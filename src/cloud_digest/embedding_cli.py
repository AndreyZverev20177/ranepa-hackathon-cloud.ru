"""Standalone embedding generation: python -m cloud_digest.embedding_cli.

Does not change baseline digest scoring or clustering. Vectors are exported
without original text; they can be consumed by stage 04 clustering.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys
from typing import Any

from .config import MSK
from .io_utils import write_json_atomic as _write_json_atomic
from .embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingConfig, EmbeddingError, EmbeddingService
from .partner import adapt
from .preprocessing import brand, prepare_publications
from .validation import InputError, load, parse_date, timestamp


def build_embedding_export(
    rows: list[dict[str, Any]], day: date, service: EmbeddingService,
    *, brand_only: bool = True,
) -> dict[str, Any]:
    daily = [row for row in rows if timestamp(row["published_at"]).astimezone(MSK).date() == day]
    selected = [row for row in daily if not brand_only or brand(row["text"]) is not None]
    prepared = prepare_publications(selected)
    embedded = service.encode(prepared)
    return {
        "embedding_schema": "cloud-digest-embeddings-v1",
        "date_msk": day.isoformat(),
        "model": service.config.model_name,
        "model_revision": service.config.revision,
        "text_representation": "semantic_text-v1",
        "normalized": True,
        "brand_only": brand_only,
        "counts": {
            "input_records": len(rows),
            "day_records": len(daily),
            "selected_records": len(selected),
            "embedded_records": sum(x.vector is not None for x in embedded),
            "skipped_records": sum(x.vector is None for x in embedded),
            "cache_hits": sum(x.from_cache for x in embedded),
        },
        "items": [
            {
                "message_id": pub.message_id,
                "text_sha256": result.text_sha256,
                "vector": list(result.vector) if result.vector is not None else None,
                "skip_reason": result.skip_reason,
            }
            for pub, result in zip(prepared, embedded)
        ],
    }




def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Подготовить эмбеддинги без изменения baseline-дайджеста")
    parser.add_argument("--input", required=True, type=Path, help="Входной JSON или CSV")
    parser.add_argument("--date", required=True, help="Дата МСК в формате YYYY-MM-DD")
    parser.add_argument("--output", required=True, type=Path, help="Выходной JSON с векторами")
    parser.add_argument("--partner-export", action="store_true", help="Адаптер исходной выгрузки Brand Analytics")
    parser.add_argument("--all", action="store_true", help="Векторизовать все записи за дату, не только упоминания бренда")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--revision", default=None, help="Фиксированная ревизия модели для воспроизводимости")
    parser.add_argument("--device", default=None, help="Например cpu или cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/embedding_cache"))
    args = parser.parse_args(argv)
    try:
        day = parse_date(args.date)
        if args.input.resolve() == args.output.resolve():
            raise InputError("Входной файл не должен совпадать с выходным")
        config = EmbeddingConfig(
            model_name=args.model, revision=args.revision, batch_size=args.batch_size,
            cache_dir=args.cache_dir, device=args.device,
        )
        rows, warnings = (adapt(args.input), []) if args.partner_export else load(args.input)
        for warning in warnings:
            # Only field names; do not print personal data from original texts.
            print("Неизвестное поле: " + warning, file=sys.stderr)
        result = build_embedding_export(rows, day, EmbeddingService(config), brand_only=not args.all)
        _write_json_atomic(args.output, result)
        counts = result["counts"]
        print("Готово: " + str(counts["embedded_records"]) + " эмбеддингов, "
              + str(counts["skipped_records"]) + " пропущено, "
              + str(counts["cache_hits"]) + " повторно использовано из кэша; "
              + str(args.output))
        return 0
    except (InputError, EmbeddingError) as exc:
        print("Ошибка embeddings: " + str(exc), file=sys.stderr)
        return 2
    except (OSError, UnicodeError):
        print("Ошибка чтения или записи файла.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
