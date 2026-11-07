"""Stage 04: candidate event clusters + optional internet RSS candidate enrichment.

Unlike normal digest CLI, this is a *research* report; internet candidates are
never automatically added to the production daily digest or scored as facts.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys
from typing import Any

from .clustering import SemanticClusterConfig, cluster_semantically
from .io_utils import write_json_atomic as _write_json_atomic
from .embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingConfig, EmbeddingError, EmbeddingService
from .internet_candidates import InternetCandidateError, fetch_internet_candidates, google_news_rss_url
from .partner import adapt
from .preprocessing import brand, prepare_publications
from .validation import InputError, load, parse_date, timestamp
from .config import MSK


def build_candidate_report(
    rows: list[dict[str, Any]], day: date, service: EmbeddingService,
    *, config: SemanticClusterConfig | None = None,
    internet_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Inspect semantically similar internal and RSS records without scoring them."""
    locals_ = [r for r in rows if timestamp(r['published_at']).astimezone(MSK).date() == day
               and brand(r['text']) is not None]
    known_ids = {r['message_id'] for r in locals_}
    internet = [r for r in (internet_rows or [])
                if timestamp(r['published_at']).astimezone(MSK).date() == day
                and brand(r['text']) is not None and r['message_id'] not in known_ids]
    combined = locals_ + internet
    prepared = prepare_publications(combined)
    vectors = service.encode(prepared)
    clustered = cluster_semantically(prepared, vectors, config)
    lookup = {r['message_id']: r for r in combined}
    groups = []
    for cluster in clustered.clusters:
        groups.append({
            'cluster_id': cluster.cluster_id,
            'first_published_at': cluster.first_published_at,
            'members': [
                {
                    'message_id': mid,
                    'published_at': lookup[mid]['published_at'],
                    'source_name': lookup[mid]['source_name'],
                    'url': lookup[mid]['url'],
                    'source': 'internet_rss' if lookup[mid].get('internet_candidate') else 'local_export',
                    'review_only': bool(lookup[mid].get('internet_candidate')),
                    'verified_full_text': not lookup[mid].get('internet_candidate', False),
                }
                for mid in cluster.message_ids
            ],
        })
    return {
        'schema': 'cloud-digest-semantic-candidates-v1',
        'date_msk': day.isoformat(),
        'model': service.config.model_name,
        'model_revision': service.config.revision,
        'thresholds': {'high': (config or SemanticClusterConfig()).high_threshold,
                       'low': (config or SemanticClusterConfig()).low_threshold},
        'counts': {'local_candidates': len(locals_), 'internet_candidates': len(internet),
                   'clusters': len(clustered.clusters), 'candidate_pairs': clustered.candidate_pairs,
                   'skipped_without_vector': clustered.skipped_without_vector,
                   'review_pairs': len(clustered.review_pairs),
                   'truncated_review_pairs': clustered.truncated_review_pairs},
        'clusters': groups,
        'review_pairs': [
            {'message_id_a': pair.message_id_a, 'message_id_b': pair.message_id_b,
             'similarity': pair.similarity, 'reason': pair.reason}
            for pair in clustered.review_pairs
        ],
        'notes': [
            'RSS contains only metadata/snippets, not verified article contents.',
            'Internet candidates are review_only and are not automatically inserted into daily digests.',
            'Fact guards test explicit event-identity conflicts, not the truth of online claims.',
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Семантические сюжеты и интернет-кандидаты для проверки')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--date', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--partner-export', action='store_true')
    parser.add_argument('--high', type=float, default=0.88)
    parser.add_argument('--low', type=float, default=0.78)
    parser.add_argument('--model', default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument('--revision', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--cache-dir', type=Path, default=Path('artifacts/embedding_cache'))
    parser.add_argument('--internet', action='store_true', help='Явно включить RSS поиск (доступ в интернет)')
    parser.add_argument('--internet-query', default='"Cloud.ru"', help='Запрос Google News RSS')
    parser.add_argument('--rss-feed', action='append', default=[], help='Дополнительный разрешённый RSS URL')
    parser.add_argument('--internet-limit', type=int, default=100)
    args = parser.parse_args(argv)
    try:
        day = parse_date(args.date)
        if args.input.resolve() == args.output.resolve():
            raise InputError('Выходной файл не должен совпадать с входным')
        if args.rss_feed and not args.internet:
            raise InputError('--rss-feed требует явного --internet')
        rows, warnings = (adapt(args.input), []) if args.partner_export else load(args.input)
        for warning in warnings:
            print('Неизвестное поле: ' + warning, file=sys.stderr)
        rss_rows = []
        if args.internet:
            urls = [google_news_rss_url(args.internet_query, day=day), *args.rss_feed]
            rss_rows = fetch_internet_candidates(urls, day=day, max_records=args.internet_limit)
        config = SemanticClusterConfig(high_threshold=args.high, low_threshold=args.low)
        service = EmbeddingService(EmbeddingConfig(model_name=args.model, revision=args.revision,
                             cache_dir=args.cache_dir, device=args.device))
        report = build_candidate_report(rows, day, service, config=config, internet_rows=rss_rows)
        _write_json_atomic(args.output, report)
        print(f"Кластеры: {report['counts']['clusters']}; интернет-кандидаты: {report['counts']['internet_candidates']}; файл: {args.output}")
        return 0
    except (InputError, InternetCandidateError, EmbeddingError, ValueError) as exc:
        print('Ошибка поиска кандидатов: ' + str(exc), file=sys.stderr)
        return 2
    except (OSError, UnicodeError):
        print('Ошибка чтения/записи или сети.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
