"""Semantic vectorization for prepared publications (stage 03).

Independent from baseline-v1: it does NOT change ranking or story_key().
Sentence-transformers is an optional dependency, imported only when needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

from .preprocessing import PreparedPublication

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_CACHE_SCHEMA = 1
_PREPROCESSING_VERSION = "semantic_text-v1"


class EmbeddingError(ValueError):
    """Validation, encoder, or cache setup error without disclosing news text."""


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    model_name: str = DEFAULT_EMBEDDING_MODEL
    revision: str | None = None
    batch_size: int = 32
    cache_dir: Path | None = None
    device: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise EmbeddingError("model_name: укажите название модели")
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise EmbeddingError("batch_size: требуется положительное целое число")
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision.strip()):
            raise EmbeddingError("revision: требуется непустая строка")


@dataclass(frozen=True, slots=True)
class EmbeddedPublication:
    message_id: str
    text_sha256: str
    vector: tuple[float, ...] | None
    from_cache: bool
    skip_reason: str | None = None


def _hash_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _cache_key(config: EmbeddingConfig, text_hash: str) -> str:
    # Separate cache for each model/revision/preprocessing strategy.
    identity = json.dumps([
        _CACHE_SCHEMA, _PREPROCESSING_VERSION, config.model_name, config.revision,
        text_hash,
    ], ensure_ascii=False, separators=(",", ":"))
    return sha256(identity.encode("utf-8")).hexdigest()


def _as_unit_vector(values: Any) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) and not hasattr(values, "tolist"):
        raise EmbeddingError("Модель вернула неверный формат вектора")
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, (list, tuple)) or not values:
        raise EmbeddingError("Модель вернула пустой или неверный вектор")
    try:
        vector = tuple(float(x) for x in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EmbeddingError("Модель вернула нечисловой вектор") from exc
    if not all(math.isfinite(v) for v in vector):
        raise EmbeddingError("Вектор содержит NaN или бесконечность")
    length = math.sqrt(math.fsum(v * v for v in vector))
    if not length or not math.isfinite(length):
        raise EmbeddingError("Вектор нулевой длины")
    return tuple(v / length for v in vector)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity of two vectors, validates dimensions and numbers."""
    if len(left) != len(right) or not left:
        raise EmbeddingError("Сравниваемые векторы имеют разную или нулевую размерность")
    a, b = _as_unit_vector(left), _as_unit_vector(right)
    return max(-1.0, min(1.0, math.fsum(x * y for x, y in zip(a, b))))


class EmbeddingService:
    """One encoder instance + batched inference + content-addressed cache.

    `encoder` can be injected for unit tests without model downloads. A real
    sentence-transformers model is initialized lazily only on a cache miss.
    """

    def __init__(self, config: EmbeddingConfig | None = None, *, encoder: Any = None):
        self.config = config or EmbeddingConfig()
        self._encoder = encoder

    def _get_encoder(self) -> Any:
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    'Для embeddings установите зависимости: python -m pip install -e ".[embeddings]"'
                ) from exc
            kwargs: dict[str, Any] = {}
            if self.config.device is not None:
                kwargs["device"] = self.config.device
            if self.config.revision is not None:
                kwargs["revision"] = self.config.revision
            try:
                self._encoder = SentenceTransformer(self.config.model_name, **kwargs)
            except Exception as exc:
                raise EmbeddingError(
                    "Не удалось загрузить модель embeddings (проверьте название, сеть и кэш модели)"
                ) from exc
        return self._encoder

    def _cache_file(self, cache_key: str) -> Path | None:
        if self.config.cache_dir is None:
            return None
        return self.config.cache_dir / cache_key[:2] / (cache_key + ".json")

    def _read_cache(self, path: Path | None, text_hash: str) -> tuple[float, ...] | None:
        if path is None or not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (not isinstance(payload, dict)
                    or payload.get("schema") != _CACHE_SCHEMA
                    or payload.get("text_sha256") != text_hash
                    or payload.get("model_name") != self.config.model_name
                    or payload.get("revision") != self.config.revision
                    or payload.get("preprocessing_version") != _PREPROCESSING_VERSION):
                return None
            return _as_unit_vector(payload.get("vector"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError, EmbeddingError):
            # Corrupted cache is a miss, not a poisoned vector.
            return None

    def _write_cache(self, path: Path | None, text_hash: str, vector: tuple[float, ...]) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": _CACHE_SCHEMA,
            "preprocessing_version": _PREPROCESSING_VERSION,
            "model_name": self.config.model_name,
            "revision": self.config.revision,
            "text_sha256": text_hash,
            "vector": vector,
        }
        name: str | None = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                             prefix=".tmp-", suffix=".json", delete=False) as tmp:
                name = tmp.name
                json.dump(payload, tmp, ensure_ascii=False, separators=(",", ":"))
            os.replace(name, path)
        finally:
            if name is not None:
                Path(name).unlink(missing_ok=True)

    def encode(self, publications: Sequence[PreparedPublication]) -> list[EmbeddedPublication]:
        """Return results in input order, with None for insufficient semantic text.

        Identical semantic texts are encoded only once even with no disk cache.
        """
        ids = [pub.message_id for pub in publications]
        if len(ids) != len(set(ids)):
            raise EmbeddingError("message_id: повторяющиеся идентификаторы")
        keys: dict[str, str] = {}
        unique: dict[str, str] = {}
        skip: set[str] = set()
        for pub in publications:
            if not pub.semantic_ready or not pub.semantic_text.strip():
                skip.add(pub.message_id)
                continue
            text_hash = _hash_text(pub.semantic_text)
            key = _cache_key(self.config, text_hash)
            keys[pub.message_id] = key
            unique.setdefault(key, pub.semantic_text)

        results_by_key: dict[str, tuple[float, ...]] = {}
        read_from_disk: set[str] = set()
        missing_keys = []
        for key, text in unique.items():
            vector = self._read_cache(self._cache_file(key), _hash_text(text))
            if vector is not None:
                results_by_key[key] = vector
                read_from_disk.add(key)
            else:
                missing_keys.append(key)

        if missing_keys:
            encoder = self._get_encoder()
            dimension: int | None = None
            for start in range(0, len(missing_keys), self.config.batch_size):
                batch_keys = missing_keys[start:start + self.config.batch_size]
                try:
                    encoded = encoder.encode(
                        [unique[key] for key in batch_keys],
                        batch_size=self.config.batch_size,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    )
                    if hasattr(encoded, "tolist"):
                        encoded = encoded.tolist()
                    if len(encoded) != len(batch_keys):
                        raise EmbeddingError("Неверное количество векторов от модели")
                    normalized_vectors = [_as_unit_vector(row) for row in encoded]
                    for vector in normalized_vectors:
                        if dimension is None:
                            dimension = len(vector)
                        elif len(vector) != dimension:
                            raise EmbeddingError("Модель вернула векторы разной размерности")
                except EmbeddingError:
                    raise
                except Exception as exc:
                    raise EmbeddingError("Ошибка при пакетной векторизации") from exc
                for key, vector in zip(batch_keys, normalized_vectors):
                    results_by_key[key] = vector
                    self._write_cache(self._cache_file(key), _hash_text(unique[key]), vector)

        # Detect a stale / inconsistent cache rather than returning vectors of mixed size.
        dimensions = {len(v) for v in results_by_key.values()}
        if len(dimensions) > 1:
            raise EmbeddingError("Размерности кэшированных и новых векторов не совпадают")

        first_in_batch: set[str] = set()
        output = []
        for pub in publications:
            if pub.message_id in skip:
                output.append(EmbeddedPublication(
                    message_id=pub.message_id, text_sha256=_hash_text(pub.semantic_text),
                    vector=None, from_cache=False, skip_reason="insufficient_semantic_text",
                ))
                continue
            key = keys[pub.message_id]
            hit = key in read_from_disk or key in first_in_batch
            output.append(EmbeddedPublication(
                message_id=pub.message_id, text_sha256=_hash_text(pub.semantic_text),
                vector=results_by_key[key], from_cache=hit,
            ))
            first_in_batch.add(key)
        return output
