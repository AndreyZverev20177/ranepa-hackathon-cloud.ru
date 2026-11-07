"""Совместимый публичный интерфейс и точка входа baseline-v1.

Реализация вынесена в отдельные модули; импорты старого кода сохранены.
"""
from .config import REQUIRED, STRINGS, METRICS, KNOWN, MSK, EMPTY, SYNTHETIC, BA
from .validation import InputError, parse_date, timestamp, load, validate_records
from .preprocessing import (
    normalized, brand, brand_context, clean_text, semantic_text, embedded_urls,
    PreparedPublication, prepare_publication, prepare_publications,
)
from .clustering import story_key
from .scoring import card
from .digest import digest, render

__all__ = [
    "REQUIRED", "STRINGS", "METRICS", "KNOWN", "MSK", "EMPTY", "SYNTHETIC", "BA",
    "InputError", "parse_date", "timestamp", "load", "normalized", "brand",
    "brand_context", "clean_text", "semantic_text", "embedded_urls",
    "PreparedPublication", "prepare_publication", "prepare_publications",
    "validate_records", "story_key", "card", "digest", "render",
]
