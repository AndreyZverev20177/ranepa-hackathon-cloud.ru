"""Скоринг отдельной публикации: совместимый baseline-v1 и opt-in publication-v2.

Publication-v2 использует топ-25 агрегированных площадок и осторожные
эвристики для остальных. Это НЕ скоринг всего сюжета: Story Score будет
добавлен на следующем этапе.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from .config import BA
from .preprocessing import normalized, brand_context
from .clustering import story_key
from .source_authority import SourceAuthorityRegistry


@dataclass(frozen=True, slots=True)
class PublicationScoreWeights:
    """Пилотные веса: максимальная оценка равна 100."""
    negative: int = 30
    crisis: int = 20
    audience: int = 15
    views: int = 10
    authority: int = 20
    historical_media_weight: int = 5
    audience_reference: int = 1_000_000
    views_reference: int = 100_000

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> PublicationScoreWeights:
        names = set(cls.__dataclass_fields__)
        if not isinstance(settings, dict) or set(settings) != names:
            raise ValueError('weights: ожидаются все и только параметры PublicationScoreWeights')
        if any(type(value) is not int or value < 0 for value in settings.values()):
            raise ValueError('weights: допускаются только неотрицательные целые')
        return cls(**settings)

    def __post_init__(self) -> None:
        values = (self.negative, self.crisis, self.audience, self.views,
                  self.authority, self.historical_media_weight)
        if any(type(v) is not int or v < 0 for v in values) or sum(values) != 100:
            raise ValueError('Веса скоринга должны быть неотрицательными и давать 100')
        if self.audience_reference < 1 or self.views_reference < 1:
            raise ValueError('Неправильные опорные значения аудитории/просмотров')


def _log_points(value: int | None, cap: int, reference: int) -> int:
    """Плавная, ограниченная оценка метрики вместо одного порога."""
    if value is None or value <= 0:
        return 0
    return min(cap, round(cap * math.log1p(value) / math.log1p(reference)))


def _baseline_score(row: dict[str, Any], negative: bool, crisis: bool) -> tuple[int, list[str]]:
    """Полностью сохраняем прежний алгоритм и текст объяснений."""
    score = 0
    explanation = []
    for condition, points, label in [
        (negative, 40, 'negative'), (crisis, 30, 'crisis_signal'),
        (row['audience'] is not None and row['audience'] >= 10000, 20, 'audience >= 10000'),
        (row['views'] is not None and row['views'] >= 1000, 10, 'views >= 1000'),
    ]:
        if condition:
            score += points
            explanation.append(f'{label}: +{points}')
    explanation.extend(f'{key}: нет данных' for key in ('audience', 'views') if row[key] is None)
    return score, explanation


def score_publication_v2(
    row: dict[str, Any], negative: bool, crisis: bool,
    registry: SourceAuthorityRegistry,
    weights: PublicationScoreWeights | None = None,
) -> tuple[int, list[str], dict[str, Any]]:
    """Оценка публикации: риск + метрики + авторитетность + вес в top25.

    media_weight из top25 — активность площадки в выборке, а не достоверность.
    Отсутствующие просмотры не превращаются в фиктивные нули для отчётности.
    """
    w = weights or PublicationScoreWeights()
    source = registry.resolve(row['url'], row['source_type'])
    breakdown = {
        'negative': w.negative if negative else 0,
        'crisis': w.crisis if crisis else 0,
        'audience': _log_points(row.get('audience'), w.audience, w.audience_reference),
        'views': _log_points(row.get('views'), w.views, w.views_reference),
        'authority': round(w.authority * source.score / 100),
        'historical_media_weight': round(w.historical_media_weight * math.sqrt(source.media_weight or 0.0)),
    }
    score = min(100, sum(breakdown.values()))
    explanation = [f'{name}: +{points}' for name, points in breakdown.items() if points > 0]
    explanation.extend(f'{key}: нет данных' for key in ('audience', 'views') if row.get(key) is None)
    explanation.append(f'authority_basis: {source.basis}')
    if source.platform_level:
        explanation.append('authority_scope: platform, not verified publisher')
    meta = {
        'scoring_version': 'publication-v2',
        'publication_score': score,
        'publication_score_breakdown': breakdown,
        'source_authority_score': source.score,
        'source_authority_basis': source.basis,
        'source_authority_scope': 'platform' if source.platform_level else 'publisher_domain_or_type',
        'source_host': source.host,
        'source_top25_rank': source.rank,
        'source_top25_media_weight': source.media_weight,
    }
    return score, explanation, meta


def card(
    row: dict[str, Any], link: tuple[str, str], *,
    scoring: str = 'baseline', source_registry: SourceAuthorityRegistry | None = None,
    scoring_weights: PublicationScoreWeights | None = None,
) -> dict[str, Any]:
    """Сформировать карточку; baseline является режимом по умолчанию."""
    text = row['text']
    norm = normalized(brand_context(text, link[1]))
    negative = any(word in norm for word in ('сбой', 'недоступ', 'утечк', 'авари'))
    positive = any(word in norm for word in ('спасибо', 'отличн', 'удобн'))
    crisis = any(word in norm for word in ('утечк', 'недоступ', 'авари'))
    sentiment = 'negative' if negative else 'positive' if positive else 'neutral'
    reasons = []
    for condition, reason in [
        (link[0] == 'historical', 'historical_brand'),
        (negative and positive, 'mixed_sentiment'),
        (row['ba_sentiment'] is not None and BA[str(row['ba_sentiment'])] != sentiment, 'ba_sentiment_mismatch'),
        (len(text) < 160, 'short_text'),
        (crisis, 'crisis_signal'),
    ]:
        if condition:
            reasons.append(reason)
    metadata: dict[str, Any] = {}
    if scoring == 'baseline':
        score, explanation = _baseline_score(row, negative, crisis)
    elif scoring == 'publication-v2':
        if source_registry is None:
            raise ValueError('Для publication-v2 требуется справочник top25_sources.json')
        score, explanation, metadata = score_publication_v2(
            row, negative, crisis, source_registry, scoring_weights)
    else:
        raise ValueError('Неизвестный режим скоринга')

    return {
        **{key: row[key] for key in ('message_id', 'published_at', 'ba_sentiment', 'source_name', 'source_type', 'audience', 'url')},
        'sentiment': sentiment, 'mixed': negative and positive,
        'importance_score': score, 'importance_explanation': explanation,
        'brand_link': link[0], 'matched_span': link[1],
        'cluster_id': story_key(row, text), 'cluster_size': 1,
        'summary_160_300': text[:300],
        'recommended_action': 'escalate_pr' if crisis else 'watch',
        'confidence': 0.4 if reasons else 0.8,
        'human_review_needed': bool(reasons), 'review_reasons': reasons,
        **metadata,
    }
