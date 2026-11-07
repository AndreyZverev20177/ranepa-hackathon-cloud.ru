"""Story Score v1: влияние *сюжета* с консервативной дедупликацией.

Это измерение приоритета для редактора, а не реальный охват и не верификация фактов.
- одна площадка/канал -> один вклад в аудиторию независимо от числа репостов;
- сумма аудиторий НЕ трактуется как уникальный охват: допускается лишь
  небольшой ограниченный proxy-бонус за дополнительные площадки;
- авторитет определяется справочником stage 05 и не приписывается всей
  социальной платформе как авторитет отдельного канала;
- новые факты: явные числовые утверждения и сервисы, без проверки истины.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from .source_authority import normalize_host
from .preprocessing import prepare_publication
from .event_facts import extract_event_facts, identity_conflicts
from .validation import timestamp


@dataclass(frozen=True, slots=True)
class StoryScoreWeights:
    distribution_per_outlet: int = 2
    distribution_cap: int = 6
    reach_cap: int = 12
    reach_reference: int = 1_000_000
    reach_secondary_fraction_pct: int = 15
    reach_max_fraction_pct: int = 25
    authority_threshold: int = 75
    authority_per_extra_publisher: int = 4
    authority_cap: int = 8
    new_fact_points: int = 3
    new_fact_cap: int = 9

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f'story weights: {field} должно быть неотрицательным целым')
        if not (0 <= self.authority_threshold <= 100):
            raise ValueError('authority_threshold должен находиться в диапазоне 0..100')
        if self.reach_reference < 1:
            raise ValueError('reach_reference должен быть положительным')
        if not (0 <= self.reach_secondary_fraction_pct <= 100 and 0 <= self.reach_max_fraction_pct <= 100):
            raise ValueError('reach fractions должны находиться в диапазоне 0..100')
        for name in ('distribution_cap', 'reach_cap', 'authority_cap', 'new_fact_cap'):
            if getattr(self, name) > 100:
                raise ValueError(f'{name}: максимум 100 баллов')

    @classmethod
    def from_settings(cls, value: Mapping[str, Any]) -> 'StoryScoreWeights':
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError('story weights: неверный набор полей')
        return cls(**value)


def load_story_weights(path: Path) -> StoryScoreWeights:
    try:
        raw = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('Не удалось прочесть настройки Story Score') from exc
    if not isinstance(raw, dict) or 'weights' not in raw:
        raise ValueError('В файле настроек Story Score отсутствует weights')
    return StoryScoreWeights.from_settings(raw['weights'])


def _source_id(row: Mapping[str, Any]) -> tuple[str, str]:
    """Один идентификатор на канал, а не на репост; основание явно указывается.

    Для Дзена нет подтверждённого ID издателя в URL /a/<article>; группируем
    по source_name как по *заявленному* издателю (не верифицированному).
    """
    name = re.sub(r'\s+', ' ', str(row.get('source_name') or '')).strip().casefold()
    url = str(row['url'])
    host = normalize_host(url)
    path = urlsplit(url).path.strip('/').split('/')
    if host == 'telegram.org':
        if path and path[0].lower() == 's' and len(path) >= 2:
            path = path[1:]
        if path and path[0].lower() == 'c' and len(path) >= 2:
            return f'telegram:c:{path[1].casefold()}', 'platform_channel'
        if path and re.fullmatch(r'[\w]{3,}', path[0], re.UNICODE) and path[0].lower() not in ('joinchat', 'share'):
            return f'telegram:{path[0].casefold()}', 'platform_channel'
        return f'telegram:unknown:{name}', 'claimed_source_name'
    if host == 'vk.com':
        first = path[0].casefold() if path else ''
        match = re.match(r'(?:wall|video|story|photo|market)(-?\d+)(?:_|$)', first)
        if match:
            return f'vk:{match.group(1)}', 'platform_account'
        match = re.fullmatch(r'(?:public|club|id)(\d+)', first)
        if match:
            return f'vk:{match.group(1)}', 'platform_account'
        if first:
            return f'vk:{first}', 'platform_account'
        return f'vk:unknown:{name}', 'claimed_source_name'
    if host == 'dzen.ru':
        return f'dzen:{name or "unknown"}', 'claimed_source_name'
    if host in ('youtube.com', 'rutube.ru', 'max.ru', 'instagram.com'):
        segment = path[0].casefold() if path else ''
        # Не трактуем id конкретного видео как отдельного издателя.
        if segment in ('watch', 'video', 'shorts', 'v', 'clip', 'embed', 'reel', ''):
            return f'{host}:unknown:{name}', 'claimed_source_name'
        return f'{host}:{segment}', 'platform_account'
    # Субдомены известных издательских групп считаем одним издателем.
    # Не используем произвольный source_name для обхода проверки домена.
    publishers = ('cnews.ru', 'rbc.ru', 'kommersant.ru', 'tass.ru',
                  'forbes.ru', 'vedomosti.ru', 'ria.ru', 'habr.com', 'vc.ru')
    for publisher in publishers:
        if host == publisher or host.endswith('.' + publisher):
            return f'site:{publisher}', 'publisher_domain'
    return f'site:{host}', 'publisher_domain'


# Сигналы: простые *явные числовые утверждения*, не внешняя верификация.
# Ограничиваем расстояние от названия метрики до значения и не приравниваем
# произвольные числа в тексте к фактам. Значения внутри одной метрики
# с разными единицами приводятся к общей шкале.
_METRIC = re.compile(r'\b(?P<label>выручк\w*|доход\w*|инвестиц\w*|капзатрат\w*|потреблен\w*|токен\w*|пользовател\w*)\b', re.I)
_VALUE = re.compile(r'(?<![\w])(?P<num>\d{1,12}(?:[.,]\d{1,3})?)\s*(?P<unit>млрд|млн|тыс\.?|%|процент(?:а|ов)?)(?!\w)', re.I)
_AI_WORD = re.compile(r'(?i)(?:\bии\b|искусствен\w*\s+интеллект\w*|ai[- ]?сервис\w*)')
_BASE_UNITS = {'млрд': Decimal('1000000000'), 'млн': Decimal('1000000'), 'тыс': Decimal('1000'), 'тыс.': Decimal('1000')}


def extract_claims(text: str) -> dict[str, str]:
    """metric -> normalized value; нет доказательств истинности утверждения.

    Игнорируем текст без явной метрики и единицы измерения, конфликтующие
    значения в *одном* сообщении, даты, случайные числа, просмотры из метаданных.
    """
    clean = re.sub(r'https?://\S+', ' ', text)
    claims: dict[str, str] = {}
    ambiguous: set[str] = set()
    for m in _METRIC.finditer(clean):
        tail = re.split(r'[!?;\n]|\.(?=\s+[А-ЯA-ZЁ])', clean[m.end():m.end()+75], maxsplit=1)[0]
        value = _VALUE.search(tail)
        if value is None:
            continue
        label = m.group('label').lower()
        if label.startswith(('выруч', 'доход')):
            metric = 'revenue'
        elif label.startswith(('инвест', 'капзатрат')):
            metric = 'investment'
        elif label.startswith(('токен', 'потреблен')):
            metric = 'token_volume'
        else:
            metric = 'users'
        # Контекст квалификатора строго около самой метрики/цифры.
        qualifier = clean[max(0, m.start()-28):m.start()] + tail[:value.start()]
        if metric == 'revenue' and _AI_WORD.search(qualifier):
            metric = 'ai_revenue'
        unit = value.group('unit').lower()
        try:
            number = Decimal(value.group('num').replace(',', '.'))
        except InvalidOperation:
            continue
        if unit in ('%', 'процента', 'процентов'):
            normalized_value = f'percent:{number.normalize()}'
        else:
            normalized_value = f'quantity:{(number * _BASE_UNITS[unit]).normalize()}'
        if metric in claims and claims[metric] != normalized_value:
            ambiguous.add(metric)
        else:
            claims[metric] = normalized_value
    return {key: value for key, value in claims.items() if key not in ambiguous}


@dataclass(frozen=True, slots=True)
class StoryScore:
    score: int
    representative_id: str
    breakdown: dict[str, int]
    metadata: dict[str, Any]
    review_reasons: tuple[str, ...]


def score_story(
    cards: Sequence[dict[str, Any]], raw_rows: Mapping[str, dict[str, Any]],
    weights: StoryScoreWeights | None = None,
) -> StoryScore:
    """Оценивает сюжет. Все карточки уже оценены publication-v2."""
    if not cards:
        raise ValueError('Пустой сюжет')
    w = weights or StoryScoreWeights()
    ordered = sorted(cards, key=lambda c: (timestamp(c['published_at']), c['message_id']))
    if any(c.get('scoring_version') != 'publication-v2' for c in ordered):
        raise ValueError('Story Score требует publication-v2 для каждой карточки')
    if any(c['message_id'] not in raw_rows for c in ordered):
        raise ValueError('Отсутствует оригинал публикации для Story Score')
    ids = [c['message_id'] for c in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError('Повторяющийся message_id внутри сюжета')

    # Лучший представитель — наиболее значимая публикация, не первая по времени.
    best = min(ordered, key=lambda c: (-c['publication_score'],
                                       -c['source_authority_score'],
                                       timestamp(c['published_at']), c['message_id']))
    outlet_map: dict[str, dict[str, Any]] = {}
    outlet_basis: dict[str, str] = {}
    for card in ordered:
        key, basis = _source_id(card)
        outlet_basis[key] = basis
        old = outlet_map.get(key)
        if old is None or (card.get('audience') or 0, card['publication_score']) > (old.get('audience') or 0, old['publication_score']):
            outlet_map[key] = card
    outlets = list(outlet_map.items())
    audiences = sorted((max(0, card.get('audience') or 0) for _, card in outlets), reverse=True)
    max_audience = audiences[0] if audiences else 0
    secondaries = sum(audiences[1:])
    extra_audience_proxy = min(
        secondaries * w.reach_secondary_fraction_pct // 100,
        max_audience * w.reach_max_fraction_pct // 100,
    ) if max_audience else 0
    # Дополнительный proxy-бонус учитывает только добавочную условную аудиторию,
    # которую НЕЛЬЗЯ выдавать за реальный непересекающийся охват.
    reach = 0
    if extra_audience_proxy and max_audience:
        absolute_factor = math.log1p(extra_audience_proxy) / math.log1p(w.reach_reference)
        # Малое добавление к большой площадке не должно давать много баллов.
        relative_factor = min(1.0, extra_audience_proxy * 4 / max_audience)
        reach = min(w.reach_cap, round(w.reach_cap * absolute_factor * relative_factor))
    distribution = min(w.distribution_cap, w.distribution_per_outlet * (len(outlets) - 1))

    high_authority_domains = {
        key for key, card in outlets
        if card['source_authority_scope'] != 'platform'
        and outlet_basis[key] == 'publisher_domain'
        and card['source_authority_score'] >= w.authority_threshold
    }
    authority = min(w.authority_cap, max(0, len(high_authority_domains) - 1) * w.authority_per_extra_publisher)

    event_facts = {card['message_id']: extract_event_facts(prepare_publication(raw_rows[card['message_id']]))
                   for card in ordered}
    identity_warnings: list[dict[str, Any]] = []
    for i, card in enumerate(ordered):
        for prior in ordered[:i]:
            blockers = identity_conflicts(event_facts[card['message_id']],
                                          event_facts[prior['message_id']])
            if blockers:
                identity_warnings.append({'message_id_a': prior['message_id'],
                                          'message_id_b': card['message_id'],
                                          'reasons': list(blockers)})

    seen_claims: dict[str, str] = {}
    seen_services: set[str] = set()
    novel_claims: list[dict[str, str]] = []
    conflicts: list[dict[str, str]] = []
    for index, card in enumerate(ordered):
        message_id = card['message_id']
        for service in sorted(event_facts[message_id].services - seen_services):
            if index > 0:
                novel_claims.append({'message_id': message_id, 'metric': f'service:{service}',
                                     'value': 'explicit_mention'})
        seen_services.update(event_facts[message_id].services)
        claims = extract_claims(raw_rows[message_id]['text'])
        for metric, value in sorted(claims.items()):
            before = seen_claims.get(metric)
            if before is None:
                seen_claims[metric] = value
                if index > 0:
                    novel_claims.append({'message_id': message_id, 'metric': metric, 'value': value})
            elif before != value:
                conflicts.append({'message_id': message_id, 'metric': metric,
                                  'first_value': before, 'other_value': value})
    # Если метрика противоречит другой публикации, премировать её как
    # 'новый факт' нельзя до редакторской проверки.
    conflicted_metrics = {item['metric'] for item in conflicts}
    novel_claims = [item for item in novel_claims if item['metric'] not in conflicted_metrics]
    new_facts = min(w.new_fact_cap, len(novel_claims) * w.new_fact_points)
    if identity_warnings:
        # Обычная хеш-группировка может соединять два разных инцидента
        # одного дня. Не начисляем им бонусов как единому сюжету.
        distribution = reach = authority = new_facts = 0
        novel_claims = []
    base = best['publication_score']
    raw_score = base + distribution + reach + authority + new_facts
    score = min(100, raw_score)
    breakdown = {'best_publication': base, 'distribution': distribution,
                 'additional_reach_proxy': reach, 'additional_high_authority_domains': authority,
                 'new_fact_signals': new_facts}
    meta: dict[str, Any] = {
        'story_score_version': 'story-v1',
        'story_score': score,
        'story_score_uncapped': raw_score,
        'story_score_breakdown': breakdown,
        'story_explanation': (
            [f'{name}: +{value}' for name, value in breakdown.items() if value > 0]
            + (['score capped at 100'] if raw_score > 100 else [])
            + ['reach is a capped audience proxy, NOT unique reach',
               'new facts are extracted claims, NOT external fact-checking']
        ),
        'first_published_at': ordered[0]['published_at'],
        'last_published_at': ordered[-1]['published_at'],
        'best_publication_id': best['message_id'],
        'story_publication_ids': ids,
        'story_publications': [
            {'message_id': c['message_id'], 'published_at': c['published_at'],
             'source_name': c['source_name'], 'source_url': c['url'],
             'publication_score': c['publication_score'], 'audience': c['audience'],
             'source_authority_scope': c['source_authority_scope']}
            for c in ordered
        ],
        'distinct_outlets': len(outlets),
        'distinct_high_authority_domains': len(high_authority_domains),
        'max_reported_audience': max_audience,
        'additional_audience_proxy': extra_audience_proxy,
        'audience_measure': 'potential_audience_proxy_not_unique_reach',
        'novel_fact_signals': novel_claims,
        'conflicting_claim_signals': conflicts,
        'incompatible_member_pairs': identity_warnings,
    }
    review_reasons = []
    if conflicts:
        review_reasons.append('conflicting_numeric_claims')
    if identity_warnings:
        review_reasons.append('incompatible_event_members')
    return StoryScore(score, best['message_id'], breakdown, meta, tuple(review_reasons))
