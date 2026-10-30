"""Локальная валидация и объяснимые эвристики; входной текст — только данные."""
import csv
import hashlib
import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

REQUIRED = ('message_id', 'published_at', 'source_type', 'source_name', 'url', 'text')
STRINGS = ('author_name', 'author_id', 'language', 'geo', 'ba_topic_hit')
METRICS = ('audience', 'views', 'likes', 'comments', 'reposts')
KNOWN = set(REQUIRED + STRINGS + METRICS + ('ba_sentiment',))
MSK = timezone(timedelta(hours=3))
EMPTY = 'Прямого или косвенного упоминания Cloud.ru не найдено'
SYNTHETIC = 'СИНТЕТИЧЕСКИЙ ПРИМЕР — ВЫМЫСЕЛ'
BA = {'negative': 'negative', '-': 'negative', '−': 'negative',
      'neutral': 'neutral', '0': 'neutral', 'positive': 'positive', '+': 'positive'}


class InputError(ValueError):
    """Ошибка без содержимого исходной записи."""


def parse_date(value: str) -> date:
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise InputError('Дата: требуется YYYY-MM-DD')
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise InputError('Дата: несуществующие сутки') from None


def timestamp(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None or 'T' not in value:
        raise ValueError('timezone')
    return dt.astimezone(timezone.utc)


def load(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    is_csv = path.suffix.lower() == '.csv'
    raw: Any
    try:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            if is_csv:
                rows = list(csv.reader(stream, strict=True))
                if not rows or len(rows[0]) != len(set(rows[0])) or not set(REQUIRED) <= set(rows[0]):
                    raise InputError('CSV: отсутствующий или повторяющийся заголовок')
                if any(len(row) != len(rows[0]) for row in rows[1:]):
                    raise InputError('CSV: число ячеек не совпадает с заголовком')
                raw = [dict(zip(rows[0], row)) for row in rows[1:]]
                extras = set(rows[0]) - KNOWN
            elif path.suffix.lower() == '.json':
                raw = json.load(stream)
                extras = set()
            else:
                raise InputError('Поддерживаются только .csv и .json')
    except (csv.Error, json.JSONDecodeError, UnicodeError):
        raise InputError('Некорректный CSV/JSON или кодировка UTF-8') from None
    if not isinstance(raw, list):
        raise InputError('JSON: ожидается массив объектов')
    result: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, row in enumerate(raw, 1):
        def fail(field: str) -> None:
            raise InputError(f'Запись {index}: неверное поле {field}')
        if not isinstance(row, dict):
            fail('record')
        extras.update(set(row) - KNOWN)
        record = {key: row.get(key) for key in KNOWN}
        for key in REQUIRED:
            if not isinstance(record[key], str) or not record[key].strip():
                fail(key)
        try:
            timestamp(record['published_at'])
        except (ValueError, OverflowError):
            fail('published_at')
        try:
            url = urlsplit(record['url'])
            if url.scheme not in ('http', 'https') or not url.hostname or any(c.isspace() for c in record['url']):
                fail('url')
        except ValueError:
            fail('url')
        if record['message_id'] in ids:
            fail('message_id (повтор)')
        ids.add(record['message_id'])
        for key in STRINGS + METRICS + ('ba_sentiment',):
            if is_csv and record[key] == '':
                record[key] = None
        for key in STRINGS:
            if record[key] is not None and not isinstance(record[key], str):
                fail(key)
        for key in METRICS:
            value = record[key]
            if is_csv and isinstance(value, str) and re.fullmatch(r'[0-9]+', value):
                value = int(value)
            if value is not None and (type(value) is not int or value < 0):
                fail(key)
            record[key] = value
        value = record['ba_sentiment']
        if value is not None and not (type(value) is int and value == 0) and not (isinstance(value, str) and value in BA):
            fail('ba_sentiment')
        result.append(record)
    return result, sorted(extras)


def normalized(text: str) -> str:
    return ' '.join(text.casefold().split())


def brand(text: str) -> tuple[str, str] | None:
    # Casefold may expand characters: map folded offsets back to exact original span.
    folded = text.casefold()
    offsets = [i for i, char in enumerate(text) for _ in char.casefold()]
    for kind, pattern in [('direct', r'(?<![\w.-])cloud\.ru(?![\w-]|\.[\w-])'),
                          ('historical', r'(?<!\w)(sbercloud|сберклауд)(?!\w)')]:
        match = re.search(pattern, folded)
        if match:
            return kind, text[offsets[match.start()]:offsets[match.end() - 1] + 1]
    return None


def card(row: dict[str, Any], link: tuple[str, str]) -> dict[str, Any]:
    text = row['text']
    norm = normalized(text)
    negative = any(word in norm for word in ('сбой', 'недоступ', 'утечк', 'авари'))
    positive = any(word in norm for word in ('спасибо', 'отличн', 'удобн'))
    crisis = any(word in norm for word in ('утечк', 'недоступ', 'авари'))
    sentiment = 'negative' if negative else 'positive' if positive else 'neutral'
    reasons = []
    for condition, reason in [(link[0] == 'historical', 'historical_brand'),
                              (negative and positive, 'mixed_sentiment'),
                              (row['ba_sentiment'] is not None and BA[str(row['ba_sentiment'])] != sentiment, 'ba_sentiment_mismatch'),
                              (len(text) < 160, 'short_text'), (crisis, 'crisis_signal')]:
        if condition:
            reasons.append(reason)
    score = 0
    explanation = []
    for condition, points, label in [(negative, 40, 'negative'), (crisis, 30, 'crisis_signal'),
                                    (row['audience'] is not None and row['audience'] >= 10000, 20, 'audience >= 10000'),
                                    (row['views'] is not None and row['views'] >= 1000, 10, 'views >= 1000')]:
        if condition:
            score += points
            explanation.append(f'{label}: +{points}')
    explanation.extend(f'{key}: нет данных' for key in ('audience', 'views') if row[key] is None)
    return {**{key: row[key] for key in ('message_id', 'published_at', 'ba_sentiment', 'source_name', 'source_type', 'audience', 'url')},
            'sentiment': sentiment, 'mixed': negative and positive, 'importance_score': score,
            'importance_explanation': explanation, 'brand_link': link[0], 'matched_span': link[1],
            'cluster_id': hashlib.sha256(norm.encode()).hexdigest(), 'cluster_size': 1,
            'summary_160_300': text[:300], 'recommended_action': 'escalate_pr' if crisis else 'watch',
            'confidence': 0.4 if reasons else 0.8, 'human_review_needed': bool(reasons), 'review_reasons': reasons}


def digest(rows: list[dict[str, Any]], day: date, top_n: int = 30, synthetic: bool = False) -> dict[str, Any]:
    if not 1 <= top_n <= 30:
        raise InputError('top-n: требуется целое 1–30')
    daily = [r for r in rows if timestamp(r['published_at']).astimezone(MSK).date() == day]
    candidates = []
    dropped = []
    queue = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in daily:
        link = brand(row['text'])
        if link is None:
            topic = bool(row['ba_topic_hit'] and row['ba_topic_hit'].strip())
            reason = 'topic_hit_without_brand' if topic else 'no_brand_match'
            dropped.append({'message_id': row['message_id'], 'reason': reason})
            if topic:
                queue.append({'message_id': row['message_id'], 'cluster_id': None, 'url': row['url'], 'reasons': [reason]})
            continue
        item = card(row, link)
        candidates.append(item)
        groups.setdefault(item['cluster_id'], []).append(item)
        if item['human_review_needed']:
            queue.append({'message_id': item['message_id'], 'cluster_id': item['cluster_id'], 'url': item['url'], 'reasons': item['review_reasons']})
    def chronological(item: dict[str, Any]) -> tuple[datetime, str]:
        return timestamp(item['published_at']), item['message_id']
    def priority(item: dict[str, Any]) -> tuple[int, datetime, str]:
        return -item['importance_score'], *chronological(item)
    representatives = []
    for group in groups.values():
        ordered = sorted(group, key=chronological)
        representative = ordered[0]
        representative['cluster_size'] = len(group)
        representatives.append(representative)
        dropped.extend({'message_id': item['message_id'], 'reason': 'duplicate'} for item in ordered[1:])
    ranked = sorted(representatives, key=priority)
    selected = sorted(ranked[:top_n], key=lambda item: (('negative', 'neutral', 'positive').index(item['sentiment']), priority(item)))
    dropped.extend({'message_id': item['message_id'], 'reason': 'outside_top_n'} for item in ranked[top_n:])
    for rank, item in enumerate(selected, 1):
        item['rank_in_digest'] = rank
    risk_ids = sorted(item['message_id'] for item in candidates if 'crisis_signal' in item['review_reasons'])
    counts = dict(input_records=len(rows), day_records=len(daily), brand_candidates=len(candidates),
                  clusters=len(groups), duplicates_removed=len(candidates) - len(groups), selected_items=len(selected),
                  review_records=len(queue), excluded_no_brand=len(daily) - len(candidates), excluded_top_n=len(ranked[top_n:]))
    return dict(digest_date=day.isoformat(), rule_version='baseline-v1', summary_counts=counts, items=selected,
                dropped_examples=sorted(dropped, key=lambda item: item['message_id'])[:10], risk_alert=bool(risk_ids),
                risk_alert_details={'message_ids': risk_ids, 'reason': 'crisis_signal' if risk_ids else None},
                review_queue=sorted(queue, key=lambda item: item['message_id']), empty_message=EMPTY if not candidates else None,
                synthetic=synthetic)


def render(result: dict[str, Any]) -> str:
    def display(value: Any) -> str:
        if value is None:
            return 'нет данных'
        if isinstance(value, dict):
            return '\n'.join(f'{key}: {display(item)}' for key, item in value.items())
        if isinstance(value, list):
            return '\n\n'.join(display(item) for item in value) or '[]'
        return str(value)
    heading = SYNTHETIC + '\n' if result['synthetic'] else ''
    return heading + 'Детерминированный baseline; косвенные связи полноценно не проверяются.\n' + display(result) + '\n'
