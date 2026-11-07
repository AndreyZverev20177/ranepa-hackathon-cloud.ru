"""Offline Brand Analytics export adapter; no network access.

Переводим поля партнёра в общую схему и применяем validate_records().
Не подменяем отсутствующие id/URL/текст вымышленными значениями.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from .validation import InputError, validate_records


def adapt(path: Path):
    try:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
    except (json.JSONDecodeError, UnicodeError):
        raise InputError('Партнёрская выгрузка: некорректный JSON/UTF-8') from None
    if not isinstance(data, list):
        raise InputError('Партнёрская выгрузка: ожидается массив объектов')
    out = []
    tones = {-1: 'negative', 0: 'neutral', 1: 'positive',
             '-1': 'negative', '0': 'neutral', '1': 'positive'}
    for i, r in enumerate(data, 1):
        if not isinstance(r, dict):
            raise InputError(f'Запись {i}: неверное поле record')
        t = r.get('timeCreate')
        if type(t) not in (int, float):
            raise InputError(f'Запись {i}: неверное поле timeCreate')
        try:
            dt = datetime.fromtimestamp(t, timezone.utc).isoformat()
        except (ValueError, OverflowError, OSError):
            raise InputError(f'Запись {i}: неверное поле timeCreate') from None
        author = r.get('authorObject') or {}
        if not isinstance(author, dict):
            raise InputError(f'Запись {i}: неверное поле authorObject')

        identifier = r.get('id')
        if not isinstance(identifier, (int, str)) or isinstance(identifier, bool) or not str(identifier).strip():
            raise InputError(f'Запись {i}: неверное поле id')
        source_url = r.get('url')
        if not isinstance(source_url, str) or not source_url.strip():
            raise InputError(f'Запись {i}: неверное поле url')
        body = r.get('text') or r.get('title')
        if not isinstance(body, str) or not body.strip():
            raise InputError(f'Запись {i}: неверное поле text/title')

        def number(field: str):
            value = r.get(field)
            if value is None or value == '':
                return None
            if type(value) is int and value >= 0:
                return value
            if isinstance(value, str) and value.isascii() and value.isdecimal():
                return int(value)
            raise InputError(f'Запись {i}: неверное поле {field}')

        tone = r.get('toneMark')
        if tone is not None and (type(tone) not in (int, str) or tone not in tones):
            raise InputError(f'Запись {i}: неверное поле toneMark')
        out.append(dict(
            message_id=str(identifier), published_at=dt,
            source_type=str(r.get('hubtype') or 'unknown'),
            source_name=str(author.get('fullname') or r.get('hub') or 'unknown'),
            url=source_url, text=body,
            author_name=str(author.get('fullname') or ''),
            audience=number('audienceCount'), views=number('viewsCount'),
            likes=number('likesCount'), comments=number('commentsCount'),
            reposts=number('repostsCount'), ba_sentiment=tones.get(tone),
            ba_topic_hit=None, language=str(r.get('language') or ''),
            geo=str(r.get('country') or ''), title=str(r.get('title') or '') or None,
        ))
    validated, _ = validate_records(out)
    return validated
