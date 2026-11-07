"""Проверка дат и входных CSV/JSON, в том числе записей партнёрского адаптера."""
import csv
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from .config import REQUIRED, STRINGS, METRICS, KNOWN, BA


class InputError(ValueError):
    """Ошибка, не раскрывающая исходный текст записи."""


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


def validate_records(
    raw: Any, *, is_csv: bool = False, known_extras: set[str] | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    """Единая валидация для CSV, JSON и partner.adapt().

    Неизвестные поля игнорируются; список их имён возвращается отдельно.
    Все ошибки содержат только номер записи и имя поля.
    """
    if not isinstance(raw, list):
        raise InputError('JSON: ожидается массив объектов')
    result: list[dict[str, Any]] = []
    extras = set() if known_extras is None else set(known_extras)
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
    return validate_records(raw, is_csv=is_csv, known_extras=extras)
