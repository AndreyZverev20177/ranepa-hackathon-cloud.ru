"""Безопасная загрузка пользовательских выгрузок через локальную панель.

Не выполняет обработку или скоринг: только проверяет поддерживаемый формат
и сохраняет исходные байты в локальном results/ui/uploads.
"""
from __future__ import annotations

import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
from typing import Literal

from .config import REQUIRED

DatasetKind = Literal['partner', 'normalized']
MAX_DATASET_BYTES = 30 * 1024 * 1024


def detect_dataset(data: bytes, filename: str) -> DatasetKind:
    """Возвращает тип входных данных или понятную ошибку для UI.

    JSON: массив исходных сообщений Brand Analytics либо массив записей
    нормализованной схемы Cloud Digest. CSV: только нормализованная схема.
    """
    ext = Path(filename).suffix.casefold()
    if ext not in ('.json', '.csv'):
        raise ValueError('Нужен датасет .json или .csv')
    if not data:
        raise ValueError('Загруженный файл пуст')
    if len(data) > MAX_DATASET_BYTES:
        raise ValueError('Максимальный размер датасета — 30 МБ')
    try:
        decoded = data.decode('utf-8-sig')
    except UnicodeError:
        raise ValueError('Для загрузки нужна кодировка UTF-8') from None

    if ext == '.json':
        try:
            records = json.loads(decoded)
        except json.JSONDecodeError:
            raise ValueError('Некорректный JSON') from None
        if isinstance(records, dict) and 'items' in records and 'digest_date' in records:
            raise ValueError('Это готовый дайджест; загрузи его через поле «Готовый дайджест»')
        if not isinstance(records, list) or not records or not isinstance(records[0], dict):
            raise ValueError('Ожидается непустой JSON-массив публикаций')
        first = records[0]
        if {'timeCreate', 'id', 'url'} <= first.keys():
            return 'partner'
        if set(REQUIRED) <= first.keys():
            return 'normalized'
        raise ValueError('Неизвестная JSON-схема. Поддерживаются выгрузка BA и нормализованный Cloud Digest JSON')

    try:
        reader = csv.DictReader(StringIO(decoded, newline=''), strict=True)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)) or not set(REQUIRED) <= set(columns):
            raise ValueError('CSV: нужны столбцы ' + ', '.join(REQUIRED))
        if next(reader, None) is None:
            raise ValueError('CSV не содержит публикаций')
    except csv.Error:
        raise ValueError('Некорректный CSV') from None
    return 'normalized'


def save_uploaded_dataset(data: bytes, filename: str, target_dir: Path) -> tuple[Path, DatasetKind]:
    """Сохраняет под безопасным именем по хешу; не использует путь клиента."""
    kind = detect_dataset(data, filename)
    ext = Path(filename).suffix.casefold()
    checksum = hashlib.sha256(data).hexdigest()
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / f'dataset-{checksum[:20]}{ext}'
    if not destination.exists() or destination.read_bytes() != data:
        destination.write_bytes(data)
    return destination, kind
