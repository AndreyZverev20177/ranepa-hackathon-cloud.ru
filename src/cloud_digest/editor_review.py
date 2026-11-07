"""Локальный журнал редакторских решений над готовым дайджестом.

Не меняет исходный JSON BA, саму модель или importance_score.
Позволяет воспроизводимо экспортировать применённые редакторские решения.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

SCHEMA = 'editor-reviews-v1'
STATUSES = frozenset({'pending', 'approved', 'needs_review', 'excluded'})
TONES = frozenset({'as_model', 'positive', 'neutral', 'negative'})
BRAND_LINKS = frozenset({'as_model', 'direct', 'historical', 'none'})
CLUSTER_ACTIONS = frozenset({'none', 'check_merge', 'check_split'})


def review_key(day: str, cluster_id: str) -> str:
    """Стабильный ключ только в рамках даты и режима кластеризации."""
    if not isinstance(day, str) or not isinstance(cluster_id, str) or not day or not cluster_id:
        raise ValueError('Нужны дата и идентификатор сюжета')
    return f'{day}:{cluster_id}'


def blank_store() -> dict[str, Any]:
    return {'schema_version': SCHEMA, 'reviews': {}}


def load_store(path: Path) -> dict[str, Any]:
    if not path.exists():
        return blank_store()
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'Не удалось прочитать журнал правок: {exc}') from exc
    if not isinstance(payload, dict) or payload.get('schema_version') != SCHEMA or not isinstance(payload.get('reviews'), dict):
        raise ValueError('Несовместимый формат журнала редакторских решений')
    return payload


def _string(value: Any, name: str, max_length: int) -> str:
    if not isinstance(value, str) or len(value) > max_length:
        raise ValueError(f'{name}: ожидалась строка длиной не более {max_length}')
    return value.strip()


def validate_review(review: Mapping[str, Any]) -> dict[str, str]:
    """Валидирует ввод из UI и выравнивает структуру сохраняемой правки."""
    if not isinstance(review, Mapping):
        raise ValueError('Неверная редакторская правка')
    values = {}
    for field, options, default in (
        ('status', STATUSES, 'pending'),
        ('tone', TONES, 'as_model'),
        ('brand_link', BRAND_LINKS, 'as_model'),
        ('cluster_action', CLUSTER_ACTIONS, 'none'),
    ):
        value = review.get(field, default)
        if not isinstance(value, str) or value not in options:
            raise ValueError(f'{field}: неизвестное значение')
        values[field] = value
    values['summary'] = _string(review.get('summary', ''), 'summary', 1000)
    values['notes'] = _string(review.get('notes', ''), 'notes', 2000)
    values['cluster_target'] = _string(review.get('cluster_target', ''), 'cluster_target', 150)
    values['split_message_ids'] = _string(review.get('split_message_ids', ''), 'split_message_ids', 1000)
    if values['cluster_action'] == 'check_merge' and not values['cluster_target']:
        raise ValueError('Для запроса объединения укажи cluster_id второго сюжета')
    if values['cluster_action'] == 'check_split' and not values['split_message_ids']:
        raise ValueError('Для запроса разделения укажи ID отделяемых публикаций')
    return values


def save_review(path: Path, day: str, cluster_id: str, review: Mapping[str, Any]) -> dict[str, Any]:
    """Атомарное сохранение для одного локального редактора, без общей БД."""
    key = review_key(day, cluster_id)
    values = validate_review(review)
    store = load_store(path)
    previous = store['reviews'].get(key, {})
    created = previous.get('created_at') if isinstance(previous, dict) else None
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    store['reviews'][key] = {
        **values,
        'digest_date': day,
        'cluster_id': cluster_id,
        'created_at': created or now,
        'updated_at': now,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, prefix='.review-', suffix='.tmp', delete=False) as tmp:
            json.dump(store, tmp, ensure_ascii=False, indent=2)
            tmp.write('\n')
            temp_name = tmp.name
        os.replace(temp_name, path)
    finally:
        if temp_name is not None and os.path.exists(temp_name):
            os.unlink(temp_name)
    return store['reviews'][key]


def review_of(store: Mapping[str, Any], day: str, item: Mapping[str, Any]) -> dict[str, Any]:
    key = review_key(day, str(item['cluster_id']))
    data = store.get('reviews', {}).get(key, {})
    return dict(data) if isinstance(data, dict) else {}


def effective_status(item: Mapping[str, Any], review: Mapping[str, Any]) -> str:
    value = review.get('status')
    if value in STATUSES and value != 'pending':
        return str(value)
    return 'needs_review' if item.get('human_review_needed') else 'pending'


def export_editorial_digest(result: Mapping[str, Any], store: Mapping[str, Any]) -> dict[str, Any]:
    """Создаёт НОВЫЙ артефакт с решениями редактора, не подменяя скор модели.

    Оригинальные importance_score, sentiment, cluster_id сохраняются.
    Исключённые из редакторского списка сюжеты указаны отдельно.
    """
    day = str(result['digest_date'])
    ready = []
    excluded = []
    for original in result.get('items', []):
        item = deepcopy(original)
        correction = review_of(store, day, item)
        status = effective_status(item, correction)
        final_tone = correction.get('tone', 'as_model')
        final_brand = correction.get('brand_link', 'as_model')
        editorial = {
            'status': status,
            'final_tone': item.get('sentiment') if final_tone == 'as_model' else final_tone,
            'final_brand_link': item.get('brand_link') if final_brand == 'as_model' else final_brand,
            'summary': correction.get('summary') or item.get('story_card', {}).get('description_excerpt') or item.get('summary_160_300') or '',
            'notes': correction.get('notes', ''),
            'cluster_action_request': correction.get('cluster_action', 'none'),
            'cluster_target': correction.get('cluster_target', ''),
            'split_message_ids': correction.get('split_message_ids', ''),
            'updated_at': correction.get('updated_at'),
            'score_recalculated': False,
            'cluster_changes_applied': False,
        }
        item['editorial'] = editorial
        if status == 'excluded':
            excluded.append({'cluster_id': item['cluster_id'], 'message_id': item['message_id'], 'reason': 'editor_excluded'})
        else:
            ready.append(item)
    return {
        'schema_version': 'editorial-digest-v1',
        'digest_date': day,
        'source_rule_version': result.get('rule_version'),
        'editorial_disclaimer': 'Тональность и связи сюжетов исправлены только в редакторском представлении; баллы модели не пересчитаны. Запись в Brand Analytics не выполнялась.',
        'items': ready,
        'excluded_by_editor': excluded,
        'counts': {'visible_stories': len(ready), 'excluded_stories': len(excluded)},
    }
