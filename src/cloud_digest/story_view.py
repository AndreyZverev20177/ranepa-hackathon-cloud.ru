"""Представление истории распространения сюжета для редактора.

Не пересчитывает Story Score, не меняет состав кластеров и не подтверждает
истинность новых фактов. Использует только результаты story_scoring.py.
"""
from __future__ import annotations

from typing import Any, Mapping

from .preprocessing import prepare_publication
from .validation import timestamp


BONUS_LABELS: dict[str, str] = {
    'best_publication': 'Лучшая отдельная публикация',
    'distribution': 'Распространение по разным площадкам',
    'additional_reach_proxy': 'Дополнительная аудитория (оценочный прокси)',
    'additional_high_authority_domains': 'Дополнительные сильные издатели',
    'new_fact_signals': 'Новые фактические сигналы (не проверены извне)',
}


def _excerpt(value: str, limit: int) -> str:
    """Не превращаем обрезанный фрагмент в якобы дословный заголовок."""
    value = ' '.join(value.split()).strip()
    if len(value) <= limit:
        return value
    return value[:limit - 1].rstrip() + '…'


def build_story_card(
    representative: Mapping[str, Any],
    original_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Формирует читаемую карточку без вымышленных метаданных.

    representative — итоговый элемент story-v1, уже содержащий Story Score.
    original_row — исходная публикация выбранного представителя.
    """
    if representative.get('story_score_version') != 'story-v1':
        raise ValueError('Карточка сюжета доступна только для story-v1')
    if original_row['message_id'] != representative['message_id']:
        raise ValueError('Основная публикация не соответствует исходной записи')

    prepared = prepare_publication(dict(original_row))
    title = _excerpt(prepared.title, 160) if prepared.title else None
    if not title:
        # Длинная первая строка — не заголовок, но её можно честно показать
        # как извлечённый фрагмент вместо бесполезного технического ID.
        first_line = next((line.strip() for line in prepared.clean_text.splitlines()
                           if line.strip()), '')
        if first_line:
            title = _excerpt(first_line, 140)
            title_kind = 'truncated_first_line_excerpt'
        else:
            title = f"Сюжет по публикации {representative['message_id']}"
            title_kind = 'technical_fallback'
    else:
        title_kind = 'explicit_or_first_line_of_main_publication'

    fact_signals = representative['novel_fact_signals']
    facts_by_message: dict[str, list[dict[str, str]]] = {}
    for fact in fact_signals:
        facts_by_message.setdefault(fact['message_id'], []).append({
            'metric': fact['metric'], 'value': fact['value'],
        })

    publications = representative['story_publications']
    primary_id = representative['best_publication_id']
    earliest_id = min(publications, key=lambda p: (timestamp(p['published_at']), p['message_id']))['message_id']
    timeline = [
        {
            'message_id': item['message_id'],
            'published_at': item['published_at'],
            'source_name': item['source_name'],
            'url': item['source_url'],
            'publication_score': item['publication_score'],
            'audience': item['audience'],
            'is_first': item['message_id'] == earliest_id,
            'is_main_source': item['message_id'] == primary_id,
            'new_fact_signals': facts_by_message.get(item['message_id'], []),
        }
        for item in sorted(publications, key=lambda p: (timestamp(p['published_at']), p['message_id']))
    ]
    main = next(item for item in timeline if item['is_main_source'])
    breakdown = representative['story_score_breakdown']
    contributions = [
        {'code': code, 'label': BONUS_LABELS.get(code, code), 'points': points}
        for code, points in breakdown.items()
    ]

    return {
        'schema_version': 'story-card-v1',
        'cluster_id': representative['cluster_id'],
        'title': title,
        'title_source': title_kind,
        'description_excerpt': _excerpt(prepared.clean_text, 300),
        'main_source': {
            'message_id': main['message_id'],
            'source_name': main['source_name'],
            'url': main['url'],
            'published_at': main['published_at'],
            'publication_score': main['publication_score'],
            'source_authority_score': representative['source_authority_score'],
            'selection_rule': 'highest_publication_score_then_authority_then_time',
        },
        'priority': {
            'story_score': representative['story_score'],
            'score_before_cap': representative['story_score_uncapped'],
            'publication_score_base': breakdown['best_publication'],
            'contributions': contributions,
            'capped_at_100': representative['story_score_uncapped'] > 100,
            'explanation': representative['story_explanation'],
        },
        'spread': {
            'publication_count': len(timeline),
            'distinct_outlets': representative['distinct_outlets'],
            'first_published_at': representative['first_published_at'],
            'last_published_at': representative['last_published_at'],
            'max_reported_audience': representative['max_reported_audience'],
            'additional_audience_proxy': representative['additional_audience_proxy'],
            'audience_measure': 'potential_audience_proxy_not_unique_reach',
        },
        'timeline': timeline,
        'fact_signals': fact_signals,
        'conflicting_claim_signals': representative['conflicting_claim_signals'],
        'incompatible_member_pairs': representative['incompatible_member_pairs'],
        'review': {
            'required': representative['human_review_needed'],
            'reasons': representative['review_reasons'],
            'contains_crisis': representative['story_contains_crisis'],
        },
        'disclaimers': [
            'Дополнительная аудитория — прокси, не уникальный охват.',
            'Фактические сигналы извлечены из текстов, но не подтверждены извне.',
            'Название взято из основной публикации, не сгенерировано как редакторский вывод.',
        ],
    }


def render_story_digest(result: Mapping[str, Any]) -> str:
    """Человекочитаемый TXT вместо рекурсивного вывода всего JSON."""
    parts: list[str] = []
    if result['synthetic']:
        parts.append('СИНТЕТИЧЕСКИЙ ПРИМЕР — ВЫМЫСЕЛ.')
    parts.extend([
        f"Дайджест Cloud.ru — {result['digest_date']}",
        'Режим: Story Score v1 | одна карточка на кластер',
        'Охват — оценочный прокси, не уникальная аудитория; фактические сигналы не верифицированы.',
        '',
    ])
    counts = result['summary_counts']
    parts.append(
        f"За день: {counts['day_records']} публикаций; "
        f"упоминаний бренда: {counts['brand_candidates']}; "
        f"сюжетов: {counts['clusters']}; "
        f"в дайджесте: {counts['selected_items']}."
    )
    if not result['items']:
        parts.extend(['', result['empty_message'] or 'Нет сюжетов для отображения.', ''])
        return '\n'.join(parts) + '\n'
    for item in result['items']:
        view = item['story_card']
        priority = view['priority']
        spread = view['spread']
        main = view['main_source']
        parts.extend([
            '',
            f"{item['rank_in_digest']:02d}. {view['title']}",
            f"Story Score: {priority['story_score']}/100 "
            f"(баллы до ограничения: {priority['score_before_cap']})",
            f"Главный источник: {main['source_name']} | {main['published_at']} "
            f"| Publication Score {main['publication_score']}/100",
            f"Ссылка: {main['url']}",
        ])
        if view['description_excerpt']:
            parts.append(f"Фрагмент основной публикации: {view['description_excerpt']}")
        visible_bonuses = [c for c in priority['contributions'] if c['points'] > 0]
        parts.append('Почему такой приоритет: ' + (
            '; '.join(f"{c['label']}: +{c['points']}" for c in visible_bonuses)
            or 'Нет положительных компонентов оценки'
        ))
        if priority['capped_at_100']:
            parts.append('Итог ограничен 100 баллами.')
        parts.append(
            f"Распространение: {spread['publication_count']} публикаций, "
            f"{spread['distinct_outlets']} различимых площадок; "
            f"с {spread['first_published_at']} до {spread['last_published_at']}."
        )
        parts.append('История публикаций (по времени):')
        for publication in view['timeline']:
            tags = []
            if publication['is_first']:
                tags.append('первая')
            if publication['is_main_source']:
                tags.append('главная')
            marker = f" [{', '.join(tags)}]" if tags else ''
            parts.append(
                f"  • {publication['published_at']} | {publication['source_name']}"
                f"{marker} | {publication['publication_score']}/100"
            )
            parts.append(f"    {publication['url']}")
            if publication['new_fact_signals']:
                parts.append('    Новые сигналы (не проверены): ' + '; '.join(
                    f"{f['metric']} = {f['value']}" for f in publication['new_fact_signals']
                ))
        if view['review']['required']:
            parts.append('Требуется проверка: ' + (
                ', '.join(view['review']['reasons']) or 'причина не указана'
            ))
        if view['conflicting_claim_signals']:
            parts.append(f"Конфликтующих числовых утверждений: {len(view['conflicting_claim_signals'])}.")
        if view['incompatible_member_pairs']:
            parts.append(f"Несовместимых пар событий: {len(view['incompatible_member_pairs'])}.")
    parts.append('')
    return '\n'.join(parts) + '\n'
