#!/usr/bin/env python3
"""Compare two clustering modes evaluated by cloud_digest.evaluate_cli.

Reads a genuine evaluation JSON. Never estimates semantic numbers when unavailable.
Standard library only; does not change the Cloud Digest processing pipeline.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def pct(v: Any) -> str:
    return '—' if v is None else f'{100 * float(v):.2f}%'.replace('.', ',')


def delta(a: Any, b: Any) -> str:
    if a is None or b is None:
        return '—'
    value = (float(b) - float(a)) * 100
    return f'{value:+.2f}'.replace('.', ',') + ' п.п.'


def raw(report: dict[str, Any], mode: str, split: str, group: str, key: str) -> Any:
    try:
        entry = report['baseline'] if mode == 'baseline' else report['semantic']
        return entry[group][split]
    except (KeyError, TypeError):
        return None


def value(report: dict[str, Any], mode: str, split: str, path: tuple[str, ...]) -> Any:
    try:
        entry = report['baseline'] if mode == 'baseline' else report['semantic']
        item = entry['clustering'][split]
        for key in path:
            item = item[key]
        return item
    except (KeyError, TypeError):
        return None


def ranking(report: dict[str, Any], mode: str, split: str, metric: str) -> Any:
    try:
        entry = report['baseline'] if mode == 'baseline' else report['semantic']
        return entry['ranking_story_v1'][split][metric]
    except (KeyError, TypeError):
        return None


METRICS = [
    ('Pairwise Precision', ('pairwise', 'precision'), True),
    ('Pairwise Recall', ('pairwise', 'recall'), True),
    ('Pairwise F1', ('pairwise', 'f1'), True),
    ('B-cubed Precision', ('b_cubed', 'precision'), True),
    ('B-cubed Recall', ('b_cubed', 'recall'), True),
    ('B-cubed F1', ('b_cubed', 'f1'), True),
    ('Ложные объединения (пары)', ('false_merge_pairs',), False),
    ('Пропущенные совпадения (пары)', ('missed_merge_pairs',), False),
]


def build(report: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    if report.get('schema') != 'cloud-digest-evaluation-v1':
        raise ValueError('Неверная схема отчёта Cloud Digest')
    status = report.get('gold_coverage', {}).get('review_status', 'unknown')
    model_ran = 'semantic' in report
    model_name = report.get('model', {}).get('name', 'не указан')
    lines = [
        '# Cloud Digest: baseline-v1 vs semantic clustering', '',
        f"**Дата набора:** {report.get('date_msk', 'не указана')}  ",
        f"**Разметка:** {status}  ",
        f"**Кандидатов BA:** {report.get('brand_candidates_on_day', '—')}  ",
        f"**Модель:** {model_name if model_ran else 'не запущена'}  ",
        '',
    ]
    if status != 'verified':
        lines += ['> ВНИМАНИЕ: разметка DRAFT; метрики диагностические, не подтверждённые человеком.', '']
    if not model_ran:
        lines += ['> Семантические метрики не получены: отсутствует успешный прогон embeddings. Пробелы не заменены расчётными оценками.', '']
    else:
        th = report['semantic'].get('thresholds', {})
        lines += [f"Пороги: high={th.get('high', '—')}, low={th.get('low', '—')}. Сравнение Story Score проводится при одинаковой формуле и разных кластерах.", '']
    rows: list[dict[str, str]] = []
    for split in ('dev', 'test'):
        lines += [f'## {split.upper()}', '', '| Метрика | Baseline | Semantic | Изменение |', '|---|---:|---:|---:|']
        for name, path, is_ratio in METRICS:
            a, b = value(report, 'baseline', split, path), value(report, 'semantic', split, path)
            fa = pct(a) if is_ratio else str(a) if a is not None else '—'
            fb = pct(b) if is_ratio else str(b) if b is not None else '—'
            d = delta(a,b) if is_ratio else (f'{int(b)-int(a):+d}' if a is not None and b is not None else '—')
            lines.append(f'| {name} | {fa} | {fb} | {d} |')
            rows.append({'split':split, 'metric':name, 'baseline':fa, 'semantic':fb, 'difference':d})
        for name, key in [('Story Recall@30', 'recall_at_k'), ('Story NDCG@30 (только размеченные карточки)', 'ndcg_at_k_judged_only')]:
            a,b=ranking(report,'baseline',split,key),ranking(report,'semantic',split,key)
            fa,fb,d=pct(a),pct(b),delta(a,b)
            lines.append(f'| {name} | {fa} | {fb} | {d} |')
            rows.append({'split':split,'metric':name,'baseline':fa,'semantic':fb,'difference':d})
        lines.append('')
    lines += ['## Ограничения', '',
              '- При подборе high/low используются только `dev`-метки. `test` должен оставаться независимым.',
              '- Статус `draft` нельзя вручную менять на `verified` без независимой редакторской проверки.',
              '- Один крупный сюжет может доминировать в попарных метриках; дополнительно анализируйте B-cubed и ошибки на малых сюжетах.',
              '- Recall@30 оценён по размеченным сюжетам, а NDCG — только по карточкам, которые сопоставились с разметкой.',
              '- Без кризисных сюжетов `critical_recall_at_k` не определён.',
              '- Модельное качество проверяется на реальных embeddings, а не на имитации в unit-тестах.', '']
    if 'calibration' in report:
        selected = report['calibration'].get('decision', {}).get('selected')
        lines += ['## Калибровка', '', f"Выбранные на dev пороги: `{selected}`" if selected else 'Ни одна конфигурация не прошла порог точности на dev.', '']
    return '\n'.join(lines)+'\n', rows


def main() -> int:
    p=argparse.ArgumentParser(description='Generate Markdown and CSV comparison from Cloud Digest evaluation')
    p.add_argument('--report',required=True,type=Path)
    p.add_argument('--output-prefix',required=True,type=Path)
    args=p.parse_args()
    report=json.loads(args.report.read_text(encoding='utf-8-sig'))
    md, rows=build(report)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    md_path=args.output_prefix.with_suffix('.md')
    csv_path=args.output_prefix.with_suffix('.csv')
    md_path.write_text(md, encoding='utf-8')
    with csv_path.open('w', encoding='utf-8-sig', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=['split','metric','baseline','semantic','difference'], delimiter=';')
        writer.writeheader(); writer.writerows(rows)
    print('Сравнение:',md_path, '\nТаблица:',csv_path)
    print('Semantic:', 'успешно оценён' if 'semantic' in report else 'НЕТ РЕЗУЛЬТАТОВ; проверьте загрузку модели или калибровку')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
