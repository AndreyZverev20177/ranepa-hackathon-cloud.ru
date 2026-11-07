"""CLI for Stage 08: transparent golden-set evaluation and dev-only calibration.

Never tunes on test. Does not mutate production configs or publish results.
No internet sources or model downloads unless --semantic is explicitly selected.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

from .clustering import SemanticClusterConfig, cluster_semantically, story_key
from .config import MSK
from .digest import digest
from .embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingConfig, EmbeddingService, EmbeddingError
from .evaluation import (EvaluationError, choose_threshold, cluster_metrics, cluster_error_examples,
                         late_repost_audit, make_gold_template, mapping_from_groups,
                         ranking_metrics, read_gold, require_dev_test, review_metrics)
from .io_utils import write_json_atomic
from .partner import adapt
from .preprocessing import brand, prepare_publications
from .scoring import PublicationScoreWeights
from .source_authority import SourceAuthorityRegistry, SourceAuthorityError, load_scoring_rules
from .story_scoring import load_story_weights
from .validation import InputError, load, parse_date, timestamp


def _grid(value: str) -> list[float]:
    try:
        result = sorted(set(float(s.strip()) for s in value.split(',')))
    except ValueError:
        raise EvaluationError('Пороги должны быть числами, разделёнными запятыми') from None
    if not result or any(not (0 < v < 1) for v in result):
        raise EvaluationError('Пороги должны лежать строго между 0 и 1')
    return result


def _quality(predictions: dict[str, str], labels: dict, *, split: str) -> dict[str, Any]:
    return cluster_metrics(predictions, labels, split)


def build_evaluation(
    rows: list[dict[str, Any]], date_msk: str, gold_path: Path, *,
    run_semantic: bool, calibrate: bool,
    high_grid: list[float], low_grid: list[float], min_precision: float,
    model_name: str, model_revision: str | None, device: str | None, cache_dir: Path,
    sources_file: Path, scoring_rules_file: Path, story_rules_file: Path,
) -> dict[str, Any]:
    day = parse_date(date_msk)
    t0 = perf_counter()
    relevant = [r for r in rows if timestamp(r['published_at']).astimezone(MSK).date() == day and brand(r['text']) is not None]
    labels, coverage = read_gold(gold_path, day=date_msk, available_ids={r['message_id'] for r in relevant})
    if calibrate and not run_semantic:
        raise EvaluationError('Подбор порогов требует --semantic')
    if calibrate:
        if coverage['review_status'] != 'verified':
            raise EvaluationError('Для --calibrate вручную проверьте разметку и установите review_status=verified')
        require_dev_test(labels)
    baseline_groups: dict[str, list[str]] = {}
    for r in relevant:
        baseline_groups.setdefault(story_key(r, r['text']), []).append(r['message_id'])
    base_pred, base_members = mapping_from_groups(baseline_groups.items())
    base_result = digest(rows, day, scoring='baseline')
    t_base = perf_counter()
    rules = load_scoring_rules(scoring_rules_file)
    registry = SourceAuthorityRegistry.from_json(sources_file, rules=rules)
    pub_weights = PublicationScoreWeights.from_settings(rules['weights'])
    story_weights = load_story_weights(story_rules_file)
    baseline_story = digest(rows, day, scoring='story-v1', source_registry=registry,
                            scoring_weights=pub_weights, story_weights=story_weights)
    t_story = perf_counter()
    splits = [name for name in ('dev', 'test') if any(l.split == name for l in labels.values())]
    report: dict[str, Any] = {
        'schema': 'cloud-digest-evaluation-v1', 'date_msk': date_msk,
        'source_records': len(rows), 'brand_candidates_on_day': len(relevant),
        'gold_coverage': {**coverage, 'unjudged_candidates': len(relevant) - len(labels),
                          'judged_candidate_fraction': round(len(labels) / len(relevant), 5) if relevant else None},
        'quality_is_from_human_gold_only': True,
        'baseline': {
            'clustering': {split: _quality(base_pred, labels, split=split) for split in splits},
            'error_examples': {split: cluster_error_examples(base_pred, labels, split) for split in splits},
            'ranking_baseline_v1': {split: ranking_metrics(base_result, base_members, labels, split) for split in splits},
            'ranking_story_v1': {split: ranking_metrics(baseline_story, base_members, labels, split) for split in splits},
        },
        'late_major_repost_audit': late_repost_audit(relevant, labels, registry, pub_weights, story_weights),
        'timings_seconds': {'baseline_digest': round(t_base-t0, 4), 'baseline_story_digest': round(t_story-t_base, 4)},
        'notes': [
            'Проверка кластеризации — только на вручную размеченных публикациях, но предсказания вычислены на всей дневной выборке.',
            'Recall@30 — доля размеченных релевантных сюжетов, обнаруженных в реальном top-30. Неполная разметка не даёт оценить полноту по всему потоку.',
            'NDCG — только по карточкам с разметкой; не выдавать за NDCG по всей выгрузке.',
            'Тесты поздних репостов ограничены вручную размеченными историями; это не оценка реального уникального охвата.',
            'При статусе разметки draft метрики диагностические и не подтверждают качество продукта.',
        ],
    }
    if run_semantic:
        start_embed = perf_counter()
        service = EmbeddingService(EmbeddingConfig(model_name=model_name, revision=model_revision,
                                                   device=device, cache_dir=cache_dir))
        prepared = prepare_publications(relevant)
        vectors = service.encode(prepared)
        report['timings_seconds']['embedding_generation_or_cache'] = round(perf_counter()-start_embed, 4)
        report['model'] = {'name': model_name, 'revision': model_revision,
                           'vector_count': sum(x.vector is not None for x in vectors),
                           'missing_vectors': sum(x.vector is None for x in vectors),
                           'cache_hits_including_within_batch': sum(x.from_cache for x in vectors)}
        if calibrate:
            candidate_rows = []
            for high in high_grid:
                for low in low_grid:
                    if low >= high:
                        continue
                    cfg = SemanticClusterConfig(high_threshold=high, low_threshold=low)
                    res = cluster_semantically(prepared, vectors, cfg)
                    predictions, _ = mapping_from_groups((c.cluster_id, c.message_ids) for c in res.clusters)
                    candidate_rows.append({'high': high, 'low': low,
                                           'dev_clusters': cluster_metrics(predictions, labels, 'dev'),
                                           'dev_review': review_metrics(res, predictions, labels, 'dev')})
            if not candidate_rows:
                raise EvaluationError('Не осталось допустимых пар high/low')
            chosen = choose_threshold(candidate_rows, labels, min_precision=min_precision)
            report['calibration'] = {
                'split_used': 'dev', 'min_precision': min_precision, 'evaluated_grid': candidate_rows,
                'decision': chosen, 'test_not_used_for_selection': True,
            }
            selected = chosen['selected']
            if selected is None:
                report['calibration']['status'] = 'no_configuration_passed_quality_gate'
                # Diagnostic only: do not evaluate test with a selected-on-test model.
                report['timings_seconds']['total'] = round(perf_counter()-t0, 4)
                return report
            high, low = selected['high'], selected['low']
        else:
            if len(high_grid) != 1 or len(low_grid) != 1:
                raise EvaluationError('Без --calibrate укажите один high и один low')
            high, low = high_grid[0], low_grid[0]
        cfg = SemanticClusterConfig(high_threshold=high, low_threshold=low)
        res = cluster_semantically(prepared, vectors, cfg)
        pred, members = mapping_from_groups((c.cluster_id, c.message_ids) for c in res.clusters)
        start_semantic_digest = perf_counter()
        # Reuses the EmbeddingService object and disk cache for actual digest ranking.
        sem_story = digest(rows, day, semantic=True, embedding_service=service,
                           clustering_config=cfg, scoring='story-v1',
                           source_registry=registry, scoring_weights=pub_weights,
                           story_weights=story_weights)
        report['timings_seconds']['semantic_story_digest'] = round(perf_counter()-start_semantic_digest, 4)
        report['semantic'] = {'thresholds': {'high': high, 'low': low},
                              'clusters': len(res.clusters),
                              'review_pairs': len(res.review_pairs),
                              'truncated_review_pairs': res.truncated_review_pairs,
                              'skipped_without_vector': res.skipped_without_vector,
                              'clustering': {split: cluster_metrics(pred, labels, split) for split in splits},
                              'error_examples': {split: cluster_error_examples(pred, labels, split) for split in splits},
                              'review': {split: review_metrics(res, pred, labels, split) for split in splits},
                              'ranking_story_v1': {split: ranking_metrics(sem_story, members, labels, split) for split in splits}}
    report['timings_seconds']['total'] = round(perf_counter()-t0, 4)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description='Этап 08: разметка, метрики и контрольная калибровка Cloud.ru')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--partner-export', action='store_true')
    p.add_argument('--date', required=True)
    p.add_argument('--output', type=Path, required=True, help='Путь JSON-отчёта или gold-шаблона')
    p.add_argument('--make-template', action='store_true', help='Создать незаполненный шаблон разметки (не метрики)')
    p.add_argument('--gold', type=Path, help='JSON вручную размеченных сюжетов')
    p.add_argument('--semantic', action='store_true', help='Загрузить модель / использовать кэш embeddings')
    p.add_argument('--calibrate', action='store_true', help='Выбрать high/low ТОЛЬКО по dev, проверить на test')
    p.add_argument('--high-grid', default=None, help='Например 0.84,0.88,0.92; без --calibrate одно число')
    p.add_argument('--low-grid', default=None, help='Например 0.72,0.78,0.82; без --calibrate одно число')
    p.add_argument('--min-precision', type=float, default=0.90)
    p.add_argument('--embedding-model', default=DEFAULT_EMBEDDING_MODEL)
    p.add_argument('--embedding-revision', default=None)
    p.add_argument('--embedding-device', default=None)
    p.add_argument('--embedding-cache-dir', type=Path, default=Path('artifacts/embedding_cache'))
    p.add_argument('--sources-file', type=Path, default=Path('partner_data/top25_sources.json'))
    p.add_argument('--scoring-rules', type=Path, default=Path('config/publication_scoring_v2.json'))
    p.add_argument('--story-rules', type=Path, default=Path('config/story_scoring_v1.json'))
    args = p.parse_args(argv)
    try:
        day = parse_date(args.date).isoformat()
        if args.output.resolve() == args.input.resolve() or (args.gold and args.output.resolve() == args.gold.resolve()):
            raise EvaluationError('Выход не может совпадать с входным файлом или файлом разметки')
        rows = adapt(args.input) if args.partner_export else load(args.input)[0]
        if args.make_template:
            write_json_atomic(args.output, make_gold_template(rows, day))
            print(f'Шаблон разметки записан: {args.output}. Требуется ручная разметка.')
            return 0
        if args.gold is None:
            raise EvaluationError('Укажите --gold с вручную заполненным JSON либо --make-template')
        highs = _grid(args.high_grid or ('0.84,0.88,0.92' if args.calibrate else '0.88'))
        lows = _grid(args.low_grid or ('0.72,0.78,0.82' if args.calibrate else '0.78'))
        report = build_evaluation(rows, day, args.gold,
                                  run_semantic=args.semantic, calibrate=args.calibrate,
                                  high_grid=highs, low_grid=lows, min_precision=args.min_precision,
                                  model_name=args.embedding_model, model_revision=args.embedding_revision,
                                  device=args.embedding_device, cache_dir=args.embedding_cache_dir,
                                  sources_file=args.sources_file, scoring_rules_file=args.scoring_rules,
                                  story_rules_file=args.story_rules)
        write_json_atomic(args.output, report)
        print('Отчёт:', args.output)
        print('Размечено:', report['gold_coverage']['reviewed'],
              '| Дневных кандидатов:', report['brand_candidates_on_day'])
        if 'calibration' in report:
            print('Подбор порогов:', report['calibration']['decision']['selected'] or 'нет допустимой конфигурации')
        for name in ('baseline', 'semantic'):
            if name in report:
                print(name + ':', json.dumps(report[name]['clustering'], ensure_ascii=False))
        return 0
    except (InputError, EvaluationError, EmbeddingError, SourceAuthorityError, ValueError) as exc:
        print(f'Ошибка этапа 08: {exc}', file=sys.stderr)
        return 2
    except (OSError, UnicodeError):
        print('Ошибка этапа 08: чтение или запись файлов', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
