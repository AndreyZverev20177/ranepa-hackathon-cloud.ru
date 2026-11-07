"""Stage 08: offline golden-set quality metrics and threshold selection.

A gold set MUST be manually annotated. Never infer ground truth from model
outputs, publisher popularity, or duplicated text. No online I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .clustering import ReviewPair, SemanticClusteringResult


class EvaluationError(ValueError):
    """Invalid or insufficient evaluation data; no raw text in errors."""


@dataclass(frozen=True, slots=True)
class Label:
    message_id: str
    story_id: str
    relevance: int
    critical: bool
    split: str


def read_gold(path: Path, *, day: str, available_ids: set[str]) -> tuple[dict[str, Label], dict[str, int]]:
    """Validate hand labels. Incomplete template entries are explicitly ignored.

    Splits are by **entire story**, never by individual articles. This prevents
    near-duplicate leakage when tuning thresholds.
    """
    try:
        raw = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationError('Эталонная разметка: невозможно прочитать JSON') from exc
    if not isinstance(raw, dict) or raw.get('schema') != 'cloud-digest-gold-v1' or raw.get('date_msk') != day:
        raise EvaluationError('Эталонная разметка: неверная схема или дата')
    status = raw.get('review_status', 'unreviewed')
    if status not in ('unreviewed', 'draft', 'verified'):
        raise EvaluationError('Эталонная разметка: review_status должен быть unreviewed/draft/verified')
    records = raw.get('annotations')
    if not isinstance(records, list):
        raise EvaluationError('Эталонная разметка: annotations должен быть массивом')
    labels: dict[str, Label] = {}
    seen_ids: set[str] = set()
    unreviewed = 0
    split_by_story: dict[str, str] = {}
    for i, entry in enumerate(records, 1):
        if not isinstance(entry, dict) or not isinstance(entry.get('message_id'), str):
            raise EvaluationError(f'Разметка {i}: неверный message_id')
        mid = entry['message_id']
        if mid not in available_ids or mid in seen_ids:
            raise EvaluationError(f'Разметка {i}: message_id отсутствует в отборе или повторяется')
        seen_ids.add(mid)
        fields = (entry.get('story_id'), entry.get('relevance'), entry.get('critical'), entry.get('split'))
        if all(x is None for x in fields):
            unreviewed += 1
            continue
        story, relevance, critical, split = fields
        if (not isinstance(story, str) or not story.strip() or type(relevance) is not int
                or not 0 <= relevance <= 3 or type(critical) is not bool or split not in ('dev', 'test')
                or (critical and relevance == 0)):
            raise EvaluationError(f'Разметка {i}: заполните story_id, relevance (0–3), critical и split (dev/test)')
        if story in split_by_story and split_by_story[story] != split:
            raise EvaluationError(f'Разметка {i}: один сюжет не может попадать и в dev, и в test')
        split_by_story[story] = split
        labels[mid] = Label(mid, story, relevance, critical, split)
    if not labels:
        raise EvaluationError('Нет заполненной эталонной разметки; заполните story_id/relevance/critical/split в gold-файле')
    return labels, {'review_status': status, 'reviewed': len(labels), 'unreviewed': unreviewed, 'dev': sum(x.split == 'dev' for x in labels.values()),
                    'test': sum(x.split == 'test' for x in labels.values()), 'stories': len(split_by_story)}


def make_gold_template(rows: Sequence[dict[str, Any]], day: str) -> dict[str, Any]:
    """Populate a review sheet without pretending to know any true labels."""
    from .preprocessing import brand, prepare_publication
    from .validation import timestamp
    from .config import MSK
    entries: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda r: (timestamp(r['published_at']), r['message_id'])):
        if timestamp(row['published_at']).astimezone(MSK).date().isoformat() != day or brand(row['text']) is None:
            continue
        prepared = prepare_publication(row)
        entries.append({
            'message_id': row['message_id'], 'source_name': row['source_name'],
            'published_at': row['published_at'], 'url': row['url'],
            'preview': ' '.join(prepared.clean_text.split())[:180],
            'story_id': None, 'relevance': None, 'critical': None, 'split': None,
        })
    return {'schema': 'cloud-digest-gold-v1', 'date_msk': day, 'review_status': 'unreviewed',
            '_instructions': 'Вручную заполните 4 поля на запись; все пересказы одного события — общий story_id. Разделяйте dev/test ПО СЮЖЕТАМ. Незаполненные строки не оцениваются.',
            'annotations': entries}


def _split_labels(labels: Mapping[str, Label], split: str) -> dict[str, Label]:
    return {mid: label for mid, label in labels.items() if label.split == split}


def cluster_metrics(predictions: Mapping[str, str], labels: Mapping[str, Label], split: str) -> dict[str, Any]:
    """Pairwise P/R/F1 and B-cubed P/R/F1 on *judged items only*."""
    gold = _split_labels(labels, split)
    ids = sorted(gold)
    if not all(mid in predictions for mid in ids):
        raise EvaluationError('Для оценки кластеризации не хватает предсказаний')
    tp = fp = fn = tn = 0
    for i, mid in enumerate(ids):
        for other in ids[i + 1:]:
            same_gold = gold[mid].story_id == gold[other].story_id
            same_pred = predictions[mid] == predictions[other]
            if same_gold and same_pred:
                tp += 1
            elif same_pred:
                fp += 1
            elif same_gold:
                fn += 1
            else:
                tn += 1
    # No positive examples -> recall is undefined, not a fabricated success.
    p = tp / (tp + fp) if tp + fp else (0.0 if tp + fn else None)
    r = tp / (tp + fn) if tp + fn else None
    f = (2 * p * r / (p + r)) if p is not None and r is not None and p + r else (0.0 if r is not None else None)
    group_gold: dict[str, set[str]] = {}
    group_pred: dict[str, set[str]] = {}
    for mid, lab in gold.items():
        group_gold.setdefault(lab.story_id, set()).add(mid)
        group_pred.setdefault(predictions[mid], set()).add(mid)
    bp = br = 0.0
    for mid, lab in gold.items():
        overlap = len(group_gold[lab.story_id] & group_pred[predictions[mid]])
        bp += overlap / len(group_pred[predictions[mid]])
        br += overlap / len(group_gold[lab.story_id])
    bp = bp / len(ids) if ids else None
    br = br / len(ids) if ids else None
    bf = (2 * bp * br / (bp + br)) if bp is not None and br is not None and bp + br else (0.0 if bp is not None else None)
    return {
        'labeled_items': len(ids), 'gold_stories': len(group_gold),
        'pairwise': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'precision': p, 'recall': r, 'f1': f},
        'b_cubed': {'precision': bp, 'recall': br, 'f1': bf},
        'false_merge_pairs': fp, 'missed_merge_pairs': fn,
    }


def review_metrics(result: SemanticClusteringResult, predictions: Mapping[str, str],
                   labels: Mapping[str, Label], split: str) -> dict[str, Any]:
    gold = _split_labels(labels, split)
    missing = {(a, b) for i, a in enumerate(sorted(gold)) for b in sorted(gold)[i + 1:]
               if gold[a].story_id == gold[b].story_id and predictions[a] != predictions[b]}
    reviewed_pairs: set[tuple[str, str]] = set()
    true_matches = 0
    for pair in result.review_pairs:
        if pair.message_id_a not in gold or pair.message_id_b not in gold:
            continue
        key = tuple(sorted((pair.message_id_a, pair.message_id_b)))
        if key in reviewed_pairs:
            continue
        reviewed_pairs.add(key)
        if gold[key[0]].story_id == gold[key[1]].story_id:
            true_matches += 1
    return {'judged_review_pairs': len(reviewed_pairs), 'judged_true_match_pairs': true_matches,
            'precision_of_review_candidates': true_matches / len(reviewed_pairs) if reviewed_pairs else None,
            'missed_positive_pairs_recovered_for_review': len(missing & reviewed_pairs),
            'missed_positive_pair_review_recall': len(missing & reviewed_pairs) / len(missing) if missing else None,
            'total_review_pairs': len(result.review_pairs), 'truncated_review_pairs': result.truncated_review_pairs}


def ranking_metrics(result: Mapping[str, Any], membership: Mapping[str, list[str]],
                    labels: Mapping[str, Label], split: str, *, k: int = 30) -> dict[str, Any]:
    """Recall@K on full top-K; NDCG on judged-only cards, clearly identified."""
    gold = _split_labels(labels, split)
    rel_by_story: dict[str, int] = {}
    critical_by_story: dict[str, bool] = {}
    for lab in gold.values():
        rel_by_story[lab.story_id] = max(rel_by_story.get(lab.story_id, 0), lab.relevance)
        critical_by_story[lab.story_id] = critical_by_story.get(lab.story_id, False) or lab.critical
    relevant = {story for story, rel in rel_by_story.items() if rel > 0}
    critical = {story for story, flag in critical_by_story.items() if flag}
    caught: set[str] = set()
    covered_critical: set[str] = set()
    judged_gains: list[int] = []
    judged_cards = ambiguous = 0
    for item in result.get('items', [])[:k]:
        members = membership.get(item['cluster_id'], [])
        matched = {gold[mid].story_id for mid in members if mid in gold}
        if not matched:
            continue
        judged_cards += 1
        if len(matched) > 1:
            # Do not reward an incorrectly merged cluster for several stories.
            ambiguous += 1
            judged_gains.append(0)
            continue
        story = next(iter(matched))
        gain = rel_by_story[story] if story not in caught else 0
        judged_gains.append(gain)
        if gain > 0:
            caught.add(story)
        if story in critical:
            covered_critical.add(story)
    def dcg(gains: Iterable[int]) -> float:
        return sum((2**grade - 1) / math.log2(i + 2) for i, grade in enumerate(gains))
    ideal = sorted(rel_by_story.values(), reverse=True)[:k]
    denom = dcg(ideal)
    return {
        'k': k, 'top_k_returned': min(k, len(result.get('items', []))),
        'judged_cards_in_top_k': judged_cards, 'ambiguous_mixed_story_cards': ambiguous,
        'relevant_gold_stories': len(relevant), 'found_relevant_stories': len(caught),
        'recall_at_k': len(caught) / len(relevant) if relevant else None,
        'critical_gold_stories': len(critical), 'found_critical_stories': len(covered_critical),
        'critical_recall_at_k': len(covered_critical) / len(critical) if critical else None,
        'ndcg_at_k_judged_only': dcg(judged_gains) / denom if denom else None,
        'ndcg_note': 'NDCG calculated on the judged cards only; full-corpus NDCG requires complete judgments.',
    }


def mapping_from_groups(groups: Iterable[tuple[str, Sequence[str]]]) -> tuple[dict[str, str], dict[str, list[str]]]:
    predictions: dict[str, str] = {}
    memberships: dict[str, list[str]] = {}
    for cluster_id, member_ids in groups:
        members = list(member_ids)
        if cluster_id in memberships or any(mid in predictions for mid in members):
            raise EvaluationError('Повторяющийся кластер или идентификатор публикации')
        memberships[cluster_id] = members
        for mid in members:
            predictions[mid] = cluster_id
    return predictions, memberships


def choose_threshold(rows: Sequence[dict[str, Any]], labels: Mapping[str, Label],
                     *, min_precision: float = 0.90) -> dict[str, Any]:
    """Choose using dev only; each row: high, low, clustering result + metrics.

    Prioritize eligibility (precision>=min_precision), then pairwise F1,
    then missed-match review capture, then fewer review pairs (efficiency).
    """
    if not (0 <= min_precision <= 1):
        raise EvaluationError('min_precision должен быть от 0 до 1')
    if not rows:
        raise EvaluationError('Нет порогов для подбора')
    candidates = []
    for row in rows:
        p = row['dev_clusters']['pairwise']['precision']
        f = row['dev_clusters']['pairwise']['f1']
        if p is None or f is None:
            raise EvaluationError('Нельзя калибровать без положительных и отрицательных пар в dev')
        rec = row['dev_review']['missed_positive_pair_review_recall']
        candidates.append((p >= min_precision, f, rec if rec is not None else 1.0,
                           -row['dev_review']['total_review_pairs'], row['low'], row['high'], row))
    eligible = [c for c in candidates if c[0]]
    if not eligible:
        return {'selected': None, 'reason': 'no_threshold_meets_min_precision',
                'best_diagnostic': max(candidates, key=lambda x: x[:-1])[-1]}
    best = max(eligible, key=lambda x: x[:-1])[-1]
    return {'selected': {'high': best['high'], 'low': best['low']}, 'reason': 'dev_only',
            'dev_metrics': best['dev_clusters'], 'dev_review': best['dev_review']}


def require_dev_test(labels: Mapping[str, Label]) -> None:
    """Refuse to present calibrated test quality on an uninformative split."""
    for split in ('dev', 'test'):
        ids = [mid for mid, x in labels.items() if x.split == split]
        positives = negatives = 0
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if labels[a].story_id == labels[b].story_id:
                    positives += 1
                else:
                    negatives += 1
        if not positives or not negatives:
            raise EvaluationError(f'{split}: необходимы и совпадающие, и разные пары размеченных сюжетов')


def late_repost_audit(rows: Sequence[dict[str, Any]], labels: Mapping[str, Label],
                      registry: Any, pub_weights: Any, story_weights: Any,
                      *, authority_min: int = 75) -> dict[str, Any]:
    """Observed within hand-labeled stories: compare score before/after a later stronger outlet.

    Diagnostic, not a model accuracy score or a causal measurement of readership.
    Does not silently include unjudged publications.
    """
    from .preprocessing import brand
    from .scoring import card
    from .story_scoring import score_story, _source_id
    from .validation import timestamp
    by_id = {r['message_id']: r for r in rows}
    by_story: dict[str, list[str]] = {}
    for mid, lab in labels.items():
        by_story.setdefault(lab.story_id, []).append(mid)
    tested = promoted = changed_representative = capped = 0
    examples: list[dict[str, Any]] = []
    for story, mids in sorted(by_story.items()):
        ordered = sorted(mids, key=lambda m: (timestamp(by_id[m]['published_at']), m))
        if len(ordered) < 2:
            continue
        cards = [card(by_id[mid], brand(by_id[mid]['text']), scoring='publication-v2',
                      source_registry=registry, scoring_weights=pub_weights) for mid in ordered]
        for index in range(1, len(cards)):
            later = cards[index]
            earlier = cards[:index]
            earlier_auth = max(c['source_authority_score'] for c in earlier)
            old_outlets = {_source_id(c)[0] for c in earlier}
            if (later['source_authority_score'] < authority_min or
                    later['source_authority_score'] <= earlier_auth or
                    _source_id(later)[0] in old_outlets):
                continue
            before = score_story(earlier, by_id, story_weights)
            after = score_story(cards[:index + 1], by_id, story_weights)
            tested += 1
            promoted += after.score > before.score
            changed_representative += after.representative_id == later['message_id']
            capped += after.score >= 100
            if len(examples) < 12:
                examples.append({'story_id': story, 'later_message_id': later['message_id'],
                                 'score_before': before.score, 'score_after': after.score,
                                 'delta': after.score - before.score,
                                 'main_source_changed_to_later': after.representative_id == later['message_id']})
            # Count first qualifying later strong publication per story only.
            break
    return {'eligible_labeled_stories': tested, 'score_increased': promoted,
            'score_increase_rate': promoted / tested if tested else None,
            'main_source_changed_to_later': changed_representative,
            'score_capped_at_100': capped,
            'examples': examples,
            'notes': 'Только полностью/частично размеченные события; тестируется появление более авторитетного независимого источника, а не доказанный дополнительный уникальный охват.'}


def cluster_error_examples(predictions: Mapping[str, str], labels: Mapping[str, Label],
                           split: str, *, limit: int = 20) -> dict[str, list[dict[str, str]]]:
    """Provide IDs of hard examples for editorial review, without leaking raw text."""
    gold = _split_labels(labels, split)
    ids = sorted(gold)
    examples: dict[str, list[dict[str, str]]] = {'false_merge': [], 'missed_merge': []}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            actual = gold[a].story_id == gold[b].story_id
            predicted = predictions[a] == predictions[b]
            if actual == predicted:
                continue
            key = 'false_merge' if predicted else 'missed_merge'
            if len(examples[key]) < limit:
                examples[key].append({'message_id_a': a, 'message_id_b': b,
                                       'gold_story_a': gold[a].story_id, 'gold_story_b': gold[b].story_id})
    return examples
