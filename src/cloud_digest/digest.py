"""Группировка, отбор, ранжирование и представление дайджеста."""
from datetime import date, datetime
from typing import Any
from .config import MSK, EMPTY, SYNTHETIC
from .validation import InputError, timestamp
from .preprocessing import brand, prepare_publication, prepare_publications
from .clustering import SemanticClusterConfig, cluster_semantically
from .embeddings import EmbeddingService
from .scoring import card, PublicationScoreWeights
from .source_authority import SourceAuthorityRegistry
from .story_scoring import StoryScoreWeights, score_story
from .story_view import build_story_card, render_story_digest

def digest(rows: list[dict[str, Any]], day: date, top_n: int = 30, synthetic: bool = False,
           *, semantic: bool = False, embedding_service: EmbeddingService | None = None,
           clustering_config: SemanticClusterConfig | None = None,
           scoring: str = 'baseline', source_registry: SourceAuthorityRegistry | None = None,
           scoring_weights: PublicationScoreWeights | None = None,
           story_weights: StoryScoreWeights | None = None) -> dict[str, Any]:
    if not 1 <= top_n <= 30:
        raise InputError('top-n: требуется целое 1–30')
    if scoring not in ('baseline', 'publication-v2', 'story-v1'):
        raise InputError('scoring: требуется baseline, publication-v2 или story-v1')
    if scoring in ('publication-v2', 'story-v1') and source_registry is None:
        raise InputError('publication-v2/story-v1: нужен справочник top25_sources.json')
    daily = [r for r in rows if timestamp(r['published_at']).astimezone(MSK).date() == day]
    candidates = []
    dropped = []
    queue = []
    groups: dict[str, list[dict[str, Any]]] = {}
    original_by_id: dict[str, dict[str, Any]] = {}
    for row in daily:
        prepared = prepare_publication(row)
        # В baseline правила отбора и скоринга не меняем. Представление
        # prepared.clean_text / prepared.semantic_text уже готово для v2.
        link = brand(prepared.raw_text)
        if link is None:
            topic = bool(row['ba_topic_hit'] and row['ba_topic_hit'].strip())
            reason = 'topic_hit_without_brand' if topic else 'no_brand_match'
            dropped.append({'message_id': row['message_id'], 'reason': reason})
            if topic:
                queue.append({'message_id': row['message_id'], 'cluster_id': None, 'url': row['url'], 'reasons': [reason]})
            continue
        item = card(row, link, scoring='publication-v2' if scoring == 'story-v1' else scoring,
                    source_registry=source_registry, scoring_weights=scoring_weights)
        candidates.append(item)
        if scoring == 'story-v1':
            original_by_id[item['message_id']] = row
        if not semantic:
            groups.setdefault(item['cluster_id'], []).append(item)
        if item['human_review_needed']:
            queue.append({'message_id': item['message_id'], 'cluster_id': item['cluster_id'], 'url': item['url'], 'reasons': item['review_reasons']})
    semantic_meta: dict[str, Any] = {}
    if semantic:
        if embedding_service is None:
            embedding_service = EmbeddingService()
        prepared_candidates = prepare_publications([row for row in daily if brand(row['text']) is not None])
        embeddings = embedding_service.encode(prepared_candidates)
        clustering = cluster_semantically(prepared_candidates, embeddings, clustering_config)
        cards_by_id = {item['message_id']: item for item in candidates}
        for group in clustering.clusters:
            members = [cards_by_id[mid] for mid in group.message_ids]
            for item in members:
                item['cluster_id'] = group.cluster_id
            groups[group.cluster_id] = members
        for entry in queue:
            if entry['message_id'] in cards_by_id:
                entry['cluster_id'] = cards_by_id[entry['message_id']]['cluster_id']
        semantic_meta = {
            'semantic_clustering': {
                'high_threshold': (clustering_config or SemanticClusterConfig()).high_threshold,
                'low_threshold': (clustering_config or SemanticClusterConfig()).low_threshold,
                'candidate_pairs': clustering.candidate_pairs,
                'comparison_pairs': clustering.comparison_pairs,
                'skipped_without_vector': clustering.skipped_without_vector,
                'truncated_review_pairs': clustering.truncated_review_pairs,
                'review_pairs': [
                    {'message_id_a': p.message_id_a, 'message_id_b': p.message_id_b,
                     'similarity': p.similarity, 'reason': p.reason}
                    for p in clustering.review_pairs
                ],
                'clusters': [
                    {'cluster_id': g.cluster_id, 'message_ids': list(g.message_ids),
                     'first_published_at': g.first_published_at}
                    for g in clustering.clusters
                ],
            },
        }
    def chronological(item: dict[str, Any]) -> tuple[datetime, str]:
        return timestamp(item['published_at']), item['message_id']
    def priority(item: dict[str, Any]) -> tuple[int, datetime, str]:
        return -item['importance_score'], *chronological(item)
    representatives = []
    story_claim_reviews = 0
    story_identity_reviews = 0
    for cluster_id, group in groups.items():
        ordered = sorted(group, key=chronological)
        if scoring == 'story-v1':
            assessment = score_story(ordered, original_by_id, story_weights)
            representative = next(item for item in ordered if item['message_id'] == assessment.representative_id)
            representative['publication_importance_score'] = representative['importance_score']
            representative['importance_score'] = assessment.score
            representative['importance_explanation'] = assessment.metadata['story_explanation']
            representative.update(assessment.metadata)
            # Риск из любого сообщения кластера не должен теряться при
            # выборе более влиятельного позднего представителя.
            story_review_reasons = set(assessment.review_reasons)
            if 'conflicting_numeric_claims' in assessment.review_reasons:
                story_claim_reviews += 1
            if 'incompatible_event_members' in assessment.review_reasons:
                story_identity_reviews += 1
            if any('crisis_signal' in c['review_reasons'] for c in ordered):
                representative['recommended_action'] = 'escalate_pr'
                representative['sentiment'] = 'negative'
                representative['story_contains_crisis'] = True
                if 'crisis_signal' not in representative['review_reasons']:
                    story_review_reasons.add('story_contains_crisis')
            else:
                representative['story_contains_crisis'] = False
            if story_review_reasons:
                representative['human_review_needed'] = True
                representative['confidence'] = min(representative['confidence'], 0.4)
                representative['review_reasons'] = sorted(set(representative['review_reasons']) | story_review_reasons)
                entry = next((q for q in queue if q['message_id'] == representative['message_id']), None)
                if entry is None:
                    queue.append({'message_id': representative['message_id'], 'cluster_id': cluster_id,
                                  'url': representative['url'], 'reasons': sorted(story_review_reasons)})
                else:
                    entry['reasons'] = sorted(set(entry['reasons']) | story_review_reasons)
            # Структурированное представление строится после обновления риска
            # и причин редакторской проверки, но без повторного скоринга.
            representative['story_card'] = build_story_card(
                representative, original_by_id[representative['message_id']]
            )
        else:
            representative = ordered[0]
        representative['cluster_size'] = len(group)
        representatives.append(representative)
        dropped.extend({'message_id': item['message_id'], 'reason': 'duplicate'} for item in ordered
                       if item['message_id'] != representative['message_id'])
    ranked = sorted(representatives, key=priority)
    # Story Score ранжируется по общему влиянию сюжета, без отдельной
    # перестановки по тональности (она уже заложена в Publication Score).
    selected = (ranked[:top_n] if scoring == 'story-v1' else sorted(
        ranked[:top_n], key=lambda item: (('negative', 'neutral', 'positive').index(item['sentiment']), priority(item))))
    dropped.extend({'message_id': item['message_id'], 'reason': 'outside_top_n'} for item in ranked[top_n:])
    for rank, item in enumerate(selected, 1):
        item['rank_in_digest'] = rank
    risk_ids = sorted(item['message_id'] for item in candidates if 'crisis_signal' in item['review_reasons'])
    counts = dict(input_records=len(rows), day_records=len(daily), brand_candidates=len(candidates),
                  clusters=len(groups), duplicates_removed=len(candidates) - len(groups), selected_items=len(selected),
                  review_records=len(queue), excluded_no_brand=len(daily) - len(candidates), excluded_top_n=len(ranked[top_n:]))
    version = 'semantic-clustering-v1' if semantic else 'baseline-v1'
    if scoring == 'publication-v2':
        version = ('semantic-clustering-v1+' if semantic else '') + 'publication-scoring-v2'
    if scoring == 'story-v1':
        version = ('semantic-clustering-v1+' if semantic else '') + 'story-scoring-v1'
        semantic_meta['story_scoring'] = {
            'scored_clusters': len(groups), 'review_clusters_with_claim_conflicts': story_claim_reviews,
            'review_clusters_with_event_conflicts': story_identity_reviews,
            'audience_method': 'capped_proxy_not_unique_reach',
            'new_facts_method': 'explicit_numeric_and_service_claim_signals_not_verified',
        }
    return dict(digest_date=day.isoformat(), rule_version=version, summary_counts=counts, items=selected,
                dropped_examples=sorted(dropped, key=lambda item: item['message_id'])[:10], risk_alert=bool(risk_ids),
                risk_alert_details={'message_ids': risk_ids, 'reason': 'crisis_signal' if risk_ids else None},
                review_queue=sorted(queue, key=lambda item: item['message_id']), empty_message=EMPTY if not candidates else None,
                synthetic=synthetic, **semantic_meta)


def render(result: dict[str, Any]) -> str:
    if result['rule_version'].endswith('story-scoring-v1'):
        return render_story_digest(result)
    def display(value: Any) -> str:
        if value is None:
            return 'нет данных'
        if isinstance(value, dict):
            return '\n'.join(f'{key}: {display(item)}' for key, item in value.items())
        if isinstance(value, list):
            return '\n\n'.join(display(item) for item in value) or '[]'
        return str(value)
    heading = SYNTHETIC + '\n' if result['synthetic'] else ''
    if result['rule_version'].endswith('story-scoring-v1'):
        label = 'Экспериментальный Story Score: охват оценочный, факты не подтверждены извне.'
    else:
        label = 'Детерминированный baseline; косвенные связи полноценно не проверяются.'
    return heading + label + '\n' + display(result) + '\n'
