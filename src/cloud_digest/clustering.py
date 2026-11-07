"""Baseline hash and opt-in semantic event clustering (stage 04).

Uses complete-link candidate acceptance to avoid transitive chain merging.
Only explicit fact conflicts are hard blockers. Scores are NOT changed here.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Any, Sequence

from .embeddings import EmbeddedPublication, cosine_similarity
from .event_facts import extract_event_facts, identity_conflicts
from .preprocessing import PreparedPublication, normalized


def story_key(item: dict[str, Any], text: str) -> str:
    """Unmodified baseline-v1 identifier, for reproducibility."""
    norm = normalized(text)
    if any(x in norm for x in ('авари', 'недоступ', 'сбой', 'отключ', 'упал', 'лежит')):
        kind = 'outage' if any(x in norm for x in ('недоступ', 'сбой', 'авари', 'отключ', 'упал', 'лежит')) else 'other'
        day = item['published_at'][:10]
        return sha256(f'incident:{day}:{kind}'.encode()).hexdigest()
    return sha256(norm.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class SemanticClusterConfig:
    high_threshold: float = 0.88
    low_threshold: float = 0.78
    max_gap_days: int = 7
    incident_gap_hours: int = 36
    max_review_pairs: int = 2000

    def __post_init__(self) -> None:
        for n in (self.low_threshold, self.high_threshold):
            if not isinstance(n, (int, float)) or isinstance(n, bool) or not math.isfinite(n) or not 0 <= n <= 1:
                raise ValueError('Пороги сходства должны быть числами в диапазоне [0, 1]')
        if self.low_threshold >= self.high_threshold:
            raise ValueError('low_threshold должен быть меньше high_threshold')
        if type(self.max_gap_days) is not int or self.max_gap_days < 1:
            raise ValueError('max_gap_days должен быть положительным целым')
        if type(self.incident_gap_hours) is not int or self.incident_gap_hours < 1:
            raise ValueError('incident_gap_hours должен быть положительным целым')
        if type(self.max_review_pairs) is not int or self.max_review_pairs < 0:
            raise ValueError('max_review_pairs должен быть неотрицательным целым')


@dataclass(frozen=True, slots=True)
class ReviewPair:
    message_id_a: str
    message_id_b: str
    similarity: float
    reason: str


@dataclass(frozen=True, slots=True)
class StoryCluster:
    cluster_id: str
    message_ids: tuple[str, ...]
    first_published_at: str


@dataclass(frozen=True, slots=True)
class SemanticClusteringResult:
    clusters: tuple[StoryCluster, ...]
    review_pairs: tuple[ReviewPair, ...]
    candidate_pairs: int
    comparison_pairs: int
    skipped_without_vector: int
    truncated_review_pairs: int


def cluster_semantically(
    publications: Sequence[PreparedPublication], embeddings: Sequence[EmbeddedPublication],
    config: SemanticClusterConfig | None = None,
) -> SemanticClusteringResult:
    """Conservative event clustering, deterministic regardless of input ordering.

    - similarity >= high and no fact conflict to *every* member => auto-merge;
    - low <= similarity < high => review candidate, no merge;
    - explicit fact contradictions block merging even at similarity 1.0;
    - no vector => merge exact clean texts only, subject to fact/time checks.
    """
    cfg = config or SemanticClusterConfig()
    if len(publications) != len(embeddings):
        raise ValueError('Число публикаций и эмбеддингов не совпадает')
    ids = [p.message_id for p in publications]
    if len(set(ids)) != len(ids) or any(p.message_id != e.message_id for p, e in zip(publications, embeddings)):
        raise ValueError('message_id в публикациях и эмбеддингах не совпадают или повторяются')
    dimensions = {len(e.vector) for e in embeddings if e.vector is not None}
    if len(dimensions) > 1:
        raise ValueError('Разная размерность эмбеддингов')
    entries = sorted(zip(publications, embeddings), key=lambda pe: (pe[0].published_at_utc, pe[0].message_id))
    fact_map = {p.message_id: extract_event_facts(p) for p, _ in entries}
    vector_map = {e.message_id: e.vector for _, e in entries}
    pub_map = {p.message_id: p for p, _ in entries}
    groups: list[list[str]] = []
    review: list[ReviewPair] = []
    comparisons = candidates = truncated = 0
    for publication, embedding in entries:
        ident = publication.message_id
        eligible: list[tuple[float, int]] = []
        for group_idx, members in enumerate(groups):
            similarities: list[float] = []
            conflicts: list[str] = []
            for other_id in members:
                other = pub_map[other_id]
                reasons = identity_conflicts(fact_map[ident], fact_map[other_id],
                                             max_gap_days=cfg.max_gap_days,
                                             incident_gap_hours=cfg.incident_gap_hours)
                left, right = embedding.vector, vector_map[other_id]
                # If no vector, only *identical* normalized content can be merged.
                if left is None or right is None:
                    similarity = 1.0 if normalized(publication.clean_text) and normalized(publication.clean_text) == normalized(other.clean_text) else 0.0
                else:
                    similarity = cosine_similarity(left, right)
                comparisons += 1
                if similarity >= cfg.low_threshold:
                    candidates += 1
                    if reasons:
                        conflicts.extend(reasons)
                        if len(review) < cfg.max_review_pairs:
                            review.append(ReviewPair(ident, other_id, round(similarity, 5), 'blocked:' + ','.join(reasons)))
                        else:
                            truncated += 1
                    elif similarity < cfg.high_threshold:
                        if len(review) < cfg.max_review_pairs:
                            review.append(ReviewPair(ident, other_id, round(similarity, 5), 'borderline_similarity'))
                        else:
                            truncated += 1
                similarities.append(similarity)
            # Complete-link rule avoids A~B~C chaining when A !~ C.
            if similarities and min(similarities) >= cfg.high_threshold and not conflicts:
                eligible.append((min(similarities), group_idx))
        if eligible:
            # Merge into single best-matching existing cluster. No cross-cluster chain unions.
            _, best = max(eligible, key=lambda x: (x[0], -x[1]))
            groups[best].append(ident)
        else:
            groups.append([ident])
    clusters = []
    for group in groups:
        anchor = pub_map[group[0]]
        cluster_id = sha256(('semantic-v1:' + anchor.message_id).encode()).hexdigest()
        clusters.append(StoryCluster(cluster_id, tuple(group), anchor.published_at_utc.isoformat()))
    return SemanticClusteringResult(tuple(clusters), tuple(review), candidates, comparisons,
                                    sum(e.vector is None for e in embeddings), truncated)
