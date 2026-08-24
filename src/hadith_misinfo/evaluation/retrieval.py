"""Retrieval evaluation metrics (Recall@k, MRR) over authentic claims."""

from __future__ import annotations

from dataclasses import dataclass

from hadith_misinfo.schemas import RetrievalResult


@dataclass
class RetrievalMetrics:
    n_authentic: int
    recall_at_1: float
    recall_at_5: float
    recall_at_k: dict[int, float]
    mrr: float


def _group_by_claim(results: list[RetrievalResult]) -> dict[str, list[RetrievalResult]]:
    grouped: dict[str, list[RetrievalResult]] = {}
    for r in results:
        grouped.setdefault(r.claim_id, []).append(r)
    for claim_id in grouped:
        grouped[claim_id].sort(key=lambda r: r.rank)
    return grouped


def recall_at_k(
    results: list[RetrievalResult],
    gold_ids: dict[str, list[str]],
    k: int,
) -> float:
    """Recall@k: fraction of authentic claims where gold evidence is in top-k."""
    if not gold_ids:
        return 0.0
    grouped = _group_by_claim(results)
    hits = sum(
        1
        for claim_id, gold in gold_ids.items()
        if any(r.evidence_id in gold for r in grouped.get(claim_id, [])[:k])
    )
    return hits / len(gold_ids)


def mean_reciprocal_rank(
    results: list[RetrievalResult],
    gold_ids: dict[str, list[str]],
) -> float:
    """Mean Reciprocal Rank (MRR) across authentic claims."""
    if not gold_ids:
        return 0.0
    grouped = _group_by_claim(results)
    reciprocal_ranks = []
    for claim_id, gold in gold_ids.items():
        rr = 0.0
        for r in grouped.get(claim_id, []):
            if r.evidence_id in gold:
                rr = 1.0 / r.rank
                break
        reciprocal_ranks.append(rr)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def compute_retrieval_metrics(
    results: list[RetrievalResult],
    gold_ids: dict[str, list[str]],
    k_values: list[int] = (1, 5),
) -> RetrievalMetrics:
    """Compute all retrieval metrics at once."""
    recall_k = {k: recall_at_k(results, gold_ids, k) for k in k_values}
    return RetrievalMetrics(
        n_authentic=len(gold_ids),
        recall_at_1=recall_k.get(1, 0.0),
        recall_at_5=recall_k.get(5, 0.0),
        recall_at_k=recall_k,
        mrr=mean_reciprocal_rank(results, gold_ids),
    )
