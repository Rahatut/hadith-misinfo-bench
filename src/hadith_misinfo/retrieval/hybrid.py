"""Reciprocal Rank Fusion (RRF) hybrid retriever.

Fuses BM25 (lexical) and Dense (semantic) ranked lists into a single
re-ranked list.  RRF is the standard fusion method used in hybrid RAG
systems (Cormack, Clarke & Buettcher 2009).

Formula: RRF(d) = Σ_r 1 / (k + rank_r(d))
where k=60 is the standard smoothing constant.

Usage
-----
>>> from hadith_misinfo.retrieval.hybrid import rrf_fuse
>>> bm25_hits = bm25_retriever.search(query, k=20)
>>> dense_hits = dense_retriever.search(query, k=20)
>>> fused = rrf_fuse(bm25_hits, dense_hits, top_k=5)
"""

from __future__ import annotations

from hadith_misinfo.schemas import EvidenceRecord, RetrievalResult
from hadith_misinfo.retrieval.bm25 import BM25Retriever
from hadith_misinfo.retrieval.dense import DenseRetriever


def rrf_fuse(
    ranked_lists: list[list[tuple[str, float]]],
    top_k: int = 5,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Parameters
    ----------
    ranked_lists:
        Each element is a ranked list of (evidence_id, score) pairs,
        ordered best-first.  The raw score is ignored; only rank matters.
    top_k:
        Number of results to return.
    k:
        RRF smoothing constant (default 60, standard value).

    Returns
    -------
    list of (evidence_id, rrf_score) sorted by rrf_score descending.
    """
    rrf_scores: dict[str, float] = {}

    for ranked in ranked_lists:
        for rank, (eid, _score) in enumerate(ranked, start=1):
            rrf_scores[eid] = rrf_scores.get(eid, 0.0) + 1.0 / (k + rank)

    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]


class HybridRetriever:
    """Hybrid BM25 + Dense retriever using RRF fusion.

    Usage
    -----
    >>> hybrid = HybridRetriever(bm25=bm25_ret, dense=dense_ret)
    >>> results = hybrid.search_to_results(claim_id, query, k=5)
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        fetch_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.fetch_k = fetch_k
        self.rrf_k = rrf_k

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Return top-k fused (evidence_id, rrf_score) pairs."""
        bm25_hits = self.bm25.search(query, self.fetch_k)
        dense_hits = self.dense.search(query, self.fetch_k)
        return rrf_fuse([bm25_hits, dense_hits], top_k=k, k=self.rrf_k)

    def search_to_results(
        self, claim_id: str, query: str, k: int = 5
    ) -> list[RetrievalResult]:
        raw = self.search(query, k)
        return [
            RetrievalResult(
                claim_id=claim_id,
                evidence_id=eid,
                rank=i + 1,
                score=score,
            )
            for i, (eid, score) in enumerate(raw)
        ]
