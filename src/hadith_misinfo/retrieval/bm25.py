"""BM25 retrieval baseline (rank_bm25).

Operates on Arabic + English text (or Arabic-only / English-only)
using the normaliser in preprocessing/normalize.py.

BM25 is used to establish whether simple lexical matching is sufficient
for Hadith verification, before moving to the dense cross-lingual models.
It is also a useful ablation: S3-BM25 vs S3-Dense and S4-BM25 vs S4-Dense.

Backend: rank_bm25 (BM25Okapi).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from hadith_misinfo.preprocessing.normalize import normalize_and_tokenize
from hadith_misinfo.schemas import EvidenceRecord, RetrievalResult


class BM25Retriever:
    """BM25 retriever over the Hadith evidence corpus.

    Usage
    -----
    >>> retriever = BM25Retriever.build(evidence_records)
    >>> results = retriever.search("Prophet said ...", k=5)
    """

    def __init__(self) -> None:
        self.evidence_ids: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._mode: str = "arabic_plus_english"

    # ── Build ─────────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        evidence_records: list[EvidenceRecord],
        mode: str = "arabic_plus_english",
        verbose: bool = True,
    ) -> "BM25Retriever":
        """Index a list of EvidenceRecords."""
        retriever = cls()
        retriever._mode = mode
        retriever.evidence_ids = [r.evidence_id for r in evidence_records]

        if verbose:
            from tqdm import tqdm
            records_iter = tqdm(evidence_records, desc="BM25 indexing")
        else:
            records_iter = evidence_records

        corpus_tokens = [
            normalize_and_tokenize(r.retrieval_text(mode))  # type: ignore[arg-type]
            for r in records_iter
        ]
        retriever._bm25 = BM25Okapi(corpus_tokens)
        return retriever

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Return top-k (evidence_id, score) pairs for a query string."""
        if self._bm25 is None:
            raise RuntimeError("Call .build() or .load() before .search()")
        query_tokens = normalize_and_tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            zip(self.evidence_ids, scores), key=lambda x: x[1], reverse=True
        )
        return ranked[:k]

    def search_to_results(
        self, claim_id: str, query: str, k: int = 5
    ) -> list[RetrievalResult]:
        """Like search(), but returns RetrievalResult objects."""
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

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "bm25.pkl").open("wb") as f:
            pickle.dump(self._bm25, f)
        with (out_dir / "evidence_ids.json").open("w", encoding="utf-8") as f:
            json.dump(self.evidence_ids, f)
        with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump({"mode": self._mode}, f)
        print(f"BM25 index saved to {out_dir}.")

    @classmethod
    def load(cls, in_dir: str | Path) -> "BM25Retriever":
        in_dir = Path(in_dir)
        retriever = cls()
        with (in_dir / "bm25.pkl").open("rb") as f:
            retriever._bm25 = pickle.load(f)
        with (in_dir / "evidence_ids.json").open("r", encoding="utf-8") as f:
            retriever.evidence_ids = json.load(f)
        meta_path = in_dir / "meta.json"
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                retriever._mode = json.load(f).get("mode", "arabic_plus_english")
        return retriever
