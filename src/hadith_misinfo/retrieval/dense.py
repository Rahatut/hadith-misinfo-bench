"""Dense multilingual retrieval using BAAI/bge-m3.

BGE-M3 is chosen as the primary model (literature review §7) because:
  • 100+ languages including Arabic and Bangla
  • Unifies dense, sparse, and multi-vector retrieval in one model
  • Strong on MIRACL/MKQA cross-lingual benchmarks
  • Handles morphologically rich languages like Arabic well

Secondary baseline: intfloat/multilingual-e5-large (swap via model_name param).

Backend: plain normalised-embedding dot-product (cosine) for the corpus sizes
here (tens of thousands of Hadith).  FAISS can be plugged in later without
changing the API — just replace _cosine_search() with a FAISS index search.

For S3 (English claim → English/Arabic evidence):
    query = English claim text
    corpus = arabic_plus_english

For S4 (Bangla claim → English/Arabic evidence):
    query = Bangla claim text
    corpus = arabic_plus_english   ← BGE-M3 handles cross-lingual alignment
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hadith_misinfo.schemas import EvidenceRecord, RetrievalResult


class DenseRetriever:
    """Dense multilingual retriever over the Hadith evidence corpus.

    Usage
    -----
    >>> retriever = DenseRetriever()
    >>> retriever.index(evidence_records)
    >>> results = retriever.search("রাসূল (সাঃ) বলেছেন ...", k=5)
    """

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        self.model_name = model_name
        self._model = None          # lazy-loaded (sentence-transformers is heavy)
        self.evidence_ids: list[str] = []
        self.embeddings: np.ndarray | None = None   # (n_docs, dim), L2-normalised
        self._mode: str = "arabic_plus_english"

    # ── Model lazy-loading ────────────────────────────────────────────────────

    def _get_model(self):
        if self._model is None:
            try:
                import os
                import torch
                torch.set_num_threads(os.cpu_count() or 8)
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is not installed.\n"
                    "Run: pip install 'hadith-misinfo-bench[dense]'"
                ) from e
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        model = self._get_model()
        emb = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
        )
        return np.asarray(emb, dtype=np.float32)

    # ── Index ─────────────────────────────────────────────────────────────────

    def index(
        self,
        evidence_records: list[EvidenceRecord],
        mode: str = "arabic_plus_english",
        batch_size: int = 64,
        chunk_size: int = 250,
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        """Encode and store all evidence records with chunked checkpointing."""
        self._mode = mode
        self.evidence_ids = [r.evidence_id for r in evidence_records]
        texts = [r.retrieval_text(mode) for r in evidence_records]  # type: ignore[arg-type]
        n_total = len(texts)
        print(f"Encoding {n_total:,} evidence records with {self.model_name}...")

        chk_path = Path(checkpoint_dir) if checkpoint_dir else None
        if chk_path:
            chk_path.mkdir(parents=True, exist_ok=True)
            chk_file = chk_path / "embeddings_partial.npy"
            if chk_file.exists():
                existing = np.load(chk_file)
                if len(existing) == n_total:
                    print(f"Loaded complete embeddings from checkpoint ({len(existing):,} records).")
                    self.embeddings = existing
                    return
                print(f"Resuming dense indexing from {len(existing):,}/{n_total:,} records...")
            else:
                existing = np.empty((0, 1024), dtype=np.float32)
        else:
            existing = np.empty((0, 1024), dtype=np.float32)

        start_idx = len(existing)
        all_chunks = [existing] if len(existing) > 0 else []

        for i in range(start_idx, n_total, chunk_size):
            chunk_texts = texts[i : i + chunk_size]
            print(f"  Encoding chunk {i+1} to {min(i + chunk_size, n_total)} / {n_total}...")
            chunk_emb = self._encode(chunk_texts, batch_size=batch_size)
            all_chunks.append(chunk_emb)
            if chk_path:
                current = np.vstack(all_chunks)
                np.save(chk_file, current)

        self.embeddings = np.vstack(all_chunks)
        print(f"Done. Embedding matrix: {self.embeddings.shape}")

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Return top-k (evidence_id, cosine_score) pairs for a query string."""
        if self.embeddings is None:
            raise RuntimeError("Call .index() or .load() before .search()")
        q_emb = self._encode([query])[0]            # (dim,)
        scores = self.embeddings @ q_emb            # (n_docs,) — cosine similarity
        top_idx = np.argsort(-scores)[:k]
        return [(self.evidence_ids[i], float(scores[i])) for i in top_idx]

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
        np.save(out_dir / "embeddings.npy", self.embeddings)
        with (out_dir / "evidence_ids.json").open("w", encoding="utf-8") as f:
            json.dump(self.evidence_ids, f)
        with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump({"model_name": self.model_name, "mode": self._mode}, f)
        print(f"Dense index saved to {out_dir} ({len(self.evidence_ids):,} vectors).")

    @classmethod
    def load(cls, in_dir: str | Path) -> "DenseRetriever":
        in_dir = Path(in_dir)
        with (in_dir / "meta.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)
        retriever = cls(model_name=meta["model_name"])
        retriever._mode = meta.get("mode", "arabic_plus_english")
        retriever.embeddings = np.load(in_dir / "embeddings.npy")
        with (in_dir / "evidence_ids.json").open("r", encoding="utf-8") as f:
            retriever.evidence_ids = json.load(f)
        print(f"Dense index loaded from {in_dir} ({len(retriever.evidence_ids):,} vectors).")
        return retriever
