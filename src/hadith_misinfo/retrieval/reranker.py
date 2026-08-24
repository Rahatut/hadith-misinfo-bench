"""Cross-encoder reranker for dense/hybrid retrieval outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hadith_misinfo.schemas import EvidenceRecord, RetrievalResult

if TYPE_CHECKING:
    from hadith_misinfo.evidence.store import EvidenceStore


class CrossEncoderReranker:
    """Reranks candidate evidence records using a cross-encoder model (e.g. BAAI/bge-reranker-v2-m3)."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for reranking.\n"
                    "Run: pip install 'hadith-misinfo-bench[dense]'"
                ) from e
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[EvidenceRecord],
        top_k: int = 5,
        mode: str = "arabic_plus_english",
    ) -> list[tuple[EvidenceRecord, float]]:
        """Rerank candidate evidence records based on cross-encoder similarity score."""
        if not candidates:
            return []

        model = self._get_model()
        pairs = [[query, c.retrieval_text(mode)] for c in candidates]  # type: ignore[arg-type]
        scores = model.predict(pairs)

        ranked = sorted(
            zip(candidates, [float(s) for s in scores]),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]
