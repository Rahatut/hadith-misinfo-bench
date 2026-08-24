"""Layer 2: Verification orchestrator.

The Verifier class coordinates retrieval (for S3/S4) and LLM verification
for all four experimental systems:

    S1  English  parametric  (no retrieval)
    S2  Bangla   parametric  (no retrieval)
    S3  English  RAG         (BM25 or Dense retrieval)
    S4  Bangla   cross-lingual RAG (Bangla query → Arabic/English evidence)

Usage
-----
>>> from hadith_misinfo.verification.verifier import Verifier
>>> verifier = Verifier(complete=complete_fn, evidence_store=store, retriever=dense_ret)
>>> result = verifier.verify(inference_record, system="S4", k=5)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from hadith_misinfo.evidence.store import EvidenceStore
from hadith_misinfo.schemas import (
    EvidenceRecord,
    InferenceRecord,
    Language,
    RetrievalResult,
    SystemId,
    VerificationResult,
)
from hadith_misinfo.verification.prompts import (
    build_no_rag_prompt,
    build_rag_prompt,
    parse_verdict,
)

CompleteFn = Callable[[str], str]

# Maps system ID to (language, uses_retrieval)
_SYSTEM_CONFIG: dict[str, tuple[Language, bool]] = {
    "S1": ("en", False),
    "S2": ("bn", False),
    "S3": ("en", True),
    "S4": ("bn", True),
}


class Verifier:
    """Orchestrates retrieval + LLM verification for one claim."""

    def __init__(
        self,
        complete: CompleteFn,
        evidence_store: EvidenceStore | None = None,
        retriever=None,                     # BM25Retriever | DenseRetriever | HybridRetriever
        retrieval_text_mode: str = "arabic_plus_english",
        default_k: int = 5,
        query_translator: Callable[[str], str] | None = None,
    ) -> None:
        self.complete = complete
        self.evidence_store = evidence_store
        self.retriever = retriever
        self.retrieval_text_mode = retrieval_text_mode
        self.default_k = default_k
        self.query_translator = query_translator

    def verify(
        self,
        record: InferenceRecord,
        system: SystemId,
        k: int | None = None,
    ) -> VerificationResult:
        """Verify one claim and return a VerificationResult.

        Parameters
        ----------
        record: InferenceRecord — the anti-leakage inference record.
        system: "S1" | "S2" | "S3" | "S4"
        k:      Number of evidence documents to retrieve (S3/S4 only).
        """
        _language, uses_retrieval = _SYSTEM_CONFIG[system]
        k = k or self.default_k

        retrieved_evidence: list[EvidenceRecord] = []
        retrieved_evidence_ids: list[str] = []

        if uses_retrieval:
            if self.retriever is None:
                raise RuntimeError(
                    f"System {system} requires a retriever, but none was provided."
                )
            if self.evidence_store is None:
                raise RuntimeError(
                    f"System {system} requires an evidence_store, but none was provided."
                )
            search_query = record.claim_text
            if self.query_translator is not None and record.language == "bn":
                search_query = self.query_translator(record.claim_text)

            raw_hits = self.retriever.search(search_query, k=k)
            retrieved_evidence_ids = [eid for eid, _score in raw_hits]
            retrieved_evidence = self.evidence_store.get_many(retrieved_evidence_ids)
            prompt = build_rag_prompt(record, retrieved_evidence, self.retrieval_text_mode)
        else:
            prompt = build_no_rag_prompt(record)

        raw_response = self.complete(prompt)

        try:
            verdict, contradicts, explanation = parse_verdict(raw_response)
        except ValueError:
            # Malformed output — abstain rather than crash the run
            verdict = "INSUFFICIENT_EVIDENCE"
            contradicts = False
            explanation = "Could not parse model output."

        return VerificationResult(
            claim_id=record.claim_id,
            system=system,
            language=record.language,
            verdict=verdict,  # type: ignore[arg-type]
            explanation=explanation,
            retrieved_evidence_ids=retrieved_evidence_ids,
            contradicts_claim=contradicts,
            raw_response=raw_response,
        )


# ── Batch runner ─────────────────────────────────────────────────────────────


def run_batch(
    verifier: Verifier,
    records: list[InferenceRecord],
    system: SystemId,
    k: int = 5,
    out_path: str | Path | None = None,
    verbose: bool = True,
    concurrency: int = 1,
) -> list[VerificationResult]:
    """Run verification for a list of InferenceRecords.

    Optionally streams results to a JSONL file as they are produced
    (safe to resume if the run is interrupted — deduplicate by claim_id
    before evaluation).
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    results: list[VerificationResult] = []
    existing_results: dict[str, VerificationResult] = {}
    if out_path and Path(out_path).exists():
        for r in load_results(out_path):
            existing_results[r.claim_id] = r
        if existing_results:
            print(f"    Resuming from {len(existing_results)} already completed records in {out_path}...")

    out_f = None
    lock = threading.Lock()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = out_path.open("a", encoding="utf-8")

    def _verify_one(record: InferenceRecord) -> VerificationResult:
        if record.claim_id in existing_results:
            return existing_results[record.claim_id]
        res = verifier.verify(record, system=system, k=k)
        if out_f:
            with lock:
                out_f.write(json.dumps(res.to_dict(), ensure_ascii=False) + "\n")
                out_f.flush()
        return res

    if concurrency > 1:
        if verbose:
            from tqdm import tqdm
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                results = list(
                    tqdm(
                        pool.map(_verify_one, records),
                        total=len(records),
                        desc=f"Verifying [{system}] (x{concurrency})",
                    )
                )
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                results = list(pool.map(_verify_one, records))
    else:
        iterator = records
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(records, desc=f"Verifying [{system}]")

        for record in iterator:
            was_cached = record.claim_id in existing_results
            result = _verify_one(record)
            results.append(result)
            if not was_cached:
                time.sleep(0.3)

    if out_f:
        out_f.close()

    return results


def load_results(path: str | Path) -> list[VerificationResult]:
    """Load VerificationResult objects from a JSONL file."""
    results = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(VerificationResult.from_dict(json.loads(line)))
    return results
