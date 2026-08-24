"""Benchmark builder — samples MAHADDAT test records and generates paired claims.

Two steps:
1. Sample balanced test records (250 authentic + 250 fabricated) from the
   MAHADDAT test split.
2. Generate English and Bangla paraphrased claims from each Arabic matn.

The critical design constraints (§4–§5 of design doc):
  • Paired structure: C001_EN and C001_BN represent the same Hadith.
  • Direct Arabic → English paraphrase (not Arabic → English → Bangla).
  • Direct Arabic → Bangla paraphrase.
  • Anti-leakage: only claim_id + language + claim_text ever reach the LLM
    at inference time (enforced by InferenceRecord / to_inference_record).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

from hadith_misinfo.ingestion.mahaddat import RawRecord
from hadith_misinfo.schemas import BenchmarkRecord

if TYPE_CHECKING:
    from hadith_misinfo.evidence.store import EvidenceStore


def sample_balanced(
    records: list[RawRecord],
    n_authentic: int = 250,
    n_fabricated: int = 250,
    seed: int = 42,
) -> list[RawRecord]:
    """Sample a balanced subset of authentic and fabricated records.

    Raises ValueError if there are not enough records of either class.
    """
    authentic = [r for r in records if r.label == "authentic"]
    fabricated = [r for r in records if r.label == "fabricated"]

    if len(authentic) < n_authentic:
        raise ValueError(
            f"Not enough authentic records: have {len(authentic)}, need {n_authentic}."
        )
    if len(fabricated) < n_fabricated:
        raise ValueError(
            f"Not enough fabricated records: have {len(fabricated)}, need {n_fabricated}."
        )

    rng = random.Random(seed)
    selected = rng.sample(authentic, n_authentic) + rng.sample(fabricated, n_fabricated)
    rng.shuffle(selected)
    return selected


def attach_gold_evidence(
    records: list[RawRecord],
    evidence_store: "EvidenceStore",
    top_k: int = 5,
    retriever=None,
) -> list[RawRecord]:
    """Populate gold_evidence_ids on authentic records.

    Strategy:
    - If a retriever is provided, run BM25 search on the Arabic matn and
      take the top-k as candidate gold evidence IDs.
    - These are approximate matches — a grounding audit should manually
      verify the top result for authentic claims.

    Fabricated records remain with empty gold_evidence_ids.
    """
    for record in records:
        if record.label != "authentic":
            continue
        if retriever is not None:
            hits = retriever.search(record.arabic_matn, k=top_k)
            record.gold_evidence_ids = [eid for eid, _score in hits]
    return records


def to_benchmark_record(
    raw: RawRecord,
    claim_id: str,
    claim_en: str,
    claim_bn: str,
    canonical_en: str = "",
) -> BenchmarkRecord:
    """Assemble a BenchmarkRecord from a sampled RawRecord + generated claims."""
    return BenchmarkRecord(
        claim_id=claim_id,
        source_id=raw.source_id,
        label=raw.label,
        claims={"en": claim_en, "bn": claim_bn},
        canonical={"ar": raw.arabic_matn, "en": canonical_en},
        gold_evidence_ids=raw.gold_evidence_ids,
    )


def save_benchmark(records: list[BenchmarkRecord], out_path: str | Path) -> None:
    """Save benchmark records to a JSONL file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} benchmark records to {out_path}.")


def load_benchmark(path: str | Path) -> list[BenchmarkRecord]:
    """Load benchmark records from a JSONL file."""
    path = Path(path)
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(BenchmarkRecord.from_dict(json.loads(line)))
    return records
