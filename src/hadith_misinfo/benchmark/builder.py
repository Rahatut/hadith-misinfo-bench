"""End-to-end benchmark construction pipeline."""

from __future__ import annotations

from pathlib import Path

from hadith_misinfo.benchmark.paraphraser import make_llm_paraphraser, paraphrase_pair
from hadith_misinfo.benchmark.sampler import (
    attach_gold_evidence,
    sample_balanced,
    save_benchmark,
    to_benchmark_record,
)
from hadith_misinfo.benchmark.validator import validate_benchmark
from hadith_misinfo.ingestion.mahaddat import iter_records
from hadith_misinfo.schemas import BenchmarkRecord


def build_benchmark_dataset(
    mahaddat_dir: str | Path,
    out_path: str | Path,
    n_authentic: int = 250,
    n_fabricated: int = 250,
    seed: int = 42,
    paraphrase: bool = True,
    complete_fn=None,
    retriever=None,
) -> list[BenchmarkRecord]:
    """Execute the full benchmark construction flow."""
    raw_records = list(iter_records(mahaddat_dir, split="test"))
    sampled = sample_balanced(raw_records, n_authentic, n_fabricated, seed)

    if retriever is not None:
        sampled = attach_gold_evidence(sampled, evidence_store=None, retriever=retriever)

    records: list[BenchmarkRecord] = []

    if paraphrase and complete_fn is not None:
        paraphraser = make_llm_paraphraser(complete_fn)
        for i, r in enumerate(sampled):
            claim_en, claim_bn = paraphrase_pair(r.arabic_matn, paraphraser)
            records.append(
                to_benchmark_record(
                    raw=r,
                    claim_id=f"C{str(i + 1).zfill(4)}",
                    claim_en=claim_en,
                    claim_bn=claim_bn,
                )
            )
    else:
        for i, r in enumerate(sampled):
            records.append(
                to_benchmark_record(
                    raw=r,
                    claim_id=f"C{str(i + 1).zfill(4)}",
                    claim_en=r.arabic_matn,
                    claim_bn=r.arabic_matn,
                )
            )

    report = validate_benchmark(records)
    if not report.is_valid:
        raise ValueError(f"Benchmark validation failed: {report.errors}")

    save_benchmark(records, out_path)
    return records
