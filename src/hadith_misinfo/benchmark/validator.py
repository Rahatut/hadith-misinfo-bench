"""Benchmark dataset integrity and anti-leakage validation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from hadith_misinfo.schemas import BenchmarkRecord


@dataclass
class ValidationReport:
    total_records: int
    label_distribution: dict[str, int]
    is_balanced: bool
    missing_claims: list[str]
    empty_claims: list[str]
    gold_evidence_counts: dict[str, int]
    is_valid: bool
    errors: list[str]


def validate_benchmark(records: list[BenchmarkRecord], target_size: int = 500) -> ValidationReport:
    """Validate integrity, balance, language completeness, and anti-leakage constraints of a benchmark."""
    errors: list[str] = []
    missing_claims: list[str] = []
    empty_claims: list[str] = []

    label_counts = Counter(r.label for r in records)
    is_balanced = (
        len(label_counts) == 2
        and label_counts.get("authentic", 0) == label_counts.get("fabricated", 0)
    )

    if not is_balanced:
        errors.append(
            f"Benchmark is not balanced: authentic={label_counts.get('authentic', 0)}, "
            f"fabricated={label_counts.get('fabricated', 0)}"
        )

    for r in records:
        for lang in ("en", "bn"):
            if lang not in r.claims:
                missing_claims.append(f"{r.claim_id}_{lang}")
            elif not r.claims[lang].strip():
                empty_claims.append(f"{r.claim_id}_{lang}")

        # Leakage check: fabricated records should never have gold evidence IDs
        if r.label == "fabricated" and r.gold_evidence_ids:
            errors.append(f"Fabricated record {r.claim_id} contains gold_evidence_ids ({r.gold_evidence_ids})")

    if missing_claims:
        errors.append(f"Missing language claims: {len(missing_claims)} instances")
    if empty_claims:
        errors.append(f"Empty language claims: {len(empty_claims)} instances")

    gold_counts = Counter(
        "has_gold" if r.gold_evidence_ids else "no_gold"
        for r in records
    )

    is_valid = len(errors) == 0

    return ValidationReport(
        total_records=len(records),
        label_distribution=dict(label_counts),
        is_balanced=is_balanced,
        missing_claims=missing_claims,
        empty_claims=empty_claims,
        gold_evidence_counts=dict(gold_counts),
        is_valid=is_valid,
        errors=errors,
    )
