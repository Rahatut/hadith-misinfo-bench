"""Grounding audit and hallucination analysis for RAG responses."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

GroundingCategory = Literal["Grounded", "Partially Grounded", "Ungrounded"]


@dataclass
class GroundingAuditItem:
    claim_id: str
    verdict: str
    retrieved_evidence_ids: list[str]
    explanation: str
    category: GroundingCategory
    citation_valid: bool
    notes: str = ""


@dataclass
class GroundingSummary:
    total_audited: int
    grounded_count: int
    partially_grounded_count: int
    ungrounded_count: int
    grounding_rate: float
    partial_grounding_rate: float
    ungrounded_rate: float
    citation_validity_rate: float


def evaluate_grounding_audit(items: list[GroundingAuditItem]) -> GroundingSummary:
    """Compute summary statistics for a manual grounding audit (Section 17 of design doc)."""
    n = len(items)
    if n == 0:
        return GroundingSummary(
            total_audited=0,
            grounded_count=0,
            partially_grounded_count=0,
            ungrounded_count=0,
            grounding_rate=0.0,
            partial_grounding_rate=0.0,
            ungrounded_rate=0.0,
            citation_validity_rate=0.0,
        )

    counts = Counter(item.category for item in items)
    valid_citations = sum(1 for item in items if item.citation_valid)

    return GroundingSummary(
        total_audited=n,
        grounded_count=counts.get("Grounded", 0),
        partially_grounded_count=counts.get("Partially Grounded", 0),
        ungrounded_count=counts.get("Ungrounded", 0),
        grounding_rate=counts.get("Grounded", 0) / n,
        partial_grounding_rate=counts.get("Partially Grounded", 0) / n,
        ungrounded_rate=counts.get("Ungrounded", 0) / n,
        citation_validity_rate=valid_citations / n,
    )
