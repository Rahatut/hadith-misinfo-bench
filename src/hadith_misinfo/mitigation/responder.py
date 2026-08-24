"""Layer 3: Evidence-grounded misinformation mitigation and structured response generation."""

from __future__ import annotations

from typing import Any

from hadith_misinfo.mitigation.prompts import (
    RECOMMENDATION_TEMPLATES,
    VERDICT_DISPLAY_LABELS,
)
from hadith_misinfo.schemas import EvidenceRecord, MitigationResult, VerdictLabel


class MitigationResponder:
    """Generates structured JSON interventions and human-readable responses."""

    def build_structured_response(
        self,
        claim_text: str,
        verdict: VerdictLabel,
        explanation: str,
        retrieved_evidence: list[EvidenceRecord],
        confidence: float = 0.85,
    ) -> dict[str, Any]:
        """Generate structured response dictionary for downstream platforms / UI."""
        abstained = (verdict == "INSUFFICIENT_EVIDENCE")

        evidence_items = [
            {
                "evidence_id": ev.evidence_id,
                "collection": ev.collection,
                "reference": ev.reference,
                "arabic_matn": ev.arabic_matn[:200],
                "english_text": (ev.english_text or "")[:200],
            }
            for ev in retrieved_evidence
        ]

        recommended_action = RECOMMENDATION_TEMPLATES.get(
            verdict,
            RECOMMENDATION_TEMPLATES["INSUFFICIENT_EVIDENCE"],
        )

        return {
            "claim": claim_text,
            "verdict": verdict,
            "confidence": confidence if not abstained else 0.4,
            "evidence": evidence_items,
            "explanation": explanation,
            "recommended_action": recommended_action,
            "abstained": abstained,
        }

    def build_intervention(
        self,
        claim_id: str,
        verdict: VerdictLabel,
        explanation: str,
        retrieved_evidence: list[EvidenceRecord],
        claim_text: str = "",
        language: str = "en",
    ) -> MitigationResult:
        """Format human-readable user-facing intervention result."""
        lines: list[str] = [
            "━" * 60,
            "HADITH VERIFICATION RESULT",
            "━" * 60,
        ]

        if claim_text:
            lines.append(f"\nClaim:\n  {claim_text}")

        label = VERDICT_DISPLAY_LABELS.get(verdict, verdict)
        lines.append(f"\nVerdict: {label}")

        if explanation:
            lines.append(f"\nAnalysis:\n  {explanation}")

        if retrieved_evidence:
            lines.append("\nRetrieved canonical evidence:")
            for i, ev in enumerate(retrieved_evidence, 1):
                ref = f"{ev.collection} — {ev.reference}"
                snippet = (ev.english_text or ev.arabic_matn or "")[:200]
                if len(ev.english_text or ev.arabic_matn or "") > 200:
                    snippet += "..."
                lines.append(f"  [{i}] {ref}")
                lines.append(f"      {snippet}")
        else:
            lines.append("\nRetrieved evidence: (none — no matching Hadith found in indexed corpus)")

        rec = RECOMMENDATION_TEMPLATES.get(
            verdict,
            RECOMMENDATION_TEMPLATES["INSUFFICIENT_EVIDENCE"],
        )
        lines.append(f"\n📋 Recommendation:\n  {rec}")

        lines.extend([
            "\n" + "━" * 60,
            "Note: This assessment is computational, not a scholarly ruling.\n"
            "For definitive Islamic guidance, consult a qualified scholar.",
            "━" * 60,
        ])

        return MitigationResult(
            claim_id=claim_id,
            verdict=verdict,
            intervention_text="\n".join(lines),
            retrieved_evidence=retrieved_evidence,
            grounded=None,
        )


# Module-level convenience functions
_default_responder = MitigationResponder()


def build_intervention(
    claim_id: str,
    verdict: VerdictLabel,
    explanation: str,
    retrieved_evidence: list[EvidenceRecord],
    claim_text: str = "",
    language: str = "en",
) -> MitigationResult:
    return _default_responder.build_intervention(
        claim_id=claim_id,
        verdict=verdict,
        explanation=explanation,
        retrieved_evidence=retrieved_evidence,
        claim_text=claim_text,
        language=language,
    )


def grounding_audit_report(results: list[MitigationResult]) -> dict:
    """Summarise grounding audit results across audited claims."""
    audited = [r for r in results if r.grounded is not None]
    if not audited:
        return {"audited": 0, "note": "No grounding labels set yet."}

    from collections import Counter
    counts = Counter(r.grounded for r in audited)
    total = len(audited)
    return {
        "audited": total,
        "grounded": counts.get(True, 0),
        "not_grounded": counts.get(False, 0),
        "grounding_rate": counts.get(True, 0) / total,
    }
