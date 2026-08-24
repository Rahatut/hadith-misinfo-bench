"""Verification policies, verdict-to-prediction mapping, and calibration rules."""

from __future__ import annotations

from hadith_misinfo.schemas import PredictionLabel, VerdictLabel


def map_verdict_to_prediction(
    verdict: str,
    contradicts_claim: bool = False,
) -> PredictionLabel:
    """Map the three-class verdict to the binary evaluation label + abstain.

    Label Mapping Policy:
        SUPPORTED             -> authentic
        INSUFFICIENT_EVIDENCE -> abstain
        NOT_SUPPORTED         -> fabricated (ONLY if contradicts_claim=True)
                              -> abstain (if contradicts_claim=False)
    """
    if verdict == "SUPPORTED":
        return "authentic"
    if verdict == "INSUFFICIENT_EVIDENCE":
        return "abstain"
    if verdict == "NOT_SUPPORTED":
        return "fabricated" if contradicts_claim else "abstain"
    return "abstain"


def should_abstain(
    verdict: VerdictLabel,
    confidence: float | None = None,
    min_confidence: float = 0.5,
) -> bool:
    """Determine whether the verification engine should abstain."""
    if verdict == "INSUFFICIENT_EVIDENCE":
        return True
    if confidence is not None and confidence < min_confidence:
        return True
    return False
