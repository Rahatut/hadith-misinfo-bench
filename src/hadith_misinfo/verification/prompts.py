"""Verification prompts and response parsing for Systems S1–S4.

S1 / S2  — parametric-only: no retrieved evidence shown to the LLM.
S3 / S4  — retrieval-augmented: top-k evidence injected into the prompt.

Anti-leakage guarantee:
    Only ``InferenceRecord`` (claim_text, language) and — for S3/S4 —
    retrieved ``EvidenceRecord`` objects ever reach these prompts.
    Source ID, gold label, and canonical Arabic text NEVER appear.

Verdict format (§9 of design doc):
    SUPPORTED | NOT_SUPPORTED | INSUFFICIENT_EVIDENCE
"""

from __future__ import annotations

from hadith_misinfo.schemas import EvidenceRecord, InferenceRecord

# ── System preamble ────────────────────────────────────────────────────────────

_SYSTEM_PREAMBLE = """\
You are a religious-text verification assistant. Your task is to assess \
whether a claim about an Islamic saying or narration (Hadith) is supported \
by canonical evidence.

Important caveats:
- You are NOT a religious authority. Your role is computational verification.
- Absence of evidence in the provided sources does NOT prove a Hadith is \
fabricated — it only means it is unverifiable from the available corpus.
- Never declare a Hadith definitively "fake" based solely on non-retrieval.

Choose exactly one verdict:
  SUPPORTED            — The provided evidence clearly supports the claim \
and its attribution.
  NOT_SUPPORTED        — The evidence explicitly contradicts the claim or \
establishes the attribution is incorrect.
  INSUFFICIENT_EVIDENCE — You cannot determine support or contradiction from \
the available information. Use this when uncertain.

Respond in this exact format (nothing else):
VERDICT: <SUPPORTED|NOT_SUPPORTED|INSUFFICIENT_EVIDENCE>
CONTRADICTS: <yes|no>
EXPLANATION: <1–3 sentences. Cite specific evidence if available.>\
"""


# ── Prompt builders ────────────────────────────────────────────────────────────

def build_no_rag_prompt(record: InferenceRecord) -> str:
    """S1 (English parametric) / S2 (Bangla parametric) — no retrieved evidence."""
    return (
        f"{_SYSTEM_PREAMBLE}\n\n"
        f"Claim (language: {record.language}):\n"
        f"{record.claim_text}\n\n"
        "Based on your knowledge, assess whether this claim reflects an \n"
        "authentic Hadith attribution. If uncertain, choose INSUFFICIENT_EVIDENCE."
    )


def build_rag_prompt(
    record: InferenceRecord,
    evidence: list[EvidenceRecord],
    retrieval_text_mode: str = "arabic_plus_english",
) -> str:
    """S3 (English RAG) / S4 (Bangla cross-lingual RAG) — evidence injected."""
    if evidence:
        evidence_block = "\n\n".join(
            f"[Evidence {i + 1}]\n"
            f"Source: {e.collection}, {e.reference}\n"
            f"{e.retrieval_text(retrieval_text_mode)}"  # type: ignore[arg-type]
            for i, e in enumerate(evidence)
        )
    else:
        evidence_block = "(No evidence retrieved from the canonical corpus.)"

    return (
        f"{_SYSTEM_PREAMBLE}\n\n"
        f"Retrieved canonical Hadith evidence:\n"
        f"{'─' * 60}\n"
        f"{evidence_block}\n"
        f"{'─' * 60}\n\n"
        f"Claim (language: {record.language}):\n"
        f"{record.claim_text}\n\n"
        "Compare the claim against the retrieved evidence above. \n"
        "Base your verdict on whether the evidence supports or contradicts \n"
        "the claim — do NOT rely on parametric knowledge not reflected in the \n"
        "retrieved sources."
    )


# ── Response parsing ───────────────────────────────────────────────────────────

def parse_verdict(raw_response: str) -> tuple[str, bool, str]:
    """Parse the VERDICT:/CONTRADICTS:/EXPLANATION: format.

    Returns
    -------
    verdict:           One of the three canonical verdict strings.
    contradicts_claim: True if the model reported the evidence contradicts the claim.
    explanation:       The explanation text.

    Raises
    ------
    ValueError if a VERDICT line cannot be found.
    """
    verdict: str | None = None
    contradicts: bool = False
    explanation_lines: list[str] = []
    in_explanation = False

    for line in raw_response.strip().splitlines():
        stripped = line.strip()
        upper = stripped.upper()

        if upper.startswith("VERDICT:"):
            raw_verdict = stripped.split(":", 1)[1].strip().upper()
            # Normalise to canonical values
            if "NOT_SUPPORTED" in raw_verdict or "NOT SUPPORTED" in raw_verdict:
                verdict = "NOT_SUPPORTED"
            elif "INSUFFICIENT" in raw_verdict:
                verdict = "INSUFFICIENT_EVIDENCE"
            elif "SUPPORTED" in raw_verdict:
                verdict = "SUPPORTED"
            else:
                verdict = raw_verdict  # keep unknown for debugging

        elif upper.startswith("CONTRADICTS:"):
            val = stripped.split(":", 1)[1].strip().lower()
            contradicts = val in ("yes", "true", "1")

        elif upper.startswith("EXPLANATION:"):
            in_explanation = True
            explanation_lines.append(stripped.split(":", 1)[1].strip())

        elif in_explanation and stripped:
            explanation_lines.append(stripped)

    if verdict is None:
        raise ValueError(
            f"Could not parse VERDICT from model output:\n{raw_response!r}"
        )

    # Validate
    valid_verdicts = {"SUPPORTED", "NOT_SUPPORTED", "INSUFFICIENT_EVIDENCE"}
    if verdict not in valid_verdicts:
        # Fallback to INSUFFICIENT_EVIDENCE rather than crashing the run
        verdict = "INSUFFICIENT_EVIDENCE"

    return verdict, contradicts, " ".join(explanation_lines).strip()
