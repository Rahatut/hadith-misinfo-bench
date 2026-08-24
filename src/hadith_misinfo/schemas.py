"""Central data schemas for HadithMisinfoBench.

Anti-leakage contract
---------------------
The only object that ever reaches the LLM at inference time is
``InferenceRecord``.  It contains *only* the claim text and its language —
no source ID, no gold label, no canonical Arabic text, no evidence IDs.

Callers are responsible for ensuring this boundary.  The ``to_inference_record``
helper enforces it in code.

Verdict three-class schema (see §9 of the design doc)
------------------------------------------------------
SUPPORTED            – retrieved canonical evidence supports the claim.
NOT_SUPPORTED        – evidence explicitly contradicts the claim or attribution.
INSUFFICIENT_EVIDENCE – available corpus cannot verify or refute the claim.

Mapping to ground-truth binary labels (§10, §12)
-------------------------------------------------
SUPPORTED            → AUTHENTIC   (evaluated as correct only for authentic claims)
NOT_SUPPORTED        → FABRICATED  *only when* the evidence explicitly contradicts
                                    (not when nothing was retrieved)
INSUFFICIENT_EVIDENCE → ABSTAIN    (never counted as correct for either class)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ── Label types ───────────────────────────────────────────────────────────────

GroundTruthLabel = Literal["authentic", "fabricated"]
VerdictLabel = Literal["SUPPORTED", "NOT_SUPPORTED", "INSUFFICIENT_EVIDENCE"]
PredictionLabel = Literal["authentic", "fabricated", "abstain"]
Language = Literal["en", "bn", "ar"]
RetrieverMode = Literal["arabic_plus_english", "arabic_only", "english_only"]
SystemId = Literal["S1", "S2", "S3", "S4"]


# ── Evidence corpus ───────────────────────────────────────────────────────────


@dataclass
class EvidenceRecord:
    """One canonical Hadith from Bukhari or Muslim.

    This is the retrieval unit.  Do NOT split Hadith into sub-chunks.
    """

    evidence_id: str             # e.g. "bukhari_1234"
    collection: str              # e.g. "Sahih al-Bukhari"
    book: str                    # book name or number
    reference: str               # human-readable reference (e.g. "Hadith 1234")
    arabic_matn: str             # original Arabic text
    english_text: str | None     # English translation (narrator + text)
    grade: str | None = None     # e.g. "Sahih", "Hasan"

    def retrieval_text(self, mode: RetrieverMode = "arabic_plus_english") -> str:
        """Return the text used for indexing and retrieval.

        Modes
        -----
        arabic_plus_english  – Arabic matn followed by English translation (default).
        arabic_only          – Arabic matn only.
        english_only         – English translation only (skips entries with no translation).
        """
        if mode == "arabic_only":
            return self.arabic_matn
        if mode == "english_only":
            return self.english_text or self.arabic_matn
        # arabic_plus_english
        parts = [self.arabic_matn]
        if self.english_text:
            parts.append(self.english_text)
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "collection": self.collection,
            "book": self.book,
            "reference": self.reference,
            "arabic_matn": self.arabic_matn,
            "english_text": self.english_text,
            "grade": self.grade,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvidenceRecord":
        return cls(
            evidence_id=d["evidence_id"],
            collection=d["collection"],
            book=d.get("book", ""),
            reference=d.get("reference", ""),
            arabic_matn=d["arabic_matn"],
            english_text=d.get("english_text"),
            grade=d.get("grade"),
        )


# ── Benchmark dataset ─────────────────────────────────────────────────────────


@dataclass
class BenchmarkRecord:
    """One paired claim in the controlled benchmark (Dataset A).

    This object lives in the benchmark JSONL and is used by the harness
    to drive evaluation.  It must NEVER be passed directly to the LLM.
    """

    claim_id: str                       # e.g. "C001"
    source_id: str                      # original MAHADDAT row ID
    label: GroundTruthLabel             # "authentic" | "fabricated"
    claims: dict[Language, str]         # {"en": "...", "bn": "..."}
    canonical: dict[Language, str]      # {"ar": "...", "en": "..."}
    gold_evidence_ids: list[str]        # populated only for authentic claims

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "source_id": self.source_id,
            "label": self.label,
            "claims": self.claims,
            "canonical": self.canonical,
            "gold_evidence_ids": self.gold_evidence_ids,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BenchmarkRecord":
        return cls(
            claim_id=d["claim_id"],
            source_id=d["source_id"],
            label=d["label"],
            claims=d["claims"],
            canonical=d.get("canonical", {}),
            gold_evidence_ids=d.get("gold_evidence_ids", []),
        )


# ── Inference boundary (anti-leakage) ────────────────────────────────────────


@dataclass
class InferenceRecord:
    """The ONLY object that should reach the LLM at inference time.

    Contains no label, no source ID, no canonical text, no evidence IDs.
    Use ``to_inference_record()`` to construct from a ``BenchmarkRecord``.
    """

    claim_id: str
    language: Language
    claim_text: str


def to_inference_record(
    record: BenchmarkRecord,
    language: Language,
) -> InferenceRecord:
    """Extract an InferenceRecord from a BenchmarkRecord, enforcing the
    anti-leakage boundary."""
    claim_text = record.claims.get(language)
    if not claim_text:
        raise KeyError(
            f"BenchmarkRecord {record.claim_id!r} has no claim for language {language!r}. "
            f"Available languages: {list(record.claims.keys())}"
        )
    return InferenceRecord(
        claim_id=record.claim_id,
        language=language,
        claim_text=claim_text,
    )


# ── Retrieval output ──────────────────────────────────────────────────────────


@dataclass
class RetrievalResult:
    """One retrieved document for a given claim query."""

    claim_id: str
    evidence_id: str
    rank: int           # 1-indexed
    score: float        # cosine similarity or BM25 score

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "evidence_id": self.evidence_id,
            "rank": self.rank,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RetrievalResult":
        return cls(**d)


# ── Verification output ───────────────────────────────────────────────────────


@dataclass
class VerificationResult:
    """Full output of one verification call (LLM + optional retrieval)."""

    claim_id: str
    system: SystemId
    language: Language
    verdict: VerdictLabel
    explanation: str
    retrieved_evidence_ids: list[str] = field(default_factory=list)
    # Set to True if the LLM explicitly stated the evidence contradicts the claim.
    # Required for mapping NOT_SUPPORTED → fabricated vs abstain (see §10).
    contradicts_claim: bool = False
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "system": self.system,
            "language": self.language,
            "verdict": self.verdict,
            "explanation": self.explanation,
            "retrieved_evidence_ids": self.retrieved_evidence_ids,
            "contradicts_claim": self.contradicts_claim,
            "raw_response": self.raw_response,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VerificationResult":
        return cls(
            claim_id=d["claim_id"],
            system=d["system"],
            language=d["language"],
            verdict=d["verdict"],
            explanation=d["explanation"],
            retrieved_evidence_ids=d.get("retrieved_evidence_ids", []),
            contradicts_claim=d.get("contradicts_claim", False),
            raw_response=d.get("raw_response", ""),
        )


# ── Mitigation output ─────────────────────────────────────────────────────────


@dataclass
class MitigationResult:
    """The evidence-grounded user-facing intervention (Layer 3)."""

    claim_id: str
    verdict: VerdictLabel
    intervention_text: str          # Human-readable explanation shown to user
    retrieved_evidence: list[EvidenceRecord] = field(default_factory=list)
    grounded: bool | None = None    # Set during manual grounding audit

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "verdict": self.verdict,
            "intervention_text": self.intervention_text,
            "retrieved_evidence": [e.to_dict() for e in self.retrieved_evidence],
            "grounded": self.grounded,
        }


# ── Social media (Dataset C) ──────────────────────────────────────────────────


@dataclass
class SocialMediaPost:
    """One raw social-media post from the Al-Zaman/Noman dataset (Dataset C)."""

    post_id: str
    raw_text: str
    source: str = "al-zaman"    # dataset provenance
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "post_id": self.post_id,
            "raw_text": self.raw_text,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SocialMediaPost":
        return cls(
            post_id=d["post_id"],
            raw_text=d["raw_text"],
            source=d.get("source", "al-zaman"),
            metadata=d.get("metadata", {}),
        )


# ── Claim extraction output ───────────────────────────────────────────────────


@dataclass
class ExtractedClaim:
    """Result of Layer 1 claim extraction from a social-media post."""

    post_id: str
    claim_text: str
    language: Language
    claim_type: Literal["hadith_attribution", "religious_general", "other", "none"]
    confidence: float           # 0.0–1.0 (LLM self-reported or heuristic)
    extraction_notes: str = ""  # Optional LLM explanation for the extraction decision

    def to_dict(self) -> dict:
        return {
            "post_id": self.post_id,
            "claim_text": self.claim_text,
            "language": self.language,
            "claim_type": self.claim_type,
            "confidence": self.confidence,
            "extraction_notes": self.extraction_notes,
        }
