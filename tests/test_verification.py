"""Tests for verification engine, prompts, policies, and evaluation metrics."""

from __future__ import annotations

import pytest

from hadith_misinfo.evaluation.verification import (
    VerificationMetrics,
    compute_delta_bn,
    compute_verification_metrics,
    results_to_predictions,
)
from hadith_misinfo.schemas import InferenceRecord, VerificationResult
from hadith_misinfo.verification.policies import map_verdict_to_prediction, should_abstain
from hadith_misinfo.verification.prompts import (
    build_no_rag_prompt,
    build_rag_prompt,
    parse_verdict,
)
from hadith_misinfo.verification.verifier import Verifier


# ── Prompt Builder Tests ───────────────────────────────────────────────────────

class TestPromptBuilders:
    def test_no_rag_prompt_contains_claim(self):
        record = InferenceRecord(claim_id="C1", language="en", claim_text="The Prophet said X.")
        prompt = build_no_rag_prompt(record)
        assert "The Prophet said X." in prompt
        assert "language: en" in prompt

    def test_no_rag_prompt_bangla(self):
        record = InferenceRecord(claim_id="C1", language="bn", claim_text="নবী বলেছেন...")
        prompt = build_no_rag_prompt(record)
        assert "নবী বলেছেন..." in prompt
        assert "language: bn" in prompt

    def test_rag_prompt_with_evidence(self, sample_evidence_records):
        record = InferenceRecord(claim_id="C1", language="en", claim_text="The Prophet said X.")
        prompt = build_rag_prompt(record, sample_evidence_records[:2])
        assert "[Evidence 1]" in prompt
        assert "[Evidence 2]" in prompt
        assert "The Prophet said X." in prompt
        claim_section = prompt.split("Claim (language")[1]
        assert "authentic" not in claim_section.lower()
        assert "fabricated" not in claim_section.lower()

    def test_rag_prompt_no_evidence(self):
        record = InferenceRecord(claim_id="C1", language="bn", claim_text="...")
        prompt = build_rag_prompt(record, [])
        assert "No evidence retrieved" in prompt


# ── Response Parser Tests ─────────────────────────────────────────────────────

class TestParseVerdict:
    def test_parse_supported(self):
        raw = "VERDICT: SUPPORTED\nCONTRADICTS: no\nEXPLANATION: The evidence clearly supports this."
        verdict, contradicts, explanation = parse_verdict(raw)
        assert verdict == "SUPPORTED"
        assert contradicts is False
        assert "evidence clearly" in explanation

    def test_parse_not_supported(self):
        raw = "VERDICT: NOT_SUPPORTED\nCONTRADICTS: yes\nEXPLANATION: Contradicted by Bukhari 1."
        verdict, contradicts, explanation = parse_verdict(raw)
        assert verdict == "NOT_SUPPORTED"
        assert contradicts is True

    def test_parse_insufficient(self):
        raw = "VERDICT: INSUFFICIENT_EVIDENCE\nCONTRADICTS: no\nEXPLANATION: Cannot determine."
        verdict, contradicts, explanation = parse_verdict(raw)
        assert verdict == "INSUFFICIENT_EVIDENCE"
        assert contradicts is False

    def test_parse_missing_verdict_raises(self):
        with pytest.raises(ValueError, match="VERDICT"):
            parse_verdict("This is not a valid response.")


# ── Policy & Metrics Tests ────────────────────────────────────────────────────

class TestVerificationPoliciesAndMetrics:
    def test_label_mapping_policies(self):
        assert map_verdict_to_prediction("SUPPORTED") == "authentic"
        assert map_verdict_to_prediction("INSUFFICIENT_EVIDENCE") == "abstain"
        assert map_verdict_to_prediction("NOT_SUPPORTED", contradicts_claim=False) == "abstain"
        assert map_verdict_to_prediction("NOT_SUPPORTED", contradicts_claim=True) == "fabricated"

    def test_abstain_policy(self):
        assert should_abstain("INSUFFICIENT_EVIDENCE") is True
        assert should_abstain("SUPPORTED", confidence=0.8) is False
        assert should_abstain("SUPPORTED", confidence=0.3, min_confidence=0.5) is True

    def test_metrics_calculation(self):
        gold = {"C1": "authentic", "C2": "fabricated", "C3": "authentic"}
        preds = {"C1": "authentic", "C2": "abstain", "C3": "fabricated"}
        m = compute_verification_metrics(gold, preds)
        assert abs(m.strict_accuracy - 1 / 3) < 1e-6
        assert abs(m.coverage - 2 / 3) < 1e-6
        assert abs(m.selective_accuracy - 1 / 2) < 1e-6
        assert abs(m.abstention_rate - 1 / 3) < 1e-6

    def test_delta_bn(self):
        en = VerificationMetrics("S1", "en", 100, 0.80, 1.0, 0.80, 0.78, 0.10)
        bn = VerificationMetrics("S2", "bn", 100, 0.65, 1.0, 0.65, 0.62, 0.20)
        delta = compute_delta_bn(en, bn)
        assert delta["Δ_BN strict_accuracy"] == pytest.approx(0.15)
        assert delta["Δ_BN macro_f1"] == pytest.approx(0.16)


# ── Verifier Engine Tests ─────────────────────────────────────────────────────

def test_mock_verifier_s1():
    mock_complete = lambda prompt: "VERDICT: SUPPORTED\nCONTRADICTS: no\nEXPLANATION: Supported by knowledge."
    verifier = Verifier(complete=mock_complete)
    record = InferenceRecord(claim_id="C001", language="en", claim_text="Actions are by intention.")
    res = verifier.verify(record, system="S1")
    assert res.verdict == "SUPPORTED"
    assert res.contradicts_claim is False
