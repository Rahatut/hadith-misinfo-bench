"""Tests for core schemas."""

from __future__ import annotations

import json

import pytest

from hadith_misinfo.schemas import (
    BenchmarkRecord,
    EvidenceRecord,
    ExtractedClaim,
    InferenceRecord,
    MitigationResult,
    RetrievalResult,
    SocialMediaPost,
    VerificationResult,
    to_inference_record,
)


class TestEvidenceRecord:
    def test_round_trip_dict(self, sample_evidence_records):
        for rec in sample_evidence_records:
            d = rec.to_dict()
            restored = EvidenceRecord.from_dict(d)
            assert restored.evidence_id == rec.evidence_id
            assert restored.arabic_matn == rec.arabic_matn

    def test_retrieval_text_modes(self, sample_evidence_records):
        rec = sample_evidence_records[0]
        ar_only = rec.retrieval_text("arabic_only")
        en_only = rec.retrieval_text("english_only")
        both = rec.retrieval_text("arabic_plus_english")
        assert rec.arabic_matn in ar_only
        assert rec.english_text in en_only
        assert rec.arabic_matn in both
        assert rec.english_text in both

    def test_retrieval_text_no_english(self):
        rec = EvidenceRecord(
            evidence_id="x_1", collection="X", book="B", reference="1",
            arabic_matn="عربي", english_text=None,
        )
        # english_only should fall back to arabic_matn
        assert rec.retrieval_text("english_only") == "عربي"


class TestBenchmarkRecord:
    def test_round_trip_dict(self, sample_benchmark_record):
        d = sample_benchmark_record.to_dict()
        restored = BenchmarkRecord.from_dict(d)
        assert restored.claim_id == sample_benchmark_record.claim_id
        assert restored.label == sample_benchmark_record.label
        assert restored.claims["en"] == sample_benchmark_record.claims["en"]
        assert restored.claims["bn"] == sample_benchmark_record.claims["bn"]

    def test_gold_evidence_ids_preserved(self, sample_benchmark_record):
        d = sample_benchmark_record.to_dict()
        restored = BenchmarkRecord.from_dict(d)
        assert restored.gold_evidence_ids == ["bukhari_1"]


class TestInferenceRecord:
    def test_to_inference_record_en(self, sample_benchmark_record):
        ir = to_inference_record(sample_benchmark_record, "en")
        assert ir.claim_id == "C0001"
        assert ir.language == "en"
        assert "Actions are judged" in ir.claim_text

    def test_to_inference_record_bn(self, sample_benchmark_record):
        ir = to_inference_record(sample_benchmark_record, "bn")
        assert ir.language == "bn"
        assert "নবী" in ir.claim_text

    def test_anti_leakage_no_label(self, sample_benchmark_record):
        ir = to_inference_record(sample_benchmark_record, "en")
        # InferenceRecord must NOT have a label attribute
        assert not hasattr(ir, "label")
        assert not hasattr(ir, "source_id")
        assert not hasattr(ir, "gold_evidence_ids")

    def test_missing_language_raises(self, sample_benchmark_record):
        with pytest.raises(KeyError):
            to_inference_record(sample_benchmark_record, "ar")  # type: ignore


class TestVerificationResult:
    def test_round_trip_dict(self):
        vr = VerificationResult(
            claim_id="C0001",
            system="S4",
            language="bn",
            verdict="SUPPORTED",
            explanation="Test explanation.",
            retrieved_evidence_ids=["bukhari_1"],
            contradicts_claim=False,
        )
        d = vr.to_dict()
        restored = VerificationResult.from_dict(d)
        assert restored.verdict == "SUPPORTED"
        assert restored.retrieved_evidence_ids == ["bukhari_1"]

    def test_valid_verdicts(self):
        for verdict in ("SUPPORTED", "NOT_SUPPORTED", "INSUFFICIENT_EVIDENCE"):
            vr = VerificationResult(
                claim_id="C0001", system="S1", language="en",
                verdict=verdict, explanation="",
            )
            assert vr.verdict == verdict


class TestSocialMediaPost:
    def test_round_trip(self):
        post = SocialMediaPost(post_id="p1", raw_text="ভাই শেয়ার করুন।", source="al-zaman")
        d = post.to_dict()
        restored = SocialMediaPost.from_dict(d)
        assert restored.post_id == "p1"
        assert restored.raw_text == "ভাই শেয়ার করুন।"
