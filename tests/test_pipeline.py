"""End-to-end integration test of the 4-layer misinformation pipeline."""

from hadith_misinfo.benchmark.validator import validate_benchmark
from hadith_misinfo.evaluation.crosslingual import compute_crosslingual_analysis
from hadith_misinfo.evaluation.grounding import GroundingAuditItem, evaluate_grounding_audit
from hadith_misinfo.evaluation.verification import VerificationMetrics
from hadith_misinfo.evidence.store import EvidenceStore
from hadith_misinfo.extraction.extractor import ClaimExtractor
from hadith_misinfo.mitigation.responder import MitigationResponder
from hadith_misinfo.retrieval.bm25 import BM25Retriever
from hadith_misinfo.schemas import BenchmarkRecord, InferenceRecord
from hadith_misinfo.verification.verifier import Verifier


def test_full_four_layer_pipeline(sample_evidence_records):
    # 1. Evidence Store & Retriever Setup
    store = EvidenceStore()
    for rec in sample_evidence_records:
        store._records[rec.evidence_id] = rec
    bm25 = BM25Retriever.build(sample_evidence_records, verbose=False)

    # 2. Layer 1: Extraction
    post_text = "ভাই সবাই শেয়ার করুন! রাসূল (সাঃ) বলেছেন: কাজগুলো তাদের নিয়তের উপর নির্ভর করে।"
    mock_extractor_llm = lambda p: (
        '{"contains_hadith_claim": true, "confidence": 0.9, "language": "bn"}'
        if "determine if it contains" in p
        else '{"extracted_claim": "রাসূল (সাঃ) বলেছেন: কাজগুলো তাদের নিয়তের উপর নির্ভর করে।", "claim_type": "hadith_attribution"}'
    )
    extractor = ClaimExtractor(complete=mock_extractor_llm)
    extracted = extractor.extract(post_id="post_001", post_text=post_text)
    assert extracted.claim_type == "hadith_attribution"
    assert "নিয়ত" in extracted.claim_text

    # 3. Layer 2: Verification
    mock_verifier_llm = lambda p: (
        "VERDICT: SUPPORTED\nCONTRADICTS: no\nEXPLANATION: Matches Bukhari 1 on intentions."
    )
    verifier = Verifier(
        complete=mock_verifier_llm,
        evidence_store=store,
        retriever=bm25,
        default_k=3,
    )
    inference = InferenceRecord(
        claim_id=extracted.post_id,
        language=extracted.language,
        claim_text=extracted.claim_text,
    )
    verif_res = verifier.verify(inference, system="S4", k=3)
    assert verif_res.verdict == "SUPPORTED"
    assert len(verif_res.retrieved_evidence_ids) == 3

    # 4. Layer 3: Mitigation
    retrieved_ev = store.get_many(verif_res.retrieved_evidence_ids)
    responder = MitigationResponder()
    mitigation_json = responder.build_structured_response(
        claim_text=extracted.claim_text,
        verdict=verif_res.verdict,
        explanation=verif_res.explanation,
        retrieved_evidence=retrieved_ev,
    )
    assert mitigation_json["verdict"] == "SUPPORTED"
    assert mitigation_json["abstained"] is False
    assert len(mitigation_json["evidence"]) == 3
    assert "consistent with" in mitigation_json["recommended_action"]


def test_crosslingual_and_grounding_evaluation():
    # Grounding evaluation
    items = [
        GroundingAuditItem(
            claim_id="C1",
            verdict="SUPPORTED",
            retrieved_evidence_ids=["bukhari_1"],
            explanation="Valid",
            category="Grounded",
            citation_valid=True,
        ),
        GroundingAuditItem(
            claim_id="C2",
            verdict="SUPPORTED",
            retrieved_evidence_ids=["bukhari_2"],
            explanation="Hallucinated text",
            category="Ungrounded",
            citation_valid=False,
        ),
    ]
    summary = evaluate_grounding_audit(items)
    assert summary.total_audited == 2
    assert summary.grounding_rate == 0.5
    assert summary.citation_validity_rate == 0.5
