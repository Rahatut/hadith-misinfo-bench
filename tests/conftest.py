"""Pytest fixtures shared across all tests."""

from __future__ import annotations

import pytest

from hadith_misinfo.schemas import (
    BenchmarkRecord,
    EvidenceRecord,
    InferenceRecord,
    RetrievalResult,
    VerificationResult,
)


@pytest.fixture
def sample_evidence_records() -> list[EvidenceRecord]:
    """A tiny 5-record evidence corpus for unit tests."""
    return [
        EvidenceRecord(
            evidence_id="bukhari_1",
            collection="Sahih al-Bukhari",
            book="Book of Revelation",
            reference="Hadith 1",
            arabic_matn="إِنَّمَا الْأَعْمَالُ بِالنِّيَّاتِ",
            english_text="Narrated Umar: The Prophet said: Actions are judged by intentions.",
            grade="Sahih",
        ),
        EvidenceRecord(
            evidence_id="bukhari_2",
            collection="Sahih al-Bukhari",
            book="Book of Prayer",
            reference="Hadith 2",
            arabic_matn="الطَّهُورُ شَطْرُ الْإِيمَانِ",
            english_text="Narrated Abu Malik: Cleanliness is half of faith.",
            grade="Sahih",
        ),
        EvidenceRecord(
            evidence_id="muslim_1",
            collection="Sahih Muslim",
            book="Book of Faith",
            reference="Hadith 34",
            arabic_matn="لَا يُؤْمِنُ أَحَدُكُمْ حَتَّى يُحِبَّ لِأَخِيهِ مَا يُحِبُّ لِنَفْسِهِ",
            english_text="None of you truly believes until he loves for his brother what he loves for himself.",
            grade="Sahih",
        ),
        EvidenceRecord(
            evidence_id="bukhari_3",
            collection="Sahih al-Bukhari",
            book="Book of Manners",
            reference="Hadith 6064",
            arabic_matn="مَنْ كَانَ يُؤْمِنُ بِاللَّهِ وَالْيَوْمِ الْآخِرِ فَلْيَقُلْ خَيْرًا أَوْ لِيَصْمُتْ",
            english_text="Whoever believes in Allah and the Last Day should say good things or be silent.",
            grade="Sahih",
        ),
        EvidenceRecord(
            evidence_id="muslim_2",
            collection="Sahih Muslim",
            book="Book of Remembrance",
            reference="Hadith 2699",
            arabic_matn="مَنْ سَلَكَ طَرِيقًا يَلْتَمِسُ فِيهِ عِلْمًا سَهَّلَ اللَّهُ لَهُ طَرِيقًا إِلَى الْجَنَّةِ",
            english_text="Whoever takes a path in seeking knowledge, Allah will make easy for him a path to Paradise.",
            grade="Sahih",
        ),
    ]


@pytest.fixture
def sample_benchmark_record() -> BenchmarkRecord:
    return BenchmarkRecord(
        claim_id="C0001",
        source_id="mahaddat_001",
        label="authentic",
        claims={
            "en": "The Prophet said: Actions are judged by their intentions.",
            "bn": "নবী (সাঃ) বলেছেন: কাজগুলো তাদের নিয়তের উপর নির্ভর করে।",
        },
        canonical={
            "ar": "إِنَّمَا الْأَعْمَالُ بِالنِّيَّاتِ",
            "en": "Actions are judged by intentions.",
        },
        gold_evidence_ids=["bukhari_1"],
    )


@pytest.fixture
def sample_inference_record() -> InferenceRecord:
    return InferenceRecord(
        claim_id="C0001",
        language="en",
        claim_text="The Prophet said: Actions are judged by their intentions.",
    )


@pytest.fixture
def sample_retrieval_results() -> list[RetrievalResult]:
    return [
        RetrievalResult(claim_id="C0001", evidence_id="bukhari_2", rank=1, score=0.85),
        RetrievalResult(claim_id="C0001", evidence_id="bukhari_1", rank=2, score=0.82),
        RetrievalResult(claim_id="C0001", evidence_id="muslim_1", rank=3, score=0.75),
        RetrievalResult(claim_id="C0001", evidence_id="bukhari_3", rank=4, score=0.70),
        RetrievalResult(claim_id="C0001", evidence_id="muslim_2", rank=5, score=0.65),
    ]
