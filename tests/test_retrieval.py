"""Tests for retrieval modules (BM25, hybrid RRF, and retrieval metrics)."""

from __future__ import annotations

import pytest

from hadith_misinfo.evaluation.retrieval import (
    compute_retrieval_metrics,
    mean_reciprocal_rank,
    recall_at_k,
)
from hadith_misinfo.retrieval.bm25 import BM25Retriever
from hadith_misinfo.retrieval.hybrid import rrf_fuse
from hadith_misinfo.schemas import RetrievalResult


class TestBM25Retriever:
    @pytest.fixture
    def retriever(self, sample_evidence_records):
        return BM25Retriever.build(sample_evidence_records, mode="arabic_plus_english", verbose=False)

    def test_search_returns_k_results(self, retriever):
        results = retriever.search("Actions are judged by intentions", k=3)
        assert len(results) == 3

    def test_search_returns_tuples(self, retriever):
        results = retriever.search("faith cleanliness", k=2)
        for eid, score in results:
            assert isinstance(eid, str)
            assert isinstance(score, float)

    def test_search_to_results(self, retriever):
        results = retriever.search_to_results("C0001", "intentions prayer", k=3)
        assert len(results) == 3
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert results[0].rank == 1
        assert results[1].rank == 2
        assert results[2].rank == 3
        assert all(r.claim_id == "C0001" for r in results)

    def test_search_english_query(self, retriever):
        results = retriever.search("cleanliness half of faith", k=5)
        eids = [eid for eid, _ in results]
        assert "bukhari_2" in eids

    def test_save_and_load(self, retriever, tmp_path):
        retriever.save(tmp_path)
        loaded = BM25Retriever.load(tmp_path)
        orig = retriever.search("faith", k=3)
        loaded_r = loaded.search("faith", k=3)
        assert [eid for eid, _ in orig] == [eid for eid, _ in loaded_r]


class TestRRFFuse:
    def test_basic_fusion(self):
        list1 = [("A", 0.9), ("B", 0.8), ("C", 0.7)]
        list2 = [("B", 0.95), ("A", 0.85), ("D", 0.6)]
        fused = rrf_fuse([list1, list2], top_k=4)
        eids = [eid for eid, _ in fused]
        assert "A" in eids[:2]
        assert "B" in eids[:2]

    def test_top_k_respected(self):
        list1 = [("A", 0.9), ("B", 0.8), ("C", 0.7), ("D", 0.6)]
        list2 = [("D", 0.9), ("C", 0.8), ("B", 0.7), ("A", 0.6)]
        fused = rrf_fuse([list1, list2], top_k=2)
        assert len(fused) == 2

    def test_scores_are_positive(self):
        list1 = [("A", 0.5), ("B", 0.4)]
        fused = rrf_fuse([list1], top_k=2)
        assert all(score > 0 for _, score in fused)

    def test_empty_list(self):
        fused = rrf_fuse([], top_k=5)
        assert fused == []


class TestRetrievalMetrics:
    def test_recall_at_1(self, sample_retrieval_results):
        gold = {"C0001": ["bukhari_1"]}
        assert recall_at_k(sample_retrieval_results, gold, k=1) == 0.0

    def test_recall_at_2(self, sample_retrieval_results):
        gold = {"C0001": ["bukhari_1"]}
        assert recall_at_k(sample_retrieval_results, gold, k=2) == 1.0

    def test_recall_at_5(self, sample_retrieval_results):
        gold = {"C0001": ["bukhari_1"]}
        assert recall_at_k(sample_retrieval_results, gold, k=5) == 1.0

    def test_mrr(self, sample_retrieval_results):
        gold = {"C0001": ["bukhari_1"]}
        mrr = mean_reciprocal_rank(sample_retrieval_results, gold)
        assert abs(mrr - 1 / 2) < 1e-6

    def test_compute_retrieval_metrics(self, sample_retrieval_results):
        gold = {"C0001": ["bukhari_1"]}
        metrics = compute_retrieval_metrics(sample_retrieval_results, gold)
        assert metrics.recall_at_1 == 0.0
        assert metrics.recall_at_5 == 1.0
        assert abs(metrics.mrr - 0.5) < 1e-6
