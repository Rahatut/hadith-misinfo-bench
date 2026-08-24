#!/usr/bin/env python
"""Evaluate retrieval performance (Recall@1, Recall@5, MRR) on authentic claims.

Compares BM25 and Dense (BGE-M3) retrievers across English (C_EN) and
Bangla (C_BN) queries against gold evidence IDs.

Usage:
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --benchmark data/processed/benchmark_dataset_a.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hadith_misinfo.benchmark.sampler import load_benchmark
from hadith_misinfo.config import settings
from hadith_misinfo.evaluation.retrieval import compute_retrieval_metrics
from hadith_misinfo.schemas import RetrievalResult


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", default=str(settings.benchmark_path),
                        help="Path to benchmark JSONL file.")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of documents to retrieve.")
    parser.add_argument("--out-dir", default=str(settings.results_dir / "retrieval"),
                        help="Directory to save retrieval results.")
    args = parser.parse_args()

    benchmark = load_benchmark(args.benchmark)
    auth_records = [r for r in benchmark if r.label == "authentic" and r.gold_evidence_ids]
    print(f"\n📊  Found {len(auth_records)} authentic benchmark records with gold evidence IDs.")

    if not auth_records:
        print("⚠️  No authentic records with gold_evidence_ids found in benchmark.")
        return

    gold_ids = {r.claim_id: r.gold_evidence_ids for r in auth_records}

    # ── Load retrievers ────────────────────────────────────────────────────────
    retrievers = {}
    if (settings.bm25_index_dir / "bm25.pkl").exists():
        from hadith_misinfo.retrieval.bm25 import BM25Retriever
        retrievers["BM25"] = BM25Retriever.load(settings.bm25_index_dir)

    if (settings.dense_index_dir / "embeddings.npy").exists():
        from hadith_misinfo.retrieval.dense import DenseRetriever
        retrievers["Dense (BGE-M3)"] = DenseRetriever.load(settings.dense_index_dir)

    if not retrievers:
        print("❌  No indices found in data/indices/bm25 or data/indices/dense.")
        return

    print("\n## Retrieval Performance on Authentic Claims (Table 4)\n")
    header = f"| {'Retriever':<22} | {'Query Lang / Mode':<24} | {'Recall@1':>10} | {'Recall@5':>10} | {'MRR':>8} |"
    sep = "|" + "|".join(["-" * (len(h) + 2) for h in header.split("|")[1:-1]]) + "|"
    print(header)
    print(sep)

    for ret_name, retriever in retrievers.items():
        # English query
        all_results_en: list[RetrievalResult] = []
        for rec in auth_records:
            query = rec.claims.get("en", "")
            res = retriever.search_to_results(claim_id=rec.claim_id, query=query, k=args.k)
            all_results_en.extend(res)
        m_en = compute_retrieval_metrics(all_results_en, gold_ids, k_values=[1, 5])
        print(
            f"| {ret_name:<22} | {'English (Direct)':<24} | "
            f"{m_en.recall_at_1:>10.3f} | {m_en.recall_at_5:>10.3f} | "
            f"{m_en.mrr:>8.3f} |"
        )

        # Bangla direct query
        all_results_bn: list[RetrievalResult] = []
        for rec in auth_records:
            query = rec.claims.get("bn", "")
            res = retriever.search_to_results(claim_id=rec.claim_id, query=query, k=args.k)
            all_results_bn.extend(res)
        m_bn = compute_retrieval_metrics(all_results_bn, gold_ids, k_values=[1, 5])
        print(
            f"| {ret_name:<22} | {'Bangla (Direct)':<24} | "
            f"{m_bn.recall_at_1:>10.3f} | {m_bn.recall_at_5:>10.3f} | "
            f"{m_bn.mrr:>8.3f} |"
        )
    print()


if __name__ == "__main__":
    main()
