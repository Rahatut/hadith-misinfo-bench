#!/usr/bin/env python
"""Run one verification experiment (S1, S2, S3, or S4).

Results are streamed to a JSONL file as they are produced — safe to
interrupt and resume (duplicate claim_ids are deduplicated at evaluation time).

Usage:
    # Parametric English (S1)
    python scripts/run_experiment.py --system S1

    # Parametric Bangla (S2)
    python scripts/run_experiment.py --system S2

    # RAG English with Dense retrieval (S3)
    python scripts/run_experiment.py --system S3 --retriever dense

    # RAG Bangla cross-lingual with BM25 (S4)
    python scripts/run_experiment.py --system S4 --retriever bm25

    # Quick smoke test (10 records)
    python scripts/run_experiment.py --system S1 --n-records 10

    # Use a specific LLM
    python scripts/run_experiment.py --system S3 --provider anthropic --model claude-3-haiku-20240307
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hadith_misinfo.benchmark.sampler import load_benchmark
from hadith_misinfo.config import settings
from hadith_misinfo.schemas import to_inference_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", required=True, choices=["S1", "S2", "S3", "S4"],
                        help="Experiment system to run.")
    parser.add_argument("--benchmark", default=str(settings.benchmark_path),
                        help="Path to the benchmark JSONL file.")
    parser.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default="dense",
                        help="Retriever to use for S3/S4. Default: dense.")
    parser.add_argument("--k", type=int, default=settings.default_k,
                        help=f"Number of evidence documents to retrieve. Default: {settings.default_k}")
    parser.add_argument("--provider", default=settings.llm_provider,
                        choices=["openai", "anthropic", "ollama"])
    parser.add_argument("--model", default=settings.llm_model)
    parser.add_argument("--temperature", type=float, default=settings.llm_temperature)
    parser.add_argument("--out", default=None,
                        help="Output JSONL path. Default: results/<system>_<retriever>.jsonl")
    parser.add_argument("--n-records", type=int, default=None,
                        help="Limit to N records (for smoke tests).")
    parser.add_argument("--mode", default=settings.retrieval_text_mode,
                        choices=["arabic_plus_english", "arabic_only", "english_only"])
    parser.add_argument("--translate-query", action="store_true",
                        help="Translate Bangla query to English search keywords for cross-lingual BM25.")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="Number of concurrent LLM verification workers. Default: 16.")
    args = parser.parse_args()

    settings.ensure_dirs()

    # ── Output path ────────────────────────────────────────────────────────────
    if args.out is None:
        ret_suffix = f"_{args.retriever}" if args.system in ("S3", "S4") else ""
        args.out = str(settings.results_dir / f"{args.system.lower()}{ret_suffix}.jsonl")

    # ── Load benchmark ─────────────────────────────────────────────────────────
    print(f"\n📂  Loading benchmark from {args.benchmark}...")
    benchmark = load_benchmark(args.benchmark)
    if args.n_records:
        benchmark = benchmark[: args.n_records]
        print(f"    ⚠️  Limiting to {args.n_records} records (smoke test mode).")
    print(f"    Loaded {len(benchmark)} benchmark records.")

    # ── Build InferenceRecords ─────────────────────────────────────────────────
    language = "en" if args.system in ("S1", "S3") else "bn"
    inference_records = [to_inference_record(rec, language) for rec in benchmark]

    # ── LLM setup ─────────────────────────────────────────────────────────────
    print(f"\n🤖  LLM: {args.provider}/{args.model}")
    from hadith_misinfo.llm.base import make_complete_fn
    complete = make_complete_fn(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_tokens=settings.llm_max_tokens,
        retry=True,
        retry_attempts=settings.llm_retry_attempts,
    )

    # ── Retriever setup ────────────────────────────────────────────────────────
    retriever = None
    evidence_store = None

    if args.system in ("S3", "S4"):
        print(f"\n🔍  Loading {args.retriever} retriever...")
        from hadith_misinfo.evidence.store import EvidenceStore
        evidence_store = EvidenceStore.load(settings.evidence_path)

        if args.retriever == "bm25":
            from hadith_misinfo.retrieval.bm25 import BM25Retriever
            retriever = BM25Retriever.load(settings.bm25_index_dir)
        elif args.retriever == "dense":
            from hadith_misinfo.retrieval.dense import DenseRetriever
            retriever = DenseRetriever.load(settings.dense_index_dir)
        elif args.retriever == "hybrid":
            from hadith_misinfo.retrieval.bm25 import BM25Retriever
            from hadith_misinfo.retrieval.dense import DenseRetriever
            from hadith_misinfo.retrieval.hybrid import HybridRetriever
            bm25 = BM25Retriever.load(settings.bm25_index_dir)
            dense = DenseRetriever.load(settings.dense_index_dir)
            retriever = HybridRetriever(bm25=bm25, dense=dense)

    # ── Run verification ───────────────────────────────────────────────────────
    from hadith_misinfo.verification.verifier import Verifier, run_batch

    query_translator = None
    if args.translate_query:
        print("    🌐  Enabled Cross-Lingual Query Translation (Bangla -> English keywords).")
        def query_translator(bn_claim: str) -> str:
            prompt = (
                "Translate the following Islamic claim from Bangla into clear English search keywords for Hadith retrieval. "
                "Output ONLY the English search keywords, nothing else.\n\n"
                f"Bangla Claim: {bn_claim}\n\nEnglish Search Query:"
            )
            return complete(prompt).strip()

    verifier = Verifier(
        complete=complete,
        evidence_store=evidence_store,
        retriever=retriever,
        retrieval_text_mode=args.mode,
        default_k=args.k,
        query_translator=query_translator,
    )

    print(f"\n🚀  Running system {args.system} on {len(inference_records)} records...")
    print(f"    Results streaming to: {args.out}\n")

    results = run_batch(
        verifier=verifier,
        records=inference_records,
        system=args.system,  # type: ignore[arg-type]
        k=args.k,
        out_path=args.out,
        verbose=True,
        concurrency=args.concurrency,
    )

    from collections import Counter
    verdict_counts = Counter(r.verdict for r in results)
    print(f"\n✅  Experiment {args.system} complete.")
    print(f"    Verdict distribution: {dict(verdict_counts)}")
    print(f"    Results saved to: {args.out}")
    print(f"\nNext step: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
