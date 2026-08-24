#!/usr/bin/env python
"""Build the controlled benchmark (Dataset A) from MAHADDAT test records.

Steps:
  1. Ingest MAHADDAT test split
  2. Sample 250 authentic + 250 fabricated records (balanced, seed=42)
  3. (Optional) Generate English + Bangla paraphrased claims via LLM
  4. (Optional) Match authentic records against evidence corpus for gold IDs
  5. Save to data/processed/benchmark_dataset_a.jsonl

The paired structure (C001_EN / C001_BN) is essential for the Δ_BN metric.

Usage:
    # Full pipeline with LLM paraphrasing
    python scripts/build_benchmark.py --provider openai --model gpt-4o-mini

    # Quick test without paraphrasing (uses Arabic matn as placeholder claim)
    python scripts/build_benchmark.py --no-paraphrase

    # Custom sample size
    python scripts/build_benchmark.py --n-authentic 10 --n-fabricated 10 --no-paraphrase
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hadith_misinfo.benchmark.sampler import (
    attach_gold_evidence,
    load_benchmark,
    sample_balanced,
    save_benchmark,
    to_benchmark_record,
)
from hadith_misinfo.config import settings
from hadith_misinfo.ingestion.mahaddat import iter_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mahaddat-dir", default=str(settings.mahaddat_raw_dir))
    parser.add_argument("--split", default="test", help="Which MAHADDAT split to use. Default: test.")
    parser.add_argument("--n-authentic", type=int, default=settings.benchmark_n_authentic)
    parser.add_argument("--n-fabricated", type=int, default=settings.benchmark_n_fabricated)
    parser.add_argument("--seed", type=int, default=settings.benchmark_seed)
    parser.add_argument("--out", default=str(settings.benchmark_path))

    # Paraphrasing
    parser.add_argument("--no-paraphrase", action="store_true",
                        help="Skip LLM paraphrasing (use Arabic matn as placeholder claim text).")
    parser.add_argument("--provider", default=settings.llm_provider,
                        choices=["openai", "anthropic", "ollama"])
    parser.add_argument("--model", default=settings.llm_model)

    # Gold evidence matching
    parser.add_argument("--no-gold-match", action="store_true",
                        help="Skip BM25 gold evidence matching for authentic records.")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Number of concurrent threads for LLM paraphrasing.")
    args = parser.parse_args()

    settings.ensure_dirs()

    # ── Step 1: Ingest MAHADDAT ───────────────────────────────────────────────
    print(f"\n📥  Step 1: Ingesting MAHADDAT '{args.split}' split from {args.mahaddat_dir}...")
    all_records = list(iter_records(args.mahaddat_dir, split=args.split))
    authentic_n = sum(1 for r in all_records if r.label == "authentic")
    fabricated_n = sum(1 for r in all_records if r.label == "fabricated")
    print(f"    Loaded {len(all_records):,} records: {authentic_n:,} authentic, {fabricated_n:,} fabricated.")

    # ── Step 2: Sample balanced set ──────────────────────────────────────────
    print(f"\n🎲  Step 2: Sampling {args.n_authentic} authentic + {args.n_fabricated} fabricated (seed={args.seed})...")
    sampled = sample_balanced(all_records, args.n_authentic, args.n_fabricated, args.seed)
    print(f"    Selected {len(sampled)} records.")

    # ── Step 3: Gold evidence matching ───────────────────────────────────────
    if not args.no_gold_match:
        print(f"\n🔍  Step 3: Matching authentic records against evidence corpus (BM25)...")
        try:
            from hadith_misinfo.retrieval.bm25 import BM25Retriever
            bm25 = BM25Retriever.load(settings.bm25_index_dir)
            sampled = attach_gold_evidence(sampled, evidence_store=None, retriever=bm25)
            n_matched = sum(1 for r in sampled if r.gold_evidence_ids)
            print(f"    Matched {n_matched} authentic records to gold evidence IDs.")
        except FileNotFoundError:
            print("    ⚠️  BM25 index not found — skipping gold match.")
            print("    Run: python scripts/build_evidence_index.py --retriever bm25")
    else:
        print("\n⏭️   Step 3: Skipping gold evidence matching (--no-gold-match).")

    # ── Step 4: Paraphrasing ─────────────────────────────────────────────────
    if args.no_paraphrase:
        print("\n⏭️   Step 4: Skipping paraphrasing — using Arabic matn as placeholder.")
        benchmark_records = [
            to_benchmark_record(
                raw=r,
                claim_id=f"C{str(i + 1).zfill(4)}",
                claim_en=f"[EN paraphrase pending] {r.arabic_matn[:100]}",
                claim_bn=f"[BN paraphrase pending] {r.arabic_matn[:100]}",
            )
            for i, r in enumerate(sampled)
        ]
    else:
        print(f"\n✍️   Step 4: Generating EN + BN paraphrases via {args.provider}/{args.model} (concurrency={args.concurrency})...")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import json
        from hadith_misinfo.benchmark.paraphraser import make_llm_paraphraser, paraphrase_pair
        from hadith_misinfo.llm.base import make_complete_fn
        from tqdm import tqdm

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        existing_records = {}
        if out_path.exists():
            try:
                for r in load_benchmark(out_path):
                    existing_records[r.claim_id] = r
                print(f"    Resuming from {len(existing_records)} existing records in {args.out}...")
            except Exception:
                pass

        complete = make_complete_fn(
            provider=args.provider,
            model=args.model,
            temperature=0.3,
            retry=True,
            retry_attempts=5,
        )
        paraphraser = make_llm_paraphraser(complete)

        def _process_item(item):
            i, r = item
            cid = f"C{str(i + 1).zfill(4)}"
            if cid in existing_records:
                return existing_records[cid]
            claim_en, claim_bn = paraphrase_pair(r.arabic_matn, paraphraser)
            rec = to_benchmark_record(
                raw=r,
                claim_id=cid,
                claim_en=claim_en,
                claim_bn=claim_bn,
            )
            # Append immediately to file
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
            return rec

        benchmark_records = [None] * len(sampled)
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_to_idx = {
                executor.submit(_process_item, (i, r)): i
                for i, r in enumerate(sampled)
            }
            for future in tqdm(as_completed(future_to_idx), total=len(sampled), desc="Building benchmark"):
                idx = future_to_idx[future]
                benchmark_records[idx] = future.result()

        # Re-save cleanly sorted by claim_id
        save_benchmark(benchmark_records, args.out)

    # Summary
    authentic_saved = sum(1 for r in benchmark_records if r.label == "authentic")
    fabricated_saved = sum(1 for r in benchmark_records if r.label == "fabricated")
    gold_matched = sum(1 for r in benchmark_records if r.gold_evidence_ids)
    print(f"\n✅  Benchmark built:")
    print(f"    Total: {len(benchmark_records)}")
    print(f"    Authentic: {authentic_saved}  |  Fabricated: {fabricated_saved}")
    print(f"    With gold evidence IDs: {gold_matched}")
    print(f"\nNext step: python scripts/run_experiment.py --system S1")


if __name__ == "__main__":
    main()
