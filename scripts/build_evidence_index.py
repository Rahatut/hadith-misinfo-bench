#!/usr/bin/env python
"""Build the evidence index (BM25 and/or Dense) from raw Hadith JSON files.

This script must be run before any experiment that uses retrieval (S3/S4).

Steps:
  1. Ingest bukhari.json + muslim.json → EvidenceStore
  2. Save EvidenceStore to data/processed/evidence.jsonl
  3. Build BM25 index → data/indices/bm25/
  4. (Optional) Build Dense (BGE-M3) index → data/indices/dense/

Usage:
    # Both indices (recommended before first run)
    python scripts/build_evidence_index.py

    # BM25 only (no GPU required)
    python scripts/build_evidence_index.py --retriever bm25

    # Dense only
    python scripts/build_evidence_index.py --retriever dense

    # Custom data directory
    python scripts/build_evidence_index.py --raw-dir data/raw/hadith-json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Make src importable when running as a script ──────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hadith_misinfo.config import settings
from hadith_misinfo.evidence.store import EvidenceStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=str(settings.hadith_json_raw_dir),
                        help="Path to hadith-json raw directory.")
    parser.add_argument("--retriever", choices=["bm25", "dense", "both"], default="both",
                        help="Which index to build. Default: both.")
    parser.add_argument("--mode", default=settings.retrieval_text_mode,
                        choices=["arabic_plus_english", "arabic_only", "english_only"],
                        help="Retrieval text mode. Default: arabic_plus_english.")
    parser.add_argument("--dense-model", default=settings.dense_model,
                        help=f"Dense model name. Default: {settings.dense_model}")
    parser.add_argument("--out-dir", default=str(settings.indices_dir),
                        help="Output directory for indices. Default: data/indices/")
    args = parser.parse_args()

    settings.ensure_dirs()

    # ── Step 1: Build and save EvidenceStore ─────────────────────────────────
    print("\n🔍  Step 1/3: Building EvidenceStore from", args.raw_dir)
    store = EvidenceStore.build(args.raw_dir, verbose=True)
    store.save(settings.evidence_path)

    summary = store.summary()
    print(f"\n📚  Evidence corpus summary:")
    for k, v in summary.items():
        print(f"    {k}: {v}")

    evidence_records = store.all_records()

    # ── Step 2: BM25 index ────────────────────────────────────────────────────
    if args.retriever in ("bm25", "both"):
        print(f"\n📦  Step 2/3: Building BM25 index (mode={args.mode})...")
        from hadith_misinfo.retrieval.bm25 import BM25Retriever
        bm25 = BM25Retriever.build(evidence_records, mode=args.mode, verbose=True)
        out = Path(args.out_dir) / "bm25"
        bm25.save(out)

    # ── Step 3: Dense index ───────────────────────────────────────────────────
    if args.retriever in ("dense", "both"):
        print(f"\n🧠  Step 3/3: Building Dense index (model={args.dense_model}, mode={args.mode})...")
        print("    ⚠️  This may take 10–30 minutes on first run (downloads the model + encodes all Hadith).")
        from hadith_misinfo.retrieval.dense import DenseRetriever
        out = Path(args.out_dir) / "dense"
        dense = DenseRetriever(model_name=args.dense_model)
        dense.index(evidence_records, mode=args.mode, checkpoint_dir=out)
        dense.save(out)

    print("\n✅  Evidence index built successfully.")
    print(f"    BM25 index: {Path(args.out_dir) / 'bm25'}")
    print(f"    Dense index: {Path(args.out_dir) / 'dense'}")
    print("\nNext step: python scripts/build_benchmark.py")


if __name__ == "__main__":
    main()
