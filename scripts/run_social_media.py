#!/usr/bin/env python
"""Run the full pipeline on Al-Zaman/Noman social-media posts (Dataset C / RQ3).

This is the out-of-distribution validation layer.  The pipeline runs:
  1. Claim extraction (Layer 1): extract clean Hadith attribution from noisy post
  2. Verification (Layer 2): S4 (Bangla cross-lingual RAG) by default
  3. Mitigation (Layer 3): format evidence-grounded intervention

Results are saved to results/dataset_c_validation.jsonl.

Usage:
    python scripts/run_social_media.py

    # Custom sample
    python scripts/run_social_media.py --n-posts 50 --system S4

    # Use a different LLM for extraction
    python scripts/run_social_media.py --provider anthropic --model claude-3-haiku-20240307
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hadith_misinfo.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alzaman-dir", default=str(settings.alzaman_raw_dir))
    parser.add_argument("--n-posts", type=int, default=settings.dataset_c_sample_size)
    parser.add_argument("--seed", type=int, default=settings.benchmark_seed)
    parser.add_argument("--system", choices=["S3", "S4"], default="S4",
                        help="Verification system to use. Default: S4 (Bangla cross-lingual RAG).")
    parser.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default="dense")
    parser.add_argument("--k", type=int, default=settings.default_k)
    parser.add_argument("--provider", default=settings.llm_provider,
                        choices=["openai", "anthropic", "ollama"])
    parser.add_argument("--model", default=settings.llm_model)
    parser.add_argument("--out", default=str(settings.results_dir / "dataset_c_validation.jsonl"))
    args = parser.parse_args()

    settings.ensure_dirs()

    # ── Sample posts ──────────────────────────────────────────────────────────
    print(f"\n📱  Sampling {args.n_posts} posts from Al-Zaman dataset ({args.alzaman_dir})...")
    from hadith_misinfo.ingestion.al_zaman import sample_posts
    posts = sample_posts(args.alzaman_dir, n=args.n_posts, seed=args.seed, require_bangla=True)
    print(f"    Sampled {len(posts)} posts.")

    # ── LLM setup ─────────────────────────────────────────────────────────────
    from hadith_misinfo.llm.base import make_complete_fn
    complete = make_complete_fn(provider=args.provider, model=args.model)

    # ── Retriever + evidence store ─────────────────────────────────────────────
    print(f"\n🔍  Loading {args.retriever} retriever...")
    from hadith_misinfo.evidence.store import EvidenceStore
    evidence_store = EvidenceStore.load(settings.evidence_path)

    if args.retriever == "bm25":
        from hadith_misinfo.retrieval.bm25 import BM25Retriever
        retriever = BM25Retriever.load(settings.bm25_index_dir)
    elif args.retriever == "dense":
        from hadith_misinfo.retrieval.dense import DenseRetriever
        retriever = DenseRetriever.load(settings.dense_index_dir)
    else:
        from hadith_misinfo.retrieval.bm25 import BM25Retriever
        from hadith_misinfo.retrieval.dense import DenseRetriever
        from hadith_misinfo.retrieval.hybrid import HybridRetriever
        retriever = HybridRetriever(
            bm25=BM25Retriever.load(settings.bm25_index_dir),
            dense=DenseRetriever.load(settings.dense_index_dir),
        )

    # ── Setup pipeline components ──────────────────────────────────────────────
    from hadith_misinfo.extraction.extractor import ClaimExtractor
    from hadith_misinfo.mitigation.responder import build_intervention
    from hadith_misinfo.schemas import InferenceRecord
    from hadith_misinfo.verification.verifier import Verifier

    extractor = ClaimExtractor(complete=complete)
    verifier = Verifier(
        complete=complete,
        evidence_store=evidence_store,
        retriever=retriever,
        default_k=args.k,
    )

    # ── Run pipeline ──────────────────────────────────────────────────────────
    print(f"\n🚀  Running full pipeline ({args.system}, {args.retriever}) on {len(posts)} posts...")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm

    n_hadith = 0
    n_skipped = 0

    with out_path.open("w", encoding="utf-8") as f:
        for post in tqdm(posts, desc="Processing posts"):
            # Layer 1: Extract claim
            extracted = extractor.extract(post.post_id, post.raw_text)

            if extracted.claim_type == "none" or not extracted.claim_text:
                n_skipped += 1
                record = {
                    "post_id": post.post_id,
                    "raw_text": post.raw_text[:300],
                    "extracted_claim": None,
                    "claim_type": "none",
                    "verdict": None,
                    "intervention": None,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            n_hadith += 1

            # Layer 2: Verify
            inference = InferenceRecord(
                claim_id=post.post_id,
                language=extracted.language,
                claim_text=extracted.claim_text,
            )
            verification = verifier.verify(inference, system=args.system, k=args.k)

            # Layer 3: Mitigate
            retrieved_evidence = evidence_store.get_many(verification.retrieved_evidence_ids)
            mitigation = build_intervention(
                claim_id=post.post_id,
                verdict=verification.verdict,
                explanation=verification.explanation,
                retrieved_evidence=retrieved_evidence,
                claim_text=extracted.claim_text,
                language=extracted.language,
            )

            record = {
                "post_id": post.post_id,
                "raw_text": post.raw_text[:500],
                "extracted_claim": extracted.to_dict(),
                "verification": verification.to_dict(),
                "intervention": mitigation.intervention_text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n✅  Dataset C pipeline complete.")
    print(f"    Posts processed: {len(posts)}")
    print(f"    Hadith claims found: {n_hadith}")
    print(f"    Non-Hadith posts skipped: {n_skipped}")
    print(f"    Results saved to: {args.out}")


if __name__ == "__main__":
    main()
