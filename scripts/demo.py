#!/usr/bin/env python
"""Interactive demo of the full 3-layer pipeline.

Paste any social-media post (or a plain Hadith claim) and see:
  Layer 1 — Claim extraction
  Layer 2 — Verification (S4: Bangla cross-lingual RAG)
  Layer 3 — Evidence-grounded mitigation intervention

Usage:
    python scripts/demo.py
    python scripts/demo.py --system S3 --retriever bm25
    python scripts/demo.py --claim "The Prophet said whoever reads X on Friday will have all sins forgiven."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hadith_misinfo.config import settings

_EXAMPLE_POSTS = [
    # Bangla social-media style
    "ভাই সবাই শেয়ার করুন!!!\nরাসূল (সাঃ) বলেছেন যে যে ব্যক্তি শুক্রবার ১০০ বার দরুদ পড়বে তার সব গুনাহ মাফ হয়ে যাবে।\nআলহামদুলিল্লাহ ❤️\nআপনার পরিচিত সবাইকে পাঠান...",
    # English social-media style
    "SubhanAllah! The Prophet (PBUH) said that whoever prays Fajr in congregation will be under the protection of Allah for the entire day. Please share with everyone!",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--claim", default=None, help="Direct claim text (skips extraction).")
    parser.add_argument("--system", choices=["S3", "S4"], default="S4")
    parser.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default="dense")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--provider", default=settings.llm_provider,
                        choices=["openai", "anthropic", "ollama"])
    parser.add_argument("--model", default=settings.llm_model)
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────────
    from hadith_misinfo.llm.base import make_complete_fn
    complete = make_complete_fn(provider=args.provider, model=args.model)

    from hadith_misinfo.evidence.store import EvidenceStore
    print("Loading evidence store...")
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

    # ── Interactive loop ──────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  HadithMisinfoBench — Interactive Demo")
    print(f"  System: {args.system}  |  Retriever: {args.retriever}  |  k={args.k}")
    print("═" * 70)

    if args.claim:
        post_text = args.claim
        post_id = "demo_claim"
    else:
        print("\nPaste a social-media post (or press Enter for an example).")
        print("Type 'exit' to quit.\n")
        try:
            post_text = input("Post text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return
        if not post_text:
            post_text = _EXAMPLE_POSTS[0]
            print(f"\n[Using example post]\n{post_text}\n")
        if post_text.lower() == "exit":
            return
        post_id = "demo"

    print("\n" + "─" * 70)
    print("▶ LAYER 1: CLAIM EXTRACTION")
    print("─" * 70)

    extracted = extractor.extract(post_id, post_text)
    print(f"Claim type:  {extracted.claim_type}")
    print(f"Language:    {extracted.language}")
    print(f"Confidence:  {extracted.confidence:.2f}")
    print(f"Claim text:  {extracted.claim_text or '(none — not a Hadith claim)'}")

    if not extracted.claim_text or extracted.claim_type == "none":
        print("\n⚠️  No Hadith attribution detected. Stopping pipeline.")
        return

    language = extracted.language if args.system == "S4" else "en"
    inference = InferenceRecord(
        claim_id=post_id,
        language=language,
        claim_text=extracted.claim_text,
    )

    print("\n" + "─" * 70)
    print("▶ LAYER 2: VERIFICATION")
    print("─" * 70)

    result = verifier.verify(inference, system=args.system, k=args.k)
    print(f"Verdict:     {result.verdict}")
    print(f"Explanation: {result.explanation}")
    if result.retrieved_evidence_ids:
        print(f"Retrieved:   {result.retrieved_evidence_ids}")

    print("\n" + "─" * 70)
    print("▶ LAYER 3: MITIGATION INTERVENTION")
    print("─" * 70)

    retrieved_evidence = evidence_store.get_many(result.retrieved_evidence_ids)
    mitigation = build_intervention(
        claim_id=post_id,
        verdict=result.verdict,
        explanation=result.explanation,
        retrieved_evidence=retrieved_evidence,
        claim_text=extracted.claim_text,
        language=language,
    )
    print(mitigation.intervention_text)


if __name__ == "__main__":
    main()
