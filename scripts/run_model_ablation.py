#!/usr/bin/env python
"""Run multi-model refusal and verification ablation across diverse LLMs for ICCIT 2026.

This script tests whether high parametric abstention rates (S1/S2) are specific
to gpt-4o-mini's safety alignment or if they persist across different model families
(Llama 3, Qwen 2.5, DeepSeek, Gemini, Mistral, etc.).

Presets available:
  - fast : Small/free models (Llama-3.1-8B, Gemma-2-9B, Mistral-7B, GPT-4o-mini)
  - full : Comprehensive suite (GPT-4o, Llama-3.3-70B, Qwen-2.5-72B, DeepSeek-V3, Gemini-Flash, etc.)
  - free : Zero-cost OpenRouter endpoints only

Usage:
    # Run fast preset on 30 balanced benchmark claims (15 authentic + 15 fabricated)
    python scripts/run_model_ablation.py --preset fast --sample-size 30

    # Run full multi-family suite on 50 balanced benchmark claims
    python scripts/run_model_ablation.py --preset full --sample-size 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hadith_misinfo.benchmark.sampler import load_benchmark
from hadith_misinfo.config import settings
from hadith_misinfo.llm.base import make_complete_fn
from hadith_misinfo.schemas import InferenceRecord, Language
from hadith_misinfo.verification.verifier import Verifier

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Model Presets across diverse LLM families
MODEL_PRESETS: dict[str, list[str]] = {
    "fast": [
        "openai/gpt-4o-mini",
        "meta-llama/llama-3.1-8b-instruct:free",
        "google/gemma-2-9b-it:free",
        "mistralai/mistral-7b-instruct:free",
    ],
    "free": [
        "meta-llama/llama-3.1-8b-instruct:free",
        "google/gemma-2-9b-it:free",
        "mistralai/mistral-7b-instruct:free",
        "qwen/qwen-2.5-7b-instruct:free",
    ],
    "full": [
        # OpenAI Family
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
        # Meta Llama Family
        "meta-llama/llama-3.1-8b-instruct:free",
        "meta-llama/llama-3.3-70b-instruct",
        # Google Family
        "google/gemini-flash-1.5",
        "google/gemma-2-9b-it:free",
        # Alibaba Qwen Family
        "qwen/qwen-2.5-72b-instruct",
        "qwen/qwen-2.5-7b-instruct:free",
        # DeepSeek Family
        "deepseek/deepseek-chat",
        # Mistral Family
        "mistralai/mistral-7b-instruct:free",
        "mistralai/mistral-large-2411",
    ],
}


def load_stratified_subset(benchmark_path: Path, sample_size: int) -> list[Any]:
    """Load benchmark records and return a balanced 50/50 authentic vs. fabricated sample."""
    full_benchmark = load_benchmark(benchmark_path)

    authentic = [r for r in full_benchmark if getattr(r, "label", "").lower() == "authentic"]
    fabricated = [r for r in full_benchmark if getattr(r, "label", "").lower() == "fabricated"]

    half = sample_size // 2
    subset_auth = authentic[:half]
    subset_fab = fabricated[:half]

    combined = subset_auth + subset_fab
    logger.info(
        f"    Stratified sample selected ({len(combined)} total): "
        f"{len(subset_auth)} authentic + {len(subset_fab)} fabricated claims."
    )
    return combined


def get_claim_text(record: Any, lang: str) -> str:
    """Extract claim text across object or dictionary representations."""
    if hasattr(record, f"claim_{lang}") and getattr(record, f"claim_{lang}"):
        return getattr(record, f"claim_{lang}")
    if hasattr(record, "claims") and isinstance(record.claims, dict):
        return record.claims.get(lang, "")
    if isinstance(record, dict):
        claims = record.get("claims", {})
        if isinstance(claims, dict):
            return claims.get(lang, "")
        return record.get(f"claim_{lang}", "")
    return ""


def compute_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute strict accuracy, selective accuracy, coverage, macro-F1, and abstention rate."""
    total = len(predictions)
    if total == 0:
        return {
            "strict_acc": 0.0,
            "sel_acc": 0.0,
            "macro_f1": 0.0,
            "coverage": 0.0,
            "abstain_rate": 0.0,
            "total": 0,
            "answered": 0,
            "abstained": 0,
            "errors": 0,
        }

    correct_strict = 0
    correct_selective = 0
    answered = 0
    abstained = 0
    errors = 0

    tp = {"SUPPORTED": 0, "NOT_SUPPORTED": 0, "INSUFFICIENT_EVIDENCE": 0}
    fp = {"SUPPORTED": 0, "NOT_SUPPORTED": 0, "INSUFFICIENT_EVIDENCE": 0}
    fn = {"SUPPORTED": 0, "NOT_SUPPORTED": 0, "INSUFFICIENT_EVIDENCE": 0}

    for item in predictions:
        gold_label = item.get("label", "").lower()
        verdict = item.get("verdict", "").upper()

        gold_target = "SUPPORTED" if gold_label == "authentic" else "NOT_SUPPORTED"

        if verdict in ("INSUFFICIENT_EVIDENCE", "ABSTAIN"):
            abstained += 1
            predicted_class = "INSUFFICIENT_EVIDENCE"
        elif verdict == "ERROR":
            errors += 1
            predicted_class = "INSUFFICIENT_EVIDENCE"
        else:
            answered += 1
            predicted_class = verdict
            if verdict == gold_target:
                correct_strict += 1
                correct_selective += 1

        for cls in ["SUPPORTED", "NOT_SUPPORTED", "INSUFFICIENT_EVIDENCE"]:
            if predicted_class == cls and gold_target == cls:
                tp[cls] += 1
            elif predicted_class == cls and gold_target != cls:
                fp[cls] += 1
            elif predicted_class != cls and gold_target == cls:
                fn[cls] += 1

    f1_scores = []
    for cls in ["SUPPORTED", "NOT_SUPPORTED"]:
        precision = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0.0
        recall = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    strict_acc = correct_strict / total
    coverage = answered / total
    sel_acc = (correct_selective / answered) if answered > 0 else 0.0
    abstain_rate = abstained / total

    return {
        "strict_acc": round(strict_acc, 4),
        "sel_acc": round(sel_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "coverage": round(coverage, 4),
        "abstain_rate": round(abstain_rate, 4),
        "total": total,
        "answered": answered,
        "abstained": abstained,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark-path", default=str(settings.benchmark_path), help="Path to benchmark_dataset_a.jsonl")
    parser.add_argument("--sample-size", type=int, default=30, help="Number of claims to evaluate per model (default: 30)")
    parser.add_argument(
        "--preset",
        choices=["fast", "full", "free", "custom"],
        default="fast",
        help="Select a predefined group of models across LLM families.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Explicit list of OpenRouter model identifiers (overrides --preset if provided).",
    )
    parser.add_argument("--out-dir", default="results/ablation", help="Directory to save ablation output artifacts.")
    args = parser.parse_args()

    if args.models:
        target_models = args.models
    else:
        target_models = MODEL_PRESETS.get(args.preset, MODEL_PRESETS["fast"])

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Stratified Benchmark Subset
    logger.info(f"Loading stratified benchmark dataset from: {args.benchmark_path}")
    eval_subset = load_stratified_subset(Path(args.benchmark_path), args.sample_size)

    summary_rows = []

    # 2. Iterate Models & Run Parametric Systems (S1: English, S2: Bangla)
    for model_index, model_name in enumerate(target_models, 1):
        clean_model_tag = model_name.split("/")[-1].replace(":", "_").replace(".", "_")
        logger.info(f"\n[{model_index}/{len(target_models)}] 🤖 Target Model: {model_name}")

        try:
            complete_fn = make_complete_fn(
                provider=settings.llm_provider,
                model=model_name,
                temperature=0.0,
                retry=True,
            )
            verifier = Verifier(complete=complete_fn)
        except Exception as exc:
            logger.error(f"Failed to initialize verifier for {model_name}: {exc}")
            continue

        for lang_code, system_tag in [("en", "S1"), ("bn", "S2")]:
            logger.info(f"   ► System [{system_tag}] ({lang_code.upper()} Parametric)...")
            preds = []

            for record in eval_subset:
                claim_text = get_claim_text(record, lang_code)
                claim_id = getattr(record, "claim_id", record.get("claim_id", "C0000")) if isinstance(record, dict) else record.claim_id
                gold_label = getattr(record, "label", record.get("label", "authentic")) if isinstance(record, dict) else record.label

                inf_record = InferenceRecord(
                    claim_id=claim_id,
                    claim_text=claim_text,
                    language=lang_code,  # type: ignore[arg-type]
                )

                try:
                    res = verifier.verify(record=inf_record, system=system_tag)  # type: ignore[arg-type]
                    verdict = res.verdict
                    explanation = res.explanation
                except Exception as exc:
                    logger.warning(f"      ⚠️ Request failed for claim {claim_id} on {model_name}: {exc}")
                    verdict = "ERROR"
                    explanation = f"Execution Exception: {exc}"

                preds.append({
                    "claim_id": claim_id,
                    "label": gold_label,
                    "lang": lang_code,
                    "verdict": verdict,
                    "explanation": explanation,
                })

            # Stream predictions to JSONL file
            save_file = out_path / f"ablation_{clean_model_tag}_{system_tag}.jsonl"
            with open(save_file, "w", encoding="utf-8") as f:
                for p in preds:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")

            metrics = compute_metrics(preds)
            metrics.update({
                "model": model_name,
                "system": system_tag,
                "language": lang_code,
            })
            summary_rows.append(metrics)

    # 3. Render Metric Summary Table
    print("\n" + "=" * 102)
    print("📊 MULTI-MODEL REFUSAL & ACCURACY ABLATION SUMMARY (ICCIT 2026)")
    print("=" * 102)
    header = f"{'Model Identifier':<38} | {'Sys':<4} | {'Strict Acc':<10} | {'Sel Acc':<8} | {'Macro-F1':<9} | {'Abstain %':<10} | {'Errors':<6}"
    print(header)
    print("-" * 102)

    for r in summary_rows:
        row_str = (
            f"{r['model']:<38} | "
            f"{r['system']:<4} | "
            f"{r['strict_acc']:<10.3f} | "
            f"{r['sel_acc']:<8.3f} | "
            f"{r['macro_f1']:<9.3f} | "
            f"{r['abstain_rate']*100:<9.1f}% | "
            f"{r['errors']:<6}"
        )
        print(row_str)

    print("-" * 102)

    # 4. Save Final Summary JSON & CSV
    summary_json = out_path / "model_ablation_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    summary_csv = out_path / "model_ablation_summary.csv"
    with open(summary_csv, "w", encoding="utf-8") as f:
        f.write("model,system,language,strict_acc,sel_acc,macro_f1,coverage,abstain_rate,errors,total\n")
        for r in summary_rows:
            f.write(
                f"{r['model']},{r['system']},{r['language']},{r['strict_acc']},{r['sel_acc']},"
                f"{r['macro_f1']},{r['coverage']},{r['abstain_rate']},{r['errors']},{r['total']}\n"
            )

    logger.info(f"\n✅ Ablation run complete. Summary exported to:\n - {summary_json}\n - {summary_csv}\n")


if __name__ == "__main__":
    main()