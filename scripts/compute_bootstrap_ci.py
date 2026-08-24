#!/usr/bin/env python
"""Compute 95% bootstrap confidence intervals across ALL prediction JSONL files.

Scans results/ and results/ablation/ for prediction files, joins them with gold ground-truth
labels from benchmark_dataset_a.jsonl, and performs 1,000 bootstrap iterations to calculate
Strict Accuracy, Selective Accuracy, Coverage, and Abstention Rate with 95% CIs.

Exports summary artifacts to:
  - results/tables/bootstrap_ci_summary.json
  - results/tables/bootstrap_ci_summary.csv
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK_PATH = Path("data/processed/benchmark_dataset_a.jsonl")
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/tables")


def load_ground_truth_labels(benchmark_path: Path) -> dict[str, str]:
    """Load gold ground-truth labels ('authentic'/'fabricated') from Dataset A."""
    labels = {}
    with open(benchmark_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                labels[item["claim_id"]] = item["label"].lower()
    return labels


def load_predictions(file_path: Path) -> dict[str, dict]:
    """Load prediction records mapped by claim_id."""
    preds = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    if "claim_id" in item and "verdict" in item:
                        preds[item["claim_id"]] = item
                except json.JSONDecodeError:
                    continue
    return preds


def compute_metrics_for_subset(
    preds_dict: dict[str, dict], gold_labels: dict[str, str], claim_ids: list[str]
) -> dict[str, float]:
    """Compute verification metrics for a given subset of claim IDs."""
    total = len(claim_ids)
    if total == 0:
        return {"strict_acc": 0.0, "coverage": 0.0, "sel_acc": 0.0, "abstain_rate": 0.0}

    correct_strict = 0
    correct_sel = 0
    answered = 0
    abstained = 0

    for cid in claim_ids:
        if cid not in preds_dict or cid not in gold_labels:
            abstained += 1
            continue

        item = preds_dict[cid]
        gold_label = gold_labels[cid]
        verdict = str(item.get("verdict", "")).upper()

        gold_target = "SUPPORTED" if gold_label == "authentic" else "NOT_SUPPORTED"

        if verdict in ("INSUFFICIENT_EVIDENCE", "ABSTAIN", "ERROR", ""):
            abstained += 1
        else:
            answered += 1
            if verdict == gold_target:
                correct_strict += 1
                correct_sel += 1

    strict_acc = correct_strict / total
    coverage = answered / total
    sel_acc = (correct_sel / answered) if answered > 0 else 0.0
    abstain_rate = abstained / total

    return {
        "strict_acc": strict_acc,
        "coverage": coverage,
        "sel_acc": sel_acc,
        "abstain_rate": abstain_rate,
    }


def run_bootstrap_for_file(
    file_path: Path,
    gold_labels: dict[str, str],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any] | None:
    """Run bootstrap resampling for a single prediction JSONL file."""
    preds = load_predictions(file_path)
    if not preds:
        return None

    common_cids = sorted([cid for cid in preds.keys() if cid in gold_labels])
    n_claims = len(common_cids)
    if n_claims == 0:
        return None

    point_metrics = compute_metrics_for_subset(preds, gold_labels, common_cids)

    rng = np.random.default_rng(seed)
    cids_array = np.array(common_cids)

    strict_accs, coverages, sel_accs, abstains = [], [], [], []

    for _ in range(n_boot):
        boot_cids = rng.choice(cids_array, size=n_claims, replace=True)
        m = compute_metrics_for_subset(preds, gold_labels, boot_cids)
        strict_accs.append(m["strict_acc"])
        coverages.append(m["coverage"])
        sel_accs.append(m["sel_acc"])
        abstains.append(m["abstain_rate"])

    acc_lo, acc_hi = np.percentile(strict_accs, 2.5), np.percentile(strict_accs, 97.5)
    cov_lo, cov_hi = np.percentile(coverages, 2.5), np.percentile(coverages, 97.5)
    sel_lo, sel_hi = np.percentile(sel_accs, 2.5), np.percentile(sel_accs, 97.5)
    abs_lo, abs_hi = np.percentile(abstains, 2.5), np.percentile(abstains, 97.5)

    rel_path = file_path.relative_to(RESULTS_DIR) if file_path.is_relative_to(RESULTS_DIR) else file_path

    return {
        "tag": rel_path.stem,
        "file": str(rel_path),
        "n_claims": n_claims,
        "strict_acc": round(point_metrics["strict_acc"], 4),
        "strict_acc_ci": [round(acc_lo, 4), round(acc_hi, 4)],
        "coverage": round(point_metrics["coverage"], 4),
        "coverage_ci": [round(cov_lo, 4), round(cov_hi, 4)],
        "sel_acc": round(point_metrics["sel_acc"], 4),
        "sel_acc_ci": [round(sel_lo, 4), round(sel_hi, 4)],
        "abstain_rate": round(point_metrics["abstain_rate"], 4),
        "abstain_rate_ci": [round(abs_lo, 4), round(abs_hi, 4)],
    }


def main():
    print("\n" + "=" * 110)
    print("📊 PAIRED 95% BOOTSTRAP CONFIDENCE INTERVALS FOR ALL PREDICTION LOGS (1,000 ITERATIONS)")
    print("=" * 110)

    if not BENCHMARK_PATH.exists():
        logger.error(f"Ground truth benchmark dataset not found at {BENCHMARK_PATH}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gold_labels = load_ground_truth_labels(BENCHMARK_PATH)
    logger.info(f"Loaded ground-truth labels for {len(gold_labels)} claims from benchmark dataset.")

    # Find all JSONL prediction files under results/
    jsonl_files = sorted(list(RESULTS_DIR.rglob("*.jsonl")))
    logger.info(f"Discovered {len(jsonl_files)} JSONL files under {RESULTS_DIR}/")

    all_results = []
    for f in jsonl_files:
        res = run_bootstrap_for_file(f, gold_labels)
        if res:
            all_results.append(res)

    if not all_results:
        logger.error("No valid prediction files processed.")
        return

    # Render Summary Table
    header = f"{'Result Artifact':<35} | {'N':<4} | {'Strict Acc (Point)':<18} | {'Strict Acc (95% CI)':<20} | {'Abstain % (95% CI)':<20}"
    print("\n" + header)
    print("-" * 110)

    for r in all_results:
        acc_ci_str = f"[{r['strict_acc_ci'][0]:.3f}, {r['strict_acc_ci'][1]:.3f}]"
        abs_ci_str = f"[{r['abstain_rate_ci'][0] * 100:.1f}%, {r['abstain_rate_ci'][1] * 100:.1f}%]"
        print(f"{r['tag']:<35} | {r['n_claims']:<4} | {r['strict_acc']:<18.4f} | {acc_ci_str:<20} | {abs_ci_str:<20}")

    print("-" * 110)

    # Save JSON and CSV output artifacts
    json_out = OUTPUT_DIR / "bootstrap_ci_summary.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    csv_out = OUTPUT_DIR / "bootstrap_ci_summary.csv"
    with open(csv_out, "w", encoding="utf-8") as f:
        f.write("tag,file,n_claims,strict_acc,strict_acc_ci_lo,strict_acc_ci_hi,sel_acc,sel_acc_ci_lo,sel_acc_ci_hi,coverage,coverage_ci_lo,coverage_ci_hi,abstain_rate,abstain_rate_ci_lo,abstain_rate_ci_hi\n")
        for r in all_results:
            f.write(
                f"{r['tag']},{r['file']},{r['n_claims']},{r['strict_acc']},{r['strict_acc_ci'][0]},{r['strict_acc_ci'][1]},"
                f"{r['sel_acc']},{r['sel_acc_ci'][0]},{r['sel_acc_ci'][1]},{r['coverage']},{r['coverage_ci'][0]},{r['coverage_ci'][1]},"
                f"{r['abstain_rate']},{r['abstain_rate_ci'][0]},{r['abstain_rate_ci'][1]}\n"
            )

    print(f"\n✅ All bootstrap CIs processed and saved to:\n - {json_out}\n - {csv_out}\n")


if __name__ == "__main__":
    main()