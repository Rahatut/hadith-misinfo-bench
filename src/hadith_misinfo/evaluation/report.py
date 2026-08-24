"""Evaluation report generator.

Reads all results JSONL files from the results/ directory, computes
metrics for each system, and produces:
  - A Markdown comparison table (Table 2 / Table 3 from the paper)
  - A CSV export
  - Δ_BN cross-lingual degradation figures
  - Ablation comparisons (BM25 vs Dense, parametric vs RAG)

Usage
-----
>>> from hadith_misinfo.evaluation.report import generate_report
>>> generate_report(results_dir="results/", benchmark_path="data/processed/benchmark_dataset_a.jsonl")
"""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path

from hadith_misinfo.benchmark.sampler import load_benchmark
from hadith_misinfo.evaluation.verification import (
    VerificationMetrics,
    compute_delta_bn,
    compute_verification_metrics,
    results_to_predictions,
)
from hadith_misinfo.schemas import VerificationResult
from hadith_misinfo.verification.verifier import load_results

# System display ordering for tables
_SYSTEM_DESCRIPTIONS = {
    "s1": "English, Parametric",
    "s2": "Bangla,  Parametric",
    "s3": "English, RAG",
    "s3_bm25": "English, RAG (BM25)",
    "s3_dense": "English, RAG (Dense)",
    "s4": "Bangla,  RAG (cross-lingual)",
    "s4_bm25": "Bangla,  RAG (Direct BM25)",
    "s4_trans": "Bangla,  RAG (Translate-BM25)",
    "s4_dense": "Bangla,  RAG (Dense)",
}


def generate_report(
    results_dir: str | Path,
    benchmark_path: str | Path,
    output_dir: str | Path | None = None,
    print_tables: bool = True,
) -> dict[str, VerificationMetrics]:
    """Load all results, compute metrics, print tables, save CSV.

    Returns a dict of file_stem → VerificationMetrics.
    """
    results_dir = Path(results_dir)
    benchmark_path = Path(benchmark_path)

    # ── Load gold labels ──────────────────────────────────────────────────────
    benchmark = load_benchmark(benchmark_path)
    gold_labels = {rec.claim_id: rec.label for rec in benchmark}

    # ── Load results files ────────────────────────────────────────────────────
    system_metrics: dict[str, VerificationMetrics] = {}

    result_files = sorted(results_dir.glob("*.jsonl"))
    for file_path in result_files:
        stem = file_path.stem.lower()
        results = load_results(file_path)
        if not results:
            continue

        predictions = results_to_predictions(results)
        benchmark_ids = set(gold_labels.keys())
        predictions = {cid: pred for cid, pred in predictions.items() if cid in benchmark_ids}
        relevant_gold = {cid: gold_labels[cid] for cid in predictions}

        language = "en" if "s1" in stem or "s3" in stem else "bn"
        metrics = compute_verification_metrics(
            relevant_gold, predictions, system=stem.upper(), language=language
        )
        system_metrics[stem] = metrics

    if print_tables:
        _print_verification_table(system_metrics)
        _print_delta_bn(system_metrics)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_csv(system_metrics, output_dir / "metrics.csv")
        _save_json(system_metrics, output_dir / "metrics.json")
        print(f"\nMetrics saved to {output_dir}/")

    return system_metrics


def _print_verification_table(system_metrics: dict[str, VerificationMetrics]) -> None:
    """Print the main verification results table in Markdown format."""
    print("\n## Verification Results (Table 2)\n")
    header = (
        f"| {'System':<10} | {'Description':<30} | "
        f"{'Strict Acc':>10} | {'Sel. Acc':>8} | "
        f"{'Macro-F1':>8} | {'Coverage':>8} | {'Abstain%':>8} |"
    )
    sep = "|" + "|".join(["-" * (len(h) + 2) for h in header.split("|")[1:-1]]) + "|"
    print(header)
    print(sep)

    for stem, m in sorted(system_metrics.items()):
        desc = _SYSTEM_DESCRIPTIONS.get(stem, stem)
        print(
            f"| {stem.upper():<10} | {desc:<30} | "
            f"{m.strict_accuracy:>10.3f} | {m.selective_accuracy:>8.3f} | "
            f"{m.macro_f1:>8.3f} | {m.coverage:>8.3f} | {m.abstention_rate * 100:>7.1f}% |"
        )


def _print_delta_bn(system_metrics: dict[str, VerificationMetrics]) -> None:
    """Print Δ_BN cross-lingual degradation."""
    print("\n## Cross-Lingual Degradation (Δ_BN)\n")
    pairs = [
        ("Parametric", "s1", "s2"),
        ("RAG (Direct BM25 Collapse)", "s3_bm25", "s4_bm25"),
        ("RAG (Translate-BM25 Recovery)", "s3_bm25", "s4_trans"),
    ]
    for label, en_sys, bn_sys in pairs:
        en_m = system_metrics.get(en_sys) or system_metrics.get(en_sys.split("_")[0])
        bn_m = system_metrics.get(bn_sys) or system_metrics.get(bn_sys.split("_")[0])
        if en_m is None or bn_m is None:
            continue
        delta = compute_delta_bn(en_m, bn_m)
        print(f"### {label} ({en_sys.upper()} → {bn_sys.upper()})")
        for key, val in delta.items():
            arrow = "↓" if val > 0 else "↑"
            print(f"  {key}: {val:+.3f} {arrow}")
        print()


def _save_csv(system_metrics: dict[str, VerificationMetrics], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "system", "language", "n", "strict_accuracy", "selective_accuracy",
            "macro_f1", "coverage", "abstention_rate",
            "f1_authentic", "f1_fabricated",
        ])
        for m in system_metrics.values():
            writer.writerow([
                m.system, m.language, m.n,
                f"{m.strict_accuracy:.4f}", f"{m.selective_accuracy:.4f}",
                f"{m.macro_f1:.4f}", f"{m.coverage:.4f}", f"{m.abstention_rate:.4f}",
                f"{m.per_class_f1.get('authentic', 0):.4f}",
                f"{m.per_class_f1.get('fabricated', 0):.4f}",
            ])


def _save_json(system_metrics: dict[str, VerificationMetrics], path: Path) -> None:
    data = {}
    for system, m in system_metrics.items():
        data[system] = {
            "system": m.system,
            "language": m.language,
            "n": m.n,
            "strict_accuracy": m.strict_accuracy,
            "selective_accuracy": m.selective_accuracy,
            "macro_f1": m.macro_f1,
            "coverage": m.coverage,
            "abstention_rate": m.abstention_rate,
            "per_class_f1": m.per_class_f1,
            "verdict_counts": m.verdict_counts,
        }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
