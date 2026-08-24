#!/usr/bin/env python
"""Run the complete experimental grid (S1, S2, S3_BM25, S4_BM25, S4_Trans) on the full benchmark.

Streams results to results/*.jsonl and generates final evaluation tables.

Usage:
    python scripts/run_all_experiments.py --concurrency 16
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent workers.")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    experiments = [
        ("S1", ["--system", "S1", "--out", "results/s1.jsonl"]),
        ("S2", ["--system", "S2", "--out", "results/s2.jsonl"]),
        ("S3_BM25", ["--system", "S3", "--retriever", "bm25", "--out", "results/s3_bm25.jsonl"]),
        ("S4_BM25", ["--system", "S4", "--retriever", "bm25", "--out", "results/s4_bm25.jsonl"]),
        ("S4_Trans", ["--system", "S4", "--retriever", "bm25", "--translate-query", "--out", "results/s4_trans.jsonl"]),
    ]

    for name, cmd_args in experiments:
        print(f"\n=======================================================")
        print(f"🚀  Running Experiment: {name}")
        print(f"=======================================================")
        cmd = [
            sys.executable,
            "scripts/run_experiment.py",
            "--provider", args.provider,
            "--model", args.model,
            "--concurrency", str(args.concurrency),
        ] + cmd_args

        t0 = time.time()
        res = subprocess.run(cmd, check=True)
        t1 = time.time()
        print(f"⏱️  Finished {name} in {t1 - t0:.1f}s.")

    print(f"\n=======================================================")
    print(f"📊  Generating Final Evaluation Report")
    print(f"=======================================================")
    subprocess.run([
        sys.executable,
        "scripts/evaluate.py",
        "--results-dir", "results/",
        "--export-dir", "results/tables/",
    ], check=True)


if __name__ == "__main__":
    main()
