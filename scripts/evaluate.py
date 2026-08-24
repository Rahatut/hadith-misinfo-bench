#!/usr/bin/env python
"""Evaluate all experiment results and print comparison tables.

Reads results JSONL files from the results/ directory and produces:
  - Table 2: Main verification results (S1–S4)
  - Table 3: Δ_BN cross-lingual degradation
  - CSV export for inclusion in paper

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --results-dir results/ --format markdown
    python scripts/evaluate.py --export-dir results/tables/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hadith_misinfo.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(settings.results_dir))
    parser.add_argument("--benchmark", default=str(settings.benchmark_path))
    parser.add_argument("--export-dir", default=None,
                        help="Directory to export metrics CSV + JSON.")
    parser.add_argument("--format", choices=["markdown", "plain"], default="markdown")
    args = parser.parse_args()

    from hadith_misinfo.evaluation.report import generate_report
    generate_report(
        results_dir=args.results_dir,
        benchmark_path=args.benchmark,
        output_dir=args.export_dir,
        print_tables=True,
    )


if __name__ == "__main__":
    main()
