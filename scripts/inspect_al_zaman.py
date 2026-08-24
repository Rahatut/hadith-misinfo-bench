#!/usr/bin/env python
"""Phase 1 sanity check: print the ACTUAL schema and sample posts of the Al-Zaman/Noman dataset.

Usage:
    python scripts/inspect_al_zaman.py
    python scripts/inspect_al_zaman.py --raw-dir data/raw/al-zaman/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--raw-dir",
        default="data/raw/al-zaman",
        help="Directory containing the Al-Zaman/Noman CSV or XLSX file(s). Default: data/raw/al-zaman",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    files = sorted(raw_dir.glob("*.csv")) + sorted(raw_dir.glob("*.xlsx"))

    if not files:
        print(f"❌  No .csv or .xlsx files found under {raw_dir}.")
        print("Download the Al-Zaman/Noman Mendeley dataset and place it there.")
        print("https://data.mendeley.com/datasets/5ykks3psks/5")
        return

    print(f"Found {len(files)} file(s) in {raw_dir}:\n")

    for path in files:
        print("=" * 80)
        print(f"File: {path}")
        df = pd.read_excel(path) if path.suffix == ".xlsx" else pd.read_csv(path, encoding="utf-8", errors="replace")
        print(f"Shape: {df.shape}  ({df.shape[0]:,} rows × {df.shape[1]} columns)")
        print(f"Columns: {list(df.columns)}")
        print(f"\nDtypes:\n{df.dtypes}")
        print("\nFirst 3 rows:")
        with pd.option_context("display.max_colwidth", 100):
            print(df.head(3).to_string())
        print()

    print("=" * 80)
    print("✅ Inspect complete.\n")


if __name__ == "__main__":
    main()
