#!/usr/bin/env python
"""Phase 1 sanity check: print the ACTUAL schema of the raw MAHADDAT files.

Run this BEFORE writing any pipeline code that depends on MAHADDAT column names.
Several published statistics (e.g. 26,561 total records, 80/10/10 split) are
self-reported by the dataset authors and should be verified against the actual files.

Usage:
    python scripts/inspect_mahaddat.py
    python scripts/inspect_mahaddat.py --raw-dir path/to/mahaddat/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--raw-dir",
        default="data/raw/mahaddat",
        help="Directory containing the raw MAHADDAT CSV file(s). Default: data/raw/mahaddat",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    csv_files = sorted(raw_dir.glob("*.csv"))

    if not csv_files:
        print(f"❌  No .csv files found under {raw_dir}.")
        print("Download/extract the MAHADDAT dataset there first, then re-run.")
        print("\nMAHADDAT is available from the IEEE Access supplementary materials:")
        print("  Gaanoun, K. & Alsuhaibani, M. (2022). IEEE Access.")
        print("  https://ieeexplore.ieee.org/document/9931123/")
        return

    print(f"Found {len(csv_files)} CSV file(s) in {raw_dir}:\n")

    for path in csv_files:
        print("=" * 80)
        print(f"File: {path}")
        df = pd.read_csv(path)
        print(f"Shape: {df.shape}  ({df.shape[0]:,} rows × {df.shape[1]} columns)")
        print(f"Columns: {list(df.columns)}")
        print(f"\nDtypes:\n{df.dtypes}")
        print(f"\nFirst 3 rows:")
        with pd.option_context("display.max_colwidth", 80):
            print(df.head(3).to_string())

        # Label column analysis
        candidate_label_cols = [
            c for c in df.columns if c.lower() in {"label", "class", "y", "target", "fabricated"}
        ]
        for col in candidate_label_cols:
            print(f"\n📊 Value counts for candidate label column '{col}':")
            print(df[col].value_counts(dropna=False).to_string())

        # Split column analysis
        candidate_split_cols = [
            c for c in df.columns if c.lower() in {"split", "set", "partition", "fold"}
        ]
        for col in candidate_split_cols:
            print(f"\n📊 Value counts for candidate split column '{col}':")
            print(df[col].value_counts(dropna=False).to_string())

        # Arabic text sample
        arabic_cols = [
            c for c in df.columns
            if any(kw in c.lower() for kw in ["arabic", "matn", "text", "hadith"])
        ]
        for col in arabic_cols[:1]:
            print(f"\n📝 Sample Arabic text from column '{col}':")
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else "(empty)"
            print(f"  {str(sample)[:300]}")

        print()

    print("=" * 80)
    print("\n✅ Next step:")
    print("  Update COLUMN_MAP and LABEL_MAP in src/hadith_misinfo/ingestion/mahaddat.py")
    print("  to match the printed column names above.\n")


if __name__ == "__main__":
    main()
