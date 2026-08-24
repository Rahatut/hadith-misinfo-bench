#!/usr/bin/env python
"""Phase 1 sanity check: print the ACTUAL schema of bukhari.json / muslim.json.

Run this BEFORE trusting any field-name assumptions in the ingestion code.

Usage:
    python scripts/inspect_hadith_json.py
    python scripts/inspect_hadith_json.py --raw-dir path/to/hadith-json/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--raw-dir",
        default="data/raw/hadith-json",
        help="Directory containing (nested) hadith JSON files. Default: data/raw/hadith-json",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    json_files = sorted(raw_dir.rglob("*.json"))

    # Target stems
    target_stems = {
        "bukhari", "muslim",
        "eng-bukhari", "eng-muslim",
        "ara-bukharibukhari", "ara-muslimmuslim",
    }
    keep = [(f, f.stem.lower()) for f in json_files if f.stem.lower() in target_stems]

    if not keep:
        print(f"❌  No recognised Hadith JSON files found under {raw_dir}.")
        print(f"    Expected file stems: {sorted(target_stems)}")
        print("\n    Clone the hadith-json / hadith-api repo there first:")
        print("    git clone https://github.com/fawazahmed0/hadith-api data/raw/hadith-json")
        if json_files:
            print(f"\n    Found {len(json_files)} other .json file(s) — showing first 20:")
            for f in json_files[:20]:
                print(f"      {f.relative_to(raw_dir)}")
        return

    print(f"Found {len(keep)} recognised Hadith JSON file(s):\n")

    for path, stem in keep:
        print("=" * 80)
        print(f"File: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            print(f"Top-level type: dict, keys = {list(data.keys())[:10]}")
        elif isinstance(data, list):
            print(f"Top-level type: list, length = {len(data)}")

        # Navigate to hadith list
        hadiths = None
        if isinstance(data, dict):
            if "hadiths" in data:
                hadiths = data["hadiths"]
                print(f"Schema: {{ 'hadiths': [...] }}   (legacy 9-books format)")
            elif "data" in data and isinstance(data["data"], dict):
                hadiths = list(data["data"].values())
                print(f"Schema: {{ 'data': {{ '1': {{...}}, '2': {{...}}, ... }} }}   (hadith-api editions format)")
        elif isinstance(data, list):
            hadiths = data
            print(f"Schema: flat list")

        if hadiths is None:
            print("⚠️  Unrecognised schema — update ingestion/hadith_json.py manually.")
            continue

        print(f"Number of hadith entries: {len(hadiths):,}")

        if hadiths:
            first = hadiths[0]
            print(f"\nKeys of first entry: {list(first.keys())}")
            print("\nFirst entry (truncated to 1500 chars):")
            pretty = json.dumps(first, ensure_ascii=False, indent=2)
            print(pretty[:1500])
            if len(pretty) > 1500:
                print("... (truncated)")

        # Count entries with Arabic text
        arabic_keys = ["arabic", "arab"]
        n_with_arabic = sum(
            1 for h in hadiths
            if any(h.get(k) for k in arabic_keys)
        )
        print(f"\n✅ Entries with Arabic text: {n_with_arabic:,} / {len(hadiths):,}")
        print()

    print("=" * 80)
    print("\n✅ Next step:")
    print("  Verify field names match the assumptions in src/hadith_misinfo/ingestion/hadith_json.py")
    print("  Update COLLECTION_STEMS and _extract_text_fields() if needed.\n")


if __name__ == "__main__":
    main()
