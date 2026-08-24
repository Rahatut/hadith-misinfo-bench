"""Ingest the MAHADDAT dataset (CSV files) into RawRecord objects.

MAHADDAT is the Fabricated Hadith detection dataset from:
    Gaanoun, K. & Alsuhaibani, M. (2022).
    "Fabricated Hadith Detection: A Novel Matn-Based Approach With
    Transformer Language Models." IEEE Access.

Run ``scripts/inspect_mahaddat.py`` FIRST to confirm the actual column
names in your downloaded CSV(s) before trusting the COLUMN_MAP below.

Expected raw layout:
    data/raw/mahaddat/
        *.csv    ← one or more CSV files from the MAHADDAT release

Key design constraint (§4 of design doc)
----------------------------------------
We use ONLY the test split for Dataset A.  Do not leak training or
development records into the benchmark.  If the CSV does not have an
explicit split column, we apply a deterministic 80/10/10 split by row
index (seed=42) and take the last 10% as test.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pandas as pd

# ── Column name mapping ────────────────────────────────────────────────────────
# Update these after running inspect_mahaddat.py to match the real CSV.
# Keys: canonical names used in this codebase.
# Values: lists of candidate column names in the actual CSV (first match wins).

COLUMN_MAP: dict[str, list[str]] = {
    "matn_arabic": ["matan", "Matan", "matn_arabic", "arabic", "arabic_matn", "text_arabic", "hadith_arabic"],
    "label":       ["degree", "Degree", "label", "class", "y", "target", "fabricated"],
    "split":       ["split", "set", "partition", "fold"],
    "source_id":   ["Unnamed: 0", "unnamed: 0", "id", "hadith_id", "row_id", "index"],
}

# Normalise raw label values → canonical "authentic" | "fabricated"
LABEL_MAP: dict[str, str] = {
    # Authentic variants
    "authentic":   "authentic",
    "real":        "authentic",
    "0":           "authentic",
    "صحيح":        "authentic",
    # Fabricated variants
    "fabricated":  "fabricated",
    "fake":        "fabricated",
    "mawdu":       "fabricated",
    "mawdoo":      "fabricated",
    "1":           "fabricated",
    "موضوع":       "fabricated",
}

# Test-split value(s) in the split column
TEST_SPLIT_VALUES = {"test", "Test", "TEST", "val", "Val", "VAL"}


@dataclass
class RawRecord:
    """One record from MAHADDAT, after column normalisation.

    ``gold_evidence_ids`` is populated later by the benchmark builder when it
    successfully matches an authentic record against the evidence corpus.
    """

    source_id: str
    label: str                             # "authentic" | "fabricated"
    arabic_matn: str
    gold_evidence_ids: list[str] = field(default_factory=list)


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _stable_id(path: Path, row_idx: int, matn: str) -> str:
    """Generate a stable source ID from file + row + content hash."""
    digest = hashlib.md5(matn.encode("utf-8", errors="replace")).hexdigest()[:8]
    return f"{path.stem}_{row_idx}_{digest}"


def iter_records(
    raw_dir: str | Path,
    split: str = "test",
    explicit_split_only: bool = False,
) -> Iterator[RawRecord]:
    """Yield RawRecord objects from all CSV files in ``raw_dir``.

    Parameters
    ----------
    raw_dir:
        Directory containing the raw MAHADDAT CSV file(s).
    split:
        Which split to return.  "test" (default) returns only test records.
        "all" returns every record (useful for debugging).
    explicit_split_only:
        If True and no split column is found, raise an error rather than
        inferring the split from row index.
    """
    raw_dir = Path(raw_dir)
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found under {raw_dir}.\n"
            "Download/extract the MAHADDAT dataset there first, then re-run.\n"
            "See README § Data download."
        )

    # Check if files are already split by filename (e.g., testFinal.csv, trainFinal.csv)
    if split != "all":
        split_matched_files = [f for f in csv_files if split.lower() in f.stem.lower()]
        if split_matched_files:
            csv_files = split_matched_files

    for path in csv_files:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]

        # ── Resolve columns ────────────────────────────────────────────────────
        matn_col = _find_col(df, COLUMN_MAP["matn_arabic"])
        label_col = _find_col(df, COLUMN_MAP["label"])
        split_col = _find_col(df, COLUMN_MAP["split"])
        id_col = _find_col(df, COLUMN_MAP["source_id"])

        if matn_col is None:
            raise KeyError(
                f"{path}: could not find an Arabic matn column. "
                f"Candidates tried: {COLUMN_MAP['matn_arabic']}. "
                f"Actual columns: {list(df.columns)}. "
                "Update COLUMN_MAP in ingestion/mahaddat.py."
            )
        if label_col is None:
            raise KeyError(
                f"{path}: could not find a label column. "
                f"Candidates tried: {COLUMN_MAP['label']}. "
                f"Actual columns: {list(df.columns)}. "
                "Update COLUMN_MAP in ingestion/mahaddat.py."
            )

        # ── Split filtering ────────────────────────────────────────────────────
        if split != "all":
            if split_col is not None:
                mask = df[split_col].isin(TEST_SPLIT_VALUES if split == "test" else {split})
                df = df[mask].reset_index(drop=True)
            elif split.lower() in path.stem.lower():
                # Entire file is this split
                pass
            elif explicit_split_only:
                raise KeyError(
                    f"{path}: no split column found and explicit_split_only=True. "
                    f"Candidates tried: {COLUMN_MAP['split']}."
                )
            else:
                # Deterministic 80/10/10 by row index (seed=42)
                n = len(df)
                test_start = int(n * 0.9)
                if split == "test":
                    df = df.iloc[test_start:].reset_index(drop=True)
                elif split == "dev":
                    df = df.iloc[int(n * 0.8): test_start].reset_index(drop=True)
                else:
                    df = df.iloc[: int(n * 0.8)].reset_index(drop=True)

        # ── Yield records ──────────────────────────────────────────────────────
        for row_idx, row in df.iterrows():
            matn = str(row[matn_col]).strip()
            if not matn or matn.lower() in {"nan", ""}:
                continue

            raw_label = str(row[label_col]).strip().lower()
            label = LABEL_MAP.get(raw_label)
            if label is None:
                # Unknown label — skip rather than crash
                continue

            sid = (
                str(row[id_col]) if id_col and pd.notna(row.get(id_col)) else None
            ) or _stable_id(path, int(str(row_idx)), matn)

            yield RawRecord(
                source_id=sid,
                label=label,
                arabic_matn=matn,
            )
