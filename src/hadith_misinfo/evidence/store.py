"""Evidence store — unified in-memory and on-disk index over all Hadith collections.

The EvidenceStore is the single source of truth for the retrieval corpus.
Build it once from raw hadith-json files, persist it to data/processed/evidence.jsonl,
and load it cheaply at experiment time.
"""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from hadith_misinfo.ingestion.hadith_json import iter_evidence_records
from hadith_misinfo.schemas import EvidenceRecord


class EvidenceStore:
    """In-memory index of EvidenceRecord objects, keyed by evidence_id."""

    def __init__(self) -> None:
        self._records: dict[str, EvidenceRecord] = {}

    # ── Build ─────────────────────────────────────────────────────────────────

    @classmethod
    def build(cls, raw_dir: str | Path, verbose: bool = True) -> "EvidenceStore":
        """Ingest all Hadith JSON files under ``raw_dir`` into a new store."""
        store = cls()
        raw_dir = Path(raw_dir)
        records = iter_evidence_records(raw_dir)
        if verbose:
            records = tqdm(records, desc="Building evidence store")
        for rec in records:
            store._records[rec.evidence_id] = rec
        if verbose:
            print(f"Evidence store: {len(store._records):,} records loaded.")
        return store

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, out_path: str | Path) -> None:
        """Persist all records to a JSONL file."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for rec in self._records.values():
                f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        print(f"Saved {len(self._records):,} evidence records to {out_path}.")

    @classmethod
    def load(cls, path: str | Path) -> "EvidenceStore":
        """Load a previously saved JSONL evidence store."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Evidence store not found at {path}.\n"
                "Run: python scripts/build_evidence_index.py"
            )
        store = cls()
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = EvidenceRecord.from_dict(json.loads(line))
                    store._records[rec.evidence_id] = rec
        return store

    # ── Access ────────────────────────────────────────────────────────────────

    def get(self, evidence_id: str) -> EvidenceRecord:
        """Return a record by ID.  Raises KeyError if not found."""
        return self._records[evidence_id]

    def get_many(self, evidence_ids: list[str]) -> list[EvidenceRecord]:
        """Return records for a list of IDs (silently skips missing IDs)."""
        return [self._records[eid] for eid in evidence_ids if eid in self._records]

    def all_records(self) -> list[EvidenceRecord]:
        """Return all records as a list (order matches insertion order)."""
        return list(self._records.values())

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, evidence_id: str) -> bool:
        return evidence_id in self._records

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return basic statistics about the store."""
        from collections import Counter
        collections = Counter(r.collection for r in self._records.values())
        return {
            "total": len(self._records),
            "collections": dict(collections),
        }
