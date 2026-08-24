"""Ingest the hadith-json (Bukhari/Muslim) evidence corpus.

Source: https://github.com/fawazahmed0/hadith-api
    → data/raw/hadith-json/

Expected raw layout (hadith-json project convention — verify with
``scripts/inspect_hadith_json.py``):

    data/raw/hadith-json/
        editions/
            eng-bukhari.json
            eng-muslim.json
            ara-bukharibukhari.json   ← Arabic
            ...

OR the older 9-books layout:
    data/raw/hadith-json/
        by_book/the_9_books/
            bukhari.json
            muslim.json

Both layouts are searched automatically.

Field names are best-guesses from the publicly documented schema — always
run ``scripts/inspect_hadith_json.py`` against your actual download to
confirm before running the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from hadith_misinfo.schemas import EvidenceRecord

# ── Known collection file stems and their display names ──────────────────────
COLLECTION_STEMS: dict[str, str] = {
    # 9-books / legacy layout
    "bukhari":             "Sahih al-Bukhari",
    "muslim":              "Sahih Muslim",
    # hadith-api editions layout (English)
    "eng-bukhari":         "Sahih al-Bukhari",
    "eng-muslim":          "Sahih Muslim",
    # hadith-api editions layout (Arabic)
    "ara-bukhari":         "Sahih al-Bukhari",
    "ara-muslim":          "Sahih Muslim",
    "ara-bukharibukhari":  "Sahih al-Bukhari",
    "ara-muslimmuslim":    "Sahih Muslim",
}


def _find_json_files(raw_dir: Path) -> list[tuple[Path, str]]:
    """Return (path, collection_key) pairs for all recognised Hadith JSON files."""
    all_json = sorted(raw_dir.rglob("*.json"))
    found: list[tuple[Path, str]] = []
    for f in all_json:
        key = f.stem.lower()
        if key in COLLECTION_STEMS:
            found.append((f, key))
    if not found:
        raise FileNotFoundError(
            f"No recognised Hadith JSON files found under {raw_dir}.\n"
            f"Expected stems like: {list(COLLECTION_STEMS.keys())}.\n"
            "Clone/extract the hadith-json repo there first, then re-run.\n"
            "See README § Data download."
        )
    return found


def _iter_raw_hadiths(path: Path) -> Iterator[dict]:
    """Yield raw hadith dicts from a JSON file (handles multiple schemas)."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Schema A: {"hadiths": [...]}  (legacy 9-books)
    if isinstance(data, dict) and "hadiths" in data:
        yield from data["hadiths"]
    # Schema B: {"data": {"1": {...}, "2": {...}}}  (hadith-api editions)
    elif isinstance(data, dict) and "data" in data and isinstance(data["data"], dict):
        for item in data["data"].values():
            yield item
    # Schema C: flat list
    elif isinstance(data, list):
        yield from data
    else:
        raise ValueError(
            f"{path}: unrecognised top-level schema. "
            f"Top-level keys: {list(data.keys()) if isinstance(data, dict) else type(data).__name__}. "
            "Run scripts/inspect_hadith_json.py and update this function."
        )


def _extract_text_fields(item: dict) -> tuple[str, str, str]:
    """Extract (number, arabic_text, english_text) from a raw hadith dict."""
    number = (
        item.get("arabicnumber")
        or item.get("hadithnumber")
        or item.get("number")
        or item.get("id")
        or ""
    )

    arabic = str(item.get("arabic") or item.get("arab") or "").strip()

    english_raw = item.get("english") or item.get("text") or {}
    if isinstance(english_raw, dict):
        narrator = str(english_raw.get("narrator") or "").strip()
        body = str(english_raw.get("text") or "").strip()
        english = f"{narrator} {body}".strip() if narrator else body
    elif isinstance(english_raw, str):
        english = english_raw.strip()
    else:
        english = ""

    return str(number), arabic, english


def iter_evidence_records(raw_dir: str | Path) -> Iterator[EvidenceRecord]:
    """Yield EvidenceRecord objects from all Bukhari/Muslim JSON files in ``raw_dir``."""
    raw_dir = Path(raw_dir)
    found = _find_json_files(raw_dir)

    # Map by collection name: e.g. "Sahih al-Bukhari" -> {'ara': path, 'eng': path}
    collections_map: dict[str, dict[str, Path]] = {}
    for path, key in found:
        coll = COLLECTION_STEMS[key]
        if coll not in collections_map:
            collections_map[coll] = {}
        if key.startswith("ara-"):
            collections_map[coll]["ara"] = path
        elif key.startswith("eng-"):
            collections_map[coll]["eng"] = path
        else:
            collections_map[coll]["single"] = path

    for coll_name, paths in collections_map.items():
        coll_prefix = "bukhari" if "Bukhari" in coll_name else "muslim"
        if "ara" in paths and "eng" in paths:
            # Merge Arabic and English editions by hadith number
            ara_items = list(_iter_raw_hadiths(paths["ara"]))
            eng_items = list(_iter_raw_hadiths(paths["eng"]))

            eng_by_num: dict[str, str] = {}
            for item in eng_items:
                num, _, eng_text = _extract_text_fields(item)
                # In english-only edition, item['text'] is the english text
                if not eng_text and isinstance(item.get("text"), str):
                    eng_text = item["text"].strip()
                if num and eng_text:
                    eng_by_num[str(num)] = eng_text

            for item in ara_items:
                num = str(item.get("hadithnumber") or item.get("arabicnumber") or item.get("id") or "")
                ara_text = str(item.get("text") or item.get("arabic") or item.get("arab") or "").strip()
                if not ara_text:
                    continue
                eng_text = eng_by_num.get(num) or None
                book_val = ""
                ref_val = str(num)
                ref_obj = item.get("reference")
                if isinstance(ref_obj, dict):
                    book_val = str(ref_obj.get("book", ""))
                    ref_val = f"Hadith {num}"
                elif item.get("book"):
                    book_val = str(item["book"])

                yield EvidenceRecord(
                    evidence_id=f"{coll_prefix}_{num}",
                    collection=coll_name,
                    book=book_val,
                    reference=ref_val,
                    arabic_matn=ara_text,
                    english_text=eng_text,
                    grade=item.get("grade") or (item.get("grades", [{}])[0].get("grade") if item.get("grades") else None),
                )
        else:
            # Legacy or single-file layout
            for key, path in [(k, p) for p, k in found if COLLECTION_STEMS[k] == coll_name]:
                for item in _iter_raw_hadiths(path):
                    number, arabic, english = _extract_text_fields(item)
                    if not arabic:
                        continue
                    yield EvidenceRecord(
                        evidence_id=f"{coll_prefix}_{number}",
                        collection=coll_name,
                        book=str(item.get("book") or item.get("bookNumber") or ""),
                        reference=str(item.get("reference") or number),
                        arabic_matn=arabic,
                        english_text=english or None,
                        grade=item.get("grade") or (item.get("grades", [{}])[0].get("grade") if item.get("grades") else None),
                    )
