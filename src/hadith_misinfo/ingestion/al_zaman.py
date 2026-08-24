"""Load and sample the Al-Zaman/Noman Dataset C."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

import pandas as pd

from hadith_misinfo.preprocessing.bangla import contains_bangla
from hadith_misinfo.schemas import SocialMediaPost

COMMENT_COLS = ["comment", "text", "post_text", "content", "message", "Comment"]
ID_COLS = ["id", "comment_id", "row_id", "ID"]


def _find_comment_col(df: pd.DataFrame) -> str:
    for c in COMMENT_COLS:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not find a comment/text column in the Al-Zaman dataset.\n"
        f"Candidates tried: {COMMENT_COLS}.\n"
        f"Actual columns: {list(df.columns)}.\n"
        "Update COMMENT_COLS in ingestion/al_zaman.py."
    )


def iter_posts(raw_dir: str | Path) -> Iterator[SocialMediaPost]:
    """Yield SocialMediaPost objects from all CSV/XLSX files in raw_dir."""
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("*.csv")) + sorted(raw_dir.glob("*.xlsx"))

    if not files:
        raise FileNotFoundError(
            f"No CSV/XLSX files found under {raw_dir}.\n"
            "Download the Al-Zaman/Noman dataset from Mendeley Data and place\n"
            "the file(s) there. See README § Data download."
        )

    for path in files:
        if path.suffix == ".xlsx":
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path, encoding="utf-8", errors="replace")

        df.columns = [c.strip() for c in df.columns]
        comment_col = _find_comment_col(df)
        id_col = next((c for c in ID_COLS if c in df.columns), None)

        for row_idx, row in df.iterrows():
            text = str(row[comment_col]).strip()
            if not text or text.lower() in {"nan", ""}:
                continue

            post_id = (
                str(row[id_col])
                if id_col and pd.notna(row.get(id_col))
                else f"{path.stem}_{row_idx}"
            )

            meta = {
                k: str(row[k])
                for k in df.columns
                if k not in {comment_col, id_col}
                and pd.notna(row.get(k))
            }

            yield SocialMediaPost(
                post_id=post_id,
                raw_text=text,
                source="al-zaman",
                metadata=meta,
            )


def sample_posts(
    raw_dir: str | Path,
    n: int = 100,
    seed: int = 42,
    require_bangla: bool = True,
) -> list[SocialMediaPost]:
    """Return a deterministic random sample of posts from Dataset C."""
    posts = list(iter_posts(raw_dir))

    if require_bangla:
        bangla = [p for p in posts if contains_bangla(p.raw_text)]
        if len(bangla) >= n:
            posts = bangla

    rng = random.Random(seed)
    return rng.sample(posts, min(n, len(posts)))
