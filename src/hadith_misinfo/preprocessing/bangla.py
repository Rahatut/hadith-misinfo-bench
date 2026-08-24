"""Bangla (Bengali) text normalization utilities."""

from __future__ import annotations

import re
import unicodedata

# Bengali punctuation and characters (e.g. Dari |, brackets, quotes)
_BANGLA_PUNCT = re.compile(r"[।॥!?,;:\"\'-()\[\]{}]")


def normalize_bangla(text: str, strip_punct: bool = False) -> str:
    """Normalize Bengali text: Unicode NFC normalization, whitespace collapsing, optional punctuation cleanup."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    if strip_punct:
        text = _BANGLA_PUNCT.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_bangla(text: str) -> bool:
    """Return True if the text contains at least one Bengali script character (U+0980–U+09FF)."""
    return any("\u0980" <= ch <= "\u09FF" for ch in text)
