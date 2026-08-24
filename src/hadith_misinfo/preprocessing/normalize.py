"""Unified text normalization and tokenization pipeline for BM25 and multilingual NLP."""

from __future__ import annotations

import re

from hadith_misinfo.preprocessing.arabic import normalize_arabic
from hadith_misinfo.preprocessing.bangla import normalize_bangla
from hadith_misinfo.preprocessing.english import normalize_english

_GENERAL_PUNCT = re.compile(r"[^\w\s\u0600-\u06FF\u0980-\u09FF]", re.UNICODE)


def normalize_text(text: str) -> str:
    """Apply all script-specific normalizations plus lowercasing."""
    if not text:
        return ""
    text = normalize_arabic(text)
    text = normalize_bangla(text)
    text = _GENERAL_PUNCT.sub(" ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Whitespace-split tokenization after normalization."""
    return [t for t in text.split() if t]


def normalize_and_tokenize(text: str) -> list[str]:
    """Full normalize-then-tokenize pipeline used by BM25Retriever."""
    return tokenize(normalize_text(text))
