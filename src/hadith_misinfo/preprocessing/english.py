"""English text normalization utilities."""

from __future__ import annotations

import re

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_english(text: str, lower: bool = True, strip_punct: bool = True) -> str:
    """Normalize English text: lowercasing, punctuation removal, whitespace collapsing."""
    if not text:
        return ""
    if lower:
        text = text.lower()
    if strip_punct:
        text = _PUNCT.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
