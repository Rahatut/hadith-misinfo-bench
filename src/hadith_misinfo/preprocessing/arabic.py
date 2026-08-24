"""Arabic text normalization utilities."""

from __future__ import annotations

import re

# Arabic diacritics / tashkeel (U+0610–U+061A, U+064B–U+065F, U+0670)
_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670]")

# Tatweel / kashida (U+0640)
_TATWEEL = re.compile(r"\u0640")

# Alef variants → plain alef (U+0627)
_ALEF_VARIANTS = re.compile(r"[\u0622\u0623\u0625\u0671]")

# Teh marbuta → heh
_TEH_MARBUTA = re.compile(r"\u0629")

# Arabic punctuation & symbols
_ARABIC_PUNCT = re.compile(r"[\u060C\u061B\u061F\u066A\u066B\u066C\u066D\u06D4]")


def normalize_arabic(text: str, strip_tashkeel: bool = True, normalize_alef: bool = True) -> str:
    """Normalize Arabic text: strip diacritics, tatweel, normalize alef variants and teh marbuta."""
    if not text:
        return ""
    if strip_tashkeel:
        text = _ARABIC_DIACRITICS.sub("", text)
    text = _TATWEEL.sub("", text)
    if normalize_alef:
        text = _ALEF_VARIANTS.sub("\u0627", text)
    text = _TEH_MARBUTA.sub("\u0647", text)
    text = _ARABIC_PUNCT.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
