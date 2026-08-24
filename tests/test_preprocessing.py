"""Tests for Arabic, Bangla, and English preprocessing."""

from hadith_misinfo.preprocessing.arabic import normalize_arabic
from hadith_misinfo.preprocessing.bangla import contains_bangla, normalize_bangla
from hadith_misinfo.preprocessing.english import normalize_english
from hadith_misinfo.preprocessing.normalize import normalize_and_tokenize, normalize_text


def test_arabic_diacritics_and_alef():
    text = "إِنَّمَا الأَعْمَالُ بِالنِّيَّاتِ"
    normalized = normalize_arabic(text)
    assert "ِ" not in normalized  # no kasra
    assert "ّ" not in normalized  # no shadda
    assert normalized.startswith("انما")


def test_bangla_unicode_and_detection():
    bn_text = "নবী (সাঃ) বলেছেন।"
    assert contains_bangla(bn_text)
    assert not contains_bangla("Prophet said in English.")
    assert normalize_bangla(bn_text, strip_punct=True) == "নবী সাঃ বলেছেন"


def test_english_normalization():
    en_text = "Actions are judged by Intentions!"
    assert normalize_english(en_text) == "actions are judged by intentions"


def test_multilingual_tokenize():
    tokens = normalize_and_tokenize("রাসূল (সাঃ) said: انما الاعمال")
    assert "রাসূল" in tokens
    assert "said" in tokens
    assert "انما" in tokens
