"""Generate controlled English and Bangla claim paraphrases from Arabic Hadith."""

from __future__ import annotations

from typing import Callable

CompleteFn = Callable[[str], str]

# ── Paraphrase prompts ─────────────────────────────────────────────────────────

_PARAPHRASE_PROMPT = """\
You are helping build a misinformation verification benchmark.

Given the Arabic religious text below, write a short, natural-sounding \
claim in {target_language} that someone might post on social media.

Rules:
- Restate who said/did what in plain language (1–2 sentences).
- Do NOT mention the words: Hadith, Sahih, authentic, fabricated, \
grading, Bukhari, Muslim, collection name, or any authenticity judgement.
- Do NOT add hedging like "it is said that" or "reportedly".
- Write in the first-person-reported style typical of religious social-media posts.
- Use {script_instruction}

Arabic text:
{arabic_matn}

Claim ({target_language}):"""

_LANGUAGE_CONFIG: dict[str, dict[str, str]] = {
    "en": {
        "target_language": "English",
        "script_instruction": "standard written English.",
    },
    "bn": {
        "target_language": "Bangla (Bengali)",
        "script_instruction": "Bengali script (বাংলা). Do NOT transliterate into Latin.",
    },
}


def make_llm_paraphraser(complete: CompleteFn):
    """Return a paraphraser function backed by the given LLM complete function."""

    def _paraphrase(arabic_matn: str, language: str) -> str:
        config = _LANGUAGE_CONFIG.get(language)
        if config is None:
            raise ValueError(
                f"Unsupported target language: {language!r}. "
                f"Supported: {list(_LANGUAGE_CONFIG.keys())}"
            )
        prompt = _PARAPHRASE_PROMPT.format(arabic_matn=arabic_matn, **config)
        return complete(prompt).strip()

    return _paraphrase


def paraphrase_pair(arabic_matn: str, paraphraser) -> tuple[str, str]:
    """Generate (claim_en, claim_bn) directly from an Arabic matn."""
    claim_en = paraphraser(arabic_matn, "en")
    claim_bn = paraphraser(arabic_matn, "bn")
    return claim_en, claim_bn
