"""Layer 1: Hadith claim extraction from noisy social-media posts."""

from __future__ import annotations

import json
import re
from typing import Callable

from hadith_misinfo.extraction.prompts import DETECTION_PROMPT, EXTRACTION_PROMPT
from hadith_misinfo.schemas import ExtractedClaim, Language

CompleteFn = Callable[[str], str]


class ClaimExtractor:
    """Two-stage LLM-based Hadith claim extractor.

    Stage 1: Detect if post contains Hadith attribution.
    Stage 2: Extract the clean claim text.
    """

    def __init__(
        self,
        complete: CompleteFn,
        min_confidence: float = 0.5,
    ) -> None:
        self.complete = complete
        self.min_confidence = min_confidence

    def extract(self, post_id: str, post_text: str) -> ExtractedClaim:
        """Run the two-stage extraction pipeline for one post."""
        detection = self._detect(post_text)
        if (
            not detection.get("contains_hadith_claim", False)
            or detection.get("confidence", 0.0) < self.min_confidence
        ):
            return ExtractedClaim(
                post_id=post_id,
                claim_text="",
                language=_map_language(detection.get("language", "bn")),
                claim_type="none",
                confidence=detection.get("confidence", 0.0),
                extraction_notes=detection.get("reasoning", ""),
            )

        extraction = self._extract(post_text)
        return ExtractedClaim(
            post_id=post_id,
            claim_text=extraction.get("extracted_claim", "").strip(),
            language=_map_language(detection.get("language", "bn")),
            claim_type=extraction.get("claim_type", "hadith_attribution"),  # type: ignore[arg-type]
            confidence=detection.get("confidence", 0.8),
            extraction_notes=extraction.get("notes", ""),
        )

    def _detect(self, post_text: str) -> dict:
        prompt = DETECTION_PROMPT.format(post_text=post_text[:3000])
        raw = self.complete(prompt)
        return _parse_json_response(raw, fallback={"contains_hadith_claim": False, "confidence": 0.0})

    def _extract(self, post_text: str) -> dict:
        prompt = EXTRACTION_PROMPT.format(post_text=post_text[:3000])
        raw = self.complete(prompt)
        return _parse_json_response(raw, fallback={"extracted_claim": "", "claim_type": "other"})


def _parse_json_response(raw: str, fallback: dict) -> dict:
    clean = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return fallback


def _map_language(raw_lang: str) -> Language:
    mapping = {
        "bn": "bn", "bangla": "bn", "bengali": "bn",
        "en": "en", "english": "en",
        "ar": "ar", "arabic": "ar",
        "mixed": "bn",
    }
    return mapping.get(raw_lang.lower(), "bn")  # type: ignore[return-value]
