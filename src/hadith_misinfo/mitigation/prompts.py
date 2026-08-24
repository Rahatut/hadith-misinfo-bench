"""Prompts and recommendation templates for Layer 3: Misinformation Mitigation."""

from __future__ import annotations

RECOMMENDATION_TEMPLATES: dict[str, str] = {
    "SUPPORTED": (
        "The claim is consistent with the retrieved canonical Hadith source(s). "
        "You may share this with appropriate attribution to the cited collection."
    ),
    "NOT_SUPPORTED": (
        "The retrieved evidence contradicts or does not corroborate this attribution. "
        "Do not circulate this statement as an established Hadith."
    ),
    "INSUFFICIENT_EVIDENCE": (
        "This claim could not be verified against the indexed canonical sources "
        "(Sahih al-Bukhari and Sahih Muslim). Absence from this corpus does not "
        "prove the Hadith is fabricated — it may exist in other collections. "
        "Do not present it as an established Hadith without further scholarly verification."
    ),
}

VERDICT_DISPLAY_LABELS: dict[str, str] = {
    "SUPPORTED": "✅ Supported by canonical evidence",
    "NOT_SUPPORTED": "❌ Not supported by canonical evidence",
    "INSUFFICIENT_EVIDENCE": "⚠️ Could not be verified from available sources",
}
