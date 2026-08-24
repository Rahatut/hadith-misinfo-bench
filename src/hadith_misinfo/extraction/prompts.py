"""Prompts for Layer 1: Social Media Hadith Claim Extraction."""

DETECTION_PROMPT = """\
You are analysing a social-media post to determine if it contains a \
claim that attributes a saying or action to the Prophet Muhammad (ﷺ) \
or other Islamic religious figures (a Hadith attribution).

Social-media post (may be in Bangla, English, or mixed):
\"\"\"
{post_text}
\"\"\"

Answer ONLY with a JSON object:
{{
  "contains_hadith_claim": true/false,
  "language": "<detected language: bn/en/ar/mixed>",
  "confidence": <float 0.0–1.0>,
  "reasoning": "<one sentence>"
}}"""

EXTRACTION_PROMPT = """\
A social-media post contains a Hadith attribution claim. \
Extract ONLY the substantive religious claim (who said/did what).

Rules:
- Remove: calls to share, emojis, expressions of praise, social commentary.
- Keep: the attribution ("The Prophet said / did...") and its substance.
- Do NOT add, remove, or change the actual claimed content.
- Output the extracted claim in the SAME language as the original claim.
- If the claim is in Bangla, output in Bangla script.

Social-media post:
\"\"\"
{post_text}
\"\"\"

Answer ONLY with a JSON object:
{{
  "extracted_claim": "<the clean claim text>",
  "claim_type": "<hadith_attribution|religious_general|other>",
  "notes": "<optional brief note>"
}}"""
