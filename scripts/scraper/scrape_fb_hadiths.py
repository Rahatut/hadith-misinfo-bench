#!/usr/bin/env python3

"""
Step 2: Facebook Hadith-related Candidate Scraper

Collects Facebook posts containing likely Hadith-attribution
expressions using Playwright.

Input:
    fb_session.json

Output:
    data/raw/facebook_hadith_candidates.jsonl

This script ONLY collects candidate posts.
It does not determine whether a Hadith is authentic/fabricated.
"""

import json
import re
import time
from pathlib import Path
from typing import Optional

from playwright.sync_api import (
    sync_playwright,
    TimeoutError as PlaywrightTimeoutError,
)


# ============================================================
# CONFIGURATION
# ============================================================

SESSION_FILE = Path("fb_session.json")

DEFAULT_OUTPUT = Path(
    "data/raw/facebook_hadith_candidates.jsonl"
)

DEFAULT_MAX_SCROLLS = 30

SCROLL_DISTANCE = 1200
SCROLL_WAIT_MS = 2500
INITIAL_WAIT_MS = 5000


# ============================================================
# HADITH TRIGGERS
# ============================================================

HADITH_TRIGGERS = {

    # -------------------------
    # Bangla
    # -------------------------

    "bn_rasul": [
        r"রাসূলুল্লাহ\s*(?:\(সাঃ\)|\(সা\.\)|ﷺ)?\s*বলেছেন",
        r"রাসুলুল্লাহ\s*(?:\(সাঃ\)|\(সা\.\)|ﷺ)?\s*বলেছেন",
        r"রাসূল\s*(?:\(সাঃ\)|\(সা\.\)|ﷺ)?\s*বলেছেন",
        r"রাসুল\s*(?:\(সাঃ\)|\(সা\.\)|ﷺ)?\s*বলেছেন",
    ],

    "bn_nobiji": [
        r"নবীজী\s*বলেছেন",
        r"নবীজি\s*বলেছেন",
        r"নবী\s*(?:\(সাঃ\)|\(সা\.\)|ﷺ)?\s*বলেছেন",
    ],

    "bn_hadith": [
        r"হাদিসে?\s*(?:এসেছে|আছে|বর্ণিত|বলা হয়েছে|বলা হয়েছে)",
        r"হাদীসে?\s*(?:এসেছে|আছে|বর্ণিত|বলা হয়েছে|বলা হয়েছে)",
    ],

    "bn_source": [
        r"সহীহ\s*বুখারী",
        r"সহিহ\s*বুখারি",
        r"বুখারী\s*হাদিস",
        r"বুখারি\s*হাদিস",
        r"সহীহ\s*মুসলিম",
        r"সহিহ\s*মুসলিম",
        r"মুসলিম\s*হাদিস",
    ],

    # -------------------------
    # English
    # -------------------------

    "en_prophet": [
        r"Prophet\s*(?:Muhammad\s*)?(?:ﷺ|pbuh)?\s*said",
        r"Prophet\s*(?:Muhammad\s*)?(?:peace be upon him)?\s*said",
    ],

    "en_messenger": [
        r"Messenger of Allah\s*said",
        r"The Messenger of Allah\s*said",
    ],

    "en_hadith": [
        r"Hadith\s*(?:states|says|narrates|mentions)",
        r"According to\s+(?:a\s+)?Hadith",
    ],
}


# Compile regex patterns once.
COMPILED_TRIGGERS = {
    name: [
        re.compile(pattern, re.IGNORECASE)
        for pattern in patterns
    ]
    for name, patterns in HADITH_TRIGGERS.items()
}


# ============================================================
# TEXT UTILITIES
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize whitespace while preserving the actual post text.
    """

    if not text:
        return ""

    # Remove zero-width spaces.
    text = text.replace("\u200b", "")

    # Replace non-breaking spaces.
    text = text.replace("\xa0", " ")

    # Collapse repeated spaces/tabs.
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def detect_language(text: str) -> str:
    """
    Lightweight script-based language detection.

    Returns:
        bn      = predominantly Bangla
        en      = predominantly Latin/English
        mixed   = mixture
        unknown = insufficient signal
    """

    if not text:
        return "unknown"

    bangla_chars = len(
        re.findall(
            r"[\u0980-\u09FF]",
            text
        )
    )

    latin_chars = len(
        re.findall(
            r"[A-Za-z]",
            text
        )
    )

    total = bangla_chars + latin_chars

    if total == 0:
        return "unknown"

    bangla_ratio = bangla_chars / total
    latin_ratio = latin_chars / total

    if bangla_ratio >= 0.60:
        return "bn"

    if latin_ratio >= 0.60:
        return "en"

    return "mixed"


def find_hadith_triggers(text: str) -> list[str]:
    """
    Return trigger categories matched by the post.
    """

    matches = []

    for trigger_name, patterns in COMPILED_TRIGGERS.items():

        for pattern in patterns:

            if pattern.search(text):

                matches.append(trigger_name)

                # Don't add the same category twice.
                break

    return matches


def contains_hadith_trigger(text: str) -> bool:
    """
    Return True if the post contains at least one
    Hadith-related trigger.
    """

    return bool(
        find_hadith_triggers(text)
    )


# ============================================================
# FACEBOOK UI HELPERS
# ============================================================

def expand_see_more_buttons(page) -> None:
    """
    Expand visible Facebook 'See more' buttons.

    Facebook changes its DOM frequently, so failures
    are intentionally ignored.
    """

    selectors = [

        'div[role="button"]:has-text("See more")',

        'div[role="button"]:has-text("See More")',

        'div[role="button"]:has-text("আরও দেখুন")',

        'span:has-text("See more")',

        'span:has-text("See More")',

        'span:has-text("আরও দেখুন")',
    ]

    for selector in selectors:

        try:

            buttons = page.locator(
                selector
            )

            count = buttons.count()

            for i in range(count):

                try:

                    button = buttons.nth(i)

                    if not button.is_visible():
                        continue

                    button.click(
                        timeout=1000
                    )

                    page.wait_for_timeout(
                        200
                    )

                except Exception:
                    continue

        except Exception:
            continue


def get_article_text(article) -> Optional[str]:
    """
    Safely extract text from a Facebook article.
    """

    try:

        text = article.inner_text(
            timeout=2000
        )

        if not text:
            return None

        text = normalize_text(text)

        # Ignore extremely short elements.
        if len(text) < 20:
            return None

        return text

    except Exception:

        return None


# ============================================================
# CANDIDATE CREATION
# ============================================================

def extract_candidate(
    article,
    target_url: str,
    scroll_index: int,
) -> Optional[dict]:

    text = get_article_text(
        article
    )

    if not text:
        return None

    triggers = find_hadith_triggers(
        text
    )

    if not triggers:
        return None

    return {

        "record_type":
            "facebook_hadith_candidate",

        "timestamp_scraped":
            time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),

        "target_url":
            target_url,

        "scroll_index":
            scroll_index,

        "language":
            detect_language(text),

        "trigger_matches":
            triggers,

        "raw_text":
            text,
    }


# ============================================================
# MAIN SCRAPER
# ============================================================

def scrape_page_posts(
    target_url: str,
    max_scrolls: int = DEFAULT_MAX_SCROLLS,
    out_path: Path = DEFAULT_OUTPUT,
) -> int:

    # --------------------------------------------------------
    # Check session
    # --------------------------------------------------------

    if not SESSION_FILE.exists():

        print(
            "\nERROR: fb_session.json was not found.\n"
            "Run your Facebook session-saving script first.\n"
        )

        return 0

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Duplicate tracking
    # --------------------------------------------------------

    seen_posts: set[str] = set()

    collected_count = 0

    # --------------------------------------------------------
    # Start Playwright
    # --------------------------------------------------------

    with sync_playwright() as p:

        print(
            "Launching Chromium..."
        )

        browser = p.chromium.launch(
            headless=True
        )

        context = browser.new_context(

            storage_state=str(
                SESSION_FILE
            ),

            viewport={
                "width": 1280,
                "height": 900,
            },

            locale="en-US",

            user_agent=(
                "Mozilla/5.0 "
                "(X11; Linux x86_64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/120.0.0.0 "
                "Safari/537.36"
            ),
        )

        page = context.new_page()

        print(
            f"\nOpening:\n{target_url}\n"
        )

        # ====================================================
        # TRY / FINALLY
        # ====================================================

        try:

            # ------------------------------------------------
            # Navigate
            # ------------------------------------------------

            try:

                page.goto(
                    target_url,
                    wait_until="domcontentloaded",
                    timeout=60_000,
                )

            except PlaywrightTimeoutError:

                print(
                    "Navigation timed out."
                    " Continuing with rendered page..."
                )

            except Exception as e:

                print(
                    f"Navigation warning: {e}"
                )

            # ------------------------------------------------
            # Allow Facebook SPA to render
            # ------------------------------------------------

            page.wait_for_timeout(
                INITIAL_WAIT_MS
            )

            # ------------------------------------------------
            # Wait for main container
            # ------------------------------------------------

            try:

                page.wait_for_selector(
                    'div[role="main"]',
                    timeout=15_000,
                )

            except PlaywrightTimeoutError:

                print(
                    "Warning: Facebook main "
                    "container was not detected."
                )

            # ------------------------------------------------
            # Open output file
            # ------------------------------------------------

            with out_path.open(
                "a",
                encoding="utf-8",
            ) as output:

                # ============================================
                # SCROLL LOOP
                # ============================================

                for scroll_index in range(
                    max_scrolls
                ):

                    print(
                        f"\n[{scroll_index + 1}/"
                        f"{max_scrolls}] "
                        f"Scanning page..."
                    )

                    # ----------------------------------------
                    # Expand posts
                    # ----------------------------------------

                    expand_see_more_buttons(
                        page
                    )

                    # ----------------------------------------
                    # Locate articles
                    # ----------------------------------------

                    try:

                        articles = page.locator(
                            'div[role="article"]'
                        )

                        article_count = (
                            articles.count()
                        )

                    except Exception:

                        article_count = 0

                    new_matches = 0

                    # ----------------------------------------
                    # Process articles
                    # ----------------------------------------

                    for i in range(
                        article_count
                    ):

                        try:

                            article = articles.nth(i)

                            text = get_article_text(
                                article
                            )

                            if not text:
                                continue

                            # --------------------------------
                            # Deduplication
                            # --------------------------------

                            if text in seen_posts:
                                continue

                            seen_posts.add(text)

                            # --------------------------------
                            # Trigger detection
                            # --------------------------------

                            candidate = (
                                extract_candidate(
                                    article,
                                    target_url,
                                    scroll_index,
                                )
                            )

                            if candidate is None:
                                continue

                            # --------------------------------
                            # Save
                            # --------------------------------

                            output.write(
                                json.dumps(
                                    candidate,
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )

                            output.flush()

                            collected_count += 1
                            new_matches += 1

                            print(
                                f"  MATCH #{collected_count}"
                                f" | language="
                                f"{candidate['language']}"
                                f" | triggers="
                                f"{candidate['trigger_matches']}"
                            )

                        except Exception:
                            continue

                    print(
                        f"  Articles found: "
                        f"{article_count}"
                    )

                    print(
                        f"  New candidates: "
                        f"{new_matches}"
                    )

                    # ----------------------------------------
                    # Scroll
                    # ----------------------------------------

                    try:

                        previous_height = (
                            page.evaluate(
                                "document.body.scrollHeight"
                            )
                        )

                    except Exception:

                        previous_height = 0

                    page.evaluate(
                        f"window.scrollBy("
                        f"0, {SCROLL_DISTANCE}"
                        f");"
                    )

                    page.wait_for_timeout(
                        SCROLL_WAIT_MS
                    )

                    # ----------------------------------------
                    # Check whether page grew
                    # ----------------------------------------

                    try:

                        current_height = (
                            page.evaluate(
                                "document.body.scrollHeight"
                            )
                        )

                    except Exception:

                        current_height = 0

                    # If no growth, wait once more.
                    if (
                        current_height
                        == previous_height
                    ):

                        page.wait_for_timeout(
                            3000
                        )

                        try:

                            current_height = (
                                page.evaluate(
                                    "document.body.scrollHeight"
                                )
                            )

                        except Exception:

                            current_height = (
                                previous_height
                            )

                    # ----------------------------------------
                    # Informational message
                    # ----------------------------------------

                    if (
                        current_height
                        == previous_height
                    ):

                        print(
                            "  Page height did not "
                            "increase."
                        )

        # ====================================================
        # CLEANUP
        # ====================================================

        finally:

            print(
                "\nClosing browser..."
            )

            try:
                context.close()
            except Exception:
                pass

            try:
                browser.close()
            except Exception:
                pass

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print(
        f"\nScraping complete."
        f"\nCandidates collected: "
        f"{collected_count}"
        f"\nOutput: {out_path}\n"
    )

    return collected_count


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    TARGET_FACEBOOK_PAGE = (
        "https://www.facebook.com/DailyProthomAlo"
    )

    scrape_page_posts(

        target_url=TARGET_FACEBOOK_PAGE,

        max_scrolls=30,

        out_path=Path(
            "data/raw/"
            "facebook_hadith_candidates.jsonl"
        ),
    )