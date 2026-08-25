#!/usr/bin/env python

"""
Targeted Facebook Hadith Claim Scraper

Pipeline:
1. Search Facebook independently for high-signal Islamic/Hadith keywords.
2. Discover Pages and Groups.
3. Validate discovered source titles using target keywords.
4. Rank sources by keyword relevance.
5. Scrape posts from validated sources.
6. Keep only posts containing Hadith-attribution triggers.
7. Save results as JSONL.

Requires:
    pip install playwright
    playwright install chromium

Requires an authenticated:
    fb_session.json
"""

import json
import re
import time
import urllib.parse
from pathlib import Path

from playwright.sync_api import sync_playwright


# ============================================================
# CONFIGURATION
# ============================================================

SESSION_FILE = "fb_session.json"

DEFAULT_OUTPUT = "data/raw/fb_targeted_hadith_claims.jsonl"


# ============================================================
# SOURCE DISCOVERY KEYWORDS
# ============================================================
#
# These are intentionally NOT combined into phrases.
#
# Facebook will be searched independently for each keyword.
#
# A result is considered relevant if its title contains one
# or more of these terms.
#
# You can add/remove terms without changing the pipeline.
# ============================================================

TARGET_KEYWORDS = [

    # -------------------------
    # English
    # -------------------------

    "islam",
    "islamic",
    "muslim",
    "sunnah",
    "sunnat",
    "hadith",
    "hadeeth",
    "sahih",
    "bukhari",
    "muslim",
    "deen",
    "ummah",
    "dawah",

    # -------------------------
    # Bangla
    # -------------------------

    "ইসলাম",
    "ইসলামী",
    "ইসলামিক",
    "মুসলিম",
    "মুসলমান",
    "মুমিন",
    "সুন্নাহ",
    "সুন্নাত",
    "হাদিস",
    "হাদীস",
    "হাদীস শরীফ",
    "সহীহ",
    "সহিহ",
    "বুখারী",
    "বুখারি",
    "মুসলিম শরীফ",
    "দ্বীন",
    "উম্মাহ",
    "দাওয়াত",
    "দাওয়াত",
    "বাংলা",
]


# ============================================================
# HADITH POST TRIGGERS
# ============================================================
#
# These are intentionally stricter than source discovery.
#
# A source can be Islamic without every post being a Hadith claim.
# ============================================================

HADITH_TRIGGERS = [

    # Bangla attribution
    r"রাসূলুল্লাহ\s*(?:\(সাঃ\)|\(সা\.\)|ﷺ)?\s*বলেছেন",
    r"রাসুলুল্লাহ\s*(?:\(সাঃ\)|\(সা\.\)|ﷺ)?\s*বলেছেন",
    r"রাসূল\s*(?:\(সাঃ\)|\(সা\.\)|ﷺ)?\s*বলেছেন",
    r"রাসুল\s*(?:\(সাঃ\)|\(সা\.\)|ﷺ)?\s*বলেছেন",

    r"নবীজী\s*বলেছেন",
    r"নবীজি\s*বলেছেন",
    r"নবী\s*(?:\(সাঃ\)|\(সা\.\)|ﷺ)?\s*বলেছেন",

    r"হাদিসে?\s*(?:এসেছে|আছে|বর্ণিত|বলা হয়েছে|বলা হয়েছে)",
    r"হাদীসে?\s*(?:এসেছে|আছে|বর্ণিত|বলা হয়েছে|বলা হয়েছে)",

    r"হাদিসে?\s*রাসূল",
    r"হাদীসে?\s*রাসূল",

    # Common Bangla Hadith introduction
    r"রাসূলুল্লাহ\s*ﷺ",
    r"রাসুলুল্লাহ\s*ﷺ",
    r"নবীজী\s*ﷺ",
    r"নবীজি\s*ﷺ",

    # English
    r"Prophet\s*(?:ﷺ|pbuh)?\s*said",
    r"Prophet Muhammad\s*(?:ﷺ|pbuh)?\s*said",
    r"Messenger of Allah\s*said",
    r"The Prophet\s*(?:ﷺ|pbuh)?\s*said",

    # Citation-style expressions
    r"Sahih\s+al[- ]Bukhari",
    r"Sahih\s+Muslim",
    r"Bukhari\s+\d+",
    r"Muslim\s+\d+",
]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize whitespace while preserving Bangla/Unicode text.
    """

    if not text:
        return ""

    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_url(url: str) -> str:
    """
    Normalize Facebook URLs for deduplication.
    """

    if not url:
        return ""

    url = url.split("?")[0]
    url = url.split("#")[0]

    if url.endswith("/"):
        url = url[:-1]

    return url.lower()


# ============================================================
# SOURCE TITLE VALIDATION
# ============================================================

def matched_target_keywords(name: str) -> list[str]:
    """
    Return all target keywords appearing in a source title.
    """

    name_lower = normalize_text(name).lower()

    matches = []

    for keyword in TARGET_KEYWORDS:
        if keyword.lower() in name_lower:
            matches.append(keyword)

    return matches


def is_target_source(name: str) -> bool:
    """
    Return True if the source title contains at least one
    target keyword.
    """

    return len(matched_target_keywords(name)) > 0


def source_relevance_score(name: str) -> int:
    """
    Score a source based on the number of target keywords
    appearing in its title.

    More matching keywords = stronger candidate.
    """

    return len(matched_target_keywords(name))


# ============================================================
# HADITH POST VALIDATION
# ============================================================

def matched_hadith_triggers(text: str) -> list[str]:
    """
    Return the Hadith trigger patterns that match the post.
    """

    text = normalize_text(text)

    matches = []

    for pattern in HADITH_TRIGGERS:

        try:
            if re.search(pattern, text, re.IGNORECASE):
                matches.append(pattern)

        except re.error:
            continue

    return matches


def contains_hadith_trigger(text: str) -> bool:
    """
    Determine whether a post appears to contain a Hadith claim.
    """

    return len(matched_hadith_triggers(text)) > 0


# ============================================================
# FACEBOOK SEARCH
# ============================================================

def build_search_url(search_type: str, query: str) -> str:
    """
    Build Facebook search URL.
    """

    encoded_query = urllib.parse.quote(query)

    return (
        f"https://www.facebook.com/search/"
        f"{search_type}/?q={encoded_query}"
    )


def discover_targets(
    page,
    query: str,
    search_type: str = "pages",
    max_results: int = 10,
) -> list[dict]:
    """
    Search Facebook for Pages or Groups using ONE keyword.

    Results are validated based on their titles.
    """

    search_url = build_search_url(search_type, query)

    print(
        f"\nSearching {search_type.upper()} "
        f"for keyword: '{query}'"
    )

    targets = []

    try:

        page.goto(
            search_url,
            wait_until="domcontentloaded",
            timeout=60000,
        )

        page.wait_for_timeout(4000)

        # Allow SPA content to render.
        try:
            page.wait_for_selector(
                'a[role="link"]',
                timeout=10000,
            )
        except Exception:
            pass

        links = page.query_selector_all(
            'a[role="link"]'
        )

        for link in links:

            try:

                href = link.get_attribute("href") or ""

                title = normalize_text(
                    link.inner_text()
                )

                if not title:
                    continue

                if len(title) < 3:
                    continue

                # Ignore obvious navigation/search links.
                if "search" in href.lower():
                    continue

                # We only want Facebook source URLs.
                if "facebook.com" not in href:
                    continue

                # Validate title against our keyword vocabulary.
                if not is_target_source(title):
                    continue

                normalized_url = normalize_url(href)

                if not normalized_url:
                    continue

                matches = matched_target_keywords(title)

                target = {
                    "title": title,
                    "url": normalized_url,
                    "source_type": search_type,
                    "discovered_by": query,
                    "matched_keywords": matches,
                    "relevance_score": len(matches),
                }

                # Deduplicate within this search.
                if any(
                    x["url"] == normalized_url
                    for x in targets
                ):
                    continue

                targets.append(target)

                print(
                    f"  + {title}"
                    f" | score={len(matches)}"
                    f" | keywords={matches}"
                )

                if len(targets) >= max_results:
                    break

            except Exception:
                continue

    except Exception as e:

        print(
            f"  Search warning for '{query}': {e}"
        )

    return targets


# ============================================================
# SOURCE DEDUPLICATION
# ============================================================

def merge_sources(sources: list[dict]) -> list[dict]:
    """
    Merge duplicate sources discovered through multiple searches.

    If a source appears through several keywords, retain all
    discovery keywords and all matched title keywords.
    """

    merged = {}

    for source in sources:

        url = normalize_url(
            source.get("url", "")
        )

        if not url:
            continue

        if url not in merged:

            merged[url] = {
                "title": source.get("title", ""),
                "url": url,
                "source_type": source.get(
                    "source_type",
                    "unknown",
                ),
                "discovered_by": set(),
                "matched_keywords": set(),
            }

        merged[url]["discovered_by"].add(
            source.get("discovered_by", "")
        )

        for keyword in source.get(
            "matched_keywords",
            [],
        ):
            merged[url]["matched_keywords"].add(
                keyword
            )

    result = []

    for source in merged.values():

        source["discovered_by"] = sorted(
            x for x in source["discovered_by"]
            if x
        )

        source["matched_keywords"] = sorted(
            source["matched_keywords"]
        )

        source["relevance_score"] = len(
            source["matched_keywords"]
        )

        result.append(source)

    # Highest relevance first.
    result.sort(
        key=lambda x: x["relevance_score"],
        reverse=True,
    )

    return result


# ============================================================
# SEE-MORE EXPANSION
# ============================================================

def expand_see_more_buttons(page):
    """
    Expand visible Facebook 'See more' buttons.
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

            buttons = page.query_selector_all(
                selector
            )

            for button in buttons:

                try:

                    if button.is_visible():

                        button.click(
                            timeout=1000
                        )

                        page.wait_for_timeout(
                            250
                        )

                except Exception:
                    continue

        except Exception:
            continue


# ============================================================
# POST EXTRACTION
# ============================================================

def extract_post_elements(page):
    """
    Find likely Facebook post containers.
    """

    articles = page.query_selector_all(
        'div[role="article"]'
    )

    if articles:
        return articles

    return page.query_selector_all(
        'div[data-ad-comet-preview="message"]'
    )


# ============================================================
# SCRAPE ONE SOURCE
# ============================================================

def scrape_target_posts(
    page,
    target_info: dict,
    max_scrolls: int,
    out_file,
) -> int:

    target_url = target_info["url"]
    title = target_info["title"]

    print(
        f"\nScraping: {title}"
    )

    print(
        f"URL: {target_url}"
    )

    saved_count = 0

    extracted_posts = set()

    try:

        page.goto(
            target_url,
            wait_until="domcontentloaded",
            timeout=60000,
        )

        page.wait_for_timeout(4000)

    except Exception as e:

        print(
            f"  Could not load source: {e}"
        )

        return 0

    for scroll_index in range(max_scrolls):

        print(
            f"  Scroll "
            f"{scroll_index + 1}/{max_scrolls}"
        )

        # ----------------------------------------------------
        # Expand posts
        # ----------------------------------------------------

        expand_see_more_buttons(page)

        # ----------------------------------------------------
        # Find posts
        # ----------------------------------------------------

        elements = extract_post_elements(page)

        print(
            f"  Found {len(elements)} post elements"
        )

        for element in elements:

            try:

                text = normalize_text(
                    element.inner_text()
                )

                if not text:
                    continue

                if len(text) < 15:
                    continue

                # Avoid processing the same post again.
                if text in extracted_posts:
                    continue

                extracted_posts.add(text)

                # ------------------------------------------------
                # Hadith trigger filtering
                # ------------------------------------------------

                triggers = matched_hadith_triggers(
                    text
                )

                if not triggers:
                    continue

                record = {

                    "timestamp_scraped": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ",
                        time.gmtime(),
                    ),

                    "source_title": title,

                    "source_url": target_url,

                    "source_type": target_info.get(
                        "source_type",
                        "unknown",
                    ),

                    "source_matched_keywords":
                        target_info.get(
                            "matched_keywords",
                            [],
                        ),

                    "source_relevance_score":
                        target_info.get(
                            "relevance_score",
                            0,
                        ),

                    "discovered_by":
                        target_info.get(
                            "discovered_by",
                            [],
                        ),

                    "hadith_trigger_count":
                        len(triggers),

                    "raw_text": text,
                }

                out_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                out_file.flush()

                saved_count += 1

                print(
                    f"    MATCH #{saved_count}"
                )

            except Exception:
                continue

        # ----------------------------------------------------
        # Infinite scroll
        # ----------------------------------------------------

        try:

            page.evaluate(
                "window.scrollBy(0, 1200);"
            )

        except Exception:
            pass

        page.wait_for_timeout(3000)

    return saved_count


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_targeted_pipeline(
    search_queries: list[str],
    max_results_per_keyword: int = 10,
    max_scrolls_per_source: int = 8,
    out_path: str = DEFAULT_OUTPUT,
):

    session_path = Path(SESSION_FILE)

    if not session_path.exists():

        print(
            f"\nERROR: '{SESSION_FILE}' not found."
        )

        print(
            "Run your Facebook session-saving script first."
        )

        return

    output_path = Path(out_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    discovered_sources = []

    # ========================================================
    # PLAYWRIGHT
    # ========================================================

    with sync_playwright() as p:

        browser = None
        context = None

        try:

            # ------------------------------------------------
            # Browser
            # ------------------------------------------------

            browser = p.chromium.launch(
                headless=False
            )

            context = browser.new_context(

                storage_state=str(
                    session_path
                ),

                viewport={
                    "width": 1280,
                    "height": 900,
                },
            )

            page = context.new_page()

            # =================================================
            # STEP 1: SOURCE DISCOVERY
            # =================================================

            print(
                "\n"
                + "=" * 70
            )

            print(
                "STEP 1 — FACEBOOK SOURCE DISCOVERY"
            )

            print(
                "=" * 70
            )

            for query in search_queries:

                # Search Pages
                pages = discover_targets(
                    page=page,
                    query=query,
                    search_type="pages",
                    max_results=max_results_per_keyword,
                )

                discovered_sources.extend(
                    pages
                )

                # Search Groups
                groups = discover_targets(
                    page=page,
                    query=query,
                    search_type="groups",
                    max_results=max_results_per_keyword,
                )

                discovered_sources.extend(
                    groups
                )

            # =================================================
            # STEP 2: MERGE / DEDUPLICATE
            # =================================================

            unique_sources = merge_sources(
                discovered_sources
            )

            print(
                "\n"
                + "=" * 70
            )

            print(
                "DISCOVERED SOURCES"
            )

            print(
                "=" * 70
            )

            print(
                f"Raw discoveries: "
                f"{len(discovered_sources)}"
            )

            print(
                f"Unique sources: "
                f"{len(unique_sources)}"
            )

            for index, source in enumerate(
                unique_sources,
                start=1,
            ):

                print(
                    f"\n[{index}] "
                    f"{source['title']}"
                )

                print(
                    f"    Type: "
                    f"{source['source_type']}"
                )

                print(
                    f"    Score: "
                    f"{source['relevance_score']}"
                )

                print(
                    f"    Keywords: "
                    f"{source['matched_keywords']}"
                )

                print(
                    f"    URL: "
                    f"{source['url']}"
                )

            # =================================================
            # STEP 3: SCRAPING
            # =================================================

            print(
                "\n"
                + "=" * 70
            )

            print(
                "STEP 2 — HADITH CLAIM SCRAPING"
            )

            print(
                "=" * 70
            )

            total_saved = 0

            with open(
                output_path,
                "a",
                encoding="utf-8",
            ) as out_file:

                for index, source in enumerate(
                    unique_sources,
                    start=1,
                ):

                    print(
                        f"\n"
                        f"Source {index}/"
                        f"{len(unique_sources)}"
                    )

                    saved = scrape_target_posts(

                        page=page,

                        target_info=source,

                        max_scrolls=
                            max_scrolls_per_source,

                        out_file=out_file,
                    )

                    total_saved += saved

            # =================================================
            # SUMMARY
            # =================================================

            print(
                "\n"
                + "=" * 70
            )

            print(
                "PIPELINE COMPLETE"
            )

            print(
                "=" * 70
            )

            print(
                f"Unique sources: "
                f"{len(unique_sources)}"
            )

            print(
                f"Hadith candidates: "
                f"{total_saved}"
            )

            print(
                f"Output: "
                f"{output_path}"
            )

        finally:

            # ------------------------------------------------
            # IMPORTANT:
            # Always close context/browser even if scraping
            # raises an exception.
            # ------------------------------------------------

            try:

                if context is not None:
                    context.close()

            except Exception:
                pass

            try:

                if browser is not None:
                    browser.close()

            except Exception:
                pass


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # IMPORTANT:
    #
    # These are INDIVIDUAL search terms.
    #
    # No manually constructed combinations.
    # ========================================================

    SEARCH_QUERIES = [

        # -------------------------
        # Bangla
        # -------------------------

        "ইসলাম",
        "ইসলামী",
        "ইসলামিক",
        "মুসলিম",
        "মুসলমান",
        "সুন্নাহ",
        "সুন্নাত",
        "হাদিস",
        "হাদীস",
        "সহীহ",
        "সহিহ",
        "বুখারী",
        "বুখারি",
        "দ্বীন",
        "উম্মাহ",
        "দাওয়াত",
        "বাংলা",

        # -------------------------
        # English
        # -------------------------

        "islam",
        "islamic",
        "muslim",
        "sunnah",
        "sunnat",
        "hadith",
        "hadeeth",
        "sahih",
        "bukhari",
        "deen",
        "ummah",
        "dawah",
        "bangla",
        "bengali",
    ]

    run_targeted_pipeline(

        search_queries=SEARCH_QUERIES,

        max_results_per_keyword=10,

        max_scrolls_per_source=8,

        out_path=(
            "data/raw/"
            "fb_targeted_hadith_claims.jsonl"
        ),
    )