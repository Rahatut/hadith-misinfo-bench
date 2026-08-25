#!/usr/bin/env python3

"""
FACEBOOK HADITH SCRAPER
=======================

Pipeline:

1. Search Facebook Pages using discovery keywords.
2. During testing, use only the first N keywords.
3. Extract candidate Facebook page URLs.
4. Reject Facebook navigation URLs (/reel, /groups, /marketplace, etc.).
5. Deduplicate discovered pages.
6. Save discovered pages to JSON.
7. Visit each discovered page.
8. Scan posts for Hadith-related keywords.
9. Save matching posts as JSONL.

IMPORTANT:
    This is deliberately configured for a small test first.

    Current discovery keywords:
        islam
        islamic

    Increase TEST_KEYWORDS later after discovery works.
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

DISCOVERY_OUTPUT = (
    "data/raw/discovered_facebook_pages.json"
)

CLAIM_OUTPUT = (
    "data/raw/fb_hadith_candidates.jsonl"
)

# ------------------------------------------------------------
# Testing parameters
# ------------------------------------------------------------

# Only the first N discovery keywords will be used.
TEST_KEYWORDS = 2

# Maximum number of pages discovered for EACH keyword.
MAX_PAGES_PER_KEYWORD = 3

# Number of scrolls performed on each page.
MAX_SCROLLS_PER_PAGE = 5

# Wait time after Facebook search loads.
SEARCH_WAIT_MS = 8000

# Wait time after opening an individual page.
PAGE_WAIT_MS = 5000

# Wait time between scrolls.
SCROLL_WAIT_MS = 3000


# ============================================================
# DISCOVERY KEYWORDS
# ============================================================

# Keep the entire list.
#
# During testing, only:
#
#     DISCOVERY_KEYWORDS[:TEST_KEYWORDS]
#
# is used.

DISCOVERY_KEYWORDS = [

    # English
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

    # Bangla
    "ইসলাম",
    "ইসলামী",
    "মুসলিম",
    "মুসলমান",
    "মুমিন",
    "সুন্নাহ",
    "সুন্নাত",
    "হাদিস",
    "হাদীস",
    "সহীহ",
    "বুখারী",
    "দ্বীন",
    "উম্মাহ",
    "দাওয়াত",
    "বাংলা",
]


# ============================================================
# HADITH CONTENT KEYWORDS
# ============================================================

# These are NOT discovery keywords.
#
# These are used AFTER a page has been discovered.
#
# A post containing one or more of these terms becomes a
# candidate for further Hadith claim processing.

HADITH_PATTERNS = [

    # --------------------------------------------------------
    # Bangla attribution
    # --------------------------------------------------------

    r"রাসূলুল্লাহ",
    r"রাসুলুল্লাহ",

    r"রাসূল",
    r"রাসুল",

    r"নবীজী",
    r"নবীজি",

    r"নবী",

    r"হযরত",
    r"হজরত",

    # --------------------------------------------------------
    # Bangla Hadith references
    # --------------------------------------------------------

    r"হাদিস",
    r"হাদীস",

    r"হাদিসে",
    r"হাদীসে",

    r"হাদিসে বর্ণিত",
    r"হাদীসে বর্ণিত",

    r"হাদিসে এসেছে",
    r"হাদীসে এসেছে",

    r"সহীহ বুখারী",
    r"সহিহ বুখারি",

    r"সহীহ মুসলিম",
    r"সহিহ মুসলিম",

    r"বুখারী",
    r"বুখারি",

    r"মুসলিম শরীফ",
    r"মুসলিম শরিফ",

    # --------------------------------------------------------
    # Arabic attribution
    # --------------------------------------------------------

    r"ﷺ",

    r"صلى الله عليه وسلم",

    # --------------------------------------------------------
    # English
    # --------------------------------------------------------

    r"\bprophet\b",
    r"\bprophet muhammad\b",

    r"\bmuhammad\b",

    r"\bmessenger of allah\b",

    r"\brasulullah\b",

    r"\bhadith\b",
    r"\bhadeeth\b",

    r"\bsahih bukhari\b",
    r"\bsahih muslim\b",

]


# ============================================================
# FACEBOOK NAVIGATION URLS TO REJECT
# ============================================================

# Facebook search pages contain many unrelated navigation links.
#
# For example:
#
#     /reel
#     /marketplace
#     /groups
#
# These are NOT search results.

BLOCKED_PREFIXES = [

    "/reel",
    "/reels",

    "/marketplace",

    "/groups",

    "/events",

    "/watch",

    "/gaming",

    "/messages",

    "/notifications",

    "/settings",

    "/help",

    "/login",

    "/home",

    "/friends",

    "/photo",
    "/photos",

    "/videos",

    "/stories",

    "/hashtag",

    "/search",

]


# ============================================================
# HADITH KEYWORD CHECK
# ============================================================

def contains_hadith_keyword(text: str) -> bool:
    """
    Check whether a post contains a Hadith-related keyword.
    """

    if not text:
        return False

    for pattern in HADITH_PATTERNS:

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            return True

    return False


# ============================================================
# URL NORMALIZATION
# ============================================================

def clean_facebook_url(url: str) -> str:
    """
    Normalize a Facebook URL.

    Removes:
        query parameters
        fragments
        tracking parameters
    """

    if not url:
        return ""

    # Relative Facebook URLs.
    if url.startswith("/"):
        url = (
            "https://www.facebook.com"
            + url
        )

    try:

        parsed = urllib.parse.urlparse(url)

        return urllib.parse.urlunparse(
            (
                "https",
                "www.facebook.com",
                parsed.path.rstrip("/"),
                "",
                "",
                "",
            )
        )

    except Exception:
        return ""


# ============================================================
# PAGE URL VALIDATION
# ============================================================

def is_real_page_url(url: str) -> bool:
    """
    Determine whether a Facebook URL looks like a real
    page/profile URL rather than Facebook navigation.

    Examples rejected:

        facebook.com/reel
        facebook.com/groups
        facebook.com/marketplace

    Examples potentially accepted:

        facebook.com/pages/...
        facebook.com/SomePageName
    """

    if not url:
        return False

    try:

        parsed = urllib.parse.urlparse(url)

        path = parsed.path.rstrip("/").lower()

    except Exception:
        return False

    if not path:
        return False

    # --------------------------------------------------------
    # Reject search itself
    # --------------------------------------------------------

    if path.startswith("/search"):
        return False

    # --------------------------------------------------------
    # Reject known Facebook navigation
    # --------------------------------------------------------

    for prefix in BLOCKED_PREFIXES:

        if path.startswith(prefix):
            return False

    # --------------------------------------------------------
    # Explicit Facebook page URL
    # --------------------------------------------------------

    if path.startswith("/pages/"):
        return True

    # --------------------------------------------------------
    # Generic one-component Facebook page/profile
    #
    # Example:
    #
    # /SomeIslamicPage
    #
    # We allow this because Facebook often uses username URLs.
    # --------------------------------------------------------

    components = [
        component
        for component in path.split("/")
        if component
    ]

    if len(components) == 1:
        return True

    return False


# ============================================================
# EXTRACT TITLE
# ============================================================

def get_link_title(link) -> str:
    """
    Safely extract visible text from a Facebook link.
    """

    try:

        title = link.inner_text()

    except Exception:

        title = ""

    if not title:
        return ""

    # Collapse newlines / excessive whitespace.
    title = re.sub(
        r"\s+",
        " ",
        title,
    )

    return title.strip()


# ============================================================
# FACEBOOK PAGE DISCOVERY
# ============================================================

def discover_pages(
    page,
    keyword: str,
    max_pages: int = 3,
):
    """
    Search Facebook Pages for one keyword.

    The function is intentionally diagnostic:
    it reports what Facebook actually returns instead
    of silently discarding everything.
    """

    encoded_keyword = urllib.parse.quote(
        keyword
    )

    search_url = (
        "https://www.facebook.com/search/pages/"
        f"?q={encoded_keyword}"
    )

    print()
    print("=" * 70)
    print(
        f"SEARCHING FACEBOOK PAGES FOR: {keyword}"
    )
    print("=" * 70)

    print(f"URL: {search_url}")

    # --------------------------------------------------------
    # Navigate
    # --------------------------------------------------------

    try:

        page.goto(
            search_url,
            wait_until="domcontentloaded",
            timeout=60000,
        )

    except Exception as e:

        print(
            f"Navigation warning: {e}"
        )

    # Facebook renders the search page dynamically.
    print(
        f"Waiting {SEARCH_WAIT_MS / 1000:.1f}s "
        "for Facebook results..."
    )

    page.wait_for_timeout(
        SEARCH_WAIT_MS
    )

    print(
        f"Current URL: {page.url}"
    )

    # --------------------------------------------------------
    # Diagnostic screenshot
    # --------------------------------------------------------

    screenshot_name = (
        "facebook_search_"
        + re.sub(
            r"[^a-zA-Z0-9_-]",
            "_",
            keyword,
        )
        + ".png"
    )

    try:

        page.screenshot(
            path=screenshot_name,
            full_page=False,
        )

        print(
            f"Screenshot saved: "
            f"{screenshot_name}"
        )

    except Exception as e:

        print(
            f"Screenshot warning: {e}"
        )

    # --------------------------------------------------------
    # Find anchors
    # --------------------------------------------------------

    try:

        links = page.locator(
            "a"
        ).all()

    except Exception as e:

        print(
            f"Could not inspect anchors: {e}"
        )

        return []

    print(
        f"Total <a> elements found: "
        f"{len(links)}"
    )

    discovered = []

    seen_urls = set()

    # --------------------------------------------------------
    # Inspect links
    # --------------------------------------------------------

    for link_number, link in enumerate(
        links,
        start=1,
    ):

        try:

            href = link.get_attribute(
                "href"
            )

            if not href:
                continue

            clean_url = clean_facebook_url(
                href
            )

            if not clean_url:
                continue

            # ------------------------------------------------
            # Check page URL
            # ------------------------------------------------

            if not is_real_page_url(
                clean_url
            ):
                continue

            # ------------------------------------------------
            # Deduplicate
            # ------------------------------------------------

            if clean_url in seen_urls:
                continue

            seen_urls.add(clean_url)

            # ------------------------------------------------
            # Extract title
            # ------------------------------------------------

            title = get_link_title(
                link
            )

            if not title:
                title = "(no visible title)"

            # ------------------------------------------------
            # Print
            # ------------------------------------------------

            print()
            print(
                f"Candidate page #{len(discovered) + 1}"
            )

            print(
                f"  Title : {title[:200]}"
            )

            print(
                f"  URL   : {clean_url}"
            )

            # ------------------------------------------------
            # Store
            # ------------------------------------------------

            discovered.append(
                {
                    "keyword": keyword,
                    "title": title,
                    "url": clean_url,
                }
            )

            print(
                "  ✓ ACCEPTED"
            )

            # ------------------------------------------------
            # Limit
            # ------------------------------------------------

            if (
                len(discovered)
                >= max_pages
            ):
                break

        except Exception:
            continue

    print()
    print(
        f"Found {len(discovered)} "
        f"candidate page(s) for "
        f"'{keyword}'."
    )

    return discovered


# ============================================================
# EXPAND "SEE MORE"
# ============================================================

def expand_see_more(page):
    """
    Expand visible Facebook 'See more' buttons.
    """

    selectors = [

        'div[role="button"]:has-text("See more")',

        'div[role="button"]:has-text("See More")',

        'span:has-text("See more")',

        'span:has-text("See More")',

        'div[role="button"]:has-text("আরও দেখুন")',

        'span:has-text("আরও দেখুন")',

    ]

    for selector in selectors:

        try:

            buttons = page.locator(
                selector
            ).all()

        except Exception:
            continue

        for button in buttons:

            try:

                if button.is_visible():

                    button.click(
                        timeout=1000
                    )

                    page.wait_for_timeout(
                        200
                    )

            except Exception:
                pass


# ============================================================
# SCRAPE ONE FACEBOOK PAGE
# ============================================================

def scrape_page(
    page,
    source,
    output_file,
    max_scrolls=5,
):
    """
    Visit one discovered page and search its posts
    for Hadith-related content.
    """

    title = source["title"]
    url = source["url"]

    print()
    print("=" * 70)
    print("SCRAPING PAGE")
    print("=" * 70)

    print(
        f"Title : {title}"
    )

    print(
        f"URL   : {url}"
    )

    # --------------------------------------------------------
    # Navigate
    # --------------------------------------------------------

    try:

        page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=60000,
        )

    except Exception as e:

        print(
            f"Navigation warning: {e}"
        )

    page.wait_for_timeout(
        PAGE_WAIT_MS
    )

    # --------------------------------------------------------
    # Screenshot
    # --------------------------------------------------------

    try:

        safe_name = re.sub(
            r"[^a-zA-Z0-9_-]",
            "_",
            title,
        )

        if not safe_name:
            safe_name = "facebook_page"

        screenshot_path = (
            f"{safe_name}_page.png"
        )

        page.screenshot(
            path=screenshot_path,
            full_page=False,
        )

        print(
            f"Page screenshot: "
            f"{screenshot_path}"
        )

    except Exception:
        pass

    # --------------------------------------------------------
    # Tracking
    # --------------------------------------------------------

    seen_posts = set()

    saved_count = 0

    # --------------------------------------------------------
    # Scroll through page
    # --------------------------------------------------------

    for scroll_number in range(
        max_scrolls
    ):

        print()
        print(
            f"  Scroll "
            f"{scroll_number + 1}/"
            f"{max_scrolls}"
        )

        # ----------------------------------------------------
        # Expand posts
        # ----------------------------------------------------

        expand_see_more(
            page
        )

        # ----------------------------------------------------
        # Find Facebook articles
        # ----------------------------------------------------

        try:

            articles = page.locator(
                'div[role="article"]'
            ).all()

        except Exception:

            articles = []

        print(
            f"  Articles currently visible: "
            f"{len(articles)}"
        )

        # ----------------------------------------------------
        # Inspect posts
        # ----------------------------------------------------

        for article in articles:

            try:

                text = article.inner_text()

            except Exception:

                continue

            if not text:
                continue

            # Normalize whitespace.
            text = re.sub(
                r"\s+",
                " ",
                text,
            ).strip()

            # Ignore tiny elements.
            if len(text) < 20:
                continue

            # Avoid processing same post repeatedly.
            if text in seen_posts:
                continue

            seen_posts.add(text)

            # ------------------------------------------------
            # Hadith detection
            # ------------------------------------------------

            if not contains_hadith_keyword(
                text
            ):
                continue

            saved_count += 1

            record = {
                "timestamp_scraped": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(),
                ),

                "source_title": title,

                "source_url": url,

                "raw_text": text,
            }

            output_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            output_file.flush()

            print()
            print(
                f"    ✓ HADITH CANDIDATE "
                f"#{saved_count}"
            )

            print(
                f"      {text[:250]}"
            )

        # ----------------------------------------------------
        # Scroll
        # ----------------------------------------------------

        try:

            page.evaluate(
                "window.scrollBy(0, 1200)"
            )

        except Exception:
            pass

        page.wait_for_timeout(
            SCROLL_WAIT_MS
        )

    print()
    print(
        f"Finished page."
    )

    print(
        f"Hadith candidates: "
        f"{saved_count}"
    )

    return saved_count


# ============================================================
# SAVE DISCOVERED PAGES
# ============================================================

def save_discovered_pages(
    pages,
    output_path,
):
    """
    Save discovered page metadata.
    """

    output = Path(
        output_path
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            pages,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline():

    print()
    print("=" * 70)
    print("FACEBOOK HADITH SCRAPER")
    print("=" * 70)

    # --------------------------------------------------------
    # Session
    # --------------------------------------------------------

    if not Path(
        SESSION_FILE
    ).exists():

        print()
        print(
            "ERROR:"
        )

        print(
            f"'{SESSION_FILE}' "
            "was not found."
        )

        print()
        print(
            "Run your Facebook session "
            "saving script first."
        )

        return

    # --------------------------------------------------------
    # Test configuration
    # --------------------------------------------------------

    test_keywords = (
        DISCOVERY_KEYWORDS[
            :TEST_KEYWORDS
        ]
    )

    print(
        f"Testing first "
        f"{len(test_keywords)} "
        f"discovery keywords:"
    )

    for keyword in test_keywords:

        print(
            f"  - {keyword}"
        )

    print()

    print(
        f"Maximum pages per keyword: "
        f"{MAX_PAGES_PER_KEYWORD}"
    )

    print(
        f"Maximum scrolls per page: "
        f"{MAX_SCROLLS_PER_PAGE}"
    )

    print()
    print("=" * 70)

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    Path(
        DISCOVERY_OUTPUT
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(
        CLAIM_OUTPUT
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Launch Playwright
    # --------------------------------------------------------

    with sync_playwright() as playwright:

        browser = playwright.chromium.launch(
            headless=False
        )

        context = browser.new_context(
            storage_state=SESSION_FILE,

            viewport={
                "width": 1280,
                "height": 900,
            },

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

        # ----------------------------------------------------
        # STEP 1
        # DISCOVER PAGES
        # ----------------------------------------------------

        discovered_pages = []

        for keyword in test_keywords:

            results = discover_pages(
                page=page,

                keyword=keyword,

                max_pages=(
                    MAX_PAGES_PER_KEYWORD
                ),
            )

            discovered_pages.extend(
                results
            )

        # ----------------------------------------------------
        # STEP 2
        # DEDUPLICATE
        # ----------------------------------------------------

        unique_pages_dict = {}

        for source in discovered_pages:

            url = source["url"]

            if url not in unique_pages_dict:

                unique_pages_dict[
                    url
                ] = source

        unique_pages = list(
            unique_pages_dict.values()
        )

        # ----------------------------------------------------
        # Print discovery result
        # ----------------------------------------------------

        print()
        print("=" * 70)

        print(
            f"DISCOVERED "
            f"{len(unique_pages)} "
            f"UNIQUE PAGES"
        )

        print("=" * 70)

        for index, source in enumerate(
            unique_pages,
            start=1,
        ):

            print(
                f"{index}. "
                f"{source['title']} "
                f"-> "
                f"{source['url']}"
            )

        # ----------------------------------------------------
        # Save pages
        # ----------------------------------------------------

        save_discovered_pages(
            unique_pages,
            DISCOVERY_OUTPUT,
        )

        print()
        print(
            f"Saved page list to: "
            f"{DISCOVERY_OUTPUT}"
        )

        # ----------------------------------------------------
        # STEP 3
        # SCRAPE DISCOVERED PAGES
        # ----------------------------------------------------

        total_candidates = 0

        with open(
            CLAIM_OUTPUT,
            "a",
            encoding="utf-8",
        ) as output_file:

            for source in unique_pages:

                count = scrape_page(
                    page=page,

                    source=source,

                    output_file=output_file,

                    max_scrolls=(
                        MAX_SCROLLS_PER_PAGE
                    ),
                )

                total_candidates += count

        # ----------------------------------------------------
        # Close browser
        # ----------------------------------------------------

        context.close()

        browser.close()

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SCRAPING COMPLETE")
    print("=" * 70)

    print(
        f"Pages discovered: "
        f"{len(unique_pages)}"
    )

    print(
        f"Hadith candidates collected: "
        f"{total_candidates}"
    )

    print(
        f"Output: "
        f"{CLAIM_OUTPUT}"
    )

    print(
        f"Page list: "
        f"{DISCOVERY_OUTPUT}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_pipeline()