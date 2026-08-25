#!/usr/bin/env python
"""Step 1: Save Facebook login session cookies to fb_session.json."""

from playwright.sync_api import sync_playwright

def save_session():
    with sync_playwright() as p:
        # Launch browser in visible mode
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()

        print("Navigating to Facebook...")
        page.goto("https://www.facebook.com")

        print("\nACTION REQUIRED:")
        print("1. Log in to Facebook manually in the opened browser window.")
        print("2. Once logged in and on the home feed, return here and press ENTER.")
        input("Press Enter after completing login...")

        # Export session cookies and localStorage
        context.storage_state(path="fb_session.json")
        print("Successfully saved browser state to 'fb_session.json'.")
        browser.close()

if __name__ == "__main__":
    save_session()