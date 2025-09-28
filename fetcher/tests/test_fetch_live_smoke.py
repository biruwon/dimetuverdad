import os
import pytest

from fetcher import fetch_tweets


@pytest.mark.skipif(os.getenv('LIVE_FETCH') != '1', reason='Live fetch tests are disabled by default')
def test_live_fetch_two_posts():
    # This test runs the real fetcher against a public user and should be
    # enabled only manually (set LIVE_FETCH=1). It attempts to fetch 2 posts.
    from playwright.sync_api import sync_playwright

    user = os.getenv('LIVE_FETCH_USER', 'vox_es')

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        results = fetch_tweets.fetch_enhanced_tweets(page, user, max_tweets=2, resume_from_last=False)
        assert isinstance(results, list)
        assert len(results) <= 2
        browser.close()
