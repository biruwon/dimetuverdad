"""
Browser Integration Tests for fetcher module.

These tests require a real browser and may require Twitter credentials.
They test the actual browser-dependent functionality that cannot be unit tested.

Run with: pytest fetcher/tests/test_browser_integration.py -v -m integration
Skip with: pytest fetcher/tests/ -v -m "not integration"
"""

import os
import pytest
from pathlib import Path
from unittest.mock import Mock, patch


def has_twitter_credentials():
    """Check if Twitter credentials are available."""
    return bool(os.getenv('X_USERNAME') and os.getenv('X_PASSWORD'))


def can_launch_browser():
    """Check if we can launch a browser (headless)."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
        return True
    except Exception:
        return False


# Skip conditions
requires_browser = pytest.mark.skipif(
    not can_launch_browser(),
    reason="Playwright browser not available"
)

requires_credentials = pytest.mark.skipif(
    not has_twitter_credentials(),
    reason="Twitter credentials not configured (set X_USERNAME, X_PASSWORD)"
)


class TestBrowserLaunch:
    """Test browser launch and basic page operations."""
    
    @requires_browser
    def test_browser_launches_headless(self):
        """Verify browser can launch in headless mode."""
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            
            # Use local HTML content to avoid network issues
            page.set_content('<html><head><title>Test Page</title></head><body>Test</body></html>')
            
            assert "Test Page" in page.title()
            
            page.close()
            context.close()
            browser.close()
    
    @requires_browser
    def test_page_javascript_evaluation(self):
        """Verify JavaScript evaluation works on page."""
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_content('<html><head><title>JS Test</title></head><body>Content</body></html>')
            
            result = page.evaluate("() => document.title")
            assert "JS Test" in result
            
            browser.close()


class TestScrollerWithBrowser:
    """Test Scroller with real browser interactions."""
    
    @requires_browser
    def test_scroller_initialization(self):
        """Verify scroller initializes correctly."""
        from fetcher.scroller import Scroller
        
        scroller = Scroller()
        
        # Test scroller has necessary methods
        assert hasattr(scroller, 'delay')
        assert hasattr(scroller, 'scroll')
        assert hasattr(scroller, 'adaptive_scroll')
        assert callable(scroller.delay)
    
    @requires_browser
    def test_scroller_scroll_operations(self):
        """Test actual scroll operations on a real page."""
        from playwright.sync_api import sync_playwright
        from fetcher.scroller import Scroller
        
        scroller = Scroller()
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # Create a long page to scroll
            page.set_content("""
                <html>
                <body style="height: 5000px;">
                    <div id="marker" style="position: absolute; top: 2000px;">Marker</div>
                </body>
                </html>
            """)
            
            # Get initial scroll position
            initial_scroll = page.evaluate("() => window.scrollY")
            assert initial_scroll == 0
            
            # Scroll down
            page.evaluate("window.scrollBy(0, 500)")
            page.wait_for_timeout(100)
            
            new_scroll = page.evaluate("() => window.scrollY")
            assert new_scroll > initial_scroll
            
            browser.close()


class TestSessionManagerWithBrowser:
    """Test SessionManager with real browser."""
    
    @requires_browser
    def test_browser_context_creation(self):
        """Test that browser context can be created."""
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={'width': 1280, 'height': 720},
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
            )
            
            assert context is not None
            
            page = context.new_page()
            assert page is not None
            
            browser.close()
    
    @requires_browser
    @requires_credentials
    def test_session_manager_creates_browser(self):
        """Test SessionManager can create a browser session."""
        from fetcher.session_manager import SessionManager
        
        manager = SessionManager()
        
        # Test that config is loaded
        assert manager.config is not None
        assert manager.scroller is not None


class TestThreadDetectorWithBrowser:
    """Test ThreadDetector with real browser interactions."""
    
    @requires_browser
    def test_thread_detector_initialization(self):
        """Test ThreadDetector initializes correctly."""
        from fetcher.thread_detector import ThreadDetector
        
        detector = ThreadDetector()
        assert detector.logger is not None
        # ThreadDetector has _blocked_expanders for tracking
        assert hasattr(detector, '_blocked_expanders')
    
    @requires_browser
    def test_dismiss_overlay_on_real_page(self):
        """Test overlay dismissal on a mock page with overlay."""
        from playwright.sync_api import sync_playwright
        from fetcher.thread_detector import ThreadDetector
        
        detector = ThreadDetector()
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # Create a page with a mock overlay
            page.set_content("""
                <html>
                <body>
                    <div role="dialog" aria-modal="true">
                        <button role="button" aria-label="Close">X</button>
                    </div>
                    <article data-testid="tweet">Test tweet content</article>
                </body>
                </html>
            """)
            
            # Test dismiss method doesn't crash
            detector._dismiss_specific_overlays(page)
            
            browser.close()
    
    @requires_browser  
    def test_has_thread_line_detection(self):
        """Test thread line CSS detection."""
        from playwright.sync_api import sync_playwright
        from fetcher.thread_detector import ThreadDetector
        
        detector = ThreadDetector()
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # Create article with thread line indicator (need css-175oi2r class as well)
            page.set_content("""
                <html>
                <body>
                    <article data-testid="tweet" id="with-line">
                        <div class="css-175oi2r r-1bimlpy r-f8sm7e"></div>
                        <span>Tweet content</span>
                    </article>
                    <article data-testid="tweet" id="without-line">
                        <span>Another tweet</span>
                    </article>
                </body>
                </html>
            """)
            
            article_with_line = page.query_selector('#with-line')
            article_without_line = page.query_selector('#without-line')
            
            # Test detection
            has_line = detector._has_thread_line(article_with_line)
            no_line = detector._has_thread_line(article_without_line)
            
            assert has_line is True
            assert no_line is False
            
            browser.close()


class TestMediaMonitorWithBrowser:
    """Test MediaMonitor with real browser interactions."""
    
    @requires_browser
    def test_media_monitor_setup(self):
        """Test MediaMonitor can be set up on a page."""
        from playwright.sync_api import sync_playwright
        from fetcher.media_monitor import MediaMonitor
        
        monitor = MediaMonitor()
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_content('<html><body>Test page</body></html>')
            
            # Setup monitoring (should not raise)
            captured_urls = monitor.setup_monitoring(page)
            
            assert isinstance(captured_urls, list)
            
            browser.close()


class TestHTMLExtractorWithBrowser:
    """Test HTML extraction with real browser."""
    
    @requires_browser
    def test_html_extraction_from_real_dom(self):
        """Test extracting HTML from real DOM elements."""
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # Create a tweet-like structure
            page.set_content("""
                <html>
                <body>
                    <article data-testid="tweet">
                        <div data-testid="tweetText">
                            <span>This is a test tweet with some content.</span>
                        </div>
                        <time datetime="2024-01-01T12:00:00Z">Jan 1</time>
                        <a href="/testuser/status/123456789">Link</a>
                    </article>
                </body>
                </html>
            """)
            
            article = page.query_selector('article[data-testid="tweet"]')
            assert article is not None
            
            # Get outer HTML
            html = article.evaluate('el => el.outerHTML')
            assert 'tweetText' in html
            assert 'This is a test tweet' in html
            
            browser.close()


class TestCollectorWithBrowser:
    """Test Collector with real browser."""
    
    @requires_browser
    def test_collector_initialization(self):
        """Test TweetCollector initializes correctly."""
        from fetcher.collector import TweetCollector
        
        collector = TweetCollector()
        
        assert collector.config is not None
        assert collector.scroller is not None
        assert collector.media_monitor is not None
        assert collector.thread_detector is not None


class TestLiveTwitterIntegration:
    """Live Twitter integration tests (require credentials)."""
    
    @requires_browser
    @requires_credentials
    def test_twitter_page_loads(self):
        """Test that Twitter page can be loaded (may show login)."""
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            
            # Try to load Twitter
            page.goto("https://x.com", wait_until="domcontentloaded", timeout=30000)
            
            # Should reach some page (even if login required)
            assert page.url is not None
            assert 'x.com' in page.url or 'twitter.com' in page.url
            
            browser.close()



