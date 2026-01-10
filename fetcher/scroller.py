"""
Scrolling and navigation logic for the fetcher module.

Handles human-like scrolling patterns, recovery strategies, and page navigation.
"""

import random
import time
from typing import Optional, Callable, List, Dict, Any

from .config import get_config
from .logging_config import get_logger

logger = get_logger('scroller')

# Performance optimization: minimum delay to avoid rate limiting
MIN_ADAPTIVE_DELAY = 0.3  # seconds

class Scroller:
    """Handles scrolling and navigation operations for tweet collection."""

    def __init__(self):
        self.config = get_config()
        # Track adaptive scroll performance
        self._adaptive_scroll_stats = {
            'total_scrolls': 0,
            'content_loaded_fast': 0,  # Loaded before timeout
            'content_loaded_timeout': 0,  # Hit timeout waiting
            'total_wait_time': 0.0,
        }

    def delay(self, min_seconds: Optional[float] = None, max_seconds: Optional[float] = None) -> None:
        """
        Add human-like random delays.

        Args:
            min_seconds: Minimum delay in seconds
            max_seconds: Maximum delay in seconds
        """
        if min_seconds is None:
            min_seconds = self.config.min_human_delay
        if max_seconds is None:
            max_seconds = self.config.max_human_delay

        delay = random.uniform(min_seconds, max_seconds)
        time.sleep(delay)

    def random_scroll_pattern(self, page, deep_scroll: bool = False) -> None:
        """
        Implement human-like scrolling patterns.

        Args:
            page: Playwright page object
            deep_scroll: Whether to use aggressive scrolling for older content
        """
        # Use conservative scroll amounts to avoid large jumps that skip visible tweets
        if deep_scroll:
            # Slightly larger steps for deep scroll but still conservative
            scroll_amounts = [
                900 + random.randint(0, 200),   # 900-1100px
                1200 + random.randint(0, 300),  # 1200-1500px
                800 + random.randint(0, 200),   # 800-1000px
            ]
        else:
            # Gentle scroll amounts for normal operation
            scroll_amounts = [
                400 + random.randint(0, 200),   # 400-600px
                600 + random.randint(0, 200),   # 600-800px
                800 + random.randint(0, 200),   # 800-1000px
            ]

        # Perform a small incremental scroll instead of jumping to the bottom
        scroll_js = f"window.scrollBy(0, {random.choice(scroll_amounts)})"
        page.evaluate(scroll_js)

        # Slight upward adjustment occasionally to mimic natural browsing
        if random.random() < 0.08:  # 8% chance
            back_scroll = 50 + random.randint(0, 100)
            page.evaluate(f"window.scrollBy(0, -{back_scroll})")

        self.delay(1.0, 2.5)

    def adaptive_scroll(self, page, deep_scroll: bool = False) -> float:
        """
        Scroll with adaptive delay - wait only until new content appears.
        
        This is a performance-optimized version of random_scroll_pattern that
        waits for content to load instead of using fixed delays.
        
        Args:
            page: Playwright page object
            deep_scroll: Whether to use aggressive scrolling for older content
            
        Returns:
            float: Time spent waiting for content (for performance tracking)
        """
        start_time = time.time()
        self._adaptive_scroll_stats['total_scrolls'] += 1
        
        # Count current articles before scroll
        try:
            prev_count = page.evaluate(
                'document.querySelectorAll("article[data-testid=\\"tweet\\"]").length'
            )
        except Exception:
            prev_count = 0
        
        # Choose scroll amount
        if deep_scroll:
            scroll_amounts = [900, 1100, 1200, 1000]
        else:
            scroll_amounts = [500, 600, 700, 800]
        
        scroll_amount = random.choice(scroll_amounts) + random.randint(0, 200)
        page.evaluate(f"window.scrollBy(0, {scroll_amount})")
        
        # Wait for new content with short timeout (instead of fixed delay)
        try:
            page.wait_for_function(
                f'document.querySelectorAll("article[data-testid=\\"tweet\\"]").length > {prev_count}',
                timeout=2000  # Max 2 seconds instead of 1.5-3 seconds fixed
            )
            self._adaptive_scroll_stats['content_loaded_fast'] += 1
        except Exception:
            # Content didn't load in time - that's OK, continue anyway
            self._adaptive_scroll_stats['content_loaded_timeout'] += 1
        
        # Minimum delay to avoid rate limiting (human-like)
        time.sleep(MIN_ADAPTIVE_DELAY)
        
        # Occasional upward adjustment (less frequent for speed)
        if random.random() < 0.05:  # 5% chance
            back_scroll = 30 + random.randint(0, 50)
            page.evaluate(f"window.scrollBy(0, -{back_scroll})")
        
        elapsed = time.time() - start_time
        self._adaptive_scroll_stats['total_wait_time'] += elapsed
        
        return elapsed

    def get_adaptive_scroll_stats(self) -> Dict[str, Any]:
        """Get statistics about adaptive scroll performance."""
        stats = self._adaptive_scroll_stats.copy()
        if stats['total_scrolls'] > 0:
            stats['avg_wait_time'] = stats['total_wait_time'] / stats['total_scrolls']
            stats['fast_load_rate'] = stats['content_loaded_fast'] / stats['total_scrolls']
        else:
            stats['avg_wait_time'] = 0.0
            stats['fast_load_rate'] = 0.0
        return stats

    def reset_adaptive_scroll_stats(self) -> None:
        """Reset adaptive scroll statistics."""
        self._adaptive_scroll_stats = {
            'total_scrolls': 0,
            'content_loaded_fast': 0,
            'content_loaded_timeout': 0,
            'total_wait_time': 0.0,
        }

    def prefetch_scroll(self, page) -> int:
        """
        P6 Performance Optimization: Trigger loading of next content batch.
        
        Scrolls ahead to trigger network loading of the next batch of tweets,
        so that network I/O happens in parallel with CPU processing of current batch.
        
        This should be called BEFORE processing tweets, then processing happens
        while content loads, and finally wait_for_prefetched_content() is called
        after processing to ensure content is ready.
        
        Args:
            page: Playwright page object
            
        Returns:
            The article count before scrolling (for later comparison)
        """
        # Get current article count
        try:
            prev_count = page.evaluate(
                'document.querySelectorAll("article[data-testid=\\"tweet\\"]").length'
            )
        except Exception:
            prev_count = 0
        
        # Scroll ahead to trigger loading (don't wait)
        scroll_amount = 1500 + random.randint(0, 500)  # Scroll far ahead
        try:
            page.evaluate(f"window.scrollBy(0, {scroll_amount})")
        except Exception as e:
            logger.warning(f"Prefetch scroll failed: {e}")
        
        return prev_count

    def wait_for_prefetched_content(self, page, prev_count: int, timeout_ms: int = 500) -> bool:
        """
        P6 Performance Optimization: Wait for prefetched content with short timeout.
        
        Called after processing is complete to ensure the next batch of content
        is ready. Uses a short timeout since content should already be loading.
        
        Args:
            page: Playwright page object
            prev_count: Article count before prefetch scroll
            timeout_ms: Maximum time to wait in milliseconds (default 500ms)
            
        Returns:
            True if new content loaded, False if timed out
        """
        try:
            page.wait_for_function(
                f'document.querySelectorAll("article[data-testid=\\"tweet\\"]").length > {prev_count}',
                timeout=timeout_ms
            )
            return True
        except Exception:
            # Content didn't load - that's OK, will be handled by normal scroll logic
            return False

    def scroll_back_for_processing(self, page, amount: int = 800) -> None:
        """
        P6 Performance Optimization: Scroll back slightly for processing.
        
        After prefetch scroll, scroll back to ensure visible tweets are
        in the viewport for processing.
        
        Args:
            page: Playwright page object
            amount: Pixels to scroll back (default 800)
        """
        back_amount = amount + random.randint(-100, 100)
        try:
            page.evaluate(f"window.scrollBy(0, -{back_amount})")
            time.sleep(MIN_ADAPTIVE_DELAY)  # Small delay for rendering
        except Exception as e:
            logger.warning(f"Scroll back failed: {e}")

    def event_scroll_cycle(self, page, iteration: int) -> None:
        """
        More varied scrolling patterns to avoid detection.

        Args:
            page: Playwright page object
            iteration: Current iteration number
        """
        # Use conservative incremental scrolls to avoid overshooting
        scroll_patterns = [
            lambda: page.evaluate("window.scrollBy(0, 600 + Math.random() * 200)"),
            lambda: page.evaluate("window.scrollBy(0, 450 + Math.random() * 200)"),
            lambda: page.evaluate("window.scrollBy(0, 700 + Math.random() * 200)"),
            lambda: page.evaluate("window.scrollBy(0, 500 + Math.random() * 200)"),
            lambda: page.evaluate("window.scrollBy(0, 550 + Math.random() * 250)"),
        ]

        try:
            # Vary the pattern based on iteration
            pattern_index = iteration % len(scroll_patterns)
            scroll_patterns[pattern_index]()
        except Exception:
            # Fallback to basic small scroll
            try:
                page.evaluate("window.scrollBy(0, 500 + Math.random() * 200)")
            except Exception:
                pass

        # Variable delays to seem more human
        if iteration % 10 == 0:
            self.delay(2.0, 3.0)  # Longer pause occasionally
        else:
            self.delay(0.8, 1.8)

    def try_recovery_strategies(self, page, attempt_number: int) -> bool:
        """
        Try different recovery strategies when Twitter stops serving content.

        Args:
            page: Playwright page object
            attempt_number: Which recovery attempt this is

        Returns:
            bool: True if recovery successful
        """
        # Safer recovery strategies: try non-disruptive actions first
        def try_click_retry():
            """Try to click retry button with short timeout - don't wait if not found."""
            try:
                retry_btn = page.query_selector('button:has-text("Retry")')
                if retry_btn and retry_btn.is_visible():
                    retry_btn.click()
                    return True
            except Exception:
                pass
            return False
        
        strategies = [
            ("click_retry_button", try_click_retry),
            ("clear_cache", lambda: page.evaluate("localStorage.clear(); sessionStorage.clear();")),
            ("small_random_scroll", lambda: page.evaluate(f"window.scrollBy(0, {200 + random.randint(200, 400)})")),
            ("jump_to_middle", lambda: page.evaluate("window.scrollTo(0, document.body.scrollHeight/2)")),
            ("refresh_page", lambda: page.reload(wait_until="domcontentloaded")),
            ("force_reload_tweets", lambda: page.evaluate("window.location.reload(true)")),
        ]

        if attempt_number <= len(strategies):
            strategy_name, strategy_func = strategies[attempt_number - 1]
            try:
                logger.info(f"Trying recovery strategy {attempt_number}: {strategy_name}")
                # Some strategies are lambdas that perform actions immediately
                strategy_func()
                self.delay(self.config.recovery_delay_min, self.config.recovery_delay_max)

                # Check if we can find tweet elements after recovery
                articles = page.query_selector_all('article[data-testid="tweet"]')
                if articles:
                    logger.info(f"Recovery successful: found {len(articles)} articles")
                    return True
                else:
                    logger.warning("Recovery failed: no articles found")
                    return False
            except Exception as e:
                logger.warning(f"Recovery strategy {strategy_name} failed: {e}")
                return False

        return False

    def check_page_height_change(self, page, last_height: int) -> int:
        """
        Check if page height has changed and log appropriately.

        Args:
            page: Playwright page object
            last_height: Previous page height

        Returns:
            int: Current page height
        """
        try:
            new_height = page.evaluate("document.body.scrollHeight")
            if new_height == last_height:
                logger.debug(f"Page height unchanged ({new_height}px)")
            else:
                logger.debug(f"Page height changed: {last_height}px → {new_height}px")
            return new_height
        except Exception:
            logger.warning("Could not check page height")
            return last_height

    def scroll(self, page, deep_scroll: bool = False) -> None:
        """
        Scroll the page using the appropriate scrolling pattern.

        Args:
            page: Playwright page object
            deep_scroll: Whether to use aggressive scrolling for older content
        """
        self.random_scroll_pattern(page, deep_scroll)

    def aggressive_scroll(self, page, consecutive_empty_scrolls: int) -> None:
        """
        Perform aggressive scrolling when having trouble finding content.

        Args:
            page: Playwright page object
            consecutive_empty_scrolls: Number of consecutive empty scrolls
        """
        # Increase scroll amount cautiously based on consecutive failures
        base_scroll = 800
        multiplier = min(consecutive_empty_scrolls, 3)  # Cap at 3x
        scroll_amount = base_scroll * multiplier

        try:
            # Try a couple of moderate aggressive scrolls
            for _ in range(2):
                page.evaluate(f"window.scrollBy(0, {scroll_amount})")
                # Use slightly longer delays to let content load
                self.delay(0.2, 0.5)
        except Exception:
            # Fallback to basic scroll
            try:
                page.evaluate("window.scrollBy(0, 1000)")
            except Exception:
                pass

# Global scroller instance
scroller = Scroller()

def get_scroller() -> Scroller:
    """Get the global scroller instance."""
    return scroller