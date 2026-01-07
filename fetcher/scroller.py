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

class Scroller:
    """Handles scrolling and navigation operations for tweet collection."""

    def __init__(self):
        self.config = get_config()

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
        strategies = [
            ("click_retry_button", lambda: (lambda: (page.locator('button:has-text("Retry")').click() if page.locator('button:has-text("Retry")') else None))()),
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