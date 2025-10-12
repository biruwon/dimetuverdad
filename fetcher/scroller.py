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
        if deep_scroll:
            # Extra aggressive scrolling for finding much older content
            scroll_amounts = [
                1500 + random.randint(0, 500),   # 1500-2000px
                2000 + random.randint(0, 800),   # 2000-2800px
                1200 + random.randint(0, 600),   # 1200-1800px
                2500 + random.randint(0, 1000)   # 2500-3500px
            ]
        else:
            # More aggressive scroll amounts for finding older content
            scroll_amounts = [
                800 + random.randint(0, 400),    # 800-1200px
                1000 + random.randint(0, 500),   # 1000-1500px
                600 + random.randint(0, 300),    # 600-900px
                1200 + random.randint(0, 600)    # 1200-1800px
            ]

        scroll_js = f"window.scrollBy(0, {random.choice(scroll_amounts)})"
        page.evaluate(scroll_js)

        # Sometimes scroll back up slightly (human behavior) - reduced frequency
        if random.random() < 0.05:  # 5% chance
            back_scroll = 100 + random.randint(0, 150)
            page.evaluate(f"window.scrollBy(0, -{back_scroll})")

        self.delay(1.5, 4.0)

    def event_scroll_cycle(self, page, iteration: int) -> None:
        """
        More varied scrolling patterns to avoid detection.

        Args:
            page: Playwright page object
            iteration: Current iteration number
        """
        scroll_patterns = [
            lambda: page.evaluate("window.scrollBy(0, 800 + Math.random() * 600)"),  # Normal scroll
            lambda: page.keyboard.press('PageDown'),  # Keyboard scroll
            lambda: page.evaluate("window.scrollBy(0, 1200 + Math.random() * 800)"),  # Larger scroll
            lambda: page.evaluate("window.scrollBy(0, 400 + Math.random() * 300)"),  # Smaller scroll
            lambda: page.evaluate("window.scrollTo(0, window.scrollY + 1000)"),  # Absolute positioning
        ]

        try:
            # Vary the pattern based on iteration
            pattern_index = iteration % len(scroll_patterns)
            scroll_patterns[pattern_index]()
        except Exception:
            # Fallback to basic scroll
            try:
                page.evaluate("window.scrollBy(0, 800 + Math.random() * 400)")
            except Exception:
                pass

        # Variable delays to seem more human
        if iteration % 10 == 0:
            self.delay(2.0, 4.0)  # Longer pause occasionally
        else:
            self.delay(0.8, 2.2)

    def try_recovery_strategies(self, page, attempt_number: int) -> bool:
        """
        Try different recovery strategies when Twitter stops serving content.

        Args:
            page: Playwright page object
            attempt_number: Which recovery attempt this is

        Returns:
            bool: True if recovery successful
        """
        strategies = [
            ("refresh_page", lambda: page.reload(wait_until="domcontentloaded")),
            ("clear_cache", lambda: page.evaluate("localStorage.clear(); sessionStorage.clear();")),
            ("jump_to_bottom", lambda: page.evaluate("window.scrollTo(0, document.body.scrollHeight)")),
            ("force_reload_tweets", lambda: page.evaluate("window.location.reload(true)")),
            ("random_scroll_pattern", lambda: page.evaluate(f"window.scrollBy(0, {1000 + random.randint(500, 2000)})")),
        ]

        if attempt_number <= len(strategies):
            strategy_name, strategy_func = strategies[attempt_number - 1]
            try:
                logger.info(f"Trying recovery strategy {attempt_number}: {strategy_name}")
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

# Global scroller instance
scroller = Scroller()

def get_scroller() -> Scroller:
    """Get the global scroller instance."""
    return scroller