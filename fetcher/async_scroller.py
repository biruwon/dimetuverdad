"""
Async scrolling and navigation logic for the fetcher module.

Async version of scroller.py for use with async_playwright.
Handles human-like scrolling patterns, recovery strategies, and page navigation.
"""

import random
import asyncio
from typing import Optional

from .config import get_config
from .logging_config import get_logger

logger = get_logger('async_scroller')

# Performance optimization: minimum delay to avoid rate limiting
MIN_ADAPTIVE_DELAY = 0.3  # seconds


class AsyncScroller:
    """Async version of Scroller - handles scrolling operations for async tweet collection."""

    def __init__(self):
        self.config = get_config()
        self._adaptive_scroll_stats = {
            'total_scrolls': 0,
            'content_loaded_fast': 0,
            'content_loaded_timeout': 0,
            'total_wait_time': 0.0,
        }

    async def delay(self, min_seconds: Optional[float] = None, max_seconds: Optional[float] = None) -> None:
        """Add human-like random delays (async version)."""
        if min_seconds is None:
            min_seconds = self.config.min_human_delay
        if max_seconds is None:
            max_seconds = self.config.max_human_delay

        delay = random.uniform(min_seconds, max_seconds)
        await asyncio.sleep(delay)

    async def random_scroll_pattern(self, page, deep_scroll: bool = False) -> None:
        """Implement human-like scrolling patterns (async version)."""
        if deep_scroll:
            scroll_amounts = [
                900 + random.randint(0, 200),
                1200 + random.randint(0, 300),
                800 + random.randint(0, 200),
            ]
        else:
            scroll_amounts = [
                400 + random.randint(0, 200),
                600 + random.randint(0, 200),
                800 + random.randint(0, 200),
            ]

        scroll_js = f"window.scrollBy(0, {random.choice(scroll_amounts)})"
        await page.evaluate(scroll_js)

        if random.random() < 0.08:
            back_scroll = 50 + random.randint(0, 100)
            await page.evaluate(f"window.scrollBy(0, -{back_scroll})")

    async def adaptive_scroll(
        self,
        page,
        target_count_selector: str = 'article[data-testid="tweet"]',
        prev_count: int = 0,
        deep_scroll: bool = False,
        max_wait: float = 5.0
    ) -> int:
        """
        Scroll and wait for new content to load (async version).
        
        Returns the new count of elements matching target_count_selector.
        """
        await self.random_scroll_pattern(page, deep_scroll=deep_scroll)
        
        self._adaptive_scroll_stats['total_scrolls'] += 1
        start_time = asyncio.get_event_loop().time()
        
        try:
            await page.wait_for_function(
                f"document.querySelectorAll('{target_count_selector}').length > {prev_count}",
                timeout=int(max_wait * 1000)
            )
            elapsed = asyncio.get_event_loop().time() - start_time
            self._adaptive_scroll_stats['content_loaded_fast'] += 1
            self._adaptive_scroll_stats['total_wait_time'] += elapsed
        except Exception:
            elapsed = asyncio.get_event_loop().time() - start_time
            self._adaptive_scroll_stats['content_loaded_timeout'] += 1
            self._adaptive_scroll_stats['total_wait_time'] += elapsed
        
        await asyncio.sleep(MIN_ADAPTIVE_DELAY)
        
        elements = await page.query_selector_all(target_count_selector)
        return len(elements)

    async def scroll_to_bottom(self, page) -> None:
        """Scroll to the bottom of the page (async version)."""
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(MIN_ADAPTIVE_DELAY)

    async def scroll_up_slightly(self, page, amount: int = 200) -> None:
        """Scroll up slightly (async version)."""
        await page.evaluate(f"window.scrollBy(0, -{amount})")
        await asyncio.sleep(0.1)

    async def wait_for_content(self, page, selector: str, timeout: int = 10000) -> bool:
        """Wait for content to appear (async version)."""
        try:
            await page.wait_for_selector(selector, timeout=timeout)
            return True
        except Exception:
            return False

    async def try_recovery_strategies(self, page, attempt: int) -> bool:
        """
        Try various recovery strategies when content isn't loading (async version).
        
        Args:
            page: Playwright page object
            attempt: Recovery attempt number
            
        Returns:
            bool: True if recovery succeeded
        """
        logger.info(f"Attempting recovery strategy #{attempt}")
        
        strategies = [
            self._scroll_jiggle_recovery,
            self._wait_longer_recovery,
            self._scroll_up_and_down_recovery,
        ]
        
        if attempt <= len(strategies):
            strategy = strategies[attempt - 1]
            return await strategy(page)
        
        return False

    async def _scroll_jiggle_recovery(self, page) -> bool:
        """Jiggle scroll to trigger content loading (async version)."""
        await page.evaluate("window.scrollBy(0, -300)")
        await asyncio.sleep(0.5)
        await page.evaluate("window.scrollBy(0, 500)")
        await asyncio.sleep(1.0)
        return True

    async def _wait_longer_recovery(self, page) -> bool:
        """Wait longer for content (async version)."""
        await asyncio.sleep(3.0)
        return True

    async def _scroll_up_and_down_recovery(self, page) -> bool:
        """Scroll up then down (async version)."""
        await page.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(1.0)
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(1.0)
        return True


# Global async scroller instance
_async_scroller = None


def get_async_scroller() -> AsyncScroller:
    """Get or create the global async scroller instance."""
    global _async_scroller
    if _async_scroller is None:
        _async_scroller = AsyncScroller()
    return _async_scroller
