"""
Async browser session management for the fetcher module.

Async version of session_manager.py for use with async_playwright.
Handles browser setup, context creation, and session persistence.
"""

import json
import random
import asyncio
from typing import Tuple, Optional, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext, Page

from .config import get_config
from .logging_config import get_logger
from .async_scroller import AsyncScroller, get_async_scroller

logger = get_logger('async_session_manager')


class AsyncSessionManager:
    """Async version of SessionManager - manages browser sessions for tweet fetching."""

    def __init__(self):
        self.config = get_config()
        self.session_file = Path("x_session.json")
        self.scroller = get_async_scroller()

    def has_valid_session(self) -> bool:
        """Check if we have a valid session file with cookies."""
        if not self.session_file.exists():
            return False
        
        try:
            with open(self.session_file, 'r') as f:
                session_data = json.load(f)
                cookies = session_data.get('cookies', [])
                auth_cookies = [c for c in cookies if c.get('name') in ('auth_token', 'ct0', 'twid')]
                return len(auth_cookies) >= 2
        except (json.JSONDecodeError, IOError):
            return False

    async def create_browser_context(
        self,
        playwright_instance,
        save_session: bool = False
    ) -> "Tuple[Browser, BrowserContext, Page]":
        """
        Create a browser context with configured settings (async version).

        Args:
            playwright_instance: Async Playwright instance
            save_session: Whether to save session state

        Returns:
            Tuple of (browser, context, page)
        """
        # Launch browser
        browser = await playwright_instance.chromium.launch(
            channel="chrome",
            headless=self.config.headless,
            slow_mo=self.config.slow_mo,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-automation",
                "--no-sandbox",
            ]
        )

        selected_user_agent = random.choice(self.config.user_agents)

        context_kwargs = {
            "user_agent": selected_user_agent,
            "viewport": {
                "width": self.config.viewport_width,
                "height": self.config.viewport_height
            },
            "locale": "es-ES",
            "timezone_id": "Europe/Madrid",
            "color_scheme": "light",
            "java_script_enabled": True,
        }

        if self.session_file.exists():
            context_kwargs["storage_state"] = str(self.session_file)

        context = await browser.new_context(**context_kwargs)
        page = await context.new_page()
        
        # Remove webdriver property
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)
        
        logger.info(f"Async browser context created (headless={self.config.headless})")
        
        return browser, context, page

    async def save_session_state(self, context) -> bool:
        """Save current session state to file (async version)."""
        try:
            state = await context.storage_state()
            with open(self.session_file, 'w') as f:
                json.dump(state, f)
            logger.info("Session state saved")
            return True
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return False

    async def cleanup_session(self, browser, context) -> None:
        """Clean up browser and context (async version)."""
        try:
            if context:
                await context.close()
            if browser:
                await browser.close()
            logger.debug("Session cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def navigate_to_profile(self, page, username: str) -> bool:
        """Navigate to a user's profile page (async version)."""
        profile_url = f"https://x.com/{username}"
        
        try:
            await page.goto(profile_url, wait_until="domcontentloaded", timeout=30000)
            await self.scroller.delay(1.0, 2.0)
            
            # Wait for tweet articles or profile content
            try:
                await page.wait_for_selector('article[data-testid="tweet"]', timeout=15000)
                return True
            except Exception:
                # Check if profile exists but has no tweets
                try:
                    await page.wait_for_selector('[data-testid="UserName"]', timeout=5000)
                    return True
                except Exception:
                    logger.warning(f"Profile @{username} not found or has no content")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to navigate to @{username}: {e}")
            return False


# Global async session manager instance
_async_session_manager = None


def get_async_session_manager() -> AsyncSessionManager:
    """Get or create the global async session manager instance."""
    global _async_session_manager
    if _async_session_manager is None:
        _async_session_manager = AsyncSessionManager()
    return _async_session_manager
