"""
Browser session management for the fetcher module.

Handles browser setup, context creation, session persistence, and cleanup.
"""

import os
import random
from typing import Tuple, Optional, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from playwright.sync_api import Browser, BrowserContext, Page

try:
    from playwright.sync_api import sync_playwright
except Exception:
    sync_playwright = None

from .config import get_config
from .logging_config import get_logger

logger = get_logger('session_manager')

class SessionManager:
    """Manages browser sessions for tweet fetching operations."""

    def __init__(self):
        self.config = get_config()
        self.session_file = Path("x_session.json")

    def create_browser_context(
        self,
        playwright_instance,
        save_session: bool = False
    ) -> "Tuple[Browser, BrowserContext, Page]":
        """
        Create a browser context with configured settings.

        Args:
            playwright_instance: Playwright instance
            save_session: Whether to save session state

        Returns:
            Tuple of (browser, context, page)
        """

        # Launch browser
        browser = playwright_instance.chromium.launch(
            headless=self.config.headless,
            slow_mo=self.config.slow_mo
        )

        # Select random user agent
        selected_user_agent = random.choice(self.config.user_agents)

        # Context configuration
        context_kwargs = {
            "user_agent": selected_user_agent,
            "viewport": {
                "width": self.config.viewport_width,
                "height": self.config.viewport_height
            },
            "locale": "en-US",
            "timezone_id": "America/New_York",
            "color_scheme": "light",
            "java_script_enabled": True,
        }

        # Load existing session if available
        if self.session_file.exists():
            context_kwargs["storage_state"] = str(self.session_file)

        # Create context and page
        context = browser.new_context(**context_kwargs)
        page = context.new_page()

        # Save session state if requested
        if save_session:
            context.storage_state(path=str(self.session_file))

        logger.info(f"Created browser session with user agent: {selected_user_agent[:50]}...")

        return browser, context, page

    def cleanup_session(
        self,
        browser: "Browser",
        context: "BrowserContext"
    ) -> None:
        """
        Clean up browser session resources.

        Args:
            browser: Browser instance
            context: Browser context instance
        """
        try:
            if context:
                context.close()
            if browser:
                browser.close()
            logger.info("Browser session cleaned up successfully")
        except Exception as e:
            logger.warning(f"Error during session cleanup: {e}")

    def clear_session_data(self, context: "BrowserContext") -> None:
        """
        Clear browser session data between operations.

        Args:
            context: Browser context instance
        """
        try:
            context.evaluate("localStorage.clear(); sessionStorage.clear();")
            logger.debug("Session data cleared")
        except Exception as e:
            logger.warning(f"Failed to clear session data: {e}")

def create_session(save_session: bool = False) -> "Tuple[Browser, BrowserContext, Page]":
    """
    Convenience function to create a browser session.

    Args:
        save_session: Whether to save session state

    Returns:
        Tuple of (browser, context, page)
    """
    manager = SessionManager()
    playwright = sync_playwright()
    playwright.start()

    try:
        return manager.create_browser_context(playwright, save_session)
    except Exception as e:
        playwright.stop()
        raise e

def cleanup_session(
    browser: "Browser",
    context: "BrowserContext",
    playwright_instance=None
) -> None:
    """
    Convenience function to clean up a browser session.

    Args:
        browser: Browser instance
        context: Browser context instance
        playwright_instance: Optional playwright instance to stop
    """
    manager = SessionManager()
    manager.cleanup_session(browser, context)

    if playwright_instance:
        try:
            playwright_instance.stop()
        except Exception as e:
            logger.warning(f"Error stopping playwright: {e}")