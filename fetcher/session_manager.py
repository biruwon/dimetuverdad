"""
Browser session management for the fetcher module.

Handles browser setup, context creation, session persistence, login operations, and cleanup.
"""

import os
import random
import time
from typing import Tuple, Optional, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from playwright.sync_api import Browser, BrowserContext, Page
from playwright.sync_api import sync_playwright, TimeoutError
from .config import get_config
from .logging_config import get_logger
from .scroller import Scroller

logger = get_logger('session_manager')

class SessionManager:
    """Manages browser sessions for tweet fetching operations."""

    def __init__(self):
        self.config = get_config()
        self.session_file = Path("x_session.json")
        self.scroller = Scroller()

    def login_and_save_session(self, page, username: str, password: str) -> bool:
        """
        Login with better anti-detection measures.
        
        Args:
            page: Playwright page object
            username: Login username
            password: Login password
            
        Returns:
            bool: True if login successful, False otherwise
        """
        
        print("🔐 Starting login process...")
        
        # Advanced stealth setup
        page.route("**/*", lambda route, request: (
            route.continue_() if not any(keyword in request.url.lower() 
                for keyword in ["webdriver", "automation", "headless"]) 
            else route.abort()
        ))
        
        # Go to login page with random delay
        page.goto("https://x.com/login")
        self.scroller.delay(2.0, 4.0)

        # Step 1: Enter username with human-like typing
        try:
            page.wait_for_selector('input[name="text"]', timeout=10000)
            username_field = page.locator('input[name="text"]')
            
            # Type with human-like delays
            for char in username:
                username_field.type(char)
                time.sleep(random.uniform(0.05, 0.15))
            
            self.scroller.delay(0.5, 1.5)
            page.click('div[data-testid="LoginForm_Login_Button"], div[role="button"]:has-text("Siguiente"), button:has-text("Next")')
            self.scroller.delay(2.0, 4.0)
            
        except TimeoutError:
            print("⚠️ Username field not found, may already be logged in")

        # Step 2: Handle unusual activity or confirmation
        try:
            page.wait_for_selector('input[name="text"]', timeout=4000)
            unusual_activity = page.query_selector('div:has-text("unusual activity"), div:has-text("actividad inusual")')
            
            if unusual_activity:
                print("⚠️ Unusual activity detected, entering email/phone...")
                confirmation_field = page.locator('input[name="text"]')
                confirmation_field.fill(self.config.email_or_phone or username)
            else:
                print("🔄 Confirming username...")
                username_field = page.locator('input[name="text"]') 
                username_field.fill(username)
            
            self.scroller.delay(1.0, 2.0)
            page.click('div[data-testid="LoginForm_Login_Button"], div[role="button"]:has-text("Siguiente"), button:has-text("Next")')
            self.scroller.delay(2.0, 4.0)
            
        except TimeoutError:
            print("✅ No additional confirmation needed")

        # Step 3: Enter password with human-like typing
        try:
            page.wait_for_selector('input[name="password"]', timeout=10000)
            password_field = page.locator('input[name="password"]')
            
            # Type password with human-like delays
            for char in password:
                password_field.type(char)
                time.sleep(random.uniform(0.05, 0.12))
            
            self.scroller.delay(0.5, 1.5)
            page.click('div[data-testid="LoginForm_Login_Button"], button:has-text("Iniciar sesión"), button:has-text("Log in")')
            self.scroller.delay(3.0, 6.0)
            
        except TimeoutError:
            print("⚠️ Password field not found")

        # Verify login success
        try:
            page.wait_for_url("https://x.com/home", timeout=15000)
            print("✅ Login successful!")
            self.scroller.delay(2.0, 4.0)
            return True
        except TimeoutError:
            print("❌ Login verification failed - check for CAPTCHA or 2FA")
            return False

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