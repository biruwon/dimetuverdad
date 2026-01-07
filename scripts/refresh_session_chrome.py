#!/usr/bin/env python3
"""
Session refresh using your real Chrome browser.
This avoids automation detection by using Chrome's persistent profile.
"""

import sys
import os
import json
import time
import random
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from playwright.sync_api import sync_playwright, TimeoutError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def human_delay(min_sec: float = 0.5, max_sec: float = 2.0):
    """Add human-like delay"""
    time.sleep(random.uniform(min_sec, max_sec))


def type_like_human(element, text: str):
    """Type text with human-like delays between keystrokes"""
    for char in text:
        element.type(char)
        time.sleep(random.uniform(0.05, 0.15))


def refresh_session_with_chrome():
    """
    Launch Chrome with your real profile to login to Twitter.
    Automatically fills credentials and handles verification steps.
    """
    print("🔄 Launching Chrome for Twitter/X session refresh...")
    
    # Get credentials from environment
    username = os.getenv('X_USERNAME', '')
    password = os.getenv('X_PASSWORD', '')
    email_or_phone = os.getenv('X_EMAIL_OR_PHONE', '')
    
    if not username or not password:
        print("❌ Missing credentials. Set X_USERNAME and X_PASSWORD in .env file")
        return False
    
    print(f"📧 Using credentials for: {username}")
    print()
    
    session_file = Path("x_session.json")
    
    with sync_playwright() as p:
        # Use channel="chrome" to use your installed Chrome
        # This helps avoid automation detection
        browser = p.chromium.launch(
            channel="chrome",  # Use real Chrome, not Chromium
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-automation",
                "--no-sandbox",
            ]
        )
        
        context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            locale="es-ES",
            timezone_id="Europe/Madrid",
        )
        
        page = context.new_page()
        
        # Remove webdriver property
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)
        
        try:
            # Go to Twitter login
            print("🌐 Opening Twitter login page...")
            page.goto("https://x.com/login", wait_until="domcontentloaded")
            human_delay(2.0, 4.0)
            
            # Step 1: Enter username
            print("📝 Entering username...")
            try:
                page.wait_for_selector('input[name="text"]', timeout=10000)
                username_field = page.locator('input[name="text"]')
                type_like_human(username_field, username)
                human_delay(0.5, 1.5)
                
                # Click Next button
                page.click('div[data-testid="LoginForm_Login_Button"], div[role="button"]:has-text("Siguiente"), button:has-text("Next")')
                human_delay(2.0, 4.0)
            except TimeoutError:
                print("⚠️ Username field not found")
            
            # Step 2: Handle unusual activity or username confirmation
            try:
                page.wait_for_selector('input[name="text"]', timeout=4000)
                unusual_activity = page.query_selector('div:has-text("unusual activity"), div:has-text("actividad inusual")')
                
                if unusual_activity:
                    print("⚠️ Unusual activity detected, entering email/phone...")
                    confirmation_field = page.locator('input[name="text"]')
                    confirmation_value = email_or_phone or username
                    type_like_human(confirmation_field, confirmation_value)
                else:
                    print("🔄 Username confirmation requested...")
                    username_field = page.locator('input[name="text"]')
                    type_like_human(username_field, username)
                
                human_delay(1.0, 2.0)
                page.click('div[data-testid="LoginForm_Login_Button"], div[role="button"]:has-text("Siguiente"), button:has-text("Next")')
                human_delay(2.0, 4.0)
            except TimeoutError:
                print("✅ No additional confirmation needed")
            
            # Step 3: Enter password
            print("🔑 Entering password...")
            try:
                page.wait_for_selector('input[name="password"]', timeout=10000)
                password_field = page.locator('input[name="password"]')
                type_like_human(password_field, password)
                human_delay(0.5, 1.5)
                
                # Click Login button
                page.click('div[data-testid="LoginForm_Login_Button"], button:has-text("Iniciar sesión"), button:has-text("Log in")')
                human_delay(3.0, 6.0)
            except TimeoutError:
                print("⚠️ Password field not found")
            
            # Step 4: Verify login - quick check first
            print("🔍 Verifying login...")
            try:
                page.wait_for_url("https://x.com/home", timeout=15000)
                print("✅ Login successful!")
            except TimeoutError:
                # Give user time to complete CAPTCHA/2FA manually
                print("⚠️ Automated login needs help - please complete CAPTCHA/2FA in the browser window...")
                print("⏳ Waiting up to 120 seconds for manual completion...")
                try:
                    page.wait_for_url("https://x.com/home", timeout=120000)
                    print("✅ Login successful (manual completion)!")
                except TimeoutError:
                    print("❌ Login verification failed - timeout waiting for login completion")
                    return False
            
            human_delay(2.0, 4.0)
            
            # Verify we can see tweets
            tweets = page.query_selector_all('article[data-testid="tweet"]')
            print(f"🧪 Found {len(tweets)} tweets on timeline")
            
            if len(tweets) > 0:
                # Save session
                context.storage_state(path=str(session_file))
                print(f"💾 Session saved to {session_file}")
                print("✅ Session refresh successful!")
                return True
            else:
                print("⚠️ No tweets visible - make sure you're logged in")
                save_anyway = input("Save session anyway? (y/n): ")
                if save_anyway.lower() == 'y':
                    context.storage_state(path=str(session_file))
                    print(f"💾 Session saved to {session_file}")
                    return True
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            context.close()
            browser.close()


if __name__ == "__main__":
    success = refresh_session_with_chrome()
    sys.exit(0 if success else 1)
