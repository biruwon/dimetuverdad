#!/usr/bin/env python3
"""
Session refresh using your real Chrome browser.
This avoids automation detection by using Chrome's persistent profile.
"""

import sys
import os
import json
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from playwright.sync_api import sync_playwright


def refresh_session_with_chrome():
    """
    Launch Chrome with your real profile to login to Twitter.
    After logging in manually, the session will be exported.
    """
    print("🔄 Launching Chrome for Twitter/X session refresh...")
    print("📝 Instructions:")
    print("   1. A Chrome window will open to Twitter login")
    print("   2. Log in manually (complete any verification)")
    print("   3. Once you see your timeline, press ENTER in this terminal")
    print("   4. The session will be saved automatically")
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
            
            # Wait for user to complete login
            input("\n✋ Press ENTER after you've logged in and can see your timeline...")
            
            # Verify we're logged in
            page.goto("https://x.com/home", wait_until="domcontentloaded")
            time.sleep(3)
            
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
            return False
            
        finally:
            context.close()
            browser.close()


if __name__ == "__main__":
    success = refresh_session_with_chrome()
    sys.exit(0 if success else 1)
