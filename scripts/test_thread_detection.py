#!/usr/bin/env python3
"""UI-based thread detection tester for Twitter/X.

This script focuses exclusively on scraping thread candidates from a user's
timeline using Twitter's native thread line indicator. When a thread line
(or inline "Show more replies" entry) is detected, the script opens the
thread in a new tab, scrolls the conversation, and collects every post from
that user until another user appears or no additional posts load.

The goal is to validate the production-ready algorithm before wiring it into
the fetcher pipeline, without writing to the database.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
from typing import Dict, List, Optional, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from playwright.async_api import BrowserContext, ElementHandle, Page, async_playwright

from fetcher.logging_config import setup_logging
from fetcher.session_manager import SessionManager
from fetcher.config import get_config
from fetcher.thread_detector import ThreadDetector

logger = setup_logging("INFO")

# Path to session file (same as SessionManager uses)
SESSION_FILE = "x_session.json"


async def refresh_session_async(playwright) -> bool:
    """Refresh the Twitter/X session by logging in again (async version)."""
    import random
    from pathlib import Path
    
    logger.info("🔄 Refreshing Twitter/X session...")

    config = get_config()

    if not config.username or not config.password:
        logger.error("❌ Missing X_USERNAME or X_PASSWORD in environment variables")
        return False

    session_manager = SessionManager()
    session_file = Path(SESSION_FILE)
    
    browser = await playwright.chromium.launch(
        headless=False,  # Show browser for login
        slow_mo=config.slow_mo
    )
    
    selected_user_agent = random.choice(config.user_agents)
    
    context_kwargs = {
        "user_agent": selected_user_agent,
        "viewport": {
            "width": config.viewport_width,
            "height": config.viewport_height
        },
        "locale": "en-US",
        "timezone_id": "America/New_York",
        "color_scheme": "light",
        "java_script_enabled": True,
    }
    
    context = await browser.new_context(**context_kwargs)
    page = await context.new_page()

    try:
        # Navigate to login page
        await page.goto("https://x.com/login", wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)
        
        # Enter username
        logger.info("📝 Entering username...")
        username_input = await page.wait_for_selector('input[autocomplete="username"]', timeout=10000)
        await username_input.fill(config.username)
        await page.wait_for_timeout(1000)
        
        # Click next button
        next_button = await page.query_selector('button:has-text("Next")')
        if next_button:
            await next_button.click()
        else:
            await page.keyboard.press("Enter")
        await page.wait_for_timeout(3000)
        
        # Check if Twitter is asking for email/phone verification (unusual login challenge)
        # Try multiple times as the page may take a moment to load the challenge
        for attempt in range(3):
            challenge_input = await page.query_selector('input[data-testid="ocfEnterTextTextInput"]')
            if challenge_input:
                logger.warning("⚠️ Twitter is requesting additional verification (email/phone)")
                # Try to enter email if available
                if hasattr(config, 'email') and config.email:
                    logger.info("📧 Entering email for verification...")
                    await challenge_input.fill(config.email)
                    verify_button = await page.query_selector('button[data-testid="ocfEnterTextNextButton"]')
                    if verify_button:
                        await verify_button.click()
                    await page.wait_for_timeout(3000)
                    break
                else:
                    logger.error("❌ Email verification required but no email configured")
                    logger.info("💡 Please run refresh_session.py manually with a visible browser")
                    return False
            
            # Check if password field is already visible
            password_input = await page.query_selector('input[type="password"]')
            if password_input:
                break
            
            await page.wait_for_timeout(1000)
        
        # Enter password - try multiple selectors
        logger.info("🔑 Entering password...")
        password_input = await page.query_selector('input[type="password"]')
        if not password_input:
            # Try waiting for it with a longer timeout
            try:
                password_input = await page.wait_for_selector('input[type="password"]', timeout=10000)
            except Exception:
                # Maybe we're on a challenge page - take screenshot for debugging
                logger.warning("⚠️ Password field not found - checking page state...")
                page_content = await page.content()
                if 'challenge' in page_content.lower() or 'verify' in page_content.lower():
                    logger.error("❌ Twitter is showing a verification challenge")
                    logger.info("💡 Please run: python scripts/refresh_session.py")
                    return False
                # Try alternative password selectors
                password_input = await page.query_selector('input[name="password"]')
                if not password_input:
                    password_input = await page.query_selector('input[autocomplete="current-password"]')
        
        if not password_input:
            logger.error("❌ Could not find password input field")
            return False
        await password_input.fill(config.password)
        await page.wait_for_timeout(1000)
        
        # Click login button
        login_button = await page.query_selector('button[data-testid="LoginForm_Login_Button"]')
        if not login_button:
            login_button = await page.query_selector('button:has-text("Log in")')
        if login_button:
            await login_button.click()
        else:
            await page.keyboard.press("Enter")
        
        # Wait for login to complete
        logger.info("⏳ Waiting for login to complete...")
        await page.wait_for_timeout(8000)
        
        # Check if we're logged in by looking for home timeline
        await page.goto("https://x.com/home", wait_until="domcontentloaded")
        await page.wait_for_timeout(5000)
        
        tweets = await page.query_selector_all('article[data-testid="tweet"]')
        
        if len(tweets) > 0:
            logger.info(f"✅ Login successful! Found {len(tweets)} tweets")
            # Save the session
            await context.storage_state(path=str(session_file))
            logger.info(f"💾 Session saved to {session_file}")
            return True
        else:
            logger.warning("⚠️ Login may have failed - no tweets visible")
            # Check for error messages or challenges
            page_text = await page.inner_text('body')
            if 'challenge' in page_text.lower() or 'verify' in page_text.lower():
                logger.error("❌ Twitter is requesting verification - manual intervention needed")
            logger.info("💡 Try running: python scripts/refresh_session.py")
            return False

    except Exception as e:
        logger.error(f"❌ Error during session refresh: {e}")
        return False

    finally:
        await context.close()
        await browser.close()

ARIA_REPLY_MARKERS = [
    "replying to",
    "respondiendo a",
    "en respuesta a",
]

MENTION_PATTERN = re.compile(r"@([a-z0-9_.]+)")

THREAD_TIME_WINDOW_MINUTES: Optional[int] = None  # Disable time-based pruning by default


@dataclass
class ThreadSummary:
    """Lightweight structure describing a discovered thread."""

    start_id: str
    url: str
    tweets: List[Dict]
    conversation_id: Optional[str] = None  # Root tweet ID that all replies share

    @property
    def size(self) -> int:
        return len(self.tweets)


class ThreadDetectionTester:
    """Timeline-driven tester that relies on UI thread indicators."""

    def __init__(
        self,
        max_threads: int = 3,
        max_timeline_scrolls: int = 20,
        thread_scroll_limit: int = 24,
        indicator_retry_limit: int = 2,
        foreign_only_cycles_limit: Optional[int] = None,
    ) -> None:
        self.session_manager = SessionManager()
        self.thread_detector = ThreadDetector()
        self.max_threads = max_threads
        self.max_timeline_scrolls = max_timeline_scrolls
        self.thread_scroll_limit = thread_scroll_limit
        self.indicator_retry_limit = indicator_retry_limit
        # Allow limited extra scroll cycles past foreign replies to capture
        # remaining author posts before giving up entirely.
        self.foreign_only_cycles_limit = (
            1 if foreign_only_cycles_limit is None else foreign_only_cycles_limit
        )
        self.processed_tweet_ids: set[str] = set()
        self.collected_threads: List[ThreadSummary] = []
        self.timeline_thread_line: Dict[str, bool] = {}
        self.indicator_retry_counts: Dict[str, int] = {}

    async def test_thread_detection(self, username: str) -> None:
        """Entry point – scan the user's timeline and expand threads."""

        async with async_playwright() as playwright:
            browser, context, page = await self._create_browser_context_async(playwright)
            try:
                await self._open_profile(page, username)
                
                # Validate session by checking if we can see tweets
                is_valid = await self._validate_session(page, username)
                
                if not is_valid:
                    logger.warning("⚠️ Session appears invalid, attempting refresh...")
                    await context.close()
                    await browser.close()
                    
                    # Refresh session using async API
                    if await refresh_session_async(playwright):
                        logger.info("✅ Session refreshed, retrying...")
                        # Recreate browser with new session
                        browser, context, page = await self._create_browser_context_async(playwright)
                        await self._open_profile(page, username)
                        
                        # Validate again
                        is_valid = await self._validate_session(page, username)
                        if not is_valid:
                            logger.error("❌ Session still invalid after refresh")
                            return
                    else:
                        logger.error("❌ Failed to refresh session")
                        return
                
                await self._scan_timeline(page, context, username)
                self._log_summary(username)
            finally:
                await context.close()
                await browser.close()

    async def _validate_session(self, page: Page, username: str) -> bool:
        """Check if current session is valid by looking for tweets on the profile."""
        try:
            # Wait longer for page to fully load
            await page.wait_for_timeout(4000)
            articles = await page.query_selector_all('article[data-testid="tweet"]')
            
            if len(articles) > 0:
                logger.info(f"✅ Session valid: Found {len(articles)} tweets on @{username}'s profile")
                return True
            
            # Check for login prompts or error messages
            page_text = await page.inner_text('body')
            login_indicators = ['log in', 'sign in', 'iniciar sesión', 'create your account']
            
            for indicator in login_indicators:
                if indicator.lower() in page_text.lower():
                    logger.warning(f"⚠️ Login prompt detected: '{indicator}'")
                    return False
            
            # No tweets but no login prompt either - might just be loading
            logger.warning(f"⚠️ No tweets found on @{username}'s profile, but no login prompt detected")
            return len(articles) > 0
            
        except Exception as e:
            logger.warning(f"Error validating session: {e}")
            return False

    async def _create_browser_context_async(self, playwright):
        """Create an async browser context with session support."""
        import random
        from pathlib import Path
        
        config = self.session_manager.config
        session_file = Path(SESSION_FILE)
        
        browser = await playwright.chromium.launch(
            headless=config.headless,
            slow_mo=config.slow_mo
        )
        
        selected_user_agent = random.choice(config.user_agents)
        
        context_kwargs = {
            "user_agent": selected_user_agent,
            "viewport": {
                "width": config.viewport_width,
                "height": config.viewport_height
            },
            "locale": "en-US",
            "timezone_id": "America/New_York",
            "color_scheme": "light",
            "java_script_enabled": True,
        }
        
        if session_file.exists():
            context_kwargs["storage_state"] = str(session_file)
        
        context = await browser.new_context(**context_kwargs)
        page = await context.new_page()
        
        logger.info(f"Created async browser session with user agent: {selected_user_agent[:50]}...")
        
        return browser, context, page

    async def _extract_tweet_from_article_async(self, article: ElementHandle) -> Optional[Dict]:
        """Async version of tweet extraction from article element."""
        try:
            tweet_data: Dict = {}

            time_link = await article.query_selector('a[href*="/status/"]')
            if time_link:
                href = await time_link.get_attribute('href')
                if href:
                    parts = href.strip('/').split('/')
                    if len(parts) >= 3 and parts[1] == 'status':
                        tweet_data['tweet_id'] = parts[2].split('?')[0]
                        if 'username' not in tweet_data and parts[0]:
                            tweet_data['username'] = parts[0]

            username = None
            user_link = await article.query_selector('a[href^="/"]')
            if user_link:
                href = await user_link.get_attribute('href')
                if href and href.startswith('/'):
                    candidate = href.strip('/').split('/')[0]
                    if candidate and not candidate.startswith('status') and candidate != 'home':
                        username = candidate

            if not username:
                article_text = await article.inner_text()
                if '@' in article_text:
                    for line in article_text.split('\n'):
                        line = line.strip()
                        if line.startswith('@') and len(line.split()) == 1:
                            username = line[1:]
                            break

            if username:
                tweet_data['username'] = username

            content_div = await article.query_selector('[data-testid="tweetText"]')
            if content_div:
                tweet_data['content'] = await content_div.inner_text()
            else:
                tweet_data['content'] = await article.inner_text()

            time_element = await article.query_selector('time')
            if time_element:
                datetime_attr = await time_element.get_attribute('datetime')
                if datetime_attr:
                    tweet_data['tweet_timestamp'] = datetime_attr

            if tweet_data.get('tweet_id'):
                return tweet_data

            return None
        except Exception as e:
            logger.warning(f"Error extracting tweet data (async): {e}")
            return None

    async def _has_thread_line_async(self, article: ElementHandle) -> bool:
        """Async version of thread line detection - NOT USED in current approach.
        
        Note: This method checks individual articles, but Twitter groups thread
        tweets together in the same cellInnerDiv on the timeline. We now use
        _scan_timeline_for_thread_groups() instead which looks at cellInnerDiv
        containers to find grouped tweets.
        """
        # This method is kept for potential future use but is not the primary detection
        try:
            article_text = await article.inner_text()
            thread_indicators = [
                'show this thread',
                'mostrar este hilo',
                'ver hilo',
            ]
            text_lower = article_text.lower()
            for indicator in thread_indicators:
                if indicator in text_lower:
                    return True
            return False
        except Exception as e:
            logger.warning(f"Error checking thread line (async): {e}")
            return False

    async def _extract_reply_handles(self, article: ElementHandle) -> List[str]:
        """Extract the usernames this tweet is replying to.
        
        Twitter shows "Replying to @user1 @user2" text above the tweet content
        when a tweet is a reply. Returns list of lowercase usernames.
        """
        handles: List[str] = []
        try:
            # Look for the "Replying to" indicator
            reply_info = await article.query_selector('[data-testid="reply"]')
            if not reply_info:
                # Alternative selector - sometimes it's just text
                text = await article.inner_text()
                lines = text.split('\n')
                for line in lines:
                    line_lower = line.lower().strip()
                    if 'replying to' in line_lower or 'respondiendo a' in line_lower:
                        # Extract @handles from this line
                        import re
                        at_handles = re.findall(r'@(\w+)', line)
                        handles.extend([h.lower() for h in at_handles])
                        break
            else:
                reply_text = await reply_info.inner_text()
                import re
                at_handles = re.findall(r'@(\w+)', reply_text)
                handles.extend([h.lower() for h in at_handles])
        except Exception as e:
            logger.debug(f"Error extracting reply handles: {e}")
        
        return handles

    async def _open_profile(self, page: Page, username: str) -> None:
        profile_url = f"https://x.com/{username}"
        await page.goto(profile_url, wait_until="domcontentloaded")
        await page.wait_for_timeout(4000)

    async def _scan_timeline(self, page: Page, context: BrowserContext, username: str) -> None:
        """Scan timeline for threads using CSS thread connector class.
        
        Detection method: CSS class r-1bimlpy indicates a tweet is visually
        shown as part of a thread on the timeline. When detected, we immediately
        visit the tweet to verify it's a valid self-reply thread.
        
        This approach:
        - Uses only Twitter's own visual indicator (the thread line)
        - Validates threads immediately when found (don't wait until end)
        - Catches thread continuations (new posts on old threads)
        """
        target_username = username.lower()
        scroll_count = 0
        seen_ids: set[str] = set()

        logger.info(f"📡 Scanning timeline for @{username}...")
        
        for scroll_num in range(self.max_timeline_scrolls):
            if len(self.collected_threads) >= self.max_threads:
                logger.info(f"🛑 Reached max threads ({self.max_threads})")
                break
                
            articles = await page.query_selector_all('article[data-testid="tweet"]')
            threads_found_this_scroll = 0
            
            for article in articles:
                tweet = await self._extract_tweet_from_article_async(article)
                if not tweet:
                    continue
                
                tweet_id = tweet.get("tweet_id")
                tweet_username = tweet.get("username", "").lower()
                
                if not tweet_id or tweet_id in seen_ids:
                    continue
                
                if tweet_username != target_username:
                    continue
                
                seen_ids.add(tweet_id)
                
                # Check for CSS thread connector class (r-1bimlpy)
                # This is Twitter's visual indicator for thread connections
                article_html = await article.inner_html()
                has_thread_connector = 'r-1bimlpy' in article_html
                
                if has_thread_connector and tweet_id not in self.processed_tweet_ids:
                    logger.info(f"🧵 Thread indicator found: {tweet_id}")
                    # Visit immediately to verify thread (in new tab, keep timeline intact)
                    thread = await self._collect_thread(context, tweet, target_username)
                    
                    if thread and thread.size >= 2:
                        self._record_thread(thread)
                        threads_found_this_scroll += 1
                        logger.info(f"✅ Thread #{len(self.collected_threads)} captured ({thread.size} posts)")
                    else:
                        logger.debug(f"Not a valid thread (only {thread.size if thread else 0} posts)")
                        self.processed_tweet_ids.add(tweet_id)
            
            logger.info(f"📜 Scroll {scroll_num}: {len(articles)} articles, {threads_found_this_scroll} threads found")
            
            await self._incremental_timeline_scroll(page)
            await page.wait_for_timeout(1500)
            scroll_count += 1
        
        logger.info(f"📊 Scan complete: {scroll_count} scrolls, {len(self.collected_threads)} threads found")

    async def _collect_thread(
        self, context: BrowserContext, start_tweet: Dict, username: str
    ) -> Optional[ThreadSummary]:
        """Open a new tab and collect the self-reply thread chain.
        
        A valid thread is a sequence where each tweet is a DIRECT REPLY to the 
        previous tweet by the SAME USER. We stop when:
        - Another user's tweet appears in the chain
        - User replies to someone else's comment (not their previous thread post)
        """
        username_lower = username.lower()
        start_id = start_tweet["tweet_id"]
        thread_url = f"https://x.com/{username}/status/{start_id}"

        page = await context.new_page()
        
        # Set up conversation_id extraction via network interception
        conversation_ids: set[str] = set()
        
        async def capture_conversation_id(response):
            try:
                url = response.url
                if '/graphql/' in url and ('TweetDetail' in url or 'TweetResultByRestId' in url):
                    body = await response.text()
                    # Find conversation_id_str in API response
                    import re
                    matches = re.findall(r'"conversation_id_str"\s*:\s*"(\d+)"', body)
                    for m in matches:
                        conversation_ids.add(m)
            except:
                pass  # Response might not be text
        
        page.on("response", capture_conversation_id)
        
        try:
            await page.goto(thread_url, wait_until="domcontentloaded")
            await page.wait_for_timeout(3000)  # Allow API calls to complete

            # Collect thread by walking the conversation structure
            posts: List[Dict] = []
            thread_chain = await self._extract_thread_chain(page, username_lower, start_id)
            
            for tweet_data in thread_chain:
                posts.append(self._format_thread_post(tweet_data))
            
            if len(posts) < 2:
                return None
            
            # Use the smallest conversation_id (usually the root tweet)
            conv_id = min(conversation_ids) if conversation_ids else None
            if conv_id:
                logger.debug(f"Thread conversation_id: {conv_id}")
                
            return ThreadSummary(
                start_id=start_id, 
                url=thread_url, 
                tweets=posts,
                conversation_id=conv_id
            )
        finally:
            await page.close()
    
    async def _extract_thread_chain(
        self, page: Page, username_lower: str, start_id: str
    ) -> List[Dict]:
        """Extract the self-reply chain from a conversation page.
        
        A valid self-thread is when:
        1. User posts tweet A
        2. User replies to tweet A with tweet B  
        3. User replies to tweet B with tweet C
        
        We detect this by checking the ACTUAL PARENT of each tweet (the tweet
        directly above in conversation view), not just the "Replying to" mentions.
        """
        thread_posts: List[Dict] = []
        
        # Get all articles in conversation order
        conversation = await page.query_selector('[aria-label="Timeline: Conversation"]')
        if conversation:
            articles = await conversation.query_selector_all('article[data-testid="tweet"]')
        else:
            articles = await page.query_selector_all('article[data-testid="tweet"]')
        
        # Build a map of tweet_id -> (username, index, tweet_data)
        tweet_map: Dict[str, tuple] = {}
        article_list = []
        
        for i, article in enumerate(articles):
            tweet = await self._extract_tweet_from_article_async(article)
            if tweet and tweet.get("tweet_id"):
                tweet_id = tweet["tweet_id"]
                tweet_username = tweet.get("username", "").lower()
                tweet_map[tweet_id] = (tweet_username, i, tweet)
                article_list.append((tweet_id, tweet_username, tweet))
        
        # Find the focal tweet (start_id) and walk the chain
        focal_idx = -1
        for i, (tid, tuser, tdata) in enumerate(article_list):
            if tid == start_id:
                focal_idx = i
                break
        
        if focal_idx < 0:
            return []
        
        # Start with focal tweet if it's from the target user
        focal_user = article_list[focal_idx][1]
        if focal_user != username_lower:
            return []
        
        thread_posts.append(article_list[focal_idx][2])
        last_thread_id = start_id
        
        # Walk forward from focal tweet, checking each subsequent tweet from same user
        for i in range(focal_idx + 1, len(article_list)):
            tweet_id, tweet_username, tweet_data = article_list[i]
            
            if tweet_username != username_lower:
                # Another user's tweet - not a thread continuation
                # But we might find more self-replies later, so continue checking
                continue
            
            # This is a tweet from the target user - check if it's a DIRECT reply
            # to our last thread post by checking what the parent tweet is
            # The parent is the tweet IMMEDIATELY BEFORE this one in conversation
            if i > 0:
                parent_id, parent_username, _ = article_list[i - 1]
                
                # Valid thread continuation: parent must be our last thread post
                if parent_id == last_thread_id and parent_username == username_lower:
                    thread_posts.append(tweet_data)
                    last_thread_id = tweet_id
                    logger.debug(f"Thread continues: {tweet_id} replies to {parent_id}")
                else:
                    # This tweet replies to something else (another user's comment)
                    logger.debug(f"Thread break: {tweet_id} parent is {parent_id} by @{parent_username}, not {last_thread_id}")
                    # Don't add this tweet, but keep looking for more valid continuations
                    # Actually, once we hit a reply to another user, the thread is broken
                    break
        
        # Also check for parent tweets (if start_id was a reply in a thread)
        if len(thread_posts) > 0:
            parent_posts = await self._collect_parent_thread_posts_v2(
                article_list, focal_idx, username_lower
            )
            if parent_posts:
                thread_posts = parent_posts + thread_posts
        
        return thread_posts
    
    async def _collect_parent_thread_posts_v2(
        self, article_list: List[tuple], focal_idx: int, username_lower: str
    ) -> List[Dict]:
        """Walk backwards from focal tweet to find parent thread posts."""
        parent_posts: List[Dict] = []
        
        # Walk backwards from focal tweet
        current_idx = focal_idx
        while current_idx > 0:
            current_id, current_user, _ = article_list[current_idx]
            parent_idx = current_idx - 1
            parent_id, parent_user, parent_data = article_list[parent_idx]
            
            # If parent is from same user, it's part of the thread
            if parent_user == username_lower:
                parent_posts.insert(0, parent_data)  # Insert at beginning
                current_idx = parent_idx
            else:
                # Parent is from another user - stop
                break
        
        return parent_posts
    
    async def _scroll_thread_view(self, page: Page, direction: str) -> None:
        """Scroll the conversation timeline toward the requested direction."""

        await self._incremental_scroll(page, direction=direction, steps=4)

    async def _harvest_thread_posts(
        self,
        page: Page,
        username_lower: str,
        seen_ids: set[str],
        posts: List[Dict],
    ) -> tuple[bool, bool]:
        """Collect newly visible tweets and signal if another user appears."""

        conversation = await page.query_selector('[aria-label="Timeline: Conversation"]')
        if conversation:
            articles = await conversation.query_selector_all('article[data-testid="tweet"]')
        else:
            articles = await page.query_selector_all('article[data-testid="tweet"]')

        new_in_cycle = False
        foreign_seen = False
        for article in articles:
            tweet = await self._extract_tweet_from_article_async(article)
            if not tweet:
                continue

            tweet_username = tweet.get("username", "").lower()
            tweet_id = tweet.get("tweet_id")

            if tweet_username == username_lower and tweet_id not in seen_ids:
                if await self._is_reply_to_other_user(article, username_lower):
                    continue
                posts.append(self._format_thread_post(tweet))
                seen_ids.add(tweet_id)
                new_in_cycle = True
            elif tweet_username and tweet_username != username_lower:
                foreign_seen = True

        return new_in_cycle, foreign_seen

    async def _incremental_scroll(
        self,
        page: Page,
        *,
        direction: str = "down",
        steps: int = 4,
        step_px: int = 1200,
    ) -> None:
        delta = step_px if direction == "down" else -step_px
        for _ in range(steps):
            try:
                await page.mouse.wheel(0, delta)
            except Exception:
                await page.evaluate("window.scrollBy(0, arguments[0])", delta)
            await page.wait_for_timeout(400)

    async def _incremental_timeline_scroll(self, page: Page, steps: int = 4, step_px: int = 1200) -> None:
        """Advance the main timeline in smaller movements to avoid skipping tweets."""

        await self._incremental_scroll(page, direction="down", steps=steps, step_px=step_px)

    def _build_thread_summary(
        self, posts: List[Dict], url: str, start_id: str
    ) -> Optional[ThreadSummary]:
        if len(posts) < 2:
            return None

        ordered: List[Dict] = []
        seen: set[str] = set()
        for post in posts:
            tweet_id = post.get("tweet_id")
            if not tweet_id or tweet_id in seen:
                continue
            ordered.append(post)
            seen.add(tweet_id)

        # Sort by tweet ID for chronological order
        ordered.sort(key=lambda p: int(p['tweet_id']) if p.get('tweet_id', '').isdigit() else 0)

        filtered = self._filter_posts_by_time_window(ordered, start_id)

        if len(filtered) < 2:
            return None

        return ThreadSummary(start_id=start_id, url=url, tweets=filtered)

    def _record_thread(self, thread: ThreadSummary) -> None:
        merged = self._merge_with_existing_thread(thread)
        if merged:
            self._mark_processed_tweets(merged.tweets)
            logger.info(
                "♻️  Expanded thread %s to %d posts -> %s",
                merged.start_id,
                merged.size,
                ",".join(tweet.get("tweet_id", "?") for tweet in merged.tweets),
            )
            return

        if len(self.collected_threads) >= self.max_threads:
            self._mark_processed_tweets(thread.tweets)
            return

        self.collected_threads.append(thread)
        self._mark_processed_tweets(thread.tweets)
        logger.info(
            "✅ Thread captured (%d posts) – start %s -> %s",
            thread.size,
            thread.start_id,
            ",".join(tweet.get("tweet_id", "?") for tweet in thread.tweets),
        )
        self._enforce_thread_limit()

    def _merge_with_existing_thread(self, thread: ThreadSummary) -> Optional[ThreadSummary]:
        new_ids = {tweet["tweet_id"] for tweet in thread.tweets}
        if not new_ids:
            return None

        for existing in self.collected_threads:
            existing_ids = {tweet["tweet_id"] for tweet in existing.tweets}
            
            # Merge if: same conversation_id OR overlapping tweet IDs
            should_merge = False
            
            # Check conversation_id match first (most reliable)
            if thread.conversation_id and existing.conversation_id:
                if thread.conversation_id == existing.conversation_id:
                    should_merge = True
                    logger.info(f"🔗 Merging threads by conversation_id: {thread.conversation_id}")
            
            # Fallback: check overlapping tweet IDs
            if not should_merge and (existing_ids & new_ids):
                should_merge = True
            
            if not should_merge:
                continue

            added = False
            for tweet in thread.tweets:
                tweet_id = tweet.get("tweet_id")
                if not tweet_id:
                    continue
                if tweet_id not in existing_ids:
                    existing.tweets.append(tweet)
                    existing_ids.add(tweet_id)
                    added = True

            if added:
                existing.tweets.sort(key=self._tweet_sort_key)
                
            # Update conversation_id if we didn't have one
            if thread.conversation_id and not existing.conversation_id:
                existing.conversation_id = thread.conversation_id
                
            return existing

        return None

    def _mark_processed_tweets(self, tweets: List[Dict]) -> None:
        for tweet in tweets:
            tweet_id = tweet.get("tweet_id")
            if not tweet_id:
                continue
            self.processed_tweet_ids.add(tweet_id)
            self.indicator_retry_counts.pop(tweet_id, None)

    def _tweet_sort_key(self, tweet: Dict) -> int:
        tweet_id = tweet.get("tweet_id", "")
        return int(tweet_id) if tweet_id.isdigit() else 0

    async def _is_reply_to_other_user(self, article: ElementHandle, username: str) -> bool:
        username = username.lower()
        handles = await self._extract_reply_handles(article)
        return any(handle != username for handle in handles)

    async def _extract_reply_handles(self, article: ElementHandle) -> List[str]:
        handles: set[str] = set()

        context_element = await article.query_selector('[data-testid="socialContext"]')
        if context_element:
            try:
                context_text = await context_element.inner_text()
            except Exception:
                context_text = ""
            handles.update(self._handles_from_text(context_text))

        labelled_elements = await article.query_selector_all('[aria-label]')
        for element in labelled_elements:
            try:
                label = await element.get_attribute("aria-label")
            except Exception:
                continue
            if not label:
                continue
            label_lower = label.lower()
            if not any(marker in label_lower for marker in ARIA_REPLY_MARKERS):
                continue
            handles.update(self._handles_from_text(label_lower))

        labelledby_attr = await article.get_attribute("aria-labelledby")
        if labelledby_attr:
            label_ids = [label.strip() for label in labelledby_attr.split() if label.strip()]
            if label_ids:
                try:
                    label_entries = await article.evaluate(
                        "(node, ids) => ids.map((id) => {\n"
                        "                    const el = node.ownerDocument ? node.ownerDocument.getElementById(id) : null;\n"
                        "                    const text = el ? (el.innerText || el.textContent || '') : '';\n"
                        "                    return { id, text };\n"
                        "                })",
                        label_ids,
                    )
                except Exception:
                    label_entries = []
                for entry in label_entries or []:
                    text = entry.get("text") or ""
                    handles.update(self._handles_from_text(text))

        return list(handles)

    def _handles_from_text(self, text: str) -> List[str]:
        if not text:
            return []
        return [handle.lower() for handle in MENTION_PATTERN.findall(text.lower())]

    def _format_thread_post(self, tweet: Dict) -> Dict:
        tweet_id = tweet["tweet_id"]
        username = tweet.get("username", "")
        tweet_url = tweet.get("tweet_url") or f"https://x.com/{username}/status/{tweet_id}"
        return {
            "tweet_id": tweet_id,
            "tweet_url": tweet_url,
            "content": tweet.get("content", ""),
            "username": username,
            "tweet_timestamp": tweet.get("tweet_timestamp"),
        }

    def _filter_posts_by_time_window(self, posts: List[Dict], start_id: str) -> List[Dict]:
        if THREAD_TIME_WINDOW_MINUTES is None:
            return posts
        if not posts:
            return posts

        start_post = next((post for post in posts if post["tweet_id"] == start_id), None)
        if not start_post:
            return posts

        start_dt = self._get_post_datetime(start_post)
        if not start_dt:
            return posts

        cutoff = start_dt + timedelta(minutes=THREAD_TIME_WINDOW_MINUTES)

        def within_window(post: Dict) -> bool:
            if post["tweet_id"] == start_id:
                return True
            post_dt = self._get_post_datetime(post)
            if not post_dt:
                return True
            return post_dt <= cutoff

        return [post for post in posts if within_window(post)]

    def _get_post_datetime(self, post: Dict) -> Optional[datetime]:
        timestamp = post.get("tweet_timestamp")
        if timestamp:
            try:
                return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                pass

        tweet_id = post.get("tweet_id")
        if tweet_id and tweet_id.isdigit():
            try:
                snowflake = int(tweet_id)
            except ValueError:
                return None
            twitter_epoch_ms = 1288834974657
            timestamp_ms = (snowflake >> 22) + twitter_epoch_ms
            return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        return None

    def _log_summary(self, username: str) -> None:
        if not self.collected_threads:
            logger.info(f"❌ No threads detected for @{username}")
            return

        self._sort_threads_by_recency()
        logger.info("\n" + "=" * 80)
        logger.info(f"THREAD SUMMARY for @{username}")
        logger.info("=" * 80)
        
        for idx, thread in enumerate(self.collected_threads, start=1):
            logger.info(f"\n🧵 Thread #{idx}: {thread.size} posts")
            logger.info(f"   URL: {thread.url}")
            logger.info(f"   Start ID: {thread.start_id}")
            if thread.conversation_id:
                logger.info(f"   Conversation ID: {thread.conversation_id}")
            logger.info("-" * 60)
            
            for post_idx, post in enumerate(thread.tweets, start=1):
                tweet_id = post.get("tweet_id", "?")
                content = post.get("content", "")
                # Truncate long content
                if len(content) > 200:
                    content = content[:200] + "..."
                # Clean up content for display
                content = content.replace("\n", " ").strip()
                timestamp = post.get("tweet_timestamp", "unknown time")
                
                logger.info(f"   [{post_idx}] {tweet_id}")
                logger.info(f"       📅 {timestamp}")
                logger.info(f"       💬 {content}")
            
            logger.info("")
        
        logger.info("=" * 80)
        logger.info(f"Total: {len(self.collected_threads)} threads detected")
        logger.info("=" * 80)

    def _sort_threads_by_recency(self) -> None:
        self.collected_threads.sort(
            key=lambda thread: int(thread.start_id) if thread.start_id.isdigit() else -1,
            reverse=True,
        )

    def _enforce_thread_limit(self) -> None:
        if len(self.collected_threads) <= self.max_threads:
            return
        self._sort_threads_by_recency()
        self.collected_threads = self.collected_threads[: self.max_threads]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("UI-based thread detection tester")
    parser.add_argument(
        "--user",
        default="infovlogger36",
        help="Twitter username to scan (default: infovlogger36)",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=3,
        help="Maximum number of threads to collect",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    tester = ThreadDetectionTester(
        max_threads=args.max_threads,
    )
    await tester.test_thread_detection(args.user)


if __name__ == "__main__":
    asyncio.run(main())
