"""Thread detection utilities for the fetcher.

Provides production-ready thread detection by:
1. Scanning timeline for CSS thread connector indicators (r-1bimlpy)
2. Opening conversation pages in new tabs to validate self-reply chains
3. Walking forward/backward through conversation structure
4. Capturing conversation_id from GraphQL API responses
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Set

from playwright.sync_api import ElementHandle, Page, BrowserContext, sync_playwright, Error as PlaywrightError
from fetcher.config import get_config


class BrowserClosedByUserError(Exception):
    """Raised when the browser is manually closed by the user.
    
    This exception should propagate up to stop thread collection
    entirely, rather than attempting to open new browser instances.
    """
    pass


def is_browser_closed_error(e: Exception) -> bool:
    """Check if an exception indicates the browser was closed by user."""
    error_msg = str(e).lower()
    # Common playwright errors when browser/page is closed
    closed_indicators = [
        'target closed',
        'target page, context or browser has been closed',
        'browser has been closed',
        'context has been closed',
        'page has been closed',
        'connection closed',
        'browser.close',
    ]
    return any(indicator in error_msg for indicator in closed_indicators)

logger = logging.getLogger(__name__)


@dataclass
class ThreadSummary:
    """Lightweight structure describing a discovered thread."""

    start_id: str
    url: str
    tweets: List[Dict]
    conversation_id: Optional[str] = None

    @property
    def size(self) -> int:
        return len(self.tweets)


def _sync_handle(handle: Any) -> Any:
    """Return Playwright sync handle when available."""
    if handle is None:
        return None
    try:
        sync_api = object.__getattribute__(handle, 'sync_api')
        if sync_api is not None:
            return sync_api
    except AttributeError:
        return handle
    except Exception:
        return getattr(handle, 'sync_api', handle)
    return handle


class ThreadDetector:
    """Provides best-effort thread detection from user timelines."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Track expanders that caused menus/overlays so we avoid clicking them again
        self._blocked_expanders = set()

    def detect_threads_on_profile(self, username: str, session_manager, max_scrolls: int = 5) -> List[Dict]:
        """Collect tweets from a profile and group them into approximate threads."""
        threads: List[Dict] = []

        with sync_playwright() as p:
            browser, context, page = session_manager.create_browser_context(p, save_session=False)
            try:
                profile_url = f"https://x.com/{username}"
                self.logger.info(f"🌐 Navigating to {profile_url}")
                page.goto(profile_url, wait_until="domcontentloaded")
                page.wait_for_timeout(2000)

                all_tweets: List[Dict] = []
                seen_ids = set()

                for scroll in range(max_scrolls):
                    self.logger.debug(f"📜 Scroll {scroll + 1}/{max_scrolls}")
                    articles = page.query_selector_all('article[data-testid="tweet"]')

                    for article in articles:
                        tweet_data = self._extract_tweet_from_article(article)
                        if not tweet_data:
                            continue

                        tweet_id = tweet_data['tweet_id']
                        if tweet_id in seen_ids:
                            continue

                        if tweet_data.get('username', '').lower() == username.lower():
                            all_tweets.append(tweet_data)
                            seen_ids.add(tweet_id)

                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    page.wait_for_timeout(2000)

                self.logger.info(f"📊 Collected {len(all_tweets)} tweets from profile")

                if len(all_tweets) < 2:
                    self.logger.info("❌ Not enough tweets for thread detection")
                    return []

                # Use reply-chain detection instead of arbitrary ID gaps
                threads = self._group_into_threads_by_reply_chain(all_tweets, username)

                self.logger.info(f"📊 Found {len(threads)} complete threads")
                return threads
            finally:
                session_manager.cleanup_session(browser, context)

    def detect_thread_start(self, article: ElementHandle, username: str) -> bool:
        """Best-effort detection of thread starts."""
        article_handle = _sync_handle(article)
        return not self._has_thread_line(article_handle)

    def extract_reply_metadata(self, article: ElementHandle) -> Optional[Dict]:
        """Extract simple reply metadata from an article."""
        article_handle = _sync_handle(article)
        try:
            reply_indicator = article_handle.query_selector('[data-testid="Tweet-User-Text"]')
            if not reply_indicator:
                return None

            reply_to_link = article_handle.query_selector('a[href*="/status/"]')
            if reply_to_link:
                href = reply_to_link.get_attribute('href')
                if href and '/status/' in href:
                    tweet_id = href.split('/status/')[1].split('?')[0]
                    return {
                        'reply_to_tweet_id': tweet_id,
                        'replied_to_id': tweet_id,
                        'is_reply': True
                    }

            return None
        except Exception as e:
            self.logger.warning(f"Error extracting reply metadata: {e}")
            return None

    def is_thread_member(self, reply_metadata: Optional[Dict], username: str) -> bool:
        """Determine if reply metadata hints that tweet is part of a thread."""
        if not reply_metadata:
            return False

        if reply_metadata.get('is_reply'):
            return True

        replied_to_username = reply_metadata.get('replies_to_username')
        return bool(replied_to_username and replied_to_username == username)

    def _has_thread_line(self, article: ElementHandle) -> bool:
        try:
            article_handle = _sync_handle(article)
            thread_line_divs = article_handle.query_selector_all('div.css-175oi2r')
            for div in thread_line_divs:
                class_attr = div.get_attribute('class')
                if class_attr and 'r-1bimlpy' in class_attr and 'r-f8sm7e' in class_attr:
                    return True
            return False
        except Exception as e:
            self.logger.warning(f"Error checking thread line: {e}")
            return False

    def _dismiss_specific_overlays(self, page: Page) -> None:
        """Dismiss specific known Twitter overlays that need special handling.
        
        Handles:
        - Cookie consent banner (BottomBar) - click "Accept all cookies"
        - Tweet compose box - close with Escape or close button
        - Grok/Chat drawers - minimize them
        """
        try:
            # 1. Handle cookie consent banner
            cookie_banner = page.query_selector('[data-testid="BottomBar"]')
            if cookie_banner and cookie_banner.is_visible():
                self.logger.debug("Found cookie consent banner, accepting cookies...")
                # Find and click "Accept all cookies" button - try multiple languages
                accept_selectors = [
                    '[data-testid="BottomBar"] button:has-text("Accept")',
                    '[data-testid="BottomBar"] button:has-text("Aceptar")',  # Spanish
                    '[data-testid="BottomBar"] button:has-text("Akzeptieren")',  # German
                    'button:has-text("Accept all cookies")',
                    'button:has-text("Aceptar todas las cookies")',  # Spanish
                    'button:has-text("Alle Cookies akzeptieren")',  # German
                ]
                accept_btn = None
                for sel in accept_selectors:
                    try:
                        accept_btn = page.query_selector(sel)
                        if accept_btn and accept_btn.is_visible():
                            break
                    except Exception:
                        continue
                if accept_btn and accept_btn.is_visible():
                    accept_btn.click()
                    page.wait_for_timeout(500)
                    self.logger.debug("Clicked accept cookies button")
            
            # 2. Handle open compose/reply tweet box (inline reply on tweet pages)
            compose_textarea = page.query_selector('[data-testid="tweetTextarea_0"]')
            if compose_textarea and compose_textarea.is_visible():
                self.logger.debug("Found open compose/reply box, closing it...")
                
                # Method 1: Blur the textarea and press Escape
                page.evaluate("""() => {
                    const textarea = document.querySelector('[data-testid="tweetTextarea_0"]');
                    if (textarea) {
                        textarea.blur();
                        // Also try to blur any focused element
                        document.activeElement?.blur();
                    }
                }""")
                page.keyboard.press('Escape')
                page.wait_for_timeout(300)
                
                # Method 2: Click on tweet text (not media) to shift focus away from compose box
                compose_textarea = page.query_selector('[data-testid="tweetTextarea_0"]')
                if compose_textarea and compose_textarea.is_visible():
                    # Find a safe text element to click (avoid media which would open image viewer)
                    safe_click_targets = [
                        'article[data-testid="tweet"] [data-testid="tweetText"]',
                        'article[data-testid="tweet"] time',
                        'article[data-testid="tweet"] [data-testid="User-Name"]',
                    ]
                    clicked = False
                    for selector in safe_click_targets:
                        try:
                            safe_el = page.query_selector(selector)
                            if safe_el and safe_el.is_visible():
                                safe_el.click()
                                clicked = True
                                page.wait_for_timeout(300)
                                break
                        except Exception:
                            continue
                    if not clicked:
                        # Last resort: press Escape again
                        page.keyboard.press('Escape')
                
                # Method 3: Scroll up to collapse inline reply
                compose_textarea = page.query_selector('[data-testid="tweetTextarea_0"]')
                if compose_textarea and compose_textarea.is_visible():
                    page.evaluate("window.scrollTo(0, 0)")
                    page.wait_for_timeout(300)
                    page.keyboard.press('Escape')
                    page.wait_for_timeout(200)
                
                # Method 4: Click the close button if it exists (modal compose)
                compose_textarea = page.query_selector('[data-testid="tweetTextarea_0"]')
                if compose_textarea and compose_textarea.is_visible():
                    close_btn = page.query_selector('[data-testid="app-bar-close"]')
                    if close_btn and close_btn.is_visible():
                        close_btn.click()
                        page.wait_for_timeout(300)
                
                # Method 5: Hide the inline reply container via JavaScript as last resort
                compose_textarea = page.query_selector('[data-testid="tweetTextarea_0"]')
                if compose_textarea and compose_textarea.is_visible():
                    self.logger.debug("Compose box still visible, hiding via CSS...")
                    page.evaluate("""() => {
                        // Find the inline_reply_offscreen container and hide it
                        const container = document.querySelector('[data-testid="inline_reply_offscreen"]');
                        if (container) {
                            container.style.display = 'none';
                        }
                        // Also hide the tweetTextarea label container
                        const label = document.querySelector('[data-testid="tweetTextarea_0_label"]');
                        if (label) {
                            const parent = label.closest('[data-testid="cellInnerDiv"]');
                            if (parent) {
                                parent.style.display = 'none';
                            }
                        }
                    }""")
                    page.wait_for_timeout(200)
                
                self.logger.debug("Compose/reply box dismissed")
            
            # 3. Minimize Grok drawer if expanded
            grok_drawer = page.query_selector('[data-testid="GrokDrawer"]')
            if grok_drawer and grok_drawer.is_visible():
                # Check if it's expanded (height > 100px typically means expanded)
                is_expanded = page.evaluate("""(el) => {
                    const rect = el.getBoundingClientRect();
                    return rect.height > 100;
                }""", grok_drawer)
                if is_expanded:
                    self.logger.debug("Found expanded Grok drawer, clicking to minimize...")
                    # Click the Grok header to minimize
                    grok_header = page.query_selector('[data-testid="GrokDrawerHeader"]')
                    if grok_header:
                        grok_header.click()
                        page.wait_for_timeout(300)
            
            # 4. Handle any other popup menus
            menu = page.query_selector('[role="menu"]')
            if menu and menu.is_visible():
                self.logger.debug("Found popup menu, pressing Escape...")
                page.keyboard.press('Escape')
                page.wait_for_timeout(300)
                
        except Exception as e:
            self.logger.debug(f"Error dismissing specific overlays: {e}")

    def _dismiss_page_modals(self, page: Page, max_attempts: int = 5) -> bool:
        """Dismiss any modals/overlays that Twitter shows on page load.
        
        Twitter often shows modals like:
        - New post/compose modal
        - User options modal  
        - Login prompts
        - Cookie consent
        
        Returns True if page is now clean, False if modals couldn't be dismissed.
        """
        # First, handle specific known overlays that need special treatment
        self._dismiss_specific_overlays(page)
        
        for attempt in range(max_attempts):
            try:
                # Check for common modal/overlay patterns
                has_modal = page.evaluate("""() => {
                    // Common modal selectors Twitter uses
                    const modalSelectors = [
                        'div[role="dialog"]',
                        '[aria-modal="true"]',
                        '[data-testid="modal"]',
                        '[data-testid="sheetDialog"]',
                        '[data-testid="confirmationSheetDialog"]',
                        'div[aria-label*="modal"]',
                        'div[aria-label*="Modal"]',
                        '[role="menu"]',
                        // Compose tweet modal
                        '[data-testid="tweetButton"]',
                        '[data-testid="toolBar"]',
                        // Layer for overlays
                        '#layers div[data-testid]'
                    ];
                    
                    for (const sel of modalSelectors) {
                        const el = document.querySelector(sel);
                        if (el && el.offsetParent !== null) {
                            // Check if it's actually a blocking modal, not just a tweet button in the sidebar
                            const rect = el.getBoundingClientRect();
                            // If it's centered or covers significant area, it's likely a modal
                            if (rect.width > 300 && rect.height > 200) {
                                return {found: true, selector: sel};
                            }
                        }
                    }
                    
                    // Check #layers specifically for Twitter's overlay system
                    const layers = document.querySelector('#layers');
                    if (layers && layers.children.length > 0) {
                        for (const child of layers.children) {
                            if (child.offsetParent !== null && child.innerHTML.length > 100) {
                                return {found: true, selector: '#layers child'};
                            }
                        }
                    }
                    
                    return {found: false};
                }""")
                
                if not has_modal or not has_modal.get('found'):
                    return True  # No modal found, page is clean
                
                self.logger.debug(f"Modal detected via {has_modal.get('selector')}, attempting to dismiss (attempt {attempt + 1})")
                
                # Try pressing Escape multiple times
                for _ in range(3):
                    try:
                        page.keyboard.press('Escape')
                        page.wait_for_timeout(300)
                    except Exception:
                        pass
                
                # Try clicking close buttons
                close_selectors = [
                    '[data-testid="app-bar-close"]',
                    'button[aria-label*="Close"]',
                    'button[aria-label*="close"]', 
                    'div[aria-label*="Close"]',
                    '[data-testid="close"]',
                    'button[data-testid="xMigrationBottomBar"] button',
                    '[role="button"][aria-label*="Close"]',
                ]
                
                for sel in close_selectors:
                    try:
                        close_btn = page.query_selector(sel)
                        if close_btn and close_btn.is_visible():
                            close_btn.click()
                            page.wait_for_timeout(500)
                            break
                    except Exception:
                        continue
                
                # Click outside modal area (top-left corner)
                try:
                    page.mouse.click(5, 5)
                    page.wait_for_timeout(300)
                except Exception:
                    pass
                
                page.wait_for_timeout(500)
                
            except Exception as e:
                self.logger.debug(f"Error dismissing modal (attempt {attempt + 1}): {e}")
                continue
        
        # Final check
        try:
            still_has_modal = page.evaluate("""() => {
                const layers = document.querySelector('#layers');
                if (layers && layers.children.length > 0) {
                    for (const child of layers.children) {
                        if (child.offsetParent !== null && child.innerHTML.length > 100) {
                            return true;
                        }
                    }
                }
                return false;
            }""")
            if still_has_modal:
                self.logger.warning("Could not dismiss all modals after max attempts")
                return False
        except Exception:
            pass
        
        return True

    def _extract_tweet_from_article(self, article: ElementHandle) -> Optional[Dict]:
        article_handle = _sync_handle(article)
        try:
            tweet_data: Dict = {}

            time_link = article_handle.query_selector('a[href*="/status/"]')
            if time_link:
                href = time_link.get_attribute('href')
                if href:
                    parts = href.strip('/').split('/')
                    if len(parts) >= 3 and parts[1] == 'status':
                        tweet_data['tweet_id'] = parts[2].split('?')[0]
                        if 'username' not in tweet_data and parts[0]:
                            tweet_data['username'] = parts[0]

            username = None
            user_link = article_handle.query_selector('a[href^="/"]')
            if user_link:
                href = user_link.get_attribute('href')
                if href and href.startswith('/'):
                    candidate = href.strip('/').split('/')[0]
                    if candidate and not candidate.startswith('status') and candidate != 'home':
                        username = candidate

            if not username:
                article_text = article_handle.inner_text()
                if '@' in article_text:
                    for line in article_text.split('\n'):
                        line = line.strip()
                        if line.startswith('@') and len(line.split()) == 1:
                            username = line[1:]
                            break

            if username:
                tweet_data['username'] = username

            content_div = article_handle.query_selector('[data-testid="tweetText"]')
            if content_div:
                tweet_data['content'] = content_div.inner_text()
            else:
                tweet_data['content'] = article_handle.inner_text()

            time_element = article_handle.query_selector('time')
            if time_element:
                datetime_attr = time_element.get_attribute('datetime')
                if datetime_attr:
                    tweet_data['tweet_timestamp'] = datetime_attr

            # Extract reply-to tweet ID for thread chain detection
            tweet_data['reply_to_tweet_id'] = self._extract_reply_to_id(article_handle)
            
            # Detect if tweet has thread continuation line
            tweet_data['has_thread_line'] = self._has_thread_line(article_handle)

            if tweet_data.get('tweet_id'):
                self.logger.debug(
                    f"Extracted tweet {tweet_data['tweet_id']} from user {tweet_data.get('username', 'unknown')}"
                )
                return tweet_data

            return None
        except Exception as e:
            self.logger.warning(f"Error extracting tweet data: {e}")
            return None
    
    def _extract_reply_to_id(self, article_handle: ElementHandle) -> Optional[str]:
        """Extract the tweet ID this tweet is replying to, if any."""
        try:
            # Look for "Replying to" section which contains links to replied tweets
            reply_section = article_handle.query_selector('[data-testid="reply"]')
            if reply_section:
                reply_link = reply_section.query_selector('a[href*="/status/"]')
                if reply_link:
                    href = reply_link.get_attribute('href')
                    if href and '/status/' in href:
                        return href.split('/status/')[1].split('?')[0].split('/')[0]
            
            # Alternative: check for in-reply-to in the tweet structure
            # Some tweets show "Replying to @username" with a link
            reply_indicator = article_handle.query_selector('div[id^="id__"] a[href*="/status/"]')
            if reply_indicator:
                href = reply_indicator.get_attribute('href')
                if href and '/status/' in href:
                    return href.split('/status/')[1].split('?')[0].split('/')[0]
            
            return None
        except Exception as e:
            self.logger.warning(f"Error extracting reply-to ID: {e}")
            return None

    def _group_into_threads_by_reply_chain(self, tweets: List[Dict], username: str) -> List[Dict]:
        """Group tweets into threads using reply-chain relationships and thread line indicators.
        
        This method uses two signals to detect threads:
        1. reply_to_tweet_id: Direct reply chain linking tweets together
        2. has_thread_line: Visual thread indicator from Twitter's UI
        
        Falls back to proximity-based grouping only for tweets with thread lines
        but no reply metadata (can happen with Twitter's DOM quirks).
        """
        threads: List[Dict] = []
        
        # Build lookup maps
        tweet_by_id = {t['tweet_id']: t for t in tweets}
        children_map: Dict[str, List[str]] = {}  # parent_id -> [child_ids]
        
        # Build the reply graph
        for tweet in tweets:
            reply_to = tweet.get('reply_to_tweet_id')
            if reply_to:
                if reply_to not in children_map:
                    children_map[reply_to] = []
                children_map[reply_to].append(tweet['tweet_id'])
        
        # Find thread roots: tweets that are replied to but don't reply to anything in our set
        # OR tweets with thread lines that aren't replying to anything
        potential_roots = set()
        
        for tweet in tweets:
            tweet_id = tweet['tweet_id']
            reply_to = tweet.get('reply_to_tweet_id')
            has_line = tweet.get('has_thread_line', False)
            
            # This tweet is a root if:
            # 1. It has children (other tweets reply to it)
            # 2. It doesn't reply to anything in our collected set
            if tweet_id in children_map:
                if not reply_to or reply_to not in tweet_by_id:
                    potential_roots.add(tweet_id)
            
            # Also consider tweets with thread lines that don't reply to anything
            # as potential thread starters
            if has_line and not reply_to:
                potential_roots.add(tweet_id)
        
        # Build threads by following reply chains from each root
        used_tweets = set()
        
        for root_id in potential_roots:
            if root_id in used_tweets:
                continue
                
            thread_tweets = []
            self._collect_thread_chain(root_id, tweet_by_id, children_map, thread_tweets, used_tweets, username)
            
            if len(thread_tweets) >= 2:
                # Sort by tweet ID to maintain chronological order
                thread_tweets.sort(key=lambda x: int(x['tweet_id']))
                threads.append(self._build_thread(username, thread_tweets))
        
        # Handle remaining tweets that have thread lines but weren't connected via replies
        # Group consecutive tweets with thread lines (fallback heuristic)
        remaining = [t for t in tweets if t['tweet_id'] not in used_tweets and t.get('has_thread_line')]
        if remaining:
            remaining.sort(key=lambda x: int(x['tweet_id']))
            current_group: List[Dict] = []
            
            for tweet in remaining:
                if not current_group:
                    current_group = [tweet]
                    continue
                
                # Check if this tweet could be part of current group
                # Use a reasonable time/ID proximity as fallback
                prev_id = int(current_group[-1]['tweet_id'])
                curr_id = int(tweet['tweet_id'])
                
                # Snowflake IDs encode timestamp - rough proximity check
                # IDs within ~1 hour of each other (rough heuristic)
                if curr_id - prev_id < 5000000000000:  # ~1 hour in snowflake time
                    current_group.append(tweet)
                else:
                    if len(current_group) >= 2:
                        threads.append(self._build_thread(username, current_group))
                    current_group = [tweet]
            
            if len(current_group) >= 2:
                threads.append(self._build_thread(username, current_group))
        
        self.logger.info(f"📊 Found {len(threads)} threads using reply-chain detection")
        return threads
    
    def _collect_thread_chain(
        self, 
        tweet_id: str, 
        tweet_by_id: Dict[str, Dict], 
        children_map: Dict[str, List[str]], 
        result: List[Dict], 
        used: set,
        username: str
    ):
        """Recursively collect tweets in a thread chain."""
        if tweet_id in used:
            return
        
        tweet = tweet_by_id.get(tweet_id)
        if not tweet:
            return
        
        # Only include tweets from the same user (self-threads)
        if tweet.get('username', '').lower() != username.lower():
            return
        
        used.add(tweet_id)
        result.append(tweet)
        
        # Follow children (tweets that reply to this one)
        children = children_map.get(tweet_id, [])
        for child_id in children:
            self._collect_thread_chain(child_id, tweet_by_id, children_map, result, used, username)

    def _build_thread(self, username: str, tweets: List[Dict]) -> Dict:
        return {
            'thread_id': tweets[0]['tweet_id'],
            'start_tweet_id': tweets[0]['tweet_id'],
            'username': username,
            'tweets': tweets,
            'tweet_count': len(tweets)
        }

    # =========================================================================
    # Enhanced Thread Detection (ported from test_thread_detection.py)
    # These methods use conversation page navigation for accurate detection
    # =========================================================================

    def detect_threads_with_conversation_validation(
        self,
        username: str,
        session_manager,
        max_threads: Optional[int] = None,
        max_timeline_scrolls: int = 20,
        existing_context: Optional[BrowserContext] = None
    ) -> List[ThreadSummary]:
        """Scan timeline for threads using CSS indicators and validate via conversation pages.
        
        This is the production-ready thread detection that:
        1. Scans user timeline for CSS thread connector (r-1bimlpy)
        2. Opens each potential thread in a new tab
        3. Validates self-reply chain structure
        4. Captures conversation_id from GraphQL API
        
        Args:
            username: Twitter username to scan
            session_manager: SessionManager instance for browser context
            max_threads: Optional maximum number of threads to collect (None = unlimited)
            max_timeline_scrolls: Maximum scroll iterations on timeline
            existing_context: Optional existing BrowserContext to reuse (avoids sync_playwright conflict)
            
        Returns:
            List of validated ThreadSummary objects
        """
        collected_threads: List[ThreadSummary] = []
        processed_tweet_ids: Set[str] = set()
        
        # If we have an existing context, use it. Otherwise create a new one.
        if existing_context is not None:
            return self._detect_threads_with_existing_context(
                username, existing_context, max_threads, max_timeline_scrolls,
                collected_threads, processed_tweet_ids
            )
        
        # No existing context - create our own playwright instance
        with sync_playwright() as p:
            browser, context, page = session_manager.create_browser_context(p, save_session=False)
            try:
                # Navigate to profile
                profile_url = f"https://x.com/{username}"
                self.logger.info(f"🌐 Navigating to {profile_url}")
                page.goto(profile_url, wait_until="domcontentloaded")
                page.wait_for_timeout(3000)
                
                # Scan timeline - collect potential thread starters
                target_username = username.lower()
                seen_ids: Set[str] = set()
                potential_thread_starts: List[Dict] = []  # Tweets without thread connector (potential starts)
                
                for scroll_num in range(max_timeline_scrolls):
                    if max_threads is not None and max_threads is not None and max_threads is not None and len(collected_threads) >= max_threads:
                        self.logger.info(f"🛑 Reached max threads ({max_threads})")
                        break
                    
                    articles = page.query_selector_all('article[data-testid="tweet"]')
                    
                    for article in articles:
                        tweet = self._extract_tweet_from_article(article)
                        if not tweet:
                            continue
                        
                        tweet_id = tweet.get("tweet_id")
                        tweet_username = tweet.get("username", "").lower()
                        
                        if not tweet_id or tweet_id in seen_ids:
                            continue
                        
                        if tweet_username != target_username:
                            continue
                        
                        seen_ids.add(tweet_id)
                        
                        # Check for CSS thread connector (r-1bimlpy)
                        # Tweets WITH connector are thread continuations
                        # Tweets WITHOUT connector could be thread starters
                        article_html = article.inner_html()
                        has_thread_connector = 'r-1bimlpy' in article_html
                        
                        if not has_thread_connector and tweet_id not in processed_tweet_ids:
                            # This could be a thread start - add to check list
                            potential_thread_starts.append(tweet)
                    
                    # Scroll down
                    self._incremental_scroll(page)
                    page.wait_for_timeout(1500)
                
                self.logger.info(f"📊 Found {len(potential_thread_starts)} potential thread starts to check")
                
                # Now check each potential thread start
                for tweet in potential_thread_starts:
                    if len(collected_threads) >= max_threads:
                        break
                    
                    tweet_id = tweet.get("tweet_id")
                    if tweet_id in processed_tweet_ids:
                        continue
                    
                    self.logger.info(f"🔍 Checking potential thread: {tweet_id}")
                    
                    # Open conversation page and check for thread
                    thread = self._collect_thread_from_conversation(
                        context, tweet, target_username
                    )
                    
                    if thread and thread.size >= 2:
                        collected_threads.append(thread)
                        # Mark all tweets in this thread as processed
                        for t in thread.tweets:
                            processed_tweet_ids.add(t.get('tweet_id', ''))
                        self.logger.info(
                            f"✅ Thread #{len(collected_threads)} captured "
                            f"({thread.size} posts)"
                        )
                    else:
                        processed_tweet_ids.add(tweet_id)
                
                self.logger.info(
                    f"📊 Scan complete: {len(collected_threads)} threads found"
                )
                return collected_threads
                
            finally:
                session_manager.cleanup_session(browser, context)

    def _detect_threads_with_existing_context(
        self,
        username: str,
        context: BrowserContext,
        max_threads: int,
        max_timeline_scrolls: int,
        collected_threads: List[ThreadSummary],
        processed_tweet_ids: Set[str]
    ) -> List[ThreadSummary]:
        """Internal method to detect threads using an existing browser context.
        
        This avoids the 'sync_playwright inside asyncio loop' error.
        """
        # Create a new page in the existing context
        page = context.new_page()
        
        try:
            # Navigate to profile
            profile_url = f"https://x.com/{username}"
            self.logger.info(f"🌐 Navigating to {profile_url}")
            page.goto(profile_url, wait_until="domcontentloaded")
            page.wait_for_timeout(3000)
            
            # Scan timeline - collect potential thread starters
            target_username = username.lower()
            seen_ids: Set[str] = set()
            potential_thread_starts: List[Dict] = []  # Tweets without thread connector (potential starts)
            
            for scroll_num in range(max_timeline_scrolls):
                if max_threads is not None and len(collected_threads) >= max_threads:
                    self.logger.info(f"🛑 Reached max threads ({max_threads})")
                    break
                
                articles = page.query_selector_all('article[data-testid="tweet"]')
                
                for article in articles:
                    tweet = self._extract_tweet_from_article(article)
                    if not tweet:
                        continue
                    
                    tweet_id = tweet.get("tweet_id")
                    tweet_username = tweet.get("username", "").lower()
                    
                    if not tweet_id or tweet_id in seen_ids:
                        continue
                    
                    if tweet_username != target_username:
                        continue
                    
                    seen_ids.add(tweet_id)
                    
                    # Check for CSS thread connector (r-1bimlpy)
                    # Tweets WITH connector are thread continuations
                    # Tweets WITHOUT connector could be thread starters
                    article_html = article.inner_html()
                    has_thread_connector = 'r-1bimlpy' in article_html
                    
                    if not has_thread_connector and tweet_id not in processed_tweet_ids:
                        # This could be a thread start - add to check list
                        potential_thread_starts.append(tweet)
                
                # Scroll down
                self._incremental_scroll(page)
                page.wait_for_timeout(1500)
            
            self.logger.info(f"📊 Found {len(potential_thread_starts)} potential thread starts to check")
            
            # Now check each potential thread start
            for tweet in potential_thread_starts:
                if len(collected_threads) >= max_threads:
                    break
                
                tweet_id = tweet.get("tweet_id")
                if tweet_id in processed_tweet_ids:
                    continue
                
                self.logger.info(f"🔍 Checking potential thread: {tweet_id}")
                
                # Open conversation page and check for thread
                thread = self._collect_thread_from_conversation(
                    context, tweet, target_username
                )
                
                if thread and thread.size >= 2:
                    collected_threads.append(thread)
                    # Mark all tweets in this thread as processed
                    for t in thread.tweets:
                        processed_tweet_ids.add(t.get('tweet_id', ''))
                    self.logger.info(
                        f"✅ Thread #{len(collected_threads)} captured "
                        f"({thread.size} posts)"
                    )
                else:
                    processed_tweet_ids.add(tweet_id)
            
            self.logger.info(
                f"📊 Scan complete: {len(collected_threads)} threads found"
            )
            return collected_threads
            
        finally:
            page.close()

    def _collect_thread_from_conversation(
        self,
        context: BrowserContext,
        start_tweet: Dict,
        username: str
    ) -> Optional[ThreadSummary]:
        """Open conversation page and collect self-reply chain.
        
        A valid thread is a sequence where each tweet is a DIRECT REPLY to the 
        previous tweet by the SAME USER. We stop when:
        - Another user's tweet appears in the chain
        - User replies to someone else's comment
        """
        username_lower = username.lower()
        start_id = start_tweet["tweet_id"]
        thread_url = f"https://x.com/{username}/status/{start_id}"
        
        page = context.new_page()
        conversation_ids: Set[str] = set()
        
        # Set up conversation_id extraction via response handler
        def capture_conversation_id(response):
            try:
                # Some GraphQL responses include conversation_id_str, but it can appear in other
                # payloads as well. Inspect body when the key is present to capture it.
                try:
                    url = response.url
                except Exception:
                    url = ''
                body = ''
                try:
                    # Only fetch body if it's likely to contain the token to reduce overhead
                    if '/graphql/' in url or 'conversation_id_str' in url:
                        body = response.text()
                    else:
                        # As a fallback, check small number of responses for the token
                        text_preview = response.text()[:200]
                        if 'conversation_id_str' in text_preview:
                            body = response.text()
                except Exception:
                    body = ''

                if body and 'conversation_id_str' in body:
                    matches = re.findall(r'"conversation_id_str"\s*:\s*"(\d+)"', body)
                    for m in matches:
                        conversation_ids.add(m)
            except Exception:
                pass
        
        page.on("response", capture_conversation_id)
        
        try:
            self.logger.info(f"🧵 Navigating to thread page: {thread_url}")
            page.goto(thread_url, wait_until="domcontentloaded")
            
            # Check actual URL after navigation
            actual_url = page.url
            self.logger.info(f"🧵 Page loaded. Actual URL: {actual_url}")
            
            # Check if we got redirected to a Twitter internal page instead of the tweet
            def is_bad_redirect(url: str) -> bool:
                bad_patterns = [
                    '/explore', '/i/connect_people', '/i/flow/', '/home',
                    '/login', '/signup', '/messages', '/notifications'
                ]
                for pattern in bad_patterns:
                    if pattern in url:
                        return True
                if url.endswith('x.com/') or url.endswith('twitter.com/'):
                    return True
                return False
            
            # If we got redirected to an internal page, try navigating again
            if is_bad_redirect(actual_url):
                self.logger.warning(f"🧵 Redirected to {actual_url}, trying direct navigation again...")
                page.goto(thread_url, wait_until="networkidle")
                actual_url = page.url
                self.logger.info(f"🧵 Second navigation. Actual URL: {actual_url}")
                
                # If still redirected, the tweet might be unavailable
                if is_bad_redirect(actual_url):
                    self.logger.error(f"🧵 Cannot access thread page, redirected to: {actual_url}")
                    return None
            
            page.wait_for_timeout(2000)
            
            # Dismiss any modals Twitter shows on page load (anti-scraping measures)
            if not self._dismiss_page_modals(page):
                self.logger.warning(f"Could not dismiss modals on thread page {thread_url}, attempting to continue")
            
            page.wait_for_timeout(1000)
            
            # Extract thread chain from conversation structure
            thread_chain = self._extract_thread_chain_from_conversation(
                page, username_lower, start_id
            )
            
            if len(thread_chain) < 2:
                return None
            
            posts = [self._format_thread_post(t) for t in thread_chain]
            
            # Use the smallest conversation_id (usually the root tweet)
            conv_id = min(conversation_ids) if conversation_ids else None
            
            return ThreadSummary(
                start_id=start_id,
                url=thread_url,
                tweets=posts,
                conversation_id=conv_id
            )
        finally:
            page.close()

    def _extract_thread_chain_from_conversation(
        self,
        page: Page,
        username_lower: str,
        start_id: str,
        expected_url: str = None
    ) -> List[Dict]:
        """Extract self-reply chain from a conversation page with scrolling.
        
        This method collects ALL consecutive posts from the same user in a thread.
        It will scroll until it finds:
        - A post from a different user (end of thread)
        - Or the page bottom with no more content
        
        There are no artificial limits - a thread can have any number of posts.
        
        A valid self-thread is when the user posts consecutive tweets replying to themselves.
        """
        collected_posts: Dict[str, Dict] = {}  # tweet_id -> tweet_data
        
        def is_on_wrong_page() -> bool:
            """Check if we've been redirected away from the thread page."""
            try:
                current_url = page.url.lower()
                # Check for known bad redirects
                bad_patterns = [
                    '/home', '/explore', '/i/connect_people', '/i/flow/',
                    '/login', '/signup', '/messages', '/notifications',
                    '/search', '/compose', 'foryou', 'for_you'
                ]
                for pattern in bad_patterns:
                    if pattern in current_url:
                        self.logger.warning(f"Detected redirect to {current_url} during thread extraction")
                        return True
                # Check we're still on a status page for the right tweet
                if f'/status/{start_id}' not in current_url and expected_url:
                    # We might be on a different tweet's page - that's ok for thread navigation
                    # But if we're on a completely different page type, abort
                    if '/status/' not in current_url:
                        self.logger.warning(f"No longer on a status page: {current_url}")
                        return True
                return False
            except Exception as e:
                self.logger.debug(f"Error checking URL: {e}")
                return False
        
        def click_show_buttons() -> int:
            """Click all 'Show' buttons that expand collapsed content. Returns count clicked.

            Uses JavaScript-side filtering for speed - avoids expensive Python-side iteration
            over thousands of DOM elements.
            """
            clicked = 0
            # Save URL before clicking to detect navigation
            url_before = page.url
            try:
                # Use JavaScript to find show-reply/expand buttons efficiently
                # Returns list of element info for buttons we should click
                show_buttons = page.evaluate(f"""(targetUser) => {{
                    const results = [];
                    const targetLower = targetUser.toLowerCase();
                    
                    // Only look within the conversation timeline - avoid nav elements
                    const conversation = document.querySelector('[aria-label="Timeline: Conversation"]');
                    const container = conversation || document.body;
                    
                    // Look for elements with "show" text patterns - much more targeted than all buttons
                    const candidates = container.querySelectorAll(
                        '[data-testid*="show"], [data-testid*="reply"], ' +
                        'div[role="button"], button, span'
                    );
                    
                    for (const el of candidates) {{
                        // Skip elements outside conversation area or in nav/header
                        const nav = el.closest('nav, header, [role="navigation"], [data-testid="primaryColumn"] > div:first-child');
                        if (nav) continue;
                        
                        // Skip tab elements
                        if (el.getAttribute('role') === 'tab') continue;
                        if (el.closest('[role="tablist"]')) continue;
                        
                        const text = (el.innerText || '').trim().toLowerCase();
                        const aria = (el.getAttribute('aria-label') || '').toLowerCase();
                        const title = (el.getAttribute('title') || '').toLowerCase();
                        const testid = (el.getAttribute('data-testid') || '').toLowerCase();
                        const href = (el.getAttribute('href') || '').toLowerCase();
                        
                        // Skip if text contains navigation-like words
                        if (/for you|trending|explore|home|search|profile|settings|premium/i.test(text)) continue;
                        
                        // Quick check: is this likely a show/expand button?
                        const combined = text + ' ' + aria + ' ' + title + ' ' + testid;
                        const isExpander = /show|repl(y|ies)|view.*thread|more.*repl/i.test(combined);
                        if (!isExpander) continue;
                        
                        // Skip menu-like elements
                        const hasPopup = (el.getAttribute('aria-haspopup') || '').includes('menu');
                        const isMenu = el.getAttribute('role') === 'menu';
                        if (hasPopup || isMenu) continue;
                        if (testid.includes('more') && !testid.includes('replies')) continue;
                        if (aria.includes('more menu')) continue;
                        
                        // Skip navigation/compose links
                        if (href && (href.includes('/compose') || href.includes('intent/tweet'))) continue;
                        if (href && (href.includes('/explore') || href.includes('/home') || href.includes('/search'))) continue;
                        if (href && !href.includes('/status/') && !href.startsWith('#') && href.length > 1) continue;
                        
                        // Must be inside an article (tweet) to be valid
                        const article = el.closest('article[data-testid="tweet"]');
                        if (!article) continue;  // Skip buttons not inside tweets
                        
                        // Check owning article's user
                        const userLink = article.querySelector('a[href*="/status/"]');
                        if (userLink) {{
                            const parts = userLink.getAttribute('href').split('/').filter(Boolean);
                            if (parts.length >= 2 && parts[1] === 'status') {{
                                const owner = parts[0].toLowerCase();
                                if (owner !== targetLower) continue;  // Skip other users' expand buttons
                            }}
                        }}
                        
                        // Element passes all checks - mark for clicking
                        results.push({{
                            text: text.slice(0, 50),
                            aria: aria.slice(0, 50),
                            testid: testid
                        }});
                        
                        // Click the element directly in JS (faster than round-tripping to Python)
                        try {{
                            el.scrollIntoView({{behavior: 'instant', block: 'center'}});
                            el.click();
                        }} catch (e) {{}}
                    }}
                    
                    return results;
                }}""", username_lower)
                
                if show_buttons:
                    clicked = len(show_buttons)
                    if clicked > 0:
                        self.logger.debug(f"Clicked {clicked} show buttons via JS: {show_buttons}")
                        # Wait for content to load after clicks
                        page.wait_for_timeout(1000)
                        
                        # Check if clicking caused navigation - if so, go back
                        url_after = page.url
                        if url_after != url_before and '/status/' not in url_after:
                            self.logger.warning(f"Button click caused navigation to {url_after}, navigating back")
                            page.go_back()
                            page.wait_for_timeout(1000)
                        
                        # Check if any overlays were opened
                        try:
                            ok = _detect_and_try_close_overlay(attempts=1)
                            if not ok:
                                self.logger.debug("Show button opened an overlay")
                        except Exception:
                            pass
                
            except Exception as e:
                self.logger.debug(f"Error clicking show buttons: {e}")
            return clicked
        
        def collect_visible_tweets():
            """Collect all visible tweets from the conversation and preserve article handles for later expansion."""
            conversation = page.query_selector('[aria-label="Timeline: Conversation"]')
            if conversation:
                articles = conversation.query_selector_all('article[data-testid="tweet"]')
            else:
                articles = page.query_selector_all('article[data-testid="tweet"]')

            for article in articles:
                try:
                    tweet = self._extract_tweet_from_article(article)
                except Exception:
                    tweet = None
                if tweet and tweet.get("tweet_id"):
                    tweet_id = tweet["tweet_id"]
                    if tweet_id not in collected_posts:
                        # preserve a reference to the article so we can click inline show buttons
                        tweet['_article'] = article
                        collected_posts[tweet_id] = tweet

        def click_per_article_show_buttons() -> int:
            """Click inline 'show replies' buttons inside each article to expand deeper content.
            
            This is a simplified version that just calls click_show_buttons() again
            since the JS-optimized version handles all button clicking efficiently.
            """
            # Just re-run the optimized click_show_buttons - it handles per-article buttons too
            return click_show_buttons()

        # First, scroll to top of page to find thread start
        self.logger.info(f"Scrolling to find thread start...")
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(1000)

        # Scroll up to ensure we're at the beginning
        for _ in range(10):
            current_scroll = page.evaluate("window.scrollY")
            if current_scroll <= 0:
                break
            page.evaluate("window.scrollBy(0, -1000)")
            page.wait_for_timeout(500)

        # Initial collection and button clicks
        self.logger.info(f"Initial tweet collection...")
        collect_visible_tweets()
        self.logger.info(f"Initial show buttons click...")
        click_show_buttons()
        self.logger.info(f"Starting main scroll loop...")
        
        # Main scroll loop - scroll down and collect until we hit a non-user post or page bottom
        last_count = 0
        consecutive_empty_scrolls = 0
        scroll_iteration = 0
        start_time = time.time()
        max_time = get_config().thread_collect_timeout_seconds

        def _detect_and_try_close_overlay(attempts: int = 3) -> bool:
            """Detect common modal/overlay patterns and attempt to close them.

            Returns True if no blocking overlay detected or if successfully closed.
            Returns False only if a true blocking modal remains after attempts.
            
            NOTE: This is intentionally conservative - we only detect actual blocking
            modals/dialogs, not just any focused element or UI components.
            """
            try:
                info = page.evaluate("""() => {
                    // Only detect actual blocking modals/dialogs
                    const modalSelectors = [
                        'div[role="dialog"]', 
                        '[aria-modal="true"]', 
                        '[data-testid="modal"]',
                        '[data-testid="sheetDialog"]',
                        '[data-testid="confirmationSheetDialog"]'
                    ];
                    for (const s of modalSelectors) {
                        const el = document.querySelector(s);
                        if (el && el.offsetParent !== null) {
                            // Check if it's actually blocking (covers significant viewport)
                            const rect = el.getBoundingClientRect();
                            if (rect.width > 300 && rect.height > 200) {
                                return {overlay: true, selector: s, blocking: true};
                            }
                        }
                    }

                    // Check for popup menus that block interaction
                    const menu = document.querySelector('[role="menu"]');
                    if (menu && menu.offsetParent !== null) {
                        // Only consider it blocking if it's a substantial popup
                        const rect = menu.getBoundingClientRect();
                        if (rect.width > 150 && rect.height > 100) {
                            return {overlay: true, selector: '[role="menu"]', blocking: true};
                        }
                    }

                    // No blocking overlay detected
                    return {overlay: false};
                }""")
            except Exception:
                return True  # Assume no overlay on error - don't block collection

            if not info or not info.get('overlay'):
                return True

            # Try to close with Escape and common close buttons
            for attempt in range(attempts):
                try:
                    try:
                        page.keyboard.press('Escape')
                    except Exception:
                        pass

                    # Click known close targets
                    close_selectors = [
                        'button[aria-label*="close"]', 'button[title*="close"]', 'button[data-testid="close"]',
                        'div[role="dialog"] button', 'button[aria-label*="dismiss"]',
                        'div[role="menu"] button', 'div[role="menu"] [role="menuitem"]', '[aria-label*="more menu"]', '[aria-label*="more menu items"]'
                    ]
                    for sel in close_selectors:
                        try:
                            btn = page.query_selector(sel)
                            if btn:
                                try:
                                    btn.click()
                                    page.wait_for_timeout(400)
                                except Exception:
                                    continue
                        except Exception:
                            continue

                    # As a last resort, attempt to click outside the overlay to dismiss (click top-left)
                    try:
                        page.mouse.click(10, 10)
                        page.wait_for_timeout(200)
                    except Exception:
                        pass

                    # Give it a moment to disappear
                    page.wait_for_timeout(600)
                    # Re-evaluate
                    try:
                        still = page.evaluate("""() => {
                            // Only check for actual blocking modals
                            const modalSelectors = [
                                'div[role="dialog"]', 
                                '[aria-modal="true"]', 
                                '[data-testid="modal"]',
                                '[data-testid="sheetDialog"]',
                                '[data-testid="confirmationSheetDialog"]'
                            ];
                            for (const s of modalSelectors) {
                                const el = document.querySelector(s);
                                if (el && el.offsetParent !== null) {
                                    const rect = el.getBoundingClientRect();
                                    if (rect.width > 300 && rect.height > 200) return true;
                                }
                            }
                            // Check for blocking popup menus
                            const menu = document.querySelector('[role="menu"]');
                            if (menu && menu.offsetParent !== null) {
                                const rect = menu.getBoundingClientRect();
                                if (rect.width > 150 && rect.height > 100) return true;
                            }
                            return false;
                        }""")
                    except Exception:
                        still = False
                    if not still:
                        self.logger.debug("Overlay detected and closed")
                        return True
                except Exception:
                    continue

            # If we reach here, overlay remains
            try:
                ts = int(time.time())
                try:
                    page.screenshot(path=f"/tmp/thread_abort_{start_id}_{ts}.png")
                except Exception:
                    pass
                try:
                    html = page.content()
                    with open(f"/tmp/thread_abort_{start_id}_{ts}.html", 'w', encoding='utf-8') as fh:
                        fh.write(html[:200000])
                except Exception:
                    pass
            except Exception:
                pass

            self.logger.warning(f"Overlay remained after attempts during thread extraction of {start_id}; captured debugging artifacts")
            return False

        # Track when we've found the end of the thread (non-user post after user posts)
        found_thread_end = False
        has_user_posts = False
        
        self.logger.info(f"Entering main scroll loop (no limits, collecting until thread ends)")
        while True:
            # Global timeout guard
            if time.time() - start_time > max_time:
                self.logger.warning(f"Aborting thread extraction for {start_id}: exceeded timeout ({max_time}s)")
                break
            
            # Check if we've been redirected away from the thread page
            if is_on_wrong_page():
                self.logger.error(f"Aborting thread extraction for {start_id}: page redirected away")
                break
            
            # If we found the thread end (non-user post after user posts), stop immediately
            if found_thread_end:
                self.logger.info(f"Thread end detected: found non-user post after consecutive user posts")
                break
            
            # If we've hit consecutive empty scrolls at the page bottom, stop
            if consecutive_empty_scrolls >= 2:
                self.logger.info(f"Reached page bottom after {consecutive_empty_scrolls} empty scrolls")
                break

            scroll_iteration += 1
            if scroll_iteration % 5 == 0:
                self.logger.info(f"Scroll iteration {scroll_iteration}, collected {len(collected_posts)} posts")
            
            # Scroll down incrementally
            page.evaluate("window.scrollBy(0, 800)")
            page.wait_for_timeout(600)
            
            # Detect and handle overlays that can capture scroll events
            try:
                ok = _detect_and_try_close_overlay()
                if not ok:
                    # Couldn't close overlay; abort
                    self.logger.warning(f"Aborting thread extraction for {start_id}: overlay blocked progress")
                    break
            except Exception as e:
                self.logger.debug(f"Overlay detection/close check failed: {e}")
            
            # Click any expand buttons that became visible
            if scroll_iteration % 2 == 0:
                buttons_clicked = click_show_buttons()
                if buttons_clicked > 0:
                    page.wait_for_timeout(1000)
            
            # Collect new tweets
            prev_count = len(collected_posts)
            collect_visible_tweets()
            current_count = len(collected_posts)
            
            # Check if we collected any user posts
            if not has_user_posts:
                has_user_posts = any(p.get('username', '').lower() == username_lower for p in collected_posts.values())
            
            # Check if any newly collected posts are from other users (indicates end of thread)
            if current_count > prev_count and has_user_posts:
                for post_id, post_data in collected_posts.items():
                    if post_data.get('username', '').lower() != username_lower:
                        # Found a non-user post after we have user posts - thread ends here
                        found_thread_end = True
                        self.logger.debug(f"Found non-user post from @{post_data.get('username', '?')}, marking thread end")
                        break
            
            # Check if we're making progress (for page bottom detection)
            if current_count == last_count:
                consecutive_empty_scrolls += 1
            else:
                consecutive_empty_scrolls = 0
            last_count = current_count
            
            self.logger.debug(f"Scroll {scroll_iteration}: collected {current_count} tweets (empty scrolls: {consecutive_empty_scrolls})")
        
        # Quick final collection pass - just collect any visible tweets we may have missed
        # No need to re-scroll entire page since main loop already did that
        collect_visible_tweets()
        
        if not collected_posts:
            return []
        
        # Build ordered list by tweet ID (chronological order)
        all_tweets = sorted(collected_posts.values(), key=lambda x: int(x.get('tweet_id', '0')))
        
        # Find the thread: consecutive tweets from the same user
        thread_posts: List[Dict] = []
        thread_started = False
        
        for tweet in all_tweets:
            tweet_username = tweet.get("username", "").lower()
            
            if tweet_username == username_lower:
                # This tweet is from our target user
                if not thread_started:
                    # Start of thread
                    thread_posts.append(tweet)
                    thread_started = True
                else:
                    # Continuation of thread
                    thread_posts.append(tweet)
            else:
                # Different user's tweet - if we've already started the thread, this marks the end
                if thread_started:
                    self.logger.debug("Encountered another user's reply; treating it as end of thread")
                    break
                # otherwise ignore non-user tweets before thread starts
                continue
        
        self.logger.info(f"Extracted {len(thread_posts)} posts from thread conversation")
        return thread_posts

    def _collect_parent_thread_posts(
        self,
        article_list: List[tuple],
        focal_idx: int,
        username_lower: str
    ) -> List[Dict]:
        """Walk backwards from focal tweet to find parent thread posts."""
        parent_posts: List[Dict] = []
        
        current_idx = focal_idx
        while current_idx > 0:
            parent_idx = current_idx - 1
            parent_id, parent_user, parent_data = article_list[parent_idx]
            
            if parent_user == username_lower:
                parent_posts.insert(0, parent_data)
                current_idx = parent_idx
            else:
                break
        
        return parent_posts

    def _incremental_scroll(
        self,
        page: Page,
        steps: int = 4,
        step_px: int = 1200
    ) -> None:
        """Scroll the page incrementally to avoid skipping content."""
        for _ in range(steps):
            try:
                page.mouse.wheel(0, step_px)
            except Exception:
                page.evaluate(f"window.scrollBy(0, {step_px})")
            page.wait_for_timeout(400)

    def _format_thread_post(self, tweet: Dict) -> Dict:
        """Format a tweet dict for thread storage."""
        tweet_id = tweet.get("tweet_id", "")
        username = tweet.get("username", "")
        tweet_url = tweet.get("tweet_url") or f"https://x.com/{username}/status/{tweet_id}"
        return {
            "tweet_id": tweet_id,
            "tweet_url": tweet_url,
            "content": tweet.get("content", ""),
            "username": username,
            "tweet_timestamp": tweet.get("tweet_timestamp"),
        }

    def collect_thread_by_id(
        self,
        username: str,
        thread_start_id: str,
        session_manager,
        existing_context=None
    ) -> Optional[ThreadSummary]:
        """Collect a full thread by navigating to its conversation page.

        If an existing BrowserContext is provided via ``existing_context``, this
        method will create a new Page in that context and reuse it for the
        extraction (avoids starting a new Playwright instance and prevents
        sync-in-async issues when called from the collector).

        Args:
            username: Twitter username
            thread_start_id: Tweet ID of the thread's first post
            session_manager: SessionManager for browser context
            existing_context: Optional existing BrowserContext to reuse

        Returns:
            ThreadSummary with all thread posts, or None if not a thread
        """

        def _collect_using_page(page, context):
            """Inner helper that runs collection logic using the provided page and context."""
            thread_url = f"https://x.com/{username}/status/{thread_start_id}"
            self.logger.info(f"🧵 Collecting thread from: {thread_url}")

            conversation_ids: Set[str] = set()

            # Capture conversation_id from GraphQL responses
            def capture_conversation_id(response):
                try:
                    try:
                        url = response.url
                    except Exception:
                        url = ''
                    body = ''
                    try:
                        if '/graphql/' in url or 'conversation_id_str' in url:
                            body = response.text()
                        else:
                            preview = response.text()[:200]
                            if 'conversation_id_str' in preview:
                                body = response.text()
                    except Exception:
                        body = ''

                    if body and 'conversation_id_str' in body:
                        matches = re.findall(r'"conversation_id_str"\s*:\s*"(\d+)"', body)
                        for m in matches:
                            conversation_ids.add(m)
                except Exception:
                    pass

            try:
                page.on("response", capture_conversation_id)
            except Exception:
                # If page.on fails, continue without capturing conversation ids
                pass

            try:
                self.logger.info(f"🧵 Navigating to thread page: {thread_url}")
                page.goto(thread_url, wait_until="domcontentloaded")
                
                # Check actual URL after navigation
                actual_url = page.url
                self.logger.info(f"🧵 Page loaded. Actual URL: {actual_url}")
                
                # Check if we got redirected to a Twitter internal page instead of the tweet
                def is_bad_redirect(url: str) -> bool:
                    bad_patterns = [
                        '/explore', '/i/connect_people', '/i/flow/', '/home',
                        '/login', '/signup', '/messages', '/notifications'
                    ]
                    for pattern in bad_patterns:
                        if pattern in url:
                            return True
                    if url.endswith('x.com/') or url.endswith('twitter.com/'):
                        return True
                    return False
                
                # If we got redirected to an internal page, try navigating again
                if is_bad_redirect(actual_url):
                    self.logger.warning(f"🧵 Redirected to {actual_url}, trying direct navigation again...")
                    page.goto(thread_url, wait_until="networkidle")
                    actual_url = page.url
                    self.logger.info(f"🧵 Second navigation. Actual URL: {actual_url}")
                    
                    # If still redirected, the tweet might be unavailable
                    if is_bad_redirect(actual_url):
                        self.logger.error(f"🧵 Cannot access thread page, redirected to: {actual_url}")
                        return None
                
                page.wait_for_timeout(2000)
                
                # Verify we're still on the right page after wait (Twitter can redirect after load)
                actual_url = page.url
                if is_bad_redirect(actual_url):
                    self.logger.error(f"🧵 Page redirected after load to: {actual_url}")
                    return None
                
                # Wait for tweet article to be present - confirms we're on a tweet page
                try:
                    page.wait_for_selector('article[data-testid="tweet"]', timeout=10000)
                except Exception as e:
                    self.logger.error(f"🧵 No tweet article found on page - may have redirected. URL: {page.url}")
                    return None
                
                # Double-check URL one more time
                actual_url = page.url
                if is_bad_redirect(actual_url):
                    self.logger.error(f"🧵 Page redirected to: {actual_url}")
                    return None
                
                # Dismiss any modals Twitter shows on page load
                self.logger.info(f"🧵 Dismissing modals...")
                self._dismiss_page_modals(page)
                page.wait_for_timeout(1000)
                
                # Final URL check after modal dismissal
                actual_url = page.url
                if is_bad_redirect(actual_url):
                    self.logger.error(f"🧵 Page redirected after modal dismissal to: {actual_url}")
                    return None

                # Extract full thread chain with scrolling
                self.logger.info(f"🧵 Extracting thread chain...")
                thread_chain = self._extract_thread_chain_from_conversation(
                    page, username.lower(), thread_start_id, expected_url=thread_url
                )
                self.logger.info(f"🧵 Extracted {len(thread_chain)} posts from thread")

                if len(thread_chain) < 2:
                    self.logger.info(f"Not a thread or only 1 post found")
                    return None

                # Multi-point expansion for long threads
                existing_ids = {t['tweet_id'] for t in thread_chain}
                expanded = False

                try:
                    conversation = page.query_selector('[aria-label="Timeline: Conversation"]')
                    if conversation:
                        articles = conversation.query_selector_all('article[data-testid="tweet"]')
                    else:
                        articles = page.query_selector_all('article[data-testid="tweet"]')

                    candidate_ids = []
                    for article in articles:
                        try:
                            tweet = self._extract_tweet_from_article(article)
                        except Exception:
                            tweet = None
                        if tweet and tweet.get('tweet_id'):
                            tid = tweet['tweet_id']
                            if tid not in existing_ids and tid not in candidate_ids:
                                if tweet.get('username', '').lower() == username.lower():
                                    candidate_ids.insert(0, tid)
                                else:
                                    candidate_ids.append(tid)

                    # Also scan for raw status links in the DOM
                    try:
                        anchors = page.query_selector_all('a[href*="/status/"]')
                        for a in anchors:
                            try:
                                href = a.get_attribute('href') or ''
                                if '/status/' in href:
                                    tid = href.split('/status/')[1].split('?')[0].split('/')[0]
                                    if tid and tid not in existing_ids and tid not in candidate_ids:
                                        candidate_ids.append(tid)
                            except Exception:
                                continue
                    except Exception:
                        pass

                    self.logger.debug(f"Candidate ids for expansion: {candidate_ids}")
                    max_attempts = 20
                    attempts = 0
                    no_new_rounds = 0

                    # Abort conditions: stop if we try too many candidates with no new tweets
                    # or if total time spent exceeds configured timeout to avoid hanging.
                    start_time = time.time()
                    max_time = get_config().thread_collect_timeout_seconds
                    max_no_progress = get_config().thread_collect_max_no_progress_attempts

                    for candidate_id in candidate_ids:
                        if attempts >= max_attempts:
                            break

                        # Time-based abort
                        if time.time() - start_time > max_time:
                            self.logger.warning(f"Aborting thread expansion for {thread_start_id}: exceeded timeout ({max_time}s)")
                            break

                        attempts += 1
                        page2 = None
                        try:
                            page2 = context.new_page()
                            page2.goto(f"https://x.com/{username}/status/{candidate_id}", wait_until="domcontentloaded")
                            page2.wait_for_timeout(800)
                            # Dismiss any modals Twitter shows on page load
                            self._dismiss_page_modals(page2, max_attempts=3)
                            extra_chain = self._extract_thread_chain_from_conversation(page2, username.lower(), candidate_id)
                            new_ids = 0
                            for t in extra_chain:
                                if t['tweet_id'] not in existing_ids:
                                    thread_chain.append(t)
                                    existing_ids.add(t['tweet_id'])
                                    new_ids += 1
                            if new_ids > 0:
                                expanded = True
                                no_new_rounds = 0
                            else:
                                no_new_rounds += 1

                            # No-progress attempt-based abort
                            if no_new_rounds >= max_no_progress:
                                self.logger.warning(f"Aborting thread expansion for {thread_start_id}: {no_new_rounds} consecutive candidates produced no new tweets")
                                break

                        except Exception as e:
                            # If a navigation or page error happens, log and continue to next candidate
                            self.logger.debug(f"Candidate expansion failed for {candidate_id}: {e}")
                        finally:
                            try:
                                if page2:
                                    page2.close()
                            except Exception:
                                pass
                except Exception:
                    pass

                # Consolidate final posts and format
                unique_by_id = {t['tweet_id']: t for t in thread_chain}
                merged = sorted(unique_by_id.values(), key=lambda x: int(x.get('tweet_id', '0')))

                posts = [self._format_thread_post(t) for t in merged]
                conv_id = min(conversation_ids) if conversation_ids else thread_start_id

                thread = ThreadSummary(
                    start_id=thread_start_id,
                    url=thread_url,
                    tweets=posts,
                    conversation_id=conv_id
                )

                self.logger.info(f"✅ Collected thread with {thread.size} posts (expanded={expanded})")
                return thread

            finally:
                try:
                    page.off("response", capture_conversation_id)
                except Exception:
                    pass

        # Use existing context if provided (preferred) to avoid creating a separate Playwright instance
        if existing_context is not None:
            page = None
            try:
                page = existing_context.new_page()
                result = _collect_using_page(page, existing_context)
                return result
            except (PlaywrightError, Exception) as e:
                if is_browser_closed_error(e):
                    self.logger.warning("Browser was closed by user - stopping thread collection")
                    raise BrowserClosedByUserError("User closed the browser") from e
                raise
            finally:
                try:
                    if page:
                        page.close()
                except Exception:
                    pass
        else:
            # Create a fresh playwright context and clean it up afterwards
            try:
                with sync_playwright() as p:
                    browser, context, page = session_manager.create_browser_context(p, save_session=False)
                    try:
                        return _collect_using_page(page, context)
                    except (PlaywrightError, Exception) as e:
                        if is_browser_closed_error(e):
                            self.logger.warning("Browser was closed by user - stopping thread collection")
                            raise BrowserClosedByUserError("User closed the browser") from e
                        raise
                    finally:
                        try:
                            session_manager.cleanup_session(browser, context)
                        except Exception:
                            pass
            except BrowserClosedByUserError:
                # Re-raise browser closed error to propagate it up
                raise
            except (PlaywrightError, Exception) as e:
                if is_browser_closed_error(e):
                    self.logger.warning("Browser was closed by user - stopping thread collection")
                    raise BrowserClosedByUserError("User closed the browser") from e
                raise

    def save_thread_to_database(
        self,
        thread: ThreadSummary,
        conn,
        username: str
    ) -> int:
        """Save thread metadata to database, updating tweets with thread info.
        
        Args:
            thread: ThreadSummary object with thread data
            conn: Database connection
            username: Username for the thread owner
            
        Returns:
            Number of tweets updated with thread metadata
        """
        from . import db as fetcher_db
        
        updated_count = 0
        thread_id = thread.start_id
        
        for position, tweet in enumerate(thread.tweets):
            tweet_id = tweet.get('tweet_id')
            if not tweet_id:
                continue
            
            # Build tweet data with thread metadata
            tweet_data = {
                'tweet_id': tweet_id,
                'username': username,
                'content': tweet.get('content'),
                'tweet_url': tweet.get('tweet_url'),
                'tweet_timestamp': tweet.get('tweet_timestamp'),
                'post_type': 'thread',
                'thread_id': thread_id,
                'thread_position': position,
                'is_thread_start': 1 if position == 0 else 0,
                'conversation_id': thread.conversation_id,
            }
            
            # Determine reply_to_tweet_id from previous tweet in thread
            if position > 0:
                prev_tweet = thread.tweets[position - 1]
                tweet_data['reply_to_tweet_id'] = prev_tweet.get('tweet_id')
            
            if fetcher_db.save_tweet(conn, tweet_data):
                updated_count += 1
        
        self.logger.info(
            f"💾 Saved thread {thread_id}: {updated_count}/{len(thread.tweets)} tweets updated"
        )
        return updated_count