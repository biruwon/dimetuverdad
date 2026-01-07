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

from playwright.sync_api import ElementHandle, Page, BrowserContext, sync_playwright
from fetcher.config import get_config

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
            page.goto(thread_url, wait_until="domcontentloaded")
            page.wait_for_timeout(3000)
            
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
        max_scroll_attempts: int = 100
    ) -> List[Dict]:
        """Extract self-reply chain from a conversation page with scrolling.
        
        This method handles long threads (50+ posts) by:
        1. First scrolling UP to find the thread start
        2. Clicking all "Show replies" buttons to expand collapsed sections
        3. Scrolling DOWN through all thread posts
        4. Collecting all consecutive posts from the same user
        
        A valid self-thread is when the user posts consecutive tweets replying to themselves.
        """
        collected_posts: Dict[str, Dict] = {}  # tweet_id -> tweet_data
        
        def click_show_buttons() -> int:
            """Click all 'Show' buttons that expand collapsed content. Returns count clicked.

            Only click expanders that are either global (no owning article) or whose owning
            article belongs to the target user. This avoids expanding replies under other
            users that can pull in unrelated content and cause long/no-ending loops.
            """
            clicked = 0
            try:
                # Search a wide set of possible elements that can act as expanders
                candidates = page.query_selector_all('div[role="button"], button, a, span')
                for el in candidates:
                    try:
                        btn_text = (el.inner_text() or '').strip().lower()
                        aria = (el.get_attribute('aria-label') or '').strip().lower()
                        title = (el.get_attribute('title') or '').strip().lower()
                        data_testid = (el.get_attribute('data-testid') or '').strip().lower()
                        href = (el.get_attribute('href') or '').strip().lower()

                        # Create a short identifying snippet for blacklisting
                        snippet = (btn_text or aria or title or data_testid or href or '')[:200]
                        if snippet in self._blocked_expanders:
                            self.logger.debug(f"Skipping previously blocked expander: {snippet}")
                            continue

                        # Avoid clicking links or controls that navigate to compose/new-post pages early
                        if href and ('/compose' in href or 'intent/tweet' in href or '/i/compose' in href or 'compose' in href):
                            self._blocked_expanders.add(snippet)
                            self.logger.debug(f"Skipping navigation expander (compose/link): {snippet} -> {href}")
                            continue
                        if href and '/status/' not in href and not href.startswith('#') and not href.startswith('/'):
                            # generic external/nav links - block to be safe (we only want in-thread expands)
                            self._blocked_expanders.add(snippet)
                            self.logger.debug(f"Skipping generic navigation expander: {snippet} -> {href}")
                            continue

                        # Decide if this element is likely an expander
                        is_expander = False
                        for candidate_text in (btn_text, aria, title, data_testid):
                            if not candidate_text:
                                continue
                            if 'show' in candidate_text or 'reply' in candidate_text or 'more' in candidate_text:
                                is_expander = True
                                break

                        if not is_expander:
                            continue

                        # Get additional attributes to decide if this is a menu-like control
                        role = (el.get_attribute('role') or '').strip().lower()
                        aria_haspopup = (el.get_attribute('aria-haspopup') or '').strip().lower()

                        # Conservative safe-expander regex matches explicit 'show replies' / 'view thread' patterns
                        safe_re = re.compile(r"(show\s(replies|conversation)|view thread|show more replies|view conversation)", re.I)

                        # If the element is menu-like, treat as risky and skip clicking to avoid opening menus
                        is_menu_like = False
                        if aria_haspopup and 'menu' in aria_haspopup:
                            is_menu_like = True
                        if role in ('menu', 'menubar'):
                            is_menu_like = True
                        if (data_testid and 'more' in data_testid) and not safe_re.search(btn_text or ''):
                            is_menu_like = True
                        if (aria and ('more menu' in aria or 'more menu items' in aria)):
                            is_menu_like = True

                        if is_menu_like and not safe_re.search(btn_text or ''):
                            # Blacklist it and skip clicking
                            self._blocked_expanders.add(snippet)
                            self.logger.debug(f"Skipping risky expander (likely menu): {snippet}")
                            continue

                        # Try to find the owning article (if any) and ensure it belongs to target user
                        owner_href = None
                        try:
                            owner_href = el.evaluate("e => { const a = e.closest('article[data-testid=\"tweet\"]'); if (!a) return null; const l = a.querySelector('a[href*=\"/status/\"]'); return l ? l.getAttribute('href') : null }")
                        except Exception:
                            owner_href = None

                        owner_username = None
                        if owner_href:
                            try:
                                parts = owner_href.strip('/').split('/')
                                if parts and len(parts) >= 2 and parts[1] == 'status':
                                    owner_username = parts[0].lower()
                            except Exception:
                                owner_username = None

                        # If the expander is attached to another user, skip it
                        if owner_username and owner_username != username_lower:
                            continue

                        # Avoid clicking unrelated UI like 'More menu' when aria suggests menu
                        if 'menu' in aria and 'more' not in btn_text:
                            continue

                        # Avoid clicking links or controls that navigate to compose/new-post pages
                        href = (el.get_attribute('href') or '').strip().lower()
                        onclick = (el.get_attribute('onclick') or '').strip().lower()
                        if href and ('/compose' in href or 'intent/tweet' in href or '/i/compose' in href or 'compose' in href):
                            self._blocked_expanders.add(snippet)
                            self.logger.debug(f"Skipping navigation expander (compose/link): {snippet} -> {href}")
                            continue
                        if data_testid and ('compose' in data_testid or 'newtweet' in data_testid or 'tweet' in data_testid) and not safe_re.search(btn_text or ''):
                            self._blocked_expanders.add(snippet)
                            self.logger.debug(f"Skipping data-test compose-like expander: {snippet} -> {data_testid}")
                            continue

                        # If element appears to be a generic link that does not point to a status, skip it
                        if href and '/status/' not in href and not href.startswith('#'):
                            self._blocked_expanders.add(snippet)
                            self.logger.debug(f"Skipping generic link expander: {snippet} -> {href}")
                            continue

                        try:
                            el.scroll_into_view_if_needed()
                        except Exception:
                            pass
                        page.wait_for_timeout(250)
                        try:
                            el.click()
                            page.wait_for_timeout(900)

                            # After clicking, if this opened a menu/overlay we should blacklist it
                            try:
                                ok = _detect_and_try_close_overlay(attempts=1)
                                if not ok:
                                    self._blocked_expanders.add(snippet)
                                    self.logger.debug(f"Clicked expander opened an overlay; blacklisting: {snippet}")
                                    continue
                            except Exception:
                                pass

                            clicked += 1
                            self.logger.debug(f"Clicked expander (text/aria/title/testid): '{btn_text}' / '{aria}' / '{title}' / '{data_testid}' (owner={owner_username})")
                        except Exception:
                            continue
                    except Exception:
                        continue
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

            Only click buttons on articles authored by the target user to avoid expanding
            other users' reply trees (which can cause unrelated expansions and loops).
            """
            clicked = 0
            for tweet_id, tweet in list(collected_posts.items()):
                # Only expand articles authored by the target user
                if tweet.get('username', '').lower() != username_lower:
                    continue

                article = tweet.get('_article')
                if not article:
                    continue
                try:
                    # Try multiple button selectors within the article
                    for btn in article.query_selector_all('button, div[role="button"], a, span'):
                        try:
                            txt = (btn.inner_text() or '').strip().lower()
                            aria = (btn.get_attribute('aria-label') or '').strip().lower()
                            title = (btn.get_attribute('title') or '').strip().lower()
                            data_testid = (btn.get_attribute('data-testid') or '').strip().lower()

                            is_expander = False
                            for candidate_text in (txt, aria, title, data_testid):
                                if not candidate_text:
                                    continue
                                if 'show' in candidate_text or 'reply' in candidate_text or 'more' in candidate_text or 'repl' in candidate_text:
                                    is_expander = True
                                    break

                            if not is_expander:
                                continue

                            # Extra checks to avoid menu-like controls
                            role = (btn.get_attribute('role') or '').strip().lower()
                            aria_haspopup = (btn.get_attribute('aria-haspopup') or '').strip().lower()
                            snippet = (txt or aria or title or data_testid or '')[:200]
                            if snippet in self._blocked_expanders:
                                self.logger.debug(f"Skipping previously blocked per-article expander: {snippet}")
                                continue

                            safe_re = re.compile(r"(show\s(replies|conversation)|view thread|show more replies|view conversation)", re.I)
                            is_menu_like = False
                            if aria_haspopup and 'menu' in aria_haspopup:
                                is_menu_like = True
                            if role in ('menu', 'menubar'):
                                is_menu_like = True
                            if (data_testid and 'more' in data_testid) and not safe_re.search(txt or ''):
                                is_menu_like = True
                            if (aria and ('more menu' in aria or 'more menu items' in aria)):
                                is_menu_like = True

                            if is_menu_like and not safe_re.search(txt or ''):
                                self._blocked_expanders.add(snippet)
                                self.logger.debug(f"Skipping risky per-article expander (likely menu): {snippet}")
                                continue

                            try:
                                btn.scroll_into_view_if_needed()
                            except Exception:
                                pass

                            # Avoid clicking compose/navigation anchors inside articles
                            href = (btn.get_attribute('href') or '').strip().lower()
                            if href and ('/compose' in href or 'intent/tweet' in href or '/i/compose' in href or 'compose' in href):
                                self._blocked_expanders.add(snippet)
                                self.logger.debug(f"Skipping per-article navigation expander: {snippet} -> {href}")
                                continue

                            try:
                                btn.click()
                                page.wait_for_timeout(600)

                                # If clicking opened an overlay/menu, blacklist it
                                try:
                                    ok = _detect_and_try_close_overlay(attempts=1)
                                    if not ok:
                                        self._blocked_expanders.add(snippet)
                                        self.logger.debug(f"Clicked per-article expander opened an overlay; blacklisting: {snippet}")
                                        continue
                                except Exception:
                                    pass

                                clicked += 1
                            except Exception:
                                continue
                        except Exception:
                            continue
                except Exception:
                    continue
            if clicked:
                self.logger.debug(f"Clicked {clicked} per-article show-buttons (target user only)")
            return clicked

        # First, scroll to top of page to find thread start
        self.logger.debug(f"Scrolling to find thread start...")
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
        collect_visible_tweets()
        click_show_buttons()
        
        # Main scroll loop - scroll down and collect
        last_count = 0
        stall_count = 0
        scroll_attempt = 0
        max_stall = 10  # Allow more stalls for long threads
        start_time = time.time()
        max_time = get_config().thread_collect_timeout_seconds

        def _detect_and_try_close_overlay(attempts: int = 3) -> bool:
            """Detect common modal/overlay patterns and attempt to close them.

            Returns True if an overlay was detected and successfully closed or
            False if an overlay remains after attempts.
            """
            try:
                info = page.evaluate("""() => {
                    // Common modal/dialog selectors
                    const selectors = ['div[role="dialog"]', '[aria-modal="true"]', '[data-testid="modal"]', '.modal', 'div[aria-label*="modal"]'];
                    for (const s of selectors) {
                        const el = document.querySelector(s);
                        if (el && el.offsetParent !== null) {
                            return {overlay: true, selector: s, snippet: el.outerHTML.substring(0, 1000)};
                        }
                    }

                    // Menus and 'More' menus (e.g., 'More menu items')
                    const menu = document.querySelector('[role="menu"]');
                    if (menu && menu.offsetParent !== null) {
                        const aria = (menu.getAttribute('aria-label') || '').toLowerCase();
                        if (aria.includes('more') || aria.includes('menu')) {
                            return {overlay: true, selector: '[role="menu"]', snippet: menu.outerHTML.substring(0, 1000)};
                        }
                        return {overlay: true, selector: '[role="menu"]', snippet: menu.outerHTML.substring(0, 1000)};
                    }

                    // More conservative aria-label scan to catch items like 'More menu items'
                    const els = document.querySelectorAll('[aria-label]');
                    for (const el of els) {
                        try {
                            const a = (el.getAttribute('aria-label') || '').toLowerCase();
                            if (a.includes('more menu') || a.includes('more menu items') || a.includes('more')) {
                                if (el && el.offsetParent !== null) {
                                    return {overlay: true, selector: '[aria-label]', label: a.slice(0, 200), snippet: el.outerHTML.substring(0, 1000)};
                                }
                            }
                        } catch (e) {
                            // ignore
                        }
                    }

                    if (document.activeElement && document.activeElement.tagName && document.activeElement.tagName.toLowerCase() !== 'body' && document.activeElement.tagName.toLowerCase() !== 'html') {
                        return {overlay: true, selector: 'activeElement', tag: document.activeElement.tagName, snippet: (document.activeElement.outerHTML||'').substring(0,1000)};
                    }
                    return {overlay: false};
                }""")
            except Exception:
                return False

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
                            const selectors = ['div[role="dialog"]', '[aria-modal="true"]', '[data-testid="modal"]', '.modal', 'div[aria-label*="modal"]', '[role="menu"]'];
                            for (const s of selectors) {
                                const el = document.querySelector(s);
                                if (el && el.offsetParent !== null) return true;
                            }
                            if (document.activeElement && document.activeElement.tagName && document.activeElement.tagName.toLowerCase() !== 'body' && document.activeElement.tagName.toLowerCase() !== 'html') return true;
                            // also check for aria-labels that indicate 'more menu'
                            const els = document.querySelectorAll('[aria-label]');
                            for (const el of els) {
                                try {
                                    const a = (el.getAttribute('aria-label')||'').toLowerCase();
                                    if (a.includes('more menu') || a.includes('more menu items') || a === 'more') return true;
                                } catch (e) {}
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

        while scroll_attempt < max_scroll_attempts and stall_count < max_stall:
            # Global timeout guard
            if time.time() - start_time > max_time:
                self.logger.warning(f"Aborting thread extraction for {start_id}: exceeded timeout ({max_time}s)")
                break

            scroll_attempt += 1
            
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
            
            # Click any expand buttons that became visible (more frequently)
            if scroll_attempt % 2 == 0:
                buttons_clicked = click_show_buttons()
                if buttons_clicked > 0:
                    # Reset stall count if we found new buttons
                    stall_count = 0
                    # Give time for content to load after clicking
                    page.wait_for_timeout(1000)
            
            # Collect new tweets
            collect_visible_tweets()
            
            # Check if we're making progress
            current_count = len(collected_posts)
            if current_count == last_count:
                stall_count += 1
                # When stalling, try per-article expansion first on early stalls
                if stall_count in (2, 4, 6):
                    per_clicked = click_per_article_show_buttons()
                    if per_clicked > 0:
                        stall_count = 0
                        page.wait_for_timeout(1000)
                        collect_visible_tweets()
                        continue
                # Additional fallback strategies
                if stall_count == 3:
                    click_show_buttons()
                    page.evaluate("window.scrollBy(0, 500)")
                elif stall_count == 5:
                    # Scroll back up a bit and try per-article expansion again
                    page.evaluate("window.scrollBy(0, -500)")
                    page.wait_for_timeout(500)
                    per_clicked = click_per_article_show_buttons()
                    if per_clicked > 0:
                        stall_count = 0
                        page.wait_for_timeout(1000)
                        collect_visible_tweets()
                elif stall_count == 7:
                    # Try scrolling to specific positions and do a per-article pass
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
                    page.wait_for_timeout(500)
                    per_clicked = click_per_article_show_buttons()
                    if per_clicked > 0:
                        stall_count = 0
                        page.wait_for_timeout(1000)
                        collect_visible_tweets()
            else:
                stall_count = 0
            last_count = current_count
            
            self.logger.debug(f"Scroll {scroll_attempt}: collected {current_count} tweets (stall: {stall_count})")
        
        # Final pass: scroll through entire page clicking buttons
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(500)
        for _ in range(20):
            click_show_buttons()
            page.evaluate("window.scrollBy(0, 500)")
            page.wait_for_timeout(400)
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
                page.goto(thread_url, wait_until="domcontentloaded")
                page.wait_for_timeout(3000)

                # Extract full thread chain with scrolling
                thread_chain = self._extract_thread_chain_from_conversation(
                    page, username.lower(), thread_start_id
                )

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
                            extra_chain = self._extract_thread_chain_from_conversation(page2, username.lower(), candidate_id, max_scroll_attempts=60)
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
            finally:
                try:
                    if page:
                        page.close()
                except Exception:
                    pass
        else:
            # Create a fresh playwright context and clean it up afterwards
            with sync_playwright() as p:
                browser, context, page = session_manager.create_browser_context(p, save_session=False)
                try:
                    return _collect_using_page(page, context)
                finally:
                    try:
                        session_manager.cleanup_session(browser, context)
                    except Exception:
                        pass

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