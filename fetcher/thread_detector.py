"""Thread detection utilities for the fetcher."""

import logging
from typing import Optional, Dict, List, Any

from playwright.sync_api import ElementHandle, sync_playwright

logger = logging.getLogger(__name__)


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