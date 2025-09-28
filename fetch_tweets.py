import os
import sqlite3
import time
import argparse
import sys
import json
import random
import re
from typing import Dict, List, Optional, Tuple
try:
    from dotenv import load_dotenv
except Exception:
    # Allow running examples and tests without python-dotenv installed.
    def load_dotenv():
        return None
try:
    from playwright.sync_api import sync_playwright, TimeoutError
except Exception:
    # Playwright may not be available in lightweight test environments.
    sync_playwright = None
    class TimeoutError(Exception):
        pass
from datetime import datetime


# Load credentials from .env
load_dotenv()
USERNAME = os.getenv("X_USERNAME")
PASSWORD = os.getenv("X_PASSWORD")
EMAIL_OR_PHONE = os.getenv("X_EMAIL_OR_PHONE")

# Randomize user agents to avoid detection
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
]

DB_PATH = "accounts.db"

# Default target handles (focusing on Spanish far-right accounts)
DEFAULT_HANDLES = [
    "vox_es",
    "Santi_ABASCAL", 
    "eduardomenoni",
    "IdiazAyuso",
    "CapitanBitcoin",
    "vitoquiles",
    "wallstwolverine", 
    "WillyTolerdoo",
    "Agenda2030_",
    "Doct_Tricornio",
    "LosMeconios"
]

def human_delay(min_seconds: float = 1.0, max_seconds: float = 3.0):
    """Add human-like random delays to avoid detection."""
    delay = random.uniform(min_seconds, max_seconds)
    time.sleep(delay)

def random_scroll_pattern(page, deep_scroll=False):
    """Implement human-like scrolling patterns with deep scrolling capability."""
    if deep_scroll:
        # Extra aggressive scrolling for finding much older content
        scroll_patterns = [
            "window.scrollBy(0, 1500 + Math.random() * 500)",   # 1500-2000px
            "window.scrollBy(0, 2000 + Math.random() * 800)",   # 2000-2800px 
            "window.scrollBy(0, 1200 + Math.random() * 600)",   # 1200-1800px
            "window.scrollBy(0, 2500 + Math.random() * 1000)"   # 2500-3500px
        ]
    else:
        # More aggressive scroll amounts for finding older content
        scroll_patterns = [
            "window.scrollBy(0, 800 + Math.random() * 400)",    # 800-1200px
            "window.scrollBy(0, 1000 + Math.random() * 500)",   # 1000-1500px 
            "window.scrollBy(0, 600 + Math.random() * 300)",    # 600-900px
            "window.scrollBy(0, 1200 + Math.random() * 600)"    # 1200-1800px
        ]
    
    pattern = random.choice(scroll_patterns)
    page.evaluate(pattern)
    
    # Sometimes scroll back up slightly (human behavior) - reduced frequency
    if random.random() < 0.05:  # 5% chance (reduced from 10%)
        page.evaluate("window.scrollBy(0, -100 - Math.random() * 150)")
    
    human_delay(1.5, 4.0)

def login_and_save_session(page, username, password):
    """Enhanced login with better anti-detection measures."""
    
    print("🔐 Starting enhanced login process...")
    
    # Advanced stealth setup
    page.route("**/*", lambda route, request: (
        route.continue_() if not any(keyword in request.url.lower() 
            for keyword in ["webdriver", "automation", "headless"]) 
        else route.abort()
    ))
    
    # Go to login page with random delay
    page.goto("https://x.com/login")
    human_delay(2.0, 4.0)

    # Step 1: Enter username with human-like typing
    try:
        page.wait_for_selector('input[name="text"]', timeout=10000)
        username_field = page.locator('input[name="text"]')
        
        # Type with human-like delays
        for char in username:
            username_field.type(char)
            time.sleep(random.uniform(0.05, 0.15))
        
        human_delay(0.5, 1.5)
        page.click('div[data-testid="LoginForm_Login_Button"], div[role="button"]:has-text("Siguiente"), button:has-text("Next")')
        human_delay(2.0, 4.0)
        
    except TimeoutError:
        print("⚠️ Username field not found, may already be logged in")

    # Step 2: Handle unusual activity or confirmation
    try:
        page.wait_for_selector('input[name="text"]', timeout=4000)
        unusual_activity = page.query_selector('div:has-text("unusual activity"), div:has-text("actividad inusual")')
        
        if unusual_activity:
            print("⚠️ Unusual activity detected, entering email/phone...")
            confirmation_field = page.locator('input[name="text"]')
            confirmation_field.fill(EMAIL_OR_PHONE or username)
        else:
            print("🔄 Confirming username...")
            username_field = page.locator('input[name="text"]') 
            username_field.fill(username)
        
        human_delay(1.0, 2.0)
        page.click('div[data-testid="LoginForm_Login_Button"], div[role="button"]:has-text("Siguiente"), button:has-text("Next")')
        human_delay(2.0, 4.0)
        
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
        
        human_delay(0.5, 1.5)
        page.click('div[data-testid="LoginForm_Login_Button"], button:has-text("Iniciar sesión"), button:has-text("Log in")')
        human_delay(3.0, 6.0)
        
    except TimeoutError:
        print("⚠️ Password field not found")

    # Verify login success
    try:
        page.wait_for_url("https://x.com/home", timeout=15000)
        print("✅ Login successful!")
        human_delay(2.0, 4.0)
        return True
    except TimeoutError:
        print("❌ Login verification failed - check for CAPTCHA or 2FA")
        return False

def init_db():
    """Initialize database with enhanced schema."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # The enhanced schema is already created by migrate_tweets_schema.py
    # Just verify it exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tweets'")
    if not c.fetchone():
        print("❌ Enhanced tweets table not found! Run migrate_tweets_schema.py first.")
        raise Exception("Database not properly initialized")
    
    print("✅ Enhanced database schema ready")
    # Ensure scrape_errors table exists for logging errors during scraping
    c.execute("""
    CREATE TABLE IF NOT EXISTS scrape_errors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        tweet_id TEXT,
        error TEXT,
        context TEXT,
        timestamp TEXT
    )
    """)
    conn.commit()
    return conn

def save_enhanced_tweet(conn, tweet_data: Dict):
    """Save tweet with enhanced data structure - simplified for current schema."""
    c = conn.cursor()
    try:
        # Skip invalid or placeholder tweet IDs (some runs produce an 'analytics' placeholder)
        if not tweet_data.get('tweet_id') or str(tweet_data.get('tweet_id')).lower() == 'analytics':
            print(f"  ⚠️ Skipping invalid/sentinel tweet_id: {tweet_data.get('tweet_id')}")
            return False

        # Check if tweet already exists. If it does, update only when new analysis differs
        c.execute("SELECT id, post_type, content, original_author, original_tweet_id FROM tweets WHERE tweet_id = ?", (tweet_data['tweet_id'],))
        existing = c.fetchone()
        if existing:
            existing_id, existing_post_type, existing_content, existing_original_author, existing_original_tweet_id = existing[0], existing[1], existing[2], existing[3], existing[4]

            # Determine whether an update is needed (post_type changed or original info updated or content changed)
            new_post_type = tweet_data.get('post_type', 'original')
            needs_update = False
            if existing_post_type != new_post_type:
                needs_update = True
            elif tweet_data.get('original_author') and tweet_data.get('original_author') != existing_original_author:
                needs_update = True
            elif tweet_data.get('original_tweet_id') and tweet_data.get('original_tweet_id') != existing_original_tweet_id:
                needs_update = True
            elif tweet_data.get('content') and tweet_data.get('content') != existing_content:
                needs_update = True

            if not needs_update:
                print(f"  ⏭️ Tweet {tweet_data['tweet_id']} already exists with same analysis, skipping")
                return False

            # Perform an update to correct analysis/fields
            print(f"  🔁 Tweet {tweet_data['tweet_id']} exists but analysis changed (\"{existing_post_type}\" -> \"{new_post_type}\"). Updating row {existing_id}...")
            c.execute("""
                UPDATE tweets SET
                    content = ?,
                    username = ?,
                    tweet_url = ?,
                    tweet_timestamp = ?,
                    post_type = ?,
                    is_pinned = ?,
                    original_author = ?,
                    original_tweet_id = ?,
                    original_content = ?,
                    reply_to_username = ?,
                    media_links = ?,
                    media_count = ?,
                    media_types = ?,
                    hashtags = ?,
                    mentions = ?,
                    external_links = ?,
                    engagement_likes = ?,
                    engagement_retweets = ?,
                    engagement_replies = ?
                WHERE tweet_id = ?
            """, (
                tweet_data['content'],
                tweet_data['username'],
                tweet_data['tweet_url'],
                tweet_data['tweet_timestamp'],
                tweet_data.get('post_type', 'original'),
                tweet_data.get('is_pinned', 0),
                tweet_data.get('original_author'),
                tweet_data.get('original_tweet_id'),
                tweet_data.get('original_content'),
                tweet_data.get('reply_to_username'),
                tweet_data.get('media_links'),
                tweet_data.get('media_count', 0),
                tweet_data.get('media_types'),
                tweet_data.get('hashtags'),
                tweet_data.get('mentions'),
                tweet_data.get('external_links'),
                tweet_data.get('engagement_likes', 0),
                tweet_data.get('engagement_retweets', 0),
                tweet_data.get('engagement_replies', 0),
                tweet_data['tweet_id']
            ))

            conn.commit()
            print(f"  ✅ Updated tweet: {tweet_data['tweet_id']}")
            return True
        
        # Insert with only the columns that exist in current schema
        c.execute("""
            INSERT INTO tweets (
                tweet_id, content, username, tweet_url, tweet_timestamp,
                post_type, is_pinned, original_author, original_tweet_id, original_content, reply_to_username,
                media_links, media_count, media_types, hashtags, mentions, external_links,
                engagement_likes, engagement_retweets, engagement_replies
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            tweet_data['tweet_id'],
            tweet_data['content'],
            tweet_data['username'],
            tweet_data['tweet_url'],
            tweet_data['tweet_timestamp'],

            # post type fields
            tweet_data.get('post_type', 'original'),
            tweet_data.get('is_pinned', 0),
            tweet_data.get('original_author'),
            tweet_data.get('original_tweet_id'),
            tweet_data.get('original_content'),
            tweet_data.get('reply_to_username'),

            # media and links
            tweet_data.get('media_links'),
            tweet_data.get('media_count', 0),
            tweet_data.get('media_types'),
            tweet_data.get('hashtags'),
            tweet_data.get('mentions'),
            tweet_data.get('external_links'),

            # engagement
            tweet_data.get('engagement_likes', 0),
            tweet_data.get('engagement_retweets', 0),
            tweet_data.get('engagement_replies', 0)
        ))

        conn.commit()
        print(f"  ✅ Saved tweet: {tweet_data['tweet_id']}")
        return True
    except Exception as e:
        print(f"  ❌ Error saving tweet {tweet_data.get('tweet_id', 'unknown')}: {e}")
        return False


def get_last_tweet_timestamp(username: str) -> Optional[str]:
    """Get the timestamp of the most recent tweet for a user."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT tweet_timestamp 
            FROM tweets 
            WHERE username = ? 
            ORDER BY tweet_timestamp DESC 
            LIMIT 1
        """, (username,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return result[0]
        return None
        
    except Exception as e:
        print(f"  ⚠️ Error getting last tweet timestamp for @{username}: {e}")
        return None


def get_oldest_tweet_timestamp(username: str) -> Optional[str]:
    """Get the timestamp of the oldest tweet for a user."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT tweet_timestamp 
            FROM tweets 
            WHERE username = ? AND tweet_timestamp IS NOT NULL
            ORDER BY tweet_timestamp ASC 
            LIMIT 1
        """, (username,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return result[0]
        return None
        
    except Exception as e:
        print(f"  ⚠️ Error getting oldest tweet timestamp for @{username}: {e}")
        return None


def check_if_tweet_exists(username: str, tweet_id: str) -> bool:
    """Check if a tweet already exists in the database."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 1 FROM tweets 
            WHERE username = ? AND tweet_id = ?
        """, (username, tweet_id))
        
        result = cursor.fetchone()
        conn.close()
        
        return result is not None
        
    except Exception as e:
        print(f"  ⚠️ Error checking if tweet exists: {e}")
        return False


def should_skip_existing_tweet(tweet_timestamp: str, oldest_timestamp: Optional[str]) -> bool:
    """Check if we should skip this tweet because it's newer than our oldest tweet (already scraped)."""
    if not oldest_timestamp:
        return False  # No previous tweets, don't skip anything
    
    try:
        # Convert timestamps to comparable format
        tweet_time = datetime.fromisoformat(tweet_timestamp.replace('Z', '+00:00'))
        oldest_time = datetime.fromisoformat(oldest_timestamp.replace('Z', '+00:00'))
        
        # Skip if this tweet is newer than or equal to our oldest scraped tweet
        return tweet_time >= oldest_time
        
    except Exception as e:
        print(f"  ⚠️ Error comparing timestamps: {e}")
        return False  # Don't skip on error

def extract_media_data(article) -> Tuple[List[str], int, List[str]]:
    """Enhanced media extraction - captures media even when not fully displayed."""
    media_links = []
    media_types = []
    
    # Enhanced image extraction - multiple selectors
    image_selectors = [
        'img[src*="pbs.twimg.com/media/"]',
        'img[src*="twimg.com/media/"]',
        'img[data-testid="tweetPhoto"]',
        '[data-testid="tweetPhoto"] img',
        '[role="img"][src*="twimg.com"]'
    ]
    
    for selector in image_selectors:
        for img in article.query_selector_all(selector):
            src = img.get_attribute('src')
            if src and 'twimg.com' in src and src not in media_links:
                media_links.append(src)
                media_types.append("image")
    
    # Enhanced video extraction
    video_selectors = [
        'video',
        '[data-testid="videoComponent"] video',
        '[data-testid="videoPlayer"] video'
    ]
    
    for selector in video_selectors:
        for video in article.query_selector_all(selector):
            # Video poster (thumbnail)
            poster = video.get_attribute('poster')
            if poster and poster not in media_links:
                media_links.append(poster)
                if 'tweet_video_thumb' in poster:
                    media_types.append("gif")
                else:
                    media_types.append("video")
            
            # Video source
            src = video.get_attribute('src')
            if src and src not in media_links:
                media_links.append(src)
                media_types.append("video")
    
    # Extract from data attributes (fallback for lazy loading)
    for elem in article.query_selector_all('[data-src*="twimg.com"], [data-url*="twimg.com"]'):
        data_src = elem.get_attribute('data-src') or elem.get_attribute('data-url')
        if data_src and data_src not in media_links:
            media_links.append(data_src)
            if 'video' in data_src or 'mp4' in data_src:
                media_types.append("video")
            else:
                media_types.append("image")
    
    # Look for background images in CSS
    for elem in article.query_selector_all('[style*="background-image"]'):
        style = elem.get_attribute('style')
        if style and 'twimg.com' in style:
            # Extract URL from background-image: url(...)
            import re
            match = re.search(r'background-image:\s*url\(["\']?(.*?twimg\.com[^"\')\s]*)', style)
            if match:
                bg_url = match.group(1)
                if bg_url not in media_links:
                    media_links.append(bg_url)
                    media_types.append("image")
    
    return media_links, len(media_links), media_types

def extract_engagement_metrics(article) -> Dict[str, int]:
    """Extract engagement numbers (likes, retweets, replies, views)."""
    engagement = {
        'retweets': 0,
        'likes': 0, 
        'replies': 0,
        'views': 0
    }
    
    # Look for engagement buttons and their counts
    engagement_selectors = {
        'replies': ['[data-testid="reply"]', 'svg[data-testid="iconMessageCircle"]'],
        'retweets': ['[data-testid="retweet"]', 'svg[data-testid="iconRetweet"]'], 
        'likes': ['[data-testid="like"]', 'svg[data-testid="iconHeart"]'],
        'views': ['[data-testid="app-text-transition-container"]']  # Views counter
    }
    
    for metric, selectors in engagement_selectors.items():
        for selector in selectors:
            elements = article.query_selector_all(selector)
            for element in elements:
                # Look for parent or sibling with count
                parent = element.query_selector('..')
                if parent:
                    text = parent.inner_text().strip()
                    # Extract numbers from text (handle K, M suffixes)
                    numbers = re.findall(r'(\d+(?:\.\d+)?[KM]?)', text)
                    if numbers:
                        count_str = numbers[0]
                        try:
                            if 'K' in count_str:
                                engagement[metric] = int(float(count_str.replace('K', '')) * 1000)
                            elif 'M' in count_str:
                                engagement[metric] = int(float(count_str.replace('M', '')) * 1000000)
                            else:
                                engagement[metric] = int(count_str)
                            break
                        except ValueError:
                            continue
    
    return engagement

def extract_full_tweet_content(article) -> str:
    """
    Extract full tweet content, handling truncation by attempting to expand first.

    Twitter/X sometimes truncates long tweets in the feed and shows "... Show more" or similar.
    This function tries to expand truncated content before extracting.
    """
    try:
        # Avoid clicking the tweet text itself: that often navigates into the
        # permalink page and breaks the timeline scraping loop. Only click
        # dedicated "show more" controls that are less likely to navigate.
        expand_selectors = [
            '[role="button"][aria-label*="Show more"]',
            '[role="button"][aria-label*="show more"]',
            '[data-testid*="showMore"]',
            '[aria-label*="Show more"]',
            'div[role="button"]:has-text("Show more")',
            'div[role="button"]:has-text("Mostrar más")'
        ]

        # First, if we somehow ended up inside a permalink view, go back.
        try:
            current_url = article.evaluate("() => window.location.href")
            if '/status/' in (current_url or ''):
                print("    ⚠️ Detected permalink view during extraction, going back to timeline")
                article.evaluate("() => window.history.back()")
                article.wait_for_timeout(800)
        except Exception:
            # Best-effort only; continue if it fails
            pass

        # Try expand buttons but skip anchors that point to /status/ (they navigate)
        for selector in expand_selectors:
            try:
                expand_button = article.query_selector(selector)
                if not expand_button:
                    continue

                # If the element is (or contains) an anchor to a status permalink, skip clicking
                try:
                    href = expand_button.get_attribute('href')
                    if href and '/status/' in href:
                        print("    ⚠️ Skipping expand element because it links to permalink")
                        continue
                except Exception:
                    pass

                print("    🔓 Found expand button, clicking to reveal full content (safe click)")
                try:
                    expand_button.click()
                    article.wait_for_timeout(500)
                except Exception:
                    # If click fails, continue - we'll still try to read text below
                    pass
                break
            except Exception:
                continue

        # Read tweet text without clicking the tweet element (avoids navigation)
        text_elem = article.query_selector('[data-testid="tweetText"]')
        if text_elem:
            try:
                current_text = text_elem.inner_text().strip()
                return current_text
            except Exception:
                return ""

        return ""

    except Exception as e:
        print(f"    ⚠️ Error in content expansion: {e}")
        # Fallback to basic extraction without clicks
        try:
            text_elem = article.query_selector('[data-testid="tweetText"]')
            return text_elem.inner_text().strip() if text_elem else ""
        except Exception:
            return ""

def extract_content_elements(article) -> Dict[str, any]:
    """Extract hashtags, mentions, and external links."""
    
    hashtags = []
    mentions = []
    external_links = []
    
    # Extract hashtags
    for hashtag_link in article.query_selector_all('a[href*="/hashtag/"]'):
        hashtag = hashtag_link.inner_text().strip()
        if hashtag and hashtag not in hashtags:
            hashtags.append(hashtag)
    
    # Extract mentions
    for mention_link in article.query_selector_all('a[href^="/"]:not([href*="/hashtag/"]):not([href*="/status/"])'):
        href = mention_link.get_attribute('href')
        if href and href.count('/') == 1:  # Just /username format
            username = href.replace('/', '@')
            if username not in mentions:
                mentions.append(username)
    
    # Extract external links
    for link in article.query_selector_all('a[href^="http"]'):
        href = link.get_attribute('href') 
        if href and 'twitter.com' not in href and 'x.com' not in href:
            if href not in external_links:
                external_links.append(href)
    
    return {
        'hashtags': json.dumps(hashtags) if hashtags else None,
        'mentions': json.dumps(mentions) if mentions else None,
        'external_links': json.dumps(external_links) if external_links else None,
        'has_external_link': 1 if external_links else 0
    }

def analyze_post_type(article, target_username: str) -> Dict[str, any]:
    """
    Sophisticated post type analysis for X/Twitter content.
    
    Detects:
    - original: Original posts by the user
    - repost_own: User reposting their own content  
    - repost_other: User reposting someone else's content
    - repost_reply: Repost of a reply (e.g. user reposting a reply to their tweet)
    - quote: removed. Quote-like tweets are treated as 'original' with reply/quoted metadata preserved
    - thread: Part of a thread (connected tweets)
    - pinned: Pinned posts (should be skipped in regular scraping)
    """
    
    post_analysis = {
        'post_type': 'original',
        'is_pinned': 0,
        'original_author': None,
        'original_tweet_id': None, 
        'original_content': None,
        'reply_to_username': None,
        'reply_to_tweet_id': None,
        'reply_to_content': None,
        'thread_position': 0,
        'thread_root_id': None,
        'should_skip': False
    }
    
    # Check for pinned post
    print("    [debug] analyze_post_type: starting checks")
    pinned_indicator = article.query_selector('[data-testid="socialContext"]:has-text("Pinned"), [aria-label*="Pinned"]')
    if pinned_indicator:
        post_analysis['is_pinned'] = 1
        post_analysis['should_skip'] = True  # Skip pinned posts
        print("    📌 Pinned post detected - will skip")
        return post_analysis
    
    # Historically the code treated quoted content as a separate post_type.
    # To simplify, we now treat quoted tweets as 'original' posts while
    # preserving the quoted/replied metadata (reply_to_username, reply_to_tweet_id,
    # reply_to_content). This keeps the database post_type space compact.
    main_text = article.query_selector('[data-testid="tweetText"]')
    quoted_content = article.query_selector('[data-testid="tweetText"] ~ div [role="article"], .css-1dbjc4n [role="article"]')
    print("    [debug] checking for quote tweet (main_text && quoted_content)")
    if main_text and quoted_content:
        # Preserve quoted metadata but keep post_type as 'original'
        try:
            quoted_author_link = None
            for a in quoted_content.query_selector_all('a[href^="/"]'):
                href = a.get_attribute('href')
                if href and href.count('/') == 1:
                    quoted_author_link = a
                    break
            if quoted_author_link:
                quoted_href = quoted_author_link.get_attribute('href')
                post_analysis['reply_to_username'] = quoted_href.replace('/', '') if quoted_href else None
        except Exception:
            post_analysis['reply_to_username'] = None

        try:
            quoted_tweet_link = quoted_content.query_selector('a[href*="/status/"]')
            if quoted_tweet_link:
                quoted_tweet_href = quoted_tweet_link.get_attribute('href')
                post_analysis['reply_to_tweet_id'] = quoted_tweet_href.split('/')[-1] if quoted_tweet_href else None
        except Exception:
            pass

        try:
            quoted_text_elem = quoted_content.query_selector('[data-testid="tweetText"]')
            if quoted_text_elem:
                post_analysis['reply_to_content'] = quoted_text_elem.inner_text().strip()
        except Exception:
            pass

        print(f"    🗨️ Quote-like tweet detected (mapped to original): @{post_analysis.get('reply_to_username')}")
        return post_analysis

    # Check for repost indicators (handle multiple languages and selector variations)
    repost_indicators = [
        '[data-testid="socialContext"]:has-text("Reposted"), [data-testid="socialContext"]:has-text("reposted"), [data-testid="socialContext"]:has-text("Retweeted"), [data-testid="socialContext"]:has-text("Retweeted" )',
        'svg[data-testid="iconRetweet"]',
        '[aria-label*="Repost"], [aria-label*="repost"], [aria-label*="Retweet"], [aria-label*="retweet"]',
        'div:has-text("Retweeted"), div:has-text("Reposted"), div:has-text("Reposteado"), div:has-text("Retuiteado")'
    ]

    repost_element = None
    for selector in repost_indicators:
        try:
            repost_element = article.query_selector(selector)
        except Exception:
            repost_element = article.query_selector(selector.split(',')[0]) if ',' in selector else None
        if repost_element:
            # Verify that the found element actually indicates a repost by checking
            # its visible text or aria-label for explicit keywords. This avoids
            # matching the generic retweet icon/button which exists on every tweet.
            try:
                elem_text = (repost_element.inner_text() or "").lower()
            except Exception:
                elem_text = ""
            try:
                elem_aria = (repost_element.get_attribute('aria-label') or "").lower()
            except Exception:
                elem_aria = ""

            keywords = ["retweet", "retweeted", "reposted", "reposteado", "retuiteado"]
            is_repost_text = any(k in elem_text for k in keywords) or any(k in elem_aria for k in keywords)
            if not is_repost_text:
                # If the repost element doesn't contain explicit text/aria, check
                # whether it contains a link to the target username (e.g. "Vito
                # Quiles reposted"). If so, treat as a repost indicator.
                try:
                    link_to_target = repost_element.query_selector(f'a[href="/{target_username}"]')
                except Exception:
                    link_to_target = None
                if not link_to_target:
                    # Not an explicit repost indicator; ignore and continue searching
                    repost_element = None
                    continue
            # Otherwise accept this element as a valid repost indicator
            break

    if repost_element:
        # This is a repost - determine if it's own content or other's content
        print(f"    [debug] repost_element found using selector fallback")

        # Try to locate the nested quoted/original tweet inside the repost block using safer heuristics
        quoted_tweet = None
        # QUICK HEURISTIC: if the repost element itself contains a link to the
        # target username (e.g. "Vito Quiles reposted") then treat this as a
        # self-repost. This helps when the embedded quoted tweet is not easily
        # discoverable via the DOM traversal below but the social context clearly
        # indicates the user reposted their own content.
        try:
            try:
                reposter_link = repost_element.query_selector(f'a[href="/{target_username}"]')
            except Exception:
                reposter_link = None
            if reposter_link:
                # Try to pull an original status link if present anywhere in the article
                try:
                    original_link_any = article.query_selector('a[href*="/status/"]')
                    if original_link_any:
                        orig_href = original_link_any.get_attribute('href')
                        post_analysis['original_tweet_id'] = orig_href.split('/')[-1] if orig_href else None
                except Exception:
                    pass

                post_analysis['post_type'] = 'repost_own'
                post_analysis['original_author'] = target_username
                post_analysis['should_skip'] = True
                print(f"    🔄 Heuristic: own repost detected via socialContext link (will skip): @{target_username}")
                return post_analysis
        except Exception:
            # Best-effort heuristic; continue to more robust detection below
            pass
        try:
            # First try: look for an article inside the repost block
            quoted_tweet = article.query_selector('article [data-testid="tweetText"], [data-testid="tweet"] article')
        except Exception:
            quoted_tweet = None

        # If not found, fall back to finding an anchor to /status/ and using its closest article
        if not quoted_tweet:
            try:
                original_tweet_link = article.query_selector('a[href*="/status/"]')
                if original_tweet_link:
                    parent = original_tweet_link.evaluate_handle('el => el.closest("article")')
                    if parent:
                        quoted_tweet = parent.as_element()
            except Exception:
                pass

        if quoted_tweet:
            print("    [debug] quoted_tweet element located")

            # First, ensure the quoted block actually contains a link to a status
            # If there is no /status/ link the block is unlikely to be an embedded tweet
            # and we should not classify it as a repost (this avoids false positives).
            try:
                original_tweet_link = quoted_tweet.query_selector('a[href*="/status/"]')
            except Exception:
                original_tweet_link = None

            if not original_tweet_link:
                # No explicit original tweet link found inside the quoted block; treat as not a repost
                print("    ⚠️ Quoted block has no /status/ link — skipping repost classification")
            else:
                # Extract original tweet ID
                try:
                    original_href = original_tweet_link.get_attribute('href')
                    original_tweet_id = original_href.split('/')[-1] if original_href else None
                    post_analysis['original_tweet_id'] = original_tweet_id
                except Exception:
                    original_tweet_id = None

                # Extract original author info robustly: collect candidate anchors like "/username" inside the quoted tweet
                original_author = None
                try:
                    candidates = []
                    for a in quoted_tweet.query_selector_all('a[href^="/"]'):
                        href = a.get_attribute('href')
                        if not href:
                            continue
                        # skip status links and media links
                        if '/status/' in href or '/photo' in href or '/video' in href:
                            continue
                        # keep only single-segment user links like '/username'
                        if href.count('/') == 1:
                            candidate = href.replace('/', '')
                            if candidate and candidate not in candidates:
                                candidates.append(candidate)

                    # Prefer a non-target candidate if there's more than one; otherwise use the sole candidate
                    if candidates:
                        if len(candidates) == 1:
                            original_author = candidates[0]
                        else:
                            # try to pick the first candidate that isn't the target_username
                            picked = None
                            for c in candidates:
                                if c != target_username:
                                    picked = c
                                    break
                            original_author = picked or candidates[0]
                except Exception:
                    original_author = None

                # Only mark as a repost when we have an explicit original tweet id AND a resolved original author
                if original_author and post_analysis.get('original_tweet_id'):
                    if original_author == target_username:
                        post_analysis['post_type'] = 'repost_own'
                        # If the user is reposting their own content, we usually
                        # don't want to store a duplicate of the original tweet.
                        # Mark as should_skip so the fetch loop can avoid saving
                        # or re-collecting this kind of self-retweet.
                        post_analysis['should_skip'] = True
                        print(f"    🔄 Own repost detected (will skip): @{original_author}")
                    else:
                        post_analysis['post_type'] = 'repost_other'
                        print(f"    🔄 Other's repost detected: @{original_author}")
                    post_analysis['original_author'] = original_author

                    # Extract original content where possible
                    try:
                        original_text_elem = quoted_tweet.query_selector('[data-testid="tweetText"]')
                        if original_text_elem:
                            post_analysis['original_content'] = original_text_elem.inner_text().strip()
                    except Exception:
                        pass

                    # When we detected a repost and extracted original info, return
                    return post_analysis

    # Check for reply indicators
    reply_context = article.query_selector('[data-testid="socialContext"]:has-text("Replying to"), [data-testid="tweetText"]:has-text("Replying to")')
    print("    [debug] checking reply indicators")
    if reply_context:
        # Rename conceptual 'reply' to 'repost_reply' to represent a repost
        # of a reply or a reply context that we might want to treat specially.
        post_analysis['post_type'] = 'repost_reply'

        # Extract who is being replied to
        reply_mention = reply_context.query_selector('a[href^="/"]')
        if reply_mention:
            reply_href = reply_mention.get_attribute('href')
            post_analysis['reply_to_username'] = reply_href.replace('/', '') if reply_href else None
            print(f"    💬 Reply detected to: @{post_analysis['reply_to_username']}")

        return post_analysis

    # Check for quote tweet (has both own text and quoted content)
    main_text = article.query_selector('[data-testid="tweetText"]')
    quoted_content = article.query_selector('[data-testid="tweetText"] ~ div [role="article"], .css-1dbjc4n [role="article"]')
    print("    [debug] checking for quote tweet (main_text && quoted_content)")
    if main_text and quoted_content:
        # Treat quote-like tweets as 'original' while preserving quoted metadata
        # (we no longer use a separate 'quote' post_type to keep the taxonomy compact)
        post_analysis['post_type'] = 'original'
        
        # Extract quoted tweet info
        quoted_author_link = quoted_content.query_selector('a[href^="/"][role="link"]')
        if quoted_author_link:
            quoted_href = quoted_author_link.get_attribute('href')
            post_analysis['reply_to_username'] = quoted_href.replace('/', '') if quoted_href else None
        
        quoted_tweet_link = quoted_content.query_selector('a[href*="/status/"]')
        if quoted_tweet_link:
            quoted_tweet_href = quoted_tweet_link.get_attribute('href')
            post_analysis['reply_to_tweet_id'] = quoted_tweet_href.split('/')[-1] if quoted_tweet_href else None
        
        quoted_text_elem = quoted_content.query_selector('[data-testid="tweetText"]')
        if quoted_text_elem:
            post_analysis['reply_to_content'] = quoted_text_elem.inner_text().strip()
        
        print(f"    🗨️ Quote-like content detected (mapped to original): @{post_analysis['reply_to_username']}")
        return post_analysis
    
    # Check for thread indicators (consecutive tweets by same user)
    thread_indicator = article.query_selector('[data-testid="socialContext"]:has-text("Show this thread")')
    print("    [debug] checking thread indicator")
    if thread_indicator:
        post_analysis['post_type'] = 'thread'
        print("    🧵 Thread detected")
        # Thread analysis would need more sophisticated logic to determine position and root
    
    # Default: original post
    print(f"    ✍️ Original post detected")
    return post_analysis

def extract_profile_picture(page, username: str) -> Optional[str]:
    """Extract profile picture URL from user's Twitter profile page."""
    try:
        # Look for profile image in various possible selectors
        profile_img_selectors = [
            '[data-testid="UserAvatar-Container-unknown"] img',
            '[data-testid="UserAvatar-Container"] img', 
            '[aria-label*="profile photo"] img',
            '[data-testid="UserProfileHeader_Items"] img',
            'img[src*="profile_images"]',
            'a[href*="/photo"] img'
        ]
        
        profile_pic_url = None
        
        for selector in profile_img_selectors:
            try:
                img_element = page.query_selector(selector)
                if img_element:
                    src = img_element.get_attribute('src')
                    if src and 'profile_images' in src:
                        # Get the full-size version (remove size parameters)
                        if '_normal.' in src:
                            profile_pic_url = src.replace('_normal.', '_400x400.')
                        elif '_bigger.' in src:
                            profile_pic_url = src.replace('_bigger.', '_400x400.')
                        else:
                            profile_pic_url = src
                        print(f"  🖼️  Profile picture found: {profile_pic_url[:80]}...")
                        break
            except Exception:
                continue
        
        if not profile_pic_url:
            print(f"  ⚠️  No profile picture found for @{username}")
            
        return profile_pic_url
        
    except Exception as e:
        print(f"  ❌ Error extracting profile picture for @{username}: {e}")
        return None

def save_account_profile_info(conn, username: str, profile_pic_url: str = None):
    """Save or update account profile information."""
    if not profile_pic_url:
        return
    
    cursor = conn.cursor()
    try:
        # Insert or update account profile information
        cursor.execute("""
            INSERT INTO accounts (username, profile_pic_url, profile_pic_updated, last_scraped)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(username) DO UPDATE SET
                profile_pic_url = excluded.profile_pic_url,
                profile_pic_updated = excluded.profile_pic_updated,
                last_scraped = excluded.last_scraped
        """, (
            username, 
            profile_pic_url, 
            datetime.now().isoformat(),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        print(f"  💾 Updated profile info for @{username}")
        
    except Exception as e:
        print(f"  ❌ Error saving profile info for @{username}: {e}")

def fetch_enhanced_tweets(page, username: str, max_tweets: int = 30, resume_from_last: bool = True) -> List[Dict]:
    """
    Enhanced tweet fetching with comprehensive post type detection.
    
    If resume_from_last is True, will skip existing tweets and fetch older ones.
    """
    print(f"\n🎯 Starting enhanced tweet collection for @{username}")
    
    # Get oldest tweet timestamp for continuing from where we left off
    oldest_timestamp = None
    if resume_from_last:
        oldest_timestamp = get_oldest_tweet_timestamp(username)
        if oldest_timestamp:
            print(f"📅 Oldest scraped tweet: {oldest_timestamp} - will fetch older tweets")
        else:
            print("🆕 No previous tweets found - scraping from beginning")
    
    url = f"https://x.com/{username}"
    page.goto(url)
    
    # Wait for tweets to load
    try:
        page.wait_for_selector('[data-testid="tweetText"], [data-testid="tweet"]', timeout=15000)
    except TimeoutError:
        print(f"❌ No tweets found for @{username} or page failed to load")
        return []

    human_delay(2.0, 4.0)
    
    # Extract profile picture before starting tweet collection
    print(f"🖼️  Extracting profile picture for @{username}...")
    profile_pic_url = extract_profile_picture(page, username)
    
    collected_tweets = []
    seen_tweet_ids = set()
    scroll_attempts = 0
    max_scroll_attempts = 200  # Much higher limit for finding older tweets
    last_height = 0
    consecutive_old_tweets = 0  # Track consecutive existing/covered tweets
    max_consecutive_old = 100   # Much higher threshold to keep trying
    found_older_tweets_yet = False  # Track if we've found any older tweets
    
    print(f"🔍 Collecting up to {max_tweets} tweets...")
    
    def event_scroll_cycle(page, iteration):
        # Alternate between deep jumps, keyboard-like PageDown, and micro scrolls
        try:
            if iteration % 7 == 0:
                # occasional deep jump
                page.evaluate("window.scrollBy(0, 1600 + Math.random() * 800)")
            elif iteration % 5 == 0:
                # subtle keyboard-like page down
                page.keyboard.press('PageDown')
            else:
                # micro wheel scroll simulation
                page.evaluate("window.scrollBy(0, 600 + Math.random() * 400)")
        except Exception:
            # Fall back to JS scroll if keyboard not permitted
            try:
                page.evaluate("window.scrollBy(0, 800 + Math.random() * 400)")
            except Exception:
                pass
        human_delay(0.8, 2.2)

    iteration = 0

    while len(collected_tweets) < max_tweets and scroll_attempts < max_scroll_attempts:
        # Find all tweet articles on current page
        articles = page.query_selector_all('article[data-testid="tweet"], [data-testid="tweet"]')
        
        print(f"  📄 Found {len(articles)} tweet elements on page")
        
        for i, article in enumerate(articles):
            if len(collected_tweets) >= max_tweets:
                break
                
            try:
                # Extract basic tweet info
                tweet_link = article.query_selector('a[href*="/status/"]')
                if not tweet_link:
                    continue
                    
                href = tweet_link.get_attribute('href')
                tweet_id = href.split('/')[-1] if href else None
                
                if not tweet_id or tweet_id in seen_tweet_ids:
                    continue
                
                # Analyze post type first 
                post_analysis = analyze_post_type(article, username)
                
                # Skip pinned posts as requested
                if post_analysis['should_skip']:
                    continue
                
                tweet_url = f"https://x.com{href}"
                
                # Extract tweet content with expansion handling
                content = extract_full_tweet_content(article)
                
                # Skip if no content (media-only posts still have some text usually)
                if not content:
                    print(f"    ⏭️ Skipping content-less tweet: {tweet_id}")
                    continue
                
                # Extract media information  
                media_links, media_count, media_types = extract_media_data(article)
                
                # Extract engagement metrics
                engagement = extract_engagement_metrics(article)
                
                # Extract content elements (hashtags, mentions, links)
                content_elements = extract_content_elements(article)
                
                # Extract timestamp if available
                time_elem = article.query_selector('time')
                tweet_timestamp = None
                if time_elem:
                    tweet_timestamp = time_elem.get_attribute('datetime')
                
                # Check if this tweet already exists (skip duplicates) but allow updates when analysis/content changed
                if resume_from_last and check_if_tweet_exists(username, tweet_id):
                    try:
                        conn_check = sqlite3.connect(DB_PATH, timeout=10.0)
                        cur_check = conn_check.cursor()
                        cur_check.execute("SELECT post_type, content, original_author, original_tweet_id FROM tweets WHERE tweet_id = ?", (tweet_id,))
                        db_row = cur_check.fetchone()
                        conn_check.close()
                    except Exception:
                        db_row = None

                    needs_update = False
                    if db_row:
                        db_post_type, db_content, db_original_author, db_original_tweet_id = db_row
                        # Compare analysis and some fields to see if we should update
                        if db_post_type != post_analysis.get('post_type'):
                            needs_update = True
                        elif content and db_content and content != db_content:
                            needs_update = True
                        elif tweet_data := None:  # placeholder to keep pylint happy
                            pass

                    if not needs_update:
                        print(f"    ⏭️ Skipping existing tweet ({tweet_id})")
                        consecutive_old_tweets += 1
                        # Be very patient - only stop if we've tried a LOT and found no older tweets
                        stop_threshold = max_consecutive_old if found_older_tweets_yet else max_consecutive_old * 3
                        if consecutive_old_tweets >= stop_threshold:
                            reason = "consecutive existing tweets"
                            print(f"    🛑 Seen {consecutive_old_tweets} {reason} - stopping")
                            return collected_tweets
                        continue

                    # We need to update existing row: build tweet_data and append so main will call save_enhanced_tweet which will perform the update
                    tweet_data = {
                        'tweet_id': tweet_id,
                        'tweet_url': tweet_url,
                        'username': username,
                        'content': content,
                        'tweet_timestamp': tweet_timestamp,
                        'profile_pic_url': profile_pic_url,
                        **post_analysis,
                        'media_links': ','.join(media_links) if media_links else None,
                        'media_count': media_count,
                        'media_types': json.dumps(media_types) if media_types else None,
                        **content_elements,
                        'engagement_retweets': engagement['retweets'],
                        'engagement_likes': engagement['likes'],
                        'engagement_replies': engagement['replies'],
                        'engagement_views': engagement['views'],
                        'is_repost': 1 if 'repost' in post_analysis['post_type'] else 0,
                        'is_comment': 1 if post_analysis['post_type'] == 'repost_reply' else 0,
                        'parent_tweet_id': post_analysis.get('reply_to_tweet_id') or post_analysis.get('original_tweet_id')
                    }

                    collected_tweets.append(tweet_data)
                    seen_tweet_ids.add(tweet_id)
                    print(f"  🔁 Queued existing tweet for update: {tweet_id} (will update DB row)")
                    continue
                
                # Check if we should skip this tweet because it's still in our scraped range
                if resume_from_last and tweet_timestamp and should_skip_existing_tweet(tweet_timestamp, oldest_timestamp):
                    print(f"    ⏭️ Skipping already covered tweet ({tweet_timestamp})")
                    consecutive_old_tweets += 1
                    # Be very patient - only stop if we've tried a LOT and found no older tweets
                    stop_threshold = max_consecutive_old if found_older_tweets_yet else max_consecutive_old * 3
                    if consecutive_old_tweets >= stop_threshold:
                        reason = "consecutive covered tweets"
                        print(f"    🛑 Seen {consecutive_old_tweets} {reason} - stopping")
                        return collected_tweets
                    continue
                
                # Reset counter and mark that we found a collectible tweet
                consecutive_old_tweets = 0
                if not found_older_tweets_yet:
                    found_older_tweets_yet = True
                    print(f"    🎉 Found first older tweet! Continuing to collect...")
                
                # Build comprehensive tweet data
                tweet_data = {
                    'tweet_id': tweet_id,
                    'tweet_url': tweet_url, 
                    'username': username,
                    'content': content,
                    'tweet_timestamp': tweet_timestamp,
                    'profile_pic_url': profile_pic_url,  # Add profile picture URL
                    
                    # Post type analysis
                    **post_analysis,
                    
                    # Media data
                    'media_links': ','.join(media_links) if media_links else None,
                    'media_count': media_count,
                    'media_types': json.dumps(media_types) if media_types else None,
                    
                    # Content elements
                    **content_elements,
                    
                    # Engagement metrics
                    'engagement_retweets': engagement['retweets'],
                    'engagement_likes': engagement['likes'], 
                    'engagement_replies': engagement['replies'],
                    'engagement_views': engagement['views'],
                    
                    # Legacy compatibility
                    'is_repost': 1 if 'repost' in post_analysis['post_type'] else 0,
                    'is_comment': 1 if post_analysis['post_type'] == 'repost_reply' else 0,
                    'parent_tweet_id': post_analysis.get('reply_to_tweet_id') or post_analysis.get('original_tweet_id')
                }
                
                collected_tweets.append(tweet_data)
                seen_tweet_ids.add(tweet_id)
                
                print(f"  ✅ [{len(collected_tweets):2d}/{max_tweets}] {post_analysis['post_type']}: {tweet_id}")
                
            except Exception as e:
                # Log processing error to DB for later inspection
                try:
                    conn_err = sqlite3.connect(DB_PATH)
                    cur_err = conn_err.cursor()
                    cur_err.execute("INSERT INTO scrape_errors (username, tweet_id, error, context, timestamp) VALUES (?, ?, ?, ?, ?)", (
                        username,
                        tweet_id if 'tweet_id' in locals() and tweet_id else None,
                        str(e),
                        'processing_article',
                        datetime.now().isoformat()
                    ))
                    conn_err.commit()
                    conn_err.close()
                except Exception:
                    pass
                print(f"    ❌ Error processing tweet {i+1}: {e}")
                continue
        
        # Human-like scrolling with event-driven cycles for finding older tweets
        if len(collected_tweets) < max_tweets:
            print(f"  📜 Scrolling for more tweets... ({len(collected_tweets)}/{max_tweets})")
            iteration += 1
            try:
                event_scroll_cycle(page, iteration)
            except Exception:
                # Fallback
                random_scroll_pattern(page, deep_scroll=False)

            new_height = page.evaluate("document.body.scrollHeight")
            if new_height == last_height:
                scroll_attempts += 1
                print(f"    ⏳ No new content, attempt {scroll_attempts}/{max_scroll_attempts}")
            else:
                scroll_attempts = 0  # Reset if we got new content
            last_height = new_height
    
    print(f"\n📊 Collection complete: {len(collected_tweets)} tweets from @{username}")
    
    # Print summary by post type
    post_type_counts = {}
    for tweet in collected_tweets:
        post_type = tweet['post_type']
        post_type_counts[post_type] = post_type_counts.get(post_type, 0) + 1
    
    print("📈 Post type breakdown:")
    for post_type, count in sorted(post_type_counts.items()):
        print(f"    {post_type}: {count}")
    
    return collected_tweets

def main():
    start_time = time.time()  # Start timing
    
    # Diagnostic: print raw argv to help debug invocations that unexpectedly
    # fall back to DEFAULT_HANDLES (e.g., when the runner didn't forward args).
    print(f"🐞 raw argv: {sys.argv}")

    parser = argparse.ArgumentParser(description="Fetch tweets from a given X (Twitter) user.")
    # Support both positional username and an explicit flag (--user / -u). Flag takes precedence.
    parser.add_argument("username", nargs='?', help="Optional positional username to fetch tweets from (without @). If omitted, default targets list will be used.")
    parser.add_argument("--user", "-u", dest="user", help="Optional single username to fetch tweets from (with or without leading @). Overrides positional username.")
    parser.add_argument("--max", type=int, default=100, help="Maximum number of tweets to fetch per user (default: 100)")
    parser.add_argument("--handles-file", help="Path to a newline-separated file with target handles (overrides defaults)")
    parser.add_argument("--no-resume", action='store_true', help="Do not resume from previous scrape; fetch recent tweets instead of older ones")
    args = parser.parse_args()

    # Resolve effective target username robustly. --user takes precedence, then positional.
    effective_user = None
    if args.user:
        effective_user = args.user.strip()
    elif args.username:
        effective_user = args.username.strip()

    if effective_user:
        # Normalize leading @ if present
        effective_user = effective_user.lstrip('@').strip()

    max_tweets = args.max

    # Diagnostic: show parsed args for debugging
    print(f"🐞 parsed args: user={args.user} username={args.username} max={max_tweets} handles_file={args.handles_file}")

    # Build handles list. If an explicit user was provided, enforce single-target.
    if effective_user:
        handles = [effective_user]
        if args.handles_file:
            print("⚠️  Ignoring --handles-file because a single username was requested via --user/positional argument")
    else:
        # No single user specified: use handles file if present, else defaults
        if args.handles_file and os.path.exists(args.handles_file):
            with open(args.handles_file, 'r') as f:
                handles = [l.strip() for l in f if l.strip()]
        else:
            handles = DEFAULT_HANDLES.copy()

    print(f"📣 Targets resolved (final): {handles}")
    print(f"📣 max_tweets set to: {max_tweets}")

    resume_from_last_flag = not args.no_resume

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=50)
        
        # Select random user agent for this session
        selected_user_agent = random.choice(USER_AGENTS)
        
        # Check if we have a saved session to reuse
        session_file = "x_session.json"
        context_kwargs = {
            "user_agent": selected_user_agent,
            "viewport": {"width": 1280, "height": 720},
            "locale": "en-US",
            "timezone_id": "America/New_York",
            "color_scheme": "light",
            "java_script_enabled": True,
        }
        
        # Load existing session if available
        if os.path.exists(session_file):
            print("🔄 Loading existing session from x_session.json")
            context_kwargs["storage_state"] = session_file
        else:
            print("🆕 No existing session found, will need to login")
        
        context = browser.new_context(**context_kwargs)

        page = context.new_page()
        # Stealth: Remove webdriver flag and spoof more properties
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            window.navigator.chrome = { runtime: {} };
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
        """)

        # Only login if we don't have a valid session
        if not os.path.exists(session_file):
            print("🔐 No session found, logging in...")
            login_and_save_session(page, USERNAME, PASSWORD)
            context.storage_state(path=session_file)
            print("💾 Session saved to x_session.json")
        else:
            # Test if the loaded session is still valid
            print("🧪 Testing existing session...")
            page.goto("https://x.com/home")
            try:
                # Wait a bit and check if we're actually logged in
                page.wait_for_timeout(3000)
                if page.url.startswith("https://x.com/home") or "home" in page.url:
                    print("✅ Existing session is valid!")
                else:
                    print("❌ Session expired, logging in again...")
                    login_and_save_session(page, USERNAME, PASSWORD)
                    context.storage_state(path=session_file)
                    print("💾 New session saved to x_session.json")
            except:
                print("❌ Session test failed, logging in again...")
                login_and_save_session(page, USERNAME, PASSWORD)
                context.storage_state(path=session_file)
                print("💾 New session saved to x_session.json")

        conn = init_db()
        # Fetch tweets for each handle in a single browser session
        total = 0
        for handle in handles:
            print(f"\nFetching up to {max_tweets} tweets for @{handle}...")
            # Add retries with exponential backoff for each handle
            max_retries = 5
            attempt = 0
            tweets = []
            while attempt <= max_retries:
                try:
                    tweets = fetch_enhanced_tweets(page, handle, max_tweets=max_tweets, resume_from_last=resume_from_last_flag)
                    break
                except Exception as e:
                    attempt += 1
                    backoff = min(60, (2 ** attempt) + random.random() * 5)
                    print(f"  ⚠️ Fetch attempt {attempt} failed for @{handle}: {e} - retrying in {backoff:.1f}s")
                    time.sleep(backoff)
            if not tweets:
                print(f"  ❌ Failed to fetch tweets for @{handle} after {max_retries} retries")
                # Log failure
                try:
                    conn_err = sqlite3.connect(DB_PATH)
                    cur_err = conn_err.cursor()
                    cur_err.execute("INSERT INTO scrape_errors (username, tweet_id, error, context, timestamp) VALUES (?, ?, ?, ?, ?)", (
                        handle,
                        None,
                        'max_retries_exceeded',
                        'fetch_enhanced_tweets',
                        datetime.now().isoformat()
                    ))
                    conn_err.commit()
                    conn_err.close()
                except Exception:
                    pass
            
            # Save each tweet with comprehensive data
            saved_count_for_handle = 0
            for i, tweet in enumerate(tweets, 1):
                try:
                    saved = save_enhanced_tweet(conn, tweet)
                    if saved:
                        saved_count_for_handle += 1
                        print(f"  💾 Saved tweet {saved_count_for_handle}/{len(tweets)}: {tweet['tweet_id']}")
                    else:
                        print(f"  ⏭️ Not saved tweet (skipped or duplicate): {tweet.get('tweet_id', 'unknown')}")
                except Exception as e:
                    print(f"  ❌ Error saving tweet {tweet.get('tweet_id', 'unknown')}: {e}")
            
            # Save profile information if tweets were collected successfully
            if tweets and 'profile_pic_url' in tweets[0]:
                profile_pic_url = tweets[0]['profile_pic_url']
                save_account_profile_info(conn, handle, profile_pic_url)
            
            total += saved_count_for_handle
        
        # Calculate and display execution time
        end_time = time.time()
        execution_time = end_time - start_time
        minutes = int(execution_time // 60)
        seconds = int(execution_time % 60)
        
        print(f"\n⏱️  Execution completed in: {minutes}m {seconds}s")
        print(f"📊 Total tweets fetched and saved: {total}")
        print(f"🎯 Accounts processed: {len(handles)}")
        print(f"📈 Average tweets per account: {total/len(handles):.1f}")
        
        conn.close()
        browser.close()

def run_in_background(username: str, max_tweets: int):
    """Run the fetch script in the background using os.fork (Unix-only)."""
    pid = os.fork()
    if pid > 0:
        # Parent: return child PID
        print(f"Started background fetch (pid={pid}) for @{username}")
        return pid
    else:
        # Child: run main-like logic for single user then exit
        try:
            # Minimal context startup for background run
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                selected_user_agent = random.choice(USER_AGENTS)
                context_kwargs = {
                    "user_agent": selected_user_agent,
                    "viewport": {"width": 1280, "height": 720},
                    "locale": "en-US",
                    "timezone_id": "America/New_York",
                    "color_scheme": "light",
                    "java_script_enabled": True,
                }
                session_file = "x_session.json"
                if os.path.exists(session_file):
                    context_kwargs["storage_state"] = session_file
                context = browser.new_context(**context_kwargs)
                page = context.new_page()
                conn = init_db()
                try:
                    # Background runs should respect the CLI resume flag if provided via env or caller; default to True
                    tweets = fetch_enhanced_tweets(page, username, max_tweets=max_tweets, resume_from_last=True)
                except Exception as e:
                    tweets = []
                    try:
                        conn_err = sqlite3.connect(DB_PATH)
                        cur_err = conn_err.cursor()
                        cur_err.execute("INSERT INTO scrape_errors (username, tweet_id, error, context, timestamp) VALUES (?, ?, ?, ?, ?)", (
                            username,
                            None,
                            str(e),
                            'background_fetch',
                            datetime.now().isoformat()
                        ))
                        conn_err.commit()
                        conn_err.close()
                    except Exception:
                        pass

                total = 0
                for i, tweet in enumerate(tweets, 1):
                    try:
                        save_enhanced_tweet(conn, tweet)
                        total += 1
                    except Exception:
                        pass

                # Print final summary to stdout (captured in parent)
                print(f"Background fetch complete for @{username}: saved {total} tweets")
                conn.close()
                browser.close()
        finally:
            os._exit(0)

if __name__ == "__main__":
    main()
