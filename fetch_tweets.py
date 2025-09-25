import os
import sqlite3
import time
import argparse
import json
import random
import re
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError
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
    return conn

def save_enhanced_tweet(conn, tweet_data: Dict):
    """Save tweet with enhanced data structure."""
    c = conn.cursor()
    try:
        # Check if tweet already exists
        c.execute("SELECT id FROM tweets WHERE tweet_id = ?", (tweet_data['tweet_id'],))
        if c.fetchone():
            print(f"  ⏭️ Tweet {tweet_data['tweet_id']} already exists, skipping")
            return
        
        # Insert with all enhanced fields including profile picture
        c.execute("""
            INSERT INTO tweets (
                tweet_id, tweet_url, username, content,
                post_type, is_pinned, 
                original_author, original_tweet_id, original_content,
                reply_to_username, reply_to_tweet_id, reply_to_content,
                thread_position, thread_root_id,
                media_links, media_count, media_types,
                has_external_link, external_links,
                hashtags, mentions,
                engagement_retweets, engagement_likes, engagement_replies, engagement_views,
                tweet_timestamp, scrape_timestamp,
                is_repost, is_like, is_comment, parent_tweet_id, profile_pic_url
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            tweet_data['tweet_id'], tweet_data['tweet_url'], tweet_data['username'], tweet_data['content'],
            tweet_data['post_type'], tweet_data['is_pinned'],
            tweet_data.get('original_author'), tweet_data.get('original_tweet_id'), tweet_data.get('original_content'),
            tweet_data.get('reply_to_username'), tweet_data.get('reply_to_tweet_id'), tweet_data.get('reply_to_content'), 
            tweet_data.get('thread_position', 0), tweet_data.get('thread_root_id'),
            tweet_data.get('media_links'), tweet_data.get('media_count', 0), tweet_data.get('media_types'),
            tweet_data.get('has_external_link', 0), tweet_data.get('external_links'),
            tweet_data.get('hashtags'), tweet_data.get('mentions'),
            tweet_data.get('engagement_retweets', 0), tweet_data.get('engagement_likes', 0), 
            tweet_data.get('engagement_replies', 0), tweet_data.get('engagement_views', 0),
            tweet_data.get('tweet_timestamp'), datetime.now().isoformat(),
            tweet_data.get('is_repost', 0), tweet_data.get('is_like', 0), 
            tweet_data.get('is_comment', 0), tweet_data.get('parent_tweet_id'),
            tweet_data.get('profile_pic_url')
        ))
        
        conn.commit()
        print(f"  ✅ Saved {tweet_data['post_type']} tweet: {tweet_data['tweet_id']}")
        
    except Exception as e:
        print(f"  ❌ Error saving tweet {tweet_data.get('tweet_id', 'unknown')}: {e}")


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
    - reply: Direct replies to other tweets
    - quote: Quote tweets with additional commentary
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
    pinned_indicator = article.query_selector('[data-testid="socialContext"]:has-text("Pinned"), [aria-label*="Pinned"]')
    if pinned_indicator:
        post_analysis['is_pinned'] = 1
        post_analysis['should_skip'] = True  # Skip pinned posts
        print("    📌 Pinned post detected - will skip")
        return post_analysis
    
    # Check for repost indicators
    repost_indicators = [
        '[data-testid="socialContext"]:has-text("Reposted"), [data-testid="socialContext"]:has-text("reposted")',
        'svg[data-testid="iconRetweet"]',
        '[aria-label*="Repost"], [aria-label*="repost"]'
    ]
    
    repost_element = None
    for selector in repost_indicators:
        repost_element = article.query_selector(selector)
        if repost_element:
            break
    
    if repost_element:
        # This is a repost - determine if it's own content or other's content
        
        # Look for the original tweet within this article
        quoted_tweet = article.query_selector('[data-testid="tweetText"] ~ div article, [role="article"] [role="article"]')
        
        if quoted_tweet:
            # Extract original author info
            original_author_link = quoted_tweet.query_selector('a[href^="/"][role="link"]')
            if original_author_link:
                original_author_href = original_author_link.get_attribute('href')
                original_author = original_author_href.replace('/', '') if original_author_href else None
                
                # Check if reposting own content
                if original_author == target_username:
                    post_analysis['post_type'] = 'repost_own'
                    print(f"    🔄 Own repost detected: @{original_author}")
                else:
                    post_analysis['post_type'] = 'repost_other'
                    print(f"    🔄 Other's repost detected: @{original_author}")
                
                post_analysis['original_author'] = original_author
                
                # Extract original tweet ID
                original_tweet_link = quoted_tweet.query_selector('a[href*="/status/"]')
                if original_tweet_link:
                    original_href = original_tweet_link.get_attribute('href')
                    post_analysis['original_tweet_id'] = original_href.split('/')[-1] if original_href else None
                
                # Extract original content
                original_text_elem = quoted_tweet.query_selector('[data-testid="tweetText"]')
                if original_text_elem:
                    post_analysis['original_content'] = original_text_elem.inner_text().strip()
        
        return post_analysis
    
    # Check for reply indicators
    reply_context = article.query_selector('[data-testid="socialContext"]:has-text("Replying to"), [data-testid="tweetText"]:has-text("Replying to")')
    
    if reply_context:
        post_analysis['post_type'] = 'reply'
        
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
    
    if main_text and quoted_content:
        post_analysis['post_type'] = 'quote'
        
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
        
        print(f"    🗨️ Quote tweet detected: @{post_analysis['reply_to_username']}")
        return post_analysis
    
    # Check for thread indicators (consecutive tweets by same user)
    thread_indicator = article.query_selector('[data-testid="socialContext"]:has-text("Show this thread")')
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
                
                # Extract tweet content
                text_elem = article.query_selector('[data-testid="tweetText"]')
                content = ""
                if text_elem:
                    # Get full text including all spans and links
                    content = text_elem.inner_text().strip()
                
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
                
                # Check if this tweet already exists (skip duplicates)
                if resume_from_last and check_if_tweet_exists(username, tweet_id):
                    print(f"    ⏭️ Skipping existing tweet ({tweet_id})")
                    consecutive_old_tweets += 1
                    # Be very patient - only stop if we've tried a LOT and found no older tweets
                    stop_threshold = max_consecutive_old if found_older_tweets_yet else max_consecutive_old * 3
                    if consecutive_old_tweets >= stop_threshold:
                        reason = "consecutive existing tweets"
                        print(f"    🛑 Seen {consecutive_old_tweets} {reason} - stopping")
                        return collected_tweets
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
                    'is_comment': 1 if post_analysis['post_type'] in ['reply', 'quote'] else 0,
                    'parent_tweet_id': post_analysis.get('reply_to_tweet_id') or post_analysis.get('original_tweet_id')
                }
                
                collected_tweets.append(tweet_data)
                seen_tweet_ids.add(tweet_id)
                
                print(f"  ✅ [{len(collected_tweets):2d}/{max_tweets}] {post_analysis['post_type']}: {tweet_id}")
                
            except Exception as e:
                print(f"    ❌ Error processing tweet {i+1}: {e}")
                continue
        
        # Human-like scrolling with deep scroll mode for finding older tweets
        if len(collected_tweets) < max_tweets:
            print(f"  📜 Scrolling for more tweets... ({len(collected_tweets)}/{max_tweets})")
            # Use deep scroll if we're in resume mode and looking for older tweets
            use_deep_scroll = resume_from_last and consecutive_old_tweets > 10
            if use_deep_scroll:
                print("  🔍 Switching to deep scroll mode to find older tweets...")
            random_scroll_pattern(page, deep_scroll=use_deep_scroll)
            
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
    
    parser = argparse.ArgumentParser(description="Fetch tweets from a given X (Twitter) user.")
    parser.add_argument("username", nargs='?', help="Optional single username to fetch tweets from (without @). If omitted, default targets list will be used.")
    parser.add_argument("--max", type=int, default=100, help="Maximum number of tweets to fetch per user (default: 100)")
    parser.add_argument("--handles-file", help="Path to a newline-separated file with target handles (overrides defaults)")
    args = parser.parse_args()

    username_to_fetch = args.username
    max_tweets = args.max

    handles = DEFAULT_HANDLES.copy()
    # If a handles file exists, use it
    if args.handles_file and os.path.exists(args.handles_file):
        with open(args.handles_file, 'r') as f:
            handles = [l.strip() for l in f if l.strip()]
    # If a single username passed on CLI, use only that
    if username_to_fetch:
        handles = [username_to_fetch]

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
            try:
                tweets = fetch_enhanced_tweets(page, handle, max_tweets=max_tweets, resume_from_last=True)
            except Exception as e:
                print(f"Failed to fetch for {handle}: {e}")
                tweets = []
            print(f"✅ Fetched {len(tweets)} tweets for @{handle}")
            
            # Save each tweet with comprehensive data
            for i, tweet in enumerate(tweets, 1):
                try:
                    save_enhanced_tweet(conn, tweet)
                    print(f"  💾 Saved tweet {i}/{len(tweets)}: {tweet['tweet_id']}")
                except Exception as e:
                    print(f"  ❌ Error saving tweet {tweet.get('tweet_id', 'unknown')}: {e}")
            
            # Save profile information if tweets were collected successfully
            if tweets and 'profile_pic_url' in tweets[0]:
                profile_pic_url = tweets[0]['profile_pic_url']
                save_account_profile_info(conn, handle, profile_pic_url)
            
            total += len(tweets)
        
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

if __name__ == "__main__":
    main()
