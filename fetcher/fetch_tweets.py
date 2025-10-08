"""
Full fetch implementation moved into the `fetcher` package.

This file contains the full implementation that used to live at the
project root `fetch_tweets.py`. It's placed inside the `fetcher` package
to consolidate fetch-related helpers and make testing/refactoring easier.
"""

import os
import sqlite3
import time
import argparse
import sys
import json
import random
import re
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports when run as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fetcher import db as fetcher_db
from fetcher import parsers as fetcher_parsers
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

def collect_tweets_from_page(page, username: str, max_tweets: int, resume_from_last: bool, oldest_timestamp: Optional[str], profile_pic_url: Optional[str]) -> List[Dict]:
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
                post_analysis = fetcher_parsers.analyze_post_type(article, username)
                
                # Skip pinned posts as requested
                if post_analysis['should_skip']:
                    continue
                
                tweet_url = f"https://x.com{href}"
                
                # Extract tweet content with expansion handling
                content = fetcher_parsers.extract_full_tweet_content(article)
                
                # Skip if no content (media-only posts still have some text usually)
                if not content:
                    print(f"    ⏭️ Skipping content-less tweet: {tweet_id}")
                    continue
                
                # Extract media information  
                media_links, media_count, media_types = fetcher_parsers.extract_media_data(article)
                
                # Extract engagement metrics
                engagement = fetcher_parsers.extract_engagement_metrics(article)
                
                # Extract content elements (hashtags, mentions, links)
                content_elements = fetcher_parsers.extract_content_elements(article)
                
                # Extract timestamp if available
                time_elem = article.query_selector('time')
                tweet_timestamp = None
                if time_elem:
                    tweet_timestamp = time_elem.get_attribute('datetime')
                
                # Check if this tweet already exists (skip duplicates) but allow updates when analysis/content changed
                if resume_from_last and fetcher_db.check_if_tweet_exists(username, tweet_id):
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
                if resume_from_last and tweet_timestamp and fetcher_parsers.should_skip_existing_tweet(tweet_timestamp, oldest_timestamp):
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
    
    return collected_tweets


def fetch_enhanced_tweets(page, username: str, max_tweets: int = 30, resume_from_last: bool = True) -> List[Dict]:
    """
    Enhanced tweet fetching with comprehensive post type detection.
    
    If resume_from_last is True, will skip existing tweets and fetch older ones.
    """
    print(f"\n🎯 Starting enhanced tweet collection for @{username}")
    
    # Get oldest tweet timestamp for continuing from where we left off
    oldest_timestamp = None
    if resume_from_last:
        oldest_timestamp = fetcher_db.get_last_tweet_timestamp(username)
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
    profile_pic_url = fetcher_parsers.extract_profile_picture(page, username)
    
    # Collect tweets from the page
    collected_tweets = collect_tweets_from_page(page, username, max_tweets, resume_from_last, oldest_timestamp, profile_pic_url)
    
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

def run_fetch_session(p, handles: List[str], max_tweets: int, resume_from_last_flag: bool):
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
    
    # Create browser context and page
    if os.path.exists(session_file):
        context_kwargs["storage_state"] = session_file
    context = browser.new_context(**context_kwargs)
    page = context.new_page()
    
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
                saved = fetcher_db.save_enhanced_tweet(conn, tweet)
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
            fetcher_db.save_account_profile_info(conn, handle, profile_pic_url)
        
        total += saved_count_for_handle
    
    conn.close()
    context.close()
    browser.close()
    return total, len(handles)


def refetch_single_tweet(tweet_id: str) -> bool:
    """
    Re-fetch a specific tweet by ID, extracting complete content including quoted tweets.
    
    Args:
        tweet_id: The tweet ID to refetch
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"🔄 REFETCH MODE: Re-fetching tweet ID {tweet_id}")
    
    # Get tweet info from database
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT username, tweet_url FROM tweets WHERE tweet_id = ?", (tweet_id,))
        row = cur.fetchone()
        conn.close()
        
        if not row:
            print(f"❌ Tweet ID {tweet_id} not found in database. Cannot refetch.")
            return False
        
        username = row['username']
        tweet_url = row['tweet_url']
        print(f"📍 Found tweet from @{username}")
        print(f"🔗 URL: {tweet_url}")
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False
    
    # Use Playwright to refetch the tweet
    try:
        with sync_playwright() as p:
            # Reuse existing browser setup (same as run_fetch_session)
            browser = p.chromium.launch(headless=False, slow_mo=50)
            
            # Select random user agent
            selected_user_agent = random.choice(USER_AGENTS)
            
            # Load session if available
            session_file = "x_session.json"
            context_kwargs = {
                "user_agent": selected_user_agent,
                "viewport": {"width": 1280, "height": 720},
                "locale": "en-US",
                "timezone_id": "America/New_York",
                "color_scheme": "light",
                "java_script_enabled": True,
            }
            if os.path.exists(session_file):
                context_kwargs["storage_state"] = session_file
            context = browser.new_context(**context_kwargs)
            page = context.new_page()
            
            # Load the tweet page
            print(f"🌐 Loading tweet page...")
            try:
                page.goto(tweet_url, wait_until="domcontentloaded", timeout=60000)
                human_delay(2.0, 3.0)
            except Exception as e:
                print(f"⚠️ Page load warning: {e}")
            
            # Extract tweet data
            tweet_data = _extract_tweet_with_quoted_content(page, tweet_id, username, tweet_url)
            
            if not tweet_data:
                print(f"❌ Failed to extract tweet data")
                context.close()
                browser.close()
                return False
            
            # Update database
            success = _update_tweet_in_database(tweet_id, tweet_data)
            
            context.close()
            browser.close()
            
            return success
            
    except Exception as e:
        print(f"❌ Error during refetch: {e}")
        import traceback
        traceback.print_exc()
        return False


def _extract_tweet_with_quoted_content(page, tweet_id: str, username: str, tweet_url: str) -> dict:
    """
    Extract complete tweet data including quoted tweet content and media.
    
    Args:
        page: Playwright page object
        tweet_id: Tweet ID
        username: Tweet author username  
        tweet_url: Tweet URL
        
    Returns:
        dict: Tweet data with all fields, or None if extraction failed
    """
    # Find main tweet article
    articles = page.query_selector_all('article[data-testid="tweet"]')
    if not articles:
        print(f"❌ Could not find tweet content on page")
        return None

    main_article = articles[0]
    print(f"✅ Found main tweet article")

    # Extract main tweet content
    content = fetcher_parsers.extract_full_tweet_content(main_article)
    post_analysis = fetcher_parsers.analyze_post_type(main_article, username)
    post_analysis['tweet_id'] = tweet_id  # Add tweet_id for filtering
    main_media_links, main_media_count, main_media_types = fetcher_parsers.extract_media_data(main_article)
    engagement = fetcher_parsers.extract_engagement_metrics(main_article)
    content_elements = fetcher_parsers.extract_content_elements(main_article)

    print(f"✅ Main tweet extracted: {len(content)} chars, {main_media_count} media")
    
    # Enhanced quoted tweet detection with multiple strategies
    quoted_tweet_data = _find_and_extract_quoted_tweet(page, main_article, post_analysis)
    
    # Combine main and quoted media (Option A: single media_links field)
    combined_media_links = list(main_media_links) if main_media_links else []
    if quoted_tweet_data and quoted_tweet_data.get('media_links'):
        quoted_media = quoted_tweet_data['media_links']
        # Add quoted media that's not already in main media
        for media_url in quoted_media:
            if media_url not in combined_media_links:
                combined_media_links.append(media_url)
        if quoted_tweet_data.get('media_count', 0) > 0:
            print(f"📎 Added {quoted_tweet_data['media_count']} media from quoted tweet")
    
    combined_media_count = len(combined_media_links)
    
    # Build complete tweet data
    tweet_data = {
        'tweet_id': tweet_id,
        'tweet_url': tweet_url,
        'username': username,
        'content': content,
        'post_type': post_analysis.get('post_type', 'original'),
        'original_author': post_analysis.get('original_author'),
        'original_tweet_id': post_analysis.get('original_tweet_id'),
        'original_content': post_analysis.get('original_content'),
        'reply_to_username': post_analysis.get('reply_to_username'),
        'media_links': ','.join(combined_media_links) if combined_media_links else None,
        'media_count': combined_media_count,
        'hashtags': ','.join(content_elements.get('hashtags', [])) if content_elements.get('hashtags') else None,
        'mentions': ','.join(content_elements.get('mentions', [])) if content_elements.get('mentions') else None,
        'external_links': ','.join(content_elements.get('external_links', [])) if content_elements.get('external_links') else None,
        'engagement_likes': engagement.get('likes', 0),
        'engagement_retweets': engagement.get('retweets', 0),
        'engagement_replies': engagement.get('replies', 0),
    }
    
    print(f"📝 Main content: {content[:100]}...")
    if tweet_data['original_content']:
        print(f"📎 Quoted content: {tweet_data['original_content'][:100]}...")
    if combined_media_count > 0:
        print(f"🖼️ Total media (main + quoted): {combined_media_count} items")
    
    return tweet_data


def _find_and_extract_quoted_tweet(page, main_article, post_analysis: dict) -> dict:
    """
    Find and extract quoted tweet using multiple detection strategies.
    
    Args:
        page: Playwright page object
        main_article: Main tweet article element
        post_analysis: Post analysis dict to update with quoted content
        
    Returns:
        dict: Quoted tweet data, or None if not found
    """
    # Strategy 1: Parser already found it
    if post_analysis.get('original_content'):
        print(f"✅ Parser found quoted content: {len(post_analysis['original_content'])} chars")
        return {'content': post_analysis['original_content']}
    
    print(f"🔍 Searching for quoted tweet with multiple strategies...")
    
    # Strategy 2: Look for various quoted tweet card selectors
    quoted_card_selectors = [
        # New: Multiple tweet texts means there's a quoted tweet
        # The page has main tweet text + quoted tweet text
        ('check_multiple_tweets', None),  # Special check
        # Original nested article selector (timeline view)
        '[role="article"] [role="article"]',
        '[data-testid="tweetText"] ~ div [role="article"]',
        '[data-testid="card.wrapper"]',  # Quote card wrapper
        'div[class*="quoted"]',  # Any div with "quoted" in class  
        'article div[data-testid="card.layoutLarge.media"]',  # Large media card
        'article a[href*="/status/"] img[alt][src*="pbs.twimg.com"]',  # Link with preview image (parent)
    ]
    
    quoted_card = None
    for selector_item in quoted_card_selectors:
        try:
            # Handle special case for multiple tweet check
            if isinstance(selector_item, tuple) and selector_item[0] == 'check_multiple_tweets':
                # Look for all tweetText elements WITHIN the main article
                tweet_texts = main_article.query_selector_all('[data-testid="tweetText"]')
                if len(tweet_texts) >= 2:
                    # Second one is the quoted tweet (nested inside main article)
                    quoted_text_elem = tweet_texts[1]
                    print(f"✅ Found quoted tweet (2nd tweetText element within main article)")
                    # Get the text content first
                    quoted_text = quoted_text_elem.inner_text() if hasattr(quoted_text_elem, 'inner_text') else None
                    if quoted_text:
                        post_analysis['original_content'] = quoted_text
                        print(f"📎 Quoted content: {quoted_text[:150]}...")
                    
                    # Now click on the quoted tweet to navigate to it for complete extraction
                    try:
                        print(f"🖱️ Clicking on quoted tweet to navigate to it...")
                        quoted_text_elem.click()
                        page.wait_for_load_state("domcontentloaded")
                        human_delay(5.0, 7.0)

                        # Try to dismiss overlays/popups
                        try:
                            overlay = page.query_selector('div:has-text("unusual activity"), div:has-text("actividad inusual"), div[role="dialog"]')
                            if overlay:
                                close_btn = overlay.query_selector('button, [role="button"]')
                                if close_btn:
                                    print("🛑 Dismissing overlay/popup...")
                                    close_btn.click()
                                    human_delay(1.0, 2.0)
                        except Exception as e:
                            print(f"⚠️ Overlay dismissal failed: {e}")

                        # Poll for media elements for up to 10 seconds
                        poll_start = time.time()
                        found_media = False
                        while time.time() - poll_start < 10:
                            quoted_articles = page.query_selector_all('article[data-testid="tweet"]')
                            if quoted_articles:
                                quoted_main = quoted_articles[0]
                                video_elems = quoted_main.query_selector_all('video, div[aria-label*="Video"], div[data-testid*="videoPlayer"]')
                                img_elems = quoted_main.query_selector_all('img[src*="twimg.com"], img[src*="pbs.twimg.com"]')
                                if video_elems or img_elems:
                                    print(f"� Poll: Found {len(video_elems)} video and {len(img_elems)} image elements.")
                                    found_media = True
                                    # Try clicking video/play if present
                                    try:
                                        if video_elems:
                                            print(f"🖱️ Clicking video element...")
                                            video_elems[0].click()
                                            human_delay(1.0, 2.0)
                                        play_btn = quoted_main.query_selector('button[aria-label*="Play"], div[role="button"][aria-label*="Play"]')
                                        if play_btn:
                                            print(f"🖱️ Clicking play button...")
                                            play_btn.click()
                                            human_delay(1.0, 2.0)
                                    except Exception as e:
                                        print(f"⚠️ Video/play click failed: {e}")
                                    break
                            human_delay(1.0, 1.5)
                        if not found_media:
                            print("⚠️ No media found after polling.")

                        # Playwright stealth mode (if available)
                        try:
                            if hasattr(page.context, 'use_stealth'):
                                print("🕵️ Enabling Playwright stealth mode...")
                                page.context.use_stealth()
                        except Exception as e:
                            print(f"⚠️ Stealth mode not available: {e}")
                        if quoted_articles:
                            quoted_main = quoted_articles[0]

                            quoted_full_content = fetcher_parsers.extract_full_tweet_content(quoted_main)
                            quoted_media_links, quoted_media_count, quoted_media_types = fetcher_parsers.extract_media_data(quoted_main)
                            quoted_elements = fetcher_parsers.extract_content_elements(quoted_main)

                            # Update post_analysis with full content
                            post_analysis['original_content'] = quoted_full_content

                            print(f"✅ Complete quoted tweet extracted: {len(quoted_full_content)} chars, {quoted_media_count} media")
                            if quoted_media_count > 0:
                                print(f"   🖼️ Media: {', '.join(quoted_media_links[:3])}")

                            return {
                                'content': quoted_full_content,
                                'media_links': quoted_media_links,
                                'media_count': quoted_media_count,
                                'media_types': quoted_media_types,
                                'hashtags': quoted_elements.get('hashtags', []),
                                'mentions': quoted_elements.get('mentions', []),
                            }
                        else:
                            print(f"⚠️ No article found after clicking quoted tweet")
                    except Exception as e:
                        print(f"⚠️ Could not click and extract quoted tweet: {e}")
                continue
            
            selector = selector_item
            if selector == 'article a[href*="/status/"] img[alt][src*="pbs.twimg.com"]':
                # Special case: image preview means there's a quoted tweet nearby
                img_elem = main_article.query_selector(selector)
                if img_elem:
                    # Get the parent link
                    parent = img_elem.evaluate_handle('el => el.closest("a[href*=\\"/status/\\"]")')
                    if parent:
                        quoted_card = parent.as_element()
                        print(f"✅ Found quoted card via image preview")
                        break
            else:
                quoted_card = main_article.query_selector(selector)
                if quoted_card:
                    print(f"✅ Found quoted card with selector: {selector}")
                    break
        except Exception as e:
            continue
    
    # Strategy 2.5: Look for ANY link to another tweet inside the main article
    if not quoted_card:
        print(f"🔍 Trying alternative: looking for any /status/ link in article...")
        try:
            all_links = main_article.query_selector_all('a[href*="/status/"]')
            for link in all_links:
                href = link.get_attribute('href')
                # Make sure it's not the main tweet's own link
                if href and '/status/' in href:
                    parts = href.strip('/').split('/')
                    if 'status' in parts:
                        status_index = parts.index('status')
                        if status_index >= 1:
                            linked_tweet_id = parts[status_index + 1].split('?')[0]
                            # If it's a different tweet ID, it might be a quoted tweet
                            if linked_tweet_id != post_analysis.get('tweet_id', ''):
                                quoted_card = link
                                print(f"✅ Found potential quoted tweet link: {href}")
                                break
        except Exception as e:
            print(f"⚠️ Alternative link search failed: {e}")
    
    if not quoted_card:
        print(f"⚠️ No quoted tweet card found in main article")
        return None
    
    # Extract quoted tweet URL and author
    try:
        # Try multiple methods to find the quoted tweet URL
        quoted_link = None
        quoted_href = None
        
        # Method 1: Direct link in quoted card
        quoted_link = quoted_card.query_selector('a[href*="/status/"]')
        if quoted_link:
            quoted_href = quoted_link.get_attribute('href')
        
        # Method 2: If quoted_card IS a link
        if not quoted_href and quoted_card.tag_name.lower() == 'a':
            quoted_href = quoted_card.get_attribute('href')
        
        # Method 3: Look for time element which usually has the link
        if not quoted_href:
            time_elem = quoted_card.query_selector('time')
            if time_elem:
                parent_link = time_elem.evaluate_handle('el => el.closest("a[href*=\\"/status/\\"]")').as_element()
                if parent_link:
                    quoted_href = parent_link.get_attribute('href')
        
        # Method 4: Search for ANY link with status inside the quoted card area
        if not quoted_href:
            all_links = quoted_card.query_selector_all('a[href*="/status/"]')
            for link in all_links:
                href = link.get_attribute('href')
                if href and '/status/' in href:
                    # Make sure it's not the main tweet
                    if post_analysis.get('tweet_id', '') not in href:
                        quoted_href = href
                        break
        
        if not quoted_href or '/status/' not in quoted_href:
            print(f"⚠️ No status link found in quoted card")
            return None
        
        # Extract author and tweet ID from URL
        parts = quoted_href.strip('/').split('/')
        status_index = parts.index('status') if 'status' in parts else -1
        if status_index < 1:
            print(f"⚠️ Cannot parse quoted tweet URL: {quoted_href}")
            return None
        
        quoted_author = parts[status_index - 1]
        quoted_tweet_id = parts[status_index + 1].split('?')[0]  # Remove query params
        quoted_tweet_url = f"https://x.com/{quoted_author}/status/{quoted_tweet_id}"
        
        post_analysis['reply_to_username'] = quoted_author
        post_analysis['reply_to_tweet_id'] = quoted_tweet_id
        
        print(f"🔗 Found quoted tweet by @{quoted_author}: {quoted_tweet_url}")
        
        # Try to extract embedded text first
        quoted_text_elem = quoted_card.query_selector('[data-testid="tweetText"]')
        if quoted_text_elem:
            embedded_text = quoted_text_elem.inner_text().strip()
            print(f"📄 Extracted embedded text: {len(embedded_text)} chars")
        else:
            embedded_text = None
        
        # Strategy 3: Visit the quoted tweet page for complete content
        print(f"🌐 Visiting quoted tweet page for complete extraction...")
        try:
            page.goto(quoted_tweet_url, wait_until="domcontentloaded", timeout=60000)
            human_delay(2.0, 3.0)
            
            # Extract complete quoted tweet
            quoted_articles = page.query_selector_all('article[data-testid="tweet"]')
            if quoted_articles:
                quoted_main = quoted_articles[0]
                
                quoted_full_content = fetcher_parsers.extract_full_tweet_content(quoted_main)
                quoted_media_links, quoted_media_count, quoted_media_types = fetcher_parsers.extract_media_data(quoted_main)
                quoted_elements = fetcher_parsers.extract_content_elements(quoted_main)
                
                # Update post_analysis with full content
                post_analysis['original_content'] = quoted_full_content
                
                print(f"✅ Complete quoted tweet: {len(quoted_full_content)} chars, {quoted_media_count} media")
                if quoted_media_count > 0:
                    print(f"   🖼️ Media: {', '.join(quoted_media_links[:3])}")
                
                return {
                    'content': quoted_full_content,
                    'media_links': quoted_media_links,
                    'media_count': quoted_media_count,
                    'media_types': quoted_media_types,
                    'hashtags': quoted_elements.get('hashtags', []),
                    'mentions': quoted_elements.get('mentions', []),
                    'external_links': quoted_elements.get('external_links', [])
                }
        except Exception as e:
            print(f"⚠️ Could not visit quoted tweet: {e}")
            # Fall back to embedded text if available
            if embedded_text:
                post_analysis['original_content'] = embedded_text
                print(f"⚠️ Using embedded text as fallback")
                return {'content': embedded_text}
            
    except Exception as e:
        print(f"⚠️ Error extracting quoted tweet: {e}")
    
    return None


def _update_tweet_in_database(tweet_id: str, tweet_data: dict) -> bool:
    """
    Update tweet in database with refetched data.
    
    Args:
        tweet_id: Tweet ID
        tweet_data: Complete tweet data dict
        
    Returns:
        bool: True if successful
    """
    try:
        conn = init_db()
        c = conn.cursor()
        
        # Direct UPDATE to force save all fields
        c.execute("""
            UPDATE tweets SET 
                original_content = ?,
                reply_to_username = ?,
                media_links = ?,
                media_count = ?,
                engagement_likes = ?,
                engagement_retweets = ?,
                engagement_replies = ?
            WHERE tweet_id = ?
        """, (
            tweet_data['original_content'],
            tweet_data.get('reply_to_username'),
            tweet_data['media_links'],
            tweet_data['media_count'],
            tweet_data['engagement_likes'],
            tweet_data['engagement_retweets'],
            tweet_data['engagement_replies'],
            tweet_id
        ))
        
        rows_updated = c.rowcount
        conn.commit()
        conn.close()
        
        if rows_updated > 0:
            print(f"💾 Tweet updated in database ({rows_updated} rows)")
            return True
        else:
            print(f"⚠️ No rows updated - tweet may not exist")
            return False
            
    except Exception as e:
        print(f"❌ Database update error: {e}")
        return False


def main():
    start_time = time.time()  # Start timing

    parser = argparse.ArgumentParser(description="Fetch tweets from a given X (Twitter) user.")
    parser.add_argument("--user", "-u", dest="user", help="Optional single username to fetch tweets from (with or without leading @). Overrides positional username.")
    parser.add_argument("--max", type=int, default=100, help="Maximum number of tweets to fetch per user (default: 100)")
    parser.add_argument("--handles-file", help="Path to a newline-separated file with target handles (overrides defaults)")
    parser.add_argument("--no-resume", action='store_true', help="Do not resume from previous scrape; fetch recent tweets instead of older ones")
    parser.add_argument("--refetch", help="Re-fetch a specific tweet ID (bypasses exists check and updates database)")
    args = parser.parse_args()

    # Handle refetch mode for specific tweet
    # Handle refetch mode for specific tweet
    if args.refetch:
        tweet_id = args.refetch.strip()
        success = refetch_single_tweet(tweet_id)
        return  # Exit after refetch

    # Resolve effective target username robustly. --user takes precedence, then positional.
    effective_user = None
    if args.user:
        effective_user = args.user.strip()
    elif getattr(args, 'username', None):
        effective_user = args.username.strip()

    if effective_user:
        # Normalize leading @ if present
        effective_user = effective_user.lstrip('@').strip()

    max_tweets = args.max

    # Diagnostic: show parsed args for debugging
    print(f"🐞 parsed args: user={args.user} username={getattr(args, 'username', None)} max={max_tweets} handles_file={args.handles_file}")

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
        total, accounts_processed = run_fetch_session(p, handles, max_tweets, resume_from_last_flag)
    
    # Calculate and display execution time
    end_time = time.time()
    execution_time = end_time - start_time
    minutes = int(execution_time // 60)
    seconds = int(execution_time % 60)
    
    print(f"\n⏱️  Execution completed in: {minutes}m {seconds}s")
    print(f"📊 Total tweets fetched and saved: {total}")
    print(f"🎯 Accounts processed: {accounts_processed}")
    print(f"📈 Average tweets per account: {total/accounts_processed:.1f}")



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
                        fetcher_db.save_enhanced_tweet(conn, tweet)
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
