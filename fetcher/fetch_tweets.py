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
    """Login with better anti-detection measures."""
    
    print("🔐 Starting login process...")
    
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

def collect_tweets_from_page(page, username: str, max_tweets: int, resume_from_last: bool, oldest_timestamp: Optional[str], profile_pic_url: Optional[str], conn) -> List[Dict]:
    collected_tweets = []
    seen_tweet_ids = set()
    consecutive_empty_scrolls = 0
    max_consecutive_empty = 15  # Stop after 15 consecutive failed scrolls
    last_height = 0
    tweets_found_this_cycle = 0
    last_tweet_count = 0
    saved_count = 0  # Track actually saved tweets
    
    if max_tweets == float('inf'):
        print(f"🔍 Collecting unlimited tweets...")
    else:
        print(f"🔍 Collecting up to {max_tweets} tweets...")
    
    def try_recovery_strategies(page, attempt_number: int) -> bool:
        """Try different recovery strategies when Twitter stops serving content."""
        strategies = [
            ("refresh_page", lambda: page.reload(wait_until="domcontentloaded")),
            ("clear_cache", lambda: page.evaluate("localStorage.clear(); sessionStorage.clear();")),
            ("jump_to_bottom", lambda: page.evaluate("window.scrollTo(0, document.body.scrollHeight)")),
            ("force_reload_tweets", lambda: page.evaluate("window.location.reload(true)")),
            ("random_scroll_pattern", lambda: page.evaluate(f"window.scrollBy(0, {1000 + random.randint(500, 2000)})")),
        ]
        
        if attempt_number <= len(strategies):
            strategy_name, strategy_func = strategies[attempt_number - 1]
            try:
                print(f"    🔧 Trying recovery strategy {attempt_number}: {strategy_name}")
                strategy_func()
                human_delay(3.0, 6.0)  # Longer delay for recovery
                
                # Check if we can find tweet elements after recovery
                articles = page.query_selector_all('article[data-testid="tweet"]')
                if articles:
                    print(f"    ✅ Recovery successful: found {len(articles)} articles")
                    return True
                else:
                    print(f"    ❌ Recovery failed: no articles found")
                    return False
            except Exception as e:
                print(f"    ⚠️ Recovery strategy {strategy_name} failed: {e}")
                return False
        
        return False
    
    def event_scroll_cycle(page, iteration):
        # More varied scrolling patterns to avoid detection
        scroll_patterns = [
            lambda: page.evaluate("window.scrollBy(0, 800 + Math.random() * 600)"),  # Normal scroll
            lambda: page.keyboard.press('PageDown'),  # Keyboard scroll
            lambda: page.evaluate("window.scrollBy(0, 1200 + Math.random() * 800)"),  # Larger scroll
            lambda: page.evaluate("window.scrollBy(0, 400 + Math.random() * 300)"),  # Smaller scroll
            lambda: page.evaluate("window.scrollTo(0, window.scrollY + 1000)"),  # Absolute positioning
        ]
        
        try:
            # Vary the pattern based on iteration
            pattern_index = iteration % len(scroll_patterns)
            scroll_patterns[pattern_index]()
        except Exception:
            # Fallback to basic scroll
            try:
                page.evaluate("window.scrollBy(0, 800 + Math.random() * 400)")
            except Exception:
                pass
        
        # Variable delays to seem more human
        if iteration % 10 == 0:
            human_delay(2.0, 4.0)  # Longer pause occasionally
        else:
            human_delay(0.8, 2.2)

    iteration = 0

    while len(collected_tweets) < max_tweets and consecutive_empty_scrolls < max_consecutive_empty:
        # Find all tweet articles on current page
        articles = page.query_selector_all('article[data-testid="tweet"], [data-testid="tweet"]')
        
        print(f"  📄 Found {len(articles)} tweet elements on page")
        
        tweets_found_this_cycle = 0
        
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
                        continue

                    # We need to update existing row: build tweet_data and save immediately
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

                    # Save update immediately
                    try:
                        saved = fetcher_db.save_tweet(conn, tweet_data)
                        if saved:
                            saved_count += 1
                            print(f"  🔁 Updated [{saved_count:2d}] {post_analysis['post_type']}: {tweet_id}")
                        else:
                            print(f"  ⏭️ Update failed or unchanged: {tweet_id}")
                    except Exception as save_e:
                        print(f"  ❌ Error updating tweet {tweet_id}: {save_e}")

                    collected_tweets.append(tweet_data)
                    seen_tweet_ids.add(tweet_id)
                    tweets_found_this_cycle += 1
                    continue

                # Check if we should skip this tweet because it's still in our scraped range
                if resume_from_last and tweet_timestamp and fetcher_parsers.should_skip_existing_tweet(tweet_timestamp, oldest_timestamp):
                    print(f"    ⏭️ Skipping already covered tweet ({tweet_timestamp})")
                    continue

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
                
                # Save tweet immediately instead of accumulating in memory
                try:
                    saved = fetcher_db.save_tweet(conn, tweet_data)
                    if saved:
                        saved_count += 1
                        print(f"  💾 Saved [{saved_count:2d}] {post_analysis['post_type']}: {tweet_id}")
                    else:
                        print(f"  ⏭️ Not saved (duplicate/unchanged): {tweet_id}")
                except Exception as save_e:
                    print(f"  ❌ Error saving tweet {tweet_id}: {save_e}")
                
                # Keep tweet in collected list for compatibility (but we've already saved it)
                collected_tweets.append(tweet_data)
                seen_tweet_ids.add(tweet_id)
                tweets_found_this_cycle += 1
                
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
        
        # Track progress and handle Twitter content limits
        current_tweet_count = len(collected_tweets)
        
        # Human-like scrolling with intelligent stopping
        if len(collected_tweets) < max_tweets:
            if max_tweets == float('inf'):
                print(f"  📜 Scrolling for more tweets... ({current_tweet_count}/∞) - Found {tweets_found_this_cycle} this cycle")
            else:
                print(f"  📜 Scrolling for more tweets... ({current_tweet_count}/{max_tweets}) - Found {tweets_found_this_cycle} this cycle")
            
            # Track consecutive empty scrolls
            if tweets_found_this_cycle == 0:
                consecutive_empty_scrolls += 1
                print(f"    ⚠️ No new tweets found ({consecutive_empty_scrolls}/{max_consecutive_empty} consecutive empty cycles)")
                
                # Try recovery strategies when we hit empty cycles
                if consecutive_empty_scrolls % 5 == 0 and consecutive_empty_scrolls <= 10:
                    recovery_attempt = consecutive_empty_scrolls // 5
                    if try_recovery_strategies(page, recovery_attempt):
                        consecutive_empty_scrolls = max(0, consecutive_empty_scrolls - 3)  # Reward successful recovery
                        print(f"    🔄 Recovery successful, continuing...")
                    else:
                        print(f"    ❌ Recovery failed, consecutive empty count: {consecutive_empty_scrolls}")
            else:
                consecutive_empty_scrolls = 0  # Reset on success
                print(f"    ✅ Found {tweets_found_this_cycle} new tweets, continuing...")
            
            iteration += 1
            try:
                # Use more aggressive scrolling when we're not finding tweets
                if consecutive_empty_scrolls > 5:
                    print(f"    🚀 Using aggressive scrolling (attempt {consecutive_empty_scrolls})")
                    page.evaluate("window.scrollBy(0, 2000 + Math.random() * 1000)")
                    human_delay(3.0, 5.0)
                else:
                    event_scroll_cycle(page, iteration)
            except Exception:
                # Fallback scrolling
                try:
                    page.evaluate("window.scrollBy(0, 1000)")
                    human_delay(2.0, 3.0)
                except Exception:
                    print(f"    ❌ All scrolling methods failed")
                    consecutive_empty_scrolls += 2  # Penalize scroll failures
            
            # Check page height changes (less important now)
            try:
                new_height = page.evaluate("document.body.scrollHeight")
                if new_height == last_height:
                    print(f"    📏 Page height unchanged ({new_height}px)")
                else:
                    print(f"    📏 Page height changed: {last_height}px → {new_height}px")
                last_height = new_height
            except Exception:
                print(f"    ⚠️ Could not check page height")
        
        last_tweet_count = current_tweet_count
    
    # Better termination reporting
    if consecutive_empty_scrolls >= max_consecutive_empty:
        print(f"  🛑 Stopped: Twitter stopped serving new content after {consecutive_empty_scrolls} empty scroll cycles")
        print(f"  📊 This is likely Twitter's content serving limit for @{username}")
        print(f"  💡 Consider using search-based resume or waiting before retrying")
    elif len(collected_tweets) >= max_tweets:
        print(f"  ✅ Completed: Reached target tweet count ({max_tweets})")
    else:
        print(f"  🏁 Stopped: Collection completed")
    
    print(f"  📈 Final count: {len(collected_tweets)} tweets processed, {saved_count} saved to database from @{username}")
    
    return collected_tweets


def convert_timestamp_to_date_filter(timestamp: str) -> Optional[str]:
    """Convert ISO timestamp to Twitter search date format (YYYY-MM-DD)."""
    try:
        # Parse ISO timestamp and convert to date string
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"⚠️ Error converting timestamp {timestamp}: {e}")
        return None


def try_resume_via_search(page, username: str, oldest_timestamp: str) -> bool:
    """
    Try to resume fetching using Twitter's search with date filters.
    
    Args:
        page: Playwright page object
        username: Twitter username to search
        oldest_timestamp: ISO timestamp of oldest scraped tweet
        
    Returns:
        bool: True if successfully navigated to target timeframe
    """
    try:
        # Convert timestamp to date for search
        since_date = convert_timestamp_to_date_filter(oldest_timestamp)
        if not since_date:
            return False
        
        # Build search query to find tweets before our oldest timestamp
        # Use "until" parameter to find tweets before the date we already have
        search_query = f"from:{username} until:{since_date}"
        search_url = f"https://x.com/search?q={search_query}&src=typed_query&f=live"
        
        print(f"🔍 Trying search-based resume: {search_url}")
        page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
        human_delay(3.0, 5.0)
        
        # Check if we got results by looking for tweet articles
        articles = page.query_selector_all('article[data-testid="tweet"]')
        if articles:
            print(f"✅ Search found {len(articles)} tweets in target timeframe")
            return True
        else:
            print("⚠️ No tweets found in search results")
            return False
            
    except Exception as e:
        print(f"⚠️ Search-based resume failed: {e}")
        return False


def resume_positioning(page, username: str, oldest_timestamp: str) -> bool:
    """
    Resume functionality using Twitter search with date filters.
    
    Args:
        page: Playwright page object
        username: Twitter username
        oldest_timestamp: ISO timestamp of oldest scraped tweet
        
    Returns:
        bool: True if successfully positioned for resume
    """
    print(f"🔄 Resume: positioning to fetch tweets older than {oldest_timestamp}")
    
    # Try search-based navigation
    if try_resume_via_search(page, username, oldest_timestamp):
        print("✅ Resume successful via search")
        return True
    
    # Fallback to regular profile load (will use normal scrolling)
    print("⚠️ Search resume failed, falling back to standard profile navigation")
    try:
        profile_url = f"https://x.com/{username}"
        page.goto(profile_url, wait_until="domcontentloaded", timeout=30000)
        human_delay(2.0, 4.0)
        print("✅ Loaded profile page for standard resume")
        return True
    except Exception as e:
        print(f"❌ All resume strategies failed: {e}")
        return False


def fetch_tweets_in_sessions(page, username: str, max_tweets: int, session_size: int = 800) -> List[Dict]:
    """
    Fetch tweets using multiple sessions to work around Twitter's content serving limits.
    
    Args:
        page: Playwright page object
        username: Twitter username
        max_tweets: Total maximum tweets to fetch
        session_size: Maximum tweets per session before refreshing
        
    Returns:
        List of collected tweets from all sessions
    """
    if max_tweets <= session_size:
        # Single session is sufficient - need to get DB connection
        conn = fetcher_db.init_db()
        try:
            result = collect_tweets_from_page(page, username, max_tweets, True, None, None, conn)
            return result
        finally:
            conn.close()
    
    print(f"🔄 Using multi-session strategy: {max_tweets} tweets in sessions of {session_size}")
    
    all_tweets = []
    sessions_completed = 0
    remaining_tweets = max_tweets
    
    # Initialize database connection for all sessions
    conn = fetcher_db.init_db()
    
    try:
        while remaining_tweets > 0 and sessions_completed < 10:  # Max 10 sessions to prevent infinite loops
            session_tweets = min(remaining_tweets, session_size)
            sessions_completed += 1
            
            print(f"\n📍 SESSION {sessions_completed}: Fetching {session_tweets} tweets (remaining: {remaining_tweets})")
            
            # Navigate to profile page for each session
            url = f"https://x.com/{username}"
            print(f"🌐 Loading profile page: {url}")
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            
            # Wait for tweets to load
            try:
                page.wait_for_selector('[data-testid="tweetText"], [data-testid="tweet"]', timeout=15000)
            except TimeoutError:
                print(f"   ❌ Session {sessions_completed}: No tweets found for @{username} or page failed to load")
                break
            
            human_delay(2.0, 4.0)
            
            # Get the oldest timestamp from our current collection to resume properly
            oldest_timestamp = None
            if all_tweets:
                # Find the oldest timestamp from our collected tweets
                timestamps = [t.get('tweet_timestamp') for t in all_tweets if t.get('tweet_timestamp')]
                if timestamps:
                    oldest_timestamp = min(timestamps)
                    print(f"   📅 Resuming from oldest collected: {oldest_timestamp}")
                    
                    # For sessions 2+, use search-based positioning
                    if not resume_positioning(page, username, oldest_timestamp):
                        print(f"   ⚠️ Session {sessions_completed} positioning failed, continuing from profile start")
            
            # Extract profile picture (once per multi-session)
            if sessions_completed == 1:
                print(f"🖼️  Extracting profile picture for @{username}...")
                profile_pic_url = fetcher_parsers.extract_profile_picture(page, username)
            else:
                profile_pic_url = None  # Use cached value from session 1
            
            # Collect tweets for this session
            session_results = collect_tweets_from_page(page, username, session_tweets, True, oldest_timestamp, profile_pic_url, conn)
            
            if not session_results:
                print(f"   ❌ Session {sessions_completed} returned no tweets, stopping multi-session")
                break
            
            # Filter out duplicates (shouldn't happen with proper resume, but safety check)
            existing_ids = {t['tweet_id'] for t in all_tweets}
            new_tweets = [t for t in session_results if t['tweet_id'] not in existing_ids]
            
            all_tweets.extend(new_tweets)
            remaining_tweets -= len(new_tweets)
            
            print(f"   ✅ Session {sessions_completed} complete: {len(new_tweets)} new tweets ({len(session_results)} total returned)")
            print(f"   📊 Progress: {len(all_tweets)}/{max_tweets} tweets collected")
            
            # Break if we didn't get new tweets (hit content limit)
            if len(new_tweets) == 0:
                print(f"   🛑 No new tweets in session {sessions_completed}, stopping")
                break
            
            # Refresh browser session between sessions to reset Twitter's limits
            if remaining_tweets > 0 and sessions_completed < 10:
                print(f"   🔄 Refreshing session for next batch...")
                try:
                    # Clear any cached data between sessions
                    page.evaluate("localStorage.clear(); sessionStorage.clear();")
                    human_delay(3.0, 6.0)  # Longer delay between sessions
                except Exception as e:
                    print(f"   ⚠️ Cache clearing failed: {e}")
        
        print(f"\n🏁 Multi-session complete: {len(all_tweets)} tweets collected in {sessions_completed} sessions")
        return all_tweets
    
    finally:
        conn.close()


def fetch_latest_tweets(page, username: str, max_tweets: int = 30) -> List[Dict]:
    """
    Strategy 1: Fetch latest tweets with simple stopping logic.
    Stops after finding 10 consecutive existing tweets in a row.
    
    Args:
        page: Playwright page object
        username: Twitter username
        max_tweets: Maximum number of tweets to fetch
        
    Returns:
        List of collected tweets
    """
    print(f"🎯 Fetching latest tweets for @{username} (max: {max_tweets})")
    
    # Navigate to profile page
    url = f"https://x.com/{username}"
    print(f"🌐 Loading profile page: {url}")
    page.goto(url, wait_until="domcontentloaded", timeout=30000)
    
    # Wait for tweets to load
    try:
        page.wait_for_selector('[data-testid="tweetText"], [data-testid="tweet"]', timeout=15000)
    except TimeoutError:
        print(f"❌ No tweets found for @{username} or page failed to load")
        return []
    
    human_delay(2.0, 4.0)
    
    # Extract profile picture
    print(f"🖼️  Extracting profile picture for @{username}...")
    profile_pic_url = fetcher_parsers.extract_profile_picture(page, username)
    
    # Initialize database connection
    conn = fetcher_db.init_db()
    
    try:
        tweets_collected = []
        saved_count = 0
        consecutive_existing = 0
        scroll_count = 0
        max_consecutive_existing = 10
        
        while saved_count < max_tweets and consecutive_existing < max_consecutive_existing:
            scroll_count += 1
            tweets_saved_this_scroll = 0
            
            # Get all tweet articles currently visible
            articles = page.query_selector_all('[data-testid="tweet"]')
            
            for article in articles:
                if saved_count >= max_tweets:
                    break
                    
                try:
                    # Extract basic tweet info
                    tweet_link = article.query_selector('a[href*="/status/"]')
                    if not tweet_link:
                        continue
                        
                    href = tweet_link.get_attribute('href')
                    tweet_id = href.split('/')[-1] if href else None
                    
                    if not tweet_id:
                        continue
                    
                    # Check if we already processed this tweet in this session
                    if any(t.get('tweet_id') == tweet_id for t in tweets_collected):
                        continue
                    
                    # Extract content
                    content = fetcher_parsers.extract_full_tweet_content(article)
                    if not content:
                        continue
                    
                    # Analyze post type
                    post_analysis = fetcher_parsers.analyze_post_type(article, username)
                    tweet_url = f"https://x.com/{username}/status/{tweet_id}"
                    
                    # Extract media information
                    media_links, media_count, media_types = fetcher_parsers.extract_media_data(article)
                    
                    # Extract content elements (hashtags, mentions, links)
                    content_elements = fetcher_parsers.extract_content_elements(article)
                    
                    # Build tweet data
                    tweet_data = {
                        'tweet_id': tweet_id,
                        'username': username,
                        'content': content,
                        'tweet_url': tweet_url,
                        'tweet_timestamp': time_elem.get_attribute('datetime') if (time_elem := article.query_selector('time')) else None,
                        'post_type': post_analysis['post_type'],
                        'media_count': media_count,
                        'hashtags': content_elements.get('hashtags'),
                        'mentions': content_elements.get('mentions'),
                        'profile_pic_url': profile_pic_url,
                        'engagement_likes': 0,
                        'engagement_retweets': 0,
                        'engagement_replies': 0,
                        'engagement_views': 0,
                        'is_repost': 1 if 'repost' in post_analysis['post_type'] else 0,
                        'is_comment': 1 if post_analysis['post_type'] == 'repost_reply' else 0,
                        'parent_tweet_id': post_analysis.get('reply_to_tweet_id') or post_analysis.get('original_tweet_id')
                    }
                    
                    # Try to save tweet
                    saved = fetcher_db.save_tweet(conn, tweet_data)
                    if saved:
                        saved_count += 1
                        tweets_saved_this_scroll += 1
                        consecutive_existing = 0  # Reset counter
                        tweets_collected.append(tweet_data)
                        print(f"  💾 Saved [{saved_count}] {post_analysis['post_type']}: {tweet_id}")
                    else:
                        print(f"  ⏭️ Existing tweet: {tweet_id}")
                    
                except Exception as e:
                    print(f"  ❌ Error processing tweet: {e}")
                    continue
            
            # Check consecutive existing logic
            if tweets_saved_this_scroll == 0:
                consecutive_existing += 1
                print(f"  📊 No new tweets saved this scroll ({consecutive_existing}/{max_consecutive_existing})")
            else:
                print(f"  ✅ Saved {tweets_saved_this_scroll} new tweets this scroll")
            
            # Check if we should stop
            if saved_count >= max_tweets:
                print(f"  ✅ Completed: Reached target of {max_tweets} tweets")
                break
            elif consecutive_existing >= max_consecutive_existing:
                print(f"  🛑 Stopped: Found {max_consecutive_existing} consecutive existing tweets")
                break
            
            # Scroll for more content
            random_scroll_pattern(page, deep_scroll=False)
            human_delay(1.0, 2.0)
        
        print(f"📊 Latest tweets collection complete: {saved_count} new tweets saved")
        
        # Update profile info
        fetcher_db.save_account_profile_info(conn, username, profile_pic_url)
        print(f"  💾 Updated profile info for @{username}")
        
        return tweets_collected
        
    finally:
        conn.close()


def fetch_tweets(page, username: str, max_tweets: int = 30, resume_from_last: bool = True) -> List[Dict]:
    """
    Tweet fetching with comprehensive post type detection and smart resume.
    For large collections (>800 tweets), automatically uses multi-session strategy.
    
    If resume_from_last is True, will fetch new tweets first, then continue from oldest timestamp.
    """
    print(f"\n🎯 Starting tweet collection for @{username} (target: {max_tweets} tweets)")
    
    # For very large collections, use multi-session approach to overcome Twitter's limits
    if max_tweets > 800:
        print(f"🔄 Large collection detected: Using multi-session strategy for {max_tweets} tweets")
        return fetch_tweets_in_sessions(page, username, max_tweets, session_size=800)
    
    all_collected_tweets = []
    
    # Initialize database connection
    conn = fetcher_db.init_db()
    
    try:
        # Get oldest tweet timestamp for continuing from where we left off
        oldest_timestamp = None
        newest_timestamp = None
        if resume_from_last:
            oldest_timestamp = fetcher_db.get_last_tweet_timestamp(username)
            if oldest_timestamp:
                # Also get the newest timestamp to detect new tweets
                try:
                    cur = conn.cursor()
                    cur.execute("SELECT tweet_timestamp FROM tweets WHERE username = ? ORDER BY tweet_timestamp ASC LIMIT 1", (username,))
                    row = cur.fetchone()
                    newest_timestamp = row[0] if row else None
                except Exception:
                    newest_timestamp = None
                
                print(f"📅 Existing tweet range: {newest_timestamp} (newest) to {oldest_timestamp} (oldest)")
                
                # PHASE 1: Fetch new tweets (from profile start)
                print(f"\n🔄 PHASE 1: Fetching new tweets (newer than {newest_timestamp})")
                url = f"https://x.com/{username}"
                print(f"🌐 Loading profile page: {url}")
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                
                # Wait for tweets to load
                try:
                    page.wait_for_selector('[data-testid="tweetText"], [data-testid="tweet"]', timeout=15000)
                except TimeoutError:
                    print(f"❌ No tweets found for @{username} or page failed to load")
                    return []

                human_delay(2.0, 4.0)
                
                # Extract profile picture
                print(f"🖼️  Extracting profile picture for @{username}...")
                profile_pic_url = fetcher_parsers.extract_profile_picture(page, username)
                
                # Collect new tweets (those newer than our newest timestamp)
                new_tweets = collect_tweets_from_page(page, username, max_tweets, False, newest_timestamp, profile_pic_url, conn)
                
                # Filter to only truly new tweets (newer than newest_timestamp)
                if newest_timestamp:
                    newest_time = datetime.fromisoformat(newest_timestamp.replace('Z', '+00:00'))
                    filtered_new_tweets = []
                    for tweet in new_tweets:
                        if tweet.get('tweet_timestamp'):
                            try:
                                tweet_time = datetime.fromisoformat(tweet['tweet_timestamp'].replace('Z', '+00:00'))
                                if tweet_time > newest_time:
                                    filtered_new_tweets.append(tweet)
                            except Exception:
                                # If timestamp parsing fails, include it to be safe
                                filtered_new_tweets.append(tweet)
                    new_tweets = filtered_new_tweets
                
                all_collected_tweets.extend(new_tweets)
                print(f"📈 Phase 1 complete: {len(new_tweets)} new tweets collected")
                
                # PHASE 2: Continue from oldest timestamp if we haven't reached max_tweets
                remaining_tweets = max_tweets - len(all_collected_tweets)
                if remaining_tweets > 0:
                    print(f"\n🔄 PHASE 2: Resuming from oldest timestamp ({remaining_tweets} tweets remaining)")
                    
                    if resume_positioning(page, username, oldest_timestamp):
                        # Wait for tweets to load after resume positioning
                        try:
                            page.wait_for_selector('[data-testid="tweetText"], [data-testid="tweet"]', timeout=15000)
                            human_delay(2.0, 4.0)
                        except TimeoutError:
                            print("⚠️ No tweets found after resume positioning")
                        
                        # Collect older tweets
                        older_tweets = collect_tweets_from_page(page, username, remaining_tweets, True, oldest_timestamp, profile_pic_url, conn)
                        all_collected_tweets.extend(older_tweets)
                        print(f"📈 Phase 2 complete: {len(older_tweets)} older tweets collected")
                    else:
                        print("❌ Resume positioning failed, skipping older tweets")
                else:
                    print(f"📊 Reached max_tweets limit with new tweets alone")
            else:
                print("🆕 No previous tweets found - scraping from beginning")
                # Load profile page for fresh start
                url = f"https://x.com/{username}"
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                
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
                
                # Collect tweets from the page (fresh start)
                all_collected_tweets = collect_tweets_from_page(page, username, max_tweets, False, None, profile_pic_url, conn)
        else:
            # Load profile page for non-resume fetch
            url = f"https://x.com/{username}"
            print(f"🌐 Loading profile page: {url}")
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            
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
            
            # Collect tweets from the page (non-resume)
            all_collected_tweets = collect_tweets_from_page(page, username, max_tweets, False, None, profile_pic_url, conn)
        
        print(f"\n📊 Collection complete: {len(all_collected_tweets)} tweets from @{username}")
        
        # Print summary by post type
        post_type_counts = {}
        for tweet in all_collected_tweets:
            post_type = tweet['post_type']
            post_type_counts[post_type] = post_type_counts.get(post_type, 0) + 1
        
        print("📈 Post type breakdown:")
        for post_type, count in sorted(post_type_counts.items()):
            print(f"    {post_type}: {count}")
        
        return all_collected_tweets
    
    finally:
        conn.close()

def run_fetch_session(p, handles: List[str], max_tweets: int, resume_from_last_flag: bool, latest: bool = False):
    # Use shared browser setup helper to consolidate browser configuration
    browser, context, page = _setup_browser_context(p, save_session=True)
    
    conn = fetcher_db.init_db()
    # Fetch tweets for each handle in a single browser session
    total_saved = 0
    for handle in handles:
        print(f"\nFetching up to {max_tweets} tweets for @{handle}...")
        # Add retries with exponential backoff for each handle
        max_retries = 5
        attempt = 0
        tweets = []
        while attempt <= max_retries:
            try:
                # Choose strategy based on latest flag
                if latest:
                    tweets = fetch_latest_tweets(page, handle, max_tweets=max_tweets)
                else:
                    tweets = fetch_tweets(page, handle, max_tweets=max_tweets, resume_from_last=resume_from_last_flag)
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
                    'fetch_tweets',
                    datetime.now().isoformat()
                ))
                conn_err.commit()
                conn_err.close()
            except Exception:
                pass
        
        # Tweets are already saved during fetch_tweets, so just count them
        # Save profile information if tweets were collected successfully
        if tweets and 'profile_pic_url' in tweets[0]:
            profile_pic_url = tweets[0]['profile_pic_url']
            fetcher_db.save_account_profile_info(conn, handle, profile_pic_url)
        
        # Count tweets that were processed (saved during collection)
        total_saved += len(tweets)
    
    conn.close()
    context.close()
    browser.close()
    return total_saved, len(handles)


def _setup_browser_context(p, save_session=False):
    """
    Helper function to set up browser context with consistent settings.
    Consolidates browser setup logic used across main fetch and refetch operations.
    
    Args:
        p: Playwright instance
        save_session: Whether to save session state (used in main fetch operations)
        
    Returns:
        tuple: (browser, context, page) instances
    """
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
    
    # Create browser context - reuse session for both main fetch and refetch
    if os.path.exists(session_file):
        context_kwargs["storage_state"] = session_file
    context = browser.new_context(**context_kwargs)
    page = context.new_page()
    
    # Save session state if requested (used in main fetch operations)
    if save_session:
        context.storage_state(path=session_file)
    
    return browser, context, page


def _get_tweet_info_from_db(tweet_id: str):
    """
    Helper function to get tweet information from database.
    
    Args:
        tweet_id: The tweet ID to look up
        
    Returns:
        tuple: (username, tweet_url) if found, (None, None) if not found
        
    Raises:
        Exception: If database error occurs
    """
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT username, tweet_url FROM tweets WHERE tweet_id = ?", (tweet_id,))
    row = cur.fetchone()
    conn.close()
    
    if not row:
        return None, None
    
    return row['username'], row['tweet_url']


def _extract_and_update_tweet(page, tweet_id: str, username: str, tweet_url: str) -> bool:
    """
    Helper function to extract tweet data and update database.
    Consolidates extraction logic used in refetch operations.
    
    Args:
        page: Playwright page instance
        tweet_id: The tweet ID
        username: The username 
        tweet_url: The tweet URL
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load the tweet page
        print(f"🌐 Loading tweet page...")
        try:
            page.goto(tweet_url, wait_until="domcontentloaded", timeout=60000)
            fetcher_parsers.human_delay(2.0, 3.0)
        except Exception as e:
            print(f"⚠️ Page load warning: {e}")
        
        # Extract tweet data
        tweet_data = fetcher_parsers.extract_tweet_with_quoted_content(page, tweet_id, username, tweet_url)
        
        if not tweet_data:
            print(f"❌ Failed to extract tweet data")
            return False
        
        # Update database
        success = fetcher_db.update_tweet_in_database(tweet_id, tweet_data)
        
        if success:
            print(f"✅ Successfully updated tweet {tweet_id}")
        else:
            print(f"❌ Failed to update tweet {tweet_id} in database")
            
        return success
        
    except Exception as e:
        print(f"❌ Error during tweet extraction: {e}")
        return False


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
        username, tweet_url = _get_tweet_info_from_db(tweet_id)
        
        if not username:
            print(f"❌ Tweet ID {tweet_id} not found in database. Cannot refetch.")
            return False
        
        print(f"📍 Found tweet from @{username}")
        print(f"🔗 URL: {tweet_url}")
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False
    
    # Use shared browser setup and extraction logic
    try:
        with sync_playwright() as p:
            browser, context, page = _setup_browser_context(p)
            
            # Extract and update tweet using shared helper
            success = _extract_and_update_tweet(page, tweet_id, username, tweet_url)
            
            context.close()
            browser.close()
            
            return success
            
    except Exception as e:
        print(f"❌ Error during refetch: {e}")
        import traceback
        traceback.print_exc()
        return False


def refetch_account_all(username: str, max_tweets: int = None) -> bool:
    """
    Delete all existing data for an account and refetch all tweets from scratch.
    
    Args:
        username: The username to refetch (without @)
        max_tweets: Maximum number of tweets to fetch (None for unlimited)
        
    Returns:
        bool: True if successful, False otherwise
    """
    username = username.lstrip('@').strip()
    print(f"🔄 REFETCH ALL MODE: Cleaning and refetching @{username}")
    
    try:
        # Delete existing data for the account
        deleted_counts = fetcher_db.delete_account_data(username)
        print(f"🗑️  Deleted {deleted_counts['tweets']} tweets and {deleted_counts['analyses']} analyses")
        
        # Handle unlimited case
        if max_tweets is None:
            max_tweets_display = "unlimited"
            max_tweets_param = float('inf')
        else:
            max_tweets_display = str(max_tweets)
            max_tweets_param = max_tweets
        
        # Fetch fresh data using the same pattern as main function
        print(f"🚀 Starting fresh fetch for @{username} (max: {max_tweets_display})")
        with sync_playwright() as p:
            total_fetched, accounts_processed = run_fetch_session(p, [username], max_tweets_param, False)
        
        if total_fetched > 0:
            print(f"✅ Successfully refetched {total_fetched} tweets for @{username}")
            return True
        else:
            print(f"❌ No tweets fetched for @{username}")
            return False
            
    except Exception as e:
        print(f"❌ Error during account refetch: {e}")
        return False


def main():
    start_time = time.time()  # Start timing

    parser = argparse.ArgumentParser(description="Fetch tweets from a given X (Twitter) user.")
    parser.add_argument("--user", "-u", dest="user", help="Optional single username to fetch tweets from (with or without leading @). Overrides positional username.")
    parser.add_argument("--max", type=int, default=None, help="Maximum number of tweets to fetch per user (default: unlimited)")
    parser.add_argument("--latest", action='store_true', help="Strategy 1: Fetch only latest tweets, stop after 10 consecutive existing tweets")
    parser.add_argument("--refetch", help="Re-fetch a specific tweet ID (bypasses exists check and updates database)")
    parser.add_argument("--refetch-all", help="Delete all data for specified username and refetch from scratch")
    args = parser.parse_args()

    # Handle refetch mode for specific tweet
    if args.refetch:
        tweet_id = args.refetch.strip()
        success = refetch_single_tweet(tweet_id)
        return  # Exit after refetch

    # Handle refetch-all mode for entire account
    if args.refetch_all:
        username = args.refetch_all.strip()
        success = refetch_account_all(username, args.max)
        return  # Exit after refetch-all

    max_tweets = args.max
    if max_tweets is None:
        max_tweets = float('inf')  # Unlimited
        print(f"📣 max_tweets set to: unlimited")
    else:
        print(f"📣 max_tweets set to: {max_tweets}")

    # Build handles list. If --user was provided, use single target, otherwise use defaults
    if args.user:
        username = args.user.strip().lstrip('@')  # Remove @ if present
        handles = [username]
    else:
        # No single user specified: use default handles
        handles = DEFAULT_HANDLES.copy()

    print(f"📣 Targets resolved (final): {handles}")

    with sync_playwright() as p:
        total, accounts_processed = run_fetch_session(p, handles, max_tweets, True, args.latest)
    
    # Calculate and display execution time
    end_time = time.time()
    execution_time = end_time - start_time
    minutes = int(execution_time // 60)
    seconds = int(execution_time % 60)
    
    print(f"\n⏱️  Execution completed in: {minutes}m {seconds}s")
    print(f"📊 Total tweets fetched and saved: {total}")
    print(f"🎯 Accounts processed: {accounts_processed}")
    print(f"📈 Average tweets per account: {total/accounts_processed:.1f}")



if __name__ == "__main__":
    main()
