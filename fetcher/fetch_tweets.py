"""
Main entry point for tweet fetching operations.

This module coordinates the fetching process by using specialized components:
- TweetCollector: Core tweet collection logic
- SessionManager: Browser session management and login
- ResumeManager: Resume positioning and search navigation
- RefetchManager: Individual tweet and account refetching
- Scroller: Scrolling and navigation
- MediaMonitor: Network request monitoring
- Parsers: Content extraction
- DB: Database operations
"""

import argparse
import random
import sqlite3
import sys
import os
import time
import traceback
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports when run as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fetcher import db as fetcher_db
from fetcher import parsers as fetcher_parsers
from fetcher.config import get_config, DEFAULT_HANDLES
from fetcher.logging_config import setup_logging
from fetcher.session_manager import SessionManager
from fetcher.scroller import Scroller
from fetcher.collector import TweetCollector
from fetcher.media_monitor import MediaMonitor
from fetcher.resume_manager import ResumeManager
from fetcher.refetch_manager import RefetchManager
from utils import paths
from database.repositories import get_tweet_repository
from playwright.sync_api import sync_playwright, TimeoutError
from datetime import datetime
from database import get_db_connection_context

# Import performance tracking utility
from utils.performance import start_tracking, stop_tracking, print_performance_summary

# Load configuration
config = get_config()

# Initialize components
session_manager = SessionManager()
scroller = Scroller()
media_monitor = MediaMonitor()
collector = TweetCollector()
resume_manager = ResumeManager()
refetch_manager = RefetchManager()

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
            result = collector.collect_tweets_from_page(page, username, max_tweets, True, None, None, conn)
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
            
            scroller.delay(2.0, 4.0)
            
            # Get the oldest timestamp from our current collection to resume properly
            oldest_timestamp = None
            if all_tweets:
                # Find the oldest timestamp from our collected tweets
                timestamps = [t.get('tweet_timestamp') for t in all_tweets if t.get('tweet_timestamp')]
                if timestamps:
                    oldest_timestamp = min(timestamps)
                    print(f"   📅 Oldest collected timestamp: {oldest_timestamp}")
                    # Note: For multi-session, we don't use resume positioning to avoid navigation issues
            
            # Extract profile picture (once per multi-session)
            if sessions_completed == 1:
                print(f"🖼️  Extracting profile picture for @{username}...")
                profile_pic_url = fetcher_parsers.extract_profile_picture(page, username)
            else:
                profile_pic_url = None  # Use cached value from session 1
            
            # Collect tweets for this session
            session_results = collector.collect_tweets_from_page(page, username, session_tweets, False, None, profile_pic_url, conn)
            
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
                    scroller.delay(3.0, 6.0)  # Longer delay between sessions
                except Exception as e:
                    print(f"   ⚠️ Cache clearing failed: {e}")
        
        print(f"\n🏁 Multi-session complete: {len(all_tweets)} tweets collected in {sessions_completed} sessions")
        return all_tweets
    
    finally:
        conn.close()


def fetch_latest_tweets(page, username: str, max_tweets: int = 30) -> List[Dict]:
    """
    Strategy 1: Fetch latest tweets with proper stopping logic.
    Stops after finding 10 consecutive existing tweets in a row (latest mode).

    Uses the same extraction method as refetch/collector for consistent media handling,
    but implements latest-mode logic at the application level.

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

    scroller.delay(2.0, 4.0)

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
        max_consecutive_existing = config.max_consecutive_existing_tweets  # Use config value (10)

        print(f"📊 Latest mode: will stop after {max_consecutive_existing} consecutive existing tweets")

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
                    if not href:
                        continue

                    # Parse author and tweet_id from URL using shared utility
                    should_process, actual_author, tweet_id = fetcher_parsers.should_process_tweet_by_author(href, username)

                    if not should_process:
                        continue

                    # Check if we already processed this tweet in this session
                    if any(t.get('tweet_id') == tweet_id for t in tweets_collected):
                        continue

                    # Check if tweet exists in database (for latest mode stopping logic)
                    exists_in_db = fetcher_db.check_if_tweet_exists(username, tweet_id)
                    if exists_in_db:
                        print(f"  ⏭️ Existing tweet: {tweet_id}")
                        consecutive_existing += 1
                        if consecutive_existing >= max_consecutive_existing:
                            print(f"  🛑 Stopped: Found {max_consecutive_existing} consecutive existing tweets (latest mode)")
                            return tweets_collected
                        continue  # Skip existing tweets

                    # Reset consecutive counter when we find a new tweet
                    consecutive_existing = 0

                    tweet_url = f"https://x.com{href}"

                    # Use the same sophisticated extraction as refetch/collector to prevent media cross-contamination
                    tweet_data = fetcher_parsers.extract_tweet_with_media_monitoring(
                        page, tweet_id, actual_author, tweet_url, media_monitor, scroller
                    )

                    if not tweet_data:
                        continue

                    # Add profile picture
                    tweet_data['profile_pic_url'] = profile_pic_url

                    # Save tweet immediately (no batching needed for latest mode)
                    saved = fetcher_db.save_tweet(conn, tweet_data)
                    if saved:
                        saved_count += 1
                        tweets_saved_this_scroll += 1
                        tweets_collected.append(tweet_data)
                        print(f"  💾 Saved [{saved_count}] {tweet_data.get('post_type', 'unknown')}: {tweet_id}")
                    else:
                        print(f"  ⏭️ Failed to save: {tweet_id}")

                except Exception as e:
                    print(f"  ❌ Error processing tweet: {e}")
                    continue

            # Check if we should stop scrolling
            if saved_count >= max_tweets:
                print(f"  ✅ Completed: Reached target of {max_tweets} tweets")
                break

            # If no new tweets were saved this scroll, increment consecutive counter
            if tweets_saved_this_scroll == 0:
                consecutive_existing += 1
                print(f"  📊 No new tweets saved this scroll ({consecutive_existing}/{max_consecutive_existing})")
                if consecutive_existing >= max_consecutive_existing:
                    print(f"  🛑 Stopped: Found {max_consecutive_existing} consecutive scrolls with no new tweets (latest mode)")
                    break
            else:
                print(f"  ✅ Saved {tweets_saved_this_scroll} new tweets this scroll")

            # Scroll for more content if we haven't reached our limits
            if saved_count < max_tweets and consecutive_existing < max_consecutive_existing:
                scroller.random_scroll_pattern(page, deep_scroll=False)
                scroller.delay(1.0, 2.0)

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
    try:
        conn = fetcher_db.init_db()
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return []
    
    try:
        # Get oldest tweet timestamp for continuing from where we left off
        oldest_timestamp = None
        newest_timestamp = None
        if resume_from_last:
            # Use repository to get tweet timestamps
            tweet_repo = get_tweet_repository()
            
            # Get tweets for this user to find timestamps
            user_tweets = tweet_repo.get_tweets_by_username(username, limit=1000)  # Get enough to find min/max
            
            if user_tweets:
                # Extract timestamps
                timestamps = []
                for tweet in user_tweets:
                    if tweet.get('tweet_timestamp'):
                        timestamps.append(tweet['tweet_timestamp'])
                
                if timestamps:
                    oldest_timestamp = min(timestamps)
                    newest_timestamp = max(timestamps)
                    
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

                scroller.delay(2.0, 4.0)
                
                # Extract profile picture
                print(f"🖼️  Extracting profile picture for @{username}...")
                profile_pic_url = fetcher_parsers.extract_profile_picture(page, username)
                
                # Collect new tweets (those newer than our newest timestamp)
                new_tweets = collector.collect_tweets_from_page(page, username, max_tweets, False, newest_timestamp, profile_pic_url, conn)
                
                # Filter to only truly new tweets (newer than newest_timestamp)
                if newest_timestamp:
                    newest_time = datetime.fromisoformat(newest_timestamp.replace('Z', '+00:00'))
                    filtered_new_tweets = []
                    for tweet in new_tweets:
                        if tweet.get('tweet_timestamp'):
                            tweet_time = datetime.fromisoformat(tweet['tweet_timestamp'].replace('Z', '+00:00'))
                            if tweet_time > newest_time:
                                filtered_new_tweets.append(tweet)
                    new_tweets = filtered_new_tweets
                
                all_collected_tweets.extend(new_tweets)
                print(f"📈 Phase 1 complete: {len(new_tweets)} new tweets collected")
                
                # PHASE 2: Continue from oldest timestamp if we haven't reached max_tweets
                remaining_tweets = max_tweets - len(all_collected_tweets)
                if remaining_tweets > 0:
                    print(f"\n🔄 PHASE 2: Resuming from oldest timestamp ({remaining_tweets} tweets remaining)")
                    
                    if resume_manager.resume_positioning(page, username, oldest_timestamp):
                        # Wait for tweets to load after resume positioning
                        try:
                            page.wait_for_selector('[data-testid="tweetText"], [data-testid="tweet"]', timeout=15000)
                            scroller.delay(2.0, 4.0)
                        except TimeoutError:
                            print("⚠️ No tweets found after resume positioning")
                        
                        # Collect older tweets
                        older_tweets = collector.collect_tweets_from_page(page, username, remaining_tweets, True, oldest_timestamp, profile_pic_url, conn)
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

                scroller.delay(2.0, 4.0)
                
                # Extract profile picture before starting tweet collection
                print(f"🖼️  Extracting profile picture for @{username}...")
                profile_pic_url = fetcher_parsers.extract_profile_picture(page, username)
                
                # Collect tweets from the page (fresh start)
                all_collected_tweets = collector.collect_tweets_from_page(page, username, max_tweets, False, None, profile_pic_url, conn)
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

            scroller.delay(2.0, 4.0)
            
            # Extract profile picture before starting tweet collection
            print(f"🖼️  Extracting profile picture for @{username}...")
            profile_pic_url = fetcher_parsers.extract_profile_picture(page, username)
            
            # Collect tweets from the page (non-resume)
            all_collected_tweets = collector.collect_tweets_from_page(page, username, max_tweets, False, None, profile_pic_url, conn)
        
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
    browser, context, page = SessionManager().create_browser_context(p, save_session=True)
    
    try:
        conn = fetcher_db.init_db()
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        SessionManager().cleanup_session(browser, context)
        return 0, 0
    # Fetch tweets for each handle in a single browser session
    total_saved = 0
    for handle in handles:
        print(f"\nFetching up to {max_tweets} tweets for @{handle}...")
        # Add retries with exponential backoff for each handle
        max_retries = 0
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
                with get_db_connection_context() as conn_err:
                    cur_err = conn_err.cursor()
                    cur_err.execute("INSERT INTO scrape_errors (username, tweet_id, error, context, timestamp) VALUES (?, ?, ?, ?, ?)", (
                        handle,
                        None,
                        'max_retries_exceeded',
                        'fetch_tweets',
                        datetime.now().isoformat()
                    ))
                    conn_err.commit()
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
    SessionManager().cleanup_session(browser, context)
    return total_saved, len(handles)


def main():
    # Setup logging and get configuration
    setup_logging()

    # Start performance tracking
    tracker = start_tracking("Tweet Fetcher")

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
        success = refetch_manager.refetch_single_tweet(tweet_id)
        return  # Exit after refetch

    # Handle refetch-all mode for entire account
    if args.refetch_all:
        username = args.refetch_all.strip()
        success = refetch_manager.refetch_account_all(username, args.max)
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
    
    # Increment performance counter with total tweets processed
    tracker.increment_operations(total)
    
    # Calculate and display execution time
    end_time = time.time()
    execution_time = end_time - start_time
    minutes = int(execution_time // 60)
    seconds = int(execution_time % 60)
    
    print(f"\n⏱️  Execution completed in: {minutes}m {seconds}s")
    print(f"📊 Total tweets fetched and saved: {total}")
    print(f"🎯 Accounts processed: {accounts_processed}")
    print(f"📈 Average tweets per account: {total/accounts_processed:.1f}")

    # Print performance summary
    metrics = stop_tracking(tracker)
    print_performance_summary(metrics)



if __name__ == "__main__":
    main()
