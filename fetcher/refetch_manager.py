"""
Refetch manager for Twitter content operations.

This module handles re-fetching individual tweets or entire accounts,
including database operations and content extraction for updates.
"""

import time
import traceback
from typing import Optional, Tuple
from playwright.sync_api import sync_playwright

from fetcher import db as fetcher_db
from fetcher import parsers as fetcher_parsers
from fetcher.session_manager import SessionManager
from fetcher.media_monitor import MediaMonitor
from fetcher.scroller import Scroller

class RefetchManager:
    """Manages re-fetching operations for tweets and accounts."""
    
    def __init__(self):
        self.session_manager = SessionManager()
        self.media_monitor = MediaMonitor()
        self.scroller = Scroller()
    
    def get_tweet_info_from_db(self, tweet_id: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Helper function to get tweet information from database.
        
        Args:
            tweet_id: The tweet ID to look up
            
        Returns:
            tuple: (username, tweet_url) if found, (None, None) if not found
            
        Raises:
            Exception: If database error occurs
        """
        from database import get_db_connection_context
        with get_db_connection_context() as conn:
            cur = conn.cursor()
            cur.execute("SELECT username, tweet_url FROM tweets WHERE tweet_id = ?", (tweet_id,))
            row = cur.fetchone()
            
            if not row:
                return None, None
            
            return row['username'], row['tweet_url']

    def refetch_single_tweet(self, tweet_id: str, max_retries: int = 3) -> bool:
        """
        Re-fetch a specific tweet by ID, extracting complete content including quoted tweets.
        Includes retry logic for handling intermittent network errors and page load issues.
        
        Args:
            tweet_id: The tweet ID to refetch
            max_retries: Maximum number of retry attempts (default: 3)
            
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"🔄 REFETCH MODE: Re-fetching tweet ID {tweet_id}")
        
        # Get tweet info from database
        try:
            username, tweet_url = self.get_tweet_info_from_db(tweet_id)
            
            if not username:
                print(f"❌ Tweet ID {tweet_id} not found in database. Cannot refetch.")
                return False
            
            print(f"📍 Found tweet from @{username}")
            print(f"🔗 URL: {tweet_url}")
            
        except Exception as e:
            print(f"❌ Database error: {e}")
            return False
        
        # Retry logic with exponential backoff
        for attempt in range(1, max_retries + 1):
            try:
                with sync_playwright() as p:
                    browser, context, page = self.session_manager.create_browser_context(p)
                    
                    # Navigate to tweet page
                    try:
                        page.goto(tweet_url, wait_until="domcontentloaded", timeout=30000)
                        print(f"📄 Page loaded: {tweet_url}")
                        # Add delay to let dynamic content load
                        self.scroller.delay(2.0, 3.0)
                    except Exception as e:
                        print(f"⚠️ Page load warning: {e}")
                        if attempt < max_retries:
                            backoff = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
                            print(f"  🔄 Retry {attempt}/{max_retries} after {backoff}s...")
                            self.session_manager.cleanup_session(browser, context)
                            time.sleep(backoff)
                            continue
                        else:
                            self.session_manager.cleanup_session(browser, context)
                            return False
                    
                    # Extract tweet data using shared parsing logic (includes media monitoring)
                    tweet_data = fetcher_parsers.extract_tweet_with_media_monitoring(
                        page, tweet_id, username, tweet_url, self.media_monitor, self.scroller
                    )
                    
                    if not tweet_data:
                        print(f"❌ Failed to extract tweet data")
                        if attempt < max_retries:
                            backoff = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
                            print(f"  🔄 Retry {attempt}/{max_retries} after {backoff}s...")
                            self.session_manager.cleanup_session(browser, context)
                            time.sleep(backoff)
                            continue
                        else:
                            self.session_manager.cleanup_session(browser, context)
                            return False
                    
                    # Update database with DOM-extracted data
                    success = fetcher_db.update_tweet_in_database(tweet_id, tweet_data)
                    
                    if success:
                        print(f"✅ Successfully updated tweet {tweet_id}")
                    else:
                        print(f"❌ Failed to update tweet {tweet_id} in database")
                    
                    self.session_manager.cleanup_session(browser, context)
                    
                    return success
                    
            except Exception as e:
                print(f"❌ Error during refetch attempt {attempt}: {e}")
                if attempt < max_retries:
                    backoff = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
                    print(f"  🔄 Retry {attempt}/{max_retries} after {backoff}s...")
                    time.sleep(backoff)
                else:
                    traceback.print_exc()
                    return False
        
        return False

    def refetch_account_all(self, username: str, max_tweets: int = None) -> bool:
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
            
            # Import here to avoid circular imports
            from fetcher.fetch_tweets import run_fetch_session
            
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