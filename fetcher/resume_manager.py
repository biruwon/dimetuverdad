"""
Resume manager for Twitter fetching operations.

This module handles resume positioning and search-based navigation
for continuing tweet collection from where previous sessions left off.
"""

from datetime import datetime
from typing import Optional
from fetcher.scroller import Scroller


class ResumeManager:
    """Manages resume positioning and search-based navigation for tweet fetching."""
    
    def __init__(self):
        self.scroller = Scroller()
    
    def convert_timestamp_to_date_filter(self, timestamp: str) -> Optional[str]:
        """Convert ISO timestamp to Twitter search date format (YYYY-MM-DD)."""
        try:
            # Parse ISO timestamp and convert to date string
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d')
        except Exception as e:
            print(f"⚠️ Error converting timestamp {timestamp}: {e}")
            return None

    def try_resume_via_search(self, page, username: str, oldest_timestamp: str) -> bool:
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
            since_date = self.convert_timestamp_to_date_filter(oldest_timestamp)
            if not since_date:
                return False
            
            # Build search query to find tweets before our oldest timestamp
            # Use "until" parameter to find tweets before the date we already have
            search_query = f"from:{username} until:{since_date}"
            search_url = f"https://x.com/search?q={search_query}&src=typed_query&f=live"
            
            print(f"🔍 Trying search-based resume: {search_url}")
            page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
            self.scroller.delay(3.0, 5.0)
            
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

    def resume_positioning(self, page, username: str, oldest_timestamp: str) -> bool:
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
        if self.try_resume_via_search(page, username, oldest_timestamp):
            print("✅ Resume successful via search")
            return True
        
        # Fallback to regular profile load (will use normal scrolling)
        print("⚠️ Search resume failed, falling back to standard profile navigation")
        try:
            profile_url = f"https://x.com/{username}"
            page.goto(profile_url, wait_until="domcontentloaded", timeout=30000)
            self.scroller.delay(2.0, 4.0)
            print("✅ Loaded profile page for standard resume")
            return True
        except Exception as e:
            print(f"❌ All resume strategies failed: {e}")
            return False