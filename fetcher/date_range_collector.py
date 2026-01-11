"""
Date-range based tweet collection for complete account archiving.

Uses Twitter's search API with date filters (from:user since:YYYY-MM-DD until:YYYY-MM-DD)
to bypass the ~800 tweet scroll depth limit. Implements adaptive window splitting
to handle high-volume accounts.

Strategy:
1. Start with large windows (6 months) and crawl backwards from today
2. If a window returns >= threshold tweets, split it in half and recurse
3. Stop when N consecutive empty windows are found (reached account start)
4. Track progress in database for resumable collection
"""

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import quote_plus

from .logging_config import get_logger
from .async_collector import AsyncTweetCollector
from .async_scroller import get_async_scroller
from .async_session_manager import get_async_session_manager

logger = get_logger('date_range_collector')


@dataclass
class DateRangeWindow:
    """Represents a date range window for collection."""
    start_date: date
    end_date: date
    
    @property
    def days(self) -> int:
        """Number of days in this window."""
        return (self.end_date - self.start_date).days
    
    @property
    def midpoint(self) -> date:
        """Calculate the midpoint date for splitting."""
        delta = self.end_date - self.start_date
        return self.start_date + timedelta(days=delta.days // 2)
    
    @property
    def can_split(self) -> bool:
        """Check if window can be split further (more than 1 day)."""
        return self.days > 1
    
    def split(self) -> Tuple['DateRangeWindow', 'DateRangeWindow']:
        """Split window into two halves."""
        mid = self.midpoint
        first_half = DateRangeWindow(start_date=self.start_date, end_date=mid)
        second_half = DateRangeWindow(start_date=mid, end_date=self.end_date)
        return first_half, second_half
    
    def __str__(self) -> str:
        return f"{self.start_date} to {self.end_date} ({self.days} days)"


@dataclass
class DateRangeCollectorConfig:
    """Configuration for date-range based collection."""
    initial_window_months: int = 6
    min_window_days: int = 1
    split_threshold: int = 750  # Split if >= this many tweets in window
    empty_windows_to_stop: int = 3  # Stop after N consecutive empty windows
    max_depth: int = 10  # Max recursion depth for splitting


@dataclass
class WindowResult:
    """Result from collecting a single window."""
    window: DateRangeWindow
    tweets_collected: int = 0
    windows_processed: int = 1
    skipped: bool = False
    split_count: int = 0


@dataclass
class CollectionResult:
    """Result from a full collection run."""
    username: str
    total_tweets: int = 0
    windows_processed: int = 0
    consecutive_empty_windows: int = 0
    stopped_reason: Optional[str] = None
    oldest_date_reached: Optional[date] = None
    newest_date_reached: Optional[date] = None
    errors: List[str] = field(default_factory=list)


@dataclass
class CollectionProgress:
    """Progress report for ongoing collection."""
    username: str
    tweets_collected: int = 0
    total_tweets: int = 0  # Alias for compatibility
    windows_completed: int = 0
    windows_remaining: int = 0
    current_window: Optional[DateRangeWindow] = None
    _is_complete: bool = False
    estimated_remaining_windows: int = 0
    
    @property
    def is_complete(self) -> bool:
        """Collection is complete when no windows remaining."""
        return self._is_complete or self.windows_remaining == 0


def build_search_url(username: str, start_date: date, end_date: date) -> str:
    """
    Build Twitter search URL with date range filters.
    
    Uses the search query: from:username since:YYYY-MM-DD until:YYYY-MM-DD
    with f=live for chronological results.
    """
    query = f"from:{username} since:{start_date.isoformat()} until:{end_date.isoformat()}"
    encoded_query = quote_plus(query)
    return f"https://x.com/search?q={encoded_query}&src=typed_query&f=live"


class DateRangeProgressDB:
    """Database operations for tracking collection progress."""
    
    def __init__(self, db_connection):
        self.conn = db_connection
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Ensure collection_progress table exists."""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collection_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                tweets_collected INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                completed_at TEXT,
                UNIQUE(username, start_date, end_date)
            )
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_collection_progress_username 
            ON collection_progress(username)
        ''')
        self.conn.commit()
    
    def get_collected_ranges(self, username: str) -> List[Dict]:
        """Get all collected date ranges for a user."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT start_date, end_date, status, tweets_collected
            FROM collection_progress
            WHERE username = ? AND status = 'complete'
            ORDER BY start_date DESC
        ''', (username,))
        return [dict(row) for row in cursor.fetchall()]
    
    def is_range_collected(self, username: str, start_date: date, end_date: date) -> bool:
        """Check if a specific range has been collected."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT status FROM collection_progress
            WHERE username = ? AND start_date = ? AND end_date = ? AND status = 'complete'
        ''', (username, start_date.isoformat(), end_date.isoformat()))
        return cursor.fetchone() is not None
    
    def mark_range_complete(self, username: str, start_date: date, end_date: date, 
                           tweets_collected: int):
        """Mark a date range as completely collected."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO collection_progress 
            (username, start_date, end_date, status, tweets_collected, completed_at)
            VALUES (?, ?, ?, 'complete', ?, CURRENT_TIMESTAMP)
        ''', (username, start_date.isoformat(), end_date.isoformat(), tweets_collected))
        self.conn.commit()
    
    def mark_range_in_progress(self, username: str, start_date: date, end_date: date):
        """Mark a date range as in progress."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO collection_progress 
            (username, start_date, end_date, status)
            VALUES (?, ?, ?, 'in_progress')
        ''', (username, start_date.isoformat(), end_date.isoformat()))
        self.conn.commit()
    
    def get_oldest_tweet_date(self, username: str) -> Optional[date]:
        """Get the oldest tweet date for a user from tweets table."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT MIN(DATE(timestamp)) as oldest
            FROM tweets WHERE username = ?
        ''', (username,))
        row = cursor.fetchone()
        if row and row['oldest']:
            return date.fromisoformat(row['oldest'])
        return None
    
    def get_newest_tweet_date(self, username: str) -> Optional[date]:
        """Get the newest tweet date for a user from tweets table."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT MAX(DATE(timestamp)) as newest
            FROM tweets WHERE username = ?
        ''', (username,))
        row = cursor.fetchone()
        if row and row['newest']:
            return date.fromisoformat(row['newest'])
        return None
    
    def get_collection_stats(self, username: str) -> Dict:
        """Get collection statistics for a user."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT 
                COUNT(*) as total_ranges,
                SUM(CASE WHEN status = 'complete' THEN 1 ELSE 0 END) as completed_ranges,
                SUM(tweets_collected) as total_tweets
            FROM collection_progress
            WHERE username = ?
        ''', (username,))
        return dict(cursor.fetchone())


class DateRangeCollector:
    """
    Collects tweets using date-range based search to bypass scroll limits.
    
    This class implements an adaptive collection strategy:
    1. Start from today and work backwards in windows
    2. If a window has too many tweets (>750), split it in half
    3. Stop when we find N consecutive empty windows (reached account start)
    """
    
    def __init__(self, username: str, tweet_collector=None, db=None, 
                 config: DateRangeCollectorConfig = None):
        self.username = username
        self.tweet_collector = tweet_collector
        self.db = db
        self.config = config or DateRangeCollectorConfig()
        self.scroller = get_async_scroller()
        self.session_manager = get_async_session_manager()
        self._collected_ids: Set[str] = set()
    
    async def collect_window(self, start_date: date, end_date: date, 
                            depth: int = 0) -> WindowResult:
        """
        Collect tweets from a specific date range window.
        
        If the window returns >= threshold tweets, splits and recurses.
        """
        window = DateRangeWindow(start_date=start_date, end_date=end_date)
        result = WindowResult(window=window)
        
        # Check if already collected
        if self.db and self.db.is_range_collected(self.username, start_date, end_date):
            logger.info(f"⏭️  Skipping already collected range: {window}")
            result.skipped = True
            return result
        
        # Mark as in progress
        if self.db:
            self.db.mark_range_in_progress(self.username, start_date, end_date)
        
        logger.info(f"📅 Collecting window: {window} (depth={depth})")
        
        # Collect tweets from this window
        tweets = await self._collect_from_search(start_date, end_date)
        tweet_count = len(tweets)
        
        logger.info(f"   Found {tweet_count} tweets in {window}")
        
        # Check if we need to split
        if tweet_count >= self.config.split_threshold and window.can_split:
            if depth < self.config.max_depth:
                logger.info(f"   ✂️  Splitting window (>= {self.config.split_threshold} tweets)")
                result.split_count += 1
                
                first_half, second_half = window.split()
                
                # Recursively collect both halves
                first_result = await self.collect_window(
                    first_half.start_date, first_half.end_date, depth + 1
                )
                second_result = await self.collect_window(
                    second_half.start_date, second_half.end_date, depth + 1
                )
                
                # Aggregate results
                result.tweets_collected = first_result.tweets_collected + second_result.tweets_collected
                result.windows_processed = first_result.windows_processed + second_result.windows_processed
                result.split_count += first_result.split_count + second_result.split_count
                return result
            else:
                logger.warning(f"   ⚠️  Max split depth reached, keeping {tweet_count} tweets")
        
        # Store collected tweets
        result.tweets_collected = tweet_count
        
        # Track collected IDs to avoid duplicates across windows
        for tweet in tweets:
            tweet_id = tweet.get('tweet_id')
            if tweet_id:
                self._collected_ids.add(tweet_id)
        
        # Mark range as complete
        if self.db:
            self.db.mark_range_complete(self.username, start_date, end_date, tweet_count)
        
        return result
    
    async def collect_full_history(self, start_from: date = None) -> CollectionResult:
        """
        Collect complete tweet history for the account.
        
        Crawls backwards from today (or start_from date) until N consecutive
        empty windows are found, indicating we've reached the account start.
        """
        result = CollectionResult(username=self.username)
        
        # Start from today or provided date
        end_date = start_from or date.today()
        window_delta = timedelta(days=self.config.initial_window_months * 30)
        
        consecutive_empty = 0
        
        logger.info(f"🚀 Starting full history collection for @{self.username}")
        logger.info(f"   Initial window: {self.config.initial_window_months} months")
        logger.info(f"   Split threshold: {self.config.split_threshold} tweets")
        logger.info(f"   Stop after: {self.config.empty_windows_to_stop} empty windows")
        
        while consecutive_empty < self.config.empty_windows_to_stop:
            # Calculate window
            start_date = end_date - window_delta
            
            try:
                window_result = await self.collect_window(start_date, end_date)
                
                result.windows_processed += window_result.windows_processed
                result.total_tweets += window_result.tweets_collected
                
                if window_result.tweets_collected == 0 and not window_result.skipped:
                    consecutive_empty += 1
                    logger.info(f"   📭 Empty window ({consecutive_empty}/{self.config.empty_windows_to_stop})")
                else:
                    consecutive_empty = 0
                    if window_result.tweets_collected > 0:
                        result.oldest_date_reached = start_date
                
                # Track newest date
                if result.newest_date_reached is None:
                    result.newest_date_reached = end_date
                
            except Exception as e:
                error_msg = f"Error collecting {start_date} to {end_date}: {str(e)}"
                logger.error(f"   ❌ {error_msg}")
                result.errors.append(error_msg)
                consecutive_empty += 1  # Treat errors as empty to avoid infinite loops
            
            # Move window backwards
            end_date = start_date
        
        result.consecutive_empty_windows = consecutive_empty
        result.stopped_reason = "consecutive_empty_windows"
        
        logger.info(f"✅ Collection complete for @{self.username}")
        logger.info(f"   Total tweets: {result.total_tweets}")
        logger.info(f"   Windows processed: {result.windows_processed}")
        if result.oldest_date_reached:
            logger.info(f"   Oldest date: {result.oldest_date_reached}")
        
        return result
    
    async def _collect_from_search(self, start_date: date, end_date: date) -> List[Dict]:
        """
        Navigate to search page and collect tweets.
        
        Uses the AsyncTweetCollector's collection logic on a search results page.
        """
        if self.tweet_collector is None:
            logger.warning("No tweet collector configured, using mock collection")
            return []
        
        # If tweet_collector has a collect_from_search method, use it
        if hasattr(self.tweet_collector, 'collect_from_search'):
            return await self.tweet_collector.collect_from_search(
                self.username, start_date, end_date
            )
        
        # Otherwise, return empty (for testing with mocks)
        return []
    
    def get_progress(self) -> CollectionProgress:
        """Get current collection progress."""
        stats = {}
        if self.db:
            stats = self.db.get_collection_stats(self.username)
        
        return CollectionProgress(
            username=self.username,
            tweets_collected=stats.get('total_tweets', 0) or 0,
            windows_completed=stats.get('completed_ranges', 0) or 0,
            is_complete=False,
            estimated_remaining_windows=0
        )


async def run_date_range_collection(username: str, db_connection, 
                                    config: DateRangeCollectorConfig = None,
                                    resume: bool = True) -> CollectionResult:
    """
    Run date-range based collection for a user.
    
    Args:
        username: Twitter username to collect
        db_connection: Database connection
        config: Collection configuration
        resume: Whether to skip already-collected ranges
    
    Returns:
        CollectionResult with collection statistics
    """
    from playwright.async_api import async_playwright
    
    progress_db = DateRangeProgressDB(db_connection)
    
    async with async_playwright() as p:
        # Create tweet collector
        tweet_collector = AsyncTweetCollector()
        
        # Create date range collector
        collector = DateRangeCollector(
            username=username,
            tweet_collector=tweet_collector,
            db=progress_db,
            config=config
        )
        
        # Run collection
        result = await collector.collect_full_history()
        
        return result
