import sqlite3
import time
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

# Import path utilities for consistent database path resolution
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from utils import paths
from database.repositories import get_tweet_repository

DB_PATH = str(paths.get_db_path())


@dataclass
class BatchWriteStats:
    """Track batch write performance statistics."""
    total_tweets: int = 0
    batches_written: int = 0
    total_time: float = 0.0
    errors: int = 0
    
    @property
    def avg_batch_time(self) -> float:
        if self.batches_written == 0:
            return 0.0
        return self.total_time / self.batches_written


class TweetBuffer:
    """
    Buffer for batch database writes (P3 Performance Optimization).
    
    Buffers tweet data and writes in batches to reduce transaction overhead.
    This can significantly improve write performance when collecting many tweets.
    
    Usage:
        buffer = TweetBuffer(conn, batch_size=50)
        for tweet in tweets:
            buffer.add(tweet)
        buffer.flush()  # Don't forget to flush remaining tweets
    """
    
    def __init__(self, conn: sqlite3.Connection, batch_size: int = 50):
        """
        Initialize the tweet buffer.
        
        Args:
            conn: SQLite database connection
            batch_size: Number of tweets to buffer before writing
        """
        self.conn = conn
        self.batch_size = batch_size
        self.buffer: List[Dict] = []
        self.stats = BatchWriteStats()
        self._insert_columns = [
            'tweet_id', 'content', 'username', 'tweet_url', 'tweet_timestamp',
            'post_type', 'original_author', 'original_tweet_id', 'reply_to_username',
            'media_links', 'media_count', 'engagement_likes', 'engagement_retweets',
            'engagement_replies', 'external_links', 'original_content', 'is_pinned',
            'reply_to_tweet_id', 'conversation_id', 'thread_id', 'thread_position',
            'is_thread_start'
        ]
    
    def add(self, tweet_data: Dict) -> bool:
        """
        Add a tweet to the buffer.
        
        Args:
            tweet_data: Tweet data dictionary
            
        Returns:
            bool: True if tweet was added (not a duplicate)
        """
        tweet_id = tweet_data.get('tweet_id')
        if not tweet_id or str(tweet_id).lower() == 'analytics':
            return False
        
        # Skip if already in buffer
        if any(t.get('tweet_id') == tweet_id for t in self.buffer):
            return False
        
        self.buffer.append(tweet_data)
        
        if len(self.buffer) >= self.batch_size:
            self.flush()
        
        return True
    
    def flush(self) -> int:
        """
        Write all buffered tweets to the database.
        
        Returns:
            int: Number of tweets written
        """
        if not self.buffer:
            return 0
        
        start_time = time.time()
        tweets_written = 0
        
        try:
            cursor = self.conn.cursor()
            
            # Prepare values for batch insert
            placeholders = ', '.join(['?'] * len(self._insert_columns))
            columns = ', '.join(self._insert_columns)
            
            for tweet_data in self.buffer:
                try:
                    # Check if exists for update logic
                    cursor.execute(
                        "SELECT id FROM tweets WHERE tweet_id = ?",
                        (tweet_data['tweet_id'],)
                    )
                    existing = cursor.fetchone()
                    
                    if existing:
                        # Update existing tweet
                        self._update_tweet(cursor, tweet_data)
                    else:
                        # Insert new tweet
                        values = tuple(
                            tweet_data.get(col, 0 if col in ('media_count', 'engagement_likes', 
                                'engagement_retweets', 'engagement_replies', 'is_pinned', 
                                'is_thread_start') else None)
                            for col in self._insert_columns
                        )
                        cursor.execute(
                            f"INSERT INTO tweets ({columns}) VALUES ({placeholders})",
                            values
                        )
                    tweets_written += 1
                    
                except Exception as e:
                    self.stats.errors += 1
                    # Continue with other tweets
                    continue
            
            # Single commit for the entire batch
            self.conn.commit()
            
            elapsed = time.time() - start_time
            self.stats.total_tweets += tweets_written
            self.stats.batches_written += 1
            self.stats.total_time += elapsed
            
        except Exception as e:
            self.stats.errors += 1
        finally:
            self.buffer = []
        
        return tweets_written
    
    def _update_tweet(self, cursor, tweet_data: Dict) -> None:
        """Update an existing tweet with new data."""
        cursor.execute("""
            UPDATE tweets SET 
                content = ?,
                post_type = ?,
                original_author = ?,
                original_tweet_id = ?,
                media_links = ?,
                media_count = ?,
                engagement_likes = ?,
                engagement_retweets = ?,
                engagement_replies = ?,
                external_links = ?,
                original_content = ?,
                is_pinned = ?,
                reply_to_tweet_id = COALESCE(?, reply_to_tweet_id),
                conversation_id = COALESCE(?, conversation_id),
                thread_id = COALESCE(?, thread_id),
                thread_position = COALESCE(?, thread_position),
                is_thread_start = COALESCE(?, is_thread_start)
            WHERE tweet_id = ?
        """, (
            tweet_data.get('content'),
            tweet_data.get('post_type', 'original'),
            tweet_data.get('original_author'),
            tweet_data.get('original_tweet_id'),
            tweet_data.get('media_links'),
            tweet_data.get('media_count', 0),
            tweet_data.get('engagement_likes', 0),
            tweet_data.get('engagement_retweets', 0),
            tweet_data.get('engagement_replies', 0),
            tweet_data.get('external_links'),
            tweet_data.get('original_content'),
            tweet_data.get('is_pinned', 0),
            tweet_data.get('reply_to_tweet_id'),
            tweet_data.get('conversation_id'),
            tweet_data.get('thread_id'),
            tweet_data.get('thread_position'),
            tweet_data.get('is_thread_start'),
            tweet_data['tweet_id']
        ))
    
    def get_stats(self) -> BatchWriteStats:
        """Get batch write statistics."""
        return self.stats
    
    def __len__(self) -> int:
        """Return number of tweets currently in buffer."""
        return len(self.buffer)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - flush remaining tweets."""
        self.flush()
        return False

def delete_account_data(username: str) -> Dict[str, int]:
    """
    Delete all data for a specific account from both tweets and content_analyses tables.
    
    Args:
        username: The username to delete data for
        
    Returns:
        Dict with counts of deleted records: {'tweets': count, 'analyses': count}
    """
    try:
        # Use standardized repositories
        tweet_repo = get_tweet_repository()
        
        # For analyses count, we need to use direct access since content analysis repo might not have username filtering
        # This is a specialized operation that may need to stay direct for now
        from database import get_db_connection_context
        with get_db_connection_context() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) AS analyses_count FROM content_analyses WHERE author_username = ?", (username,))
            row = cur.fetchone()
            analyses_count = row['analyses_count'] if row else 0

        # Delete tweets using direct access (for now, since repository doesn't have delete method)
        from database import get_db_connection_context
        with get_db_connection_context() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM tweets WHERE username = ?", (username,))
            tweets_before = cur.fetchone()[0]
            cur.execute("DELETE FROM tweets WHERE username = ?", (username,))
            deleted_tweets = cur.rowcount
            conn.commit()

        # Delete analyses using direct access (for now)
        from database import get_db_connection_context
        with get_db_connection_context() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM content_analyses WHERE author_username = ?", (username,))
            conn.commit()

        print(f"✅ Deleted {deleted_tweets} tweets and {analyses_count} analyses for @{username}")

        return {'tweets': deleted_tweets, 'analyses': analyses_count}

    except Exception as e:
        print(f"❌ Error deleting data for @{username}: {e}")
        raise

def save_tweet(conn: sqlite3.Connection, tweet_data: Dict) -> bool:
    """Simplified save/update function for tests and main logic.
    
    Supports thread-related fields: reply_to_tweet_id, conversation_id,
    thread_id, thread_position, is_thread_start.
    """
    c = conn.cursor()
    try:
        if not tweet_data.get('tweet_id') or str(tweet_data.get('tweet_id')).lower() == 'analytics':
            return False
        
        c.execute("SELECT id, post_type, content, original_author, original_tweet_id FROM tweets WHERE tweet_id = ?", (tweet_data['tweet_id'],))
        existing = c.fetchone()
        if existing:
            existing_post_type = existing['post_type']
            existing_content = existing['content']
            existing_original_author = existing['original_author']
            existing_original_tweet_id = existing['original_tweet_id']
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
            # Also update if thread metadata is being set
            elif tweet_data.get('thread_id') or tweet_data.get('conversation_id'):
                needs_update = True
            if not needs_update:
                return False
            # perform update (minimal fields + thread fields)
            c.execute("""UPDATE tweets SET content = ?, post_type = ?, original_author = ?, original_tweet_id = ?,
                        media_links = ?, media_count = ?,
                        engagement_likes = ?, engagement_retweets = ?, engagement_replies = ?,
                        external_links = ?, original_content = ?, is_pinned = ?,
                        reply_to_tweet_id = COALESCE(?, reply_to_tweet_id),
                        conversation_id = COALESCE(?, conversation_id),
                        thread_id = COALESCE(?, thread_id),
                        thread_position = COALESCE(?, thread_position),
                        is_thread_start = COALESCE(?, is_thread_start)
                        WHERE tweet_id = ?""", (
                tweet_data.get('content'),
                new_post_type,
                tweet_data.get('original_author'),
                tweet_data.get('original_tweet_id'),
                tweet_data.get('media_links'),
                tweet_data.get('media_count', 0),
                tweet_data.get('engagement_likes', 0),
                tweet_data.get('engagement_retweets', 0),
                tweet_data.get('engagement_replies', 0),
                tweet_data.get('external_links'),
                tweet_data.get('original_content'),
                tweet_data.get('is_pinned', 0),
                tweet_data.get('reply_to_tweet_id'),
                tweet_data.get('conversation_id'),
                tweet_data.get('thread_id'),
                tweet_data.get('thread_position'),
                tweet_data.get('is_thread_start'),
                tweet_data['tweet_id']
            ))
            conn.commit()
            return True
        # insert
        c.execute("""INSERT INTO tweets (tweet_id, content, username, tweet_url, tweet_timestamp, post_type,
                        original_author, original_tweet_id, reply_to_username,
                        media_links, media_count,
                        engagement_likes, engagement_retweets, engagement_replies,
                        external_links, original_content, is_pinned,
                        reply_to_tweet_id, conversation_id, thread_id, thread_position, is_thread_start)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
            tweet_data['tweet_id'],
            tweet_data.get('content'),
            tweet_data.get('username'),
            tweet_data.get('tweet_url'),
            tweet_data.get('tweet_timestamp'),
            tweet_data.get('post_type', 'original'),
            tweet_data.get('original_author'),
            tweet_data.get('original_tweet_id'),
            tweet_data.get('reply_to_username'),
            tweet_data.get('media_links'),
            tweet_data.get('media_count', 0),
            tweet_data.get('engagement_likes', 0),
            tweet_data.get('engagement_retweets', 0),
            tweet_data.get('engagement_replies', 0),
            tweet_data.get('external_links'),
            tweet_data.get('original_content'),
            tweet_data.get('is_pinned', 0),
            tweet_data.get('reply_to_tweet_id'),
            tweet_data.get('conversation_id'),
            tweet_data.get('thread_id'),
            tweet_data.get('thread_position'),
            tweet_data.get('is_thread_start', 0)
        ))
        conn.commit()
        return True
    except Exception:
        return False

def check_if_tweet_exists(username: str, tweet_id: str) -> bool:
    """Check if a tweet already exists in the database."""
    try:
        # Use repository pattern
        tweet_repo = get_tweet_repository()
        tweet = tweet_repo.get_tweet_by_id(tweet_id)
        return tweet is not None and tweet.get('username') == username

    except Exception as e:
        print(f"  ⚠️ Error checking if tweet exists: {e}")
        return False


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


def init_db():
    """Initialize database with schema."""
    from database import get_db_connection
    conn = get_db_connection()
    c = conn.cursor()
    
    # The schema is already created by migrate_tweets_schema.py
    # Just verify it exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tweets'")
    if not c.fetchone():
        print("❌ Tweets table not found! Run migrate_tweets_schema.py first.")
        raise Exception("Database not properly initialized")
    
    print("✅ Database schema ready")
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


def update_tweet_in_database(tweet_id: str, tweet_data: dict) -> bool:
    """
    Update tweet in database with refetched data.
    
    Args:
        tweet_id: Tweet ID
        tweet_data: Complete tweet data dict
        
    Returns:
        bool: True if successful
    """
    try:
        from database import get_db_connection_context
        with get_db_connection_context() as conn:
            c = conn.cursor()
                        
            # Direct UPDATE to force save all fields
            c.execute("""
                UPDATE tweets SET 
                    content = ?,
                    original_content = ?,
                    original_author = ?,
                    original_tweet_id = ?,
                    reply_to_username = ?,
                    media_links = ?,
                    media_count = ?,
                    engagement_likes = ?,
                    engagement_retweets = ?,
                    engagement_replies = ?
                WHERE tweet_id = ?
            """, (
                tweet_data.get('content'),
                tweet_data.get('original_content'),
                tweet_data.get('original_author'),
                tweet_data.get('original_tweet_id'),
                tweet_data.get('reply_to_username'),
                tweet_data.get('media_links'),
                tweet_data.get('media_count', 0),
                tweet_data.get('engagement_likes', 0),
                tweet_data.get('engagement_retweets', 0),
                tweet_data.get('engagement_replies', 0),
                tweet_id
            ))
            
            rows_updated = c.rowcount
            conn.commit()
            
            if rows_updated > 0:
                print(f"💾 Tweet updated in database ({rows_updated} rows)")
                return True
            else:
                print(f"⚠️ No rows updated - tweet may not exist")
                return False
                
    except Exception as e:
        print(f"❌ Database update error: {e}")
        return False
