import sqlite3
from typing import Dict
from datetime import datetime

# Import path utilities for consistent database path resolution
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from utils import paths
from database.repositories import get_tweet_repository

DB_PATH = str(paths.get_db_path())

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
