import sqlite3
from typing import Optional, Dict, List
from datetime import datetime

# Import path utilities for consistent database path resolution
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from utils import paths
from utils.database import get_db_connection
# Import repository interfaces
from repositories import get_tweet_repository, get_account_repository

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
        account_repo = get_account_repository()
        
        # Get counts before deletion
        tweets = tweet_repo.get_tweets_by_username(username)
        tweets_count = len(tweets)
        
        # For analyses count, we need to use direct access since content analysis repo might not have username filtering
        # This is a specialized operation that may need to stay direct for now
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS analyses_count FROM content_analyses WHERE username = ?", (username,))
        row = cur.fetchone()
        analyses_count = row['analyses_count'] if row and 'analyses_count' in row else 0
        conn.close()

        # Delete tweets using repository
        deleted_tweets = tweet_repo.delete_tweets_by_username(username)

        # Delete analyses using direct access (for now)
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM content_analyses WHERE username = ?", (username,))
        conn.commit()
        conn.close()

        print(f"✅ Deleted {deleted_tweets} tweets and {analyses_count} analyses for @{username}")

        return {'tweets': deleted_tweets, 'analyses': analyses_count}

    except Exception as e:
        print(f"❌ Error deleting data for @{username}: {e}")
        raise

def save_tweet(conn: sqlite3.Connection, tweet_data: Dict) -> bool:
    """Simplified save/update function for tests and main logic."""
    c = conn.cursor()
    try:
        if not tweet_data.get('tweet_id') or str(tweet_data.get('tweet_id')).lower() == 'analytics':
            return False
        c.execute("SELECT id, post_type, content, original_author, original_tweet_id FROM tweets WHERE tweet_id = ?", (tweet_data['tweet_id'],))
        existing = c.fetchone()
        if existing:
            existing_id = existing['id']
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
            if not needs_update:
                return False
            # perform update (minimal fields)
            c.execute("""UPDATE tweets SET content = ?, post_type = ?, original_author = ?, original_tweet_id = ?,
                        media_links = ?, media_count = ?,
                        engagement_likes = ?, engagement_retweets = ?, engagement_replies = ?,
                        external_links = ?, original_content = ?, is_pinned = ?
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
                tweet_data['tweet_id']
            ))
            conn.commit()
            return True
        # insert
        c.execute("""INSERT INTO tweets (tweet_id, content, username, tweet_url, tweet_timestamp, post_type,
                        media_links, media_count,
                        engagement_likes, engagement_retweets, engagement_replies,
                        external_links, original_content, is_pinned)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
            tweet_data['tweet_id'],
            tweet_data.get('content'),
            tweet_data.get('username'),
            tweet_data.get('tweet_url'),
            tweet_data.get('tweet_timestamp'),
            tweet_data.get('post_type', 'original'),
            tweet_data.get('media_links'),
            tweet_data.get('media_count', 0),
            tweet_data.get('engagement_likes', 0),
            tweet_data.get('engagement_retweets', 0),
            tweet_data.get('engagement_replies', 0),
            tweet_data.get('external_links'),
            tweet_data.get('original_content'),
            tweet_data.get('is_pinned', 0)
        ))
        conn.commit()
        return True
    except Exception:
        return False

def check_if_tweet_exists(username: str, tweet_id: str) -> bool:
    """Check if a tweet already exists in the database."""
    try:
        conn = get_db_connection()
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


def update_tweet_in_database(tweet_id: str, tweet_data: dict, db_path: str = None) -> bool:
    """
    Update tweet in database with refetched data.
    
    Args:
        tweet_id: Tweet ID
        tweet_data: Complete tweet data dict
        db_path: Optional database path (defaults to current environment path)
        
    Returns:
        bool: True if successful
    """
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Get current media_links to combine with any new video URLs
        c.execute("SELECT media_links FROM tweets WHERE tweet_id = ?", (tweet_id,))
        row = c.fetchone()
        current_media = row['media_links'] if row and 'media_links' in row and row['media_links'] else ""
        
        # If tweet_data contains media_links, combine them with existing ones
        if tweet_data.get('media_links'):
            existing_urls = current_media.split(',') if current_media else []
            new_urls = tweet_data['media_links'].split(',') if tweet_data['media_links'] else []
            combined_urls = list(set(existing_urls + new_urls))  # Remove duplicates
            tweet_data['media_links'] = ','.join([url for url in combined_urls if url.strip()])
            tweet_data['media_count'] = len([u for u in combined_urls if u.strip()])
        
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
            tweet_data.get('media_links'),
            tweet_data.get('media_count', 0),
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
