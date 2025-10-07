import sqlite3
from typing import Optional, Dict
from datetime import datetime

DB_PATH = "accounts.db"

def get_connection(timeout: float = 10.0) -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, timeout=timeout)


def get_last_tweet_timestamp(username: str) -> Optional[str]:
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT tweet_timestamp FROM tweets WHERE username = ? ORDER BY tweet_timestamp DESC LIMIT 1", (username,))
        row = cur.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def save_tweet(conn: sqlite3.Connection, tweet_data: Dict) -> bool:
    """Simplified save/update function for tests and main logic."""
    c = conn.cursor()
    try:
        if not tweet_data.get('tweet_id') or str(tweet_data.get('tweet_id')).lower() == 'analytics':
            return False
        c.execute("SELECT id, post_type, content, original_author, original_tweet_id FROM tweets WHERE tweet_id = ?", (tweet_data['tweet_id'],))
        existing = c.fetchone()
        if existing:
            existing_id, existing_post_type, existing_content, existing_original_author, existing_original_tweet_id = existing[0], existing[1], existing[2], existing[3], existing[4]
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
                        media_links = ?, media_count = ?, media_types = ?,
                        engagement_likes = ?, engagement_retweets = ?, engagement_replies = ?,
                        external_links = ?, original_content = ?, is_pinned = ?
                        WHERE tweet_id = ?""", (
                tweet_data.get('content'),
                new_post_type,
                tweet_data.get('original_author'),
                tweet_data.get('original_tweet_id'),
                tweet_data.get('media_links'),
                tweet_data.get('media_count', 0),
                tweet_data.get('media_types'),
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
                        media_links, media_count, media_types,
                        engagement_likes, engagement_retweets, engagement_replies,
                        external_links, original_content, is_pinned)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
            tweet_data['tweet_id'],
            tweet_data.get('content'),
            tweet_data.get('username'),
            tweet_data.get('tweet_url'),
            tweet_data.get('tweet_timestamp'),
            tweet_data.get('post_type', 'original'),
            tweet_data.get('media_links'),
            tweet_data.get('media_count', 0),
            tweet_data.get('media_types'),
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


def save_enhanced_tweet(conn, tweet_data: Dict) -> bool:
    """Save tweet with enhanced data structure - simplified for current schema."""
    try:
        saved = save_tweet(conn, tweet_data)
        if saved:
            print(f"  ✅ Saved/Updated tweet: {tweet_data.get('tweet_id')}")
        else:
            print(f"  ⏭️ Not saved (duplicate/unchanged): {tweet_data.get('tweet_id')}")
        return saved
    except Exception as e:
        print(f"  ❌ Error saving tweet via fetcher_db: {e}")
        return False


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
