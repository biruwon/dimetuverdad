"""
Database utilities for the dimetuverdad project.
Centralized database connection and operation patterns.
"""

import sqlite3
import os
from typing import Optional

# Database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'accounts.db')

def get_db_connection(timeout: float = 30.0) -> sqlite3.Connection:
    """Get database connection with row factory and specified timeout."""
    conn = sqlite3.connect(DB_PATH, timeout=timeout)
    conn.row_factory = sqlite3.Row
    return conn

def get_tweet_data(tweet_id: str, timeout: float = 30.0) -> Optional[dict]:
    """Get tweet data for analysis."""
    conn = get_db_connection(timeout)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT tweet_id, content, username, media_links FROM tweets WHERE tweet_id = ?
        """, (tweet_id,))
        result = cursor.fetchone()
        if result:
            data = dict(result)
            # Parse media_links into a list
            media_links = data.get('media_links', '')
            if media_links:
                data['media_urls'] = [url.strip() for url in media_links.split(',') if url.strip()]
            else:
                data['media_urls'] = []
            return data
        return None
    finally:
        conn.close()

def delete_existing_analysis(tweet_id: str, timeout: float = 30.0) -> bool:
    """Delete existing analysis for a tweet. Returns True if deleted."""
    conn = get_db_connection(timeout)
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM content_analyses WHERE tweet_id = ?", (tweet_id,))
        conn.commit()
        return conn.total_changes > 0
    finally:
        conn.close()