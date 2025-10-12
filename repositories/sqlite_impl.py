"""
SQLite implementations of repository interfaces.
Concrete implementations using SQLite database.
"""

import sqlite3
from typing import Optional, Dict, List, Any
from datetime import datetime
from contextlib import contextmanager

from .interfaces import (
    TweetRepositoryInterface,
    ContentAnalysisRepositoryInterface,
    AccountRepositoryInterface,
    PostEditRepositoryInterface
)


class SQLiteRepositoryBase:
    """Base class for SQLite repositories."""

    def __init__(self, connection_factory=None):
        """Initialize with connection factory."""
        self._connection_factory = connection_factory or self._default_connection_factory

    def _default_connection_factory(self):
        """Default connection factory - import utils.database."""
        from utils.database import get_db_connection
        return get_db_connection()

    @contextmanager
    def get_connection(self):
        """Get database connection context manager."""
        conn = self._connection_factory()
        try:
            yield conn
        finally:
            conn.close()

    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert SQLite Row to dictionary."""
        if hasattr(row, 'keys'):
            return dict(row)
        return row


class SQLiteTweetRepository(SQLiteRepositoryBase, TweetRepositoryInterface):
    """SQLite implementation of TweetRepositoryInterface."""

    def get_tweet_by_id(self, tweet_id: str) -> Optional[Dict[str, Any]]:
        """Get tweet data by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM tweets WHERE tweet_id = ?
            """, (tweet_id,))
            result = cursor.fetchone()
            return self._row_to_dict(result) if result else None

    def get_tweets_by_username(self, username: str, limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        """Get tweets for a specific user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT * FROM tweets WHERE username = ?
                ORDER BY tweet_timestamp DESC
            """
            params = [username]

            if limit:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])

            cursor.execute(query, params)
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_tweet_count_by_username(self, username: str) -> int:
        """Get total tweet count for a user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM tweets WHERE username = ?
            """, (username,))
            return cursor.fetchone()[0]

    def update_tweet_status(self, tweet_id: str, is_deleted: bool = None, is_edited: bool = None) -> bool:
        """Update tweet status flags."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            updates = {}
            if is_deleted is not None:
                updates['is_deleted'] = 1 if is_deleted else 0
            if is_edited is not None:
                updates['is_edited'] = 1 if is_edited else 0

            if not updates:
                return False

            set_clause = ', '.join(f"{k} = ?" for k in updates.keys())
            values = list(updates.values()) + [tweet_id]

            cursor.execute(f"""
                UPDATE tweets SET {set_clause} WHERE tweet_id = ?
            """, values)

            conn.commit()
            return cursor.rowcount > 0

    def get_recent_tweets(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recently scraped tweets."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM tweets
                ORDER BY scraped_at DESC
                LIMIT ?
            """, (limit,))
            return [self._row_to_dict(row) for row in cursor.fetchall()]


class SQLiteContentAnalysisRepository(SQLiteRepositoryBase, ContentAnalysisRepositoryInterface):
    """SQLite implementation of ContentAnalysisRepositoryInterface."""

    def get_analysis_by_post_id(self, post_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis for a specific post."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM content_analyses WHERE post_id = ?
            """, (post_id,))
            result = cursor.fetchone()
            return self._row_to_dict(result) if result else None

    def save_analysis(self, analysis_data: Dict[str, Any]) -> bool:
        """Save or update content analysis."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Check if analysis already exists
            existing = self.get_analysis_by_post_id(analysis_data['post_id'])
            if existing:
                return self.update_analysis(analysis_data['post_id'], analysis_data)

            # Insert new analysis
            columns = ', '.join(analysis_data.keys())
            placeholders = ', '.join('?' * len(analysis_data))
            values = list(analysis_data.values())

            cursor.execute(f"""
                INSERT INTO content_analyses ({columns}) VALUES ({placeholders})
            """, values)

            conn.commit()
            return cursor.rowcount > 0

    def update_analysis(self, post_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing analysis."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            set_clause = ', '.join(f"{k} = ?" for k in updates.keys())
            values = list(updates.values()) + [post_id]

            cursor.execute(f"""
                UPDATE content_analyses SET {set_clause}, analysis_timestamp = datetime('now')
                WHERE post_id = ?
            """, values)

            conn.commit()
            return cursor.rowcount > 0

    def get_analyses_by_category(self, category: str, limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        """Get analyses by category."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT ca.*, t.content, t.username, t.tweet_url
                FROM content_analyses ca
                JOIN tweets t ON ca.post_id = t.tweet_id
                WHERE ca.category = ?
                ORDER BY ca.analysis_timestamp DESC
            """
            params = [category]

            if limit:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])

            cursor.execute(query, params)
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_analyses_by_username(self, username: str) -> List[Dict[str, Any]]:
        """Get all analyses for a user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT ca.*, t.content, t.tweet_url
                FROM content_analyses ca
                JOIN tweets t ON ca.post_id = t.tweet_id
                WHERE ca.author_username = ?
                ORDER BY ca.analysis_timestamp DESC
            """, (username,))
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get overall analysis statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Total stats
            cursor.execute("""
                SELECT
                    COUNT(DISTINCT t.tweet_id) as total_tweets,
                    COUNT(CASE WHEN ca.post_id IS NOT NULL THEN 1 END) as analyzed_tweets,
                    COUNT(CASE WHEN ca.analysis_method = 'pattern' THEN 1 END) as pattern_analyzed,
                    COUNT(CASE WHEN ca.analysis_method = 'llm' THEN 1 END) as llm_analyzed,
                    COUNT(CASE WHEN ca.analysis_method = 'gemini' THEN 1 END) as gemini_analyzed
                FROM tweets t
                LEFT JOIN content_analyses ca ON t.tweet_id = ca.post_id
            """)
            total_stats = self._row_to_dict(cursor.fetchone())

            # Category distribution
            cursor.execute("""
                SELECT category, COUNT(*) as count
                FROM content_analyses
                WHERE category IS NOT NULL
                GROUP BY category
                ORDER BY count DESC
            """)
            categories = [self._row_to_dict(row) for row in cursor.fetchall()]

            return {
                'total_stats': total_stats,
                'categories': categories
            }

    def delete_analysis(self, post_id: str) -> bool:
        """Delete analysis for a post."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM content_analyses WHERE post_id = ?
            """, (post_id,))
            conn.commit()
            return cursor.rowcount > 0


class SQLiteAccountRepository(SQLiteRepositoryBase, AccountRepositoryInterface):
    """SQLite implementation of AccountRepositoryInterface."""

    def get_account_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get account data by username."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM accounts WHERE username = ?
            """, (username,))
            result = cursor.fetchone()
            return self._row_to_dict(result) if result else None

    def get_all_accounts(self, limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        """Get all accounts with pagination."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT * FROM accounts
                ORDER BY username
            """
            params = []

            if limit:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])

            cursor.execute(query, params)
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def save_account(self, account_data: Dict[str, Any]) -> bool:
        """Save or update account."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Check if account exists
            existing = self.get_account_by_username(account_data['username'])
            if existing:
                # Update existing
                updates = {k: v for k, v in account_data.items() if k != 'username'}
                set_clause = ', '.join(f"{k} = ?" for k in updates.keys())
                values = list(updates.values()) + [account_data['username']]

                cursor.execute(f"""
                    UPDATE accounts SET {set_clause} WHERE username = ?
                """, values)
            else:
                # Insert new
                columns = ', '.join(account_data.keys())
                placeholders = ', '.join('?' * len(account_data))
                values = list(account_data.values())

                cursor.execute(f"""
                    INSERT INTO accounts ({columns}) VALUES ({placeholders})
                """, values)

            conn.commit()
            return cursor.rowcount > 0

    def update_last_scraped(self, username: str, timestamp: datetime = None) -> bool:
        """Update account's last scraped timestamp."""
        if timestamp is None:
            timestamp = datetime.now()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE accounts SET last_scraped = ? WHERE username = ?
            """, (timestamp.isoformat(), username))
            conn.commit()
            return cursor.rowcount > 0

    def get_accounts_with_stats(self, limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        """Get accounts with tweet and analysis statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT
                    a.*,
                    COUNT(t.tweet_id) as tweet_count,
                    MAX(t.tweet_timestamp) as last_activity,
                    COUNT(CASE WHEN ca.category IS NOT NULL AND ca.category != 'general' THEN 1 END) as problematic_posts,
                    COUNT(CASE WHEN ca.post_id IS NOT NULL THEN 1 END) as analyzed_posts
                FROM accounts a
                LEFT JOIN tweets t ON a.username = t.username
                LEFT JOIN content_analyses ca ON t.tweet_id = ca.post_id
                GROUP BY a.username
                ORDER BY problematic_posts DESC, analyzed_posts DESC, tweet_count DESC
            """
            params = []

            if limit:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])

            cursor.execute(query, params)
            return [self._row_to_dict(row) for row in cursor.fetchall()]


class SQLitePostEditRepository(SQLiteRepositoryBase, PostEditRepositoryInterface):
    """SQLite implementation of PostEditRepositoryInterface."""

    def save_edit(self, post_id: str, previous_content: str, version_number: int = None) -> bool:
        """Save a post edit record."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if version_number is None:
                # Get next version number
                cursor.execute("""
                    SELECT MAX(version_number) FROM post_edits WHERE post_id = ?
                """, (post_id,))
                result = cursor.fetchone()
                version_number = (result[0] or 0) + 1

            cursor.execute("""
                INSERT INTO post_edits (post_id, version_number, previous_content)
                VALUES (?, ?, ?)
            """, (post_id, version_number, previous_content))

            conn.commit()
            return cursor.rowcount > 0

    def get_edits_by_post_id(self, post_id: str) -> List[Dict[str, Any]]:
        """Get edit history for a post."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM post_edits
                WHERE post_id = ?
                ORDER BY version_number DESC
            """, (post_id,))
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_latest_version(self, post_id: str) -> Optional[int]:
        """Get the latest version number for a post."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT MAX(version_number) FROM post_edits WHERE post_id = ?
            """, (post_id,))
            result = cursor.fetchone()
            return result[0] if result and result[0] else None