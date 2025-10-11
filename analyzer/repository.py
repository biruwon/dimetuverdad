"""
Database operations for content analysis storage and retrieval.
"""

import sqlite3
import json
from typing import Optional, Dict, Any, List
from .models import ContentAnalysis
from .constants import DatabaseConstants


class ContentAnalysisRepository:
    """
    Repository for ContentAnalysis database operations.

    Handles all database interactions for storing and retrieving
    content analysis results with proper error handling and retries.
    """

    def __init__(self, db_path: str, timeout: float = DatabaseConstants.CONNECTION_TIMEOUT):
        """
        Initialize repository with database path.

        Args:
            db_path: Path to SQLite database file
            timeout: Database connection timeout in seconds
        """
        self.db_path = db_path
        self.timeout = timeout

    def save(self, analysis: ContentAnalysis) -> None:
        """
        Save content analysis to database with retry logic.

        Args:
            analysis: ContentAnalysis object to save

        Raises:
            sqlite3.OperationalError: If database remains locked after retries
        """
        for attempt in range(DatabaseConstants.MAX_RETRIES):
            try:
                conn = sqlite3.connect(self.db_path, timeout=self.timeout)
                cursor = conn.cursor()

                cursor.execute(f'''
                INSERT OR REPLACE INTO {DatabaseConstants.TABLE_NAME}
                (tweet_id, tweet_url, username, tweet_content, category,
                 llm_explanation, analysis_method, analysis_json, analysis_timestamp,
                 categories_detected, media_urls, media_analysis, media_type, multimodal_analysis)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    analysis.tweet_id,
                    analysis.tweet_url,
                    analysis.username,
                    analysis.tweet_content,
                    analysis.category,
                    analysis.llm_explanation,
                    analysis.analysis_method,
                    analysis.analysis_json,
                    analysis.analysis_timestamp,
                    json.dumps(analysis.categories_detected, ensure_ascii=False),
                    json.dumps(analysis.media_urls, ensure_ascii=False),
                    analysis.media_analysis,
                    analysis.media_type,
                    analysis.multimodal_analysis
                ))

                conn.commit()
                conn.close()
                return  # Success

            except sqlite3.OperationalError as e:
                if attempt < DatabaseConstants.MAX_RETRIES - 1:
                    print(f"⚠️ Database locked, retrying in {DatabaseConstants.RETRY_DELAY}s... "
                          f"(attempt {attempt + 1}/{DatabaseConstants.MAX_RETRIES})")
                    import time
                    time.sleep(DatabaseConstants.RETRY_DELAY)
                    # Exponential backoff
                    DatabaseConstants.RETRY_DELAY *= 2
                else:
                    print(f"❌ Database remains locked after {DatabaseConstants.MAX_RETRIES} attempts: {e}")
                    raise

    def get_by_tweet_id(self, tweet_id: str) -> Optional[ContentAnalysis]:
        """
        Retrieve content analysis by tweet ID.

        Args:
            tweet_id: Twitter tweet ID

        Returns:
            ContentAnalysis object if found, None otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path, timeout=self.timeout)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            cursor = conn.cursor()

            cursor.execute(f'''
            SELECT * FROM {DatabaseConstants.TABLE_NAME}
            WHERE tweet_id = ?
            ''', (tweet_id,))

            row = cursor.fetchone()
            conn.close()

            if row:
                return self._row_to_content_analysis(row)
            return None

        except Exception as e:
            print(f"❌ Error retrieving analysis for tweet {tweet_id}: {e}")
            return None

    def get_recent_analyses(self, limit: int = 100) -> List[ContentAnalysis]:
        """
        Get recent content analyses ordered by timestamp.

        Args:
            limit: Maximum number of analyses to return

        Returns:
            List of ContentAnalysis objects
        """
        try:
            conn = sqlite3.connect(self.db_path, timeout=self.timeout)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(f'''
            SELECT * FROM {DatabaseConstants.TABLE_NAME}
            ORDER BY analysis_timestamp DESC
            LIMIT ?
            ''', (limit,))

            rows = cursor.fetchall()
            conn.close()

            return [self._row_to_content_analysis(row) for row in rows]

        except Exception as e:
            print(f"❌ Error retrieving recent analyses: {e}")
            return []

    def get_analyses_by_category(self, category: str, limit: int = 100) -> List[ContentAnalysis]:
        """
        Get content analyses by category.

        Args:
            category: Category to filter by
            limit: Maximum number of analyses to return

        Returns:
            List of ContentAnalysis objects
        """
        try:
            conn = sqlite3.connect(self.db_path, timeout=self.timeout)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(f'''
            SELECT * FROM {DatabaseConstants.TABLE_NAME}
            WHERE category = ?
            ORDER BY analysis_timestamp DESC
            LIMIT ?
            ''', (category, limit))

            rows = cursor.fetchall()
            conn.close()

            return [self._row_to_content_analysis(row) for row in rows]

        except Exception as e:
            print(f"❌ Error retrieving analyses for category {category}: {e}")
            return []

    def get_analysis_count(self) -> int:
        """
        Get total number of stored analyses.

        Returns:
            Total count of analyses in database
        """
        try:
            conn = sqlite3.connect(self.db_path, timeout=self.timeout)
            cursor = conn.cursor()

            cursor.execute(f'SELECT COUNT(*) FROM {DatabaseConstants.TABLE_NAME}')
            count = cursor.fetchone()[0]
            conn.close()

            return count

        except Exception as e:
            print(f"❌ Error getting analysis count: {e}")
            return 0

    def delete_analysis(self, tweet_id: str) -> bool:
        """
        Delete analysis by tweet ID.

        Args:
            tweet_id: Twitter tweet ID to delete

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path, timeout=self.timeout)
            cursor = conn.cursor()

            cursor.execute(f'''
            DELETE FROM {DatabaseConstants.TABLE_NAME}
            WHERE tweet_id = ?
            ''', (tweet_id,))

            deleted = cursor.rowcount > 0
            conn.commit()
            conn.close()

            return deleted

        except Exception as e:
            print(f"❌ Error deleting analysis for tweet {tweet_id}: {e}")
            return False

    def _row_to_content_analysis(self, row) -> ContentAnalysis:
        """
        Convert database row to ContentAnalysis object.

        Args:
            row: SQLite Row object

        Returns:
            ContentAnalysis object
        """
        return ContentAnalysis(
            tweet_id=row['tweet_id'],
            tweet_url=row['tweet_url'],
            username=row['username'],
            tweet_content=row['tweet_content'],
            analysis_timestamp=row['analysis_timestamp'],
            category=row['category'],
            categories_detected=json.loads(row['categories_detected'] or '[]'),
            llm_explanation=row['llm_explanation'],
            analysis_method=row['analysis_method'],
            media_urls=json.loads(row['media_urls'] or '[]'),
            media_analysis=row['media_analysis'],
            media_type=row['media_type'],
            multimodal_analysis=bool(row['multimodal_analysis']),
            pattern_matches=json.loads(row['analysis_json']).get('pattern_matches', []) if row['analysis_json'] else [],
            topic_classification=json.loads(row['analysis_json']).get('topic_classification', {}) if row['analysis_json'] else {},
            analysis_json=row['analysis_json'],
            analysis_time_seconds=0.0,  # Not stored in DB
            model_used="",  # Not stored in DB
            tokens_used=0  # Not stored in DB
        )