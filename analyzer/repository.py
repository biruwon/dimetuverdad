"""
Database operations for content analysis storage and retrieval.
"""

import sqlite3
import json
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
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

    def save_failed_analysis(self, tweet_id: str, tweet_url: str, username: str, content: str, 
                           error_message: str, media_urls: Optional[List[str]] = None) -> None:
        """
        Save failed analysis attempt to database for debugging.
        
        Args:
            tweet_id: ID of the tweet that failed analysis
            tweet_url: URL of the tweet
            username: Username of the tweet author
            content: Tweet content
            error_message: Error message from the failed analysis
            media_urls: List of media URLs if any
        """
        try:
            # Create a ContentAnalysis object for failed analysis
            failed_analysis = ContentAnalysis(
                tweet_id=tweet_id,
                tweet_url=tweet_url,
                username=username,
                tweet_content=content,
                analysis_timestamp=datetime.now().isoformat(),
                category="ERROR",
                categories_detected=["ERROR"],
                llm_explanation=f"Analysis failed: {error_message}",
                analysis_method="error",
                media_urls=media_urls or [],
                media_analysis="",
                media_type="",
                multimodal_analysis=bool(media_urls),
                pattern_matches=[],
                topic_classification={},
                analysis_json=f'{{"error": "{error_message[:500]}", "media_urls": {len(media_urls or [])}}}'
            )
            
            # Save to database
            self.save(failed_analysis)
            print(f"Saved failed analysis for tweet {tweet_id}: {error_message[:100]}")
            
        except Exception as save_error:
            print(f"Failed to save error analysis for tweet {tweet_id}: {save_error}")

    def get_tweets_for_analysis(self, username: Optional[str] = None, max_tweets: Optional[int] = None, 
                               force_reanalyze: bool = False) -> List[Tuple[str, str, str, str, str, str]]:
        """
        Get tweets from database that need analysis.
        
        Args:
            username: Specific username to analyze (None for all)
            max_tweets: Maximum number of tweets to return (None for all)
            force_reanalyze: If True, return all tweets (including already analyzed)
            
        Returns:
            List of tuples: (tweet_id, tweet_url, username, content, media_links, original_content)
        """
        try:
            conn = sqlite3.connect(self.db_path, timeout=self.timeout)
            cursor = conn.cursor()
            
            # Build query - exclude already analyzed tweets unless force_reanalyze is True
            # Also skip RT posts where the original content has already been analyzed
            if force_reanalyze:
                query = """
                    SELECT t.tweet_id, t.tweet_url, t.username, t.content, t.media_links, t.original_content FROM tweets t
                """
                params = []
                if username:
                    query += " WHERE t.username = ?"
                    params.append(username)
            else:
                query = """
                    SELECT t.tweet_id, t.tweet_url, t.username, t.content, t.media_links, t.original_content FROM tweets t 
                    LEFT JOIN content_analyses ca ON t.tweet_id = ca.tweet_id 
                    WHERE ca.tweet_id IS NULL
                    AND NOT (
                        t.post_type IN ('repost_other', 'repost_own') 
                        AND t.rt_original_analyzed = 1
                    )
                """
                params = []
                if username:
                    query += " AND t.username = ?"
                    params.append(username)
            
            query += " ORDER BY t.tweet_id DESC"
            
            if max_tweets:
                query += " LIMIT ?"
                params.append(max_tweets)
            
            cursor.execute(query, params)
            tweets = cursor.fetchall()
            conn.close()
            
            return tweets
            
        except Exception as e:
            print(f"❌ Error retrieving tweets for analysis: {e}")
            return []

    def get_tweet_data(self, tweet_id: str) -> Optional[Dict[str, Any]]:
        """
        Get tweet data by tweet ID.
        
        Args:
            tweet_id: Twitter tweet ID
            
        Returns:
            Dictionary with tweet data or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path, timeout=self.timeout)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT tweet_id, tweet_url, username, content, media_links, original_content 
                FROM tweets 
                WHERE tweet_id = ?
            """, (tweet_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'tweet_id': row['tweet_id'],
                    'tweet_url': row['tweet_url'],
                    'username': row['username'],
                    'content': row['content'],
                    'media_links': row['media_links'],
                    'original_content': row['original_content']
                }
            return None
            
        except Exception as e:
            print(f"❌ Error retrieving tweet data for {tweet_id}: {e}")
            return None

    def delete_existing_analysis(self, tweet_id: str) -> bool:
        """
        Delete existing analysis for a tweet (used for reanalysis).
        
        Args:
            tweet_id: Twitter tweet ID
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path, timeout=self.timeout)
            cursor = conn.cursor()
            
            cursor.execute(f'DELETE FROM {DatabaseConstants.TABLE_NAME} WHERE tweet_id = ?', (tweet_id,))
            
            conn.commit()
            conn.close()
            
            return cursor.rowcount > 0
            
        except Exception as e:
            print(f"❌ Error deleting analysis for tweet {tweet_id}: {e}")
            return False

    def get_analysis_count_by_username(self, username: Optional[str] = None) -> int:
        """
        Get count of analyses, optionally filtered by username.
        
        Args:
            username: Username to filter by (None for all)
            
        Returns:
            Count of analyses
        """
        try:
            conn = sqlite3.connect(self.db_path, timeout=self.timeout)
            cursor = conn.cursor()
            
            query = f'SELECT COUNT(*) FROM {DatabaseConstants.TABLE_NAME}'
            params = []
            
            if username:
                query += " WHERE username = ?"
                params.append(username)
            
            cursor.execute(query, params)
            count = cursor.fetchone()[0]
            conn.close()
            
            return count
            
        except Exception as e:
            print(f"❌ Error getting analysis count: {e}")
            return 0

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