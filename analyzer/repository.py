"""
Database operations for content analysis storage and retrieval.
"""

import os
import sqlite3
import json
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from .models import ContentAnalysis
from .constants import DatabaseConstants
from database.repositories import get_tweet_repository, get_content_analysis_repository
from database import get_db_connection_context

class ContentAnalysisRepository:
    """
    Repository for ContentAnalysis database operations.

    Handles all database interactions for storing and retrieving
    content analysis results with proper error handling and retries.
    """

    def __init__(self, timeout: float = DatabaseConstants.CONNECTION_TIMEOUT):
        """
        Initialize repository with automatic database detection.

        Args:
            timeout: Database connection timeout in seconds
        """
        self.timeout = timeout        
        # Initialize standardized repositories
        self.tweet_repo = get_tweet_repository()
        self.content_analysis_repo = get_content_analysis_repository()

    def _get_connection(self):
        """Get database connection context manager with proper path."""
        return get_db_connection_context()

    def save(self, analysis: ContentAnalysis) -> None:
        """
        Save content analysis to database (fail-fast, no retries).

        Args:
            analysis: ContentAnalysis object to save

        Raises:
            sqlite3.OperationalError: If database operation fails
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(f'''
                INSERT OR REPLACE INTO {DatabaseConstants.TABLE_NAME}
                (post_id, post_url, author_username, platform, post_content, category,
                 local_explanation, external_explanation, analysis_stages, external_analysis_used,
                 analysis_json, analysis_timestamp, categories_detected, 
                 media_urls, media_type, media_description,
                 verification_data, verification_confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    analysis.post_id,
                    analysis.post_url,
                    analysis.author_username,
                    'twitter',  # Default platform for existing Twitter data
                    analysis.post_content,
                    analysis.category,
                    analysis.local_explanation,
                    analysis.external_explanation or '',
                    analysis.analysis_stages,
                    analysis.external_analysis_used,
                    analysis.analysis_json,
                    analysis.analysis_timestamp,
                    json.dumps(analysis.categories_detected, ensure_ascii=False),
                    json.dumps(analysis.media_urls, ensure_ascii=False),
                    analysis.media_type,
                    analysis.media_description,
                    json.dumps(analysis.verification_data, ensure_ascii=False, default=str) if analysis.verification_data else None,
                    analysis.verification_confidence
                ))

                conn.commit()

        except sqlite3.OperationalError as e:
            print(f"❌ Database operation failed (no retry): {e}")
            raise

    def get_by_post_id(self, post_id: str) -> Optional[ContentAnalysis]:
        """
        Retrieve content analysis by post ID.

        Args:
            post_id: Platform-agnostic post ID

        Returns:
            ContentAnalysis object if found, None otherwise
        """
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row  # Enable column access by name
                cursor = conn.cursor()

                cursor.execute(f'''
                SELECT * FROM {DatabaseConstants.TABLE_NAME}
                WHERE post_id = ?
                ''', (post_id,))

                row = cursor.fetchone()

                if row:
                    return self._row_to_content_analysis(row)
                return None

        except Exception as e:
            print(f"❌ Error retrieving analysis for post {post_id}: {e}")
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
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute(f'''
                SELECT * FROM {DatabaseConstants.TABLE_NAME}
                ORDER BY analysis_timestamp DESC
                LIMIT ?
                ''', (limit,))

                rows = cursor.fetchall()

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
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute(f'''
                SELECT * FROM {DatabaseConstants.TABLE_NAME}
                WHERE category = ?
                ORDER BY analysis_timestamp DESC
                LIMIT ?
                ''', (category, limit))

                rows = cursor.fetchall()

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
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(f'SELECT COUNT(*) FROM {DatabaseConstants.TABLE_NAME}')
                count = cursor.fetchone()[0]

                return count

        except Exception as e:
            print(f"❌ Error getting analysis count: {e}")
            return 0

    def save_failed_analysis(self, post_id: str, post_url: str, author_username: str, content: str, 
                           error_message: str, media_urls: Optional[List[str]] = None) -> None:
        """
        Save failed analysis attempt to database for debugging.
        
        Args:
            post_id: ID of the post that failed analysis
            post_url: URL of the post
            author_username: Username of the post author
            content: Post content
            error_message: Error message from the failed analysis
            media_urls: List of media URLs if any
        """
        try:
            # Create a ContentAnalysis object for failed analysis
            failed_analysis = ContentAnalysis(
                post_id=post_id,
                post_url=post_url,
                author_username=author_username,
                post_content=content,
                analysis_timestamp=datetime.now().isoformat(),
                category="ERROR",
                categories_detected=["ERROR"],
                local_explanation=f"Analysis failed: {error_message}",
                external_explanation="",
                analysis_stages="error",
                external_analysis_used=False,
                media_urls=media_urls or [],
                media_type="",
                pattern_matches=[],
                topic_classification={},
                analysis_json=f'{{"error": "{error_message[:500]}", "media_urls": {len(media_urls or [])}}}'
            )
            
            # Save to database
            self.save(failed_analysis)
            print(f"Saved failed analysis for post {post_id}: {error_message[:100]}")
            
        except Exception as save_error:
            print(f"Failed to save error analysis for post {post_id}: {save_error}")

    def get_tweets_for_analysis(self, usernames: Optional[List[str]] = None, max_tweets: Optional[int] = None, 
                               force_reanalyze: bool = False) -> List[Tuple[str, str, str, str, str, str]]:
        """
        Get tweets from database that need analysis.
        
        Args:
            usernames: List of usernames to analyze (None for all)
            max_tweets: Maximum number of tweets to return (None for all)
            force_reanalyze: If True, return all tweets (including already analyzed)
            
        Returns:
            List of tuples: (tweet_id, tweet_url, username, content, media_links, original_content)
        """
        try:
            # Use standardized repository for basic tweet operations
            if force_reanalyze:
                # Get all tweets for the user or all users
                if usernames:
                    tweets_data = []
                    for username in usernames:
                        user_tweets = self.tweet_repo.get_tweets_by_username(username=username, limit=max_tweets)
                        tweets_data.extend(user_tweets)
                        if max_tweets and len(tweets_data) >= max_tweets:
                            tweets_data = tweets_data[:max_tweets]
                            break
                else:
                    # When no username specified and force_reanalyze=True, get all tweets
                    with self._get_connection() as conn:
                        cursor = conn.cursor()
                        query = """
                            SELECT tweet_id, tweet_url, username, content, media_links, original_content 
                            FROM tweets 
                            ORDER BY tweet_timestamp DESC
                        """
                        params = []
                        if max_tweets:
                            query += " LIMIT ?"
                            params.append(max_tweets)
                        
                        cursor.execute(query, params)
                        tweets_data = cursor.fetchall()
                        
                        # Convert to dict format for consistency
                        tweets_data = [
                            {
                                'tweet_id': row[0],
                                'tweet_url': row[1], 
                                'username': row[2],
                                'content': row[3],
                                'media_links': row[4],
                                'original_content': row[5]
                            }
                            for row in tweets_data
                        ]
            else:
                # Get unanalyzed tweets - this requires custom logic, so we still need some direct access
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    
                    query = """
                        SELECT t.tweet_id, t.tweet_url, t.username, t.content, t.media_links, t.original_content FROM tweets t 
                        LEFT JOIN content_analyses ca ON t.tweet_id = ca.post_id 
                        WHERE ca.post_id IS NULL
                        AND NOT (
                            t.post_type IN ('repost_other', 'repost_own') 
                            AND t.rt_original_analyzed = 1
                        )
                    """
                    params = []
                    if usernames:
                        # Use IN clause for multiple usernames
                        placeholders = ','.join('?' * len(usernames))
                        query += f" AND t.username IN ({placeholders})"
                        params.extend(usernames)
                    
                    query += " ORDER BY t.tweet_id DESC"
                    
                    if max_tweets:
                        query += " LIMIT ?"
                        params.append(max_tweets)
                    
                    cursor.execute(query, params)
                    tweets = cursor.fetchall()
                    
                    return tweets
            
            # Convert tweet data to expected format; support dicts, Rows, or objects
            result = []
            for tweet in tweets_data:
                def _val(obj, key):
                    if isinstance(obj, dict):
                        return obj.get(key)
                    try:
                        return getattr(obj, key)
                    except Exception:
                        try:
                            return obj[key]
                        except Exception:
                            return None

                result.append((
                    _val(tweet, 'tweet_id'),
                    _val(tweet, 'tweet_url'),
                    _val(tweet, 'username'),
                    _val(tweet, 'content'),
                    _val(tweet, 'media_links') or '',
                    _val(tweet, 'original_content') or ''
                ))
            
            return result
            
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
            # Use standardized repository for tweet data
            tweet = self.tweet_repo.get_tweet_by_id(tweet_id)
            if tweet:
                return {
                    'tweet_id': tweet['tweet_id'],
                    'tweet_url': tweet['tweet_url'],
                    'username': tweet['username'],
                    'content': tweet['content'],
                    'media_links': tweet.get('media_links') or '',
                    'original_content': tweet.get('original_content', '')
                }
            return None
            
        except Exception as e:
            print(f"❌ Error retrieving tweet data for {tweet_id}: {e}")
            return None

    def delete_existing_analysis(self, post_id: str) -> bool:
        """
        Delete existing analysis for a post (used for reanalysis).
        
        Args:
            post_id: Platform-agnostic post ID
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute(f'DELETE FROM {DatabaseConstants.TABLE_NAME} WHERE post_id = ?', (post_id,))
                
                conn.commit()
                
                return cursor.rowcount > 0
                
        except Exception as e:
            print(f"❌ Error deleting analysis for post {post_id}: {e}")
            return False

    def get_analysis_count_by_author(self, author_username: Optional[str] = None) -> int:
        """
        Get count of analyses, optionally filtered by author username.
        
        Args:
            author_username: Author username to filter by (None for all)
            
        Returns:
            Count of analyses
        """
        try:
            if author_username:
                # Use standardized repository for user-specific analysis count
                analyses = self.content_analysis_repo.get_analyses_by_username(author_username)
                return len(analyses)
            else:
                # Use standardized repository for total count
                stats = self.content_analysis_repo.get_analysis_stats()
                return stats.get('total_analyses', 0)
            
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
        # Access values robustly for sqlite3.Row, dicts, or simple mocks implementing __getitem__
        def _g(key, default=None):
            try:
                return row[key]
            except Exception:
                try:
                    return getattr(row, key)
                except Exception:
                    try:
                        # Mapping-like get
                        return row.get(key, default)  # type: ignore[attr-defined]
                    except Exception:
                        return default

        verification_data_raw = _g('verification_data')
        analysis_json_raw = _g('analysis_json')
        categories_detected_raw = _g('categories_detected')
        media_urls_raw = _g('media_urls')
        verification_confidence_val = _g('verification_confidence', 0.0) or 0.0

        return ContentAnalysis(
            post_id=_g('post_id'),
            post_url=_g('post_url'),
            author_username=_g('author_username'),
            post_content=_g('post_content'),
            analysis_timestamp=_g('analysis_timestamp'),
            category=_g('category'),
            categories_detected=json.loads(categories_detected_raw or '[]'),
            local_explanation=_g('local_explanation', ''),
            external_explanation=_g('external_explanation', ''),
            analysis_stages=_g('analysis_stages', ''),
            external_analysis_used=bool(_g('external_analysis_used', False)),
            media_urls=json.loads(media_urls_raw or '[]'),
            media_type=_g('media_type', ''),
            media_description=_g('media_description', ''),
            pattern_matches=json.loads(analysis_json_raw).get('pattern_matches', []) if analysis_json_raw else [],
            topic_classification=json.loads(analysis_json_raw).get('topic_classification', {}) if analysis_json_raw else {},
            analysis_json=analysis_json_raw,
            analysis_time_seconds=0.0,  # Not stored in DB
            model_used="",  # Not stored in DB
            tokens_used=0,  # Not stored in DB
            verification_data=json.loads(verification_data_raw) if verification_data_raw else None,
            verification_confidence=verification_confidence_val
        )