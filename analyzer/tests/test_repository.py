"""
Tests for analyzer/repository.py - Comprehensive test coverage for database operations.
"""

import unittest
import pytest
import sqlite3
from unittest.mock import Mock, patch
from analyzer.repository import ContentAnalysisRepository
from analyzer.models import ContentAnalysis
from analyzer.categories import Categories
from analyzer.constants import DatabaseConstants
from utils.database import get_db_connection_context

class TestContentAnalysisRepository:
    """Test the ContentAnalysisRepository class."""

    def test_init(self):
        """Test repository initialization."""
        repo = ContentAnalysisRepository()
        assert repo.timeout == DatabaseConstants.CONNECTION_TIMEOUT
        assert hasattr(repo, 'tweet_repo')
        assert hasattr(repo, 'content_analysis_repo')

    @patch('analyzer.repository.get_tweet_repository')
    @patch('analyzer.repository.get_content_analysis_repository')
    def test_init_with_dependencies(self, mock_content_repo, mock_tweet_repo):
        """Test repository initialization with mocked dependencies."""
        mock_tweet_repo.return_value = Mock()
        mock_content_repo.return_value = Mock()

        # Create repository instance to trigger initialization
        repo = ContentAnalysisRepository()

        mock_tweet_repo.assert_called_once()
        mock_content_repo.assert_called_once()

    def test_save_success(self):
        """Test successful analysis saving."""
        # Create test tweet data first
        with get_db_connection_context() as conn:
            c = conn.cursor()
            # Create account first
            c.execute('INSERT OR REPLACE INTO accounts (username, platform) VALUES (?, ?)', ("test_user", "twitter"))
            c.execute('''
                INSERT OR REPLACE INTO tweets
                (tweet_id, tweet_url, username, content, tweet_timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', ("test_123", "https://twitter.com/test/status/test_123", "test_user", "Test content", "2024-01-01T12:00:00"))
            conn.commit()

        repo = ContentAnalysisRepository()

        analysis = ContentAnalysis(
            post_id="test_123",
            post_url="https://twitter.com/test/status/test_123",
            author_username="test_user",
            post_content="Test content",
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.HATE_SPEECH,
            categories_detected=[Categories.HATE_SPEECH],
            local_explanation="Test explanation",
            analysis_stages="pattern->local_llm",
            media_urls=["https://example.com/image.jpg"],
            media_type="image",
            multimodal_analysis=True,
            pattern_matches=[],
            topic_classification={}
        )

        repo.save(analysis)

        # Verify saved using environment-based database connection
        with get_db_connection_context() as conn:
            cursor = conn.cursor()
            cursor.execute(f'SELECT * FROM {DatabaseConstants.TABLE_NAME} WHERE post_id = ?', ("test_123",))
            row = cursor.fetchone()

            assert row is not None
            assert row['post_id'] == "test_123"
            assert row['category'] == Categories.HATE_SPEECH
            assert row['local_explanation'] == "Test explanation"
            assert row['multimodal_analysis'] == 1

    def test_save_fails_immediately_on_lock(self):
        """Test that save fails immediately on database lock (no retry logic)."""
        repo = ContentAnalysisRepository()

        # Mock the _get_connection method to raise OperationalError
        with patch.object(repo, '_get_connection') as mock_get_conn:
            mock_conn = Mock()
            mock_conn.__enter__ = Mock(side_effect=sqlite3.OperationalError("database is locked"))
            mock_conn.__exit__ = Mock(return_value=None)
            mock_get_conn.return_value = mock_conn

            analysis = ContentAnalysis(
                post_id="test_123",
                post_url="https://twitter.com/test/status/test_123",
                author_username="test_user",
                post_content="Test content",
                analysis_timestamp="2024-01-01T12:00:00",
                category=Categories.GENERAL
            )

            # Should fail immediately without retries
            with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                repo.save(analysis)

    def test_save_max_retries_exceeded(self):
        """Test save failure after max retries."""
        # Create test tweet data first
        with get_db_connection_context() as conn:
            c = conn.cursor()
            # Create account first
            c.execute('INSERT OR REPLACE INTO accounts (username, platform) VALUES (?, ?)', ("test_user", "twitter"))
            c.execute('''
                INSERT OR REPLACE INTO tweets
                (tweet_id, tweet_url, username, content, tweet_timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', ("test_123", "https://twitter.com/test/status/test_123", "test_user", "Test content", "2024-01-01T12:00:00"))
            conn.commit()

        repo = ContentAnalysisRepository()

        with patch('sqlite3.connect', side_effect=sqlite3.OperationalError("database is locked")):
            analysis = ContentAnalysis(
                post_id="test_123",
                post_url="https://twitter.com/test/status/test_123",
                author_username="test_user",
                post_content="Test content",
                analysis_timestamp="2024-01-01T12:00:00",
                category=Categories.GENERAL
            )

            with pytest.raises(sqlite3.OperationalError):
                repo.save(analysis)

    def test_get_by_post_id_success(self):
        """Test successful retrieval by post ID."""
        # Create test tweet data first
        with get_db_connection_context() as conn:
            c = conn.cursor()
            # Create account first
            c.execute('INSERT OR REPLACE INTO accounts (username, platform) VALUES (?, ?)', ("test_user", "twitter"))
            c.execute('''
                INSERT OR REPLACE INTO tweets
                (tweet_id, tweet_url, username, content, tweet_timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', ("test_123", "https://twitter.com/test/status/test_123", "test_user", "Test content", "2024-01-01T12:00:00"))
            conn.commit()

        # First save an analysis
        repo = ContentAnalysisRepository()
        
        analysis = ContentAnalysis(
            post_id="test_123",
            post_url="https://twitter.com/test/status/test_123",
            author_username="test_user",
            post_content="Test content",
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.HATE_SPEECH,
            categories_detected=[Categories.HATE_SPEECH],
            local_explanation="Test explanation",
            analysis_stages="pattern->local_llm",
            media_urls=["https://example.com/image.jpg"],
            media_type="image",
            multimodal_analysis=True,
            pattern_matches=[{"matched_text": "test", "category": "hate_speech"}],
            topic_classification={"topic": "politics"}
        )
        repo.save(analysis)

        # Now retrieve it
        retrieved = repo.get_by_post_id("test_123")
        assert retrieved is not None
        assert retrieved.post_id == "test_123"
        assert retrieved.category == Categories.HATE_SPEECH
        assert retrieved.local_explanation == "Test explanation"
        assert retrieved.multimodal_analysis == True
        assert len(retrieved.media_urls) == 1

    def test_get_by_post_id_not_found(self):
        """Test retrieval when post ID not found."""
        repo = ContentAnalysisRepository()
        result = repo.get_by_post_id("nonexistent")
        assert result is None

    def test_get_by_post_id_error(self):
        """Test error handling in get_by_post_id."""
        repo = ContentAnalysisRepository()

        with patch('sqlite3.connect', side_effect=Exception("Connection error")):
            result = repo.get_by_post_id("test_123")
            assert result is None

    def test_get_recent_analyses(self):
        """Test retrieval of recent analyses."""
        # Create test tweet data first
        with get_db_connection_context() as conn:
            c = conn.cursor()
            # Create account first
            c.execute('INSERT OR REPLACE INTO accounts (username, platform) VALUES (?, ?)', ("test_user", "twitter"))
            for i in range(3):
                c.execute('''
                    INSERT OR REPLACE INTO tweets
                    (tweet_id, tweet_url, username, content, tweet_timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (f"test_{i}", f"https://twitter.com/test/status/test_{i}", "test_user", f"Test content {i}", f"2024-01-01T12:0{i}:00"))
            conn.commit()

        repo = ContentAnalysisRepository()

        # Save multiple analyses
        for i in range(3):
            analysis = ContentAnalysis(
                post_id=f"test_{i}",
                post_url=f"https://twitter.com/test/status/test_{i}",
                author_username="test_user",
                post_content=f"Test content {i}",
                analysis_timestamp=f"2024-01-01T12:0{i}:00",
                category=Categories.GENERAL
            )
            repo.save(analysis)

        # Retrieve recent analyses
        recent = repo.get_recent_analyses(limit=2)
        assert len(recent) == 2
        # Should be ordered by timestamp DESC
        assert recent[0].post_id == "test_2"
        assert recent[1].post_id == "test_1"

    def test_get_recent_analyses_error(self):
        """Test error handling in get_recent_analyses."""
        repo = ContentAnalysisRepository()

        with patch('sqlite3.connect', side_effect=Exception("Connection error")):
            result = repo.get_recent_analyses()
            assert result == []

    def test_get_analyses_by_category(self):
        """Test retrieval of analyses by category."""
        # Clean up any existing analyses first
        with get_db_connection_context() as conn:
            c = conn.cursor()
            c.execute('DELETE FROM content_analyses')
            conn.commit()

        # Create test tweet data first
        with get_db_connection_context() as conn:
            c = conn.cursor()
            # Create account first
            c.execute('INSERT OR REPLACE INTO accounts (username, platform) VALUES (?, ?)', ("test_user", "twitter"))
            categories = [Categories.HATE_SPEECH, Categories.DISINFORMATION, Categories.HATE_SPEECH]
            for i, category in enumerate(categories):
                c.execute('''
                    INSERT OR REPLACE INTO tweets
                    (tweet_id, tweet_url, username, content, tweet_timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (f"test_{i}", f"https://twitter.com/test/status/test_{i}", "test_user", f"Test content {i}", f"2024-01-01T12:0{i}:00"))
            conn.commit()

        repo = ContentAnalysisRepository()

        # Save analyses with different categories
        categories = [Categories.HATE_SPEECH, Categories.DISINFORMATION, Categories.HATE_SPEECH]
        for i, category in enumerate(categories):
            analysis = ContentAnalysis(
                post_id=f"test_{i}",
                post_url=f"https://twitter.com/test/status/test_{i}",
                author_username="test_user",
                post_content=f"Test content {i}",
                analysis_timestamp=f"2024-01-01T12:0{i}:00",
                category=category,
                categories_detected=[category]
            )
            repo.save(analysis)

        # Retrieve by category
        hate_speech_analyses = repo.get_analyses_by_category(Categories.HATE_SPEECH)
        assert len(hate_speech_analyses) == 2
        assert all(a.category == Categories.HATE_SPEECH for a in hate_speech_analyses)

    def test_get_analyses_by_category_error(self):
        """Test error handling in get_analyses_by_category."""
        repo = ContentAnalysisRepository()

        with patch('sqlite3.connect', side_effect=Exception("Connection error")):
            result = repo.get_analyses_by_category(Categories.HATE_SPEECH)
            assert result == []

    def test_get_analysis_count(self):
        """Test getting total analysis count."""
        # Clean up any existing analyses first
        with get_db_connection_context() as conn:
            c = conn.cursor()
            c.execute('DELETE FROM content_analyses')
            conn.commit()

        repo = ContentAnalysisRepository()

        # Initially should be 0
        assert repo.get_analysis_count() == 0

        # Create test tweet data first
        with get_db_connection_context() as conn:
            c = conn.cursor()
            # Create account first
            c.execute('INSERT OR REPLACE INTO accounts (username, platform) VALUES (?, ?)', ("test_user", "twitter"))
            for i in range(3):
                c.execute('''
                    INSERT OR REPLACE INTO tweets
                    (tweet_id, tweet_url, username, content, tweet_timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (f"test_{i}", f"https://twitter.com/test/status/test_{i}", "test_user", f"Test content {i}", f"2024-01-01T12:0{i}:00"))
            conn.commit()

        # Save some analyses
        for i in range(3):
            analysis = ContentAnalysis(
                post_id=f"test_{i}",
                post_url=f"https://twitter.com/test/status/test_{i}",
                author_username="test_user",
                post_content=f"Test content {i}",
                analysis_timestamp=f"2024-01-01T12:0{i}:00",
                category=Categories.GENERAL
            )
            repo.save(analysis)

        assert repo.get_analysis_count() == 3

    def test_get_analysis_count_error(self):
        """Test error handling in get_analysis_count."""
        repo = ContentAnalysisRepository()

        with patch('sqlite3.connect', side_effect=Exception("Connection error")):
            result = repo.get_analysis_count()
            assert result == 0

    def test_save_failed_analysis(self):
        """Test saving failed analysis."""
        # Create test tweet data first
        with get_db_connection_context() as conn:
            c = conn.cursor()
            # Create account first
            c.execute('INSERT OR REPLACE INTO accounts (username, platform) VALUES (?, ?)', ("test_user", "twitter"))
            c.execute('''
                INSERT OR REPLACE INTO tweets
                (tweet_id, tweet_url, username, content, tweet_timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', ("test_123", "https://twitter.com/test/status/test_123", "test_user", "Test content", "2024-01-01T12:00:00"))
            conn.commit()

        repo = ContentAnalysisRepository()

        repo.save_failed_analysis(
            post_id="test_123",
            post_url="https://twitter.com/test/status/test_123",
            author_username="test_user",
            content="Test content",
            error_message="Analysis error occurred",
            media_urls=["https://example.com/image.jpg"]
        )

        # Verify saved as error using environment-based database connection
        with get_db_connection_context() as conn:
            cursor = conn.cursor()
            cursor.execute(f'SELECT * FROM {DatabaseConstants.TABLE_NAME} WHERE post_id = ?', ("test_123",))
            row = cursor.fetchone()

            assert row is not None
            assert row['category'] == "ERROR"
            assert "Analysis error occurred" in row['local_explanation']
            assert row['analysis_stages'] == "error"
            assert row['multimodal_analysis'] == 1

    def test_save_failed_analysis_no_media(self):
        """Test saving failed analysis without media."""
        # Create test tweet data first
        with get_db_connection_context() as conn:
            c = conn.cursor()
            # Create account first
            c.execute('INSERT OR REPLACE INTO accounts (username, platform) VALUES (?, ?)', ("test_user", "twitter"))
            c.execute('''
                INSERT OR REPLACE INTO tweets
                (tweet_id, tweet_url, username, content, tweet_timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', ("failed_456", "https://twitter.com/test/status/failed_456", "test_user", "Failed content", "2024-01-01T12:00:00"))
            conn.commit()

        repo = ContentAnalysisRepository()

        repo.save_failed_analysis(
            post_id="failed_456",
            post_url="https://twitter.com/test/status/failed_456",
            author_username="test_user",
            content="Failed content",
            error_message="Analysis error occurred"
        )

        # Verify saved using environment-based database connection
        with get_db_connection_context() as conn:
            cursor = conn.cursor()
            cursor.execute(f'SELECT * FROM {DatabaseConstants.TABLE_NAME} WHERE post_id = ?', ("failed_456",))
            row = cursor.fetchone()

            assert row is not None
            assert row['multimodal_analysis'] == 0

    @patch('analyzer.repository.get_tweet_repository')
    def test_get_tweets_for_analysis_force_reanalyze(self, mock_tweet_repo):
        """Test getting tweets for analysis with force reanalyze."""
        mock_repo = Mock()
        mock_tweet_repo.return_value = mock_repo

        # Mock tweet data
        mock_tweets = [
            Mock(
                tweet_id="123",
                tweet_url="https://twitter.com/test/status/123",
                username="testuser",
                content="Test content",
                media_links="url1.jpg",
                original_content="Original content"
            )
        ]
        mock_repo.get_tweets_by_username.return_value = mock_tweets

        repo = ContentAnalysisRepository()
        result = repo.get_tweets_for_analysis(force_reanalyze=True, max_tweets=10)

        assert len(result) == 1
        assert result[0] == ("123", "https://twitter.com/test/status/123", "testuser", "Test content", "url1.jpg", "Original content")

    @patch('sqlite3.connect')
    def test_get_tweets_for_analysis_unanalyzed(self, mock_connect):
        """Test getting unanalyzed tweets."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            ("123", "https://twitter.com/test/status/123", "testuser", "Test content", "url1.jpg", "Original content")
        ]
        mock_connect.return_value = mock_conn

        repo = ContentAnalysisRepository()
        result = repo.get_tweets_for_analysis(max_tweets=10)

        assert len(result) == 1
        assert result[0] == ("123", "https://twitter.com/test/status/123", "testuser", "Test content", "url1.jpg", "Original content")

        # Verify the complex query was executed
        mock_cursor.execute.assert_called_once()
        query = mock_cursor.execute.call_args[0][0]
        assert "LEFT JOIN content_analyses" in query
        assert "WHERE ca.post_id IS NULL" in query

    @patch('sqlite3.connect')
    def test_get_tweets_for_analysis_error(self, mock_connect):
        """Test error handling in get_tweets_for_analysis."""
        mock_connect.side_effect = Exception("Connection error")

        repo = ContentAnalysisRepository()
        result = repo.get_tweets_for_analysis()

        assert result == []

    @patch('analyzer.repository.get_tweet_repository')
    def test_get_tweet_data_success(self, mock_tweet_repo):
        """Test successful tweet data retrieval."""
        mock_repo = Mock()
        mock_tweet_repo.return_value = mock_repo

        mock_tweet = {
            'tweet_id': '123',
            'tweet_url': 'https://twitter.com/test/status/123',
            'username': 'testuser',
            'content': 'Test content',
            'media_links': 'url1.jpg',
            'original_content': 'Original content'
        }
        mock_repo.get_tweet_by_id.return_value = mock_tweet

        repo = ContentAnalysisRepository()
        result = repo.get_tweet_data('123')

        assert result is not None
        assert result['tweet_id'] == '123'
        assert result['content'] == 'Test content'

    @patch('analyzer.repository.get_tweet_repository')
    def test_get_tweet_data_not_found(self, mock_tweet_repo):
        """Test tweet data retrieval when not found."""
        mock_repo = Mock()
        mock_tweet_repo.return_value = mock_repo
        mock_repo.get_tweet_by_id.return_value = None

        repo = ContentAnalysisRepository()
        result = repo.get_tweet_data('123')

        assert result is None

    @patch('analyzer.repository.get_tweet_repository')
    def test_get_tweet_data_error(self, mock_tweet_repo):
        """Test error handling in get_tweet_data."""
        mock_repo = Mock()
        mock_tweet_repo.return_value = mock_repo
        mock_repo.get_tweet_by_id.side_effect = Exception("Repository error")

        repo = ContentAnalysisRepository()
        result = repo.get_tweet_data('123')

        assert result is None

    def test_delete_existing_analysis_success(self):
        """Test successful deletion of existing analysis."""
        # Create test tweet data first
        with get_db_connection_context() as conn:
            c = conn.cursor()
            # Create account first
            c.execute('INSERT OR REPLACE INTO accounts (username, platform) VALUES (?, ?)', ("test_user", "twitter"))
            c.execute('''
                INSERT OR REPLACE INTO tweets
                (tweet_id, tweet_url, username, content, tweet_timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', ("test_123", "https://twitter.com/test/status/test_123", "test_user", "Test content", "2024-01-01T12:00:00"))
            conn.commit()

        # First save an analysis
        repo = ContentAnalysisRepository()
        
        analysis = ContentAnalysis(
            post_id="test_123",
            post_url="https://twitter.com/test/status/test_123",
            author_username="test_user",
            post_content="Test content",
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.GENERAL
        )
        repo.save(analysis)

        # Now delete it
        result = repo.delete_existing_analysis("test_123")
        assert result is True

        # Verify deleted
        retrieved = repo.get_by_post_id("test_123")
        assert retrieved is None

    def test_delete_existing_analysis_not_found(self):
        """Test deletion when analysis doesn't exist."""
        repo = ContentAnalysisRepository()
        result = repo.delete_existing_analysis("nonexistent")
        assert result is False

    def test_delete_existing_analysis_error(self):
        """Test error handling in delete_existing_analysis."""
        repo = ContentAnalysisRepository()

        with patch('sqlite3.connect', side_effect=Exception("Connection error")):
            result = repo.delete_existing_analysis("test_123")
            assert result is False

    @patch('analyzer.repository.get_content_analysis_repository')
    def test_get_analysis_count_by_author_all(self, mock_content_repo):
        """Test getting analysis count for all authors."""
        mock_repo = Mock()
        mock_content_repo.return_value = mock_repo
        mock_repo.get_analysis_stats.return_value = {'total_analyses': 42}

        repo = ContentAnalysisRepository()
        result = repo.get_analysis_count_by_author()

        assert result == 42

    @patch('analyzer.repository.get_content_analysis_repository')
    def test_get_analysis_count_by_author_specific(self, mock_content_repo):
        """Test getting analysis count for specific author."""
        mock_repo = Mock()
        mock_content_repo.return_value = mock_repo
        mock_analyses = [Mock(), Mock(), Mock()]  # 3 analyses
        mock_repo.get_analyses_by_username.return_value = mock_analyses

        repo = ContentAnalysisRepository()
        result = repo.get_analysis_count_by_author("testuser")

        assert result == 3

    @patch('analyzer.repository.get_content_analysis_repository')
    def test_get_analysis_count_by_author_error(self, mock_content_repo):
        """Test error handling in get_analysis_count_by_author."""
        mock_repo = Mock()
        mock_content_repo.return_value = mock_repo
        mock_repo.get_analysis_stats.side_effect = Exception("Repository error")

        repo = ContentAnalysisRepository()
        result = repo.get_analysis_count_by_author()

        assert result == 0

    def test_row_to_content_analysis(self):
        """Test conversion from database row to ContentAnalysis."""
        repo = ContentAnalysisRepository()

        # Create a mock row
        mock_row = Mock()
        mock_row.__getitem__ = Mock(side_effect=lambda key: {
            'post_id': 'test_123',
            'post_url': 'https://twitter.com/test/status/test_123',
            'author_username': 'test_user',
            'post_content': 'Test content',
            'analysis_timestamp': '2024-01-01T12:00:00',
            'category': Categories.HATE_SPEECH,
            'categories_detected': '["hate_speech"]',
            'local_explanation': 'Test explanation',
            'analysis_stages': 'llm',
            'media_urls': '["https://example.com/image.jpg"]',
            'media_analysis': 'Test media analysis',
            'media_type': 'image',
            'multimodal_analysis': 1,
            'analysis_json': '{"pattern_matches": [], "topic_classification": {}}',
            'verification_data': None,
            'verification_confidence': 0.0
        }[key])

        result = repo._row_to_content_analysis(mock_row)

        assert isinstance(result, ContentAnalysis)
        assert result.post_id == 'test_123'
        assert result.category == Categories.HATE_SPEECH
        assert result.local_explanation == 'Test explanation'
        assert result.multimodal_analysis == True
        assert len(result.media_urls) == 1
        assert result.media_urls[0] == 'https://example.com/image.jpg'