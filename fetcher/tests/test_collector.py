"""
Comprehensive tests for fetcher/collector.py module.

Tests the TweetCollector class and its methods for tweet collection,
data extraction, and database operations.
"""

import json
import pytest
import sqlite3
from unittest.mock import Mock, patch
from datetime import datetime

from fetcher.collector import TweetCollector, get_collector
from fetcher.config import FetcherConfig


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = Mock(spec=FetcherConfig)
    config.db_timeout = 30.0
    config.max_consecutive_empty_scrolls = 10
    return config


@pytest.fixture
def mock_page():
    """Mock Playwright page object."""
    page = Mock()
    return page


@pytest.fixture
def mock_article():
    """Mock tweet article element."""
    article = Mock()
    return article


@pytest.fixture
def mock_database_connection():
    """Mock database connection."""
    conn = Mock(spec=sqlite3.Connection)
    cur = Mock(spec=sqlite3.Cursor)
    conn.cursor.return_value = cur
    return conn


@pytest.fixture
def collector(mock_config):
    """Create TweetCollector instance with mocked dependencies."""
    with patch('fetcher.collector.get_config', return_value=mock_config), \
         patch('fetcher.collector.get_scroller') as mock_scroller, \
         patch('fetcher.collector.get_media_monitor') as mock_monitor, \
         patch('utils.paths.get_db_path', return_value=Mock()):

        mock_scroller.return_value = Mock()
        mock_monitor.return_value = Mock()

        collector = TweetCollector()
        return collector


class TestTweetCollector:
    """Test suite for TweetCollector class."""

    def test_init(self, collector, mock_config):
        """Test TweetCollector initialization."""
        assert collector.config == mock_config
        assert collector.scroller is not None
        assert collector.media_monitor is not None
        assert collector.db_path is not None

    def test_setup_media_url_monitoring(self, collector, mock_page):
        """Test media URL monitoring setup."""
        expected_urls = ['url1', 'url2']
        collector.media_monitor.setup_monitoring.return_value = expected_urls

        result = collector.setup_media_url_monitoring(mock_page)

        collector.media_monitor.setup_monitoring.assert_called_once_with(mock_page)
        assert result == expected_urls

    def test_should_process_tweet_new_tweet(self, collector):
        """Test should_process_tweet with new tweet."""
        seen_ids = {'existing1', 'existing2'}
        tweet_id = 'new_tweet'

        result = collector.should_process_tweet(tweet_id, seen_ids)

        assert result is True

    def test_should_process_tweet_already_seen(self, collector):
        """Test should_process_tweet with already seen tweet."""
        seen_ids = {'existing1', 'existing2'}
        tweet_id = 'existing1'

        result = collector.should_process_tweet(tweet_id, seen_ids)

        assert result is False

    def test_should_process_tweet_empty_id(self, collector):
        """Test should_process_tweet with empty tweet ID."""
        seen_ids = {'existing1', 'existing2'}
        tweet_id = ''

        result = collector.should_process_tweet(tweet_id, seen_ids)

        assert result is False

    def test_should_process_tweet_none_id(self, collector):
        """Test should_process_tweet with None tweet ID."""
        seen_ids = {'existing1', 'existing2'}
        tweet_id = None

        result = collector.should_process_tweet(tweet_id, seen_ids)

        assert result is False

    @patch('fetcher.collector.get_tweet_repository')
    def test_check_tweet_exists_in_db_no_resume(self, mock_repo, collector):
        """Test check_tweet_exists_in_db when not resuming."""
        result = collector.check_tweet_exists_in_db('username', 'tweet_id', False)

        assert result == (False, None)
        mock_repo.assert_not_called()

    @patch('fetcher.collector.get_tweet_repository')
    def test_check_tweet_exists_in_db_resume_not_found(self, mock_repo, collector):
        """Test check_tweet_exists_in_db when resuming but tweet not found."""
        mock_repo.return_value.get_tweet_by_id.return_value = None

        result = collector.check_tweet_exists_in_db('username', 'tweet_id', True)

        assert result == (False, None)
        mock_repo.return_value.get_tweet_by_id.assert_called_once_with('tweet_id')

    @patch('fetcher.collector.get_tweet_repository')
    def test_check_tweet_exists_in_db_resume_found(self, mock_repo, collector):
        """Test check_tweet_exists_in_db when resuming and tweet found."""
        mock_tweet = {
            'username': 'testuser',
            'post_type': 'original',
            'content': 'test content',
            'original_author': None,
            'original_tweet_id': None
        }
        mock_repo.return_value.get_tweet_by_id.return_value = mock_tweet

        result = collector.check_tweet_exists_in_db('testuser', 'tweet_id', True)

        assert result[0] is True  # exists
        assert result[1]['post_type'] == 'original'
        assert result[1]['content'] == 'test content'

    @patch('fetcher.collector.fetcher_db.save_tweet')
    def test_save_tweet_data_success(self, mock_save, collector, mock_database_connection):
        """Test successful tweet data saving."""
        mock_save.return_value = True
        tweet_data = {'tweet_id': '123', 'content': 'test'}

        result = collector.save_tweet_data(mock_database_connection, tweet_data)

        assert result is True
        mock_save.assert_called_once_with(mock_database_connection, tweet_data)

    @patch('fetcher.collector.fetcher_db.save_tweet')
    def test_save_tweet_data_failure(self, mock_save, collector, mock_database_connection):
        """Test tweet data saving failure."""
        mock_save.return_value = False
        tweet_data = {'tweet_id': '123', 'content': 'test'}

        result = collector.save_tweet_data(mock_database_connection, tweet_data)

        assert result is False

    @patch('fetcher.collector.sqlite3.connect')
    def test_log_processing_error(self, mock_connect, collector, mock_config):
        """Test processing error logging."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value = mock_cur
        mock_connect.return_value = mock_conn

        error = Exception("Test error")
        collector.log_processing_error('tweet123', 'testuser', error)

        mock_connect.assert_called_once_with(collector.db_path, timeout=mock_config.db_timeout)
        mock_cur.execute.assert_called_once()
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch('fetcher.collector.sqlite3.connect')
    def test_log_processing_error_no_tweet_id(self, mock_connect, collector, mock_config):
        """Test processing error logging without tweet ID."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value = mock_cur
        mock_connect.return_value = mock_conn

        error = Exception("Test error")
        collector.log_processing_error(None, 'testuser', error)

        # Should still log with None tweet_id
        call_args = mock_cur.execute.call_args[0]
        assert call_args[1][1] is None  # tweet_id parameter

    @patch('fetcher.collector.sqlite3.connect')
    def test_log_processing_error_db_failure(self, mock_connect, collector):
        """Test processing error logging when DB fails."""
        mock_connect.side_effect = Exception("Connection failed")

        error = Exception("Test error")
        # Should not raise exception
        collector.log_processing_error('tweet123', 'testuser', error)

    def test_collect_tweets_from_page_max_tweets_reached(self, collector, mock_page, mock_database_connection):
        """Test collection stops when max tweets reached."""
        mock_page.query_selector_all.return_value = []
        collector.media_monitor.setup_monitoring.return_value = []

        result = collector.collect_tweets_from_page(
            mock_page, 'testuser', 0, False, None, None, mock_database_connection
        )

        assert len(result) == 0

    def test_get_collector(self):
        """Test get_collector function returns TweetCollector instance."""
        collector = get_collector()
        assert isinstance(collector, TweetCollector)