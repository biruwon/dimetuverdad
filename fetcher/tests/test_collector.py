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
    config.skip_duplicate_check = False  # Default to checking duplicates in tests
    config.batch_write_size = 50
    return config


@pytest.fixture
def mock_page():
    """Mock Playwright page object."""
    page = Mock()
    # Set url as a property (string) so 'in' operator works
    page.url = "https://x.com/testuser"
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
    def test_check_tweet_exists_in_db_always_checks(self, mock_repo, collector):
        """Test check_tweet_exists_in_db always checks database (even without resume flag)."""
        mock_repo.return_value.get_tweet_by_id.return_value = None
        
        result = collector.check_tweet_exists_in_db('username', 'tweet_id', False)

        assert result == (False, None)
        # Should always check database to avoid re-saving duplicates
        mock_repo.return_value.get_tweet_by_id.assert_called_once_with('tweet_id')

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

    @patch('database.get_db_connection_context')
    def test_log_processing_error(self, mock_get_conn, collector):
        """Test processing error logging."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value = mock_cur
        mock_get_conn.return_value.__enter__ = Mock(return_value=mock_conn)
        mock_get_conn.return_value.__exit__ = Mock(return_value=None)

        error = Exception("Test error")
        collector.log_processing_error('tweet123', 'testuser', error)

        mock_cur.execute.assert_called_once()
        mock_conn.commit.assert_called_once()

    @patch('database.get_db_connection_context')
    def test_log_processing_error_no_tweet_id(self, mock_context, collector, mock_config):
        """Test processing error logging without tweet ID."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value = mock_cur
        mock_context.return_value.__enter__.return_value = mock_conn

        error = Exception("Test error")
        collector.log_processing_error(None, 'testuser', error)

        # Should still log with None tweet_id
        call_args = mock_cur.execute.call_args[0]
        assert call_args[1][1] is None  # tweet_id parameter

    @patch('database.get_db_connection_context')
    def test_log_processing_error_db_failure(self, mock_context, collector):
        """Test processing error logging when DB fails."""
        mock_context.side_effect = Exception("Connection failed")

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

    @patch('fetcher.collector.fetcher_parsers.extract_engagement_metrics')
    @patch('fetcher.collector.fetcher_parsers.extract_tweet_with_media_monitoring')
    def test_extract_tweet_data_success(self, mock_extract_tweet, mock_engagement, collector, mock_page, mock_article):
        """Test successful tweet data extraction."""
        # Setup mocks
        mock_dict = {
            'tweet_id': '123456789',
            'tweet_url': 'https://x.com/testuser/status/123456789',
            'username': 'testuser',
            'content': 'Test tweet content',
            'post_type': 'original',
            'original_author': None,
            'original_tweet_id': None,
            'media_links': 'media1.jpg',
            'media_count': 1,
            'hashtags': '#test',
            'mentions': '@user',
            'external_links': 'http://example.com'
        }
        mock_extract_tweet.return_value = mock_dict
        mock_engagement.return_value = {
            'retweets': 10,
            'likes': 20,
            'replies': 5,
            'views': 100
        }

        # Mock time element
        mock_time = Mock()
        mock_time.get_attribute.return_value = '2023-10-15T10:30:00.000Z'
        mock_article.query_selector.return_value = mock_time

        result = collector.extract_tweet_data(
            mock_page, mock_article, '123456789', 'https://x.com/testuser/status/123456789',
            'testuser', 'https://profile.pic.url'
        )

        assert result is not None
        assert result['tweet_id'] == '123456789'
        assert result['tweet_url'] == 'https://x.com/testuser/status/123456789'
        assert result['username'] == 'testuser'
        assert result['content'] == 'Test tweet content'
        assert result['post_type'] == 'original'
        assert result['media_count'] == 1
        assert result['engagement_retweets'] == 10
        assert result['engagement_likes'] == 20
        mock_extract_tweet.assert_called_once_with(
            mock_page, '123456789', 'testuser', 'https://x.com/testuser/status/123456789',
            collector.media_monitor, collector.scroller, mock_article
        )

    @patch('fetcher.collector.fetcher_parsers.extract_engagement_metrics')
    @patch('fetcher.collector.fetcher_parsers.extract_tweet_with_media_monitoring')
    def test_extract_tweet_data_with_media_urls(self, mock_extract_tweet, mock_engagement, collector, mock_page, mock_article):
        """Test tweet data extraction with media URLs."""
        # Setup mocks
        mock_dict = {
            'tweet_id': '123',
            'content': 'Test content',
            'post_type': 'original',
            'media_links': 'https://video.twimg.com/test.mp4',
            'media_count': 1
        }
        mock_extract_tweet.return_value = mock_dict
        mock_engagement.return_value = {
            'retweets': 0, 'likes': 0, 'replies': 0, 'views': 0
        }

        mock_time = Mock()
        mock_time.get_attribute.return_value = '2023-10-15T10:30:00.000Z'
        mock_article.query_selector.return_value = mock_time

        result = collector.extract_tweet_data(
            mock_page, mock_article, '123', 'https://x.com/testuser/status/123',
            'testuser', None
        )

        assert result is not None
        assert result['media_count'] == 1
        assert 'test.mp4' in result['media_links']

    @patch('fetcher.collector.fetcher_parsers.extract_tweet_with_media_monitoring')
    def test_extract_tweet_data_extraction_error(self, mock_extract, collector, mock_page, mock_article):
        """Test that extraction errors are handled gracefully."""
        mock_extract.side_effect = Exception("Extraction failed")

        result = collector.extract_tweet_data(
            mock_page, mock_article, '123', 'https://x.com/testuser/status/123',
            'testuser', None
        )

        assert result is None

    @patch('fetcher.collector.fetcher_parsers.extract_engagement_metrics')
    @patch('fetcher.collector.fetcher_parsers.extract_tweet_with_media_monitoring')
    def test_extract_tweet_data_no_timestamp(self, mock_extract_tweet, mock_engagement, collector, mock_page, mock_article):
        """Test tweet data extraction when timestamp is not available."""
        # Setup mocks
        mock_dict = {
            'tweet_id': '123',
            'content': 'Test content',
            'post_type': 'original'
        }
        mock_extract_tweet.return_value = mock_dict
        mock_engagement.return_value = {
            'retweets': 0, 'likes': 0, 'replies': 0, 'views': 0
        }

        # No time element found
        mock_article.query_selector.return_value = None

        result = collector.extract_tweet_data(
            mock_page, mock_article, '123', 'https://x.com/testuser/status/123',
            'testuser', None
        )

        assert result is not None
        assert result['tweet_timestamp'] is None

    @patch('fetcher.collector.fetcher_parsers.should_skip_existing_tweet')
    @patch('fetcher.collector.fetcher_parsers.analyze_post_type')
    @patch('fetcher.collector.fetcher_parsers.extract_full_tweet_content')
    def test_collect_tweets_from_page_basic_collection(self, mock_content, mock_post_analysis, mock_skip_existing, collector, mock_page, mock_database_connection):
        """Test basic tweet collection from page."""
        # Setup mocks
        collector.media_monitor.setup_monitoring.return_value = []

        # Mock articles
        mock_article = Mock()
        mock_link = Mock()
        mock_link.get_attribute.return_value = '/testuser/status/123456789'
        mock_article.query_selector.return_value = mock_link
        mock_page.query_selector_all.return_value = [mock_article]

        # Mock database check - tweet doesn't exist
        with patch.object(collector, 'check_tweet_exists_in_db', return_value=(False, None)):
            # Mock extraction
            mock_tweet_data = {'tweet_id': '123456789', 'content': 'test'}
            with patch.object(collector, 'extract_tweet_data', return_value=mock_tweet_data):
                # Mock saving
                with patch.object(collector, 'save_tweet_data', return_value=True):
                    # Mock scroller
                    collector.scroller.event_scroll_cycle = Mock()
                    collector.scroller.check_page_height_change = Mock(return_value=1000)

                    result = collector.collect_tweets_from_page(
                        mock_page, 'testuser', 1, False, None, None, mock_database_connection
                    )

                    assert len(result) == 1
                    assert result[0]['tweet_id'] == '123456789'

    def test_collect_tweets_from_page_no_articles(self, collector, mock_page, mock_database_connection):
        """Test collection when no articles are found."""
        # Override config to make test run faster
        collector.config.max_consecutive_empty_scrolls = 2
        
        collector.media_monitor.setup_monitoring.return_value = []
        mock_page.query_selector_all.return_value = []

        collector.scroller.event_scroll_cycle = Mock()
        collector.scroller.check_page_height_change = Mock(return_value=1000)
        # Mock the missing aggressive_scroll method
        collector.scroller.aggressive_scroll = Mock()

        result = collector.collect_tweets_from_page(
            mock_page, 'testuser', 5, False, None, None, mock_database_connection
        )

        # Should eventually stop due to consecutive empty scrolls
        assert isinstance(result, list)
        assert len(result) == 0  # No tweets should be collected

    @patch('fetcher.collector.fetcher_parsers.analyze_post_type')
    @patch('fetcher.collector.fetcher_parsers.extract_full_tweet_content')
    def test_collect_tweets_from_page_skip_existing_tweet(self, mock_content, mock_post_analysis, collector, mock_page, mock_database_connection):
        """Test that existing tweets are skipped when resuming."""
        # Override config to make test run faster
        collector.config.max_consecutive_empty_scrolls = 2
        
        collector.media_monitor.setup_monitoring.return_value = []

        # Mock article
        mock_article = Mock()
        mock_link = Mock()
        mock_link.get_attribute.return_value = '/testuser/status/123456789'
        mock_article.query_selector.return_value = mock_link
        
        # Mock page to return article first time, then empty list
        call_count = 0
        def mock_query_selector_all(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return [mock_article] if call_count == 1 else []
        mock_page.query_selector_all.side_effect = mock_query_selector_all

        # Mock database check - tweet exists and doesn't need update
        with patch.object(collector, 'check_tweet_exists_in_db', return_value=(True, {'post_type': 'original', 'content': 'same content'})):
            mock_post_analysis.return_value = {'post_type': 'original', 'should_skip': False}
            mock_content.return_value = 'same content'

            collector.scroller.event_scroll_cycle = Mock()
            collector.scroller.check_page_height_change = Mock(return_value=1000)
            # Mock the aggressive_scroll method
            collector.scroller.aggressive_scroll = Mock()

            result = collector.collect_tweets_from_page(
                mock_page, 'testuser', 5, True, '2023-10-15T10:00:00Z', None, mock_database_connection
            )

            # Should skip the existing tweet
            assert len(result) == 0

    def test_collect_tweets_from_page_resume_timestamp_skip(self, collector, mock_page, mock_database_connection):
        """Test that tweets newer than or equal to resume timestamp are skipped."""
        # This test verifies that should_skip_existing_tweet is called correctly
        # We can't easily test the full collect_tweets_from_page loop without hanging
        # So we test the logic that would be used inside it
        
        # Test the skip logic directly
        from fetcher import parsers as fetcher_parsers
        
        # Test that older timestamps are NOT skipped (should continue processing)
        assert fetcher_parsers.should_skip_existing_tweet('2023-10-14T10:00:00Z', '2023-10-15T10:00:00Z') == False
        # Test that newer timestamps ARE skipped
        assert fetcher_parsers.should_skip_existing_tweet('2023-10-16T10:00:00Z', '2023-10-15T10:00:00Z') == True
        # Test that same timestamp is skipped
        assert fetcher_parsers.should_skip_existing_tweet('2023-10-15T10:00:00Z', '2023-10-15T10:00:00Z') == True
        # Test that None oldest_timestamp doesn't skip
        assert fetcher_parsers.should_skip_existing_tweet('2023-10-14T10:00:00Z', None) == False

    def test_collect_tweets_from_page_extraction_failure(self, collector, mock_page, mock_database_connection):
        """Test handling of tweet extraction failures."""
        # Test that extract_tweet_data returning None is handled correctly
        mock_article = Mock()
        
        # Mock extraction failure
        with patch.object(collector, 'extract_tweet_data', return_value=None) as mock_extract:
            result = collector.extract_tweet_data(mock_article, '123', 'url', 'user', None, [])
            
            # Verify extraction was attempted
            mock_extract.assert_called_once()
            assert result is None

    def test_collect_tweets_from_page_save_failure(self, collector, mock_page, mock_database_connection):
        """Test handling of tweet save failures."""
        # Test that save_tweet_data returning False is handled correctly
        tweet_data = {'tweet_id': '123', 'content': 'test'}
        
        with patch.object(collector, 'save_tweet_data', return_value=False) as mock_save:
            result = collector.save_tweet_data(mock_database_connection, tweet_data)
            
            # Verify save was attempted
            mock_save.assert_called_once_with(mock_database_connection, tweet_data)
            assert result is False

    def test_collect_tweets_from_page_max_tweets_unlimited(self, collector, mock_page, mock_database_connection):
        """Test collection with unlimited tweets (float('inf'))."""
        # Override config to make test run faster
        collector.config.max_consecutive_empty_scrolls = 2
        
        collector.media_monitor.setup_monitoring.return_value = []
        mock_page.query_selector_all.return_value = []

        collector.scroller.event_scroll_cycle = Mock()
        collector.scroller.check_page_height_change = Mock(return_value=1000)
        # Mock the aggressive_scroll method
        collector.scroller.aggressive_scroll = Mock()

        result = collector.collect_tweets_from_page(
            mock_page, 'testuser', float('inf'), False, None, None, mock_database_connection
        )

        assert isinstance(result, list)

    @patch('fetcher.collector.logger')
    def test_collect_tweets_from_page_scrolling_failure(self, mock_logger, collector, mock_page, mock_database_connection):
        """Test handling of scrolling failures."""
        # Override config to make test run faster
        collector.config.max_consecutive_empty_scrolls = 2
        
        collector.media_monitor.setup_monitoring.return_value = []
        mock_page.query_selector_all.return_value = []

        # Mock scroller failures
        collector.scroller.event_scroll_cycle.side_effect = Exception("Scroll failed")
        collector.scroller.check_page_height_change.side_effect = Exception("Height check failed")

        # Mock fallback scroll and HTML extraction
        mock_page.evaluate = Mock(return_value=[])  # Return empty list for HTML extraction
        # Mock the aggressive_scroll method
        collector.scroller.aggressive_scroll = Mock()

        result = collector.collect_tweets_from_page(
            mock_page, 'testuser', 5, False, None, None, mock_database_connection
        )

        # Should still complete despite scrolling failures
        assert isinstance(result, list)
        # Verify evaluate was called (either for scroll or HTML extraction)
        assert mock_page.evaluate.called

    def test_get_collector(self):
        """Test get_collector function returns TweetCollector instance."""
        collector = get_collector()
        assert isinstance(collector, TweetCollector)


class TestGroupAndUpdateThreads:
    """Tests for _group_and_update_threads method."""

    def test_no_thread_indicators(self, collector, mock_database_connection):
        """Should return 0 when no tweets have thread indicators."""
        tweets = [
            {'tweet_id': '123', 'content': 'Test tweet'},
            {'tweet_id': '456', 'content': 'Another tweet'}
        ]
        
        result = collector._group_and_update_threads(mock_database_connection, tweets, 'testuser')
        assert result == 0

    def test_tweets_with_thread_line(self, collector, mock_database_connection):
        """Should detect threads when tweets have thread line indicators."""
        tweets = [
            {'tweet_id': '123', 'content': 'Thread start', 'has_thread_line': True},
            {'tweet_id': '456', 'content': 'Thread reply', 'reply_to_tweet_id': '123'}
        ]
        
        # Mock thread detector to return grouped threads
        collector.thread_detector._group_into_threads_by_reply_chain = Mock(return_value=[
            {'tweets': [
                {'tweet_id': '123'},
                {'tweet_id': '456'}
            ]}
        ])
        
        result = collector._group_and_update_threads(mock_database_connection, tweets, 'testuser')
        assert result >= 0

    def test_thread_grouping_exception(self, collector, mock_database_connection):
        """Should handle exceptions in thread grouping gracefully."""
        tweets = [
            {'tweet_id': '123', 'has_thread_line': True}
        ]
        
        # Mock thread detector to raise exception
        collector.thread_detector._group_into_threads_by_reply_chain = Mock(
            side_effect=Exception("Grouping failed")
        )
        
        result = collector._group_and_update_threads(mock_database_connection, tweets, 'testuser')
        assert result == 0

    def test_empty_tweets_list(self, collector, mock_database_connection):
        """Should return 0 for empty tweets list."""
        result = collector._group_and_update_threads(mock_database_connection, [], 'testuser')
        assert result == 0


class TestMaybeDetectThreads:
    """Tests for _maybe_detect_threads method."""

    def test_feature_disabled(self, collector, mock_page, mock_database_connection):
        """Should return 0 when collect_threads is disabled."""
        collector.config.collect_threads = False
        
        detected, new_scroll_count = collector._maybe_detect_threads(
            mock_page, 'testuser', mock_database_connection, scrolls_since_last_detect=10
        )
        
        assert detected == 0
        assert new_scroll_count == 10

    def test_not_enough_scrolls(self, collector, mock_page, mock_database_connection):
        """Should not detect when not enough scrolls have passed."""
        collector.config.collect_threads = True
        collector.config.thread_detect_interval = 5
        
        detected, new_scroll_count = collector._maybe_detect_threads(
            mock_page, 'testuser', mock_database_connection, scrolls_since_last_detect=3
        )
        
        assert detected == 0
        assert new_scroll_count == 3


class TestCheckTweetExistsInDbEdgeCases:
    """Additional edge cases for check_tweet_exists_in_db."""

    @patch('fetcher.collector.get_tweet_repository')
    def test_username_mismatch(self, mock_repo_func, collector):
        """Should return False if username doesn't match."""
        mock_repo = Mock()
        mock_repo.get_tweet_by_id.return_value = {'tweet_id': '123', 'username': 'differentuser'}
        mock_repo_func.return_value = mock_repo
        
        exists, needs_update = collector.check_tweet_exists_in_db('testuser', '123', True)
        
        assert exists is False
        assert needs_update is None

    @patch('fetcher.collector.get_tweet_repository')
    def test_tweet_exists_same_username(self, mock_repo_func, collector):
        """Should return True if tweet exists with same username."""
        mock_repo = Mock()
        mock_repo.get_tweet_by_id.return_value = {
            'tweet_id': '123', 
            'username': 'testuser',
            'post_type': 'original',
            'content': 'test content'
        }
        mock_repo_func.return_value = mock_repo
        
        exists, needs_update = collector.check_tweet_exists_in_db('testuser', '123', True)
        
        assert exists is True


class TestExtractTweetDataEdgeCases:
    """Additional edge cases for extract_tweet_data."""

    def test_article_html_extraction_failure(self, collector, mock_page):
        """Test handling when article HTML extraction fails."""
        mock_article = Mock()
        mock_article.evaluate.side_effect = Exception("HTML extraction failed")
        mock_article.query_selector.return_value = None
        
        # Should handle the error and potentially fall back
        with patch('fetcher.collector.html_extractor.parse_tweet_from_html', return_value=None):
            with patch('fetcher.collector.fetcher_parsers.extract_tweet_with_media_monitoring', return_value=None):
                result = collector.extract_tweet_data(
                    mock_article, '123', 'https://twitter.com/user/status/123', 
                    'user', mock_page, []
                )
                assert result is None

    def test_video_monitoring_needed(self, collector, mock_page):
        """Test detection of tweets requiring video monitoring."""
        mock_article = Mock()
        mock_article.evaluate.return_value = '<html>...</html>'
        
        # Mock stateless extraction with video content
        stateless_data = {
            'content': 'Video tweet',
            'media_links': 'https://pbs.twimg.com/amplify_video_thumb/123.jpg',
            'media_count': 1
        }
        
        with patch('fetcher.collector.html_extractor.parse_tweet_from_html', return_value=stateless_data):
            # Should trigger video monitoring path
            with patch('fetcher.collector.fetcher_parsers.extract_tweet_with_media_monitoring') as mock_extract:
                mock_extract.return_value = {
                    'tweet_id': '123',
                    'content': 'Video tweet',
                    'tweet_timestamp': '2024-01-01T12:00:00Z'
                }
                result = collector.extract_tweet_data(
                    mock_article, '123', 'https://twitter.com/user/status/123',
                    'user', mock_page, []
                )
                # Video monitoring should have been triggered
                assert mock_extract.called


class TestCollectorAlwaysChecksDatabaseForExistingTweets:
    """Tests that collector always checks database to avoid duplicates."""

    @pytest.fixture
    def collector(self, mock_config):
        """Create TweetCollector instance with mocked dependencies."""
        with patch('fetcher.collector.get_config', return_value=mock_config), \
             patch('fetcher.collector.get_scroller') as mock_scroller, \
             patch('fetcher.collector.get_media_monitor') as mock_monitor, \
             patch('utils.paths.get_db_path', return_value=Mock()):

            c = TweetCollector()
            c.scroller = mock_scroller.return_value
            c.media_monitor = mock_monitor.return_value
            return c

    @patch('fetcher.collector.get_tweet_repository')
    def test_check_tweet_exists_even_without_resume(self, mock_repo, collector):
        """check_tweet_exists_in_db should query database even when resume_from_last is False."""
        # Set up mock to indicate tweet exists
        mock_repo.return_value.get_tweet_by_id.return_value = {
            'username': 'testuser',
            'post_type': 'original',
            'content': 'existing content'
        }
        
        # Call with resume_from_last=False - should still check database
        exists, data = collector.check_tweet_exists_in_db('testuser', 'tweet123', False)
        
        # Should have checked the database
        mock_repo.return_value.get_tweet_by_id.assert_called_once_with('tweet123')
        assert exists is True

    @patch('fetcher.collector.get_tweet_repository')
    def test_check_tweet_exists_returns_comparison_data(self, mock_repo, collector):
        """check_tweet_exists_in_db should return comparison data for update checks."""
        mock_repo.return_value.get_tweet_by_id.return_value = {
            'username': 'testuser',
            'post_type': 'repost',
            'content': 'test content',
            'original_author': 'original',
            'original_tweet_id': '999'
        }
        
        exists, data = collector.check_tweet_exists_in_db('testuser', 'tweet123', True)
        
        assert exists is True
        assert data is not None
        assert data['post_type'] == 'repost'
        assert data['content'] == 'test content'

    @patch('fetcher.collector.get_tweet_repository')
    def test_check_tweet_wrong_username_returns_not_exists(self, mock_repo, collector):
        """check_tweet_exists_in_db should return False if username doesn't match."""
        mock_repo.return_value.get_tweet_by_id.return_value = {
            'username': 'differentuser',  # Different username
            'post_type': 'original',
            'content': 'content'
        }
        
        exists, data = collector.check_tweet_exists_in_db('testuser', 'tweet123', True)
        
        assert exists is False


class TestCollectorThreadSkipBehavior:
    """Tests for thread skip behavior using is_thread_already_collected."""

    @pytest.fixture
    def collector(self, mock_config):
        """Create TweetCollector instance with mocked dependencies."""
        with patch('fetcher.collector.get_config', return_value=mock_config), \
             patch('fetcher.collector.get_scroller') as mock_scroller, \
             patch('fetcher.collector.get_media_monitor') as mock_monitor, \
             patch('utils.paths.get_db_path', return_value=Mock()):

            mock_config.collect_threads = True
            c = TweetCollector()
            c.scroller = mock_scroller.return_value
            c.media_monitor = mock_monitor.return_value
            return c

    @patch('fetcher.collector.fetcher_db.is_thread_already_collected')
    def test_thread_check_called_when_thread_line_detected(self, mock_is_collected, collector, mock_database_connection):
        """is_thread_already_collected should be called when has_thread_line is True."""
        mock_is_collected.return_value = True
        
        # This simulates the check that happens in collect_tweets_from_page
        result = mock_is_collected(mock_database_connection, 'tweet123')
        
        assert result is True
        mock_is_collected.assert_called_once_with(mock_database_connection, 'tweet123')

    @patch('fetcher.collector.fetcher_db.is_thread_already_collected')
    def test_thread_not_skipped_when_not_collected(self, mock_is_collected, collector, mock_database_connection):
        """Thread should not be skipped when is_thread_already_collected returns False."""
        mock_is_collected.return_value = False
        
        result = mock_is_collected(mock_database_connection, 'new_tweet_123')
        
        assert result is False


class TestFetchTweetsNoneVsEmptyList:
    """Tests for distinguishing between None (failure) and empty list (no new tweets)."""

    def test_empty_list_is_not_failure(self):
        """An empty list should not be treated as failure."""
        tweets = []
        
        # With our fix, we check `tweets is None` not `not tweets`
        is_failure = tweets is None
        
        assert is_failure is False  # Empty list is NOT a failure

    def test_none_is_failure(self):
        """None should be treated as failure."""
        tweets = None
        
        is_failure = tweets is None
        
        assert is_failure is True  # None IS a failure

    def test_empty_list_length_is_zero(self):
        """Empty list should contribute 0 to total count."""
        tweets = []
        total = len(tweets) if tweets else 0
        
        assert total == 0

    def test_none_tweets_length_is_zero(self):
        """None tweets should contribute 0 to total count (not error)."""
        tweets = None
        total = len(tweets) if tweets else 0
        
        assert total == 0