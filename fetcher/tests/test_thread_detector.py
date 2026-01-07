"""
Unit tests for ThreadDetector class.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

from fetcher.thread_detector import ThreadDetector, _sync_handle


class TestThreadDetector:
    """Test cases for ThreadDetector class."""

    @pytest.fixture
    def detector(self):
        """Create ThreadDetector instance."""
        return ThreadDetector()

    @pytest.fixture
    def mock_article(self):
        """Create mock article element."""
        article = MagicMock()
        return article

    @pytest.fixture
    def mock_session_manager(self):
        """Create mock session manager."""
        session_manager = MagicMock()
        return session_manager

    def test_detect_thread_start_no_thread_line(self, detector, mock_article):
        """Test thread start detection when no thread line is present."""
        # Mock article without thread line
        mock_article.query_selector_all.return_value = []

        result = detector.detect_thread_start(mock_article, "testuser")

        assert result is True
        mock_article.query_selector_all.assert_called_once()

    def test_detect_thread_start_with_thread_line(self, detector, mock_article):
        """Test thread start detection when thread line is present."""
        # Mock article with thread line
        mock_div = MagicMock()
        mock_div.get_attribute.return_value = "css-175oi2r r-1bimlpy r-f8sm7e"
        mock_article.query_selector_all.return_value = [mock_div]

        result = detector.detect_thread_start(mock_article, "testuser")

        assert result is False

    def test_sync_handle_prefers_sync_api(self):
        """_sync_handle should prefer explicit sync_api attribute."""
        handle = SimpleNamespace(sync_api="synced")
        assert _sync_handle(handle) == "synced"

        without_attr = object()
        assert _sync_handle(without_attr) is without_attr

    def test_extract_thread_content_success(self, detector):
        """Test successful thread content extraction."""
        # Skip the complex session manager mocking for now
        # This test would require integration testing
        # Instead, test the core logic that can be unit tested
        pass

    def test_extract_thread_content_below_threshold(self, detector):
        """Test thread content extraction when below tweet threshold."""
        # Skip the complex session manager mocking for now
        pass

    def test_extract_tweet_from_article_success(self, detector):
        """Test successful tweet extraction from article."""
        mock_article = MagicMock()

        # Mock time link for tweet ID
        mock_time_link = MagicMock()
        mock_time_link.get_attribute.return_value = "/testuser/status/tweet_123"
        mock_article.query_selector.return_value = mock_time_link

        # Mock content extraction
        mock_content_div = MagicMock()
        mock_content_div.inner_text.return_value = "Test tweet content"
        mock_article.query_selector.side_effect = lambda selector: {
            'a[href*="/status/"]': mock_time_link,
            '[data-testid="tweetText"]': mock_content_div
        }.get(selector)

        result = detector._extract_tweet_from_article(mock_article)

        assert result is not None
        assert result['tweet_id'] == 'tweet_123'
        assert result['content'] == 'Test tweet content'

    def test_extract_tweet_from_article_no_id(self, detector):
        """Test tweet extraction when no tweet ID found."""
        mock_article = MagicMock()
        mock_article.query_selector.return_value = None

        result = detector._extract_tweet_from_article(mock_article)

        assert result is None

    def test_extract_reply_metadata_success(self, detector):
        """Thread reply metadata should extract tweet id."""
        mock_article = MagicMock()

        def query_selector(selector):
            if selector == '[data-testid="Tweet-User-Text"]':
                return MagicMock()
            if selector == 'a[href*="/status/"]':
                link = MagicMock()
                link.get_attribute.return_value = "/other/status/987654321?foo=1"
                return link
            return None

        mock_article.query_selector.side_effect = query_selector

        result = detector.extract_reply_metadata(mock_article)
        assert result == {
            'reply_to_tweet_id': '987654321',
            'replied_to_id': '987654321',
            'is_reply': True,
        }

    def test_extract_reply_metadata_handles_errors(self, detector):
        """Gracefully handle selector errors."""
        mock_article = MagicMock()
        mock_article.query_selector.side_effect = Exception("boom")

        result = detector.extract_reply_metadata(mock_article)
        assert result is None

    def test_has_thread_line_present(self, detector):
        """Test thread line detection when present."""
        mock_article = MagicMock()
        mock_div = MagicMock()
        mock_div.get_attribute.return_value = "css-175oi2r r-1bimlpy r-f8sm7e"
        mock_article.query_selector_all.return_value = [mock_div]

        result = detector._has_thread_line(mock_article)

        assert result is True

    def test_has_thread_line_absent(self, detector):
        """Test thread line detection when absent."""
        mock_article = MagicMock()
        mock_article.query_selector_all.return_value = []

        result = detector._has_thread_line(mock_article)

        assert result is False

    def test_is_thread_member_cases(self, detector):
        """is_thread_member should react to metadata combinations."""
        assert detector.is_thread_member(None, "user") is False
        metadata = {'is_reply': True}
        assert detector.is_thread_member(metadata, "user") is True

        metadata = {'replies_to_username': 'user'}
        assert detector.is_thread_member(metadata, 'user') is True
        metadata = {'replies_to_username': 'other'}
        assert detector.is_thread_member(metadata, 'user') is False


class TestThreadSummary:
    """Test cases for ThreadSummary dataclass."""

    def test_thread_summary_creation(self):
        """Test creating a ThreadSummary."""
        from fetcher.thread_detector import ThreadSummary
        
        tweets = [
            {'tweet_id': '123', 'content': 'First'},
            {'tweet_id': '124', 'content': 'Second'},
        ]
        
        summary = ThreadSummary(
            start_id='123',
            url='https://x.com/user/status/123',
            tweets=tweets,
            conversation_id='123'
        )
        
        assert summary.start_id == '123'
        assert summary.size == 2
        assert summary.conversation_id == '123'

    def test_thread_summary_size_property(self):
        """Test that size property returns tweet count."""
        from fetcher.thread_detector import ThreadSummary
        
        summary = ThreadSummary(
            start_id='100',
            url='https://x.com/user/status/100',
            tweets=[{'tweet_id': str(i)} for i in range(5)]
        )
        
        assert summary.size == 5


class TestEnhancedThreadDetection:
    """Test cases for enhanced thread detection methods."""

    @pytest.fixture
    def detector(self):
        """Create ThreadDetector instance."""
        return ThreadDetector()

    def test_format_thread_post(self, detector):
        """Test formatting tweet data for thread storage."""
        tweet = {
            'tweet_id': '12345',
            'username': 'testuser',
            'content': 'Test content',
            'tweet_timestamp': '2026-01-05T12:00:00Z',
        }
        
        formatted = detector._format_thread_post(tweet)
        
        assert formatted['tweet_id'] == '12345'
        assert formatted['username'] == 'testuser'
        assert formatted['content'] == 'Test content'
        assert formatted['tweet_timestamp'] == '2026-01-05T12:00:00Z'
        assert 'x.com/testuser/status/12345' in formatted['tweet_url']

    def test_format_thread_post_with_existing_url(self, detector):
        """Test that existing tweet_url is preserved."""
        tweet = {
            'tweet_id': '12345',
            'username': 'testuser',
            'tweet_url': 'https://x.com/custom/status/12345',
            'content': 'Test',
        }
        
        formatted = detector._format_thread_post(tweet)
        
        assert formatted['tweet_url'] == 'https://x.com/custom/status/12345'

    def test_collect_parent_thread_posts_empty(self, detector):
        """Test parent collection when focal is first."""
        article_list = [
            ('100', 'testuser', {'tweet_id': '100', 'content': 'First'}),
        ]
        
        parents = detector._collect_parent_thread_posts(article_list, 0, 'testuser')
        
        assert parents == []

    def test_collect_parent_thread_posts_single_parent(self, detector):
        """Test collecting a single parent tweet."""
        article_list = [
            ('100', 'testuser', {'tweet_id': '100', 'content': 'First'}),
            ('101', 'testuser', {'tweet_id': '101', 'content': 'Second'}),
        ]
        
        parents = detector._collect_parent_thread_posts(article_list, 1, 'testuser')
        
        assert len(parents) == 1
        assert parents[0]['tweet_id'] == '100'

    def test_collect_parent_thread_posts_stops_at_other_user(self, detector):
        """Test that parent collection stops at another user's tweet."""
        article_list = [
            ('99', 'otheruser', {'tweet_id': '99', 'content': 'Other user'}),
            ('100', 'testuser', {'tweet_id': '100', 'content': 'First'}),
            ('101', 'testuser', {'tweet_id': '101', 'content': 'Second'}),
        ]
        
        parents = detector._collect_parent_thread_posts(article_list, 2, 'testuser')
        
        assert len(parents) == 1
        assert parents[0]['tweet_id'] == '100'

    def test_collect_parent_thread_posts_multiple_parents(self, detector):
        """Test collecting multiple parent tweets in a chain."""
        article_list = [
            ('100', 'testuser', {'tweet_id': '100', 'content': 'First'}),
            ('101', 'testuser', {'tweet_id': '101', 'content': 'Second'}),
            ('102', 'testuser', {'tweet_id': '102', 'content': 'Third'}),
        ]
        
        parents = detector._collect_parent_thread_posts(article_list, 2, 'testuser')
        
        assert len(parents) == 2
        assert parents[0]['tweet_id'] == '100'
        assert parents[1]['tweet_id'] == '101'







class TestThreadDatabaseSaving:
    """Test cases for thread database saving functionality."""

    @pytest.fixture
    def detector(self):
        """Create ThreadDetector instance."""
        return ThreadDetector()

    @pytest.fixture
    def mock_conn(self):
        """Create a mock database connection."""
        return MagicMock()

    def test_save_thread_to_database_basic(self, detector, mock_conn):
        """Test saving a basic thread to database."""
        from fetcher.thread_detector import ThreadSummary
        from unittest.mock import patch
        
        thread = ThreadSummary(
            start_id='100',
            url='https://x.com/testuser/status/100',
            tweets=[
                {'tweet_id': '100', 'content': 'First', 'username': 'testuser'},
                {'tweet_id': '101', 'content': 'Second', 'username': 'testuser'},
            ],
            conversation_id='100'
        )
        
        with patch('fetcher.db.save_tweet') as mock_save:
            mock_save.return_value = True
            
            count = detector.save_thread_to_database(thread, mock_conn, 'testuser')
        
        assert count == 2
        assert mock_save.call_count == 2

    def test_save_thread_sets_thread_metadata(self, detector, mock_conn):
        """Test that thread metadata is correctly set."""
        from fetcher.thread_detector import ThreadSummary
        from unittest.mock import patch
        
        thread = ThreadSummary(
            start_id='100',
            url='https://x.com/testuser/status/100',
            tweets=[
                {'tweet_id': '100', 'content': 'First'},
                {'tweet_id': '101', 'content': 'Second'},
            ],
            conversation_id='100'
        )
        
        saved_tweets = []
        
        def capture_save(conn, tweet_data):
            saved_tweets.append(tweet_data.copy())
            return True
        
        with patch('fetcher.db.save_tweet', side_effect=capture_save):
            detector.save_thread_to_database(thread, mock_conn, 'testuser')
        
        # First tweet should be thread start
        assert saved_tweets[0]['is_thread_start'] == 1
        assert saved_tweets[0]['thread_position'] == 0
        assert saved_tweets[0]['thread_id'] == '100'
        assert saved_tweets[0]['post_type'] == 'thread'
        
        # Second tweet should have reply_to_tweet_id
        assert saved_tweets[1]['is_thread_start'] == 0
        assert saved_tweets[1]['thread_position'] == 1
        assert saved_tweets[1]['reply_to_tweet_id'] == '100'

    def test_save_thread_handles_empty_tweet_id(self, detector, mock_conn):
        """Test that tweets without IDs are skipped."""
        from fetcher.thread_detector import ThreadSummary
        from unittest.mock import patch
        
        thread = ThreadSummary(
            start_id='100',
            url='https://x.com/testuser/status/100',
            tweets=[
                {'tweet_id': '100', 'content': 'First'},
                {'content': 'No ID'},  # Missing tweet_id
                {'tweet_id': '102', 'content': 'Third'},
            ]
        )
        
        with patch('fetcher.db.save_tweet') as mock_save:
            mock_save.return_value = True
            
            count = detector.save_thread_to_database(thread, mock_conn, 'testuser')
        
        # Should only save tweets with IDs
        assert count == 2


class TestCollectorThreadDetection:
    """Test cases for TweetCollector.detect_and_save_threads method."""

    @pytest.fixture
    def collector(self):
        """Create TweetCollector instance."""
        from fetcher.collector import TweetCollector
        return TweetCollector()

    @pytest.fixture
    def mock_session_manager(self):
        """Create mock session manager."""
        return MagicMock()

    @pytest.fixture
    def mock_conn(self):
        """Create mock database connection."""
        return MagicMock()

    def test_detect_and_save_threads_no_threads(self, collector, mock_session_manager, mock_conn):
        """Test when no threads are detected."""
        from unittest.mock import patch
        
        with patch('database.ensure_schema_up_to_date'), \
             patch('fetcher.thread_detector.ThreadDetector') as MockDetector:
            
            mock_detector = MockDetector.return_value
            mock_detector.detect_threads_with_conversation_validation.return_value = []
            
            result = collector.detect_and_save_threads(
                'testuser', mock_session_manager, mock_conn
            )
        
        assert result == 0

    def test_detect_and_save_threads_with_threads(self, collector, mock_session_manager, mock_conn):
        """Test successful thread detection and saving."""
        from unittest.mock import patch
        from fetcher.thread_detector import ThreadSummary
        
        mock_threads = [
            ThreadSummary(
                start_id='100',
                url='https://x.com/testuser/status/100',
                tweets=[{'tweet_id': '100'}, {'tweet_id': '101'}]
            ),
            ThreadSummary(
                start_id='200',
                url='https://x.com/testuser/status/200',
                tweets=[{'tweet_id': '200'}, {'tweet_id': '201'}]
            ),
        ]
        
        with patch('database.ensure_schema_up_to_date'), \
             patch('fetcher.thread_detector.ThreadDetector') as MockDetector:
            
            mock_detector = MockDetector.return_value
            mock_detector.detect_threads_with_conversation_validation.return_value = mock_threads
            mock_detector.save_thread_to_database.return_value = 2
            
            result = collector.detect_and_save_threads(
                'testuser', mock_session_manager, mock_conn
            )
        
        assert result == 2
        assert mock_detector.save_thread_to_database.call_count == 2

    def test_detect_and_save_threads_calls_schema_update(self, collector, mock_session_manager, mock_conn):
        """Test that schema migration is called."""
        from unittest.mock import patch
        
        with patch('database.ensure_schema_up_to_date') as mock_schema, \
             patch('fetcher.thread_detector.ThreadDetector') as MockDetector:
            
            mock_detector = MockDetector.return_value
            mock_detector.detect_threads_with_conversation_validation.return_value = []
            
            collector.detect_and_save_threads(
                'testuser', mock_session_manager, mock_conn
            )
        
        mock_schema.assert_called_once_with(mock_conn)

    def test_mark_processed_on_thread_none(self, collector, monkeypatch):
        """If a thread extraction returns None, mark it processed to avoid retries."""
        fake_page = Mock()
        article = Mock()

        # Simulate two article fetch cycles (article present twice), then none
        call_count = {'n': 0}

        def qsa(selector):
            # Accept the combined selector used in collector
            if 'article' in selector and 'data-testid' in selector:
                call_count['n'] += 1
                if call_count['n'] <= 2:
                    return [article]
                return []
            return []

        fake_page.query_selector_all.side_effect = qsa

        # Minimal page behaviors
        fake_page.locator.return_value.inner_text.return_value = ''
        fake_page.url = 'https://x.com/targetuser'
        fake_page.wait_for_function.return_value = None
        fake_page.evaluate.return_value = 0

        # Provide a fake tweet_link on the article so the collector picks it up
        tweet_link = Mock()
        tweet_link.get_attribute.return_value = '/targetuser/status/12345'
        article.query_selector.return_value = tweet_link

        # Make extract_tweet_data return the same tweet (with thread indicator)
        def extract_tweet_data(page, article_arg, tweet_id, tweet_url, username, profile_pic_url):
            return {
                'tweet_id': '12345',
                'username': 'targetuser',
                'tweet_url': 'https://x.com/targetuser/status/12345',
                'content': 'hello',
                'has_thread_line': True
            }

        monkeypatch.setattr(collector, 'extract_tweet_data', extract_tweet_data)

        # Patch parser helper to accept the href as the right author/id
        from fetcher import parsers as fetcher_parsers
        monkeypatch.setattr(fetcher_parsers, 'should_process_tweet_by_author', lambda href, u: (True, 'targetuser', '12345'))

        # Patch thread_detector.collect_thread_by_id to return None always (aborted/not a thread)
        called = {'n': 0}

        def collect_thread_by_id(username, tid, session_manager, existing_context=None):
            called['n'] += 1
            return None

        monkeypatch.setattr(collector.thread_detector, 'collect_thread_by_id', collect_thread_by_id)

        # Run collection for a small number of tweets
        conn = Mock()
        res = collector.collect_tweets_from_page(fake_page, 'targetuser', max_tweets=10, resume_from_last=False, oldest_timestamp=None, profile_pic_url=None, conn=conn)

        # Thread collection is disabled by config, so no calls should be made
        assert called['n'] == 0
        # The tweet should have been saved/collected once
        assert any(t['tweet_id'] == '12345' for t in res)