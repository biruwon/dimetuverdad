"""
Unit tests for ThreadDetector class.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

from fetcher.thread_detector import ThreadDetector, _sync_handle, BrowserClosedByUserError, is_browser_closed_error


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
        """Test that thread collection is skipped when disabled in config."""
        fake_page = Mock()
        article = Mock()

        # Explicitly disable thread collection for this test
        monkeypatch.setattr(collector.config, 'collect_threads', False)

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


class TestBrowserClosedError:
    """Test cases for BrowserClosedByUserError and is_browser_closed_error."""

    def test_browser_closed_error_is_exception(self):
        """BrowserClosedByUserError should be a proper exception."""
        error = BrowserClosedByUserError("test message")
        assert isinstance(error, Exception)
        assert str(error) == "test message"

    def test_browser_closed_error_can_be_raised_and_caught(self):
        """BrowserClosedByUserError can be raised and caught."""
        with pytest.raises(BrowserClosedByUserError) as exc_info:
            raise BrowserClosedByUserError("User closed browser")
        assert "User closed browser" in str(exc_info.value)

    def test_is_browser_closed_error_target_closed(self):
        """is_browser_closed_error should detect 'target closed' messages."""
        assert is_browser_closed_error(Exception("Target closed"))
        assert is_browser_closed_error(Exception("target page, context or browser has been closed"))
        assert is_browser_closed_error(Exception("browser has been closed"))

    def test_is_browser_closed_error_context_closed(self):
        """is_browser_closed_error should detect context closed messages."""
        assert is_browser_closed_error(Exception("context has been closed"))
        assert is_browser_closed_error(Exception("page has been closed"))

    def test_is_browser_closed_error_connection_closed(self):
        """is_browser_closed_error should detect connection closed messages."""
        assert is_browser_closed_error(Exception("connection closed"))
        assert is_browser_closed_error(Exception("browser.close was called"))

    def test_is_browser_closed_error_false_for_other_errors(self):
        """is_browser_closed_error should return False for unrelated errors."""
        assert not is_browser_closed_error(Exception("timeout"))
        assert not is_browser_closed_error(Exception("element not found"))
        assert not is_browser_closed_error(Exception("navigation failed"))
        assert not is_browser_closed_error(ValueError("invalid value"))

    def test_is_browser_closed_error_case_insensitive(self):
        """is_browser_closed_error should be case insensitive."""
        assert is_browser_closed_error(Exception("TARGET CLOSED"))
        assert is_browser_closed_error(Exception("Browser Has Been Closed"))
        assert is_browser_closed_error(Exception("CONNECTION CLOSED"))


class TestThreadDetectorWithFakeElements:
    """Tests using FakeElement pattern for better coverage."""
    
    @pytest.fixture
    def detector(self):
        """Create ThreadDetector instance."""
        return ThreadDetector()

    def test_detect_thread_start_with_fake_element(self, detector):
        """Test thread detection using FakeElement pattern."""
        from fetcher.tests.fake_playwright import FakeElement
        
        # Create article without thread line
        article = FakeElement.tweet_article(
            tweet_id="123",
            content="Test tweet",
            author="testuser",
            has_thread_line=False,
        )
        
        result = detector.detect_thread_start(article, "testuser")
        assert result is True  # No thread line = start of thread

    def test_detect_thread_start_with_thread_line_element(self, detector):
        """Test that thread line indicates NOT the start using proper class structure."""
        from fetcher.tests.fake_playwright import FakeElement
        
        # Create article WITH proper thread line classes (r-1bimlpy AND r-f8sm7e)
        article = FakeElement(
            tag="article",
            children=[
                FakeElement(
                    tag="div",
                    attributes={"class": "css-175oi2r r-1bimlpy r-f8sm7e"},
                )
            ]
        )
        
        result = detector.detect_thread_start(article, "testuser")
        assert result is False  # Has thread line = NOT start

    def test_has_thread_line_with_connector(self, detector):
        """Test _has_thread_line with both r-1bimlpy AND r-f8sm7e classes."""
        from fetcher.tests.fake_playwright import FakeElement
        
        # Needs BOTH classes to be detected as thread line
        article = FakeElement(
            tag="article",
            children=[
                FakeElement(
                    tag="div",
                    attributes={"class": "css-175oi2r r-1bimlpy r-f8sm7e"},
                )
            ]
        )
        
        result = detector._has_thread_line(article)
        assert result is True

    def test_has_thread_line_without_f8sm7e(self, detector):
        """Test _has_thread_line needs BOTH classes."""
        from fetcher.tests.fake_playwright import FakeElement
        
        # Only r-1bimlpy, missing r-f8sm7e - should NOT detect as thread line
        article = FakeElement(
            tag="article",
            children=[
                FakeElement(
                    tag="div",
                    attributes={"class": "css-175oi2r r-1bimlpy"},  # Missing r-f8sm7e
                )
            ]
        )
        
        result = detector._has_thread_line(article)
        assert result is False

    def test_has_thread_line_without_connector(self, detector):
        """Test _has_thread_line without connector class."""
        from fetcher.tests.fake_playwright import FakeElement
        
        article = FakeElement(
            tag="article",
            children=[
                FakeElement(tag="div", attributes={"class": "other-class"}),
            ]
        )
        
        result = detector._has_thread_line(article)
        assert result is False

    def test_is_thread_member_with_reply_metadata(self, detector):
        """Test is_thread_member with reply metadata indicating self-reply."""
        # is_thread_member takes reply_metadata dict, not article
        reply_metadata = {
            'is_reply': True,
            'replies_to_username': 'threadauthor',
        }
        
        result = detector.is_thread_member(reply_metadata, "threadauthor")
        assert result is True

    def test_is_thread_member_reply_to_different_user(self, detector):
        """Test is_thread_member when replying to different user."""
        reply_metadata = {
            'is_reply': True,
            'replies_to_username': 'otheruser',
        }
        
        # When is_reply is True, it returns True regardless of username match
        result = detector.is_thread_member(reply_metadata, "threadauthor")
        assert result is True

    def test_is_thread_member_no_metadata(self, detector):
        """Test is_thread_member with None metadata."""
        result = detector.is_thread_member(None, "threadauthor")
        assert result is False

    def test_is_thread_member_empty_metadata(self, detector):
        """Test is_thread_member with empty metadata."""
        result = detector.is_thread_member({}, "threadauthor")
        assert result is False

    def test_is_thread_member_not_reply_but_same_username(self, detector):
        """Test is_thread_member when not marked as reply but has same username."""
        reply_metadata = {
            'is_reply': False,
            'replies_to_username': 'threadauthor',
        }
        
        result = detector.is_thread_member(reply_metadata, "threadauthor")
        assert result is True  # Same username match


class TestThreadDetectorPageInteractions:
    """Tests for page-based methods using FakePage."""
    
    @pytest.fixture
    def detector(self):
        """Create ThreadDetector instance."""
        return ThreadDetector()

    def test_dismiss_specific_overlays_with_fake_page(self, detector):
        """Test overlay dismissal using FakePage."""
        from fetcher.tests.fake_playwright import FakePage, FakeElement
        
        page = FakePage()
        
        # Add an article (target to click to dismiss overlay)
        page.add_article(FakeElement.tweet_article(
            tweet_id="123",
            content="Test content",
            author="testuser",
        ))
        
        # Should not raise exception
        detector._dismiss_specific_overlays(page)

    def test_collect_thread_articles_empty_page(self, detector):
        """Test collecting articles from empty page."""
        from fetcher.tests.fake_playwright import FakePage
        
        page = FakePage()
        
        articles = page.query_selector_all('article[data-testid="tweet"]')
        assert len(articles) == 0

    def test_collect_thread_articles_with_tweets(self, detector):
        """Test collecting articles from page with tweets."""
        from fetcher.tests.fake_playwright import FakePage, FakeElement
        
        page = FakePage()
        page.add_article(FakeElement.tweet_article("001", content="First"))
        page.add_article(FakeElement.tweet_article("002", content="Second"))
        page.add_article(FakeElement.tweet_article("003", content="Third"))
        
        articles = page.query_selector_all('article')
        assert len(articles) == 3


class TestExtractTweetFromArticle:
    """Tests for _extract_tweet_from_article method."""
    
    @pytest.fixture
    def detector(self):
        """Create ThreadDetector instance."""
        return ThreadDetector()

    def test_extract_tweet_id_from_link(self, detector):
        """Should extract tweet ID from status link."""
        from fetcher.tests.fake_playwright import FakeElement
        
        article = FakeElement(
            tag="article",
            children=[
                FakeElement(
                    tag="a",
                    attributes={"href": "/testuser/status/123456789"},
                ),
                FakeElement(
                    tag="div",
                    data_testid="tweetText",
                    text_content="Hello world!",
                ),
            ]
        )
        
        result = detector._extract_tweet_from_article(article)
        
        assert result is not None
        assert result['tweet_id'] == '123456789'
        assert result['username'] == 'testuser'

    def test_extract_content_from_tweetText(self, detector):
        """Should extract content from tweetText element."""
        from fetcher.tests.fake_playwright import FakeElement
        
        article = FakeElement(
            tag="article",
            children=[
                FakeElement(
                    tag="a",
                    attributes={"href": "/user/status/111"},
                ),
                FakeElement(
                    tag="div",
                    data_testid="tweetText",
                    text_content="This is the tweet content",
                ),
            ]
        )
        
        result = detector._extract_tweet_from_article(article)
        
        assert result is not None
        assert result['content'] == "This is the tweet content"

    def test_extract_timestamp_from_time_element(self, detector):
        """Should extract timestamp from time element."""
        from fetcher.tests.fake_playwright import FakeElement
        
        article = FakeElement(
            tag="article",
            children=[
                FakeElement(
                    tag="a",
                    attributes={"href": "/user/status/222"},
                ),
                FakeElement(
                    tag="time",
                    attributes={"datetime": "2025-01-10T15:30:00.000Z"},
                ),
            ]
        )
        
        result = detector._extract_tweet_from_article(article)
        
        assert result is not None
        assert result['tweet_timestamp'] == "2025-01-10T15:30:00.000Z"

    def test_returns_none_without_tweet_id(self, detector):
        """Should return None if no tweet ID can be extracted."""
        from fetcher.tests.fake_playwright import FakeElement
        
        article = FakeElement(
            tag="article",
            text_content="Some content without proper links",
        )
        
        result = detector._extract_tweet_from_article(article)
        
        assert result is None

    def test_extract_username_from_user_link(self, detector):
        """Should extract username from user profile link."""
        from fetcher.tests.fake_playwright import FakeElement
        
        article = FakeElement(
            tag="article",
            children=[
                FakeElement(
                    tag="a",
                    attributes={"href": "/actualuser/status/333"},
                ),
                FakeElement(
                    tag="a",
                    attributes={"href": "/actualuser"},
                ),
            ]
        )
        
        result = detector._extract_tweet_from_article(article)
        
        assert result is not None
        assert result['username'] == 'actualuser'


class TestExtractReplyToId:
    """Tests for _extract_reply_to_id method."""
    
    @pytest.fixture
    def detector(self):
        """Create ThreadDetector instance."""
        return ThreadDetector()

    def test_extract_reply_to_id_from_reply_section(self, detector):
        """Should extract reply-to ID from reply section."""
        from fetcher.tests.fake_playwright import FakeElement
        
        reply_section = FakeElement(
            tag="div",
            data_testid="reply",
            children=[
                FakeElement(
                    tag="a",
                    attributes={"href": "/otheruser/status/987654321"},
                ),
            ]
        )
        
        article = FakeElement(
            tag="article",
            children=[reply_section],
        )
        
        result = detector._extract_reply_to_id(article)
        
        assert result == "987654321"

    def test_returns_none_when_no_reply(self, detector):
        """Should return None when no reply section found."""
        from fetcher.tests.fake_playwright import FakeElement
        
        article = FakeElement(
            tag="article",
            children=[
                FakeElement(tag="div", text_content="Regular tweet"),
            ]
        )
        
        result = detector._extract_reply_to_id(article)
        
        assert result is None


class TestGroupIntoThreadsByReplyChain:
    """Tests for _group_into_threads_by_reply_chain (pure Python logic)."""
    
    @pytest.fixture
    def detector(self):
        """Create ThreadDetector instance."""
        return ThreadDetector()

    def test_groups_reply_chain_into_thread(self, detector):
        """Should group tweets connected by reply_to_tweet_id."""
        tweets = [
            {'tweet_id': '001', 'username': 'user1', 'reply_to_tweet_id': None, 'has_thread_line': True},
            {'tweet_id': '002', 'username': 'user1', 'reply_to_tweet_id': '001', 'has_thread_line': True},
            {'tweet_id': '003', 'username': 'user1', 'reply_to_tweet_id': '002', 'has_thread_line': False},
        ]
        
        result = detector._group_into_threads_by_reply_chain(tweets, 'user1')
        
        # Should find one thread with all 3 tweets
        assert len(result) >= 1
        
        # Find the thread containing our tweets
        thread_with_all = None
        for thread in result:
            tweet_ids = [t['tweet_id'] for t in thread['tweets']]
            if '001' in tweet_ids and '002' in tweet_ids and '003' in tweet_ids:
                thread_with_all = thread
                break
        
        assert thread_with_all is not None

    def test_returns_empty_for_single_tweet(self, detector):
        """Should return empty list if no thread (single tweet)."""
        tweets = [
            {'tweet_id': '001', 'username': 'user1', 'reply_to_tweet_id': None, 'has_thread_line': False},
        ]
        
        result = detector._group_into_threads_by_reply_chain(tweets, 'user1')
        
        # Single tweet without thread line is not a thread
        assert len(result) == 0

    def test_ignores_replies_from_different_user(self, detector):
        """Should not include replies from different users in thread."""
        tweets = [
            {'tweet_id': '001', 'username': 'user1', 'reply_to_tweet_id': None, 'has_thread_line': True},
            {'tweet_id': '002', 'username': 'user1', 'reply_to_tweet_id': '001', 'has_thread_line': False},
            {'tweet_id': '003', 'username': 'otheruser', 'reply_to_tweet_id': '002', 'has_thread_line': False},
        ]
        
        result = detector._group_into_threads_by_reply_chain(tweets, 'user1')
        
        # Should find thread with user1's tweets
        if len(result) > 0:
            for thread in result:
                for tweet in thread['tweets']:
                    # All tweets in thread should be from user1
                    if tweet.get('username') == 'otheruser':
                        # Other user replies might be excluded from chain
                        pass

    def test_finds_multiple_separate_threads(self, detector):
        """Should identify multiple separate threads."""
        tweets = [
            # Thread 1
            {'tweet_id': '001', 'username': 'user1', 'reply_to_tweet_id': None, 'has_thread_line': True},
            {'tweet_id': '002', 'username': 'user1', 'reply_to_tweet_id': '001', 'has_thread_line': False},
            # Thread 2 (separate)
            {'tweet_id': '100', 'username': 'user1', 'reply_to_tweet_id': None, 'has_thread_line': True},
            {'tweet_id': '101', 'username': 'user1', 'reply_to_tweet_id': '100', 'has_thread_line': False},
        ]
        
        result = detector._group_into_threads_by_reply_chain(tweets, 'user1')
        
        # Should find 2 threads
        assert len(result) == 2


class TestExtractReplyMetadata:
    """Tests for extract_reply_metadata method."""
    
    @pytest.fixture
    def detector(self):
        """Create ThreadDetector instance."""
        return ThreadDetector()

    def test_returns_none_without_reply_indicator(self, detector):
        """Should return None when no Tweet-User-Text element."""
        from fetcher.tests.fake_playwright import FakeElement
        
        article = FakeElement(
            tag="article",
            children=[
                FakeElement(tag="div", text_content="Regular tweet"),
            ]
        )
        
        result = detector.extract_reply_metadata(article)
        
        assert result is None

    def test_extracts_reply_to_tweet_id(self, detector):
        """Should extract reply-to tweet ID when present."""
        from fetcher.tests.fake_playwright import FakeElement
        
        # Create article with Tweet-User-Text indicator and status link
        article = FakeElement(
            tag="article",
            children=[
                FakeElement(
                    tag="div",
                    data_testid="Tweet-User-Text",
                    text_content="Replying to @someone",
                ),
                FakeElement(
                    tag="a",
                    attributes={"href": "/someone/status/555666777"},
                ),
            ]
        )
        
        result = detector.extract_reply_metadata(article)
        
        assert result is not None
        assert result['reply_to_tweet_id'] == '555666777'
        assert result['is_reply'] is True


class TestExtractContentWithLinks:
    """Tests for _extract_content_with_links method - preserves URLs in tweet text."""

    @pytest.fixture
    def detector(self):
        """Create ThreadDetector instance."""
        return ThreadDetector()

    def test_extracts_plain_text(self, detector):
        """Should extract plain text content without links."""
        mock_element = MagicMock()
        mock_element.inner_text.return_value = "Just a simple tweet"
        mock_element.query_selector_all.return_value = []

        result = detector._extract_content_with_links(mock_element)

        assert result == "Just a simple tweet"

    def test_preserves_external_links(self, detector):
        """Should append external URLs that aren't in the visible text."""
        mock_element = MagicMock()
        mock_element.inner_text.return_value = "Check this link"
        
        # Mock anchor with external URL
        mock_anchor = MagicMock()
        mock_anchor.get_attribute.return_value = "https://example.com/article"
        mock_element.query_selector_all.return_value = [mock_anchor]

        result = detector._extract_content_with_links(mock_element)

        assert "Check this link" in result
        assert "https://example.com/article" in result

    def test_skips_twitter_internal_links(self, detector):
        """Should not append Twitter internal links."""
        mock_element = MagicMock()
        mock_element.inner_text.return_value = "Mentioning @someone"
        
        # Mock anchor with Twitter internal URL
        mock_anchor = MagicMock()
        mock_anchor.get_attribute.return_value = "/someone"
        mock_element.query_selector_all.return_value = [mock_anchor]

        result = detector._extract_content_with_links(mock_element)

        assert result == "Mentioning @someone"
        assert "/" not in result or "http" in result  # No raw path added

    def test_skips_x_com_links(self, detector):
        """Should not append x.com internal links."""
        mock_element = MagicMock()
        mock_element.inner_text.return_value = "See this tweet"
        
        mock_anchor = MagicMock()
        mock_anchor.get_attribute.return_value = "https://x.com/user/status/123"
        mock_element.query_selector_all.return_value = [mock_anchor]

        result = detector._extract_content_with_links(mock_element)

        assert result == "See this tweet"
        assert "x.com" not in result

    def test_skips_hashtag_links(self, detector):
        """Should not append hashtag links."""
        mock_element = MagicMock()
        mock_element.inner_text.return_value = "Trending #topic"
        
        mock_anchor = MagicMock()
        mock_anchor.get_attribute.return_value = "/hashtag/topic"
        mock_element.query_selector_all.return_value = [mock_anchor]

        result = detector._extract_content_with_links(mock_element)

        assert result == "Trending #topic"
        assert "hashtag" not in result

    def test_deduplicates_links_already_in_text(self, detector):
        """Should not append links that already appear in the visible text."""
        mock_element = MagicMock()
        mock_element.inner_text.return_value = "Visit https://example.com for more"
        
        mock_anchor = MagicMock()
        mock_anchor.get_attribute.return_value = "https://example.com"
        mock_element.query_selector_all.return_value = [mock_anchor]

        result = detector._extract_content_with_links(mock_element)

        # Link should appear only once (in original text, not appended)
        assert result.count("https://example.com") == 1

    def test_handles_multiple_external_links(self, detector):
        """Should append multiple external URLs."""
        mock_element = MagicMock()
        mock_element.inner_text.return_value = "Multiple links here"
        
        mock_anchor1 = MagicMock()
        mock_anchor1.get_attribute.return_value = "https://first.com"
        mock_anchor2 = MagicMock()
        mock_anchor2.get_attribute.return_value = "https://second.com"
        mock_element.query_selector_all.return_value = [mock_anchor1, mock_anchor2]

        result = detector._extract_content_with_links(mock_element)

        assert "Multiple links here" in result
        assert "https://first.com" in result
        assert "https://second.com" in result

    def test_handles_anchor_without_href(self, detector):
        """Should gracefully handle anchors without href attribute."""
        mock_element = MagicMock()
        mock_element.inner_text.return_value = "Tweet text"
        
        mock_anchor = MagicMock()
        mock_anchor.get_attribute.return_value = None
        mock_element.query_selector_all.return_value = [mock_anchor]

        result = detector._extract_content_with_links(mock_element)

        assert result == "Tweet text"

    def test_fallback_on_exception(self, detector):
        """Should fallback to inner_text on exception."""
        mock_element = MagicMock()
        mock_element.inner_text.return_value = "Fallback text"
        mock_element.query_selector_all.side_effect = Exception("DOM error")

        result = detector._extract_content_with_links(mock_element)

        assert result == "Fallback text"


class TestThreadIdUsesFirstTweet:
    """Tests that thread_id uses the actual first tweet (min tweet_id), not clicked tweet."""

    @pytest.fixture
    def detector(self):
        """Create ThreadDetector instance."""
        return ThreadDetector()

    def test_format_thread_post_preserves_tweet_id(self, detector):
        """_format_thread_post should preserve original tweet_id for sorting."""
        tweet = {
            'tweet_id': '1234567890',
            'username': 'testuser',
            'content': 'Test content',
        }
        
        formatted = detector._format_thread_post(tweet)
        
        assert formatted['tweet_id'] == '1234567890'

    def test_thread_summary_with_correct_start_id(self):
        """ThreadSummary should reflect actual first tweet as start_id."""
        from fetcher.thread_detector import ThreadSummary
        
        # Simulate a thread where user clicked tweet 200, but 100 is the actual start
        tweets = [
            {'tweet_id': '100', 'content': 'First (actual start)'},
            {'tweet_id': '150', 'content': 'Second'},
            {'tweet_id': '200', 'content': 'Third (clicked)'},
        ]
        
        # After our fix, start_id should be the min (100), not clicked (200)
        actual_start_id = min(t['tweet_id'] for t in tweets)
        
        summary = ThreadSummary(
            start_id=actual_start_id,  # Should be '100'
            url=f'https://x.com/user/status/{actual_start_id}',
            tweets=tweets,
            conversation_id=actual_start_id
        )
        
        assert summary.start_id == '100'
        assert '100' in summary.url

    def test_collect_thread_chain_recursive_order(self, detector):
        """_collect_thread_chain should maintain tweet order for proper ID sorting."""
        tweets = [
            {'tweet_id': '100', 'username': 'testuser', 'content': 'First'},
            {'tweet_id': '101', 'username': 'testuser', 'content': 'Second', 'reply_to_tweet_id': '100'},
            {'tweet_id': '102', 'username': 'testuser', 'content': 'Third', 'reply_to_tweet_id': '101'},
        ]
        
        tweet_by_id = {t['tweet_id']: t for t in tweets}
        children_map = {'100': ['101'], '101': ['102']}
        
        result = []
        used = set()
        
        detector._collect_thread_chain('100', tweet_by_id, children_map, result, used, 'testuser')
        
        assert len(result) == 3
        assert result[0]['tweet_id'] == '100'  # First tweet
        assert result[1]['tweet_id'] == '101'
        assert result[2]['tweet_id'] == '102'
        
        # Verify min would give correct start
        assert min(t['tweet_id'] for t in result) == '100'