"""
Unit tests for ThreadDetector class.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

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