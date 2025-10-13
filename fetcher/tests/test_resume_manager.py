"""
Tests for ResumeManager class in fetcher/resume_manager.py
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from fetcher.resume_manager import ResumeManager


class TestResumeManager:
    """Test cases for ResumeManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.resume_manager = ResumeManager()

    def test_init(self):
        """Test ResumeManager initialization."""
        assert self.resume_manager.scroller is not None
        assert hasattr(self.resume_manager.scroller, 'delay')

    def test_convert_timestamp_to_date_filter_valid_iso(self):
        """Test converting valid ISO timestamp to date filter."""
        timestamp = "2023-10-15T14:30:00Z"
        result = self.resume_manager.convert_timestamp_to_date_filter(timestamp)
        assert result == "2023-10-15"

    def test_convert_timestamp_to_date_filter_valid_iso_with_timezone(self):
        """Test converting ISO timestamp with timezone to date filter."""
        timestamp = "2023-10-15T14:30:00+02:00"
        result = self.resume_manager.convert_timestamp_to_date_filter(timestamp)
        assert result == "2023-10-15"

    def test_convert_timestamp_to_date_filter_invalid_timestamp(self):
        """Test handling of invalid timestamp."""
        timestamp = "invalid-timestamp"
        result = self.resume_manager.convert_timestamp_to_date_filter(timestamp)
        assert result is None

    def test_convert_timestamp_to_date_filter_empty_string(self):
        """Test handling of empty string."""
        timestamp = ""
        result = self.resume_manager.convert_timestamp_to_date_filter(timestamp)
        assert result is None

    def test_convert_timestamp_to_date_filter_none(self):
        """Test handling of None input."""
        timestamp = None
        result = self.resume_manager.convert_timestamp_to_date_filter(timestamp)
        assert result is None

    @patch('fetcher.resume_manager.print')
    def test_try_resume_via_search_successful(self, mock_print):
        """Test successful search-based resume."""
        # Mock page object
        mock_page = Mock()
        mock_articles = [Mock(), Mock(), Mock()]  # 3 mock articles
        mock_page.query_selector_all.return_value = mock_articles
        mock_page.goto = Mock()

        # Mock scroller delay
        self.resume_manager.scroller.delay = Mock()

        username = "testuser"
        oldest_timestamp = "2023-10-15T14:30:00Z"

        result = self.resume_manager.try_resume_via_search(mock_page, username, oldest_timestamp)

        assert result is True
        mock_page.goto.assert_called_once()
        expected_url = "https://x.com/search?q=from:testuser until:2023-10-15&src=typed_query&f=live"
        assert mock_page.goto.call_args[0][0] == expected_url
        mock_page.query_selector_all.assert_called_once_with('article[data-testid="tweet"]')
        self.resume_manager.scroller.delay.assert_called_once_with(3.0, 5.0)

    @patch('fetcher.resume_manager.print')
    def test_try_resume_via_search_no_results(self, mock_print):
        """Test search-based resume with no results."""
        # Mock page object
        mock_page = Mock()
        mock_page.query_selector_all.return_value = []  # No articles found
        mock_page.goto = Mock()

        # Mock scroller delay
        self.resume_manager.scroller.delay = Mock()

        username = "testuser"
        oldest_timestamp = "2023-10-15T14:30:00Z"

        result = self.resume_manager.try_resume_via_search(mock_page, username, oldest_timestamp)

        assert result is False
        mock_page.goto.assert_called_once()
        mock_page.query_selector_all.assert_called_once_with('article[data-testid="tweet"]')

    @patch('fetcher.resume_manager.print')
    def test_try_resume_via_search_invalid_timestamp(self, mock_print):
        """Test search-based resume with invalid timestamp."""
        # Mock page object
        mock_page = Mock()
        mock_page.goto = Mock()

        username = "testuser"
        oldest_timestamp = "invalid-timestamp"

        result = self.resume_manager.try_resume_via_search(mock_page, username, oldest_timestamp)

        assert result is False
        mock_page.goto.assert_not_called()  # Should not navigate if timestamp conversion fails

    @patch('fetcher.resume_manager.print')
    def test_try_resume_via_search_goto_exception(self, mock_print):
        """Test search-based resume with navigation exception."""
        # Mock page object that raises exception on goto
        mock_page = Mock()
        mock_page.goto.side_effect = Exception("Navigation failed")

        username = "testuser"
        oldest_timestamp = "2023-10-15T14:30:00Z"

        result = self.resume_manager.try_resume_via_search(mock_page, username, oldest_timestamp)

        assert result is False
        mock_page.goto.assert_called_once()

    @patch('fetcher.resume_manager.print')
    def test_resume_positioning_search_success(self, mock_print):
        """Test resume positioning with successful search."""
        # Mock page object
        mock_page = Mock()
        mock_articles = [Mock()]
        mock_page.query_selector_all.return_value = mock_articles
        mock_page.goto = Mock()

        # Mock scroller delay
        self.resume_manager.scroller.delay = Mock()

        username = "testuser"
        oldest_timestamp = "2023-10-15T14:30:00Z"

        result = self.resume_manager.resume_positioning(mock_page, username, oldest_timestamp)

        assert result is True
        # Should have called search-based resume first
        assert mock_page.goto.call_count == 1

    @patch('fetcher.resume_manager.print')
    def test_resume_positioning_fallback_to_profile(self, mock_print):
        """Test resume positioning falling back to profile navigation."""
        # Mock page object - search fails, profile succeeds
        mock_page = Mock()
        mock_page.query_selector_all.return_value = []  # Search finds no results
        mock_page.goto = Mock()

        # Mock scroller delay
        self.resume_manager.scroller.delay = Mock()

        username = "testuser"
        oldest_timestamp = "2023-10-15T14:30:00Z"

        result = self.resume_manager.resume_positioning(mock_page, username, oldest_timestamp)

        assert result is True
        # Should have called goto twice: once for search, once for profile
        assert mock_page.goto.call_count == 2
        # Second call should be to profile URL
        profile_call = mock_page.goto.call_args_list[1]
        assert profile_call[0][0] == "https://x.com/testuser"

    @patch('fetcher.resume_manager.print')
    def test_resume_positioning_all_strategies_fail(self, mock_print):
        """Test resume positioning when all strategies fail."""
        # Mock page object that fails on all operations
        mock_page = Mock()
        mock_page.query_selector_all.return_value = []  # Search finds no results
        mock_page.goto.side_effect = Exception("Navigation failed")

        username = "testuser"
        oldest_timestamp = "2023-10-15T14:30:00Z"

        result = self.resume_manager.resume_positioning(mock_page, username, oldest_timestamp)

        assert result is False
        # Should have attempted both search and profile navigation
        assert mock_page.goto.call_count == 2

    @patch('fetcher.resume_manager.print')
    def test_resume_positioning_invalid_timestamp_fallback(self, mock_print):
        """Test resume positioning with invalid timestamp falls back to profile."""
        # Mock page object - invalid timestamp causes search to fail
        mock_page = Mock()
        mock_page.goto = Mock()

        # Mock scroller delay
        self.resume_manager.scroller.delay = Mock()

        username = "testuser"
        oldest_timestamp = "invalid-timestamp"

        result = self.resume_manager.resume_positioning(mock_page, username, oldest_timestamp)

        assert result is True
        # Should have fallen back to profile navigation
        assert mock_page.goto.call_count == 1
        profile_call = mock_page.goto.call_args_list[0]
        assert profile_call[0][0] == "https://x.com/testuser"