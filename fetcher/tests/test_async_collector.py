"""
Unit tests for the async tweet collector module.
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from pathlib import Path
import asyncio


class TestAsyncTweetCollector:
    """Tests for AsyncTweetCollector class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        config = Mock()
        config.skip_duplicate_check = False
        config.batch_write_size = 50
        config.max_consecutive_empty_scrolls = 15
        config.min_human_delay = 0.5
        config.max_human_delay = 1.5
        return config

    @pytest.fixture
    def collector(self, mock_config):
        """Create an AsyncTweetCollector with mocked dependencies."""
        with patch('fetcher.async_collector.get_config', return_value=mock_config), \
             patch('fetcher.async_collector.get_async_scroller'), \
             patch('fetcher.async_collector.get_async_session_manager'):
            from fetcher.async_collector import AsyncTweetCollector
            return AsyncTweetCollector()

    def test_should_process_tweet_with_valid_id(self, collector):
        """Test should_process_tweet returns True for valid unseen tweet."""
        seen_ids = set()
        result = collector.should_process_tweet("123456789", seen_ids)
        assert result is True

    def test_should_process_tweet_with_empty_id(self, collector):
        """Test should_process_tweet returns False for empty tweet ID."""
        seen_ids = set()
        result = collector.should_process_tweet("", seen_ids)
        assert result is False

    def test_should_process_tweet_with_none_id(self, collector):
        """Test should_process_tweet returns False for None tweet ID."""
        seen_ids = set()
        result = collector.should_process_tweet(None, seen_ids)
        assert result is False

    def test_should_process_tweet_already_seen(self, collector):
        """Test should_process_tweet returns False for already seen tweet."""
        seen_ids = {"123456789"}
        result = collector.should_process_tweet("123456789", seen_ids)
        assert result is False

    def test_check_tweet_exists_skips_when_configured(self, collector, mock_config):
        """Test check_tweet_exists_in_db returns False when skip_duplicate_check is True."""
        mock_config.skip_duplicate_check = True
        
        exists, data = collector.check_tweet_exists_in_db("user1", "12345")
        assert exists is False
        assert data is None

    def test_check_tweet_exists_returns_true_when_found(self, collector, mock_config):
        """Test check_tweet_exists_in_db returns True when tweet exists."""
        mock_config.skip_duplicate_check = False
        
        mock_tweet = {
            'username': 'user1',
            'post_type': 'original',
            'content': 'Test content',
            'original_author': None,
            'original_tweet_id': None
        }
        
        with patch('fetcher.async_collector.get_tweet_repository') as mock_repo:
            mock_repo.return_value.get_tweet_by_id.return_value = mock_tweet
            
            exists, data = collector.check_tweet_exists_in_db("user1", "12345")
            
            assert exists is True
            assert data['post_type'] == 'original'
            assert data['content'] == 'Test content'

    def test_check_tweet_exists_returns_false_when_not_found(self, collector, mock_config):
        """Test check_tweet_exists_in_db returns False when tweet doesn't exist."""
        mock_config.skip_duplicate_check = False
        
        with patch('fetcher.async_collector.get_tweet_repository') as mock_repo:
            mock_repo.return_value.get_tweet_by_id.return_value = None
            
            exists, data = collector.check_tweet_exists_in_db("user1", "12345")
            
            assert exists is False
            assert data is None

    def test_check_tweet_exists_returns_false_for_different_user(self, collector, mock_config):
        """Test check_tweet_exists_in_db returns False when tweet belongs to different user."""
        mock_config.skip_duplicate_check = False
        
        mock_tweet = {
            'username': 'different_user',
            'post_type': 'original',
            'content': 'Test content',
            'original_author': None,
            'original_tweet_id': None
        }
        
        with patch('fetcher.async_collector.get_tweet_repository') as mock_repo:
            mock_repo.return_value.get_tweet_by_id.return_value = mock_tweet
            
            exists, data = collector.check_tweet_exists_in_db("user1", "12345")
            
            assert exists is False
            assert data is None


class TestAsyncTweetCollectorDebugInfo:
    """Tests for debug info saving functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        config = Mock()
        config.skip_duplicate_check = False
        config.batch_write_size = 50
        config.max_consecutive_empty_scrolls = 15
        return config

    @pytest.fixture
    def collector(self, mock_config):
        """Create an AsyncTweetCollector with mocked dependencies."""
        with patch('fetcher.async_collector.get_config', return_value=mock_config), \
             patch('fetcher.async_collector.get_async_scroller'), \
             patch('fetcher.async_collector.get_async_session_manager'):
            from fetcher.async_collector import AsyncTweetCollector
            return AsyncTweetCollector()

    @pytest.mark.asyncio
    async def test_save_debug_info_creates_directory(self, collector, tmp_path, monkeypatch):
        """Test _save_debug_info creates debug directory."""
        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock()
        mock_page.content = AsyncMock(return_value="<html></html>")
        mock_page.url = "https://x.com/testuser"
        mock_page.locator.return_value.inner_text = AsyncMock(return_value="Test content")
        
        # Change to tmp_path so debug files are created there
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            await collector._save_debug_info(mock_page, "testuser", 100)
            
            mock_page.screenshot.assert_called_once()
            # Verify debug directory was created
            assert (tmp_path / "logs" / "debug").exists()
        finally:
            os.chdir(original_cwd)

    @pytest.mark.asyncio
    async def test_save_debug_info_handles_errors(self, collector):
        """Test _save_debug_info handles errors gracefully."""
        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock(side_effect=Exception("Screenshot failed"))
        
        # Should not raise
        await collector._save_debug_info(mock_page, "testuser", 100)


class TestAsyncFetchSession:
    """Tests for async fetch session functions."""

    @pytest.mark.asyncio
    async def test_run_async_fetch_session_processes_users(self):
        """Test run_async_fetch_session processes multiple users."""
        with patch('fetcher.async_collector.async_playwright') as mock_pw, \
             patch('fetcher.async_collector.AsyncTweetCollector') as mock_collector_class, \
             patch('fetcher.async_collector.get_async_session_manager') as mock_sm, \
             patch('fetcher.async_collector.get_db_connection_context'):
            
            # Setup mocks
            mock_browser = AsyncMock()
            mock_context = AsyncMock()
            mock_page = AsyncMock()
            
            mock_sm_instance = AsyncMock()
            mock_sm_instance.create_browser_context = AsyncMock(
                return_value=(mock_browser, mock_context, mock_page)
            )
            mock_sm_instance.navigate_to_profile = AsyncMock(return_value=True)
            mock_sm_instance.cleanup_session = AsyncMock()
            mock_sm.return_value = mock_sm_instance
            
            mock_collector = Mock()
            mock_collector.collect_tweets_from_page = AsyncMock(return_value=[{'id': '1'}])
            mock_collector_class.return_value = mock_collector
            
            # Setup async context manager
            mock_pw_context = AsyncMock()
            mock_pw.return_value.__aenter__ = AsyncMock(return_value=mock_pw_context)
            mock_pw.return_value.__aexit__ = AsyncMock()
            
            from fetcher.async_collector import run_async_fetch_session
            total, processed = await run_async_fetch_session(["user1", "user2"], max_tweets=10)
            
            assert processed == 2
            assert total == 2

    def test_run_async_fetch_wrapper(self):
        """Test synchronous wrapper for async fetch."""
        with patch('fetcher.async_collector.asyncio.run') as mock_run:
            mock_run.return_value = (100, 5)
            
            from fetcher.async_collector import run_async_fetch
            total, processed = run_async_fetch(["user1"], max_tweets=10)
            
            assert total == 100
            assert processed == 5
            mock_run.assert_called_once()
