"""
Unit tests for the async session manager module.
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import json
from pathlib import Path


class TestAsyncSessionManager:
    """Tests for AsyncSessionManager class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        config = Mock()
        config.headless = True
        config.slow_mo = 0
        config.viewport_width = 1280
        config.viewport_height = 800
        config.user_agents = ["Mozilla/5.0 Test Agent"]
        return config

    @pytest.fixture
    def session_manager(self, mock_config, tmp_path):
        """Create an AsyncSessionManager with mocked config."""
        with patch('fetcher.async_session_manager.get_config', return_value=mock_config), \
             patch('fetcher.async_session_manager.get_async_scroller'):
            from fetcher.async_session_manager import AsyncSessionManager
            manager = AsyncSessionManager()
            # Replace session file path with tmp_path based one
            manager.session_file = tmp_path / "x_session.json"
            return manager

    def test_has_valid_session_no_file(self, session_manager):
        """Test has_valid_session returns False when no session file."""
        # File doesn't exist by default in tmp_path
        result = session_manager.has_valid_session()
        assert result is False

    def test_has_valid_session_valid_cookies(self, session_manager):
        """Test has_valid_session returns True with valid auth cookies."""
        session_data = {
            'cookies': [
                {'name': 'auth_token', 'value': 'token123'},
                {'name': 'ct0', 'value': 'csrf123'},
                {'name': 'twid', 'value': 'user123'},
            ]
        }
        
        # Create the session file with valid cookies
        session_manager.session_file.write_text(json.dumps(session_data))
        
        result = session_manager.has_valid_session()
        assert result is True

    def test_has_valid_session_missing_auth_cookies(self, session_manager):
        """Test has_valid_session returns False with missing auth cookies."""
        session_data = {
            'cookies': [
                {'name': 'some_cookie', 'value': 'value1'},
            ]
        }
        
        # Create the session file with missing auth cookies
        session_manager.session_file.write_text(json.dumps(session_data))
        
        result = session_manager.has_valid_session()
        assert result is False

    def test_has_valid_session_invalid_json(self, session_manager):
        """Test has_valid_session returns False on invalid JSON."""
        # Create a file with invalid JSON
        session_manager.session_file.write_text("not valid json {{{")
        
        result = session_manager.has_valid_session()
        assert result is False

    @pytest.mark.asyncio
    async def test_create_browser_context_launches_browser(self, session_manager, mock_config):
        """Test create_browser_context launches browser with correct settings."""
        mock_playwright = AsyncMock()
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        
        mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_page.add_init_script = AsyncMock()
        
        # Session file doesn't exist
        browser, context, page = await session_manager.create_browser_context(mock_playwright)
        
        assert browser is mock_browser
        assert context is mock_context
        assert page is mock_page
        
        # Verify browser launch args
        launch_call = mock_playwright.chromium.launch.call_args
        assert launch_call.kwargs['headless'] is True
        assert '--disable-blink-features=AutomationControlled' in launch_call.kwargs['args']

    @pytest.mark.asyncio
    async def test_create_browser_context_loads_session(self, session_manager):
        """Test create_browser_context loads session state if exists."""
        mock_playwright = AsyncMock()
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        
        mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_page.add_init_script = AsyncMock()
        
        # Create a session file
        session_manager.session_file.write_text(json.dumps({'cookies': []}))
        
        await session_manager.create_browser_context(mock_playwright)
        
        # Verify storage_state was included
        context_call = mock_browser.new_context.call_args
        assert 'storage_state' in context_call.kwargs

    @pytest.mark.asyncio
    async def test_save_session_state_success(self, session_manager):
        """Test save_session_state saves context state to file."""
        mock_context = AsyncMock()
        mock_context.storage_state = AsyncMock(return_value={'cookies': []})
        
        mock_file = MagicMock()
        with patch('builtins.open', return_value=mock_file):
            result = await session_manager.save_session_state(mock_context)
        
        assert result is True
        mock_context.storage_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_session_state_failure(self, session_manager):
        """Test save_session_state returns False on error."""
        mock_context = AsyncMock()
        mock_context.storage_state = AsyncMock(side_effect=Exception("Save failed"))
        
        result = await session_manager.save_session_state(mock_context)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_cleanup_session_closes_resources(self, session_manager):
        """Test cleanup_session closes browser and context."""
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        
        await session_manager.cleanup_session(mock_browser, mock_context)
        
        mock_context.close.assert_called_once()
        mock_browser.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_session_handles_errors(self, session_manager):
        """Test cleanup_session handles errors gracefully."""
        mock_browser = AsyncMock()
        mock_browser.close = AsyncMock(side_effect=Exception("Close failed"))
        mock_context = AsyncMock()
        
        # Should not raise
        await session_manager.cleanup_session(mock_browser, mock_context)

    @pytest.mark.asyncio
    async def test_cleanup_session_handles_none_values(self, session_manager):
        """Test cleanup_session handles None browser/context."""
        # Should not raise
        await session_manager.cleanup_session(None, None)

    @pytest.mark.asyncio
    async def test_navigate_to_profile_success(self, session_manager):
        """Test navigate_to_profile navigates and waits for content."""
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.wait_for_selector = AsyncMock()
        
        with patch.object(session_manager.scroller, 'delay', new_callable=AsyncMock):
            result = await session_manager.navigate_to_profile(mock_page, "testuser")
        
        assert result is True
        mock_page.goto.assert_called_once_with(
            "https://x.com/testuser",
            wait_until="domcontentloaded",
            timeout=30000
        )

    @pytest.mark.asyncio
    async def test_navigate_to_profile_no_tweets_but_profile_exists(self, session_manager):
        """Test navigate_to_profile returns True if profile exists but no tweets."""
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        # First wait fails (no tweets), second succeeds (profile exists)
        mock_page.wait_for_selector = AsyncMock(
            side_effect=[Exception("No tweets"), None]
        )
        
        with patch.object(session_manager.scroller, 'delay', new_callable=AsyncMock):
            result = await session_manager.navigate_to_profile(mock_page, "testuser")
        
        assert result is True

    @pytest.mark.asyncio
    async def test_navigate_to_profile_not_found(self, session_manager):
        """Test navigate_to_profile returns False when profile not found."""
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.wait_for_selector = AsyncMock(side_effect=Exception("Not found"))
        
        with patch.object(session_manager.scroller, 'delay', new_callable=AsyncMock):
            result = await session_manager.navigate_to_profile(mock_page, "nonexistent")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_navigate_to_profile_navigation_error(self, session_manager):
        """Test navigate_to_profile handles navigation errors."""
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(side_effect=Exception("Navigation failed"))
        
        result = await session_manager.navigate_to_profile(mock_page, "testuser")
        
        assert result is False


class TestAsyncSessionManagerInit:
    """Tests for AsyncSessionManager initialization."""

    def test_init_sets_session_file(self):
        """Test __init__ sets session file path."""
        with patch('fetcher.async_session_manager.get_config'), \
             patch('fetcher.async_session_manager.get_async_scroller'):
            from fetcher.async_session_manager import AsyncSessionManager
            
            manager = AsyncSessionManager()
            
            assert manager.session_file == Path("x_session.json")

    def test_init_gets_config(self):
        """Test __init__ gets config instance."""
        mock_config = Mock()
        with patch('fetcher.async_session_manager.get_config', return_value=mock_config), \
             patch('fetcher.async_session_manager.get_async_scroller'):
            from fetcher.async_session_manager import AsyncSessionManager
            
            manager = AsyncSessionManager()
            
            assert manager.config is mock_config
