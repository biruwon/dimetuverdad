"""Tests for the session_manager module."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from fetcher.session_manager import SessionManager, create_session, cleanup_session


class TestHasValidSession:
    """Tests for SessionManager.has_valid_session()."""

    def test_returns_false_when_no_session_file(self, tmp_path, monkeypatch):
        """Should return False when session file doesn't exist."""
        manager = SessionManager()
        manager.session_file = tmp_path / "nonexistent.json"
        
        assert manager.has_valid_session() is False

    def test_returns_false_when_empty_cookies(self, tmp_path, monkeypatch):
        """Should return False when session file has no cookies."""
        session_file = tmp_path / "session.json"
        session_file.write_text(json.dumps({"cookies": []}))
        
        manager = SessionManager()
        manager.session_file = session_file
        
        assert manager.has_valid_session() is False

    def test_returns_false_when_missing_auth_cookies(self, tmp_path):
        """Should return False when auth cookies are missing."""
        session_file = tmp_path / "session.json"
        session_file.write_text(json.dumps({
            "cookies": [{"name": "random_cookie", "value": "value"}]
        }))
        
        manager = SessionManager()
        manager.session_file = session_file
        
        assert manager.has_valid_session() is False

    def test_returns_false_with_only_one_auth_cookie(self, tmp_path):
        """Should return False when only one auth cookie exists (need at least 2)."""
        session_file = tmp_path / "session.json"
        session_file.write_text(json.dumps({
            "cookies": [{"name": "auth_token", "value": "token123"}]
        }))
        
        manager = SessionManager()
        manager.session_file = session_file
        
        assert manager.has_valid_session() is False

    def test_returns_true_with_valid_auth_cookies(self, tmp_path):
        """Should return True when auth_token and ct0 cookies exist."""
        session_file = tmp_path / "session.json"
        session_file.write_text(json.dumps({
            "cookies": [
                {"name": "auth_token", "value": "token123"},
                {"name": "ct0", "value": "ct0value"}
            ]
        }))
        
        manager = SessionManager()
        manager.session_file = session_file
        
        assert manager.has_valid_session() is True

    def test_returns_true_with_all_three_auth_cookies(self, tmp_path):
        """Should return True when auth_token, ct0, and twid cookies exist."""
        session_file = tmp_path / "session.json"
        session_file.write_text(json.dumps({
            "cookies": [
                {"name": "auth_token", "value": "token123"},
                {"name": "ct0", "value": "ct0value"},
                {"name": "twid", "value": "twidvalue"}
            ]
        }))
        
        manager = SessionManager()
        manager.session_file = session_file
        
        assert manager.has_valid_session() is True

    def test_returns_false_on_json_decode_error(self, tmp_path):
        """Should return False when session file contains invalid JSON."""
        session_file = tmp_path / "session.json"
        session_file.write_text("not valid json {{{")
        
        manager = SessionManager()
        manager.session_file = session_file
        
        assert manager.has_valid_session() is False

    def test_returns_false_on_io_error(self, tmp_path, monkeypatch):
        """Should return False when there's an IO error reading the file."""
        session_file = tmp_path / "session.json"
        session_file.write_text("{}")
        
        manager = SessionManager()
        manager.session_file = session_file
        
        # Make the file unreadable by mocking open
        def raise_io_error(*args, **kwargs):
            raise IOError("Permission denied")
        
        with patch("builtins.open", raise_io_error):
            assert manager.has_valid_session() is False


class TestEnsureLoggedIn:
    """Tests for SessionManager.ensure_logged_in()."""

    def test_returns_true_when_valid_session_exists(self, tmp_path):
        """Should return True without login when valid session exists."""
        session_file = tmp_path / "session.json"
        session_file.write_text(json.dumps({
            "cookies": [
                {"name": "auth_token", "value": "token123"},
                {"name": "ct0", "value": "ct0value"}
            ]
        }))
        
        manager = SessionManager()
        manager.session_file = session_file
        
        mock_page = MagicMock()
        result = manager.ensure_logged_in(mock_page)
        
        assert result is True
        # Should not attempt to navigate to login page
        mock_page.goto.assert_not_called()

    def test_returns_false_when_no_credentials(self, tmp_path, monkeypatch):
        """Should return False when no credentials are configured."""
        session_file = tmp_path / "nonexistent.json"
        
        manager = SessionManager()
        manager.session_file = session_file
        
        # Mock config without credentials
        mock_config = MagicMock()
        mock_config.username = None
        mock_config.password = None
        manager.config = mock_config
        
        mock_page = MagicMock()
        result = manager.ensure_logged_in(mock_page)
        
        assert result is False

    def test_calls_login_when_no_session_but_has_credentials(self, tmp_path):
        """Should attempt login when no session but credentials exist."""
        session_file = tmp_path / "nonexistent.json"
        
        manager = SessionManager()
        manager.session_file = session_file
        
        # Mock config with credentials
        mock_config = MagicMock()
        mock_config.username = "testuser"
        mock_config.password = "testpass"
        manager.config = mock_config
        
        # Mock login method
        manager.login_and_save_session = MagicMock(return_value=True)
        
        mock_page = MagicMock()
        result = manager.ensure_logged_in(mock_page)
        
        assert result is True
        manager.login_and_save_session.assert_called_once_with(
            mock_page, "testuser", "testpass"
        )


class TestCleanupSession:
    """Tests for SessionManager.cleanup_session()."""

    def test_closes_context_and_browser(self):
        """Should close both context and browser."""
        manager = SessionManager()
        
        mock_browser = MagicMock()
        mock_context = MagicMock()
        
        manager.cleanup_session(mock_browser, mock_context)
        
        mock_context.close.assert_called_once()
        mock_browser.close.assert_called_once()

    def test_handles_none_context(self):
        """Should handle None context gracefully."""
        manager = SessionManager()
        
        mock_browser = MagicMock()
        
        # Should not raise exception
        manager.cleanup_session(mock_browser, None)
        
        mock_browser.close.assert_called_once()

    def test_handles_none_browser(self):
        """Should handle None browser gracefully."""
        manager = SessionManager()
        
        mock_context = MagicMock()
        
        # Should not raise exception
        manager.cleanup_session(None, mock_context)
        
        mock_context.close.assert_called_once()

    def test_handles_exception_during_cleanup(self):
        """Should log warning and not raise on cleanup error."""
        manager = SessionManager()
        
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_context.close.side_effect = Exception("Close error")
        
        # Should not raise exception - the method catches exceptions
        manager.cleanup_session(mock_browser, mock_context)
        
        # Context close was attempted (and raised)
        mock_context.close.assert_called_once()


class TestClearSessionData:
    """Tests for SessionManager.clear_session_data()."""

    def test_clears_local_and_session_storage(self):
        """Should execute JS to clear storage."""
        manager = SessionManager()
        
        mock_context = MagicMock()
        
        manager.clear_session_data(mock_context)
        
        mock_context.evaluate.assert_called_once()
        call_args = mock_context.evaluate.call_args[0][0]
        assert "localStorage.clear()" in call_args
        assert "sessionStorage.clear()" in call_args

    def test_handles_exception_gracefully(self):
        """Should log warning and not raise on error."""
        manager = SessionManager()
        
        mock_context = MagicMock()
        mock_context.evaluate.side_effect = Exception("Evaluate error")
        
        # Should not raise exception
        manager.clear_session_data(mock_context)


class TestSessionManagerInit:
    """Tests for SessionManager initialization."""

    def test_creates_with_default_session_file(self):
        """Should create manager with default session file path."""
        manager = SessionManager()
        
        assert manager.session_file == Path("x_session.json")

    def test_creates_scroller_instance(self):
        """Should create a Scroller instance."""
        manager = SessionManager()
        
        from fetcher.scroller import Scroller
        assert isinstance(manager.scroller, Scroller)

    def test_loads_config(self):
        """Should load config on initialization."""
        manager = SessionManager()
        
        assert manager.config is not None


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_cleanup_session_function_calls_manager(self):
        """cleanup_session() should delegate to SessionManager."""
        mock_browser = MagicMock()
        mock_context = MagicMock()
        
        with patch.object(SessionManager, 'cleanup_session') as mock_cleanup:
            cleanup_session(mock_browser, mock_context)
            mock_cleanup.assert_called_once_with(mock_browser, mock_context)

    def test_cleanup_session_function_stops_playwright(self):
        """cleanup_session() should stop playwright if provided."""
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_playwright = MagicMock()
        
        with patch.object(SessionManager, 'cleanup_session'):
            cleanup_session(mock_browser, mock_context, mock_playwright)
            mock_playwright.stop.assert_called_once()

    def test_cleanup_session_handles_playwright_stop_error(self):
        """cleanup_session() should handle playwright stop errors."""
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_playwright = MagicMock()
        mock_playwright.stop.side_effect = Exception("Stop error")
        
        with patch.object(SessionManager, 'cleanup_session'):
            # Should not raise exception
            cleanup_session(mock_browser, mock_context, mock_playwright)


class TestCreateBrowserContext:
    """Tests for SessionManager.create_browser_context()."""

    def test_uses_chrome_channel(self):
        """Should launch Chrome browser with chrome channel."""
        manager = SessionManager()
        
        mock_playwright = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        
        mock_playwright.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        
        browser, context, page = manager.create_browser_context(mock_playwright)
        
        # Check chrome channel is used
        call_kwargs = mock_playwright.chromium.launch.call_args[1]
        assert call_kwargs['channel'] == 'chrome'

    def test_applies_anti_detection_args(self):
        """Should apply anti-automation detection browser args."""
        manager = SessionManager()
        
        mock_playwright = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        
        mock_playwright.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        
        manager.create_browser_context(mock_playwright)
        
        call_kwargs = mock_playwright.chromium.launch.call_args[1]
        args = call_kwargs['args']
        assert "--disable-blink-features=AutomationControlled" in args
        assert "--disable-automation" in args

    def test_uses_configured_viewport(self):
        """Should use viewport from config."""
        manager = SessionManager()
        
        mock_playwright = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        
        mock_playwright.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        
        manager.create_browser_context(mock_playwright)
        
        context_kwargs = mock_browser.new_context.call_args[1]
        assert 'viewport' in context_kwargs
        assert context_kwargs['viewport']['width'] == manager.config.viewport_width
        assert context_kwargs['viewport']['height'] == manager.config.viewport_height

    def test_sets_spanish_locale(self):
        """Should set Spanish locale and Madrid timezone."""
        manager = SessionManager()
        
        mock_playwright = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        
        mock_playwright.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        
        manager.create_browser_context(mock_playwright)
        
        context_kwargs = mock_browser.new_context.call_args[1]
        assert context_kwargs['locale'] == 'es-ES'
        assert context_kwargs['timezone_id'] == 'Europe/Madrid'

    def test_loads_existing_session_file(self, tmp_path):
        """Should load existing session file if available."""
        session_file = tmp_path / "session.json"
        session_file.write_text('{"cookies": []}')
        
        manager = SessionManager()
        manager.session_file = session_file
        
        mock_playwright = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        
        mock_playwright.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        
        manager.create_browser_context(mock_playwright)
        
        context_kwargs = mock_browser.new_context.call_args[1]
        assert context_kwargs['storage_state'] == str(session_file)

    def test_adds_webdriver_hiding_script(self):
        """Should add init script to hide webdriver property."""
        manager = SessionManager()
        
        mock_playwright = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        
        mock_playwright.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        
        manager.create_browser_context(mock_playwright)
        
        # Verify init script was added
        mock_page.add_init_script.assert_called_once()
        script = mock_page.add_init_script.call_args[0][0]
        assert 'webdriver' in script

    def test_saves_session_when_requested(self, tmp_path):
        """Should save session state when save_session=True."""
        manager = SessionManager()
        manager.session_file = tmp_path / "session.json"
        
        mock_playwright = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        
        mock_playwright.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        
        manager.create_browser_context(mock_playwright, save_session=True)
        
        mock_context.storage_state.assert_called_once_with(
            path=str(manager.session_file)
        )

    def test_returns_browser_context_page_tuple(self):
        """Should return tuple of (browser, context, page)."""
        manager = SessionManager()
        
        mock_playwright = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        
        mock_playwright.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        
        result = manager.create_browser_context(mock_playwright)
        
        assert result == (mock_browser, mock_context, mock_page)
