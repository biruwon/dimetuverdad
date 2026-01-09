"""
Tests for fetcher/config.py module.

Tests configuration loading, environment variable overrides, and runtime updates.
"""

import pytest
import os
from unittest.mock import patch


class TestFetcherConfig:
    """Test FetcherConfig dataclass and configuration management."""

    def test_headless_default_true_from_env(self):
        """Test that headless defaults to True when FETCHER_HEADLESS is not set."""
        with patch.dict(os.environ, {}, clear=False):
            # Remove the env var if it exists to test default
            if 'FETCHER_HEADLESS' in os.environ:
                del os.environ['FETCHER_HEADLESS']
            
            # Import fresh config
            from importlib import reload
            import fetcher.config as config_module
            reload(config_module)
            
            # Default is True (headless for production speed)
            assert config_module.config.headless is True

    def test_headless_false_from_env(self):
        """Test that FETCHER_HEADLESS=false disables headless mode."""
        with patch.dict(os.environ, {'FETCHER_HEADLESS': 'false'}, clear=False):
            from importlib import reload
            import fetcher.config as config_module
            reload(config_module)
            
            assert config_module.config.headless is False

    def test_headless_true_explicit_from_env(self):
        """Test that FETCHER_HEADLESS=true enables headless mode."""
        with patch.dict(os.environ, {'FETCHER_HEADLESS': 'true'}, clear=False):
            from importlib import reload
            import fetcher.config as config_module
            reload(config_module)
            
            assert config_module.config.headless is True

    def test_slow_mo_default(self):
        """Test default slow_mo value."""
        with patch.dict(os.environ, {}, clear=False):
            if 'FETCHER_SLOW_MO' in os.environ:
                del os.environ['FETCHER_SLOW_MO']
            
            from importlib import reload
            import fetcher.config as config_module
            reload(config_module)
            
            assert config_module.config.slow_mo == 50

    def test_slow_mo_from_env(self):
        """Test slow_mo can be overridden via environment."""
        with patch.dict(os.environ, {'FETCHER_SLOW_MO': '100'}, clear=False):
            from importlib import reload
            import fetcher.config as config_module
            reload(config_module)
            
            assert config_module.config.slow_mo == 100

    def test_update_config_runtime(self):
        """Test runtime configuration updates."""
        from fetcher.config import get_config, update_config
        
        cfg = get_config()
        original_headless = cfg.headless
        
        # Update headless to opposite value
        update_config(headless=not original_headless)
        assert cfg.headless == (not original_headless)
        
        # Restore
        update_config(headless=original_headless)
        assert cfg.headless == original_headless

    def test_update_config_unknown_key_raises(self):
        """Test that updating an unknown key raises ValueError."""
        from fetcher.config import update_config
        
        with pytest.raises(ValueError, match="Unknown configuration key"):
            update_config(unknown_key="value")

    def test_viewport_dimensions(self):
        """Test viewport dimensions are set correctly."""
        from fetcher.config import get_config
        
        cfg = get_config()
        assert cfg.viewport_width == 1280
        assert cfg.viewport_height == 720

    def test_timeout_defaults(self):
        """Test timeout defaults are reasonable."""
        from fetcher.config import get_config
        
        cfg = get_config()
        assert cfg.page_load_timeout == 30000
        assert cfg.element_wait_timeout == 15000
        assert cfg.login_verification_timeout == 15000

    def test_scroll_amounts_initialization(self):
        """Test scroll amounts are initialized properly."""
        from fetcher.config import get_config
        
        cfg = get_config()
        assert cfg.scroll_amounts is not None
        assert len(cfg.scroll_amounts) > 0
        assert all(isinstance(x, int) for x in cfg.scroll_amounts)

    def test_user_agents_initialization(self):
        """Test user agents list is initialized."""
        from fetcher.config import get_config
        
        cfg = get_config()
        assert cfg.user_agents is not None
        assert len(cfg.user_agents) > 0
        assert all("Mozilla" in ua for ua in cfg.user_agents)
