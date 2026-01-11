"""
Unit tests for the async scroller module.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio


class TestAsyncScroller:
    """Tests for AsyncScroller class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        config = Mock()
        config.min_human_delay = 0.5
        config.max_human_delay = 1.5
        return config

    @pytest.fixture
    def scroller(self, mock_config):
        """Create an AsyncScroller with mocked config."""
        with patch('fetcher.async_scroller.get_config', return_value=mock_config):
            from fetcher.async_scroller import AsyncScroller
            return AsyncScroller()

    @pytest.mark.asyncio
    async def test_delay_uses_config_defaults(self, scroller, mock_config):
        """Test delay uses config values when no args provided."""
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            await scroller.delay()
            
            mock_sleep.assert_called_once()
            delay_value = mock_sleep.call_args[0][0]
            assert mock_config.min_human_delay <= delay_value <= mock_config.max_human_delay

    @pytest.mark.asyncio
    async def test_delay_uses_custom_values(self, scroller):
        """Test delay uses custom values when provided."""
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            await scroller.delay(min_seconds=0.1, max_seconds=0.2)
            
            mock_sleep.assert_called_once()
            delay_value = mock_sleep.call_args[0][0]
            assert 0.1 <= delay_value <= 0.2

    @pytest.mark.asyncio
    async def test_random_scroll_pattern_normal(self, scroller):
        """Test random_scroll_pattern with normal scroll."""
        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock()
        
        await scroller.random_scroll_pattern(mock_page, deep_scroll=False)
        
        mock_page.evaluate.assert_called()
        # Check scroll was positive (scrolling down)
        call_args = mock_page.evaluate.call_args_list[0][0][0]
        assert 'scrollBy' in call_args

    @pytest.mark.asyncio
    async def test_random_scroll_pattern_deep(self, scroller):
        """Test random_scroll_pattern with deep scroll."""
        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock()
        
        await scroller.random_scroll_pattern(mock_page, deep_scroll=True)
        
        mock_page.evaluate.assert_called()

    @pytest.mark.asyncio
    async def test_adaptive_scroll_returns_element_count(self, scroller):
        """Test adaptive_scroll returns count of matching elements."""
        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock()
        mock_page.wait_for_function = AsyncMock()
        mock_page.query_selector_all = AsyncMock(return_value=[Mock(), Mock(), Mock()])
        
        count = await scroller.adaptive_scroll(mock_page, prev_count=0)
        
        assert count == 3

    @pytest.mark.asyncio
    async def test_adaptive_scroll_handles_timeout(self, scroller):
        """Test adaptive_scroll handles timeout gracefully."""
        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock()
        mock_page.wait_for_function = AsyncMock(side_effect=Exception("Timeout"))
        mock_page.query_selector_all = AsyncMock(return_value=[Mock(), Mock()])
        
        count = await scroller.adaptive_scroll(mock_page, prev_count=0, max_wait=0.1)
        
        assert count == 2
        assert scroller._adaptive_scroll_stats['content_loaded_timeout'] == 1

    @pytest.mark.asyncio
    async def test_adaptive_scroll_tracks_stats(self, scroller):
        """Test adaptive_scroll updates statistics."""
        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock()
        mock_page.wait_for_function = AsyncMock()
        mock_page.query_selector_all = AsyncMock(return_value=[])
        
        initial_scrolls = scroller._adaptive_scroll_stats['total_scrolls']
        
        await scroller.adaptive_scroll(mock_page)
        
        assert scroller._adaptive_scroll_stats['total_scrolls'] == initial_scrolls + 1

    @pytest.mark.asyncio
    async def test_scroll_to_bottom(self, scroller):
        """Test scroll_to_bottom scrolls to page bottom."""
        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock()
        
        await scroller.scroll_to_bottom(mock_page)
        
        mock_page.evaluate.assert_called_once()
        call_args = mock_page.evaluate.call_args[0][0]
        assert 'scrollTo' in call_args
        assert 'scrollHeight' in call_args

    @pytest.mark.asyncio
    async def test_scroll_up_slightly(self, scroller):
        """Test scroll_up_slightly scrolls up by specified amount."""
        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock()
        
        await scroller.scroll_up_slightly(mock_page, amount=300)
        
        mock_page.evaluate.assert_called_once_with("window.scrollBy(0, -300)")

    @pytest.mark.asyncio
    async def test_wait_for_content_success(self, scroller):
        """Test wait_for_content returns True when content found."""
        mock_page = AsyncMock()
        mock_page.wait_for_selector = AsyncMock()
        
        result = await scroller.wait_for_content(mock_page, 'article', timeout=5000)
        
        assert result is True
        mock_page.wait_for_selector.assert_called_once_with('article', timeout=5000)

    @pytest.mark.asyncio
    async def test_wait_for_content_timeout(self, scroller):
        """Test wait_for_content returns False on timeout."""
        mock_page = AsyncMock()
        mock_page.wait_for_selector = AsyncMock(side_effect=Exception("Timeout"))
        
        result = await scroller.wait_for_content(mock_page, 'article', timeout=100)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_try_recovery_strategies_first(self, scroller):
        """Test try_recovery_strategies uses first strategy for attempt 1."""
        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock()
        
        result = await scroller.try_recovery_strategies(mock_page, attempt=1)
        
        assert result is True
        # First strategy does scroll jiggle
        assert mock_page.evaluate.call_count >= 2

    @pytest.mark.asyncio
    async def test_try_recovery_strategies_second(self, scroller):
        """Test try_recovery_strategies uses second strategy for attempt 2."""
        mock_page = AsyncMock()
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await scroller.try_recovery_strategies(mock_page, attempt=2)
        
        assert result is True

    @pytest.mark.asyncio
    async def test_try_recovery_strategies_third(self, scroller):
        """Test try_recovery_strategies uses third strategy for attempt 3."""
        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock()
        
        result = await scroller.try_recovery_strategies(mock_page, attempt=3)
        
        assert result is True
        # Third strategy scrolls to top then bottom
        assert mock_page.evaluate.call_count >= 2

    @pytest.mark.asyncio
    async def test_try_recovery_strategies_beyond_available(self, scroller):
        """Test try_recovery_strategies returns False when no more strategies."""
        mock_page = AsyncMock()
        
        result = await scroller.try_recovery_strategies(mock_page, attempt=10)
        
        assert result is False


class TestAsyncScrollerSingleton:
    """Tests for AsyncScroller singleton pattern."""

    def test_get_async_scroller_returns_singleton(self):
        """Test get_async_scroller returns same instance."""
        with patch('fetcher.async_scroller.get_config'):
            # Reset singleton
            import fetcher.async_scroller as scroller_module
            scroller_module._async_scroller = None
            
            from fetcher.async_scroller import get_async_scroller
            
            scroller1 = get_async_scroller()
            scroller2 = get_async_scroller()
            
            assert scroller1 is scroller2

    def test_get_async_scroller_creates_instance(self):
        """Test get_async_scroller creates new instance when None."""
        with patch('fetcher.async_scroller.get_config'):
            import fetcher.async_scroller as scroller_module
            scroller_module._async_scroller = None
            
            from fetcher.async_scroller import get_async_scroller, AsyncScroller
            
            scroller = get_async_scroller()
            
            assert isinstance(scroller, AsyncScroller)
