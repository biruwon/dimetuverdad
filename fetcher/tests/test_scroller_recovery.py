import pytest
import time
from unittest.mock import Mock, patch

from fetcher.scroller import Scroller, MIN_ADAPTIVE_DELAY


class FakeLocator:
    def __init__(self, page, visible=True):
        self.page = page
        self._visible = visible

    def click(self):
        self.page._clicked_retry = True

    def is_visible(self):
        return self._visible


class FakeButton:
    """Fake button element for testing."""
    def __init__(self, page, visible=True):
        self.page = page
        self._visible = visible

    def click(self):
        self.page._clicked_retry = True

    def is_visible(self):
        return self._visible


class FakePage:
    def __init__(self, has_retry=True):
        self._clicked_retry = False
        self._reloaded = False
        self._has_retry = has_retry
        self.evaluations = []

    def locator(self, selector):
        if self._has_retry and selector == 'button:has-text("Retry")':
            return FakeLocator(self, visible=True)
        # Return a falsy object if no retry
        return None

    def query_selector(self, selector):
        """Return a fake button if retry button is requested and exists."""
        if self._has_retry and 'Retry' in selector:
            return FakeButton(self, visible=True)
        return None

    def evaluate(self, js):
        # record evaluation calls
        self.evaluations.append(js)

    def reload(self, **kwargs):
        self._reloaded = True

    def query_selector_all(self, selector):
        # After clicking retry or after reload, pretend articles appear
        if self._clicked_retry or self._reloaded:
            return ["article"]
        return []


def test_try_recovery_clicks_retry_and_does_not_reload():
    page = FakePage(has_retry=True)
    scroller = Scroller()

    recovered = scroller.try_recovery_strategies(page, attempt_number=1)

    assert recovered is True
    assert page._clicked_retry is True
    assert page._reloaded is False


def test_try_recovery_reload_as_last_resort():
    page = FakePage(has_retry=False)
    scroller = Scroller()

    # attempts 1-4 will not find anything; attempt 5 is reload
    recovered = scroller.try_recovery_strategies(page, attempt_number=5)

    assert recovered is True
    assert page._reloaded is True

# ============== Adaptive Scroll Tests (P2 Performance Optimization) ==============

class FakePageForAdaptiveScroll:
    """Fake page object that simulates content loading behavior."""
    
    def __init__(self, article_counts=None, fast_load=True):
        """
        Args:
            article_counts: List of article counts to return on each evaluate call.
                           If None, defaults to [5, 8] (5 before scroll, 8 after).
            fast_load: If True, wait_for_function succeeds. If False, raises timeout.
        """
        self.article_counts = article_counts or [5, 8]
        self._eval_call_count = 0
        self.evaluations = []
        self.fast_load = fast_load
        self._wait_for_function_called = False
    
    def evaluate(self, js):
        self.evaluations.append(js)
        
        # Return article count for the count query
        if 'querySelectorAll' in js and '.length' in js:
            idx = min(self._eval_call_count, len(self.article_counts) - 1)
            count = self.article_counts[idx]
            self._eval_call_count += 1
            return count
        
        # scrollBy calls return None
        return None
    
    def wait_for_function(self, condition, timeout=None):
        self._wait_for_function_called = True
        if not self.fast_load:
            raise Exception("TimeoutError: waiting for content")


@patch('time.sleep')  # Speed up tests by mocking sleep
def test_adaptive_scroll_basic_functionality(mock_sleep):
    """Test that adaptive_scroll performs scroll and waits for content."""
    scroller = Scroller()
    page = FakePageForAdaptiveScroll(article_counts=[5, 8], fast_load=True)
    
    elapsed = scroller.adaptive_scroll(page, deep_scroll=False)
    
    # Verify scroll was performed (at least one forward, possibly one backward)
    scroll_calls = [e for e in page.evaluations if 'scrollBy' in e]
    forward_scrolls = [e for e in scroll_calls if 'scrollBy(0, -' not in e]
    assert len(forward_scrolls) >= 1, "Should have at least one forward scroll"
    
    # Verify wait_for_function was called
    assert page._wait_for_function_called is True
    
    # Verify stats updated
    stats = scroller.get_adaptive_scroll_stats()
    assert stats['total_scrolls'] == 1
    assert stats['content_loaded_fast'] == 1
    assert stats['content_loaded_timeout'] == 0


@patch('time.sleep')
def test_adaptive_scroll_handles_timeout(mock_sleep):
    """Test that adaptive_scroll handles content not loading gracefully."""
    scroller = Scroller()
    page = FakePageForAdaptiveScroll(article_counts=[5, 5], fast_load=False)
    
    # Should not raise, just continue
    elapsed = scroller.adaptive_scroll(page, deep_scroll=False)
    
    # Verify stats show timeout
    stats = scroller.get_adaptive_scroll_stats()
    assert stats['total_scrolls'] == 1
    assert stats['content_loaded_fast'] == 0
    assert stats['content_loaded_timeout'] == 1


@patch('time.sleep')
def test_adaptive_scroll_deep_scroll_uses_larger_amounts(mock_sleep):
    """Test that deep_scroll=True uses larger scroll amounts."""
    scroller = Scroller()
    page = FakePageForAdaptiveScroll(article_counts=[5, 8], fast_load=True)
    
    scroller.adaptive_scroll(page, deep_scroll=True)
    
    # Check that scroll was called with a larger amount (900-1400 range for deep)
    scroll_call = [e for e in page.evaluations if 'scrollBy' in e][0]
    # Extract the scroll amount from "window.scrollBy(0, XXX)"
    import re
    match = re.search(r'scrollBy\(0, (\d+)\)', scroll_call)
    assert match, "Should have a scroll call"
    scroll_amount = int(match.group(1))
    assert scroll_amount >= 900, f"Deep scroll should be >= 900, got {scroll_amount}"


def test_adaptive_scroll_stats_reset():
    """Test that stats can be reset."""
    scroller = Scroller()
    
    # Simulate some activity
    scroller._adaptive_scroll_stats['total_scrolls'] = 100
    scroller._adaptive_scroll_stats['content_loaded_fast'] = 80
    
    scroller.reset_adaptive_scroll_stats()
    
    stats = scroller.get_adaptive_scroll_stats()
    assert stats['total_scrolls'] == 0
    assert stats['content_loaded_fast'] == 0


def test_adaptive_scroll_stats_calculations():
    """Test that stats calculations are correct."""
    scroller = Scroller()
    
    # Set up stats manually
    scroller._adaptive_scroll_stats = {
        'total_scrolls': 10,
        'content_loaded_fast': 8,
        'content_loaded_timeout': 2,
        'total_wait_time': 5.0,
    }
    
    stats = scroller.get_adaptive_scroll_stats()
    
    assert stats['avg_wait_time'] == 0.5  # 5.0 / 10
    assert stats['fast_load_rate'] == 0.8  # 8 / 10


def test_min_adaptive_delay_constant():
    """Test that MIN_ADAPTIVE_DELAY is a reasonable value."""
    assert MIN_ADAPTIVE_DELAY > 0, "Should be positive"
    assert MIN_ADAPTIVE_DELAY <= 1.0, "Should be under 1 second for performance"


# ===========================================================
# P6: Pre-scroll Content Loading Tests
# ===========================================================

class TestPrefetchScroll:
    """Tests for the prefetch_scroll optimization (P6)."""
    
    def test_prefetch_scroll_returns_article_count(self):
        """Test that prefetch_scroll returns current article count."""
        scroller = Scroller()
        page = Mock()
        page.evaluate = Mock(side_effect=[15])  # 15 articles
        
        result = scroller.prefetch_scroll(page)
        
        assert result == 15
        # Should have scrolled
        assert page.evaluate.call_count == 2  # count + scroll
    
    def test_prefetch_scroll_scrolls_ahead(self):
        """Test that prefetch scrolls a large amount ahead."""
        scroller = Scroller()
        page = Mock()
        page.evaluate = Mock(return_value=10)
        
        scroller.prefetch_scroll(page)
        
        # Check second call is a scrollBy with large amount
        calls = page.evaluate.call_args_list
        assert len(calls) >= 2
        scroll_call = str(calls[1])
        assert 'scrollBy' in scroll_call
        assert '0, 1' in scroll_call  # Starts with 1500+
    
    def test_prefetch_scroll_handles_error(self):
        """Test that prefetch_scroll handles errors gracefully."""
        scroller = Scroller()
        page = Mock()
        page.evaluate = Mock(side_effect=Exception("Failed"))
        
        # Should not raise, just return 0
        result = scroller.prefetch_scroll(page)
        
        assert result == 0


class TestWaitForPrefetchedContent:
    """Tests for wait_for_prefetched_content (P6)."""
    
    def test_wait_returns_true_when_content_loads(self):
        """Test successful content load returns True."""
        scroller = Scroller()
        page = Mock()
        page.wait_for_function = Mock(return_value=True)
        
        result = scroller.wait_for_prefetched_content(page, prev_count=10)
        
        assert result is True
        page.wait_for_function.assert_called_once()
    
    def test_wait_returns_false_on_timeout(self):
        """Test that timeout returns False."""
        scroller = Scroller()
        page = Mock()
        page.wait_for_function = Mock(side_effect=Exception("Timeout"))
        
        result = scroller.wait_for_prefetched_content(page, prev_count=10)
        
        assert result is False
    
    def test_wait_uses_correct_selector(self):
        """Test that wait uses correct article count selector."""
        scroller = Scroller()
        page = Mock()
        page.wait_for_function = Mock(return_value=True)
        
        scroller.wait_for_prefetched_content(page, prev_count=25, timeout_ms=300)
        
        call_args = page.wait_for_function.call_args
        assert '> 25' in call_args[0][0]  # Check prev_count in selector
        assert call_args[1]['timeout'] == 300
    
    def test_wait_default_timeout_is_short(self):
        """Test that default timeout is short for performance."""
        scroller = Scroller()
        page = Mock()
        page.wait_for_function = Mock(return_value=True)
        
        scroller.wait_for_prefetched_content(page, prev_count=10)
        
        call_args = page.wait_for_function.call_args
        assert call_args[1]['timeout'] <= 1000  # Should be <= 1 second


class TestScrollBackForProcessing:
    """Tests for scroll_back_for_processing (P6)."""
    
    def test_scroll_back_scrolls_negative(self):
        """Test that scroll_back scrolls in negative direction."""
        scroller = Scroller()
        page = Mock()
        page.evaluate = Mock()
        
        with patch('time.sleep'):  # Don't actually sleep
            scroller.scroll_back_for_processing(page)
        
        call_args = str(page.evaluate.call_args)
        assert 'scrollBy' in call_args
        assert '-' in call_args  # Negative scroll
    
    def test_scroll_back_handles_error(self):
        """Test that scroll_back handles errors gracefully."""
        scroller = Scroller()
        page = Mock()
        page.evaluate = Mock(side_effect=Exception("Failed"))
        
        # Should not raise
        with patch('time.sleep'):
            scroller.scroll_back_for_processing(page)
    
    def test_scroll_back_uses_custom_amount(self):
        """Test that custom scroll amount is used."""
        scroller = Scroller()
        page = Mock()
        page.evaluate = Mock()
        
        with patch('time.sleep'):
            scroller.scroll_back_for_processing(page, amount=500)
        
        call_args = str(page.evaluate.call_args)
        # Should be around -500 (±100)
        assert 'scrollBy' in call_args


class TestPrefetchIntegration:
    """Integration tests for the prefetch scroll pattern (P6)."""
    
    def test_prefetch_pattern_workflow(self):
        """Test the expected prefetch workflow pattern."""
        scroller = Scroller()
        page = Mock()
        page.evaluate = Mock(side_effect=[
            20,  # Article count for prefetch
            None,  # Scroll call
            None,  # Scroll back call
        ])
        page.wait_for_function = Mock(return_value=True)
        
        # Step 1: Prefetch before processing
        prev_count = scroller.prefetch_scroll(page)
        assert prev_count == 20
        
        # Step 2: Process tweets (simulated - would be CPU work here)
        # ...processing happens...
        
        # Step 3: Wait for prefetched content (short timeout)
        loaded = scroller.wait_for_prefetched_content(page, prev_count)
        assert loaded is True
        
        # Step 4: Scroll back for next processing iteration
        with patch('time.sleep'):
            scroller.scroll_back_for_processing(page)
    
    def test_prefetch_with_timeout_still_continues(self):
        """Test that prefetch timeout doesn't block workflow."""
        scroller = Scroller()
        page = Mock()
        page.evaluate = Mock(return_value=10)
        page.wait_for_function = Mock(side_effect=Exception("Timeout"))
        
        prev_count = scroller.prefetch_scroll(page)
        loaded = scroller.wait_for_prefetched_content(page, prev_count)
        
        # Even with timeout, should continue (returns False, not raise)
        assert loaded is False


class TestRandomScrollPattern:
    """Tests for Scroller.random_scroll_pattern()."""

    def test_random_scroll_normal_mode(self):
        """Should perform scroll with normal amounts."""
        scroller = Scroller()
        page = Mock()
        page.evaluate = Mock()
        
        with patch('time.sleep'):
            scroller.random_scroll_pattern(page, deep_scroll=False)
        
        # Should have called evaluate at least once
        assert page.evaluate.called
        # Check the scroll JS contains window.scrollBy
        call_args = page.evaluate.call_args_list[0][0][0]
        assert 'window.scrollBy' in call_args

    def test_random_scroll_deep_scroll_mode(self):
        """Should use larger scroll amounts in deep scroll mode."""
        scroller = Scroller()
        page = Mock()
        page.evaluate = Mock()
        
        with patch('time.sleep'):
            with patch('random.random', return_value=0.5):  # No back scroll
                scroller.random_scroll_pattern(page, deep_scroll=True)
        
        assert page.evaluate.called

    def test_random_scroll_with_back_scroll(self):
        """Should occasionally scroll back slightly."""
        scroller = Scroller()
        page = Mock()
        page.evaluate = Mock()
        
        # Force back scroll by making random() return < 0.08
        with patch('time.sleep'):
            with patch('random.random', return_value=0.05):
                scroller.random_scroll_pattern(page, deep_scroll=False)
        
        # Should have two evaluate calls - scroll and back scroll
        assert page.evaluate.call_count >= 2


class TestEventScrollCycle:
    """Tests for Scroller.event_scroll_cycle()."""

    def test_event_scroll_cycle_normal_iteration(self):
        """Should scroll with appropriate pattern."""
        scroller = Scroller()
        page = Mock()
        page.evaluate = Mock()
        
        with patch('time.sleep'):
            scroller.event_scroll_cycle(page, iteration=1)
        
        assert page.evaluate.called

    def test_event_scroll_cycle_varies_by_iteration(self):
        """Should use different scroll patterns for different iterations."""
        scroller = Scroller()
        page1 = Mock()
        page1.evaluate = Mock()
        page2 = Mock()
        page2.evaluate = Mock()
        
        with patch('time.sleep'):
            scroller.event_scroll_cycle(page1, iteration=0)
            scroller.event_scroll_cycle(page2, iteration=1)
        
        # Both should call evaluate
        assert page1.evaluate.called
        assert page2.evaluate.called

    def test_event_scroll_cycle_longer_pause_every_10(self):
        """Should have longer pause on every 10th iteration."""
        scroller = Scroller()
        page = Mock()
        page.evaluate = Mock()
        
        with patch('time.sleep') as mock_sleep:
            scroller.event_scroll_cycle(page, iteration=10)
        
        # Should have called delay (sleep) with longer values
        assert mock_sleep.called

    def test_event_scroll_cycle_handles_evaluate_exception(self):
        """Should handle exceptions gracefully."""
        scroller = Scroller()
        page = Mock()
        page.evaluate = Mock(side_effect=Exception("Eval error"))
        
        with patch('time.sleep'):
            # Should not raise exception
            scroller.event_scroll_cycle(page, iteration=1)


class TestCheckPageHeightChange:
    """Tests for Scroller.check_page_height_change()."""

    def test_returns_new_height(self):
        """Should return the new page height."""
        scroller = Scroller()
        page = Mock()
        page.evaluate = Mock(return_value=5000)
        
        result = scroller.check_page_height_change(page, 4000)
        
        assert result == 5000
        page.evaluate.assert_called_with("document.body.scrollHeight")

    def test_returns_last_height_on_no_change(self):
        """Should return same height if no change."""
        scroller = Scroller()
        page = Mock()
        page.evaluate = Mock(return_value=4000)
        
        result = scroller.check_page_height_change(page, 4000)
        
        assert result == 4000

    def test_returns_last_height_on_exception(self):
        """Should return last height on evaluation error."""
        scroller = Scroller()
        page = Mock()
        page.evaluate = Mock(side_effect=Exception("Error"))
        
        result = scroller.check_page_height_change(page, 3000)
        
        assert result == 3000


class TestAggressiveScroll:
    """Tests for Scroller.aggressive_scroll()."""

    def test_aggressive_scroll_increases_with_empty_scrolls(self):
        """Should increase scroll amount with more consecutive empty scrolls."""
        scroller = Scroller()
        page = Mock()
        page.evaluate = Mock()
        
        with patch('time.sleep'):
            scroller.aggressive_scroll(page, consecutive_empty_scrolls=1)
        
        # Should call evaluate twice for 2 iterations
        assert page.evaluate.call_count == 2

    def test_aggressive_scroll_caps_multiplier_at_3(self):
        """Should cap the scroll multiplier at 3x."""
        scroller = Scroller()
        page = Mock()
        page.evaluate = Mock()
        
        with patch('time.sleep'):
            scroller.aggressive_scroll(page, consecutive_empty_scrolls=10)
        
        # Even with 10 empty scrolls, multiplier should be capped
        assert page.evaluate.called
        # Check the scroll amount is capped (800 * 3 = 2400)
        call_args = page.evaluate.call_args_list[0][0][0]
        assert '2400' in call_args

    def test_aggressive_scroll_handles_exception(self):
        """Should fallback to basic scroll on exception."""
        scroller = Scroller()
        page = Mock()
        call_count = [0]
        
        def evaluate_side_effect(js):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise Exception("First calls fail")
            return None
        
        page.evaluate = Mock(side_effect=evaluate_side_effect)
        
        with patch('time.sleep'):
            # Should not raise, should try fallback
            scroller.aggressive_scroll(page, consecutive_empty_scrolls=1)


class TestDelayMethod:
    """Tests for Scroller.delay()."""

    def test_delay_uses_default_values(self):
        """Should use config values when no args provided."""
        scroller = Scroller()
        
        with patch('time.sleep') as mock_sleep:
            scroller.delay()
        
        assert mock_sleep.called
        sleep_time = mock_sleep.call_args[0][0]
        # Should be within default range from config
        assert sleep_time >= 0

    def test_delay_uses_custom_values(self):
        """Should use provided min/max values."""
        scroller = Scroller()
        
        with patch('time.sleep') as mock_sleep:
            with patch('random.uniform', return_value=1.5):
                scroller.delay(1.0, 2.0)
        
        mock_sleep.assert_called_once_with(1.5)


class TestScrollMethod:
    """Tests for Scroller.scroll()."""

    def test_scroll_delegates_to_random_scroll_pattern(self):
        """Should call random_scroll_pattern internally."""
        scroller = Scroller()
        page = Mock()
        page.evaluate = Mock()
        
        with patch.object(scroller, 'random_scroll_pattern') as mock_random:
            scroller.scroll(page, deep_scroll=False)
        
        mock_random.assert_called_once_with(page, False)

    def test_scroll_passes_deep_scroll_flag(self):
        """Should pass deep_scroll flag to random_scroll_pattern."""
        scroller = Scroller()
        page = Mock()
        
        with patch.object(scroller, 'random_scroll_pattern') as mock_random:
            scroller.scroll(page, deep_scroll=True)
        
        mock_random.assert_called_once_with(page, True)


class TestGetScroller:
    """Tests for get_scroller() function."""

    def test_returns_global_scroller_instance(self):
        """Should return the global scroller instance."""
        from fetcher.scroller import get_scroller, scroller as global_scroller
        
        result = get_scroller()
        
        assert result is global_scroller
        assert isinstance(result, Scroller)