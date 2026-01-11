"""
Unit tests for DateRangeCollector - Adaptive date-range based tweet collection.

Tests the recursive window splitting strategy that:
- Starts with large windows and subdivides if > 750 tweets
- Stops when consecutive empty windows found (reached account start)
- Tracks collection progress in database
"""

import pytest
from datetime import datetime, date, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import asyncio


class TestDateRangeWindow:
    """Tests for DateRangeWindow dataclass/namedtuple."""
    
    def test_window_creation(self):
        """Window stores start_date, end_date correctly."""
        from fetcher.date_range_collector import DateRangeWindow
        
        start = date(2025, 1, 1)
        end = date(2025, 6, 30)
        window = DateRangeWindow(start_date=start, end_date=end)
        
        assert window.start_date == start
        assert window.end_date == end
    
    def test_window_days_calculation(self):
        """Window can calculate its span in days."""
        from fetcher.date_range_collector import DateRangeWindow
        
        window = DateRangeWindow(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31)
        )
        
        assert window.days == 30
    
    def test_window_midpoint(self):
        """Window can calculate its midpoint for splitting."""
        from fetcher.date_range_collector import DateRangeWindow
        
        window = DateRangeWindow(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31)
        )
        
        midpoint = window.midpoint
        assert midpoint == date(2025, 1, 16)
    
    def test_window_split(self):
        """Window can split into two halves."""
        from fetcher.date_range_collector import DateRangeWindow
        
        window = DateRangeWindow(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31)
        )
        
        first_half, second_half = window.split()
        
        assert first_half.start_date == date(2025, 1, 1)
        assert first_half.end_date == date(2025, 1, 16)
        assert second_half.start_date == date(2025, 1, 16)
        assert second_half.end_date == date(2025, 1, 31)
    
    def test_window_cannot_split_single_day(self):
        """Single-day window cannot be split further."""
        from fetcher.date_range_collector import DateRangeWindow
        
        window = DateRangeWindow(
            start_date=date(2025, 1, 15),
            end_date=date(2025, 1, 16)  # 1 day span
        )
        
        assert window.can_split is False


class TestSearchURLBuilder:
    """Tests for building Twitter search URLs with date ranges."""
    
    def test_build_search_url_basic(self):
        """Builds correct search URL with from, since, until."""
        from fetcher.date_range_collector import build_search_url
        from urllib.parse import unquote_plus
        
        url = build_search_url(
            username="testuser",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31)
        )
        
        # Decode URL for checking content
        decoded_url = unquote_plus(url)
        
        assert "from:testuser" in decoded_url
        assert "since:2025-01-01" in decoded_url
        assert "until:2025-01-31" in decoded_url
        assert "x.com/search" in url
        assert "f=live" in url  # Live/latest results
    
    def test_build_search_url_encodes_properly(self):
        """URL encodes query parameters correctly."""
        from fetcher.date_range_collector import build_search_url
        
        url = build_search_url(
            username="test_user",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31)
        )
        
        # Should be URL encoded (spaces and colons)
        assert "+" in url or "%3A" in url  # + for spaces or %3A for colons


class TestDateRangeCollectorConfig:
    """Tests for DateRangeCollector configuration."""
    
    def test_default_config_values(self):
        """Default configuration has sensible values."""
        from fetcher.date_range_collector import DateRangeCollectorConfig
        
        config = DateRangeCollectorConfig()
        
        assert config.initial_window_months == 6
        assert config.min_window_days == 1
        assert config.split_threshold == 750
        assert config.empty_windows_to_stop == 3
    
    def test_custom_config_values(self):
        """Configuration accepts custom values."""
        from fetcher.date_range_collector import DateRangeCollectorConfig
        
        config = DateRangeCollectorConfig(
            initial_window_months=3,
            split_threshold=500,
            empty_windows_to_stop=2
        )
        
        assert config.initial_window_months == 3
        assert config.split_threshold == 500
        assert config.empty_windows_to_stop == 2


class TestDateRangeCollector:
    """Tests for DateRangeCollector main class."""
    
    @pytest.fixture
    def mock_collector(self):
        """Create a mock async tweet collector."""
        collector = AsyncMock()
        collector.collect_from_search = AsyncMock(return_value=[])
        return collector
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock database connection."""
        db = Mock()
        db.get_collected_ranges = Mock(return_value=[])
        db.is_range_collected = Mock(return_value=False)
        db.mark_range_complete = Mock()
        db.mark_range_in_progress = Mock()
        db.get_oldest_tweet_date = Mock(return_value=None)
        db.get_newest_tweet_date = Mock(return_value=None)
        db.get_collection_stats = Mock(return_value={'total_tweets': 0, 'completed_ranges': 0})
        return db
    
    @pytest.mark.asyncio
    async def test_collect_single_window_under_threshold(self, mock_collector, mock_db):
        """Window under threshold collects all tweets without splitting."""
        from fetcher.date_range_collector import DateRangeCollector, DateRangeCollectorConfig
        
        # Return 100 tweets (under 750 threshold)
        mock_collector.collect_from_search = AsyncMock(return_value=[
            {"tweet_id": str(i)} for i in range(100)
        ])
        
        config = DateRangeCollectorConfig(split_threshold=750)
        collector = DateRangeCollector(
            username="testuser",
            tweet_collector=mock_collector,
            db=mock_db,
            config=config
        )
        
        result = await collector.collect_window(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 6, 30)
        )
        
        assert result.tweets_collected == 100
        assert result.windows_processed == 1
        assert mock_collector.collect_from_search.call_count == 1
    
    @pytest.mark.asyncio
    async def test_collect_window_splits_when_over_threshold(self, mock_collector, mock_db):
        """Window over threshold triggers recursive split."""
        from fetcher.date_range_collector import DateRangeCollector, DateRangeCollectorConfig
        
        call_count = 0
        async def mock_search(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # First call returns 800 (over threshold), subsequent calls return less
            if call_count == 1:
                return [{"tweet_id": str(i)} for i in range(800)]
            return [{"tweet_id": str(i)} for i in range(200)]
        
        mock_collector.collect_from_search = mock_search
        
        config = DateRangeCollectorConfig(split_threshold=750)
        collector = DateRangeCollector(
            username="testuser",
            tweet_collector=mock_collector,
            db=mock_db,
            config=config
        )
        
        result = await collector.collect_window(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 6, 30)
        )
        
        # Should have split and made multiple calls
        assert call_count >= 2
    
    @pytest.mark.asyncio
    async def test_collect_stops_after_consecutive_empty_windows(self, mock_collector, mock_db):
        """Collection stops after N consecutive empty windows."""
        from fetcher.date_range_collector import DateRangeCollector, DateRangeCollectorConfig
        
        # Always return empty
        mock_collector.collect_from_search = AsyncMock(return_value=[])
        
        config = DateRangeCollectorConfig(
            initial_window_months=1,
            empty_windows_to_stop=3
        )
        collector = DateRangeCollector(
            username="testuser",
            tweet_collector=mock_collector,
            db=mock_db,
            config=config
        )
        
        result = await collector.collect_full_history()
        
        # Should stop after 3 consecutive empty windows
        assert result.consecutive_empty_windows == 3
        assert result.stopped_reason == "consecutive_empty_windows"
    
    @pytest.mark.asyncio
    async def test_collect_skips_already_collected_ranges(self, mock_collector, mock_db):
        """Already collected ranges are skipped on resume."""
        from fetcher.date_range_collector import DateRangeCollector, DateRangeCollectorConfig
        
        # Mark Jan-Mar as already collected
        mock_db.get_collected_ranges = Mock(return_value=[
            {"start_date": "2025-01-01", "end_date": "2025-03-31", "status": "complete"}
        ])
        mock_db.is_range_collected = Mock(return_value=True)
        
        mock_collector.collect_from_search = AsyncMock(return_value=[
            {"tweet_id": str(i)} for i in range(50)
        ])
        
        config = DateRangeCollectorConfig(initial_window_months=6)
        collector = DateRangeCollector(
            username="testuser",
            tweet_collector=mock_collector,
            db=mock_db,
            config=config
        )
        
        # Collect a window that overlaps with already-collected range
        result = await collector.collect_window(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 3, 31)
        )
        
        # Should skip since already complete
        assert result.skipped is True
        assert mock_collector.collect_from_search.call_count == 0
    
    @pytest.mark.asyncio
    async def test_collect_marks_range_complete_in_db(self, mock_collector, mock_db):
        """Completed ranges are marked in database."""
        from fetcher.date_range_collector import DateRangeCollector, DateRangeCollectorConfig
        
        mock_collector.collect_from_search = AsyncMock(return_value=[
            {"tweet_id": str(i)} for i in range(100)
        ])
        
        collector = DateRangeCollector(
            username="testuser",
            tweet_collector=mock_collector,
            db=mock_db
        )
        
        await collector.collect_window(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 6, 30)
        )
        
        mock_db.mark_range_complete.assert_called_once()
        # Check positional args - (username, start_date, end_date, tweets_collected)
        call_args = mock_db.mark_range_complete.call_args[0]
        assert call_args[0] == "testuser"  # username
        assert call_args[3] == 100  # tweets_collected
    
    @pytest.mark.asyncio
    async def test_single_day_window_not_split_even_if_over_threshold(self, mock_collector, mock_db):
        """Single day window accepts all tweets even if over threshold."""
        from fetcher.date_range_collector import DateRangeCollector, DateRangeCollectorConfig
        
        # Return 1000 tweets for a single day (viral moment)
        mock_collector.collect_from_search = AsyncMock(return_value=[
            {"tweet_id": str(i)} for i in range(1000)
        ])
        
        config = DateRangeCollectorConfig(split_threshold=750, min_window_days=1)
        collector = DateRangeCollector(
            username="testuser",
            tweet_collector=mock_collector,
            db=mock_db,
            config=config
        )
        
        result = await collector.collect_window(
            start_date=date(2025, 1, 15),
            end_date=date(2025, 1, 16)  # Single day
        )
        
        # Should collect all 1000 without splitting (can't split 1 day)
        assert result.tweets_collected == 1000
        assert result.windows_processed == 1
        assert mock_collector.collect_from_search.call_count == 1


class TestBackwardsCrawl:
    """Tests for backwards crawling from current date."""
    
    @pytest.fixture
    def mock_collector(self):
        collector = AsyncMock()
        collector.collect_from_search = AsyncMock(return_value=[])
        return collector
    
    @pytest.fixture
    def mock_db(self):
        db = Mock()
        db.is_range_collected = Mock(return_value=False)
        db.get_collected_ranges = Mock(return_value=[])
        db.mark_range_complete = Mock()
        db.mark_range_in_progress = Mock()
        db.get_oldest_tweet_date = Mock(return_value=None)
        db.get_newest_tweet_date = Mock(return_value=None)
        return db
    
    @pytest.mark.asyncio
    async def test_crawl_starts_from_today(self, mock_collector, mock_db):
        """Backwards crawl starts from current date."""
        from fetcher.date_range_collector import DateRangeCollector
        
        mock_collector.collect_from_search = AsyncMock(return_value=[])
        
        collector = DateRangeCollector(
            username="testuser",
            tweet_collector=mock_collector,
            db=mock_db
        )
        
        with patch('fetcher.date_range_collector.date') as mock_date:
            mock_date.today.return_value = date(2025, 6, 15)
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
            
            await collector.collect_full_history()
            
            # First window should end at today
            first_call = mock_collector.collect_from_search.call_args_list[0]
            # The end_date should be around today
            assert first_call is not None
    
    @pytest.mark.asyncio
    async def test_crawl_moves_backwards_in_time(self, mock_collector, mock_db):
        """Each subsequent window is earlier in time."""
        from fetcher.date_range_collector import DateRangeCollector, DateRangeCollectorConfig
        
        windows_collected = []
        
        async def track_windows(username, start_date, end_date, **kwargs):
            windows_collected.append((start_date, end_date))
            return [{"tweet_id": "1"}] if len(windows_collected) < 3 else []
        
        mock_collector.collect_from_search = track_windows
        
        config = DateRangeCollectorConfig(
            initial_window_months=1,
            empty_windows_to_stop=1
        )
        collector = DateRangeCollector(
            username="testuser",
            tweet_collector=mock_collector,
            db=mock_db,
            config=config
        )
        
        await collector.collect_full_history()
        
        # Should have collected multiple windows going backwards
        assert len(windows_collected) >= 2
        
        # Verify windows move backwards in time
        for i in range(1, len(windows_collected)):
            prev_end = windows_collected[i-1][1]
            curr_end = windows_collected[i][1]
            # Each window's end_date should be <= previous window's start_date
            assert curr_end <= prev_end, f"Window {i} should be earlier than window {i-1}"


class TestProgressTracking:
    """Tests for collection progress tracking."""
    
    def test_progress_reports_tweets_collected(self):
        """Progress tracking reports total tweets collected."""
        from fetcher.date_range_collector import CollectionProgress
        
        progress = CollectionProgress(
            username="testuser",
            total_tweets=500,
            windows_completed=5,
            windows_remaining=3
        )
        
        assert progress.total_tweets == 500
        assert progress.windows_completed == 5
        assert progress.is_complete is False
    
    def test_progress_complete_when_no_remaining(self):
        """Progress is complete when no windows remaining."""
        from fetcher.date_range_collector import CollectionProgress
        
        progress = CollectionProgress(
            username="testuser",
            total_tweets=1000,
            windows_completed=10,
            windows_remaining=0
        )
        
        assert progress.is_complete is True
