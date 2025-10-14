"""
Tests for performance metrics utility module.
Tests the PerformanceTracker, PerformanceMetrics, and convenience functions.
"""

import time
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from utils.performance import (
    PerformanceMetrics,
    PerformanceTracker,
    track_performance,
    start_tracking,
    stop_tracking,
    print_performance_summary
)


class TestPerformanceMetrics:
    """Test the PerformanceMetrics dataclass."""

    def test_performance_metrics_creation(self):
        """Test creating a PerformanceMetrics instance."""
        start_time = datetime.now()
        metrics = PerformanceMetrics(
            script_name="test_script",
            start_time=start_time,
            duration_seconds=10.5,
            peak_memory_mb=150.0,
            cpu_percent=25.0,
            operations_count=100,
            operations_per_second=9.52,
            success=True
        )

        assert metrics.script_name == "test_script"
        assert metrics.start_time == start_time
        assert metrics.duration_seconds == 10.5
        assert metrics.peak_memory_mb == 150.0
        assert metrics.cpu_percent == 25.0
        assert metrics.operations_count == 100
        assert metrics.operations_per_second == 9.52
        assert metrics.success is True
        assert metrics.error_message is None

    def test_performance_metrics_to_dict(self):
        """Test converting PerformanceMetrics to dictionary."""
        start_time = datetime(2025, 10, 14, 12, 0, 0)
        end_time = datetime(2025, 10, 14, 12, 0, 10)

        metrics = PerformanceMetrics(
            script_name="test_script",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=10.0,
            peak_memory_mb=100.0,
            cpu_percent=20.0,
            operations_count=50,
            operations_per_second=5.0,
            success=True,
            error_message=None
        )

        data = metrics.to_dict()

        assert data['script_name'] == "test_script"
        assert data['start_time'] == "2025-10-14T12:00:00"
        assert data['end_time'] == "2025-10-14T12:00:10"
        assert data['duration_seconds'] == 10.0
        assert data['peak_memory_mb'] == 100.0
        assert data['cpu_percent'] == 20.0
        assert data['operations_count'] == 50
        assert data['operations_per_second'] == 5.0
        assert data['success'] is True
        assert data['error_message'] is None


class TestPerformanceTracker:
    """Test the PerformanceTracker class."""

    @patch('utils.performance.psutil.Process')
    def test_performance_tracker_initialization(self, mock_process_class):
        """Test PerformanceTracker initialization."""
        mock_process = MagicMock()
        mock_process_class.return_value = mock_process

        tracker = PerformanceTracker("test_script")

        assert tracker.script_name == "test_script"
        assert isinstance(tracker.start_time, datetime)
        assert tracker.peak_memory == 0
        assert tracker.cpu_percent == 0
        assert tracker.operations_count == 0
        assert tracker._monitoring is False
        mock_process_class.assert_called_once()

    @patch('utils.performance.psutil.Process')
    def test_increment_operations(self, mock_process_class):
        """Test incrementing operations counter."""
        mock_process_class.return_value = MagicMock()

        tracker = PerformanceTracker("test_script")

        assert tracker.operations_count == 0

        tracker.increment_operations(5)
        assert tracker.operations_count == 5

        tracker.increment_operations()  # Default increment by 1
        assert tracker.operations_count == 6

    @patch('utils.performance.psutil.Process')
    @patch('utils.performance.datetime')
    def test_stop_monitoring(self, mock_datetime, mock_process_class):
        """Test stopping monitoring and calculating metrics."""
        # Setup mocks
        mock_process_class.return_value = MagicMock()

        start_time = datetime(2025, 10, 14, 12, 0, 0)
        end_time = datetime(2025, 10, 14, 12, 0, 15)  # 15 seconds later

        mock_datetime.now.side_effect = [start_time, end_time]

        tracker = PerformanceTracker("test_script")
        tracker.peak_memory = 200.0
        tracker.cpu_percent = 30.0
        tracker.operations_count = 10

        metrics = tracker.stop_monitoring()

        assert metrics.script_name == "test_script"
        assert metrics.start_time == start_time
        assert metrics.end_time == end_time
        assert metrics.duration_seconds == 15.0
        assert metrics.peak_memory_mb == 200.0
        assert metrics.cpu_percent == 30.0
        assert metrics.operations_count == 10
        assert metrics.operations_per_second == 10 / 15.0  # 0.666...
        assert metrics.success is True

    @patch('utils.performance.psutil.Process')
    def test_monitor_resources(self, mock_process_class):
        """Test resource monitoring."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024  # 100 MB in bytes
        mock_process.cpu_percent.return_value = 25.0
        mock_process_class.return_value = mock_process

        tracker = PerformanceTracker("test_script")
        tracker._monitoring = True

        tracker._monitor_resources()

        assert tracker.peak_memory == 100.0  # 100 MB
        assert tracker.cpu_percent == 25.0

        # Test peak memory tracking
        mock_process.memory_info.return_value.rss = 150 * 1024 * 1024  # 150 MB
        tracker._monitor_resources()
        assert tracker.peak_memory == 150.0  # Should update to higher value

    @patch('utils.performance.psutil.Process')
    def test_monitor_resources_disabled(self, mock_process_class):
        """Test that monitoring doesn't happen when disabled."""
        mock_process_class.return_value = MagicMock()

        tracker = PerformanceTracker("test_script")
        tracker._monitoring = False

        tracker._monitor_resources()

        # Values should remain at defaults
        assert tracker.peak_memory == 0
        assert tracker.cpu_percent == 0

    @patch('utils.performance.psutil.Process')
    def test_monitor_resources_exception_handling(self, mock_process_class):
        """Test that exceptions during monitoring are handled gracefully."""
        mock_process = MagicMock()
        mock_process.memory_info.side_effect = Exception("Test error")
        mock_process_class.return_value = mock_process

        tracker = PerformanceTracker("test_script")
        tracker._monitoring = True

        # Should not raise exception
        tracker._monitor_resources()

        # Values should remain at defaults
        assert tracker.peak_memory == 0
        assert tracker.cpu_percent == 0


class TestConvenienceFunctions:
    """Test convenience functions."""

    @patch('utils.performance.PerformanceTracker')
    def test_start_tracking(self, mock_tracker_class):
        """Test start_tracking function."""
        mock_tracker = MagicMock()
        mock_tracker_class.return_value = mock_tracker

        result = start_tracking("test_script")

        mock_tracker_class.assert_called_once_with("test_script")
        mock_tracker.start_monitoring.assert_called_once()
        assert result == mock_tracker

    @patch('utils.performance.PerformanceTracker')
    def test_stop_tracking(self, mock_tracker_class):
        """Test stop_tracking function."""
        mock_tracker = MagicMock()
        mock_metrics = MagicMock()
        mock_tracker.stop_monitoring.return_value = mock_metrics

        result = stop_tracking(mock_tracker)

        mock_tracker.stop_monitoring.assert_called_once()
        assert result == mock_metrics

    @patch('utils.performance.PerformanceTracker')
    def test_print_performance_summary(self, mock_tracker_class):
        """Test print_performance_summary function."""
        mock_tracker = MagicMock()
        mock_tracker_class.return_value = mock_tracker
        mock_metrics = MagicMock()

        print_performance_summary(mock_metrics)

        mock_tracker_class.assert_called_once_with(mock_metrics.script_name)
        mock_tracker.print_summary.assert_called_once_with(mock_metrics)


class TestTrackPerformanceDecorator:
    """Test the track_performance decorator."""

    @patch('utils.performance.PerformanceTracker')
    def test_decorator_success(self, mock_tracker_class):
        """Test decorator with successful function execution."""
        mock_tracker = MagicMock()
        mock_metrics = MagicMock()
        mock_tracker.stop_monitoring.return_value = mock_metrics
        mock_tracker_class.return_value = mock_tracker

        @track_performance("test_function")
        def test_func():
            return "success"

        result = test_func()

        assert result == "success"
        mock_tracker_class.assert_called_once_with("test_function")
        mock_tracker.start_monitoring.assert_called_once()
        mock_tracker.stop_monitoring.assert_called_once()
        mock_tracker.print_summary.assert_called_once_with(mock_metrics)

    @patch('utils.performance.PerformanceTracker')
    def test_decorator_exception(self, mock_tracker_class):
        """Test decorator with function that raises exception."""
        mock_tracker = MagicMock()
        mock_metrics = MagicMock()
        mock_tracker.stop_monitoring.return_value = mock_metrics
        mock_tracker_class.return_value = mock_tracker

        @track_performance("test_function")
        def test_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            test_func()

        mock_tracker_class.assert_called_once_with("test_function")
        mock_tracker.start_monitoring.assert_called_once()
        mock_tracker.stop_monitoring.assert_called_once()
        assert mock_metrics.success is False
        assert mock_metrics.error_message == "Test error"
        mock_tracker.print_summary.assert_called_once_with(mock_metrics)


class TestPrintSummary:
    """Test the print_summary method."""

    @patch('builtins.print')
    def test_print_summary_success(self, mock_print):
        """Test printing summary for successful execution."""
        start_time = datetime(2025, 10, 14, 12, 0, 0)
        end_time = datetime(2025, 10, 14, 12, 0, 30)

        metrics = PerformanceMetrics(
            script_name="Test Script",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=30.0,
            peak_memory_mb=256.5,
            cpu_percent=15.2,
            operations_count=150,
            operations_per_second=5.0,
            success=True
        )

        tracker = PerformanceTracker("Test Script")
        tracker.print_summary(metrics)

        # Check that print was called with expected content
        calls = mock_print.call_args_list
        assert len(calls) >= 6  # Should have multiple print calls

        # Check key content
        print_calls = [call[0][0] for call in calls]
        assert any("📊 Test Script Performance Summary" in call for call in print_calls)
        assert any("⏱️  Duration: 30.00 seconds" in call for call in print_calls)
        assert any("🧠 Peak Memory: 256.5 MB" in call for call in print_calls)
        assert any("⚡ CPU Usage: 15.2%" in call for call in print_calls)
        assert any("🔢 Operations: 150" in call for call in print_calls)
        assert any("🚀 Throughput: 5.00 ops/sec" in call for call in print_calls)
        assert any("📈 Status: ✅ Success" in call for call in print_calls)

    @patch('builtins.print')
    def test_print_summary_failure(self, mock_print):
        """Test printing summary for failed execution."""
        start_time = datetime(2025, 10, 14, 12, 0, 0)
        end_time = datetime(2025, 10, 14, 12, 0, 10)

        metrics = PerformanceMetrics(
            script_name="Failed Script",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=10.0,
            peak_memory_mb=128.0,
            cpu_percent=5.0,
            operations_count=0,
            operations_per_second=0.0,
            success=False,
            error_message="Test failure"
        )

        tracker = PerformanceTracker("Failed Script")
        tracker.print_summary(metrics)

        # Check that print was called with expected content
        calls = mock_print.call_args_list
        print_calls = [call[0][0] for call in calls]

        assert any("📊 Failed Script Performance Summary" in call for call in print_calls)
        assert any("📈 Status: ❌ Failed" in call for call in print_calls)
        assert any("💥 Error: Test failure" in call for call in print_calls)

    @patch('builtins.print')
    def test_print_summary_no_operations(self, mock_print):
        """Test printing summary when no operations were counted."""
        start_time = datetime(2025, 10, 14, 12, 0, 0)
        end_time = datetime(2025, 10, 14, 12, 0, 5)

        metrics = PerformanceMetrics(
            script_name="No Ops Script",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=5.0,
            peak_memory_mb=64.0,
            cpu_percent=2.0,
            operations_count=0,
            operations_per_second=0.0,
            success=True
        )

        tracker = PerformanceTracker("No Ops Script")
        tracker.print_summary(metrics)

        # Check that print was called with expected content
        calls = mock_print.call_args_list
        printed_lines = [call[0][0] for call in calls]

        # Should not include operations or throughput lines
        has_operations = any("🔢 Operations:" in line for line in printed_lines)
        has_throughput = any("🚀 Throughput:" in line for line in printed_lines)
        assert not has_operations
        assert not has_throughput