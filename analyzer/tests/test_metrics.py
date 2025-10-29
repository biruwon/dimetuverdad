"""
Tests for MetricsCollector class.

Comprehensive test suite for metrics collection and reporting functionality.
"""

import time
import pytest
from unittest.mock import patch

from analyzer.metrics import MetricsCollector
from analyzer.constants import MetricsKeys


class TestMetricsCollectorInitialization:
    """Test MetricsCollector initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        collector = MetricsCollector()

        assert collector.max_timings == 10000
        assert isinstance(collector._metrics, dict)
        assert isinstance(collector._timings, list)
        assert isinstance(collector._category_counts, dict)
        assert isinstance(collector._model_usage, dict)
        assert isinstance(collector._method_counts, dict)
        assert MetricsKeys.START_TIME in collector._metrics

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        collector = MetricsCollector(max_timings=5000)

        assert collector.max_timings == 5000


class TestRecordAnalysis:
    """Test analysis recording functionality."""

    def test_record_analysis_basic(self):
        """Test basic analysis recording."""
        collector = MetricsCollector()

        collector.record_analysis("pattern", 1.5, "hate_speech")

        summary = collector.get_summary()
        assert summary[MetricsKeys.TOTAL_ANALYSES] == 1
        assert summary[MetricsKeys.TOTAL_TIME] == 1.5
        assert summary[MetricsKeys.AVG_TIME_PER_ANALYSIS] == 1.5
        assert summary[MetricsKeys.METHOD_COUNTS]["pattern"] == 1
        assert summary[MetricsKeys.CATEGORY_COUNTS]["hate_speech"] == 1

    def test_record_analysis_with_model(self):
        """Test analysis recording with model usage."""
        collector = MetricsCollector()

        collector.record_analysis("llm", 2.0, "disinformation", model_used="gpt-4")

        summary = collector.get_summary()
        assert summary[MetricsKeys.MODEL_USAGE]["gpt-4"] == 1

    def test_record_analysis_multimodal(self):
        """Test multimodal analysis recording."""
        collector = MetricsCollector()

        collector.record_analysis("gemini", 3.0, "hate_speech", is_multimodal=True)

        summary = collector.get_summary()
        assert summary[MetricsKeys.MULTIMODAL_COUNT] == 1

    def test_record_analysis_multiple(self):
        """Test recording multiple analyses."""
        collector = MetricsCollector()

        collector.record_analysis("pattern", 1.0, "hate_speech")
        collector.record_analysis("llm", 2.0, "disinformation", model_used="gpt-4")
        collector.record_analysis("pattern", 1.5, "general")

        summary = collector.get_summary()
        assert summary[MetricsKeys.TOTAL_ANALYSES] == 3
        assert summary[MetricsKeys.TOTAL_TIME] == 4.5
        assert summary[MetricsKeys.AVG_TIME_PER_ANALYSIS] == 1.5
        assert summary[MetricsKeys.METHOD_COUNTS]["pattern"] == 2
        assert summary[MetricsKeys.METHOD_COUNTS]["llm"] == 1
        assert summary[MetricsKeys.CATEGORY_COUNTS]["hate_speech"] == 1
        assert summary[MetricsKeys.CATEGORY_COUNTS]["disinformation"] == 1
        assert summary[MetricsKeys.CATEGORY_COUNTS]["general"] == 1
        assert summary[MetricsKeys.MODEL_USAGE]["gpt-4"] == 1

    def test_record_analysis_timing_memory_management(self):
        """Test timing memory management with max_timings limit."""
        collector = MetricsCollector(max_timings=3)

        # Record more timings than the limit
        for i in range(5):
            collector.record_analysis("pattern", float(i + 1), "test")

        timings = collector.get_timings()
        assert len(timings) == 3  # Should only keep the last 3
        assert timings == [3.0, 4.0, 5.0]  # Most recent timings


class TestGetSummary:
    """Test metrics summary functionality."""

    def test_get_summary_empty(self):
        """Test summary with no analyses."""
        collector = MetricsCollector()

        summary = collector.get_summary()

        assert summary[MetricsKeys.TOTAL_ANALYSES] == 0
        assert summary[MetricsKeys.TOTAL_TIME] == 0
        assert summary[MetricsKeys.AVG_TIME_PER_ANALYSIS] == 0
        assert summary[MetricsKeys.MULTIMODAL_COUNT] == 0
        assert isinstance(summary[MetricsKeys.METHOD_COUNTS], dict)
        assert isinstance(summary[MetricsKeys.CATEGORY_COUNTS], dict)
        assert isinstance(summary[MetricsKeys.MODEL_USAGE], dict)
        assert 'current_time' in summary
        assert 'runtime_seconds' in summary

    def test_get_summary_with_data(self):
        """Test summary with analysis data."""
        collector = MetricsCollector()

        collector.record_analysis("pattern", 1.5, "hate_speech")
        collector.record_analysis("llm", 2.0, "disinformation", model_used="gpt-4", is_multimodal=True)

        summary = collector.get_summary()

        assert summary[MetricsKeys.TOTAL_ANALYSES] == 2
        assert summary[MetricsKeys.TOTAL_TIME] == 3.5
        assert summary[MetricsKeys.AVG_TIME_PER_ANALYSIS] == 1.75
        assert summary[MetricsKeys.MULTIMODAL_COUNT] == 1
        assert summary[MetricsKeys.METHOD_COUNTS]["pattern"] == 1
        assert summary[MetricsKeys.METHOD_COUNTS]["llm"] == 1
        assert summary[MetricsKeys.CATEGORY_COUNTS]["hate_speech"] == 1
        assert summary[MetricsKeys.CATEGORY_COUNTS]["disinformation"] == 1
        assert summary[MetricsKeys.MODEL_USAGE]["gpt-4"] == 1


class TestGetterMethods:
    """Test various getter methods."""

    def test_get_timings(self):
        """Test getting timing data."""
        collector = MetricsCollector()

        collector.record_analysis("pattern", 1.5, "test")
        collector.record_analysis("llm", 2.0, "test")

        timings = collector.get_timings()
        assert timings == [1.5, 2.0]

    def test_get_category_breakdown(self):
        """Test category breakdown."""
        collector = MetricsCollector()

        collector.record_analysis("pattern", 1.0, "hate_speech")
        collector.record_analysis("pattern", 1.0, "hate_speech")
        collector.record_analysis("pattern", 1.0, "disinformation")

        breakdown = collector.get_category_breakdown()
        assert breakdown["hate_speech"] == 2
        assert breakdown["disinformation"] == 1

    def test_get_method_breakdown(self):
        """Test method breakdown."""
        collector = MetricsCollector()

        collector.record_analysis("pattern", 1.0, "test")
        collector.record_analysis("llm", 1.0, "test")
        collector.record_analysis("llm", 1.0, "test")

        breakdown = collector.get_method_breakdown()
        assert breakdown["pattern"] == 1
        assert breakdown["llm"] == 2

    def test_get_model_usage(self):
        """Test model usage breakdown."""
        collector = MetricsCollector()

        collector.record_analysis("llm", 1.0, "test", model_used="gpt-4")
        collector.record_analysis("llm", 1.0, "test", model_used="gpt-4")
        collector.record_analysis("llm", 1.0, "test", model_used="claude")

        usage = collector.get_model_usage()
        assert usage["gpt-4"] == 2
        assert usage["claude"] == 1


class TestReset:
    """Test metrics reset functionality."""

    def test_reset(self):
        """Test resetting all metrics."""
        collector = MetricsCollector()

        # Add some data
        collector.record_analysis("pattern", 1.5, "hate_speech", model_used="test-model", is_multimodal=True)

        # Verify data exists
        summary = collector.get_summary()
        assert summary[MetricsKeys.TOTAL_ANALYSES] == 1

        # Reset
        collector.reset()

        # Verify data is cleared
        summary = collector.get_summary()
        assert summary[MetricsKeys.TOTAL_ANALYSES] == 0
        assert summary[MetricsKeys.TOTAL_TIME] == 0
        assert summary[MetricsKeys.MULTIMODAL_COUNT] == 0
        assert len(summary[MetricsKeys.METHOD_COUNTS]) == 0
        assert len(summary[MetricsKeys.CATEGORY_COUNTS]) == 0
        assert len(summary[MetricsKeys.MODEL_USAGE]) == 0
        assert len(collector.get_timings()) == 0

        # Start time should be reset
        assert summary[MetricsKeys.START_TIME] > summary['current_time'] - 1  # Within last second


class TestGenerateReport:
    """Test report generation functionality."""

    def test_generate_report_empty(self):
        """Test report generation with no data."""
        collector = MetricsCollector()

        report = collector.generate_report()

        assert "📊 ANALYSIS METRICS REPORT" in report
        assert "Total analyses: 0" in report

    def test_generate_report_with_data(self):
        """Test report generation with analysis data."""
        collector = MetricsCollector()

        collector.record_analysis("pattern", 1.0, "hate_speech")
        collector.record_analysis("llm", 2.0, "disinformation", model_used="gpt-4", is_multimodal=True)
        collector.record_analysis("pattern", 1.5, "general")

        report = collector.generate_report()

        # Check basic structure
        assert "📊 ANALYSIS METRICS REPORT" in report
        assert "Total analyses: 3" in report
        assert "Average time per analysis:" in report

        # Check method breakdown
        assert "PATTERN:" in report
        assert "LLM:" in report

        # Check multimodal
        assert "Multimodal:" in report

        # Check model usage
        assert "gpt-4:" in report

        # Check categories
        assert "hate_speech:" in report
        assert "disinformation:" in report
        assert "general:" in report

    def test_generate_report_performance_insights(self):
        """Test performance insights in report."""
        collector = MetricsCollector()

        # Add different types of analyses
        collector.record_analysis("pattern", 0.5, "test")
        collector.record_analysis("llm", 2.0, "test")
        collector.record_analysis("gemini", 3.0, "test", is_multimodal=True)

        report = collector.generate_report()

        # Check performance insights
        assert "Average Pattern analysis time:" in report
        assert "Average LLM analysis time:" in report
        assert "Average Multimodal analysis time:" in report

    @patch('time.time')
    def test_generate_report_runtime_calculation(self, mock_time):
        """Test runtime calculation in report."""
        mock_time.return_value = 1000.0

        collector = MetricsCollector()
        # Manually set start time to simulate elapsed time
        collector._metrics[MetricsKeys.START_TIME] = 900.0

        report = collector.generate_report()

        assert "Total runtime: 100.00s" in report