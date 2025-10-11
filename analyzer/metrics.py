"""
Metrics collection and reporting for the analyzer.
"""

import time
from collections import defaultdict
from typing import Dict, Any, List
from .constants import MetricsKeys


class MetricsCollector:
    """
    Collects and reports metrics for analyzer operations.

    Provides efficient metrics tracking with memory management
    and comprehensive reporting capabilities.
    """

    def __init__(self, max_timings: int = 10000):
        """
        Initialize metrics collector.

        Args:
            max_timings: Maximum number of timing records to keep
        """
        self.max_timings = max_timings
        self._metrics = defaultdict(int)
        self._timings: List[float] = []
        self._category_counts: Dict[str, int] = defaultdict(int)
        self._model_usage: Dict[str, int] = defaultdict(int)
        self._method_counts: Dict[str, int] = defaultdict(int)

        # Set start time
        self._metrics[MetricsKeys.START_TIME] = time.time()

    def record_analysis(self, method: str, duration: float, category: str,
                       model_used: str = "", is_multimodal: bool = False) -> None:
        """
        Record a completed analysis.

        Args:
            method: Analysis method used (pattern, llm, gemini)
            duration: Time taken for analysis in seconds
            category: Content category detected
            model_used: Specific model name used
            is_multimodal: Whether this was a multimodal analysis
        """
        # Update counters
        self._metrics[MetricsKeys.TOTAL_ANALYSES] += 1
        self._method_counts[method] += 1

        if is_multimodal:
            self._metrics[MetricsKeys.MULTIMODAL_COUNT] += 1

        # Update category counts
        self._category_counts[category] += 1

        # Update model usage
        if model_used:
            self._model_usage[model_used] += 1

        # Add timing (with memory management)
        self._timings.append(duration)
        if len(self._timings) > self.max_timings:
            # Keep only the most recent timings
            self._timings = self._timings[-self.max_timings:]

        # Update total time
        self._metrics[MetricsKeys.TOTAL_TIME] += duration

        # Update average time
        total_analyses = self._metrics[MetricsKeys.TOTAL_ANALYSES]
        if total_analyses > 0:
            self._metrics[MetricsKeys.AVG_TIME_PER_ANALYSIS] = (
                self._metrics[MetricsKeys.TOTAL_TIME] / total_analyses
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        return {
            MetricsKeys.TOTAL_ANALYSES: self._metrics[MetricsKeys.TOTAL_ANALYSES],
            MetricsKeys.METHOD_COUNTS: dict(self._method_counts),
            MetricsKeys.MULTIMODAL_COUNT: self._metrics[MetricsKeys.MULTIMODAL_COUNT],
            MetricsKeys.CATEGORY_COUNTS: dict(self._category_counts),
            MetricsKeys.TOTAL_TIME: self._metrics[MetricsKeys.TOTAL_TIME],
            MetricsKeys.AVG_TIME_PER_ANALYSIS: self._metrics[MetricsKeys.AVG_TIME_PER_ANALYSIS],
            MetricsKeys.MODEL_USAGE: dict(self._model_usage),
            MetricsKeys.START_TIME: self._metrics[MetricsKeys.START_TIME],
            'current_time': time.time(),
            'runtime_seconds': time.time() - self._metrics[MetricsKeys.START_TIME]
        }

    def get_timings(self) -> List[float]:
        """Get all recorded timing values."""
        return self._timings.copy()

    def get_category_breakdown(self) -> Dict[str, int]:
        """Get category usage breakdown."""
        return dict(self._category_counts)

    def get_method_breakdown(self) -> Dict[str, int]:
        """Get analysis method usage breakdown."""
        return dict(self._method_counts)

    def get_model_usage(self) -> Dict[str, int]:
        """Get model usage breakdown."""
        return dict(self._model_usage)

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()
        self._timings.clear()
        self._category_counts.clear()
        self._model_usage.clear()
        self._method_counts.clear()
        self._metrics[MetricsKeys.START_TIME] = time.time()

    def generate_report(self) -> str:
        """Generate a comprehensive metrics report."""
        summary = self.get_summary()
        total_time = summary['runtime_seconds']
        total_analyses = summary[MetricsKeys.TOTAL_ANALYSES]

        report_lines = [
            "📊 ANALYSIS METRICS REPORT",
            "=" * 50,
            f"⏱️  Total runtime: {total_time:.2f}s",
            f"📈 Total analyses: {total_analyses}",
        ]

        if total_analyses > 0:
            avg_time = summary[MetricsKeys.AVG_TIME_PER_ANALYSIS]
            total_analysis_time = summary[MetricsKeys.TOTAL_TIME]
            report_lines.extend([
                f"⚡ Average time per analysis: {avg_time:.2f}s",
                f"💰 Total analysis time: {total_analysis_time:.2f}s",
                ""
            ])

            # Method breakdown
            report_lines.append("🔧 Analysis Methods Used:")
            method_counts = summary[MetricsKeys.METHOD_COUNTS]
            for method, count in method_counts.items():
                if count > 0:
                    percentage = (count / total_analyses) * 100
                    report_lines.append(f"  {method.upper()}: {count} ({percentage:.1f}%)")

            # Multimodal breakdown
            multimodal_count = summary[MetricsKeys.MULTIMODAL_COUNT]
            if multimodal_count > 0:
                multimodal_percentage = (multimodal_count / total_analyses) * 100
                report_lines.append(f"  🎥 Multimodal: {multimodal_count} ({multimodal_percentage:.1f}%)")
            report_lines.append("")

            # Model usage
            model_usage = summary[MetricsKeys.MODEL_USAGE]
            if model_usage:
                report_lines.append("🤖 Models Used:")
                for model, count in model_usage.items():
                    percentage = (count / total_analyses) * 100
                    report_lines.append(f"  {model}: {count} ({percentage:.1f}%)")
                report_lines.append("")

            # Category breakdown
            category_counts = summary[MetricsKeys.CATEGORY_COUNTS]
            if category_counts:
                report_lines.append("🏷️  Categories Detected:")
                sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
                for category, count in sorted_categories:
                    percentage = (count / total_analyses) * 100
                    report_lines.append(f"  {category}: {count} ({percentage:.1f}%)")
                report_lines.append("")

            # Performance insights
            report_lines.append("⚡ Performance Insights:")
            if multimodal_count > 0:
                avg_multimodal_time = total_analysis_time / multimodal_count
                report_lines.append(f"  🎥 Average Multimodal analysis time: {avg_multimodal_time:.2f}s")

            llm_count = method_counts.get('llm', 0)
            if llm_count > 0:
                avg_llm_time = total_analysis_time / llm_count
                report_lines.append(f"  🧠 Average LLM analysis time: {avg_llm_time:.2f}s")

            pattern_count = method_counts.get('pattern', 0)
            if pattern_count > 0:
                avg_pattern_time = total_analysis_time / pattern_count
                report_lines.append(f"  🔍 Average Pattern analysis time: {avg_pattern_time:.2f}s")

        return "\n".join(report_lines)