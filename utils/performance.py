"""
Performance metrics utility for tracking execution time and system resources.
Used by main scripts to provide performance insights on every run.
"""

import time
import psutil
import os
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class PerformanceMetrics:
    """Container for performance metrics data."""
    script_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    operations_count: Optional[int] = None
    operations_per_second: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        if isinstance(data['start_time'], datetime):
            data['start_time'] = data['start_time'].isoformat()
        if isinstance(data['end_time'], datetime):
            data['end_time'] = data['end_time'].isoformat()
        return data


class PerformanceTracker:
    """Tracks performance metrics for script execution."""

    def __init__(self, script_name: str):
        self.script_name = script_name
        self.start_time = datetime.now()
        self.process = psutil.Process(os.getpid())
        self.peak_memory = 0
        self.cpu_percent = 0
        self.operations_count = 0
        self._monitoring = False

    def start_monitoring(self):
        """Start monitoring system resources."""
        self._monitoring = True
        self._monitor_resources()

    def stop_monitoring(self):
        """Stop monitoring and calculate final metrics."""
        self._monitoring = False
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        return PerformanceMetrics(
            script_name=self.script_name,
            start_time=self.start_time,
            end_time=end_time,
            duration_seconds=duration,
            peak_memory_mb=self.peak_memory,
            cpu_percent=self.cpu_percent,
            operations_count=self.operations_count,
            operations_per_second=self.operations_count / duration if duration > 0 else 0
        )

    def increment_operations(self, count: int = 1):
        """Increment the operations counter."""
        self.operations_count += count

    def _monitor_resources(self):
        """Monitor system resources in the background."""
        if not self._monitoring:
            return

        try:
            # Get memory usage
            memory_mb = self.process.memory_info().rss / (1024 * 1024)
            self.peak_memory = max(self.peak_memory, memory_mb)

            # Get CPU usage (over a short interval)
            cpu = self.process.cpu_percent(interval=0.1)
            if cpu > 0:  # Only update if we got a valid reading
                self.cpu_percent = cpu

        except Exception:
            # Ignore monitoring errors
            pass

    def print_summary(self, metrics: PerformanceMetrics):
        """Print a performance summary."""
        print(f"\n📊 {self.script_name} Performance Summary")
        print("=" * 50)
        print(f"⏱️  Duration: {metrics.duration_seconds:.2f} seconds")
        print(f"🧠 Peak Memory: {metrics.peak_memory_mb:.1f} MB")
        print(f"⚡ CPU Usage: {metrics.cpu_percent:.1f}%")

        if metrics.operations_count > 0:
            print(f"🔢 Operations: {metrics.operations_count}")
            print(f"🚀 Throughput: {metrics.operations_per_second:.2f} ops/sec")

        status = "✅ Success" if metrics.success else "❌ Failed"
        print(f"📈 Status: {status}")

        if metrics.error_message:
            print(f"💥 Error: {metrics.error_message}")


def track_performance(script_name: str):
    """
    Decorator to track performance of a function.

    Usage:
        @track_performance("my_script")
        def my_function():
            # do work
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracker = PerformanceTracker(script_name)
            tracker.start_monitoring()

            try:
                result = func(*args, **kwargs)
                metrics = tracker.stop_monitoring()
                tracker.print_summary(metrics)
                return result
            except Exception as e:
                metrics = tracker.stop_monitoring()
                metrics.success = False
                metrics.error_message = str(e)
                tracker.print_summary(metrics)
                raise

        return wrapper
    return decorator


# Convenience functions for manual tracking
def start_tracking(script_name: str) -> PerformanceTracker:
    """Start manual performance tracking."""
    tracker = PerformanceTracker(script_name)
    tracker.start_monitoring()
    return tracker


def stop_tracking(tracker: PerformanceTracker) -> PerformanceMetrics:
    """Stop manual performance tracking and return metrics."""
    return tracker.stop_monitoring()


def print_performance_summary(metrics: PerformanceMetrics):
    """Print performance summary for given metrics."""
    tracker = PerformanceTracker(metrics.script_name)
    tracker.print_summary(metrics)