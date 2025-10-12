"""
Integration components for connecting with the main analyzer.
"""

from .analyzer_hooks import AnalyzerHooks, AnalysisResult, create_analyzer_hooks

__all__ = [
    "AnalyzerHooks",
    "AnalysisResult",
    "create_analyzer_hooks",
]