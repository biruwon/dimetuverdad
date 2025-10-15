"""
Constants and enums for the analyzer module.

This module contains all the magic strings, constants, and enums used throughout
the analyzer to improve maintainability and reduce duplication.
"""

from enum import Enum
from typing import Final


class AnalysisMethods(Enum):
    """Analysis method types."""
    PATTERN = "pattern"
    LLM = "llm"
    GEMINI = "gemini"
    MULTIMODAL = "multimodal"
    ERROR = "error"


class MediaTypes(Enum):
    """Media type classifications."""
    IMAGE = "image"
    VIDEO = "video"
    MIXED = "mixed"
    UNKNOWN = "unknown"
    EMPTY = ""


class ModelNames(Enum):
    """Model name constants."""
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    PATTERN_MATCHING = "pattern-matching"


class LogMessages:
    """Common log message templates."""
    MULTIMODAL_START = "Starting multimodal analysis for tweet {tweet_id} with {media_count} media URLs"
    MULTIMODAL_SUCCESS = "Multimodal analysis successful for tweet {tweet_id}, category: {category}, time: {time:.2f}s"
    MULTIMODAL_FAILURE = "Multimodal analysis returned None for tweet {tweet_id}, falling back to text-only analysis"
    MULTIMODAL_ERROR = "Multimodal analysis failed for tweet {tweet_id}: {error_type}: {error_message}"
    FALLBACK_TO_TEXT = "Fallback: Processing tweet {tweet_id} with text-only pipeline"


class ErrorMessages:
    """Error message templates."""
    LLM_PIPELINE_NOT_AVAILABLE = "ERROR: LLM pipeline not available - explanation generation impossible"
    LLM_EXPLANATION_FAILED = "ERROR: LLM explanation generation failed - {details} (length: {length})"
    LLM_EXPLANATION_EXCEPTION = "ERROR: LLM explanation generation exception - {error_type}: {error_message}"
    LLM_CATEGORY_ERROR = "⚠️ Error en categorización LLM: {error}"
    ANALYSIS_FAILED = "Analysis failed: {error}"
    MULTIMODAL_ANALYSIS_FAILED = "Multimodal analysis failed: {error}"
    MULTIMODAL_EXPLANATION_INSUFFICIENT = "Insufficient multimodal analysis data for explanation"
    MULTIMODAL_EXPLANATION_EXCEPTION = "Multimodal explanation generation failed: {error}"


class ConfigDefaults:
    """Default configuration values."""
    VERBOSE: Final[bool] = False
    MAX_RETRIES: Final[int] = 3
    RETRY_DELAY: Final[int] = 1
    ANALYSIS_TIMEOUT: Final[float] = 300.0
    DATABASE_TIMEOUT: Final[float] = 30.0
    DOWNLOAD_TIMEOUT: Final[float] = 120.0
    REQUEST_TIMEOUT: Final[float] = 30.0
    MAX_CONCURRENCY: Final[int] = 4
    MAX_LLM_CONCURRENCY: Final[int] = 2


class MetricsKeys:
    """Keys for metrics tracking."""
    TOTAL_ANALYSES = "total_analyses"
    METHOD_COUNTS = "method_counts"
    MULTIMODAL_COUNT = "multimodal_count"
    CATEGORY_COUNTS = "category_counts"
    TOTAL_TIME = "total_time"
    AVG_TIME_PER_ANALYSIS = "avg_time_per_analysis"
    MODEL_USAGE = "model_usage"
    START_TIME = "start_time"


class DatabaseConstants:
    """Database-related constants."""
    TABLE_NAME = "content_analyses"
    CONNECTION_TIMEOUT = 30.0
    MAX_RETRIES = 3
    RETRY_DELAY = 1


class GeminiKeywords:
    """Keywords for Gemini analysis category extraction."""
    HATE_SPEECH = ['hate_speech', 'odio', 'racismo', 'discriminación']
    DISINFORMATION = ['disinformation', 'desinformación', 'fake news', 'mentira']
    CONSPIRACY_THEORY = ['conspiracy', 'conspiración', 'teoría conspirativa']
    FAR_RIGHT_BIAS = ['far_right', 'extrema derecha', 'ultraderecha']
    CALL_TO_ACTION = ['call_to_action', 'llamado a la acción', 'llamados a la acción']


class MediaExtensions:
    """File extensions for media type detection."""
    VIDEO_EXTENSIONS = ['.mp4', '.m3u8', '.mov', '.avi', '.webm']
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
    VIDEO_KEYWORDS = ['video', 'vid/', 'amplify_video']
    IMAGE_FORMAT_PARAMS = ['format=jpg', 'format=jpeg', 'format=png']