"""
Error handling utilities for the dimetuverdad analyzer.

Provides standardized error classification and handling patterns used across
all analyzer components.
"""

import logging
from enum import Enum
from typing import Optional


class ErrorCategory(Enum):
    """Enumeration of different error categories for better error handling."""
    NETWORK_ERROR = "network_error"
    AUTHENTICATION_ERROR = "authentication_error"
    QUOTA_ERROR = "quota_error"
    MODEL_ERROR = "model_error"
    TIMEOUT_ERROR = "timeout_error"
    MEDIA_ERROR = "media_error"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN_ERROR = "unknown_error"


class AnalysisError(Exception):
    """Custom exception for analysis errors with categorization."""

    def __init__(self, message: str, category: ErrorCategory, recoverable: bool = False, retry_delay: Optional[int] = None):
        super().__init__(message)
        self.category = category
        self.recoverable = recoverable
        self.retry_delay = retry_delay
        self.message = message

    def __str__(self):
        return f"[{self.category.value}] {self.message}"


def classify_error(error: Exception, context: str = "") -> AnalysisError:
    """
    Classify an exception into a specific error category with recovery information.

    Args:
        error: The exception to classify
        context: Additional context about where the error occurred

    Returns:
        AnalysisError with appropriate category and recovery information
    """
    error_str = str(error).lower()

    # Network-related errors
    if any(pattern in error_str for pattern in ['connection', 'timeout', 'network', 'dns', 'ssl']):
        if 'timeout' in error_str:
            return AnalysisError(
                f"Network timeout in {context}: {error}",
                ErrorCategory.TIMEOUT_ERROR,
                recoverable=True,
                retry_delay=5
            )
        else:
            return AnalysisError(
                f"Network error in {context}: {error}",
                ErrorCategory.NETWORK_ERROR,
                recoverable=True,
                retry_delay=2
            )

    # Authentication errors
    elif any(pattern in error_str for pattern in ['unauthorized', 'forbidden', 'authentication', 'api key', 'credentials']):
        return AnalysisError(
            f"Authentication error in {context}: {error}",
            ErrorCategory.AUTHENTICATION_ERROR,
            recoverable=False
        )

    # Quota/rate limit errors
    elif any(pattern in error_str for pattern in ['quota', 'rate limit', 'exceeded', 'limit']):
        return AnalysisError(
            f"Quota exceeded in {context}: {error}",
            ErrorCategory.QUOTA_ERROR,
            recoverable=True,
            retry_delay=60  # Wait 1 minute for quota reset
        )

    # Model availability errors
    elif any(pattern in error_str for pattern in ['model not found', 'model not available', 'unsupported model']):
        return AnalysisError(
            f"Model error in {context}: {error}",
            ErrorCategory.MODEL_ERROR,
            recoverable=True,
            retry_delay=10
        )

    # Media processing errors
    elif any(pattern in error_str for pattern in ['media', 'file', 'upload', 'processing']):
        return AnalysisError(
            f"Media processing error in {context}: {error}",
            ErrorCategory.MEDIA_ERROR,
            recoverable=False
        )

    # Configuration errors
    elif any(pattern in error_str for pattern in ['configuration', 'config', 'environment']):
        return AnalysisError(
            f"Configuration error in {context}: {error}",
            ErrorCategory.CONFIGURATION_ERROR,
            recoverable=False
        )

    # Default to unknown error
    return AnalysisError(
        f"Unknown error in {context}: {error}",
        ErrorCategory.UNKNOWN_ERROR,
        recoverable=False
    )