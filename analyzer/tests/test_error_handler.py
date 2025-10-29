"""
Tests for error_handler.py module.

Comprehensive test suite covering error classification and handling functionality.
"""

import pytest
from analyzer.error_handler import ErrorCategory, AnalysisError, classify_error


class TestErrorCategory:
    """Test ErrorCategory enum values."""

    def test_error_category_values(self):
        """Test that all expected error categories are defined."""
        expected_categories = [
            'NETWORK_ERROR',
            'AUTHENTICATION_ERROR',
            'QUOTA_ERROR',
            'MODEL_ERROR',
            'TIMEOUT_ERROR',
            'MEDIA_ERROR',
            'CONFIGURATION_ERROR',
            'UNKNOWN_ERROR'
        ]

        actual_categories = [category.name for category in ErrorCategory]

        assert set(actual_categories) == set(expected_categories)

    def test_error_category_string_values(self):
        """Test that error categories have correct string values."""
        assert ErrorCategory.NETWORK_ERROR.value == "network_error"
        assert ErrorCategory.AUTHENTICATION_ERROR.value == "authentication_error"
        assert ErrorCategory.QUOTA_ERROR.value == "quota_error"
        assert ErrorCategory.MODEL_ERROR.value == "model_error"
        assert ErrorCategory.TIMEOUT_ERROR.value == "timeout_error"
        assert ErrorCategory.MEDIA_ERROR.value == "media_error"
        assert ErrorCategory.CONFIGURATION_ERROR.value == "configuration_error"
        assert ErrorCategory.UNKNOWN_ERROR.value == "unknown_error"


class TestAnalysisError:
    """Test AnalysisError exception class."""

    def test_analysis_error_basic(self):
        """Test basic AnalysisError creation."""
        error = AnalysisError("Test message", ErrorCategory.NETWORK_ERROR)

        assert str(error) == "[network_error] Test message"
        assert error.category == ErrorCategory.NETWORK_ERROR
        assert error.recoverable is False
        assert error.retry_delay is None
        assert error.message == "Test message"

    def test_analysis_error_with_recovery(self):
        """Test AnalysisError with recovery information."""
        error = AnalysisError(
            "Timeout occurred",
            ErrorCategory.TIMEOUT_ERROR,
            recoverable=True,
            retry_delay=5
        )

        assert str(error) == "[timeout_error] Timeout occurred"
        assert error.category == ErrorCategory.TIMEOUT_ERROR
        assert error.recoverable is True
        assert error.retry_delay == 5

    def test_analysis_error_inheritance(self):
        """Test that AnalysisError inherits from Exception."""
        error = AnalysisError("Test", ErrorCategory.UNKNOWN_ERROR)

        assert isinstance(error, Exception)


class TestClassifyError:
    """Test classify_error function with various error types."""

    def test_classify_network_timeout_error(self):
        """Test classification of timeout network errors."""
        error = Exception("Connection timeout")
        result = classify_error(error, "test_context")

        assert isinstance(result, AnalysisError)
        assert result.category == ErrorCategory.TIMEOUT_ERROR
        assert result.recoverable is True
        assert result.retry_delay == 5
        assert "Connection timeout" in result.message
        assert "test_context" in result.message

    def test_classify_network_connection_error(self):
        """Test classification of general network connection errors."""
        error = Exception("Failed to establish connection")
        result = classify_error(error, "api_call")

        assert isinstance(result, AnalysisError)
        assert result.category == ErrorCategory.NETWORK_ERROR
        assert result.recoverable is True
        assert result.retry_delay == 2
        assert "Failed to establish connection" in result.message
        assert "api_call" in result.message

    def test_classify_network_dns_error(self):
        """Test classification of DNS-related network errors."""
        error = Exception("DNS resolution failed")
        result = classify_error(error)

        assert isinstance(result, AnalysisError)
        assert result.category == ErrorCategory.NETWORK_ERROR
        assert result.recoverable is True
        assert result.retry_delay == 2

    def test_classify_network_ssl_error(self):
        """Test classification of SSL-related network errors."""
        error = Exception("SSL certificate verification failed")
        result = classify_error(error)

        assert isinstance(result, AnalysisError)
        assert result.category == ErrorCategory.NETWORK_ERROR
        assert result.recoverable is True
        assert result.retry_delay == 2

    def test_classify_authentication_error(self):
        """Test classification of authentication errors."""
        error = Exception("Unauthorized access - invalid API key")
        result = classify_error(error, "authentication")

        assert isinstance(result, AnalysisError)
        assert result.category == ErrorCategory.AUTHENTICATION_ERROR
        assert result.recoverable is False
        assert result.retry_delay is None
        assert "authentication" in result.message

    def test_classify_forbidden_error(self):
        """Test classification of forbidden access errors."""
        error = Exception("403 Forbidden")
        result = classify_error(error)

        assert isinstance(result, AnalysisError)
        assert result.category == ErrorCategory.AUTHENTICATION_ERROR
        assert result.recoverable is False

    def test_classify_credentials_error(self):
        """Test classification of credential-related errors."""
        error = Exception("Invalid credentials provided")
        result = classify_error(error)

        assert isinstance(result, AnalysisError)
        assert result.category == ErrorCategory.AUTHENTICATION_ERROR
        assert result.recoverable is False

    def test_classify_quota_error(self):
        """Test classification of quota/rate limit errors."""
        error = Exception("Rate limit exceeded")
        result = classify_error(error, "api_request")

        assert isinstance(result, AnalysisError)
        assert result.category == ErrorCategory.QUOTA_ERROR
        assert result.recoverable is True
        assert result.retry_delay == 60
        assert "api_request" in result.message

    def test_classify_quota_exceeded_error(self):
        """Test classification of quota exceeded errors."""
        error = Exception("Daily quota exceeded for this API")
        result = classify_error(error)

        assert isinstance(result, AnalysisError)
        assert result.category == ErrorCategory.QUOTA_ERROR
        assert result.recoverable is True
        assert result.retry_delay == 60

    def test_classify_limit_error(self):
        """Test classification of general limit errors."""
        error = Exception("Request limit reached")
        result = classify_error(error)

        assert isinstance(result, AnalysisError)
        assert result.category == ErrorCategory.QUOTA_ERROR
        assert result.recoverable is True
        assert result.retry_delay == 60

    def test_classify_model_not_found_error(self):
        """Test classification of model not found errors."""
        error = Exception("The model not found in registry")
        result = classify_error(error, "model_selection")

        assert isinstance(result, AnalysisError)
        assert result.category == ErrorCategory.MODEL_ERROR
        assert result.recoverable is True
        assert result.retry_delay == 10
        assert "model_selection" in result.message

    def test_classify_model_not_available_error(self):
        """Test classification of model not available errors."""
        error = Exception("Model not available currently")
        result = classify_error(error)

        assert isinstance(result, AnalysisError)
        assert result.category == ErrorCategory.MODEL_ERROR
        assert result.recoverable is True
        assert result.retry_delay == 10

    def test_classify_unsupported_model_error(self):
        """Test classification of unsupported model errors."""
        error = Exception("This model is unsupported model type")
        result = classify_error(error)

        assert isinstance(result, AnalysisError)
        assert result.category == ErrorCategory.MODEL_ERROR
        assert result.recoverable is True
        assert result.retry_delay == 10

    def test_classify_media_processing_error(self):
        """Test classification of media processing errors."""
        error = Exception("Failed to process uploaded media file")
        result = classify_error(error, "media_upload")

        assert isinstance(result, AnalysisError)
        assert result.category == ErrorCategory.MEDIA_ERROR
        assert result.recoverable is False
        assert result.retry_delay is None
        assert "media_upload" in result.message

    def test_classify_media_file_error(self):
        """Test classification of media file errors."""
        error = Exception("Invalid media file format")
        result = classify_error(error)

        assert isinstance(result, AnalysisError)
        assert result.category == ErrorCategory.MEDIA_ERROR
        assert result.recoverable is False

    def test_classify_media_upload_error(self):
        """Test classification of media upload errors."""
        error = Exception("Media upload failed")
        result = classify_error(error)

        assert isinstance(result, AnalysisError)
        assert result.category == ErrorCategory.MEDIA_ERROR
        assert result.recoverable is False

    def test_classify_configuration_error(self):
        """Test classification of configuration errors."""
        error = Exception("Configuration settings invalid")
        result = classify_error(error, "config_load")

        assert isinstance(result, AnalysisError)
        assert result.category == ErrorCategory.CONFIGURATION_ERROR
        assert result.recoverable is False
        assert result.retry_delay is None
        assert "config_load" in result.message

    def test_classify_config_error(self):
        """Test classification of config-related errors."""
        error = Exception("Invalid config parameter")
        result = classify_error(error)

        assert isinstance(result, AnalysisError)
        assert result.category == ErrorCategory.CONFIGURATION_ERROR
        assert result.recoverable is False

    def test_classify_environment_error(self):
        """Test classification of environment-related errors."""
        error = Exception("Required environment variable not set")
        result = classify_error(error)

        assert isinstance(result, AnalysisError)
        assert result.category == ErrorCategory.CONFIGURATION_ERROR
        assert result.recoverable is False

    def test_classify_unknown_error(self):
        """Test classification of unknown/unmatched errors."""
        error = Exception("Some unexpected error occurred")
        result = classify_error(error, "unknown_operation")

        assert isinstance(result, AnalysisError)
        assert result.category == ErrorCategory.UNKNOWN_ERROR
        assert result.recoverable is False
        assert result.retry_delay is None
        assert "unknown_operation" in result.message

    def test_classify_error_empty_context(self):
        """Test classify_error with empty context."""
        error = Exception("Test error")
        result = classify_error(error, "")

        assert isinstance(result, AnalysisError)
        assert "Test error" in result.message

    def test_classify_error_case_insensitive_matching(self):
        """Test that error classification is case insensitive."""
        error = Exception("CONNECTION TIMEOUT")
        result = classify_error(error)

        assert isinstance(result, AnalysisError)
        assert result.category == ErrorCategory.TIMEOUT_ERROR

    def test_classify_error_partial_matches(self):
        """Test classification with partial string matches."""
        # Test partial matches for different categories
        test_cases = [
            ("network timeout", ErrorCategory.TIMEOUT_ERROR),
            ("unauthorized access", ErrorCategory.AUTHENTICATION_ERROR),
            ("quota limit exceeded", ErrorCategory.QUOTA_ERROR),
            ("model not found", ErrorCategory.MODEL_ERROR),
            ("media processing failed", ErrorCategory.MEDIA_ERROR),
            ("configuration invalid", ErrorCategory.CONFIGURATION_ERROR),
        ]

        for error_text, expected_category in test_cases:
            error = Exception(error_text)
            result = classify_error(error)
            assert result.category == expected_category, f"Failed for error: {error_text}"

    def test_classify_error_complex_messages(self):
        """Test classification with complex error messages containing multiple keywords."""
        error = Exception("Connection timeout occurred during API call with invalid credentials")
        result = classify_error(error)

        # Should match timeout first (appears earlier in the classification logic)
        assert result.category == ErrorCategory.TIMEOUT_ERROR
        assert result.recoverable is True
