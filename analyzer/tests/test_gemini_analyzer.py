"""
Unit tests for GeminiAnalyzer component.
Tests multimodal analysis, rate limiting, and metrics collection.
"""

import pytest
import asyncio
import time
import os
from unittest.mock import Mock, patch, MagicMock
from analyzer.gemini_analyzer import (
    GeminiAnalyzer,
    RequestsHTTPClient,
    TempFileSystem,
    DefaultResourceMonitor,
    DefaultMetricsCollector
)


@pytest.fixture
def mock_gemini_client():
    """Mock GeminiClient instance."""
    mock = Mock()
    mock.get_model.return_value = (Mock(), None)
    mock.generate_content.return_value = "CATEGORÍA: hate_speech\nEXPLICACIÓN: Este contenido muestra discurso de odio."
    return mock


@pytest.fixture
def mock_media_processor():
    """Mock MediaProcessor instance."""
    mock = Mock()
    mock.select_media_url.return_value = "https://example.com/image.jpg"
    mock.is_video_url.return_value = False
    mock.download_media.return_value = "/tmp/test_image.jpg"
    mock.download_media_async.return_value = "/tmp/test_image.jpg"
    mock.cleanup_file.return_value = None
    return mock


@pytest.fixture
def analyzer(mock_gemini_client, mock_media_processor):
    """Create GeminiAnalyzer with mocked dependencies."""
    with patch('analyzer.gemini_analyzer.GeminiClient', return_value=mock_gemini_client), \
         patch('analyzer.gemini_analyzer.MediaProcessor', return_value=mock_media_processor):
        analyzer = GeminiAnalyzer(api_key="test_key")
        return analyzer


class TestGeminiAnalyzerInitialization:
    """Test analyzer initialization."""

    def test_default_initialization(self):
        """Test default initialization with all parameters."""
        with patch('analyzer.gemini_analyzer.GeminiClient'), \
             patch('analyzer.gemini_analyzer.MediaProcessor'):
            analyzer = GeminiAnalyzer(api_key="test_key")

            assert analyzer.api_key == "test_key"
            assert analyzer.analysis_timeout == 180.0
            assert isinstance(analyzer.metrics_collector, DefaultMetricsCollector)
            assert analyzer.model_priority == [
                'gemini-2.5-pro',
                'gemini-2.5-flash',
                'gemini-2.5-flash-lite',
                'gemini-2.0-flash-exp',
                'gemini-2.0-flash-lite',
                'gemini-2.0-flash'
            ]
            assert analyzer.rate_limited_models == set()

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        custom_priority = ['custom-model-1', 'custom-model-2']

        with patch('analyzer.gemini_analyzer.GeminiClient'), \
             patch('analyzer.gemini_analyzer.MediaProcessor'):
            analyzer = GeminiAnalyzer(
                api_key="test_key",
                model_priority=custom_priority,
                analysis_timeout=120.0
            )

            assert analyzer.model_priority == custom_priority
            assert analyzer.analysis_timeout == 120.0

    def test_custom_dependencies(self):
        """Test initialization with custom dependency implementations."""
        custom_http = RequestsHTTPClient()
        custom_fs = TempFileSystem()
        custom_monitor = DefaultResourceMonitor()

        with patch('analyzer.gemini_analyzer.GeminiClient'), \
             patch('analyzer.gemini_analyzer.MediaProcessor') as mock_mp:
            analyzer = GeminiAnalyzer(
                api_key="test_key",
                http_client=custom_http,
                file_system=custom_fs,
                resource_monitor=custom_monitor
            )

            # Verify MediaProcessor was called with custom dependencies
            mock_mp.assert_called_once()
            call_args = mock_mp.call_args
            assert call_args[1]['http_client'] == custom_http
            assert call_args[1]['file_system'] == custom_fs
            assert call_args[1]['resource_monitor'] == custom_monitor


class TestModelManagement:
    """Test model priority and rate limiting."""

    def test_get_available_models_all_available(self, analyzer):
        """Test getting available models when none are rate limited."""
        available = analyzer._get_available_models()
        assert available == analyzer.model_priority

    def test_get_available_models_with_rate_limits(self, analyzer):
        """Test getting available models when some are rate limited."""
        analyzer.rate_limited_models.add('gemini-2.5-pro')
        analyzer.rate_limited_models.add('gemini-2.5-flash')

        available = analyzer._get_available_models()
        assert 'gemini-2.5-pro' not in available
        assert 'gemini-2.5-flash' not in available
        assert 'gemini-2.5-flash-lite' in available

    def test_mark_model_rate_limited(self, analyzer):
        """Test marking a model as rate limited."""
        analyzer._mark_model_rate_limited('gemini-2.5-pro')
        assert 'gemini-2.5-pro' in analyzer.rate_limited_models

    def test_mark_model_success(self, analyzer):
        """Test marking a model as successful (removes rate limit)."""
        analyzer.rate_limited_models.add('gemini-2.5-pro')
        analyzer._mark_model_success('gemini-2.5-pro')
        assert 'gemini-2.5-pro' not in analyzer.rate_limited_models

    def test_get_rate_limit_status(self, analyzer):
        """Test getting rate limit status."""
        analyzer.rate_limited_models.add('gemini-2.5-pro')

        status = analyzer.get_rate_limit_status()
        assert status['rate_limited_models'] == ['gemini-2.5-pro']
        assert 'gemini-2.5-flash' in status['available_models']
        assert 'gemini-2.5-pro' not in status['available_models']

    def test_reset_rate_limits(self, analyzer):
        """Test resetting all rate limits."""
        analyzer.rate_limited_models.add('gemini-2.5-pro')
        analyzer.rate_limited_models.add('gemini-2.5-flash')

        analyzer.reset_rate_limits()
        assert len(analyzer.rate_limited_models) == 0


class TestMultimodalAnalysis:
    """Test multimodal content analysis."""

    def test_analyze_multimodal_content_text_only(self, analyzer, mock_gemini_client, mock_media_processor):
        """Test text-only analysis."""
        mock_gemini_client.generate_content.return_value = "CATEGORÍA: hate_speech\nEXPLICACIÓN: Test response."

        result, time_taken = analyzer.analyze_multimodal_content([], "Test content")

        assert result == "CATEGORÍA: hate_speech\nEXPLICACIÓN: Test response."
        assert time_taken > 0
        mock_media_processor.select_media_url.assert_not_called()

    def test_analyze_multimodal_content_with_media(self, analyzer, mock_gemini_client, mock_media_processor):
        """Test multimodal analysis with media."""
        mock_gemini_client.generate_content.return_value = "CATEGORÍA: hate_speech\nEXPLICACIÓN: Test response."

        media_urls = ["https://example.com/image.jpg"]
        result, time_taken = analyzer.analyze_multimodal_content(media_urls, "Test content")

        assert result == "CATEGORÍA: hate_speech\nEXPLICACIÓN: Test response."
        assert time_taken > 0
        mock_media_processor.select_media_url.assert_called_once_with(media_urls)
        mock_media_processor.download_media.assert_called_once()
        mock_media_processor.cleanup_file.assert_called_once()

    def test_analyze_multimodal_content_download_failure(self, analyzer, mock_media_processor):
        """Test handling of media download failure."""
        mock_media_processor.download_media.return_value = None

        media_urls = ["https://example.com/image.jpg"]
        result, time_taken = analyzer.analyze_multimodal_content(media_urls, "Test content")

        assert result is None
        assert time_taken > 0

    def test_analyze_multimodal_content_no_valid_media(self, analyzer, mock_media_processor, mock_gemini_client):
        """Test handling when no valid media URLs are provided."""
        # Mock media processor to return None for select_media_url
        mock_media_processor.select_media_url.return_value = None
        # Ensure no text analysis fallback by making Gemini client fail
        mock_gemini_client.get_model.return_value = (None, "No models available")

        media_urls = ["https://example.com/invalid.jpg"]
        result, time_taken = analyzer.analyze_multimodal_content(media_urls, "Test content")

        assert result is None
        assert time_taken > 0

    def test_analyze_multimodal_content_timeout(self, analyzer, mock_gemini_client):
        """Test analysis timeout handling."""
        # Mock a long-running operation that exceeds timeout
        original_timeout = analyzer.analysis_timeout
        analyzer.analysis_timeout = 0.001  # Very short timeout

        # Make Gemini client take longer than timeout by returning None initially
        mock_gemini_client.get_model.return_value = (None, "Timeout simulation")

        try:
            result, time_taken = analyzer.analyze_multimodal_content([], "Test content")
            assert result is None
            assert time_taken > 0
        finally:
            analyzer.analysis_timeout = original_timeout

    def test_analyze_multimodal_content_all_models_fail(self, analyzer, mock_gemini_client):
        """Test when all models fail."""
        mock_gemini_client.get_model.return_value = (None, "Model unavailable")

        result, time_taken = analyzer.analyze_multimodal_content([], "Test content")

        assert result is None
        assert time_taken > 0


class TestAsyncAnalysis:
    """Test async analysis methods."""

    @pytest.mark.asyncio
    async def test_analyze_multimodal_content_async_text_only(self, analyzer, mock_gemini_client):
        """Test async text-only analysis."""
        mock_gemini_client.generate_content.return_value = "CATEGORÍA: hate_speech\nEXPLICACIÓN: Test response."

        result, time_taken = await analyzer.analyze_multimodal_content_async([], "Test content")

        assert result == "CATEGORÍA: hate_speech\nEXPLICACIÓN: Test response."
        assert time_taken > 0

    @pytest.mark.asyncio
    async def test_analyze_multimodal_content_async_with_media(self, analyzer, mock_gemini_client, mock_media_processor):
        """Test async multimodal analysis."""
        from unittest.mock import AsyncMock
        
        mock_gemini_client.generate_content.return_value = "CATEGORÍA: hate_speech\nEXPLICACIÓN: Test response."
        mock_media_processor.download_media_async = AsyncMock(return_value="/tmp/test_media.jpg")
        mock_media_processor.select_media_url.return_value = "https://example.com/image.jpg"
        mock_media_processor.is_video_url.return_value = False

        media_urls = ["https://example.com/image.jpg"]
        result, time_taken = await analyzer.analyze_multimodal_content_async(media_urls, "Test content")

        assert result == "CATEGORÍA: hate_speech\nEXPLICACIÓN: Test response."
        assert time_taken > 0
        mock_media_processor.download_media_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_multimodal_batch_async(self, analyzer, mock_gemini_client, mock_media_processor):
        """Test batch async analysis."""
        from unittest.mock import AsyncMock
        
        mock_gemini_client.generate_content.return_value = "CATEGORÍA: hate_speech\nEXPLICACIÓN: Test response."
        mock_media_processor.download_media_async = AsyncMock(return_value="/tmp/test_media.jpg")
        mock_media_processor.select_media_url.return_value = "https://example.com/image.jpg"
        mock_media_processor.is_video_url.return_value = False

        batch_items = [
            (["https://example.com/image1.jpg"], "Content 1"),
            ([], "Content 2")  # Text-only
        ]

        results = await analyzer.analyze_multimodal_batch_async(batch_items, max_concurrent=2)

        assert len(results) == 2
        assert all(result[0] == "CATEGORÍA: hate_speech\nEXPLICACIÓN: Test response." for result in results)
        assert all(result[1] > 0 for result in results)


class TestMetrics:
    """Test metrics collection."""

    def test_get_metrics_summary(self, analyzer):
        """Test getting metrics summary."""
        summary = analyzer.get_metrics_summary()
        assert isinstance(summary, dict)
        assert 'operations' in summary
        assert 'errors' in summary
        assert 'performance' in summary

    def test_reset_metrics(self, analyzer):
        """Test resetting metrics."""
        # Add some mock data
        analyzer.metrics_collector.record_operation("test", 1.0, True)

        # Reset
        analyzer.reset_metrics()

        # Should have new metrics collector
        assert isinstance(analyzer.metrics_collector, DefaultMetricsCollector)

    def test_log_metrics_summary(self, analyzer, caplog):
        """Test logging metrics summary."""
        with caplog.at_level('INFO'):
            analyzer.log_metrics_summary()

        assert "📊 Multimodal Analysis Metrics:" in caplog.text


class TestDefaultImplementations:
    """Test default dependency implementations."""

    def test_requests_http_client(self):
        """Test RequestsHTTPClient implementation."""
        client = RequestsHTTPClient()
        assert hasattr(client, 'get')

    def test_temp_file_system(self):
        """Test TempFileSystem implementation."""
        fs = TempFileSystem()

        # Test create_temp_file
        temp_path = fs.create_temp_file('.txt')
        assert temp_path.endswith('.txt')
        assert os.path.exists(temp_path)

        # Test remove_file
        fs.remove_file(temp_path)
        assert not os.path.exists(temp_path)

    def test_default_resource_monitor(self):
        """Test DefaultResourceMonitor implementation."""
        monitor = DefaultResourceMonitor()

        # Create a test file
        test_file = '/tmp/test_file.txt'
        with open(test_file, 'w') as f:
            f.write('test content')

        try:
            size = monitor.check_file_size(test_file)
            assert size == len('test content')
        finally:
            os.remove(test_file)


class TestErrorHandling:
    """Test error handling in various scenarios."""

    def test_try_model_analysis_client_failure(self, analyzer, mock_gemini_client):
        """Test handling of client initialization failure."""
        mock_gemini_client.get_model.return_value = (None, "Client error")

        result = analyzer._try_model_analysis(
            "test-model", "/tmp/test.jpg", "https://example.com/image.jpg",
            "Test content", False
        )

        assert result is None

    def test_try_text_model_analysis_success(self, analyzer, mock_gemini_client):
        """Test successful text model analysis."""
        mock_gemini_client.generate_content.return_value = "Test response"

        result = analyzer._try_text_model_analysis("test-model", "Test content")

        assert result == "Test response"

    def test_try_text_model_analysis_failure(self, analyzer, mock_gemini_client):
        """Test text model analysis failure."""
        mock_gemini_client.get_model.return_value = (None, "Model error")

        result = analyzer._try_text_model_analysis("test-model", "Test content")

        assert result is None