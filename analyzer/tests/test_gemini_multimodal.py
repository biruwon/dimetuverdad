#!/usr/bin/env python3
"""
Unit tests for Gemini multimodal analysis functionality.
"""

import unittest
import os
from unittest.mock import patch, MagicMock
from analyzer.gemini_multimodal import (
    GeminiMultimodal,
    GeminiMultimodalConfig,
    DependencyContainer,
    classify_error,
    ErrorCategory,
    AnalysisError
)
from analyzer.prompts import EnhancedPromptGenerator


class TestGeminiMultimodal(unittest.TestCase):
    """Test cases for Gemini multimodal analysis class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_image_url = "https://pbs.twimg.com/media/1973243448871284736/bPeZHL3l?format=jpg&name=small"
        self.test_video_url = "https://video.twimg.com/amplify_video/1972307252796141568/vid/avc1/320x568/GftH9VZYZuygizQc.mp4"
        self.test_text = "A juicio el hermano, la mujer, el fiscal y dos secretarios de organización del presidente."

    def _create_mock_dependencies(self):
        """Create mock dependencies for testing."""
        config = GeminiMultimodalConfig(api_key="test_key")

        # Create mock dependencies
        mock_http_client = MagicMock()
        mock_file_system = MagicMock()
        mock_resource_monitor = MagicMock()
        mock_metrics_collector = MagicMock()

        # Set up default behaviors
        mock_resource_monitor.check_memory_usage.return_value = 100.0  # 100MB
        mock_resource_monitor.check_file_size.return_value = 1024  # 1KB
        mock_metrics_collector.record_operation.return_value = None
        mock_metrics_collector.get_summary.return_value = {}

        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {'content-type': 'image/jpeg', 'content-length': '1024'}
        mock_response.iter_content.return_value = [b'test', b'data']
        mock_http_client.get.return_value = mock_response

        # Mock file system
        mock_file_system.create_temp_file.return_value = "/tmp/test.jpg"
        mock_file_system.remove_file.return_value = None

        return DependencyContainer(
            http_client=mock_http_client,
            file_system=mock_file_system,
            resource_monitor=mock_resource_monitor,
            metrics_collector=mock_metrics_collector,
            config=config
        )

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = GeminiMultimodalConfig(api_key="test_key")
        self.assertEqual(config.api_key, "test_key")

        # Invalid config - no API key
        with self.assertRaises(ValueError):
            GeminiMultimodalConfig(api_key="")

    def test_analyze_multimodal_success(self):
        """Test successful multimodal analysis."""
        deps = self._create_mock_dependencies()
        analyzer = GeminiMultimodal(deps)

        # Setup Gemini mocks
        with patch('analyzer.gemini_multimodal.genai.configure'), \
             patch('analyzer.gemini_multimodal.genai.GenerativeModel') as mock_generative_model, \
             patch('analyzer.gemini_multimodal.genai.upload_file') as mock_upload_file:

            mock_model = MagicMock()
            mock_generative_model.return_value = mock_model

            mock_file = MagicMock()
            mock_file.state.name = "ACTIVE"
            mock_upload_file.return_value = mock_file

            # Mock the generate_content response
            mock_response = MagicMock()
            mock_response.text = "Test analysis result"
            mock_model.generate_content.return_value = mock_response

            result, time_taken = analyzer.analyze_multimodal_content([self.test_image_url], self.test_text)

            self.assertEqual(result, "Test analysis result")
            self.assertIsInstance(time_taken, float)
            self.assertGreater(time_taken, 0)

    def test_get_metrics_summary(self):
        """Test metrics summary retrieval."""
        deps = self._create_mock_dependencies()
        analyzer = GeminiMultimodal(deps)

        summary = analyzer.get_metrics_summary()
        self.assertIsInstance(summary, dict)

    def test_error_classification_network_error(self):
        """Test error classification for network errors."""
        error = Exception("Connection timeout")
        analysis_error = classify_error(error, "test context")

        self.assertEqual(analysis_error.category, ErrorCategory.TIMEOUT_ERROR)
        self.assertTrue(analysis_error.recoverable)
        self.assertEqual(analysis_error.retry_delay, 5)


class TestGeminiMultimodalConfig(unittest.TestCase):
    """Test cases for GeminiMultimodalConfig class."""

    def test_config_validation_invalid_timeout(self):
        """Test configuration validation with invalid timeout values."""
        # Invalid analysis timeout
        with self.assertRaises(ValueError):
            GeminiMultimodalConfig(api_key="test_key", analysis_timeout=0)

        with self.assertRaises(ValueError):
            GeminiMultimodalConfig(api_key="test_key", analysis_timeout=-1)

        # Invalid download timeout
        with self.assertRaises(ValueError):
            GeminiMultimodalConfig(api_key="test_key", download_timeout=0)

    def test_config_validation_invalid_retries(self):
        """Test configuration validation with invalid retry values."""
        with self.assertRaises(ValueError):
            GeminiMultimodalConfig(api_key="test_key", max_retries=-1)

    def test_config_validation_invalid_file_size_limits(self):
        """Test configuration validation with invalid file size limits."""
        # Invalid video size limit
        with self.assertRaises(ValueError):
            GeminiMultimodalConfig(api_key="test_key", max_video_size_mb=0)

        with self.assertRaises(ValueError):
            GeminiMultimodalConfig(api_key="test_key", max_video_size_mb=-1)

        # Invalid image size limit
        with self.assertRaises(ValueError):
            GeminiMultimodalConfig(api_key="test_key", max_image_size_mb=0)

        with self.assertRaises(ValueError):
            GeminiMultimodalConfig(api_key="test_key", max_image_size_mb=-5)

    def test_config_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = GeminiMultimodalConfig(
            api_key="test_key",
            model_name="gemini-test",
            max_retries=5,
            max_video_size_mb=50.0,
            max_image_size_mb=5.0
        )
        config_dict = config.to_dict()

        # API key should be redacted
        self.assertEqual(config_dict['api_key'], '***REDACTED***')
        self.assertEqual(config_dict['model_name'], 'gemini-test')
        self.assertEqual(config_dict['max_retries'], 5)
        self.assertEqual(config_dict['max_video_size_mb'], 50.0)
        self.assertEqual(config_dict['max_image_size_mb'], 5.0)

    def test_config_str_representation(self):
        """Test string representation of configuration."""
        config = GeminiMultimodalConfig(api_key="test_key")
        str_repr = str(config)

        self.assertIn('GeminiMultimodalConfig', str_repr)
        self.assertIn('***REDACTED***', str_repr)  # API key should be redacted
        self.assertNotIn('test_key', str_repr)  # Actual API key should not appear

    def test_config_default_model_priority(self):
        """Test that default model priority is set correctly."""
        config = GeminiMultimodalConfig(api_key="test_key")
        expected_priority = [
            'gemini-2.5-pro',
            'gemini-2.5-flash',
            'gemini-2.5-flash-lite',
            'gemini-2.0-flash-exp',
            'gemini-2.0-flash-lite',
            'gemini-2.0-flash'
        ]
        self.assertEqual(config.model_priority, expected_priority)

    def test_config_custom_model_priority(self):
        """Test custom model priority setting."""
        custom_priority = ['custom-model-1', 'custom-model-2']
        config = GeminiMultimodalConfig(
            api_key="test_key",
            model_priority=custom_priority
        )
        self.assertEqual(config.model_priority, custom_priority)


class TestGeminiMultimodalInitialization(unittest.TestCase):
    """Test cases for GeminiMultimodal initialization."""

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_env_key'})
    def test_init_with_default_dependencies(self):
        """Test initialization with default dependencies."""
        analyzer = GeminiMultimodal()
        self.assertIsNotNone(analyzer.dependencies)
        self.assertEqual(analyzer.dependencies.config.api_key, 'test_env_key')

    @patch.dict(os.environ, {}, clear=True)
    def test_init_missing_api_key(self):
        """Test initialization failure when API key is missing."""
        with self.assertRaises(ValueError):
            GeminiMultimodal()

    def test_init_with_custom_dependencies(self):
        """Test initialization with custom dependencies."""
        deps = DependencyContainer(
            http_client=MagicMock(),
            file_system=MagicMock(),
            resource_monitor=MagicMock(),
            metrics_collector=MagicMock(),
            config=GeminiMultimodalConfig(api_key="custom_key")
        )
        analyzer = GeminiMultimodal(deps)
        self.assertEqual(analyzer.dependencies.config.api_key, "custom_key")


class TestGeminiMultimodalMediaSelection(unittest.TestCase):
    """Test cases for media URL selection logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.deps = DependencyContainer(
            http_client=MagicMock(),
            file_system=MagicMock(),
            resource_monitor=MagicMock(),
            metrics_collector=MagicMock(),
            config=GeminiMultimodalConfig(api_key="test_key")
        )
        self.analyzer = GeminiMultimodal(self.deps)

    def test_select_media_url_mp4_priority(self):
        """Test MP4 file priority in media URL selection."""
        urls = [
            "https://example.com/image.jpg",
            "https://example.com/video.mp4",
            "https://example.com/other.m3u8"
        ]
        selected = self.analyzer._select_media_url(urls)
        self.assertEqual(selected, "https://example.com/video.mp4")

    def test_select_media_url_m3u8_priority(self):
        """Test M3U8 file priority in media URL selection."""
        urls = [
            "https://example.com/image.jpg",
            "https://example.com/video.m3u8",
            "https://example.com/other.mp4"
        ]
        # MP4 has higher priority than M3U8, so other.mp4 should be selected
        selected = self.analyzer._select_media_url(urls)
        self.assertEqual(selected, "https://example.com/other.mp4")

    def test_select_media_url_video_pattern(self):
        """Test video URL pattern priority in media URL selection."""
        urls = [
            "https://example.com/image.jpg",
            "https://example.com/video_content",
            "https://example.com/other.jpg"
        ]
        selected = self.analyzer._select_media_url(urls)
        self.assertEqual(selected, "https://example.com/video_content")

    def test_select_media_url_empty_list(self):
        """Test empty URL list error handling."""
        with self.assertRaises(ValueError):
            self.analyzer._select_media_url([])

    def test_is_video_url_various_formats(self):
        """Test video URL detection for various formats."""
        # Video formats
        self.assertTrue(self.analyzer._is_video_url("https://example.com/video.mp4"))
        self.assertTrue(self.analyzer._is_video_url("https://example.com/video.M3U8"))
        self.assertTrue(self.analyzer._is_video_url("https://example.com/video.mov"))
        self.assertTrue(self.analyzer._is_video_url("https://example.com/video.avi"))
        self.assertTrue(self.analyzer._is_video_url("https://example.com/video.webm"))

        # Non-video formats
        self.assertFalse(self.analyzer._is_video_url("https://example.com/image.jpg"))
        self.assertFalse(self.analyzer._is_video_url("https://example.com/image.png"))


class TestGeminiMultimodalDownload(unittest.TestCase):
    """Test cases for media download functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.deps = DependencyContainer(
            http_client=MagicMock(),
            file_system=MagicMock(),
            resource_monitor=MagicMock(),
            metrics_collector=MagicMock(),
            config=GeminiMultimodalConfig(api_key="test_key")
        )
        self.analyzer = GeminiMultimodal(self.deps)

    def test_download_media_success(self):
        """Test successful media download."""
        # Mock successful HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = [b'test', b'data']
        self.deps.http_client.get.return_value = mock_response

        # Mock file system
        self.deps.file_system.create_temp_file.return_value = "/tmp/test.jpg"
        self.deps.resource_monitor.check_file_size.return_value = 1024

        result = self.analyzer._download_media("https://example.com/image.jpg", False)

        self.assertEqual(result, "/tmp/test.jpg")
        self.deps.http_client.get.assert_called_once()
        self.deps.file_system.create_temp_file.assert_called_once_with('.jpg')

    def test_download_media_failure(self):
        """Test media download failure handling."""
        # Mock failed HTTP response
        self.deps.http_client.get.side_effect = Exception("Connection failed")

        result = self.analyzer._download_media("https://example.com/image.jpg", False)

        self.assertIsNone(result)

    def test_download_media_empty_file(self):
        """Test handling of empty downloaded files."""
        # Mock response with empty content
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = []
        self.deps.http_client.get.return_value = mock_response

        self.deps.file_system.create_temp_file.return_value = "/tmp/test.jpg"
        self.deps.resource_monitor.check_file_size.return_value = 0  # Empty file

        result = self.analyzer._download_media("https://example.com/image.jpg", False)

        self.assertIsNone(result)
        # File should be cleaned up
        self.deps.file_system.remove_file.assert_called_once_with("/tmp/test.jpg")

    def test_download_media_video_size_limit_exceeded(self):
        """Test video file size limit enforcement."""
        # Mock response with large video content
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = [b'test'] * 1000
        self.deps.http_client.get.return_value = mock_response

        self.deps.file_system.create_temp_file.return_value = "/tmp/test.mp4"
        # Simulate file size of 150MB (exceeds 100MB limit)
        self.deps.resource_monitor.check_file_size.return_value = 150 * 1024 * 1024

        result = self.analyzer._download_media("https://example.com/video.mp4", True)

        self.assertIsNone(result)
        # File should be cleaned up due to size limit
        self.deps.file_system.remove_file.assert_called_once_with("/tmp/test.mp4")

    def test_download_media_image_size_limit_exceeded(self):
        """Test image file size limit enforcement."""
        # Mock response with large image content
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = [b'test'] * 100
        self.deps.http_client.get.return_value = mock_response

        self.deps.file_system.create_temp_file.return_value = "/tmp/test.jpg"
        # Simulate file size of 15MB (exceeds 10MB limit)
        self.deps.resource_monitor.check_file_size.return_value = 15 * 1024 * 1024

        result = self.analyzer._download_media("https://example.com/image.jpg", False)

        self.assertIsNone(result)
        # File should be cleaned up due to size limit
        self.deps.file_system.remove_file.assert_called_once_with("/tmp/test.jpg")

    def test_download_media_size_limit_within_bounds(self):
        """Test that files within size limits are accepted."""
        # Mock response with acceptable content
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = [b'test'] * 100
        self.deps.http_client.get.return_value = mock_response

        self.deps.file_system.create_temp_file.return_value = "/tmp/test.jpg"
        # Simulate file size of 5MB (within 10MB limit for images)
        self.deps.resource_monitor.check_file_size.return_value = 5 * 1024 * 1024

        result = self.analyzer._download_media("https://example.com/image.jpg", False)

        self.assertEqual(result, "/tmp/test.jpg")
        # File should NOT be cleaned up
        self.deps.file_system.remove_file.assert_not_called()


class TestGeminiMultimodalModelAnalysis(unittest.TestCase):
    """Test cases for model analysis functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.deps = DependencyContainer(
            http_client=MagicMock(),
            file_system=MagicMock(),
            resource_monitor=MagicMock(),
            metrics_collector=MagicMock(),
            config=GeminiMultimodalConfig(api_key="test_key")
        )
        self.analyzer = GeminiMultimodal(self.deps)

    @patch('analyzer.gemini_multimodal.genai.configure')
    @patch('analyzer.gemini_multimodal.genai.GenerativeModel')
    @patch('analyzer.gemini_multimodal.genai.upload_file')
    def test_try_model_analysis_success(self, mock_upload_file, mock_generative_model, mock_configure):
        """Test successful model analysis."""
        # Mock Gemini components
        mock_model = MagicMock()
        mock_generative_model.return_value = mock_model

        mock_file = MagicMock()
        mock_file.state.name = "ACTIVE"
        mock_upload_file.return_value = mock_file

        mock_response = MagicMock()
        mock_response.text = "Analysis result"
        mock_model.generate_content.return_value = mock_response

        result = self.analyzer._try_model_analysis(
            "gemini-test", "/tmp/test.jpg", "https://example.com/image.jpg",
            "Test content", False
        )

        self.assertEqual(result, "Analysis result")

    @patch('analyzer.gemini_multimodal.genai.configure')
    @patch('analyzer.gemini_multimodal.genai.GenerativeModel')
    def test_try_model_analysis_timeout(self, mock_generative_model, mock_configure):
        """Test model analysis timeout handling."""
        mock_model = MagicMock()
        mock_generative_model.return_value = mock_model

        # Mock timeout
        mock_model.generate_content.side_effect = TimeoutError()

        result = self.analyzer._try_model_analysis(
            "gemini-test", "/tmp/test.jpg", "https://example.com/image.jpg",
            "Test content", False
        )

        self.assertIsNone(result)

    @patch('analyzer.gemini_multimodal.genai.configure')
    def test_try_model_analysis_client_failure(self, mock_configure):
        """Test model analysis when client creation fails."""
        mock_configure.side_effect = Exception("API key invalid")

        result = self.analyzer._try_model_analysis(
            "gemini-test", "/tmp/test.jpg", "https://example.com/image.jpg",
            "Test content", False
        )

        self.assertIsNone(result)


class TestGeminiMultimodalGeminiClient(unittest.TestCase):
    """Test cases for Gemini client functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.deps = DependencyContainer(
            http_client=MagicMock(),
            file_system=MagicMock(),
            resource_monitor=MagicMock(),
            metrics_collector=MagicMock(),
            config=GeminiMultimodalConfig(api_key="test_key")
        )
        self.analyzer = GeminiMultimodal(self.deps)

    @patch('analyzer.gemini_multimodal.genai.configure')
    @patch('analyzer.gemini_multimodal.genai.GenerativeModel')
    def test_get_gemini_client_success(self, mock_generative_model, mock_configure):
        """Test successful Gemini client creation."""
        mock_model = MagicMock()
        mock_generative_model.return_value = mock_model

        model, error = self.analyzer._get_gemini_client("gemini-test")

        self.assertEqual(model, mock_model)
        self.assertIsNone(error)

    @patch('analyzer.gemini_multimodal.genai.configure')
    def test_get_gemini_client_failure(self, mock_configure):
        """Test Gemini client creation failure."""
        mock_configure.side_effect = Exception("Invalid API key")

        model, error = self.analyzer._get_gemini_client("gemini-test")

        self.assertIsNone(model)
        self.assertIsNotNone(error)

    @patch('analyzer.gemini_multimodal.genai.configure')
    @patch('analyzer.gemini_multimodal.genai.GenerativeModel')
    @patch('analyzer.gemini_multimodal.genai.upload_file')
    @patch('analyzer.gemini_multimodal.genai.get_file')
    def test_upload_media_to_gemini_success(self, mock_get_file, mock_upload_file, mock_generative_model, mock_configure):
        """Test successful media upload to Gemini."""
        mock_model = MagicMock()
        mock_generative_model.return_value = mock_model

        mock_file = MagicMock()
        mock_file.state.name = "ACTIVE"
        mock_upload_file.return_value = mock_file
        mock_get_file.return_value = mock_file

        result = self.analyzer._upload_media_to_gemini(mock_model, "/tmp/test.jpg", "https://example.com/image.jpg")

        self.assertEqual(result, mock_file)

    @patch('analyzer.gemini_multimodal.genai.configure')
    @patch('analyzer.gemini_multimodal.genai.GenerativeModel')
    @patch('analyzer.gemini_multimodal.genai.upload_file')
    def test_upload_media_to_gemini_processing_failure(self, mock_upload_file, mock_generative_model, mock_configure):
        """Test media upload failure due to processing issues."""
        mock_model = MagicMock()
        mock_generative_model.return_value = mock_model

        mock_file = MagicMock()
        mock_file.state.name = "FAILED"
        mock_upload_file.return_value = mock_file

        result = self.analyzer._upload_media_to_gemini(mock_model, "/tmp/test.jpg", "https://example.com/image.jpg")

        self.assertIsNone(result)


class TestGeminiMultimodalAsync(unittest.TestCase):
    """Test cases for async functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.deps = DependencyContainer(
            http_client=MagicMock(),
            file_system=MagicMock(),
            resource_monitor=MagicMock(),
            metrics_collector=MagicMock(),
            config=GeminiMultimodalConfig(api_key="test_key")
        )
        self.analyzer = GeminiMultimodal(self.deps)

    @patch('analyzer.gemini_multimodal.genai.configure')
    @patch('analyzer.gemini_multimodal.genai.GenerativeModel')
    @patch('analyzer.gemini_multimodal.genai.upload_file')
    @patch('analyzer.gemini_multimodal.genai.get_file')
    def test_analyze_multimodal_content_async_success(self, mock_get_file, mock_upload_file, mock_generative_model, mock_configure):
        """Test successful async multimodal analysis."""
        import asyncio

        # Mock Gemini components
        mock_model = MagicMock()
        mock_generative_model.return_value = mock_model

        mock_file = MagicMock()
        mock_file.state.name = "ACTIVE"
        mock_upload_file.return_value = mock_file
        mock_get_file.return_value = mock_file

        mock_response = MagicMock()
        mock_response.text = "Async analysis result"
        mock_model.generate_content.return_value = mock_response

        # Mock download
        mock_response_dl = MagicMock()
        mock_response_dl.status_code = 200
        mock_response_dl.raise_for_status.return_value = None
        mock_response_dl.iter_content.return_value = [b'test']
        self.deps.http_client.get.return_value = mock_response_dl
        self.deps.file_system.create_temp_file.return_value = "/tmp/test.jpg"
        self.deps.resource_monitor.check_file_size.return_value = 1024

        async def run_test():
            result, time_taken = await self.analyzer.analyze_multimodal_content_async(
                ["https://example.com/image.jpg"], "Test content"
            )
            self.assertEqual(result, "Async analysis result")
            self.assertIsInstance(time_taken, float)

        asyncio.run(run_test())


class TestGeminiMultimodalErrorHandling(unittest.TestCase):
    """Test cases for comprehensive error handling."""

    def test_analyze_multimodal_content_no_media_urls(self):
        """Test analysis with no media URLs."""
        deps = DependencyContainer(
            http_client=MagicMock(),
            file_system=MagicMock(),
            resource_monitor=MagicMock(),
            metrics_collector=MagicMock(),
            config=GeminiMultimodalConfig(api_key="test_key")
        )
        analyzer = GeminiMultimodal(deps)

        result, time_taken = analyzer.analyze_multimodal_content([], "Test content")

        self.assertIsNone(result)
        self.assertGreater(time_taken, 0)

    def test_analyze_multimodal_content_filtered_urls_empty(self):
        """Test analysis when all URLs are filtered out."""
        deps = DependencyContainer(
            http_client=MagicMock(),
            file_system=MagicMock(),
            resource_monitor=MagicMock(),
            metrics_collector=MagicMock(),
            config=GeminiMultimodalConfig(api_key="test_key")
        )
        analyzer = GeminiMultimodal(deps)

        # URLs that will be filtered out
        filtered_urls = [
            "https://example.com/profile_images/test.jpg",
            "https://example.com/card_img/test.png"
        ]

        result, time_taken = analyzer.analyze_multimodal_content(filtered_urls, "Test content")

        self.assertIsNone(result)
        self.assertGreater(time_taken, 0)

    def test_error_classification_authentication_error(self):
        """Test authentication error classification."""
        error = Exception("API key invalid")
        analysis_error = classify_error(error, "test context")

        self.assertEqual(analysis_error.category, ErrorCategory.AUTHENTICATION_ERROR)
        self.assertFalse(analysis_error.recoverable)

    def test_error_classification_quota_error(self):
        """Test quota error classification."""
        error = Exception("Quota exceeded")
        analysis_error = classify_error(error, "test context")

        self.assertEqual(analysis_error.category, ErrorCategory.QUOTA_ERROR)
        self.assertTrue(analysis_error.recoverable)
        self.assertEqual(analysis_error.retry_delay, 60)

    def test_error_classification_model_error(self):
        """Test model error classification."""
        error = Exception("Model not found")
        analysis_error = classify_error(error, "test context")

        self.assertEqual(analysis_error.category, ErrorCategory.MODEL_ERROR)
        self.assertTrue(analysis_error.recoverable)
        self.assertEqual(analysis_error.retry_delay, 10)

    def test_error_classification_media_error(self):
        """Test media error classification."""
        error = Exception("File upload failed")
        analysis_error = classify_error(error, "test context")

        self.assertEqual(analysis_error.category, ErrorCategory.MEDIA_ERROR)
        self.assertFalse(analysis_error.recoverable)

    def test_error_classification_configuration_error(self):
        """Test configuration error classification."""
        error = Exception("Environment variable missing")
        analysis_error = classify_error(error, "test context")

        self.assertEqual(analysis_error.category, ErrorCategory.CONFIGURATION_ERROR)
        self.assertFalse(analysis_error.recoverable)

    def test_error_classification_unknown_error(self):
        """Test unknown error classification."""
        error = Exception("Some random error")
        analysis_error = classify_error(error, "test context")

        self.assertEqual(analysis_error.category, ErrorCategory.UNKNOWN_ERROR)
        self.assertFalse(analysis_error.recoverable)


class TestGeminiMultimodalUtilities(unittest.TestCase):
    """Test cases for utility methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.deps = DependencyContainer(
            http_client=MagicMock(),
            file_system=MagicMock(),
            resource_monitor=MagicMock(),
            metrics_collector=MagicMock(),
            config=GeminiMultimodalConfig(api_key="test_key")
        )
        self.analyzer = GeminiMultimodal(self.deps)

    def test_cleanup_file(self):
        """Test file cleanup functionality."""
        self.analyzer._cleanup_file("/tmp/test.jpg")
        self.deps.file_system.remove_file.assert_called_once_with("/tmp/test.jpg")

    def test_log_metrics_summary(self):
        """Test metrics summary logging."""
        with patch.object(self.analyzer.logger, 'info') as mock_info:
            self.analyzer.log_metrics_summary()
            mock_info.assert_called()

    def test_reset_metrics(self):
        """Test metrics reset functionality."""
        old_collector = self.deps.metrics_collector
        self.analyzer.reset_metrics()

        # Metrics collector should be replaced
        self.assertNotEqual(self.deps.metrics_collector, old_collector)


if __name__ == '__main__':
    unittest.main()
