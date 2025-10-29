"""
Tests for GeminiClient class.

Comprehensive test suite for the Gemini API client functionality.
"""

import os
import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from analyzer.gemini_client import GeminiClient
from analyzer.error_handler import ErrorCategory


class TestGeminiClientInitialization:
    """Test GeminiClient initialization."""

    def test_default_initialization(self):
        """Test initialization with default logger."""
        client = GeminiClient("test-api-key")
        assert client.api_key == "test-api-key"
        assert isinstance(client.logger, logging.Logger)

    def test_custom_logger_initialization(self):
        """Test initialization with custom logger."""
        custom_logger = logging.getLogger("custom")
        client = GeminiClient("test-api-key", custom_logger)
        assert client.api_key == "test-api-key"
        assert client.logger == custom_logger


class TestGetModel:
    """Test model initialization functionality."""

    def test_get_model_success(self):
        """Test successful model initialization."""
        client = GeminiClient("test-api-key")

        with patch('google.generativeai.configure') as mock_configure, \
             patch('google.generativeai.GenerativeModel') as mock_model_class:

            mock_model = Mock()
            mock_model_class.return_value = mock_model

            result, error = client.get_model("gemini-1.5-flash")

            assert result == mock_model
            assert error is None
            mock_configure.assert_called_once_with(api_key="test-api-key")
            mock_model_class.assert_called_once_with("gemini-1.5-flash")

    def test_get_model_failure(self):
        """Test model initialization failure."""
        client = GeminiClient("test-api-key")

        with patch('google.generativeai.configure') as mock_configure, \
             patch('google.generativeai.GenerativeModel') as mock_model_class:

            mock_model_class.side_effect = Exception("API Error")

            result, error = client.get_model("gemini-1.5-flash")

            assert result is None
            assert error is not None
            assert "API Error" in error

    def test_get_model_quota_error(self):
        """Test model initialization with quota error."""
        client = GeminiClient("test-api-key")

        with patch('google.generativeai.configure') as mock_configure, \
             patch('google.generativeai.GenerativeModel') as mock_model_class:

            mock_model_class.side_effect = Exception("Quota exceeded")

            result, error = client.get_model("gemini-1.5-flash")

            assert result is None
            assert error is not None


class TestPrepareMediaForAnalysis:
    """Test media preparation functionality."""

    def test_prepare_media_file_not_exists(self):
        """Test handling when media file doesn't exist."""
        client = GeminiClient("test-api-key")

        result = client.prepare_media_for_analysis("/nonexistent/file.jpg", "https://example.com/image.jpg")

        assert result is None

    def test_prepare_media_empty_file(self, tmp_path):
        """Test handling of empty media file."""
        client = GeminiClient("test-api-key")

        # Create empty file
        empty_file = tmp_path / "empty.jpg"
        empty_file.write_bytes(b"")

        result = client.prepare_media_for_analysis(str(empty_file), "https://example.com/image.jpg")

        assert result is None

    def test_prepare_media_image_success(self, tmp_path):
        """Test successful image preparation."""
        client = GeminiClient("test-api-key")

        # Create a simple test image
        test_image = tmp_path / "test.jpg"
        img = Image.new('RGB', (10, 10), color='red')
        img.save(test_image)

        result = client.prepare_media_for_analysis(str(test_image), "https://example.com/image.jpg")

        assert result is not None
        assert isinstance(result, Image.Image)

    def test_prepare_media_image_corrupted(self, tmp_path):
        """Test handling of corrupted image file."""
        client = GeminiClient("test-api-key")

        # Create file with invalid image data
        corrupted_file = tmp_path / "corrupted.jpg"
        corrupted_file.write_bytes(b"not an image")

        result = client.prepare_media_for_analysis(str(corrupted_file), "https://example.com/image.jpg")

        # Should fallback to file path
        assert result == str(corrupted_file)

    def test_prepare_media_video_file(self, tmp_path):
        """Test video file preparation."""
        client = GeminiClient("test-api-key")

        # Create a dummy video file
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video content")

        result = client.prepare_media_for_analysis(str(video_file), "https://example.com/video.mp4")

        assert result == str(video_file)

    def test_prepare_media_unsupported_format(self, tmp_path):
        """Test unsupported media format."""
        client = GeminiClient("test-api-key")

        # Create a file with unsupported extension
        unsupported_file = tmp_path / "test.xyz"
        unsupported_file.write_bytes(b"unsupported content")

        result = client.prepare_media_for_analysis(str(unsupported_file), "https://example.com/file.xyz")

        assert result == str(unsupported_file)

    def test_prepare_media_pil_import_error(self, tmp_path, monkeypatch):
        """Test handling when PIL import fails."""
        client = GeminiClient("test-api-key")

        # Mock PIL import to fail
        def mock_import(name, *args, **kwargs):
            if name == 'PIL':
                raise ImportError("PIL not available")
            return __import__(name, *args, **kwargs)

        monkeypatch.setattr('builtins.__import__', mock_import)

        # Create a test image file
        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"fake image content")

        result = client.prepare_media_for_analysis(str(test_file), "https://example.com/image.jpg")

        # Should fallback to file path
        assert result == str(test_file)


class TestGenerateContent:
    """Test content generation functionality."""

    def test_generate_content_success(self):
        """Test successful content generation."""
        client = GeminiClient("test-api-key")
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Generated content"
        mock_model.generate_content.return_value = mock_response

        result = client.generate_content(mock_model, "Test prompt")

        assert result == "Generated content"
        mock_model.generate_content.assert_called_once_with("Test prompt")

    def test_generate_content_empty_response(self):
        """Test handling of empty response."""
        client = GeminiClient("test-api-key")
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = ""
        mock_model.generate_content.return_value = mock_response

        result = client.generate_content(mock_model, "Test prompt")

        assert result is None

    def test_generate_content_none_response(self):
        """Test handling of None response."""
        client = GeminiClient("test-api-key")
        mock_model = Mock()
        mock_model.generate_content.return_value = None

        result = client.generate_content(mock_model, "Test prompt")

        assert result is None

    def test_generate_content_timeout(self):
        """Test content generation timeout."""
        client = GeminiClient("test-api-key")
        mock_model = Mock()

        # Mock generate_content to never return (simulate timeout)
        mock_model.generate_content.side_effect = Exception("Timeout")

        result = client.generate_content(mock_model, "Test prompt", timeout=0.001)

        assert result is None

    def test_generate_content_quota_error(self):
        """Test content generation with quota error."""
        client = GeminiClient("test-api-key")
        mock_model = Mock()

        # Mock quota exceeded error
        mock_model.generate_content.side_effect = Exception("Quota exceeded")

        with pytest.raises(Exception, match="Quota exceeded"):
            client.generate_content(mock_model, "Test prompt")

    def test_generate_content_generic_error(self):
        """Test content generation with generic error."""
        client = GeminiClient("test-api-key")
        mock_model = Mock()

        # Mock generic API error
        mock_model.generate_content.side_effect = Exception("API Error")

        result = client.generate_content(mock_model, "Test prompt")

        assert result is None

    def test_generate_content_with_media(self):
        """Test content generation with media content."""
        client = GeminiClient("test-api-key")
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Media analysis result"
        mock_model.generate_content.return_value = mock_response

        # Test with list content (media + text)
        media_content = [Mock(), "Analyze this image"]
        result = client.generate_content(mock_model, media_content)

        assert result == "Media analysis result"
        mock_model.generate_content.assert_called_once_with(media_content)