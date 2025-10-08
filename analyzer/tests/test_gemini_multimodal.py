#!/usr/bin/env python3
"""
Unit tests for Gemini multimodal analysis functionality.
"""

import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock
from analyzer.gemini_multimodal import (
    download_media_to_temp_file,
    _get_gemini_client,
    _upload_media_to_gemini,
    analyze_multimodal_content,
    extract_media_type
)
from analyzer.prompts import EnhancedPromptGenerator


class TestGeminiMultimodal(unittest.TestCase):
    """Test cases for Gemini multimodal analysis functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_image_url = "https://pbs.twimg.com/card_img/1973243448871284736/bPeZHL3l?format=jpg&name=small"
        self.test_video_url = "https://video.twimg.com/amplify_video/1972307252796141568/vid/avc1/320x568/GftH9VZYZuygizQc.mp4"
        self.test_text = "A juicio el hermano, la mujer, el fiscal y dos secretarios de organización del presidente."

    def test_extract_media_type_image(self):
        """Test media type extraction for images."""
        media_urls = [self.test_image_url]
        result = extract_media_type(media_urls)
        self.assertEqual(result, "image")

    def test_extract_media_type_video(self):
        """Test media type extraction for videos."""
        media_urls = [self.test_video_url]
        result = extract_media_type(media_urls)
        self.assertEqual(result, "video")

    def test_extract_media_type_mixed(self):
        """Test media type extraction for mixed media."""
        media_urls = [self.test_image_url, self.test_video_url]
        result = extract_media_type(media_urls)
        self.assertEqual(result, "mixed")

    def test_extract_media_type_empty(self):
        """Test media type extraction for empty list."""
        media_urls = []
        result = extract_media_type(media_urls)
        self.assertEqual(result, "")

    def test_extract_media_type_unknown(self):
        """Test media type extraction for unknown URLs."""
        media_urls = ["https://example.com/file.unknown"]
        result = extract_media_type(media_urls)
        self.assertEqual(result, "unknown")

    def test_create_analysis_prompt_image(self):
        """Test analysis prompt creation for images."""
        prompt = EnhancedPromptGenerator.build_gemini_analysis_prompt(self.test_text, is_video=False)
        self.assertIn("imagen", prompt)
        self.assertIn("Descripción detallada del contenido visual de la imagen", prompt)
        self.assertIn(self.test_text, prompt)

    def test_create_analysis_prompt_video(self):
        """Test analysis prompt creation for videos."""
        prompt = EnhancedPromptGenerator.build_gemini_analysis_prompt(self.test_text, is_video=True)
        self.assertIn("video", prompt)
        self.assertIn("Resumen del contenido visual del video", prompt)
        self.assertIn(self.test_text, prompt)

    @patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'}, clear=True)
    @patch('analyzer.gemini_multimodal.genai.configure')
    @patch('analyzer.gemini_multimodal.genai.GenerativeModel')
    def test_get_gemini_client_success(self, mock_model_class, mock_configure):
        """Test successful Gemini client creation."""
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        result = _get_gemini_client()
        self.assertEqual(result, mock_model)
        # Don't assert exact API key - just verify configure was called
        mock_configure.assert_called_once()
        mock_model_class.assert_called_once_with('gemini-1.5-flash')

    @patch.dict(os.environ, {}, clear=True)
    def test_get_gemini_client_no_key(self):
        """Test Gemini client creation without API key."""
        result = _get_gemini_client()
        self.assertIsNone(result)

    @patch('analyzer.gemini_multimodal.requests.get')
    def test_download_media_success(self, mock_get):
        """Test successful media download."""
        # Mock response with proper status_code attribute
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {'content-type': 'image/jpeg'}
        mock_response.iter_content.return_value = [b'test', b'data']
        mock_get.return_value = mock_response

        result = download_media_to_temp_file("https://example.com/image.jpg")

        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(result))

        # Clean up
        if result and os.path.exists(result):
            os.unlink(result)

    @patch('analyzer.gemini_multimodal.requests.get')
    def test_download_media_failure(self, mock_get):
        """Test media download failure."""
        mock_get.side_effect = Exception("Network error")

        result = download_media_to_temp_file("https://example.com/image.jpg")
        self.assertIsNone(result)

    @patch('analyzer.gemini_multimodal._get_gemini_client')
    @patch('analyzer.gemini_multimodal.download_media_to_temp_file')
    @patch('analyzer.gemini_multimodal._upload_media_to_gemini')
    def test_analyze_multimodal_success(self, mock_upload, mock_download, mock_get_client):
        """Test successful multimodal analysis."""
        # Setup mocks
        mock_model = MagicMock()
        mock_get_client.return_value = mock_model

        mock_download.return_value = "/tmp/test.jpg"

        mock_file = MagicMock()
        mock_upload.return_value = mock_file

        # Mock the generate_content response properly
        mock_response = MagicMock()
        mock_response.text = "Test analysis result"
        mock_model.generate_content.return_value = mock_response

        # Test
        result, time_taken = analyze_multimodal_content([self.test_image_url], self.test_text)

        self.assertEqual(result, "Test analysis result")
        self.assertIsInstance(time_taken, float)
        self.assertGreater(time_taken, 0)

    @patch('analyzer.gemini_multimodal._get_gemini_client')
    def test_analyze_multimodal_no_urls(self, mock_get_client):
        """Test multimodal analysis with no URLs."""
        result, time_taken = analyze_multimodal_content([], self.test_text)

        self.assertIsNone(result)
        self.assertEqual(time_taken, 0.0)
        mock_get_client.assert_not_called()

    @patch('analyzer.gemini_multimodal._get_gemini_client')
    def test_analyze_multimodal_client_failure(self, mock_get_client):
        """Test multimodal analysis when client creation fails."""
        mock_get_client.return_value = None

        result, time_taken = analyze_multimodal_content([self.test_image_url], self.test_text)

        self.assertIsNone(result)
        self.assertIsInstance(time_taken, float)


if __name__ == '__main__':
    unittest.main()