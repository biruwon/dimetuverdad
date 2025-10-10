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
        self.test_image_url = "https://pbs.twimg.com/media/1973243448871284736/bPeZHL3l?format=jpg&name=small"
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
        self.assertIn("contenido político problemático en la imagen", prompt)
        self.assertIn(self.test_text, prompt)

    def test_create_analysis_prompt_video(self):
        """Test analysis prompt creation for videos."""
        prompt = EnhancedPromptGenerator.build_gemini_analysis_prompt(self.test_text, is_video=True)
        self.assertIn("video", prompt)
        self.assertIn("contenido político problemático en la video", prompt)
        self.assertIn(self.test_text, prompt)

    @patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'}, clear=True)
    @patch('analyzer.gemini_multimodal.genai.configure')
    @patch('analyzer.gemini_multimodal.genai.GenerativeModel')
    def test_get_gemini_client_success(self, mock_model_class, mock_configure):
        """Test successful Gemini client creation."""
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        result, error = _get_gemini_client()
        self.assertEqual(result, mock_model)
        self.assertIsNone(error)
        # Don't assert exact API key - just verify configure was called
        mock_configure.assert_called_once()
        mock_model_class.assert_called_once_with('gemini-2.0-flash-exp')

    @patch.dict(os.environ, {}, clear=True)
    def test_get_gemini_client_no_key(self):
        """Test Gemini client creation without API key."""
        result, error = _get_gemini_client()
        self.assertIsNone(result)
        self.assertIsNotNone(error)

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
        mock_get_client.return_value = (mock_model, None)

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
    @patch('analyzer.gemini_multimodal.download_media_to_temp_file')
    @patch('analyzer.gemini_multimodal._upload_media_to_gemini')
    def test_analyze_multimodal_video_detection_priority_mp4(self, mock_upload, mock_download, mock_get_client):
        """Test that MP4 files are prioritized over thumbnails in video detection."""
        # Setup mocks
        mock_model = MagicMock()
        mock_get_client.return_value = (mock_model, None)

        mock_download.return_value = "/tmp/test.mp4"

        mock_file = MagicMock()
        mock_upload.return_value = mock_file

        mock_response = MagicMock()
        mock_response.text = "Video analysis result"
        mock_model.generate_content.return_value = mock_response

        # Test URLs: thumbnail first, then MP4
        media_urls = [
            "https://pbs.twimg.com/amplify_video_thumb/123/img.jpg",  # thumbnail
            "https://video.twimg.com/test.mp4"  # actual video
        ]

        result, time_taken = analyze_multimodal_content(media_urls, self.test_text)

        # Should download the MP4 file, not the thumbnail
        mock_download.assert_called_once_with("https://video.twimg.com/test.mp4", True)
        self.assertEqual(result, "Video analysis result")

    @patch('analyzer.gemini_multimodal._get_gemini_client')
    @patch('analyzer.gemini_multimodal.download_media_to_temp_file')
    @patch('analyzer.gemini_multimodal._upload_media_to_gemini')
    def test_analyze_multimodal_video_detection_fallback_to_video_url(self, mock_upload, mock_download, mock_get_client):
        """Test fallback to video URLs when no MP4/M3U8 files exist."""
        # Setup mocks
        mock_model = MagicMock()
        mock_get_client.return_value = (mock_model, None)

        mock_download.return_value = "/tmp/test.jpg"

        mock_file = MagicMock()
        mock_upload.return_value = mock_file

        mock_response = MagicMock()
        mock_response.text = "Video analysis result"
        mock_model.generate_content.return_value = mock_response

        # Test URLs: only video URLs (no MP4), should pick first video URL
        media_urls = [
            "https://pbs.twimg.com/profile.jpg",  # profile image
            "https://video.twimg.com/amplify_video/123/vid/"  # video URL
        ]

        result, time_taken = analyze_multimodal_content(media_urls, self.test_text)

        # Should download the video URL
        mock_download.assert_called_once_with("https://video.twimg.com/amplify_video/123/vid/", True)
        self.assertEqual(result, "Video analysis result")

    @patch('analyzer.gemini_multimodal._get_gemini_client')
    @patch('analyzer.gemini_multimodal.download_media_to_temp_file')
    @patch('analyzer.gemini_multimodal._upload_media_to_gemini')
    def test_analyze_multimodal_image_fallback(self, mock_upload, mock_download, mock_get_client):
        """Test fallback to first URL when no video content detected."""
        # Setup mocks
        mock_model = MagicMock()
        mock_get_client.return_value = (mock_model, None)

        mock_download.return_value = "/tmp/test.jpg"

        mock_file = MagicMock()
        mock_upload.return_value = mock_file

        mock_response = MagicMock()
        mock_response.text = "Image analysis result"
        mock_model.generate_content.return_value = mock_response

        # Test URLs: only images
        media_urls = [
            "https://pbs.twimg.com/profile.jpg",
            "https://pbs.twimg.com/card_img.jpg"
        ]

        result, time_taken = analyze_multimodal_content(media_urls, self.test_text)

        # Should download the first URL and treat as image
        mock_download.assert_called_once_with("https://pbs.twimg.com/profile.jpg", False)
        self.assertEqual(result, "Image analysis result")

    @patch('analyzer.gemini_multimodal._get_gemini_client')
    @patch('analyzer.gemini_multimodal.download_media_to_temp_file')
    @patch('analyzer.gemini_multimodal._upload_media_to_gemini')
    def test_analyze_multimodal_m3u8_detection(self, mock_upload, mock_download, mock_get_client):
        """Test detection of M3U8 video files."""
        # Setup mocks
        mock_model = MagicMock()
        mock_get_client.return_value = (mock_model, None)

        mock_download.return_value = "/tmp/test.m3u8"

        mock_file = MagicMock()
        mock_upload.return_value = mock_file

        mock_response = MagicMock()
        mock_response.text = "M3U8 video analysis result"
        mock_model.generate_content.return_value = mock_response

        # Test URLs with M3U8 file
        media_urls = [
            "https://pbs.twimg.com/profile.jpg",
            "https://video.twimg.com/test.m3u8"
        ]

        result, time_taken = analyze_multimodal_content(media_urls, self.test_text)

        # Should prioritize M3U8 file and treat as video
        mock_download.assert_called_once_with("https://video.twimg.com/test.m3u8", True)
        self.assertEqual(result, "M3U8 video analysis result")


    @patch('analyzer.gemini_multimodal._get_gemini_client')
    @patch('analyzer.gemini_multimodal.download_media_to_temp_file')
    @patch('analyzer.gemini_multimodal._upload_media_to_gemini')
    def test_analyze_multimodal_video_priority_order(self, mock_upload, mock_download, mock_get_client):
        """Test video detection priority: MP4 > M3U8 > video URLs > thumbnails."""
        # Setup mocks
        mock_model = MagicMock()
        mock_get_client.return_value = (mock_model, None)

        mock_download.return_value = "/tmp/test.mp4"

        mock_file = MagicMock()
        mock_upload.return_value = mock_file

        mock_response = MagicMock()
        mock_response.text = "Priority video analysis result"
        mock_model.generate_content.return_value = mock_response

        # Test URLs in reverse priority order (thumbnail first, then MP4)
        media_urls = [
            "https://pbs.twimg.com/amplify_video_thumb/123/img.jpg",  # thumbnail
            "https://video.twimg.com/amplify_video/456/vid/",        # video URL
            "https://video.twimg.com/test.m3u8",                     # M3U8
            "https://video.twimg.com/test.mp4"                       # MP4 (highest priority)
        ]

        result, time_taken = analyze_multimodal_content(media_urls, self.test_text)

        # Should prioritize MP4 file over all others
        mock_download.assert_called_once_with("https://video.twimg.com/test.mp4", True)
        self.assertEqual(result, "Priority video analysis result")


    @patch('analyzer.gemini_multimodal._get_gemini_client')
    @patch('analyzer.gemini_multimodal.download_media_to_temp_file')
    @patch('analyzer.gemini_multimodal._upload_media_to_gemini')
    def test_analyze_multimodal_m3u8_over_video_url(self, mock_upload, mock_download, mock_get_client):
        """Test that M3U8 files are prioritized over generic video URLs."""
        # Setup mocks
        mock_model = MagicMock()
        mock_get_client.return_value = (mock_model, None)

        mock_download.return_value = "/tmp/test.m3u8"

        mock_file = MagicMock()
        mock_upload.return_value = mock_file

        mock_response = MagicMock()
        mock_response.text = "M3U8 priority result"
        mock_model.generate_content.return_value = mock_response

        # Test URLs: video URL first, then M3U8
        media_urls = [
            "https://video.twimg.com/amplify_video/123/vid/",  # video URL
            "https://video.twimg.com/test.m3u8"                 # M3U8 (higher priority)
        ]

        result, time_taken = analyze_multimodal_content(media_urls, self.test_text)

        # Should prioritize M3U8 over video URL
        mock_download.assert_called_once_with("https://video.twimg.com/test.m3u8", True)
        self.assertEqual(result, "M3U8 priority result")


    @patch('analyzer.gemini_multimodal._get_gemini_client')
    @patch('analyzer.gemini_multimodal.download_media_to_temp_file')
    @patch('analyzer.gemini_multimodal._upload_media_to_gemini')
    def test_analyze_multimodal_no_video_content(self, mock_upload, mock_download, mock_get_client):
        """Test that non-video URLs fall back to first available media."""
        # Setup mocks
        mock_model = MagicMock()
        mock_get_client.return_value = (mock_model, None)

        mock_download.return_value = "/tmp/test.jpg"

        mock_file = MagicMock()
        mock_upload.return_value = mock_file

        mock_response = MagicMock()
        mock_response.text = "Image fallback result"
        mock_model.generate_content.return_value = mock_response

        # Test URLs: only images, no video content
        media_urls = [
            "https://pbs.twimg.com/card_img/123.jpg",    # card image (filtered out)
            "https://pbs.twimg.com/profile_images/456.jpg", # profile image (filtered out)
            "https://pbs.twimg.com/media/789.jpg"        # regular media image (kept)
        ]

        result, time_taken = analyze_multimodal_content(media_urls, self.test_text)

        # Should download the regular media image and treat as image
        mock_download.assert_called_once_with("https://pbs.twimg.com/media/789.jpg", False)
        self.assertEqual(result, "Image fallback result")


    def test_extract_media_type_priority_logic(self):
        """Test media type extraction considers video priority."""
        # Pure video URLs should be detected as video
        video_urls = ["https://video.twimg.com/test.mp4"]
        result = extract_media_type(video_urls)
        self.assertEqual(result, "video")

        # Mixed content with both images and videos should be "mixed"
        mixed_urls = [
            "https://pbs.twimg.com/media/test.jpg",
            "https://video.twimg.com/test.mp4"
        ]
        result = extract_media_type(mixed_urls)
        self.assertEqual(result, "mixed")  # Mixed content returns "mixed"

        # Only images should be image
        image_urls = ["https://pbs.twimg.com/media/test.jpg"]
        result = extract_media_type(image_urls)
        self.assertEqual(result, "image")

        # Unknown URLs
        unknown_urls = ["https://example.com/file.unknown"]
        result = extract_media_type(unknown_urls)
        self.assertEqual(result, "unknown")


if __name__ == '__main__':
    unittest.main()