"""
Tests for OllamaAnalyzer - high-level analysis with media preparation and parsing.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from analyzer.ollama_analyzer import OllamaAnalyzer
from analyzer.categories import Categories


class TestOllamaAnalyzerInitialization:
    """Test OllamaAnalyzer initialization."""
    
    def test_default_initialization(self):
        """Test analyzer initializes with default model."""
        analyzer = OllamaAnalyzer()
        assert analyzer.model == "gemma3:27b-it-q4_K_M"
        assert analyzer.verbose is False
        assert analyzer.client is not None
    
    def test_custom_model_initialization(self):
        """Test analyzer initializes with custom model."""
        analyzer = OllamaAnalyzer(model="gpt-oss:20b", verbose=True)
        assert analyzer.model == "gpt-oss:20b"
        assert analyzer.verbose is True


class TestMediaPreparation:
    """Test media preparation functionality."""
    
    @pytest.mark.asyncio
    async def test_prepare_image_success(self):
        """Test successful image preparation."""
        analyzer = OllamaAnalyzer()
        
        mock_response = Mock()
        mock_response.content = b"image_data"
        mock_response.headers = {'content-length': '1024'}
        mock_response.raise_for_status.return_value = None
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client_class.return_value.__aexit__.return_value = None
            mock_client.get.return_value = mock_response
            
            result = await analyzer._prepare_media_content(["http://example.com/image.jpg"])
            
            assert len(result) == 1
            assert result[0]["type"] == "image_url"
            assert "url" in result[0]["image_url"]
    
    @pytest.mark.asyncio
    async def test_prepare_media_skips_videos(self):
        """Test that video URLs are skipped."""
        analyzer = OllamaAnalyzer()
        
        result = await analyzer._prepare_media_content([
            "http://example.com/video.mp4",
            "http://example.com/video.m3u8"
        ])
        
        assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_prepare_media_handles_thumbnails(self):
        """Test that video thumbnails with image extensions are processed."""
        analyzer = OllamaAnalyzer()
        
        mock_response = Mock()
        mock_response.content = b"thumbnail_data"
        mock_response.headers = {'content-length': '512'}
        mock_response.raise_for_status.return_value = None
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client_class.return_value.__aexit__.return_value = None
            mock_client.get.return_value = mock_response
            
            result = await analyzer._prepare_media_content([
                "http://example.com/amplify_video_thumb/image.jpg"
            ])
            
            assert len(result) == 1


class TestHasOnlyVideos:
    """Test video detection logic."""
    
    def test_has_only_videos_returns_true_for_videos_only(self):
        """Test detection of video-only content."""
        analyzer = OllamaAnalyzer()
        
        result = analyzer._has_only_videos([
            "http://example.com/video.mp4",
            "http://example.com/stream.m3u8"
        ])
        
        assert result is True
    
    def test_has_only_videos_returns_false_with_images(self):
        """Test detection returns false when images are present."""
        analyzer = OllamaAnalyzer()
        
        result = analyzer._has_only_videos([
            "http://example.com/video.mp4",
            "http://example.com/image.jpg"
        ])
        
        assert result is False
    
    def test_has_only_videos_handles_thumbnails(self):
        """Test that video thumbnails are recognized as images."""
        analyzer = OllamaAnalyzer()
        
        result = analyzer._has_only_videos([
            "http://example.com/amplify_video_thumb/image.jpg"
        ])
        
        assert result is False
