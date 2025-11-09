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


class TestCategorizeAndExplain:
    """Test categorize_and_explain method."""
    
    @pytest.mark.asyncio
    async def test_text_only_analysis(self):
        """Test text-only analysis."""
        analyzer = OllamaAnalyzer()
        
        with patch.object(analyzer.client, 'generate_text', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "CATEGORÍA: hate_speech\nEXPLICACIÓN: Este es un análisis de prueba."
            
            category, explanation = await analyzer.categorize_and_explain("Test content")
            
            assert category == Categories.HATE_SPEECH
            assert "análisis de prueba" in explanation
            mock_gen.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_multimodal_analysis_with_images(self):
        """Test multimodal analysis with images."""
        analyzer = OllamaAnalyzer()
        
        with patch.object(analyzer, '_prepare_media_content', new_callable=AsyncMock) as mock_prepare:
            with patch.object(analyzer.client, 'generate_multimodal', new_callable=AsyncMock) as mock_gen:
                mock_prepare.return_value = [{"type": "image_url", "image_url": {"url": "base64data"}}]
                mock_gen.return_value = "CATEGORÍA: anti_immigration\nEXPLICACIÓN: Contenido anti-inmigración."
                
                category, explanation = await analyzer.categorize_and_explain(
                    "Test content",
                    media_urls=["http://example.com/image.jpg"]
                )
                
                assert category == Categories.ANTI_IMMIGRATION
                assert "anti-inmigración" in explanation
                mock_prepare.assert_called_once()
                mock_gen.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_video_only_falls_back_to_text(self):
        """Test that video-only content falls back to text analysis."""
        analyzer = OllamaAnalyzer()
        
        with patch.object(analyzer, '_has_only_videos', return_value=True):
            with patch.object(analyzer.client, 'generate_text', new_callable=AsyncMock) as mock_gen:
                mock_gen.return_value = "CATEGORÍA: general\nEXPLICACIÓN: Análisis de texto."
                
                category, explanation = await analyzer.categorize_and_explain(
                    "Test content",
                    media_urls=["http://example.com/video.mp4"]
                )
                
                assert category == Categories.GENERAL
                mock_gen.assert_called_once()


class TestExplainOnly:
    """Test explain_only method."""
    
    @pytest.mark.asyncio
    async def test_text_only_explanation(self):
        """Test text-only explanation generation."""
        analyzer = OllamaAnalyzer()
        
        with patch.object(analyzer.client, 'generate_text', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "Esta es la explicación generada por el modelo."
            
            explanation = await analyzer.explain_only("Test content", Categories.HATE_SPEECH)
            
            assert "explicación generada" in explanation
            mock_gen.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_multimodal_explanation(self):
        """Test multimodal explanation generation."""
        analyzer = OllamaAnalyzer()
        
        with patch.object(analyzer, '_prepare_media_content', new_callable=AsyncMock) as mock_prepare:
            with patch.object(analyzer.client, 'generate_multimodal', new_callable=AsyncMock) as mock_gen:
                mock_prepare.return_value = [{"type": "image_url", "image_url": {"url": "base64data"}}]
                mock_gen.return_value = "Explicación multimodal del contenido."
                
                explanation = await analyzer.explain_only(
                    "Test content",
                    Categories.ANTI_IMMIGRATION,
                    media_urls=["http://example.com/image.jpg"]
                )
                
                assert "multimodal" in explanation
                mock_prepare.assert_called_once()


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


class TestResponseParsing:
    """Test response parsing logic."""
    
    def test_parse_structured_response(self):
        """Test parsing of properly formatted response."""
        analyzer = OllamaAnalyzer()
        
        response = "CATEGORÍA: hate_speech\nEXPLICACIÓN: Esta es la explicación."
        category, explanation = analyzer._parse_category_and_explanation(response)
        
        assert category == Categories.HATE_SPEECH
        assert explanation == "Esta es la explicación."
    
    def test_parse_response_case_insensitive(self):
        """Test parsing is case-insensitive."""
        analyzer = OllamaAnalyzer()
        
        response = "categoría: anti_immigration\nexplicación: Texto anti-inmigración."
        category, explanation = analyzer._parse_category_and_explanation(response)
        
        assert category == Categories.ANTI_IMMIGRATION
        assert "anti-inmigración" in explanation
    
    def test_parse_empty_response(self):
        """Test parsing of empty response returns default."""
        analyzer = OllamaAnalyzer()
        
        category, explanation = analyzer._parse_category_and_explanation("")
        
        assert category == Categories.GENERAL
        assert "empty response" in explanation.lower()
    
    def test_parse_unstructured_response(self):
        """Test parsing of response without proper format."""
        analyzer = OllamaAnalyzer()
        
        response = "This is just some random text without structure."
        category, explanation = analyzer._parse_category_and_explanation(response)
        
        assert category == Categories.GENERAL
        assert len(explanation) > 0
