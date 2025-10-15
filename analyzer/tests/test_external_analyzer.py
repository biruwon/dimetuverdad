"""
Unit tests for ExternalAnalyzer component.
Tests Gemini multimodal analysis integration.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch, MagicMock

from analyzer.external_analyzer import ExternalAnalyzer


@pytest.fixture
def mock_gemini():
    """Mock GeminiMultimodal instance."""
    mock = Mock()
    mock.analyze_multimodal_content = Mock(return_value=(
        "CATEGORÍA: hate_speech\nEXPLICACIÓN: Este contenido muestra discurso de odio xenófobo.",
        0.5  # analysis_time
    ))
    mock._select_media_url = Mock(return_value="https://example.com/image.jpg")
    return mock


@pytest.fixture
def mock_genai():
    """Mock google.generativeai module."""
    with patch('google.generativeai') as mock_genai:
        # Mock the GenerativeModel
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "CATEGORÍA: hate_speech\nEXPLICACIÓN: Este contenido muestra discurso de odio xenófobo."
        mock_model.generate_content.return_value = mock_response
        
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.configure = Mock()
        yield mock_genai


@pytest.fixture
def analyzer(mock_gemini):
    """Create ExternalAnalyzer with mocked Gemini."""
    with patch('analyzer.external_analyzer.GeminiMultimodal', return_value=mock_gemini):
        analyzer = ExternalAnalyzer(verbose=False)
        return analyzer


@pytest.fixture
def analyzer_with_genai_mock(mock_gemini):
    """Create ExternalAnalyzer with both GeminiMultimodal and genai mocked."""
    # Mock the analyze_multimodal_content method to return text-only results
    mock_gemini.analyze_multimodal_content.return_value = (
        "CATEGORÍA: hate_speech\nEXPLICACIÓN: Este contenido muestra discurso de odio xenófobo.",
        0.5
    )
    
    with patch('analyzer.external_analyzer.GeminiMultimodal', return_value=mock_gemini):
        analyzer = ExternalAnalyzer(verbose=False)
        yield analyzer


class TestExternalAnalyzerInitialization:
    """Test analyzer initialization."""
    
    def test_default_initialization(self):
        """Test default initialization with Gemini."""
        with patch('analyzer.external_analyzer.GeminiMultimodal'):
            analyzer = ExternalAnalyzer()
            
            assert analyzer.verbose is False
            assert analyzer.gemini is not None
    
    def test_verbose_initialization(self):
        """Test initialization with verbose logging."""
        with patch('analyzer.external_analyzer.GeminiMultimodal'):
            analyzer = ExternalAnalyzer(verbose=True)
            
            assert analyzer.verbose is True


class TestAnalyzeMethod:
    """Test main analyze method."""
    
    @pytest.mark.asyncio
    async def test_analyze_with_media(self, analyzer, mock_gemini):
        """Test analysis with media URLs triggers multimodal analysis."""
        content = "Test content"
        media_urls = ["https://example.com/image1.jpg", "https://example.com/image2.jpg"]
        
        result = await analyzer.analyze(content, media_urls)
        
        assert result == "Este contenido muestra discurso de odio xenófobo."
        
        # Verify multimodal analysis was called
        mock_gemini.analyze_multimodal_content.assert_called_once()
        # Verify media URL selection
        mock_gemini._select_media_url.assert_called_once_with(media_urls)
    
    @pytest.mark.asyncio
    async def test_analyze_text_only(self, analyzer_with_genai_mock, mock_gemini):
        """Test text-only analysis (no media)."""
        content = "Test content without media"
        
        result = await analyzer_with_genai_mock.analyze(content, media_urls=None)
        
        assert result == "Este contenido muestra discurso de odio xenófobo."
        
        # Verify analyze_multimodal_content was called with empty media list
        mock_gemini.analyze_multimodal_content.assert_called_once()
        call_args = mock_gemini.analyze_multimodal_content.call_args
        assert call_args[0][0] == []  # Empty media list
        assert call_args[0][1] == content
    
    @pytest.mark.asyncio
    async def test_analyze_with_empty_media_list(self, analyzer_with_genai_mock, mock_gemini):
        """Test analysis with empty media list treated as text-only."""
        content = "Test content"
        media_urls = []
        
        result = await analyzer_with_genai_mock.analyze(content, media_urls)
        
        # Empty list should trigger text-only analysis
        mock_gemini.analyze_multimodal_content.assert_called_once()
        call_args = mock_gemini.analyze_multimodal_content.call_args
        assert call_args[0][0] == []
    
    @pytest.mark.asyncio
    async def test_analyze_error_handling(self, analyzer_with_genai_mock, mock_gemini):
        """Test error handling in analyze method."""
        # Make analyze_multimodal_content return None (failure)
        mock_gemini.analyze_multimodal_content.return_value = (None, 0.5)
        
        content = "Test content"
        result = await analyzer_with_genai_mock.analyze(content)
        
        assert "No se pudo completar el análisis de texto externo." in result


class TestMultimodalAnalysis:
    """Test multimodal analysis flow."""
    
    @pytest.mark.asyncio
    async def test_multimodal_with_single_media(self, analyzer, mock_gemini):
        """Test multimodal analysis with single media URL."""
        content = "Test image content"
        media_urls = ["https://example.com/image.jpg"]
        
        result = await analyzer._analyze_multimodal(content, media_urls)
        
        assert "discurso de odio xenófobo" in result
        
        # Verify media selection was called
        mock_gemini._select_media_url.assert_called_once_with(media_urls)
        
        # Verify Gemini was called with selected media
        call_args = mock_gemini.analyze_multimodal_content.call_args
        assert call_args[0][0] == ["https://example.com/image.jpg"]
        assert call_args[0][1] == content
    
    @pytest.mark.asyncio
    async def test_multimodal_with_multiple_media(self, analyzer, mock_gemini):
        """Test multimodal analysis with multiple media URLs."""
        mock_gemini._select_media_url.return_value = "https://example.com/image2.jpg"
        
        content = "Test content"
        media_urls = [
            "https://example.com/image1.jpg",
            "https://example.com/image2.jpg",
            "https://example.com/image3.jpg"
        ]
        
        result = await analyzer._analyze_multimodal(content, media_urls)
        
        # Verify best media was selected
        mock_gemini._select_media_url.assert_called_once_with(media_urls)
        
        # Verify selected media was used
        call_args = mock_gemini.analyze_multimodal_content.call_args
        assert call_args[0][0] == ["https://example.com/image2.jpg"]
    
    @pytest.mark.asyncio
    async def test_multimodal_null_response(self, analyzer, mock_gemini):
        """Test handling of null response from Gemini."""
        mock_gemini.analyze_multimodal_content.return_value = (None, 0)
        
        content = "Test content"
        media_urls = ["https://example.com/image.jpg"]
        
        result = await analyzer._analyze_multimodal(content, media_urls)
        
        assert result == "No se pudo completar el análisis multimodal externo."


class TestTextOnlyAnalysis:
    """Test text-only analysis flow."""
    
    @pytest.mark.asyncio
    async def test_text_only_success(self, analyzer_with_genai_mock, mock_gemini):
        """Test successful text-only analysis."""
        content = "Pure text content"
        
        result = await analyzer_with_genai_mock._analyze_text_only(content)
        
        assert "discurso de odio xenófobo" in result
        
        # Verify analyze_multimodal_content was called with empty media list
        mock_gemini.analyze_multimodal_content.assert_called_once()
        call_args = mock_gemini.analyze_multimodal_content.call_args
        assert call_args[0][0] == []
        assert call_args[0][1] == content
    
    @pytest.mark.asyncio
    async def test_text_only_null_response(self, analyzer_with_genai_mock, mock_gemini):
        """Test handling of null response in text-only analysis."""
        # Make analyze_multimodal_content return None
        mock_gemini.analyze_multimodal_content.return_value = (None, 0.5)
        
        content = "Test content"
        result = await analyzer_with_genai_mock._analyze_text_only(content)
        
        assert result == "No se pudo completar el análisis de texto externo."


class TestResponseParsing:
    """Test parsing of Gemini responses."""
    
    def test_parse_structured_response(self, analyzer):
        """Test parsing structured Gemini response."""
        response = """CATEGORÍA: disinformation
EXPLICACIÓN: Este contenido presenta información falsa sobre vacunas."""
        
        result = analyzer._parse_gemini_response(response)
        
        assert result == "Este contenido presenta información falsa sobre vacunas."
    
    def test_parse_response_with_extra_lines(self, analyzer):
        """Test parsing response with extra content."""
        response = """CATEGORÍA: conspiracy_theory
EXPLICACIÓN: El contenido promueve teorías conspirativas sin evidencia.
Additional context or metadata."""
        
        result = analyzer._parse_gemini_response(response)
        
        assert "teorías conspirativas sin evidencia" in result
    
    def test_parse_unstructured_response(self, analyzer):
        """Test parsing unstructured response."""
        response = "Este es un análisis directo sin formato estructurado."
        
        result = analyzer._parse_gemini_response(response)
        
        assert result == "Este es un análisis directo sin formato estructurado."
    
    def test_parse_response_removes_category_line(self, analyzer):
        """Test parsing removes category line from unstructured response."""
        response = """CATEGORÍA: hate_speech
Este contenido muestra ataques xenófobos directos.
Incluye lenguaje deshumanizante."""
        
        result = analyzer._parse_gemini_response(response)
        
        # Should remove category line and return rest
        assert "CATEGORÍA:" not in result
        assert "xenófobos" in result
    
    def test_parse_empty_response(self, analyzer):
        """Test parsing empty or whitespace-only response."""
        result = analyzer._parse_gemini_response("")
        assert result == "Análisis externo no disponible."
        
        result = analyzer._parse_gemini_response("   \n   ")
        assert result == "Análisis externo no disponible."
    
    def test_parse_short_response(self, analyzer):
        """Test parsing very short response (less than 10 chars)."""
        response = "CATEGORÍA: general\nEXPLICACIÓN: OK"
        
        result = analyzer._parse_gemini_response(response)
        
        assert result == "Análisis externo completado sin detalles adicionales."


class TestVerboseLogging:
    """Test verbose logging output."""
    
    @pytest.mark.asyncio
    async def test_analyze_verbose_output(self, mock_gemini, capsys):
        """Test verbose logging in analyze method."""
        with patch('analyzer.external_analyzer.GeminiMultimodal', return_value=mock_gemini):
            analyzer = ExternalAnalyzer(verbose=True)
            
            await analyzer.analyze("Test content", media_urls=["https://example.com/image.jpg"])
            
            captured = capsys.readouterr()
            assert "🌐 Running external analysis (Gemini)" in captured.out
            assert "📝 Content:" in captured.out
            assert "🖼️  Media:" in captured.out
            assert "✅ External analysis complete:" in captured.out
    
    @pytest.mark.asyncio
    async def test_analyze_verbose_without_media(self, mock_gemini, capsys):
        """Test verbose logging without media."""
        with patch('analyzer.external_analyzer.GeminiMultimodal', return_value=mock_gemini):
            analyzer = ExternalAnalyzer(verbose=True)
            
            await analyzer.analyze("Test content")
            
            captured = capsys.readouterr()
            assert "🌐 Running external analysis (Gemini)" in captured.out
            # Should not mention media when none provided
            assert "🖼️  Media:" not in captured.out


class TestAsyncThreading:
    """Test async/threading integration."""
    
    @pytest.mark.asyncio
    async def test_multimodal_uses_threading(self, analyzer, mock_gemini):
        """Test that Gemini calls run in thread pool."""
        content = "Test content"
        media_urls = ["https://example.com/image.jpg"]
        
        # Create a mock that tracks whether it was called
        call_tracker = []
        
        def track_call(*args, **kwargs):
            call_tracker.append("called")
            return ("Result", 0.5)
        
        mock_gemini.analyze_multimodal_content = Mock(side_effect=track_call)
        
        with patch('asyncio.to_thread', wraps=asyncio.to_thread) as mock_to_thread:
            result = await analyzer._analyze_multimodal(content, media_urls)
            
            # Verify to_thread was used
            assert mock_to_thread.called
            
            # Verify Gemini was called
            assert len(call_tracker) == 1
