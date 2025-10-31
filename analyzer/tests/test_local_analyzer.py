"""
Unit tests for LocalMultimodalAnalyzer component.
Tests category detection and explanation generation using multiple Ollama models.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch

from analyzer.local_analyzer import LocalMultimodalAnalyzer
from analyzer.categories import Categories


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama AsyncClient for responses."""
    mock_client = AsyncMock()
    
    # Mock the generate method to return a dict with 'response' key
    mock_client.generate.return_value = {
        "response": "CATEGORÍA: hate_speech\nEXPLICACIÓN: Este contenido contiene discurso de odio xenófobo."
    }
    
    return mock_client


@pytest.fixture
def analyzer(mock_ollama_client):
    """Create LocalMultimodalAnalyzer with mocked Ollama client."""
    with patch('ollama.AsyncClient', return_value=mock_ollama_client):
        analyzer = LocalMultimodalAnalyzer(verbose=False)
        return analyzer


class TestLocalMultimodalAnalyzerInitialization:
    """Test analyzer initialization and configuration."""
    
    def test_default_initialization(self):
        """Test default initialization with Ollama."""
        with patch('ollama.AsyncClient'):
            analyzer = LocalMultimodalAnalyzer()
            
            assert analyzer.primary_model == "gemma3:4b"
            assert analyzer.verbose is False
            assert analyzer.prompt_generator is not None


class TestCategorizeAndExplain:
    """Test combined category detection and explanation generation."""
    
    @pytest.mark.asyncio
    async def test_successful_categorization(self, analyzer, mock_ollama_client):
        """Test successful category detection and explanation."""
        content = "Los inmigrantes destruyen nuestra nación"
        
        category, explanation = await analyzer.categorize_and_explain(content)
        
        assert category == Categories.HATE_SPEECH
        assert explanation == "Este contenido contiene discurso de odio xenófobo."
    
        # Verify Ollama was called
        mock_ollama_client.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_general_category_fallback(self, analyzer, mock_ollama_client):
        """Test fallback to general category when LLM returns unrecognized category."""
        # Mock response with unrecognized category
        mock_ollama_client.generate.return_value = {
            "response": "CATEGORÍA: unknown_category\nEXPLICACIÓN: Some explanation."
        }
        
        content = "Neutral political statement"
        category, explanation = await analyzer.categorize_and_explain(content)
        
        # Should fallback to general category when category not recognized
        assert category == Categories.GENERAL
        assert "Some explanation" in explanation or "Contenido clasificado como" in explanation
    
    @pytest.mark.asyncio
    async def test_error_handling(self, analyzer, mock_ollama_client):
        """Test error handling when Ollama fails."""
        mock_ollama_client.generate.side_effect = RuntimeError("Ollama connection failed")
        
        content = "Test content"
        
        # Should re-raise the exception instead of graceful fallback
        with pytest.raises(RuntimeError, match="Analysis failed: Ollama text generation failed"):
            await analyzer.categorize_and_explain(content)


class TestExplainOnly:
    """Test explanation generation for known categories."""
    
    @pytest.mark.asyncio
    async def test_successful_explanation(self, analyzer, mock_ollama_client):
        """Test generating explanation for known category."""
        # Mock response for explanation-only
        mock_ollama_client.generate.return_value = {
            "response": "Este contenido muestra retórica extremista con ataques xenófobos directos."
        }
        
        content = "Los inmigrantes son criminales"
        category = Categories.HATE_SPEECH
        
        explanation = await analyzer.explain_only(content, category)
        
        assert "extremista" in explanation or "xenófobos" in explanation
    
    @pytest.mark.asyncio
    async def test_explanation_error_handling(self, analyzer, mock_ollama_client):
        """Test error handling in explanation generation."""
        mock_ollama_client.generate.side_effect = RuntimeError("Generation failed")
        
        content = "Test content"
        
        # Should re-raise the exception instead of graceful fallback
        with pytest.raises(RuntimeError, match="Explanation generation failed: Generation failed"):
            await analyzer.explain_only(content, Categories.DISINFORMATION)


class TestPromptBuilding:
    """Test prompt generation methods."""
    
    def test_categorization_prompt_structure(self, analyzer):
        """Test categorization prompt contains all required elements."""
        content = "Test content for analysis"
        user_prompt = analyzer.prompt_generator.build_ollama_categorization_prompt(content)
        system_prompt = analyzer.prompt_generator.build_ollama_text_analysis_system_prompt()
        
        # User prompt should contain only the content (optimized structure)
        assert content in user_prompt
        assert "CONTENIDO A ANALIZAR:" in user_prompt
        
        # System prompt should contain all instructions
        assert "CATEGORÍAS:" in system_prompt
        assert "FORMATO OBLIGATORIO:" in system_prompt
        assert "CATEGORÍA:" in system_prompt
        assert "EXPLICACIÓN:" in system_prompt
        
        # Check all categories are listed in system prompt
        assert Categories.HATE_SPEECH in system_prompt
        assert Categories.DISINFORMATION in system_prompt
        assert Categories.CONSPIRACY_THEORY in system_prompt
        assert Categories.GENERAL in system_prompt
    
    def test_explanation_prompt_delegates(self, analyzer):
        """Test explanation prompt uses EnhancedPromptGenerator."""
        content = "Test content"
        category = Categories.ANTI_IMMIGRATION
        
        # Should delegate to prompt generator
        prompt = analyzer._build_explanation_prompt(content, category)
        
        # Basic validation - prompt should contain content and reference category
        assert isinstance(prompt, str)
        assert len(prompt) > 0


class TestResponseParsing:
    """Test parsing of LLM responses."""
    
    def test_parse_structured_response(self, analyzer):
        """Test parsing correctly structured response."""
        response = """CATEGORÍA: hate_speech
EXPLICACIÓN: Este contenido contiene discurso de odio con ataques xenófobos directos."""
        
        category, explanation = analyzer._parse_category_and_explanation(response)
        
        assert category == Categories.HATE_SPEECH
        assert explanation == "Este contenido contiene discurso de odio con ataques xenófobos directos."
    
    def test_parse_response_with_extra_whitespace(self, analyzer):
        """Test parsing response with extra whitespace and formatting."""
        response = """
        
        CATEGORÍA:   disinformation   
        
        EXPLICACIÓN:    Este contenido presenta desinformación verificable.   
        
        """
        
        category, explanation = analyzer._parse_category_and_explanation(response)
        
        assert category == Categories.DISINFORMATION
        assert "desinformación verificable" in explanation
    
    def test_parse_response_case_insensitive_category(self, analyzer):
        """Test category parsing is case-insensitive."""
        response = "CATEGORÍA: HATE_SPEECH\nEXPLICACIÓN: Test explanation."
        
        category, explanation = analyzer._parse_category_and_explanation(response)
        
        assert category == Categories.HATE_SPEECH
    
    def test_parse_response_with_invalid_category(self, analyzer):
        """Test parsing with invalid category defaults to general."""
        response = "CATEGORÍA: invalid_cat\nEXPLICACIÓN: Some explanation."
        
        category, explanation = analyzer._parse_category_and_explanation(response)
        
        assert category == Categories.GENERAL
    
    def test_parse_unstructured_response(self, analyzer):
        """Test parsing unstructured response falls back gracefully."""
        response = "This is just text without structure."
        
        category, explanation = analyzer._parse_category_and_explanation(response)
        
        assert category == Categories.GENERAL
        assert explanation == response
    
    def test_parse_empty_explanation(self, analyzer):
        """Test parsing with empty explanation gets fallback text."""
        response = "CATEGORÍA: conspiracy_theory\nEXPLICACIÓN: "
        
        category, explanation = analyzer._parse_category_and_explanation(response)
        
        assert category == Categories.CONSPIRACY_THEORY
        assert "Contenido clasificado como" in explanation


class TestOllamaGeneration:
    """Test Ollama API interaction."""
    
    @pytest.mark.asyncio
    async def test_generate_with_ollama_success(self, analyzer, mock_ollama_client):
        """Test successful Ollama generation."""
        prompt = "Test prompt"
    
        result = await analyzer._generate_with_ollama(prompt, "gemma3:4b")
    
        assert result == "CATEGORÍA: hate_speech\nEXPLICACIÓN: Este contenido contiene discurso de odio xenófobo."
    
    @pytest.mark.asyncio
    async def test_generate_with_ollama_error(self, analyzer, mock_ollama_client):
        """Test error handling in Ollama generation."""
        mock_ollama_client.generate.side_effect = Exception("Connection timeout")
        
        prompt = "Test prompt"
        
        with pytest.raises(RuntimeError, match="Ollama text generation failed"):
            await analyzer._generate_with_ollama(prompt, "gemma3:4b")


class TestVerboseLogging:
    """Test verbose logging output."""
    
    @pytest.mark.asyncio
    async def test_categorize_verbose_output(self, mock_ollama_client, capsys):
        """Test verbose logging in categorize_and_explain."""
        with patch('ollama.AsyncClient', return_value=mock_ollama_client):
            analyzer = LocalMultimodalAnalyzer(verbose=True)
            
            await analyzer.categorize_and_explain("Test content")
            
            captured = capsys.readouterr()
            assert "🔍 Running local text categorization + explanation" in captured.out
            assert "✅ Category detected:" in captured.out
    
    @pytest.mark.asyncio
    async def test_explain_verbose_output(self, mock_ollama_client, capsys):
        """Test verbose logging in explain_only."""
        with patch('ollama.AsyncClient', return_value=mock_ollama_client):
            analyzer = LocalMultimodalAnalyzer(verbose=True)
            
            await analyzer.explain_only("Test content", Categories.HATE_SPEECH)
            
            captured = capsys.readouterr()
            assert "🔍 Generating local text explanation for category:" in captured.out


class TestMultimodalSupport:
    """Test multimodal analysis capabilities."""
    
    @pytest.mark.asyncio
    async def test_text_fallback_when_no_media(self, analyzer):
        """Test that text-only analysis works when no media provided."""
        content = "Contenido político normal"
        
        category, explanation = await analyzer.categorize_and_explain(content)
        
        # Should use text analysis and get some result
        assert category in Categories.get_all_categories()
        assert isinstance(explanation, str)
        assert len(explanation) > 0
    
    @pytest.mark.asyncio
    async def test_explanation_without_media(self, analyzer):
        """Test explanation generation without media."""
        content = "Contenido de prueba"
        category = Categories.GENERAL
        
        explanation = await analyzer.explain_only(content, category)
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0


class TestErrorHandling:
    """Test error handling methods."""

    def test_handle_error_with_fallback(self):
        """Test _handle_error method with fallback value."""
        with patch('ollama.AsyncClient'):
            analyzer = LocalMultimodalAnalyzer(verbose=False)

            result = analyzer._handle_error("test operation", Exception("test error"), fallback_value="fallback")
            assert result == "fallback"

    def test_handle_error_without_fallback(self):
        """Test _handle_error method without fallback (should raise)."""
        with patch('ollama.AsyncClient'):
            analyzer = LocalMultimodalAnalyzer(verbose=False)

            with pytest.raises(RuntimeError, match="test operation failed"):
                analyzer._handle_error("test operation", Exception("test error"))

    def test_handle_error_verbose(self):
        """Test _handle_error method with verbose output."""
        with patch('ollama.AsyncClient'):
            analyzer = LocalMultimodalAnalyzer(verbose=True)

            with patch('builtins.print') as mock_print:
                result = analyzer._handle_error("test operation", Exception("test error"), fallback_value="fallback")
                assert result == "fallback"
                mock_print.assert_called_with("❌ test operation failed: test error")


class TestRetryLogic:
    """Test Ollama retry logic."""

    @pytest.mark.asyncio
    async def test_retry_ollama_call_success_first_attempt(self):
        """Test successful call on first attempt."""
        with patch('ollama.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.generate.return_value = {"response": "test response"}
            mock_client_class.return_value = mock_client

            analyzer = LocalMultimodalAnalyzer(verbose=False)

            result = await analyzer._retry_ollama_call("test op", mock_client.generate, model="test")

            assert result == "test response"
            assert mock_client.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_ollama_call_empty_response_retry(self):
        """Test retry on empty response."""
        with patch('ollama.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            # First call returns empty, second succeeds
            mock_client.generate.side_effect = [
                {"response": ""},
                {"response": "success response"}
            ]
            mock_client_class.return_value = mock_client

            analyzer = LocalMultimodalAnalyzer(verbose=False)

            with patch('asyncio.sleep') as mock_sleep:
                result = await analyzer._retry_ollama_call("test op", mock_client.generate, model="test")

                assert result == "success response"
                assert mock_client.generate.call_count == 2
                mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_ollama_call_max_retries_exhausted(self):
        """Test max retries exhausted."""
        with patch('ollama.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.generate.return_value = {"response": ""}  # Always empty
            mock_client_class.return_value = mock_client

            analyzer = LocalMultimodalAnalyzer(verbose=False)

            from analyzer.local_analyzer import OllamaRetryError
            with pytest.raises(OllamaRetryError, match="returned empty response after 3 attempts"):
                await analyzer._retry_ollama_call("test op", mock_client.generate, model="test")

    @pytest.mark.asyncio
    async def test_retry_ollama_call_network_error_retry(self):
        """Test retry on network error."""
        with patch('ollama.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            # First call fails with network error, second succeeds
            mock_client.generate.side_effect = [
                Exception("Connection timeout"),
                {"response": "success response"}
            ]
            mock_client_class.return_value = mock_client

            analyzer = LocalMultimodalAnalyzer(verbose=False)

            with patch('asyncio.sleep') as mock_sleep:
                result = await analyzer._retry_ollama_call("test op", mock_client.generate, model="test")

                assert result == "success response"
                assert mock_client.generate.call_count == 2
                mock_sleep.assert_called_once()


class TestRetryableErrorDetection:
    """Test retryable error detection."""

    def test_is_retryable_ollama_error_empty_response(self):
        """Test OllamaEmptyResponseError is retryable."""
        with patch('ollama.AsyncClient'):
            analyzer = LocalMultimodalAnalyzer(verbose=False)

            from analyzer.local_analyzer import OllamaEmptyResponseError
            assert analyzer._is_retryable_ollama_error(OllamaEmptyResponseError("empty")) is True

    def test_is_retryable_ollama_error_status_codes(self):
        """Test HTTP status codes for retryable errors."""
        with patch('ollama.AsyncClient'):
            analyzer = LocalMultimodalAnalyzer(verbose=False)

            # 5xx errors should be retryable
            error_500 = Exception("Server error")
            error_500.status_code = 500
            assert analyzer._is_retryable_ollama_error(error_500) is True

            # 429 should be retryable
            error_429 = Exception("Rate limited")
            error_429.status_code = 429
            assert analyzer._is_retryable_ollama_error(error_429) is True

            # 408 should be retryable
            error_408 = Exception("Request timeout")
            error_408.status_code = 408
            assert analyzer._is_retryable_ollama_error(error_408) is True

            # 404 should not be retryable
            error_404 = Exception("Not found")
            error_404.status_code = 404
            assert analyzer._is_retryable_ollama_error(error_404) is False

    def test_is_retryable_ollama_error_message_patterns(self):
        """Test error message patterns for retryable errors."""
        with patch('ollama.AsyncClient'):
            analyzer = LocalMultimodalAnalyzer(verbose=False)

            retryable_messages = [
                "Connection failed",
                "Network timeout",
                "Server unavailable",
                "Temporary error",
                "Rate limit exceeded"
            ]

            for msg in retryable_messages:
                assert analyzer._is_retryable_ollama_error(Exception(msg)) is True

            non_retryable_messages = [
                "Invalid model",
                "Bad request",
                "Authentication failed"
            ]

            for msg in non_retryable_messages:
                assert analyzer._is_retryable_ollama_error(Exception(msg)) is False


class TestMediaContentPreparation:
    """Test media content preparation methods."""

    @pytest.mark.asyncio
    async def test_prepare_media_content_image_success(self):
        """Test successful image download and preparation."""
        with patch('ollama.AsyncClient'):
            analyzer = LocalMultimodalAnalyzer(verbose=False)

            # Mock successful response
            mock_response = Mock()
            mock_response.content = b"fake image data"
            mock_response.headers = {'content-length': '1000'}
            mock_response.raise_for_status.return_value = None

            with patch('requests.get', return_value=mock_response) as mock_get:
                media_urls = ["https://example.com/image.jpg"]
                result = await analyzer._prepare_media_content(media_urls)

                assert len(result) == 1
                assert result[0]["type"] == "image_url"
                assert "image_url" in result[0]
                assert "url" in result[0]["image_url"]
                mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_prepare_media_content_skip_videos(self):
        """Test that videos are skipped."""
        with patch('ollama.AsyncClient'):
            analyzer = LocalMultimodalAnalyzer(verbose=False)

            media_urls = ["https://example.com/video.mp4", "https://example.com/image.jpg"]
            result = await analyzer._prepare_media_content(media_urls)

            # Should skip the video and only process the image
            assert len(result) == 0  # No valid images in this test

    @pytest.mark.asyncio
    async def test_prepare_media_content_large_file_skip(self):
        """Test skipping large files."""
        with patch('ollama.AsyncClient'):
            analyzer = LocalMultimodalAnalyzer(verbose=False)

            # Mock response with large content-length
            mock_response = Mock()
            mock_response.headers = {'content-length': str(20 * 1024 * 1024)}  # 20MB
            mock_response.raise_for_status.return_value = None

            with patch('requests.get', return_value=mock_response):
                media_urls = ["https://example.com/large.jpg"]
                result = await analyzer._prepare_media_content(media_urls)

                assert len(result) == 0  # Should skip large file

    @pytest.mark.asyncio
    async def test_prepare_media_content_timeout(self):
        """Test handling of download timeouts."""
        with patch('ollama.AsyncClient'):
            analyzer = LocalMultimodalAnalyzer(verbose=False)

            with patch('requests.get', side_effect=Exception("Timeout")):
                media_urls = ["https://example.com/image.jpg"]
                result = await analyzer._prepare_media_content(media_urls)

                assert len(result) == 0  # Should handle timeout gracefully

    @pytest.mark.asyncio
    async def test_prepare_media_content_limit_to_three(self):
        """Test that only first 3 media URLs are processed."""
        with patch('ollama.AsyncClient'):
            analyzer = LocalMultimodalAnalyzer(verbose=False)

            # Mock successful response
            mock_response = Mock()
            mock_response.content = b"fake image data"
            mock_response.headers = {'content-length': '1000'}
            mock_response.raise_for_status.return_value = None

            with patch('requests.get', return_value=mock_response) as mock_get:
                media_urls = ["https://example.com/1.jpg", "https://example.com/2.jpg",
                             "https://example.com/3.jpg", "https://example.com/4.jpg"]
                result = await analyzer._prepare_media_content(media_urls)

                assert len(result) == 3  # Should limit to 3
                assert mock_get.call_count == 3

    @pytest.mark.asyncio
    async def test_prepare_media_content_twitter_format_urls(self):
        """Test that Twitter URLs with format=jpg are properly handled."""
        with patch('ollama.AsyncClient'):
            analyzer = LocalMultimodalAnalyzer(verbose=False)

            # Mock successful response
            mock_response = Mock()
            mock_response.content = b"fake image data"
            mock_response.headers = {'content-length': '1000'}
            mock_response.raise_for_status.return_value = None

            with patch('requests.get', return_value=mock_response) as mock_get:
                # Test Twitter-style URLs with format parameter
                twitter_urls = [
                    "https://pbs.twimg.com/media/G3eSgyyXYAAWpzY?format=jpg&name=large",
                    "https://pbs.twimg.com/media/ABC123?format=png&name=small"
                ]
                result = await analyzer._prepare_media_content(twitter_urls)

                assert len(result) == 2  # Both should be processed
                assert result[0]["type"] == "image_url"
                assert result[1]["type"] == "image_url"
                assert mock_get.call_count == 2


class TestMultimodalGeneration:
    """Test multimodal generation methods."""

    @pytest.mark.asyncio
    async def test_generate_multimodal_with_ollama_success(self):
        """Test successful multimodal generation."""
        with patch('ollama.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.generate.return_value = {"response": "multimodal response"}
            mock_client_class.return_value = mock_client

            analyzer = LocalMultimodalAnalyzer(verbose=False)

            media_content = [{"type": "image_url", "image_url": {"url": "base64data"}}]
            result = await analyzer._generate_multimodal_with_ollama("test prompt", media_content, "gemma3:4b")

            assert result == "multimodal response"
            mock_client.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_multimodal_no_valid_images(self):
        """Test multimodal generation with no valid images."""
        with patch('ollama.AsyncClient'):
            analyzer = LocalMultimodalAnalyzer(verbose=False)

            media_content = [{"type": "invalid"}]
            with pytest.raises(RuntimeError, match="No valid images found"):
                await analyzer._generate_multimodal_with_ollama("test prompt", media_content, "gemma3:4b")


class TestVideoDetection:
    """Test video detection logic."""

    def test_has_only_videos_true(self):
        """Test detection of video-only content."""
        with patch('ollama.AsyncClient'):
            analyzer = LocalMultimodalAnalyzer(verbose=False)

            video_urls = ["https://example.com/video.mp4", "https://example.com/stream.m3u8"]
            assert analyzer._has_only_videos(video_urls) is True

    def test_has_only_videos_false_with_images(self):
        """Test detection when images are present."""
        with patch('ollama.AsyncClient'):
            analyzer = LocalMultimodalAnalyzer(verbose=False)

            mixed_urls = ["https://example.com/video.mp4", "https://example.com/image.jpg"]
            assert analyzer._has_only_videos(mixed_urls) is False

    def test_has_only_videos_empty_list(self):
        """Test empty media URL list."""
        with patch('ollama.AsyncClient'):
            analyzer = LocalMultimodalAnalyzer(verbose=False)

            assert analyzer._has_only_videos([]) is False

    def test_has_only_videos_none(self):
        """Test None media URL list."""
        with patch('ollama.AsyncClient'):
            analyzer = LocalMultimodalAnalyzer(verbose=False)

            assert analyzer._has_only_videos(None) is False


class TestResponseParsing:
    """Test response parsing methods."""

    def test_parse_category_and_explanation_valid(self):
        """Test parsing valid structured response."""
        with patch('ollama.AsyncClient'):
            analyzer = LocalMultimodalAnalyzer(verbose=False)

            response = "CATEGORÍA: hate_speech\nEXPLICACIÓN: Este contenido es discriminatorio."
            category, explanation = analyzer._parse_category_and_explanation(response)

            assert category == "hate_speech"
            assert explanation == "Este contenido es discriminatorio."

    def test_parse_category_and_explanation_empty_response(self):
        """Test parsing empty response."""
        with patch('ollama.AsyncClient'):
            analyzer = LocalMultimodalAnalyzer(verbose=False)

            category, explanation = analyzer._parse_category_and_explanation("")

            assert category == Categories.GENERAL
            assert "Error: Model returned empty response" in explanation

    def test_parse_category_and_explanation_invalid_category(self):
        """Test parsing with invalid category."""
        with patch('ollama.AsyncClient'):
            analyzer = LocalMultimodalAnalyzer(verbose=False)

            response = "CATEGORÍA: invalid_category\nEXPLICACIÓN: Some explanation."
            category, explanation = analyzer._parse_category_and_explanation(response)

            assert category == Categories.GENERAL  # Should fallback to GENERAL

    def test_parse_category_and_explanation_short_explanation(self):
        """Test parsing with very short explanation."""
        with patch('ollama.AsyncClient'):
            analyzer = LocalMultimodalAnalyzer(verbose=False)

            response = "CATEGORÍA: hate_speech\nEXPLICACIÓN: Short"
            category, explanation = analyzer._parse_category_and_explanation(response)

            assert category == "hate_speech"
            assert "Contenido clasificado como hate_speech" in explanation  # Should generate fallback


class TestAnalysisContent:
    """Test the unified _analyze_content method."""

    @pytest.mark.asyncio
    async def test_analyze_content_text_only_categorization(self):
        """Test text-only categorization."""
        with patch('ollama.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.generate.return_value = {"response": "test response"}
            mock_client_class.return_value = mock_client

            analyzer = LocalMultimodalAnalyzer(verbose=False)

            result = await analyzer._analyze_content("test content")

            assert result == "test response"

    @pytest.mark.asyncio
    async def test_analyze_content_text_only_explanation(self):
        """Test text-only explanation."""
        with patch('ollama.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.generate.return_value = {"response": "test explanation"}
            mock_client_class.return_value = mock_client

            analyzer = LocalMultimodalAnalyzer(verbose=False)

            result = await analyzer._analyze_content("test content", category="hate_speech")

            assert result == "test explanation"

    @pytest.mark.asyncio
    async def test_analyze_content_multimodal_fallback(self):
        """Test multimodal fallback to text-only when no valid media."""
        with patch('ollama.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.generate.return_value = {"response": "text response"}
            mock_client_class.return_value = mock_client

            analyzer = LocalMultimodalAnalyzer(verbose=False)

            # Mock _prepare_media_content to return empty list
            with patch.object(analyzer, '_prepare_media_content', return_value=[]):
                result = await analyzer._analyze_content("test content", ["invalid.jpg"])

                assert result == "text response"


class TestVerboseOutput:
    """Test verbose output paths."""

    @pytest.mark.asyncio
    async def test_categorize_and_explain_verbose_output(self):
        """Test verbose output in categorize_and_explain."""
        with patch('ollama.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.generate.return_value = {"response": "CATEGORÍA: hate_speech\nEXPLICACIÓN: Test explanation."}
            mock_client_class.return_value = mock_client

            analyzer = LocalMultimodalAnalyzer(verbose=True)

            with patch('builtins.print') as mock_print:
                category, explanation = await analyzer.categorize_and_explain("test content", ["image.jpg"])

                # Should print verbose messages
                assert mock_print.call_count >= 3  # At least initialization, content, and result prints

    @pytest.mark.asyncio
    async def test_explain_only_verbose_output(self):
        """Test verbose output in explain_only."""
        with patch('ollama.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.generate.return_value = {"response": "Test explanation."}
            mock_client_class.return_value = mock_client

            analyzer = LocalMultimodalAnalyzer(verbose=True)

            with patch('builtins.print') as mock_print:
                explanation = await analyzer.explain_only("test content", "hate_speech", ["image.jpg"])

                # Should print verbose messages
                assert mock_print.call_count >= 3

    @pytest.mark.asyncio
    async def test_analyze_content_multimodal_verbose_fallback(self):
        """Test verbose fallback message in _analyze_content."""
        with patch('ollama.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.generate.return_value = {"response": "text response"}
            mock_client_class.return_value = mock_client

            analyzer = LocalMultimodalAnalyzer(verbose=True)

            # Mock _prepare_media_content to return empty list (triggers fallback)
            with patch.object(analyzer, '_prepare_media_content', return_value=[]):
                with patch('builtins.print') as mock_print:
                    result = await analyzer._analyze_content("test content", ["invalid.jpg"])

                    # Should print fallback message
                    fallback_calls = [call for call in mock_print.call_args_list
                                    if "No valid media content found" in str(call)]
                    assert len(fallback_calls) > 0


class TestRetryVerboseOutput:
    """Test verbose output in retry logic."""

    @pytest.mark.asyncio
    async def test_retry_ollama_call_verbose_retry_attempt(self):
        """Test verbose output during retry attempts."""
        with patch('ollama.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            # First call returns empty, second succeeds
            mock_client.generate.side_effect = [
                {"response": ""},
                {"response": "success response"}
            ]
            mock_client_class.return_value = mock_client

            analyzer = LocalMultimodalAnalyzer(verbose=True)

            with patch('builtins.print') as mock_print:
                with patch('asyncio.sleep'):
                    result = await analyzer._retry_ollama_call("test op", mock_client.generate, model="test")

                    # Should print retry attempt message
                    retry_calls = [call for call in mock_print.call_args_list
                                 if "Retry attempt" in str(call)]
                    assert len(retry_calls) > 0

    @pytest.mark.asyncio
    async def test_retry_ollama_call_verbose_empty_response_retry(self):
        """Test verbose output for empty response retry."""
        with patch('ollama.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            # First call returns empty, second succeeds
            mock_client.generate.side_effect = [
                {"response": ""},
                {"response": "success response"}
            ]
            mock_client_class.return_value = mock_client

            analyzer = LocalMultimodalAnalyzer(verbose=True)

            with patch('builtins.print') as mock_print:
                with patch('asyncio.sleep'):
                    result = await analyzer._retry_ollama_call("test op", mock_client.generate, model="test")

                    # Should print empty response retry message
                    empty_retry_calls = [call for call in mock_print.call_args_list
                                       if "returned empty response, retrying" in str(call)]
                    assert len(empty_retry_calls) > 0


class TestMediaPreparationVerbose:
    """Test verbose output in media preparation."""

    @pytest.mark.asyncio
    async def test_prepare_media_content_verbose_skipping_video(self):
        """Test verbose output when skipping videos."""
        with patch('ollama.AsyncClient'):
            analyzer = LocalMultimodalAnalyzer(verbose=True)

            with patch('builtins.print') as mock_print:
                result = await analyzer._prepare_media_content(["https://example.com/video.mp4"])

                # Should print skipping video message
                skip_calls = [call for call in mock_print.call_args_list
                            if "Skipping video file" in str(call)]
                assert len(skip_calls) > 0

    @pytest.mark.asyncio
    async def test_prepare_media_content_verbose_unsupported_format(self):
        """Test verbose output when skipping unsupported formats."""
        with patch('ollama.AsyncClient'):
            analyzer = LocalMultimodalAnalyzer(verbose=True)

            with patch('builtins.print') as mock_print:
                result = await analyzer._prepare_media_content(["https://example.com/file.txt"])

                # Should print skipping unsupported format message
                skip_calls = [call for call in mock_print.call_args_list
                            if "Skipping unsupported media format" in str(call)]
                assert len(skip_calls) > 0

    @pytest.mark.asyncio
    async def test_prepare_media_content_verbose_large_file(self):
        """Test verbose output when skipping large files."""
        with patch('ollama.AsyncClient'):
            analyzer = LocalMultimodalAnalyzer(verbose=True)

            # Mock response with large content-length
            mock_response = Mock()
            mock_response.headers = {'content-length': str(20 * 1024 * 1024)}  # 20MB
            mock_response.raise_for_status.return_value = None

            with patch('requests.get', return_value=mock_response):
                with patch('builtins.print') as mock_print:
                    result = await analyzer._prepare_media_content(["https://example.com/large.jpg"])

                    # Should print skipping large file message
                    skip_calls = [call for call in mock_print.call_args_list
                                if "Skipping large file" in str(call)]
                    assert len(skip_calls) > 0

    @pytest.mark.asyncio
    async def test_prepare_media_content_verbose_downloading(self):
        """Test verbose output during download."""
        with patch('ollama.AsyncClient'):
            analyzer = LocalMultimodalAnalyzer(verbose=True)

            # Mock successful response
            mock_response = Mock()
            mock_response.content = b"fake image data"
            mock_response.headers = {'content-length': '1000'}
            mock_response.raise_for_status.return_value = None

            with patch('requests.get', return_value=mock_response) as mock_get:
                with patch('builtins.print') as mock_print:
                    result = await analyzer._prepare_media_content(["https://example.com/image.jpg"])

                    # Should print downloading and success messages
                    download_calls = [call for call in mock_print.call_args_list
                                    if "Downloading image" in str(call)]
                    assert len(download_calls) > 0

                    success_calls = [call for call in mock_print.call_args_list
                                   if "Downloaded" in str(call)]
                    assert len(success_calls) > 0

    @pytest.mark.asyncio
    async def test_prepare_media_content_verbose_timeout(self):
        """Test verbose output on download timeout."""
        with patch('ollama.AsyncClient'):
            analyzer = LocalMultimodalAnalyzer(verbose=True)

            import requests
            with patch('requests.get', side_effect=requests.exceptions.Timeout("Timeout")):
                with patch('builtins.print') as mock_print:
                    result = await analyzer._prepare_media_content(["https://example.com/image.jpg"])

                    # Should print timeout message
                    timeout_calls = [call for call in mock_print.call_args_list
                                   if "Timeout downloading media" in str(call)]
                    assert len(timeout_calls) > 0

    @pytest.mark.asyncio
    async def test_prepare_media_content_verbose_prepared_items(self):
        """Test verbose output when media items are prepared."""
        with patch('ollama.AsyncClient'):
            analyzer = LocalMultimodalAnalyzer(verbose=True)

            # Mock successful response
            mock_response = Mock()
            mock_response.content = b"fake image data"
            mock_response.headers = {'content-length': '1000'}
            mock_response.raise_for_status.return_value = None

            with patch('requests.get', return_value=mock_response):
                with patch('builtins.print') as mock_print:
                    result = await analyzer._prepare_media_content(["https://example.com/image.jpg"])

                    # Should print prepared items message
                    prepared_calls = [call for call in mock_print.call_args_list
                                    if "Prepared" in str(call) and "media items" in str(call)]
                    assert len(prepared_calls) > 0


class TestMultimodalGenerationKeepAlive:
    """Test multimodal generation with keep_alive parameter."""

    @pytest.mark.asyncio
    async def test_generate_multimodal_keep_alive_parameter(self):
        """Test that keep_alive parameter is passed to Ollama."""
        with patch('ollama.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.generate.return_value = {"response": "multimodal response"}
            mock_client_class.return_value = mock_client

            analyzer = LocalMultimodalAnalyzer(verbose=False)

            media_content = [{"type": "image_url", "image_url": {"url": "base64data"}}]
            result = await analyzer._generate_multimodal_with_ollama("test prompt", media_content, "gemma3:4b")

            # Verify keep_alive parameter was passed
            call_args = mock_client.generate.call_args
            assert "keep_alive" in call_args.kwargs
            assert call_args.kwargs["keep_alive"] == "24h"
