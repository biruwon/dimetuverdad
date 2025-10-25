"""
Unit tests for LocalLLMAnalyzer component.
Tests category detection and explanation generation using local LLM (gpt-oss:20b).
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch

from analyzer.local_llm_analyzer import LocalLLMAnalyzer
from analyzer.categories import Categories


@pytest.fixture
def mock_ollama_client():
    """Mock OpenAI client for Ollama responses."""
    mock_client = Mock()
    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    
    # Setup nested mock structure: client.chat.completions.create()
    mock_message.content = "CATEGORÍA: hate_speech\nEXPLICACIÓN: Este contenido contiene discurso de odio xenófobo."
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    
    return mock_client


@pytest.fixture
def analyzer(mock_ollama_client):
    """Create LocalLLMAnalyzer with mocked Ollama client."""
    with patch('analyzer.local_llm_analyzer.OpenAI', return_value=mock_ollama_client):
        analyzer = LocalLLMAnalyzer(verbose=False)
        return analyzer


class TestLocalLLMAnalyzerInitialization:
    """Test analyzer initialization and configuration."""
    
    def test_default_initialization(self):
        """Test default initialization with gpt-oss:20b model."""
        with patch('analyzer.local_llm_analyzer.OpenAI'):
            analyzer = LocalLLMAnalyzer()
            
            assert analyzer.model == "gpt-oss:20b"
            assert analyzer.verbose is False
            assert analyzer.prompt_generator is not None
    
    def test_custom_model_initialization(self):
        """Test initialization with custom model name."""
        with patch('analyzer.local_llm_analyzer.OpenAI'):
            analyzer = LocalLLMAnalyzer(model="gpt-oss:7b", verbose=True)
            
            assert analyzer.model == "gpt-oss:7b"
            assert analyzer.verbose is True


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
        mock_ollama_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_ollama_client.chat.completions.create.call_args.kwargs
        assert call_kwargs['model'] == "gpt-oss:20b"
        assert call_kwargs['temperature'] == 0.3
        assert call_kwargs['max_tokens'] == 512
    
    @pytest.mark.asyncio
    async def test_general_category_fallback(self, analyzer, mock_ollama_client):
        """Test fallback to general category when LLM returns unrecognized category."""
        # Mock response with unrecognized category
        mock_ollama_client.chat.completions.create.return_value.choices[0].message.content = (
            "CATEGORÍA: unknown_category\nEXPLICACIÓN: Some explanation."
        )
        
        content = "Neutral political statement"
        category, explanation = await analyzer.categorize_and_explain(content)
        
        # Should fallback to general category when category not recognized
        assert category == Categories.GENERAL
        assert "Some explanation" in explanation or "Contenido clasificado como" in explanation
    
    @pytest.mark.asyncio
    async def test_error_handling(self, analyzer, mock_ollama_client):
        """Test error handling when Ollama fails."""
        mock_ollama_client.chat.completions.create.side_effect = RuntimeError("Ollama connection failed")
        
        content = "Test content"
        category, explanation = await analyzer.categorize_and_explain(content)
        
        assert category == Categories.GENERAL
        assert "Error en análisis local" in explanation


class TestExplainOnly:
    """Test explanation generation for known categories."""
    
    @pytest.mark.asyncio
    async def test_successful_explanation(self, analyzer, mock_ollama_client):
        """Test generating explanation for known category."""
        # Mock response for explanation-only
        mock_ollama_client.chat.completions.create.return_value.choices[0].message.content = (
            "Este contenido muestra retórica extremista con ataques xenófobos directos."
        )
        
        content = "Los inmigrantes son criminales"
        category = Categories.HATE_SPEECH
        
        explanation = await analyzer.explain_only(content, category)
        
        assert "extremista" in explanation or "xenófobos" in explanation
        
        # Verify prompt was category-specific
        mock_ollama_client.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_explanation_error_handling(self, analyzer, mock_ollama_client):
        """Test error handling in explanation generation."""
        mock_ollama_client.chat.completions.create.side_effect = RuntimeError("Generation failed")
        
        content = "Test content"
        explanation = await analyzer.explain_only(content, Categories.DISINFORMATION)
        
        assert "Error generando explicación local" in explanation


class TestPromptBuilding:
    """Test prompt generation methods."""
    
    def test_categorization_prompt_structure(self, analyzer):
        """Test categorization prompt contains all required elements."""
        content = "Test content for analysis"
        prompt = analyzer.prompt_generator.build_categorization_prompt(content)
        
        # Check prompt structure
        assert "CATEGORÍAS:" in prompt
        assert content in prompt
        assert "FORMATO OBLIGATORIO:" in prompt
        assert "CATEGORÍA:" in prompt
        assert "EXPLICACIÓN:" in prompt
        
        # Check all categories are listed
        assert Categories.HATE_SPEECH in prompt
        assert Categories.DISINFORMATION in prompt
        assert Categories.CONSPIRACY_THEORY in prompt
        assert Categories.GENERAL in prompt
    
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
    
    def test_generate_with_ollama_success(self, analyzer, mock_ollama_client):
        """Test successful Ollama generation."""
        prompt = "Test prompt"
        
        result = analyzer._generate_with_ollama(prompt)
        
        assert result == "CATEGORÍA: hate_speech\nEXPLICACIÓN: Este contenido contiene discurso de odio xenófobo."
        
        # Verify API call structure
        call_kwargs = mock_ollama_client.chat.completions.create.call_args.kwargs
        assert call_kwargs['model'] == "gpt-oss:20b"
        assert len(call_kwargs['messages']) == 2
        assert call_kwargs['messages'][0]['role'] == "system"
        assert call_kwargs['messages'][1]['role'] == "user"
        assert call_kwargs['messages'][1]['content'] == prompt
    
    def test_generate_with_ollama_error(self, analyzer, mock_ollama_client):
        """Test error handling in Ollama generation."""
        mock_ollama_client.chat.completions.create.side_effect = Exception("Connection timeout")
        
        prompt = "Test prompt"
        
        with pytest.raises(RuntimeError, match="Ollama generation failed"):
            analyzer._generate_with_ollama(prompt)


class TestVerboseLogging:
    """Test verbose logging output."""
    
    @pytest.mark.asyncio
    async def test_categorize_verbose_output(self, mock_ollama_client, capsys):
        """Test verbose logging in categorize_and_explain."""
        with patch('analyzer.local_llm_analyzer.OpenAI', return_value=mock_ollama_client):
            analyzer = LocalLLMAnalyzer(verbose=True)
            
            await analyzer.categorize_and_explain("Test content")
            
            captured = capsys.readouterr()
            assert "🔍 Running local LLM categorization + explanation" in captured.out
            assert "✅ Category detected:" in captured.out
    
    @pytest.mark.asyncio
    async def test_explain_verbose_output(self, mock_ollama_client, capsys):
        """Test verbose logging in explain_only."""
        with patch('analyzer.local_llm_analyzer.OpenAI', return_value=mock_ollama_client):
            analyzer = LocalLLMAnalyzer(verbose=True)
            
            await analyzer.explain_only("Test content", Categories.HATE_SPEECH)
            
            captured = capsys.readouterr()
            assert "🔍 Generating local explanation for category:" in captured.out
