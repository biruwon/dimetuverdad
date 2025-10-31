"""
Tests for MultiModelAnalyzer - orchestrates multi-model analysis for comparison.
"""

import pytest
from unittest.mock import AsyncMock, patch, Mock
from analyzer.multi_model_analyzer import MultiModelAnalyzer
from analyzer.categories import Categories


class TestMultiModelConfiguration:
    """Test multi-model configuration and capabilities."""
    
    def test_available_models_configuration(self):
        """Test that AVAILABLE_MODELS is properly configured."""
        analyzer = MultiModelAnalyzer(verbose=False)
        
        # Check that all expected models are present
        expected_models = ["gemma3:4b", "gemma3:12b", "gemma3:27b-it-qat", "gpt-oss:20b"]
        for model in expected_models:
            assert model in MultiModelAnalyzer.AVAILABLE_MODELS
        
        # Check model configurations
        assert MultiModelAnalyzer.AVAILABLE_MODELS["gemma3:4b"]["type"] == "fast"
        assert MultiModelAnalyzer.AVAILABLE_MODELS["gemma3:12b"]["type"] == "fast"
        assert MultiModelAnalyzer.AVAILABLE_MODELS["gemma3:27b-it-qat"]["type"] == "accurate"
        assert MultiModelAnalyzer.AVAILABLE_MODELS["gpt-oss:20b"]["type"] == "balanced"
    
    def test_multimodal_capabilities(self):
        """Test multimodal capability flags."""
        # Gemma models support multimodal
        assert MultiModelAnalyzer.AVAILABLE_MODELS["gemma3:4b"]["multimodal"] is True
        assert MultiModelAnalyzer.AVAILABLE_MODELS["gemma3:12b"]["multimodal"] is True
        assert MultiModelAnalyzer.AVAILABLE_MODELS["gemma3:27b-it-qat"]["multimodal"] is True
        
        # GPT-OSS is text-only
        assert MultiModelAnalyzer.AVAILABLE_MODELS["gpt-oss:20b"]["multimodal"] is False


class TestAnalyzeWithMultipleModels:
    """Test the analyze_with_multiple_models method."""
    
    @pytest.mark.asyncio
    async def test_analyze_with_default_models(self):
        """Test analysis with all default models."""
        analyzer = MultiModelAnalyzer(verbose=False)
        
        with patch.object(analyzer, '_analyze_with_specific_model', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = (Categories.HATE_SPEECH, "Test explanation", 10.5)
            
            results = await analyzer.analyze_with_multiple_models("Test content")
            
            # Should analyze with all 4 default models
            assert len(results) == 4
            assert "gemma3:4b" in results
            assert "gpt-oss:20b" in results
    
    @pytest.mark.asyncio
    async def test_analyze_with_specific_models(self):
        """Test analysis with specific model list."""
        analyzer = MultiModelAnalyzer(verbose=False)
        
        with patch.object(analyzer, '_analyze_with_specific_model', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = (Categories.ANTI_IMMIGRATION, "Test", 15.2)
            
            results = await analyzer.analyze_with_multiple_models(
                "Test content",
                models=["gemma3:4b", "gpt-oss:20b"]
            )
            
            assert len(results) == 2
            assert "gemma3:4b" in results
            assert "gpt-oss:20b" in results
    
    @pytest.mark.asyncio
    async def test_analyze_handles_model_failure(self):
        """Test that failures in one model don't stop analysis of other models."""
        analyzer = MultiModelAnalyzer(verbose=False)
        
        async def mock_analyze_side_effect(content, media, model):
            if model == "gemma3:4b":
                raise RuntimeError("Model failed")
            return (Categories.GENERAL, "Success", 5.0)
        
        with patch.object(analyzer, '_analyze_with_specific_model', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.side_effect = mock_analyze_side_effect
            
            results = await analyzer.analyze_with_multiple_models(
                "Test content",
                models=["gemma3:4b", "gpt-oss:20b"]
            )
            
            # Should have results for both models (failed one has error message)
            assert len(results) == 2
            assert results["gemma3:4b"][0] == Categories.GENERAL  # Error fallback
            assert "Error" in results["gemma3:4b"][1]
            assert results["gpt-oss:20b"][0] == Categories.GENERAL


class TestAnalyzeWithSpecificModel:
    """Test the _analyze_with_specific_model method."""
    
    @pytest.mark.asyncio
    async def test_text_only_analysis(self):
        """Test text-only analysis with specific model."""
        analyzer = MultiModelAnalyzer(verbose=False)
        
        with patch('analyzer.multi_model_analyzer.OllamaAnalyzer') as MockAnalyzer:
            mock_instance = MockAnalyzer.return_value
            mock_instance.client.generate_text = AsyncMock(return_value="CATEGORÍA: hate_speech\nEXPLICACIÓN: Test")
            mock_instance._parse_category_and_explanation = Mock(
                return_value=(Categories.HATE_SPEECH, "Test explanation")
            )
            mock_instance.prompt_generator = Mock()
            mock_instance.prompt_generator.build_ollama_categorization_prompt = Mock(return_value="prompt")
            mock_instance.prompt_generator.build_ollama_text_analysis_system_prompt = Mock(return_value="system")
            mock_instance.DEFAULT_TEMPERATURE_TEXT = 0.3
            mock_instance.DEFAULT_MAX_TOKENS = 512
            
            category, explanation, time_taken = await analyzer._analyze_with_specific_model(
                "Test content",
                prepared_media_content=None,
                model="gpt-oss:20b"
            )
            
            assert category == Categories.HATE_SPEECH
            assert explanation == "Test explanation"
            assert time_taken > 0
    
    @pytest.mark.asyncio
    async def test_multimodal_analysis(self):
        """Test multimodal analysis with images."""
        analyzer = MultiModelAnalyzer(verbose=False)
        
        prepared_media = [{"type": "image_url", "image_url": {"url": "base64data"}}]
        
        with patch('analyzer.multi_model_analyzer.OllamaAnalyzer') as MockAnalyzer:
            mock_instance = MockAnalyzer.return_value
            mock_instance.client.generate_multimodal = AsyncMock(
                return_value="CATEGORÍA: anti_immigration\nEXPLICACIÓN: Test"
            )
            mock_instance._parse_category_and_explanation = Mock(
                return_value=(Categories.ANTI_IMMIGRATION, "Test explanation")
            )
            mock_instance.prompt_generator = Mock()
            mock_instance.prompt_generator.build_multimodal_categorization_prompt = Mock(return_value="prompt")
            mock_instance.prompt_generator.build_ollama_multimodal_system_prompt = Mock(return_value="system")
            mock_instance.DEFAULT_TEMPERATURE_MULTIMODAL = 0.2
            mock_instance.DEFAULT_TOP_P_MULTIMODAL = 0.7
            mock_instance.DEFAULT_NUM_PREDICT_MULTIMODAL = 250
            mock_instance.DEFAULT_KEEP_ALIVE = "24h"
            
            category, explanation, time_taken = await analyzer._analyze_with_specific_model(
                "Test content",
                prepared_media_content=prepared_media,
                model="gemma3:4b"
            )
            
            assert category == Categories.ANTI_IMMIGRATION
            assert explanation == "Test explanation"
            assert time_taken > 0


class TestGetAnalyzer:
    """Test analyzer caching functionality."""
    
    def test_get_analyzer_creates_new_instance(self):
        """Test that _get_analyzer creates new analyzer instances."""
        multi_analyzer = MultiModelAnalyzer(verbose=False)
        
        analyzer1 = multi_analyzer._get_analyzer("gemma3:4b")
        assert analyzer1 is not None
        assert analyzer1.model == "gemma3:4b"
    
    def test_get_analyzer_caches_instances(self):
        """Test that _get_analyzer caches analyzer instances."""
        multi_analyzer = MultiModelAnalyzer(verbose=False)
        
        analyzer1 = multi_analyzer._get_analyzer("gemma3:4b")
        analyzer2 = multi_analyzer._get_analyzer("gemma3:4b")
        
        # Should return the same cached instance
        assert analyzer1 is analyzer2
    
    def test_get_analyzer_different_models(self):
        """Test that different models get different analyzer instances."""
        multi_analyzer = MultiModelAnalyzer(verbose=False)
        
        analyzer1 = multi_analyzer._get_analyzer("gemma3:4b")
        analyzer2 = multi_analyzer._get_analyzer("gpt-oss:20b")
        
        # Should be different instances
        assert analyzer1 is not analyzer2
        assert analyzer1.model == "gemma3:4b"
        assert analyzer2.model == "gpt-oss:20b"
