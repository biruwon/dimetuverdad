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
        expected_models = ["gemma3:4b", "gemma3:27b-it-q4_K_M"]
        for model in expected_models:
            assert model in MultiModelAnalyzer.AVAILABLE_MODELS
        
        # Check model configurations
        assert MultiModelAnalyzer.AVAILABLE_MODELS["gemma3:4b"]["type"] == "fast"
        assert MultiModelAnalyzer.AVAILABLE_MODELS["gemma3:27b-it-q4_K_M"]["type"] == "accurate"
    
    def test_multimodal_capabilities(self):
        """Test multimodal capability flags."""
        # All current Gemma models support multimodal
        assert MultiModelAnalyzer.AVAILABLE_MODELS["gemma3:4b"]["multimodal"] is True
        assert MultiModelAnalyzer.AVAILABLE_MODELS["gemma3:27b-it-q4_K_M"]["multimodal"] is True


class TestAnalyzeWithMultipleModels:
    """Test the analyze_with_multiple_models method."""
    
    @pytest.mark.asyncio
    async def test_analyze_with_default_models(self):
        """Test analysis with all default models."""
        analyzer = MultiModelAnalyzer(verbose=False)
        
        with patch.object(analyzer, '_analyze_with_specific_model', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = (Categories.HATE_SPEECH, "Test explanation", 10.5)
            
            results = await analyzer.analyze_with_multiple_models("Test content")
            
            # Should analyze with all 2 default models
            assert len(results) == 2
            assert "gemma3:4b" in results
            assert "gemma3:27b-it-q4_K_M" in results
    
    @pytest.mark.asyncio
    async def test_analyze_with_specific_models(self):
        """Test analysis with specific model list."""
        analyzer = MultiModelAnalyzer(verbose=False)
        
        with patch.object(analyzer, '_analyze_with_specific_model', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = (Categories.ANTI_IMMIGRATION, "Test", 15.2)
            
            results = await analyzer.analyze_with_multiple_models(
                "Test content",
                models=["gemma3:4b", "gemma3:27b-it-q4_K_M"]
            )
            
            assert len(results) == 2
            assert "gemma3:4b" in results
            assert "gemma3:27b-it-q4_K_M" in results
    
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
                models=["gemma3:4b", "gemma3:27b-it-q4_K_M"]
            )
            
            # Should have results for both models (failed one has error message)
            assert len(results) == 2
            assert results["gemma3:4b"][0] == Categories.GENERAL  # Error fallback
            assert "Error" in results["gemma3:4b"][1]
            assert results["gemma3:27b-it-q4_K_M"][0] == Categories.GENERAL


class TestAnalyzeWithSpecificModel:
    """Test the _analyze_with_specific_model method."""
    
    @pytest.mark.asyncio
    async def test_text_only_analysis(self):
        """Test text-only analysis with multi-stage approach."""
        analyzer = MultiModelAnalyzer(verbose=False)
        
        with patch('analyzer.multi_model_analyzer.OllamaAnalyzer') as MockAnalyzer:
            mock_instance = MockAnalyzer.return_value
            # Mock the multi-stage methods
            mock_instance.detect_category_only = AsyncMock(return_value=Categories.HATE_SPEECH)
            mock_instance.generate_explanation_with_context = AsyncMock(return_value="Test explanation")
            
            category, explanation, time_taken = await analyzer._analyze_with_specific_model(
                "Test content",
                prepared_media_content=None,
                original_media_urls=None,
                model="gpt-oss:20b"
            )
            
            assert category == Categories.HATE_SPEECH
            assert explanation == "Test explanation"
            assert time_taken > 0
            
            # Verify the multi-stage methods were called
            mock_instance.detect_category_only.assert_called_once()
            mock_instance.generate_explanation_with_context.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_multimodal_analysis(self):
        """Test multimodal analysis with images."""
        analyzer = MultiModelAnalyzer(verbose=False)
        
        prepared_media = [{"type": "image_url", "image_url": {"url": "base64data"}}]
        original_urls = ["https://example.com/image.jpg"]
        
        with patch('analyzer.multi_model_analyzer.OllamaAnalyzer') as MockAnalyzer:
            mock_instance = MockAnalyzer.return_value
            # Mock the multi-stage methods
            mock_instance.detect_category_only = AsyncMock(return_value=Categories.ANTI_IMMIGRATION)
            mock_instance.describe_media = AsyncMock(return_value="Test media description")
            mock_instance.generate_explanation_with_context = AsyncMock(return_value="Test explanation")
            
            category, explanation, time_taken = await analyzer._analyze_with_specific_model(
                "Test content",
                prepared_media_content=prepared_media,
                original_media_urls=original_urls,
                model="gemma3:4b"
            )
            
            assert category == Categories.ANTI_IMMIGRATION
            assert explanation == "Test explanation"
            assert time_taken > 0
            
            # Verify the multi-stage methods were called
            mock_instance.detect_category_only.assert_called_once()
            mock_instance.describe_media.assert_called_once()
            mock_instance.generate_explanation_with_context.assert_called_once()


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
        analyzer2 = multi_analyzer._get_analyzer("gemma3:27b-it-q4_K_M")
        
        # Should be different instances
        assert analyzer1 is not analyzer2
        assert analyzer1.model == "gemma3:4b"
        assert analyzer2.model == "gemma3:27b-it-q4_K_M"
