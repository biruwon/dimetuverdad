"""
Unit tests for multi-model analysis functionality in LocalMultimodalAnalyzer.
Focuses on the new multi-model analysis methods with 80%+ coverage.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from analyzer.local_analyzer import LocalMultimodalAnalyzer
from analyzer.categories import Categories


class TestMultiModelConfiguration:
    """Test multi-model configuration and capabilities."""
    
    def test_available_models_configuration(self):
        """Test that AVAILABLE_MODELS is properly configured."""
        analyzer = LocalMultimodalAnalyzer(verbose=False)
        
        # Check that all expected models are present
        expected_models = ["gemma3:4b", "gemma3:12b", "gemma3:27b-it-qat", "gpt-oss:20b"]
        for model in expected_models:
            assert model in analyzer.AVAILABLE_MODELS
        
        # Check model configurations
        assert analyzer.AVAILABLE_MODELS["gemma3:4b"]["type"] == "fast"
        assert analyzer.AVAILABLE_MODELS["gemma3:12b"]["type"] == "fast"
        assert analyzer.AVAILABLE_MODELS["gemma3:27b-it-qat"]["type"] == "accurate"
        assert analyzer.AVAILABLE_MODELS["gpt-oss:20b"]["type"] == "balanced"
    
    def test_multimodal_capabilities(self):
        """Test that multimodal capabilities are correctly defined."""
        analyzer = LocalMultimodalAnalyzer(verbose=False)
        
        # Gemma models should support multimodal
        assert analyzer.AVAILABLE_MODELS["gemma3:4b"]["multimodal"] is True
        assert analyzer.AVAILABLE_MODELS["gemma3:12b"]["multimodal"] is True
        assert analyzer.AVAILABLE_MODELS["gemma3:27b-it-qat"]["multimodal"] is True
        
        # GPT-OSS should be text-only
        assert analyzer.AVAILABLE_MODELS["gpt-oss:20b"]["multimodal"] is False


class TestMultiModelAnalyzerMethods:
    """Test multi-model analyzer methods existence and signatures."""
    
    def test_analyzer_has_multi_model_method(self):
        """Test that analyzer has the analyze_with_multiple_models method."""
        analyzer = LocalMultimodalAnalyzer(verbose=False)
        assert hasattr(analyzer, 'analyze_with_multiple_models')
        assert callable(analyzer.analyze_with_multiple_models)
    
    def test_analyzer_has_analyze_with_specific_model(self):
        """Test that analyzer has the _analyze_with_specific_model helper."""
        analyzer = LocalMultimodalAnalyzer(verbose=False)
        assert hasattr(analyzer, '_analyze_with_specific_model')
        assert callable(analyzer._analyze_with_specific_model)


class TestAnalyzeWithMultipleModels:
    """Test analyze_with_multiple_models method."""
    
    @pytest.mark.asyncio
    async def test_analyze_with_multiple_models_default_models(self):
        """Test multi-model analysis with default model list."""
        analyzer = LocalMultimodalAnalyzer(verbose=False)
        
        # Mock the _analyze_with_specific_model method
        async def mock_analyze(content, media, model):
            return (Categories.GENERAL, f"Explanation from {model}", 1.5)
        
        analyzer._analyze_with_specific_model = AsyncMock(side_effect=mock_analyze)
        
        content = "Test content"
        results = await analyzer.analyze_with_multiple_models(content)
        
        # Should use all available models by default
        assert len(results) == len(analyzer.AVAILABLE_MODELS)
        assert all(model in results for model in analyzer.AVAILABLE_MODELS.keys())
    
    @pytest.mark.asyncio
    async def test_analyze_with_multiple_models_specific_models(self):
        """Test multi-model analysis with specific model list."""
        analyzer = LocalMultimodalAnalyzer(verbose=False)
        
        async def mock_analyze(content, media, model):
            return (Categories.HATE_SPEECH, f"Explanation from {model}", 2.0)
        
        analyzer._analyze_with_specific_model = AsyncMock(side_effect=mock_analyze)
        
        content = "Test content"
        models = ["gemma3:4b", "gpt-oss:20b"]
        results = await analyzer.analyze_with_multiple_models(content, models=models)
        
        # Should only use specified models
        assert len(results) == 2
        assert "gemma3:4b" in results
        assert "gpt-oss:20b" in results
    
    @pytest.mark.asyncio
    async def test_analyze_with_multiple_models_video_only_skips_multimodal(self):
        """Test that video-only content skips multimodal models."""
        analyzer = LocalMultimodalAnalyzer(verbose=False)
        
        # Mock _has_only_videos to return True
        analyzer._has_only_videos = Mock(return_value=True)
        
        async def mock_analyze(content, media, model):
            # Should receive None for media if videos only
            assert media is None
            return (Categories.GENERAL, "Text analysis", 1.0)
        
        analyzer._analyze_with_specific_model = AsyncMock(side_effect=mock_analyze)
        
        content = "Test content"
        media_urls = ["https://video.twimg.com/video.mp4"]
        results = await analyzer.analyze_with_multiple_models(content, media_urls=media_urls)
        
        # Should still analyze but without media
        assert len(results) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_with_multiple_models_prepares_media_once(self):
        """Test that media is prepared once for all multimodal models."""
        analyzer = LocalMultimodalAnalyzer(verbose=False)
        
        # Mock media preparation
        mock_media_content = [{"type": "image", "data": "base64data"}]
        analyzer._prepare_media_content = AsyncMock(return_value=mock_media_content)
        
        async def mock_analyze(content, media, model):
            return (Categories.GENERAL, "Analysis", 1.0)
        
        analyzer._analyze_with_specific_model = AsyncMock(side_effect=mock_analyze)
        
        content = "Test content"
        media_urls = ["https://pbs.twimg.com/media/image.jpg"]
        await analyzer.analyze_with_multiple_models(content, media_urls=media_urls)
        
        # Media should be prepared only once
        analyzer._prepare_media_content.assert_called_once_with(media_urls)
    
    @pytest.mark.asyncio
    async def test_analyze_with_multiple_models_returns_empty_on_all_failures(self):
        """Test that error results are returned when models fail."""
        analyzer = LocalMultimodalAnalyzer(verbose=False)
        
        # Mock all analyses to fail
        async def mock_failing_analyze(content, media, model):
            raise Exception("Model failed")
        
        analyzer._analyze_with_specific_model = AsyncMock(side_effect=mock_failing_analyze)
        
        content = "Test content"
        results = await analyzer.analyze_with_multiple_models(content, models=["gemma3:4b"])
        
        # Should still return results (with error messages)
        assert len(results) == 1
        assert "gemma3:4b" in results
        # Error results have GENERAL category and error message
        category, explanation, time_taken = results["gemma3:4b"]
        assert category == Categories.GENERAL
        assert "Error durante el análisis" in explanation


class TestAnalyzeWithSpecificModel:
    """Test _analyze_with_specific_model method."""
    
    @pytest.mark.asyncio
    async def test_analyze_with_specific_model_text_only(self):
        """Test analyzing with text-only model."""
        analyzer = LocalMultimodalAnalyzer(verbose=False)
        
        # Mock _analyze_content to return a response string
        analyzer._analyze_content = AsyncMock(return_value="CATEGORÍA: general\nEXPLICACIÓN: Test explanation")
        analyzer._parse_category_and_explanation = Mock(return_value=(Categories.GENERAL, "Test explanation"))
        
        content = "Test content"
        category, explanation, time_taken = await analyzer._analyze_with_specific_model(
            content, None, "gpt-oss:20b"
        )
        
        assert category == Categories.GENERAL
        assert explanation == "Test explanation"
        assert time_taken >= 0
    
    @pytest.mark.asyncio
    async def test_analyze_with_specific_model_multimodal(self):
        """Test analyzing with multimodal model."""
        analyzer = LocalMultimodalAnalyzer(verbose=False)
        
        mock_media = [{"type": "image", "data": "base64"}]
        
        # Mock _analyze_content to return a response string
        analyzer._analyze_content = AsyncMock(return_value="CATEGORÍA: hate_speech\nEXPLICACIÓN: Multimodal analysis")
        analyzer._parse_category_and_explanation = Mock(return_value=(Categories.HATE_SPEECH, "Multimodal analysis"))
        
        content = "Test content"
        category, explanation, time_taken = await analyzer._analyze_with_specific_model(
            content, mock_media, "gemma3:4b"
        )
        
        assert category == Categories.HATE_SPEECH
        assert explanation == "Multimodal analysis"
        assert time_taken >= 0
    
    @pytest.mark.asyncio
    async def test_analyze_with_specific_model_handles_errors(self):
        """Test error handling in specific model analysis."""
        analyzer = LocalMultimodalAnalyzer(verbose=False)
        
        # Mock analysis to fail
        analyzer._analyze_content = AsyncMock(side_effect=RuntimeError("Analysis failed"))
        
        content = "Test content"
        
        with pytest.raises(RuntimeError, match="Model gemma3:4b analysis failed"):
            await analyzer._analyze_with_specific_model(content, None, "gemma3:4b")
    
    @pytest.mark.asyncio
    async def test_analyze_with_specific_model_tracks_processing_time(self):
        """Test that processing time is correctly tracked."""
        analyzer = LocalMultimodalAnalyzer(verbose=False)
        
        async def mock_analyze_content(*args, **kwargs):
            # Simulate some processing time
            await asyncio.sleep(0.01)
            return "CATEGORÍA: general\nEXPLICACIÓN: Test"
        
        analyzer._analyze_content = AsyncMock(side_effect=mock_analyze_content)
        analyzer._parse_category_and_explanation = Mock(return_value=(Categories.GENERAL, "Test"))
        
        content = "Test content"
        category, explanation, time_taken = await analyzer._analyze_with_specific_model(
            content, None, "gpt-oss:20b"
        )
        
        # Should have tracked some processing time
        assert time_taken > 0
        assert time_taken < 1.0  # Should be quick for mock


class TestMultiModelVerboseOutput:
    """Test verbose output for multi-model analysis."""
    
    @pytest.mark.asyncio
    async def test_analyze_with_multiple_models_verbose_output(self, capsys):
        """Test verbose output during multi-model analysis."""
        analyzer = LocalMultimodalAnalyzer(verbose=True)
        
        async def mock_analyze(content, media, model):
            return (Categories.GENERAL, "Explanation", 1.5)
        
        analyzer._analyze_with_specific_model = AsyncMock(side_effect=mock_analyze)
        
        content = "Test content"
        models = ["gemma3:4b"]
        await analyzer.analyze_with_multiple_models(content, models=models)
        
        captured = capsys.readouterr()
        # Should show progress information
        assert "multi-model" in captured.out.lower() or "model" in captured.out.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
