"""
Tests for analyzer LLM components
Comprehensive test coverage for LLM models, response parsing, and pipeline functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, patch as mock_patch
import torch
from analyzer.model_configs import LLMModelConfig
from analyzer.response_parser import ResponseParser
from analyzer.llm_pipeline import EnhancedLLMPipeline
from analyzer.categories import Categories


class TestLLMModelConfig:
    """Test LLMModelConfig class methods."""

    def test_get_recommended_model_generation_speed(self):
        """Test getting recommended model for generation with speed priority."""
        config = LLMModelConfig.get_recommended_model("generation", "speed")
        assert isinstance(config, dict)
        assert config["task_type"] == "generation"
        assert config["speed"] in ["ultra_fast", "very_fast", "fast"]

    def test_get_recommended_model_generation_quality(self):
        """Test getting recommended model for generation with quality priority."""
        config = LLMModelConfig.get_recommended_model("generation", "quality")
        assert isinstance(config, dict)
        assert config["task_type"] == "generation"

    def test_get_recommended_model_generation_balanced(self):
        """Test getting recommended model for generation with balanced priority."""
        config = LLMModelConfig.get_recommended_model("generation", "balanced")
        assert isinstance(config, dict)
        assert config["task_type"] == "generation"

    def test_get_recommended_model_classification_speed(self):
        """Test getting recommended model for classification with speed priority."""
        config = LLMModelConfig.get_recommended_model("classification", "speed")
        assert isinstance(config, dict)
        assert config["task_type"] == "classification"

    def test_get_recommended_model_classification_quality(self):
        """Test getting recommended model for classification with quality priority."""
        config = LLMModelConfig.get_recommended_model("classification", "quality")
        assert isinstance(config, dict)
        assert config["task_type"] == "classification"

    def test_get_recommended_model_classification_balanced(self):
        """Test getting recommended model for classification with balanced priority."""
        config = LLMModelConfig.get_recommended_model("classification", "balanced")
        assert isinstance(config, dict)
        assert config["task_type"] == "classification"

    def test_get_fast_models(self):
        """Test getting list of fast models."""
        models = LLMModelConfig.get_fast_models()
        assert isinstance(models, list)
        assert len(models) > 0
        # Verify all returned models are actually fast
        for model_name in models:
            config = LLMModelConfig.MODELS[model_name]
            assert config["speed"] in ["ultra_fast", "very_fast", "fast"]

    def test_get_models_by_task_generation(self):
        """Test getting models by generation task."""
        models = LLMModelConfig.get_models_by_task("generation")
        assert isinstance(models, list)
        for model_name in models:
            assert LLMModelConfig.MODELS[model_name]["task_type"] == "generation"

    def test_get_models_by_task_classification(self):
        """Test getting models by classification task."""
        models = LLMModelConfig.get_models_by_task("classification")
        assert isinstance(models, list)
        for model_name in models:
            assert LLMModelConfig.MODELS[model_name]["task_type"] == "classification"

    def test_get_spanish_models(self):
        """Test getting Spanish-optimized models."""
        models = LLMModelConfig.get_spanish_models()
        assert isinstance(models, list)
        for model_name in models:
            assert LLMModelConfig.MODELS[model_name]["language"] in ["spanish", "multilingual"]

    def test_get_models_by_size(self):
        """Test getting models under size limit."""
        models = LLMModelConfig.get_models_by_size(1.0)
        assert isinstance(models, list)
        for model_name in models:
            assert LLMModelConfig.MODELS[model_name]["size_gb"] <= 1.0

    def test_get_fastest_model_for_task_generation(self):
        """Test getting fastest model for generation task."""
        model_name = LLMModelConfig.get_fastest_model_for_task("generation")
        assert isinstance(model_name, str)
        assert model_name in LLMModelConfig.MODELS
        config = LLMModelConfig.MODELS[model_name]
        assert config["task_type"] == "generation"

    def test_get_fastest_model_for_task_classification(self):
        """Test getting fastest model for classification task."""
        model_name = LLMModelConfig.get_fastest_model_for_task("classification")
        assert isinstance(model_name, str)
        assert model_name in LLMModelConfig.MODELS
        config = LLMModelConfig.MODELS[model_name]
        assert config["task_type"] == "classification"

    def test_get_ollama_models(self):
        """Test getting Ollama models."""
        models = LLMModelConfig.get_ollama_models()
        assert isinstance(models, list)
        for model_name in models:
            config = LLMModelConfig.MODELS[model_name]
            assert config.get("pipeline_type") == "ollama"

    def test_get_recommended_model_unknown_priority(self):
        """Test get_recommended_model handles unknown priority gracefully."""
        config = LLMModelConfig.get_recommended_model("generation", "unknown")
        assert isinstance(config, dict)
        # Should still return a valid config (defaults to balanced behavior)


class TestResponseParser:
    """Test ResponseParser class methods."""

    def test_parse_classification_response_valid(self):
        """Test parsing valid classification response."""
        response = [
            [
                {"label": "toxic", "score": 0.8},
                {"label": "hate", "score": 0.6},
                {"label": "positive", "score": 0.2}
            ]
        ]
        result = ResponseParser.parse_classification_response(response)
        assert isinstance(result, dict)
        assert "llm_categories" in result
        assert isinstance(result["llm_categories"], list)

    def test_parse_classification_response_empty(self):
        """Test parsing empty classification response."""
        response = []
        result = ResponseParser.parse_classification_response(response)
        assert isinstance(result, dict)
        assert result["llm_categories"] == []

    def test_parse_classification_response_none(self):
        """Test parsing None classification response."""
        response = None
        result = ResponseParser.parse_classification_response(response)
        assert isinstance(result, dict)
        assert result["llm_categories"] == []

    def test_parse_text_generation_response_valid(self):
        """Test parsing valid text generation response."""
        response = [{"generated_text": "This is a generated response."}]
        result = ResponseParser.parse_text_generation_response(response, "prompt")
        assert isinstance(result, str)
        assert "generated response" in result

    def test_parse_text_generation_response_with_prompt_removal(self):
        """Test parsing text generation response with prompt removal."""
        response = [{"generated_text": "Prompt text. This is the response."}]
        model_config = {"prompt_removal_strategy": "remove_prompt"}
        result = ResponseParser.parse_text_generation_response(response, "Prompt text.", model_config)
        assert isinstance(result, str)
        assert result.strip() == "This is the response."

    def test_parse_text_generation_response_empty(self):
        """Test parsing empty text generation response."""
        response = []
        result = ResponseParser.parse_text_generation_response(response)
        assert result is None

    def test_parse_text2text_generation_response_valid(self):
        """Test parsing valid text2text generation response."""
        response = [{"generated_text": "Generated response"}]
        result = ResponseParser.parse_text2text_generation_response(response)
        assert isinstance(result, str)
        assert result == "Generated response"

    def test_parse_text2text_generation_response_empty(self):
        """Test parsing empty text2text generation response."""
        response = []
        result = ResponseParser.parse_text2text_generation_response(response)
        assert result is None

    def test_parse_response_classification(self):
        """Test main parse_response method for classification."""
        response = [{"label": "toxic", "score": 0.8}]
        result = ResponseParser.parse_response(response, "classification")
        assert isinstance(result, dict)
        assert "llm_categories" in result

    def test_parse_response_text_generation(self):
        """Test main parse_response method for text generation."""
        response = [{"generated_text": "Response"}]
        result = ResponseParser.parse_response(response, "text_generation")
        assert isinstance(result, str)

    def test_parse_response_text2text_generation(self):
        """Test main parse_response method for text2text generation."""
        response = [{"generated_text": "Response"}]
        result = ResponseParser.parse_response(response, "text2text_generation")
        assert isinstance(result, str)

    def test_parse_response_unknown_type(self):
        """Test main parse_response method with unknown parser type."""
        response = [{"data": "test"}]
        result = ResponseParser.parse_response(response, "unknown")
        assert result is None

    # Additional detailed ResponseParser tests from unit test file
    def test_parse_classification_response_empty_input(self):
        """Test parse_classification_response with empty input."""
        result = ResponseParser.parse_classification_response(None)
        expected = {"llm_categories": []}
        assert result == expected

    def test_parse_classification_response_empty_list(self):
        """Test parse_classification_response with empty list."""
        result = ResponseParser.parse_classification_response([])
        expected = {"llm_categories": []}
        assert result == expected

    def test_parse_classification_response_hate_speech_high_score(self):
        """Test parse_classification_response detects hate speech with high score."""
        response = [
            [
                {"label": "hate", "score": 0.8},
                {"label": "normal", "score": 0.2}
            ]
        ]
        result = ResponseParser.parse_classification_response(response)
        assert Categories.HATE_SPEECH in result["llm_categories"]

    def test_parse_classification_response_toxic_content_high_score(self):
        """Test parse_classification_response detects toxic content with high score."""
        response = [
            [
                {"label": "toxic", "score": 0.7},
                {"label": "normal", "score": 0.3}
            ]
        ]
        result = ResponseParser.parse_classification_response(response)
        assert "toxic_content" in result["llm_categories"]

    def test_parse_classification_response_normal_content_high_score(self):
        """Test parse_classification_response detects normal content."""
        response = [
            [
                {"label": "positive", "score": 0.8},
                {"label": "hate", "score": 0.1}
            ]
        ]
        result = ResponseParser.parse_classification_response(response)
        assert "normal_content" in result["llm_categories"]

    def test_parse_classification_response_low_scores_fallback_to_general(self):
        """Test parse_classification_response falls back to general with low scores."""
        response = [
            [
                {"label": "hate", "score": 0.2},
                {"label": "normal", "score": 0.4}
            ]
        ]
        result = ResponseParser.parse_classification_response(response)
        assert result["llm_categories"] == [Categories.GENERAL]

    def test_parse_classification_response_exception_handling(self):
        """Test parse_classification_response handles exceptions gracefully."""
        response = "invalid_response_format"
        result = ResponseParser.parse_classification_response(response)
        expected = {"llm_categories": []}
        assert result == expected

    def test_parse_text_generation_response_empty_input(self):
        """Test parse_text_generation_response with empty input."""
        result = ResponseParser.parse_text_generation_response(None)
        assert result is None

    def test_parse_text_generation_response_empty_list(self):
        """Test parse_text_generation_response with empty list."""
        result = ResponseParser.parse_text_generation_response([])
        assert result is None

    def test_parse_text_generation_response_dict_with_generated_text(self):
        """Test parse_text_generation_response with dict containing generated_text."""
        response = [{"generated_text": "This is generated text"}]
        result = ResponseParser.parse_text_generation_response(response)
        assert result == "This is generated text"

    def test_parse_text_generation_response_dict_with_prompt_removal(self):
        """Test parse_text_generation_response removes prompt when configured."""
        response = [{"generated_text": "Prompt text: This is the response"}]
        model_config = {"prompt_removal_strategy": "remove_prompt"}
        result = ResponseParser.parse_text_generation_response(response, "Prompt text:", model_config)
        assert result == "This is the response"

    def test_parse_text_generation_response_string_response(self):
        """Test parse_text_generation_response with string response."""
        response = ["Direct string response"]
        result = ResponseParser.parse_text_generation_response(response)
        assert result == "Direct string response"

    def test_parse_text_generation_response_exception_handling(self):
        """Test parse_text_generation_response handles exceptions gracefully."""
        # Use an input that will actually cause an exception in the parsing logic
        response = [{"invalid_key": "no_generated_text"}]
        result = ResponseParser.parse_text_generation_response(response)
        # The function falls back to str(response[0]) when expected keys are missing
        assert result == str(response[0])

    def test_parse_text2text_generation_response_empty_input(self):
        """Test parse_text2text_generation_response with empty input."""
        result = ResponseParser.parse_text2text_generation_response(None)
        assert result is None

    def test_parse_text2text_generation_response_dict_with_generated_text(self):
        """Test parse_text2text_generation_response with dict containing generated_text."""
        response = [{"generated_text": "T5 generated text"}]
        result = ResponseParser.parse_text2text_generation_response(response)
        assert result == "T5 generated text"

    def test_parse_text2text_generation_response_dict_with_text_key(self):
        """Test parse_text2text_generation_response with dict containing text key."""
        response = [{"text": "Alternative text key"}]
        result = ResponseParser.parse_text2text_generation_response(response)
        assert result == "Alternative text key"

    def test_parse_text2text_generation_response_string_response(self):
        """Test parse_text2text_generation_response with string response."""
        response = ["String response"]
        result = ResponseParser.parse_text2text_generation_response(response)
        assert result == "String response"

    def test_parse_text2text_generation_response_exception_handling(self):
        """Test parse_text2text_generation_response handles exceptions gracefully."""
        # Use an input that will actually cause an exception in the parsing logic
        response = [{"invalid_key": "no_generated_text"}]
        result = ResponseParser.parse_text2text_generation_response(response)
        # The function falls back to str(response[0]) when expected keys are missing
        assert result == str(response[0])

    def test_parse_response_classification_type(self):
        """Test parse_response with classification parser type."""
        response = [[{"label": "hate", "score": 0.8}]]
        result = ResponseParser.parse_response(response, "classification")
        assert Categories.HATE_SPEECH in result["llm_categories"]

    def test_parse_response_text_generation_type(self):
        """Test parse_response with text_generation parser type."""
        response = [{"generated_text": "Generated response"}]
        result = ResponseParser.parse_response(response, "text_generation")
        assert result == "Generated response"

    def test_parse_response_text2text_generation_type(self):
        """Test parse_response with text2text_generation parser type."""
        response = [{"generated_text": "T5 response"}]
        result = ResponseParser.parse_response(response, "text2text_generation")
        assert result == "T5 response"


class TestEnhancedLLMPipeline:
    """Test EnhancedLLMPipeline class."""

    def test_init_basic(self):
        """Test basic initialization."""
        pipeline = EnhancedLLMPipeline()
        assert pipeline.model_name is None
        assert pipeline.device in ["cpu", "cuda", "mps"]
        assert pipeline.is_loaded is False

    def test_init_with_model_name(self):
        """Test initialization with model name."""
        pipeline = EnhancedLLMPipeline(model_name="tiny-bert")
        assert pipeline.model_name == "tiny-bert"
        assert pipeline.model_config is not None
        assert pipeline.model_config["model_name"] == "huawei-noah/TinyBERT_General_4L_312D"

    def test_determine_device_cpu(self):
        """Test device determination on CPU."""
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=False):
            pipeline = EnhancedLLMPipeline()
            assert pipeline._determine_device() == "cpu"

    def test_determine_device_cuda(self):
        """Test device determination on CUDA."""
        with patch('torch.cuda.is_available', return_value=True):
            pipeline = EnhancedLLMPipeline()
            assert pipeline._determine_device() == "cuda"

    def test_determine_device_mps(self):
        """Test device determination on MPS."""
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=True):
            pipeline = EnhancedLLMPipeline()
            assert pipeline._determine_device() == "mps"