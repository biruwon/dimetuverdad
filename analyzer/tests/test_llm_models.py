"""
Tests for analyzer/llm_models.py
Comprehensive test coverage for LLM models and pipeline functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
from analyzer.llm_models import (
    LLMModelConfig,
    ResponseParser,
    EnhancedLLMPipeline
)
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
            assert config["speed"] in ["ultra_fast", "very_fast"]

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
            assert LLMModelConfig.MODELS[model_name]["language"] == "spanish"

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


class TestEnhancedLLMPipeline:
    """Test EnhancedLLMPipeline class."""

    @patch('analyzer.llm_models.pipeline')
    @patch('analyzer.llm_models.torch.cuda.is_available', return_value=False)
    @patch('analyzer.llm_models.torch.backends.mps.is_available', return_value=False)
    def test_init_cpu_device(self, mock_mps, mock_cuda, mock_pipeline):
        """Test initialization on CPU device."""
        mock_pipeline.return_value = Mock()
        pipeline = EnhancedLLMPipeline()
        assert pipeline.device == "cpu"
        assert pipeline.model_priority == "balanced"

    @patch('analyzer.llm_models.pipeline')
    @patch('analyzer.llm_models.torch.cuda.is_available', return_value=True)
    def test_init_cuda_device(self, mock_cuda, mock_pipeline):
        """Test initialization on CUDA device."""
        mock_pipeline.return_value = Mock()
        pipeline = EnhancedLLMPipeline()
        assert pipeline.device == "cuda"

    @patch('analyzer.llm_models.pipeline')
    @patch('analyzer.llm_models.torch.backends.mps.is_available', return_value=True)
    @patch('analyzer.llm_models.torch.cuda.is_available', return_value=False)
    def test_init_mps_device(self, mock_cuda, mock_mps, mock_pipeline):
        """Test initialization on MPS device."""
        mock_pipeline.return_value = Mock()
        pipeline = EnhancedLLMPipeline()
        assert pipeline.device == "mps"

    @patch('analyzer.llm_models.pipeline')
    def test_init_with_specific_models(self, mock_pipeline):
        """Test initialization with specific model overrides."""
        mock_pipeline.return_value = Mock()
        specific_models = {
            "generation": "flan-t5-small",
            "classification": "distilbert-multilingual"
        }
        pipeline = EnhancedLLMPipeline(specific_models=specific_models)
        assert pipeline.specific_models == specific_models

    @patch('analyzer.llm_models.pipeline')
    def test_load_generation_model_ollama(self, mock_pipeline):
        """Test loading Ollama generation model."""
        from unittest.mock import patch as mock_patch
        with mock_patch('analyzer.llm_models.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            pipeline = EnhancedLLMPipeline()
            pipeline._load_generation_model(LLMModelConfig.MODELS["gpt-oss-20b"])

            assert pipeline.ollama_client == mock_client
            assert pipeline.ollama_model_name == "gpt-oss:20b"
            assert pipeline.generation_model == "ollama"

    @patch('analyzer.llm_models.pipeline')
    def test_load_generation_model_transformers(self, mock_pipeline):
        """Test loading transformers generation model."""
        mock_gen_model = Mock()
        mock_class_model = Mock()
        mock_pipeline.side_effect = [mock_class_model, mock_gen_model]  # classification first, then generation

        pipeline = EnhancedLLMPipeline()
        pipeline._load_generation_model(LLMModelConfig.MODELS["flan-t5-small"])

        # Should be called twice: once for classification (during init), once for generation
        assert mock_pipeline.call_count == 2
        assert pipeline.generation_model == mock_gen_model
        assert pipeline.generation_model == mock_gen_model

    @patch('analyzer.llm_models.pipeline')
    def test_load_classification_model(self, mock_pipeline):
        """Test loading classification model."""
        mock_gen_model = Mock()
        mock_class_model = Mock()
        mock_pipeline.side_effect = [mock_gen_model, mock_class_model]  # generation first, then classification

        pipeline = EnhancedLLMPipeline()
        pipeline._load_classification_model(LLMModelConfig.MODELS["distilbert-multilingual"])

        # Should be called twice: once for generation (during init), once for classification
        assert mock_pipeline.call_count == 2
        assert pipeline.classification_model == mock_class_model

    @patch('analyzer.llm_models.pipeline')
    def test_get_category_with_ollama(self, mock_pipeline):
        """Test get_category method using Ollama."""
        from unittest.mock import patch as mock_patch

        # Mock Ollama client
        with mock_patch('analyzer.llm_models.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            # Mock Ollama response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "hate_speech"
            mock_client.chat.completions.create.return_value = mock_response

            pipeline = EnhancedLLMPipeline()
            pipeline._load_generation_model(LLMModelConfig.MODELS["gpt-oss-20b"])

            result = pipeline.get_category("test text")
            assert isinstance(result, str)

    @patch('analyzer.llm_models.pipeline')
    def test_get_category_with_generation_model(self, mock_pipeline):
        """Test get_category method using generation model."""
        mock_gen_model = Mock()
        mock_pipeline.return_value = mock_gen_model

        # Mock generation response
        mock_gen_model.return_value = [{"generated_text": "hate_speech"}]

        pipeline = EnhancedLLMPipeline()
        pipeline._load_generation_model(LLMModelConfig.MODELS["flan-t5-small"])

        result = pipeline.get_category("test text")
        assert isinstance(result, str)

    @patch('analyzer.llm_models.pipeline')
    def test_get_explanation_with_ollama(self, mock_pipeline):
        """Test get_explanation method using Ollama."""
        from unittest.mock import patch as mock_patch

        with mock_patch('analyzer.llm_models.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            # Mock Ollama response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "This is a detailed explanation."
            mock_client.chat.completions.create.return_value = mock_response

            pipeline = EnhancedLLMPipeline()
            pipeline._load_generation_model(LLMModelConfig.MODELS["gpt-oss-20b"])

            result = pipeline.get_explanation("test text", "hate_speech")
            assert isinstance(result, str)
            assert len(result) > 0

    @patch('analyzer.llm_models.pipeline')
    def test_get_explanation_with_transformers(self, mock_pipeline):
        """Test get_explanation method using transformers."""
        mock_gen_model = Mock()
        mock_pipeline.return_value = mock_gen_model

        # Mock transformers response
        mock_gen_model.return_value = [{"generated_text": "This is an explanation."}]

        pipeline = EnhancedLLMPipeline()
        pipeline._load_generation_model(LLMModelConfig.MODELS["flan-t5-small"])

        result = pipeline.get_explanation("test text", "hate_speech")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_explanation_no_generation_model(self):
        """Test get_explanation when no generation model is available."""
        pipeline = EnhancedLLMPipeline()
        pipeline.generation_model = None

        result = pipeline.get_explanation("test text", "hate_speech")
        assert "ERROR" in result
        assert "Generation model not available" in result

    @patch('analyzer.llm_models.torch.cuda.empty_cache')
    @patch('analyzer.llm_models.torch.cuda.synchronize')
    @patch('analyzer.llm_models.torch.cuda.is_available', return_value=True)
    def test_cleanup_memory_cuda(self, mock_cuda_available, mock_sync, mock_empty_cache):
        """Test memory cleanup on CUDA."""
        pipeline = EnhancedLLMPipeline()
        pipeline.cleanup_memory()

        mock_empty_cache.assert_called_once()
        mock_sync.assert_called_once()

    @patch('analyzer.llm_models.torch.cuda.is_available', return_value=False)
    def test_cleanup_memory_cpu(self, mock_cuda_available):
        """Test memory cleanup on CPU."""
        pipeline = EnhancedLLMPipeline()
        pipeline.cleanup_memory()

        # Should not raise any errors

    def test_get_model_info(self):
        """Test get_model_info method."""
        pipeline = EnhancedLLMPipeline()
        info = pipeline.get_model_info()

        assert isinstance(info, dict)
        assert "device" in info
        assert "generation_model" in info
        assert "classification_model" in info
        assert "quantization_enabled" in info
        assert "model_configs" in info

    @patch('analyzer.llm_models.pipeline')
    def test_load_models_exception_handling(self, mock_pipeline):
        """Test exception handling in _load_models."""
        mock_pipeline.side_effect = Exception("Model loading failed")

        pipeline = EnhancedLLMPipeline()
        # Should not raise exception, should set models to None
        assert pipeline.generation_model is None
        assert pipeline.classification_model is None

    @patch('analyzer.llm_models.pipeline')
    def test_load_generation_model_exception_handling(self, mock_pipeline):
        """Test exception handling in _load_generation_model."""
        mock_pipeline.side_effect = Exception("Generation model failed")

        pipeline = EnhancedLLMPipeline()
        with pytest.raises(Exception):
            pipeline._load_generation_model(LLMModelConfig.MODELS["flan-t5-small"])

    @patch('analyzer.llm_models.pipeline')
    def test_load_classification_model_exception_handling(self, mock_pipeline):
        """Test exception handling in _load_classification_model."""
        mock_pipeline.side_effect = Exception("Classification model failed")

        pipeline = EnhancedLLMPipeline()
        with pytest.raises(Exception):
            pipeline._load_classification_model(LLMModelConfig.MODELS["distilbert-multilingual"])

    def test_classify_with_ollama_exception_handling(self):
        """Test exception handling in _classify_with_ollama."""
        pipeline = EnhancedLLMPipeline()
        pipeline.ollama_client = Mock()
        pipeline.ollama_client.chat.completions.create.side_effect = Exception("Ollama error")

        result = pipeline._classify_with_ollama("test text")
        assert result == {"llm_categories": []}

    @patch('analyzer.llm_models.pipeline')
    def test_classify_with_generation_model_exception_handling(self, mock_pipeline):
        """Test exception handling in _classify_with_generation_model."""
        mock_gen_model = Mock()
        mock_pipeline.return_value = mock_gen_model
        mock_gen_model.side_effect = Exception("Generation error")

        pipeline = EnhancedLLMPipeline()
        pipeline._load_generation_model(LLMModelConfig.MODELS["flan-t5-small"])

        result = pipeline._classify_with_generation_model("test text")
        assert result == Categories.GENERAL

    def test_generate_explanation_with_specific_prompt_exception_handling(self):
        """Test exception handling in _generate_explanation_with_specific_prompt."""
        pipeline = EnhancedLLMPipeline()
        pipeline.generation_model = None

        result = pipeline._generate_explanation_with_specific_prompt("test text", "hate_speech")
        assert "ERROR" in result
        assert "No generation model available" in result

    @patch('analyzer.llm_models.pipeline')
    def test_get_category_fallback_to_general(self, mock_pipeline):
        """Test get_category fallback to general category."""
        mock_gen_model = Mock()
        mock_pipeline.return_value = mock_gen_model
        mock_gen_model.return_value = [{"generated_text": "invalid_category"}]

        pipeline = EnhancedLLMPipeline()
        pipeline._load_generation_model(LLMModelConfig.MODELS["flan-t5-small"])

        result = pipeline.get_category("test text")
        assert result == Categories.GENERAL

    def test_get_category_exception_handling(self):
        """Test exception handling in get_category."""
        pipeline = EnhancedLLMPipeline()
        pipeline.generation_model = Mock()
        pipeline.generation_model.side_effect = Exception("Unexpected error")

        result = pipeline.get_category("test text")
        assert result == Categories.GENERAL

    def test_get_explanation_exception_handling(self):
        """Test exception handling in get_explanation."""
        pipeline = EnhancedLLMPipeline()
        pipeline.generation_model = Mock()
        pipeline.generation_model.side_effect = Exception("Unexpected error")

        result = pipeline.get_explanation("test text", "hate_speech")
        assert "ERROR" in result
        assert "Exception in explanation prompt generation" in result

    @patch('analyzer.llm_models.pipeline')
    def test_ollama_model_info(self, mock_pipeline):
        """Test model info with Ollama model."""
        from unittest.mock import patch as mock_patch

        with mock_patch('analyzer.llm_models.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            pipeline = EnhancedLLMPipeline()
            pipeline._load_generation_model(LLMModelConfig.MODELS["gpt-oss-20b"])

            info = pipeline.get_model_info()
            assert info["generation_model"] == "loaded"