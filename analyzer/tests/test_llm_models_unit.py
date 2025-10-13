"""Unit tests for analyzer/llm_models.py ResponseParser and core logic."""

import unittest
from unittest.mock import MagicMock
from analyzer.llm_models import ResponseParser
from analyzer.categories import Categories


class TestResponseParser(unittest.TestCase):
    """Test cases for ResponseParser static methods."""

    def test_parse_classification_response_empty_input(self):
        """Test parse_classification_response with empty input."""
        result = ResponseParser.parse_classification_response(None)
        expected = {"llm_categories": []}
        self.assertEqual(result, expected)

    def test_parse_classification_response_empty_list(self):
        """Test parse_classification_response with empty list."""
        result = ResponseParser.parse_classification_response([])
        expected = {"llm_categories": []}
        self.assertEqual(result, expected)

    def test_parse_classification_response_hate_speech_high_score(self):
        """Test parse_classification_response detects hate speech with high score."""
        response = [
            [
                {"label": "hate", "score": 0.8},
                {"label": "normal", "score": 0.2}
            ]
        ]
        result = ResponseParser.parse_classification_response(response)
        self.assertIn(Categories.HATE_SPEECH, result["llm_categories"])

    def test_parse_classification_response_toxic_content_high_score(self):
        """Test parse_classification_response detects toxic content with high score."""
        response = [
            [
                {"label": "toxic", "score": 0.7},
                {"label": "normal", "score": 0.3}
            ]
        ]
        result = ResponseParser.parse_classification_response(response)
        self.assertIn("toxic_content", result["llm_categories"])

    def test_parse_classification_response_normal_content_high_score(self):
        """Test parse_classification_response detects normal content."""
        response = [
            [
                {"label": "positive", "score": 0.8},
                {"label": "hate", "score": 0.1}
            ]
        ]
        result = ResponseParser.parse_classification_response(response)
        self.assertIn("normal_content", result["llm_categories"])

    def test_parse_classification_response_low_scores_fallback_to_general(self):
        """Test parse_classification_response falls back to general with low scores."""
        response = [
            [
                {"label": "hate", "score": 0.2},
                {"label": "normal", "score": 0.4}
            ]
        ]
        result = ResponseParser.parse_classification_response(response)
        self.assertEqual(result["llm_categories"], [Categories.GENERAL])

    def test_parse_classification_response_exception_handling(self):
        """Test parse_classification_response handles exceptions gracefully."""
        response = "invalid_response_format"
        result = ResponseParser.parse_classification_response(response)
        expected = {"llm_categories": []}
        self.assertEqual(result, expected)

    def test_parse_text_generation_response_empty_input(self):
        """Test parse_text_generation_response with empty input."""
        result = ResponseParser.parse_text_generation_response(None)
        self.assertIsNone(result)

    def test_parse_text_generation_response_empty_list(self):
        """Test parse_text_generation_response with empty list."""
        result = ResponseParser.parse_text_generation_response([])
        self.assertIsNone(result)

    def test_parse_text_generation_response_dict_with_generated_text(self):
        """Test parse_text_generation_response with dict containing generated_text."""
        response = [{"generated_text": "This is generated text"}]
        result = ResponseParser.parse_text_generation_response(response)
        self.assertEqual(result, "This is generated text")

    def test_parse_text_generation_response_dict_with_prompt_removal(self):
        """Test parse_text_generation_response removes prompt when configured."""
        response = [{"generated_text": "Prompt text: This is the response"}]
        model_config = {"prompt_removal_strategy": "remove_prompt"}
        result = ResponseParser.parse_text_generation_response(response, "Prompt text:", model_config)
        self.assertEqual(result, "This is the response")

    def test_parse_text_generation_response_string_response(self):
        """Test parse_text_generation_response with string response."""
        response = ["Direct string response"]
        result = ResponseParser.parse_text_generation_response(response)
        self.assertEqual(result, "Direct string response")

    def test_parse_text_generation_response_exception_handling(self):
        """Test parse_text_generation_response handles exceptions gracefully."""
        # Use an input that will actually cause an exception in the parsing logic
        response = [{"invalid_key": "no_generated_text"}]
        result = ResponseParser.parse_text_generation_response(response)
        # The function falls back to str(response[0]) when expected keys are missing
        self.assertEqual(result, str(response[0]))

    def test_parse_text2text_generation_response_empty_input(self):
        """Test parse_text2text_generation_response with empty input."""
        result = ResponseParser.parse_text2text_generation_response(None)
        self.assertIsNone(result)

    def test_parse_text2text_generation_response_dict_with_generated_text(self):
        """Test parse_text2text_generation_response with dict containing generated_text."""
        response = [{"generated_text": "T5 generated text"}]
        result = ResponseParser.parse_text2text_generation_response(response)
        self.assertEqual(result, "T5 generated text")

    def test_parse_text2text_generation_response_dict_with_text_key(self):
        """Test parse_text2text_generation_response with dict containing text key."""
        response = [{"text": "Alternative text key"}]
        result = ResponseParser.parse_text2text_generation_response(response)
        self.assertEqual(result, "Alternative text key")

    def test_parse_text2text_generation_response_string_response(self):
        """Test parse_text2text_generation_response with string response."""
        response = ["String response"]
        result = ResponseParser.parse_text2text_generation_response(response)
        self.assertEqual(result, "String response")

    def test_parse_text2text_generation_response_exception_handling(self):
        """Test parse_text2text_generation_response handles exceptions gracefully."""
        # Use an input that will actually cause an exception in the parsing logic
        response = [{"invalid_key": "no_generated_text"}]
        result = ResponseParser.parse_text2text_generation_response(response)
        # The function falls back to str(response[0]) when expected keys are missing
        self.assertEqual(result, str(response[0]))

    def test_parse_response_classification_type(self):
        """Test parse_response with classification parser type."""
        response = [[{"label": "hate", "score": 0.8}]]
        result = ResponseParser.parse_response(response, "classification")
        self.assertIn(Categories.HATE_SPEECH, result["llm_categories"])

    def test_parse_response_text_generation_type(self):
        """Test parse_response with text_generation parser type."""
        response = [{"generated_text": "Generated response"}]
        result = ResponseParser.parse_response(response, "text_generation")
        self.assertEqual(result, "Generated response")

    def test_parse_response_text2text_generation_type(self):
        """Test parse_response with text2text_generation parser type."""
        response = [{"generated_text": "T5 response"}]
        result = ResponseParser.parse_response(response, "text2text_generation")
        self.assertEqual(result, "T5 response")

    def test_parse_response_unknown_type(self):
        """Test parse_response with unknown parser type."""
        response = [{"data": "test"}]
        result = ResponseParser.parse_response(response, "unknown_type")
        self.assertIsNone(result)


class TestLLMModelConfig(unittest.TestCase):
    """Test cases for LLMModelConfig class."""

    def test_get_recommended_model_generation_speed(self):
        """Test get_recommended_model returns valid config for generation speed."""
        from analyzer.llm_models import LLMModelConfig
        config = LLMModelConfig.get_recommended_model("generation", "speed")
        self.assertIsInstance(config, dict)
        self.assertEqual(config["task_type"], "generation")
        self.assertIn(config["speed"], ["ultra_fast", "very_fast", "fast"])

    def test_get_recommended_model_generation_quality(self):
        """Test get_recommended_model returns valid config for generation quality."""
        from analyzer.llm_models import LLMModelConfig
        config = LLMModelConfig.get_recommended_model("generation", "quality")
        self.assertIsInstance(config, dict)
        self.assertEqual(config["task_type"], "generation")

    def test_get_recommended_model_generation_balanced(self):
        """Test get_recommended_model returns valid config for generation balanced."""
        from analyzer.llm_models import LLMModelConfig
        config = LLMModelConfig.get_recommended_model("generation", "balanced")
        self.assertIsInstance(config, dict)
        self.assertEqual(config["task_type"], "generation")

    def test_get_recommended_model_classification_speed(self):
        """Test get_recommended_model returns valid config for classification speed."""
        from analyzer.llm_models import LLMModelConfig
        config = LLMModelConfig.get_recommended_model("classification", "speed")
        self.assertIsInstance(config, dict)
        self.assertEqual(config["task_type"], "classification")

    def test_get_recommended_model_classification_quality(self):
        """Test get_recommended_model returns valid config for classification quality."""
        from analyzer.llm_models import LLMModelConfig
        config = LLMModelConfig.get_recommended_model("classification", "quality")
        self.assertIsInstance(config, dict)
        self.assertEqual(config["task_type"], "classification")

    def test_get_recommended_model_classification_balanced(self):
        """Test get_recommended_model returns valid config for classification balanced."""
        from analyzer.llm_models import LLMModelConfig
        config = LLMModelConfig.get_recommended_model("classification", "balanced")
        self.assertIsInstance(config, dict)
        self.assertEqual(config["task_type"], "classification")

    def test_get_recommended_model_unknown_priority(self):
        """Test get_recommended_model handles unknown priority gracefully."""
        from analyzer.llm_models import LLMModelConfig
        config = LLMModelConfig.get_recommended_model("generation", "unknown")
        self.assertIsInstance(config, dict)
        # Should still return a valid config (defaults to balanced behavior)


if __name__ == '__main__':
    unittest.main()