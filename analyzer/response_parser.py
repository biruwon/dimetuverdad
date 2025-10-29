"""
Response parsing utilities for different LLM pipeline types.

Handles parsing responses from classification, text generation, and text2text generation models.
"""

from .categories import Categories


class ResponseParser:
    """Handles different types of response parsing based on pipeline type."""

    @staticmethod
    def parse_classification_response(response, text_input="", model_config=None):
        """Parse classification pipeline responses."""
        result = {"llm_categories": []}

        if not response or not isinstance(response, list) or len(response) == 0:
            return result

        try:
            scores = response[0] if isinstance(response[0], list) else response

            # Find highest scoring negative class
            toxic_score = 0.0
            hate_score = 0.0
            positive_score = 0.0

            for item in scores:
                label = item.get('label', '').lower()
                score = item.get('score', 0.0)

                # Check for various negative indicators
                if any(word in label for word in ['toxic', 'hate', 'negative', '1', 'offensive', 'bad']):
                    toxic_score = max(toxic_score, score)
                if any(word in label for word in ['hate', 'odio', 'discrimin']):
                    hate_score = max(hate_score, score)
                if any(word in label for word in ['positive', '0', 'good', 'normal']):
                    positive_score = max(positive_score, score)

            # Determine final scores and categories
            final_score = max(toxic_score, hate_score)
            categories = []

            # Use lower thresholds for detection
            if hate_score > 0.3:
                categories.append(Categories.HATE_SPEECH)
            if toxic_score > 0.3:
                categories.append("toxic_content")
            if final_score < 0.3 and positive_score > 0.6:
                categories.append("normal_content")

            if not categories:
                categories = [Categories.GENERAL]

            return {
                "llm_categories": categories
            }
        except Exception as e:
            print(f"⚠️ Classification parsing error: {e}")
            return result

    @staticmethod
    def parse_text_generation_response(response, text_input="", model_config=None):
        """Parse text generation pipeline responses."""
        try:
            if not response or len(response) == 0:
                return None

            # Extract generated text
            if isinstance(response[0], dict) and 'generated_text' in response[0]:
                generated_text = response[0]['generated_text']

                # Remove prompt if configured to do so
                prompt_removal = model_config.get("prompt_removal_strategy") if model_config else None
                if prompt_removal == "remove_prompt" and text_input and text_input in generated_text:
                    generated_text = generated_text.replace(text_input, '').strip()

                return generated_text
            elif isinstance(response[0], str):
                return response[0]
            else:
                return str(response[0]) if response[0] else ""

        except Exception as e:
            print(f"⚠️ Text generation parsing error: {e}")
            return None

    @staticmethod
    def parse_text2text_generation_response(response, text_input="", model_config=None):
        """Parse text2text generation pipeline responses (T5, etc.)."""
        try:
            if not response or len(response) == 0:
                return None

            # T5 models don't include the prompt in response, so no removal needed
            if isinstance(response[0], dict):
                # Try different possible keys
                for key in ['generated_text', 'text', 'output']:
                    if key in response[0]:
                        return response[0][key]
                return str(response[0])
            elif isinstance(response[0], str):
                return response[0]
            else:
                return str(response[0]) if response[0] else ""

        except Exception as e:
            print(f"⚠️ Text2text generation parsing error: {e}")
            return None

    @classmethod
    def parse_response(cls, response, parser_type, text_input="", model_config=None):
        """Main method to parse responses based on parser type."""
        if parser_type == "classification":
            return cls.parse_classification_response(response, text_input, model_config)
        elif parser_type == "text_generation":
            return cls.parse_text_generation_response(response, text_input, model_config)
        elif parser_type == "text2text_generation":
            return cls.parse_text2text_generation_response(response, text_input, model_config)
        else:
            print(f"⚠️ Unknown parser type: {parser_type}")
            return None