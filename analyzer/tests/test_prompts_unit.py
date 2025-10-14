"""Unit tests for analyzer/prompts.py EnhancedPromptGenerator static methods."""

import unittest
from analyzer.prompts import EnhancedPromptGenerator
from analyzer.categories import Categories, CATEGORY_INFO


class TestEnhancedPromptGenerator(unittest.TestCase):
    """Test cases for EnhancedPromptGenerator static methods."""

    def test_build_category_list(self):
        """Test build_category_list returns comma-separated category list."""
        result = EnhancedPromptGenerator.build_category_list()
        expected_categories = Categories.get_all_categories()
        expected = ", ".join(expected_categories)
        self.assertEqual(result, expected)
        # Ensure it's a non-empty string
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_build_ollama_system_prompt(self):
        """Test build_ollama_system_prompt includes all required elements."""
        result = EnhancedPromptGenerator.build_ollama_system_prompt()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

        # Check that it includes the expert classifier description
        self.assertIn("clasificador experto", result)
        self.assertIn("contenido problemático", result)

        # Check that it includes category list
        categories = EnhancedPromptGenerator.build_category_list()
        self.assertIn(categories, result)

        # Check that it includes detection guidelines
        self.assertIn("GUÍAS DE DETECCIÓN", result)
        self.assertIn("HATE_SPEECH", result)
        self.assertIn("DISINFORMATION", result)
        self.assertIn("CONSPIRACY_THEORY", result)
        self.assertIn("FAR_RIGHT_BIAS", result)
        self.assertIn("CALL_TO_ACTION", result)
        self.assertIn("GENERAL", result)

    def test_build_generation_system_prompt(self):
        """Test build_generation_system_prompt includes all required elements."""
        result = EnhancedPromptGenerator.build_generation_system_prompt()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

        # Check that it includes the expert classifier description
        self.assertIn("expert content classifier", result)
        self.assertIn("problematic Spanish content", result)

        # Check that it includes category list
        categories = EnhancedPromptGenerator.build_category_list()
        self.assertIn(categories, result)

        # Check that it includes detection rules
        self.assertIn("ENHANCED DETECTION RULES", result)
        self.assertIn("HATE_SPEECH:", result)
        self.assertIn("DISINFORMATION:", result)
        self.assertIn("CONSPIRACY_THEORY:", result)
        self.assertIn("FAR_RIGHT_BIAS:", result)
        self.assertIn("CALL_TO_ACTION:", result)
        self.assertIn("GENERAL:", result)

    def test_build_spanish_classification_prompt(self):
        """Test build_spanish_classification_prompt formats correctly."""
        test_text = "Este es un texto de prueba"
        result = EnhancedPromptGenerator.build_spanish_classification_prompt(test_text)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

        # Check that it includes the text
        self.assertIn(test_text, result)

        # Check that it includes category list
        categories = EnhancedPromptGenerator.build_category_list()
        self.assertIn(categories, result)

        # Check that it includes the instruction
        self.assertIn("Clasifica el siguiente texto", result)
        self.assertIn("Responde SOLO con el nombre", result)

    def test_build_gemini_analysis_prompt_text_only(self):
        """Test build_gemini_analysis_prompt for text-only content."""
        test_text = "Texto de prueba para análisis"
        result = EnhancedPromptGenerator.build_gemini_analysis_prompt(test_text, is_video=False)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

        # Check that it identifies as image analysis (but may mention video in questions)
        self.assertIn("imagen", result)
        # Note: The prompt mentions both image/video in questions, so we can't check for absence

        # Check that it includes the text
        self.assertIn(test_text, result)

        # Check that it includes analysis questions
        self.assertIn("ANÁLISIS DETALLADO", result)
        self.assertIn("elementos visuales específicos", result)

        # Check that it includes categories
        categories = EnhancedPromptGenerator.build_category_list()
        self.assertIn(categories, result)

        # Check format instructions
        self.assertIn("FORMATO REQUERIDO", result)
        self.assertIn("CATEGORÍA:", result)
        self.assertIn("EXPLICACIÓN:", result)

    def test_build_gemini_analysis_prompt_video(self):
        """Test build_gemini_analysis_prompt for video content."""
        test_text = "Texto de prueba para video"
        result = EnhancedPromptGenerator.build_gemini_analysis_prompt(test_text, is_video=True)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

        # Check that it identifies as video analysis (but may mention image in questions)
        self.assertIn("video", result)
        # Note: The prompt mentions both image/video in questions, so we can't check for absence

        # Check that it includes the text
        self.assertIn(test_text, result)

    def test_build_gemini_analysis_prompt_empty_text(self):
        """Test build_gemini_analysis_prompt with empty text."""
        result = EnhancedPromptGenerator.build_gemini_analysis_prompt("", is_video=False)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        # Should still include the analysis structure
        self.assertIn("ANÁLISIS DETALLADO", result)


class TestEnhancedPromptGeneratorInstance(unittest.TestCase):
    """Test cases for EnhancedPromptGenerator instance methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = EnhancedPromptGenerator()

    def test_init_creates_base_context(self):
        """Test that __init__ creates base Spanish context."""
        self.assertIsInstance(self.generator.base_context, str)
        self.assertGreater(len(self.generator.base_context), 0)
        self.assertIn("especializado", self.generator.base_context)

    def test_init_creates_prompt_templates(self):
        """Test that __init__ creates prompt templates."""
        self.assertIsInstance(self.generator.prompt_templates, dict)
        self.assertGreater(len(self.generator.prompt_templates), 0)

        # Check that all categories have templates
        for category_name in CATEGORY_INFO.keys():
            self.assertIn(category_name, self.generator.prompt_templates)
            template = self.generator.prompt_templates[category_name]
            self.assertIn("system", template)
            self.assertIn("focus", template)
            self.assertIn("questions", template)

    def test_generate_classification_prompt(self):
        """Test generate_classification_prompt creates detailed prompt."""
        test_text = "Texto de prueba para clasificación"
        result = self.generator.generate_classification_prompt(test_text)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

        # Check that it includes the text
        self.assertIn(test_text, result)

        # Check that it includes analysis steps
        self.assertIn("PROCESO DE ANÁLISIS PASO A PASO", result)
        self.assertIn("HATE_SPEECH - ¿Contiene el texto", result)
        self.assertIn("DISINFORMATION - ¿Presenta", result)
        self.assertIn("CONSPIRACY_THEORY - ¿Menciona", result)
        self.assertIn("FAR_RIGHT_BIAS - ¿Muestra", result)
        self.assertIn("CALL_TO_ACTION - ¿Incluye", result)
        self.assertIn("GENERAL - Solo si", result)

        # Check that it includes decision instructions
        self.assertIn("DECISIÓN:", result)
        self.assertIn("RESPUESTA FINAL", result)

    def test_generate_explanation_prompt_hate_speech(self):
        """Test generate_explanation_prompt for hate_speech category."""
        test_text = "Texto de prueba con odio"
        result = self.generator.generate_explanation_prompt(test_text, Categories.HATE_SPEECH)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

        # Check that it includes the text and category
        self.assertIn(test_text, result)
        self.assertIn(Categories.HATE_SPEECH, result)

        # Check that it includes hate speech focus
        self.assertIn("elementos de odio", result)
        self.assertIn("discriminación", result)

        # Check format instructions
        self.assertIn("INSTRUCCIONES DE FORMATO", result)
        self.assertIn("EXPLICACIÓN:", result)

    def test_generate_explanation_prompt_disinformation(self):
        """Test generate_explanation_prompt for disinformation category."""
        test_text = "Texto de prueba con desinformación"
        result = self.generator.generate_explanation_prompt(test_text, Categories.DISINFORMATION)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

        # Check that it includes disinformation focus
        self.assertIn("afirmaciones falsas", result)
        self.assertIn("datos manipulados", result)

    def test_generate_explanation_prompt_general(self):
        """Test generate_explanation_prompt for general category."""
        test_text = "Texto neutral"
        result = self.generator.generate_explanation_prompt(test_text, Categories.GENERAL)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

        # Check that it includes general focus
        self.assertIn("contenido neutral", result)

    def test_generate_explanation_prompt_unknown_category(self):
        """Test generate_explanation_prompt with unknown category falls back to general."""
        test_text = "Texto de prueba"
        result = self.generator.generate_explanation_prompt(test_text, "unknown_category")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        # Should use general category context
        self.assertIn("contenido neutral", result)


if __name__ == '__main__':
    unittest.main()