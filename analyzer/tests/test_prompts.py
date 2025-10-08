#!/usr/bin/env python3
"""
Unit tests for Enhanced Prompt Generator.
Tests all static and instance methods for prompt generation.
"""

import unittest
from analyzer.prompts import EnhancedPromptGenerator, PromptContext
from analyzer.categories import Categories


class TestEnhancedPromptGeneratorStatic(unittest.TestCase):
    """Test cases for static methods of EnhancedPromptGenerator."""

    def test_build_category_list(self):
        """Test building dynamic category list."""
        result = EnhancedPromptGenerator.build_category_list()
        
        # Should be a comma-separated string
        self.assertIsInstance(result, str)
        self.assertIn(",", result)
        
        # Should contain all major categories
        self.assertIn("hate_speech", result)
        self.assertIn("disinformation", result)
        self.assertIn("conspiracy_theory", result)
        self.assertIn("far_right_bias", result)
        self.assertIn("call_to_action", result)
        self.assertIn("general", result)

    def test_build_ollama_system_prompt(self):
        """Test Ollama system prompt generation."""
        result = EnhancedPromptGenerator.build_ollama_system_prompt()
        
        # Should be a non-empty string
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 100)
        
        # Should contain key Spanish instructions
        self.assertIn("clasificador experto", result.lower())
        self.assertIn("español", result.lower())
        
        # Should mention detection categories
        self.assertIn("HATE_SPEECH", result)
        self.assertIn("DISINFORMATION", result)
        self.assertIn("CONSPIRACY_THEORY", result)
        self.assertIn("FAR_RIGHT_BIAS", result)
        self.assertIn("CALL_TO_ACTION", result)
        self.assertIn("GENERAL", result)
        
        # Should have detection guidelines
        self.assertIn("GUÍAS DE DETECCIÓN", result)
        self.assertIn("Identifica:", result)

    def test_build_generation_system_prompt(self):
        """Test generation model system prompt."""
        result = EnhancedPromptGenerator.build_generation_system_prompt()
        
        # Should be a non-empty string
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 100)
        
        # Should be in English (for generation models)
        self.assertIn("expert content classifier", result.lower())
        self.assertIn("spanish content", result.lower())
        
        # Should mention detection rules
        self.assertIn("DETECTION RULES", result)
        self.assertIn("HATE_SPEECH", result)
        self.assertIn("DISINFORMATION", result)

    def test_build_spanish_classification_prompt(self):
        """Test simple Spanish classification prompt."""
        test_text = "Este es un texto de prueba"
        result = EnhancedPromptGenerator.build_spanish_classification_prompt(test_text)
        
        # Should contain the text
        self.assertIn(test_text, result)
        
        # Should be in Spanish
        self.assertIn("Clasifica", result)
        self.assertIn("categorías", result)
        
        # Should mention category list
        self.assertIn("hate_speech", result)
        self.assertIn("general", result)

    def test_build_gemini_analysis_prompt_image(self):
        """Test Gemini analysis prompt for images."""
        test_text = "Contenido de ejemplo"
        result = EnhancedPromptGenerator.build_gemini_analysis_prompt(test_text, is_video=False)
        
        # Should contain the text
        self.assertIn(test_text, result)
        
        # Should be for image analysis
        self.assertIn("imagen", result)
        self.assertNotIn("video", result.lower().replace("videos", ""))  # Avoid false positive from "videos"
        
        # Should have image-specific instructions
        self.assertIn("Descripción detallada del contenido visual de la imagen", result)
        
        # Should mention Twitter/X context
        self.assertIn("Twitter/X", result)
        
        # Should include analysis requirements
        self.assertIn("ANÁLISIS REQUERIDO", result)
        self.assertIn("extrema derecha", result)

    def test_build_gemini_analysis_prompt_video(self):
        """Test Gemini analysis prompt for videos."""
        test_text = "Contenido de video"
        result = EnhancedPromptGenerator.build_gemini_analysis_prompt(test_text, is_video=True)
        
        # Should contain the text
        self.assertIn(test_text, result)
        
        # Should be for video analysis
        self.assertIn("video", result.lower())
        self.assertNotIn("imagen", result)
        
        # Should have video-specific instructions
        self.assertIn("Resumen del contenido visual del video", result)
        
        # Should include same analysis requirements
        self.assertIn("ANÁLISIS REQUERIDO", result)
        self.assertIn("extrema derecha", result)


class TestEnhancedPromptGeneratorInstance(unittest.TestCase):
    """Test cases for instance methods of EnhancedPromptGenerator."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = EnhancedPromptGenerator()
        self.test_text = "Los musulmanes están invadiendo España"

    def test_initialization(self):
        """Test proper initialization of generator."""
        self.assertIsNotNone(self.generator.base_context)
        self.assertIsNotNone(self.generator.prompt_templates)
        self.assertIsInstance(self.generator.prompt_templates, dict)
        
        # Should have templates for major categories
        self.assertIn(Categories.HATE_SPEECH, self.generator.prompt_templates)
        self.assertIn(Categories.DISINFORMATION, self.generator.prompt_templates)
        self.assertIn(Categories.GENERAL, self.generator.prompt_templates)

    def test_generate_classification_prompt_structure(self):
        """Test classification prompt has correct structure."""
        result = self.generator.generate_classification_prompt(self.test_text)
        
        # Should contain the text
        self.assertIn(self.test_text, result)
        
        # Should have step-by-step analysis
        self.assertIn("PROCESO DE ANÁLISIS PASO A PASO", result)
        
        # Should list all major categories with emoji indicators
        self.assertIn("1️⃣ HATE_SPEECH", result)
        self.assertIn("2️⃣ DISINFORMATION", result)
        self.assertIn("3️⃣ CONSPIRACY_THEORY", result)
        self.assertIn("4️⃣ FAR_RIGHT_BIAS", result)
        self.assertIn("5️⃣ CALL_TO_ACTION", result)
        self.assertIn("6️⃣ GENERAL", result)
        
        # Should have decision instructions
        self.assertIn("DECISIÓN", result)
        self.assertIn("RESPUESTA FINAL", result)

    def test_generate_classification_prompt_model_types(self):
        """Test classification prompt works for different model types."""
        # Test ollama
        result_ollama = self.generator.generate_classification_prompt(self.test_text, model_type="ollama")
        self.assertIn(self.test_text, result_ollama)
        
        # Test transformers
        result_transformers = self.generator.generate_classification_prompt(self.test_text, model_type="transformers")
        self.assertIn(self.test_text, result_transformers)
        
        # Both should have same core structure
        self.assertIn("PROCESO DE ANÁLISIS", result_ollama)
        self.assertIn("PROCESO DE ANÁLISIS", result_transformers)

    def test_generate_explanation_prompt_hate_speech(self):
        """Test explanation prompt for hate speech category."""
        result = self.generator.generate_explanation_prompt(
            self.test_text, 
            Categories.HATE_SPEECH
        )
        
        # Should contain text and category
        self.assertIn(self.test_text, result)
        self.assertIn(Categories.HATE_SPEECH, result)
        
        # Should have category-specific focus
        self.assertIn("odio", result.lower())
        self.assertIn("discriminación", result.lower())
        
        # Should have analysis questions
        self.assertIn("lenguaje específico", result.lower())
        self.assertIn("grupo", result.lower())

    def test_generate_explanation_prompt_disinformation(self):
        """Test explanation prompt for disinformation category."""
        result = self.generator.generate_explanation_prompt(
            "Las vacunas causan autismo",
            Categories.DISINFORMATION
        )
        
        # Should have disinformation-specific focus
        self.assertIn("afirmaciones falsas", result.lower())
        self.assertIn("médica", result.lower())
        self.assertIn("científica", result.lower())

    def test_generate_explanation_prompt_conspiracy(self):
        """Test explanation prompt for conspiracy theory category."""
        result = self.generator.generate_explanation_prompt(
            "Soros controla el gobierno",
            Categories.CONSPIRACY_THEORY
        )
        
        # Should have conspiracy-specific focus
        self.assertIn("teorías", result.lower())
        self.assertIn("control secreto", result.lower())
        self.assertIn("plan oculto", result.lower())

    def test_generate_explanation_prompt_far_right(self):
        """Test explanation prompt for far-right bias category."""
        result = self.generator.generate_explanation_prompt(
            "Los rojos quieren destruir España",
            Categories.FAR_RIGHT_BIAS
        )
        
        # Should have far-right specific focus
        self.assertIn("retórica extremista", result.lower())
        self.assertIn("nacionalismo", result.lower())
        self.assertIn("nosotros vs ellos", result.lower())

    def test_generate_explanation_prompt_call_to_action(self):
        """Test explanation prompt for call to action category."""
        result = self.generator.generate_explanation_prompt(
            "Todos a la calle mañana a las 18:00",
            Categories.CALL_TO_ACTION
        )
        
        # Should have call-to-action specific focus
        self.assertIn("movilización", result.lower())
        self.assertIn("acción colectiva", result.lower())

    def test_generate_explanation_prompt_general(self):
        """Test explanation prompt for general category."""
        result = self.generator.generate_explanation_prompt(
            "Hoy hace buen tiempo",
            Categories.GENERAL
        )
        
        # Should explain why it's general/neutral
        self.assertIn("neutral", result.lower())
        self.assertIn("moderado", result.lower())

    def test_generate_explanation_prompt_model_types(self):
        """Test explanation prompt works for different model types."""
        # Test ollama
        result_ollama = self.generator.generate_explanation_prompt(
            self.test_text, 
            Categories.HATE_SPEECH, 
            model_type="ollama"
        )
        self.assertIn(self.test_text, result_ollama)
        
        # Test transformers
        result_transformers = self.generator.generate_explanation_prompt(
            self.test_text, 
            Categories.HATE_SPEECH, 
            model_type="transformers"
        )
        self.assertIn(self.test_text, result_transformers)


class TestPromptContext(unittest.TestCase):
    """Test cases for PromptContext dataclass."""

    def test_prompt_context_creation(self):
        """Test creating PromptContext instance."""
        context = PromptContext(
            detected_categories=["hate_speech", "far_right_bias"],
            political_topic="inmigración",
            uncertainty_areas=["Nivel de amenaza", "Intención"]
        )
        
        self.assertEqual(len(context.detected_categories), 2)
        self.assertIn("hate_speech", context.detected_categories)
        self.assertEqual(context.political_topic, "inmigración")
        self.assertEqual(len(context.uncertainty_areas), 2)


if __name__ == '__main__':
    unittest.main()
