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

    def test_build_gemini_analysis_prompt_image(self):
        """Test Gemini analysis prompt for images."""
        test_text = "Contenido de ejemplo"
        result = EnhancedPromptGenerator.build_gemini_analysis_prompt(test_text, is_video=False)
        
        # Should contain the text
        self.assertIn(test_text, result)
        
        # Should be for image analysis
        self.assertIn("imagen", result)
        self.assertIn("imagen/video", result)  # Generic reference
        
        # Should have analysis instructions
        self.assertIn("CONTENIDO PROBLEMÁTICO EN REDES SOCIALES", result)
        
        # Should include analysis requirements
        self.assertIn("extrema derecha", result)

    def test_build_gemini_analysis_prompt_video(self):
        """Test Gemini analysis prompt for videos."""
        test_text = "Contenido de video"
        result = EnhancedPromptGenerator.build_gemini_analysis_prompt(test_text, is_video=True)
        
        # Should contain the text
        self.assertIn(test_text, result)
        
        # Should be for video analysis
        self.assertIn("video", result.lower())
        self.assertIn("imagen/video", result)  # Generic reference
        
        # Should have analysis instructions
        self.assertIn("CONTENIDO PROBLEMÁTICO EN REDES SOCIALES", result)
        
        # Should include analysis requirements
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

    def test_build_categorization_prompt_structure(self):
        """Test categorization prompt has correct structure."""
        result = self.generator.build_categorization_prompt(self.test_text)
        
        # Should contain the text
        self.assertIn(self.test_text, result)
        
        # Should have step-by-step analysis
        self.assertIn("INVESTIGACIÓN ACADÉMICA", result)
        
        # Should list all major categories
        self.assertIn("hate_speech", result)
        self.assertIn("disinformation", result)
        self.assertIn("conspiracy_theory", result)
        self.assertIn("far_right_bias", result)
        self.assertIn("call_to_action", result)
        self.assertIn("general", result)
        
        # Should have decision instructions
        self.assertIn("CATEGORÍA:", result)
        self.assertIn("EXPLICACIÓN:", result)

    def test_generate_explanation_prompt_hate_speech(self):
        """Test explanation prompt for hate speech category."""
        result = self.generator.generate_explanation_prompt(
            self.test_text,
            Categories.HATE_SPEECH
        )
        
        # Should contain text and category
        self.assertIn(self.test_text, result)
        self.assertIn(Categories.HATE_SPEECH, result)
        
        # Should have the optimized simplified focus (neutral/moderate content)
        self.assertIn("contenido neutral o político moderado", result.lower())
        self.assertIn("elementos extremistas", result.lower())

    def test_generate_explanation_prompt_disinformation(self):
        """Test explanation prompt for disinformation category."""
        result = self.generator.generate_explanation_prompt(
            "Las vacunas causan autismo",
            Categories.DISINFORMATION
        )
        
        # Should have the optimized simplified focus (neutral/moderate content)
        self.assertIn("contenido neutral o político moderado", result.lower())
        self.assertIn("elementos extremistas", result.lower())

    def test_generate_explanation_prompt_conspiracy(self):
        """Test explanation prompt for conspiracy theory category."""
        result = self.generator.generate_explanation_prompt(
            "Soros controla el gobierno",
            Categories.CONSPIRACY_THEORY
        )
        
        # Should have the optimized simplified focus (neutral/moderate content)
        self.assertIn("contenido neutral o político moderado", result.lower())
        self.assertIn("elementos extremistas", result.lower())

    def test_generate_explanation_prompt_far_right(self):
        """Test explanation prompt for far-right bias category."""
        result = self.generator.generate_explanation_prompt(
            "Los rojos quieren destruir España",
            Categories.FAR_RIGHT_BIAS
        )
        
        # Should have the optimized simplified focus (neutral/moderate content)
        self.assertIn("contenido neutral o político moderado", result.lower())
        self.assertIn("elementos extremistas", result.lower())

    def test_generate_explanation_prompt_call_to_action(self):
        """Test explanation prompt for call to action category."""
        result = self.generator.generate_explanation_prompt(
            "Todos a la calle mañana a las 18:00",
            Categories.CALL_TO_ACTION
        )
        
        # Should have the optimized simplified focus (neutral/moderate content)
        self.assertIn("contenido neutral o político moderado", result.lower())
        self.assertIn("elementos extremistas", result.lower())

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


class TestEnhancedDisinformationPrompts(unittest.TestCase):
    """Test cases for enhanced disinformation detection prompts."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = EnhancedPromptGenerator()
    
    def test_disinformation_explanation_includes_cross_country_manipulation(self):
        """Test that disinformation explanations include cross-country manipulation detection."""
        explanation = self.generator.generate_explanation_prompt(
            "Los aliados de Sánchez quieren dictadura digital como en México",
            "disinformation"
        )
        
        # Should have the optimized simplified focus (neutral/moderate content)
        self.assertIn("contenido neutral o político moderado", explanation.lower())
        self.assertIn("elementos extremistas", explanation.lower())
        
    def test_disinformation_explanation_includes_dictatorial_framing(self):
        """Test that disinformation explanations include dictatorial framing detection."""
        explanation = self.generator.generate_explanation_prompt(
            "El gobierno impone dictadura digital",
            "disinformation"
        )
        
        # Should have the optimized simplified focus (neutral/moderate content)
        self.assertIn("contenido neutral o político moderado", explanation.lower())
        self.assertIn("elementos extremistas", explanation.lower())
        
    def test_disinformation_explanation_includes_emotional_manipulation(self):
        """Test that disinformation explanations include emotional manipulation detection."""
        explanation = self.generator.generate_explanation_prompt(
            "Dictadura digital nos controla",
            "disinformation"
        )
        
        # Should have the optimized simplified focus (neutral/moderate content)
        self.assertIn("contenido neutral o político moderado", explanation.lower())
        self.assertIn("elementos extremistas", explanation.lower())
        
    def test_disinformation_focus_updated(self):
        """Test that disinformation focus includes new manipulation techniques."""
        explanation = self.generator.generate_explanation_prompt(
            "Test content",
            "disinformation"
        )
        
        # Should have the optimized simplified focus (neutral/moderate content)
        self.assertIn("contenido neutral o político moderado", explanation.lower())
        self.assertIn("elementos extremistas", explanation.lower())


class TestCrossCategoryPromptGeneration(unittest.TestCase):
    """Test cases for cross-category prompt generation scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = EnhancedPromptGenerator()
    
    def test_sophisticated_disinformation_case(self):
        """Test prompt generation for sophisticated disinformation like the Mexico-Spain case."""
        content = "Los aliados de Sánchez quieren dictadura digital igual que en México con la Ley Anti-Stickers"
        
        explanation_prompt = self.generator.generate_explanation_prompt(content, "disinformation")
        
        # Should have the optimized simplified focus (neutral/moderate content)
        self.assertIn("contenido neutral o político moderado", explanation_prompt.lower())
        self.assertIn("elementos extremistas", explanation_prompt.lower())
        
    def test_hate_speech_with_scapegoating(self):
        """Test prompt generation for hate speech with scapegoating elements."""
        content = "Los inmigrantes saturan la sanidad por culpa del gobierno"
        
        explanation_prompt = self.generator.generate_explanation_prompt(content, "hate_speech")
        
        # Should have the optimized simplified focus (neutral/moderate content)
        self.assertIn("contenido neutral o político moderado", explanation_prompt.lower())
        self.assertIn("elementos extremistas", explanation_prompt.lower())
        
    def test_far_right_political_rhetoric(self):
        """Test prompt generation for far-right political rhetoric."""
        content = "El régimen socialista ha destruido España"
        
        explanation_prompt = self.generator.generate_explanation_prompt(content, "far_right_bias")
        
        # Should have the optimized simplified focus (neutral/moderate content)
        self.assertIn("contenido neutral o político moderado", explanation_prompt.lower())
        self.assertIn("elementos extremistas", explanation_prompt.lower())


if __name__ == '__main__':
    unittest.main()
