#!/usr/bin/env python3
"""
Unit tests for Enhanced Prompt Generator.
Tests all static and instance methods for prompt generation.
"""

import unittest
from analyzer.prompts import EnhancedPromptGenerator, PromptContext
from analyzer.categories import Categories, CATEGORY_INFO


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
        self.assertIn("anti_immigration", result)
        self.assertIn("anti_lgbtq", result)
        self.assertIn("anti_feminism", result)
        self.assertIn("disinformation", result)
        self.assertIn("conspiracy_theory", result)
        self.assertIn("call_to_action", result)
        self.assertIn("general", result)

    def test_build_ollama_text_analysis_system_prompt(self):
        """Test Ollama system prompt generation."""
        result = EnhancedPromptGenerator.build_ollama_text_analysis_system_prompt()
        
        # Should be a non-empty string
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 100)
        
        # Should contain key Spanish instructions
        self.assertIn("clasificador experto", result.lower())
        self.assertIn("español", result.lower())
        
        # Should mention detection categories
        self.assertIn("HATE_SPEECH", result)
        self.assertIn("ANTI_IMMIGRATION", result)
        self.assertIn("ANTI_LGBTQ", result)
        self.assertIn("ANTI_FEMINISM", result)
        self.assertIn("DISINFORMATION", result)
        self.assertIn("CONSPIRACY_THEORY", result)
        self.assertIn("**call_to_action**", result)
        self.assertIn("GENERAL", result)
        
        # Should have detection guidelines
        self.assertIn("GUÍAS DE DETECCIÓN", result)
        self.assertIn("Identifica:", result)

    def test_build_gemini_multimodal_analysis_prompt_image(self):
        """Test Gemini analysis prompt for images."""
        test_text = "Contenido de ejemplo"
        result = EnhancedPromptGenerator.build_gemini_multimodal_analysis_prompt(test_text, is_video=False)
        
        # Should contain the text
        self.assertIn(test_text, result)
        
        # Should be for image analysis
        self.assertIn("imagen", result)
        self.assertIn("imagen/video", result)  # Generic reference
        
        # Should have analysis instructions
        self.assertIn("OBJETIVO DE INVESTIGACIÓN", result)
        
        # Should include analysis requirements
        self.assertIn("extrema derecha", result)

    def test_build_gemini_multimodal_analysis_prompt_video(self):
        """Test Gemini analysis prompt for videos."""
        test_text = "Contenido de video"
        result = EnhancedPromptGenerator.build_gemini_multimodal_analysis_prompt(test_text, is_video=True)
        
        # Should contain the text
        self.assertIn(test_text, result)
        
        # Should be for video analysis
        self.assertIn("video", result.lower())
        self.assertIn("imagen/video", result)  # Generic reference
        
        # Should have analysis instructions
        self.assertIn("OBJETIVO DE INVESTIGACIÓN", result)
        
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

    def test_build_categorization_prompt_structure(self):
        """Test categorization prompt has correct structure."""
        result = self.generator.build_ollama_categorization_prompt(self.test_text)
        
        # Should contain the text
        self.assertIn(self.test_text, result)
        
        # Should have step-by-step analysis
        self.assertIn("INVESTIGACIÓN ACADÉMICA", result)
        
        # Should list all major categories
        self.assertIn("hate_speech", result)
        self.assertIn("anti_immigration", result)
        self.assertIn("anti_lgbtq", result)
        self.assertIn("anti_feminism", result)
        self.assertIn("disinformation", result)
        self.assertIn("conspiracy_theory", result)
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
        
        # Should have the category-specific explanation structure
        self.assertIn("lenguaje de odio", result.lower())
        self.assertIn("grupos específicos", result.lower())

    def test_generate_explanation_prompt_disinformation(self):
        """Test explanation prompt for disinformation category."""
        result = self.generator.generate_explanation_prompt(
            "Las vacunas causan autismo",
            Categories.DISINFORMATION
        )
        
        # Should have the category-specific explanation structure for disinformation
        self.assertIn("desinformación", result.lower())
        self.assertIn("hechos verificables", result.lower())

    def test_generate_explanation_prompt_conspiracy(self):
        """Test explanation prompt for conspiracy theory category."""
        result = self.generator.generate_explanation_prompt(
            "Soros controla el gobierno",
            Categories.CONSPIRACY_THEORY
        )
        
        # Should have the category-specific explanation structure for conspiracy
        self.assertIn("teoría conspirativa", result.lower())
        self.assertIn("agenda secreta", result.lower())

    def test_generate_explanation_prompt_anti_government(self):
        """Test explanation prompt for anti-government category."""
        result = self.generator.generate_explanation_prompt(
            "Los rojos quieren destruir España",
            Categories.ANTI_GOVERNMENT
        )
        
        # Should have the category-specific explanation structure for anti-government
        self.assertIn("anti-gubernamental", result.lower())
        self.assertIn("legitimidad del gobierno", result.lower())

    def test_generate_explanation_prompt_anti_immigration(self):
        """Test explanation prompt for anti-immigration category."""
        result = self.generator.generate_explanation_prompt(
            "Los inmigrantes están invadiendo nuestro país",
            Categories.ANTI_IMMIGRATION
        )
        
        # Should have the category-specific explanation structure for anti-immigration
        self.assertIn("retórica anti-inmigración", result.lower())
        self.assertIn("amenaza existencial", result.lower())

    def test_generate_explanation_prompt_anti_lgbtq(self):
        """Test explanation prompt for anti-LGBTQ category."""
        result = self.generator.generate_explanation_prompt(
            "La ideología de género adoctrina a los niños",
            Categories.ANTI_LGBTQ
        )
        
        # Should have the category-specific explanation structure for anti-LGBTQ
        self.assertIn("comunidad lgbtq", result.lower())
        self.assertIn("diversidad de género", result.lower())

    def test_generate_explanation_prompt_anti_feminism(self):
        """Test explanation prompt for anti-feminism category."""
        result = self.generator.generate_explanation_prompt(
            "Las feminazis quieren destruir la familia tradicional",
            Categories.ANTI_FEMINISM
        )
        
        # Should have the category-specific explanation structure for anti-feminism
        self.assertIn("feminista", result.lower())
        self.assertIn("igualdad de género", result.lower())

    def test_generate_explanation_prompt_nationalism(self):
        """Test explanation prompt for nationalism category."""
        result = self.generator.generate_explanation_prompt(
            "España es superior a otros países",
            Categories.NATIONALISM
        )
        
        # Should have the category-specific explanation structure for nationalism
        self.assertIn("nacionalismo", result.lower())
        self.assertIn("orgullo nacional", result.lower())

    def test_generate_explanation_prompt_historical_revisionism(self):
        """Test explanation prompt for historical revisionism category."""
        result = self.generator.generate_explanation_prompt(
            "Franco salvó España de los comunistas",
            Categories.HISTORICAL_REVISIONISM
        )
        
        # Should have the category-specific explanation structure for historical revisionism
        self.assertIn("revisionismo histórico", result.lower())
        self.assertIn("regímenes autoritarios", result.lower())

    def test_generate_explanation_prompt_political_general(self):
        """Test explanation prompt for political general category."""
        result = self.generator.generate_explanation_prompt(
            "El gobierno debería bajar los impuestos",
            Categories.POLITICAL_GENERAL
        )
        
        # Should have the category-specific explanation structure for political general
        self.assertIn("político general", result.lower())
        self.assertIn("perspectivas políticas moderadas", result.lower())

    def test_generate_explanation_prompt_call_to_action(self):
        """Test explanation prompt for call to action category."""
        result = self.generator.generate_explanation_prompt(
            "Todos a la calle mañana a las 18:00",
            Categories.CALL_TO_ACTION
        )
        
        # Should have the category-specific explanation structure for call to action
        self.assertIn("incita a la acción", result.lower())
        self.assertIn("acción específica", result.lower())

    def test_generate_explanation_prompt_general(self):
        """Test explanation prompt for general category."""
        result = self.generator.generate_explanation_prompt(
            "Hoy hace buen tiempo",
            Categories.GENERAL
        )
        
        # Should explain why it's general/neutral
        self.assertIn("contenido neutral", result.lower())
        self.assertIn("elementos extremistas", result.lower())

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
            detected_categories=["hate_speech", "anti_government"],
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
        
        # Should have the category-specific explanation structure for disinformation
        self.assertIn("desinformación", explanation.lower())
        self.assertIn("hechos verificables", explanation.lower())
        
    def test_disinformation_explanation_includes_dictatorial_framing(self):
        """Test that disinformation explanations include dictatorial framing detection."""
        explanation = self.generator.generate_explanation_prompt(
            "El gobierno impone dictadura digital",
            "disinformation"
        )
        
        # Should have the category-specific explanation structure for disinformation
        self.assertIn("desinformación", explanation.lower())
        self.assertIn("hechos verificables", explanation.lower())
        
    def test_disinformation_explanation_includes_emotional_manipulation(self):
        """Test that disinformation explanations include emotional manipulation detection."""
        explanation = self.generator.generate_explanation_prompt(
            "Dictadura digital nos controla",
            "disinformation"
        )
        
        # Should have the category-specific explanation structure for disinformation
        self.assertIn("desinformación", explanation.lower())
        self.assertIn("hechos verificables", explanation.lower())
        
    def test_disinformation_focus_updated(self):
        """Test that disinformation focus includes new manipulation techniques."""
        explanation = self.generator.generate_explanation_prompt(
            "Test content",
            "disinformation"
        )
        
        # Should have the category-specific explanation structure for disinformation
        self.assertIn("desinformación", explanation.lower())
        self.assertIn("hechos verificables", explanation.lower())


class TestCrossCategoryPromptGeneration(unittest.TestCase):
    """Test cases for cross-category prompt generation scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = EnhancedPromptGenerator()
    
    def test_sophisticated_disinformation_case(self):
        """Test prompt generation for sophisticated disinformation like the Mexico-Spain case."""
        content = "Los aliados de Sánchez quieren dictadura digital igual que en México con la Ley Anti-Stickers"
        
        explanation_prompt = self.generator.generate_explanation_prompt(content, "disinformation")
        
        # Should have the category-specific explanation structure for disinformation
        self.assertIn("desinformación", explanation_prompt.lower())
        self.assertIn("hechos verificables", explanation_prompt.lower())
        
    def test_hate_speech_with_scapegoating(self):
        """Test prompt generation for hate speech with scapegoating elements."""
        content = "Los inmigrantes saturan la sanidad por culpa del gobierno"
        
        explanation_prompt = self.generator.generate_explanation_prompt(content, "hate_speech")
        
        # Should have the category-specific explanation structure for hate speech
        self.assertIn("lenguaje de odio", explanation_prompt.lower())
        self.assertIn("grupos específicos", explanation_prompt.lower())
        
    def test_anti_government_political_rhetoric(self):
        """Test prompt generation for anti-government political rhetoric."""
        content = "El régimen socialista ha destruido España"
        
        explanation_prompt = self.generator.generate_explanation_prompt(content, "anti_government")
        
        # Should have the category-specific explanation structure for anti-government
        self.assertIn("anti-gubernamental", explanation_prompt.lower())
        self.assertIn("legitimidad del gobierno", explanation_prompt.lower())


if __name__ == '__main__':
    unittest.main()
