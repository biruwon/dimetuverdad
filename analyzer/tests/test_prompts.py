#!/usr/bin/env python3
"""
Unit tests for Enhanced Prompt Generator.
Tests all static and instance methods for prompt generation.
"""

import unittest
from analyzer.prompts import EnhancedPromptGenerator
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

    def test_build_gemini_multimodal_analysis_prompt_empty_text(self):
        """Test build_gemini_multimodal_analysis_prompt with empty text."""
        result = EnhancedPromptGenerator.build_gemini_multimodal_analysis_prompt("", is_video=False)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        # Should still include the analysis structure
        self.assertIn("OBJETIVO DE INVESTIGACIÓN", result)


class TestEnhancedPromptGeneratorInstance(unittest.TestCase):
    """Test cases for instance methods of EnhancedPromptGenerator."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = EnhancedPromptGenerator()
        self.test_text = "Los musulmanes están invadiendo España"

    def test_initialization(self):
        """Test proper initialization of generator."""
        # Instance is properly created
        self.assertIsNotNone(self.generator)


if __name__ == '__main__':
    unittest.main()
