#!/usr/bin/env python3
"""
Tests for Phase 3: Analyzer Integration
Comprehensive test coverage for evidence retrieval integration in the main Analyzer class.
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from dataclasses import dataclass
from typing import Dict, Optional
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from analyzer.analyze_twitter import Analyzer
from analyzer.config import AnalyzerConfig
from analyzer.models import ContentAnalysis
from analyzer.categories import Categories


class TestAnalyzerIntegration(unittest.TestCase):
    """Test Phase 3: Analyzer Integration functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = AnalyzerConfig(use_llm=False)  # Disable LLM for faster tests
        self.analyzer = Analyzer(config=self.config)

    def test_analyzer_initialization_with_retrieval_hooks(self):
        """Test that analyzer initializes with retrieval hooks."""
        analyzer = Analyzer(config=self.config)
        # Check that retrieval hooks are initialized
        self.assertTrue(hasattr(analyzer, 'retrieval_hooks'))
        self.assertIsNotNone(analyzer.retrieval_hooks)

    @patch('analyzer.analyze_twitter.create_analyzer_hooks')
    def test_retrieval_hooks_initialization(self, mock_create_hooks):
        """Test that AnalyzerHooks is properly initialized."""
        mock_hooks_instance = Mock()
        mock_create_hooks.return_value = mock_hooks_instance

        analyzer = Analyzer(config=self.config)

        mock_create_hooks.assert_called_once()
        self.assertEqual(analyzer.retrieval_hooks, mock_hooks_instance)


class TestTriggerConditions(unittest.TestCase):
    """Test evidence retrieval trigger conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = AnalyzerConfig(use_llm=False)
        self.analyzer = Analyzer(config=self.config)

    def test_should_trigger_evidence_retrieval_hate_speech(self):
        """Test that hate speech alone does not trigger evidence retrieval."""
        analysis = ContentAnalysis(
            post_id="test_123",
            post_url="https://twitter.com/test/status/test_123",
            author_username="test_user",
            post_content="Test content",
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.HATE_SPEECH
        )

        result = self.analyzer._should_trigger_evidence_retrieval(analysis, "Test content")
        self.assertFalse(result)  # Hate speech alone doesn't trigger

    def test_should_trigger_evidence_retrieval_disinformation(self):
        """Test trigger conditions for disinformation category."""
        analysis = ContentAnalysis(
            post_id="test_123",
            post_url="https://twitter.com/test/status/test_123",
            author_username="test_user",
            post_content="Test content",
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.DISINFORMATION
        )

        result = self.analyzer._should_trigger_evidence_retrieval(analysis, "Test content")
        self.assertTrue(result)

    def test_should_trigger_evidence_retrieval_conspiracy_theory(self):
        """Test trigger conditions for conspiracy theory category."""
        analysis = ContentAnalysis(
            post_id="test_123",
            post_url="https://twitter.com/test/status/test_123",
            author_username="test_user",
            post_content="Test content",
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.CONSPIRACY_THEORY
        )

        result = self.analyzer._should_trigger_evidence_retrieval(analysis, "Test content")
        self.assertTrue(result)

    def test_should_trigger_evidence_retrieval_far_right_bias(self):
        """Test trigger conditions for far right bias category."""
        analysis = ContentAnalysis(
            post_id="test_123",
            post_url="https://twitter.com/test/status/test_123",
            author_username="test_user",
            post_content="Test content",
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.FAR_RIGHT_BIAS
        )

        result = self.analyzer._should_trigger_evidence_retrieval(analysis, "Test content")
        self.assertTrue(result)

    def test_should_trigger_evidence_retrieval_general_no_trigger(self):
        """Test that general category does not trigger evidence retrieval."""
        analysis = ContentAnalysis(
            post_id="test_123",
            post_url="https://twitter.com/test/status/test_123",
            author_username="test_user",
            post_content="Test content",
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.GENERAL
        )

        result = self.analyzer._should_trigger_evidence_retrieval(analysis, "Test content")
        self.assertFalse(result)

    def test_should_trigger_evidence_retrieval_call_to_action_no_trigger(self):
        """Test that call to action category does not trigger evidence retrieval."""
        analysis = ContentAnalysis(
            post_id="test_123",
            post_url="https://twitter.com/test/status/test_123",
            author_username="test_user",
            post_content="Test content",
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.CALL_TO_ACTION
        )

        result = self.analyzer._should_trigger_evidence_retrieval(analysis, "Test content")
        self.assertFalse(result)

    def test_should_trigger_evidence_retrieval_low_confidence_no_trigger(self):
        """Test that low confidence analysis does not trigger evidence retrieval."""
        analysis = ContentAnalysis(
            post_id="test_123",
            post_url="https://twitter.com/test/status/test_123",
            author_username="test_user",
            post_content="Test content",
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.HATE_SPEECH
        )

        result = self.analyzer._should_trigger_evidence_retrieval(analysis, "Simple content without claims")
        self.assertFalse(result)

    def test_should_trigger_evidence_retrieval_high_confidence_trigger(self):
        """Test that high confidence analysis triggers evidence retrieval."""
        analysis = ContentAnalysis(
            post_id="test_123",
            post_url="https://twitter.com/test/status/test_123",
            author_username="test_user",
            post_content="Test content",
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.DISINFORMATION
        )

        result = self.analyzer._should_trigger_evidence_retrieval(analysis, "Test content with numbers and claims")
        self.assertTrue(result)


class TestEvidenceEnhancement(unittest.TestCase):
    """Test LLM explanation enhancement with evidence."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = AnalyzerConfig(use_llm=False)
        self.analyzer = Analyzer(config=self.config)

    @patch('analyzer.analyze_twitter.create_analyzer_hooks')
    def test_enhance_with_evidence_retrieval_success(self, mock_create_hooks):
        """Test successful evidence retrieval and enhancement."""
        mock_hooks_instance = AsyncMock()
        mock_create_hooks.return_value = mock_hooks_instance
        
        # Mock the analyze_with_verification method
        from retrieval.integration.analyzer_hooks import AnalysisResult
        mock_result = AnalysisResult(
            original_result={'category': 'disinformation'},
            verification_data={
                'verification_report': {
                    'overall_verdict': 'debunked',
                    'confidence_score': 0.8,
                    'claims_verified': [],
                    'evidence_sources': [],
                    'temporal_consistency': True,
                    'contradictions_found': [],
                    'processing_time': 1.0,
                    'verification_method': 'fact_check'
                },
                'sources_cited': ['fact-check.org'],
                'contradictions_detected': [],
                'verification_confidence': 0.8
            },
            explanation_with_verification="This appears to be disinformation. Verification confirms this is false."
        )
        mock_hooks_instance.analyze_with_verification.return_value = mock_result

        analyzer = Analyzer(config=self.config)

        analysis = ContentAnalysis(
            post_id="test_123",
            post_url="https://twitter.com/test/status/test_123",
            author_username="test_user",
            post_content="Test content with disinformation",
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.DISINFORMATION,
            llm_explanation="This appears to be disinformation."
        )

        # Run the async method
        async def test_async():
            result = await analyzer._enhance_with_evidence_retrieval(analysis, "Test content with disinformation")
            return result

        result = asyncio.run(test_async())

        # Check that verification was performed
        mock_hooks_instance.analyze_with_verification.assert_called_once()

        # Check that analysis was enhanced
        self.assertIsNotNone(result.verification_data)
        self.assertEqual(result.verification_confidence, 0.8)
        self.assertIn("Verification confirms this is false", result.llm_explanation)

    @patch('analyzer.analyze_twitter.create_analyzer_hooks')
    def test_enhance_with_evidence_retrieval_no_evidence(self, mock_create_hooks):
        """Test enhancement when no evidence is found."""
        mock_hooks_instance = AsyncMock()
        mock_create_hooks.return_value = mock_hooks_instance
        
        # Mock the analyze_with_verification method to return no verification data
        from retrieval.integration.analyzer_hooks import AnalysisResult
        mock_result = AnalysisResult(
            original_result={'category': 'hate_speech'},
            verification_data=None,
            explanation_with_verification="This contains hate speech."
        )
        mock_hooks_instance.analyze_with_verification.return_value = mock_result

        analyzer = Analyzer(config=self.config)

        analysis = ContentAnalysis(
            post_id="test_123",
            post_url="https://twitter.com/test/status/test_123",
            author_username="test_user",
            post_content="Test content",
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.HATE_SPEECH,
            llm_explanation="This contains hate speech."
        )

        async def test_async():
            result = await analyzer._enhance_with_evidence_retrieval(analysis, "Test content")
            return result

        result = asyncio.run(test_async())

        # Check that verification was attempted
        mock_hooks_instance.analyze_with_verification.assert_called_once()

        # Check that analysis indicates no verification data
        self.assertIsNone(result.verification_data)
        self.assertEqual(result.verification_confidence, 0.0)

    @patch('analyzer.analyze_twitter.create_analyzer_hooks')
    def test_enhance_with_evidence_retrieval_error_handling(self, mock_create_hooks):
        """Test error handling during evidence retrieval."""
        mock_hooks_instance = AsyncMock()
        mock_hooks_instance.analyze_with_verification.side_effect = Exception("Retrieval failed")
        mock_create_hooks.return_value = mock_hooks_instance

        analyzer = Analyzer(config=self.config)

        analysis = ContentAnalysis(
            post_id="test_123",
            post_url="https://twitter.com/test/status/test_123",
            author_username="test_user",
            post_content="Test content",
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.CONSPIRACY_THEORY,
            llm_explanation="This appears to be a conspiracy theory."
        )

        async def test_async():
            result = await analyzer._enhance_with_evidence_retrieval(analysis, "Test content")
            return result

        result = asyncio.run(test_async())

        # Check that error was handled gracefully - should return original analysis
        self.assertIsNone(result.verification_data)
        self.assertEqual(result.verification_confidence, 0.0)
        self.assertEqual(result.llm_explanation, "This appears to be a conspiracy theory.")


class TestAsyncAnalyzeContent(unittest.TestCase):
    """Test the async analyze_content method with evidence retrieval integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = AnalyzerConfig(use_llm=False)
        self.analyzer = Analyzer(config=self.config)

    @patch.object(Analyzer, '_should_trigger_evidence_retrieval', return_value=True)
    @patch.object(Analyzer, '_enhance_with_evidence_retrieval')
    def test_analyze_content_with_evidence_retrieval(self, mock_enhance, mock_should_trigger):
        """Test analyze_content with evidence retrieval triggered."""
        # Mock the text analyzer to return a basic analysis
        with patch.object(self.analyzer.text_analyzer, 'analyze') as mock_text_analyze:
            mock_text_analyze.return_value = ContentAnalysis(
                post_id="test_123",
                post_url="https://twitter.com/test/status/test_123",
                author_username="test_user",
                post_content="Test content with disinformation",
                analysis_timestamp="2024-01-01T12:00:00",
                category=Categories.DISINFORMATION,
                llm_explanation="This contains disinformation."
            )

            # Mock the enhancement to return enhanced analysis
            enhanced_analysis = ContentAnalysis(
                post_id="test_123",
                post_url="https://twitter.com/test/status/test_123",
                author_username="test_user",
                post_content="Test content with disinformation",
                analysis_timestamp="2024-01-01T12:00:00",
                category=Categories.DISINFORMATION,
                llm_explanation="This contains disinformation. Evidence confirms this is false.",
                verification_data={"evidence_found": True},
                verification_confidence=0.8
            )
            mock_enhance.return_value = enhanced_analysis

            async def test_async():
                result = await self.analyzer.analyze_content(
                    tweet_id="test_123",
                    tweet_url="https://twitter.com/test/status/test_123",
                    username="test_user",
                    content="Test content with disinformation"
                )
                return result

            result = asyncio.run(test_async())

            # Verify the flow
            mock_text_analyze.assert_called_once()
            mock_should_trigger.assert_called_once()
            mock_enhance.assert_called_once()

            # Verify the result is enhanced
            self.assertEqual(result.verification_confidence, 0.8)
            self.assertEqual(result.verification_data, {"evidence_found": True})

    @patch.object(Analyzer, '_should_trigger_evidence_retrieval', return_value=False)
    def test_analyze_content_without_evidence_retrieval(self, mock_should_trigger):
        """Test analyze_content when evidence retrieval is not triggered."""
        # Mock the text analyzer to return a basic analysis
        with patch.object(self.analyzer.text_analyzer, 'analyze') as mock_text_analyze:
            analysis_result = ContentAnalysis(
                post_id="test_123",
                post_url="https://twitter.com/test/status/test_123",
                author_username="test_user",
                post_content="Test content",
                analysis_timestamp="2024-01-01T12:00:00",
                category=Categories.GENERAL,
                llm_explanation="This is general content."
            )
            mock_text_analyze.return_value = analysis_result

            async def test_async():
                result = await self.analyzer.analyze_content(
                    tweet_id="test_123",
                    tweet_url="https://twitter.com/test/status/test_123",
                    username="test_user",
                    content="Test content"
                )
                return result

            result = asyncio.run(test_async())

            # Verify the flow
            mock_text_analyze.assert_called_once()
            mock_should_trigger.assert_called_once()
            # _enhance_with_evidence_retrieval should not be called

            # Verify the result is unchanged
            self.assertEqual(result.category, Categories.GENERAL)
            self.assertIsNone(result.verification_data)
            self.assertEqual(result.verification_confidence, 0.0)


class TestContentAnalysisVerificationFields(unittest.TestCase):
    """Test the new verification fields in ContentAnalysis."""

    def test_content_analysis_with_verification_fields(self):
        """Test ContentAnalysis creation with verification fields."""
        analysis = ContentAnalysis(
            post_id="test_123",
            post_url="https://twitter.com/test/status/test_123",
            author_username="test_user",
            post_content="Test content",
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.DISINFORMATION,
            verification_data={"evidence_found": True, "sources": ["maldita.es"]},
            verification_confidence=0.85
        )

        self.assertEqual(analysis.verification_data, {"evidence_found": True, "sources": ["maldita.es"]})
        self.assertEqual(analysis.verification_confidence, 0.85)

    def test_content_analysis_default_verification_fields(self):
        """Test ContentAnalysis default verification fields."""
        analysis = ContentAnalysis(
            post_id="test_123",
            post_url="https://twitter.com/test/status/test_123",
            author_username="test_user",
            post_content="Test content",
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.GENERAL
        )

        self.assertIsNone(analysis.verification_data)
        self.assertEqual(analysis.verification_confidence, 0.0)

    def test_content_analysis_verification_data_types(self):
        """Test ContentAnalysis verification field types."""
        # Test with None
        analysis = ContentAnalysis(
            post_id="test_123",
            post_url="https://twitter.com/test/status/test_123",
            author_username="test_user",
            post_content="Test content",
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.GENERAL,
            verification_data=None,
            verification_confidence=0.0
        )
        self.assertIsNone(analysis.verification_data)
        self.assertEqual(analysis.verification_confidence, 0.0)

        # Test with dict
        analysis = ContentAnalysis(
            post_id="test_123",
            post_url="https://twitter.com/test/status/test_123",
            author_username="test_user",
            post_content="Test content",
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.GENERAL,
            verification_data={"key": "value"},
            verification_confidence=0.75
        )
        self.assertEqual(analysis.verification_data, {"key": "value"})
        self.assertEqual(analysis.verification_confidence, 0.75)


class TestIntegrationEndToEnd(unittest.TestCase):
    """Test end-to-end integration scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = AnalyzerConfig(use_llm=False)
        self.analyzer = Analyzer(config=self.config)

    @patch.object(Analyzer, '_should_trigger_evidence_retrieval', return_value=True)
    @patch.object(Analyzer, '_enhance_with_evidence_retrieval')
    def test_end_to_end_disinformation_with_evidence(self, mock_enhance, mock_should_trigger):
        """Test complete flow for disinformation content with evidence retrieval."""
        # Mock initial analysis
        with patch.object(self.analyzer.text_analyzer, 'analyze') as mock_text_analyze:
            initial_analysis = ContentAnalysis(
                post_id="disinfo_123",
                post_url="https://twitter.com/test/status/disinfo_123",
                author_username="bad_actor",
                post_content="COVID vaccines contain microchips from Bill Gates",
                analysis_timestamp="2024-01-01T12:00:00",
                category=Categories.DISINFORMATION,
                llm_explanation="This appears to be a conspiracy theory about vaccines."
            )
            mock_text_analyze.return_value = initial_analysis

            # Mock evidence enhancement
            enhanced_analysis = ContentAnalysis(
                post_id="disinfo_123",
                post_url="https://twitter.com/test/status/disinfo_123",
                author_username="bad_actor",
                post_content="COVID vaccines contain microchips from Bill Gates",
                analysis_timestamp="2024-01-01T12:00:00",
                category=Categories.DISINFORMATION,
                llm_explanation="This appears to be a conspiracy theory about vaccines. Verification from fact-checkers confirms this claim is false and has been repeatedly debunked.",
                verification_data={
                    "evidence_found": True,
                    "sources": ["maldita.es", "newtral.es"],
                    "verdict": "debunked"
                },
                verification_confidence=0.95
            )
            mock_enhance.return_value = enhanced_analysis

            async def test_async():
                result = await self.analyzer.analyze_content(
                    tweet_id="disinfo_123",
                    tweet_url="https://twitter.com/test/status/disinfo_123",
                    username="bad_actor",
                    content="COVID vaccines contain microchips from Bill Gates"
                )
                return result

            result = asyncio.run(test_async())

            # Verify complete flow
            self.assertEqual(result.category, Categories.DISINFORMATION)
            self.assertEqual(result.verification_confidence, 0.95)
            self.assertIn("Verification from fact-checkers", result.llm_explanation)
            self.assertIn("debunked", result.verification_data["verdict"])

    @patch.object(Analyzer, '_should_trigger_evidence_retrieval', return_value=False)
    def test_end_to_end_general_content_no_evidence(self, mock_should_trigger):
        """Test complete flow for general content without evidence retrieval."""
        with patch.object(self.analyzer.text_analyzer, 'analyze') as mock_text_analyze:
            # Mock analysis
            analysis = ContentAnalysis(
                post_id="general_123",
                post_url="https://twitter.com/test/status/general_123",
                author_username="normal_user",
                post_content="Having a great day at the park!",
                analysis_timestamp="2024-01-01T12:00:00",
                category=Categories.GENERAL,
                llm_explanation="This is a positive, general statement."
            )
            mock_text_analyze.return_value = analysis

            async def test_async():
                result = await self.analyzer.analyze_content(
                    tweet_id="general_123",
                    tweet_url="https://twitter.com/test/status/general_123",
                    username="normal_user",
                    content="Having a great day at the park!"
                )
                return result

            result = asyncio.run(test_async())

            # Verify no evidence retrieval occurred
            self.assertEqual(result.category, Categories.GENERAL)
            self.assertIsNone(result.verification_data)
            self.assertEqual(result.verification_confidence, 0.0)
            self.assertNotIn("verification", result.llm_explanation.lower())


if __name__ == '__main__':
    unittest.main()
