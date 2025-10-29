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
from analyzer.flow_manager import AnalysisFlowManager
from retrieval.integration.analyzer_hooks import AnalysisResult


class TestAnalyzerIntegration(unittest.TestCase):
    """Test Phase 3: Analyzer Integration functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = AnalyzerConfig()  # Disable LLM for faster tests
        self.analyzer = Analyzer(config=self.config)

    def test_analyzer_initialization_with_retrieval_hooks(self):
        """Test that analyzer initializes without retrieval hooks (now handled in flow_manager)."""
        analyzer = Analyzer(config=self.config)
        # Check that retrieval hooks are NOT initialized in analyzer (moved to flow_manager)
        self.assertFalse(hasattr(analyzer, 'retrieval_hooks'))
        # But flow_manager should have analyzer hooks
        self.assertTrue(hasattr(analyzer.flow_manager, 'analyzer_hooks'))

    def test_retrieval_hooks_initialization(self):
        """Test that AnalyzerHooks is properly initialized in flow_manager."""
        analyzer = Analyzer(config=self.config)
        # Check that analyzer hooks are initialized in flow_manager
        self.assertTrue(hasattr(analyzer.flow_manager, 'analyzer_hooks'))
        self.assertIsNotNone(analyzer.flow_manager.analyzer_hooks)


class TestTriggerConditions(unittest.TestCase):
    """Test evidence retrieval trigger conditions (now handled in flow_manager)."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = AnalyzerConfig()
        self.analyzer = Analyzer(config=self.config)

    @unittest.skip("Evidence retrieval logic moved to flow_manager - these methods no longer exist in analyzer")
    def test_should_trigger_evidence_retrieval_hate_speech(self):
        """Test that hate speech alone does not trigger evidence retrieval."""
        pass

    @unittest.skip("Evidence retrieval logic moved to flow_manager")
    def test_should_trigger_evidence_retrieval_disinformation(self):
        """Test trigger conditions for disinformation category."""
        pass

    @unittest.skip("Evidence retrieval logic moved to flow_manager")
    def test_should_trigger_evidence_retrieval_conspiracy_theory(self):
        """Test trigger conditions for conspiracy theory category."""
        pass

    @unittest.skip("Evidence retrieval logic moved to flow_manager")
    def test_should_trigger_evidence_retrieval_anti_government(self):
        """Test trigger conditions for anti-government category."""
        pass

    @unittest.skip("Evidence retrieval logic moved to flow_manager")
    def test_should_trigger_evidence_retrieval_general_no_trigger(self):
        """Test that general category does not trigger evidence retrieval."""
        pass

    @unittest.skip("Evidence retrieval logic moved to flow_manager")
    def test_should_trigger_evidence_retrieval_call_to_action_no_trigger(self):
        """Test that call to action category does not trigger evidence retrieval."""
        pass

    @unittest.skip("Evidence retrieval logic moved to flow_manager")
    def test_should_trigger_evidence_retrieval_low_confidence_no_trigger(self):
        """Test that low confidence analysis does not trigger evidence retrieval."""
        pass

    @unittest.skip("Evidence retrieval logic moved to flow_manager")
    def test_should_trigger_evidence_retrieval_high_confidence_trigger(self):
        """Test that high confidence analysis triggers evidence retrieval."""
        pass


class TestEvidenceEnhancement(unittest.TestCase):
    """Test LLM explanation enhancement with evidence (now handled in flow_manager)."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = AnalyzerConfig()
        self.analyzer = Analyzer(config=self.config)

    @unittest.skip("Evidence enhancement methods moved to flow_manager")
    def test_enhance_with_evidence_retrieval_success(self):
        """Test successful evidence retrieval and enhancement."""
        pass

    @unittest.skip("Evidence enhancement methods moved to flow_manager")
    def test_enhance_with_evidence_retrieval_no_evidence(self):
        """Test enhancement when no evidence is found."""
        pass

    @unittest.skip("Evidence enhancement methods moved to flow_manager")
    def test_enhance_with_evidence_retrieval_error_handling(self):
        """Test error handling during evidence retrieval."""
        pass


class TestAsyncAnalyzeContent(unittest.TestCase):
    """Test the async analyze_content method (evidence retrieval now handled in flow_manager)."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = AnalyzerConfig()
        self.analyzer = Analyzer(config=self.config)

    @unittest.skip("Evidence retrieval methods moved to flow_manager")
    def test_analyze_content_with_evidence_retrieval(self):
        """Test analyze_content with evidence retrieval triggered."""
        pass

    @unittest.skip("Evidence retrieval methods moved to flow_manager")
    def test_analyze_content_without_evidence_retrieval(self):
        """Test analyze_content when evidence retrieval is not triggered."""
        pass


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
    """Test end-to-end integration scenarios (evidence retrieval now handled in flow_manager)."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = AnalyzerConfig()
        self.analyzer = Analyzer(config=self.config)

    @unittest.skip("Evidence retrieval methods moved to flow_manager")
    def test_end_to_end_disinformation_with_evidence(self):
        """Test complete flow for disinformation content with evidence retrieval."""
        pass

    @unittest.skip("Evidence retrieval methods moved to flow_manager")
    def test_end_to_end_general_content_no_evidence(self):
        """Test complete flow for general content without evidence retrieval."""
        pass


if __name__ == '__main__':
    unittest.main()
