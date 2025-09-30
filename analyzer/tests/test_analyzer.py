#!/usr/bin/env python3
"""
Comprehensive unit tests for the Analyzer class and related functionality.
Tests all methods, edge cases, and integration scenarios.
"""

import unittest
import tempfile
import os
import sqlite3
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from analyzer.analyzer import Analyzer, ContentAnalysis, migrate_database_schema, save_content_analysis, init_content_analysis_table
from analyzer.categories import Categories


class TestContentAnalysis(unittest.TestCase):
    """Test the ContentAnalysis dataclass."""

    def test_content_analysis_creation(self):
        """Test basic ContentAnalysis creation."""
        analysis = ContentAnalysis(
            tweet_id="test_123",
            tweet_url="https://twitter.com/test/status/test_123",
            username="test_user",
            tweet_content="Test content",
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.GENERAL
        )

        self.assertEqual(analysis.tweet_id, "test_123")
        self.assertEqual(analysis.category, Categories.GENERAL)  # default
        self.assertEqual(analysis.analysis_method, "pattern")  # default
        self.assertEqual(len(analysis.categories_detected), 0)
        self.assertEqual(len(analysis.pattern_matches), 0)

    def test_content_analysis_with_categories(self):
        """Test ContentAnalysis with category data."""
        analysis = ContentAnalysis(
            tweet_id="test_123",
            tweet_url="https://twitter.com/test/status/test_123",
            username="test_user",
            tweet_content="Test content",
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.HATE_SPEECH,
            categories_detected=[Categories.HATE_SPEECH, Categories.FAR_RIGHT_BIAS],
            pattern_matches=[{"matched_text": "test", "category": "hate_speech"}]
        )

        self.assertEqual(analysis.category, Categories.HATE_SPEECH)
        self.assertEqual(len(analysis.categories_detected), 2)
        self.assertTrue(analysis.has_multiple_categories)
        self.assertEqual(analysis.get_secondary_categories(), [Categories.FAR_RIGHT_BIAS])

    def test_content_analysis_single_category(self):
        """Test ContentAnalysis with single category."""
        analysis = ContentAnalysis(
            tweet_id="test_123",
            tweet_url="https://twitter.com/test/status/test_123",
            username="test_user",
            tweet_content="Test content",
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.GENERAL,
            categories_detected=[Categories.GENERAL]
        )

        self.assertFalse(analysis.has_multiple_categories)
        self.assertEqual(analysis.get_secondary_categories(), [])


class TestAnalyzerInitialization(unittest.TestCase):
    """Test Analyzer initialization with different parameters."""

    @patch('analyzer.analyzer.EnhancedLLMPipeline')
    def test_analyzer_init_default(self, mock_llm_pipeline):
        """Test default analyzer initialization."""
        mock_llm_pipeline.return_value = Mock()

        analyzer = Analyzer()

        self.assertTrue(analyzer.use_llm)
        self.assertEqual(analyzer.model_priority, "balanced")
        self.assertFalse(analyzer.verbose)
        mock_llm_pipeline.assert_called_once_with(model_priority="balanced")

    @patch('analyzer.analyzer.EnhancedLLMPipeline')
    def test_analyzer_init_no_llm(self, mock_llm_pipeline):
        """Test analyzer initialization without LLM."""
        mock_llm_pipeline.side_effect = Exception("LLM init failed")

        analyzer = Analyzer(use_llm=False)

        self.assertFalse(analyzer.use_llm)
        self.assertIsNone(analyzer.llm_pipeline)
        # Should be called twice - once for main model, once for fallback
        self.assertEqual(mock_llm_pipeline.call_count, 2)

    @patch('analyzer.analyzer.EnhancedLLMPipeline')
    def test_analyzer_init_fast_priority(self, mock_llm_pipeline):
        """Test analyzer initialization with fast model priority."""
        mock_llm_pipeline.return_value = Mock()

        analyzer = Analyzer(model_priority="fast")

        self.assertEqual(analyzer.model_priority, "fast")
        mock_llm_pipeline.assert_called_once_with(model_priority="fast")

    @patch('analyzer.analyzer.EnhancedLLMPipeline')
    def test_analyzer_init_verbose(self, mock_llm_pipeline):
        """Test analyzer initialization with verbose output."""
        mock_llm_pipeline.return_value = Mock()

        analyzer = Analyzer(verbose=True)

        self.assertTrue(analyzer.verbose)

    @patch('analyzer.analyzer.EnhancedLLMPipeline')
    def test_analyzer_init_llm_failure(self, mock_llm_pipeline):
        """Test analyzer initialization when LLM fails."""
        mock_llm_pipeline.side_effect = Exception("LLM init failed")

        analyzer = Analyzer()

        self.assertIsNone(analyzer.llm_pipeline)


class TestAnalyzerAnalysis(unittest.TestCase):
    """Test the main analysis functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = Analyzer(use_llm=False)  # Disable LLM for faster tests

    def test_analyze_content_empty(self):
        """Test analysis of empty content."""
        result = self.analyzer.analyze_content(
            tweet_id="test_123",
            tweet_url="https://twitter.com/test/status/test_123",
            username="test_user",
            content=""
        )

        self.assertEqual(result.category, Categories.GENERAL)
        self.assertIn("too short", result.llm_explanation.lower())

    def test_analyze_content_short(self):
        """Test analysis of very short content."""
        result = self.analyzer.analyze_content(
            tweet_id="test_123",
            tweet_url="https://twitter.com/test/status/test_123",
            username="test_user",
            content="Hi"
        )

        self.assertEqual(result.category, Categories.GENERAL)
        self.assertIn("too short", result.llm_explanation.lower())

    @patch('analyzer.analyzer.PatternAnalyzer')
    def test_analyze_content_with_patterns(self, mock_pattern_analyzer):
        """Test analysis when patterns are detected."""
        # Mock pattern analyzer to return hate speech detection
        mock_instance = Mock()
        mock_instance.analyze_content.return_value = Mock(
            categories=[Categories.HATE_SPEECH],
            pattern_matches=[Mock(matched_text="test", category="hate_speech", description="test")]
        )
        mock_pattern_analyzer.return_value = mock_instance

        analyzer = Analyzer(use_llm=False)
        analyzer.pattern_analyzer = mock_instance

        result = analyzer.analyze_content(
            tweet_id="test_123",
            tweet_url="https://twitter.com/test/status/test_123",
            username="test_user",
            content="Test content with hate speech"
        )

        self.assertEqual(result.category, Categories.HATE_SPEECH)
        self.assertEqual(result.analysis_method, "pattern")
        self.assertEqual(len(result.categories_detected), 1)

    @patch('analyzer.analyzer.PatternAnalyzer')
    @patch('analyzer.analyzer.EnhancedLLMPipeline')
    def test_analyze_content_no_patterns_llm_fallback(self, mock_llm_pipeline, mock_pattern_analyzer):
        """Test analysis when no patterns detected, using LLM fallback."""
        # Mock pattern analyzer to return no patterns
        mock_pattern_instance = Mock()
        mock_pattern_instance.analyze_content.return_value = Mock(
            categories=[],
            pattern_matches=[]
        )
        mock_pattern_analyzer.return_value = mock_pattern_instance

        # Mock LLM pipeline
        mock_llm_instance = Mock()
        mock_llm_instance.get_category.return_value = Categories.DISINFORMATION
        mock_llm_instance.get_explanation.return_value = "Test explanation"
        mock_llm_pipeline.return_value = mock_llm_instance

        analyzer = Analyzer(use_llm=True)
        analyzer.llm_pipeline = mock_llm_instance

        result = analyzer.analyze_content(
            tweet_id="test_123",
            tweet_url="https://twitter.com/test/status/test_123",
            username="test_user",
            content="Test content with no patterns"
        )

        self.assertEqual(result.category, Categories.DISINFORMATION)
        self.assertEqual(result.analysis_method, "llm")
        mock_llm_instance.get_category.assert_called_once()
        mock_llm_instance.get_explanation.assert_called_once()


class TestAnalyzerInternalMethods(unittest.TestCase):
    """Test internal analyzer methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = Analyzer(use_llm=False)

    @patch('analyzer.analyzer.PatternAnalyzer')
    def test_run_pattern_analysis(self, mock_pattern_analyzer):
        """Test pattern analysis execution."""
        mock_instance = Mock()
        mock_result = Mock()
        mock_instance.analyze_content.return_value = mock_result
        mock_pattern_analyzer.return_value = mock_instance

        analyzer = Analyzer(use_llm=False)
        analyzer.pattern_analyzer = mock_instance

        result = analyzer._run_pattern_analysis("test content")

        self.assertEqual(result['pattern_result'], mock_result)
        mock_instance.analyze_content.assert_called_once_with("test content")

    def test_categorize_content_with_patterns(self):
        """Test categorization when patterns are found."""
        pattern_results = {
            'pattern_result': Mock(categories=[Categories.HATE_SPEECH])
        }

        category, method = self.analyzer._categorize_content("test content", pattern_results)

        self.assertEqual(category, Categories.HATE_SPEECH)
        self.assertEqual(method, "pattern")

    def test_categorize_content_no_patterns(self):
        """Test categorization when no patterns are found."""
        pattern_results = {
            'pattern_result': Mock(categories=[])
        }

        with patch.object(self.analyzer, '_get_llm_category') as mock_get_llm:
            mock_get_llm.return_value = Categories.GENERAL

            category, method = self.analyzer._categorize_content("test content", pattern_results)

            self.assertEqual(category, Categories.GENERAL)
            self.assertEqual(method, "llm")
            mock_get_llm.assert_called_once_with("test content", pattern_results)

    @patch('analyzer.analyzer.EnhancedLLMPipeline')
    def test_get_llm_category_success(self, mock_llm_pipeline):
        """Test successful LLM category retrieval."""
        mock_instance = Mock()
        mock_instance.get_category.return_value = Categories.CONSPIRACY_THEORY
        mock_llm_pipeline.return_value = mock_instance

        analyzer = Analyzer(use_llm=True)
        analyzer.llm_pipeline = mock_instance

        result = analyzer._get_llm_category("test content", {})

        self.assertEqual(result, Categories.CONSPIRACY_THEORY)
        mock_instance.get_category.assert_called_once_with("test content")

    def test_get_llm_category_no_llm(self):
        """Test LLM category retrieval when no LLM available."""
        analyzer = Analyzer(use_llm=False)

        result = analyzer._get_llm_category("test content", {})

        self.assertEqual(result, Categories.GENERAL)

    @patch('analyzer.analyzer.EnhancedLLMPipeline')
    def test_get_llm_category_llm_error(self, mock_llm_pipeline):
        """Test LLM category retrieval when LLM fails."""
        mock_instance = Mock()
        mock_instance.get_category.side_effect = Exception("LLM error")
        mock_llm_pipeline.return_value = mock_instance

        analyzer = Analyzer(use_llm=True)
        analyzer.llm_pipeline = mock_instance

        result = analyzer._get_llm_category("test content", {})

        self.assertEqual(result, Categories.GENERAL)

    @patch('analyzer.analyzer.EnhancedLLMPipeline')
    def test_generate_llm_explanation_success(self, mock_llm_pipeline):
        """Test successful LLM explanation generation."""
        mock_instance = Mock()
        mock_instance.get_explanation.return_value = "Test explanation"
        mock_llm_pipeline.return_value = mock_instance

        analyzer = Analyzer(use_llm=True)
        analyzer.llm_pipeline = mock_instance

        pattern_results = {'pattern_result': Mock(categories=[Categories.HATE_SPEECH])}

        result = analyzer._generate_llm_explanation("test content", Categories.HATE_SPEECH, pattern_results)

        self.assertEqual(result, "Test explanation")
        mock_instance.get_explanation.assert_called_once()

    @patch('analyzer.analyzer.EnhancedLLMPipeline')
    def test_generate_llm_explanation_no_llm(self, mock_llm_pipeline):
        """Test LLM explanation generation when no LLM available."""
        mock_llm_pipeline.side_effect = Exception("LLM init failed")

        analyzer = Analyzer(use_llm=False)
        # Ensure llm_pipeline is None
        self.assertIsNone(analyzer.llm_pipeline)

        result = analyzer._generate_llm_explanation("test content", Categories.HATE_SPEECH, {})

        self.assertIn("LLM pipeline not available", result)

    @patch('analyzer.analyzer.EnhancedLLMPipeline')
    def test_generate_llm_explanation_llm_error(self, mock_llm_pipeline):
        """Test LLM explanation generation when LLM fails."""
        mock_instance = Mock()
        mock_instance.get_explanation.side_effect = Exception("LLM error")
        mock_llm_pipeline.return_value = mock_instance

        analyzer = Analyzer(use_llm=True)
        analyzer.llm_pipeline = mock_instance

        result = analyzer._generate_llm_explanation("test content", Categories.HATE_SPEECH, {})

        self.assertIn("LLM explanation generation exception", result)
        self.assertIn("LLM error", result)

    def test_build_analysis_data(self):
        """Test building analysis data structure."""
        pattern_result = Mock()
        pattern_result.pattern_matches = [
            Mock(matched_text="test1", category="hate_speech", description="desc1"),
            Mock(matched_text="test2", category="disinformation", description="desc2")
        ]
        pattern_result.categories = [Categories.HATE_SPEECH, Categories.DISINFORMATION]
        pattern_result.political_context = ["test_context"]

        pattern_results = {'pattern_result': pattern_result}

        result = self.analyzer._build_analysis_data(pattern_results)

        self.assertEqual(result['category'], None)
        self.assertEqual(len(result['pattern_matches']), 2)
        self.assertEqual(result['unified_categories'], [Categories.HATE_SPEECH, Categories.DISINFORMATION])


class TestAnalyzerUtilityMethods(unittest.TestCase):
    """Test analyzer utility methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = Analyzer(use_llm=False)

    @patch('analyzer.analyzer.EnhancedLLMPipeline')
    def test_cleanup_resources(self, mock_llm_pipeline):
        """Test resource cleanup."""
        mock_instance = Mock()
        mock_instance.cleanup = Mock()
        mock_llm_pipeline.return_value = mock_instance

        analyzer = Analyzer(use_llm=True)
        analyzer.llm_pipeline = mock_instance

        analyzer.cleanup_resources()

        mock_instance.cleanup.assert_called_once()

    def test_cleanup_resources_no_llm(self):
        """Test resource cleanup when no LLM."""
        analyzer = Analyzer(use_llm=False)

        # Should not raise exception
        analyzer.cleanup_resources()

    @patch('builtins.print')
    def test_print_system_status(self, mock_print):
        """Test system status printing."""
        analyzer = Analyzer(use_llm=False)

        analyzer.print_system_status()

        # Check that print was called with the expected header
        mock_print.assert_any_call("🔧 ANALYZER SYSTEM STATUS")


class TestDatabaseFunctions(unittest.TestCase):
    """Test database-related functions."""

    def setUp(self):
        """Set up temporary database for testing."""
        self.db_fd, self.db_path = tempfile.mkstemp()
        # Override the DB_PATH for testing
        import analyzer.analyzer
        analyzer.analyzer.DB_PATH = self.db_path

    def tearDown(self):
        """Clean up temporary database."""
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_init_content_analysis_table(self):
        """Test content analysis table initialization."""
        init_content_analysis_table()

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Check table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='content_analyses'")
        self.assertIsNotNone(c.fetchone())

        # Check columns exist
        c.execute("PRAGMA table_info(content_analyses)")
        columns = [row[1] for row in c.fetchall()]

        expected_columns = [
            'id', 'tweet_id', 'tweet_url', 'username', 'tweet_content',
            'category', 'subcategory', 'llm_explanation', 'calls_to_action',
            'analysis_json', 'analysis_timestamp', 'evidence_sources',
            'verification_status', 'misinformation_risk', 'analysis_method',
            'categories_detected'
        ]

        for col in expected_columns:
            self.assertIn(col, columns)

        conn.close()

    def test_migrate_database_schema_create_table(self):
        """Test database schema migration when table doesn't exist."""
        migrate_database_schema()

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Check table was created
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='content_analyses'")
        self.assertIsNotNone(c.fetchone())

        conn.close()

    def test_save_content_analysis(self):
        """Test saving content analysis to database."""
        # Initialize table first
        init_content_analysis_table()

        analysis = ContentAnalysis(
            tweet_id="test_123",
            tweet_url="https://twitter.com/test/status/test_123",
            username="test_user",
            tweet_content="Test content",
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.HATE_SPEECH,
            llm_explanation="Test explanation",
            analysis_method="pattern",
            categories_detected=[Categories.HATE_SPEECH]
        )

        save_content_analysis(analysis)

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("SELECT * FROM content_analyses WHERE tweet_id = ?", ("test_123",))
        row = c.fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row[1], "test_123")  # tweet_id
        self.assertEqual(row[5], Categories.HATE_SPEECH)  # category
        self.assertEqual(row[7], "Test explanation")  # llm_explanation
        self.assertEqual(row[14], "pattern")  # analysis_method

        conn.close()

    def test_save_content_analysis_duplicate(self):
        """Test saving duplicate content analysis (should replace)."""
        # Initialize table first
        init_content_analysis_table()

        analysis1 = ContentAnalysis(
            tweet_id="test_123",
            tweet_url="https://twitter.com/test/status/test_123",
            username="test_user",
            tweet_content="Test content 1",
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.HATE_SPEECH
        )

        analysis2 = ContentAnalysis(
            tweet_id="test_123",
            tweet_url="https://twitter.com/test/status/test_123",
            username="test_user",
            tweet_content="Test content 2",
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.DISINFORMATION
        )

        save_content_analysis(analysis1)
        save_content_analysis(analysis2)  # Should replace

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("SELECT COUNT(*) FROM content_analyses WHERE tweet_id = ?", ("test_123",))
        count = c.fetchone()[0]
        self.assertEqual(count, 1)

        c.execute("SELECT category FROM content_analyses WHERE tweet_id = ?", ("test_123",))
        category = c.fetchone()[0]
        self.assertEqual(category, Categories.DISINFORMATION)

        conn.close()


class TestAnalyzerIntegration(unittest.TestCase):
    """Integration tests for the analyzer."""

    def test_analyzer_full_workflow(self):
        """Test complete analyzer workflow."""
        analyzer = Analyzer(use_llm=False)

        result = analyzer.analyze_content(
            tweet_id="integration_test_123",
            tweet_url="https://twitter.com/test/status/integration_test_123",
            username="integration_user",
            content="This is a test content for integration testing."
        )

        # Verify result structure
        self.assertIsInstance(result, ContentAnalysis)
        self.assertEqual(result.tweet_id, "integration_test_123")
        self.assertEqual(result.username, "integration_user")
        self.assertIsInstance(result.analysis_timestamp, str)
        self.assertIsInstance(result.categories_detected, list)
        self.assertIsInstance(result.pattern_matches, list)

    def test_analyzer_with_real_pattern_analyzer(self):
        """Test analyzer with real pattern analyzer (no mocking)."""
        analyzer = Analyzer(use_llm=False)

        # Test with content that should trigger patterns
        result = analyzer.analyze_content(
            tweet_id="real_test_123",
            tweet_url="https://twitter.com/test/status/real_test_123",
            username="real_user",
            content="Los inmigrantes nos están invadiendo y robando nuestros trabajos."
        )

        # Should detect some category (may vary based on pattern analyzer)
        self.assertIsNotNone(result.category)
        self.assertIsInstance(result.categories_detected, list)


if __name__ == '__main__':
    # Set up test environment
    os.environ.setdefault('PYTHONPATH', str(project_root))

    # Run tests
    unittest.main(verbosity=2)