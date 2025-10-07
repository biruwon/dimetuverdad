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

from analyzer.analyzer import Analyzer, ContentAnalysis, save_content_analysis
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

    def test_content_analysis_with_metrics(self):
        """Test ContentAnalysis with performance metrics."""
        analysis = ContentAnalysis(
            tweet_id="test_123",
            tweet_url="https://twitter.com/test/status/test_123",
            username="test_user",
            tweet_content="Test content",
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.HATE_SPEECH,
            analysis_time_seconds=2.5,
            model_used="ollama-balanced",
            tokens_used=150
        )

        self.assertEqual(analysis.analysis_time_seconds, 2.5)
        self.assertEqual(analysis.model_used, "ollama-balanced")
        self.assertEqual(analysis.tokens_used, 150)


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
    def test_analyze_content_with_metrics_tracking(self, mock_llm_pipeline, mock_pattern_analyzer):
        """Test that analyze_content properly tracks metrics."""
        # Mock pattern analyzer to return hate speech detection
        mock_instance = Mock()
        mock_instance.analyze_content.return_value = Mock(
            categories=[Categories.HATE_SPEECH],
            pattern_matches=[Mock(matched_text="test", category="hate_speech", description="test")]
        )
        mock_pattern_analyzer.return_value = mock_instance

        # Mock LLM pipeline
        mock_llm_instance = Mock()
        mock_llm_pipeline.return_value = mock_llm_instance

        analyzer = Analyzer(use_llm=True)
        analyzer.llm_pipeline = mock_llm_instance

        # Analyze content
        result = analyzer.analyze_content(
            tweet_id="test_123",
            tweet_url="https://twitter.com/test/status/test_123",
            username="test_user",
            content="Test content with hate speech"
        )

        # Check that metrics were updated
        self.assertEqual(analyzer.metrics['total_analyses'], 1)
        self.assertEqual(analyzer.metrics['method_counts']['pattern'], 1)
        self.assertEqual(analyzer.metrics['category_counts'][Categories.HATE_SPEECH], 1)
        self.assertEqual(analyzer.metrics['model_usage']['pattern-matching'], 1)
        self.assertGreater(analyzer.metrics['total_time'], 0)
        self.assertGreater(result.analysis_time_seconds, 0)
        self.assertEqual(result.model_used, "pattern-matching")


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

    def test_update_metrics(self):
        """Test metrics update functionality."""
        analyzer = Analyzer(use_llm=False)
        
        # Create a mock result
        result = ContentAnalysis(
            tweet_id="test_123",
            tweet_url="https://twitter.com/test/status/test_123",
            username="test_user",
            tweet_content="Test content",
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.HATE_SPEECH,
            analysis_method="pattern",
            multimodal_analysis=False
        )
        
        # Update metrics
        analyzer._update_metrics(result, 1.5)
        
        # Check metrics were updated
        self.assertEqual(analyzer.metrics['total_analyses'], 1)
        self.assertEqual(analyzer.metrics['method_counts']['pattern'], 1)
        self.assertEqual(analyzer.metrics['category_counts'][Categories.HATE_SPEECH], 1)
        self.assertEqual(analyzer.metrics['model_usage']['pattern-matching'], 1)
        self.assertEqual(analyzer.metrics['total_time'], 1.5)
        self.assertEqual(analyzer.metrics['avg_time_per_analysis'], 1.5)

    def test_get_model_name_pattern(self):
        """Test model name generation for pattern analysis."""
        analyzer = Analyzer(use_llm=False)
        result = analyzer._get_model_name("pattern", False)
        self.assertEqual(result, "pattern-matching")

    def test_get_model_name_llm(self):
        """Test model name generation for LLM analysis."""
        analyzer = Analyzer(use_llm=True, model_priority="fast")
        result = analyzer._get_model_name("llm", False)
        self.assertEqual(result, "ollama-fast")

    def test_get_model_name_multimodal(self):
        """Test model name generation for multimodal analysis."""
        analyzer = Analyzer(use_llm=False)
        result = analyzer._get_model_name("llm", True)
        self.assertEqual(result, "gemini-2.5-flash")

    def test_get_metrics_report(self):
        """Test metrics report generation."""
        analyzer = Analyzer(use_llm=False)
        
        # Add some mock data
        analyzer.metrics = {
            'total_analyses': 2,
            'method_counts': {'pattern': 1, 'llm': 1},
            'multimodal_count': 0,
            'category_counts': {Categories.HATE_SPEECH: 1, Categories.DISINFORMATION: 1},
            'total_time': 3.0,
            'avg_time_per_analysis': 1.5,
            'model_usage': {'pattern-matching': 1, 'ollama-balanced': 1},
            'start_time': analyzer.metrics['start_time']  # Keep original start time
        }
        
        report = analyzer.get_metrics_report()
        
        # Check report contains expected content
        self.assertIn("ANALYSIS METRICS REPORT", report)
        self.assertIn("Total analyses: 2", report)
        self.assertIn("Average time per analysis: 1.50s", report)
        self.assertIn("PATTERN: 1 (50.0%)", report)
        self.assertIn("LLM: 1 (50.0%)", report)
        self.assertIn("pattern-matching: 1 (50.0%)", report)
        self.assertIn("ollama-balanced: 1 (50.0%)", report)


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

    def test_save_content_analysis(self):
        """Test saving content analysis to database."""
        # Create table manually for test
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
        CREATE TABLE IF NOT EXISTS content_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tweet_id TEXT UNIQUE,
            tweet_url TEXT,
            username TEXT,
            tweet_content TEXT,
            category TEXT,
            llm_explanation TEXT,
            analysis_json TEXT,
            analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            analysis_method TEXT DEFAULT "pattern",
            categories_detected TEXT,
            media_urls TEXT,
            media_analysis TEXT,
            media_type TEXT,
            multimodal_analysis BOOLEAN DEFAULT FALSE
        )
        ''')
        conn.commit()
        conn.close()

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
        self.assertEqual(row['tweet_id'], "test_123")
        self.assertEqual(row['category'], Categories.HATE_SPEECH)
        self.assertEqual(row['llm_explanation'], "Test explanation")
        self.assertEqual(row['analysis_method'], "pattern")

        conn.close()

    def test_save_content_analysis_duplicate(self):
        """Test saving duplicate content analysis (should replace)."""
        # Create table manually for test
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
        CREATE TABLE IF NOT EXISTS content_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tweet_id TEXT UNIQUE,
            tweet_url TEXT,
            username TEXT,
            tweet_content TEXT,
            category TEXT,
            llm_explanation TEXT,
            analysis_json TEXT,
            analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            analysis_method TEXT DEFAULT "pattern",
            categories_detected TEXT,
            media_urls TEXT,
            media_analysis TEXT,
            media_type TEXT,
            multimodal_analysis BOOLEAN DEFAULT FALSE
        )
        ''')
        conn.commit()
        conn.close()

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


class TestAnalyzerMultimodal(unittest.TestCase):
    """Test multimodal analysis functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = Analyzer(use_llm=False)

    @patch('analyzer.gemini_multimodal.analyze_multimodal_content')
    @patch('analyzer.gemini_multimodal.extract_media_type')
    def test_analyze_multi_modal_success(self, mock_extract_type, mock_analyze):
        """Test successful multimodal analysis."""
        # Mock the multimodal analysis
        mock_analyze.return_value = ("Test Gemini analysis", 2.5)
        mock_extract_type.return_value = "image"

        result = self.analyzer.analyze_multi_modal(
            tweet_id="test_123",
            tweet_url="https://twitter.com/test/status/test_123",
            username="test_user",
            content="Test content",
            media_urls=["https://example.com/image.jpg"]
        )

        self.assertEqual(result.tweet_id, "test_123")
        self.assertEqual(result.category, Categories.GENERAL)  # Default for now
        self.assertEqual(result.analysis_method, "gemini")
        self.assertEqual(result.media_analysis, "Test Gemini analysis")
        self.assertEqual(result.media_type, "image")
        self.assertTrue(result.multimodal_analysis)
        self.assertEqual(result.media_urls, ["https://example.com/image.jpg"])

    @patch('analyzer.gemini_multimodal.analyze_multimodal_content')
    def test_analyze_multi_modal_failure_fallback(self, mock_analyze):
        """Test multimodal analysis failure with fallback to text-only."""
        # Mock failure
        mock_analyze.return_value = (None, 1.0)

        with patch.object(self.analyzer, '_analyze_text_only') as mock_text_only:
            mock_text_result = ContentAnalysis(
                tweet_id="test_123",
                tweet_url="https://twitter.com/test/status/test_123",
                username="test_user",
                tweet_content="Test content",
                analysis_timestamp="2024-01-01T12:00:00",
                category=Categories.GENERAL
            )
            mock_text_only.return_value = mock_text_result

            result = self.analyzer.analyze_multi_modal(
                tweet_id="test_123",
                tweet_url="https://twitter.com/test/status/test_123",
                username="test_user",
                content="Test content",
                media_urls=["https://example.com/image.jpg"]
            )

            # Should fallback to text-only result
            self.assertEqual(result, mock_text_result)
            mock_text_only.assert_called_once()

    def test_analyze_content_routing_text_only(self):
        """Test that analyze_content routes to text-only for no media."""
        with patch.object(self.analyzer, '_analyze_text_only') as mock_text_only:
            mock_result = ContentAnalysis(
                tweet_id="test_123",
                tweet_url="https://twitter.com/test/status/test_123",
                username="test_user",
                tweet_content="Test content",
                analysis_timestamp="2024-01-01T12:00:00",
                category=Categories.GENERAL
            )
            mock_text_only.return_value = mock_result

            result = self.analyzer.analyze_content(
                tweet_id="test_123",
                tweet_url="https://twitter.com/test/status/test_123",
                username="test_user",
                content="Test content",
                media_urls=[]
            )

            self.assertEqual(result, mock_result)
            mock_text_only.assert_called_once()

    def test_analyze_content_routing_multimodal(self):
        """Test that analyze_content routes to multimodal for media."""
        with patch.object(self.analyzer, 'analyze_multi_modal') as mock_multimodal:
            mock_result = ContentAnalysis(
                tweet_id="test_123",
                tweet_url="https://twitter.com/test/status/test_123",
                username="test_user",
                tweet_content="Test content",
                analysis_timestamp="2024-01-01T12:00:00",
                category=Categories.GENERAL,
                multimodal_analysis=True
            )
            mock_multimodal.return_value = mock_result

            result = self.analyzer.analyze_content(
                tweet_id="test_123",
                tweet_url="https://twitter.com/test/status/test_123",
                username="test_user",
                content="Test content",
                media_urls=["https://example.com/image.jpg"]
            )

            self.assertEqual(result, mock_result)
            mock_multimodal.assert_called_once()

    def test_extract_category_from_gemini_hate_speech(self):
        """Test category extraction from Gemini analysis - hate speech."""
        analysis = "Este contenido contiene discurso de odio y hate_speech contra inmigrantes."
        result = self.analyzer._extract_category_from_gemini(analysis)
        self.assertEqual(result, Categories.HATE_SPEECH)

    def test_extract_category_from_gemini_disinformation(self):
        """Test category extraction from Gemini analysis - disinformation."""
        analysis = "El post contiene desinformación y fake news sobre política."
        result = self.analyzer._extract_category_from_gemini(analysis)
        self.assertEqual(result, Categories.DISINFORMATION)

    def test_extract_category_from_gemini_conspiracy(self):
        """Test category extraction from Gemini analysis - conspiracy."""
        analysis = "Esto es una teoría conspirativa sobre el gobierno."
        result = self.analyzer._extract_category_from_gemini(analysis)
        self.assertEqual(result, Categories.CONSPIRACY_THEORY)

    def test_extract_category_from_gemini_far_right(self):
        """Test category extraction from Gemini analysis - far right."""
        analysis = "Contenido de extrema derecha y far_right_bias político."
        result = self.analyzer._extract_category_from_gemini(analysis)
        self.assertEqual(result, Categories.FAR_RIGHT_BIAS)

    def test_extract_category_from_gemini_call_to_action(self):
        """Test category extraction from Gemini analysis - call to action."""
        analysis = "El post incluye llamados a la acción política."
        result = self.analyzer._extract_category_from_gemini(analysis)
        self.assertEqual(result, Categories.CALL_TO_ACTION)

    def test_extract_category_from_gemini_general(self):
        """Test category extraction from Gemini analysis - general (fallback)."""
        analysis = "Este es un post normal sobre política general."
        result = self.analyzer._extract_category_from_gemini(analysis)
        self.assertEqual(result, Categories.GENERAL)


if __name__ == '__main__':
    # Set up test environment
    os.environ.setdefault('PYTHONPATH', str(project_root))

    # Run tests
    unittest.main(verbosity=2)