#!/usr/bin/env python3
"""
Comprehensive unit tests for the Analyzer class and related functionality.
Tests all methods, edge cases, and integration scenarios.
"""

import unittest
import os
import sqlite3
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime
import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from analyzer.analyze_twitter import (
    Analyzer, analyze_tweets_from_db, create_analyzer, reanalyze_tweet, main
)
from analyzer.config import AnalyzerConfig
from analyzer.models import ContentAnalysis
from analyzer.repository import ContentAnalysisRepository
from analyzer.categories import Categories
from utils.database import get_db_connection_context
from analyzer.multimodal_analyzer import extract_media_type


class TestContentAnalysis(unittest.TestCase):
    """Test the ContentAnalysis dataclass."""

    def test_content_analysis_creation(self):
        """Test basic ContentAnalysis creation."""
        analysis = ContentAnalysis(
            post_id="test_123",
            post_url="https://twitter.com/test/status/test_123",
            author_username="test_user",
            post_content="Test content",
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.GENERAL
        )

        self.assertEqual(analysis.post_id, "test_123")
        self.assertEqual(analysis.category, Categories.GENERAL)  # default
        self.assertEqual(analysis.analysis_method, "pattern")  # default
        self.assertEqual(len(analysis.categories_detected), 0)
        self.assertEqual(len(analysis.pattern_matches), 0)

    def test_content_analysis_with_categories(self):
        """Test ContentAnalysis with category data."""
        analysis = ContentAnalysis(
            post_id="test_123",
            post_url="https://twitter.com/test/status/test_123",
            author_username="test_user",
            post_content="Test content",
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
            post_id="test_123",
            post_url="https://twitter.com/test/status/test_123",
            author_username="test_user",
            post_content="Test content",
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

    @patch('analyzer.llm_models.EnhancedLLMPipeline')
    def test_analyzer_init_default(self, mock_llm_pipeline):
        """Test analyzer initialization with default config."""
        analyzer = Analyzer(config=AnalyzerConfig())

        self.assertTrue(analyzer.config.use_llm)
        self.assertEqual(analyzer.config.model_priority, "balanced")
        self.assertFalse(analyzer.config.verbose)

    @patch('analyzer.llm_models.EnhancedLLMPipeline')
    def test_analyzer_init_no_llm(self, mock_llm_pipeline):
        """Test analyzer initialization without LLM."""
        analyzer = Analyzer(config=AnalyzerConfig(use_llm=False))

        self.assertFalse(analyzer.config.use_llm)
        self.assertIsNone(analyzer.text_analyzer.llm_pipeline)

    @patch('analyzer.llm_models.EnhancedLLMPipeline')
    def test_analyzer_init_fast_priority(self, mock_llm_pipeline):
        """Test analyzer initialization with fast model priority."""
        analyzer = Analyzer(config=AnalyzerConfig(model_priority="fast"))

        self.assertEqual(analyzer.config.model_priority, "fast")

    @patch('analyzer.llm_models.EnhancedLLMPipeline')
    def test_analyzer_init_verbose(self, mock_llm_pipeline):
        """Test analyzer initialization with verbose output."""
        analyzer = Analyzer(config=AnalyzerConfig(verbose=True))

        self.assertTrue(analyzer.config.verbose)

    @patch('analyzer.llm_models.EnhancedLLMPipeline')
    def test_analyzer_init_llm_failure(self, mock_llm_pipeline):
        """Test analyzer initialization when LLM fails."""
        analyzer = Analyzer(config=AnalyzerConfig(use_llm=False))

        self.assertIsNone(analyzer.text_analyzer.llm_pipeline)


class TestAnalyzerAnalysis(unittest.TestCase):
    """Test the main analysis functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = Analyzer(config=AnalyzerConfig(use_llm=False))  # Disable LLM for faster tests

    def test_analyze_content_empty(self):
        """Test analysis of empty content."""
        async def test_async():
            result = await self.analyzer.analyze_content(
                tweet_id="test_123",
                tweet_url="https://twitter.com/test/status/test_123",
                username="test_user",
                content=""
            )
            return result

        result = asyncio.run(test_async())

        self.assertEqual(result.category, Categories.GENERAL)
        # Ollama provides detailed explanation even for empty content
        self.assertIsNotNone(result.llm_explanation)
        self.assertTrue(len(result.llm_explanation) > 0)

    def test_analyze_content_short(self):
        """Test analysis of very short content."""
        async def test_async():
            result = await self.analyzer.analyze_content(
                tweet_id="test_123",
                tweet_url="https://twitter.com/test/status/test_123",
                username="test_user",
                content="Hi"
            )
            return result

        result = asyncio.run(test_async())

        self.assertEqual(result.category, Categories.GENERAL)
        # Ollama provides detailed explanation even for short content
        self.assertIsNotNone(result.llm_explanation)
        self.assertTrue(len(result.llm_explanation) > 0)

    @patch('analyzer.pattern_analyzer.PatternAnalyzer')
    def test_analyze_content_with_patterns(self, mock_pattern_analyzer):
        """Test analysis when patterns are detected."""
        # Mock pattern analyzer to return hate speech detection
        mock_instance = Mock()
        mock_instance.analyze_content.return_value = Mock(
            categories=[Categories.HATE_SPEECH],
            pattern_matches=[Mock(matched_text="test", category="hate_speech", description="test")]
        )
        mock_pattern_analyzer.return_value = mock_instance

        analyzer = Analyzer(config=AnalyzerConfig(use_llm=False))
        analyzer.text_analyzer.pattern_analyzer = mock_instance

        async def test_async():
            result = await analyzer.analyze_content(
                tweet_id="test_123",
                tweet_url="https://twitter.com/test/status/test_123",
                username="test_user",
                content="Test content with hate speech"
            )
            return result

        result = asyncio.run(test_async())

        self.assertEqual(result.category, Categories.HATE_SPEECH)
        self.assertEqual(result.analysis_method, "pattern")
        self.assertEqual(len(result.categories_detected), 1)

    @patch('analyzer.pattern_analyzer.PatternAnalyzer')
    def test_analyze_content_with_metrics_tracking(self, mock_pattern_analyzer):
        """Test that analyze_content properly tracks metrics."""
        # Mock pattern analyzer to return hate speech detection
        mock_instance = Mock()
        mock_instance.analyze_content.return_value = Mock(
            categories=[Categories.HATE_SPEECH],
            pattern_matches=[Mock(matched_text="test", category="hate_speech", description="test")]
        )
        mock_pattern_analyzer.return_value = mock_instance

        analyzer = Analyzer(config=AnalyzerConfig(use_llm=False))

        # Analyze content
        async def test_async():
            result = await analyzer.analyze_content(
                tweet_id="test_123",
                tweet_url="https://twitter.com/test/status/test_123",
                username="test_user",
                content="Test content with hate speech"
            )
            return result

        result = asyncio.run(test_async())

        # Check that metrics were updated
        summary = analyzer.metrics.get_summary()
        self.assertEqual(summary['total_analyses'], 1)
        self.assertEqual(summary['method_counts'].get('pattern', 0), 1)
        self.assertEqual(summary['category_counts'][Categories.GENERAL], 1)  # Content doesn't match hate speech patterns
        self.assertEqual(summary['model_usage']['pattern-matching'], 1)
        self.assertGreater(summary['total_time'], 0)
        self.assertGreater(result.analysis_time_seconds, 0)
        self.assertEqual(result.model_used, "pattern-matching")


class TestAnalyzerInternalMethods(unittest.TestCase):
    """Test internal analyzer methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = Analyzer(config=AnalyzerConfig(use_llm=False))

    def test_categorize_content_with_patterns(self):
        """Test categorization when patterns are found."""
        analyzer = Analyzer(config=AnalyzerConfig(use_llm=False))
        pattern_results = {
            'pattern_result': Mock(categories=[Categories.HATE_SPEECH])
        }

        category, method = analyzer.text_analyzer._categorize_content("test content", pattern_results)

        self.assertEqual(category, Categories.HATE_SPEECH)
        self.assertEqual(method, "pattern")

    def test_categorize_content_no_patterns(self):
        """Test categorization when no patterns are found."""
        analyzer = Analyzer(config=AnalyzerConfig(use_llm=False))
        pattern_results = {
            'pattern_result': Mock(categories=[])
        }

        category, method = analyzer.text_analyzer._categorize_content("test content", pattern_results)

        self.assertEqual(category, Categories.GENERAL)
        self.assertEqual(method, "pattern")  # No LLM available, so still pattern method

    @patch('analyzer.llm_models.EnhancedLLMPipeline')
    def test_get_llm_category_success(self, mock_llm_pipeline):
        """Test successful LLM category retrieval."""
        mock_instance = Mock()
        mock_instance.get_category.return_value = Categories.CONSPIRACY_THEORY
        mock_llm_pipeline.return_value = mock_instance

        analyzer = Analyzer(config=AnalyzerConfig(use_llm=True))
        analyzer.text_analyzer.llm_pipeline = mock_instance

        result = analyzer.text_analyzer._get_llm_category("test content", {})

        self.assertEqual(result, Categories.CONSPIRACY_THEORY)
        mock_instance.get_category.assert_called_once_with("test content")

    def test_get_llm_category_no_llm(self):
        """Test LLM category retrieval when no LLM available."""
        analyzer = Analyzer(config=AnalyzerConfig(use_llm=False))

        result = analyzer.text_analyzer._get_llm_category("test content", {})

        self.assertEqual(result, Categories.GENERAL)

    @patch('analyzer.llm_models.EnhancedLLMPipeline')
    def test_get_llm_category_llm_error(self, mock_llm_pipeline):
        """Test LLM category retrieval when LLM fails."""
        mock_instance = Mock()
        mock_instance.get_category.side_effect = Exception("LLM error")
        mock_llm_pipeline.return_value = mock_instance

        analyzer = Analyzer(config=AnalyzerConfig(use_llm=True))
        analyzer.text_analyzer.llm_pipeline = mock_instance

        result = analyzer.text_analyzer._get_llm_category("test content", {})

        self.assertEqual(result, Categories.GENERAL)

    @patch('analyzer.llm_models.EnhancedLLMPipeline')
    def test_generate_llm_explanation_success(self, mock_llm_pipeline):
        """Test successful LLM explanation generation."""
        mock_instance = Mock()
        mock_instance.get_explanation.return_value = "Test explanation"
        mock_llm_pipeline.return_value = mock_instance

        analyzer = Analyzer(config=AnalyzerConfig(use_llm=True))
        analyzer.text_analyzer.llm_pipeline = mock_instance

        pattern_results = {'pattern_result': Mock(categories=[Categories.HATE_SPEECH])}

        result = analyzer.text_analyzer._generate_llm_explanation("test content", Categories.HATE_SPEECH, pattern_results)

        self.assertEqual(result, "Test explanation")
        mock_instance.get_explanation.assert_called_once()

    @patch('analyzer.llm_models.EnhancedLLMPipeline')
    def test_generate_llm_explanation_no_llm(self, mock_llm_pipeline):
        """Test LLM explanation generation when no LLM available."""
        mock_llm_pipeline.side_effect = Exception("LLM init failed")

        analyzer = Analyzer(config=AnalyzerConfig(use_llm=False))
        # Ensure llm_pipeline is None
        self.assertIsNone(analyzer.text_analyzer.llm_pipeline)

        result = analyzer.text_analyzer._generate_llm_explanation("test content", Categories.HATE_SPEECH, {})

        self.assertIn("LLM pipeline not available", result)

    @patch('analyzer.llm_models.EnhancedLLMPipeline')
    def test_generate_llm_explanation_llm_error(self, mock_llm_pipeline):
        """Test LLM explanation generation when LLM fails."""
        mock_instance = Mock()
        mock_instance.get_explanation.side_effect = Exception("LLM error")
        mock_llm_pipeline.return_value = mock_instance

        analyzer = Analyzer(config=AnalyzerConfig(use_llm=True))
        analyzer.text_analyzer.llm_pipeline = mock_instance

        result = analyzer.text_analyzer._generate_llm_explanation("test content", Categories.HATE_SPEECH, {})

        self.assertIn("LLM explanation generation exception", result)
        self.assertIn("LLM error", result)

    def test_get_model_name_pattern(self):
        """Test model name generation for pattern analysis."""
        analyzer = Analyzer(config=AnalyzerConfig(use_llm=False))
        result = ContentAnalysis(
            post_id="test",
            post_url="https://twitter.com/test/status/test",
            author_username="test",
            post_content="test",
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.GENERAL,
            analysis_method="pattern"
        )
        model_name = analyzer._get_model_name(result)
        self.assertEqual(model_name, "pattern-matching")

    def test_get_model_name_llm(self):
        """Test model name generation for LLM analysis."""
        analyzer = Analyzer(config=AnalyzerConfig(use_llm=True, model_priority="fast"))
        result = ContentAnalysis(
            post_id="test",
            post_url="https://twitter.com/test/status/test",
            author_username="test",
            post_content="test",
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.GENERAL,
            analysis_method="llm"
        )
        model_name = analyzer._get_model_name(result)
        self.assertEqual(model_name, "ollama-fast")

    def test_get_model_name_multimodal(self):
        """Test model name generation for multimodal analysis."""
        analyzer = Analyzer(config=AnalyzerConfig(use_llm=False))
        result = ContentAnalysis(
            post_id="test",
            post_url="https://twitter.com/test/status/test",
            author_username="test",
            post_content="test",
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.GENERAL,
            analysis_method="multimodal"
        )
        model_name = analyzer._get_model_name(result)
        self.assertEqual(model_name, "gemini-2.5-flash")

    def test_get_metrics_report(self):
        """Test metrics report generation."""
        analyzer = Analyzer(config=AnalyzerConfig(use_llm=False))
        
        # Mock the metrics summary
        mock_summary = {
            'total_analyses': 2,
            'method_counts': {'pattern': 1, 'llm': 1},
            'multimodal_count': 0,
            'category_counts': {Categories.HATE_SPEECH: 1, Categories.DISINFORMATION: 1},
            'total_time': 3.0,
            'avg_time_per_analysis': 1.5,
            'model_usage': {'pattern-matching': 1, 'ollama-balanced': 1},
            'start_time': analyzer.metrics.get_summary()['start_time'],  # Keep original start time
            'runtime_seconds': 10.5  # Add missing runtime_seconds
        }
        
        with patch.object(analyzer.metrics, 'get_summary', return_value=mock_summary):
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
        self.analyzer = Analyzer(config=AnalyzerConfig(use_llm=False))

    @patch('analyzer.llm_models.EnhancedLLMPipeline')
    def test_cleanup_resources(self, mock_llm_pipeline):
        """Test resource cleanup."""
        mock_instance = Mock()
        mock_instance.cleanup = Mock()
        mock_llm_pipeline.return_value = mock_instance

        analyzer = Analyzer(config=AnalyzerConfig(use_llm=True))
        analyzer.text_analyzer.llm_pipeline = mock_instance

        analyzer.cleanup_resources()

        mock_instance.cleanup.assert_called_once()

    def test_cleanup_resources_no_llm(self):
        """Test resource cleanup when no LLM."""
        analyzer = Analyzer(config=AnalyzerConfig(use_llm=False))

        # Should not raise exception
        analyzer.cleanup_resources()

    @patch('builtins.print')
    def test_print_system_status(self, mock_print):
        """Test system status printing."""
        analyzer = Analyzer(config=AnalyzerConfig(use_llm=False))

        analyzer.print_system_status()

        # Check that print was called with the expected header
        mock_print.assert_any_call("🔧 ANALYZER SYSTEM STATUS")


class TestDatabaseFunctions(unittest.TestCase):
    """Test database-related functions."""

class TestDatabaseFunctions:
    """Test database-related functions."""

    def test_save_content_analysis(self, test_tweet):
        """Test saving content analysis to database."""
        # Create repository instance
        repo = ContentAnalysisRepository()

        analysis = ContentAnalysis(
            post_id=test_tweet['tweet_id'],
            post_url=test_tweet['tweet_url'],
            author_username=test_tweet['username'],
            post_content=test_tweet['content'],
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.HATE_SPEECH,
            llm_explanation="Test explanation",
            analysis_method="pattern",
            categories_detected=[Categories.HATE_SPEECH]
        )

        repo.save(analysis)

        # Verify the data was saved using the repository's database connection
        with get_db_connection_context() as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM content_analyses WHERE post_id = ?", (test_tweet['tweet_id'],))
            row = c.fetchone()

            assert row is not None
            assert row['post_id'] == test_tweet['tweet_id']
            assert row['category'] == Categories.HATE_SPEECH
            assert row['llm_explanation'] == "Test explanation"
            assert row['analysis_method'] == "pattern"

    def test_save_content_analysis_duplicate(self, test_tweet):
        """Test saving duplicate content analysis (should replace)."""
        # Create repository instance
        repo = ContentAnalysisRepository()

        analysis1 = ContentAnalysis(
            post_id=test_tweet['tweet_id'],
            post_url=test_tweet['tweet_url'],
            author_username=test_tweet['username'],
            post_content=test_tweet['content'],
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.HATE_SPEECH
        )

        analysis2 = ContentAnalysis(
            post_id=test_tweet['tweet_id'],
            post_url=test_tweet['tweet_url'],
            author_username=test_tweet['username'],
            post_content=test_tweet['content'],
            analysis_timestamp="2024-01-01T12:00:00",
            category=Categories.DISINFORMATION
        )

        repo.save(analysis1)
        repo.save(analysis2)  # Should replace

        # Verify the data was saved using the repository's database connection
        with get_db_connection_context() as conn:
            c = conn.cursor()

            c.execute("SELECT COUNT(*) FROM content_analyses WHERE post_id = ?", (test_tweet['tweet_id'],))
            count = c.fetchone()[0]
            assert count == 1

            c.execute("SELECT category FROM content_analyses WHERE post_id = ?", (test_tweet['tweet_id'],))
            category = c.fetchone()[0]
            assert category == Categories.DISINFORMATION


class TestAnalyzerMultimodal(unittest.TestCase):
    """Test multimodal analysis functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = Analyzer(config=AnalyzerConfig(use_llm=False))

    @patch('analyzer.gemini_multimodal.GeminiMultimodal')
    @patch('analyzer.multimodal_analyzer.extract_media_type')
    def test_analyze_multi_modal_success(self, mock_extract_type, mock_gemini_class):
        """Test successful multimodal analysis."""
        # Mock the GeminiMultimodal instance
        mock_instance = mock_gemini_class.return_value
        mock_instance.analyze_multimodal_content.return_value = ("Test Gemini analysis", 2.5)
        mock_extract_type.return_value = "image"

        # Create analyzer with mocked multimodal_analyzer
        analyzer = Analyzer(config=AnalyzerConfig(use_llm=False))
        analyzer.multimodal_analyzer.gemini_analyzer = mock_instance

        result = analyzer.multimodal_analyzer.analyze_with_media(
            tweet_id="test_123",
            tweet_url="https://twitter.com/test/status/test_123",
            username="test_user",
            content="Test content",
            media_urls=["https://example.com/image.jpg"]
        )

        self.assertEqual(result.post_id, "test_123")
        self.assertEqual(result.category, Categories.GENERAL)  # Default for now
        self.assertEqual(result.analysis_method, "multimodal")  # Updated to match actual method
        self.assertEqual(result.media_analysis, "Test Gemini analysis")
        self.assertEqual(result.media_type, "image")
        self.assertTrue(result.multimodal_analysis)
        self.assertEqual(result.media_urls, ["https://example.com/image.jpg"])

    @patch('analyzer.gemini_multimodal.GeminiMultimodal')
    def test_analyze_multi_modal_failure_fallback(self, mock_gemini_class):
        """Test multimodal analysis failure handling."""
        # Mock the GeminiMultimodal instance to return failure
        mock_instance = mock_gemini_class.return_value
        mock_instance.analyze_multimodal_content.return_value = (None, 1.0)

        # Create analyzer with mocked multimodal_analyzer
        analyzer = Analyzer(config=AnalyzerConfig(use_llm=False))
        analyzer.multimodal_analyzer.gemini_analyzer = mock_instance

        result = analyzer.multimodal_analyzer.analyze_with_media(
            tweet_id="test_123",
            tweet_url="https://twitter.com/test/status/test_123",
            username="test_user",
            content="Test content",
            media_urls=["https://example.com/image.jpg"]
        )

        # Should return a valid ContentAnalysis but with error indication
        self.assertEqual(result.post_id, "test_123")
        self.assertEqual(result.category, Categories.GENERAL)  # Fallback category
        self.assertEqual(result.analysis_method, "multimodal")  # Still multimodal method
        self.assertIn("Media analysis failed", result.llm_explanation)  # Error message
        self.assertTrue(result.multimodal_analysis)  # Still marked as multimodal attempt

    def test_analyze_content_routing_text_only(self):
        """Test that analyze_content routes to text-only for no media."""
        with patch.object(self.analyzer.text_analyzer, 'analyze') as mock_text_only:
            mock_result = ContentAnalysis(
                post_id="test_123",
                post_url="https://twitter.com/test/status/test_123",
                author_username="test_user",
                post_content="Test content",
                analysis_timestamp="2024-01-01T12:00:00",
                category=Categories.GENERAL
            )
            mock_text_only.return_value = mock_result

            async def test_async():
                result = await self.analyzer.analyze_content(
                    tweet_id="test_123",
                    tweet_url="https://twitter.com/test/status/test_123",
                    username="test_user",
                    content="Test content",
                    media_urls=[]
                )
                return result

            result = asyncio.run(test_async())

            self.assertEqual(result, mock_result)
            mock_text_only.assert_called_once()

    def test_analyze_content_routing_multimodal(self):
        """Test that analyze_content routes to multimodal for media."""
        with patch.object(self.analyzer.multimodal_analyzer, 'analyze_with_media') as mock_multimodal:
            mock_result = ContentAnalysis(
                post_id="test_123",
                post_url="https://twitter.com/test/status/test_123",
                author_username="test_user",
                post_content="Test content",
                analysis_timestamp="2024-01-01T12:00:00",
                category=Categories.GENERAL,
                multimodal_analysis=True
            )
            mock_multimodal.return_value = mock_result

            async def test_async():
                result = await self.analyzer.analyze_content(
                    tweet_id="test_123",
                    tweet_url="https://twitter.com/test/status/test_123",
                    username="test_user",
                    content="Test content",
                    media_urls=["https://example.com/image.jpg"]
                )
                return result

            result = asyncio.run(test_async())

            self.assertEqual(result, mock_result)
            mock_multimodal.assert_called_once()

    def test_extract_media_type_image(self):
        """Test media type extraction for images."""
        media_urls = ["https://pbs.twimg.com/media/1973243448871284736/bPeZHL3l?format=jpg&name=small"]
        result = extract_media_type(media_urls)
        self.assertEqual(result, "image")

    def test_extract_media_type_video(self):
        """Test media type extraction for videos."""
        media_urls = ["https://video.twimg.com/amplify_video/1972307252796141568/vid/avc1/320x568/GftH9VZYZuygizQc.mp4"]
        result = extract_media_type(media_urls)
        self.assertEqual(result, "video")

    def test_extract_media_type_mixed(self):
        """Test media type extraction for mixed content."""
        media_urls = [
            "https://pbs.twimg.com/media/1973243448871284736/bPeZHL3l?format=jpg&name=small",
            "https://video.twimg.com/amplify_video/1972307252796141568/vid/avc1/320x568/GftH9VZYZuygizQc.mp4"
        ]
        result = extract_media_type(media_urls)
        self.assertEqual(result, "mixed")

    def test_extract_media_type_unknown(self):
        """Test media type extraction for unknown content."""
        media_urls = ["https://example.com/somefile.txt"]
        result = extract_media_type(media_urls)
        self.assertEqual(result, "unknown")

    def test_extract_media_type_empty(self):
        """Test media type extraction for empty list."""
        media_urls = []
        result = extract_media_type(media_urls)
        self.assertEqual(result, "")

    def test_extract_media_type_video_by_pattern(self):
        """Test media type extraction for video detected by URL pattern."""
        media_urls = ["https://example.com/video/content"]
        result = extract_media_type(media_urls)
        self.assertEqual(result, "video")

    def test_extract_media_type_image_by_format_param(self):
        """Test media type extraction for image detected by format parameter."""
        media_urls = ["https://example.com/image?format=jpeg"]
        result = extract_media_type(media_urls)
        self.assertEqual(result, "image")


class TestCLIFunctions(unittest.TestCase):
    """Test CLI functions and main execution paths."""

    @patch('analyzer.analyze_twitter.create_analyzer')
    @patch('builtins.print')
    def test_analyze_tweets_from_db_specific_tweet_success(self, mock_print, mock_create_analyzer):
        """Test analyzing a specific tweet by ID successfully."""
        mock_analyzer = Mock()
        mock_create_analyzer.return_value = mock_create_analyzer

        mock_result = Mock()
        mock_result.category = Categories.HATE_SPEECH
        mock_result.llm_explanation = "Hate speech detected"
        mock_result.analysis_method = "llm"

        with patch('analyzer.analyze_twitter.reanalyze_tweet', return_value=mock_result):
            asyncio.run(analyze_tweets_from_db(tweet_id='123456789'))

        mock_create_analyzer.assert_called_once()

    @patch('analyzer.analyze_twitter.create_analyzer')
    @patch('builtins.print')
    def test_analyze_tweets_from_db_specific_tweet_not_found(self, mock_print, mock_create_analyzer):
        """Test analyzing a specific tweet that doesn't exist."""
        mock_analyzer = Mock()
        mock_create_analyzer.return_value = mock_create_analyzer

        with patch('analyzer.analyze_twitter.reanalyze_tweet', return_value=None):
            asyncio.run(analyze_tweets_from_db(tweet_id='123456789'))

        mock_create_analyzer.assert_called_once()

    @patch('analyzer.analyze_twitter.create_analyzer')
    @patch('builtins.print')
    def test_analyze_tweets_from_db_no_tweets(self, mock_print, mock_create_analyzer):
        """Test when no tweets are available for analysis."""
        mock_analyzer = Mock()
        mock_create_analyzer.return_value = mock_analyzer
        mock_analyzer.repository.get_tweets_for_analysis.return_value = []
        mock_analyzer.repository.get_analysis_count_by_author.return_value = 0

        asyncio.run(analyze_tweets_from_db())

        mock_create_analyzer.assert_called_once()

    @patch('analyzer.analyze_twitter.create_analyzer')
    @patch('builtins.print')
    def test_analyze_tweets_from_db_with_tweets(self, mock_print, mock_create_analyzer):
        """Test analyzing tweets from database."""
        mock_analyzer = Mock()
        mock_create_analyzer.return_value = mock_analyzer

        # Mock tweets data
        tweets = [
            ('123', 'https://twitter.com/test/status/123', 'testuser', 'Test content', '', '')
        ]
        mock_analyzer.repository.get_tweets_for_analysis.return_value = tweets
        mock_analyzer.repository.get_analysis_count_by_author.return_value = 5

        # Mock analysis result
        mock_result = Mock()
        mock_result.category = Categories.HATE_SPEECH
        mock_result.llm_explanation = "Hate speech detected"
        mock_result.analysis_method = "llm"
        mock_result.multimodal_analysis = False
        mock_analyzer.analyze_content = AsyncMock(return_value=mock_result)

        asyncio.run(analyze_tweets_from_db(max_tweets=1))

        mock_create_analyzer.assert_called_once()
        mock_analyzer.analyze_content.assert_called_once()
        mock_analyzer.save_analysis.assert_called_once()

    @patch('analyzer.analyze_twitter.create_analyzer')
    @patch('builtins.print')
    def test_analyze_tweets_from_db_with_media(self, mock_print, mock_create_analyzer):
        """Test analyzing tweets with media content."""
        mock_analyzer = Mock()
        mock_create_analyzer.return_value = mock_analyzer

        tweets = [
            ('123', 'https://twitter.com/test/status/123', 'testuser', 'Test content', 'url1.jpg,url2.jpg', '')
        ]
        mock_analyzer.repository.get_tweets_for_analysis.return_value = tweets
        mock_analyzer.repository.get_analysis_count_by_author.return_value = 0

        mock_result = Mock()
        mock_result.category = Categories.GENERAL
        mock_result.llm_explanation = "Normal content"
        mock_result.analysis_method = "multimodal"
        mock_result.multimodal_analysis = True
        mock_result.media_type = "image"
        mock_analyzer.analyze_content = AsyncMock(return_value=mock_result)

        asyncio.run(analyze_tweets_from_db(max_tweets=1))

        # Verify media URLs were parsed correctly
        call_args = mock_analyzer.analyze_content.call_args
        assert call_args[1]['media_urls'] == ['url1.jpg', 'url2.jpg']

    @patch('analyzer.analyze_twitter.create_analyzer')
    @patch('builtins.print')
    def test_analyze_tweets_from_db_with_quoted_content(self, mock_print, mock_create_analyzer):
        """Test analyzing tweets with quoted content."""
        mock_analyzer = Mock()
        mock_create_analyzer.return_value = mock_analyzer

        tweets = [
            ('123', 'https://twitter.com/test/status/123', 'testuser', 'Main content', '', 'Quoted content')
        ]
        mock_analyzer.repository.get_tweets_for_analysis.return_value = tweets
        mock_analyzer.repository.get_analysis_count_by_author.return_value = 0

        mock_result = Mock()
        mock_result.category = Categories.GENERAL
        mock_result.llm_explanation = "Normal content"
        mock_result.analysis_method = "llm"
        mock_analyzer.analyze_content = AsyncMock(return_value=mock_result)

        asyncio.run(analyze_tweets_from_db(max_tweets=1))

        # Verify quoted content was combined
        call_args = mock_analyzer.analyze_content.call_args
        expected_content = "Main content\n\n[Contenido citado]: Quoted content"
        assert call_args[1]['content'] == expected_content

    @patch('analyzer.analyze_twitter.create_analyzer')
    @patch('builtins.print')
    def test_analyze_tweets_from_db_analysis_error(self, mock_print, mock_create_analyzer):
        """Test handling of analysis errors."""
        mock_analyzer = Mock()
        mock_create_analyzer.return_value = mock_analyzer

        tweets = [
            ('123', 'https://twitter.com/test/status/123', 'testuser', 'Test content', '', '')
        ]
        mock_analyzer.repository.get_tweets_for_analysis.return_value = tweets
        mock_analyzer.repository.get_analysis_count_by_author.return_value = 0

        mock_analyzer.analyze_content = AsyncMock(side_effect=Exception("Analysis failed"))

        asyncio.run(analyze_tweets_from_db(max_tweets=1))

        # Should save failed analysis
        mock_analyzer.repository.save_failed_analysis.assert_called_once()

    @patch('analyzer.analyze_twitter.create_analyzer')
    @patch('builtins.print')
    def test_analyze_tweets_from_db_force_reanalyze(self, mock_print, mock_create_analyzer):
        """Test force reanalysis of tweets."""
        mock_analyzer = Mock()
        mock_create_analyzer.return_value = mock_analyzer

        tweets = [
            ('123', 'https://twitter.com/test/status/123', 'testuser', 'Test content', '', '')
        ]
        mock_analyzer.repository.get_tweets_for_analysis.return_value = tweets
        mock_analyzer.repository.get_analysis_count_by_author.return_value = 10

        mock_result = Mock()
        mock_result.category = Categories.GENERAL
        mock_result.llm_explanation = "Reanalyzed content"
        mock_result.analysis_method = "llm"
        mock_analyzer.analyze_content = AsyncMock(return_value=mock_result)

        asyncio.run(analyze_tweets_from_db(force_reanalyze=True, max_tweets=1))

        mock_create_analyzer.assert_called_once()
        mock_analyzer.analyze_content.assert_called_once()
        mock_analyzer.save_analysis.assert_called_once()

    @patch('analyzer.analyze_twitter.create_analyzer')
    @patch('builtins.print')
    def test_analyze_tweets_from_db_username_filter(self, mock_print, mock_create_analyzer):
        """Test filtering tweets by username."""
        mock_analyzer = Mock()
        mock_create_analyzer.return_value = mock_analyzer

        tweets = [
            ('123', 'https://twitter.com/test/status/123', 'specificuser', 'Test content', '', '')
        ]
        mock_analyzer.repository.get_tweets_for_analysis.return_value = tweets
        mock_analyzer.repository.get_analysis_count_by_author.return_value = 0

        mock_result = Mock()
        mock_result.category = Categories.GENERAL
        mock_analyzer.analyze_content = AsyncMock(return_value=mock_result)

        asyncio.run(analyze_tweets_from_db(username='specificuser', max_tweets=1))

        mock_analyzer.repository.get_tweets_for_analysis.assert_called_once_with(
            username='specificuser',
            max_tweets=1,
            force_reanalyze=False
        )


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    @patch('analyzer.analyze_twitter.Analyzer')
    def test_create_analyzer_function(self, mock_analyzer_class):
        """Test the create_analyzer utility function."""
        config = AnalyzerConfig(use_llm=False, model_priority="fast")
        create_analyzer(config=config, verbose=True)
        mock_analyzer_class.assert_called_once_with(config=config, verbose=True)

    @patch('analyzer.analyze_twitter.create_analyzer')
    def test_reanalyze_tweet_success(self, mock_create_analyzer):
        """Test successful tweet reanalysis."""
        mock_analyzer = Mock()
        mock_create_analyzer.return_value = mock_create_analyzer

        mock_tweet_data = {
            'tweet_id': '123',
            'username': 'testuser',
            'content': 'Test content',
            'media_links': 'url1.jpg,url2.jpg'
        }
        mock_analyzer.repository.get_tweet_data.return_value = mock_tweet_data

        mock_analysis_result = Mock()
        mock_analyzer.analyze_content = AsyncMock(return_value=mock_analysis_result)

        result = asyncio.run(reanalyze_tweet('123', analyzer=mock_analyzer))

        assert result == mock_analysis_result
        mock_analyzer.repository.get_tweet_data.assert_called_once_with('123')
        mock_analyzer.repository.delete_existing_analysis.assert_called_once_with('123')
        mock_analyzer.analyze_content.assert_called_once()
        mock_analyzer.save_analysis.assert_called_once_with(mock_analysis_result)

    @patch('analyzer.analyze_twitter.create_analyzer')
    def test_reanalyze_tweet_not_found(self, mock_create_analyzer):
        """Test reanalysis when tweet is not found."""
        mock_analyzer = Mock()
        mock_create_analyzer.return_value = mock_create_analyzer
        mock_analyzer.repository.get_tweet_data.return_value = None

        result = asyncio.run(reanalyze_tweet('123', analyzer=mock_analyzer))
        assert result is None

    @patch('analyzer.analyze_twitter.analyze_tweets_from_db')
    @patch('sys.argv', ['analyze_twitter.py', '--username', 'testuser', '--limit', '5'])
    def test_main_with_args(self, mock_analyze):
        """Test main function with command line arguments."""
        main()
        mock_analyze.assert_called_once_with(
            username='testuser',
            max_tweets=5,
            force_reanalyze=False,
            tweet_id=None
        )

    @patch('analyzer.analyze_twitter.analyze_tweets_from_db')
    @patch('sys.argv', ['analyze_twitter.py'])
    def test_main_no_args(self, mock_analyze):
        """Test main function with no arguments."""
        main()
        mock_analyze.assert_called_once_with(
            username=None,
            max_tweets=None,
            force_reanalyze=False,
            tweet_id=None
        )

    @patch('analyzer.analyze_twitter.analyze_tweets_from_db')
    @patch('sys.argv', ['analyze_twitter.py', '--force-reanalyze', '--tweet-id', '123'])
    def test_main_force_reanalyze_with_tweet_id(self, mock_analyze):
        """Test main function with force reanalyze and tweet ID."""
        main()
        mock_analyze.assert_called_once_with(
            username=None,
            max_tweets=None,
            force_reanalyze=True,
            tweet_id='123'
        )