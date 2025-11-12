#!/usr/bin/env python3
"""
Comprehensive unit tests for the Analyzer class and related functionality.
Tests all methods, edge cases, and integration scenarios.
"""

import unittest
from unittest.mock import Mock, patch, AsyncMock
import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from analyzer.analyze_twitter import (
    Analyzer, create_analyzer, reanalyze_tweet
)
from scripts.analyzer_cli import analyze_tweets_cli, main
from analyzer.config import AnalyzerConfig
from analyzer.models import ContentAnalysis
from analyzer.repository import ContentAnalysisRepository
from analyzer.categories import Categories
from analyzer.external_analyzer import ExternalAnalysisResult
from database import get_db_connection_context

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
        self.assertEqual(analysis.analysis_stages, "")  # Empty by default, set by analyzer
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
            categories_detected=[Categories.HATE_SPEECH, Categories.ANTI_IMMIGRATION],
            pattern_matches=[{"matched_text": "test", "category": "hate_speech"}]
        )

        self.assertEqual(analysis.category, Categories.HATE_SPEECH)
        self.assertEqual(len(analysis.categories_detected), 2)
        self.assertTrue(analysis.has_multiple_categories)
        self.assertEqual(analysis.get_secondary_categories(), [Categories.ANTI_IMMIGRATION])

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

    @patch('analyzer.llm_pipeline.EnhancedLLMPipeline')
    def test_analyzer_init_default(self, mock_llm_pipeline):
        """Test analyzer initialization with default config."""
        analyzer = Analyzer(config=AnalyzerConfig())

        self.assertFalse(analyzer.config.enable_external_analysis)  # External analysis disabled by default
        # REMOVED: self.assertEqual(analyzer.config.model_priority, "balanced")  # model_priority no longer exists
        self.assertFalse(analyzer.config.verbose)

    @patch('analyzer.llm_pipeline.EnhancedLLMPipeline')
    def test_analyzer_init_no_llm(self, mock_llm_pipeline):
        """Test analyzer initialization without external analysis."""
        analyzer = Analyzer(config=AnalyzerConfig(enable_external_analysis=False))

        self.assertFalse(analyzer.config.enable_external_analysis)

    @patch('analyzer.llm_pipeline.EnhancedLLMPipeline')
    def test_analyzer_init_fast_priority(self, mock_llm_pipeline):
        """Test analyzer initialization with fast model priority."""
        analyzer = Analyzer(config=AnalyzerConfig())

        # REMOVED: self.assertEqual(analyzer.config.model_priority, "fast")  # model_priority no longer exists

    @patch('analyzer.llm_pipeline.EnhancedLLMPipeline')
    def test_analyzer_init_verbose(self, mock_llm_pipeline):
        """Test analyzer initialization with verbose output."""
        analyzer = Analyzer(config=AnalyzerConfig(verbose=True))

        self.assertTrue(analyzer.config.verbose)

    @patch('analyzer.llm_pipeline.EnhancedLLMPipeline')
    def test_analyzer_init_llm_failure(self, mock_llm_pipeline):
        """Test analyzer initialization when LLM fails."""
        analyzer = Analyzer(config=AnalyzerConfig())

        # REMOVED: self.assertIsNone(analyzer.text_analyzer.llm_pipeline)  # text_analyzer component removed



# Integration tests that require Ollama LLM to be running
# These tests verify end-to-end analysis functionality
class TestAnalyzerAnalysis(unittest.TestCase):
    """Test the main analysis functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = Analyzer(config=AnalyzerConfig())  # Disable LLM for faster tests

    def test_analyze_content_with_patterns(self):
        """Test analysis when patterns are detected."""
        analyzer = Analyzer(config=AnalyzerConfig(enable_external_analysis=True))

        # Mock the entire flow manager analyze_full method to avoid real LLM calls
        mock_result = Mock()
        mock_result.category = Categories.HATE_SPEECH
        mock_result.local_explanation = "Mock explanation"
        mock_result.external_explanation = "Mock external explanation"
        mock_result.stages.to_string.return_value = "pattern,category_detection,explanation,external"
        mock_result.pattern_data = {
            'pattern_matches': [{"matched_text": "test", "category": "hate_speech", "description": "test"}],
            'topic_classification': {'categories': [Categories.HATE_SPEECH], 'primary_category': Categories.HATE_SPEECH}
        }
        mock_result.verification_data = None
        mock_result.media_description = ""
        analyzer.flow_manager.analyze_full = AsyncMock(return_value=mock_result)

        async def test_async():
            result = await analyzer.analyze_content(
                tweet_id="test_123",
                tweet_url="https://twitter.com/test/status/test_123",
                username="test_user",
                content="Test content with hate speech"
            )
            return result

        result = asyncio.run(test_async())

        # Should detect hate speech
        self.assertEqual(result.category, Categories.HATE_SPEECH)
        self.assertEqual(result.analysis_stages, "pattern,category_detection,explanation,external")
        self.assertEqual(len(result.categories_detected), 1)
        
        # Verify the flow manager was called correctly
        analyzer.flow_manager.analyze_full.assert_called_once_with(
            content="Test content with hate speech",
            media_urls=None,
            admin_override=False,
            force_disable_external=False
        )


class TestAnalyzerUtilityMethods(unittest.TestCase):
    """Test analyzer utility methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = Analyzer(config=AnalyzerConfig())

    def test_cleanup_resources(self):
        """Test resource cleanup."""
        analyzer = Analyzer(config=AnalyzerConfig())
        
        # Mock cleanup method on flow_manager
        analyzer.flow_manager.cleanup = Mock()

        analyzer.cleanup_resources()

        analyzer.flow_manager.cleanup.assert_called_once()

    def test_cleanup_resources_no_llm(self):
        """Test resource cleanup when no cleanup method exists."""
        analyzer = Analyzer(config=AnalyzerConfig())

        # Should not raise exception even without cleanup method
        analyzer.cleanup_resources()

    @patch('builtins.print')
    def test_print_system_status(self, mock_print):
        """Test system status printing."""
        analyzer = Analyzer(config=AnalyzerConfig())

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
            local_explanation="Test explanation",
            analysis_stages="pattern",
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
            assert row['local_explanation'] == "Test explanation"
            assert row['analysis_stages'] == "pattern"

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


class TestCLIFunctions(unittest.TestCase):
    """Test CLI functions and main execution paths."""

    @patch('scripts.analyzer_cli._handle_single_tweet_analysis')
    @patch('builtins.print')
    def test_analyze_tweets_cli_specific_tweet_success(self, mock_print, mock_handle):
        """Test analyzing a specific tweet by ID successfully."""
        mock_result = Mock()
        mock_result.category = Categories.HATE_SPEECH
        mock_result.local_explanation = "Hate speech detected"
        mock_result.analysis_stages = "llm"

        mock_handle.return_value = None  # The function doesn't return anything

        asyncio.run(analyze_tweets_cli(tweet_id='123456789'))

        mock_handle.assert_called_once_with('123456789', False)

    @patch('scripts.analyzer_cli._handle_single_tweet_analysis')
    @patch('builtins.print')
    def test_analyze_tweets_cli_specific_tweet_not_found(self, mock_print, mock_handle):
        """Test analyzing a specific tweet that doesn't exist."""
        mock_handle.return_value = None

        asyncio.run(analyze_tweets_cli(tweet_id='123456789'))

        mock_handle.assert_called_once_with('123456789', False)

    @patch('scripts.analyzer_cli._setup_analyzer_and_get_tweets')
    @patch('builtins.print')
    def test_analyze_tweets_cli_no_tweets(self, mock_print, mock_setup):
        """Test when no tweets are available for analysis."""
        mock_analyzer = Mock()
        mock_analyzer.repository.get_tweets_for_analysis.return_value = []
        mock_analyzer.repository.get_analysis_count_by_author.return_value = 0
        
        mock_setup.return_value = (mock_analyzer, [], 0, 0)

        asyncio.run(analyze_tweets_cli())

        mock_setup.assert_called_once()

    @patch('scripts.analyzer_cli._setup_analyzer_and_get_tweets')
    @patch('builtins.print')
    def test_analyze_tweets_cli_with_tweets(self, mock_print, mock_setup):
        """Test analyzing tweets from database."""
        mock_analyzer = Mock()
        mock_setup.return_value = (mock_analyzer, [
            ('123', 'https://twitter.com/test/status/123', 'testuser', 'Test content', '', '')
        ], 5, 10)

        # Mock analysis result
        mock_result = Mock()
        mock_result.category = Categories.HATE_SPEECH
        mock_result.local_explanation = "Hate speech detected"
        mock_result.analysis_stages = "llm"
        mock_result.multimodal_analysis = False
        mock_result.verification_data = {'stage_timings': {'pattern_detection': 0.1, 'category_detection': 0.5, 'explanation': 0.3}}
        mock_analyzer.analyze_content = AsyncMock(return_value=mock_result)

        asyncio.run(analyze_tweets_cli(max_tweets=1))

        mock_setup.assert_called_once()
        mock_analyzer.analyze_content.assert_called_once()
        mock_analyzer.save_analysis.assert_called_once()

    @patch('scripts.analyzer_cli._setup_analyzer_and_get_tweets')
    @patch('builtins.print')
    def test_analyze_tweets_cli_with_media(self, mock_print, mock_setup):
        """Test analyzing tweets with media content."""
        mock_analyzer = Mock()
        mock_setup.return_value = (mock_analyzer, [
            ('123', 'https://twitter.com/test/status/123', 'testuser', 'Test content', 'url1.jpg,url2.jpg', '')
        ], 0, 5)

        mock_result = Mock()
        mock_result.category = Categories.GENERAL
        mock_result.local_explanation = "Normal content"
        mock_result.analysis_stages = "multimodal"
        mock_result.multimodal_analysis = True
        mock_result.media_type = "image"
        mock_result.verification_data = {'stage_timings': {'pattern_detection': 0.1, 'category_detection': 0.4, 'media_analysis': 1.2, 'explanation': 0.3}}
        mock_analyzer.analyze_content = AsyncMock(return_value=mock_result)

        asyncio.run(analyze_tweets_cli(max_tweets=1))

        # Verify media URLs were parsed correctly
        call_args = mock_analyzer.analyze_content.call_args
        assert call_args[1]['media_urls'] == ['url1.jpg', 'url2.jpg']

    @patch('scripts.analyzer_cli._setup_analyzer_and_get_tweets')
    @patch('builtins.print')
    def test_analyze_tweets_cli_with_quoted_content(self, mock_print, mock_setup):
        """Test analyzing tweets with quoted content."""
        mock_analyzer = Mock()
        mock_setup.return_value = (mock_analyzer, [
            ('123', 'https://twitter.com/test/status/123', 'testuser', 'Main content', '', 'Quoted content')
        ], 0, 3)

        mock_result = Mock()
        mock_result.category = Categories.GENERAL
        mock_result.local_explanation = "Normal content"
        mock_result.analysis_stages = "llm"
        mock_result.verification_data = {'stage_timings': {'pattern_detection': 0.1, 'category_detection': 0.4, 'explanation': 0.3}}
        mock_analyzer.analyze_content = AsyncMock(return_value=mock_result)

        asyncio.run(analyze_tweets_cli(max_tweets=1))

        # Verify quoted content was combined
        call_args = mock_analyzer.analyze_content.call_args
        expected_content = "Main content\n\n[Contenido citado]: Quoted content"
        assert call_args[1]['content'] == expected_content

    @patch('scripts.analyzer_cli._setup_analyzer_and_get_tweets')
    @patch('builtins.print')
    def test_analyze_tweets_cli_analysis_error(self, mock_print, mock_setup):
        """Test that analysis errors are handled gracefully and saved to database."""
        mock_analyzer = Mock()
        mock_setup.return_value = (mock_analyzer, [
            ('123', 'https://twitter.com/test/status/123', 'testuser', 'Test content', '', '')
        ], 0, 2)

        mock_analyzer.analyze_content = AsyncMock(side_effect=Exception("Analysis failed"))

        # Should handle error gracefully and continue (not raise exception)
        asyncio.run(analyze_tweets_cli(max_tweets=1))
        
        # Verify that save_failed_analysis was called
        mock_analyzer.repository.save_failed_analysis.assert_called_once()
        call_args = mock_analyzer.repository.save_failed_analysis.call_args
        assert call_args[1]['post_id'] == '123'
        assert call_args[1]['author_username'] == 'testuser'
        assert 'Analysis failed' in call_args[1]['error_message']

    @patch('scripts.analyzer_cli._setup_analyzer_and_get_tweets')
    @patch('builtins.print')
    def test_analyze_tweets_cli_force_reanalyze(self, mock_print, mock_setup):
        """Test force reanalysis of tweets."""
        mock_analyzer = Mock()
        mock_setup.return_value = (mock_analyzer, [
            ('123', 'https://twitter.com/test/status/123', 'testuser', 'Test content', '', '')
        ], 10, 15)

        mock_result = Mock()
        mock_result.category = Categories.GENERAL
        mock_result.local_explanation = "Reanalyzed content"
        mock_result.analysis_stages = "llm"
        mock_result.verification_data = {'stage_timings': {'pattern_detection': 0.1, 'category_detection': 0.4, 'explanation': 0.3}}
        mock_analyzer.analyze_content = AsyncMock(return_value=mock_result)

        asyncio.run(analyze_tweets_cli(force_reanalyze=True, max_tweets=1))

        mock_setup.assert_called_once()
        mock_analyzer.analyze_content.assert_called_once()
        mock_analyzer.save_analysis.assert_called_once()

    @patch('scripts.analyzer_cli._setup_analyzer_and_get_tweets')
    @patch('builtins.print')
    def test_analyze_tweets_cli_username_filter(self, mock_print, mock_setup):
        """Test filtering tweets by username."""
        mock_analyzer = Mock()
        mock_setup.return_value = (mock_analyzer, [
            ('123', 'https://twitter.com/test/status/123', 'specificuser', 'Test content', '', '')
        ], 0, 8)

        mock_result = Mock()
        mock_result.category = Categories.GENERAL
        mock_result.verification_data = {'stage_timings': {'pattern_detection': 0.1, 'category_detection': 0.4, 'explanation': 0.3}}
        mock_analyzer.analyze_content = AsyncMock(return_value=mock_result)

        asyncio.run(analyze_tweets_cli(usernames=['specificuser'], max_tweets=1))

        mock_setup.assert_called_once_with(['specificuser'], 1, False, False, False)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    @patch('analyzer.analyze_twitter.Analyzer')
    def test_create_analyzer_function(self, mock_analyzer_class):
        """Test the create_analyzer utility function."""
        config = AnalyzerConfig()
        create_analyzer(config=config, verbose=True)
        mock_analyzer_class.assert_called_once_with(config=config, verbose=True, fast_mode=False)

    @patch('analyzer.analyze_twitter.create_analyzer')
    def test_reanalyze_tweet_success(self, mock_create_analyzer):
        """Test successful tweet reanalysis."""
        mock_analyzer = Mock()
        mock_create_analyzer.return_value = mock_create_analyzer

        mock_tweet_data = {
            'tweet_id': '123',
            'username': 'testuser',
            'content': 'Test content',
            'media_links': 'url1.jpg,url2.jpg',
            'original_content': 'Quoted context'
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

        call_kwargs = mock_analyzer.analyze_content.call_args.kwargs
        expected_combined = "Test content\n\n[Contenido citado]: Quoted context"
        assert call_kwargs['content'] == expected_combined
        assert mock_analysis_result.post_content == 'Test content'

    @patch('analyzer.analyze_twitter.create_analyzer')
    def test_reanalyze_tweet_not_found(self, mock_create_analyzer):
        """Test reanalysis when tweet is not found."""
        mock_analyzer = Mock()
        mock_create_analyzer.return_value = mock_create_analyzer
        mock_analyzer.repository.get_tweet_data.return_value = None

        result = asyncio.run(reanalyze_tweet('123', analyzer=mock_analyzer))
        assert result is None

    @patch('scripts.analyzer_cli.analyze_tweets_cli')
    @patch('sys.argv', ['analyze_twitter.py', '--username', 'testuser', '--limit', '5'])
    def test_main_with_args(self, mock_analyze):
        """Test main function with command line arguments."""
        main()
        mock_analyze.assert_called_once_with(
            usernames=['testuser'],
            max_tweets=5,
            force_reanalyze=False,
            tweet_id=None,
            verbose=False,
            fast_mode=False
        )

    @patch('scripts.analyzer_cli.analyze_tweets_cli')
    @patch('sys.argv', ['analyze_twitter.py'])
    def test_main_no_args(self, mock_analyze):
        """Test main function with no arguments."""
        main()
        mock_analyze.assert_called_once_with(
            usernames=None,
            max_tweets=None,
            force_reanalyze=False,
            tweet_id=None,
            verbose=False,
            fast_mode=False
        )

    @patch('scripts.analyzer_cli.analyze_tweets_cli')
    @patch('sys.argv', ['analyze_twitter.py', '--force-reanalyze', '--tweet-id', '123'])
    def test_main_force_reanalyze_with_tweet_id(self, mock_analyze):
        """Test main function with force reanalyze and tweet ID."""
        main()
        mock_analyze.assert_called_once_with(
            usernames=None,
            max_tweets=None,
            force_reanalyze=True,
            tweet_id='123',
            verbose=False,
            fast_mode=False
        )