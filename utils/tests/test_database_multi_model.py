"""
Unit tests for database_multi_model module.
Tests all database operations for multi-model analysis storage and retrieval.
"""

import pytest
import sqlite3
from datetime import datetime
from unittest.mock import Mock, patch
from utils import database_multi_model
from utils.database import get_db_connection_context


@pytest.fixture
def test_db_connection():
    """Create a temporary in-memory database for testing."""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    
    # Create necessary tables
    conn.execute('''
        CREATE TABLE IF NOT EXISTS model_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            category TEXT NOT NULL,
            explanation TEXT,
            confidence_score REAL,
            processing_time_seconds REAL,
            error_message TEXT,
            analysis_timestamp TEXT NOT NULL,
            UNIQUE(post_id, model_name)
        )
    ''')
    
    conn.execute('''
        CREATE TABLE IF NOT EXISTS content_analyses (
            post_id TEXT PRIMARY KEY,
            category TEXT,
            multi_model_analysis INTEGER DEFAULT 0,
            model_consensus_category TEXT
        )
    ''')
    
    conn.execute('''
        CREATE TABLE IF NOT EXISTS tweets (
            tweet_id TEXT PRIMARY KEY,
            content TEXT,
            username TEXT,
            media_links TEXT,
            tweet_timestamp TEXT
        )
    ''')
    
    conn.commit()
    return conn


class TestSaveModelAnalysis:
    """Test save_model_analysis function."""
    
    def test_save_model_analysis_success(self, test_db_connection):
        """Test saving a model analysis successfully."""
        database_multi_model.save_model_analysis(
            conn=test_db_connection,
            post_id="123",
            model_name="gemma3:4b",
            category="general",
            explanation="Test explanation",
            processing_time=1.5
        )
        
        cursor = test_db_connection.cursor()
        cursor.execute("SELECT * FROM model_analyses WHERE post_id = ?", ("123",))
        row = cursor.fetchone()
        
        assert row is not None
        assert row['model_name'] == "gemma3:4b"
        assert row['category'] == "general"
        assert row['explanation'] == "Test explanation"
        assert row['processing_time_seconds'] == 1.5
    
    def test_save_model_analysis_with_confidence(self, test_db_connection):
        """Test saving analysis with confidence score."""
        database_multi_model.save_model_analysis(
            conn=test_db_connection,
            post_id="123",
            model_name="gpt-oss:20b",
            category="hate_speech",
            explanation="Detected hate speech",
            processing_time=2.0,
            confidence_score=0.85
        )
        
        cursor = test_db_connection.cursor()
        cursor.execute("SELECT * FROM model_analyses WHERE post_id = ?", ("123",))
        row = cursor.fetchone()
        
        assert row['confidence_score'] == 0.85
    
    def test_save_model_analysis_with_error(self, test_db_connection):
        """Test saving failed analysis with error message."""
        database_multi_model.save_model_analysis(
            conn=test_db_connection,
            post_id="123",
            model_name="gemma3:27b-it-qat",
            category="general",
            explanation="",
            processing_time=0.0,
            error_message="Model timeout"
        )
        
        cursor = test_db_connection.cursor()
        cursor.execute("SELECT * FROM model_analyses WHERE post_id = ?", ("123",))
        row = cursor.fetchone()
        
        assert row['error_message'] == "Model timeout"
    
    def test_save_model_analysis_replaces_existing(self, test_db_connection):
        """Test that saving replaces existing analysis for same post/model."""
        # Save first analysis
        database_multi_model.save_model_analysis(
            conn=test_db_connection,
            post_id="123",
            model_name="gemma3:4b",
            category="general",
            explanation="First explanation",
            processing_time=1.0
        )
        
        # Save second analysis (should replace)
        database_multi_model.save_model_analysis(
            conn=test_db_connection,
            post_id="123",
            model_name="gemma3:4b",
            category="hate_speech",
            explanation="Second explanation",
            processing_time=2.0
        )
        
        cursor = test_db_connection.cursor()
        cursor.execute("SELECT * FROM model_analyses WHERE post_id = ? AND model_name = ?", 
                      ("123", "gemma3:4b"))
        rows = cursor.fetchall()
        
        # Should only have one row (replaced)
        assert len(rows) == 1
        assert rows[0]['category'] == "hate_speech"
        assert rows[0]['explanation'] == "Second explanation"


class TestGetModelAnalyses:
    """Test get_model_analyses function."""
    
    def test_get_model_analyses_success(self, test_db_connection):
        """Test retrieving model analyses for a post."""
        # Save multiple analyses
        database_multi_model.save_model_analysis(
            test_db_connection, "123", "gemma3:4b", "general", "Exp1", 1.0
        )
        database_multi_model.save_model_analysis(
            test_db_connection, "123", "gpt-oss:20b", "hate_speech", "Exp2", 2.0
        )
        
        results = database_multi_model.get_model_analyses(test_db_connection, "123")
        
        assert len(results) == 2
        assert any(r['model_name'] == "gemma3:4b" for r in results)
        assert any(r['model_name'] == "gpt-oss:20b" for r in results)
    
    def test_get_model_analyses_empty(self, test_db_connection):
        """Test retrieving analyses for non-existent post."""
        results = database_multi_model.get_model_analyses(test_db_connection, "999")
        
        assert results == []
    
    def test_get_model_analyses_includes_all_fields(self, test_db_connection):
        """Test that all fields are returned."""
        database_multi_model.save_model_analysis(
            test_db_connection, "123", "gemma3:4b", "general", "Exp", 1.5,
            confidence_score=0.9, error_message=None
        )
        
        results = database_multi_model.get_model_analyses(test_db_connection, "123")
        
        assert len(results) == 1
        result = results[0]
        assert 'model_name' in result
        assert 'category' in result
        assert 'explanation' in result
        assert 'confidence_score' in result
        assert 'processing_time' in result
        assert 'timestamp' in result
        assert 'error' in result


class TestGetModelConsensus:
    """Test get_model_consensus function."""
    
    def test_get_model_consensus_full_agreement(self, test_db_connection):
        """Test consensus with full agreement."""
        # All models agree
        database_multi_model.save_model_analysis(
            test_db_connection, "123", "gemma3:4b", "hate_speech", "Exp1", 1.0
        )
        database_multi_model.save_model_analysis(
            test_db_connection, "123", "gpt-oss:20b", "hate_speech", "Exp2", 2.0
        )
        
        consensus = database_multi_model.get_model_consensus(test_db_connection, "123")
        
        assert consensus is not None
        assert consensus['category'] == "hate_speech"
        assert consensus['agreement_score'] == 1.0
        assert consensus['total_models'] == 2
    
    def test_get_model_consensus_majority_vote(self, test_db_connection):
        """Test consensus with majority vote."""
        # 2 vote general, 1 votes hate_speech
        database_multi_model.save_model_analysis(
            test_db_connection, "123", "gemma3:4b", "general", "Exp1", 1.0
        )
        database_multi_model.save_model_analysis(
            test_db_connection, "123", "gemma3:12b", "general", "Exp2", 1.5
        )
        database_multi_model.save_model_analysis(
            test_db_connection, "123", "gpt-oss:20b", "hate_speech", "Exp3", 2.0
        )
        
        consensus = database_multi_model.get_model_consensus(test_db_connection, "123")
        
        assert consensus['category'] == "general"
        assert consensus['agreement_score'] == pytest.approx(2.0 / 3.0)
        assert consensus['total_models'] == 3
    
    def test_get_model_consensus_no_analyses(self, test_db_connection):
        """Test consensus with no analyses."""
        consensus = database_multi_model.get_model_consensus(test_db_connection, "999")
        
        assert consensus is None
    
    def test_get_model_consensus_excludes_errors(self, test_db_connection):
        """Test that failed analyses are excluded from consensus."""
        database_multi_model.save_model_analysis(
            test_db_connection, "123", "gemma3:4b", "general", "Exp1", 1.0
        )
        database_multi_model.save_model_analysis(
            test_db_connection, "123", "gpt-oss:20b", "hate_speech", "", 0.0,
            error_message="Failed"
        )
        
        consensus = database_multi_model.get_model_consensus(test_db_connection, "123")
        
        # Should only count successful analysis
        assert consensus['total_models'] == 1
        assert consensus['category'] == "general"


class TestUpdateConsensusInContentAnalyses:
    """Test update_consensus_in_content_analyses function."""
    
    def test_update_consensus_success(self, test_db_connection):
        """Test updating content_analyses with consensus."""
        # Insert content analysis
        test_db_connection.execute(
            "INSERT INTO content_analyses (post_id, category) VALUES (?, ?)",
            ("123", "general")
        )
        
        # Save model analyses
        database_multi_model.save_model_analysis(
            test_db_connection, "123", "gemma3:4b", "hate_speech", "Exp", 1.0
        )
        
        # Update consensus
        result = database_multi_model.update_consensus_in_content_analyses(
            test_db_connection, "123"
        )
        
        assert result is True
        
        # Verify update
        cursor = test_db_connection.cursor()
        cursor.execute("SELECT * FROM content_analyses WHERE post_id = ?", ("123",))
        row = cursor.fetchone()
        
        assert row['multi_model_analysis'] == 1
        assert row['model_consensus_category'] == "hate_speech"
    
    def test_update_consensus_no_analyses(self, test_db_connection):
        """Test updating when no model analyses exist."""
        result = database_multi_model.update_consensus_in_content_analyses(
            test_db_connection, "999"
        )
        
        assert result is False


class TestGetModelPerformanceStats:
    """Test get_model_performance_stats function."""
    
    def test_get_performance_stats_all_models(self, test_db_connection):
        """Test getting stats for all models."""
        # Add analyses for multiple models
        database_multi_model.save_model_analysis(
            test_db_connection, "1", "gemma3:4b", "general", "Exp", 1.0
        )
        database_multi_model.save_model_analysis(
            test_db_connection, "2", "gemma3:4b", "hate_speech", "Exp", 1.5
        )
        database_multi_model.save_model_analysis(
            test_db_connection, "3", "gpt-oss:20b", "general", "Exp", 2.0
        )
        
        stats = database_multi_model.get_model_performance_stats(test_db_connection)
        
        assert "gemma3:4b" in stats
        assert "gpt-oss:20b" in stats
        assert stats["gemma3:4b"]["total_analyses"] == 2
        assert stats["gpt-oss:20b"]["total_analyses"] == 1
    
    def test_get_performance_stats_specific_model(self, test_db_connection):
        """Test getting stats for specific model."""
        database_multi_model.save_model_analysis(
            test_db_connection, "1", "gemma3:4b", "general", "Exp", 1.0
        )
        
        stats = database_multi_model.get_model_performance_stats(
            test_db_connection, model_name="gemma3:4b"
        )
        
        assert len(stats) == 1
        assert "gemma3:4b" in stats
    
    def test_get_performance_stats_includes_error_rate(self, test_db_connection):
        """Test that error rate is calculated correctly."""
        # 2 successful, 1 failed
        database_multi_model.save_model_analysis(
            test_db_connection, "1", "gemma3:4b", "general", "Exp", 1.0
        )
        database_multi_model.save_model_analysis(
            test_db_connection, "2", "gemma3:4b", "hate_speech", "Exp", 1.5
        )
        database_multi_model.save_model_analysis(
            test_db_connection, "3", "gemma3:4b", "general", "", 0.0,
            error_message="Failed"
        )
        
        stats = database_multi_model.get_model_performance_stats(test_db_connection)
        
        gemma_stats = stats["gemma3:4b"]
        assert gemma_stats["successful"] == 2
        assert gemma_stats["failed"] == 1
        assert gemma_stats["error_rate"] == pytest.approx(1.0 / 3.0)


class TestGetModelAgreementStats:
    """Test get_model_agreement_stats function."""
    
    def test_get_agreement_stats_full_agreement(self, test_db_connection):
        """Test agreement stats with full model agreement."""
        # Post 1: Full agreement
        database_multi_model.save_model_analysis(
            test_db_connection, "1", "gemma3:4b", "general", "Exp", 1.0
        )
        database_multi_model.save_model_analysis(
            test_db_connection, "1", "gpt-oss:20b", "general", "Exp", 2.0
        )
        
        stats = database_multi_model.get_model_agreement_stats(test_db_connection)
        
        assert stats['total_posts_analyzed'] == 1
        assert stats['full_agreement_count'] == 1
        assert stats['avg_agreement_score'] == 1.0
    
    def test_get_agreement_stats_no_agreement(self, test_db_connection):
        """Test agreement stats with partial agreement (50% threshold)."""
        # Post 1: 50% agreement (one model each)
        database_multi_model.save_model_analysis(
            test_db_connection, "1", "gemma3:4b", "general", "Exp", 1.0
        )
        database_multi_model.save_model_analysis(
            test_db_connection, "1", "gpt-oss:20b", "hate_speech", "Exp", 2.0
        )
        
        stats = database_multi_model.get_model_agreement_stats(test_db_connection)
        
        assert stats['total_posts_analyzed'] == 1
        # With 50% agreement (0.5), it counts as partial agreement not no agreement
        assert stats['partial_agreement_count'] == 1
    
    def test_get_agreement_stats_empty_database(self, test_db_connection):
        """Test agreement stats with no data."""
        stats = database_multi_model.get_model_agreement_stats(test_db_connection)
        
        assert stats['total_posts_analyzed'] == 0
        assert stats['full_agreement_count'] == 0
        assert stats['avg_agreement_score'] == 0.0


class TestGetPostsForMultiModelAnalysis:
    """Test get_posts_for_multi_model_analysis function."""
    
    def test_get_posts_basic(self, test_db_connection):
        """Test getting posts for analysis."""
        test_db_connection.execute(
            "INSERT INTO tweets (tweet_id, content, username, media_links, tweet_timestamp) VALUES (?, ?, ?, ?, ?)",
            ("123", "Test content", "user1", "url1,url2", "2024-01-01")
        )
        
        posts = database_multi_model.get_posts_for_multi_model_analysis(test_db_connection)
        
        assert len(posts) == 1
        assert posts[0]['post_id'] == "123"
        assert posts[0]['content'] == "Test content"
        assert posts[0]['media_urls'] == ["url1", "url2"]
    
    def test_get_posts_with_limit(self, test_db_connection):
        """Test getting posts with limit."""
        for i in range(5):
            test_db_connection.execute(
                "INSERT INTO tweets (tweet_id, content, username, tweet_timestamp) VALUES (?, ?, ?, ?)",
                (str(i), f"Content {i}", "user1", "2024-01-01")
            )
        
        posts = database_multi_model.get_posts_for_multi_model_analysis(
            test_db_connection, limit=2
        )
        
        assert len(posts) == 2
    
    def test_get_posts_filter_by_username(self, test_db_connection):
        """Test filtering posts by username."""
        test_db_connection.execute(
            "INSERT INTO tweets (tweet_id, content, username, tweet_timestamp) VALUES (?, ?, ?, ?)",
            ("1", "Content 1", "user1", "2024-01-01")
        )
        test_db_connection.execute(
            "INSERT INTO tweets (tweet_id, content, username, tweet_timestamp) VALUES (?, ?, ?, ?)",
            ("2", "Content 2", "user2", "2024-01-01")
        )
        
        posts = database_multi_model.get_posts_for_multi_model_analysis(
            test_db_connection, username="user1"
        )
        
        assert len(posts) == 1
        assert posts[0]['post_id'] == "1"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
