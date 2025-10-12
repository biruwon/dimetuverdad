"""
Integration tests for analyzer-retrieval system integration.
Tests end-to-end functionality with controlled test data.
"""

import asyncio
import pytest
import sqlite3
import tempfile
import os

from retrieval.api import create_retrieval_api
from retrieval.integration.analyzer_hooks import AnalyzerHooks, AnalysisResult, create_analyzer_hooks


class TestAnalyzerIntegration:
    """Test integration between analyzer and retrieval system with controlled data."""

    def setup_method(self):
        """Set up test database with controlled test data."""
        # Create temporary database
        self.db_fd, self.db_path = tempfile.mkstemp()
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._setup_test_database()

        # Create retrieval API
        self.api = create_analyzer_hooks()

    def teardown_method(self):
        """Clean up test database."""
        self.conn.close()
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def _setup_test_database(self):
        """Set up test database with sample disinformation content."""
        c = self.conn.cursor()

        # Create tables
        c.execute('''
            CREATE TABLE tweets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tweet_id TEXT UNIQUE NOT NULL,
                tweet_url TEXT NOT NULL,
                username TEXT NOT NULL,
                content TEXT NOT NULL,
                tweet_timestamp TEXT,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        c.execute('''
            CREATE TABLE content_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id TEXT NOT NULL,
                author_username TEXT,
                category TEXT,
                llm_explanation TEXT,
                analysis_method TEXT DEFAULT "pattern",
                analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(post_id)
            )
        ''')

        # Insert test data with disinformation content containing numerical claims
        test_tweets = [
            {
                'tweet_id': 'test_001',
                'tweet_url': 'https://twitter.com/test1/status/test_001',
                'username': 'FakeNewsUser',
                'content': 'Según fuentes secretas, la población española ha alcanzado los 100 millones de habitantes. ¡El gobierno lo oculta!',
                'tweet_timestamp': '2024-01-15T10:00:00Z',
                'category': 'disinformation',
                'llm_explanation': 'Contenido que afirma datos demográficos falsos sobre la población española, sugiriendo conspiración gubernamental.'
            },
            {
                'tweet_id': 'test_002',
                'tweet_url': 'https://twitter.com/test2/status/test_002',
                'username': 'ConspiracyBot',
                'content': '¡ALERTA! El PIB de España creció un 25% en 2023 según datos reales que el BCE no quiere que sepan. ¡Despierten!',
                'tweet_timestamp': '2024-01-16T14:30:00Z',
                'category': 'conspiracy_theory',
                'llm_explanation': 'Afirmación conspirativa sobre crecimiento económico exagerado, acusando a instituciones de ocultar información.'
            },
            {
                'tweet_id': 'test_003',
                'tweet_url': 'https://twitter.com/test3/status/test_003',
                'username': 'Agenda2030',
                'content': 'La tasa de desempleo en España es del 50% según informes confidenciales. El gobierno socialista lo niega todo.',
                'tweet_timestamp': '2024-01-17T09:15:00Z',
                'category': 'disinformation',
                'llm_explanation': 'Afirmación falsa sobre desempleo con acusaciones políticas, típica de narrativa conspirativa.'
            }
        ]

        for tweet in test_tweets:
            c.execute('''
                INSERT INTO tweets (tweet_id, tweet_url, username, content, tweet_timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (tweet['tweet_id'], tweet['tweet_url'], tweet['username'],
                  tweet['content'], tweet['tweet_timestamp']))

            c.execute('''
                INSERT INTO content_analyses (post_id, author_username, category, llm_explanation)
                VALUES (?, ?, ?, ?)
            ''', (tweet['tweet_id'], tweet['username'], tweet['category'], tweet['llm_explanation']))

        self.conn.commit()

    @pytest.mark.asyncio
    async def test_analyzer_verification_with_disinformation(self):
        """Test analyzer result verification with disinformation content."""
        # Query test data
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT t.tweet_id, t.content, t.username, ca.category, ca.llm_explanation
            FROM tweets t
            LEFT JOIN content_analyses ca ON t.tweet_id = ca.post_id
            WHERE ca.category IN ('disinformation', 'conspiracy_theory')
            ORDER BY t.tweet_timestamp
        ''')

        test_tweets = cursor.fetchall()
        assert len(test_tweets) == 3, "Should have 3 test tweets"

        for tweet in test_tweets:
            # Create mock analyzer result
            analyzer_result = {
                "category": tweet['category'],
                "confidence": 0.8,
                "explanation": tweet['llm_explanation']
            }

            # Test verification
            result = await self.api.analyze_with_verification(tweet['content'], analyzer_result)

            # Verify verification structure
            assert isinstance(result, AnalysisResult)
            assert result.original_result['category'] == tweet['category']
            assert result.original_result.get('explanation') == tweet['llm_explanation']
            assert isinstance(result.verification_data.get('verification_confidence', 0), (int, float))
            assert isinstance(result.verification_data.get('sources_cited', []), list)
            assert isinstance(result.verification_data.get('contradictions_detected', []), list)
            assert len(result.explanation_with_verification) > 0

            # Should have some form of verification
            assert result.explanation_with_verification != result.original_result.get('explanation', '')

    @pytest.mark.asyncio
    async def test_disinformation_triggering(self):
        """Test that disinformation content triggers verification."""
        content = "Según fuentes secretas, hay 200 millones de españoles viviendo en España."
        analyzer_result = {
            "category": "disinformation",
            "confidence": 0.9,
            "explanation": "Contenido con afirmaciones demográficas falsas"
        }

        result = await self.api.analyze_with_verification(content, analyzer_result)

        # Should trigger verification for disinformation
        assert result.verification_data['verification_confidence'] >= 0
        assert "disinformation" in result.explanation_with_verification.lower() or "falsas" in result.explanation_with_verification.lower()

    @pytest.mark.asyncio
    async def test_conspiracy_theory_triggering(self):
        """Test that conspiracy theory content triggers verification."""
        content = "El gobierno oculta que el PIB creció un 50% gracias a la tecnología secreta 5G."
        analyzer_result = {
            "category": "conspiracy_theory",
            "confidence": 0.85,
            "explanation": "Contenido conspirativo sobre economía y tecnología"
        }

        result = await self.api.analyze_with_verification(content, analyzer_result)

        # Should trigger verification for conspiracy theory
        assert result.verification_data['verification_confidence'] >= 0
        assert len(result.explanation_with_verification) > len(result.original_result.get('explanation', ''))