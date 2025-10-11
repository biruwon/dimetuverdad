"""
Web Layer Testing Configuration
Comprehensive testing for Flask application routes, templates, and functionality.
"""

import pytest
import tempfile
import os
import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from web.app import create_app
from config import ADMIN_TOKEN

def init_test_db(db_path: str):
    """Initialize test database with minimal schema."""
    # Remove existing test database
    if os.path.exists(db_path):
        os.remove(db_path)

    # Create fresh database
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    try:
        # Create minimal tables for testing
        c.execute('''
            CREATE TABLE accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                profile_pic_url TEXT,
                last_scraped TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        c.execute('''
            CREATE TABLE tweets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tweet_id TEXT UNIQUE NOT NULL,
                tweet_url TEXT NOT NULL,
                username TEXT NOT NULL,
                content TEXT NOT NULL,
                post_type TEXT DEFAULT 'original',
                media_links TEXT,
                media_count INTEGER DEFAULT 0,
                hashtags TEXT,
                mentions TEXT,
                tweet_timestamp TEXT,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (username) REFERENCES accounts (username)
            )
        ''')

        c.execute('''
            CREATE TABLE content_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tweet_id TEXT NOT NULL,
                tweet_url TEXT,
                username TEXT,
                tweet_content TEXT,
                category TEXT,
                categories_detected TEXT,
                llm_explanation TEXT,
                analysis_method TEXT DEFAULT "pattern",
                analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                analysis_json TEXT,
                pattern_matches TEXT,
                topic_classification TEXT,
                media_urls TEXT,
                media_analysis TEXT,
                media_type TEXT,
                multimodal_analysis BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (tweet_id) REFERENCES tweets (tweet_id),
                FOREIGN KEY (username) REFERENCES accounts (username),
                UNIQUE(tweet_id)
            )
        ''')

        # Create indexes
        c.execute('CREATE INDEX idx_tweets_username ON tweets(username)')
        c.execute('CREATE INDEX idx_tweets_timestamp ON tweets(scraped_at)')
        c.execute('CREATE INDEX idx_analyses_tweet ON content_analyses(tweet_id)')
        c.execute('CREATE INDEX idx_analyses_category ON content_analyses(category)')
        c.execute('CREATE INDEX idx_analyses_username ON content_analyses(username)')

        conn.commit()

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


@pytest.fixture
def app():
    """Create and configure a test app instance."""
    # Create temporary database for testing
    db_fd, db_path = tempfile.mkstemp()

    # Configure test app
    test_config = {
        'TESTING': True,
        'DATABASE_PATH': db_path,
        'SECRET_KEY': 'test-secret-key',
        'ADMIN_TOKEN': 'test-admin-token',
        'CACHE_TYPE': 'SimpleCache',
        'CACHE_DEFAULT_TIMEOUT': 300,
        'DB_TIMEOUT': 30.0,
        'DB_CHECK_SAME_THREAD': False
    }

    flask_app = create_app()
    flask_app.config.update(test_config)

    # Initialize test database
    with flask_app.app_context():
        init_test_db(db_path)

    yield flask_app

    # Cleanup
    os.close(db_fd)
    os.unlink(db_path)


@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()


@pytest.fixture
def runner(app):
    """A test runner for the app's Click commands."""
    return app.test_cli_runner()


@pytest.fixture
def admin_client(client):
    """A test client with admin authentication."""
    with client:
        with client.session_transaction() as sess:
            sess['admin_authenticated'] = True
        return client


@pytest.fixture
def mock_analyzer():
    """Mock analyzer for testing."""
    with patch('web.app.get_analyzer') as mock_get_analyzer:
        mock_analyzer = Mock()
        mock_analyzer.analyze_content.return_value = Mock(
            category='general',
            explanation='Test analysis result'
        )
        mock_get_analyzer.return_value = mock_analyzer
        yield mock_analyzer


class MockRow:
    """Mock sqlite3.Row object for testing."""
    def __init__(self, data_dict):
        self._data = data_dict
        self._keys = list(data_dict.keys())
        self._values = list(data_dict.values())

    def __getitem__(self, key):
        if isinstance(key, int):
            # Integer indexing like row[0]
            return self._values[key]
        else:
            # String indexing like row['column_name']
            return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def keys(self):
        """Return list of column names."""
        return self._keys

    def __iter__(self):
        """Iterate over (key, value) pairs."""
        return iter(self._data.items())

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"MockRow({self._data})"


@pytest.fixture
def mock_database():
    """Mock database operations."""
    with patch('web.app.get_db_connection') as mock_conn:
        mock_cursor = Mock()
        mock_connection = Mock()

        # Set up cursor methods to return proper data structures
        # Use MockRow for fetchone to support dict() conversion and indexing
        mock_cursor.fetchone.return_value = MockRow({'total_accounts': 10, 'analyzed_tweets': 100})
        mock_cursor.fetchall.return_value = []  # Default empty list
        mock_cursor.rowcount = 0
        mock_cursor.total_changes = 0

        # Set up connection methods
        mock_connection.cursor.return_value = mock_cursor
        mock_connection.execute.return_value = mock_cursor
        mock_connection.commit.return_value = None
        mock_connection.close.return_value = None

        mock_conn.return_value.__enter__ = Mock(return_value=mock_connection)
        mock_conn.return_value.__exit__ = Mock(return_value=None)
        mock_conn.return_value = mock_connection

        yield mock_connection


@pytest.fixture
def sample_tweet_data():
    """Sample tweet data for testing."""
    return {
        'tweet_id': '1234567890',
        'content': 'This is a test tweet content',
        'username': 'testuser',
        'tweet_timestamp': '2024-01-01 12:00:00',
        'tweet_url': 'https://twitter.com/testuser/status/1234567890',
        'category': 'general',
        'llm_explanation': 'Test explanation',
        'analysis_method': 'pattern'
    }


@pytest.fixture
def sample_account_data():
    """Sample account data for testing."""
    return {
        'username': 'testuser',
        'tweet_count': 100,
        'problematic_posts': 5,
        'analyzed_posts': 95,
        'profile_pic_url': 'https://example.com/avatar.jpg',
        'last_activity': '2024-01-01 12:00:00'
    }


class TestHelpers:
    """Helper methods for web layer tests."""

    @staticmethod
    def create_test_tweet(db_connection, tweet_data):
        """Create a test tweet in the database."""
        db_connection.execute("""
            INSERT INTO tweets (
                tweet_id, content, username, tweet_timestamp, tweet_url
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            tweet_data['tweet_id'],
            tweet_data['content'],
            tweet_data['username'],
            tweet_data['tweet_timestamp'],
            tweet_data['tweet_url']
        ))

        if 'category' in tweet_data:
            db_connection.execute("""
                INSERT INTO content_analyses (
                    tweet_id, category, llm_explanation, analysis_method,
                    analysis_timestamp, username
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                tweet_data['tweet_id'],
                tweet_data['category'],
                tweet_data.get('llm_explanation', ''),
                tweet_data.get('analysis_method', 'pattern'),
                tweet_data.get('analysis_timestamp', '2024-01-01 12:00:00'),
                tweet_data['username']
            ))

        db_connection.commit()

    @staticmethod
    def create_test_account(db_connection, account_data):
        """Create a test account in the database."""
        db_connection.execute("""
            INSERT INTO accounts (
                username, profile_pic_url, last_updated
            ) VALUES (?, ?, ?)
        """, (
            account_data['username'],
            account_data.get('profile_pic_url', ''),
            account_data.get('last_activity', '2024-01-01 12:00:00')
        ))
        db_connection.commit()