"""
Web Layer Testing Configuration
Comprehensive testing for Flask application routes, templates, and functionality.
"""

import pytest
import tempfile
import os
import atexit
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from web.app import create_app
from config import ADMIN_TOKEN
from utils.database import init_test_database, cleanup_test_databases


@pytest.fixture(scope="session", autouse=True)
def cleanup_web_test_databases():
    """Clean up web test databases at the end of the test session."""
    yield  # Run tests first

    # Clean up Flask test databases in temp directory
    temp_dir = tempfile.gettempdir()
    flask_test_pattern = os.path.join(temp_dir, 'flask_test_*.db')

    import glob
    for db_file in glob.glob(flask_test_pattern):
        try:
            os.remove(db_file)
            print(f"🗑️  Cleaned up Flask test database: {os.path.basename(db_file)}")
        except OSError as e:
            print(f"⚠️  Could not remove Flask test database {db_file}: {e}")


@pytest.fixture
def app():
    """Create and configure a test app instance."""

    # For Flask tests, use a deterministic test database path that's consistent across workers
    import tempfile
    worker_id = os.environ.get('PYTEST_XDIST_WORKER', 'master')
    test_db_path = os.path.join(tempfile.gettempdir(), f'flask_test_{worker_id}.db')

    # Remove any existing test database for this worker
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    # Create fresh schema using the same initialization function
    from utils.database import _create_test_database_schema
    _create_test_database_schema(test_db_path)

    # Configure test app
    test_config = {
        'TESTING': True,
        'DATABASE_PATH': test_db_path,
        'SECRET_KEY': 'test-secret-key',
        'ADMIN_TOKEN': 'test-admin-token',
        'CACHE_TYPE': 'SimpleCache',
        'CACHE_DEFAULT_TIMEOUT': 300,
        'DB_TIMEOUT': 30.0,
        'DB_CHECK_SAME_THREAD': False
    }

    flask_app = create_app()
    flask_app.config.update(test_config)

    yield flask_app
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


def make_count_row(count: int = 0, key: str = 'cnt') -> MockRow:
    """Return a MockRow representing a simple COUNT(*) AS cnt result."""
    return MockRow({key: count})


def make_post_export_row(post_id: str, author: str, category: str = 'general', explanation: str = '',
                         method: str = 'pattern', timestamp: str = '2024-01-01 12:00:00',
                         content: str = None, url: str = None, post_ts: str = None, categories_detected=None) -> MockRow:
    """Return a MockRow shaped for export endpoints (CSV/JSON)."""
    return MockRow({
        'post_id': post_id,
        'author_username': author,
        'category': category,
        'llm_explanation': explanation,
        'analysis_method': method,
        'analysis_timestamp': timestamp,
        'post_content': content or '',
        'post_url': url or '',
        'post_timestamp': post_ts or timestamp,
        'categories_detected': categories_detected
    })


def tuple_to_mockrow(seq: tuple, fields: list) -> MockRow:
    """Convert a sequence/tuple into a MockRow given a list of field names."""
    return MockRow({k: (seq[i] if i < len(seq) else None) for i, k in enumerate(fields)})


@pytest.fixture
def mock_database():
    """Mock database operations."""
    with patch('utils.database.get_db_connection') as mock_conn:
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
                    post_id, category, llm_explanation, analysis_method,
                    analysis_timestamp, author_username
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
                username, profile_pic_url, last_scraped
            ) VALUES (?, ?, ?)
        """, (
            account_data['username'],
            account_data.get('profile_pic_url', ''),
            account_data.get('last_activity', '2024-01-01 12:00:00')
        ))
        db_connection.commit()