"""
Web Layer Testing Configuration
Comprehensive testing for Flask application routes, templates, and functionality.
"""

import pytest
import tempfile
import os
import glob
import uuid
import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from web.app import create_app
from utils.database import _create_test_database_schema


@pytest.fixture(scope="session", autouse=True)
def cleanup_web_test_databases():
    """Clean up web test databases at the end of the test session."""
    yield  # Run tests first

    # Clean up Flask test databases in temp directory
    temp_dir = tempfile.gettempdir()
    flask_test_pattern = os.path.join(temp_dir, 'flask_session_test_*.db')

    for db_file in glob.glob(flask_test_pattern):
        try:
            os.remove(db_file)
            print(f"🗑️  Cleaned up Flask test database: {os.path.basename(db_file)}")
        except OSError as e:
            print(f"⚠️  Could not remove Flask test database {db_file}: {e}")


@pytest.fixture(scope="session")
def session_test_db_path():
    """Create a single test database for the entire web test session."""

    # Create unique test database per session
    session_id = str(uuid.uuid4())[:8]
    test_db_path = os.path.join(tempfile.gettempdir(), f'flask_session_test_{session_id}.db')

    # Remove any existing test database
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    # Create fresh schema using the same initialization function
    _create_test_database_schema(test_db_path)

    yield test_db_path

    # Clean up after session
    try:
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
    except OSError:
        pass


@pytest.fixture
def app(session_test_db_path):
    """Create and configure a test app instance."""

    # Configure test app with session database
    test_config = {
        'TESTING': True,
        'DATABASE_PATH': session_test_db_path,
        'SECRET_KEY': 'test-secret-key',
        'ADMIN_TOKEN': 'test-admin-token',
        'CACHE_TYPE': 'flask_caching.backends.SimpleCache',
        'CACHE_DEFAULT_TIMEOUT': 300,
        'DB_TIMEOUT': 30.0,
        'DB_CHECK_SAME_THREAD': False
    }

    # Set DATABASE_PATH environment variable to ensure it takes precedence
    # over any global DATABASE_PATH set by other test modules
    old_db_path = os.environ.get('DATABASE_PATH')
    os.environ['DATABASE_PATH'] = session_test_db_path

    flask_app = create_app()
    flask_app.config.update(test_config)

    yield flask_app

    # Restore original DATABASE_PATH if it existed
    if old_db_path is not None:
        os.environ['DATABASE_PATH'] = old_db_path
    elif 'DATABASE_PATH' in os.environ:
        del os.environ['DATABASE_PATH']
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
                         content: str = None, url: str = None, post_ts: str = None, categories_detected=None,
                         local_explanation: str = '', external_explanation: str = '', analysis_stages: str = 'pattern') -> MockRow:
    """Return a MockRow shaped for export endpoints (CSV/JSON)."""
    return MockRow({
        'post_id': post_id,
        'author_username': author,
        'category': category,
        'local_explanation': local_explanation or explanation,
        'external_explanation': external_explanation,
        'analysis_stages': analysis_stages,
        'analysis_timestamp': timestamp,
        'post_content': content or '',
        'post_url': url or '',
        'post_timestamp': post_ts or timestamp,
        'categories_detected': categories_detected,
        'external_analysis_used': bool(external_explanation)
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


@pytest.fixture(autouse=True)
def cleanup_session_db_tables(session_test_db_path):
    """Clean up database tables between tests to ensure test isolation."""
    # Truncate all tables to ensure clean state between tests
    try:
        conn = sqlite3.connect(session_test_db_path, timeout=5.0)
        conn.execute("PRAGMA busy_timeout = 5000")  # 5 second timeout
    except sqlite3.Error:
        # If we can't connect to the database (e.g., it's mocked or locked), skip cleanup
        return

    try:
        cursor = conn.cursor()
        # Disable foreign key checks temporarily
        cursor.execute("PRAGMA foreign_keys = OFF")

        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = cursor.fetchall()

        # Truncate all tables
        for table in tables:
            table_name = table[0]
            try:
                cursor.execute(f"DELETE FROM {table_name}")
            except sqlite3.Error:
                # If table deletion fails (e.g., due to mocking), continue
                continue

        # Reset auto-increment counters only if sqlite_sequence table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'")
        if cursor.fetchone():
            try:
                cursor.execute("DELETE FROM sqlite_sequence")
            except sqlite3.Error:
                pass  # Ignore if this fails

        # Re-enable foreign key checks
        cursor.execute("PRAGMA foreign_keys = ON")
        conn.commit()
    except sqlite3.Error:
        # If any database operation fails, skip cleanup entirely
        pass
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass


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
                    post_id, category, local_explanation, external_explanation, 
                    analysis_stages, external_analysis_used,
                    analysis_timestamp, author_username
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tweet_data['tweet_id'],
                tweet_data['category'],
                tweet_data.get('local_explanation', tweet_data.get('llm_explanation', '')),
                tweet_data.get('external_explanation', ''),
                tweet_data.get('analysis_stages', 'pattern'),
                tweet_data.get('external_analysis_used', False),
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