"""
Shared pytest fixtures for analyzer tests.
Provides common test data setup and database management.
"""

import pytest
import sqlite3
import os
from typing import Dict, Any
from analyzer.models import ContentAnalysis
from analyzer.categories import Categories
from utils.database import init_test_database, cleanup_test_database, _create_test_database_schema
from utils import paths


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_databases():
    """Clean up test databases at the end of the test session."""
    yield  # Run tests first

    # Clean up test database after tests complete
    import os
    test_db_path = paths.get_db_path(env='testing')
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
            print(f"🗑️  Cleaned up test database: {os.path.basename(test_db_path)}")
        except OSError as e:
            print(f"⚠️  Could not remove test database {test_db_path}: {e}")


@pytest.fixture(scope="session")
def session_db_path(tmp_path_factory):
    """Provide a unique database path for the entire test session."""
    # Create a unique database path for this test session
    db_dir = tmp_path_factory.getbasetemp() / "test_dbs"
    db_dir.mkdir(exist_ok=True)
    db_path = db_dir / "test_accounts.db"
    return str(db_path)


@pytest.fixture
def test_db_path(session_db_path):
    """Provide the session database path for each test."""
    return session_db_path


@pytest.fixture(autouse=True)
def setup_test_database(test_db_path):
    """Set up the test database environment for each test."""
    # Set the DATABASE_PATH environment variable to point to our test database
    old_db_path = os.environ.get('DATABASE_PATH')
    os.environ['DATABASE_PATH'] = test_db_path

    # Ensure database exists and has schema
    if not os.path.exists(test_db_path):
        _create_test_database_schema(test_db_path)

    yield

    # Restore original DATABASE_PATH if it existed
    if old_db_path is not None:
        os.environ['DATABASE_PATH'] = old_db_path
    elif 'DATABASE_PATH' in os.environ:
        del os.environ['DATABASE_PATH']


@pytest.fixture
def test_db(test_db_path):
    """Provide a test database connection for each test."""
    conn = sqlite3.connect(test_db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    yield conn

    # Clean up any changes made during the test by truncating tables
    # Use a separate connection for cleanup to avoid issues
    try:
        cleanup_conn = sqlite3.connect(test_db_path)
        cleanup_conn.execute("DELETE FROM content_analyses")
        cleanup_conn.execute("DELETE FROM tweets")
        cleanup_conn.execute("DELETE FROM accounts")
        cleanup_conn.commit()
        cleanup_conn.close()
    except sqlite3.OperationalError:
        # Tables might not exist or be locked, skip cleanup
        pass
    finally:
        conn.close()


@pytest.fixture
def test_account(test_db):
    """Create a test account record."""
    account_data = {
        'username': 'test_user',
        'platform': 'twitter'
    }

    test_db.execute('''
        INSERT OR REPLACE INTO accounts (username, platform)
        VALUES (?, ?)
    ''', (account_data['username'], account_data['platform']))

    test_db.commit()
    return account_data


@pytest.fixture
def test_tweet(test_db, test_account):
    """Create a test tweet record."""
    tweet_data = {
        'tweet_id': 'test_123',
        'tweet_url': f"https://twitter.com/{test_account['username']}/status/test_123",
        'username': test_account['username'],
        'content': 'Test content',
        'tweet_timestamp': '2024-01-01T12:00:00'
    }

    test_db.execute('''
        INSERT OR REPLACE INTO tweets
        (tweet_id, tweet_url, username, content, tweet_timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        tweet_data['tweet_id'],
        tweet_data['tweet_url'],
        tweet_data['username'],
        tweet_data['content'],
        tweet_data['tweet_timestamp']
    ))

    test_db.commit()
    return tweet_data


@pytest.fixture
def test_tweet_2(test_db, test_account):
    """Create a second test tweet record."""
    tweet_data = {
        'tweet_id': 'failed_456',
        'tweet_url': f"https://twitter.com/{test_account['username']}/status/failed_456",
        'username': test_account['username'],
        'content': 'Failed content',
        'tweet_timestamp': '2024-01-01T12:00:00'
    }

    test_db.execute('''
        INSERT OR REPLACE INTO tweets
        (tweet_id, tweet_url, username, content, tweet_timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        tweet_data['tweet_id'],
        tweet_data['tweet_url'],
        tweet_data['username'],
        tweet_data['content'],
        tweet_data['tweet_timestamp']
    ))

    test_db.commit()
    return tweet_data


@pytest.fixture
def test_multiple_tweets(test_db, test_account):
    """Create multiple test tweets for batch testing."""
    tweets_data = []
    for i in range(3):
        tweet_data = {
            'tweet_id': f'test_{i}',
            'tweet_url': f"https://twitter.com/{test_account['username']}/status/test_{i}",
            'username': test_account['username'],
            'content': f'Test content {i}',
            'tweet_timestamp': f'2024-01-01T12:0{i}:00'
        }

        test_db.execute('''
            INSERT OR REPLACE INTO tweets
            (tweet_id, tweet_url, username, content, tweet_timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            tweet_data['tweet_id'],
            tweet_data['tweet_url'],
            tweet_data['username'],
            tweet_data['content'],
            tweet_data['tweet_timestamp']
        ))

        tweets_data.append(tweet_data)

    test_db.commit()
    return tweets_data


@pytest.fixture
def test_different_accounts(test_db):
    """Create multiple test accounts with different usernames."""
    accounts_data = []
    usernames = ['user1', 'user2', 'test_user']

    for username in usernames:
        account_data = {
            'username': username,
            'platform': 'twitter'
        }

        test_db.execute('''
            INSERT OR REPLACE INTO accounts (username, platform)
            VALUES (?, ?)
        ''', (account_data['username'], account_data['platform']))

        accounts_data.append(account_data)

    test_db.commit()
    return accounts_data


@pytest.fixture
def test_tweets_different_users(test_db, test_different_accounts):
    """Create tweets for different users."""
    tweets_data = []

    for i, account in enumerate(test_different_accounts):
        tweet_data = {
            'tweet_id': f'tweet_user_{i}',
            'tweet_url': f"https://twitter.com/{account['username']}/status/tweet_user_{i}",
            'username': account['username'],
            'content': f'Content from {account["username"]}',
            'tweet_timestamp': f'2024-01-01T12:0{i}:00'
        }

        test_db.execute('''
            INSERT OR REPLACE INTO tweets
            (tweet_id, tweet_url, username, content, tweet_timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            tweet_data['tweet_id'],
            tweet_data['tweet_url'],
            tweet_data['username'],
            tweet_data['content'],
            tweet_data['tweet_timestamp']
        ))

        tweets_data.append(tweet_data)

    test_db.commit()
    return tweets_data


@pytest.fixture
def test_tweets_different_categories(test_db, test_account):
    """Create tweets with different categories for testing category filtering."""
    tweets_data = []
    categories = [Categories.HATE_SPEECH, Categories.DISINFORMATION, Categories.HATE_SPEECH]
    
    for i, category in enumerate(categories):
        tweet_data = {
            'tweet_id': f'test_{i}',
            'tweet_url': f"https://twitter.com/{test_account['username']}/status/test_{i}",
            'username': test_account['username'],
            'content': f'Test content {i}',
            'tweet_timestamp': f'2024-01-01T12:0{i}:00',
            'category': category
        }

        test_db.execute('''
            INSERT OR REPLACE INTO tweets
            (tweet_id, tweet_url, username, content, tweet_timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            tweet_data['tweet_id'],
            tweet_data['tweet_url'],
            tweet_data['username'],
            tweet_data['content'],
            tweet_data['tweet_timestamp']
        ))

        tweets_data.append(tweet_data)

    test_db.commit()
    return tweets_data