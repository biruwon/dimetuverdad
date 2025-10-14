"""
Shared pytest fixtures for analyzer tests.
Provides common test data setup and database management.
"""

import pytest
import sqlite3
import os
import glob
from typing import Dict, Any
from analyzer.models import ContentAnalysis
from analyzer.categories import Categories
from utils.database import init_test_database, cleanup_test_database
from utils import paths


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_databases():
    """Clean up test databases at the end of the test session."""
    yield  # Run tests first

    # Clean up all test databases after tests complete
    base_db_path = paths.get_db_path(env='testing')
    test_pattern = f"{base_db_path}.pid_*"

    for db_file in glob.glob(test_pattern):
        try:
            os.remove(db_file)
            print(f"🗑️  Cleaned up test database: {os.path.basename(db_file)}")
        except OSError as e:
            print(f"⚠️  Could not remove test database {db_file}: {e}")


@pytest.fixture
def test_db_path():
    """Provide a test database path for each test."""
    db_path = init_test_database(fixtures=False)
    yield db_path
    # Cleanup happens automatically via atexit in init_test_database


@pytest.fixture(autouse=True)
def test_db(test_db_path):
    """Provide a fresh test database connection for each test."""
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