import os
import sqlite3
import tempfile
import pytest
from datetime import datetime
from fetcher import db
from utils.database import init_test_database, cleanup_test_database


# ===== TEST HELPERS =====

@pytest.fixture(scope="class")
def test_db():
    """Create a test database for all tests in this class"""
    # Initialize test database
    db_path = init_test_database()
    yield db_path
    # Cleanup after all tests in this class
    cleanup_test_database()


@pytest.fixture(autouse=True)
def override_db_path(test_db):
    """Override DB_PATH for all tests"""
    original_path = db.DB_PATH
    db.DB_PATH = test_db
    yield
    db.DB_PATH = original_path


def get_test_connection():
    """Get a connection to the test database"""
    return db.get_connection()


# ===== DATABASE CONNECTION TESTS =====

def test_get_connection():
    """Test database connection creation"""
    conn = db.get_connection()
    assert conn is not None
    assert isinstance(conn, sqlite3.Connection)
    conn.close()

# ===== SAVE TWEET TESTS =====

def test_save_tweet_new_tweet():
    """Test saving a new tweet"""
    conn = get_test_connection()

    tweet_data = {
        'tweet_id': '123',
        'content': 'Hello world',
        'username': 'testuser',
        'tweet_url': 'https://twitter.com/testuser/status/123',
        'tweet_timestamp': '2024-01-01T10:00:00Z',
        'post_type': 'original'
    }

    result = db.save_tweet(conn, tweet_data)
    assert result is True

    # Verify tweet was saved
    c = conn.cursor()
    c.execute("SELECT * FROM tweets WHERE tweet_id = ?", ('123',))
    row = c.fetchone()
    assert row is not None
    assert row[1] == '123'  # tweet_id
    assert row[4] == 'Hello world'  # content
    assert row[3] == 'testuser'  # username

    conn.close()


def test_save_tweet_duplicate_no_changes():
    """Test saving duplicate tweet with no changes"""
    conn = get_test_connection()

    tweet_data = {
        'tweet_id': '123',
        'content': 'Hello world',
        'username': 'testuser',
        'tweet_url': 'https://twitter.com/testuser/status/123',
        'tweet_timestamp': '2024-01-01T10:00:00Z',
        'post_type': 'original'
    }

    # Save first time
    result1 = db.save_tweet(conn, tweet_data)
    assert result1 is True

    # Save again with same data
    result2 = db.save_tweet(conn, tweet_data)
    assert result2 is False  # Should return False for no changes

    conn.close()


def test_save_tweet_update_content():
    """Test updating tweet content"""
    conn = get_test_connection()

    tweet_data = {
        'tweet_id': '123',
        'content': 'Hello world',
        'username': 'testuser',
        'tweet_url': 'https://twitter.com/testuser/status/123',
        'tweet_timestamp': '2024-01-01T10:00:00Z',
        'post_type': 'original'
    }

    # Save first time
    db.save_tweet(conn, tweet_data)

    # Update content
    tweet_data['content'] = 'Updated content'
    result = db.save_tweet(conn, tweet_data)
    assert result is True

    # Verify update
    c = conn.cursor()
    c.execute("SELECT content FROM tweets WHERE tweet_id = ?", ('123',))
    row = c.fetchone()
    assert row[0] == 'Updated content'

    conn.close()


def test_save_tweet_update_post_type():
    """Test updating tweet post type"""
    conn = get_test_connection()

    tweet_data = {
        'tweet_id': '123',
        'content': 'Hello world',
        'username': 'testuser',
        'tweet_url': 'https://twitter.com/testuser/status/123',
        'tweet_timestamp': '2024-01-01T10:00:00Z',
        'post_type': 'original'
    }

    # Save first time
    db.save_tweet(conn, tweet_data)

    # Update post type
    tweet_data['post_type'] = 'repost'
    result = db.save_tweet(conn, tweet_data)
    assert result is True

    # Verify update
    c = conn.cursor()
    c.execute("SELECT post_type FROM tweets WHERE tweet_id = ?", ('123',))
    row = c.fetchone()
    assert row[0] == 'repost'

    conn.close()


def test_save_tweet_invalid_tweet_id():
    """Test saving tweet with invalid tweet_id"""
    conn = get_test_connection()

    # Test with None tweet_id
    tweet_data = {
        'tweet_id': None,
        'content': 'Hello world',
        'username': 'testuser'
    }
    result = db.save_tweet(conn, tweet_data)
    assert result is False

    # Test with 'analytics' tweet_id
    tweet_data['tweet_id'] = 'analytics'
    result = db.save_tweet(conn, tweet_data)
    assert result is False

    conn.close()

# ===== TWEET EXISTENCE CHECKS =====

def test_check_if_tweet_exists_tweet_exists():
    """Test checking if tweet exists when it does"""
    conn = get_test_connection()
    c = conn.cursor()
    c.execute("INSERT INTO tweets (tweet_id, username, tweet_url, content) VALUES (?, ?, ?, ?)", ('123', 'testuser', 'https://x.com/testuser/status/123', 'test'))
    conn.commit()
    conn.close()

    result = db.check_if_tweet_exists('testuser', '123')
    assert result is True


def test_check_if_tweet_exists_tweet_not_exists():
    """Test checking if tweet exists when it doesn't"""
    result = db.check_if_tweet_exists('testuser', '123')
    assert result is False


def test_check_if_tweet_exists_wrong_user():
    """Test checking if tweet exists for wrong user"""
    conn = get_test_connection()
    c = conn.cursor()
    c.execute("INSERT INTO tweets (tweet_id, username, tweet_url, content) VALUES (?, ?, ?, ?)", ('123', 'testuser', 'https://x.com/testuser/status/123', 'test'))
    conn.commit()
    conn.close()

    result = db.check_if_tweet_exists('otheruser', '123')
    assert result is False


# ===== ACCOUNT PROFILE TESTS =====

def test_save_account_profile_info_new_account(capsys):
    """Test saving profile info for new account"""
    conn = get_test_connection()

    db.save_account_profile_info(conn, 'testuser', 'https://example.com/pic.jpg')

    # Verify account was saved
    c = conn.cursor()
    c.execute("SELECT * FROM accounts WHERE username = ?", ('testuser',))
    row = c.fetchone()
    assert row is not None
    assert row[1] == 'testuser'  # username is at index 1
    assert row[3] == 'https://example.com/pic.jpg'  # profile_pic_url is at index 3
    assert row[4] is not None  # profile_pic_updated
    assert row[5] is not None  # last_scraped

    # Check output
    captured = capsys.readouterr()
    assert '💾 Updated profile info for @testuser' in captured.out

    conn.close()


def test_save_account_profile_info_update_existing():
    """Test updating profile info for existing account"""
    conn = get_test_connection()
    c = conn.cursor()

    # Insert existing account
    c.execute("INSERT INTO accounts (username, profile_pic_url) VALUES (?, ?)",
              ('testuser', 'https://example.com/old.jpg'))
    conn.commit()

    # Update profile
    db.save_account_profile_info(conn, 'testuser', 'https://example.com/new.jpg')

    # Verify update
    c.execute("SELECT profile_pic_url FROM accounts WHERE username = ?", ('testuser',))
    row = c.fetchone()
    assert row[0] == 'https://example.com/new.jpg'

    conn.close()


def test_save_account_profile_info_no_url():
    """Test saving profile info with no URL (should do nothing)"""
    conn = get_test_connection()

    db.save_account_profile_info(conn, 'testuser', None)

    # Verify no account was created
    c = conn.cursor()
    c.execute("SELECT * FROM accounts WHERE username = ?", ('testuser',))
    row = c.fetchone()
    assert row is None

    conn.close()


def test_save_account_profile_info_empty_url():
    """Test saving profile info with empty URL (should do nothing)"""
    conn = get_test_connection()

    db.save_account_profile_info(conn, 'testuser', '')

    # Verify no account was created
    c = conn.cursor()
    c.execute("SELECT * FROM accounts WHERE username = ?", ('testuser',))
    row = c.fetchone()
    assert row is None

    conn.close()