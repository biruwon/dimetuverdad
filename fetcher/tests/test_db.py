import os
import sqlite3
import tempfile
import pytest
from datetime import datetime
from fetcher import db
from utils.database import init_test_database, cleanup_test_database, _create_test_database_schema
from utils.paths import get_db_path


# ===== TEST HELPERS =====

@pytest.fixture
def test_db(tmp_path):
    """Provide a unique test database for each test."""
    # Use the standard testing database path
    db_path = get_db_path(env='testing')
    
    # But modify it to be unique per test by adding a suffix
    base, ext = os.path.splitext(db_path)
    unique_db_path = f"{base}_{tmp_path.name}{ext}"
    
    # Ensure database exists and has schema
    if not os.path.exists(unique_db_path):
        _create_test_database_schema(unique_db_path)
    
    yield unique_db_path
    
    # Clean up test database after each test
    try:
        if os.path.exists(unique_db_path):
            os.remove(unique_db_path)
    except OSError:
        pass


@pytest.fixture(autouse=True)
def override_db_path(test_db):
    """This fixture is kept for compatibility but doesn't override TEST_DATABASE_PATH anymore"""
    # Since we removed TEST_DATABASE_PATH support, this fixture is now a no-op
    # The tests will use the environment-based database path
    yield


def get_test_connection(db_path):
    """Get a connection to the test database"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def test_save_tweet_new_tweet(test_db):
    """Test saving a new tweet"""
    conn = get_test_connection(test_db)

    # Create account first (required by foreign key constraint)
    c = conn.cursor()
    c.execute("INSERT INTO accounts (username) VALUES (?)", ('testuser',))
    conn.commit()

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
    assert row['tweet_id'] == '123'
    assert row['content'] == 'Hello world'
    assert row['username'] == 'testuser'

    conn.close()


def test_save_tweet_duplicate_no_changes(test_db):
    """Test saving duplicate tweet with no changes"""
    conn = get_test_connection(test_db)

    # Create account first (required by foreign key constraint)
    c = conn.cursor()
    c.execute("INSERT INTO accounts (username) VALUES (?)", ('testuser',))
    conn.commit()

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


def test_save_tweet_update_content(test_db):
    """Test updating tweet content"""
    conn = get_test_connection(test_db)

    # Create account first (required by foreign key constraint)
    c = conn.cursor()
    c.execute("INSERT INTO accounts (username) VALUES (?)", ('testuser',))
    conn.commit()

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
    assert row['content'] == 'Updated content'

    conn.close()


def test_save_tweet_update_post_type(test_db):
    """Test updating tweet post type"""
    conn = get_test_connection(test_db)

    # Create account first (required by foreign key constraint)
    c = conn.cursor()
    c.execute("INSERT INTO accounts (username) VALUES (?)", ('testuser',))
    conn.commit()

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
    assert row['post_type'] == 'repost'

    conn.close()


def test_save_tweet_invalid_tweet_id(test_db):
    """Test saving tweet with invalid tweet_id"""
    conn = get_test_connection(test_db)

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

def test_check_if_tweet_exists_tweet_exists(test_db):
    """Test checking if tweet exists when it does"""
    # Set DATABASE_PATH environment variable to use the test database
    old_db_path = os.environ.get('DATABASE_PATH')
    os.environ['DATABASE_PATH'] = test_db
    
    try:
        conn = get_test_connection(test_db)
        c = conn.cursor()
        # Create account first
        c.execute("INSERT INTO accounts (username) VALUES (?)", ('testuser',))
        c.execute("INSERT INTO tweets (tweet_id, username, tweet_url, content) VALUES (?, ?, ?, ?)", ('123', 'testuser', 'https://x.com/testuser/status/123', 'test'))
        conn.commit()
        conn.close()

        result = db.check_if_tweet_exists('testuser', '123')
        assert result is True
    finally:
        # Restore original DATABASE_PATH
        if old_db_path is not None:
            os.environ['DATABASE_PATH'] = old_db_path
        elif 'DATABASE_PATH' in os.environ:
            del os.environ['DATABASE_PATH']


def test_check_if_tweet_exists_tweet_not_exists():
    """Test checking if tweet exists when it doesn't"""
    result = db.check_if_tweet_exists('testuser', '123')
    assert result is False


def test_check_if_tweet_exists_wrong_user(test_db):
    """Test checking if tweet exists for wrong user"""
    # Set DATABASE_PATH environment variable to use the test database
    old_db_path = os.environ.get('DATABASE_PATH')
    os.environ['DATABASE_PATH'] = test_db
    
    try:
        conn = get_test_connection(test_db)
        c = conn.cursor()
        # Create account first
        c.execute("INSERT INTO accounts (username) VALUES (?)", ('testuser',))
        c.execute("INSERT INTO tweets (tweet_id, username, tweet_url, content) VALUES (?, ?, ?, ?)", ('123', 'testuser', 'https://x.com/testuser/status/123', 'test'))
        conn.commit()
        conn.close()

        result = db.check_if_tweet_exists('otheruser', '123')
        assert result is False
    finally:
        # Restore original DATABASE_PATH
        if old_db_path is not None:
            os.environ['DATABASE_PATH'] = old_db_path
        elif 'DATABASE_PATH' in os.environ:
            del os.environ['DATABASE_PATH']


# ===== ACCOUNT PROFILE TESTS =====

def test_save_account_profile_info_new_account(test_db, capsys):
    """Test saving profile info for new account"""
    conn = get_test_connection(test_db)

    db.save_account_profile_info(conn, 'testuser', 'https://example.com/pic.jpg')

    # Verify account was saved
    c = conn.cursor()
    c.execute("SELECT * FROM accounts WHERE username = ?", ('testuser',))
    row = c.fetchone()
    assert row is not None
    assert row['username'] == 'testuser'
    assert row['profile_pic_url'] == 'https://example.com/pic.jpg'
    assert row['profile_pic_updated'] is not None
    assert row['last_scraped'] is not None

    # Check output
    captured = capsys.readouterr()
    assert '💾 Updated profile info for @testuser' in captured.out

    conn.close()


def test_save_account_profile_info_update_existing(test_db):
    """Test updating profile info for existing account"""
    conn = get_test_connection(test_db)
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
    assert row['profile_pic_url'] == 'https://example.com/new.jpg'

    conn.close()


def test_save_account_profile_info_no_url(test_db):
    """Test saving profile info with no URL (should do nothing)"""
    conn = get_test_connection(test_db)

    db.save_account_profile_info(conn, 'testuser', None)

    # Verify no account was created
    c = conn.cursor()
    c.execute("SELECT * FROM accounts WHERE username = ?", ('testuser',))
    row = c.fetchone()
    assert row is None

    conn.close()


def test_save_account_profile_info_empty_url(test_db):
    """Test saving profile info with empty URL (should do nothing)"""
    conn = get_test_connection(test_db)

    db.save_account_profile_info(conn, 'testuser', '')

    # Verify no account was created
    c = conn.cursor()
    c.execute("SELECT * FROM accounts WHERE username = ?", ('testuser',))
    row = c.fetchone()
    assert row is None

    conn.close()