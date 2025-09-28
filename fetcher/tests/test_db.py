import os
import sqlite3
import tempfile
import pytest
from datetime import datetime
from fetcher import db


# ===== TEST HELPERS =====

@pytest.fixture
def temp_db():
    """Create a temporary database for testing"""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)

    # Create tables
    conn = sqlite3.connect(path)
    c = conn.cursor()

    # Create tweets table
    c.execute('''
        CREATE TABLE tweets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tweet_id TEXT UNIQUE NOT NULL,
            content TEXT,
            username TEXT,
            tweet_url TEXT,
            tweet_timestamp TEXT,
            post_type TEXT DEFAULT 'original',
            original_author TEXT,
            original_tweet_id TEXT
        )
    ''')

    # Create accounts table
    c.execute('''
        CREATE TABLE accounts (
            username TEXT PRIMARY KEY,
            profile_pic_url TEXT,
            profile_pic_updated TEXT,
            last_scraped TEXT
        )
    ''')

    conn.commit()
    conn.close()

    # Override DB_PATH for testing
    original_path = db.DB_PATH
    db.DB_PATH = path

    yield path

    # Cleanup
    db.DB_PATH = original_path
    os.remove(path)


def get_test_connection(db_path):
    """Get a connection to the test database"""
    return sqlite3.connect(db_path, timeout=10.0)


# ===== DATABASE CONNECTION TESTS =====

def test_get_connection(temp_db):
    """Test database connection creation"""
    conn = db.get_connection()
    assert conn is not None
    assert isinstance(conn, sqlite3.Connection)
    conn.close()


# ===== TWEET TIMESTAMP TESTS =====

def test_get_last_tweet_timestamp_no_tweets(temp_db):
    """Test getting last tweet timestamp when no tweets exist"""
    result = db.get_last_tweet_timestamp('testuser')
    assert result is None


def test_get_last_tweet_timestamp_with_tweets(temp_db):
    """Test getting last tweet timestamp with existing tweets"""
    conn = get_test_connection(temp_db)
    c = conn.cursor()

    # Insert test tweets
    c.execute("INSERT INTO tweets (tweet_id, username, tweet_timestamp) VALUES (?, ?, ?)",
              ('1', 'testuser', '2024-01-01T10:00:00Z'))
    c.execute("INSERT INTO tweets (tweet_id, username, tweet_timestamp) VALUES (?, ?, ?)",
              ('2', 'testuser', '2024-01-02T10:00:00Z'))
    c.execute("INSERT INTO tweets (tweet_id, username, tweet_timestamp) VALUES (?, ?, ?)",
              ('3', 'otheruser', '2024-01-03T10:00:00Z'))
    conn.commit()
    conn.close()

    # Test getting latest for testuser
    result = db.get_last_tweet_timestamp('testuser')
    assert result == '2024-01-02T10:00:00Z'

    # Test getting latest for otheruser
    result = db.get_last_tweet_timestamp('otheruser')
    assert result == '2024-01-03T10:00:00Z'

    # Test non-existent user
    result = db.get_last_tweet_timestamp('nonexistent')
    assert result is None


# ===== SAVE TWEET TESTS =====

def test_save_tweet_new_tweet(temp_db):
    """Test saving a new tweet"""
    conn = get_test_connection(temp_db)

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
    assert row[2] == 'Hello world'  # content
    assert row[3] == 'testuser'  # username

    conn.close()


def test_save_tweet_duplicate_no_changes(temp_db):
    """Test saving duplicate tweet with no changes"""
    conn = get_test_connection(temp_db)

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


def test_save_tweet_update_content(temp_db):
    """Test updating tweet content"""
    conn = get_test_connection(temp_db)

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


def test_save_tweet_update_post_type(temp_db):
    """Test updating tweet post type"""
    conn = get_test_connection(temp_db)

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


def test_save_tweet_invalid_tweet_id(temp_db):
    """Test saving tweet with invalid tweet_id"""
    conn = get_test_connection(temp_db)

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


# ===== ENHANCED TWEET TESTS =====

def test_save_enhanced_tweet_success(temp_db, capsys):
    """Test saving enhanced tweet successfully"""
    conn = get_test_connection(temp_db)

    tweet_data = {
        'tweet_id': '123',
        'content': 'Hello world',
        'username': 'testuser',
        'tweet_url': 'https://twitter.com/testuser/status/123',
        'tweet_timestamp': '2024-01-01T10:00:00Z'
    }

    result = db.save_enhanced_tweet(conn, tweet_data)
    assert result is True

    # Check output
    captured = capsys.readouterr()
    assert '✅ Saved/Updated tweet: 123' in captured.out

    conn.close()


def test_save_enhanced_tweet_duplicate(temp_db, capsys):
    """Test saving duplicate enhanced tweet"""
    conn = get_test_connection(temp_db)

    tweet_data = {
        'tweet_id': '123',
        'content': 'Hello world',
        'username': 'testuser'
    }

    # Save first time
    db.save_enhanced_tweet(conn, tweet_data)

    # Save again
    result = db.save_enhanced_tweet(conn, tweet_data)
    assert result is False

    # Check output
    captured = capsys.readouterr()
    assert '⏭️ Not saved (duplicate/unchanged): 123' in captured.out

    conn.close()


# ===== TWEET EXISTENCE CHECKS =====

def test_check_if_tweet_exists_tweet_exists(temp_db):
    """Test checking if tweet exists when it does"""
    conn = get_test_connection(temp_db)
    c = conn.cursor()
    c.execute("INSERT INTO tweets (tweet_id, username) VALUES (?, ?)", ('123', 'testuser'))
    conn.commit()
    conn.close()

    result = db.check_if_tweet_exists('testuser', '123')
    assert result is True


def test_check_if_tweet_exists_tweet_not_exists(temp_db):
    """Test checking if tweet exists when it doesn't"""
    result = db.check_if_tweet_exists('testuser', '123')
    assert result is False


def test_check_if_tweet_exists_wrong_user(temp_db):
    """Test checking if tweet exists for wrong user"""
    conn = get_test_connection(temp_db)
    c = conn.cursor()
    c.execute("INSERT INTO tweets (tweet_id, username) VALUES (?, ?)", ('123', 'testuser'))
    conn.commit()
    conn.close()

    result = db.check_if_tweet_exists('otheruser', '123')
    assert result is False


# ===== ACCOUNT PROFILE TESTS =====

def test_save_account_profile_info_new_account(temp_db, capsys):
    """Test saving profile info for new account"""
    conn = get_test_connection(temp_db)

    db.save_account_profile_info(conn, 'testuser', 'https://example.com/pic.jpg')

    # Verify account was saved
    c = conn.cursor()
    c.execute("SELECT * FROM accounts WHERE username = ?", ('testuser',))
    row = c.fetchone()
    assert row is not None
    assert row[0] == 'testuser'
    assert row[1] == 'https://example.com/pic.jpg'
    assert row[2] is not None  # profile_pic_updated
    assert row[3] is not None  # last_scraped

    # Check output
    captured = capsys.readouterr()
    assert '💾 Updated profile info for @testuser' in captured.out

    conn.close()


def test_save_account_profile_info_update_existing(temp_db):
    """Test updating profile info for existing account"""
    conn = get_test_connection(temp_db)
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


def test_save_account_profile_info_no_url(temp_db):
    """Test saving profile info with no URL (should do nothing)"""
    conn = get_test_connection(temp_db)

    db.save_account_profile_info(conn, 'testuser', None)

    # Verify no account was created
    c = conn.cursor()
    c.execute("SELECT * FROM accounts WHERE username = ?", ('testuser',))
    row = c.fetchone()
    assert row is None

    conn.close()


def test_save_account_profile_info_empty_url(temp_db):
    """Test saving profile info with empty URL (should do nothing)"""
    conn = get_test_connection(temp_db)

    db.save_account_profile_info(conn, 'testuser', '')

    # Verify no account was created
    c = conn.cursor()
    c.execute("SELECT * FROM accounts WHERE username = ?", ('testuser',))
    row = c.fetchone()
    assert row is None

    conn.close()