import os
import sqlite3
import tempfile
import pytest
from datetime import datetime
from fetcher import db
from database import init_test_database, cleanup_test_database, create_fresh_database_schema
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
        create_fresh_database_schema(unique_db_path)
    
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


# ===== BATCH WRITE TESTS (P3 Performance Optimization) =====

class TestTweetBuffer:
    """Tests for the TweetBuffer batch write class."""
    
    def test_buffer_basic_functionality(self, test_db):
        """Test basic buffer add and flush."""
        conn = get_test_connection(test_db)
        
        # Create account first
        c = conn.cursor()
        c.execute("INSERT INTO accounts (username) VALUES (?)", ('batchuser',))
        conn.commit()
        
        buffer = db.TweetBuffer(conn, batch_size=5)
        
        # Add a tweet
        tweet = {
            'tweet_id': 'batch001',
            'content': 'Batch test tweet',
            'username': 'batchuser',
            'tweet_url': 'https://x.com/batchuser/status/batch001',
            'tweet_timestamp': '2024-01-01T12:00:00Z',
            'post_type': 'original'
        }
        
        result = buffer.add(tweet)
        assert result is True
        assert len(buffer) == 1
        
        # Flush and verify
        written = buffer.flush()
        assert written == 1
        assert len(buffer) == 0
        
        # Check database
        c.execute("SELECT * FROM tweets WHERE tweet_id = ?", ('batch001',))
        row = c.fetchone()
        assert row is not None
        assert row['content'] == 'Batch test tweet'
        
        conn.close()
    
    def test_buffer_auto_flush_on_size(self, test_db):
        """Test that buffer auto-flushes when batch_size is reached."""
        conn = get_test_connection(test_db)
        
        # Create account
        c = conn.cursor()
        c.execute("INSERT INTO accounts (username) VALUES (?)", ('batchuser',))
        conn.commit()
        
        buffer = db.TweetBuffer(conn, batch_size=3)
        
        # Add tweets up to batch size
        for i in range(3):
            buffer.add({
                'tweet_id': f'auto{i}',
                'content': f'Tweet {i}',
                'username': 'batchuser',
                'tweet_url': f'https://x.com/batchuser/status/auto{i}',
                'tweet_timestamp': '2024-01-01T12:00:00Z',
                'post_type': 'original'
            })
        
        # Buffer should be empty after auto-flush
        assert len(buffer) == 0
        
        # Verify all tweets were written
        c.execute("SELECT COUNT(*) FROM tweets WHERE tweet_id LIKE 'auto%'")
        count = c.fetchone()[0]
        assert count == 3
        
        conn.close()
    
    def test_buffer_skips_duplicates(self, test_db):
        """Test that duplicate tweet IDs are skipped in buffer."""
        conn = get_test_connection(test_db)
        
        # Create account
        c = conn.cursor()
        c.execute("INSERT INTO accounts (username) VALUES (?)", ('batchuser',))
        conn.commit()
        
        buffer = db.TweetBuffer(conn, batch_size=10)
        
        tweet = {
            'tweet_id': 'dup001',
            'content': 'First version',
            'username': 'batchuser',
            'tweet_url': 'https://x.com/batchuser/status/dup001',
            'post_type': 'original'
        }
        
        # Add same tweet twice
        result1 = buffer.add(tweet)
        result2 = buffer.add(tweet)
        
        assert result1 is True
        assert result2 is False
        assert len(buffer) == 1
        
        conn.close()
    
    def test_buffer_skips_analytics(self, test_db):
        """Test that 'analytics' tweet IDs are skipped."""
        conn = get_test_connection(test_db)
        
        buffer = db.TweetBuffer(conn, batch_size=10)
        
        result = buffer.add({
            'tweet_id': 'analytics',
            'content': 'Should not be saved',
            'username': 'user'
        })
        
        assert result is False
        assert len(buffer) == 0
        
        conn.close()
    
    def test_buffer_context_manager(self, test_db):
        """Test buffer as context manager (auto-flush on exit)."""
        conn = get_test_connection(test_db)
        
        # Create account
        c = conn.cursor()
        c.execute("INSERT INTO accounts (username) VALUES (?)", ('ctxuser',))
        conn.commit()
        
        with db.TweetBuffer(conn, batch_size=100) as buffer:
            buffer.add({
                'tweet_id': 'ctx001',
                'content': 'Context manager test',
                'username': 'ctxuser',
                'tweet_url': 'https://x.com/ctxuser/status/ctx001',
                'post_type': 'original'
            })
            # Not flushed yet (batch_size=100)
            assert len(buffer) == 1
        
        # Should be auto-flushed on exit
        c.execute("SELECT * FROM tweets WHERE tweet_id = ?", ('ctx001',))
        assert c.fetchone() is not None
        
        conn.close()
    
    def test_buffer_stats_tracking(self, test_db):
        """Test that buffer tracks statistics."""
        conn = get_test_connection(test_db)
        
        # Create account
        c = conn.cursor()
        c.execute("INSERT INTO accounts (username) VALUES (?)", ('statsuser',))
        conn.commit()
        
        buffer = db.TweetBuffer(conn, batch_size=2)
        
        # Add 3 tweets (will trigger 1 auto-flush at 2, then manual flush)
        for i in range(3):
            buffer.add({
                'tweet_id': f'stats{i}',
                'content': f'Stats test {i}',
                'username': 'statsuser',
                'tweet_url': f'https://x.com/statsuser/status/stats{i}',
                'post_type': 'original'
            })
        
        buffer.flush()  # Flush remaining
        
        stats = buffer.get_stats()
        assert stats.total_tweets == 3
        assert stats.batches_written == 2  # 2 at auto-flush + 1 at manual
        assert stats.total_time > 0
        
        conn.close()
    
    def test_buffer_update_existing(self, test_db):
        """Test that buffer updates existing tweets."""
        conn = get_test_connection(test_db)
        
        # Create account and existing tweet
        c = conn.cursor()
        c.execute("INSERT INTO accounts (username) VALUES (?)", ('updateuser',))
        c.execute("""
            INSERT INTO tweets (tweet_id, content, username, tweet_url, post_type)
            VALUES ('existing001', 'Original content', 'updateuser', 
                    'https://x.com/updateuser/status/existing001', 'original')
        """)
        conn.commit()
        
        buffer = db.TweetBuffer(conn, batch_size=10)
        
        # Add tweet with same ID but different content
        buffer.add({
            'tweet_id': 'existing001',
            'content': 'Updated content',
            'username': 'updateuser',
            'tweet_url': 'https://x.com/updateuser/status/existing001',
            'post_type': 'original'
        })
        buffer.flush()
        
        # Verify update
        c.execute("SELECT content FROM tweets WHERE tweet_id = ?", ('existing001',))
        row = c.fetchone()
        assert row['content'] == 'Updated content'
        
        conn.close()


class TestBatchWriteStats:
    """Tests for BatchWriteStats dataclass."""
    
    def test_avg_batch_time_zero_division(self):
        """Test avg_batch_time handles zero batches."""
        stats = db.BatchWriteStats()
        assert stats.avg_batch_time == 0.0
    
    def test_avg_batch_time_calculation(self):
        """Test avg_batch_time calculation."""
        stats = db.BatchWriteStats(
            total_tweets=100,
            batches_written=10,
            total_time=5.0
        )
        assert stats.avg_batch_time == 0.5


class TestDeleteAccountData:
    """Tests for delete_account_data function."""
    
    def test_deletes_account_data_with_mocking(self, monkeypatch):
        """Should delete tweets and analyses for a user using mocking."""
        from unittest.mock import MagicMock, patch
        
        # Mock the tweet repository
        mock_tweet_repo = MagicMock()
        
        # Track rowcount values - the function reads rowcount after each DELETE
        # Order: SELECT, model_analyses DELETE, user_feedback DELETE, post_edits DELETE, analyses DELETE, tweets DELETE
        rowcount_values = iter([0, 1, 2, 3, 5, 10])  # values for each execute that reads rowcount
        
        class MockCursor:
            def __init__(self):
                self._rowcount = 0
                
            def execute(self, sql, params=None):
                # After execute, set rowcount based on which query this is
                try:
                    self._rowcount = next(rowcount_values)
                except StopIteration:
                    self._rowcount = 0
                    
            def fetchall(self):
                return [('tweet1',), ('tweet2',)]  # Return some tweet_ids
                
            @property
            def rowcount(self):
                return self._rowcount
        
        mock_cursor = MockCursor()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        
        def mock_context_manager():
            return mock_conn
        
        with patch('fetcher.db.get_tweet_repository', return_value=mock_tweet_repo):
            with patch('database.get_db_connection_context', mock_context_manager):
                result = db.delete_account_data('testuser')
        
        assert result['tweets'] == 10
        assert result['analyses'] == 5
    
    def test_delete_account_data_handles_exception(self, monkeypatch):
        """Should raise exception on database error."""
        from unittest.mock import MagicMock, patch
        
        mock_tweet_repo = MagicMock()
        
        def raise_error():
            raise Exception("Database error")
        
        with patch('fetcher.db.get_tweet_repository', return_value=mock_tweet_repo):
            with patch('database.get_db_connection_context', side_effect=raise_error):
                with pytest.raises(Exception):
                    db.delete_account_data('testuser')


class TestUpdateTweetInDatabase:
    """Tests for update_tweet_in_database function."""
    
    def test_updates_tweet_with_mocking(self):
        """Should update tweet and return True on success."""
        from unittest.mock import MagicMock, patch
        
        # Create mock cursor and connection
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        
        def mock_context_manager():
            return mock_conn
        
        with patch('database.get_db_connection_context', mock_context_manager):
            result = db.update_tweet_in_database('tweet123', {
                'content': 'Updated',
                'engagement_likes': 100,
            })
        
        assert result is True
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()
    
    def test_returns_false_when_no_rows_updated(self):
        """Should return False when tweet doesn't exist."""
        from unittest.mock import MagicMock, patch
        
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        
        def mock_context_manager():
            return mock_conn
        
        with patch('database.get_db_connection_context', mock_context_manager):
            result = db.update_tweet_in_database('nonexistent', {'content': 'Test'})
        
        assert result is False
    
    def test_returns_false_on_exception(self):
        """Should return False on database error."""
        from unittest.mock import patch
        
        def raise_error():
            raise Exception("Database error")
        
        with patch('database.get_db_connection_context', side_effect=raise_error):
            result = db.update_tweet_in_database('tweet123', {'content': 'Test'})
        
        assert result is False


class TestInitDb:
    """Tests for init_db function."""
    
    def test_init_db_creates_scrape_errors_table(self):
        """Should create scrape_errors table."""
        from unittest.mock import MagicMock, patch
        
        mock_cursor = MagicMock()
        # First query checks for tweets table - return a result
        mock_cursor.fetchone.return_value = ('tweets',)
        
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        
        with patch('database.get_db_connection', return_value=mock_conn):
            result = db.init_db()
        
        assert result == mock_conn
        # Verify CREATE TABLE was called
        calls = [str(call) for call in mock_cursor.execute.call_args_list]
        assert any('scrape_errors' in str(call) for call in calls)
    
    def test_init_db_raises_when_tweets_table_missing(self):
        """Should raise exception when tweets table doesn't exist."""
        from unittest.mock import MagicMock, patch
        
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None  # No tweets table
        
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        
        with patch('database.get_db_connection', return_value=mock_conn):
            with pytest.raises(Exception, match="Database not properly initialized"):
                db.init_db()


# ===== Tests for get_tweet_by_id =====

class TestGetTweetById:
    """Tests for get_tweet_by_id function."""
    
    def test_returns_tweet_when_exists(self, test_db):
        """Test that existing tweet is returned."""
        conn = get_test_connection(test_db)
        
        # Create account first
        c = conn.cursor()
        c.execute("INSERT INTO accounts (username) VALUES (?)", ('testuser',))
        conn.commit()
        
        tweet_data = {
            'tweet_id': '123456789',
            'username': 'testuser',
            'content': 'Test tweet content',
            'tweet_url': 'https://x.com/testuser/status/123456789',
        }
        db.save_tweet(conn, tweet_data)
        
        result = db.get_tweet_by_id(conn, '123456789')
        assert result is not None
        assert result['tweet_id'] == '123456789'
        assert result['username'] == 'testuser'
        assert result['content'] == 'Test tweet content'
        conn.close()
    
    def test_returns_none_when_not_exists(self, test_db):
        """Test that None is returned for non-existent tweet."""
        conn = get_test_connection(test_db)
        
        result = db.get_tweet_by_id(conn, 'nonexistent123')
        assert result is None
        conn.close()


# ===== Tests for is_thread_already_collected =====

class TestIsThreadAlreadyCollected:
    """Tests for is_thread_already_collected function."""
    
    def test_returns_false_when_no_thread(self, test_db):
        """Test returns False when thread doesn't exist."""
        conn = get_test_connection(test_db)
        
        result = db.is_thread_already_collected(conn, '999999')
        assert result is False
        conn.close()
    
    def test_returns_false_when_tweet_exists_but_no_thread_metadata(self, test_db):
        """Test returns False when tweet exists but has no thread metadata."""
        conn = get_test_connection(test_db)
        
        # Create account first
        c = conn.cursor()
        c.execute("INSERT INTO accounts (username) VALUES (?)", ('testuser',))
        conn.commit()
        
        tweet_data = {
            'tweet_id': '123456789',
            'username': 'testuser',
            'content': 'Test tweet',
            'tweet_url': 'https://x.com/testuser/status/123456789',
        }
        db.save_tweet(conn, tweet_data)
        
        result = db.is_thread_already_collected(conn, '123456789')
        assert result is False
        conn.close()
    
    def test_returns_true_when_thread_start_exists(self, test_db):
        """Test returns True when thread start tweet exists."""
        conn = get_test_connection(test_db)
        
        # Create account first
        c = conn.cursor()
        c.execute("INSERT INTO accounts (username) VALUES (?)", ('testuser',))
        conn.commit()
        
        tweet_data = {
            'tweet_id': '123456789',
            'username': 'testuser',
            'content': 'Thread start',
            'tweet_url': 'https://x.com/testuser/status/123456789',
            'thread_id': '123456789',
            'is_thread_start': 1,
            'thread_position': 0,
        }
        db.save_tweet(conn, tweet_data)
        
        result = db.is_thread_already_collected(conn, '123456789')
        assert result is True
        conn.close()
    
    def test_returns_true_when_tweet_is_continuation_in_collected_thread(self, test_db):
        """Test returns True when checking a continuation tweet that's part of a collected thread."""
        conn = get_test_connection(test_db)
        
        # Create account first
        c = conn.cursor()
        c.execute("INSERT INTO accounts (username) VALUES (?)", ('testuser',))
        conn.commit()
        
        # Save a thread continuation (has thread_id set)
        tweet_data = {
            'tweet_id': '123456790',
            'username': 'testuser',
            'content': 'Thread continuation',
            'tweet_url': 'https://x.com/testuser/status/123456790',
            'thread_id': '123456789',
            'is_thread_start': 0,
            'thread_position': 1,
        }
        db.save_tweet(conn, tweet_data)
        
        # Check for the continuation tweet itself - should be True since it has thread_id
        result = db.is_thread_already_collected(conn, '123456790')
        assert result is True
        conn.close()
    
    def test_returns_true_when_checking_thread_start_id_with_existing_members(self, test_db):
        """Test returns True when checking thread_id and thread members exist."""
        conn = get_test_connection(test_db)
        
        # Create account first
        c = conn.cursor()
        c.execute("INSERT INTO accounts (username) VALUES (?)", ('testuser',))
        conn.commit()
        
        # Save a thread continuation referencing thread_id = 123456789
        tweet_data = {
            'tweet_id': '123456790',
            'username': 'testuser',
            'content': 'Thread continuation',
            'tweet_url': 'https://x.com/testuser/status/123456790',
            'thread_id': '123456789',
            'is_thread_start': 0,
            'thread_position': 1,
        }
        db.save_tweet(conn, tweet_data)
        
        # Check for the thread_id (start tweet) - should be True since we have members referencing it
        result = db.is_thread_already_collected(conn, '123456789')
        assert result is True
        conn.close()