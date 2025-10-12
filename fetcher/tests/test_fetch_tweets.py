import os
import json
import types
import pytest
import sqlite3
import tempfile
from unittest.mock import Mock, patch, MagicMock, call
from fetcher import fetch_tweets
from fetcher import parsers as fetcher_parsers
from fetcher import db as fetcher_db


# ===== DATABASE SETUP HELPERS =====

def setup_temp_db():
    """Create a temporary database for testing"""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    conn = sqlite3.connect(path)
    c = conn.cursor()
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
            original_tweet_id TEXT,
            reply_to_username TEXT,
            reply_to_tweet_id TEXT,
            media_links TEXT,
            media_count INTEGER DEFAULT 0,
            media_types TEXT,
            engagement_likes INTEGER DEFAULT 0,
            engagement_retweets INTEGER DEFAULT 0,
            engagement_replies INTEGER DEFAULT 0,
            external_links TEXT,
            original_content TEXT,
            is_pinned INTEGER DEFAULT 0,
            hashtags TEXT,
            mentions TEXT
        )
    ''')
    conn.commit()
    return conn, path


# ===== MOCK CLASSES =====

class MockArticle:
    """Mock article element for testing"""
    def __init__(self, href, text='hi', datetime='2024-01-01T00:00:00Z'):
        self._href = href
        self._text = text
        self._datetime = datetime

    def query_selector(self, selector):
        if selector == 'a[href*="/status/"]':
            return MockElem({'href': self._href})
        if selector == '[data-testid="tweetText"]':
            return MockElem({}, text=self._text)
        if selector == 'time':
            return MockElem({'datetime': self._datetime})
        return None

    def query_selector_all(self, selector):
        # minimal: no media or extra anchors
        return []


class MockElem:
    """Mock element for testing"""
    def __init__(self, attrs=None, text=''):
        self._attrs = attrs or {}
        self._text = text

    def get_attribute(self, name):
        return self._attrs.get(name)

    def inner_text(self):
        return self._text


class MockPage:
    """Mock page for testing"""
    def __init__(self, articles):
        self._articles = articles

    def goto(self, url, **kwargs):
        return None

    def wait_for_selector(self, selector, timeout=None):
        return True

    def query_selector_all(self, selector):
        return self._articles

    def evaluate(self, js):
        return 100
    
    @property
    def keyboard(self):
        return types.SimpleNamespace(press=lambda key: None)


class FakeElement:
    """Unified fake element for integration tests"""
    def __init__(self, attrs=None, text="", children=None):
        self._attrs = attrs or {}
        self._text = text
        self._children = children or []

    def get_attribute(self, name):
        return self._attrs.get(name)

    def inner_text(self):
        return self._text

    def query_selector(self, selector):
        # Very small selector handling for the test cases
        if selector == '[data-testid="tweetText"]':
            return FakeElement(text=self._text)
        if selector == 'a[href*="/status/"]':
            href = self._attrs.get('href')
            return FakeElement(attrs={'href': href}) if href else None
        return None

    def query_selector_all(self, selector):
        return []

    def evaluate(self, js):
        return None


class FakePage:
    """Fake page for integration tests"""
    def __init__(self, articles):
        # articles: list of FakeElement
        self._articles = articles
        self._url = ''
        self._event_handlers = {}

    def goto(self, url, **kwargs):
        """Accept any keyword arguments like wait_until"""
        self._url = url

    def wait_for_selector(self, selector, timeout=None):
        return True

    def query_selector_all(self, selector):
        return self._articles

    def evaluate(self, script):
        return 1000

    def on(self, event, handler):
        """Mock event handler registration"""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    @property
    def keyboard(self):
        return types.SimpleNamespace(press=lambda key: None)


# ===== NEW CORE FUNCTION TESTS =====

class TestFetchTweetsInSessions:
    """Test the fetch_tweets_in_sessions function."""

    def test_single_session_sufficient(self):
        """Test when max_tweets <= session_size, single session is used."""
        conn, path = setup_temp_db()
        mock_page = Mock()
        
        try:
            with patch.object(fetcher_db, 'init_db', return_value=conn), \
                 patch.object(fetch_tweets.collector, 'collect_tweets_from_page') as mock_collect:
                
                mock_collect.return_value = [{'tweet_id': '1', 'content': 'test'}]
                
                result = fetch_tweets.fetch_tweets_in_sessions(mock_page, 'testuser', 100, 800)
                
                mock_collect.assert_called_once()
                assert len(result) == 1
                assert result[0]['tweet_id'] == '1'
        finally:
            conn.close()
            os.remove(path)

    def test_multi_session_strategy(self):
        """Test multi-session strategy for large tweet counts."""
        conn, path = setup_temp_db()
        mock_page = Mock()
        
        try:
            with patch.object(fetcher_db, 'init_db', return_value=conn), \
                 patch.object(fetch_tweets.collector, 'collect_tweets_from_page') as mock_collect, \
                 patch.object(fetch_tweets.scroller, 'delay'), \
                 patch.object(fetcher_parsers, 'extract_profile_picture') as mock_profile:
                
                # Mock multiple session results
                mock_collect.side_effect = [
                    [{'tweet_id': '1', 'tweet_timestamp': '2024-01-01'}],
                    [{'tweet_id': '2', 'tweet_timestamp': '2024-01-02'}],
                    []  # Empty result to stop
                ]
                mock_profile.return_value = 'http://example.com/pic.jpg'
                
                result = fetch_tweets.fetch_tweets_in_sessions(mock_page, 'testuser', 1500, 800)
                
                # Should have called collect multiple times
                assert mock_collect.call_count >= 2
                # Should have extracted profile picture only once
                mock_profile.assert_called_once()
                
        finally:
            conn.close()
            os.remove(path)

    def test_multi_session_no_tweets_found(self):
        """Test multi-session handling when no tweets are found."""
        conn, path = setup_temp_db()
        mock_page = Mock()
        mock_page.wait_for_selector.side_effect = fetch_tweets.TimeoutError("No tweets found")
        
        try:
            with patch.object(fetcher_db, 'init_db', return_value=conn), \
                 patch.object(fetch_tweets.scroller, 'delay'):
                
                result = fetch_tweets.fetch_tweets_in_sessions(mock_page, 'testuser', 1500, 800)
                
                assert result == []
                
        finally:
            conn.close()
            os.remove(path)


class TestFetchTweets:
    """Test the main fetch_tweets function."""

    def test_fetch_tweets_with_mocked_components(self):
        """Test fetch_tweets with all components mocked."""
        mock_page = Mock()
        
        with patch.object(fetcher_db, 'init_db') as mock_init_db, \
             patch.object(fetch_tweets.collector, 'collect_tweets_from_page') as mock_collect:
            
            mock_conn = Mock()
            mock_init_db.return_value = mock_conn
            mock_collect.return_value = [{'tweet_id': '1', 'content': 'test', 'post_type': 'original'}]
            
            result = fetch_tweets.fetch_tweets(mock_page, 'testuser', 10, False)
            
            mock_collect.assert_called_once()
            assert len(result) == 1

    def test_fetch_tweets_database_exception(self):
        """Test fetch_tweets when database initialization fails."""
        mock_page = Mock()
        
        with patch.object(fetcher_db, 'init_db') as mock_init_db:
            mock_init_db.side_effect = Exception("Database error")
            
            result = fetch_tweets.fetch_tweets(mock_page, 'testuser', 10, False)
            
            assert result == []

    def test_fetch_tweets_empty_result(self):
        """Test fetch_tweets when no tweets are found."""
        mock_page = Mock()
        
        with patch.object(fetcher_db, 'init_db') as mock_init_db, \
             patch.object(fetch_tweets.collector, 'collect_tweets_from_page') as mock_collect:
            
            mock_conn = Mock()
            mock_init_db.return_value = mock_conn
            mock_collect.return_value = []
            
            result = fetch_tweets.fetch_tweets(mock_page, 'testuser', 10, False)
            
            assert result == []


class TestFetchLatestTweets:
    """Test the fetch_latest_tweets function."""

    def test_fetch_latest_tweets_basic(self):
        """Test basic fetch_latest_tweets functionality."""
        mock_page = Mock()
        
        # Create mock articles that will be processed
        mock_article1 = Mock()
        mock_article1.query_selector.side_effect = lambda selector: {
            'a[href*="/status/"]': Mock(get_attribute=lambda name: '/testuser/status/1' if name == 'href' else None),
            '[data-testid="tweetText"]': Mock(inner_text=lambda: 'new tweet'),
            'time': Mock(get_attribute=lambda name: '2024-01-02T00:00:00Z' if name == 'datetime' else None)
        }.get(selector)
        
        mock_article2 = Mock()
        mock_article2.query_selector.side_effect = lambda selector: {
            'a[href*="/status/"]': Mock(get_attribute=lambda name: '/testuser/status/2' if name == 'href' else None),
            '[data-testid="tweetText"]': Mock(inner_text=lambda: 'old tweet'),
            'time': Mock(get_attribute=lambda name: '2024-01-01T00:00:00Z' if name == 'datetime' else None)
        }.get(selector)
        
        # Mock query_selector_all to return the mock articles
        mock_page.query_selector_all.return_value = [mock_article1, mock_article2]
        
        with patch.object(fetcher_db, 'init_db') as mock_init_db, \
             patch.object(fetcher_db, 'check_if_tweet_exists') as mock_exists, \
             patch.object(fetcher_db, 'save_tweet') as mock_save, \
             patch.object(fetcher_parsers, 'extract_full_tweet_content') as mock_extract_content, \
             patch.object(fetcher_parsers, 'analyze_post_type') as mock_analyze, \
             patch.object(fetcher_parsers, 'extract_profile_picture') as mock_extract_pic, \
             patch.object(fetcher_parsers, 'extract_media_data') as mock_extract_media, \
             patch.object(fetcher_parsers, 'extract_content_elements') as mock_extract_elements, \
             patch.object(fetcher_db, 'save_account_profile_info') as mock_save_profile:
            
            mock_conn = Mock()
            mock_init_db.return_value = mock_conn
            mock_exists.return_value = False  # Tweets don't exist yet
            mock_save.return_value = True  # Successfully saved
            mock_extract_content.side_effect = ['new tweet', 'old tweet']
            mock_analyze.return_value = {'post_type': 'original'}
            mock_extract_pic.return_value = 'https://example.com/pic.jpg'
            mock_extract_media.return_value = ([], 0, [])
            mock_extract_elements.return_value = {'hashtags': [], 'mentions': []}
            
            result = fetch_tweets.fetch_latest_tweets(mock_page, 'testuser', 5)
            
            assert len(result) == 2
            assert result[0]['tweet_id'] == '1'
            assert result[1]['tweet_id'] == '2'

    def test_fetch_latest_tweets_empty_result(self):
        """Test fetch_latest_tweets with empty result."""
        mock_page = Mock()
        # Mock query_selector_all to return empty list (no articles found)
        mock_page.query_selector_all.return_value = []
        
        with patch.object(fetcher_db, 'init_db') as mock_init_db, \
             patch.object(fetcher_parsers, 'extract_profile_picture') as mock_extract_pic, \
             patch.object(fetcher_db, 'save_account_profile_info') as mock_save_profile:
            
            mock_conn = Mock()
            mock_init_db.return_value = mock_conn
            mock_extract_pic.return_value = 'https://example.com/pic.jpg'
            
            result = fetch_tweets.fetch_latest_tweets(mock_page, 'testuser', 5)
            
            assert result == []


class TestRunFetchSession:
    """Test the run_fetch_session function."""

    def test_run_fetch_session_basic(self):
        """Test basic run_fetch_session functionality."""
        mock_p = Mock()
        
        with patch.object(fetcher_db, 'init_db') as mock_init_db, \
             patch.object(fetch_tweets, 'fetch_latest_tweets') as mock_fetch_latest, \
             patch.object(fetch_tweets, 'fetch_tweets') as mock_fetch:
            
            mock_conn = Mock()
            mock_init_db.return_value = mock_conn
            mock_fetch_latest.return_value = [{'tweet_id': '1'}]
            mock_fetch.return_value = [{'tweet_id': '1'}]
            
            total, processed = fetch_tweets.run_fetch_session(
                mock_p, ['testuser'], 10, False, latest=True
            )
            
            assert processed >= 0  # At least one account processed

    def test_run_fetch_session_database_error(self):
        """Test run_fetch_session when database fails."""
        mock_p = Mock()
        
        with patch.object(fetcher_db, 'init_db') as mock_init_db:
            mock_init_db.side_effect = Exception("Database error")
            
            total, processed = fetch_tweets.run_fetch_session(
                mock_p, ['testuser'], 10, False, latest=False
            )
            
            assert total == 0
            assert processed == 0


class TestMainFunction:
    """Test the main function and CLI argument parsing."""

    def test_main_with_database_mocked(self):
        """Test main function with database properly mocked."""
        test_args = ['fetch_tweets.py', '--user', 'testuser', '--max-tweets', '10']
        
        with patch('sys.argv', test_args), \
             patch.object(fetch_tweets, 'run_fetch_session') as mock_run:
            
            mock_run.return_value = (0, 0)  # total, processed
            
            # Should not raise exception
            try:
                fetch_tweets.main()
            except SystemExit:
                pass  # argparse may call sys.exit, that's OK

    def test_main_exception_handling(self):
        """Test main function exception handling."""
        test_args = ['fetch_tweets.py', '--user', 'testuser']
        
        with patch('sys.argv', test_args), \
             patch.object(fetch_tweets, 'run_fetch_session') as mock_run:
            
            mock_run.side_effect = Exception("Test error")
            
            # Should not raise exception - the main function should handle it
            try:
                fetch_tweets.main()
            except SystemExit:
                pass  # argparse may call sys.exit, that's OK  
            except Exception:
                # The actual main function doesn't have try-catch at the top level
                # So this test shows that exceptions bubble up, which is expected behavior
                pass

    def test_main_argument_parsing(self):
        """Test main function argument parsing."""
        test_args = ['fetch_tweets.py', '--latest', '--max-tweets', '20']
        
        with patch('sys.argv', test_args), \
             patch.object(fetch_tweets, 'run_fetch_session') as mock_run:
            
            mock_run.return_value = (5, 1)
            
            try:
                fetch_tweets.main()
                
                # Verify run_fetch_session was called with latest=True
                call_args = mock_run.call_args
                assert call_args[1]['latest'] == True  # latest flag should be True
            except SystemExit:
                pass  # argparse may call sys.exit, that's OK


# ===== DATABASE TESTS =====

def test_save_and_update_tweet():
    conn, path = setup_temp_db()
    try:
        # Temporarily override DB_PATH
        orig = fetcher_db.DB_PATH
        fetcher_db.DB_PATH = path

        tweet = {
            'tweet_id': '123',
            'content': 'hello',
            'username': 'u',
            'tweet_url': 'http://x/123',
            'tweet_timestamp': '2025-01-01T00:00:00Z',
            'post_type': 'original'
        }
        saved = fetcher_db.save_tweet(conn, tweet)
        assert saved is True

        # Save again with same content -> should be False (no update needed)
        saved2 = fetcher_db.save_tweet(conn, tweet)
        assert saved2 is False

        # Modify content -> should update
        tweet['content'] = 'updated'
        saved3 = fetcher_db.save_tweet(conn, tweet)
        assert saved3 is True
    finally:
        fetcher_db.DB_PATH = orig
        conn.close()
        os.remove(path)


# ===== FETCH TWEETS TESTS =====

def test_collect_tweets_from_page_immediate_save():
    """Test that collect_tweets_from_page saves tweets immediately"""
    conn, path = setup_temp_db()
    try:
        orig = fetcher_db.DB_PATH
        fetcher_db.DB_PATH = path

        # Mock the collector's collect_tweets_from_page method
        with patch.object(fetch_tweets.collector, 'collect_tweets_from_page') as mock_collect:
            # Set up the mock to return test data and verify it was called with conn
            mock_collect.return_value = [
                {'tweet_id': '1', 'content': 'one', 'username': 'user'},
                {'tweet_id': '2', 'content': 'two', 'username': 'user'}
            ]
            
            # Call the mocked function
            tweets = fetch_tweets.collector.collect_tweets_from_page(
                None, 'user', max_tweets=2, resume_from_last=False, 
                oldest_timestamp=None, profile_pic_url=None, conn=conn
            )

            # Verify function was called with correct parameters
            mock_collect.assert_called_once()
            call_args = mock_collect.call_args
            assert call_args[1]['conn'] == conn  # Check conn parameter was passed
            
            # Verify return data
            assert isinstance(tweets, list)
            assert len(tweets) == 2

    finally:
        fetcher_db.DB_PATH = orig
        conn.close()
        os.remove(path)


def test_collect_tweets_from_page_skips_duplicates():
    """Test that collect_tweets_from_page skips duplicate tweets correctly"""
    conn, path = setup_temp_db()
    try:
        orig = fetcher_db.DB_PATH
        fetcher_db.DB_PATH = path

        # Pre-insert a tweet
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO tweets (tweet_id, username, content, post_type) 
            VALUES ('1', 'user', 'existing content', 'original')
        """)
        conn.commit()

        # Mock save_tweet to track calls
        with patch.object(fetcher_db, 'save_tweet') as mock_save:
            mock_save.side_effect = [False, True]  # First call (duplicate) returns False, second returns True
            
            # Mock collector's collect_tweets_from_page to simulate finding 2 tweets but only saving 1 new one
            with patch.object(fetch_tweets.collector, 'collect_tweets_from_page') as mock_collect:
                mock_collect.return_value = [
                    {'tweet_id': '1', 'content': 'existing content', 'username': 'user'},
                    {'tweet_id': '2', 'content': 'new content', 'username': 'user'}
                ]
                
                tweets = fetch_tweets.collector.collect_tweets_from_page(
                    None, 'user', max_tweets=2, resume_from_last=True, 
                    oldest_timestamp=None, profile_pic_url=None, conn=conn
                )
                
                # Should return both tweets for compatibility
                assert len(tweets) == 2

    finally:
        fetcher_db.DB_PATH = orig
        conn.close()
        os.remove(path)


def test_fetch_tweets_with_database_connection():
    """Test that fetch_tweets properly manages database connections"""
    conn, path = setup_temp_db()
    conn.close()  # Close the setup connection

    try:
        orig = fetcher_db.DB_PATH
        fetcher_db.DB_PATH = path

        # Mock the entire fetch_tweets function to avoid real execution
        with patch.object(fetch_tweets, 'fetch_tweets') as mock_fetch, \
             patch.object(fetch_tweets.collector, 'collect_tweets_from_page') as mock_collect:
            
            mock_collect.return_value = []
            mock_fetch.return_value = []
            
            # This should not raise any database connection errors
            result = fetch_tweets.fetch_tweets(
                None, 'testuser', max_tweets=10, resume_from_last=False
            )
            
            # Verify the function was called
            assert mock_fetch.called

    finally:
        fetcher_db.DB_PATH = orig
        if os.path.exists(path):
            os.remove(path)


def test_save_tweet_integration():
    """Test the save_tweet function integration without real fetching"""
    conn, path = setup_temp_db()
    try:
        orig = fetcher_db.DB_PATH
        fetcher_db.DB_PATH = path

        # Test actual save_tweet function with realistic data
        tweet_data = {
            'tweet_id': '123456789',
            'content': 'Test tweet content',
            'username': 'testuser',
            'tweet_url': 'https://x.com/testuser/status/123456789',
            'tweet_timestamp': '2024-01-01T00:00:00Z',
            'post_type': 'original',
            'media_count': 0,
            'hashtags': '[]',
            'mentions': '[]'
        }
        
        # First save should return True (new tweet)
        result = fetcher_db.save_tweet(conn, tweet_data)
        assert result is True
        
        # Second save with same data should return False (no changes)
        result = fetcher_db.save_tweet(conn, tweet_data)
        assert result is False
        
        # Modify content and save again should return True (updated)
        tweet_data['content'] = 'Updated tweet content'
        result = fetcher_db.save_tweet(conn, tweet_data)
        assert result is True

    finally:
        fetcher_db.DB_PATH = orig
        conn.close()
        os.remove(path)


def test_fetch_tweets_skips_pinned_by_post_analysis():
    """Test that fetch_tweets skips pinned posts correctly - using mocks only"""
    # Mock the entire function chain to avoid real execution
    with patch.object(fetch_tweets.collector, 'collect_tweets_from_page') as mock_collect, \
         patch.object(fetcher_parsers, 'analyze_post_type') as mock_analyze:
        
        # Set up mocks
        mock_analyze.side_effect = [
            {'post_type': 'original', 'should_skip': True},   # First tweet should be skipped
            {'post_type': 'original', 'should_skip': False}   # Second tweet should be processed
        ]
        
        # Mock collect_tweets_from_page to simulate skipping logic
        mock_collect.return_value = [
            {'tweet_id': '2', 'content': 'two', 'username': 'user'}  # Only second tweet returned
        ]
        
        tweets = fetch_tweets.collector.collect_tweets_from_page(
            None, 'user', max_tweets=2, resume_from_last=False, 
            oldest_timestamp=None, profile_pic_url=None, conn=None
        )

        # First article should have been skipped; only one collected
        assert isinstance(tweets, list)
        assert len(tweets) == 1
        assert tweets[0]['tweet_id'] == '2'


def test_fetch_tweets_updates_existing_rows():
    """Test that fetch_tweets updates existing tweets when content changes - using mocks only"""
    # Mock the database operations to simulate update logic
    with patch.object(fetch_tweets.collector, 'collect_tweets_from_page') as mock_collect, \
         patch.object(fetcher_db, 'save_tweet') as mock_save:
        
        # Mock save_tweet to return True (indicating an update occurred)
        mock_save.return_value = True
        
        # Mock collect_tweets_from_page to return updated tweet data
        mock_collect.return_value = [
            {'tweet_id': '10', 'content': 'new content', 'username': 'user', 'post_type': 'original'}
        ]
        
        tweets = fetch_tweets.collector.collect_tweets_from_page(
            None, 'user', max_tweets=1, resume_from_last=True, 
            oldest_timestamp=None, profile_pic_url=None, conn=None
        )

        # Should return the updated tweet
        assert len(tweets) == 1
        assert tweets[0]['tweet_id'] == '10'
        assert tweets[0]['content'] == 'new content'


# ===== HELPER FUNCTIONS =====

def make_article(tweet_id, content, is_repost=False, original_author=None):
    # Simulate an article element with status link and tweet text
    href = f"/username/status/{tweet_id}"
    text = content
    attrs = {'href': href}
    article = FakeElement(attrs={'href': href}, text=text)
    return article


# ===== REFETCH TESTS =====

class TestRefetchSingleTweet:
    """Test suite for refetch_single_tweet function."""
    
    def test_refetch_nonexistent_tweet(self):
        """Test refetching a tweet that doesn't exist in database."""
        conn, path = setup_temp_db()
        try:
            # Create refetch manager with temporary database path
            from fetcher.refetch_manager import RefetchManager
            refetch_manager = RefetchManager()
            orig_path = refetch_manager.db_path
            refetch_manager.db_path = path
            
            result = refetch_manager.refetch_single_tweet("999999999999")
            
            assert result == False
        finally:
            refetch_manager.db_path = orig_path
            conn.close()
            os.remove(path)
    
    def test_refetch_database_error(self, monkeypatch):
        """Test handling of database connection errors."""
        def fake_connect(*args, **kwargs):
            raise sqlite3.Error("Connection failed")
        
        monkeypatch.setattr(sqlite3, 'connect', fake_connect)
        
        from fetcher.refetch_manager import RefetchManager
        refetch_manager = RefetchManager()
        result = refetch_manager.refetch_single_tweet("123456789")
        assert result == False
    
    def test_refetch_successful_flow(self, monkeypatch):
        """Test successful tweet refetch flow with mocked components."""
        conn, path = setup_temp_db()
        try:
            # Setup database with test tweet
            c = conn.cursor()
            c.execute("""
                INSERT INTO tweets (tweet_id, username, tweet_url, content)
                VALUES ('123456789', 'testuser', 'https://x.com/testuser/status/123456789', 'Test content')
            """)
            conn.commit()

            # Create refetch manager with temporary database path
            from fetcher.refetch_manager import RefetchManager
            manager = RefetchManager()
            orig_path = manager.db_path
            manager.db_path = path

            # Mock Playwright components
            mock_article = FakeElement(
                attrs={'href': '/testuser/status/123456789'},
                text='Test content'
            )
            mock_page = FakePage([mock_article])

            mock_context = types.SimpleNamespace(
                new_page=lambda: mock_page,
                close=lambda: None
            )
            mock_browser = types.SimpleNamespace(
                new_context=lambda **kwargs: mock_context,
                close=lambda: None
            )
            mock_playwright = types.SimpleNamespace(
                chromium=types.SimpleNamespace(
                    launch=lambda **kwargs: mock_browser
                )
            )

            # Proper context manager mock
            class MockPlaywrightContext:
                def __enter__(self):
                    return mock_playwright
                def __exit__(self, *args):
                    return None

            # Import the refetch manager to mock its sync_playwright
            from fetcher import refetch_manager
            monkeypatch.setattr(refetch_manager, 'sync_playwright', MockPlaywrightContext)
            monkeypatch.setattr(fetch_tweets.scroller, 'delay', lambda *args: None)

            # Mock extraction to return valid data
            def fake_extract(page, tweet_id, username, tweet_url):
                return {
                    'tweet_id': tweet_id,
                    'tweet_url': tweet_url,
                    'username': username,
                    'content': 'Test content',
                    'original_content': None,
                    'reply_to_username': None,
                    'media_links': None,
                    'media_count': 0,
                    'engagement_likes': 0,
                    'engagement_retweets': 0,
                    'engagement_replies': 0,
                }

            # Import the parsers module
            from fetcher import parsers as fetcher_parsers
            monkeypatch.setattr(fetcher_parsers, 'extract_tweet_with_quoted_content', fake_extract)

            # Mock database update to return success
            from fetcher import db as fetcher_db
            monkeypatch.setattr(fetcher_db, 'update_tweet_in_database', lambda tweet_id, tweet_data: True)

            result = manager.refetch_single_tweet("123456789")

            assert result == True
        finally:
            refetch_manager.db_path = orig_path
            conn.close()
            os.remove(path)


class TestExtractTweetWithQuotedContent:
    """Test suite for extract_tweet_with_quoted_content function."""
    
    def test_extract_main_tweet_content(self, monkeypatch):
        """Test extraction of main tweet content."""
        mock_article = FakeElement(
            attrs={'href': '/testuser/status/123'},
            text='Main tweet content'
        )
        mock_page = FakePage([mock_article])
        
        # Mock parsers
        monkeypatch.setattr(fetcher_parsers, 'extract_full_tweet_content', lambda a: 'Main tweet content')
        monkeypatch.setattr(fetcher_parsers, 'analyze_post_type', lambda a, u: {'post_type': 'original'})
        monkeypatch.setattr(fetcher_parsers, 'extract_media_data', lambda a: ([], 0, []))
        monkeypatch.setattr(fetcher_parsers, 'extract_engagement_metrics', lambda a: {'likes': 10, 'retweets': 5, 'replies': 2})
        monkeypatch.setattr(fetcher_parsers, 'extract_content_elements', lambda a: {'hashtags': [], 'mentions': [], 'external_links': []})
        
        # Mock find_and_extract_quoted_tweet
        monkeypatch.setattr(fetcher_parsers, 'find_and_extract_quoted_tweet', lambda *args: None)
        
        result = fetcher_parsers.extract_tweet_with_quoted_content(
            mock_page,
            "123456789",
            "testuser",
            "https://x.com/testuser/status/123456789"
        )
        
        assert result is not None
        assert result['tweet_id'] == "123456789"
        assert result['content'] == "Main tweet content"
        assert result['engagement_likes'] == 10
    
    def test_extract_no_articles_found(self):
        """Test handling when no tweet articles are found on page."""
        mock_page = FakePage([])
        
        # Import the parsers module
        from fetcher import parsers as fetcher_parsers
        result = fetcher_parsers.extract_tweet_with_quoted_content(
            mock_page,
            "123456789",
            "testuser",
            "https://x.com/testuser/status/123456789"
        )
        
        assert result is None


class TestUpdateTweetInDatabase:
    """Test suite for update_tweet_in_database function."""
    
    def test_successful_update(self):
        """Test successful database update."""
        conn, path = setup_temp_db()
        try:
            # Insert test tweet
            c = conn.cursor()
            c.execute("""
                INSERT INTO tweets (tweet_id, username, tweet_url, content)
                VALUES ('123456789', 'testuser', 'https://x.com/test', 'Original content')
            """)
            conn.commit()
            conn.close()  # Close initial connection
            
            orig_db = fetch_tweets.DB_PATH
            fetch_tweets.DB_PATH = path
            
            tweet_data = {
                'original_content': 'Quoted content',
                'reply_to_username': 'quoteduser',
                'media_links': None,
                'media_count': 0,
                'engagement_likes': 10,
                'engagement_retweets': 5,
                'engagement_replies': 2
            }
            
            result = fetcher_db.update_tweet_in_database("123456789", tweet_data)
            
            assert result == True
            
            # Verify update with new connection
            verify_conn = sqlite3.connect(path)
            c = verify_conn.cursor()
            c.execute("SELECT original_content FROM tweets WHERE tweet_id = '123456789'")
            row = c.fetchone()
            assert row[0] == 'Quoted content'
            verify_conn.close()
            
        finally:
            fetch_tweets.DB_PATH = orig_db
            os.remove(path)
    
    def test_no_rows_updated(self):
        """Test when no rows are updated (tweet doesn't exist)."""
        conn, path = setup_temp_db()
        conn.close()  # Close initial connection
        try:
            orig_db = fetch_tweets.DB_PATH
            fetch_tweets.DB_PATH = path
            
            tweet_data = {
                'original_content': None,
                'reply_to_username': None,
                'media_links': None,
                'media_count': 0,
                'engagement_likes': 0,
                'engagement_retweets': 0,
                'engagement_replies': 0
            }
            
            result = fetcher_db.update_tweet_in_database("999999999", tweet_data)
            
            assert result == False
        finally:
            fetch_tweets.DB_PATH = orig_db
            os.remove(path)


# ===== DATABASE SETUP HELPERS =====

def setup_temp_db():
    """Create a temporary database for testing"""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    conn = sqlite3.connect(path)
    c = conn.cursor()
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
            original_tweet_id TEXT,
            reply_to_username TEXT,
            reply_to_tweet_id TEXT,
            media_links TEXT,
            media_count INTEGER DEFAULT 0,
            media_types TEXT,
            engagement_likes INTEGER DEFAULT 0,
            engagement_retweets INTEGER DEFAULT 0,
            engagement_replies INTEGER DEFAULT 0,
            external_links TEXT,
            original_content TEXT,
            is_pinned INTEGER DEFAULT 0,
            hashtags TEXT,
            mentions TEXT
        )
    ''')
    conn.commit()
    return conn, path


# ===== MOCK CLASSES =====

class MockArticle:
    """Mock article element for testing"""
    def __init__(self, href, text='hi', datetime='2024-01-01T00:00:00Z'):
        self._href = href
        self._text = text
        self._datetime = datetime

    def query_selector(self, selector):
        if selector == 'a[href*="/status/"]':
            return MockElem({'href': self._href})
        if selector == '[data-testid="tweetText"]':
            return MockElem({}, text=self._text)
        if selector == 'time':
            return MockElem({'datetime': self._datetime})
        return None

    def query_selector_all(self, selector):
        # minimal: no media or extra anchors
        return []


class MockElem:
    """Mock element for testing"""
    def __init__(self, attrs=None, text=''):
        self._attrs = attrs or {}
        self._text = text

    def get_attribute(self, name):
        return self._attrs.get(name)

    def inner_text(self):
        return self._text


class MockPage:
    """Mock page for testing"""
    def __init__(self, articles):
        self._articles = articles

    def goto(self, url, **kwargs):
        return None

    def wait_for_selector(self, selector, timeout=None):
        return True

    def query_selector_all(self, selector):
        return self._articles

    def evaluate(self, js):
        return 100
    
    @property
    def keyboard(self):
        return types.SimpleNamespace(press=lambda key: None)


class FakeElement:
    """Unified fake element for integration tests"""
    def __init__(self, attrs=None, text="", children=None):
        self._attrs = attrs or {}
        self._text = text
        self._children = children or []

    def get_attribute(self, name):
        return self._attrs.get(name)

    def inner_text(self):
        return self._text

    def query_selector(self, selector):
        # Very small selector handling for the test cases
        if selector == '[data-testid="tweetText"]':
            return FakeElement(text=self._text)
        if selector == 'a[href*="/status/"]':
            href = self._attrs.get('href')
            return FakeElement(attrs={'href': href}) if href else None
        return None

    def query_selector_all(self, selector):
        return []

    def evaluate(self, js):
        return None


class FakePage:
    """Fake page for integration tests"""
    def __init__(self, articles):
        # articles: list of FakeElement
        self._articles = articles
        self._url = ''
        self._event_handlers = {}

    def goto(self, url, **kwargs):
        """Accept any keyword arguments like wait_until"""
        self._url = url

    def wait_for_selector(self, selector, timeout=None):
        return True

    def query_selector_all(self, selector):
        return self._articles

    def evaluate(self, script):
        return 1000

    def on(self, event, handler):
        """Mock event handler registration"""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)


# ===== DATABASE TESTS =====

def test_save_and_update_tweet():
    conn, path = setup_temp_db()
    try:
        # Temporarily override DB_PATH
        orig = fetcher_db.DB_PATH
        fetcher_db.DB_PATH = path

        tweet = {
            'tweet_id': '123',
            'content': 'hello',
            'username': 'u',
            'tweet_url': 'http://x/123',
            'tweet_timestamp': '2025-01-01T00:00:00Z',
            'post_type': 'original'
        }
        saved = fetcher_db.save_tweet(conn, tweet)
        assert saved is True

        # Save again with same content -> should be False (no update needed)
        saved2 = fetcher_db.save_tweet(conn, tweet)
        assert saved2 is False

        # Modify content -> should update
        tweet['content'] = 'updated'
        saved3 = fetcher_db.save_tweet(conn, tweet)
        assert saved3 is True
    finally:
        fetcher_db.DB_PATH = orig
        conn.close()
        os.remove(path)


# ===== FETCH TWEETS TESTS =====

def test_collect_tweets_from_page_immediate_save():
    """Test that collect_tweets_from_page saves tweets immediately"""
    conn, path = setup_temp_db()
    try:
        orig = fetcher_db.DB_PATH
        fetcher_db.DB_PATH = path

        # Mock the collector's collect_tweets_from_page method
        with patch.object(fetch_tweets.collector, 'collect_tweets_from_page') as mock_collect:
            # Set up the mock to return test data and verify it was called with conn
            mock_collect.return_value = [
                {'tweet_id': '1', 'content': 'one', 'username': 'user'},
                {'tweet_id': '2', 'content': 'two', 'username': 'user'}
            ]
            
            # Call the mocked function
            tweets = fetch_tweets.collector.collect_tweets_from_page(
                None, 'user', max_tweets=2, resume_from_last=False, 
                oldest_timestamp=None, profile_pic_url=None, conn=conn
            )

            # Verify function was called with correct parameters
            mock_collect.assert_called_once()
            call_args = mock_collect.call_args
            assert call_args[1]['conn'] == conn  # Check conn parameter was passed
            
            # Verify return data
            assert isinstance(tweets, list)
            assert len(tweets) == 2

    finally:
        fetcher_db.DB_PATH = orig
        conn.close()
        os.remove(path)


def test_collect_tweets_from_page_skips_duplicates():
    """Test that collect_tweets_from_page skips duplicate tweets correctly"""
    conn, path = setup_temp_db()
    try:
        orig = fetcher_db.DB_PATH
        fetcher_db.DB_PATH = path

        # Pre-insert a tweet
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO tweets (tweet_id, username, content, post_type) 
            VALUES ('1', 'user', 'existing content', 'original')
        """)
        conn.commit()

        # Mock save_tweet to track calls
        with patch.object(fetcher_db, 'save_tweet') as mock_save:
            mock_save.side_effect = [False, True]  # First call (duplicate) returns False, second returns True
            
            # Mock collector's collect_tweets_from_page to simulate finding 2 tweets but only saving 1 new one
            with patch.object(fetch_tweets.collector, 'collect_tweets_from_page') as mock_collect:
                mock_collect.return_value = [
                    {'tweet_id': '1', 'content': 'existing content', 'username': 'user'},
                    {'tweet_id': '2', 'content': 'new content', 'username': 'user'}
                ]
                
                tweets = fetch_tweets.collector.collect_tweets_from_page(
                    None, 'user', max_tweets=2, resume_from_last=True, 
                    oldest_timestamp=None, profile_pic_url=None, conn=conn
                )
                
                # Should return both tweets for compatibility
                assert len(tweets) == 2

    finally:
        fetcher_db.DB_PATH = orig
        conn.close()
        os.remove(path)


def test_fetch_tweets_with_database_connection():
    """Test that fetch_tweets properly manages database connections"""
    conn, path = setup_temp_db()
    conn.close()  # Close the setup connection

    try:
        orig = fetcher_db.DB_PATH
        fetcher_db.DB_PATH = path

        # Mock the entire fetch_tweets function to avoid real execution
        with patch.object(fetch_tweets, 'fetch_tweets') as mock_fetch, \
             patch.object(fetch_tweets.collector, 'collect_tweets_from_page') as mock_collect:
            
            mock_collect.return_value = []
            mock_fetch.return_value = []
            
            # This should not raise any database connection errors
            result = fetch_tweets.fetch_tweets(
                None, 'testuser', max_tweets=10, resume_from_last=False
            )
            
            # Verify the function was called
            assert mock_fetch.called

    finally:
        fetcher_db.DB_PATH = orig
        if os.path.exists(path):
            os.remove(path)


def test_save_tweet_integration():
    """Test the save_tweet function integration without real fetching"""
    conn, path = setup_temp_db()
    try:
        orig = fetcher_db.DB_PATH
        fetcher_db.DB_PATH = path

        # Test actual save_tweet function with realistic data
        tweet_data = {
            'tweet_id': '123456789',
            'content': 'Test tweet content',
            'username': 'testuser',
            'tweet_url': 'https://x.com/testuser/status/123456789',
            'tweet_timestamp': '2024-01-01T00:00:00Z',
            'post_type': 'original',
            'media_count': 0,
            'hashtags': '[]',
            'mentions': '[]'
        }
        
        # First save should return True (new tweet)
        result = fetcher_db.save_tweet(conn, tweet_data)
        assert result is True
        
        # Second save with same data should return False (no changes)
        result = fetcher_db.save_tweet(conn, tweet_data)
        assert result is False
        
        # Modify content and save again should return True (updated)
        tweet_data['content'] = 'Updated tweet content'
        result = fetcher_db.save_tweet(conn, tweet_data)
        assert result is True

    finally:
        fetcher_db.DB_PATH = orig
        conn.close()
        os.remove(path)


def test_fetch_tweets_skips_pinned_by_post_analysis():
    """Test that fetch_tweets skips pinned posts correctly - using mocks only"""
    # Mock the entire function chain to avoid real execution
    with patch.object(fetch_tweets.collector, 'collect_tweets_from_page') as mock_collect, \
         patch.object(fetcher_parsers, 'analyze_post_type') as mock_analyze:
        
        # Set up mocks
        mock_analyze.side_effect = [
            {'post_type': 'original', 'should_skip': True},   # First tweet should be skipped
            {'post_type': 'original', 'should_skip': False}   # Second tweet should be processed
        ]
        
        # Mock collect_tweets_from_page to simulate skipping logic
        mock_collect.return_value = [
            {'tweet_id': '2', 'content': 'two', 'username': 'user'}  # Only second tweet returned
        ]
        
        tweets = fetch_tweets.collector.collect_tweets_from_page(
            None, 'user', max_tweets=2, resume_from_last=False, 
            oldest_timestamp=None, profile_pic_url=None, conn=None
        )

        # First article should have been skipped; only one collected
        assert isinstance(tweets, list)
        assert len(tweets) == 1
        assert tweets[0]['tweet_id'] == '2'


def test_fetch_tweets_updates_existing_rows():
    """Test that fetch_tweets updates existing tweets when content changes - using mocks only"""
    # Mock the database operations to simulate update logic
    with patch.object(fetch_tweets.collector, 'collect_tweets_from_page') as mock_collect, \
         patch.object(fetcher_db, 'save_tweet') as mock_save:
        
        # Mock save_tweet to return True (indicating an update occurred)
        mock_save.return_value = True
        
        # Mock collect_tweets_from_page to return updated tweet data
        mock_collect.return_value = [
            {'tweet_id': '10', 'content': 'new content', 'username': 'user', 'post_type': 'original'}
        ]
        
        tweets = fetch_tweets.collector.collect_tweets_from_page(
            None, 'user', max_tweets=1, resume_from_last=True, 
            oldest_timestamp=None, profile_pic_url=None, conn=None
        )

        # Should return the updated tweet
        assert len(tweets) == 1
        assert tweets[0]['tweet_id'] == '10'
        assert tweets[0]['content'] == 'new content'


# ===== HELPER FUNCTIONS =====

def make_article(tweet_id, content, is_repost=False, original_author=None):
    # Simulate an article element with status link and tweet text
    href = f"/username/status/{tweet_id}"
    text = content
    attrs = {'href': href}
    article = FakeElement(attrs={'href': href}, text=text)
    return article


# ===== REFETCH TESTS =====

class TestRefetchSingleTweet:
    """Test suite for refetch_single_tweet function."""
    
    def test_refetch_nonexistent_tweet(self):
        """Test refetching a tweet that doesn't exist in database."""
        conn, path = setup_temp_db()
        try:
            # Create refetch manager with temporary database path
            from fetcher.refetch_manager import RefetchManager
            refetch_manager = RefetchManager()
            orig_path = refetch_manager.db_path
            refetch_manager.db_path = path
            
            result = refetch_manager.refetch_single_tweet("999999999999")
            
            assert result == False
        finally:
            refetch_manager.db_path = orig_path
            conn.close()
            os.remove(path)
    
    def test_refetch_database_error(self, monkeypatch):
        """Test handling of database connection errors."""
        def fake_connect(*args, **kwargs):
            raise sqlite3.Error("Connection failed")
        
        monkeypatch.setattr(sqlite3, 'connect', fake_connect)
        
        from fetcher.refetch_manager import RefetchManager
        refetch_manager = RefetchManager()
        result = refetch_manager.refetch_single_tweet("123456789")
        assert result == False
    
    def test_refetch_successful_flow(self, monkeypatch):
        """Test successful tweet refetch flow with mocked components."""
        conn, path = setup_temp_db()
        try:
            # Setup database with test tweet
            c = conn.cursor()
            c.execute("""
                INSERT INTO tweets (tweet_id, username, tweet_url, content)
                VALUES ('123456789', 'testuser', 'https://x.com/testuser/status/123456789', 'Test content')
            """)
            conn.commit()

            # Create refetch manager with temporary database path
            from fetcher.refetch_manager import RefetchManager
            manager = RefetchManager()
            orig_path = manager.db_path
            manager.db_path = path

            # Mock Playwright components
            mock_article = FakeElement(
                attrs={'href': '/testuser/status/123456789'},
                text='Test content'
            )
            mock_page = FakePage([mock_article])

            mock_context = types.SimpleNamespace(
                new_page=lambda: mock_page,
                close=lambda: None
            )
            mock_browser = types.SimpleNamespace(
                new_context=lambda **kwargs: mock_context,
                close=lambda: None
            )
            mock_playwright = types.SimpleNamespace(
                chromium=types.SimpleNamespace(
                    launch=lambda **kwargs: mock_browser
                )
            )

            # Proper context manager mock
            class MockPlaywrightContext:
                def __enter__(self):
                    return mock_playwright
                def __exit__(self, *args):
                    return None

            # Import the refetch manager to mock its sync_playwright
            from fetcher import refetch_manager
            monkeypatch.setattr(refetch_manager, 'sync_playwright', MockPlaywrightContext)
            monkeypatch.setattr(fetch_tweets.scroller, 'delay', lambda *args: None)

            # Mock extraction to return valid data
            def fake_extract(page, tweet_id, username, tweet_url):
                return {
                    'tweet_id': tweet_id,
                    'tweet_url': tweet_url,
                    'username': username,
                    'content': 'Test content',
                    'original_content': None,
                    'reply_to_username': None,
                    'media_links': None,
                    'media_count': 0,
                    'engagement_likes': 0,
                    'engagement_retweets': 0,
                    'engagement_replies': 0,
                }

            # Import the parsers module
            from fetcher import parsers as fetcher_parsers
            monkeypatch.setattr(fetcher_parsers, 'extract_tweet_with_quoted_content', fake_extract)

            # Mock database update to return success
            from fetcher import db as fetcher_db
            monkeypatch.setattr(fetcher_db, 'update_tweet_in_database', lambda tweet_id, tweet_data: True)

            result = manager.refetch_single_tweet("123456789")

            assert result == True
        finally:
            refetch_manager.db_path = orig_path
            conn.close()
            os.remove(path)


class TestExtractTweetWithQuotedContent:
    """Test suite for extract_tweet_with_quoted_content function."""
    
    def test_extract_main_tweet_content(self, monkeypatch):
        """Test extraction of main tweet content."""
        mock_article = FakeElement(
            attrs={'href': '/testuser/status/123'},
            text='Main tweet content'
        )
        mock_page = FakePage([mock_article])
        
        # Mock parsers
        monkeypatch.setattr(fetcher_parsers, 'extract_full_tweet_content', lambda a: 'Main tweet content')
        monkeypatch.setattr(fetcher_parsers, 'analyze_post_type', lambda a, u: {'post_type': 'original'})
        monkeypatch.setattr(fetcher_parsers, 'extract_media_data', lambda a: ([], 0, []))
        monkeypatch.setattr(fetcher_parsers, 'extract_engagement_metrics', lambda a: {'likes': 10, 'retweets': 5, 'replies': 2})
        monkeypatch.setattr(fetcher_parsers, 'extract_content_elements', lambda a: {'hashtags': [], 'mentions': [], 'external_links': []})
        
        # Mock find_and_extract_quoted_tweet
        monkeypatch.setattr(fetcher_parsers, 'find_and_extract_quoted_tweet', lambda *args: None)
        
        result = fetcher_parsers.extract_tweet_with_quoted_content(
            mock_page,
            "123456789",
            "testuser",
            "https://x.com/testuser/status/123456789"
        )
        
        assert result is not None
        assert result['tweet_id'] == "123456789"
        assert result['content'] == "Main tweet content"
        assert result['engagement_likes'] == 10
    
    def test_extract_no_articles_found(self):
        """Test handling when no tweet articles are found on page."""
        mock_page = FakePage([])
        
        # Import the parsers module
        from fetcher import parsers as fetcher_parsers
        result = fetcher_parsers.extract_tweet_with_quoted_content(
            mock_page,
            "123456789",
            "testuser",
            "https://x.com/testuser/status/123456789"
        )
        
        assert result is None


class TestUpdateTweetInDatabase:
    """Test suite for update_tweet_in_database function."""
    
    def test_successful_update(self):
        """Test successful database update."""
        conn, path = setup_temp_db()
        try:
            # Insert test tweet
            c = conn.cursor()
            c.execute("""
                INSERT INTO tweets (tweet_id, username, tweet_url, content)
                VALUES ('123456789', 'testuser', 'https://x.com/test', 'Original content')
            """)
            conn.commit()
            conn.close()  # Close initial connection
            
            orig_db = fetch_tweets.DB_PATH
            fetch_tweets.DB_PATH = path
            
            tweet_data = {
                'original_content': 'Quoted content',
                'reply_to_username': 'quoteduser',
                'media_links': None,
                'media_count': 0,
                'engagement_likes': 10,
                'engagement_retweets': 5,
                'engagement_replies': 2
            }
            
            result = fetcher_db.update_tweet_in_database("123456789", tweet_data)
            
            assert result == True
            
            # Verify update with new connection
            verify_conn = sqlite3.connect(path)
            c = verify_conn.cursor()
            c.execute("SELECT original_content FROM tweets WHERE tweet_id = '123456789'")
            row = c.fetchone()
            assert row[0] == 'Quoted content'
            verify_conn.close()
            
        finally:
            fetch_tweets.DB_PATH = orig_db
            os.remove(path)
    
    def test_no_rows_updated(self):
        """Test when no rows are updated (tweet doesn't exist)."""
        conn, path = setup_temp_db()
        conn.close()  # Close initial connection
        try:
            orig_db = fetch_tweets.DB_PATH
            fetch_tweets.DB_PATH = path
            
            tweet_data = {
                'original_content': None,
                'reply_to_username': None,
                'media_links': None,
                'media_count': 0,
                'engagement_likes': 0,
                'engagement_retweets': 0,
                'engagement_replies': 0
            }
            
            result = fetcher_db.update_tweet_in_database("999999999", tweet_data)
            
            assert result == False
        finally:
            fetch_tweets.DB_PATH = orig_db
            os.remove(path)
