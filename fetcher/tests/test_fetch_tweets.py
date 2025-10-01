import os
import json
import types
import pytest
import sqlite3
import tempfile
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
            original_tweet_id TEXT
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

    def goto(self, url):
        return None

    def wait_for_selector(self, selector, timeout=None):
        return True

    def query_selector_all(self, selector):
        return self._articles

    def evaluate(self, js):
        return 100


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

    def goto(self, url):
        self._url = url

    def wait_for_selector(self, selector, timeout=None):
        return True

    def query_selector_all(self, selector):
        return self._articles

    def evaluate(self, script):
        return 1000


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

def test_fetch_enhanced_tweets_collects_articles():
    a1 = MockArticle('/user/status/1', text='one', datetime='2023-01-01T00:00:00Z')
    a2 = MockArticle('/user/status/2', text='two', datetime='2022-01-01T00:00:00Z')
    page = MockPage([a1, a2])
    tweets = fetch_tweets.fetch_enhanced_tweets(page, 'user', max_tweets=2, resume_from_last=False)
    assert isinstance(tweets, list)
    assert len(tweets) == 2
    ids = {t['tweet_id'] for t in tweets}
    assert '1' in ids and '2' in ids


def test_fetch_enhanced_tweets_skips_pinned_by_post_analysis(monkeypatch):
    # Make analyze_post_type return should_skip for first article
    a1 = MockArticle('/user/status/1', text='one', datetime='2023-01-01T00:00:00Z')
    a2 = MockArticle('/user/status/2', text='two', datetime='2022-01-01T00:00:00Z')
    page = MockPage([a1, a2])

    def fake_analyze(article, username):
        if article._href.endswith('/1'):
            return {'post_type': 'original', 'should_skip': True}
        return {'post_type': 'original', 'should_skip': False}

    monkeypatch.setattr(fetcher_parsers, 'analyze_post_type', fake_analyze)

    tweets = fetch_tweets.fetch_enhanced_tweets(page, 'user', max_tweets=2, resume_from_last=False)
    # first article should have been skipped; only one collected
    assert isinstance(tweets, list)
    assert len(tweets) == 1


def test_fetch_enhanced_tweets_updates_existing_rows(monkeypatch):
    # Simulate existing tweet in DB and ensure update path queues the tweet_data
    a1 = MockArticle('/user/status/10', text='ten', datetime='2023-01-02T00:00:00Z')
    page = MockPage([a1])

    # Force check_if_tweet_exists to True
    from fetcher import db as fetcher_db
    monkeypatch.setattr(fetcher_db, 'check_if_tweet_exists', lambda u, tid: True)

    # Fake sqlite3.connect to return a cursor that yields a different post_type so needs_update becomes True
    class FakeCursor:
        def execute(self, *args, **kwargs):
            return None
        def fetchone(self):
            # db_post_type, db_content, db_original_author, db_original_tweet_id
            return ('different_type', 'old content', None, None)

    class FakeConn:
        def cursor(self):
            return FakeCursor()
        def close(self):
            pass

    orig_connect = sqlite3.connect
    sqlite3.connect = lambda *a, **k: FakeConn()

    # Ensure analyze_post_type yields 'original' so it's different from db_post_type
    monkeypatch.setattr(fetcher_parsers, 'analyze_post_type', lambda a, u: {'post_type': 'original', 'should_skip': False})

    try:
        tweets = fetch_tweets.fetch_enhanced_tweets(page, 'user', max_tweets=1, resume_from_last=True)
        assert isinstance(tweets, list)
        # since check_if_tweet_exists True and db row differs, the tweet should be queued for update
        assert len(tweets) == 1
        assert tweets[0]['tweet_id'] == '10'
    finally:
        sqlite3.connect = orig_connect


def test_fetch_enhanced_tweets_basic():
    # Create two fake articles: one original, one self-repost (which should be skipped)
    a1 = make_article('t1', 'Hello world')
    a2 = make_article('t2', 'Reposted my own content')

    page = FakePage([a1, a2])

    # Set max_tweets to 2 (the number of fake articles) so the fetch loop
    # finishes immediately and doesn't enter long scrolling loops during tests.
    results = fetch_tweets.fetch_enhanced_tweets(page, 'someuser', max_tweets=2, resume_from_last=False)

    # We expect both to be returned for this fake page since our FakeElement doesn't mark reposts
    assert isinstance(results, list)
    assert len(results) == 2
    ids = {r['tweet_id'] for r in results}
    assert 't1' in ids and 't2' in ids

# ===== HELPER FUNCTIONS =====

def make_article(tweet_id, content, is_repost=False, original_author=None):
    # Simulate an article element with status link and tweet text
    href = f"/username/status/{tweet_id}"
    text = content
    attrs = {'href': href}
    article = FakeElement(attrs={'href': href}, text=text)
    return article