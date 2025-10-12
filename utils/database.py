"""
Database configuration and connection management.
Provides environment-isolated database connections and configuration.
"""

import sqlite3
import os
import threading
from contextlib import contextmanager
from typing import Optional, Dict, Any
from utils import paths

# Thread-local storage for database connections
_local = threading.local()

class DatabaseConfig:
    """Database configuration for different environments."""

    def __init__(self, env: str = None):
        self.env = env or paths.get_environment()
        self.db_path = paths.get_db_path(env=self.env)

        # Environment-specific settings
        self.settings = {
            'development': {
                'timeout': 30.0,
                'isolation_level': None,  # Autocommit mode for development
                'check_same_thread': False,
                'enable_foreign_keys': True,
            },
            'testing': {
                'timeout': 10.0,
                'isolation_level': None,
                'check_same_thread': False,
                'enable_foreign_keys': True,
            },
            'production': {
                'timeout': 60.0,
                'isolation_level': None,
                'check_same_thread': True,
                'enable_foreign_keys': True,
            }
        }

    def get_connection_params(self) -> Dict[str, Any]:
        """Get connection parameters for the current environment."""
        return self.settings.get(self.env, self.settings['development'])

    def get_db_path(self) -> str:
        """Get the database path for the current environment."""
        return self.db_path

def get_db_connection(env: str = None, row_factory: bool = True) -> sqlite3.Connection:
    """
    Get a database connection for the specified environment.

    Args:
        env: Environment name ('development', 'testing', 'production')
        row_factory: Whether to enable row factory for dict-like access

    Returns:
        SQLite database connection
    """
    if env is None:
        env = paths.get_environment()

    config = DatabaseConfig(env)
    params = config.get_connection_params()
    db_path = config.get_db_path()

    # Create connection
    conn = sqlite3.connect(db_path, **params)

    if row_factory:
        conn.row_factory = sqlite3.Row

    # Enable foreign keys
    if params.get('enable_foreign_keys', True):
        conn.execute("PRAGMA foreign_keys = ON")

    # Environment-specific optimizations
    if env == 'production':
        # Production optimizations
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = -64000")  # 64MB cache
    elif env == 'testing':
        # Testing optimizations - faster, less durable
        conn.execute("PRAGMA journal_mode = MEMORY")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA cache_size = -1000")  # 1MB cache

    return conn

@contextmanager
def get_db_connection_context(env: str = None, row_factory: bool = True):
    """
    Context manager for database connections.

    Automatically handles connection cleanup and error handling.
    """
    conn = None
    try:
        conn = get_db_connection(env, row_factory)
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()

def cleanup_test_databases():
    """Clean up test databases (useful for testing teardown)."""
    import glob
    import tempfile

    temp_dir = tempfile.gettempdir()
    test_pattern = f"{temp_dir}/dimetuverdad_test_*.db"

    for db_file in glob.glob(test_pattern):
        try:
            os.remove(db_file)
        except OSError:
            pass  # Ignore if file doesn't exist or can't be removed

def init_test_database(fixtures: bool = True) -> str:
    """
    Initialize a test database with schema and optional fixtures.

    Returns:
        Path to the test database
    """
    # Get test database path
    test_db_path = paths.get_db_path(env='testing')

    # Ensure test database is clean
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    # Create fresh schema directly
    _create_test_database_schema(test_db_path)

    # Load fixtures if requested
    if fixtures:
        _load_test_fixtures(test_db_path)

    return test_db_path

def _create_test_database_schema(db_path: str):
    """Create test database schema directly."""
    print(f"🏗️  Creating test database schema at {db_path}...")

    # Create connection directly to the test database
    conn = sqlite3.connect(db_path, timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    try:
        # Core accounts table (multi-platform support)
        print("  📝 Creating accounts table...")
        c.execute('''
            CREATE TABLE accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                platform TEXT DEFAULT 'twitter',  -- Multi-platform support
                profile_pic_url TEXT,
                profile_pic_updated TIMESTAMP,
                last_scraped TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Core tweets table (simplified)
        print("  📝 Creating tweets table...")
        c.execute('''
            CREATE TABLE tweets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tweet_id TEXT UNIQUE NOT NULL,
                tweet_url TEXT NOT NULL,
                username TEXT NOT NULL,
                content TEXT NOT NULL,

                -- Post classification (simplified)
                post_type TEXT DEFAULT 'original', -- original, repost_own, repost_other, repost_reply, thread
                is_pinned INTEGER DEFAULT 0,

                -- RT / embedded/referenced content data (only when needed)
                original_author TEXT,     -- For reposts or referenced tweets
                original_tweet_id TEXT,   -- For reposts or referenced tweets
                original_content TEXT,    -- For reposts or referenced tweets (if different)
                reply_to_username TEXT,   -- For replies

                -- Media and content
                media_links TEXT,         -- Comma-separated URLs
                media_count INTEGER DEFAULT 0,
                hashtags TEXT,           -- JSON array
                mentions TEXT,           -- JSON array
                external_links TEXT,     -- JSON array

                -- Basic engagement (optional)
                engagement_likes INTEGER DEFAULT 0,
                engagement_retweets INTEGER DEFAULT 0,
                engagement_replies INTEGER DEFAULT 0,

                -- Essential status tracking
                is_deleted INTEGER DEFAULT 0,
                is_edited INTEGER DEFAULT 0,

                -- RT optimization
                rt_original_analyzed INTEGER DEFAULT 0, -- Avoid duplicate analysis

                -- Timestamps (minimal)
                tweet_timestamp TEXT,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (username) REFERENCES accounts (username)
            )
        ''')

        # Content analysis results (platform-agnostic)
        print("  📝 Creating content_analyses table...")
        c.execute('''
            CREATE TABLE content_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id TEXT NOT NULL,           -- Platform-agnostic post identifier
                post_url TEXT,                   -- Platform-agnostic post URL
                author_username TEXT,            -- Platform-agnostic author identifier
                platform TEXT DEFAULT 'twitter', -- Multi-platform support
                post_content TEXT,               -- Platform-agnostic content
                category TEXT,                   -- Primary category (backward compatibility)
                categories_detected TEXT,        -- JSON array of all detected categories
                llm_explanation TEXT,
                analysis_method TEXT DEFAULT "pattern", -- "pattern", "llm", or "gemini"
                analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                analysis_json TEXT,

                -- Multi-category support
                pattern_matches TEXT,            -- JSON array of pattern matches
                topic_classification TEXT,       -- JSON topic classification data

                -- Media analysis (multimodal support)
                media_urls TEXT,                 -- JSON array of media URLs
                media_analysis TEXT,             -- Gemini multimodal analysis result
                media_type TEXT,                 -- "image", "video", or ""
                multimodal_analysis BOOLEAN DEFAULT FALSE, -- Whether media was analyzed

                FOREIGN KEY (post_id) REFERENCES tweets (tweet_id),  -- Keep FK for now, will be updated
                FOREIGN KEY (author_username) REFERENCES accounts (username),  -- Keep FK for now, will be updated
                UNIQUE(post_id) -- One analysis per post
            )
        ''')

        # Performance indexes
        print("  📝 Creating indexes...")
        indexes = [
            ('idx_tweets_username', 'tweets', 'username'),
            ('idx_tweets_timestamp', 'tweets', 'scraped_at'),
            ('idx_analyses_post', 'content_analyses', 'post_id'),
            ('idx_analyses_category', 'content_analyses', 'category'),
            ('idx_analyses_author', 'content_analyses', 'author_username'),
        ]

        for idx_name, table, columns in indexes:
            c.execute(f'CREATE INDEX {idx_name} ON {table}({columns})')

        print(f"    ✅ Created {len(indexes)} performance indexes")

        conn.commit()
        print("✅ Test database schema created successfully!")

    except Exception as e:
        conn.rollback()
        print(f"❌ Test database creation failed: {e}")
        raise
    finally:
        conn.close()

def _load_test_fixtures(db_path: str):
    """Load test fixtures into the database."""
    # This will be implemented when we have test data
    pass

# Legacy compatibility - keep old functions for now
def get_tweet_data(tweet_id: str, timeout: float = 30.0) -> Optional[dict]:
    """Get tweet data for analysis (legacy compatibility)."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT tweet_id, content, username, media_links FROM tweets WHERE tweet_id = ?
        """, (tweet_id,))
        result = cursor.fetchone()
        if result:
            data = dict(result)
            # Parse media_links into a list
            media_links = data.get('media_links', '')
            if media_links:
                data['media_urls'] = [url.strip() for url in media_links.split(',') if url.strip()]
            else:
                data['media_urls'] = []
            return data
        return None
    finally:
        conn.close()

def cleanup_thread_connections():
    """Clean up thread-local database connections (legacy compatibility)."""
    pass  # No longer needed with new connection management

# Global config instance
db_config = DatabaseConfig()

import sqlite3
import os
import threading
from typing import Optional
from queue import Queue, Empty

# Import centralized path management
from .paths import get_db_path

# Database path - now uses centralized path management
DB_PATH = get_db_path()

# Connection pool settings
POOL_SIZE = 5
CONNECTION_TIMEOUT = 30.0

# Thread-local storage for connections
_local = threading.local()

# Thread-local storage for connections
_local = threading.local()

class ConnectionPool:
    """Simple connection pool for SQLite connections."""

    def __init__(self, pool_size: int = POOL_SIZE):
        self.pool_size = pool_size
        self._pool = Queue(maxsize=pool_size)
        self._lock = threading.Lock()

        # Pre-populate the pool
        for _ in range(pool_size):
            conn = self._create_connection()
            self._pool.put(conn)

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection."""
        conn = sqlite3.connect(DB_PATH, timeout=CONNECTION_TIMEOUT, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrency
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        conn.execute('PRAGMA cache_size=1000')
        conn.execute('PRAGMA temp_store=memory')
        return conn

    def get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool."""
        try:
            # Try to get an existing connection
            conn = self._pool.get_nowait()
            # Test if connection is still valid
            try:
                conn.execute('SELECT 1').fetchone()
                return conn
            except sqlite3.Error:
                # Connection is bad, create a new one
                conn.close()
                return self._create_connection()
        except Empty:
            # Pool is empty, create a new connection (beyond pool size if needed)
            return self._create_connection()

    def return_connection(self, conn: sqlite3.Connection):
        """Return a connection to the pool."""
        try:
            # Test if connection is still valid before returning to pool
            conn.execute('SELECT 1').fetchone()
            # Only return to pool if we haven't exceeded pool size
            if self._pool.qsize() < self.pool_size:
                self._pool.put_nowait(conn)
            else:
                conn.close()
        except (sqlite3.Error, Exception):
            # Connection is bad, close it
            try:
                conn.close()
            except:
                pass

    def close_all(self):
        """Close all connections in the pool."""
        while True:
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Empty:
                break

def get_thread_local_connection_pool() -> ConnectionPool:
    """Get a thread-local connection pool."""
    if not hasattr(_local, 'connection_pool'):
        _local.connection_pool = ConnectionPool()
    return _local.connection_pool

class PooledConnection:
    """Wrapper for SQLite connections that returns to pool on close."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._pool = get_thread_local_connection_pool()

    def __getattr__(self, name):
        """Delegate attribute access to the underlying connection."""
        return getattr(self._conn, name)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - return connection to pool."""
        self.close()

    def close(self):
        """Return connection to pool instead of closing it."""
        if self._conn:
            self._pool.return_connection(self._conn)
            self._conn = None

    def real_close(self):
        """Actually close the connection (for cleanup)."""
        if self._conn:
            self._conn.close()
            self._conn = None

def get_db_connection(timeout: float = CONNECTION_TIMEOUT) -> PooledConnection:
    """Get database connection from thread-local pool with row factory and specified timeout."""
    pool = get_thread_local_connection_pool()
    conn = pool.get_connection()
    conn.execute(f'PRAGMA busy_timeout={int(timeout * 1000)}')  # Convert to milliseconds
    return PooledConnection(conn)

def get_tweet_data(tweet_id: str, timeout: float = CONNECTION_TIMEOUT) -> Optional[dict]:
    """Get tweet data for analysis."""
    conn = get_db_connection(timeout)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT tweet_id, content, username, media_links FROM tweets WHERE tweet_id = ?
        """, (tweet_id,))
        result = cursor.fetchone()
        if result:
            data = dict(result)
            # Parse media_links into a list
            media_links = data.get('media_links', '')
            if media_links:
                data['media_urls'] = [url.strip() for url in media_links.split(',') if url.strip()]
            else:
                data['media_urls'] = []
            return data
        return None
    finally:
        conn.close()

def cleanup_thread_connections():
    """Clean up thread-local database connections."""
    if hasattr(_local, 'connection_pool'):
        _local.connection_pool.close_all()
        delattr(_local, 'connection_pool')