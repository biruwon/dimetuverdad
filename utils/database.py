"""
Database configuration and connection management.
Provides environment-isolated database connections and configuration.
"""

import sqlite3
import os
import threading
import fcntl
import glob
import tempfile
import uuid
import atexit
import stat
from contextlib import contextmanager
from typing import Optional, Dict, Any
from utils import paths
from utils.config import config

# Direct import of schema creation function for tests
# from scripts.init_database import create_fresh_database_schema  # Circular import - removed

def create_fresh_database_schema(db_path: str):
    """Create a clean database schema for the specified path."""
    print(f"🏗️  Creating fresh database schema at {db_path}...")

    # Create connection directly to the target database
    conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
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
        CREATE TABLE IF NOT EXISTS content_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id TEXT UNIQUE NOT NULL,
            post_url TEXT,
            author_username TEXT,
            platform TEXT DEFAULT 'twitter',
            post_content TEXT,
            category TEXT,
            categories_detected TEXT,
            local_explanation TEXT,
            external_explanation TEXT,
            analysis_stages TEXT,
            external_analysis_used BOOLEAN DEFAULT FALSE,
            analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            analysis_json TEXT,
            pattern_matches TEXT,
            topic_classification TEXT,
            media_urls TEXT,
            media_type TEXT,
            verification_data TEXT,
            verification_confidence REAL DEFAULT 0.0
        )
        ''')

        # Post edits detection (renamed for clarity - tracks post content changes)
        print("  📝 Creating post_edits table...")
        c.execute('''
            CREATE TABLE post_edits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id TEXT NOT NULL,
                version_number INTEGER NOT NULL,
                previous_content TEXT NOT NULL,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (post_id) REFERENCES tweets (tweet_id),
                UNIQUE(post_id, version_number)
            )
        ''')

        # User feedback table for model improvement (platform-agnostic)
        print("  📝 Creating user_feedback table...")
        c.execute('''
            CREATE TABLE user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id TEXT NOT NULL,          -- Platform-agnostic post identifier
                feedback_type TEXT NOT NULL,    -- 'correction', 'flag', 'improvement'
                original_category TEXT,
                corrected_category TEXT,
                user_comment TEXT,
                user_ip TEXT,                   -- For rate limiting and analytics
                submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (post_id) REFERENCES tweets (tweet_id)
            )
        ''')

        # Platforms table for multi-platform support (hierarchical)
        print("  📝 Creating platforms table...")
        c.execute('''
            CREATE TABLE platforms (
                platform_id TEXT PRIMARY KEY,
                category TEXT NOT NULL,  -- 'social_media', 'messenger', 'news', etc.
                name TEXT NOT NULL,
                display_name TEXT NOT NULL,
                description TEXT,
                api_base_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Populate platforms table with defaults
        print("  📝 Populating platforms table...")
        platforms_data = [
            ('twitter', 'social_media', 'Twitter', 'Twitter/X', 'Social media platform for short-form content', 'https://twitter.com'),
            ('telegram', 'messenger', 'Telegram', 'Telegram', 'Messaging platform with channels and groups', 'https://telegram.org'),
            ('news', 'news', 'News', 'News Sources', 'Newspaper and news website sources', None),
        ]

        c.executemany('''
            INSERT INTO platforms (platform_id, category, name, display_name, description, api_base_url)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', platforms_data)

        # Performance indexes
        print("  📝 Creating indexes...")
        indexes = [
            ('idx_tweets_username', 'tweets', 'username'),
            ('idx_tweets_post_type', 'tweets', 'post_type'),
            ('idx_tweets_timestamp', 'tweets', 'scraped_at'),
            ('idx_tweets_tweet_timestamp', 'tweets', 'tweet_timestamp'),
            ('idx_tweets_deleted', 'tweets', 'is_deleted'),
            ('idx_tweets_edited', 'tweets', 'is_edited'),
            ('idx_analyses_post', 'content_analyses', 'post_id'),
            ('idx_analyses_category', 'content_analyses', 'category'),
            ('idx_analyses_author', 'content_analyses', 'author_username'),
            ('idx_analyses_platform', 'content_analyses', 'platform'),
            ('idx_content_analyses_timestamp', 'content_analyses', 'analysis_timestamp'),
            ('idx_content_analyses_stages', 'content_analyses', 'analysis_stages'),
            ('idx_content_analyses_external', 'content_analyses', 'external_analysis_used'),
            ('idx_post_edits_post', 'post_edits', 'post_id'),
            ('idx_user_feedback_post', 'user_feedback', 'post_id'),
            ('idx_user_feedback_type', 'user_feedback', 'feedback_type'),
            ('idx_user_feedback_submitted', 'user_feedback', 'submitted_at'),
            ('idx_platforms_name', 'platforms', 'name')
        ]

        for idx_name, table, columns in indexes:
            c.execute(f'CREATE INDEX {idx_name} ON {table}({columns})')

        print(f"    ✅ Created {len(indexes)} performance indexes")

        conn.commit()
        print("✅ Clean database schema created successfully!")

        # Show summary
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row['name'] for row in c.fetchall()]
        print(f"📊 Created {len(tables)} tables: {', '.join(tables)}")

    except Exception as e:
        conn.rollback()
        print(f"❌ Database creation failed: {e}")
        raise
    finally:
        conn.close()

# Thread-local storage for database connections
_local = threading.local()

class DatabaseConfig:
    """Database configuration for different environments."""

    def __init__(self, env: str = None):
        self.env = env or config.get_environment()
        self.db_path = paths.get_db_path(env=self.env)

        # Environment-specific settings
        self.settings = {
            'development': {
                'timeout': 30.0,
                'isolation_level': None,  # Autocommit mode for development
                'check_same_thread': False,
                'enable_foreign_keys': True,
                'pragma_settings': {
                    'journal_mode': 'WAL',
                    'synchronous': 'NORMAL',
                    'cache_size': -8000,  # 8MB cache
                }
            },
            'testing': {
                'timeout': 10.0,
                'isolation_level': None,
                'check_same_thread': False,
                'enable_foreign_keys': True,
                'pragma_settings': {
                    'journal_mode': 'MEMORY',
                    'synchronous': 'OFF',
                    'cache_size': -1000,  # 1MB cache
                }
            },
            'production': {
                'timeout': 60.0,
                'isolation_level': None,
                'check_same_thread': True,
                'enable_foreign_keys': True,
                'pragma_settings': {
                    'journal_mode': 'WAL',
                    'synchronous': 'NORMAL',
                    'cache_size': -64000,  # 64MB cache
                }
            }
        }

    def get_connection_params(self) -> Dict[str, Any]:
        """Get connection parameters for the current environment."""
        return self.settings.get(self.env, self.settings['development'])

    def get_pragma_settings(self) -> Dict[str, Any]:
        """Get PRAGMA settings for the current environment."""
        return self.get_connection_params().get('pragma_settings', {})

    def get_db_path(self) -> str:
        """Get the database path for the current environment."""
        # Check if DATABASE_PATH environment variable is set (for testing)
        env_db_path = os.environ.get('DATABASE_PATH')
        if env_db_path:
            return env_db_path
        return self.db_path

def _get_flask_database_path() -> Optional[str]:
    """Get database path from Flask app config if available."""
    try:
        from flask import current_app
        if current_app and hasattr(current_app, 'config') and 'DATABASE_PATH' in current_app.config:
            return current_app.config['DATABASE_PATH']
    except (ImportError, RuntimeError):
        # Not in Flask context or Flask not available
        pass
    
    
    return None

def get_db_connection() -> sqlite3.Connection:
    """
    Get a database connection using automatic environment detection.

    Returns:
        SQLite database connection
    """
    env = config.get_environment()

    config_obj = DatabaseConfig(env)
    params = config_obj.get_connection_params().copy()
    # Store enable_foreign_keys before removing it from params
    enable_foreign_keys = params.pop('enable_foreign_keys', True)
    pragma_settings = params.pop('pragma_settings', {})

    # Determine db_path with priority: Flask config > environment-based path
    flask_db_path = _get_flask_database_path()
    env_db_path = config_obj.get_db_path()
    
    final_db_path = flask_db_path or env_db_path
    conn = sqlite3.connect(final_db_path, **params)

    # Always enable row factory for dict-like access
    conn.row_factory = sqlite3.Row

    # Enable foreign keys
    if enable_foreign_keys:
        conn.execute("PRAGMA foreign_keys = ON")

    # Always set a reasonable busy timeout for concurrent writers/readers
    # This helps avoid immediate 'database is locked' errors under parallel tasks
    try:
        conn.execute("PRAGMA busy_timeout = 30000")  # 30 seconds
    except Exception:
        # Best-effort; ignore if not supported
        pass

    # Apply environment-specific PRAGMA settings
    for pragma_name, pragma_value in pragma_settings.items():
        try:
            if isinstance(pragma_value, int):
                conn.execute(f"PRAGMA {pragma_name} = {pragma_value}")
            else:
                conn.execute(f"PRAGMA {pragma_name} = {pragma_value}")
        except Exception:
            # Best-effort; ignore PRAGMA errors
            pass

    return conn

@contextmanager
def get_db_connection_context():
    """
    Context manager for database connections.

    Automatically handles connection cleanup and error handling.
    """
    conn = None
    try:
        conn = get_db_connection()
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
    
    # Clean up all test databases with the actual naming pattern
    base_db_path = paths.get_db_path(env='testing')
    test_pattern = f"{base_db_path}.session_*"
    
    for db_file in glob.glob(test_pattern):
        try:
            os.remove(db_file)
            print(f"🗑️  Cleaned up test database: {os.path.basename(db_file)}")
        except OSError as e:
            print(f"⚠️  Could not remove test database {db_file}: {e}")
    
    # Also clean up any leftover lock files
    lock_pattern = f"{base_db_path}*.lock"
    for lock_file in glob.glob(lock_pattern):
        try:
            os.remove(lock_file)
        except OSError:
            pass

def init_test_database(fixtures: bool = False) -> str:
    """
    Initialize a test database with schema and optional fixtures.

    Returns:
        Path to the test database
    """

    # Create unique test database per process/thread to avoid conflicts
    # Use process ID, thread ID, and random UUID for uniqueness
    process_id = os.getpid()
    thread_id = threading.get_ident()
    unique_id = str(uuid.uuid4())[:8]

    # Get base test database path and make it unique
    base_db_path = paths.get_db_path(env='testing')
    test_db_path = f"{base_db_path}.pid_{process_id}.tid_{thread_id}.{unique_id}"

    # Clean up any existing database for this unique path
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except OSError:
            pass  # Ignore if we can't remove it

    # Create fresh schema directly
    _create_test_database_schema(test_db_path)

    # Ensure proper permissions for test database
    try:
        os.chmod(test_db_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP)
    except OSError:
        pass  # Ignore permission errors

    # Load fixtures if requested
    if fixtures:
        _load_test_fixtures(test_db_path)

    # Register for automatic cleanup when process exits
    def cleanup_this_db():
        try:
            if os.path.exists(test_db_path):
                os.remove(test_db_path)
                print(f"🗑️  Cleaned up test database: {os.path.basename(test_db_path)}")
        except OSError:
            pass

    atexit.register(cleanup_this_db)

    return test_db_path

def _create_test_database_schema(db_path: str):
    """Create test database schema using the centralized schema from init_database.py."""
    # Direct call to the centrally imported schema creator
    create_fresh_database_schema(db_path)

def _load_test_fixtures(db_path: str):
    """Load test fixtures into the database."""
    # This will be implemented when we have test data
    pass

# Legacy compatibility - keep old functions for now
def get_tweet_data(tweet_id: str, timeout: float = 30.0) -> Optional[dict]:
    """Get tweet data for analysis (legacy compatibility)."""
    with get_db_connection_context() as conn:
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

def cleanup_test_database():
    """Clean up the test database by removing it entirely."""

    # Clean up all test databases for this process
    process_id = os.getpid()
    base_db_path = paths.get_db_path(env='testing')

    # Find all test databases for this process
    pattern = f"{base_db_path}.pid_{process_id}.*"
    test_db_files = glob.glob(pattern)

    for test_db_path in test_db_files:
        try:
            os.remove(test_db_path)
            print(f"🗑️  Removed test database: {test_db_path}")
        except OSError as e:
            print(f"⚠️  Could not remove test database {test_db_path}: {e}")

    # Also clean up any leftover lock files
    lock_pattern = f"{base_db_path}*.lock"
    lock_files = glob.glob(lock_pattern)
    for lock_file in lock_files:
        try:
            os.remove(lock_file)
        except OSError:
            pass

# Global config instance
db_config = DatabaseConfig()