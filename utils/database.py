"""
Database configuration and connection management.
Provides environment-isolated database connections and configuration.
"""

import sqlite3
import os
import threading
import fcntl
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

def get_db_connection(env: str = None, row_factory: bool = True, db_path: str = None) -> sqlite3.Connection:
    """
    Get a database connection for the specified environment.

    Args:
        env: Environment name ('development', 'testing', 'production')
        row_factory: Whether to enable row factory for dict-like access
        db_path: Optional explicit database path to use

    Returns:
        SQLite database connection
    """
    if env is None:
        env = paths.get_environment()

    config = DatabaseConfig(env)
    params = config.get_connection_params().copy()
    # Store enable_foreign_keys before removing it from params
    enable_foreign_keys = params.pop('enable_foreign_keys', True)

    # Use explicit db_path if provided and do not modify it
    if db_path:
        conn = sqlite3.connect(db_path, **params)
    else:
        # Check for Flask app DATABASE_PATH config first (for web tests)
        final_db_path = _get_flask_database_path() or config.get_db_path()
        conn = sqlite3.connect(final_db_path, **params)

    if row_factory:
        conn.row_factory = sqlite3.Row

    # Enable foreign keys
    if enable_foreign_keys:
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
    
    # Clean up all test databases with the actual naming pattern
    base_db_path = paths.get_db_path(env='testing')
    test_pattern = f"{base_db_path}.pid_*"
    
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
    import os
    import tempfile
    import threading
    import uuid
    import atexit

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
        import stat
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
        except OSError:
            pass
    
    atexit.register(cleanup_this_db)

    return test_db_path

def _create_test_database_schema(db_path: str):
    """Create test database schema using the centralized schema from init_database.py."""
    # Lazy import to avoid circular dependency
    try:
        from scripts.init_database import create_fresh_database_schema
    except ImportError:
        # Fallback for when running from different directories
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from scripts.init_database import create_fresh_database_schema

    # Use the same schema creation function as the main database initialization
    create_fresh_database_schema(db_path)

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

def cleanup_test_database():
    """Clean up the test database by removing it entirely."""
    import os
    import glob
    import threading

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