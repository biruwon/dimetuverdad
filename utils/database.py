"""
Database utilities for the dimetuverdad project.
Centralized database connection and operation patterns.
"""

import sqlite3
import os
import threading
from typing import Optional
from queue import Queue, Empty

# Database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'accounts.db')

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

# Global connection pool (deprecated - use thread-local instead)
_connection_pool = None
_pool_lock = threading.Lock()

def get_connection_pool() -> ConnectionPool:
    """Get the global connection pool (singleton pattern)."""
    global _connection_pool
    if _connection_pool is None:
        with _pool_lock:
            if _connection_pool is None:
                _connection_pool = ConnectionPool()
    return _connection_pool

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