"""Unit tests for database module."""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from utils.database import (
    DatabaseConfig,
    get_db_connection,
    get_db_connection_context,
    cleanup_test_databases,
    init_test_database,
    get_tweet_data,
    cleanup_test_database
)


class TestDatabaseConfig(unittest.TestCase):
    """Test cases for DatabaseConfig class."""

    @patch('utils.database.config')
    def test_database_config_init(self, mock_config):
        """Test DatabaseConfig initialization."""
        mock_config.get_environment.return_value = 'testing'

        config = DatabaseConfig('testing')
        self.assertEqual(config.env, 'testing')
        self.assertIn('testing', config.settings)

    @patch('utils.database.config')
    def test_database_config_default_env(self, mock_config):
        """Test DatabaseConfig with default environment."""
        mock_config.get_environment.return_value = 'development'

        config = DatabaseConfig()
        self.assertEqual(config.env, 'development')

    def test_get_connection_params_development(self):
        """Test get_connection_params for development."""
        config = DatabaseConfig('development')
        params = config.get_connection_params()

        expected_keys = ['timeout', 'isolation_level', 'check_same_thread', 'enable_foreign_keys']
        for key in expected_keys:
            self.assertIn(key, params)

        self.assertEqual(params['timeout'], 30.0)
        self.assertIsNone(params['isolation_level'])

    def test_get_connection_params_testing(self):
        """Test get_connection_params for testing."""
        config = DatabaseConfig('testing')
        params = config.get_connection_params()

        self.assertEqual(params['timeout'], 10.0)
        self.assertIsNone(params['isolation_level'])

    def test_get_connection_params_production(self):
        """Test get_connection_params for production."""
        config = DatabaseConfig('production')
        params = config.get_connection_params()

        self.assertEqual(params['timeout'], 60.0)
        self.assertIsNone(params['isolation_level'])


class TestDatabaseFunctions(unittest.TestCase):
    """Test cases for database utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('utils.database.sqlite3.connect')
    @patch('utils.database.DatabaseConfig')
    @patch('utils.database.config.get_environment')
    def test_get_db_connection(self, mock_get_env, mock_config_class, mock_connect):
        """Test get_db_connection function."""
        # Mock the environment detection to return 'testing'
        mock_get_env.return_value = 'testing'
        
        # Mock the database config
        mock_config = MagicMock()
        mock_config.get_connection_params.return_value = {
            'timeout': 10.0,  # Testing environment
            'check_same_thread': False,
            'enable_foreign_keys': True,
            'pragma_settings': {
                'journal_mode': 'MEMORY',
                'synchronous': 'OFF',
                'cache_size': -1000,
            }
        }
        mock_config.get_db_path.return_value = '/fake/db/path'
        mock_config_class.return_value = mock_config

        # Mock the connection
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        result = get_db_connection()

        mock_connect.assert_called_once()
        args, kwargs = mock_connect.call_args
        self.assertEqual(args[0], '/fake/db/path')
        self.assertEqual(kwargs['timeout'], 10.0)

        # Check that PRAGMA statements were executed (order may vary)
        execute_calls = [call.args[0] for call in mock_conn.execute.call_args_list]
        self.assertIn("PRAGMA foreign_keys = ON", execute_calls)
        self.assertIn("PRAGMA journal_mode = MEMORY", execute_calls)
        self.assertIn("PRAGMA synchronous = OFF", execute_calls)
        self.assertIn("PRAGMA cache_size = -1000", execute_calls)

    @patch('utils.database.get_db_connection')
    def test_get_db_connection_context(self, mock_get_conn):
        """Test get_db_connection_context manager."""
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        with get_db_connection_context() as conn:
            self.assertEqual(conn, mock_conn)

        # Connection should be closed after context
        mock_conn.close.assert_called_once()

    @patch('utils.database.get_db_connection')
    def test_get_db_connection_context_exception(self, mock_get_conn):
        """Test get_db_connection_context with exception."""
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        with self.assertRaises(ValueError):
            with get_db_connection_context() as conn:
                raise ValueError("Test exception")

        # Connection should be rolled back and closed
        mock_conn.rollback.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch('utils.database.paths.get_db_path')
    @patch('glob.glob')
    @patch('os.remove')
    def test_cleanup_test_databases(self, mock_remove, mock_glob, mock_get_db_path):
        """Test cleanup_test_databases function."""
        mock_get_db_path.return_value = '/fake/base/db'
        mock_glob.return_value = ['/fake/base/db.pid_12345.test1', '/fake/base/db.pid_12345.test2']

        cleanup_test_databases()

        # Should attempt to remove the files found by glob
        mock_remove.assert_any_call('/fake/base/db.pid_12345.test1')
        mock_remove.assert_any_call('/fake/base/db.pid_12345.test2')

    @patch('utils.database.paths.get_db_path')
    @patch('os.getpid')
    @patch('threading.get_ident')
    @patch('uuid.uuid4')
    @patch('utils.database.sqlite3.connect')
    @patch('utils.database.os.path.exists')
    @patch('utils.database.os.remove')
    @patch('utils.database.os.chmod')
    @patch('utils.database.DatabaseConfig')
    def test_init_test_database(self, mock_config_class, mock_chmod, mock_remove,
                               mock_exists, mock_connect, mock_uuid, mock_thread_id, mock_pid, mock_get_db_path):
        """Test init_test_database function."""
        # Mock all the external dependencies
        mock_pid.return_value = 12345
        mock_thread_id.return_value = 67890
        mock_uuid.return_value = 'abcd1234-5678-9012-3456-789012345678'  # Full UUID string
        mock_get_db_path.return_value = '/fake/base/db'

        mock_config = MagicMock()
        mock_config.get_db_path.return_value = '/fake/base/db'
        mock_config_class.return_value = mock_config

        mock_exists.return_value = False  # Database doesn't exist initially

        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        with patch('utils.database.create_fresh_database_schema') as mock_create_schema:
            result = init_test_database()

            expected_path = '/fake/base/db.pid_12345.tid_67890.abcd1234'
            self.assertEqual(result, expected_path)

            # Should create fresh schema
            mock_create_schema.assert_called_once_with(expected_path)

            # Should set permissions
            mock_chmod.assert_called_once()

    @patch('utils.database.get_db_connection')
    def test_get_tweet_data_found(self, mock_get_conn):
        """Test get_tweet_data when tweet is found."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        # Create a mock row that behaves like sqlite3.Row
        mock_row = MagicMock()
        row_data = {
            'tweet_id': '123',
            'content': 'test content',
            'username': 'testuser',
            'media_links': 'link1.jpg,link2.png'
        }
        mock_row.__getitem__ = MagicMock(side_effect=lambda key: row_data[key])
        mock_row.keys = MagicMock(return_value=row_data.keys())
        # Make it behave like a dict for dict() conversion
        mock_row.__iter__ = MagicMock(return_value=iter(row_data.items()))
        mock_cursor.fetchone.return_value = mock_row

        mock_get_conn.return_value = mock_conn

        result = get_tweet_data('123')

        expected = {
            'tweet_id': '123',
            'content': 'test content',
            'username': 'testuser',
            'media_links': 'link1.jpg,link2.png',
            'media_urls': ['link1.jpg', 'link2.png']
        }
        self.assertEqual(result, expected)

    @patch('utils.database.get_db_connection')
    def test_get_tweet_data_not_found(self, mock_get_conn):
        """Test get_tweet_data when tweet is not found."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None

        mock_get_conn.return_value = mock_conn

        result = get_tweet_data('123')
        self.assertIsNone(result)

    @patch('utils.database.paths.get_db_path')
    @patch('glob.glob')
    @patch('os.remove')
    @patch('os.getpid')
    def test_cleanup_test_database(self, mock_pid, mock_remove, mock_glob, mock_get_db_path):
        """Test cleanup_test_database function."""
        mock_pid.return_value = 12345
        mock_get_db_path.return_value = '/fake/base/db'
        mock_glob.return_value = ['/fake/base/db.pid_12345.test1', '/fake/base/db.pid_12345.test2']

        cleanup_test_database()

        # Should attempt to remove the files found by glob
        mock_remove.assert_any_call('/fake/base/db.pid_12345.test1')
        mock_remove.assert_any_call('/fake/base/db.pid_12345.test2')


if __name__ == '__main__':
    unittest.main()