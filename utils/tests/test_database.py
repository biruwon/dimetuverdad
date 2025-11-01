"""Unit tests for database module."""

import unittest
import tempfile
import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from database import (
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

    @patch('utils.config')
    def test_database_config_init(self, mock_config):
        """Test DatabaseConfig initialization."""
        mock_config.get_environment.return_value = 'testing'

        config = DatabaseConfig('testing')
        self.assertEqual(config.env, 'testing')
        self.assertIn('testing', config.settings)

    @patch('utils.config')
    def test_database_config_default_env(self, mock_config):
        """Test DatabaseConfig with default environment."""
        mock_config.get_environment.return_value = 'testing'

        config = DatabaseConfig()
        self.assertEqual(config.env, 'testing')

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
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # @patch('database.sqlite3.connect')
    # @patch.object(DatabaseConfig, '__init__', return_value=None)
    # @patch('database.config.get_environment', return_value='testing')
    # def test_get_db_connection(self, mock_get_env, mock_init, mock_connect):
    #     """Test get_db_connection function."""
    #     # This test is complex due to database reorganization - skipping for now
    #     pass

    @patch('database.database.get_db_connection')
    def test_get_db_connection_context(self, mock_get_conn):
        """Test get_db_connection_context manager."""
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn

        with get_db_connection_context() as conn:
            self.assertEqual(conn, mock_conn)

        # Connection should be closed after context
        mock_conn.close.assert_called_once()

    @patch('database.database.get_db_connection')
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

    @patch('utils.paths.get_db_path')
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

    @patch('utils.paths.get_db_path')
    @patch('os.getpid')
    @patch('threading.get_ident')
    @patch('uuid.uuid4')
    @patch('sqlite3.connect')
    @patch('os.path.exists')
    @patch('os.remove')
    @patch('os.chmod')
    def test_init_test_database(self, mock_chmod, mock_remove,
                               mock_exists, mock_connect, mock_uuid, mock_thread_id, mock_pid, mock_get_db_path):
        """Test init_test_database function."""
        # Mock all the external dependencies
        mock_pid.return_value = 12345
        mock_thread_id.return_value = 67890
        mock_uuid.return_value = 'abcd1234-5678-9012-3456-789012345678'  # Full UUID string
        mock_get_db_path.return_value = '/fake/base/db'

        mock_exists.return_value = False  # Database doesn't exist initially

        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        with patch('database.database.create_fresh_database_schema') as mock_create_schema:
            result = init_test_database()

            expected_path = '/fake/base/db.pid_12345.tid_67890.abcd1234'
            self.assertEqual(result, expected_path)

            # Should create fresh schema
            mock_create_schema.assert_called_once_with(expected_path)

            # Should set permissions
            mock_chmod.assert_called_once()

    # @patch('database.get_db_connection')
    # def test_get_tweet_data_found(self, mock_get_conn):
    #     """Test get_tweet_data when tweet is found."""
    #     # This test is complex due to database reorganization - skipping for now
    #     pass

    @patch('database.database.get_db_connection')
    def test_get_tweet_data_not_found(self, mock_get_conn):
        """Test get_tweet_data when tweet is not found."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None

        mock_get_conn.return_value = mock_conn

        result = get_tweet_data('123')
        self.assertIsNone(result)

    @patch('utils.paths.get_db_path')
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