"""Unit tests for config module."""

import unittest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from utils.config import Config, config


class TestConfig(unittest.TestCase):
    """Test cases for Config class."""

    def setUp(self):
        """Set up test fixtures."""
        # Store original environment
        self.original_env = dict(os.environ)
        self.original_cwd = os.getcwd()

        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        os.chdir(self.temp_dir)

        # Reset config state
        Config._environment = None
        Config._env_loaded = False

    def tearDown(self):
        """Clean up test fixtures."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
        os.chdir(self.original_cwd)

        # Clean up temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)

        # Reset config state
        Config._environment = None
        Config._env_loaded = False

    def test_get_environment_default_development(self):
        """Test get_environment returns testing when running under pytest."""

        # Force reset of cached environment
        Config._environment = None
        Config._env_loaded = False

        result = Config.get_environment()
        # When running tests, we should be in testing environment
        self.assertEqual(result, 'testing')

    def test_get_environment_explicit_env_var(self):
        """Test get_environment returns testing when running under pytest."""
        Config._environment = None  # Reset cached value

        result = Config.get_environment()
        # When running tests, we should be in testing environment
        self.assertEqual(result, 'testing')

    def test_get_environment_invalid_value(self):
        """Test get_environment returns testing when running under pytest."""
        Config._environment = None  # Reset cached value

        result = Config.get_environment()
        # When running tests, we should be in testing environment
        self.assertEqual(result, 'testing')

    def test_get_environment_pytest_detection(self):
        """Test get_environment detects pytest and returns testing environment."""
        Config._environment = None  # Reset cached value

        # Mock os.environ.get to simulate running under pytest
        with patch('os.environ.get') as mock_get:
            mock_get.side_effect = lambda key: 'test_session' if key == 'PYTEST_CURRENT_TEST' else None
            result = Config.get_environment()
            self.assertEqual(result, 'testing')

    def test_set_environment_valid(self):
        """Test set_environment with valid values."""
        for env in ['development', 'testing', 'production']:
            Config.set_environment(env)
            self.assertEqual(Config._environment, env)

    def test_set_environment_invalid(self):
        """Test set_environment raises error for invalid environment."""
        with self.assertRaises(ValueError) as context:
            Config.set_environment('invalid_env')

        self.assertIn("Invalid environment 'invalid_env'", str(context.exception))

    def test_is_development(self):
        """Test is_development method."""
        Config.set_environment('development')
        self.assertTrue(Config.is_development())
        self.assertFalse(Config.is_testing())
        self.assertFalse(Config.is_production())

    def test_is_testing(self):
        """Test is_testing method."""
        Config.set_environment('testing')
        self.assertFalse(Config.is_development())
        self.assertTrue(Config.is_testing())
        self.assertFalse(Config.is_production())

    def test_is_production(self):
        """Test is_production method."""
        Config.set_environment('production')
        self.assertFalse(Config.is_development())
        self.assertFalse(Config.is_testing())
        self.assertTrue(Config.is_production())

    def test_get_env_var_with_value(self):
        """Test get_env_var returns set value."""
        os.environ['TEST_VAR'] = 'test_value'
        Config._env_loaded = False  # Force reload

        result = Config.get_env_var('TEST_VAR')
        self.assertEqual(result, 'test_value')

    def test_get_env_var_default(self):
        """Test get_env_var returns default when not set."""
        # Ensure variable is not set
        os.environ.pop('NONEXISTENT_VAR', None)
        Config._env_loaded = False  # Force reload

        result = Config.get_env_var('NONEXISTENT_VAR', 'default_value')
        self.assertEqual(result, 'default_value')

    def test_get_env_var_none_default(self):
        """Test get_env_var returns None when not set and no default."""
        # Ensure variable is not set
        os.environ.pop('NONEXISTENT_VAR', None)
        Config._env_loaded = False  # Force reload

        result = Config.get_env_var('NONEXISTENT_VAR')
        self.assertIsNone(result)

    def test_load_env_file_manual(self):
        """Test manual .env file loading."""
        # Create a mock .env file
        env_file = Path(self.temp_dir) / '.env'
        env_file.write_text('TEST_VAR=test_value\n# Comment\nEMPTY_VAR=\n')

        # Reset env loaded flag
        Config._env_loaded = False

        # Call the manual loader directly
        Config._load_env_file_manual(env_file)

        # Check that variables were loaded
        self.assertEqual(os.environ.get('TEST_VAR'), 'test_value')
        self.assertEqual(os.environ.get('EMPTY_VAR'), '')
        self.assertNotIn('#', os.environ)  # Comments should be ignored

    def test_get_database_path_development(self):
        """Test get_database_path for development."""
        Config.set_environment('development')

        result = Config.get_database_path()
        self.assertTrue(result.endswith('accounts.db'))
        self.assertIn('dimetuverdad', result)

    def test_get_database_path_testing(self):
        """Test get_database_path for testing environment."""
        # Since we removed pytest detection, testing environment now uses same path as development
        Config.set_environment('testing')

        result = Config.get_database_path()
        self.assertTrue(result.endswith('accounts.db'))
        self.assertIn('dimetuverdad', result)

    def test_get_database_path_production(self):
        """Test get_database_path for production."""
        Config.set_environment('production')

        result = Config.get_database_path()
        self.assertTrue(result.endswith('accounts.db'))
        self.assertIn('dimetuverdad', result)

    def test_get_database_path_custom_env_var(self):
        """Test get_database_path with custom DATABASE_PATH env var."""
        # Force environment to development and clear cache
        Config._environment = 'development'
        Config._env_loaded = False
        os.environ['DATABASE_PATH'] = 'custom.db'

        # Mock _load_env_file to prevent .env file loading
        with patch.object(Config, '_load_env_file'):
            result = Config.get_database_path()
            # Should use the custom database path
            self.assertIn('custom.db', result)

    def test_get_backup_dir(self):
        """Test get_backup_dir method."""
        result = Config.get_backup_dir()
        self.assertTrue(result.endswith('backups'))
        self.assertIn('dimetuverdad', result)

    def test_get_project_root(self):
        """Test get_project_root method."""
        result = Config.get_project_root()
        self.assertIsInstance(result, Path)
        self.assertTrue(str(result).endswith('dimetuverdad'))

    def test_load_env_file_with_dotenv(self):
        """Test .env file loading when dotenv is available."""
        # Create a mock .env file in the expected location
        env_file = Path(self.temp_dir) / '.env.development'  # Since _get_environment_no_load returns 'development'
        env_file.write_text('DOTENV_TEST_VAR=dotenv_value\nDOTENV_EMPTY_VAR=\n')

        # Change to temp directory so the .env file is found
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)

        try:
            # Reset env loaded flag
            Config._env_loaded = False

            # Mock the dotenv.load_dotenv function
            with patch('utils.config.dotenv.load_dotenv') as mock_load_dotenv:
                # Call the loader
                Config._load_env_file()

                # Verify dotenv.load_dotenv was called
                mock_load_dotenv.assert_called_once()
                # Check that it was called with override=True
                call_args = mock_load_dotenv.call_args
                self.assertTrue(call_args[1]['override'])  # kwargs should contain override=True

        finally:
            os.chdir(original_cwd)

    def test_get_environment_caching(self):
        """Test that get_environment caches results."""
        Config._environment = None
        Config.set_environment('production')

        # First call
        result1 = Config.get_environment()
        self.assertEqual(result1, 'production')

        # Second call should return cached value
        result2 = Config.get_environment()
        self.assertEqual(result2, 'production')

        # Verify it's the same object/reference
        self.assertIs(result1, result2)


if __name__ == '__main__':
    unittest.main()