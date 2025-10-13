"""Unit tests for paths module."""

import unittest
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from utils.paths import (
    setup_project_paths,
    get_project_root,
    get_db_path,
    get_backup_dir,
    get_scripts_dir,
    get_web_dir,
    get_test_data_dir,
    ensure_directories_exist
)


class TestPaths(unittest.TestCase):
    """Test cases for path utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.original_cwd = os.getcwd()
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        os.chdir(self.temp_dir)

        # Create a mock project structure
        self.project_root = Path(self.temp_dir) / "test_project"
        self.project_root.mkdir()

        # Create utils subdirectory
        utils_dir = self.project_root / "utils"
        utils_dir.mkdir()

        # Create a mock paths.py file
        paths_file = utils_dir / "paths.py"
        paths_file.write_text("# Mock paths.py for testing")

    def tearDown(self):
        """Clean up test fixtures."""
        os.chdir(self.original_cwd)
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_project_root(self):
        """Test get_project_root function."""
        # Test that it returns a Path object pointing to the project root
        result = get_project_root()
        self.assertIsInstance(result, Path)
        # Should be the parent.parent of the utils directory
        expected_parts = ['utils', 'tests', 'test_paths.py']
        self.assertTrue(str(result).endswith('dimetuverdad'))

    def test_setup_project_paths(self):
        """Test setup_project_paths function."""
        # Store original sys.path
        original_path = sys.path.copy()

        try:
            result = setup_project_paths()
            self.assertIsInstance(result, Path)

            # Check that project root was added to sys.path
            project_root_str = str(result)
            self.assertIn(project_root_str, sys.path)
        finally:
            # Restore original sys.path
            sys.path[:] = original_path

    def test_setup_project_paths_already_in_path(self):
        """Test setup_project_paths when path is already in sys.path."""
        # Store original sys.path
        original_path = sys.path.copy()

        try:
            # Add project root to sys.path first
            project_root = get_project_root()
            project_root_str = str(project_root)
            sys.path.insert(0, project_root_str)

            # Now call setup_project_paths
            result = setup_project_paths()
            self.assertIsInstance(result, Path)

            # Should still have the project root in sys.path
            self.assertIn(project_root_str, sys.path)
        finally:
            # Restore original sys.path
            sys.path[:] = original_path

    @patch('utils.paths.config')
    def test_get_db_path_development(self, mock_config):
        """Test get_db_path for development environment."""
        mock_config.get_environment.return_value = 'development'

        with patch('utils.paths.get_project_root', return_value=Path('/fake/project')):
            result = get_db_path()
            expected = str(Path('/fake/project') / 'accounts.db')
            self.assertEqual(result, expected)

    @patch('utils.paths.config')
    def test_get_db_path_testing(self, mock_config):
        """Test get_db_path for testing environment."""
        mock_config.get_environment.return_value = 'testing'

        with patch('utils.paths.get_project_root', return_value=Path('/fake/project')):
            result = get_db_path()
            expected = str(Path('/fake/project') / 'test_accounts.db')
            self.assertEqual(result, expected)

    @patch('utils.paths.config')
    def test_get_db_path_production(self, mock_config):
        """Test get_db_path for production environment."""
        mock_config.get_environment.return_value = 'production'

        with patch('utils.paths.get_project_root', return_value=Path('/fake/project')):
            result = get_db_path()
            expected = str(Path('/fake/project') / 'accounts.db')
            self.assertEqual(result, expected)

    @patch('utils.paths.config')
    def test_get_db_path_explicit_env(self, mock_config):
        """Test get_db_path with explicit environment parameter."""
        with patch('utils.paths.get_project_root', return_value=Path('/fake/project')):
            result = get_db_path(env='testing')
            expected = str(Path('/fake/project') / 'test_accounts.db')
            self.assertEqual(result, expected)

    @patch('utils.paths.config')
    def test_get_db_path_unknown_env(self, mock_config):
        """Test get_db_path with unknown environment (should default to accounts.db)."""
        mock_config.get_environment.return_value = 'unknown'

        with patch('utils.paths.get_project_root', return_value=Path('/fake/project')):
            result = get_db_path()
            expected = str(Path('/fake/project') / 'accounts.db')
            self.assertEqual(result, expected)

    def test_get_backup_dir(self):
        """Test get_backup_dir function."""
        with patch('utils.paths.get_project_root', return_value=Path('/fake/project')):
            result = get_backup_dir()
            expected = str(Path('/fake/project') / 'backups')
            self.assertEqual(result, expected)

    def test_get_scripts_dir(self):
        """Test get_scripts_dir function."""
        with patch('utils.paths.get_project_root', return_value=Path('/fake/project')):
            result = get_scripts_dir()
            expected = Path('/fake/project') / 'scripts'
            self.assertEqual(result, expected)

    def test_get_web_dir(self):
        """Test get_web_dir function."""
        with patch('utils.paths.get_project_root', return_value=Path('/fake/project')):
            result = get_web_dir()
            expected = Path('/fake/project') / 'web'
            self.assertEqual(result, expected)

    def test_get_test_data_dir(self):
        """Test get_test_data_dir function."""
        with patch('utils.paths.get_project_root', return_value=Path('/fake/project')):
            result = get_test_data_dir()
            expected = Path('/fake/project') / 'tests' / 'data'
            self.assertEqual(result, expected)

    @patch('utils.paths.Path.mkdir')
    def test_ensure_directories_exist(self, mock_mkdir):
        """Test ensure_directories_exist function."""
        with patch('utils.paths.get_project_root', return_value=Path('/fake/project')):
            with patch('utils.paths.get_backup_dir', return_value='/fake/project/backups'):
                with patch('utils.paths.get_test_data_dir', return_value=Path('/fake/project/tests/data')):
                    ensure_directories_exist()

                    # Check that mkdir was called for each directory
                    expected_calls = [
                        unittest.mock.call(parents=True, exist_ok=True),
                        unittest.mock.call(parents=True, exist_ok=True),
                        unittest.mock.call(parents=True, exist_ok=True),
                        unittest.mock.call(parents=True, exist_ok=True)
                    ]
                    mock_mkdir.assert_has_calls(expected_calls)


if __name__ == '__main__':
    unittest.main()