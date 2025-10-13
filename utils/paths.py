"""
Path utilities for the dimetuverdad project.
Centralized path management with environment isolation.
"""

import sys
import os
from pathlib import Path
from typing import Optional

def setup_project_paths():
    """Set up sys.path to include project directories for imports."""
    # Get the project root directory
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent

    # Add project root to sys.path if not already there
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    return project_root

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent

def get_environment() -> str:
    """Get the current environment (development, testing, production)."""
    # Check environment variables first
    env = os.environ.get('DIMETUVERDAD_ENV', '').lower()

    # Auto-detect testing environment
    if os.environ.get('PYTEST_CURRENT_TEST') or 'pytest' in sys.argv[0]:
        return 'testing'

    # Default to development
    if not env:
        return 'development'

    # Validate environment
    valid_envs = ['development', 'testing', 'production']
    if env not in valid_envs:
        raise ValueError(f"Invalid environment '{env}'. Must be one of: {valid_envs}")

    return env

def get_db_path(env: Optional[str] = None, test_mode: bool = False) -> str:
    """Get the database file path for the specified environment."""
    if env is None:
        env = get_environment()

    # Environment-specific database paths
    db_names = {
        'development': 'accounts.db',
        'testing': 'test_accounts.db',
        'production': 'accounts.db'
    }

    db_name = db_names.get(env, 'accounts.db')
    return str(get_project_root() / db_name)

def get_backup_dir() -> str:
    """Get the backup directory path."""
    return str(get_project_root() / 'backups')

def get_scripts_dir() -> Path:
    """Get the scripts directory."""
    return get_project_root() / 'scripts'

def get_web_dir() -> Path:
    """Get the web directory."""
    return get_project_root() / 'web'

def get_test_data_dir() -> Path:
    """Get the test data directory."""
    return get_project_root() / 'tests' / 'data'

def ensure_directories_exist():
    """Ensure all necessary directories exist."""
    dirs = [
        get_backup_dir(),
        str(get_test_data_dir()),
        str(get_project_root() / 'logs'),
        str(get_project_root() / 'temp')
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)