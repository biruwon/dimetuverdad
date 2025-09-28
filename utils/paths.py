"""
Path utilities for the dimetuverdad project.
Centralized path management.
"""

from pathlib import Path

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent

def get_db_path() -> Path:
    """Get the database file path."""
    return get_project_root() / 'accounts.db'

import sys
import os
from pathlib import Path

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

def get_db_path() -> str:
    """Get the database file path."""
    return str(get_project_root() / 'accounts.db')

def get_scripts_dir() -> Path:
    """Get the scripts directory."""
    return get_project_root() / 'scripts'

def get_web_dir() -> Path:
    """Get the web directory."""
    return get_project_root() / 'web'