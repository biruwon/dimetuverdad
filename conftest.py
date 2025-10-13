"""
Global pytest configuration for dimetuverdad project.
Handles test database cleanup and other global test settings.
"""

import pytest
import os
import atexit
from utils.database import cleanup_test_databases


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_databases_on_exit():
    """Automatically clean up test databases when pytest session ends."""
    
    def cleanup():
        """Cleanup function to be called on session end."""
        print("\n🧹 Cleaning up test databases...")
        cleanup_test_databases()
    
    # Register cleanup to run when pytest exits
    atexit.register(cleanup)
    
    yield
    
    # Also run cleanup at the end of this fixture
    cleanup()


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment for all tests."""
    # Ensure we're in testing mode
    original_env = os.environ.get('DIMETUVERDAD_ENV')
    os.environ['DIMETUVERDAD_ENV'] = 'testing'
    
    yield
    
    # Restore original environment
    if original_env:
        os.environ['DIMETUVERDAD_ENV'] = original_env
    else:
        os.environ.pop('DIMETUVERDAD_ENV', None)