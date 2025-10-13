"""
Configuration management for dimetuverdad.
Best practices for environment configuration in Python.
"""

import os
from typing import Optional
from pathlib import Path


class Config:
    """Centralized configuration management."""

    # Environment settings
    _environment: Optional[str] = None
    _env_loaded: bool = False

    @classmethod
    def _load_env_file(cls) -> None:
        """Load environment-specific .env file if it exists."""
        if cls._env_loaded:
            return

        # Get environment without loading .env files first (to avoid recursion)
        env = cls._get_environment_no_load()
        env_file = Path(__file__).resolve().parent.parent / f'.env.{env}'

        if env_file.exists():
            try:
                import dotenv
                dotenv.load_dotenv(env_file, override=True)
            except ImportError:
                # dotenv not installed, load manually
                cls._load_env_file_manual(env_file)
        else:
            # Only load default .env if no environment-specific file exists
            default_env_file = Path(__file__).resolve().parent.parent / '.env'
            if default_env_file.exists():
                try:
                    import dotenv
                    dotenv.load_dotenv(default_env_file, override=True)
                except ImportError:
                    # dotenv not installed, load manually
                    cls._load_env_file_manual(default_env_file)

        cls._env_loaded = True

    @classmethod
    def _get_environment_no_load(cls) -> str:
        """Get environment without loading .env files (to avoid recursion)."""
        # Priority order: explicit env var > auto-detection > default
        env = os.environ.get('DIMETUVERDAD_ENV', '').strip().lower()

        # Auto-detect testing environment
        if not env:
            if (os.environ.get('PYTEST_CURRENT_TEST') or
                'pytest' in os.sys.argv[0] or
                'test' in os.sys.argv[0]):
                env = 'testing'

        # Default to development
        if not env:
            env = 'development'

        return env

    @classmethod
    def _load_env_file_manual(cls, env_file: Path) -> None:
        """Manually load .env file without python-dotenv dependency."""
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ.setdefault(key, value)

    @classmethod
    def get_environment(cls) -> str:
        """Get the current environment with proper priority order."""
        if cls._environment is not None:
            return cls._environment

        # Load environment files first
        cls._load_env_file()

        # Priority order: explicit env var > auto-detection > default
        env = os.environ.get('DIMETUVERDAD_ENV', '').strip().lower()

        # Default to development
        if not env:
            env = 'development'

        # Validate
        valid_envs = {'development', 'testing', 'production'}
        if env not in valid_envs:
            raise ValueError(f"Invalid environment '{env}'. Must be one of: {valid_envs}")

        cls._environment = env
        return env

    @classmethod
    def set_environment(cls, env: str) -> None:
        """Override the environment (useful for testing)."""
        valid_envs = {'development', 'testing', 'production'}
        if env not in valid_envs:
            raise ValueError(f"Invalid environment '{env}'. Must be one of: {valid_envs}")
        cls._environment = env

    @classmethod
    def is_development(cls) -> bool:
        """Check if running in development environment."""
        return cls.get_environment() == 'development'

    @classmethod
    def is_testing(cls) -> bool:
        """Check if running in testing environment."""
        return cls.get_environment() == 'testing'

    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production environment."""
        return cls.get_environment() == 'production'

    @classmethod
    def get_env_var(cls, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable with automatic .env file loading."""
        cls._load_env_file()
        return os.environ.get(key, default)

    @classmethod
    def get_database_path(cls) -> str:
        """Get database path for current environment."""
        env = cls.get_environment()

        # Check for environment-specific database path
        db_path = cls.get_env_var('DATABASE_PATH')
        if db_path:
            project_root = Path(__file__).resolve().parent.parent
            return str(project_root / db_path)

        # Fallback to environment-specific database names
        db_names = {
            'development': 'accounts.db',
            'testing': 'test_accounts.db',
            'production': 'accounts.db'
        }

        db_name = db_names.get(env, 'accounts.db')
        project_root = Path(__file__).resolve().parent.parent
        return str(project_root / db_name)

    @classmethod
    def get_backup_dir(cls) -> str:
        """Get backup directory path."""
        project_root = Path(__file__).resolve().parent.parent
        return str(project_root / 'backups')

    @classmethod
    def get_project_root(cls) -> Path:
        """Get project root directory."""
        return Path(__file__).resolve().parent.parent


# Global instance for easy access
config = Config()