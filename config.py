"""
Configuration settings for dimetuverdad Flask application.
All configuration values should be defined here for easy management.
"""
import os
from pathlib import Path

# Import environment config
from utils.config import config

# Base directory
BASE_DIR = Path(__file__).parent

# =============================================================================
# SECURITY SETTINGS
# =============================================================================

# Admin authentication
ADMIN_TOKEN = os.environ.get('ADMIN_TOKEN', 'change_this_in_production')

# Flask secret key
SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', os.urandom(32).hex())

# =============================================================================
# DATABASE SETTINGS
# =============================================================================

# Database connection settings
DB_TIMEOUT = float(os.environ.get('DB_TIMEOUT', '30.0'))
DB_CHECK_SAME_THREAD = os.environ.get('DB_CHECK_SAME_THREAD', 'False').lower() == 'true'

# =============================================================================
# CACHE SETTINGS
# =============================================================================

# Flask-Caching configuration - disabled for development/testing, enabled for production
if config.is_production():
    CACHE_TYPE = 'flask_caching.backends.SimpleCache'
else:
    CACHE_TYPE = 'flask_caching.backends.NullCache'  # Disabled for development/testing
CACHE_DEFAULT_TIMEOUT = int(os.environ.get('CACHE_DEFAULT_TIMEOUT', '300'))  # 5 minutes

# =============================================================================
# RATE LIMITING SETTINGS
# =============================================================================

# Rate limits for different endpoints (requests per window)
RATE_LIMITS = {
    'admin_login': {'max_requests': 5, 'window_seconds': 300},  # 5 attempts per 5 minutes
    'admin_actions': {'max_requests': 5, 'window_seconds': 300},  # 5 requests per 5 minutes
    'main_dashboard': {'max_requests': 30, 'window_seconds': 60},  # 30 requests per minute
    'user_pages': {'max_requests': 20, 'window_seconds': 60},  # 20 requests per minute
    'api_endpoints': {'max_requests': 10, 'window_seconds': 60},  # 10 requests per minute
    'api_versions': {'max_requests': 20, 'window_seconds': 60},  # 20 requests per minute
    'api_usernames': {'max_requests': 30, 'window_seconds': 60},  # 30 requests per minute
    'export_endpoints': {'max_requests': 3, 'window_seconds': 600},  # 3 requests per 10 minutes
}

# =============================================================================
# PAGINATION SETTINGS
# =============================================================================

# Default pagination settings
DEFAULT_PER_PAGE = {
    'accounts': 10,
    'tweets': 10,
    'admin_category': 20,
}

# =============================================================================
# TIMEOUT SETTINGS
# =============================================================================

# Command execution timeouts (seconds)
COMMAND_TIMEOUTS = {
    'fetch': 600,  # 10 minutes
    'analyze': 300,  # 5 minutes
    'user_analysis': 180,  # 3 minutes
}

# =============================================================================
# LOGGING SETTINGS
# =============================================================================

# Logging configuration
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# =============================================================================
# FLASK SETTINGS
# =============================================================================

# Flask application settings
DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
PORT = int(os.environ.get('FLASK_PORT', '5000'))

# =============================================================================
# ANALYSIS SETTINGS
# =============================================================================

# Analysis categories
ANALYSIS_CATEGORIES = [
    'general',
    'hate_speech',
    'anti_immigration',
    'anti_lgbtq',
    'anti_feminism',
    'disinformation',
    'conspiracy_theory',
    'call_to_action',
    'nationalism',
    'anti_government',
    'historical_revisionism',
    'political_general'
]

# =============================================================================
# WEB SCRAPING SETTINGS
# =============================================================================

# Web scraping settings
WEB_SCRAPE_MAX_RESULTS = int(os.environ.get('WEB_SCRAPE_MAX_RESULTS', '5'))
WEB_SCRAPE_TIMEOUT = int(os.environ.get('WEB_SCRAPE_TIMEOUT', '10'))

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_db_path() -> str:
    """Get the database path from configuration."""
    return os.environ.get('DATABASE_PATH', str(BASE_DIR / 'accounts.db'))

def get_rate_limit(endpoint_type: str) -> dict:
    """Get rate limit configuration for an endpoint type."""
    return RATE_LIMITS.get(endpoint_type, {'max_requests': 10, 'window_seconds': 60})

def get_pagination_limit(page_type: str) -> int:
    """Get pagination limit for a page type."""
    return DEFAULT_PER_PAGE.get(page_type, 10)

def get_command_timeout(command_type: str) -> int:
    """Get timeout for a command type."""
    return COMMAND_TIMEOUTS.get(command_type, 300)