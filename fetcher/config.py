"""
Configuration management for the fetcher module.

Centralizes all configuration values, timeouts, limits, and settings
used throughout the tweet fetching system.
"""

import os
from typing import Dict, List, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class FetcherConfig:
    """Configuration class for tweet fetching operations."""

    # Authentication
    username: str = os.getenv("X_USERNAME", "")
    password: str = os.getenv("X_PASSWORD", "")
    email_or_phone: str = os.getenv("X_EMAIL_OR_PHONE", "")

    # Browser settings
    # Default headless=True for production (faster), override with --visible flag
    headless: bool = os.getenv("FETCHER_HEADLESS", "true").lower() in ("1", "true", "yes")
    slow_mo: int = int(os.getenv("FETCHER_SLOW_MO", "50"))
    viewport_width: int = 1280
    viewport_height: int = 720
    user_agents: List[str] = None

    # Timeouts (in milliseconds)
    page_load_timeout: int = 30000
    element_wait_timeout: int = 15000
    login_verification_timeout: int = 15000

    # Delays (in seconds)
    min_human_delay: float = 1.0
    max_human_delay: float = 3.0
    recovery_delay_min: float = 3.0
    recovery_delay_max: float = 6.0
    session_refresh_delay_min: float = 3.0
    session_refresh_delay_max: float = 6.0

    # Collection limits
    max_consecutive_empty_scrolls: int = 15
    max_consecutive_existing_tweets: int = 10
    max_recovery_attempts: int = 5
    max_session_retries: int = 5
    max_sessions: int = 10

    # Thread detection tuning
    thread_detect_interval: int = 5  # number of scroll cycles between inline thread detection
    thread_collect_timeout_seconds: int = 600  # max time (10 min) for collecting a single thread, generous to avoid cutting off long threads

    # Feature flags
    # Thread collection is enabled by default. Set FETCHER_COLLECT_THREADS=false to disable.
    collect_threads: bool = os.getenv("FETCHER_COLLECT_THREADS", "true").lower() in ("1","true","yes")
    
    # Performance optimization: use adaptive scroll delays (wait for content instead of fixed delays)
    use_adaptive_scroll: bool = os.getenv("FETCHER_ADAPTIVE_SCROLL", "true").lower() in ("1", "true", "yes")

    # Session management
    session_size: int = 800
    large_collection_threshold: int = 800

    # Scrolling parameters
    scroll_amounts: List[int] = None
    aggressive_scroll_multiplier: float = 2.0

    # Database settings
    db_timeout: float = 10.0

    def __post_init__(self):
        """Initialize default values that depend on other attributes."""
        if self.user_agents is None:
            self.user_agents = [
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
            ]

        if self.scroll_amounts is None:
            self.scroll_amounts = [800, 1000, 600, 1200]

# Global configuration instance
config = FetcherConfig()

# Default target handles (focusing on Spanish far-right accounts)
DEFAULT_HANDLES = [
    "vox_es",
    "Santi_ABASCAL",
    "eduardomenoni",
    "IdiazAyuso",
    "CapitanBitcoin",
    "vitoquiles",
    "wallstwolverine",
    "WillyTolerdoo",
    "Agenda2030_",
    "Doct_Tricornio",
    "LosMeconios",
    "infovlogger36",
    "daniesdsk"
]

def get_config() -> FetcherConfig:
    """Get the global configuration instance."""
    return config

def update_config(**kwargs) -> None:
    """Update configuration values."""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")