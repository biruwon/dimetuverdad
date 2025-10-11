"""
Logging configuration for the fetcher module.

Provides centralized logging setup with appropriate levels and formatting
for different components of the tweet fetching system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Set up logging configuration for the fetcher module.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        console: Whether to enable console logging

    Returns:
        Configured logger instance
    """

    # Create logger
    logger = logging.getLogger('fetcher')
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def get_logger(name: str = 'fetcher') -> logging.Logger:
    """
    Get a logger instance for the specified name.

    Args:
        name: Logger name (will be prefixed with 'fetcher.')

    Returns:
        Logger instance
    """
    return logging.getLogger(f'fetcher.{name}')

# Global logger instance
logger = setup_logging()