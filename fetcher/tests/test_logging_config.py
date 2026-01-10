"""Tests for the logging_config module."""

import logging
import pytest
from pathlib import Path

from fetcher.logging_config import setup_logging, get_logger


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_returns_logger_instance(self):
        """Should return a logger instance."""
        logger = setup_logging()
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'fetcher'

    def test_sets_correct_level(self):
        """Should set the specified logging level."""
        logger = setup_logging(level="DEBUG")
        
        assert logger.level == logging.DEBUG
        
        logger = setup_logging(level="ERROR")
        assert logger.level == logging.ERROR

    def test_level_case_insensitive(self):
        """Should handle different case for level."""
        logger = setup_logging(level="warning")
        
        assert logger.level == logging.WARNING

    def test_adds_console_handler_by_default(self):
        """Should add console handler when console=True (default)."""
        logger = setup_logging(console=True)
        
        # Should have at least one handler
        assert len(logger.handlers) >= 1
        
        # Should have a StreamHandler
        stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) >= 1

    def test_no_console_handler_when_disabled(self):
        """Should not add console handler when console=False."""
        logger = setup_logging(console=False)
        
        # Clear any existing handlers first
        logger.handlers.clear()
        logger = setup_logging(console=False)
        
        # Should have no handlers
        assert len(logger.handlers) == 0

    def test_adds_file_handler_when_log_file_specified(self, tmp_path):
        """Should add file handler when log_file is specified."""
        log_file = tmp_path / "test.log"
        
        logger = setup_logging(log_file=str(log_file), console=False)
        
        # Should have a FileHandler
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1

    def test_creates_log_directory_if_not_exists(self, tmp_path):
        """Should create the log file's parent directory."""
        log_file = tmp_path / "logs" / "subdir" / "test.log"
        
        # Directory shouldn't exist yet
        assert not log_file.parent.exists()
        
        setup_logging(log_file=str(log_file), console=False)
        
        # Directory should be created
        assert log_file.parent.exists()

    def test_file_handler_writes_to_file(self, tmp_path):
        """Should actually write log messages to file."""
        log_file = tmp_path / "test.log"
        
        logger = setup_logging(log_file=str(log_file), console=False, level="INFO")
        logger.info("Test message")
        
        # Flush and close handlers
        for handler in logger.handlers:
            handler.flush()
        
        # Check file was written
        content = log_file.read_text()
        assert "Test message" in content

    def test_clears_existing_handlers(self):
        """Should clear existing handlers to avoid duplicates."""
        logger1 = setup_logging()
        initial_handler_count = len(logger1.handlers)
        
        # Call setup again
        logger2 = setup_logging()
        
        # Should have same number of handlers, not doubled
        assert len(logger2.handlers) == initial_handler_count

    def test_both_console_and_file_handlers(self, tmp_path):
        """Should add both console and file handlers when both requested."""
        log_file = tmp_path / "test.log"
        
        logger = setup_logging(log_file=str(log_file), console=True)
        
        stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        
        assert len(stream_handlers) >= 1
        assert len(file_handlers) == 1


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_child_logger(self):
        """Should return a logger with fetcher prefix."""
        logger = get_logger('test_component')
        
        assert logger.name == 'fetcher.test_component'
        assert isinstance(logger, logging.Logger)

    def test_default_name(self):
        """Should use 'fetcher' as default name."""
        logger = get_logger()
        
        assert logger.name == 'fetcher.fetcher'

    def test_different_names_return_different_loggers(self):
        """Should return different logger instances for different names."""
        logger1 = get_logger('component1')
        logger2 = get_logger('component2')
        
        assert logger1 is not logger2
        assert logger1.name != logger2.name

    def test_same_name_returns_same_logger(self):
        """Should return same logger instance for same name."""
        logger1 = get_logger('same_component')
        logger2 = get_logger('same_component')
        
        assert logger1 is logger2
