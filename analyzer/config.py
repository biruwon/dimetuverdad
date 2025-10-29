"""
Configuration management for the analyzer module.
"""

from dataclasses import dataclass
from typing import Optional
from .llm_models import EnhancedLLMPipeline
from .constants import ConfigDefaults


@dataclass
class AnalyzerConfig:
    """
    Configuration for the Analyzer and its components.

    Centralizes all configuration options to improve maintainability
    and make the system more testable.
    """

    # Core settings
    verbose: bool = ConfigDefaults.VERBOSE

    # External analysis settings
    enable_external_analysis: bool = False  # Auto-trigger for non-general/political_general categories
    external_analysis_timeout: float = 120.0  # Timeout for external analysis (Gemini)

    # Performance settings
    max_retries: int = ConfigDefaults.MAX_RETRIES
    retry_delay: int = ConfigDefaults.RETRY_DELAY
    analysis_timeout: float = ConfigDefaults.ANALYSIS_TIMEOUT
    database_timeout: float = ConfigDefaults.DATABASE_TIMEOUT
    download_timeout: float = ConfigDefaults.DOWNLOAD_TIMEOUT
    request_timeout: float = ConfigDefaults.REQUEST_TIMEOUT
    max_concurrency: int = ConfigDefaults.MAX_CONCURRENCY
    max_llm_concurrency: int = ConfigDefaults.MAX_LLM_CONCURRENCY

    # External dependencies (for dependency injection)
    llm_pipeline: Optional['EnhancedLLMPipeline'] = None  # Forward reference to avoid circular import

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self):
        """Validate configuration values."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        if self.retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")

        if self.analysis_timeout <= 0:
            raise ValueError("analysis_timeout must be positive")
        
        if self.external_analysis_timeout <= 0:
            raise ValueError("external_analysis_timeout must be positive")

        if self.database_timeout <= 0:
            raise ValueError("database_timeout must be positive")

        if self.download_timeout <= 0:
            raise ValueError("download_timeout must be positive")

        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")

        if self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        if self.max_llm_concurrency <= 0:
            raise ValueError("max_llm_concurrency must be positive")

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'AnalyzerConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'verbose': self.verbose,
            'enable_external_analysis': self.enable_external_analysis,
            'external_analysis_timeout': self.external_analysis_timeout,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'analysis_timeout': self.analysis_timeout,
            'database_timeout': self.database_timeout,
            'download_timeout': self.download_timeout,
            'request_timeout': self.request_timeout,
            'max_concurrency': self.max_concurrency,
            'max_llm_concurrency': self.max_llm_concurrency,
            # Don't include llm_pipeline in dict representation
        }

    def __str__(self) -> str:
        """String representation of configuration."""
        config_dict = self.to_dict()
        return f"AnalyzerConfig({config_dict})"