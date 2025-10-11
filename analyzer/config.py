"""
Configuration management for the analyzer module.
"""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
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
    use_llm: bool = ConfigDefaults.USE_LLM
    model_priority: str = ConfigDefaults.MODEL_PRIORITY
    verbose: bool = ConfigDefaults.VERBOSE

    # Performance settings
    max_retries: int = ConfigDefaults.MAX_RETRIES
    retry_delay: int = ConfigDefaults.RETRY_DELAY
    analysis_timeout: float = ConfigDefaults.ANALYSIS_TIMEOUT
    database_timeout: float = ConfigDefaults.DATABASE_TIMEOUT
    download_timeout: float = ConfigDefaults.DOWNLOAD_TIMEOUT
    request_timeout: float = ConfigDefaults.REQUEST_TIMEOUT

    # External dependencies (for dependency injection)
    db_path: Optional[str] = None
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

        if self.database_timeout <= 0:
            raise ValueError("database_timeout must be positive")

        if self.download_timeout <= 0:
            raise ValueError("download_timeout must be positive")

        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")

        if self.model_priority not in ["fast", "balanced", "quality"]:
            raise ValueError("model_priority must be one of: fast, balanced, quality")

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'AnalyzerConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'use_llm': self.use_llm,
            'model_priority': self.model_priority,
            'verbose': self.verbose,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'analysis_timeout': self.analysis_timeout,
            'database_timeout': self.database_timeout,
            'download_timeout': self.download_timeout,
            'request_timeout': self.request_timeout,
            'db_path': self.db_path,
            # Don't include llm_pipeline in dict representation
        }

    def __str__(self) -> str:
        """String representation of configuration."""
        config_dict = self.to_dict()
        return f"AnalyzerConfig({config_dict})"