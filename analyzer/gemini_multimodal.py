#!/usr/bin/env python3
"""
Gemini Multimodal Analysis Module for dimetuverdad.

Provides unified analysis for images and video using Google Gemini 2.5 Flash.
Handles media download, upload to Gemini, and comprehensive political content analysis.
"""

import os
import tempfile
import time
import requests
import logging
from typing import Optional, Tuple, List, Protocol, Dict, Any
from enum import Enum
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import psutil

# Simple warning suppression for Google Cloud libraries
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

import google.generativeai as genai
from dotenv import load_dotenv

# Import prompt generation
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyzer.prompts import EnhancedPromptGenerator

# Load environment variables
load_dotenv()

# Set up structured logging
logger = logging.getLogger(__name__)

# Dependency Injection Interfaces
class HTTPClientProtocol(Protocol):
    """Protocol for HTTP client operations."""
    def get(self, url: str, timeout: float = 30.0, **kwargs) -> requests.Response:
        """Make a GET request."""
        ...

class AsyncHTTPClientProtocol(Protocol):
    """Protocol for async HTTP client operations."""
    async def get(self, url: str, timeout: float = 30.0, **kwargs) -> aiohttp.ClientResponse:
        """Make an async GET request."""
        ...

class FileSystemProtocol(Protocol):
    """Protocol for file system operations."""
    def create_temp_file(self, suffix: str = "") -> str:
        """Create a temporary file and return its path."""
        ...

    def remove_file(self, file_path: str) -> None:
        """Remove a file."""
        ...

class ResourceMonitorProtocol(Protocol):
    """Protocol for resource monitoring."""
    def check_memory_usage(self) -> float:
        """Check current memory usage in MB."""
        ...

    def check_file_size(self, file_path: str) -> int:
        """Check file size in bytes."""
        ...

class MetricsCollectorProtocol(Protocol):
    """Protocol for metrics collection."""
    def record_operation(self, operation: str, duration: float, success: bool, **kwargs) -> None:
        """Record an operation with metrics."""
        ...

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        ...

# Configuration Classes
@dataclass
class GeminiMultimodalConfig:
    """Configuration for GeminiMultimodal analyzer."""
    # Required settings
    api_key: str

    # Model settings
    model_name: str = "gemini-2.5-pro"
    model_priority: List[str] = None

    # Performance settings
    max_retries: int = 3
    retry_delay: float = 1.0
    request_timeout: float = 30.0
    analysis_timeout: float = 180.0  # 3 minutes max for analysis
    download_timeout: float = 180.0  # 3 minutes max for download

    # File size limits (in MB)
    max_video_size_mb: float = 100.0  # Videos up to 100MB
    max_image_size_mb: float = 10.0   # Images up to 10MB

    # Operational settings
    cleanup_delay: float = 5.0
    log_level: str = "INFO"

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()

        # Set default model priority if not provided
        if self.model_priority is None:
            self.model_priority = [
                'gemini-2.5-pro',
                'gemini-2.5-flash',
                'gemini-2.5-flash-lite',
                'gemini-2.0-flash-exp',
                'gemini-2.0-flash-lite',
                'gemini-2.0-flash'
            ]

    def _validate_config(self):
        """Validate configuration values."""
        if not self.api_key or not self.api_key.strip():
            raise ValueError("API key is required and cannot be empty")

        if self.analysis_timeout <= 0:
            raise ValueError("analysis_timeout must be positive")

        if self.download_timeout <= 0:
            raise ValueError("download_timeout must be positive")

        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")

        if self.max_video_size_mb <= 0:
            raise ValueError("max_video_size_mb must be positive")

        if self.max_image_size_mb <= 0:
            raise ValueError("max_image_size_mb must be positive")

        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(f"log_level must be one of: {', '.join(valid_log_levels)}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            'api_key': '***REDACTED***',  # Don't expose API key
            'model_name': self.model_name,
            'model_priority': self.model_priority,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'request_timeout': self.request_timeout,
            'analysis_timeout': self.analysis_timeout,
            'download_timeout': self.download_timeout,
            'max_video_size_mb': self.max_video_size_mb,
            'max_image_size_mb': self.max_image_size_mb,
            'cleanup_delay': self.cleanup_delay,
            'log_level': self.log_level
        }

    def __str__(self) -> str:
        """String representation of configuration (without sensitive data)."""
        config_dict = self.to_dict()
        return f"GeminiMultimodalConfig({config_dict})"

@dataclass
class DependencyContainer:
    """Container for all dependencies."""
    http_client: HTTPClientProtocol
    file_system: FileSystemProtocol
    resource_monitor: ResourceMonitorProtocol
    metrics_collector: MetricsCollectorProtocol
    config: GeminiMultimodalConfig
    async_http_client: Optional[AsyncHTTPClientProtocol] = None

# Concrete Implementations
class RequestsHTTPClient:
    """Concrete HTTP client using requests library."""
    def get(self, url: str, timeout: float = 30.0, **kwargs) -> requests.Response:
        return requests.get(url, timeout=timeout, **kwargs)

class AsyncAIOHTTPClient:
    """Concrete async HTTP client using aiohttp library."""
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get(self, url: str, timeout: float = 30.0, **kwargs) -> aiohttp.ClientResponse:
        if not self.session:
            raise RuntimeError("AsyncHTTPClient must be used as async context manager")
        return await self.session.get(url, timeout=aiohttp.ClientTimeout(total=timeout), **kwargs)

class TempFileSystem:
    """Concrete file system using tempfile module."""
    def create_temp_file(self, suffix: str = "") -> str:
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)  # Close the file descriptor, keep the path
        return path

    def remove_file(self, file_path: str) -> None:
        if os.path.exists(file_path):
            os.remove(file_path)

class DefaultResourceMonitor(ResourceMonitorProtocol):
    """Default resource monitor implementation."""
    def check_memory_usage(self) -> float:
        """Check current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # Convert to MB
        except ImportError:
            return 0.0  # Return 0 if psutil not available

    def check_file_size(self, file_path: str) -> int:
        """Check file size in bytes."""
        return os.path.getsize(file_path) if os.path.exists(file_path) else 0

class DefaultMetricsCollector(MetricsCollectorProtocol):
    """Default metrics collector implementation."""
    def __init__(self):
        self.metrics = {
            "operations": [],
            "errors": {},
            "performance": {}
        }

    def record_operation(self, operation: str, duration: float, success: bool, **kwargs) -> None:
        """Record an operation with metrics."""
        self.metrics["operations"].append({
            "operation": operation,
            "duration": duration,
            "success": success,
            "timestamp": time.time(),
            **kwargs
        })

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return self.metrics.copy()


class ErrorCategory(Enum):
    """Enumeration of different error categories for better error handling."""
    NETWORK_ERROR = "network_error"
    AUTHENTICATION_ERROR = "authentication_error"
    QUOTA_ERROR = "quota_error"
    MODEL_ERROR = "model_error"
    TIMEOUT_ERROR = "timeout_error"
    MEDIA_ERROR = "media_error"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN_ERROR = "unknown_error"


class AnalysisError(Exception):
    """Custom exception for analysis errors with categorization."""

    def __init__(self, message: str, category: ErrorCategory, recoverable: bool = False, retry_delay: Optional[int] = None):
        super().__init__(message)
        self.category = category
        self.recoverable = recoverable
        self.retry_delay = retry_delay
        self.message = message

    def __str__(self):
        return f"[{self.category.value}] {self.message}"


def classify_error(error: Exception, context: str = "") -> AnalysisError:
    """
    Classify an exception into a specific error category with recovery information.

    Args:
        error: The exception to classify
        context: Additional context about where the error occurred

    Returns:
        AnalysisError with appropriate category and recovery information
    """
    error_str = str(error).lower()

    # Network-related errors
    if any(pattern in error_str for pattern in ['connection', 'timeout', 'network', 'dns', 'ssl']):
        if 'timeout' in error_str:
            return AnalysisError(
                f"Network timeout in {context}: {error}",
                ErrorCategory.TIMEOUT_ERROR,
                recoverable=True,
                retry_delay=5
            )
        else:
            return AnalysisError(
                f"Network error in {context}: {error}",
                ErrorCategory.NETWORK_ERROR,
                recoverable=True,
                retry_delay=2
            )

    # Authentication errors
    elif any(pattern in error_str for pattern in ['unauthorized', 'forbidden', 'authentication', 'api key', 'credentials']):
        return AnalysisError(
            f"Authentication error in {context}: {error}",
            ErrorCategory.AUTHENTICATION_ERROR,
            recoverable=False
        )

    # Quota/rate limit errors
    elif any(pattern in error_str for pattern in ['quota', 'rate limit', 'exceeded', 'limit']):
        return AnalysisError(
            f"Quota exceeded in {context}: {error}",
            ErrorCategory.QUOTA_ERROR,
            recoverable=True,
            retry_delay=60  # Wait 1 minute for quota reset
        )

    # Model availability errors
    elif any(pattern in error_str for pattern in ['model not found', 'model not available', 'unsupported model']):
        return AnalysisError(
            f"Model error in {context}: {error}",
            ErrorCategory.MODEL_ERROR,
            recoverable=True,
            retry_delay=10
        )

    # Media processing errors
    elif any(pattern in error_str for pattern in ['media', 'file', 'upload', 'processing']):
        return AnalysisError(
            f"Media processing error in {context}: {error}",
            ErrorCategory.MEDIA_ERROR,
            recoverable=False
        )

    # Configuration errors
    elif any(pattern in error_str for pattern in ['configuration', 'config', 'environment']):
        return AnalysisError(
            f"Configuration error in {context}: {error}",
            ErrorCategory.CONFIGURATION_ERROR,
            recoverable=False
        )

    # Default to unknown error
    return AnalysisError(
        f"Unknown error in {context}: {error}",
        ErrorCategory.UNKNOWN_ERROR,
        recoverable=False
    )


class GeminiMultimodal:
    """
    Main class for Gemini multimodal analysis with dependency injection.

    This class encapsulates all multimodal analysis functionality and allows
    for dependency injection to improve testability and configurability.
    """

    def __init__(self, dependencies: Optional[DependencyContainer] = None):
        """
        Initialize GeminiMultimodal with dependencies.

        Args:
            dependencies: Dependency container with all required components.
                         If None, uses default implementations.
        """
        if dependencies is None:
            # Create default dependencies
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable is required for default GeminiMultimodal configuration")
            config = GeminiMultimodalConfig(api_key=api_key)
            dependencies = DependencyContainer(
                http_client=RequestsHTTPClient(),
                file_system=TempFileSystem(),
                resource_monitor=DefaultResourceMonitor(),
                metrics_collector=DefaultMetricsCollector(),
                config=config
            )

        self.dependencies = dependencies
        self.logger = logging.getLogger(__name__)

        # Configure logging level
        self.logger.setLevel(getattr(logging, dependencies.config.log_level.upper()))

        # Track rate-limited models to avoid repeated failures
        self.rate_limited_models = set()

    def _get_available_models(self) -> List[str]:
        """
        Get list of available models, excluding currently rate-limited ones.

        Returns:
            List of model names that are not currently rate-limited
        """
        available_models = []

        for model_name in self.dependencies.config.model_priority:
            if model_name not in self.rate_limited_models:
                available_models.append(model_name)
            else:
                self.logger.debug(f"⏳ Skipping rate-limited model {model_name}")

        return available_models

    def _mark_model_rate_limited(self, model_name: str):
        """
        Mark a model as rate-limited.

        Args:
            model_name: Name of the model that was rate-limited
        """
        self.rate_limited_models.add(model_name)
        self.logger.warning(f"🚫 Marked model as rate-limited: {model_name}")

    def _mark_model_success(self, model_name: str):
        """
        Mark a model as successful, clearing any rate-limit status.

        Args:
            model_name: Name of the model that succeeded
        """
        if model_name in self.rate_limited_models:
            self.rate_limited_models.discard(model_name)
            self.logger.info(f"✅ Model recovered from rate-limit: {model_name}")

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """
        Get current rate limit status for all models.

        Returns:
            Dict with rate limit information
        """
        status = {
            "rate_limited_models": list(self.rate_limited_models),
            "available_models": self._get_available_models()
        }

        return status

    def reset_rate_limits(self):
        """Reset all rate limit tracking."""
        self.rate_limited_models.clear()
        self.logger.info("🔄 Reset all rate limit tracking")

    def analyze_multimodal_content(self, media_urls: List[str], text_content: str) -> Tuple[Optional[str], float]:
        """
        Analyze multimodal content (images/videos + text) using Gemini models with fallback.

        Args:
            media_urls: List of media URLs to analyze
            text_content: Text content accompanying the media

        Returns:
            Tuple of (analysis_result, analysis_time_seconds)
            Returns (None, error_time) if analysis failed
        """
        analysis_start_time = time.time()

        # Record analysis start
        self.dependencies.metrics_collector.record_operation(
            "analysis_start", 0.0, True, media_count=len(media_urls)
        )

        if not media_urls:
            self.logger.warning("No media URLs provided")
            self.dependencies.metrics_collector.record_operation(
                "analysis_failure", time.time() - analysis_start_time, False,
                error_category="no_media_urls"
            )
            return None, time.time() - analysis_start_time

        # Filter out unwanted media
        filtered_urls = []
        for url in media_urls:
            if 'profile_images' in url or 'card_img' in url:
                continue
            filtered_urls.append(url)

        if not filtered_urls:
            self.logger.warning("No valid media URLs found after filtering")
            self.dependencies.metrics_collector.record_operation(
                "analysis_failure", time.time() - analysis_start_time, False,
                error_category="no_valid_media_urls"
            )
            return None, time.time() - analysis_start_time

        # Select media URL with priority logic
        media_url = self._select_media_url(filtered_urls)
        is_video = self._is_video_url(media_url)

        # Download media
        media_path = self._download_media(media_url, is_video)
        if not media_path:
            self.dependencies.metrics_collector.record_operation(
                "analysis_failure", time.time() - analysis_start_time, False,
                error_category="media_download_failed"
            )
            return None, time.time() - analysis_start_time

        try:
            # Try each available model in priority order
            available_models = self._get_available_models()
            if not available_models:
                self.logger.error("💔 No available models (all rate-limited)")
                return None, time.time() - analysis_start_time

            for model_name in available_models:
                # Check timeout
                if time.time() - analysis_start_time > self.dependencies.config.analysis_timeout:
                    self.logger.error("❌ Analysis timeout exceeded")
                    return None, time.time() - analysis_start_time

                result = self._try_model_analysis(model_name, media_path, media_url, text_content, is_video)
                if result:
                    # Mark model as successful
                    self._mark_model_success(model_name)
                    total_time = time.time() - analysis_start_time
                    self.dependencies.metrics_collector.record_operation(
                        "analysis_success", total_time, True,
                        model_used=model_name, media_type="video" if is_video else "image"
                    )
                    return result, total_time

            # All available models failed
            self.logger.error("💔 All available models failed")
            return None, time.time() - analysis_start_time

        finally:
            # Clean up
            self._cleanup_file(media_path)

    async def analyze_multimodal_content_async(self, media_urls: List[str], text_content: str) -> Tuple[Optional[str], float]:
        """
        Async version of analyze_multimodal_content with concurrent processing.

        This method can process multiple media URLs concurrently for better performance.

        Args:
            media_urls: List of media URLs to analyze
            text_content: Text content accompanying the media

        Returns:
            Tuple of (analysis_result, analysis_time_seconds)
            Returns (None, error_time) if analysis failed
        """
        analysis_start_time = time.time()

        # Record analysis start
        self.dependencies.metrics_collector.record_operation(
            "async_analysis_start", 0.0, True, media_count=len(media_urls)
        )

        if not media_urls:
            self.logger.warning("No media URLs provided")
            self.dependencies.metrics_collector.record_operation(
                "async_analysis_failure", time.time() - analysis_start_time, False,
                error_category="no_media_urls"
            )
            return None, time.time() - analysis_start_time

        # Filter out unwanted media
        filtered_urls = []
        for url in media_urls:
            if 'profile_images' in url or 'card_img' in url:
                continue
            filtered_urls.append(url)

        if not filtered_urls:
            self.logger.warning("No valid media URLs found after filtering")
            self.dependencies.metrics_collector.record_operation(
                "async_analysis_failure", time.time() - analysis_start_time, False,
                error_category="no_valid_media_urls"
            )
            return None, time.time() - analysis_start_time

        # For async processing, we can download multiple media files concurrently
        # and analyze them, but for now we'll focus on the primary media file
        media_url = self._select_media_url(filtered_urls)
        is_video = self._is_video_url(media_url)

        # Download media asynchronously
        media_path = await self._download_media_async(media_url, is_video)
        if not media_path:
            self.dependencies.metrics_collector.record_operation(
                "async_analysis_failure", time.time() - analysis_start_time, False,
                error_category="media_download_failed"
            )
            return None, time.time() - analysis_start_time

        try:
            # Try each available model in priority order
            available_models = self._get_available_models()
            if not available_models:
                self.logger.error("💔 No available models (all rate-limited)")
                return None, time.time() - analysis_start_time

            for model_name in available_models:
                # Check timeout
                if time.time() - analysis_start_time > self.dependencies.config.analysis_timeout:
                    self.logger.error("❌ Analysis timeout exceeded")
                    return None, time.time() - analysis_start_time

                result = await self._try_model_analysis_async(model_name, media_path, media_url, text_content, is_video)
                if result:
                    # Mark model as successful
                    self._mark_model_success(model_name)
                    total_time = time.time() - analysis_start_time
                    self.dependencies.metrics_collector.record_operation(
                        "async_analysis_success", total_time, True,
                        model_used=model_name, media_type="video" if is_video else "image"
                    )
                    return result, total_time

            # All available models failed
            self.logger.error("💔 All available async models failed")
            return None, time.time() - analysis_start_time

        finally:
            # Clean up
            self._cleanup_file(media_path)

    def _select_media_url(self, media_urls: List[str]) -> str:
        """Select the best media URL based on priority logic.

        Priority order:
        1. MP4 files (highest priority)
        2. M3U8 files
        3. Video URLs (containing 'video' or specific patterns)
        4. Image URLs (fallback)

        Args:
            media_urls: List of media URLs to choose from

        Returns:
            Selected media URL
        """
        if not media_urls:
            raise ValueError("No media URLs provided")

        # Priority 1: MP4 files
        for url in media_urls:
            if '.mp4' in url.lower():
                return url

        # Priority 2: M3U8 files
        for url in media_urls:
            if '.m3u8' in url.lower():
                return url

        # Priority 3: Video URLs (containing video patterns)
        for url in media_urls:
            if any(pattern in url.lower() for pattern in ['video', 'vid/', 'amplify_video']):
                return url

        # Priority 4: First available URL (images)
        return media_urls[0]

    def _is_video_url(self, url: str) -> bool:
        """Check if URL points to video content."""
        return any(pattern in url.lower() for pattern in ['.mp4', '.m3u8', '.mov', '.avi', '.webm', 'video'])

    def _download_media(self, media_url: str, is_video: bool) -> Optional[str]:
        """Download media using injected dependencies."""
        try:
            self.logger.info(f"🔗 Downloading media: {media_url}")
            download_start = time.time()

            # Use injected HTTP client
            response = self.dependencies.http_client.get(
                media_url,
                timeout=self.dependencies.config.download_timeout
            )
            response.raise_for_status()

            # Create temp file
            suffix = '.mp4' if is_video else '.jpg'
            temp_path = self.dependencies.file_system.create_temp_file(suffix)

            # Download content
            downloaded = 0
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

            # Verify file
            file_size = self.dependencies.resource_monitor.check_file_size(temp_path)
            if file_size == 0:
                self.logger.error("❌ Downloaded file is empty")
                self.dependencies.file_system.remove_file(temp_path)
                return None

            # Check file size limits
            file_size_mb = file_size / (1024 * 1024)  # Convert to MB
            if is_video and file_size_mb > self.dependencies.config.max_video_size_mb:
                self.logger.error(f"❌ Video file too large: {file_size_mb:.2f}MB (max: {self.dependencies.config.max_video_size_mb}MB)")
                self.dependencies.file_system.remove_file(temp_path)
                return None
            elif not is_video and file_size_mb > self.dependencies.config.max_image_size_mb:
                self.logger.error(f"❌ Image file too large: {file_size_mb:.2f}MB (max: {self.dependencies.config.max_image_size_mb}MB)")
                self.dependencies.file_system.remove_file(temp_path)
                return None

            download_time = time.time() - download_start
            self.logger.info(f"✅ Downloaded: {temp_path} ({file_size_mb:.2f}MB) in {download_time:.2f}s")

            self.dependencies.metrics_collector.record_operation(
                "download_success", download_time, True,
                file_size=file_size, media_type="video" if is_video else "image"
            )

            return temp_path

        except Exception as e:
            download_time = time.time() - download_start
            error = classify_error(e, "media download")
            self.logger.error(f"❌ {error}")

            self.dependencies.metrics_collector.record_operation(
                "download_failure", download_time, False,
                error_category=error.category.value
            )
            return None

    async def _download_media_async(self, media_url: str, is_video: bool) -> Optional[str]:
        """Download media asynchronously using injected async HTTP client."""
        if not self.dependencies.async_http_client:
            # Fallback to sync download in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                return await loop.run_in_executor(executor, self._download_media, media_url, is_video)

        try:
            self.logger.info(f"🔗 Downloading media async: {media_url}")
            download_start = time.time()

            # Use injected async HTTP client
            async with self.dependencies.async_http_client as client:
                response = await client.get(
                    media_url,
                    timeout=self.dependencies.config.download_timeout
                )
                response.raise_for_status()

                # Create temp file
                suffix = '.mp4' if is_video else '.jpg'
                temp_path = self.dependencies.file_system.create_temp_file(suffix)

                # Download content
                downloaded = 0
                with open(temp_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                # Verify file
                file_size = self.dependencies.resource_monitor.check_file_size(temp_path)
                if file_size == 0:
                    self.logger.error("❌ Downloaded file is empty")
                    self.dependencies.file_system.remove_file(temp_path)
                    return None

                # Check file size limits
                file_size_mb = file_size / (1024 * 1024)  # Convert to MB
                if is_video and file_size_mb > self.dependencies.config.max_video_size_mb:
                    self.logger.error(f"❌ Video file too large: {file_size_mb:.2f}MB (max: {self.dependencies.config.max_video_size_mb}MB)")
                    self.dependencies.file_system.remove_file(temp_path)
                    return None
                elif not is_video and file_size_mb > self.dependencies.config.max_image_size_mb:
                    self.logger.error(f"❌ Image file too large: {file_size_mb:.2f}MB (max: {self.dependencies.config.max_image_size_mb}MB)")
                    self.dependencies.file_system.remove_file(temp_path)
                    return None

                download_time = time.time() - download_start
                self.logger.info(f"✅ Downloaded async: {temp_path} ({file_size_mb:.2f}MB) in {download_time:.2f}s")

                self.dependencies.metrics_collector.record_operation(
                    "async_download_success", download_time, True,
                    file_size=file_size, media_type="video" if is_video else "image"
                )

                return temp_path

        except Exception as e:
            download_time = time.time() - download_start
            error = classify_error(e, "async media download")
            self.logger.error(f"❌ {error}")

            self.dependencies.metrics_collector.record_operation(
                "async_download_failure", download_time, False,
                error_category=error.category.value
            )
            return None

    def _try_model_analysis(self, model_name: str, media_path: str, media_url: str,
                          text_content: str, is_video: bool) -> Optional[str]:
        """Try analysis with a specific model."""
        try:
            self.logger.info(f"🔄 Trying model: {model_name}")

            # Get Gemini client
            model, error = self._get_gemini_client(model_name)
            if not model:
                self.logger.error(f"❌ Failed to initialize {model_name}: {error}")
                return None

            # Upload media
            media_file = self._upload_media_to_gemini(model, media_path, media_url)
            if not media_file:
                return None

            # Generate analysis
            prompt = EnhancedPromptGenerator.build_gemini_analysis_prompt(text_content, is_video)

            # Use ThreadPoolExecutor with timeout instead of signal-based timeout
            # to avoid "signal only works in main thread" error
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(model.generate_content, [media_file, prompt])
                try:
                    response = future.result(timeout=60.0)
                    self.logger.info(f"✅ {model_name} analysis completed")
                    return response.text
                except TimeoutError:
                    self.logger.warning(f"⏰ {model_name} timed out")
                    future.cancel()
                    return None

        except Exception as e:
            error = classify_error(e, f"{model_name} analysis")

            # Check if this is a rate limiting error
            if error.category == ErrorCategory.QUOTA_ERROR:
                self._mark_model_rate_limited(model_name)

            self.logger.error(f"❌ {error}")
            return None

    async def _try_model_analysis_async(self, model_name: str, media_path: str, media_url: str,
                                      text_content: str, is_video: bool) -> Optional[str]:
        """Try analysis with a specific model asynchronously."""
        try:
            self.logger.info(f"🔄 Trying model async: {model_name}")

            # Get Gemini client (run in thread pool since genai is sync)
            loop = asyncio.get_event_loop()
            model, error = await loop.run_in_executor(
                None, self._get_gemini_client, model_name
            )

            if not model:
                self.logger.error(f"❌ Failed to initialize {model_name}: {error}")
                return None

            # Upload media (run in thread pool)
            media_file = await loop.run_in_executor(
                None, self._upload_media_to_gemini, model, media_path, media_url
            )
            if not media_file:
                return None

            # Generate analysis (run in thread pool with timeout)
            prompt = EnhancedPromptGenerator.build_gemini_analysis_prompt(text_content, is_video)

            try:
                response = await asyncio.wait_for(
                    loop.run_in_executor(None, model.generate_content, [media_file, prompt]),
                    timeout=60.0
                )

                self.logger.info(f"✅ {model_name} async analysis completed")
                return response.text

            except asyncio.TimeoutError:
                self.logger.warning(f"⏰ {model_name} async timed out")
                return None

        except Exception as e:
            error = classify_error(e, f"async {model_name} analysis")

            # Check if this is a rate limiting error
            if error.category == ErrorCategory.QUOTA_ERROR:
                self._mark_model_rate_limited(model_name)

            self.logger.error(f"❌ {error}")
            return None

    def _get_gemini_client(self, model_name: str) -> Tuple[Optional[genai.GenerativeModel], Optional[str]]:
        """Get initialized Gemini client."""
        try:
            genai.configure(api_key=self.dependencies.config.api_key)
            model = genai.GenerativeModel(model_name)
            return model, None
        except Exception as e:
            error = classify_error(e, f"client initialization ({model_name})")
            return None, str(error)

    def _upload_media_to_gemini(self, client: genai.GenerativeModel, media_path: str, media_url: str) -> Optional[genai.types.File]:
        """Upload media to Gemini."""
        try:
            self.logger.info("📤 Uploading media to Gemini...")
            media_file = genai.upload_file(media_path)

            # Wait for processing
            max_wait = 60
            start_wait = time.time()

            while media_file.state.name == "PROCESSING" and (time.time() - start_wait) < max_wait:
                time.sleep(1)
                media_file = genai.get_file(media_file.name)

            if media_file.state.name != "ACTIVE":
                self.logger.error(f"❌ Media processing failed: {media_file.state.name}")
                return None

            self.logger.info("✅ Media uploaded successfully")
            return media_file

        except Exception as e:
            error = classify_error(e, "media upload")
            self.logger.error(f"❌ {error}")
            return None

    async def analyze_multimodal_batch_async(self, media_url_text_pairs: List[Tuple[List[str], str]],
                                           max_concurrent: int = 3) -> List[Tuple[Optional[str], float]]:
        """
        Analyze multiple multimodal content items concurrently.

        Args:
            media_url_text_pairs: List of (media_urls, text_content) tuples
            max_concurrent: Maximum number of concurrent analyses

        Returns:
            List of (result, time) tuples in the same order as input
        """
        self.logger.info(f"🔄 Starting batch analysis of {len(media_url_text_pairs)} items (max concurrent: {max_concurrent})")

        async def analyze_single(item: Tuple[List[str], str]) -> Tuple[Optional[str], float]:
            """Analyze a single item."""
            media_urls, text_content = item
            return await self.analyze_multimodal_content_async(media_urls, text_content)

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_with_semaphore(item: Tuple[List[str], str]) -> Tuple[Optional[str], float]:
            async with semaphore:
                return await analyze_single(item)

        # Run all analyses concurrently with semaphore
        tasks = [analyze_with_semaphore(item) for item in media_url_text_pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that occurred
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"❌ Batch item {i} failed with exception: {result}")
                final_results.append((None, 0.0))
            else:
                final_results.append(result)

        self.logger.info(f"✅ Batch analysis completed: {len(final_results)} results")
        return final_results

    def _cleanup_file(self, file_path: str):
        """Clean up temporary file."""
        try:
            self.dependencies.file_system.remove_file(file_path)
            self.logger.debug(f"🧹 Cleaned up: {file_path}")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not clean up {file_path}: {e}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return self.dependencies.metrics_collector.get_summary()

    def log_metrics_summary(self):
        """Log metrics summary."""
        summary = self.get_metrics_summary()
        self.logger.info("📊 Multimodal Analysis Metrics:")
        for key, value in summary.items():
            self.logger.info(f"   {key}: {value}")

    def reset_metrics(self):
        """Reset metrics."""
        # Create new metrics collector instance
        self.dependencies.metrics_collector = DefaultMetricsCollector()
        self.logger.info("📊 Metrics reset")
