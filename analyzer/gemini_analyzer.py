"""
Gemini multimodal analyzer for the dimetuverdad system.

Provides orchestration for multimodal content analysis using Gemini models
with proper error handling, rate limiting, and metrics collection.
"""

import os
import time
import asyncio
import logging
import tempfile
from typing import Optional, Tuple, List, Dict, Any
import requests
from .error_handler import classify_error, ErrorCategory
from .gemini_client import GeminiClient
from .media_processor import (
    MediaProcessor,
    HTTPClientProtocol,
    FileSystemProtocol,
    ResourceMonitorProtocol
)

# Import prompt generation
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyzer.prompts import EnhancedPromptGenerator

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class RequestsHTTPClient(HTTPClientProtocol):
    """Default HTTP client implementation using requests."""
    def get(self, url: str, timeout: float = 30.0, **kwargs) -> requests.Response:
        return requests.get(url, timeout=timeout, **kwargs)


class TempFileSystem(FileSystemProtocol):
    """Default file system implementation using tempfile."""
    def create_temp_file(self, suffix: str = "") -> str:
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)  # Close the file descriptor, keep the path
        return path
    
    def remove_file(self, file_path: str) -> None:
        if os.path.exists(file_path):
            os.remove(file_path)


class DefaultResourceMonitor(ResourceMonitorProtocol):
    """Default resource monitor implementation."""
    def check_file_size(self, file_path: str) -> int:
        return os.path.getsize(file_path)


class MetricsCollectorProtocol:
    """Protocol for metrics collection."""
    def record_operation(self, operation: str, duration: float, success: bool, **kwargs) -> None:
        """Record an operation with metrics."""
        ...

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        ...


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


class GeminiAnalyzer:
    """
    Main orchestrator for Gemini multimodal analysis.

    Handles model management, rate limiting, analysis orchestration,
    and metrics collection.
    """

    def __init__(self,
                 api_key: str,
                 model_priority: Optional[List[str]] = None,
                 analysis_timeout: float = 180.0,
                 metrics_collector: Optional[MetricsCollectorProtocol] = None,
                 logger: Optional[logging.Logger] = None,
                 http_client: Optional[HTTPClientProtocol] = None,
                 file_system: Optional[FileSystemProtocol] = None,
                 resource_monitor: Optional[ResourceMonitorProtocol] = None,
                 max_video_size_mb: float = 100.0,
                 max_image_size_mb: float = 10.0,
                 download_timeout: float = 180.0):
        """
        Initialize Gemini analyzer.

        Args:
            api_key: Gemini API key
            model_priority: List of models to try in order of preference
            analysis_timeout: Maximum time for analysis in seconds
            metrics_collector: Metrics collection interface
            logger: Optional logger instance
            http_client: HTTP client for media downloads
            file_system: File system operations
            resource_monitor: Resource monitoring
            max_video_size_mb: Maximum video file size in MB
            max_image_size_mb: Maximum image file size in MB
            download_timeout: Download timeout in seconds
        """
        self.api_key = api_key
        self.analysis_timeout = analysis_timeout
        self.metrics_collector = metrics_collector or DefaultMetricsCollector()
        self.logger = logger or logging.getLogger(__name__)

        # Set default model priority if not provided
        self.model_priority = model_priority or [
            'gemini-2.5-pro',
            'gemini-2.5-flash',
            'gemini-2.5-flash-lite',
            'gemini-2.0-flash-exp',
            'gemini-2.0-flash-lite',
            'gemini-2.0-flash'
        ]

        # Initialize components
        self.client = GeminiClient(api_key, self.logger)
        self.media_processor = MediaProcessor(
            http_client=http_client or RequestsHTTPClient(),
            file_system=file_system or TempFileSystem(),
            resource_monitor=resource_monitor or DefaultResourceMonitor(),
            max_video_size_mb=max_video_size_mb,
            max_image_size_mb=max_image_size_mb,
            download_timeout=download_timeout,
            logger=self.logger
        )

        # Track rate-limited models to avoid repeated failures
        self.rate_limited_models = set()

    def _get_available_models(self) -> List[str]:
        """
        Get list of available models, excluding currently rate-limited ones.

        Returns:
            List of model names that are not currently rate-limited
        """
        available_models = []

        for model_name in self.model_priority:
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
        Analyze multimodal content (images/videos + text) or text-only content using Gemini models with fallback.

        Args:
            media_urls: List of media URLs to analyze (empty list for text-only)
            text_content: Text content to analyze

        Returns:
            Tuple of (analysis_result, analysis_time_seconds)
            Returns (None, error_time) if analysis failed
        """
        analysis_start_time = time.time()

        # Record analysis start
        self.metrics_collector.record_operation(
            "analysis_start", 0.0, True, media_count=len(media_urls) if media_urls else 0
        )

        # Check if this is text-only analysis (no media URLs)
        if not media_urls:
            self.logger.info("📝 Performing text-only analysis")
            return self._analyze_text_only(text_content, analysis_start_time)

        # Filter out unwanted media
        filtered_urls = []
        for url in media_urls:
            if 'profile_images' in url or 'card_img' in url:
                continue
            filtered_urls.append(url)

        if not filtered_urls:
            self.logger.warning("No valid media URLs found after filtering")
            self.metrics_collector.record_operation(
                "analysis_failure", time.time() - analysis_start_time, False,
                error_category="no_valid_media_urls"
            )
            return None, time.time() - analysis_start_time

        # Select media URL with priority logic
        media_url = self.media_processor.select_media_url(filtered_urls)
        is_video = self.media_processor.is_video_url(media_url)

        # Download media
        media_path = self.media_processor.download_media(media_url, is_video)
        if not media_path:
            self.metrics_collector.record_operation(
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
                if time.time() - analysis_start_time > self.analysis_timeout:
                    self.logger.error("❌ Analysis timeout exceeded")
                    return None, time.time() - analysis_start_time

                result = self._try_model_analysis(model_name, media_path, media_url, text_content, is_video)
                if result:
                    # Mark model as successful
                    self._mark_model_success(model_name)
                    total_time = time.time() - analysis_start_time
                    self.metrics_collector.record_operation(
                        "analysis_success", total_time, True,
                        model_used=model_name, media_type="video" if is_video else "image"
                    )
                    return result, total_time

            # All available models failed
            self.logger.error("💔 All available models failed")
            return None, time.time() - analysis_start_time

        finally:
            # Clean up
            self.media_processor.cleanup_file(media_path)

    def _analyze_text_only(self, text_content: str, analysis_start_time: float) -> Tuple[Optional[str], float]:
        """
        Analyze text-only content using Gemini models with fallback and retries.

        Args:
            text_content: Text content to analyze
            analysis_start_time: When analysis started (for timeout tracking)

        Returns:
            Tuple of (analysis_result, analysis_time_seconds)
        """
        try:
            # Try each available model in priority order
            available_models = self._get_available_models()
            if not available_models:
                self.logger.error("💔 No available models (all rate-limited)")
                self.metrics_collector.record_operation(
                    "text_analysis_failure", time.time() - analysis_start_time, False,
                    error_category="no_available_models"
                )
                return None, time.time() - analysis_start_time

            for model_name in available_models:
                # Check timeout
                if time.time() - analysis_start_time > self.analysis_timeout:
                    self.logger.error("❌ Text analysis timeout exceeded")
                    self.metrics_collector.record_operation(
                        "text_analysis_failure", time.time() - analysis_start_time, False,
                        error_category="timeout"
                    )
                    return None, time.time() - analysis_start_time

                result = self._try_text_model_analysis(model_name, text_content)
                if result:
                    # Mark model as successful
                    self._mark_model_success(model_name)
                    total_time = time.time() - analysis_start_time
                    self.metrics_collector.record_operation(
                        "text_analysis_success", total_time, True,
                        model_used=model_name
                    )
                    return result, total_time

            # All available models failed
            self.logger.error("💔 All available models failed for text analysis")
            self.metrics_collector.record_operation(
                "text_analysis_failure", time.time() - analysis_start_time, False,
                error_category="all_models_failed"
            )
            return None, time.time() - analysis_start_time

        except Exception as e:
            error = classify_error(e, "text analysis")
            self.logger.error(f"❌ Text analysis failed: {error}")
            self.metrics_collector.record_operation(
                "text_analysis_failure", time.time() - analysis_start_time, False,
                error_category=error.category.value
            )
            return None, time.time() - analysis_start_time

    def _try_text_model_analysis(self, model_name: str, text_content: str) -> Optional[str]:
        """
        Try text-only analysis with a specific model.

        Args:
            model_name: Name of the model to use
            text_content: Text content to analyze

        Returns:
            Analysis result or None if failed
        """
        try:
            self.logger.info(f"🔄 Trying text model: {model_name}")

            # Get Gemini client
            model, error = self.client.get_model(model_name)
            if not model:
                self.logger.error(f"❌ Failed to initialize {model_name}: {error}")
                return None

            # Build text-only analysis prompt
            prompt = EnhancedPromptGenerator.build_gemini_multimodal_analysis_prompt(text_content, is_video=False)

            # Generate content
            result = self.client.generate_content(model, prompt)
            if result:
                self.logger.info(f"✅ {model_name} text analysis completed")
                return result
            else:
                self.logger.warning(f"⚠️ {model_name} returned empty response")
                return None

        except Exception as e:
            error = classify_error(e, f"text {model_name} analysis")

            # Check if this is a rate limiting error
            if error.category == ErrorCategory.QUOTA_ERROR:
                self._mark_model_rate_limited(model_name)

            self.logger.error(f"❌ {error}")
            return None

    def _try_model_analysis(self, model_name: str, media_path: str, media_url: str,
                          text_content: str, is_video: bool) -> Optional[str]:
        """Try analysis with a specific model."""
        media_file = None
        try:
            self.logger.info(f"🔄 Trying model: {model_name}")

            # Get Gemini client
            model, error = self.client.get_model(model_name)
            if not model:
                self.logger.error(f"❌ Failed to initialize {model_name}: {error}")
                return None

            # Prepare media
            media_file = self.client.prepare_media_for_analysis(media_path, media_url)
            if not media_file:
                return None

            # Generate analysis
            prompt = EnhancedPromptGenerator.build_gemini_multimodal_analysis_prompt(text_content, is_video)

            # Generate content
            result = self.client.generate_content(model, [media_file, prompt])
            if result:
                self.logger.info(f"✅ {model_name} analysis completed")
                return result
            else:
                self.logger.warning(f"⚠️ {model_name} returned empty response")
                return None

        except Exception as e:
            error = classify_error(e, f"{model_name} analysis")

            # Check if this is a rate limiting error
            if error.category == ErrorCategory.QUOTA_ERROR:
                self._mark_model_rate_limited(model_name)

            self.logger.error(f"❌ {error}")
            return None
        finally:
            # Clean up PIL Image if it was created
            if media_file and hasattr(media_file, 'close'):
                try:
                    media_file.close()
                except Exception:
                    pass  # Ignore cleanup errors

    async def analyze_multimodal_content_async(self, media_urls: List[str], text_content: str) -> Tuple[Optional[str], float]:
        """
        Async version of analyze_multimodal_content with concurrent processing.
        Supports both multimodal and text-only analysis.

        Args:
            media_urls: List of media URLs to analyze (empty list for text-only)
            text_content: Text content to analyze

        Returns:
            Tuple of (analysis_result, analysis_time_seconds)
            Returns (None, error_time) if analysis failed
        """
        analysis_start_time = time.time()

        # Record analysis start
        self.metrics_collector.record_operation(
            "async_analysis_start", 0.0, True, media_count=len(media_urls) if media_urls else 0
        )

        # Check if this is text-only analysis (no media URLs)
        if not media_urls:
            self.logger.info("📝 Performing async text-only analysis")
            # Run text analysis in thread pool since it's sync
            loop = asyncio.get_event_loop()
            result, time_taken = await loop.run_in_executor(
                None, self._analyze_text_only, text_content, analysis_start_time
            )
            return result, time_taken

        # Filter out unwanted media
        filtered_urls = []
        for url in media_urls:
            if 'profile_images' in url or 'card_img' in url:
                continue
            filtered_urls.append(url)

        if not filtered_urls:
            self.logger.warning("No valid media URLs found after filtering")
            self.metrics_collector.record_operation(
                "async_analysis_failure", time.time() - analysis_start_time, False,
                error_category="no_valid_media_urls"
            )
            return None, time.time() - analysis_start_time

        # For async processing, we can download multiple media files concurrently
        # and analyze them, but for now we'll focus on the primary media file
        media_url = self.media_processor.select_media_url(filtered_urls)
        is_video = self.media_processor.is_video_url(media_url)

        # Download media asynchronously
        media_path = await self.media_processor.download_media_async(media_url, is_video)
        if not media_path:
            self.metrics_collector.record_operation(
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
                if time.time() - analysis_start_time > self.analysis_timeout:
                    self.logger.error("❌ Analysis timeout exceeded")
                    return None, time.time() - analysis_start_time

                result = await self._try_model_analysis_async(model_name, media_path, media_url, text_content, is_video)
                if result:
                    # Mark model as successful
                    self._mark_model_success(model_name)
                    total_time = time.time() - analysis_start_time
                    self.metrics_collector.record_operation(
                        "async_analysis_success", total_time, True,
                        model_used=model_name, media_type="video" if is_video else "image"
                    )
                    return result, total_time

            # All available models failed
            self.logger.error("💔 All available async models failed")
            return None, time.time() - analysis_start_time

        finally:
            # Clean up
            self.media_processor.cleanup_file(media_path)

    async def _try_model_analysis_async(self, model_name: str, media_path: str, media_url: str,
                                      text_content: str, is_video: bool) -> Optional[str]:
        """Try analysis with a specific model asynchronously."""
        media_file = None
        try:
            self.logger.info(f"🔄 Trying model async: {model_name}")

            # Get Gemini client (run in thread pool since genai is sync)
            loop = asyncio.get_event_loop()
            model, error = await loop.run_in_executor(
                None, self.client.get_model, model_name
            )

            if not model:
                self.logger.error(f"❌ Failed to initialize {model_name}: {error}")
                return None

            # Prepare media (run in thread pool)
            media_file = await loop.run_in_executor(
                None, self.client.prepare_media_for_analysis, media_path, media_url
            )
            if not media_file:
                return None

            # Generate analysis (run in thread pool with timeout)
            prompt = EnhancedPromptGenerator.build_gemini_multimodal_analysis_prompt(text_content, is_video)

            try:
                response = await asyncio.wait_for(
                    loop.run_in_executor(None, self.client.generate_content, model, [media_file, prompt]),
                    timeout=60.0
                )

                self.logger.info(f"✅ {model_name} async analysis completed")
                return response

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
        finally:
            # Clean up PIL Image if it was created
            if media_file and hasattr(media_file, 'close'):
                try:
                    media_file.close()
                except Exception:
                    pass  # Ignore cleanup errors

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

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return self.metrics_collector.get_summary()

    def log_metrics_summary(self):
        """Log metrics summary."""
        summary = self.get_metrics_summary()
        self.logger.info("📊 Multimodal Analysis Metrics:")
        for key, value in summary.items():
            self.logger.info(f"   {key}: {value}")

    def reset_metrics(self):
        """Reset metrics."""
        # Create new metrics collector instance
        self.metrics_collector = DefaultMetricsCollector()
        self.logger.info("📊 Metrics reset")