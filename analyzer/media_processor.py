"""
Media processing utilities for the dimetuverdad analyzer.

Handles media download, validation, processing, and cleanup operations.
"""

import asyncio
import logging
from typing import Optional, List, Protocol
from concurrent.futures import ThreadPoolExecutor

import requests
import aiohttp

from .error_handler import classify_error, ErrorCategory


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
    def check_file_size(self, file_path: str) -> int:
        """Check file size in bytes."""
        ...


class MediaProcessor:
    """
    Handles media download, validation, and processing operations.
    """

    def __init__(self,
                 http_client: HTTPClientProtocol,
                 file_system: FileSystemProtocol,
                 resource_monitor: ResourceMonitorProtocol,
                 async_http_client: Optional[AsyncHTTPClientProtocol] = None,
                 max_video_size_mb: float = 100.0,
                 max_image_size_mb: float = 10.0,
                 download_timeout: float = 180.0,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize media processor with dependencies.

        Args:
            http_client: HTTP client for downloads
            file_system: File system operations
            resource_monitor: Resource monitoring
            async_http_client: Optional async HTTP client
            max_video_size_mb: Maximum video file size in MB
            max_image_size_mb: Maximum image file size in MB
            download_timeout: Download timeout in seconds
            logger: Optional logger instance
        """
        self.http_client = http_client
        self.async_http_client = async_http_client
        self.file_system = file_system
        self.resource_monitor = resource_monitor
        self.max_video_size_mb = max_video_size_mb
        self.max_image_size_mb = max_image_size_mb
        self.download_timeout = download_timeout
        self.logger = logger or logging.getLogger(__name__)

    def select_media_url(self, media_urls: List[str]) -> str:
        """
        Select the best media URL based on priority logic.

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

    def is_video_url(self, url: str) -> bool:
        """Check if URL points to video content."""
        return any(pattern in url.lower() for pattern in ['.mp4', '.m3u8', '.mov', '.avi', '.webm', 'video'])

    def download_media(self, media_url: str, is_video: bool) -> Optional[str]:
        """
        Download media using injected dependencies.

        Args:
            media_url: URL of media to download
            is_video: Whether the media is a video file

        Returns:
            Path to downloaded file or None if download failed
        """
        try:
            self.logger.info(f"🔗 Downloading media: {media_url}")

            # Use injected HTTP client
            response = self.http_client.get(
                media_url,
                timeout=self.download_timeout
            )
            response.raise_for_status()

            # Create temp file
            suffix = '.mp4' if is_video else '.jpg'
            temp_path = self.file_system.create_temp_file(suffix)

            # Download content
            downloaded = 0
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

            # Verify file
            file_size = self.resource_monitor.check_file_size(temp_path)
            if file_size == 0:
                self.logger.error("❌ Downloaded file is empty")
                self.file_system.remove_file(temp_path)
                return None

            # Check file size limits
            file_size_mb = file_size / (1024 * 1024)  # Convert to MB
            if is_video and file_size_mb > self.max_video_size_mb:
                self.logger.error(".2f")
                self.file_system.remove_file(temp_path)
                return None
            elif not is_video and file_size_mb > self.max_image_size_mb:
                self.logger.error(".2f")
                self.file_system.remove_file(temp_path)
                return None

            self.logger.info(".2f")
            return temp_path

        except Exception as e:
            error = classify_error(e, "media download")
            self.logger.error(f"❌ {error}")
            return None

    async def download_media_async(self, media_url: str, is_video: bool) -> Optional[str]:
        """
        Download media asynchronously using injected async HTTP client.

        Args:
            media_url: URL of media to download
            is_video: Whether the media is a video file

        Returns:
            Path to downloaded file or None if download failed
        """
        if not self.async_http_client:
            # Fallback to sync download in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                return await loop.run_in_executor(executor, self.download_media, media_url, is_video)

        try:
            self.logger.info(f"🔗 Downloading media async: {media_url}")

            # Use injected async HTTP client
            async with self.async_http_client as client:
                response = await client.get(
                    media_url,
                    timeout=self.download_timeout
                )
                response.raise_for_status()

                # Create temp file
                suffix = '.mp4' if is_video else '.jpg'
                temp_path = self.file_system.create_temp_file(suffix)

                # Download content
                downloaded = 0
                with open(temp_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                # Verify file
                file_size = self.resource_monitor.check_file_size(temp_path)
                if file_size == 0:
                    self.logger.error("❌ Downloaded file is empty")
                    self.file_system.remove_file(temp_path)
                    return None

                # Check file size limits
                file_size_mb = file_size / (1024 * 1024)  # Convert to MB
                if is_video and file_size_mb > self.max_video_size_mb:
                    self.logger.error(".2f")
                    self.file_system.remove_file(temp_path)
                    return None
                elif not is_video and file_size_mb > self.max_image_size_mb:
                    self.logger.error(".2f")
                    self.file_system.remove_file(temp_path)
                    return None

                self.logger.info(".2f")
                return temp_path

        except Exception as e:
            error = classify_error(e, "async media download")
            self.logger.error(f"❌ {error}")
            return None

    def cleanup_file(self, file_path: str):
        """
        Clean up temporary file.

        Args:
            file_path: Path to file to clean up
        """
        try:
            self.file_system.remove_file(file_path)
            self.logger.debug(f"🧹 Cleaned up: {file_path}")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not clean up {file_path}: {e}")