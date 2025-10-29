"""
Tests for MediaProcessor class.

Comprehensive test suite for media processing functionality.
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch
from io import BytesIO

from analyzer.media_processor import (
    MediaProcessor,
    HTTPClientProtocol,
    AsyncHTTPClientProtocol,
    FileSystemProtocol,
    ResourceMonitorProtocol
)


class MockHTTPClient:
    """Mock HTTP client for testing."""
    def __init__(self, response_data=b"test content", status_code=200):
        self.response_data = response_data
        self.status_code = status_code

    def get(self, url, timeout=30.0, **kwargs):
        response = Mock()
        response.status_code = self.status_code
        response.raise_for_status = Mock()
        response.iter_content = Mock(return_value=[self.response_data])
        return response


class MockAsyncHTTPClient:
    """Mock async HTTP client for testing."""
    def __init__(self, response_data=b"test content", status_code=200):
        self.response_data = response_data
        self.status_code = status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def get(self, url, timeout=30.0, **kwargs):
        response = AsyncMock()
        response.status_code = self.status_code
        response.raise_for_status = Mock()

        # Mock content with proper async iteration
        async def mock_iter_chunked(chunk_size):
            yield self.response_data

        content_mock = AsyncMock()
        content_mock.iter_chunked = mock_iter_chunked
        response.content = content_mock

        return response


class MockFileSystem:
    """Mock file system for testing."""
    def __init__(self):
        self.created_files = []
        self.removed_files = []

    def create_temp_file(self, suffix=""):
        temp_path = f"/tmp/test_file{suffix}"
        self.created_files.append(temp_path)
        return temp_path

    def remove_file(self, file_path):
        self.removed_files.append(file_path)


class MockResourceMonitor:
    """Mock resource monitor for testing."""
    def __init__(self, file_size=1024):
        self.file_size = file_size

    def check_file_size(self, file_path):
        return self.file_size


class TestMediaProcessorInitialization:
    """Test MediaProcessor initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        http_client = MockHTTPClient()
        file_system = MockFileSystem()
        resource_monitor = MockResourceMonitor()

        processor = MediaProcessor(http_client, file_system, resource_monitor)

        assert processor.http_client == http_client
        assert processor.file_system == file_system
        assert processor.resource_monitor == resource_monitor
        assert processor.async_http_client is None
        assert processor.max_video_size_mb == 100.0
        assert processor.max_image_size_mb == 10.0
        assert processor.download_timeout == 180.0

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        http_client = MockHTTPClient()
        file_system = MockFileSystem()
        resource_monitor = MockResourceMonitor()
        async_client = MockAsyncHTTPClient()

        processor = MediaProcessor(
            http_client=http_client,
            file_system=file_system,
            resource_monitor=resource_monitor,
            async_http_client=async_client,
            max_video_size_mb=50.0,
            max_image_size_mb=5.0,
            download_timeout=60.0
        )

        assert processor.async_http_client == async_client
        assert processor.max_video_size_mb == 50.0
        assert processor.max_image_size_mb == 5.0
        assert processor.download_timeout == 60.0


class TestSelectMediaUrl:
    """Test media URL selection functionality."""

    def test_select_media_url_mp4_priority(self):
        """Test MP4 files have highest priority."""
        http_client = MockHTTPClient()
        file_system = MockFileSystem()
        resource_monitor = MockResourceMonitor()

        processor = MediaProcessor(http_client, file_system, resource_monitor)

        urls = [
            "https://example.com/image.jpg",
            "https://example.com/video.mp4",
            "https://example.com/other.m3u8"
        ]

        result = processor.select_media_url(urls)
        assert result == "https://example.com/video.mp4"

    def test_select_media_url_m3u8_priority(self):
        """Test M3U8 files have second priority."""
        http_client = MockHTTPClient()
        file_system = MockFileSystem()
        resource_monitor = MockResourceMonitor()

        processor = MediaProcessor(http_client, file_system, resource_monitor)

        urls = [
            "https://example.com/image.jpg",
            "https://example.com/other.m3u8"
        ]

        result = processor.select_media_url(urls)
        assert result == "https://example.com/other.m3u8"

    def test_select_media_url_video_pattern_priority(self):
        """Test video pattern URLs have third priority."""
        http_client = MockHTTPClient()
        file_system = MockFileSystem()
        resource_monitor = MockResourceMonitor()

        processor = MediaProcessor(http_client, file_system, resource_monitor)

        urls = [
            "https://example.com/image.jpg",
            "https://example.com/video_content"
        ]

        result = processor.select_media_url(urls)
        assert result == "https://example.com/video_content"

    def test_select_media_url_fallback_to_first(self):
        """Test fallback to first URL when no priorities match."""
        http_client = MockHTTPClient()
        file_system = MockFileSystem()
        resource_monitor = MockResourceMonitor()

        processor = MediaProcessor(http_client, file_system, resource_monitor)

        urls = [
            "https://example.com/image1.jpg",
            "https://example.com/image2.png"
        ]

        result = processor.select_media_url(urls)
        assert result == "https://example.com/image1.jpg"

    def test_select_media_url_empty_list(self):
        """Test error when no URLs provided."""
        http_client = MockHTTPClient()
        file_system = MockFileSystem()
        resource_monitor = MockResourceMonitor()

        processor = MediaProcessor(http_client, file_system, resource_monitor)

        with pytest.raises(ValueError, match="No media URLs provided"):
            processor.select_media_url([])


class TestIsVideoUrl:
    """Test video URL detection."""

    def test_is_video_url_mp4(self):
        """Test MP4 detection."""
        http_client = MockHTTPClient()
        file_system = MockFileSystem()
        resource_monitor = MockResourceMonitor()

        processor = MediaProcessor(http_client, file_system, resource_monitor)

        assert processor.is_video_url("https://example.com/video.mp4") is True

    def test_is_video_url_m3u8(self):
        """Test M3U8 detection."""
        http_client = MockHTTPClient()
        file_system = MockFileSystem()
        resource_monitor = MockResourceMonitor()

        processor = MediaProcessor(http_client, file_system, resource_monitor)

        assert processor.is_video_url("https://example.com/stream.m3u8") is True

    def test_is_video_url_video_pattern(self):
        """Test video pattern detection."""
        http_client = MockHTTPClient()
        file_system = MockFileSystem()
        resource_monitor = MockResourceMonitor()

        processor = MediaProcessor(http_client, file_system, resource_monitor)

        assert processor.is_video_url("https://example.com/video") is True

    def test_is_video_url_image(self):
        """Test image URL returns False."""
        http_client = MockHTTPClient()
        file_system = MockFileSystem()
        resource_monitor = MockResourceMonitor()

        processor = MediaProcessor(http_client, file_system, resource_monitor)

        assert processor.is_video_url("https://example.com/image.jpg") is False


class TestDownloadMedia:
    """Test synchronous media download functionality."""

    def test_download_media_success_image(self):
        """Test successful image download."""
        http_client = MockHTTPClient(b"fake image data")
        file_system = MockFileSystem()
        resource_monitor = MockResourceMonitor(1024)  # 1KB file

        processor = MediaProcessor(http_client, file_system, resource_monitor)

        result = processor.download_media("https://example.com/image.jpg", is_video=False)

        assert result is not None
        assert result in file_system.created_files

    def test_download_media_success_video(self):
        """Test successful video download."""
        http_client = MockHTTPClient(b"fake video data")
        file_system = MockFileSystem()
        resource_monitor = MockResourceMonitor(1024)  # 1KB file

        processor = MediaProcessor(http_client, file_system, resource_monitor)

        result = processor.download_media("https://example.com/video.mp4", is_video=True)

        assert result is not None
        assert result in file_system.created_files

    def test_download_media_empty_response(self):
        """Test handling of empty response."""
        http_client = MockHTTPClient(b"")
        file_system = MockFileSystem()
        resource_monitor = MockResourceMonitor(0)  # Empty file

        processor = MediaProcessor(http_client, file_system, resource_monitor)

        result = processor.download_media("https://example.com/image.jpg", is_video=False)

        assert result is None

    def test_download_media_video_too_large(self):
        """Test video size limit enforcement."""
        http_client = MockHTTPClient(b"x" * (200 * 1024 * 1024))  # 200MB
        file_system = MockFileSystem()
        resource_monitor = MockResourceMonitor(200 * 1024 * 1024)  # 200MB

        processor = MediaProcessor(http_client, file_system, resource_monitor, max_video_size_mb=100.0)

        result = processor.download_media("https://example.com/video.mp4", is_video=True)

        assert result is None

    def test_download_media_image_too_large(self):
        """Test image size limit enforcement."""
        http_client = MockHTTPClient(b"x" * (20 * 1024 * 1024))  # 20MB
        file_system = MockFileSystem()
        resource_monitor = MockResourceMonitor(20 * 1024 * 1024)  # 20MB

        processor = MediaProcessor(http_client, file_system, resource_monitor, max_image_size_mb=10.0)

        result = processor.download_media("https://example.com/image.jpg", is_video=False)

        assert result is None

    def test_download_media_http_error(self):
        """Test HTTP error handling."""
        # Create a mock response that raises an exception
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")

        http_client = Mock()
        http_client.get.return_value = mock_response

        file_system = MockFileSystem()
        resource_monitor = MockResourceMonitor()

        processor = MediaProcessor(http_client, file_system, resource_monitor)

        result = processor.download_media("https://example.com/image.jpg", is_video=False)

        assert result is None


class TestDownloadMediaAsync:
    """Test asynchronous media download functionality."""

    @pytest.mark.asyncio
    async def test_download_media_async_with_async_client(self):
        """Test async download with async HTTP client."""
        http_client = MockHTTPClient()
        async_client = MockAsyncHTTPClient(b"fake async data")
        file_system = MockFileSystem()
        resource_monitor = MockResourceMonitor(1024)

        processor = MediaProcessor(http_client, file_system, resource_monitor, async_http_client=async_client)

        result = await processor.download_media_async("https://example.com/image.jpg", is_video=False)

        assert result is not None
        assert result in file_system.created_files

    @pytest.mark.asyncio
    async def test_download_media_async_fallback_to_sync(self):
        """Test async download fallback to sync when no async client."""
        http_client = MockHTTPClient(b"fake sync data")
        file_system = MockFileSystem()
        resource_monitor = MockResourceMonitor(1024)

        processor = MediaProcessor(http_client, file_system, resource_monitor)

        result = await processor.download_media_async("https://example.com/image.jpg", is_video=False)

        assert result is not None
        assert result in file_system.created_files

    @pytest.mark.asyncio
    async def test_download_media_async_empty_response(self):
        """Test async download with empty response."""
        http_client = MockHTTPClient()
        async_client = MockAsyncHTTPClient(b"")
        file_system = MockFileSystem()
        resource_monitor = MockResourceMonitor(0)

        processor = MediaProcessor(http_client, file_system, resource_monitor, async_http_client=async_client)

        result = await processor.download_media_async("https://example.com/image.jpg", is_video=False)

        assert result is None

    @pytest.mark.asyncio
    async def test_download_media_async_video_too_large(self):
        """Test async download video size limit."""
        http_client = MockHTTPClient()
        async_client = MockAsyncHTTPClient(b"x" * (200 * 1024 * 1024))
        file_system = MockFileSystem()
        resource_monitor = MockResourceMonitor(200 * 1024 * 1024)

        processor = MediaProcessor(
            http_client, file_system, resource_monitor,
            async_http_client=async_client, max_video_size_mb=100.0
        )

        result = await processor.download_media_async("https://example.com/video.mp4", is_video=True)

        assert result is None


class TestCleanupFile:
    """Test file cleanup functionality."""

    def test_cleanup_file_success(self):
        """Test successful file cleanup."""
        http_client = MockHTTPClient()
        file_system = MockFileSystem()
        resource_monitor = MockResourceMonitor()

        processor = MediaProcessor(http_client, file_system, resource_monitor)

        processor.cleanup_file("/tmp/test_file.jpg")

        assert "/tmp/test_file.jpg" in file_system.removed_files

    def test_cleanup_file_error(self):
        """Test cleanup error handling."""
        http_client = MockHTTPClient()
        file_system = MockFileSystem()
        resource_monitor = MockResourceMonitor()

        # Mock remove_file to raise an exception
        file_system.remove_file = Mock(side_effect=Exception("Permission denied"))

        processor = MediaProcessor(http_client, file_system, resource_monitor)

        # Should not raise exception, just log warning
        processor.cleanup_file("/tmp/test_file.jpg")