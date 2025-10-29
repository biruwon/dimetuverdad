"""
Tests for MediaMonitor class in fetcher/media_monitor.py
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from fetcher.media_monitor import MediaMonitor


class TestMediaMonitor:
    """Test cases for MediaMonitor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.media_monitor = MediaMonitor()

    def test_init(self):
        """Test MediaMonitor initialization."""
        assert self.media_monitor.media_extensions == ['.mp4', '.m3u8', '.webm', '.mov', '.jpg', '.jpeg', '.png', '.gif', '.webp']
        assert self.media_monitor.media_keywords == ['video.twimg.com', 'pbs.twimg.com', 'mediadelivery']

    def test_setup_monitoring(self):
        """Test setting up network request monitoring."""
        mock_page = Mock()
        mock_page.on = Mock()

        media_urls = self.media_monitor.setup_monitoring(mock_page)

        assert isinstance(media_urls, list)
        assert len(media_urls) == 0
        # Should be called twice - once for request, once for response
        assert mock_page.on.call_count == 2

    @patch('fetcher.media_monitor.logger')
    def test_setup_monitoring_captures_video_url(self, mock_logger):
        """Test that the first video URL is captured during monitoring."""
        mock_page = Mock()
        captured_requests = []

        def mock_on(event, handler):
            # Simulate capturing multiple requests - only first should be captured
            urls = [
                "https://video.twimg.com/first_video.mp4",
                "https://video.twimg.com/second_video.mp4",  # This should be ignored
            ]
            for url in urls:
                mock_request = Mock()
                mock_request.url = url
                captured_requests.append(mock_request)
                handler(mock_request)

        mock_page.on = mock_on

        media_urls = self.media_monitor.setup_monitoring(mock_page)

        assert len(media_urls) == 1
        assert media_urls[0] == "https://video.twimg.com/first_video.mp4"
        mock_logger.debug.assert_called_once()

    @patch('fetcher.media_monitor.logger')
    def test_setup_monitoring_ignores_non_video_urls(self, mock_logger):
        """Test that non-video URLs are ignored."""
        mock_page = Mock()
        captured_requests = []

        def mock_on(event, handler):
            urls = [
                "https://twitter.com/api/timeline",
                "https://x.com/user/profile", 
                "https://example.com/script.js",
                "https://pbs.twimg.com/media/test_image.jpg",  # Image - should be ignored
                "https://video.twimg.com/first_video.mp4",  # This should be captured
                "https://video.twimg.com/second_video.mp4",  # This should be ignored (only first)
            ]
            for url in urls:
                mock_request = Mock()
                mock_request.url = url
                captured_requests.append(mock_request)
                handler(mock_request)

        mock_page.on = mock_on

        media_urls = self.media_monitor.setup_monitoring(mock_page)

        assert len(media_urls) == 1
        assert media_urls[0] == "https://video.twimg.com/first_video.mp4"

    @patch('fetcher.media_monitor.logger')
    def test_setup_monitoring_avoids_duplicates(self, mock_logger):
        """Test that only the first URL is captured, even with duplicates."""
        mock_page = Mock()
        captured_requests = []

        def mock_on(event, handler):
            # Simulate multiple requests with same URL - only first should be captured
            for _ in range(3):
                mock_request = Mock()
                mock_request.url = "https://video.twimg.com/test.mp4"
                captured_requests.append(mock_request)
                handler(mock_request)

        mock_page.on = mock_on

        media_urls = self.media_monitor.setup_monitoring(mock_page)

        assert len(media_urls) == 1  # Should only capture first one
        assert media_urls[0] == "https://video.twimg.com/test.mp4"
        mock_logger.debug.assert_called_once()  # Only called once for first URL

    @patch('fetcher.media_monitor.logger')
    def test_setup_monitoring_ignores_non_media_urls(self, mock_logger):
        """Test that non-media URLs are ignored and only first video URL is captured."""
        mock_page = Mock()
        captured_requests = []

        def mock_on(event, handler):
            urls = [
                "https://twitter.com/api/timeline",
                "https://x.com/user/profile",
                "https://example.com/script.js",
                "https://video.twimg.com/first_video.mp4",  # This should be captured
                "https://video.twimg.com/second_video.mp4",  # This should be ignored
            ]
            for url in urls:
                mock_request = Mock()
                mock_request.url = url
                captured_requests.append(mock_request)
                handler(mock_request)

        mock_page.on = mock_on

        media_urls = self.media_monitor.setup_monitoring(mock_page)

        assert len(media_urls) == 1
        assert media_urls[0] == "https://video.twimg.com/first_video.mp4"

    def test_determine_media_type_video(self):
        """Test determining media type for video URLs."""
        video_urls = [
            "https://video.twimg.com/test.mp4",
            "https://example.com/video.m3u8",
            "https://test.com/movie.webm",
            "https://example.com/clip.MOV"
        ]

        for url in video_urls:
            result = self.media_monitor._determine_media_type(url)
            assert result == "video"

    def test_determine_media_type_image(self):
        """Test determining media type for image URLs."""
        image_urls = [
            "https://pbs.twimg.com/media/test.jpg",
            "https://example.com/photo.jpeg",
            "https://test.com/image.png",
            "https://example.com/picture.gif",
            "https://test.com/animated.WEBP"
        ]

        for url in image_urls:
            result = self.media_monitor._determine_media_type(url)
            assert result == "image"

    def test_determine_media_type_unknown_extension(self):
        """Test determining media type for unknown extensions."""
        url = "https://example.com/file.unknown"
        result = self.media_monitor._determine_media_type(url)
        assert result == "image"  # Default fallback

    def test_setup_and_monitor(self, monkeypatch):
        """Test setup_and_monitor method."""
        mock_page = Mock()
        mock_scroller = Mock()
        mock_scroller.delay = Mock()
        
        # Mock setup_monitoring to return empty list initially
        self.media_monitor.setup_monitoring = Mock(return_value=[])
        
        # Mock video element
        mock_video_element = Mock()
        mock_page.query_selector.return_value = mock_video_element
        
        result = self.media_monitor.setup_and_monitor(mock_page, mock_scroller)
        
        # Should call setup_monitoring
        self.media_monitor.setup_monitoring.assert_called_once_with(mock_page)
        # Should call delay twice (once for wait, once after hover)
        assert mock_scroller.delay.call_count == 2
        # Should query for video element
        mock_page.query_selector.assert_called_once_with('[data-testid="videoPlayer"]')
        # Should hover over video element
        mock_video_element.hover.assert_called_once()

    def test_process_video_urls_new_media(self):
        """Test adding new media URLs to tweet data."""
        tweet_data = {
            "text": "Test tweet",
            "media_count": 0,
            "media_links": "",
        }

        media_urls = ["https://video.twimg.com/test.mp4"]

        result = self.media_monitor.process_video_urls(media_urls, tweet_data)

        assert result["media_count"] == 1
        assert result["media_links"] == "https://video.twimg.com/test.mp4"

    def test_process_video_urls_with_existing_media(self):
        """Test adding media URLs to tweet data that already has media."""
        tweet_data = {
            "text": "Test tweet",
            "media_count": 1,
            "media_links": "https://existing.com/old.jpg",
        }

        media_urls = ["https://video.twimg.com/test.mp4"]

        result = self.media_monitor.process_video_urls(media_urls, tweet_data)

        assert result["media_count"] == 2
        assert result["media_links"] == "https://existing.com/old.jpg,https://video.twimg.com/test.mp4"

    def test_process_video_urls_malformed_existing_data(self):
        """Test handling existing media data."""
        tweet_data = {
            "text": "Test tweet",
            "media_count": 1,
            "media_links": "https://existing.com/old.jpg",
        }

        media_urls = ["https://video.twimg.com/test.mp4"]

        result = self.media_monitor.process_video_urls(media_urls, tweet_data)

        assert result["media_count"] == 2
        assert "https://video.twimg.com/test.mp4" in result["media_links"]

    @patch('fetcher.media_monitor.logger')
    def test_process_video_urls_logging(self, mock_logger):
        """Test that adding media URLs triggers appropriate logging."""
        tweet_data = {"text": "Test tweet", "media_count": 0}
        media_urls = ["https://video.twimg.com/test.mp4"]

        result = self.media_monitor.process_video_urls(media_urls, tweet_data)

        mock_logger.info.assert_called_once()

    def test_get_media_monitor(self):
        """Test getting the global media monitor instance."""
        from fetcher.media_monitor import get_media_monitor

        monitor = get_media_monitor()
        assert isinstance(monitor, MediaMonitor)
        assert monitor is not None