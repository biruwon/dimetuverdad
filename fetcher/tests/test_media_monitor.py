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
        mock_page.on.assert_called_once_with("request", mock_page.on.call_args[0][1])

    @patch('fetcher.media_monitor.logger')
    def test_setup_monitoring_captures_video_url(self, mock_logger):
        """Test that video URLs are captured during monitoring."""
        mock_page = Mock()
        captured_requests = []

        def mock_on(event, handler):
            # Simulate capturing a request
            mock_request = Mock()
            mock_request.url = "https://video.twimg.com/test_video.mp4"
            captured_requests.append(mock_request)
            # Call the handler to simulate the request
            handler(mock_request)

        mock_page.on = mock_on

        media_urls = self.media_monitor.setup_monitoring(mock_page)

        assert len(media_urls) == 1
        assert media_urls[0] == "https://video.twimg.com/test_video.mp4"
        mock_logger.debug.assert_called_once()

    @patch('fetcher.media_monitor.logger')
    def test_setup_monitoring_captures_image_url(self, mock_logger):
        """Test that image URLs are captured during monitoring."""
        mock_page = Mock()
        captured_requests = []

        def mock_on(event, handler):
            mock_request = Mock()
            mock_request.url = "https://pbs.twimg.com/media/test_image.jpg"
            captured_requests.append(mock_request)
            handler(mock_request)

        mock_page.on = mock_on

        media_urls = self.media_monitor.setup_monitoring(mock_page)

        assert len(media_urls) == 1
        assert media_urls[0] == "https://pbs.twimg.com/media/test_image.jpg"

    @patch('fetcher.media_monitor.logger')
    def test_setup_monitoring_avoids_duplicates(self, mock_logger):
        """Test that duplicate URLs are not captured."""
        mock_page = Mock()
        captured_requests = []

        def mock_on(event, handler):
            # Simulate multiple requests with same URL
            for _ in range(3):
                mock_request = Mock()
                mock_request.url = "https://video.twimg.com/test.mp4"
                captured_requests.append(mock_request)
                handler(mock_request)

        mock_page.on = mock_on

        media_urls = self.media_monitor.setup_monitoring(mock_page)

        assert len(media_urls) == 1  # Should only capture once
        assert media_urls[0] == "https://video.twimg.com/test.mp4"

    @patch('fetcher.media_monitor.logger')
    def test_setup_monitoring_ignores_non_media_urls(self, mock_logger):
        """Test that non-media URLs are ignored."""
        mock_page = Mock()
        captured_requests = []

        def mock_on(event, handler):
            urls = [
                "https://twitter.com/api/timeline",
                "https://x.com/user/profile",
                "https://example.com/script.js",
                "https://video.twimg.com/test.mp4",  # This should be captured
            ]
            for url in urls:
                mock_request = Mock()
                mock_request.url = url
                captured_requests.append(mock_request)
                handler(mock_request)

        mock_page.on = mock_on

        media_urls = self.media_monitor.setup_monitoring(mock_page)

        assert len(media_urls) == 1
        assert media_urls[0] == "https://video.twimg.com/test.mp4"

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

    def test_add_media_to_tweet_data_no_media(self):
        """Test adding media to tweet data when no media URLs provided."""
        tweet_data = {"text": "Test tweet", "media_count": 0}

        result = self.media_monitor.add_media_to_tweet_data(tweet_data, [])

        assert result == tweet_data
        assert result["media_count"] == 0

    def test_add_media_to_tweet_data_new_media(self):
        """Test adding new media URLs to tweet data."""
        tweet_data = {
            "text": "Test tweet",
            "media_count": 0,
            "media_links": "",
            "media_types": "[]"
        }

        media_urls = [
            "https://video.twimg.com/test.mp4",
            "https://pbs.twimg.com/media/test.jpg"
        ]

        result = self.media_monitor.add_media_to_tweet_data(tweet_data, media_urls)

        assert result["media_count"] == 2
        assert result["media_links"] == "https://video.twimg.com/test.mp4,https://pbs.twimg.com/media/test.jpg"
        assert result["media_types"] == "['video', 'image']"
        assert len(media_urls) == 0  # Should be cleared

    def test_add_media_to_tweet_data_with_existing_media(self):
        """Test adding media URLs to tweet data that already has media."""
        tweet_data = {
            "text": "Test tweet",
            "media_count": 1,
            "media_links": "https://existing.com/old.jpg",
            "media_types": "['image']"
        }

        media_urls = [
            "https://video.twimg.com/test.mp4",
            "https://existing.com/old.jpg"  # Duplicate - should not be added
        ]

        result = self.media_monitor.add_media_to_tweet_data(tweet_data, media_urls)

        assert result["media_count"] == 2
        assert result["media_links"] == "https://existing.com/old.jpg,https://video.twimg.com/test.mp4"
        assert result["media_types"] == "['image', 'video']"
        assert len(media_urls) == 0  # Should be cleared

    def test_add_media_to_tweet_data_malformed_existing_types(self):
        """Test handling malformed existing media types."""
        tweet_data = {
            "text": "Test tweet",
            "media_count": 1,
            "media_links": "https://existing.com/old.jpg",
            "media_types": "invalid-json"
        }

        media_urls = ["https://video.twimg.com/test.mp4"]

        result = self.media_monitor.add_media_to_tweet_data(tweet_data, media_urls)

        assert result["media_count"] == 2
        assert "https://video.twimg.com/test.mp4" in result["media_links"]
        assert len(media_urls) == 0

    @patch('fetcher.media_monitor.logger')
    def test_add_media_to_tweet_data_logging(self, mock_logger):
        """Test that adding media URLs triggers appropriate logging."""
        tweet_data = {"text": "Test tweet", "media_count": 0}
        media_urls = ["https://video.twimg.com/test.mp4"]

        result = self.media_monitor.add_media_to_tweet_data(tweet_data, media_urls)

        mock_logger.info.assert_called_once_with("Added 1 media URLs from network monitoring")

    def test_get_media_monitor(self):
        """Test getting the global media monitor instance."""
        from fetcher.media_monitor import get_media_monitor

        monitor = get_media_monitor()
        assert isinstance(monitor, MediaMonitor)
        assert monitor is not None