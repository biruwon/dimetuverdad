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
        """Test setup_and_monitor method when P4 fast extraction fails."""
        mock_page = Mock()
        mock_scroller = Mock()
        mock_scroller.delay = Mock()
        
        # Mock P4 fast extraction to fail (return None)
        self.media_monitor.try_extract_video_from_poster = Mock(return_value=None)
        
        # Mock setup_monitoring to return empty list initially
        self.media_monitor.setup_monitoring = Mock(return_value=[])
        
        # Mock video element for hover fallback
        mock_video_element = Mock()
        mock_page.query_selector.return_value = mock_video_element
        
        result = self.media_monitor.setup_and_monitor(mock_page, mock_scroller)
        
        # Should first try P4 fast extraction
        self.media_monitor.try_extract_video_from_poster.assert_called_once_with(mock_page)
        # Should call setup_monitoring as fallback
        self.media_monitor.setup_monitoring.assert_called_once_with(mock_page)
        # Should call delay twice (once for wait, once after hover)
        assert mock_scroller.delay.call_count == 2
        # Should query for video element to hover
        mock_page.query_selector.assert_called_with('[data-testid="videoPlayer"]')
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


# ============================================================================
# P4: Skip Video Hover Tests
# ============================================================================

class TestExtractVideoUrlFromPoster:
    """Tests for P4 fast video URL extraction from poster."""
    
    def test_amplify_video_thumb_pattern(self):
        """Should extract video ID from amplify_video_thumb pattern."""
        from fetcher.media_monitor import extract_video_url_from_poster
        
        poster_url = "https://pbs.twimg.com/amplify_video_thumb/1234567890/img/abc123.jpg"
        result = extract_video_url_from_poster(poster_url)
        
        assert result is not None
        assert "1234567890" in result
        assert "video.twimg.com" in result
        assert ".mp4" in result
    
    def test_tweet_video_thumb_pattern(self):
        """Should extract video ID from tweet_video_thumb pattern."""
        from fetcher.media_monitor import extract_video_url_from_poster
        
        poster_url = "https://pbs.twimg.com/tweet_video_thumb/98765/5555555/img/thumb.jpg"
        result = extract_video_url_from_poster(poster_url)
        
        assert result is not None
        assert "5555555" in result
        assert ".mp4" in result
    
    def test_ext_tw_video_pattern(self):
        """Should extract video ID from ext_tw_video pattern."""
        from fetcher.media_monitor import extract_video_url_from_poster
        
        poster_url = "https://pbs.twimg.com/ext_tw_video/7777777/pu/img/preview.jpg"
        result = extract_video_url_from_poster(poster_url)
        
        assert result is not None
        assert "7777777" in result
    
    def test_no_match_returns_none(self):
        """Should return None for unrecognized poster patterns."""
        from fetcher.media_monitor import extract_video_url_from_poster
        
        poster_url = "https://pbs.twimg.com/media/regular_image.jpg"
        result = extract_video_url_from_poster(poster_url)
        
        assert result is None
    
    def test_empty_url_returns_none(self):
        """Should return None for empty or None input."""
        from fetcher.media_monitor import extract_video_url_from_poster
        
        assert extract_video_url_from_poster("") is None
        assert extract_video_url_from_poster(None) is None
    
    def test_real_world_url_format(self):
        """Should handle real-world URL with query params."""
        from fetcher.media_monitor import extract_video_url_from_poster
        
        poster_url = "https://pbs.twimg.com/amplify_video_thumb/1853445123456/img/abc.jpg?format=jpg&name=small"
        result = extract_video_url_from_poster(poster_url)
        
        assert result is not None
        assert "1853445123456" in result


class TestTryExtractVideoFromPoster:
    """Tests for MediaMonitor.try_extract_video_from_poster method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.media_monitor = MediaMonitor()
    
    def test_extracts_from_video_player_nested_video(self):
        """Should extract from [data-testid='videoPlayer'] video element."""
        mock_video = Mock()
        mock_video.get_attribute = Mock(return_value="https://pbs.twimg.com/amplify_video_thumb/123456/img/test.jpg")
        
        mock_page = Mock()
        mock_page.query_selector = Mock(side_effect=lambda s: mock_video if 'videoPlayer' in s else None)
        
        result = self.media_monitor.try_extract_video_from_poster(mock_page)
        
        assert result is not None
        assert "123456" in result
        assert ".mp4" in result
    
    def test_extracts_from_plain_video_element(self):
        """Should extract from video[poster*='twimg'] if videoPlayer selector fails."""
        mock_video = Mock()
        mock_video.get_attribute = Mock(return_value="https://pbs.twimg.com/amplify_video_thumb/789012/img/test.jpg")
        
        mock_page = Mock()
        # First selector returns None, second returns the video
        mock_page.query_selector = Mock(side_effect=lambda s: mock_video if 'poster' in s else None)
        
        result = self.media_monitor.try_extract_video_from_poster(mock_page)
        
        assert result is not None
        assert "789012" in result
    
    def test_returns_none_when_no_video_element(self):
        """Should return None when no video element found."""
        mock_page = Mock()
        mock_page.query_selector = Mock(return_value=None)
        
        result = self.media_monitor.try_extract_video_from_poster(mock_page)
        
        assert result is None
    
    def test_returns_none_when_poster_has_no_pattern(self):
        """Should return None when poster URL doesn't match known patterns."""
        mock_video = Mock()
        mock_video.get_attribute = Mock(return_value="https://pbs.twimg.com/media/random_image.jpg")
        
        mock_page = Mock()
        mock_page.query_selector = Mock(return_value=mock_video)
        
        result = self.media_monitor.try_extract_video_from_poster(mock_page)
        
        assert result is None
    
    def test_handles_exception_gracefully(self):
        """Should handle exceptions and return None."""
        mock_page = Mock()
        mock_page.query_selector = Mock(side_effect=Exception("Network error"))
        
        result = self.media_monitor.try_extract_video_from_poster(mock_page)
        
        assert result is None


class TestSetupAndMonitorP4:
    """Tests for P4 optimization in setup_and_monitor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.media_monitor = MediaMonitor()
    
    def test_skips_network_monitoring_when_fast_extraction_succeeds(self):
        """P4: Should skip network monitoring if poster extraction succeeds."""
        mock_video = Mock()
        mock_video.get_attribute = Mock(return_value="https://pbs.twimg.com/amplify_video_thumb/999999/img/test.jpg")
        
        mock_page = Mock()
        mock_page.query_selector = Mock(return_value=mock_video)
        mock_page.on = Mock()  # Track if monitoring is set up
        
        mock_scroller = Mock()
        
        result = self.media_monitor.setup_and_monitor(mock_page, mock_scroller)
        
        # Should return the fast-extracted URL
        assert len(result) == 1
        assert "999999" in result[0]
        
        # Network monitoring should NOT be set up (page.on not called)
        mock_page.on.assert_not_called()
        
        # Scroller delays should NOT be called (no waiting for network)
        mock_scroller.delay.assert_not_called()
    
    def test_falls_back_to_network_monitoring(self):
        """Should fall back to network monitoring when fast extraction fails."""
        mock_page = Mock()
        mock_page.query_selector = Mock(return_value=None)  # No video element
        mock_page.on = Mock()
        
        mock_scroller = Mock()
        
        result = self.media_monitor.setup_and_monitor(mock_page, mock_scroller)
        
        # Network monitoring should be set up
        assert mock_page.on.call_count >= 1
        
        # Scroller should be called for waiting
        assert mock_scroller.delay.called
    
    def test_p4_performance_expectation_no_hover_when_fast_succeeds(self):
        """P4 saves 8-10 seconds by skipping hover when poster extraction works."""
        mock_video = Mock()
        mock_video.get_attribute = Mock(return_value="https://pbs.twimg.com/amplify_video_thumb/123/img/test.jpg")
        
        mock_page = Mock()
        # Return video for poster extraction
        mock_page.query_selector = Mock(return_value=mock_video)
        mock_page.on = Mock()
        
        mock_scroller = Mock()
        
        # Run the method
        result = self.media_monitor.setup_and_monitor(mock_page, mock_scroller)
        
        # Verify NO hover was triggered
        mock_video.hover.assert_not_called()
        
        # Verify we got a result (fast path worked)
        assert len(result) == 1