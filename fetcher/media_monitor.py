"""
Media URL monitoring for the fetcher module.

Handles network request monitoring to capture media URLs during page interactions.
"""

from typing import List
from .logging_config import get_logger

logger = get_logger('media_monitor')

class MediaMonitor:
    """Monitors network requests to capture media URLs."""

    def __init__(self):
        self.media_extensions = ['.mp4', '.m3u8', '.webm', '.mov', '.jpg', '.jpeg', '.png', '.gif', '.webp']
        self.media_keywords = ['video.twimg.com', 'pbs.twimg.com', 'mediadelivery']

    def setup_monitoring(self, page) -> List[str]:
        """
        Set up network request monitoring on a page.

        Args:
            page: Playwright page object

        Returns:
            List that will be populated with captured media URLs
        """
        media_urls = []

        def handle_request(request):
            url = request.url.lower()

            # Check for media file extensions
            has_media_extension = any(ext in url for ext in self.media_extensions)

            # Check for media-related keywords
            has_media_keyword = any(keyword in url for keyword in self.media_keywords)

            if has_media_extension or has_media_keyword:
                if request.url not in media_urls:
                    media_urls.append(request.url)
                    media_type = self._determine_media_type(request.url)
                    logger.debug(f"Captured {media_type} URL: {request.url[:100]}...")

        page.on("request", handle_request)
        return media_urls

    def _determine_media_type(self, url: str) -> str:
        """
        Determine the media type from URL.

        Args:
            url: Media URL

        Returns:
            str: 'video' or 'image'
        """
        video_extensions = ['.mp4', '.m3u8', '.webm', '.mov']
        return "video" if any(ext in url.lower() for ext in video_extensions) else "image"

    def add_media_to_tweet_data(self, tweet_data: dict, media_urls: List[str]) -> dict:
        """
        Add captured media URLs to tweet data.

        Args:
            tweet_data: Tweet data dictionary
            media_urls: List of captured media URLs

        Returns:
            Updated tweet data dictionary
        """
        if not media_urls:
            return tweet_data

        # Get existing media data
        existing_media = tweet_data.get('media_links', '')
        existing_urls = existing_media.split(',') if existing_media else []
        existing_count = tweet_data.get('media_count', 0)
        existing_types = tweet_data.get('media_types', '[]')
        if isinstance(existing_types, str):
            try:
                existing_types = eval(existing_types)  # Safe since we control the data
            except:
                existing_types = []

        # Add new media URLs
        new_urls = []
        new_types = []

        for media_url in media_urls:
            if media_url not in existing_urls:
                new_urls.append(media_url)
                new_types.append(self._determine_media_type(media_url))

        # Update tweet data
        if new_urls:
            combined_urls = existing_urls + new_urls
            combined_types = existing_types + new_types

            tweet_data['media_links'] = ','.join(combined_urls)
            tweet_data['media_count'] = len(combined_urls)
            tweet_data['media_types'] = str(combined_types)  # JSON serializable

            logger.info(f"Added {len(new_urls)} media URLs from network monitoring")

        # Clear the media URLs list for next tweet
        media_urls.clear()

        return tweet_data

# Global media monitor instance
media_monitor = MediaMonitor()

def get_media_monitor() -> MediaMonitor:
    """Get the global media monitor instance."""
    return media_monitor