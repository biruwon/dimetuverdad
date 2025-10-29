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
        Set up network request monitoring on a page to capture the first video URL.

        Args:
            page: Playwright page object

        Returns:
            List that will be populated with captured media URLs (max 1 video URL)
        """
        media_urls = []

        def handle_request(request):
            # Only capture if we haven't found a video URL yet
            if media_urls:
                return
                
            url = request.url.lower()

            # Only capture video files, not images
            video_extensions = ['.mp4', '.m3u8', '.webm', '.mov']
            video_keywords = ['video.twimg.com']

            has_video_extension = any(ext in url for ext in video_extensions)
            has_video_keyword = any(keyword in url for keyword in video_keywords)

            if has_video_extension or has_video_keyword:
                media_urls.append(request.url)
                print(f"🎥 CAPTURED VIDEO URL: {request.url[:100]}...")
                logger.debug(f"Captured first video URL: {request.url[:100]}...")

        page.on("request", handle_request)
        
        # Also try response events as backup
        def handle_response(response):
            if media_urls:
                return
                
            url = response.url.lower()
            video_extensions = ['.mp4', '.m3u8', '.webm', '.mov']
            video_keywords = ['video.twimg.com']

            has_video_extension = any(ext in url for ext in video_extensions)
            has_video_keyword = any(keyword in url for keyword in video_keywords)

            if has_video_extension or has_video_keyword:
                media_urls.append(response.url)
                print(f"🎥 CAPTURED VIDEO URL (response): {response.url[:100]}...")
        
        page.on("response", handle_response)
        print(f"🎥 Network monitoring set up on page")
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

    def setup_and_monitor(self, page, scroller) -> List[str]:
        """
        Set up network monitoring and trigger video loading if needed.
        Captures the first video URL found.

        Args:
            page: Playwright page object
            scroller: Scroller instance for delays

        Returns:
            List containing at most 1 video URL
        """
        # Set up monitoring BEFORE any interactions
        video_urls = self.setup_monitoring(page)
        
        # Wait for automatic video loading
        print("⏳ Waiting for automatic video loading...")
        scroller.delay(3.0, 5.0)
        
        # If no videos captured, try minimal hover to trigger loading
        if not video_urls:
            print("🎬 No videos captured, trying minimal hover to trigger loading...")
            try:
                video_element = page.query_selector('[data-testid="videoPlayer"]')
                if video_element:
                    video_element.hover()
                    scroller.delay(2.0, 3.0)  # Wait for video requests after hover
                    print(f"🎬 Hovered over video element, captured {len(video_urls)} video URLs")
                else:
                    print("🎬 No video element found to hover")
            except Exception as e:
                print(f"⚠️ Video hover failed: {e}")
        else:
            print(f"🎬 Videos captured during initial wait: {len(video_urls)} items")
        
        return video_urls

    def process_video_urls(self, video_urls: List[str], tweet_data: dict) -> dict:
        """
        Process captured video URL (first one found) and combine with existing tweet media.
        
        Args:
            video_urls: List of captured video URLs (should contain at most 1 URL)
            tweet_data: Tweet data dictionary to update
            
        Returns:
            Updated tweet data dictionary
        """
        if not video_urls:
            return tweet_data

        # Get existing media from DOM extraction
        existing_media = tweet_data.get('media_links', '')
        existing_urls = existing_media.split(',') if existing_media else []
        
        # Take only the first video URL
        video_url = video_urls[0]
        
        # Add video URL if not already captured by DOM extraction
        if video_url not in existing_urls:
            combined_urls = existing_urls + [video_url]
            tweet_data['media_links'] = ','.join(combined_urls)
            tweet_data['media_count'] = len(combined_urls)
            
            logger.info(f"Added video URL from network monitoring: {video_url[:100]}...")
        
        return tweet_data

# Global media monitor instance
media_monitor = MediaMonitor()

def get_media_monitor() -> MediaMonitor:
    """Get the global media monitor instance."""
    return media_monitor