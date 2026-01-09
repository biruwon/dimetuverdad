"""
Media URL monitoring for the fetcher module.

Handles network request monitoring to capture media URLs during page interactions.
"""

import re
from typing import List, Optional
from .logging_config import get_logger

logger = get_logger('media_monitor')


# P4 Optimization: URL patterns for fast video extraction from poster
# Twitter video poster URLs contain video IDs that can be used to construct video URLs
VIDEO_POSTER_PATTERNS = [
    # Pattern: amplify_video_thumb/{VIDEO_ID}/img/...
    (r'amplify_video_thumb/(\d+)/', 'https://video.twimg.com/amplify_video/{video_id}/vid/avc1/720x720/video.mp4'),
    # Pattern: tweet_video_thumb/{TWEET_ID}/{VIDEO_ID}/img/...
    (r'tweet_video_thumb/\d+/(\d+)/', 'https://video.twimg.com/tweet_video/{video_id}.mp4'),
    # Pattern: ext_tw_video/{VIDEO_ID}/...
    (r'ext_tw_video/(\d+)/', 'https://video.twimg.com/ext_tw_video/{video_id}/vid/avc1/720x720/video.mp4'),
]


def extract_video_url_from_poster(poster_url: str) -> Optional[str]:
    """
    P4 Performance Optimization: Try to construct video URL from poster thumbnail URL.
    
    Twitter video thumbnails often contain the video ID, which can be used to
    construct the actual video URL without network monitoring or hover.
    
    Args:
        poster_url: Video poster/thumbnail URL
        
    Returns:
        Constructed video URL if pattern matches, None otherwise
    """
    if not poster_url:
        return None
    
    for pattern, url_template in VIDEO_POSTER_PATTERNS:
        match = re.search(pattern, poster_url)
        if match:
            video_id = match.group(1)
            video_url = url_template.format(video_id=video_id)
            print(f"🎬 P4: Extracted video URL from poster pattern: {video_url[:80]}...")
            return video_url
    
    return None

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

            # Only capture video files, not images or thumbnails
            video_extensions = ['.mp4', '.m3u8', '.webm', '.mov']
            video_keywords = ['video.twimg.com']
            
            # Exclude image extensions and video thumbnails
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
            is_image = any(ext in url for ext in image_extensions)
            is_thumbnail = 'amplify_video_thumb' in url or 'thumb' in url

            has_video_extension = any(ext in url for ext in video_extensions)
            has_video_keyword = any(keyword in url for keyword in video_keywords)

            # Only capture if it's a video and NOT an image/thumbnail
            if (has_video_extension or has_video_keyword) and not is_image and not is_thumbnail:
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
            
            # Exclude image extensions and video thumbnails
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
            is_image = any(ext in url for ext in image_extensions)
            is_thumbnail = 'amplify_video_thumb' in url or 'thumb' in url

            has_video_extension = any(ext in url for ext in video_extensions)
            has_video_keyword = any(keyword in url for keyword in video_keywords)

            # Only capture if it's a video and NOT an image/thumbnail
            if (has_video_extension or has_video_keyword) and not is_image and not is_thumbnail:
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

    def try_extract_video_from_poster(self, page) -> Optional[str]:
        """
        P4 Performance Optimization: Try to extract video URL from poster thumbnail.
        
        This is much faster than network monitoring + hover (saves 8-10 seconds per video).
        
        Args:
            page: Playwright page object
            
        Returns:
            Video URL if extracted from poster, None otherwise
        """
        try:
            video_element = page.query_selector('[data-testid="videoPlayer"] video')
            if video_element:
                poster_url = video_element.get_attribute('poster')
                if poster_url:
                    video_url = extract_video_url_from_poster(poster_url)
                    if video_url:
                        return video_url
            
            # Also try video elements without the nested structure
            video_element = page.query_selector('video[poster*="twimg"]')
            if video_element:
                poster_url = video_element.get_attribute('poster')
                if poster_url:
                    video_url = extract_video_url_from_poster(poster_url)
                    if video_url:
                        return video_url
        except Exception as e:
            print(f"⚠️ P4: Error extracting video from poster: {e}")
        
        return None

    def setup_and_monitor(self, page, scroller) -> List[str]:
        """
        Set up network monitoring and trigger video loading if needed.
        Captures the first video URL found.
        
        P4 Optimization: Tries fast poster extraction before slow network monitoring.

        Args:
            page: Playwright page object
            scroller: Scroller instance for delays

        Returns:
            List containing at most 1 video URL
        """
        # P4 Optimization: Try fast extraction from poster first (saves 8-10 seconds)
        fast_video_url = self.try_extract_video_from_poster(page)
        if fast_video_url:
            print(f"🎬 P4: Video URL extracted from poster (skipped network monitoring)")
            return [fast_video_url]
        
        # Fall back to network monitoring if fast extraction failed
        print("🎬 P4: Fast extraction failed, falling back to network monitoring...")
        
        # Set up monitoring BEFORE any interactions
        video_urls = self.setup_monitoring(page)
        
        # Wait for automatic video loading (reduced wait time)
        print("⏳ Waiting for automatic video loading...")
        scroller.delay(2.0, 3.0)  # Reduced from 3.0-5.0
        
        # If no videos captured, try minimal hover to trigger loading
        if not video_urls:
            print("🎬 No videos captured, trying minimal hover to trigger loading...")
            try:
                video_element = page.query_selector('[data-testid="videoPlayer"]')
                if video_element:
                    video_element.hover()
                    scroller.delay(1.5, 2.5)  # Reduced from 2.0-3.0
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
        Deduplicates URLs to prevent saving the same media twice.
        
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
        
        # Deduplicate: collect unique URLs preserving order
        unique_urls = []
        seen = set()
        
        # First add all existing URLs (from DOM extraction)
        for url in existing_urls:
            if url and url not in seen:
                unique_urls.append(url)
                seen.add(url)
        
        # Then add captured video URLs if not already present
        for video_url in video_urls[:1]:  # Only process first video URL
            if video_url and video_url not in seen:
                unique_urls.append(video_url)
                seen.add(video_url)
                logger.info(f"Added video URL from network monitoring: {video_url[:100]}...")
        
        # Update tweet data with deduplicated URLs
        if unique_urls != existing_urls:
            tweet_data['media_links'] = ','.join(unique_urls)
            tweet_data['media_count'] = len(unique_urls)
        
        return tweet_data

# Global media monitor instance
media_monitor = MediaMonitor()

def get_media_monitor() -> MediaMonitor:
    """Get the global media monitor instance."""
    return media_monitor