"""
Multimodal content analysis utilities.

NOTE: The MultimodalAnalyzer class has been deprecated in favor of the new
dual-flow architecture (LocalLLMAnalyzer + ExternalAnalyzer + AnalysisFlowManager).
This file now contains only utility functions.
"""

from typing import List


def extract_media_type(media_urls: List[str]) -> str:
    """
    Extract media type from a list of media URLs.

    Args:
        media_urls: List of media URLs to analyze

    Returns:
        Media type string: "image", "video", "mixed", "unknown", or ""
    """
    if not media_urls:
        return ""

    media_types = set()

    for url in media_urls:
        url_lower = url.lower()

        # Check for image formats
        if any(fmt in url_lower for fmt in ['.jpg', '.jpeg', '.png', '.gif', '.webp', 'format=jpg', 'format=jpeg', 'format=png']):
            media_types.add("image")
        # Check for video formats
        elif any(fmt in url_lower for fmt in ['.mp4', '.webm', '.avi', '.mov', 'video', '/vid/']):
            media_types.add("video")
        else:
            media_types.add("unknown")

    # Determine overall type
    if len(media_types) == 1:
        return media_types.pop()
    elif "video" in media_types:
        return "mixed"  # If we have both video and other types
    elif "image" in media_types:
        return "mixed"  # If we have both image and unknown
    else:
        return "unknown"
