"""
Multimodal content analysis functionality for images and media.
"""

import json
from typing import Dict, Tuple, Optional, List
from datetime import datetime

from .gemini_multimodal import GeminiMultimodal
from .models import ContentAnalysis
from .categories import Categories
from .constants import AnalysisMethods, ErrorMessages


class MultimodalAnalyzer:
    """
    Handles multimodal content analysis combining text and media analysis.

    This class manages the integration of text analysis with multimodal
    capabilities for images, videos, and other media content.
    """

    def __init__(self, gemini_analyzer: Optional[GeminiMultimodal] = None,
                 verbose: bool = False):
        """
        Initialize multimodal analyzer.

        Args:
            gemini_analyzer: Gemini multimodal analyzer instance (created if None)
            verbose: Whether to enable verbose logging
        """
        self.gemini_analyzer = gemini_analyzer or GeminiMultimodal()
        self.verbose = verbose

    def analyze_with_media(self, tweet_id: str, tweet_url: str, username: str,
                          content: str, media_urls: List[str]) -> ContentAnalysis:
        """
        Perform multimodal analysis combining text and media.

        Args:
            tweet_id: Twitter tweet ID
            tweet_url: Twitter tweet URL
            username: Twitter username
            content: Tweet content
            media_urls: List of media URLs to analyze

        Returns:
            ContentAnalysis result with multimodal insights
        """
        if self.verbose:
            print(f"\n🖼️  Multimodal analysis: @{username}")
            print(f"📝 Content: {content[:80]}...")
            print(f"🖼️  Media URLs: {len(media_urls)} items")

        # Step 1: Analyze media content
        media_analysis = self._analyze_media_content(media_urls, content)

        # Step 2: Combine with text analysis (if available)
        combined_category = self._determine_combined_category(content, media_analysis)

        # Step 3: Generate multimodal explanation
        multimodal_explanation = self._generate_multimodal_explanation(
            content, media_analysis, combined_category
        )

        # Step 4: Build analysis data
        analysis_data = self._build_multimodal_analysis_data(media_analysis)

        # Extract media analysis text
        media_result = media_analysis.get('media_analysis', {})
        media_analysis_text = media_result.get('description', '') if media_result else ''

        return ContentAnalysis(
            tweet_id=tweet_id,
            tweet_url=tweet_url,
            username=username,
            tweet_content=content,
            analysis_timestamp=datetime.now().isoformat(),
            category=combined_category,
            categories_detected=[combined_category],  # Multimodal focuses on primary category
            llm_explanation=multimodal_explanation,
            analysis_method=AnalysisMethods.MULTIMODAL.value,
            pattern_matches=[],  # Pattern matching not applicable for multimodal
            topic_classification=analysis_data['topic_classification'],
            analysis_json=json.dumps(analysis_data, ensure_ascii=False, default=str),
            media_urls=media_urls,
            media_analysis=media_analysis_text,
            media_type=extract_media_type(media_urls),
            multimodal_analysis=True
        )

    def _analyze_media_content(self, media_urls: List[str], text_context: str) -> Dict:
        """
        Analyze media content using multimodal capabilities.

        Args:
            media_urls: List of media URLs to analyze
            text_context: Associated text content for context

        Returns:
            Dictionary containing media analysis results
        """
        if not media_urls:
            return {'media_analysis': None, 'error': 'No media URLs provided'}

        try:
            if self.verbose:
                print(f"🖼️  Analyzing {len(media_urls)} media items...")

            # Use Gemini's URL selection logic to pick the best media URL
            selected_media_url = self.gemini_analyzer._select_media_url(media_urls)

            media_analysis_result, analysis_time = self.gemini_analyzer.analyze_multimodal_content(
                [selected_media_url], text_context
            )

            if media_analysis_result is None:
                return {'media_analysis': None, 'error': 'Media analysis failed'}

            # Convert tuple result to dict format expected by the rest of the code
            # Parse structured response: "CATEGORÍA: [category]\nEXPLICACIÓN: [explanation]"
            analysis_text = media_analysis_result.strip()

            # Extract category from structured response
            category = Categories.GENERAL  # default
            explanation = analysis_text  # fallback to full text

            if "CATEGORÍA:" in analysis_text.upper():
                lines = analysis_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.upper().startswith("CATEGORÍA:"):
                        # Extract category after "CATEGORÍA:"
                        category_part = line.split(":", 1)[1].strip().lower()
                        # Validate it's a known category
                        if category_part in [cat.lower() for cat in Categories.get_all_categories()]:
                            # Find the exact case category
                            for cat in Categories.get_all_categories():
                                if cat.lower() == category_part:
                                    category = cat
                                    break
                        break

                # Extract explanation part
                explanation_start = analysis_text.upper().find("EXPLICACIÓN:")
                if explanation_start != -1:
                    explanation = analysis_text[explanation_start + len("EXPLICACIÓN:"):].strip()

            media_analysis = {
                'description': explanation,
                'category': category
            }

            if self.verbose:
                print(f"🖼️  Media analysis complete: {media_analysis.get('category', 'unknown')}")

            return {'media_analysis': media_analysis}

        except Exception as e:
            if self.verbose:
                print(f"❌ Media analysis error: {e}")
            return {'media_analysis': None, 'error': str(e)}

    def _determine_combined_category(self, text_content: str, media_analysis: Dict) -> str:
        """
        Determine the final category combining text and media analysis.

        Args:
            text_content: Original text content
            media_analysis: Results from media analysis

        Returns:
            Combined category determination
        """
        media_result = media_analysis.get('media_analysis')

        if not media_result:
            # Fallback to general if media analysis failed
            return Categories.GENERAL

        # Use media analysis category if available
        media_category = media_result.get('category')
        if media_category and media_category != Categories.GENERAL:
            if self.verbose:
                print(f"🎯 Media-determined category: {media_category}")
            return media_category

        # Fallback to general
        return Categories.GENERAL

    def _generate_multimodal_explanation(self, content: str, media_analysis: Dict, category: str) -> str:
        """
        Generate explanation combining text and media analysis.

        Args:
            content: Text content
            media_analysis: Media analysis results
            category: Determined category

        Returns:
            Combined multimodal explanation
        """
        media_result = media_analysis.get('media_analysis')

        if not media_result:
            error_msg = media_analysis.get('error', 'Unknown media analysis error')
            return ErrorMessages.MULTIMODAL_ANALYSIS_FAILED.format(error=error_msg)

        try:
            # Extract media insights
            media_description = media_result.get('description', '')
            media_category = media_result.get('category', Categories.GENERAL)

            # Build multimodal explanation
            explanation_parts = []

            if media_description:
                explanation_parts.append(media_description)

            if not explanation_parts:
                return ErrorMessages.MULTIMODAL_EXPLANATION_INSUFFICIENT

            return " ".join(explanation_parts)

        except Exception as e:
            return ErrorMessages.MULTIMODAL_EXPLANATION_EXCEPTION.format(error=str(e))

    def _build_multimodal_analysis_data(self, media_analysis: Dict) -> Dict:
        """
        Build analysis data structure for multimodal results.

        Args:
            media_analysis: Media analysis results

        Returns:
            Analysis data dictionary
        """
        media_result = media_analysis.get('media_analysis', {})

        # Handle case where media_result is None (analysis failed)
        if media_result is None:
            media_result = {}

        return {
            'category': None,  # Set by caller
            'multimodal_analysis': {
                'media_description': media_result.get('description', ''),
                'media_category': media_result.get('category', Categories.GENERAL),
                'analysis_type': 'multimodal'
            },
            'topic_classification': {
                'primary_topic': media_result.get('category', Categories.GENERAL),
                'all_topics': [{'category': media_result.get('category', Categories.GENERAL)}],
                'political_context': []
            },
            'unified_categories': [media_result.get('category', Categories.GENERAL)]
        }


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