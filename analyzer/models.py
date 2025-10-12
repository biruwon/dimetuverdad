"""
Data models for the analyzer system.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class ContentAnalysis:
    """Content analysis result structure with multi-category support."""
    # Post metadata (platform-agnostic)
    post_id: str
    post_url: str
    author_username: str
    post_content: str
    analysis_timestamp: str

    # Content categories (consolidated and multi-category support)
    category: str  # Primary category (backward compatibility)
    categories_detected: List[str] = None  # All detected categories

    # Analysis results
    llm_explanation: str = ""
    analysis_method: str = "pattern"  # "pattern", "llm", or "gemini"

    # Media analysis fields
    media_urls: List[str] = None  # List of media URLs
    media_analysis: str = ""      # Gemini multimodal analysis result
    media_type: str = ""          # "image", "video", or ""
    multimodal_analysis: bool = False  # Whether media was analyzed

    # Technical data
    pattern_matches: List[Dict] = None
    topic_classification: Dict = None
    analysis_json: str = ""

    # Performance metrics
    analysis_time_seconds: float = 0.0  # Total analysis time
    model_used: str = ""                # Which model was used (gpt-oss:20b, gemini-2.5-flash, etc.)
    tokens_used: int = 0                # Approximate tokens used (if available)

    def __post_init__(self):
        # Initialize lists to avoid None values
        if self.categories_detected is None:
            self.categories_detected = []
        if self.pattern_matches is None:
            self.pattern_matches = []
        if self.media_urls is None:
            self.media_urls = []

    @property
    def has_multiple_categories(self) -> bool:
        """Check if content was classified with multiple categories."""
        return len(self.categories_detected) > 1

    def get_secondary_categories(self) -> List[str]:
        """Get all categories except the primary one."""
        return [cat for cat in self.categories_detected if cat != self.category]