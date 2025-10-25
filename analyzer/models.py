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
    category: str  # Primary category
    categories_detected: List[str] = None  # All detected categories

    # Dual analysis results
    local_explanation: str = ""        # Local LLM explanation (gpt-oss:20b)
    external_explanation: str = ""     # External LLM explanation (Gemini, optional)
    analysis_stages: str = ""          # Comma-separated stages executed (e.g., "pattern,local_llm,external")
    external_analysis_used: bool = False  # Whether external analysis was triggered

    # Media analysis fields
    media_urls: List[str] = None  # List of media URLs
    media_type: str = ""          # "image", "video", or ""

    # Technical data
    pattern_matches: List[Dict] = None
    topic_classification: Dict = None
    analysis_json: str = ""

    # Performance metrics
    analysis_time_seconds: float = 0.0  # Total analysis time
    model_used: str = ""                # Which model was used (gpt-oss:20b, gemini-2.5-flash, etc.)
    tokens_used: int = 0                # Approximate tokens used (if available)

    # Evidence verification fields
    verification_data: Optional[Dict] = None     # Verification results from retrieval system
    verification_confidence: float = 0.0         # Confidence score from verification (0.0-1.0)

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

    @property
    def multimodal_analysis(self) -> bool:
        """Whether media was analyzed (derived from media_urls)."""
        return bool(self.media_urls)

    def get_secondary_categories(self) -> List[str]:
        """Get all categories except the primary one."""
        return [cat for cat in self.categories_detected if cat != self.category]
    
    def get_best_explanation(self) -> str:
        """
        Get the best available explanation.
        Prefers external over local, falls back to local if external not available.
        """
        if self.external_explanation and len(self.external_explanation.strip()) > 0:
            return self.external_explanation
        return self.local_explanation
    
    def has_dual_explanations(self) -> bool:
        """Check if both local and external explanations are available."""
        return (
            len(self.local_explanation.strip()) > 0 and 
            len(self.external_explanation.strip()) > 0
        )