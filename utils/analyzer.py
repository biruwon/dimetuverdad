"""
Analyzer utilities for the dimetuverdad project.
Centralized analyzer initialization and common operations.
"""

import sys
import os
from typing import Optional
from pathlib import Path

# Setup paths for imports
current_dir = Path(__file__).resolve().parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from enhanced_analyzer import EnhancedAnalyzer, save_content_analysis, ContentAnalysis

def create_analyzer(use_llm: bool = True, model_priority: str = "balanced", verbose: bool = False) -> EnhancedAnalyzer:
    """Create and return an EnhancedAnalyzer instance with specified parameters."""
    return EnhancedAnalyzer(use_llm=use_llm, model_priority=model_priority, verbose=verbose)

def reanalyze_tweet(tweet_id: str, analyzer: Optional[EnhancedAnalyzer] = None) -> Optional[ContentAnalysis]:
    """Reanalyze a single tweet and return the result."""
    from .database import get_tweet_data, delete_existing_analysis

    # Get tweet data
    tweet_data = get_tweet_data(tweet_id)
    if not tweet_data:
        return None

    # Delete existing analysis
    delete_existing_analysis(tweet_id)

    # Use provided analyzer or create default one
    if analyzer is None:
        analyzer = create_analyzer()

    # Reanalyze
    analysis_result = analyzer.analyze_content(
        tweet_id=tweet_data['tweet_id'],
        tweet_url=f"https://twitter.com/placeholder/status/{tweet_data['tweet_id']}",
        username=tweet_data['username'],
        content=tweet_data['content']
    )

    # Save result
    save_content_analysis(analysis_result)

    return analysis_result