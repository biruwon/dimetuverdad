"""
Analyze Twitter: Comprehensive Twitter content analysis system.
Combines content analysis, database operations, and CLI interface.
"""

import json
import sqlite3
import time
import warnings
import sys
import logging
import argparse
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import our components
from .pattern_analyzer import PatternAnalyzer
from utils.text_utils import normalize_text
from .llm_models import EnhancedLLMPipeline
from .categories import Categories

# Import new modular components
from .config import AnalyzerConfig
from .metrics import MetricsCollector
from .repository import ContentAnalysisRepository
from .text_analyzer import TextAnalyzer
from .multimodal_analyzer import MultimodalAnalyzer
from .models import ContentAnalysis
from .constants import AnalysisMethods, ErrorMessages

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging for multimodal analysis debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# DB path (same as other scripts)
from utils.paths import get_db_path
DB_PATH = get_db_path()


class Analyzer:
    """
    Uses specialized components for different analysis types:
    - TextAnalyzer: Text-only content analysis
    - MultimodalAnalyzer: Media content analysis
    - MetricsCollector: Performance tracking
    - ContentAnalysisRepository: Database operations
    """

    def __init__(self, config: Optional[AnalyzerConfig] = None, verbose: bool = False):
        """
        Initialize analyzer with configuration.

        Args:
            config: Analyzer configuration (created with defaults if None)
            verbose: Whether to enable verbose logging
        """
        self.config = config or AnalyzerConfig()
        self.verbose = verbose

        # Initialize components
        self.metrics = MetricsCollector()
        self.repository = ContentAnalysisRepository(DB_PATH)

        # Initialize analysis components
        self.text_analyzer = TextAnalyzer(config=self.config, verbose=verbose)
        self.multimodal_analyzer = MultimodalAnalyzer(verbose=verbose)

        if self.verbose:
            print("🚀 Iniciando Analyzer con componentes modulares...")
            print("Componentes cargados:")
            print("- ✓ Analizador de texto especializado")
            print("- ✓ Analizador multimodal para medios")
            print("- ✓ Recolector de métricas de rendimiento")
            print("- ✓ Repositorio de análisis de contenido")
            print(f"- ✓ Configuración: {self.config.model_priority} priority")

    def analyze_content(self,
                       tweet_id: str,
                       tweet_url: str,
                       username: str,
                       content: str,
                       media_urls: List[str] = None) -> ContentAnalysis:
        """
        Main content analysis pipeline with conditional multimodal analysis.

        Routes to appropriate analyzer based on content type.
        """
        analysis_start_time = time.time()

        try:
            # Route to appropriate analyzer
            if media_urls and len(media_urls) > 0:
                result = self.multimodal_analyzer.analyze_with_media(
                    tweet_id, tweet_url, username, content, media_urls
                )
            else:
                result = self.text_analyzer.analyze(
                    tweet_id, tweet_url, username, content
                )

            # Track metrics
            analysis_time = time.time() - analysis_start_time
            self.metrics.record_analysis(
                method=result.analysis_method,
                duration=analysis_time,
                category=result.category,
                model_used=self._get_model_name(result),
                is_multimodal=result.analysis_method == AnalysisMethods.MULTIMODAL.value
            )

            # Add performance data to result
            result.analysis_time_seconds = analysis_time
            result.model_used = self._get_model_name(result)

            return result

        except Exception as e:
            # Handle analysis errors gracefully
            if self.verbose:
                print(f"❌ Error en análisis: {e}")
                import traceback
                traceback.print_exc()

            # Return error result
            return ContentAnalysis(
                tweet_id=tweet_id,
                tweet_url=tweet_url,
                username=username,
                tweet_content=content,
                analysis_timestamp=datetime.now().isoformat(),
                category=Categories.GENERAL,
                categories_detected=[Categories.GENERAL],
                llm_explanation=ErrorMessages.ANALYSIS_FAILED.format(error=str(e)),
                analysis_method=AnalysisMethods.ERROR.value,
                pattern_matches=[],
                topic_classification={},
                analysis_json=json.dumps({'error': str(e)}, ensure_ascii=False)
            )

    def _get_model_name(self, result: ContentAnalysis) -> str:
        """Get the model name based on analysis result."""
        if result.analysis_method == AnalysisMethods.MULTIMODAL.value:
            return "gemini-2.5-flash"
        elif result.analysis_method == AnalysisMethods.LLM.value:
            return f"ollama-{self.config.model_priority}"
        else:
            return "pattern-matching"

    def get_metrics_report(self) -> str:
        """Generate comprehensive metrics report."""
        return self.metrics.generate_report()

    def save_analysis(self, analysis: ContentAnalysis) -> bool:
        """
        Save analysis result to database.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.repository.save(analysis)
            return True
        except Exception as e:
            if self.verbose:
                print(f"❌ Error saving analysis: {e}")
            return False

    def get_analysis(self, tweet_id: str) -> Optional[ContentAnalysis]:
        """
        Retrieve analysis result from database.

        Returns:
            ContentAnalysis if found, None otherwise
        """
        try:
            return self.repository.get_by_tweet_id(tweet_id)
        except Exception as e:
            if self.verbose:
                print(f"❌ Error retrieving analysis: {e}")
            return None

    def cleanup_resources(self):
        """Clean up any resources used by the analyzer."""
        try:
            if hasattr(self.text_analyzer, 'llm_pipeline') and self.text_analyzer.llm_pipeline:
                if hasattr(self.text_analyzer.llm_pipeline, 'cleanup'):
                    self.text_analyzer.llm_pipeline.cleanup()
        except Exception as e:
            print(f"⚠️ Warning during cleanup: {e}")

    def print_system_status(self):
        """Print current system status for debugging."""
        print("🔧 ANALYZER SYSTEM STATUS")
        print("=" * 50)
        print(f"🤖 LLM Enabled: {self.config.use_llm}")
        print(f"🔧 Model Priority: {self.config.model_priority}")
        print(f"🧠 LLM Pipeline: {'✓' if self.text_analyzer.llm_pipeline else '❌'}")
        print(f"🔍 Pattern analyzer: {'✓' if self.text_analyzer.pattern_analyzer else '❌'}")
        print(f"🎥 Multimodal analyzer: {'✓' if self.multimodal_analyzer.gemini_analyzer else '❌'}")

        # Check database
        try:
            conn = sqlite3.connect(DB_PATH, timeout=5.0)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM content_analyses")
            count = c.fetchone()[0]
            conn.close()
            print(f"📊 Database: {count} analyses stored")
        except Exception as e:
            print(f"❌ Database: Error - {e}")


def analyze_tweets_from_db(username=None, max_tweets=None, force_reanalyze=False, tweet_id=None):
    """
    Analyze tweets from the database using the analyzer with LLM enabled.

    Args:
        username: Specific username to analyze (None for all)
        max_tweets: Maximum number of tweets to analyze (None for all)
        force_reanalyze: If True, reanalyze already processed tweets (useful when prompts change)
        tweet_id: Specific tweet ID to analyze/reanalyze (overrides other filters)
    """

    print("🔍 Enhanced Tweet Analysis Pipeline")
    print("=" * 50)

    # Special case: specific tweet ID requested - use reanalyze_tweet utility
    if tweet_id:
        print(f"🎯 Reanalyzing specific tweet: {tweet_id}")
        print("🚀 Initializing Analyzer...")
        try:
            config = AnalyzerConfig(use_llm=True, verbose=False)
            analyzer_instance = create_analyzer(config=config, verbose=False)
            print("✅ Analyzer ready!")

            result = reanalyze_tweet(tweet_id, analyzer=analyzer_instance)
            if result:
                print(f"\n📝 Tweet: {tweet_id}")
                print(f"    🏷️ Category: {result.category}")
                print(f"    💭 {result.llm_explanation[:100]}...")
                print(f"    🔍 Method: {result.analysis_method}")
                if result.multimodal_analysis:
                    print(f"    🎥 Multimodal analysis: Yes ({result.media_type})")
                print("\n✅ Analysis complete and saved to database")
            else:
                print(f"❌ Tweet {tweet_id} not found in database")
        except Exception as e:
            print(f"❌ Error reanalyzing tweet: {e}")
            import traceback
            traceback.print_exc()
        return

    # Initialize analyzer for bulk processing
    print("🚀 Initializing Analyzer...")
    try:
        config = AnalyzerConfig(use_llm=True, verbose=False)
        analyzer_instance = create_analyzer(config=config, verbose=False)  # Always use LLM for better explanations
        print("✅ Analyzer ready!")
    except Exception as e:
        print(f"❌ Error initializing analyzer: {e}")
        return

    # Get tweets for analysis
    tweets = analyzer_instance.repository.get_tweets_for_analysis(
        username=username,
        max_tweets=max_tweets,
        force_reanalyze=force_reanalyze
    )

    # Also get count of already analyzed tweets for reporting
    analyzed_count = analyzer_instance.repository.get_analysis_count_by_username(username)

    if not tweets:
        search_desc = "tweets" if force_reanalyze else "unanalyzed tweets"
        print(f"✅ No {search_desc} found{f' for @{username}' if username else ''}")
        if analyzed_count > 0 and not force_reanalyze:
            print(f"📊 Already analyzed: {analyzed_count} tweets")
        return

    tweet_type = "tweets" if force_reanalyze else "unanalyzed tweets"
    print(f"📊 Found {len(tweets)} {tweet_type}")
    if analyzed_count > 0 and not force_reanalyze:
        print(f"📊 Already analyzed: {analyzed_count} tweets")
        print(f"📊 Total tweets: {len(tweets) + analyzed_count}")
    elif force_reanalyze:
        print(f"📊 Will reanalyze ALL selected tweets")

    print(f"🔧 Analysis Mode: LLM + Patterns")
    print()

    # Analyze each tweet
    results = []
    category_counts = {}

    for i, (tweet_id, tweet_url, tweet_username, content, media_links, original_content) in enumerate(tweets, 1):
        print(f"📝 [{i:2d}/{len(tweets)}] Analyzing: {tweet_id}")

        # Combine main content with quoted content if available
        analysis_content = content
        if original_content and original_content.strip():
            analysis_content = f"{content}\n\n[Contenido citado]: {original_content}"
            print(f"    📎 Including quoted tweet content")

        # Parse media URLs
        media_urls = []
        if media_links:
            media_urls = [url.strip() for url in media_links.split(',') if url.strip()]

        if media_urls:
            print(f"    🖼️ Found {len(media_urls)} media files")

        try:
            # Run analysis (suppress verbose output)
            result = analyzer_instance.analyze_content(
                tweet_id=tweet_id,
                tweet_url=tweet_url,
                username=tweet_username,
                content=analysis_content,
                media_urls=media_urls
            )

            # Count categories
            category = result.category
            category_counts[category] = category_counts.get(category, 0) + 1

            # Create ContentAnalysis object for saving (copy all fields from result)
            analysis = ContentAnalysis(
                tweet_id=tweet_id,
                tweet_url=tweet_url,
                username=tweet_username,
                tweet_content=content,
                analysis_timestamp=datetime.now().isoformat(),
                category=result.category,
                categories_detected=getattr(result, 'categories_detected', []),
                llm_explanation=result.llm_explanation,
                analysis_method=result.analysis_method,
                media_urls=getattr(result, 'media_urls', []),
                media_analysis=getattr(result, 'media_analysis', ''),
                media_type=getattr(result, 'media_type', ''),
                multimodal_analysis=getattr(result, 'multimodal_analysis', False),
                pattern_matches=getattr(result, 'pattern_matches', []),
                topic_classification=getattr(result, 'topic_classification', {}),
                analysis_json=getattr(result, 'analysis_json', '')
            )

            # Save to database
            analyzer_instance.save_analysis(analysis)

            results.append(result)

            # Show result
            category_emoji = {
                Categories.HATE_SPEECH: '🚫',
                Categories.DISINFORMATION: '❌',
                Categories.CONSPIRACY_THEORY: '🕵️',
                Categories.FAR_RIGHT_BIAS: '⚡',
                Categories.CALL_TO_ACTION: '📢',
                Categories.NATIONALISM: '🏴',
                Categories.ANTI_GOVERNMENT: '🏛️',
                Categories.HISTORICAL_REVISIONISM: '📜',
                Categories.POLITICAL_GENERAL: '🗳️',
                Categories.GENERAL: '✅'
            }.get(category, '❓')

            print(f"    {category_emoji} {category}")

            # Show LLM explanation if available (truncated for readability)
            if result.llm_explanation and len(result.llm_explanation.strip()) > 0:
                explanation = result.llm_explanation[:120] + "..." if len(result.llm_explanation) > 120 else result.llm_explanation
                print(f"    💭 {explanation}")

            # Show analysis method
            method_emoji = "🧠" if result.analysis_method == "llm" else "🔍"
            print(f"    {method_emoji} Method: {result.analysis_method}")

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"    ❌ Error analyzing tweet: {error_msg}")

            # Log detailed error information
            logger.error(f"Analysis failed for tweet {tweet_id} (@{tweet_username}): {error_msg}")
            logger.info(f"Tweet content: {content[:200]}...")
            if media_urls:
                logger.info(f"Media URLs that may have caused failure: {media_urls}")

            # Save failed analysis to database for debugging
            analyzer_instance.repository.save_failed_analysis(
                tweet_id=tweet_id,
                tweet_url=tweet_url,
                username=tweet_username,
                content=content,
                error_message=error_msg,
                media_urls=media_urls
            )

            # Count as failed analysis
            category_counts["ERROR"] = category_counts.get("ERROR", 0) + 1

        print()

    # Summary
    print("📊 Analysis Complete!")
    print("=" * 50)
    total_processed = len(results) + category_counts.get("ERROR", 0)
    print(f"📈 Tweets processed: {total_processed}")
    print(f"✅ Successful analyses: {len(results)}")
    if category_counts.get("ERROR", 0) > 0:
        print(f"❌ Failed analyses: {category_counts['ERROR']}")
    print("📋 Category breakdown:")

    for category, count in sorted(category_counts.items()):
        emoji = {
            Categories.HATE_SPEECH: '🚫',
            Categories.DISINFORMATION: '❌',
            Categories.CONSPIRACY_THEORY: '🕵️',
            Categories.FAR_RIGHT_BIAS: '⚡',
            Categories.CALL_TO_ACTION: '📢',
            Categories.GENERAL: '✅',
            'ERROR': '💥'
        }.get(category, '❓')

        percentage = (count / total_processed) * 100 if total_processed > 0 else 0
        print(f"    {emoji} {category}: {count} ({percentage:.1f}%)")

    print()
    print("✅ Results saved to content_analyses table in database")
    return results


# Utility functions for analyzer operations
def create_analyzer(config: Optional[AnalyzerConfig] = None, verbose: bool = False) -> Analyzer:
    """Create and return an Analyzer instance with specified configuration."""
    return Analyzer(config=config, verbose=verbose)


def reanalyze_tweet(tweet_id: str, analyzer: Optional[Analyzer] = None) -> Optional[ContentAnalysis]:
    """Reanalyze a single tweet and return the result."""
    from utils.database import get_tweet_data

    # Use provided analyzer or create default one
    if analyzer is None:
        analyzer = create_analyzer()

    # Get tweet data
    tweet_data = analyzer.repository.get_tweet_data(tweet_id)
    if not tweet_data:
        return None

    # Delete existing analysis
    analyzer.repository.delete_existing_analysis(tweet_id)

    # Reanalyze
    analysis_result = analyzer.analyze_content(
        tweet_id=tweet_data['tweet_id'],
        tweet_url=f"https://twitter.com/placeholder/status/{tweet_data['tweet_id']}",
        username=tweet_data['username'],
        content=tweet_data['content'],
        media_urls=tweet_data.get('media_urls', [])
    )

    # Save result
    analyzer.save_analysis(analysis_result)

    return analysis_result


def main():
    parser = argparse.ArgumentParser(
        description="Analyze tweets from database using Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_twitter.py                                    # Analyze all unanalyzed tweets with LLM
  python analyze_twitter.py --username Santi_ABASCAL          # Analyze specific user's unanalyzed tweets with LLM
  python analyze_twitter.py --limit 5                         # Analyze 5 unanalyzed tweets with LLM
  python analyze_twitter.py --force-reanalyze --limit 10      # Reanalyze 10 tweets (including already analyzed)
  python analyze_twitter.py --username Santi_ABASCAL -f       # Reanalyze all tweets from specific user
  python analyze_twitter.py --tweet-id 1234567890123456789    # Analyze/reanalyze specific tweet by ID
        """
    )

    parser.add_argument('--username', '-u', help='Analyze tweets from specific username only')
    parser.add_argument('--limit', '-l', type=int, help='Maximum number of tweets to process')
    parser.add_argument('--force-reanalyze', '-f', action='store_true',
                       help='Reanalyze already processed tweets (useful when prompts change)')
    parser.add_argument('--tweet-id', '-t', help='Analyze/reanalyze a specific tweet by ID')

    args = parser.parse_args()

    analyze_tweets_from_db(
        username=args.username,
        max_tweets=args.limit,
        force_reanalyze=args.force_reanalyze,
        tweet_id=args.tweet_id
    )


if __name__ == '__main__':
    main()