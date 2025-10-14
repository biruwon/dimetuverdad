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
import asyncio
import traceback
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
# Import repository interfaces
from repositories import get_tweet_repository

# Import new modular components
from .config import AnalyzerConfig
from .metrics import MetricsCollector
from .repository import ContentAnalysisRepository
from .text_analyzer import TextAnalyzer
from .multimodal_analyzer import MultimodalAnalyzer
from .models import ContentAnalysis
from .constants import AnalysisMethods, ErrorMessages, ConfigDefaults

# Import retrieval integration
from retrieval.integration.analyzer_hooks import AnalyzerHooks, create_analyzer_hooks

# Import database utilities
from utils.database import get_db_connection_context, get_tweet_data

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging for multimodal analysis debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        self.repository = ContentAnalysisRepository()

        # Initialize analysis components
        self.text_analyzer = TextAnalyzer(config=self.config, verbose=verbose)
        self.multimodal_analyzer = MultimodalAnalyzer(verbose=verbose)

        # Initialize retrieval integration
        self.retrieval_hooks = create_analyzer_hooks()

        if self.verbose:
            print("🚀 Iniciando Analyzer con componentes modulares...")
            print("Componentes cargados:")
            print("- ✓ Analizador de texto especializado")
            print("- ✓ Analizador multimodal para medios")
            print("- ✓ Recolector de métricas de rendimiento")
            print("- ✓ Repositorio de análisis de contenido")
            print("- ✓ Integración de recuperación de evidencia")
            print(f"- ✓ Configuración: {self.config.model_priority} priority")

    async def analyze_content(self,
                       tweet_id: str,
                       tweet_url: str,
                       username: str,
                       content: str,
                       media_urls: List[str] = None) -> ContentAnalysis:
        """
        Main content analysis pipeline with conditional multimodal analysis and evidence retrieval.

        Routes to appropriate analyzer based on content type, then conditionally triggers
        evidence retrieval for verification.
        """
        analysis_start_time = time.time()

        try:
            # Route to appropriate analyzer (execute in thread to avoid blocking event loop)
            if media_urls and len(media_urls) > 0:
                result = await asyncio.to_thread(
                    self.multimodal_analyzer.analyze_with_media,
                    tweet_id, tweet_url, username, content, media_urls
                )
            else:
                result = await asyncio.to_thread(
                    self.text_analyzer.analyze,
                    tweet_id, tweet_url, username, content
                )

            # Check if evidence retrieval should be triggered
            if self._should_trigger_evidence_retrieval(result, content):
                if self.verbose:
                    print("🔍 Triggering evidence retrieval for verification...")

                # Perform evidence retrieval and enhance explanation
                enhanced_result = await self._enhance_with_evidence_retrieval(result, content)

                # Update result with enhanced explanation and verification data
                result.llm_explanation = enhanced_result.llm_explanation
                result.verification_data = enhanced_result.verification_data
                result.verification_confidence = enhanced_result.verification_confidence

                if self.verbose:
                    print("✅ Evidence retrieval completed and explanation enhanced")

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
                traceback.print_exc()

            # Return error result
    def _should_trigger_evidence_retrieval(self, analysis_result: ContentAnalysis, content: str) -> bool:
        """
        Determine if evidence retrieval should be triggered based on analysis results and content.

        Triggers verification for text-only analyses with:
        1. High-confidence disinformation detection
        2. Conspiracy theory content
        3. Content with numerical/statistical claims
        4. Far-right bias content with potential factual claims

        Note: Multimodal analyses are excluded as they already include media-based verification.
        """
        # Skip verification for multimodal analyses (they already include media verification)
        if analysis_result.analysis_method == AnalysisMethods.MULTIMODAL.value:
            return False

        # Convert ContentAnalysis to dict format expected by analyzer hooks
        analyzer_result_dict = {
            'category': analysis_result.category,
            'confidence': getattr(analysis_result, 'confidence', 0.5),  # Default confidence
            'explanation': analysis_result.llm_explanation,
            'analysis_method': analysis_result.analysis_method
        }

        # Use the analyzer hooks to determine if verification should be triggered
        should_trigger, reason = self.retrieval_hooks.should_trigger_verification(
            content, analyzer_result_dict
        )

        if self.verbose and should_trigger:
            print(f"🔍 Evidence retrieval triggered: {reason}")

        return should_trigger

    async def _enhance_with_evidence_retrieval(self, analysis_result: ContentAnalysis, content: str):
        """
        Enhance analysis result with evidence retrieval and verification.

        Args:
            analysis_result: Original analysis result
            content: Original content text

        Returns:
            Enhanced analysis result with verification data
        """
        try:
            # Convert ContentAnalysis to dict format for analyzer hooks
            original_result_dict = {
                'category': analysis_result.category,
                'confidence': getattr(analysis_result, 'confidence', 0.5),
                'explanation': analysis_result.llm_explanation,
                'analysis_method': analysis_result.analysis_method
            }

            # Perform analysis with verification
            enhanced_result = await self.retrieval_hooks.analyze_with_verification(
                content, original_result_dict
            )

            # Convert verification data to dict for JSON serialization
            verification_data_dict = None
            if enhanced_result.verification_data:
                verification_data_dict = dict(enhanced_result.verification_data)  # Convert to regular dict
                
                # Handle nested VerificationReport object
                if 'verification_report' in verification_data_dict and hasattr(verification_data_dict['verification_report'], 'overall_verdict'):
                    report = verification_data_dict['verification_report']
                    
                    # Safely serialize claims_verified
                    claims_verified = []
                    for claim in getattr(report, 'claims_verified', []):
                        if hasattr(claim, '__dict__'):
                            claim_dict = dict(claim.__dict__)
                            # Convert enum values to strings
                            if 'verdict' in claim_dict and hasattr(claim_dict['verdict'], 'value'):
                                claim_dict['verdict'] = claim_dict['verdict'].value
                            elif 'verdict' in claim_dict:
                                claim_dict['verdict'] = str(claim_dict['verdict'])
                            claims_verified.append(claim_dict)
                        else:
                            claims_verified.append(str(claim))
                    
                    verification_data_dict['verification_report'] = {
                        'overall_verdict': report.overall_verdict.value if hasattr(report.overall_verdict, 'value') else str(report.overall_verdict),
                        'confidence_score': report.confidence_score,
                        'claims_verified': claims_verified,
                        'evidence_sources': [{'source_name': s.source_name, 'source_type': getattr(s, 'source_type', 'unknown'), 'reliability_score': getattr(s, 'reliability_score', 0.5)} for s in getattr(report, 'evidence_sources', [])],
                        'temporal_consistency': getattr(report, 'temporal_consistency', True),
                        'contradictions_found': getattr(report, 'contradictions_found', []),
                        'processing_time': getattr(report, 'processing_time', 0.0),
                        'verification_method': getattr(report, 'verification_method', 'unknown')
                    }

            # Create enhanced ContentAnalysis with verification data
            enhanced_analysis = ContentAnalysis(
                post_id=analysis_result.post_id,
                post_url=analysis_result.post_url,
                author_username=analysis_result.author_username,
                post_content=analysis_result.post_content,
                analysis_timestamp=analysis_result.analysis_timestamp,
                category=analysis_result.category,
                categories_detected=analysis_result.categories_detected,
                llm_explanation=enhanced_result.explanation_with_verification,
                analysis_method=analysis_result.analysis_method,
                media_urls=analysis_result.media_urls,
                media_analysis=analysis_result.media_analysis,
                media_type=analysis_result.media_type,
                multimodal_analysis=analysis_result.multimodal_analysis,
                pattern_matches=analysis_result.pattern_matches,
                topic_classification=analysis_result.topic_classification,
                analysis_json=analysis_result.analysis_json,
                analysis_time_seconds=analysis_result.analysis_time_seconds,
                model_used=analysis_result.model_used,
                tokens_used=analysis_result.tokens_used,
                verification_data=verification_data_dict,
                verification_confidence=verification_data_dict.get('verification_confidence', 0.0) if verification_data_dict else 0.0
            )

            return enhanced_analysis

        except Exception as e:
            if self.verbose:
                print(f"⚠️ Evidence retrieval failed: {e}")
            # Return original result if verification fails
            return analysis_result

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
            return self.repository.get_by_post_id(tweet_id)
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
            with get_db_connection_context() as conn:
                c = conn.cursor()
                c.execute("SELECT COUNT(*) FROM content_analyses")
                count = c.fetchone()[0]
            print(f"📊 Database: {count} analyses stored")
        except Exception as e:
            print(f"❌ Database: Error - {e}")


async def analyze_tweets_from_db(username=None, max_tweets=None, force_reanalyze=False, tweet_id=None):
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

            result = await reanalyze_tweet(tweet_id, analyzer=analyzer_instance)
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
    analyzed_count = analyzer_instance.repository.get_analysis_count_by_author(username)

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

    # Analyze each tweet with bounded concurrency
    results = []
    category_counts = {}

    # Tunables from configuration, with safe fallbacks for mocked analyzers
    cfg = getattr(analyzer_instance, 'config', None)
    raw_max_conc = getattr(cfg, 'max_concurrency', None) if cfg is not None else None
    raw_max_llm_conc = getattr(cfg, 'max_llm_concurrency', None) if cfg is not None else None

    def _coerce_positive_int(val, default):
        try:
            iv = int(val)
            return iv if iv > 0 else default
        except Exception:
            return default

    max_concurrency = _coerce_positive_int(raw_max_conc, ConfigDefaults.MAX_CONCURRENCY)
    max_llm_concurrency = _coerce_positive_int(raw_max_llm_conc, ConfigDefaults.MAX_LLM_CONCURRENCY)

    analysis_sema = asyncio.Semaphore(max_concurrency)
    llm_sema = asyncio.Semaphore(max_llm_concurrency)

    # Retry/backoff tunables with safe fallbacks
    raw_max_retries = getattr(cfg, 'max_retries', None) if cfg is not None else None
    raw_retry_delay = getattr(cfg, 'retry_delay', None) if cfg is not None else None
    max_retries = _coerce_positive_int(raw_max_retries, ConfigDefaults.MAX_RETRIES)
    # allow 0 for retry_delay, so special-case non-negative int coercion
    try:
        retry_delay = int(raw_retry_delay)
        if retry_delay < 0:
            retry_delay = ConfigDefaults.RETRY_DELAY
    except Exception:
        retry_delay = ConfigDefaults.RETRY_DELAY

    async def analyze_one(idx: int, entry):
        (tw_id, tw_url, tw_user, content, media_links, original_content) = entry
        ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        print(f"{ts} 📝 [{idx:2d}/{len(tweets)}] Analyzing: {tw_id}")

        # Combine content with quoted content if available
        analysis_content = content
        if original_content and original_content.strip():
            analysis_content = f"{content}\n\n[Contenido citado]: {original_content}"
            print("    📎 Including quoted tweet content")

        # Parse media URLs
        media_urls = []
        if media_links:
            media_urls = [u.strip() for u in media_links.split(',') if u.strip()]
        if media_urls:
            print(f"    🖼️ Found {len(media_urls)} media files")

        try:
            async with analysis_sema:
                # Guard heavy I/O phases (LLM, multimodal, retrieval)
                async with llm_sema:
                    # Basic retry honoring config
                    attempts = 0
                    last_exc = None
                    while attempts <= max_retries:
                        try:
                            result = await analyzer_instance.analyze_content(
                                tweet_id=tw_id,
                                tweet_url=tw_url,
                                username=tw_user,
                                content=analysis_content,
                                media_urls=media_urls
                            )
                            break
                        except Exception as inner_e:
                            last_exc = inner_e
                            attempts += 1
                            if attempts > max_retries:
                                raise inner_e
                            # brief backoff
                            await asyncio.sleep(retry_delay)

            # Count categories
            category = result.category
            category_counts[category] = category_counts.get(category, 0) + 1

            # Create ContentAnalysis object for saving
            analysis = ContentAnalysis(
                post_id=tw_id,
                post_url=tw_url,
                author_username=tw_user,
                post_content=content,
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

            analyzer_instance.save_analysis(analysis)
            # Mark save completion with timestamp for concurrency visibility
            ts_done = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            print(f"{ts_done} 💾 Saved analysis for {tw_id}")

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

            if result.llm_explanation and len(result.llm_explanation.strip()) > 0:
                explanation = result.llm_explanation[:120] + "..." if len(result.llm_explanation) > 120 else result.llm_explanation
                print(f"    💭 {explanation}")
            method_emoji = "🧠" if result.analysis_method == "llm" else "🔍"
            print(f"    {method_emoji} Method: {result.analysis_method}")

            return ("ok", tw_id, category)

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"    ❌ Error analyzing tweet: {error_msg}")
            logger.error(f"Analysis failed for tweet {tw_id} (@{tw_user}): {error_msg}")
            logger.info(f"Tweet content: {content[:200]}...")
            if media_urls:
                logger.info(f"Media URLs that may have caused failure: {media_urls}")
            analyzer_instance.repository.save_failed_analysis(
                tweet_id=tw_id,
                tweet_url=tw_url,
                username=tw_user,
                content=content,
                error_message=error_msg,
                media_urls=media_urls
            )
            category_counts["ERROR"] = category_counts.get("ERROR", 0) + 1
            return ("err", tw_id, error_msg)
        finally:
            print()

    # Kick off tasks and wait for all to complete
    tasks = [analyze_one(i, entry) for i, entry in enumerate(tweets, 1)]
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect successful results for reporting compatibility
    results = []
    for out in outcomes:
        if isinstance(out, tuple) and out[0] == "ok":
            # For now we don't reconstruct full ContentAnalysis; we maintain counts and logs
            results.append(out)

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


async def reanalyze_tweet(tweet_id: str, analyzer: Optional[Analyzer] = None) -> Optional[ContentAnalysis]:
    """Reanalyze a single tweet and return the result."""

    # Use provided analyzer or create default one
    if analyzer is None:
        analyzer = create_analyzer()

    # Get tweet data
    tweet_data = analyzer.repository.get_tweet_data(tweet_id)
    if not tweet_data:
        return None

    # Delete existing analysis
    analyzer.repository.delete_existing_analysis(tweet_id)

    # Parse media URLs from media_links string
    media_urls = []
    if tweet_data.get('media_links'):
        media_urls = [url.strip() for url in tweet_data['media_links'].split(',') if url.strip()]
    
    # Debug print
    if media_urls:
        print(f"    🖼️ Found {len(media_urls)} media files for analysis")
        for i, url in enumerate(media_urls):
            print(f"      {i+1}. {url[:50]}...")

    # Reanalyze
    analysis_result = await analyzer.analyze_content(
        tweet_id=tweet_data['tweet_id'],
        tweet_url=f"https://twitter.com/placeholder/status/{tweet_data['tweet_id']}",
        username=tweet_data['username'],
        content=tweet_data['content'],
        media_urls=media_urls
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

    # Run async analysis
    asyncio.run(analyze_tweets_from_db(
        username=args.username,
        max_tweets=args.limit,
        force_reanalyze=args.force_reanalyze,
        tweet_id=args.tweet_id
    ))


if __name__ == '__main__':
    main()