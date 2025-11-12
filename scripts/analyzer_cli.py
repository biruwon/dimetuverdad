"""
Analyzer CLI: Command-line interface for the dimetuverdad analyzer.

This module provides the CLI interface for tweet analysis, separating
the command-line logic from the core Analyzer class.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import warnings
import logging
import argparse
import asyncio
import traceback
from typing import Optional
from datetime import datetime
from analyzer.analyze_twitter import create_analyzer, reanalyze_tweet
from analyzer.config import AnalyzerConfig
from analyzer.categories import Categories
from analyzer.models import ContentAnalysis
from analyzer.constants import ConfigDefaults
from utils.performance import start_tracking, stop_tracking, print_performance_summary

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging for multimodal analysis debugging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def _handle_single_tweet_analysis(tweet_id: str, verbose: bool = False):
    """Handle analysis of a single specific tweet."""
    print(f"🎯 Reanalyzing specific tweet: {tweet_id}")
    print("🚀 Initializing Analyzer...")
    try:
        config = AnalyzerConfig(verbose=verbose)
        analyzer_instance = create_analyzer(config=config, verbose=verbose)
        print("✅ Analyzer ready!")

        result = await reanalyze_tweet(tweet_id, analyzer=analyzer_instance)
        if result:
            print(f"\n📝 Tweet: {tweet_id}")
            print(f"    🏷️ Category: {result.category}")

            # Display best available explanation
            best_explanation = result.external_explanation if result.external_explanation else result.local_explanation
            if best_explanation:
                print(f"    💭 {best_explanation[:100]}...")

            # Show analysis stages
            stages_display = result.analysis_stages or "pattern"
            if result.external_analysis_used:
                print(f"    🌐 Stages: {stages_display} (with external)")
            else:
                print(f"    🔍 Stages: {stages_display}")

            if result.multimodal_analysis:
                print(f"    🎥 Multimodal analysis: Yes ({result.media_type})")
            print("\n✅ Analysis complete and saved to database")
        else:
            print(f"❌ Tweet {tweet_id} not found in database")
    except Exception as e:
        print(f"❌ Error reanalyzing tweet: {e}")
        traceback.print_exc()


async def _setup_analyzer_and_get_tweets(usernames, max_tweets, force_reanalyze, verbose, fast_mode=False):
    """Initialize analyzer and retrieve tweets for analysis."""
    print("🚀 Initializing Analyzer...")
    try:
        config = AnalyzerConfig(verbose=verbose)
        analyzer_instance = create_analyzer(config=config, verbose=verbose, fast_mode=fast_mode)  # Always use LLM for better explanations
        print("✅ Analyzer ready!")
    except Exception as e:
        print(f"❌ Error initializing analyzer: {e}")
        logger.error(f"Error initializing analyzer: {e}")
        return None, [], 0, 0

    # Get tweets for analysis
    tweets = analyzer_instance.repository.get_tweets_for_analysis(
        usernames=usernames,
        max_tweets=max_tweets,
        force_reanalyze=force_reanalyze
    )

    # Get total pending count (all unanalyzed tweets for selected users)
    total_pending = 0
    if usernames:
        for username in usernames:
            # Get count of unanalyzed tweets for this user
            user_tweets = analyzer_instance.repository.get_tweets_for_analysis(
                usernames=[username],
                max_tweets=None,  # Get all
                force_reanalyze=False  # Only unanalyzed
            )
            total_pending += len(user_tweets)
    else:
        # Get all unanalyzed tweets across all users
        all_tweets = analyzer_instance.repository.get_tweets_for_analysis(
            usernames=None,
            max_tweets=None,
            force_reanalyze=False
        )
        total_pending = len(all_tweets)

    # Also get count of already analyzed tweets for reporting
    analyzed_count = 0
    if usernames:
        for username in usernames:
            analyzed_count += analyzer_instance.repository.get_analysis_count_by_author(username)
    else:
        analyzed_count = analyzer_instance.repository.get_analysis_count()

    return analyzer_instance, tweets, analyzed_count, total_pending


def _print_no_tweets_message(usernames, force_reanalyze, analyzed_count):
    """Print message when no tweets are found for analysis."""
    search_desc = "tweets" if force_reanalyze else "unanalyzed tweets"
    if usernames:
        user_list = ", ".join(f"@{u}" for u in usernames)
        print(f"✅ No {search_desc} found for {user_list}")
    else:
        print(f"✅ No {search_desc} found")
    if analyzed_count > 0 and not force_reanalyze:
        print(f"📊 Already analyzed: {analyzed_count} tweets")


def _print_analysis_start_info(tweets, usernames, force_reanalyze, analyzed_count, total_pending):
    """Print information about the analysis that is about to start."""
    tweet_type = "tweets" if force_reanalyze else "unanalyzed tweets"
    print(f"📊 Found {len(tweets)} {tweet_type} to process")
    if analyzed_count > 0 and not force_reanalyze:
        print(f"📊 Already analyzed: {analyzed_count} tweets")
        print(f"📊 Total pending: {total_pending} tweets")
    elif force_reanalyze:
        print(f"📊 Will reanalyze ALL selected tweets")

    print(f"🔧 Analysis Mode: LLM + Patterns")
    print()


def _setup_concurrency_config(analyzer_instance):
    """Setup concurrency configuration and return semaphores and retry settings."""

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

    return analysis_sema, llm_sema, max_retries, retry_delay


async def _execute_analysis_tasks(tweets, analyzer_instance, analysis_sema, llm_sema, max_retries, retry_delay, tracker, verbose=False, total_pending=0):
    """Execute concurrent analysis tasks and return results."""

    results = []
    category_counts = {}
    processed_count = 0

    async def analyze_one(idx: int, entry):
        nonlocal processed_count
        (tw_id, tw_url, tw_user, content, media_links, original_content) = entry
        
        # Update progress counter
        processed_count += 1
        
        # Show progress with tweet ID and overall progress
        progress_info = f"[{processed_count:3d}/{len(tweets):3d}]"
        if total_pending > 0 and total_pending != len(tweets):
            progress_info += f" ({total_pending - processed_count + 1:3d} remaining)"
        
        ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        print(f"{ts} 📝 {progress_info} Analyzing: {tw_id}")
        
        logger.info(f"Starting analysis of tweet {tw_id} by @{tw_user}")

        # Combine content with quoted content if available
        analysis_content = content
        if original_content and original_content.strip():
            analysis_content = f"{content}\n\n[Contenido citado]: {original_content}"
            if verbose:
                print("    📎 Including quoted tweet content")
                logger.info(f"Tweet {tw_id}: Including quoted content")

        # Parse media URLs
        media_urls = []
        if media_links:
            media_urls = [u.strip() for u in media_links.split(',') if u.strip()]
        if media_urls and verbose:
            print(f"    🖼️ Found {len(media_urls)} media files")
            logger.info(f"Tweet {tw_id}: Found {len(media_urls)} media files: {media_urls}")

        try:
            async with analysis_sema:
                async with llm_sema:
                    # Basic retry honoring config
                    attempts = 0
                    last_exc = None
                    while attempts <= max_retries:
                        try:
                            logger.info(f"Tweet {tw_id}: Starting analysis attempt {attempts + 1}")
                            # Add overall timeout to prevent infinite retries from multiplying timeouts
                            # Max single attempt: category(60s) + media(100s) + explanation(60s) + verification(100s) = 320s
                            result = await asyncio.wait_for(
                                analyzer_instance.analyze_content(
                                    tweet_id=tw_id,
                                    tweet_url=tw_url,
                                    username=tw_user,
                                    content=analysis_content,
                                    media_urls=media_urls
                                ),
                                timeout=ConfigDefaults.ANALYSIS_TIMEOUT
                            )
                            logger.info(f"Tweet {tw_id}: Analysis completed successfully - category: {result.category}")
                            break
                        except asyncio.TimeoutError:
                            last_exc = asyncio.TimeoutError(f"Analysis timed out after {ConfigDefaults.ANALYSIS_TIMEOUT}s")
                            attempts += 1
                            logger.warning(f"Tweet {tw_id}: Analysis attempt {attempts} timed out after {ConfigDefaults.ANALYSIS_TIMEOUT}s")
                            if attempts > max_retries:
                                logger.error(f"Tweet {tw_id}: All {max_retries + 1} analysis attempts timed out")
                                raise last_exc
                            await asyncio.sleep(retry_delay)
                        except Exception as inner_e:
                            last_exc = inner_e
                            attempts += 1
                            logger.warning(f"Tweet {tw_id}: Analysis attempt {attempts} failed: {str(inner_e)}")
                            if attempts > max_retries:
                                logger.error(f"Tweet {tw_id}: All {max_retries + 1} analysis attempts failed")
                                raise inner_e
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
                local_explanation=result.local_explanation,
                external_explanation=result.external_explanation,
                analysis_stages=result.analysis_stages,
                external_analysis_used=result.external_analysis_used,
                media_urls=getattr(result, 'media_urls', []),
                media_type=getattr(result, 'media_type', ''),
                pattern_matches=getattr(result, 'pattern_matches', []),
                topic_classification=getattr(result, 'topic_classification', {}),
                analysis_json=getattr(result, 'analysis_json', '')
            )

            analyzer_instance.save_analysis(analysis)
            logger.info(f"Tweet {tw_id}: Analysis saved to database - category: {category}")
            if verbose:
                ts_done = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                print(f"{ts_done} 💾 Saved analysis for {tw_id}")

            # Increment performance counter
            tracker.increment_operations(1)

            # Show result (only in verbose mode)
            if verbose:
                category_emoji = {
                    Categories.HATE_SPEECH: '🚫',
                    Categories.DISINFORMATION: '❌',
                    Categories.CONSPIRACY_THEORY: '🕵️',
                    Categories.ANTI_IMMIGRATION: '🌍',
                    Categories.ANTI_LGBTQ: '🏳️‍🌈',
                    Categories.ANTI_FEMINISM: '👩',
                    Categories.CALL_TO_ACTION: '📢',
                    Categories.GENERAL: '✅'
                }.get(category, '❓')
                print(f"    {category_emoji} {category}")

                # Display best available explanation
                best_explanation = result.external_explanation if result.external_explanation else result.local_explanation
                if best_explanation and isinstance(best_explanation, str) and len(best_explanation.strip()) > 0:
                    explanation = best_explanation[:120] + "..." if len(best_explanation) > 120 else best_explanation
                    print(f"    💭 {explanation}")

                # Show analysis stages
                stages_display = result.analysis_stages or "pattern"
                if result.external_analysis_used:
                    print(f"    🌐 Stages: {stages_display} (with external)")
                else:
                    print(f"    🔍 Stages: {stages_display}")
                
                # Show stage timings if available
                if hasattr(result, 'verification_data') and result.verification_data:
                    stage_timings = result.verification_data.get('stage_timings', {})
                    if stage_timings:
                        total_time = sum(stage_timings.values())
                        print(f"    ⏱️  Timings: Total={total_time:.1f}s", end="")
                        if 'pattern_detection' in stage_timings:
                            print(f" | Pattern={stage_timings['pattern_detection']:.2f}s", end="")
                        if 'category_detection' in stage_timings:
                            print(f" | Category={stage_timings['category_detection']:.1f}s", end="")
                        if 'media_analysis' in stage_timings:
                            print(f" | Media={stage_timings['media_analysis']:.1f}s", end="")
                        if 'explanation' in stage_timings:
                            print(f" | Explanation={stage_timings['explanation']:.1f}s", end="")
                        print()
            else:
                # Non-verbose mode: show category and brief content/explanation
                category_emoji = {
                    Categories.HATE_SPEECH: '🚫',
                    Categories.DISINFORMATION: '❌',
                    Categories.CONSPIRACY_THEORY: '🕵️',
                    Categories.ANTI_IMMIGRATION: '🌍',
                    Categories.ANTI_LGBTQ: '🏳️‍🌈',
                    Categories.ANTI_FEMINISM: '👩',
                    Categories.CALL_TO_ACTION: '📢',
                    Categories.GENERAL: '✅'
                }.get(category, '❓')

                # Show content preview (first 60 chars)
                content_preview = content[:60] + "..." if len(content) > 60 else content
                print(f"    📝 {content_preview}")

                # Show category
                print(f"    {category_emoji} {category}")

                # Show explanation preview (first 80 chars)
                best_explanation = result.external_explanation if result.external_explanation else result.local_explanation
                if best_explanation and isinstance(best_explanation, str) and len(best_explanation.strip()) > 0:
                    explanation_preview = best_explanation[:80] + "..." if len(best_explanation) > 80 else best_explanation
                    print(f"    💭 {explanation_preview}")
                
                # Show timing summary
                if hasattr(result, 'verification_data') and result.verification_data:
                    stage_timings = result.verification_data.get('stage_timings', {})
                    if stage_timings:
                        total_time = sum(stage_timings.values())
                        print(f"    ⏱️  {total_time:.1f}s", end="")
                        if 'media_analysis' in stage_timings:
                            print(f" (media: {stage_timings['media_analysis']:.1f}s)", end="")
                        print()

            return ("ok", tw_id, category, stage_timings if hasattr(result, 'verification_data') and result.verification_data and result.verification_data.get('stage_timings') else {})

        except Exception as e:
            # Log the error
            logger.error(f"Tweet {tw_id}: Analysis failed - {str(e)}")
            # Re-raise all errors to stop the entire analysis pipeline
            raise RuntimeError(f"Analysis failed for tweet {tw_id}: {str(e)}") from e
        finally:
            print()

    # Kick off tasks and wait for all to complete
    tasks = [analyze_one(i, entry) for i, entry in enumerate(tweets, 1)]
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    # Check for exceptions and re-raise the first one encountered
    for out in outcomes:
        if isinstance(out, Exception):
            logger.error(f"Analysis run failed: {str(out)}")
            raise out

    # Collect successful results for reporting compatibility
    results = []
    all_timings = []
    for out in outcomes:
        if isinstance(out, tuple) and out[0] == "ok":
            results.append(out)
            if len(out) > 3:  # Has timing data
                all_timings.append(out[3])

    logger.info(f"Analysis run completed: {len(results)} successful analyses out of {len(tweets)} tweets")
    return results, category_counts, all_timings


def _print_analysis_summary(results, category_counts, tracker, all_timings=None):
    """Print the final analysis summary and statistics."""
    print("📊 Analysis Complete!")
    print("=" * 50)
    total_processed = len(results) + category_counts.get("ERROR", 0)
    print(f"📈 Tweets processed: {total_processed}")
    print(f"✅ Successful analyses: {len(results)}")
    if category_counts.get("ERROR", 0) > 0:
        print(f"❌ Failed analyses: {category_counts['ERROR']}")
    
    # Log summary
    logger.info(f"Analysis summary: {total_processed} processed, {len(results)} successful, {category_counts.get('ERROR', 0)} failed")
    
    # Print timing analysis
    if all_timings:
        print("\n⏱️  Performance Analysis:")
        print("=" * 50)
        
        # Calculate aggregates
        stage_names = ['pattern_detection', 'category_detection', 'media_analysis', 'explanation', 'verification']
        stage_aggregates = {stage: [] for stage in stage_names}
        total_times = []
        
        for timing_dict in all_timings:
            if timing_dict:
                total_time = sum(timing_dict.values())
                total_times.append(total_time)
                for stage in stage_names:
                    if stage in timing_dict:
                        stage_aggregates[stage].append(timing_dict[stage])
        
        if total_times:
            avg_total = sum(total_times) / len(total_times)
            max_total = max(total_times)
            min_total = min(total_times)
            
            print(f"📊 Average time per post: {avg_total:.1f}s")
            print(f"   ⬆️  Slowest: {max_total:.1f}s")
            print(f"   ⬇️  Fastest: {min_total:.1f}s")
            
            print("\n📋 Stage Breakdown (averages):")
            for stage in stage_names:
                if stage_aggregates[stage]:
                    avg = sum(stage_aggregates[stage]) / len(stage_aggregates[stage])
                    count = len(stage_aggregates[stage])
                    percentage = (avg / avg_total * 100) if avg_total > 0 else 0
                    stage_display = stage.replace('_', ' ').title()
                    print(f"   • {stage_display}: {avg:.1f}s ({percentage:.1f}%) - {count}/{len(total_times)} posts")
    
    print("\n📋 Category breakdown:")
    print("=" * 50)

    for category, count in sorted(category_counts.items()):
        emoji = {
            Categories.HATE_SPEECH: '🚫',
            Categories.DISINFORMATION: '❌',
            Categories.CONSPIRACY_THEORY: '🕵️',
            Categories.ANTI_IMMIGRATION: '🌍',
            Categories.ANTI_LGBTQ: '🏳️‍🌈',
            Categories.ANTI_FEMINISM: '👩',
            Categories.CALL_TO_ACTION: '📢',
            Categories.GENERAL: '✅',
            'ERROR': '💥'
        }.get(category, '❓')

        percentage = (count / total_processed) * 100 if total_processed > 0 else 0
        print(f"    {emoji} {category}: {count} ({percentage:.1f}%)")

    print()
    print("✅ Results saved to content_analyses table in database")

    # Print performance summary
    metrics = stop_tracking(tracker)
    print_performance_summary(metrics)


async def analyze_tweets_cli(usernames=None, max_tweets=None, force_reanalyze=False, tweet_id=None, verbose=False, fast_mode=False):
    """
    CLI wrapper for tweet analysis functionality.

    Args:
        username: List of usernames to analyze (None for all)
        max_tweets: Maximum number of tweets to analyze (None for all)
        force_reanalyze: If True, reanalyze already processed tweets
        tweet_id: Specific tweet ID to analyze/reanalyze
        verbose: Enable verbose output
    """
    # Setup logging for this analysis run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"logs/analysis_run_{timestamp}.log"
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Configure file logging
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Add file handler to logger
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    logger.info(f"Starting analysis run - usernames: {usernames}, max_tweets: {max_tweets}, force_reanalyze: {force_reanalyze}, tweet_id: {tweet_id}, verbose: {verbose}, fast_mode: {fast_mode}")
    
    print(f"📝 Analysis logs will be saved to: {log_filename}")
    
    # Start performance tracking
    tracker = start_tracking("Tweet Analyzer")

    print("🔍 Enhanced Tweet Analysis Pipeline")
    print("=" * 50)

    # Special case: specific tweet ID requested
    if tweet_id:
        await _handle_single_tweet_analysis(tweet_id, verbose)
        return

    # Initialize analyzer and get tweets for bulk processing
    analyzer_instance, tweets, analyzed_count, total_pending = await _setup_analyzer_and_get_tweets(
        usernames, max_tweets, force_reanalyze, verbose, fast_mode
    )

    if not tweets:
        _print_no_tweets_message(usernames, force_reanalyze, analyzed_count)
        return

    _print_analysis_start_info(tweets, usernames, force_reanalyze, analyzed_count, total_pending)

    # Setup concurrency configuration
    analysis_sema, llm_sema, max_retries, retry_delay = _setup_concurrency_config(analyzer_instance)

    # Execute analysis tasks
    results, category_counts, all_timings = await _execute_analysis_tasks(
        tweets, analyzer_instance, analysis_sema, llm_sema, max_retries, retry_delay, tracker, verbose, total_pending
    )

    # Print final summary
    _print_analysis_summary(results, category_counts, tracker, all_timings)
    
    logger.info("Analysis run finished successfully")


def create_analyzer_cli(config: Optional[AnalyzerConfig] = None, verbose: bool = False, fast_mode: bool = False):
    """
    CLI wrapper for analyzer creation.

    Args:
        config: Analyzer configuration
        verbose: Enable verbose output
        fast_mode: Use simplified prompts for faster bulk processing

    Returns:
        Analyzer instance
    """
    return create_analyzer(config=config, verbose=verbose, fast_mode=fast_mode)


async def reanalyze_tweet_cli(tweet_id: str, analyzer=None):
    """
    CLI wrapper for tweet reanalysis.

    Args:
        tweet_id: Tweet ID to reanalyze
        analyzer: Optional analyzer instance

    Returns:
        Analysis result
    """
    return await reanalyze_tweet(tweet_id, analyzer)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze tweets from database using Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m analyzer.cli                                    # Analyze all unanalyzed tweets with LLM
  python -m analyzer.cli --username Santi_ABASCAL          # Analyze specific user's unanalyzed tweets with LLM
  python -m analyzer.cli --username "user1,user2,user3"    # Analyze multiple users' unanalyzed tweets with LLM
  python -m analyzer.cli --limit 5                         # Analyze 5 unanalyzed tweets with LLM
  python -m analyzer.cli --force-reanalyze --limit 10      # Reanalyze 10 tweets (including already analyzed)
  python -m analyzer.cli --username Santi_ABASCAL -f       # Reanalyze all tweets from specific user
  python -m analyzer.cli --tweet-id 1234567890123456789    # Analyze/reanalyze specific tweet by ID
        """
    )

    parser.add_argument('--username', '-u', help='Analyze tweets from specific username(s) only (comma-separated for multiple users)')
    parser.add_argument('--limit', '-l', type=int, help='Maximum number of tweets to process')
    parser.add_argument('--force-reanalyze', '-f', action='store_true',
                       help='Reanalyze already processed tweets (useful when prompts change)')
    parser.add_argument('--tweet-id', '-t', help='Analyze/reanalyze a specific tweet by ID')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed analysis output for each tweet')
    parser.add_argument('--fast-mode', action='store_true',
                       help='Use simplified prompts for faster bulk processing')

    args = parser.parse_args()

    # Split username by comma if provided
    usernames = None
    if args.username:
        usernames = [u.strip() for u in args.username.split(',') if u.strip()]

    # Run async analysis
    asyncio.run(analyze_tweets_cli(
        usernames=usernames,
        max_tweets=args.limit,
        force_reanalyze=args.force_reanalyze,
        tweet_id=args.tweet_id,
        verbose=args.verbose,
        fast_mode=args.fast_mode
    ))


if __name__ == '__main__':
    main()