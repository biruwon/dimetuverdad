#!/usr/bin/env python3
"""
Analyze Database Tweets - Analyzer Pipeline
Analyzes all tweets from the database using the analyzer system.
"""

import sqlite3
import argparse
import logging
from datetime import datetime

# Import utility modules
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from analyzer.analyzer import Analyzer, ContentAnalysis, create_analyzer, reanalyze_tweet
from analyzer.config import AnalyzerConfig
from analyzer.categories import Categories
from utils import database, paths

# Configure logging for error tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = paths.get_db_path()

def save_failed_analysis(analyzer: Analyzer, tweet_id: str, tweet_url: str, username: str, content: str, 
                        error_message: str, media_urls: list = None):
    """
    Save failed analysis attempt to database for debugging.
    
    Args:
        tweet_id: ID of the tweet that failed analysis
        tweet_url: URL of the tweet
        username: Username of the tweet author
        content: Tweet content
        error_message: Error message from the failed analysis
        media_urls: List of media URLs if any
    """
    try:
        # Create a ContentAnalysis object for failed analysis
        failed_analysis = ContentAnalysis(
            tweet_id=tweet_id,
            tweet_url=tweet_url,
            username=username,
            tweet_content=content,
            analysis_timestamp=datetime.now().isoformat(),
            category="ERROR",
            categories_detected=["ERROR"],
            llm_explanation=f"Analysis failed: {error_message}",
            analysis_method="error",
            media_urls=media_urls or [],
            media_analysis="",
            media_type="",
            multimodal_analysis=bool(media_urls),
            pattern_matches=[],
            topic_classification={},
            analysis_json=f'{{"error": "{error_message[:500]}", "media_urls": {len(media_urls or [])}}}'
        )
        
        # Save to database using analyzer instance
        analyzer.save_analysis(failed_analysis)
        logger.info(f"Saved failed analysis for tweet {tweet_id}: {error_message[:100]}")
        
    except Exception as save_error:
        logger.error(f"Failed to save error analysis for tweet {tweet_id}: {save_error}")

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
    
    # Connect to database and get tweets
    conn = database.get_db_connection()
    c = conn.cursor()
    
    # Build query - exclude already analyzed tweets unless force_reanalyze is True
    # Also skip RT posts where the original content has already been analyzed
    if force_reanalyze:
        print("🔄 Force reanalyze mode: Will process ALL tweets (including already analyzed)")
        query = """
            SELECT t.tweet_id, t.tweet_url, t.username, t.content, t.media_links, t.original_content FROM tweets t
        """
        params = []
        if username:
            query += " WHERE t.username = ?"
            params.append(username)
    else:
        query = """
            SELECT t.tweet_id, t.tweet_url, t.username, t.content, t.media_links, t.original_content FROM tweets t 
            LEFT JOIN content_analyses ca ON t.tweet_id = ca.tweet_id 
            WHERE ca.tweet_id IS NULL
            AND NOT (
                t.post_type IN ('repost_other', 'repost_own') 
                AND t.rt_original_analyzed = 1
            )
        """
        params = []
        if username:
            query += " AND t.username = ?"
            params.append(username)
    
    query += " ORDER BY t.tweet_id DESC"
    
    if max_tweets:
        query += " LIMIT ?"
        params.append(max_tweets)
    
    search_type = "all tweets" if force_reanalyze else "unanalyzed tweets"
    print(f"🔍 Searching {search_type}{f' for @{username}' if username else ''}...")
    
    c.execute(query, params)
    tweets = c.fetchall()
    
    # Also get count of already analyzed tweets for reporting
    count_query = "SELECT COUNT(*) FROM content_analyses"
    count_params = []
    if username:
        count_query += " WHERE username = ?"
        count_params.append(username)
    
    c.execute(count_query, count_params)
    analyzed_count = c.fetchone()[0]
    
    conn.close()
    
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
            save_failed_analysis(
                analyzer=analyzer_instance,
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

def main():
    parser = argparse.ArgumentParser(
        description="Analyze tweets from database using Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_db_tweets.py                                    # Analyze all unanalyzed tweets with LLM
  python analyze_db_tweets.py --username Santi_ABASCAL          # Analyze specific user's unanalyzed tweets with LLM
  python analyze_db_tweets.py --limit 5                         # Analyze 5 unanalyzed tweets with LLM
  python analyze_db_tweets.py --force-reanalyze --limit 10      # Reanalyze 10 tweets (including already analyzed)
  python analyze_db_tweets.py --username Santi_ABASCAL -f       # Reanalyze all tweets from specific user
  python analyze_db_tweets.py --tweet-id 1234567890123456789    # Analyze/reanalyze specific tweet by ID
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