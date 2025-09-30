#!/usr/bin/env python3
"""
Analyze Database Tweets - Analyzer Pipeline
Analyzes all tweets from the database using the analyzer system.
"""

import sqlite3
import argparse
from datetime import datetime

# Import utility modules
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from analyzer.analyzer import Analyzer, save_content_analysis, ContentAnalysis, migrate_database_schema, create_analyzer
from analyzer.categories import Categories
from utils import database, analyzer, paths

DB_PATH = paths.get_db_path()

def analyze_tweets_from_db(username=None, max_tweets=None, force_reanalyze=False):
    """
    Analyze tweets from the database using the analyzer with LLM enabled.
    
    Args:
        username: Specific username to analyze (None for all)
        max_tweets: Maximum number of tweets to analyze (None for all)
        force_reanalyze: If True, reanalyze already processed tweets (useful when prompts change)
    """
    
    print("🔍 Enhanced Tweet Analysis Pipeline")
    print("=" * 50)
    
    # Ensure database schema is ready
    migrate_database_schema()
    
    # Initialize analyzer
    print("🚀 Initializing Analyzer...")
    try:
        analyzer_instance = create_analyzer(use_llm=True, verbose=False)  # Always use LLM for better explanations
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
            SELECT t.tweet_id, t.tweet_url, t.username, t.content 
            FROM tweets t
        """
        if username:
            query += " WHERE t.username = ?"
    else:
        query = """
            SELECT t.tweet_id, t.tweet_url, t.username, t.content 
            FROM tweets t 
            LEFT JOIN content_analyses ca ON t.tweet_id = ca.tweet_id 
            WHERE ca.tweet_id IS NULL
            AND NOT (
                t.post_type IN ('repost_other', 'repost_own') 
                AND t.rt_original_analyzed = 1
            )
        """
        if username:
            query += " AND t.username = ?"
    
    params = []
    if username:
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
    
    for i, (tweet_id, tweet_url, tweet_username, content) in enumerate(tweets, 1):
        print(f"📝 [{i:2d}/{len(tweets)}] Analyzing: {tweet_id}")
        
        try:
            # Run analysis (suppress verbose output)
            result = analyzer_instance.analyze_content(
                tweet_id=tweet_id,
                tweet_url=tweet_url,
                username=tweet_username,
                content=content
            )
            
            # Count categories
            category = result.category
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # Create ContentAnalysis object for saving
            analysis = ContentAnalysis(
                tweet_id=tweet_id,
                tweet_url=tweet_url,
                username=tweet_username,
                tweet_content=content,
                analysis_timestamp=datetime.now().isoformat(),
                category=result.category,
                llm_explanation=result.llm_explanation,
                analysis_method=result.analysis_method,
                pattern_matches=getattr(result, 'pattern_matches', []),
                topic_classification=getattr(result, 'topic_classification', {})
            )
            
            # Save to database
            save_content_analysis(analysis)
            
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
            print(f"    ❌ Error analyzing tweet: {e}")
            
        print()
    
    # Summary
    print("📊 Analysis Complete!")
    print("=" * 50)
    print(f"📈 Tweets analyzed: {len(results)}")
    print("📋 Category breakdown:")
    
    for category, count in sorted(category_counts.items()):
        emoji = {
            Categories.HATE_SPEECH: '🚫',
            Categories.DISINFORMATION: '❌', 
            Categories.CONSPIRACY_THEORY: '🕵️',
            Categories.FAR_RIGHT_BIAS: '⚡',
            Categories.CALL_TO_ACTION: '📢',
            Categories.GENERAL: '✅'
        }.get(category, '❓')
        
        percentage = (count / len(results)) * 100
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
        """
    )
    
    parser.add_argument('--username', '-u', help='Analyze tweets from specific username only')
    parser.add_argument('--limit', '-l', type=int, help='Maximum number of tweets to process')
    parser.add_argument('--force-reanalyze', '-f', action='store_true', 
                       help='Reanalyze already processed tweets (useful when prompts change)')
    
    args = parser.parse_args()
    
    analyze_tweets_from_db(
        username=args.username,
        max_tweets=args.limit,
        force_reanalyze=args.force_reanalyze
    )

if __name__ == '__main__':
    main()