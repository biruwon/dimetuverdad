#!/usr/bin/env python3
"""
Analyze Database Tweets - Enhanced Analyzer Pipeline
Analyzes all tweets from the database using the enhanced analyzer system.
"""

import sys
import sqlite3
import argparse
from enhanced_analyzer import EnhancedAnalyzer, save_content_analysis, ContentAnalysis, migrate_database_schema
import json
from datetime import datetime

DB_PATH = "accounts.db"

def analyze_tweets_from_db(username=None, max_tweets=None, force_reanalyze=False):
    """
    Analyze tweets from the database using the enhanced analyzer with LLM enabled.
    
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
    print("🚀 Initializing Enhanced Analyzer...")
    try:
        analyzer = EnhancedAnalyzer(use_llm=True, verbose=False)  # Always use LLM for better explanations
        print("✅ Analyzer ready!")
    except Exception as e:
        print(f"❌ Error initializing analyzer: {e}")
        return
    
    # Connect to database and get tweets
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Build query - exclude already analyzed tweets unless force_reanalyze is True
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
            result = analyzer.analyze_content(
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
                targeted_groups=getattr(result, 'targeted_groups', []),
                pattern_matches=getattr(result, 'pattern_matches', []),
                topic_classification=getattr(result, 'topic_classification', {}),
                claims_detected=getattr(result, 'claims_detected', [])
            )
            
            # Save to database
            save_content_analysis(analysis)
            
            results.append(result)
            
            # Show result
            category_emoji = {
                'hate_speech': '🚫',
                'disinformation': '❌',
                'conspiracy_theory': '🕵️',
                'far_right_bias': '⚡',
                'call_to_action': '📢',
                'general': '✅'
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
            'hate_speech': '🚫',
            'disinformation': '❌', 
            'conspiracy_theory': '🕵️',
            'far_right_bias': '⚡',
            'call_to_action': '📢',
            'general': '✅'
        }.get(category, '❓')
        
        percentage = (count / len(results)) * 100
        print(f"    {emoji} {category}: {count} ({percentage:.1f}%)")
    
    print()
    print("✅ Results saved to content_analyses table in database")
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Analyze tweets from database using Enhanced Analyzer",
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