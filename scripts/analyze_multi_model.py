"""
Multi-Model Analyzer CLI: Command-line interface for multi-model tweet analysis.

Analyzes tweets using multiple local LLM models in parallel for comparison.
Results are stored in the model_analyses table for comparison and consensus building.
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
from typing import Optional, List, Dict
from datetime import datetime
from analyzer.multi_model_analyzer import MultiModelAnalyzer
from analyzer.categories import Categories
from database import get_db_connection_context
from database import database_multi_model

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_tweets_for_multi_model_analysis(
    username: Optional[str] = None,
    limit: Optional[int] = None,
    force_reanalyze: bool = False,
    post_id: Optional[str] = None
) -> List[Dict]:
    """
    Get tweets that need multi-model analysis.
    
    Args:
        username: Optional specific username filter
        limit: Maximum number of tweets to analyze
        force_reanalyze: If True, reanalyze tweets already processed
        post_id: Optional specific post ID to analyze
        
    Returns:
        List of tweet dictionaries
    """
    with get_db_connection_context() as conn:
        cursor = conn.cursor()
        
        # Build query
        where_clauses = []
        params = []
        
        if username:
            where_clauses.append("t.username = ?")
            params.append(username)
        
        if post_id:
            where_clauses.append("t.tweet_id = ?")
            params.append(post_id)
        
        if not force_reanalyze and not post_id:
            where_clauses.append("ca.multi_model_analysis IS NULL OR ca.multi_model_analysis = 0")
        
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        limit_sql = f"LIMIT {limit}" if limit and not post_id else ""
        
        query = f'''
            SELECT 
                t.tweet_id,
                t.tweet_url,
                t.username,
                t.content,
                t.media_links,
                t.original_content
            FROM tweets t
            LEFT JOIN content_analyses ca ON t.tweet_id = ca.post_id
            {where_sql}
            ORDER BY t.tweet_timestamp DESC
            {limit_sql}
        '''
        
        cursor.execute(query, params)
        
        tweets = []
        for row in cursor.fetchall():
            media_urls = []
            if row['media_links']:
                media_urls = [u.strip() for u in row['media_links'].split(',') if u.strip()]
            
            tweets.append({
                'tweet_id': row['tweet_id'],
                'tweet_url': row['tweet_url'],
                'username': row['username'],
                'content': row['content'],
                'media_urls': media_urls,
                'original_content': row['original_content']
            })
        
        return tweets


async def analyze_tweet_multi_model(
    tweet_data: Dict,
    analyzer: MultiModelAnalyzer,
    models: Optional[List[str]] = None,
    verbose: bool = False
) -> Dict:
    """
    Analyze a single tweet with multiple models.
    
    Args:
        tweet_data: Dictionary with tweet information
        analyzer: LocalMultimodalAnalyzer instance
        models: Optional list of specific models to use
        verbose: Enable verbose output
        
    Returns:
        Dictionary with analysis results and consensus
    """
    tweet_id = tweet_data['tweet_id']
    content = tweet_data['content']
    media_urls = tweet_data['media_urls']
    
    # Combine content with quoted content if available
    original_content = tweet_data.get('original_content')
    if original_content and original_content.strip():
        analysis_content = f"{content}\n\n[Contenido citado]: {original_content}"
        if verbose:
            print(f"    📎 Including quoted tweet content")
    else:
        analysis_content = content
    
    if verbose:
        print(f"🔍 Analyzing tweet {tweet_id} with multiple models")
        if media_urls:
            print(f"    🖼️  Found {len(media_urls)} media files")
    
    # Run multi-model analysis
    model_results = await analyzer.analyze_with_multiple_models(
        content=analysis_content,
        media_urls=media_urls if media_urls else None,
        models=models
    )
    
    # Save individual model results to database
    with get_db_connection_context() as conn:
        cursor = conn.cursor()
        
        # If reanalyzing, delete old analyses for these models
        if models:
            placeholders = ','.join('?' * len(models))
            cursor.execute(
                f"DELETE FROM model_analyses WHERE post_id = ? AND model_name IN ({placeholders})",
                [tweet_id] + models
            )
            if verbose and cursor.rowcount > 0:
                print(f"    🗑️  Deleted {cursor.rowcount} old analyses before reanalyzing")
        
        for model_name, (category, explanation, processing_time) in model_results.items():
            database_multi_model.save_model_analysis(
                conn=conn,
                post_id=tweet_id,
                model_name=model_name,
                category=category,
                explanation=explanation,
                processing_time=processing_time
            )
        
        # Calculate and save consensus
        consensus = database_multi_model.get_model_consensus(conn, tweet_id)
        if consensus:
            database_multi_model.update_consensus_in_content_analyses(conn, tweet_id)
    
    return {
        'tweet_id': tweet_id,
        'model_results': model_results,
        'consensus': consensus
    }


async def analyze_tweets_multi_model_cli(
    username: Optional[str] = None,
    limit: Optional[int] = None,
    force_reanalyze: bool = False,
    models: Optional[List[str]] = None,
    verbose: bool = False,
    post_id: Optional[str] = None
):
    """
    CLI wrapper for multi-model tweet analysis.
    
    Args:
        username: Specific username to analyze
        limit: Maximum number of tweets to analyze
        force_reanalyze: If True, reanalyze already processed tweets
        models: List of models to use (None = all available)
        verbose: Enable verbose output
    """
    print("🔍 Multi-Model Tweet Analysis Pipeline")
    print("=" * 50)
    
    # Initialize analyzer
    print("🚀 Initializing Multi-Model Analyzer...")
    analyzer = MultiModelAnalyzer(verbose=verbose)
    
    # Determine which models to use
    if models is None:
        models = list(MultiModelAnalyzer.AVAILABLE_MODELS.keys())
    
    print(f"✅ Using {len(models)} models: {', '.join(models)}")
    print()
    
    # Get tweets for analysis
    tweets = get_tweets_for_multi_model_analysis(
        username=username,
        limit=limit,
        force_reanalyze=force_reanalyze,
        post_id=post_id
    )
    
    if not tweets:
        search_desc = "tweets" if force_reanalyze else "unanalyzed tweets"
        print(f"✅ No {search_desc} found{f' for @{username}' if username else ''}")
        return
    
    print(f"📊 Found {len(tweets)} tweets for multi-model analysis")
    print()
    
    # Analyze each tweet
    results = []
    category_votes = {}
    model_stats = {model: {'total': 0, 'categories': {}} for model in models}
    
    for idx, tweet_data in enumerate(tweets, 1):
        tweet_id = tweet_data['tweet_id']
        content = tweet_data['content']
        
        ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        print(f"{ts} 📝 [{idx:2d}/{len(tweets)}] Analyzing: {tweet_id}")
        
        if not verbose:
            # Show content preview in non-verbose mode
            content_preview = content[:60] + "..." if len(content) > 60 else content
            print(f"    📝 {content_preview}")
        
        try:
            result = await analyze_tweet_multi_model(
                tweet_data=tweet_data,
                analyzer=analyzer,
                models=models,
                verbose=verbose
            )
            
            results.append(result)
            
            # Update statistics
            consensus = result.get('consensus')
            if consensus:
                consensus_cat = consensus['category']
                category_votes[consensus_cat] = category_votes.get(consensus_cat, 0) + 1
                
                # Show consensus
                category_emoji = {
                    Categories.HATE_SPEECH: '🚫',
                    Categories.DISINFORMATION: '❌',
                    Categories.CONSPIRACY_THEORY: '🕵️',
                    Categories.ANTI_IMMIGRATION: '🌍',
                    Categories.ANTI_LGBTQ: '🏳️‍🌈',
                    Categories.ANTI_FEMINISM: '👩',
                    Categories.CALL_TO_ACTION: '📢',
                    Categories.GENERAL: '✅'
                }.get(consensus_cat, '❓')
                
                agreement_pct = consensus['agreement_score'] * 100
                print(f"    🎯 Consensus: {category_emoji} {consensus_cat} ({agreement_pct:.0f}% agreement)")
                
                # Show individual model results in verbose mode
                if verbose:
                    print(f"    📊 Model votes:")
                    for model_name, (cat, exp, time_taken) in result['model_results'].items():
                        print(f"        • {model_name}: {cat} ({time_taken:.1f}s)")
                        print(f"          {exp[:80]}...")
                
                # Update model stats
                for model_name, (cat, _, _) in result['model_results'].items():
                    model_stats[model_name]['total'] += 1
                    model_stats[model_name]['categories'][cat] = \
                        model_stats[model_name]['categories'].get(cat, 0) + 1
            
            print()
            
        except Exception as e:
            print(f"    ❌ Error analyzing tweet: {e}")
            print()
            continue
    
    # Print summary
    print("📊 Multi-Model Analysis Complete!")
    print("=" * 50)
    print(f"📈 Tweets processed: {len(results)}")
    print()
    
    if category_votes:
        print("📋 Consensus category breakdown:")
        total_analyzed = len(results)
        for category, count in sorted(category_votes.items(), key=lambda x: x[1], reverse=True):
            emoji = {
                Categories.HATE_SPEECH: '🚫',
                Categories.DISINFORMATION: '❌',
                Categories.CONSPIRACY_THEORY: '🕵️',
                Categories.ANTI_IMMIGRATION: '🌍',
                Categories.ANTI_LGBTQ: '🏳️‍🌈',
                Categories.ANTI_FEMINISM: '👩',
                Categories.CALL_TO_ACTION: '📢',
                Categories.GENERAL: '✅'
            }.get(category, '❓')
            
            percentage = (count / total_analyzed) * 100 if total_analyzed > 0 else 0
            print(f"    {emoji} {category}: {count} ({percentage:.1f}%)")
    
    print()
    print("🤖 Model performance summary:")
    for model_name in models:
        stats = model_stats[model_name]
        if stats['total'] > 0:
            print(f"    • {model_name}: {stats['total']} analyses")
            # Show top 3 categories for this model
            top_cats = sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True)[:3]
            if top_cats:
                cats_str = ", ".join([f"{cat} ({count})" for cat, count in top_cats])
                print(f"      Top categories: {cats_str}")
    
    print()
    print("✅ Results saved to model_analyses table in database")
    print("💡 View comparisons at: http://localhost:5000/models/comparison")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze tweets using multiple LLM models for comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/analyze_multi_model.py                           # Analyze all unanalyzed tweets with all models
  python scripts/analyze_multi_model.py --username Santi_ABASCAL  # Analyze specific user with all models
  python scripts/analyze_multi_model.py --limit 5                 # Analyze 5 tweets
  python scripts/analyze_multi_model.py --force-reanalyze         # Reanalyze all tweets
  python scripts/analyze_multi_model.py --models gemma3:4b,gpt-oss:20b  # Use specific models
  python scripts/analyze_multi_model.py --post-id 1234567890      # Analyze specific post ID
  python scripts/analyze_multi_model.py -v                        # Verbose output with per-model details
        """
    )
    
    parser.add_argument('--username', '-u', help='Analyze tweets from specific username only')
    parser.add_argument('--limit', '-l', type=int, help='Maximum number of tweets to process')
    parser.add_argument('--force-reanalyze', '-f', action='store_true',
                       help='Reanalyze already processed tweets')
    parser.add_argument('--models', '-m', help='Comma-separated list of models to use (default: all)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed analysis output for each model')
    parser.add_argument('--post-id', '-p', help='Analyze specific post ID only')
    
    args = parser.parse_args()
    
    # Parse models list
    models = None
    if args.models:
        models = [m.strip() for m in args.models.split(',') if m.strip()]
    
    # Run async analysis
    asyncio.run(analyze_tweets_multi_model_cli(
        username=args.username,
        limit=args.limit,
        force_reanalyze=args.force_reanalyze,
        models=models,
        verbose=args.verbose,
        post_id=args.post_id
    ))


if __name__ == '__main__':
    main()
