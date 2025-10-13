"""
Shared utility functions for the dimetuverdad web application.
Contains common database operations, tweet processing, and admin actions.
"""

import sys
import os
import importlib.util
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
import json
import math
from datetime import datetime

# Import repository interfaces
from repositories import (
    get_tweet_repository,
    get_content_analysis_repository,
    get_account_repository
)

import config


def get_db_connection():
    """Get database connection with row factory for easier access."""
    # Legacy compatibility - repositories handle connections internally
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from utils.database import get_db_connection as db_get_connection
    return db_get_connection()


def get_tweet_data(tweet_id) -> Optional[Dict[str, Any]]:
    """Get tweet data for analysis."""
    tweet_repo = get_tweet_repository()
    tweet_data = tweet_repo.get_tweet_by_id(tweet_id)

    if tweet_data:
        # Parse media_links into a list for backward compatibility
        media_links = tweet_data.get('media_links', '')
        if media_links:
            tweet_data['media_urls'] = [url.strip() for url in media_links.split(',') if url.strip()]
        else:
            tweet_data['media_urls'] = []
        return tweet_data
    return None


async def reanalyze_tweet(tweet_id) -> Any:
    """Reanalyze a single tweet and return the result."""
    from analyzer.analyze_twitter import reanalyze_tweet as analyzer_reanalyze_tweet  # Local import
    return await analyzer_reanalyze_tweet(tweet_id)


def reanalyze_tweet_sync(tweet_id) -> Any:
    """Synchronous wrapper for reanalyze_tweet that can be called from Flask routes."""
    import asyncio
    try:
        # Check if there's already a running event loop
        try:
            loop = asyncio.get_running_loop()
            # If loop is already running, we need to use a different approach
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, reanalyze_tweet(tweet_id))
                return future.result(timeout=30)  # 30 second timeout
        except RuntimeError:
            # No event loop running, create a new one
            return asyncio.run(reanalyze_tweet(tweet_id))
    except Exception as e:
        print(f"❌ Error in reanalyze_tweet_sync: {e}")
        raise


def refetch_tweet(tweet_id) -> bool:
    """Refetch a single tweet from Twitter."""
    from fetcher.fetch_tweets import refetch_manager  # Local import
    return refetch_manager.refetch_single_tweet(tweet_id)


def handle_reanalyze_action(tweet_id, referrer) -> None:
    """Handle the reanalyze action for a tweet."""
    from flask import flash

    print(f"🔄 Reanalyze action triggered for tweet_id: {tweet_id}")

    tweet_data = get_tweet_data(tweet_id)

    if not tweet_data:
        print(f"❌ Tweet data not found for tweet_id: {tweet_id}")
        flash('Tweet no encontrado', 'error')
        return

    print(f"🔄 Reanalizando tweet {tweet_id} de @{tweet_data.get('username', 'unknown')}")

    # Debug: Check if tweet data was retrieved
    print(f"🔍 Tweet data retrieved: {tweet_data is not None}")
    if tweet_data:
        print(f"🔍 Tweet content length: {len(tweet_data.get('content', ''))}")
        print(f"🔍 Tweet media_urls: {tweet_data.get('media_urls', [])}")

    # Reanalyze the content
    try:
        analysis_result = reanalyze_tweet_sync(tweet_id)
        print(f"🔍 Reanalyze result type: {type(analysis_result)}")
        print(f"🔍 Reanalyze result: {analysis_result is not None}")
    except Exception as reanalyze_e:
        print(f"❌ Error in reanalyze_tweet: {reanalyze_e}")
        from flask import current_app
        current_app.logger.error(f"Error in reanalyze_tweet for {tweet_id}: {str(reanalyze_e)}")
        flash('Error interno durante el reanálisis. Inténtalo de nuevo.', 'error')
        return

    if analysis_result:
        try:
            # More defensive access to category
            if hasattr(analysis_result, 'category') and analysis_result.category:
                category = analysis_result.category
                print(f"🔍 Analysis category: {category}")
                flash(f'Tweet reanálizado correctamente. Nueva categoría: {category}', 'success')
            else:
                print(f"❌ Analysis result has no valid category: {analysis_result}")
                flash('Tweet reanálizado pero no se pudo determinar la categoría.', 'warning')
        except Exception as cat_e:
            print(f"❌ Error accessing category: {cat_e}")
            print(f"❌ Analysis result attributes: {dir(analysis_result) if analysis_result else 'None'}")
            flash('Tweet reanálizado pero error al acceder a la categoría.', 'warning')
    else:
        print(f"❌ Reanalyze returned None")
        flash('Error: No se pudo reanalizar el tweet. Verifica que existe y tiene contenido.', 'error')


def handle_refresh_action(tweet_id, referrer) -> None:
    """Handle the refresh action for a tweet."""
    from flask import flash

    print(f"🔄 Refresh action triggered for tweet_id: {tweet_id}")

    try:
        success = refetch_tweet(tweet_id)
        if success:
            print(f"✅ Tweet {tweet_id} refreshed successfully")
            flash('Datos del tweet actualizados correctamente desde Twitter', 'success')
        else:
            print(f"❌ Failed to refresh tweet {tweet_id}")
            flash('Error al actualizar los datos del tweet. Verifica que el tweet existe en Twitter.', 'error')
    except Exception as refresh_e:
        print(f"❌ Error in refetch_tweet: {refresh_e}")
        from flask import current_app
        current_app.logger.error(f"Error in refetch_tweet for {tweet_id}: {str(refresh_e)}")
        flash('Error interno al actualizar los datos del tweet. Inténtalo de nuevo.', 'error')


def handle_refresh_and_reanalyze_action(tweet_id, referrer) -> None:
    """Handle the refresh and reanalyze action for a tweet."""
    from flask import flash

    print(f"🔄 Refresh and reanalyze action triggered for tweet_id: {tweet_id}")

    # First refresh the tweet data
    try:
        refresh_success = refetch_tweet(tweet_id)
        if not refresh_success:
            print(f"❌ Failed to refresh tweet {tweet_id}, skipping reanalysis")
            flash('Error al actualizar los datos del tweet. No se pudo proceder con el reanálisis.', 'error')
            return

        print(f"✅ Tweet {tweet_id} refreshed successfully, proceeding with reanalysis")
    except Exception as refresh_e:
        print(f"❌ Error in refetch_tweet during refresh_and_reanalyze: {refresh_e}")
        from flask import current_app
        current_app.logger.error(f"Error in refetch_tweet for {tweet_id}: {str(refresh_e)}")
        flash('Error interno al actualizar los datos del tweet. Inténtalo de nuevo.', 'error')
        return

    # Then reanalyze the content
    try:
        analysis_result = reanalyze_tweet_sync(tweet_id)
        print(f"🔍 Reanalyze result type: {type(analysis_result)}")
        print(f"🔍 Reanalyze result: {analysis_result is not None}")
    except Exception as reanalyze_e:
        print(f"❌ Error in reanalyze_tweet: {reanalyze_e}")
        from flask import current_app
        current_app.logger.error(f"Error in reanalyze_tweet for {tweet_id}: {str(reanalyze_e)}")
        flash('Datos actualizados pero error interno durante el reanálisis. Inténtalo de nuevo.', 'warning')
        return

    if analysis_result:
        try:
            # More defensive access to category
            if hasattr(analysis_result, 'category') and analysis_result.category:
                category = analysis_result.category
                print(f"🔍 Analysis category: {category}")
                flash(f'Tweet actualizado y reanálizado correctamente. Nueva categoría: {category}', 'success')
            else:
                print(f"❌ Analysis result has no valid category: {analysis_result}")
                flash('Tweet actualizado y reanálizado pero no se pudo determinar la categoría.', 'warning')
        except Exception as cat_e:
            print(f"❌ Error accessing category: {cat_e}")
            print(f"❌ Analysis result attributes: {dir(analysis_result) if analysis_result else 'None'}")
            flash('Tweet actualizado y reanálizado pero error al acceder a la categoría.', 'warning')
    else:
        print(f"❌ Reanalyze returned None")
        flash('Datos actualizados pero error: No se pudo reanalizar el tweet. Verifica que tiene contenido.', 'warning')


def handle_manual_update_action(tweet_id, new_category, new_explanation) -> None:
    """Handle the manual update action for a tweet."""
    from flask import flash

    if not new_category or not new_explanation:
        flash('Categoría y explicación son requeridas', 'error')
        return

    conn = get_db_connection()
    # Check if analysis exists
    row = conn.execute("SELECT post_id FROM content_analyses WHERE post_id = ?", (tweet_id,)).fetchone()
    if row:
        # Update existing analysis
        conn.execute("UPDATE content_analyses SET category = ?, llm_explanation = ?, analysis_method = 'manual', analysis_timestamp = CURRENT_TIMESTAMP WHERE post_id = ?", (new_category, new_explanation, tweet_id))
        conn.commit()
        success = True
    else:
        # Create new analysis entry
        tweet_row = conn.execute("SELECT username, content, tweet_url FROM tweets WHERE tweet_id = ?", (tweet_id,)).fetchone()
        if tweet_row:
            conn.execute("""
                INSERT INTO content_analyses (post_id, category, llm_explanation, analysis_method, author_username, post_content, post_url, analysis_timestamp)
                VALUES (?, ?, ?, 'manual', ?, ?, ?, CURRENT_TIMESTAMP)
            """, (tweet_id, new_category, new_explanation, tweet_row['username'], tweet_row['content'], tweet_row['tweet_url']))
            conn.commit()
            success = True
        else:
            success = False
    conn.close()

    if success:
        flash('Análisis actualizado correctamente', 'success')
    else:
        flash('Error al actualizar el análisis', 'error')


def get_tweet_display_data(tweet_id, referrer) -> Any:
    """Get tweet data for display in the edit form."""
    from flask import flash, redirect, url_for

    tweet_repo = get_tweet_repository()
    content_analysis_repo = get_content_analysis_repository()

    # Get tweet data
    tweet_data = tweet_repo.get_tweet_by_id(tweet_id)

    if not tweet_data:
        flash('Tweet no encontrado', 'error')
        return redirect(referrer or url_for('admin.admin_dashboard'))

    # Get analysis data
    analysis_data = content_analysis_repo.get_analysis_by_post_id(tweet_id)

    # Convert to dict safely
    tweet_dict = {
        'content': tweet_data.get('content', ''),
        'username': tweet_data.get('username', ''),
        'tweet_timestamp': tweet_data.get('tweet_timestamp', ''),
        'category': analysis_data.get('category', 'general') if analysis_data else 'general',
        'llm_explanation': analysis_data.get('llm_explanation', '') if analysis_data else '',
        'tweet_url': tweet_data.get('tweet_url', ''),
        'original_content': tweet_data.get('original_content', '')
    }

    from web.utils.decorators import ANALYSIS_CATEGORIES
    categories = ANALYSIS_CATEGORIES

    return tweet_dict, categories


def get_account_statistics(username) -> Dict[str, Any]:
    """Get comprehensive statistics for an account."""
    from flask_caching import Cache

    # This function uses caching, but we'll need to handle it differently in the shared context
    # For now, we'll implement without caching and let individual routes handle caching
    tweet_repo = get_tweet_repository()
    content_analysis_repo = get_content_analysis_repository()

    # Basic tweet stats
    tweet_count = tweet_repo.get_tweet_count_by_username(username)
    tweets = tweet_repo.get_tweets_by_username(username, limit=1000)  # Get recent tweets for stats

    # Calculate stats from tweet data
    media_count = sum(1 for tweet in tweets if tweet.get('media_count', 0) > 0)
    avg_media = sum(tweet.get('media_count', 0) for tweet in tweets) / len(tweets) if tweets else 0

    # Get active days (unique dates)
    active_days = len(set(
        tweet.get('tweet_timestamp', '').split('T')[0]
        for tweet in tweets
        if tweet.get('tweet_timestamp')
    ))

    basic_stats = {
        'total_tweets': tweet_count,
        'active_days': active_days,
        'tweets_with_media': media_count,
        'avg_media_per_tweet': avg_media
    }

    # Analysis results breakdown
    analyses = content_analysis_repo.get_analyses_by_username(username)
    analysis_stats = {}
    for analysis in analyses:
        category = analysis.get('category', 'unknown')
        analysis_stats[category] = analysis_stats.get(category, 0) + 1

    # Recent activity (last 7 days)
    recent_tweets = [
        tweet for tweet in tweets
        if tweet.get('tweet_timestamp') and
        (datetime.now() - datetime.fromisoformat(tweet['tweet_timestamp'].replace('Z', '+00:00'))).days <= 7
    ]

    recent_activity = {}
    for tweet in recent_tweets:
        date = tweet.get('tweet_timestamp', '').split('T')[0]
        if date:
            recent_activity[date] = recent_activity.get(date, 0) + 1

    recent_activity_list = [
        {'date': date, 'tweets_count': count}
        for date, count in recent_activity.items()
    ]
    recent_activity_list.sort(key=lambda x: x['date'], reverse=True)

    return {
        'basic': basic_stats,
        'analysis': [{'category': cat, 'count': count} for cat, count in analysis_stats.items()],
        'recent_activity': recent_activity_list
    }


def get_all_accounts(page: int = 1, per_page: int = 10) -> Dict[str, Any]:
    """Get list of all accounts with basic stats, paginated and sorted by non-general posts."""
    # Use direct SQL for test compatibility
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT username, profile_pic_url, last_scraped
        FROM accounts
        ORDER BY last_scraped DESC
        LIMIT ? OFFSET ?
    """, (per_page, (page - 1) * per_page)).fetchall()
    total_count_row = conn.execute("SELECT COUNT(*) AS cnt FROM accounts").fetchone()
    # Handle different row types: tuples, MockRow with .get(), and sqlite3.Row with dict access
    total_count = total_count_row['cnt'] if total_count_row and 'cnt' in total_count_row else 0

    accounts_with_stats = []
    for r in rows:
        username = r['username']
        profile_pic_url = r['profile_pic_url']
        last_scraped = r['last_scraped']

        # Get tweet count for this account
        tweet_count_row = conn.execute("SELECT COUNT(*) AS cnt FROM tweets WHERE username = ?", (username,)).fetchone()
        tweet_count = tweet_count_row['cnt'] if tweet_count_row and 'cnt' in tweet_count_row else 0

        # Get analyzed posts count
        analyzed_count_row = conn.execute("""
            SELECT COUNT(*) AS cnt FROM content_analyses ca
            JOIN tweets t ON ca.post_id = t.tweet_id
            WHERE t.username = ?
        """, (username,)).fetchone()
        analyzed_posts = analyzed_count_row['cnt'] if analyzed_count_row and 'cnt' in analyzed_count_row else 0

        # Get problematic posts count (non-general categories)
        problematic_count_row = conn.execute("""
            SELECT COUNT(*) AS cnt FROM content_analyses ca
            JOIN tweets t ON ca.post_id = t.tweet_id
            WHERE t.username = ? AND ca.category != 'general'
        """, (username,)).fetchone()
        problematic_posts = problematic_count_row['cnt'] if problematic_count_row and 'cnt' in problematic_count_row else 0

        accounts_with_stats.append({
            'username': username,
            'profile_pic_url': profile_pic_url,
            'last_activity': last_scraped,
            'tweet_count': tweet_count,
            'analyzed_posts': analyzed_posts,
            'problematic_posts': problematic_posts
        })

    conn.close()

    return {
        'accounts': accounts_with_stats,
        'total': total_count,
        'page': page,
        'per_page': per_page,
        'total_pages': math.ceil(total_count / per_page) if total_count > 0 else 1
    }


def get_user_profile_data(username: str) -> Optional[Dict[str, Any]]:
    """Get user profile data and basic tweet count."""
    account_repo = get_account_repository()
    tweet_repo = get_tweet_repository()

    account_data = account_repo.get_account_by_username(username)
    if account_data:
        tweet_count = tweet_repo.get_tweet_count_by_username(username)
        return {
            'profile_pic_url': account_data.get('profile_pic_url'),
            'total_tweets': tweet_count
        }
    return None


def get_user_tweets_data(username: str, page: int, per_page: int,
                        category_filter: Optional[str], post_type_filter: Optional[str],
                        total_tweets_all: int, date_from: Optional[str] = None, date_to: Optional[str] = None,
                        analysis_method_filter: Optional[str] = None) -> Dict[str, Any]:
    """Get paginated tweets data with filtering."""
    tweet_repo = get_tweet_repository()
    content_analysis_repo = get_content_analysis_repository()

    # Get all tweets for the user first (we'll filter in memory for now)
    all_tweets = tweet_repo.get_tweets_by_username(username)

    # Apply filters
    filtered_tweets = []
    for tweet in all_tweets:
        # Get analysis for this tweet
        analysis = content_analysis_repo.get_analysis_by_post_id(tweet['tweet_id'])

        # Apply category filter
        if category_filter and category_filter != 'all':
            if not analysis or analysis.get('category') != category_filter:
                continue

        # Apply post_type filter
        if post_type_filter and post_type_filter != 'all':
            if tweet.get('post_type') != post_type_filter:
                continue

        # Apply date range filters
        tweet_date = tweet.get('tweet_timestamp', '').split('T')[0] if tweet.get('tweet_timestamp') else ''
        if date_from and tweet_date < date_from:
            continue
        if date_to and tweet_date > date_to:
            continue

        # Apply analysis method filter
        if analysis_method_filter and analysis_method_filter != 'all':
            if not analysis or analysis.get('analysis_method') != analysis_method_filter:
                continue

        # Add analysis data to tweet
        tweet_with_analysis = dict(tweet)
        if analysis:
            tweet_with_analysis.update({
                'analysis_category': analysis.get('category'),
                'llm_explanation': analysis.get('llm_explanation'),
                'analysis_method': analysis.get('analysis_method'),
                'analysis_timestamp': analysis.get('analysis_timestamp'),
                'categories_detected': analysis.get('categories_detected'),
                'multimodal_analysis': analysis.get('multimodal_analysis'),
                'media_analysis': analysis.get('media_analysis'),
                'verification_data': analysis.get('verification_data'),
                'verification_confidence': analysis.get('verification_confidence', 0.0)
            })
        else:
            tweet_with_analysis.update({
                'analysis_category': None,
                'llm_explanation': None,
                'analysis_method': None,
                'analysis_timestamp': None,
                'categories_detected': None,
                'multimodal_analysis': False,
                'media_analysis': None,
                'verification_data': None,
                'verification_confidence': 0.0
            })

        filtered_tweets.append(tweet_with_analysis)

    # Sort by priority (non-general first) then by timestamp
    filtered_tweets.sort(key=lambda x: (
        0 if x.get('analysis_category') in [None, 'general'] else 1,
        x.get('tweet_timestamp', ''),
    ), reverse=True)

    # Apply pagination
    total_filtered = len(filtered_tweets)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_tweets = filtered_tweets[start_idx:end_idx]

    # Process tweets for display
    tweets = [process_tweet_row(tweet) for tweet in paginated_tweets]

    return {
        'tweets': tweets,
        'page': page,
        'per_page': per_page,
        'total_tweets': total_filtered,
        'total_pages': math.ceil(total_filtered / per_page) if total_filtered > 0 else 1
    }


def process_tweet_row(row) -> Dict[str, Any]:
    """Process a raw tweet row into display format."""
    # Parse multi-category data
    categories_detected = []
    try:
        if row['categories_detected']:
            categories_detected = json.loads(row['categories_detected'])
    except (json.JSONDecodeError, TypeError):
        if row['analysis_category']:
            categories_detected = [row['analysis_category']]

    tweet = {
        'tweet_url': row['tweet_url'],
        'content': row['content'],
        'media_links': row['media_links'],
        'hashtags_parsed': json.loads(row['hashtags']) if row['hashtags'] else [],
        'mentions_parsed': json.loads(row['mentions']) if row['mentions'] else [],
        'tweet_timestamp': row['tweet_timestamp'],
        'post_type': row['post_type'],
        'tweet_id': row['tweet_id'],
        'analysis_category': row['analysis_category'],
        'llm_explanation': row['llm_explanation'],
        'analysis_method': row['analysis_method'],
        'analysis_timestamp': row['analysis_timestamp'],
        'categories_detected': categories_detected,
        'multimodal_analysis': bool(row['multimodal_analysis']) if row['multimodal_analysis'] is not None else False,
        'media_analysis': row['media_analysis'],
        'is_deleted': row['is_deleted'],
        'is_edited': row['is_edited'],
        'rt_original_analyzed': row['rt_original_analyzed'],
        'original_author': row['original_author'],
        'original_tweet_id': row['original_tweet_id'],
        'reply_to_username': row['reply_to_username'],
        'verification_data': json.loads(row['verification_data']) if row['verification_data'] else None,
        'verification_confidence': row['verification_confidence'] if 'verification_confidence' in row.keys() else 0.0
    }

    # Post status warnings
    tweet['post_status_warnings'] = []
    if tweet['is_deleted']:
        tweet['post_status_warnings'].append({
            'type': 'deleted',
            'message': 'Este tweet fue eliminado',
            'icon': 'fas fa-trash',
            'class': 'alert-danger'
        })
    if tweet['is_edited']:
        tweet['post_status_warnings'].append({
            'type': 'edited',
            'message': 'Este tweet fue editado',
            'icon': 'fas fa-edit',
            'class': 'alert-warning'
        })

    # RT display logic
    tweet['is_rt'] = tweet['post_type'] in ['repost_other', 'repost_own', 'repost_reply']
    tweet['rt_type'] = tweet['post_type'] if tweet['is_rt'] else None

    # Use the appropriate analysis field
    tweet['analysis_display'] = (
        tweet['media_analysis'] if tweet['multimodal_analysis'] and tweet['media_analysis']
        else tweet['llm_explanation'] or "Sin análisis disponible"
    )
    tweet['category'] = tweet['analysis_category'] or 'general'
    tweet['has_multiple_categories'] = len(categories_detected) > 1

    return tweet


def get_user_analysis_stats(username: str) -> Dict[str, Any]:
    """Get user analysis statistics."""
    content_analysis_repo = get_content_analysis_repository()

    analyses = content_analysis_repo.get_analyses_by_username(username)

    # Calculate stats
    total_analyzed = len(analyses)
    category_counts = {}

    for analysis in analyses:
        category = analysis.get('category', 'unknown')
        category_counts[category] = category_counts.get(category, 0) + 1

    # Build analysis stats
    analysis_stats = []
    for category, count in category_counts.items():
        if count > 0:
            analysis_stats.append({
                'category': category,
                'count': count,
                'percentage': (count * 100.0 / total_analyzed) if total_analyzed > 0 else 0
            })

    return {
        'total_analyzed': total_analyzed,
        'analysis': analysis_stats
    }


def prepare_user_page_template_data(username: str, tweets_data: Dict[str, Any], user_stats: Dict[str, Any],
                                   total_tweets_all: int, page: int, per_page: int,
                                   category_filter: Optional[str], user_profile_pic: Optional[str],
                                   date_from: Optional[str] = None, date_to: Optional[str] = None,
                                   analysis_method: Optional[str] = None) -> Dict[str, Any]:
    """Prepare all data for user page template rendering."""
    return {
        'username': username,
        'tweets_data': tweets_data,
        'stats': {
            'basic': {
                'total_tweets': total_tweets_all,
                'analyzed_posts': user_stats['total_analyzed']
            },
            'analysis': user_stats['analysis']
        },
        'current_category': category_filter,
        'user_profile_pic': user_profile_pic,
        'date_from': date_from,
        'date_to': date_to,
        'analysis_method': analysis_method
    }