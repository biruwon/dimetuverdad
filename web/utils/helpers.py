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

# Import database module dynamically to avoid path issues
project_root = Path(__file__).parent.parent.parent
database_path = project_root / "utils" / "database.py"
spec = importlib.util.spec_from_file_location("utils.database", database_path)
database_module = importlib.util.module_from_spec(spec)
sys.modules["utils.database"] = database_module
spec.loader.exec_module(database_module)

# Import functions from the loaded module
db_get_connection = database_module.get_db_connection
db_get_tweet_data = database_module.get_tweet_data

import config


def get_db_connection():
    """Get database connection with row factory for easier access."""
    return db_get_connection()


def get_tweet_data(tweet_id) -> Optional[Dict[str, Any]]:
    """Get tweet data for analysis."""
    return db_get_tweet_data(tweet_id)


def reanalyze_tweet(tweet_id) -> Any:
    """Reanalyze a single tweet and return the result."""
    from analyzer.analyze_twitter import reanalyze_tweet as analyzer_reanalyze_tweet  # Local import
    return analyzer_reanalyze_tweet(tweet_id)


def refetch_tweet(tweet_id) -> bool:
    """Refetch a single tweet from Twitter."""
    from fetcher.fetch_tweets import refetch_single_tweet  # Local import
    return refetch_single_tweet(tweet_id)


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
        analysis_result = reanalyze_tweet(tweet_id)
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
        analysis_result = reanalyze_tweet(tweet_id)
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

    # Update analysis manually
    conn.execute("""
        UPDATE content_analyses
        SET category = ?, llm_explanation = ?, analysis_method = 'manual', analysis_timestamp = datetime('now')
        WHERE tweet_id = ?
    """, (new_category, new_explanation, tweet_id))

    if conn.total_changes == 0:
        # Create new analysis if none exists
        conn.execute("""
            INSERT INTO content_analyses (tweet_id, category, llm_explanation, analysis_method, analysis_timestamp, username)
            SELECT ?, ?, ?, 'manual', datetime('now'), username FROM tweets WHERE tweet_id = ?
        """, (tweet_id, new_category, new_explanation, tweet_id))

    conn.commit()
    conn.close()

    flash('Análisis actualizado correctamente', 'success')


def get_tweet_display_data(tweet_id, referrer) -> Any:
    """Get tweet data for display in the edit form."""
    from flask import flash, redirect, url_for

    conn = get_db_connection()

    try:
        # Get tweet and current analysis
        tweet_data = conn.execute("""
            SELECT
                t.content, t.username, t.tweet_timestamp,
                ca.category, ca.llm_explanation, t.tweet_url, t.original_content
            FROM tweets t
            LEFT JOIN content_analyses ca ON t.tweet_id = ca.tweet_id
            WHERE t.tweet_id = ?
        """, (tweet_id,)).fetchone()

        conn.close()

        if not tweet_data:
            flash('Tweet no encontrado', 'error')
            return redirect(referrer or url_for('admin_dashboard'))

        # Convert to dict safely
        tweet_dict = {
            'content': tweet_data[0] or '',
            'username': tweet_data[1] or '',
            'tweet_timestamp': tweet_data[2] or '',
            'category': tweet_data[3] or 'general',
            'llm_explanation': tweet_data[4] or '',
            'tweet_url': tweet_data[5] or '',
            'original_content': tweet_data[6] or ''
        }

        from web.utils.decorators import ANALYSIS_CATEGORIES
        categories = ANALYSIS_CATEGORIES

        return tweet_dict, categories

    except Exception as e:
        from flask import current_app
        current_app.logger.error(f"Error loading edit analysis for {tweet_id}: {str(e)}")
        flash('No se pudo cargar la información del tweet. Inténtalo de nuevo.', 'error')
        return redirect(referrer or url_for('admin_dashboard'))


def get_account_statistics(username) -> Dict[str, Any]:
    """Get comprehensive statistics for an account."""
    from flask_caching import Cache

    # This function uses caching, but we'll need to handle it differently in the shared context
    # For now, we'll implement without caching and let individual routes handle caching
    conn = get_db_connection()

    # Basic tweet stats (removed engagement metrics focus)
    basic_stats = conn.execute("""
        SELECT
            COUNT(*) as total_tweets,
            COUNT(DISTINCT DATE(tweet_timestamp)) as active_days,
            COUNT(CASE WHEN media_count > 0 THEN 1 END) as tweets_with_media,
            AVG(media_count) as avg_media_per_tweet
        FROM tweets
        WHERE username = ?
    """, (username,)).fetchone()

    # Analysis results breakdown
    analysis_stats = conn.execute("""
        SELECT category, COUNT(*) as count
        FROM content_analyses
        WHERE username = ?
        GROUP BY category
        ORDER BY count DESC
    """, (username,)).fetchall()

    # Recent activity (last 7 days)
    recent_activity = conn.execute("""
        SELECT
            DATE(tweet_timestamp) as date,
            COUNT(*) as tweets_count
        FROM tweets
        WHERE username = ? AND tweet_timestamp >= datetime('now', '-7 days')
        GROUP BY DATE(tweet_timestamp)
        ORDER BY date DESC
    """, (username,)).fetchall()

    conn.close()

    return {
        'basic': dict(basic_stats) if basic_stats else {},
        'analysis': [dict(row) for row in analysis_stats],
        'recent_activity': [dict(row) for row in recent_activity]
    }


def get_all_accounts(page: int = 1, per_page: int = 10) -> Dict[str, Any]:
    """Get list of all accounts with basic stats, paginated and sorted by non-general posts."""
    conn = get_db_connection()
    offset = (page - 1) * per_page

    # Get accounts sorted by number of non-general posts, including profile pictures
    accounts_query = """
        SELECT
            t.username,
            COUNT(t.tweet_id) as tweet_count,
            MAX(t.tweet_timestamp) as last_activity,
            COUNT(CASE WHEN ca.category IS NOT NULL AND ca.category != 'general' THEN 1 END) as problematic_posts,
            COUNT(CASE WHEN ca.tweet_id IS NOT NULL THEN 1 END) as analyzed_posts,
            COALESCE(a.profile_pic_url,
                     (SELECT profile_pic_url FROM tweets WHERE username = t.username AND profile_pic_url IS NOT NULL LIMIT 1),
                     '') as profile_pic_url
        FROM tweets t
        LEFT JOIN content_analyses ca ON t.tweet_id = ca.tweet_id
        LEFT JOIN accounts a ON t.username = a.username
        GROUP BY t.username
        ORDER BY problematic_posts DESC, analyzed_posts DESC, tweet_count DESC
        LIMIT ? OFFSET ?
    """

    accounts = conn.execute(accounts_query, (per_page, offset)).fetchall()

    # Get total count for pagination
    total_count = conn.execute("SELECT COUNT(DISTINCT username) FROM tweets").fetchone()[0]

    conn.close()

    return {
        'accounts': [dict(row) for row in accounts],
        'total': total_count,
        'page': page,
        'per_page': per_page,
        'total_pages': math.ceil(total_count / per_page)
    }


def get_user_profile_data(username: str) -> Optional[Dict[str, Any]]:
    """Get user profile data and basic tweet count."""
    conn = get_db_connection()
    try:
        user_query = """
        SELECT
            a.profile_pic_url,
            COUNT(t.tweet_id) as total_tweets
        FROM accounts a
        LEFT JOIN tweets t ON a.username = t.username
        WHERE a.username = ?
        GROUP BY a.username, a.profile_pic_url
        """
        user_result = conn.execute(user_query, [username]).fetchone()
        if user_result:
            return {
                'profile_pic_url': user_result[0],
                'total_tweets': user_result[1]
            }
        return None
    finally:
        conn.close()


def get_user_tweets_data(username: str, page: int, per_page: int,
                        category_filter: Optional[str], post_type_filter: Optional[str],
                        total_tweets_all: int, date_from: Optional[str] = None, date_to: Optional[str] = None,
                        analysis_method_filter: Optional[str] = None) -> Dict[str, Any]:
    """Get paginated tweets data with filtering."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        # Build optimized main query with proper filtering and ordering
        base_query = '''
        SELECT
            t.tweet_url, t.content, t.media_links, t.hashtags, t.mentions,
            t.tweet_timestamp, t.post_type, t.tweet_id,
            ca.category as analysis_category, ca.llm_explanation, ca.analysis_method, ca.analysis_timestamp,
            ca.categories_detected, ca.multimodal_analysis, ca.media_analysis,
            t.is_deleted, t.is_edited, t.rt_original_analyzed,
            t.original_author, t.original_tweet_id, t.reply_to_username,
            CASE
                WHEN ca.category IS NULL OR ca.category = 'general' THEN 1
                ELSE 0
            END as priority_order
        FROM tweets t
        LEFT JOIN content_analyses ca ON t.tweet_id = ca.tweet_id
        WHERE t.username = ?
        '''

        query_params = [username]

        # Add category filter if specified
        if category_filter and category_filter != 'all':
            base_query += ' AND ca.category = ?'
            query_params.append(category_filter)

        # Add post_type filter if specified
        if post_type_filter and post_type_filter != 'all':
            base_query += ' AND t.post_type = ?'
            query_params.append(post_type_filter)

        # Add date range filters if specified
        if date_from:
            base_query += ' AND date(t.tweet_timestamp) >= date(?)'
            query_params.append(date_from)
        if date_to:
            base_query += ' AND date(t.tweet_timestamp) <= date(?)'
            query_params.append(date_to)

        # Add analysis method filter if specified
        if analysis_method_filter and analysis_method_filter != 'all':
            base_query += ' AND ca.analysis_method = ?'
            query_params.append(analysis_method_filter)

        # Get filtered count for pagination (only when filters are applied)
        filters_applied = (
            (category_filter and category_filter != 'all') or
            (post_type_filter and post_type_filter != 'all') or
            date_from or date_to or
            (analysis_method_filter and analysis_method_filter != 'all')
        )

        if filters_applied:
            count_query = f"SELECT COUNT(*) FROM ({base_query}) as subquery"
            total_tweets = cursor.execute(count_query, query_params).fetchone()[0]
        else:
            total_tweets = total_tweets_all

        # Add optimized ordering and pagination
        offset = (page - 1) * per_page
        paginated_query = f"{base_query} ORDER BY priority_order ASC, t.tweet_timestamp DESC LIMIT ? OFFSET ?"
        query_params.extend([per_page, offset])

        cursor.execute(paginated_query, query_params)
        results = cursor.fetchall()

        # Process tweets
        tweets = [process_tweet_row(row) for row in results]

        return {
            'tweets': tweets,
            'page': page,
            'per_page': per_page,
            'total_tweets': total_tweets,
            'total_pages': math.ceil(total_tweets / per_page) if total_tweets > 0 else 1
        }
    finally:
        conn.close()


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
        'reply_to_username': row['reply_to_username']
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
    conn = get_db_connection()
    stats_result = conn.execute("""
        SELECT
            COUNT(CASE WHEN ca.tweet_id IS NOT NULL THEN 1 END) as analyzed_posts,
            COUNT(CASE WHEN ca.category = 'hate_speech' THEN 1 END) as hate_speech_count,
            COUNT(CASE WHEN ca.category = 'disinformation' THEN 1 END) as disinformation_count,
            COUNT(CASE WHEN ca.category = 'conspiracy_theory' THEN 1 END) as conspiracy_count,
            COUNT(CASE WHEN ca.category = 'far_right_bias' THEN 1 END) as far_right_count,
            COUNT(CASE WHEN ca.category = 'call_to_action' THEN 1 END) as call_to_action_count,
            COUNT(CASE WHEN ca.category = 'general' THEN 1 END) as general_count
        FROM tweets t
        LEFT JOIN content_analyses ca ON t.tweet_id = ca.tweet_id
        WHERE t.username = ?
        """, [username]).fetchone()
    conn.close()
    stats_dict = dict(stats_result) if stats_result else {}

    # Build analysis stats from single query result
    total_analyzed = stats_dict.get('analyzed_posts', 0)
    analysis_stats = []

    # Only include categories with counts > 0
    category_mapping = [
        ('hate_speech', 'Hate Speech', stats_dict.get('hate_speech_count', 0)),
        ('disinformation', 'Disinformation', stats_dict.get('disinformation_count', 0)),
        ('conspiracy_theory', 'Conspiracy Theory', stats_dict.get('conspiracy_count', 0)),
        ('far_right_bias', 'Far Right Bias', stats_dict.get('far_right_count', 0)),
        ('call_to_action', 'Call to Action', stats_dict.get('call_to_action_count', 0)),
        ('general', 'General', stats_dict.get('general_count', 0))
    ]

    for category, display_name, count in category_mapping:
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