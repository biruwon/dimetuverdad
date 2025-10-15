"""
Main routes for the dimetuverdad web application.
Contains dashboard and user profile pages.
"""

from flask import Blueprint, render_template, request, current_app
from flask_caching import Cache
from web.utils.decorators import rate_limit, handle_db_errors, validate_input
from web.utils.helpers import (
    get_all_accounts, get_user_profile_data,
    get_user_tweets_data, get_user_analysis_stats,
    prepare_user_page_template_data
)
import config

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
@rate_limit(**config.get_rate_limit('main_dashboard'))
def index() -> str:
    """Main dashboard with account overview (focus on analysis, not engagement)."""
    page = request.args.get('page', 1, type=int)
    category_filter = request.args.get('category', None)

    accounts_data = get_all_accounts(page=page, per_page=config.get_pagination_limit('accounts'))

    # Filter accounts by category if specified
    if category_filter and category_filter != 'all':
        filtered_accounts = []
        from utils.database import get_db_connection_context
        with get_db_connection_context() as conn:
            for account in accounts_data['accounts']:
                # Check if account has posts in this category
                has_category = conn.execute("""
                    SELECT COUNT(*) FROM content_analyses ca
                    JOIN tweets t ON ca.post_id = t.tweet_id
                    WHERE t.username = ? AND ca.category = ?
                """, (account['username'], category_filter)).fetchone()[0]

                if has_category > 0:
                    filtered_accounts.append(account)
        accounts_data['accounts'] = filtered_accounts

    # Overall statistics - simplified to just accounts and analyzed posts (with caching)
    cache = getattr(current_app, 'cache', None)
    if not cache:
        # Fallback: create cache instance if not available
        cache = Cache(current_app, config={
            'CACHE_TYPE': current_app.config.get('CACHE_TYPE', 'null'),
            'CACHE_DEFAULT_TIMEOUT': current_app.config.get('CACHE_DEFAULT_TIMEOUT', 300)
        })

    @cache.memoize(timeout=600)
    def get_overall_stats_cached():
        from utils.database import get_db_connection_context
        with get_db_connection_context() as conn:
            overall_stats = conn.execute("""
            SELECT
                COUNT(DISTINCT t.username) as total_accounts,
                COUNT(CASE WHEN ca.post_id IS NOT NULL THEN 1 END) as analyzed_tweets
            FROM tweets t
            LEFT JOIN content_analyses ca ON t.tweet_id = ca.post_id
            """).fetchone()
        return dict(overall_stats) if overall_stats else {}

    @cache.memoize(timeout=600)
    def get_analysis_distribution_cached():
        from utils.database import get_db_connection_context
        with get_db_connection_context() as conn:
            analysis_distribution = conn.execute("""
            SELECT
                category,
                COUNT(*) as count,
                COUNT(*) * 100.0 / (SELECT COUNT(*) FROM content_analyses) as percentage
            FROM content_analyses
            GROUP BY category
            ORDER BY count DESC
            """).fetchall()
        return [dict(row) for row in analysis_distribution]

    overall_stats = get_overall_stats_cached()
    analysis_distribution = get_analysis_distribution_cached()

    return render_template('index.html',
                         accounts_data=accounts_data,
                         overall_stats=overall_stats,
                         analysis_distribution=analysis_distribution,
                         current_category=category_filter)

@main_bp.route('/user/<username>')
@handle_db_errors
@validate_input('username')
@rate_limit(**config.get_rate_limit('user_pages'))
def user_page(username: str) -> str:
    """User profile page with tweets and analysis focus."""
    page = request.args.get('page', 1, type=int)
    category_filter = request.args.get('category', None)
    post_type_filter = request.args.get('post_type', None)
    date_from = request.args.get('date_from', None)
    date_to = request.args.get('date_to', None)
    per_page = config.get_pagination_limit('tweets')

    try:
        # Get user profile data
        user_profile_data = get_user_profile_data(username)
        if not user_profile_data:
            return render_template('error.html',
                                 error_code=404,
                                 error_title="Usuario no encontrado",
                                 error_message="El usuario solicitado no existe en la base de datos.",
                                 error_icon="fas fa-user-times",
                                 show_back_button=True), 404

        user_profile_pic = user_profile_data['profile_pic_url']
        total_tweets_all = user_profile_data['total_tweets']

        # Build and execute tweets query
        tweets_data = get_user_tweets_data(username, page, per_page, category_filter, post_type_filter, total_tweets_all, date_from, date_to)

        # Get user statistics
        user_stats = get_user_analysis_stats(username)

        # Prepare final data for template
        template_data = prepare_user_page_template_data(
            username, tweets_data, user_stats, total_tweets_all,
            page, per_page, category_filter, user_profile_pic, date_from, date_to
        )

        return render_template('user.html', **template_data)

    except Exception as e:
        current_app.logger.error(f"Error in user_page for {username}: {str(e)}")
        return render_template('error.html',
                             error_code=500,
                             error_title="Error interno del servidor",
                             error_message="Ocurrió un error al cargar los detalles del usuario.",
                             error_icon="fas fa-exclamation-triangle",
                             show_back_button=True), 500