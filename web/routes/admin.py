"""
Admin routes for the dimetuverdad web application.
Contains administrative functions for system management and data operations.
"""

from flask import Blueprint, render_template, request, flash, redirect, url_for, session
import json
import math
import threading
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

from web.utils.decorators import admin_required, rate_limit, handle_db_errors, validate_input
from web.utils.helpers import (
    get_db_connection, get_tweet_data, reanalyze_tweet,
    handle_reanalyze_action, handle_refresh_action, handle_refresh_and_reanalyze_action,
    handle_manual_update_action, get_tweet_display_data
)
import config

admin_bp = Blueprint('admin', __name__)

from flask import Blueprint, render_template, request, flash, redirect, url_for, session, jsonify, Response
import json
import math
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

from web.utils.decorators import admin_required, rate_limit, handle_db_errors, validate_input, ANALYSIS_CATEGORIES
from utils import database
import config

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

@admin_bp.route('/login', methods=['GET', 'POST'])
@rate_limit(**config.get_rate_limit('admin_login'))
def admin_login() -> str:
    """Simple admin login page."""
    if request.method == 'POST':
        token = request.form.get('token')
        print(f"DEBUG: Received token: '{token}'")
        print(f"DEBUG: Expected token: '{config.ADMIN_TOKEN}'")
        print(f"DEBUG: Tokens match: {token == config.ADMIN_TOKEN}")
        if token == config.ADMIN_TOKEN:
            session['admin_authenticated'] = True
            flash('Acceso administrativo concedido', 'success')
            return redirect(url_for('admin.admin_dashboard'))
        else:
            flash('Token administrativo incorrecto', 'error')

    return render_template('admin/login.html')

@admin_bp.route('/logout')
def admin_logout() -> str:
    """Admin logout."""
    session.pop('admin_authenticated', None)
    flash('Sesión administrativa cerrada', 'info')
    return redirect(url_for('main.index'))

@admin_bp.route('/')
@admin_required
def admin_dashboard() -> str:
    """Admin dashboard with reanalysis and management options."""
    conn = get_db_connection()

    # Get analysis statistics
    stats = conn.execute("""
        SELECT
            COUNT(DISTINCT t.tweet_id) as total_tweets,
            COUNT(CASE WHEN ca.tweet_id IS NOT NULL THEN 1 END) as analyzed_tweets,
            COUNT(CASE WHEN ca.analysis_method = 'pattern' THEN 1 END) as pattern_analyzed,
            COUNT(CASE WHEN ca.analysis_method = 'llm' THEN 1 END) as llm_analyzed
        FROM tweets t
        LEFT JOIN content_analyses ca ON t.tweet_id = ca.tweet_id
    """).fetchone()

    # Get category distribution
    categories = conn.execute("""
        SELECT category, COUNT(*) as count
        FROM content_analyses
        GROUP BY category
        ORDER BY count DESC
    """).fetchall()

    # Get recent analysis activity
    recent_analyses = conn.execute("""
        SELECT
            ca.analysis_timestamp,
            ca.category,
            ca.analysis_method,
            t.username,
            SUBSTR(t.content, 1, 100) as content_preview
        FROM content_analyses ca
        JOIN tweets t ON ca.tweet_id = t.tweet_id
        ORDER BY ca.analysis_timestamp DESC
        LIMIT 10
    """).fetchall()

    conn.close()

    return render_template('admin/dashboard.html',
                         stats=dict(stats) if stats else {},
                         categories=[dict(row) for row in categories],
                         recent_analyses=[dict(row) for row in recent_analyses])

@admin_bp.route('/fetch', methods=['POST'])
@admin_required
@rate_limit(**config.get_rate_limit('admin_actions'))
@handle_db_errors
@validate_input('username')
def admin_fetch() -> str:
    """Fetch tweets from a user, optionally with analysis."""
    username = request.form.get('username')
    action = request.form.get('action', 'fetch_and_analyze')  # Default to fetch and analyze

    if not username:
        flash('Nombre de usuario requerido para fetch', 'error')
        return redirect(url_for('admin.admin_dashboard'))

    import subprocess
    import threading
    from pathlib import Path

    base_dir = Path(__file__).parent.parent.parent

    def run_user_fetch():
        try:
            # Check if user exists in database
            conn = get_db_connection()
            user_exists = conn.execute("""
                SELECT COUNT(*) FROM tweets WHERE username = ?
            """, (username,)).fetchone()[0] > 0
            conn.close()

            # Choose fetch strategy based on user existence
            if user_exists:
                # User exists, fetch latest content
                cmd = ["./run_in_venv.sh", "fetch", "--user", username, "--latest"]
                strategy = "latest content"
            else:
                # User doesn't exist, fetch all history
                cmd = ["./run_in_venv.sh", "fetch", "--refetch-all", username]
                strategy = "complete history"

            result = subprocess.run(cmd, cwd=base_dir, check=True, timeout=config.get_command_timeout('fetch'))  # 10 minute timeout for fetch
            admin_bp.logger.info(f"User fetch completed for @{username} ({strategy})")

            # Only trigger analysis if action is fetch_and_analyze
            if action == 'fetch_and_analyze':
                analysis_cmd = ["./run_in_venv.sh", "analyze-twitter", "--username", username]
                analysis_result = subprocess.run(analysis_cmd, cwd=base_dir, check=True, timeout=config.get_command_timeout('analyze'))
                admin_bp.logger.info(f"Analysis completed for @{username} after fetch")

        except subprocess.TimeoutExpired:
            admin_bp.logger.error(f"Fetch/analysis timed out for @{username}")
        except subprocess.CalledProcessError as e:
            admin_bp.logger.error(f"Fetch/analysis failed for @{username}: {e}")
        except Exception as e:
            admin_bp.logger.error(f"Unexpected error in fetch/analysis for @{username}: {str(e)}")

    thread = threading.Thread(target=run_user_fetch, daemon=True)
    thread.start()

    if action == 'fetch_only':
        flash(f'Fetch de usuario "@{username}" iniciado (solo datos)', 'success')
    else:
        flash(f'Fetch de usuario "@{username}" iniciado (con análisis automático)', 'success')

    # Return loading page instead of immediate redirect
    return render_template('loading.html',
                         message=f"Procesando datos de @{username}...",
                         redirect_url=url_for('admin.admin_dashboard'))

@admin_bp.route('/reanalyze', methods=['POST'])
@admin_required
@rate_limit(**config.get_rate_limit('admin_actions'))
@handle_db_errors
@validate_input('action')
def admin_reanalyze() -> str:
    """Trigger reanalysis of tweets."""
    action = request.form.get('action')

    if action == 'category':
        category = request.form.get('category')
        if not category:
            flash('Categoría requerida para reanálisis', 'error')
            return redirect(url_for('admin.admin_dashboard'))

        # Reanalyze tweets from specific category using direct analysis
        import threading

        def reanalyze_category():
            try:
                # Get tweets from specific category
                conn = get_db_connection()
                tweets = conn.execute("""
                    SELECT t.tweet_id, t.content, t.username
                    FROM tweets t
                    JOIN content_analyses ca ON t.tweet_id = ca.tweet_id
                    WHERE ca.category = ?
                    LIMIT 20
                """, (category,)).fetchall()
                conn.close()

                if not tweets:
                    admin_bp.logger.warning(f"No tweets found for category: {category}")
                    return

                reanalyzed_count = 0

                for tweet in tweets:
                    try:
                        result = reanalyze_tweet(tweet[0])
                        if result:
                            reanalyzed_count += 1
                            admin_bp.logger.info(f"✅ Reanálizado tweet {tweet[0]} de @{tweet[2]}: {result.category}")
                    except Exception as e:
                        admin_bp.logger.error(f"Failed to reanalyze tweet {tweet[0]}: {str(e)}")
                        continue

                admin_bp.logger.info(f"Category reanalysis completed: {reanalyzed_count}/{len(tweets)} tweets processed")

            except Exception as e:
                admin_bp.logger.error(f"Error in category reanalysis: {str(e)}")

        thread = threading.Thread(target=reanalyze_category, daemon=True)
        thread.start()
        flash(f'Reanálisis de categoría "{category}" iniciado (máximo 20 tweets)', 'success')

    elif action == 'user':
        username = request.form.get('username')
        if not username:
            flash('Nombre de usuario requerido para reanálisis', 'error')
            return redirect(url_for('admin.admin_dashboard'))

        import subprocess
        import threading
        from pathlib import Path

        base_dir = Path(__file__).parent.parent.parent

        def run_user_analysis():
            try:
                cmd = ["./run_in_venv.sh", "analyze-twitter", "--username", username, "--force-reanalyze"]
                result = subprocess.run(cmd, cwd=base_dir, check=True, timeout=config.get_command_timeout('user_analysis'))
                admin_bp.logger.info(f"User analysis completed for @{username}")
            except subprocess.TimeoutExpired:
                admin_bp.logger.error(f"User analysis timed out for @{username}")
            except subprocess.CalledProcessError as e:
                admin_bp.logger.error(f"User analysis failed for @{username}: {e}")
            except Exception as e:
                admin_bp.logger.error(f"Unexpected error in user analysis for @{username}: {str(e)}")

        thread = threading.Thread(target=run_user_analysis, daemon=True)
        thread.start()
        flash(f'Reanálisis de usuario "@{username}" iniciado', 'success')

    else:
        flash('Acción no válida', 'error')
        return redirect(url_for('admin.admin_dashboard'))

    # Return loading page instead of immediate redirect
    return render_template('loading.html',
                         message="Procesando reanálisis...",
                         redirect_url=url_for('admin.admin_dashboard'))

@admin_bp.route('/edit-analysis/<tweet_id>', methods=['GET', 'POST'])
@admin_required
def admin_edit_analysis(tweet_id: str) -> str:
    """Edit analysis results for a specific tweet."""
    conn = get_db_connection()

    # Get referrer (user page) for proper redirect
    referrer = request.args.get('from') or request.referrer

    if request.method == 'POST':
        action = request.form.get('action', 'update')
        new_category = request.form.get('category')
        new_explanation = request.form.get('explanation')

        try:
            if action == 'reanalyze':
                handle_reanalyze_action(tweet_id, referrer)
            elif action == 'refresh':
                handle_refresh_action(tweet_id, referrer)
            elif action == 'refresh_and_reanalyze':
                handle_refresh_and_reanalyze_action(tweet_id, referrer)
            else:
                # Manual update
                handle_manual_update_action(tweet_id, new_category, new_explanation)

            # Redirect back to user view if possible, otherwise admin dashboard
            if referrer and '/user/' in referrer:
                return redirect(referrer)
            else:
                return redirect(url_for('admin.admin_dashboard'))

        except Exception as e:
            admin_bp.logger.error(f"Error in admin_edit_analysis: {str(e)}")
            admin_bp.logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            admin_bp.logger.error(f"Traceback: {traceback.format_exc()}")
            flash('Ocurrió un error al procesar la solicitud. Inténtalo de nuevo.', 'error')
            return redirect(referrer or url_for('admin.admin_dashboard'))

    # GET request - show edit form
    result = get_tweet_display_data(tweet_id, referrer)
    if isinstance(result, tuple):
        tweet_dict, categories = result
        return render_template('admin/edit_analysis.html',
                             tweet=tweet_dict,
                             tweet_id=tweet_id,
                             categories=categories,
                             referrer=referrer)
    else:
        # It's a redirect response
        return result

@admin_bp.route('/reanalyze-single/<tweet_id>', methods=['POST'])
@admin_required
def admin_reanalyze_single(tweet_id: str) -> str:
    """Reanalyze a single tweet using the analysis pipeline directly."""
    try:
        tweet_data = get_tweet_data(tweet_id)

        if not tweet_data:
            flash('Tweet no encontrado', 'error')
            return redirect(request.referrer or url_for('main.index'))

        print(f"🔄 Reanalizando tweet {tweet_id} de @{tweet_data.get('username', 'unknown')}")

        # Reanalyze the content
        analysis_result = reanalyze_tweet(tweet_id)

        if analysis_result and hasattr(analysis_result, 'category') and analysis_result.category:
            flash(f'Tweet reanálizado correctamente. Nueva categoría: {analysis_result.category}', 'success')
        else:
            flash('Tweet reanálizado pero no se pudo determinar la categoría.', 'warning')

    except Exception as e:
        print(f"Error durante reanálisis: {e}")
        admin_bp.logger.error(f"Error in admin_reanalyze_single for {tweet_id}: {str(e)}")
        flash('El reanálisis falló. Inténtalo de nuevo más tarde.', 'error')

    return redirect(request.referrer or url_for('main.index'))

@admin_bp.route('/category/<category_name>')
@admin_required
def admin_view_category(category_name: str) -> str:
    """View all tweets from a specific category (admin only)."""
    page = request.args.get('page', 1, type=int)
    per_page = config.get_pagination_limit('admin_category')

    try:
        conn = get_db_connection()

        # Debug: Check if category exists
        category_check = conn.execute("""
            SELECT COUNT(*) FROM content_analyses WHERE category = ?
        """, (category_name,)).fetchone()

        if not category_check or category_check[0] == 0:
            conn.close()
            flash(f'No se encontraron tweets para la categoría "{category_name}"', 'info')
            return redirect(url_for('admin.admin_dashboard'))

        # Get total count for pagination
        total_count_result = conn.execute("""
            SELECT COUNT(*)
            FROM content_analyses ca
            JOIN tweets t ON ca.tweet_id = t.tweet_id
            WHERE ca.category = ?
        """, (category_name,)).fetchone()

        total_count = 0
        if total_count_result and len(total_count_result) > 0 and total_count_result[0] is not None:
            total_count = total_count_result[0]

        if total_count == 0:
            conn.close()
            flash(f'No hay tweets disponibles para la categoría "{category_name}"', 'info')
            return redirect(url_for('admin.admin_dashboard'))

        # Get tweets from this category across all users
        offset = (page - 1) * per_page
        tweets_query = """
            SELECT
                t.tweet_url, t.content, t.username, t.tweet_timestamp, t.tweet_id,
                ca.category, ca.llm_explanation, ca.analysis_method, ca.analysis_timestamp,
                t.is_deleted, t.is_edited, t.post_type
            FROM content_analyses ca
            JOIN tweets t ON ca.tweet_id = t.tweet_id
            WHERE ca.category = ?
            ORDER BY ca.analysis_timestamp DESC
            LIMIT ? OFFSET ?
        """

        tweets = conn.execute(tweets_query, (category_name, per_page, offset)).fetchall()

        # Get category statistics with comprehensive null safety
        category_stats_query = """
            SELECT
                COUNT(*) as total_tweets,
                COUNT(DISTINCT t.username) as unique_users,
                COUNT(CASE WHEN ca.analysis_method = 'llm' THEN 1 END) as llm_analyzed,
                COUNT(CASE WHEN ca.analysis_method = 'pattern' THEN 1 END) as pattern_analyzed
            FROM content_analyses ca
            JOIN tweets t ON ca.tweet_id = t.tweet_id
            WHERE ca.category = ?
        """

        category_stats_result = conn.execute(category_stats_query, (category_name,)).fetchone()

        # Build category stats with comprehensive null checking
        category_stats = {
            'total_tweets': 0,
            'unique_users': 0,
            'llm_analyzed': 0,
            'pattern_analyzed': 0
        }

        if category_stats_result and len(category_stats_result) >= 4:
            category_stats = {
                'total_tweets': category_stats_result[0] if category_stats_result[0] is not None else 0,
                'unique_users': category_stats_result[1] if category_stats_result[1] is not None else 0,
                'llm_analyzed': category_stats_result[2] if category_stats_result[2] is not None else 0,
                'pattern_analyzed': category_stats_result[3] if category_stats_result[3] is not None else 0
            }

        # Get top users in this category
        top_users_query = """
            SELECT t.username, COUNT(*) as tweet_count
            FROM content_analyses ca
            JOIN tweets t ON ca.tweet_id = t.tweet_id
            WHERE ca.category = ?
            GROUP BY t.username
            ORDER BY tweet_count DESC
            LIMIT 10
        """

        top_users_result = conn.execute(top_users_query, (category_name,)).fetchall()
        top_users = top_users_result if top_users_result else []

        conn.close()

        # Process tweets for display with null safety
        processed_tweets = []
        if tweets:
            for row in tweets:
                if len(row) >= 12:  # Ensure row has enough columns
                    tweet = {
                        'tweet_url': row['tweet_url'] or '',
                        'content': row['content'] or '',
                        'username': row['username'] or '',
                        'tweet_timestamp': row['tweet_timestamp'] or '',
                        'tweet_id': row['tweet_id'] or '',
                        'category': row['category'] or category_name,
                        'llm_explanation': row['llm_explanation'] or '',
                        'analysis_method': row['analysis_method'] or 'unknown',
                        'analysis_timestamp': row['analysis_timestamp'] or '',
                        'is_deleted': bool(row['is_deleted']) if row['is_deleted'] is not None else False,
                        'is_edited': bool(row['is_edited']) if row['is_edited'] is not None else False,
                        'post_type': row['post_type'] or 'original'
                    }
                    processed_tweets.append(tweet)

        pagination = {
            'page': page,
            'per_page': per_page,
            'total': total_count,
            'total_pages': math.ceil(total_count / per_page) if total_count > 0 else 1
        }

        return render_template('admin/category_view.html',
                             category_name=category_name,
                             tweets=processed_tweets,
                             pagination=pagination,
                             category_stats=category_stats,
                             top_users=[dict(row) for row in top_users] if top_users else [])

    except Exception as e:
        admin_bp.logger.error(f"Error in admin_view_category for {category_name}: {str(e)}")
        admin_bp.logger.error(f"Error details: {type(e).__name__}: {e}")
        import traceback
        admin_bp.logger.error(f"Traceback: {traceback.format_exc()}")
        flash('No se pudo cargar la información de la categoría. Inténtalo de nuevo.', 'error')
        return redirect(url_for('admin.admin_dashboard'))

@admin_bp.route('/user-category/<username>/<category>')
@admin_required
def admin_view_user_category(username: str, category: str) -> str:
    """View specific user's tweets in a specific category (admin only)."""
    return redirect(url_for('main.user_page', username=username, category=category))

@admin_bp.route('/quick-edit-category/<tweet_id>', methods=['POST'])
@admin_required
def admin_quick_edit_category(tweet_id: str) -> str:
    """Quickly change the category of a tweet."""
    new_category = request.form.get('category')

    if not new_category:
        flash('Categoría requerida', 'error')
        return redirect(request.referrer or url_for('main.index'))

    conn = get_db_connection()

    # Check if analysis exists, if not create one
    existing = conn.execute("""
        SELECT tweet_id FROM content_analyses WHERE tweet_id = ?
    """, (tweet_id,)).fetchone()

    if existing:
        # Update existing analysis
        conn.execute("""
            UPDATE content_analyses
            SET category = ?, analysis_timestamp = datetime('now')
            WHERE tweet_id = ?
        """, (new_category, tweet_id))
    else:
        # Create new analysis entry
        conn.execute("""
            INSERT INTO content_analyses (tweet_id, category, llm_explanation, analysis_method, analysis_timestamp)
            SELECT ?, ?, 'Categoría asignada manualmente por administrador', 'manual', datetime('now')
        """, (tweet_id, new_category))

    conn.commit()
    conn.close()

    flash(f'Categoría cambiada a "{new_category}" correctamente', 'success')
    return redirect(request.referrer or url_for('main.index'))

@admin_bp.route('/export/csv')
@admin_required
@rate_limit(**config.get_rate_limit('export_endpoints'))
@handle_db_errors
def export_csv() -> str:
    """Export analysis results as CSV."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get all analysis results with tweet data
        cursor.execute('''
            SELECT
                ca.tweet_id,
                ca.username,
                ca.category,
                ca.llm_explanation,
                ca.analysis_method,
                ca.analysis_timestamp,
                t.content as tweet_content,
                t.tweet_url,
                t.tweet_timestamp
            FROM content_analyses ca
            JOIN tweets t ON ca.tweet_id = t.tweet_id
            ORDER BY ca.analysis_timestamp DESC
        ''')

        results = cursor.fetchall()
        conn.close()

        # Create CSV response
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow([
            'Tweet ID', 'Username', 'Category', 'LLM Explanation',
            'Analysis Method', 'Analysis Timestamp', 'Tweet Content',
            'Tweet URL', 'Tweet Timestamp'
        ])

        # Write data
        for row in results:
            writer.writerow([
                row['tweet_id'],
                row['username'],
                row['category'],
                row['llm_explanation'],
                row['analysis_method'],
                row['analysis_timestamp'],
                row['tweet_content'],
                row['tweet_url'],
                row['tweet_timestamp']
            ])

        output.seek(0)
        csv_data = output.getvalue()
        output.close()

        # Create response
        response = Response(
            csv_data,
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename=dimetuverdad_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            }
        )
        return response

    except Exception as e:
        admin_bp.logger.error(f"Error exporting CSV: {str(e)}")
        flash(f'Error exporting CSV: {str(e)}', 'error')
        return redirect(url_for('admin.admin_dashboard'))

@admin_bp.route('/export/json')
@admin_required
@rate_limit(**config.get_rate_limit('export_endpoints'))
@handle_db_errors
def export_json() -> str:
    """Export analysis results as JSON."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get all analysis results with tweet data
        cursor.execute('''
            SELECT
                ca.tweet_id,
                ca.username,
                ca.category,
                ca.llm_explanation,
                ca.analysis_method,
                ca.analysis_timestamp,
                t.content as tweet_content,
                t.tweet_url,
                t.tweet_timestamp,
                ca.categories_detected
            FROM content_analyses ca
            JOIN tweets t ON ca.tweet_id = t.tweet_id
            ORDER BY ca.analysis_timestamp DESC
        ''')

        results = cursor.fetchall()
        conn.close()

        # Convert to JSON-serializable format
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_records': len(results),
            'data': []
        }

        for row in results:
            record = {
                'tweet_id': row['tweet_id'],
                'username': row['username'],
                'category': row['category'],
                'llm_explanation': row['llm_explanation'],
                'analysis_method': row['analysis_method'],
                'analysis_timestamp': row['analysis_timestamp'],
                'tweet_content': row['tweet_content'],
                'tweet_url': row['tweet_url'],
                'tweet_timestamp': row['tweet_timestamp'],
                'categories_detected': row['categories_detected']
            }

            export_data['data'].append(record)

        # Create response
        json_data = json.dumps(export_data, indent=2, ensure_ascii=False)
        response = Response(
            json_data,
            mimetype='application/json',
            headers={
                'Content-Disposition': f'attachment; filename=dimetuverdad_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            }
        )
        return response

    except Exception as e:
        admin_bp.logger.error(f"Error exporting JSON: {str(e)}")
        flash(f'Error exporting JSON: {str(e)}', 'error')
        return redirect(url_for('admin.admin_dashboard'))