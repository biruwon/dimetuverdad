"""
Admin routes for the dimetuverdad web application.
Contains administrative functions for system management and data operations.
"""

from flask import Blueprint, render_template, request, flash, redirect, url_for, session, jsonify, Response
import json
import math
import threading
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

from web.utils.decorators import admin_required, rate_limit, handle_db_errors, validate_input, ANALYSIS_CATEGORIES
from web.utils.helpers import (
    get_db_connection, get_tweet_data, reanalyze_tweet, reanalyze_tweet_sync,
    handle_reanalyze_action, handle_refresh_action, handle_refresh_and_reanalyze_action,
    handle_manual_update_action
)
import config

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

# Set up logger for admin blueprint
import logging
admin_bp.logger = logging.getLogger('web.routes.admin')

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

    # Basic stats for dashboard
    stats_row = conn.execute("""
        SELECT
            COUNT(*) AS total_tweets,
            SUM(CASE WHEN ca.post_id IS NOT NULL THEN 1 ELSE 0 END) AS analyzed_tweets,
            SUM(CASE WHEN ca.analysis_method = 'pattern' THEN 1 ELSE 0 END) AS pattern_analyzed,
            SUM(CASE WHEN ca.analysis_method = 'llm' THEN 1 ELSE 0 END) AS llm_analyzed
        FROM tweets t
        LEFT JOIN content_analyses ca ON ca.post_id = t.tweet_id
    """).fetchone()

    stats = {
        'total_tweets': stats_row['total_tweets'] if stats_row and 'total_tweets' in stats_row else 0,
        'analyzed_tweets': stats_row['analyzed_tweets'] if stats_row and 'analyzed_tweets' in stats_row else 0,
        'pattern_analyzed': stats_row['pattern_analyzed'] if stats_row and 'pattern_analyzed' in stats_row else 0,
        'llm_analyzed': stats_row['llm_analyzed'] if stats_row and 'llm_analyzed' in stats_row else 0,
    }

    # Recent analyses (last 10)
    recent_rows = conn.execute("""
        SELECT ca.analysis_timestamp, ca.category, ca.analysis_method, t.username,
               SUBSTR(t.content, 1, 100) AS content_preview
        FROM content_analyses ca
        JOIN tweets t ON t.tweet_id = ca.post_id
        ORDER BY ca.analysis_timestamp DESC
        LIMIT 10
    """).fetchall()

    # Category distribution
    category_rows = conn.execute("""
        SELECT category, COUNT(*) as count
        FROM content_analyses
        GROUP BY category
        ORDER BY count DESC
    """).fetchall()
    conn.close()

    recent_analyses = []
    for r in recent_rows or []:
        recent_analyses.append({
            'analysis_timestamp': r['analysis_timestamp'],
            'category': r['category'],
            'analysis_method': r['analysis_method'],
            'username': r['username'],
            'content_preview': r['content_preview'] if 'content_preview' in r else r['content'][:100] if 'content' in r else ''
        })

    categories = []
    for r in category_rows or []:
        categories.append({
            'category': r['category'],
            'count': r['count']
        })

    return render_template('admin/dashboard.html', stats=stats, categories=categories, recent_analyses=recent_analyses)

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
            user_exists_row = conn.execute("SELECT COUNT(*) FROM tweets WHERE username = ?", (username,)).fetchone()
            user_exists = user_exists_row['COUNT(*)'] if user_exists_row and 'COUNT(*)' in user_exists_row else user_exists_row[0] if user_exists_row else 0
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
                rows = conn.execute("SELECT post_id FROM content_analyses WHERE category = ? LIMIT 20", (category,)).fetchall()
                conn.close()

                if not rows:
                    admin_bp.logger.warning(f"No tweets found for category: {category}")
                    return

                reanalyzed_count = 0

                for r in rows:
                    tweet_id = r['post_id'] if 'post_id' in r else r[0]
                    try:
                        result = reanalyze_tweet_sync(tweet_id)
                        if result:
                            reanalyzed_count += 1
                            admin_bp.logger.info(f"✅ Reanálizado tweet {tweet_id}: {getattr(result, 'category', None)}")
                    except Exception as e:
                        admin_bp.logger.error(f"Failed to reanalyze tweet {tweet_id}: {str(e)}")
                        continue

                admin_bp.logger.info(f"Category reanalysis completed: {reanalyzed_count}/{len(rows)} tweets processed")

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

    # GET request - show edit form (use direct SQL to align with tests)
    conn = get_db_connection()
    row = conn.execute("""
        SELECT 
            t.content,
            t.username,
            t.tweet_timestamp,
            ca.category,
            ca.llm_explanation,
            t.tweet_url,
            t.original_content,
            ca.verification_data,
            ca.verification_confidence
        FROM tweets t
        LEFT JOIN content_analyses ca ON ca.post_id = t.tweet_id
        WHERE t.tweet_id = ?
    """, (tweet_id,)).fetchone()
    conn.close()

    if not row:
        flash('Tweet no encontrado', 'error')
        return redirect(url_for('admin.admin_dashboard'))

    # Support tuple or mapping rows as in tests
    try:
        tweet_dict = {
            'content': row['content'] if 'content' in row else row[0],
            'username': row['username'] if 'username' in row else row[1],
            'tweet_timestamp': row['tweet_timestamp'] if 'tweet_timestamp' in row else row[2],
            'category': row['category'] if 'category' in row else (row[3] if len(row) > 3 else 'general'),
            'llm_explanation': row['llm_explanation'] if 'llm_explanation' in row else (row[4] if len(row) > 4 else ''),
            'tweet_url': row['tweet_url'] if 'tweet_url' in row else (row[5] if len(row) > 5 else ''),
            'original_content': row['original_content'] if 'original_content' in row else (row[6] if len(row) > 6 else ''),
            'verification_data': json.loads(row['verification_data']) if row['verification_data'] and 'verification_data' in row else None,
            'verification_confidence': row['verification_confidence'] if 'verification_confidence' in row else (row[8] if len(row) > 8 else 0.0)
        }
    except Exception:
        # Fallback mapping for strict tuples
        tweet_dict = {
            'content': row[0],
            'username': row[1],
            'tweet_timestamp': row[2],
            'category': row[3] if len(row) > 3 else 'general',
            'llm_explanation': row[4] if len(row) > 4 else '',
            'tweet_url': row[5] if len(row) > 5 else '',
            'original_content': row[6] if len(row) > 6 else '',
            'verification_data': json.loads(row[7]) if row[7] and len(row) > 7 else None,
            'verification_confidence': row[8] if len(row) > 8 else 0.0
        }

    from web.utils.decorators import ANALYSIS_CATEGORIES
    return render_template('admin/edit_analysis.html',
                           tweet=tweet_dict,
                           tweet_id=tweet_id,
                           categories=ANALYSIS_CATEGORIES,
                           referrer=referrer)

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
        analysis_result = reanalyze_tweet_sync(tweet_id)

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

        # 1) Check if category exists
        exists_row = conn.execute("SELECT COUNT(*) FROM content_analyses WHERE category = ?", (category_name,)).fetchone()
        exists_count = exists_row['COUNT(*)'] if exists_row and 'COUNT(*)' in exists_row else exists_row[0] if exists_row else 0
        if not exists_count:
            conn.close()
            flash(f'No se encontraron tweets para la categoría "{category_name}"', 'info')
            return redirect(url_for('admin.admin_dashboard'))

        # 2) Total count
        total_count_row = conn.execute("SELECT COUNT(*) FROM content_analyses WHERE category = ?", (category_name,)).fetchone()
        total_count = total_count_row['COUNT(*)'] if total_count_row and 'COUNT(*)' in total_count_row else total_count_row[0] if total_count_row else 0

        # Pagination
        offset = (page - 1) * per_page

        # 3) Recent analyses for page
        recent_rows = conn.execute("""
            SELECT 
                t.tweet_url,
                t.content,
                t.username,
                t.tweet_timestamp,
                t.tweet_id,
                ca.category,
                ca.llm_explanation,
                ca.analysis_method,
                ca.analysis_timestamp,
                t.is_deleted,
                t.is_edited,
                t.post_type
            FROM content_analyses ca
            JOIN tweets t ON t.tweet_id = ca.post_id
            WHERE ca.category = ?
            ORDER BY ca.analysis_timestamp DESC
            LIMIT ? OFFSET ?
        """, (category_name, per_page, offset)).fetchall()

        # 4) Category stats (llm vs pattern, unique users)
        stats_row = conn.execute("""
            SELECT 
                SUM(CASE WHEN analysis_method='llm' THEN 1 ELSE 0 END) AS llm_count,
                SUM(CASE WHEN analysis_method='pattern' THEN 1 ELSE 0 END) AS pattern_count,
                COUNT(DISTINCT author_username) AS unique_users
            FROM content_analyses
            WHERE category = ?
        """, (category_name,)).fetchone()

        # 5) Top users
        top_user_rows = conn.execute("""
            SELECT author_username as username, COUNT(*) as tweet_count
            FROM content_analyses
            WHERE category = ?
            GROUP BY author_username
            ORDER BY tweet_count DESC
            LIMIT 10
        """, (category_name,)).fetchall()
        conn.close()

        # Build tweets list
        processed_tweets = []
        for r in recent_rows or []:
            processed_tweets.append({
                'tweet_url': r['tweet_url'],
                'content': r['content'],
                'username': r['username'],
                'tweet_timestamp': r['tweet_timestamp'],
                'tweet_id': r['tweet_id'],
                'category': r['category'],
                'llm_explanation': r['llm_explanation'],
                'analysis_method': r['analysis_method'],
                'analysis_timestamp': r['analysis_timestamp'],
                'is_deleted': r['is_deleted'],
                'is_edited': r['is_edited'],
                'post_type': r['post_type']
            })

        # Category stats
        llm_count = stats_row['llm_count'] if stats_row and 'llm_count' in stats_row else 0
        pattern_count = stats_row['pattern_count'] if stats_row and 'pattern_count' in stats_row else 0
        unique_users = stats_row['unique_users'] if stats_row and 'unique_users' in stats_row else 0

        category_stats = {
            'total_tweets': total_count,
            'unique_users': unique_users,
            'llm_analyzed': llm_count,
            'pattern_analyzed': pattern_count
        }

        top_users = []
        for r in top_user_rows or []:
            top_users.append({'username': r['username'], 'tweet_count': r['tweet_count']})

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
                               top_users=top_users)

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

    # Use direct SQL for quick edit
    conn = get_db_connection()
    # Check if analysis exists
    row = conn.execute("SELECT post_id FROM content_analyses WHERE post_id = ?", (tweet_id,)).fetchone()
    if row:
        # Update existing analysis
        conn.execute("UPDATE content_analyses SET category = ?, llm_explanation = ?, analysis_method = 'manual', analysis_timestamp = CURRENT_TIMESTAMP WHERE post_id = ?", (new_category, 'Categoría asignada manualmente por administrador', tweet_id))
        conn.commit()
        success = True
    else:
        # Create new analysis entry
        tweet_row = conn.execute("SELECT username, content, tweet_url FROM tweets WHERE tweet_id = ?", (tweet_id,)).fetchone()
        if tweet_row:
            username = tweet_row['username']
            content = tweet_row['content']
            tweet_url = tweet_row['tweet_url']
            
            conn.execute("""
                INSERT INTO content_analyses (post_id, category, llm_explanation, analysis_method, author_username, post_content, post_url, analysis_timestamp)
                VALUES (?, ?, ?, 'manual', ?, ?, ?, CURRENT_TIMESTAMP)
            """, (tweet_id, new_category, 'Categoría asignada manualmente por administrador', username, content, tweet_url))
            conn.commit()
            success = True
        else:
            success = False
    conn.close()

    if success:
        flash(f'Categoría cambiada a "{new_category}" correctamente', 'success')
    else:
        flash('Error al cambiar la categoría', 'error')

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
        cursor.execute("""
            SELECT 
                ca.post_id,
                ca.author_username,
                ca.category,
                ca.llm_explanation,
                ca.analysis_method,
                ca.analysis_timestamp,
                t.content as tweet_content,
                t.tweet_url,
                t.tweet_timestamp
            FROM content_analyses ca
            JOIN tweets t ON t.tweet_id = ca.post_id
            ORDER BY ca.analysis_timestamp DESC
        """)
        rows = cursor.fetchall()
        conn.close()

        # Create CSV response
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow([
            'Post ID', 'Author Username', 'Category', 'LLM Explanation',
            'Analysis Method', 'Analysis Timestamp', 'Post Content',
            'Post URL', 'Post Timestamp'
        ])

        # Write data
        for r in rows:
            try:
                writer.writerow([
                    r['post_id'], r['author_username'], r['category'], r['llm_explanation'],
                    r['analysis_method'], r['analysis_timestamp'], r['tweet_content'], r['tweet_url'], r['tweet_timestamp']
                ])
            except Exception:
                # Support MockRow/tuple
                writer.writerow([r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]])

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
        cursor.execute("""
            SELECT 
                ca.post_id,
                ca.author_username,
                ca.category,
                ca.llm_explanation,
                ca.analysis_method,
                ca.analysis_timestamp,
                t.content as post_content,
                t.tweet_url as post_url,
                t.tweet_timestamp as post_timestamp,
                ca.categories_detected
            FROM content_analyses ca
            JOIN tweets t ON t.tweet_id = ca.post_id
            ORDER BY ca.analysis_timestamp DESC
        """)
        rows = cursor.fetchall()
        conn.close()

        # Convert to JSON-serializable format
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_records': len(rows),
            'data': []
        }

        for r in rows:
            try:
                record = {
                    'post_id': r['post_id'],
                    'author_username': r['author_username'],
                    'category': r['category'],
                    'llm_explanation': r['llm_explanation'],
                    'analysis_method': r['analysis_method'],
                    'analysis_timestamp': r['analysis_timestamp'],
                    'post_content': r['post_content'],
                    'post_url': r['post_url'],
                    'post_timestamp': r['post_timestamp'],
                    'categories_detected': r['categories_detected']
                }
            except Exception:
                record = {
                    'post_id': r[0],
                    'author_username': r[1],
                    'category': r[2],
                    'llm_explanation': r[3],
                    'analysis_method': r[4],
                    'analysis_timestamp': r[5],
                    'post_content': r[6],
                    'post_url': r[7],
                    'post_timestamp': r[8],
                    'categories_detected': r[9] if len(r) > 9 else None
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