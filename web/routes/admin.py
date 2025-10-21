"""
Admin routes for the dimetuverdad web application.
Contains administrative functions for system management and data operations.
"""

from flask import Blueprint, render_template, request, flash, redirect, url_for, session, jsonify, Response
import json
import math
import threading
import subprocess
import traceback
import csv
import io
import asyncio
import concurrent.futures
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

from analyzer.external_analyzer import ExternalAnalyzer
from utils.database import get_db_connection_context
from web.utils.decorators import admin_required, rate_limit, handle_db_errors, validate_input, ANALYSIS_CATEGORIES
from web.utils.helpers import (
    get_tweet_data, reanalyze_tweet_sync,
    handle_reanalyze_action, handle_refresh_action, handle_refresh_and_reanalyze_action,
    handle_manual_update_action
)
import config

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

# Set up logger for admin blueprint
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
    from utils.database import get_db_connection_context
    with get_db_connection_context() as conn:
        # Basic stats for dashboard
        stats_row = conn.execute("""
            SELECT
                COUNT(*) AS total_tweets,
                SUM(CASE WHEN ca.post_id IS NOT NULL THEN 1 ELSE 0 END) AS analyzed_tweets,
                SUM(CASE WHEN ca.external_analysis_used = 1 THEN 1 ELSE 0 END) AS external_analyzed,
                SUM(CASE WHEN ca.analysis_stages LIKE '%local_llm%' THEN 1 ELSE 0 END) AS local_llm_analyzed
            FROM tweets t
            LEFT JOIN content_analyses ca ON ca.post_id = t.tweet_id
        """).fetchone()

        stats = {
            'total_tweets': stats_row['total_tweets'] if stats_row else 0,
            'analyzed_tweets': stats_row['analyzed_tweets'] if stats_row else 0,
            'external_analyzed': stats_row['external_analyzed'] if stats_row else 0,
            'local_llm_analyzed': stats_row['local_llm_analyzed'] if stats_row else 0,
        }

        # Recent analyses (last 10)
        recent_rows = conn.execute("""
            SELECT ca.analysis_timestamp, ca.category, ca.analysis_stages, ca.external_analysis_used, t.username,
                   SUBSTR(t.content, 1, 100) AS content_preview
            FROM content_analyses ca
            JOIN tweets t ON t.tweet_id = ca.post_id
            ORDER BY ca.analysis_timestamp DESC
            LIMIT 10
        """).fetchall()

        # Recent feedback submissions (last 5)
        feedback_rows = conn.execute("""
            SELECT uf.submitted_at, uf.feedback_type, uf.original_category, uf.corrected_category,
                   uf.user_comment, uf.post_id, t.username, t.content
            FROM user_feedback uf
            LEFT JOIN tweets t ON t.tweet_id = uf.post_id
            ORDER BY uf.submitted_at DESC
            LIMIT 5
        """).fetchall()

        # Recent fetch operations (last 5 users with recent activity)
        fetch_rows = conn.execute("""
            SELECT username, COUNT(*) as tweet_count, MAX(scraped_at) as latest_scraped
            FROM tweets
            WHERE scraped_at >= datetime('now', '-7 days')
            GROUP BY username
            ORDER BY latest_scraped DESC
            LIMIT 5
        """).fetchall()

        # Category distribution
        category_rows = conn.execute("""
            SELECT category, COUNT(*) as count
            FROM content_analyses
            GROUP BY category
            ORDER BY count DESC
        """).fetchall()

    recent_analyses = []
    for r in recent_rows or []:
        # Determine display method from analysis stages
        stages = r['analysis_stages'] or 'pattern'
        if r['external_analysis_used']:
            display_method = f"{stages} (external)"
        else:
            display_method = stages
            
        recent_analyses.append({
            'analysis_timestamp': r['analysis_timestamp'],
            'category': r['category'],
            'analysis_method': display_method,
            'username': r['username'],
            'content_preview': r['content_preview'] if 'content_preview' in r else r['content'][:100] if 'content' in r else '',
            'activity_type': 'analysis'
        })

    # Add feedback submissions to recent activity
    for r in feedback_rows or []:
        recent_analyses.append({
            'analysis_timestamp': r['submitted_at'],
            'category': f"{r['original_category']} → {r['corrected_category']}",
            'analysis_method': f"feedback ({r['feedback_type']})",
            'username': r['username'] or 'unknown',
            'content_preview': r['user_comment'][:100] if r['user_comment'] else (r['content'][:100] if r['content'] else ''),
            'activity_type': 'feedback',
            'feedback_type': r['feedback_type'],
            'original_category': r['original_category'],
            'corrected_category': r['corrected_category'],
            'post_id': r['post_id']
        })

    # Add fetch operations to recent activity
    for r in fetch_rows or []:
        recent_analyses.append({
            'analysis_timestamp': r['latest_scraped'],
            'category': f"Fetched {r['tweet_count']} tweets",
            'analysis_method': 'fetch',
            'username': r['username'],
            'content_preview': f"Content collection from @{r['username']}",
            'activity_type': 'fetch',
            'tweet_count': r['tweet_count']
        })

    # Sort all activities by timestamp and take the most recent 15
    recent_analyses.sort(key=lambda x: x['analysis_timestamp'], reverse=True)
    recent_analyses = recent_analyses[:15]

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
    max_tweets = request.form.get('max')
    action = request.form.get('action', 'fetch_and_analyze')  # Default to fetch and analyze

    if not username:
        flash('Nombre de usuario requerido para fetch', 'error')
        return redirect(url_for('admin.admin_dashboard'))

    base_dir = Path(__file__).parent.parent.parent

    def run_user_fetch():
        try:
            # Check if user exists in database
            from utils.database import get_db_connection_context
            with get_db_connection_context() as conn:
                user_exists_row = conn.execute("SELECT COUNT(*) AS cnt FROM tweets WHERE username = ?", (username,)).fetchone()
                user_exists = user_exists_row['cnt'] if user_exists_row else 0

            # Choose fetch strategy based on user existence
            if user_exists:
                # User exists, fetch latest content
                cmd = ["./run_in_venv.sh", "fetch", "--user", username, "--latest"]
                strategy = "latest content"
            else:
                # User doesn't exist, fetch all history
                cmd = ["./run_in_venv.sh", "fetch", "--refetch-all", username]
                strategy = "complete history"

            # Add max parameter if provided
            if max_tweets and max_tweets.strip():
                try:
                    max_val = int(max_tweets.strip())
                    if max_val > 0:
                        cmd.extend(["--max", str(max_val)])
                        strategy += f" (max {max_val} tweets)"
                except ValueError:
                    admin_bp.logger.warning(f"Invalid max_tweets value: {max_tweets}")

            result = subprocess.run(cmd, cwd=base_dir, check=True, timeout=config.get_command_timeout('fetch'))  # 10 minute timeout for fetch
            admin_bp.logger.info(f"User fetch completed for @{username} ({strategy})")

            # Only trigger analysis if action is fetch_and_analyze
            if action == 'fetch_and_analyze':
                # Run analysis in a separate subprocess after fetch completes
                analysis_cmd = ["./run_in_venv.sh", "analyze-twitter", "--username", username]
                analysis_result = subprocess.run(analysis_cmd, cwd=base_dir, check=True, timeout=config.get_command_timeout('analyze'))
                admin_bp.logger.info(f"Analysis completed for @{username}")

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

        def reanalyze_category():
            try:
                # Get tweets from specific category
                from utils.database import get_db_connection_context
                with get_db_connection_context() as conn:
                    rows = conn.execute("SELECT post_id FROM content_analyses WHERE category = ? LIMIT 20", (category,)).fetchall()

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
            admin_bp.logger.error(f"Traceback: {traceback.format_exc()}")
            flash('Ocurrió un error al procesar la solicitud. Inténtalo de nuevo.', 'error')
            return redirect(referrer or url_for('admin.admin_dashboard'))

    # GET request - show edit form (use direct SQL to align with tests)
    from utils.database import get_db_connection_context
    with get_db_connection_context() as conn:
        row = conn.execute("""
            SELECT 
                t.content,
                t.username,
                t.tweet_timestamp,
                ca.category,
                ca.local_explanation,
                ca.external_explanation,
                t.tweet_url,
                t.original_content,
                ca.verification_data,
                ca.verification_confidence
            FROM tweets t
            LEFT JOIN content_analyses ca ON ca.post_id = t.tweet_id
            WHERE t.tweet_id = ?
        """, (tweet_id,)).fetchone()

    if not row:
        flash('Tweet no encontrado', 'error')
        return redirect(url_for('admin.admin_dashboard'))

    # Extract data from row using column names (sqlite3.Row supports dict-like access)
    # Use best explanation for display (external if available, otherwise local)
    best_explanation = row['external_explanation'] if row['external_explanation'] else row['local_explanation'] if row['local_explanation'] else ''
    
    tweet_dict = {
        'content': row['content'],
        'username': row['username'],
        'tweet_timestamp': row['tweet_timestamp'],
        'category': row['category'] if row['category'] is not None else 'general',
        'best_explanation': best_explanation,
        'local_explanation': row['local_explanation'] if row['local_explanation'] is not None else '',
        'external_explanation': row['external_explanation'] if row['external_explanation'] is not None else '',
        'tweet_url': row['tweet_url'] if row['tweet_url'] is not None else '',
        'original_content': row['original_content'] if row['original_content'] is not None else '',
        'verification_data': json.loads(row['verification_data']) if row['verification_data'] is not None else None,
        'verification_confidence': row['verification_confidence'] if row['verification_confidence'] is not None else 0.0
    }

    from web.utils.decorators import ANALYSIS_CATEGORIES
    return render_template('admin/edit_analysis.html',
                           tweet=tweet_dict,
                           tweet_id=tweet_id,
                           categories=ANALYSIS_CATEGORIES,
                           referrer=referrer)

@admin_bp.route('/trigger-external/<tweet_id>', methods=['POST'])
@admin_required
def trigger_external_analysis(tweet_id: str):
    """Manually trigger external Gemini analysis on a tweet (admin-only)."""
    try:
        # Get tweet content and media
        tweet_data = get_tweet_data(tweet_id)
        if not tweet_data:
            return jsonify({'success': False, 'error': 'Tweet no encontrado'}), 404
        
        content = tweet_data.get('content', '')
        media_urls = tweet_data.get('media_urls', [])
        
        # Run external analysis with admin override
        external_analyzer = ExternalAnalyzer(verbose=True)
        
        # Use asyncio to run the async analysis
        try:
            loop = asyncio.get_running_loop()
            # If loop is already running, create a task
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, external_analyzer.analyze(content, media_urls))
                external_explanation = future.result(timeout=120)  # 2 minute timeout
        except RuntimeError:
            # No event loop running, create a new one
            external_explanation = asyncio.run(external_analyzer.analyze(content, media_urls))
        
        if not external_explanation:
            return jsonify({'success': False, 'error': 'No se pudo generar el análisis externo'}), 500
        
        # Save external explanation to database
        with get_db_connection_context() as conn:
            # Update content_analyses with external explanation
            conn.execute("""
                UPDATE content_analyses 
                SET external_explanation = ?,
                    external_analysis_used = 1,
                    analysis_stages = CASE 
                        WHEN analysis_stages LIKE '%external%' THEN analysis_stages
                        ELSE analysis_stages || ',external'
                    END,
                    analysis_timestamp = CURRENT_TIMESTAMP
                WHERE post_id = ?
            """, (external_explanation, tweet_id))
            conn.commit()
        
        return jsonify({
            'success': True, 
            'message': 'Análisis externo completado',
            'external_explanation': external_explanation
        })
        
    except Exception as e:
        admin_bp.logger.error(f"Error triggering external analysis: {str(e)}")
        admin_bp.logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

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
        from utils.database import get_db_connection_context
        with get_db_connection_context() as conn:
            # 1) Check if category exists
            exists_row = conn.execute("SELECT COUNT(*) AS cnt FROM content_analyses WHERE category = ?", (category_name,)).fetchone()
            exists_count = exists_row['cnt'] if exists_row else 0
            if not exists_count:
                flash(f'No se encontraron tweets para la categoría "{category_name}"', 'info')
                return redirect(url_for('admin.admin_dashboard'))

            # 2) Total count
            total_count_row = conn.execute("SELECT COUNT(*) AS cnt FROM content_analyses WHERE category = ?", (category_name,)).fetchone()
            total_count = total_count_row['cnt'] if total_count_row else 0

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
                    ca.local_explanation,
                    ca.external_explanation,
                    ca.analysis_stages,
                    ca.external_analysis_used,
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

            # 4) Category stats (external vs local, unique users)
            stats_row = conn.execute("""
                SELECT 
                    SUM(CASE WHEN external_analysis_used=1 THEN 1 ELSE 0 END) AS external_count,
                    SUM(CASE WHEN analysis_stages LIKE '%local_llm%' THEN 1 ELSE 0 END) AS local_llm_count,
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

        # Build tweets list
        processed_tweets = []
        for r in recent_rows or []:
            # Determine best explanation
            best_explanation = r['external_explanation'] if r['external_explanation'] else r['local_explanation']
            # Determine display method from analysis stages
            stages = r['analysis_stages'] or 'pattern'
            if r['external_analysis_used']:
                display_method = f"{stages} (external)"
            else:
                display_method = stages
                
            processed_tweets.append({
                'tweet_url': r['tweet_url'],
                'content': r['content'],
                'username': r['username'],
                'tweet_timestamp': r['tweet_timestamp'],
                'tweet_id': r['tweet_id'],
                'category': r['category'],
                'best_explanation': best_explanation,
                'analysis_stages': display_method,
                'analysis_timestamp': r['analysis_timestamp'],
                'is_deleted': r['is_deleted'],
                'is_edited': r['is_edited'],
                'post_type': r['post_type']
            })

        # Category stats
        external_count = stats_row['external_count'] if stats_row else 0
        local_llm_count = stats_row['local_llm_count'] if stats_row else 0
        unique_users = stats_row['unique_users'] if stats_row else 0

        category_stats = {
            'total_tweets': total_count,
            'unique_users': unique_users,
            'local_llm_analyzed': local_llm_count,
            'pattern_analyzed': total_count - external_count - local_llm_count,
            'external_analyzed': external_count
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
    from utils.database import get_db_connection_context
    with get_db_connection_context() as conn:
        # Check if analysis exists
        row = conn.execute("SELECT post_id FROM content_analyses WHERE post_id = ?", (tweet_id,)).fetchone()
        if row:
            # Update existing analysis with manual annotation
            conn.execute("""
                UPDATE content_analyses 
                SET category = ?, 
                    local_explanation = ?, 
                    analysis_stages = 'manual', 
                    analysis_timestamp = CURRENT_TIMESTAMP 
                WHERE post_id = ?
            """, (new_category, 'Categoría asignada manualmente por administrador', tweet_id))
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
                    INSERT INTO content_analyses (
                        post_id, category, local_explanation, analysis_stages, 
                        author_username, post_content, post_url, analysis_timestamp
                    )
                    VALUES (?, ?, ?, 'manual', ?, ?, ?, CURRENT_TIMESTAMP)
                """, (tweet_id, new_category, 'Categoría asignada manualmente por administrador', username, content, tweet_url))
                conn.commit()
                success = True
            else:
                success = False

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
        from utils.database import get_db_connection_context
        with get_db_connection_context() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    ca.post_id,
                    ca.author_username,
                    ca.category,
                    ca.local_explanation,
                    ca.external_explanation,
                    ca.analysis_stages,
                    ca.external_analysis_used,
                    ca.analysis_timestamp,
                    t.content as tweet_content,
                    t.tweet_url,
                    t.tweet_timestamp
                FROM content_analyses ca
                JOIN tweets t ON t.tweet_id = ca.post_id
                ORDER BY ca.analysis_timestamp DESC
            """)
            rows = cursor.fetchall()

        # Create CSV response
        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow([
            'Post ID', 'Author Username', 'Category', 'Local Explanation', 'External Explanation',
            'Analysis Stages', 'External Analysis Used', 'Analysis Timestamp', 'Post Content',
            'Post URL', 'Post Timestamp'
        ])

        # Write data
        for r in rows:
            writer.writerow([
                r['post_id'], r['author_username'], r['category'], r['local_explanation'], r['external_explanation'],
                r['analysis_stages'], r['external_analysis_used'], r['analysis_timestamp'], r['tweet_content'], 
                r['tweet_url'], r['tweet_timestamp']
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
        from utils.database import get_db_connection_context
        with get_db_connection_context() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    ca.post_id,
                    ca.author_username,
                    ca.category,
                    ca.local_explanation,
                    ca.external_explanation,
                    ca.analysis_stages,
                    ca.external_analysis_used,
                    ca.analysis_timestamp,
                    ca.post_content,
                    ca.post_url,
                    ca.categories_detected,
                    ca.verification_data,
                    ca.verification_confidence,
                    t.content as tweet_content,
                    t.tweet_url,
                    t.tweet_timestamp
                FROM content_analyses ca
                LEFT JOIN tweets t ON t.tweet_id = ca.post_id
                ORDER BY ca.analysis_timestamp DESC
            """)
            rows = cursor.fetchall()

        # Convert to JSON-serializable format
        data = []
        for r in rows:
            data.append({
                'post_id': r['post_id'],
                'author_username': r['author_username'],
                'category': r['category'],
                'local_explanation': r['local_explanation'],
                'external_explanation': r['external_explanation'],
                'analysis_stages': r['analysis_stages'],
                'external_analysis_used': bool(r['external_analysis_used']),
                'analysis_timestamp': r['analysis_timestamp'],
                'post_content': r['post_content'],
                'post_url': r['post_url'],
                'categories_detected': json.loads(r['categories_detected']) if r['categories_detected'] else None,
                'verification_data': json.loads(r['verification_data']) if r['verification_data'] else None,
                'verification_confidence': r['verification_confidence']
            })

        # Create JSON response
        json_data = json.dumps({
            'export_timestamp': datetime.now().isoformat(),
            'total_records': len(data),
            'data': data
        }, indent=2, ensure_ascii=False)

        # Create response
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

@admin_bp.route('/feedback')
@admin_required
def admin_view_feedback() -> str:
    """View all submitted feedback (admin only)."""
    page = request.args.get('page', 1, type=int)
    per_page = config.get_pagination_limit('admin_category')

    try:
        from utils.database import get_db_connection_context
        with get_db_connection_context() as conn:
            # Total count
            total_count_row = conn.execute("SELECT COUNT(*) AS cnt FROM user_feedback").fetchone()
            total_count = total_count_row['cnt'] if total_count_row else 0

            # Pagination
            offset = (page - 1) * per_page

            # Get feedback submissions
            feedback_rows = conn.execute("""
                SELECT uf.id, uf.post_id, uf.feedback_type, uf.original_category, uf.corrected_category,
                       uf.user_comment, uf.user_ip, uf.submitted_at, t.username, t.content, t.tweet_url
                FROM user_feedback uf
                LEFT JOIN tweets t ON t.tweet_id = uf.post_id
                ORDER BY uf.submitted_at DESC
                LIMIT ? OFFSET ?
            """, (per_page, offset)).fetchall()

            # Feedback stats
            stats_row = conn.execute("""
                SELECT
                    COUNT(*) as total_feedback,
                    SUM(CASE WHEN feedback_type = 'correction' THEN 1 ELSE 0 END) as corrections,
                    SUM(CASE WHEN feedback_type = 'improvement' THEN 1 ELSE 0 END) as improvements,
                    SUM(CASE WHEN feedback_type = 'bug_report' THEN 1 ELSE 0 END) as bug_reports,
                    COUNT(DISTINCT post_id) as unique_posts
                FROM user_feedback
            """).fetchone()

        # Process feedback
        feedback_list = []
        for r in feedback_rows or []:
            feedback_list.append({
                'id': r['id'],
                'post_id': r['post_id'],
                'feedback_type': r['feedback_type'],
                'original_category': r['original_category'],
                'corrected_category': r['corrected_category'],
                'user_comment': r['user_comment'],
                'user_ip': r['user_ip'],
                'submitted_at': r['submitted_at'],
                'username': r['username'],
                'tweet_content': r['content'],
                'tweet_url': r['tweet_url']
            })

        # Stats
        feedback_stats = {
            'total_feedback': stats_row['total_feedback'] if stats_row else 0,
            'corrections': stats_row['corrections'] if stats_row else 0,
            'improvements': stats_row['improvements'] if stats_row else 0,
            'bug_reports': stats_row['bug_reports'] if stats_row else 0,
            'unique_posts': stats_row['unique_posts'] if stats_row else 0
        }

        pagination = {
            'page': page,
            'per_page': per_page,
            'total': total_count,
            'total_pages': math.ceil(total_count / per_page) if total_count > 0 else 1
        }

        # Calculate pagination range for template
        start_page = max(1, page - 2)
        end_page = min(pagination['total_pages'] + 1, page + 3)
        pagination['page_range'] = list(range(start_page, end_page))

        return render_template('admin/feedback_view.html',
                               feedback=feedback_list,
                               pagination=pagination,
                               feedback_stats=feedback_stats)

    except Exception as e:
        admin_bp.logger.error(f"Error in admin_view_feedback: {str(e)}")
        flash('No se pudo cargar la información de feedback. Inténtalo de nuevo.', 'error')
        return redirect(url_for('admin.admin_dashboard'))