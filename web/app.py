from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import sqlite3
import os
import json
from datetime import datetime, timedelta
from collections import Counter
import re
import math
from functools import wraps
import secrets
from pathlib import Path

# Load environment variables from .env file
def load_env_file():
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

load_env_file()

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(32))

# Admin configuration
ADMIN_TOKEN = os.environ.get('ADMIN_TOKEN', 'admin123')  # Change this in production!

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'accounts.db')

def admin_required(f):
    """Decorator to require admin access for certain routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_authenticated'):
            flash('Acceso administrativo requerido', 'error')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

def get_db_connection():
    """Get database connection with row factory for easier access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_account_statistics(username):
    """Get comprehensive statistics for an account."""
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
        SELECT 
            category, 
            COUNT(*) as count,
            COUNT(*) * 100.0 / (SELECT COUNT(*) FROM content_analyses WHERE username = ?) as percentage
        FROM content_analyses 
        WHERE username = ? 
        GROUP BY category 
        ORDER BY count DESC
    """, (username, username)).fetchall()
    
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

def get_all_accounts(page=1, per_page=10):
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
            COUNT(CASE WHEN ca.category IS NOT NULL THEN 1 END) as analyzed_posts,
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

# Admin Routes
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Simple admin login page."""
    if request.method == 'POST':
        token = request.form.get('token')
        print(f"DEBUG: Received token: '{token}'")
        print(f"DEBUG: Expected token: '{ADMIN_TOKEN}'")
        print(f"DEBUG: Tokens match: {token == ADMIN_TOKEN}")
        if token == ADMIN_TOKEN:
            session['admin_authenticated'] = True
            flash('Acceso administrativo concedido', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Token administrativo incorrecto', 'error')
    
    return render_template('admin/login.html')

@app.route('/admin/logout')
def admin_logout():
    """Admin logout."""
    session.pop('admin_authenticated', None)
    flash('Sesión administrativa cerrada', 'info')
    return redirect(url_for('index'))

@app.route('/admin')
@admin_required
def admin_dashboard():
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

@app.route('/admin/reanalyze', methods=['POST'])
@admin_required
def admin_reanalyze():
    """Trigger reanalysis of tweets."""
    action = request.form.get('action')
    
    if action == 'all':
        # Reanalyze all tweets with background subprocess
        import subprocess
        import threading
        from pathlib import Path
        
        base_dir = Path(__file__).parent.parent
        
        def run_analysis_background():
            try:
                cmd = ["./run_in_venv.sh", "analyze-db", "--force-reanalyze"]
                subprocess.run(cmd, cwd=base_dir, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"Analysis error: {e}")
        
        thread = threading.Thread(target=run_analysis_background)
        thread.daemon = True
        thread.start()
        flash('Reanálisis de TODOS los tweets iniciado (sin límite)', 'success')
    
    elif action == 'category':
        category = request.form.get('category')
        if category:
            # Reanalyze tweets from specific category using direct analysis
            import threading
            
            def reanalyze_category():
                try:
                    import sys
                    import os
                    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
                    
                    from enhanced_analyzer import EnhancedAnalyzer, save_content_analysis
                    
                    # Get tweets from specific category
                    conn = get_db_connection()
                    tweets = conn.execute("""
                        SELECT t.tweet_id, t.content, t.username 
                        FROM tweets t
                        JOIN content_analyses ca ON t.tweet_id = ca.tweet_id
                        WHERE ca.category = ?
                        LIMIT 20
                    """, (category,)).fetchall()
                    
                    if tweets:
                        analyzer = EnhancedAnalyzer(model_priority="balanced")
                        
                        for tweet in tweets:
                            # Delete existing analysis
                            conn.execute("DELETE FROM content_analyses WHERE tweet_id = ?", (tweet[0],))
                            conn.commit()
                            
                            # Reanalyze
                            analysis_result = analyzer.analyze_content(tweet[1])
                            save_content_analysis(
                                tweet_id=tweet[0],
                                username=tweet[2],
                                analysis_result=analysis_result
                            )
                            print(f"✅ Reanálizado tweet {tweet[0]} de @{tweet[2]}: {analysis_result.category}")
                    
                    conn.close()
                    
                except Exception as e:
                    print(f"Error en reanálisis por categoría: {e}")
            
            thread = threading.Thread(target=reanalyze_category)
            thread.daemon = True
            thread.start()
            flash(f'Reanálisis de categoría "{category}" iniciado (máximo 20 tweets)', 'success')
    
    elif action == 'user':
        username = request.form.get('username')
        if username:
            import subprocess
            import threading
            from pathlib import Path
            
            base_dir = Path(__file__).parent.parent
            
            def run_user_analysis():
                try:
                    cmd = ["./run_in_venv.sh", "analyze-db", "--username", username, "--force-reanalyze"]
                    subprocess.run(cmd, cwd=base_dir, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    print(f"Analysis error: {e}")
            
            thread = threading.Thread(target=run_user_analysis)
            thread.daemon = True
            thread.start()
            flash(f'Reanálisis de usuario "@{username}" iniciado', 'success')
    
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/edit-analysis/<tweet_id>', methods=['GET', 'POST'])
@admin_required
def admin_edit_analysis(tweet_id):
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
                # Trigger full reanalysis using the analysis pipeline
                import sys
                import os
                sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
                
                from enhanced_analyzer import EnhancedAnalyzer, save_content_analysis
                
                # Get tweet data for reanalysis
                tweet_data = conn.execute("""
                    SELECT tweet_id, content, username FROM tweets WHERE tweet_id = ?
                """, (tweet_id,)).fetchone()
                
                if not tweet_data:
                    flash('Tweet no encontrado', 'error')
                    conn.close()
                    return redirect(referrer or url_for('admin_dashboard'))
                
                # Delete existing analysis to force reanalysis
                conn.execute("DELETE FROM content_analyses WHERE tweet_id = ?", (tweet_id,))
                conn.commit()
                conn.close()
                
                # Initialize analyzer and reanalyze
                analyzer = EnhancedAnalyzer(model_priority="balanced")
                print(f"🔄 Reanalizando tweet {tweet_id} de @{tweet_data[2]}")
                
                # Analyze the content
                analysis_result = analyzer.analyze_content(tweet_data[1])  # content
                
                # Save the new analysis
                save_content_analysis(
                    tweet_id=tweet_data[0],
                    username=tweet_data[2], 
                    analysis_result=analysis_result
                )
                
                flash(f'Tweet reanálizado correctamente. Nueva categoría: {analysis_result.category}', 'success')
                
            else:
                # Manual update
                if not new_category or not new_explanation:
                    flash('Categoría y explicación son requeridas', 'error')
                    conn.close()
                    return redirect(request.url)
                
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
            
            # Redirect back to user view if possible, otherwise admin dashboard
            if referrer and '/user/' in referrer:
                return redirect(referrer)
            else:
                return redirect(url_for('admin_dashboard'))
                
        except Exception as e:
            app.logger.error(f"Error in admin_edit_analysis: {str(e)}")
            flash('Ocurrió un error al procesar la solicitud. Inténtalo de nuevo.', 'error')
            conn.close()
            return redirect(referrer or url_for('admin_dashboard'))
    
    # GET request - show edit form
    try:
        # Get tweet and current analysis
        tweet_data = conn.execute("""
            SELECT 
                t.content, t.username, t.tweet_timestamp,
                ca.category, ca.llm_explanation, t.tweet_url
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
            'tweet_url': tweet_data[5] or ''
        }
        
        categories = ['general', 'hate_speech', 'disinformation', 'conspiracy_theory', 
                      'far_right_bias', 'call_to_action', 'political_general']
        
        return render_template('admin/edit_analysis.html',
                             tweet=tweet_dict,
                             tweet_id=tweet_id,
                             categories=categories,
                             referrer=referrer)
    
    except Exception as e:
        app.logger.error(f"Error loading edit analysis for {tweet_id}: {str(e)}")
        flash('No se pudo cargar la información del tweet. Inténtalo de nuevo.', 'error')
        return redirect(referrer or url_for('admin_dashboard'))

@app.route('/admin/reanalyze-single/<tweet_id>', methods=['POST'])
@admin_required
def admin_reanalyze_single(tweet_id):
    """Reanalyze a single tweet using the analysis pipeline directly."""
    try:
        # Import the analyzer directly
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        
        from enhanced_analyzer import EnhancedAnalyzer, save_content_analysis
        
        # Get tweet data
        conn = get_db_connection()
        tweet_data = conn.execute("""
            SELECT tweet_id, content, username FROM tweets WHERE tweet_id = ?
        """, (tweet_id,)).fetchone()
        
        if not tweet_data:
            conn.close()
            flash('Tweet no encontrado', 'error')
            return redirect(request.referrer or url_for('index'))
        
        # Delete existing analysis to force reanalysis
        conn.execute("DELETE FROM content_analyses WHERE tweet_id = ?", (tweet_id,))
        conn.commit()
        conn.close()
        
        # Initialize analyzer and reanalyze
        analyzer = EnhancedAnalyzer(model_priority="balanced")
        
        print(f"🔄 Reanalizando tweet {tweet_id} de @{tweet_data[2]}")
        
        # Analyze the content
        analysis_result = analyzer.analyze_content(tweet_data[1])  # content
        
        # Save the new analysis
        save_content_analysis(
            tweet_id=tweet_data[0],
            username=tweet_data[2], 
            analysis_result=analysis_result
        )
        
        flash(f'Tweet reanálizado correctamente. Nueva categoría: {analysis_result.category}', 'success')
        
    except Exception as e:
        print(f"Error durante reanálisis: {e}")
        flash('El reanálisis falló. Inténtalo de nuevo más tarde.', 'error')
    
    return redirect(request.referrer or url_for('index'))

@app.route('/admin/category/<category_name>')
@admin_required
def admin_view_category(category_name):
    """View all tweets from a specific category (admin only)."""
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    try:
        conn = get_db_connection()
        
        # Debug: Check if category exists
        category_check = conn.execute("""
            SELECT COUNT(*) FROM content_analyses WHERE category = ?
        """, (category_name,)).fetchone()
        
        if not category_check or category_check[0] == 0:
            conn.close()
            flash(f'No se encontraron tweets para la categoría "{category_name}"', 'info')
            return redirect(url_for('admin_dashboard'))
        
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
            return redirect(url_for('admin_dashboard'))
        
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
                        'tweet_url': row[0] or '',
                        'content': row[1] or '',
                        'username': row[2] or '',
                        'tweet_timestamp': row[3] or '',
                        'tweet_id': row[4] or '',
                        'category': row[5] or category_name,
                        'llm_explanation': row[6] or '',
                        'analysis_method': row[7] or 'unknown',
                        'analysis_timestamp': row[8] or '',
                        'is_deleted': bool(row[9]) if row[9] is not None else False,
                        'is_edited': bool(row[10]) if row[10] is not None else False,
                        'post_type': row[11] or 'original'
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
        app.logger.error(f"Error in admin_view_category for {category_name}: {str(e)}")
        app.logger.error(f"Error details: {type(e).__name__}: {e}")
        import traceback
        app.logger.error(f"Traceback: {traceback.format_exc()}")
        flash('No se pudo cargar la información de la categoría. Inténtalo de nuevo.', 'error')
        return redirect(url_for('admin_dashboard'))

@app.route('/admin/user-category/<username>/<category>')
@admin_required  
def admin_view_user_category(username, category):
    """View specific user's tweets in a specific category (admin only)."""
    return redirect(url_for('user_page', username=username, category=category))

@app.route('/admin/quick-edit-category/<tweet_id>', methods=['POST'])
@admin_required
def admin_quick_edit_category(tweet_id):
    """Quickly change the category of a tweet."""
    new_category = request.form.get('category')
    
    if not new_category:
        flash('Categoría requerida', 'error')
        return redirect(request.referrer or url_for('index'))
    
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
    return redirect(request.referrer or url_for('index'))

@app.route('/')
def index():
    """Main dashboard with account overview (focus on analysis, not engagement)."""
    page = request.args.get('page', 1, type=int)
    category_filter = request.args.get('category', None)
    
    accounts_data = get_all_accounts(page=page, per_page=10)
    
    # Filter accounts by category if specified
    if category_filter and category_filter != 'all':
        filtered_accounts = []
        conn = get_db_connection()
        for account in accounts_data['accounts']:
            # Check if account has posts in this category
            has_category = conn.execute("""
                SELECT COUNT(*) FROM content_analyses ca
                JOIN tweets t ON ca.tweet_id = t.tweet_id  
                WHERE t.username = ? AND ca.category = ?
            """, (account['username'], category_filter)).fetchone()[0]
            
            if has_category > 0:
                filtered_accounts.append(account)
        conn.close()
        accounts_data['accounts'] = filtered_accounts
    
    # Overall statistics - simplified to just accounts and analyzed posts
    conn = get_db_connection()
    overall_stats = conn.execute("""
        SELECT 
            COUNT(DISTINCT t.username) as total_accounts,
            COUNT(CASE WHEN ca.tweet_id IS NOT NULL THEN 1 END) as analyzed_tweets
        FROM tweets t
        LEFT JOIN content_analyses ca ON t.tweet_id = ca.tweet_id
    """).fetchone()
    
    # Analysis distribution
    analysis_distribution = conn.execute("""
        SELECT 
            category,
            COUNT(*) as count,
            COUNT(*) * 100.0 / (SELECT COUNT(*) FROM content_analyses) as percentage
        FROM content_analyses 
        GROUP BY category 
        ORDER BY count DESC
    """).fetchall()
    
    conn.close()
    
    return render_template('index.html', 
                         accounts_data=accounts_data,
                         overall_stats=dict(overall_stats) if overall_stats else {},
                         analysis_distribution=[dict(row) for row in analysis_distribution],
                         current_category=category_filter)

@app.route('/user/<username>')
def user_page(username):
    """User profile page with tweets and analysis focus."""
    page = request.args.get('page', 1, type=int)
    category_filter = request.args.get('category', None)
    per_page = 10
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get user profile picture from accounts table first
        user_profile = cursor.execute("""
            SELECT profile_pic_url FROM accounts WHERE username = ?
        """, [username]).fetchone()
        user_profile_pic = user_profile[0] if user_profile else None
        
        # Base query with optional category filter including profile picture
        base_query = '''
        SELECT 
            t.tweet_url, t.content, t.media_links, t.hashtags, t.mentions,
            t.tweet_timestamp, t.post_type, t.tweet_id,
            ca.category as analysis_category, ca.llm_explanation, ca.analysis_method, ca.analysis_timestamp,
            ca.categories_detected,
            t.is_deleted, t.is_edited, t.rt_original_analyzed,
            t.original_author, t.original_tweet_id, t.reply_to_username
        FROM tweets t
        LEFT JOIN content_analyses ca ON t.tweet_id = ca.tweet_id
        WHERE t.username = ?
        '''
        
        query_params = [username]
        
        # Add category filter if specified
        if category_filter and category_filter != 'all':
            base_query += ' AND ca.category = ?'
            query_params.append(category_filter)
        
        # Order by truly problematic content first, then general content last
        order_clause = '''
        ORDER BY 
            CASE 
                WHEN ca.category IS NULL OR ca.category = 'general' THEN 1 
                ELSE 0 
            END,
            t.tweet_timestamp DESC
        '''
        
        # Get total count for pagination
        count_query = f"SELECT COUNT(*) FROM ({base_query}) as subquery"
        total_tweets = cursor.execute(count_query, query_params).fetchone()[0]
        
        # Add pagination
        offset = (page - 1) * per_page
        paginated_query = f"{base_query} {order_clause} LIMIT ? OFFSET ?"
        query_params.extend([per_page, offset])
        
        cursor.execute(paginated_query, query_params)
        results = cursor.fetchall()
        
        # Process tweets with enhanced multi-category analysis
        tweets = []
        for row in results:
            # Parse multi-category data
            categories_detected = []
            try:
                if row[12]:  # categories_detected
                    categories_detected = json.loads(row[12])
            except (json.JSONDecodeError, TypeError):
                # Fallback to single category for backward compatibility
                if row[8]:  # analysis_category
                    categories_detected = [row[8]]
            
            tweet = {
                'tweet_url': row[0],
                'content': row[1],
                'media_links': row[2],
                'hashtags_parsed': json.loads(row[3]) if row[3] else [],
                'mentions_parsed': json.loads(row[4]) if row[4] else [],
                'tweet_timestamp': row[5],
                'post_type': row[6],
                'tweet_id': row[7],
                'analysis_category': row[8],
                'llm_explanation': row[9],
                'analysis_method': row[10],
                'analysis_timestamp': row[11],
                'categories_detected': categories_detected,
                'profile_pic_url': user_profile_pic,  # Use profile from accounts table
                # Post status fields from simplified schema (corrected indexes)
                'is_deleted': row[13],  # Fixed: was row[12], now row[13]
                'is_edited': row[14],   # Fixed: was row[13], now row[14] 
                'rt_original_analyzed': row[15],  # Fixed: was row[14], now row[15]
                'original_author': row[16],       # Fixed: was row[15], now row[16]
                'original_tweet_id': row[17],     # Fixed: was row[16], now row[17]
                'reply_to_username': row[18]      # Fixed: was row[17], now row[18]
            }
            
            # Post status warnings (simplified schema)
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
            if tweet['post_type'] in ['repost_other', 'repost_own', 'quote']:
                tweet['is_rt'] = True
                tweet['rt_type'] = tweet['post_type']
            else:
                tweet['is_rt'] = False
                tweet['rt_type'] = None
            
            # Use the llm_explanation directly - it contains the best analysis available
            tweet['analysis_display'] = tweet['llm_explanation'] or "Sin análisis disponible"
            tweet['category'] = tweet['analysis_category'] or 'general'
            tweet['has_multiple_categories'] = len(categories_detected) > 1
            
            tweets.append(tweet)
        
        # Get simplified account statistics (only analyzed posts count)
        stats_query = """
        SELECT 
            COUNT(CASE WHEN ca.tweet_id IS NOT NULL THEN 1 END) as analyzed_posts
        FROM tweets t
        LEFT JOIN content_analyses ca ON t.tweet_id = ca.tweet_id
        WHERE t.username = ?
        """
        basic_stats = cursor.execute(stats_query, [username]).fetchone()
        
        # Analysis results breakdown for pie chart
        analysis_stats = conn.execute("""
            SELECT 
                category, 
                COUNT(*) as count,
                COUNT(*) * 100.0 / (SELECT COUNT(*) FROM content_analyses WHERE username = ?) as percentage
            FROM content_analyses 
            WHERE username = ? 
            GROUP BY category 
            ORDER BY count DESC
        """, (username, username)).fetchall()
        
        conn.close()
        
        # Prepare tweets data with pagination info
        tweets_data = {
            'tweets': tweets,
            'page': page,
            'per_page': per_page,
            'total_tweets': total_tweets,
            'total_pages': math.ceil(total_tweets / per_page) if total_tweets > 0 else 1
        }
        
        stats = {
            'basic': {
                'total_tweets': total_tweets,
                'analyzed_posts': basic_stats[0] if basic_stats else 0
            },
            'analysis': [dict(row) for row in analysis_stats]
        }
        
        return render_template('user.html', 
                             username=username, 
                             tweets_data=tweets_data,
                             stats=stats,
                             current_category=category_filter,
                             user_profile_pic=user_profile_pic)
    
    except Exception as e:
        app.logger.error(f"Error in user_page for {username}: {str(e)}")
        return render_template('user.html', 
                             username=username, 
                             tweets_data={
                                 'tweets': [],
                                 'page': 1,
                                 'per_page': per_page,
                                 'total_tweets': 0,
                                 'total_pages': 1
                             },
                             stats={'basic': {}, 'analysis': []},
                             current_category=None,
                             error=f"Error loading data: {str(e)}")

@app.route('/api/tweet-status/<tweet_id>')
def check_tweet_status(tweet_id):
    """API endpoint to check if a tweet is still available."""
    # This would be used by JavaScript to fallback to stored content if tweet is deleted
    return jsonify({'exists': True})  # Placeholder - would need actual Twitter API integration

@app.route('/admin/export/csv')
@admin_required
def export_csv():
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
        from flask import Response
        response = Response(
            csv_data,
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename=dimetuverdad_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            }
        )
        return response
        
    except Exception as e:
        app.logger.error(f"Error exporting CSV: {str(e)}")
        flash(f'Error exporting CSV: {str(e)}', 'error')
        return redirect(url_for('admin_dashboard'))

@app.route('/admin/export/json')
@admin_required
def export_json():
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
                ca.subcategory,
                ca.targeted_groups,
                ca.calls_to_action,
                ca.evidence_sources,
                ca.verification_status,
                ca.misinformation_risk,
                ca.categories_detected,
                ca.category_scores
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
                'tweet_timestamp': row['tweet_timestamp']
            }
            
            # Add optional fields if they exist
            if row['subcategory']:
                record['subcategory'] = row['subcategory']
            if row['targeted_groups']:
                record['targeted_groups'] = row['targeted_groups']
            if row['calls_to_action'] is not None:
                record['calls_to_action'] = bool(row['calls_to_action'])
            if row['evidence_sources']:
                record['evidence_sources'] = row['evidence_sources']
            if row['verification_status']:
                record['verification_status'] = row['verification_status']
            if row['misinformation_risk']:
                record['misinformation_risk'] = row['misinformation_risk']
            if row['categories_detected']:
                record['categories_detected'] = row['categories_detected']
            if row['category_scores']:
                record['category_scores'] = row['category_scores']
                
            export_data['data'].append(record)
        
        # Create response
        from flask import Response
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
        app.logger.error(f"Error exporting JSON: {str(e)}")
        flash(f'Error exporting JSON: {str(e)}', 'error')
        return redirect(url_for('admin_dashboard'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
