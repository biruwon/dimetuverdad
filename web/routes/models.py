"""
Multi-Model Analysis Routes: Web interface for viewing and comparing multi-model analysis results.
"""

from flask import Blueprint, render_template, jsonify, request
from utils.database import get_db_connection_context
from utils import database_multi_model
from analyzer.categories import Categories

models_bp = Blueprint('models', __name__, url_prefix='/models')


@models_bp.route('/tweet/<tweet_id>')
def tweet_model_comparison(tweet_id):
    """
    Display side-by-side comparison of all model analyses for a specific tweet.
    
    Args:
        tweet_id: Tweet identifier
        
    Returns:
        Rendered template with model comparison data
    """
    with get_db_connection_context() as conn:
        # Get tweet data
        cursor = conn.cursor()
        cursor.execute('''
            SELECT 
                tweet_id,
                tweet_url,
                username,
                content,
                media_links,
                tweet_timestamp
            FROM tweets
            WHERE tweet_id = ?
        ''', (tweet_id,))
        
        tweet_row = cursor.fetchone()
        
        if not tweet_row:
            return render_template('error.html', 
                                 error_message=f"Tweet {tweet_id} not found"), 404
        
        tweet_data = {
            'tweet_id': tweet_row['tweet_id'],
            'tweet_url': tweet_row['tweet_url'],
            'username': tweet_row['username'],
            'content': tweet_row['content'],
            'media_links': tweet_row['media_links'].split(',') if tweet_row['media_links'] else [],
            'timestamp': tweet_row['tweet_timestamp']
        }
        
        # Get all model analyses
        model_analyses = database_multi_model.get_model_analyses(conn, tweet_id)
        
        # Get consensus
        consensus = database_multi_model.get_model_consensus(conn, tweet_id)
        
        # Get main analysis (if exists)
        cursor.execute('''
            SELECT category, local_explanation, external_explanation
            FROM content_analyses
            WHERE post_id = ?
        ''', (tweet_id,))
        
        main_analysis_row = cursor.fetchone()
        main_analysis = None
        if main_analysis_row:
            main_analysis = {
                'category': main_analysis_row['category'],
                'local_explanation': main_analysis_row['local_explanation'],
                'external_explanation': main_analysis_row['external_explanation']
            }
    
    return render_template(
        'tweet_model_comparison.html',
        tweet=tweet_data,
        model_analyses=model_analyses,
        consensus=consensus,
        main_analysis=main_analysis
    )


@models_bp.route('/comparison')
def models_comparison_dashboard():
    """
    Dashboard showing model performance and agreement statistics across all tweets.
    
    Returns:
        Rendered template with model comparison dashboard
    """
    with get_db_connection_context() as conn:
        # Get model performance stats
        performance_stats = database_multi_model.get_model_performance_stats(conn)
        
        # Get agreement stats
        agreement_stats = database_multi_model.get_model_agreement_stats(conn)
        
        # Convert agreement percentages for chart (multiply by 100)
        agreement_by_category_percent = {}
        if agreement_stats.get('agreement_by_category'):
            for category, score in agreement_stats['agreement_by_category'].items():
                agreement_by_category_percent[category] = round(score * 100, 1)
        
        # Get pagination parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)  # Default to 10 to avoid loading too many images
        per_page = min(per_page, 50)  # Maximum 50 per page
        offset = (page - 1) * per_page
        
        # Get total count for pagination
        cursor = conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) as total
            FROM content_analyses
            WHERE multi_model_analysis = 1
        ''')
        total_analyses = cursor.fetchone()['total']
        total_pages = (total_analyses + per_page - 1) // per_page  # Ceiling division
        
        # Get recent multi-model analyses with all model results and media
        cursor.execute('''
            SELECT 
                ca.post_id,
                ca.author_username,
                ca.model_consensus_category,
                t.content,
                t.tweet_url,
                t.media_links,
                t.media_count,
                t.original_content,
                t.original_author,
                ca.analysis_timestamp
            FROM content_analyses ca
            JOIN tweets t ON ca.post_id = t.tweet_id
            WHERE ca.multi_model_analysis = 1
            ORDER BY ca.analysis_timestamp DESC
            LIMIT ? OFFSET ?
        ''', (per_page, offset))
        
        recent_analyses = []
        for row in cursor.fetchall():
            post_id = row['post_id']
            
            # Get all model analyses for this post
            model_results = database_multi_model.get_model_analyses(conn, post_id)
            
            # Get consensus
            consensus = database_multi_model.get_model_consensus(conn, post_id)
            
            # Parse media links
            media_urls = []
            if row['media_links']:
                media_urls = [url.strip() for url in row['media_links'].split(',') if url.strip()]
            
            recent_analyses.append({
                'post_id': post_id,
                'author': row['author_username'],
                'consensus_category': row['model_consensus_category'],
                'content': row['content'],
                'content_preview': row['content'][:100] + '...' if len(row['content']) > 100 else row['content'],
                'tweet_url': row['tweet_url'],
                'timestamp': row['analysis_timestamp'],
                'model_results': model_results,
                'consensus': consensus,
                'media_urls': media_urls,
                'media_count': row['media_count'],
                'quoted_content': row['original_content'],
                'quoted_author': row['original_author']
            })
    
    return render_template(
        'models_dashboard.html',
        performance_stats=performance_stats,
        agreement_stats=agreement_stats,
        agreement_by_category_percent=agreement_by_category_percent,
        recent_analyses=recent_analyses,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        total_analyses=total_analyses
    )


@models_bp.route('/stats')
def model_stats():
    """
    Detailed model statistics and performance metrics.
    
    Returns:
        JSON response with detailed model statistics
    """
    with get_db_connection_context() as conn:
        # Get performance stats
        performance_stats = database_multi_model.get_model_performance_stats(conn)
        
        # Get agreement stats
        agreement_stats = database_multi_model.get_model_agreement_stats(conn)
        
        # Calculate additional metrics
        total_multi_model = agreement_stats.get('total_posts_analyzed', 0)
        
        stats = {
            'models': performance_stats,
            'agreement': agreement_stats,
            'summary': {
                'total_multi_model_analyses': total_multi_model,
                'total_models': len(performance_stats),
                'avg_agreement_score': agreement_stats.get('avg_agreement_score', 0.0)
            }
        }
    
    return jsonify(stats)


@models_bp.route('/api/tweet/<tweet_id>/models')
def api_tweet_models(tweet_id):
    """
    API endpoint for getting model analyses for a specific tweet.
    
    Args:
        tweet_id: Tweet identifier
        
    Returns:
        JSON response with model analyses and consensus
    """
    with get_db_connection_context() as conn:
        # Get model analyses
        model_analyses = database_multi_model.get_model_analyses(conn, tweet_id)
        
        # Get consensus
        consensus = database_multi_model.get_model_consensus(conn, tweet_id)
        
        return jsonify({
            'tweet_id': tweet_id,
            'models': model_analyses,
            'consensus': consensus
        })


@models_bp.route('/api/comparison/category/<category>')
def api_category_model_comparison(category):
    """
    API endpoint for comparing model agreement on a specific category.
    
    Args:
        category: Content category to analyze
        
    Returns:
        JSON response with model agreement statistics for the category
    """
    with get_db_connection_context() as conn:
        cursor = conn.cursor()
        
        # Get all multi-model posts in this category
        cursor.execute('''
            SELECT post_id
            FROM content_analyses
            WHERE model_consensus_category = ? AND multi_model_analysis = 1
        ''', (category,))
        
        post_ids = [row['post_id'] for row in cursor.fetchall()]
        
        # Calculate agreement for each post
        agreements = []
        for post_id in post_ids:
            consensus = database_multi_model.get_model_consensus(conn, post_id)
            if consensus and consensus['category'] == category:
                agreements.append(consensus['agreement_score'])
        
        # Calculate statistics
        if agreements:
            avg_agreement = sum(agreements) / len(agreements)
            full_agreement = sum(1 for a in agreements if a == 1.0)
        else:
            avg_agreement = 0.0
            full_agreement = 0
        
        return jsonify({
            'category': category,
            'total_posts': len(post_ids),
            'avg_agreement_score': avg_agreement,
            'full_agreement_count': full_agreement,
            'full_agreement_percentage': (full_agreement / len(post_ids) * 100) if post_ids else 0
        })
