from flask import Flask, render_template, request, jsonify, redirect, url_for
import sqlite3
import os
import json
from datetime import datetime, timedelta

app = Flask(__name__)
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'accounts.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    return conn

@app.route('/')
def index():
    """Dashboard for journalists"""
    conn = get_db_connection()
    
    # Get summary statistics
    stats = {}
    
    # Total tweets
    stats['total_tweets'] = conn.execute('SELECT COUNT(*) FROM tweets').fetchone()[0]
    
    # Analyzed tweets
    stats['analyzed_tweets'] = conn.execute('SELECT COUNT(*) FROM journalist_analyses').fetchone()[0]
    
    # High priority fact-checks
    stats['high_priority'] = conn.execute(
        'SELECT COUNT(*) FROM journalist_analyses WHERE fact_check_priority IN ("high", "critical")'
    ).fetchone()[0]
    
    # Recent categories (last 24h)
    yesterday = (datetime.now() - timedelta(days=1)).isoformat()
    categories = conn.execute('''
        SELECT category, COUNT(*) as count 
        FROM journalist_analyses 
        WHERE analysis_timestamp > ? 
        GROUP BY category 
        ORDER BY count DESC
    ''', (yesterday,)).fetchall()
    
    # Users with most content needing fact-checking
    fact_check_users = conn.execute('''
        SELECT username, COUNT(*) as count 
        FROM journalist_analyses 
        WHERE fact_check_priority IN ("high", "critical", "medium")
        GROUP BY username 
        ORDER BY count DESC 
        LIMIT 10
    ''').fetchall()
    
    # Recent high-priority items
    recent_priority = conn.execute('''
        SELECT tweet_url, username, category, subcategory, confidence, llm_explanation
        FROM journalist_analyses 
        WHERE fact_check_priority IN ("high", "critical")
        ORDER BY analysis_timestamp DESC 
        LIMIT 5
    ''').fetchall()
    
    conn.close()
    
    return render_template('index.html', 
                         stats=stats, 
                         categories=categories,
                         fact_check_users=fact_check_users,
                         recent_priority=recent_priority)

@app.route('/user/<username>')
def user_analysis(username):
    """Detailed analysis for a specific user"""
    conn = get_db_connection()
    
    # Get user's analyzed tweets
    analyses = conn.execute('''
        SELECT * FROM journalist_analyses 
        WHERE username = ? 
        ORDER BY analysis_timestamp DESC
    ''', (username,)).fetchall()
    
    # Get user stats
    user_stats = {
        'total_analyzed': len(analyses),
        'high_priority': len([a for a in analyses if a['fact_check_priority'] == 'high']),
        'disinformation': len([a for a in analyses if a['category'] == 'disinformation']),
        'hate_speech': len([a for a in analyses if a['category'] == 'hate_speech']),
        'avg_far_right_score': sum(a['far_right_score'] for a in analyses) / len(analyses) if analyses else 0
    }
    
    conn.close()
    
    return render_template('user_analysis.html', 
                         username=username, 
                         analyses=analyses,
                         user_stats=user_stats)

@app.route('/category/<category>')
def category_view(category):
    """View tweets by category"""
    conn = get_db_connection()
    
    analyses = conn.execute('''
        SELECT * FROM journalist_analyses 
        WHERE category = ? 
        ORDER BY confidence DESC, analysis_timestamp DESC
    ''', (category,)).fetchall()
    
    # Category stats
    subcategories = {}
    for analysis in analyses:
        subcat = analysis['subcategory']
        subcategories[subcat] = subcategories.get(subcat, 0) + 1
    
    conn.close()
    
    return render_template('category_view.html',
                         category=category,
                         analyses=analyses,
                         subcategories=subcategories)

@app.route('/fact-check')
def fact_check_queue():
    """Fact-checking priority queue for journalists"""
    conn = get_db_connection()
    
    # Critical priority items
    critical_priority = conn.execute('''
        SELECT * FROM journalist_analyses 
        WHERE fact_check_priority = "critical"
        ORDER BY confidence DESC, far_right_score DESC
    ''').fetchall()
    
    # High priority items
    high_priority = conn.execute('''
        SELECT * FROM journalist_analyses 
        WHERE fact_check_priority = "high"
        ORDER BY confidence DESC, far_right_score DESC
    ''').fetchall()
    
    # Medium priority items
    medium_priority = conn.execute('''
        SELECT * FROM journalist_analyses 
        WHERE fact_check_priority = "medium"
        ORDER BY confidence DESC, far_right_score DESC
        LIMIT 20
    ''').fetchall()
    
    conn.close()
    
    return render_template('fact_check.html',
                         critical_priority=critical_priority,
                         high_priority=high_priority,
                         medium_priority=medium_priority)

@app.route('/search')
def search():
    """Search interface"""
    query = request.args.get('q', '')
    category_filter = request.args.get('category', '')
    
    if not query:
        return render_template('search.html', results=[], query='')
    
    conn = get_db_connection()
    
    # Build search query
    sql = '''
        SELECT ja.*, t.content 
        FROM journalist_analyses ja
        JOIN tweets t ON ja.tweet_id = t.tweet_id
        WHERE 1=1
    '''
    params = []
    
    # Text search
    if query:
        sql += ' AND (t.content LIKE ? OR ja.llm_explanation LIKE ?)'
        params.extend([f'%{query}%', f'%{query}%'])
    
    # Category filter
    if category_filter:
        sql += ' AND ja.category = ?'
        params.append(category_filter)
    
    sql += ' ORDER BY ja.confidence DESC LIMIT 50'
    
    results = conn.execute(sql, params).fetchall()
    
    # Get available categories for filter
    categories = conn.execute(
        'SELECT DISTINCT category FROM journalist_analyses ORDER BY category'
    ).fetchall()
    
    conn.close()
    
    return render_template('search.html', 
                         results=results, 
                         query=query,
                         category_filter=category_filter,
                         categories=categories)

@app.route('/api/analyze/<username>')
def api_analyze_user(username):
    """API endpoint to trigger analysis for a user"""
    try:
        from enhanced_analyzer import EnhancedAnalyzer, save_journalist_analysis
        
        # Get user's recent tweets
        conn = get_db_connection()
        tweets = conn.execute('''
            SELECT tweet_id, tweet_url, username, content 
            FROM tweets 
            WHERE username = ? AND length(content) > 20
            ORDER BY created_at DESC 
            LIMIT 20
        ''', (username,)).fetchall()
        conn.close()
        
        if not tweets:
            return jsonify({
                'success': False,
                'message': f'No tweets found for @{username}'
            })
        
        # Analyze with enhanced analyzer
        analyzer = EnhancedAnalyzer(use_llm=False, journalism_mode=True)
        results = []
        
        for tweet in tweets:
            try:
                analysis = analyzer.analyze_for_journalism(
                    tweet_id=tweet['tweet_id'],
                    tweet_url=tweet['tweet_url'],
                    username=tweet['username'],
                    content=tweet['content'],
                    retrieve_evidence=False  # Skip for speed
                )
                
                save_journalist_analysis(analysis)
                results.append(analysis)
                
            except Exception as e:
                print(f"Error analyzing tweet {tweet['tweet_id']}: {e}")
                continue
        
        return jsonify({
            'success': True,
            'analyzed': len(results),
            'message': f'Analyzed {len(results)} tweets for @{username}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/tweet/<tweet_id>/evidence')
def api_get_evidence(tweet_id):
    """Get evidence details for a specific tweet"""
    conn = get_db_connection()
    
    analysis = conn.execute(
        'SELECT evidence_sources, analysis_json FROM journalist_analyses WHERE tweet_id = ?',
        (tweet_id,)
    ).fetchone()
    
    conn.close()
    
    if not analysis:
        return jsonify({'error': 'Analysis not found'}), 404
    
    try:
        # Try to get evidence from the evidence_sources column first
        evidence_sources = []
        if analysis['evidence_sources']:
            evidence_sources = json.loads(analysis['evidence_sources'])
        
        # Fallback to analysis_json if no evidence_sources
        if not evidence_sources and analysis['analysis_json']:
            analysis_data = json.loads(analysis['analysis_json'])
            evidence_sources = analysis_data.get('evidence_sources', [])
        
        return jsonify({
            'evidence_sources': evidence_sources,
            'evidence_found': len(evidence_sources) > 0
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
