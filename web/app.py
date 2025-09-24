from flask import Flask, render_template, request, jsonify
import sqlite3
import os
import json
from datetime import datetime, timedelta
from collections import Counter
import re

app = Flask(__name__)
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'accounts.db')

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

def get_all_accounts():
    """Get list of all accounts with basic stats."""
    conn = get_db_connection()
    accounts = conn.execute("""
        SELECT 
            username,
            COUNT(*) as tweet_count,
            MAX(tweet_timestamp) as last_activity,
            COUNT(DISTINCT post_type) as post_type_variety
        FROM tweets 
        GROUP BY username 
        ORDER BY tweet_count DESC
    """).fetchall()
    conn.close()
    return [dict(row) for row in accounts]

@app.route('/')
def index():
    """Main dashboard with account overview (focus on analysis, not engagement)."""
    accounts = get_all_accounts()
    
    # Overall statistics - focus on analysis coverage
    conn = get_db_connection()
    overall_stats = conn.execute("""
        SELECT 
            COUNT(*) as total_tweets,
            COUNT(DISTINCT username) as total_accounts,
            COUNT(CASE WHEN media_count > 0 THEN 1 END) as tweets_with_media,
            (SELECT COUNT(*) FROM content_analyses) as analyzed_tweets
        FROM tweets
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
                         accounts=accounts,
                         overall_stats=dict(overall_stats) if overall_stats else {},
                         analysis_distribution=[dict(row) for row in analysis_distribution])

@app.route('/user/<username>')
def user_page(username):
    """User profile page with tweets and analysis focus."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get tweets for this user with analysis - removed engagement metrics
        query = '''
        SELECT 
            t.tweet_url, t.content, t.media_links, t.hashtags, t.mentions,
            t.tweet_timestamp, t.post_type,
            ca.category as analysis_category, ca.llm_explanation, ca.analysis_method, ca.analysis_timestamp
        FROM tweets t
        LEFT JOIN content_analyses ca ON t.tweet_id = ca.tweet_id
        WHERE t.username = ?
        ORDER BY t.tweet_timestamp DESC
        '''
        
        cursor.execute(query, (username,))
        results = cursor.fetchall()
        
        # Process tweets with enhanced analysis
        tweets = []
        for row in results:
            tweet = {
                'tweet_url': row[0],
                'content': row[1],
                'media_links': row[2],
                'hashtags_parsed': json.loads(row[3]) if row[3] else [],
                'mentions_parsed': json.loads(row[4]) if row[4] else [],
                'tweet_timestamp': row[5],
                'post_type': row[6],
                'analysis_category': row[7],
                'llm_explanation': row[8],
                'analysis_method': row[9],
                'analysis_timestamp': row[10]
            }
            
            # Use the llm_explanation directly - it contains the best analysis available
            tweet['analysis_display'] = tweet['llm_explanation'] or "Sin análisis disponible"
            tweet['category'] = tweet['analysis_category'] or 'general'
            
            tweets.append(tweet)
        
        # Get account statistics (without engagement focus)
        stats = get_account_statistics(username)
        
        conn.close()
        
        return render_template('user.html', 
                             username=username, 
                             tweets=tweets,
                             stats=stats)
    
    except Exception as e:
        app.logger.error(f"Error in user_page for {username}: {str(e)}")
        return render_template('user.html', 
                             username=username, 
                             tweets=[],
                             stats={},
                             error=f"Error loading data: {str(e)}")

@app.route('/api/tweet-status/<tweet_id>')
def check_tweet_status(tweet_id):
    """API endpoint to check if a tweet is still available."""
    # This would be used by JavaScript to fallback to stored content if tweet is deleted
    return jsonify({'exists': True})  # Placeholder - would need actual Twitter API integration

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
