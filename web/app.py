from flask import Flask, render_template, request, jsonify
import sqlite3
import os
import json
from datetime import datetime, timedelta
from collections import Counter
import re
import math

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
            ca.categories_detected, t.profile_pic_url
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
                'profile_pic_url': row[13]  # Add profile picture URL
            }
            
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
