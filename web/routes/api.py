"""
API routes for the dimetuverdad web application.
Contains REST API endpoints for external integrations.
"""

from flask import Blueprint, request, jsonify
import json
from datetime import datetime
from typing import Optional, Dict, List, Any

from web.utils.decorators import rate_limit, ANALYSIS_CATEGORIES
from utils import database
import config

import logging
api_bp = Blueprint('api', __name__, url_prefix='/api')
api_bp.logger = logging.getLogger('web.routes.api')

def get_db_connection():
    """Get database connection with row factory for easier access."""
    return database.get_db_connection()

@api_bp.route('/feedback', methods=['POST'])
@rate_limit(**config.get_rate_limit('api_endpoints'))
def submit_feedback() -> str:
    """API endpoint for users to submit feedback on analysis accuracy."""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        tweet_id = data.get('tweet_id') or data.get('post_id')
        feedback_type = data.get('feedback_type', 'correction')
        original_category = data.get('original_category')
        corrected_category = data.get('corrected_category')
        user_comment = data.get('user_comment', '')

        # Validate required fields
        if not tweet_id:
            return jsonify({'error': 'tweet_id or post_id is required'}), 400

        if feedback_type == 'correction' and not corrected_category:
            return jsonify({'error': 'corrected_category is required for correction feedback'}), 400

        # Get user IP for rate limiting
        user_ip = request.remote_addr

        # Check if tweet exists
        conn = get_db_connection()
        tweet_exists = conn.execute("""
            SELECT tweet_id FROM tweets WHERE tweet_id = ?
        """, (tweet_id,)).fetchone()

        if not tweet_exists:
            conn.close()
            return jsonify({'error': 'Tweet not found'}), 404

        # Check for recent feedback from this IP for this tweet (rate limiting)
        recent_feedback = conn.execute("""
            SELECT id FROM user_feedback
            WHERE post_id = ? AND user_ip = ? AND submitted_at > datetime('now', '-1 hour')
        """, (tweet_id, user_ip)).fetchone()

        if recent_feedback:
            conn.close()
            return jsonify({'error': 'Feedback already submitted recently for this tweet'}), 429

        # Insert feedback
        conn.execute("""
            INSERT INTO user_feedback (post_id, feedback_type, original_category, corrected_category, user_comment, user_ip)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (tweet_id, feedback_type, original_category, corrected_category, user_comment, user_ip))

        conn.commit()
        conn.close()

        api_bp.logger.info(f"Feedback submitted for tweet {tweet_id}: {feedback_type}")

        return jsonify({
            'success': True,
            'message': 'Feedback submitted successfully. Thank you for helping improve our analysis!'
        })

    except Exception as e:
        api_bp.logger.error(f"Error submitting feedback: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/tweet-versions/<tweet_id>')
@rate_limit(**config.get_rate_limit('api_versions'))
def get_tweet_versions(tweet_id: str) -> str:
    """API endpoint to get version history for edited tweets."""
    try:
        conn = get_db_connection()

        # Get current tweet data
        current_tweet = conn.execute("""
            SELECT
                t.content, t.username, t.tweet_timestamp, t.tweet_url,
                ca.category, ca.llm_explanation
            FROM tweets t
            LEFT JOIN content_analyses ca ON t.tweet_id = ca.post_id
            WHERE t.tweet_id = ?
        """, (tweet_id,)).fetchone()

        if not current_tweet:
            conn.close()
            return jsonify({'error': 'Tweet not found'}), 404

        # Get version history from edit_history table
        versions = conn.execute("""
            SELECT
                version_number, content, hashtags, mentions, media_links,
                media_count, external_links, detected_at
            FROM edit_history
            WHERE original_tweet_id = ?
            ORDER BY version_number DESC
        """, (tweet_id,)).fetchall()

        conn.close()

        # Format response
        version_history = []
        for version in versions:
            version_history.append({
                'version_number': version['version_number'],
                'content': version['content'],
                'hashtags': version['hashtags'],
                'mentions': version['mentions'],
                'media_links': version['media_links'],
                'media_count': version['media_count'],
                'external_links': version['external_links'],
                'detected_at': version['detected_at']
            })

        response_data = {
            'tweet_id': tweet_id,
            'current_version': {
                'content': current_tweet['content'],
                'username': current_tweet['username'],
                'tweet_timestamp': current_tweet['tweet_timestamp'],
                'tweet_url': current_tweet['tweet_url'],
                'category': current_tweet['category'],
                'llm_explanation': current_tweet['llm_explanation']
            },
            'previous_versions': version_history,
            'total_versions': len(version_history) + 1  # +1 for current version
        }

        return jsonify(response_data)

    except Exception as e:
        api_bp.logger.error(f"Error getting tweet versions for {tweet_id}: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/tweet-status/<tweet_id>')
@rate_limit(**config.get_rate_limit('api_endpoints'))
def get_tweet_status(tweet_id: str) -> str:
    """API endpoint to check if a tweet exists and get basic status."""
    try:
        conn = get_db_connection()
        
        # Check if tweet exists
        tweet = conn.execute("""
            SELECT 
                t.tweet_id, t.username, t.content,
                ca.category, ca.analysis_method
            FROM tweets t
            LEFT JOIN content_analyses ca ON t.tweet_id = ca.post_id
            WHERE t.tweet_id = ?
        """, (tweet_id,)).fetchone()
        
        conn.close()
        
        if tweet:
            return jsonify({
                'exists': True,
                'tweet_id': tweet['tweet_id'],
                'username': tweet['username'],
                'analyzed': tweet['category'] is not None,
                'category': tweet['category'],
                'analysis_method': tweet['analysis_method']
            })
        else:
            return jsonify({'exists': False}), 404
            
    except Exception as e:
        api_bp.logger.error(f"Error checking tweet status for {tweet_id}: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/usernames')
@rate_limit(**config.get_rate_limit('api_endpoints'))
def get_usernames() -> str:
    """API endpoint to get list of usernames for autocomplete."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get distinct usernames from tweets table
        cursor.execute("""
            SELECT DISTINCT username
            FROM tweets
            ORDER BY username
        """)

        usernames = [row['username'] for row in cursor.fetchall()]
        conn.close()

        return jsonify(usernames)

    except Exception as e:
        api_bp.logger.error(f"Error getting usernames: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500