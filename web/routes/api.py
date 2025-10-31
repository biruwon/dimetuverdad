"""
API routes for the dimetuverdad web application.
Contains REST API endpoints for external integrations.
"""

from flask import Blueprint, request, jsonify
from web.utils.decorators import rate_limit
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
        from utils.database import get_db_connection_context
        with get_db_connection_context() as conn:
            tweet_exists = conn.execute("""
                SELECT tweet_id FROM tweets WHERE tweet_id = ?
            """, (tweet_id,)).fetchone()

            if not tweet_exists:
                return jsonify({'error': 'Tweet not found'}), 404

            # Check for recent feedback from this IP for this tweet (rate limiting)
            recent_feedback = conn.execute("""
                SELECT id FROM user_feedback
                WHERE post_id = ? AND user_ip = ? AND submitted_at > datetime('now', '-1 hour')
            """, (tweet_id, user_ip)).fetchone()

            if recent_feedback:
                return jsonify({'error': 'Feedback already submitted recently for this tweet'}), 429

            # Insert feedback
            conn.execute("""
                INSERT INTO user_feedback (post_id, feedback_type, original_category, corrected_category, user_comment, user_ip)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (tweet_id, feedback_type, original_category, corrected_category, user_comment, user_ip))

            conn.commit()

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
        from utils.database import get_db_connection_context
        with get_db_connection_context() as conn:
            # Get current tweet data
            current_tweet = conn.execute("""
                SELECT
                    t.content, t.username, t.tweet_timestamp, t.tweet_url,
                    ca.category, ca.local_explanation, ca.external_explanation,
                    ca.analysis_stages, ca.external_analysis_used
                FROM tweets t
                LEFT JOIN content_analyses ca ON t.tweet_id = ca.post_id
                WHERE t.tweet_id = ?
            """, (tweet_id,)).fetchone()

            if not current_tweet:
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
                'local_explanation': current_tweet['local_explanation'],
                'external_explanation': current_tweet['external_explanation'],
                'analysis_stages': current_tweet['analysis_stages'],
                'external_analysis_used': current_tweet['external_analysis_used']
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
        from utils.database import get_db_connection_context
        with get_db_connection_context() as conn:
            # Check if tweet exists
            tweet = conn.execute("""
                SELECT 
                    t.tweet_id, t.username, t.content,
                    ca.category, ca.analysis_stages, ca.external_analysis_used
                FROM tweets t
                LEFT JOIN content_analyses ca ON t.tweet_id = ca.post_id
                WHERE t.tweet_id = ?
            """, (tweet_id,)).fetchone()
        
        if tweet:
            return jsonify({
                'exists': True,
                'tweet_id': tweet['tweet_id'],
                'username': tweet['username'],
                'analyzed': tweet['category'] is not None,
                'category': tweet['category'],
                'analysis_stages': tweet['analysis_stages'],
                'external_analysis_used': tweet['external_analysis_used']
            })
        else:
            return jsonify({'exists': False}), 404
            
    except Exception as e:
        api_bp.logger.error(f"Error checking tweet status for {tweet_id}: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/reanalyze-multi/<post_id>', methods=['POST'])
@rate_limit(**config.get_rate_limit('api_endpoints'))
def reanalyze_multi_model(post_id: str) -> str:
    """API endpoint to re-run multi-model analysis for a specific post."""
    try:
        # Import here to avoid circular imports
        import subprocess
        import sys
        import os
        
        # Check if post exists
        from utils.database import get_db_connection_context
        with get_db_connection_context() as conn:
            tweet_exists = conn.execute("""
                SELECT tweet_id FROM tweets WHERE tweet_id = ?
            """, (post_id,)).fetchone()

            if not tweet_exists:
                return jsonify({'error': 'Post not found'}), 404

        # Run the multi-model analysis script
        try:
            # Get the project root directory
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            # Run the analysis script
            cmd = [
                sys.executable, 
                os.path.join(project_root, 'scripts', 'analyze_multi_model.py'),
                '--post-id', post_id,
                '--force-reanalyze'
            ]
            
            api_bp.logger.info(f"Running multi-model reanalysis for post {post_id}")
            
            # Run the command and capture output
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                api_bp.logger.info(f"Multi-model reanalysis completed for post {post_id}")
                return jsonify({
                    'success': True,
                    'message': 'Multi-model analysis completed successfully',
                    'output': result.stdout[-500:]  # Last 500 chars of output
                })
            else:
                error_msg = result.stderr or result.stdout
                api_bp.logger.error(f"Multi-model reanalysis failed for post {post_id}: {error_msg}")
                return jsonify({
                    'success': False,
                    'error': f'Analysis failed: {error_msg[-200:]}'
                }), 500
                
        except subprocess.TimeoutExpired:
            api_bp.logger.error(f"Multi-model reanalysis timed out for post {post_id}")
            return jsonify({'error': 'Analysis timed out'}), 504
        except Exception as e:
            api_bp.logger.error(f"Error running multi-model reanalysis for {post_id}: {str(e)}")
            return jsonify({'error': f'Failed to run analysis: {str(e)}'}), 500

    except Exception as e:
        api_bp.logger.error(f"Error in reanalyze-multi endpoint for {post_id}: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500