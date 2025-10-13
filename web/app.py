"""
Main Flask application for dimetuverdad.
Uses Flask app factory pattern with blueprints for modular organization.
"""

from flask import Flask
from flask_caching import Cache
import logging
import sys
import os
from pathlib import Path

# Import blueprints
from web.routes.main import main_bp
from web.routes.admin import admin_bp
from web.routes.api import api_bp
from web.routes.loading import loading_bp

# Import utilities
from web.utils.decorators import register_error_handlers

def create_app(config_name=None):
    """Application factory pattern for Flask app creation."""
    app = Flask(__name__)

    # Load configuration
    app.config.from_object('config')

    # Initialize Flask-Caching
    cache = Cache(app, config={
        'CACHE_TYPE': app.config['CACHE_TYPE'],
        'CACHE_DEFAULT_TIMEOUT': app.config['CACHE_DEFAULT_TIMEOUT']
    })

    # Make cache available through app
    app.cache = cache

    # Register error handlers
    register_error_handlers(app)

    # Register blueprints
    app.register_blueprint(main_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(loading_bp)

    # Export helpers for test mocks
    import web.utils.helpers as helpers
    app.get_user_profile_data = helpers.get_user_profile_data
    app.get_user_tweets_data = helpers.get_user_tweets_data
    app.get_user_analysis_stats = helpers.get_user_analysis_stats
    app.get_tweet_data = helpers.get_tweet_data
    app.reanalyze_tweet = helpers.reanalyze_tweet
    
    # Add placeholder functions for route-level operations
    def get_analyzer():
        """Placeholder for analyzer access."""
        return None
    
    def reanalyze_category():
        """Placeholder for category reanalysis."""
        return None
        
    def run_user_analysis():
        """Placeholder for user analysis."""
        return None
    
    app.get_analyzer = get_analyzer
    app.reanalyze_category = reanalyze_category
    app.run_user_analysis = run_user_analysis

    # Set up logging
    _setup_logging(app)

    return app

# Module level exports for tests
import web.utils.helpers as helpers

get_user_profile_data = helpers.get_user_profile_data
get_user_tweets_data = helpers.get_user_tweets_data
get_user_analysis_stats = helpers.get_user_analysis_stats
get_tweet_data = helpers.get_tweet_data
reanalyze_tweet = helpers.reanalyze_tweet

def get_analyzer():
    """Placeholder for analyzer access."""
    return None

def reanalyze_category():
    """Placeholder for category reanalysis."""
    return None
    
def run_user_analysis():
    """Placeholder for user analysis."""
    return None

def _setup_logging(app):
    """Configure logging for the Flask application."""
    # Set up logging to stdout for development
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

    # Ensure Flask logger outputs to stdout
    app.logger.setLevel(logging.INFO)
    app.logger.addHandler(logging.StreamHandler(sys.stdout))

    # Also ensure werkzeug (Flask's underlying server) logs to stdout
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.setLevel(logging.INFO)
    werkzeug_logger.addHandler(logging.StreamHandler(sys.stdout))

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)