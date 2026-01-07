"""
Database module for dimetuverdad.
Provides centralized access to database connections and repository interfaces.
"""

# Re-export connection functions from database module
from .database import (
    get_db_connection,
    get_db_connection_context,
    init_test_database,
    cleanup_test_database,
    create_fresh_database_schema,
    DatabaseConfig,
    get_tweet_data,
    cleanup_test_databases,
    ensure_schema_up_to_date,
)

# Re-export repository interfaces and factory
from .repositories import (
    get_tweet_repository,
    get_content_analysis_repository,
    get_account_repository,
    get_post_edit_repository,
    TweetRepositoryInterface,
    ContentAnalysisRepositoryInterface,
    AccountRepositoryInterface,
    PostEditRepositoryInterface
)

# Re-export multi-model database functions
from .database_multi_model import (
    save_model_analysis,
    get_model_analyses,
    get_model_consensus,
    get_model_performance_stats,
    get_model_agreement_stats,
    update_consensus_in_content_analyses,
    get_posts_for_multi_model_analysis
)

__all__ = [
    # Connection functions
    'get_db_connection',
    'get_db_connection_context',
    'init_test_database',
    'cleanup_test_database',
    'create_fresh_database_schema',
    'DatabaseConfig',
    'get_tweet_data',
    'cleanup_test_databases',

    # Repository interfaces
    'TweetRepositoryInterface',
    'ContentAnalysisRepositoryInterface',
    'AccountRepositoryInterface',
    'PostEditRepositoryInterface',

    # Repository factory functions
    'get_tweet_repository',
    'get_content_analysis_repository',
    'get_account_repository',
    'get_post_edit_repository',

    # Multi-model functions
    'save_model_analysis',
    'get_model_analyses',
    'get_model_consensus',
    'get_model_performance_stats',
    'get_model_agreement_stats',
    'update_consensus_in_content_analyses',
    'get_posts_for_multi_model_analysis'
]