"""
Repository factory and dependency injection container.
Provides centralized access to repository instances.
"""

from typing import Optional
from .interfaces import (
    TweetRepositoryInterface,
    ContentAnalysisRepositoryInterface,
    AccountRepositoryInterface,
    PostEditRepositoryInterface
)
from .sqlite_impl import (
    SQLiteTweetRepository,
    SQLiteContentAnalysisRepository,
    SQLiteAccountRepository,
    SQLitePostEditRepository
)


class RepositoryFactory:
    """Factory for creating repository instances."""

    def __init__(self, connection_factory=None):
        """Initialize with optional connection factory."""
        self._connection_factory = connection_factory
        self._tweet_repo: Optional[TweetRepositoryInterface] = None
        self._content_analysis_repo: Optional[ContentAnalysisRepositoryInterface] = None
        self._account_repo: Optional[AccountRepositoryInterface] = None
        self._post_edit_repo: Optional[PostEditRepositoryInterface] = None

    def get_tweet_repository(self) -> TweetRepositoryInterface:
        """Get tweet repository instance."""
        if self._tweet_repo is None:
            self._tweet_repo = SQLiteTweetRepository(self._connection_factory)
        return self._tweet_repo

    def get_content_analysis_repository(self) -> ContentAnalysisRepositoryInterface:
        """Get content analysis repository instance."""
        if self._content_analysis_repo is None:
            self._content_analysis_repo = SQLiteContentAnalysisRepository(self._connection_factory)
        return self._content_analysis_repo

    def get_account_repository(self) -> AccountRepositoryInterface:
        """Get account repository instance."""
        if self._account_repo is None:
            self._account_repo = SQLiteAccountRepository(self._connection_factory)
        return self._account_repo

    def get_post_edit_repository(self) -> PostEditRepositoryInterface:
        """Get post edit repository instance."""
        if self._post_edit_repo is None:
            self._post_edit_repo = SQLitePostEditRepository(self._connection_factory)
        return self._post_edit_repo


# Global repository factory instance
_repository_factory = RepositoryFactory()

def get_tweet_repository() -> TweetRepositoryInterface:
    """Get global tweet repository instance."""
    return _repository_factory.get_tweet_repository()

def get_content_analysis_repository() -> ContentAnalysisRepositoryInterface:
    """Get global content analysis repository instance."""
    return _repository_factory.get_content_analysis_repository()

def get_account_repository() -> AccountRepositoryInterface:
    """Get global account repository instance."""
    return _repository_factory.get_account_repository()

def get_post_edit_repository() -> PostEditRepositoryInterface:
    """Get global post edit repository instance."""
    return _repository_factory.get_post_edit_repository()