"""
Repository pattern interfaces for database abstraction.
Provides platform-agnostic data access interfaces.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any, Protocol
from datetime import datetime


class RepositoryInterface(Protocol):
    """Base protocol for all repository interfaces."""

    @abstractmethod
    def get_connection(self):
        """Get database connection."""
        pass


class TweetRepositoryInterface(RepositoryInterface):
    """Interface for tweet/post data operations."""

    @abstractmethod
    def get_tweet_by_id(self, tweet_id: str) -> Optional[Dict[str, Any]]:
        """Get tweet data by ID."""
        pass

    @abstractmethod
    def get_tweets_by_username(self, username: str, limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        """Get tweets for a specific user."""
        pass

    @abstractmethod
    def get_tweet_count_by_username(self, username: str) -> int:
        """Get total tweet count for a user."""
        pass

    @abstractmethod
    def update_tweet_status(self, tweet_id: str, is_deleted: bool = None, is_edited: bool = None) -> bool:
        """Update tweet status flags."""
        pass

    @abstractmethod
    def get_recent_tweets(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recently scraped tweets."""
        pass


class ContentAnalysisRepositoryInterface(RepositoryInterface):
    """Interface for content analysis operations."""

    @abstractmethod
    def get_analysis_by_post_id(self, post_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis for a specific post."""
        pass

    @abstractmethod
    def save_analysis(self, analysis_data: Dict[str, Any]) -> bool:
        """Save or update content analysis."""
        pass

    @abstractmethod
    def update_analysis(self, post_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing analysis."""
        pass

    @abstractmethod
    def get_analyses_by_category(self, category: str, limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        """Get analyses by category."""
        pass

    @abstractmethod
    def get_analyses_by_username(self, username: str) -> List[Dict[str, Any]]:
        """Get all analyses for a user."""
        pass

    @abstractmethod
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get overall analysis statistics."""
        pass

    @abstractmethod
    def delete_analysis(self, post_id: str) -> bool:
        """Delete analysis for a post."""
        pass


class AccountRepositoryInterface(RepositoryInterface):
    """Interface for account operations."""

    @abstractmethod
    def get_account_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get account data by username."""
        pass

    @abstractmethod
    def get_all_accounts(self, limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        """Get all accounts with pagination."""
        pass

    @abstractmethod
    def save_account(self, account_data: Dict[str, Any]) -> bool:
        """Save or update account."""
        pass

    @abstractmethod
    def update_last_scraped(self, username: str, timestamp: datetime = None) -> bool:
        """Update account's last scraped timestamp."""
        pass

    @abstractmethod
    def get_accounts_with_stats(self, limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        """Get accounts with tweet and analysis statistics."""
        pass


class PostEditRepositoryInterface(RepositoryInterface):
    """Interface for post edit tracking operations."""

    @abstractmethod
    def save_edit(self, post_id: str, previous_content: str, version_number: int = None) -> bool:
        """Save a post edit record."""
        pass

    @abstractmethod
    def get_edits_by_post_id(self, post_id: str) -> List[Dict[str, Any]]:
        """Get edit history for a post."""
        pass

    @abstractmethod
    def get_latest_version(self, post_id: str) -> Optional[int]:
        """Get the latest version number for a post."""
        pass