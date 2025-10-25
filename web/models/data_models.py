"""
Data models for dimetuverdad Flask application.
Provides structured data handling and validation for tweets, analyses, and user interactions.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import json


@dataclass
class Tweet:
    """Represents a tweet with all its metadata and content."""
    tweet_id: str
    content: str
    username: str
    tweet_url: str
    tweet_timestamp: str
    media_count: int = 0
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    media_links: Optional[str] = None
    post_type: str = "original"
    is_deleted: bool = False
    is_edited: bool = False
    rt_original_analyzed: bool = False
    original_author: Optional[str] = None
    original_tweet_id: Optional[str] = None
    reply_to_username: Optional[str] = None
    original_content: Optional[str] = None

    @classmethod
    def from_row(cls, row) -> 'Tweet':
        """Create Tweet instance from database row."""
        return cls(
            tweet_id=row['tweet_id'],
            content=row['content'] or '',
            username=row['username'] or '',
            tweet_url=row['tweet_url'] or '',
            tweet_timestamp=row['tweet_timestamp'] or '',
            media_count=row['media_count'] or 0,
            hashtags=json.loads(row['hashtags']) if row['hashtags'] else [],
            mentions=json.loads(row['mentions']) if row['mentions'] else [],
            media_links=row['media_links'],
            post_type=row['post_type'] or 'original',
            is_deleted=bool(row['is_deleted']) if row['is_deleted'] is not None else False,
            is_edited=bool(row['is_edited']) if row['is_edited'] is not None else False,
            rt_original_analyzed=bool(row['rt_original_analyzed']) if row['rt_original_analyzed'] is not None else False,
            original_author=row['original_author'],
            original_tweet_id=row['original_tweet_id'],
            reply_to_username=row['reply_to_username'],
            original_content=row['original_content']
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for template rendering."""
        return {
            'tweet_url': self.tweet.tweet_url,
            'content': self.tweet.content,
            'media_links': self.tweet.media_links,
            'hashtags_parsed': self.tweet.hashtags,
            'mentions_parsed': self.tweet.mentions,
            'tweet_timestamp': self.tweet.tweet_timestamp,
            'post_type': self.tweet.post_type,
            'tweet_id': self.tweet.tweet_id,
            'analysis_category': self.category,
            'local_explanation': self.local_explanation,
            'external_explanation': self.external_explanation,
            'best_explanation': self.best_explanation,
            'has_dual_explanations': self.has_dual_explanations,
            'analysis_stages': self.analysis_stages,
            'external_analysis_used': self.external_analysis_used,
            'analysis_timestamp': self.analysis.analysis_timestamp if self.analysis else '',
            'categories_detected': self.categories_detected,
            'multimodal_analysis': self.analysis.multimodal_analysis if self.analysis else False,
            'is_deleted': self.tweet.is_deleted,
            'is_edited': self.tweet.is_edited,
            'rt_original_analyzed': self.tweet.rt_original_analyzed,
            'original_author': self.tweet.original_author,
            'original_tweet_id': self.tweet.original_tweet_id,
            'reply_to_username': self.tweet.reply_to_username,
            'post_status_warnings': self.post_status_warnings,
            'is_rt': self.is_rt,
            'rt_type': self.rt_type,
        }


@dataclass
class ContentAnalysis:
    """Represents the analysis results for a tweet with dual explanation architecture."""
    post_id: str
    category: str
    local_explanation: str
    author_username: str
    analysis_timestamp: str
    external_explanation: Optional[str] = ''
    analysis_stages: Optional[str] = ''
    external_analysis_used: bool = False
    categories_detected: Optional[str] = None
    multimodal_analysis: bool = False

    @classmethod
    def from_row(cls, row) -> 'ContentAnalysis':
        """Create ContentAnalysis instance from database row."""
        # Parse media_urls from JSON if present
        media_urls = []
        if row.get('media_urls'):
            try:
                media_urls = json.loads(row['media_urls'])
            except (json.JSONDecodeError, TypeError):
                media_urls = []
        
        return cls(
            post_id=row['post_id'],
            category=row['category'] or 'general',
            local_explanation=row['local_explanation'] or '',
            external_explanation=row['external_explanation'] or '',
            analysis_stages=row['analysis_stages'] or '',
            external_analysis_used=bool(row['external_analysis_used']) if row['external_analysis_used'] is not None else False,
            author_username=row['author_username'] or '',
            analysis_timestamp=row['analysis_timestamp'] or '',
            categories_detected=row['categories_detected'],
            multimodal_analysis=bool(media_urls)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'post_id': self.post_id,
            'category': self.category,
            'local_explanation': self.local_explanation,
            'external_explanation': self.external_explanation,
            'analysis_stages': self.analysis_stages,
            'external_analysis_used': self.external_analysis_used,
            'author_username': self.author_username,
            'analysis_timestamp': self.analysis_timestamp,
            'categories_detected': self.categories_detected,
            'multimodal_analysis': self.multimodal_analysis
        }

    @property
    def categories_list(self) -> List[str]:
        """Get list of detected categories."""
        if self.categories_detected:
            try:
                return json.loads(self.categories_detected)
            except (json.JSONDecodeError, TypeError):
                pass
        return [self.category] if self.category else []
    
    @property
    def best_explanation(self) -> str:
        """Get the best available explanation (prefers external over local)."""
        if self.external_explanation and len(self.external_explanation.strip()) > 0:
            return self.external_explanation
        return self.local_explanation
    
    @property
    def has_dual_explanations(self) -> bool:
        """Check if both local and external explanations are available."""
        return (
            len(self.local_explanation.strip()) > 0 and 
            len(self.external_explanation.strip()) > 0
        )


@dataclass
class Account:
    """Represents a Twitter/X account."""
    username: str
    profile_pic_url: Optional[str] = None
    total_tweets: int = 0
    analyzed_tweets: int = 0
    problematic_posts: int = 0
    last_activity: Optional[str] = None

    @classmethod
    def from_row(cls, row) -> 'Account':
        """Create Account instance from database row."""
        return cls(
            username=row['username'] or '',
            profile_pic_url=row['profile_pic_url'],
            total_tweets=row['tweet_count'] or 0,
            analyzed_tweets=row['analyzed_posts'] or 0,
            problematic_posts=row['problematic_posts'] or 0,
            last_activity=row['last_activity']
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'username': self.username,
            'profile_pic_url': self.profile_pic_url,
            'total_tweets': self.total_tweets,
            'analyzed_tweets': self.analyzed_tweets,
            'problematic_posts': self.problematic_posts,
            'last_activity': self.last_activity
        }


@dataclass
class UserFeedback:
    """Represents user feedback on analysis accuracy."""
    tweet_id: str
    feedback_type: str
    original_category: Optional[str] = None
    corrected_category: Optional[str] = None
    user_comment: str = ""
    user_ip: Optional[str] = None
    submitted_at: Optional[str] = None

    @classmethod
    def from_row(cls, row) -> 'UserFeedback':
        """Create UserFeedback instance from database row."""
        return cls(
            tweet_id=row['tweet_id'],
            feedback_type=row['feedback_type'] or 'correction',
            original_category=row['original_category'],
            corrected_category=row['corrected_category'],
            user_comment=row['user_comment'] or '',
            user_ip=row['user_ip'],
            submitted_at=row['submitted_at']
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'tweet_id': self.tweet_id,
            'feedback_type': self.feedback_type,
            'original_category': self.original_category,
            'corrected_category': self.corrected_category,
            'user_comment': self.user_comment,
            'user_ip': self.user_ip,
            'submitted_at': self.submitted_at
        }


@dataclass
class AnalysisStatistics:
    """Represents analysis statistics for accounts or categories."""
    total_tweets: int = 0
    analyzed_tweets: int = 0
    unique_users: int = 0
    category_breakdown: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_category_stats(cls, row) -> 'AnalysisStatistics':
        """Create from category statistics query result."""
        return cls(
            total_tweets=row.get('total_tweets', 0),
            unique_users=row.get('unique_users', 0)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'total_tweets': self.total_tweets,
            'analyzed_tweets': self.analyzed_tweets,
            'unique_users': self.unique_users,
            'category_breakdown': self.category_breakdown
        }


@dataclass
class TweetDisplay:
    """Represents a tweet with all display-related data."""
    tweet: Tweet
    analysis: Optional[ContentAnalysis] = None

    @property
    def category(self) -> str:
        """Get the analysis category."""
        return self.analysis.category if self.analysis else 'general'

    @property
    def local_explanation(self) -> str:
        """Get the local LLM explanation."""
        return self.analysis.local_explanation if self.analysis else ''
    
    @property
    def external_explanation(self) -> str:
        """Get the external LLM explanation."""
        return self.analysis.external_explanation if self.analysis else ''
    
    @property
    def best_explanation(self) -> str:
        """Get the best available explanation."""
        return self.analysis.best_explanation if self.analysis else ''
    
    @property
    def has_dual_explanations(self) -> bool:
        """Check if both local and external explanations are available."""
        return self.analysis.has_dual_explanations if self.analysis else False

    @property
    def analysis_stages(self) -> str:
        """Get the analysis stages."""
        return self.analysis.analysis_stages if self.analysis else ''
    
    @property
    def external_analysis_used(self) -> bool:
        """Check if external analysis was used."""
        return self.analysis.external_analysis_used if self.analysis else False

    @property
    def categories_detected(self) -> List[str]:
        """Get list of detected categories."""
        return self.analysis.categories_list if self.analysis else []

    @property
    def has_multiple_categories(self) -> bool:
        """Check if tweet has multiple detected categories."""
        return len(self.categories_detected) > 1

    @property
    def is_rt(self) -> bool:
        """Check if this is a retweet."""
        return self.tweet.post_type in ['repost_other', 'repost_own', 'repost_reply']

    @property
    def rt_type(self) -> Optional[str]:
        """Get retweet type."""
        return self.tweet.post_type if self.is_rt else None

    @property
    def post_status_warnings(self) -> List[Dict[str, str]]:
        """Get post status warnings."""
        warnings = []
        if self.tweet.is_deleted:
            warnings.append({
                'type': 'deleted',
                'message': 'Este tweet fue eliminado',
                'icon': 'fas fa-trash',
                'class': 'alert-danger'
            })
        if self.tweet.is_edited:
            warnings.append({
                'type': 'edited',
                'message': 'Este tweet fue editado',
                'icon': 'fas fa-edit',
                'class': 'alert-warning'
            })
        return warnings

    @property
    def analysis_display(self) -> str:
        """Get the appropriate analysis text for display."""
        # Prefer external explanation if available, otherwise use local
        return self.best_explanation or "Sin análisis disponible"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for template rendering."""
        return {
            'tweet_url': self.tweet.tweet_url,
            'content': self.tweet.content,
            'media_links': self.tweet.media_links,
            'hashtags_parsed': self.tweet.hashtags,
            'mentions_parsed': self.tweet.mentions,
            'tweet_timestamp': self.tweet.tweet_timestamp,
            'post_type': self.tweet.post_type,
            'tweet_id': self.tweet.tweet_id,
            'analysis_category': self.category,
            'local_explanation': self.local_explanation,
            'external_explanation': self.external_explanation,
            'best_explanation': self.best_explanation,
            'has_dual_explanations': self.has_dual_explanations,
            'analysis_stages': self.analysis_stages,
            'external_analysis_used': self.external_analysis_used,
            'analysis_timestamp': self.analysis.analysis_timestamp if self.analysis else '',
            'categories_detected': self.categories_detected,
            'multimodal_analysis': self.analysis.multimodal_analysis if self.analysis else False,
            'is_deleted': self.tweet.is_deleted,
            'is_edited': self.tweet.is_edited,
            'rt_original_analyzed': self.tweet.rt_original_analyzed,
            'original_author': self.tweet.original_author,
            'original_tweet_id': self.tweet.original_tweet_id,
            'reply_to_username': self.tweet.reply_to_username,
            'post_status_warnings': self.post_status_warnings,
            'is_rt': self.is_rt,
            'rt_type': self.rt_type,
            'analysis_display': self.analysis_display,
            'category': self.category,
            'has_multiple_categories': self.has_multiple_categories
        }