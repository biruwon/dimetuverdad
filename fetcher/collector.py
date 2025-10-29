"""
Tweet collection logic for the fetcher module.

Handles the core tweet collection workflow, processing individual tweets,
and managing collection state.
"""

import json
import sqlite3
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
from playwright.sync_api import Page, TimeoutError
from .config import get_config
from .logging_config import get_logger
from .scroller import get_scroller
from .media_monitor import get_media_monitor
from . import db as fetcher_db
from . import parsers as fetcher_parsers
from utils import paths
# Import repository interfaces
from repositories import get_tweet_repository

logger = get_logger('collector')

class TweetCollector:
    """Handles tweet collection operations."""

    def __init__(self):
        self.config = get_config()
        self.scroller = get_scroller()
        self.media_monitor = get_media_monitor()

    def setup_media_url_monitoring(self, page) -> List[str]:
        """
        Set up network request monitoring to capture media URLs.

        Args:
            page: Playwright page object

        Returns:
            List of captured media URLs
        """
        return self.media_monitor.setup_and_monitor(page, self.scroller)

    def should_process_tweet(self, tweet_id: str, seen_ids: Set[str]) -> bool:
        """
        Check if a tweet should be processed.

        Args:
            tweet_id: Tweet ID to check
            seen_ids: Set of already seen tweet IDs

        Returns:
            bool: True if tweet should be processed
        """
        if not tweet_id or tweet_id in seen_ids:
            return False
        return True

    def check_tweet_exists_in_db(self, username: str, tweet_id: str, resume_from_last: bool) -> Tuple[bool, Optional[Dict]]:
        """
        Check if tweet exists in database and determine if update is needed.

        Args:
            username: Twitter username
            tweet_id: Tweet ID
            resume_from_last: Whether we're resuming from last collection

        Returns:
            Tuple of (exists, needs_update_data)
        """
        if not resume_from_last:
            return False, None

        # Use repository pattern
        tweet_repo = get_tweet_repository()
        tweet = tweet_repo.get_tweet_by_id(tweet_id)
        
        exists = tweet is not None and tweet.get('username') == username
        if not exists:
            return False, None

        # Return existing data for comparison
        return True, {
            'post_type': tweet.get('post_type'),
            'content': tweet.get('content'),
            'original_author': tweet.get('original_author'),
            'original_tweet_id': tweet.get('original_tweet_id')
        }

    def extract_tweet_data(self, article, tweet_id: str, tweet_url: str, username: str, profile_pic_url: Optional[str], media_urls: List[str]) -> Optional[Dict]:
        """
        Extract comprehensive tweet data from a tweet article.

        Args:
            article: Playwright article element
            tweet_id: Tweet ID
            tweet_url: Tweet URL
            username: Username
            profile_pic_url: Profile picture URL
            media_urls: Captured media URLs from monitoring

        Returns:
            Dict of tweet data or None if extraction failed
        """
        try:
            tweet_data = fetcher_parsers.extract_tweet_with_quoted_content(article, tweet_id, username, tweet_url)
            
            if not tweet_data:
                return None
            
            # Process video URLs like refetch does
            tweet_data = self.media_monitor.process_video_urls(media_urls, tweet_data)
            
            # Extract timestamp
            time_element = article.query_selector('time')
            if time_element:
                tweet_data['tweet_timestamp'] = time_element.get_attribute('datetime')
            else:
                tweet_data['tweet_timestamp'] = None
            
            # Add profile picture URL
            tweet_data['profile_pic_url'] = profile_pic_url
            
            # Extract engagement metrics (additional data not in refetch)
            engagement = fetcher_parsers.extract_engagement_metrics(article)
            tweet_data.update({
                'engagement_retweets': engagement['retweets'],
                'engagement_likes': engagement['likes'],
                'engagement_replies': engagement['replies'],
                'engagement_views': engagement['views'],
            })

            return tweet_data

        except Exception as e:
            logger.error(f"Error extracting tweet data for {tweet_id}: {e}")
            return None

    def save_tweet_data(self, conn, tweet_data: Dict) -> bool:
        """
        Save tweet data to database.

        Args:
            conn: Database connection
            tweet_data: Tweet data dictionary

        Returns:
            bool: True if saved successfully
        """
        try:
            saved = fetcher_db.save_tweet(conn, tweet_data)
            if saved:
                post_type = tweet_data.get('post_type', 'unknown')
                tweet_id = tweet_data.get('tweet_id', 'unknown')
                logger.info(f"Saved {post_type}: {tweet_id}")
            else:
                logger.debug(f"Not saved (duplicate/unchanged): {tweet_data.get('tweet_id')}")
            return saved
        except Exception as e:
            tweet_id = tweet_data.get('tweet_id', 'unknown')
            logger.error(f"Error saving tweet {tweet_id}: {e}")
            return False

    def log_processing_error(self, tweet_id: Optional[str], username: str, error: Exception) -> None:
        """
        Log tweet processing errors to database.

        Args:
            tweet_id: Tweet ID (may be None)
            username: Username
            error: Exception that occurred
        """
        try:
            from utils.database import get_db_connection_context
            with get_db_connection_context() as conn:
                cur = conn.cursor()
                cur.execute("INSERT INTO scrape_errors (username, tweet_id, error, context, timestamp) VALUES (?, ?, ?, ?, ?)", (
                    username,
                    tweet_id if tweet_id else None,
                    str(error),
                    'processing_article',
                    datetime.now().isoformat()
                ))
                conn.commit()
        except Exception:
            pass  # Don't log logging errors

    def collect_tweets_from_page(
        self,
        page,
        username: str,
        max_tweets: int,
        resume_from_last: bool,
        oldest_timestamp: Optional[str],
        profile_pic_url: Optional[str],
        conn
    ) -> List[Dict]:
        """
        Collect tweets from a page with comprehensive error handling.

        Args:
            page: Playwright page object
            username: Twitter username
            max_tweets: Maximum tweets to collect
            resume_from_last: Whether to resume from last collection
            oldest_timestamp: Oldest timestamp for resume logic
            profile_pic_url: Profile picture URL
            conn: Database connection

        Returns:
            List of collected tweet data
        """
        collected_tweets = []
        seen_tweet_ids: Set[str] = set()
        consecutive_empty_scrolls = 0
        last_height = 0
        tweets_found_this_cycle = 0
        saved_count = 0

        # Set up media URL monitoring
        media_urls = self.setup_media_url_monitoring(page)

        if max_tweets == float('inf'):
            logger.info("Collecting unlimited tweets...")
        else:
            logger.info(f"Collecting up to {max_tweets} tweets...")

        iteration = 0

        while len(collected_tweets) < max_tweets and consecutive_empty_scrolls < self.config.max_consecutive_empty_scrolls:
            # Find tweet articles
            articles = page.query_selector_all('article[data-testid="tweet"], [data-testid="tweet"]')
            logger.debug(f"Found {len(articles)} tweet elements on page")

            tweets_found_this_cycle = 0

            # Process each article
            for article in articles:
                if len(collected_tweets) >= max_tweets:
                    break

                try:
                    # Extract basic tweet info
                    tweet_link = article.query_selector('a[href*="/status/"]')
                    if not tweet_link:
                        continue

                    href = tweet_link.get_attribute('href')
                    tweet_id = href.split('/')[-1] if href else None

                    if not self.should_process_tweet(tweet_id, seen_tweet_ids):
                        continue

                    tweet_url = f"https://x.com{href}"

                    # Check database existence and update needs
                    exists_in_db, db_data = self.check_tweet_exists_in_db(username, tweet_id, resume_from_last)

                    if exists_in_db:
                        # Check if update is needed
                        post_analysis = fetcher_parsers.analyze_post_type(article, username)
                        if post_analysis.get('should_skip'):
                            continue

                        content = fetcher_parsers.extract_full_tweet_content(article)
                        if not content:
                            logger.debug(f"Skipping content-less tweet: {tweet_id}")
                            continue

                        # Compare with database
                        needs_update = False
                        if db_data:
                            db_post_type = db_data.get('post_type')
                            db_content = db_data.get('content')
                            if db_post_type != post_analysis.get('post_type') or (content and db_content and content != db_content):
                                needs_update = True

                        if not needs_update:
                            logger.debug(f"Skipping existing tweet ({tweet_id})")
                            continue

                        logger.info(f"Tweet {tweet_id} exists but needs update")

                    # Extract full tweet data
                    tweet_data = self.extract_tweet_data(
                        article, tweet_id, tweet_url, username,
                        profile_pic_url, media_urls
                    )

                    if not tweet_data:
                        continue

                    # Check resume timestamp logic
                    if resume_from_last and tweet_data.get('tweet_timestamp') and oldest_timestamp:
                        if fetcher_parsers.should_skip_existing_tweet(tweet_data['tweet_timestamp'], oldest_timestamp):
                            logger.debug(f"Skipping already covered tweet ({tweet_data['tweet_timestamp']})")
                            continue

                    # Save tweet
                    if self.save_tweet_data(conn, tweet_data):
                        saved_count += 1
                        if exists_in_db:
                            logger.info(f"Updated {tweet_data.get('post_type', 'unknown')}: {tweet_id}")
                        else:
                            logger.info(f"Saved {tweet_data.get('post_type', 'unknown')}: {tweet_id}")

                    collected_tweets.append(tweet_data)
                    seen_tweet_ids.add(tweet_id)
                    tweets_found_this_cycle += 1

                except Exception as e:
                    self.log_processing_error(tweet_id if 'tweet_id' in locals() else None, username, e)
                    logger.error(f"Error processing tweet: {e}")
                    continue

            # Handle scrolling and progress tracking
            current_tweet_count = len(collected_tweets)

            if len(collected_tweets) < max_tweets:
                if max_tweets == float('inf'):
                    logger.info(f"Scrolling for more tweets... ({current_tweet_count}/∞) - Found {tweets_found_this_cycle} this cycle")
                else:
                    logger.info(f"Scrolling for more tweets... ({current_tweet_count}/{max_tweets}) - Found {tweets_found_this_cycle} this cycle")

                # Track consecutive empty scrolls
                if tweets_found_this_cycle == 0:
                    consecutive_empty_scrolls += 1
                    logger.warning(f"No new tweets found ({consecutive_empty_scrolls}/{self.config.max_consecutive_empty_scrolls} consecutive empty cycles)")

                    # Try recovery strategies
                    if consecutive_empty_scrolls % 5 == 0 and consecutive_empty_scrolls <= 10:
                        recovery_attempt = consecutive_empty_scrolls // 5
                        if self.scroller.try_recovery_strategies(page, recovery_attempt):
                            consecutive_empty_scrolls = max(0, consecutive_empty_scrolls - 3)
                            logger.info("Recovery successful, continuing...")
                        else:
                            logger.warning("Recovery failed")
                else:
                    consecutive_empty_scrolls = 0
                    logger.info(f"Found {tweets_found_this_cycle} new tweets, continuing...")

                iteration += 1

                try:
                    # Use aggressive scrolling when needed
                    if consecutive_empty_scrolls > 5:
                        self.scroller.aggressive_scroll(page, consecutive_empty_scrolls)
                    else:
                        self.scroller.event_scroll_cycle(page, iteration)
                except Exception:
                    try:
                        page.evaluate("window.scrollBy(0, 1000)")
                        self.scroller.human_delay(2.0, 3.0)
                    except Exception:
                        logger.error("All scrolling methods failed")
                        consecutive_empty_scrolls += 2

                # Check page height
                try:
                    last_height = self.scroller.check_page_height_change(page, last_height)
                except Exception as e:
                    logger.warning(f"Failed to check page height change: {e}")
                    # Assume no height change if we can't check it
                    last_height = last_height

            last_tweet_count = current_tweet_count

        # Log completion status
        if consecutive_empty_scrolls >= self.config.max_consecutive_empty_scrolls:
            logger.warning(f"Stopped: Twitter stopped serving new content after {consecutive_empty_scrolls} empty scroll cycles")
            logger.info("This is likely Twitter's content serving limit")
        elif len(collected_tweets) >= max_tweets:
            logger.info(f"Completed: Reached target tweet count ({max_tweets})")
        else:
            logger.info("Stopped: Collection completed")

        logger.info(f"Final count: {len(collected_tweets)} tweets processed, {saved_count} saved to database from @{username}")

        return collected_tweets

# Global collector instance
collector = TweetCollector()

def get_collector() -> TweetCollector:
    """Get the global collector instance."""
    return collector