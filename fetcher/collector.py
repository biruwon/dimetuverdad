"""
Tweet collection logic for the fetcher module.

Handles the core tweet collection workflow, processing individual tweets,
and managing collection state.
"""

from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
from .config import get_config
from .logging_config import get_logger
from .scroller import get_scroller
from .media_monitor import get_media_monitor
from .thread_detector import ThreadDetector, _sync_handle
from . import db as fetcher_db
from . import parsers as fetcher_parsers
from .session_manager import SessionManager
from database.repositories import get_tweet_repository

logger = get_logger('collector')

class TweetCollector:
    """Handles tweet collection operations."""

    def __init__(self):
        self.config = get_config()
        self.scroller = get_scroller()
        self.media_monitor = get_media_monitor()
        self.thread_detector = ThreadDetector()

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

    def extract_tweet_data(self, page, article, tweet_id: str, tweet_url: str, username: str, profile_pic_url: Optional[str]) -> Optional[Dict]:
        """
        Extract comprehensive tweet data from a tweet article using the same approach as refetch.
        This ensures consistent media extraction across regular fetch and refetch operations.

        Args:
            page: Playwright page object (for media monitoring)
            article: Playwright article element
            tweet_id: Tweet ID
            tweet_url: Tweet URL
            username: Username
            profile_pic_url: Profile picture URL

        Returns:
            Dict of tweet data or None if extraction failed
        """
        try:
            # Use the same extraction method as refetch_manager for consistency
            # Pass article element to ensure we extract the correct tweet from timeline
            tweet_data = fetcher_parsers.extract_tweet_with_media_monitoring(
                page, tweet_id, username, tweet_url, self.media_monitor, self.scroller, article
            )
            
            if not tweet_data:
                return None
            
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

            # Extract thread-related metadata inline
            try:
                article_handle = _sync_handle(article)
                
                # Check for thread continuation line (CSS indicator r-1bimlpy)
                has_thread_line = self.thread_detector._has_thread_line(article_handle)
                tweet_data['has_thread_line'] = has_thread_line
                
                # Extract reply-to tweet ID if this is a reply
                reply_to_id = self.thread_detector._extract_reply_to_id(article_handle)
                if reply_to_id:
                    tweet_data['reply_to_tweet_id'] = reply_to_id
                
                # If tweet has thread line, mark as potential thread member
                if has_thread_line:
                    # The thread_id and conversation_id will be determined during
                    # post-collection thread grouping, but mark it for now
                    tweet_data['is_thread_start'] = 0  # Not the start if it has a line above
                    logger.debug(f"Tweet {tweet_id} has thread continuation line")
                    
            except Exception as thread_err:
                logger.debug(f"Could not extract thread metadata for {tweet_id}: {thread_err}")

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
            from database import get_db_connection_context
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

    def _group_and_update_threads(self, conn, collected_tweets: List[Dict], username: str) -> int:
        """
        Group collected tweets into threads and update database with thread metadata.
        
        Uses reply chain relationships and thread line indicators to detect threads,
        then updates the database with thread_id, conversation_id, thread_position,
        and is_thread_start for each tweet in a thread.
        
        Args:
            conn: Database connection
            collected_tweets: List of tweet dicts from collection
            username: Username being collected
            
        Returns:
            Number of threads detected and updated
        """
        # Filter tweets that have thread indicators
        tweets_with_thread_info = [
            t for t in collected_tweets 
            if t.get('has_thread_line') or t.get('reply_to_tweet_id')
        ]
        
        if not tweets_with_thread_info:
            logger.debug("No thread indicators found in collected tweets")
            return 0
        
        # Use thread detector's grouping logic
        try:
            threads = self.thread_detector._group_into_threads_by_reply_chain(
                collected_tweets, username
            )
        except Exception as e:
            logger.warning(f"Thread grouping algorithm failed: {e}")
            return 0
        
        if not threads:
            return 0
        
        threads_updated = 0
        cursor = conn.cursor()
        
        for thread in threads:
            thread_tweets = thread.get('tweets', [])
            if len(thread_tweets) < 2:
                continue
            
            # Use the first tweet's ID as thread_id and conversation_id
            thread_start_id = thread_tweets[0].get('tweet_id')
            if not thread_start_id:
                continue
            
            # Update each tweet in the thread
            for position, tweet in enumerate(thread_tweets):
                tweet_id = tweet.get('tweet_id')
                if not tweet_id:
                    continue
                
                is_start = 1 if position == 0 else 0
                
                try:
                    cursor.execute("""
                        UPDATE tweets 
                        SET thread_id = ?, 
                            conversation_id = ?,
                            thread_position = ?,
                            is_thread_start = ?
                        WHERE tweet_id = ?
                    """, (thread_start_id, thread_start_id, position, is_start, tweet_id))
                except Exception as e:
                    logger.debug(f"Failed to update thread info for {tweet_id}: {e}")
            
            threads_updated += 1
            logger.debug(f"Updated thread {thread_start_id} with {len(thread_tweets)} tweets")
        
        try:
            conn.commit()
        except Exception as e:
            logger.warning(f"Failed to commit thread updates: {e}")
        
        return threads_updated

    def _maybe_detect_threads(self, page, username: str, conn, scrolls_since_last_detect: int):
        """Run inline thread detection if enough scroll cycles have passed.

        Returns a tuple (detected_count, new_scrolls_since_last_detect)
        """
        # Skip detection if feature flag disabled
        if not getattr(self.config, 'collect_threads', False):
            return 0, scrolls_since_last_detect

        if scrolls_since_last_detect < self.config.thread_detect_interval:
            return 0, scrolls_since_last_detect

        try:
            session_manager = SessionManager()
            detected = self.detect_and_save_threads(
                username=username,
                session_manager=session_manager,
                conn=conn,
                max_timeline_scrolls=self.config.max_consecutive_empty_scrolls * 2,
                existing_context=page.context if hasattr(page, 'context') else None
            )
            if detected:
                logger.info(f"  🧵 Detected {detected} threads during collection (intermediate pass)")
            detected_count = int(detected or 0)
            return detected_count, 0
        except Exception as e:
            logger.debug("Intermediate thread detection failed: %s", e)
            return 0, 0

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
        processed_thread_ids: Set[str] = set()
        consecutive_empty_scrolls = 0
        last_height = 0
        tweets_found_this_cycle = 0
        saved_count = 0
        scrolls_since_last_detect = 0

        if max_tweets == float('inf'):
            logger.info("Collecting unlimited tweets...")
        else:
            logger.info(f"Collecting up to {max_tweets} tweets...")

        iteration = 0
        timeline_url = f"https://x.com/{username}"

        # Attempt recovery if Twitter returns the 'Something went wrong' page
        try:
            try_text = ''
            try:
                try_text = page.locator('main').inner_text() or ''
            except Exception:
                try_text = ''

            if 'Something went wrong' in try_text or 'Try reloading' in try_text:
                logger.warning("Twitter served 'Something went wrong' page; attempting recovery")
                recovered = False

                # Try a series of safer recovery strategies (no full reload as first option)
                for attempt in range(1, 6):
                    try:
                        if self.scroller.try_recovery_strategies(page, attempt):
                            recovered = True
                            break
                    except Exception:
                        # Continue to next strategy if one fails
                        pass

                if not recovered:
                    # Try login as a last resort when creds are present
                    try:
                        if self.config.username and self.config.password:
                            logger.info('Attempting login as part of recovery')
                            login_ok = SessionManager().login_and_save_session(page, self.config.username, self.config.password)
                            if login_ok:
                                try:
                                    page.goto(timeline_url, wait_until='domcontentloaded', timeout=20000)
                                    page.wait_for_selector('article[data-testid="tweet"]', timeout=15000)
                                    recovered = True
                                except Exception:
                                    recovered = False
                    except Exception as e:
                        logger.warning(f'Login recovery attempt failed: {e}')

                    if not recovered:
                        logger.warning('Recovery attempts failed; continuing (may be rate-limited)')
        except Exception:
            pass

        post_page_cycles = 0  # Track how many cycles we've been on a post page
        
        while len(collected_tweets) < max_tweets and consecutive_empty_scrolls < self.config.max_consecutive_empty_scrolls:
            # Check if we're on a post page instead of timeline
            current_url = page.url
            if '/status/' in current_url and username.lower() not in current_url.lower():
                post_page_cycles += 1
                # Only navigate back if we've been stuck on post page for 3+ cycles
                # This avoids reacting to temporary URL states
                if post_page_cycles >= 3:
                    logger.warning(f"Stuck on post page for {post_page_cycles} cycles ({current_url}), navigating back to timeline...")
                    try:
                        page.goto(timeline_url, wait_until="domcontentloaded", timeout=30000)
                        page.wait_for_selector('article[data-testid="tweet"]', timeout=15000)
                        logger.info(f"✅ Returned to timeline: {timeline_url}")
                        post_page_cycles = 0
                    except Exception as nav_err:
                        logger.error(f"Failed to navigate back to timeline: {nav_err}")
                        break  # Can't recover, stop collection
                else:
                    logger.debug(f"On post page ({current_url}), waiting before navigating back (cycle {post_page_cycles}/3)")
            else:
                post_page_cycles = 0  # Reset counter when on timeline
            
            # Find tweet articles
            articles = page.query_selector_all('article[data-testid="tweet"], [data-testid="tweet"]')
            logger.debug(f"Found {len(articles)} tweet elements on page")

            # Debug: collect tweet_ids found this cycle
            found_ids = []
            for article in articles:
                try:
                    tweet_link = article.query_selector('a[href*="/status/"]')
                    if tweet_link:
                        href = tweet_link.get_attribute('href')
                        if href:
                            should_process, actual_author, tweet_id = fetcher_parsers.should_process_tweet_by_author(href, username)
                            if should_process:
                                found_ids.append(tweet_id)
                except Exception:
                    pass
            logger.debug(f"DEBUG: Found tweet IDs this cycle: {found_ids[:10]}... (total {len(found_ids)})")

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
                    if not href:
                        continue
                    
                    # Parse author and tweet_id from URL using shared utility
                    should_process, actual_author, tweet_id = fetcher_parsers.should_process_tweet_by_author(href, username)
                    
                    if not should_process:
                        logger.debug(f"Skipping tweet from @{actual_author} (not target user @{username})")
                        continue

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
                        page, article, tweet_id, tweet_url, actual_author,
                        profile_pic_url
                    )

                    if not tweet_data:
                        continue

                    # Check resume timestamp logic
                    if resume_from_last and tweet_data.get('tweet_timestamp') and oldest_timestamp:
                        if fetcher_parsers.should_skip_existing_tweet(tweet_data['tweet_timestamp'], oldest_timestamp):
                            logger.debug(f"Skipping already covered tweet ({tweet_data['tweet_timestamp']})")
                            continue

                    # If thread collection is enabled, and this tweet shows a thread connector, collect the full thread first
                    if getattr(self.config, 'collect_threads', False) and tweet_data.get('has_thread_line') and tweet_id not in processed_thread_ids and tweet_id not in seen_tweet_ids:
                        logger.info(f"Thread connector detected at {tweet_id} — collecting thread")
                        try:
                            session_manager = SessionManager()
                            # Reuse current page's context to avoid creating nested Playwright instances
                            existing_context = None
                            try:
                                existing_context = page.context
                            except Exception:
                                existing_context = None

                            thread_summary = self.thread_detector.collect_thread_by_id(
                                username, tweet_id, session_manager, existing_context=existing_context
                            )
                            if thread_summary and thread_summary.tweets:
                                saved_threads = self.thread_detector.save_thread_to_database(thread_summary, conn, username)
                                # Add collected thread tweets to our local lists and seen ids
                                for t in thread_summary.tweets:
                                    tid = t.get('tweet_id')
                                    if tid and tid not in seen_tweet_ids:
                                        collected_tweets.append(t)
                                        seen_tweet_ids.add(tid)
                                processed_thread_ids.add(thread_summary.start_id)
                                saved_count += saved_threads
                                logger.info(f"Saved {saved_threads} tweets from thread starting at {thread_summary.start_id}")
                                # Skip saving the individual tweet_data below to avoid dupes
                                continue
                            else:
                                # No thread found or extraction aborted; mark this tweet as processed
                                # to avoid repeated attempts reopening the same thread in this run.
                                processed_thread_ids.add(tweet_id)
                                logger.debug(f"No thread collected for {tweet_id}; marking as processed to avoid retries")
                        except Exception as e:
                            logger.warning(f"Thread collection for {tweet_id} failed: {e}")
                            processed_thread_ids.add(tweet_id)

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
                            logger.info("Recovery attempted - checking if it helped...")
                        else:
                            logger.warning("Recovery failed")
                else:
                    consecutive_empty_scrolls = 0
                    logger.info(f"Found {tweets_found_this_cycle} new tweets, continuing...")

                iteration += 1

                # Record article count before scrolling
                prev_article_count = len(articles)

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

                # Wait for new tweets to load after scrolling
                try:
                    logger.debug(f"Waiting for more tweets to load (was {prev_article_count})...")
                    page.wait_for_function(f"document.querySelectorAll('article[data-testid=\"tweet\"]').length > {prev_article_count}", timeout=20000)
                    new_count = page.evaluate("document.querySelectorAll('article[data-testid=\"tweet\"]').length")
                    logger.debug(f"New tweets loaded: now {new_count} articles")
                except Exception:
                    new_count = page.evaluate("document.querySelectorAll('article[data-testid=\"tweet\"]').length")
                    logger.debug(f"Timeout waiting for new tweets, continuing with {new_count} articles...")

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

        # Phase 3 (post-collection grouping) has been removed — thread detection is performed inline during collection
        logger.debug("Thread grouping is now performed inline during collection; no post-collection Phase 3 run.")

        return collected_tweets

    def detect_and_save_threads(
        self,
        username: str,
        session_manager,
        conn,
        max_threads: int | None = None,
        max_timeline_scrolls: int = 20,
        existing_context=None
    ) -> int:
        """Detect threads from user timeline and save to database.
        
        Uses the enhanced thread detection that:
        1. Scans timeline for CSS thread indicators
        2. Opens conversation pages to validate self-reply chains
        3. Saves thread metadata to database
        
        Args:
            username: Twitter username to scan
            session_manager: SessionManager instance
            conn: Database connection
            max_threads: Optional maximum threads to detect (None = unlimited)
            max_timeline_scrolls: Maximum scroll iterations
            existing_context: Optional existing BrowserContext to reuse
            
        Returns:
            Number of threads detected and saved
        """
        from .thread_detector import ThreadDetector
        from database import ensure_schema_up_to_date
        
        # Ensure database has thread columns
        ensure_schema_up_to_date(conn)
        
        detector = ThreadDetector()
        
        logger.info(f"🧵 Starting thread detection for @{username}")
        
        try:
            # Use the enhanced conversation-validated detection
            threads = detector.detect_threads_with_conversation_validation(
                username=username,
                session_manager=session_manager,
                max_threads=max_threads,
                max_timeline_scrolls=max_timeline_scrolls,
                existing_context=existing_context
            )
            
            if not threads:
                logger.info(f"No threads detected for @{username}")
                return 0
            
            # Save each thread to database
            total_saved = 0
            for thread in threads:
                saved_count = detector.save_thread_to_database(thread, conn, username)
                if saved_count > 0:
                    total_saved += 1
                    logger.info(
                        f"✅ Thread {thread.start_id}: {thread.size} posts "
                        f"(conversation_id: {thread.conversation_id or 'N/A'})"
                    )
            
            logger.info(f"📊 Thread detection complete: {total_saved} threads saved")
            return total_saved
            
        except Exception as e:
            logger.error(f"❌ Thread detection failed: {e}")
            raise


# Global collector instance
collector = TweetCollector()

def get_collector() -> TweetCollector:
    """Get the global collector instance."""
    return collector