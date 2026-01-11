"""
Async tweet collection logic for the fetcher module.

Async version of collector.py for use with async_playwright.
Provides improved I/O handling during network operations.
"""

import asyncio
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime

from playwright.async_api import async_playwright

from .config import get_config
from .logging_config import get_logger
from .async_scroller import get_async_scroller
from .async_session_manager import get_async_session_manager
from . import db as fetcher_db
from .db import TweetBuffer
from . import parsers as fetcher_parsers
from . import html_extractor
from database.repositories import get_tweet_repository
from database import get_db_connection_context

logger = get_logger('async_collector')


class AsyncTweetCollector:
    """Async version of TweetCollector - handles async tweet collection operations."""

    def __init__(self):
        self.config = get_config()
        self.scroller = get_async_scroller()
        self.session_manager = get_async_session_manager()

    def should_process_tweet(self, tweet_id: str, seen_ids: Set[str]) -> bool:
        """Check if a tweet should be processed."""
        if not tweet_id or tweet_id in seen_ids:
            return False
        return True

    def check_tweet_exists_in_db(self, username: str, tweet_id: str) -> Tuple[bool, Optional[Dict]]:
        """Check if tweet exists in database."""
        if self.config.skip_duplicate_check:
            return False, None
        
        tweet_repo = get_tweet_repository()
        tweet = tweet_repo.get_tweet_by_id(tweet_id)
        
        exists = tweet is not None and tweet.get('username') == username
        if not exists:
            return False, None

        return True, {
            'post_type': tweet.get('post_type'),
            'content': tweet.get('content'),
            'original_author': tweet.get('original_author'),
            'original_tweet_id': tweet.get('original_tweet_id')
        }

    async def _save_debug_info(self, page, username: str, tweet_count: int) -> None:
        """Save debug HTML and screenshot when empty scrolls are detected."""
        from pathlib import Path
        import time
        
        debug_dir = Path("logs/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        base_name = f"empty_scroll_{username}_{tweet_count}_{timestamp}"
        
        try:
            # Save screenshot
            screenshot_path = debug_dir / f"{base_name}.png"
            await page.screenshot(path=str(screenshot_path), full_page=False)
            logger.warning(f"📸 Debug screenshot saved: {screenshot_path}")
            
            # Save page HTML
            html_path = debug_dir / f"{base_name}.html"
            html_content = await page.content()
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.warning(f"📄 Debug HTML saved: {html_path}")
            
            # Log current URL
            current_url = page.url
            logger.warning(f"🔗 Current URL: {current_url}")
            
            # Check for common error indicators
            try:
                main_text = await page.locator('main').inner_text()
                if 'Something went wrong' in main_text:
                    logger.error("❌ Twitter shows 'Something went wrong' error!")
                elif 'Rate limit' in main_text.lower():
                    logger.error("❌ Twitter rate limit detected!")
                elif 'Try again' in main_text:
                    logger.error("❌ Twitter shows 'Try again' prompt!")
                else:
                    # Log first 500 chars of main content
                    logger.warning(f"📝 Main content preview: {main_text[:500]}...")
            except Exception:
                pass
                
        except Exception as e:
            logger.error(f"Failed to save debug info: {e}")

    async def collect_tweets_from_page(
        self,
        page,
        username: str,
        max_tweets: int,
        conn
    ) -> List[Dict]:
        """
        Collect tweets from a page (async version).

        Args:
            page: Async Playwright page object
            username: Twitter username
            max_tweets: Maximum tweets to collect
            conn: Database connection

        Returns:
            List of collected tweet data
        """
        collected_tweets = []
        seen_tweet_ids: Set[str] = set()
        consecutive_empty_scrolls = 0
        tweets_found_this_cycle = 0
        saved_count = 0
        
        tweet_buffer = TweetBuffer(conn, batch_size=self.config.batch_write_size)

        if max_tweets == float('inf'):
            logger.info("Collecting unlimited tweets (async)...")
        else:
            logger.info(f"Collecting up to {max_tweets} tweets (async)...")

        timeline_url = f"https://x.com/{username}"

        while len(collected_tweets) < max_tweets:
            if consecutive_empty_scrolls >= self.config.max_consecutive_empty_scrolls:
                logger.info("Stopping: too many empty scrolls")
                break

            # Find tweet articles
            articles = await page.query_selector_all('article[data-testid="tweet"]')
            logger.debug(f"Found {len(articles)} tweet elements")

            tweets_found_this_cycle = 0

            for article in articles:
                if len(collected_tweets) >= max_tweets:
                    break

                try:
                    tweet_link = await article.query_selector('a[href*="/status/"]')
                    if not tweet_link:
                        continue

                    href = await tweet_link.get_attribute('href')
                    if not href:
                        continue
                    
                    should_process, actual_author, tweet_id = fetcher_parsers.should_process_tweet_by_author(href, username)
                    
                    if not should_process:
                        continue

                    if not self.should_process_tweet(tweet_id, seen_tweet_ids):
                        continue

                    tweet_url = f"https://x.com{href}"
                    exists_in_db, _ = self.check_tweet_exists_in_db(username, tweet_id)

                    # Extract article HTML for stateless parsing
                    article_html = await article.evaluate('el => el.outerHTML')
                    tweet_data = html_extractor.parse_tweet_from_html(article_html, username)
                    
                    if not tweet_data:
                        continue
                    
                    tweet_data['tweet_id'] = tweet_id
                    tweet_data['tweet_url'] = tweet_url
                    tweet_data['username'] = username

                    # Add to buffer
                    if tweet_buffer.add(tweet_data):
                        saved_count += 1
                        post_type = tweet_data.get('post_type', 'unknown')
                        logger.info(f"Buffered {post_type}: {tweet_id}")

                    collected_tweets.append(tweet_data)
                    seen_tweet_ids.add(tweet_id)
                    tweets_found_this_cycle += 1

                except Exception as e:
                    logger.error(f"Error processing tweet: {e}")
                    continue

            # Handle scrolling
            if len(collected_tweets) < max_tweets:
                current_count = len(collected_tweets)
                if max_tweets == float('inf'):
                    logger.info(f"Scrolling for more (async)... ({current_count}/∞)")
                else:
                    logger.info(f"Scrolling for more (async)... ({current_count}/{max_tweets})")

                if tweets_found_this_cycle == 0:
                    consecutive_empty_scrolls += 1
                    logger.warning(f"No new tweets ({consecutive_empty_scrolls}/{self.config.max_consecutive_empty_scrolls})")
                    
                    # Debug: Save page HTML and screenshot on 3rd consecutive empty scroll
                    if consecutive_empty_scrolls == 3:
                        await self._save_debug_info(page, username, current_count)
                    
                    if consecutive_empty_scrolls % 5 == 0:
                        await self.scroller.try_recovery_strategies(page, consecutive_empty_scrolls // 5)
                else:
                    consecutive_empty_scrolls = 0

                prev_count = len(articles)
                await self.scroller.adaptive_scroll(
                    page,
                    target_count_selector='article[data-testid="tweet"]',
                    prev_count=prev_count
                )

        # Flush remaining tweets
        remaining = tweet_buffer.flush()
        if remaining > 0:
            logger.info(f"Flushed final batch: {remaining} tweets")
        
        stats = tweet_buffer.stats
        if stats.batches_written > 0:
            logger.info(f"Batch stats: {stats.total_tweets} tweets, {stats.batches_written} batches")

        logger.info(f"Async collection complete: {len(collected_tweets)} tweets, {saved_count} saved")
        return collected_tweets

    async def collect_from_search(
        self,
        username: str,
        start_date,
        end_date,
        page=None,
        conn=None
    ) -> List[Dict]:
        """
        Collect tweets from a search results page with date range filters.
        
        Uses Twitter's search with: from:username since:YYYY-MM-DD until:YYYY-MM-DD
        
        Args:
            username: Twitter username to search for
            start_date: Start date (inclusive)
            end_date: End date (exclusive)
            page: Optional Playwright page (if None, creates new browser)
            conn: Optional database connection
        
        Returns:
            List of collected tweet data
        """
        from urllib.parse import quote_plus
        from datetime import date
        
        # Build search URL
        query = f"from:{username} since:{start_date.isoformat()} until:{end_date.isoformat()}"
        encoded_query = quote_plus(query)
        search_url = f"https://x.com/search?q={encoded_query}&src=typed_query&f=live"
        
        logger.info(f"🔍 Search: {query}")
        
        collected_tweets = []
        
        # If no page provided, we need to create a browser session
        if page is None:
            from playwright.async_api import async_playwright
            
            async with async_playwright() as p:
                browser, context, page = await self.session_manager.create_browser_context(p)
                
                try:
                    # Navigate to search
                    await page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
                    await self.scroller.delay(2, 3)  # Wait for results
                    
                    # Collect from this page
                    if conn is None:
                        with get_db_connection_context() as conn:
                            collected_tweets = await self._collect_search_results(
                                page, username, conn
                            )
                    else:
                        collected_tweets = await self._collect_search_results(
                            page, username, conn
                        )
                finally:
                    await context.close()
                    await browser.close()
        else:
            # Use provided page
            await page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
            await self.scroller.delay(2, 3)
            
            if conn is None:
                with get_db_connection_context() as conn:
                    collected_tweets = await self._collect_search_results(
                        page, username, conn
                    )
            else:
                collected_tweets = await self._collect_search_results(
                    page, username, conn
                )
        
        logger.info(f"   Collected {len(collected_tweets)} tweets from search")
        return collected_tweets

    async def _collect_search_results(
        self,
        page,
        username: str,
        conn,
        max_tweets: int = 1000
    ) -> List[Dict]:
        """
        Collect tweets from search results page.
        
        Similar to collect_tweets_from_page but for search results.
        """
        collected_tweets = []
        seen_tweet_ids: Set[str] = set()
        consecutive_empty_scrolls = 0
        
        tweet_buffer = TweetBuffer(conn, batch_size=self.config.batch_write_size)
        
        while len(collected_tweets) < max_tweets:
            if consecutive_empty_scrolls >= 10:  # Fewer scrolls for search
                break
            
            # Find tweet articles
            articles = await page.query_selector_all('article[data-testid="tweet"]')
            tweets_found_this_cycle = 0
            
            for article in articles:
                if len(collected_tweets) >= max_tweets:
                    break
                
                try:
                    tweet_link = await article.query_selector('a[href*="/status/"]')
                    if not tweet_link:
                        continue
                    
                    href = await tweet_link.get_attribute('href')
                    if not href:
                        continue
                    
                    # Parse tweet ID
                    parts = href.strip('/').split('/')
                    tweet_id = parts[-1].split('?')[0] if parts else None
                    
                    if not tweet_id or tweet_id in seen_tweet_ids:
                        continue
                    
                    # Extract tweet data
                    article_html = await article.evaluate('el => el.outerHTML')
                    tweet_data = html_extractor.parse_tweet_from_html(article_html, username)
                    
                    if not tweet_data:
                        continue
                    
                    tweet_data['tweet_id'] = tweet_id
                    tweet_data['tweet_url'] = f"https://x.com{href}"
                    tweet_data['username'] = username
                    
                    # Add to buffer
                    tweet_buffer.add(tweet_data)
                    collected_tweets.append(tweet_data)
                    seen_tweet_ids.add(tweet_id)
                    tweets_found_this_cycle += 1
                    
                except Exception as e:
                    logger.debug(f"Error processing search result: {e}")
                    continue
            
            # Scroll for more
            if tweets_found_this_cycle == 0:
                consecutive_empty_scrolls += 1
            else:
                consecutive_empty_scrolls = 0
            
            if len(collected_tweets) < max_tweets:
                await self.scroller.scroll_to_bottom(page)
                await self.scroller.delay(1, 2)
        
        # Flush remaining tweets
        tweet_buffer.flush()
        
        return collected_tweets


async def run_async_fetch_session(
    usernames: List[str],
    max_tweets: int = float('inf')
) -> Tuple[int, int]:
    """
    Run an async fetch session for multiple users.
    
    Args:
        usernames: List of usernames to fetch
        max_tweets: Maximum tweets per user
        
    Returns:
        Tuple of (total_tweets, accounts_processed)
    """
    total_tweets = 0
    accounts_processed = 0
    
    collector = AsyncTweetCollector()
    session_manager = get_async_session_manager()
    
    async with async_playwright() as p:
        browser, context, page = await session_manager.create_browser_context(p)
        
        try:
            for username in usernames:
                logger.info(f"Starting async collection for @{username}")
                
                # Navigate to profile
                if not await session_manager.navigate_to_profile(page, username):
                    logger.warning(f"Skipping @{username} - could not navigate")
                    continue
                
                # Collect tweets
                with get_db_connection_context() as conn:
                    tweets = await collector.collect_tweets_from_page(
                        page, username, max_tweets, conn
                    )
                
                total_tweets += len(tweets)
                accounts_processed += 1
                logger.info(f"Completed @{username}: {len(tweets)} tweets")
                
        finally:
            await session_manager.cleanup_session(browser, context)
    
    return total_tweets, accounts_processed


def run_async_fetch(usernames: List[str], max_tweets: int = float('inf')) -> Tuple[int, int]:
    """
    Synchronous wrapper for async fetch session.
    
    This allows calling async fetch from synchronous code.
    
    Args:
        usernames: List of usernames to fetch
        max_tweets: Maximum tweets per user
        
    Returns:
        Tuple of (total_tweets, accounts_processed)
    """
    return asyncio.run(run_async_fetch_session(usernames, max_tweets))
