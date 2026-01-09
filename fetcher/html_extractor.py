"""
Parallel HTML Extraction for Performance Optimization (P1).

This module provides stateless HTML parsing using BeautifulSoup to avoid
multiple round-trips to the browser. Instead of calling page.query_selector()
multiple times, we extract all article HTML in a single JavaScript call
and parse it in Python.

Benefits:
- Reduces browser communication overhead
- Enables parallel parsing (no shared state)
- More robust to timing issues
- Easier to test with fixture HTML
"""

import re
import time
from typing import Dict, List, Optional, Tuple, Any
from bs4 import BeautifulSoup
from dataclasses import dataclass, field

from .logging_config import get_logger

logger = get_logger('html_extractor')


@dataclass
class ExtractionStats:
    """Track extraction performance statistics."""
    total_extractions: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    total_time: float = 0.0
    
    @property
    def avg_time_per_extraction(self) -> float:
        if self.total_extractions == 0:
            return 0.0
        return self.total_time / self.total_extractions
    
    @property
    def success_rate(self) -> float:
        if self.total_extractions == 0:
            return 0.0
        return self.successful_extractions / self.total_extractions


# Module-level stats for tracking (internal use)
_extraction_stats = ExtractionStats()


def extract_all_article_html(page) -> List[Dict[str, Any]]:
    """
    Extract HTML and basic info from all tweet articles in a single browser call.
    
    This is the key performance optimization - instead of making many
    query_selector calls, we get everything at once.
    
    Args:
        page: Playwright page object
        
    Returns:
        List of dicts with 'html', 'rect', 'index' for each article
    """
    try:
        result = page.evaluate('''() => {
            const articles = document.querySelectorAll('article[data-testid="tweet"]');
            return Array.from(articles).map((article, index) => ({
                html: article.outerHTML,
                rect: {
                    top: article.getBoundingClientRect().top,
                    left: article.getBoundingClientRect().left,
                    width: article.getBoundingClientRect().width,
                    height: article.getBoundingClientRect().height
                },
                index: index
            }));
        }''')
        return result or []
    except Exception as e:
        logger.error(f"Failed to extract article HTML: {e}")
        return []


def parse_tweet_from_html(html: str, target_username: str) -> Optional[Dict[str, Any]]:
    """
    Parse tweet data from article HTML using BeautifulSoup.
    
    This is a stateless function that can be called in parallel.
    
    Args:
        html: Raw HTML of a tweet article
        target_username: Username we're collecting tweets for
        
    Returns:
        Dict with tweet data or None if parsing failed/not target user
    """
    global _extraction_stats
    _extraction_stats.total_extractions += 1
    start_time = time.time()
    
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find the tweet link to get tweet ID and author
        tweet_link = soup.select_one('a[href*="/status/"]')
        if not tweet_link:
            _extraction_stats.failed_extractions += 1
            return None
        
        href = tweet_link.get('href', '')
        author, tweet_id = _parse_author_and_id_from_href(href)
        
        if not author or not tweet_id:
            _extraction_stats.failed_extractions += 1
            return None
        
        # Skip if not from target user
        if author.lower() != target_username.lower():
            # Not a failure, just not our target
            _extraction_stats.successful_extractions += 1
            _extraction_stats.total_time += time.time() - start_time
            return None
        
        # Extract basic tweet data
        tweet_data = {
            'tweet_id': tweet_id,
            'username': author,
            'tweet_url': f'https://x.com{href}',
        }
        
        # Extract text content
        tweet_data['content'] = _extract_text_from_soup(soup)
        
        # Extract timestamp
        time_elem = soup.select_one('time')
        if time_elem:
            tweet_data['tweet_timestamp'] = time_elem.get('datetime')
        
        # Extract media
        media_links, media_count, media_types = _extract_media_from_soup(soup)
        tweet_data['media_links'] = ','.join(media_links) if media_links else None
        tweet_data['media_count'] = media_count
        
        # Extract engagement metrics
        tweet_data.update(_extract_engagement_from_soup(soup))
        
        # Detect post type
        post_type_data = _analyze_post_type_from_soup(soup, target_username)
        tweet_data.update(post_type_data)
        
        # Detect thread indicators
        has_thread_line = _has_thread_line_from_soup(soup)
        tweet_data['has_thread_line'] = has_thread_line
        
        _extraction_stats.successful_extractions += 1
        _extraction_stats.total_time += time.time() - start_time
        
        return tweet_data
        
    except Exception as e:
        logger.error(f"Error parsing tweet HTML: {e}")
        _extraction_stats.failed_extractions += 1
        _extraction_stats.total_time += time.time() - start_time
        return None


def _parse_author_and_id_from_href(href: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse author username and tweet ID from a tweet URL."""
    if not href:
        return None, None
    
    # URL format: /{author}/status/{tweet_id}
    parts = href.strip('/').split('/')
    if len(parts) >= 3 and parts[1] == 'status':
        author = parts[0]
        tweet_id = parts[2].split('?')[0]  # Remove query params
        return author, tweet_id
    
    return None, None


def _extract_text_from_soup(soup: BeautifulSoup) -> str:
    """Extract tweet text content from BeautifulSoup parsed HTML."""
    # Try the main tweet text selector
    text_elem = soup.select_one('[data-testid="tweetText"]')
    if text_elem:
        return text_elem.get_text(strip=True)
    
    # Fallback: look for text in spans within the article
    text_spans = soup.select('div[lang] span')
    if text_spans:
        return ' '.join(span.get_text(strip=True) for span in text_spans)
    
    return ''


def _extract_media_from_soup(soup: BeautifulSoup) -> Tuple[List[str], int, List[str]]:
    """Extract media URLs from BeautifulSoup parsed HTML."""
    media_links = []
    media_types = []
    
    # Extract images
    image_selectors = [
        'img[src*="pbs.twimg.com/media/"]',
        'img[src*="twimg.com/media/"]',
        '[data-testid="tweetPhoto"] img',
    ]
    
    for selector in image_selectors:
        for img in soup.select(selector):
            src = img.get('src') or img.get('data-src')
            if src and 'twimg.com' in src and src not in media_links:
                # Skip video thumbnails
                if 'amplify_video_thumb' in src or '/thumb/' in src:
                    continue
                # Skip profile images
                if 'profile_images' in src:
                    continue
                media_links.append(src)
                media_types.append('image')
    
    # Extract video posters/thumbnails (note: actual video URLs need browser)
    video_elems = soup.select('[data-testid="videoPlayer"] video')
    for video in video_elems:
        poster = video.get('poster')
        if poster and poster not in media_links:
            media_links.append(poster)
            media_types.append('image')  # Poster is an image
        
        src = video.get('src')
        if src and 'video.twimg.com' in src and src not in media_links:
            media_links.append(src)
            media_types.append('video')
    
    return media_links, len(media_links), media_types


def _extract_engagement_from_soup(soup: BeautifulSoup) -> Dict[str, int]:
    """Extract engagement metrics from BeautifulSoup parsed HTML."""
    engagement = {
        'engagement_retweets': 0,
        'engagement_likes': 0,
        'engagement_replies': 0,
        'engagement_views': 0,
    }
    
    def parse_count(text: str) -> int:
        if not text:
            return 0
        text = text.strip().upper()
        try:
            # Extract the numeric part (may include decimal point)
            match = re.search(r'([\d,.]+)\s*([KM])?', text)
            if not match:
                return 0
            
            number_str = match.group(1).replace(',', '.')
            suffix = match.group(2)
            
            if suffix == 'K':
                return int(float(number_str) * 1000)
            elif suffix == 'M':
                return int(float(number_str) * 1000000)
            else:
                # Remove any remaining non-digit characters
                clean_num = re.sub(r'[^0-9]', '', number_str)
                return int(clean_num) if clean_num else 0
        except Exception:
            return 0
    
    # Try to find engagement elements by data-testid
    selectors = {
        'engagement_replies': '[data-testid="reply"]',
        'engagement_retweets': '[data-testid="retweet"]',
        'engagement_likes': '[data-testid="like"]',
    }
    
    for key, selector in selectors.items():
        elem = soup.select_one(selector)
        if elem:
            # Look for aria-label with count
            aria = elem.get('aria-label', '')
            if aria:
                count = parse_count(aria)
                engagement[key] = count
            else:
                # Try to find text content
                text = elem.get_text(strip=True)
                count = parse_count(text)
                engagement[key] = count
    
    # Views are usually in a different container
    view_elem = soup.select_one('[data-testid="app-text-transition-container"]')
    if view_elem:
        text = view_elem.get_text(strip=True)
        engagement['engagement_views'] = parse_count(text)
    
    return engagement


def _analyze_post_type_from_soup(soup: BeautifulSoup, target_username: str) -> Dict[str, Any]:
    """Analyze post type (original, retweet, quote, pinned) from HTML."""
    result = {
        'post_type': 'original',
        'original_author': None,
        'original_tweet_id': None,
    }
    
    # Check for pinned indicator first
    social_context = soup.select_one('[data-testid="socialContext"]')
    if social_context:
        context_text = social_context.get_text(strip=True).lower()
        
        # Check for pinned
        if 'pinned' in context_text or 'fijado' in context_text:
            result['post_type'] = 'pinned'
            return result
        
        # Check for retweet indicator
        if 'retweeted' in context_text or 'retuiteó' in context_text:
            result['post_type'] = 'retweet'
            # Find the original author from the tweet
            # In retweets, the first user link is often the original author
            user_links = soup.select('a[href^="/"][role="link"]')
            for link in user_links:
                href = link.get('href', '')
                if href.startswith('/') and '/status/' not in href:
                    username = href.strip('/')
                    if username and username.lower() != target_username.lower():
                        result['original_author'] = username
                        break
    
    # Also check for pinned via aria-label
    pinned_indicator = soup.select_one('[aria-label*="Pinned"], [aria-label*="fijado"]')
    if pinned_indicator:
        result['post_type'] = 'pinned'
        return result
    
    # Check for self-retweet (repost_own) - when target user retweets their own tweet
    # This happens when the retweeter (from social context) equals the tweet author
    if result['post_type'] == 'retweet' and social_context:
        context_text = social_context.get_text(strip=True)
        # Extract retweeter name from context (e.g., "Retweeted by username" or "username retweeted")
        retweeter = None
        # Pattern: "Retweeted by X" or "X retweeted"
        import re
        match = re.search(r'(?:retweeted by|retuiteó)\s+@?(\w+)', context_text, re.IGNORECASE)
        if match:
            retweeter = match.group(1)
        else:
            match = re.search(r'@?(\w+)\s+(?:retweeted|retuiteó)', context_text, re.IGNORECASE)
            if match:
                retweeter = match.group(1)
        
        if retweeter:
            # Find the tweet author
            tweet_link = soup.select_one('a[href*="/status/"]')
            if tweet_link:
                href = tweet_link.get('href', '')
                author, _ = _parse_author_and_id_from_href(href)
                # Self-retweet: retweeter == author AND that's our target user
                if author and retweeter.lower() == author.lower() and author.lower() == target_username.lower():
                    result['post_type'] = 'repost_own'
    
    # Check for quote tweet (embedded tweet card)
    quoted_card = soup.select_one('[data-testid="card.wrapper"]')
    if quoted_card:
        quoted_link = quoted_card.select_one('a[href*="/status/"]')
        if quoted_link:
            href = quoted_link.get('href', '')
            author, tweet_id = _parse_author_and_id_from_href(href)
            if author:
                result['post_type'] = 'quote'
                result['original_author'] = author
                result['original_tweet_id'] = tweet_id
    
    return result


def _has_thread_line_from_soup(soup: BeautifulSoup) -> bool:
    """Check if tweet has a thread continuation line."""
    # The thread line uses specific CSS class r-1bimlpy
    thread_indicator = soup.select_one('[class*="r-1bimlpy"]')
    if thread_indicator:
        return True
    
    # Also check for reply-to indicators
    reply_indicator = soup.select_one('[data-testid="tweet"] [data-testid="tweet"]')
    if reply_indicator:
        return True
    
    return False