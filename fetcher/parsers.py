import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def should_skip_existing_tweet(tweet_timestamp: str, oldest_timestamp: Optional[str]) -> bool:
    if not oldest_timestamp:
        return False
    try:
        tweet_time = datetime.fromisoformat(tweet_timestamp.replace('Z', '+00:00'))
        oldest_time = datetime.fromisoformat(oldest_timestamp.replace('Z', '+00:00'))
        return tweet_time >= oldest_time
    except Exception:
        return False


def extract_media_data(article) -> Tuple[List[str], int, List[str]]:
    media_links = []
    media_types = []

    image_selectors = [
        'img[src*="pbs.twimg.com/media/"]',
        'img[src*="twimg.com/media/"]',
        'img[data-testid="tweetPhoto"]',
        '[data-testid="tweetPhoto"] img',
        '[role="img"][src*="twimg.com"]'
    ]

    for selector in image_selectors:
        for img in article.query_selector_all(selector):
            src = img.get_attribute('src')
            if src and 'twimg.com' in src and src not in media_links:
                media_links.append(src)
                media_types.append('image')

    video_selectors = ['video', '[data-testid="videoComponent"] video', '[data-testid="videoPlayer"] video']
    for selector in video_selectors:
        for video in article.query_selector_all(selector):
            poster = video.get_attribute('poster')
            if poster and poster not in media_links:
                media_links.append(poster)
                media_types.append('image')
            src = video.get_attribute('src')
            if src and src not in media_links:
                media_links.append(src)
                media_types.append('video')

    for elem in article.query_selector_all('[style*="background-image"]'):
        style = elem.get_attribute('style')
        if style and 'twimg.com' in style:
            match = re.search(r'background-image:\s*url\(["\']?(.*?twimg\.com[^"\')\s]*)', style)
            if match:
                bg_url = match.group(1)
                if bg_url not in media_links:
                    media_links.append(bg_url)
                    media_types.append('image')

    # Filter out unwanted media types: profile images and card previews
    filtered_media_links = []
    filtered_media_types = []
    
    for url, media_type in zip(media_links, media_types):
        # Skip profile images (user avatars)
        if 'profile_images' in url:
            continue
        # Skip card images (link previews, thumbnails)
        if 'card_img' in url:
            continue
        # Keep only actual content media
        filtered_media_links.append(url)
        filtered_media_types.append(media_type)

    return filtered_media_links, len(filtered_media_links), filtered_media_types


def extract_engagement_metrics(article) -> Dict[str, int]:
    engagement = {'retweets': 0, 'likes': 0, 'replies': 0, 'views': 0}
    engagement_selectors = {
        'replies': ['[data-testid="reply"]', 'svg[data-testid="iconMessageCircle"]'],
        'retweets': ['[data-testid="retweet"]', 'svg[data-testid="iconRetweet"]'],
        'likes': ['[data-testid="like"]', 'svg[data-testid="iconHeart"]'],
        'views': ['[data-testid="app-text-transition-container"]']
    }

    def parse_int(text: str) -> int:
        if not text:
            return 0
        text = text.strip().upper()
        try:
            if text.endswith('K'):
                return int(float(text[:-1].replace(',', '.')) * 1000)
            if text.endswith('M'):
                return int(float(text[:-1].replace(',', '.')) * 1000000)
            return int(re.sub(r'[^0-9]', '', text) or 0)
        except Exception:
            return 0

    for metric, selectors in engagement_selectors.items():
        for selector in selectors:
            for element in article.query_selector_all(selector):
                # try to find a numeric parent text
                parent_text = None
                try:
                    parent_text = element.inner_text()
                except Exception:
                    parent_text = element.get_attribute('aria-label') or ''
                val = parse_int(parent_text)
                engagement[metric] = max(engagement.get(metric, 0), val)

    return engagement


def extract_full_tweet_content(article) -> str:
    try:
        text_elem = article.query_selector('[data-testid="tweetText"]')
        if text_elem:
            try:
                return text_elem.inner_text()
            except Exception:
                return text_elem.get_attribute('text') or ''
        # fallback: try generic inner_text
        try:
            return article.inner_text()
        except Exception:
            return ''
    except Exception:
        return ''


def extract_content_elements(article) -> Dict[str, any]:
    hashtags = []
    mentions = []
    external_links = []

    for hashtag_link in article.query_selector_all('a[href*="/hashtag/"]'):
        text = hashtag_link.inner_text() if hasattr(hashtag_link, 'inner_text') else hashtag_link.get_attribute('text')
        if text and text.startswith('#'):
            hashtags.append(text)

    for mention_link in article.query_selector_all('a[href^="/"]:not([href*="/hashtag/"]):not([href*="/status/"])'):
        href = mention_link.get_attribute('href')
        if href:
            # normalize to @username
            parts = href.strip('/').split('/')
            if parts:
                mentions.append('@' + parts[0])

    for link in article.query_selector_all('a[href^="http"]'):
        href = link.get_attribute('href')
        if href:
            external_links.append(href)

    return {
        'hashtags': json.dumps(hashtags) if hashtags else None,
        'mentions': json.dumps(mentions) if mentions else None,
        'external_links': json.dumps(external_links) if external_links else None,
        'has_external_link': 1 if external_links else 0
    }


def extract_profile_picture(page, username: str) -> Optional[str]:
    # best-effort extraction without navigation
    selectors = [
        'img[src*="profile_images"]',
        'img[src*="pbs.twimg.com/profile_images"]',
        'img[alt*="Profile photo"]'
    ]
    try:
        for sel in selectors:
            el = page.query_selector(sel)
            if el:
                src = el.get_attribute('src')
                if src:
                    return src
    except Exception:
        return None
    return None


def analyze_post_type(article, target_username: str) -> Dict[str, any]:
    post_analysis = {
        'post_type': 'original',
        'is_pinned': 0,
        'original_author': None,
        'original_tweet_id': None,
        'original_content': None,
        'reply_to_username': None,
        'reply_to_tweet_id': None,
        'reply_to_content': None,
        'thread_position': 0,
        'thread_root_id': None,
        'should_skip': False
    }

    # Check for pinned
    try:
        pinned_indicator = article.query_selector('[data-testid="socialContext"]:has-text("Pinned"), [aria-label*="Pinned"]')
        if pinned_indicator:
            post_analysis['is_pinned'] = 1
            post_analysis['should_skip'] = True
            return post_analysis
    except Exception:
        pass

    # Quote-like content: preserve metadata but keep as original
    try:
        main_text = article.query_selector('[data-testid="tweetText"]')
        quoted_content = article.query_selector('[data-testid="tweetText"] ~ div [role="article"], .css-1dbjc4n [role="article"]')
        if main_text and quoted_content:
            try:
                quoted_author_link = None
                for a in quoted_content.query_selector_all('a[href^="/"]'):
                    href = a.get_attribute('href')
                    if href and href.count('/') == 1:
                        quoted_author_link = a
                        break
                if quoted_author_link:
                    quoted_href = quoted_author_link.get_attribute('href')
                    post_analysis['reply_to_username'] = quoted_href.replace('/', '') if quoted_href else None
            except Exception:
                post_analysis['reply_to_username'] = None

            try:
                quoted_tweet_link = quoted_content.query_selector('a[href*="/status/"]')
                if quoted_tweet_link:
                    quoted_tweet_href = quoted_tweet_link.get_attribute('href')
                    post_analysis['reply_to_tweet_id'] = quoted_tweet_href.split('/')[-1] if quoted_tweet_href else None
            except Exception:
                pass

            try:
                quoted_text_elem = quoted_content.query_selector('[data-testid="tweetText"]')
                if quoted_text_elem:
                    post_analysis['reply_to_content'] = quoted_text_elem.inner_text().strip()
            except Exception:
                pass

            return post_analysis
    except Exception:
        pass

    # Repost detection (simplified heuristic)
    try:
        repost_indicators = [
            '[data-testid="socialContext"]:has-text("Reposted"), [data-testid="socialContext"]:has-text("reposted"), [data-testid="socialContext"]:has-text("Retweeted"), [data-testid="socialContext"]:has-text("Retweeted" )',
            'svg[data-testid="iconRetweet"]',
            '[aria-label*="Repost"], [aria-label*="repost"], [aria-label*="Retweet"], [aria-label*="retweet"]',
            'div:has-text("Retweeted"), div:has-text("Reposted"), div:has-text("Reposteado"), div:has-text("Retuiteado")'
        ]
        repost_element = None
        for selector in repost_indicators:
            try:
                repost_element = article.query_selector(selector)
            except Exception:
                repost_element = article.query_selector(selector.split(',')[0]) if ',' in selector else None
            if repost_element:
                try:
                    elem_text = (repost_element.inner_text() or "").lower()
                except Exception:
                    elem_text = ""
                try:
                    elem_aria = (repost_element.get_attribute('aria-label') or "").lower()
                except Exception:
                    elem_aria = ""

                keywords = ["retweet", "retweeted", "reposted", "reposteado", "retuiteado"]
                is_repost_text = any(k in elem_text for k in keywords) or any(k in elem_aria for k in keywords)
                if not is_repost_text:
                    try:
                        link_to_target = repost_element.query_selector(f'a[href="/{target_username}"]')
                    except Exception:
                        link_to_target = None
                    if not link_to_target:
                        repost_element = None
                        continue
                break

        if repost_element:
            try:
                reposter_link = repost_element.query_selector(f'a[href="/{target_username}"]')
            except Exception:
                reposter_link = None
            if reposter_link:
                try:
                    original_link_any = article.query_selector('a[href*="/status/"]')
                    if original_link_any:
                        orig_href = original_link_any.get_attribute('href')
                        post_analysis['original_tweet_id'] = orig_href.split('/')[-1] if orig_href else None
                except Exception:
                    pass
                post_analysis['post_type'] = 'repost_own'
                post_analysis['original_author'] = target_username
                post_analysis['should_skip'] = True
                return post_analysis

            quoted_tweet = None
            try:
                quoted_tweet = article.query_selector('article [data-testid="tweetText"], [data-testid="tweet"] article')
            except Exception:
                quoted_tweet = None

            if not quoted_tweet:
                try:
                    original_tweet_link = article.query_selector('a[href*="/status/"]')
                    if original_tweet_link:
                        parent = original_tweet_link.evaluate('el => el.closest("article")')
                        quoted_tweet = parent
                except Exception:
                    pass

            if quoted_tweet:
                try:
                    original_tweet_link = quoted_tweet.query_selector('a[href*="/status/"]')
                except Exception:
                    original_tweet_link = None

                if original_tweet_link:
                    try:
                        original_href = original_tweet_link.get_attribute('href')
                        original_tweet_id = original_href.split('/')[-1] if original_href else None
                        post_analysis['original_tweet_id'] = original_tweet_id
                    except Exception:
                        original_tweet_id = None

                    original_author = None
                    try:
                        candidates = []
                        for a in quoted_tweet.query_selector_all('a[href^="/"]'):
                            href = a.get_attribute('href')
                            if not href:
                                continue
                            if '/status/' in href or '/photo' in href or '/video' in href:
                                continue
                            if href.count('/') == 1:
                                candidate = href.replace('/', '')
                                if candidate and candidate not in candidates:
                                    candidates.append(candidate)
                        if candidates:
                            if len(candidates) == 1:
                                original_author = candidates[0]
                            else:
                                picked = None
                                for c in candidates:
                                    if c != target_username:
                                        picked = c
                                        break
                                original_author = picked or candidates[0]
                    except Exception:
                        original_author = None

                    if original_author and post_analysis.get('original_tweet_id'):
                        if original_author == target_username:
                            post_analysis['post_type'] = 'repost_own'
                            post_analysis['should_skip'] = True
                        else:
                            post_analysis['post_type'] = 'repost_other'
                        post_analysis['original_author'] = original_author
                        try:
                            original_text_elem = quoted_tweet.query_selector('[data-testid="tweetText"]')
                            if original_text_elem:
                                post_analysis['original_content'] = original_text_elem.inner_text().strip()
                        except Exception:
                            pass
                        return post_analysis
    except Exception:
        pass

    # Reply indicators -> repost_reply
    try:
        reply_context = article.query_selector('[data-testid="socialContext"]:has-text("Replying to"), [data-testid="tweetText"]:has-text("Replying to")')
        if reply_context:
            post_analysis['post_type'] = 'repost_reply'
            try:
                reply_mention = reply_context.query_selector('a[href^="/"]')
                if reply_mention:
                    reply_href = reply_mention.get_attribute('href')
                    post_analysis['reply_to_username'] = reply_href.replace('/', '') if reply_href else None
            except Exception:
                pass
            return post_analysis
    except Exception:
        pass

    # Quote-like (again) handled above; thread indicator:
    try:
        thread_indicator = article.query_selector('[data-testid="socialContext"]:has-text("Show this thread")')
        if thread_indicator:
            post_analysis['post_type'] = 'thread'
            return post_analysis
    except Exception:
        pass

    return post_analysis
