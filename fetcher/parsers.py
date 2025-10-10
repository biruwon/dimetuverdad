import json
import re
import time
import random
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
            # NEW: Check for <source> tags inside <video>
            source_elems = video.query_selector_all('source')
            for source in source_elems:
                source_src = source.get_attribute('src')
                if source_src and source_src not in media_links:
                    media_links.append(source_src)
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
                    # Store quoted tweet content in original_content for analysis
                    quoted_text = quoted_text_elem.inner_text().strip()
                    post_analysis['original_content'] = quoted_text
                    # Keep reply_to_content for backward compatibility
                    post_analysis['reply_to_content'] = quoted_text
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


def human_delay(min_seconds: float = 1.0, max_seconds: float = 3.0):
    """Sleep for a random amount of time to mimic human behavior."""
    delay = random.uniform(min_seconds, max_seconds)
    time.sleep(delay)


def extract_tweet_with_quoted_content(page, tweet_id: str, username: str, tweet_url: str) -> dict:
    """
    Extract complete tweet data including quoted tweet content and media.
    
    Args:
        page: Playwright page object
        tweet_id: Tweet ID
        username: Tweet author username  
        tweet_url: Tweet URL
        
    Returns:
        dict: Tweet data with all fields, or None if extraction failed
    """
    # Find main tweet article
    articles = page.query_selector_all('article[data-testid="tweet"]')
    if not articles:
        print(f"❌ Could not find tweet content on page")
        return None

    main_article = articles[0]
    print(f"✅ Found main tweet article")

    # Extract main tweet content
    content = extract_full_tweet_content(main_article)
    post_analysis = analyze_post_type(main_article, username)
    post_analysis['tweet_id'] = tweet_id  # Add tweet_id for filtering
    main_media_links, main_media_count, main_media_types = extract_media_data(main_article)
    engagement = extract_engagement_metrics(main_article)
    content_elements = extract_content_elements(main_article)

    print(f"✅ Main tweet extracted: {len(content)} chars, {main_media_count} media")
    
    # Multi-strategy quoted tweet detection
    quoted_tweet_data = find_and_extract_quoted_tweet(page, main_article, post_analysis)
    
    # Combine main and quoted media (Option A: single media_links field)
    combined_media_links = list(main_media_links) if main_media_links else []
    if quoted_tweet_data and quoted_tweet_data.get('media_links'):
        quoted_media = quoted_tweet_data['media_links']
        # Add quoted media that's not already in main media
        for media_url in quoted_media:
            if media_url not in combined_media_links:
                combined_media_links.append(media_url)
        if quoted_tweet_data.get('media_count', 0) > 0:
            print(f"📎 Added {quoted_tweet_data['media_count']} media from quoted tweet")
    
    combined_media_count = len(combined_media_links)
    
    # Build complete tweet data
    tweet_data = {
        'tweet_id': tweet_id,
        'tweet_url': tweet_url,
        'username': username,
        'content': content,
        'post_type': post_analysis.get('post_type', 'original'),
        'original_author': post_analysis.get('original_author'),
        'original_tweet_id': post_analysis.get('original_tweet_id'),
        'original_content': post_analysis.get('original_content'),
        'reply_to_username': post_analysis.get('reply_to_username'),
        'media_links': ','.join(combined_media_links) if combined_media_links else None,
        'media_count': combined_media_count,
        'hashtags': ','.join(content_elements.get('hashtags', [])) if content_elements.get('hashtags') else None,
        'mentions': ','.join(content_elements.get('mentions', [])) if content_elements.get('mentions') else None,
        'external_links': ','.join(content_elements.get('external_links', [])) if content_elements.get('external_links') else None,
        'engagement_likes': engagement.get('likes', 0),
        'engagement_retweets': engagement.get('retweets', 0),
        'engagement_replies': engagement.get('replies', 0),
    }
    
    print(f"📝 Main content: {content[:100]}...")
    if tweet_data['original_content']:
        print(f"📎 Quoted content: {tweet_data['original_content'][:100]}...")
    if combined_media_count > 0:
        print(f"🖼️ Total media (main + quoted): {combined_media_count} items")
    
    return tweet_data


def find_and_extract_quoted_tweet(page, main_article, post_analysis: dict) -> dict:
    """
    Find and extract quoted tweet using multiple detection strategies.
    
    Args:
        page: Playwright page object
        main_article: Main tweet article element
        post_analysis: Post analysis dict to update with quoted content
        
    Returns:
        dict: Quoted tweet data, or None if not found
    """
    # Strategy 1: Parser already found it
    if post_analysis.get('original_content'):
        print(f"✅ Parser found quoted content: {len(post_analysis['original_content'])} chars")
        return {'content': post_analysis['original_content']}
    
    print(f"🔍 Searching for quoted tweet with multiple strategies...")
    
    # Strategy 2: Look for various quoted tweet card selectors
    quoted_card_selectors = [
        # New: Multiple tweet texts means there's a quoted tweet
        # The page has main tweet text + quoted tweet text
        ('check_multiple_tweets', None),  # Special check
        # Original nested article selector (timeline view)
        '[role="article"] [role="article"]',
        '[data-testid="tweetText"] ~ div [role="article"]',
        '[data-testid="card.wrapper"]',  # Quote card wrapper
        'div[class*="quoted"]',  # Any div with "quoted" in class  
        'article div[data-testid="card.layoutLarge.media"]',  # Large media card
        'article a[href*="/status/"] img[alt][src*="pbs.twimg.com"]',  # Link with preview image (parent)
    ]
    
    quoted_card = None
    for selector_item in quoted_card_selectors:
        try:
            # Handle special case for multiple tweet check
            if isinstance(selector_item, tuple) and selector_item[0] == 'check_multiple_tweets':
                # Look for all tweetText elements WITHIN the main article
                tweet_texts = main_article.query_selector_all('[data-testid="tweetText"]')
                if len(tweet_texts) >= 2:
                    # Second one is the quoted tweet (nested inside main article)
                    quoted_text_elem = tweet_texts[1]
                    print(f"✅ Found quoted tweet (2nd tweetText element within main article)")
                    # Get the text content first
                    quoted_text = quoted_text_elem.inner_text() if hasattr(quoted_text_elem, 'inner_text') else None
                    if quoted_text:
                        post_analysis['original_content'] = quoted_text
                        print(f"📎 Quoted content: {quoted_text[:150]}...")
                    
                    # Now click on the quoted tweet to navigate to it for complete extraction
                    try:
                        print(f"🖱️ Clicking on quoted tweet to navigate to it...")
                        quoted_text_elem.click()
                        page.wait_for_load_state("domcontentloaded")
                        human_delay(5.0, 7.0)

                        # Try to dismiss overlays/popups
                        try:
                            overlay = page.query_selector('div:has-text("unusual activity"), div:has-text("actividad inusual"), div[role="dialog"]')
                            if overlay:
                                close_btn = overlay.query_selector('button, [role="button"]')
                                if close_btn:
                                    print("🛑 Dismissing overlay/popup...")
                                    close_btn.click()
                                    human_delay(1.0, 2.0)
                        except Exception as e:
                            print(f"⚠️ Overlay dismissal failed: {e}")

                        # Poll for media elements for up to 10 seconds
                        poll_start = time.time()
                        found_media = False
                        while time.time() - poll_start < 10:
                            quoted_articles = page.query_selector_all('article[data-testid="tweet"]')
                            if quoted_articles:
                                quoted_main = quoted_articles[0]
                                video_elems = quoted_main.query_selector_all('video, div[aria-label*="Video"], div[data-testid*="videoPlayer"]')
                                img_elems = quoted_main.query_selector_all('img[src*="twimg.com"], img[src*="pbs.twimg.com"]')
                                if video_elems or img_elems:
                                    print(f"📸 Poll: Found {len(video_elems)} video and {len(img_elems)} image elements.")
                                    found_media = True
                                    # Try clicking video/play if present
                                    try:
                                        if video_elems:
                                            print(f"🖱️ Clicking video element...")
                                            video_elems[0].click()
                                            human_delay(1.0, 2.0)
                                        play_btn = quoted_main.query_selector('button[aria-label*="Play"], div[role="button"][aria-label*="Play"]')
                                        if play_btn:
                                            print(f"🖱️ Clicking play button...")
                                            play_btn.click()
                                            human_delay(1.0, 2.0)
                                    except Exception as e:
                                        print(f"⚠️ Video/play click failed: {e}")
                                    break
                            human_delay(1.0, 1.5)
                        if not found_media:
                            print("⚠️ No media found after polling.")

                        # Playwright stealth mode (if available)
                        try:
                            if hasattr(page.context, 'use_stealth'):
                                print("🕵️ Enabling Playwright stealth mode...")
                                page.context.use_stealth()
                        except Exception as e:
                            print(f"⚠️ Stealth mode not available: {e}")
                        if quoted_articles:
                            quoted_main = quoted_articles[0]

                            quoted_full_content = extract_full_tweet_content(quoted_main)
                            quoted_media_links, quoted_media_count, quoted_media_types = extract_media_data(quoted_main)
                            quoted_elements = extract_content_elements(quoted_main)

                            # Update post_analysis with full content
                            post_analysis['original_content'] = quoted_full_content

                            print(f"✅ Complete quoted tweet extracted: {len(quoted_full_content)} chars, {quoted_media_count} media")
                            if quoted_media_count > 0:
                                print(f"   🖼️ Media: {', '.join(quoted_media_links[:3])}")

                            return {
                                'content': quoted_full_content,
                                'media_links': quoted_media_links,
                                'media_count': quoted_media_count,
                                'media_types': quoted_media_types,
                                'hashtags': quoted_elements.get('hashtags', []),
                                'mentions': quoted_elements.get('mentions', []),
                            }
                        else:
                            print(f"⚠️ No article found after clicking quoted tweet")
                    except Exception as e:
                        print(f"⚠️ Could not click and extract quoted tweet: {e}")
                continue
            
            selector = selector_item
            if selector == 'article a[href*="/status/"] img[alt][src*="pbs.twimg.com"]':
                # Special case: image preview means there's a quoted tweet nearby
                img_elem = main_article.query_selector(selector)
                if img_elem:
                    # Get the parent link
                    parent = img_elem.evaluate_handle('el => el.closest("a[href*=\\"/status/\\"]")')
                    if parent:
                        quoted_card = parent.as_element()
                        print(f"✅ Found quoted card via image preview")
                        break
            else:
                quoted_card = main_article.query_selector(selector)
                if quoted_card:
                    print(f"✅ Found quoted card with selector: {selector}")
                    break
        except Exception as e:
            continue
    
    # Strategy 2.5: Look for ANY link to another tweet inside the main article
    if not quoted_card:
        print(f"🔍 Trying alternative: looking for any /status/ link in article...")
        try:
            all_links = main_article.query_selector_all('a[href*="/status/"]')
            for link in all_links:
                href = link.get_attribute('href')
                # Make sure it's not the main tweet's own link
                if href and '/status/' in href:
                    parts = href.strip('/').split('/')
                    if 'status' in parts:
                        status_index = parts.index('status')
                        if status_index >= 1:
                            linked_tweet_id = parts[status_index + 1].split('?')[0]
                            # If it's a different tweet ID, it might be a quoted tweet
                            if linked_tweet_id != post_analysis.get('tweet_id', ''):
                                quoted_card = link
                                print(f"✅ Found potential quoted tweet link: {href}")
                                break
        except Exception as e:
            print(f"⚠️ Alternative link search failed: {e}")
    
    if not quoted_card:
        print(f"⚠️ No quoted tweet card found in main article")
        return None
    
    # Extract quoted tweet URL and author
    try:
        # Try multiple methods to find the quoted tweet URL
        quoted_link = None
        quoted_href = None
        
        # Method 1: Direct link in quoted card
        quoted_link = quoted_card.query_selector('a[href*="/status/"]')
        if quoted_link:
            quoted_href = quoted_link.get_attribute('href')
        
        # Method 2: If quoted_card IS a link
        if not quoted_href and quoted_card.tag_name.lower() == 'a':
            quoted_href = quoted_card.get_attribute('href')
        
        # Method 3: Look for time element which usually has the link
        if not quoted_href:
            time_elem = quoted_card.query_selector('time')
            if time_elem:
                parent_link = time_elem.evaluate_handle('el => el.closest("a[href*=\\"/status/\\"]")').as_element()
                if parent_link:
                    quoted_href = parent_link.get_attribute('href')
        
        # Method 4: Search for ANY link with status inside the quoted card area
        if not quoted_href:
            all_links = quoted_card.query_selector_all('a[href*="/status/"]')
            for link in all_links:
                href = link.get_attribute('href')
                if href and '/status/' in href:
                    # Make sure it's not the main tweet
                    if post_analysis.get('tweet_id', '') not in href:
                        quoted_href = href
                        break
        
        if not quoted_href or '/status/' not in quoted_href:
            print(f"⚠️ No status link found in quoted card")
            return None
        
        # Extract author and tweet ID from URL
        parts = quoted_href.strip('/').split('/')
        status_index = parts.index('status') if 'status' in parts else -1
        if status_index < 1:
            print(f"⚠️ Cannot parse quoted tweet URL: {quoted_href}")
            return None
        
        quoted_author = parts[status_index - 1]
        quoted_tweet_id = parts[status_index + 1].split('?')[0]  # Remove query params
        quoted_tweet_url = f"https://x.com/{quoted_author}/status/{quoted_tweet_id}"
        
        post_analysis['reply_to_username'] = quoted_author
        post_analysis['reply_to_tweet_id'] = quoted_tweet_id
        
        print(f"🔗 Found quoted tweet by @{quoted_author}: {quoted_tweet_url}")
        
        # Try to extract embedded text first
        quoted_text_elem = quoted_card.query_selector('[data-testid="tweetText"]')
        if quoted_text_elem:
            embedded_text = quoted_text_elem.inner_text().strip()
            print(f"📄 Extracted embedded text: {len(embedded_text)} chars")
        else:
            embedded_text = None
        
        # Strategy 3: Visit the quoted tweet page for complete content
        print(f"🌐 Visiting quoted tweet page for complete extraction...")
        try:
            page.goto(quoted_tweet_url, wait_until="domcontentloaded", timeout=60000)
            human_delay(2.0, 3.0)
            
            # Extract complete quoted tweet
            quoted_articles = page.query_selector_all('article[data-testid="tweet"]')
            if quoted_articles:
                quoted_main = quoted_articles[0]
                
                quoted_full_content = extract_full_tweet_content(quoted_main)
                quoted_media_links, quoted_media_count, quoted_media_types = extract_media_data(quoted_main)
                quoted_elements = extract_content_elements(quoted_main)
                
                # Update post_analysis with full content
                post_analysis['original_content'] = quoted_full_content
                
                print(f"✅ Complete quoted tweet: {len(quoted_full_content)} chars, {quoted_media_count} media")
                if quoted_media_count > 0:
                    print(f"   🖼️ Media: {', '.join(quoted_media_links[:3])}")
                
                return {
                    'content': quoted_full_content,
                    'media_links': quoted_media_links,
                    'media_count': quoted_media_count,
                    'media_types': quoted_media_types,
                    'hashtags': quoted_elements.get('hashtags', []),
                    'mentions': quoted_elements.get('mentions', []),
                    'external_links': quoted_elements.get('external_links', [])
                }
        except Exception as e:
            print(f"⚠️ Could not visit quoted tweet: {e}")
            # Fall back to embedded text if available
            if embedded_text:
                post_analysis['original_content'] = embedded_text
                print(f"⚠️ Using embedded text as fallback")
                return {'content': embedded_text}
            
    except Exception as e:
        print(f"⚠️ Error extracting quoted tweet: {e}")
    
    return None
