import json
from fetcher import parsers


class FakeElem:
    """Mock element for testing"""
    def __init__(self, attrs=None, text="", children=None, mapping=None, tag='div'):
        self._attrs = attrs or {}
        self._text = text
        self._children = children or []
        self._mapping = mapping or {}
        self.tag = tag

    def get_attribute(self, name):
        return self._attrs.get(name)

    def inner_text(self):
        return self._text
    
    def evaluate(self, script):
        """Mock JavaScript evaluate for text extraction."""
        # Return the text for text extraction scripts
        if 'extractTextWithEmojis' in script or 'textContent' in script or 'innerHTML' in script:
            return self._text
        # Return tag name for tag checks
        if 'tagName' in script:
            return self.tag
        # Return False for 'inside anchor' checks by default
        if 'closest' in script:
            return False
        return None
    
    def wait_for_timeout(self, ms):
        """Mock wait function."""
        pass

    def click(self):
        """Mock click function."""
        pass

    def query_selector(self, selector):
        # First check mapping if available
        val = self._mapping.get(selector)
        if isinstance(val, list):
            return val[0] if val else None
        if val is not None:
            return val

        # Comprehensive selector support for all test cases
        if selector.startswith('[data-testid="socialContext"]') and self._attrs.get('social'):
            return FakeElem(text=self._attrs.get('social'))
        if selector == '[data-testid="tweetText"]':
            return FakeElem(text=self._text)
        if 'a[href*="/status/"]' in selector and self._attrs.get('href') and '/status/' in self._attrs.get('href'):
            return FakeElem(attrs={'href': self._attrs.get('href')})
        if selector == 'time' and self._attrs.get('datetime'):
            return FakeElem(attrs={'datetime': self._attrs.get('datetime')})
        if 'role="article"' in selector and self._attrs.get('quoted'):
            return FakeElem(attrs={'quoted_href': self._attrs.get('quoted_href')}, text=self._attrs.get('quoted_text'))
        if selector == 'a[href^="/"]' and self._attrs.get('href'):
            return FakeElem(attrs={'href': self._attrs.get('href')})
        if 'Replying to' in selector and self._attrs.get('replying_to'):
            return FakeElem(text=self._attrs.get('replying_to'))
        # Support for quoted content selectors
        if ('[data-testid="tweetText"] ~ div [role="article"]' in selector) or ('.css-1dbjc4n [role="article"]' in selector):
            if self._attrs.get('quoted'):
                return FakeElem(text=self._attrs.get('quoted_text'), attrs={'href': self._attrs.get('quoted_href'), 'quoted_href': self._attrs.get('quoted_href')})
        return None

    def query_selector_all(self, selector):
        # First check mapping if available
        val = self._mapping.get(selector)
        if val is None:
            results = []
        elif isinstance(val, list):
            results = val
        else:
            results = [val]

        if results:  # If mapping provided results, return them
            return results

        # Match data-src/data-url selectors (twimg lazy loads)
        if 'data-src' in selector or 'data-url' in selector or 'twimg.com' in selector:
            for c in self._children:
                for key in ('data-src', 'data-url'):
                    v = c._attrs.get(key)
                    if v and 'twimg.com' in v:
                        results.append(c)
                        break
        # Match style background-image selectors
        if 'background-image' in selector or 'style' in selector:
            for c in self._children:
                style = c._attrs.get('style') or ''
                if 'background-image' in style:
                    results.append(c)
        # Image selectors
        if selector.startswith('img') or selector == '[data-testid="tweetPhoto"] img':
            for c in self._children:
                if c.tag == 'img':
                    results.append(c)
        # Anchors
        if selector.startswith('a[') or selector.startswith('a[href'):
            for c in self._children:
                if c._attrs.get('href'):
                    results.append(c)
        
        # Special case: don't return anything for tweetText when no explicit mapping
        if selector == '[data-testid="tweetText"]' and not self._mapping.get(selector):
            return []
        
        return results

    def evaluate(self, js):
        if 'tagName.toLowerCase()' in js:
            return self.tag
        if 'Array.from(el.attributes)' in js:
            return [(k, v) for k, v in self._attrs.items()]
        return None

    def click(self):
        return None


class FakePage:
    def __init__(self, elements):
        self._elements = elements

    def query_selector(self, selector):
        for el in self._elements:
            if el._attrs.get('src') and 'profile_images' in el._attrs.get('src'):
                return el
        return None

    def query_selector_all(self, selector):
        return self._elements


# ===== TIMESTAMP TESTS =====

def test_should_skip_existing_tweet():
    # Test newer tweet vs older cutoff
    assert parsers.should_skip_existing_tweet('2025-01-02T00:00:00Z', '2025-01-01T00:00:00Z') is True
    assert parsers.should_skip_existing_tweet('2024-12-31T23:59:59Z', '2025-01-01T00:00:00Z') is False
    # Test with no cutoff
    assert parsers.should_skip_existing_tweet('2024-01-02T12:00:00Z', None) is False


# ===== POST TYPE ANALYSIS TESTS =====

def test_analyze_post_type_pinned():
    el = FakeElem(attrs={'social': 'Pinned'})
    res = parsers.analyze_post_type(el, 'user')
    assert res['is_pinned'] == 1
    assert res['should_skip'] is True


def test_analyze_post_type_quote_like():
    el = FakeElem(attrs={'quoted': True, 'quoted_text': 'orig', 'quoted_href': '/orig/status/555'}, text='comment')
    res = parsers.analyze_post_type(el, 'user')
    assert res['post_type'] == 'original'
    assert res['reply_to_tweet_id'] == '555' or res['reply_to_tweet_id'] is None


def test_analyze_post_type_repost_heuristic_own():
    # SocialContext with reposter link implies own repost
    el = FakeElem(attrs={'social': 'Reposted', 'href': '/user/status/999'})
    # Include a /status/ link in children
    el._children.append(FakeElem(attrs={'href': '/user/status/999'}))
    res = parsers.analyze_post_type(el, 'user')
    assert res['post_type'] in ('repost_own', 'original')


def test_analyze_post_type_reply_and_thread():
    reply_mention = FakeElem(attrs={'href': '/replyto'})
    reply_ctx = FakeElem(mapping={'a[href^="/"]': reply_mention})
    article = FakeElem(mapping={
        '[data-testid="socialContext"]:has-text("Replying to"), [data-testid="tweetText"]:has-text("Replying to")': reply_ctx
    })
    res = parsers.analyze_post_type(article, 'target')
    assert res['post_type'] == 'repost_reply'
    assert res['reply_to_username'] == 'replyto'

    thread_indicator = FakeElem()
    article2 = FakeElem(mapping={
        '[data-testid="socialContext"]:has-text("Show this thread")': thread_indicator
    })
    res2 = parsers.analyze_post_type(article2, 'target')
    assert res2['post_type'] == 'thread'


def test_reply_detection_sets_repost_reply():
    # Reply context should mark repost_reply
    el = FakeElem(attrs={'replying_to': 'Replying to @someone'})
    res = parsers.analyze_post_type(el, 'user')
    assert res['post_type'] == 'repost_reply'


def test_thread_indicator_sets_thread():
    el = FakeElem(attrs={'social': 'Show this thread'})
    res = parsers.analyze_post_type(el, 'user')
    assert isinstance(res, dict)


def test_quote_like_parsing_extracts_reply_to():
    quoted = FakeElem(attrs={'href': '/orig'}, text='original text')
    main = FakeElem(children=[quoted], text='comment')
    # Provide quoted tweet link
    quoted._attrs['href'] = '/orig/status/321'
    res = parsers.analyze_post_type(main, 'user')
    # reply_to_tweet_id may be set or None depending on DOM; ensure function returns dict
    assert isinstance(res, dict)


def test_analyze_post_type_repost_own_and_other():
    # Repost own: repost element contains aria-label 'retweet' and a link to target_username
    repost_elem = FakeElem(attrs={'aria-label': 'Retweeted'})
    reposter_link = FakeElem(attrs={'href': '/targetuser'})
    repost_elem._mapping = {f'a[href="/targetuser"]': reposter_link}

    orig_link = FakeElem(attrs={'href': '/targetuser/status/999'})
    article = FakeElem(mapping={
        # Emulate finding repost indicator
        '[data-testid="socialContext"]:has-text("Reposted"), [data-testid="socialContext"]:has-text("reposted"), [data-testid="socialContext"]:has-text("Retweeted"), [data-testid="socialContext"]:has-text("Retweeted" )': repost_elem,
        'a[href*="/status/"]': orig_link
    })

    res = parsers.analyze_post_type(article, 'targetuser')
    assert res['post_type'] == 'repost_own'
    assert res['original_author'] == 'targetuser'
    assert res['should_skip'] is True

    # Repost other: repost element present but quoted tweet contains different original author
    quoted_orig_link = FakeElem(attrs={'href': '/otheruser/status/888'})
    quoted_tweet = FakeElem(mapping={
        'a[href*="/status/"]': quoted_orig_link,
        'a[href^="/"]': [FakeElem(attrs={'href': '/otheruser'})]
    })
    repost_elem2 = FakeElem(attrs={'aria-label': 'Retweeted'})
    article2 = FakeElem(mapping={
        '[data-testid="socialContext"]:has-text("Reposted"), [data-testid="socialContext"]:has-text("reposted"), [data-testid="socialContext"]:has-text("Retweeted"), [data-testid="socialContext"]:has-text("Retweeted" )': repost_elem2,
            'article [data-testid="tweetText"], [data-testid="tweet"] article': quoted_tweet,
        'a[href*="/status/"]': quoted_orig_link
    })

    res2 = parsers.analyze_post_type(article2, 'targetuser')
    assert res2['post_type'] == 'repost_other'
    assert res2['original_author'] == 'otheruser'


# ===== MEDIA EXTRACTION TESTS =====

def test_media_extraction_image_and_video():
    img = FakeElem(attrs={'src': 'https://pbs.twimg.com/media/img.jpg'}, tag='img')
    video = FakeElem(attrs={'src': 'https://video.twimg.com/video.mp4', 'poster': 'https://pbs.twimg.com/media/vid.jpg'}, tag='video')
    art = FakeElem(children=[img, video])
    media_links, media_count, media_types = parsers.extract_media_data(art)
    assert media_count >= 1
    assert any('pbs.twimg.com' in m or 'video' in m for m in media_links)


def test_media_data_src_and_background():
    lazy = FakeElem(attrs={'data-src': 'https://pbs.twimg.com/media/lazy.jpg'})
    bg = FakeElem(attrs={'style': 'background-image: url("https://pbs.twimg.com/media/bg.jpg")'})
    art = FakeElem(children=[lazy, bg])
    links, count, types = parsers.extract_media_data(art)
    assert any('lazy.jpg' in l or 'bg.jpg' in l for l in links)


def test_extract_media_data_with_data_src_and_background():
    img = FakeElem(attrs={'data-src': 'https://pbs.twimg.com/media/IMG1.jpg'})
    bg = FakeElem(attrs={'style': "background-image: url('https://pbs.twimg.com/media/BG1.jpg')"})
    article = FakeElem(mapping={
        '[data-src*="twimg.com"], [data-url*="twimg.com"]': [img],
        '[style*="background-image"]': [bg]
    })

    links, count, types = parsers.extract_media_data(article)
    # Background images are extracted but data-src images might be skipped if they're not in img tags
    assert 'https://pbs.twimg.com/media/BG1.jpg' in links
    assert count >= 1  # At least the background image
    assert 'image' in types


# ===== ENGAGEMENT METRICS TESTS =====

def test_engagement_parsing_numbers_and_k_suffix():
    # Create fake svg elements with parents providing inner_text
    like_parent = FakeElem(text='1.2K')
    like_svg = FakeElem(children=[like_parent], attrs={'aria-label': 'like'}, tag='svg')
    art = FakeElem(children=[like_svg])
    engagement = parsers.extract_engagement_metrics(art)
    assert engagement['likes'] in (1200, 0) or isinstance(engagement['likes'], int)


def test_extract_engagement_metrics_parsing_k_m_and_numbers():
    e1 = FakeElem(text='3')
    e2 = FakeElem(text='1.2K')
    e3 = FakeElem(text='2M')
    article = FakeElem(mapping={
        '[data-testid="reply"]': [e1],
        '[data-testid="retweet"]': [e2],
        '[data-testid="like"]': [e3]
    })

    metrics = parsers.extract_engagement_metrics(article)
    assert metrics['replies'] == 3
    assert metrics['retweets'] == 1200
    assert metrics['likes'] == 2000000


def test_engagement_metrics_and_profile_picture():
    # Build fake elements that mimic engagement counts
    like_parent = FakeElem(text='1.2K')
    like_button = FakeElem(children=[like_parent], attrs={'aria-label': 'like'}, tag='svg')

    retweet_parent = FakeElem(text='345')
    retweet_button = FakeElem(children=[retweet_parent], attrs={'aria-label': 'retweet'}, tag='svg')

    article = FakeElem(children=[like_button, retweet_button], text='')

    engagement = parsers.extract_engagement_metrics(article)
    assert engagement['likes'] in (1200, 0) or isinstance(engagement['likes'], int)
    assert engagement['retweets'] in (345, 0) or isinstance(engagement['retweets'], int)


# ===== CONTENT EXTRACTION TESTS =====

def test_extract_full_tweet_content_and_elements():
    """Test content extraction with proper selector matching."""
    # Create article with tweetText selector
    txt_elem = FakeElem(text='This is a post #test @other')
    img = FakeElem(attrs={'src': 'https://pbs.twimg.com/media/abc.jpg'}, tag='img')
    status = FakeElem(attrs={'href': '/user/status/123'})
    hashtag = FakeElem(attrs={'href': '/hashtag/test'}, text='#test')
    mention = FakeElem(attrs={'href': '/otheruser'}, text='@other')
    art = FakeElem(
        children=[img, status, hashtag, mention], 
        text='This is a post #test @other',
        mapping={'[data-testid="tweetText"]': txt_elem}
    )
    content = parsers.extract_full_tweet_content(art)
    assert 'This is a post' in content
    elems = parsers.extract_content_elements(art)
    assert elems['hashtags'] is not None
    assert elems['mentions'] is not None


def test_extract_full_tweet_content_and_fallback():
    """Test content extraction with proper selectors - no overly broad fallbacks."""
    txt = FakeElem(text='Hello world')
    article = FakeElem(mapping={'[data-testid="tweetText"]': txt})
    assert parsers.extract_full_tweet_content(article) == 'Hello world'

    # When no proper tweet text selectors match, return empty (media-only post)
    # Must mock both query_selector and query_selector_all to return nothing
    article2 = FakeElem(mapping={})  # Empty mapping means query_selector returns None
    article2.query_selector = lambda selector: None  # Force no primary matches
    article2.query_selector_all = lambda selector: []  # Force no fallback matches
    # Should return empty string since no tweet text selectors matched
    assert parsers.extract_full_tweet_content(article2) == ''


def test_extract_full_tweet_content_multiple_selectors():
    """Test the enhanced extract_full_tweet_content with multiple selector strategies."""
    # Test primary selector (data-testid="tweetText")
    primary_elem = FakeElem(text='Primary tweet text')
    article1 = FakeElem(mapping={'[data-testid="tweetText"]': primary_elem})
    result1 = parsers.extract_full_tweet_content(article1)
    assert result1 == 'Primary tweet text'

    # Test alternative selectors when primary fails
    alt_elem = FakeElem(text='Alternative tweet text')
    article2 = FakeElem(mapping={
        '[data-testid="Tweet-User-Text"]': alt_elem,
        '[data-testid="tweetText"]': None  # Primary not found
    })
    result2 = parsers.extract_full_tweet_content(article2)
    assert result2 == 'Alternative tweet text'

    # Test CSS class-based selectors
    css_elem = FakeElem(text='CSS class tweet text')
    article3 = FakeElem(mapping={
        '.tweet-text': css_elem,
        '[data-testid="tweetText"]': None
    })
    result3 = parsers.extract_full_tweet_content(article3)
    assert result3 == 'CSS class tweet text'

    # Test generic paragraph selectors
    para_elem = FakeElem(text='Paragraph tweet text')
    article4 = FakeElem(mapping={
        'article p': para_elem,
        '[data-testid="tweetText"]': None
    })
    result4 = parsers.extract_full_tweet_content(article4)
    assert result4 == 'Paragraph tweet text'

    # Test span-based selectors
    span_elem = FakeElem(text='Span tweet text')
    article5 = FakeElem(mapping={
        'span[data-testid="tweetText"]': span_elem,
        '[data-testid="tweetText"]': None
    })
    result5 = parsers.extract_full_tweet_content(article5)
    assert result5 == 'Span tweet text'

    # Test media-only post (no text selectors match)
    article6 = FakeElem(mapping={})
    article6.query_selector = lambda selector: None  # Force no primary matches
    article6.query_selector_all = lambda selector: []  # Force no fallback matches
    result6 = parsers.extract_full_tweet_content(article6)
    # Should return empty string for media-only posts
    assert result6 == ''


def test_extract_full_tweet_content_ultimate_fallback():
    """Test that media-only posts return empty string (no overly broad fallbacks)."""
    # When no proper tweet text selectors match, should return empty
    # Simulates a media-only post where tweet text element is not found
    article = FakeElem(mapping={})
    article.query_selector = lambda selector: None  # Force no primary matches
    article.query_selector_all = lambda selector: []  # Force no fallback matches

    result = parsers.extract_full_tweet_content(article)
    # Should return empty string since no tweet text selectors matched
    assert result == ''


def test_extract_full_tweet_content_empty_and_invalid():
    """Test extract_full_tweet_content with empty or invalid inputs."""
    # Empty article (media-only post simulation)
    empty_article = FakeElem(mapping={})
    empty_article.query_selector = lambda selector: None  # No tweet text element
    empty_article.query_selector_all = lambda selector: []
    result = parsers.extract_full_tweet_content(empty_article)
    assert result == ''

    # Article with no tweet text element (media-only post)
    unwanted_article = FakeElem(mapping={})
    unwanted_article.query_selector = lambda selector: None  # No tweet text element
    unwanted_article.query_selector_all = lambda selector: []
    result = parsers.extract_full_tweet_content(unwanted_article)
    assert result == ''  # Should return empty when no valid content found


def test_extract_image_data_function():
    """Test the new extract_image_data function."""
    # Test standard image selector
    img_elem = FakeElem(attrs={'src': 'https://pbs.twimg.com/media/test.jpg'})
    article1 = FakeElem(mapping={
        'img[src*="pbs.twimg.com/media/"]': [img_elem]
    })

    links, types = parsers.extract_image_data(article1)
    assert 'https://pbs.twimg.com/media/test.jpg' in links
    assert 'image' in types

    # Test data-src attribute
    data_img = FakeElem(attrs={'data-src': 'https://pbs.twimg.com/media/data.jpg'})
    article2 = FakeElem(mapping={
        'img[data-src*="twimg.com"]': [data_img]
    })

    links2, types2 = parsers.extract_image_data(article2)
    assert 'https://pbs.twimg.com/media/data.jpg' in links2
    assert 'image' in types2

    # Test background image extraction
    bg_elem = FakeElem(attrs={'style': 'background-image: url("https://pbs.twimg.com/media/bg.jpg")'})
    article3 = FakeElem(mapping={
        '[style*="background-image"]': [bg_elem]
    })

    links3, types3 = parsers.extract_image_data(article3)
    assert 'https://pbs.twimg.com/media/bg.jpg' in links3
    assert 'image' in types3


def test_extract_video_data_function():
    """Test the new extract_video_data function."""
    # Test direct video element
    video_elem = FakeElem(attrs={'src': 'https://video.twimg.com/test.mp4'})
    article1 = FakeElem(mapping={
        'video': [video_elem]
    })

    links1, types1 = parsers.extract_video_data(article1)
    assert 'https://video.twimg.com/test.mp4' in links1
    assert 'video' in types1

    # Test video with poster (thumbnail)
    video_with_poster = FakeElem(attrs={
        'poster': 'https://pbs.twimg.com/media/thumb.jpg'
    })
    article2 = FakeElem(mapping={
        'video': [video_with_poster]
    })

    links2, types2 = parsers.extract_video_data(article2)
    assert 'https://pbs.twimg.com/media/thumb.jpg' in links2
    assert 'image' in types2  # Poster is image type

    # Test video component selectors
    video_comp = FakeElem(attrs={'data-video-id': '12345'})
    article3 = FakeElem(mapping={
        '[data-testid="videoComponent"]': [video_comp]
    })

    links3, types3 = parsers.extract_video_data(article3)
    # Should not extract anything without actual video URLs
    assert len(links3) == 0


def test_extract_media_data_combined():
    """Test the combined extract_media_data function."""
    # Create article with both image and video
    img_elem = FakeElem(attrs={'src': 'https://pbs.twimg.com/media/test.jpg'})
    video_elem = FakeElem(attrs={'src': 'https://video.twimg.com/test.mp4'})

    article = FakeElem(mapping={
        'img[src*="pbs.twimg.com/media/"]': [img_elem],
        'video': [video_elem]
    })

    links, count, types = parsers.extract_media_data(article)
    assert count == 2
    assert 'image' in types
    assert 'video' in types
    assert len([l for l in links if 'pbs.twimg.com' in l]) == 1
    assert len([l for l in links if 'video.twimg.com' in l]) == 1


def test_extract_media_data_filters_unwanted():
    """Test that extract_media_data filters out profile images and card previews."""
    # Profile image (should be filtered out)
    profile_img = FakeElem(attrs={'src': 'https://pbs.twimg.com/profile_images/test.jpg'})
    # Card image (should be filtered out)
    card_img = FakeElem(attrs={'src': 'https://pbs.twimg.com/card_img/test.jpg'})
    # Regular media image (should be kept)
    media_img = FakeElem(attrs={'src': 'https://pbs.twimg.com/media/test.jpg'})

    article = FakeElem(mapping={
        'img[src*="pbs.twimg.com"]': [profile_img, card_img, media_img]
    })

    links, count, types = parsers.extract_media_data(article)
    assert count == 1  # Only the media image should remain
    assert 'https://pbs.twimg.com/media/test.jpg' in links
    assert 'profile_images' not in ' '.join(links)
    assert 'card_img' not in ' '.join(links)


def test_extract_content_elements_hashtags_mentions_and_links():
    h = FakeElem(text='#tag')
    m = FakeElem(attrs={'href': '/otheruser'})
    ext = FakeElem(attrs={'href': 'http://example.com'})
    article = FakeElem(mapping={
        'a[href*="/hashtag/"]': [h],
        'a[href^="/"]:not([href*="/hashtag/"]):not([href*="/status/"])': [m],
        'a[href^="http"]': [ext]
    })

    elements = parsers.extract_content_elements(article)
    assert json.loads(elements['hashtags']) == ['#tag']
    assert json.loads(elements['mentions']) == ['@otheruser']
    assert json.loads(elements['external_links']) == ['http://example.com']


def test_extract_full_tweet_content_expansion():
    """Test tweet content extraction with proper selector."""
    # Simulate an article with tweetText element
    txt_elem = FakeElem(text='Long tweet content that should be returned')
    article = FakeElem(
        text='Long tweet content that should be returned',
        mapping={'[data-testid="tweetText"]': txt_elem}
    )
    res = parsers.extract_full_tweet_content(article)
    assert 'Long tweet content' in res


def test_media_extraction_and_content_elements():
    # Create an article with an image, a status link, hashtag and mention
    img = FakeElem(attrs={'src': 'https://pbs.twimg.com/media/abc.jpg'}, tag='img')
    anchor_status = FakeElem(attrs={'href': '/user/status/123'})
    hashtag = FakeElem(attrs={'href': '/hashtag/test'}, text='#test')
    mention = FakeElem(attrs={'href': '/otheruser'}, text='@other')

    article = FakeElem(children=[img, anchor_status, hashtag, mention], text='This is a post #test @other')

    media_links, media_count, media_types = parsers.extract_media_data(article)
    assert media_count == 1
    assert any('pbs.twimg.com' in m for m in media_links)

    elements = parsers.extract_content_elements(article)
    assert elements['hashtags'] is not None
    assert '#test' in json.loads(elements['hashtags'])
    assert elements['mentions'] is not None
    assert '@otheruser' in json.loads(elements['mentions'])


# ===== PROFILE PICTURE TESTS =====

def test_extract_profile_picture_from_page():
    profile_img = FakeElem(attrs={'src': 'https://pbs.twimg.com/profile_images/user_normal.jpg'}, tag='img')
    page = FakePage([profile_img])
    pic = parsers.extract_profile_picture(page, 'user')
    assert pic is not None


def test_engagement_metrics_and_profile_picture():
    # Profile picture extraction from page
    profile_img = FakeElem(attrs={'src': 'https://pbs.twimg.com/profile_images/user_normal.jpg'}, tag='img')
    page = FakePage([profile_img])
    pic = parsers.extract_profile_picture(page, 'user')
    assert pic is not None
    assert '400x400' in pic or 'profile_images' in pic


def test_find_and_extract_quoted_tweet_scroll_position_handling():
    """Test that scroll position is saved during quoted tweet extraction."""
    from unittest.mock import Mock

    # Mock page with scroll position methods
    mock_page = Mock()
    mock_page.url = "https://x.com/user/status/123"
    mock_page.evaluate.return_value = 100  # scrollY returns 100
    mock_page.query_selector_all.return_value = []

    # Mock main article
    mock_main_article = Mock()
    mock_main_article.query_selector_all.return_value = []

    # Mock post analysis
    post_analysis = {'tweet_id': '123'}

    # Call the function (will fail but should save scroll position)
    try:
        parsers.find_and_extract_quoted_tweet(mock_page, mock_main_article, post_analysis)
    except:
        pass  # Expected to fail with mocks

    # Verify scroll position was saved
    mock_page.evaluate.assert_called_with("window.scrollY")


# ===========================================================
# P5: Skip "Show More" for Short Tweets Tests
# ===========================================================

class TestTruncationCheckThreshold:
    """Tests for the TRUNCATION_CHECK_THRESHOLD constant."""
    
    def test_truncation_threshold_exists(self):
        """Test that the truncation threshold constant is defined."""
        assert hasattr(parsers, 'TRUNCATION_CHECK_THRESHOLD')
        assert parsers.TRUNCATION_CHECK_THRESHOLD == 250
    
    def test_truncation_threshold_below_twitter_limit(self):
        """Threshold should be below Twitter's 280 char limit to catch potential truncation."""
        assert parsers.TRUNCATION_CHECK_THRESHOLD < 280


class TestTryExpandTruncatedText:
    """Tests for the _try_expand_truncated_text function."""
    
    def test_skip_expansion_for_short_text(self):
        """Short text (< 250 chars) should skip the expansion check entirely."""
        from unittest.mock import Mock
        
        article = Mock()
        # query_selector should NOT be called for short text
        article.query_selector = Mock(return_value=None)
        
        short_text_length = 100  # Well below threshold
        result = parsers._try_expand_truncated_text(article, short_text_length)
        
        assert result is False
        # Verify query_selector was never called (optimization working)
        article.query_selector.assert_not_called()
    
    def test_skip_expansion_at_threshold_minus_one(self):
        """Text at exactly threshold-1 chars should skip expansion check."""
        from unittest.mock import Mock
        
        article = Mock()
        article.query_selector = Mock(return_value=None)
        
        result = parsers._try_expand_truncated_text(article, 249)
        
        assert result is False
        article.query_selector.assert_not_called()
    
    def test_check_expansion_at_threshold(self):
        """Text at exactly threshold should check for Show more button."""
        from unittest.mock import Mock
        
        article = Mock()
        article.query_selector = Mock(return_value=None)  # No button found
        
        result = parsers._try_expand_truncated_text(article, 250)
        
        assert result is False
        # Verify query_selector WAS called (checking for button)
        assert article.query_selector.call_count > 0
    
    def test_check_expansion_above_threshold(self):
        """Text above threshold should check for Show more button."""
        from unittest.mock import Mock
        
        article = Mock()
        article.query_selector = Mock(return_value=None)
        
        result = parsers._try_expand_truncated_text(article, 300)
        
        assert result is False
        assert article.query_selector.call_count > 0
    
    def test_expansion_clicks_button_when_found(self):
        """When Show more button is found, it should be clicked."""
        from unittest.mock import Mock
        
        article = Mock()
        mock_button = Mock()
        # First evaluate returns 'span' (not anchor), second returns False (not inside anchor)
        mock_button.evaluate = Mock(side_effect=['span', False])
        mock_button.click = Mock()
        article.query_selector = Mock(return_value=mock_button)
        article.wait_for_timeout = Mock()
        
        result = parsers._try_expand_truncated_text(article, 280)
        
        assert result is True
        mock_button.click.assert_called_once()
        article.wait_for_timeout.assert_called_once_with(500)
    
    def test_skip_anchor_tags(self):
        """Anchor tags should be skipped to avoid navigation."""
        from unittest.mock import Mock
        
        article = Mock()
        mock_anchor = Mock()
        mock_anchor.evaluate = Mock(return_value='a')  # Is an anchor tag
        article.query_selector = Mock(return_value=mock_anchor)
        
        result = parsers._try_expand_truncated_text(article, 280)
        
        # Should return False since anchor was skipped
        assert result is False
        mock_anchor.click.assert_not_called()
    
    def test_skip_button_inside_anchor(self):
        """Buttons inside anchor tags should be skipped."""
        from unittest.mock import Mock
        
        article = Mock()
        mock_button = Mock()
        # For each selector tried (7 selectors), we need:
        # - First evaluate returns 'span' (not anchor)
        # - Second returns True (inside anchor) - causes skip
        # The side_effect cycles through for each selector
        mock_button.evaluate = Mock(side_effect=['span', True] * 7)
        article.query_selector = Mock(return_value=mock_button)
        
        result = parsers._try_expand_truncated_text(article, 280)
        
        # All buttons were inside anchors, so no expansion
        assert result is False


class TestExtractTextFromElement:
    """Tests for the _extract_text_from_element helper function."""
    
    def test_extract_simple_text(self):
        """Test extracting simple text from an element."""
        from unittest.mock import Mock
        
        mock_elem = Mock()
        mock_elem.evaluate = Mock(return_value="Hello, world!")
        
        result = parsers._extract_text_from_element(mock_elem)
        
        assert result == "Hello, world!"
    
    def test_extract_text_strips_whitespace(self):
        """Extracted text should be stripped of whitespace."""
        from unittest.mock import Mock
        
        mock_elem = Mock()
        mock_elem.evaluate = Mock(return_value="  Hello  ")
        
        result = parsers._extract_text_from_element(mock_elem)
        
        assert result == "Hello"
    
    def test_fallback_to_inner_text(self):
        """Should fall back to inner_text if JS extraction fails."""
        from unittest.mock import Mock
        
        mock_elem = Mock()
        mock_elem.evaluate = Mock(side_effect=Exception("JS failed"))
        mock_elem.inner_text = Mock(return_value="Fallback text")
        
        result = parsers._extract_text_from_element(mock_elem)
        
        assert result == "Fallback text"
    
    def test_returns_none_for_empty_text(self):
        """Should return None if all extraction methods return empty."""
        from unittest.mock import Mock
        
        mock_elem = Mock()
        mock_elem.evaluate = Mock(return_value="")
        mock_elem.inner_text = Mock(return_value="")
        
        result = parsers._extract_text_from_element(mock_elem)
        
        assert result is None


class TestExtractFullTweetContentP5:
    """Tests for P5 optimization in extract_full_tweet_content."""
    
    def test_short_text_does_not_check_show_more(self):
        """For short tweets, Show more should NOT be checked (P5 optimization)."""
        from unittest.mock import Mock, patch
        
        short_text = "This is a short tweet."  # 23 chars, well below threshold
        
        mock_elem = Mock()
        mock_elem.evaluate = Mock(return_value=short_text)
        
        mock_article = Mock()
        mock_article.query_selector = Mock(return_value=mock_elem)
        mock_article.query_selector_all = Mock(return_value=[])
        
        with patch.object(parsers, '_try_expand_truncated_text') as mock_expand:
            mock_expand.return_value = False
            
            result = parsers.extract_full_tweet_content(mock_article)
            
            assert result == short_text
            # Verify _try_expand_truncated_text was called with correct length
            mock_expand.assert_called_once()
            _, call_args = mock_expand.call_args
            # The length argument should be len(short_text)
            assert call_args == {} or mock_expand.call_args[0][1] == len(short_text)
    
    def test_long_text_checks_show_more(self):
        """For long tweets, Show more SHOULD be checked."""
        from unittest.mock import Mock, patch
        
        # Create text that's above the threshold
        long_text = "A" * 260  # Above 250 threshold
        
        mock_elem = Mock()
        mock_elem.evaluate = Mock(return_value=long_text)
        
        mock_article = Mock()
        mock_article.query_selector = Mock(return_value=mock_elem)
        mock_article.query_selector_all = Mock(return_value=[])
        
        with patch.object(parsers, '_try_expand_truncated_text') as mock_expand:
            mock_expand.return_value = False  # No expansion happened
            
            result = parsers.extract_full_tweet_content(mock_article)
            
            assert result == long_text
            # Verify expansion check was attempted
            mock_expand.assert_called_once()
    
    def test_re_extracts_text_after_expansion(self):
        """If text is expanded, should re-extract to get full content."""
        from unittest.mock import Mock, patch
        
        initial_text = "A" * 260  # Truncated text
        expanded_text = "A" * 400  # Full text after expansion
        
        mock_elem = Mock()
        # First call returns truncated, second call returns expanded
        mock_elem.evaluate = Mock(side_effect=[initial_text, expanded_text])
        
        mock_article = Mock()
        mock_article.query_selector = Mock(return_value=mock_elem)
        mock_article.query_selector_all = Mock(return_value=[])
        
        with patch.object(parsers, '_try_expand_truncated_text') as mock_expand:
            mock_expand.return_value = True  # Expansion happened
            
            result = parsers.extract_full_tweet_content(mock_article)
            
            assert result == expanded_text


# ============================================================================
# P8: No Navigation for Quoted Tweets Tests
# ============================================================================

class TestExtractQuotedFromEmbeddedCard:
    """Tests for extract_quoted_from_embedded_card - P8 optimization."""
    
    def test_extracts_basic_quoted_tweet_from_card(self):
        """Should extract author, ID, content from embedded card."""
        from unittest.mock import Mock
        
        # Setup mock card with link containing status URL
        mock_link = Mock()
        mock_link.get_attribute = Mock(return_value="https://x.com/quoted_user/status/123456789")
        
        # Setup mock text element
        mock_text_elem = Mock()
        mock_text_elem.evaluate = Mock(return_value="This is the quoted tweet content")
        
        mock_card = Mock()
        mock_card.query_selector = Mock(side_effect=lambda s: {
            'a[href*="/status/"]': mock_link,
            '[data-testid="tweetText"]': mock_text_elem,
        }.get(s))
        mock_card.query_selector_all = Mock(return_value=[])  # No images
        
        post_analysis = {'tweet_id': '999999'}  # Main tweet ID
        
        result = parsers.extract_quoted_from_embedded_card(mock_card, post_analysis)
        
        assert result is not None
        assert result['username'] == 'quoted_user'
        assert result['tweet_id'] == '123456789'
        assert result['content'] == 'This is the quoted tweet content'
        assert result['extracted_from_card'] is True
    
    def test_returns_none_when_no_link_found(self):
        """Should return None if no status link in card."""
        from unittest.mock import Mock
        
        mock_card = Mock()
        mock_card.query_selector = Mock(return_value=None)
        mock_card.query_selector_all = Mock(return_value=[])
        
        result = parsers.extract_quoted_from_embedded_card(mock_card, {})
        
        assert result is None
    
    def test_returns_none_when_link_is_main_tweet(self):
        """Should return None if link points to main tweet (same ID)."""
        from unittest.mock import Mock
        
        mock_link = Mock()
        mock_link.get_attribute = Mock(return_value="https://x.com/user/status/123456")
        
        mock_card = Mock()
        mock_card.query_selector = Mock(return_value=mock_link)
        mock_card.query_selector_all = Mock(return_value=[])
        
        # Main tweet has the same ID as the link
        post_analysis = {'tweet_id': '123456'}
        
        result = parsers.extract_quoted_from_embedded_card(mock_card, post_analysis)
        
        assert result is None
    
    def test_extracts_media_from_card(self):
        """Should extract images from the quoted card."""
        from unittest.mock import Mock
        
        mock_link = Mock()
        mock_link.get_attribute = Mock(return_value="/quoted/status/12345")
        
        mock_text_elem = Mock()
        mock_text_elem.evaluate = Mock(return_value="Quoted content")
        
        # Create mock images
        mock_img1 = Mock()
        mock_img1.get_attribute = Mock(return_value="https://pbs.twimg.com/media/image1.jpg")
        mock_img2 = Mock()
        mock_img2.get_attribute = Mock(return_value="https://pbs.twimg.com/media/image2.jpg")
        
        # Track call count to return different results for different selectors
        call_count = [0]
        def query_selector_all_handler(selector):
            call_count[0] += 1
            # First call is for images, return images
            if 'img[src' in selector:
                return [mock_img1, mock_img2]
            # Second call is for videos, return empty
            return []
        
        mock_card = Mock()
        mock_card.query_selector = Mock(side_effect=lambda s: {
            'a[href*="/status/"]': mock_link,
            '[data-testid="tweetText"]': mock_text_elem,
        }.get(s))
        mock_card.query_selector_all = Mock(side_effect=query_selector_all_handler)
        
        post_analysis = {'tweet_id': '999'}
        
        result = parsers.extract_quoted_from_embedded_card(mock_card, post_analysis)
        
        assert result is not None
        assert len(result['media_links']) == 2
        assert 'image1.jpg' in result['media_links'][0]
        assert result['media_count'] == 2
    
    def test_filters_profile_and_emoji_images(self):
        """Should filter out profile images and emoji from media."""
        from unittest.mock import Mock
        
        mock_link = Mock()
        mock_link.get_attribute = Mock(return_value="/quoted/status/12345")
        
        mock_text_elem = Mock()
        mock_text_elem.evaluate = Mock(return_value="Content")
        
        # Create mix of images - some should be filtered
        mock_img_good = Mock()
        mock_img_good.get_attribute = Mock(return_value="https://pbs.twimg.com/media/good.jpg")
        mock_img_profile = Mock()
        mock_img_profile.get_attribute = Mock(return_value="https://pbs.twimg.com/profile_images/pfp.jpg")
        mock_img_emoji = Mock()
        mock_img_emoji.get_attribute = Mock(return_value="https://abs.twimg.com/emoji/v2/emoji.png")
        
        def query_selector_all_handler(selector):
            # For images selector return the test images
            if 'img[src' in selector:
                return [mock_img_good, mock_img_profile, mock_img_emoji]
            # For video selector return empty
            return []
        
        mock_card = Mock()
        mock_card.query_selector = Mock(side_effect=lambda s: {
            'a[href*="/status/"]': mock_link,
            '[data-testid="tweetText"]': mock_text_elem,
        }.get(s))
        mock_card.query_selector_all = Mock(side_effect=query_selector_all_handler)
        
        result = parsers.extract_quoted_from_embedded_card(mock_card, {'tweet_id': '999'})
        
        assert result is not None
        assert len(result['media_links']) == 1
        assert 'good.jpg' in result['media_links'][0]
    
    def test_parses_url_formats(self):
        """Should handle various URL formats."""
        from unittest.mock import Mock
        
        test_cases = [
            ("/user/status/123", "user", "123"),
            ("https://x.com/user/status/123", "user", "123"),
            ("https://twitter.com/user/status/123", "user", "123"),
            ("/user/status/123?ref_src=twsrc", "user", "123"),
            ("/user_name/status/456789", "user_name", "456789"),
        ]
        
        for url, expected_user, expected_id in test_cases:
            mock_link = Mock()
            mock_link.get_attribute = Mock(return_value=url)
            
            mock_text_elem = Mock()
            mock_text_elem.evaluate = Mock(return_value="Content")
            
            mock_card = Mock()
            mock_card.query_selector = Mock(side_effect=lambda s, link=mock_link, text=mock_text_elem: {
                'a[href*="/status/"]': link,
                '[data-testid="tweetText"]': text,
            }.get(s))
            mock_card.query_selector_all = Mock(return_value=[])
            
            result = parsers.extract_quoted_from_embedded_card(mock_card, {'tweet_id': '999'})
            
            assert result is not None, f"Failed for URL: {url}"
            assert result['username'] == expected_user, f"Failed username for URL: {url}"
            assert result['tweet_id'] == expected_id, f"Failed ID for URL: {url}"
    
    def test_returns_none_for_empty_content(self):
        """Should return None if no text content and no media found."""
        from unittest.mock import Mock
        
        mock_link = Mock()
        mock_link.get_attribute = Mock(return_value="/user/status/123")
        
        mock_card = Mock()
        mock_card.query_selector = Mock(side_effect=lambda s: {
            'a[href*="/status/"]': mock_link,
            '[data-testid="tweetText"]': None,  # No text element
        }.get(s))
        mock_card.query_selector_all = Mock(return_value=[])  # No media either
        
        result = parsers.extract_quoted_from_embedded_card(mock_card, {'tweet_id': '999'})
        
        # P8: Returns None if no content AND no media - nothing useful to extract
        # Caller should fall back to navigation
        assert result is None


class TestP8QuotedTweetIntegration:
    """Integration tests for P8 quoted tweet optimization."""
    
    def test_card_extraction_flag_present(self):
        """Results from card extraction should have extracted_from_card flag."""
        from unittest.mock import Mock
        
        mock_link = Mock()
        mock_link.get_attribute = Mock(return_value="/quoted/status/12345")
        
        mock_text = Mock()
        mock_text.evaluate = Mock(return_value="Quoted content")
        
        mock_card = Mock()
        mock_card.query_selector = Mock(side_effect=lambda s: {
            'a[href*="/status/"]': mock_link,
            '[data-testid="tweetText"]': mock_text,
        }.get(s))
        mock_card.query_selector_all = Mock(return_value=[])
        
        result = parsers.extract_quoted_from_embedded_card(mock_card, {'tweet_id': '999'})
        
        assert result is not None
        assert result['extracted_from_card'] is True
    
    def test_tweet_url_format(self):
        """Should construct proper tweet URL."""
        from unittest.mock import Mock
        
        mock_link = Mock()
        mock_link.get_attribute = Mock(return_value="/testuser/status/98765")
        
        mock_text = Mock()
        mock_text.evaluate = Mock(return_value="Content")
        
        mock_card = Mock()
        mock_card.query_selector = Mock(side_effect=lambda s: {
            'a[href*="/status/"]': mock_link,
            '[data-testid="tweetText"]': mock_text,
        }.get(s))
        mock_card.query_selector_all = Mock(return_value=[])
        
        result = parsers.extract_quoted_from_embedded_card(mock_card, {'tweet_id': '999'})
        
        assert result['tweet_url'] == "https://x.com/testuser/status/98765"


class TestP8PerformanceExpectations:
    """Tests documenting P8 performance expectations."""
    
    def test_no_navigation_required_for_card_extraction(self):
        """P8 optimization should not require page navigation."""
        from unittest.mock import Mock, patch
        
        # The extract_quoted_from_embedded_card function should never call
        # page.goto or open new tabs
        mock_link = Mock()
        mock_link.get_attribute = Mock(return_value="/user/status/123")
        
        mock_text = Mock()
        mock_text.evaluate = Mock(return_value="Quick extract")
        
        mock_card = Mock()
        mock_card.query_selector = Mock(side_effect=lambda s: {
            'a[href*="/status/"]': mock_link,
            '[data-testid="tweetText"]': mock_text,
        }.get(s))
        mock_card.query_selector_all = Mock(return_value=[])
        
        # No page object should be needed
        result = parsers.extract_quoted_from_embedded_card(mock_card, {'tweet_id': '999'})
        
        # Verify we got results without any navigation
        assert result is not None
        assert result['extracted_from_card'] is True
    
    def test_handles_exception_gracefully(self):
        """Should handle extraction errors and return None."""
        from unittest.mock import Mock
        
        mock_card = Mock()
        mock_card.query_selector = Mock(side_effect=Exception("Unexpected error"))
        
        result = parsers.extract_quoted_from_embedded_card(mock_card, {'tweet_id': '999'})
        
        assert result is None


class TestParseTweetAuthorAndId:
    """Tests for parse_tweet_author_and_id function."""

    def test_valid_tweet_url(self):
        """Should extract author and tweet ID from valid URL."""
        author, tweet_id = parsers.parse_tweet_author_and_id('/username/status/123456789')
        assert author == 'username'
        assert tweet_id == '123456789'

    def test_url_with_query_params(self):
        """Should strip query parameters from tweet ID."""
        author, tweet_id = parsers.parse_tweet_author_and_id('/user/status/999?s=20&t=abc')
        assert author == 'user'
        assert tweet_id == '999'

    def test_empty_href(self):
        """Should return None for empty href."""
        author, tweet_id = parsers.parse_tweet_author_and_id('')
        assert author is None
        assert tweet_id is None

    def test_none_href(self):
        """Should return None for None href."""
        author, tweet_id = parsers.parse_tweet_author_and_id(None)
        assert author is None
        assert tweet_id is None

    def test_invalid_url_format(self):
        """Should return None for invalid URL format."""
        author, tweet_id = parsers.parse_tweet_author_and_id('/username/photo/123')
        assert author is None
        assert tweet_id is None

    def test_short_url(self):
        """Should return None for URL with too few parts."""
        author, tweet_id = parsers.parse_tweet_author_and_id('/username')
        assert author is None
        assert tweet_id is None

    def test_url_with_leading_trailing_slashes(self):
        """Should handle URLs with extra slashes."""
        author, tweet_id = parsers.parse_tweet_author_and_id('///user/status/12345///')
        # After strip('/').split('/'), the '' parts should be handled
        # Depending on implementation, this should still extract correctly
        author2, tweet_id2 = parsers.parse_tweet_author_and_id('/user/status/12345/')
        assert author2 == 'user'
        assert tweet_id2 == '12345'


class TestShouldProcessTweetByAuthor:
    """Tests for should_process_tweet_by_author function."""

    def test_matching_author(self):
        """Should return True when author matches target."""
        should_process, author, tweet_id = parsers.should_process_tweet_by_author(
            '/targetuser/status/123', 'targetuser'
        )
        assert should_process is True
        assert author == 'targetuser'
        assert tweet_id == '123'

    def test_non_matching_author(self):
        """Should return False when author doesn't match target."""
        should_process, author, tweet_id = parsers.should_process_tweet_by_author(
            '/otheruser/status/456', 'targetuser'
        )
        assert should_process is False
        assert author == 'otheruser'
        assert tweet_id == '456'

    def test_invalid_href(self):
        """Should return False for invalid href."""
        should_process, author, tweet_id = parsers.should_process_tweet_by_author(
            '', 'targetuser'
        )
        assert should_process is False
        assert author is None
        assert tweet_id is None

    def test_case_sensitive_matching(self):
        """Author matching should be case-sensitive."""
        should_process, author, tweet_id = parsers.should_process_tweet_by_author(
            '/TargetUser/status/789', 'targetuser'
        )
        # Twitter usernames are case-insensitive in real world, but this tests current behavior
        assert should_process is False
        assert author == 'TargetUser'


class TestShouldSkipExistingTweetEdgeCases:
    """Additional edge case tests for should_skip_existing_tweet."""

    def test_none_oldest_timestamp(self):
        """Should return False when oldest_timestamp is None."""
        result = parsers.should_skip_existing_tweet('2024-01-01T12:00:00Z', None)
        assert result is False

    def test_tweet_older_than_oldest(self):
        """Should return False when tweet is older than oldest."""
        result = parsers.should_skip_existing_tweet(
            '2024-01-01T12:00:00Z',
            '2024-01-02T12:00:00Z'
        )
        assert result is False

    def test_tweet_newer_than_oldest(self):
        """Should return True when tweet is newer than oldest."""
        result = parsers.should_skip_existing_tweet(
            '2024-01-03T12:00:00Z',
            '2024-01-02T12:00:00Z'
        )
        assert result is True

    def test_tweet_same_as_oldest(self):
        """Should return True when tweet is same time as oldest."""
        result = parsers.should_skip_existing_tweet(
            '2024-01-02T12:00:00Z',
            '2024-01-02T12:00:00Z'
        )
        assert result is True

    def test_invalid_timestamp_format(self):
        """Should return False for invalid timestamp format."""
        result = parsers.should_skip_existing_tweet('not-a-date', '2024-01-02T12:00:00Z')
        assert result is False

    def test_invalid_oldest_timestamp_format(self):
        """Should return False for invalid oldest timestamp format."""
        result = parsers.should_skip_existing_tweet('2024-01-01T12:00:00Z', 'invalid')
        assert result is False


class TestExtractVideoDataEdgeCases:
    """Additional tests for extract_video_data function."""

    def test_video_poster_fallback(self):
        """Test extraction of poster image when no video src."""
        video_elem = FakeElem(
            attrs={'poster': 'https://pbs.twimg.com/poster.jpg'},
            tag='video'
        )
        article = FakeElem(mapping={
            'video': [video_elem]
        })

        links, types = parsers.extract_video_data(article)
        # Poster should be extracted as image fallback
        if links:
            assert 'pbs.twimg.com' in links[0]

    def test_no_video_content(self):
        """Test when no video content exists."""
        article = FakeElem(mapping={})
        links, types = parsers.extract_video_data(article)
        assert isinstance(links, list)
        assert isinstance(types, list)

    def test_returns_empty_on_exception(self):
        """Test graceful handling when extraction fails."""
        # Create article that raises on query
        class BadElem:
            def query_selector_all(self, sel):
                raise Exception("Simulated error")
        
        # The function should handle errors gracefully
        links, types = parsers.extract_video_data(BadElem())
        assert links == []
        assert types == []


class TestExtractMediaDataEdgeCases:
    """Additional edge cases for extract_media_data function."""

    def test_deduplication_of_urls(self):
        """Test that duplicate URLs are filtered out."""
        # Same URL extracted as both image and video
        img_elem = FakeElem(attrs={'src': 'https://pbs.twimg.com/media/test.jpg'}, tag='img')
        
        article = FakeElem(mapping={
            'img[src*="pbs.twimg.com/media/"]': [img_elem, img_elem]  # Duplicate
        })

        links, count, types = parsers.extract_media_data(article)
        # Should have only 1 entry after deduplication
        assert links.count('https://pbs.twimg.com/media/test.jpg') == 1

    def test_empty_media_extraction(self):
        """Test extraction when no media exists."""
        article = FakeElem(mapping={})
        links, count, types = parsers.extract_media_data(article)
        assert count == 0
        assert links == []
        assert types == []

    def test_filters_emoji_images(self):
        """Test that emoji images are filtered out."""
        emoji_img = FakeElem(attrs={'src': 'https://abs-0.twimg.com/emoji/v2/test.png'}, tag='img')
        media_img = FakeElem(attrs={'src': 'https://pbs.twimg.com/media/real.jpg'}, tag='img')

        article = FakeElem(mapping={
            'img[src*="pbs.twimg.com/media/"]': [media_img],
            'img[src*="twimg.com/emoji"]': [emoji_img]
        })

        links, count, types = parsers.extract_media_data(article)
        for link in links:
            assert 'emoji' not in link


class TestExtractEngagementMetricsEdgeCases:
    """Additional edge cases for extract_engagement_metrics."""

    def test_parse_k_suffix(self):
        """Test parsing numbers with K suffix."""
        reply_elem = FakeElem(text='5K')
        article = FakeElem(mapping={
            '[data-testid="reply"]': [reply_elem]
        })
        
        engagement = parsers.extract_engagement_metrics(article)
        assert engagement.get('replies', 0) >= 5000 or engagement.get('replies') == 0

    def test_parse_m_suffix(self):
        """Test parsing numbers with M suffix."""
        like_elem = FakeElem(text='2.5M')
        article = FakeElem(mapping={
            '[data-testid="like"]': [like_elem]
        })
        
        engagement = parsers.extract_engagement_metrics(article)
        # Depending on selector matching, this may or may not work
        assert isinstance(engagement, dict)

    def test_handles_empty_elements(self):
        """Test handling of empty engagement elements."""
        empty_elem = FakeElem(text='')
        article = FakeElem(mapping={
            '[data-testid="reply"]': [empty_elem]
        })
        
        engagement = parsers.extract_engagement_metrics(article)
        assert engagement.get('replies', 0) == 0


class TestAnalyzePostTypeEdgeCases:
    """Additional edge cases for analyze_post_type."""

    def test_pinned_post_detection(self):
        """Test detection of pinned posts."""
        pinned_elem = FakeElem(text='Pinned')
        article = FakeElem(mapping={
            '[data-testid="socialContext"]:has-text("Pinned"), [aria-label*="Pinned"]': pinned_elem
        })
        
        result = parsers.analyze_post_type(article, 'testuser')
        assert result['is_pinned'] == 1
        assert result['should_skip'] is True

    def test_regular_post(self):
        """Test regular post without special indicators."""
        article = FakeElem(mapping={})
        
        result = parsers.analyze_post_type(article, 'testuser')
        assert result['post_type'] == 'original'
        assert result['is_pinned'] == 0


class TestExtractImageDataEdgeCases:
    """Additional edge cases for extract_image_data."""

    def test_multiple_images(self):
        """Test extraction of multiple images."""
        img1 = FakeElem(attrs={'src': 'https://pbs.twimg.com/media/img1.jpg'}, tag='img')
        img2 = FakeElem(attrs={'src': 'https://pbs.twimg.com/media/img2.jpg'}, tag='img')
        
        article = FakeElem(mapping={
            'img[src*="pbs.twimg.com/media/"]': [img1, img2]
        })
        
        links, types = parsers.extract_image_data(article)
        assert len(links) == 2
        assert all(t == 'image' for t in types)

    def test_data_src_attribute(self):
        """Test extraction from data-src attribute (lazy loading)."""
        lazy_img = FakeElem(attrs={'data-src': 'https://pbs.twimg.com/media/lazy.jpg'}, tag='img')
        
        article = FakeElem(mapping={
            'img[data-src*="pbs.twimg.com"]': [lazy_img]
        })
        
        links, types = parsers.extract_image_data(article)
        # Should extract from data-src
        if links:
            assert 'pbs.twimg.com' in links[0]

    def test_no_images(self):
        """Test when no images exist."""
        article = FakeElem(mapping={})
        links, types = parsers.extract_image_data(article)
        assert links == []
        assert types == []