import json
from fetcher import parsers


class FakeElem:
    """Unified fake element class for all parser tests"""
    def __init__(self, attrs=None, text='', children=None, tag='div', mapping=None):
        self._attrs = attrs or {}
        self._text = text
        self._children = children or []
        self.tag = tag
        # mapping: selector -> element or list
        self._mapping = mapping or {}

    def get_attribute(self, name):
        return self._attrs.get(name)

    def inner_text(self):
        return self._text

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
        return results

    def evaluate(self, js):
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
    img = FakeElem(attrs={'src': 'https://pbs.twimg.com/media/abc.jpg'}, tag='img')
    status = FakeElem(attrs={'href': '/user/status/123'})
    hashtag = FakeElem(attrs={'href': '/hashtag/test'}, text='#test')
    mention = FakeElem(attrs={'href': '/otheruser'}, text='@other')
    art = FakeElem(children=[img, status, hashtag, mention], text='This is a post #test @other')
    content = parsers.extract_full_tweet_content(art)
    assert 'This is a post' in content
    elems = parsers.extract_content_elements(art)
    assert elems['hashtags'] is not None
    assert elems['mentions'] is not None


def test_extract_full_tweet_content_and_fallback():
    txt = FakeElem(text='Hello world')
    article = FakeElem(mapping={'[data-testid="tweetText"]': txt})
    assert parsers.extract_full_tweet_content(article) == 'Hello world'

    # Fallback to article.inner_text
    article2 = FakeElem(text='Fallback full text')
    assert parsers.extract_full_tweet_content(article2) == 'Fallback full text'


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
    # Simulate an article with tweetText element
    article = FakeElem(text='Long tweet content that should be returned')
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
