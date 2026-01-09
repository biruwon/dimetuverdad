"""
Tests for fetcher/html_extractor.py module.

Tests the parallel HTML extraction performance optimization (P1).
"""

import pytest
from unittest.mock import Mock, patch

from fetcher.html_extractor import (
    parse_tweet_from_html,
    extract_all_article_html,
    _parse_author_and_id_from_href,
    _extract_text_from_soup,
    _extract_media_from_soup,
    _extract_engagement_from_soup,
    _analyze_post_type_from_soup,
    _has_thread_line_from_soup,
)
from bs4 import BeautifulSoup


# Sample tweet HTML fixtures
SAMPLE_TWEET_HTML = '''
<article data-testid="tweet">
    <div>
        <a href="/testuser/status/123456789" role="link">
            <time datetime="2024-01-15T10:30:00.000Z">Jan 15</time>
        </a>
        <div data-testid="tweetText">This is a test tweet with some content.</div>
        <div data-testid="reply" aria-label="5 replies">
            <span>5</span>
        </div>
        <div data-testid="retweet" aria-label="10 retweets">
            <span>10</span>
        </div>
        <div data-testid="like" aria-label="100 likes">
            <span>100</span>
        </div>
    </div>
</article>
'''

SAMPLE_RETWEET_HTML = '''
<article data-testid="tweet">
    <div data-testid="socialContext">Retweeted by testuser</div>
    <div>
        <a href="/originaluser" role="link">Original User</a>
        <a href="/originaluser/status/987654321" role="link">
            <time datetime="2024-01-14T08:00:00.000Z">Jan 14</time>
        </a>
        <div data-testid="tweetText">Original tweet content</div>
    </div>
</article>
'''

SAMPLE_QUOTE_TWEET_HTML = '''
<article data-testid="tweet">
    <div>
        <a href="/testuser/status/111222333" role="link">
            <time datetime="2024-01-15T12:00:00.000Z">Jan 15</time>
        </a>
        <div data-testid="tweetText">My commentary on this tweet</div>
        <div data-testid="card.wrapper">
            <a href="/quoteduser/status/444555666">Quoted tweet</a>
            <div data-testid="tweetText">Quoted content</div>
        </div>
    </div>
</article>
'''

SAMPLE_TWEET_WITH_MEDIA_HTML = '''
<article data-testid="tweet">
    <div>
        <a href="/testuser/status/999888777" role="link">
            <time datetime="2024-01-16T14:00:00.000Z">Jan 16</time>
        </a>
        <div data-testid="tweetText">Check out this image!</div>
        <div data-testid="tweetPhoto">
            <img src="https://pbs.twimg.com/media/abc123.jpg" />
        </div>
    </div>
</article>
'''

SAMPLE_THREAD_TWEET_HTML = '''
<article data-testid="tweet">
    <div class="r-1bimlpy">Thread line indicator</div>
    <div>
        <a href="/testuser/status/777666555" role="link">
            <time datetime="2024-01-17T09:00:00.000Z">Jan 17</time>
        </a>
        <div data-testid="tweetText">This is part of a thread</div>
    </div>
</article>
'''


class TestParseTweetFromHtml:
    """Test the main parse_tweet_from_html function."""
    
    def test_parse_basic_tweet(self):
        """Test parsing a basic tweet."""
        result = parse_tweet_from_html(SAMPLE_TWEET_HTML, 'testuser')
        
        assert result is not None
        assert result['tweet_id'] == '123456789'
        assert result['username'] == 'testuser'
        assert 'test tweet' in result['content'].lower()
        assert result['tweet_timestamp'] == '2024-01-15T10:30:00.000Z'
    
    def test_parse_tweet_wrong_user(self):
        """Test that tweets from wrong user return None."""
        result = parse_tweet_from_html(SAMPLE_TWEET_HTML, 'differentuser')
        
        assert result is None  # Should return None for non-target user
    
    def test_parse_retweet(self):
        """Test parsing a retweet."""
        result = parse_tweet_from_html(SAMPLE_RETWEET_HTML, 'originaluser')
        
        assert result is not None
        assert result['post_type'] == 'retweet'
    
    def test_parse_quote_tweet(self):
        """Test parsing a quote tweet."""
        result = parse_tweet_from_html(SAMPLE_QUOTE_TWEET_HTML, 'testuser')
        
        assert result is not None
        assert result['post_type'] == 'quote'
        assert result['original_author'] == 'quoteduser'
        assert result['original_tweet_id'] == '444555666'
    
    def test_parse_tweet_with_media(self):
        """Test parsing a tweet with media."""
        result = parse_tweet_from_html(SAMPLE_TWEET_WITH_MEDIA_HTML, 'testuser')
        
        assert result is not None
        assert result['media_count'] >= 1
        assert 'twimg.com' in result['media_links']
    
    def test_parse_thread_tweet(self):
        """Test parsing a tweet in a thread."""
        result = parse_tweet_from_html(SAMPLE_THREAD_TWEET_HTML, 'testuser')
        
        assert result is not None
        assert result['has_thread_line'] is True


class TestParseAuthorAndIdFromHref:
    """Test the href parsing utility."""
    
    def test_valid_href(self):
        """Test parsing a valid tweet href."""
        author, tweet_id = _parse_author_and_id_from_href('/username/status/123456')
        
        assert author == 'username'
        assert tweet_id == '123456'
    
    def test_href_with_query_params(self):
        """Test parsing href with query parameters."""
        author, tweet_id = _parse_author_and_id_from_href('/user/status/789?s=20')
        
        assert author == 'user'
        assert tweet_id == '789'
    
    def test_empty_href(self):
        """Test handling empty href."""
        author, tweet_id = _parse_author_and_id_from_href('')
        
        assert author is None
        assert tweet_id is None
    
    def test_invalid_href(self):
        """Test handling invalid href format."""
        author, tweet_id = _parse_author_and_id_from_href('/just/some/path')
        
        assert author is None
        assert tweet_id is None


class TestExtractTextFromSoup:
    """Test text extraction from BeautifulSoup."""
    
    def test_extract_from_tweet_text_element(self):
        """Test extraction from data-testid='tweetText'."""
        html = '<div data-testid="tweetText">Hello world!</div>'
        soup = BeautifulSoup(html, 'html.parser')
        
        text = _extract_text_from_soup(soup)
        
        assert text == 'Hello world!'
    
    def test_extract_handles_whitespace(self):
        """Test that whitespace is stripped."""
        html = '<div data-testid="tweetText">  \n  Test content  \n  </div>'
        soup = BeautifulSoup(html, 'html.parser')
        
        text = _extract_text_from_soup(soup)
        
        assert text == 'Test content'
    
    def test_extract_empty_content(self):
        """Test handling empty content."""
        html = '<div>No tweet text element here</div>'
        soup = BeautifulSoup(html, 'html.parser')
        
        text = _extract_text_from_soup(soup)
        
        assert text == ''


class TestExtractMediaFromSoup:
    """Test media extraction from BeautifulSoup."""
    
    def test_extract_image(self):
        """Test image URL extraction."""
        html = '<img src="https://pbs.twimg.com/media/test.jpg" />'
        soup = BeautifulSoup(html, 'html.parser')
        
        links, count, types = _extract_media_from_soup(soup)
        
        assert count == 1
        assert 'test.jpg' in links[0]
        assert types[0] == 'image'
    
    def test_skip_profile_images(self):
        """Test that profile images are skipped."""
        html = '<img src="https://pbs.twimg.com/profile_images/user.jpg" />'
        soup = BeautifulSoup(html, 'html.parser')
        
        links, count, types = _extract_media_from_soup(soup)
        
        assert count == 0
    
    def test_skip_video_thumbnails(self):
        """Test that video thumbnails are skipped."""
        html = '<img src="https://pbs.twimg.com/amplify_video_thumb/thumb.jpg" />'
        soup = BeautifulSoup(html, 'html.parser')
        
        links, count, types = _extract_media_from_soup(soup)
        
        assert count == 0


class TestExtractEngagementFromSoup:
    """Test engagement metrics extraction."""
    
    def test_extract_engagement_with_aria_labels(self):
        """Test extracting engagement from aria-labels."""
        html = '''
        <div>
            <div data-testid="reply" aria-label="5 replies"></div>
            <div data-testid="retweet" aria-label="10 Retweets"></div>
            <div data-testid="like" aria-label="100 likes"></div>
        </div>
        '''
        soup = BeautifulSoup(html, 'html.parser')
        
        engagement = _extract_engagement_from_soup(soup)
        
        assert engagement['engagement_replies'] == 5
        assert engagement['engagement_retweets'] == 10
        assert engagement['engagement_likes'] == 100
    
    def test_extract_engagement_with_k_suffix(self):
        """Test parsing counts with K suffix (thousands)."""
        html = '<div data-testid="like" aria-label="5.2K likes"></div>'
        soup = BeautifulSoup(html, 'html.parser')
        
        engagement = _extract_engagement_from_soup(soup)
        
        assert engagement['engagement_likes'] == 5200


class TestAnalyzePostTypeFromSoup:
    """Test post type analysis."""
    
    def test_detect_original_post(self):
        """Test detecting original post."""
        html = '<article><div data-testid="tweetText">Content</div></article>'
        soup = BeautifulSoup(html, 'html.parser')
        
        result = _analyze_post_type_from_soup(soup, 'testuser')
        
        assert result['post_type'] == 'original'
    
    def test_detect_retweet(self):
        """Test detecting retweet."""
        html = '''
        <article>
            <div data-testid="socialContext">Retweeted</div>
        </article>
        '''
        soup = BeautifulSoup(html, 'html.parser')
        
        result = _analyze_post_type_from_soup(soup, 'testuser')
        
        assert result['post_type'] == 'retweet'
    
    def test_detect_quote_tweet(self):
        """Test detecting quote tweet."""
        html = '''
        <article>
            <div data-testid="card.wrapper">
                <a href="/quoteduser/status/12345">Quote</a>
            </div>
        </article>
        '''
        soup = BeautifulSoup(html, 'html.parser')
        
        result = _analyze_post_type_from_soup(soup, 'testuser')
        
        assert result['post_type'] == 'quote'
        assert result['original_author'] == 'quoteduser'
        assert result['original_tweet_id'] == '12345'


class TestHasThreadLineFromSoup:
    """Test thread line detection."""
    
    def test_detect_thread_line_by_class(self):
        """Test detecting thread line by CSS class."""
        html = '<div class="r-1bimlpy">Thread indicator</div>'
        soup = BeautifulSoup(html, 'html.parser')
        
        result = _has_thread_line_from_soup(soup)
        
        assert result is True
    
    def test_no_thread_line(self):
        """Test no thread line detected."""
        html = '<div>Regular tweet</div>'
        soup = BeautifulSoup(html, 'html.parser')
        
        result = _has_thread_line_from_soup(soup)
        
        assert result is False


class TestExtractAllArticleHtml:
    """Test the batch HTML extraction."""
    
    def test_extract_all_articles(self):
        """Test extracting all articles from page."""
        mock_page = Mock()
        mock_page.evaluate.return_value = [
            {'html': '<article>Tweet 1</article>', 'rect': {}, 'index': 0},
            {'html': '<article>Tweet 2</article>', 'rect': {}, 'index': 1},
        ]
        
        result = extract_all_article_html(mock_page)
        
        assert len(result) == 2
        assert result[0]['html'] == '<article>Tweet 1</article>'
    
    def test_extract_handles_empty_page(self):
        """Test handling page with no articles."""
        mock_page = Mock()
        mock_page.evaluate.return_value = []
        
        result = extract_all_article_html(mock_page)
        
        assert result == []
    
    def test_extract_handles_error(self):
        """Test handling evaluation error."""
        mock_page = Mock()
        mock_page.evaluate.side_effect = Exception("Browser error")
        
        result = extract_all_article_html(mock_page)
        
        assert result == []
