"""
Fake Playwright objects for testing.

This module provides fake implementations of Playwright's Page, Context, 
ElementHandle, and Locator classes that can be used for unit testing
without requiring a real browser.

Usage:
    from fetcher.tests.fake_playwright import FakePage, FakeElement
    
    # Create a fake page with articles
    page = FakePage()
    page.add_article(FakeElement.tweet_article(
        tweet_id="123",
        content="Hello world",
        author="testuser"
    ))
    
    # Test your code
    articles = page.query_selector_all('article')
"""

from typing import Dict, List, Optional, Any, Callable
import re


class FakeElement:
    """
    Fake Playwright ElementHandle for testing.
    
    Simulates Playwright's element queries and attribute access
    without requiring a real browser.
    """
    
    def __init__(
        self,
        tag: str = "div",
        attributes: Optional[Dict[str, str]] = None,
        text_content: str = "",
        inner_html: str = "",
        children: Optional[List["FakeElement"]] = None,
        data_testid: Optional[str] = None,
    ):
        self.tag = tag
        self.attributes = attributes or {}
        self._text_content = text_content
        self._inner_html = inner_html
        self.children = children or []
        self._visible = True
        self._click_callback: Optional[Callable] = None
        
        if data_testid:
            self.attributes["data-testid"] = data_testid

    def get_attribute(self, name: str) -> Optional[str]:
        """Get an attribute value."""
        return self.attributes.get(name)

    def text_content(self) -> str:
        """Get the text content of the element."""
        return self._text_content

    def inner_text(self) -> str:
        """Get the inner text of the element."""
        return self._text_content

    def inner_html(self) -> str:
        """Get the inner HTML of the element."""
        return self._inner_html or self._text_content

    def is_visible(self) -> bool:
        """Check if element is visible."""
        return self._visible

    def click(self):
        """Simulate clicking the element."""
        if self._click_callback:
            self._click_callback()

    def query_selector(self, selector: str) -> Optional["FakeElement"]:
        """Find first matching child element."""
        results = self.query_selector_all(selector)
        return results[0] if results else None

    def query_selector_all(self, selector: str) -> List["FakeElement"]:
        """Find all matching child elements."""
        results = []
        
        # Parse selector (simplified)
        for child in self.children:
            if self._matches_selector(child, selector):
                results.append(child)
            # Recursively search children
            results.extend(child.query_selector_all(selector))
        
        return results

    def _matches_selector(self, element: "FakeElement", selector: str) -> bool:
        """Check if an element matches a CSS selector (simplified)."""
        # Handle data-testid selector
        if 'data-testid=' in selector:
            match = re.search(r'data-testid=["\']?([^"\'}\]]+)', selector)
            if match:
                testid = match.group(1)
                return element.attributes.get("data-testid") == testid
        
        # Handle tag.class selector (e.g., "div.css-175oi2r")
        if '.' in selector and not selector.startswith('.'):
            parts = selector.split('.', 1)
            tag_part = parts[0]
            class_part = parts[1] if len(parts) > 1 else ""
            
            # Must match tag
            if element.tag != tag_part:
                return False
            
            # Must have the class
            if class_part:
                element_classes = element.attributes.get("class", "").split()
                return class_part in element_classes
            
            return True
        
        # Handle tag selector
        if selector in ['article', 'div', 'span', 'a', 'img', 'time', 'video', 'button']:
            return element.tag == selector
        
        # Handle class selector
        if selector.startswith('.'):
            class_name = selector[1:]
            element_classes = element.attributes.get("class", "").split()
            return class_name in element_classes
        
        # Handle attribute selectors like [href*="status"]
        attr_match = re.search(r'\[(\w+)\*?=?"?([^"\]]+)"?\]', selector)
        if attr_match:
            attr_name, attr_value = attr_match.groups()
            element_attr = element.attributes.get(attr_name, "")
            if '*=' in selector:
                return attr_value in element_attr
            return element_attr == attr_value
        
        # Handle :has-text() pseudo-selector
        if ':has-text(' in selector:
            match = re.search(r':has-text\(["\']?([^"\']+)["\']?\)', selector)
            if match:
                text = match.group(1)
                return text.lower() in element._text_content.lower()
        
        return False

    def evaluate(self, expression: str) -> Any:
        """Evaluate JavaScript expression (simplified simulation)."""
        if "getBoundingClientRect" in expression:
            return {"y": 100, "height": 200, "top": 100}
        if "scrollHeight" in expression:
            return 1000
        return None

    # Factory methods for common tweet elements
    @classmethod
    def tweet_article(
        cls,
        tweet_id: str,
        content: str = "Test tweet content",
        author: str = "testuser",
        timestamp: str = "2025-01-10T12:00:00Z",
        likes: int = 0,
        retweets: int = 0,
        replies: int = 0,
        has_media: bool = False,
        media_urls: Optional[List[str]] = None,
        has_thread_line: bool = False,
        is_retweet: bool = False,
        original_author: Optional[str] = None,
    ) -> "FakeElement":
        """Create a fake tweet article element."""
        children = []
        
        # Add tweet link
        tweet_link = cls(
            tag="a",
            attributes={
                "href": f"/{author}/status/{tweet_id}",
                "role": "link",
            },
            data_testid="User-Name",
        )
        children.append(tweet_link)
        
        # Add tweet text
        text_element = cls(
            tag="div",
            text_content=content,
            data_testid="tweetText",
        )
        children.append(text_element)
        
        # Add timestamp
        time_element = cls(
            tag="time",
            attributes={"datetime": timestamp},
            text_content=timestamp,
        )
        children.append(time_element)
        
        # Add engagement metrics
        if likes > 0:
            children.append(cls(
                tag="span",
                text_content=str(likes),
                attributes={"data-testid": "like"},
            ))
        
        # Add media if present
        if has_media and media_urls:
            for url in media_urls:
                children.append(cls(
                    tag="img",
                    attributes={"src": url},
                    data_testid="tweetPhoto",
                ))
        
        # Add thread line indicator
        if has_thread_line:
            children.append(cls(
                tag="div",
                attributes={"class": "r-1bimlpy"},  # Thread connector class
            ))
        
        # Add retweet indicator
        if is_retweet and original_author:
            children.append(cls(
                tag="span",
                text_content=f"{author} reposted",
                data_testid="socialContext",
            ))
        
        return cls(
            tag="article",
            data_testid="tweet",
            children=children,
            text_content=content,
        )

    @classmethod
    def retry_button(cls) -> "FakeElement":
        """Create a fake retry button."""
        return cls(
            tag="button",
            text_content="Retry",
            attributes={"type": "button"},
        )


class FakeLocator:
    """Fake Playwright Locator for testing."""
    
    def __init__(self, elements: List[FakeElement] = None, visible: bool = True):
        self.elements = elements or []
        self._visible = visible
        self._first_element = elements[0] if elements else None

    @property
    def first(self) -> "FakeLocator":
        """Get first matching element."""
        return FakeLocator(
            elements=[self.elements[0]] if self.elements else [],
            visible=self._visible
        )

    def is_visible(self, timeout: int = 1000) -> bool:
        """Check if any element is visible."""
        return self._visible and len(self.elements) > 0

    def click(self):
        """Click the first element."""
        if self.elements:
            self.elements[0].click()

    def count(self) -> int:
        """Return number of matching elements."""
        return len(self.elements)

    def all(self) -> List[FakeElement]:
        """Return all matching elements."""
        return self.elements


class FakePage:
    """
    Fake Playwright Page for testing.
    
    Simulates a browser page with articles, navigation, and JavaScript evaluation.
    """
    
    def __init__(self, url: str = "https://x.com/testuser"):
        self._url = url
        self._articles: List[FakeElement] = []
        self._elements: Dict[str, List[FakeElement]] = {}
        self.evaluations: List[str] = []
        self._reloaded = False
        self._goto_calls: List[str] = []
        self._click_count = 0
        self._wait_callbacks: Dict[str, Callable] = {}

    def url(self) -> str:
        """Get current URL."""
        return self._url

    def goto(self, url: str, **kwargs) -> None:
        """Navigate to URL."""
        self._url = url
        self._goto_calls.append(url)

    def reload(self, **kwargs) -> None:
        """Reload the page."""
        self._reloaded = True

    def add_article(self, article: FakeElement) -> None:
        """Add a tweet article to the page."""
        self._articles.append(article)

    def add_element(self, selector: str, element: FakeElement) -> None:
        """Add an element that matches a specific selector."""
        if selector not in self._elements:
            self._elements[selector] = []
        self._elements[selector].append(element)

    def query_selector(self, selector: str) -> Optional[FakeElement]:
        """Find first element matching selector."""
        results = self.query_selector_all(selector)
        return results[0] if results else None

    def query_selector_all(self, selector: str) -> List[FakeElement]:
        """Find all elements matching selector."""
        # Check pre-registered elements first
        if selector in self._elements:
            return self._elements[selector]
        
        # Check for article queries
        if 'article' in selector or 'tweet' in selector:
            return self._articles
        
        # Search within articles
        results = []
        for article in self._articles:
            results.extend(article.query_selector_all(selector))
        
        return results

    def locator(self, selector: str) -> FakeLocator:
        """Get a locator for elements matching selector."""
        elements = self.query_selector_all(selector)
        return FakeLocator(elements=elements)

    def evaluate(self, expression: str) -> Any:
        """Evaluate JavaScript expression."""
        self.evaluations.append(expression)
        
        if "scrollHeight" in expression:
            return 5000 + len(self._articles) * 200
        if "scrollBy" in expression:
            return None
        if "scrollTo" in expression:
            return None
        if "localStorage" in expression:
            return None
        if "document.body.scrollHeight" in expression:
            return 5000
        if "querySelectorAll" in expression:
            return len(self._articles)
        
        return None

    def wait_for_selector(self, selector: str, timeout: int = 30000) -> FakeElement:
        """Wait for an element to appear."""
        elements = self.query_selector_all(selector)
        if elements:
            return elements[0]
        from playwright.sync_api import TimeoutError
        raise TimeoutError(f"Timeout waiting for selector: {selector}")

    def wait_for_timeout(self, timeout: int) -> None:
        """Wait for a specified time (simulated)."""
        pass  # No-op in tests

    def wait_for_url(self, url: str, timeout: int = 30000) -> None:
        """Wait for URL to match."""
        if url not in self._url and url != self._url:
            from playwright.sync_api import TimeoutError
            raise TimeoutError(f"Timeout waiting for URL: {url}")

    def wait_for_function(self, expression: str, timeout: int = 5000) -> bool:
        """Wait for a JavaScript function to return truthy."""
        return True

    def route(self, pattern: str, handler: Callable) -> None:
        """Register a route handler (no-op in tests)."""
        pass

    def keyboard(self) -> "FakeKeyboard":
        """Get keyboard interface."""
        return FakeKeyboard()

    def mouse(self) -> "FakeMouse":
        """Get mouse interface."""
        return FakeMouse()


class FakeKeyboard:
    """Fake keyboard for testing."""
    
    def press(self, key: str) -> None:
        """Press a key."""
        pass

    def type(self, text: str) -> None:
        """Type text."""
        pass


class FakeMouse:
    """Fake mouse for testing."""
    
    def click(self, x: int, y: int) -> None:
        """Click at coordinates."""
        pass

    def wheel(self, delta_x: int, delta_y: int) -> None:
        """Scroll the mouse wheel."""
        pass


class FakeContext:
    """Fake Playwright BrowserContext for testing."""
    
    def __init__(self):
        self.pages: List[FakePage] = []
        self._storage_state: Optional[str] = None

    def new_page(self) -> FakePage:
        """Create a new page."""
        page = FakePage()
        self.pages.append(page)
        return page

    def close(self) -> None:
        """Close the context."""
        self.pages.clear()

    def storage_state(self, path: str = None) -> Dict:
        """Get or save storage state."""
        if path:
            self._storage_state = path
        return {"cookies": []}

    def evaluate(self, expression: str) -> Any:
        """Evaluate JavaScript in context."""
        return None


class FakeBrowser:
    """Fake Playwright Browser for testing."""
    
    def __init__(self):
        self.contexts: List[FakeContext] = []
        self._closed = False

    def new_context(self, **kwargs) -> FakeContext:
        """Create a new browser context."""
        context = FakeContext()
        self.contexts.append(context)
        return context

    def close(self) -> None:
        """Close the browser."""
        self._closed = True
        for context in self.contexts:
            context.close()
        self.contexts.clear()


# HTML Fixtures for testing parsers with real-like HTML
class HTMLFixtures:
    """
    Pre-built HTML snippets for testing parsing functions.
    
    Use these to test parsers without needing real browser output.
    """
    
    @staticmethod
    def tweet_html(
        tweet_id: str,
        author: str,
        content: str,
        timestamp: str = "2025-01-10T12:00:00.000Z",
        likes: int = 42,
        retweets: int = 10,
        replies: int = 5,
    ) -> str:
        """Generate realistic tweet HTML."""
        return f'''
        <article data-testid="tweet" role="article">
            <div class="css-175oi2r">
                <a href="/{author}/status/{tweet_id}" role="link">
                    <time datetime="{timestamp}">{timestamp}</time>
                </a>
                <div data-testid="User-Name">
                    <a href="/{author}">@{author}</a>
                </div>
                <div data-testid="tweetText" class="css-1jxf684">
                    <span>{content}</span>
                </div>
                <div class="css-175oi2r r-1awozwy">
                    <span data-testid="reply" aria-label="{replies} replies">{replies}</span>
                    <span data-testid="retweet" aria-label="{retweets} reposts">{retweets}</span>
                    <span data-testid="like" aria-label="{likes} likes">{likes}</span>
                </div>
            </div>
        </article>
        '''

    @staticmethod
    def retweet_html(
        tweet_id: str,
        retweeter: str,
        original_author: str,
        content: str,
    ) -> str:
        """Generate retweet HTML."""
        return f'''
        <article data-testid="tweet" role="article">
            <div data-testid="socialContext">
                <span>{retweeter} reposted</span>
            </div>
            <a href="/{original_author}/status/{tweet_id}" role="link">
                <time datetime="2025-01-10T12:00:00.000Z"></time>
            </a>
            <div data-testid="User-Name">
                <a href="/{original_author}">@{original_author}</a>
            </div>
            <div data-testid="tweetText">
                <span>{content}</span>
            </div>
        </article>
        '''

    @staticmethod
    def tweet_with_media_html(
        tweet_id: str,
        author: str,
        content: str,
        image_urls: List[str],
    ) -> str:
        """Generate tweet with images HTML."""
        images_html = "\n".join([
            f'<img src="{url}" data-testid="tweetPhoto" />'
            for url in image_urls
        ])
        return f'''
        <article data-testid="tweet" role="article">
            <a href="/{author}/status/{tweet_id}" role="link">
                <time datetime="2025-01-10T12:00:00.000Z"></time>
            </a>
            <div data-testid="tweetText">
                <span>{content}</span>
            </div>
            <div data-testid="tweetPhoto">
                {images_html}
            </div>
        </article>
        '''

    @staticmethod
    def thread_indicator_html() -> str:
        """Generate HTML with thread connector line."""
        return '''
        <div class="css-175oi2r r-1bimlpy" style="border-color: rgb(207, 217, 222);">
        </div>
        '''

    @staticmethod
    def cookie_banner_html() -> str:
        """Generate cookie consent banner HTML."""
        return '''
        <div role="dialog" aria-modal="true" data-testid="sheetDialog">
            <div role="group">
                <button type="button" role="button">
                    <span>Accept all cookies</span>
                </button>
                <button type="button" role="button">
                    <span>Refuse non-essential cookies</span>
                </button>
            </div>
        </div>
        '''
