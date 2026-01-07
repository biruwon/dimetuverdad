# Fetcher Architecture Improvements

## Overview

This document outlines architectural improvements for the fetcher module based on lessons learned from debugging and maintaining the current implementation.

---

## 1. Separation of Concerns

### Current Problem
The code mixes navigation, extraction, scrolling, and recovery logic across multiple files with overlapping responsibilities.

### Better Approach
```
fetcher/
  browser/
    driver.py          # Browser lifecycle only (start, stop, session)
    navigator.py       # URL navigation, page state detection
  extraction/
    tweet_parser.py    # Pure DOM → data extraction (no side effects)
    media_extractor.py # Media URLs extraction
    quoted_tweet.py    # Quoted tweet handling
  scrolling/
    scroll_strategy.py # Abstract scroll behavior
    timeline_scroller.py
  persistence/
    session_store.py   # Cookie/session management
  pipeline.py          # Orchestrates everything
```

### Benefits
- Each module has single responsibility
- Easier to test in isolation
- Clear dependencies between components

---

## 2. Stateless Extraction Functions

### Current Problem
Extraction functions receive `page` object and can accidentally click/navigate, causing state corruption.

### Better Approach
Extract HTML first, then parse stateless:

```python
# Instead of:
def extract_tweet(page, article):  # Can accidentally navigate
    text = article.query_selector('[data-testid="tweetText"]').inner_text()
    ...
    
# Do:
def extract_tweet(html: str) -> TweetData:  # Pure function, no side effects
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.select_one('[data-testid="tweetText"]').get_text()
    ...
```

### Benefits
- No accidental navigation/clicks
- Easier to unit test with fixture HTML
- Can cache HTML for debugging
- Parallelizable

---

## 3. Explicit State Machine for Collection

### Current Problem
Complex nested loops with many flags (`consecutive_empty`, `post_page_cycles`, etc.) that are hard to reason about.

### Better Approach
State machine with clear transitions:

```python
from enum import Enum
from typing import Callable, Dict

class CollectorState(Enum):
    INITIALIZING = "initializing"
    SCROLLING = "scrolling"
    EXTRACTING = "extracting"
    RECOVERING = "recovering"
    RATE_LIMITED = "rate_limited"
    DONE = "done"
    ERROR = "error"

class Collector:
    def __init__(self):
        self.state = CollectorState.INITIALIZING
        self.transitions: Dict[CollectorState, Callable] = {
            CollectorState.INITIALIZING: self._initialize,
            CollectorState.SCROLLING: self._scroll,
            CollectorState.EXTRACTING: self._extract,
            CollectorState.RECOVERING: self._recover,
            CollectorState.RATE_LIMITED: self._wait_rate_limit,
        }
    
    def run(self):
        while self.state not in (CollectorState.DONE, CollectorState.ERROR):
            handler = self.transitions.get(self.state)
            self.state = handler()
```

### Benefits
- Clear visualization of possible states
- Transitions are explicit and testable
- Easy to add logging/metrics per state
- No hidden flag interactions

---

## 4. No Navigation for Quoted Tweets

### Current Problem
We had to build complex "new tab" logic to avoid breaking the timeline state.

### Better Approach
Never navigate away from the timeline. Extract everything from the embedded card:

```python
def extract_quoted_from_card(card_html: str) -> Optional[QuotedTweetData]:
    """
    The quoted tweet card already contains:
    - Author username
    - Tweet text preview
    - Media thumbnails
    - Tweet ID (from href)
    
    Only open new tab for full media that's not in the card.
    """
    soup = BeautifulSoup(card_html, 'html.parser')
    
    # Extract from card - no navigation needed
    username = soup.select_one('[data-testid="User-Name"]')
    text = soup.select_one('[data-testid="tweetText"]')
    link = soup.select_one('a[href*="/status/"]')
    
    return QuotedTweetData(
        username=extract_username(username),
        text=text.get_text() if text else None,
        tweet_id=extract_id_from_href(link['href']) if link else None,
    )
```

### Benefits
- No tab management complexity
- No risk of losing scroll position
- Faster extraction
- More reliable

---

## 5. Retry/Recovery as Decorators

### Current Problem
Recovery logic is scattered in multiple places (collector, fetch_tweets, scroller).

### Better Approach
Single retry decorator:

```python
from functools import wraps
from typing import List, Callable

def with_recovery(
    max_attempts: int = 3,
    strategies: List[Callable] = None
):
    def decorator(func):
        @wraps(func)
        def wrapper(page, *args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(page, *args, **kwargs)
                except RecoverableError as e:
                    if attempt < max_attempts - 1:
                        strategy = strategies[attempt % len(strategies)]
                        strategy(page)
                    else:
                        raise
        return wrapper
    return decorator

# Usage
@with_recovery(max_attempts=3, strategies=[click_retry, small_scroll, reload])
def scroll_and_extract(page):
    ...
```

### Benefits
- Single place to modify recovery logic
- Consistent behavior across all operations
- Easy to test strategies in isolation

---

## 6. Event-Driven Architecture

### Current Problem
Synchronous loop that blocks on each operation, hard to add cross-cutting concerns.

### Better Approach
Event-based with observers:

```python
from typing import Callable, List

class Event:
    def __init__(self):
        self._handlers: List[Callable] = []
    
    def __iadd__(self, handler: Callable):
        self._handlers.append(handler)
        return self
    
    def emit(self, *args, **kwargs):
        for handler in self._handlers:
            handler(*args, **kwargs)

class TweetCollector:
    def __init__(self):
        self.on_tweet_found = Event()
        self.on_scroll = Event()
        self.on_error = Event()
        self.on_recovery = Event()
    
    def collect(self, page, username):
        for tweet in self._scroll_and_extract(page):
            self.on_tweet_found.emit(tweet)

# Separate concerns via observers
collector = TweetCollector()
collector.on_tweet_found += save_to_db
collector.on_tweet_found += log_progress
collector.on_tweet_found += update_metrics
collector.on_error += recovery_handler
collector.on_error += alert_on_repeated_errors
```

### Benefits
- Decoupled components
- Easy to add/remove behaviors
- Better for async operations
- Natural place for metrics/logging

---

## 7. Configuration over Code

### Current Problem
Selectors and timeouts are hardcoded throughout the codebase.

### Better Approach
Single config file:

```yaml
# fetcher_config.yaml
selectors:
  tweet_text: '[data-testid="tweetText"]'
  article: 'article[data-testid="tweet"]'
  retry_button: 'button:has-text("Retry")'
  show_more: '[data-testid="tweet-text-show-more-link"]'
  card_wrapper: '[data-testid="card.wrapper"]'

timeouts:
  page_load: 30000
  element_wait: 5000
  login_manual: 120000

scroll:
  min_amount: 300
  max_amount: 600
  delay_min: 1.0
  delay_max: 3.0

recovery:
  max_empty_cycles: 10
  max_recovery_attempts: 3
  strategies:
    - click_retry
    - small_scroll
    - reload

rate_limiting:
  max_tweets_per_session: 800
  cooldown_seconds: 60
```

```python
# Load config
import yaml
from dataclasses import dataclass

@dataclass
class FetcherConfig:
    selectors: dict
    timeouts: dict
    scroll: dict
    recovery: dict
    rate_limiting: dict
    
    @classmethod
    def load(cls, path: str = "fetcher_config.yaml"):
        with open(path) as f:
            return cls(**yaml.safe_load(f))

config = FetcherConfig.load()
page.wait_for_selector(config.selectors['tweet_text'], timeout=config.timeouts['element_wait'])
```

### Benefits
- Easy to tune without code changes
- Environment-specific configs (dev/prod)
- Selectors can be updated when Twitter changes
- Self-documenting

---

## 8. Better Testing Strategy

### Current Problem
Tests mock too much internal state, making them brittle and not representative.

### Better Approach

**Unit tests**: Pure functions only (HTML → data)
```python
def test_extract_tweet_from_html():
    html = load_fixture("tweet_with_media.html")
    result = extract_tweet(html)
    
    assert result.text == "Expected tweet text"
    assert len(result.media) == 2
```

**Integration tests**: Record/replay with Playwright traces
```bash
# Record a trace once
playwright codegen --save-trace=twitter_scroll.trace https://x.com/kikollan
```

```python
def test_extraction_with_recorded_trace():
    html = load_trace_html("twitter_scroll.trace", step=5)
    result = extract_tweet(html)
    assert result.text == "expected..."
```

**Contract tests**: Verify selector assumptions
```python
@pytest.mark.live
def test_selectors_still_work():
    """Run periodically to detect Twitter UI changes."""
    page = create_live_page()
    page.goto("https://x.com/someone")
    
    # Verify our selectors still find elements
    assert page.query_selector(config.selectors['article'])
    assert page.query_selector(config.selectors['tweet_text'])
```

### Benefits
- Unit tests are fast and reliable
- Integration tests catch real issues
- Contract tests alert on Twitter changes
- Less mocking, more confidence

---

## 9. Observability First

### Current Problem
Debugging requires adding print statements; no metrics collection.

### Better Approach
Structured logging and metrics from day one:

```python
import structlog
from dataclasses import dataclass, field
from time import time

logger = structlog.get_logger()

@dataclass
class CollectionMetrics:
    tweets_found: int = 0
    tweets_saved: int = 0
    scroll_cycles: int = 0
    recoveries: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time)
    
    @property
    def elapsed(self) -> float:
        return time() - self.start_time
    
    @property
    def tweets_per_minute(self) -> float:
        return (self.tweets_found / self.elapsed) * 60 if self.elapsed > 0 else 0

# Structured logging
logger.info(
    "tweet_extracted",
    tweet_id=tweet.id,
    username=tweet.username,
    media_count=len(tweet.media),
    has_quoted=tweet.quoted is not None,
)

# Metrics export
def export_metrics(metrics: CollectionMetrics):
    """Export to Prometheus, CloudWatch, or file."""
    print(f"""
    === Collection Metrics ===
    Tweets found: {metrics.tweets_found}
    Tweets saved: {metrics.tweets_saved}
    Scroll cycles: {metrics.scroll_cycles}
    Recoveries: {metrics.recoveries}
    Errors: {metrics.errors}
    Duration: {metrics.elapsed:.1f}s
    Rate: {metrics.tweets_per_minute:.1f} tweets/min
    """)
```

### Benefits
- Easy to debug issues
- Can detect performance regressions
- Alerting on anomalies
- Historical analysis

---

## 10. Graceful Degradation

### Current Problem
If quoted tweet extraction fails, we might lose the entire tweet or retry excessively.

### Better Approach
Always save what we have:

```python
@dataclass
class TweetData:
    id: str
    username: str
    text: str
    timestamp: str
    media: List[str] = field(default_factory=list)
    quoted: Optional['TweetData'] = None
    extraction_errors: List[str] = field(default_factory=list)

def extract_tweet_gracefully(article_html: str) -> TweetData:
    """Extract tweet, capturing partial data even on errors."""
    tweet = TweetData(
        id=extract_id(article_html),
        username=extract_username(article_html),
        text="",
        timestamp="",
    )
    
    # Each extraction can fail independently
    try:
        tweet.text = extract_text(article_html)
    except ExtractionError as e:
        tweet.extraction_errors.append(f"text: {e}")
    
    try:
        tweet.timestamp = extract_timestamp(article_html)
    except ExtractionError as e:
        tweet.extraction_errors.append(f"timestamp: {e}")
    
    try:
        tweet.media = extract_media(article_html)
    except ExtractionError as e:
        tweet.extraction_errors.append(f"media: {e}")
    
    try:
        tweet.quoted = extract_quoted(article_html)
    except ExtractionError as e:
        tweet.extraction_errors.append(f"quoted: {e}")
    
    # Always save - partial data is better than no data
    return tweet
```

### Benefits
- Never lose data due to minor failures
- Can analyze extraction_errors to fix issues
- More resilient to Twitter changes
- Can retry failed parts later

---

## Implementation Priority

| Priority | Improvement | Effort | Impact |
|----------|-------------|--------|--------|
| 1 | Configuration over Code | Low | High |
| 2 | Stateless Extraction | Medium | High |
| 3 | Graceful Degradation | Low | Medium |
| 4 | Observability | Medium | High |
| 5 | State Machine | High | High |
| 6 | Event-Driven | High | Medium |
| 7 | Better Testing | Medium | Medium |
| 8 | Retry Decorators | Low | Medium |
| 9 | Separation of Concerns | High | High |
| 10 | No Navigation for Quoted | Medium | Medium |

---

## Summary

The main architectural changes would be:

1. **Separate navigation from extraction** - No accidental clicks/navigation
2. **State machine** - Replace nested loops with explicit states
3. **Configuration-driven** - Selectors and timeouts in config files
4. **Event-driven** - Better separation of concerns
5. **Structured logging** - Observability from day one
6. **Graceful degradation** - Always save partial data

These changes would make the fetcher more maintainable, testable, and resilient to Twitter's frequent UI changes.
