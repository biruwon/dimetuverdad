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

---

# Performance & Speed Optimizations

This section focuses specifically on improvements that reduce fetching time.

**Current baseline**: ~5-7 minutes for 200 posts (~1.5-2 sec/post)
**Target**: ~2-3 minutes for 200 posts (~0.6-0.9 sec/post)

---

## P1. Parallel HTML Extraction (High Impact, Low Effort)

### Current Problem
Each tweet extraction makes multiple round-trips to the browser (query selector, get attribute, inner text, etc.).

### Better Approach
Grab all article HTML in a single JavaScript call, then parse stateless:

```python
# Current (slow) - multiple round-trips per tweet
for article in articles:
    text = article.query_selector('[data-testid="tweetText"]').inner_text()  # Round-trip 1
    link = article.query_selector('a[href*="/status/"]').get_attribute('href')  # Round-trip 2
    time = article.query_selector('time').get_attribute('datetime')  # Round-trip 3
    # ... more round-trips

# Faster - single round-trip for all tweets
all_html = page.evaluate('''() => {
    return Array.from(document.querySelectorAll('article[data-testid="tweet"]'))
        .map(a => ({
            html: a.outerHTML,
            rect: a.getBoundingClientRect()
        }))
}''')

# Parse in Python (no browser communication)
from bs4 import BeautifulSoup
tweets = []
for item in all_html:
    soup = BeautifulSoup(item['html'], 'html.parser')
    tweets.append(extract_tweet_from_soup(soup))
```

### Expected Savings
- **Per scroll cycle**: 0.5-1 second (10-15 tweets)
- **Per 200 tweets**: 30-60 seconds

---

## P2. Adaptive Scroll Delays (Medium Impact, Low Effort)

### Current Problem
Fixed random delays between scrolls waste time when content loads quickly.

### Better Approach
Wait only until new content appears, with a short minimum delay:

```python
# Current (slow)
page.evaluate('window.scrollBy(0, 800)')
time.sleep(random.uniform(1.5, 3.0))  # Always wait 1.5-3 seconds

# Faster - wait only for content
prev_count = len(page.query_selector_all('article[data-testid="tweet"]'))
page.evaluate('window.scrollBy(0, 800)')

try:
    page.wait_for_function(
        f'document.querySelectorAll("article[data-testid=\\"tweet\\"]").length > {prev_count}',
        timeout=3000
    )
except TimeoutError:
    pass  # Content didn't load, continue anyway

# Minimum human-like delay to avoid rate limiting
time.sleep(0.3)
```

### Expected Savings
- **Per scroll cycle**: 1-2 seconds
- **Per 200 tweets**: 60-120 seconds

---

## P3. Batch Database Writes (Medium Impact, Low Effort)

### Current Problem
Each tweet is written individually, creating transaction overhead.

### Better Approach
Buffer tweets and write in batches:

```python
# Current (slow)
for tweet in tweets:
    save_tweet(conn, tweet)  # Individual transaction per tweet
    conn.commit()

# Faster - batch writes
class TweetBuffer:
    def __init__(self, conn, batch_size=50):
        self.conn = conn
        self.batch_size = batch_size
        self.buffer = []
    
    def add(self, tweet):
        self.buffer.append(tweet)
        if len(self.buffer) >= self.batch_size:
            self.flush()
    
    def flush(self):
        if not self.buffer:
            return
        cursor = self.conn.cursor()
        cursor.executemany('''
            INSERT OR REPLACE INTO tweets 
            (tweet_id, username, content, tweet_url, tweet_timestamp, ...)
            VALUES (?, ?, ?, ?, ?, ...)
        ''', [(t['tweet_id'], t['username'], ...) for t in self.buffer])
        self.conn.commit()
        self.buffer = []
```

### Expected Savings
- **Per 200 tweets**: 10-20 seconds

---

## P4. Skip Video Hover for Thumbnail Extraction (Medium Impact, Medium Effort)

### Current Problem
We hover over every video element to trigger network requests, adding 8-10 seconds per video.

### Better Approach
Extract video info from thumbnail URL when possible:

```python
# Current (slow) - hover to capture network requests
video_player = article.query_selector('[data-testid="videoPlayer"]')
if video_player:
    video_player.hover()
    time.sleep(2)  # Wait for network request
    # Then capture from network monitor

# Faster - derive from thumbnail when possible
def extract_video_url_fast(article_html: str) -> Optional[str]:
    soup = BeautifulSoup(article_html, 'html.parser')
    
    # Video thumbnail contains video ID
    poster = soup.select_one('[data-testid="videoPlayer"] video')
    if poster:
        poster_url = poster.get('poster', '')
        # Pattern: amplify_video_thumb/{VIDEO_ID}/img/...
        match = re.search(r'amplify_video_thumb/(\d+)/', poster_url)
        if match:
            video_id = match.group(1)
            # Construct playable URL (common pattern)
            return f"https://video.twimg.com/amplify_video/{video_id}/vid/avc1/720x720/video.mp4"
    
    # Fallback to hover method only if needed
    return None
```

### Expected Savings
- **Per video**: 8-10 seconds (when pattern matches)
- **Per 200 tweets**: 30-60 seconds (depends on video count)

---

## P5. Skip "Show More" for Short Tweets (Low-Medium Impact, Low Effort)

### Current Problem
We check for "Show more" button on every tweet, even short ones.

### Better Approach
Only look for expand button if visible text is near truncation threshold:

```python
# Current (slow) - check every tweet
show_more = article.query_selector('[data-testid="tweet-text-show-more-link"]')
if show_more:
    show_more.click()
    time.sleep(0.5)

# Faster - only check if likely truncated
def extract_text_smart(article_html: str, page=None, article=None) -> str:
    soup = BeautifulSoup(article_html, 'html.parser')
    text_el = soup.select_one('[data-testid="tweetText"]')
    
    if not text_el:
        return ""
    
    visible_text = text_el.get_text()
    
    # Twitter truncates at ~280 chars, check if near limit
    if len(visible_text) < 250:
        return visible_text  # Definitely not truncated
    
    # Only if near truncation AND we have page access, try expanding
    if page and article:
        show_more = article.query_selector('[data-testid="tweet-text-show-more-link"]')
        if show_more:
            show_more.click()
            page.wait_for_timeout(300)
            # Re-extract
            return article.query_selector('[data-testid="tweetText"]').inner_text()
    
    return visible_text
```

### Expected Savings
- **Per non-truncated tweet**: ~0.3-0.5 seconds
- **Per 200 tweets**: 20-40 seconds (assuming 70% not truncated)

---

## P6. Pre-scroll Content Loading (Medium Impact, Low Effort)

### Current Problem
We wait for content to load after each scroll, sequentially.

### Better Approach
Trigger loading of next batch while processing current batch:

```python
# Current (sequential)
tweets = extract_visible_tweets(page)
process_tweets(tweets)
scroll_down(page)
wait_for_load(page)  # Blocking wait

# Faster - overlap loading with processing
def collect_with_prefetch(page, max_tweets):
    collected = []
    
    while len(collected) < max_tweets:
        # Extract current visible tweets
        tweets = extract_visible_tweets(page)
        
        # Trigger next load BEFORE processing
        page.evaluate('window.scrollBy(0, 2000)')  # Scroll far ahead
        
        # Process current batch (network loading happens in parallel)
        for tweet in tweets:
            if tweet['id'] not in seen:
                collected.append(tweet)
                seen.add(tweet['id'])
        
        # Small wait for triggered content
        page.wait_for_timeout(300)
        
        # Scroll back to process newly loaded
        page.evaluate('window.scrollBy(0, -1200)')
```

### Expected Savings
- **Per scroll cycle**: 0.5-1 second
- **Per 200 tweets**: 20-40 seconds

---

## P7. Headless Mode for Production (Low Impact, Trivial Effort)

### Current Problem
Running with visible browser adds rendering overhead.

### Better Approach
Use headless mode when not debugging:

```python
# In config.py
class FetcherConfig:
    headless: bool = True  # Default to headless in production
    
# Override for debugging
# ./run_in_venv.sh fetch --user kikollan --visible
```

### Expected Savings
- **Overall**: ~10-20% faster (30-60 seconds per 200 tweets)

---

## P8. No Navigation for Quoted Tweets (High Impact, Medium Effort)

### Current Problem
We open a new tab for each quoted tweet, adding 5-6 seconds each.

### Better Approach
Extract everything from the embedded quote card without navigation:

```python
def extract_quoted_from_card(article_html: str) -> Optional[Dict]:
    """Extract quoted tweet data from the embedded card."""
    soup = BeautifulSoup(article_html, 'html.parser')
    
    # Find the quote card
    card = soup.select_one('[data-testid="card.wrapper"]')
    if not card:
        return None
    
    # Extract link to get tweet ID and username
    link = card.select_one('a[href*="/status/"]')
    if not link:
        return None
    
    href = link.get('href', '')
    match = re.match(r'/([^/]+)/status/(\d+)', href)
    if not match:
        return None
    
    username, tweet_id = match.groups()
    
    # Extract visible content from card
    text_el = card.select_one('[data-testid="tweetText"]')
    text = text_el.get_text() if text_el else ""
    
    # Extract media thumbnails
    media = []
    for img in card.select('img[src*="twimg.com"]'):
        media.append(img.get('src'))
    
    return {
        'tweet_id': tweet_id,
        'username': username,
        'content': text,
        'media_urls': media,
        'tweet_url': f'https://x.com/{username}/status/{tweet_id}'
    }
```

### Expected Savings
- **Per quoted tweet**: 5-6 seconds
- **Per 200 tweets**: 15-30 seconds (depends on quote frequency)

---

## Performance Improvement Summary

| Optimization | Time Saved (200 tweets) | Effort | Priority |
|--------------|-------------------------|--------|----------|
| P1. Parallel HTML Extraction | 30-60 sec | Low | 🔴 High |
| P2. Adaptive Scroll Delays | 60-120 sec | Low | 🔴 High |
| P3. Batch Database Writes | 10-20 sec | Low | 🟡 Medium |
| P4. Skip Video Hover | 30-60 sec | Medium | 🟡 Medium |
| P5. Skip "Show More" Check | 20-40 sec | Low | 🟢 Low |
| P6. Pre-scroll Loading | 20-40 sec | Low | 🟡 Medium |
| P7. Headless Mode | 30-60 sec | Trivial | 🟢 Low |
| P8. No Navigation for Quoted | 15-30 sec | Medium | 🟡 Medium |

**Total potential savings**: 215-430 seconds per 200 tweets (50-70% reduction)

### Projected Times for Large Fetches

| Tweets | Current Time | Optimized Time |
|--------|--------------|----------------|
| 200 | 5-7 min | 2-3 min |
| 1,000 | 25-35 min | 10-15 min |
| 10,000 | 4-6 hours | 1.5-2.5 hours |
| 85,000 | 35-50 hours | 13-20 hours |

---

## Quick Wins (Implement First)

1. **P2. Adaptive Scroll Delays** - Easiest, biggest impact
2. **P1. Parallel HTML Extraction** - Requires BeautifulSoup, high impact
3. **P3. Batch Database Writes** - Simple buffer class
4. **P7. Headless Mode** - Config flag only
