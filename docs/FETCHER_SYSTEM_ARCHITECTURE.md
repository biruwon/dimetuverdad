# Fetcher System Architecture

## Overview

The **dimetuverdad fetcher system** is a sophisticated web scraping pipeline designed to collect tweets from Twitter/X with high reliability and anti-detection measures. It combines browser automation, intelligent content extraction, and robust error handling to gather comprehensive tweet data while respecting platform limits and avoiding detection.

## 🏗️ Core Architecture

### Component Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                     Fetcher (Main Orchestrator)              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Components:                                            │  │
│  │ • TweetCollector (core collection logic)              │  │
│  │ • SessionManager (browser session management)         │  │
│  │ • Scroller (scrolling and navigation)                 │  │
│  │ • MediaMonitor (network request monitoring)           │  │
│  │ • ResumeManager (resume positioning)                  │  │
│  │ • RefetchManager (individual tweet refetching)        │  │
│  │ • Parsers (content extraction)                        │  │
│  │ • Database Layer (data persistence)                   │  │
│  │ • Config (configuration management)                   │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Configuration System

**FetcherConfig** (`config.py`)
- Centralized configuration management using Python dataclass
- Environment variable integration with `.env` file support
- Anti-detection settings (user agents, viewport sizes, delays)
- Collection limits and timeouts
- Session management parameters

**Key Configuration Areas:**
```python
@dataclass
class FetcherConfig:
    # Authentication (from environment variables)
    username: str
    password: str
    email_or_phone: str
    
    # Browser settings (headless=True default for performance)
    headless: bool = True  # Override with --visible flag
    user_agents: List[str] = None  # Randomized for anti-detection
    
    # Feature flags
    collect_threads: bool = False  # Enable with FETCHER_COLLECT_THREADS=true
    use_adaptive_scroll: bool = True  # P2 optimization
    
    # Timeouts and delays
    page_load_timeout: int = 30000
    min_human_delay: float = 1.0
    max_human_delay: float = 3.0
    
    # Collection limits
    max_consecutive_empty_scrolls: int = 15
    max_consecutive_existing_tweets: int = 10
    large_collection_threshold: int = 800
```

---

## 📊 Collection Strategies

### 1. **Latest Mode Strategy** (`fetch_latest_tweets()`)
**Purpose:** Fetch only the most recent tweets, optimized for frequent updates

```
Input: Username, max_tweets (default: 30)
  ↓
Navigate to Profile Page (/username)
  ↓
Extract Profile Picture
  ↓
Scroll and Collect Loop:
  • Extract visible tweets
  • Check database existence (for stopping logic)
  • Skip existing tweets (don't count toward consecutive)
  • Save new tweets only
  • Stop after 10 consecutive existing tweets in database
  ↓
Update Profile Information
  ↓
Output: New tweets saved to database
```

**Key Features:**
- **Smart Stopping:** Stops after `max_consecutive_existing_tweets` (default: 10) consecutive scrolls with no new tweets
- **Database Integration:** Checks tweet existence before processing to avoid duplicates
- **Profile Updates:** Updates user profile picture and metadata on each run
- **Fast Updates:** Optimized for frequent checking of account activity

### 2. **Full History Strategy** (`fetch_tweets()`)
**Purpose:** Comprehensive collection of user tweet history with intelligent resume

```
Input: Username, max_tweets, resume_from_last (default: True)
  ↓
Check Existing Tweet Range in Database
  ↓
PHASE 1: Fetch New Tweets (newer than newest existing)
  ↓
PHASE 2: Resume from Oldest (if needed and resume_from_last=True)
  • Use ResumeManager for timestamp-based navigation
  • Search Twitter for tweets before oldest timestamp
  • Continue scrolling from resume position
  ↓
Output: Complete tweet history with resume capability
```

**Key Features:**
- **Two-Phase Collection:** New tweets first, then historical
- **Resume Logic:** Continues from last collected timestamp using search-based navigation
- **Timestamp Tracking:** Maintains oldest/newest timestamps for efficient resuming
- **Automatic Fallback:** Falls back to profile start if resume fails

### 3. **Multi-Session Strategy** (`fetch_tweets_in_sessions()`)
**Purpose:** Overcome Twitter's content serving limits for large collections

```
Input: Username, max_tweets (>800), session_size (default: 800)
  ↓
Calculate Session Size (800 tweets per session)
  ↓
For Each Session:
  • Navigate to profile page
  • Collect up to session_size tweets
  • Refresh browser session to reset limits
  • Continue until target reached
  ↓
Output: Large tweet collection across multiple sessions
```

**Key Features:**
- **Session Rotation:** Refreshes browser context between sessions to bypass rate limits
- **Progress Tracking:** Maintains collection progress across sessions
- **Duplicate Prevention:** Filters out duplicates between sessions
- **Automatic Trigger:** Activates automatically when `max_tweets > large_collection_threshold`

---

## 🔍 Core Components

### TweetCollector (`collector.py`)

**Primary Responsibilities:**
- Orchestrate tweet collection from web pages
- Process individual tweet articles
- Manage collection state and progress
- Handle error recovery and retries

**Key Methods:**

#### `collect_tweets_from_page()`
```python
def collect_tweets_from_page(
    self, page, username: str, max_tweets: int,
    resume_from_last: bool, oldest_timestamp: Optional[str],
    profile_pic_url: Optional[str], conn
) -> List[Dict]:
```

**Collection Flow:**
```
Setup Media Monitoring
  ↓
Main Collection Loop:
  • Find tweet articles on page
  • Process each article
  • Extract tweet data
  • Check database existence
  • Save/update tweet
  • Track progress and empty scrolls
  ↓
Handle Scrolling and Recovery
  ↓
Return Collected Tweets
```

#### `extract_tweet_data()`
**Comprehensive Data Extraction:**
- Post type analysis (original, repost, reply, etc.)
- Content extraction with full text reconstruction
- Media data (images, videos, links)
- Engagement metrics (likes, retweets, replies, views)
- Content elements (hashtags, mentions, links)
- Timestamp extraction

### SessionManager (`session_manager.py`)

**Browser Session Management:**
- Playwright browser initialization
- Anti-detection configuration
- Session persistence and recovery
- Cookie and localStorage management

**Key Features:**
- **Stealth Mode:** Randomized user agents, viewport sizes
- **Session Persistence:** Saves and restores browser sessions
- **Context Isolation:** Separate browser contexts per session
- **Resource Management:** Proper cleanup and resource limits

### Scroller (`scroller.py`)

**Intelligent Scrolling System:**
- Human-like scrolling patterns
- Content loading detection
- Recovery strategies for stuck pages
- Performance optimization

**Scrolling Strategies:**
- **Event-based Scrolling:** Simulates user scroll events
- **Random Patterns:** Variable scroll distances and timing
- **Aggressive Scrolling:** For difficult-to-load content
- **Recovery Mechanisms:** Page refresh, cache clearing

### MediaMonitor (`media_monitor.py`)

**Advanced Network Request Monitoring:**
- Captures media URLs from network requests during page interactions
- Video and image URL extraction with content type filtering
- Intelligent deduplication and URL validation
- Integration with DOM-based media extraction

**Key Features:**
- **Selective Monitoring:** Only monitors when main tweet contains video elements
- **Video-First Approach:** Prioritizes video URL capture over images
- **URL Deduplication:** Prevents duplicate media URLs from DOM and network sources
- **Content Type Filtering:** Excludes thumbnails, avatars, and non-content media

**Monitoring Process:**
```
Check for Video Elements in Main Tweet
  ↓
Setup Request/Response Interception
  ↓
Trigger Video Loading (hover/play attempts)
  ↓
Capture Video URLs from Network Requests
  ↓
Filter and Validate URLs
  ↓
Deduplicate with DOM-extracted Media
  ↓
Combine Results for Tweet Data
```

**URL Filtering Logic:**
```python
# Video detection criteria
has_video_extension = any(ext in url.lower() for ext in ['.mp4', '.m3u8', '.webm', '.mov'])
has_video_keyword = 'video.twimg.com' in url
is_not_thumbnail = not ('amplify_video_thumb' in url or 'thumb' in url)
is_not_image = not any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp'])

# Only capture if video AND not image/thumbnail
if (has_video_extension or has_video_keyword) and not is_image and not is_thumbnail:
    capture_video_url(url)
```

### Parsers (`parsers.py`)

**Advanced Content Extraction Engine:**
- HTML parsing and data extraction with emoji preservation
- Comprehensive post type classification (9+ types)
- Multi-strategy media extraction (DOM + network monitoring)
- Engagement metrics parsing with unit conversion
- Content element extraction (hashtags, mentions, links)

**Key Parsing Functions:**

#### `extract_full_tweet_content()`
**Multi-Strategy Text Extraction:**
- Expands truncated content ("Show more" button handling)
- Preserves emojis using JavaScript evaluation
- Fallback strategies: JavaScript → BeautifulSoup → Regex
- Handles different Twitter layouts and languages

#### `analyze_post_type()`
**Comprehensive Post Classification:**
- **Original:** Standalone content
- **Repost_Own:** Self-retweets (skipped)
- **Repost_Other:** Content shared from others
- **Repost_Reply:** Comments on retweets
- **Pinned:** Pinned tweets (skipped)
- **Thread:** Multi-part conversations

#### `extract_media_data()`
**Dual-Mode Media Extraction:**
- **DOM Extraction:** Parses visible media elements
- **Network Monitoring:** Captures dynamically loaded content
- **Deduplication:** Combines results without duplicates
- **Content Filtering:** Excludes avatars, previews, thumbnails

#### `extract_tweet_with_media_monitoring()`
**Complete Tweet Processing:**
```python
# Combined extraction workflow
tweet_data = extract_tweet_with_quoted_content(page, tweet_id, username, tweet_url)
if has_video_elements(tweet_data):
    video_urls = media_monitor.setup_and_monitor(page, scroller)
    tweet_data = media_monitor.process_video_urls(video_urls, tweet_data)
return tweet_data
```

#### `find_and_extract_quoted_tweet()`
**Multi-Strategy Quoted Content Detection:**
- Parser metadata analysis
- DOM structure inspection
- Click-and-navigate extraction
- Fallback URL parsing
- Complete media extraction from quoted tweets

### ResumeManager (`resume_manager.py`)

**Smart Resume Positioning:**
- Timestamp-based navigation
- Search functionality for old tweets
- Position recovery after interruptions

**Resume Strategies:**
- **Timestamp Search:** Uses Twitter search to find specific dates
- **Scroll Positioning:** Calculates scroll distance to target tweets
- **Fallback Mechanisms:** Graceful degradation when resume fails

### RefetchManager (`refetch_manager.py`)

**Individual Tweet Management:**
- Single tweet refetching
- Account data refresh
- Update existing tweet data

**Refetch Operations:**
- **Single Tweet:** Update specific tweet by ID
- **Account Refetch:** Delete and re-scrape entire account
- **Data Updates:** Refresh engagement metrics, media data

---

## 💾 Database Integration

### Data Persistence Layer

**Tweet Storage Schema:**
```sql
CREATE TABLE tweets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tweet_id TEXT UNIQUE NOT NULL,
    username TEXT NOT NULL,
    content TEXT,
    tweet_url TEXT,
    tweet_timestamp TEXT,
    post_type TEXT DEFAULT 'original',
    original_author TEXT,
    original_tweet_id TEXT,
    reply_to_username TEXT,
    media_links TEXT,
    media_count INTEGER DEFAULT 0,
    hashtags TEXT,
    mentions TEXT,
    external_links TEXT,
    engagement_likes INTEGER DEFAULT 0,
    engagement_retweets INTEGER DEFAULT 0,
    engagement_replies INTEGER DEFAULT 0,
    engagement_views INTEGER DEFAULT 0,
    original_content TEXT,
    is_pinned INTEGER DEFAULT 0,
    profile_pic_url TEXT
);
```

**Account Profile Storage:**
```sql
CREATE TABLE accounts (
    username TEXT PRIMARY KEY,
    profile_pic_url TEXT,
    profile_pic_updated TEXT,
    last_scraped TEXT,
    last_updated TEXT DEFAULT CURRENT_TIMESTAMP
);
```

**Error Logging:**
```sql
CREATE TABLE scrape_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    tweet_id TEXT,
    error TEXT,
    context TEXT,
    timestamp TEXT
);
```

### Database Operations

**Repository Pattern:**
- **TweetRepository:** Handles all tweet CRUD operations
- **Standardized Connections:** `get_db_connection_context()` for consistent access
- **Environment-Aware Paths:** Automatic database path resolution
- **Connection Pooling:** Context managers for proper resource management

**Data Validation & Updates:**
- **Duplicate Detection:** Tweet ID uniqueness constraints
- **Content Change Detection:** Compares existing vs new content for updates
- **Timestamp Validation:** ISO format validation and normalization
- **Media URL Verification:** Validates and deduplicates media links

**Update Logic:**
```python
# Check for updates needed
needs_update = (
    existing_post_type != new_post_type or
    existing_content != new_content or
    existing_original_author != new_original_author
)

# Perform targeted update
if needs_update:
    UPDATE tweets SET content=?, post_type=?, ... WHERE tweet_id=?
```

---

## 🔄 Collection Workflows

### Single User Collection

```python
# Latest tweets only (fast updates)
tweets = fetch_latest_tweets(page, username, max_tweets=30)

# Full history collection
tweets = fetch_tweets(page, username, max_tweets=1000, resume_from_last=True)

# Multi-session for large collections
tweets = fetch_tweets_in_sessions(page, username, max_tweets=5000, session_size=800)
```

### Multi-User Batch Processing

```python
# Process multiple accounts in single session
total_saved, accounts_processed = run_fetch_session(
    playwright, handles, max_tweets, resume_from_last_flag, latest_mode
)
```

### Error Recovery and Retries

**Retry Logic:**
- Exponential backoff (2^attempt + random jitter)
- Maximum 5 retry attempts per operation
- Error logging to database for analysis
- Graceful degradation on persistent failures

**Recovery Strategies:**
- Session refresh between collections
- Browser context recreation
- Cache clearing and page reloads
- Alternative scrolling methods

---

## 📊 Performance Characteristics

### Performance Optimizations (P1-P8)

The fetcher includes 8 performance optimizations that significantly improve collection speed:

| Optimization | Description | Time Saved |
|-------------|-------------|------------|
| **P1. Parallel HTML Extraction** | BeautifulSoup parsing off main thread | 30-60 sec |
| **P2. Adaptive Scroll Delays** | Wait for content instead of fixed 2s delays | 60-120 sec |
| **P3. Batch Database Writes** | Buffer tweets and write in batches | 10-20 sec |
| **P4. Skip Video Hover** | Extract video URL from poster pattern | 30-60 sec |
| **P5. Skip "Show More" Check** | Skip for tweets < 250 chars | 20-40 sec |
| **P6. Pre-scroll Content Loading** | Detect new content before scrolling | 20-40 sec |
| **P7. Headless Mode** | No browser rendering overhead | 30-60 sec |
| **P8. No Navigation for Quoted Tweets** | Extract quoted content inline | 15-30 sec |

### Collection Speeds (M1 Pro, 32GB RAM)

**Benchmark Results (200 tweets):**

| User | Time | Rate | Notes |
|------|------|------|-------|
| kikollan | 187 sec | ~0.94 sec/tweet | Clean run, text-heavy content |
| infovlogger36 | 306 sec | ~1.53 sec/tweet | Media-heavy with quoted tweets |

**Estimated Collection Times:**
- 200 tweets: ~3-5 minutes
- 1,000 tweets: ~15-25 minutes
- 5,000 tweets: ~1.5-2.5 hours
- 85,000 tweets: ~22-30 hours

**Latest Mode (fast updates):**
- 30 tweets: ~10-15 seconds
- 100 tweets: ~30-45 seconds
- Optimized stopping logic prevents unnecessary work

**Multi-Session Mode:**
- 2000 tweets: ~10-15 minutes (across 3 sessions)
- 5000 tweets: ~25-35 minutes (across 7 sessions)
- Automatic session management and refresh

### Memory & CPU Usage

**Resource Requirements:**
- **RAM:** 32GB+ recommended for large collections
- **CPU:** M1 Pro optimized (30-60s per LLM analysis when combined)
- **Storage:** SQLite database scales efficiently
- **Network:** Stable connection required for media downloads

### Optimization Strategies

**Core Strategies:**
1. **Smart Stopping:** Latest mode prevents over-collection of existing tweets
2. **Resume Logic:** Continues from last collected timestamp, not from beginning
3. **Session Management:** Refreshes browser context to bypass Twitter's limits
4. **Parallel Processing:** Multiple accounts per browser session
5. **Resource Limits:** Memory and CPU monitoring with automatic cleanup
6. **Database Indexing:** Optimized queries for existence checks and updates

**P1-P8 Performance Optimizations (Implemented):**
- **P1. Parallel HTML Extraction:** Uses BeautifulSoup for off-thread HTML parsing
- **P2. Adaptive Scroll Delays:** Waits for content to load instead of fixed 2-second delays (2000ms timeout)
- **P3. Batch Database Writes:** Buffers tweets and writes in batches to reduce I/O
- **P4. Skip Video Hover:** Extracts video URLs from poster pattern (`amplify_video_thumb` → `amplify_video/vid/avc1/720x720/video.mp4`)
- **P5. Skip "Show More" Check:** Skips expansion click for tweets under 250 characters
- **P6. Pre-scroll Content Loading:** Detects new content availability before scrolling
- **P7. Headless Mode:** Enabled by default (use `--visible` to disable for debugging)
- **P8. No Navigation for Quoted Tweets:** Extracts quoted content inline from DOM without opening new tabs

### Rate Limiting & Anti-Detection

**Advanced Anti-Detection Measures:**
- **User Agent Rotation:** Randomized from curated list of modern browsers
- **Viewport Variation:** Dynamic sizing to mimic different devices
- **Timing Patterns:** Human-like delays with randomization
- **Session Rotation:** Fresh browser contexts between collections
- **Request Throttling:** Progressive delays on detection patterns

**Rate Limiting Handling:**
- **Consecutive Empty Scroll Detection:** Automatic stopping when Twitter stops serving content
- **Session Refresh:** Browser context recreation to reset limits
- **Progressive Delays:** Exponential backoff for retry attempts
- **Error-Based Adaptation:** Adjusts behavior based on response patterns

---

## 🛠️ Configuration System

### FetcherConfig Dataclass

**Centralized Configuration Management:**
```python
@dataclass
class FetcherConfig:
    # Authentication (from environment variables)
    username: str = os.getenv("X_USERNAME", "")
    password: str = os.getenv("X_PASSWORD", "")
    email_or_phone: str = os.getenv("X_EMAIL_OR_PHONE", "")
    
    # Browser settings
    # Default headless=True for production (faster), override with --visible flag
    headless: bool = os.getenv("FETCHER_HEADLESS", "true").lower() in ("1", "true", "yes")
    slow_mo: int = 50
    viewport_width: int = 1280
    viewport_height: int = 720
    user_agents: List[str] = None
    
    # Timeouts (milliseconds)
    page_load_timeout: int = 30000
    element_wait_timeout: int = 15000
    login_verification_timeout: int = 15000
    
    # Human-like delays (seconds)
    min_human_delay: float = 1.0
    max_human_delay: float = 3.0
    recovery_delay_min: float = 3.0
    recovery_delay_max: float = 6.0
    session_refresh_delay_min: float = 3.0
    session_refresh_delay_max: float = 6.0
    
    # Collection limits
    max_consecutive_empty_scrolls: int = 15
    max_consecutive_existing_tweets: int = 10
    max_recovery_attempts: int = 5
    max_session_retries: int = 5
    max_sessions: int = 10
    
    # Thread detection tuning
    thread_detect_interval: int = 5  # scroll cycles between inline thread detection
    
    # Feature flags
    collect_threads: bool = os.getenv("FETCHER_COLLECT_THREADS", "false").lower() in ("1","true","yes")
    use_adaptive_scroll: bool = os.getenv("FETCHER_ADAPTIVE_SCROLL", "true").lower() in ("1", "true", "yes")
    
    # Session management
    session_size: int = 800
    large_collection_threshold: int = 800
    
    # Scrolling parameters
    scroll_amounts: List[int] = None
    aggressive_scroll_multiplier: float = 2.0
    
    # Database settings
    db_timeout: float = 10.0
```

### Environment Variables

**Required:**
- `X_USERNAME`: X/Twitter login username
- `X_PASSWORD`: X/Twitter login password  
- `X_EMAIL_OR_PHONE`: Associated email or phone for verification

**Optional:**
- `FETCHER_MAX_TWEETS`: Default maximum tweets per user
- `FETCHER_SESSION_SIZE`: Multi-session chunk size
- `FETCHER_HEADLESS`: Set to `false` to show browser window (default: `true`)
- `FETCHER_COLLECT_THREADS`: Set to `true` to enable thread detection (default: `false`)
- `FETCHER_ADAPTIVE_SCROLL`: Set to `false` to use fixed scroll delays (default: `true`)
- `FETCHER_SLOW_MO`: Browser slow motion delay in ms (default: `50`)
- `DATABASE_PATH`: Custom database location

### Configuration Features

**Anti-Detection Measures:**
- Randomized user agent rotation
- Dynamic viewport sizing
- Configurable delays and timing
- Session persistence and rotation
- Headless mode (default) for reduced fingerprint

**Performance Tuning (P1-P8 Optimizations):**
- **P1**: Parallel HTML extraction with BeautifulSoup
- **P2**: Adaptive scroll delays (wait for content, not fixed time)
- **P3**: Batch database writes (buffered inserts)
- **P4**: Skip video hover (extract URL from poster pattern)
- **P5**: Skip "Show More" check for short tweets (<250 chars)
- **P6**: Pre-scroll content loading detection
- **P7**: Headless mode enabled by default
- **P8**: No navigation for quoted tweets (inline extraction)
- Adjustable timeouts and retry limits
- Memory and resource management
- Collection thresholds and limits
- Database connection pooling

---

## 🔧 Usage Examples

### Command Line Interface

```bash
# Fetch latest tweets from default accounts (fast updates)
./run_in_venv.sh fetch --latest

# Fetch specific user (full history)
./run_in_venv.sh fetch --user "username"

# Fetch latest from specific user only
./run_in_venv.sh fetch --user "username" --latest

# Fetch with custom limits
./run_in_venv.sh fetch --user "username" --max 500

# Re-fetch a specific tweet (updates existing data)
./run_in_venv.sh fetch --refetch "1234567890123456789"

# Delete all data for account and refetch from scratch
./run_in_venv.sh fetch --refetch-all "username"

# Enable resume mode to fetch older tweets beyond current collection
./run_in_venv.sh fetch --user "username" --resume

# Show browser window for debugging (disables headless mode)
./run_in_venv.sh fetch --user "username" --visible

# Disable thread detection for faster collection
./run_in_venv.sh fetch --user "username" --no-threads

# Unlimited collection (use with caution)
./run_in_venv.sh fetch --user "username" --max 10000
```

**Command Line Options:**
- `--user, -u`: Single username to fetch (overrides defaults)
- `--max`: Maximum tweets per user (default: unlimited)
- `--latest`: Use latest mode (stops after 10 consecutive existing tweets)
- `--resume`: Enable resume/seek mode to fetch older tweets (disabled by default)
- `--refetch`: Re-fetch specific tweet ID with updated data
- `--refetch-all`: Delete and refetch entire account from scratch
- `--visible`: Show browser window (disable headless mode) for debugging
- `--no-threads`: Disable thread detection and collection for this run

### Performance Tracking

**Built-in Performance Monitoring:**
- **Operation Counting:** Tracks total tweets processed
- **Execution Timing:** Measures total collection time
- **Metrics Collection:** CPU, memory, and throughput statistics
- **Summary Reports:** Detailed performance breakdowns

**Performance Integration:**
```python
from utils.performance import start_tracking, stop_tracking, print_performance_summary

# Start tracking at beginning of collection
tracker = start_tracking("Tweet Fetcher")

# Increment counter for each tweet processed
tracker.increment_operations(total_tweets)

# Print summary at completion
metrics = stop_tracking(tracker)
print_performance_summary(metrics)
```

**Output Example:**
```
⏱️  Execution completed in: 5m 23s
📊 Total tweets fetched and saved: 1247
🎯 Accounts processed: 3
📈 Average tweets per account: 415.7

Performance Summary:
- Total Operations: 1247
- Execution Time: 323.45s
- Operations/Second: 3.85
- Peak Memory Usage: 487MB
```

---

## 🧪 Testing & Validation

### Test Coverage

**Comprehensive Test Suite:**
- **`test_fetch_tweets.py`**: 908 lines - Main collection workflows and CLI
- **`test_parsers.py`**: 599 lines - Content extraction and post type analysis  
- **`test_collector.py`**: 520 lines - Tweet collection logic and error handling
- **`test_db.py`**: 334 lines - Database operations and data persistence
- **`test_media_monitor.py`**: 254 lines - Network monitoring and media capture
- **`test_refetch_manager.py`**: 238 lines - Individual tweet refetching
- **`test_resume_manager.py`**: 206 lines - Resume positioning and search navigation
- **`test_fetch_integration.py`**: 136 lines - End-to-end integration tests

**Test Categories:**

1. **Unit Tests:** Individual component behavior
   - Parser functions and data extraction
   - Database CRUD operations
   - Media URL processing and validation
   - Configuration management

2. **Integration Tests:** Component interaction
   - End-to-end tweet collection workflows
   - Multi-session collection validation
   - Cross-component data flow
   - Error recovery scenarios

3. **Browser Tests:** Playwright-based web interaction
   - DOM element detection and extraction
   - Dynamic content loading
   - Anti-detection measure validation
   - Network request monitoring

4. **Database Tests:** Data persistence and retrieval
   - Schema validation and migrations
   - Concurrent access handling
   - Data integrity and consistency
   - Performance optimization

5. **Performance Tests:** Collection speed and reliability
   - Large-scale collection efficiency
   - Memory usage monitoring
   - Rate limiting and anti-detection
   - Error recovery under load

**Test Infrastructure:**
- **conftest.py**: Shared test fixtures and setup
- **Mock Objects**: Simulated browser interactions for CI/CD
- **Test Data**: Realistic tweet structures and edge cases
- **Coverage Reporting**: Detailed coverage analysis and reporting

---

## 🚀 Advanced Features

### Intelligent Content Detection

**Post Type Classification:**
The system analyzes tweet structure and context to classify content types with high accuracy:

- **`original`**: Standalone content created by the account
- **`repost_own`**: Self-retweets (automatically skipped to avoid duplicates)
- **`repost_other`**: Content shared from other accounts (retweets)
- **`repost_reply`**: Comments on retweeted content (replies to retweets)
- **`thread`**: Multi-part conversations and thread continuations
- **`pinned`**: Pinned tweets (automatically skipped)

**Advanced Classification Logic:**
```python
# Multi-stage analysis with fallback strategies
post_analysis = {
    'post_type': 'original',  # Default assumption
    'is_pinned': 0,
    'original_author': None,   # For quotes/retweets
    'original_tweet_id': None, # For quotes/retweets
    'original_content': None,  # Quoted tweet text
    'should_skip': False       # Skip flag for duplicates/pinned
}
```

**Detection Strategies:**
1. **Pinned Detection**: Identifies pinned tweets via social context indicators
2. **Quote Tweet Analysis**: Extracts quoted content and metadata from nested structures
3. **Repost Classification**: Distinguishes self-retweets from external retweets
4. **Reply Detection**: Identifies replies to other tweets vs. standalone content
5. **Thread Recognition**: Detects thread continuations via social context

### Media Handling

**Supported Media Types:**
- Images (JPG, PNG, GIF, WebP)
- Videos (MP4, WebM, AVI, MOV)
- GIFs and animated content
- Multiple media per tweet

**Media Processing:**
- URL extraction from page elements
- Network request monitoring
- Content type detection
- Media count and type tracking

### Error Recovery

**Multi-Level Recovery Strategies:**
1. **Page Refresh:** Reload current page state
2. **Session Recreation:** New browser context with fresh session
3. **Cache Clearing:** Remove stored data and cookies
4. **Alternative Navigation:** Different access methods and URLs
5. **Graceful Degradation:** Continue with partial data collection

**Recovery Implementation:**
```python
def try_recovery_strategies(self, page, attempt_number: int) -> bool:
    strategies = [
        ("refresh_page", lambda: page.reload(wait_until="domcontentloaded")),
        ("clear_cache", lambda: page.evaluate("localStorage.clear(); sessionStorage.clear();")),
        ("jump_to_bottom", lambda: page.evaluate("window.scrollTo(0, document.body.scrollHeight)")),
        ("force_reload_tweets", lambda: page.evaluate("window.location.reload(true)")),
        ("random_scroll_pattern", lambda: page.evaluate(f"window.scrollBy(0, {1000 + random.randint(500, 2000)})")),
    ]
    
    if attempt_number <= len(strategies):
        strategy_name, strategy_func = strategies[attempt_number - 1]
        strategy_func()
        # Check if recovery successful by finding tweet elements
        articles = page.query_selector_all('article[data-testid="tweet"]')
        return len(articles) > 0
    return False
```

**Error Logging & Monitoring:**
- **Database Error Table:** `scrape_errors` for tracking failures
- **Context Preservation:** Error context and metadata storage
- **Retry Logic:** Exponential backoff with jitter
- **Failure Analysis:** Pattern recognition for systemic issues

---

## 📈 Monitoring & Metrics

### Performance Tracking

**Collection Metrics:**
- Tweets collected per minute
- Success rate by account
- Error frequency and types
- Session duration and efficiency

**System Health:**
- Browser memory usage
- Network request success
- Database operation latency
- Anti-detection effectiveness

### Logging System

**Log Levels:**
- **DEBUG:** Detailed operation tracing
- **INFO:** Collection progress and results
- **WARNING:** Recoverable errors and issues
- **ERROR:** Critical failures requiring attention

**Log Categories:**
- Collection progress
- Error conditions
- Performance metrics
- Database operations
- Browser interactions

---

## 🔒 Security & Compliance

### Anti-Detection Measures

**Browser Fingerprinting:**
- Randomized user agents
- Dynamic viewport sizes
- Canvas and WebGL randomization
- Font and plugin spoofing

**Behavioral Patterns:**
- Human-like delays and timing
- Natural scrolling patterns
- Session management and rotation
- Request rate limiting

### Data Privacy

**Collection Ethics:**
- Public data only (no private messages)
- Research and analysis purposes
- Transparent data usage
- User data protection

---

## 🔄 Future Enhancements

### Planned Improvements

#### Fetcher-Specific Enhancements
1. **Real-Time Streaming:** WebSocket-based live tweet monitoring
2. **Multi-Platform Support:** Extend beyond Twitter/X to Telegram, Facebook, Instagram
3. **Advanced Resume:** Bookmark-based continuation for interrupted collections
4. **Distributed Collection:** Multi-instance coordination for large-scale harvesting
5. **API Integration:** RESTful collection endpoints for programmatic access
6. **Content Archiving:** Long-term tweet preservation with metadata

#### Analysis Engine Improvements
1. **Batch Processing:** Process multiple tweets simultaneously for improved throughput
2. **Multi-Category Support:** Allow tweets to belong to multiple categories
3. **Thread Analysis:** Analyze complete conversation threads instead of individual posts
4. **Local Multimodal Processing:** Enhanced video and image content analysis
5. **Link Verification:** Automatic verification and preview of embedded links
6. **Disinformation Tracking:** Real-time fact-checking and source verification

#### Platform Integration
1. **Slack/Discord Alerts:** Real-time notifications for high-priority content
2. **IFTTT/Zapier Integration:** Connect with external workflow automation tools
3. **User Submission Portal:** Allow public submission of accounts/posts for analysis
4. **Research Paper Generation:** Automated analysis summary reports
5. **Academic Partnerships:** Integration with university research databases

#### Advanced Features
1. **Sentiment Analysis:** Add emotional tone detection beyond category classification
2. **Topic Modeling:** Implement unsupervised topic discovery
3. **Event Detection:** Automatically identify significant political events
4. **Predictive Modeling:** Forecast content trends and narrative shifts
5. **Network Analysis:** Analyze account interaction patterns and influence networks
6. **Trend Analysis:** Temporal analysis to track narrative evolution

#### Technical Improvements
1. **Memory Optimization:** Reduce memory usage during large batch analysis
2. **Model Quantization:** Optimize LLM response times through model compression
3. **A/B Testing Framework:** Compare different LLM models and prompting strategies
4. **Multi-Language Support:** Enhanced handling of Catalan, Galician, and other languages
5. **Context Awareness:** Implement thread/conversation context for better analysis

---

## 📚 Related Documentation

- [Analyzer Pipeline Architecture](ANALYZER_PIPELINE_ARCHITECTURE.md) - Content analysis and classification system
- [Retrieval System Architecture](RETRIEVAL_SYSTEM_ARCHITECTURE.md) - Evidence retrieval and verification system
- [Multi-Model Analysis](MULTI_MODEL_ANALYSIS.md) - Advanced LLM integration and model management
- [Command Reference](COMMAND_REFERENCE.md) - Complete CLI command documentation
- [Development Guide](DEVELOPMENT.md) - Development setup and contribution guidelines
- [Docker Deployment](DOCKER_DEPLOYMENT.md) - Containerization and deployment instructions
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues and solutions
- [Database Schema Documentation](../database/) - Database structure and migrations
- [API Reference](../web/) - Web interface and API documentation</content>
<parameter name="filePath">/Users/antonio/projects/bulos/dimetuverdad/docs/FETCHER_SYSTEM_ARCHITECTURE.md