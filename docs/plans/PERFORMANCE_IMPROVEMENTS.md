# Fetcher Performance Improvement Plan

## Current State

- **Average rate**: ~1.5 tweets/second (observed)
- **13.4k tweets**: ~2.5 hours
- **100 users × 20k tweets**: ~370 hours (~15 days)

## Goal

- **Target rate**: 5-10 tweets/second (with parallelization)
- **100 users × 20k tweets**: ~50-80 hours (2-3 days)

---

## Phase 1: Safe Quick Wins (Implemented)

### 1.1 Disable Thread Collection (--fast or --no-threads)

Thread collection opens new pages and scrolls through entire threads, adding significant overhead.

```bash
./run_in_venv.sh fetch --user infovlogger36 --fast
```

**Impact**: Saves 5-30 seconds per thread detected

### 1.2 Reduce Human-Like Delays ✅ (Now Default)

Reduced delays are now the **default** configuration:

```python
# In fetcher/config.py (default values)
min_human_delay: float = 0.5  # was 1.0
max_human_delay: float = 1.5  # was 3.0
```

**Impact**: ~20% speedup on scroll cycles (built-in)

### ⚠️ Things We Should NOT Skip

**Video Monitoring**: Captures real MP4 URLs like:
```
https://video.twimg.com/amplify_video/{id}/vid/avc1/720x720/video.mp4
```
These are **usable downloadable video URLs**, not blob references!

**Quoted Tweet Tab Navigation**: Timeline view truncates quoted content. Test showed:
- Inline P8 extraction: 128 chars (truncated)
- Tab navigation: 421 chars + 2 media (complete)

**Aggressive Scroll Amounts**: Past testing showed larger scroll amounts (>1000px) cause missed tweets. Current conservative amounts (400-800px) are tuned.

---

## Phase 2: Parallelization (4x+ speedup) - Main Opportunity

### 2.1 Multiple Browser Instances

Run multiple Playwright browsers in parallel, each handling different users:

```python
# parallel_fetcher.py
import asyncio
from concurrent.futures import ProcessPoolExecutor

async def fetch_users_parallel(users: List[str], max_workers: int = 4):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_single_user, user) for user in users]
        results = [f.result() for f in futures]
    return results
```

**Impact**: 4 parallel workers = 4x throughput

### 2.2 Async Playwright ✅

**Implemented**: Use `--async` flag for async Playwright.

```bash
./run_in_venv.sh fetch --user username --async
```

Files added:
- `fetcher/async_scroller.py` - Async scroll operations
- `fetcher/async_session_manager.py` - Async browser management
- `fetcher/async_collector.py` - Async tweet collection

```python
# In async_collector.py
async with async_playwright() as p:
    browser, context, page = await session_manager.create_browser_context(p)
    tweets = await collector.collect_tweets_from_page(page, username, max_tweets, conn)
```

---

## Phase 3: Twitter API Alternative (10-50x speedup)

### 3.1 Twitter API v2 (If Available)

If you have API access, use direct API calls:

```python
# Much faster than scraping
import tweepy

client = tweepy.Client(bearer_token=BEARER_TOKEN)
tweets = client.get_users_tweets(user_id, max_results=100)
```

**Impact**: ~100 tweets per API call, rate limited but much faster

### 3.2 Nitter/RSS Proxy

Use Nitter instances or RSS feeds for basic text content:

```python
# RSS-based collection (text only, no media)
import feedparser
feed = feedparser.parse(f"https://nitter.net/{username}/rss")
```

---

## Phase 4: Infrastructure Optimizations (✅ Implemented)

### 4.1 Persistent Browser Session

Keep browser open between users instead of restarting:

```python
# Reuse browser across all users
browser = p.chromium.launch()
for user in users:
    context = browser.new_context()
    page = context.new_page()
    fetch_user(page, user)
    context.close()  # Only close context, not browser
browser.close()
```

### 4.2 Database Batch Writes ✅

**Implemented**: Tweets are now buffered using `TweetBuffer` and written in batches.

```python
# In fetcher/collector.py
tweet_buffer = TweetBuffer(conn, batch_size=self.config.batch_write_size)

# During collection
tweet_buffer.add(tweet_data)  # Auto-flushes at batch_size

# At end
tweet_buffer.flush()  # Flush remaining
```

Config option: `batch_write_size` (default: 50)

### 4.3 Skip Duplicate Checks During Refetch ✅

**Implemented**: `--refetch-all` automatically enables `skip_duplicate_check`.

```python
# In fetcher/refetch_manager.py
cfg.skip_duplicate_check = True  # Set for refetch-all mode

# In collector.py check_tweet_exists_in_db()
if self.config.skip_duplicate_check:
    return False, None  # Skip DB lookup
```

---

## Implementation Priority

| Priority | Improvement | Speedup | Effort | Data Loss? |
|----------|-------------|---------|--------|------------|
| 🔴 High | **Parallel browsers** | **4x** | Medium | ❌ None |
| 🔴 High | Disable threads (--fast) | 1.3x | ✅ Done | ❌ None |
| 🟡 Medium | Reduce delays | 1.2x | ✅ Default | ❌ None |
| 🟡 Medium | Batch DB writes | 1.1x | ✅ Done | ❌ None |
| 🟡 Medium | Skip dupe checks (refetch) | 1.1x | ✅ Done | ❌ None |
| 🟢 Low | Async Playwright (--async) | 1.5x | ✅ Done | ❌ None |
| 🟢 Low | API integration | 10x | Medium | ❌ None |
| ❌ Don't | Skip video monitoring | - | - | ⚠️ Loses MP4 URLs |
| ❌ Don't | Skip quoted navigation | - | - | ⚠️ Loses truncated text |
| ❌ Don't | Aggressive scroll amounts | - | - | ⚠️ Misses tweets |

---

## Implemented Optimizations

### Default: Reduced Delays (Now Standard)

Human-like delays are now reduced by default:
- Default delays: 0.5-1.5s (was 1.0-3.0s)
- No flag needed - this is the new standard

### --fast Flag

Disables thread collection for faster bulk fetching:

```bash
./run_in_venv.sh fetch --user infovlogger36 --fast
```

### --async Flag

Uses async Playwright for improved I/O handling:

```bash
./run_in_venv.sh fetch --user infovlogger36 --async
```

### Database Batch Writes (Automatic)

Tweets are buffered and written in batches of 50 (configurable via `batch_write_size`).
Reduces transaction overhead and improves write performance.

### Skip Duplicate Checks (--refetch-all)

When using `--refetch-all`, duplicate checks are automatically skipped since the database
is cleared first. This saves one DB query per tweet.

```bash
./run_in_venv.sh fetch --refetch-all username
```

### What We Do NOT Optimize (Data Quality)

- ❌ Does NOT skip video monitoring (captures real MP4 URLs)
- ❌ Does NOT skip quoted navigation (timeline truncates content)
- ❌ Does NOT increase scroll amounts (would miss tweets)

Expected combined impact: **~1.5-2x faster** for single-threaded collection

---

## Parallel Fetcher Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Parallel Fetcher                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User Queue: [user1, user2, user3, ... user100]            │
│                                                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │Worker 1 │  │Worker 2 │  │Worker 3 │  │Worker 4 │       │
│  │Browser 1│  │Browser 2│  │Browser 3│  │Browser 4│       │
│  │→ user1  │  │→ user2  │  │→ user3  │  │→ user4  │       │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘       │
│       │            │            │            │             │
│       └────────────┴────────────┴────────────┘             │
│                         │                                   │
│                    ┌────▼────┐                              │
│                    │ SQLite  │                              │
│                    │   DB    │                              │
│                    └─────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

With 4 parallel workers in fast mode:
- **100 users × 20k tweets** = 2,000,000 tweets
- **Rate**: 4 workers × 3 tweets/s = 12 tweets/s
- **Time**: 2,000,000 / 12 = ~167,000s = **~46 hours (2 days)**
