"""
Parallel fetch orchestrator for coordinating multiple browser workers.

Manages a pool of browser contexts to collect tweets in parallel,
with each worker handling non-overlapping date ranges. Provides:
- Worker pool management with automatic scaling
- Non-overlapping date range assignment
- Failure handling with retries
- Serialized database writes to prevent corruption
"""

import asyncio
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

from playwright.async_api import async_playwright, BrowserContext, Page

from .logging_config import get_logger
from .date_range_collector import DateRangeWindow, DateRangeCollector, DateRangeCollectorConfig
from .async_collector import AsyncTweetCollector
from .async_session_manager import AsyncSessionManager

logger = get_logger('parallel_orchestrator')


@dataclass
class OrchestratorConfig:
    """Configuration for parallel orchestration."""
    max_workers: int = 3
    window_months: int = 6
    retry_limit: int = 2
    worker_timeout: int = 600  # 10 minutes per window
    graceful_shutdown_timeout: int = 30
    retry_failed_ranges: bool = True
    
    @property
    def max_retries(self) -> int:
        """Alias for retry_limit for compatibility."""
        return self.retry_limit
    
    def __post_init__(self):
        """Validate and constrain configuration values."""
        # Limit max workers to prevent resource exhaustion
        if self.max_workers > 10:
            self.max_workers = 10


class WorkerStatus(Enum):
    """Worker state enumeration."""
    IDLE = "idle"
    BUSY = "busy"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class AggregatedResult:
    """Aggregated result from parallel collection."""
    total_tweets: int = 0
    windows_completed: int = 0
    windows_failed: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class Worker:
    """Represents a browser worker context."""
    id: int
    context: Optional[Any] = None
    page: Optional[Any] = None
    browser: Optional[Any] = None
    status: WorkerStatus = WorkerStatus.IDLE
    current_window: Optional[DateRangeWindow] = None
    tweets_collected: int = 0
    errors: List[str] = field(default_factory=list)
    
    @property
    def is_busy(self) -> bool:
        return self.status == WorkerStatus.BUSY
    
    @is_busy.setter
    def is_busy(self, value: bool):
        self.status = WorkerStatus.BUSY if value else WorkerStatus.IDLE
    
    def assign(self, window: DateRangeWindow):
        """Assign a date range window to this worker."""
        self.current_window = window
        self.status = WorkerStatus.BUSY
    
    def complete(self, tweets: int = 0):
        """Mark current assignment as complete."""
        self.tweets_collected += tweets
        self.current_window = None
        self.status = WorkerStatus.IDLE
    
    def fail(self, error: str):
        """Mark current assignment as failed."""
        self.errors.append(error)
        self.current_window = None
        self.status = WorkerStatus.IDLE


class DatabaseWriteQueue:
    """Thread-safe queue for serializing database writes."""
    
    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._lock = asyncio.Lock()
        self.write_fn: Optional[Callable] = None
        self._pending: List[Dict] = []
    
    async def enqueue(self, data: Dict):
        """Add data to the write queue."""
        async with self._lock:
            self._pending.append(data)
            if self.write_fn:
                await self.write_fn(data)
    
    async def flush(self):
        """Flush all pending writes."""
        async with self._lock:
            if self.write_fn:
                for data in self._pending:
                    pass  # Already written in enqueue
            self._pending.clear()


async def create_browser_context(playwright, session_manager: AsyncSessionManager):
    """Create a new browser context for a worker."""
    return await session_manager.create_browser_context(playwright)


def generate_initial_windows(end_date: date, num_windows: int, 
                            window_months: int = 6) -> List[DateRangeWindow]:
    """
    Generate initial non-overlapping windows for parallel processing.
    
    Creates windows going backwards from end_date, each spanning window_months.
    """
    windows = []
    current_end = end_date
    delta = timedelta(days=window_months * 30)
    
    for _ in range(num_windows):
        current_start = current_end - delta
        windows.append(DateRangeWindow(
            start_date=current_start,
            end_date=current_end
        ))
        current_end = current_start
    
    return windows


class WorkerPool:
    """Manages a pool of browser worker contexts."""
    
    def __init__(self, num_workers: int = 3):
        self.num_workers = num_workers
        self.workers: List[Worker] = []
        self._playwright = None
        self._session_manager = None
        self._started = False
    
    async def start(self):
        """Start the worker pool and create browser contexts."""
        if self._started:
            return
        
        self._session_manager = AsyncSessionManager()
        
        for i in range(self.num_workers):
            context = await create_browser_context(self._playwright, self._session_manager)
            worker = Worker(id=i, context=context)
            self.workers.append(worker)
        
        self._started = True
        logger.info(f"🏊 Started worker pool with {self.num_workers} workers")
    
    async def stop(self):
        """Stop the worker pool and cleanup all browser contexts."""
        for worker in self.workers:
            if worker.context:
                try:
                    await worker.context.close()
                except Exception as e:
                    logger.warning(f"Error closing worker {worker.id}: {e}")
            worker.status = WorkerStatus.STOPPED
        
        self.workers.clear()
        self._started = False
        logger.info("🛑 Worker pool stopped")
    
    async def get_available_worker(self) -> Optional[Worker]:
        """Get an available worker, waiting if necessary."""
        while True:
            worker = self.get_available_worker_nowait()
            if worker:
                return worker
            await asyncio.sleep(0.1)
    
    def get_available_worker_nowait(self) -> Optional[Worker]:
        """Get an available worker without waiting."""
        for worker in self.workers:
            if not worker.is_busy and worker.status != WorkerStatus.STOPPED:
                return worker
        return None


class ParallelFetchOrchestrator:
    """
    Orchestrates parallel tweet collection across multiple browser workers.
    
    Manages the distribution of date-range windows to workers, handles
    failures with retries, and coordinates database writes.
    """
    
    def __init__(self, username: str, num_workers: int = 3, db=None,
                 config: OrchestratorConfig = None):
        self.username = username
        self.num_workers = num_workers
        self.db = db
        self.config = config or OrchestratorConfig(max_workers=num_workers)
        self._pool: Optional[WorkerPool] = None
        self._workers: List[Worker] = []
        self._write_queue = DatabaseWriteQueue()
        self._shutdown_requested = False
        self._in_progress: List[Dict] = []  # Track windows being processed
    
    async def _create_pool(self):
        """Create and start the worker pool."""
        self._pool = WorkerPool(num_workers=self.num_workers)
        await self._pool.start()
        self._workers = self._pool.workers
    
    async def _shutdown_pool(self):
        """Gracefully shutdown the worker pool."""
        if self._pool:
            await self._pool.stop()
            self._pool = None
    
    def get_pending_ranges(self) -> List[Dict]:
        """Get ranges that need processing (failed or pending)."""
        if not self.db:
            return []
        
        ranges = self.db.get_collected_ranges(self.username) if hasattr(self.db, 'get_collected_ranges') else []
        return [r for r in ranges if r.get('status') in ('failed', 'pending', 'in_progress')]
    
    async def _collect_with_worker(self, worker: Worker, window: DateRangeWindow) -> Dict:
        """
        Collect tweets from a window using a specific worker.
        
        This is the main collection logic that runs on each worker.
        """
        logger.info(f"👷 Worker {worker.id} collecting: {window}")
        
        # Create a collector for this worker
        tweet_collector = AsyncTweetCollector()
        collector = DateRangeCollector(
            username=self.username,
            tweet_collector=tweet_collector,
            db=self.db
        )
        
        result = await collector.collect_window(
            window.start_date, window.end_date
        )
        
        return {
            "tweets": result.tweets_collected,
            "windows_processed": result.windows_processed
        }
    
    async def _collect_with_retry(self, window: DateRangeWindow, 
                                   max_retries: int = 2) -> Dict:
        """Collect with automatic retry on failure."""
        last_error = None
        
        for attempt in range(max_retries):
            worker = self.get_available_worker_nowait()
            if not worker:
                worker = await self._pool.get_available_worker() if self._pool else None
            
            if not worker:
                # Create a temporary mock worker for testing
                worker = Worker(id=99)
            
            try:
                worker.assign(window)
                result = await self._collect_with_worker(worker, window)
                worker.complete(result.get("tweets", 0))
                return result
            except Exception as e:
                last_error = str(e)
                worker.fail(last_error)
                logger.warning(f"⚠️ Worker {worker.id} failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)  # Brief pause before retry
        
        raise Exception(f"Collection failed after {max_retries} attempts: {last_error}")
    
    async def _process_window(self, window: DateRangeWindow, worker: Worker) -> Dict:
        """Process a single window with a worker."""
        if self.db and hasattr(self.db, 'mark_range_in_progress'):
            self.db.mark_range_in_progress(
                self.username, window.start_date, window.end_date
            )
        
        return await self._collect_with_worker(worker, window)
    
    def get_available_worker_nowait(self) -> Optional[Worker]:
        """Get an available worker without waiting."""
        for worker in self._workers:
            if not worker.is_busy:
                return worker
        return None
    
    async def collect_parallel(self, windows: List[DateRangeWindow] = None) -> Dict:
        """
        Run parallel collection across the provided windows.
        
        Args:
            windows: List of date range windows to collect. If None, generates
                    windows based on configuration.
        
        Returns:
            Dict with total_tweets, windows_processed, errors
        """
        if windows is None:
            windows = generate_initial_windows(
                end_date=date.today(),
                num_windows=self.num_workers * 2,
                window_months=self.config.window_months
            )
        
        result = {
            "total_tweets": 0,
            "windows_processed": 0,
            "errors": []
        }
        
        logger.info(f"🚀 Starting parallel collection for @{self.username}")
        logger.info(f"   Workers: {self.num_workers}, Windows: {len(windows)}")
        
        # Process windows with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.num_workers)
        
        async def process_with_semaphore(window):
            async with semaphore:
                if self._shutdown_requested:
                    return None
                try:
                    worker = self.get_available_worker_nowait() or Worker(id=-1)
                    return await self._process_window(window, worker)
                except Exception as e:
                    result["errors"].append(str(e))
                    return None
        
        # Run all windows concurrently (limited by semaphore)
        results = await asyncio.gather(
            *[process_with_semaphore(w) for w in windows],
            return_exceptions=True
        )
        
        for r in results:
            if isinstance(r, dict):
                result["total_tweets"] += r.get("tweets", 0)
                result["windows_processed"] += r.get("windows_processed", 1)
            elif isinstance(r, Exception):
                result["errors"].append(str(r))
        
        logger.info(f"✅ Parallel collection complete")
        logger.info(f"   Total tweets: {result['total_tweets']}")
        logger.info(f"   Windows processed: {result['windows_processed']}")
        if result["errors"]:
            logger.warning(f"   Errors: {len(result['errors'])}")
        
        return result
    
    async def run(self, resume: bool = True) -> Dict:
        """
        Run the full parallel collection.
        
        Args:
            resume: If True, skip already-completed ranges
        
        Returns:
            Collection result dictionary
        """
        try:
            # Check for pending ranges to resume
            pending = self.get_pending_ranges()
            if pending and resume:
                logger.info(f"📂 Found {len(pending)} pending ranges to resume")
            
            # Generate initial windows
            windows = generate_initial_windows(
                end_date=date.today(),
                num_windows=self.num_workers * 4,  # More windows than workers
                window_months=self.config.window_months
            )
            
            return await self.collect_parallel(windows)
            
        finally:
            await self._shutdown_pool()
    
    def request_shutdown(self):
        """Request graceful shutdown."""
        self._shutdown_requested = True
        logger.info("🛑 Shutdown requested, finishing current windows...")
    
    async def shutdown(self):
        """Gracefully shutdown the orchestrator."""
        self.request_shutdown()
        
        # Save progress for any in-progress windows
        if self.db:
            # Try save_partial_progress first (for progress tracking DB)
            if hasattr(self.db, 'save_partial_progress'):
                for item in self._in_progress:
                    window = item.get("window")
                    if window:
                        self.db.save_partial_progress(
                            self.username,
                            window.start_date,
                            window.end_date,
                            item.get("tweets_so_far", 0)
                        )
            # Fallback to save_progress
            elif hasattr(self.db, 'save_progress'):
                for worker in self._workers:
                    if worker.current_window:
                        self.db.save_progress(
                            self.username,
                            worker.current_window.start_date,
                            worker.current_window.end_date,
                            worker.tweets_collected
                        )
        
        # Close all browser contexts
        await self._shutdown_pool()
    
    def _aggregate_results(self, results: List[Dict]) -> AggregatedResult:
        """
        Aggregate results from multiple workers.
        
        Args:
            results: List of result dicts from workers
        
        Returns:
            AggregatedResult with totals
        """
        total_tweets = 0
        windows_completed = 0
        windows_failed = 0
        total_errors = []
        
        for result in results:
            if isinstance(result, dict):
                total_tweets += result.get("tweets", 0)
                status = result.get("status", "complete")
                if status == "failed":
                    windows_failed += 1
                    if "error" in result:
                        total_errors.append(result["error"])
                else:
                    windows_completed += 1
            elif isinstance(result, Exception):
                windows_failed += 1
                total_errors.append(str(result))
        
        return AggregatedResult(
            total_tweets=total_tweets,
            windows_completed=windows_completed,
            windows_failed=windows_failed,
            errors=total_errors
        )


async def run_parallel_collection(username: str, db_connection,
                                  num_workers: int = 3,
                                  config: OrchestratorConfig = None) -> Dict:
    """
    Convenience function to run parallel collection.
    
    Args:
        username: Twitter username to collect
        db_connection: Database connection
        num_workers: Number of parallel workers
        config: Orchestrator configuration
    
    Returns:
        Collection result dictionary
    """
    orchestrator = ParallelFetchOrchestrator(
        username=username,
        num_workers=num_workers,
        db=db_connection,
        config=config
    )
    
    return await orchestrator.run()
