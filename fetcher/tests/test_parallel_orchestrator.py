"""
Unit tests for ParallelFetchOrchestrator - Coordinate parallel date-range collection.

Tests the orchestrator that:
- Manages pool of browser worker contexts
- Assigns non-overlapping date ranges to workers
- Handles worker failures with reassignment
- Coordinates database writes safely
"""

import pytest
from datetime import date, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import asyncio


class TestWorkerPool:
    """Tests for worker pool management."""
    
    def test_pool_creation_with_worker_count(self):
        """Pool creates specified number of workers."""
        from fetcher.parallel_orchestrator import WorkerPool
        
        pool = WorkerPool(num_workers=3)
        
        assert pool.num_workers == 3
        assert len(pool.workers) == 0  # Not started yet
    
    @pytest.mark.asyncio
    async def test_pool_starts_workers(self):
        """Pool can start all workers."""
        from fetcher.parallel_orchestrator import WorkerPool
        
        pool = WorkerPool(num_workers=2)
        
        with patch('fetcher.parallel_orchestrator.create_browser_context', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = AsyncMock()
            
            await pool.start()
            
            assert len(pool.workers) == 2
            assert mock_create.call_count == 2
    
    @pytest.mark.asyncio
    async def test_pool_stops_workers(self):
        """Pool can stop and cleanup all workers."""
        from fetcher.parallel_orchestrator import WorkerPool
        
        pool = WorkerPool(num_workers=2)
        mock_context = AsyncMock()
        
        with patch('fetcher.parallel_orchestrator.create_browser_context', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_context
            
            await pool.start()
            await pool.stop()
            
            assert mock_context.close.call_count == 2
    
    @pytest.mark.asyncio
    async def test_get_available_worker(self):
        """Pool returns available worker for assignment."""
        from fetcher.parallel_orchestrator import WorkerPool
        
        pool = WorkerPool(num_workers=2)
        
        with patch('fetcher.parallel_orchestrator.create_browser_context', new_callable=AsyncMock):
            await pool.start()
            
            worker = await pool.get_available_worker()
            
            assert worker is not None
            assert worker.is_busy is False
    
    @pytest.mark.asyncio
    async def test_no_available_worker_when_all_busy(self):
        """Pool waits when all workers are busy."""
        from fetcher.parallel_orchestrator import WorkerPool
        
        pool = WorkerPool(num_workers=1)
        
        with patch('fetcher.parallel_orchestrator.create_browser_context', new_callable=AsyncMock):
            await pool.start()
            
            # Mark the only worker as busy
            pool.workers[0].is_busy = True
            
            # Should return None or wait (depending on implementation)
            worker = pool.get_available_worker_nowait()
            
            assert worker is None


class TestDateRangeAssignment:
    """Tests for assigning date ranges to workers."""
    
    def test_generate_initial_windows(self):
        """Generates initial large windows for distribution."""
        from fetcher.parallel_orchestrator import generate_initial_windows
        
        windows = generate_initial_windows(
            end_date=date(2025, 6, 30),
            num_windows=4,
            window_months=6
        )
        
        assert len(windows) == 4
        # Windows should be consecutive going backwards
        assert windows[0].end_date == date(2025, 6, 30)
        assert windows[1].end_date == windows[0].start_date
    
    def test_windows_dont_overlap(self):
        """Generated windows have no overlap."""
        from fetcher.parallel_orchestrator import generate_initial_windows
        
        windows = generate_initial_windows(
            end_date=date(2025, 6, 30),
            num_windows=4,
            window_months=3
        )
        
        for i in range(len(windows) - 1):
            assert windows[i].start_date >= windows[i + 1].end_date
    
    def test_assign_window_to_worker(self):
        """Window can be assigned to specific worker."""
        from fetcher.parallel_orchestrator import Worker, DateRangeWindow
        
        worker = Worker(id=1, context=Mock())
        window = DateRangeWindow(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 6, 30)
        )
        
        worker.assign(window)
        
        assert worker.current_window == window
        assert worker.is_busy is True


class TestParallelOrchestrator:
    """Tests for the main orchestrator class."""
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database."""
        db = Mock()
        db.get_collected_ranges = Mock(return_value=[])
        db.mark_range_complete = Mock()
        db.mark_range_in_progress = Mock()
        db.mark_range_failed = Mock()
        return db
    
    @pytest.mark.asyncio
    async def test_orchestrator_creates_pool(self, mock_db):
        """Orchestrator creates worker pool with configured size."""
        from fetcher.parallel_orchestrator import ParallelFetchOrchestrator
        
        orchestrator = ParallelFetchOrchestrator(
            username="testuser",
            num_workers=3,
            db=mock_db
        )
        
        assert orchestrator.num_workers == 3
    
    @pytest.mark.asyncio
    async def test_orchestrator_distributes_windows(self, mock_db):
        """Orchestrator distributes windows across workers."""
        from fetcher.parallel_orchestrator import ParallelFetchOrchestrator
        
        assigned_windows = []
        
        async def mock_collect(worker, window):
            assigned_windows.append((worker.id, window))
            return {"tweets": 100}
        
        orchestrator = ParallelFetchOrchestrator(
            username="testuser",
            num_workers=2,
            db=mock_db
        )
        
        with patch.object(orchestrator, '_collect_with_worker', side_effect=mock_collect):
            with patch.object(orchestrator, '_create_pool', new_callable=AsyncMock):
                orchestrator._workers = [
                    Mock(id=0, is_busy=False),
                    Mock(id=1, is_busy=False)
                ]
                
                await orchestrator.collect_parallel(
                    windows=[
                        Mock(start_date=date(2025, 1, 1), end_date=date(2025, 3, 31)),
                        Mock(start_date=date(2025, 4, 1), end_date=date(2025, 6, 30))
                    ]
                )
        
        # Both windows should be assigned
        assert len(assigned_windows) == 2
    
    @pytest.mark.asyncio
    async def test_orchestrator_handles_worker_failure(self, mock_db):
        """Failed window is reassigned to another worker."""
        from fetcher.parallel_orchestrator import ParallelFetchOrchestrator
        
        attempt_count = 0
        
        async def failing_then_success(worker, window):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                raise Exception("Worker failed")
            return {"tweets": 50}
        
        orchestrator = ParallelFetchOrchestrator(
            username="testuser",
            num_workers=2,
            db=mock_db
        )
        
        with patch.object(orchestrator, '_collect_with_worker', side_effect=failing_then_success):
            window = Mock(start_date=date(2025, 1, 1), end_date=date(2025, 3, 31))
            
            result = await orchestrator._collect_with_retry(window, max_retries=2)
            
            assert attempt_count == 2
            assert result["tweets"] == 50
    
    @pytest.mark.asyncio
    async def test_orchestrator_marks_progress_in_db(self, mock_db):
        """Orchestrator marks range as in-progress before starting."""
        from fetcher.parallel_orchestrator import ParallelFetchOrchestrator
        
        orchestrator = ParallelFetchOrchestrator(
            username="testuser",
            num_workers=1,
            db=mock_db
        )
        
        async def mock_collect(*args, **kwargs):
            return {"tweets": 100}
        
        with patch.object(orchestrator, '_collect_with_worker', side_effect=mock_collect):
            window = Mock(start_date=date(2025, 1, 1), end_date=date(2025, 3, 31))
            
            await orchestrator._process_window(window, Mock())
            
            mock_db.mark_range_in_progress.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_orchestrator_resumes_from_failed_ranges(self, mock_db):
        """Orchestrator picks up failed ranges on resume."""
        from fetcher.parallel_orchestrator import ParallelFetchOrchestrator
        
        # Simulate a failed range in DB
        mock_db.get_collected_ranges = Mock(return_value=[
            {"start_date": "2025-01-01", "end_date": "2025-03-31", "status": "failed"},
            {"start_date": "2025-04-01", "end_date": "2025-06-30", "status": "complete"}
        ])
        
        orchestrator = ParallelFetchOrchestrator(
            username="testuser",
            num_workers=1,
            db=mock_db
        )
        
        pending = orchestrator.get_pending_ranges()
        
        # Should include the failed range, not the complete one
        assert len(pending) == 1
        assert pending[0]["status"] == "failed"


class TestConcurrencyControl:
    """Tests for concurrent access handling."""
    
    @pytest.mark.asyncio
    async def test_database_writes_are_serialized(self):
        """Database writes don't corrupt with concurrent workers."""
        from fetcher.parallel_orchestrator import DatabaseWriteQueue
        
        write_order = []
        
        async def mock_write(data):
            write_order.append(data)
            await asyncio.sleep(0.01)  # Simulate I/O
        
        queue = DatabaseWriteQueue()
        queue.write_fn = mock_write
        
        # Simulate concurrent writes
        await asyncio.gather(
            queue.enqueue({"id": 1}),
            queue.enqueue({"id": 2}),
            queue.enqueue({"id": 3})
        )
        
        await queue.flush()
        
        # All writes should complete
        assert len(write_order) == 3
    
    @pytest.mark.asyncio
    async def test_workers_dont_get_same_range(self):
        """Two workers never process the same date range."""
        from fetcher.parallel_orchestrator import ParallelFetchOrchestrator
        
        processed_ranges = []
        lock = asyncio.Lock()
        
        async def track_range(worker, window):
            async with lock:
                # Check no overlap with already processed
                for prev in processed_ranges:
                    assert not (
                        window.start_date < prev.end_date and
                        window.end_date > prev.start_date
                    ), "Overlapping ranges detected!"
                processed_ranges.append(window)
            await asyncio.sleep(0.01)
            return {"tweets": 10}
        
        orchestrator = ParallelFetchOrchestrator(
            username="testuser",
            num_workers=3,
            db=Mock(get_collected_ranges=Mock(return_value=[]))
        )
        
        with patch.object(orchestrator, '_collect_with_worker', side_effect=track_range):
            windows = [
                Mock(start_date=date(2025, 1, 1), end_date=date(2025, 2, 1)),
                Mock(start_date=date(2025, 2, 1), end_date=date(2025, 3, 1)),
                Mock(start_date=date(2025, 3, 1), end_date=date(2025, 4, 1)),
            ]
            
            await orchestrator.collect_parallel(windows)
            
            assert len(processed_ranges) == 3


class TestOrchestratorConfig:
    """Tests for orchestrator configuration."""
    
    def test_default_config(self):
        """Default configuration values."""
        from fetcher.parallel_orchestrator import OrchestratorConfig
        
        config = OrchestratorConfig()
        
        assert config.max_workers == 3
        assert config.retry_failed_ranges is True
        assert config.max_retries == 2
    
    def test_worker_count_limited(self):
        """Worker count has reasonable upper limit."""
        from fetcher.parallel_orchestrator import OrchestratorConfig
        
        config = OrchestratorConfig(max_workers=100)
        
        # Should cap at reasonable limit
        assert config.max_workers <= 10


class TestResultAggregation:
    """Tests for aggregating results from parallel workers."""
    
    @pytest.mark.asyncio
    async def test_aggregate_tweet_counts(self):
        """Total tweets aggregated from all workers."""
        from fetcher.parallel_orchestrator import ParallelFetchOrchestrator
        
        orchestrator = ParallelFetchOrchestrator(
            username="testuser",
            num_workers=2,
            db=Mock(get_collected_ranges=Mock(return_value=[]))
        )
        
        results = [
            {"tweets": 100, "window": Mock()},
            {"tweets": 200, "window": Mock()},
            {"tweets": 150, "window": Mock()}
        ]
        
        total = orchestrator._aggregate_results(results)
        
        assert total.total_tweets == 450
        assert total.windows_completed == 3
    
    @pytest.mark.asyncio
    async def test_aggregate_includes_failures(self):
        """Aggregation tracks failed windows."""
        from fetcher.parallel_orchestrator import ParallelFetchOrchestrator
        
        orchestrator = ParallelFetchOrchestrator(
            username="testuser",
            num_workers=2,
            db=Mock(get_collected_ranges=Mock(return_value=[]))
        )
        
        results = [
            {"tweets": 100, "window": Mock(), "status": "complete"},
            {"tweets": 0, "window": Mock(), "status": "failed", "error": "timeout"},
        ]
        
        total = orchestrator._aggregate_results(results)
        
        assert total.windows_completed == 1
        assert total.windows_failed == 1


class TestGracefulShutdown:
    """Tests for graceful shutdown handling."""
    
    @pytest.mark.asyncio
    async def test_shutdown_saves_progress(self):
        """In-progress work is saved on shutdown."""
        from fetcher.parallel_orchestrator import ParallelFetchOrchestrator
        
        mock_db = Mock()
        mock_db.save_partial_progress = Mock()
        
        orchestrator = ParallelFetchOrchestrator(
            username="testuser",
            num_workers=2,
            db=mock_db
        )
        
        # Simulate work in progress
        orchestrator._in_progress = [
            {"window": Mock(start_date=date(2025, 1, 1), end_date=date(2025, 3, 31)), "tweets_so_far": 50}
        ]
        
        await orchestrator.shutdown()
        
        mock_db.save_partial_progress.assert_called()
    
    @pytest.mark.asyncio
    async def test_shutdown_closes_browsers(self):
        """All browser contexts closed on shutdown."""
        from fetcher.parallel_orchestrator import ParallelFetchOrchestrator
        
        mock_pool = AsyncMock()
        
        orchestrator = ParallelFetchOrchestrator(
            username="testuser",
            num_workers=2,
            db=Mock()
        )
        orchestrator._pool = mock_pool
        
        await orchestrator.shutdown()
        
        mock_pool.stop.assert_called_once()
