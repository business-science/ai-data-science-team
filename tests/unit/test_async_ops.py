"""
Unit tests for async and parallel execution utilities.
"""

import asyncio
import time
import pytest
import pandas as pd

from ai_data_science_team.async_ops import (
    AsyncExecutor,
    ParallelExecutor,
    TaskResult,
    ExecutionStatus,
    parallel_map,
    parallel_apply,
    gather_results,
    async_retry,
    timeout,
    rate_limit,
    batch_process,
)
from ai_data_science_team.async_ops.utils import (
    retry,
    RateLimiter,
    BatchProcessor,
    CircuitBreaker,
)


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_task_result_creation(self):
        """Test creating a TaskResult."""
        result = TaskResult(
            task_id="test-1",
            status=ExecutionStatus.COMPLETED,
            result="success",
        )

        assert result.task_id == "test-1"
        assert result.status == ExecutionStatus.COMPLETED
        assert result.result == "success"
        assert result.succeeded is True
        assert result.failed is False

    def test_task_result_failed(self):
        """Test TaskResult in failed state."""
        result = TaskResult(
            task_id="test-2",
            status=ExecutionStatus.FAILED,
            error=ValueError("test error"),
        )

        assert result.succeeded is False
        assert result.failed is True
        assert isinstance(result.error, ValueError)

    def test_task_result_duration(self):
        """Test TaskResult duration calculation."""
        result = TaskResult(
            task_id="test-3",
            status=ExecutionStatus.COMPLETED,
            start_time=100.0,
            end_time=105.5,
        )

        assert result.duration == 5.5

    def test_task_result_duration_none(self):
        """Test TaskResult duration when times not set."""
        result = TaskResult(
            task_id="test-4",
            status=ExecutionStatus.PENDING,
        )

        assert result.duration is None


class TestAsyncExecutor:
    """Tests for AsyncExecutor."""

    @pytest.mark.asyncio
    async def test_async_executor_run(self):
        """Test running a single coroutine."""
        executor = AsyncExecutor()

        async def simple_task():
            return "completed"

        result = await executor.run(simple_task)

        assert result.succeeded
        assert result.result == "completed"
        assert result.duration is not None

    @pytest.mark.asyncio
    async def test_async_executor_with_args(self):
        """Test running coroutine with arguments."""
        executor = AsyncExecutor()

        async def add(a, b):
            return a + b

        result = await executor.run(add, 2, 3)

        assert result.succeeded
        assert result.result == 5

    @pytest.mark.asyncio
    async def test_async_executor_timeout(self):
        """Test timeout handling."""
        executor = AsyncExecutor(timeout=0.1)

        async def slow_task():
            await asyncio.sleep(1.0)
            return "done"

        result = await executor.run(slow_task)

        assert result.status == ExecutionStatus.TIMEOUT
        assert result.failed

    @pytest.mark.asyncio
    async def test_async_executor_exception(self):
        """Test exception handling."""
        executor = AsyncExecutor()

        async def failing_task():
            raise ValueError("test error")

        result = await executor.run(failing_task)

        assert result.status == ExecutionStatus.FAILED
        assert isinstance(result.error, ValueError)

    @pytest.mark.asyncio
    async def test_async_executor_map(self):
        """Test mapping coroutine over items."""
        executor = AsyncExecutor()

        async def double(x):
            return x * 2

        results = await executor.map(double, [1, 2, 3, 4])

        assert len(results) == 4
        values = [r.result for r in results if r.succeeded]
        assert sorted(values) == [2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_async_executor_concurrency_limit(self):
        """Test that concurrency is limited."""
        executor = AsyncExecutor(max_concurrency=2)
        concurrent_count = 0
        max_concurrent = 0

        async def track_concurrency():
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.1)
            concurrent_count -= 1
            return True

        await executor.map(track_concurrency, range(5))

        assert max_concurrent <= 2


class TestParallelExecutor:
    """Tests for ParallelExecutor."""

    def test_parallel_executor_run(self):
        """Test running a single function."""
        with ParallelExecutor() as executor:
            result = executor.run(lambda x: x * 2, 5)

        assert result.succeeded
        assert result.result == 10

    def test_parallel_executor_map(self):
        """Test mapping function over items."""
        def square(x):
            return x ** 2

        with ParallelExecutor(max_workers=4) as executor:
            results = executor.map(square, [1, 2, 3, 4, 5])

        values = [r.result for r in results if r.succeeded]
        assert sorted(values) == [1, 4, 9, 16, 25]

    def test_parallel_executor_exception(self):
        """Test exception handling in parallel execution."""
        def failing(x):
            if x == 3:
                raise ValueError("error at 3")
            return x

        with ParallelExecutor() as executor:
            results = executor.map(failing, [1, 2, 3, 4])

        succeeded = [r for r in results if r.succeeded]
        failed = [r for r in results if r.failed]

        assert len(failed) >= 1

    def test_parallel_executor_run_batch(self):
        """Test running batch of different tasks."""
        tasks = [
            {"func": lambda: 1 + 1},
            {"func": lambda: 2 * 2},
            {"func": lambda: 3 ** 2},
        ]

        with ParallelExecutor() as executor:
            results = executor.run_batch(tasks)

        values = sorted([r.result for r in results if r.succeeded])
        assert values == [2, 4, 9]


class TestParallelMap:
    """Tests for parallel_map function."""

    def test_parallel_map_basic(self):
        """Test basic parallel mapping."""
        def process(x):
            return x * 2

        results = parallel_map(process, [1, 2, 3, 4])

        assert len(results) == 4
        values = [r.result for r in results if r.succeeded]
        assert sorted(values) == [2, 4, 6, 8]

    def test_parallel_map_with_workers(self):
        """Test parallel mapping with limited workers."""
        def slow_process(x):
            time.sleep(0.05)
            return x

        start = time.time()
        results = parallel_map(slow_process, list(range(10)), max_workers=5)
        elapsed = time.time() - start

        # With 5 workers, 10 items should take ~2 batches
        assert elapsed < 0.5  # Much faster than sequential
        assert all(r.succeeded for r in results)


class TestParallelApply:
    """Tests for parallel_apply on DataFrames."""

    def test_parallel_apply_basic(self):
        """Test basic parallel DataFrame processing."""
        df = pd.DataFrame({
            "a": range(100),
            "b": range(100, 200),
        })

        def double_values(partition):
            return partition * 2

        result = parallel_apply(df, double_values, n_partitions=4)

        assert len(result) == 100
        assert result["a"].sum() == sum(range(100)) * 2

    def test_parallel_apply_with_filter(self):
        """Test parallel apply with filtering."""
        df = pd.DataFrame({
            "value": range(100),
        })

        def filter_even(partition):
            return partition[partition["value"] % 2 == 0]

        result = parallel_apply(df, filter_even, n_partitions=4)

        assert len(result) == 50
        assert all(v % 2 == 0 for v in result["value"])


class TestGatherResults:
    """Tests for gather_results function."""

    @pytest.mark.asyncio
    async def test_gather_results_basic(self):
        """Test gathering results from coroutines."""
        async def task1():
            return 1

        async def task2():
            return 2

        results = await gather_results(task1(), task2())

        assert len(results) == 2
        values = [r.result for r in results if r.succeeded]
        assert sorted(values) == [1, 2]

    @pytest.mark.asyncio
    async def test_gather_results_with_exception(self):
        """Test gathering with exception handling."""
        async def good_task():
            return "good"

        async def bad_task():
            raise ValueError("bad")

        results = await gather_results(good_task(), bad_task(), return_exceptions=True)

        assert len(results) == 2
        succeeded = [r for r in results if r.succeeded]
        failed = [r for r in results if r.failed]

        assert len(succeeded) == 1
        assert len(failed) == 1


class TestAsyncRetry:
    """Tests for async_retry decorator."""

    @pytest.mark.asyncio
    async def test_async_retry_success(self):
        """Test retry on eventual success."""
        call_count = 0

        @async_retry(max_retries=3, delay=0.01)
        async def flaky_task():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("temporary error")
            return "success"

        result = await flaky_task()

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_async_retry_all_fail(self):
        """Test retry exhaustion."""
        call_count = 0

        @async_retry(max_retries=2, delay=0.01)
        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("permanent error")

        with pytest.raises(ValueError):
            await always_fail()

        assert call_count == 3  # Initial + 2 retries


class TestRetry:
    """Tests for sync retry decorator."""

    def test_retry_success(self):
        """Test retry on eventual success."""
        call_count = 0

        @retry(max_retries=3, delay=0.01)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("temporary")
            return "done"

        result = flaky_function()

        assert result == "done"
        assert call_count == 2

    def test_retry_all_fail(self):
        """Test retry exhaustion."""
        @retry(max_retries=2, delay=0.01)
        def always_fail():
            raise ValueError("error")

        with pytest.raises(ValueError):
            always_fail()


class TestTimeout:
    """Tests for timeout decorator."""

    @pytest.mark.asyncio
    async def test_timeout_success(self):
        """Test function completing within timeout."""
        @timeout(1.0)
        async def quick_task():
            return "done"

        result = await quick_task()
        assert result == "done"

    @pytest.mark.asyncio
    async def test_timeout_exceeded(self):
        """Test timeout exceeded."""
        @timeout(0.1)
        async def slow_task():
            await asyncio.sleep(1.0)
            return "done"

        with pytest.raises(TimeoutError):
            await slow_task()


class TestRateLimiter:
    """Tests for RateLimiter."""

    def test_rate_limiter_basic(self):
        """Test basic rate limiting."""
        limiter = RateLimiter(calls_per_second=10)

        @limiter.limit
        def limited_func():
            return True

        # First call should be immediate
        start = time.time()
        limited_func()
        first_call_time = time.time() - start

        assert first_call_time < 0.1

    def test_rate_limiter_burst(self):
        """Test burst handling."""
        limiter = RateLimiter(calls_per_second=100, burst=5)

        # Should allow burst of 5 immediately
        start = time.time()
        for _ in range(5):
            limiter.acquire()
        burst_time = time.time() - start

        assert burst_time < 0.1

    @pytest.mark.asyncio
    async def test_rate_limiter_async(self):
        """Test async rate limiting."""
        limiter = RateLimiter(calls_per_second=100)

        @limiter.limit
        async def async_limited():
            return True

        result = await async_limited()
        assert result is True


class TestBatchProcess:
    """Tests for batch processing."""

    def test_batch_process_basic(self):
        """Test basic batch processing."""
        processed_batches = []

        def process_batch(batch):
            processed_batches.append(batch)
            return len(batch)

        items = list(range(25))
        results = batch_process(items, batch_size=10, processor=process_batch)

        assert len(processed_batches) == 3  # 10 + 10 + 5
        assert sum(results) == 25

    def test_batch_process_parallel(self):
        """Test parallel batch processing."""
        def slow_process(batch):
            time.sleep(0.05)
            return sum(batch)

        items = list(range(100))
        start = time.time()
        results = batch_process(
            items,
            batch_size=25,
            processor=slow_process,
            max_workers=4,
        )
        elapsed = time.time() - start

        assert sum(results) == sum(range(100))
        assert elapsed < 0.2  # Parallel should be fast


class TestBatchProcessor:
    """Tests for BatchProcessor class."""

    @pytest.mark.asyncio
    async def test_batch_processor_auto_flush(self):
        """Test automatic batch flushing on size."""
        processed = []

        async def process(batch):
            processed.append(list(batch))

        processor = BatchProcessor(
            batch_size=3,
            max_wait=10.0,
            processor=process,
        )

        for i in range(5):
            await processor.add(i)

        assert len(processed) == 1  # First batch of 3
        assert processed[0] == [0, 1, 2]

        # Flush remaining
        await processor.flush()
        assert len(processed) == 2
        assert processed[1] == [3, 4]


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_circuit_breaker_closed(self):
        """Test circuit breaker in closed state."""
        breaker = CircuitBreaker(failure_threshold=3)

        @breaker.protect
        def working_func():
            return "success"

        result = working_func()
        assert result == "success"
        assert breaker.state == "closed"

    def test_circuit_breaker_opens(self):
        """Test circuit breaker opening after failures."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)

        @breaker.protect
        def failing_func():
            raise ValueError("error")

        # Cause failures to open circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                failing_func()

        assert breaker.state == "open"

        # Should raise without calling function
        with pytest.raises(RuntimeError, match="Circuit breaker is open"):
            failing_func()

    def test_circuit_breaker_half_open(self):
        """Test circuit breaker half-open state."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,
            half_open_calls=1,
        )

        call_count = 0

        @breaker.protect
        def controlled_func():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError("error")
            return "recovered"

        # Open the circuit
        for _ in range(2):
            try:
                controlled_func()
            except ValueError:
                pass

        assert breaker.state == "open"

        # Wait for recovery timeout
        time.sleep(0.15)
        assert breaker.state == "half-open"

        # Successful call should close circuit
        result = controlled_func()
        assert result == "recovered"
        assert breaker.state == "closed"

    def test_circuit_breaker_reset(self):
        """Test manual circuit breaker reset."""
        breaker = CircuitBreaker(failure_threshold=1)

        @breaker.protect
        def failing():
            raise ValueError("error")

        try:
            failing()
        except ValueError:
            pass

        assert breaker.state == "open"

        breaker.reset()
        assert breaker.state == "closed"

    @pytest.mark.asyncio
    async def test_circuit_breaker_async(self):
        """Test circuit breaker with async functions."""
        breaker = CircuitBreaker(failure_threshold=2)

        @breaker.protect
        async def async_func():
            return "async success"

        result = await async_func()
        assert result == "async success"
