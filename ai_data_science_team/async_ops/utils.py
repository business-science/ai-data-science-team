"""
Async utilities for robust execution.

This module provides decorators and utilities for handling
retries, timeouts, rate limiting, and batch processing.
"""

import asyncio
import functools
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


def async_retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """
    Decorator for retrying async functions on failure.

    Parameters
    ----------
    max_retries : int, default 3
        Maximum number of retry attempts.
    delay : float, default 1.0
        Initial delay between retries in seconds.
    backoff : float, default 2.0
        Multiplier for delay after each retry.
    exceptions : tuple, default (Exception,)
        Exception types to catch and retry.
    on_retry : callable, optional
        Callback called on each retry with (exception, attempt_number).

    Returns
    -------
    callable
        Decorated function.

    Example
    -------
    >>> @async_retry(max_retries=3, delay=1.0)
    ... async def fetch_data(url):
    ...     response = await http_client.get(url)
    ...     return response.json()
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        if on_retry:
                            on_retry(e, attempt + 1)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for "
                            f"{func.__name__}: {e}. Retrying in {current_delay}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}"
                        )

            raise last_exception

        return wrapper
    return decorator


def retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """
    Decorator for retrying synchronous functions on failure.

    Parameters
    ----------
    max_retries : int, default 3
        Maximum number of retry attempts.
    delay : float, default 1.0
        Initial delay between retries in seconds.
    backoff : float, default 2.0
        Multiplier for delay after each retry.
    exceptions : tuple, default (Exception,)
        Exception types to catch and retry.
    on_retry : callable, optional
        Callback called on each retry.

    Example
    -------
    >>> @retry(max_retries=3)
    ... def call_api(data):
    ...     return api_client.post(data)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        if on_retry:
                            on_retry(e, attempt + 1)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff

            raise last_exception

        return wrapper
    return decorator


def timeout(seconds: float):
    """
    Decorator to add timeout to async functions.

    Parameters
    ----------
    seconds : float
        Timeout in seconds.

    Returns
    -------
    callable
        Decorated function.

    Example
    -------
    >>> @timeout(30.0)
    ... async def long_operation():
    ...     await some_long_task()
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds,
                )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Function {func.__name__} timed out after {seconds}s"
                )
        return wrapper
    return decorator


@dataclass
class RateLimiter:
    """
    Rate limiter for controlling request frequency.

    Parameters
    ----------
    calls_per_second : float
        Maximum calls per second.
    burst : int, optional
        Maximum burst size. Defaults to calls_per_second.

    Example
    -------
    >>> limiter = RateLimiter(calls_per_second=10)
    >>>
    >>> @limiter.limit
    ... async def api_call():
    ...     return await fetch_data()
    """
    calls_per_second: float
    burst: Optional[int] = None
    _tokens: float = field(default=0, init=False)
    _last_update: float = field(default_factory=time.time, init=False)
    _lock: Lock = field(default_factory=Lock, init=False)

    def __post_init__(self):
        if self.burst is None:
            self.burst = int(self.calls_per_second)
        self._tokens = float(self.burst)

    def _refill(self) -> None:
        """Refill tokens based on time elapsed."""
        now = time.time()
        elapsed = now - self._last_update
        self._tokens = min(
            self.burst,
            self._tokens + elapsed * self.calls_per_second,
        )
        self._last_update = now

    def acquire(self) -> float:
        """
        Acquire a token, blocking if necessary.

        Returns
        -------
        float
            Time waited in seconds.
        """
        with self._lock:
            self._refill()
            wait_time = 0.0

            if self._tokens < 1:
                wait_time = (1 - self._tokens) / self.calls_per_second
                time.sleep(wait_time)
                self._refill()

            self._tokens -= 1
            return wait_time

    async def acquire_async(self) -> float:
        """
        Acquire a token asynchronously.

        Returns
        -------
        float
            Time waited in seconds.
        """
        with self._lock:
            self._refill()
            wait_time = 0.0

            if self._tokens < 1:
                wait_time = (1 - self._tokens) / self.calls_per_second

        if wait_time > 0:
            await asyncio.sleep(wait_time)
            with self._lock:
                self._refill()

        with self._lock:
            self._tokens -= 1
            return wait_time

    def limit(self, func: Callable) -> Callable:
        """
        Decorator to rate limit a function.

        Works with both sync and async functions.
        """
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                await self.acquire_async()
                return await func(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                self.acquire()
                return func(*args, **kwargs)
            return sync_wrapper


def rate_limit(calls_per_second: float, burst: Optional[int] = None):
    """
    Decorator factory for rate limiting.

    Parameters
    ----------
    calls_per_second : float
        Maximum calls per second.
    burst : int, optional
        Maximum burst size.

    Returns
    -------
    callable
        Decorator function.

    Example
    -------
    >>> @rate_limit(10)  # 10 calls per second
    ... async def api_call():
    ...     return await fetch_data()
    """
    limiter = RateLimiter(calls_per_second=calls_per_second, burst=burst)
    return limiter.limit


@dataclass
class BatchProcessor:
    """
    Process items in batches with configurable size and timing.

    Parameters
    ----------
    batch_size : int
        Maximum items per batch.
    max_wait : float
        Maximum time to wait for batch to fill (seconds).
    processor : callable
        Function to process each batch.

    Example
    -------
    >>> async def process_batch(items):
    ...     await db.insert_many(items)
    >>>
    >>> processor = BatchProcessor(batch_size=100, max_wait=1.0, processor=process_batch)
    >>> for item in items:
    ...     await processor.add(item)
    >>> await processor.flush()
    """
    batch_size: int
    max_wait: float
    processor: Callable[[List[Any]], Any]
    _buffer: List[Any] = field(default_factory=list, init=False)
    _last_flush: float = field(default_factory=time.time, init=False)
    _lock: Lock = field(default_factory=Lock, init=False)

    async def add(self, item: Any) -> Optional[Any]:
        """
        Add an item to the batch.

        Returns batch result if batch was processed.
        """
        result = None
        should_flush = False

        with self._lock:
            self._buffer.append(item)
            if len(self._buffer) >= self.batch_size:
                should_flush = True
            elif time.time() - self._last_flush >= self.max_wait:
                should_flush = True

        if should_flush:
            result = await self.flush()

        return result

    async def flush(self) -> Optional[Any]:
        """Process and clear the current batch."""
        with self._lock:
            if not self._buffer:
                return None
            batch = self._buffer.copy()
            self._buffer.clear()
            self._last_flush = time.time()

        if asyncio.iscoroutinefunction(self.processor):
            return await self.processor(batch)
        else:
            return self.processor(batch)

    def add_sync(self, item: Any) -> Optional[Any]:
        """Synchronous version of add."""
        result = None
        should_flush = False

        with self._lock:
            self._buffer.append(item)
            if len(self._buffer) >= self.batch_size:
                should_flush = True

        if should_flush:
            result = self.flush_sync()

        return result

    def flush_sync(self) -> Optional[Any]:
        """Synchronous version of flush."""
        with self._lock:
            if not self._buffer:
                return None
            batch = self._buffer.copy()
            self._buffer.clear()
            self._last_flush = time.time()

        return self.processor(batch)


def batch_process(
    items: List[T],
    batch_size: int,
    processor: Callable[[List[T]], Any],
    max_workers: int = 4,
) -> List[Any]:
    """
    Process items in batches using parallel workers.

    Parameters
    ----------
    items : list
        Items to process.
    batch_size : int
        Size of each batch.
    processor : callable
        Function to process each batch.
    max_workers : int, default 4
        Number of parallel workers.

    Returns
    -------
    list
        Results from each batch.

    Example
    -------
    >>> def insert_batch(records):
    ...     db.insert_many(records)
    >>>
    >>> results = batch_process(all_records, batch_size=100, processor=insert_batch)
    """
    from concurrent.futures import ThreadPoolExecutor

    # Create batches
    batches = [
        items[i:i + batch_size]
        for i in range(0, len(items), batch_size)
    ]

    # Process in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(processor, batch) for batch in batches]
        for future in futures:
            results.append(future.result())

    return results


async def batch_process_async(
    items: List[T],
    batch_size: int,
    processor: Callable[[List[T]], Any],
    max_concurrency: int = 4,
) -> List[Any]:
    """
    Process items in batches asynchronously.

    Parameters
    ----------
    items : list
        Items to process.
    batch_size : int
        Size of each batch.
    processor : callable
        Async function to process each batch.
    max_concurrency : int, default 4
        Maximum concurrent batches.

    Returns
    -------
    list
        Results from each batch.
    """
    # Create batches
    batches = [
        items[i:i + batch_size]
        for i in range(0, len(items), batch_size)
    ]

    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_with_semaphore(batch):
        async with semaphore:
            if asyncio.iscoroutinefunction(processor):
                return await processor(batch)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, processor, batch)

    tasks = [process_with_semaphore(batch) for batch in batches]
    return await asyncio.gather(*tasks)


class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance.

    Prevents repeated calls to failing services by "opening" the circuit
    after a threshold of failures.

    Parameters
    ----------
    failure_threshold : int, default 5
        Number of failures before opening circuit.
    recovery_timeout : float, default 30.0
        Time in seconds before attempting recovery.
    half_open_calls : int, default 1
        Number of calls to allow in half-open state.

    Example
    -------
    >>> breaker = CircuitBreaker(failure_threshold=5)
    >>>
    >>> @breaker.protect
    ... async def call_service():
    ...     return await external_api.call()
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_calls: int = 1,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_calls = half_open_calls

        self._failures = 0
        self._last_failure_time: Optional[float] = None
        self._state = "closed"  # closed, open, half-open
        self._half_open_count = 0
        self._lock = Lock()

    @property
    def state(self) -> str:
        """Get current circuit state."""
        with self._lock:
            self._check_recovery()
            return self._state

    def _check_recovery(self) -> None:
        """Check if circuit should transition from open to half-open."""
        if self._state == "open" and self._last_failure_time:
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = "half-open"
                self._half_open_count = 0

    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == "half-open":
                self._half_open_count += 1
                if self._half_open_count >= self.half_open_calls:
                    # Recovery successful
                    self._state = "closed"
                    self._failures = 0
            elif self._state == "closed":
                self._failures = 0

    def _record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failures += 1
            self._last_failure_time = time.time()

            if self._state == "half-open":
                # Recovery failed
                self._state = "open"
            elif self._failures >= self.failure_threshold:
                self._state = "open"

    def protect(self, func: Callable) -> Callable:
        """
        Decorator to protect a function with circuit breaker.

        Works with both sync and async functions.
        """
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                state = self.state
                if state == "open":
                    raise RuntimeError("Circuit breaker is open")

                try:
                    result = await func(*args, **kwargs)
                    self._record_success()
                    return result
                except Exception as e:
                    self._record_failure()
                    raise

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                state = self.state
                if state == "open":
                    raise RuntimeError("Circuit breaker is open")

                try:
                    result = func(*args, **kwargs)
                    self._record_success()
                    return result
                except Exception as e:
                    self._record_failure()
                    raise

            return sync_wrapper

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        with self._lock:
            self._state = "closed"
            self._failures = 0
            self._last_failure_time = None
            self._half_open_count = 0
