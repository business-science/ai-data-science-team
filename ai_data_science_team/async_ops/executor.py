"""
Async and parallel execution engines.

This module provides executors for running tasks concurrently
using threads, processes, or async/await patterns.
"""

import asyncio
import logging
import time
import uuid
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    Future,
    as_completed,
)
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
    Union,
)
from threading import Lock
import traceback

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ExecutionStatus(Enum):
    """Status of a task execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class TaskResult(Generic[T]):
    """
    Result of a task execution.

    Attributes
    ----------
    task_id : str
        Unique identifier for the task.
    status : ExecutionStatus
        Current status of the task.
    result : T, optional
        The result value if successful.
    error : Exception, optional
        The exception if failed.
    start_time : float, optional
        Unix timestamp when task started.
    end_time : float, optional
        Unix timestamp when task ended.
    duration : float, optional
        Execution duration in seconds.
    """
    task_id: str
    status: ExecutionStatus
    result: Optional[T] = None
    error: Optional[Exception] = None
    error_traceback: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Calculate task duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def succeeded(self) -> bool:
        """Check if task completed successfully."""
        return self.status == ExecutionStatus.COMPLETED

    @property
    def failed(self) -> bool:
        """Check if task failed."""
        return self.status in (ExecutionStatus.FAILED, ExecutionStatus.TIMEOUT)


class AsyncExecutor:
    """
    Async executor for running coroutines concurrently.

    This executor uses asyncio for concurrent execution of async tasks.
    Ideal for I/O-bound operations like API calls.

    Parameters
    ----------
    max_concurrency : int, default 10
        Maximum number of concurrent tasks.
    timeout : float, optional
        Default timeout for tasks in seconds.

    Example
    -------
    >>> async def fetch_data(url):
    ...     # async operation
    ...     return data
    >>>
    >>> executor = AsyncExecutor(max_concurrency=5)
    >>> urls = ["http://example.com/1", "http://example.com/2"]
    >>> results = await executor.map(fetch_data, urls)
    """

    def __init__(
        self,
        max_concurrency: int = 10,
        timeout: Optional[float] = None,
    ):
        self.max_concurrency = max_concurrency
        self.default_timeout = timeout
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._results: Dict[str, TaskResult] = {}
        self._lock = Lock()

    async def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create the semaphore for concurrency control."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore

    async def run(
        self,
        coro: Callable[..., Any],
        *args,
        timeout: Optional[float] = None,
        task_id: Optional[str] = None,
        **kwargs,
    ) -> TaskResult:
        """
        Run a single coroutine.

        Parameters
        ----------
        coro : callable
            Async function to execute.
        *args
            Positional arguments for the coroutine.
        timeout : float, optional
            Timeout in seconds.
        task_id : str, optional
            Custom task ID.
        **kwargs
            Keyword arguments for the coroutine.

        Returns
        -------
        TaskResult
            Result of the execution.
        """
        task_id = task_id or str(uuid.uuid4())[:8]
        timeout = timeout or self.default_timeout
        semaphore = await self._get_semaphore()

        result = TaskResult(
            task_id=task_id,
            status=ExecutionStatus.PENDING,
        )

        async with semaphore:
            result.status = ExecutionStatus.RUNNING
            result.start_time = time.time()

            try:
                if timeout:
                    value = await asyncio.wait_for(
                        coro(*args, **kwargs),
                        timeout=timeout,
                    )
                else:
                    value = await coro(*args, **kwargs)

                result.result = value
                result.status = ExecutionStatus.COMPLETED

            except asyncio.TimeoutError:
                result.status = ExecutionStatus.TIMEOUT
                result.error = TimeoutError(f"Task timed out after {timeout}s")
                result.error_traceback = traceback.format_exc()

            except asyncio.CancelledError:
                result.status = ExecutionStatus.CANCELLED
                result.error = Exception("Task was cancelled")

            except Exception as e:
                result.status = ExecutionStatus.FAILED
                result.error = e
                result.error_traceback = traceback.format_exc()
                logger.error(f"Task {task_id} failed: {e}")

            finally:
                result.end_time = time.time()

        with self._lock:
            self._results[task_id] = result

        return result

    async def map(
        self,
        coro: Callable[..., Any],
        items: List[Any],
        timeout: Optional[float] = None,
    ) -> List[TaskResult]:
        """
        Apply a coroutine to multiple items concurrently.

        Parameters
        ----------
        coro : callable
            Async function to apply.
        items : list
            Items to process.
        timeout : float, optional
            Timeout per item.

        Returns
        -------
        list of TaskResult
            Results for each item.
        """
        tasks = [
            self.run(coro, item, timeout=timeout, task_id=f"map-{i}")
            for i, item in enumerate(items)
        ]
        return await asyncio.gather(*tasks)

    async def gather(
        self,
        coros: List[Callable[..., Any]],
        timeout: Optional[float] = None,
    ) -> List[TaskResult]:
        """
        Run multiple coroutines concurrently.

        Parameters
        ----------
        coros : list of callables
            Coroutines to execute.
        timeout : float, optional
            Timeout for all tasks.

        Returns
        -------
        list of TaskResult
            Results for each coroutine.
        """
        tasks = [
            self.run(coro, timeout=timeout, task_id=f"gather-{i}")
            for i, coro in enumerate(coros)
        ]
        return await asyncio.gather(*tasks)

    def get_results(self) -> Dict[str, TaskResult]:
        """Get all stored results."""
        with self._lock:
            return dict(self._results)

    def clear_results(self) -> None:
        """Clear stored results."""
        with self._lock:
            self._results.clear()


class ParallelExecutor:
    """
    Parallel executor using thread or process pools.

    This executor uses concurrent.futures for parallel execution.
    Use threads for I/O-bound tasks and processes for CPU-bound tasks.

    Parameters
    ----------
    max_workers : int, optional
        Maximum number of workers. Defaults to CPU count.
    use_processes : bool, default False
        Use processes instead of threads.
    timeout : float, optional
        Default timeout for tasks.

    Example
    -------
    >>> def process_data(df):
    ...     return df.describe()
    >>>
    >>> executor = ParallelExecutor(max_workers=4)
    >>> dataframes = [df1, df2, df3, df4]
    >>> results = executor.map(process_data, dataframes)
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_processes: bool = False,
        timeout: Optional[float] = None,
    ):
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.default_timeout = timeout
        self._executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]] = None
        self._results: Dict[str, TaskResult] = {}
        self._lock = Lock()

    def _get_executor(self) -> Union[ThreadPoolExecutor, ProcessPoolExecutor]:
        """Get or create the executor."""
        if self._executor is None:
            if self.use_processes:
                self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
            else:
                self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self._executor

    def submit(
        self,
        func: Callable[..., T],
        *args,
        task_id: Optional[str] = None,
        **kwargs,
    ) -> Future:
        """
        Submit a task for execution.

        Parameters
        ----------
        func : callable
            Function to execute.
        *args
            Positional arguments.
        task_id : str, optional
            Custom task ID.
        **kwargs
            Keyword arguments.

        Returns
        -------
        Future
            Future object representing the execution.
        """
        executor = self._get_executor()
        task_id = task_id or str(uuid.uuid4())[:8]

        def wrapper():
            result = TaskResult(
                task_id=task_id,
                status=ExecutionStatus.RUNNING,
                start_time=time.time(),
            )
            try:
                value = func(*args, **kwargs)
                result.result = value
                result.status = ExecutionStatus.COMPLETED
            except Exception as e:
                result.status = ExecutionStatus.FAILED
                result.error = e
                result.error_traceback = traceback.format_exc()
                logger.error(f"Task {task_id} failed: {e}")
            finally:
                result.end_time = time.time()

            with self._lock:
                self._results[task_id] = result

            return result

        return executor.submit(wrapper)

    def map(
        self,
        func: Callable[..., T],
        items: List[Any],
        timeout: Optional[float] = None,
    ) -> List[TaskResult]:
        """
        Apply a function to multiple items in parallel.

        Parameters
        ----------
        func : callable
            Function to apply.
        items : list
            Items to process.
        timeout : float, optional
            Timeout for all tasks.

        Returns
        -------
        list of TaskResult
            Results for each item.
        """
        timeout = timeout or self.default_timeout
        futures = [
            self.submit(func, item, task_id=f"map-{i}")
            for i, item in enumerate(items)
        ]

        results = []
        for future in as_completed(futures, timeout=timeout):
            try:
                results.append(future.result())
            except Exception as e:
                # Handle timeout or other errors
                result = TaskResult(
                    task_id="unknown",
                    status=ExecutionStatus.FAILED,
                    error=e,
                    error_traceback=traceback.format_exc(),
                )
                results.append(result)

        return results

    def run(
        self,
        func: Callable[..., T],
        *args,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> TaskResult:
        """
        Run a single function and wait for result.

        Parameters
        ----------
        func : callable
            Function to execute.
        *args
            Positional arguments.
        timeout : float, optional
            Timeout in seconds.
        **kwargs
            Keyword arguments.

        Returns
        -------
        TaskResult
            Result of the execution.
        """
        timeout = timeout or self.default_timeout
        future = self.submit(func, *args, **kwargs)
        return future.result(timeout=timeout)

    def run_batch(
        self,
        tasks: List[Dict[str, Any]],
        timeout: Optional[float] = None,
    ) -> List[TaskResult]:
        """
        Run a batch of tasks with different functions and arguments.

        Parameters
        ----------
        tasks : list of dict
            Each dict contains 'func', 'args' (optional), 'kwargs' (optional).
        timeout : float, optional
            Timeout for all tasks.

        Returns
        -------
        list of TaskResult
            Results for each task.
        """
        futures = []
        for i, task in enumerate(tasks):
            func = task["func"]
            args = task.get("args", ())
            kwargs = task.get("kwargs", {})
            futures.append(self.submit(func, *args, task_id=f"batch-{i}", **kwargs))

        timeout = timeout or self.default_timeout
        results = []
        for future in as_completed(futures, timeout=timeout):
            results.append(future.result())

        return results

    def get_results(self) -> Dict[str, TaskResult]:
        """Get all stored results."""
        with self._lock:
            return dict(self._results)

    def clear_results(self) -> None:
        """Clear stored results."""
        with self._lock:
            self._results.clear()

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the executor.

        Parameters
        ----------
        wait : bool, default True
            Wait for all tasks to complete.
        """
        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)
        return False
