"""
Parallel operations for data science workflows.

This module provides high-level functions for running operations
in parallel, including agent execution and data processing.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from ai_data_science_team.async_ops.executor import (
    AsyncExecutor,
    ParallelExecutor,
    TaskResult,
    ExecutionStatus,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def parallel_map(
    func: Callable[[T], Any],
    items: List[T],
    max_workers: Optional[int] = None,
    timeout: Optional[float] = None,
    use_processes: bool = False,
) -> List[TaskResult]:
    """
    Apply a function to items in parallel.

    Parameters
    ----------
    func : callable
        Function to apply to each item.
    items : list
        Items to process.
    max_workers : int, optional
        Maximum number of parallel workers.
    timeout : float, optional
        Timeout for all operations.
    use_processes : bool, default False
        Use processes instead of threads for CPU-bound operations.

    Returns
    -------
    list of TaskResult
        Results for each item, preserving order.

    Example
    -------
    >>> def process(x):
    ...     return x * 2
    >>> results = parallel_map(process, [1, 2, 3, 4])
    >>> values = [r.result for r in results if r.succeeded]
    """
    with ParallelExecutor(
        max_workers=max_workers,
        use_processes=use_processes,
        timeout=timeout,
    ) as executor:
        return executor.map(func, items, timeout=timeout)


def parallel_apply(
    dataframe,
    func: Callable,
    axis: int = 0,
    n_partitions: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> Any:
    """
    Apply a function to a DataFrame in parallel.

    Splits the DataFrame into partitions and processes them in parallel.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame to process.
    func : callable
        Function to apply. Should accept a DataFrame and return a DataFrame.
    axis : int, default 0
        Axis along which to split (0 for rows, 1 for columns).
    n_partitions : int, optional
        Number of partitions. Defaults to number of workers.
    max_workers : int, optional
        Maximum number of workers.

    Returns
    -------
    pandas.DataFrame
        Combined result from all partitions.

    Example
    -------
    >>> def clean_partition(df):
    ...     return df.dropna()
    >>> result = parallel_apply(large_df, clean_partition, n_partitions=4)
    """
    import pandas as pd
    import numpy as np

    # Determine number of partitions
    if n_partitions is None:
        import os
        n_partitions = max_workers or os.cpu_count() or 4

    # Split DataFrame
    if axis == 0:
        partitions = np.array_split(dataframe, n_partitions)
    else:
        partitions = [
            dataframe.iloc[:, i::n_partitions]
            for i in range(n_partitions)
        ]

    # Process in parallel
    results = parallel_map(func, partitions, max_workers=max_workers)

    # Combine results
    successful_results = [r.result for r in results if r.succeeded]
    if not successful_results:
        raise RuntimeError("All partitions failed to process")

    if axis == 0:
        return pd.concat(successful_results, ignore_index=True)
    else:
        return pd.concat(successful_results, axis=1)


async def run_agents_parallel(
    agents: List[Any],
    inputs: Optional[List[Dict[str, Any]]] = None,
    shared_input: Optional[Dict[str, Any]] = None,
    max_concurrency: int = 5,
    timeout: Optional[float] = None,
) -> List[TaskResult]:
    """
    Run multiple AI agents in parallel.

    Parameters
    ----------
    agents : list
        List of agent instances with an `invoke` or `run` method.
    inputs : list of dict, optional
        Input for each agent (must match length of agents).
    shared_input : dict, optional
        Shared input for all agents (used if inputs is None).
    max_concurrency : int, default 5
        Maximum number of agents running concurrently.
    timeout : float, optional
        Timeout per agent in seconds.

    Returns
    -------
    list of TaskResult
        Results from each agent.

    Example
    -------
    >>> from ai_data_science_team import DataCleaningAgent, DataWranglingAgent
    >>>
    >>> agents = [DataCleaningAgent(model=llm), DataWranglingAgent(model=llm)]
    >>> results = await run_agents_parallel(
    ...     agents,
    ...     shared_input={"data": df, "instructions": "process this"},
    ... )
    """
    if inputs is None:
        if shared_input is None:
            raise ValueError("Either inputs or shared_input must be provided")
        inputs = [shared_input] * len(agents)

    if len(inputs) != len(agents):
        raise ValueError(
            f"Number of inputs ({len(inputs)}) must match number of agents ({len(agents)})"
        )

    async def run_agent(agent, input_data):
        """Run a single agent."""
        # Check for different agent interfaces
        if hasattr(agent, "ainvoke"):
            return await agent.ainvoke(input_data)
        elif hasattr(agent, "invoke"):
            # Run sync invoke in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, agent.invoke, input_data)
        elif hasattr(agent, "run"):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, agent.run, input_data)
        else:
            raise TypeError(f"Agent {type(agent)} has no invoke or run method")

    executor = AsyncExecutor(max_concurrency=max_concurrency, timeout=timeout)
    tasks = [
        executor.run(run_agent, agent, input_data, task_id=f"agent-{i}")
        for i, (agent, input_data) in enumerate(zip(agents, inputs))
    ]

    return await asyncio.gather(*tasks)


def run_agents_parallel_sync(
    agents: List[Any],
    inputs: Optional[List[Dict[str, Any]]] = None,
    shared_input: Optional[Dict[str, Any]] = None,
    max_workers: int = 5,
    timeout: Optional[float] = None,
) -> List[TaskResult]:
    """
    Run multiple AI agents in parallel (synchronous version).

    Parameters
    ----------
    agents : list
        List of agent instances.
    inputs : list of dict, optional
        Input for each agent.
    shared_input : dict, optional
        Shared input for all agents.
    max_workers : int, default 5
        Maximum number of workers.
    timeout : float, optional
        Timeout per agent.

    Returns
    -------
    list of TaskResult
        Results from each agent.
    """
    if inputs is None:
        if shared_input is None:
            raise ValueError("Either inputs or shared_input must be provided")
        inputs = [shared_input] * len(agents)

    if len(inputs) != len(agents):
        raise ValueError("Number of inputs must match number of agents")

    def run_agent(args):
        agent, input_data = args
        if hasattr(agent, "invoke"):
            return agent.invoke(input_data)
        elif hasattr(agent, "run"):
            return agent.run(input_data)
        else:
            raise TypeError(f"Agent {type(agent)} has no invoke or run method")

    with ParallelExecutor(max_workers=max_workers, timeout=timeout) as executor:
        return executor.map(run_agent, list(zip(agents, inputs)), timeout=timeout)


async def gather_results(
    *coros,
    timeout: Optional[float] = None,
    return_exceptions: bool = True,
) -> List[TaskResult]:
    """
    Gather results from multiple coroutines.

    Parameters
    ----------
    *coros
        Coroutines to execute.
    timeout : float, optional
        Timeout for all operations.
    return_exceptions : bool, default True
        Return exceptions as results instead of raising.

    Returns
    -------
    list of TaskResult
        Results from each coroutine.

    Example
    -------
    >>> async def task1():
    ...     return await some_operation()
    >>> async def task2():
    ...     return await another_operation()
    >>>
    >>> results = await gather_results(task1(), task2())
    """
    async def wrap_coro(coro, index):
        result = TaskResult(
            task_id=f"gather-{index}",
            status=ExecutionStatus.RUNNING,
        )
        import time
        result.start_time = time.time()

        try:
            if timeout:
                value = await asyncio.wait_for(coro, timeout=timeout)
            else:
                value = await coro
            result.result = value
            result.status = ExecutionStatus.COMPLETED
        except asyncio.TimeoutError:
            result.status = ExecutionStatus.TIMEOUT
            result.error = TimeoutError(f"Operation timed out after {timeout}s")
        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error = e
            if not return_exceptions:
                raise
        finally:
            result.end_time = time.time()

        return result

    wrapped = [wrap_coro(coro, i) for i, coro in enumerate(coros)]
    return await asyncio.gather(*wrapped, return_exceptions=return_exceptions)


def run_pipeline_parallel(
    steps: List[Dict[str, Any]],
    initial_data: Any,
    max_parallel_branches: int = 4,
) -> Dict[str, TaskResult]:
    """
    Run a data pipeline with parallel branches.

    Parameters
    ----------
    steps : list of dict
        Pipeline steps. Each step has:
        - 'name': Step name
        - 'func': Function to execute
        - 'depends_on': List of step names this depends on (optional)
    initial_data : Any
        Initial data to pass to steps with no dependencies.
    max_parallel_branches : int, default 4
        Maximum parallel branches.

    Returns
    -------
    dict
        Results keyed by step name.

    Example
    -------
    >>> steps = [
    ...     {'name': 'load', 'func': load_data},
    ...     {'name': 'clean', 'func': clean_data, 'depends_on': ['load']},
    ...     {'name': 'analyze', 'func': analyze, 'depends_on': ['clean']},
    ...     {'name': 'visualize', 'func': plot, 'depends_on': ['clean']},
    ... ]
    >>> results = run_pipeline_parallel(steps, "data.csv")
    """
    import time
    from collections import defaultdict

    # Build dependency graph
    step_map = {s["name"]: s for s in steps}
    dependents = defaultdict(list)
    dependencies = {}

    for step in steps:
        name = step["name"]
        deps = step.get("depends_on", [])
        dependencies[name] = set(deps)
        for dep in deps:
            dependents[dep].append(name)

    # Track results and completion
    results: Dict[str, TaskResult] = {}
    completed = set()

    def get_ready_steps():
        """Get steps whose dependencies are all completed."""
        ready = []
        for step in steps:
            name = step["name"]
            if name not in completed:
                if all(dep in completed for dep in dependencies[name]):
                    ready.append(step)
        return ready

    def run_step(step):
        """Execute a single step."""
        name = step["name"]
        func = step["func"]

        result = TaskResult(
            task_id=name,
            status=ExecutionStatus.RUNNING,
            start_time=time.time(),
        )

        try:
            # Get input data
            deps = dependencies[name]
            if not deps:
                input_data = initial_data
            elif len(deps) == 1:
                dep_result = results[list(deps)[0]]
                input_data = dep_result.result
            else:
                # Multiple dependencies - pass as dict
                input_data = {
                    dep: results[dep].result
                    for dep in deps
                }

            # Execute
            value = func(input_data)
            result.result = value
            result.status = ExecutionStatus.COMPLETED

        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error = e
            logger.error(f"Step {name} failed: {e}")

        finally:
            result.end_time = time.time()

        return result

    # Execute pipeline
    with ThreadPoolExecutor(max_workers=max_parallel_branches) as executor:
        while len(completed) < len(steps):
            ready = get_ready_steps()
            if not ready:
                # Check for deadlock
                remaining = set(s["name"] for s in steps) - completed
                raise RuntimeError(f"Pipeline deadlock. Remaining: {remaining}")

            # Submit ready steps
            futures = {
                executor.submit(run_step, step): step["name"]
                for step in ready
            }

            # Wait for completion
            for future in as_completed(futures):
                name = futures[future]
                result = future.result()
                results[name] = result
                completed.add(name)

                if result.failed:
                    logger.warning(f"Step {name} failed, dependent steps may fail")

    return results
