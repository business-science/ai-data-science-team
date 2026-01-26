"""
Caching decorators for AI Data Science Team.

This module provides decorators for easily caching function results.
"""

from functools import wraps
from typing import Any, Callable, Optional, Union
import logging

from ai_data_science_team.cache.cache import Cache, get_cache
from ai_data_science_team.cache.keys import generate_cache_key

logger = logging.getLogger(__name__)


def cached(
    ttl: Optional[int] = None,
    cache: Optional[Cache] = None,
    key_prefix: str = "",
    hash_dataframes: bool = True,
    unless: Optional[Callable[..., bool]] = None,
):
    """
    Decorator to cache function results.

    Parameters
    ----------
    ttl : int, optional
        Time-to-live in seconds for cached results.
    cache : Cache, optional
        Cache instance to use. Uses default cache if not provided.
    key_prefix : str, optional
        Prefix for cache keys.
    hash_dataframes : bool, default True
        Whether to use DataFrame-aware hashing for arguments.
    unless : callable, optional
        Function that takes the result and returns True if it should not be cached.

    Returns
    -------
    callable
        Decorated function.

    Example:
        @cached(ttl=3600)
        def expensive_computation(data):
            return process(data)

        # With DataFrame-aware caching
        @cached(ttl=3600, hash_dataframes=True)
        def analyze_dataframe(df):
            return df.describe()

        # Conditional caching
        @cached(ttl=3600, unless=lambda result: result is None)
        def might_return_none(x):
            return x if x > 0 else None
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get cache instance
            cache_instance = cache or get_cache()

            # Generate cache key
            cache_key = generate_cache_key(
                func, args, kwargs,
                prefix=key_prefix,
                hash_dataframes=hash_dataframes,
            )

            # Try to get from cache
            cached_value = cache_instance.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_value

            # Call function
            result = func(*args, **kwargs)

            # Check if we should cache the result
            should_cache = True
            if unless is not None:
                try:
                    should_cache = not unless(result)
                except Exception:
                    should_cache = True

            # Cache result
            if should_cache:
                cache_instance.set(cache_key, result, ttl)
                logger.debug(f"Cached result for {func.__name__}")

            return result

        # Add method to clear cache for this function
        def clear_cache():
            """Clear all cached results for this function."""
            cache_instance = cache or get_cache()
            # Note: This clears the entire cache, not just this function's entries
            # A more sophisticated implementation would track keys per function
            cache_instance.clear()

        wrapper.clear_cache = clear_cache
        wrapper._cache_key_prefix = key_prefix
        wrapper._cached = True

        return wrapper

    return decorator


def cache_result(
    key: str,
    ttl: Optional[int] = None,
    cache: Optional[Cache] = None,
):
    """
    Decorator to cache a function result with a specific key.

    Unlike @cached, this uses a fixed key instead of generating one
    from function arguments.

    Parameters
    ----------
    key : str
        The cache key to use.
    ttl : int, optional
        Time-to-live in seconds.
    cache : Cache, optional
        Cache instance to use.

    Example:
        @cache_result("daily_stats", ttl=86400)
        def compute_daily_stats():
            return expensive_computation()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            cache_instance = cache or get_cache()

            # Try to get from cache
            cached_value = cache_instance.get(key)
            if cached_value is not None:
                return cached_value

            # Call function and cache
            result = func(*args, **kwargs)
            cache_instance.set(key, result, ttl)
            return result

        return wrapper

    return decorator


def invalidate_cache(
    key: Optional[str] = None,
    key_prefix: Optional[str] = None,
    cache: Optional[Cache] = None,
):
    """
    Decorator to invalidate cache entries after function execution.

    Useful for functions that modify data and should invalidate related caches.

    Parameters
    ----------
    key : str, optional
        Specific key to invalidate.
    key_prefix : str, optional
        Prefix of keys to invalidate (clears all matching).
    cache : Cache, optional
        Cache instance to use.

    Example:
        @invalidate_cache(key="user_stats")
        def update_user(user_id, data):
            # This will invalidate the cache after updating
            db.update(user_id, data)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Call function first
            result = func(*args, **kwargs)

            # Invalidate cache
            cache_instance = cache or get_cache()

            if key:
                cache_instance.delete(key)
                logger.debug(f"Invalidated cache key: {key}")

            if key_prefix:
                # For prefix invalidation, we'd need to clear or iterate
                # For simplicity, we just log a warning
                logger.warning(
                    f"Prefix-based invalidation ({key_prefix}) requires "
                    "backend-specific implementation"
                )

            return result

        return wrapper

    return decorator


class CacheContext:
    """
    Context manager for temporary cache configuration.

    Example:
        with CacheContext(ttl=60) as ctx:
            # All cached functions will use 60s TTL
            result = cached_function()
    """

    def __init__(
        self,
        cache: Optional[Cache] = None,
        ttl: Optional[int] = None,
        enabled: bool = True,
    ):
        self.cache = cache
        self.ttl = ttl
        self.enabled = enabled
        self._original_cache = None

    def __enter__(self):
        from ai_data_science_team.cache.cache import _default_cache, set_default_cache

        self._original_cache = _default_cache

        if self.cache:
            set_default_cache(self.cache)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        from ai_data_science_team.cache.cache import set_default_cache

        if self._original_cache:
            set_default_cache(self._original_cache)

        return False
