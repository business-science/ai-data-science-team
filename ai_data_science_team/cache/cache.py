"""
Main Cache class for AI Data Science Team.

This module provides the main Cache interface that wraps different backends.
"""

from typing import Any, Optional, Type
import logging

from ai_data_science_team.cache.backends import CacheBackend, MemoryBackend

logger = logging.getLogger(__name__)

# Global default cache instance
_default_cache: Optional["Cache"] = None


class Cache:
    """
    Main cache interface.

    This class provides a unified interface for caching with different backends.

    Parameters
    ----------
    backend : CacheBackend, optional
        The cache backend to use. Defaults to MemoryBackend.
    default_ttl : int, optional
        Default time-to-live in seconds for cached entries.
    namespace : str, optional
        Namespace prefix for all keys in this cache instance.

    Example:
        # Create a cache with default memory backend
        cache = Cache()

        # Set and get values
        cache.set("my_key", {"data": [1, 2, 3]}, ttl=3600)
        value = cache.get("my_key")

        # Use with custom backend
        from ai_data_science_team.cache import DiskBackend
        disk_cache = Cache(backend=DiskBackend("./my_cache"))
    """

    def __init__(
        self,
        backend: Optional[CacheBackend] = None,
        default_ttl: Optional[int] = None,
        namespace: str = "",
    ):
        self.backend = backend or MemoryBackend()
        self.default_ttl = default_ttl
        self.namespace = namespace

    def _make_key(self, key: str) -> str:
        """Create the full key with namespace."""
        if self.namespace:
            return f"{self.namespace}:{key}"
        return key

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the cache.

        Parameters
        ----------
        key : str
            The cache key.
        default : Any, optional
            Value to return if key not found.

        Returns
        -------
        Any
            The cached value or default.
        """
        full_key = self._make_key(key)
        hit, value = self.backend.get(full_key)
        if hit:
            logger.debug(f"Cache hit: {key}")
            return value
        logger.debug(f"Cache miss: {key}")
        return default

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a value in the cache.

        Parameters
        ----------
        key : str
            The cache key.
        value : Any
            The value to cache.
        ttl : int, optional
            Time-to-live in seconds. Uses default_ttl if not specified.

        Returns
        -------
        bool
            True if the value was cached successfully.
        """
        full_key = self._make_key(key)
        actual_ttl = ttl if ttl is not None else self.default_ttl
        success = self.backend.set(full_key, value, actual_ttl)
        if success:
            logger.debug(f"Cached: {key} (ttl={actual_ttl})")
        return success

    def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.

        Parameters
        ----------
        key : str
            The cache key.

        Returns
        -------
        bool
            True if the key was deleted.
        """
        full_key = self._make_key(key)
        return self.backend.delete(full_key)

    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache.

        Parameters
        ----------
        key : str
            The cache key.

        Returns
        -------
        bool
            True if the key exists and is not expired.
        """
        full_key = self._make_key(key)
        return self.backend.exists(full_key)

    def clear(self) -> None:
        """Clear all entries from the cache."""
        self.backend.clear()
        logger.info("Cache cleared")

    def get_or_set(
        self,
        key: str,
        factory: callable,
        ttl: Optional[int] = None,
    ) -> Any:
        """
        Get a value from cache, or compute and cache it if not found.

        Parameters
        ----------
        key : str
            The cache key.
        factory : callable
            Function to call to compute the value if not cached.
        ttl : int, optional
            Time-to-live for the cached value.

        Returns
        -------
        Any
            The cached or computed value.
        """
        value = self.get(key)
        if value is not None:
            return value

        # Compute and cache
        value = factory()
        self.set(key, value, ttl)
        return value

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns
        -------
        dict
            Statistics including hits, misses, hit rate, and size.
        """
        stats = self.backend.get_stats()
        stats["namespace"] = self.namespace
        stats["default_ttl"] = self.default_ttl
        return stats

    def __repr__(self) -> str:
        return f"Cache(backend={type(self.backend).__name__}, namespace={self.namespace!r})"


def get_cache() -> Cache:
    """
    Get the default cache instance.

    Creates a new MemoryBackend-based cache if none exists.

    Returns
    -------
    Cache
        The default cache instance.
    """
    global _default_cache
    if _default_cache is None:
        _default_cache = Cache()
    return _default_cache


def set_default_cache(cache: Cache) -> None:
    """
    Set the default cache instance.

    Parameters
    ----------
    cache : Cache
        The cache to use as default.
    """
    global _default_cache
    _default_cache = cache
