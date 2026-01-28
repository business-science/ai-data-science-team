"""
Cache backends for AI Data Science Team.

This module provides different storage backends for the caching system.
"""

import json
import hashlib
import pickle
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import threading
import logging

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: str) -> Tuple[bool, Any]:
        """
        Get a value from the cache.

        Parameters
        ----------
        key : str
            The cache key.

        Returns
        -------
        tuple of (bool, Any)
            (hit, value) - hit is True if key was found, value is the cached value.
        """
        pass

    @abstractmethod
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
            Time-to-live in seconds. None means no expiration.

        Returns
        -------
        bool
            True if the value was set successfully.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries from the cache."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {}


class MemoryBackend(CacheBackend):
    """
    In-memory LRU cache backend.

    This is the default backend, suitable for single-process applications.
    Uses an LRU (Least Recently Used) eviction policy.

    Parameters
    ----------
    max_size : int, default 1000
        Maximum number of entries in the cache.
    max_memory_mb : float, optional
        Maximum memory usage in megabytes (approximate).
    """

    def __init__(self, max_size: int = 1000, max_memory_mb: Optional[float] = None):
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self._cache: OrderedDict = OrderedDict()
        self._expiry: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Tuple[bool, Any]:
        with self._lock:
            # Check if key exists
            if key not in self._cache:
                self._misses += 1
                return (False, None)

            # Check expiry
            if key in self._expiry:
                if time.time() > self._expiry[key]:
                    # Expired - delete and return miss
                    del self._cache[key]
                    del self._expiry[key]
                    self._misses += 1
                    return (False, None)

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return (True, self._cache[key])

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        with self._lock:
            # Remove if exists (to update position)
            if key in self._cache:
                del self._cache[key]

            # Add new entry
            self._cache[key] = value

            # Set expiry if TTL provided
            if ttl is not None:
                self._expiry[key] = time.time() + ttl
            elif key in self._expiry:
                del self._expiry[key]

            # Evict oldest entries if over max size
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                if oldest_key in self._expiry:
                    del self._expiry[oldest_key]

            return True

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._expiry:
                    del self._expiry[key]
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._expiry.clear()
            self._hits = 0
            self._misses = 0

    def exists(self, key: str) -> bool:
        with self._lock:
            if key not in self._cache:
                return False
            # Check expiry
            if key in self._expiry and time.time() > self._expiry[key]:
                del self._cache[key]
                del self._expiry[key]
                return False
            return True

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0
            return {
                "backend": "memory",
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count of removed entries."""
        with self._lock:
            now = time.time()
            expired_keys = [
                k for k, exp_time in self._expiry.items()
                if now > exp_time
            ]
            for key in expired_keys:
                if key in self._cache:
                    del self._cache[key]
                del self._expiry[key]
            return len(expired_keys)


class DiskBackend(CacheBackend):
    """
    Disk-based cache backend using pickle serialization.

    Suitable for caching large objects or persisting cache across restarts.

    Parameters
    ----------
    cache_dir : str or Path
        Directory to store cache files.
    max_size_mb : float, default 1000
        Maximum total cache size in megabytes.
    """

    def __init__(
        self,
        cache_dir: str = ".cache/ai_ds_team",
        max_size_mb: float = 1000,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def _get_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        # Use hash to create safe filename
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def _get_meta_path(self, key: str) -> Path:
        """Get the metadata file path for a cache key."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.meta"

    def get(self, key: str) -> Tuple[bool, Any]:
        with self._lock:
            cache_path = self._get_path(key)
            meta_path = self._get_meta_path(key)

            if not cache_path.exists():
                self._misses += 1
                return (False, None)

            # Check expiry from metadata
            if meta_path.exists():
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    if meta.get("expiry") and time.time() > meta["expiry"]:
                        # Expired - delete and return miss
                        cache_path.unlink(missing_ok=True)
                        meta_path.unlink(missing_ok=True)
                        self._misses += 1
                        return (False, None)
                except Exception:
                    pass

            # Load cached value
            try:
                with open(cache_path, "rb") as f:
                    value = pickle.load(f)
                self._hits += 1
                return (True, value)
            except Exception as e:
                logger.warning(f"Failed to load cache for {key}: {e}")
                self._misses += 1
                return (False, None)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        with self._lock:
            cache_path = self._get_path(key)
            meta_path = self._get_meta_path(key)

            try:
                # Write value
                with open(cache_path, "wb") as f:
                    pickle.dump(value, f)

                # Write metadata
                meta = {
                    "key": key,
                    "created": time.time(),
                    "expiry": time.time() + ttl if ttl else None,
                }
                with open(meta_path, "w") as f:
                    json.dump(meta, f)

                return True
            except Exception as e:
                logger.warning(f"Failed to cache {key}: {e}")
                return False

    def delete(self, key: str) -> bool:
        with self._lock:
            cache_path = self._get_path(key)
            meta_path = self._get_meta_path(key)

            deleted = False
            if cache_path.exists():
                cache_path.unlink()
                deleted = True
            if meta_path.exists():
                meta_path.unlink()
                deleted = True

            return deleted

    def clear(self) -> None:
        with self._lock:
            for path in self.cache_dir.glob("*.cache"):
                path.unlink(missing_ok=True)
            for path in self.cache_dir.glob("*.meta"):
                path.unlink(missing_ok=True)
            self._hits = 0
            self._misses = 0

    def exists(self, key: str) -> bool:
        cache_path = self._get_path(key)
        if not cache_path.exists():
            return False

        # Check expiry
        meta_path = self._get_meta_path(key)
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                if meta.get("expiry") and time.time() > meta["expiry"]:
                    return False
            except Exception:
                pass

        return True

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            # Calculate total size
            total_size = sum(
                f.stat().st_size for f in self.cache_dir.glob("*.cache")
            )
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0

            return {
                "backend": "disk",
                "cache_dir": str(self.cache_dir),
                "size_mb": total_size / (1024 * 1024),
                "max_size_mb": self.max_size_mb,
                "file_count": len(list(self.cache_dir.glob("*.cache"))),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count of removed entries."""
        with self._lock:
            removed = 0
            now = time.time()

            for meta_path in self.cache_dir.glob("*.meta"):
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    if meta.get("expiry") and now > meta["expiry"]:
                        # Get corresponding cache file
                        cache_path = meta_path.with_suffix(".cache")
                        cache_path.unlink(missing_ok=True)
                        meta_path.unlink(missing_ok=True)
                        removed += 1
                except Exception:
                    pass

            return removed
