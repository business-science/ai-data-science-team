"""
Unit tests for the caching system.
"""

import time
import pytest
from pathlib import Path
import pandas as pd
import numpy as np


class TestMemoryBackend:
    """Tests for the memory cache backend."""

    def test_memory_backend_set_get(self):
        """Test basic set and get operations."""
        from ai_data_science_team.cache import MemoryBackend

        backend = MemoryBackend()
        backend.set("key1", "value1")

        hit, value = backend.get("key1")
        assert hit is True
        assert value == "value1"

    def test_memory_backend_miss(self):
        """Test cache miss."""
        from ai_data_science_team.cache import MemoryBackend

        backend = MemoryBackend()
        hit, value = backend.get("nonexistent")

        assert hit is False
        assert value is None

    def test_memory_backend_delete(self):
        """Test delete operation."""
        from ai_data_science_team.cache import MemoryBackend

        backend = MemoryBackend()
        backend.set("key1", "value1")
        backend.delete("key1")

        hit, _ = backend.get("key1")
        assert hit is False

    def test_memory_backend_ttl(self):
        """Test TTL expiration."""
        from ai_data_science_team.cache import MemoryBackend

        backend = MemoryBackend()
        backend.set("key1", "value1", ttl=1)

        # Should be available immediately
        hit, _ = backend.get("key1")
        assert hit is True

        # Wait for expiration
        time.sleep(1.1)

        hit, _ = backend.get("key1")
        assert hit is False

    def test_memory_backend_lru_eviction(self):
        """Test LRU eviction when max size reached."""
        from ai_data_science_team.cache import MemoryBackend

        backend = MemoryBackend(max_size=3)

        backend.set("key1", "value1")
        backend.set("key2", "value2")
        backend.set("key3", "value3")

        # Access key1 to make it recently used
        backend.get("key1")

        # Add new key, should evict key2 (least recently used)
        backend.set("key4", "value4")

        assert backend.exists("key1")  # Recently accessed
        assert not backend.exists("key2")  # Should be evicted
        assert backend.exists("key3")
        assert backend.exists("key4")

    def test_memory_backend_clear(self):
        """Test clearing the cache."""
        from ai_data_science_team.cache import MemoryBackend

        backend = MemoryBackend()
        backend.set("key1", "value1")
        backend.set("key2", "value2")
        backend.clear()

        assert not backend.exists("key1")
        assert not backend.exists("key2")

    def test_memory_backend_stats(self):
        """Test cache statistics."""
        from ai_data_science_team.cache import MemoryBackend

        backend = MemoryBackend()
        backend.set("key1", "value1")
        backend.get("key1")  # Hit
        backend.get("key2")  # Miss

        stats = backend.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1


class TestDiskBackend:
    """Tests for the disk cache backend."""

    def test_disk_backend_set_get(self, tmp_path):
        """Test basic set and get operations."""
        from ai_data_science_team.cache import DiskBackend

        backend = DiskBackend(cache_dir=str(tmp_path / "cache"))
        backend.set("key1", {"data": [1, 2, 3]})

        hit, value = backend.get("key1")
        assert hit is True
        assert value == {"data": [1, 2, 3]}

    def test_disk_backend_complex_objects(self, tmp_path):
        """Test caching complex objects."""
        from ai_data_science_team.cache import DiskBackend

        backend = DiskBackend(cache_dir=str(tmp_path / "cache"))

        # Cache a DataFrame
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        backend.set("dataframe", df)

        hit, value = backend.get("dataframe")
        assert hit is True
        pd.testing.assert_frame_equal(value, df)

    def test_disk_backend_ttl(self, tmp_path):
        """Test TTL expiration."""
        from ai_data_science_team.cache import DiskBackend

        backend = DiskBackend(cache_dir=str(tmp_path / "cache"))
        backend.set("key1", "value1", ttl=1)

        hit, _ = backend.get("key1")
        assert hit is True

        time.sleep(1.1)

        hit, _ = backend.get("key1")
        assert hit is False

    def test_disk_backend_persistence(self, tmp_path):
        """Test that cache persists across backend instances."""
        from ai_data_science_team.cache import DiskBackend

        cache_dir = str(tmp_path / "cache")

        # First backend instance
        backend1 = DiskBackend(cache_dir=cache_dir)
        backend1.set("persistent_key", "persistent_value")

        # New backend instance pointing to same directory
        backend2 = DiskBackend(cache_dir=cache_dir)
        hit, value = backend2.get("persistent_key")

        assert hit is True
        assert value == "persistent_value"


class TestCache:
    """Tests for the main Cache class."""

    def test_cache_basic_operations(self):
        """Test basic cache operations."""
        from ai_data_science_team.cache import Cache, MemoryBackend

        cache = Cache(backend=MemoryBackend())

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.get("nonexistent") is None
        assert cache.get("nonexistent", "default") == "default"

    def test_cache_namespace(self):
        """Test cache with namespace."""
        from ai_data_science_team.cache import Cache, MemoryBackend

        backend = MemoryBackend()
        cache1 = Cache(backend=backend, namespace="ns1")
        cache2 = Cache(backend=backend, namespace="ns2")

        cache1.set("key", "value1")
        cache2.set("key", "value2")

        assert cache1.get("key") == "value1"
        assert cache2.get("key") == "value2"

    def test_cache_get_or_set(self):
        """Test get_or_set operation."""
        from ai_data_science_team.cache import Cache, MemoryBackend

        cache = Cache(backend=MemoryBackend())
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return "computed_value"

        # First call should execute factory
        result1 = cache.get_or_set("key", factory)
        assert result1 == "computed_value"
        assert call_count == 1

        # Second call should return cached value
        result2 = cache.get_or_set("key", factory)
        assert result2 == "computed_value"
        assert call_count == 1  # Factory not called again

    def test_cache_default_ttl(self):
        """Test cache with default TTL."""
        from ai_data_science_team.cache import Cache, MemoryBackend

        cache = Cache(backend=MemoryBackend(), default_ttl=1)
        cache.set("key", "value")

        assert cache.exists("key")
        time.sleep(1.1)
        assert not cache.exists("key")


class TestCacheKeys:
    """Tests for cache key generation."""

    def test_hash_dataframe(self):
        """Test DataFrame hashing."""
        from ai_data_science_team.cache import hash_dataframe

        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df3 = pd.DataFrame({"a": [1, 2, 4], "b": [4, 5, 6]})

        hash1 = hash_dataframe(df1)
        hash2 = hash_dataframe(df2)
        hash3 = hash_dataframe(df3)

        assert hash1 == hash2  # Same data should have same hash
        assert hash1 != hash3  # Different data should have different hash

    def test_hash_args(self):
        """Test argument hashing."""
        from ai_data_science_team.cache import hash_args

        hash1 = hash_args((1, 2), {"a": 3})
        hash2 = hash_args((1, 2), {"a": 3})
        hash3 = hash_args((1, 2), {"a": 4})

        assert hash1 == hash2
        assert hash1 != hash3

    def test_generate_cache_key(self):
        """Test cache key generation."""
        from ai_data_science_team.cache import generate_cache_key

        def my_func(x, y):
            return x + y

        key1 = generate_cache_key(my_func, (1, 2), {})
        key2 = generate_cache_key(my_func, (1, 2), {})
        key3 = generate_cache_key(my_func, (1, 3), {})

        assert key1 == key2
        assert key1 != key3


class TestCacheDecorators:
    """Tests for caching decorators."""

    def test_cached_decorator(self):
        """Test @cached decorator."""
        from ai_data_science_team.cache import cached, Cache, MemoryBackend

        cache = Cache(backend=MemoryBackend())
        call_count = 0

        @cached(cache=cache)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call with same args should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1

        # Call with different args
        result3 = expensive_function(6)
        assert result3 == 12
        assert call_count == 2

    def test_cached_with_dataframe(self):
        """Test @cached decorator with DataFrame arguments."""
        from ai_data_science_team.cache import cached, Cache, MemoryBackend

        cache = Cache(backend=MemoryBackend())
        call_count = 0

        @cached(cache=cache, hash_dataframes=True)
        def process_df(df):
            nonlocal call_count
            call_count += 1
            return df.sum().sum()

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        result1 = process_df(df)
        assert result1 == 10
        assert call_count == 1

        # Same DataFrame should hit cache
        result2 = process_df(df)
        assert result2 == 10
        assert call_count == 1

    def test_cached_with_ttl(self):
        """Test @cached decorator with TTL."""
        from ai_data_science_team.cache import cached, Cache, MemoryBackend

        cache = Cache(backend=MemoryBackend())
        call_count = 0

        @cached(cache=cache, ttl=1)
        def short_lived(x):
            nonlocal call_count
            call_count += 1
            return x

        result1 = short_lived(1)
        assert call_count == 1

        # Should use cache
        result2 = short_lived(1)
        assert call_count == 1

        # Wait for TTL
        time.sleep(1.1)

        # Should call function again
        result3 = short_lived(1)
        assert call_count == 2

    def test_cached_unless(self):
        """Test @cached decorator with unless condition."""
        from ai_data_science_team.cache import cached, Cache, MemoryBackend

        cache = Cache(backend=MemoryBackend())

        @cached(cache=cache, unless=lambda x: x is None)
        def maybe_none(return_none):
            return None if return_none else "value"

        # Should cache "value"
        result1 = maybe_none(False)
        assert result1 == "value"

        # Should not cache None
        result2 = maybe_none(True)
        assert result2 is None

    def test_cache_result_decorator(self):
        """Test @cache_result decorator."""
        from ai_data_science_team.cache import cache_result, Cache, MemoryBackend

        cache = Cache(backend=MemoryBackend())
        call_count = 0

        @cache_result("fixed_key", cache=cache)
        def compute_value():
            nonlocal call_count
            call_count += 1
            return "expensive_result"

        result1 = compute_value()
        assert result1 == "expensive_result"
        assert call_count == 1

        result2 = compute_value()
        assert result2 == "expensive_result"
        assert call_count == 1


class TestGlobalCache:
    """Tests for global cache functions."""

    def test_get_default_cache(self):
        """Test getting default cache."""
        from ai_data_science_team.cache import get_cache

        cache = get_cache()
        assert cache is not None

    def test_set_default_cache(self):
        """Test setting default cache."""
        from ai_data_science_team.cache import (
            Cache, MemoryBackend, get_cache, set_default_cache
        )
        import ai_data_science_team.cache.cache as cache_module

        # Save original
        original = cache_module._default_cache

        try:
            new_cache = Cache(backend=MemoryBackend(), namespace="test")
            set_default_cache(new_cache)

            retrieved = get_cache()
            assert retrieved.namespace == "test"
        finally:
            # Restore original
            cache_module._default_cache = original
