"""
AI Data Science Team - Caching System

This module provides a flexible caching layer for expensive operations
like LLM calls, ML model training, and data transformations.

Example usage:
    from ai_data_science_team.cache import Cache, cached, MemoryBackend

    # Use the default cache
    cache = Cache()
    cache.set("key", "value", ttl=3600)
    value = cache.get("key")

    # Use decorator for automatic caching
    @cached(ttl=3600)
    def expensive_operation(data):
        # This result will be cached
        return process(data)

    # Use with DataFrame hashing for data-aware caching
    @cached(ttl=3600, hash_dataframes=True)
    def analyze_data(df):
        return df.describe()
"""

from ai_data_science_team.cache.backends import (
    CacheBackend,
    MemoryBackend,
    DiskBackend,
)
from ai_data_science_team.cache.cache import (
    Cache,
    get_cache,
    set_default_cache,
)
from ai_data_science_team.cache.decorators import (
    cached,
    cache_result,
    invalidate_cache,
)
from ai_data_science_team.cache.keys import (
    generate_cache_key,
    hash_dataframe,
    hash_args,
)

__all__ = [
    # Backends
    "CacheBackend",
    "MemoryBackend",
    "DiskBackend",
    # Cache
    "Cache",
    "get_cache",
    "set_default_cache",
    # Decorators
    "cached",
    "cache_result",
    "invalidate_cache",
    # Keys
    "generate_cache_key",
    "hash_dataframe",
    "hash_args",
]
