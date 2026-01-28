"""
Cache key generation utilities.

This module provides utilities for generating consistent cache keys
from function arguments, including special handling for DataFrames.
"""

import hashlib
import json
from typing import Any, Callable, Tuple
import logging

logger = logging.getLogger(__name__)


def hash_dataframe(df) -> str:
    """
    Generate a hash for a pandas DataFrame.

    Uses a combination of shape, column names, dtypes, and sampled data
    to create a unique identifier for the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to hash.

    Returns
    -------
    str
        A hex string hash of the DataFrame.
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df)}")

    # Create a signature from DataFrame properties
    signature_parts = [
        f"shape:{df.shape}",
        f"columns:{list(df.columns)}",
        f"dtypes:{df.dtypes.to_dict()}",
    ]

    # Add hash of actual data (sample if large)
    if len(df) > 1000:
        # Sample for large DataFrames
        sample = df.sample(n=1000, random_state=42)
        signature_parts.append(f"sample_hash:{pd.util.hash_pandas_object(sample).sum()}")
    else:
        signature_parts.append(f"data_hash:{pd.util.hash_pandas_object(df).sum()}")

    # Combine and hash
    signature = "|".join(str(p) for p in signature_parts)
    return hashlib.sha256(signature.encode()).hexdigest()[:16]


def hash_value(value: Any) -> str:
    """
    Generate a hash for an arbitrary value.

    Parameters
    ----------
    value : Any
        The value to hash.

    Returns
    -------
    str
        A hex string hash of the value.
    """
    try:
        import pandas as pd
        if isinstance(value, pd.DataFrame):
            return hash_dataframe(value)
    except ImportError:
        pass

    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return hashlib.sha256(value.tobytes()).hexdigest()[:16]
    except ImportError:
        pass

    # For other types, use JSON serialization
    try:
        serialized = json.dumps(value, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
    except (TypeError, ValueError):
        # Fallback to repr
        return hashlib.sha256(repr(value).encode()).hexdigest()[:16]


def hash_args(args: Tuple, kwargs: dict, hash_dataframes: bool = True) -> str:
    """
    Generate a hash for function arguments.

    Parameters
    ----------
    args : tuple
        Positional arguments.
    kwargs : dict
        Keyword arguments.
    hash_dataframes : bool, default True
        Whether to use special DataFrame hashing.

    Returns
    -------
    str
        A hex string hash of the arguments.
    """
    parts = []

    # Hash positional arguments
    for i, arg in enumerate(args):
        arg_hash = hash_value(arg) if hash_dataframes else hashlib.sha256(
            repr(arg).encode()
        ).hexdigest()[:16]
        parts.append(f"arg{i}:{arg_hash}")

    # Hash keyword arguments (sorted for consistency)
    for key in sorted(kwargs.keys()):
        val_hash = hash_value(kwargs[key]) if hash_dataframes else hashlib.sha256(
            repr(kwargs[key]).encode()
        ).hexdigest()[:16]
        parts.append(f"{key}:{val_hash}")

    combined = "|".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:32]


def generate_cache_key(
    func: Callable,
    args: Tuple,
    kwargs: dict,
    prefix: str = "",
    hash_dataframes: bool = True,
) -> str:
    """
    Generate a cache key for a function call.

    Parameters
    ----------
    func : callable
        The function being cached.
    args : tuple
        Positional arguments to the function.
    kwargs : dict
        Keyword arguments to the function.
    prefix : str, optional
        Prefix to add to the cache key.
    hash_dataframes : bool, default True
        Whether to use special DataFrame hashing.

    Returns
    -------
    str
        A unique cache key for this function call.
    """
    # Get function identifier
    func_name = f"{func.__module__}.{func.__qualname__}"

    # Hash arguments
    args_hash = hash_args(args, kwargs, hash_dataframes)

    # Combine into key
    key_parts = [prefix, func_name, args_hash] if prefix else [func_name, args_hash]
    return ":".join(filter(None, key_parts))
