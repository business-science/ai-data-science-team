"""
Connector factory for creating data connectors.

This module provides factory functions for creating and managing
data connectors in a unified way.
"""

import logging
from typing import Any, Dict, Optional, Type

from ai_data_science_team.connectors.base import DataConnector, ConnectionConfig

logger = logging.getLogger(__name__)

# Registry of connector types
_CONNECTOR_REGISTRY: Dict[str, Type[DataConnector]] = {}


def _register_builtin_connectors() -> None:
    """Register built-in connectors."""
    from ai_data_science_team.connectors.snowflake import SnowflakeConnector
    from ai_data_science_team.connectors.bigquery import BigQueryConnector
    from ai_data_science_team.connectors.redshift import RedshiftConnector
    from ai_data_science_team.connectors.postgres import PostgresConnector
    from ai_data_science_team.connectors.s3 import S3Connector
    from ai_data_science_team.connectors.base import MockConnector

    _CONNECTOR_REGISTRY.update({
        "snowflake": SnowflakeConnector,
        "bigquery": BigQueryConnector,
        "bq": BigQueryConnector,
        "redshift": RedshiftConnector,
        "postgres": PostgresConnector,
        "postgresql": PostgresConnector,
        "pg": PostgresConnector,
        "s3": S3Connector,
        "mock": MockConnector,
    })


def register_connector(name: str, connector_class: Type[DataConnector]) -> None:
    """
    Register a custom connector type.

    Parameters
    ----------
    name : str
        Name for the connector type.
    connector_class : type
        Connector class (must inherit from DataConnector).

    Example
    -------
    >>> from ai_data_science_team.connectors import register_connector
    >>> register_connector("mydb", MyCustomConnector)
    """
    if not issubclass(connector_class, DataConnector):
        raise TypeError(f"{connector_class} must inherit from DataConnector")

    _CONNECTOR_REGISTRY[name.lower()] = connector_class
    logger.debug(f"Registered connector: {name}")


def get_connector(
    connector_type: str,
    config: Optional[ConnectionConfig] = None,
    auto_connect: bool = True,
    **kwargs,
) -> DataConnector:
    """
    Get a connector instance by type.

    Parameters
    ----------
    connector_type : str
        Type of connector (snowflake, bigquery, redshift, postgres, s3).
    config : ConnectionConfig, optional
        Connection configuration.
    auto_connect : bool, default True
        Connect automatically.
    **kwargs
        Additional arguments passed to connector constructor.

    Returns
    -------
    DataConnector
        Connector instance.

    Example
    -------
    >>> # Create Snowflake connector
    >>> connector = get_connector(
    ...     "snowflake",
    ...     account="myaccount",
    ...     user="myuser",
    ...     password="mypassword",
    ...     database="mydb",
    ... )
    >>>
    >>> # Create BigQuery connector
    >>> connector = get_connector(
    ...     "bigquery",
    ...     project="my-project",
    ...     credentials_path="/path/to/creds.json",
    ... )
    >>>
    >>> # Create S3 connector
    >>> connector = get_connector("s3", bucket="my-bucket")
    """
    # Ensure built-in connectors are registered
    if not _CONNECTOR_REGISTRY:
        _register_builtin_connectors()

    connector_type = connector_type.lower()

    if connector_type not in _CONNECTOR_REGISTRY:
        available = ", ".join(sorted(_CONNECTOR_REGISTRY.keys()))
        raise ValueError(
            f"Unknown connector type: {connector_type}. "
            f"Available types: {available}"
        )

    connector_class = _CONNECTOR_REGISTRY[connector_type]
    return connector_class(config=config, auto_connect=auto_connect, **kwargs)


def list_connectors() -> Dict[str, Type[DataConnector]]:
    """
    List all registered connector types.

    Returns
    -------
    dict
        Dictionary of connector name to class.
    """
    if not _CONNECTOR_REGISTRY:
        _register_builtin_connectors()

    return dict(_CONNECTOR_REGISTRY)


def get_connector_from_url(
    url: str,
    auto_connect: bool = True,
    **kwargs,
) -> DataConnector:
    """
    Create a connector from a connection URL.

    Supports URLs like:
    - snowflake://user:pass@account/database/schema?warehouse=wh
    - bigquery://project/dataset
    - postgres://user:pass@host:port/database
    - redshift://user:pass@host:port/database
    - s3://bucket/prefix

    Parameters
    ----------
    url : str
        Connection URL.
    auto_connect : bool, default True
        Connect automatically.
    **kwargs
        Additional arguments.

    Returns
    -------
    DataConnector
        Connector instance.
    """
    from urllib.parse import urlparse, parse_qs

    parsed = urlparse(url)
    scheme = parsed.scheme.lower()

    if scheme in ("snowflake", "sf"):
        # snowflake://user:pass@account/database/schema
        parts = parsed.path.strip("/").split("/")
        return get_connector(
            "snowflake",
            account=parsed.hostname,
            user=parsed.username,
            password=parsed.password,
            database=parts[0] if parts else None,
            schema=parts[1] if len(parts) > 1 else None,
            auto_connect=auto_connect,
            **kwargs,
        )

    elif scheme in ("bigquery", "bq"):
        # bigquery://project/dataset
        parts = parsed.path.strip("/").split("/")
        return get_connector(
            "bigquery",
            project=parsed.hostname or (parts[0] if parts else None),
            dataset=parts[1] if len(parts) > 1 else (parts[0] if parts else None),
            auto_connect=auto_connect,
            **kwargs,
        )

    elif scheme in ("postgres", "postgresql", "pg"):
        # postgres://user:pass@host:port/database
        return get_connector(
            "postgres",
            host=parsed.hostname,
            port=parsed.port or 5432,
            user=parsed.username,
            password=parsed.password,
            database=parsed.path.strip("/"),
            auto_connect=auto_connect,
            **kwargs,
        )

    elif scheme == "redshift":
        # redshift://user:pass@host:port/database
        return get_connector(
            "redshift",
            host=parsed.hostname,
            port=parsed.port or 5439,
            user=parsed.username,
            password=parsed.password,
            database=parsed.path.strip("/"),
            auto_connect=auto_connect,
            **kwargs,
        )

    elif scheme == "s3":
        # s3://bucket/prefix
        return get_connector(
            "s3",
            bucket=parsed.hostname,
            auto_connect=auto_connect,
            **kwargs,
        )

    else:
        raise ValueError(f"Unsupported URL scheme: {scheme}")


class ConnectorPool:
    """
    Pool of reusable connectors.

    Manages a pool of connector instances for efficient reuse
    and automatic cleanup.

    Parameters
    ----------
    max_connections : int, default 10
        Maximum number of connections per type.

    Example
    -------
    >>> pool = ConnectorPool()
    >>>
    >>> # Get a connector from pool (creates if needed)
    >>> with pool.get("snowflake", account="acc", user="user") as conn:
    ...     df = conn.fetch("SELECT * FROM table")
    >>>
    >>> # Connector is returned to pool for reuse
    """

    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self._pools: Dict[str, list] = {}
        self._in_use: Dict[str, list] = {}

    def _pool_key(self, connector_type: str, **kwargs) -> str:
        """Generate a unique key for connection parameters."""
        import hashlib
        import json

        params = {"type": connector_type, **kwargs}
        param_str = json.dumps(params, sort_keys=True, default=str)
        return hashlib.md5(param_str.encode()).hexdigest()[:16]

    def get(
        self,
        connector_type: str,
        **kwargs,
    ) -> DataConnector:
        """
        Get a connector from the pool.

        Parameters
        ----------
        connector_type : str
            Type of connector.
        **kwargs
            Connector parameters.

        Returns
        -------
        DataConnector
            Pooled connector instance.
        """
        key = self._pool_key(connector_type, **kwargs)

        # Check pool for available connection
        if key in self._pools and self._pools[key]:
            connector = self._pools[key].pop()
            if connector.is_connected or connector.ping():
                self._in_use.setdefault(key, []).append(connector)
                return connector

        # Create new connection
        connector = get_connector(connector_type, **kwargs)
        self._in_use.setdefault(key, []).append(connector)
        return connector

    def release(self, connector: DataConnector) -> None:
        """
        Return a connector to the pool.

        Parameters
        ----------
        connector : DataConnector
            Connector to return.
        """
        for key, in_use_list in self._in_use.items():
            if connector in in_use_list:
                in_use_list.remove(connector)
                pool = self._pools.setdefault(key, [])
                if len(pool) < self.max_connections:
                    pool.append(connector)
                else:
                    connector.disconnect()
                return

    def close_all(self) -> None:
        """Close all connections in the pool."""
        for pool in self._pools.values():
            for connector in pool:
                try:
                    connector.disconnect()
                except Exception:
                    pass
            pool.clear()

        for in_use in self._in_use.values():
            for connector in in_use:
                try:
                    connector.disconnect()
                except Exception:
                    pass
            in_use.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all()
        return False


# Global connector pool
_global_pool: Optional[ConnectorPool] = None


def get_global_pool() -> ConnectorPool:
    """Get the global connector pool."""
    global _global_pool
    if _global_pool is None:
        _global_pool = ConnectorPool()
    return _global_pool
