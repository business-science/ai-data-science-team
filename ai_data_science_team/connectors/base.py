"""
Base classes for data connectors.

This module provides abstract base classes and common interfaces
for all data connectors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class ConnectionConfig:
    """
    Configuration for database/cloud connections.

    Parameters
    ----------
    host : str, optional
        Host address for the connection.
    port : int, optional
        Port number.
    database : str, optional
        Database name.
    schema : str, optional
        Schema name.
    username : str, optional
        Username for authentication.
    password : str, optional
        Password for authentication.
    warehouse : str, optional
        Warehouse name (for Snowflake).
    project : str, optional
        Project ID (for BigQuery).
    region : str, optional
        Cloud region.
    credentials_path : str, optional
        Path to credentials file.
    extra : dict, optional
        Additional connection parameters.
    """
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    schema: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    warehouse: Optional[str] = None
    project: Optional[str] = None
    region: Optional[str] = None
    credentials_path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary, excluding None values."""
        return {
            k: v for k, v in {
                "host": self.host,
                "port": self.port,
                "database": self.database,
                "schema": self.schema,
                "username": self.username,
                "password": self.password,
                "warehouse": self.warehouse,
                "project": self.project,
                "region": self.region,
                "credentials_path": self.credentials_path,
                **self.extra,
            }.items() if v is not None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConnectionConfig":
        """Create config from dictionary."""
        known_fields = {
            "host", "port", "database", "schema", "username", "password",
            "warehouse", "project", "region", "credentials_path",
        }
        known = {k: v for k, v in data.items() if k in known_fields}
        extra = {k: v for k, v in data.items() if k not in known_fields}
        return cls(**known, extra=extra)

    @classmethod
    def from_env(cls, prefix: str = "") -> "ConnectionConfig":
        """
        Create config from environment variables.

        Parameters
        ----------
        prefix : str
            Prefix for environment variable names (e.g., "SNOWFLAKE_").
        """
        import os

        def get_env(name: str) -> Optional[str]:
            return os.environ.get(f"{prefix}{name}")

        return cls(
            host=get_env("HOST"),
            port=int(get_env("PORT")) if get_env("PORT") else None,
            database=get_env("DATABASE"),
            schema=get_env("SCHEMA"),
            username=get_env("USERNAME") or get_env("USER"),
            password=get_env("PASSWORD"),
            warehouse=get_env("WAREHOUSE"),
            project=get_env("PROJECT"),
            region=get_env("REGION"),
            credentials_path=get_env("CREDENTIALS_PATH"),
        )


@dataclass
class QueryResult:
    """
    Result from a database query.

    Parameters
    ----------
    data : Any
        Query result data (DataFrame, list, etc.).
    rows_affected : int
        Number of rows affected/returned.
    columns : list, optional
        Column names if applicable.
    execution_time : float
        Query execution time in seconds.
    query : str, optional
        The executed query.
    metadata : dict
        Additional metadata about the query.
    """
    data: Any
    rows_affected: int = 0
    columns: Optional[List[str]] = None
    execution_time: float = 0.0
    query: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if query was successful."""
        return self.data is not None or self.rows_affected >= 0


class DataConnector(ABC):
    """
    Abstract base class for data connectors.

    All connectors should inherit from this class and implement
    the abstract methods.

    Parameters
    ----------
    config : ConnectionConfig
        Connection configuration.
    auto_connect : bool, default True
        Whether to connect automatically on initialization.
    """

    def __init__(
        self,
        config: Optional[ConnectionConfig] = None,
        auto_connect: bool = True,
        **kwargs,
    ):
        self.config = config or ConnectionConfig(**kwargs)
        self._connection = None
        self._connected = False

        if auto_connect:
            self.connect()

    @property
    def is_connected(self) -> bool:
        """Check if connector is connected."""
        return self._connected

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the data source.

        Returns
        -------
        bool
            True if connection successful.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection."""
        pass

    @abstractmethod
    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        """
        Execute a query.

        Parameters
        ----------
        query : str
            SQL or query string to execute.
        params : dict, optional
            Query parameters.

        Returns
        -------
        QueryResult
            Query execution result.
        """
        pass

    @abstractmethod
    def fetch(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        as_dataframe: bool = True,
    ) -> Union[Any, "pd.DataFrame"]:
        """
        Fetch data from a query.

        Parameters
        ----------
        query : str
            Query to execute.
        params : dict, optional
            Query parameters.
        as_dataframe : bool, default True
            Return results as pandas DataFrame.

        Returns
        -------
        DataFrame or list
            Query results.
        """
        pass

    def fetch_one(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a single row from a query.

        Parameters
        ----------
        query : str
            Query to execute.
        params : dict, optional
            Query parameters.

        Returns
        -------
        dict or None
            Single row as dictionary, or None if no results.
        """
        result = self.fetch(query, params, as_dataframe=False)
        if result and len(result) > 0:
            return result[0]
        return None

    def table_exists(self, table_name: str, schema: Optional[str] = None) -> bool:
        """
        Check if a table exists.

        Parameters
        ----------
        table_name : str
            Name of the table.
        schema : str, optional
            Schema name.

        Returns
        -------
        bool
            True if table exists.
        """
        # Default implementation - subclasses may override
        try:
            self.fetch(f"SELECT 1 FROM {schema + '.' if schema else ''}{table_name} LIMIT 1")
            return True
        except Exception:
            return False

    def get_tables(self, schema: Optional[str] = None) -> List[str]:
        """
        Get list of tables.

        Parameters
        ----------
        schema : str, optional
            Schema to list tables from.

        Returns
        -------
        list of str
            Table names.
        """
        # Default implementation - subclasses should override
        raise NotImplementedError("get_tables not implemented for this connector")

    def get_columns(
        self,
        table_name: str,
        schema: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get column information for a table.

        Parameters
        ----------
        table_name : str
            Name of the table.
        schema : str, optional
            Schema name.

        Returns
        -------
        list of dict
            Column information (name, type, nullable, etc.).
        """
        # Default implementation - subclasses should override
        raise NotImplementedError("get_columns not implemented for this connector")

    def write(
        self,
        data: "pd.DataFrame",
        table_name: str,
        schema: Optional[str] = None,
        if_exists: str = "append",
        **kwargs,
    ) -> int:
        """
        Write a DataFrame to the data source.

        Parameters
        ----------
        data : DataFrame
            Data to write.
        table_name : str
            Target table name.
        schema : str, optional
            Target schema.
        if_exists : str, default 'append'
            Behavior if table exists ('fail', 'replace', 'append').

        Returns
        -------
        int
            Number of rows written.
        """
        # Default implementation - subclasses should override
        raise NotImplementedError("write not implemented for this connector")

    def ping(self) -> bool:
        """
        Test the connection.

        Returns
        -------
        bool
            True if connection is alive.
        """
        try:
            self.execute("SELECT 1")
            return True
        except Exception:
            return False

    def __enter__(self):
        """Context manager entry."""
        if not self.is_connected:
            self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(connected={self.is_connected})"


class MockConnector(DataConnector):
    """
    Mock connector for testing purposes.

    Stores data in memory and simulates database operations.
    """

    def __init__(
        self,
        config: Optional[ConnectionConfig] = None,
        auto_connect: bool = True,
        **kwargs,
    ):
        self._tables: Dict[str, Any] = {}
        # Remove auto_connect from kwargs if present to avoid duplicate
        kwargs.pop("auto_connect", None)
        super().__init__(config, auto_connect=False, **kwargs)
        self._connected = True

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        start = time.time()
        logger.debug(f"Mock execute: {query}")
        return QueryResult(
            data=None,
            rows_affected=0,
            execution_time=time.time() - start,
            query=query,
        )

    def fetch(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        as_dataframe: bool = True,
    ) -> Any:
        import pandas as pd

        # Simple mock: return empty DataFrame
        if as_dataframe:
            return pd.DataFrame()
        return []

    def get_tables(self, schema: Optional[str] = None) -> List[str]:
        return list(self._tables.keys())

    def write(
        self,
        data: Any,
        table_name: str,
        schema: Optional[str] = None,
        if_exists: str = "append",
        **kwargs,
    ) -> int:
        import pandas as pd

        key = f"{schema}.{table_name}" if schema else table_name

        if if_exists == "replace" or key not in self._tables:
            self._tables[key] = data.copy() if isinstance(data, pd.DataFrame) else data
        elif if_exists == "append":
            if isinstance(data, pd.DataFrame):
                self._tables[key] = pd.concat([self._tables[key], data], ignore_index=True)
        elif if_exists == "fail":
            if key in self._tables:
                raise ValueError(f"Table {key} already exists")

        return len(data) if hasattr(data, "__len__") else 1
