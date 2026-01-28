"""
Snowflake data warehouse connector.

This module provides a connector for Snowflake data warehouse.
"""

import logging
from typing import Any, Dict, List, Optional, Union
import time

from ai_data_science_team.connectors.base import (
    DataConnector,
    ConnectionConfig,
    QueryResult,
)

logger = logging.getLogger(__name__)


class SnowflakeConnector(DataConnector):
    """
    Connector for Snowflake data warehouse.

    Parameters
    ----------
    config : ConnectionConfig, optional
        Connection configuration. Can also pass individual parameters.
    account : str, optional
        Snowflake account identifier.
    user : str, optional
        Username.
    password : str, optional
        Password.
    database : str, optional
        Database name.
    schema : str, optional
        Schema name.
    warehouse : str, optional
        Warehouse name.
    role : str, optional
        Role to use.
    auto_connect : bool, default True
        Connect automatically on initialization.

    Example
    -------
    >>> connector = SnowflakeConnector(
    ...     account="myaccount",
    ...     user="myuser",
    ...     password="mypassword",
    ...     database="mydb",
    ...     warehouse="compute_wh",
    ... )
    >>> df = connector.fetch("SELECT * FROM mytable")
    """

    def __init__(
        self,
        config: Optional[ConnectionConfig] = None,
        account: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        warehouse: Optional[str] = None,
        role: Optional[str] = None,
        auto_connect: bool = True,
        **kwargs,
    ):
        # Build config from parameters if not provided
        if config is None:
            extra = kwargs.copy()
            if account:
                extra["account"] = account
            if role:
                extra["role"] = role

            config = ConnectionConfig(
                host=account,  # Snowflake uses account as identifier
                username=user,
                password=password,
                database=database,
                schema=schema,
                warehouse=warehouse,
                extra=extra,
            )

        super().__init__(config, auto_connect=False)

        if auto_connect:
            self.connect()

    @property
    def account(self) -> Optional[str]:
        """Get Snowflake account."""
        return self.config.extra.get("account") or self.config.host

    def connect(self) -> bool:
        """
        Establish connection to Snowflake.

        Returns
        -------
        bool
            True if connection successful.

        Raises
        ------
        ImportError
            If snowflake-connector-python is not installed.
        """
        try:
            import snowflake.connector
        except ImportError:
            raise ImportError(
                "snowflake-connector-python is required for SnowflakeConnector. "
                "Install with: pip install snowflake-connector-python"
            )

        try:
            conn_params = {
                "account": self.account,
                "user": self.config.username,
                "password": self.config.password,
            }

            if self.config.database:
                conn_params["database"] = self.config.database
            if self.config.schema:
                conn_params["schema"] = self.config.schema
            if self.config.warehouse:
                conn_params["warehouse"] = self.config.warehouse
            if self.config.extra.get("role"):
                conn_params["role"] = self.config.extra["role"]

            # Add any extra parameters
            for key, value in self.config.extra.items():
                if key not in conn_params and key not in ("account", "role"):
                    conn_params[key] = value

            self._connection = snowflake.connector.connect(**conn_params)
            self._connected = True
            logger.info(f"Connected to Snowflake account: {self.account}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {e}")
            self._connected = False
            raise

    def disconnect(self) -> None:
        """Close the Snowflake connection."""
        if self._connection:
            try:
                self._connection.close()
            except Exception as e:
                logger.warning(f"Error closing Snowflake connection: {e}")
            finally:
                self._connection = None
                self._connected = False

    def execute(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """
        Execute a query on Snowflake.

        Parameters
        ----------
        query : str
            SQL query to execute.
        params : dict, optional
            Query parameters.

        Returns
        -------
        QueryResult
            Execution result.
        """
        if not self.is_connected:
            self.connect()

        start_time = time.time()
        cursor = self._connection.cursor()

        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            rows_affected = cursor.rowcount or 0
            execution_time = time.time() - start_time

            return QueryResult(
                data=None,
                rows_affected=rows_affected,
                execution_time=execution_time,
                query=query,
                metadata={"query_id": cursor.sfqid},
            )

        except Exception as e:
            logger.error(f"Snowflake query failed: {e}")
            raise

        finally:
            cursor.close()

    def fetch(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        as_dataframe: bool = True,
    ) -> Union[Any, "pd.DataFrame"]:
        """
        Fetch data from Snowflake.

        Parameters
        ----------
        query : str
            SQL query to execute.
        params : dict, optional
            Query parameters.
        as_dataframe : bool, default True
            Return as pandas DataFrame.

        Returns
        -------
        DataFrame or list
            Query results.
        """
        if not self.is_connected:
            self.connect()

        cursor = self._connection.cursor()

        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            if as_dataframe:
                import pandas as pd
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                data = cursor.fetchall()
                return pd.DataFrame(data, columns=columns)
            else:
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                rows = cursor.fetchall()
                return [dict(zip(columns, row)) for row in rows]

        finally:
            cursor.close()

    def get_tables(self, schema: Optional[str] = None) -> List[str]:
        """
        Get list of tables in the database.

        Parameters
        ----------
        schema : str, optional
            Schema to list tables from. Uses current schema if not provided.

        Returns
        -------
        list of str
            Table names.
        """
        schema = schema or self.config.schema
        query = "SHOW TABLES"
        if schema:
            query += f" IN SCHEMA {schema}"

        result = self.fetch(query, as_dataframe=False)
        return [row.get("name") for row in result if row.get("name")]

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
            Table name.
        schema : str, optional
            Schema name.

        Returns
        -------
        list of dict
            Column information.
        """
        schema = schema or self.config.schema
        full_name = f"{schema}.{table_name}" if schema else table_name

        query = f"DESCRIBE TABLE {full_name}"
        result = self.fetch(query, as_dataframe=False)

        columns = []
        for row in result:
            columns.append({
                "name": row.get("name"),
                "type": row.get("type"),
                "nullable": row.get("null?") == "Y",
                "default": row.get("default"),
                "primary_key": row.get("primary key") == "Y",
            })
        return columns

    def write(
        self,
        data: "pd.DataFrame",
        table_name: str,
        schema: Optional[str] = None,
        if_exists: str = "append",
        **kwargs,
    ) -> int:
        """
        Write a DataFrame to Snowflake.

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
        from snowflake.connector.pandas_tools import write_pandas

        if not self.is_connected:
            self.connect()

        schema = schema or self.config.schema

        # Handle if_exists
        full_name = f"{schema}.{table_name}" if schema else table_name
        if if_exists == "replace":
            try:
                self.execute(f"DROP TABLE IF EXISTS {full_name}")
            except Exception:
                pass

        success, num_chunks, num_rows, output = write_pandas(
            conn=self._connection,
            df=data,
            table_name=table_name,
            schema=schema,
            auto_create_table=True,
            **kwargs,
        )

        logger.info(f"Wrote {num_rows} rows to {full_name}")
        return num_rows

    def use_warehouse(self, warehouse: str) -> None:
        """Switch to a different warehouse."""
        self.execute(f"USE WAREHOUSE {warehouse}")
        self.config.warehouse = warehouse

    def use_database(self, database: str) -> None:
        """Switch to a different database."""
        self.execute(f"USE DATABASE {database}")
        self.config.database = database

    def use_schema(self, schema: str) -> None:
        """Switch to a different schema."""
        self.execute(f"USE SCHEMA {schema}")
        self.config.schema = schema
