"""
Amazon Redshift connector.

This module provides a connector for Amazon Redshift.
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


class RedshiftConnector(DataConnector):
    """
    Connector for Amazon Redshift.

    Parameters
    ----------
    config : ConnectionConfig, optional
        Connection configuration.
    host : str, optional
        Redshift cluster endpoint.
    port : int, default 5439
        Port number.
    database : str, optional
        Database name.
    user : str, optional
        Username.
    password : str, optional
        Password.
    schema : str, optional
        Default schema.
    auto_connect : bool, default True
        Connect automatically on initialization.

    Example
    -------
    >>> connector = RedshiftConnector(
    ...     host="my-cluster.xxx.region.redshift.amazonaws.com",
    ...     database="mydb",
    ...     user="admin",
    ...     password="password",
    ... )
    >>> df = connector.fetch("SELECT * FROM mytable")
    """

    def __init__(
        self,
        config: Optional[ConnectionConfig] = None,
        host: Optional[str] = None,
        port: int = 5439,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        schema: Optional[str] = None,
        auto_connect: bool = True,
        **kwargs,
    ):
        if config is None:
            config = ConnectionConfig(
                host=host,
                port=port,
                database=database,
                username=user,
                password=password,
                schema=schema,
                extra=kwargs,
            )

        super().__init__(config, auto_connect=False)

        if auto_connect:
            self.connect()

    def connect(self) -> bool:
        """
        Establish connection to Redshift.

        Returns
        -------
        bool
            True if connection successful.

        Raises
        ------
        ImportError
            If redshift_connector is not installed.
        """
        try:
            import redshift_connector
        except ImportError:
            raise ImportError(
                "redshift_connector is required for RedshiftConnector. "
                "Install with: pip install redshift-connector"
            )

        try:
            conn_params = {
                "host": self.config.host,
                "port": self.config.port or 5439,
                "database": self.config.database,
                "user": self.config.username,
                "password": self.config.password,
            }

            # Add any extra parameters
            conn_params.update(self.config.extra)

            self._connection = redshift_connector.connect(**conn_params)
            self._connected = True
            logger.info(f"Connected to Redshift: {self.config.host}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Redshift: {e}")
            self._connected = False
            raise

    def disconnect(self) -> None:
        """Close the Redshift connection."""
        if self._connection:
            try:
                self._connection.close()
            except Exception as e:
                logger.warning(f"Error closing Redshift connection: {e}")
            finally:
                self._connection = None
                self._connected = False

    def execute(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """
        Execute a query on Redshift.

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

            self._connection.commit()
            rows_affected = cursor.rowcount or 0
            execution_time = time.time() - start_time

            return QueryResult(
                data=None,
                rows_affected=rows_affected,
                execution_time=execution_time,
                query=query,
            )

        except Exception as e:
            self._connection.rollback()
            logger.error(f"Redshift query failed: {e}")
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
        Fetch data from Redshift.

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
        schema = schema or self.config.schema or "public"
        query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """
        result = self.fetch(query, {"schema": schema}, as_dataframe=False)
        return [row["table_name"] for row in result]

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
        schema = schema or self.config.schema or "public"
        query = """
            SELECT
                column_name,
                data_type,
                is_nullable,
                column_default,
                character_maximum_length
            FROM information_schema.columns
            WHERE table_schema = %s
            AND table_name = %s
            ORDER BY ordinal_position
        """
        result = self.fetch(query, {"schema": schema, "table": table_name}, as_dataframe=False)

        columns = []
        for row in result:
            columns.append({
                "name": row["column_name"],
                "type": row["data_type"],
                "nullable": row["is_nullable"] == "YES",
                "default": row["column_default"],
                "max_length": row["character_maximum_length"],
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
        Write a DataFrame to Redshift.

        Parameters
        ----------
        data : DataFrame
            Data to write.
        table_name : str
            Target table name.
        schema : str, optional
            Target schema.
        if_exists : str, default 'append'
            Behavior if table exists.

        Returns
        -------
        int
            Number of rows written.
        """
        if not self.is_connected:
            self.connect()

        schema = schema or self.config.schema or "public"
        full_name = f"{schema}.{table_name}"

        cursor = self._connection.cursor()

        try:
            # Handle if_exists
            if if_exists == "replace":
                cursor.execute(f"DROP TABLE IF EXISTS {full_name}")
                self._connection.commit()
            elif if_exists == "fail":
                cursor.execute(f"""
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = '{schema}' AND table_name = '{table_name}'
                """)
                if cursor.fetchone():
                    raise ValueError(f"Table {full_name} already exists")

            # Create table if needed
            if if_exists in ("replace", "fail") or not self.table_exists(table_name, schema):
                columns_sql = ", ".join([
                    f'"{col}" VARCHAR(65535)' for col in data.columns
                ])
                cursor.execute(f"CREATE TABLE {full_name} ({columns_sql})")
                self._connection.commit()

            # Insert data
            columns = ", ".join([f'"{col}"' for col in data.columns])
            placeholders = ", ".join(["%s"] * len(data.columns))
            insert_query = f"INSERT INTO {full_name} ({columns}) VALUES ({placeholders})"

            for _, row in data.iterrows():
                cursor.execute(insert_query, tuple(row))

            self._connection.commit()
            logger.info(f"Wrote {len(data)} rows to {full_name}")
            return len(data)

        except Exception as e:
            self._connection.rollback()
            raise

        finally:
            cursor.close()

    def copy_from_s3(
        self,
        table_name: str,
        s3_path: str,
        iam_role: str,
        schema: Optional[str] = None,
        format: str = "CSV",
        **kwargs,
    ) -> int:
        """
        Load data from S3 using COPY command.

        Parameters
        ----------
        table_name : str
            Target table name.
        s3_path : str
            S3 path (e.g., 's3://bucket/path/').
        iam_role : str
            IAM role ARN for S3 access.
        schema : str, optional
            Target schema.
        format : str, default 'CSV'
            File format (CSV, JSON, PARQUET, etc.).

        Returns
        -------
        int
            Number of rows loaded.
        """
        schema = schema or self.config.schema or "public"
        full_name = f"{schema}.{table_name}"

        options = []
        if format.upper() == "CSV":
            options.append("CSV")
            if kwargs.get("header"):
                options.append("IGNOREHEADER 1")
        elif format.upper() == "JSON":
            options.append("JSON 'auto'")
        elif format.upper() == "PARQUET":
            options.append("FORMAT AS PARQUET")

        options_str = " ".join(options)

        query = f"""
            COPY {full_name}
            FROM '{s3_path}'
            IAM_ROLE '{iam_role}'
            {options_str}
        """

        result = self.execute(query)
        return result.rows_affected
