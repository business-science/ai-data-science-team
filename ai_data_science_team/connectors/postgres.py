"""
PostgreSQL database connector.

This module provides a connector for PostgreSQL databases.
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


class PostgresConnector(DataConnector):
    """
    Connector for PostgreSQL databases.

    Parameters
    ----------
    config : ConnectionConfig, optional
        Connection configuration.
    host : str, default 'localhost'
        Database host.
    port : int, default 5432
        Port number.
    database : str, optional
        Database name.
    user : str, optional
        Username.
    password : str, optional
        Password.
    schema : str, default 'public'
        Default schema.
    auto_connect : bool, default True
        Connect automatically on initialization.

    Example
    -------
    >>> connector = PostgresConnector(
    ...     host="localhost",
    ...     database="mydb",
    ...     user="postgres",
    ...     password="password",
    ... )
    >>> df = connector.fetch("SELECT * FROM mytable")
    """

    def __init__(
        self,
        config: Optional[ConnectionConfig] = None,
        host: str = "localhost",
        port: int = 5432,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        schema: str = "public",
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
        Establish connection to PostgreSQL.

        Returns
        -------
        bool
            True if connection successful.

        Raises
        ------
        ImportError
            If psycopg2 is not installed.
        """
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "psycopg2 is required for PostgresConnector. "
                "Install with: pip install psycopg2-binary"
            )

        try:
            conn_params = {
                "host": self.config.host,
                "port": self.config.port or 5432,
                "dbname": self.config.database,
                "user": self.config.username,
                "password": self.config.password,
            }

            # Add any extra parameters
            conn_params.update(self.config.extra)

            self._connection = psycopg2.connect(**conn_params)
            self._connected = True
            logger.info(f"Connected to PostgreSQL: {self.config.host}:{self.config.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            self._connected = False
            raise

    def disconnect(self) -> None:
        """Close the PostgreSQL connection."""
        if self._connection:
            try:
                self._connection.close()
            except Exception as e:
                logger.warning(f"Error closing PostgreSQL connection: {e}")
            finally:
                self._connection = None
                self._connected = False

    def execute(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """
        Execute a query on PostgreSQL.

        Parameters
        ----------
        query : str
            SQL query to execute.
        params : dict or tuple, optional
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
            logger.error(f"PostgreSQL query failed: {e}")
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
        Fetch data from PostgreSQL.

        Parameters
        ----------
        query : str
            SQL query to execute.
        params : dict or tuple, optional
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
        result = self.fetch(query, (schema,), as_dataframe=False)
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
        result = self.fetch(query, (schema, table_name), as_dataframe=False)

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
        Write a DataFrame to PostgreSQL.

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
        from sqlalchemy import create_engine

        if not self.is_connected:
            self.connect()

        schema = schema or self.config.schema or "public"

        # Create SQLAlchemy engine for pandas to_sql
        engine_url = (
            f"postgresql://{self.config.username}:{self.config.password}@"
            f"{self.config.host}:{self.config.port}/{self.config.database}"
        )
        engine = create_engine(engine_url)

        try:
            data.to_sql(
                table_name,
                engine,
                schema=schema,
                if_exists=if_exists,
                index=False,
                **kwargs,
            )
            logger.info(f"Wrote {len(data)} rows to {schema}.{table_name}")
            return len(data)
        finally:
            engine.dispose()

    def copy_from_csv(
        self,
        table_name: str,
        file_path: str,
        schema: Optional[str] = None,
        delimiter: str = ",",
        header: bool = True,
        **kwargs,
    ) -> int:
        """
        Load data from CSV using COPY command (fast bulk load).

        Parameters
        ----------
        table_name : str
            Target table name.
        file_path : str
            Path to CSV file.
        schema : str, optional
            Target schema.
        delimiter : str, default ','
            Column delimiter.
        header : bool, default True
            Whether file has header row.

        Returns
        -------
        int
            Number of rows loaded.
        """
        if not self.is_connected:
            self.connect()

        schema = schema or self.config.schema or "public"
        full_name = f"{schema}.{table_name}"

        cursor = self._connection.cursor()

        try:
            with open(file_path, "r") as f:
                cursor.copy_expert(
                    f"COPY {full_name} FROM STDIN WITH CSV HEADER DELIMITER '{delimiter}'",
                    f,
                )
            self._connection.commit()
            return cursor.rowcount
        except Exception as e:
            self._connection.rollback()
            raise
        finally:
            cursor.close()

    def get_schemas(self) -> List[str]:
        """Get list of schemas in the database."""
        query = """
            SELECT schema_name
            FROM information_schema.schemata
            WHERE schema_name NOT LIKE 'pg_%'
            AND schema_name != 'information_schema'
            ORDER BY schema_name
        """
        result = self.fetch(query, as_dataframe=False)
        return [row["schema_name"] for row in result]
