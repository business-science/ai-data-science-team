"""
Google BigQuery connector.

This module provides a connector for Google BigQuery.
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


class BigQueryConnector(DataConnector):
    """
    Connector for Google BigQuery.

    Parameters
    ----------
    config : ConnectionConfig, optional
        Connection configuration.
    project : str, optional
        Google Cloud project ID.
    credentials_path : str, optional
        Path to service account JSON file.
    location : str, optional
        BigQuery location (e.g., 'US', 'EU').
    auto_connect : bool, default True
        Connect automatically on initialization.

    Example
    -------
    >>> connector = BigQueryConnector(
    ...     project="my-project",
    ...     credentials_path="/path/to/credentials.json",
    ... )
    >>> df = connector.fetch("SELECT * FROM `my-project.dataset.table`")
    """

    def __init__(
        self,
        config: Optional[ConnectionConfig] = None,
        project: Optional[str] = None,
        credentials_path: Optional[str] = None,
        location: Optional[str] = None,
        dataset: Optional[str] = None,
        auto_connect: bool = True,
        **kwargs,
    ):
        if config is None:
            extra = kwargs.copy()
            if location:
                extra["location"] = location

            config = ConnectionConfig(
                project=project,
                credentials_path=credentials_path,
                database=dataset,  # Map dataset to database
                extra=extra,
            )

        self._client = None
        super().__init__(config, auto_connect=False)

        if auto_connect:
            self.connect()

    @property
    def project(self) -> Optional[str]:
        """Get project ID."""
        return self.config.project

    @property
    def dataset(self) -> Optional[str]:
        """Get default dataset."""
        return self.config.database

    @property
    def location(self) -> Optional[str]:
        """Get BigQuery location."""
        return self.config.extra.get("location")

    def connect(self) -> bool:
        """
        Establish connection to BigQuery.

        Returns
        -------
        bool
            True if connection successful.

        Raises
        ------
        ImportError
            If google-cloud-bigquery is not installed.
        """
        try:
            from google.cloud import bigquery
            from google.oauth2 import service_account
        except ImportError:
            raise ImportError(
                "google-cloud-bigquery is required for BigQueryConnector. "
                "Install with: pip install google-cloud-bigquery"
            )

        try:
            client_kwargs = {}

            if self.config.project:
                client_kwargs["project"] = self.config.project

            if self.config.credentials_path:
                credentials = service_account.Credentials.from_service_account_file(
                    self.config.credentials_path
                )
                client_kwargs["credentials"] = credentials

            if self.location:
                client_kwargs["location"] = self.location

            self._client = bigquery.Client(**client_kwargs)
            self._connected = True
            logger.info(f"Connected to BigQuery project: {self._client.project}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to BigQuery: {e}")
            self._connected = False
            raise

    def disconnect(self) -> None:
        """Close the BigQuery connection."""
        if self._client:
            try:
                self._client.close()
            except Exception as e:
                logger.warning(f"Error closing BigQuery connection: {e}")
            finally:
                self._client = None
                self._connected = False

    def execute(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """
        Execute a query on BigQuery.

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

        from google.cloud import bigquery

        start_time = time.time()

        job_config = bigquery.QueryJobConfig()
        if params:
            job_config.query_parameters = [
                bigquery.ScalarQueryParameter(k, self._infer_type(v), v)
                for k, v in params.items()
            ]

        query_job = self._client.query(query, job_config=job_config)
        query_job.result()  # Wait for completion

        execution_time = time.time() - start_time

        return QueryResult(
            data=None,
            rows_affected=query_job.num_dml_affected_rows or 0,
            execution_time=execution_time,
            query=query,
            metadata={
                "job_id": query_job.job_id,
                "bytes_processed": query_job.total_bytes_processed,
                "bytes_billed": query_job.total_bytes_billed,
            },
        )

    def fetch(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        as_dataframe: bool = True,
    ) -> Union[Any, "pd.DataFrame"]:
        """
        Fetch data from BigQuery.

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

        from google.cloud import bigquery

        job_config = bigquery.QueryJobConfig()
        if params:
            job_config.query_parameters = [
                bigquery.ScalarQueryParameter(k, self._infer_type(v), v)
                for k, v in params.items()
            ]

        query_job = self._client.query(query, job_config=job_config)

        if as_dataframe:
            return query_job.to_dataframe()
        else:
            results = query_job.result()
            return [dict(row) for row in results]

    def _infer_type(self, value: Any) -> str:
        """Infer BigQuery type from Python value."""
        if isinstance(value, bool):
            return "BOOL"
        elif isinstance(value, int):
            return "INT64"
        elif isinstance(value, float):
            return "FLOAT64"
        elif isinstance(value, str):
            return "STRING"
        else:
            return "STRING"

    def get_tables(self, schema: Optional[str] = None) -> List[str]:
        """
        Get list of tables in a dataset.

        Parameters
        ----------
        schema : str, optional
            Dataset name. Uses default dataset if not provided.

        Returns
        -------
        list of str
            Table names.
        """
        dataset = schema or self.dataset
        if not dataset:
            raise ValueError("Dataset not specified")

        tables = self._client.list_tables(f"{self.project}.{dataset}")
        return [table.table_id for table in tables]

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
            Dataset name.

        Returns
        -------
        list of dict
            Column information.
        """
        dataset = schema or self.dataset
        if not dataset:
            raise ValueError("Dataset not specified")

        table_ref = f"{self.project}.{dataset}.{table_name}"
        table = self._client.get_table(table_ref)

        columns = []
        for field in table.schema:
            columns.append({
                "name": field.name,
                "type": field.field_type,
                "mode": field.mode,
                "description": field.description,
                "nullable": field.mode != "REQUIRED",
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
        Write a DataFrame to BigQuery.

        Parameters
        ----------
        data : DataFrame
            Data to write.
        table_name : str
            Target table name.
        schema : str, optional
            Target dataset.
        if_exists : str, default 'append'
            Behavior if table exists ('fail', 'replace', 'append').

        Returns
        -------
        int
            Number of rows written.
        """
        from google.cloud import bigquery

        if not self.is_connected:
            self.connect()

        dataset = schema or self.dataset
        if not dataset:
            raise ValueError("Dataset not specified")

        table_ref = f"{self.project}.{dataset}.{table_name}"

        job_config = bigquery.LoadJobConfig()

        if if_exists == "replace":
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
        elif if_exists == "append":
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
        elif if_exists == "fail":
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_EMPTY

        job = self._client.load_table_from_dataframe(
            data,
            table_ref,
            job_config=job_config,
        )
        job.result()  # Wait for completion

        logger.info(f"Wrote {len(data)} rows to {table_ref}")
        return len(data)

    def get_datasets(self) -> List[str]:
        """Get list of datasets in the project."""
        datasets = self._client.list_datasets()
        return [ds.dataset_id for ds in datasets]

    def create_dataset(
        self,
        dataset_name: str,
        location: Optional[str] = None,
        exists_ok: bool = True,
    ) -> None:
        """
        Create a new dataset.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to create.
        location : str, optional
            Location for the dataset.
        exists_ok : bool, default True
            Don't raise error if dataset exists.
        """
        from google.cloud import bigquery

        dataset_ref = f"{self.project}.{dataset_name}"
        dataset = bigquery.Dataset(dataset_ref)

        if location:
            dataset.location = location
        elif self.location:
            dataset.location = self.location

        self._client.create_dataset(dataset, exists_ok=exists_ok)
        logger.info(f"Created dataset: {dataset_ref}")
