"""
Amazon S3 storage connector.

This module provides a connector for Amazon S3 cloud storage.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import io

from ai_data_science_team.connectors.base import (
    DataConnector,
    ConnectionConfig,
    QueryResult,
)

logger = logging.getLogger(__name__)


class S3Connector(DataConnector):
    """
    Connector for Amazon S3 cloud storage.

    Parameters
    ----------
    config : ConnectionConfig, optional
        Connection configuration.
    bucket : str, optional
        Default S3 bucket name.
    region : str, optional
        AWS region.
    access_key : str, optional
        AWS access key ID.
    secret_key : str, optional
        AWS secret access key.
    profile : str, optional
        AWS profile name from credentials file.
    auto_connect : bool, default True
        Connect automatically on initialization.

    Example
    -------
    >>> connector = S3Connector(bucket="my-bucket", region="us-east-1")
    >>> df = connector.read_csv("path/to/data.csv")
    >>> connector.write_parquet(df, "path/to/output.parquet")
    """

    def __init__(
        self,
        config: Optional[ConnectionConfig] = None,
        bucket: Optional[str] = None,
        region: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        profile: Optional[str] = None,
        auto_connect: bool = True,
        **kwargs,
    ):
        if config is None:
            extra = kwargs.copy()
            if profile:
                extra["profile"] = profile

            config = ConnectionConfig(
                database=bucket,  # Map bucket to database field
                region=region,
                username=access_key,
                password=secret_key,
                extra=extra,
            )

        self._client = None
        self._resource = None
        super().__init__(config, auto_connect=False)

        if auto_connect:
            self.connect()

    @property
    def bucket(self) -> Optional[str]:
        """Get default bucket name."""
        return self.config.database

    @property
    def region(self) -> Optional[str]:
        """Get AWS region."""
        return self.config.region

    def connect(self) -> bool:
        """
        Establish connection to S3.

        Returns
        -------
        bool
            True if connection successful.

        Raises
        ------
        ImportError
            If boto3 is not installed.
        """
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for S3Connector. "
                "Install with: pip install boto3"
            )

        try:
            session_kwargs = {}
            if self.config.extra.get("profile"):
                session_kwargs["profile_name"] = self.config.extra["profile"]
            if self.config.region:
                session_kwargs["region_name"] = self.config.region

            session = boto3.Session(**session_kwargs)

            client_kwargs = {}
            if self.config.username and self.config.password:
                client_kwargs["aws_access_key_id"] = self.config.username
                client_kwargs["aws_secret_access_key"] = self.config.password

            self._client = session.client("s3", **client_kwargs)
            self._resource = session.resource("s3", **client_kwargs)
            self._connected = True
            logger.info("Connected to S3")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to S3: {e}")
            self._connected = False
            raise

    def disconnect(self) -> None:
        """Close the S3 connection."""
        self._client = None
        self._resource = None
        self._connected = False

    def execute(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """
        Execute S3 Select query on a file.

        Parameters
        ----------
        query : str
            SQL query for S3 Select.
        params : dict, optional
            Must include 'bucket' and 'key'.

        Returns
        -------
        QueryResult
            Query result.
        """
        if not params or "key" not in params:
            raise ValueError("params must include 'key'")

        bucket = params.get("bucket", self.bucket)
        key = params["key"]
        input_format = params.get("format", "CSV")

        if not bucket:
            raise ValueError("Bucket not specified")

        import time
        start_time = time.time()

        # Build input serialization
        if input_format.upper() == "CSV":
            input_ser = {"CSV": {"FileHeaderInfo": "USE"}}
        elif input_format.upper() == "JSON":
            input_ser = {"JSON": {"Type": "DOCUMENT"}}
        elif input_format.upper() == "PARQUET":
            input_ser = {"Parquet": {}}
        else:
            input_ser = {"CSV": {}}

        response = self._client.select_object_content(
            Bucket=bucket,
            Key=key,
            Expression=query,
            ExpressionType="SQL",
            InputSerialization=input_ser,
            OutputSerialization={"JSON": {}},
        )

        # Collect results
        records = []
        for event in response["Payload"]:
            if "Records" in event:
                records.append(event["Records"]["Payload"].decode("utf-8"))

        execution_time = time.time() - start_time

        return QueryResult(
            data="".join(records),
            rows_affected=len(records),
            execution_time=execution_time,
            query=query,
        )

    def fetch(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        as_dataframe: bool = True,
    ) -> Union[Any, "pd.DataFrame"]:
        """
        Fetch data using S3 Select.

        Parameters
        ----------
        query : str
            SQL query.
        params : dict, optional
            Must include 'key'.
        as_dataframe : bool, default True
            Return as DataFrame.

        Returns
        -------
        DataFrame or list
            Query results.
        """
        result = self.execute(query, params)

        if as_dataframe:
            import pandas as pd
            import json

            lines = result.data.strip().split("\n")
            records = [json.loads(line) for line in lines if line]
            return pd.DataFrame(records)
        else:
            import json
            lines = result.data.strip().split("\n")
            return [json.loads(line) for line in lines if line]

    def read_csv(
        self,
        key: str,
        bucket: Optional[str] = None,
        **kwargs,
    ) -> "pd.DataFrame":
        """
        Read a CSV file from S3.

        Parameters
        ----------
        key : str
            S3 object key.
        bucket : str, optional
            Bucket name.
        **kwargs
            Additional arguments for pandas.read_csv.

        Returns
        -------
        DataFrame
            Loaded data.
        """
        import pandas as pd

        bucket = bucket or self.bucket
        if not bucket:
            raise ValueError("Bucket not specified")

        obj = self._client.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(obj["Body"], **kwargs)

    def read_parquet(
        self,
        key: str,
        bucket: Optional[str] = None,
        **kwargs,
    ) -> "pd.DataFrame":
        """
        Read a Parquet file from S3.

        Parameters
        ----------
        key : str
            S3 object key.
        bucket : str, optional
            Bucket name.
        **kwargs
            Additional arguments for pandas.read_parquet.

        Returns
        -------
        DataFrame
            Loaded data.
        """
        import pandas as pd

        bucket = bucket or self.bucket
        if not bucket:
            raise ValueError("Bucket not specified")

        s3_path = f"s3://{bucket}/{key}"
        return pd.read_parquet(s3_path, **kwargs)

    def read_json(
        self,
        key: str,
        bucket: Optional[str] = None,
        lines: bool = True,
        **kwargs,
    ) -> "pd.DataFrame":
        """
        Read a JSON file from S3.

        Parameters
        ----------
        key : str
            S3 object key.
        bucket : str, optional
            Bucket name.
        lines : bool, default True
            Whether file is JSON lines format.
        **kwargs
            Additional arguments for pandas.read_json.

        Returns
        -------
        DataFrame
            Loaded data.
        """
        import pandas as pd

        bucket = bucket or self.bucket
        if not bucket:
            raise ValueError("Bucket not specified")

        obj = self._client.get_object(Bucket=bucket, Key=key)
        return pd.read_json(obj["Body"], lines=lines, **kwargs)

    def write_csv(
        self,
        data: "pd.DataFrame",
        key: str,
        bucket: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Write a DataFrame to CSV in S3.

        Parameters
        ----------
        data : DataFrame
            Data to write.
        key : str
            S3 object key.
        bucket : str, optional
            Bucket name.
        **kwargs
            Additional arguments for DataFrame.to_csv.

        Returns
        -------
        str
            S3 URI of written file.
        """
        bucket = bucket or self.bucket
        if not bucket:
            raise ValueError("Bucket not specified")

        buffer = io.StringIO()
        data.to_csv(buffer, index=False, **kwargs)
        buffer.seek(0)

        self._client.put_object(
            Bucket=bucket,
            Key=key,
            Body=buffer.getvalue(),
        )

        logger.info(f"Wrote {len(data)} rows to s3://{bucket}/{key}")
        return f"s3://{bucket}/{key}"

    def write_parquet(
        self,
        data: "pd.DataFrame",
        key: str,
        bucket: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Write a DataFrame to Parquet in S3.

        Parameters
        ----------
        data : DataFrame
            Data to write.
        key : str
            S3 object key.
        bucket : str, optional
            Bucket name.
        **kwargs
            Additional arguments for DataFrame.to_parquet.

        Returns
        -------
        str
            S3 URI of written file.
        """
        bucket = bucket or self.bucket
        if not bucket:
            raise ValueError("Bucket not specified")

        buffer = io.BytesIO()
        data.to_parquet(buffer, index=False, **kwargs)
        buffer.seek(0)

        self._client.put_object(
            Bucket=bucket,
            Key=key,
            Body=buffer.getvalue(),
        )

        logger.info(f"Wrote {len(data)} rows to s3://{bucket}/{key}")
        return f"s3://{bucket}/{key}"

    def write(
        self,
        data: "pd.DataFrame",
        table_name: str,
        schema: Optional[str] = None,
        if_exists: str = "append",
        format: str = "parquet",
        **kwargs,
    ) -> int:
        """
        Write a DataFrame to S3.

        Parameters
        ----------
        data : DataFrame
            Data to write.
        table_name : str
            Object key (path).
        schema : str, optional
            Bucket name.
        if_exists : str, default 'append'
            Not used for S3.
        format : str, default 'parquet'
            Output format (parquet, csv, json).

        Returns
        -------
        int
            Number of rows written.
        """
        bucket = schema or self.bucket

        if format.lower() == "parquet":
            self.write_parquet(data, table_name, bucket, **kwargs)
        elif format.lower() == "csv":
            self.write_csv(data, table_name, bucket, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return len(data)

    def list_objects(
        self,
        prefix: str = "",
        bucket: Optional[str] = None,
        max_keys: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        List objects in a bucket.

        Parameters
        ----------
        prefix : str
            Key prefix filter.
        bucket : str, optional
            Bucket name.
        max_keys : int, default 1000
            Maximum number of keys to return.

        Returns
        -------
        list of dict
            Object metadata.
        """
        bucket = bucket or self.bucket
        if not bucket:
            raise ValueError("Bucket not specified")

        response = self._client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            MaxKeys=max_keys,
        )

        objects = []
        for obj in response.get("Contents", []):
            objects.append({
                "key": obj["Key"],
                "size": obj["Size"],
                "last_modified": obj["LastModified"],
                "storage_class": obj.get("StorageClass"),
            })

        return objects

    def delete_object(self, key: str, bucket: Optional[str] = None) -> bool:
        """
        Delete an object from S3.

        Parameters
        ----------
        key : str
            Object key.
        bucket : str, optional
            Bucket name.

        Returns
        -------
        bool
            True if deleted.
        """
        bucket = bucket or self.bucket
        if not bucket:
            raise ValueError("Bucket not specified")

        self._client.delete_object(Bucket=bucket, Key=key)
        logger.info(f"Deleted s3://{bucket}/{key}")
        return True

    def get_tables(self, schema: Optional[str] = None) -> List[str]:
        """Get list of objects (tables are keys)."""
        bucket = schema or self.bucket
        objects = self.list_objects(bucket=bucket)
        return [obj["key"] for obj in objects]

    def copy_object(
        self,
        source_key: str,
        dest_key: str,
        source_bucket: Optional[str] = None,
        dest_bucket: Optional[str] = None,
    ) -> str:
        """
        Copy an object within S3.

        Parameters
        ----------
        source_key : str
            Source object key.
        dest_key : str
            Destination object key.
        source_bucket : str, optional
            Source bucket.
        dest_bucket : str, optional
            Destination bucket.

        Returns
        -------
        str
            URI of copied object.
        """
        source_bucket = source_bucket or self.bucket
        dest_bucket = dest_bucket or self.bucket

        if not source_bucket or not dest_bucket:
            raise ValueError("Bucket not specified")

        self._client.copy_object(
            CopySource={"Bucket": source_bucket, "Key": source_key},
            Bucket=dest_bucket,
            Key=dest_key,
        )

        logger.info(f"Copied s3://{source_bucket}/{source_key} to s3://{dest_bucket}/{dest_key}")
        return f"s3://{dest_bucket}/{dest_key}"
