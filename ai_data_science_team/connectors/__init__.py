"""
Cloud and database connectors for AI Data Science Team.

This module provides unified interfaces for connecting to various
data sources including cloud data warehouses and databases.
"""

from ai_data_science_team.connectors.base import (
    DataConnector,
    ConnectionConfig,
    QueryResult,
)
from ai_data_science_team.connectors.snowflake import SnowflakeConnector
from ai_data_science_team.connectors.bigquery import BigQueryConnector
from ai_data_science_team.connectors.redshift import RedshiftConnector
from ai_data_science_team.connectors.postgres import PostgresConnector
from ai_data_science_team.connectors.s3 import S3Connector
from ai_data_science_team.connectors.factory import (
    get_connector,
    register_connector,
    list_connectors,
)

__all__ = [
    # Base classes
    "DataConnector",
    "ConnectionConfig",
    "QueryResult",
    # Cloud connectors
    "SnowflakeConnector",
    "BigQueryConnector",
    "RedshiftConnector",
    "PostgresConnector",
    "S3Connector",
    # Factory functions
    "get_connector",
    "register_connector",
    "list_connectors",
]
