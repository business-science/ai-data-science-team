"""
Unit tests for cloud and database connectors.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from ai_data_science_team.connectors import (
    ConnectionConfig,
    QueryResult,
    get_connector,
    register_connector,
    list_connectors,
)
from ai_data_science_team.connectors.base import (
    DataConnector,
    MockConnector,
)
from ai_data_science_team.connectors.factory import (
    get_connector_from_url,
    ConnectorPool,
    _CONNECTOR_REGISTRY,
)


class TestConnectionConfig:
    """Tests for ConnectionConfig."""

    def test_config_creation(self):
        """Test creating a connection config."""
        config = ConnectionConfig(
            host="localhost",
            port=5432,
            database="testdb",
            username="user",
            password="pass",
        )

        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "testdb"
        assert config.username == "user"
        assert config.password == "pass"

    def test_config_to_dict(self):
        """Test converting config to dict."""
        config = ConnectionConfig(
            host="localhost",
            database="testdb",
        )

        d = config.to_dict()

        assert d["host"] == "localhost"
        assert d["database"] == "testdb"
        assert "port" not in d  # None values excluded

    def test_config_from_dict(self):
        """Test creating config from dict."""
        d = {
            "host": "localhost",
            "port": 5432,
            "database": "testdb",
            "custom_param": "value",
        }

        config = ConnectionConfig.from_dict(d)

        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "testdb"
        assert config.extra["custom_param"] == "value"

    def test_config_from_env(self):
        """Test creating config from environment."""
        import os

        os.environ["TEST_HOST"] = "envhost"
        os.environ["TEST_PORT"] = "1234"
        os.environ["TEST_DATABASE"] = "envdb"

        try:
            config = ConnectionConfig.from_env(prefix="TEST_")

            assert config.host == "envhost"
            assert config.port == 1234
            assert config.database == "envdb"
        finally:
            del os.environ["TEST_HOST"]
            del os.environ["TEST_PORT"]
            del os.environ["TEST_DATABASE"]


class TestQueryResult:
    """Tests for QueryResult."""

    def test_query_result_creation(self):
        """Test creating a query result."""
        result = QueryResult(
            data=[{"a": 1}, {"a": 2}],
            rows_affected=2,
            execution_time=0.5,
            query="SELECT * FROM table",
        )

        assert len(result.data) == 2
        assert result.rows_affected == 2
        assert result.execution_time == 0.5
        assert result.success is True

    def test_query_result_metadata(self):
        """Test query result with metadata."""
        result = QueryResult(
            data=None,
            rows_affected=10,
            metadata={"query_id": "abc123"},
        )

        assert result.metadata["query_id"] == "abc123"


class TestMockConnector:
    """Tests for MockConnector."""

    def test_mock_connector_basic(self):
        """Test basic mock connector operations."""
        connector = MockConnector()

        assert connector.is_connected is True
        assert connector.ping() is True

    def test_mock_connector_write_read(self):
        """Test mock connector write and table tracking."""
        connector = MockConnector()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        rows = connector.write(df, "test_table")

        assert rows == 3
        assert "test_table" in connector.get_tables()

    def test_mock_connector_execute(self):
        """Test mock connector execute."""
        connector = MockConnector()

        result = connector.execute("SELECT 1")

        assert result.query == "SELECT 1"
        assert result.rows_affected == 0

    def test_mock_connector_context_manager(self):
        """Test mock connector as context manager."""
        with MockConnector() as connector:
            assert connector.is_connected is True

        assert connector.is_connected is False


class TestConnectorFactory:
    """Tests for connector factory functions."""

    def test_list_connectors(self):
        """Test listing available connectors."""
        connectors = list_connectors()

        assert "snowflake" in connectors
        assert "bigquery" in connectors
        assert "postgres" in connectors
        assert "s3" in connectors
        assert "mock" in connectors

    def test_get_mock_connector(self):
        """Test getting mock connector."""
        connector = get_connector("mock")

        assert isinstance(connector, MockConnector)
        assert connector.is_connected is True

    def test_register_custom_connector(self):
        """Test registering a custom connector."""
        class CustomConnector(DataConnector):
            def connect(self):
                self._connected = True
                return True

            def disconnect(self):
                self._connected = False

            def execute(self, query, params=None):
                return QueryResult(data=None, rows_affected=0)

            def fetch(self, query, params=None, as_dataframe=True):
                return pd.DataFrame()

        register_connector("custom", CustomConnector)
        connector = get_connector("custom", auto_connect=False)

        assert isinstance(connector, CustomConnector)

        # Cleanup
        del _CONNECTOR_REGISTRY["custom"]

    def test_get_unknown_connector(self):
        """Test getting unknown connector type."""
        with pytest.raises(ValueError, match="Unknown connector type"):
            get_connector("unknown_type")


class TestConnectorFromUrl:
    """Tests for creating connectors from URLs."""

    def test_postgres_url(self):
        """Test parsing PostgreSQL URL."""
        with patch("ai_data_science_team.connectors.factory.get_connector") as mock_get:
            mock_get.return_value = MockConnector()

            get_connector_from_url("postgres://user:pass@localhost:5432/mydb")

            mock_get.assert_called_once()
            call_kwargs = mock_get.call_args[1]
            assert call_kwargs["host"] == "localhost"
            assert call_kwargs["port"] == 5432
            assert call_kwargs["user"] == "user"
            assert call_kwargs["database"] == "mydb"

    def test_s3_url(self):
        """Test parsing S3 URL."""
        with patch("ai_data_science_team.connectors.factory.get_connector") as mock_get:
            mock_get.return_value = MockConnector()

            get_connector_from_url("s3://my-bucket")

            mock_get.assert_called_once()
            call_kwargs = mock_get.call_args[1]
            assert call_kwargs["bucket"] == "my-bucket"

    def test_unknown_url_scheme(self):
        """Test unknown URL scheme."""
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            get_connector_from_url("unknown://host/db")


class TestConnectorPool:
    """Tests for ConnectorPool."""

    def test_pool_get_connector(self):
        """Test getting connector from pool."""
        pool = ConnectorPool()

        connector = pool.get("mock")

        assert isinstance(connector, MockConnector)
        assert connector.is_connected is True

    def test_pool_release_and_reuse(self):
        """Test releasing connector back to pool."""
        pool = ConnectorPool()

        # Get and release connector
        connector1 = pool.get("mock")
        pool.release(connector1)

        # Get again - should reuse
        connector2 = pool.get("mock")

        assert connector1 is connector2

    def test_pool_close_all(self):
        """Test closing all connections."""
        pool = ConnectorPool()

        connector = pool.get("mock")
        pool.close_all()

        # Pool should be empty
        assert connector.is_connected is False

    def test_pool_context_manager(self):
        """Test pool as context manager."""
        with ConnectorPool() as pool:
            connector = pool.get("mock")
            assert connector.is_connected is True

        # All connections closed after exit
        assert connector.is_connected is False


class TestSnowflakeConnector:
    """Tests for SnowflakeConnector (mocked)."""

    def test_snowflake_init_params(self):
        """Test Snowflake connector initialization."""
        from ai_data_science_team.connectors.snowflake import SnowflakeConnector

        with patch("ai_data_science_team.connectors.snowflake.SnowflakeConnector.connect"):
            connector = SnowflakeConnector(
                account="myaccount",
                user="myuser",
                password="mypassword",
                database="mydb",
                warehouse="mywh",
                auto_connect=False,
            )

            assert connector.account == "myaccount"
            assert connector.config.username == "myuser"
            assert connector.config.database == "mydb"
            assert connector.config.warehouse == "mywh"

    def test_snowflake_connect_import_error(self):
        """Test Snowflake connector import error."""
        from ai_data_science_team.connectors.snowflake import SnowflakeConnector

        connector = SnowflakeConnector(auto_connect=False)

        with patch.dict("sys.modules", {"snowflake": None, "snowflake.connector": None}):
            with pytest.raises(ImportError, match="snowflake-connector-python is required"):
                connector.connect()


class TestBigQueryConnector:
    """Tests for BigQueryConnector (mocked)."""

    def test_bigquery_init_params(self):
        """Test BigQuery connector initialization."""
        from ai_data_science_team.connectors.bigquery import BigQueryConnector

        with patch("ai_data_science_team.connectors.bigquery.BigQueryConnector.connect"):
            connector = BigQueryConnector(
                project="my-project",
                dataset="my_dataset",
                location="US",
                auto_connect=False,
            )

            assert connector.project == "my-project"
            assert connector.dataset == "my_dataset"
            assert connector.location == "US"

    def test_bigquery_connect_import_error(self):
        """Test BigQuery connector import error."""
        from ai_data_science_team.connectors.bigquery import BigQueryConnector

        connector = BigQueryConnector(auto_connect=False)

        with patch.dict("sys.modules", {"google": None, "google.cloud": None}):
            with pytest.raises(ImportError, match="google-cloud-bigquery is required"):
                connector.connect()


class TestRedshiftConnector:
    """Tests for RedshiftConnector (mocked)."""

    def test_redshift_init_params(self):
        """Test Redshift connector initialization."""
        from ai_data_science_team.connectors.redshift import RedshiftConnector

        with patch("ai_data_science_team.connectors.redshift.RedshiftConnector.connect"):
            connector = RedshiftConnector(
                host="cluster.region.redshift.amazonaws.com",
                database="mydb",
                user="admin",
                password="password",
                auto_connect=False,
            )

            assert connector.config.host == "cluster.region.redshift.amazonaws.com"
            assert connector.config.database == "mydb"
            assert connector.config.port == 5439

    def test_redshift_connect_import_error(self):
        """Test Redshift connector import error."""
        from ai_data_science_team.connectors.redshift import RedshiftConnector

        connector = RedshiftConnector(auto_connect=False)

        with patch.dict("sys.modules", {"redshift_connector": None}):
            with pytest.raises(ImportError, match="redshift_connector is required"):
                connector.connect()


class TestPostgresConnector:
    """Tests for PostgresConnector (mocked)."""

    def test_postgres_init_params(self):
        """Test PostgreSQL connector initialization."""
        from ai_data_science_team.connectors.postgres import PostgresConnector

        with patch("ai_data_science_team.connectors.postgres.PostgresConnector.connect"):
            connector = PostgresConnector(
                host="localhost",
                port=5432,
                database="mydb",
                user="postgres",
                password="password",
                auto_connect=False,
            )

            assert connector.config.host == "localhost"
            assert connector.config.port == 5432
            assert connector.config.database == "mydb"

    def test_postgres_connect_import_error(self):
        """Test PostgreSQL connector import error."""
        from ai_data_science_team.connectors.postgres import PostgresConnector

        connector = PostgresConnector(auto_connect=False)

        with patch.dict("sys.modules", {"psycopg2": None}):
            with pytest.raises(ImportError, match="psycopg2 is required"):
                connector.connect()


class TestS3Connector:
    """Tests for S3Connector (mocked)."""

    def test_s3_init_params(self):
        """Test S3 connector initialization."""
        from ai_data_science_team.connectors.s3 import S3Connector

        with patch("ai_data_science_team.connectors.s3.S3Connector.connect"):
            connector = S3Connector(
                bucket="my-bucket",
                region="us-east-1",
                auto_connect=False,
            )

            assert connector.bucket == "my-bucket"
            assert connector.region == "us-east-1"

    def test_s3_connect_import_error(self):
        """Test S3 connector import error."""
        from ai_data_science_team.connectors.s3 import S3Connector

        connector = S3Connector(auto_connect=False)

        with patch.dict("sys.modules", {"boto3": None}):
            with pytest.raises(ImportError, match="boto3 is required"):
                connector.connect()

    def test_s3_connect_success(self):
        """Test successful S3 connection."""
        pytest.importorskip("boto3")
        from ai_data_science_team.connectors.s3 import S3Connector

        with patch("boto3.Session") as mock_session:
            mock_session.return_value.client.return_value = MagicMock()
            mock_session.return_value.resource.return_value = MagicMock()

            connector = S3Connector(bucket="test-bucket", auto_connect=True)

            assert connector.is_connected is True

    def test_s3_list_objects(self):
        """Test S3 list objects."""
        pytest.importorskip("boto3")
        from ai_data_science_team.connectors.s3 import S3Connector

        with patch("boto3.Session") as mock_session:
            mock_client = MagicMock()
            mock_client.list_objects_v2.return_value = {
                "Contents": [
                    {"Key": "file1.csv", "Size": 100, "LastModified": "2024-01-01"},
                    {"Key": "file2.csv", "Size": 200, "LastModified": "2024-01-02"},
                ]
            }
            mock_session.return_value.client.return_value = mock_client
            mock_session.return_value.resource.return_value = MagicMock()

            connector = S3Connector(bucket="test-bucket")
            objects = connector.list_objects()

            assert len(objects) == 2
            assert objects[0]["key"] == "file1.csv"


class TestDataConnectorAbstract:
    """Tests for DataConnector abstract base class."""

    def test_connector_repr(self):
        """Test connector string representation."""
        connector = MockConnector()
        repr_str = repr(connector)

        assert "MockConnector" in repr_str
        assert "connected=True" in repr_str

    def test_connector_ping_success(self):
        """Test connector ping success."""
        connector = MockConnector()

        with patch.object(connector, "execute", return_value=QueryResult(data=1, rows_affected=1)):
            assert connector.ping() is True

    def test_connector_ping_failure(self):
        """Test connector ping failure."""
        connector = MockConnector()

        with patch.object(connector, "execute", side_effect=Exception("Connection failed")):
            assert connector.ping() is False

    def test_connector_fetch_one(self):
        """Test fetch_one method."""
        connector = MockConnector()

        with patch.object(connector, "fetch", return_value=[{"a": 1}, {"a": 2}]):
            result = connector.fetch_one("SELECT * FROM table")

            assert result == {"a": 1}

    def test_connector_fetch_one_empty(self):
        """Test fetch_one with no results."""
        connector = MockConnector()

        with patch.object(connector, "fetch", return_value=[]):
            result = connector.fetch_one("SELECT * FROM table")

            assert result is None
