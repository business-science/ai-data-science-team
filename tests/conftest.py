"""
Shared pytest fixtures for AI Data Science Team tests.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_df():
    """Basic sample DataFrame for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "age": [25, 30, 35, None, 45],
        "salary": [50000, 60000, 70000, 80000, None],
        "department": ["Engineering", "Sales", "Engineering", "HR", "Sales"],
        "hire_date": pd.to_datetime([
            "2020-01-15", "2019-06-20", "2021-03-10", "2018-11-05", "2022-07-01"
        ]),
    })


@pytest.fixture
def sample_df_with_missing():
    """DataFrame with various missing value patterns."""
    return pd.DataFrame({
        "complete": [1, 2, 3, 4, 5],
        "some_missing": [1.0, None, 3.0, None, 5.0],
        "mostly_missing": [None, None, None, 4.0, None],
        "all_missing": [None, None, None, None, None],
        "category": ["A", "B", None, "A", "B"],
    })


@pytest.fixture
def sample_df_with_duplicates():
    """DataFrame with duplicate rows."""
    return pd.DataFrame({
        "id": [1, 2, 2, 3, 3, 3],
        "value": [10, 20, 20, 30, 30, 30],
        "category": ["A", "B", "B", "C", "C", "C"],
    })


@pytest.fixture
def sample_df_with_outliers():
    """DataFrame with outliers for testing outlier detection."""
    return pd.DataFrame({
        "normal": [10, 12, 11, 13, 10, 11, 12, 100, -50],  # 100 and -50 are outliers
        "category": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
    })


@pytest.fixture
def sample_df_timeseries():
    """Time series DataFrame for testing temporal operations."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    return pd.DataFrame({
        "date": dates,
        "value": range(100),
        "category": ["A", "B", "C", "D"] * 25,
    })


@pytest.fixture
def sample_df_for_ml():
    """DataFrame suitable for ML testing (classification)."""
    import numpy as np
    np.random.seed(42)
    n_samples = 100

    return pd.DataFrame({
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
        "feature3": np.random.choice(["A", "B", "C"], n_samples),
        "target": np.random.choice([0, 1], n_samples),
    })


# ============================================================================
# File System Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_csv_file(temp_dir, sample_df):
    """Create a temporary CSV file with sample data."""
    filepath = temp_dir / "sample_data.csv"
    sample_df.to_csv(filepath, index=False)
    return filepath


@pytest.fixture
def sample_excel_file(temp_dir, sample_df):
    """Create a temporary Excel file with sample data."""
    filepath = temp_dir / "sample_data.xlsx"
    sample_df.to_excel(filepath, index=False)
    return filepath


@pytest.fixture
def sample_parquet_file(temp_dir, sample_df):
    """Create a temporary Parquet file with sample data."""
    filepath = temp_dir / "sample_data.parquet"
    sample_df.to_parquet(filepath, index=False)
    return filepath


@pytest.fixture
def sample_json_file(temp_dir, sample_df):
    """Create a temporary JSON file with sample data."""
    filepath = temp_dir / "sample_data.json"
    sample_df.to_json(filepath, orient="records")
    return filepath


# ============================================================================
# Mock LLM Fixtures
# ============================================================================

@pytest.fixture
def mock_llm():
    """Create a mock LLM that returns predictable responses."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="Mock LLM response")
    mock.bind_tools.return_value = mock
    return mock


@pytest.fixture
def mock_llm_with_code():
    """Create a mock LLM that returns code responses."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(
        content='''```python
# Generated code
import pandas as pd

def process_data(df):
    return df.dropna()

result = process_data(data_raw)
```'''
    )
    mock.bind_tools.return_value = mock
    return mock


@pytest.fixture
def mock_chat_openai():
    """Mock ChatOpenAI for testing without API calls."""
    with patch("langchain_openai.ChatOpenAI") as mock:
        instance = MagicMock()
        instance.invoke.return_value = MagicMock(content="Mock response")
        instance.bind_tools.return_value = instance
        mock.return_value = instance
        yield mock


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture
def sample_sqlite_db(temp_dir, sample_df):
    """Create a temporary SQLite database with sample data."""
    import sqlite3

    db_path = temp_dir / "test.db"
    conn = sqlite3.connect(str(db_path))
    sample_df.to_sql("users", conn, index=False, if_exists="replace")

    # Add another table
    pd.DataFrame({
        "order_id": [1, 2, 3],
        "user_id": [1, 2, 1],
        "amount": [100.0, 200.0, 150.0],
    }).to_sql("orders", conn, index=False, if_exists="replace")

    conn.close()
    return db_path


@pytest.fixture
def sqlite_connection_string(sample_sqlite_db):
    """SQLAlchemy connection string for the test database."""
    return f"sqlite:///{sample_sqlite_db}"


# ============================================================================
# Agent Fixtures
# ============================================================================

@pytest.fixture
def agent_config():
    """Base configuration for agent testing."""
    return {
        "model": MagicMock(),
        "human_in_the_loop": False,
        "log": False,
        "log_path": None,
    }


# ============================================================================
# Environment Fixtures
# ============================================================================

@pytest.fixture
def env_with_openai_key():
    """Set up environment with OpenAI API key for testing."""
    original = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "test-api-key"
    yield
    if original is not None:
        os.environ["OPENAI_API_KEY"] = original
    else:
        os.environ.pop("OPENAI_API_KEY", None)


# ============================================================================
# Utility Functions
# ============================================================================

def assert_dataframe_equal(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    """Assert two DataFrames are equal with helpful error messages."""
    pd.testing.assert_frame_equal(df1, df2)


def assert_columns_exist(df: pd.DataFrame, columns: list) -> None:
    """Assert that specified columns exist in the DataFrame."""
    missing = set(columns) - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
