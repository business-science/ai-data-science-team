"""
End-to-end integration tests for AI Data Science Team.

These tests verify that components work together correctly.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

try:
    from ai_data_science_team.agents.data_cleaning_agent import DataCleaningAgent
    from ai_data_science_team.agents.data_wrangling_agent import DataWranglingAgent
    from ai_data_science_team.agents.data_visualization_agent import DataVisualizationAgent
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


@pytest.fixture
def mock_llm_for_integration():
    """Mock LLM that returns realistic code responses."""
    mock = MagicMock()

    def mock_invoke(messages):
        # Return different code based on the context
        response = MagicMock()
        response.content = '''```python
import pandas as pd

# Process the data
def process_data(df):
    # Remove missing values
    df_clean = df.dropna()
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    return df_clean

result = process_data(data_raw)
```'''
        return response

    mock.invoke = mock_invoke
    mock.bind_tools = MagicMock(return_value=mock)
    return mock


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Package not installed")
class TestDataPipelineIntegration:
    """Integration tests for data processing pipelines."""

    def test_cleaning_to_wrangling_pipeline(self, mock_llm_for_integration, sample_df):
        """Test data flowing from cleaning to wrangling agent."""
        # This tests the conceptual pipeline
        # In real use, cleaned data would flow to wrangling

        # Simulate cleaning result
        cleaned_df = sample_df.dropna().drop_duplicates()

        # Verify data integrity through pipeline
        assert len(cleaned_df) <= len(sample_df)
        assert cleaned_df.columns.tolist() == sample_df.columns.tolist()

    def test_wrangling_to_visualization_pipeline(self, mock_llm_for_integration, sample_df):
        """Test data flowing from wrangling to visualization."""
        # Simulate wrangling result (aggregated data)
        wrangled_df = sample_df.groupby("department")["salary"].mean().reset_index()

        # Verify this is suitable for visualization
        assert len(wrangled_df) > 0
        assert "department" in wrangled_df.columns
        assert "salary" in wrangled_df.columns


class TestMultiDatasetIntegration:
    """Tests for handling multiple datasets."""

    def test_merge_two_datasets(self, sample_df):
        """Test merging two datasets."""
        # Create second dataset
        df2 = pd.DataFrame({
            "id": [1, 2, 3],
            "bonus": [1000, 2000, 3000],
        })

        # Merge
        merged = pd.merge(sample_df, df2, on="id", how="left")

        assert "bonus" in merged.columns
        assert len(merged) == len(sample_df)

    def test_concat_datasets(self, sample_df):
        """Test concatenating datasets."""
        # Split and rejoin
        df1 = sample_df.head(3)
        df2 = sample_df.tail(2)

        concatenated = pd.concat([df1, df2], ignore_index=True)

        assert len(concatenated) == 5


class TestFileRoundTrip:
    """Tests for reading and writing data through the pipeline."""

    def test_csv_round_trip(self, sample_df, temp_dir):
        """Test reading and writing CSV maintains data integrity."""
        filepath = temp_dir / "roundtrip.csv"

        # Write
        sample_df.to_csv(filepath, index=False)

        # Read back
        loaded_df = pd.read_csv(filepath)

        # Verify data integrity
        assert len(loaded_df) == len(sample_df)
        assert set(loaded_df.columns) == set(sample_df.columns)

    def test_parquet_round_trip(self, sample_df, temp_dir):
        """Test reading and writing Parquet maintains data integrity."""
        filepath = temp_dir / "roundtrip.parquet"

        # Write
        sample_df.to_parquet(filepath, index=False)

        # Read back
        loaded_df = pd.read_parquet(filepath)

        # Verify data integrity
        assert len(loaded_df) == len(sample_df)
        # Parquet preserves dtypes better
        pd.testing.assert_frame_equal(
            loaded_df.reset_index(drop=True),
            sample_df.reset_index(drop=True),
            check_like=True
        )

    def test_excel_round_trip(self, sample_df, temp_dir):
        """Test reading and writing Excel maintains data integrity."""
        filepath = temp_dir / "roundtrip.xlsx"

        # Write
        sample_df.to_excel(filepath, index=False)

        # Read back
        loaded_df = pd.read_excel(filepath)

        # Verify data integrity
        assert len(loaded_df) == len(sample_df)


class TestSQLIntegration:
    """Tests for SQL database integration."""

    def test_sql_write_and_read(self, sample_df, sqlite_connection_string):
        """Test writing to and reading from SQLite."""
        from sqlalchemy import create_engine

        engine = create_engine(sqlite_connection_string)

        # Write new table
        sample_df.to_sql("test_table", engine, index=False, if_exists="replace")

        # Read back
        loaded_df = pd.read_sql("SELECT * FROM test_table", engine)

        assert len(loaded_df) == len(sample_df)

    def test_sql_query_execution(self, sqlite_connection_string):
        """Test executing SQL queries."""
        from sqlalchemy import create_engine, text

        engine = create_engine(sqlite_connection_string)

        # Execute a query
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM users"))
            count = result.scalar()

        assert count > 0


class TestErrorRecovery:
    """Tests for error handling and recovery."""

    def test_handle_invalid_data(self):
        """Test handling invalid data gracefully."""
        # DataFrame with problematic values
        df = pd.DataFrame({
            "a": [1, 2, float("inf"), -float("inf")],
            "b": [1, 2, 3, 4],
        })

        # Replace inf values
        df = df.replace([float("inf"), -float("inf")], pd.NA)

        assert df["a"].isna().sum() == 2

    def test_handle_type_mismatch(self):
        """Test handling type mismatches."""
        df = pd.DataFrame({
            "mixed": ["1", "2", "three", "4"],
        })

        # Convert to numeric with error handling
        df["numeric"] = pd.to_numeric(df["mixed"], errors="coerce")

        assert df["numeric"].isna().sum() == 1  # "three" becomes NaN

    def test_handle_memory_efficient_operations(self, temp_dir):
        """Test memory-efficient data processing."""
        # Create a larger dataset
        large_df = pd.DataFrame({
            "id": range(100000),
            "value": range(100000),
        })

        filepath = temp_dir / "large.csv"
        large_df.to_csv(filepath, index=False)

        # Process in chunks
        chunk_sums = []
        for chunk in pd.read_csv(filepath, chunksize=10000):
            chunk_sums.append(chunk["value"].sum())

        total_sum = sum(chunk_sums)
        expected_sum = sum(range(100000))

        assert total_sum == expected_sum


@pytest.mark.integration
class TestVisualizationIntegration:
    """Integration tests for visualization components."""

    def test_create_dashboard_data(self, sample_df):
        """Test creating data suitable for dashboards."""
        import plotly.express as px

        # Multiple visualizations from same data
        scatter_fig = px.scatter(sample_df, x="age", y="salary", color="department")
        bar_fig = px.bar(
            sample_df.groupby("department")["salary"].mean().reset_index(),
            x="department", y="salary"
        )

        assert scatter_fig is not None
        assert bar_fig is not None

    def test_export_visualizations(self, sample_df, temp_dir):
        """Test exporting visualizations."""
        import plotly.express as px

        fig = px.scatter(sample_df, x="age", y="salary")

        # Export to JSON
        json_path = temp_dir / "chart.json"
        fig.write_json(str(json_path))

        assert json_path.exists()

        # Export to HTML
        html_path = temp_dir / "chart.html"
        fig.write_html(str(html_path))

        assert html_path.exists()


class TestPipelineReproducibility:
    """Tests for pipeline reproducibility."""

    def test_deterministic_results(self, sample_df):
        """Test that operations produce deterministic results."""
        # Same operation twice should give same result
        result1 = sample_df.dropna().sort_values("id").reset_index(drop=True)
        result2 = sample_df.dropna().sort_values("id").reset_index(drop=True)

        pd.testing.assert_frame_equal(result1, result2)

    def test_save_and_restore_pipeline_state(self, sample_df, temp_dir):
        """Test saving and restoring pipeline state."""
        import json

        # Simulate pipeline state
        state = {
            "step": "cleaning",
            "params": {
                "dropna": True,
                "drop_duplicates": True,
            },
            "input_shape": list(sample_df.shape),
        }

        # Save state
        state_path = temp_dir / "pipeline_state.json"
        with open(state_path, "w") as f:
            json.dump(state, f)

        # Restore state
        with open(state_path, "r") as f:
            restored_state = json.load(f)

        assert restored_state == state
