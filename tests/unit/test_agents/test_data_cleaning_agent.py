"""
Unit tests for DataCleaningAgent.
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

# Test imports - these should work after proper installation
try:
    from ai_data_science_team.agents.data_cleaning_agent import (
        DataCleaningAgent,
        make_data_cleaning_agent,
    )
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Package not installed")
class TestDataCleaningAgent:
    """Tests for DataCleaningAgent class."""

    def test_agent_initialization(self, mock_llm):
        """Test that agent initializes with correct parameters."""
        agent = DataCleaningAgent(
            model=mock_llm,
            human_in_the_loop=False,
            log=False,
        )
        assert agent is not None
        assert agent._params["human_in_the_loop"] is False

    def test_agent_initialization_with_defaults(self, mock_llm):
        """Test agent initializes with default parameters."""
        agent = DataCleaningAgent(model=mock_llm)
        assert agent is not None

    def test_agent_has_required_methods(self, mock_llm):
        """Test that agent has all required interface methods."""
        agent = DataCleaningAgent(model=mock_llm)

        assert hasattr(agent, "invoke_agent")
        assert hasattr(agent, "get_data_cleaned")
        assert hasattr(agent, "get_data_cleaner_function")
        assert hasattr(agent, "get_recommended_cleaning_steps")
        assert hasattr(agent, "get_response")
        assert callable(agent.invoke_agent)

    def test_agent_params_structure(self, mock_llm):
        """Test that agent params have expected structure."""
        agent = DataCleaningAgent(
            model=mock_llm,
            human_in_the_loop=True,
            log=True,
            log_path="./logs",
            n_samples=5,
        )

        assert agent._params["human_in_the_loop"] is True
        assert agent._params["log"] is True
        assert agent._params["n_samples"] == 5

    def test_agent_with_bypass_recommended_steps(self, mock_llm):
        """Test agent with bypass_recommended_steps option."""
        agent = DataCleaningAgent(
            model=mock_llm,
            bypass_recommended_steps=True,
        )

        assert agent._params["bypass_recommended_steps"] is True


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Package not installed")
class TestDataCleaningAgentInvocation:
    """Tests for DataCleaningAgent invocation and data processing."""

    def test_get_data_cleaned_before_invoke(self, mock_llm):
        """Test that get_data_cleaned returns None before invocation."""
        agent = DataCleaningAgent(model=mock_llm)
        result = agent.get_data_cleaned()
        assert result is None

    def test_get_recommended_cleaning_steps_before_invoke(self, mock_llm):
        """Test that get_recommended_cleaning_steps returns None before invocation."""
        agent = DataCleaningAgent(model=mock_llm)
        result = agent.get_recommended_cleaning_steps()
        assert result is None


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Package not installed")
class TestDataCleaningAgentGraph:
    """Tests for the agent graph construction."""

    def test_make_data_cleaning_agent_creates_graph(self, mock_llm):
        """Test that make_data_cleaning_agent creates a valid graph."""
        # This tests the graph factory function
        graph = make_data_cleaning_agent(model=mock_llm)
        assert graph is not None
        assert hasattr(graph, "invoke")


class TestDataCleaningLogic:
    """Tests for data cleaning logic without LLM."""

    def test_remove_duplicates(self, sample_df_with_duplicates):
        """Test duplicate removal logic."""
        df = sample_df_with_duplicates
        cleaned = df.drop_duplicates()

        assert len(cleaned) < len(df)
        assert len(cleaned) == 3  # 3 unique rows

    def test_handle_missing_values_dropna(self, sample_df_with_missing):
        """Test dropping rows with missing values."""
        df = sample_df_with_missing
        # Drop the all_missing column since it's all NaN
        df_subset = df.drop(columns=["all_missing", "mostly_missing"])
        cleaned_subset = df_subset.dropna(how="any")

        # After dropping all_missing and mostly_missing:
        # Remaining: complete (no NaN), some_missing (has NaN), category (has NaN)
        # Rows with all non-null: rows 0 and 4
        assert len(cleaned_subset) == 2

    def test_handle_missing_values_fillna(self, sample_df_with_missing):
        """Test filling missing values."""
        df = sample_df_with_missing.copy()

        # Fill numeric with mean
        df["some_missing"] = df["some_missing"].fillna(df["some_missing"].mean())

        assert df["some_missing"].isna().sum() == 0
        assert df["some_missing"].mean() == 3.0  # (1+3+5)/3 = 3

    def test_detect_outliers_iqr(self, sample_df_with_outliers):
        """Test IQR-based outlier detection."""
        df = sample_df_with_outliers
        col = "normal"

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

        # 100 and -50 should be detected as outliers
        assert len(outliers) >= 2

    def test_type_conversion(self, sample_df):
        """Test data type conversions."""
        df = sample_df.copy()

        # Convert age to int (after handling NaN)
        df["age"] = df["age"].fillna(0).astype(int)

        assert df["age"].dtype == "int64" or df["age"].dtype == "int32"

    def test_remove_high_missing_columns(self, sample_df_with_missing):
        """Test removing columns with high missing rate."""
        df = sample_df_with_missing
        threshold = 0.4  # 40%

        # Calculate missing rate per column
        missing_rate = df.isna().sum() / len(df)

        # Columns to drop (missing rate > threshold)
        cols_to_drop = missing_rate[missing_rate > threshold].index.tolist()

        # Should include 'mostly_missing' and 'all_missing'
        assert "mostly_missing" in cols_to_drop
        assert "all_missing" in cols_to_drop

    def test_standardize_column_names(self):
        """Test column name standardization."""
        df = pd.DataFrame({
            "First Name": [1],
            "LAST_NAME": [2],
            "email Address": [3],
            "Phone-Number": [4],
        })

        # Standardize: lowercase, replace spaces and hyphens with underscores
        df.columns = (
            df.columns
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("-", "_")
        )

        expected_cols = ["first_name", "last_name", "email_address", "phone_number"]
        assert list(df.columns) == expected_cols


class TestDataCleaningEdgeCases:
    """Tests for edge cases in data cleaning."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()

        # Operations should not raise errors
        cleaned = df.dropna()
        assert cleaned.empty
        assert len(cleaned) == 0

    def test_single_row_dataframe(self):
        """Test handling of single-row DataFrame."""
        df = pd.DataFrame({"a": [1], "b": [2]})

        cleaned = df.drop_duplicates()
        assert len(cleaned) == 1

    def test_all_missing_dataframe(self):
        """Test handling of DataFrame with all missing values."""
        df = pd.DataFrame({
            "a": [None, None],
            "b": [None, None],
        })

        cleaned = df.dropna()
        assert cleaned.empty

    def test_mixed_types_column(self):
        """Test handling of columns with mixed types."""
        df = pd.DataFrame({
            "mixed": [1, "two", 3.0, None, "5"],
        })

        # This should not raise
        assert len(df) == 5
        assert df["mixed"].dtype == object

    def test_very_long_strings(self):
        """Test handling of very long string values."""
        long_string = "a" * 10000
        df = pd.DataFrame({
            "text": [long_string, "short", None],
        })

        # Should handle without issues
        cleaned = df.dropna()
        assert len(cleaned) == 2
