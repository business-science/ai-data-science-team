"""
Unit tests for DataWranglingAgent.
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

try:
    from ai_data_science_team.agents.data_wrangling_agent import (
        DataWranglingAgent,
        make_data_wrangling_agent,
    )
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Package not installed")
class TestDataWranglingAgent:
    """Tests for DataWranglingAgent class."""

    def test_agent_initialization(self, mock_llm):
        """Test that agent initializes correctly."""
        agent = DataWranglingAgent(model=mock_llm)
        assert agent is not None

    def test_agent_has_required_methods(self, mock_llm):
        """Test that agent has required interface methods."""
        agent = DataWranglingAgent(model=mock_llm)

        assert hasattr(agent, "invoke_agent")
        assert hasattr(agent, "get_data_wrangled")
        assert hasattr(agent, "get_data_wrangler_function")
        assert hasattr(agent, "get_response")

    def test_agent_with_bypass_recommended_steps(self, mock_llm):
        """Test agent with bypass_recommended_steps option."""
        agent = DataWranglingAgent(
            model=mock_llm,
            bypass_recommended_steps=True,
        )
        assert agent is not None


class TestDataWranglingOperations:
    """Tests for data wrangling operations without LLM."""

    def test_select_columns(self, sample_df):
        """Test column selection."""
        df = sample_df
        selected = df[["name", "age", "department"]]

        assert list(selected.columns) == ["name", "age", "department"]
        assert len(selected) == len(df)

    def test_filter_rows(self, sample_df):
        """Test row filtering."""
        df = sample_df

        # Filter by department
        filtered = df[df["department"] == "Engineering"]
        assert len(filtered) == 2

        # Filter by age
        filtered = df[df["age"] > 30]
        assert len(filtered) >= 2

    def test_sort_values(self, sample_df):
        """Test sorting."""
        df = sample_df

        # Sort by age descending
        sorted_df = df.sort_values("age", ascending=False)

        # First non-null value should be highest
        non_null_ages = sorted_df["age"].dropna()
        assert non_null_ages.iloc[0] == non_null_ages.max()

    def test_groupby_aggregation(self, sample_df):
        """Test groupby operations."""
        df = sample_df

        # Group by department and calculate mean salary
        grouped = df.groupby("department")["salary"].mean().reset_index()

        assert "department" in grouped.columns
        assert "salary" in grouped.columns
        assert len(grouped) == df["department"].nunique()

    def test_pivot_table(self, sample_df):
        """Test pivot table creation."""
        df = sample_df

        pivot = pd.pivot_table(
            df,
            values="salary",
            index="department",
            aggfunc="mean"
        )

        assert len(pivot) == df["department"].nunique()

    def test_melt_unpivot(self):
        """Test melting/unpivoting data."""
        df = pd.DataFrame({
            "id": [1, 2],
            "jan": [100, 200],
            "feb": [110, 210],
            "mar": [120, 220],
        })

        melted = pd.melt(
            df,
            id_vars=["id"],
            value_vars=["jan", "feb", "mar"],
            var_name="month",
            value_name="value"
        )

        assert len(melted) == 6  # 2 ids * 3 months
        assert "month" in melted.columns
        assert "value" in melted.columns

    def test_merge_dataframes(self, sample_df):
        """Test merging DataFrames."""
        # Create a second DataFrame to merge
        df2 = pd.DataFrame({
            "id": [1, 2, 3],
            "bonus": [5000, 6000, 7000],
        })

        merged = pd.merge(sample_df, df2, on="id", how="left")

        assert "bonus" in merged.columns
        assert len(merged) == len(sample_df)

    def test_concat_dataframes(self, sample_df):
        """Test concatenating DataFrames."""
        df1 = sample_df.head(2)
        df2 = sample_df.tail(2)

        concatenated = pd.concat([df1, df2], ignore_index=True)

        assert len(concatenated) == 4

    def test_rename_columns(self, sample_df):
        """Test column renaming."""
        df = sample_df.copy()

        df = df.rename(columns={
            "name": "full_name",
            "department": "dept",
        })

        assert "full_name" in df.columns
        assert "dept" in df.columns
        assert "name" not in df.columns

    def test_add_computed_column(self, sample_df):
        """Test adding computed columns."""
        df = sample_df.copy()

        # Add bonus column (10% of salary)
        df["bonus"] = df["salary"] * 0.1

        assert "bonus" in df.columns
        # Check calculation for non-null values
        non_null_mask = df["salary"].notna()
        assert (df.loc[non_null_mask, "bonus"] == df.loc[non_null_mask, "salary"] * 0.1).all()

    def test_string_operations(self, sample_df):
        """Test string column operations."""
        df = sample_df.copy()

        # Uppercase names
        df["name_upper"] = df["name"].str.upper()

        assert df["name_upper"].iloc[0] == "ALICE"

    def test_datetime_operations(self, sample_df):
        """Test datetime column operations."""
        df = sample_df.copy()

        # Extract year from hire_date
        df["hire_year"] = df["hire_date"].dt.year
        df["hire_month"] = df["hire_date"].dt.month

        assert "hire_year" in df.columns
        assert "hire_month" in df.columns
        assert df["hire_year"].min() >= 2018

    def test_apply_function(self, sample_df):
        """Test applying custom functions."""
        df = sample_df.copy()

        # Apply function to categorize ages
        def age_category(age):
            if pd.isna(age):
                return "Unknown"
            elif age < 30:
                return "Young"
            elif age < 40:
                return "Middle"
            else:
                return "Senior"

        df["age_category"] = df["age"].apply(age_category)

        assert "age_category" in df.columns
        assert set(df["age_category"].unique()).issubset({"Young", "Middle", "Senior", "Unknown"})


class TestDataWranglingEdgeCases:
    """Tests for edge cases in data wrangling."""

    def test_empty_merge_result(self):
        """Test merge that produces empty result."""
        df1 = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
        df2 = pd.DataFrame({"id": [3, 4], "other": [30, 40]})

        merged = pd.merge(df1, df2, on="id", how="inner")

        assert merged.empty

    def test_groupby_empty_groups(self):
        """Test groupby with resulting empty groups."""
        df = pd.DataFrame({
            "category": ["A", "A", "B"],
            "value": [1, 2, 3],
        })

        grouped = df.groupby("category")["value"].sum()

        assert len(grouped) == 2

    def test_sort_with_nulls(self, sample_df):
        """Test sorting with null values."""
        df = sample_df.copy()

        # Sort with nulls first
        sorted_df = df.sort_values("age", na_position="first")
        assert pd.isna(sorted_df["age"].iloc[0])

        # Sort with nulls last
        sorted_df = df.sort_values("age", na_position="last")
        assert pd.isna(sorted_df["age"].iloc[-1])

    def test_duplicate_column_names_after_merge(self):
        """Test handling duplicate column names in merge."""
        df1 = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
        df2 = pd.DataFrame({"id": [1, 2], "value": [100, 200]})

        merged = pd.merge(df1, df2, on="id", suffixes=("_left", "_right"))

        assert "value_left" in merged.columns
        assert "value_right" in merged.columns
