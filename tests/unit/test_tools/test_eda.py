"""
Unit tests for EDA (Exploratory Data Analysis) tools.
"""

import pandas as pd
import numpy as np
import pytest

try:
    from ai_data_science_team.tools.eda import (
        explain_data,
        describe_dataset,
        visualize_missing,
        generate_correlation_funnel,
    )
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


class TestBasicEDA:
    """Tests for basic EDA operations."""

    def test_dataframe_info(self, sample_df):
        """Test getting DataFrame info."""
        info = sample_df.dtypes.to_dict()

        assert "id" in info
        assert "name" in info
        assert "salary" in info

    def test_describe_numeric(self, sample_df):
        """Test describing numeric columns."""
        description = sample_df.describe()

        assert "count" in description.index
        assert "mean" in description.index
        assert "std" in description.index
        assert "min" in description.index
        assert "max" in description.index

    def test_describe_all_columns(self, sample_df):
        """Test describing all columns including non-numeric."""
        description = sample_df.describe(include="all")

        assert "name" in description.columns
        assert "department" in description.columns

    def test_value_counts(self, sample_df):
        """Test value counts for categorical columns."""
        counts = sample_df["department"].value_counts()

        assert isinstance(counts, pd.Series)
        assert counts.sum() == len(sample_df)

    def test_unique_values(self, sample_df):
        """Test getting unique values."""
        unique = sample_df["department"].unique()

        assert len(unique) == sample_df["department"].nunique()


class TestMissingValueAnalysis:
    """Tests for missing value analysis."""

    def test_missing_count(self, sample_df_with_missing):
        """Test counting missing values."""
        missing = sample_df_with_missing.isna().sum()

        assert missing["complete"] == 0
        assert missing["some_missing"] > 0
        assert missing["all_missing"] == len(sample_df_with_missing)

    def test_missing_percentage(self, sample_df_with_missing):
        """Test calculating missing percentage."""
        df = sample_df_with_missing
        missing_pct = (df.isna().sum() / len(df)) * 100

        assert missing_pct["complete"] == 0
        assert missing_pct["all_missing"] == 100

    def test_missing_pattern_matrix(self, sample_df_with_missing):
        """Test creating missing pattern matrix."""
        df = sample_df_with_missing
        missing_matrix = df.isna().astype(int)

        assert missing_matrix.shape == df.shape
        assert missing_matrix.values.max() == 1
        assert missing_matrix.values.min() == 0


class TestCorrelationAnalysis:
    """Tests for correlation analysis."""

    def test_pearson_correlation(self, sample_df):
        """Test Pearson correlation calculation."""
        numeric_df = sample_df.select_dtypes(include=[np.number])
        corr = numeric_df.corr(method="pearson")

        # Diagonal should be 1
        assert all(np.isclose(np.diag(corr), 1.0))

        # Should be symmetric
        assert np.allclose(corr.values, corr.values.T, equal_nan=True)

    def test_spearman_correlation(self, sample_df):
        """Test Spearman correlation calculation."""
        numeric_df = sample_df.select_dtypes(include=[np.number])
        corr = numeric_df.corr(method="spearman")

        assert corr.shape[0] == corr.shape[1]

    def test_correlation_with_target(self, sample_df_for_ml):
        """Test correlation with a target variable."""
        df = sample_df_for_ml
        numeric_df = df.select_dtypes(include=[np.number])

        # Correlation with target
        target_corr = numeric_df.corr()["target"].drop("target")

        assert len(target_corr) == len(numeric_df.columns) - 1


class TestDataDistribution:
    """Tests for data distribution analysis."""

    def test_histogram_bins(self, sample_df):
        """Test histogram binning."""
        values = sample_df["salary"].dropna()
        hist, bin_edges = np.histogram(values, bins=10)

        assert len(hist) == 10
        assert len(bin_edges) == 11

    def test_quantiles(self, sample_df):
        """Test quantile calculation."""
        quantiles = sample_df["salary"].quantile([0.25, 0.5, 0.75])

        assert len(quantiles) == 3
        assert quantiles[0.25] <= quantiles[0.5] <= quantiles[0.75]

    def test_skewness(self, sample_df):
        """Test skewness calculation."""
        skew = sample_df["salary"].skew()

        assert isinstance(skew, float)

    def test_kurtosis(self, sample_df):
        """Test kurtosis calculation."""
        kurt = sample_df["salary"].kurtosis()

        assert isinstance(kurt, float)


class TestCategoricalAnalysis:
    """Tests for categorical data analysis."""

    def test_category_frequency(self, sample_df):
        """Test category frequency analysis."""
        freq = sample_df["department"].value_counts(normalize=True)

        assert freq.sum() == pytest.approx(1.0)
        assert all(freq >= 0) and all(freq <= 1)

    def test_crosstab(self, sample_df):
        """Test cross-tabulation."""
        # Need to bin age first for crosstab
        df = sample_df.copy()
        df["age_group"] = pd.cut(df["age"], bins=[0, 30, 40, 100], labels=["<30", "30-40", ">40"])

        crosstab = pd.crosstab(df["department"], df["age_group"])

        assert isinstance(crosstab, pd.DataFrame)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Package not installed")
class TestEDATools:
    """Tests for the EDA tool functions."""

    def test_explain_data_function(self, sample_df):
        """Test the explain_data tool function."""
        result = explain_data(sample_df)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_describe_dataset_function(self, sample_df):
        """Test the describe_dataset tool function."""
        result = describe_dataset(sample_df)

        assert isinstance(result, str)


class TestEDAEdgeCases:
    """Tests for edge cases in EDA."""

    def test_eda_single_column(self):
        """Test EDA with single column DataFrame."""
        df = pd.DataFrame({"single": [1, 2, 3, 4, 5]})

        description = df.describe()
        assert len(description.columns) == 1

    def test_eda_single_row(self):
        """Test EDA with single row DataFrame."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})

        description = df.describe()
        # count should be 1
        assert description.loc["count"].values[0] == 1

    def test_eda_constant_column(self):
        """Test EDA with constant column (zero variance)."""
        df = pd.DataFrame({"constant": [5, 5, 5, 5, 5]})

        description = df.describe()
        assert description.loc["std", "constant"] == 0

    def test_eda_all_null_column(self):
        """Test EDA with all-null column."""
        df = pd.DataFrame({
            "valid": [1, 2, 3],
            "all_null": [None, None, None],
        })

        # describe() may exclude all-null columns, so check missing count instead
        missing_count = df["all_null"].isna().sum()
        assert missing_count == 3

    def test_correlation_with_constant_column(self):
        """Test correlation when one column is constant."""
        df = pd.DataFrame({
            "varying": [1, 2, 3, 4, 5],
            "constant": [1, 1, 1, 1, 1],
        })

        corr = df.corr()

        # Correlation with constant column should be NaN
        assert pd.isna(corr.loc["varying", "constant"])
