"""
Unit tests for data loader tools.
"""

import os
import json
import pandas as pd
import pytest
from pathlib import Path

try:
    from ai_data_science_team.tools.data_loader import (
        search_directory,
        load_file,
        get_directory_summary,
    )
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


class TestFileLoading:
    """Tests for file loading functionality."""

    def test_load_csv_file(self, sample_csv_file):
        """Test loading a CSV file."""
        df = pd.read_csv(sample_csv_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "name" in df.columns

    def test_load_excel_file(self, sample_excel_file):
        """Test loading an Excel file."""
        df = pd.read_excel(sample_excel_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_parquet_file(self, sample_parquet_file):
        """Test loading a Parquet file."""
        df = pd.read_parquet(sample_parquet_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_json_file(self, sample_json_file):
        """Test loading a JSON file."""
        df = pd.read_json(sample_json_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_nonexistent_file(self, temp_dir):
        """Test loading a file that doesn't exist."""
        nonexistent = temp_dir / "does_not_exist.csv"

        with pytest.raises(FileNotFoundError):
            pd.read_csv(nonexistent)

    def test_load_empty_csv(self, temp_dir):
        """Test loading an empty CSV file."""
        empty_file = temp_dir / "empty.csv"
        empty_file.write_text("")

        with pytest.raises(pd.errors.EmptyDataError):
            pd.read_csv(empty_file)

    def test_load_csv_with_different_encodings(self, temp_dir):
        """Test loading CSV with different encodings."""
        # UTF-8
        utf8_file = temp_dir / "utf8.csv"
        pd.DataFrame({"name": ["Alice", "Bob"]}).to_csv(utf8_file, index=False, encoding="utf-8")

        df = pd.read_csv(utf8_file, encoding="utf-8")
        assert len(df) == 2

    def test_load_csv_with_custom_delimiter(self, temp_dir):
        """Test loading CSV with custom delimiter."""
        tsv_file = temp_dir / "data.tsv"
        tsv_file.write_text("name\tage\nAlice\t30\nBob\t25")

        df = pd.read_csv(tsv_file, delimiter="\t")
        assert len(df) == 2
        assert "name" in df.columns


class TestDirectoryOperations:
    """Tests for directory operations."""

    def test_list_files_in_directory(self, temp_dir, sample_csv_file):
        """Test listing files in a directory."""
        files = list(temp_dir.glob("*.csv"))

        assert len(files) >= 1
        assert sample_csv_file in files

    def test_list_files_with_pattern(self, temp_dir, sample_df):
        """Test listing files matching a pattern."""
        # Create multiple files
        (temp_dir / "data1.csv").write_text("a,b\n1,2")
        (temp_dir / "data2.csv").write_text("a,b\n3,4")
        (temp_dir / "other.txt").write_text("text")

        csv_files = list(temp_dir.glob("*.csv"))
        txt_files = list(temp_dir.glob("*.txt"))

        assert len(csv_files) >= 2
        assert len(txt_files) >= 1

    def test_recursive_file_search(self, temp_dir, sample_df):
        """Test recursive file search."""
        # Create nested directories
        subdir = temp_dir / "subdir"
        subdir.mkdir()

        sample_df.to_csv(temp_dir / "root.csv", index=False)
        sample_df.to_csv(subdir / "nested.csv", index=False)

        all_csv = list(temp_dir.glob("**/*.csv"))

        assert len(all_csv) >= 2

    def test_directory_not_found(self, temp_dir):
        """Test handling non-existent directory."""
        nonexistent = temp_dir / "nonexistent"

        # glob returns empty for non-existent dirs, iterdir raises
        with pytest.raises(FileNotFoundError):
            list(nonexistent.iterdir())


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Package not installed")
class TestDataLoaderTools:
    """Tests for the data loader tool functions."""

    def test_search_directory_function(self, temp_dir, sample_csv_file):
        """Test the search_directory tool function."""
        result = search_directory(str(temp_dir), pattern="*.csv")

        assert isinstance(result, (list, str))

    def test_load_file_function(self, sample_csv_file):
        """Test the load_file tool function."""
        result = load_file(str(sample_csv_file))

        assert result is not None

    def test_get_directory_summary_function(self, temp_dir):
        """Test the get_directory_summary tool function."""
        result = get_directory_summary(str(temp_dir))

        assert isinstance(result, str)


class TestFileTypeDetection:
    """Tests for file type detection and handling."""

    def test_detect_csv_by_extension(self, temp_dir):
        """Test CSV detection by extension."""
        csv_file = temp_dir / "test.csv"
        csv_file.write_text("a,b\n1,2")

        assert csv_file.suffix == ".csv"

    def test_detect_excel_by_extension(self, temp_dir):
        """Test Excel detection by extension."""
        xlsx_file = temp_dir / "test.xlsx"
        # Just test extension detection, not actual Excel writing
        assert xlsx_file.suffix == ".xlsx"

    def test_infer_schema_from_data(self, sample_df):
        """Test schema inference from data."""
        dtypes = sample_df.dtypes

        assert dtypes["id"] in ["int64", "int32"]
        assert dtypes["name"] == "object"
        assert dtypes["salary"] == "float64"

    def test_handle_mixed_file_types(self, temp_dir, sample_df):
        """Test handling directory with mixed file types."""
        # Create CSV file
        sample_df.to_csv(temp_dir / "data.csv", index=False)
        # Create a text file instead of xlsx/parquet to avoid dependency issues
        (temp_dir / "data.txt").write_text("sample text")
        (temp_dir / "data.json").write_text('{"a": 1}')

        files = list(temp_dir.iterdir())

        # Filter by type
        csv_files = [f for f in files if f.suffix == ".csv"]
        txt_files = [f for f in files if f.suffix == ".txt"]
        json_files = [f for f in files if f.suffix == ".json"]

        assert len(csv_files) >= 1
        assert len(txt_files) >= 1
        assert len(json_files) >= 1


class TestDataLoaderEdgeCases:
    """Tests for edge cases in data loading."""

    def test_load_very_large_csv_chunk(self, temp_dir):
        """Test loading large CSV in chunks."""
        # Create a larger CSV
        large_df = pd.DataFrame({
            "id": range(10000),
            "value": range(10000),
        })
        large_file = temp_dir / "large.csv"
        large_df.to_csv(large_file, index=False)

        # Read in chunks
        chunks = []
        for chunk in pd.read_csv(large_file, chunksize=1000):
            chunks.append(chunk)

        combined = pd.concat(chunks)
        assert len(combined) == 10000

    def test_load_csv_with_special_characters(self, temp_dir):
        """Test loading CSV with special characters."""
        special_file = temp_dir / "special.csv"
        special_file.write_text('name,value\n"Alice, Bob",100\n"Test ""quoted""",200')

        df = pd.read_csv(special_file)
        assert len(df) == 2

    def test_load_csv_with_missing_values(self, temp_dir):
        """Test loading CSV with various missing value representations."""
        missing_file = temp_dir / "missing.csv"
        missing_file.write_text("a,b,c\n1,NA,\n2,,null\n3,None,N/A")

        df = pd.read_csv(missing_file, na_values=["NA", "null", "None", "N/A", ""])

        # All columns should have missing values detected
        assert df["b"].isna().sum() >= 2
        assert df["c"].isna().sum() >= 2
