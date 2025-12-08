import io
import pandas as pd
from typing_extensions import Union, List, Dict


def get_dataframe_summary(
    dataframes: Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]],
    n_sample: int = 30,
    skip_stats: bool = False,
) -> List[str]:
    """
    Generate a summary for one or more DataFrames. Accepts a single DataFrame, a list of DataFrames,
    or a dictionary mapping names to DataFrames.

    Parameters
    ----------
    dataframes : pandas.DataFrame or list of pandas.DataFrame or dict of (str -> pandas.DataFrame)
        - Single DataFrame: produce a single summary (returned within a one-element list).
        - List of DataFrames: produce a summary for each DataFrame, using index-based names.
        - Dictionary of DataFrames: produce a summary for each DataFrame, using dictionary keys as names.
    n_sample : int, default 30
        Number of rows to display in the "Data (first 30 rows)" section.
    skip_stats : bool, default False
        If True, skip the descriptive statistics and DataFrame info sections.

    Example:
    --------
    ``` python
    import pandas as pd
    from sklearn.datasets import load_iris
    data = load_iris(as_frame=True)
    dataframes = {
        "iris": data.frame,
        "iris_target": data.target,
    }
    summaries = get_dataframe_summary(dataframes)
    print(summaries[0])
    ```

    Returns
    -------
    list of str
        A list of summaries, one for each provided DataFrame. Each summary includes:
        - Shape of the DataFrame (rows, columns)
        - Column data types
        - Missing value percentage
        - Unique value counts
        - First 30 rows
        - Descriptive statistics
        - DataFrame info output
    """

    summaries = []

    # --- Dictionary Case ---
    if isinstance(dataframes, dict):
        for dataset_name, df in dataframes.items():
            summaries.append(
                _summarize_dataframe(df, dataset_name, n_sample, skip_stats)
            )

    # --- Single DataFrame Case ---
    elif isinstance(dataframes, pd.DataFrame):
        summaries.append(
            _summarize_dataframe(dataframes, "Single_Dataset", n_sample, skip_stats)
        )

    # --- List of DataFrames Case ---
    elif isinstance(dataframes, list):
        for idx, df in enumerate(dataframes):
            dataset_name = f"Dataset_{idx}"
            summaries.append(
                _summarize_dataframe(df, dataset_name, n_sample, skip_stats)
            )

    else:
        raise TypeError(
            "Input must be a single DataFrame, a list of DataFrames, or a dictionary of DataFrames."
        )

    return summaries


def _summarize_dataframe(
    df: pd.DataFrame, dataset_name: str, n_sample=30, skip_stats=False
) -> str:
    """Generate a summary string for a single DataFrame."""
    # 1. Convert dictionary-type cells to strings
    #    This prevents unhashable dict errors during df.nunique().
    df = df.apply(lambda col: col.map(lambda x: str(x) if isinstance(x, dict) else x))

    # 2. Capture df.info() output
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_text = buffer.getvalue()

    # 3. Calculate missing value stats
    missing_stats = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    missing_summary = "\n".join(
        [f"{col}: {val:.2f}%" for col, val in missing_stats.items()]
    )

    # 4. Get column data types
    column_types = "\n".join([f"{col}: {dtype}" for col, dtype in df.dtypes.items()])

    # 5. Get unique value counts
    unique_counts = df.nunique()  # Will no longer fail on unhashable dict
    unique_counts_summary = "\n".join(
        [f"{col}: {count}" for col, count in unique_counts.items()]
    )

    # 6. Generate the summary text
    if not skip_stats:
        summary_text = f"""
        Dataset Name: {dataset_name}
        ----------------------------
        Shape: {df.shape[0]} rows x {df.shape[1]} columns

        Column Data Types:
        {column_types}

        Missing Value Percentage:
        {missing_summary}

        Unique Value Counts:
        {unique_counts_summary}

        Data (first {n_sample} rows):
        {df.head(n_sample).to_string()}

        Data Description:
        {df.describe().to_string()}

        Data Info:
        {info_text}
        """
    else:
        summary_text = f"""
        Dataset Name: {dataset_name}
        ----------------------------
        Shape: {df.shape[0]} rows x {df.shape[1]} columns

        Column Data Types:
        {column_types}

        Data (first {n_sample} rows):
        {df.head(n_sample).to_string()}
        """

    return summary_text.strip()
