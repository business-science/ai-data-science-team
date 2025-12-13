from typing_extensions import Annotated, Dict, Tuple, Union

import os
import tempfile
import warnings

from langchain.tools import tool

from langgraph.prebuilt import InjectedState

from ai_data_science_team.tools.dataframe import get_dataframe_summary


@tool(response_format="content")
def explain_data(
    data_raw: Annotated[dict, InjectedState("data_raw")],
    n_sample: int = 30,
    skip_stats: bool = False,
):
    """
    Tool: explain_data
    Description:
        Provides an extensive, narrative summary of a DataFrame including its shape, column types,
        missing value percentages, unique counts, sample rows, and (if not skipped) descriptive stats/info.

    Parameters:
        data_raw (dict): Raw data.
        n_sample (int, default=30): Number of rows to display.
        skip_stats (bool, default=False): If True, omit descriptive stats/info.

    LLM Guidance:
        Use when a detailed, human-readable explanation is needed—i.e., a full overview is preferred over a concise numerical summary.

    Returns:
        str: Detailed DataFrame summary.
    """
    print("    * Tool: explain_data")
    import pandas as pd

    result = get_dataframe_summary(
        pd.DataFrame(data_raw), n_sample=n_sample, skip_stats=skip_stats
    )

    return result


@tool(response_format="content_and_artifact")
def describe_dataset(
    data_raw: Annotated[dict, InjectedState("data_raw")],
) -> Tuple[str, Dict]:
    """
    Tool: describe_dataset
    Description:
        Compute and return summary statistics for the dataset using pandas' describe() method.
        The tool provides both a textual summary and a structured artifact (a dictionary) for further processing.

    Parameters:
    -----------
    data_raw : dict
        The raw data in dictionary format.

    LLM Selection Guidance:
    ------------------------
    Use this tool when:
      - The request emphasizes numerical descriptive statistics (e.g., count, mean, std, min, quartiles, max).
      - The user needs a concise statistical snapshot rather than a detailed narrative.
      - Both a brief text explanation and a structured data artifact (for downstream tasks) are required.

    Returns:
    -------
    Tuple[str, Dict]:
        - content: A textual summary indicating that summary statistics have been computed.
        - artifact: A dictionary (derived from DataFrame.describe()) containing detailed statistical measures.
    """
    print("    * Tool: describe_dataset")
    import pandas as pd

    df = pd.DataFrame(data_raw)
    description_df = df.describe(include="all")
    content = "Summary statistics computed using pandas describe()."
    # Flatten: orient="index" gives rows=stat, columns=columns
    flattened = description_df.reset_index().rename(columns={"index": "stat"})
    artifact = {"describe_df": flattened.to_dict(orient="list")}
    return content, artifact


@tool(response_format="content_and_artifact")
def visualize_missing(
    data_raw: Annotated[dict, InjectedState("data_raw")], n_sample: int = None
) -> Tuple[str, Dict]:
    """
    Tool: visualize_missing
    Description:
        Missing value analysis using the missingno library. Generates a matrix plot, bar plot, and heatmap plot.

    Parameters:
    -----------
    data_raw : dict
        The raw data in dictionary format.
    n_sample : int, optional (default: None)
        The number of rows to sample from the dataset if it is large.

    Returns:
    -------
    Tuple[str, Dict]:
        content: A message describing the generated plots.
        artifact: A dict with keys 'matrix_plot', 'bar_plot', and 'heatmap_plot' each containing the
                  corresponding base64 encoded PNG image.
    """
    print("    * Tool: visualize_missing")

    try:
        import missingno as msno  # Ensure missingno is installed
    except ImportError:
        raise ImportError(
            "Please install the 'missingno' package to use this tool. pip install missingno"
        )

    import pandas as pd
    import base64
    from io import BytesIO
    import matplotlib.pyplot as plt

    # Create the DataFrame and sample if n_sample is provided.
    df = pd.DataFrame(data_raw)
    if n_sample is not None:
        df = df.sample(n=n_sample, random_state=42)

    # Dictionary to store the base64 encoded images for each plot.
    encoded_plots = {}

    # Define a helper function to create a plot, save it, and encode it.
    def create_and_encode_plot(plot_func, plot_name: str):
        plt.figure(figsize=(8, 6))
        # Call the missingno plotting function.
        plot_func(df)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # Create and encode the matrix plot.
    encoded_plots["matrix_plot"] = create_and_encode_plot(msno.matrix, "matrix")

    # Create and encode the bar plot.
    encoded_plots["bar_plot"] = create_and_encode_plot(msno.bar, "bar")

    # Create and encode the heatmap plot.
    encoded_plots["heatmap_plot"] = create_and_encode_plot(msno.heatmap, "heatmap")

    content = (
        "Missing data visualizations (matrix, bar, and heatmap) have been generated."
    )
    artifact = encoded_plots
    return content, artifact


@tool(response_format="content_and_artifact")
def generate_correlation_funnel(
    data_raw: Annotated[dict, InjectedState("data_raw")],
    target: str,
    target_bin_index: Union[int, str] = -1,
    corr_method: str = "pearson",
    n_bins: int = 4,
    thresh_infreq: float = 0.01,
    name_infreq: str = "-OTHER",
) -> Tuple[str, Dict]:
    """
    Tool: generate_correlation_funnel
    Description:
        Correlation analysis using the correlation funnel method. The tool binarizes the data and computes correlation versus a target column.

    Parameters:
    ----------
    target : str
        The base target column name (e.g., 'Member_Status'). The tool will look for columns that begin
        with this string followed by '__' (e.g., 'Member_Status__Gold', 'Member_Status__Platinum').
    target_bin_index : int or str, default -1
        If an integer, selects the target level by position from the matching columns.
        If a string (e.g., "Yes"), attempts to match to the suffix of a column name
        (i.e., 'target__Yes').
    corr_method : str
        The correlation method ('pearson', 'kendall', or 'spearman'). Default is 'pearson'.
    n_bins : int
        The number of bins to use for binarization. Default is 4.
    thresh_infreq : float
        The threshold for infrequent levels. Default is 0.01.
    name_infreq : str
        The name to use for infrequent levels. Default is '-OTHER'.
    """
    print("    * Tool: generate_correlation_funnel")
    try:
        import pytimetk as tk
    except ImportError:
        raise ImportError(
            "Please install the 'pytimetk' package to use this tool. pip install pytimetk"
        )
    import pandas as pd
    import base64
    from io import BytesIO
    import matplotlib.pyplot as plt
    import json
    import plotly.io as pio

    # Convert the raw injected state into a DataFrame.
    df = pd.DataFrame(data_raw)

    # Apply the binarization method.
    df_binarized = df.binarize(
        n_bins=n_bins,
        thresh_infreq=thresh_infreq,
        name_infreq=name_infreq,
        one_hot=True,
    )

    # Determine the full target column name.
    # Look for all columns that start with "target__"
    matching_columns = [
        col for col in df_binarized.columns if col.startswith(f"{target}__")
    ]
    if not matching_columns:
        # If no matching columns are found, warn and use the provided target as-is.
        full_target = target
    else:
        # Determine the full target based on target_bin_index.
        if isinstance(target_bin_index, str):
            # Build the candidate column name
            candidate = f"{target}__{target_bin_index}"
            if candidate in matching_columns:
                full_target = candidate
            else:
                # If no matching candidate is found, default to the last matching column.
                full_target = matching_columns[-1]
        else:
            # target_bin_index is an integer.
            try:
                full_target = matching_columns[target_bin_index]
            except IndexError:
                # If index is out of bounds, use the last matching column.
                full_target = matching_columns[-1]

    # Compute correlation funnel using the full target column name.
    df_correlated = df_binarized.correlate(target=full_target, method=corr_method)

    # Attempt to generate a static plot.
    encoded = None
    try:
        # Here we assume that your DataFrame has a method plot_correlation_funnel.
        fig = df_correlated.plot_correlation_funnel(engine="plotnine", height=600)
        buf = BytesIO()
        # Use the appropriate save method for your figure object.
        fig.save(buf, format="png")
        plt.close()
        buf.seek(0)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        encoded = {"error": str(e)}

    # Attempt to generate a Plotly plot.
    fig_dict = None
    try:
        fig = df_correlated.plot_correlation_funnel(engine="plotly", base_size=14)

        fig_json = pio.to_json(fig)
        fig_dict = json.loads(fig_json)
    except Exception as e:
        fig_dict = {"error": str(e)}

    content = (
        f"Correlation funnel computed using method '{corr_method}' for target level '{full_target}'. "
        f"Base target was '{target}' with target_bin_index '{target_bin_index}'."
    )
    artifact = {
        "correlation_data": df_correlated.to_dict(orient="list"),
        "plot_image": encoded,
        "plotly_figure": fig_dict,
    }
    return content, artifact


@tool(response_format="content_and_artifact")
def generate_sweetviz_report(
    data_raw: Annotated[dict, InjectedState("data_raw")],
    target: str = None,
    report_name: str = "sweetviz_report.html",
    report_directory: str = None,  # <-- Default to None
    open_browser: bool = False,
    include_html: bool = False,
) -> Tuple[str, Dict]:
    """
    Tool: generate_sweetviz_report
    Description:
        Make an Exploratory Data Analysis (EDA) report using the Sweetviz library.

    Parameters:
    -----------
    data_raw : dict
        The raw data injected as a dictionary (converted from a DataFrame).
    target : str, optional
        The target feature to analyze. Default is None.
    report_name : str, optional
        The file name to save the Sweetviz HTML report. Default is "sweetviz_report.html".
    report_directory : str, optional
        The directory where the report should be saved.
        If None, a temporary directory is created and used.
    open_browser : bool, optional
        Whether to open the report in a web browser. Default is False.
    include_html : bool, optional
        If True, includes the full HTML content in the returned artifact. Default is False.

    Returns:
    --------
    Tuple[str, Dict]:
        content: A summary message describing the generated report.
        artifact: A dictionary with the report file path and optionally the report's HTML content.
    """
    print("    * Tool: generate_sweetviz_report")

    # Import sweetviz
    try:
        import sweetviz as sv
    except ImportError:
        raise ImportError(
            "Please install the 'sweetviz' package to use this tool. Run: pip install sweetviz"
        )

    import pandas as pd

    # Convert injected raw data to a DataFrame.
    df = pd.DataFrame(data_raw)

    # If no directory is specified, use a temporary directory.
    if not report_directory:
        report_directory = tempfile.mkdtemp()
        print(f"    * Using temporary directory: {report_directory}")
    else:
        # Ensure user-specified directory exists.
        if not os.path.exists(report_directory):
            os.makedirs(report_directory)

    # Create the Sweetviz report.
    # Sweetviz internally calls warnings.filterwarnings on np.VisibleDeprecationWarning; this is removed in numpy>=2.
    import numpy as np

    # NumPy >= 2 removed VisibleDeprecationWarning; Sweetviz still references it directly.
    if not hasattr(np, "VisibleDeprecationWarning"):
        # Provide a compatible placeholder to avoid AttributeError inside Sweetviz.
        np.VisibleDeprecationWarning = DeprecationWarning

    visible_dep = getattr(np, "VisibleDeprecationWarning", DeprecationWarning)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=visible_dep)
        report = sv.analyze(df, target_feat=target)

    # Determine the full path for the report.
    full_report_path = os.path.join(report_directory, report_name)

    # Save the report to the specified HTML file.
    report.show_html(
        filepath=full_report_path,
        open_browser=open_browser,
    )

    html_content = None
    if include_html:
        try:
            with open(full_report_path, "r", encoding="utf-8") as f:
                html_content = f.read()
        except Exception:
            html_content = None

    content = (
        f"Sweetviz EDA report generated and saved as '{os.path.abspath(full_report_path)}'. "
        f"{'This was saved in a temporary directory.' if 'tmp' in report_directory else ''}"
    )
    artifact = {
        "report_file": os.path.abspath(full_report_path),
        "report_html": html_content,
    }
    return content, artifact


@tool(response_format="content_and_artifact")
def generate_dtale_report(
    data_raw: Annotated[dict, InjectedState("data_raw")],
    host: str = "localhost",
    port: int = 40000,
    open_browser: bool = False,
) -> Tuple[str, Dict]:
    """
    Tool: generate_dtale_report
    Description:
        Creates an interactive data exploration report using the dtale library.

    Parameters:
    -----------
    data_raw : dict
        The raw data in dictionary format.
    host : str, optional
        The host IP address to serve the dtale app. Default is "localhost".
    port : int, optional
        The port number to serve the dtale app. Default is 40000.
    open_browser : bool, optional
        Whether to open the report in a web browser. Default is False.

    Returns:
    --------
    Tuple[str, Dict]:
        content: A summary message describing the dtale report.
        artifact: A dictionary containing the URL of the dtale report.
    """
    print("    * Tool: generate_dtale_report")

    try:
        import dtale
    except ImportError:
        raise ImportError(
            "Please install the 'dtale' package to use this tool. Run: pip install dtale"
        )

    import pandas as pd

    df = pd.DataFrame(data_raw)

    # Create the dtale report
    d = dtale.show(df, host=host, port=port, open_browser=open_browser)

    content = f"Dtale report generated and available at: {d.main_url()}"
    artifact = {"dtale_url": d.main_url()}

    return content, artifact
