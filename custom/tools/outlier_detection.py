"""
Outlier Detection Tools

Tools for detecting, visualizing, and treating outliers using various statistical
and machine learning methods.
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from langchain.tools import tool
import json


@tool
def detect_outliers_zscore(
    data: dict,
    columns: list,
    threshold: float = 3.0
) -> str:
    """
    Detect outliers using Z-score method (parametric).

    Args:
        data: Dictionary representation of DataFrame
        columns: List of numeric columns to check
        threshold: Z-score threshold (default: 3.0)

    Returns:
        JSON string with outlier indices and statistics
    """
    df = pd.DataFrame(data)
    outlier_info = {}

    for col in columns:
        if col not in df.columns:
            continue

        clean_data = df[col].dropna()
        if len(clean_data) == 0:
            continue

        mean = clean_data.mean()
        std = clean_data.std()

        if std == 0:
            outlier_info[col] = {
                'method': 'z-score',
                'outlier_count': 0,
                'outlier_pct': 0.0,
                'outlier_indices': []
            }
            continue

        z_scores = np.abs((df[col] - mean) / std)
        outliers = z_scores > threshold

        outlier_info[col] = {
            'method': 'z-score',
            'threshold': threshold,
            'outlier_count': int(outliers.sum()),
            'outlier_pct': float(outliers.sum() / len(df) * 100),
            'outlier_indices': df[outliers].index.tolist(),
            'min_outlier_value': float(df.loc[outliers, col].min()) if outliers.sum() > 0 else None,
            'max_outlier_value': float(df.loc[outliers, col].max()) if outliers.sum() > 0 else None
        }

    return json.dumps(outlier_info, indent=2)


@tool
def detect_outliers_iqr(
    data: dict,
    columns: list,
    multiplier: float = 1.5
) -> str:
    """
    Detect outliers using Interquartile Range (IQR) method (non-parametric).

    Args:
        data: Dictionary representation of DataFrame
        columns: List of numeric columns to check
        multiplier: IQR multiplier (default: 1.5, use 3.0 for extreme outliers)

    Returns:
        JSON string with outlier indices and statistics
    """
    df = pd.DataFrame(data)
    outlier_info = {}

    for col in columns:
        if col not in df.columns:
            continue

        clean_data = df[col].dropna()
        if len(clean_data) == 0:
            continue

        Q1 = clean_data.quantile(0.25)
        Q3 = clean_data.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)

        outlier_info[col] = {
            'method': 'iqr',
            'multiplier': multiplier,
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'outlier_count': int(outliers.sum()),
            'outlier_pct': float(outliers.sum() / len(df) * 100),
            'outlier_indices': df[outliers].index.tolist(),
            'below_lower': int((df[col] < lower_bound).sum()),
            'above_upper': int((df[col] > upper_bound).sum())
        }

    return json.dumps(outlier_info, indent=2)


@tool
def detect_outliers_isolation_forest(
    data: dict,
    columns: list,
    contamination: float = 0.1,
    random_state: int = 42
) -> str:
    """
    Detect multivariate outliers using Isolation Forest algorithm.

    Args:
        data: Dictionary representation of DataFrame
        columns: List of numeric columns to use for detection
        contamination: Expected proportion of outliers (default: 0.1)
        random_state: Random seed for reproducibility

    Returns:
        JSON string with outlier indices and statistics
    """
    from sklearn.ensemble import IsolationForest

    df = pd.DataFrame(data)

    # Select numeric columns
    available_cols = [col for col in columns if col in df.columns]
    if not available_cols:
        return json.dumps({"error": "No valid columns found"})

    X = df[available_cols].dropna()
    if len(X) == 0:
        return json.dumps({"error": "No data after dropping NaN values"})

    # Fit Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )

    # Predict outliers (-1 for outliers, 1 for inliers)
    predictions = iso_forest.fit_predict(X)
    outlier_scores = iso_forest.score_samples(X)

    outliers = predictions == -1

    result = {
        'method': 'isolation_forest',
        'contamination': contamination,
        'columns_used': available_cols,
        'outlier_count': int(outliers.sum()),
        'outlier_pct': float(outliers.sum() / len(X) * 100),
        'outlier_indices': X[outliers].index.tolist(),
        'outlier_scores': {
            'mean': float(outlier_scores[outliers].mean()) if outliers.sum() > 0 else None,
            'min': float(outlier_scores[outliers].min()) if outliers.sum() > 0 else None,
            'max': float(outlier_scores[outliers].max()) if outliers.sum() > 0 else None
        }
    }

    return json.dumps(result, indent=2)


@tool
def detect_outliers_lof(
    data: dict,
    columns: list,
    n_neighbors: int = 20,
    contamination: float = 0.1
) -> str:
    """
    Detect outliers using Local Outlier Factor (LOF) algorithm.

    Args:
        data: Dictionary representation of DataFrame
        columns: List of numeric columns to use
        n_neighbors: Number of neighbors to consider
        contamination: Expected proportion of outliers

    Returns:
        JSON string with outlier indices and statistics
    """
    from sklearn.neighbors import LocalOutlierFactor

    df = pd.DataFrame(data)

    # Select numeric columns
    available_cols = [col for col in columns if col in df.columns]
    if not available_cols:
        return json.dumps({"error": "No valid columns found"})

    X = df[available_cols].dropna()
    if len(X) == 0:
        return json.dumps({"error": "No data after dropping NaN values"})

    # Fit LOF
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        n_jobs=-1
    )

    predictions = lof.fit_predict(X)
    outlier_scores = lof.negative_outlier_factor_

    outliers = predictions == -1

    result = {
        'method': 'local_outlier_factor',
        'n_neighbors': n_neighbors,
        'contamination': contamination,
        'columns_used': available_cols,
        'outlier_count': int(outliers.sum()),
        'outlier_pct': float(outliers.sum() / len(X) * 100),
        'outlier_indices': X[outliers].index.tolist(),
        'outlier_factors': {
            'mean': float(outlier_scores[outliers].mean()) if outliers.sum() > 0 else None,
            'min': float(outlier_scores[outliers].min()) if outliers.sum() > 0 else None,
            'max': float(outlier_scores[outliers].max()) if outliers.sum() > 0 else None
        }
    }

    return json.dumps(result, indent=2)


@tool
def visualize_outliers(
    data: dict,
    outlier_indices: list,
    columns: list,
    method_name: str = "Outlier Detection"
) -> str:
    """
    Create visualizations showing outliers in the data.

    Args:
        data: Dictionary representation of DataFrame
        outlier_indices: List of outlier row indices
        columns: Columns to visualize
        method_name: Name of detection method for title

    Returns:
        JSON string of Plotly figure
    """
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    df = pd.DataFrame(data)

    # Create binary outlier column
    df['is_outlier'] = False
    df.loc[outlier_indices, 'is_outlier'] = True

    # Create subplots
    n_cols = min(len(columns), 2)
    n_rows = (len(columns) + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=columns
    )

    for idx, col in enumerate(columns):
        if col not in df.columns:
            continue

        row = idx // n_cols + 1
        col_num = idx % n_cols + 1

        # Normal points
        normal_data = df[~df['is_outlier']]
        fig.add_trace(
            go.Box(
                y=normal_data[col],
                name='Normal',
                marker=dict(color='lightblue'),
                boxmean='sd'
            ),
            row=row,
            col=col_num
        )

        # Outlier points
        if len(outlier_indices) > 0:
            outlier_data = df[df['is_outlier']]
            fig.add_trace(
                go.Scatter(
                    y=outlier_data[col],
                    x=['Outliers'] * len(outlier_data),
                    mode='markers',
                    name='Outliers',
                    marker=dict(color='red', size=10, symbol='x')
                ),
                row=row,
                col=col_num
            )

    fig.update_layout(
        title=f'Outlier Visualization - {method_name}',
        height=400 * n_rows,
        showlegend=True
    )

    return pio.to_json(fig)


@tool
def suggest_outlier_treatment(
    outlier_info: str,
    treatment_strategy: str = "auto"
) -> str:
    """
    Suggest appropriate treatment strategies for detected outliers.

    Args:
        outlier_info: JSON string with outlier detection results
        treatment_strategy: 'auto', 'remove', 'cap', 'transform', 'impute'

    Returns:
        Treatment recommendations
    """
    info = json.loads(outlier_info)

    report = "OUTLIER TREATMENT RECOMMENDATIONS\n"
    report += "=" * 80 + "\n\n"

    for col, col_info in info.items():
        if isinstance(col_info, dict) and 'outlier_count' in col_info:
            outlier_count = col_info['outlier_count']
            outlier_pct = col_info['outlier_pct']

            report += f"Column: {col}\n"
            report += f"  Outliers detected: {outlier_count} ({outlier_pct:.2f}%)\n"

            # Auto-suggest based on percentage
            if treatment_strategy == "auto":
                if outlier_pct < 1:
                    strategy = "REMOVE"
                    report += f"  Recommended: {strategy}\n"
                    report += f"    Rationale: Very few outliers (<1%), safe to remove\n"
                elif outlier_pct < 5:
                    strategy = "CAP (Winsorization)"
                    report += f"  Recommended: {strategy}\n"
                    report += f"    Rationale: Moderate outliers (1-5%), cap to threshold values\n"
                    if 'lower_bound' in col_info:
                        report += f"    Lower bound: {col_info['lower_bound']:.4f}\n"
                        report += f"    Upper bound: {col_info['upper_bound']:.4f}\n"
                elif outlier_pct < 10:
                    strategy = "TRANSFORM"
                    report += f"  Recommended: {strategy}\n"
                    report += f"    Rationale: Significant outliers (5-10%), try log/sqrt transform\n"
                else:
                    strategy = "INVESTIGATE"
                    report += f"  Recommended: {strategy}\n"
                    report += f"    Rationale: Many outliers (>10%), may not be outliers but valid extreme values\n"

            else:
                report += f"  Strategy: {treatment_strategy.upper()}\n"

            report += "\n"

    # General recommendations
    report += "-" * 80 + "\n"
    report += "GENERAL GUIDELINES:\n"
    report += "  1. REMOVE: Use when outliers are errors/mistakes (<1% of data)\n"
    report += "  2. CAP: Limit values to upper/lower bounds (1-5% outliers)\n"
    report += "  3. TRANSFORM: Apply log, sqrt, or Box-Cox transformation (5-10% outliers)\n"
    report += "  4. SEPARATE MODEL: Build separate model for outlier segment\n"
    report += "  5. ROBUST METHODS: Use algorithms resistant to outliers (e.g., tree-based)\n"
    report += "\n" + "=" * 80 + "\n"

    return report
