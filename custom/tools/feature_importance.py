"""
Feature Importance Visualization Tools

Tools for analyzing and visualizing feature importance from ML models.
Supports multiple model types and visualization methods.
"""

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from langchain.tools import tool
import json


@tool
def calculate_permutation_importance(
    model: Any,
    X: dict,
    y: dict,
    n_repeats: int = 10,
    random_state: int = 42
) -> str:
    """
    Calculate permutation feature importance for any scikit-learn compatible model.

    Args:
        model: Trained model with predict method
        X: Feature data as dictionary
        y: Target data as dictionary
        n_repeats: Number of times to permute each feature
        random_state: Random seed for reproducibility

    Returns:
        JSON string with feature importance scores
    """
    from sklearn.inspection import permutation_importance

    X_df = pd.DataFrame(X)
    y_series = pd.Series(y) if isinstance(y, dict) else y

    result = permutation_importance(
        model, X_df, y_series,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        'feature': X_df.columns,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False)

    return importance_df.to_json(orient='records')


@tool
def extract_tree_importance(model: Any, feature_names: list) -> str:
    """
    Extract feature importance from tree-based models (Random Forest, XGBoost, etc.).

    Args:
        model: Trained tree-based model
        feature_names: List of feature names

    Returns:
        JSON string with feature importance scores
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'get_score'):
        # XGBoost specific
        importance_dict = model.get_score(importance_type='gain')
        importances = [importance_dict.get(f, 0) for f in feature_names]
    else:
        return json.dumps({"error": "Model does not have feature_importances_ attribute"})

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    return importance_df.to_json(orient='records')


@tool
def calculate_shap_importance(
    model: Any,
    X: dict,
    sample_size: int = 100
) -> str:
    """
    Calculate SHAP (SHapley Additive exPlanations) feature importance.

    Args:
        model: Trained model
        X: Feature data as dictionary
        sample_size: Number of samples to use for SHAP calculation

    Returns:
        JSON string with SHAP importance values
    """
    try:
        import shap
    except ImportError:
        return json.dumps({"error": "SHAP library not installed. Run: pip install shap"})

    X_df = pd.DataFrame(X)

    # Sample data if too large
    if len(X_df) > sample_size:
        X_sample = X_df.sample(n=sample_size, random_state=42)
    else:
        X_sample = X_df

    try:
        # Try TreeExplainer first (faster for tree models)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except:
        # Fall back to KernelExplainer
        explainer = shap.KernelExplainer(model.predict, X_sample)
        shap_values = explainer.shap_values(X_sample)

    # Handle multi-class case
    if isinstance(shap_values, list):
        shap_values = np.abs(shap_values[0])

    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        'feature': X_df.columns,
        'shap_importance': mean_abs_shap
    }).sort_values('shap_importance', ascending=False)

    return importance_df.to_json(orient='records')


@tool
def plot_feature_importance_bar(
    importance_data: str,
    top_n: int = 20,
    title: str = "Feature Importance"
) -> str:
    """
    Create a bar plot of feature importance.

    Args:
        importance_data: JSON string of importance scores
        top_n: Number of top features to display
        title: Plot title

    Returns:
        JSON string of Plotly figure
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    # Parse importance data
    importance_df = pd.read_json(importance_data)

    # Get column with importance scores
    importance_col = [col for col in importance_df.columns if 'importance' in col.lower()][0]

    # Select top N features
    top_features = importance_df.nlargest(top_n, importance_col)

    # Create bar plot
    fig = go.Figure([
        go.Bar(
            x=top_features[importance_col],
            y=top_features['feature'],
            orientation='h',
            marker=dict(
                color=top_features[importance_col],
                colorscale='Viridis',
                showscale=True
            )
        )
    ])

    fig.update_layout(
        title=title,
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        height=max(400, top_n * 20),
        yaxis={'categoryorder': 'total ascending'}
    )

    return pio.to_json(fig)


@tool
def compare_importance_methods(
    importance_data_list: List[str],
    method_names: List[str]
) -> str:
    """
    Compare feature importance across different methods.

    Args:
        importance_data_list: List of JSON strings with importance scores
        method_names: Names of each importance method

    Returns:
        JSON string of comparison DataFrame and plot
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    # Parse all importance data
    dfs = [pd.read_json(data) for data in importance_data_list]

    # Standardize column names
    for df, method_name in zip(dfs, method_names):
        importance_col = [col for col in df.columns if 'importance' in col.lower()][0]
        df.rename(columns={importance_col: method_name}, inplace=True)

    # Merge all dataframes
    comparison_df = dfs[0][['feature', method_names[0]]]
    for df, method_name in zip(dfs[1:], method_names[1:]):
        comparison_df = comparison_df.merge(
            df[['feature', method_name]],
            on='feature',
            how='outer'
        )

    # Fill missing values
    comparison_df.fillna(0, inplace=True)

    # Normalize scores to 0-1 range for each method
    for method in method_names:
        max_val = comparison_df[method].max()
        if max_val > 0:
            comparison_df[f'{method}_normalized'] = comparison_df[method] / max_val

    # Calculate average importance
    normalized_cols = [f'{m}_normalized' for m in method_names]
    comparison_df['avg_importance'] = comparison_df[normalized_cols].mean(axis=1)

    # Sort by average importance
    comparison_df = comparison_df.sort_values('avg_importance', ascending=False)

    # Create comparison plot
    fig = go.Figure()

    top_20 = comparison_df.head(20)
    for method in method_names:
        fig.add_trace(go.Bar(
            name=method,
            x=top_20['feature'],
            y=top_20[f'{method}_normalized'],
        ))

    fig.update_layout(
        title='Feature Importance Comparison Across Methods',
        xaxis_title='Feature',
        yaxis_title='Normalized Importance',
        barmode='group',
        height=500,
        xaxis={'tickangle': -45}
    )

    result = {
        'comparison_table': comparison_df.to_dict(orient='records'),
        'plot': json.loads(pio.to_json(fig))
    }

    return json.dumps(result)


@tool
def generate_importance_report(
    importance_data: str,
    model_name: str = "Model",
    method: str = "Unknown"
) -> str:
    """
    Generate a comprehensive feature importance report.

    Args:
        importance_data: JSON string of importance scores
        model_name: Name of the model
        method: Importance calculation method used

    Returns:
        Formatted text report
    """
    importance_df = pd.read_json(importance_data)
    importance_col = [col for col in importance_df.columns if 'importance' in col.lower()][0]

    report = f"FEATURE IMPORTANCE REPORT\n"
    report += f"{'=' * 80}\n\n"
    report += f"Model: {model_name}\n"
    report += f"Method: {method}\n"
    report += f"Total Features: {len(importance_df)}\n\n"

    # Top features
    report += "TOP 20 MOST IMPORTANT FEATURES:\n"
    report += f"{'-' * 80}\n"
    report += f"{'Rank':<6} {'Feature':<40} {'Importance':<15}\n"
    report += f"{'-' * 80}\n"

    top_20 = importance_df.head(20)
    for idx, row in enumerate(top_20.itertuples(), 1):
        feature = row.feature
        importance = getattr(row, importance_col)
        report += f"{idx:<6} {feature:<40} {importance:<15.6f}\n"

    # Statistics
    report += f"\n{'-' * 80}\n"
    report += "IMPORTANCE STATISTICS:\n"
    report += f"  Mean: {importance_df[importance_col].mean():.6f}\n"
    report += f"  Std:  {importance_df[importance_col].std():.6f}\n"
    report += f"  Min:  {importance_df[importance_col].min():.6f}\n"
    report += f"  Max:  {importance_df[importance_col].max():.6f}\n"

    # Cumulative importance
    cumsum = importance_df[importance_col].cumsum()
    total = importance_df[importance_col].sum()

    features_for_80pct = (cumsum / total <= 0.8).sum()
    features_for_90pct = (cumsum / total <= 0.9).sum()

    report += f"\nCUMULATIVE IMPORTANCE:\n"
    report += f"  Features for 80% importance: {features_for_80pct} ({features_for_80pct/len(importance_df)*100:.1f}%)\n"
    report += f"  Features for 90% importance: {features_for_90pct} ({features_for_90pct/len(importance_df)*100:.1f}%)\n"

    # Low importance features
    threshold = importance_df[importance_col].quantile(0.1)
    low_importance = importance_df[importance_df[importance_col] < threshold]

    report += f"\nLOW IMPORTANCE FEATURES (bottom 10%):\n"
    report += f"  Count: {len(low_importance)}\n"
    report += f"  Features: {', '.join(low_importance['feature'].head(10).tolist())}\n"

    report += f"\n{'=' * 80}\n"

    return report
