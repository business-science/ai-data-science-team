"""
Model Comparison Tools

Tools for comparing multiple ML models across various metrics and visualizations.
"""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from langchain.tools import tool
import json


@tool
def compare_classification_metrics(
    models_predictions: dict,
    y_true: dict,
    model_names: list
) -> str:
    """
    Compare classification models across multiple metrics.

    Args:
        models_predictions: Dict mapping model names to their predictions
        y_true: True labels
        model_names: List of model names

    Returns:
        JSON string with comparison metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )

    y_true_array = np.array(list(y_true.values()) if isinstance(y_true, dict) else y_true)

    results = []

    for model_name in model_names:
        y_pred = np.array(models_predictions[model_name])

        metrics = {
            'model': model_name,
            'accuracy': accuracy_score(y_true_array, y_pred),
            'precision': precision_score(y_true_array, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true_array, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true_array, y_pred, average='weighted', zero_division=0),
        }

        # Try to calculate AUC if probability predictions available
        try:
            if f"{model_name}_proba" in models_predictions:
                y_proba = np.array(models_predictions[f"{model_name}_proba"])
                metrics['roc_auc'] = roc_auc_score(y_true_array, y_proba, multi_class='ovr', average='weighted')
        except:
            metrics['roc_auc'] = None

        results.append(metrics)

    results_df = pd.DataFrame(results)
    return results_df.to_json(orient='records')


@tool
def compare_regression_metrics(
    models_predictions: dict,
    y_true: dict,
    model_names: list
) -> str:
    """
    Compare regression models across multiple metrics.

    Args:
        models_predictions: Dict mapping model names to their predictions
        y_true: True values
        model_names: List of model names

    Returns:
        JSON string with comparison metrics
    """
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error,
        r2_score, mean_absolute_percentage_error
    )

    y_true_array = np.array(list(y_true.values()) if isinstance(y_true, dict) else y_true)

    results = []

    for model_name in model_names:
        y_pred = np.array(models_predictions[model_name])

        metrics = {
            'model': model_name,
            'mse': mean_squared_error(y_true_array, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true_array, y_pred)),
            'mae': mean_absolute_error(y_true_array, y_pred),
            'r2': r2_score(y_true_array, y_pred),
            'mape': mean_absolute_percentage_error(y_true_array, y_pred)
        }

        results.append(metrics)

    results_df = pd.DataFrame(results)
    return results_df.to_json(orient='records')


@tool
def plot_roc_curves(
    models_predictions: dict,
    y_true: dict,
    model_names: list
) -> str:
    """
    Plot ROC curves for multiple classification models on one chart.

    Args:
        models_predictions: Dict with model predictions and probabilities
        y_true: True labels
        model_names: List of model names

    Returns:
        JSON string of Plotly figure
    """
    from sklearn.metrics import roc_curve, auc
    import plotly.graph_objects as go
    import plotly.io as pio

    y_true_array = np.array(list(y_true.values()) if isinstance(y_true, dict) else y_true)

    fig = go.Figure()

    for model_name in model_names:
        proba_key = f"{model_name}_proba"
        if proba_key in models_predictions:
            y_proba = np.array(models_predictions[proba_key])

            # Handle binary classification
            if len(y_proba.shape) > 1 and y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]

            fpr, tpr, _ = roc_curve(y_true_array, y_proba)
            roc_auc = auc(fpr, tpr)

            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'{model_name} (AUC = {roc_auc:.3f})',
                line=dict(width=2)
            ))

    # Add diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray')
    ))

    fig.update_layout(
        title='ROC Curve Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=600,
        width=800,
        hovermode='closest'
    )

    return pio.to_json(fig)


@tool
def plot_prediction_comparison(
    models_predictions: dict,
    y_true: dict,
    model_names: list,
    task_type: str = "regression"
) -> str:
    """
    Plot actual vs predicted values for regression or classification models.

    Args:
        models_predictions: Dict with model predictions
        y_true: True values/labels
        model_names: List of model names
        task_type: 'regression' or 'classification'

    Returns:
        JSON string of Plotly figure
    """
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    y_true_array = np.array(list(y_true.values()) if isinstance(y_true, dict) else y_true)

    n_models = len(model_names)
    cols = min(2, n_models)
    rows = (n_models + cols - 1) // cols

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=model_names
    )

    for idx, model_name in enumerate(model_names):
        row = idx // cols + 1
        col = idx % cols + 1

        y_pred = np.array(models_predictions[model_name])

        if task_type == "regression":
            # Scatter plot for regression
            fig.add_trace(
                go.Scatter(
                    x=y_true_array,
                    y=y_pred,
                    mode='markers',
                    name=model_name,
                    marker=dict(size=5, opacity=0.6)
                ),
                row=row,
                col=col
            )

            # Add perfect prediction line
            min_val = min(y_true_array.min(), y_pred.min())
            max_val = max(y_true_array.max(), y_pred.max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(dash='dash', color='red'),
                    showlegend=False
                ),
                row=row,
                col=col
            )

        fig.update_xaxes(title_text="Actual", row=row, col=col)
        fig.update_yaxes(title_text="Predicted", row=row, col=col)

    fig.update_layout(
        title='Prediction Comparison',
        height=400 * rows,
        width=800,
        showlegend=False
    )

    return pio.to_json(fig)


@tool
def create_model_comparison_table(
    metrics_data: str,
    highlight_best: bool = True
) -> str:
    """
    Create a formatted comparison table from metrics data.

    Args:
        metrics_data: JSON string with model metrics
        highlight_best: Whether to highlight best values

    Returns:
        HTML table string
    """
    metrics_df = pd.read_json(metrics_data)

    # Determine which metrics should be minimized vs maximized
    minimize_metrics = ['mse', 'rmse', 'mae', 'mape']
    maximize_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'r2', 'roc_auc']

    if highlight_best:
        styled_df = metrics_df.copy()

        for col in metrics_df.columns:
            if col == 'model':
                continue

            if col in minimize_metrics:
                best_val = metrics_df[col].min()
            elif col in maximize_metrics:
                best_val = metrics_df[col].max()
            else:
                continue

            # Mark best values
            styled_df[f'{col}_is_best'] = metrics_df[col] == best_val

    # Convert to HTML
    html = metrics_df.to_html(index=False, float_format=lambda x: f'{x:.4f}')

    return html


@tool
def generate_model_comparison_report(
    metrics_data: str,
    model_names: list,
    task_type: str = "classification"
) -> str:
    """
    Generate a comprehensive model comparison report.

    Args:
        metrics_data: JSON string with model metrics
        model_names: List of model names
        task_type: 'classification' or 'regression'

    Returns:
        Formatted text report
    """
    metrics_df = pd.read_json(metrics_data)

    report = f"MODEL COMPARISON REPORT\n"
    report += f"{'=' * 80}\n\n"
    report += f"Task Type: {task_type.capitalize()}\n"
    report += f"Models Compared: {len(model_names)}\n\n"

    # Summary table
    report += "PERFORMANCE METRICS:\n"
    report += f"{'-' * 80}\n"
    report += metrics_df.to_string(index=False, float_format=lambda x: f'{x:.4f}')
    report += f"\n{'-' * 80}\n\n"

    # Best models per metric
    report += "BEST MODELS PER METRIC:\n"
    report += f"{'-' * 80}\n"

    if task_type == "classification":
        metrics_to_check = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in metrics_to_check:
            if metric in metrics_df.columns:
                best_idx = metrics_df[metric].idxmax()
                best_model = metrics_df.loc[best_idx, 'model']
                best_value = metrics_df.loc[best_idx, metric]
                report += f"  {metric:15s}: {best_model:20s} ({best_value:.4f})\n"
    else:
        # Regression
        minimize = ['mse', 'rmse', 'mae', 'mape']
        maximize = ['r2']

        for metric in minimize:
            if metric in metrics_df.columns:
                best_idx = metrics_df[metric].idxmin()
                best_model = metrics_df.loc[best_idx, 'model']
                best_value = metrics_df.loc[best_idx, metric]
                report += f"  {metric:15s}: {best_model:20s} ({best_value:.4f})\n"

        for metric in maximize:
            if metric in metrics_df.columns:
                best_idx = metrics_df[metric].idxmax()
                best_model = metrics_df.loc[best_idx, 'model']
                best_value = metrics_df.loc[best_idx, metric]
                report += f"  {metric:15s}: {best_model:20s} ({best_value:.4f})\n"

    # Overall recommendation
    report += f"\n{'-' * 80}\n"
    report += "RECOMMENDATION:\n"

    if task_type == "classification" and 'f1_score' in metrics_df.columns:
        best_overall_idx = metrics_df['f1_score'].idxmax()
        best_overall = metrics_df.loc[best_overall_idx, 'model']
        report += f"  Based on F1-Score, '{best_overall}' appears to be the best overall model.\n"
    elif task_type == "regression" and 'r2' in metrics_df.columns:
        best_overall_idx = metrics_df['r2'].idxmax()
        best_overall = metrics_df.loc[best_overall_idx, 'model']
        report += f"  Based on R², '{best_overall}' appears to be the best overall model.\n"

    report += f"\n{'=' * 80}\n"

    return report
