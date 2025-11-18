"""
Custom Tools

Your custom tools and utilities.
"""

# Data Quality Tools
from .data_quality import (
    check_schema_compliance,
    detect_data_anomalies,
    validate_business_rules,
    calculate_data_quality_score,
    generate_data_quality_report,
)

# Feature Importance Tools
from .feature_importance import (
    calculate_permutation_importance,
    extract_tree_importance,
    calculate_shap_importance,
    plot_feature_importance_bar,
    compare_importance_methods,
    generate_importance_report,
)

# Model Comparison Tools
from .model_comparison import (
    compare_classification_metrics,
    compare_regression_metrics,
    plot_roc_curves,
    plot_prediction_comparison,
    create_model_comparison_table,
    generate_model_comparison_report,
)

# Outlier Detection Tools
from .outlier_detection import (
    detect_outliers_zscore,
    detect_outliers_iqr,
    detect_outliers_isolation_forest,
    detect_outliers_lof,
    visualize_outliers,
    suggest_outlier_treatment,
)

# Time Series Tools
from .time_series import (
    detect_seasonality,
    decompose_time_series,
    check_stationarity,
    forecast_baseline,
    visualize_time_series,
    calculate_time_series_metrics,
)

__all__ = [
    # Data Quality
    'check_schema_compliance',
    'detect_data_anomalies',
    'validate_business_rules',
    'calculate_data_quality_score',
    'generate_data_quality_report',
    # Feature Importance
    'calculate_permutation_importance',
    'extract_tree_importance',
    'calculate_shap_importance',
    'plot_feature_importance_bar',
    'compare_importance_methods',
    'generate_importance_report',
    # Model Comparison
    'compare_classification_metrics',
    'compare_regression_metrics',
    'plot_roc_curves',
    'plot_prediction_comparison',
    'create_model_comparison_table',
    'generate_model_comparison_report',
    # Outlier Detection
    'detect_outliers_zscore',
    'detect_outliers_iqr',
    'detect_outliers_isolation_forest',
    'detect_outliers_lof',
    'visualize_outliers',
    'suggest_outlier_treatment',
    # Time Series
    'detect_seasonality',
    'decompose_time_series',
    'check_stationarity',
    'forecast_baseline',
    'visualize_time_series',
    'calculate_time_series_metrics',
]
