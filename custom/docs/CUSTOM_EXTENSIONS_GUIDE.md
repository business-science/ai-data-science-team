
# Custom Extensions Guide

This document provides comprehensive documentation for all custom agents and tools in your fork.

## Table of Contents

1. [Overview](#overview)
2. [Data Quality Agent](#1-data-quality-agent)
3. [Feature Importance Agent](#2-feature-importance-agent)
4. [Model Comparison Agent](#3-model-comparison-agent)
5. [Outlier Detection Agent](#4-outlier-detection-agent)
6. [Time Series Agent](#5-time-series-agent)
7. [Installation & Setup](#installation--setup)
8. [Quick Start Examples](#quick-start-examples)

---

## Overview

This custom extension package adds 5 specialized AI agents, each with 5-6 dedicated tools. All agents follow the same pattern: they use LangChain for agent orchestration and provide both full agent-based analysis and quick utility methods.

### Custom Agents

| Agent | Purpose | Tools Count |
|-------|---------|-------------|
| **Data Quality Agent** | Validate and assess data quality | 5 tools |
| **Feature Importance Agent** | Analyze feature importance from ML models | 6 tools |
| **Model Comparison Agent** | Compare multiple ML models | 6 tools |
| **Outlier Detection Agent** | Detect and treat outliers | 6 tools |
| **Time Series Agent** | Analyze temporal data and forecast | 6 tools |

**Total:** 29 custom tools across 5 agents

---

## 1. Data Quality Agent

### Purpose
Comprehensive data quality assessment, validation, and reporting.

### Tools (5)

1. **check_schema_compliance** - Validate DataFrame schema against expected definition
2. **detect_data_anomalies** - Detect missing values, duplicates, outliers, format issues
3. **validate_business_rules** - Enforce custom business logic rules
4. **calculate_data_quality_score** - Calculate overall quality score (0-100)
5. **generate_data_quality_report** - Create comprehensive quality report

### Usage Example

```python
from langchain_openai import ChatOpenAI
from custom.agents import DataQualityAgent
import pandas as pd

# Initialize
llm = ChatOpenAI(model="gpt-4")
agent = DataQualityAgent(model=llm)

# Load data
df = pd.read_csv("customer_data.csv")

# Full quality assessment with AI reasoning
result = agent.invoke(
    data=df,
    task="Perform comprehensive quality check and provide recommendations"
)
print(result['output'])

# Quick check without AI reasoning
quick_result = agent.quick_check(df)
print(quick_result)

# Schema validation
schema = {
    "customer_id": "int64",
    "age": "int64",
    "email": "object",
    "signup_date": "datetime64[ns]"
}
schema_result = agent.validate_schema(df, schema)
print(schema_result)

# Business rules validation
rules = {
    "age_valid": lambda df: (df['age'] >= 18) & (df['age'] <= 120),
    "email_format": lambda df: df['email'].str.contains('@', na=False)
}
rules_result = agent.check_business_rules(df, rules)
print(rules_result)
```

### Key Features

- ✅ Schema validation with type checking
- ✅ Anomaly detection (missing values, outliers, duplicates)
- ✅ Business rule enforcement
- ✅ Quality scoring (0-100 scale)
- ✅ Detailed quality reports

---

## 2. Feature Importance Agent

### Purpose
Analyze and visualize which features are most important in ML models.

### Tools (6)

1. **extract_tree_importance** - Extract importance from tree-based models (RF, XGBoost)
2. **calculate_permutation_importance** - Model-agnostic permutation importance
3. **calculate_shap_importance** - SHAP (SHapley Additive exPlanations) values
4. **plot_feature_importance_bar** - Create bar plots of importance
5. **compare_importance_methods** - Compare different importance methods
6. **generate_importance_report** - Comprehensive importance analysis report

### Usage Example

```python
from langchain_openai import ChatOpenAI
from custom.agents import FeatureImportanceAgent
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Train a model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Initialize agent
llm = ChatOpenAI(model="gpt-4")
agent = FeatureImportanceAgent(model=llm)

# Full analysis with AI insights
result = agent.analyze_importance(
    model=rf,
    X=X_test,
    y=y_test,
    feature_names=X_train.columns.tolist(),
    compare_methods=True
)
print(result['output'])

# Quick importance extraction
quick_result = agent.quick_importance(
    model=rf,
    feature_names=X_train.columns.tolist(),
    top_n=15
)
print(quick_result['report'])

# Compare multiple methods
comparison = agent.compare_methods(
    model=rf,
    X=X_test,
    y=y_test,
    feature_names=X_train.columns.tolist(),
    methods=['tree', 'permutation', 'shap']
)
print(comparison)

# Identify low-importance features
redundant = agent.identify_redundant_features(
    model=rf,
    feature_names=X_train.columns.tolist(),
    threshold=0.01
)
print(f"Consider removing: {redundant}")
```

### Key Features

- ✅ Multiple importance calculation methods
- ✅ Interactive Plotly visualizations
- ✅ Cross-method comparison
- ✅ SHAP integration for model-agnostic analysis
- ✅ Feature selection recommendations

---

## 3. Model Comparison Agent

### Purpose
Compare multiple ML models across various metrics with visualizations.

### Tools (6)

1. **compare_classification_metrics** - Compare accuracy, precision, recall, F1, AUC
2. **compare_regression_metrics** - Compare MSE, RMSE, MAE, R², MAPE
3. **plot_roc_curves** - Multi-model ROC curve comparison
4. **plot_prediction_comparison** - Actual vs predicted plots
5. **create_model_comparison_table** - HTML comparison tables
6. **generate_model_comparison_report** - Comprehensive comparison report

### Usage Example

```python
from langchain_openai import ChatOpenAI
from custom.agents import ModelComparisonAgent
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

# Train multiple models
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100),
    'LogisticRegression': LogisticRegression()
}

for name, model in models.items():
    model.fit(X_train, y_train)

# Get predictions
predictions = {}
for name, model in models.items():
    predictions[name] = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        predictions[f'{name}_proba'] = model.predict_proba(X_test)

# Initialize agent
llm = ChatOpenAI(model="gpt-4")
agent = ModelComparisonAgent(model=llm)

# Full comparison with AI insights
result = agent.compare_models(
    predictions=predictions,
    y_true=y_test,
    model_names=list(models.keys()),
    task_type='classification',
    generate_plots=True
)
print(result['output'])

# Quick comparison
quick_result = agent.quick_comparison(
    predictions=predictions,
    y_true=y_test,
    model_names=list(models.keys()),
    task_type='classification'
)
print(quick_result['report'])

# ROC curve comparison
roc_plot = agent.plot_roc_comparison(
    predictions=predictions,
    y_true=y_test,
    model_names=list(models.keys())
)

# Rank models
rankings = agent.rank_models(
    predictions=predictions,
    y_true=y_test,
    model_names=list(models.keys()),
    task_type='classification',
    ranking_metric='f1_score'
)
print("Model Rankings:", rankings)
```

### Key Features

- ✅ Classification and regression support
- ✅ ROC curves and prediction plots
- ✅ Side-by-side metric comparison
- ✅ Automatic best model recommendation
- ✅ Exportable HTML tables

---

## 4. Outlier Detection Agent

### Purpose
Detect, visualize, and treat outliers using multiple statistical and ML methods.

### Tools (6)

1. **detect_outliers_zscore** - Parametric Z-score method
2. **detect_outliers_iqr** - Non-parametric IQR method
3. **detect_outliers_isolation_forest** - Multivariate Isolation Forest
4. **detect_outliers_lof** - Local Outlier Factor (density-based)
5. **visualize_outliers** - Interactive outlier visualizations
6. **suggest_outlier_treatment** - Treatment strategy recommendations

### Usage Example

```python
from langchain_openai import ChatOpenAI
from custom.agents import OutlierDetectionAgent
import pandas as pd
import numpy as np

# Create sample data with outliers
np.random.seed(42)
data = {
    'price': np.concatenate([np.random.normal(100, 15, 95), [500, 600, 700, 800, 900]]),
    'age': np.concatenate([np.random.normal(35, 10, 95), [150, 160, -5, -10, 200]]),
    'income': np.concatenate([np.random.normal(50000, 10000, 95), [500000, 600000, 700000, 800000, 900000]])
}
df = pd.DataFrame(data)

# Initialize agent
llm = ChatOpenAI(model="gpt-4")
agent = OutlierDetectionAgent(model=llm)

# Full analysis with AI reasoning
result = agent.detect_outliers(
    data=df,
    columns=['price', 'age', 'income'],
    methods=['zscore', 'iqr', 'isolation_forest'],
    compare_methods=True,
    visualize=True
)
print(result['output'])

# Quick detection
quick_result = agent.quick_detect(
    data=df,
    columns=['price', 'age'],
    method='iqr'
)
print(quick_result['treatment_recommendations'])

# Compare detection methods
comparison = agent.compare_methods(
    data=df,
    columns=['price', 'age', 'income'],
    methods=['zscore', 'iqr', 'isolation_forest']
)
print(comparison)

# Get consensus outliers (flagged by multiple methods)
consensus = agent.get_consensus_outliers(
    data=df,
    columns=['price', 'age', 'income'],
    methods=['zscore', 'iqr'],
    min_methods=2
)
print(f"High-confidence outliers: {consensus}")
```

### Key Features

- ✅ Multiple detection methods (univariate & multivariate)
- ✅ Interactive visualizations
- ✅ Cross-method consensus detection
- ✅ Automatic treatment recommendations
- ✅ Configurable sensitivity thresholds

---

## 5. Time Series Agent

### Purpose
Analyze temporal data including trend, seasonality, stationarity, and forecasting.

### Tools (6)

1. **detect_seasonality** - Autocorrelation-based seasonality detection
2. **decompose_time_series** - Trend, seasonal, residual decomposition
3. **check_stationarity** - Augmented Dickey-Fuller test
4. **forecast_baseline** - Simple forecast methods (naive, mean, drift, seasonal)
5. **visualize_time_series** - Time series and forecast visualizations
6. **calculate_time_series_metrics** - Comprehensive temporal metrics

### Usage Example

```python
from langchain_openai import ChatOpenAI
from custom.agents import TimeSeriesAgent
import pandas as pd
import numpy as np

# Create sample time series
dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
trend = np.linspace(100, 150, 365)
seasonality = 10 * np.sin(2 * np.pi * np.arange(365) / 7)  # Weekly
noise = np.random.normal(0, 3, 365)
values = trend + seasonality + noise

df = pd.DataFrame({'date': dates, 'sales': values})

# Initialize agent
llm = ChatOpenAI(model="gpt-4")
agent = TimeSeriesAgent(model=llm)

# Full analysis with AI insights
result = agent.analyze_series(
    data=df,
    value_column='sales',
    date_column='date',
    forecast_periods=30,
    check_seasonality=True,
    decompose=True
)
print(result['output'])

# Quick analysis
quick_result = agent.quick_analysis(
    data=df,
    value_column='sales',
    date_column='date'
)
print(f"Mean: {quick_result['metrics']['mean']}")
print(f"Stationary: {quick_result['stationarity']['is_stationary']}")

# Detect patterns
patterns = agent.detect_patterns(
    data=df,
    value_column='sales',
    date_column='date'
)
print(f"Seasonality: {patterns['seasonality_found']}")

# Create forecasts
forecasts = agent.create_forecast(
    data=df,
    value_column='sales',
    date_column='date',
    forecast_periods=30,
    methods=['naive', 'mean', 'drift']
)

# Decompose series
decomposition = agent.decompose_series(
    data=df,
    value_column='sales',
    date_column='date',
    model='additive'
)

# Generate report
report = agent.generate_report(
    data=df,
    value_column='sales',
    date_column='date'
)
print(report)
```

### Key Features

- ✅ Seasonality detection (daily, weekly, monthly, yearly)
- ✅ Trend decomposition (additive/multiplicative)
- ✅ Stationarity testing (ADF test)
- ✅ Multiple baseline forecast methods
- ✅ Interactive Plotly visualizations

---

## Installation & Setup

### Prerequisites

```bash
# Core dependencies (already in main package)
pip install langchain langchain-openai pandas numpy scikit-learn plotly

# Additional dependencies for custom extensions
pip install scipy statsmodels shap
```

### Import Custom Agents

```python
# Import all agents
from custom.agents import (
    DataQualityAgent,
    FeatureImportanceAgent,
    ModelComparisonAgent,
    OutlierDetectionAgent,
    TimeSeriesAgent
)

# Import specific tools
from custom.tools import (
    detect_data_anomalies,
    calculate_permutation_importance,
    compare_classification_metrics,
    detect_outliers_iqr,
    detect_seasonality
)
```

---

## Quick Start Examples

### Example 1: End-to-End ML Pipeline

```python
from langchain_openai import ChatOpenAI
from custom.agents import *
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Initialize LLM
llm = ChatOpenAI(model="gpt-4")

# Load data
df = pd.read_csv("data.csv")

# Step 1: Data Quality Check
dq_agent = DataQualityAgent(model=llm)
quality_report = dq_agent.generate_report(df, "ml_dataset")
print(quality_report)

# Step 2: Outlier Detection
outlier_agent = OutlierDetectionAgent(model=llm)
outliers = outlier_agent.quick_detect(
    data=df,
    columns=['feature1', 'feature2'],
    method='isolation_forest'
)
print(f"Outliers: {outliers['outlier_info']}")

# Step 3: Train Model
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Step 4: Feature Importance
fi_agent = FeatureImportanceAgent(model=llm)
importance = fi_agent.quick_importance(
    model=rf,
    feature_names=X.columns.tolist(),
    top_n=10
)
print(importance['report'])

# Step 5: Compare with Other Models
# ... train additional models ...
# comparison_agent = ModelComparisonAgent(model=llm)
# results = comparison_agent.quick_comparison(...)
```

### Example 2: Time Series Analysis Pipeline

```python
from langchain_openai import ChatOpenAI
from custom.agents import TimeSeriesAgent, DataQualityAgent
import pandas as pd

llm = ChatOpenAI(model="gpt-4")

# Load time series
df = pd.read_csv("sales_data.csv")

# Check data quality
dq_agent = DataQualityAgent(model=llm)
quality = dq_agent.quick_check(df)

# Analyze time series
ts_agent = TimeSeriesAgent(model=llm)
report = ts_agent.generate_report(
    data=df,
    value_column='sales',
    date_column='date'
)
print(report)

# Create forecasts
forecasts = ts_agent.create_forecast(
    data=df,
    value_column='sales',
    date_column='date',
    forecast_periods=30,
    methods=['naive', 'drift']
)
```

---

## Agent Architecture

All custom agents follow this pattern:

```python
class CustomAgent:
    def __init__(self, model, verbose=True):
        self.model = model  # LLM
        self.tools = self._create_tools()  # LangChain tools
        self.agent_executor = self._create_agent()  # Agent with reasoning

    # Full AI-powered analysis
    def invoke(self, ...):
        # Uses LLM reasoning to select and use tools
        return self.agent_executor.invoke(...)

    # Quick utility methods (no AI reasoning)
    def quick_method(self, ...):
        # Direct tool invocation
        return tool.invoke(...)
```

**Benefits:**
- **Flexibility:** Use full AI reasoning or quick utilities
- **Explainability:** AI explains its analysis steps
- **Composability:** Combine multiple agents in workflows
- **Extensibility:** Easy to add new tools to existing agents

---

## Best Practices

### 1. **Use Quick Methods for Known Tasks**
```python
# Good: Fast, predictable
result = agent.quick_check(df)

# When you need reasoning
result = agent.invoke(df, task="Find unusual patterns and explain them")
```

### 2. **Combine Agents in Pipelines**
```python
# Quality → Outliers → Modeling → Comparison
quality_agent.invoke(df, ...)
outlier_agent.detect_outliers(df, ...)
# ... train models ...
comparison_agent.compare_models(...)
```

### 3. **Cache Expensive Operations**
```python
# Calculate once, reuse
importance_data = fi_agent.quick_importance(model, features)
# Use for multiple visualizations
```

### 4. **Leverage Agent Reasoning for Complex Analysis**
```python
# Let AI figure out the best approach
agent.invoke(
    data=df,
    task="This is financial data. Find anomalies that might indicate fraud."
)
```

---

## Troubleshooting

### Import Errors

```python
# If you see: ModuleNotFoundError: No module named 'custom'

# Add project root to path
import sys
sys.path.insert(0, '/path/to/ai-data-science-team')

# Then import
from custom.agents import DataQualityAgent
```

### Missing Dependencies

```bash
# SHAP for feature importance
pip install shap

# Time series analysis
pip install statsmodels scipy

# Visualization
pip install plotly
```

### Agent Timeout

```python
# Increase max_iterations for complex tasks
agent = DataQualityAgent(model=llm)
agent.agent_executor.max_iterations = 15  # Default: 10
```

---

## Contributing

### Adding a New Tool

```python
# custom/tools/your_module.py
from langchain.tools import tool

@tool
def your_new_tool(data: dict, param: str) -> str:
    """
    Tool description for LLM.

    Args:
        data: Input data
        param: Parameter description

    Returns:
        Result description
    """
    # Implementation
    return result
```

### Adding to an Existing Agent

```python
# custom/agents/data_quality_agent.py

from custom.tools.your_module import your_new_tool

class DataQualityAgent:
    def _create_tools(self):
        return [
            # ... existing tools ...
            your_new_tool,
        ]
```

---

## License

These custom extensions follow the same MIT license as the parent project.

## Support

For issues or questions:
1. Check the main project documentation: `/CUSTOM_WORKFLOW.md`
2. Review examples in this guide
3. Check tool docstrings for detailed parameter info

---

**Version:** 1.0.0
**Last Updated:** 2025-11-18
**Maintained By:** Your Fork
