# 🚀 Fork Features - Enhanced AI Data Science Team

**This is an enhanced fork of [business-science/ai-data-science-team](https://github.com/business-science/ai-data-science-team)** with additional custom agents and tools.

## 🆕 What's New in This Fork?

This fork extends the original package with **5 additional AI agents** and **29 specialized tools** that complement the core functionality.

### Original Package Features ✅
- Data Wrangling Agent
- Data Visualization Agent
- Data Cleaning Agent
- Feature Engineering Agent
- SQL Database Agent
- H2O ML Agent
- MLflow Tools Agent
- EDA Tools Agent
- Pandas Data Analyst (Multi-Agent)

### 🎉 NEW: Custom Extensions (This Fork)

| Agent | Tools | Purpose |
|-------|-------|---------|
| **Data Quality Agent** | 5 | Schema validation, anomaly detection, quality scoring |
| **Feature Importance Agent** | 6 | Model interpretation, SHAP, importance comparison |
| **Model Comparison Agent** | 6 | Multi-model evaluation, ROC curves, rankings |
| **Outlier Detection Agent** | 6 | Multiple detection methods, treatment recommendations |
| **Time Series Agent** | 6 | Seasonality, decomposition, forecasting, stationarity |

**Total Custom Extensions:** 5 agents, 29 tools, 5,000+ lines of code

---

## 🎯 Why Use This Fork?

### Original Package → Great for:
- Data wrangling and cleaning
- Building ML models with H2O
- Basic EDA and visualization
- SQL data extraction

### This Fork → Adds:
- **Data Quality Validation** - Catch issues before modeling
- **Model Interpretation** - Understand feature importance with SHAP
- **Model Selection** - Compare multiple models scientifically
- **Outlier Management** - Advanced detection and treatment
- **Time Series Analysis** - Comprehensive temporal data tools

**Perfect for:** Teams that need end-to-end data science workflows with quality control and model validation.

---

## 📦 Installation

### Install from This Fork

```bash
# Clone this fork
git clone https://github.com/NicoLeeVaz/ai-data-science-team.git
cd ai-data-science-team

# Install with custom extensions
pip install -e .

# Install additional dependencies for custom features
pip install scipy statsmodels shap flask flask-cors streamlit
```

### Or Install Original Package

```bash
# Original package (without custom extensions)
pip install ai-data-science-team
```

---

## 🚀 Quick Start

### Using Original Package Agents

```python
from langchain_openai import ChatOpenAI
from ai_data_science_team.agents import DataWranglingAgent
import pandas as pd

llm = ChatOpenAI(model="gpt-4")
agent = DataWranglingAgent(model=llm)

# Use original agents as normal
result = agent.invoke(df, "Clean and prepare this data")
```

### Using Custom Fork Extensions

```python
from langchain_openai import ChatOpenAI
from custom.agents import (
    DataQualityAgent,
    FeatureImportanceAgent,
    ModelComparisonAgent,
    OutlierDetectionAgent,
    TimeSeriesAgent
)
import pandas as pd

llm = ChatOpenAI(model="gpt-4")

# 1. Check Data Quality
dq_agent = DataQualityAgent(model=llm)
quality_report = dq_agent.quick_check(df)
print(quality_report)

# 2. Detect Outliers
outlier_agent = OutlierDetectionAgent(model=llm)
outliers = outlier_agent.quick_detect(
    data=df,
    columns=['price', 'quantity'],
    method='iqr'
)
print(outliers['treatment_recommendations'])

# 3. Analyze Feature Importance (after training model)
fi_agent = FeatureImportanceAgent(model=llm)
importance = fi_agent.quick_importance(
    model=trained_model,
    feature_names=X.columns.tolist()
)
print(importance['report'])

# 4. Compare Models
comparison_agent = ModelComparisonAgent(model=llm)
results = comparison_agent.quick_comparison(
    predictions=predictions_dict,
    y_true=y_test,
    model_names=['RandomForest', 'XGBoost', 'LogisticRegression'],
    task_type='classification'
)
print(results['report'])

# 5. Time Series Analysis
ts_agent = TimeSeriesAgent(model=llm)
report = ts_agent.generate_report(
    data=df,
    value_column='sales',
    date_column='date'
)
print(report)
```

---

## 📖 Complete Documentation

### Custom Extensions Documentation
- **[Custom Extensions Guide](custom/docs/CUSTOM_EXTENSIONS_GUIDE.md)** - Detailed docs for all 5 custom agents
- **[Usage Examples](custom/examples/README.md)** - Python, Jupyter, Streamlit, CLI, REST API examples
- **[Quick Start](custom/QUICKSTART.md)** - Build your first custom agent

### Original Package Documentation
- **[Main README](README.md)** - Original package features and usage
- **[Examples](examples/)** - Original package examples

---

## 🛠️ Usage Methods

All custom agents work with multiple interfaces:

### 1. **Python Scripts**
```python
# analyze.py
from custom.agents import DataQualityAgent
import pandas as pd

df = pd.read_excel("data.xlsx")
agent = DataQualityAgent(model=llm)
print(agent.quick_check(df))
```

### 2. **Jupyter Notebooks**
```python
# Interactive analysis
from custom.agents import *
# ... explore data interactively
```

### 3. **Streamlit Web App** 🌐
```bash
streamlit run custom/examples/streamlit_quality_checker.py
# Opens browser with drag & drop interface!
```

### 4. **Command-Line Tool** 💻
```bash
python custom/examples/cli_quality_check.py data.csv --outliers
```

### 5. **REST API + JavaScript** 🌐
```bash
# Start API server
python custom/examples/api_server.py

# Use from JavaScript/React/any web app
# See: custom/examples/web_client.html
```

**Full examples:** See `custom/examples/README.md`

---

## 🔑 Key Features Comparison

| Feature | Original Package | This Fork |
|---------|-----------------|-----------|
| Data Wrangling | ✅ | ✅ |
| Data Cleaning | ✅ | ✅ |
| Feature Engineering | ✅ | ✅ |
| H2O AutoML | ✅ | ✅ |
| MLflow Integration | ✅ | ✅ |
| EDA Tools | ✅ | ✅ |
| **Data Quality Validation** | ❌ | ✅ **NEW** |
| **Schema Compliance** | ❌ | ✅ **NEW** |
| **Business Rule Validation** | ❌ | ✅ **NEW** |
| **Feature Importance (SHAP)** | ❌ | ✅ **NEW** |
| **Multi-Model Comparison** | ❌ | ✅ **NEW** |
| **ROC Curve Comparison** | ❌ | ✅ **NEW** |
| **Advanced Outlier Detection** | ❌ | ✅ **NEW** |
| **Isolation Forest** | ❌ | ✅ **NEW** |
| **LOF (Local Outlier Factor)** | ❌ | ✅ **NEW** |
| **Time Series Analysis** | ❌ | ✅ **NEW** |
| **Seasonality Detection** | ❌ | ✅ **NEW** |
| **Stationarity Testing** | ❌ | ✅ **NEW** |
| **Baseline Forecasting** | ❌ | ✅ **NEW** |
| **Streamlit Examples** | ❌ | ✅ **NEW** |
| **REST API Server** | ❌ | ✅ **NEW** |
| **CLI Tools** | ❌ | ✅ **NEW** |

---

## 💡 Common Workflows

### Workflow 1: Data Quality → Modeling → Comparison

```python
from langchain_openai import ChatOpenAI
from custom.agents import DataQualityAgent, OutlierDetectionAgent, ModelComparisonAgent
from ai_data_science_team.ml_agents import H2OMLAgent
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

llm = ChatOpenAI(model="gpt-4")

# Step 1: Load data
df = pd.read_csv("customer_data.csv")

# Step 2: Check quality (CUSTOM)
dq_agent = DataQualityAgent(model=llm)
quality = dq_agent.quick_check(df)
print("Quality Score:", quality)

# Step 3: Handle outliers (CUSTOM)
outlier_agent = OutlierDetectionAgent(model=llm)
outlier_indices = outlier_agent.get_consensus_outliers(
    data=df,
    columns=['age', 'income', 'credit_score'],
    methods=['zscore', 'iqr'],
    min_methods=2
)
df_clean = df.drop(outlier_indices)

# Step 4: Train models (ORIGINAL + Custom)
# Option A: Use original H2O ML Agent
h2o_agent = H2OMLAgent(model=llm)
h2o_agent.invoke_agent(
    data_raw=df_clean,
    user_instructions="Build classification model for 'churn'",
    target_variable="churn"
)

# Option B: Train your own models
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Step 5: Compare models (CUSTOM)
comparison_agent = ModelComparisonAgent(model=llm)
results = comparison_agent.quick_comparison(
    predictions={
        'H2O_Best': h2o_predictions,
        'RandomForest': rf.predict(X_test)
    },
    y_true=y_test,
    model_names=['H2O_Best', 'RandomForest'],
    task_type='classification'
)
print(results['report'])
```

### Workflow 2: Time Series Pipeline

```python
from langchain_openai import ChatOpenAI
from custom.agents import DataQualityAgent, TimeSeriesAgent
from ai_data_science_team.agents import DataWranglingAgent
import pandas as pd

llm = ChatOpenAI(model="gpt-4")

# Step 1: Wrangle data (ORIGINAL)
wrangling_agent = DataWranglingAgent(model=llm)
df_clean = wrangling_agent.invoke(df, "Clean and prepare time series data")

# Step 2: Quality check (CUSTOM)
dq_agent = DataQualityAgent(model=llm)
quality = dq_agent.quick_check(df_clean)

# Step 3: Time series analysis (CUSTOM)
ts_agent = TimeSeriesAgent(model=llm)

# Detect patterns
patterns = ts_agent.detect_patterns(
    data=df_clean,
    value_column='sales',
    date_column='date'
)
print(f"Seasonality: {patterns['seasonality_found']}")

# Create forecasts
forecasts = ts_agent.create_forecast(
    data=df_clean,
    value_column='sales',
    date_column='date',
    forecast_periods=30,
    methods=['naive', 'drift', 'seasonal_naive']
)

# Generate report
report = ts_agent.generate_report(df_clean, 'sales', 'date')
print(report)
```

---

## 🏗️ Architecture

### Custom Extensions Structure

```
ai-data-science-team/
├── ai_data_science_team/          # Original package
│   ├── agents/                    # Original agents
│   ├── ml_agents/                 # H2O, MLflow agents
│   ├── tools/                     # Original tools
│   └── ...
│
├── custom/                        # 🆕 CUSTOM EXTENSIONS (This Fork)
│   ├── agents/                    # 5 custom agents
│   │   ├── data_quality_agent.py
│   │   ├── feature_importance_agent.py
│   │   ├── model_comparison_agent.py
│   │   ├── outlier_detection_agent.py
│   │   └── time_series_agent.py
│   │
│   ├── tools/                     # 29 custom tools
│   │   ├── data_quality.py        # 5 tools
│   │   ├── feature_importance.py  # 6 tools
│   │   ├── model_comparison.py    # 6 tools
│   │   ├── outlier_detection.py   # 6 tools
│   │   └── time_series.py         # 6 tools
│   │
│   ├── examples/                  # Usage examples
│   │   ├── streamlit_quality_checker.py
│   │   ├── cli_quality_check.py
│   │   ├── api_server.py
│   │   ├── web_client.html
│   │   └── README.md
│   │
│   ├── docs/                      # Documentation
│   │   └── CUSTOM_EXTENSIONS_GUIDE.md
│   │
│   └── private/                   # Your private work (gitignored)
│
└── FORK_FEATURES.md               # This file
```

---

## 🔐 Safety & Isolation

This fork is configured with safety guardrails to prevent accidental merging back to upstream:

✅ **Upstream push disabled** - Cannot accidentally push to original repo
✅ **Custom code isolated** - All custom work in `custom/` directory
✅ **Private workspace** - `custom/private/` for sensitive code (gitignored)
✅ **Independent versioning** - Your fork, your releases

See: [CUSTOM_WORKFLOW.md](CUSTOM_WORKFLOW.md) for development guidelines.

---

## 🤝 Contributing

### To This Fork
1. Fork this repository (NicoLeeVaz/ai-data-science-team)
2. Create feature branch: `git checkout -b feature/my-feature`
3. Work in `custom/` directory for new features
4. Submit PR to this fork

### To Original Package
For improvements to the core package, submit PRs to:
- **Upstream:** [business-science/ai-data-science-team](https://github.com/business-science/ai-data-science-team)

---

## 📊 Examples Gallery

### Data Quality Validation
```python
from custom.agents import DataQualityAgent

agent = DataQualityAgent(model=llm)
report = agent.generate_report(df, "customer_data")

# Output:
# DATA QUALITY SCORECARD
# ================================================
# Overall Quality Score: 87.3/100
#
# Breakdown:
#   Completeness   : 95.2/100 ★★★★★
#   Uniqueness     : 89.5/100 ★★★★☆
#   Validity       : 82.1/100 ★★★★☆
#   Consistency    : 82.5/100 ★★★★☆
```

### Feature Importance Analysis
```python
from custom.agents import FeatureImportanceAgent

agent = FeatureImportanceAgent(model=llm)
importance = agent.quick_importance(model, feature_names)

# Output:
# TOP 20 MOST IMPORTANT FEATURES:
# ────────────────────────────────────────────────
# Rank   Feature                    Importance
# ────────────────────────────────────────────────
# 1      customer_lifetime_value    0.234567
# 2      purchase_frequency         0.198234
# 3      avg_order_value           0.156789
```

### Model Comparison
```python
from custom.agents import ModelComparisonAgent

agent = ModelComparisonAgent(model=llm)
results = agent.quick_comparison(predictions, y_test, models, 'classification')

# Output:
# BEST MODELS PER METRIC:
# ────────────────────────────────────────────────
#   accuracy       : RandomForest         (0.9234)
#   precision      : GradientBoosting     (0.9156)
#   recall         : RandomForest         (0.9089)
#   f1_score       : RandomForest         (0.9161)
```

---

## 📈 Roadmap

### Planned Features (This Fork)
- [ ] Automated model deployment agent
- [ ] A/B testing comparison tools
- [ ] Advanced feature selection algorithms
- [ ] Causal inference tools
- [ ] Drift detection for deployed models

### Staying Updated with Upstream
```bash
# Pull latest from original package
git fetch upstream main
git merge upstream/main

# Your custom/ directory won't conflict!
```

---

## 📄 License

Same as original: **MIT License**

This fork maintains the MIT license from the original project. You're free to use, modify, and distribute both the original package features and custom extensions.

---

## 🙏 Acknowledgments

**Original Package:** [business-science/ai-data-science-team](https://github.com/business-science/ai-data-science-team) by Matt Dancho

**This Fork:** Enhanced with custom agents for comprehensive data science workflows

---

## 📞 Support

### For Custom Extensions (This Fork)
- Issues: [NicoLeeVaz/ai-data-science-team/issues](https://github.com/NicoLeeVaz/ai-data-science-team/issues)
- Documentation: `custom/docs/CUSTOM_EXTENSIONS_GUIDE.md`
- Examples: `custom/examples/`

### For Original Package Features
- Issues: [business-science/ai-data-science-team/issues](https://github.com/business-science/ai-data-science-team/issues)
- Documentation: See main [README.md](README.md)

---

## ⭐ Star This Fork!

If you find these custom extensions useful, please star this repository!

[⭐ Star on GitHub](https://github.com/NicoLeeVaz/ai-data-science-team)

---

**Quick Links:**
- [Custom Extensions Guide](custom/docs/CUSTOM_EXTENSIONS_GUIDE.md)
- [Usage Examples](custom/examples/README.md)
- [Development Workflow](CUSTOM_WORKFLOW.md)
- [Original README](README.md)
