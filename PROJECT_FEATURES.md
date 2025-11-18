# 🚀 AI Data Science Team - Features

**A comprehensive AI-powered data science platform with specialized agents for end-to-end workflows.**

## 📦 What's Included?

This package includes **14 specialized AI agents** and **29+ custom tools** for comprehensive data science workflows.

### Core Data Agents
- **Data Wrangling Agent** - Merge, join, and prepare data
- **Data Visualization Agent** - Create interactive visualizations
- **Data Cleaning Agent** - Handle missing values, outliers, types
- **Feature Engineering Agent** - Build ML-ready features
- **SQL Database Agent** - Query and extract from databases
- **Data Loader Tools Agent** - Load CSV, Excel, Parquet, Pickle

### Machine Learning Agents
- **H2O ML Agent** - Build hundreds of ML models with AutoML
- **MLflow Tools Agent** - MLOps and model management (11+ tools)

### Data Science Agents
- **EDA Tools Agent** - Automated exploratory data analysis

### Multi-Agents
- **Pandas Data Analyst** - Combined wrangling + visualization
- **SQL Data Analyst** - SQL operations + visualization

### 🎉 Advanced Custom Agents

| Agent | Tools | Purpose |
|-------|-------|---------|
| **Data Quality Agent** | 5 | Schema validation, anomaly detection, quality scoring |
| **Feature Importance Agent** | 6 | Model interpretation, SHAP, importance comparison |
| **Model Comparison Agent** | 6 | Multi-model evaluation, ROC curves, rankings |
| **Outlier Detection Agent** | 6 | Multiple detection methods, treatment recommendations |
| **Time Series Agent** | 6 | Seasonality, decomposition, forecasting, stationarity |

**Total Agents:** 14
**Total Tools:** 58+

---

## 🎯 Key Capabilities

### Data Quality & Validation
- Schema compliance checking
- Business rule validation
- Automated quality scoring (0-100)
- Comprehensive quality reports
- Data anomaly detection

### Model Interpretation & Selection
- Feature importance (tree-based, SHAP, permutation)
- Cross-method importance comparison
- Multi-model performance comparison
- ROC curve generation
- Automated model ranking

### Outlier Management
- Z-score detection (parametric)
- IQR detection (non-parametric)
- Isolation Forest (multivariate)
- Local Outlier Factor (density-based)
- Treatment recommendations

### Time Series Analysis
- Seasonality detection (daily/weekly/monthly/yearly)
- Time series decomposition
- Stationarity testing (ADF)
- Baseline forecasting methods
- Comprehensive temporal metrics

### Machine Learning
- H2O AutoML integration
- MLflow model tracking
- Feature engineering
- Model deployment

### Interactive Tools
- Streamlit web applications
- REST API server
- Command-line interface
- JavaScript/React integration

---

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/NicoLeeVaz/ai-data-science-team.git
cd ai-data-science-team

# Install
pip install -e .
pip install scipy statsmodels shap streamlit flask flask-cors
```

See [INSTALLATION.md](INSTALLATION.md) for detailed instructions.

---

## 🚀 Quick Start

### Using Core Agents

```python
from langchain_openai import ChatOpenAI
from ai_data_science_team.agents import DataWranglingAgent
import pandas as pd

llm = ChatOpenAI(model="gpt-4")
agent = DataWranglingAgent(model=llm)

df = pd.read_csv("data.csv")
result = agent.invoke(df, "Clean and prepare this data")
```

### Using Custom Agents

```python
from langchain_openai import ChatOpenAI
from custom.agents import (
    DataQualityAgent,
    FeatureImportanceAgent,
    ModelComparisonAgent,
    OutlierDetectionAgent,
    TimeSeriesAgent
)

llm = ChatOpenAI(model="gpt-4")

# Check data quality
dq_agent = DataQualityAgent(model=llm)
quality = dq_agent.quick_check(df)

# Detect outliers
outlier_agent = OutlierDetectionAgent(model=llm)
outliers = outlier_agent.quick_detect(df, columns=['price', 'age'], method='iqr')

# Analyze feature importance
fi_agent = FeatureImportanceAgent(model=llm)
importance = fi_agent.quick_importance(model, feature_names)

# Compare models
comp_agent = ModelComparisonAgent(model=llm)
comparison = comp_agent.quick_comparison(predictions, y_test, model_names, 'classification')

# Time series analysis
ts_agent = TimeSeriesAgent(model=llm)
report = ts_agent.generate_report(df, 'sales', 'date')
```

---

## 🛠️ Usage Methods

### 1. Python Scripts
```python
# analyze.py
from custom.agents import DataQualityAgent
import pandas as pd

df = pd.read_excel("data.xlsx")
agent = DataQualityAgent(model=llm)
print(agent.quick_check(df))
```

### 2. Jupyter Notebooks
Interactive data exploration and analysis

### 3. Streamlit Web App
```bash
streamlit run custom/examples/streamlit_quality_checker.py
```
Drag & drop interface for non-technical users

### 4. Command-Line Tool
```bash
python custom/examples/cli_quality_check.py data.csv --outliers
```

### 5. REST API + JavaScript
```bash
python custom/examples/api_server.py
```
HTTP API for web applications

---

## 📊 Common Workflows

### Workflow 1: Data Quality → Modeling → Comparison

```python
# Step 1: Check quality
dq_agent = DataQualityAgent(model=llm)
quality = dq_agent.quick_check(df)

# Step 2: Handle outliers
outlier_agent = OutlierDetectionAgent(model=llm)
outlier_indices = outlier_agent.get_consensus_outliers(df, columns, methods=['zscore', 'iqr'])
df_clean = df.drop(outlier_indices)

# Step 3: Train models
h2o_agent = H2OMLAgent(model=llm)
h2o_agent.invoke_agent(df_clean, "Build classification model", target="churn")

# Step 4: Compare models
comparison_agent = ModelComparisonAgent(model=llm)
results = comparison_agent.quick_comparison(predictions, y_test, model_names, 'classification')
```

### Workflow 2: Time Series Pipeline

```python
# Step 1: Wrangle data
wrangling_agent = DataWranglingAgent(model=llm)
df_clean = wrangling_agent.invoke(df, "Clean time series data")

# Step 2: Analyze patterns
ts_agent = TimeSeriesAgent(model=llm)
patterns = ts_agent.detect_patterns(df_clean, 'sales', 'date')

# Step 3: Create forecasts
forecasts = ts_agent.create_forecast(df_clean, 'sales', 'date', 30, ['naive', 'drift'])
```

---

## 🏗️ Architecture

```
ai-data-science-team/
├── ai_data_science_team/          # Core package
│   ├── agents/                    # Core agents
│   ├── ml_agents/                 # ML agents
│   └── tools/                     # Core tools
│
├── custom/                        # Custom extensions
│   ├── agents/                    # 5 custom agents
│   ├── tools/                     # 29 custom tools
│   ├── examples/                  # Usage examples
│   └── docs/                      # Documentation
```

---

## 📚 Documentation

- **[Installation Guide](INSTALLATION.md)** - Setup instructions
- **[Custom Extensions Guide](custom/docs/CUSTOM_EXTENSIONS_GUIDE.md)** - Detailed agent documentation
- **[Usage Examples](custom/examples/README.md)** - Python, Jupyter, Streamlit, CLI, API
- **[Custom Workflow](CUSTOM_WORKFLOW.md)** - Development guidelines

---

## 🔑 Key Features Comparison

| Feature | Available |
|---------|-----------|
| Data Wrangling | ✅ |
| Data Cleaning | ✅ |
| Feature Engineering | ✅ |
| H2O AutoML | ✅ |
| MLflow Integration | ✅ |
| EDA Tools | ✅ |
| Data Quality Validation | ✅ |
| Schema Compliance | ✅ |
| Business Rule Validation | ✅ |
| Feature Importance (SHAP) | ✅ |
| Multi-Model Comparison | ✅ |
| ROC Curve Comparison | ✅ |
| Advanced Outlier Detection | ✅ |
| Isolation Forest | ✅ |
| LOF (Local Outlier Factor) | ✅ |
| Time Series Analysis | ✅ |
| Seasonality Detection | ✅ |
| Stationarity Testing | ✅ |
| Baseline Forecasting | ✅ |
| Streamlit Apps | ✅ |
| REST API Server | ✅ |
| CLI Tools | ✅ |

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 📞 Support

- Documentation: See docs/ directory
- Examples: See custom/examples/
- Issues: Use GitHub Issues

---

**Built for production data science workflows.**
