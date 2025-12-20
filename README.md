<div align="center">
  <a href="https://github.com/business-science/ai-data-science-team">
    <picture>
      <img src="./img/ai_data_science_logo.png" alt="AI Data Science Team" width="360">
    </picture>
  </a>
</div>
<div align="center">
  <em>AI Data Science Team + AI Pipeline Studio</em>
</div>
<div align="center">
  <a href="https://pypi.python.org/pypi/ai-data-science-team"><img src="https://img.shields.io/pypi/v/ai-data-science-team.svg?style=for-the-badge" alt="PyPI"></a>
  <a href="https://github.com/business-science/ai-data-science-team"><img src="https://img.shields.io/pypi/pyversions/ai-data-science-team.svg?style=for-the-badge" alt="versions"></a>
  <a href="https://github.com/business-science/ai-data-science-team/blob/main/LICENSE"><img src="https://img.shields.io/github/license/business-science/ai-data-science-team.svg?style=for-the-badge" alt="license"></a>
  <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/business-science/ai-data-science-team?style=for-the-badge">
</div>

# AI Data Science Team

AI Data Science Team is a Python library of specialized agents for common data science workflows, plus a flagship app: **AI Pipeline Studio**. The Studio turns your work into a visual, reproducible pipeline, while the AI team handles data loading, cleaning, visualization, and modeling.

**Status:** Beta. Breaking changes may occur until 0.1.0.

[**Please ⭐ us on GitHub (it takes 2 seconds and means a lot).**](https://github.com/business-science/ai-data-science-team)

## AI Pipeline Studio (Flagship App)

AI Pipeline Studio is the main example of the AI Data Science Team in action.

![AI Pipeline Studio](/img/apps/ai_pipeline_studio_app.jpg)

Highlights:
- Pipeline-first workspace: Visual Editor, Table, Chart, EDA, Code, Model, Predictions, MLflow
- Manual + AI steps with lineage and reproducible scripts
- Multi-dataset handling and merge workflows
- Project saves: metadata-only or full-data
- Storage footprint controls and rehydrate workflows

Run it:
```bash
streamlit run apps/ai-pipeline-studio-app/app.py
```

Full app docs: `apps/ai-pipeline-studio-app/README.md`

## Quickstart

### Requirements
- Python 3.10+
- OpenAI API key (or Ollama for local models)

### Install the app and library
Clone the repo and install in editable mode:
```bash
pip install -e .
```

### Run the AI Pipeline Studio app
```bash
streamlit run apps/ai-pipeline-studio-app/app.py
```

## Library Overview

The repository includes both the **AI Pipeline Studio** app and the underlying **AI Data Science Team** library. The library provides agent building blocks and multi-agent workflows for:
- Data loading and inspection
- Cleaning, wrangling, and feature engineering
- Visualization and EDA
- Modeling and evaluation (H2O + MLflow tools)
- SQL database interaction

### Agents (Snapshot)

Agent examples live in `examples/`. Notable agents:
- Data Loader Tools Agent
- Data Wrangling Agent
- Data Cleaning Agent
- Data Visualization Agent
- EDA Tools Agent
- Feature Engineering Agent
- SQL Database Agent
- H2O ML Agent
- MLflow Tools Agent
- Multi-agent workflows (e.g., Pandas Data Analyst, SQL Data Analyst)
- Supervisor Agent (oversees other agents)
- Custom tools for data science tasks

## Apps

See all apps in `apps/`. Notable apps:
- AI Pipeline Studio: `apps/ai-pipeline-studio-app/`
- EDA Explorer App: `apps/exploratory-copilot-app/`
- Pandas Data Analyst App: `apps/pandas-data-analyst-app/`

## Use OpenAI

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model_name="gpt-4.1-mini",
)
```

## Use Ollama (Local LLM)

```bash
ollama serve
ollama pull llama3.1:8b
```

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.1:8b",
)
```

## Next-Gen AI Agentic Workshop

Want to learn how to build AI agents and AI apps for real data science workflows? Join my next‑gen AI workshop:
https://learn.business-science.io/ai-register
