# AI Pipeline Studio App

Streamlit chat app to exercise a data pipeline workflow with AI assistance.

## Requirements
- Python 3.10+
- Dependencies from the project (installable via `pip install -e .`)
- OpenAI API key available as `OPENAI_API_KEY`

## Install
```bash
pip install -e .
```

## Run
```bash
streamlit run apps/ai-pipeline-studio-app/app.py
```

## Notes
- Enter your OpenAI API key in the sidebar, choose the model, and set recursion limit.
- You can upload a CSV or load the sample Telco churn data (`data/churn_data.csv`) for quick tests; a preview is shown before invoking the team.
- Short-term memory (LangGraph MemorySaver) can be toggled in the UI.
- Artifacts (dataframes, charts, etc.) are listed under the “Artifacts” section after each run.
