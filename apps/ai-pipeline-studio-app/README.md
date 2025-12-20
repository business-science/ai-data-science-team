# AI Pipeline Studio

AI Pipeline Studio is a Streamlit app that centers your workflow around a visual pipeline, while the AI Data Science Team handles data loading, cleaning, visualization, and modeling. Use it to create, inspect, and evolve datasets step-by-step with full lineage, reproducible scripts, and project saves.

## Highlights
- Pipeline-first workspace: visual editor, table, chart, EDA, code, model, predictions, MLflow
- Manual + AI steps: add nodes, edit code drafts, rerun, and compare versions
- Multi-dataset handling: switch active datasets, merge datasets, and inspect lineage
- Project saves: metadata-only or full-data, with rehydrate and relink workflows
- Storage controls: opt-in dataset cache with size limits

## Quickstart
### Requirements
- Python 3.10+
- Dependencies from the project (installable via `pip install -e .`)
- OpenAI API key (or Ollama if running local models)

### Install
```bash
pip install -e .
```

### Run
```bash
streamlit run apps/ai-pipeline-studio-app/app.py
```

## Using Pipeline Studio
1) Load data (upload CSV or sample dataset).
2) Open Pipeline Studio from the sidebar or the floating button.
3) Select a pipeline step and switch views (Table, Chart, EDA, Code).
4) Add manual nodes or ask the AI team to create steps.
5) Save a project when you want to pick up later.

## Project Modes (Storage Footprint)
Pipeline Studio supports two save modes:
- Metadata-only: save lineage and steps without dataset files (rehydrate from source).
- Full-data: save dataset snapshots (Parquet preferred, pickle fallback).

You can convert a full-data project to metadata-only at any time.

## Screenshots
![AI Pipeline Studio](../../img/apps/ai_pipeline_studio_app.jpg)

## Configuration Tips
- Choose a model in the sidebar (OpenAI or Ollama).
- Enable short-term memory if you want multi-turn context.
- Turn on verbose logs to debug agent failures (see `logs/`).
- For large datasets, keep dataset caching off unless needed.

## Troubleshooting
- No charts showing: enable verbose logs and check the visualization error panel.
- Missing data on load: relink missing sources in the project panel.
- Cache size: adjust limits under Pipeline behaviors.

## License
See repository root for licensing details.
