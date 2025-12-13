# Data Science Supervisor Team Plan

## Goal
Build a LangGraph supervisor-led data science team (message-first, tool-aware) that can route work across core sub-agents (data loading, wrangling/cleaning, EDA/visualization, SQL, feature engineering, ML training/serving) while remaining backward compatible with existing agent APIs.

## Team & Roles (initial)
- `Data_Loader_Tools_Agent` – directory/file discovery and loading (csv/parquet/etc).
- `Data_Wrangling_Agent` – pandas transformations; light cleaning.
- `Data_Cleaning_Agent` – robust cleaning/imputation; user constraints.
- `EDA_Tools_Agent` – describe, missingness, correlation, Sweetviz.
- `Data_Visualization_Agent` – plotly/matplotlib chart generation.
- `SQL_Database_Agent` – SQL generation/execution; returns data + SQL code.
- `Feature_Engineering_Agent` – feature creation for modeling.
- `H2O_ML_Agent` – AutoML training/eval; optional MLflow logging.
- `MLflow_Tools_Agent` – experiment/registry operations (list/search/runs, artifacts, stage transitions, UI status).

## Supervisor Design
- State: `messages: Sequence[BaseMessage]`, `next: str`, plus shared payload slots (`data_raw`, `data_sql`, `chart_json`, `artifacts`, `errors`).
- Routing rules:
  - Default entry: supervisor inspects last human message and chooses a worker.
  - Avoid same worker twice in a row unless explicitly requested.
  - Prefer table-first workflows unless user explicitly asks for charts/models.
  - If data missing, route to Data_Loader; if data needs shaping, to Data_Wrangling/Cleaning; if query needed, to SQL; if summary needed, to EDA/Visualization; if features/models requested, to Feature_Engineering/H2O_ML; if experiment ops requested, to MLflow_Tools.
- Output format: supervisor returns `messages` with appended AI decision trace; sub-agents return their `messages` and artifacts; supervisor aggregates a concise summary.

## Implementation Steps
1) Draft supervisor prompt & router function (JSON route schema; names must match sub-agent nodes). ✅ in `supervisor_ds_team.py`
2) Wire sub-agents as nodes (use their `invoke_messages` / `ainvoke_messages`). ✅
3) Define state schema with additive `messages` and optional slots (`data_raw`, `data_sql`, `plotly_graph`, `model_info`, `mlflow_artifacts`). ✅
4) Add guardrails: if a sub-agent returns empty data, reroute or respond with guidance; cap recursion. 🔄 (basic guards via supervisor routing; future: explicit empties)
5) Logging: minimal progress prints (`* SUPERVISOR`, chosen worker; sub-agent tool logging already exists). ✅
6) Demo: create `temp/30_supervisor_ds_team_demo.py` showing a table request, a chart request, and a quick model run. ✅ (demo created; modeling step optional)

## Backward Compatibility
- Keep sub-agents’ legacy entrypoints intact; supervisor uses message-first.
- Do not change artifact shapes beyond existing shims (single-tool unwrapping).
- Supervisor outputs should not break existing getters; add a helper to extract the last AI message if needed.

## Open Questions
- Do we include sandboxed code execution for modeling agents by default? (currently opt-in).
- Should we add a lightweight summarizer node to produce a final “answer” after worker responses?
- Memory: use optional `MemorySaver` checkpointer for short-term conversation continuity.

## Notes / Fixes (post-plan)
- Conversation state: `messages` now uses LangGraph's ID-aware message reducer (prevents duplicated history when nodes return full message lists).
- Message-first sub-agent calls: `invoke_messages(...)` for coding-style agents now forwards `user_instructions` (or infers it from the last user message), which fixes generic/incorrect outputs (especially charts/SQL).
- Data correctness: the supervisor tracks an `active_data_key` so downstream agents use the most recently “active” dataset (raw vs SQL vs wrangled/cleaned/features), avoiding stale plots.
