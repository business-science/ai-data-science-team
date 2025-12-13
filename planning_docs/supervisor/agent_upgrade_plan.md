# Supervisor DS Team — Agent Upgrade Plan (Gap Analysis)

## Objective
Deliver a reliable end-to-end data science workflow in the supervisor-led team:
ingest/load → wrangle/clean → EDA → visualization → model training (H2O) → evaluation → MLflow logging/inspection.

This plan focuses on gaps that cause incorrect outputs, missing workflow steps, or unreliable “end-to-end” execution.

## Current State (What Exists)
- Supervisor router + shared state: `ai_data_science_team/multiagents/supervisor_ds_team.py`
- Sub-agents integrated:
  - `Data_Loader_Tools_Agent` (file discovery/loading)
  - `Data_Wrangling_Agent`, `Data_Cleaning_Agent`
  - `EDA_Tools_Agent` (describe/missing/correlation/Sweetviz/D-Tale)
  - `Data_Visualization_Agent` (Plotly codegen + semantic chart validation)
  - `SQL_Database_Agent`
  - `Feature_Engineering_Agent`
  - `H2O_ML_Agent` (AutoML; optional MLflow logging)
  - `MLflow_Tools_Agent` (inspection/UI/registry operations)
- Streamlit UX: `apps/supervisor-ds-team-app/app.py` (chat + “Analysis Details”)

## Key Gaps (Symptoms → Root Cause)
1. **No explicit workflow plan**
   - Symptom: multi-step prompts can stop early, skip prerequisites, or do steps out of order.
   - Root: supervisor routing is intent/step based, but not a durable plan with step verification and explicit prerequisites.

2. **Problem setup is underspecified**
   - Symptom: modeling requests fail or produce low-quality results without clarity on target/metric/splits/leakage.
   - Root: “target discovery” and “model spec” are not first-class artifacts; H2O agent relies on prompt inference.

3. **Evaluation artifacts are missing/weak**
   - Symptom: users cannot trust or compare models; no consistent confusion matrix/ROC/metrics table/error slices.
   - Root: no dedicated evaluation agent; results are mostly “leaderboard” and ad-hoc summaries.

4. **MLflow is not end-to-end for the workflow**
   - Symptom: “log everything to MLflow” is inconsistent; charts/EDA/datasets are not logged reliably.
   - Root: MLflow tools primarily support inspection/UI/predict; logging tools for params/metrics/tables/figures are missing.

5. **EDA report rendering is incomplete in Streamlit**
   - Symptom: Sweetviz/D-Tale outputs aren’t visible/embedded; users can’t easily inspect reports.
   - Root: Streamlit UI only renders Plotly JSON and JSON blobs; doesn’t handle HTML report artifacts.

## Upgrade Priorities

### P0 — Must Have (Unblocks true end-to-end workflows)
1. **WorkflowPlannerAgent (new)**
   - Output: a structured plan object with ordered steps, prerequisites, and required inputs.
   - Example steps: `load`, `validate_schema`, `clean`, `eda_summary`, `viz_requests`, `feature_engineering`, `train`, `evaluate`, `log_mlflow`.
   - Supervisor executes steps sequentially and marks completion with deterministic checks (e.g., “data_cleaned exists and non-empty”).

2. **ModelEvaluationAgent (new)**
   - Input: trained model artifact + dataset + target + split strategy.
   - Output: standardized artifacts (metrics table, confusion matrix/ROC for classification, residuals for regression, error slices).
   - Ensure evaluation results are used in the final answer and optionally logged to MLflow.

3. **Expand MLflow tooling to support logging (upgrade)**
   - Add tools: `mlflow_log_params`, `mlflow_log_metrics`, `mlflow_log_table`, `mlflow_log_artifact`, `mlflow_set_tags`, `mlflow_log_figure`.
   - Goal: logging can be performed deterministically (tools), not via LLM free-form code.

4. **Supervisor “Workflow Mode” toggle (app + supervisor)**
   - Add a UI toggle: **Proactive workflow mode** (off by default).
   - When on: supervisor is allowed to propose and run the full workflow even if user request is underspecified; it asks for missing inputs (e.g., target column) as needed.

### P1 — High Value (Quality + reliability improvements)
1. **Dataset registry & selection UX (upgrade)**
   - Maintain a dataset registry in supervisor state: `datasets[{name}] = {data, schema, provenance}` and `active_dataset_id`.
   - Explicitly route “use dataset X” and prevent silent dataset switches.

2. **DataQualityAgent (new, optional but high ROI)**
   - Output: schema/type inference, missingness rules, leakage checks (IDs), cardinality checks, target viability checks.
   - Can gate modeling: “data is not ready for training because …”

3. **EDA report rendering in Streamlit (upgrade)**
   - Detect Sweetviz/D-Tale outputs and render with `st.components.v1.html(...)` or provide download links.
   - Add a dedicated tab in “Analysis Details” for reports.

4. **Better “done-ness” checks (upgrade)**
   - Replace LLM “looks done” with explicit checks per step (data exists, plot type matches request, model metrics present, MLflow run id present).

### P2 — Nice to Have (Polish + scale)
1. **Task queue / long-running job handling**
   - Support longer workflows with progress updates and cancellation.

2. **Experiment comparison & model registry workflows**
   - “Compare last 5 runs”, “promote best to staging”, “register model”, “serve via MLflow”.

3. **Reproducibility packs**
   - Auto-export notebook/script with executed code for each step + metadata.

## Proposed Integration Order (Milestones)
1. P0.1–P0.2: WorkflowPlannerAgent + ModelEvaluationAgent wired into supervisor graph.
2. P0.3: Add MLflow logging tools and have supervisor log (tables/figures/metrics) deterministically.
3. P0.4: Add “Workflow Mode” toggle in Streamlit and supervisor behavior gates.
4. P1.1–P1.3: Dataset registry + DataQuality + EDA report rendering.
5. P1.4–P2: Hardening and advanced MLflow registry workflows.

## Acceptance Criteria (Definition of Done)
- A single prompt like: “Load churn, clean, EDA, plot MonthlyCharges by Churn, train a churn model, evaluate it, and log everything to MLflow”
  - executes without message-order issues,
  - produces correct chart types,
  - returns evaluation artifacts (not just a leaderboard),
  - logs a run with metrics + params + at least one table + one figure + model artifact in MLflow.

