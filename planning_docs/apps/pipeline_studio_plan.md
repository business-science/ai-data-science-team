# Pipeline Studio Plan (Pipeline-Centered UX)

## Context / Motivation
A user request came in for “a tab with live figures and updated table” that’s easy to toggle between while working through an analysis. The current app already captures dataset lineage + a reproducible script (the Pipeline snapshot), but the UI is mostly “after the fact” (Analysis Details per turn) rather than a unified workspace centered on the evolving pipeline.

This plan proposes a **Pipeline Studio** experience that:
- Treats the pipeline as the primary navigation structure (what happened, in what order, and how to reproduce).
- Provides a “Workspace” that toggles between **Table ↔ Chart ↔ EDA ↔ Code ↔ Model/Predictions** for the currently selected pipeline node.
- Enables quick **compare/diff** between pipeline nodes (e.g., raw vs feature vs predictions) and/or between turns.

## Goals
1. **Pipeline-first navigation**: pipeline becomes the main “state timeline” a user interacts with.
2. **Toggle between artifacts**: view the current dataset as a table and/or related figures/EDA in one place.
3. **Make outputs feel “live”**: as new steps finish, the workspace automatically reflects the latest pipeline node(s).
4. **Reproducibility**: every node should show inputs, parameters, and runnable code snippets when available.
5. **Predictions as first-class nodes**: scoring steps (MLflow/H2O) should appear in the pipeline and be inspectable the same way as data transforms.

## Non-Goals (for the first iteration)
- Real-time streaming of intermediate DataFrame rows while a transform is running (Streamlit + sandbox/LLM calls make this non-trivial).
- Full DAG editor / drag-and-drop pipeline editing.
- Remote model serving or production-grade model registry flows (beyond existing MLflow/H2O integration).

## Current Building Blocks (Already Present)
- Dataset registry + `active_dataset_id` and `provenance.transform.kind` in supervisor state.
- Pipeline snapshot + reproducible script generation (`build_pipeline_snapshot`, `build_reproducible_pipeline_script`).
- Analysis Details tabs with Data/Charts/EDA/Models/Predictions/MLflow panels.
- Prediction transforms recorded in provenance:
  - `transform.kind = "mlflow_predict"` (run_id/model_uri)
  - `transform.kind = "h2o_predict"` (model_id)

## Proposed UX

### A) New top-level tab: “Pipeline Studio”
Add a dedicated tab/pane that replaces the “find the right turn” workflow with a pipeline-centered flow.

**Left rail (Navigator)**
- Pipeline target selector: Model / Active / Latest (existing behavior)
- Pipeline “nodes list” (lineage) with:
  - label, stage, shape, created_by, created_at
  - transform type badge (load/sql/python_function/python_merge/mlflow_predict/h2o_predict)
- Node selection changes the workspace context.

**Main workspace (Toggle)**
- Segmented control / tabs: `Table | Chart | EDA | Code | Model | Predictions | MLflow`
- Table: dataframe preview (with row count and column summary)
- Chart: if a plotly artifact exists for this node/turn, render it; otherwise show a helpful empty state
- EDA: link or embed Sweetviz / D-Tale if present for that node/turn
- Code:
  - Node transform code snippet (function code, SQL query, merge code)
  - Reproducible pipeline script section (download + view)
- Model:
  - training summary + leaderboard (if available)
  - mlflow run id / model uri (if available)
- Predictions:
  - prediction preview table
  - schema alignment warnings (categorical mismatch, missing columns)

### B) Compare Mode (phase 2)
Enable selecting two nodes:
- Show side-by-side previews (Table + key stats)
- Show schema diff:
  - columns added/removed
  - dtype changes (best-effort)
  - missingness delta
- Optionally: show row-level diff for a set of key columns (limited; expensive)

### C) “Auto-follow latest” (phase 1)
When a run completes, auto-select the newest pipeline node (or newest node within the selected target pipeline).

## Data Model / Plumbing Changes

### 1) Pipeline snapshot should optionally include “node artifacts”
We currently show charts/EDA/models per turn, not per pipeline node. For Pipeline Studio, we need a best-effort mapping between a lineage node and the artifacts produced “around” that step.

Proposed approach:
- Add `node_artifacts` in the pipeline snapshot (display-only, lightweight):
  - `plotly_graph` pointer (if chart created in the same turn and the dataset id matches)
  - `eda_reports` pointers (if EDA created and the dataset id matches)
  - `model_info` pointer (if training created and the model dataset id matches)
  - `mlflow_model_uri` pointer (if training created and run id exists)
  - `predictions` pointer (if scoring created and predictions dataset id matches)

This will likely require:
- Tracking a per-dataset “last artifacts” index in supervisor state (or in Streamlit session_state) so we can associate artifacts to dataset ids deterministically.
- Ensuring all artifacts stored remain msgpack/json serializable (no DataFrames).

### 2) Consistent provenance for all steps
Ensure every node in lineage can provide:
- `source_type` + `source` for roots
- `transform.kind` for transforms
- transform metadata fields depending on kind:
  - python_function: function_name/path/code_sha256
  - sql_query: sql_sha256 / query text
  - python_merge: merge_code/code_sha256 and parent ids
  - mlflow_predict: run_id/model_uri
  - h2o_predict: model_id

## Implementation Plan (Phased)

### Phase 0 — Design alignment (1–2 hours)
- Confirm top-level UX: add a dedicated “Pipeline Studio” tab vs reworking Analysis Details.
- Decide whether Studio replaces existing bottom Analysis Details or complements it.
- Decide “minimum viable” toggles for v1 (Table + Chart + Code + Predictions recommended).

### Phase 1 — MVP Pipeline Studio (1–2 days)
1) UI: Add “Pipeline Studio” to the app layout
   - Pipeline selector (Model/Active/Latest)
   - Node list selector (lineage)
   - Workspace toggles: Table / Chart / Code / Predictions
2) Auto-follow latest node after run completion
3) Map node selection -> dataset id -> preview dataframe
4) Render empty states when artifacts don’t exist for a node

### Phase 2 — Artifact linking + better previews (1–2 days)
1) Implement “best-effort node artifacts” mapping via dataset ids
2) Add schema summary panel:
   - n_rows, n_cols
   - column names (collapsible)
   - missingness count (fast)
3) Add “Compare mode” (two-node compare):
   - schema diff + side-by-side head()

### Phase 3 — Reproducibility center (1–2 days)
1) Add “Repro script” section with:
   - download spec JSON + repro python script
   - include ML/predict steps in script (already partially supported; keep improving)
2) Add “Copy snippet” buttons for node transform code (Streamlit UI convenience)

## Acceptance Criteria (MVP)
- Users can open Pipeline Studio and:
  - Select pipeline target (Model/Active/Latest)
  - Select a node from lineage
  - Toggle between Table and Code views for the node
  - If a chart exists for the corresponding dataset, show it in Chart view
- After a run finishes, Pipeline Studio auto-selects the latest node.
- No new serialization errors introduced (no DataFrames in state/artifacts payloads).

## Risks / Open Questions
- **Artifact ↔ dataset mapping**: current artifacts are “per turn”; we’ll need deterministic association rules (dataset id matching is safest).
- **Streamlit reruns**: ensure selection state doesn’t reset unexpectedly; use stable widget keys and avoid session_state mutation after instantiation.
- **Performance**: avoid rendering full tables; keep previews small and cache expensive computations (schema summaries).
- **True “live”**: people may mean streaming incremental intermediate results. MVP focuses on “live toggling of latest outputs” rather than streaming intermediate DataFrame updates.

## Notes
- The existing Pipeline snapshot feature is a strong foundation and likely what viewers are intuiting as the “toggle between steps” concept.
- “Pipeline Studio” formalizes that mental model and reduces the need to hunt through prior turns to find the right artifact.
