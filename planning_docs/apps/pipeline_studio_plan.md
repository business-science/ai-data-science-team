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
- Full DAG editor / drag-and-drop pipeline editing. (Planned as a later phase.)
- Remote model serving or production-grade model registry flows (beyond existing MLflow/H2O integration).

## Current Building Blocks (Already Present)
- Dataset registry + `active_dataset_id` and `provenance.transform.kind` in supervisor state.
- Pipeline snapshot + reproducible script generation (`build_pipeline_snapshot`, `build_reproducible_pipeline_script`).
- Analysis Details tabs with Data/Charts/EDA/Models/Predictions/MLflow panels.
- Prediction transforms recorded in provenance:
  - `transform.kind = "mlflow_predict"` (run_id/model_uri)
  - `transform.kind = "h2o_predict"` (model_id)

## Status (What’s Implemented In-Repo)
✅ A working MVP Pipeline Studio UI exists in `apps/supervisor-ds-team-app/app.py`:
- Studio navigation: modal dialog (`st.dialog`) launched from the chat UI (Pipeline Studio button in main area + sidebar)
- Auto-seed on upload/sample: selecting an uploaded CSV or the sample Telco churn dataset seeds a `raw` dataset node so Studio can be used without running chat first (best effort)
- Pipeline target selector: Model / Active / Latest
- Pipeline step selector (lineage nodes)
- Workspace toggles: Visual Editor / Table / Chart / EDA / Code / Model / Predictions / MLflow
- Visual Editor is the default view; clicking a node opens an Inspector (Preview / Code / Metadata) with a draft code editor and “Ask AI” → sends the draft to chat
- Draft code persistence (implemented): “Save draft” writes to `pipeline_store/pipeline_studio_code_drafts.json` (keyed by dataset fingerprint) so edits survive Streamlit session resets
- Draft execution (implemented): “Run draft” creates a new dataset node (active) for `python_function`, `python_merge`, and `sql_query` (read-only guardrails)
- Manual node creation (beta): “New node” in the Visual Editor supports Python/SQL/Merge nodes (validation + templates + file upload) and creates `python_function`, `sql_query`, or `python_merge` nodes from selected parent dataset(s)
- Insert between nodes (beta): manual node creation can re-link a downstream node to the new node (inserts between parent → child and marks downstream stale)
- Project save/load (implemented, best effort): save datasets + Studio state to `pipeline_store/pipeline_projects/` (pickle; local-only)
- Downstream invalidation + “Run downstream” (implemented): rerunning a node marks downstream nodes stale, provides a replay button (best effort; skips unsupported transforms), and captures a `{old→new}` mapping with quick actions to hide stale nodes / the old branch
- Undo/redo (implemented, best effort): “Undo run / Redo run” supports undo/redo for single-node runs and grouped downstream runs (hide/delete uses Restore/Unhide)
- “Auto-follow latest step” behavior after each run
- Code pane renders provenance-backed snippets for `python_function`, `sql_query`, `python_merge`, and `*_predict` steps (best effort)
- Reproducibility: download pipeline spec (JSON) + pipeline script; copy/download buttons for per-node code snippets
- Table pane includes a schema summary (columns + sampled missingness)
- Compare mode: pick two nodes to view schema diff + dtype changes + missingness delta + chart/code compare + key-based row diff + side-by-side preview tables
- Deterministic artifact linking (lightweight): maintains a per-dataset artifact index in `st.session_state`, with a file-backed fingerprint index as a cross-session fallback (`pipeline_store/pipeline_studio_artifact_store.json`)

🧪 Phase 5 started (beta):
- Visual workflow editor/canvas (“Visual Editor” workspace view) using `streamlit-flow-component` (`streamlit_flow`)
  - Drag nodes to arrange layout (persisted best effort to `pipeline_store/pipeline_studio_flow_layout.json`)
  - Right-click node menu (soft-delete) + Inspector hide/unhide (semantic across Studio; persisted to `pipeline_store/pipeline_registry.json`)
  - Node inspector moved below the canvas; `Preview | Code | Metadata` spans full width
  - Edit actions (best effort): rename node (label/stage), branch actions (hide/unhide/soft-delete/restore + hard delete), run draft-only or run draft + downstream replay with “Replace mode” (auto-hide old branch)
  - Node badges: `ACTIVE` / `TARGET` / `STALE` / `HIDDEN` / `DELETED` + “Stale only” canvas filter
  - Downstream replacements: show `{old→new}` mapping + buttons to hide stale nodes / hide the old branch
  - Canvas actions: Reset layout, Show all nodes

✅ Recently implemented:
- Rich node inspector actions (edit code + rerun / delete subgraph) + semantic graph model
- Templates palette (quick add) with prebuilt Python/SQL/Merge snippets that open the manual node editor
- Project management + footprint controls:
  - Metadata-only project saves with preview samples + rehydrate from source
  - “Save project with data” (Parquet preferred; pickle fallback)
  - Project dashboard (search/sort, tags/notes/archive, rename/duplicate, bulk delete, mode badges)
  - Relink missing sources UI + rehydrate status summary
  - Last-load status includes missing data files for full-data projects
  - Dataset cache controls (persist/restore off by default, cache limits, parquet/pickle)
  - Factory reset + start-new-project flow
- Workspace tab badges indicate available artifacts per node (charts/EDA/model/predictions/MLflow)

## Proposed UX

### A) Where Pipeline Studio “lives”
**v1 (implemented):** Pipeline Studio opens as a modal (`st.dialog`) so it stays accessible without cluttering the chat layout (works well on desktop + mobile).

**v2 (beta):** Dockable “drawer” mode (inline expander) for wide screens, plus modal on mobile.

**v3 (future):** Promote Pipeline Studio into a top-level workspace with a visual pipeline editor (graph/canvas) and a right-side inspector.

### B) Declutter: Modal vs Tabs
Implemented Option 2 (modal). Other options remain useful depending on UX needs:

**Option 1 — Collapsible expander (quick win)**
- Keep Studio in the chat page but move it into `st.expander("Pipeline Studio", expanded=False)`.
- Lowest-risk; doesn’t change navigation; still keeps “one page” mental model.

**Option 2 — Modal inspector (`st.dialog`)**
- ✅ Implemented: “Pipeline Studio” button opens Studio in a modal (works well on desktop + mobile).
- Implementation note: avoid `st.rerun()` for most in-modal interactions (rerun closes the dialog); use `on_click` callbacks to mutate `st.session_state`.

**Option 3 — Top-level tabs (`st.tabs`)**
- Tabs: `Chat | Pipeline Studio` (optional: `Settings`).
- Useful fallback, but the current implementation favors a modal so Studio doesn’t fight the chat layout.

**Option 4 — Multi-page app (Streamlit Pages)**
- Separate `Chat` page and `Pipeline Studio` page.
- Best “long-term” maintainability; introduces page-level state/navigation considerations.

**Recommendation**
- Keep Option 2 (modal) as the default navigation; consider Option 1 (expander) for quick summaries and Option 3 (tabs) if/when Studio becomes the primary workflow surface.

**Left rail (Navigator)**
- Pipeline target selector: Model / Active / Latest (existing behavior)
- Pipeline “nodes list” (lineage) with:
  - label, stage, shape, created_by, created_at
  - transform type badge (load/sql/python_function/python_merge/mlflow_predict/h2o_predict)
- Node selection changes the workspace context.

**Main workspace (Toggle)**
**v1 (implemented):** `Visual Editor | Table | Chart | EDA | Code | Model | Predictions | MLflow`
- Visual Editor: draggable canvas view of the current pipeline, with an Inspector for Preview/Code/Metadata on node click
- Table: dataframe preview (with row count and column summary)
- Chart: if a plotly artifact exists for this node/turn, render it; otherwise show a helpful empty state
- EDA: link or embed Sweetviz / D-Tale if present for that node/turn
- Code:
  - Node transform code snippet (function code, SQL query, merge code)
  - Download snippet button (per node)
  - Reproducible pipeline script + spec download (view + download)
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

### D) Visual Pipeline Editor (phase 5)
Eventually replace the “nodes list” with an interactive, draggable graph editor:
- Drag nodes, pan/zoom, select nodes/edges
- Node inspector (edit metadata/code, view artifacts, run/fork, delete/hide)
- Optional edge edits (rewire downstream to a new node)

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

**Note (current implementation):** Studio uses an explicit `dataset_id → artifacts` index in `st.session_state` first, and falls back to scanning recent turn details only when the index is missing an artifact.

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

## Technical Design Notes (Phase 2)

### A) Artifact index (dataset_id → artifacts)
**Problem:** artifacts are currently attached “per turn”, but Studio needs “per dataset node” rendering.

**Proposed minimal structure (UI-owned, JSON/pointers only):**
```json
{
  "<dataset_id>": {
    "plotly_graph": {"json": {...}, "created_ts": 0, "turn_idx": 0},
    "eda_reports": {"reports": [...], "created_ts": 0, "turn_idx": 0},
    "model_info": {"info": {...}, "created_ts": 0, "turn_idx": 0},
    "eval_artifacts": {"artifacts": {...}, "created_ts": 0, "turn_idx": 0},
    "eval_plotly_graph": {"json": {...}, "created_ts": 0, "turn_idx": 0},
    "mlflow_artifacts": {"artifacts": {...}, "created_ts": 0, "turn_idx": 0}
  }
}
```

**Write path (when to update):**
- After each supervisor run completes, compute the “artifact context dataset id” (typically the run’s `active_dataset_id`) and, if the turn produced `plotly_graph` / `eda_reports` / `model_info` / `eval_artifacts` / `eval_plotly_graph` / `mlflow_artifacts`, update the index for that dataset id.
- For prediction steps: also index artifacts under the predictions dataset id returned by `_register_dataset` so the Predictions pane can render deterministically.

**Read path (how Studio uses it):**
- Workspace panes pull artifacts from the index first, and fall back to “scan history” only if missing.

**Ownership decision:**
- Prefer keeping this index in `st.session_state` (UI concern) unless another UI surface needs it; if it must be shared, store only lightweight pointers in supervisor `team_state`.
- Optional persistence (implemented): write a lightweight JSON index to disk keyed by dataset `fingerprint` (used as a best-effort fallback across sessions) at `pipeline_store/pipeline_studio_artifact_store.json`.

**Local output folders**
- Pipeline files (spec + repro script): default to `pipeline_reports/pipelines/` (user-configurable in the sidebar).
- EDA reports (Sweetviz HTML): default to a unique subfolder under `pipeline_reports/`.

### B) Schema summaries (fast, cached)
- Use existing dataset registry metadata when possible (`shape`, `schema`, `schema_hash`, `fingerprint`).
- Compute extra per-node stats only on demand (and cache): missingness counts, basic numeric summary.
- Guardrails: cap columns (e.g., first 200) and use small row samples for expensive stats.

### C) Compare mode (two-node diff)
**Inputs:** two dataset ids (A, B) from the same pipeline snapshot.

**Outputs:**
- UI shows changes in the selected step (A) relative to the compare step (B):
  - `removed_cols = B − A`, `added_cols = A − B`
- dtype changes by comparing `schema` entries on intersecting columns
- optional missingness delta for intersecting columns (sample-based)
- chart compare (side-by-side Plotly charts when available)
- code compare (side-by-side snippets + unified diff)
- row diff (key-based): keys only in A/B + per-key value mismatches for selected columns
- side-by-side `head(n)` previews (small `n`, user-controlled)

**UX (ideal):** a “Compare” toggle that switches the right pane into a compare view (2-column layout).

**UX (current implementation note):** compare mode fully replaces the workspace (via a clean `if/return`) to avoid `st.stop()` and keep the rest of the app rendering normally.

## Technical Design Notes (Visual Pipeline Editor)

### A) Candidate UI components
Based on the desired UX (drag, pan/zoom, right-click edit/delete, returns state to Python), a React Flow wrapper is a strong fit:
- Canvas: `streamlit-flow-component` (React Flow wrapper) for nodes/edges editing and layout persistence.
- Code editing: `streamlit-code-editor` or `streamlit-ace` inside a modal (`st.dialog`) or a right-side inspector panel.
  - Note: built-in context menus typically cover edit/delete; custom actions like “Run node” may need to live in the Inspector (click node → Inspector shows Run/Delete/Edit code).

### B) Graph representation (separate UI state from semantics)
Keep **two** distinct pieces of state:

1) **Flow UI state** (layout + selection; owned by the canvas component)
- Node positions, viewport, edge geometry
- Stored in `st.session_state.flow_state`

2) **Pipeline registry** (the “truth”; owned by the app)
- Nodes/edges with stable ids, provenance, parameters, and artifact pointers
- Stored in `st.session_state.pipeline_registry` (and later persisted to JSON/SQLite)

### C) Node types (progressive complexity)
**v1 graph (fastest):** dataset-only nodes
- Nodes are dataset ids; edges are parent→child (from dataset registry `parent_id`/`parent_ids`)
- Node badge shows transform kind + artifact availability

**v2 graph (clearer semantics):** dataset + transform nodes
- Dataset nodes represent data frames
- Transform nodes represent operations (`sql_query`, `python_function`, `python_merge`, `*_predict`)
- Edges: dataset_in → transform → dataset_out; artifacts hang off dataset_out

### D) Inspector actions (delete / edit / rerun)
Suggested behavior (safe + versionable):
- **Delete**: soft-delete/hide a node in the registry (do not destroy data by default); optionally “delete subgraph” with confirmation.
- **Edit code**: edits create a new “draft” version (fork) instead of mutating history. Store `code_sha256` and mark downstream as “stale”.
- **Run node**:
  - “Run node” executes this transform using its parent dataset(s) and produces a new dataset id.
  - “Run downstream” executes the subgraph from this node to selected sinks.

### E) Execution + caching/invalidation
- Cache key: `(transform_code_hash, input_fingerprint(s), params_hash)`
- If the user edits code or changes input wiring, invalidate downstream nodes by marking them “stale” (but keep previous outputs available).

### F) Persistence
Persist two files (or SQLite tables):
- `pipeline_store/pipeline_registry.json` (semantic graph + provenance + artifact pointers; no DataFrames)
- `pipeline_store/pipeline_studio_flow_layout.json` (node positions + hidden nodes; viewport not yet persisted)

## Implementation Plan (Phased)

### Phase 0 — Design alignment (1–2 hours)
- Confirm top-level UX: add a dedicated “Pipeline Studio” tab vs reworking Analysis Details.
- Decide whether Studio replaces existing bottom Analysis Details or complements it.
- Decide “minimum viable” toggles for v1 (Table + Chart + Code + Predictions recommended).

### Phase 1 — MVP Pipeline Studio ✅ (implemented)
1) UI surface (section) with:
   - Pipeline selector (Model/Active/Latest)
   - Node list selector (lineage)
   - Workspace toggles: Table / Chart / EDA / Code / Model / Predictions / MLflow
2) Auto-follow latest node after run completion
3) Map node selection → dataset id → preview dataframe
4) Render empty states when artifacts don’t exist for a node

### Phase 2 — Artifact linking + better previews ✅ (implemented)
1) Deterministic node artifacts mapping (dataset id → artifacts index with history fallback)
2) Schema summary panel (columns + sampled missingness)
3) Compare mode (two-node compare): schema diff + side-by-side preview

### Phase 3 — Reproducibility center ✅ (implemented)
1) Add “Repro script” section with:
   - ✅ download spec JSON + repro python script (Pipeline Studio)
   - ✅ include `*_predict` steps in script generation (MLflow/H2O)
2) Add “Copy snippet” buttons for node transform code (Streamlit UI convenience)
   - ✅ copy + download snippet buttons (fallback: code block copy UI)

### Phase 4 — Declutter UI ✅ (implemented)
1) Move Studio out of the bottom of the chat page:
   - implemented: modal (`st.dialog`) launched from chat
2) Preserve Studio selection state across reruns (target, node id, view, compare selection)
3) Keep the artifact index stable across navigation changes

### Phase 5 — Visual Workflow Editor 🧪 (in progress)
1) Canvas view (implemented, beta)
   - Workspace view: `Visual Editor` (Studio)
   - Dependency: `streamlit-flow-component`
   - State:
     - Node positions: `st.session_state["pipeline_studio_flow_positions"]`
     - Hidden nodes: `st.session_state["pipeline_studio_flow_hidden_ids"]`
     - Persisted layout store: `pipeline_store/pipeline_studio_flow_layout.json` (best effort, keyed by pipeline hash)
2) Semantic graph model (implemented, v0)
   - Persist a lightweight `pipeline_registry` graph model from the dataset registry + artifact index at `pipeline_store/pipeline_registry.json`
   - Exposes a “Download pipeline registry (JSON)” button in Studio (semantic DAG metadata; no DataFrames)
3) Inspector actions (next)
   - Show provenance + artifacts
   - Edit code (fork) + run
   - Delete/hide nodes (semantic, not just UI)
4) Execution (future)
   - Implement `run_from(node_id)` and downstream invalidation
   - Persist outputs/artifacts by `(run_id, node_id)`

### Phase 6 — Project management + footprint controls ✅ (implemented)
1) Metadata-only project saves (default) with preview samples
2) “Save project with data” using Parquet when available (pickle fallback)
3) Project dashboard (search/sort, tags/notes/archive, rename/duplicate, bulk delete)
4) Relink missing sources + rehydrate summary
5) Dataset cache controls (persist/restore defaults off, cache size limits, format)
6) Factory reset + start-new-project actions

## Acceptance Criteria (MVP)
✅ Users can open Pipeline Studio and:
- Select pipeline target (Model/Active/Latest)
- Select a node from lineage
- Toggle between Table and Code views for the node
- If a chart exists for the corresponding dataset, show it in Chart view (best effort)

✅ After a run finishes, Pipeline Studio auto-selects the latest node.

🔄 No new serialization errors introduced:
- Keep heavy objects (DataFrames, figures) out of any persisted or LLM-facing state; store lightweight JSON/pointers and reconstruct for display.

## Risks / Open Questions
- **Artifact ↔ dataset mapping**: current artifacts are “per turn”; we’ll need deterministic association rules (dataset id matching is safest).
- **Streamlit reruns**: ensure selection state doesn’t reset unexpectedly; use stable widget keys and avoid session_state mutation after instantiation.
- **Performance**: avoid rendering full tables; keep previews small and cache expensive computations (schema summaries).
- **True “live”**: people may mean streaming incremental intermediate results. MVP focuses on “live toggling of latest outputs” rather than streaming intermediate DataFrame updates.
- **Workflow editor complexity**: interactive canvas introduces heavy state management and requires a custom Streamlit component.
- **Code execution safety**: “edit code + rerun” requires clear guardrails (local-only, confirmations, and a limited execution model).

## Release Checklist (Remaining to Ship)
- Smoke-test metadata-only save → reload → rehydrate flow (including missing-source relink).
- Smoke-test full-data save/load (missing data files should surface clearly).
- Smoke-test cache pruning + Parquet fallback + factory reset/start-new-project.
- Add short user-facing docs or README snippet describing:
  - Metadata-only vs full-data saves
  - Rehydrate/relink workflow
  - Cache controls + storage footprint expectations
- (Optional) Add lightweight tests or a manual QA script for core workflows.

## Next Design Decisions (to unblock Phase 4/5)
- **Navigation**: tabs vs multi-page vs modal (decide what becomes the “default” workspace).
- **Graph model**: dataset-only nodes vs dataset+transform nodes.
- **Edit semantics**: mutate-in-place vs fork new nodes (recommend fork).
- **Persistence**: JSON files vs SQLite tables; where artifact files live.

## Notes
- The existing Pipeline snapshot feature is a strong foundation and likely what viewers are intuiting as the “toggle between steps” concept.
- “Pipeline Studio” formalizes that mental model and reduces the need to hunt through prior turns to find the right artifact.
