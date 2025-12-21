"""
Streamlit app for the AI Data Science.

Command:
    streamlit run apps/supervisor-ds-team-app/app.py
"""

from __future__ import annotations

import re
import uuid
import os
import json
import inspect
import shutil
from openai import OpenAI
import pandas as pd
import sqlalchemy as sql
import plotly.colors as pc
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI

try:
    from langchain_ollama import ChatOllama  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ChatOllama = None
from langgraph.checkpoint.memory import MemorySaver

from ai_data_science_team.agents.data_loader_tools_agent import DataLoaderToolsAgent
from ai_data_science_team.agents.data_wrangling_agent import DataWranglingAgent
from ai_data_science_team.agents.data_cleaning_agent import DataCleaningAgent
from ai_data_science_team.ds_agents.eda_tools_agent import EDAToolsAgent
from ai_data_science_team.agents.data_visualization_agent import DataVisualizationAgent
from ai_data_science_team.agents.sql_database_agent import SQLDatabaseAgent
from ai_data_science_team.agents.feature_engineering_agent import (
    FeatureEngineeringAgent,
)
from ai_data_science_team.agents.workflow_planner_agent import WorkflowPlannerAgent
from ai_data_science_team.ml_agents.h2o_ml_agent import H2OMLAgent
from ai_data_science_team.ml_agents.mlflow_tools_agent import MLflowToolsAgent
from ai_data_science_team.ml_agents.model_evaluation_agent import ModelEvaluationAgent
from ai_data_science_team.multiagents.supervisor_ds_team import make_supervisor_ds_team
from ai_data_science_team.utils.pipeline import build_pipeline_snapshot

APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TITLE = "AI Pipeline Studio"
LOGO_PATH = os.path.join(APP_ROOT, "img", "ai_pipeline_studio_logo.png")
page_icon = LOGO_PATH if os.path.exists(LOGO_PATH) else ":bar_chart:"
st.set_page_config(page_title=TITLE, page_icon=page_icon, layout="wide")
st.title(TITLE)
st.markdown('<div id="page-top"></div>', unsafe_allow_html=True)
st.markdown(
    "\n".join(
        [
            "<style>",
            "@media (min-width: 1100px) {",
            "  [data-testid=\"stDialog\"] [role=\"dialog\"] {",
            "    width: calc(100vw - 2rem) !important;",
            "    max-width: calc(100vw - 2rem) !important;",
            "  }",
            "}",
            "</style>",
        ]
    ),
    unsafe_allow_html=True,
)

UI_DETAIL_MARKER_PREFIX = "DETAILS_INDEX:"
DEFAULT_SQL_URL = "sqlite:///:memory:"
SQL_URL_INPUT_KEY = "sql_url_input"
SQL_URL_SYNC_FLAG = "_sync_sql_url_input"
ACTIVE_DATASET_OVERRIDE_SYNC_FLAG = "_sync_active_dataset_override"
ACTIVE_DATASET_OVERRIDE_PENDING_KEY = "active_dataset_id_override_pending"
PIPELINE_STUDIO_ARTIFACT_STORE_VERSION = 1
PIPELINE_STUDIO_ARTIFACT_STORE_PATH = os.path.join(
    APP_ROOT, "pipeline_store", "pipeline_studio_artifact_store.json"
)
PIPELINE_STUDIO_ARTIFACT_STORE_LEGACY_PATH = os.path.join(
    APP_ROOT, "temp", "pipeline_studio_artifact_store.json"
)
PIPELINE_STUDIO_ARTIFACT_STORE_MAX_ITEMS = 250
PIPELINE_STUDIO_FLOW_LAYOUT_VERSION = 1
PIPELINE_STUDIO_FLOW_LAYOUT_PATH = os.path.join(
    APP_ROOT, "pipeline_store", "pipeline_studio_flow_layout.json"
)
PIPELINE_STUDIO_FLOW_LAYOUT_MAX_ITEMS = 100
PIPELINE_STUDIO_PIPELINE_REGISTRY_VERSION = 1
PIPELINE_STUDIO_PIPELINE_REGISTRY_PATH = os.path.join(
    APP_ROOT, "pipeline_store", "pipeline_registry.json"
)
PIPELINE_STUDIO_PIPELINE_REGISTRY_MAX_ITEMS = 50
PIPELINE_STUDIO_CODE_DRAFTS_VERSION = 1
PIPELINE_STUDIO_CODE_DRAFTS_PATH = os.path.join(
    APP_ROOT, "pipeline_store", "pipeline_studio_code_drafts.json"
)
PIPELINE_STUDIO_CODE_DRAFTS_MAX_ITEMS = 250
PIPELINE_STUDIO_DATASET_STORE_VERSION = 1
PIPELINE_STUDIO_DATASET_STORE_PATH = os.path.join(
    APP_ROOT, "pipeline_store", "pipeline_studio_dataset_store.json"
)
PIPELINE_STUDIO_DATASET_STORE_DIR = os.path.join(
    APP_ROOT, "pipeline_store", "pipeline_datasets"
)
PIPELINE_STUDIO_DATASET_STORE_MAX_ITEMS = 0
PIPELINE_STUDIO_DATASET_CACHE_MAX_ITEMS_DEFAULT = 5
PIPELINE_STUDIO_DATASET_CACHE_MAX_MB_DEFAULT = 500
PIPELINE_STUDIO_PROJECT_PREVIEW_MAX_ROWS = 20
PIPELINE_STUDIO_PROJECT_PREVIEW_MAX_COLS = 50
PIPELINE_STUDIO_PROJECTS_VERSION = 1
PIPELINE_STUDIO_PROJECTS_DIR = os.path.join(
    APP_ROOT, "pipeline_store", "pipeline_projects"
)
PIPELINE_STUDIO_PROJECTS_MAX_ITEMS = 25
PIPELINE_STUDIO_HISTORY_MAX_ITEMS = 25


def _pipeline_studio_history_init() -> None:
    if "pipeline_studio_undo_stack" not in st.session_state:
        st.session_state["pipeline_studio_undo_stack"] = []
    if "pipeline_studio_redo_stack" not in st.session_state:
        st.session_state["pipeline_studio_redo_stack"] = []


def _pipeline_studio_push_history(action: dict) -> None:
    """
    Record a reversible Pipeline Studio operation (best effort).
    This is intentionally in-memory only (may include DataFrames).
    """
    try:
        _pipeline_studio_history_init()
        undo = st.session_state.get("pipeline_studio_undo_stack")
        undo = undo if isinstance(undo, list) else []
        undo.append(action if isinstance(action, dict) else {})
        if len(undo) > int(PIPELINE_STUDIO_HISTORY_MAX_ITEMS):
            undo = undo[-int(PIPELINE_STUDIO_HISTORY_MAX_ITEMS) :]
        st.session_state["pipeline_studio_undo_stack"] = undo
        st.session_state["pipeline_studio_redo_stack"] = []
    except Exception:
        pass


def _pipeline_studio_build_pipelines_from_team_state(team_state: dict) -> dict:
    team_state = team_state if isinstance(team_state, dict) else {}
    ds = team_state.get("datasets")
    ds = ds if isinstance(ds, dict) else {}
    active_id = team_state.get("active_dataset_id")
    active_id = active_id if isinstance(active_id, str) else None
    if not ds:
        return {}
    return {
        "model": build_pipeline_snapshot(ds, active_dataset_id=active_id),
        "active": build_pipeline_snapshot(
            ds, active_dataset_id=active_id, target="active"
        ),
        "latest": build_pipeline_snapshot(
            ds, active_dataset_id=active_id, target="latest"
        ),
    }


def _pipeline_studio_undo_last_action() -> None:
    try:
        _pipeline_studio_history_init()
        undo = st.session_state.get("pipeline_studio_undo_stack")
        undo = undo if isinstance(undo, list) else []
        if not undo:
            return

        action = undo.pop()
        action = action if isinstance(action, dict) else {}

        action_type = str(action.get("type") or "")
        if action_type not in {"create_dataset", "create_datasets"}:
            st.session_state["pipeline_studio_history_notice"] = (
                f"Undo not implemented for action type `{action_type}`."
            )
            undo.append(action)
            st.session_state["pipeline_studio_undo_stack"] = undo
            return

        prev_active = action.get("prev_active_dataset_id")
        prev_active = (
            prev_active if isinstance(prev_active, str) and prev_active else None
        )

        remove_ids: list[str] = []
        if action_type == "create_dataset":
            dataset_id = action.get("dataset_id")
            dataset_id = (
                dataset_id if isinstance(dataset_id, str) and dataset_id else None
            )
            if dataset_id:
                remove_ids = [dataset_id]
        else:
            ids = action.get("dataset_ids")
            ids = ids if isinstance(ids, list) else []
            remove_ids = [str(x) for x in ids if isinstance(x, str) and x]

        if not remove_ids:
            st.session_state["pipeline_studio_history_notice"] = (
                "Undo failed: missing dataset id(s)."
            )
            return

        team_state = st.session_state.get("team_state", {})
        team_state = team_state if isinstance(team_state, dict) else {}
        datasets = team_state.get("datasets")
        datasets = datasets if isinstance(datasets, dict) else {}

        remove_set = set(remove_ids)
        existing_to_remove = [did for did in remove_ids if did in datasets]
        if not existing_to_remove:
            st.session_state["pipeline_studio_history_notice"] = (
                f"Undo skipped: dataset(s) already gone: {', '.join([f'`{x}`' for x in remove_ids])}."
            )
        else:
            # Defensive: don't remove datasets that have downstream children outside the removal set.
            for did, ent in datasets.items():
                if did in remove_set or not isinstance(ent, dict):
                    continue
                pids = ent.get("parent_ids")
                pids = pids if isinstance(pids, list) else []
                pid = ent.get("parent_id")
                parents = []
                if isinstance(pid, str) and pid:
                    parents.append(pid)
                parents.extend([p for p in pids if isinstance(p, str) and p])
                if any(p in remove_set for p in parents):
                    st.session_state["pipeline_studio_history_notice"] = (
                        f"Cannot undo: dataset(s) have downstream dataset `{did}`."
                    )
                    undo.append(action)
                    st.session_state["pipeline_studio_undo_stack"] = undo
                    return

            datasets = dict(datasets)
            for did in existing_to_remove:
                datasets.pop(did, None)
            team_state = {**team_state, "datasets": datasets}

            active_now = team_state.get("active_dataset_id")
            active_now = active_now if isinstance(active_now, str) else None
            if prev_active and prev_active in datasets:
                team_state["active_dataset_id"] = prev_active
            elif active_now in remove_set or (
                active_now and active_now not in datasets
            ):
                # pick newest by created_ts
                best_id = None
                best_ts = -1.0
                for did, ent in datasets.items():
                    if not isinstance(ent, dict):
                        continue
                    try:
                        ts = float(ent.get("created_ts") or 0.0)
                    except Exception:
                        ts = 0.0
                    if ts >= best_ts:
                        best_ts = ts
                        best_id = did
                team_state["active_dataset_id"] = best_id

            st.session_state["team_state"] = team_state

            # Remove per-dataset artifact index entries (best effort).
            try:
                idx_map = st.session_state.get("pipeline_studio_artifacts")
                if isinstance(idx_map, dict):
                    idx_map = dict(idx_map)
                    for did in existing_to_remove:
                        idx_map.pop(did, None)
                    st.session_state["pipeline_studio_artifacts"] = idx_map
            except Exception:
                pass

            # Update persisted registry (best effort).
            try:
                pipelines_new = _pipeline_studio_build_pipelines_from_team_state(
                    team_state
                )
                ds_new = team_state.get("datasets")
                ds_new = ds_new if isinstance(ds_new, dict) else {}
                _update_pipeline_registry_store_for_pipelines(
                    pipelines=pipelines_new, datasets=ds_new
                )
            except Exception:
                pass

            new_active = team_state.get("active_dataset_id")
            new_active = new_active if isinstance(new_active, str) else None
            if new_active:
                st.session_state["pipeline_studio_node_id_pending"] = new_active
                st.session_state["pipeline_studio_autofollow_pending"] = True

            st.session_state["pipeline_studio_history_notice"] = (
                f"Undid last action: removed {len(existing_to_remove)} dataset(s)."
            )

        redo = st.session_state.get("pipeline_studio_redo_stack")
        redo = redo if isinstance(redo, list) else []
        redo.append(action)
        st.session_state["pipeline_studio_redo_stack"] = redo
        st.session_state["pipeline_studio_undo_stack"] = undo
    except Exception as e:
        st.session_state["pipeline_studio_history_notice"] = f"Undo failed: {e}"


def _pipeline_studio_redo_last_action() -> None:
    try:
        _pipeline_studio_history_init()
        redo = st.session_state.get("pipeline_studio_redo_stack")
        redo = redo if isinstance(redo, list) else []
        if not redo:
            return

        action = redo.pop()
        action = action if isinstance(action, dict) else {}
        action_type = str(action.get("type") or "")
        if action_type not in {"create_dataset", "create_datasets"}:
            st.session_state["pipeline_studio_history_notice"] = (
                f"Redo not implemented for action type `{action_type}`."
            )
            redo.append(action)
            st.session_state["pipeline_studio_redo_stack"] = redo
            return

        dataset_ids: list[str] = []
        entries_by_id: dict[str, dict] = {}
        if action_type == "create_dataset":
            dataset_id = action.get("dataset_id")
            dataset_id = (
                dataset_id if isinstance(dataset_id, str) and dataset_id else None
            )
            dataset_entry = action.get("dataset_entry")
            dataset_entry = dataset_entry if isinstance(dataset_entry, dict) else None
            if dataset_id and dataset_entry is not None:
                dataset_ids = [dataset_id]
                entries_by_id = {dataset_id: dataset_entry}
        else:
            ids = action.get("dataset_ids")
            ids = ids if isinstance(ids, list) else []
            dataset_ids = [str(x) for x in ids if isinstance(x, str) and x]
            eby = action.get("dataset_entries_by_id")
            eby = eby if isinstance(eby, dict) else {}
            entries_by_id = {
                str(k): v
                for k, v in eby.items()
                if isinstance(k, str) and isinstance(v, dict)
            }

        if not dataset_ids or any(did not in entries_by_id for did in dataset_ids):
            st.session_state["pipeline_studio_history_notice"] = (
                "Redo failed: missing dataset payload."
            )
            return

        team_state = st.session_state.get("team_state", {})
        team_state = team_state if isinstance(team_state, dict) else {}
        datasets = team_state.get("datasets")
        datasets = datasets if isinstance(datasets, dict) else {}
        datasets = dict(datasets)
        for did in dataset_ids:
            datasets[did] = entries_by_id[did]
        team_state = {
            **team_state,
            "datasets": datasets,
            "active_dataset_id": dataset_ids[-1],
        }
        st.session_state["team_state"] = team_state

        # Update persisted registry (best effort).
        try:
            pipelines_new = _pipeline_studio_build_pipelines_from_team_state(team_state)
            _update_pipeline_registry_store_for_pipelines(
                pipelines=pipelines_new, datasets=datasets
            )
        except Exception:
            pass

        st.session_state["pipeline_studio_node_id_pending"] = dataset_ids[-1]
        st.session_state["pipeline_studio_autofollow_pending"] = True
        st.session_state["pipeline_studio_history_notice"] = (
            f"Redid last action: restored {len(dataset_ids)} dataset(s)."
        )

        undo = st.session_state.get("pipeline_studio_undo_stack")
        undo = undo if isinstance(undo, list) else []
        undo.append(action)
        if len(undo) > int(PIPELINE_STUDIO_HISTORY_MAX_ITEMS):
            undo = undo[-int(PIPELINE_STUDIO_HISTORY_MAX_ITEMS) :]
        st.session_state["pipeline_studio_undo_stack"] = undo
        st.session_state["pipeline_studio_redo_stack"] = redo
    except Exception as e:
        st.session_state["pipeline_studio_history_notice"] = f"Redo failed: {e}"


def _load_pipeline_studio_code_drafts_store() -> dict:
    """
    Load a small, file-backed store of per-dataset code drafts keyed by dataset fingerprint.
    """
    loaded_flag = "_pipeline_studio_code_drafts_store_loaded"
    if bool(st.session_state.get(loaded_flag)):
        store = st.session_state.get("pipeline_studio_code_drafts_store")
        return store if isinstance(store, dict) else {}

    store: dict = {
        "version": PIPELINE_STUDIO_CODE_DRAFTS_VERSION,
        "path": PIPELINE_STUDIO_CODE_DRAFTS_PATH,
        "by_fingerprint": {},
    }
    try:
        out_dir = os.path.dirname(PIPELINE_STUDIO_CODE_DRAFTS_PATH) or "."
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(PIPELINE_STUDIO_CODE_DRAFTS_PATH):
            with open(PIPELINE_STUDIO_CODE_DRAFTS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                if isinstance(data.get("by_fingerprint"), dict):
                    store.update(
                        {
                            "version": int(
                                data.get("version")
                                or PIPELINE_STUDIO_CODE_DRAFTS_VERSION
                            ),
                            "by_fingerprint": data.get("by_fingerprint") or {},
                        }
                    )
                else:
                    # legacy: direct mapping {fingerprint: draft_record}
                    store["by_fingerprint"] = data
    except Exception:
        pass

    st.session_state["pipeline_studio_code_drafts_store"] = store
    st.session_state[loaded_flag] = True
    return store


def _save_pipeline_studio_code_drafts_store(store: dict) -> None:
    try:
        import tempfile
        import time

        if not isinstance(store, dict):
            return
        by_fp = store.get("by_fingerprint")
        by_fp = by_fp if isinstance(by_fp, dict) else {}

        if len(by_fp) > int(PIPELINE_STUDIO_CODE_DRAFTS_MAX_ITEMS):
            items: list[tuple[float, str]] = []
            for fp, rec in by_fp.items():
                ts = 0.0
                if isinstance(rec, dict):
                    try:
                        ts = float(
                            rec.get("updated_ts") or rec.get("created_ts") or 0.0
                        )
                    except Exception:
                        ts = 0.0
                items.append((ts, str(fp)))
            items.sort(reverse=True)
            keep = {
                fp for _ts, fp in items[: int(PIPELINE_STUDIO_CODE_DRAFTS_MAX_ITEMS)]
            }
            by_fp = {fp: by_fp[fp] for fp in keep if fp in by_fp}
            store["by_fingerprint"] = by_fp

        store["version"] = PIPELINE_STUDIO_CODE_DRAFTS_VERSION
        store["updated_ts"] = time.time()
        store["path"] = PIPELINE_STUDIO_CODE_DRAFTS_PATH

        out_dir = os.path.dirname(PIPELINE_STUDIO_CODE_DRAFTS_PATH) or "."
        os.makedirs(out_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix="._pipeline_code_drafts_", suffix=".json", dir=out_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(store, f, indent=2, default=str)
            os.replace(tmp_path, PIPELINE_STUDIO_CODE_DRAFTS_PATH)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
    except Exception:
        pass


def _get_pipeline_studio_code_draft(*, fingerprint: str) -> dict:
    try:
        if not isinstance(fingerprint, str) or not fingerprint:
            return {}
        store = _load_pipeline_studio_code_drafts_store()
        by_fp = store.get("by_fingerprint")
        by_fp = by_fp if isinstance(by_fp, dict) else {}
        rec = by_fp.get(fingerprint)
        return rec if isinstance(rec, dict) else {}
    except Exception:
        return {}


def _save_pipeline_studio_code_draft(
    *,
    fingerprint: str,
    dataset_id: str | None,
    transform_kind: str | None,
    lang: str | None,
    draft_code: str,
) -> None:
    try:
        import time

        fingerprint = fingerprint.strip() if isinstance(fingerprint, str) else ""
        if not fingerprint:
            return
        draft_code = draft_code.strip() if isinstance(draft_code, str) else ""
        if not draft_code:
            return

        store = _load_pipeline_studio_code_drafts_store()
        by_fp = store.get("by_fingerprint")
        by_fp = by_fp if isinstance(by_fp, dict) else {}
        prev = by_fp.get(fingerprint)
        prev = prev if isinstance(prev, dict) else {}
        rec = {
            "fingerprint": fingerprint,
            "dataset_id": dataset_id,
            "transform_kind": transform_kind,
            "lang": lang,
            "draft_code": draft_code,
            "updated_ts": time.time(),
            "created_ts": prev.get("created_ts") or time.time(),
        }
        by_fp[fingerprint] = rec
        store["by_fingerprint"] = by_fp
        st.session_state["pipeline_studio_code_drafts_store"] = store
        _save_pipeline_studio_code_drafts_store(store)
    except Exception:
        pass


def _delete_pipeline_studio_code_draft(*, fingerprint: str) -> None:
    try:
        fingerprint = fingerprint.strip() if isinstance(fingerprint, str) else ""
        if not fingerprint:
            return
        store = _load_pipeline_studio_code_drafts_store()
        by_fp = store.get("by_fingerprint")
        by_fp = by_fp if isinstance(by_fp, dict) else {}
        if fingerprint in by_fp:
            by_fp.pop(fingerprint, None)
            store["by_fingerprint"] = by_fp
            st.session_state["pipeline_studio_code_drafts_store"] = store
            _save_pipeline_studio_code_drafts_store(store)
    except Exception:
        pass


def _pipeline_studio_project_slug(name: str) -> str:
    name = name.strip() if isinstance(name, str) else ""
    name = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-_")
    name = re.sub(r"-{2,}", "-", name).strip("-_")
    return name.lower() if name else "project"


def _pipeline_studio_load_project_manifest(*, project_dir: str) -> dict | None:
    project_dir = project_dir.strip() if isinstance(project_dir, str) else ""
    if not project_dir:
        return None
    manifest_path = os.path.join(project_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return None
    return manifest if isinstance(manifest, dict) else None


def _pipeline_studio_write_project_manifest(
    *, project_dir: str, manifest: dict
) -> bool:
    try:
        import tempfile

        project_dir = project_dir.strip() if isinstance(project_dir, str) else ""
        if not project_dir or not isinstance(manifest, dict):
            return False
        manifest_path = os.path.join(project_dir, "manifest.json")
        fd, tmp_path = tempfile.mkstemp(
            prefix="._manifest_", suffix=".json", dir=project_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, default=str)
            os.replace(tmp_path, manifest_path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
        return True
    except Exception:
        return False


def _pipeline_studio_update_project_manifest(
    *,
    project_dir: str,
    updates: dict | None = None,
    dataset_source_updates: dict[str, str] | None = None,
) -> dict | None:
    manifest = _pipeline_studio_load_project_manifest(project_dir=project_dir)
    if not isinstance(manifest, dict):
        return None
    updates = updates if isinstance(updates, dict) else {}
    if updates:
        manifest.update(updates)
    if isinstance(dataset_source_updates, dict) and dataset_source_updates:
        team = manifest.get("team_state")
        team = team if isinstance(team, dict) else {}
        datasets_meta = team.get("datasets")
        datasets_meta = datasets_meta if isinstance(datasets_meta, dict) else {}
        for did, new_source in dataset_source_updates.items():
            if not isinstance(did, str) or not did:
                continue
            if not isinstance(new_source, str) or not new_source.strip():
                continue
            entry = datasets_meta.get(did)
            entry = entry if isinstance(entry, dict) else {}
            prov = entry.get("provenance")
            prov = prov if isinstance(prov, dict) else {}
            prov["source"] = new_source.strip()
            prov["source_type"] = prov.get("source_type") or "file"
            entry["provenance"] = prov
            datasets_meta[did] = entry
        team["datasets"] = datasets_meta
        manifest["team_state"] = team
    _pipeline_studio_write_project_manifest(project_dir=project_dir, manifest=manifest)
    return manifest


def _pipeline_studio_project_disk_usage(dir_path: str) -> int:
    total = 0
    dir_path = dir_path.strip() if isinstance(dir_path, str) else ""
    if not dir_path or not os.path.isdir(dir_path):
        return 0
    for root, _dirs, files in os.walk(dir_path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except Exception:
                pass
    return total


def _pipeline_studio_format_bytes(value: int | float | None) -> str:
    try:
        size = float(value or 0.0)
    except Exception:
        return "0 B"
    if size <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    return f"{size:.1f} {units[idx]}"


def _pipeline_studio_dataset_cache_usage() -> int:
    total = 0
    cache_dir = PIPELINE_STUDIO_DATASET_STORE_DIR
    if not cache_dir or not os.path.isdir(cache_dir):
        return 0
    for root, _dirs, files in os.walk(cache_dir):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except Exception:
                pass
    return total


def _pipeline_studio_format_ts(value: float | int | None) -> str:
    try:
        ts = float(value or 0.0)
    except Exception:
        return "-"
    if ts <= 0:
        return "-"
    try:
        from datetime import datetime

        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(int(ts))


def _pipeline_studio_rename_project(*, dir_name: str, new_name: str) -> dict:
    import time

    dir_name = dir_name.strip() if isinstance(dir_name, str) else ""
    new_name = new_name.strip() if isinstance(new_name, str) else ""
    if not dir_name or not new_name:
        return {"error": "Select a project and enter a new name."}
    project_dir = os.path.join(PIPELINE_STUDIO_PROJECTS_DIR, dir_name)
    if not os.path.isdir(project_dir):
        return {"error": "Project not found on disk."}
    slug = _pipeline_studio_project_slug(new_name)
    suffix = ""
    if "_" in dir_name:
        suffix = dir_name.split("_")[-1]
    if not suffix.isdigit():
        suffix = str(int(time.time()))
    new_dir_name = f"{slug}_{suffix}"
    new_dir = os.path.join(PIPELINE_STUDIO_PROJECTS_DIR, new_dir_name)
    if os.path.exists(new_dir):
        new_dir_name = f"{slug}_{suffix}_{uuid.uuid4().hex[:6]}"
        new_dir = os.path.join(PIPELINE_STUDIO_PROJECTS_DIR, new_dir_name)
    try:
        os.rename(project_dir, new_dir)
    except Exception as exc:
        return {"error": str(exc)}
    _pipeline_studio_update_project_manifest(
        project_dir=new_dir,
        updates={"name": new_name, "dir_name": new_dir_name},
    )
    return {"dir_name": new_dir_name, "dir_path": new_dir}


def _pipeline_studio_duplicate_project(*, dir_name: str, new_name: str) -> dict:
    import shutil
    import time

    dir_name = dir_name.strip() if isinstance(dir_name, str) else ""
    new_name = new_name.strip() if isinstance(new_name, str) else ""
    if not dir_name or not new_name:
        return {"error": "Select a project and enter a name."}
    project_dir = os.path.join(PIPELINE_STUDIO_PROJECTS_DIR, dir_name)
    if not os.path.isdir(project_dir):
        return {"error": "Project not found on disk."}
    slug = _pipeline_studio_project_slug(new_name)
    ts = str(int(time.time()))
    new_dir_name = f"{slug}_{ts}"
    new_dir = os.path.join(PIPELINE_STUDIO_PROJECTS_DIR, new_dir_name)
    if os.path.exists(new_dir):
        new_dir_name = f"{slug}_{ts}_{uuid.uuid4().hex[:6]}"
        new_dir = os.path.join(PIPELINE_STUDIO_PROJECTS_DIR, new_dir_name)
    try:
        shutil.copytree(project_dir, new_dir)
    except Exception as exc:
        return {"error": str(exc)}
    _pipeline_studio_update_project_manifest(
        project_dir=new_dir,
        updates={
            "name": new_name,
            "dir_name": new_dir_name,
            "saved_ts": time.time(),
            "last_opened_ts": None,
            "archived": False,
        },
    )
    return {"dir_name": new_dir_name, "dir_path": new_dir}


def _pipeline_studio_save_project_metadata(
    *, dir_name: str, tags: list[str], notes: str, archived: bool
) -> None:
    dir_name = dir_name.strip() if isinstance(dir_name, str) else ""
    if not dir_name:
        st.session_state["pipeline_studio_project_notice"] = (
            "Error: Select a project to update."
        )
        return
    project_dir = os.path.join(PIPELINE_STUDIO_PROJECTS_DIR, dir_name)
    if not os.path.isdir(project_dir):
        st.session_state["pipeline_studio_project_notice"] = (
            "Error: Project not found on disk."
        )
        return
    clean_tags = [t.strip() for t in tags if isinstance(t, str) and t.strip()]
    _pipeline_studio_update_project_manifest(
        project_dir=project_dir,
        updates={
            "tags": clean_tags,
            "notes": notes if isinstance(notes, str) else "",
            "archived": bool(archived),
        },
    )
    st.session_state["pipeline_studio_project_notice"] = (
        f"Updated project metadata for `{dir_name}`."
    )


def _pipeline_studio_convert_project_to_metadata_only(*, dir_name: str) -> None:
    dir_name = dir_name.strip() if isinstance(dir_name, str) else ""
    if not dir_name:
        st.session_state["pipeline_studio_project_notice"] = (
            "Error: Select a project to convert."
        )
        return
    project_dir = os.path.join(PIPELINE_STUDIO_PROJECTS_DIR, dir_name)
    if not os.path.isdir(project_dir):
        st.session_state["pipeline_studio_project_notice"] = (
            "Error: Project not found on disk."
        )
        return
    manifest = _pipeline_studio_load_project_manifest(project_dir=project_dir)
    if not isinstance(manifest, dict):
        st.session_state["pipeline_studio_project_notice"] = (
            "Error: Manifest not found."
        )
        return
    team = manifest.get("team_state")
    team = team if isinstance(team, dict) else {}
    datasets_meta = team.get("datasets")
    datasets_meta = datasets_meta if isinstance(datasets_meta, dict) else {}

    removed_files = 0
    for did, meta in datasets_meta.items():
        if not isinstance(did, str) or not did or not isinstance(meta, dict):
            continue
        rel_path = meta.get("data_path")
        rel_path = rel_path if isinstance(rel_path, str) and rel_path else None
        if rel_path:
            abs_path = os.path.join(project_dir, rel_path)
            try:
                if os.path.exists(abs_path):
                    os.remove(abs_path)
                    removed_files += 1
            except Exception:
                pass
        meta.pop("data_path", None)
        meta.pop("data_format", None)
        meta.pop("data_bytes", None)
        meta["data_saved"] = False
        datasets_meta[did] = meta

    team["datasets"] = datasets_meta
    manifest["team_state"] = team
    manifest["data_mode"] = "metadata_only"
    manifest["datasets_saved"] = 0
    _pipeline_studio_write_project_manifest(project_dir=project_dir, manifest=manifest)
    st.session_state["pipeline_studio_project_notice"] = (
        f"Converted project to metadata-only (removed {removed_files} data file(s))."
    )


def _pipeline_studio_bulk_delete_projects(dir_names: list[str]) -> None:
    dir_names = [d for d in (dir_names or []) if isinstance(d, str) and d.strip()]
    if not dir_names:
        st.session_state["pipeline_studio_project_notice"] = (
            "Error: Select project(s) to delete."
        )
        return
    deleted = 0
    errors: list[str] = []
    for dir_name in dir_names:
        project_dir = os.path.join(PIPELINE_STUDIO_PROJECTS_DIR, dir_name)
        if not os.path.isdir(project_dir):
            errors.append(f"{dir_name}: not found")
            continue
        try:
            shutil.rmtree(project_dir)
            deleted += 1
        except Exception as exc:
            errors.append(f"{dir_name}: {exc}")
    if errors:
        st.session_state["pipeline_studio_project_notice"] = (
            f"Deleted {deleted} project(s). Errors: {', '.join(errors)}"
        )
    else:
        st.session_state["pipeline_studio_project_notice"] = (
            f"Deleted {deleted} project(s)."
        )


def _pipeline_studio_list_projects() -> list[dict]:
    """
    Best-effort list of Pipeline Studio projects saved under `pipeline_store/pipeline_projects/`.
    """
    try:
        root = PIPELINE_STUDIO_PROJECTS_DIR
        if not os.path.isdir(root):
            return []
        items: list[dict] = []
        for dir_name in os.listdir(root):
            if not isinstance(dir_name, str) or not dir_name:
                continue
            dir_path = os.path.join(root, dir_name)
            if not os.path.isdir(dir_path):
                continue
            manifest = _pipeline_studio_load_project_manifest(project_dir=dir_path)
            if not isinstance(manifest, dict):
                continue
            try:
                saved_ts = float(
                    manifest.get("saved_ts") or manifest.get("created_ts") or 0.0
                )
            except Exception:
                saved_ts = 0.0
            try:
                last_opened_ts = float(manifest.get("last_opened_ts") or 0.0)
            except Exception:
                last_opened_ts = 0.0
            datasets_total = manifest.get("datasets_total")
            if not datasets_total:
                team_state = manifest.get("team_state")
                team_state = team_state if isinstance(team_state, dict) else {}
                ds_meta = team_state.get("datasets")
                ds_meta = ds_meta if isinstance(ds_meta, dict) else {}
                datasets_total = len(ds_meta)
            data_mode = manifest.get("data_mode")
            if not data_mode:
                team_state = manifest.get("team_state")
                team_state = team_state if isinstance(team_state, dict) else {}
                ds_meta = team_state.get("datasets")
                ds_meta = ds_meta if isinstance(ds_meta, dict) else {}
                data_mode = (
                    "full"
                    if any(
                        isinstance(rec, dict) and rec.get("data_path")
                        for rec in ds_meta.values()
                    )
                    else "metadata_only"
                )
            items.append(
                {
                    "dir_name": dir_name,
                    "dir_path": dir_path,
                    "manifest_path": os.path.join(dir_path, "manifest.json"),
                    "saved_ts": saved_ts,
                    "last_opened_ts": last_opened_ts,
                    "name": manifest.get("name")
                    if isinstance(manifest.get("name"), str)
                    else dir_name,
                    "pipeline_hash": manifest.get("pipeline_hash")
                    if isinstance(manifest.get("pipeline_hash"), str)
                    else "",
                    "data_mode": data_mode or "full",
                    "datasets_total": datasets_total or 0,
                    "datasets_saved": manifest.get("datasets_saved") or 0,
                    "tags": manifest.get("tags") or [],
                    "notes": manifest.get("notes") or "",
                    "archived": bool(manifest.get("archived", False)),
                    "manifest": manifest,
                }
            )
        items.sort(key=lambda x: float(x.get("saved_ts") or 0.0), reverse=True)
        return items
    except Exception:
        return []


def _pipeline_studio_prune_projects(*, max_items: int) -> None:
    try:
        import shutil

        max_items = int(max_items or 0)
        if max_items <= 0:
            return
        projects = _pipeline_studio_list_projects()
        if len(projects) <= max_items:
            return
        for rec in projects[max_items:]:
            dir_path = rec.get("dir_path")
            if isinstance(dir_path, str) and dir_path and os.path.isdir(dir_path):
                shutil.rmtree(dir_path, ignore_errors=True)
    except Exception:
        pass


def _pipeline_studio_save_project(
    *, name: str, team_state: dict, include_data: bool = False
) -> dict:
    """
    Save a best-effort, local "project" bundle:
    - dataset frames (pickle) + dataset metadata (manifest.json), or metadata-only
    - pipeline registry + flow layout + code drafts (scoped) for portability

    Note: full-data saves store Parquet when available (fallback to pickle); do not load untrusted project files.
    """
    try:
        import tempfile
        import time

        team_state = team_state if isinstance(team_state, dict) else {}
        datasets = team_state.get("datasets")
        datasets = datasets if isinstance(datasets, dict) else {}
        active_id = team_state.get("active_dataset_id")
        active_id = active_id if isinstance(active_id, str) and active_id else None

        os.makedirs(PIPELINE_STUDIO_PROJECTS_DIR, exist_ok=True)
        slug = _pipeline_studio_project_slug(name)
        ts = int(time.time())
        dir_name = f"{slug}_{ts}"
        project_dir = os.path.join(PIPELINE_STUDIO_PROJECTS_DIR, dir_name)
        if os.path.exists(project_dir):
            dir_name = f"{slug}_{ts}_{uuid.uuid4().hex[:6]}"
            project_dir = os.path.join(PIPELINE_STUDIO_PROJECTS_DIR, dir_name)
        os.makedirs(project_dir, exist_ok=True)
        if include_data:
            ds_dir = os.path.join(project_dir, "datasets")
            os.makedirs(ds_dir, exist_ok=True)

        datasets_out: dict[str, dict] = {}
        fingerprints: set[str] = set()
        saved_count = 0
        for did, entry in datasets.items():
            if not isinstance(did, str) or not did:
                continue
            entry = entry if isinstance(entry, dict) else {}
            data = entry.get("data")
            df: pd.DataFrame | None = None
            try:
                if isinstance(data, pd.DataFrame):
                    df = data
                elif isinstance(data, dict):
                    df = pd.DataFrame.from_dict(data)
                elif isinstance(data, list):
                    df = pd.DataFrame(data)
            except Exception:
                df = None
            meta = {k: v for k, v in entry.items() if k != "data"}
            if not include_data:
                meta.pop("data_path", None)
                meta.pop("data_format", None)
                meta.pop("data_bytes", None)
            meta["data_saved"] = False
            if include_data and df is not None:
                rel_path = os.path.join("datasets", f"{did}.parquet")
                abs_path = os.path.join(project_dir, rel_path)

                def _write_parquet(path: str) -> bool:
                    try:
                        for compression in ("zstd", "snappy", "gzip", None):
                            try:
                                df.to_parquet(
                                    path,
                                    index=False,
                                    compression=compression,
                                )
                                return True
                            except Exception:
                                continue
                    except Exception:
                        return False
                    return False

                if _write_parquet(abs_path):
                    meta["data_path"] = rel_path
                    meta["data_format"] = "parquet"
                    meta["data_saved"] = True
                    saved_count += 1
                else:
                    rel_path = os.path.join("datasets", f"{did}.pkl")
                    abs_path = os.path.join(project_dir, rel_path)
                    df.to_pickle(abs_path)
                    meta["data_path"] = rel_path
                    meta["data_format"] = "pickle"
                    meta["data_saved"] = True
                    saved_count += 1
            elif df is not None:
                max_rows = int(PIPELINE_STUDIO_PROJECT_PREVIEW_MAX_ROWS)
                max_cols = int(PIPELINE_STUDIO_PROJECT_PREVIEW_MAX_COLS)
                try:
                    df_preview = df.iloc[:max_rows, :max_cols].copy()
                    meta["preview_rows"] = int(df_preview.shape[0])
                    meta["preview_cols"] = [str(c) for c in df_preview.columns]
                    meta["preview_data"] = df_preview.to_dict(orient="records")
                    meta["preview_truncated"] = bool(
                        df.shape[0] > max_rows or df.shape[1] > max_cols
                    )
                except Exception:
                    pass
            elif include_data:
                meta.pop("data_path", None)
                meta.pop("data_format", None)
                meta.pop("data_bytes", None)
            datasets_out[did] = meta

            fp = entry.get("fingerprint")
            if isinstance(fp, str) and fp:
                fingerprints.add(fp)

        pipelines = _pipeline_studio_build_pipelines_from_team_state(team_state)
        pipeline_hashes = {}
        registry_by_ph = {}
        flow_layout_by_ph = {}
        for name_key, pipe in pipelines.items():
            pipe = pipe if isinstance(pipe, dict) else {}
            ph = pipe.get("pipeline_hash")
            ph = ph if isinstance(ph, str) and ph.strip() else None
            if not ph:
                continue
            pipeline_hashes[name_key] = ph
            try:
                registry_by_ph[ph] = _get_persisted_pipeline_registry(pipeline_hash=ph)
            except Exception:
                pass
            try:
                flow_layout_by_ph[ph] = _get_persisted_pipeline_studio_flow_layout(
                    pipeline_hash=ph
                )
            except Exception:
                pass

        code_drafts_by_fp: dict[str, dict] = {}
        try:
            store = _load_pipeline_studio_code_drafts_store()
            by_fp = store.get("by_fingerprint")
            by_fp = by_fp if isinstance(by_fp, dict) else {}
            for fp in fingerprints:
                rec = by_fp.get(fp)
                if isinstance(rec, dict):
                    code_drafts_by_fp[fp] = rec
        except Exception:
            code_drafts_by_fp = {}

        manifest = {
            "version": PIPELINE_STUDIO_PROJECTS_VERSION,
            "name": name.strip()
            if isinstance(name, str) and name.strip()
            else dir_name,
            "saved_ts": time.time(),
            "dir_name": dir_name,
            "data_mode": "full" if include_data else "metadata_only",
            "datasets_saved": int(saved_count),
            "datasets_total": int(len(datasets_out)),
            "tags": [],
            "notes": "",
            "archived": False,
            "last_opened_ts": None,
            "pipeline_hashes": pipeline_hashes,
            "team_state": {
                "active_dataset_id": active_id,
                "datasets": datasets_out,
            },
            "registry_by_pipeline_hash": registry_by_ph,
            "flow_layout_by_pipeline_hash": flow_layout_by_ph,
            "code_drafts_by_fingerprint": code_drafts_by_fp,
        }

        manifest_path = os.path.join(project_dir, "manifest.json")
        fd, tmp_path = tempfile.mkstemp(
            prefix="._manifest_", suffix=".json", dir=project_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, default=str)
            os.replace(tmp_path, manifest_path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

        _pipeline_studio_prune_projects(max_items=PIPELINE_STUDIO_PROJECTS_MAX_ITEMS)
        return {
            "project_dir": project_dir,
            "manifest_path": manifest_path,
            "dir_name": dir_name,
        }
    except Exception as e:
        return {"error": str(e)}


def _pipeline_studio_load_project(*, project_dir: str, rehydrate: bool = True) -> dict:
    """
    Load a previously saved project bundle into the current session_state `team_state`.
    """
    try:
        import time as _time

        project_dir = project_dir.strip() if isinstance(project_dir, str) else ""
        if not project_dir:
            return {"error": "Missing project_dir."}
        manifest = _pipeline_studio_load_project_manifest(project_dir=project_dir)
        if not isinstance(manifest, dict):
            manifest_path = os.path.join(project_dir, "manifest.json")
            return {"error": f"Project manifest not found: {manifest_path}"}
        team = manifest.get("team_state")
        team = team if isinstance(team, dict) else {}
        datasets_meta = team.get("datasets")
        datasets_meta = datasets_meta if isinstance(datasets_meta, dict) else {}
        active_id = team.get("active_dataset_id")
        active_id = active_id if isinstance(active_id, str) and active_id else None
        data_mode = manifest.get("data_mode") or "full"
        metadata_only = str(data_mode).lower() == "metadata_only"

        datasets: dict[str, dict] = {}
        missing_files: list[dict] = []
        loaded_data_files = 0
        for did, meta in datasets_meta.items():
            if not isinstance(did, str) or not did or not isinstance(meta, dict):
                continue
            entry = dict(meta)
            entry["id"] = did
            if metadata_only:
                entry["data"] = None
                datasets[did] = entry
                continue
            rel_path = meta.get("data_path")
            rel_path = rel_path if isinstance(rel_path, str) and rel_path else None
            fmt = meta.get("data_format")
            fmt = fmt if isinstance(fmt, str) and fmt else "pickle"
            df = None
            if rel_path:
                abs_path = os.path.join(project_dir, rel_path)
                if os.path.exists(abs_path):
                    try:
                        if fmt == "parquet":
                            df = pd.read_parquet(abs_path)
                        elif fmt == "pickle":
                            df = pd.read_pickle(abs_path)
                        elif fmt == "csv":
                            df = pd.read_csv(abs_path)
                    except Exception:
                        df = None
                else:
                    missing_files.append(
                        {"dataset_id": did, "data_path": rel_path, "format": fmt}
                    )
            entry["data"] = df
            if isinstance(df, pd.DataFrame):
                loaded_data_files += 1
            datasets[did] = entry

        def _entry_parent_ids(entry_obj: dict) -> list[str]:
            entry_obj = entry_obj if isinstance(entry_obj, dict) else {}
            parents: list[str] = []
            pids = entry_obj.get("parent_ids")
            if isinstance(pids, list):
                parents.extend([str(p) for p in pids if isinstance(p, str) and p])
            pid = entry_obj.get("parent_id")
            if isinstance(pid, str) and pid and pid not in parents:
                parents.insert(0, pid)
            return [p for p in parents if p]

        def _coerce_entry_df(entry_obj: dict) -> pd.DataFrame | None:
            if not isinstance(entry_obj, dict):
                return None
            data = entry_obj.get("data")
            if isinstance(data, pd.DataFrame):
                return data
            try:
                if isinstance(data, dict):
                    return pd.DataFrame.from_dict(data)
                if isinstance(data, list):
                    return pd.DataFrame(data)
            except Exception:
                return None
            return None

        rehydrate_stats = {
            "roots_loaded": 0,
            "transforms_run": 0,
            "missing_sources": 0,
            "transform_failures": 0,
        }
        missing_sources: list[dict] = []

        def _load_root_data(entry_obj: dict) -> pd.DataFrame | None:
            prov = entry_obj.get("provenance")
            prov = prov if isinstance(prov, dict) else {}
            source_type = str(prov.get("source_type") or "")
            source = prov.get("source")
            if source_type not in {"file", "directory_load"}:
                return None
            if not isinstance(source, str) or not source:
                return None
            try:
                from ai_data_science_team.tools.data_loader import auto_load_file
            except Exception:
                return None
            try:
                df = auto_load_file(source, max_rows=None)
            except Exception:
                df = None
            return df if isinstance(df, pd.DataFrame) else None

        def _infer_first_def_name(code: str) -> str | None:
            if not isinstance(code, str) or not code:
                return None
            m = re.search(
                r"^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
                code,
                flags=re.MULTILINE,
            )
            return m.group(1) if m else None

        def _read_text_file(path: object, *, max_bytes: int = 500_000) -> str | None:
            try:
                if not isinstance(path, str) or not path:
                    return None
                if not os.path.exists(path):
                    return None
                if os.path.getsize(path) > max_bytes:
                    return None
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                return None

        def _exec_python_function(
            *, code: str, df_in: pd.DataFrame, fn_name_hint: str | None
        ) -> pd.DataFrame | None:
            code = code if isinstance(code, str) else ""
            code = code.strip()
            if not code:
                return None
            inferred = _infer_first_def_name(code)
            fn_name_hint = (
                fn_name_hint.strip() if isinstance(fn_name_hint, str) else None
            )
            exec_env: dict[str, object] = {"pd": pd}
            try:
                import numpy as np  # type: ignore

                exec_env["np"] = np
            except Exception:
                pass
            exec(code, exec_env, exec_env)
            fn = None
            for candidate in [fn_name_hint, inferred]:
                if not candidate:
                    continue
                obj = exec_env.get(candidate)
                if callable(obj):
                    fn = obj
                    break
            if fn is None:
                for name, obj in exec_env.items():
                    if callable(obj):
                        fn = obj
                        break
            if fn is None:
                return None
            out = fn(df_in)
            if isinstance(out, tuple) and out:
                out = out[0]
            if isinstance(out, pd.DataFrame):
                return out
            try:
                if isinstance(out, dict):
                    return pd.DataFrame.from_dict(out)
                if isinstance(out, list):
                    return pd.DataFrame(out)
            except Exception:
                return None
            return None

        def _exec_merge(
            *, code: str, parents: list[pd.DataFrame]
        ) -> pd.DataFrame | None:
            code = code if isinstance(code, str) else ""
            code = code.strip()
            if not code:
                return None
            exec_env: dict[str, object] = {"pd": pd}
            for idx, df in enumerate(parents):
                exec_env[f"df_{idx}"] = df
            exec(code, exec_env, exec_env)
            df_out = exec_env.get("df")
            if isinstance(df_out, pd.DataFrame):
                return df_out
            try:
                if isinstance(df_out, dict):
                    return pd.DataFrame.from_dict(df_out)
                if isinstance(df_out, list):
                    return pd.DataFrame(df_out)
            except Exception:
                return None
            return None

        def _exec_sql_query(*, sql_code: str, sql_url: str) -> pd.DataFrame | None:
            sql_code = sql_code if isinstance(sql_code, str) else ""
            sql_code = sql_code.strip()
            if not sql_code:
                return None
            try:
                engine = sql.create_engine(sql_url)
                return pd.read_sql_query(sql_code, engine)
            except Exception:
                return None

        def _rehydrate_datasets(
            datasets_map: dict[str, dict],
        ) -> tuple[dict[str, dict], dict]:
            datasets_map = datasets_map if isinstance(datasets_map, dict) else {}
            parents_map: dict[str, list[str]] = {}
            children_map: dict[str, set[str]] = {}
            for did, entry_obj in datasets_map.items():
                if not isinstance(did, str) or not did:
                    continue
                parents = _entry_parent_ids(entry_obj)
                parents_map[did] = parents
                for pid in parents:
                    children_map.setdefault(pid, set()).add(did)

            # Load root datasets from source.
            for did, entry_obj in datasets_map.items():
                if parents_map.get(did):
                    continue
                if _coerce_entry_df(entry_obj) is not None:
                    continue
                prov = entry_obj.get("provenance")
                prov = prov if isinstance(prov, dict) else {}
                source = prov.get("source")
                if isinstance(source, str) and source and not os.path.exists(source):
                    missing_sources.append(
                        {
                            "dataset_id": did,
                            "label": entry_obj.get("label") or did,
                            "source": source,
                            "source_type": prov.get("source_type"),
                            "original_name": prov.get("original_name"),
                        }
                    )
                    rehydrate_stats["missing_sources"] += 1
                    continue
                df_root = _load_root_data(entry_obj)
                if df_root is None:
                    rehydrate_stats["missing_sources"] += 1
                    continue
                entry_obj["data"] = df_root
                shape, cols, schema, schema_hash, fingerprint = (
                    _pipeline_studio_dataset_meta(df_root)
                )
                entry_obj["shape"] = shape
                entry_obj["columns"] = cols
                entry_obj["schema"] = schema
                entry_obj["schema_hash"] = schema_hash
                entry_obj["fingerprint"] = fingerprint
                datasets_map[did] = entry_obj
                rehydrate_stats["roots_loaded"] += 1

            # Topological replay (best effort).
            in_degree: dict[str, int] = {
                did: len(parents_map.get(did) or []) for did in datasets_map.keys()
            }
            queue: list[str] = [did for did, deg in in_degree.items() if deg == 0]
            order: list[str] = []
            while queue:
                cur = queue.pop(0)
                order.append(cur)
                for child in children_map.get(cur, set()):
                    if child not in in_degree:
                        continue
                    in_degree[child] -= 1
                    if in_degree[child] <= 0:
                        queue.append(child)

            sql_url = st.session_state.get("sql_url") or DEFAULT_SQL_URL
            for did in order:
                entry_obj = datasets_map.get(did)
                if not isinstance(entry_obj, dict):
                    continue
                if _coerce_entry_df(entry_obj) is not None:
                    continue
                parents = parents_map.get(did) or []
                if not parents:
                    continue
                parent_dfs: list[pd.DataFrame] = []
                for pid in parents:
                    parent_entry = datasets_map.get(pid)
                    parent_df = _coerce_entry_df(parent_entry) if parent_entry else None
                    if parent_df is None:
                        parent_dfs = []
                        break
                    parent_dfs.append(parent_df)
                if not parent_dfs:
                    continue
                prov = entry_obj.get("provenance")
                prov = prov if isinstance(prov, dict) else {}
                transform = prov.get("transform")
                transform = transform if isinstance(transform, dict) else {}
                kind = str(transform.get("kind") or "")
                out_df = None
                if kind == "python_function":
                    fn_code = transform.get("function_code")
                    if not isinstance(fn_code, str) or not fn_code.strip():
                        fn_code = _read_text_file(transform.get("function_path"))
                    out_df = _exec_python_function(
                        code=fn_code,
                        df_in=parent_dfs[0],
                        fn_name_hint=transform.get("function_name"),
                    )
                elif kind == "python_merge":
                    merge_code = transform.get("merge_code")
                    if not isinstance(merge_code, str) or not merge_code.strip():
                        merge_code = _read_text_file(transform.get("merge_path"))
                    out_df = _exec_merge(code=merge_code, parents=parent_dfs)
                elif kind == "sql_query":
                    sql_code = transform.get("sql_query_code")
                    if not isinstance(sql_code, str) or not sql_code.strip():
                        sql_code = _read_text_file(transform.get("sql_query_path"))
                    out_df = _exec_sql_query(sql_code=sql_code, sql_url=sql_url)
                if not isinstance(out_df, pd.DataFrame):
                    rehydrate_stats["transform_failures"] += 1
                    continue
                entry_obj["data"] = out_df
                shape, cols, schema, schema_hash, fingerprint = (
                    _pipeline_studio_dataset_meta(out_df)
                )
                entry_obj["shape"] = shape
                entry_obj["columns"] = cols
                entry_obj["schema"] = schema
                entry_obj["schema_hash"] = schema_hash
                entry_obj["fingerprint"] = fingerprint
                datasets_map[did] = entry_obj
                rehydrate_stats["transforms_run"] += 1

            return datasets_map, rehydrate_stats

        missing_data = any(
            _coerce_entry_df(entry_obj) is None for entry_obj in datasets.values()
        )
        if bool(rehydrate) and (metadata_only or missing_data):
            datasets, rehydrate_stats = _rehydrate_datasets(datasets)

        prev = st.session_state.get("team_state", {})
        prev = prev if isinstance(prev, dict) else {}
        new_team_state = dict(prev)
        new_team_state["datasets"] = datasets
        data_ids = [
            did
            for did, entry_obj in datasets.items()
            if _coerce_entry_df(entry_obj) is not None
        ]
        if active_id and active_id in datasets:
            if data_ids and active_id not in data_ids:
                new_team_state["active_dataset_id"] = data_ids[0]
            else:
                new_team_state["active_dataset_id"] = active_id
        else:
            new_team_state["active_dataset_id"] = (
                data_ids[0] if data_ids else next(iter(datasets.keys()), None)
            )
        st.session_state["team_state"] = new_team_state

        # Restore code drafts (merge).
        drafts = manifest.get("code_drafts_by_fingerprint")
        drafts = drafts if isinstance(drafts, dict) else {}
        if drafts:
            try:
                store = _load_pipeline_studio_code_drafts_store()
                by_fp = store.get("by_fingerprint")
                by_fp = by_fp if isinstance(by_fp, dict) else {}
                for fp, rec in drafts.items():
                    if isinstance(fp, str) and fp and isinstance(rec, dict):
                        by_fp[fp] = rec
                store["by_fingerprint"] = by_fp
                st.session_state["pipeline_studio_code_drafts_store"] = store
                _save_pipeline_studio_code_drafts_store(store)
            except Exception:
                pass

        # Restore registry records (merge).
        reg_by_ph = manifest.get("registry_by_pipeline_hash")
        reg_by_ph = reg_by_ph if isinstance(reg_by_ph, dict) else {}
        if reg_by_ph:
            try:
                store = _load_pipeline_studio_pipeline_registry_store()
                by_ph = store.get("by_pipeline_hash")
                by_ph = by_ph if isinstance(by_ph, dict) else {}
                for ph, rec in reg_by_ph.items():
                    if isinstance(ph, str) and ph and isinstance(rec, dict):
                        by_ph[ph] = rec
                store["by_pipeline_hash"] = by_ph
                st.session_state["pipeline_studio_pipeline_registry_store"] = store
                _save_pipeline_studio_pipeline_registry_store(store)
            except Exception:
                pass

        # Restore flow layout records (best effort).
        layout_by_ph = manifest.get("flow_layout_by_pipeline_hash")
        layout_by_ph = layout_by_ph if isinstance(layout_by_ph, dict) else {}
        if layout_by_ph:
            for ph, layout in layout_by_ph.items():
                if not isinstance(ph, str) or not ph or not isinstance(layout, dict):
                    continue
                _update_persisted_pipeline_studio_flow_layout(
                    pipeline_hash=ph,
                    positions=layout.get("positions"),
                    hidden_ids=layout.get("hidden_ids"),
                )

        # Recompute registry from loaded datasets (keeps prior UI state if present).
        try:
            pipelines_new = _pipeline_studio_build_pipelines_from_team_state(
                new_team_state
            )
            _update_pipeline_registry_store_for_pipelines(
                pipelines=pipelines_new, datasets=datasets
            )
        except Exception:
            pass

        # Reset canvas caches so the Visual Editor rehydrates.
        st.session_state.pop("pipeline_studio_flow_state", None)
        st.session_state.pop("pipeline_studio_flow_signature", None)
        st.session_state.pop("pipeline_studio_flow_positions", None)
        st.session_state.pop("pipeline_studio_flow_hidden_ids", None)
        st.session_state.pop("pipeline_studio_flow_layout_sig", None)
        st.session_state["pipeline_studio_flow_force_layout"] = True
        st.session_state["pipeline_studio_flow_fit_view_pending"] = True
        st.session_state["pipeline_studio_flow_ts"] = int(_time.time() * 1000)

        try:
            _pipeline_studio_update_project_manifest(
                project_dir=project_dir, updates={"last_opened_ts": _time.time()}
            )
        except Exception:
            pass

        return {
            "loaded_datasets": len(datasets),
            "active_dataset_id": new_team_state.get("active_dataset_id"),
            "data_mode": "metadata_only" if metadata_only else "full",
            "rehydrate_stats": rehydrate_stats,
            "missing_sources": missing_sources,
            "missing_files": missing_files,
            "data_files_loaded": loaded_data_files,
        }
    except Exception as e:
        return {"error": str(e)}


def _pipeline_studio_dataset_meta(
    data: object,
) -> tuple[
    list[int] | None,
    list[str] | None,
    list[dict[str, str]] | None,
    str | None,
    str | None,
]:
    """
    Best-effort dataset metadata matching the supervisor's dataset registry conventions.
    Returns (shape, cols, schema, schema_hash, fingerprint).
    """
    try:
        import hashlib
        import pandas as _pd
        from pandas.util import hash_pandas_object

        if not isinstance(data, _pd.DataFrame):
            return None, None, None, None, None

        shape = [int(data.shape[0]), int(data.shape[1])]
        cols = [str(c) for c in list(data.columns)]
        cols = cols[:200] if cols else None

        col_order = sorted([str(c) for c in list(data.columns)])
        schema = [
            {"name": c, "dtype": str(data[c].dtype) if c in data.columns else ""}
            for c in col_order[:200]
        ]
        schema_str = "|".join(f"{r['name']}:{r['dtype']}" for r in schema)
        schema_hash = (
            hashlib.sha256(schema_str.encode("utf-8")).hexdigest()
            if schema_str
            else None
        )

        df_sample = data.reindex(columns=col_order).head(2000).reset_index(drop=True)
        try:
            row_hashes = hash_pandas_object(df_sample, index=False).values
            fingerprint = hashlib.sha256(row_hashes.tobytes()).hexdigest()
        except Exception:
            snap = df_sample.to_json(orient="split", date_format="iso")
            fingerprint = hashlib.sha256(snap.encode("utf-8")).hexdigest()

        return shape, cols, schema, schema_hash, fingerprint
    except Exception:
        return None, None, None, None, None


def _pipeline_studio_register_dataset(
    *,
    team_state: dict,
    data: pd.DataFrame,
    stage: str,
    label: str,
    created_by: str,
    provenance: dict[str, object],
    parent_id: str | None = None,
    parent_ids: list[str] | None = None,
    make_active: bool = True,
) -> tuple[dict, str]:
    import time
    from datetime import datetime, timezone

    team_state = team_state if isinstance(team_state, dict) else {}
    datasets = team_state.get("datasets")
    datasets = datasets if isinstance(datasets, dict) else {}

    stage = (stage or "custom").strip() or "custom"
    did = f"{stage}_{uuid.uuid4().hex[:8]}"

    shape, cols, schema, schema_hash, fingerprint = _pipeline_studio_dataset_meta(data)
    ts = time.time()
    normalized_parents: list[str] = []
    if isinstance(parent_ids, list):
        normalized_parents.extend(
            [str(p) for p in parent_ids if isinstance(p, str) and p]
        )
    if isinstance(parent_id, str) and parent_id and parent_id not in normalized_parents:
        normalized_parents.insert(0, parent_id)
    normalized_parents = [p for p in normalized_parents if p]
    parent_id = normalized_parents[0] if normalized_parents else parent_id

    datasets = {
        **datasets,
        did: {
            "id": did,
            "label": label or did,
            "stage": stage,
            "data": data,
            "shape": shape,
            "columns": cols,
            "schema": schema,
            "schema_hash": schema_hash,
            "fingerprint": fingerprint,
            "created_ts": ts,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": created_by,
            "provenance": provenance or {},
            "parent_id": parent_id,
            "parent_ids": normalized_parents,
        },
    }
    active_id = did if make_active else team_state.get("active_dataset_id")
    out_state = {**team_state, "datasets": datasets, "active_dataset_id": active_id}
    return out_state, did


def _load_pipeline_studio_pipeline_registry_store() -> dict:
    """
    Load a small, file-backed semantic graph store for Pipeline Studio (best effort).
    The registry is intended to persist a lightweight DAG (no DataFrames) across sessions.
    """
    loaded_flag = "_pipeline_studio_pipeline_registry_store_loaded"
    if bool(st.session_state.get(loaded_flag)):
        store = st.session_state.get("pipeline_studio_pipeline_registry_store")
        return store if isinstance(store, dict) else {}

    store: dict = {
        "version": PIPELINE_STUDIO_PIPELINE_REGISTRY_VERSION,
        "path": PIPELINE_STUDIO_PIPELINE_REGISTRY_PATH,
        "by_pipeline_hash": {},
    }
    try:
        out_dir = os.path.dirname(PIPELINE_STUDIO_PIPELINE_REGISTRY_PATH) or "."
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(PIPELINE_STUDIO_PIPELINE_REGISTRY_PATH):
            with open(
                PIPELINE_STUDIO_PIPELINE_REGISTRY_PATH, "r", encoding="utf-8"
            ) as f:
                data = json.load(f)
            if isinstance(data, dict):
                if isinstance(data.get("by_pipeline_hash"), dict):
                    store.update(
                        {
                            "version": int(
                                data.get("version")
                                or PIPELINE_STUDIO_PIPELINE_REGISTRY_VERSION
                            ),
                            "by_pipeline_hash": data.get("by_pipeline_hash") or {},
                        }
                    )
                else:
                    # legacy: direct mapping {pipeline_hash: registry_record}
                    store["by_pipeline_hash"] = data
    except Exception:
        pass

    st.session_state["pipeline_studio_pipeline_registry_store"] = store
    st.session_state[loaded_flag] = True
    return store


def _save_pipeline_studio_pipeline_registry_store(store: dict) -> None:
    try:
        import tempfile
        import time

        if not isinstance(store, dict):
            return
        by_ph = store.get("by_pipeline_hash")
        by_ph = by_ph if isinstance(by_ph, dict) else {}

        if len(by_ph) > int(PIPELINE_STUDIO_PIPELINE_REGISTRY_MAX_ITEMS):
            items: list[tuple[float, str]] = []
            for ph, rec in by_ph.items():
                ts = 0.0
                if isinstance(rec, dict):
                    try:
                        ts = float(
                            rec.get("updated_ts") or rec.get("created_ts") or 0.0
                        )
                    except Exception:
                        ts = 0.0
                items.append((ts, str(ph)))
            items.sort(reverse=True)
            keep = {
                ph
                for _ts, ph in items[: int(PIPELINE_STUDIO_PIPELINE_REGISTRY_MAX_ITEMS)]
            }
            by_ph = {ph: by_ph[ph] for ph in keep if ph in by_ph}
            store["by_pipeline_hash"] = by_ph

        store["version"] = PIPELINE_STUDIO_PIPELINE_REGISTRY_VERSION
        store["updated_ts"] = time.time()
        store["path"] = PIPELINE_STUDIO_PIPELINE_REGISTRY_PATH

        out_dir = os.path.dirname(PIPELINE_STUDIO_PIPELINE_REGISTRY_PATH) or "."
        os.makedirs(out_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix="._pipeline_registry_", suffix=".json", dir=out_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(store, f, indent=2, default=str)
            os.replace(tmp_path, PIPELINE_STUDIO_PIPELINE_REGISTRY_PATH)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
    except Exception:
        pass


def _get_persisted_pipeline_registry(*, pipeline_hash: str) -> dict:
    try:
        if not isinstance(pipeline_hash, str) or not pipeline_hash:
            return {}
        store = _load_pipeline_studio_pipeline_registry_store()
        by_ph = store.get("by_pipeline_hash")
        by_ph = by_ph if isinstance(by_ph, dict) else {}
        rec = by_ph.get(pipeline_hash)
        return rec if isinstance(rec, dict) else {}
    except Exception:
        return {}


def _pipeline_studio_registry_signature(record: dict) -> str:
    import hashlib

    payload = dict(record) if isinstance(record, dict) else {}
    payload.pop("signature", None)
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _pipeline_studio_get_registry_ui(
    *, pipeline_hash: str
) -> tuple[set[str], set[str]]:
    try:
        rec = _get_persisted_pipeline_registry(pipeline_hash=pipeline_hash)
        ui = rec.get("ui") if isinstance(rec.get("ui"), dict) else {}
        hidden_ids = ui.get("hidden_ids")
        deleted_ids = ui.get("deleted_ids")
        hidden_set = {
            str(x)
            for x in (hidden_ids if isinstance(hidden_ids, list) else [])
            if str(x).strip()
        }
        deleted_set = {
            str(x)
            for x in (deleted_ids if isinstance(deleted_ids, list) else [])
            if str(x).strip()
        }
        return hidden_set, deleted_set
    except Exception:
        return set(), set()


def _pipeline_studio_set_registry_ui(
    *,
    pipeline_hash: str,
    hidden_ids: set[str] | list[str],
    deleted_ids: set[str] | list[str],
) -> None:
    try:
        import time

        pipeline_hash = pipeline_hash.strip() if isinstance(pipeline_hash, str) else ""
        if not pipeline_hash:
            return

        store = _load_pipeline_studio_pipeline_registry_store()
        by_ph = store.get("by_pipeline_hash")
        by_ph = by_ph if isinstance(by_ph, dict) else {}
        rec = by_ph.get(pipeline_hash)
        if not isinstance(rec, dict):
            return

        hidden_set = {
            str(x)
            for x in (hidden_ids if isinstance(hidden_ids, (list, set, tuple)) else [])
            if str(x).strip()
        }
        deleted_set = {
            str(x)
            for x in (
                deleted_ids if isinstance(deleted_ids, (list, set, tuple)) else []
            )
            if str(x).strip()
        }

        rec = dict(rec)
        rec["ui"] = {
            "hidden_ids": sorted(hidden_set),
            "deleted_ids": sorted(deleted_set),
        }
        rec["updated_ts"] = time.time()
        rec["signature"] = _pipeline_studio_registry_signature(rec)
        by_ph = dict(by_ph)
        by_ph[pipeline_hash] = rec
        store["by_pipeline_hash"] = by_ph
        st.session_state["pipeline_studio_pipeline_registry_store"] = store
        _save_pipeline_studio_pipeline_registry_store(store)
    except Exception:
        pass


def _build_pipeline_registry_record(
    *,
    pipeline: dict,
    datasets: dict,
    artifacts_by_dataset_id: dict | None,
    ui: dict | None = None,
) -> dict:
    import time
    import hashlib

    pipeline = pipeline if isinstance(pipeline, dict) else {}
    datasets = datasets if isinstance(datasets, dict) else {}
    artifacts_by_dataset_id = (
        artifacts_by_dataset_id if isinstance(artifacts_by_dataset_id, dict) else {}
    )

    pipeline_hash = pipeline.get("pipeline_hash")
    pipeline_hash = (
        pipeline_hash if isinstance(pipeline_hash, str) and pipeline_hash else None
    )

    lineage = pipeline.get("lineage")
    lineage = lineage if isinstance(lineage, list) else []
    lineage_ids = [
        str(x.get("id"))
        for x in lineage
        if isinstance(x, dict) and isinstance(x.get("id"), str) and x.get("id")
    ]

    def _sanitize_transform(transform: dict) -> dict:
        transform = transform if isinstance(transform, dict) else {}
        kind = transform.get("kind")
        kind = str(kind) if isinstance(kind, str) and kind else ""
        out: dict = {"kind": kind} if kind else {}
        for k in (
            "code_sha256",
            "sql_sha256",
            "sql_database_function_sha256",
            "function_name",
            "function_path",
            "merge_strategy",
            "supersedes_dataset_id",
            "run_id",
            "model_uri",
            "model_id",
        ):
            v = transform.get(k)
            if isinstance(v, (str, int, float, bool)) and str(v).strip():
                out[k] = v

        # Keep optional code payloads, but cap size so the registry stays lightweight.
        max_chars = 12_000
        for code_key in ("function_code", "sql_query_code", "merge_code"):
            code = transform.get(code_key)
            if not isinstance(code, str):
                continue
            code = code.strip()
            if not code:
                continue
            if len(code) > max_chars:
                code = code[:max_chars].rstrip() + "\n\n# ... truncated ..."
            out[code_key] = code
        return out

    def _sanitize_provenance(prov: dict) -> dict:
        prov = prov if isinstance(prov, dict) else {}
        out: dict = {}
        for k in (
            "source_type",
            "source",
            "source_label",
            "original_name",
            "sha256",
        ):
            v = prov.get(k)
            if isinstance(v, (str, int, float, bool)) and str(v).strip():
                out[k] = v
        transform = prov.get("transform")
        if isinstance(transform, dict):
            out["transform"] = _sanitize_transform(transform)
        return out

    nodes: dict[str, dict] = {}
    edges: list[dict[str, str]] = []
    for did in lineage_ids:
        entry = datasets.get(did)
        entry = entry if isinstance(entry, dict) else {}

        parent_ids = entry.get("parent_ids")
        parent_ids = parent_ids if isinstance(parent_ids, list) else []
        parent_ids_clean = [str(p) for p in parent_ids if isinstance(p, str) and p]
        for pid in parent_ids_clean:
            edges.append({"source": pid, "target": did})

        shape = entry.get("shape")
        shape_clean = None
        if isinstance(shape, (list, tuple)) and len(shape) == 2:
            try:
                shape_clean = [int(shape[0]), int(shape[1])]
            except Exception:
                shape_clean = None

        artifacts = artifacts_by_dataset_id.get(did)
        artifacts = artifacts if isinstance(artifacts, dict) else {}
        artifacts_slim: dict[str, dict] = {}
        for k, rec in artifacts.items():
            if not isinstance(k, str) or not k:
                continue
            if not isinstance(rec, dict):
                continue
            slim = {}
            if "turn_idx" in rec:
                slim["turn_idx"] = rec.get("turn_idx")
            if "created_ts" in rec:
                slim["created_ts"] = rec.get("created_ts")
            if "updated_ts" in rec:
                slim["updated_ts"] = rec.get("updated_ts")
            if slim:
                artifacts_slim[k] = slim

        prov = entry.get("provenance")
        prov = prov if isinstance(prov, dict) else {}

        nodes[did] = {
            "id": did,
            "label": entry.get("label"),
            "stage": entry.get("stage"),
            "shape": shape_clean,
            "schema_hash": entry.get("schema_hash"),
            "fingerprint": entry.get("fingerprint"),
            "parent_ids": parent_ids_clean,
            "created_ts": entry.get("created_ts"),
            "created_at": entry.get("created_at"),
            "created_by": entry.get("created_by"),
            "provenance": _sanitize_provenance(prov),
            "artifacts": artifacts_slim,
        }

    ui = ui if isinstance(ui, dict) else {}
    hidden_ids = ui.get("hidden_ids")
    deleted_ids = ui.get("deleted_ids")
    hidden_clean = sorted(
        {
            str(x)
            for x in (hidden_ids if isinstance(hidden_ids, list) else [])
            if str(x).strip()
        }
    )
    deleted_clean = sorted(
        {
            str(x)
            for x in (deleted_ids if isinstance(deleted_ids, list) else [])
            if str(x).strip()
        }
    )

    record: dict = {
        "pipeline_hash": pipeline_hash,
        "target": pipeline.get("target"),
        "target_dataset_id": pipeline.get("target_dataset_id"),
        "active_dataset_id": pipeline.get("active_dataset_id"),
        "model_dataset_id": pipeline.get("model_dataset_id"),
        "inputs": pipeline.get("inputs")
        if isinstance(pipeline.get("inputs"), list)
        else [],
        "lineage": [
            {k: v for k, v in (x or {}).items() if k != "script"}
            for x in lineage
            if isinstance(x, dict)
        ],
        "nodes": nodes,
        "edges": edges,
        "ui": {
            "hidden_ids": hidden_clean,
            "deleted_ids": deleted_clean,
        },
        "updated_ts": time.time(),
    }
    sig = hashlib.sha1(
        json.dumps(record, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    record["signature"] = sig
    return record


def _update_pipeline_registry_store_for_pipelines(
    *, pipelines: dict, datasets: dict
) -> None:
    """
    Persist a lightweight semantic registry keyed by pipeline_hash for each pipeline snapshot.
    """
    try:
        import time

        pipelines = pipelines if isinstance(pipelines, dict) else {}
        datasets = datasets if isinstance(datasets, dict) else {}

        artifacts_by_dataset_id = st.session_state.get("pipeline_studio_artifacts")
        artifacts_by_dataset_id = (
            artifacts_by_dataset_id if isinstance(artifacts_by_dataset_id, dict) else {}
        )

        store = _load_pipeline_studio_pipeline_registry_store()
        by_ph = store.get("by_pipeline_hash")
        by_ph = by_ph if isinstance(by_ph, dict) else {}

        changed = False
        for _name, pipe in pipelines.items():
            pipe = pipe if isinstance(pipe, dict) else {}
            if not pipe.get("lineage"):
                continue
            pipeline_hash = pipe.get("pipeline_hash")
            pipeline_hash = (
                pipeline_hash
                if isinstance(pipeline_hash, str) and pipeline_hash.strip()
                else None
            )
            if not pipeline_hash:
                continue

            prev = by_ph.get(pipeline_hash)
            prev = prev if isinstance(prev, dict) else {}
            prev_ui = prev.get("ui") if isinstance(prev.get("ui"), dict) else None
            rec = _build_pipeline_registry_record(
                pipeline=pipe,
                datasets=datasets,
                artifacts_by_dataset_id=artifacts_by_dataset_id,
                ui=prev_ui,
            )
            if prev.get("signature") == rec.get("signature"):
                continue
            if "created_ts" not in prev:
                rec["created_ts"] = rec.get("updated_ts") or time.time()
            else:
                rec["created_ts"] = prev.get("created_ts")
            by_ph[pipeline_hash] = rec
            changed = True

        if changed:
            store["by_pipeline_hash"] = by_ph
            store["updated_ts"] = time.time()
            st.session_state["pipeline_studio_pipeline_registry_store"] = store
            _save_pipeline_studio_pipeline_registry_store(store)
    except Exception:
        pass


def _load_pipeline_studio_artifact_store() -> dict:
    """
    Load a small, file-backed artifact index so Pipeline Studio can restore charts/EDA/model pointers
    across Streamlit sessions (best effort).
    """
    loaded_flag = "_pipeline_studio_artifact_store_loaded"
    if bool(st.session_state.get(loaded_flag)):
        store = st.session_state.get("pipeline_studio_artifact_store")
        return store if isinstance(store, dict) else {}

    store: dict = {
        "version": PIPELINE_STUDIO_ARTIFACT_STORE_VERSION,
        "path": PIPELINE_STUDIO_ARTIFACT_STORE_PATH,
        "by_fingerprint": {},
    }
    try:
        # Ensure the folder exists even before the first write so it's discoverable on disk.
        out_dir = os.path.dirname(PIPELINE_STUDIO_ARTIFACT_STORE_PATH) or "."
        os.makedirs(out_dir, exist_ok=True)

        source_path = None
        if os.path.exists(PIPELINE_STUDIO_ARTIFACT_STORE_PATH):
            source_path = PIPELINE_STUDIO_ARTIFACT_STORE_PATH
        elif os.path.exists(PIPELINE_STUDIO_ARTIFACT_STORE_LEGACY_PATH):
            source_path = PIPELINE_STUDIO_ARTIFACT_STORE_LEGACY_PATH
        if source_path:
            with open(source_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                # v1 format
                if isinstance(data.get("by_fingerprint"), dict):
                    store.update(
                        {
                            "version": int(
                                data.get("version")
                                or PIPELINE_STUDIO_ARTIFACT_STORE_VERSION
                            ),
                            "by_fingerprint": data.get("by_fingerprint") or {},
                        }
                    )
                # legacy format: direct mapping {fingerprint: artifacts_record}
                else:
                    store["by_fingerprint"] = data
            # If we loaded from legacy, migrate to the new location best-effort.
            if (
                source_path == PIPELINE_STUDIO_ARTIFACT_STORE_LEGACY_PATH
                and not os.path.exists(PIPELINE_STUDIO_ARTIFACT_STORE_PATH)
            ):
                _save_pipeline_studio_artifact_store(store)
    except Exception:
        pass

    st.session_state["pipeline_studio_artifact_store"] = store
    st.session_state[loaded_flag] = True
    return store


def _save_pipeline_studio_artifact_store(store: dict) -> None:
    try:
        import tempfile
        import time

        if not isinstance(store, dict):
            return
        by_fp = store.get("by_fingerprint")
        by_fp = by_fp if isinstance(by_fp, dict) else {}

        # Enforce a small cap to avoid unbounded growth.
        if len(by_fp) > int(PIPELINE_STUDIO_ARTIFACT_STORE_MAX_ITEMS):
            items: list[tuple[float, str]] = []
            for fp, rec in by_fp.items():
                ts = 0.0
                if isinstance(rec, dict):
                    try:
                        ts = float(
                            rec.get("updated_ts") or rec.get("created_ts") or 0.0
                        )
                    except Exception:
                        ts = 0.0
                items.append((ts, fp))
            items.sort(reverse=True)
            keep = {
                fp for _ts, fp in items[: int(PIPELINE_STUDIO_ARTIFACT_STORE_MAX_ITEMS)]
            }
            by_fp = {fp: by_fp[fp] for fp in keep if fp in by_fp}
            store["by_fingerprint"] = by_fp

        store["version"] = PIPELINE_STUDIO_ARTIFACT_STORE_VERSION
        store["updated_ts"] = time.time()
        store["path"] = PIPELINE_STUDIO_ARTIFACT_STORE_PATH

        out_dir = os.path.dirname(PIPELINE_STUDIO_ARTIFACT_STORE_PATH) or "."
        os.makedirs(out_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix="._pipeline_studio_artifacts_", suffix=".json", dir=out_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(store, f, indent=2, default=str)
            os.replace(tmp_path, PIPELINE_STUDIO_ARTIFACT_STORE_PATH)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
    except Exception:
        pass


def _update_pipeline_studio_artifact_store_for_dataset(
    dataset_id: str, artifacts: dict
) -> None:
    """
    Merge artifacts for a dataset into the persisted store, keyed by dataset fingerprint when available.
    """
    try:
        import time

        if not isinstance(dataset_id, str) or not dataset_id:
            return
        if not isinstance(artifacts, dict) or not artifacts:
            return

        team_state = st.session_state.get("team_state", {})
        team_state = team_state if isinstance(team_state, dict) else {}
        datasets = team_state.get("datasets")
        datasets = datasets if isinstance(datasets, dict) else {}
        entry = datasets.get(dataset_id)
        entry = entry if isinstance(entry, dict) else {}

        fingerprint = entry.get("fingerprint")
        fingerprint = (
            fingerprint if isinstance(fingerprint, str) and fingerprint else None
        )
        if not fingerprint:
            return

        store = _load_pipeline_studio_artifact_store()
        by_fp = store.get("by_fingerprint")
        by_fp = by_fp if isinstance(by_fp, dict) else {}
        rec = by_fp.get(fingerprint) if isinstance(by_fp.get(fingerprint), dict) else {}
        rec_art = rec.get("artifacts") if isinstance(rec.get("artifacts"), dict) else {}
        rec_art.update(artifacts)
        rec.update(
            {
                "fingerprint": fingerprint,
                "schema_hash": entry.get("schema_hash")
                if isinstance(entry.get("schema_hash"), str)
                else rec.get("schema_hash"),
                "stage": entry.get("stage")
                if isinstance(entry.get("stage"), str)
                else rec.get("stage"),
                "label": entry.get("label")
                if isinstance(entry.get("label"), str)
                else rec.get("label"),
                "last_dataset_id": dataset_id,
                "updated_ts": time.time(),
                "artifacts": rec_art,
            }
        )
        by_fp[fingerprint] = rec
        store["by_fingerprint"] = by_fp
        st.session_state["pipeline_studio_artifact_store"] = store
        _save_pipeline_studio_artifact_store(store)
    except Exception:
        pass


def _coerce_dataset_entry_df(entry: dict) -> pd.DataFrame | None:
    entry = entry if isinstance(entry, dict) else {}
    data = entry.get("data")
    try:
        if isinstance(data, pd.DataFrame):
            return data
    except Exception:
        pass
    try:
        if isinstance(data, dict):
            return pd.DataFrame.from_dict(data)
        if isinstance(data, list):
            return pd.DataFrame(data)
    except Exception:
        return None
    return None


def _load_pipeline_studio_dataset_store() -> dict:
    loaded_flag = "_pipeline_studio_dataset_store_loaded"
    if bool(st.session_state.get(loaded_flag)):
        store = st.session_state.get("pipeline_studio_dataset_store")
        return store if isinstance(store, dict) else {}

    store: dict = {
        "version": PIPELINE_STUDIO_DATASET_STORE_VERSION,
        "path": PIPELINE_STUDIO_DATASET_STORE_PATH,
        "by_dataset_id": {},
        "active_dataset_id": None,
        "locked_node_ids": [],
    }
    try:
        out_dir = os.path.dirname(PIPELINE_STUDIO_DATASET_STORE_PATH) or "."
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(PIPELINE_STUDIO_DATASET_STORE_PATH):
            with open(PIPELINE_STUDIO_DATASET_STORE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                by_id = data.get("by_dataset_id")
                if isinstance(by_id, dict):
                    store.update(
                        {
                            "version": int(
                                data.get("version")
                                or PIPELINE_STUDIO_DATASET_STORE_VERSION
                            ),
                            "by_dataset_id": by_id,
                            "active_dataset_id": data.get("active_dataset_id"),
                            "locked_node_ids": data.get("locked_node_ids") or [],
                        }
                    )
                else:
                    store["by_dataset_id"] = data
    except Exception:
        pass

    st.session_state["pipeline_studio_dataset_store"] = store
    st.session_state[loaded_flag] = True
    return store


def _save_pipeline_studio_dataset_store(store: dict) -> None:
    try:
        import tempfile
        import time

        if not isinstance(store, dict):
            return
        store = _pipeline_studio_prune_dataset_cache(store)

        store["version"] = PIPELINE_STUDIO_DATASET_STORE_VERSION
        store["updated_ts"] = time.time()

        out_dir = os.path.dirname(PIPELINE_STUDIO_DATASET_STORE_PATH) or "."
        os.makedirs(out_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix="._pipeline_studio_dataset_store_", suffix=".json", dir=out_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(store, f, indent=2, default=str)
            os.replace(tmp_path, PIPELINE_STUDIO_DATASET_STORE_PATH)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
    except Exception:
        pass


def _pipeline_studio_prune_dataset_cache(store: dict) -> dict:
    try:
        store = store if isinstance(store, dict) else {}
        by_id = store.get("by_dataset_id")
        by_id = by_id if isinstance(by_id, dict) else {}
        max_items = int(
            st.session_state.get(
                "pipeline_dataset_cache_max_items",
                PIPELINE_STUDIO_DATASET_CACHE_MAX_ITEMS_DEFAULT,
            )
        )
        max_mb = float(
            st.session_state.get(
                "pipeline_dataset_cache_max_mb",
                PIPELINE_STUDIO_DATASET_CACHE_MAX_MB_DEFAULT,
            )
        )
        max_bytes = int(max_mb * 1024 * 1024) if max_mb > 0 else 0
        if max_items <= 0 and max_bytes <= 0:
            return store

        items: list[tuple[float, str, int, str | None]] = []
        for did, rec in by_id.items():
            if not isinstance(did, str) or not did:
                continue
            rec = rec if isinstance(rec, dict) else {}
            ts = 0.0
            try:
                ts = float(
                    rec.get("saved_ts")
                    or rec.get("created_ts")
                    or rec.get("created_at")
                    or 0.0
                )
            except Exception:
                ts = 0.0
            rel_path = rec.get("data_path")
            rel_path = rel_path if isinstance(rel_path, str) and rel_path else None
            abs_path = os.path.join(APP_ROOT, rel_path) if rel_path else None
            size = 0
            if abs_path and os.path.exists(abs_path):
                try:
                    size = int(os.path.getsize(abs_path))
                except Exception:
                    size = 0
            items.append((ts, did, size, abs_path))

        items.sort(reverse=True)
        keep: set[str] = set()
        kept_bytes = 0
        for _ts, did, size, _abs in items:
            if max_items > 0 and len(keep) >= max_items:
                continue
            if max_bytes > 0 and size > 0 and (kept_bytes + size) > max_bytes:
                continue
            keep.add(did)
            kept_bytes += size

        remove_ids = set(by_id.keys()) - keep
        if remove_ids:
            for did in list(remove_ids):
                rec = by_id.pop(did, None)
                rec = rec if isinstance(rec, dict) else {}
                rel_path = rec.get("data_path")
                rel_path = rel_path if isinstance(rel_path, str) and rel_path else None
                if rel_path:
                    abs_path = os.path.join(APP_ROOT, rel_path)
                    try:
                        if os.path.exists(abs_path):
                            os.remove(abs_path)
                    except Exception:
                        pass
            store["by_dataset_id"] = by_id
        return store
    except Exception:
        return store


def _persist_pipeline_studio_dataset_entry(
    *, dataset_id: str, entry: dict, store: dict | None = None
) -> None:
    try:
        if not bool(st.session_state.get("pipeline_dataset_persist_enabled", False)):
            return
        dataset_id = dataset_id.strip() if isinstance(dataset_id, str) else ""
        if not dataset_id:
            return
        entry = entry if isinstance(entry, dict) else {}
        df = _coerce_dataset_entry_df(entry)
        if df is None:
            return

        store = (
            store if isinstance(store, dict) else _load_pipeline_studio_dataset_store()
        )
        by_id = store.get("by_dataset_id")
        by_id = by_id if isinstance(by_id, dict) else {}

        prev = by_id.get(dataset_id)
        prev = prev if isinstance(prev, dict) else {}
        prev_fp = prev.get("fingerprint") or prev.get("schema_hash")
        cur_fp = entry.get("fingerprint") or entry.get("schema_hash")
        if prev_fp and cur_fp and str(prev_fp) == str(cur_fp) and prev.get("data_path"):
            rel_prev = prev.get("data_path")
            rel_prev = rel_prev if isinstance(rel_prev, str) and rel_prev else None
            abs_prev = os.path.join(APP_ROOT, rel_prev) if rel_prev else None
            if abs_prev and os.path.exists(abs_prev):
                return

        os.makedirs(PIPELINE_STUDIO_DATASET_STORE_DIR, exist_ok=True)
        cache_format = (
            st.session_state.get("pipeline_dataset_cache_format") or "parquet"
        )
        cache_format = str(cache_format).strip().lower()

        import tempfile
        import time

        data_format = "pickle"
        rel_path = os.path.join(
            "pipeline_store", "pipeline_datasets", f"{dataset_id}.pkl"
        )
        abs_path = os.path.join(APP_ROOT, rel_path)

        def _write_parquet(path: str) -> bool:
            try:
                for compression in ("zstd", "snappy", "gzip", None):
                    try:
                        df.to_parquet(
                            path,
                            index=False,
                            compression=compression,
                        )
                        return True
                    except Exception:
                        continue
            except Exception:
                return False
            return False

        if cache_format == "parquet":
            rel_path = os.path.join(
                "pipeline_store", "pipeline_datasets", f"{dataset_id}.parquet"
            )
            abs_path = os.path.join(APP_ROOT, rel_path)
            fd, tmp_path = tempfile.mkstemp(
                prefix="._dataset_",
                suffix=".parquet",
                dir=PIPELINE_STUDIO_DATASET_STORE_DIR,
            )
            try:
                os.close(fd)
                if _write_parquet(tmp_path):
                    os.replace(tmp_path, abs_path)
                    data_format = "parquet"
                else:
                    raise ValueError("Parquet write failed")
            except Exception:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
                rel_path = os.path.join(
                    "pipeline_store", "pipeline_datasets", f"{dataset_id}.pkl"
                )
                abs_path = os.path.join(APP_ROOT, rel_path)
                fd, tmp_path = tempfile.mkstemp(
                    prefix="._dataset_",
                    suffix=".pkl",
                    dir=PIPELINE_STUDIO_DATASET_STORE_DIR,
                )
                try:
                    os.close(fd)
                    df.to_pickle(tmp_path)
                    os.replace(tmp_path, abs_path)
                    data_format = "pickle"
                finally:
                    try:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except Exception:
                        pass
            else:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
        else:
            fd, tmp_path = tempfile.mkstemp(
                prefix="._dataset_",
                suffix=".pkl",
                dir=PIPELINE_STUDIO_DATASET_STORE_DIR,
            )
            try:
                os.close(fd)
                df.to_pickle(tmp_path)
                os.replace(tmp_path, abs_path)
            finally:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass

        data_bytes = None
        try:
            if os.path.exists(abs_path):
                data_bytes = int(os.path.getsize(abs_path))
        except Exception:
            data_bytes = None

        meta = {k: v for k, v in entry.items() if k != "data"}
        meta["data_path"] = rel_path
        meta["data_format"] = data_format
        meta["data_bytes"] = data_bytes
        meta["saved_ts"] = time.time()
        by_id[dataset_id] = meta
        store["by_dataset_id"] = by_id
        st.session_state["pipeline_studio_dataset_store"] = store
    except Exception:
        pass


def _persist_pipeline_studio_team_state(*, team_state: dict) -> None:
    try:
        if not bool(st.session_state.get("pipeline_dataset_persist_enabled", False)):
            return
        team_state = team_state if isinstance(team_state, dict) else {}
        datasets = team_state.get("datasets")
        datasets = datasets if isinstance(datasets, dict) else {}
        store = _load_pipeline_studio_dataset_store()
        by_id = store.get("by_dataset_id")
        by_id = by_id if isinstance(by_id, dict) else {}

        for did, entry in datasets.items():
            if not isinstance(did, str) or not did:
                continue
            entry = entry if isinstance(entry, dict) else {}
            entry = {**entry, "id": entry.get("id") or did}
            _persist_pipeline_studio_dataset_entry(
                dataset_id=did, entry=entry, store=store
            )
            by_id = store.get("by_dataset_id")
            by_id = by_id if isinstance(by_id, dict) else {}

        # Remove datasets no longer present (hard delete).
        current_ids = {did for did in datasets.keys() if isinstance(did, str) and did}
        stale_ids = set(by_id.keys()) - current_ids
        if stale_ids:
            for did in list(stale_ids):
                rec = by_id.pop(did, None)
                rec = rec if isinstance(rec, dict) else {}
                rel_path = rec.get("data_path")
                if isinstance(rel_path, str) and rel_path:
                    abs_path = os.path.join(APP_ROOT, rel_path)
                    try:
                        if os.path.exists(abs_path):
                            os.remove(abs_path)
                    except Exception:
                        pass
            store["by_dataset_id"] = by_id

        active_id = team_state.get("active_dataset_id")
        if isinstance(active_id, str) and active_id:
            store["active_dataset_id"] = active_id
        locked_ids = st.session_state.get("pipeline_studio_locked_node_ids")
        if isinstance(locked_ids, list):
            store["locked_node_ids"] = locked_ids

        st.session_state["pipeline_studio_dataset_store"] = store
        _save_pipeline_studio_dataset_store(store)
    except Exception:
        pass


def _maybe_restore_pipeline_studio_datasets() -> None:
    try:
        if not bool(st.session_state.get("pipeline_dataset_restore_enabled", False)):
            return
        restored_flag = "_pipeline_studio_dataset_store_restored"
        if bool(st.session_state.get(restored_flag)):
            return
        store = _load_pipeline_studio_dataset_store()
        by_id = store.get("by_dataset_id")
        by_id = by_id if isinstance(by_id, dict) else {}
        if not by_id:
            return

        team_state = st.session_state.get("team_state", {})
        team_state = team_state if isinstance(team_state, dict) else {}
        datasets = team_state.get("datasets")
        datasets = datasets if isinstance(datasets, dict) else {}
        merged = dict(datasets)
        added = False

        for did, meta in by_id.items():
            if not isinstance(did, str) or not did or did in merged:
                continue
            meta = meta if isinstance(meta, dict) else {}
            rel_path = meta.get("data_path")
            rel_path = rel_path if isinstance(rel_path, str) and rel_path else None
            if not rel_path:
                continue
            abs_path = os.path.join(APP_ROOT, rel_path)
            if not os.path.exists(abs_path):
                continue
            fmt = meta.get("data_format")
            fmt = fmt if isinstance(fmt, str) and fmt else "pickle"
            try:
                if fmt == "parquet":
                    df = pd.read_parquet(abs_path)
                elif fmt == "csv":
                    df = pd.read_csv(abs_path)
                else:
                    df = pd.read_pickle(abs_path)
            except Exception:
                continue
            if not isinstance(df, pd.DataFrame):
                continue
            entry = dict(meta)
            entry["id"] = did
            entry["data"] = df
            merged[did] = entry
            added = True

        if added:
            team_state = dict(team_state)
            team_state["datasets"] = merged
            active_id = team_state.get("active_dataset_id")
            if not isinstance(active_id, str) or active_id not in merged:
                store_active = store.get("active_dataset_id")
                if isinstance(store_active, str) and store_active in merged:
                    team_state["active_dataset_id"] = store_active
            st.session_state["team_state"] = team_state

        locked_ids = store.get("locked_node_ids")
        if isinstance(locked_ids, list):
            st.session_state["pipeline_studio_locked_node_ids"] = locked_ids

        st.session_state[restored_flag] = True
    except Exception:
        pass


def _get_persisted_pipeline_studio_artifacts(*, fingerprint: str) -> dict:
    try:
        if not isinstance(fingerprint, str) or not fingerprint:
            return {}
        store = _load_pipeline_studio_artifact_store()
        by_fp = store.get("by_fingerprint")
        by_fp = by_fp if isinstance(by_fp, dict) else {}
        rec = by_fp.get(fingerprint)
        rec = rec if isinstance(rec, dict) else {}
        artifacts = rec.get("artifacts")
        return artifacts if isinstance(artifacts, dict) else {}
    except Exception:
        return {}


def _load_pipeline_studio_flow_layout_store() -> dict:
    """
    Load a small, file-backed layout store so the Pipeline Studio Visual Editor can restore
    node positions/hidden nodes across Streamlit sessions (best effort).
    """
    loaded_flag = "_pipeline_studio_flow_layout_store_loaded"
    if bool(st.session_state.get(loaded_flag)):
        store = st.session_state.get("pipeline_studio_flow_layout_store")
        return store if isinstance(store, dict) else {}

    store: dict = {
        "version": PIPELINE_STUDIO_FLOW_LAYOUT_VERSION,
        "path": PIPELINE_STUDIO_FLOW_LAYOUT_PATH,
        "by_pipeline_hash": {},
    }
    try:
        out_dir = os.path.dirname(PIPELINE_STUDIO_FLOW_LAYOUT_PATH) or "."
        os.makedirs(out_dir, exist_ok=True)

        if os.path.exists(PIPELINE_STUDIO_FLOW_LAYOUT_PATH):
            with open(PIPELINE_STUDIO_FLOW_LAYOUT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                # v1 format
                if isinstance(data.get("by_pipeline_hash"), dict):
                    store.update(
                        {
                            "version": int(
                                data.get("version")
                                or PIPELINE_STUDIO_FLOW_LAYOUT_VERSION
                            ),
                            "by_pipeline_hash": data.get("by_pipeline_hash") or {},
                        }
                    )
                # legacy: direct mapping {pipeline_hash: layout_record}
                else:
                    store["by_pipeline_hash"] = data
    except Exception:
        pass

    st.session_state["pipeline_studio_flow_layout_store"] = store
    st.session_state[loaded_flag] = True
    return store


def _save_pipeline_studio_flow_layout_store(store: dict) -> None:
    try:
        import tempfile
        import time

        if not isinstance(store, dict):
            return
        by_ph = store.get("by_pipeline_hash")
        by_ph = by_ph if isinstance(by_ph, dict) else {}

        # Enforce a small cap to avoid unbounded growth.
        if len(by_ph) > int(PIPELINE_STUDIO_FLOW_LAYOUT_MAX_ITEMS):
            items: list[tuple[float, str]] = []
            for ph, rec in by_ph.items():
                ts = 0.0
                if isinstance(rec, dict):
                    try:
                        ts = float(
                            rec.get("updated_ts") or rec.get("created_ts") or 0.0
                        )
                    except Exception:
                        ts = 0.0
                items.append((ts, str(ph)))
            items.sort(reverse=True)
            keep = {
                ph for _ts, ph in items[: int(PIPELINE_STUDIO_FLOW_LAYOUT_MAX_ITEMS)]
            }
            by_ph = {ph: by_ph[ph] for ph in keep if ph in by_ph}
            store["by_pipeline_hash"] = by_ph

        store["version"] = PIPELINE_STUDIO_FLOW_LAYOUT_VERSION
        store["updated_ts"] = time.time()
        store["path"] = PIPELINE_STUDIO_FLOW_LAYOUT_PATH

        out_dir = os.path.dirname(PIPELINE_STUDIO_FLOW_LAYOUT_PATH) or "."
        os.makedirs(out_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix="._pipeline_studio_flow_", suffix=".json", dir=out_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(store, f, indent=2, default=str)
            os.replace(tmp_path, PIPELINE_STUDIO_FLOW_LAYOUT_PATH)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
    except Exception:
        pass


def _get_persisted_pipeline_studio_flow_layout(*, pipeline_hash: str) -> dict:
    try:
        if not isinstance(pipeline_hash, str) or not pipeline_hash:
            return {}
        store = _load_pipeline_studio_flow_layout_store()
        by_ph = store.get("by_pipeline_hash")
        by_ph = by_ph if isinstance(by_ph, dict) else {}
        rec = by_ph.get(pipeline_hash)
        return rec if isinstance(rec, dict) else {}
    except Exception:
        return {}


def _delete_persisted_pipeline_studio_flow_layout(*, pipeline_hash: str) -> None:
    try:
        if not isinstance(pipeline_hash, str) or not pipeline_hash:
            return
        store = _load_pipeline_studio_flow_layout_store()
        by_ph = store.get("by_pipeline_hash")
        by_ph = by_ph if isinstance(by_ph, dict) else {}
        if pipeline_hash in by_ph:
            by_ph.pop(pipeline_hash, None)
            store["by_pipeline_hash"] = by_ph
            st.session_state["pipeline_studio_flow_layout_store"] = store
            _save_pipeline_studio_flow_layout_store(store)
    except Exception:
        pass


def _update_persisted_pipeline_studio_flow_layout(
    *, pipeline_hash: str, positions: dict, hidden_ids: list[str]
) -> None:
    try:
        import time

        if not isinstance(pipeline_hash, str) or not pipeline_hash:
            return
        positions = positions if isinstance(positions, dict) else {}
        hidden_ids = hidden_ids if isinstance(hidden_ids, list) else []

        pos_clean: dict[str, dict[str, float]] = {}
        for nid, p in positions.items():
            if not isinstance(nid, str) or not nid:
                continue
            if not isinstance(p, dict):
                continue
            try:
                x = float(p.get("x", 0.0))
                y = float(p.get("y", 0.0))
            except Exception:
                continue
            pos_clean[nid] = {"x": x, "y": y}

        hid_clean = [str(x) for x in hidden_ids if isinstance(x, str) and x]

        store = _load_pipeline_studio_flow_layout_store()
        by_ph = store.get("by_pipeline_hash")
        by_ph = by_ph if isinstance(by_ph, dict) else {}
        rec = by_ph.get(pipeline_hash)
        rec = rec if isinstance(rec, dict) else {}
        rec.update(
            {
                "pipeline_hash": pipeline_hash,
                "positions": pos_clean,
                "hidden_ids": hid_clean,
                "updated_ts": time.time(),
            }
        )
        if "created_ts" not in rec:
            rec["created_ts"] = rec.get("updated_ts")
        by_ph[pipeline_hash] = rec
        store["by_pipeline_hash"] = by_ph
        st.session_state["pipeline_studio_flow_layout_store"] = store
        _save_pipeline_studio_flow_layout_store(store)
    except Exception:
        pass


def _pipeline_studio_transform_code_snippet(
    transform: dict,
) -> tuple[str | None, str | None, str, str]:
    """
    Best-effort extraction of runnable code for a pipeline transform.
    Returns (title, code_text, code_lang, kind).
    """
    transform = transform if isinstance(transform, dict) else {}
    kind = str(transform.get("kind") or "")
    code_lang = "python"
    title = None
    code_text = None

    if kind == "python_function":
        title = "Transform function (Python)"
        code_text = transform.get("function_code")
    elif kind == "sql_query":
        title = "SQL query"
        code_text = transform.get("sql_query_code")
        code_lang = "sql"
    elif kind == "python_merge":
        title = "Merge code (Python)"
        code_text = transform.get("merge_code")
    elif kind == "mlflow_predict":
        run_id = transform.get("run_id")
        run_id = run_id.strip() if isinstance(run_id, str) else ""
        title = "Prediction (MLflow) snippet"
        code_text = (
            "\n".join(
                [
                    "import pandas as pd",
                    "import mlflow",
                    "",
                    f"model_uri = 'runs:/{run_id}/model'",
                    "model = mlflow.pyfunc.load_model(model_uri)",
                    "preds = model.predict(df)",
                    "df_preds = preds if isinstance(preds, pd.DataFrame) else pd.DataFrame(preds)",
                ]
            ).strip()
            + "\n"
        )
    elif kind == "h2o_predict":
        model_id = transform.get("model_id")
        model_id = model_id.strip() if isinstance(model_id, str) else ""
        title = "Prediction (H2O) snippet"
        code_text = (
            "\n".join(
                [
                    "import h2o",
                    "",
                    "h2o.init()",
                    f"model = h2o.get_model('{model_id}')",
                    "frame = h2o.H2OFrame(df)",
                    "preds = model.predict(frame)",
                    "df_preds = preds.as_data_frame(use_pandas=True)",
                ]
            ).strip()
            + "\n"
        )

    code_text = code_text if isinstance(code_text, str) and code_text.strip() else None
    return title, code_text, code_lang, kind


def _pipeline_studio_chat_context(*, include_code: bool = False) -> str:
    """
    Return a small context block describing the current Pipeline Studio selection, suitable
    for appending to a chat prompt (kept lightweight; best effort).
    """
    try:
        sel = st.session_state.get("pipeline_studio_node_id")
        sel = sel.strip() if isinstance(sel, str) and sel.strip() else None
        team_state = st.session_state.get("team_state", {})
        team_state = team_state if isinstance(team_state, dict) else {}
        datasets = team_state.get("datasets")
        datasets = datasets if isinstance(datasets, dict) else {}
        entry = datasets.get(sel) if sel else {}
        entry = entry if isinstance(entry, dict) else {}
        if not datasets and not entry:
            return ""

        label = entry.get("label") if isinstance(entry.get("label"), str) else None
        stage = entry.get("stage") if isinstance(entry.get("stage"), str) else None
        shape = entry.get("shape")
        shape_str = None
        if isinstance(shape, (list, tuple)) and len(shape) == 2:
            shape_str = f"{shape[0]}x{shape[1]}"

        prov = (
            entry.get("provenance") if isinstance(entry.get("provenance"), dict) else {}
        )
        transform = (
            prov.get("transform") if isinstance(prov.get("transform"), dict) else {}
        )
        _title, code_text, code_lang, kind = _pipeline_studio_transform_code_snippet(
            transform
        )

        preserve_all = bool(st.session_state.get("pipeline_preserve_all_nodes", True))
        preserve_studio = bool(
            st.session_state.get("pipeline_preserve_studio_nodes", True)
        )
        locked_ids = st.session_state.get("pipeline_studio_locked_node_ids") or []
        locked_ids = [str(x) for x in locked_ids if isinstance(x, str) and x.strip()]

        lines = [
            "[Pipeline Studio context]",
            f"pipeline_target: {st.session_state.get('pipeline_studio_target')}",
            f"active_dataset_id: {team_state.get('active_dataset_id')}",
            f"selected_node_id: {sel}" if sel else None,
            f"label: {label}" if label else None,
            f"stage: {stage}" if stage else None,
            f"transform_kind: {kind}" if kind else None,
            f"shape: {shape_str}" if shape_str else None,
            "preserve_nodes: all" if preserve_all else None,
            "preserve_nodes: studio_only"
            if (not preserve_all and preserve_studio)
            else None,
            f"locked_node_ids: {', '.join(locked_ids[:10])}" if locked_ids else None,
        ]
        lines = [x for x in lines if isinstance(x, str) and x.strip()]

        if len(datasets) > 1:
            ordered = sorted(
                datasets.items(),
                key=lambda kv: float(kv[1].get("created_ts") or 0.0)
                if isinstance(kv[1], dict)
                else 0.0,
                reverse=True,
            )
            lines.append(f"available_datasets: {len(datasets)}")
            for did, ent in ordered[:8]:
                if not isinstance(ent, dict):
                    continue
                dlabel = ent.get("label") if isinstance(ent.get("label"), str) else did
                dstage = ent.get("stage") if isinstance(ent.get("stage"), str) else None
                dshape = ent.get("shape")
                dshape_str = None
                if isinstance(dshape, (list, tuple)) and len(dshape) == 2:
                    dshape_str = f"{dshape[0]}x{dshape[1]}"
                meta_bits = [x for x in (dstage, dshape_str) if x]
                meta_txt = f" ({', '.join(meta_bits)})" if meta_bits else ""
                lines.append(f"- {did}: {dlabel}{meta_txt}")

        if include_code and isinstance(code_text, str) and code_text.strip():
            max_chars = 4000
            trimmed = (
                code_text
                if len(code_text) <= max_chars
                else (code_text[:max_chars] + "\n# ...trimmed...\n")
            )
            lines.append("code_snippet:")
            lines.append(f"```{code_lang}\n{trimmed}\n```")

        return "\n".join(lines).strip()
    except Exception:
        return ""


def _strip_ui_marker_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Remove Streamlit-only marker messages (e.g., DETAILS_INDEX) from the LLM context.
    These are useful for UI rendering, but can confuse the supervisor/router when memory is off.
    """
    cleaned: list[BaseMessage] = []
    for m in messages or []:
        content = getattr(m, "content", "")
        if isinstance(content, str) and content.startswith(UI_DETAIL_MARKER_PREFIX):
            continue
        cleaned.append(m)
    return cleaned


def _redact_sqlalchemy_url(url: str) -> str:
    """
    Render a SQLAlchemy URL while hiding any password component.
    """
    if not isinstance(url, str) or not url.strip():
        return ""
    try:
        parsed = sql.engine.make_url(url.strip())
        return parsed.render_as_string(hide_password=True)
    except Exception:
        return url.strip()


def _queue_active_dataset_override(value: str | None) -> None:
    st.session_state[ACTIVE_DATASET_OVERRIDE_PENDING_KEY] = value or ""
    st.session_state[ACTIVE_DATASET_OVERRIDE_SYNC_FLAG] = True


def _safe_rerun() -> None:
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass


def _keep_pipeline_studio_open() -> None:
    st.session_state["pipeline_studio_manual_node_open"] = True
    st.session_state["pipeline_studio_view_pending"] = "Visual Editor"
    if _pipeline_studio_is_docked():
        st.session_state["pipeline_studio_drawer_open"] = True
    else:
        st.session_state["pipeline_studio_open_requested"] = True


def _pipeline_studio_reset_project(*, clear_cache: bool, add_memory: bool) -> None:
    docked = bool(st.session_state.get("pipeline_studio_docked", False))
    target_label = st.session_state.get("pipeline_studio_target")
    show_hidden = bool(st.session_state.get("pipeline_studio_show_hidden", False))
    show_deleted = bool(st.session_state.get("pipeline_studio_show_deleted", False))

    for key in list(st.session_state.keys()):
        if key.startswith("pipeline_studio_"):
            st.session_state.pop(key, None)

    st.session_state["pipeline_studio_docked"] = docked
    if isinstance(target_label, str) and target_label:
        st.session_state["pipeline_studio_target"] = target_label
    st.session_state["pipeline_studio_show_hidden"] = show_hidden
    st.session_state["pipeline_studio_show_deleted"] = show_deleted

    st.session_state["team_state"] = {}
    st.session_state["details"] = []
    st.session_state["chat_history"] = []
    st.session_state["selected_data_raw"] = None
    st.session_state["selected_data_provenance"] = None
    st.session_state["last_pipeline_persist_dir"] = None

    _queue_active_dataset_override("")
    st.session_state.pop("chat_dataset_selector", None)

    msgs = StreamlitChatMessageHistory(key="supervisor_ds_msgs")
    msgs.clear()
    msgs.add_ai_message("How can the data science team help today?")
    st.session_state["thread_id"] = str(uuid.uuid4())
    st.session_state["checkpointer"] = get_checkpointer() if add_memory else None

    dataset_store = {
        "version": PIPELINE_STUDIO_DATASET_STORE_VERSION,
        "path": PIPELINE_STUDIO_DATASET_STORE_PATH,
        "by_dataset_id": {},
        "active_dataset_id": None,
        "locked_node_ids": [],
    }
    registry_store = {
        "version": PIPELINE_STUDIO_PIPELINE_REGISTRY_VERSION,
        "path": PIPELINE_STUDIO_PIPELINE_REGISTRY_PATH,
        "by_pipeline_hash": {},
    }
    artifact_store = {
        "version": PIPELINE_STUDIO_ARTIFACT_STORE_VERSION,
        "path": PIPELINE_STUDIO_ARTIFACT_STORE_PATH,
        "by_fingerprint": {},
    }
    flow_store = {
        "version": PIPELINE_STUDIO_FLOW_LAYOUT_VERSION,
        "path": PIPELINE_STUDIO_FLOW_LAYOUT_PATH,
        "by_pipeline_hash": {},
    }
    drafts_store = {
        "version": PIPELINE_STUDIO_CODE_DRAFTS_VERSION,
        "path": PIPELINE_STUDIO_CODE_DRAFTS_PATH,
        "by_fingerprint": {},
    }

    st.session_state["pipeline_studio_dataset_store"] = dataset_store
    st.session_state["_pipeline_studio_dataset_store_loaded"] = True
    st.session_state["_pipeline_studio_dataset_store_restored"] = True
    st.session_state["pipeline_studio_pipeline_registry_store"] = registry_store
    st.session_state["_pipeline_studio_pipeline_registry_store_loaded"] = True
    st.session_state["pipeline_studio_artifact_store"] = artifact_store
    st.session_state["_pipeline_studio_artifact_store_loaded"] = True
    st.session_state["pipeline_studio_flow_layout_store"] = flow_store
    st.session_state["_pipeline_studio_flow_layout_store_loaded"] = True
    st.session_state["pipeline_studio_code_drafts_store"] = drafts_store
    st.session_state["_pipeline_studio_code_drafts_store_loaded"] = True
    st.session_state["pipeline_studio_artifacts"] = {}
    st.session_state["pipeline_studio_semantic_graph"] = None
    st.session_state["pipeline_studio_locked_node_ids"] = []

    if clear_cache:
        _save_pipeline_studio_dataset_store(dataset_store)
        _save_pipeline_studio_pipeline_registry_store(registry_store)
        _save_pipeline_studio_artifact_store(artifact_store)
        _save_pipeline_studio_flow_layout_store(flow_store)
        _save_pipeline_studio_code_drafts_store(drafts_store)
        try:
            if os.path.isdir(PIPELINE_STUDIO_DATASET_STORE_DIR):
                shutil.rmtree(PIPELINE_STUDIO_DATASET_STORE_DIR)
        except Exception:
            pass

    st.session_state["pipeline_studio_project_notice"] = (
        "Started a new project (current session cleared)."
    )
    _keep_pipeline_studio_open()


def _pipeline_studio_delete_project(*, dir_name: str) -> None:
    dir_name = dir_name.strip() if isinstance(dir_name, str) else ""
    if not dir_name:
        st.session_state["pipeline_studio_project_notice"] = (
            "Error: Select a project to delete."
        )
        return
    project_dir = os.path.join(PIPELINE_STUDIO_PROJECTS_DIR, dir_name)
    if not os.path.isdir(project_dir):
        st.session_state["pipeline_studio_project_notice"] = (
            "Error: Project not found on disk."
        )
        return
    try:
        shutil.rmtree(project_dir)
    except Exception as exc:
        st.session_state["pipeline_studio_project_notice"] = f"Error: {exc}"
        return
    st.session_state["pipeline_studio_project_notice"] = (
        f"Deleted project `{dir_name}`."
    )
    st.session_state.pop("pipeline_studio_project_select", None)
    _keep_pipeline_studio_open()


def _pipeline_studio_factory_reset(add_memory: bool | None = None) -> None:
    add_memory = bool(add_memory)
    docked = bool(st.session_state.get("pipeline_studio_docked", False))
    target_label = st.session_state.get("pipeline_studio_target")
    show_hidden = bool(st.session_state.get("pipeline_studio_show_hidden", False))
    show_deleted = bool(st.session_state.get("pipeline_studio_show_deleted", False))

    for key in list(st.session_state.keys()):
        if key.startswith("pipeline_studio_") or key.startswith("_pipeline_studio_"):
            st.session_state.pop(key, None)

    st.session_state["pipeline_studio_docked"] = docked
    if isinstance(target_label, str) and target_label:
        st.session_state["pipeline_studio_target"] = target_label
    st.session_state["pipeline_studio_show_hidden"] = show_hidden
    st.session_state["pipeline_studio_show_deleted"] = show_deleted

    st.session_state["team_state"] = {}
    st.session_state["details"] = []
    st.session_state["chat_history"] = []
    st.session_state["selected_data_raw"] = None
    st.session_state["selected_data_provenance"] = None
    st.session_state["last_pipeline_persist_dir"] = None

    _queue_active_dataset_override("")
    st.session_state.pop("chat_dataset_selector", None)

    msgs = StreamlitChatMessageHistory(key="supervisor_ds_msgs")
    msgs.clear()
    msgs.add_ai_message("How can the data science team help today?")
    st.session_state["thread_id"] = str(uuid.uuid4())
    st.session_state["checkpointer"] = get_checkpointer() if add_memory else None

    pipeline_store_root = os.path.join(APP_ROOT, "pipeline_store")
    pipeline_reports_root = os.path.join(APP_ROOT, "pipeline_reports")
    for path in (pipeline_store_root, pipeline_reports_root):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
        except Exception:
            pass

    st.session_state["pipeline_studio_project_notice"] = (
        "Factory reset complete. Removed `pipeline_store/` and `pipeline_reports/`."
    )
    _keep_pipeline_studio_open()


def _parse_merge_shortcut(prompt: str, *, datasets: dict) -> dict | None:
    if not isinstance(prompt, str) or not prompt.strip():
        return None
    text = prompt.strip()
    text_lower = text.lower()
    if not any(w in text_lower for w in ("merge", "join", "concat", "append", "union")):
        return None

    dataset_ids = [did for did in datasets.keys() if isinstance(did, str) and did]
    if not dataset_ids:
        return None

    def _match_dataset_ids() -> list[str]:
        matches: list[str] = []
        for did in dataset_ids:
            if did.lower() in text_lower:
                matches.append(did)
        if len(matches) >= 2:
            return matches

        for did in dataset_ids:
            entry = datasets.get(did)
            entry = entry if isinstance(entry, dict) else {}
            prov = entry.get("provenance")
            prov = prov if isinstance(prov, dict) else {}
            candidates = [
                str(entry.get("label") or ""),
                str(prov.get("original_name") or ""),
                str(prov.get("source") or ""),
            ]
            for cand in candidates:
                cand_clean = cand.strip()
                if not cand_clean or len(cand_clean) < 3:
                    continue
                if cand_clean.lower() in text_lower:
                    matches.append(did)
                    break
        return list(dict.fromkeys(matches))

    selected = _match_dataset_ids()
    if len(selected) < 2:
        return None

    op = "join"
    if any(w in text_lower for w in ("concat", "append", "union")):
        op = "concat"

    join_keys: list[str] = []
    if op == "join":
        m = re.search(r"(?i)\\bon\\s+([a-zA-Z0-9_,\\s]+)", text)
        if m:
            raw = m.group(1)
            parts = re.split(r"[,&]|\\band\\b", raw, flags=re.IGNORECASE)
            join_keys = [p.strip() for p in parts if p.strip()]
    return {
        "dataset_ids": selected[:4],
        "operation": op,
        "on": join_keys[:3],
    }


def _extract_db_target_from_prompt(
    prompt: str,
) -> tuple[str | None, tuple[int, int] | None]:
    """
    Extract a DB target (SQLAlchemy URL or sqlite file path) from a natural-language prompt.
    Returns (target, (start, end)) where span refers to the target substring in prompt.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        return None, None

    # Quoted targets after connect/use/switch (allows spaces)
    m = re.search(
        r"(?is)\b(?:connect|switch|use)\b[^\n]*?(?:\bto\b\s*)?(?P<q>['\"`])(?P<target>.+?)(?P=q)",
        prompt,
    )
    if m:
        return (m.group("target") or "").strip(), m.span("target")

    # SQLAlchemy URLs (no spaces)
    m = re.search(
        r"(?is)\b(?P<target>(?:sqlite|postgresql|mysql|mssql|oracle|duckdb|snowflake)(?:\+\w+)?://[^\s'\"`]+)",
        prompt,
    )
    if m:
        return (m.group("target") or "").strip(), m.span("target")

    # SQLite file path mention (no spaces unless quoted above)
    m = re.search(
        r"(?is)\b(?P<target>(?:[a-zA-Z]:)?[\w./~\\:-]+\.(?:db|sqlite|sqlite3))\b",
        prompt,
    )
    if m:
        return (m.group("target") or "").strip(), m.span("target")

    return None, None


def _resolve_existing_sqlite_path(target: str, *, cwd: str) -> str | None:
    """
    Resolve a sqlite file target to an existing file path (best effort).
    Does not create new files.
    """
    if not isinstance(target, str) or not target.strip():
        return None
    raw = target.strip().strip("'\"`")
    raw = os.path.expandvars(os.path.expanduser(raw))
    if os.path.isabs(raw):
        return os.path.abspath(raw) if os.path.exists(raw) else None

    # Relative path as-given
    rel = os.path.abspath(os.path.join(cwd, raw))
    if os.path.exists(rel):
        return rel

    # Basename search in common dirs
    base = os.path.basename(raw)
    if base == raw:
        for d in (cwd, os.path.join(cwd, "data"), os.path.join(cwd, "temp")):
            cand = os.path.abspath(os.path.join(d, base))
            if os.path.exists(cand):
                return cand

    return None


def _normalize_db_target_to_sqlalchemy_url(
    target: str, *, cwd: str
) -> tuple[str | None, str | None]:
    """
    Convert a DB target (SQLAlchemy URL or sqlite file path) into a SQLAlchemy URL.
    Returns (sql_url, error).
    """
    if not isinstance(target, str) or not target.strip():
        return None, "No database target found."

    t = target.strip().strip("'\"`")

    # Treat anything that looks like a URL as a SQLAlchemy URL.
    if "://" in t or t.lower().startswith("sqlite:"):
        try:
            parsed = sql.engine.make_url(t)
        except Exception as e:
            return None, f"Invalid SQLAlchemy URL: {e}"

        # Avoid accidentally creating new sqlite files for typos.
        if str(parsed.drivername or "").startswith(
            "sqlite"
        ) and parsed.database not in (None, "", ":memory:"):
            # database may be absolute or relative; interpret relative to cwd
            db_path = str(parsed.database)
            if not os.path.isabs(db_path):
                db_path = os.path.abspath(os.path.join(cwd, db_path))
            if not os.path.exists(db_path):
                return None, f"SQLite file not found: {db_path}"
        return t, None

    # Otherwise assume it's a local sqlite file path.
    resolved = _resolve_existing_sqlite_path(t, cwd=cwd)
    if not resolved:
        return None, f"SQLite file not found: {t}"

    # sqlite:///relative or sqlite:////absolute
    posix_path = resolved.replace("\\", "/")
    return f"sqlite:///{posix_path}", None


def _parse_db_connect_command(prompt: str) -> dict | None:
    """
    Parse DB connect/disconnect intent from the user prompt.
    Returns a dict with:
      - action: 'connect'|'disconnect'
      - sql_url: str (for connect)
      - display_prompt: str (redacted)
      - team_prompt: str (with connect clause removed when possible)
      - message: str (assistant confirmation)
    """
    if not isinstance(prompt, str) or not prompt.strip():
        return None

    p = prompt.strip()
    low = p.lower()

    # Disconnect/reset
    if re.search(r"(?i)\b(disconnect|reset|clear)\b.*\b(db|database|sql)\b", p):
        display_prompt = p
        message = f"Disconnected database. Reset SQL URL to `{DEFAULT_SQL_URL}`."
        return {
            "action": "disconnect",
            "sql_url": DEFAULT_SQL_URL,
            "display_prompt": display_prompt,
            "team_prompt": "",
            "message": message,
        }

    # Connect/switch/use with a target
    if not re.search(r"(?i)\b(connect|switch|use)\b", p):
        return None

    target, span = _extract_db_target_from_prompt(p)
    if not target:
        return None

    sql_url, err = _normalize_db_target_to_sqlalchemy_url(target, cwd=os.getcwd())
    if err:
        display_prompt = p
        return {
            "action": "connect",
            "sql_url": None,
            "display_prompt": display_prompt,
            "team_prompt": "",
            "message": f"Could not connect database: {err}",
            "error": err,
        }

    redacted_url = _redact_sqlalchemy_url(sql_url or "")

    # Redact secrets in what we store/show for the user message.
    display_prompt = p
    if span and redacted_url:
        try:
            display_prompt = p[: span[0]] + redacted_url + p[span[1] :]
        except Exception:
            display_prompt = p

    # Best-effort: strip the connect clause so the team focuses on the actual task.
    team_prompt = ""
    if span:
        tail_start = span[1]
        # If the target was quoted, skip the closing quote/backtick.
        if tail_start < len(p) and p[tail_start] in ("'", '"', "`"):
            tail_start += 1
        tail = p[tail_start:].strip()
        tail = tail.lstrip(" .,:;-\n\t")
        for kw in ("and", "then", "also"):
            if tail.lower().startswith(kw + " "):
                tail = tail[len(kw) + 1 :].lstrip()
                break
        team_prompt = tail.strip()

    message = f"Connected database. Set SQL URL to `{redacted_url}`."
    return {
        "action": "connect",
        "sql_url": sql_url,
        "display_prompt": display_prompt,
        "team_prompt": team_prompt,
        "message": message,
    }


def _replace_last_human_message(
    messages: list[BaseMessage], new_content: str
) -> list[BaseMessage]:
    """
    Return a shallow-copied messages list with the last HumanMessage content replaced.
    """
    if not messages:
        return []
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        m = out[i]
        if isinstance(m, HumanMessage):
            out[i] = HumanMessage(content=new_content, id=getattr(m, "id", None))
            break
    return out


def _apply_streamlit_plot_style(fig):
    """
    Streamlit dark theme + Plotly can yield black-on-black traces when the
    figure template/colors aren't set. Normalize styling for readability.
    """
    if fig is None:
        return None

    try:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            legend=dict(font=dict(color="white")),
        )
    except Exception:
        return fig

    colorway = list(getattr(pc.qualitative, "Plotly", [])) or [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
    ]

    def _is_black(val) -> bool:
        if val is None:
            return True
        if isinstance(val, str):
            v = val.strip().lower()
            return v in ("black", "#000", "#000000", "rgb(0,0,0)", "rgba(0,0,0,1)")
        return False

    try:
        for i, tr in enumerate(fig.data or []):
            c = colorway[i % len(colorway)]
            # Line-like traces
            if hasattr(tr, "line"):
                line_color = getattr(getattr(tr, "line", None), "color", None)
                if _is_black(line_color):
                    tr.update(line=dict(color=c))
            # Marker-like traces
            if hasattr(tr, "marker"):
                marker_color = getattr(getattr(tr, "marker", None), "color", None)
                if _is_black(marker_color):
                    tr.update(marker=dict(color=c))
            # Fill (e.g., area)
            fillcolor = getattr(tr, "fillcolor", None)
            if _is_black(fillcolor):
                tr.update(fillcolor=c)
    except Exception:
        pass

    return fig


def persist_pipeline_artifacts(
    pipeline: dict,
    *,
    base_dir: str | None,
    overwrite: bool = False,
    include_sql: bool = True,
    sql_query: str | None = None,
    sql_executor: str | None = None,
) -> dict:
    """
    Persist the pipeline spec + repro script to disk (best effort).

    Returns metadata:
      - persisted_dir, spec_path, script_path
      - sql_query_path, sql_executor_path (optional)
      - error (if any)
    """
    try:
        if not isinstance(pipeline, dict) or not pipeline.get("lineage"):
            return {}

        base_dir = (base_dir or "").strip()
        if not base_dir:
            return {}

        base_dir = os.path.abspath(os.path.expanduser(base_dir))
        if os.path.exists(base_dir) and not os.path.isdir(base_dir):
            return {
                "error": f"Pipeline persist path exists and is not a directory: {base_dir}"
            }

        pipeline_hash = pipeline.get("pipeline_hash")
        model_id = pipeline.get("model_dataset_id") or pipeline.get("active_dataset_id")
        suffix = (
            str(pipeline_hash)
            if isinstance(pipeline_hash, str) and pipeline_hash
            else str(model_id or "pipeline")
        )
        persisted_dir = os.path.join(base_dir, f"pipeline_{suffix}")
        os.makedirs(persisted_dir, exist_ok=True)

        # Prepare file payloads
        spec = dict(pipeline)
        script = spec.pop("script", "") or ""
        try:
            from datetime import datetime, timezone

            spec["saved_at"] = datetime.now(timezone.utc).isoformat()
        except Exception:
            pass

        spec_path = os.path.join(persisted_dir, "pipeline_spec.json")
        script_path = os.path.join(persisted_dir, "pipeline_repro.py")

        if overwrite or not os.path.exists(spec_path):
            with open(spec_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(spec, indent=2))

        if isinstance(script, str) and script.strip():
            if overwrite or not os.path.exists(script_path):
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(script)

        out = {
            "persisted_dir": persisted_dir,
            "spec_path": spec_path,
            "script_path": script_path if os.path.exists(script_path) else None,
        }

        if include_sql and (sql_query or sql_executor):
            sql_dir = os.path.join(persisted_dir, "sql")
            os.makedirs(sql_dir, exist_ok=True)
            if sql_query:
                sql_query_path = os.path.join(sql_dir, "query.sql")
                if overwrite or not os.path.exists(sql_query_path):
                    with open(sql_query_path, "w", encoding="utf-8") as f:
                        f.write(str(sql_query))
                out["sql_query_path"] = sql_query_path
            if sql_executor:
                sql_executor_path = os.path.join(sql_dir, "sql_executor.py")
                if overwrite or not os.path.exists(sql_executor_path):
                    with open(sql_executor_path, "w", encoding="utf-8") as f:
                        f.write(str(sql_executor))
                out["sql_executor_path"] = sql_executor_path

        return out
    except Exception as e:
        return {"error": str(e)}


@st.cache_resource(show_spinner=False)
def get_checkpointer():
    """Cache the LangGraph MemorySaver checkpointer."""
    return MemorySaver()


with st.expander(
    "I’m a full data science copilot. Load data, wrangle/clean, run EDA, visualize, engineer features, and train/evaluate models (H2O/MLflow). Try these on the sample Telco churn data:"
):
    st.markdown(
        """
        #### Data loading / discovery
        - What files are in `./data`? List only CSVs.
        - Load `data/churn_data.csv` and show the first 5 rows.
        - Load multiple files: `Load data/bike_sales_data.csv and data/bike_model_specs.csv`

        #### Wrangling / cleaning
        - Clean the churn data; fix TotalCharges numeric conversion and missing values; summarize changes.
        - Standardize column names and impute missing TotalCharges = MonthlyCharges * tenure when possible.

        #### EDA
        - Describe the dataset and give key stats for MonthlyCharges and tenure.
        - Show missingness summary and top 5 correlations with `Churn`.
        - Generate a Sweetviz report with `Churn` as the target.

        #### Visualization
        - Make a violin+box plot of MonthlyCharges by Churn.
        - Plot tenure distribution split by InternetService.
        
        #### SQL (if a DB is connected)
        - Connect to a DB: `connect to data/northwind.db`
        - Show the tables in the connected database (do not call other agents).
        - Write SQL to count customers by Contract type and execute it.      

        #### Feature engineering
        - Create model-ready features for churn (encode categoricals, handle totals/averages).
        
        #### Machine Learning (Logs To MLFlow by default)
        - Train an H2O AutoML classifier on Churn with max runtime 30 seconds and report leaderboard.
        - Score using an H2O model id (current cluster): ``predict with model `XGBoost_grid_...` on the dataset``
        
        #### MLflow
        - What runs are available in the H2O AutoML experiment?
        - Score using MLflow (latest run): `predict using mlflow on the dataset`
        - Score using MLflow (specific run): `predict using mlflow run <run_id> on the dataset`
        """
    )

drawer_placeholder = st.empty()


# ---------------- Pipeline Studio helpers ----------------
def _pipeline_studio_is_docked() -> bool:
    return bool(st.session_state.get("pipeline_studio_docked", False))


def _request_open_pipeline_studio() -> None:
    if _pipeline_studio_is_docked():
        st.session_state["pipeline_studio_drawer_open"] = True
    else:
        st.session_state["pipeline_studio_open_requested"] = True


def _pipeline_studio_target_key_from_label(label: str | None) -> str:
    mapping = {
        "Model (latest feature)": "model",
        "Active dataset": "active",
        "Latest dataset": "latest",
    }
    return mapping.get(label or "", "model")


def _pipeline_studio_ui_state(
    *, team_state: dict | None = None
) -> tuple[list[str], set[str], set[str], str, list[str]]:
    team_state = (
        team_state
        if isinstance(team_state, dict)
        else st.session_state.get("team_state", {})
    )
    team_state = team_state if isinstance(team_state, dict) else {}
    datasets = team_state.get("datasets")
    datasets = datasets if isinstance(datasets, dict) else {}
    active_id = team_state.get("active_dataset_id")
    active_id = active_id if isinstance(active_id, str) else None
    target_label = st.session_state.get("pipeline_studio_target")
    target_key = _pipeline_studio_target_key_from_label(
        target_label if isinstance(target_label, str) else None
    )
    pipe = build_pipeline_snapshot(
        datasets, active_dataset_id=active_id, target=target_key
    )
    lineage = pipe.get("lineage") if isinstance(pipe, dict) else None
    lineage = lineage if isinstance(lineage, list) else []
    node_ids = [
        str(x.get("id")) for x in lineage if isinstance(x, dict) and x.get("id")
    ]
    pipeline_hash = pipe.get("pipeline_hash") if isinstance(pipe, dict) else None
    pipeline_hash = (
        pipeline_hash.strip()
        if isinstance(pipeline_hash, str) and pipeline_hash.strip()
        else ""
    )
    hidden_ids, deleted_ids = (
        _pipeline_studio_get_registry_ui(pipeline_hash=pipeline_hash)
        if pipeline_hash
        else (set(), set())
    )
    hidden_ids = set(hidden_ids)
    deleted_ids = set(deleted_ids)
    visible_ids = [
        did for did in node_ids if did not in hidden_ids and did not in deleted_ids
    ]
    if not visible_ids and datasets:
        visible_ids = [
            did
            for did in datasets.keys()
            if did not in hidden_ids and did not in deleted_ids
        ]
    return visible_ids, hidden_ids, deleted_ids, pipeline_hash, node_ids


def _pick_latest_dataset_id(
    datasets: dict, candidate_ids: set[str] | list[str]
) -> str | None:
    datasets = datasets if isinstance(datasets, dict) else {}
    best_id = None
    best_ts = -1.0
    for did in list(candidate_ids):
        if not isinstance(did, str) or did not in datasets:
            continue
        entry = datasets.get(did)
        entry = entry if isinstance(entry, dict) else {}
        try:
            ts = float(entry.get("created_ts") or 0.0)
        except Exception:
            ts = 0.0
        if ts >= best_ts:
            best_ts = ts
            best_id = did
    if best_id is None:
        for did in candidate_ids:
            if isinstance(did, str) and did in datasets:
                return did
    return best_id


def _sync_pipeline_targets_after_ui_change() -> None:
    try:
        team_state = st.session_state.get("team_state", {})
        team_state = team_state if isinstance(team_state, dict) else {}
        datasets = team_state.get("datasets")
        datasets = datasets if isinstance(datasets, dict) else {}
        if not datasets:
            return
        visible_ids, hidden_ids, deleted_ids, _p_hash, _node_ids = (
            _pipeline_studio_ui_state(team_state=team_state)
        )
        visible_set = set(visible_ids) if visible_ids else set(datasets.keys())
        if hidden_ids or deleted_ids:
            visible_set = {
                did
                for did in visible_set
                if did not in hidden_ids and did not in deleted_ids
            }
        active_id = team_state.get("active_dataset_id")
        active_id = active_id if isinstance(active_id, str) else None
        if visible_set and (not active_id or active_id not in visible_set):
            new_active = _pick_latest_dataset_id(datasets, visible_set)
            if new_active and new_active != active_id:
                team_state = dict(team_state)
                team_state["active_dataset_id"] = new_active
                st.session_state["team_state"] = team_state
                _persist_pipeline_studio_team_state(team_state=team_state)
        current_sel = st.session_state.get("pipeline_studio_node_id")
        if visible_set and (
            not isinstance(current_sel, str) or current_sel not in visible_set
        ):
            fallback = (
                team_state.get("active_dataset_id")
                if isinstance(team_state.get("active_dataset_id"), str)
                else None
            )
            if fallback and fallback in visible_set:
                st.session_state["pipeline_studio_node_id"] = fallback
        override = st.session_state.get("active_dataset_id_override")
        if isinstance(override, str) and override and override not in visible_set:
            _queue_active_dataset_override("")
    except Exception:
        pass


def _get_query_params_dict() -> dict:
    try:
        qp = st.query_params
        return {k: qp.get(k) for k in qp.keys()}
    except Exception:
        pass
    try:
        params = st.experimental_get_query_params()
        return params if isinstance(params, dict) else {}
    except Exception:
        return {}


def _clear_query_param(param: str) -> None:
    try:
        qp = st.query_params
        if param in qp:
            del qp[param]
        return
    except Exception:
        pass
    try:
        current = st.experimental_get_query_params()
        if isinstance(current, dict) and param in current:
            current.pop(param, None)
        if hasattr(st, "experimental_set_query_params"):
            st.experimental_set_query_params(**(current or {}))
    except Exception:
        try:
            if hasattr(st, "experimental_set_query_params"):
                st.experimental_set_query_params()
        except Exception:
            pass


def _maybe_open_pipeline_studio_from_query() -> None:
    params = _get_query_params_dict()
    value = params.get("open_pipeline_studio")
    if isinstance(value, list):
        value = value[0] if value else ""
    flag = str(value or "").strip().lower()
    if flag and flag not in {"0", "false", "no"}:
        _request_open_pipeline_studio()
        _clear_query_param("open_pipeline_studio")


_maybe_open_pipeline_studio_from_query()


# ---------------- Sidebar ----------------
key_status = None
with st.sidebar:
    if st.button(
        "Pipeline Studio",
        key="pipeline_studio_open_sidebar",
        width="stretch",
        help="Open Pipeline Studio using the selected mode.",
    ):
        _request_open_pipeline_studio()
    st.toggle(
        "Dock Pipeline Studio (inline)",
        value=False,
        key="pipeline_studio_docked",
        help="Docked mode keeps Studio inline; undocked opens a modal.",
    )
    st.divider()

    st.header("LLM")
    llm_provider = st.selectbox(
        "Provider",
        ["OpenAI", "Ollama"],
        index=0,
        key="llm_provider",
        help="Choose OpenAI (cloud) or Ollama (local).",
    )

    ollama_base_url = None
    if llm_provider == "OpenAI":
        openai_key_input = st.text_input(
            "OpenAI API key",
            type="password",
            value=st.session_state.get("OPENAI_API_KEY") or "",
            key="openai_api_key_input",
            help="Required when using OpenAI models.",
        )
        openai_key = (openai_key_input or "").strip()
        st.session_state["OPENAI_API_KEY"] = openai_key

        if openai_key:
            try:
                _ = OpenAI(api_key=openai_key).models.list()
                key_status = "ok"
                st.success("API Key is valid!")
            except Exception as e:
                key_status = "bad"
                st.error(f"Invalid API Key: {e}")
        else:
            st.info(
                "Please enter your OpenAI API key to proceed (or switch to Ollama)."
            )
            st.stop()

        model_choice = st.selectbox(
            "Model",
            [
                "gpt-4.1-mini",
                "gpt-4.1",
                "gpt-4o-mini",
                "gpt-4o",
                "gpt-5-mini",
                "gpt-5.1",
                # "gpt-5.1-codex-mini",
                "gpt-5.2",
            ],
            key="openai_model_choice",
        )
    else:
        if ChatOllama is None:
            st.error(
                "Ollama support requires `langchain-ollama`. Install it with `pip install langchain-ollama`."
            )
            st.stop()

        default_ollama_url = (
            st.session_state.get("ollama_base_url") or "http://localhost:11434"
        )
        default_ollama_model = st.session_state.get("ollama_model") or "llama3.1:8b"
        ollama_base_url = st.text_input(
            "Ollama base URL",
            value=default_ollama_url,
            key="ollama_base_url_input",
            help="Usually `http://localhost:11434`.",
        ).strip()
        ollama_model = st.text_input(
            "Ollama model",
            value=default_ollama_model,
            key="ollama_model_input",
            help="Example: `llama3.1:8b` (run `ollama list` to see what's installed).",
        ).strip()

        st.session_state["ollama_base_url"] = ollama_base_url
        st.session_state["ollama_model"] = ollama_model

        model_choice = ollama_model

        if st.button(
            "Check Ollama connection", width="stretch", key="ollama_check"
        ):
            try:
                from urllib.request import Request, urlopen
                import json as _json

                url = f"{ollama_base_url.rstrip('/')}/api/tags"
                req = Request(url, headers={"Accept": "application/json"})
                with urlopen(req, timeout=3) as resp:
                    payload = _json.loads(resp.read().decode("utf-8", errors="replace"))
                models = [
                    m.get("name")
                    for m in (payload.get("models") or [])
                    if isinstance(m, dict) and isinstance(m.get("name"), str)
                ]
                if models:
                    st.success(f"Connected. Found {len(models)} model(s).")
                    st.write(models[:20])
                else:
                    st.warning(
                        "Connected, but no models were returned. Run `ollama list` to confirm."
                    )
            except Exception as e:
                st.error(f"Could not connect to Ollama at `{ollama_base_url}`: {e}")

        st.caption(
            "Tip: start Ollama with `ollama serve` and pull a model with `ollama pull <model>`."
        )

    # Settings
    st.header("Settings")
    recursion_limit = st.slider("Recursion limit", 4, 20, 10, 1)
    add_memory = st.checkbox("Enable short-term memory", value=True)
    proactive_workflow_mode = st.checkbox(
        "Proactive workflow mode",
        value=False,
        help="When enabled, the supervisor may propose and run a multi-step end-to-end workflow for broad requests (and will ask clarifying questions when needed).",
    )
    st.session_state["proactive_workflow_mode"] = proactive_workflow_mode
    use_llm_intent_parser = st.checkbox(
        "LLM intent parsing",
        value=True,
        help="When enabled, the supervisor uses a lightweight LLM call to classify user intent for routing. Can improve ambiguous requests, but adds latency/cost.",
    )
    st.session_state["use_llm_intent_parser"] = use_llm_intent_parser
    st.markdown("---")
    st.markdown("**Data options**")
    use_sample = st.checkbox("Load sample Telco churn data", value=False)
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    preview_rows = st.number_input("Preview rows", 1, 20, 5)

    st.markdown("**Dataset selection**")
    team_state = st.session_state.get("team_state", {})
    team_state = team_state if isinstance(team_state, dict) else {}
    datasets = team_state.get("datasets")
    datasets = datasets if isinstance(datasets, dict) else {}
    current_active_id = team_state.get("active_dataset_id")
    current_active_id = (
        current_active_id if isinstance(current_active_id, str) else None
    )
    current_active_key = team_state.get("active_data_key")
    visible_ids, ui_hidden_ids, ui_deleted_ids, _p_hash, _node_ids = (
        _pipeline_studio_ui_state(team_state=team_state)
    )
    visible_set = set(visible_ids) if visible_ids else set(datasets.keys())
    if ui_hidden_ids or ui_deleted_ids:
        visible_set = {
            did
            for did in visible_set
            if did not in ui_hidden_ids and did not in ui_deleted_ids
        }
    if visible_set and (not current_active_id or current_active_id not in visible_set):
        new_active = _pick_latest_dataset_id(datasets, visible_set)
        if new_active and new_active != current_active_id:
            team_state = dict(team_state)
            team_state["active_dataset_id"] = new_active
            st.session_state["team_state"] = team_state
            _persist_pipeline_studio_team_state(team_state=team_state)
            current_active_id = new_active

    if (
        current_active_id
        and current_active_id in datasets
        and isinstance(datasets[current_active_id], dict)
    ):
        entry = datasets[current_active_id]
        label = entry.get("label") or current_active_id
        stage = entry.get("stage")
        shape = entry.get("shape")
        meta_bits = []
        if stage:
            meta_bits.append(f"stage={stage}")
        if shape:
            meta_bits.append(f"shape={shape}")
        meta = f" ({', '.join(meta_bits)})" if meta_bits else ""
        st.caption(f"Current active dataset: `{label}` (`{current_active_id}`){meta}")
    elif current_active_key:
        st.caption(f"Current active dataset: `{current_active_key}`")

    if st.session_state.pop(ACTIVE_DATASET_OVERRIDE_SYNC_FLAG, False):
        pending_override = st.session_state.pop(
            ACTIVE_DATASET_OVERRIDE_PENDING_KEY, None
        )
        pending_override = pending_override if pending_override is not None else ""
        st.session_state["active_dataset_id_override"] = pending_override

    if datasets:
        ordered = sorted(
            [(did, ent) for did, ent in datasets.items() if did in visible_set],
            key=lambda kv: float(kv[1].get("created_ts") or 0.0)
            if isinstance(kv[1], dict)
            else 0.0,
            reverse=True,
        )
        options = [""] + [did for did, _ in ordered]
        current_override = st.session_state.get("active_dataset_id_override")
        if current_override and current_override not in options:
            st.session_state["active_dataset_id_override"] = ""

        def _fmt_dataset(did: str) -> str:
            if not did:
                return "Auto (use supervisor active)"
            e = datasets.get(did)
            if not isinstance(e, dict):
                return str(did)
            label = e.get("label") or did
            stage = e.get("stage") or "dataset"
            shape = e.get("shape")
            shape_txt = f" {shape}" if shape else ""
            return f"{stage}: {label}{shape_txt} ({did})"

        st.selectbox(
            "Active dataset (override)",
            options=options,
            format_func=_fmt_dataset,
            help="Overrides which dataset is considered active for downstream steps (EDA/viz/wrangle/clean).",
            key="active_dataset_id_override",
        )
        st.caption(
            "For merges, use Pipeline Studio (create a node with multiple parents)."
        )
    else:
        st.session_state["active_dataset_id_override"] = ""
        st.selectbox(
            "Active dataset (override)",
            options=[""],
            format_func=lambda _k: "Auto (load data to populate datasets)",
            disabled=True,
            key="active_dataset_id_override",
        )

    st.markdown("**Pipeline options**")
    default_pipeline_dir = os.path.join(APP_ROOT, "pipeline_reports", "pipelines")
    st.text_input(
        "Persist pipeline directory (optional)",
        value=default_pipeline_dir,
        key="pipeline_persist_dir",
        help="When enabled, writes `pipeline_spec.json` and `pipeline_repro.py` to this folder for reproducibility.",
    )
    with st.expander("Pipeline behaviors", expanded=False):
        st.checkbox(
            "Auto-save pipeline files",
            value=True,
            key="pipeline_persist_enabled",
            help="Saves the latest pipeline on each new pipeline hash.",
        )
        st.checkbox(
            "Overwrite existing pipeline files",
            value=False,
            key="pipeline_persist_overwrite",
            help="If off, existing files are left untouched.",
        )
        st.checkbox(
            "Also save SQL artifacts",
            value=True,
            key="pipeline_persist_include_sql",
            help="If SQL is generated, also saves `sql/query.sql` and `sql/sql_executor.py` under the pipeline folder.",
        )
        st.checkbox(
            "Preserve all nodes on AI runs",
            value=bool(st.session_state.get("pipeline_preserve_all_nodes", True)),
            key="pipeline_preserve_all_nodes",
            help="When enabled, agent updates never drop existing nodes (append-only graph unless you delete nodes manually).",
        )
        st.checkbox(
            "Preserve Pipeline Studio nodes on AI runs",
            value=bool(st.session_state.get("pipeline_preserve_studio_nodes", True)),
            key="pipeline_preserve_studio_nodes",
            help="Keeps Pipeline Studio-created nodes (manual/edited) and their parents when agent results arrive.",
        )
        persist_enabled = st.checkbox(
            "Persist pipeline nodes to disk",
            value=bool(st.session_state.get("pipeline_dataset_persist_enabled", False)),
            key="pipeline_dataset_persist_enabled",
            help="Stores datasets under `pipeline_store/` for recovery across sessions (Parquet when available, else pickle).",
        )
        st.checkbox(
            "Restore pipeline nodes on start",
            value=bool(st.session_state.get("pipeline_dataset_restore_enabled", False)),
            key="pipeline_dataset_restore_enabled",
            help="Restores persisted datasets into the current session when available.",
            disabled=not bool(persist_enabled),
        )
        cache_format = st.session_state.get("pipeline_dataset_cache_format")
        cache_format = (
            cache_format if cache_format in {"parquet", "pickle"} else "parquet"
        )
        st.selectbox(
            "Dataset cache format",
            options=["parquet", "pickle"],
            index=0 if cache_format == "parquet" else 1,
            key="pipeline_dataset_cache_format",
            help="Parquet is smaller; pickle is fastest but larger.",
            disabled=not bool(persist_enabled),
        )
        st.number_input(
            "Dataset cache max items",
            min_value=0,
            step=1,
            value=int(
                st.session_state.get(
                    "pipeline_dataset_cache_max_items",
                    PIPELINE_STUDIO_DATASET_CACHE_MAX_ITEMS_DEFAULT,
                )
            ),
            key="pipeline_dataset_cache_max_items",
            help="0 disables pruning; older items are removed first.",
            disabled=not bool(persist_enabled),
        )
        st.number_input(
            "Dataset cache max size (MB)",
            min_value=0.0,
            step=50.0,
            value=float(
                st.session_state.get(
                    "pipeline_dataset_cache_max_mb",
                    PIPELINE_STUDIO_DATASET_CACHE_MAX_MB_DEFAULT,
                )
            ),
            key="pipeline_dataset_cache_max_mb",
            help="0 disables pruning; older items are removed first.",
            disabled=not bool(persist_enabled),
        )
    if st.session_state.get("last_pipeline_persist_dir"):
        st.caption(
            f"Last saved pipeline: `{st.session_state.get('last_pipeline_persist_dir')}`"
        )

    st.markdown("**Chat ↔ Pipeline context**")
    st.checkbox(
        "Include Pipeline Studio context in chat",
        value=bool(st.session_state.get("pipeline_chat_context_enabled", True)),
        key="pipeline_chat_context_enabled",
        help="When enabled, the current Pipeline Studio selection is appended (lightly) to your chat prompt.",
    )
    st.checkbox(
        "Include selected node code snippet",
        value=bool(st.session_state.get("pipeline_chat_context_include_code", False)),
        key="pipeline_chat_context_include_code",
        help="If enabled, includes a trimmed code snippet for the selected node in the chat context block.",
    )
    st.checkbox(
        "Use selected Pipeline Studio node for chat",
        value=bool(st.session_state.get("pipeline_use_selected_node_for_chat", True)),
        key="pipeline_use_selected_node_for_chat",
        help="When enabled, chat/AI uses the currently selected Pipeline Studio node as the active dataset.",
    )
    st.checkbox(
        "Sync Pipeline Studio state to AI",
        value=bool(st.session_state.get("pipeline_sync_state_to_agents", True)),
        key="pipeline_sync_state_to_agents",
        help="Keeps the agent's dataset registry aligned with Pipeline Studio (recommended to preserve manual steps).",
    )
    st.markdown("**SQL options**")
    if "sql_url" not in st.session_state:
        st.session_state["sql_url"] = DEFAULT_SQL_URL
    if SQL_URL_INPUT_KEY not in st.session_state:
        st.session_state[SQL_URL_INPUT_KEY] = st.session_state.get(
            "sql_url", DEFAULT_SQL_URL
        )
    if st.session_state.pop(SQL_URL_SYNC_FLAG, False):
        st.session_state[SQL_URL_INPUT_KEY] = st.session_state.get(
            "sql_url", DEFAULT_SQL_URL
        )

    sql_url_input = st.text_input(
        "SQLAlchemy URL (optional)",
        key=SQL_URL_INPUT_KEY,
        help="Tip: you can also type in chat `connect to data/northwind.db` or paste a SQLAlchemy URL (e.g., `postgresql://...`).",
    )
    sql_url_input = (sql_url_input or "").strip()
    st.session_state["sql_url"] = sql_url_input or DEFAULT_SQL_URL

    st.markdown("**MLflow options**")
    # Use separate widget keys so we can normalize/sync into the internal config keys
    # without violating Streamlit's "no session_state mutation after widget instantiation" rule.
    enable_mlflow_logging = st.checkbox(
        "Enable MLflow logging in training",
        value=bool(st.session_state.get("enable_mlflow_logging", True)),
        key="enable_mlflow_logging_input",
    )
    # Prefer a local SQLite backend store to avoid MLflow FileStore deprecation warnings.
    # Note: this changes where runs are tracked; existing `file:.../mlruns` runs won't
    # appear unless you switch the URI back (or migrate).
    default_mlflow_uri = f"sqlite:///{os.path.abspath('mlflow.db')}"
    mlflow_tracking_uri = st.text_input(
        "MLflow tracking URI",
        value=st.session_state.get("mlflow_tracking_uri") or default_mlflow_uri,
        key="mlflow_tracking_uri_input",
    ).strip()
    default_mlflow_artifact_root = os.path.abspath("mlflow_artifacts")
    mlflow_artifact_root = st.text_input(
        "MLflow artifact root (local path)",
        value=st.session_state.get("mlflow_artifact_root")
        or default_mlflow_artifact_root,
        key="mlflow_artifact_root_input",
        help=(
            "Where MLflow stores artifacts (models, tables, plots). "
            "This is used when creating new experiments; existing experiments keep their current artifact location."
        ),
    ).strip()
    mlflow_experiment_name = st.text_input(
        "MLflow experiment name",
        value=st.session_state.get("mlflow_experiment_name") or "H2O AutoML",
        key="mlflow_experiment_name_input",
    ).strip()
    st.session_state["enable_mlflow_logging"] = bool(enable_mlflow_logging)
    st.session_state["mlflow_tracking_uri"] = (
        mlflow_tracking_uri or ""
    ).strip() or None
    st.session_state["mlflow_artifact_root"] = (
        mlflow_artifact_root or ""
    ).strip() or None
    st.session_state["mlflow_experiment_name"] = (
        mlflow_experiment_name or ""
    ).strip() or "H2O AutoML"

    st.markdown("**Debug options**")
    st.checkbox(
        "Verbose console logs",
        value=bool(st.session_state.get("debug_mode", False)),
        key="debug_mode",
        help="Print extra debug info to the terminal to troubleshoot DB connect and multi-file loads.",
    )
    st.checkbox(
        "Show progress in chat",
        value=bool(st.session_state.get("show_progress", True)),
        key="show_progress",
        help="Shows which agent is running while the team works (best effort).",
    )
    st.checkbox(
        "Show live logs while running",
        value=bool(st.session_state.get("show_live_logs", True)),
        key="show_live_logs",
        help="Streams console output into the app during execution (clears after the run finishes).",
    )

    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.session_state.details = []
        st.session_state.team_state = {}
        _queue_active_dataset_override("")
        st.session_state.selected_data_provenance = None
        st.session_state.last_pipeline_persist_dir = None
        msgs = StreamlitChatMessageHistory(key="supervisor_ds_msgs")
        msgs.clear()
        msgs.add_ai_message("How can the data science team help today?")
        st.session_state.thread_id = str(uuid.uuid4())
        # Reset checkpointer when clearing chat
        st.session_state.checkpointer = get_checkpointer() if add_memory else None

# LLM credentials are only required when running chat (Pipeline Studio + previews should still work).
llm_provider_selected = st.session_state.get("llm_provider") or "OpenAI"
resolved_api_key = (st.session_state.get("OPENAI_API_KEY") or "").strip() or None
resolved_ollama_model = (st.session_state.get("ollama_model") or "").strip() or None


def build_team(
    llm_provider: str,
    model_name: str,
    openai_api_key: str | None,
    ollama_base_url: str | None,
    use_memory: bool,
    sql_url: str,
    checkpointer,
    enable_mlflow_logging: bool,
    mlflow_tracking_uri: str | None,
    mlflow_artifact_root: str | None,
    mlflow_experiment_name: str,
    debug_mode: bool = False,
):
    llm_provider = (llm_provider or "OpenAI").strip()
    if llm_provider.lower() == "ollama":
        if ChatOllama is None:
            raise RuntimeError(
                "Ollama provider selected but `langchain-ollama` is not installed."
            )
        kwargs: dict[str, object] = {"model": model_name}
        base_url = (ollama_base_url or "").strip()
        if base_url:
            try:
                sig = inspect.signature(ChatOllama)
                if "base_url" in sig.parameters:
                    kwargs["base_url"] = base_url
                elif "ollama_base_url" in sig.parameters:
                    kwargs["ollama_base_url"] = base_url
            except Exception:
                kwargs["base_url"] = base_url
        llm = ChatOllama(**kwargs)
    else:

        def _openai_requires_responses(model: str | None) -> bool:
            model = model.strip().lower() if isinstance(model, str) else ""
            if not model:
                return False
            if "codex" in model:
                return True
            return model in {"gpt-5.1-codex-mini"}

        llm_kwargs: dict[str, object] = {
            "model": model_name,
            "api_key": openai_api_key,
        }
        if _openai_requires_responses(model_name):
            llm_kwargs["use_responses_api"] = True
            llm_kwargs["output_version"] = "responses/v1"
        llm = ChatOpenAI(**llm_kwargs)
    workflow_planner_agent = WorkflowPlannerAgent(llm)
    data_loader_agent = DataLoaderToolsAgent(
        llm, invoke_react_agent_kwargs={"recursion_limit": 4}
    )
    data_wrangling_agent = DataWranglingAgent(llm, log=False)
    data_cleaning_agent = DataCleaningAgent(llm, log=False)
    eda_tools_agent = EDAToolsAgent(llm, log_tool_calls=True)
    data_visualization_agent = DataVisualizationAgent(
        llm, log=bool(debug_mode)
    )
    # SQL connection is optional; default to in-memory sqlite to satisfy constructor.
    resolved_sql_url = (sql_url or DEFAULT_SQL_URL).strip() or DEFAULT_SQL_URL
    engine_kwargs: dict = {}
    try:
        url_obj = sql.engine.make_url(resolved_sql_url)
        if str(getattr(url_obj, "drivername", "")).startswith("sqlite"):
            # Use check_same_thread=False so the connection can be reused safely in Streamlit threads.
            engine_kwargs["connect_args"] = {"check_same_thread": False}
    except Exception:
        # Best effort: if URL parsing fails, assume sqlite when it looks like sqlite.
        if resolved_sql_url.lower().startswith("sqlite"):
            engine_kwargs["connect_args"] = {"check_same_thread": False}
    conn = sql.create_engine(resolved_sql_url, **engine_kwargs).connect()
    sql_database_agent = SQLDatabaseAgent(llm, connection=conn, log=False)
    feature_engineering_agent = FeatureEngineeringAgent(llm, log=False)
    h2o_ml_agent = H2OMLAgent(
        llm,
        log=False,
        enable_mlflow=enable_mlflow_logging,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_artifact_root=mlflow_artifact_root,
        mlflow_experiment_name=mlflow_experiment_name,
    )
    model_evaluation_agent = ModelEvaluationAgent()
    mlflow_tools_agent = MLflowToolsAgent(
        llm, log_tool_calls=True, mlflow_tracking_uri=mlflow_tracking_uri
    )

    team = make_supervisor_ds_team(
        model=llm,
        workflow_planner_agent=workflow_planner_agent,
        data_loader_agent=data_loader_agent,
        data_wrangling_agent=data_wrangling_agent,
        data_cleaning_agent=data_cleaning_agent,
        eda_tools_agent=eda_tools_agent,
        data_visualization_agent=data_visualization_agent,
        sql_database_agent=sql_database_agent,
        feature_engineering_agent=feature_engineering_agent,
        h2o_ml_agent=h2o_ml_agent,
        mlflow_tools_agent=mlflow_tools_agent,
        model_evaluation_agent=model_evaluation_agent,
        checkpointer=checkpointer if use_memory else None,
    )
    return team


# ---------------- Session state ----------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "details" not in st.session_state:
    st.session_state.details = []
if "team_state" not in st.session_state:
    st.session_state.team_state = {}
if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = get_checkpointer() if add_memory else None
if add_memory and st.session_state.checkpointer is None:
    st.session_state.checkpointer = get_checkpointer()
if not add_memory:
    st.session_state.checkpointer = None

_load_pipeline_studio_artifact_store()
_load_pipeline_studio_dataset_store()
_maybe_restore_pipeline_studio_datasets()

msgs = StreamlitChatMessageHistory(key="supervisor_ds_msgs")
if not msgs.messages:
    msgs.add_ai_message("How can the data science team help today?")


def get_input_data():
    """
    Resolve data_raw based on user selections: uploaded CSV or sample dataset.
    Returns tuple (data_raw_dict, preview_df_or_None, provenance_dict_or_None).
    """
    df = None
    provenance = None
    if uploaded_file is not None:
        try:
            # Persist uploads to disk so the pipeline can be reproduced later.
            raw_bytes = uploaded_file.getvalue()
            import hashlib

            digest = hashlib.sha256(raw_bytes).hexdigest()[:12]
            safe_name = os.path.basename(getattr(uploaded_file, "name", "upload.csv"))
            upload_dir = os.path.join("temp", "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            saved_path = os.path.abspath(
                os.path.join(upload_dir, f"{digest}_{safe_name}")
            )
            if not os.path.exists(saved_path):
                with open(saved_path, "wb") as f:
                    f.write(raw_bytes)

            df = pd.read_csv(saved_path)
            provenance = {
                "source_type": "file",
                "source": saved_path,
                "source_label": "upload",
                "original_name": safe_name,
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            }
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
    elif use_sample:
        sample_path = os.path.join("data", "churn_data.csv")
        if os.path.exists(sample_path):
            try:
                abs_path = os.path.abspath(sample_path)
                df = pd.read_csv(abs_path)
                provenance = {
                    "source_type": "file",
                    "source": abs_path,
                    "source_label": "sample",
                    "original_name": os.path.basename(abs_path),
                }
            except Exception as e:
                st.error(f"Error loading sample data: {e}")
        else:
            st.warning(f"Sample file not found at {sample_path}")

    if df is not None:
        # Best-effort: seed the Pipeline Studio dataset registry so Studio can open
        # immediately after a file upload/sample selection (no chat run required).
        try:
            team_state = st.session_state.get("team_state", {})
            team_state = team_state if isinstance(team_state, dict) else {}
            datasets = team_state.get("datasets")
            datasets = datasets if isinstance(datasets, dict) else {}

            prov = provenance if isinstance(provenance, dict) else {}
            prov_source = (
                prov.get("source") if isinstance(prov.get("source"), str) else None
            )
            prov_sha = (
                prov.get("sha256") if isinstance(prov.get("sha256"), str) else None
            )
            seed_key = (prov_sha or prov_source or "").strip() or None
            prev_seed_key = st.session_state.get("pipeline_studio_seed_key")
            prev_seed_key = (
                prev_seed_key.strip() if isinstance(prev_seed_key, str) else None
            )
            is_new_selection = bool(seed_key and seed_key != prev_seed_key)

            match_id = None
            for did, ent in datasets.items():
                if not isinstance(did, str) or not did or not isinstance(ent, dict):
                    continue
                ent_prov = ent.get("provenance")
                ent_prov = ent_prov if isinstance(ent_prov, dict) else {}
                if prov_sha and ent_prov.get("sha256") == prov_sha:
                    match_id = did
                    break
                if prov_source and ent_prov.get("source") == prov_source:
                    match_id = did
                    break

            if seed_key:
                # Only force-select the seeded dataset when the user *changes* the selected input.
                # Otherwise, leave the active dataset alone so Pipeline Studio edits/runs aren't overwritten.
                if is_new_selection:
                    st.session_state["pipeline_studio_seed_key"] = seed_key
                    if match_id and team_state.get("active_dataset_id") != match_id:
                        st.session_state["team_state"] = {
                            **team_state,
                            "active_dataset_id": match_id,
                        }
                        _persist_pipeline_studio_team_state(
                            team_state=st.session_state.get("team_state", {})
                        )

                # If the selected dataset isn't present in the registry (e.g., after clearing chat),
                # seed it again regardless of selection history.
                if not match_id:
                    st.session_state["pipeline_studio_seed_key"] = seed_key
                    label = (
                        prov.get("original_name")
                        if isinstance(prov.get("original_name"), str)
                        else None
                    )
                    label = (
                        label
                        or (os.path.basename(prov_source) if prov_source else None)
                        or "data_raw"
                    )
                    new_state, _new_id = _pipeline_studio_register_dataset(
                        team_state=team_state,
                        data=df,
                        stage="raw",
                        label=label,
                        created_by="User",
                        provenance=prov
                        or {"source_type": "file", "source": prov_source or "upload"},
                        parent_id=None,
                        parent_ids=None,
                        make_active=True,
                    )
                    st.session_state["team_state"] = new_state
                    try:
                        pipelines_new = (
                            _pipeline_studio_build_pipelines_from_team_state(new_state)
                        )
                        ds_new = (
                            new_state.get("datasets")
                            if isinstance(new_state, dict)
                            else {}
                        )
                        ds_new = ds_new if isinstance(ds_new, dict) else {}
                        _update_pipeline_registry_store_for_pipelines(
                            pipelines=pipelines_new, datasets=ds_new
                        )
                    except Exception:
                        pass
                    _persist_pipeline_studio_team_state(team_state=new_state)
        except Exception:
            pass

        st.markdown("**Data preview**")
        st.dataframe(df.head(preview_rows))
        return df.to_dict(), df.head(preview_rows), provenance

    return None, None, None


def _render_analysis_detail(detail: dict, key_suffix: str) -> None:
    tabs = st.tabs(
        [
            "AI Reasoning",
            "Pipeline",
            "Data (raw/sql/wrangle/clean/features)",
            "SQL",
            "Charts",
            "EDA Reports",
            "Models",
            "Predictions",
            "MLflow",
        ]
    )
    # AI Reasoning
    with tabs[0]:
        reasoning_items = detail.get("reasoning_items", [])
        if reasoning_items:
            for name, text in reasoning_items:
                if not text:
                    continue
                st.markdown(f"**{name}:**")
                st.write(text)
                st.markdown("---")
        else:
            txt = detail.get("reasoning", detail.get("ai_reply", ""))
            if txt:
                st.write(txt)
            else:
                st.info("No reasoning available.")
    # Pipeline
    with tabs[1]:
        pipelines = detail.get("pipelines") if isinstance(detail, dict) else None
        if not isinstance(pipelines, dict):
            pipelines = {}
        pipe = detail.get("pipeline") if isinstance(detail, dict) else None
        target = None
        if pipelines:
            options = [
                ("Model (latest feature)", "model"),
                ("Active dataset", "active"),
                ("Latest dataset", "latest"),
                ("All datasets", "all"),
            ]
            target = st.radio(
                "Pipeline target",
                options=[k for k, _v in options],
                index=0,
                horizontal=True,
                key=f"pipeline_target_{key_suffix}",
            )
            target_key = dict(options).get(target, "model")
            if target_key == "model":
                pipe = detail.get("pipeline") or pipelines.get("model") or pipe
            else:
                pipe = pipelines.get(target_key) or pipe
        team_state_now = st.session_state.get("team_state", {})
        team_state_now = team_state_now if isinstance(team_state_now, dict) else {}
        datasets_now = team_state_now.get("datasets")
        datasets_now = datasets_now if isinstance(datasets_now, dict) else {}
        active_now = team_state_now.get("active_dataset_id")
        active_now = active_now if isinstance(active_now, str) else None
        if datasets_now:
            ordered_now = sorted(
                datasets_now.items(),
                key=lambda kv: float(kv[1].get("created_ts") or 0.0)
                if isinstance(kv[1], dict)
                else 0.0,
                reverse=True,
            )
            options_now = [did for did, _e in ordered_now if isinstance(did, str)]
            if active_now not in options_now:
                active_now = options_now[0] if options_now else None

            def _fmt_dataset_now(did: str) -> str:
                e = datasets_now.get(did)
                if not isinstance(e, dict):
                    return str(did)
                label = e.get("label") or did
                stage = e.get("stage") or "dataset"
                shape = e.get("shape")
                shape_txt = f" {shape}" if shape else ""
                return f"{stage}: {label}{shape_txt} ({did})"

            def _set_active_dataset_now(dataset_id: str | None) -> None:
                dataset_id = (
                    dataset_id.strip()
                    if isinstance(dataset_id, str) and dataset_id.strip()
                    else None
                )
                notice_key = f"pipeline_active_dataset_notice_{key_suffix}"
                if not dataset_id or dataset_id not in datasets_now:
                    st.session_state[notice_key] = "Select a valid dataset id."
                    return
                updated = dict(team_state_now)
                updated["active_dataset_id"] = dataset_id
                st.session_state["team_state"] = updated
                _queue_active_dataset_override("")
                _persist_pipeline_studio_team_state(team_state=updated)
                _sync_pipeline_targets_after_ui_change()
                st.session_state[notice_key] = f"Active dataset set to `{dataset_id}`."

            cols = st.columns([0.64, 0.18, 0.18], gap="small")
            with cols[0]:
                selected_now = st.selectbox(
                    "Set active dataset",
                    options=options_now,
                    index=options_now.index(active_now)
                    if active_now in options_now
                    else 0,
                    format_func=_fmt_dataset_now,
                    key=f"pipeline_active_pick_{key_suffix}",
                )
            with cols[1]:
                if st.button(
                    "Set active",
                    key=f"pipeline_active_set_{key_suffix}",
                    width="stretch",
                ):
                    _set_active_dataset_now(selected_now)
            with cols[2]:
                target_id = (
                    pipe.get("target_dataset_id") if isinstance(pipe, dict) else None
                )
                if isinstance(target_id, str) and target_id in datasets_now:
                    if st.button(
                        "Use target",
                        key=f"pipeline_active_target_{key_suffix}",
                        width="stretch",
                    ):
                        _set_active_dataset_now(target_id)
                else:
                    st.button(
                        "Use target",
                        key=f"pipeline_active_target_{key_suffix}",
                        width="stretch",
                        disabled=True,
                    )
            if st.session_state.get("active_dataset_id_override"):
                st.caption(
                    "Active dataset override is set in the sidebar and may supersede this selection."
                )
            notice = st.session_state.pop(
                f"pipeline_active_dataset_notice_{key_suffix}", None
            )
            if isinstance(notice, str) and notice.strip():
                st.success(notice)
        if isinstance(pipe, dict) and pipe.get("lineage"):
            inputs = pipe.get("inputs") or []
            inputs_txt = ""
            if isinstance(inputs, list) and inputs:
                inputs_txt = (
                    f"  \n**Inputs:** {', '.join([f'`{i}`' for i in inputs if i])}"
                )
            target_display = (
                "all"
                if str(pipe.get("target") or "").strip().lower() == "all"
                else pipe.get("target_dataset_id")
            )
            st.markdown(
                f"**Pipeline hash:** `{pipe.get('pipeline_hash')}`  \n"
                f"**Target dataset id:** `{target_display}`  \n"
                f"**Model dataset id:** `{pipe.get('model_dataset_id')}`  \n"
                f"**Active dataset id:** `{pipe.get('active_dataset_id')}`"
                f"{inputs_txt}"
            )
            if pipe.get("persisted_dir"):
                st.caption(f"Saved to: `{pipe.get('persisted_dir')}`")
            try:
                st.dataframe(pd.DataFrame(pipe.get("lineage") or []))
            except Exception:
                st.json(pipe.get("lineage"))
            script = pipe.get("script")
            try:
                spec = dict(pipe)
                spec.pop("script", None)
                st.download_button(
                    "Download pipeline spec (JSON)",
                    data=json.dumps(spec, indent=2).encode("utf-8"),
                    file_name=f"pipeline_spec_{pipe.get('target') or 'model'}.json",
                    mime="application/json",
                    key=f"download_pipeline_spec_{key_suffix}",
                )
            except Exception:
                pass
            if isinstance(script, str) and script.strip():
                st.download_button(
                    "Download pipeline script",
                    data=script.encode("utf-8"),
                    file_name=f"pipeline_repro_{pipe.get('target') or 'model'}.py",
                    mime="text/x-python",
                    key=f"download_pipeline_{key_suffix}",
                )
            st.markdown("---")
            if st.button(
                "Open Pipeline Studio",
                key=f"open_pipeline_studio_from_details_{key_suffix}",
                type="primary",
                help="Open Pipeline Studio using the selected mode.",
            ):
                # Best-effort: align Studio target selector with the Analysis Details selector.
                if isinstance(target, str) and target:
                    st.session_state["pipeline_studio_target_pending"] = target
                _request_open_pipeline_studio()

            if isinstance(script, str) and script.strip():
                st.code(script, language="python")

            # ML & prediction code steps (best effort)
            fe_code = detail.get("feature_engineering_code")
            train_code = detail.get("model_training_code")
            pred_code = detail.get("prediction_code")
            if any(
                isinstance(x, str) and x.strip()
                for x in (fe_code, train_code, pred_code)
            ):
                st.markdown("---")
                st.markdown("**ML / Prediction Steps (best effort)**")
                if isinstance(fe_code, str) and fe_code.strip():
                    with st.expander("Feature engineering code", expanded=False):
                        st.code(fe_code, language="python")
                if isinstance(train_code, str) and train_code.strip():
                    with st.expander(
                        "Model training code (H2O AutoML)", expanded=False
                    ):
                        st.code(train_code, language="python")
                if isinstance(pred_code, str) and pred_code.strip():
                    with st.expander("Prediction code", expanded=False):
                        st.code(pred_code, language="python")
        else:
            st.info(
                "No pipeline available yet. Load data and run a transform (wrangle/clean/features)."
            )
            if st.button(
                "Open Pipeline Studio",
                key=f"open_pipeline_studio_from_details_{key_suffix}",
                type="primary",
                help="Open Pipeline Studio using the selected mode.",
            ):
                st.session_state["pipeline_studio_target_pending"] = (
                    target if isinstance(target, str) and target else "Active dataset"
                )
                _request_open_pipeline_studio()

    # Data
    with tabs[2]:
        raw_df = detail.get("data_raw_df")
        sql_df = detail.get("data_sql_df")
        wrangled_df = detail.get("data_wrangled_df")
        cleaned_df = detail.get("data_cleaned_df")
        feature_df = detail.get("feature_data_df")
        if raw_df is not None:
            st.markdown("**Raw Preview**")
            st.dataframe(raw_df)
        if sql_df is not None:
            st.markdown("**SQL Preview**")
            st.dataframe(sql_df)
        if wrangled_df is not None:
            st.markdown("**Wrangled Preview**")
            st.dataframe(wrangled_df)
        if cleaned_df is not None:
            st.markdown("**Cleaned Preview**")
            st.dataframe(cleaned_df)
        if feature_df is not None:
            st.markdown("**Feature-engineered Preview**")
            st.dataframe(feature_df)
        if (
            raw_df is None
            and sql_df is None
            and wrangled_df is None
            and cleaned_df is None
            and feature_df is None
        ):
            st.info("No data frames returned.")
    # SQL
    with tabs[3]:
        sql_query = detail.get("sql_query_code")
        sql_fn = detail.get("sql_database_function")
        sql_fn_name = detail.get("sql_database_function_name")
        sql_fn_path = detail.get("sql_database_function_path")

        if sql_query:
            st.markdown("**SQL Query**")
            st.code(sql_query, language="sql")
            try:
                st.download_button(
                    "Download query (.sql)",
                    data=str(sql_query).encode("utf-8"),
                    file_name="query.sql",
                    mime="application/sql",
                    key=f"download_sql_query_{key_suffix}",
                )
            except Exception:
                pass
        else:
            st.info("No SQL query generated for this turn.")

        if sql_fn:
            st.markdown("**SQL Executor (Python)**")
            if sql_fn_name or sql_fn_path:
                st.caption(
                    "  ".join(
                        [
                            f"name={sql_fn_name}" if sql_fn_name else "",
                            f"path={sql_fn_path}" if sql_fn_path else "",
                        ]
                    ).strip()
                )
            st.code(sql_fn, language="python")
            try:
                st.download_button(
                    "Download executor (.py)",
                    data=str(sql_fn).encode("utf-8"),
                    file_name="sql_executor.py",
                    mime="text/x-python",
                    key=f"download_sql_executor_{key_suffix}",
                )
            except Exception:
                pass
    # Charts
    with tabs[4]:
        graph_json = detail.get("plotly_graph")
        viz_error = detail.get("data_visualization_error")
        viz_error_path = detail.get("data_visualization_error_log_path")
        viz_warning = detail.get("data_visualization_warning")
        if isinstance(viz_error, str) and viz_error:
            err_bits = [viz_error]
            if isinstance(viz_error_path, str) and viz_error_path:
                err_bits.append(f"Log: {viz_error_path}")
            st.error("Visualization error:\n" + "\n".join(err_bits))
        if isinstance(viz_warning, str) and viz_warning:
            st.warning(viz_warning)
        if graph_json:
            try:
                payload = (
                    json.dumps(graph_json)
                    if isinstance(graph_json, dict)
                    else graph_json
                )
                fig = _apply_streamlit_plot_style(pio.from_json(payload))
                st.plotly_chart(
                    fig,
                    width="stretch",
                    key=f"detail_chart_{key_suffix}",
                )
            except Exception as e:
                st.error(f"Error rendering chart: {e}")
        else:
            st.info("No charts returned.")
    # EDA Reports
    with tabs[5]:
        reports = detail.get("eda_reports") if isinstance(detail, dict) else None
        sweetviz_file = (
            reports.get("sweetviz_report_file") if isinstance(reports, dict) else None
        )
        dtale_url = reports.get("dtale_url") if isinstance(reports, dict) else None

        if sweetviz_file:
            st.markdown("**Sweetviz report**")
            st.write(sweetviz_file)
            try:
                with open(sweetviz_file, "r", encoding="utf-8") as f:
                    html = f.read()
                components.html(html, height=800, scrolling=True)
                st.download_button(
                    "Download Sweetviz HTML",
                    data=html.encode("utf-8"),
                    file_name=os.path.basename(sweetviz_file),
                    mime="text/html",
                    key=f"download_sweetviz_{key_suffix}",
                )
            except Exception as e:
                st.warning(f"Could not render Sweetviz report: {e}")

        if dtale_url:
            st.markdown("**D-Tale**")
            st.markdown(f"[Open D-Tale]({dtale_url})")

        if not sweetviz_file and not dtale_url:
            st.info("No EDA reports returned.")

    # Models
    with tabs[6]:
        model_info = detail.get("model_info")
        eval_art = detail.get("eval_artifacts")
        eval_graph = detail.get("eval_plotly_graph")
        if model_info is not None:
            st.markdown("**Model Info**")
            try:
                if isinstance(model_info, dict):
                    st.dataframe(pd.DataFrame(model_info), width="stretch")
                elif isinstance(model_info, list):
                    st.dataframe(pd.DataFrame(model_info), width="stretch")
                else:
                    st.json(model_info)
            except Exception:
                st.json(model_info)
        if eval_art is not None:
            st.markdown("**Evaluation**")
            st.json(eval_art)
        if eval_graph:
            try:
                payload = (
                    json.dumps(eval_graph)
                    if isinstance(eval_graph, dict)
                    else eval_graph
                )
                fig = _apply_streamlit_plot_style(pio.from_json(payload))
                st.plotly_chart(
                    fig,
                    width="stretch",
                    key=f"eval_chart_{key_suffix}",
                )
            except Exception as e:
                st.error(f"Error rendering evaluation chart: {e}")

        if model_info is None and eval_art is None and eval_graph is None:
            st.info("No model or evaluation artifacts.")

    # Predictions
    with tabs[7]:
        preds_df = detail.get("data_wrangled_df")
        if isinstance(preds_df, pd.DataFrame) and not preds_df.empty:
            lower_cols = {str(c).lower() for c in preds_df.columns}
            looks_like_preds = (
                "predict" in lower_cols
                or any(str(c).lower().startswith("p") for c in preds_df.columns)
                or any(str(c).lower().startswith("actual_") for c in preds_df.columns)
            )
            if looks_like_preds:
                st.markdown("**Predictions Preview**")
                st.dataframe(preds_df)
            else:
                st.info(
                    "No predictions detected for this turn. (Tip: ask `predict using mlflow on the dataset` or `predict with model <id> on the dataset`.)"
                )
        else:
            st.info("No predictions returned.")

    # MLflow
    with tabs[8]:
        mlflow_art = detail.get("mlflow_artifacts")
        if mlflow_art is None:
            st.info("No MLflow artifacts.")
        else:
            st.markdown("**MLflow Artifacts**")

            def _render_mlflow_artifact(obj):
                try:
                    if isinstance(obj, dict) and isinstance(obj.get("runs"), list):
                        df = pd.DataFrame(obj["runs"])
                        preferred_cols = [
                            c
                            for c in [
                                "run_id",
                                "run_name",
                                "status",
                                "start_time",
                                "duration_seconds",
                                "has_model",
                                "model_uri",
                                "params_preview",
                                "metrics_preview",
                            ]
                            if c in df.columns
                        ]
                        st.dataframe(
                            df[preferred_cols] if preferred_cols else df,
                            width="stretch",
                        )
                        if any(
                            c in df.columns
                            for c in ("params", "metrics", "tags", "artifact_uri")
                        ):
                            with st.expander("Raw run details", expanded=False):
                                st.json(obj)
                        return
                    if isinstance(obj, dict) and isinstance(
                        obj.get("experiments"), list
                    ):
                        df = pd.DataFrame(obj["experiments"])
                        preferred_cols = [
                            c
                            for c in [
                                "experiment_id",
                                "name",
                                "lifecycle_stage",
                                "creation_time",
                                "last_update_time",
                                "artifact_location",
                            ]
                            if c in df.columns
                        ]
                        st.dataframe(
                            df[preferred_cols] if preferred_cols else df,
                            width="stretch",
                        )
                        return
                    if isinstance(obj, list):
                        st.dataframe(pd.DataFrame(obj), width="stretch")
                        return
                except Exception:
                    pass
                st.json(obj)

            if isinstance(mlflow_art, dict) and not any(
                k in mlflow_art for k in ("runs", "experiments")
            ):
                is_tool_map = all(
                    isinstance(k, str) and k.startswith("mlflow_")
                    for k in mlflow_art.keys()
                )
                if is_tool_map:
                    for tool_name, tool_art in mlflow_art.items():
                        st.markdown(f"`{tool_name}`")
                        _render_mlflow_artifact(tool_art)
                else:
                    _render_mlflow_artifact(mlflow_art)
            else:
                _render_mlflow_artifact(mlflow_art)


def render_history(history: list[BaseMessage]):
    for m in history:
        role = getattr(m, "role", getattr(m, "type", "assistant"))
        content = getattr(m, "content", "")
        with st.chat_message("assistant" if role in ("assistant", "ai") else "human"):
            if isinstance(content, str) and content.startswith(UI_DETAIL_MARKER_PREFIX):
                try:
                    idx = int(content.split(":")[1])
                    detail = st.session_state.details[idx]
                except Exception:
                    # If detail is missing (e.g., state not restored), skip showing the raw marker
                    continue
                with st.expander("Analysis Details", expanded=False):
                    _render_analysis_detail(detail, key_suffix=str(idx))
            else:
                st.write(content)


@st.dialog("Pipeline Studio", width="large")
def _open_pipeline_studio_dialog() -> None:
    st.subheader("Pipeline Studio")
    _render_pipeline_studio_fragment()


render_history(msgs.messages)

# Show data preview (if selected) and store for reuse on submit
data_raw_dict, _, input_provenance = get_input_data()
# If no new data selected, reuse previously loaded data_raw from session
if data_raw_dict is None:
    data_raw_dict = st.session_state.get("selected_data_raw")
    input_provenance = st.session_state.get("selected_data_provenance")
st.session_state.selected_data_raw = data_raw_dict
st.session_state.selected_data_provenance = input_provenance


# ---------------- User input ----------------
def _resolve_chat_target_dataset() -> tuple[str | None, str | None, dict]:
    team_state = st.session_state.get("team_state", {})
    team_state = team_state if isinstance(team_state, dict) else {}
    datasets = team_state.get("datasets")
    datasets = datasets if isinstance(datasets, dict) else {}
    visible_ids, _ui_h, _ui_d, _p_hash, _node_ids = _pipeline_studio_ui_state(
        team_state=team_state
    )
    visible_set = set(visible_ids) if visible_ids else set(datasets.keys())
    if not visible_set:
        visible_set = set(datasets.keys())

    def _is_visible(did: str | None) -> bool:
        return isinstance(did, str) and did in visible_set

    active_override = st.session_state.get("active_dataset_id_override") or None
    if _is_visible(active_override):
        return active_override, "override", datasets
    if bool(st.session_state.get("pipeline_use_selected_node_for_chat", True)):
        selected = st.session_state.get("pipeline_studio_node_id")
        if _is_visible(selected):
            return selected, "studio_selected", datasets
        active_id = team_state.get("active_dataset_id")
        if _is_visible(active_id):
            return active_id, "studio_active", datasets
    return None, None, datasets


chat_target_id, chat_target_source, chat_target_datasets = (
    _resolve_chat_target_dataset()
)
if chat_target_id and isinstance(chat_target_datasets, dict):
    entry = chat_target_datasets.get(chat_target_id)
    entry = entry if isinstance(entry, dict) else {}
    label = entry.get("label") or chat_target_id
    stage = entry.get("stage") or ""
    shape = entry.get("shape")
    meta_bits = []
    if stage:
        meta_bits.append(str(stage))
    if isinstance(shape, (list, tuple)) and len(shape) == 2:
        meta_bits.append(f"{shape[0]}×{shape[1]}")
    meta_txt = f" ({', '.join(meta_bits)})" if meta_bits else ""
    source_txt = (
        f" - {str(chat_target_source).replace('_', ' ')}" if chat_target_source else ""
    )
    st.markdown(
        "\n".join(
            [
                "<style>",
                ".chat-dataset-badge {",
                "  display: inline-flex;",
                "  align-items: center;",
                "  gap: 0.35rem;",
                "  padding: 0.2rem 0.6rem;",
                "  border-radius: 999px;",
                "  background: rgba(28, 61, 102, 0.35);",
                "  color: #dfe7f2;",
                "  font-size: 0.8rem;",
                "  border: 1px solid rgba(223, 231, 242, 0.2);",
                "}",
                ".chat-dataset-badge strong {",
                "  font-weight: 600;",
                "}",
                "</style>",
                f'<div class="chat-dataset-badge">Chat target: <strong>{label}</strong>{meta_txt}{source_txt}</div>',
            ]
        ),
        unsafe_allow_html=True,
    )
else:
    st.caption("Chat target: Auto (supervisor active dataset)")

team_state_for_chat = st.session_state.get("team_state", {})
team_state_for_chat = (
    team_state_for_chat if isinstance(team_state_for_chat, dict) else {}
)
datasets_for_chat = team_state_for_chat.get("datasets")
datasets_for_chat = datasets_for_chat if isinstance(datasets_for_chat, dict) else {}
visible_ids, ui_hidden_ids, ui_deleted_ids, _p_hash, _node_ids = (
    _pipeline_studio_ui_state(team_state=team_state_for_chat)
)
visible_set = set(visible_ids) if visible_ids else set(datasets_for_chat.keys())
if ui_hidden_ids or ui_deleted_ids:
    visible_set = {
        did
        for did in visible_set
        if did not in ui_hidden_ids and did not in ui_deleted_ids
    }
chat_dataset_options = [did for did in datasets_for_chat.keys() if did in visible_set]
if chat_dataset_options:
    ordered_chat = sorted(
        [(did, datasets_for_chat.get(did)) for did in chat_dataset_options],
        key=lambda kv: float(kv[1].get("created_ts") or 0.0)
        if isinstance(kv[1], dict)
        else 0.0,
        reverse=True,
    )
    chat_dataset_options = [did for did, _e in ordered_chat]
    override_now = st.session_state.get("active_dataset_id_override") or ""
    desired = override_now if override_now in chat_dataset_options else ""
    if "chat_dataset_selector" not in st.session_state:
        st.session_state["chat_dataset_selector"] = desired

    def _format_chat_dataset(did: str) -> str:
        if not did:
            return "Auto (use active dataset)"
        entry = datasets_for_chat.get(did)
        entry = entry if isinstance(entry, dict) else {}
        label = entry.get("label") or did
        stage = entry.get("stage") or "dataset"
        shape = entry.get("shape")
        shape_txt = f" {shape}" if shape else ""
        return f"{stage}: {label}{shape_txt} ({did})"

    def _apply_chat_dataset_selection() -> None:
        selected = st.session_state.get("chat_dataset_selector") or ""
        if not selected:
            _queue_active_dataset_override("")
        else:
            _queue_active_dataset_override(str(selected))
            st.session_state["pipeline_studio_node_id_pending"] = str(selected)

    st.selectbox(
        "Chat dataset",
        options=[""] + chat_dataset_options,
        format_func=_format_chat_dataset,
        key="chat_dataset_selector",
        help="Controls which dataset chat operations target.",
        on_change=_apply_chat_dataset_selection,
    )
    if len(chat_dataset_options) >= 2:
        st.caption(
            "Tip: merge datasets in chat like `merge <id1> <id2> on <key>` or use the Pipeline Studio Merge wizard."
        )

pending_prompt = st.session_state.pop("chat_prompt_pending", None)
pending_prompt = pending_prompt.strip() if isinstance(pending_prompt, str) else ""
prompt = (
    pending_prompt if pending_prompt else st.chat_input("Ask the data science team...")
)
if prompt:
    if llm_provider_selected == "OpenAI":
        if not resolved_api_key or key_status == "bad":
            st.error(
                "OpenAI API key is required and must be valid. Enter it in the sidebar."
            )
            st.stop()
    else:
        if not resolved_ollama_model:
            st.error("Ollama model name is required. Enter it in the sidebar.")
            st.stop()

    debug_mode = bool(st.session_state.get("debug_mode", False))
    raw_prompt = prompt
    if debug_mode:
        print(f"[APP] raw_prompt={raw_prompt!r}")
    db_cmd = _parse_db_connect_command(raw_prompt)
    if debug_mode:
        print(f"[APP] db_cmd={db_cmd}")
    display_prompt = (
        db_cmd.get("display_prompt") if isinstance(db_cmd, dict) else raw_prompt
    ) or raw_prompt
    if isinstance(db_cmd, dict) and db_cmd.get("action") in ("connect", "disconnect"):
        team_prompt = (db_cmd.get("team_prompt") or "").strip()
    else:
        team_prompt = display_prompt.strip()
    merge_shortcut = None
    if team_prompt:
        team_state_for_merge = st.session_state.get("team_state", {})
        team_state_for_merge = (
            team_state_for_merge if isinstance(team_state_for_merge, dict) else {}
        )
        merge_shortcut = _parse_merge_shortcut(
            team_prompt, datasets=team_state_for_merge.get("datasets") or {}
        )

    st.chat_message("human").write(display_prompt)
    msgs.add_user_message(display_prompt)

    # Apply DB connect/disconnect updates before building the team.
    if isinstance(db_cmd, dict) and db_cmd.get("action") in ("connect", "disconnect"):
        new_url = db_cmd.get("sql_url")
        if isinstance(new_url, str) and new_url.strip():
            st.session_state["sql_url"] = new_url.strip()
            st.session_state[SQL_URL_SYNC_FLAG] = True
            if debug_mode:
                print(
                    f"[APP] Updated sql_url={_redact_sqlalchemy_url(st.session_state.get('sql_url', ''))!r}"
                )
        msg_text = db_cmd.get("message")
        if isinstance(msg_text, str) and msg_text.strip():
            st.chat_message("assistant").write(msg_text)
            msgs.add_ai_message(msg_text)

    data_raw_dict = st.session_state.get("selected_data_raw")
    input_provenance = st.session_state.get("selected_data_provenance")

    if (
        team_prompt
        and bool(st.session_state.get("pipeline_chat_context_enabled", True))
        and not (
            isinstance(db_cmd, dict)
            and db_cmd.get("action") in ("connect", "disconnect")
        )
    ):
        ctx = _pipeline_studio_chat_context(
            include_code=bool(
                st.session_state.get("pipeline_chat_context_include_code", False)
            )
        )
        if ctx:
            team_prompt = f"{team_prompt}\n\n{ctx}"

    result = None
    # If this was only a connect/disconnect command, skip running the team.
    if team_prompt:
        team = build_team(
            llm_provider_selected,
            model_choice,
            resolved_api_key if llm_provider_selected == "OpenAI" else None,
            st.session_state.get("ollama_base_url"),
            add_memory,
            st.session_state.get("sql_url", DEFAULT_SQL_URL),
            st.session_state.checkpointer if add_memory else None,
            st.session_state.get("enable_mlflow_logging", True),
            st.session_state.get("mlflow_tracking_uri"),
            st.session_state.get("mlflow_artifact_root"),
            st.session_state.get("mlflow_experiment_name", "H2O AutoML"),
            debug_mode=bool(st.session_state.get("debug_mode", False)),
        )
        try:
            # If LangGraph memory is enabled, pass only the new user message.
            # The checkpointer will supply prior state/messages for continuity.
            input_messages = (
                [HumanMessage(content=team_prompt, id=str(uuid.uuid4()))]
                if add_memory
                else _replace_last_human_message(
                    _strip_ui_marker_messages(msgs.messages), team_prompt
                )
            )
            active_dataset_override = (
                st.session_state.get("active_dataset_id_override") or None
            )
            persisted = st.session_state.get("team_state", {})
            persisted = persisted if isinstance(persisted, dict) else {}
            if not active_dataset_override and chat_target_id:
                active_dataset_override = chat_target_id

            def _payload_safe(obj):
                try:
                    if isinstance(obj, pd.DataFrame):
                        return obj.to_dict()
                except Exception:
                    pass
                return obj

            def _normalize_payload_datasets(ds):
                if not isinstance(ds, dict):
                    return ds
                out = {}
                for did, entry in ds.items():
                    if not isinstance(entry, dict):
                        out[did] = entry
                        continue
                    data = entry.get("data")
                    out[did] = {**entry, "data": _payload_safe(data)}
                return out

            invoke_payload = {
                "messages": input_messages,
                "artifacts": {
                    "config": {
                        "mlflow_tracking_uri": st.session_state.get(
                            "mlflow_tracking_uri"
                        ),
                        "mlflow_artifact_root": st.session_state.get(
                            "mlflow_artifact_root"
                        ),
                        "mlflow_experiment_name": st.session_state.get(
                            "mlflow_experiment_name", "H2O AutoML"
                        ),
                        "enable_mlflow_logging": st.session_state.get(
                            "enable_mlflow_logging", True
                        ),
                        "proactive_workflow_mode": st.session_state.get(
                            "proactive_workflow_mode", True
                        ),
                        "use_llm_intent_parser": st.session_state.get(
                            "use_llm_intent_parser", True
                        ),
                        "debug": bool(st.session_state.get("debug_mode", False)),
                        "sql_url": _redact_sqlalchemy_url(
                            st.session_state.get("sql_url", DEFAULT_SQL_URL)
                        ),
                    }
                },
                "data_raw": data_raw_dict,
            }
            if isinstance(merge_shortcut, dict):
                invoke_payload["artifacts"]["config"]["merge"] = {
                    "dataset_ids": merge_shortcut.get("dataset_ids") or [],
                    "operation": merge_shortcut.get("operation") or "join",
                    "on": ",".join(merge_shortcut.get("on") or []),
                    "how": "left",
                    "left_on": "",
                    "right_on": "",
                    "suffixes": "_x,_y",
                    "axis": 0,
                    "ignore_index": True,
                }
            if input_provenance:
                invoke_payload["artifacts"]["input_dataset"] = input_provenance
            sync_state = bool(
                st.session_state.get("pipeline_sync_state_to_agents", True)
            )
            if sync_state and persisted:
                datasets_payload = (
                    _normalize_payload_datasets(persisted.get("datasets"))
                    if "datasets" in persisted
                    else None
                )
                if "datasets" in persisted:
                    invoke_payload["datasets"] = datasets_payload
                if "active_dataset_id" in persisted:
                    invoke_payload["active_dataset_id"] = persisted.get(
                        "active_dataset_id"
                    )
                if "active_data_key" in persisted:
                    invoke_payload["active_data_key"] = persisted.get("active_data_key")
            # Provide continuity when memory is disabled (no checkpointer).
            if not add_memory and persisted:
                invoke_payload.update(
                    {
                        k: persisted.get(k)
                        for k in (
                            "data_sql",
                            "data_wrangled",
                            "data_cleaned",
                            "feature_data",
                            "target_variable",
                        )
                        if k in persisted
                    }
                )
            # Apply explicit user override last.
            if active_dataset_override:
                invoke_payload["active_dataset_id"] = active_dataset_override
            run_config = {
                "recursion_limit": recursion_limit,
                "configurable": {"thread_id": st.session_state.thread_id},
            }
            show_progress = bool(st.session_state.get("show_progress", True))
            progress_box = st.empty() if show_progress else None
            if progress_box is not None:
                progress_box.info("Working…")

            show_live_logs = bool(st.session_state.get("show_live_logs", False))
            log_container = st.empty() if show_live_logs else None
            log_placeholder = None

            import sys
            import io
            from collections import deque
            from contextlib import redirect_stdout, redirect_stderr
            import threading
            import time

            class _TeeCapture(io.TextIOBase):
                def __init__(self, passthrough, max_chars: int = 50_000):
                    super().__init__()
                    self._passthrough = passthrough
                    self._buf = deque()
                    self._n_chars = 0
                    self._max_chars = int(max_chars)
                    self._lock = threading.Lock()

                def write(self, s: str) -> int:
                    if not s:
                        return 0
                    try:
                        self._passthrough.write(s)
                    except Exception:
                        pass
                    with self._lock:
                        self._buf.append(s)
                        self._n_chars += len(s)
                        while self._n_chars > self._max_chars and self._buf:
                            removed = self._buf.popleft()
                            self._n_chars -= len(removed)
                    return len(s)

                def flush(self) -> None:
                    try:
                        self._passthrough.flush()
                    except Exception:
                        pass

                def get_text(self) -> str:
                    with self._lock:
                        return "".join(self._buf)

            stdout_cap = _TeeCapture(sys.stdout)
            stderr_cap = _TeeCapture(sys.stderr)

            if log_container is not None:
                with log_container.container():
                    st.markdown("**Live logs**")
                    st.caption("Showing the most recent output (tail).")
                    log_status_placeholder = st.empty()
                    log_placeholder = st.empty()
                    log_placeholder.code("", language="text")
            else:
                log_status_placeholder = None

            def _run_with_stream() -> dict | None:
                last_event = None
                for event in team.stream(
                    invoke_payload, config=run_config, stream_mode="values"
                ):
                    if not isinstance(event, dict):
                        continue
                    last_event = event
                    label = None
                    nxt = event.get("next")
                    if (
                        isinstance(nxt, str)
                        and nxt.strip()
                        and nxt.strip().upper() != "FINISH"
                    ):
                        label = f"Routing → {nxt.strip()}"
                    elif (
                        isinstance(event.get("last_worker"), str)
                        and event.get("last_worker").strip()
                    ):
                        label = event.get("last_worker").strip()
                    shared["label"] = label
                return last_event

            def _run_with_invoke():
                return team.invoke(invoke_payload, config=run_config)

            if show_live_logs:
                shared = {"done": False, "label": None, "result": None, "error": None}
                start_ts = time.time()
                last_output_ts = start_ts
                last_status_ts = 0.0

                def _worker():
                    try:
                        with redirect_stdout(stdout_cap), redirect_stderr(stderr_cap):
                            if show_progress and hasattr(team, "stream"):
                                shared["result"] = _run_with_stream()
                            else:
                                shared["result"] = _run_with_invoke()
                    except Exception as e:
                        shared["error"] = e
                    finally:
                        shared["done"] = True

                t = threading.Thread(target=_worker, daemon=True)
                t.start()

                last_rendered = None
                while not shared.get("done"):
                    if (
                        progress_box is not None
                        and isinstance(shared.get("label"), str)
                        and shared["label"]
                    ):
                        progress_box.info(f"Working: `{shared['label']}`")
                    if log_placeholder is not None:
                        text = (stdout_cap.get_text() + stderr_cap.get_text()).strip()
                        if text:
                            tail = "\n".join(text.splitlines()[-250:])
                            if tail and tail != last_rendered:
                                log_placeholder.code(tail, language="text")
                                last_rendered = tail
                                last_output_ts = time.time()
                    if log_status_placeholder is not None:
                        now = time.time()
                        if (now - last_status_ts) >= 0.5:
                            elapsed = now - start_ts
                            since_out = now - last_output_ts
                            log_status_placeholder.caption(
                                f"Elapsed: {elapsed:0.1f}s • Last output: {since_out:0.1f}s ago"
                            )
                            last_status_ts = now
                    time.sleep(0.15)

                t.join(timeout=0.1)
                if log_placeholder is not None:
                    text = (stdout_cap.get_text() + stderr_cap.get_text()).strip()
                    if text:
                        tail = "\n".join(text.splitlines()[-250:])
                        if tail and tail != last_rendered:
                            log_placeholder.code(tail, language="text")
                            last_output_ts = time.time()
                if log_status_placeholder is not None:
                    now = time.time()
                    elapsed = now - start_ts
                    since_out = now - last_output_ts
                    log_status_placeholder.caption(
                        f"Elapsed: {elapsed:0.1f}s • Last output: {since_out:0.1f}s ago"
                    )
                if shared.get("error") is not None:
                    raise shared["error"]
                result = shared.get("result")
            elif show_progress and hasattr(team, "stream"):
                last_event = None
                with redirect_stdout(stdout_cap), redirect_stderr(stderr_cap):
                    for event in team.stream(
                        invoke_payload, config=run_config, stream_mode="values"
                    ):
                        if isinstance(event, dict):
                            last_event = event
                            label = None
                            nxt = event.get("next")
                            if (
                                isinstance(nxt, str)
                                and nxt.strip()
                                and nxt.strip().upper() != "FINISH"
                            ):
                                label = f"Routing → {nxt.strip()}"
                            elif (
                                isinstance(event.get("last_worker"), str)
                                and event.get("last_worker").strip()
                            ):
                                label = event.get("last_worker").strip()
                            if (
                                progress_box is not None
                                and isinstance(label, str)
                                and label.strip()
                            ):
                                progress_box.info(f"Working: `{label}`")
                    result = last_event
            else:
                with redirect_stdout(stdout_cap), redirect_stderr(stderr_cap):
                    result = team.invoke(invoke_payload, config=run_config)

            if progress_box is not None:
                progress_box.empty()
            if log_container is not None:
                log_container.empty()
        except Exception as e:
            try:
                if "progress_box" in locals() and progress_box is not None:
                    progress_box.empty()
            except Exception:
                pass
            try:
                if "log_container" in locals() and log_container is not None:
                    log_container.empty()
            except Exception:
                pass
            msg = str(e)
            if (
                "rate_limit_exceeded" in msg
                or "tokens per min" in msg
                or "tpm" in msg.lower()
                or "request too large" in msg.lower()
            ):
                st.error(f"Error running team (rate limit): {e}")
                st.info(
                    "Try again in ~60s, or reduce load by disabling memory, lowering recursion, "
                    "or switching to a smaller model."
                )
            else:
                st.error(f"Error running team: {e}")
            result = None

    if result:
        # Persist data_raw from result for follow-on requests
        if result.get("data_raw") is not None:
            try:
                st.session_state.selected_data_raw = result.get("data_raw").to_dict()
            except Exception:
                st.session_state.selected_data_raw = result.get("data_raw")

        # Persist additional state slots for continuity when memory is off
        # (and to support dataset selection UX in the sidebar).
        def _maybe_df_to_dict(obj):
            try:
                if isinstance(obj, pd.DataFrame):
                    return obj.to_dict()
            except Exception:
                pass
            return obj

        def _normalize_datasets(ds):
            if not isinstance(ds, dict):
                return ds
            out = {}
            for did, entry in ds.items():
                if not isinstance(entry, dict):
                    out[did] = entry
                    continue
                data = entry.get("data")
                out[did] = {**entry, "data": _maybe_df_to_dict(data)}
            return out

        def _is_pipeline_studio_node(entry: dict) -> bool:
            if not isinstance(entry, dict):
                return False
            prov = entry.get("provenance")
            prov = prov if isinstance(prov, dict) else {}
            return prov.get("source_type") == "pipeline_studio"

        def _entry_parent_ids(entry_obj: dict) -> list[str]:
            entry_obj = entry_obj if isinstance(entry_obj, dict) else {}
            parents: list[str] = []
            pids = entry_obj.get("parent_ids")
            if isinstance(pids, list):
                parents.extend([str(p) for p in pids if isinstance(p, str) and p])
            pid = entry_obj.get("parent_id")
            if isinstance(pid, str) and pid and pid not in parents:
                parents.insert(0, pid)
            return [p for p in parents if p]

        def _collect_ancestor_ids(datasets: dict, seed_ids: set[str]) -> set[str]:
            if not isinstance(datasets, dict) or not seed_ids:
                return set()
            seen: set[str] = set()
            stack = [sid for sid in seed_ids if isinstance(sid, str) and sid]
            while stack:
                nid = stack.pop()
                if nid in seen:
                    continue
                seen.add(nid)
                entry = datasets.get(nid)
                if not isinstance(entry, dict):
                    continue
                for pid in _entry_parent_ids(entry):
                    if pid and pid not in seen and pid in datasets:
                        stack.append(pid)
            return seen

        try:
            state_updates = {}
            for k in (
                "data_sql",
                "data_wrangled",
                "data_cleaned",
                "feature_data",
                "active_data_key",
                "active_dataset_id",
                "datasets",
                "target_variable",
            ):
                if k in result:
                    if k == "datasets":
                        normalized = _normalize_datasets(result.get(k))
                        persisted = st.session_state.team_state or {}
                        persisted = persisted if isinstance(persisted, dict) else {}
                        existing = persisted.get("datasets")
                        existing = existing if isinstance(existing, dict) else {}
                        if isinstance(normalized, dict) and existing:
                            merged = dict(normalized)
                            preserve_all = bool(
                                st.session_state.get(
                                    "pipeline_preserve_all_nodes", True
                                )
                            )
                            preserve_studio = bool(
                                st.session_state.get(
                                    "pipeline_preserve_studio_nodes", True
                                )
                            )
                            protected_ids: set[str] = set()
                            locked_ids = st.session_state.get(
                                "pipeline_studio_locked_node_ids", []
                            )
                            if isinstance(locked_ids, (list, set, tuple)):
                                protected_ids |= {
                                    str(x)
                                    for x in locked_ids
                                    if isinstance(x, str) and x
                                }
                            if preserve_all:
                                protected_ids |= {
                                    str(x)
                                    for x in existing.keys()
                                    if isinstance(x, str) and x
                                }
                            elif preserve_studio:
                                for did, entry in existing.items():
                                    if _is_pipeline_studio_node(entry):
                                        protected_ids.add(did)
                            if protected_ids:
                                if not preserve_all:
                                    protected_ids = _collect_ancestor_ids(
                                        existing, protected_ids
                                    )
                                for did in protected_ids:
                                    if did not in merged and did in existing:
                                        merged[did] = existing[did]
                            normalized = merged
                        state_updates[k] = normalized
                    else:
                        state_updates[k] = _maybe_df_to_dict(result.get(k))
            if state_updates:
                st.session_state.team_state = {
                    **(st.session_state.team_state or {}),
                    **state_updates,
                }
                _persist_pipeline_studio_team_state(
                    team_state=st.session_state.team_state
                )
        except Exception:
            pass

        # append last AI message to chat history for display
        last_ai = None
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) or getattr(msg, "role", None) in (
                "assistant",
                "ai",
            ):
                last_ai = msg
                break
        if last_ai:
            msgs.add_ai_message(getattr(last_ai, "content", ""))
            st.chat_message("assistant").write(getattr(last_ai, "content", ""))

        # Collect reasoning from AI messages after latest human
        reasoning = ""
        reasoning_items = []
        latest_human_index = -1
        for i, message in enumerate(result.get("messages", [])):
            role = getattr(message, "role", getattr(message, "type", None))
            if role in ("human", "user"):
                latest_human_index = i
        # Collapse multiple messages from the same agent into the latest one
        ordered_names = []
        latest_by_name = {}
        for message in result.get("messages", [])[latest_human_index + 1 :]:
            role = getattr(message, "role", getattr(message, "type", None))
            if role in ("assistant", "ai"):
                name = getattr(message, "name", None) or "assistant"
                if name == "assistant":
                    txt_lower = (getattr(message, "content", "") or "").lower()
                    if "loader" in txt_lower:
                        name = "data_loader_agent"
                content = getattr(message, "content", "")
                if not content:
                    continue
                latest_by_name[name] = content
                if name not in ordered_names:
                    ordered_names.append(name)

        for name in ordered_names:
            content = latest_by_name.get(name, "")
            if not content:
                continue
            display_name = name.replace("_", " ").title()
            reasoning_items.append((display_name, content))
            reasoning += f"##### {display_name}:\n\n{content}\n\n---\n\n"

        # Collect detail snapshot for tabbed display
        artifacts = result.get("artifacts", {}) or {}
        ran_agents = set(latest_by_name.keys())
        sql_payload = artifacts.get("sql") if isinstance(artifacts, dict) else None
        sql_payload = sql_payload if isinstance(sql_payload, dict) else None

        def _to_df(obj):
            try:
                return pd.DataFrame(obj) if obj is not None else None
            except Exception:
                return None

        def _extract_eda_reports(artifacts: dict) -> dict:
            if not isinstance(artifacts, dict):
                return {}
            eda_payload = artifacts.get("eda")
            if not isinstance(eda_payload, dict):
                return {}

            sweetviz_report_file = None
            dtale_url = None

            candidates = []
            if isinstance(eda_payload.get("generate_sweetviz_report"), dict):
                candidates.append(eda_payload.get("generate_sweetviz_report"))
            candidates.extend(list(eda_payload.values()))
            for v in candidates:
                if isinstance(v, dict) and v.get("report_file"):
                    sweetviz_report_file = v.get("report_file")
                    break

            candidates = []
            if isinstance(eda_payload.get("generate_dtale_report"), dict):
                candidates.append(eda_payload.get("generate_dtale_report"))
            candidates.extend(list(eda_payload.values()))
            for v in candidates:
                if isinstance(v, dict) and v.get("dtale_url"):
                    dtale_url = v.get("dtale_url")
                    break

            out = {}
            if sweetviz_report_file:
                out["sweetviz_report_file"] = sweetviz_report_file
            if dtale_url:
                out["dtale_url"] = dtale_url
            return out

        def _summarize_artifacts(artifacts: dict) -> dict:
            """
            Produce a lightweight summary to keep the UI responsive.
            """
            if not isinstance(artifacts, dict):
                return {}
            summary = {}
            for k, v in artifacts.items():
                # Table-like payload
                if isinstance(v, dict) and "data" in v:
                    try:
                        df_tmp = pd.DataFrame(v["data"])
                        summary[k] = {
                            "type": "table",
                            "shape": tuple(df_tmp.shape),
                            "preview_head": df_tmp.head(5).to_dict(),
                        }
                    except Exception:
                        summary[k] = {"type": "table", "note": "preview unavailable"}
                # Plotly figure
                elif isinstance(v, dict) and "plotly_graph" in v:
                    summary[k] = {"type": "plot", "note": "plotly figure returned"}
                else:
                    summary[k] = (
                        v if isinstance(v, (str, int, float, list, dict)) else str(v)
                    )
            return summary

        def _truncate_code(text: object, limit: int = 12000) -> str | None:
            if not isinstance(text, str):
                return None
            t = text.strip()
            if not t:
                return None
            if len(t) <= limit:
                return t
            return t[:limit].rstrip() + "\n\n# ... truncated ..."

        datasets_dict = (
            result.get("datasets")
            if isinstance(result, dict) and isinstance(result.get("datasets"), dict)
            else {}
        )
        active_dataset_id = (
            result.get("active_dataset_id") if isinstance(result, dict) else None
        )

        pipeline_model = (
            build_pipeline_snapshot(datasets_dict, active_dataset_id=active_dataset_id)
            if isinstance(datasets_dict, dict) and datasets_dict
            else None
        )
        pipelines = (
            {
                "model": pipeline_model,
                "active": build_pipeline_snapshot(
                    datasets_dict, active_dataset_id=active_dataset_id, target="active"
                ),
                "latest": build_pipeline_snapshot(
                    datasets_dict, active_dataset_id=active_dataset_id, target="latest"
                ),
                "all": build_pipeline_snapshot(
                    datasets_dict, active_dataset_id=active_dataset_id, target="all"
                ),
            }
            if isinstance(datasets_dict, dict) and datasets_dict
            else None
        )

        def _extract_feature_engineering_code() -> str | None:
            if not isinstance(pipeline_model, dict):
                return None
            did = pipeline_model.get("model_dataset_id")
            if not isinstance(did, str) or not did:
                return None
            entry = datasets_dict.get(did)
            if not isinstance(entry, dict):
                return None
            prov = (
                entry.get("provenance")
                if isinstance(entry.get("provenance"), dict)
                else {}
            )
            transform = (
                prov.get("transform") if isinstance(prov.get("transform"), dict) else {}
            )
            if str(transform.get("kind") or "") != "python_function":
                return None
            return _truncate_code(transform.get("function_code"))

        def _extract_prediction_code() -> str | None:
            if not isinstance(artifacts, dict):
                return None
            mp = artifacts.get("mlflow_predictions")
            if (
                isinstance(mp, dict)
                and isinstance(mp.get("run_id"), str)
                and mp.get("run_id").strip()
            ):
                run_id = mp["run_id"].strip()
                return _truncate_code(
                    "\n".join(
                        [
                            "import pandas as pd",
                            "import mlflow",
                            "",
                            f"model_uri = 'runs:/{run_id}/model'",
                            "model = mlflow.pyfunc.load_model(model_uri)",
                            "preds = model.predict(df)",
                            "df = preds if isinstance(preds, pd.DataFrame) else pd.DataFrame(preds)",
                        ]
                    ),
                    limit=6000,
                )
            hp = artifacts.get("h2o_predictions")
            if (
                isinstance(hp, dict)
                and isinstance(hp.get("model_id"), str)
                and hp.get("model_id").strip()
            ):
                model_id = hp["model_id"].strip()
                return _truncate_code(
                    "\n".join(
                        [
                            "import h2o",
                            "",
                            "h2o.init()",
                            f"model = h2o.get_model('{model_id}')",
                            "frame = h2o.H2OFrame(df)",
                            "preds = model.predict(frame)",
                            "df = preds.as_data_frame(use_pandas=True)",
                        ]
                    ),
                    limit=6000,
                )
            return None

        feature_engineering_code = _extract_feature_engineering_code()
        model_training_code = _truncate_code(
            artifacts.get("h2o", {}).get("h2o_train_function")
            if isinstance(artifacts.get("h2o"), dict)
            else None
        )
        prediction_code = _extract_prediction_code()

        detail = {
            "ai_reply": getattr(last_ai, "content", "") if last_ai else "",
            "reasoning": reasoning or getattr(last_ai, "content", ""),
            "reasoning_items": reasoning_items,
            "data_raw_df": _to_df(result.get("data_raw")),
            "data_sql_df": _to_df(result.get("data_sql")),
            "data_wrangled_df": _to_df(result.get("data_wrangled")),
            "data_cleaned_df": _to_df(result.get("data_cleaned")),
            "feature_data_df": _to_df(result.get("feature_data")),
            # Only show artifacts produced during this invocation to avoid stale charts/models.
            "eda_reports": (
                _extract_eda_reports(artifacts)
                if "eda_tools_agent" in ran_agents
                else None
            ),
            "plotly_graph": (
                artifacts.get("viz", {}).get("plotly_graph")
                if "data_visualization_agent" in ran_agents
                and isinstance(artifacts.get("viz"), dict)
                else None
            ),
            "data_visualization_error": (
                artifacts.get("viz", {}).get("error")
                if "data_visualization_agent" in ran_agents
                and isinstance(artifacts.get("viz"), dict)
                else None
            ),
            "data_visualization_error_log_path": (
                artifacts.get("viz", {}).get("error_log_path")
                if "data_visualization_agent" in ran_agents
                and isinstance(artifacts.get("viz"), dict)
                else None
            ),
            "data_visualization_warning": (
                artifacts.get("viz", {}).get("warning")
                if "data_visualization_agent" in ran_agents
                and isinstance(artifacts.get("viz"), dict)
                else None
            ),
            "model_info": (
                (result.get("model_info") or artifacts.get("h2o"))
                if "h2o_ml_agent" in ran_agents
                else None
            ),
            "eval_artifacts": (
                artifacts.get("eval", {}).get("eval_artifacts")
                if "model_evaluation_agent" in ran_agents
                and isinstance(artifacts.get("eval"), dict)
                else None
            ),
            "eval_plotly_graph": (
                artifacts.get("eval", {}).get("plotly_graph")
                if "model_evaluation_agent" in ran_agents
                and isinstance(artifacts.get("eval"), dict)
                else None
            ),
            "feature_engineering_code": feature_engineering_code,
            "model_training_code": model_training_code,
            "prediction_code": prediction_code,
            "mlflow_artifacts": (
                (
                    result.get("mlflow_artifacts")
                    or artifacts.get("mlflow")
                    or artifacts.get("mlflow_log")
                )
                if (
                    "mlflow_tools_agent" in ran_agents
                    or "mlflow_logging_agent" in ran_agents
                )
                else None
            ),
            # Store only a summarized version to avoid rendering huge payloads
            "artifacts": _summarize_artifacts(artifacts),
            "pipeline": pipeline_model,
            "pipelines": pipelines,
            "sql_query_code": (
                sql_payload.get("sql_query_code")
                if sql_payload and "sql_database_agent" in ran_agents
                else None
            ),
            "sql_database_function": (
                sql_payload.get("sql_database_function")
                if sql_payload and "sql_database_agent" in ran_agents
                else None
            ),
            "sql_database_function_path": (
                sql_payload.get("sql_database_function_path")
                if sql_payload and "sql_database_agent" in ran_agents
                else None
            ),
            "sql_database_function_name": (
                sql_payload.get("sql_database_function_name")
                if sql_payload and "sql_database_agent" in ran_agents
                else None
            ),
        }

        # Persist pipeline files to a user-configurable directory (best effort).
        try:
            if (
                st.session_state.get("pipeline_persist_enabled")
                and isinstance(detail.get("pipeline"), dict)
                and detail["pipeline"].get("lineage")
            ):
                saved = persist_pipeline_artifacts(
                    detail["pipeline"],
                    base_dir=st.session_state.get("pipeline_persist_dir"),
                    overwrite=bool(st.session_state.get("pipeline_persist_overwrite")),
                    include_sql=bool(
                        st.session_state.get("pipeline_persist_include_sql", True)
                    ),
                    sql_query=detail.get("sql_query_code"),
                    sql_executor=detail.get("sql_database_function"),
                )
                if isinstance(saved, dict) and saved.get("persisted_dir"):
                    detail["pipeline"]["persisted_dir"] = saved.get("persisted_dir")
                    detail["pipeline"]["persisted_spec_path"] = saved.get("spec_path")
                    detail["pipeline"]["persisted_script_path"] = saved.get(
                        "script_path"
                    )
                    detail["pipeline"]["persisted_sql_query_path"] = saved.get(
                        "sql_query_path"
                    )
                    detail["pipeline"]["persisted_sql_executor_path"] = saved.get(
                        "sql_executor_path"
                    )
                    st.session_state.last_pipeline_persist_dir = saved.get(
                        "persisted_dir"
                    )
                    if isinstance(detail.get("pipelines"), dict):
                        detail["pipelines"]["model"] = detail["pipeline"]
                if isinstance(saved, dict) and saved.get("error"):
                    st.sidebar.warning(f"Pipeline save failed: {saved.get('error')}")
        except Exception:
            pass

        idx = len(st.session_state.details)

        try:
            import time

            dsid = None
            if isinstance(result, dict):
                dsid = result.get("active_dataset_id")
            if isinstance(detail.get("pipelines"), dict):
                active_pipe = detail["pipelines"].get("active")
                if isinstance(active_pipe, dict) and isinstance(
                    active_pipe.get("target_dataset_id"), str
                ):
                    dsid = active_pipe.get("target_dataset_id")
            dsid = dsid if isinstance(dsid, str) and dsid else None

            def _safe_json(obj):
                try:
                    if isinstance(obj, pd.DataFrame):
                        return obj.to_dict()
                except Exception:
                    pass
                return obj

            if dsid:
                idx_map = st.session_state.get("pipeline_studio_artifacts")
                idx_map = idx_map if isinstance(idx_map, dict) else {}
                cur = idx_map.get(dsid) if isinstance(idx_map.get(dsid), dict) else {}
                ts = time.time()
                if detail.get("plotly_graph") is not None:
                    cur["plotly_graph"] = {
                        "json": _safe_json(detail.get("plotly_graph")),
                        "turn_idx": idx,
                        "created_ts": ts,
                    }
                if detail.get("data_visualization_error") is not None:
                    cur["viz_error"] = {
                        "message": _safe_json(detail.get("data_visualization_error")),
                        "log_path": _safe_json(
                            detail.get("data_visualization_error_log_path")
                        ),
                        "turn_idx": idx,
                        "created_ts": ts,
                    }
                if detail.get("data_visualization_warning") is not None:
                    cur["viz_warning"] = {
                        "message": _safe_json(
                            detail.get("data_visualization_warning")
                        ),
                        "turn_idx": idx,
                        "created_ts": ts,
                    }
                if detail.get("eda_reports") is not None:
                    cur["eda_reports"] = {
                        "reports": _safe_json(detail.get("eda_reports")),
                        "turn_idx": idx,
                        "created_ts": ts,
                    }
                if detail.get("model_info") is not None:
                    cur["model_info"] = {
                        "info": _safe_json(detail.get("model_info")),
                        "turn_idx": idx,
                        "created_ts": ts,
                    }
                if detail.get("eval_artifacts") is not None:
                    cur["eval_artifacts"] = {
                        "artifacts": _safe_json(detail.get("eval_artifacts")),
                        "turn_idx": idx,
                        "created_ts": ts,
                    }
                if detail.get("eval_plotly_graph") is not None:
                    cur["eval_plotly_graph"] = {
                        "json": _safe_json(detail.get("eval_plotly_graph")),
                        "turn_idx": idx,
                        "created_ts": ts,
                    }
                if detail.get("mlflow_artifacts") is not None:
                    cur["mlflow_artifacts"] = {
                        "artifacts": _safe_json(detail.get("mlflow_artifacts")),
                        "turn_idx": idx,
                        "created_ts": ts,
                    }
                if cur:
                    idx_map[dsid] = cur
                    st.session_state["pipeline_studio_artifacts"] = idx_map
                    _update_pipeline_studio_artifact_store_for_dataset(dsid, cur)
        except Exception:
            pass

        # Persist a lightweight semantic pipeline registry (best effort).
        try:
            if isinstance(pipelines, dict) and isinstance(datasets_dict, dict):
                _update_pipeline_registry_store_for_pipelines(
                    pipelines=pipelines, datasets=datasets_dict
                )
        except Exception:
            pass

        st.session_state.details.append(detail)
        msgs.add_ai_message(f"{UI_DETAIL_MARKER_PREFIX}{idx}")

        # Sidebar widgets render before the team run in Streamlit's top-to-bottom execution.
        # Rerun once after saving results so the sidebar reflects the latest active dataset immediately.
        st.rerun()

# ---------------- Pipeline Studio ----------------


def _render_pipeline_studio() -> None:
    studio_state = st.session_state.get("team_state", {})
    studio_state = studio_state if isinstance(studio_state, dict) else {}
    studio_datasets = studio_state.get("datasets")
    studio_datasets = studio_datasets if isinstance(studio_datasets, dict) else {}
    studio_active_id = studio_state.get("active_dataset_id")
    studio_active_id = studio_active_id if isinstance(studio_active_id, str) else None

    # Make editable code areas look like code (monospace).
    st.markdown(
        """
        <style>
          div[data-testid="stTextArea"] textarea {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace !important;
            font-size: 0.9rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def _dataset_entry_to_df(entry: dict) -> pd.DataFrame | None:
        if not isinstance(entry, dict):
            return None
        data = entry.get("data")
        try:
            if isinstance(data, pd.DataFrame):
                return data
        except Exception:
            pass
        try:
            if isinstance(data, dict):
                return pd.DataFrame.from_dict(data)
            if isinstance(data, list):
                return pd.DataFrame(data)
        except Exception:
            return None
        return None

    def _infer_first_def_name(code: str) -> str | None:
        if not isinstance(code, str) or not code:
            return None
        m = re.search(
            r"^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
            code,
            flags=re.MULTILINE,
        )
        return m.group(1) if m else None

    def _normalize_pipeline_stage(stage: str) -> str:
        stage = stage.strip().lower() if isinstance(stage, str) else ""
        stage = re.sub(r"[^a-z0-9_]+", "_", stage)
        stage = re.sub(r"_+", "_", stage).strip("_")
        return stage or "custom"

    def _exec_python_transform(
        *,
        code: str,
        df_in: pd.DataFrame,
        fn_name_hint: str | None = None,
    ) -> tuple[pd.DataFrame, str | None]:
        code = code if isinstance(code, str) else ""
        code = code.strip()
        if not code:
            raise ValueError("Draft code is empty; nothing to run.")

        inferred = _infer_first_def_name(code)
        fn_name_hint = fn_name_hint.strip() if isinstance(fn_name_hint, str) else None

        # Use a single namespace for globals so top-level imports are visible to the function.
        exec_env: dict[str, object] = {"pd": pd}
        try:
            import numpy as np  # type: ignore

            exec_env["np"] = np
        except Exception:
            pass
        exec(code, exec_env, exec_env)

        fn = None
        fn_used: str | None = None
        for candidate in [fn_name_hint, inferred]:
            if not candidate:
                continue
            obj = exec_env.get(candidate)
            if callable(obj):
                fn = obj
                fn_used = candidate
                break
        if fn is None:
            # Fall back: pick the first callable defined in locals.
            for name, obj in exec_env.items():
                if callable(obj):
                    fn = obj
                    fn_used = name if isinstance(name, str) and name else None
                    break
        if fn is None:
            raise ValueError(
                "Could not find a callable function in the draft. Define a function like `def transform(df): ...`."
            )

        out = fn(df_in)
        if isinstance(out, tuple) and out:
            out = out[0]
        if not isinstance(out, pd.DataFrame):
            try:
                if isinstance(out, dict):
                    out = pd.DataFrame.from_dict(out)
                elif isinstance(out, list):
                    out = pd.DataFrame(out)
            except Exception:
                out = None
        if not isinstance(out, pd.DataFrame):
            raise ValueError("Draft function did not return a pandas DataFrame.")
        return out, fn_used

    def _entry_parent_ids(entry_obj: dict) -> list[str]:
        entry_obj = entry_obj if isinstance(entry_obj, dict) else {}
        parents: list[str] = []
        pids = entry_obj.get("parent_ids")
        if isinstance(pids, list):
            parents.extend([str(p) for p in pids if isinstance(p, str) and p])
        pid = entry_obj.get("parent_id")
        if isinstance(pid, str) and pid and pid not in parents:
            parents.insert(0, pid)
        return [p for p in parents if p]

    def _build_children_index(datasets: dict) -> dict[str, set[str]]:
        datasets = datasets if isinstance(datasets, dict) else {}
        children: dict[str, set[str]] = {}
        for did, ent in datasets.items():
            if not isinstance(did, str) or not did or not isinstance(ent, dict):
                continue
            for pid in _entry_parent_ids(ent):
                children.setdefault(pid, set()).add(did)
        return children

    def _descendants(start_id: str, children_index: dict[str, set[str]]) -> set[str]:
        start_id = start_id.strip() if isinstance(start_id, str) else ""
        if not start_id:
            return set()
        seen: set[str] = set()
        stack = list(children_index.get(start_id, set()))
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            stack.extend(list(children_index.get(nid, set())))
        return seen

    def _pipeline_studio_branch_ids(root_id: str) -> set[str]:
        root_id = root_id.strip() if isinstance(root_id, str) else ""
        if not root_id:
            return set()
        child_idx = _build_children_index(studio_datasets)
        return {root_id} | _descendants(root_id, child_idx)

    def _pipeline_studio_soft_delete_branch(
        *, pipeline_hash: str, root_id: str
    ) -> None:
        try:
            import time as _time

            p_hash = pipeline_hash.strip() if isinstance(pipeline_hash, str) else ""
            root_id = root_id.strip() if isinstance(root_id, str) else ""
            if not p_hash or not root_id:
                return
            branch_ids = _pipeline_studio_branch_ids(root_id)
            if not branch_ids:
                return
            ui_h, ui_d = _pipeline_studio_get_registry_ui(pipeline_hash=p_hash)
            ui_h = set(ui_h)
            ui_d = set(ui_d) | {str(x) for x in branch_ids if str(x).strip()}
            _pipeline_studio_set_registry_ui(
                pipeline_hash=p_hash,
                hidden_ids=ui_h,
                deleted_ids=ui_d,
            )
            st.session_state["pipeline_studio_flow_hidden_ids"] = sorted(ui_h | ui_d)
            st.session_state["pipeline_studio_flow_fit_view_pending"] = True
            st.session_state["pipeline_studio_flow_ts"] = int(_time.time() * 1000)
            st.session_state["pipeline_studio_history_notice"] = (
                f"Soft-deleted {len(branch_ids)} node(s) under `{root_id}`."
            )
            _sync_pipeline_targets_after_ui_change()
        except Exception:
            pass

    def _pipeline_studio_restore_branch(*, pipeline_hash: str, root_id: str) -> None:
        try:
            import time as _time

            p_hash = pipeline_hash.strip() if isinstance(pipeline_hash, str) else ""
            root_id = root_id.strip() if isinstance(root_id, str) else ""
            if not p_hash or not root_id:
                return
            branch_ids = _pipeline_studio_branch_ids(root_id)
            if not branch_ids:
                return
            ui_h, ui_d = _pipeline_studio_get_registry_ui(pipeline_hash=p_hash)
            ui_h = set(ui_h) - branch_ids
            ui_d = set(ui_d) - branch_ids
            _pipeline_studio_set_registry_ui(
                pipeline_hash=p_hash,
                hidden_ids=ui_h,
                deleted_ids=ui_d,
            )
            st.session_state["pipeline_studio_flow_hidden_ids"] = sorted(ui_h | ui_d)
            st.session_state["pipeline_studio_flow_fit_view_pending"] = True
            st.session_state["pipeline_studio_flow_ts"] = int(_time.time() * 1000)
            st.session_state["pipeline_studio_history_notice"] = (
                f"Restored {len(branch_ids)} node(s) under `{root_id}`."
            )
            _sync_pipeline_targets_after_ui_change()
        except Exception:
            pass

    def _pipeline_studio_hide_branch(*, pipeline_hash: str, root_id: str) -> None:
        try:
            import time as _time

            p_hash = pipeline_hash.strip() if isinstance(pipeline_hash, str) else ""
            root_id = root_id.strip() if isinstance(root_id, str) else ""
            if not p_hash or not root_id:
                return
            branch_ids = _pipeline_studio_branch_ids(root_id)
            if not branch_ids:
                return
            ui_h, ui_d = _pipeline_studio_get_registry_ui(pipeline_hash=p_hash)
            ui_h = set(ui_h) | branch_ids
            ui_d = set(ui_d)
            _pipeline_studio_set_registry_ui(
                pipeline_hash=p_hash,
                hidden_ids=ui_h,
                deleted_ids=ui_d,
            )
            st.session_state["pipeline_studio_flow_hidden_ids"] = sorted(ui_h | ui_d)
            st.session_state["pipeline_studio_flow_fit_view_pending"] = True
            st.session_state["pipeline_studio_flow_ts"] = int(_time.time() * 1000)
            st.session_state["pipeline_studio_history_notice"] = (
                f"Hid {len(branch_ids)} node(s) under `{root_id}`."
            )
            _sync_pipeline_targets_after_ui_change()
        except Exception:
            pass

    def _pipeline_studio_unhide_branch(*, pipeline_hash: str, root_id: str) -> None:
        try:
            import time as _time

            p_hash = pipeline_hash.strip() if isinstance(pipeline_hash, str) else ""
            root_id = root_id.strip() if isinstance(root_id, str) else ""
            if not p_hash or not root_id:
                return
            branch_ids = _pipeline_studio_branch_ids(root_id)
            if not branch_ids:
                return
            ui_h, ui_d = _pipeline_studio_get_registry_ui(pipeline_hash=p_hash)
            ui_h = set(ui_h) - branch_ids
            ui_d = set(ui_d)
            _pipeline_studio_set_registry_ui(
                pipeline_hash=p_hash,
                hidden_ids=ui_h,
                deleted_ids=ui_d,
            )
            st.session_state["pipeline_studio_flow_hidden_ids"] = sorted(ui_h | ui_d)
            st.session_state["pipeline_studio_flow_fit_view_pending"] = True
            st.session_state["pipeline_studio_flow_ts"] = int(_time.time() * 1000)
            st.session_state["pipeline_studio_history_notice"] = (
                f"Unhid {len(branch_ids)} node(s) under `{root_id}`."
            )
            _sync_pipeline_targets_after_ui_change()
        except Exception:
            pass

    def _pipeline_studio_hard_delete_branch(
        *, pipeline_hash: str | None, root_id: str, clear_history: bool
    ) -> None:
        try:
            import time as _time

            root_id = root_id.strip() if isinstance(root_id, str) else ""
            if not root_id:
                return
            team_state = st.session_state.get("team_state", {})
            team_state = team_state if isinstance(team_state, dict) else {}
            ds = team_state.get("datasets")
            ds = ds if isinstance(ds, dict) else {}
            if root_id not in ds:
                st.session_state["pipeline_studio_history_notice"] = (
                    f"Hard delete skipped: `{root_id}` not found."
                )
                return
            child_idx = _build_children_index(ds)
            branch_ids = {root_id} | _descendants(root_id, child_idx)
            ds_new = {k: v for k, v in ds.items() if k not in branch_ids}
            if not ds_new:
                st.session_state["pipeline_studio_history_notice"] = (
                    "Hard delete skipped: would remove all datasets."
                )
                return

            active_id = team_state.get("active_dataset_id")
            active_id = active_id if isinstance(active_id, str) and active_id else None
            if active_id in branch_ids:
                best_id = None
                best_ts = -1.0
                for did, ent in ds_new.items():
                    if not isinstance(ent, dict):
                        continue
                    try:
                        ts = float(ent.get("created_ts") or 0.0)
                    except Exception:
                        ts = 0.0
                    if ts >= best_ts:
                        best_ts = ts
                        best_id = did
                active_id = best_id or next(iter(ds_new.keys()))

            team_state = dict(team_state)
            team_state["datasets"] = ds_new
            team_state["active_dataset_id"] = active_id
            st.session_state["team_state"] = team_state
            _persist_pipeline_studio_team_state(team_state=team_state)

            p_hash_clean = (
                pipeline_hash.strip()
                if isinstance(pipeline_hash, str) and pipeline_hash.strip()
                else None
            )
            if p_hash_clean:
                ui_h, ui_d = _pipeline_studio_get_registry_ui(
                    pipeline_hash=p_hash_clean
                )
                ui_h = set(ui_h) - branch_ids
                ui_d = set(ui_d) - branch_ids
                _pipeline_studio_set_registry_ui(
                    pipeline_hash=p_hash_clean,
                    hidden_ids=ui_h,
                    deleted_ids=ui_d,
                )

            pos_store = st.session_state.get("pipeline_studio_flow_positions")
            pos_store = pos_store if isinstance(pos_store, dict) else {}
            for did in list(branch_ids):
                pos_store.pop(str(did), None)
            st.session_state["pipeline_studio_flow_positions"] = pos_store

            if bool(clear_history):
                st.session_state["pipeline_studio_undo_stack"] = []
                st.session_state["pipeline_studio_redo_stack"] = []

            try:
                pipelines_new = _pipeline_studio_build_pipelines_from_team_state(
                    team_state
                )
                _update_pipeline_registry_store_for_pipelines(
                    pipelines=pipelines_new, datasets=ds_new
                )
            except Exception:
                pass

            st.session_state.pop("pipeline_studio_flow_state", None)
            st.session_state.pop("pipeline_studio_flow_signature", None)
            st.session_state.pop("pipeline_studio_flow_hidden_ids", None)
            st.session_state.pop("pipeline_studio_flow_layout_sig", None)
            st.session_state["pipeline_studio_flow_force_layout"] = True
            st.session_state["pipeline_studio_flow_fit_view_pending"] = True
            st.session_state["pipeline_studio_flow_ts"] = int(_time.time() * 1000)

            if isinstance(active_id, str) and active_id:
                st.session_state["pipeline_studio_node_id_pending"] = active_id
                st.session_state["pipeline_studio_autofollow_pending"] = False
            st.session_state["pipeline_studio_history_notice"] = (
                f"Permanently deleted {len(branch_ids)} node(s) under `{root_id}`."
            )
            _sync_pipeline_targets_after_ui_change()
        except Exception as e:
            st.session_state["pipeline_studio_history_notice"] = (
                f"Hard delete failed: {e}"
            )

    def _pipeline_studio_build_semantic_graph(
        *,
        pipeline_hash: str,
        node_ids: list[str],
        meta_by_id: dict,
        datasets: dict,
        hidden_ids: set[str],
        deleted_ids: set[str],
    ) -> dict:
        import time as _time

        node_ids = [str(x) for x in node_ids if isinstance(x, str) and x]
        nodes: dict[str, dict] = {}
        edges: list[dict[str, str]] = []
        node_set = set(node_ids)
        for did in node_ids:
            meta = meta_by_id.get(did) if isinstance(meta_by_id, dict) else {}
            meta = meta if isinstance(meta, dict) else {}
            entry = datasets.get(did) if isinstance(datasets, dict) else {}
            entry = entry if isinstance(entry, dict) else {}
            parents = _entry_parent_ids(entry)
            for pid in parents:
                if pid in node_set:
                    edges.append({"source": pid, "target": did})
            nodes[did] = {
                "id": did,
                "label": meta.get("label") or entry.get("label") or did,
                "stage": meta.get("stage") or entry.get("stage"),
                "transform_kind": meta.get("transform_kind"),
                "parent_ids": parents,
                "schema_hash": entry.get("schema_hash"),
                "fingerprint": entry.get("fingerprint"),
                "created_ts": entry.get("created_ts"),
                "hidden": did in hidden_ids,
                "deleted": did in deleted_ids,
            }
        return {
            "pipeline_hash": pipeline_hash,
            "nodes": nodes,
            "edges": edges,
            "hidden_ids": sorted(hidden_ids),
            "deleted_ids": sorted(deleted_ids),
            "updated_ts": _time.time(),
        }

    def _finalize_created_dataset(
        *,
        new_state: dict,
        new_id: str,
        prev_active_id: str | None,
        source: str,
        supersedes_node_id: str | None = None,
    ) -> None:
        st.session_state["team_state"] = new_state
        try:
            ds_new = new_state.get("datasets") if isinstance(new_state, dict) else None
            ds_new = ds_new if isinstance(ds_new, dict) else {}
            created_entry = ds_new.get(new_id)
            created_entry = created_entry if isinstance(created_entry, dict) else {}
            _pipeline_studio_push_history(
                {
                    "type": "create_dataset",
                    "dataset_id": new_id,
                    "dataset_entry": created_entry,
                    "prev_active_dataset_id": prev_active_id,
                    "source": source,
                }
            )
        except Exception:
            pass
        try:
            ds_new = new_state.get("datasets") if isinstance(new_state, dict) else None
            ds_new = ds_new if isinstance(ds_new, dict) else {}
            active_new = (
                new_state.get("active_dataset_id")
                if isinstance(new_state, dict)
                else None
            )
            active_new = active_new if isinstance(active_new, str) else None
            pipelines_new = {
                "model": build_pipeline_snapshot(ds_new, active_dataset_id=active_new),
                "active": build_pipeline_snapshot(
                    ds_new, active_dataset_id=active_new, target="active"
                ),
                "latest": build_pipeline_snapshot(
                    ds_new, active_dataset_id=active_new, target="latest"
                ),
            }
            _update_pipeline_registry_store_for_pipelines(
                pipelines=pipelines_new, datasets=ds_new
            )
        except Exception:
            pass
        _persist_pipeline_studio_team_state(team_state=new_state)
        st.session_state["pipeline_studio_node_id_pending"] = new_id
        st.session_state["pipeline_studio_autofollow_pending"] = True
        st.session_state["pipeline_studio_run_success"] = new_id
        if isinstance(supersedes_node_id, str) and supersedes_node_id.strip():
            try:
                ds_new = (
                    new_state.get("datasets") if isinstance(new_state, dict) else None
                )
                ds_new = ds_new if isinstance(ds_new, dict) else {}
                child_idx = _build_children_index(ds_new)
                stale_ids = sorted(_descendants(supersedes_node_id, child_idx))
                st.session_state["pipeline_studio_last_replacement"] = {
                    "old_id": supersedes_node_id,
                    "new_id": new_id,
                }
                st.session_state["pipeline_studio_stale_ids"] = stale_ids
            except Exception:
                pass

    def _pipeline_studio_relink_child_parent(
        *,
        child_id: str,
        old_parent_id: str,
        new_parent_id: str,
    ) -> None:
        try:
            child_id = child_id.strip() if isinstance(child_id, str) else ""
            old_parent_id = (
                old_parent_id.strip() if isinstance(old_parent_id, str) else ""
            )
            new_parent_id = (
                new_parent_id.strip() if isinstance(new_parent_id, str) else ""
            )
            if not child_id or not old_parent_id or not new_parent_id:
                return

            team_state = st.session_state.get("team_state", {})
            team_state = team_state if isinstance(team_state, dict) else {}
            datasets = team_state.get("datasets")
            datasets = datasets if isinstance(datasets, dict) else {}
            entry_obj = datasets.get(child_id)
            entry_obj = entry_obj if isinstance(entry_obj, dict) else {}
            if not entry_obj:
                st.session_state["pipeline_studio_run_error"] = (
                    f"Cannot insert: dataset `{child_id}` not found."
                )
                return

            parents = _entry_parent_ids(entry_obj)
            if old_parent_id not in parents:
                st.session_state["pipeline_studio_run_error"] = (
                    f"Cannot insert: `{child_id}` is not downstream of `{old_parent_id}`."
                )
                return

            new_parents = [
                new_parent_id if pid == old_parent_id else pid for pid in parents
            ]
            if new_parent_id not in new_parents:
                new_parents = [new_parent_id] + [
                    pid for pid in parents if pid != old_parent_id
                ]

            updated = dict(entry_obj)
            updated["parent_ids"] = new_parents
            updated["parent_id"] = new_parent_id
            datasets = dict(datasets)
            datasets[child_id] = updated
            team_state = dict(team_state)
            team_state["datasets"] = datasets
            st.session_state["team_state"] = team_state
            try:
                pipelines_new = _pipeline_studio_build_pipelines_from_team_state(
                    team_state
                )
                _update_pipeline_registry_store_for_pipelines(
                    pipelines=pipelines_new, datasets=datasets
                )
            except Exception:
                pass

            child_idx = _build_children_index(datasets)
            stale_ids = sorted({child_id} | _descendants(child_id, child_idx))
            st.session_state["pipeline_studio_stale_ids"] = stale_ids
            st.session_state["pipeline_studio_history_notice"] = (
                f"Inserted node between `{old_parent_id}` and `{child_id}`. "
                f"Marked {len(stale_ids)} node(s) stale."
            )
            st.session_state["pipeline_studio_flow_force_layout"] = True
            st.session_state["pipeline_studio_flow_fit_view_pending"] = True
        except Exception as e:
            st.session_state["pipeline_studio_run_error"] = str(e)

    def _run_python_function_draft(*, node_id: str, editor_key: str) -> None:
        """
        Run a user-edited python_function draft against the node's parent dataset and
        register a new dataset in `st.session_state.team_state`.
        """
        try:
            st.session_state.pop("pipeline_studio_run_error", None)
            st.session_state.pop("pipeline_studio_run_success", None)

            if not isinstance(node_id, str) or not node_id:
                return
            team_state = st.session_state.get("team_state", {})
            team_state = team_state if isinstance(team_state, dict) else {}
            prev_active_id = team_state.get("active_dataset_id")
            prev_active_id = (
                prev_active_id
                if isinstance(prev_active_id, str) and prev_active_id
                else None
            )
            datasets = team_state.get("datasets")
            datasets = datasets if isinstance(datasets, dict) else {}
            entry_obj = datasets.get(node_id)
            entry_obj = entry_obj if isinstance(entry_obj, dict) else {}

            prov = (
                entry_obj.get("provenance")
                if isinstance(entry_obj.get("provenance"), dict)
                else {}
            )
            transform = (
                prov.get("transform") if isinstance(prov.get("transform"), dict) else {}
            )
            kind = str(transform.get("kind") or "")
            if kind != "python_function":
                st.session_state["pipeline_studio_run_error"] = (
                    "Only `python_function` nodes can be executed from Pipeline Studio right now."
                )
                return

            parents = _entry_parent_ids(entry_obj)
            parent_id = parents[0] if parents else None
            if not isinstance(parent_id, str) or not parent_id:
                st.session_state["pipeline_studio_run_error"] = (
                    "Could not find a parent dataset for this node."
                )
                return

            parent_entry = datasets.get(parent_id)
            parent_entry = parent_entry if isinstance(parent_entry, dict) else {}
            df_in = _dataset_entry_to_df(parent_entry)
            if df_in is None:
                st.session_state["pipeline_studio_run_error"] = (
                    f"Parent dataset `{parent_id}` has no tabular data to run against."
                )
                return

            draft_code = st.session_state.get(editor_key)
            draft_code = draft_code if isinstance(draft_code, str) else ""
            draft_code = draft_code.strip()
            if not draft_code:
                st.session_state["pipeline_studio_run_error"] = (
                    "Draft code is empty; nothing to run."
                )
                return

            fn_name = (
                transform.get("function_name")
                if isinstance(transform.get("function_name"), str)
                else None
            )
            fn_name = fn_name.strip() if isinstance(fn_name, str) else None
            inferred = _infer_first_def_name(draft_code)
            try:
                out, fn_used = _exec_python_transform(
                    code=draft_code, df_in=df_in, fn_name_hint=fn_name or inferred
                )
            except Exception as e:
                st.session_state["pipeline_studio_run_error"] = str(e)
                return

            import hashlib

            code_sha = hashlib.sha256(draft_code.encode("utf-8")).hexdigest()
            stage = entry_obj.get("stage")
            stage = stage if isinstance(stage, str) and stage else "custom"
            label = entry_obj.get("label")
            label = label if isinstance(label, str) and label else node_id

            new_state, new_id = _pipeline_studio_register_dataset(
                team_state=team_state,
                data=out,
                stage=stage,
                label=f"{label}_edited",
                created_by="Pipeline_Studio",
                provenance={
                    "source_type": "pipeline_studio",
                    "source": f"rerun:{node_id}",
                    "user_request": "Pipeline Studio: run edited transform draft",
                    "transform": {
                        "kind": "python_function",
                        "supersedes_dataset_id": node_id,
                        "function_name": fn_used or fn_name or inferred or None,
                        "function_code": draft_code[:12000],
                        "code_sha256": code_sha,
                    },
                },
                parent_id=parent_id,
                parent_ids=[parent_id],
                make_active=True,
            )
            _finalize_created_dataset(
                new_state=new_state,
                new_id=new_id,
                prev_active_id=prev_active_id,
                source="pipeline_studio_run_draft",
                supersedes_node_id=node_id,
            )
        except Exception as e:
            st.session_state["pipeline_studio_run_error"] = str(e)

    def _pipeline_studio_create_manual_python_node(
        *,
        parent_id: str,
        stage: str,
        label: str,
        code: str,
        insert_child_id: str | None = None,
    ) -> None:
        try:
            st.session_state.pop("pipeline_studio_run_error", None)
            st.session_state.pop("pipeline_studio_run_success", None)

            parent_id = parent_id.strip() if isinstance(parent_id, str) else ""
            if not parent_id:
                st.session_state["pipeline_studio_run_error"] = (
                    "Select a parent dataset to run against."
                )
                return

            team_state = st.session_state.get("team_state", {})
            team_state = team_state if isinstance(team_state, dict) else {}
            prev_active_id = team_state.get("active_dataset_id")
            prev_active_id = (
                prev_active_id
                if isinstance(prev_active_id, str) and prev_active_id
                else None
            )
            datasets = team_state.get("datasets")
            datasets = datasets if isinstance(datasets, dict) else {}
            parent_entry = datasets.get(parent_id)
            parent_entry = parent_entry if isinstance(parent_entry, dict) else {}
            df_in = _dataset_entry_to_df(parent_entry)
            if df_in is None:
                st.session_state["pipeline_studio_run_error"] = (
                    f"Parent dataset `{parent_id}` has no tabular data to run against."
                )
                return

            code = code if isinstance(code, str) else ""
            try:
                out, fn_used = _exec_python_transform(code=code, df_in=df_in)
            except Exception as e:
                st.session_state["pipeline_studio_run_error"] = str(e)
                return

            import hashlib

            code = code.strip()
            code_sha = hashlib.sha256(code.encode("utf-8")).hexdigest() if code else ""

            stage = _normalize_pipeline_stage(stage or "custom")
            label = label.strip() if isinstance(label, str) else ""
            if not label:
                label = f"{stage}_manual"

            make_active = not bool(insert_child_id)
            new_state, new_id = _pipeline_studio_register_dataset(
                team_state=team_state,
                data=out,
                stage=stage,
                label=label,
                created_by="Pipeline_Studio",
                provenance={
                    "source_type": "pipeline_studio",
                    "source": f"manual:{parent_id}",
                    "user_request": "Pipeline Studio: manual python transform",
                    "transform": {
                        "kind": "python_function",
                        "function_name": fn_used or None,
                        "function_code": code[:12000],
                        "code_sha256": code_sha,
                    },
                },
                parent_id=parent_id,
                parent_ids=[parent_id],
                make_active=make_active,
            )
            _finalize_created_dataset(
                new_state=new_state,
                new_id=new_id,
                prev_active_id=prev_active_id,
                source="pipeline_studio_manual_python_node",
            )
            if isinstance(insert_child_id, str) and insert_child_id.strip():
                _pipeline_studio_relink_child_parent(
                    child_id=insert_child_id,
                    old_parent_id=parent_id,
                    new_parent_id=new_id,
                )
            st.session_state["pipeline_studio_manual_last_created_id"] = new_id
            # Ensure the newly created node is visible immediately even when the user is
            # viewing the "Model (latest feature)" pipeline target (which only follows `stage=feature`).
            if insert_child_id:
                st.session_state["pipeline_studio_target_pending"] = "Active dataset"
            elif stage != "feature":
                st.session_state["pipeline_studio_target_pending"] = "Active dataset"
            st.session_state["pipeline_studio_manual_node_open"] = False
            st.session_state["pipeline_studio_flow_force_layout"] = True
            st.session_state["pipeline_studio_flow_fit_view_pending"] = True
        except Exception as e:
            st.session_state["pipeline_studio_run_error"] = str(e)

    def _pipeline_studio_validate_readonly_sql(
        sql_text: str,
    ) -> tuple[str | None, str | None]:
        sql_text = sql_text if isinstance(sql_text, str) else ""
        sql_text = sql_text.strip()
        if not sql_text:
            return None, "SQL is empty; nothing to run."

        parts = [p.strip() for p in sql_text.split(";")]
        non_empty = [p for p in parts if p]
        if len(non_empty) > 1:
            return None, "Only single-statement queries are allowed in Pipeline Studio."
        sql_text = non_empty[0] if non_empty else ""
        first = re.sub(r"^\s*\(+\s*", "", sql_text).strip().lower()
        if not re.match(r"^(select|with|pragma|explain)\b", first):
            return (
                None,
                "Only read-only queries are allowed (SELECT/WITH/PRAGMA/EXPLAIN).",
            )
        return sql_text, None

    def _pipeline_studio_create_manual_sql_node(
        *,
        parent_id: str,
        stage: str,
        label: str,
        sql_text: str,
        insert_child_id: str | None = None,
    ) -> None:
        try:
            st.session_state.pop("pipeline_studio_run_error", None)
            st.session_state.pop("pipeline_studio_run_success", None)

            parent_id = parent_id.strip() if isinstance(parent_id, str) else ""
            if not parent_id:
                st.session_state["pipeline_studio_run_error"] = (
                    "Select a parent dataset to attach this SQL query to."
                )
                return

            sql_text, err = _pipeline_studio_validate_readonly_sql(sql_text)
            if err:
                st.session_state["pipeline_studio_run_error"] = err
                return

            team_state = st.session_state.get("team_state", {})
            team_state = team_state if isinstance(team_state, dict) else {}
            prev_active_id = team_state.get("active_dataset_id")
            prev_active_id = (
                prev_active_id
                if isinstance(prev_active_id, str) and prev_active_id
                else None
            )

            sql_url = st.session_state.get("sql_url", DEFAULT_SQL_URL)
            sql_url = (sql_url or DEFAULT_SQL_URL).strip() or DEFAULT_SQL_URL

            df_out = pd.read_sql_query(sql_text, sql.create_engine(sql_url))
            if not isinstance(df_out, pd.DataFrame):
                st.session_state["pipeline_studio_run_error"] = (
                    "SQL query did not return a pandas DataFrame."
                )
                return

            import hashlib

            sql_sha = hashlib.sha256(sql_text.encode("utf-8")).hexdigest()
            stage = _normalize_pipeline_stage(stage or "sql")
            label = label.strip() if isinstance(label, str) else ""
            if not label:
                label = f"{stage}_manual"

            make_active = not bool(insert_child_id)
            new_state, new_id = _pipeline_studio_register_dataset(
                team_state=team_state,
                data=df_out,
                stage=stage,
                label=label,
                created_by="Pipeline_Studio",
                provenance={
                    "source_type": "pipeline_studio",
                    "source": f"manual_sql:{parent_id}",
                    "user_request": "Pipeline Studio: manual SQL query",
                    "transform": {
                        "kind": "sql_query",
                        "sql_query_code": sql_text[:12000],
                        "sql_sha256": sql_sha,
                    },
                },
                parent_id=parent_id,
                parent_ids=[parent_id],
                make_active=make_active,
            )
            _finalize_created_dataset(
                new_state=new_state,
                new_id=new_id,
                prev_active_id=prev_active_id,
                source="pipeline_studio_manual_sql_node",
            )
            if isinstance(insert_child_id, str) and insert_child_id.strip():
                _pipeline_studio_relink_child_parent(
                    child_id=insert_child_id,
                    old_parent_id=parent_id,
                    new_parent_id=new_id,
                )
            st.session_state["pipeline_studio_manual_last_created_id"] = new_id
            if insert_child_id:
                st.session_state["pipeline_studio_target_pending"] = "Active dataset"
            elif stage != "feature":
                st.session_state["pipeline_studio_target_pending"] = "Active dataset"
            st.session_state["pipeline_studio_manual_node_open"] = False
            st.session_state["pipeline_studio_flow_force_layout"] = True
            st.session_state["pipeline_studio_flow_fit_view_pending"] = True
        except Exception as e:
            st.session_state["pipeline_studio_run_error"] = str(e)

    def _pipeline_studio_create_manual_merge_node(
        *,
        parent_ids: list[str],
        stage: str,
        label: str,
        code: str,
        insert_child_id: str | None = None,
    ) -> None:
        try:
            st.session_state.pop("pipeline_studio_run_error", None)
            st.session_state.pop("pipeline_studio_run_success", None)

            parent_ids = [
                p for p in (parent_ids or []) if isinstance(p, str) and p.strip()
            ]
            if len(parent_ids) < 2:
                st.session_state["pipeline_studio_run_error"] = (
                    "Select at least two parent datasets to merge."
                )
                return

            team_state = st.session_state.get("team_state", {})
            team_state = team_state if isinstance(team_state, dict) else {}
            prev_active_id = team_state.get("active_dataset_id")
            prev_active_id = (
                prev_active_id
                if isinstance(prev_active_id, str) and prev_active_id
                else None
            )
            datasets = team_state.get("datasets")
            datasets = datasets if isinstance(datasets, dict) else {}

            parent_dfs: list[pd.DataFrame] = []
            for pid in parent_ids:
                p_entry = datasets.get(pid)
                p_entry = p_entry if isinstance(p_entry, dict) else {}
                df_in = _dataset_entry_to_df(p_entry)
                if df_in is None:
                    st.session_state["pipeline_studio_run_error"] = (
                        f"Parent dataset `{pid}` has no tabular data to run against."
                    )
                    return
                parent_dfs.append(df_in)

            code = code if isinstance(code, str) else ""
            code = code.strip()
            if not code:
                st.session_state["pipeline_studio_run_error"] = (
                    "Merge code is empty; nothing to run."
                )
                return

            exec_globals: dict[str, object] = {"pd": pd}
            for i, df_i in enumerate(parent_dfs):
                exec_globals[f"df_{i}"] = df_i
            exec_locals: dict[str, object] = {}
            exec(code, exec_globals, exec_locals)

            out = exec_locals["df"] if "df" in exec_locals else exec_globals.get("df")
            if isinstance(out, tuple) and out:
                out = out[0]
            if not isinstance(out, pd.DataFrame):
                try:
                    if isinstance(out, dict):
                        out = pd.DataFrame.from_dict(out)
                    elif isinstance(out, list):
                        out = pd.DataFrame(out)
                except Exception:
                    out = None
            if not isinstance(out, pd.DataFrame):
                st.session_state["pipeline_studio_run_error"] = (
                    "Merge code did not produce a pandas DataFrame named `df`."
                )
                return

            import hashlib

            code_sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
            stage = _normalize_pipeline_stage(stage or "custom")
            label = label.strip() if isinstance(label, str) else ""
            if not label:
                label = f"{stage}_manual"

            new_state, new_id = _pipeline_studio_register_dataset(
                team_state=team_state,
                data=out,
                stage=stage,
                label=label,
                created_by="Pipeline_Studio",
                provenance={
                    "source_type": "pipeline_studio",
                    "source": f"manual_merge:{parent_ids[0]}",
                    "user_request": "Pipeline Studio: manual merge",
                    "transform": {
                        "kind": "python_merge",
                        "merge_code": code[:12000],
                        "code_sha256": code_sha,
                    },
                },
                parent_id=parent_ids[0],
                parent_ids=parent_ids,
                make_active=True,
            )
            _finalize_created_dataset(
                new_state=new_state,
                new_id=new_id,
                prev_active_id=prev_active_id,
                source="pipeline_studio_manual_merge_node",
            )
            if isinstance(insert_child_id, str) and insert_child_id.strip():
                _pipeline_studio_relink_child_parent(
                    child_id=insert_child_id,
                    old_parent_id=parent_ids[0],
                    new_parent_id=new_id,
                )
            st.session_state["pipeline_studio_manual_last_created_id"] = new_id
            if stage != "feature":
                st.session_state["pipeline_studio_target_pending"] = "Active dataset"
            st.session_state["pipeline_studio_manual_node_open"] = False
            st.session_state["pipeline_studio_flow_force_layout"] = True
            st.session_state["pipeline_studio_flow_fit_view_pending"] = True
        except Exception as e:
            st.session_state["pipeline_studio_run_error"] = str(e)

    def _run_python_merge_draft(*, node_id: str, editor_key: str) -> None:
        """
        Run a user-edited python_merge draft against the node's parent datasets and
        register a new dataset in `st.session_state.team_state`.
        """
        try:
            st.session_state.pop("pipeline_studio_run_error", None)
            st.session_state.pop("pipeline_studio_run_success", None)

            if not isinstance(node_id, str) or not node_id:
                return
            team_state = st.session_state.get("team_state", {})
            team_state = team_state if isinstance(team_state, dict) else {}
            prev_active_id = team_state.get("active_dataset_id")
            prev_active_id = (
                prev_active_id
                if isinstance(prev_active_id, str) and prev_active_id
                else None
            )
            datasets = team_state.get("datasets")
            datasets = datasets if isinstance(datasets, dict) else {}
            entry_obj = datasets.get(node_id)
            entry_obj = entry_obj if isinstance(entry_obj, dict) else {}

            prov = (
                entry_obj.get("provenance")
                if isinstance(entry_obj.get("provenance"), dict)
                else {}
            )
            transform = (
                prov.get("transform") if isinstance(prov.get("transform"), dict) else {}
            )
            kind = str(transform.get("kind") or "")
            if kind != "python_merge":
                st.session_state["pipeline_studio_run_error"] = (
                    "Only `python_merge` nodes can be executed with this action."
                )
                return

            parents = _entry_parent_ids(entry_obj)
            if len(parents) < 2:
                st.session_state["pipeline_studio_run_error"] = (
                    "Merge draft requires 2+ parent datasets."
                )
                return

            parent_dfs: list[pd.DataFrame] = []
            for pid in parents:
                p_entry = datasets.get(pid)
                p_entry = p_entry if isinstance(p_entry, dict) else {}
                df_in = _dataset_entry_to_df(p_entry)
                if df_in is None:
                    st.session_state["pipeline_studio_run_error"] = (
                        f"Parent dataset `{pid}` has no tabular data to run against."
                    )
                    return
                parent_dfs.append(df_in)

            draft_code = st.session_state.get(editor_key)
            draft_code = draft_code if isinstance(draft_code, str) else ""
            draft_code = draft_code.strip()
            if not draft_code:
                st.session_state["pipeline_studio_run_error"] = (
                    "Draft code is empty; nothing to run."
                )
                return

            exec_globals: dict[str, object] = {"pd": pd}
            for i, df_i in enumerate(parent_dfs):
                exec_globals[f"df_{i}"] = df_i
            exec_locals: dict[str, object] = {}
            exec(draft_code, exec_globals, exec_locals)

            out = exec_locals["df"] if "df" in exec_locals else exec_globals.get("df")
            if isinstance(out, tuple) and out:
                out = out[0]
            if not isinstance(out, pd.DataFrame):
                try:
                    if isinstance(out, dict):
                        out = pd.DataFrame.from_dict(out)
                    elif isinstance(out, list):
                        out = pd.DataFrame(out)
                except Exception:
                    out = None
            if not isinstance(out, pd.DataFrame):
                st.session_state["pipeline_studio_run_error"] = (
                    "Merge draft did not produce a pandas DataFrame named `df`."
                )
                return

            import hashlib

            code_sha = hashlib.sha256(draft_code.encode("utf-8")).hexdigest()
            stage = entry_obj.get("stage")
            stage = stage if isinstance(stage, str) and stage else "custom"
            label = entry_obj.get("label")
            label = label if isinstance(label, str) and label else node_id

            merge_meta = (
                transform.get("merge")
                if isinstance(transform.get("merge"), dict)
                else None
            )

            new_state, new_id = _pipeline_studio_register_dataset(
                team_state=team_state,
                data=out,
                stage=stage,
                label=f"{label}_edited",
                created_by="Pipeline_Studio",
                provenance={
                    "source_type": "pipeline_studio",
                    "source": f"rerun:{node_id}",
                    "user_request": "Pipeline Studio: run edited merge draft",
                    "transform": {
                        "kind": "python_merge",
                        "supersedes_dataset_id": node_id,
                        "merge": merge_meta,
                        "merge_code": draft_code[:12000],
                        "code_sha256": code_sha,
                    },
                },
                parent_id=parents[0],
                parent_ids=parents,
                make_active=True,
            )
            _finalize_created_dataset(
                new_state=new_state,
                new_id=new_id,
                prev_active_id=prev_active_id,
                source="pipeline_studio_run_merge_draft",
                supersedes_node_id=node_id,
            )
        except Exception as e:
            st.session_state["pipeline_studio_run_error"] = str(e)

    def _run_sql_query_draft(*, node_id: str, editor_key: str) -> None:
        """
        Run a user-edited sql_query draft (read-only) and register a new dataset in `st.session_state.team_state`.
        """
        try:
            st.session_state.pop("pipeline_studio_run_error", None)
            st.session_state.pop("pipeline_studio_run_success", None)

            if not isinstance(node_id, str) or not node_id:
                return
            team_state = st.session_state.get("team_state", {})
            team_state = team_state if isinstance(team_state, dict) else {}
            prev_active_id = team_state.get("active_dataset_id")
            prev_active_id = (
                prev_active_id
                if isinstance(prev_active_id, str) and prev_active_id
                else None
            )
            datasets = team_state.get("datasets")
            datasets = datasets if isinstance(datasets, dict) else {}
            entry_obj = datasets.get(node_id)
            entry_obj = entry_obj if isinstance(entry_obj, dict) else {}

            prov = (
                entry_obj.get("provenance")
                if isinstance(entry_obj.get("provenance"), dict)
                else {}
            )
            transform = (
                prov.get("transform") if isinstance(prov.get("transform"), dict) else {}
            )
            kind = str(transform.get("kind") or "")
            if kind != "sql_query":
                st.session_state["pipeline_studio_run_error"] = (
                    "Only `sql_query` nodes can be executed with this action."
                )
                return

            draft_sql = st.session_state.get(editor_key)
            draft_sql = draft_sql if isinstance(draft_sql, str) else ""
            sql_text, err = _pipeline_studio_validate_readonly_sql(draft_sql)
            if err:
                st.session_state["pipeline_studio_run_error"] = err.replace(
                    "SQL is empty", "Draft SQL is empty"
                )
                return

            sql_url = st.session_state.get("sql_url", DEFAULT_SQL_URL)
            sql_url = (sql_url or DEFAULT_SQL_URL).strip() or DEFAULT_SQL_URL

            df_out = pd.read_sql_query(sql_text, sql.create_engine(sql_url))
            if not isinstance(df_out, pd.DataFrame):
                st.session_state["pipeline_studio_run_error"] = (
                    "SQL query did not return a pandas DataFrame."
                )
                return

            import hashlib

            sql_sha = hashlib.sha256(sql_text.encode("utf-8")).hexdigest()
            stage = entry_obj.get("stage")
            stage = stage if isinstance(stage, str) and stage else "sql"
            label = entry_obj.get("label")
            label = label if isinstance(label, str) and label else node_id

            parents = _entry_parent_ids(entry_obj)

            new_state, new_id = _pipeline_studio_register_dataset(
                team_state=team_state,
                data=df_out,
                stage=stage,
                label=f"{label}_edited",
                created_by="Pipeline_Studio",
                provenance={
                    "source_type": "pipeline_studio",
                    "source": f"rerun:{node_id}",
                    "user_request": "Pipeline Studio: run edited SQL draft",
                    "transform": {
                        "kind": "sql_query",
                        "supersedes_dataset_id": node_id,
                        "sql_query_code": sql_text[:12000],
                        "sql_sha256": sql_sha,
                    },
                },
                parent_id=parents[0] if parents else None,
                parent_ids=parents,
                make_active=True,
            )
            _finalize_created_dataset(
                new_state=new_state,
                new_id=new_id,
                prev_active_id=prev_active_id,
                source="pipeline_studio_run_sql_draft",
                supersedes_node_id=node_id,
            )
        except Exception as e:
            st.session_state["pipeline_studio_run_error"] = str(e)

    def _run_downstream_transforms(
        superseded_node_id: str, replacement_node_id: str
    ) -> None:
        """
        Best-effort re-run of downstream transform nodes starting from `superseded_node_id`,
        using `replacement_node_id` as the new upstream input.
        """
        try:
            st.session_state.pop("pipeline_studio_run_error", None)
            st.session_state.pop("pipeline_studio_run_success", None)

            superseded_node_id = (
                superseded_node_id.strip()
                if isinstance(superseded_node_id, str)
                else ""
            )
            replacement_node_id = (
                replacement_node_id.strip()
                if isinstance(replacement_node_id, str)
                else ""
            )
            if not superseded_node_id or not replacement_node_id:
                return

            team_state = st.session_state.get("team_state", {})
            team_state = team_state if isinstance(team_state, dict) else {}
            prev_active_id = team_state.get("active_dataset_id")
            prev_active_id = (
                prev_active_id
                if isinstance(prev_active_id, str) and prev_active_id
                else None
            )
            datasets = team_state.get("datasets")
            datasets = datasets if isinstance(datasets, dict) else {}

            if superseded_node_id not in datasets:
                st.session_state["pipeline_studio_run_error"] = (
                    f"Run downstream failed: `{superseded_node_id}` is not a known dataset id."
                )
                return
            if replacement_node_id not in datasets:
                st.session_state["pipeline_studio_run_error"] = (
                    f"Run downstream failed: `{replacement_node_id}` is not a known dataset id."
                )
                return

            child_idx = _build_children_index(datasets)
            stale_set = _descendants(superseded_node_id, child_idx)
            if not stale_set:
                st.session_state["pipeline_studio_history_notice"] = (
                    f"No downstream nodes to rerun from `{superseded_node_id}`."
                )
                st.session_state["pipeline_studio_stale_ids"] = []
                return

            parents_by_id: dict[str, list[str]] = {}
            for did, ent in datasets.items():
                if not isinstance(did, str) or not did or not isinstance(ent, dict):
                    continue
                parents_by_id[did] = _entry_parent_ids(ent)

            # Topo-sort within the downstream subgraph.
            def _created_ts(did: str) -> float:
                ent = datasets.get(did)
                ent = ent if isinstance(ent, dict) else {}
                try:
                    return float(ent.get("created_ts") or 0.0)
                except Exception:
                    return 0.0

            indeg: dict[str, int] = {}
            for did in stale_set:
                indeg[did] = sum(
                    1 for pid in parents_by_id.get(did, []) if pid in stale_set
                )
            queue = [did for did in stale_set if indeg.get(did, 0) == 0]
            queue.sort(key=_created_ts)
            order: list[str] = []
            while queue:
                did = queue.pop(0)
                order.append(did)
                for child in child_idx.get(did, set()):
                    if child not in stale_set:
                        continue
                    indeg[child] = max(0, int(indeg.get(child, 0)) - 1)
                    if indeg[child] == 0:
                        queue.append(child)
                        queue.sort(key=_created_ts)
            if len(order) < len(stale_set):
                remaining = [did for did in stale_set if did not in set(order)]
                remaining.sort(key=_created_ts)
                order.extend(remaining)

            replacement_map: dict[str, str] = {superseded_node_id: replacement_node_id}
            created_ids: list[str] = []
            created_entries_by_id: dict[str, dict] = {}
            skipped: list[dict[str, str]] = []
            failed: list[dict[str, str]] = []

            working_state = dict(team_state)
            working_datasets = dict(datasets)
            working_state["datasets"] = working_datasets

            def _safe_df_from_id(did: str) -> pd.DataFrame | None:
                ent = working_datasets.get(did)
                ent = ent if isinstance(ent, dict) else {}
                return _dataset_entry_to_df(ent)

            def _exec_python_function(
                code: str, *, fn_name_hint: str | None, df_in: pd.DataFrame
            ) -> pd.DataFrame:
                fn_name_hint = (
                    fn_name_hint.strip() if isinstance(fn_name_hint, str) else None
                )
                inferred = _infer_first_def_name(code)
                exec_globals: dict[str, object] = {"pd": pd}
                exec_locals: dict[str, object] = {}
                exec(code, exec_globals, exec_locals)
                fn = None
                for candidate in [fn_name_hint, inferred]:
                    if not candidate:
                        continue
                    obj = exec_locals.get(candidate) or exec_globals.get(candidate)
                    if callable(obj):
                        fn = obj
                        break
                if fn is None:
                    for obj in exec_locals.values():
                        if callable(obj):
                            fn = obj
                            break
                if fn is None:
                    raise ValueError("Could not find a callable function in the code.")
                out = fn(df_in)
                if isinstance(out, tuple) and out:
                    out = out[0]
                if not isinstance(out, pd.DataFrame):
                    if isinstance(out, dict):
                        out = pd.DataFrame.from_dict(out)
                    elif isinstance(out, list):
                        out = pd.DataFrame(out)
                if not isinstance(out, pd.DataFrame):
                    raise ValueError("Transform did not return a pandas DataFrame.")
                return out

            def _exec_python_merge(
                code: str, *, parent_dfs: list[pd.DataFrame]
            ) -> pd.DataFrame:
                exec_globals: dict[str, object] = {"pd": pd}
                for i, df_i in enumerate(parent_dfs):
                    exec_globals[f"df_{i}"] = df_i
                exec_locals: dict[str, object] = {}
                exec(code, exec_globals, exec_locals)
                out = (
                    exec_locals["df"] if "df" in exec_locals else exec_globals.get("df")
                )
                if isinstance(out, tuple) and out:
                    out = out[0]
                if not isinstance(out, pd.DataFrame):
                    if isinstance(out, dict):
                        out = pd.DataFrame.from_dict(out)
                    elif isinstance(out, list):
                        out = pd.DataFrame(out)
                if not isinstance(out, pd.DataFrame):
                    raise ValueError(
                        "Merge code did not produce a pandas DataFrame named `df`."
                    )
                return out

            def _normalize_readonly_sql(sql_text: str) -> str:
                sql_text = (sql_text or "").strip()
                parts = [p.strip() for p in sql_text.split(";")]
                non_empty = [p for p in parts if p]
                if len(non_empty) > 1:
                    raise ValueError("Only single-statement queries are allowed.")
                sql_text = non_empty[0] if non_empty else ""
                first = re.sub(r"^\s*\(+\s*", "", sql_text).strip().lower()
                if not re.match(r"^(select|with|pragma|explain)\b", first):
                    raise ValueError(
                        "Only read-only queries are allowed (SELECT/WITH/PRAGMA/EXPLAIN)."
                    )
                return sql_text

            for did in order:
                ent = datasets.get(did)
                ent = ent if isinstance(ent, dict) else {}
                prov = (
                    ent.get("provenance")
                    if isinstance(ent.get("provenance"), dict)
                    else {}
                )
                transform = (
                    prov.get("transform")
                    if isinstance(prov.get("transform"), dict)
                    else {}
                )
                kind = str(transform.get("kind") or "")
                if kind not in {"python_function", "python_merge", "sql_query"}:
                    skipped.append(
                        {
                            "node_id": did,
                            "kind": kind or "unknown",
                            "reason": "unsupported",
                        }
                    )
                    continue

                # Map parents: if a parent is within the stale subtree, require a replacement.
                orig_parents = parents_by_id.get(did, [])
                mapped_parents: list[str] = []
                blocked = None
                for pid in orig_parents:
                    if pid == superseded_node_id or pid in stale_set:
                        if pid not in replacement_map:
                            blocked = pid
                            break
                        mapped_parents.append(replacement_map[pid])
                    else:
                        mapped_parents.append(pid)
                if blocked:
                    skipped.append(
                        {
                            "node_id": did,
                            "kind": kind,
                            "reason": f"blocked_by:{blocked}",
                        }
                    )
                    continue

                try:
                    stage = ent.get("stage")
                    stage = stage if isinstance(stage, str) and stage else "custom"
                    label = ent.get("label")
                    label = label if isinstance(label, str) and label else did

                    if kind == "python_function":
                        code = transform.get("function_code")
                        code = code if isinstance(code, str) else ""
                        code = code.strip()
                        if not code:
                            skipped.append(
                                {"node_id": did, "kind": kind, "reason": "missing_code"}
                            )
                            continue
                        if not mapped_parents:
                            skipped.append(
                                {
                                    "node_id": did,
                                    "kind": kind,
                                    "reason": "missing_parent",
                                }
                            )
                            continue
                        parent_id = mapped_parents[0]
                        df_in = _safe_df_from_id(parent_id)
                        if df_in is None:
                            skipped.append(
                                {
                                    "node_id": did,
                                    "kind": kind,
                                    "reason": f"missing_data:{parent_id}",
                                }
                            )
                            continue
                        fn_name = (
                            transform.get("function_name")
                            if isinstance(transform.get("function_name"), str)
                            else None
                        )
                        out_df = _exec_python_function(
                            code, fn_name_hint=fn_name, df_in=df_in
                        )

                        new_state, new_id = _pipeline_studio_register_dataset(
                            team_state=working_state,
                            data=out_df,
                            stage=stage,
                            label=f"{label}_rerun",
                            created_by="Pipeline_Studio",
                            provenance={
                                "source_type": "pipeline_studio",
                                "source": f"rerun_downstream:{did}",
                                "user_request": "Pipeline Studio: run downstream transforms",
                                "transform": {
                                    "kind": "python_function",
                                    "supersedes_dataset_id": did,
                                    "function_name": fn_name,
                                    "function_code": code[:12000],
                                    "code_sha256": transform.get("code_sha256"),
                                },
                            },
                            parent_id=parent_id,
                            parent_ids=[parent_id],
                            make_active=False,
                        )
                    elif kind == "python_merge":
                        code = transform.get("merge_code")
                        code = code if isinstance(code, str) else ""
                        code = code.strip()
                        if not code:
                            skipped.append(
                                {"node_id": did, "kind": kind, "reason": "missing_code"}
                            )
                            continue
                        if len(mapped_parents) < 2:
                            skipped.append(
                                {
                                    "node_id": did,
                                    "kind": kind,
                                    "reason": "missing_parents",
                                }
                            )
                            continue
                        parent_dfs = []
                        for pid in mapped_parents:
                            df_in = _safe_df_from_id(pid)
                            if df_in is None:
                                raise ValueError(
                                    f"Missing tabular data for parent `{pid}`."
                                )
                            parent_dfs.append(df_in)
                        out_df = _exec_python_merge(code, parent_dfs=parent_dfs)
                        merge_meta = (
                            transform.get("merge")
                            if isinstance(transform.get("merge"), dict)
                            else None
                        )

                        new_state, new_id = _pipeline_studio_register_dataset(
                            team_state=working_state,
                            data=out_df,
                            stage=stage,
                            label=f"{label}_rerun",
                            created_by="Pipeline_Studio",
                            provenance={
                                "source_type": "pipeline_studio",
                                "source": f"rerun_downstream:{did}",
                                "user_request": "Pipeline Studio: run downstream transforms",
                                "transform": {
                                    "kind": "python_merge",
                                    "supersedes_dataset_id": did,
                                    "merge": merge_meta,
                                    "merge_code": code[:12000],
                                    "code_sha256": transform.get("code_sha256"),
                                },
                            },
                            parent_id=mapped_parents[0],
                            parent_ids=mapped_parents,
                            make_active=False,
                        )
                    else:  # sql_query
                        sql_code = transform.get("sql_query_code")
                        sql_code = sql_code if isinstance(sql_code, str) else ""
                        sql_code = _normalize_readonly_sql(sql_code)
                        sql_url = st.session_state.get("sql_url", DEFAULT_SQL_URL)
                        sql_url = (
                            sql_url or DEFAULT_SQL_URL
                        ).strip() or DEFAULT_SQL_URL
                        out_df = pd.read_sql_query(sql_code, sql.create_engine(sql_url))

                        new_state, new_id = _pipeline_studio_register_dataset(
                            team_state=working_state,
                            data=out_df,
                            stage=stage,
                            label=f"{label}_rerun",
                            created_by="Pipeline_Studio",
                            provenance={
                                "source_type": "pipeline_studio",
                                "source": f"rerun_downstream:{did}",
                                "user_request": "Pipeline Studio: run downstream transforms",
                                "transform": {
                                    "kind": "sql_query",
                                    "supersedes_dataset_id": did,
                                    "sql_query_code": sql_code[:12000],
                                    "sql_sha256": transform.get("sql_sha256"),
                                },
                            },
                            parent_id=mapped_parents[0] if mapped_parents else None,
                            parent_ids=mapped_parents,
                            make_active=False,
                        )

                    # Update working state for subsequent nodes.
                    working_state = new_state
                    working_datasets = (
                        new_state.get("datasets") if isinstance(new_state, dict) else {}
                    )
                    working_datasets = (
                        working_datasets if isinstance(working_datasets, dict) else {}
                    )
                    working_state["datasets"] = working_datasets
                    working_state["active_dataset_id"] = new_state.get(
                        "active_dataset_id"
                    )
                    replacement_map[did] = new_id
                    created_ids.append(new_id)
                    created_entries_by_id[new_id] = (
                        working_datasets.get(new_id)
                        if isinstance(working_datasets.get(new_id), dict)
                        else {}
                    )
                except Exception as e:
                    failed.append({"node_id": did, "kind": kind, "error": str(e)})

            # Update active dataset to the newest created node if any.
            if created_ids:
                working_state = dict(working_state)
                working_state["active_dataset_id"] = created_ids[-1]

            st.session_state["team_state"] = working_state

            # Persist semantic registry (best effort).
            try:
                ds_new = (
                    working_state.get("datasets")
                    if isinstance(working_state, dict)
                    else None
                )
                ds_new = ds_new if isinstance(ds_new, dict) else {}
                pipelines_new = _pipeline_studio_build_pipelines_from_team_state(
                    working_state
                )
                if pipelines_new:
                    _update_pipeline_registry_store_for_pipelines(
                        pipelines=pipelines_new, datasets=ds_new
                    )
            except Exception:
                pass

            # History: group undo/redo.
            if created_ids:
                _pipeline_studio_push_history(
                    {
                        "type": "create_datasets",
                        "dataset_ids": created_ids,
                        "dataset_entries_by_id": created_entries_by_id,
                        "prev_active_dataset_id": prev_active_id,
                        "source": "pipeline_studio_run_downstream",
                    }
                )
                st.session_state["pipeline_studio_node_id_pending"] = created_ids[-1]
                st.session_state["pipeline_studio_autofollow_pending"] = True
                st.session_state["pipeline_studio_run_success"] = created_ids[-1]

            try:
                import time as _time

                st.session_state["pipeline_studio_last_downstream_mapping"] = {
                    str(k): str(v)
                    for k, v in replacement_map.items()
                    if isinstance(k, str) and isinstance(v, str) and k and v
                }
                st.session_state["pipeline_studio_last_downstream_source"] = {
                    "old_id": superseded_node_id,
                    "new_id": replacement_node_id,
                    "created_ids": list(created_ids),
                    "ts": _time.time(),
                }
            except Exception:
                pass

            # Refresh stale list: remaining nodes that could not be rerun.
            rerun_old_nodes = set(replacement_map.keys()).intersection(stale_set)
            remaining_stale = sorted(stale_set - rerun_old_nodes)
            st.session_state["pipeline_studio_stale_ids"] = remaining_stale
            if remaining_stale:
                st.session_state["pipeline_studio_history_notice"] = (
                    f"Downstream run complete: created {len(created_ids)} dataset(s). "
                    f"Remaining stale: {len(remaining_stale)} (skipped/blocked)."
                )
            else:
                st.session_state["pipeline_studio_history_notice"] = (
                    f"Downstream run complete: created {len(created_ids)} dataset(s)."
                )

            if failed:
                st.session_state["pipeline_studio_run_error"] = (
                    f"Downstream rerun had {len(failed)} failure(s). "
                    "Open the latest node and check the code/provenance."
                )
                st.session_state["pipeline_studio_last_downstream_failures"] = failed
            if skipped:
                st.session_state["pipeline_studio_last_downstream_skipped"] = skipped
        except Exception as e:
            st.session_state["pipeline_studio_run_error"] = str(e)

    def _latest_detail_for_dataset_id(
        dataset_id: str, *, require_key: str | None = None
    ) -> dict | None:
        details = st.session_state.get("details") or []
        if not isinstance(details, list):
            return None
        for d in reversed(details):
            if not isinstance(d, dict):
                continue
            pipelines = (
                d.get("pipelines") if isinstance(d.get("pipelines"), dict) else {}
            )
            active_pipe = (
                pipelines.get("active")
                if isinstance(pipelines.get("active"), dict)
                else None
            )
            active_target_id = (
                active_pipe.get("target_dataset_id")
                if isinstance(active_pipe, dict)
                else None
            )
            if active_target_id == dataset_id:
                if require_key and not d.get(require_key):
                    continue
                return d
        return None

    def _render_copy_to_clipboard(text: str, *, label: str = "Copy") -> None:
        if not isinstance(text, str) or not text:
            st.info("Nothing to copy.")
            return
        try:
            payload = json.dumps(text)
        except Exception:
            payload = "''"
        components.html(
            "\n".join(
                [
                    "<style>",
                    "@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600&display=swap');",
                    "body {",
                    "  margin: 0;",
                    "  font-family: 'Source Sans 3', 'Source Sans Pro', sans-serif;",
                    "  -webkit-font-smoothing: antialiased;",
                    "}",
                    "#pipeline-copy-wrap { width: 100%; }",
                    "#pipeline-copy-btn {",
                    "  width: 100%;",
                    "  min-height: 2.2rem;",
                    "  padding: 0.25rem 0.8rem;",
                    "  border-radius: 0.5rem;",
                    "  border: 1px solid rgba(255, 255, 255, 0.18);",
                    "  background: rgba(18, 25, 34, 0.85);",
                    "  color: #e6edf3;",
                    "  cursor: pointer;",
                    "  font-size: 0.95rem;",
                    "  font-family: inherit;",
                    "  font-weight: 600;",
                    "  line-height: 1.1;",
                    "  display: inline-flex;",
                    "  align-items: center;",
                    "  justify-content: center;",
                    "  box-sizing: border-box;",
                    "}",
                    "#pipeline-copy-btn:hover {",
                    "  background: rgba(32, 42, 56, 0.95);",
                    "}",
                    "</style>",
                    '<div id="pipeline-copy-wrap">',
                    f'<button id="pipeline-copy-btn">{label}</button>',
                    "</div>",
                    "<script>",
                    f"const text = {payload};",
                    "const btn = document.getElementById('pipeline-copy-btn');",
                    "const original = btn ? btn.textContent : '';",
                    "async function doCopy() {",
                    "  try {",
                    "    if (navigator.clipboard && navigator.clipboard.writeText) {",
                    "      await navigator.clipboard.writeText(text);",
                    "    } else {",
                    "      const textarea = document.createElement('textarea');",
                    "      textarea.value = text;",
                    "      document.body.appendChild(textarea);",
                    "      textarea.select();",
                    "      document.execCommand('copy');",
                    "      document.body.removeChild(textarea);",
                    "    }",
                    "    if (btn) btn.textContent = 'Copied';",
                    "    setTimeout(() => { if (btn) btn.textContent = original; }, 1500);",
                    "  } catch (e) {",
                    "    if (btn) btn.textContent = 'Copy failed';",
                    "    setTimeout(() => { if (btn) btn.textContent = original; }, 2000);",
                    "  }",
                    "}",
                    "if (btn) btn.addEventListener('click', doCopy);",
                    "</script>",
                ]
            ),
            height=44,
        )

    if not studio_datasets:
        st.info("No pipeline yet. Load data and run a transform to build one.")
    else:
        target_options = [
            ("Model (latest feature)", "model"),
            ("Active dataset", "active"),
            ("Latest dataset", "latest"),
            ("All datasets", "all"),
        ]
        pending_target = st.session_state.pop("pipeline_studio_target_pending", None)
        if isinstance(pending_target, str):
            valid_labels = [k for k, _v in target_options]
            if pending_target in valid_labels:
                st.session_state["pipeline_studio_target"] = pending_target
        target_label = st.radio(
            "Pipeline target",
            options=[k for k, _v in target_options],
            index=0,
            horizontal=True,
            key="pipeline_studio_target",
        )
        target_key = dict(target_options).get(target_label, "model")
        pipe = build_pipeline_snapshot(
            studio_datasets, active_dataset_id=studio_active_id, target=target_key
        )
        lineage = pipe.get("lineage") if isinstance(pipe, dict) else None
        lineage = lineage if isinstance(lineage, list) else []

        if not lineage:
            st.info(
                "No pipeline available yet. Load data and run a transform (wrangle/clean/features/model/predict)."
            )
        else:
            meta_by_id = {
                str(x.get("id")): x
                for x in lineage
                if isinstance(x, dict) and x.get("id")
            }
            node_ids = [did for did in meta_by_id.keys() if did]

            left, right = st.columns([0.35, 0.65], gap="large")

            with left:
                pipeline_hash = (
                    pipe.get("pipeline_hash") if isinstance(pipe, dict) else None
                )
                pipeline_hash = (
                    pipeline_hash.strip()
                    if isinstance(pipeline_hash, str) and pipeline_hash.strip()
                    else ""
                )
                ui_hidden_ids, ui_deleted_ids = (
                    _pipeline_studio_get_registry_ui(pipeline_hash=pipeline_hash)
                    if pipeline_hash
                    else (set(), set())
                )
                if isinstance(pipe, dict):
                    target_id = pipe.get("target_dataset_id")
                    if isinstance(target_id, str) and target_id in set(
                        ui_hidden_ids
                    ) | set(ui_deleted_ids):
                        visible_ids = [
                            did
                            for did in node_ids
                            if did not in ui_hidden_ids and did not in ui_deleted_ids
                        ]
                        fallback_id = _pick_latest_dataset_id(
                            studio_datasets, visible_ids
                        )
                        if fallback_id:
                            pipe = dict(pipe)
                            pipe["target_dataset_id"] = fallback_id
                st.session_state["pipeline_studio_semantic_graph"] = (
                    _pipeline_studio_build_semantic_graph(
                        pipeline_hash=pipeline_hash,
                        node_ids=node_ids,
                        meta_by_id=meta_by_id,
                        datasets=studio_datasets,
                        hidden_ids=set(ui_hidden_ids),
                        deleted_ids=set(ui_deleted_ids),
                    )
                )
                target_display = (
                    f"all ({len(studio_datasets)} datasets)"
                    if str(pipe.get("target") or "").strip().lower() == "all"
                    else pipe.get("target_dataset_id")
                )
                st.markdown(
                    f"**Pipeline hash:** `{pipe.get('pipeline_hash')}`  \n"
                    f"**Target dataset id:** `{target_display}`  \n"
                    f"**Active dataset id:** `{pipe.get('active_dataset_id')}`"
                )
                show_hidden_pick = bool(
                    st.session_state.get("pipeline_studio_show_hidden", False)
                )
                show_deleted_pick = bool(
                    st.session_state.get("pipeline_studio_show_deleted", False)
                )
                ordered_ids = sorted(
                    studio_datasets.items(),
                    key=lambda kv: float(kv[1].get("created_ts") or 0.0)
                    if isinstance(kv[1], dict)
                    else 0.0,
                    reverse=True,
                )
                pick_ids = [
                    did
                    for did, _e in ordered_ids
                    if isinstance(did, str)
                    and (show_hidden_pick or did not in ui_hidden_ids)
                    and (show_deleted_pick or did not in ui_deleted_ids)
                ]
                if not pick_ids:
                    pick_ids = [did for did, _e in ordered_ids if isinstance(did, str)]

                def _fmt_studio_dataset(did: str) -> str:
                    e = studio_datasets.get(did)
                    if not isinstance(e, dict):
                        return str(did)
                    label = e.get("label") or did
                    stage = e.get("stage") or "dataset"
                    shape = e.get("shape")
                    shape_txt = f" {shape}" if shape else ""
                    status = []
                    if did in ui_deleted_ids:
                        status.append("deleted")
                    elif did in ui_hidden_ids:
                        status.append("hidden")
                    status_txt = f" ({', '.join(status)})" if status else ""
                    return f"{stage}: {label}{shape_txt} ({did}){status_txt}"

                def _set_active_dataset_studio(dataset_id: str | None) -> None:
                    dataset_id = (
                        dataset_id.strip()
                        if isinstance(dataset_id, str) and dataset_id.strip()
                        else None
                    )
                    if not dataset_id or dataset_id not in studio_datasets:
                        st.session_state["pipeline_studio_active_notice"] = (
                            "Select a valid dataset id."
                        )
                        return
                    team_state = st.session_state.get("team_state", {})
                    team_state = team_state if isinstance(team_state, dict) else {}
                    updated = dict(team_state)
                    updated["active_dataset_id"] = dataset_id
                    st.session_state["team_state"] = updated
                    _queue_active_dataset_override("")
                    st.session_state["pipeline_studio_node_id_pending"] = dataset_id
                    _persist_pipeline_studio_team_state(team_state=updated)
                    _sync_pipeline_targets_after_ui_change()
                    st.session_state["pipeline_studio_active_notice"] = (
                        f"Active dataset set to `{dataset_id}`."
                    )

                if pick_ids:
                    default_pick = st.session_state.get("pipeline_studio_node_id")
                    if (
                        not isinstance(default_pick, str)
                        or default_pick not in pick_ids
                    ):
                        default_pick = (
                            studio_active_id
                            if isinstance(studio_active_id, str)
                            and studio_active_id in pick_ids
                            else pick_ids[0]
                        )
                    pick_cols = st.columns([0.64, 0.18, 0.18], gap="small")
                    with pick_cols[0]:
                        picked_id = st.selectbox(
                            "Set active dataset",
                            options=pick_ids,
                            index=pick_ids.index(default_pick)
                            if default_pick in pick_ids
                            else 0,
                            format_func=_fmt_studio_dataset,
                            key="pipeline_studio_active_picker",
                        )
                    with pick_cols[1]:
                        if st.button(
                            "Set active",
                            key="pipeline_studio_active_set",
                            width="stretch",
                        ):
                            _set_active_dataset_studio(picked_id)
                    with pick_cols[2]:
                        target_id = (
                            pipe.get("target_dataset_id")
                            if isinstance(pipe, dict)
                            else None
                        )
                        if isinstance(target_id, str) and target_id in studio_datasets:
                            if st.button(
                                "Use target",
                                key="pipeline_studio_active_target",
                                width="stretch",
                            ):
                                _set_active_dataset_studio(target_id)
                        else:
                            st.button(
                                "Use target",
                                key="pipeline_studio_active_target",
                                width="stretch",
                                disabled=True,
                            )
                    if st.session_state.get("active_dataset_id_override"):
                        st.caption(
                            "Active dataset override is set in the sidebar and may supersede this selection."
                        )
                    notice = st.session_state.pop("pipeline_studio_active_notice", None)
                    if isinstance(notice, str) and notice.strip():
                        st.success(notice)
                if (
                    target_key == "model"
                    and isinstance(studio_active_id, str)
                    and studio_active_id
                    and studio_active_id not in node_ids
                ):
                    st.info(
                        "Your *active dataset* is not part of the current **Model (latest feature)** pipeline view. "
                        "Switch **Pipeline target → Active dataset** (or set stage to `feature`) to see the newest node."
                    )
                if (
                    target_key != "all"
                    and isinstance(studio_datasets, dict)
                    and len(studio_datasets) > len(node_ids)
                ):
                    st.info(
                        "Showing only the target lineage. Switch **Pipeline target → All datasets** to see every dataset."
                    )
                run_err = st.session_state.pop("pipeline_studio_run_error", None)
                if isinstance(run_err, str) and run_err.strip():
                    st.error(run_err)
                run_ok = st.session_state.pop("pipeline_studio_run_success", None)
                if isinstance(run_ok, str) and run_ok.strip():
                    active_now = None
                    try:
                        team_state = st.session_state.get("team_state", {})
                        team_state = team_state if isinstance(team_state, dict) else {}
                        active_now = team_state.get("active_dataset_id")
                    except Exception:
                        active_now = None
                    suffix = " (set active)" if active_now == run_ok else ""
                    st.success(f"Created new dataset: `{run_ok}`{suffix}.")
                history_notice = st.session_state.pop(
                    "pipeline_studio_history_notice", None
                )
                if isinstance(history_notice, str) and history_notice.strip():
                    st.info(history_notice)

                project_notice = st.session_state.pop(
                    "pipeline_studio_project_notice", None
                )
                if isinstance(project_notice, str) and project_notice.strip():
                    if project_notice.lower().startswith("error:"):
                        st.error(project_notice.replace("Error:", "", 1).strip())
                    else:
                        st.success(project_notice)

                with st.expander("Projects (save/load)", expanded=False):
                    st.caption(
                        "Saves Pipeline Studio state to `pipeline_store/pipeline_projects/` "
                        "(pickles optional; do not load untrusted files)."
                    )

                    def _save_project_click(
                        project_name: str, include_data: bool
                    ) -> None:
                        team_state = st.session_state.get("team_state", {})
                        res = _pipeline_studio_save_project(
                            name=project_name or "project",
                            team_state=team_state,
                            include_data=bool(include_data),
                        )
                        err = res.get("error")
                        if isinstance(err, str) and err:
                            st.session_state["pipeline_studio_project_notice"] = (
                                f"Error: {err}"
                            )
                            return
                        project_dir = res.get("project_dir")
                        if isinstance(project_dir, str) and project_dir:
                            data_mode = (
                                "metadata-only" if not bool(include_data) else "full"
                            )
                            st.session_state["pipeline_studio_project_notice"] = (
                                f"Saved {data_mode} project to `{project_dir}`."
                            )
                        else:
                            st.session_state["pipeline_studio_project_notice"] = (
                                "Error: Project save failed (unknown error)."
                            )

                    def _load_project_click(dir_name: str, rehydrate: bool) -> None:
                        dir_name = dir_name.strip() if isinstance(dir_name, str) else ""
                        if not dir_name:
                            st.session_state["pipeline_studio_project_notice"] = (
                                "Error: Select a project to load."
                            )
                            return
                        project_dir = os.path.join(
                            PIPELINE_STUDIO_PROJECTS_DIR, dir_name
                        )
                        res = _pipeline_studio_load_project(
                            project_dir=project_dir, rehydrate=bool(rehydrate)
                        )
                        err = res.get("error")
                        if isinstance(err, str) and err:
                            st.session_state["pipeline_studio_project_notice"] = (
                                f"Error: {err}"
                            )
                            return
                        loaded_n = res.get("loaded_datasets")
                        data_mode = str(res.get("data_mode") or "full")
                        stats = res.get("rehydrate_stats")
                        stats = stats if isinstance(stats, dict) else {}
                        missing_files = res.get("missing_files")
                        missing_files = (
                            missing_files if isinstance(missing_files, list) else []
                        )
                        data_files_loaded = int(res.get("data_files_loaded") or 0)
                        if data_mode == "metadata_only":
                            roots_loaded = int(stats.get("roots_loaded") or 0)
                            transforms_run = int(stats.get("transforms_run") or 0)
                            missing_sources = int(stats.get("missing_sources") or 0)
                            failures = int(stats.get("transform_failures") or 0)
                            suffix_bits = []
                            if bool(rehydrate):
                                if roots_loaded or transforms_run:
                                    suffix_bits.append(
                                        f"rehydrated {roots_loaded + transforms_run}"
                                    )
                                if missing_sources:
                                    suffix_bits.append(
                                        f"missing sources: {missing_sources}"
                                    )
                                if failures:
                                    suffix_bits.append(f"transform errors: {failures}")
                                suffix = (
                                    f" ({', '.join(suffix_bits)})"
                                    if suffix_bits
                                    else ""
                                )
                                st.session_state["pipeline_studio_project_notice"] = (
                                    f"Loaded metadata-only project: {int(loaded_n or 0)} dataset(s){suffix}."
                                )
                            else:
                                st.session_state["pipeline_studio_project_notice"] = (
                                    f"Loaded metadata-only project: {int(loaded_n or 0)} dataset(s) (data not rehydrated)."
                                )
                        else:
                            suffix_bits = []
                            if data_files_loaded:
                                suffix_bits.append(
                                    f"loaded {data_files_loaded} data file(s)"
                                )
                            if missing_files:
                                suffix_bits.append(
                                    f"missing data files: {len(missing_files)}"
                                )
                            suffix = (
                                f" ({', '.join(suffix_bits)})" if suffix_bits else ""
                            )
                            st.session_state["pipeline_studio_project_notice"] = (
                                f"Loaded project: {int(loaded_n or 0)} dataset(s){suffix}."
                            )
                        st.session_state["pipeline_studio_last_load_summary"] = {
                            "data_mode": data_mode,
                            "rehydrate_stats": stats,
                            "data_files_loaded": data_files_loaded,
                            "missing_files": missing_files,
                        }
                        st.session_state["pipeline_studio_loaded_project_dir"] = (
                            project_dir
                        )
                        missing_sources = res.get("missing_sources")
                        missing_sources = (
                            missing_sources if isinstance(missing_sources, list) else []
                        )
                        st.session_state["pipeline_studio_missing_sources"] = (
                            missing_sources
                        )
                        st.session_state["pipeline_studio_autofollow_pending"] = True
                        st.session_state["pipeline_studio_view_pending"] = (
                            "Visual Editor"
                        )

                    default_project_name = st.session_state.get(
                        "pipeline_studio_project_name"
                    )
                    if (
                        not isinstance(default_project_name, str)
                        or not default_project_name.strip()
                    ):
                        default_project_name = (
                            f"pipeline_{pipeline_hash[:8]}"
                            if pipeline_hash
                            else "project"
                        )
                    project_name = st.text_input(
                        "Project name",
                        value=default_project_name,
                        key="pipeline_studio_project_name",
                    )
                    c_save_meta, c_save_full = st.columns(2)
                    with c_save_meta:
                        st.button(
                            "Save project (metadata-only)",
                            key="pipeline_studio_project_save_meta",
                            on_click=_save_project_click,
                            args=(project_name, False),
                            width="stretch",
                            help="Stores lineage + steps without dataset pickles; reloads from source on demand.",
                        )
                    with c_save_full:
                        st.button(
                            "Save project with data",
                            key="pipeline_studio_project_save_full",
                            on_click=_save_project_click,
                            args=(project_name, True),
                            width="stretch",
                            help="Stores dataset snapshots (Parquet when available, else pickle).",
                        )

                    projects = _pipeline_studio_list_projects()
                    if not projects:
                        st.caption("No saved projects yet.")
                    else:
                        dir_options = [
                            p.get("dir_name") for p in projects if p.get("dir_name")
                        ]
                        dir_options = [
                            x for x in dir_options if isinstance(x, str) and x
                        ]

                        def _fmt_project(dir_name: str) -> str:
                            rec = next(
                                (p for p in projects if p.get("dir_name") == dir_name),
                                None,
                            )
                            if not isinstance(rec, dict):
                                return dir_name
                            label = (
                                rec.get("name")
                                if isinstance(rec.get("name"), str)
                                else dir_name
                            )
                            try:
                                ts = float(rec.get("saved_ts") or 0.0)
                            except Exception:
                                ts = 0.0
                            return f"{label} ({int(ts)})" if ts else str(label)

                        selected_project = st.selectbox(
                            "Load project",
                            options=dir_options,
                            format_func=_fmt_project,
                            key="pipeline_studio_project_select",
                        )
                        selected_manifest = None
                        selected_project_dir = None
                        if isinstance(selected_project, str) and selected_project:
                            selected_project_dir = os.path.join(
                                PIPELINE_STUDIO_PROJECTS_DIR, selected_project
                            )
                            selected_manifest = _pipeline_studio_load_project_manifest(
                                project_dir=selected_project_dir
                            )
                        selected_mode = (
                            str((selected_manifest or {}).get("data_mode") or "full")
                            .strip()
                            .lower()
                        )
                        selected_mode_label = (
                            "metadata-only"
                            if selected_mode == "metadata_only"
                            else "full"
                        )
                        prev_selected = st.session_state.get(
                            "pipeline_studio_project_select_prev"
                        )
                        if selected_project != prev_selected:
                            st.session_state["pipeline_studio_project_select_prev"] = (
                                selected_project
                            )
                            st.session_state["pipeline_studio_project_rehydrate"] = (
                                selected_mode == "metadata_only"
                            )
                        st.caption(f"Project mode: **{selected_mode_label}**")
                        rehydrate = st.checkbox(
                            "Rebuild datasets from sources (metadata-only)",
                            key="pipeline_studio_project_rehydrate",
                            help="Attempts to reload source files and replay transforms when the project contains metadata only.",
                        )
                        st.button(
                            "Load selected project",
                            key="pipeline_studio_project_load",
                            on_click=_load_project_click,
                            args=(selected_project, bool(rehydrate)),
                            width="stretch",
                        )
                        delete_confirm = st.checkbox(
                            "I understand this will delete the selected project",
                            value=False,
                            key="pipeline_studio_project_delete_confirm",
                        )
                        st.button(
                            "Delete selected project",
                            key="pipeline_studio_project_delete",
                            on_click=_pipeline_studio_delete_project,
                            args=(selected_project,),
                            width="stretch",
                            disabled=not bool(delete_confirm),
                        )

                    st.markdown("---")
                    st.markdown("**Project dashboard**")
                    projects = _pipeline_studio_list_projects()
                    load_summary = st.session_state.get(
                        "pipeline_studio_last_load_summary"
                    )
                    if isinstance(load_summary, dict):
                        mode = str(load_summary.get("data_mode") or "full").replace(
                            "_", "-"
                        )
                        stats = load_summary.get("rehydrate_stats")
                        stats = stats if isinstance(stats, dict) else {}
                        missing_files = load_summary.get("missing_files")
                        missing_files = (
                            missing_files if isinstance(missing_files, list) else []
                        )
                        data_files_loaded = int(
                            load_summary.get("data_files_loaded") or 0
                        )
                        stat_bits = []
                        if stats:
                            roots = int(stats.get("roots_loaded") or 0)
                            transforms = int(stats.get("transforms_run") or 0)
                            misses = int(stats.get("missing_sources") or 0)
                            fails = int(stats.get("transform_failures") or 0)
                            if roots or transforms:
                                stat_bits.append(f"rehydrated {roots + transforms}")
                            if misses:
                                stat_bits.append(f"missing sources: {misses}")
                            if fails:
                                stat_bits.append(f"transform errors: {fails}")
                        if mode != "metadata-only":
                            if data_files_loaded:
                                stat_bits.append(
                                    f"loaded data files: {data_files_loaded}"
                                )
                            if missing_files:
                                stat_bits.append(
                                    f"missing data files: {len(missing_files)}"
                                )
                        badge = f"Last load: {mode}"
                        if stat_bits:
                            badge = f"{badge} · {', '.join(stat_bits)}"
                        st.info(badge)
                    search_term = st.text_input(
                        "Search projects",
                        value=st.session_state.get(
                            "pipeline_studio_project_search", ""
                        ),
                        key="pipeline_studio_project_search",
                    )
                    show_archived = st.checkbox(
                        "Show archived projects",
                        value=bool(
                            st.session_state.get(
                                "pipeline_studio_project_show_archived", False
                            )
                        ),
                        key="pipeline_studio_project_show_archived",
                    )
                    show_sizes = st.checkbox(
                        "Show disk usage",
                        value=bool(
                            st.session_state.get(
                                "pipeline_studio_project_show_sizes", False
                            )
                        ),
                        key="pipeline_studio_project_show_sizes",
                    )
                    sort_by = st.selectbox(
                        "Sort by",
                        options=[
                            "Last saved",
                            "Last opened",
                            "Name",
                            "Datasets",
                            "Disk usage",
                        ],
                        index=0,
                        key="pipeline_studio_project_sort",
                    )

                    if search_term:
                        query = search_term.strip().lower()
                    else:
                        query = ""
                    filtered_projects = []
                    for rec in projects:
                        if not isinstance(rec, dict):
                            continue
                        if not show_archived and bool(rec.get("archived")):
                            continue
                        name = str(rec.get("name") or rec.get("dir_name") or "")
                        tags = rec.get("tags") or []
                        tags = [str(t).lower() for t in tags if isinstance(t, str)]
                        hay = " ".join([name.lower()] + tags)
                        if query and query not in hay:
                            continue
                        filtered_projects.append(rec)

                    if show_sizes:
                        for rec in filtered_projects:
                            dir_path = rec.get("dir_path")
                            if isinstance(dir_path, str):
                                rec["disk_bytes"] = _pipeline_studio_project_disk_usage(
                                    dir_path
                                )

                    def _sort_key(rec: dict) -> tuple:
                        if sort_by == "Name":
                            return (str(rec.get("name") or ""),)
                        if sort_by == "Last opened":
                            return (float(rec.get("last_opened_ts") or 0.0),)
                        if sort_by == "Datasets":
                            return (int(rec.get("datasets_total") or 0),)
                        if sort_by == "Disk usage":
                            return (int(rec.get("disk_bytes") or 0),)
                        return (float(rec.get("saved_ts") or 0.0),)

                    filtered_projects.sort(key=_sort_key, reverse=sort_by != "Name")

                    if not filtered_projects:
                        st.caption("No matching projects.")
                    else:
                        rows = []
                        for rec in filtered_projects:
                            mode = str(rec.get("data_mode") or "full").replace("_", "-")
                            mode_badge = "META" if mode == "metadata-only" else "FULL"
                            rows.append(
                                {
                                    "name": rec.get("name") or rec.get("dir_name"),
                                    "mode": mode_badge,
                                    "datasets": int(rec.get("datasets_total") or 0),
                                    "saved": _pipeline_studio_format_ts(
                                        rec.get("saved_ts")
                                    ),
                                    "opened": _pipeline_studio_format_ts(
                                        rec.get("last_opened_ts")
                                    ),
                                    "tags": ", ".join(
                                        [
                                            t
                                            for t in (rec.get("tags") or [])
                                            if isinstance(t, str) and t.strip()
                                        ]
                                    ),
                                    "archived": bool(rec.get("archived")),
                                    "size": _pipeline_studio_format_bytes(
                                        rec.get("disk_bytes")
                                    )
                                    if show_sizes
                                    else "-",
                                }
                            )
                        df_rows = pd.DataFrame(rows)

                        def _style_mode(val: object) -> str:
                            val_str = str(val).strip().upper()
                            if val_str == "META":
                                return (
                                    "background-color: rgba(88, 166, 255, 0.15);"
                                    "color: #9cc9ff; font-weight: 600;"
                                )
                            return (
                                "background-color: rgba(46, 160, 67, 0.15);"
                                "color: #7ee787; font-weight: 600;"
                            )

                        st.dataframe(
                            df_rows.style.map(_style_mode, subset=["mode"]),
                            width="stretch",
                            hide_index=True,
                        )
                        cache_bytes = _pipeline_studio_dataset_cache_usage()
                        if cache_bytes:
                            st.caption(
                                f"Dataset cache size: {_pipeline_studio_format_bytes(cache_bytes)}"
                            )

                        def _project_label(dir_name: str) -> str:
                            rec = next(
                                (
                                    p
                                    for p in filtered_projects
                                    if p.get("dir_name") == dir_name
                                ),
                                None,
                            )
                            if not isinstance(rec, dict):
                                return dir_name
                            name = rec.get("name") or dir_name
                            mode = str(rec.get("data_mode") or "full").replace("_", "-")
                            data_ct = rec.get("datasets_total") or 0
                            archived = " • archived" if rec.get("archived") else ""
                            return f"{name} ({mode}, {data_ct} datasets){archived}"

                        manage_options = [
                            rec.get("dir_name")
                            for rec in filtered_projects
                            if rec.get("dir_name")
                        ]
                        manage_options = [
                            x
                            for x in manage_options
                            if isinstance(x, str) and x.strip()
                        ]
                        manage_choice = st.selectbox(
                            "Manage project",
                            options=[""] + manage_options,
                            format_func=_project_label,
                            key="pipeline_studio_project_manage",
                        )
                        if manage_choice:
                            rec = next(
                                (
                                    p
                                    for p in filtered_projects
                                    if p.get("dir_name") == manage_choice
                                ),
                                None,
                            )
                            rec = rec if isinstance(rec, dict) else {}
                            tags_str = ", ".join(
                                [
                                    t
                                    for t in (rec.get("tags") or [])
                                    if isinstance(t, str) and t.strip()
                                ]
                            )
                            tags_input = st.text_input(
                                "Tags (comma-separated)",
                                value=tags_str,
                                key=f"pipeline_studio_project_tags_{manage_choice}",
                            )
                            notes_input = st.text_area(
                                "Notes",
                                value=rec.get("notes") or "",
                                key=f"pipeline_studio_project_notes_{manage_choice}",
                            )
                            archived_input = st.checkbox(
                                "Archived",
                                value=bool(rec.get("archived")),
                                key=f"pipeline_studio_project_archived_{manage_choice}",
                            )
                            if st.button(
                                "Save metadata",
                                key=f"pipeline_studio_project_meta_save_{manage_choice}",
                                width="stretch",
                            ):
                                tags_list = [
                                    t.strip()
                                    for t in tags_input.split(",")
                                    if t.strip()
                                ]
                                _pipeline_studio_save_project_metadata(
                                    dir_name=manage_choice,
                                    tags=tags_list,
                                    notes=notes_input,
                                    archived=bool(archived_input),
                                )

                            rename_name = st.text_input(
                                "Rename to",
                                value=rec.get("name") or manage_choice,
                                key=f"pipeline_studio_project_rename_{manage_choice}",
                            )
                            if st.button(
                                "Rename project",
                                key=f"pipeline_studio_project_rename_btn_{manage_choice}",
                                width="stretch",
                            ):
                                res = _pipeline_studio_rename_project(
                                    dir_name=manage_choice, new_name=rename_name
                                )
                                err = res.get("error")
                                if err:
                                    st.session_state[
                                        "pipeline_studio_project_notice"
                                    ] = f"Error: {err}"
                                else:
                                    st.session_state[
                                        "pipeline_studio_project_notice"
                                    ] = "Project renamed."

                            dup_name = st.text_input(
                                "Duplicate as",
                                value=f"{rec.get('name') or manage_choice} copy",
                                key=f"pipeline_studio_project_duplicate_{manage_choice}",
                            )
                            if st.button(
                                "Duplicate project",
                                key=f"pipeline_studio_project_duplicate_btn_{manage_choice}",
                                width="stretch",
                            ):
                                res = _pipeline_studio_duplicate_project(
                                    dir_name=manage_choice, new_name=dup_name
                                )
                                err = res.get("error")
                                if err:
                                    st.session_state[
                                        "pipeline_studio_project_notice"
                                    ] = f"Error: {err}"
                                else:
                                    st.session_state[
                                        "pipeline_studio_project_notice"
                                    ] = "Project duplicated."

                            if st.button(
                                "Convert to metadata-only",
                                key=f"pipeline_studio_project_convert_{manage_choice}",
                                width="stretch",
                            ):
                                _pipeline_studio_convert_project_to_metadata_only(
                                    dir_name=manage_choice
                                )

                        bulk_options = manage_options
                        bulk_delete = st.multiselect(
                            "Bulk delete projects",
                            options=bulk_options,
                            format_func=_project_label,
                            key="pipeline_studio_project_bulk_delete",
                        )
                        bulk_confirm = st.checkbox(
                            "I understand this will delete the selected projects",
                            value=False,
                            key="pipeline_studio_project_bulk_confirm",
                        )
                        st.button(
                            "Delete selected projects",
                            key="pipeline_studio_project_bulk_delete_btn",
                            on_click=_pipeline_studio_bulk_delete_projects,
                            args=(bulk_delete,),
                            width="stretch",
                            disabled=not (bulk_confirm and bulk_delete),
                        )

                    missing_sources = st.session_state.get(
                        "pipeline_studio_missing_sources"
                    )
                    missing_sources = (
                        missing_sources if isinstance(missing_sources, list) else []
                    )
                    project_dir = st.session_state.get(
                        "pipeline_studio_loaded_project_dir"
                    )
                    if missing_sources and isinstance(project_dir, str):
                        with st.expander("Relink missing sources", expanded=True):
                            st.caption(
                                "Update file paths for datasets that could not be rehydrated."
                            )
                            for rec in missing_sources:
                                did = rec.get("dataset_id")
                                if not isinstance(did, str) or not did:
                                    continue
                                label = rec.get("label") or did
                                old_src = rec.get("source") or ""
                                st.text_input(
                                    f"{label} ({did})",
                                    value=old_src,
                                    key=f"pipeline_studio_relink_{did}",
                                    help=f"Previous source: {old_src}",
                                )
                            if st.button(
                                "Relink + reload project",
                                key="pipeline_studio_relink_apply",
                                width="stretch",
                            ):
                                updates: dict[str, str] = {}
                                for rec in missing_sources:
                                    did = rec.get("dataset_id")
                                    if not isinstance(did, str) or not did:
                                        continue
                                    new_val = st.session_state.get(
                                        f"pipeline_studio_relink_{did}"
                                    )
                                    old_val = rec.get("source") or ""
                                    if (
                                        isinstance(new_val, str)
                                        and new_val.strip()
                                        and new_val.strip() != old_val
                                    ):
                                        updates[did] = new_val.strip()
                                if updates:
                                    _pipeline_studio_update_project_manifest(
                                        project_dir=project_dir,
                                        dataset_source_updates=updates,
                                    )
                                res = _pipeline_studio_load_project(
                                    project_dir=project_dir, rehydrate=True
                                )
                                err = res.get("error")
                                if err:
                                    st.session_state[
                                        "pipeline_studio_project_notice"
                                    ] = f"Error: {err}"
                                else:
                                    st.session_state[
                                        "pipeline_studio_project_notice"
                                    ] = "Relinked sources and reloaded project."
                                missing_next = res.get("missing_sources")
                                missing_next = (
                                    missing_next
                                    if isinstance(missing_next, list)
                                    else []
                                )
                                st.session_state["pipeline_studio_missing_sources"] = (
                                    missing_next
                                )

                    project_dir = st.session_state.get(
                        "pipeline_studio_loaded_project_dir"
                    )
                    if isinstance(project_dir, str) and project_dir:
                        if st.button(
                            "Rehydrate now",
                            key="pipeline_studio_rehydrate_now",
                            width="stretch",
                        ):
                            res = _pipeline_studio_load_project(
                                project_dir=project_dir, rehydrate=True
                            )
                            err = res.get("error")
                            if err:
                                st.session_state["pipeline_studio_project_notice"] = (
                                    f"Error: {err}"
                                )
                            else:
                                st.session_state["pipeline_studio_project_notice"] = (
                                    "Rehydrated project."
                                )
                            missing_next = res.get("missing_sources")
                            missing_next = (
                                missing_next if isinstance(missing_next, list) else []
                            )
                            st.session_state["pipeline_studio_missing_sources"] = (
                                missing_next
                            )
                            st.session_state["pipeline_studio_last_load_summary"] = {
                                "data_mode": res.get("data_mode"),
                                "rehydrate_stats": res.get("rehydrate_stats") or {},
                            }

                        manifest = _pipeline_studio_load_project_manifest(
                            project_dir=project_dir
                        )
                        team_state = (
                            manifest.get("team_state")
                            if isinstance(manifest, dict)
                            else {}
                        )
                        datasets_meta = (
                            team_state.get("datasets")
                            if isinstance(team_state, dict)
                            else {}
                        )
                        preview_sets = [
                            (did, meta)
                            for did, meta in datasets_meta.items()
                            if isinstance(meta, dict)
                            and isinstance(meta.get("preview_data"), list)
                            and meta.get("preview_data")
                        ]
                        if preview_sets:
                            with st.expander(
                                "Data previews (metadata-only)", expanded=False
                            ):
                                for did, meta in preview_sets:
                                    label = meta.get("label") or did
                                    st.caption(f"{label} ({did})")
                                    st.dataframe(
                                        meta.get("preview_data"),
                                        width="stretch",
                                        hide_index=True,
                                    )

                    st.markdown("---")
                    st.markdown("**Start new project**")
                    st.caption(
                        "Clears current datasets, Pipeline Studio state, and chat history. "
                        "Saved projects remain available."
                    )
                    clear_cache = st.checkbox(
                        "Also clear Pipeline Studio cache on disk",
                        value=False,
                        key="pipeline_studio_reset_clear_cache",
                        help="Removes cached datasets/layout/artifacts stored in `pipeline_store/`.",
                    )
                    confirm_reset = st.checkbox(
                        "I understand this will clear the current project",
                        value=False,
                        key="pipeline_studio_reset_confirm",
                    )
                    st.button(
                        "Start new project",
                        key="pipeline_studio_project_reset",
                        on_click=_pipeline_studio_reset_project,
                        args=(bool(clear_cache), bool(add_memory)),
                        width="stretch",
                        disabled=not bool(confirm_reset),
                    )

                    st.markdown("---")
                    st.markdown("**Factory reset (delete cache + reports)**")
                    st.caption(
                        "Deletes `pipeline_store/` (datasets/projects/cache) and "
                        "`pipeline_reports/` (pipeline specs/scripts). This is permanent."
                    )
                    factory_confirm = st.checkbox(
                        "I understand this will permanently delete pipeline_store/ and pipeline_reports/",
                        value=False,
                        key="pipeline_studio_factory_reset_confirm",
                    )
                    st.button(
                        "Factory reset",
                        key="pipeline_studio_factory_reset",
                        on_click=_pipeline_studio_factory_reset,
                        args=(bool(add_memory),),
                        width="stretch",
                        disabled=not bool(factory_confirm),
                    )

                def _pipeline_studio_hide_nodes(
                    p_hash: str,
                    node_ids_to_hide: list[str] | set[str] | tuple[str, ...],
                ) -> None:
                    try:
                        import time as _time

                        p_hash = p_hash.strip() if isinstance(p_hash, str) else ""
                        if not p_hash:
                            return
                        node_ids_clean = {
                            str(x).strip()
                            for x in (
                                node_ids_to_hide
                                if isinstance(node_ids_to_hide, (list, set, tuple))
                                else []
                            )
                            if str(x).strip()
                        }
                        if not node_ids_clean:
                            return
                        ui_h, ui_d = _pipeline_studio_get_registry_ui(
                            pipeline_hash=p_hash
                        )
                        ui_h = set(ui_h) | set(node_ids_clean)
                        _pipeline_studio_set_registry_ui(
                            pipeline_hash=p_hash,
                            hidden_ids=ui_h,
                            deleted_ids=ui_d,
                        )
                        st.session_state["pipeline_studio_flow_hidden_ids"] = sorted(
                            set(ui_h) | set(ui_d)
                        )
                        st.session_state["pipeline_studio_flow_fit_view_pending"] = True
                        if "pipeline_studio_flow_ts" in st.session_state:
                            st.session_state["pipeline_studio_flow_ts"] = int(
                                _time.time() * 1000
                            )
                    except Exception:
                        pass

                def _pipeline_studio_hide_old_branch(p_hash: str, root_id: str) -> None:
                    root_id = root_id.strip() if isinstance(root_id, str) else ""
                    if not root_id:
                        return
                    child_idx = _build_children_index(studio_datasets)
                    old_branch = {root_id} | _descendants(root_id, child_idx)
                    _pipeline_studio_hide_nodes(p_hash, sorted(old_branch))

                last_repl = st.session_state.get("pipeline_studio_last_replacement")
                stale_ids = st.session_state.get("pipeline_studio_stale_ids")
                stale_ids = stale_ids if isinstance(stale_ids, list) else []
                if isinstance(last_repl, dict):
                    old_id = last_repl.get("old_id")
                    new_id = last_repl.get("new_id")
                    old_id = old_id if isinstance(old_id, str) and old_id else None
                    new_id = new_id if isinstance(new_id, str) and new_id else None
                    if old_id and new_id and stale_ids:
                        st.warning(
                            f"Downstream invalidation: {len(stale_ids)} node(s) downstream of `{old_id}` are stale."
                        )
                        with st.expander("Stale / skipped details", expanded=False):
                            st.code(
                                "\n".join(stale_ids[:100])
                                + ("\n..." if len(stale_ids) > 100 else "")
                            )
                            skipped = st.session_state.get(
                                "pipeline_studio_last_downstream_skipped"
                            )
                            failed = st.session_state.get(
                                "pipeline_studio_last_downstream_failures"
                            )
                            if isinstance(skipped, list) and skipped:
                                st.markdown("**Skipped**")
                                st.dataframe(
                                    pd.DataFrame(skipped), width="stretch"
                                )
                            if isinstance(failed, list) and failed:
                                st.markdown("**Failures**")
                                st.dataframe(
                                    pd.DataFrame(failed), width="stretch"
                                )
                            if pipeline_hash and stale_ids:
                                st.markdown("---")
                                st.button(
                                    "Hide these stale nodes",
                                    key="pipeline_studio_hide_stale_nodes",
                                    help="Marks these stale ids as hidden in the pipeline registry (reversible).",
                                    on_click=_pipeline_studio_hide_nodes,
                                    args=(pipeline_hash, stale_ids),
                                )

                        confirm_downstream = st.checkbox(
                            "I understand this will rerun downstream steps locally",
                            key="pipeline_studio_run_downstream_confirm",
                            help="Replays downstream transforms where possible, starting from the last rerun node.",
                        )
                        st.button(
                            "Run downstream transforms",
                            key="pipeline_studio_run_downstream",
                            type="primary",
                            disabled=not bool(confirm_downstream),
                            on_click=_run_downstream_transforms,
                            args=(old_id, new_id),
                        )

                last_mapping = st.session_state.get(
                    "pipeline_studio_last_downstream_mapping"
                )
                last_mapping = last_mapping if isinstance(last_mapping, dict) else {}
                last_source = st.session_state.get(
                    "pipeline_studio_last_downstream_source"
                )
                last_source = last_source if isinstance(last_source, dict) else {}
                if last_mapping:
                    with st.expander("Downstream replacements", expanded=False):
                        src_old = last_source.get("old_id")
                        src_new = last_source.get("new_id")
                        src_old = (
                            src_old if isinstance(src_old, str) and src_old else None
                        )
                        src_new = (
                            src_new if isinstance(src_new, str) and src_new else None
                        )
                        created_ids = last_source.get("created_ids")
                        created_ids = (
                            created_ids if isinstance(created_ids, list) else []
                        )
                        if src_old and src_new:
                            st.caption(f"From `{src_old}` → `{src_new}`")
                        if created_ids:
                            st.caption(f"Created {len(created_ids)} new dataset(s).")

                        mapping_rows = [
                            {"old_id": str(k), "new_id": str(v)}
                            for k, v in sorted(
                                last_mapping.items(), key=lambda x: str(x[0])
                            )
                            if isinstance(k, str) and isinstance(v, str) and k and v
                        ]
                        if mapping_rows:
                            st.dataframe(
                                pd.DataFrame(mapping_rows), width="stretch"
                            )

                        c_hide_stale, c_hide_old = st.columns(2)
                        with c_hide_stale:
                            st.button(
                                "Hide remaining stale",
                                key="pipeline_studio_hide_remaining_stale",
                                help="Hides any still-stale ids shown in the warning above (if present).",
                                disabled=not bool(pipeline_hash and stale_ids),
                                on_click=_pipeline_studio_hide_nodes,
                                args=(pipeline_hash, stale_ids),
                                width="stretch",
                            )
                        with c_hide_old:
                            st.button(
                                "Hide old branch",
                                key="pipeline_studio_hide_old_branch",
                                help="Hides the superseded branch (the old node and its downstream descendants).",
                                disabled=not bool(pipeline_hash and src_old),
                                on_click=_pipeline_studio_hide_old_branch,
                                args=(pipeline_hash, src_old or ""),
                                width="stretch",
                            )

                _pipeline_studio_history_init()
                undo_stack = st.session_state.get("pipeline_studio_undo_stack")
                undo_stack = undo_stack if isinstance(undo_stack, list) else []
                redo_stack = st.session_state.get("pipeline_studio_redo_stack")
                redo_stack = redo_stack if isinstance(redo_stack, list) else []

                def _history_action_summary(action: dict) -> tuple[str, list[str]]:
                    action = action if isinstance(action, dict) else {}
                    action_type = str(action.get("type") or "")
                    if action_type == "create_dataset":
                        did = action.get("dataset_id")
                        did = did if isinstance(did, str) and did else ""
                        return (
                            f"Remove `{did}`" if did else "Remove last dataset",
                            [did] if did else [],
                        )
                    if action_type == "create_datasets":
                        ids = action.get("dataset_ids")
                        ids = ids if isinstance(ids, list) else []
                        clean = [str(x) for x in ids if isinstance(x, str) and x]
                        return (f"Remove {len(clean)} dataset(s)", clean)
                    return (f"Action `{action_type}`", [])

                undo_summary, undo_ids = (
                    _history_action_summary(undo_stack[-1]) if undo_stack else ("", [])
                )
                redo_summary, _redo_ids = (
                    _history_action_summary(redo_stack[-1]) if redo_stack else ("", [])
                )

                undo_blocked_reason = None
                if undo_ids:
                    try:
                        remove_set = set(undo_ids)
                        for did, ent in studio_datasets.items():
                            if did in remove_set or not isinstance(ent, dict):
                                continue
                            parents = _entry_parent_ids(ent)
                            if any(p in remove_set for p in parents):
                                undo_blocked_reason = f"Undo blocked: downstream dataset `{did}` depends on the last run."
                                break
                    except Exception:
                        undo_blocked_reason = None

                c_undo, c_redo = st.columns(2)
                with c_undo:
                    st.button(
                        "Undo run",
                        key="pipeline_studio_undo",
                        help=(
                            "Undo the most recent run (removes the last created dataset node). "
                            "Hide/Delete actions are handled separately via Restore/Unhide."
                            + (f" ({undo_summary})" if undo_summary else "")
                        ),
                        disabled=not bool(undo_stack) or bool(undo_blocked_reason),
                        on_click=_pipeline_studio_undo_last_action,
                    )
                with c_redo:
                    st.button(
                        "Redo run",
                        key="pipeline_studio_redo",
                        help=(
                            "Redo the most recently undone run (restores the dataset node)."
                            + (f" ({redo_summary})" if redo_summary else "")
                        ),
                        disabled=not bool(redo_stack),
                        on_click=_pipeline_studio_redo_last_action,
                    )
                if isinstance(undo_blocked_reason, str) and undo_blocked_reason.strip():
                    st.caption(undo_blocked_reason)

                script = pipe.get("script") if isinstance(pipe, dict) else None
                spec_bytes = None
                try:
                    spec = dict(pipe) if isinstance(pipe, dict) else {}
                    spec.pop("script", None)
                    spec_bytes = json.dumps(spec, indent=2, default=str).encode("utf-8")
                except Exception:
                    spec_bytes = None
                if spec_bytes:
                    st.download_button(
                        "Download pipeline spec (JSON)",
                        data=spec_bytes,
                        file_name=f"pipeline_spec_{pipe.get('target') or 'model'}.json",
                        mime="application/json",
                        key="pipeline_studio_download_spec",
                    )
                registry_bytes = None
                try:
                    ph = pipeline_hash
                    rec = (
                        _get_persisted_pipeline_registry(pipeline_hash=ph) if ph else {}
                    )
                    if not rec:
                        rec = _build_pipeline_registry_record(
                            pipeline=pipe,
                            datasets=studio_datasets,
                            artifacts_by_dataset_id=st.session_state.get(
                                "pipeline_studio_artifacts"
                            ),
                        )
                    if isinstance(rec, dict) and rec:
                        registry_bytes = json.dumps(rec, indent=2, default=str).encode(
                            "utf-8"
                        )
                except Exception:
                    registry_bytes = None
                if registry_bytes:
                    st.download_button(
                        "Download pipeline registry (JSON)",
                        data=registry_bytes,
                        file_name=f"pipeline_registry_{pipe.get('target') or 'model'}.json",
                        mime="application/json",
                        key="pipeline_studio_download_registry",
                        help="Semantic DAG metadata + artifact pointers (no DataFrames).",
                    )
                if isinstance(script, str) and script.strip():
                    st.download_button(
                        "Download pipeline script",
                        data=script.encode("utf-8"),
                        file_name=f"pipeline_repro_{pipe.get('target') or 'model'}.py",
                        mime="text/x-python",
                        key="pipeline_studio_download_repro",
                    )

                pending_node = st.session_state.pop(
                    "pipeline_studio_node_id_pending", None
                )
                if (
                    isinstance(pending_node, str)
                    and pending_node
                    and isinstance(node_ids, list)
                    and pending_node in node_ids
                ):
                    if pending_node in ui_hidden_ids:
                        st.session_state["pipeline_studio_show_hidden"] = True
                    if pending_node in ui_deleted_ids:
                        st.session_state["pipeline_studio_show_deleted"] = True
                    st.session_state["pipeline_studio_node_id"] = pending_node
                    # Disable auto-follow when a user explicitly selects a node.
                    st.session_state["pipeline_studio_autofollow"] = False

                pending_autofollow = st.session_state.pop(
                    "pipeline_studio_autofollow_pending", None
                )
                if pending_autofollow is not None:
                    st.session_state["pipeline_studio_autofollow"] = bool(
                        pending_autofollow
                    )

                auto_follow_default = bool(
                    st.session_state.get("pipeline_studio_autofollow", True)
                )
                auto_follow = st.checkbox(
                    "Auto-follow latest step",
                    value=auto_follow_default,
                    key="pipeline_studio_autofollow",
                    help="When enabled, the studio auto-selects the newest pipeline node after each run.",
                )

                show_hidden = st.checkbox(
                    "Show hidden steps",
                    value=bool(
                        st.session_state.get("pipeline_studio_show_hidden", False)
                    ),
                    key="pipeline_studio_show_hidden",
                    help="Include hidden steps in the step selector and compare options.",
                )
                show_deleted = st.checkbox(
                    "Show deleted steps",
                    value=bool(
                        st.session_state.get("pipeline_studio_show_deleted", False)
                    ),
                    key="pipeline_studio_show_deleted",
                    help="Include deleted steps in the step selector and compare options.",
                )
                selectable_node_ids = [
                    did
                    for did in node_ids
                    if (show_hidden or did not in ui_hidden_ids)
                    and (show_deleted or did not in ui_deleted_ids)
                ]
                if not selectable_node_ids:
                    selectable_node_ids = list(node_ids)
                    st.warning(
                        "All steps are hidden/deleted; showing all for selection."
                    )

                # Keep selection valid and optionally auto-follow newest node.
                if selectable_node_ids:
                    desired = selectable_node_ids[-1]
                    current = st.session_state.get("pipeline_studio_node_id")
                    if auto_follow:
                        st.session_state["pipeline_studio_node_id"] = desired
                    elif (
                        not isinstance(current, str)
                        or current not in selectable_node_ids
                    ):
                        st.session_state["pipeline_studio_node_id"] = desired

                def _node_label(did: str) -> str:
                    m = meta_by_id.get(did) or {}
                    label = m.get("label") or did
                    stage = m.get("stage") or ""
                    shape = m.get("shape")
                    tk = m.get("transform_kind") or ""
                    bits = []
                    if stage:
                        bits.append(str(stage))
                    if tk:
                        bits.append(str(tk))
                    if isinstance(shape, (list, tuple)) and len(shape) == 2:
                        bits.append(f"{shape[0]}×{shape[1]}")
                    if did in ui_deleted_ids:
                        bits.append("deleted")
                    elif did in ui_hidden_ids:
                        bits.append("hidden")
                    if did in stale_ids:
                        bits.append("stale")
                    meta = f" ({', '.join(bits)})" if bits else ""
                    return f"{label}{meta}"

                selected_node_id = st.selectbox(
                    "Pipeline step",
                    options=selectable_node_ids,
                    format_func=_node_label,
                    key="pipeline_studio_node_id",
                )

                with st.expander("Templates (quick add)", expanded=False):
                    template_catalog = [
                        {
                            "id": "py_drop_columns",
                            "title": "Drop columns",
                            "kind": "python_function",
                            "stage": "wrangled",
                            "label": "drop_columns",
                            "desc": "Remove a list of columns (ignore missing).",
                            "code": (
                                "import pandas as pd\n\n"
                                "def transform(df: pd.DataFrame) -> pd.DataFrame:\n"
                                "    df = df.copy()\n"
                                '    cols_to_drop = ["col_a", "col_b"]\n'
                                "    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])\n"
                            ),
                        },
                        {
                            "id": "py_filter_rows",
                            "title": "Filter rows",
                            "kind": "python_function",
                            "stage": "wrangled",
                            "label": "filter_rows",
                            "desc": "Keep rows that match a condition.",
                            "code": (
                                "import pandas as pd\n\n"
                                "def transform(df: pd.DataFrame) -> pd.DataFrame:\n"
                                "    df = df.copy()\n"
                                '    return df[df["col_a"] > 0]\n'
                            ),
                        },
                        {
                            "id": "py_fill_missing",
                            "title": "Fill missing values",
                            "kind": "python_function",
                            "stage": "cleaned",
                            "label": "fill_missing",
                            "desc": "Fill missing values with defaults.",
                            "code": (
                                "import pandas as pd\n\n"
                                "def transform(df: pd.DataFrame) -> pd.DataFrame:\n"
                                "    df = df.copy()\n"
                                '    return df.fillna({"col_a": 0, "col_b": "Unknown"})\n'
                            ),
                        },
                        {
                            "id": "py_groupby_agg",
                            "title": "Groupby aggregate",
                            "kind": "python_function",
                            "stage": "feature",
                            "label": "groupby_aggregate",
                            "desc": "Aggregate values by a category.",
                            "code": (
                                "import pandas as pd\n\n"
                                "def transform(df: pd.DataFrame) -> pd.DataFrame:\n"
                                "    df = df.copy()\n"
                                '    return df.groupby("category", as_index=False)["value"].mean()\n'
                            ),
                        },
                        {
                            "id": "sql_filter",
                            "title": "SQL: filter + limit",
                            "kind": "sql_query",
                            "stage": "sql",
                            "label": "sql_filter",
                            "desc": "Read-only SQL filter with LIMIT.",
                            "code": (
                                "SELECT *\nFROM my_table\nWHERE col_a > 0\nLIMIT 100\n"
                            ),
                        },
                        {
                            "id": "sql_aggregate",
                            "title": "SQL: aggregate",
                            "kind": "sql_query",
                            "stage": "sql",
                            "label": "sql_aggregate",
                            "desc": "Read-only SQL aggregation.",
                            "code": (
                                "SELECT category, COUNT(*) AS n\n"
                                "FROM my_table\n"
                                "GROUP BY category\n"
                            ),
                        },
                        {
                            "id": "merge_left_join",
                            "title": "Merge: left join",
                            "kind": "python_merge",
                            "stage": "wrangled",
                            "label": "merge_left_join",
                            "desc": "Left join df_0 with df_1 on a key.",
                            "code": (
                                "import pandas as pd\n\n"
                                "# df_0, df_1, ... are available\n"
                                'df = df_0.merge(df_1, on="id", how="left")\n'
                            ),
                        },
                    ]
                    template_map = {t["id"]: t for t in template_catalog}
                    template_ids = [t["id"] for t in template_catalog]
                    template_choice = st.selectbox(
                        "Template",
                        options=template_ids,
                        format_func=lambda tid: template_map.get(tid, {}).get(
                            "title", tid
                        ),
                        key="pipeline_studio_template_choice",
                    )
                    chosen_template = template_map.get(template_choice)
                    if isinstance(chosen_template, dict):
                        kind = chosen_template.get("kind")
                        title = chosen_template.get("title")
                        desc = chosen_template.get("desc")
                        if title:
                            st.caption(f"{title} · {kind}")
                        if desc:
                            st.caption(desc)

                    def _apply_template() -> None:
                        if not isinstance(chosen_template, dict):
                            return
                        kind = chosen_template.get("kind")
                        kind = kind if isinstance(kind, str) else "python_function"
                        code_key_map = {
                            "python_function": "pipeline_studio_manual_python_code",
                            "sql_query": "pipeline_studio_manual_sql_code",
                            "python_merge": "pipeline_studio_manual_merge_code",
                        }
                        code_key = code_key_map.get(kind)
                        st.session_state["pipeline_studio_manual_kind"] = kind
                        st.session_state["pipeline_studio_manual_kind_prev"] = kind
                        st.session_state["pipeline_studio_manual_seed_defaults"] = False
                        st.session_state["pipeline_studio_manual_stage"] = (
                            chosen_template.get("stage") or "custom"
                        )
                        st.session_state["pipeline_studio_manual_label"] = (
                            chosen_template.get("label") or "manual_transform"
                        )
                        st.session_state["pipeline_studio_manual_confirm_run"] = False
                        if code_key:
                            st.session_state[code_key] = (
                                chosen_template.get("code") or ""
                            )
                        if (
                            kind in {"python_function", "sql_query"}
                            and isinstance(selected_node_id, str)
                            and selected_node_id
                        ):
                            st.session_state["pipeline_studio_manual_parent_id"] = (
                                selected_node_id
                            )
                        st.session_state["pipeline_studio_manual_node_open"] = True
                        st.session_state["pipeline_studio_view_pending"] = (
                            "Visual Editor"
                        )
                        st.session_state["pipeline_studio_history_notice"] = (
                            f"Loaded template: {chosen_template.get('title') or 'template'}."
                        )

                    st.button(
                        "Use template",
                        key="pipeline_studio_template_apply",
                        on_click=_apply_template,
                        width="stretch",
                    )

                with st.expander("Merge wizard", expanded=False):
                    if len(studio_datasets) < 2:
                        st.info("Load at least two datasets to enable merges.")
                    else:
                        ordered_ids = sorted(
                            studio_datasets.items(),
                            key=lambda kv: float(kv[1].get("created_ts") or 0.0)
                            if isinstance(kv[1], dict)
                            else 0.0,
                            reverse=True,
                        )
                        wizard_ids = [
                            did for did, _e in ordered_ids if isinstance(did, str)
                        ]
                        if not wizard_ids:
                            st.info("No datasets available.")
                        else:

                            def _wiz_label(did: str) -> str:
                                entry = studio_datasets.get(did)
                                entry = entry if isinstance(entry, dict) else {}
                                label = entry.get("label") or did
                                stage = entry.get("stage") or "dataset"
                                shape = entry.get("shape")
                                shape_txt = f" {shape}" if shape else ""
                                return f"{stage}: {label}{shape_txt} ({did})"

                            default_left = (
                                studio_active_id
                                if isinstance(studio_active_id, str)
                                and studio_active_id in wizard_ids
                                else wizard_ids[0]
                            )
                            left_id = st.selectbox(
                                "Left dataset",
                                options=wizard_ids,
                                index=wizard_ids.index(default_left)
                                if default_left in wizard_ids
                                else 0,
                                format_func=_wiz_label,
                                key="pipeline_studio_merge_left_id",
                            )
                            right_options = [
                                did for did in wizard_ids if did != left_id
                            ]
                            right_default = (
                                right_options[0] if right_options else left_id
                            )
                            right_id = st.selectbox(
                                "Right dataset",
                                options=right_options or wizard_ids,
                                index=right_options.index(right_default)
                                if right_default in right_options
                                else 0,
                                format_func=_wiz_label,
                                key="pipeline_studio_merge_right_id",
                            )
                            op = st.selectbox(
                                "Merge operation",
                                options=["join", "concat"],
                                index=0,
                                key="pipeline_studio_merge_op",
                            )

                            def _dataset_columns(did: str) -> list[str]:
                                entry = studio_datasets.get(did)
                                entry = entry if isinstance(entry, dict) else {}
                                cols = entry.get("columns")
                                if isinstance(cols, list):
                                    return [str(c) for c in cols if str(c)]
                                df = _dataset_entry_to_df(entry)
                                if isinstance(df, pd.DataFrame):
                                    return [str(c) for c in df.columns]
                                return []

                            join_keys: list[str] = []
                            join_how = "left"
                            concat_axis = 0
                            concat_ignore_index = True
                            if op == "join":
                                left_cols = _dataset_columns(left_id)
                                right_cols = _dataset_columns(right_id)
                                common_cols = [
                                    c for c in left_cols if c in set(right_cols)
                                ]
                                preferred = sorted(
                                    common_cols,
                                    key=lambda c: (
                                        0 if "id" in c.lower() else 1,
                                        c.lower(),
                                    ),
                                )
                                join_keys = st.multiselect(
                                    "Join keys",
                                    options=preferred,
                                    default=preferred[:1],
                                    key="pipeline_studio_merge_keys",
                                )
                                join_how = st.selectbox(
                                    "Join type",
                                    options=["inner", "left", "right", "outer"],
                                    index=1,
                                    key="pipeline_studio_merge_how",
                                )
                            else:
                                concat_axis = st.selectbox(
                                    "Concat axis",
                                    options=[0, 1],
                                    index=0,
                                    key="pipeline_studio_merge_axis",
                                )
                                concat_ignore_index = st.checkbox(
                                    "Ignore index (axis=0)",
                                    value=True,
                                    key="pipeline_studio_merge_ignore_index",
                                )

                            stage_default = "wrangled"
                            label_default = "merge_wizard"
                            stage = st.text_input(
                                "Stage",
                                value=stage_default,
                                key="pipeline_studio_merge_stage",
                            )
                            label = st.text_input(
                                "Label",
                                value=label_default,
                                key="pipeline_studio_merge_label",
                            )

                            if op == "join":
                                key_expr = join_keys or ["id"]
                                join_code = (
                                    "import pandas as pd\n\n"
                                    "# df_0, df_1, ... are available\n"
                                    f"df = df_0.merge(df_1, on={key_expr!r}, how={join_how!r})\n"
                                )
                            else:
                                join_code = (
                                    "import pandas as pd\n\n"
                                    "# df_0, df_1, ... are available\n"
                                    f"df = pd.concat([df_0, df_1], axis={concat_axis}, ignore_index={concat_ignore_index if concat_axis == 0 else False})\n"
                                )
                            st.code(join_code, language="python")
                            if st.button(
                                "Create merge node",
                                key="pipeline_studio_merge_create",
                                width="stretch",
                            ):
                                _pipeline_studio_create_manual_merge_node(
                                    parent_ids=[left_id, right_id],
                                    stage=stage,
                                    label=label,
                                    code=join_code,
                                )

                m = meta_by_id.get(selected_node_id) or {}
                with st.expander("Selected step details", expanded=False):
                    st.json(m)

                entry_obj = {}
                prov = {}
                transform = {}
                try:
                    entry_obj = (
                        studio_datasets.get(selected_node_id)
                        if isinstance(studio_datasets, dict)
                        else None
                    )
                    entry_obj = entry_obj if isinstance(entry_obj, dict) else {}
                    prov = (
                        entry_obj.get("provenance")
                        if isinstance(entry_obj.get("provenance"), dict)
                        else {}
                    )
                    transform = (
                        prov.get("transform")
                        if isinstance(prov.get("transform"), dict)
                        else {}
                    )
                    title, code_text, code_lang, kind = (
                        _pipeline_studio_transform_code_snippet(transform)
                    )
                except Exception:
                    title, code_text, code_lang, kind = None, None, "python", ""

                with st.expander("Node inspector actions", expanded=False):
                    st.markdown("**Edit code + rerun**")
                    if kind not in {"python_function", "python_merge", "sql_query"}:
                        st.info("No editable code for this step.")
                    elif not (isinstance(code_text, str) and code_text.strip()):
                        st.info("No runnable code recorded for this step yet.")
                    else:
                        inline_key = f"pipeline_studio_inline_editor_{selected_node_id}"
                        fp = entry_obj.get("fingerprint")
                        fp = fp if isinstance(fp, str) and fp else None
                        saved_draft = None
                        saved_meta = {}
                        if fp:
                            saved_meta = _get_pipeline_studio_code_draft(fingerprint=fp)
                            saved_draft = (
                                saved_meta.get("draft_code")
                                if isinstance(saved_meta, dict)
                                else None
                            )
                            saved_draft = (
                                saved_draft
                                if isinstance(saved_draft, str) and saved_draft.strip()
                                else None
                            )
                        if inline_key not in st.session_state:
                            st.session_state[inline_key] = (
                                saved_draft or code_text or ""
                            )
                        draft_code = st.text_area(
                            "Draft editor",
                            key=inline_key,
                            height=220,
                            help="Edit the snippet and rerun this step locally.",
                        )
                        show_preview = st.checkbox(
                            "Show formatted preview",
                            value=False,
                            key=f"pipeline_studio_inline_preview_{selected_node_id}",
                        )
                        if show_preview:
                            st.code(
                                draft_code if isinstance(draft_code, str) else "",
                                language="sql" if code_lang == "sql" else "python",
                            )
                        if saved_draft:
                            st.caption("Loaded saved draft from `pipeline_store/`.")
                        c_save, c_reset = st.columns(2)
                        with c_save:

                            def _save_inline_draft(
                                fingerprint: str | None,
                                node_id: str,
                                e_key: str,
                                t_kind: str,
                                lang: str,
                            ) -> None:
                                if not isinstance(fingerprint, str) or not fingerprint:
                                    return
                                code = st.session_state.get(e_key)
                                code = code if isinstance(code, str) else ""
                                _save_pipeline_studio_code_draft(
                                    fingerprint=fingerprint,
                                    dataset_id=node_id,
                                    transform_kind=t_kind,
                                    lang=lang,
                                    draft_code=code,
                                )
                                st.session_state["pipeline_studio_code_draft_saved"] = (
                                    node_id
                                )

                            st.button(
                                "Save draft",
                                key=f"pipeline_studio_inline_save_{selected_node_id}",
                                on_click=_save_inline_draft,
                                args=(
                                    fp,
                                    selected_node_id,
                                    inline_key,
                                    kind,
                                    code_lang,
                                ),
                                width="stretch",
                            )
                        with c_reset:

                            def _reset_inline_draft(
                                node_id: str, fingerprint: str | None
                            ) -> None:
                                st.session_state.pop(inline_key, None)
                                if isinstance(fingerprint, str) and fingerprint:
                                    _delete_pipeline_studio_code_draft(
                                        fingerprint=fingerprint
                                    )
                                st.session_state["pipeline_studio_history_notice"] = (
                                    f"Reset draft for `{node_id}`."
                                )

                            st.button(
                                "Reset draft",
                                key=f"pipeline_studio_inline_reset_{selected_node_id}",
                                on_click=_reset_inline_draft,
                                args=(selected_node_id, fp),
                                width="stretch",
                            )

                        st.markdown("**Run draft (local)**")
                        confirm_label = (
                            "I understand this runs a read-only SQL query"
                            if kind == "sql_query"
                            else "I understand this executes code locally"
                        )
                        confirmed = st.checkbox(
                            confirm_label,
                            key=f"pipeline_studio_inline_confirm_{selected_node_id}",
                        )
                        replace_mode = st.checkbox(
                            "Replace mode (auto-hide old branch)",
                            value=True,
                            key=f"pipeline_studio_inline_replace_{selected_node_id}",
                            help="After a successful rerun, hide the superseded branch.",
                        )

                        def _run_inline_draft(nid: str, e_key: str, k: str) -> None:
                            if k == "python_function":
                                _run_python_function_draft(
                                    node_id=nid, editor_key=e_key
                                )
                            elif k == "python_merge":
                                _run_python_merge_draft(node_id=nid, editor_key=e_key)
                            elif k == "sql_query":
                                _run_sql_query_draft(node_id=nid, editor_key=e_key)

                        def _run_inline_draft_downstream(
                            nid: str, e_key: str, k: str, do_replace: bool
                        ) -> None:
                            _run_inline_draft(nid, e_key, k)
                            err = st.session_state.get("pipeline_studio_run_error")
                            if isinstance(err, str) and err.strip():
                                return
                            last_repl = st.session_state.get(
                                "pipeline_studio_last_replacement"
                            )
                            if not isinstance(last_repl, dict):
                                return
                            old_id = last_repl.get("old_id")
                            new_id = last_repl.get("new_id")
                            old_id = (
                                old_id.strip()
                                if isinstance(old_id, str) and old_id.strip()
                                else None
                            )
                            new_id = (
                                new_id.strip()
                                if isinstance(new_id, str) and new_id.strip()
                                else None
                            )
                            if not old_id or not new_id:
                                return
                            _run_downstream_transforms(old_id, new_id)
                            if bool(do_replace and pipeline_hash):
                                err = st.session_state.get("pipeline_studio_run_error")
                                if not (isinstance(err, str) and err.strip()):
                                    _pipeline_studio_hide_branch(
                                        pipeline_hash=pipeline_hash, root_id=old_id
                                    )

                        r1, r2 = st.columns(2)
                        with r1:
                            st.button(
                                "Run draft only",
                                key=f"pipeline_studio_inline_run_{selected_node_id}",
                                disabled=not bool(confirmed),
                                on_click=_run_inline_draft,
                                args=(selected_node_id, inline_key, kind),
                                width="stretch",
                            )
                        with r2:
                            st.button(
                                "Run draft + downstream",
                                key=f"pipeline_studio_inline_run_downstream_{selected_node_id}",
                                type="primary",
                                disabled=not bool(confirmed),
                                on_click=_run_inline_draft_downstream,
                                args=(
                                    selected_node_id,
                                    inline_key,
                                    kind,
                                    bool(replace_mode),
                                ),
                                width="stretch",
                            )

                    st.markdown("---")
                    st.markdown("**Delete subgraph**")
                    branch_ids = _pipeline_studio_branch_ids(selected_node_id)
                    st.caption(f"Branch size: {len(branch_ids)} node(s).")
                    confirm_soft = st.checkbox(
                        "Confirm soft-delete",
                        key=f"pipeline_studio_inline_delete_confirm_{selected_node_id}",
                        help="Marks this node and descendants as deleted in the registry (reversible).",
                    )
                    st.button(
                        "Delete subgraph (soft)",
                        key=f"pipeline_studio_inline_delete_{selected_node_id}",
                        type="secondary",
                        disabled=not bool(confirm_soft and pipeline_hash),
                        on_click=_pipeline_studio_soft_delete_branch,
                        args=(),
                        kwargs={
                            "pipeline_hash": pipeline_hash,
                            "root_id": selected_node_id,
                        },
                        width="stretch",
                    )
                    st.button(
                        "Restore subgraph",
                        key=f"pipeline_studio_inline_restore_{selected_node_id}",
                        disabled=not bool(pipeline_hash),
                        on_click=_pipeline_studio_restore_branch,
                        args=(),
                        kwargs={
                            "pipeline_hash": pipeline_hash,
                            "root_id": selected_node_id,
                        },
                        width="stretch",
                    )
                    st.markdown("---")
                    clear_history = st.checkbox(
                        "Also clear undo/redo history",
                        value=True,
                        key=f"pipeline_studio_inline_clear_history_{selected_node_id}",
                    )
                    confirm_hard = st.checkbox(
                        "I understand this permanently deletes data",
                        key=f"pipeline_studio_inline_hard_confirm_{selected_node_id}",
                    )
                    st.button(
                        "Hard delete subgraph (permanent)",
                        key=f"pipeline_studio_inline_hard_delete_{selected_node_id}",
                        type="primary",
                        disabled=not bool(confirm_hard),
                        on_click=_pipeline_studio_hard_delete_branch,
                        args=(),
                        kwargs={
                            "pipeline_hash": pipeline_hash,
                            "root_id": selected_node_id,
                            "clear_history": bool(clear_history),
                        },
                        width="stretch",
                    )

                if st.button(
                    "Ask AI about this step",
                    key="pipeline_studio_ask_ai_about_step",
                    help="Sends the selected pipeline step (and code, when available) to the chat for help.",
                ):
                    prompt_lines = [
                        "Pipeline Studio request: help me understand and improve this pipeline step.",
                        f"selected_node_id: {selected_node_id}",
                        f"label: {m.get('label')}"
                        if isinstance(m, dict) and m.get("label")
                        else "",
                        f"stage: {m.get('stage')}"
                        if isinstance(m, dict) and m.get("stage")
                        else "",
                        f"transform_kind: {kind}" if kind else "",
                    ]
                    if isinstance(code_text, str) and code_text.strip():
                        prompt_lines.extend(
                            [
                                "",
                                f"{title or 'Code'}:",
                                f"```{code_lang}\n{code_text}\n```",
                            ]
                        )
                    st.session_state["chat_prompt_pending"] = "\n".join(
                        [x for x in prompt_lines if isinstance(x, str) and x.strip()]
                    ).strip()
                    st.rerun()

                compare_node_id = None
                compare_mode = st.checkbox(
                    "Compare mode",
                    value=bool(
                        st.session_state.get("pipeline_studio_compare_mode", False)
                    ),
                    key="pipeline_studio_compare_mode",
                    help="Compare two pipeline steps side-by-side (schema + table preview).",
                )
                if compare_mode:
                    compare_candidates = (
                        selectable_node_ids
                        if isinstance(selectable_node_ids, list)
                        else node_ids
                    )
                    compare_options = [
                        did for did in compare_candidates if did != selected_node_id
                    ]
                    compare_options = compare_options or compare_candidates
                    default_compare = (
                        compare_options[-1] if compare_options else selected_node_id
                    )
                    current_compare = st.session_state.get(
                        "pipeline_studio_compare_node_id"
                    )
                    if (
                        not isinstance(current_compare, str)
                        or current_compare not in compare_options
                    ):
                        st.session_state["pipeline_studio_compare_node_id"] = (
                            default_compare
                        )

                    compare_node_id = st.selectbox(
                        "Compare with",
                        options=compare_options,
                        format_func=_node_label,
                        key="pipeline_studio_compare_node_id",
                    )

            with right:
                if (
                    compare_mode
                    and isinstance(compare_node_id, str)
                    and compare_node_id
                ):
                    a_id = selected_node_id
                    b_id = compare_node_id

                    a_entry = (
                        studio_datasets.get(a_id)
                        if isinstance(studio_datasets, dict)
                        else None
                    )
                    b_entry = (
                        studio_datasets.get(b_id)
                        if isinstance(studio_datasets, dict)
                        else None
                    )
                    a_entry = a_entry if isinstance(a_entry, dict) else {}
                    b_entry = b_entry if isinstance(b_entry, dict) else {}
                    df_a = _dataset_entry_to_df(a_entry)
                    df_b = _dataset_entry_to_df(b_entry)

                    st.caption(f"Comparing `{a_id}` vs `{b_id}`")

                    col_map_a = (
                        {str(c): c for c in list(df_a.columns)}
                        if df_a is not None
                        else {}
                    )
                    col_map_b = (
                        {str(c): c for c in list(df_b.columns)}
                        if df_b is not None
                        else {}
                    )
                    cols_a = (
                        [str(c) for c in list(df_a.columns)]
                        if df_a is not None
                        else (
                            [str(c) for c in a_entry.get("columns")]
                            if isinstance(a_entry.get("columns"), list)
                            else []
                        )
                    )
                    cols_b = (
                        [str(c) for c in list(df_b.columns)]
                        if df_b is not None
                        else (
                            [str(c) for c in b_entry.get("columns")]
                            if isinstance(b_entry.get("columns"), list)
                            else []
                        )
                    )
                    removed: list[str] = []
                    added: list[str] = []
                    shared: list[str] = []
                    if cols_a and cols_b:
                        set_a = set(cols_a)
                        set_b = set(cols_b)
                        # Interpret diff as changes in the selected step (A) relative to the compare step (B).
                        # - Removed: present in B, missing in A
                        # - Added: present in A, missing in B
                        removed = sorted(set_b - set_a)
                        added = sorted(set_a - set_b)
                        shared = sorted(set_a.intersection(set_b))

                    def _get_plotly_graph_json(dataset_id: str, entry_obj: dict):
                        graph_json = None
                        idx_map = st.session_state.get("pipeline_studio_artifacts")
                        if isinstance(idx_map, dict):
                            entry_art = idx_map.get(dataset_id)
                            entry_art = entry_art if isinstance(entry_art, dict) else {}
                            pg = entry_art.get("plotly_graph")
                            pg = pg if isinstance(pg, dict) else {}
                            graph_json = pg.get("json")
                        if not graph_json:
                            detail = _latest_detail_for_dataset_id(
                                dataset_id, require_key="plotly_graph"
                            )
                            graph_json = (
                                detail.get("plotly_graph")
                                if isinstance(detail, dict)
                                else None
                            )
                        if not graph_json and isinstance(entry_obj, dict):
                            fp = entry_obj.get("fingerprint")
                            fp = fp if isinstance(fp, str) and fp else None
                            if fp:
                                persisted = _get_persisted_pipeline_studio_artifacts(
                                    fingerprint=fp
                                )
                                pg = persisted.get("plotly_graph")
                                pg = pg if isinstance(pg, dict) else {}
                                graph_json = pg.get("json")
                        return graph_json

                    def _render_plotly_graph(graph_json, *, widget_key: str) -> None:
                        payload = (
                            json.dumps(graph_json)
                            if isinstance(graph_json, dict)
                            else graph_json
                        )
                        fig = _apply_streamlit_plot_style(pio.from_json(payload))
                        st.plotly_chart(fig, width="stretch", key=widget_key)

                    def _build_code_snippet(entry_obj: dict):
                        prov = (
                            entry_obj.get("provenance")
                            if isinstance(entry_obj.get("provenance"), dict)
                            else {}
                        )
                        transform = (
                            prov.get("transform")
                            if isinstance(prov.get("transform"), dict)
                            else {}
                        )
                        kind = str(transform.get("kind") or "")
                        code_lang = "python"
                        title = None
                        code_text = None
                        if kind == "python_function":
                            title = "Transform function (Python)"
                            code_text = transform.get("function_code")
                        elif kind == "sql_query":
                            title = "SQL query"
                            code_text = transform.get("sql_query_code")
                            code_lang = "sql"
                        elif kind == "python_merge":
                            title = "Merge code (Python)"
                            code_text = transform.get("merge_code")
                        elif kind == "mlflow_predict":
                            run_id = transform.get("run_id")
                            run_id = run_id.strip() if isinstance(run_id, str) else ""
                            title = "Prediction (MLflow) snippet"
                            code_text = (
                                "\n".join(
                                    [
                                        "import pandas as pd",
                                        "import mlflow",
                                        "",
                                        f"model_uri = 'runs:/{run_id}/model'",
                                        "model = mlflow.pyfunc.load_model(model_uri)",
                                        "preds = model.predict(df)",
                                        "df_preds = preds if isinstance(preds, pd.DataFrame) else pd.DataFrame(preds)",
                                    ]
                                ).strip()
                                + "\n"
                            )
                        elif kind == "h2o_predict":
                            model_id = transform.get("model_id")
                            model_id = (
                                model_id.strip() if isinstance(model_id, str) else ""
                            )
                            title = "Prediction (H2O) snippet"
                            code_text = (
                                "\n".join(
                                    [
                                        "import h2o",
                                        "",
                                        "h2o.init()",
                                        f"model = h2o.get_model('{model_id}')",
                                        "frame = h2o.H2OFrame(df)",
                                        "preds = model.predict(frame)",
                                        "df_preds = preds.as_data_frame(use_pandas=True)",
                                    ]
                                ).strip()
                                + "\n"
                            )
                        code_text = (
                            code_text
                            if isinstance(code_text, str) and code_text.strip()
                            else None
                        )
                        return title, code_text, code_lang, kind

                    cmp_tabs = st.tabs(
                        [
                            "Schema diff",
                            "Table preview",
                            "Chart compare",
                            "Code compare",
                            "Row diff",
                        ]
                    )
                    with cmp_tabs[0]:
                        st.caption(
                            "Diff is shown as changes in the selected step (A) relative to the compare step (B)."
                        )
                        if not cols_a or not cols_b:
                            st.info(
                                "Could not compute schema diff (missing column metadata). "
                                "Try selecting steps with tabular data."
                            )
                        else:
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown(f"**Removed columns ({len(removed)})**")
                                st.code(
                                    "\n".join(removed) if removed else "—",
                                    language="text",
                                )
                            with c2:
                                st.markdown(f"**Added columns ({len(added)})**")
                                st.code(
                                    "\n".join(added) if added else "—",
                                    language="text",
                                )

                            if df_a is not None and df_b is not None and shared:
                                changes = []
                                try:
                                    for col in shared:
                                        col_a = col_map_a.get(col)
                                        col_b = col_map_b.get(col)
                                        dt_a = (
                                            str(df_a[col_a].dtype)
                                            if col_a is not None
                                            else ""
                                        )
                                        dt_b = (
                                            str(df_b[col_b].dtype)
                                            if col_b is not None
                                            else ""
                                        )
                                        if dt_a != dt_b:
                                            changes.append(
                                                {
                                                    "column": col,
                                                    "dtype_compare": dt_b,
                                                    "dtype_selected": dt_a,
                                                }
                                            )
                                except Exception:
                                    changes = []
                                if changes:
                                    st.markdown("**Dtype changes**")
                                    st.dataframe(
                                        pd.DataFrame(changes),
                                        width="stretch",
                                    )
                                else:
                                    st.caption(
                                        "No dtype changes detected (pandas dtypes)."
                                    )

                                with st.expander(
                                    "Missingness delta (sampled)", expanded=False
                                ):
                                    try:
                                        max_rows = 5000
                                        max_cols = 200
                                        shared_sample = [
                                            c
                                            for c in shared[:max_cols]
                                            if c in col_map_a and c in col_map_b
                                        ]
                                        if not shared_sample:
                                            st.caption(
                                                "No shared columns available for missingness comparison."
                                            )
                                        else:
                                            cols_a_sel = [
                                                col_map_a[c] for c in shared_sample
                                            ]
                                            cols_b_sel = [
                                                col_map_b[c] for c in shared_sample
                                            ]
                                            sample_a = df_a[cols_a_sel].head(max_rows)
                                            sample_b = df_b[cols_b_sel].head(max_rows)
                                            sample_a.columns = shared_sample
                                            sample_b.columns = shared_sample
                                            miss_a = sample_a.isna().sum()
                                            miss_b = sample_b.isna().sum()
                                            miss_df = pd.DataFrame(
                                                {
                                                    "column": shared_sample,
                                                    f"missing_compare (n={len(sample_b)})": miss_b.values,
                                                    f"missing_selected (n={len(sample_a)})": miss_a.values,
                                                    "delta (selected-compare)": (
                                                        miss_a - miss_b
                                                    ).values,
                                                }
                                            )
                                            miss_df = miss_df[
                                                miss_df["delta (selected-compare)"] != 0
                                            ]
                                            if len(miss_df) == 0:
                                                st.caption(
                                                    "No missingness delta detected in sampled rows."
                                                )
                                            else:
                                                miss_df["abs_delta"] = miss_df[
                                                    "delta (selected-compare)"
                                                ].abs()
                                                miss_df = miss_df.sort_values(
                                                    "abs_delta", ascending=False
                                                ).drop(columns=["abs_delta"])
                                                st.dataframe(
                                                    miss_df.head(50),
                                                    width="stretch",
                                                )
                                                if len(shared) > max_cols:
                                                    st.caption(
                                                        f"Missingness computed on first {max_cols} shared columns."
                                                    )
                                    except Exception as e:
                                        st.caption(
                                            f"Could not compute missingness delta: {e}"
                                        )

                    with cmp_tabs[1]:
                        rows = st.slider(
                            "Preview rows",
                            min_value=5,
                            max_value=200,
                            value=25,
                            step=5,
                            key="pipeline_studio_compare_preview_rows",
                        )
                        ca, cb = st.columns(2, gap="large")
                        with ca:
                            st.markdown(f"**A (selected): {_node_label(a_id)}**")
                            if df_a is None:
                                st.info("No tabular data available for A.")
                            else:
                                st.caption(f"Shape: {df_a.shape[0]} × {df_a.shape[1]}")
                                st.dataframe(
                                    df_a.head(int(rows)), width="stretch"
                                )
                        with cb:
                            st.markdown(f"**B (compare): {_node_label(b_id)}**")
                            if df_b is None:
                                st.info("No tabular data available for B.")
                            else:
                                st.caption(f"Shape: {df_b.shape[0]} × {df_b.shape[1]}")
                                st.dataframe(
                                    df_b.head(int(rows)), width="stretch"
                                )

                    with cmp_tabs[2]:
                        ga = _get_plotly_graph_json(a_id, a_entry)
                        gb = _get_plotly_graph_json(b_id, b_entry)
                        if not ga and not gb:
                            st.info(
                                "No charts found for either step. Try generating a chart while each dataset is active."
                            )
                        else:
                            ca, cb = st.columns(2, gap="large")
                            with ca:
                                st.markdown(f"**A (selected): {_node_label(a_id)}**")
                                if not ga:
                                    st.info("No chart found for A.")
                                else:
                                    try:
                                        _render_plotly_graph(
                                            ga,
                                            widget_key=f"pipeline_studio_compare_chart_a_{a_id}",
                                        )
                                    except Exception as e:
                                        st.error(f"Error rendering chart A: {e}")
                            with cb:
                                st.markdown(f"**B (compare): {_node_label(b_id)}**")
                                if not gb:
                                    st.info("No chart found for B.")
                                else:
                                    try:
                                        _render_plotly_graph(
                                            gb,
                                            widget_key=f"pipeline_studio_compare_chart_b_{b_id}",
                                        )
                                    except Exception as e:
                                        st.error(f"Error rendering chart B: {e}")

                    with cmp_tabs[3]:
                        title_a, code_a, lang_a, kind_a = _build_code_snippet(a_entry)
                        title_b, code_b, lang_b, kind_b = _build_code_snippet(b_entry)
                        ca, cb = st.columns(2, gap="large")
                        with ca:
                            st.markdown(
                                f"**A (selected): {title_a or kind_a or 'Code'}**"
                            )
                            if code_a:
                                st.code(code_a, language=lang_a)
                                _render_copy_to_clipboard(code_a, label="Copy A")
                            else:
                                st.info("No runnable code recorded for A.")
                        with cb:
                            st.markdown(
                                f"**B (compare): {title_b or kind_b or 'Code'}**"
                            )
                            if code_b:
                                st.code(code_b, language=lang_b)
                                _render_copy_to_clipboard(code_b, label="Copy B")
                            else:
                                st.info("No runnable code recorded for B.")

                        if code_a and code_b:
                            with st.expander(
                                "Unified diff (compare → selected)", expanded=False
                            ):
                                try:
                                    import difflib

                                    diff_lines = difflib.unified_diff(
                                        code_b.splitlines(),
                                        code_a.splitlines(),
                                        fromfile="compare",
                                        tofile="selected",
                                        lineterm="",
                                    )
                                    diff_text = "\n".join(diff_lines).strip()
                                    st.code(
                                        diff_text if diff_text else "—", language="diff"
                                    )
                                except Exception as e:
                                    st.caption(f"Could not compute diff: {e}")

                    with cmp_tabs[4]:
                        if df_a is None or df_b is None:
                            st.info(
                                "Row diff requires tabular data for both A and B (DataFrames)."
                            )
                        elif not shared:
                            st.info("No shared columns available for row diff.")
                        else:
                            key_options = shared[:200]
                            current_key = st.session_state.get(
                                "pipeline_studio_compare_rowdiff_key"
                            )
                            if (
                                not isinstance(current_key, str)
                                or current_key not in key_options
                            ):
                                st.session_state[
                                    "pipeline_studio_compare_rowdiff_key"
                                ] = key_options[0]
                            key_col = st.selectbox(
                                "Key column",
                                options=key_options,
                                key="pipeline_studio_compare_rowdiff_key",
                                help="Used to align rows between A (selected) and B (compare).",
                            )
                            compare_candidates = [
                                c for c in key_options if c != key_col
                            ]
                            compare_candidates = compare_candidates[:50]
                            default_cols = compare_candidates[:10]
                            current_cols = st.session_state.get(
                                "pipeline_studio_compare_rowdiff_cols"
                            )
                            if not isinstance(current_cols, list) or any(
                                c not in compare_candidates for c in current_cols
                            ):
                                st.session_state[
                                    "pipeline_studio_compare_rowdiff_cols"
                                ] = default_cols
                            cols_to_compare = st.multiselect(
                                "Columns to compare",
                                options=compare_candidates,
                                key="pipeline_studio_compare_rowdiff_cols",
                                help="Limit columns for faster diffing.",
                            )
                            preview_rows = st.slider(
                                "Preview rows",
                                min_value=5,
                                max_value=200,
                                value=25,
                                step=5,
                                key="pipeline_studio_compare_rowdiff_preview_rows",
                            )

                            col_a = col_map_a.get(key_col)
                            col_b = col_map_b.get(key_col)
                            if col_a is None or col_b is None:
                                st.info("Key column not available in both DataFrames.")
                            else:
                                cols_a_actual = [col_a] + [
                                    col_map_a[c]
                                    for c in cols_to_compare
                                    if c in col_map_a
                                ]
                                cols_b_actual = [col_b] + [
                                    col_map_b[c]
                                    for c in cols_to_compare
                                    if c in col_map_b
                                ]
                                a_small = df_a[cols_a_actual].copy()
                                b_small = df_b[cols_b_actual].copy()
                                a_small.columns = [key_col] + [
                                    c for c in cols_to_compare if c in col_map_a
                                ]
                                b_small.columns = [key_col] + [
                                    c for c in cols_to_compare if c in col_map_b
                                ]

                                dup_a = bool(a_small[key_col].duplicated().any())
                                dup_b = bool(b_small[key_col].duplicated().any())
                                if dup_a or dup_b:
                                    st.warning(
                                        "Key column contains duplicates; row diff uses the first occurrence per key."
                                    )
                                    a_small = a_small.drop_duplicates(
                                        subset=[key_col], keep="first"
                                    )
                                    b_small = b_small.drop_duplicates(
                                        subset=[key_col], keep="first"
                                    )

                                a_keys = a_small[key_col].dropna()
                                b_keys = b_small[key_col].dropna()
                                try:
                                    set_a_keys = set(a_keys.unique().tolist())
                                except Exception:
                                    set_a_keys = set(
                                        a_keys.astype(str).unique().tolist()
                                    )
                                try:
                                    set_b_keys = set(b_keys.unique().tolist())
                                except Exception:
                                    set_b_keys = set(
                                        b_keys.astype(str).unique().tolist()
                                    )

                                only_a = set_a_keys - set_b_keys
                                only_b = set_b_keys - set_a_keys
                                both = set_a_keys.intersection(set_b_keys)

                                m1, m2, m3, m4 = st.columns(4)
                                with m1:
                                    st.metric("Keys in A", len(set_a_keys))
                                with m2:
                                    st.metric("Keys in B", len(set_b_keys))
                                with m3:
                                    st.metric("Only in A", len(only_a))
                                with m4:
                                    st.metric("Only in B", len(only_b))

                                if only_a:
                                    with st.expander(
                                        "Sample rows only in A (selected)",
                                        expanded=False,
                                    ):
                                        sample_keys = list(only_a)[: int(preview_rows)]
                                        st.dataframe(
                                            a_small[
                                                a_small[key_col].isin(sample_keys)
                                            ].head(int(preview_rows)),
                                            width="stretch",
                                        )
                                if only_b:
                                    with st.expander(
                                        "Sample rows only in B (compare)",
                                        expanded=False,
                                    ):
                                        sample_keys = list(only_b)[: int(preview_rows)]
                                        st.dataframe(
                                            b_small[
                                                b_small[key_col].isin(sample_keys)
                                            ].head(int(preview_rows)),
                                            width="stretch",
                                        )

                                if not cols_to_compare:
                                    st.caption(
                                        "Select one or more columns to compare for per-key value diffs."
                                    )
                                elif not both:
                                    st.caption(
                                        "No overlapping keys between A and B for detailed value comparison."
                                    )
                                else:
                                    merged = a_small.merge(
                                        b_small,
                                        on=key_col,
                                        how="inner",
                                        suffixes=("_selected", "_compare"),
                                    )
                                    diff_counts = []
                                    for col in cols_to_compare:
                                        a_col = f"{col}_selected"
                                        b_col = f"{col}_compare"
                                        if (
                                            a_col not in merged.columns
                                            or b_col not in merged.columns
                                        ):
                                            continue
                                        a_vals = merged[a_col]
                                        b_vals = merged[b_col]
                                        try:
                                            neq = ~(
                                                a_vals.eq(b_vals)
                                                | (a_vals.isna() & b_vals.isna())
                                            )
                                        except Exception:
                                            neq = a_vals.astype(str) != b_vals.astype(
                                                str
                                            )
                                        n_bad = int(neq.sum())
                                        if n_bad:
                                            diff_counts.append(
                                                {
                                                    "column": col,
                                                    "mismatched_rows": n_bad,
                                                }
                                            )
                                    if not diff_counts:
                                        st.caption(
                                            "No per-key value differences detected in the selected compare columns."
                                        )
                                    else:
                                        diff_df = pd.DataFrame(diff_counts).sort_values(
                                            "mismatched_rows", ascending=False
                                        )
                                        st.markdown("**Value diffs (by key)**")
                                        st.dataframe(diff_df, width="stretch")
                                        inspect_default = str(diff_df.iloc[0]["column"])
                                        current_inspect = st.session_state.get(
                                            "pipeline_studio_compare_rowdiff_inspect_col"
                                        )
                                        inspect_options = [
                                            str(x) for x in diff_df["column"].tolist()
                                        ]
                                        if (
                                            not isinstance(current_inspect, str)
                                            or current_inspect not in inspect_options
                                        ):
                                            st.session_state[
                                                "pipeline_studio_compare_rowdiff_inspect_col"
                                            ] = inspect_default
                                        inspect_col = st.selectbox(
                                            "Inspect column",
                                            options=inspect_options,
                                            key="pipeline_studio_compare_rowdiff_inspect_col",
                                        )
                                        a_col = f"{inspect_col}_selected"
                                        b_col = f"{inspect_col}_compare"
                                        try:
                                            neq = ~(
                                                merged[a_col].eq(merged[b_col])
                                                | (
                                                    merged[a_col].isna()
                                                    & merged[b_col].isna()
                                                )
                                            )
                                        except Exception:
                                            neq = merged[a_col].astype(str) != merged[
                                                b_col
                                            ].astype(str)
                                        preview = (
                                            merged.loc[neq, [key_col, a_col, b_col]]
                                            .head(int(preview_rows))
                                            .copy()
                                        )
                                        preview = preview.rename(
                                            columns={
                                                a_col: f"{inspect_col} (selected)",
                                                b_col: f"{inspect_col} (compare)",
                                            }
                                        )
                                        st.dataframe(preview, width="stretch")

                    # Compare mode replaces the workspace when enabled.
                    return
                if "pipeline_studio_view" not in st.session_state:
                    st.session_state["pipeline_studio_view"] = "Visual Editor"
                pending_view = st.session_state.pop(
                    "pipeline_studio_view_pending", None
                )
                valid_views = {
                    "Table",
                    "Chart",
                    "EDA",
                    "Code",
                    "Model",
                    "Predictions",
                    "MLflow",
                    "Visual Editor",
                }
                if isinstance(pending_view, str) and pending_view in valid_views:
                    st.session_state["pipeline_studio_view"] = pending_view
                entry = (
                    studio_datasets.get(selected_node_id)
                    if isinstance(studio_datasets, dict)
                    else None
                )
                entry = entry if isinstance(entry, dict) else {}
                df_sel = _dataset_entry_to_df(entry)
                idx_map = st.session_state.get("pipeline_studio_artifacts")
                idx_map = idx_map if isinstance(idx_map, dict) else {}
                entry_art = idx_map.get(selected_node_id)
                entry_art = entry_art if isinstance(entry_art, dict) else {}
                fp = entry.get("fingerprint")
                fp = fp if isinstance(fp, str) and fp else None
                persisted_art = (
                    _get_persisted_pipeline_studio_artifacts(fingerprint=fp)
                    if fp
                    else {}
                )

                def _artifact_payload(
                    key: str, *, detail_key: str | None = None, field: str | None = None
                ) -> object | None:
                    rec = entry_art.get(key) if isinstance(entry_art, dict) else None
                    rec = rec if isinstance(rec, dict) else {}
                    if field and isinstance(rec.get(field), (dict, list, str)):
                        return rec.get(field)
                    if detail_key:
                        detail = _latest_detail_for_dataset_id(
                            selected_node_id, require_key=detail_key
                        )
                        if (
                            isinstance(detail, dict)
                            and detail.get(detail_key) is not None
                        ):
                            return detail.get(detail_key)
                    persisted = (
                        persisted_art.get(key)
                        if isinstance(persisted_art, dict)
                        else None
                    )
                    persisted = persisted if isinstance(persisted, dict) else {}
                    if field and persisted.get(field) is not None:
                        return persisted.get(field)
                    return None

                chart_json = _artifact_payload(
                    "plotly_graph", detail_key="plotly_graph", field="json"
                )
                chart_count = 1 if chart_json else 0
                eda_reports = _artifact_payload(
                    "eda_reports", detail_key="eda_reports", field="reports"
                )
                if isinstance(eda_reports, dict):
                    eda_count = sum(1 for v in eda_reports.values() if v)
                else:
                    eda_count = 1 if eda_reports else 0
                model_info = _artifact_payload(
                    "model_info", detail_key="model_info", field="info"
                )
                eval_art = _artifact_payload("eval_artifacts", field="artifacts")
                eval_graph = _artifact_payload("eval_plotly_graph", field="json")
                model_count = sum(1 for v in (model_info, eval_art, eval_graph) if v)
                mlflow_art = _artifact_payload(
                    "mlflow_artifacts", detail_key="mlflow_artifacts", field="artifacts"
                )
                mlflow_count = 1 if mlflow_art else 0
                prov = (
                    entry.get("provenance")
                    if isinstance(entry.get("provenance"), dict)
                    else {}
                )
                transform = (
                    prov.get("transform")
                    if isinstance(prov.get("transform"), dict)
                    else {}
                )
                kind = str(transform.get("kind") or "")
                pred_count = 1 if kind in {"mlflow_predict", "h2o_predict"} else 0
                view_badges = {
                    "Chart": int(chart_count),
                    "EDA": int(eda_count),
                    "Model": int(model_count),
                    "Predictions": int(pred_count),
                    "MLflow": int(mlflow_count),
                }

                def _view_label(view_name: str) -> str:
                    count = view_badges.get(view_name, 0)
                    return f"{view_name} ({count})" if count else view_name

                view = st.radio(
                    "Workspace",
                    [
                        "Visual Editor",
                        "Table",
                        "Chart",
                        "EDA",
                        "Code",
                        "Model",
                        "Predictions",
                        "MLflow",
                    ],
                    horizontal=True,
                    format_func=_view_label,
                    key="pipeline_studio_view",
                )

                if view == "Table":
                    if df_sel is None:
                        st.info("No tabular data available for this pipeline step.")
                    else:
                        n_rows = int(getattr(df_sel, "shape", (0, 0))[0] or 0)
                        n_cols = int(getattr(df_sel, "shape", (0, 0))[1] or 0)
                        st.caption(f"Shape: {n_rows} rows × {n_cols} columns")
                        rows = st.slider(
                            "Preview rows",
                            min_value=5,
                            max_value=200,
                            value=25,
                            step=5,
                            key="pipeline_studio_preview_rows",
                        )
                        st.dataframe(df_sel.head(int(rows)), width="stretch")
                        try:
                            cols = (
                                entry.get("columns")
                                if isinstance(entry.get("columns"), list)
                                else [str(c) for c in list(df_sel.columns)]
                            )
                            schema_hash = (
                                entry.get("schema_hash")
                                if isinstance(entry.get("schema_hash"), str)
                                else None
                            )
                            fingerprint = (
                                entry.get("fingerprint")
                                if isinstance(entry.get("fingerprint"), str)
                                else None
                            )
                            with st.expander("Schema summary", expanded=False):
                                if schema_hash or fingerprint:
                                    st.json(
                                        {
                                            "schema_hash": schema_hash,
                                            "fingerprint": fingerprint,
                                        }
                                    )
                                if cols:
                                    max_cols = 200
                                    shown = [str(c) for c in cols[:max_cols]]
                                    st.markdown(f"**Columns ({len(cols)})**")
                                    st.code("\n".join(shown), language="text")
                                    if len(cols) > max_cols:
                                        st.caption(f"Showing first {max_cols} columns.")
                                else:
                                    st.info("No column metadata available.")

                                max_rows = 5000
                                sample = df_sel.head(max_rows)
                                missing = None
                                try:
                                    missing = sample.isna().sum()
                                except Exception:
                                    missing = None
                                if missing is not None:
                                    try:
                                        missing = missing[missing > 0].sort_values(
                                            ascending=False
                                        )
                                    except Exception:
                                        missing = None
                                if missing is not None and len(missing) > 0:
                                    miss_df = missing.head(25).reset_index()
                                    miss_df.columns = [
                                        "column",
                                        f"missing_count (first {len(sample)} rows)",
                                    ]
                                    st.markdown("**Missingness (sampled)**")
                                    st.dataframe(miss_df, width="stretch")
                                else:
                                    st.caption(
                                        f"No missing values detected in first {len(sample)} rows (sampled)."
                                    )
                        except Exception:
                            pass

                elif view == "Chart":
                    graph_json = None
                    idx_map = st.session_state.get("pipeline_studio_artifacts")
                    if isinstance(idx_map, dict):
                        entry_art = idx_map.get(selected_node_id)
                        entry_art = entry_art if isinstance(entry_art, dict) else {}
                        pg = entry_art.get("plotly_graph")
                        pg = pg if isinstance(pg, dict) else {}
                        graph_json = pg.get("json")
                        viz_err = entry_art.get("viz_error")
                        viz_err = viz_err if isinstance(viz_err, dict) else {}
                        viz_err_msg = viz_err.get("message")
                        viz_err_path = viz_err.get("log_path")
                        viz_warn = entry_art.get("viz_warning")
                        viz_warn = viz_warn if isinstance(viz_warn, dict) else {}
                        viz_warn_msg = viz_warn.get("message")
                    else:
                        viz_err_msg = None
                        viz_err_path = None
                        viz_warn_msg = None
                    if not graph_json:
                        detail = _latest_detail_for_dataset_id(
                            selected_node_id, require_key="plotly_graph"
                        )
                        graph_json = (
                            detail.get("plotly_graph")
                            if isinstance(detail, dict)
                            else None
                        )
                        if isinstance(detail, dict) and not viz_err_msg:
                            viz_err_msg = detail.get("data_visualization_error")
                            viz_err_path = detail.get(
                                "data_visualization_error_log_path"
                            )
                        if isinstance(detail, dict) and not viz_warn_msg:
                            viz_warn_msg = detail.get("data_visualization_warning")
                    if not graph_json:
                        fp = entry.get("fingerprint")
                        fp = fp if isinstance(fp, str) and fp else None
                        if fp:
                            persisted = _get_persisted_pipeline_studio_artifacts(
                                fingerprint=fp
                            )
                            pg = persisted.get("plotly_graph")
                            pg = pg if isinstance(pg, dict) else {}
                            graph_json = pg.get("json")
                            viz_err = persisted.get("viz_error")
                            viz_err = viz_err if isinstance(viz_err, dict) else {}
                            if not viz_err_msg:
                                viz_err_msg = viz_err.get("message")
                                viz_err_path = viz_err.get("log_path")
                            viz_warn = persisted.get("viz_warning")
                            viz_warn = viz_warn if isinstance(viz_warn, dict) else {}
                            if not viz_warn_msg:
                                viz_warn_msg = viz_warn.get("message")
                    if isinstance(viz_err_msg, str) and viz_err_msg:
                        err_bits = [viz_err_msg]
                        if isinstance(viz_err_path, str) and viz_err_path:
                            err_bits.append(f"Log: {viz_err_path}")
                        st.error("Visualization error:\n" + "\n".join(err_bits))
                    if isinstance(viz_warn_msg, str) and viz_warn_msg:
                        st.warning(viz_warn_msg)
                    if not graph_json:
                        st.info(
                            "No chart found for this dataset yet. Try: `plot ...` while this dataset is active."
                        )
                    else:
                        try:
                            payload = (
                                json.dumps(graph_json)
                                if isinstance(graph_json, dict)
                                else graph_json
                            )
                            fig = _apply_streamlit_plot_style(pio.from_json(payload))
                            st.plotly_chart(
                                fig,
                                width="stretch",
                                key=f"pipeline_studio_chart_{selected_node_id}",
                            )
                        except Exception as e:
                            st.error(f"Error rendering chart: {e}")

                elif view == "EDA":
                    reports = None
                    idx_map = st.session_state.get("pipeline_studio_artifacts")
                    if isinstance(idx_map, dict):
                        entry_art = idx_map.get(selected_node_id)
                        entry_art = entry_art if isinstance(entry_art, dict) else {}
                        er = entry_art.get("eda_reports")
                        er = er if isinstance(er, dict) else {}
                        reports = er.get("reports")
                    if not reports:
                        detail = _latest_detail_for_dataset_id(
                            selected_node_id, require_key="eda_reports"
                        )
                        reports = (
                            detail.get("eda_reports")
                            if isinstance(detail, dict)
                            else None
                        )
                    if not reports:
                        fp = entry.get("fingerprint")
                        fp = fp if isinstance(fp, str) and fp else None
                        if fp:
                            persisted = _get_persisted_pipeline_studio_artifacts(
                                fingerprint=fp
                            )
                            er = persisted.get("eda_reports")
                            er = er if isinstance(er, dict) else {}
                            reports = er.get("reports")
                    reports = reports if isinstance(reports, dict) else {}
                    sweetviz_file = (
                        reports.get("sweetviz_report_file")
                        if isinstance(reports.get("sweetviz_report_file"), str)
                        else None
                    )
                    dtale_url = (
                        reports.get("dtale_url")
                        if isinstance(reports.get("dtale_url"), str)
                        else None
                    )
                    if sweetviz_file:
                        st.markdown("**Sweetviz report**")
                        st.write(sweetviz_file)
                        try:
                            with open(sweetviz_file, "r", encoding="utf-8") as f:
                                html = f.read()
                            components.html(html, height=800, scrolling=True)
                            st.download_button(
                                "Download Sweetviz HTML",
                                data=html.encode("utf-8"),
                                file_name=os.path.basename(sweetviz_file),
                                mime="text/html",
                                key=f"pipeline_studio_download_sweetviz_{selected_node_id}",
                            )
                        except Exception as e:
                            st.warning(f"Could not render Sweetviz report: {e}")
                    if dtale_url:
                        st.markdown("**D-Tale**")
                        st.markdown(f"[Open D-Tale]({dtale_url})")
                    if not sweetviz_file and not dtale_url:
                        st.info(
                            "No EDA reports found for this dataset yet. Try: `generate a Sweetviz report` while this dataset is active."
                        )

                elif view == "Code":
                    prov = (
                        entry.get("provenance")
                        if isinstance(entry.get("provenance"), dict)
                        else {}
                    )
                    transform = (
                        prov.get("transform")
                        if isinstance(prov.get("transform"), dict)
                        else {}
                    )
                    kind = str(transform.get("kind") or "")

                    st.markdown("**Provenance**")
                    st.json(
                        {
                            "source_type": prov.get("source_type"),
                            "source": prov.get("source"),
                            "transform_kind": kind or None,
                            "created_by": entry.get("created_by"),
                            "created_at": entry.get("created_at"),
                        }
                    )

                    code_text = None
                    code_lang = "python"
                    title = None
                    if kind == "python_function":
                        title = "Transform function (Python)"
                        code_text = transform.get("function_code")
                    elif kind == "sql_query":
                        title = "SQL query"
                        code_text = transform.get("sql_query_code")
                        code_lang = "sql"
                    elif kind == "python_merge":
                        title = "Merge code (Python)"
                        code_text = transform.get("merge_code")
                    elif kind == "mlflow_predict":
                        run_id = transform.get("run_id")
                        run_id = run_id.strip() if isinstance(run_id, str) else ""
                        title = "Prediction (MLflow) snippet"
                        code_text = (
                            "\n".join(
                                [
                                    "import pandas as pd",
                                    "import mlflow",
                                    "",
                                    f"model_uri = 'runs:/{run_id}/model'",
                                    "model = mlflow.pyfunc.load_model(model_uri)",
                                    "preds = model.predict(df)",
                                    "df_preds = preds if isinstance(preds, pd.DataFrame) else pd.DataFrame(preds)",
                                ]
                            ).strip()
                            + "\n"
                        )
                    elif kind == "h2o_predict":
                        model_id = transform.get("model_id")
                        model_id = model_id.strip() if isinstance(model_id, str) else ""
                        title = "Prediction (H2O) snippet"
                        code_text = (
                            "\n".join(
                                [
                                    "import h2o",
                                    "",
                                    "h2o.init()",
                                    f"model = h2o.get_model('{model_id}')",
                                    "frame = h2o.H2OFrame(df)",
                                    "preds = model.predict(frame)",
                                    "df_preds = preds.as_data_frame(use_pandas=True)",
                                ]
                            ).strip()
                            + "\n"
                        )

                    if isinstance(code_text, str) and code_text.strip():
                        st.markdown(f"**{title or 'Code'}**")
                        editor_key = f"pipeline_studio_code_editor_{selected_node_id}"
                        reset_pending = st.session_state.pop(
                            "pipeline_studio_code_reset_pending", None
                        )
                        if reset_pending == selected_node_id:
                            st.session_state.pop(editor_key, None)
                        fp = entry.get("fingerprint")
                        fp = fp if isinstance(fp, str) and fp else None
                        saved_draft = None
                        saved_meta = {}
                        if fp:
                            saved_meta = _get_pipeline_studio_code_draft(fingerprint=fp)
                            saved_draft = (
                                saved_meta.get("draft_code")
                                if isinstance(saved_meta, dict)
                                else None
                            )
                            saved_draft = (
                                saved_draft
                                if isinstance(saved_draft, str) and saved_draft.strip()
                                else None
                            )
                        if (
                            saved_draft
                            and editor_key not in st.session_state
                            and reset_pending != selected_node_id
                        ):
                            st.session_state[editor_key] = saved_draft
                        draft_saved_flag = st.session_state.pop(
                            "pipeline_studio_code_draft_saved", None
                        )
                        if draft_saved_flag == selected_node_id:
                            st.success("Draft saved to `pipeline_store/`.")
                        if saved_draft:
                            ts = None
                            try:
                                ts = float(
                                    (saved_meta or {}).get("updated_ts")
                                    or (saved_meta or {}).get("created_ts")
                                    or 0.0
                                )
                            except Exception:
                                ts = None
                            st.caption(
                                "Loaded saved draft from `pipeline_store/`."
                                + (
                                    f" (updated_ts={ts:.0f})"
                                    if isinstance(ts, float) and ts
                                    else ""
                                )
                            )
                        draft_code = st.text_area(
                            "Draft editor",
                            value=code_text,
                            key=editor_key,
                            height=320,
                            help="Edit the snippet and use Ask AI to refine it. You can run drafts for python_function/python_merge/sql_query.",
                        )
                        show_preview = st.checkbox(
                            "Show formatted preview",
                            value=False,
                            key=f"pipeline_studio_draft_preview_{selected_node_id}",
                        )
                        if show_preview:
                            st.code(
                                draft_code if isinstance(draft_code, str) else "",
                                language="sql" if code_lang == "sql" else "python",
                            )
                        try:
                            ext = "sql" if code_lang == "sql" else "py"
                            mime = (
                                "application/sql"
                                if code_lang == "sql"
                                else "text/x-python"
                            )
                            c_ask, c_save, c_reset, c_copy, c_download = st.columns(
                                [0.16, 0.16, 0.14, 0.20, 0.34]
                            )
                            with c_ask:
                                if st.button(
                                    "Ask AI",
                                    key=f"pipeline_studio_code_ask_ai_{selected_node_id}",
                                    help="Send this draft to chat for review and improvement.",
                                ):
                                    prompt_lines = [
                                        "Pipeline Studio request: review and improve this draft code for the selected pipeline step.",
                                        f"selected_node_id: {selected_node_id}",
                                        f"transform_kind: {kind}" if kind else "",
                                        "",
                                        f"```{code_lang}\n{draft_code}\n```",
                                        "",
                                        "Return the full improved code snippet and explain key changes briefly.",
                                    ]
                                    st.session_state["chat_prompt_pending"] = "\n".join(
                                        [
                                            x
                                            for x in prompt_lines
                                            if isinstance(x, str) and x.strip()
                                        ]
                                    ).strip()
                                    st.rerun()
                            with c_save:

                                def _save_code_draft(
                                    fingerprint: str | None,
                                    node_id: str,
                                    e_key: str,
                                    t_kind: str,
                                    lang: str,
                                ) -> None:
                                    if (
                                        not isinstance(fingerprint, str)
                                        or not fingerprint
                                    ):
                                        return
                                    code = st.session_state.get(e_key)
                                    code = code if isinstance(code, str) else ""
                                    _save_pipeline_studio_code_draft(
                                        fingerprint=fingerprint,
                                        dataset_id=node_id,
                                        transform_kind=t_kind,
                                        lang=lang,
                                        draft_code=code,
                                    )
                                    st.session_state[
                                        "pipeline_studio_code_draft_saved"
                                    ] = node_id

                                st.button(
                                    "Save",
                                    key=f"pipeline_studio_code_save_{selected_node_id}",
                                    help="Persist this draft to `pipeline_store/` (keyed by dataset fingerprint).",
                                    on_click=_save_code_draft,
                                    args=(
                                        fp,
                                        selected_node_id,
                                        editor_key,
                                        kind,
                                        code_lang,
                                    ),
                                )
                            with c_reset:

                                def _queue_code_reset(
                                    node_id: str, fingerprint: str | None
                                ) -> None:
                                    st.session_state[
                                        "pipeline_studio_code_reset_pending"
                                    ] = node_id
                                    if isinstance(fingerprint, str) and fingerprint:
                                        _delete_pipeline_studio_code_draft(
                                            fingerprint=fingerprint
                                        )

                                st.button(
                                    "Reset",
                                    key=f"pipeline_studio_code_reset_{selected_node_id}",
                                    help="Discard edits and restore the recorded code for this node.",
                                    on_click=_queue_code_reset,
                                    args=(selected_node_id, fp),
                                )
                            with c_copy:
                                _render_copy_to_clipboard(
                                    draft_code, label="Copy draft"
                                )
                            with c_download:
                                st.download_button(
                                    "Download draft",
                                    data=draft_code.encode("utf-8"),
                                    file_name=f"{selected_node_id}_{kind or 'step'}.{ext}",
                                    mime=mime,
                                    key=f"pipeline_studio_download_snippet_{selected_node_id}",
                                )
                        except Exception:
                            pass

                        if kind in {"python_function", "python_merge", "sql_query"}:
                            st.markdown("---")
                            st.markdown("**Run draft (local)**")
                            parents = _entry_parent_ids(entry)
                            if kind == "python_function":
                                st.caption(
                                    f"Input dataset: `{parents[0]}` → creates a new `{entry.get('stage')}` node"
                                    if parents
                                    else "Input dataset: (missing parent)"
                                )
                            elif kind == "python_merge":
                                st.caption(
                                    "Inputs: "
                                    + ", ".join([f"`{p}`" for p in parents])
                                    + f" → creates a new `{entry.get('stage')}` node"
                                    if parents
                                    else "Inputs: (missing parents)"
                                )
                            elif kind == "sql_query":
                                st.caption(
                                    f"SQL URL: `{_redact_sqlalchemy_url(st.session_state.get('sql_url', DEFAULT_SQL_URL))}`"
                                )

                            confirm_key = (
                                f"pipeline_studio_run_confirm_{selected_node_id}"
                            )
                            confirm_label = (
                                "I understand this executes code locally"
                                if kind in {"python_function", "python_merge"}
                                else "I understand this runs a read-only SQL query"
                            )
                            confirmed = st.checkbox(
                                confirm_label,
                                key=confirm_key,
                                help="Creates a new dataset node from the draft output (active).",
                            )

                            def _run_draft_click(nid: str, e_key: str, k: str) -> None:
                                if k == "python_function":
                                    _run_python_function_draft(
                                        node_id=nid, editor_key=e_key
                                    )
                                elif k == "python_merge":
                                    _run_python_merge_draft(
                                        node_id=nid, editor_key=e_key
                                    )
                                elif k == "sql_query":
                                    _run_sql_query_draft(node_id=nid, editor_key=e_key)

                            def _run_draft_and_downstream(
                                nid: str, e_key: str, k: str
                            ) -> None:
                                _run_draft_click(nid, e_key, k)
                                err = st.session_state.get("pipeline_studio_run_error")
                                if isinstance(err, str) and err.strip():
                                    return
                                last_repl = st.session_state.get(
                                    "pipeline_studio_last_replacement"
                                )
                                if not isinstance(last_repl, dict):
                                    return
                                old_id = last_repl.get("old_id")
                                new_id = last_repl.get("new_id")
                                run_ok = st.session_state.get(
                                    "pipeline_studio_run_success"
                                )
                                old_id = (
                                    old_id.strip()
                                    if isinstance(old_id, str) and old_id.strip()
                                    else None
                                )
                                new_id = (
                                    new_id.strip()
                                    if isinstance(new_id, str) and new_id.strip()
                                    else None
                                )
                                if (
                                    not old_id
                                    or not new_id
                                    or (isinstance(run_ok, str) and run_ok != new_id)
                                ):
                                    return
                                _run_downstream_transforms(old_id, new_id)

                            r1, r2 = st.columns(2)
                            with r1:
                                st.button(
                                    "Run draft only",
                                    key=f"pipeline_studio_run_draft_{selected_node_id}",
                                    disabled=not bool(confirmed),
                                    on_click=_run_draft_click,
                                    args=(selected_node_id, editor_key, kind),
                                    help="Runs the draft and registers the output as a new dataset (active).",
                                    width="stretch",
                                )
                            with r2:
                                st.button(
                                    "Run draft + run downstream",
                                    key=f"pipeline_studio_run_draft_downstream_{selected_node_id}",
                                    type="primary",
                                    disabled=not bool(confirmed),
                                    on_click=_run_draft_and_downstream,
                                    args=(selected_node_id, editor_key, kind),
                                    help="Runs the draft, then best-effort reruns downstream steps.",
                                    width="stretch",
                                )
                    else:
                        st.info("No runnable code recorded for this step.")

                    if isinstance(script, str) and script.strip():
                        with st.expander("Full pipeline repro script", expanded=False):
                            _render_copy_to_clipboard(script, label="Copy script")
                            st.code(script, language="python")

                elif view == "Model":
                    model_info = None
                    eval_art = None
                    eval_graph = None
                    idx_map = st.session_state.get("pipeline_studio_artifacts")
                    if isinstance(idx_map, dict):
                        entry_art = idx_map.get(selected_node_id)
                        entry_art = entry_art if isinstance(entry_art, dict) else {}
                        mi = entry_art.get("model_info")
                        mi = mi if isinstance(mi, dict) else {}
                        model_info = mi.get("info")
                        ea = entry_art.get("eval_artifacts")
                        ea = ea if isinstance(ea, dict) else {}
                        eval_art = ea.get("artifacts")
                        eg = entry_art.get("eval_plotly_graph")
                        eg = eg if isinstance(eg, dict) else {}
                        eval_graph = eg.get("json")
                    if model_info is None and eval_art is None and eval_graph is None:
                        detail = _latest_detail_for_dataset_id(
                            selected_node_id, require_key="model_info"
                        )
                        if isinstance(detail, dict):
                            model_info = detail.get("model_info")
                            eval_art = detail.get("eval_artifacts")
                            eval_graph = detail.get("eval_plotly_graph")
                    if model_info is None and eval_art is None and eval_graph is None:
                        fp = entry.get("fingerprint")
                        fp = fp if isinstance(fp, str) and fp else None
                        if fp:
                            persisted = _get_persisted_pipeline_studio_artifacts(
                                fingerprint=fp
                            )
                            mi = persisted.get("model_info")
                            mi = mi if isinstance(mi, dict) else {}
                            model_info = mi.get("info")
                            ea = persisted.get("eval_artifacts")
                            ea = ea if isinstance(ea, dict) else {}
                            eval_art = ea.get("artifacts")
                            eg = persisted.get("eval_plotly_graph")
                            eg = eg if isinstance(eg, dict) else {}
                            eval_graph = eg.get("json")

                    if model_info is not None:
                        st.markdown("**Model Info**")
                        try:
                            if isinstance(model_info, dict):
                                st.dataframe(
                                    pd.DataFrame(model_info),
                                    width="stretch",
                                )
                            elif isinstance(model_info, list):
                                st.dataframe(
                                    pd.DataFrame(model_info),
                                    width="stretch",
                                )
                            else:
                                st.json(model_info)
                        except Exception:
                            st.json(model_info)
                    if eval_art is not None:
                        st.markdown("**Evaluation**")
                        st.json(eval_art)
                    if eval_graph:
                        try:
                            payload = (
                                json.dumps(eval_graph)
                                if isinstance(eval_graph, dict)
                                else eval_graph
                            )
                            fig = _apply_streamlit_plot_style(pio.from_json(payload))
                            st.plotly_chart(
                                fig,
                                width="stretch",
                                key=f"pipeline_studio_eval_chart_{selected_node_id}",
                            )
                        except Exception as e:
                            st.error(f"Error rendering evaluation chart: {e}")
                    if model_info is None and eval_art is None and eval_graph is None:
                        st.info(
                            "No model artifacts found for this dataset yet. Try: `train a model` while this dataset is active."
                        )

                elif view == "Predictions":
                    prov = (
                        entry.get("provenance")
                        if isinstance(entry.get("provenance"), dict)
                        else {}
                    )
                    transform = (
                        prov.get("transform")
                        if isinstance(prov.get("transform"), dict)
                        else {}
                    )
                    kind = str(transform.get("kind") or "")
                    is_pred = kind in {"mlflow_predict", "h2o_predict"}
                    if not is_pred:
                        st.info(
                            "This pipeline step does not look like a predictions dataset. "
                            "Select a `*_predict` step to view predictions."
                        )
                    elif df_sel is None:
                        st.info("No tabular predictions data available for this step.")
                    else:
                        st.markdown("**Predictions preview**")
                        meta = {
                            "kind": kind,
                            "run_id": transform.get("run_id")
                            if kind == "mlflow_predict"
                            else None,
                            "model_uri": transform.get("model_uri")
                            if kind == "mlflow_predict"
                            else None,
                            "model_id": transform.get("model_id")
                            if kind == "h2o_predict"
                            else None,
                        }
                        st.json({k: v for k, v in meta.items() if v})
                        st.dataframe(df_sel.head(50), width="stretch")

                elif view == "MLflow":
                    mlflow_art = None
                    idx_map = st.session_state.get("pipeline_studio_artifacts")
                    if isinstance(idx_map, dict):
                        entry_art = idx_map.get(selected_node_id)
                        entry_art = entry_art if isinstance(entry_art, dict) else {}
                        ma = entry_art.get("mlflow_artifacts")
                        ma = ma if isinstance(ma, dict) else {}
                        mlflow_art = ma.get("artifacts")
                    if mlflow_art is None:
                        detail = _latest_detail_for_dataset_id(
                            selected_node_id, require_key="mlflow_artifacts"
                        )
                        if isinstance(detail, dict):
                            mlflow_art = detail.get("mlflow_artifacts")
                    if mlflow_art is None:
                        fp = entry.get("fingerprint")
                        fp = fp if isinstance(fp, str) and fp else None
                        if fp:
                            persisted = _get_persisted_pipeline_studio_artifacts(
                                fingerprint=fp
                            )
                            ma = persisted.get("mlflow_artifacts")
                            ma = ma if isinstance(ma, dict) else {}
                            mlflow_art = ma.get("artifacts")

                    if mlflow_art is None:
                        st.info(
                            "No MLflow artifacts found for this dataset yet. Try: `what runs are available?`"
                        )
                    else:
                        st.markdown("**MLflow Artifacts**")

                        def _render_mlflow_artifact(obj):
                            try:
                                if isinstance(obj, dict) and isinstance(
                                    obj.get("runs"), list
                                ):
                                    df = pd.DataFrame(obj["runs"])
                                    preferred_cols = [
                                        c
                                        for c in [
                                            "run_id",
                                            "run_name",
                                            "status",
                                            "start_time",
                                            "duration_seconds",
                                            "has_model",
                                            "model_uri",
                                            "params_preview",
                                            "metrics_preview",
                                        ]
                                        if c in df.columns
                                    ]
                                    st.dataframe(
                                        df[preferred_cols] if preferred_cols else df,
                                        width="stretch",
                                    )
                                    if any(
                                        c in df.columns
                                        for c in (
                                            "params",
                                            "metrics",
                                            "tags",
                                            "artifact_uri",
                                        )
                                    ):
                                        with st.expander(
                                            "Raw run details", expanded=False
                                        ):
                                            st.json(obj)
                                    return
                                if isinstance(obj, dict) and isinstance(
                                    obj.get("experiments"), list
                                ):
                                    df = pd.DataFrame(obj["experiments"])
                                    preferred_cols = [
                                        c
                                        for c in [
                                            "experiment_id",
                                            "name",
                                            "lifecycle_stage",
                                            "creation_time",
                                            "last_update_time",
                                            "artifact_location",
                                        ]
                                        if c in df.columns
                                    ]
                                    st.dataframe(
                                        df[preferred_cols] if preferred_cols else df,
                                        width="stretch",
                                    )
                                    return
                                if isinstance(obj, list):
                                    st.dataframe(
                                        pd.DataFrame(obj),
                                        width="stretch",
                                    )
                                    return
                            except Exception:
                                pass
                            st.json(obj)

                        if isinstance(mlflow_art, dict) and not any(
                            k in mlflow_art for k in ("runs", "experiments")
                        ):
                            is_tool_map = all(
                                isinstance(k, str) and k.startswith("mlflow_")
                                for k in mlflow_art.keys()
                            )
                            if is_tool_map:
                                for tool_name, tool_art in mlflow_art.items():
                                    st.markdown(f"`{tool_name}`")
                                    _render_mlflow_artifact(tool_art)
                            else:
                                _render_mlflow_artifact(mlflow_art)
                        else:
                            _render_mlflow_artifact(mlflow_art)

                elif view == "Visual Editor":
                    st.caption(
                        "Beta: drag nodes to arrange layout; right-click for node actions; click a node to inspect."
                    )
                    try:
                        from streamlit_flow import (
                            StreamlitFlowEdge,
                            StreamlitFlowNode,
                            StreamlitFlowState,
                            streamlit_flow,
                        )
                        from streamlit_flow.layouts import LayeredLayout, ManualLayout
                    except Exception:
                        st.info(
                            "Visual Editor dependency not installed. Add `streamlit-flow-component` and restart the app."
                        )
                        return

                    all_node_ids = [
                        did for did in node_ids if isinstance(did, str) and did
                    ]
                    if not all_node_ids:
                        st.info("No pipeline nodes available to render.")
                        return
                    show_hidden_pick = bool(
                        st.session_state.get("pipeline_studio_show_hidden", False)
                    )
                    show_deleted_pick = bool(
                        st.session_state.get("pipeline_studio_show_deleted", False)
                    )
                    ordered_dataset_ids = sorted(
                        studio_datasets.items(),
                        key=lambda kv: float(kv[1].get("created_ts") or 0.0)
                        if isinstance(kv[1], dict)
                        else 0.0,
                        reverse=True,
                    )
                    all_dataset_ids = [
                        did for did, _e in ordered_dataset_ids if isinstance(did, str)
                    ]
                    dataset_ids = [
                        did
                        for did in all_dataset_ids
                        if (show_hidden_pick or did not in ui_hidden_ids)
                        and (show_deleted_pick or did not in ui_deleted_ids)
                    ]
                    if not dataset_ids:
                        dataset_ids = list(all_dataset_ids)

                    pipeline_hash = (
                        pipe.get("pipeline_hash") if isinstance(pipe, dict) else None
                    )
                    pipeline_hash = (
                        pipeline_hash
                        if isinstance(pipeline_hash, str) and pipeline_hash.strip()
                        else None
                    )
                    prev_pipeline_hash = st.session_state.get(
                        "pipeline_studio_flow_pipeline_hash"
                    )
                    if pipeline_hash and prev_pipeline_hash != pipeline_hash:
                        st.session_state["pipeline_studio_flow_pipeline_hash"] = (
                            pipeline_hash
                        )
                        # Reset in-session layout cache when switching pipelines.
                        st.session_state.pop("pipeline_studio_flow_state", None)
                        st.session_state.pop("pipeline_studio_flow_signature", None)
                        st.session_state.pop("pipeline_studio_flow_positions", None)
                        st.session_state.pop("pipeline_studio_flow_hidden_ids", None)
                        st.session_state.pop("pipeline_studio_flow_layout_sig", None)

                        persisted_layout = _get_persisted_pipeline_studio_flow_layout(
                            pipeline_hash=pipeline_hash
                        )
                        pos = persisted_layout.get("positions")
                        if isinstance(pos, dict):
                            st.session_state["pipeline_studio_flow_positions"] = pos
                        hid = persisted_layout.get("hidden_ids")
                        hid = hid if isinstance(hid, list) else []
                        hid_clean = {
                            str(x) for x in hid if isinstance(x, str) and x.strip()
                        }
                        ui_hidden, ui_deleted = _pipeline_studio_get_registry_ui(
                            pipeline_hash=pipeline_hash
                        )
                        if hid_clean and not (ui_hidden or ui_deleted):
                            ui_hidden = set(hid_clean)
                            _pipeline_studio_set_registry_ui(
                                pipeline_hash=pipeline_hash,
                                hidden_ids=ui_hidden,
                                deleted_ids=set(),
                            )
                        elif hid_clean and (hid_clean - (ui_hidden | ui_deleted)):
                            ui_hidden = set(ui_hidden) | set(hid_clean)
                            _pipeline_studio_set_registry_ui(
                                pipeline_hash=pipeline_hash,
                                hidden_ids=ui_hidden,
                                deleted_ids=ui_deleted,
                            )
                        st.session_state["pipeline_studio_flow_hidden_ids"] = sorted(
                            (ui_hidden | ui_deleted)
                            if (ui_hidden or ui_deleted)
                            else hid_clean
                        )
                        st.session_state["pipeline_studio_flow_fit_view_pending"] = True

                    hidden_ids = st.session_state.get("pipeline_studio_flow_hidden_ids")
                    hidden_ids = hidden_ids if isinstance(hidden_ids, list) else []
                    hidden_set = {
                        str(x) for x in hidden_ids if isinstance(x, str) and x
                    }
                    all_set = set(all_node_ids)
                    hidden_set = {hid for hid in hidden_set if hid in all_set}

                    ui_hidden_ids, ui_deleted_ids = (
                        _pipeline_studio_get_registry_ui(pipeline_hash=pipeline_hash)
                        if pipeline_hash
                        else (set(), set())
                    )
                    ui_hidden_ids = {hid for hid in ui_hidden_ids if hid in all_set}
                    ui_deleted_ids = {hid for hid in ui_deleted_ids if hid in all_set}
                    if ui_hidden_ids or ui_deleted_ids:
                        hidden_set = set(ui_hidden_ids) | set(ui_deleted_ids)
                    st.session_state["pipeline_studio_flow_hidden_ids"] = sorted(
                        hidden_set
                    )
                    if not (ui_hidden_ids or ui_deleted_ids):
                        ui_hidden_ids = set(hidden_set)
                        ui_deleted_ids = set()

                    show_hidden_steps = bool(
                        st.session_state.get("pipeline_studio_show_hidden", False)
                    )
                    show_deleted_steps = bool(
                        st.session_state.get("pipeline_studio_show_deleted", False)
                    )
                    canvas_hidden_set: set[str] = set()
                    if not show_hidden_steps:
                        canvas_hidden_set |= set(ui_hidden_ids)
                    if not show_deleted_steps:
                        canvas_hidden_set |= set(ui_deleted_ids)

                    stale_ids_list = st.session_state.get("pipeline_studio_stale_ids")
                    stale_ids_list = (
                        stale_ids_list if isinstance(stale_ids_list, list) else []
                    )
                    stale_set = {
                        str(x)
                        for x in stale_ids_list
                        if isinstance(x, str) and x in all_set
                    }

                    stale_only_key = "pipeline_studio_flow_stale_only"
                    stale_only = bool(st.session_state.get(stale_only_key, False))
                    flow_node_ids = list(all_node_ids)
                    focus_set: set[str] = set(stale_set)
                    last_repl = st.session_state.get("pipeline_studio_last_replacement")
                    if isinstance(last_repl, dict):
                        old_focus = last_repl.get("old_id")
                        new_focus = last_repl.get("new_id")
                        if isinstance(old_focus, str) and old_focus in all_set:
                            focus_set.add(old_focus)
                        if isinstance(new_focus, str) and new_focus in all_set:
                            focus_set.add(new_focus)
                    stale_only_empty = False
                    if stale_only:
                        if focus_set:
                            flow_node_ids = [
                                did for did in all_node_ids if did in focus_set
                            ]
                        else:
                            stale_only_empty = True

                    def _node_has_artifact(did: str, key: str) -> bool:
                        try:
                            idx_map = st.session_state.get("pipeline_studio_artifacts")
                            if isinstance(idx_map, dict):
                                entry_art = idx_map.get(did)
                                entry_art = (
                                    entry_art if isinstance(entry_art, dict) else {}
                                )
                                if isinstance(entry_art.get(key), dict):
                                    return True
                            ds_entry = (
                                studio_datasets.get(did)
                                if isinstance(studio_datasets, dict)
                                else None
                            )
                            ds_entry = ds_entry if isinstance(ds_entry, dict) else {}
                            fp = ds_entry.get("fingerprint")
                            fp = fp if isinstance(fp, str) and fp else None
                            if fp:
                                persisted = _get_persisted_pipeline_studio_artifacts(
                                    fingerprint=fp
                                )
                                return isinstance(persisted.get(key), dict)
                        except Exception:
                            return False
                        return False

                    def _node_style(
                        stage: str,
                        *,
                        is_active: bool,
                        is_target: bool,
                        is_stale: bool,
                        is_hidden: bool,
                        is_deleted: bool,
                    ) -> dict:
                        stage = (stage or "").strip().lower()
                        bg = "rgba(255,255,255,0.04)"
                        border = "1px solid rgba(255,255,255,0.15)"
                        if stage in {"raw", "sql"}:
                            bg = "rgba(148,163,184,0.12)"
                        elif stage in {"wrangled", "cleaned"}:
                            bg = "rgba(56,189,248,0.10)"
                        elif stage in {"feature"}:
                            bg = "rgba(34,197,94,0.10)"
                        elif stage in {"model"}:
                            bg = "rgba(168,85,247,0.10)"
                        style = {
                            "background": bg,
                            "border": border,
                            "borderRadius": "12px",
                            "color": "rgba(255,255,255,0.92)",
                            "fontSize": "12px",
                            "padding": "10px 12px",
                            "minWidth": "160px",
                        }
                        if is_deleted:
                            style["opacity"] = "0.35"
                            style["border"] = "1px dashed rgba(255,255,255,0.25)"
                        elif is_hidden:
                            style["opacity"] = "0.55"
                            style["border"] = "1px dashed rgba(255,255,255,0.25)"

                        accent = None
                        if is_active:
                            accent = "rgba(34,197,94,0.85)"  # green
                        elif is_target:
                            accent = "rgba(236,72,153,0.85)"  # pink
                        elif is_stale:
                            accent = "rgba(245,158,11,0.85)"  # amber
                        if accent:
                            if is_hidden or is_deleted:
                                style["boxShadow"] = f"0 0 0 2px {accent}"
                            else:
                                style["border"] = f"2px solid {accent}"
                        return style

                    def _parent_ids_for(did: str) -> list[str]:
                        entry = (
                            studio_datasets.get(did)
                            if isinstance(studio_datasets, dict)
                            else None
                        )
                        entry = entry if isinstance(entry, dict) else {}
                        parents: list[str] = []
                        pids = entry.get("parent_ids")
                        if isinstance(pids, (list, tuple)):
                            parents.extend(
                                [str(p) for p in pids if isinstance(p, str) and p]
                            )
                        pid = entry.get("parent_id")
                        if isinstance(pid, str) and pid and pid not in parents:
                            parents.insert(0, pid)
                        return [p for p in parents if p]

                    base_set = set(flow_node_ids)
                    base_edges: list[StreamlitFlowEdge] = []
                    for did in flow_node_ids:
                        for pid in _parent_ids_for(did):
                            if pid in base_set:
                                base_edges.append(
                                    StreamlitFlowEdge(
                                        id=f"e:{pid}->{did}",
                                        source=pid,
                                        target=did,
                                        edge_type="smoothstep",
                                        animated=False,
                                        deletable=False,
                                    )
                                )

                    import hashlib
                    import time

                    prev_state = st.session_state.get("pipeline_studio_flow_state")
                    if not isinstance(prev_state, StreamlitFlowState):
                        prev_state = None

                    pos_store = st.session_state.get("pipeline_studio_flow_positions")
                    pos_store = pos_store if isinstance(pos_store, dict) else {}
                    positions_by_id: dict[str, tuple[float, float]] = {}
                    for nid, pos_val in pos_store.items():
                        if not isinstance(nid, str) or not nid:
                            continue
                        x = None
                        y = None
                        try:
                            if isinstance(pos_val, dict):
                                x = float(pos_val.get("x", 0.0))
                                y = float(pos_val.get("y", 0.0))
                            elif (
                                isinstance(pos_val, (list, tuple)) and len(pos_val) == 2
                            ):
                                x = float(pos_val[0])
                                y = float(pos_val[1])
                        except Exception:
                            x = None
                            y = None
                        if x is None or y is None:
                            continue
                        positions_by_id[nid] = (x, y)
                    if not positions_by_id and prev_state is not None:
                        try:
                            for n in prev_state.nodes:
                                positions_by_id[str(n.id)] = (
                                    float(n.position.get("x", 0.0)),
                                    float(n.position.get("y", 0.0)),
                                )
                        except Exception:
                            positions_by_id = {}

                    flow_ts_key = "pipeline_studio_flow_ts"
                    if flow_ts_key not in st.session_state:
                        # Python-controlled timestamp: only bump when we want to force the component
                        # to accept an updated graph definition (new nodes/labels/show-all/reset).
                        st.session_state[flow_ts_key] = 0
                    flow_ts = int(st.session_state.get(flow_ts_key) or 0)

                    force_layout = bool(
                        st.session_state.pop("pipeline_studio_flow_force_layout", False)
                    )
                    use_auto_layout = force_layout or not positions_by_id
                    flow_layer_spacing = 320.0
                    flow_node_spacing = 140.0
                    layout_obj = (
                        LayeredLayout(
                            direction="right",
                            node_node_spacing=flow_node_spacing,
                            node_layer_spacing=flow_layer_spacing,
                        )
                        if use_auto_layout
                        else ManualLayout()
                    )
                    should_fit_view = use_auto_layout or bool(
                        st.session_state.pop(
                            "pipeline_studio_flow_fit_view_pending", False
                        )
                    )

                    selected_default = None
                    if prev_state is not None and isinstance(
                        prev_state.selected_id, str
                    ):
                        selected_default = prev_state.selected_id
                    elif isinstance(selected_node_id, str):
                        selected_default = selected_node_id

                    active_id = (
                        pipe.get("active_dataset_id")
                        if isinstance(pipe, dict)
                        else None
                    )
                    active_id = active_id if isinstance(active_id, str) else None
                    target_id = (
                        pipe.get("target_dataset_id")
                        if isinstance(pipe, dict)
                        else None
                    )
                    target_id = target_id if isinstance(target_id, str) else None

                    base_nodes: list[StreamlitFlowNode] = []
                    sig_nodes = []
                    for idx, did in enumerate(flow_node_ids):
                        meta = (
                            meta_by_id.get(did) if isinstance(meta_by_id, dict) else {}
                        )
                        meta = meta if isinstance(meta, dict) else {}
                        label = meta.get("label") or did
                        stage = str(meta.get("stage") or "")
                        tk = str(meta.get("transform_kind") or "")
                        shape = meta.get("shape")
                        shape_str = ""
                        if isinstance(shape, (list, tuple)) and len(shape) == 2:
                            shape_str = f"{shape[0]}×{shape[1]}"
                        flags = []
                        if _node_has_artifact(did, "plotly_graph"):
                            flags.append("chart")
                        if _node_has_artifact(did, "eda_reports"):
                            flags.append("eda")
                        if _node_has_artifact(did, "model_info") or _node_has_artifact(
                            did, "eval_artifacts"
                        ):
                            flags.append("model")
                        if _node_has_artifact(did, "mlflow_artifacts"):
                            flags.append("mlflow")
                        flags_str = f"Artifacts: {', '.join(flags)}" if flags else ""

                        status_tags: list[str] = []
                        is_active = bool(active_id and did == active_id)
                        is_target = bool(target_id and did == target_id)
                        is_stale = did in stale_set
                        is_deleted = did in ui_deleted_ids
                        is_hidden = did in ui_hidden_ids and not is_deleted
                        if is_active:
                            status_tags.append("ACTIVE")
                        if is_target:
                            status_tags.append("TARGET")
                        if is_stale:
                            status_tags.append("STALE")
                        if is_deleted:
                            status_tags.append("DELETED")
                        elif is_hidden:
                            status_tags.append("HIDDEN")
                        status_str = " • ".join(status_tags).strip()

                        bits = [x for x in [stage, tk, shape_str, flags_str] if x]
                        content_lines = [str(label)]
                        if status_str:
                            content_lines.append(status_str)
                        content_lines.extend(bits)
                        content = "\n".join([x for x in content_lines if x]).strip()

                        parents = _parent_ids_for(did)
                        node_type = "input" if not parents else "default"
                        if target_id and did == target_id:
                            node_type = "output"

                        if use_auto_layout:
                            pos = (0.0, 0.0)
                        else:
                            pos = positions_by_id.get(did)
                            if pos is None:
                                pos = None
                                for pid in parents:
                                    ppos = positions_by_id.get(pid)
                                    if ppos is None:
                                        continue
                                    pos = (ppos[0] + flow_layer_spacing, ppos[1])
                                    break
                            if pos is None:
                                pos = (0.0, float(idx * flow_node_spacing))
                        sig_nodes.append(
                            {
                                "id": did,
                                "label": str(label),
                                "stage": stage,
                                "transform_kind": tk,
                                "shape": shape_str,
                                "flags": sorted([str(x) for x in flags]),
                                "status": status_tags,
                                "node_type": node_type,
                                "hidden": did in canvas_hidden_set,
                            }
                        )
                        base_nodes.append(
                            StreamlitFlowNode(
                                id=did,
                                pos=pos,
                                data={"content": content},
                                node_type=node_type,
                                source_position="right",
                                target_position="left",
                                hidden=did in canvas_hidden_set,
                                selectable=True,
                                draggable=True,
                                deletable=True,
                                style=_node_style(
                                    stage,
                                    is_active=is_active,
                                    is_target=is_target,
                                    is_stale=is_stale,
                                    is_hidden=is_hidden,
                                    is_deleted=is_deleted,
                                ),
                            )
                        )

                    sig_obj = {
                        "nodes": sorted(
                            sig_nodes, key=lambda x: str(x.get("id") or "")
                        ),
                        "edges": sorted(
                            [(e.source, e.target) for e in base_edges],
                            key=lambda x: (str(x[0]), str(x[1])),
                        ),
                        "hidden": sorted(canvas_hidden_set),
                        "stale": sorted(stale_set),
                        "target_id": target_id,
                        "active_id": active_id,
                        "stale_only": bool(stale_only),
                        "show_hidden": bool(show_hidden_steps),
                        "show_deleted": bool(show_deleted_steps),
                    }
                    sig = hashlib.sha1(
                        json.dumps(sig_obj, sort_keys=True, default=str).encode("utf-8")
                    ).hexdigest()
                    if st.session_state.get("pipeline_studio_flow_signature") != sig:
                        st.session_state["pipeline_studio_flow_signature"] = sig
                        flow_ts = int(time.time() * 1000)
                        st.session_state[flow_ts_key] = flow_ts

                    flow_state = StreamlitFlowState(
                        nodes=base_nodes,
                        edges=base_edges,
                        selected_id=selected_default,
                        timestamp=flow_ts,
                    )

                    c1, c2, c3, c4, c5 = st.columns([0.16, 0.16, 0.16, 0.16, 0.36])
                    with c1:

                        def _flow_reset_layout(p_hash: str | None) -> None:
                            st.session_state.pop("pipeline_studio_flow_state", None)
                            st.session_state.pop("pipeline_studio_flow_positions", None)
                            st.session_state.pop("pipeline_studio_flow_signature", None)
                            if isinstance(p_hash, str) and p_hash:
                                _delete_persisted_pipeline_studio_flow_layout(
                                    pipeline_hash=p_hash
                                )
                            st.session_state["pipeline_studio_flow_force_layout"] = True
                            st.session_state[
                                "pipeline_studio_flow_fit_view_pending"
                            ] = True
                            st.session_state[flow_ts_key] = int(time.time() * 1000)

                        st.button(
                            "Reset layout",
                            key="pipeline_studio_flow_reset_layout",
                            help="Rebuilds the canvas layout (keeps pipeline data intact).",
                            on_click=_flow_reset_layout,
                            args=(pipeline_hash,),
                        )
                    with c2:

                        def _flow_show_all_nodes(p_hash: str | None) -> None:
                            st.session_state["pipeline_studio_flow_hidden_ids"] = []
                            if isinstance(p_hash, str) and p_hash:
                                _pipeline_studio_set_registry_ui(
                                    pipeline_hash=p_hash,
                                    hidden_ids=set(),
                                    deleted_ids=set(),
                                )
                            st.session_state[
                                "pipeline_studio_flow_fit_view_pending"
                            ] = True
                            st.session_state[flow_ts_key] = int(time.time() * 1000)

                        st.button(
                            "Show all nodes",
                            key="pipeline_studio_flow_show_all",
                            help="Restores any hidden/deleted nodes in the canvas.",
                            on_click=_flow_show_all_nodes,
                            args=(pipeline_hash,),
                        )
                    with c3:

                        def _open_manual_node_editor() -> None:
                            st.session_state["pipeline_studio_manual_node_open"] = True
                            st.session_state["pipeline_studio_manual_seed_defaults"] = (
                                True
                            )

                        st.button(
                            "New node",
                            key="pipeline_studio_flow_new_node_open",
                            help="Create a manual transform node (Python/SQL/Merge).",
                            width="stretch",
                            on_click=_open_manual_node_editor,
                        )
                    with c4:
                        st.checkbox(
                            "Stale only",
                            key=stale_only_key,
                            help="Filter the canvas to stale nodes (plus the last rerun old/new nodes when available).",
                        )
                        if stale_only and stale_only_empty:
                            st.caption("No stale nodes to focus; showing all.")
                    with c5:
                        st.caption(
                            f"Nodes: {len(flow_node_ids)}/{len(all_node_ids)} | "
                            f"Edges: {len(base_edges)} | "
                            f"Hidden: {len(ui_hidden_ids)} | "
                            f"Deleted: {len(ui_deleted_ids)} | "
                            f"Stale: {len(stale_set)}"
                        )

                    manual_open = bool(
                        st.session_state.get("pipeline_studio_manual_node_open", False)
                    )
                    if manual_open:
                        with st.expander("New node (manual transform)", expanded=False):
                            kind_labels = {
                                "python_function": "Python transform",
                                "sql_query": "SQL query",
                                "python_merge": "Merge (Python)",
                            }
                            manual_kind = st.selectbox(
                                "Node type",
                                options=list(kind_labels.keys()),
                                format_func=lambda k: kind_labels.get(k, str(k)),
                                key="pipeline_studio_manual_kind",
                                help="Choose the transform type for this manual node.",
                            )
                            prev_kind = st.session_state.get(
                                "pipeline_studio_manual_kind_prev"
                            )
                            if prev_kind != manual_kind:
                                st.session_state["pipeline_studio_manual_kind_prev"] = (
                                    manual_kind
                                )
                                st.session_state[
                                    "pipeline_studio_manual_seed_defaults"
                                ] = True

                            default_parent = (
                                selected_default
                                if isinstance(selected_default, str)
                                and selected_default in dataset_ids
                                else (
                                    active_id
                                    if isinstance(active_id, str)
                                    and active_id in dataset_ids
                                    else (dataset_ids[0] if dataset_ids else None)
                                )
                            )
                            parent_index = (
                                int(dataset_ids.index(default_parent))
                                if default_parent in dataset_ids
                                else 0
                            )

                            def _parent_fmt(did: str) -> str:
                                entry = (
                                    studio_datasets.get(did)
                                    if isinstance(studio_datasets, dict)
                                    else None
                                )
                                entry = entry if isinstance(entry, dict) else {}
                                m = (
                                    meta_by_id.get(did)
                                    if isinstance(meta_by_id, dict)
                                    else {}
                                )
                                m = m if isinstance(m, dict) else {}
                                lbl = entry.get("label") or m.get("label") or did
                                stg = str(entry.get("stage") or m.get("stage") or "")
                                shp = entry.get("shape") or m.get("shape")
                                shp_str = ""
                                if isinstance(shp, (list, tuple)) and len(shp) == 2:
                                    shp_str = f"{shp[0]}×{shp[1]}"
                                parts = [p for p in [stg, shp_str] if p]
                                meta = f" ({', '.join(parts)})" if parts else ""
                                return f"{lbl}{meta} — {did}"

                            parent_id_new = ""
                            parent_ids_new: list[str] = []
                            if manual_kind == "python_merge":
                                pending_parent_ids = st.session_state.pop(
                                    "pipeline_studio_manual_parent_ids_pending", None
                                )
                                if isinstance(pending_parent_ids, list):
                                    pending_clean = [
                                        did
                                        for did in pending_parent_ids
                                        if isinstance(did, str) and did in dataset_ids
                                    ]
                                    if pending_clean:
                                        st.session_state[
                                            "pipeline_studio_manual_parent_ids"
                                        ] = pending_clean

                                def _queue_manual_merge_parents(
                                    ids: list[str], *, notice: str | None = None
                                ) -> None:
                                    cleaned: list[str] = []
                                    seen: set[str] = set()
                                    for did in ids:
                                        if (
                                            not isinstance(did, str)
                                            or did not in dataset_ids
                                        ):
                                            continue
                                        if did in seen:
                                            continue
                                        seen.add(did)
                                        cleaned.append(did)
                                    if len(cleaned) < 2:
                                        st.session_state[
                                            "pipeline_studio_manual_parent_notice"
                                        ] = "Select at least two datasets."
                                        return
                                    st.session_state[
                                        "pipeline_studio_manual_parent_ids_pending"
                                    ] = cleaned
                                    st.session_state[
                                        "pipeline_studio_manual_parent_notice"
                                    ] = notice or "Updated parent datasets."
                                    _keep_pipeline_studio_open()

                                def _parse_parent_hint(text: str) -> list[str]:
                                    text = text if isinstance(text, str) else ""
                                    tokens = [
                                        t.strip()
                                        for t in re.split(r"[,\n]+", text)
                                        if t.strip()
                                    ]
                                    if len(tokens) <= 1:
                                        tokens = [
                                            t.strip()
                                            for t in re.split(r"\s+", text)
                                            if t.strip()
                                        ]
                                    if not tokens:
                                        return []
                                    resolved: list[str] = []
                                    for tok in tokens:
                                        if tok in dataset_ids:
                                            resolved.append(tok)
                                            continue
                                        tok_lower = tok.lower()
                                        match_id = None
                                        if tok_lower:
                                            for did in dataset_ids:
                                                did_lower = did.lower()
                                                if (
                                                    tok_lower == did_lower
                                                    or tok_lower in did_lower
                                                ):
                                                    match_id = did
                                                    break
                                        if match_id:
                                            resolved.append(match_id)
                                            continue
                                        for did in dataset_ids:
                                            entry = studio_datasets.get(did)
                                            entry = (
                                                entry if isinstance(entry, dict) else {}
                                            )
                                            label = entry.get("label") or ""
                                            prov = entry.get("provenance")
                                            prov = (
                                                prov if isinstance(prov, dict) else {}
                                            )
                                            candidates = [
                                                str(label),
                                                str(prov.get("original_name") or ""),
                                                str(prov.get("source") or ""),
                                            ]
                                            if any(
                                                tok_lower in c.lower()
                                                for c in candidates
                                                if c
                                            ):
                                                match_id = did
                                                break
                                        if match_id:
                                            resolved.append(match_id)
                                    return resolved

                                hint_cols = st.columns([0.62, 0.19, 0.19], gap="small")
                                with hint_cols[0]:
                                    hint_text = st.text_input(
                                        "Quick parent select (ids or labels)",
                                        key="pipeline_studio_manual_parent_hint",
                                        help="Paste dataset ids or labels (comma/space separated). Example: raw_abcd, raw_efgh.",
                                    )
                                with hint_cols[1]:
                                    if st.button(
                                        "Use IDs",
                                        key="pipeline_studio_manual_parent_apply",
                                        width="stretch",
                                    ):
                                        resolved = _parse_parent_hint(hint_text)
                                        notice = (
                                            f"Matched {len(resolved)} dataset(s)."
                                            if resolved
                                            else "No matches found."
                                        )
                                        _queue_manual_merge_parents(
                                            resolved,
                                            notice=notice,
                                        )
                                with hint_cols[2]:
                                    if st.button(
                                        "Auto pick",
                                        key="pipeline_studio_manual_parent_auto",
                                        width="stretch",
                                    ):
                                        auto_ids: list[str] = []
                                        if (
                                            isinstance(active_id, str)
                                            and active_id in dataset_ids
                                        ):
                                            auto_ids.append(active_id)
                                        for did in dataset_ids:
                                            if did not in auto_ids:
                                                auto_ids.append(did)
                                            if len(auto_ids) >= 2:
                                                break
                                        _queue_manual_merge_parents(
                                            auto_ids,
                                            notice="Auto-picked parent datasets.",
                                        )

                                default_merge = (
                                    [default_parent] if default_parent else []
                                )
                                for pid in dataset_ids:
                                    if pid != default_parent:
                                        default_merge.append(pid)
                                        break
                                parent_ids_new = st.multiselect(
                                    "Parent datasets (2+ required)",
                                    options=dataset_ids,
                                    default=default_merge[:2],
                                    format_func=_parent_fmt,
                                    key="pipeline_studio_manual_parent_ids",
                                    help="Ordered parent datasets available as df_0, df_1, ... in the merge code.",
                                )
                                st.caption(
                                    "Parent options include all loaded datasets. Switch Pipeline target → All datasets to visualize them."
                                )
                                hint_notice = st.session_state.pop(
                                    "pipeline_studio_manual_parent_notice", None
                                )
                                if isinstance(hint_notice, str) and hint_notice.strip():
                                    st.success(hint_notice)
                                parent_id_new = (
                                    parent_ids_new[0] if parent_ids_new else ""
                                )
                            else:
                                parent_id_new = st.selectbox(
                                    "Parent dataset",
                                    options=dataset_ids,
                                    index=parent_index,
                                    format_func=_parent_fmt,
                                    key="pipeline_studio_manual_parent_id",
                                    help="The input dataset passed into your transform function.",
                                )
                            insert_child_id = ""
                            insert_between = False
                            if manual_kind in {"python_function", "sql_query"}:
                                child_idx = _build_children_index(studio_datasets)
                                child_candidates = sorted(
                                    child_idx.get(parent_id_new, set())
                                )
                                insert_between = st.checkbox(
                                    "Insert between parent and child",
                                    key="pipeline_studio_manual_insert_between",
                                    help="Re-links a downstream node to this new node and marks it stale.",
                                )
                                if insert_between:
                                    if child_candidates:
                                        insert_child_id = st.selectbox(
                                            "Downstream node",
                                            options=child_candidates,
                                            format_func=_parent_fmt,
                                            key="pipeline_studio_manual_insert_child_id",
                                        )
                                        st.caption(
                                            "Downstream node will be relinked to the new node (marked stale)."
                                        )
                                    else:
                                        st.info(
                                            "No downstream nodes available for the selected parent."
                                        )
                            seed_defaults = bool(
                                st.session_state.pop(
                                    "pipeline_studio_manual_seed_defaults", False
                                )
                            )
                            parent_meta = (
                                meta_by_id.get(parent_id_new)
                                if isinstance(meta_by_id, dict)
                                and isinstance(parent_id_new, str)
                                else {}
                            )
                            parent_meta = (
                                parent_meta if isinstance(parent_meta, dict) else {}
                            )
                            parent_stage_guess = str(
                                parent_meta.get("stage") or ""
                            ).strip()
                            if manual_kind == "sql_query":
                                stage_default = "sql"
                            else:
                                stage_default = (
                                    _normalize_pipeline_stage(parent_stage_guess)
                                    if parent_stage_guess
                                    else "custom"
                                )
                            label_defaults = {
                                "python_function": "manual_transform",
                                "sql_query": "manual_sql",
                                "python_merge": "manual_merge",
                            }
                            label_default = label_defaults.get(
                                manual_kind, "manual_transform"
                            )
                            if seed_defaults:
                                try:
                                    st.session_state["pipeline_studio_manual_stage"] = (
                                        stage_default
                                    )
                                    st.session_state["pipeline_studio_manual_label"] = (
                                        label_default
                                    )
                                    st.session_state[
                                        "pipeline_studio_manual_confirm_run"
                                    ] = False
                                except Exception:
                                    pass
                            c_stage, c_label = st.columns([0.25, 0.75])
                            with c_stage:
                                if (
                                    "pipeline_studio_manual_stage"
                                    not in st.session_state
                                    or not str(
                                        st.session_state.get(
                                            "pipeline_studio_manual_stage"
                                        )
                                        or ""
                                    ).strip()
                                ):
                                    st.session_state["pipeline_studio_manual_stage"] = (
                                        stage_default
                                    )
                                stage_new = st.text_input(
                                    "Stage",
                                    key="pipeline_studio_manual_stage",
                                    help="Used as the dataset id prefix (e.g. `cleaned_...`, `features_...`).",
                                )
                                stage_norm = _normalize_pipeline_stage(stage_new or "")
                                if (
                                    stage_new
                                    and stage_norm != (stage_new or "").strip().lower()
                                ):
                                    st.caption(f"Normalized stage: `{stage_norm}`")
                            with c_label:
                                if (
                                    "pipeline_studio_manual_label"
                                    not in st.session_state
                                    or not str(
                                        st.session_state.get(
                                            "pipeline_studio_manual_label"
                                        )
                                        or ""
                                    ).strip()
                                ):
                                    st.session_state["pipeline_studio_manual_label"] = (
                                        label_default
                                    )
                                label_new = st.text_input(
                                    "Label",
                                    key="pipeline_studio_manual_label",
                                    help="Human-friendly name shown in the pipeline.",
                                )

                            code_templates = {
                                "python_function": (
                                    "import pandas as pd\n\n"
                                    "def transform(df: pd.DataFrame) -> pd.DataFrame:\n"
                                    "    df = df.copy()\n"
                                    "    # TODO: modify df\n"
                                    "    return df\n"
                                ),
                                "sql_query": ("SELECT *\nFROM my_table\nLIMIT 100\n"),
                                "python_merge": (
                                    "import pandas as pd\n\n"
                                    "# df_0, df_1, ... are available\n"
                                    'df = df_0.merge(df_1, on="id", how="left")\n'
                                ),
                            }
                            code_key_map = {
                                "python_function": "pipeline_studio_manual_python_code",
                                "sql_query": "pipeline_studio_manual_sql_code",
                                "python_merge": "pipeline_studio_manual_merge_code",
                            }
                            template_key = code_key_map.get(
                                manual_kind, "pipeline_studio_manual_python_code"
                            )
                            if template_key not in st.session_state:
                                st.session_state[template_key] = code_templates.get(
                                    manual_kind, code_templates["python_function"]
                                )

                            upload_label = (
                                "Load a SQL script (optional)"
                                if manual_kind == "sql_query"
                                else "Load a Python script (optional)"
                            )
                            upload_types = (
                                ["sql", "txt"]
                                if manual_kind == "sql_query"
                                else ["py", "txt"]
                            )
                            upload_key = f"pipeline_studio_manual_upload_{manual_kind}"
                            load_key = f"pipeline_studio_manual_load_{manual_kind}"
                            up = st.file_uploader(
                                upload_label,
                                type=upload_types,
                                key=upload_key,
                                help="Loads the file content into the editor below.",
                            )
                            if up is not None:
                                if st.button(
                                    "Load into editor",
                                    key=load_key,
                                    width="stretch",
                                ):
                                    try:
                                        st.session_state[template_key] = (
                                            up.getvalue().decode(
                                                "utf-8", errors="replace"
                                            )
                                        )
                                    except Exception:
                                        pass

                            if manual_kind == "sql_query":
                                st.caption(
                                    f"SQL URL: `{_redact_sqlalchemy_url(st.session_state.get('sql_url', DEFAULT_SQL_URL))}`"
                                )
                            code_label = (
                                "SQL query"
                                if manual_kind == "sql_query"
                                else "Merge code (Python)"
                                if manual_kind == "python_merge"
                                else "Python transform code"
                            )
                            code_help = (
                                "Provide a read-only SQL query to execute."
                                if manual_kind == "sql_query"
                                else "Use df_0, df_1, ... and assign the merged DataFrame to `df`."
                                if manual_kind == "python_merge"
                                else "Define a function (e.g. `def transform(df): ...`) that returns a pandas DataFrame."
                            )
                            code_new = st.text_area(
                                code_label,
                                key=template_key,
                                height=260,
                                help=code_help,
                            )
                            show_preview = st.checkbox(
                                "Show formatted preview",
                                value=False,
                                key="pipeline_studio_manual_code_preview",
                            )
                            if show_preview:
                                st.code(
                                    code_new if isinstance(code_new, str) else "",
                                    language="sql"
                                    if manual_kind == "sql_query"
                                    else "python",
                                )
                            manual_errors: list[str] = []
                            if (
                                manual_kind == "python_merge"
                                and len(parent_ids_new) < 2
                            ):
                                manual_errors.append(
                                    "Select at least two parent datasets."
                                )
                            if insert_between and not insert_child_id:
                                manual_errors.append(
                                    "Select a downstream node to insert between."
                                )
                            if manual_kind in {
                                "python_function",
                                "python_merge",
                            } and not (isinstance(code_new, str) and code_new.strip()):
                                manual_errors.append("Python code is empty.")
                            if manual_kind == "sql_query":
                                _sql_text, sql_err = (
                                    _pipeline_studio_validate_readonly_sql(code_new)
                                )
                                if sql_err:
                                    manual_errors.append(sql_err)
                            last_created = st.session_state.get(
                                "pipeline_studio_manual_last_created_id"
                            )
                            if isinstance(last_created, str) and last_created.strip():
                                st.caption(
                                    f"Last manual node created: `{last_created}`"
                                )
                            confirm_label = (
                                "I understand this runs a read-only SQL query"
                                if manual_kind == "sql_query"
                                else "I understand this executes code locally"
                            )
                            confirm_run = st.checkbox(
                                confirm_label,
                                key="pipeline_studio_manual_confirm_run",
                            )
                            for err in manual_errors:
                                st.error(err)
                            c_run, c_close = st.columns([0.7, 0.3])
                            with c_run:

                                def _manual_create_click() -> None:
                                    if manual_kind == "sql_query":
                                        _pipeline_studio_create_manual_sql_node(
                                            parent_id=st.session_state.get(
                                                "pipeline_studio_manual_parent_id", ""
                                            ),
                                            stage=st.session_state.get(
                                                "pipeline_studio_manual_stage", ""
                                            ),
                                            label=st.session_state.get(
                                                "pipeline_studio_manual_label", ""
                                            ),
                                            sql_text=st.session_state.get(
                                                template_key, ""
                                            ),
                                            insert_child_id=insert_child_id or None,
                                        )
                                    elif manual_kind == "python_merge":
                                        _pipeline_studio_create_manual_merge_node(
                                            parent_ids=st.session_state.get(
                                                "pipeline_studio_manual_parent_ids", []
                                            ),
                                            stage=st.session_state.get(
                                                "pipeline_studio_manual_stage", ""
                                            ),
                                            label=st.session_state.get(
                                                "pipeline_studio_manual_label", ""
                                            ),
                                            code=st.session_state.get(template_key, ""),
                                            insert_child_id=insert_child_id or None,
                                        )
                                    else:
                                        _pipeline_studio_create_manual_python_node(
                                            parent_id=st.session_state.get(
                                                "pipeline_studio_manual_parent_id", ""
                                            ),
                                            stage=st.session_state.get(
                                                "pipeline_studio_manual_stage", ""
                                            ),
                                            label=st.session_state.get(
                                                "pipeline_studio_manual_label", ""
                                            ),
                                            code=st.session_state.get(template_key, ""),
                                            insert_child_id=insert_child_id or None,
                                        )

                                st.button(
                                    "Create node",
                                    key="pipeline_studio_manual_create_node",
                                    type="primary",
                                    disabled=not bool(confirm_run)
                                    or bool(manual_errors),
                                    width="stretch",
                                    on_click=_manual_create_click,
                                )
                            with c_close:
                                if st.button(
                                    "Close",
                                    key="pipeline_studio_manual_close",
                                    width="stretch",
                                ):
                                    st.session_state[
                                        "pipeline_studio_manual_node_open"
                                    ] = False

                    new_state = streamlit_flow(
                        key="pipeline_studio_flow",
                        state=flow_state,
                        height=650,
                        fit_view=should_fit_view,
                        show_controls=True,
                        show_minimap=True,
                        layout=layout_obj,
                        get_node_on_click=True,
                        enable_pane_menu=True,
                        enable_node_menu=True,
                        enable_edge_menu=False,
                        hide_watermark=True,
                    )
                    pos_store = st.session_state.get("pipeline_studio_flow_positions")
                    pos_store = pos_store if isinstance(pos_store, dict) else {}
                    try:
                        for n in new_state.nodes:
                            pos_store[str(n.id)] = {
                                "x": float(n.position.get("x", 0.0)),
                                "y": float(n.position.get("y", 0.0)),
                            }
                    except Exception:
                        pass
                    st.session_state["pipeline_studio_flow_positions"] = pos_store
                    new_ids = {n.id for n in new_state.nodes}
                    removed_by_user = base_set - new_ids
                    if removed_by_user:
                        ui_hidden, ui_deleted = _pipeline_studio_get_registry_ui(
                            pipeline_hash=pipeline_hash
                        )
                        ui_deleted = set(ui_deleted) | {str(x) for x in removed_by_user}
                        _pipeline_studio_set_registry_ui(
                            pipeline_hash=pipeline_hash,
                            hidden_ids=ui_hidden,
                            deleted_ids=ui_deleted,
                        )
                        hidden_set = (set(ui_hidden) | set(ui_deleted)) | set(
                            hidden_set
                        )
                        st.session_state["pipeline_studio_flow_hidden_ids"] = sorted(
                            hidden_set
                        )
                    st.session_state["pipeline_studio_flow_state"] = new_state
                    if pipeline_hash:
                        try:
                            hid_save = st.session_state.get(
                                "pipeline_studio_flow_hidden_ids"
                            )
                            hid_save = (
                                hid_save
                                if isinstance(hid_save, list)
                                else sorted(hidden_set)
                            )
                            _update_persisted_pipeline_studio_flow_layout(
                                pipeline_hash=pipeline_hash,
                                positions=pos_store,
                                hidden_ids=[
                                    str(x) for x in hid_save if isinstance(x, str) and x
                                ],
                            )
                        except Exception:
                            pass

                    st.markdown("---")
                    sel = (
                        new_state.selected_id
                        if isinstance(new_state.selected_id, str)
                        else None
                    )
                    if not sel:
                        st.info("Click a node to inspect it.")
                    else:
                        st.markdown("**Node Inspector**")
                        st.code(sel, language="text")

                        meta = (
                            meta_by_id.get(sel) if isinstance(meta_by_id, dict) else {}
                        )
                        meta = meta if isinstance(meta, dict) else {}
                        entry_obj = (
                            studio_datasets.get(sel)
                            if isinstance(studio_datasets, dict)
                            else None
                        )
                        entry_obj = entry_obj if isinstance(entry_obj, dict) else {}
                        prov = (
                            entry_obj.get("provenance")
                            if isinstance(entry_obj.get("provenance"), dict)
                            else {}
                        )
                        transform = (
                            prov.get("transform")
                            if isinstance(prov.get("transform"), dict)
                            else {}
                        )
                        kind = str(transform.get("kind") or "")

                        if kind:
                            st.caption(f"Transform: `{kind}`")

                        available_views = ["Table", "Code"]
                        if _node_has_artifact(sel, "plotly_graph"):
                            available_views.append("Chart")
                        if _node_has_artifact(sel, "eda_reports"):
                            available_views.append("EDA")
                        if _node_has_artifact(sel, "model_info") or _node_has_artifact(
                            sel, "eval_artifacts"
                        ):
                            available_views.append("Model")
                        if kind in {"mlflow_predict", "h2o_predict"}:
                            available_views.append("Predictions")
                        if _node_has_artifact(sel, "mlflow_artifacts"):
                            available_views.append("MLflow")

                        open_view_key = "pipeline_studio_flow_open_view"
                        current_open_view = st.session_state.get(open_view_key)
                        if (
                            not isinstance(current_open_view, str)
                            or current_open_view not in available_views
                        ):
                            st.session_state[open_view_key] = available_views[0]
                        ui_hidden, ui_deleted = (
                            _pipeline_studio_get_registry_ui(
                                pipeline_hash=pipeline_hash
                            )
                            if pipeline_hash
                            else (set(), set())
                        )
                        in_hidden = sel in ui_hidden
                        in_deleted = sel in ui_deleted

                        def _flow_open_in_workspace(
                            node_id: str, view_name: str
                        ) -> None:
                            st.session_state["pipeline_studio_node_id_pending"] = (
                                node_id
                            )
                            st.session_state["pipeline_studio_autofollow_pending"] = (
                                False
                            )
                            st.session_state["pipeline_studio_view_pending"] = view_name

                        def _flow_toggle_hidden(
                            p_hash: str, node_id: str, currently_hidden: bool
                        ) -> None:
                            ui_h, ui_d = _pipeline_studio_get_registry_ui(
                                pipeline_hash=p_hash
                            )
                            ui_h = set(ui_h)
                            ui_d = set(ui_d)
                            if currently_hidden:
                                ui_h.discard(node_id)
                            else:
                                ui_h.add(node_id)
                            _pipeline_studio_set_registry_ui(
                                pipeline_hash=p_hash,
                                hidden_ids=ui_h,
                                deleted_ids=ui_d,
                            )
                            st.session_state["pipeline_studio_flow_hidden_ids"] = (
                                sorted(ui_h | ui_d)
                            )
                            st.session_state[
                                "pipeline_studio_flow_fit_view_pending"
                            ] = True
                            st.session_state[flow_ts_key] = int(time.time() * 1000)
                            _sync_pipeline_targets_after_ui_change()

                        def _flow_toggle_deleted(
                            p_hash: str, node_id: str, currently_deleted: bool
                        ) -> None:
                            ui_h, ui_d = _pipeline_studio_get_registry_ui(
                                pipeline_hash=p_hash
                            )
                            ui_h = set(ui_h)
                            ui_d = set(ui_d)
                            if currently_deleted:
                                ui_d.discard(node_id)
                            else:
                                ui_d.add(node_id)
                            _pipeline_studio_set_registry_ui(
                                pipeline_hash=p_hash,
                                hidden_ids=ui_h,
                                deleted_ids=ui_d,
                            )
                            st.session_state["pipeline_studio_flow_hidden_ids"] = (
                                sorted(ui_h | ui_d)
                            )
                            st.session_state[
                                "pipeline_studio_flow_fit_view_pending"
                            ] = True
                            st.session_state[flow_ts_key] = int(time.time() * 1000)
                            _sync_pipeline_targets_after_ui_change()

                        def _flow_save_node_metadata(
                            node_id: str,
                            label_key: str,
                            stage_key: str,
                        ) -> None:
                            try:
                                node_id = (
                                    node_id.strip() if isinstance(node_id, str) else ""
                                )
                                if not node_id:
                                    return
                                team_state = st.session_state.get("team_state", {})
                                team_state = (
                                    team_state if isinstance(team_state, dict) else {}
                                )
                                ds = team_state.get("datasets")
                                ds = ds if isinstance(ds, dict) else {}
                                if node_id not in ds:
                                    return
                                entry = ds.get(node_id)
                                entry = entry if isinstance(entry, dict) else {}
                                new_label = st.session_state.get(label_key)
                                new_label = (
                                    new_label if isinstance(new_label, str) else ""
                                )
                                new_label = new_label.strip()
                                new_stage = st.session_state.get(stage_key)
                                new_stage = (
                                    new_stage if isinstance(new_stage, str) else ""
                                )
                                new_stage = new_stage.strip() or (
                                    entry.get("stage")
                                    if isinstance(entry.get("stage"), str)
                                    else "custom"
                                )
                                updated = dict(entry)
                                if new_label:
                                    updated["label"] = new_label
                                updated["stage"] = new_stage
                                ds = dict(ds)
                                ds[node_id] = updated
                                team_state = dict(team_state)
                                team_state["datasets"] = ds
                                st.session_state["team_state"] = team_state
                                try:
                                    pipelines_new = _pipeline_studio_build_pipelines_from_team_state(
                                        team_state
                                    )
                                    _update_pipeline_registry_store_for_pipelines(
                                        pipelines=pipelines_new, datasets=ds
                                    )
                                except Exception:
                                    pass
                                _persist_pipeline_studio_team_state(
                                    team_state=team_state
                                )
                                st.session_state["pipeline_studio_history_notice"] = (
                                    f"Updated metadata for `{node_id}`."
                                )
                                st.session_state[flow_ts_key] = int(time.time() * 1000)
                            except Exception:
                                pass

                        def _flow_delete_branch(p_hash: str, root_id: str) -> None:
                            try:
                                p_hash = (
                                    p_hash.strip() if isinstance(p_hash, str) else ""
                                )
                                root_id = (
                                    root_id.strip() if isinstance(root_id, str) else ""
                                )
                                if not p_hash or not root_id:
                                    return
                                child_idx = _build_children_index(studio_datasets)
                                branch_ids = {root_id} | _descendants(
                                    root_id, child_idx
                                )
                                ui_h, ui_d = _pipeline_studio_get_registry_ui(
                                    pipeline_hash=p_hash
                                )
                                ui_h = set(ui_h)
                                ui_d = set(ui_d) | {
                                    str(x) for x in branch_ids if str(x).strip()
                                }
                                _pipeline_studio_set_registry_ui(
                                    pipeline_hash=p_hash,
                                    hidden_ids=ui_h,
                                    deleted_ids=ui_d,
                                )
                                st.session_state["pipeline_studio_flow_hidden_ids"] = (
                                    sorted(ui_h | ui_d)
                                )
                                st.session_state[
                                    "pipeline_studio_flow_fit_view_pending"
                                ] = True
                                st.session_state[flow_ts_key] = int(time.time() * 1000)
                                st.session_state["pipeline_studio_history_notice"] = (
                                    f"Soft-deleted {len(branch_ids)} node(s) under `{root_id}`."
                                )
                                _sync_pipeline_targets_after_ui_change()
                            except Exception:
                                pass

                        def _flow_restore_branch(p_hash: str, root_id: str) -> None:
                            try:
                                p_hash = (
                                    p_hash.strip() if isinstance(p_hash, str) else ""
                                )
                                root_id = (
                                    root_id.strip() if isinstance(root_id, str) else ""
                                )
                                if not p_hash or not root_id:
                                    return
                                child_idx = _build_children_index(studio_datasets)
                                branch_ids = {root_id} | _descendants(
                                    root_id, child_idx
                                )
                                ui_h, ui_d = _pipeline_studio_get_registry_ui(
                                    pipeline_hash=p_hash
                                )
                                ui_h = set(ui_h) - branch_ids
                                ui_d = set(ui_d) - branch_ids
                                _pipeline_studio_set_registry_ui(
                                    pipeline_hash=p_hash,
                                    hidden_ids=ui_h,
                                    deleted_ids=ui_d,
                                )
                                st.session_state["pipeline_studio_flow_hidden_ids"] = (
                                    sorted(ui_h | ui_d)
                                )
                                st.session_state[
                                    "pipeline_studio_flow_fit_view_pending"
                                ] = True
                                st.session_state[flow_ts_key] = int(time.time() * 1000)
                                st.session_state["pipeline_studio_history_notice"] = (
                                    f"Restored {len(branch_ids)} node(s) under `{root_id}`."
                                )
                                _sync_pipeline_targets_after_ui_change()
                            except Exception:
                                pass

                        def _flow_hide_branch(p_hash: str, root_id: str) -> None:
                            try:
                                p_hash = (
                                    p_hash.strip() if isinstance(p_hash, str) else ""
                                )
                                root_id = (
                                    root_id.strip() if isinstance(root_id, str) else ""
                                )
                                if not p_hash or not root_id:
                                    return
                                child_idx = _build_children_index(studio_datasets)
                                branch_ids = {root_id} | _descendants(
                                    root_id, child_idx
                                )
                                ui_h, ui_d = _pipeline_studio_get_registry_ui(
                                    pipeline_hash=p_hash
                                )
                                ui_h = set(ui_h) | branch_ids
                                ui_d = set(ui_d)
                                _pipeline_studio_set_registry_ui(
                                    pipeline_hash=p_hash,
                                    hidden_ids=ui_h,
                                    deleted_ids=ui_d,
                                )
                                st.session_state["pipeline_studio_flow_hidden_ids"] = (
                                    sorted(ui_h | ui_d)
                                )
                                st.session_state[
                                    "pipeline_studio_flow_fit_view_pending"
                                ] = True
                                st.session_state[flow_ts_key] = int(time.time() * 1000)
                                st.session_state["pipeline_studio_history_notice"] = (
                                    f"Hid {len(branch_ids)} node(s) under `{root_id}`."
                                )
                                _sync_pipeline_targets_after_ui_change()
                            except Exception:
                                pass

                        def _flow_unhide_branch(p_hash: str, root_id: str) -> None:
                            try:
                                p_hash = (
                                    p_hash.strip() if isinstance(p_hash, str) else ""
                                )
                                root_id = (
                                    root_id.strip() if isinstance(root_id, str) else ""
                                )
                                if not p_hash or not root_id:
                                    return
                                child_idx = _build_children_index(studio_datasets)
                                branch_ids = {root_id} | _descendants(
                                    root_id, child_idx
                                )
                                ui_h, ui_d = _pipeline_studio_get_registry_ui(
                                    pipeline_hash=p_hash
                                )
                                ui_h = set(ui_h) - branch_ids
                                ui_d = set(ui_d)
                                _pipeline_studio_set_registry_ui(
                                    pipeline_hash=p_hash,
                                    hidden_ids=ui_h,
                                    deleted_ids=ui_d,
                                )
                                st.session_state["pipeline_studio_flow_hidden_ids"] = (
                                    sorted(ui_h | ui_d)
                                )
                                st.session_state[
                                    "pipeline_studio_flow_fit_view_pending"
                                ] = True
                                st.session_state[flow_ts_key] = int(time.time() * 1000)
                                st.session_state["pipeline_studio_history_notice"] = (
                                    f"Unhid {len(branch_ids)} node(s) under `{root_id}`."
                                )
                                _sync_pipeline_targets_after_ui_change()
                            except Exception:
                                pass

                        def _flow_hard_delete_branch(
                            p_hash: str | None, root_id: str, clear_history: bool
                        ) -> None:
                            try:
                                import time as _time

                                root_id = (
                                    root_id.strip() if isinstance(root_id, str) else ""
                                )
                                if not root_id:
                                    return
                                team_state = st.session_state.get("team_state", {})
                                team_state = (
                                    team_state if isinstance(team_state, dict) else {}
                                )
                                ds = team_state.get("datasets")
                                ds = ds if isinstance(ds, dict) else {}
                                if root_id not in ds:
                                    st.session_state[
                                        "pipeline_studio_history_notice"
                                    ] = f"Hard delete skipped: `{root_id}` not found."
                                    return
                                child_idx = _build_children_index(ds)
                                branch_ids = {root_id} | _descendants(
                                    root_id, child_idx
                                )
                                ds_new = {
                                    k: v for k, v in ds.items() if k not in branch_ids
                                }
                                if not ds_new:
                                    st.session_state[
                                        "pipeline_studio_history_notice"
                                    ] = "Hard delete skipped: would remove all datasets."
                                    return

                                active_id = team_state.get("active_dataset_id")
                                active_id = (
                                    active_id
                                    if isinstance(active_id, str) and active_id
                                    else None
                                )
                                if active_id in branch_ids:
                                    # Pick newest remaining dataset as the new active id.
                                    best_id = None
                                    best_ts = -1.0
                                    for did, ent in ds_new.items():
                                        if not isinstance(ent, dict):
                                            continue
                                        try:
                                            ts = float(ent.get("created_ts") or 0.0)
                                        except Exception:
                                            ts = 0.0
                                        if ts >= best_ts:
                                            best_ts = ts
                                            best_id = did
                                    active_id = best_id or next(iter(ds_new.keys()))

                                team_state = dict(team_state)
                                team_state["datasets"] = ds_new
                                team_state["active_dataset_id"] = active_id
                                st.session_state["team_state"] = team_state
                                _persist_pipeline_studio_team_state(
                                    team_state=team_state
                                )

                                # Clean up registry UI for the current pipeline hash (best effort).
                                p_hash_clean = (
                                    p_hash.strip()
                                    if isinstance(p_hash, str) and p_hash.strip()
                                    else None
                                )
                                if p_hash_clean:
                                    ui_h, ui_d = _pipeline_studio_get_registry_ui(
                                        pipeline_hash=p_hash_clean
                                    )
                                    ui_h = set(ui_h) - branch_ids
                                    ui_d = set(ui_d) - branch_ids
                                    _pipeline_studio_set_registry_ui(
                                        pipeline_hash=p_hash_clean,
                                        hidden_ids=ui_h,
                                        deleted_ids=ui_d,
                                    )

                                # Drop cached positions for deleted nodes.
                                pos_store = st.session_state.get(
                                    "pipeline_studio_flow_positions"
                                )
                                pos_store = (
                                    pos_store if isinstance(pos_store, dict) else {}
                                )
                                for did in list(branch_ids):
                                    pos_store.pop(str(did), None)
                                st.session_state["pipeline_studio_flow_positions"] = (
                                    pos_store
                                )

                                # Clear in-session history if requested (frees memory).
                                if bool(clear_history):
                                    st.session_state["pipeline_studio_undo_stack"] = []
                                    st.session_state["pipeline_studio_redo_stack"] = []

                                # Refresh semantic registry for new pipelines.
                                try:
                                    pipelines_new = _pipeline_studio_build_pipelines_from_team_state(
                                        team_state
                                    )
                                    _update_pipeline_registry_store_for_pipelines(
                                        pipelines=pipelines_new, datasets=ds_new
                                    )
                                except Exception:
                                    pass

                                # Reset flow caches so the editor rehydrates.
                                st.session_state.pop("pipeline_studio_flow_state", None)
                                st.session_state.pop(
                                    "pipeline_studio_flow_signature", None
                                )
                                st.session_state.pop(
                                    "pipeline_studio_flow_hidden_ids", None
                                )
                                st.session_state.pop(
                                    "pipeline_studio_flow_layout_sig", None
                                )
                                st.session_state[
                                    "pipeline_studio_flow_force_layout"
                                ] = True
                                st.session_state[
                                    "pipeline_studio_flow_fit_view_pending"
                                ] = True
                                st.session_state[flow_ts_key] = int(_time.time() * 1000)

                                if isinstance(active_id, str) and active_id:
                                    st.session_state[
                                        "pipeline_studio_node_id_pending"
                                    ] = active_id
                                    st.session_state[
                                        "pipeline_studio_autofollow_pending"
                                    ] = False
                                st.session_state["pipeline_studio_history_notice"] = (
                                    f"Permanently deleted {len(branch_ids)} node(s) under `{root_id}`."
                                )
                                _sync_pipeline_targets_after_ui_change()
                            except Exception as e:
                                st.session_state["pipeline_studio_history_notice"] = (
                                    f"Hard delete failed: {e}"
                                )

                        c_view, c_open, c_hide, c_delete = st.columns(
                            [0.45, 0.25, 0.15, 0.15]
                        )
                        with c_view:
                            open_view = st.selectbox(
                                "Open workspace view",
                                options=available_views,
                                key=open_view_key,
                                help="Jump from the canvas to a standard Pipeline Studio workspace view.",
                            )
                        with c_open:
                            st.button(
                                "Open in workspace",
                                key="pipeline_studio_flow_open_in_workspace",
                                help="Selects this node in the left rail and turns off auto-follow.",
                                on_click=_flow_open_in_workspace,
                                args=(sel, str(open_view)),
                                width="stretch",
                            )
                        with c_hide:
                            st.button(
                                "Unhide" if in_hidden else "Hide",
                                key="pipeline_studio_flow_hide_toggle",
                                help="Hide/unhide this node across the Studio (soft; does not delete pipeline data).",
                                on_click=_flow_toggle_hidden,
                                args=(pipeline_hash, sel, bool(in_hidden)),
                                disabled=bool(in_deleted) or not bool(pipeline_hash),
                                width="stretch",
                            )
                        with c_delete:
                            st.button(
                                "Restore" if in_deleted else "Delete",
                                key="pipeline_studio_flow_delete_toggle",
                                help="Soft-delete this node across the Studio (can be restored).",
                                on_click=_flow_toggle_deleted,
                                args=(pipeline_hash, sel, bool(in_deleted)),
                                disabled=not bool(pipeline_hash),
                                width="stretch",
                            )

                        node_tabs = st.tabs(["Preview", "Code", "Metadata"])

                        with node_tabs[0]:
                            df_node = _dataset_entry_to_df(entry_obj)
                            if df_node is None:
                                st.info("No tabular data available for this node.")
                            else:
                                n_rows = int(getattr(df_node, "shape", (0, 0))[0] or 0)
                                n_cols = int(getattr(df_node, "shape", (0, 0))[1] or 0)
                                st.caption(f"Shape: {n_rows} rows × {n_cols} columns")
                                preview_rows = st.slider(
                                    "Preview rows",
                                    min_value=5,
                                    max_value=200,
                                    value=25,
                                    step=5,
                                    key="pipeline_studio_flow_preview_rows",
                                )
                                st.dataframe(
                                    df_node.head(int(preview_rows)),
                                    width="stretch",
                                )

                        with node_tabs[1]:
                            title, code_text, code_lang, _kind = (
                                _pipeline_studio_transform_code_snippet(transform)
                            )
                            if not code_text:
                                st.info("No runnable code recorded for this node.")
                            else:
                                editor_key = f"pipeline_studio_code_editor_{sel}"
                                reset_pending = st.session_state.pop(
                                    "pipeline_studio_code_reset_pending", None
                                )
                                if reset_pending == sel:
                                    st.session_state.pop(editor_key, None)
                                fp = entry_obj.get("fingerprint")
                                fp = fp if isinstance(fp, str) and fp else None
                                saved_draft = None
                                saved_meta = {}
                                if fp:
                                    saved_meta = _get_pipeline_studio_code_draft(
                                        fingerprint=fp
                                    )
                                    saved_draft = (
                                        saved_meta.get("draft_code")
                                        if isinstance(saved_meta, dict)
                                        else None
                                    )
                                    saved_draft = (
                                        saved_draft
                                        if isinstance(saved_draft, str)
                                        and saved_draft.strip()
                                        else None
                                    )
                                if (
                                    saved_draft
                                    and editor_key not in st.session_state
                                    and reset_pending != sel
                                ):
                                    st.session_state[editor_key] = saved_draft
                                draft_saved_flag = st.session_state.pop(
                                    "pipeline_studio_code_draft_saved", None
                                )
                                if draft_saved_flag == sel:
                                    st.success("Draft saved to `pipeline_store/`.")
                                has_saved_draft = bool(saved_draft)
                                if not saved_draft:
                                    saved_draft = code_text

                                if saved_draft:
                                    if has_saved_draft:
                                        ts = None
                                        try:
                                            ts = float(
                                                (saved_meta or {}).get("updated_ts")
                                                or (saved_meta or {}).get("created_ts")
                                                or 0.0
                                            )
                                        except Exception:
                                            ts = None
                                        st.caption(
                                            "Loaded saved draft from `pipeline_store/`."
                                            + (
                                                f" (updated_ts={ts:.0f})"
                                                if isinstance(ts, float) and ts
                                                else ""
                                            )
                                        )

                                    draft_code = st.text_area(
                                        title or "Code",
                                        value=code_text,
                                        key=editor_key,
                                        height=260,
                                        help="Edit this draft and ask chat for improvements. You can run drafts for python_function/python_merge/sql_query.",
                                    )
                                    show_preview = st.checkbox(
                                        "Show formatted preview",
                                        value=False,
                                        key=f"pipeline_studio_flow_draft_preview_{sel}",
                                    )
                                    if show_preview:
                                        st.code(
                                            draft_code
                                            if isinstance(draft_code, str)
                                            else "",
                                            language="sql"
                                            if code_lang == "sql"
                                            else "python",
                                        )

                                    c_ask, c_save, c_reset, c_copy = st.columns(
                                        [2, 1, 1, 1]
                                    )
                                    with c_ask:
                                        if st.button(
                                            "Ask AI about this code",
                                            key=f"pipeline_studio_flow_code_ask_ai_{sel}",
                                            help="Send this draft to chat for review and improvement.",
                                        ):
                                            prompt_lines = [
                                                "Pipeline Studio request: review and improve this code draft for the selected pipeline node.",
                                                f"selected_node_id: {sel}",
                                                f"transform_kind: {_kind}"
                                                if _kind
                                                else "",
                                                "",
                                                f"```{code_lang}\n{draft_code}\n```",
                                            ]
                                            st.session_state["chat_prompt_pending"] = (
                                                "\n".join(
                                                    [
                                                        x
                                                        for x in prompt_lines
                                                        if isinstance(x, str)
                                                        and x.strip()
                                                    ]
                                                ).strip()
                                            )
                                            st.rerun()
                                    with c_save:

                                        def _save_code_draft_from_flow(
                                            fingerprint: str | None,
                                            node_id: str,
                                            e_key: str,
                                            t_kind: str,
                                            lang: str,
                                        ) -> None:
                                            if (
                                                not isinstance(fingerprint, str)
                                                or not fingerprint
                                            ):
                                                return
                                            code = st.session_state.get(e_key)
                                            code = code if isinstance(code, str) else ""
                                            _save_pipeline_studio_code_draft(
                                                fingerprint=fingerprint,
                                                dataset_id=node_id,
                                                transform_kind=t_kind,
                                                lang=lang,
                                                draft_code=code,
                                            )
                                            st.session_state[
                                                "pipeline_studio_code_draft_saved"
                                            ] = node_id

                                        st.button(
                                            "Save draft",
                                            key=f"pipeline_studio_flow_code_save_{sel}",
                                            help="Persist this draft to `pipeline_store/` (keyed by dataset fingerprint).",
                                            on_click=_save_code_draft_from_flow,
                                            args=(
                                                fp,
                                                sel,
                                                editor_key,
                                                _kind,
                                                code_lang,
                                            ),
                                        )
                                    with c_reset:

                                        def _queue_code_reset_from_flow(
                                            node_id: str, fingerprint: str | None
                                        ) -> None:
                                            st.session_state[
                                                "pipeline_studio_code_reset_pending"
                                            ] = node_id
                                            if (
                                                isinstance(fingerprint, str)
                                                and fingerprint
                                            ):
                                                _delete_pipeline_studio_code_draft(
                                                    fingerprint=fingerprint
                                                )

                                        st.button(
                                            "Reset draft",
                                            key=f"pipeline_studio_flow_code_reset_{sel}",
                                            help="Discard edits and restore the recorded code for this node.",
                                            on_click=_queue_code_reset_from_flow,
                                            args=(sel, fp),
                                        )
                                    with c_copy:
                                        _render_copy_to_clipboard(
                                            draft_code, label="Copy draft"
                                        )
                                    if _kind in {
                                        "python_function",
                                        "python_merge",
                                        "sql_query",
                                    }:
                                        st.markdown("---")
                                        st.markdown("**Run draft (local)**")
                                        parents = _entry_parent_ids(entry_obj)
                                        if _kind == "python_function":
                                            st.caption(
                                                f"Input dataset: `{parents[0]}` → creates a new `{entry_obj.get('stage')}` node"
                                                if parents
                                                else "Input dataset: (missing parent)"
                                            )
                                        elif _kind == "python_merge":
                                            st.caption(
                                                "Inputs: "
                                                + ", ".join([f"`{p}`" for p in parents])
                                                + f" → creates a new `{entry_obj.get('stage')}` node"
                                                if parents
                                                else "Inputs: (missing parents)"
                                            )
                                        elif _kind == "sql_query":
                                            st.caption(
                                                f"SQL URL: `{_redact_sqlalchemy_url(st.session_state.get('sql_url', DEFAULT_SQL_URL))}`"
                                            )

                                        confirm_key = (
                                            f"pipeline_studio_run_confirm_{sel}"
                                        )
                                        confirm_label = (
                                            "I understand this executes code locally"
                                            if _kind
                                            in {"python_function", "python_merge"}
                                            else "I understand this runs a read-only SQL query"
                                        )
                                        confirmed = st.checkbox(
                                            confirm_label,
                                            key=confirm_key,
                                            help="Creates a new dataset node from the draft output (active).",
                                        )
                                        replace_mode = st.checkbox(
                                            "Replace mode (auto-hide old branch)",
                                            value=True,
                                            key="pipeline_studio_flow_replace_mode",
                                            help="After a successful run, hides the superseded branch (reversible via 'Show hidden steps').",
                                        )

                                        def _run_draft_click_from_flow(
                                            nid: str, e_key: str, k: str
                                        ) -> None:
                                            if k == "python_function":
                                                _run_python_function_draft(
                                                    node_id=nid, editor_key=e_key
                                                )
                                            elif k == "python_merge":
                                                _run_python_merge_draft(
                                                    node_id=nid, editor_key=e_key
                                                )
                                            elif k == "sql_query":
                                                _run_sql_query_draft(
                                                    node_id=nid, editor_key=e_key
                                                )

                                        def _run_draft_only_with_replace_from_flow(
                                            nid: str,
                                            e_key: str,
                                            k: str,
                                            do_replace: bool,
                                        ) -> None:
                                            _run_draft_click_from_flow(nid, e_key, k)
                                            if not bool(do_replace and pipeline_hash):
                                                return
                                            err = st.session_state.get(
                                                "pipeline_studio_run_error"
                                            )
                                            if isinstance(err, str) and err.strip():
                                                return
                                            _flow_hide_branch(pipeline_hash or "", nid)

                                        def _run_draft_and_downstream_from_flow(
                                            nid: str, e_key: str, k: str
                                        ) -> None:
                                            _run_draft_click_from_flow(nid, e_key, k)
                                            err = st.session_state.get(
                                                "pipeline_studio_run_error"
                                            )
                                            if isinstance(err, str) and err.strip():
                                                return
                                            last_repl = st.session_state.get(
                                                "pipeline_studio_last_replacement"
                                            )
                                            if not isinstance(last_repl, dict):
                                                return
                                            old_id = last_repl.get("old_id")
                                            new_id = last_repl.get("new_id")
                                            run_ok = st.session_state.get(
                                                "pipeline_studio_run_success"
                                            )
                                            old_id = (
                                                old_id.strip()
                                                if isinstance(old_id, str)
                                                and old_id.strip()
                                                else None
                                            )
                                            new_id = (
                                                new_id.strip()
                                                if isinstance(new_id, str)
                                                and new_id.strip()
                                                else None
                                            )
                                            if (
                                                not old_id
                                                or not new_id
                                                or (
                                                    isinstance(run_ok, str)
                                                    and run_ok != new_id
                                                )
                                            ):
                                                return
                                            _run_downstream_transforms(old_id, new_id)
                                            if bool(replace_mode and pipeline_hash):
                                                err = st.session_state.get(
                                                    "pipeline_studio_run_error"
                                                )
                                                if not (
                                                    isinstance(err, str) and err.strip()
                                                ):
                                                    _flow_hide_branch(
                                                        pipeline_hash or "", old_id
                                                    )

                                        r1, r2 = st.columns(2)
                                        with r1:
                                            st.button(
                                                "Run draft only",
                                                key=f"pipeline_studio_flow_run_draft_only_{sel}",
                                                disabled=not bool(confirmed),
                                                on_click=_run_draft_only_with_replace_from_flow,
                                                args=(
                                                    sel,
                                                    editor_key,
                                                    _kind,
                                                    bool(replace_mode),
                                                ),
                                                help="Runs the draft and registers the output as a new dataset (active).",
                                                width="stretch",
                                            )
                                        with r2:
                                            st.button(
                                                "Run draft + run downstream",
                                                key=f"pipeline_studio_flow_run_draft_downstream_{sel}",
                                                type="primary",
                                                disabled=not bool(confirmed),
                                                on_click=_run_draft_and_downstream_from_flow,
                                                args=(sel, editor_key, _kind),
                                                help="Runs the draft, then best-effort reruns downstream steps (python_function/python_merge/sql_query).",
                                                width="stretch",
                                            )

                        with node_tabs[2]:
                            locked_ids = st.session_state.get(
                                "pipeline_studio_locked_node_ids", []
                            )
                            locked_set = {
                                str(x) for x in locked_ids if isinstance(x, str) and x
                            }
                            lock_key = f"pipeline_studio_lock_node_{sel}"
                            lock_default = sel in locked_set
                            lock_value = st.checkbox(
                                "Lock node (preserve on AI runs)",
                                value=lock_default,
                                key=lock_key,
                                help="Locked nodes are preserved when chat/AI updates the pipeline.",
                            )
                            if lock_value:
                                locked_set.add(sel)
                            else:
                                locked_set.discard(sel)
                            st.session_state["pipeline_studio_locked_node_ids"] = (
                                sorted(locked_set)
                            )
                            if lock_value != lock_default:
                                _persist_pipeline_studio_team_state(
                                    team_state=st.session_state.get("team_state", {})
                                )
                            st.markdown("---")
                            st.markdown("**Edit node metadata**")
                            label_key = f"pipeline_studio_flow_label_{sel}"
                            stage_key = f"pipeline_studio_flow_stage_{sel}"

                            current_label = (
                                entry_obj.get("label") or meta.get("label") or sel
                            )
                            current_label = (
                                str(current_label) if current_label is not None else sel
                            )

                            current_stage = (
                                entry_obj.get("stage") or meta.get("stage") or "custom"
                            )
                            current_stage = (
                                str(current_stage).strip()
                                if isinstance(current_stage, str)
                                else "custom"
                            )
                            stage_options = [
                                "raw",
                                "sql",
                                "wrangled",
                                "cleaned",
                                "feature",
                                "model",
                                "custom",
                            ]
                            if current_stage and current_stage not in stage_options:
                                stage_options = [current_stage] + stage_options

                            st.text_input(
                                "Label",
                                value=current_label,
                                key=label_key,
                                help="Update the display label for this node.",
                            )
                            st.selectbox(
                                "Stage",
                                options=stage_options,
                                index=stage_options.index(current_stage)
                                if current_stage in stage_options
                                else 0,
                                key=stage_key,
                                help="Updates the node stage (used for coloring + grouping).",
                            )
                            st.button(
                                "Save metadata",
                                key=f"pipeline_studio_flow_save_meta_{sel}",
                                on_click=_flow_save_node_metadata,
                                args=(sel, label_key, stage_key),
                                width="stretch",
                            )

                            st.markdown("---")
                            with st.expander("Branch actions", expanded=False):
                                child_idx = _build_children_index(studio_datasets)
                                branch_ids = {sel} | _descendants(sel, child_idx)
                                ui_h, ui_d = (
                                    _pipeline_studio_get_registry_ui(
                                        pipeline_hash=pipeline_hash
                                    )
                                    if pipeline_hash
                                    else (set(), set())
                                )
                                st.caption(
                                    f"Branch size: {len(branch_ids)} | hidden: {len(branch_ids & set(ui_h))} | "
                                    f"deleted: {len(branch_ids & set(ui_d))}"
                                )

                                c_hide, c_unhide = st.columns(2)
                                with c_hide:
                                    st.button(
                                        "Hide branch",
                                        key=f"pipeline_studio_flow_hide_branch_{sel}",
                                        disabled=not bool(pipeline_hash),
                                        on_click=_flow_hide_branch,
                                        args=(pipeline_hash or "", sel),
                                        width="stretch",
                                    )
                                with c_unhide:
                                    st.button(
                                        "Unhide branch",
                                        key=f"pipeline_studio_flow_unhide_branch_{sel}",
                                        disabled=not bool(pipeline_hash),
                                        on_click=_flow_unhide_branch,
                                        args=(pipeline_hash or "", sel),
                                        width="stretch",
                                    )

                                c_delete, c_restore = st.columns(2)
                                with c_delete:
                                    confirm_soft = st.checkbox(
                                        "Confirm soft-delete",
                                        key=f"pipeline_studio_flow_delete_branch_confirm_{sel}",
                                        help="Marks this node and descendants as deleted in the pipeline registry (reversible).",
                                    )
                                    st.button(
                                        "Delete branch (soft)",
                                        key=f"pipeline_studio_flow_delete_branch_{sel}",
                                        type="secondary",
                                        disabled=not bool(
                                            confirm_soft and pipeline_hash
                                        ),
                                        on_click=_flow_delete_branch,
                                        args=(pipeline_hash or "", sel),
                                        width="stretch",
                                    )
                                with c_restore:
                                    st.button(
                                        "Restore branch",
                                        key=f"pipeline_studio_flow_restore_branch_{sel}",
                                        disabled=not bool(pipeline_hash),
                                        on_click=_flow_restore_branch,
                                        args=(pipeline_hash or "", sel),
                                        width="stretch",
                                    )

                                st.markdown("---")
                                st.caption(
                                    "Hard delete permanently removes dataset data from memory (and will change the pipeline)."
                                )
                                clear_history = st.checkbox(
                                    "Also clear undo/redo history",
                                    value=True,
                                    key=f"pipeline_studio_flow_hard_delete_clear_history_{sel}",
                                    help="Clears Pipeline Studio undo/redo stacks (may free memory).",
                                )
                                confirm_hard = st.checkbox(
                                    "I understand this permanently deletes data",
                                    key=f"pipeline_studio_flow_hard_delete_confirm_{sel}",
                                )
                                st.button(
                                    "Hard delete branch (permanent)",
                                    key=f"pipeline_studio_flow_hard_delete_branch_{sel}",
                                    type="primary",
                                    disabled=not bool(confirm_hard),
                                    on_click=_flow_hard_delete_branch,
                                    args=(pipeline_hash, sel, bool(clear_history)),
                                    width="stretch",
                                )

                            st.markdown("---")
                            st.markdown("**Snapshot metadata**")
                            st.json(meta)


def _render_pipeline_studio_safe() -> None:
    try:
        _render_pipeline_studio()
    except Exception as e:
        st.error(f"Could not render Pipeline Studio: {e}")


if hasattr(st, "fragment"):

    @st.fragment
    def _render_pipeline_studio_fragment() -> None:
        _render_pipeline_studio_safe()

else:

    def _render_pipeline_studio_fragment() -> None:
        _render_pipeline_studio_safe()


open_studio_requested = bool(
    st.session_state.pop("pipeline_studio_open_requested", False)
)
if open_studio_requested:
    _open_pipeline_studio_dialog()

st.markdown("---")
st.subheader("Analysis Details")
details = st.session_state.get("details")
details = details if isinstance(details, list) else []
if not details:
    st.info("No analysis details yet.")
else:
    default_idx = len(details) - 1
    selected = st.selectbox(
        "Inspect a prior turn",
        options=list(range(len(details))),
        index=default_idx,
        format_func=lambda i: f"Turn {i + 1}",
        key="analysis_details_turn_select",
    )
    try:
        detail = details[int(selected)]
        _render_analysis_detail(detail, key_suffix=f"bottom_{selected}")
    except Exception as e:
        st.warning(f"Could not render analysis details: {e}")
# Note: analysis details rendering is best-effort; errors should not break the app.

drawer_open = bool(st.session_state.get("pipeline_studio_drawer_open", False))
if not _pipeline_studio_is_docked():
    drawer_open = False
    st.session_state["pipeline_studio_drawer_open"] = False
if drawer_open and _pipeline_studio_is_docked():
    with drawer_placeholder.container():
        st.markdown(
            "\n".join(
                [
                    "<style>",
                    'div[data-testid="stExpanderDetails"]:has(#pipeline-studio-drawer-marker) {',
                    "  height: calc(100vh - 220px);",
                    "  max-height: none;",
                    "  min-height: 320px;",
                    "  overflow-y: auto;",
                    "  padding: 0.75rem 0.75rem 1rem 0.75rem;",
                    "  border: 1px solid rgba(255,255,255,0.08);",
                    "  border-radius: 12px;",
                    "  background: rgba(15, 15, 15, 0.45);",
                    "}",
                    "#pipeline-studio-drawer-marker {",
                    "  display: none;",
                    "}",
                    "</style>",
                ]
            ),
            unsafe_allow_html=True,
        )
        with st.expander("Pipeline Studio (Docked)", expanded=True):
            st.markdown(
                '<span id="pipeline-studio-drawer-marker"></span>',
                unsafe_allow_html=True,
            )
            c_drawer_close, _c_drawer_hint = st.columns(
                [0.18, 0.82], vertical_alignment="center"
            )
            with c_drawer_close:
                if st.button(
                    "Close Studio",
                    key="pipeline_studio_drawer_close",
                    width="stretch",
                ):
                    st.session_state["pipeline_studio_drawer_open"] = False
            with _c_drawer_hint:
                st.caption("Docked view keeps Studio inline while you chat.")
            _render_pipeline_studio_fragment()
else:
    drawer_placeholder.empty()

st.markdown('<span id="pipeline-studio-fab-anchor"></span>', unsafe_allow_html=True)
if st.button(
    "",
    key="pipeline_studio_fab_trigger",
    # help="Open Pipeline Studio",
):
    _request_open_pipeline_studio()

st.markdown(
    "\n".join(
        [
            '<div id="page-bottom"></div>',
            '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">',
            "<style>",
            ":root {",
            "  --pipeline-fab-size: 2.5rem;",
            "  --pipeline-fab-gap: 0.5rem;",
            "}",
            ".pipeline-studio-fab {",
            "  position: fixed;",
            "  right: 1.5rem;",
            "  bottom: 2.5rem;",
            "  z-index: 9999;",
            "  display: flex;",
            "  flex-direction: column;",
            "  gap: var(--pipeline-fab-gap);",
            "}",
            ".pipeline-studio-fab a {",
            "  background-color: #1c3d66;",
            "  color: #f5f7fa;",
            "  border: none;",
            "  border-radius: 999px;",
            "  width: var(--pipeline-fab-size);",
            "  height: var(--pipeline-fab-size);",
            "  padding: 0;",
            "  font-size: 1rem;",
            "  cursor: pointer;",
            "  text-decoration: none;",
            "  text-align: center;",
            "  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);",
            "  display: flex;",
            "  align-items: center;",
            "  justify-content: center;",
            "}",
            ".pipeline-studio-fab a:hover {",
            "  background-color: #27527f;",
            "}",
            ".element-container:has(#pipeline-studio-fab-anchor) {",
            "  height: 0;",
            "  margin: 0;",
            "}",
            ".element-container:has(#pipeline-studio-fab-anchor) + div {",
            "  height: 0;",
            "  margin: 0;",
            "}",
            ".element-container:has(#pipeline-studio-fab-anchor) + div button {",
            "  position: fixed;",
            "  right: 1.5rem;",
            "  bottom: calc(2.5rem + (var(--pipeline-fab-size) + var(--pipeline-fab-gap)) * 2);",
            "  z-index: 9999;",
            "  width: var(--pipeline-fab-size);",
            "  height: var(--pipeline-fab-size);",
            "  padding: 0;",
            "  border-radius: 999px;",
            "  border: none;",
            "  background-color: #1c3d66;",
            "  color: #f5f7fa;",
            "  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);",
            "  display: flex;",
            "  align-items: center;",
            "  justify-content: center;",
            "}",
            ".element-container:has(#pipeline-studio-fab-anchor) + div button:hover {",
            "  background-color: #27527f;",
            "}",
            ".element-container:has(#pipeline-studio-fab-anchor) + div button span {",
            "  font-size: 0;",
            "  line-height: 0;",
            "}",
            ".element-container:has(#pipeline-studio-fab-anchor) + div button::before {",
            '  content: "\\f542";',
            '  font-family: "Font Awesome 6 Free";',
            "  font-weight: 900;",
            "  font-size: 1rem;",
            "}",
            "</style>",
            '<div class="pipeline-studio-fab">',
            '  <a href="#page-top" title="Scroll to top"><i class="fa-solid fa-arrow-up"></i></a>',
            '  <a href="#page-bottom" title="Scroll to bottom"><i class="fa-solid fa-arrow-down"></i></a>',
            "</div>",
        ]
    ),
    unsafe_allow_html=True,
)
