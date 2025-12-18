"""
Streamlit app for the Supervisor-led AI Data Science Team.

Command:
    streamlit run apps/supervisor-ds-team-app/app.py
"""

from __future__ import annotations

import re
import uuid
import os
import json
import inspect
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


st.set_page_config(
    page_title="Supervisor Data Science Team", page_icon=":bar_chart:", layout="wide"
)
TITLE = "Supervisor-led AI Data Science Team"
st.title(TITLE)

UI_DETAIL_MARKER_PREFIX = "DETAILS_INDEX:"
DEFAULT_SQL_URL = "sqlite:///:memory:"
SQL_URL_INPUT_KEY = "sql_url_input"
SQL_URL_SYNC_FLAG = "_sync_sql_url_input"

APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
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
                        ts = float(rec.get("updated_ts") or rec.get("created_ts") or 0.0)
                    except Exception:
                        ts = 0.0
                items.append((ts, fp))
            items.sort(reverse=True)
            keep = {fp for _ts, fp in items[: int(PIPELINE_STUDIO_ARTIFACT_STORE_MAX_ITEMS)]}
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


def _update_pipeline_studio_artifact_store_for_dataset(dataset_id: str, artifacts: dict) -> None:
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
        fingerprint = fingerprint if isinstance(fingerprint, str) and fingerprint else None
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
                "stage": entry.get("stage") if isinstance(entry.get("stage"), str) else rec.get("stage"),
                "label": entry.get("label") if isinstance(entry.get("label"), str) else rec.get("label"),
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
                                data.get("version") or PIPELINE_STUDIO_FLOW_LAYOUT_VERSION
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
                        ts = float(rec.get("updated_ts") or rec.get("created_ts") or 0.0)
                    except Exception:
                        ts = 0.0
                items.append((ts, str(ph)))
            items.sort(reverse=True)
            keep = {ph for _ts, ph in items[: int(PIPELINE_STUDIO_FLOW_LAYOUT_MAX_ITEMS)]}
            by_ph = {ph: by_ph[ph] for ph in keep if ph in by_ph}
            store["by_pipeline_hash"] = by_ph

        store["version"] = PIPELINE_STUDIO_FLOW_LAYOUT_VERSION
        store["updated_ts"] = time.time()
        store["path"] = PIPELINE_STUDIO_FLOW_LAYOUT_PATH

        out_dir = os.path.dirname(PIPELINE_STUDIO_FLOW_LAYOUT_PATH) or "."
        os.makedirs(out_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix="._pipeline_studio_flow_", suffix=".json", dir=out_dir)
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


# ---------------- Sidebar ----------------
key_status = None
with st.sidebar:
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
            st.info("Please enter your OpenAI API key to proceed (or switch to Ollama).")
            st.stop()

        model_choice = st.selectbox(
            "Model",
            [
                "gpt-4.1-mini",
                "gpt-4.1",
                "gpt-4o-mini",
                "gpt-4o",
                "gpt-5.2",
                "gpt-5.1-mini",
                "gpt-5.1",
            ],
            key="openai_model_choice",
        )
    else:
        if ChatOllama is None:
            st.error(
                "Ollama support requires `langchain-ollama`. Install it with `pip install langchain-ollama`."
            )
            st.stop()

        default_ollama_url = st.session_state.get("ollama_base_url") or "http://localhost:11434"
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

        if st.button("Check Ollama connection", use_container_width=True, key="ollama_check"):
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

        st.caption("Tip: start Ollama with `ollama serve` and pull a model with `ollama pull <model>`.")

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

    if datasets:
        ordered = sorted(
            datasets.items(),
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

        st.markdown("**Merge options**")
        merge_default_ids = st.session_state.get("merge_dataset_ids") or []
        merge_ids = st.multiselect(
            "Datasets to merge/join",
            options=[did for did, _e in ordered],
            default=[d for d in merge_default_ids if d in datasets],
            format_func=_fmt_dataset,
            help="Select 2+ datasets to merge. Then ask in chat to merge (or describe how to join).",
            key="merge_dataset_ids",
        )
        merge_operation = st.selectbox(
            "Merge operation",
            options=["join", "concat"],
            index=0,
            help="Use join for relational merges (SQL-like). Use concat to stack or combine columns by index.",
            key="merge_operation",
        )
        if merge_operation == "join":
            st.selectbox(
                "Join type",
                options=["inner", "left", "right", "outer"],
                index=0,
                help="How to join datasets when using 'join'.",
                key="merge_how",
            )
            st.text_input(
                "Join keys (optional, comma-separated)",
                value=st.session_state.get("merge_on", ""),
                help="Example: customerID. If blank, the team will try to infer common keys or ask.",
                key="merge_on",
            )
            with st.expander("Advanced join options", expanded=False):
                st.text_input(
                    "Left keys (optional, comma-separated)",
                    value=st.session_state.get("merge_left_on", ""),
                    help="Use when left/right key names differ.",
                    key="merge_left_on",
                )
                st.text_input(
                    "Right keys (optional, comma-separated)",
                    value=st.session_state.get("merge_right_on", ""),
                    help="Use when left/right key names differ.",
                    key="merge_right_on",
                )
                st.text_input(
                    "Suffixes (optional, comma-separated)",
                    value=st.session_state.get("merge_suffixes", "_x,_y"),
                    help="Example: _left,_right",
                    key="merge_suffixes",
                )
        else:
            st.selectbox(
                "Concat axis",
                options=[0, 1],
                index=0,
                help="axis=0 stacks rows; axis=1 concatenates columns by index.",
                key="merge_axis",
            )
            st.checkbox(
                "Ignore index (axis=0)",
                value=True,
                help="When stacking rows, reset the index for a clean result.",
                key="merge_ignore_index",
            )
    else:
        st.session_state["active_dataset_id_override"] = ""
        st.session_state["merge_dataset_ids"] = []
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
    if st.session_state.get("last_pipeline_persist_dir"):
        st.caption(
            f"Last saved pipeline: `{st.session_state.get('last_pipeline_persist_dir')}`"
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
        st.session_state["active_dataset_id_override"] = ""
        st.session_state.selected_data_provenance = None
        st.session_state.last_pipeline_persist_dir = None
        msgs = StreamlitChatMessageHistory(key="supervisor_ds_msgs")
        msgs.clear()
        msgs.add_ai_message("How can the data science team help today?")
        st.session_state.thread_id = str(uuid.uuid4())
        # Reset checkpointer when clearing chat
        st.session_state.checkpointer = get_checkpointer() if add_memory else None

# Hard gate: only require an API key for OpenAI provider
llm_provider_selected = st.session_state.get("llm_provider") or "OpenAI"
resolved_api_key = st.session_state.get("OPENAI_API_KEY")
if llm_provider_selected == "OpenAI":
    if not resolved_api_key:
        st.info("Please enter your OpenAI API key in the sidebar to proceed.")
        st.stop()
    if key_status == "bad":
        st.error("Invalid OpenAI API key. Please fix it in the sidebar.")
        st.stop()
else:
    if not (st.session_state.get("ollama_model") or "").strip():
        st.info("Please enter an Ollama model name in the sidebar (e.g., `llama3.1:8b`).")
        st.stop()


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
        llm = ChatOpenAI(model=model_name, api_key=openai_api_key)
    workflow_planner_agent = WorkflowPlannerAgent(llm)
    data_loader_agent = DataLoaderToolsAgent(
        llm, invoke_react_agent_kwargs={"recursion_limit": 4}
    )
    data_wrangling_agent = DataWranglingAgent(llm, log=False)
    data_cleaning_agent = DataCleaningAgent(llm, log=False)
    eda_tools_agent = EDAToolsAgent(llm, log_tool_calls=True)
    data_visualization_agent = DataVisualizationAgent(llm, log=False)
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
        st.markdown("**Data preview**")
        st.dataframe(df.head(preview_rows))
        return df.to_dict(), df.head(preview_rows), provenance

    return None, None, None


def render_history(history: list[BaseMessage]):
    def _render_analysis_detail(detail: dict, key_suffix: str):
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
            if pipelines:
                options = [
                    ("Model (latest feature)", "model"),
                    ("Active dataset", "active"),
                    ("Latest dataset", "latest"),
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
            if isinstance(pipe, dict) and pipe.get("lineage"):
                inputs = pipe.get("inputs") or []
                inputs_txt = ""
                if isinstance(inputs, list) and inputs:
                    inputs_txt = (
                        f"  \n**Inputs:** {', '.join([f'`{i}`' for i in inputs if i])}"
                    )
                st.markdown(
                    f"**Pipeline hash:** `{pipe.get('pipeline_hash')}`  \n"
                    f"**Target dataset id:** `{pipe.get('target_dataset_id')}`  \n"
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
                        use_container_width=True,
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
                reports.get("sweetviz_report_file")
                if isinstance(reports, dict)
                else None
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
                        st.dataframe(
                            pd.DataFrame(model_info), use_container_width=True
                        )
                    elif isinstance(model_info, list):
                        st.dataframe(
                            pd.DataFrame(model_info), use_container_width=True
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
                        use_container_width=True,
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
                    or any(
                        str(c).lower().startswith("actual_") for c in preds_df.columns
                    )
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
                                use_container_width=True,
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
                                use_container_width=True,
                            )
                            return
                        if isinstance(obj, list):
                            st.dataframe(pd.DataFrame(obj), use_container_width=True)
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


tab_chat, tab_pipeline_studio = st.tabs(["Chat", "Pipeline Studio"])

with tab_chat:
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
prompt = tab_chat.chat_input("Ask the data science team...")
if prompt:
    if llm_provider_selected == "OpenAI" and (
        not resolved_api_key or key_status == "bad"
    ):
        tab_chat.error(
            "OpenAI API key is required and must be valid. Enter it in the sidebar."
        )
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

    tab_chat.chat_message("human").write(display_prompt)
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
            tab_chat.chat_message("assistant").write(msg_text)
            msgs.add_ai_message(msg_text)

    data_raw_dict = st.session_state.get("selected_data_raw")
    input_provenance = st.session_state.get("selected_data_provenance")

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
                        "merge": {
                            "dataset_ids": st.session_state.get("merge_dataset_ids")
                            or [],
                            "operation": st.session_state.get("merge_operation")
                            or "join",
                            "how": st.session_state.get("merge_how") or "inner",
                            "on": st.session_state.get("merge_on") or "",
                            "left_on": st.session_state.get("merge_left_on") or "",
                            "right_on": st.session_state.get("merge_right_on") or "",
                            "suffixes": st.session_state.get("merge_suffixes")
                            or "_x,_y",
                            "axis": st.session_state.get("merge_axis", 0),
                            "ignore_index": bool(
                                st.session_state.get("merge_ignore_index", True)
                            ),
                        },
                        "sql_url": _redact_sqlalchemy_url(
                            st.session_state.get("sql_url", DEFAULT_SQL_URL)
                        ),
                    }
                },
                "data_raw": data_raw_dict,
            }
            if input_provenance:
                invoke_payload["artifacts"]["input_dataset"] = input_provenance
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
                            "active_data_key",
                            "active_dataset_id",
                            "datasets",
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
            progress_box = tab_chat.empty() if show_progress else None
            if progress_box is not None:
                progress_box.info("Working…")

            show_live_logs = bool(st.session_state.get("show_live_logs", False))
            log_container = tab_chat.empty() if show_live_logs else None
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
                tab_chat.error(f"Error running team (rate limit): {e}")
                tab_chat.info(
                    "Try again in ~60s, or reduce load by disabling memory, lowering recursion, "
                    "or switching to a smaller model."
                )
            else:
                tab_chat.error(f"Error running team: {e}")
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
                        state_updates[k] = _normalize_datasets(result.get(k))
                    else:
                        state_updates[k] = _maybe_df_to_dict(result.get(k))
            if state_updates:
                st.session_state.team_state = {
                    **(st.session_state.team_state or {}),
                    **state_updates,
                }
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
            tab_chat.chat_message("assistant").write(getattr(last_ai, "content", ""))

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

    def _latest_detail_for_dataset_id(
        dataset_id: str, *, require_key: str | None = None
    ) -> dict | None:
        details = st.session_state.get("details") or []
        if not isinstance(details, list):
            return None
        for d in reversed(details):
            if not isinstance(d, dict):
                continue
            pipelines = d.get("pipelines") if isinstance(d.get("pipelines"), dict) else {}
            active_pipe = pipelines.get("active") if isinstance(pipelines.get("active"), dict) else None
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
                    "<div style=\"display:flex; align-items:center; gap:0.5rem;\">",
                    f"<button id=\"copy-btn\" style=\"padding:0.25rem 0.75rem; border-radius:0.5rem; border:1px solid rgba(49,51,63,0.2); background:rgba(255,255,255,0.0); cursor:pointer; font-size:0.875rem;\">{label}</button>",
                    "<span id=\"copy-status\" style=\"font-size:0.85rem; color:rgba(49,51,63,0.65);\"></span>",
                    "</div>",
                    "<script>",
                    f"const text = {payload};",
                    "const btn = document.getElementById('copy-btn');",
                    "const status = document.getElementById('copy-status');",
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
                    "    if (status) status.textContent = 'Copied';",
                    "    setTimeout(() => { if (status) status.textContent = ''; }, 1500);",
                    "  } catch (e) {",
                    "    if (status) status.textContent = 'Copy failed';",
                    "    setTimeout(() => { if (status) status.textContent = ''; }, 2000);",
                    "  }",
                    "}",
                    "if (btn) btn.addEventListener('click', doCopy);",
                    "</script>",
                ]
            ),
            height=40,
        )

    if not studio_datasets:
        st.info("No pipeline yet. Load data and run a transform to build one.")
    else:
        target_options = [
            ("Model (latest feature)", "model"),
            ("Active dataset", "active"),
            ("Latest dataset", "latest"),
        ]
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
                str(x.get("id")): x for x in lineage if isinstance(x, dict) and x.get("id")
            }
            node_ids = [did for did in meta_by_id.keys() if did]

            left, right = st.columns([0.35, 0.65], gap="large")

            with left:
                st.markdown(
                    f"**Pipeline hash:** `{pipe.get('pipeline_hash')}`  \n"
                    f"**Target dataset id:** `{pipe.get('target_dataset_id')}`  \n"
                    f"**Active dataset id:** `{pipe.get('active_dataset_id')}`"
                )

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
                if isinstance(script, str) and script.strip():
                    st.download_button(
                        "Download pipeline script",
                        data=script.encode("utf-8"),
                        file_name=f"pipeline_repro_{pipe.get('target') or 'model'}.py",
                        mime="text/x-python",
                        key="pipeline_studio_download_repro",
                    )

                pending_node = st.session_state.pop("pipeline_studio_node_id_pending", None)
                if (
                    isinstance(pending_node, str)
                    and pending_node
                    and isinstance(node_ids, list)
                    and pending_node in node_ids
                ):
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

                # Keep selection valid and optionally auto-follow newest node.
                if node_ids:
                    desired = node_ids[-1]
                    current = st.session_state.get("pipeline_studio_node_id")
                    if auto_follow:
                        st.session_state["pipeline_studio_node_id"] = desired
                    elif not isinstance(current, str) or current not in node_ids:
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
                    meta = f" ({', '.join(bits)})" if bits else ""
                    return f"{label}{meta}"

                selected_node_id = st.selectbox(
                    "Pipeline step",
                    options=node_ids,
                    format_func=_node_label,
                    key="pipeline_studio_node_id",
                )

                m = meta_by_id.get(selected_node_id) or {}
                with st.expander("Selected step details", expanded=False):
                    st.json(m)

                compare_node_id = None
                compare_mode = st.checkbox(
                    "Compare mode",
                    value=bool(st.session_state.get("pipeline_studio_compare_mode", False)),
                    key="pipeline_studio_compare_mode",
                    help="Compare two pipeline steps side-by-side (schema + table preview).",
                )
                if compare_mode:
                    compare_options = [did for did in node_ids if did != selected_node_id]
                    compare_options = compare_options or node_ids
                    default_compare = compare_options[-1] if compare_options else selected_node_id
                    current_compare = st.session_state.get("pipeline_studio_compare_node_id")
                    if (
                        not isinstance(current_compare, str)
                        or current_compare not in compare_options
                    ):
                        st.session_state["pipeline_studio_compare_node_id"] = default_compare

                    compare_node_id = st.selectbox(
                        "Compare with",
                        options=compare_options,
                        format_func=_node_label,
                        key="pipeline_studio_compare_node_id",
                    )

            with right:
                if compare_mode and isinstance(compare_node_id, str) and compare_node_id:
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
                        st.plotly_chart(fig, use_container_width=True, key=widget_key)

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
                            code_text if isinstance(code_text, str) and code_text.strip() else None
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
                                        use_container_width=True,
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
                                                    "delta (selected-compare)": (miss_a - miss_b).values,
                                                }
                                            )
                                            miss_df = miss_df[
                                                miss_df["delta (selected-compare)"]
                                                != 0
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
                                                    use_container_width=True,
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
                                    df_a.head(int(rows)), use_container_width=True
                                )
                        with cb:
                            st.markdown(f"**B (compare): {_node_label(b_id)}**")
                            if df_b is None:
                                st.info("No tabular data available for B.")
                            else:
                                st.caption(f"Shape: {df_b.shape[0]} × {df_b.shape[1]}")
                                st.dataframe(
                                    df_b.head(int(rows)), use_container_width=True
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
                            st.markdown(f"**B (compare): {title_b or kind_b or 'Code'}**")
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
                                    st.code(diff_text if diff_text else "—", language="diff")
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
                                st.session_state["pipeline_studio_compare_rowdiff_key"] = (
                                    key_options[0]
                                )
                            key_col = st.selectbox(
                                "Key column",
                                options=key_options,
                                key="pipeline_studio_compare_rowdiff_key",
                                help="Used to align rows between A (selected) and B (compare).",
                            )
                            compare_candidates = [c for c in key_options if c != key_col]
                            compare_candidates = compare_candidates[:50]
                            default_cols = compare_candidates[:10]
                            current_cols = st.session_state.get(
                                "pipeline_studio_compare_rowdiff_cols"
                            )
                            if not isinstance(current_cols, list) or any(
                                c not in compare_candidates for c in current_cols
                            ):
                                st.session_state["pipeline_studio_compare_rowdiff_cols"] = (
                                    default_cols
                                )
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
                                    set_a_keys = set(a_keys.astype(str).unique().tolist())
                                try:
                                    set_b_keys = set(b_keys.unique().tolist())
                                except Exception:
                                    set_b_keys = set(b_keys.astype(str).unique().tolist())

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
                                            a_small[a_small[key_col].isin(sample_keys)].head(
                                                int(preview_rows)
                                            ),
                                            use_container_width=True,
                                        )
                                if only_b:
                                    with st.expander(
                                        "Sample rows only in B (compare)", expanded=False
                                    ):
                                        sample_keys = list(only_b)[: int(preview_rows)]
                                        st.dataframe(
                                            b_small[b_small[key_col].isin(sample_keys)].head(
                                                int(preview_rows)
                                            ),
                                            use_container_width=True,
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
                                        if a_col not in merged.columns or b_col not in merged.columns:
                                            continue
                                        a_vals = merged[a_col]
                                        b_vals = merged[b_col]
                                        try:
                                            neq = ~(
                                                a_vals.eq(b_vals)
                                                | (a_vals.isna() & b_vals.isna())
                                            )
                                        except Exception:
                                            neq = a_vals.astype(str) != b_vals.astype(str)
                                        n_bad = int(neq.sum())
                                        if n_bad:
                                            diff_counts.append(
                                                {"column": col, "mismatched_rows": n_bad}
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
                                        st.dataframe(diff_df, use_container_width=True)
                                        inspect_default = str(diff_df.iloc[0]["column"])
                                        current_inspect = st.session_state.get(
                                            "pipeline_studio_compare_rowdiff_inspect_col"
                                        )
                                        inspect_options = [str(x) for x in diff_df["column"].tolist()]
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
                                                | (merged[a_col].isna() & merged[b_col].isna())
                                            )
                                        except Exception:
                                            neq = merged[a_col].astype(str) != merged[b_col].astype(str)
                                        preview = merged.loc[
                                            neq, [key_col, a_col, b_col]
                                        ].head(int(preview_rows)).copy()
                                        preview = preview.rename(
                                            columns={
                                                a_col: f"{inspect_col} (selected)",
                                                b_col: f"{inspect_col} (compare)",
                                            }
                                        )
                                        st.dataframe(preview, use_container_width=True)

                    # Compare mode replaces the workspace when enabled.
                    return
                pending_view = st.session_state.pop("pipeline_studio_view_pending", None)
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
                view = st.radio(
                    "Workspace",
                    [
                        "Table",
                        "Chart",
                        "EDA",
                        "Code",
                        "Model",
                        "Predictions",
                        "MLflow",
                        "Visual Editor",
                    ],
                    horizontal=True,
                    key="pipeline_studio_view",
                )

                entry = (
                    studio_datasets.get(selected_node_id)
                    if isinstance(studio_datasets, dict)
                    else None
                )
                entry = entry if isinstance(entry, dict) else {}
                df_sel = _dataset_entry_to_df(entry)

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
                        st.dataframe(
                            df_sel.head(int(rows)), use_container_width=True
                        )
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
                                        st.caption(
                                            f"Showing first {max_cols} columns."
                                        )
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
                                    st.dataframe(
                                        miss_df, use_container_width=True
                                    )
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
                    if not graph_json:
                        detail = _latest_detail_for_dataset_id(
                            selected_node_id, require_key="plotly_graph"
                        )
                        graph_json = (
                            detail.get("plotly_graph") if isinstance(detail, dict) else None
                        )
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
                                use_container_width=True,
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
                            detail.get("eda_reports") if isinstance(detail, dict) else None
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
                    prov = entry.get("provenance") if isinstance(entry.get("provenance"), dict) else {}
                    transform = prov.get("transform") if isinstance(prov.get("transform"), dict) else {}
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
                        st.code(code_text, language=code_lang)
                        try:
                            ext = "sql" if code_lang == "sql" else "py"
                            mime = (
                                "application/sql"
                                if code_lang == "sql"
                                else "text/x-python"
                            )
                            c_copy, c_download = st.columns([0.22, 0.26])
                            with c_copy:
                                _render_copy_to_clipboard(code_text, label="Copy snippet")
                            with c_download:
                                st.download_button(
                                    "Download snippet",
                                    data=code_text.encode("utf-8"),
                                    file_name=f"{selected_node_id}_{kind or 'step'}.{ext}",
                                    mime=mime,
                                    key=f"pipeline_studio_download_snippet_{selected_node_id}",
                                )
                        except Exception:
                            pass
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
                                    use_container_width=True,
                                )
                            elif isinstance(model_info, list):
                                st.dataframe(
                                    pd.DataFrame(model_info),
                                    use_container_width=True,
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
                            fig = _apply_streamlit_plot_style(
                                pio.from_json(payload)
                            )
                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                                key=f"pipeline_studio_eval_chart_{selected_node_id}",
                            )
                        except Exception as e:
                            st.error(f"Error rendering evaluation chart: {e}")
                    if model_info is None and eval_art is None and eval_graph is None:
                        st.info(
                            "No model artifacts found for this dataset yet. Try: `train a model` while this dataset is active."
                        )

                elif view == "Predictions":
                    prov = entry.get("provenance") if isinstance(entry.get("provenance"), dict) else {}
                    transform = prov.get("transform") if isinstance(prov.get("transform"), dict) else {}
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
                        st.dataframe(df_sel.head(50), use_container_width=True)

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
                                        df[preferred_cols]
                                        if preferred_cols
                                        else df,
                                        use_container_width=True,
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
                                        df[preferred_cols]
                                        if preferred_cols
                                        else df,
                                        use_container_width=True,
                                    )
                                    return
                                if isinstance(obj, list):
                                    st.dataframe(
                                        pd.DataFrame(obj),
                                        use_container_width=True,
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

                    base_node_ids = [did for did in node_ids if isinstance(did, str) and did]
                    if not base_node_ids:
                        st.info("No pipeline nodes available to render.")
                        return

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
                        st.session_state["pipeline_studio_flow_pipeline_hash"] = pipeline_hash
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
                        if isinstance(hid, list):
                            st.session_state["pipeline_studio_flow_hidden_ids"] = hid
                        st.session_state["pipeline_studio_flow_fit_view_pending"] = True

                    hidden_ids = st.session_state.get("pipeline_studio_flow_hidden_ids")
                    hidden_ids = hidden_ids if isinstance(hidden_ids, list) else []
                    hidden_set = {str(x) for x in hidden_ids if isinstance(x, str) and x}

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

                    def _node_style(stage: str) -> dict:
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
                        return {
                            "background": bg,
                            "border": border,
                            "borderRadius": "12px",
                            "color": "rgba(255,255,255,0.92)",
                            "fontSize": "12px",
                            "padding": "10px 12px",
                            "minWidth": "160px",
                        }

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

                    base_set = set(base_node_ids)
                    hidden_set = {hid for hid in hidden_set if hid in base_set}
                    st.session_state["pipeline_studio_flow_hidden_ids"] = sorted(hidden_set)
                    base_edges: list[StreamlitFlowEdge] = []
                    for did in base_node_ids:
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
                            elif isinstance(pos_val, (list, tuple)) and len(pos_val) == 2:
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
                    layout_obj = (
                        LayeredLayout(direction="right")
                        if use_auto_layout
                        else ManualLayout()
                    )
                    should_fit_view = use_auto_layout or bool(
                        st.session_state.pop(
                            "pipeline_studio_flow_fit_view_pending", False
                        )
                    )

                    selected_default = None
                    if prev_state is not None and isinstance(prev_state.selected_id, str):
                        selected_default = prev_state.selected_id
                    elif isinstance(selected_node_id, str):
                        selected_default = selected_node_id

                    target_id = (
                        pipe.get("target_dataset_id") if isinstance(pipe, dict) else None
                    )
                    target_id = target_id if isinstance(target_id, str) else None

                    base_nodes: list[StreamlitFlowNode] = []
                    sig_nodes = []
                    for idx, did in enumerate(base_node_ids):
                        meta = meta_by_id.get(did) if isinstance(meta_by_id, dict) else {}
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
                        bits = [x for x in [stage, tk, shape_str, flags_str] if x]
                        content = "\n".join([str(label)] + bits).strip()

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
                                    pos = (ppos[0] + 280.0, ppos[1])
                                    break
                            if pos is None:
                                pos = (0.0, float(idx * 80))
                        sig_nodes.append(
                            {
                                "id": did,
                                "label": str(label),
                                "stage": stage,
                                "transform_kind": tk,
                                "shape": shape_str,
                                "flags": sorted([str(x) for x in flags]),
                                "node_type": node_type,
                                "hidden": did in hidden_set,
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
                                hidden=did in hidden_set,
                                selectable=True,
                                draggable=True,
                                deletable=True,
                                style=_node_style(stage),
                            )
                        )

                    sig_obj = {
                        "nodes": sorted(sig_nodes, key=lambda x: str(x.get("id") or "")),
                        "edges": sorted(
                            [(e.source, e.target) for e in base_edges],
                            key=lambda x: (str(x[0]), str(x[1])),
                        ),
                        "hidden": sorted(hidden_set),
                        "target_id": target_id,
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

                    c_canvas, c_inspect = st.columns([0.72, 0.28], gap="large")
                    with c_canvas:
                        c1, c2, c3 = st.columns([0.22, 0.22, 0.56])
                        with c1:
                            if st.button(
                                "Reset layout",
                                key="pipeline_studio_flow_reset_layout",
                                help="Rebuilds the canvas layout (keeps pipeline data intact).",
                            ):
                                st.session_state.pop("pipeline_studio_flow_state", None)
                                st.session_state.pop("pipeline_studio_flow_positions", None)
                                st.session_state.pop("pipeline_studio_flow_signature", None)
                                if pipeline_hash:
                                    _delete_persisted_pipeline_studio_flow_layout(
                                        pipeline_hash=pipeline_hash
                                    )
                                st.session_state["pipeline_studio_flow_force_layout"] = True
                                st.session_state["pipeline_studio_flow_fit_view_pending"] = True
                                st.session_state[flow_ts_key] = int(time.time() * 1000)
                                st.rerun()
                        with c2:
                            if st.button(
                                "Show all nodes",
                                key="pipeline_studio_flow_show_all",
                                help="Restores any hidden/deleted nodes in the canvas.",
                            ):
                                st.session_state["pipeline_studio_flow_hidden_ids"] = []
                                st.session_state["pipeline_studio_flow_fit_view_pending"] = True
                                st.session_state[flow_ts_key] = int(time.time() * 1000)
                                st.rerun()
                        with c3:
                            st.caption(
                                f"Nodes: {len(base_node_ids)} | Edges: {len(base_edges)} | Hidden: {len(hidden_set)}"
                            )

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
                            hidden_set.update(removed_by_user)
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

                    with c_inspect:
                        sel = (
                            new_state.selected_id
                            if isinstance(new_state.selected_id, str)
                            else None
                        )
                        if not sel:
                            st.info("Click a node to inspect it.")
                        else:
                            st.markdown("**Selected node**")
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
                            open_view = st.selectbox(
                                "Open workspace view",
                                options=available_views,
                                key=open_view_key,
                                help="Jump from the canvas to a standard Pipeline Studio workspace view.",
                            )

                            a, b = st.columns(2)
                            with a:
                                if st.button(
                                    "Open in workspace",
                                    key="pipeline_studio_flow_open_in_workspace",
                                    help="Selects this node in the left rail and turns off auto-follow.",
                                ):
                                    st.session_state["pipeline_studio_node_id_pending"] = sel
                                    st.session_state[
                                        "pipeline_studio_autofollow_pending"
                                    ] = False
                                    st.session_state["pipeline_studio_view_pending"] = open_view
                                    st.rerun()
                            with b:
                                in_hidden = sel in hidden_set
                                if st.button(
                                    "Unhide" if in_hidden else "Hide",
                                    key="pipeline_studio_flow_hide_toggle",
                                    help="Toggles this node visibility in the canvas (does not delete pipeline data).",
                                ):
                                    if in_hidden:
                                        hidden_set.discard(sel)
                                    else:
                                        hidden_set.add(sel)
                                    st.session_state["pipeline_studio_flow_hidden_ids"] = sorted(
                                        hidden_set
                                    )
                                    st.session_state["pipeline_studio_flow_fit_view_pending"] = True
                                    st.rerun()

                            with st.expander("Metadata (debug)", expanded=False):
                                st.json(meta)

with tab_pipeline_studio:
    st.subheader("Pipeline Studio")
    try:
        _render_pipeline_studio()
    except Exception as e:
        st.error(f"Could not render Pipeline Studio: {e}")

with tab_chat:
    st.markdown("---")
    st.subheader("Analysis Details")
    if st.session_state.get("details"):
        details = st.session_state.details
        default_idx = len(details) - 1
        selected = st.selectbox(
            "Inspect a prior turn",
            options=list(range(len(details))),
            index=default_idx,
            format_func=lambda i: f"Turn {i + 1}",
            key="analysis_details_turn_select",
        )
        try:
            # Reuse the same rendering logic as chat history by calling render_history's helper pattern.
            # Minimal duplication: render via a small inline function to avoid leaking outer scope.
            detail = details[int(selected)]
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
                    st.write(txt if txt else "No reasoning available.")

            with tabs[1]:
                pipelines = (
                    detail.get("pipelines") if isinstance(detail, dict) else None
                )
                if not isinstance(pipelines, dict):
                    pipelines = {}
                pipe = detail.get("pipeline") if isinstance(detail, dict) else None
                if pipelines:
                    options = [
                        ("Model (latest feature)", "model"),
                        ("Active dataset", "active"),
                        ("Latest dataset", "latest"),
                    ]
                    target = st.radio(
                        "Pipeline target",
                        options=[k for k, _v in options],
                        index=0,
                        horizontal=True,
                        key=f"bottom_pipeline_target_{selected}",
                    )
                    target_key = dict(options).get(target, "model")
                    if target_key == "model":
                        pipe = detail.get("pipeline") or pipelines.get("model") or pipe
                    else:
                        pipe = pipelines.get(target_key) or pipe

                if isinstance(pipe, dict) and pipe.get("lineage"):
                    inputs = pipe.get("inputs") or []
                    inputs_txt = ""
                    if isinstance(inputs, list) and inputs:
                        inputs_txt = f"  \n**Inputs:** {', '.join([f'`{i}`' for i in inputs if i])}"
                    st.markdown(
                        f"**Pipeline hash:** `{pipe.get('pipeline_hash')}`  \n"
                        f"**Target dataset id:** `{pipe.get('target_dataset_id')}`  \n"
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
                            key=f"bottom_download_pipeline_spec_{selected}",
                        )
                    except Exception:
                        pass
                    if isinstance(script, str) and script.strip():
                        st.download_button(
                            "Download pipeline script",
                            data=script.encode("utf-8"),
                            file_name=f"pipeline_repro_{pipe.get('target') or 'model'}.py",
                            mime="text/x-python",
                            key=f"bottom_download_pipeline_{selected}",
                        )
                        st.code(script, language="python")

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
                            with st.expander(
                                "Feature engineering code", expanded=False
                            ):
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
        finally:
            pass
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
                        key=f"bottom_download_sql_query_{selected}",
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
                        key=f"bottom_download_sql_executor_{selected}",
                    )
                except Exception:
                    pass
        with tabs[4]:
            graph_json = detail.get("plotly_graph")
            if graph_json:
                payload = (
                    json.dumps(graph_json)
                    if isinstance(graph_json, dict)
                    else graph_json
                )
                fig = _apply_streamlit_plot_style(pio.from_json(payload))
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"bottom_detail_chart_{selected}",
                )
            else:
                st.info("No charts returned.")
        with tabs[5]:
            reports = detail.get("eda_reports") if isinstance(detail, dict) else None
            sweetviz_file = (
                reports.get("sweetviz_report_file")
                if isinstance(reports, dict)
                else None
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
                        key=f"bottom_download_sweetviz_{selected}",
                    )
                except Exception as e:
                    st.warning(f"Could not render Sweetviz report: {e}")

            if dtale_url:
                st.markdown("**D-Tale**")
                st.markdown(f"[Open D-Tale]({dtale_url})")

            if not sweetviz_file and not dtale_url:
                st.info("No EDA reports returned.")

        with tabs[6]:
            model_info = detail.get("model_info")
            eval_art = detail.get("eval_artifacts")
            eval_graph = detail.get("eval_plotly_graph")
            if model_info is not None:
                st.markdown("**Model Info**")
                try:
                    if isinstance(model_info, dict):
                        st.dataframe(
                            pd.DataFrame(model_info), use_container_width=True
                        )
                    elif isinstance(model_info, list):
                        st.dataframe(
                            pd.DataFrame(model_info), use_container_width=True
                        )
                    else:
                        st.json(model_info)
                except Exception:
                    st.json(model_info)
            if eval_art is not None:
                st.markdown("**Evaluation**")
                st.json(eval_art)
            if eval_graph:
                payload = (
                    json.dumps(eval_graph)
                    if isinstance(eval_graph, dict)
                    else eval_graph
                )
                fig = _apply_streamlit_plot_style(pio.from_json(payload))
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"bottom_eval_chart_{selected}",
                )
            if model_info is None and eval_art is None and eval_graph is None:
                st.info("No model or evaluation artifacts.")

        with tabs[7]:
            preds_df = detail.get("data_wrangled_df")
            if isinstance(preds_df, pd.DataFrame) and not preds_df.empty:
                lower_cols = {str(c).lower() for c in preds_df.columns}
                looks_like_preds = (
                    "predict" in lower_cols
                    or any(str(c).lower().startswith("p") for c in preds_df.columns)
                    or any(
                        str(c).lower().startswith("actual_") for c in preds_df.columns
                    )
                )
                if looks_like_preds:
                    st.markdown("**Predictions Preview**")
                    st.dataframe(preds_df)
                else:
                    st.info("No predictions detected for this turn.")
            else:
                st.info("No predictions returned.")

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
                                use_container_width=True,
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
                                use_container_width=True,
                            )
                            return
                        if isinstance(obj, list):
                            st.dataframe(pd.DataFrame(obj), use_container_width=True)
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
    # Note: analysis details rendering is best-effort; errors should not break the app.
