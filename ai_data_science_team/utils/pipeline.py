from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def strip_markdown_code_fences(code: str) -> str:
    """
    Remove ```python ... ``` or ``` ... ``` fences if present.
    """
    if not isinstance(code, str) or not code:
        return ""
    text = code.strip()
    if text.startswith("```"):
        # Remove first fence line
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        # Remove trailing fence
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def _as_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


def pick_latest_dataset_id(datasets: Dict[str, Any], *, stage: str) -> Optional[str]:
    """
    Pick the newest dataset id for a given stage by created_ts.
    """
    best_id: Optional[str] = None
    best_ts: float = -1.0
    for did, entry in (datasets or {}).items():
        if not isinstance(entry, dict):
            continue
        if entry.get("stage") != stage:
            continue
        ts = _as_float(entry.get("created_ts"), 0.0)
        if ts >= best_ts:
            best_ts = ts
            best_id = did
    return best_id


def pick_latest_dataset_id_any_stage(datasets: Dict[str, Any]) -> Optional[str]:
    """
    Pick the newest dataset id across all stages by created_ts.
    """
    best_id: Optional[str] = None
    best_ts: float = -1.0
    for did, entry in (datasets or {}).items():
        if not isinstance(entry, dict):
            continue
        ts = _as_float(entry.get("created_ts"), 0.0)
        if ts >= best_ts:
            best_ts = ts
            best_id = did
    return best_id


def build_dataset_lineage_ids(datasets: Dict[str, Any], target_dataset_id: str) -> List[str]:
    """
    Walk parent_id links from target -> root, returning ids in root->target order.
    """
    if not isinstance(datasets, dict) or not target_dataset_id:
        return []
    lineage_rev: List[str] = []
    current: Optional[str] = target_dataset_id
    visited: set[str] = set()
    while current and current not in visited:
        visited.add(current)
        entry = datasets.get(current)
        if not isinstance(entry, dict):
            break
        lineage_rev.append(current)
        parent = entry.get("parent_id")
        current = parent if isinstance(parent, str) and parent else None
    return list(reversed(lineage_rev))


def compute_pipeline_hash(datasets: Dict[str, Any], lineage_ids: List[str]) -> Optional[str]:
    """
    Compute a stable-ish pipeline hash from source + transform hashes across lineage.
    """
    if not lineage_ids or not isinstance(datasets, dict):
        return None
    import hashlib
    import json

    items: List[Dict[str, Any]] = []
    for idx, did in enumerate(lineage_ids):
        entry = datasets.get(did)
        if not isinstance(entry, dict):
            continue
        prov = entry.get("provenance") if isinstance(entry.get("provenance"), dict) else {}
        transform = prov.get("transform") if isinstance(prov.get("transform"), dict) else {}
        step = {
            "id": did,
            "stage": entry.get("stage"),
            "label": entry.get("label"),
            "schema_hash": entry.get("schema_hash"),
            "fingerprint": entry.get("fingerprint"),
        }
        if idx == 0:
            step["source_type"] = prov.get("source_type")
            step["source"] = prov.get("source")
        if transform:
            step["transform_kind"] = transform.get("kind")
            step["code_sha256"] = transform.get("code_sha256") or transform.get("sql_sha256")
            step["sql_sha256"] = transform.get("sql_sha256")
        items.append(step)

    payload = json.dumps(items, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _infer_function_name(code: str) -> Optional[str]:
    import re

    if not isinstance(code, str) or not code:
        return None
    m = re.search(r"^\\s*def\\s+([a-zA-Z_][a-zA-Z0-9_]*)\\s*\\(", code, flags=re.MULTILINE)
    return m.group(1) if m else None


def _read_text_file(path: Any, *, max_bytes: int = 500_000) -> Optional[str]:
    try:
        import os

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


def build_reproducible_pipeline_script(
    datasets: Dict[str, Any],
    *,
    target_dataset_id: str,
) -> str:
    """
    Generate a best-effort python script that replays the lineage from root -> target.

    Notes:
    - If the root dataset was loaded from a file, uses pandas readers.
    - If the root dataset came from SQL, emits a SQLAlchemy placeholder.
    - For transformation stages, embeds the recorded python function code (if present) and calls it.
    """
    lineage_ids = build_dataset_lineage_ids(datasets, target_dataset_id)
    if not lineage_ids:
        return ""

    lines: List[str] = []
    lines.append("# Auto-generated pipeline script (best effort).")
    lines.append("# Edit file paths / connection strings as needed.")
    lines.append(f"# Target dataset id: {target_dataset_id}")
    lines.append("")
    lines.append("import pandas as pd")
    lines.append("")
    lines.append("df = None")
    lines.append("")

    for idx, did in enumerate(lineage_ids):
        entry = datasets.get(did)
        if not isinstance(entry, dict):
            continue

        stage = str(entry.get("stage") or "")
        label = str(entry.get("label") or did)
        prov = entry.get("provenance") if isinstance(entry.get("provenance"), dict) else {}
        transform = prov.get("transform") if isinstance(prov.get("transform"), dict) else {}

        lines.append(f"# Step {idx + 1}: {stage or 'dataset'} — {label} ({did})")
        if entry.get("schema_hash"):
            lines.append(f"#   schema_hash: {entry.get('schema_hash')}")
        if entry.get("fingerprint"):
            lines.append(f"#   fingerprint: {entry.get('fingerprint')}")
        if isinstance(transform, dict) and transform.get("kind"):
            lines.append(f"#   transform: {transform.get('kind')}")
            if transform.get("code_sha256"):
                lines.append(f"#   code_sha256: {transform.get('code_sha256')}")
            if transform.get("sql_sha256"):
                lines.append(f"#   sql_sha256: {transform.get('sql_sha256')}")

        if idx == 0:
            source_type = str(prov.get("source_type") or "")
            source = prov.get("source")
            if source_type == "file" and isinstance(source, str) and source:
                s = source.lower()
                if s.endswith(".csv") or s.endswith(".csv.gz"):
                    lines.append(f"df = pd.read_csv({source!r})")
                elif s.endswith(".tsv") or s.endswith(".tsv.gz"):
                    lines.append(f"df = pd.read_csv({source!r}, sep='\\t')")
                elif s.endswith(".parquet"):
                    lines.append(f"df = pd.read_parquet({source!r})")
                elif s.endswith(".json") or s.endswith(".jsonl") or s.endswith(".ndjson"):
                    lines.append(f"df = pd.read_json({source!r})")
                elif s.endswith(".xlsx") or s.endswith(".xls"):
                    lines.append(f"df = pd.read_excel({source!r})")
                else:
                    lines.append(f"# TODO: add reader for: {source!r}")
                    lines.append(f"df = pd.read_csv({source!r})")
            elif stage == "sql" and isinstance(transform, dict) and transform.get("sql_query_code"):
                sql_query = str(transform.get("sql_query_code") or "")
                lines.append("import sqlalchemy as sql")
                lines.append("engine = sql.create_engine('YOUR_SQLALCHEMY_URL')")
                lines.append(f"sql_query = {sql_query!r}")
                lines.append("df = pd.read_sql_query(sql_query, engine)")
            else:
                lines.append(
                    "# TODO: root dataset source not recorded as a file/sql; provide your own df here."
                )
                lines.append("df = pd.DataFrame()")
        else:
            kind = str(transform.get("kind") or "")
            if kind == "python_function":
                code = ""
                file_code = _read_text_file(transform.get("function_path"))
                if isinstance(file_code, str) and file_code.strip():
                    code = strip_markdown_code_fences(file_code)
                else:
                    code = strip_markdown_code_fences(str(transform.get("function_code") or ""))
                fn_name = transform.get("function_name")
                fn_name = str(fn_name) if isinstance(fn_name, str) and fn_name else _infer_function_name(code)
                if code and fn_name:
                    lines.append("")
                    lines.append(code)
                    lines.append("")
                    lines.append(f"df = {fn_name}(df)")
                else:
                    lines.append(
                        "# TODO: missing function code/name for this step; see datasets provenance."
                    )
            elif kind == "sql_query" and transform.get("sql_query_code"):
                sql_query = str(transform.get("sql_query_code") or "")
                lines.append("import sqlalchemy as sql")
                lines.append("engine = sql.create_engine('YOUR_SQLALCHEMY_URL')")
                lines.append(f"sql_query = {sql_query!r}")
                lines.append("df = pd.read_sql_query(sql_query, engine)")
            else:
                lines.append(
                    "# TODO: transform not recorded in a runnable form; see datasets provenance."
                )

        lines.append("")

    lines.append("# Final output")
    lines.append("print('Final shape:', getattr(df, 'shape', None))")
    lines.append("# df.to_csv('final_dataset.csv', index=False)")
    return "\n".join(lines).strip() + "\n"


def build_pipeline_snapshot(
    datasets: Dict[str, Any],
    *,
    active_dataset_id: Optional[str],
    target: str = "model",
) -> Dict[str, Any]:
    """
    Build a lightweight pipeline snapshot for display:
    - Computes the latest modeling dataset as the newest 'feature' dataset if present; else uses active_dataset_id.
    - `target` controls which dataset the lineage/script is built for:
        - "model": the modeling dataset (default)
        - "active": the active dataset
        - "latest": the newest dataset across all stages by created_ts
    - Returns lineage metadata and an exportable script.
    """
    datasets = datasets if isinstance(datasets, dict) else {}
    model_dataset_id = pick_latest_dataset_id(datasets, stage="feature") or active_dataset_id

    target = (target or "model").strip().lower()
    if target == "active":
        target_dataset_id = active_dataset_id
    elif target == "latest":
        target_dataset_id = pick_latest_dataset_id_any_stage(datasets) or model_dataset_id or active_dataset_id
    else:
        target = "model"
        target_dataset_id = model_dataset_id

    lineage_ids = (
        build_dataset_lineage_ids(datasets, target_dataset_id)
        if isinstance(target_dataset_id, str) and target_dataset_id
        else []
    )
    pipeline_hash = compute_pipeline_hash(datasets, lineage_ids) if lineage_ids else None

    def _entry_meta(did: str) -> Dict[str, Any]:
        e = datasets.get(did)
        if not isinstance(e, dict):
            return {"id": did}
        prov = e.get("provenance") if isinstance(e.get("provenance"), dict) else {}
        transform = prov.get("transform") if isinstance(prov.get("transform"), dict) else {}
        return {
            "id": did,
            "label": e.get("label"),
            "stage": e.get("stage"),
            "shape": e.get("shape"),
            "schema_hash": e.get("schema_hash"),
            "fingerprint": e.get("fingerprint"),
            "source": prov.get("source") if isinstance(prov, dict) else None,
            "transform_kind": transform.get("kind") if isinstance(transform, dict) else None,
            "transform_hash": (
                transform.get("code_sha256")
                or transform.get("sql_sha256")
                or transform.get("sql_database_function_sha256")
            )
            if isinstance(transform, dict)
            else None,
            "created_at": e.get("created_at"),
            "created_by": e.get("created_by"),
        }

    lineage = [_entry_meta(did) for did in lineage_ids]
    script = (
        build_reproducible_pipeline_script(datasets, target_dataset_id=target_dataset_id)
        if isinstance(target_dataset_id, str) and target_dataset_id
        else ""
    )

    return {
        "pipeline_hash": pipeline_hash,
        "active_dataset_id": active_dataset_id,
        "model_dataset_id": model_dataset_id,
        "target": target,
        "target_dataset_id": target_dataset_id,
        "lineage": lineage,
        "script": script,
    }
