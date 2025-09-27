from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import math
import numpy as np
from uuid import uuid4
import os
import sys
import io
import pandas as pd

from langchain_openai import ChatOpenAI
from openai import OpenAI

from ai_data_science_team.ds_agents import EDAToolsAgent
from ai_data_science_team.utils.matplotlib import matplotlib_from_base64
from ai_data_science_team.utils.plotly import plotly_from_dict

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")

app = FastAPI(title="EDA Copilot API", version="0.1.0")
def _json_safe(value: Any) -> Any:
    """Recursively convert value into JSON-safe types.

    - NaN/Inf -> None
    - numpy scalars -> Python scalars
    - numpy arrays -> lists
    - pandas NA/NaT -> None
    - dict/list/tuple/set -> recurse
    """
    try:
        # pandas NA detection
        import pandas as _pd
        if _pd.isna(value):
            return None
    except Exception:
        pass

    if value is None:
        return None

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.ndarray,)):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return value


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class SessionCreateResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: str
    question: str
    api_key: Optional[str] = None


class DataframePayload(BaseModel):
    columns: List[str]
    rows: List[Dict[str, Any]]


class ChatResponse(BaseModel):
    ai_message: str
    tool: Optional[str] = None
    plotly_figure: Optional[Dict[str, Any]] = None
    dataframe: Optional[DataframePayload] = None
    report_url: Optional[str] = None
    dtale_url: Optional[str] = None
    missing_matrix_url: Optional[str] = None
    missing_bar_url: Optional[str] = None
    missing_heatmap_url: Optional[str] = None


# In-memory session store. For production, replace with Redis or DB.
SESSIONS: Dict[str, Dict[str, Any]] = {}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/session", response_model=SessionCreateResponse)
def create_session():
    session_id = uuid4().hex
    SESSIONS[session_id] = {"data": None}
    return SessionCreateResponse(session_id=session_id)


@app.post("/api/upload")
async def upload_file(session_id: str = Form(...), file: UploadFile = File(...)):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="session not found")

    content = await file.read()
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise ValueError("unsupported file type")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to parse file: {e}")

    SESSIONS[session_id]["data"] = df
    # 清洗 NaN/Inf 以避免 JSON 序列化失败
    preview_df = (
        df.head(10)
        .replace([pd.NA, pd.NaT, float('inf'), float('-inf')], None)
        .where(pd.notna(df.head(10)), None)
    )
    return _json_safe({
        "status": "ok",
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "preview": {
            "columns": [str(c) for c in preview_df.columns],
            "rows": preview_df.to_dict(orient="records"),
        },
    })


@app.post("/api/demo-data")
def load_demo_data(session_id: str = Form(...), name: str = Form("churn")):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="session not found")
    # Map demo names to local files
    demo_map = {
        "churn": os.path.join(ROOT_DIR if 'ROOT_DIR' in globals() else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")), "data", "churn_data.csv"),
    }
    path = demo_map.get(name)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"demo data not found: {name}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to load demo: {e}")
    SESSIONS[session_id]["data"] = df
    # 清洗 NaN/Inf 以避免 JSON 序列化失败
    preview_df = (
        df.head(10)
        .replace([pd.NA, pd.NaT, float('inf'), float('-inf')], None)
        .where(pd.notna(df.head(10)), None)
    )
    return _json_safe({
        "status": "ok",
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "preview": {
            "columns": [str(c) for c in preview_df.columns],
            "rows": preview_df.to_dict(orient="records"),
        },
    })


@app.post("/api/validate-key")
def validate_key(api_key: str = Form(...)):
    try:
        client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        _ = client.models.list()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid key: {e}")


def _get_llm(override_key: Optional[str] = None) -> ChatOpenAI:
    key = override_key or API_KEY
    if not key:
        raise HTTPException(status_code=400, detail="missing OPENAI_API_KEY/DEEPSEEK_API_KEY")
    return ChatOpenAI(model=DEEPSEEK_MODEL, api_key=key, base_url=DEEPSEEK_BASE_URL)


def _df_to_payload(df: pd.DataFrame, limit: int = 200) -> DataframePayload:
    df_limited = df.head(limit).replace([pd.NA, pd.NaT, float('inf'), float('-inf')], None)
    # 将 numpy.nan 转为 None，确保 JSON 可序列化
    df_limited = df_limited.where(pd.notna(df_limited), None)
    return DataframePayload(
        columns=[str(c) for c in df_limited.columns],
        rows=df_limited.to_dict(orient="records"),
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    session = SESSIONS.get(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    df: Optional[pd.DataFrame] = session.get("data")
    if df is None:
        raise HTTPException(status_code=400, detail="no data uploaded for this session")

    llm = _get_llm(override_key=body.api_key)
    eda_agent = EDAToolsAgent(llm, invoke_react_agent_kwargs={"recursion_limit": 10})

    question = body.question.strip() + " Don't return hyperlinks to files in the response."
    eda_agent.invoke_agent(user_instructions=question, data_raw=df)

    tool_calls = eda_agent.get_tool_calls() or []
    ai_message = eda_agent.get_ai_message(markdown=False) or ""
    artifacts = eda_agent.get_artifacts(as_dataframe=False) or {}

    tool_name = tool_calls[-1] if tool_calls else None

    resp: Dict[str, Any] = {
        "ai_message": ai_message,
        "tool": tool_name,
    }

    # Common artifact mappings
    if isinstance(artifacts, dict):
        # describe_dataset -> describe_df
        if "describe_df" in artifacts:
            try:
                resp["dataframe"] = _df_to_payload(pd.DataFrame(artifacts["describe_df"]))
            except Exception:
                pass

        # correlation funnel
        if "correlation_data" in artifacts:
            try:
                resp["dataframe"] = _df_to_payload(pd.DataFrame(artifacts["correlation_data"]))
            except Exception:
                pass

        # generic dataframe
        if "dataframe" in artifacts and resp.get("dataframe") is None:
            try:
                resp["dataframe"] = _df_to_payload(pd.DataFrame(artifacts["dataframe"]))
            except Exception:
                pass

        # plotly figure
        if "plotly_figure" in artifacts:
            resp["plotly_figure"] = artifacts["plotly_figure"]

        # visualize_missing: base64 images -> static files
        def _save_b64_image(b64_str: Optional[str], filename: str) -> Optional[str]:
            if not b64_str:
                return None
            try:
                import base64
                from pathlib import Path
                sess_dir = os.path.join(STATIC_DIR, body.session_id)
                os.makedirs(sess_dir, exist_ok=True)
                out_path = os.path.join(sess_dir, filename)
                data = base64.b64decode(b64_str)
                with open(out_path, "wb") as f:
                    f.write(data)
                return f"/static/{body.session_id}/{filename}"
            except Exception:
                return None

        if "matrix_plot" in artifacts or "bar_plot" in artifacts or "heatmap_plot" in artifacts:
            resp["missing_matrix_url"] = _save_b64_image(artifacts.get("matrix_plot"), "missing_matrix.png")
            resp["missing_bar_url"] = _save_b64_image(artifacts.get("bar_plot"), "missing_bar.png")
            resp["missing_heatmap_url"] = _save_b64_image(artifacts.get("heatmap_plot"), "missing_heatmap.png")

        # sweetviz report
        if "report_html" in artifacts:
            # save to static/session_id/sweetviz_report.html
            sess_dir = os.path.join(STATIC_DIR, body.session_id)
            os.makedirs(sess_dir, exist_ok=True)
            report_path = os.path.join(sess_dir, "sweetviz_report.html")
            try:
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(artifacts["report_html"])
                resp["report_url"] = f"/static/{body.session_id}/sweetviz_report.html"
            except Exception:
                pass

        # dtale url passthrough
        if "dtale_url" in artifacts:
            resp["dtale_url"] = artifacts["dtale_url"]

    return _json_safe(resp)


