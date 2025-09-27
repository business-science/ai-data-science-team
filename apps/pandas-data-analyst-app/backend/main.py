from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from uuid import uuid4
import os
import io
import math
import numpy as np
import pandas as pd

from langchain_openai import ChatOpenAI
from openai import OpenAI

from ai_data_science_team.agents import DataWranglingAgent, DataVisualizationAgent
from ai_data_science_team.multiagents.pandas_data_analyst import PandasDataAnalyst


DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")

app = FastAPI(title="Pandas Data Analyst API", version="0.1.0")

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


# In-memory session store. For production, replace with Redis or DB.
SESSIONS: Dict[str, Dict[str, Any]] = {}


@app.get("/health")
def health_check():
    return {"status": "ok"}


def _json_safe(value: Any) -> Any:
    try:
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


def _get_llm(override_key: Optional[str] = None) -> ChatOpenAI:
    key = override_key or API_KEY
    if not key:
        raise HTTPException(status_code=400, detail="missing OPENAI_API_KEY/DEEPSEEK_API_KEY")
    return ChatOpenAI(model=DEEPSEEK_MODEL, api_key=key, base_url=DEEPSEEK_BASE_URL)


def _df_to_payload(df: pd.DataFrame, limit: int = 200) -> DataframePayload:
    df_limited = (
        df.head(limit)
        .replace([pd.NA, pd.NaT, float("inf"), float("-inf")], None)
        .where(pd.notna(df.head(limit)), None)
    )
    return DataframePayload(
        columns=[str(c) for c in df_limited.columns],
        rows=df_limited.to_dict(orient="records"),
    )


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
    preview_df = (
        df.head(10)
        .replace([pd.NA, pd.NaT, float("inf"), float("-inf")], None)
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
def load_demo_data(session_id: str = Form(...), name: str = Form("bikes")):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="session not found")
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    demo_map = {
        "bikes": os.path.join(root, "data", "bike_sales_data.csv"),
        "churn": os.path.join(root, "data", "churn_data.csv"),
    }
    path = demo_map.get(name)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"demo data not found: {name}")
    try:
        df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to load demo: {e}")
    SESSIONS[session_id]["data"] = df
    preview_df = (
        df.head(10)
        .replace([pd.NA, pd.NaT, float("inf"), float("-inf")], None)
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


@app.post("/api/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    session = SESSIONS.get(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    df: Optional[pd.DataFrame] = session.get("data")
    if df is None:
        raise HTTPException(status_code=400, detail="no data uploaded for this session")

    llm = _get_llm(override_key=body.api_key)
    wrangler = DataWranglingAgent(model=llm, bypass_recommended_steps=True, n_samples=100)
    viz = DataVisualizationAgent(model=llm, n_samples=100)
    pda = PandasDataAnalyst(model=llm, data_wrangling_agent=wrangler, data_visualization_agent=viz)

    question = body.question.strip()
    pda.invoke_agent(user_instructions=question, data_raw=df)
    result = pda.response or {}

    resp: Dict[str, Any] = {
        "ai_message": "",
        "tool": result.get("routing_preprocessor_decision"),
    }

    if result.get("plotly_graph") and not result.get("plotly_error"):
        resp["plotly_figure"] = result.get("plotly_graph")

    if result.get("data_wrangled") is not None:
        try:
            resp["dataframe"] = _df_to_payload(pd.DataFrame(result.get("data_wrangled")))
        except Exception:
            pass

    # Optional: include a compact text summary
    try:
        summary = pda.get_workflow_summary(markdown=False)
        if isinstance(summary, str):
            resp["ai_message"] = summary
    except Exception:
        pass

    return _json_safe(resp)


