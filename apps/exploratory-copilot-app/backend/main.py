from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from uuid import uuid4
import os
import io
import pandas as pd

from langchain_openai import ChatOpenAI
from ai_data_science_team.ds_agents import EDAToolsAgent

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")

app = FastAPI(title="EDA Copilot API", version="0.1.0")

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
    return {"status": "ok", "rows": int(df.shape[0]), "cols": int(df.shape[1])}


def _get_llm() -> ChatOpenAI:
    if not API_KEY:
        raise HTTPException(status_code=400, detail="missing OPENAI_API_KEY/DEEPSEEK_API_KEY")
    return ChatOpenAI(model=DEEPSEEK_MODEL, api_key=API_KEY, base_url=DEEPSEEK_BASE_URL)


def _df_to_payload(df: pd.DataFrame, limit: int = 200) -> DataframePayload:
    df_limited = df.head(limit)
    return DataframePayload(columns=[str(c) for c in df_limited.columns], rows=df_limited.to_dict(orient="records"))


@app.post("/api/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    session = SESSIONS.get(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    df: Optional[pd.DataFrame] = session.get("data")
    if df is None:
        raise HTTPException(status_code=400, detail="no data uploaded for this session")

    llm = _get_llm()
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

    return resp


