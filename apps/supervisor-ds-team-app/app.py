"""
Streamlit app for the Supervisor-led Data Science Team.

Command:
    streamlit run apps/supervisor-ds-team-app/app.py
"""

from __future__ import annotations

import uuid
import os
import json
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


st.set_page_config(
    page_title="Supervisor Data Science Team", page_icon=":bar_chart:", layout="wide"
)
TITLE = "Supervisor-led Data Science Team"
st.title(TITLE)

UI_DETAIL_MARKER_PREFIX = "DETAILS_INDEX:"


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
        - Load `data/churn_data.csv` and show the first 5 rows.
        - What files are in `./data`? List only CSVs.

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
        - Show the tables in the connected database (do not call other agents).
        - Write SQL to count customers by Contract type and execute it.      

        #### Feature engineering
        - Create model-ready features for churn (encode categoricals, handle totals/averages).
        
        #### Machine Learning / MLflow
        - Train an H2O AutoML classifier on Churn with max runtime 30 seconds and report leaderboard.
        - Log best model to MLflow and show run info.
        """
    )


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Enter your OpenAI API Key")
    # Keep API key in session_state
    st.session_state["OPENAI_API_KEY"] = st.text_input(
        "API Key",
        type="password",
        help="Your OpenAI API key is required for the app to function.",
    )

    key_status = None
    if st.session_state["OPENAI_API_KEY"]:
        try:
            _ = OpenAI(api_key=st.session_state["OPENAI_API_KEY"]).models.list()
            key_status = "ok"
            st.success("API Key is valid!")
        except Exception as e:
            key_status = "bad"
            st.error(f"Invalid API Key: {e}")
    else:
        st.info("Please enter your OpenAI API Key to proceed.")
        st.stop()

    # Settings
    st.header("Settings")
    model_choice = st.selectbox(
        "OpenAI model",
        [
            "gpt-4.1-mini",
            "gpt-4.1",
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-5.2",
            "gpt-5.1-mini",
            "gpt-5.1",
        ],
    )
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
    current_active_id = current_active_id if isinstance(current_active_id, str) else None
    current_active_key = team_state.get("active_data_key")

    if current_active_id and current_active_id in datasets and isinstance(datasets[current_active_id], dict):
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
    else:
        st.session_state["active_dataset_id_override"] = ""
        st.selectbox(
            "Active dataset (override)",
            options=[""],
            format_func=lambda _k: "Auto (load data to populate datasets)",
            disabled=True,
            key="active_dataset_id_override",
        )
    st.markdown("**SQL options**")
    sql_url = st.text_input("SQLAlchemy URL (optional)", value="sqlite:///:memory:")
    st.session_state["sql_url"] = sql_url

    st.markdown("**MLflow options**")
    enable_mlflow_logging = st.checkbox("Enable MLflow logging in training", value=True)
    default_mlflow_uri = f"file:{os.path.abspath('mlruns')}"
    mlflow_tracking_uri = st.text_input(
        "MLflow tracking URI", value=default_mlflow_uri
    ).strip()
    mlflow_experiment_name = st.text_input(
        "MLflow experiment name", value="H2O AutoML"
    ).strip()
    st.session_state["enable_mlflow_logging"] = enable_mlflow_logging
    st.session_state["mlflow_tracking_uri"] = mlflow_tracking_uri or None
    st.session_state["mlflow_experiment_name"] = mlflow_experiment_name or "H2O AutoML"

    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.session_state.details = []
        st.session_state.team_state = {}
        st.session_state["active_dataset_id_override"] = ""
        msgs = StreamlitChatMessageHistory(key="supervisor_ds_msgs")
        msgs.clear()
        msgs.add_ai_message("How can the data science team help today?")
        st.session_state.thread_id = str(uuid.uuid4())
        # Reset checkpointer when clearing chat
        st.session_state.checkpointer = get_checkpointer() if add_memory else None

# Hard gate: require valid API key before rendering the rest of the app
resolved_api_key = st.session_state.get("OPENAI_API_KEY")
if not resolved_api_key:
    st.info("Please enter your OpenAI API key in the sidebar to proceed.")
    st.stop()
if key_status == "bad":
    st.error("Invalid OpenAI API key. Please fix it in the sidebar.")
    st.stop()


def build_team(
    model_name: str,
    use_memory: bool,
    sql_url: str,
    checkpointer,
    enable_mlflow_logging: bool,
    mlflow_tracking_uri: str | None,
    mlflow_experiment_name: str,
):
    llm = ChatOpenAI(model=model_name)
    workflow_planner_agent = WorkflowPlannerAgent(llm)
    data_loader_agent = DataLoaderToolsAgent(
        llm, invoke_react_agent_kwargs={"recursion_limit": 4}
    )
    data_wrangling_agent = DataWranglingAgent(llm, log=False)
    data_cleaning_agent = DataCleaningAgent(llm, log=False)
    eda_tools_agent = EDAToolsAgent(llm, log_tool_calls=True)
    data_visualization_agent = DataVisualizationAgent(llm, log=False)
    # SQL connection is optional; default to in-memory sqlite to satisfy constructor.
    # Use check_same_thread=False so the connection can be reused safely in Streamlit threads.
    conn = sql.create_engine(
        sql_url or "sqlite:///:memory:", connect_args={"check_same_thread": False}
    ).connect()
    sql_database_agent = SQLDatabaseAgent(llm, connection=conn, log=False)
    feature_engineering_agent = FeatureEngineeringAgent(llm, log=False)
    h2o_ml_agent = H2OMLAgent(
        llm,
        log=False,
        enable_mlflow=enable_mlflow_logging,
        mlflow_tracking_uri=mlflow_tracking_uri,
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

msgs = StreamlitChatMessageHistory(key="supervisor_ds_msgs")
if not msgs.messages:
    msgs.add_ai_message("How can the data science team help today?")


def get_input_data():
    """
    Resolve data_raw based on user selections: uploaded CSV or sample dataset.
    Returns tuple (data_raw_dict, preview_df_or_None).
    """
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
    elif use_sample:
        sample_path = os.path.join("data", "churn_data.csv")
        if os.path.exists(sample_path):
            try:
                df = pd.read_csv(sample_path)
            except Exception as e:
                st.error(f"Error loading sample data: {e}")
        else:
            st.warning(f"Sample file not found at {sample_path}")

    if df is not None:
        st.markdown("**Data preview**")
        st.dataframe(df.head(preview_rows))
        return df.to_dict(), df.head(preview_rows)

    return None, None


def render_history(history: list[BaseMessage]):
    def _render_analysis_detail(detail: dict, key_suffix: str):
        tabs = st.tabs(
            [
                "AI Reasoning",
                "Data (raw/sql/wrangle/clean/features)",
                "Charts",
                "Reports",
                "Models/MLflow",
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
        # Data
        with tabs[1]:
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
        # Charts
        with tabs[2]:
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
                        width="stretch",
                        key=f"detail_chart_{key_suffix}",
                    )
                except Exception as e:
                    st.error(f"Error rendering chart: {e}")
            else:
                st.info("No charts returned.")
        # Reports
        with tabs[3]:
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

        # Models / MLflow
        with tabs[4]:
            model_info = detail.get("model_info")
            eval_art = detail.get("eval_artifacts")
            eval_graph = detail.get("eval_plotly_graph")
            mlflow_art = detail.get("mlflow_artifacts")
            if model_info is not None:
                st.markdown("**Model Info**")
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
                        fig, width="stretch", key=f"eval_chart_{key_suffix}"
                    )
                except Exception as e:
                    st.error(f"Error rendering evaluation chart: {e}")
            if mlflow_art is not None:
                st.markdown("**MLflow Artifacts**")
                st.json(mlflow_art)
            if model_info is None and eval_art is None and mlflow_art is None:
                st.info("No model/evaluation/MLflow artifacts.")

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


render_history(msgs.messages)

# Show data preview (if selected) and store for reuse on submit
data_raw_dict, _ = get_input_data()
# If no new data selected, reuse previously loaded data_raw from session
if data_raw_dict is None:
    data_raw_dict = st.session_state.get("selected_data_raw")
st.session_state.selected_data_raw = data_raw_dict

# ---------------- User input ----------------
prompt = st.chat_input("Ask the data science team...")
if prompt:
    if not resolved_api_key or key_status == "bad":
        st.error(
            "OpenAI API key is required and must be valid. Enter it in the sidebar."
        )
        st.stop()

    st.chat_message("human").write(prompt)
    msgs.add_user_message(prompt)

    data_raw_dict = st.session_state.get("selected_data_raw")

    team = build_team(
        model_choice,
        add_memory,
        st.session_state.get("sql_url", "sqlite:///:memory:"),
        st.session_state.checkpointer if add_memory else None,
        st.session_state.get("enable_mlflow_logging", True),
        st.session_state.get("mlflow_tracking_uri"),
        st.session_state.get("mlflow_experiment_name", "H2O AutoML"),
    )
    try:
        # If LangGraph memory is enabled, pass only the new user message.
        # The checkpointer will supply prior state/messages for continuity.
        input_messages = (
            [HumanMessage(content=prompt, id=str(uuid.uuid4()))]
            if add_memory
            else _strip_ui_marker_messages(msgs.messages)
        )
        active_dataset_override = st.session_state.get("active_dataset_id_override") or None
        persisted = st.session_state.get("team_state", {})
        persisted = persisted if isinstance(persisted, dict) else {}
        invoke_payload = {
            "messages": input_messages,
            "artifacts": {
                "config": {
                    "mlflow_tracking_uri": st.session_state.get("mlflow_tracking_uri"),
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
                }
            },
            "data_raw": data_raw_dict,
        }
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
        result = team.invoke(
            invoke_payload,
            config={
                "recursion_limit": recursion_limit,
                "configurable": {"thread_id": st.session_state.thread_id},
            },
        )
    except Exception as e:
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
        }
        idx = len(st.session_state.details)
        st.session_state.details.append(detail)
        msgs.add_ai_message(f"{UI_DETAIL_MARKER_PREFIX}{idx}")

# ---------------- Always-on analysis panel (bottom) ----------------
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
        with st.expander("Open analysis details", expanded=True):
            # Reuse the same rendering logic as chat history by calling render_history's helper pattern.
            # Minimal duplication: render via a small inline function to avoid leaking outer scope.
            detail = details[int(selected)]
            tabs = st.tabs(
                [
                    "AI Reasoning",
                    "Data (raw/sql/wrangle/clean/features)",
                    "Charts",
                    "Reports",
                    "Models/MLflow",
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
            with tabs[2]:
                graph_json = detail.get("plotly_graph")
                if graph_json:
                    payload = (
                        json.dumps(graph_json)
                        if isinstance(graph_json, dict)
                        else graph_json
                    )
                    fig = _apply_streamlit_plot_style(pio.from_json(payload))
                    st.plotly_chart(
                        fig, width="stretch", key=f"bottom_detail_chart_{selected}"
                    )
                else:
                    st.info("No charts returned.")
            with tabs[3]:
                reports = (
                    detail.get("eda_reports") if isinstance(detail, dict) else None
                )
                sweetviz_file = (
                    reports.get("sweetviz_report_file")
                    if isinstance(reports, dict)
                    else None
                )
                dtale_url = (
                    reports.get("dtale_url") if isinstance(reports, dict) else None
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
                            key=f"bottom_download_sweetviz_{selected}",
                        )
                    except Exception as e:
                        st.warning(f"Could not render Sweetviz report: {e}")

                if dtale_url:
                    st.markdown("**D-Tale**")
                    st.markdown(f"[Open D-Tale]({dtale_url})")

                if not sweetviz_file and not dtale_url:
                    st.info("No EDA reports returned.")

            with tabs[4]:
                model_info = detail.get("model_info")
                eval_art = detail.get("eval_artifacts")
                eval_graph = detail.get("eval_plotly_graph")
                mlflow_art = detail.get("mlflow_artifacts")
                if model_info is not None:
                    st.markdown("**Model Info**")
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
                        fig, width="stretch", key=f"bottom_eval_chart_{selected}"
                    )
                if mlflow_art is not None:
                    st.markdown("**MLflow Artifacts**")
                    st.json(mlflow_art)
                if model_info is None and eval_art is None and mlflow_art is None:
                    st.info("No model/evaluation/MLflow artifacts.")
    except Exception as e:
        st.error(f"Could not render analysis details: {e}")
else:
    st.info("No analysis details yet. Run a request to generate them.")
