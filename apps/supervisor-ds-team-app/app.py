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
from ai_data_science_team.ml_agents.h2o_ml_agent import H2OMLAgent
from ai_data_science_team.ml_agents.mlflow_tools_agent import MLflowToolsAgent
from ai_data_science_team.multiagents.supervisor_ds_team import make_supervisor_ds_team


st.set_page_config(
    page_title="Supervisor Data Science Team", page_icon=":bar_chart:", layout="wide"
)
TITLE = "Supervisor-led Data Science Team"
st.title(TITLE)

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
    "I’m a full data science copilot. Load data, wrangle/clean, run EDA, visualize, query SQL, engineer features, and train/evaluate models (H2O/MLflow). Try these on the sample Telco churn data:"
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

        #### Feature engineering / ML
        - Create model-ready features for churn (encode categoricals, handle totals/averages).
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
        ],
    )
    recursion_limit = st.slider("Recursion limit", 4, 20, 10, 1)
    add_memory = st.checkbox("Enable short-term memory", value=True)
    st.markdown("---")
    st.markdown("**Data options**")
    use_sample = st.checkbox("Load sample Telco churn data", value=False)
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    preview_rows = st.number_input("Preview rows", 1, 20, 5)
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
    mlflow_tools_agent = MLflowToolsAgent(
        llm, log_tool_calls=True, mlflow_tracking_uri=mlflow_tracking_uri
    )

    team = make_supervisor_ds_team(
        model=llm,
        data_loader_agent=data_loader_agent,
        data_wrangling_agent=data_wrangling_agent,
        data_cleaning_agent=data_cleaning_agent,
        eda_tools_agent=eda_tools_agent,
        data_visualization_agent=data_visualization_agent,
        sql_database_agent=sql_database_agent,
        feature_engineering_agent=feature_engineering_agent,
        h2o_ml_agent=h2o_ml_agent,
        mlflow_tools_agent=mlflow_tools_agent,
        checkpointer=checkpointer if use_memory else None,
    )
    return team


# ---------------- Session state ----------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "details" not in st.session_state:
    st.session_state.details = []
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
                "Data (raw/clean)",
                "Charts",
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
            wrangled_df = detail.get("data_wrangled_df")
            cleaned_df = detail.get("data_cleaned_df")
            if raw_df is not None:
                st.markdown("**Raw Preview**")
                st.dataframe(raw_df)
            if wrangled_df is not None:
                st.markdown("**Wrangled Preview**")
                st.dataframe(wrangled_df)
            if cleaned_df is not None:
                st.markdown("**Cleaned Preview**")
                st.dataframe(cleaned_df)
            if raw_df is None and wrangled_df is None and cleaned_df is None:
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
        # Models / MLflow
        with tabs[3]:
            model_info = detail.get("model_info")
            mlflow_art = detail.get("mlflow_artifacts")
            if model_info is not None:
                st.markdown("**Model Info**")
                st.json(model_info)
            if mlflow_art is not None:
                st.markdown("**MLflow Artifacts**")
                st.json(mlflow_art)
            if model_info is None and mlflow_art is None:
                st.info("No model/MLflow artifacts.")

    for m in history:
        role = getattr(m, "role", getattr(m, "type", "assistant"))
        content = getattr(m, "content", "")
        with st.chat_message("assistant" if role in ("assistant", "ai") else "human"):
            if isinstance(content, str) and content.startswith("DETAILS_INDEX:"):
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
        input_messages = [HumanMessage(content=prompt)] if add_memory else msgs.messages
        result = team.invoke(
            {
                "messages": input_messages,
                "artifacts": {},
                "data_raw": data_raw_dict,
            },
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
            "data_wrangled_df": _to_df(result.get("data_wrangled")),
            "data_cleaned_df": _to_df(result.get("data_cleaned")),
            # Only show artifacts produced during this invocation to avoid stale charts/models.
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
            "mlflow_artifacts": (
                (result.get("mlflow_artifacts") or artifacts.get("mlflow"))
                if "mlflow_tools_agent" in ran_agents
                else None
            ),
            # Store only a summarized version to avoid rendering huge payloads
            "artifacts": _summarize_artifacts(artifacts),
        }
        idx = len(st.session_state.details)
        st.session_state.details.append(detail)
        msgs.add_ai_message(f"DETAILS_INDEX:{idx}")

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
            tabs = st.tabs(["AI Reasoning", "Data (raw/clean)", "Charts", "Models/MLflow"])
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
                wrangled_df = detail.get("data_wrangled_df")
                cleaned_df = detail.get("data_cleaned_df")
                if raw_df is not None:
                    st.markdown("**Raw Preview**")
                    st.dataframe(raw_df)
                if wrangled_df is not None:
                    st.markdown("**Wrangled Preview**")
                    st.dataframe(wrangled_df)
                if cleaned_df is not None:
                    st.markdown("**Cleaned Preview**")
                    st.dataframe(cleaned_df)
                if raw_df is None and wrangled_df is None and cleaned_df is None:
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
                    st.plotly_chart(fig, width="stretch", key=f"bottom_detail_chart_{selected}")
                else:
                    st.info("No charts returned.")
            with tabs[3]:
                model_info = detail.get("model_info")
                mlflow_art = detail.get("mlflow_artifacts")
                if model_info is not None:
                    st.markdown("**Model Info**")
                    st.json(model_info)
                if mlflow_art is not None:
                    st.markdown("**MLflow Artifacts**")
                    st.json(mlflow_art)
                if model_info is None and mlflow_art is None:
                    st.info("No model/MLflow artifacts.")
    except Exception as e:
        st.error(f"Could not render analysis details: {e}")
else:
    st.info("No analysis details yet. Run a request to generate them.")
