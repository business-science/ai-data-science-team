"""
Streamlit app for the Supervisor-led Data Science Team.

Command:
    streamlit run apps/supervisor-ds-team-app/app.py
"""

from __future__ import annotations

import uuid
import os
from openai import OpenAI
import pandas as pd
import sqlalchemy as sql
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
from ai_data_science_team.agents.feature_engineering_agent import FeatureEngineeringAgent
from ai_data_science_team.ml_agents.h2o_ml_agent import H2OMLAgent
from ai_data_science_team.ml_agents.mlflow_tools_agent import MLflowToolsAgent
from ai_data_science_team.multiagents.supervisor_ds_team import make_supervisor_ds_team


st.set_page_config(page_title="Supervisor Data Science Team", page_icon=":bar_chart:", layout="wide")
TITLE = "Supervisor-led Data Science Team"
st.title(TITLE)

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
    model_choice = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"])
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

    if st.button("Clear chat"):
        st.session_state.chat_history = []
        msgs = StreamlitChatMessageHistory(key="supervisor_ds_msgs")
        msgs.clear()
        msgs.add_ai_message("How can the data science team help today?")
        st.session_state.thread_id = str(uuid.uuid4())

# Hard gate: require valid API key before rendering the rest of the app
resolved_api_key = st.session_state.get("OPENAI_API_KEY")
if not resolved_api_key:
    st.info("Please enter your OpenAI API key in the sidebar to proceed.")
    st.stop()
if key_status == "bad":
    st.error("Invalid OpenAI API key. Please fix it in the sidebar.")
    st.stop()


@st.cache_resource(show_spinner=False)
def get_checkpointer():
    return MemorySaver()


@st.cache_resource(show_spinner=False)
def build_team(model_name: str, use_memory: bool, sql_url: str):
    llm = ChatOpenAI(model=model_name)
    data_loader_agent = DataLoaderToolsAgent(llm, invoke_react_agent_kwargs={"recursion_limit": 4})
    data_wrangling_agent = DataWranglingAgent(llm, log=False)
    data_cleaning_agent = DataCleaningAgent(llm, log=False)
    eda_tools_agent = EDAToolsAgent(llm, log_tool_calls=True)
    data_visualization_agent = DataVisualizationAgent(llm, log=False)
    # SQL connection is optional; default to in-memory sqlite to satisfy constructor
    conn = sql.create_engine(sql_url or "sqlite:///:memory:").connect()
    sql_database_agent = SQLDatabaseAgent(llm, connection=conn, log=False)
    feature_engineering_agent = FeatureEngineeringAgent(llm, log=False)
    h2o_ml_agent = H2OMLAgent(llm, log=False)
    mlflow_tools_agent = MLflowToolsAgent(llm, log_tool_calls=True)

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
        checkpointer=get_checkpointer() if use_memory else None,
    )
    return team


# ---------------- Session state ----------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "details" not in st.session_state:
    st.session_state.details = []

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
    for m in history:
        role = getattr(m, "role", getattr(m, "type", "assistant"))
        content = getattr(m, "content", "")
        with st.chat_message("assistant" if role in ("assistant", "ai") else "human"):
            if isinstance(content, str) and content.startswith("DETAILS_INDEX:"):
                try:
                    idx = int(content.split(":")[1])
                    detail = st.session_state.details[idx]
                except Exception:
                    st.write(content)
                    continue
                with st.expander("Analysis Details", expanded=False):
                    tabs = st.tabs(
                        ["AI Reply", "Data (raw/clean)", "Charts", "Models/MLflow", "Artifacts"]
                    )
                    with tabs[0]:
                        st.write(detail.get("ai_reply", "_(No reply)_"))
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
                            try:
                                fig = pio.from_json(graph_json)
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error rendering chart: {e}")
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
                    with tabs[4]:
                        st.json(detail.get("artifacts", {}))
            else:
                st.write(content)


render_history(msgs.messages)

# Show data preview (if selected) and store for reuse on submit
data_raw_dict, _ = get_input_data()
st.session_state.selected_data_raw = data_raw_dict

# ---------------- User input ----------------
prompt = st.chat_input("Ask the data science team...")
if prompt:
    if not resolved_api_key or key_status == "bad":
        st.error("OpenAI API key is required and must be valid. Enter it in the sidebar.")
        st.stop()

    st.chat_message("human").write(prompt)
    msgs.add_user_message(prompt)

    data_raw_dict = st.session_state.get("selected_data_raw")

    team = build_team(model_choice, add_memory, st.session_state.get("sql_url", "sqlite:///:memory:"))
    try:
        result = team.invoke(
            {
                "messages": msgs.messages,
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
        # append last AI message to chat history for display
        last_ai = None
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) or getattr(msg, "role", None) in ("assistant", "ai"):
                last_ai = msg
                break
        if last_ai:
            msgs.add_ai_message(getattr(last_ai, "content", ""))
            st.chat_message("assistant").write(getattr(last_ai, "content", ""))

        # Collect detail snapshot for tabbed display
        artifacts = result.get("artifacts", {}) or {}
        def _to_df(obj):
            try:
                return pd.DataFrame(obj) if obj is not None else None
            except Exception:
                return None

        detail = {
            "ai_reply": getattr(last_ai, "content", "") if last_ai else "",
            "data_raw_df": _to_df(result.get("data_raw")),
            "data_wrangled_df": _to_df(result.get("data_wrangled")),
            "data_cleaned_df": _to_df(result.get("data_cleaned")),
            "plotly_graph": artifacts.get("viz", {}).get("plotly_graph") if isinstance(artifacts.get("viz"), dict) else None,
            "model_info": result.get("model_info") or artifacts.get("h2o"),
            "mlflow_artifacts": result.get("mlflow_artifacts") or artifacts.get("mlflow"),
            "artifacts": artifacts,
        }
        idx = len(st.session_state.details)
        st.session_state.details.append(detail)
        msgs.add_ai_message(f"DETAILS_INDEX:{idx}")

        # Artifacts preview
        art = result.get("artifacts", {}) or {}
        if art:
            st.subheader("Artifacts")
            st.write("Keys:", list(art.keys()))
            # Show first table-like artifact if present
            for key, val in art.items():
                if isinstance(val, dict) and "data" in val:
                    try:
                        df = pd.DataFrame(val["data"])
                        st.markdown(f"**{key}**")
                        st.dataframe(df.head())
                        break
                    except Exception:
                        continue
                if isinstance(val, dict) and "plotly_graph" in val:
                    try:
                        fig = pio.from_json(val["plotly_graph"])
                        st.markdown(f"**{key}**")
                        st.plotly_chart(fig, use_container_width=True)
                        break
                    except Exception:
                        continue
