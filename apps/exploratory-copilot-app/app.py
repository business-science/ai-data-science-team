# BUSINESS SCIENCE
# Exploratory Data Analysis (EDA) Copilot App
# -----------------------

# This app helps you search for data and produces exploratory analysis reports.

# Imports
# !pip install git+https://github.com/business-science/ai-data-science-team.git --upgrade

from openai import OpenAI
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pathlib import Path
import html

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI

import os
import sys
# 优先从本地工程导入 ai_data_science_team 源码
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ai_data_science_team.ds_agents import EDAToolsAgent
from ai_data_science_team.utils.matplotlib import matplotlib_from_base64
from ai_data_science_team.utils.plotly import plotly_from_dict

# Helpers

MODEL_LIST = ["deepseek-chat", "deepseek-reasoner"]
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

def render_report_iframe(
    report_src, src_type="url", height=620, title="Interactive Report"
):
    """
    Render a report iframe with expandable fullscreen functionality.

    Parameters:
    ----------
    report_src : str
        Either the URL of the report (for src_type='url') or the raw HTML (for src_type='html').

    src_type : str
        Type of the source: 'url' or 'html'.

    height : int
        Height of the iframe component.
    """

    if src_type == "html":
        iframe_src = f'srcdoc="{html.escape(report_src, quote=True)}"'
    else:
        iframe_src = f'src="{report_src}"'

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{title}</title>
        <style>
            body, html {{
                margin: 0;
                padding: 0;
                height: 100%;
            }}
            #iframe-container {{
                position: relative;
                width: 100%;
                height: {height}px;
            }}
            #myIframe {{
                width: 100%;
                height: 100%;
                border: none;
            }}
            #fullscreen-btn {{
                position: absolute;
                top: 10px;
                right: 10px;
                z-index: 1000;
                padding: 8px 12px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }}
        </style>
    </head>
    <body>
        <div id="iframe-container">
            <button id="fullscreen-btn" onclick="toggleFullscreen()">Full Screen</button>
            <iframe id="myIframe" {iframe_src} allowfullscreen></iframe>
        </div>
        <script>
            function toggleFullscreen() {{
                var container = document.getElementById("iframe-container");
                if (!document.fullscreenElement) {{
                    container.requestFullscreen().catch(err => {{
                        alert("Error attempting to enable full-screen mode: " + err.message);
                    }});
                    document.getElementById("fullscreen-btn").innerText = "Exit Full Screen";
                }} else {{
                    document.exitFullscreen();
                    document.getElementById("fullscreen-btn").innerText = "Full Screen";
                }}
            }}
            document.addEventListener('fullscreenchange', () => {{
                if (!document.fullscreenElement) {{
                    document.getElementById("fullscreen-btn").innerText = "Full Screen";
                }}
            }});
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=height, scrolling=True)


# =============================================================================
# STREAMLIT APP SETUP (including data upload, API key, etc.)
# =============================================================================

TITLE = "Your Exploratory Data Analysis (EDA) Copilot"
st.set_page_config(page_title=TITLE, page_icon="📊")
st.title("📊 " + TITLE)

st.markdown("""
Welcome to the EDA Copilot. This AI agent is designed to help you find and load data 
and return exploratory analysis reports that can be used to understand the data 
prior to other analysis (e.g. modeling, feature engineering, etc).
""")

with st.expander("Example Questions", expanded=False):
    st.write(
        """
        - What tools do you have access to? Return a table.
        - Give me information on the correlation funnel tool.
        - Explain the dataset.
        - What do the first 5 rows contain?
        - Describe the dataset.
        - Analyze missing data in the dataset.
        - Generate a correlation funnel. Use the Churn feature as the target.
        - Generate a Sweetviz report for the dataset. Use the Churn feature as the target.
        - Generate a Dtale report for the dataset.
        """
    )

# Sidebar for file upload / demo data
st.sidebar.header("EDA Copilot: Data Upload/Selection", divider=True)
st.sidebar.header("Upload Data (CSV or Excel)")
use_demo_data = st.sidebar.checkbox("Use demo data", value=False)

if "DATA_RAW" not in st.session_state:
    st.session_state["DATA_RAW"] = None

if use_demo_data:
    demo_file_path = Path("data/churn_data.csv")
    if demo_file_path.exists():
        df = pd.read_csv(demo_file_path)
        file_name = "churn_data"
        st.session_state["DATA_RAW"] = df.copy()
        st.write(f"## Preview of {file_name} data:")
        st.dataframe(st.session_state["DATA_RAW"])
    else:
        st.error(
            f"Demo data file not found at {demo_file_path}. Please ensure it exists."
        )
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Excel file", type=["csv", "xlsx"]
    )
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        st.session_state["DATA_RAW"] = df.copy()
        file_name = Path(uploaded_file.name).stem
        st.write(f"## Preview of {file_name} data:")
        st.dataframe(st.session_state["DATA_RAW"])
    else:
        st.info("Please upload a CSV or Excel file or Use Demo Data to proceed.")

# Sidebar: OpenAI API Key and Model Selection
st.sidebar.header("Enter your OpenAI API Key")
st.session_state["OPENAI_API_KEY"] = st.sidebar.text_input(
    "API Key",
    type="password",
    help="Your OpenAI API key is required for the app to function.",
)

if st.session_state["OPENAI_API_KEY"]:
    client = OpenAI(api_key=st.session_state["OPENAI_API_KEY"], base_url=DEEPSEEK_BASE_URL)
    try:
        models = client.models.list()
        st.success("API Key is valid!")
    except Exception as e:
        st.error(f"Invalid API Key: {e}")
else:
    st.info("Please enter your OpenAI API Key to proceed.")
    st.stop()



model_option = st.sidebar.selectbox("Choose OpenAI model", MODEL_LIST, index=0)
llm = ChatOpenAI(model=model_option, api_key=st.session_state["OPENAI_API_KEY"], base_url=DEEPSEEK_BASE_URL)

# =============================================================================
# CHAT MESSAGE HISTORY AND ARTIFACT STORAGE
# =============================================================================

msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

if "chat_artifacts" not in st.session_state:
    st.session_state["chat_artifacts"] = {}


def display_chat_history():
    """
    Renders the entire chat history along with any artifacts attached to messages.
    Artifacts (e.g., plots, dataframes, Sweetviz reports) are rendered inside expanders.
    """
    for i, msg in enumerate(msgs.messages):
        with st.chat_message(msg.type):
            st.write(msg.content)
            if (
                "chat_artifacts" in st.session_state
                and i in st.session_state["chat_artifacts"]
            ):
                for artifact in st.session_state["chat_artifacts"][i]:
                    with st.expander(artifact["title"], expanded=True):
                        if artifact["render_type"] == "dataframe":
                            st.dataframe(artifact["data"])
                        elif artifact["render_type"] == "matplotlib":
                            st.pyplot(artifact["data"])
                        elif artifact["render_type"] == "plotly":
                            st.plotly_chart(artifact["data"])
                        elif artifact["render_type"] == "sweetviz":
                            report_file = artifact["data"].get("report_file")
                            try:
                                with open(report_file, "r", encoding="utf-8") as f:
                                    report_html = f.read()
                            except Exception as e:
                                st.error(f"Could not open report file: {e}")
                                report_html = "<h1>Report not found</h1>"

                            render_report_iframe(
                                report_html,
                                src_type="html",
                                height=620,
                                title="Sweetviz Report",
                            )
                        elif artifact["render_type"] == "dtale":
                            dtale_url = artifact["data"]["dtale_url"]
                            render_report_iframe(
                                dtale_url,
                                src_type="url",
                                height=620,
                                title="Dtale Report",
                            )

                        else:
                            st.write("Artifact of unknown type.")


# =============================================================================
# PROCESS AGENTS AND ARTIFACTS
# =============================================================================


def process_exploratory(question: str, llm, data: pd.DataFrame) -> dict:
    """
    Initializes and calls the EDA agent using the provided question and data.
    Processes any returned artifacts (plots, dataframes, etc.) and returns a result dict.
    """
    eda_agent = EDAToolsAgent(
        llm,
        invoke_react_agent_kwargs={"recursion_limit": 10},
    )

    question += " Don't return hyperlinks to files in the response."

    eda_agent.invoke_agent(
        user_instructions=question,
        data_raw=data,
    )

    tool_calls = eda_agent.get_tool_calls()
    ai_message = eda_agent.get_ai_message(markdown=False)
    artifacts = eda_agent.get_artifacts(as_dataframe=False)

    result = {
        "ai_message": ai_message,
        "tool_calls": tool_calls,
        "artifacts": artifacts,
    }

    if tool_calls:
        last_tool_call = tool_calls[-1]
        result["last_tool_call"] = last_tool_call
        tool_name = last_tool_call

        print(f"Tool Name: {tool_name}")

        if tool_name == "explain_data":
            result["explanation"] = ai_message

        elif tool_name == "describe_dataset":
            if artifacts and isinstance(artifacts, dict) and "describe_df" in artifacts:
                try:
                    df = pd.DataFrame(artifacts["describe_df"])
                    result["describe_df"] = df
                except Exception as e:
                    st.error(f"Error processing describe_dataset artifact: {e}")

        elif tool_name == "visualize_missing":
            if artifacts and isinstance(artifacts, dict):
                try:
                    matrix_fig = matplotlib_from_base64(artifacts.get("matrix_plot"))
                    bar_fig = matplotlib_from_base64(artifacts.get("bar_plot"))
                    heatmap_fig = matplotlib_from_base64(artifacts.get("heatmap_plot"))
                    result["matrix_plot_fig"] = matrix_fig[0]
                    result["bar_plot_fig"] = bar_fig[0]
                    result["heatmap_plot_fig"] = heatmap_fig[0]
                except Exception as e:
                    st.error(f"Error processing visualize_missing artifact: {e}")

        elif tool_name == "generate_correlation_funnel":
            if artifacts and isinstance(artifacts, dict):
                if "correlation_data" in artifacts:
                    try:
                        corr_df = pd.DataFrame(artifacts["correlation_data"])
                        result["correlation_data"] = corr_df
                    except Exception as e:
                        st.error(f"Error processing correlation_data: {e}")
                if "plotly_figure" in artifacts:
                    try:
                        corr_plotly = plotly_from_dict(artifacts["plotly_figure"])
                        result["correlation_plotly"] = corr_plotly
                    except Exception as e:
                        st.error(
                            f"Error processing correlation funnel Plotly figure: {e}"
                        )

        elif tool_name == "generate_sweetviz_report":
            if artifacts and isinstance(artifacts, dict):
                result["report_file"] = artifacts.get("report_file")
                result["report_html"] = artifacts.get("report_html")

        elif tool_name == "generate_dtale_report":
            if artifacts and isinstance(artifacts, dict):
                result["dtale_url"] = artifacts.get("dtale_url")

        else:
            if artifacts and isinstance(artifacts, dict):
                if "plotly_figure" in artifacts:
                    try:
                        plotly_fig = plotly_from_dict(artifacts["plotly_figure"])
                        result["plotly_fig"] = plotly_fig
                    except Exception as e:
                        st.error(f"Error processing Plotly figure: {e}")
                if "plot_image" in artifacts:
                    try:
                        fig = matplotlib_from_base64(artifacts["plot_image"])
                        result["matplotlib_fig"] = fig
                    except Exception as e:
                        st.error(f"Error processing matplotlib image: {e}")
                if "dataframe" in artifacts:
                    try:
                        df = pd.DataFrame(artifacts["dataframe"])
                        result["dataframe"] = df
                    except Exception as e:
                        st.error(f"Error converting artifact to dataframe: {e}")
    else:
        result["plain_response"] = ai_message

    return result


# =============================================================================
# MAIN INTERACTION: GET USER QUESTION AND HANDLE RESPONSE
# =============================================================================

if st.session_state["DATA_RAW"] is not None:
    # Use the built-in chat input widget
    question = st.chat_input("Enter your question here:", key="query_input")
    if question:
        if not st.session_state["OPENAI_API_KEY"]:
            st.error("Please enter your OpenAI API Key to proceed.")
            st.stop()

        with st.spinner("Thinking..."):
            # Add the user's question to the message history
            msgs.add_user_message(question)
            result = process_exploratory(question, llm, st.session_state["DATA_RAW"])

            tool_name = None
            if "last_tool_call" in result:
                tool_name = result["last_tool_call"]

            # Append the AI response and (if available) tool usage info
            ai_msg = result.get("ai_message", "")
            if tool_name:
                ai_msg += f"\n\n*Tool Used: {tool_name}*"

            msgs.add_ai_message(ai_msg)

            # Build an artifact list to attach to the latest AI message
            artifact_list = []
            if "last_tool_call" in result:
                tool_name = result["last_tool_call"]
                if tool_name == "describe_dataset":
                    if "describe_df" in result:
                        artifact_list.append(
                            {
                                "title": "Dataset Description",
                                "render_type": "dataframe",
                                "data": result["describe_df"],
                            }
                        )
                elif tool_name == "visualize_missing":
                    if "matrix_plot_fig" in result:
                        artifact_list.append(
                            {
                                "title": "Missing Data Matrix",
                                "render_type": "matplotlib",
                                "data": result["matrix_plot_fig"],
                            }
                        )
                    if "bar_plot_fig" in result:
                        artifact_list.append(
                            {
                                "title": "Missing Data Bar Plot",
                                "render_type": "matplotlib",
                                "data": result["bar_plot_fig"],
                            }
                        )
                    if "heatmap_plot_fig" in result:
                        artifact_list.append(
                            {
                                "title": "Missing Data Heatmap",
                                "render_type": "matplotlib",
                                "data": result["heatmap_plot_fig"],
                            }
                        )
                elif tool_name == "generate_correlation_funnel":
                    if "correlation_data" in result:
                        artifact_list.append(
                            {
                                "title": "Correlation Data",
                                "render_type": "dataframe",
                                "data": result["correlation_data"],
                            }
                        )
                    if "correlation_plotly" in result:
                        artifact_list.append(
                            {
                                "title": "Correlation Funnel (Interactive Plotly)",
                                "render_type": "plotly",
                                "data": result["correlation_plotly"],
                            }
                        )
                elif tool_name == "generate_sweetviz_report":
                    artifact_list.append(
                        {
                            "title": "Sweetviz Report",
                            "render_type": "sweetviz",
                            "data": {
                                "report_file": result.get("report_file"),
                                "report_html": result.get("report_html"),
                            },
                        }
                    )
                elif tool_name == "generate_dtale_report":
                    artifact_list.append(
                        {
                            "title": "Dtale Interactive Report",
                            "render_type": "dtale",
                            "data": {"dtale_url": result.get("dtale_url")},
                        }
                    )

                else:
                    if "plotly_fig" in result:
                        artifact_list.append(
                            {
                                "title": "Plotly Figure",
                                "render_type": "plotly",
                                "data": result["plotly_fig"],
                            }
                        )
                    if "matplotlib_fig" in result:
                        artifact_list.append(
                            {
                                "title": "Matplotlib Figure",
                                "render_type": "matplotlib",
                                "data": result["matplotlib_fig"],
                            }
                        )
                    if "dataframe" in result:
                        artifact_list.append(
                            {
                                "title": "Dataframe",
                                "render_type": "dataframe",
                                "data": result["dataframe"],
                            }
                        )

            # Attach artifacts to the most recent AI message (so they show immediately)
            if artifact_list:
                msg_index = len(msgs.messages) - 1
                st.session_state["chat_artifacts"][msg_index] = artifact_list


# =============================================================================
# FINAL RENDER: DISPLAY THE COMPLETE CHAT HISTORY WITH ARTIFACTS
# =============================================================================

display_chat_history()
