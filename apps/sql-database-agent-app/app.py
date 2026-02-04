# BUSINESS SCIENCE
# SQL Database Agent App
# -----------------------

# This app is designed to help you query your SQL database and return data frames that you can interactively inspect and download.

# Imports
# !pip install git+https://github.com/business-science/ai-data-science-team.git --upgrade

from openai import OpenAI

import inspect
import os
import streamlit as st
import sqlalchemy as sql
import pandas as pd
import asyncio

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI

from ai_data_science_team.agents import SQLDatabaseAgent

# * APP INPUTS ----

# MODIFY THIS TO YOUR DATABASE PATH IF YOU WANT TO USE A DIFFERENT DATABASE
DB_OPTIONS = {
    "Northwind Database": "sqlite:///data/northwind.db",
}

MODEL_LIST = ['gpt-4o-mini', 'gpt-4o']

TITLE = "Your SQL Database Agent"

# * STREAMLIT APP SETUP ----

st.set_page_config(page_title=TITLE, page_icon="📊", )
st.title(TITLE)

st.markdown("""
Welcome to the SQL Database Agent. This AI agent is designed to help you query your SQL database and return data frames that you can interactively inspect and download.
""")

with st.expander("Example Questions", expanded=False):
    st.write(
        """
        - What tables exist in the database?
        - What are the first 10 rows in the territory table?
        - Aggregate sales for each territory. 
        - Aggregate sales by month for each territory.
        """
    )

# * STREAMLIT APP SIDEBAR ----

# Database Selection

db_option = st.sidebar.selectbox(
    "Select a Database",
    list(DB_OPTIONS.keys()),
)

st.session_state["PATH_DB"] = DB_OPTIONS.get(db_option)

sql_engine = sql.create_engine(st.session_state["PATH_DB"])

conn = sql_engine.connect()

# * OpenAI API Key

st.sidebar.header("Enter your OpenAI API Key")

st.session_state["OPENAI_API_KEY"] = st.sidebar.text_input("API Key", type="password", help="Your OpenAI API key is required for the app to function.")
default_openai_base_url = st.session_state.get("OPENAI_BASE_URL")
if default_openai_base_url is None:
    default_openai_base_url = os.environ.get("OPENAI_BASE_URL", "")
st.session_state["OPENAI_BASE_URL"] = st.sidebar.text_input(
    "Base URL (optional)",
    value=default_openai_base_url,
    help="Override OpenAI base URL, e.g. http://localhost:8000/v1",
)

# Test OpenAI API Key
openai_base_url = (st.session_state.get("OPENAI_BASE_URL") or "").strip() or None
if st.session_state["OPENAI_API_KEY"]:
    # Set the API key for OpenAI
    client_kwargs = {"api_key": st.session_state["OPENAI_API_KEY"]}
    if openai_base_url:
        client_kwargs["base_url"] = openai_base_url
    client = OpenAI(**client_kwargs)
    
    # Test the API key (optional)
    try:
        # Example: Fetch models to validate the key
        models = client.models.list()
        st.success("API Key is valid!")
    except Exception as e:
        st.error(f"Invalid API Key: {e}")
else:
    st.info("Please enter your OpenAI API Key to proceed.")
    st.stop()


# * OpenAI Model Selection

model_option = st.sidebar.selectbox(
    "Choose OpenAI model",
    MODEL_LIST,
    index=0
)

def _chat_openai_kwargs(model_name, api_key, base_url):
    kwargs = {"model": model_name, "api_key": api_key}
    if base_url:
        try:
            sig = inspect.signature(ChatOpenAI)
            if "base_url" in sig.parameters:
                kwargs["base_url"] = base_url
            elif "openai_api_base" in sig.parameters:
                kwargs["openai_api_base"] = base_url
        except Exception:
            kwargs["base_url"] = base_url
    return kwargs


OPENAI_LLM = ChatOpenAI(
    **_chat_openai_kwargs(
        model_option, st.session_state["OPENAI_API_KEY"], openai_base_url
    )
)

llm = OPENAI_LLM

# * STREAMLIT 

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

# Initialize dataframe storage in session state
if "dataframes" not in st.session_state:
    st.session_state.dataframes = []

# Function to display chat messages including Plotly charts and dataframes
def display_chat_history():
    for i, msg in enumerate(msgs.messages):
        with st.chat_message(msg.type):
            if "DATAFRAME_INDEX:" in msg.content:
                df_index = int(msg.content.split("DATAFRAME_INDEX:")[1])
                st.dataframe(st.session_state.dataframes[df_index])
            else:
                st.write(msg.content)

# Render current messages from StreamlitChatMessageHistory
display_chat_history()

# Create the SQL Database Agent
sql_db_agent = SQLDatabaseAgent(
    model = llm,
    connection=conn,
    n_samples=1,
    log = False,
    bypass_recommended_steps=True,
)

# Handle the question async
async def handle_question(question):
    await sql_db_agent.ainvoke_agent(
        user_instructions=question,
    )
    return sql_db_agent


if st.session_state["PATH_DB"] and (question := st.chat_input("Enter your question here:", key="query_input")):
    
    if not st.session_state["OPENAI_API_KEY"]:
        st.error("Please enter your OpenAI API Key to proceed.")
        st.stop()
    
    with st.spinner("Thinking..."):
        
        st.chat_message("human").write(question)
        msgs.add_user_message(question)
        
        # Run the app       
        error_occured = False
        try: 
            print(st.session_state["PATH_DB"])
            result = asyncio.run(handle_question(question))
        except Exception as e:
            error_occured = True
            print(e)
            
            response_text = f"""
            I'm sorry. I am having difficulty answering that question. You can try providing more details and I'll do my best to provide an answer.
            
            Error: {e}
            """
            msgs.add_ai_message(response_text)
            st.chat_message("ai").write(response_text)
            st.error(f"Error: {e}")
        
        # Generate the Results
        if not error_occured:
            
            sql_query = result.get_sql_query_code()
            response_df = result.get_data_sql()
            
            if sql_query:
                
                # Store the SQL
                response_1 = f"### SQL Results:\n\nSQL Query:\n\n```sql\n{sql_query}\n```\n\nResult:"
                
                # Store the forecast df and keep its index
                df_index = len(st.session_state.dataframes)
                st.session_state.dataframes.append(response_df)

                # Store response
                msgs.add_ai_message(response_1)
                msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                
                # Write Results
                st.chat_message("ai").write(response_1)
                st.dataframe(response_df)
        
