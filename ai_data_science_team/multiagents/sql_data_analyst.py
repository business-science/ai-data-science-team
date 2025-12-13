from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Checkpointer

from typing_extensions import TypedDict, Annotated, Sequence

import pandas as pd
import json
from IPython.display import Markdown

from ai_data_science_team.templates import BaseAgent
from ai_data_science_team.agents import SQLDatabaseAgent, DataVisualizationAgent
from ai_data_science_team.utils.plotly import plotly_from_dict
from ai_data_science_team.utils.regex import (
    remove_consecutive_duplicates,
    get_generic_summary,
)

AGENT_NAME = "sql_data_analyst"


class SQLDataAnalyst(BaseAgent):
    """
    SQLDataAnalyst is a multi-agent class that combines SQL database querying and data visualization capabilities.

    Parameters:
    -----------
    model:
        The language model to be used for the agents.
    sql_database_agent: SQLDatabaseAgent
        The SQL Database Agent.
    data_visualization_agent: DataVisualizationAgent
        The Data Visualization Agent.
    checkpointer: Checkpointer (optional)
        The checkpointer to save the state of the multi-agent system.

    Methods:
    --------
    ainvoke_agent(user_instructions, **kwargs)
        Asynchronously invokes the SQL Data Analyst Multi-Agent with the given user instructions.
    invoke_agent(user_instructions, **kwargs)
        Invokes the SQL Data Analyst Multi-Agent with the given user instructions.
    get_data_sql()
        Returns the SQL data as a Pandas DataFrame.
    get_plotly_graph()
        Returns the Plotly graph as a Plotly object.
    get_sql_query_code(markdown=False)
        Returns the SQL query code as a string, optionally formatted as a Markdown code block.
    get_sql_database_function(markdown=False)
        Returns the SQL database function as a string, optionally formatted as a Markdown code block.
    get_data_visualization_function(markdown=False)
        Returns the data visualization function as a string, optionally formatted as a Markdown code block.
    """

    def __init__(
        self,
        model,
        sql_database_agent: SQLDatabaseAgent,
        data_visualization_agent: DataVisualizationAgent,
        checkpointer: Checkpointer = None,
    ):
        self._params = {
            "model": model,
            "sql_database_agent": sql_database_agent,
            "data_visualization_agent": data_visualization_agent,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        """
        Create or rebuild the compiled graph for the SQL Data Analyst Multi-Agent.
        Running this method resets the response to None.
        """
        self.response = None
        return make_sql_data_analyst(
            model=self._params["model"],
            sql_database_agent=self._params["sql_database_agent"]._compiled_graph,
            data_visualization_agent=self._params[
                "data_visualization_agent"
            ]._compiled_graph,
            checkpointer=self._params["checkpointer"],
        )

    def update_params(self, **kwargs):
        """
        Updates the agent's parameters (e.g. model, sql_database_agent, etc.)
        and rebuilds the compiled graph.
        """
        for k, v in kwargs.items():
            self._params[k] = v
        self._compiled_graph = self._make_compiled_graph()

    async def ainvoke_agent(
        self, user_instructions, max_retries: int = 3, retry_count: int = 0, **kwargs
    ):
        """
        Asynchronosly nvokes the SQL Data Analyst Multi-Agent.

        Parameters:
        ----------
        user_instructions: str
            The user's instructions for the combined SQL and (optionally) Data Visualization agents.
        **kwargs:
            Additional keyword arguments to pass to the compiled graph's `ainvoke` method.

        Returns:
        -------
        None. The response is stored in the `response` attribute.

        Example:
        --------
        ``` python
        from langchain_openai import ChatOpenAI
        import sqlalchemy as sql
        from ai_data_science_team.multiagents import SQLDataAnalyst
        from ai_data_science_team.agents import SQLDatabaseAgent, DataVisualizationAgent

        llm = ChatOpenAI(model = "gpt-4o-mini")

        sql_engine = sql.create_engine("sqlite:///data/northwind.db")

        conn = sql_engine.connect()

        sql_data_analyst = SQLDataAnalyst(
            model = llm,
            sql_database_agent = SQLDatabaseAgent(
                model = llm,
                connection = conn,
                n_samples = 1,
            ),
            data_visualization_agent = DataVisualizationAgent(
                model = llm,
                n_samples = 10,
            )
        )

        sql_data_analyst.ainvoke_agent(
            user_instructions = "Make a plot of sales revenue by month by territory. Make a dropdown for the user to select the territory.",
        )

        sql_data_analyst.get_sql_query_code()

        sql_data_analyst.get_data_sql()

        sql_data_analyst.get_plotly_graph()
        ```
        """
        response = await self._compiled_graph.ainvoke(
            {
                "user_instructions": user_instructions,
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )

        if response.get("messages"):
            response["messages"] = remove_consecutive_duplicates(response["messages"])

        self.response = response

    def invoke_agent(
        self, user_instructions, max_retries: int = 3, retry_count: int = 0, **kwargs
    ):
        """
        Invokes the SQL Data Analyst Multi-Agent.

        Parameters:
        ----------
        user_instructions: str
            The user's instructions for the combined SQL and (optionally) Data Visualization agents.
        max_retries (int):
                Maximum retry attempts for cleaning.
        retry_count (int):
            Current retry attempt.
        **kwargs:
            Additional keyword arguments to pass to the compiled graph's `invoke` method.

        Returns:
        -------
        None. The response is stored in the `response` attribute.

        Example:
        --------
        ``` python
        from langchain_openai import ChatOpenAI
        import sqlalchemy as sql
        from ai_data_science_team.multiagents import SQLDataAnalyst
        from ai_data_science_team.agents import SQLDatabaseAgent, DataVisualizationAgent

        llm = ChatOpenAI(model = "gpt-4o-mini")

        sql_engine = sql.create_engine("sqlite:///data/northwind.db")

        conn = sql_engine.connect()

        sql_data_analyst = SQLDataAnalyst(
            model = llm,
            sql_database_agent = SQLDatabaseAgent(
                model = llm,
                connection = conn,
                n_samples = 1,
            ),
            data_visualization_agent = DataVisualizationAgent(
                model = llm,
                n_samples = 10,
            )
        )

        sql_data_analyst.invoke_agent(
            user_instructions = "Make a plot of sales revenue by month by territory. Make a dropdown for the user to select the territory.",
        )

        sql_data_analyst.get_sql_query_code()

        sql_data_analyst.get_data_sql()

        sql_data_analyst.get_plotly_graph()
        ```
        """
        response = self._compiled_graph.invoke(
            {
                "user_instructions": user_instructions,
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )

        if response.get("messages"):
            response["messages"] = remove_consecutive_duplicates(response["messages"])

        self.response = response

    def invoke_messages(
        self,
        messages: Sequence[BaseMessage],
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
        """
        Invoke the multi-agent with an explicit message list (preferred for supervisors/teams).
        """
        # If user_instructions not provided, derive from last human message if present
        user_instructions = kwargs.pop("user_instructions", None)
        if user_instructions is None:
            for msg in reversed(messages):
                if getattr(msg, "type", None) == "human" or getattr(msg, "role", None) == "user":
                    user_instructions = msg.content
                    break
        response = self._compiled_graph.invoke(
            {
                "messages": messages,
                "user_instructions": user_instructions,
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        if response.get("messages"):
            response["messages"] = remove_consecutive_duplicates(response["messages"])
        self.response = response
        return None

    async def ainvoke_messages(
        self,
        messages: Sequence[BaseMessage],
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
        """
        Async version of invoke_messages.
        """
        user_instructions = kwargs.pop("user_instructions", None)
        if user_instructions is None:
            for msg in reversed(messages):
                if getattr(msg, "type", None) == "human" or getattr(msg, "role", None) == "user":
                    user_instructions = msg.content
                    break
        response = await self._compiled_graph.ainvoke(
            {
                "messages": messages,
                "user_instructions": user_instructions,
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        if response.get("messages"):
            response["messages"] = remove_consecutive_duplicates(response["messages"])
        self.response = response
        return None

    def get_data_sql(self):
        """
        Returns the SQL data as a Pandas DataFrame.
        """
        if self.response:
            if self.response.get("data_sql"):
                return pd.DataFrame(self.response.get("data_sql"))

    def get_plotly_graph(self):
        """
        Returns the Plotly graph as a Plotly object.
        """
        if self.response:
            if self.response.get("plotly_graph"):
                return plotly_from_dict(self.response.get("plotly_graph"))

    def get_sql_query_code(self, markdown=False):
        """
        Returns the SQL query code as a string.

        Parameters:
        ----------
        markdown: bool
            If True, returns the code as a Markdown code block for Jupyter (IPython).
            For streamlit, use `st.code()` instead.
        """
        if self.response:
            if self.response.get("sql_query_code"):
                if markdown:
                    return Markdown(
                        f"```sql\n{self.response.get('sql_query_code')}\n```"
                    )
                return self.response.get("sql_query_code")

    def get_sql_database_function(self, markdown=False):
        """
        Returns the SQL database function as a string.

        Parameters:
        ----------
        markdown: bool
            If True, returns the function as a Markdown code block for Jupyter (IPython).
            For streamlit, use `st.code()` instead.
        """
        if self.response:
            if self.response.get("sql_database_function"):
                if markdown:
                    return Markdown(
                        f"```python\n{self.response.get('sql_database_function')}\n```"
                    )
                return self.response.get("sql_database_function")

    def get_data_visualization_function(self, markdown=False):
        """
        Returns the data visualization function as a string.

        Parameters:
        ----------
        markdown: bool
            If True, returns the function as a Markdown code block for Jupyter (IPython).
            For streamlit, use `st.code()` instead.
        """
        if self.response:
            if self.response.get("data_visualization_function"):
                if markdown:
                    return Markdown(
                        f"```python\n{self.response.get('data_visualization_function')}\n```"
                    )
                return self.response.get("data_visualization_function")

    def get_workflow_summary(self, markdown=False):
        """
        Returns a summary of the SQL Data Analyst workflow.

        Parameters:
        ----------
        markdown: bool
            If True, returns the summary as a Markdown-formatted string.
        """
        if self.response and self.response.get("messages"):
            agents = []
            seen = set()
            allowed = {
                "sql_database_agent",
                "data_visualization_agent",
            }
            role_to_content = {}
            for msg in self.response["messages"]:
                role = getattr(msg, "role", None) or getattr(msg, "type", None)
                if role in allowed and role not in seen:
                    agents.append(role)
                    seen.add(role)
                if role in allowed and role not in role_to_content:
                    role_to_content[role] = getattr(msg, "content", "")
            agent_labels = [
                f"- **Agent {i + 1}:** {role}\n" for i, role in enumerate(agents)
            ]
            header = (
                f"# SQL Data Analyst Workflow Summary\n\nThis workflow contains {len(agents)} agents:\n\n"
                + "\n".join(agent_labels)
            )
            reports = []
            for role in agents:
                content = role_to_content.get(role, "")
                try:
                    reports.append(get_generic_summary(json.loads(content)))
                except Exception:
                    reports.append(content)
            summary = "\n\n" + header + "\n\n".join(reports)
            return Markdown(summary) if markdown else summary


def make_sql_data_analyst(
    model,
    sql_database_agent: CompiledStateGraph,
    data_visualization_agent: CompiledStateGraph,
    checkpointer: Checkpointer = None,
):
    """
    Creates a multi-agent system that takes in a SQL query and returns a plot or table.

    - Agent 1: SQL Database Agent made with `SQLDatabaseAgent()`
    - Agent 2: Data Visualization Agent made with `DataVisualizationAgent()`

    Parameters:
    ----------
    model:
        The language model to be used for the agents.
    sql_database_agent: CompiledStateGraph
        The SQL Database Agent made with `SQLDatabaseAgent()`.
    data_visualization_agent: CompiledStateGraph
        The Data Visualization Agent made with `DataVisualizationAgent()`.
    checkpointer: Checkpointer (optional)
        The checkpointer to save the state of the multi-agent system.
        Default: None

    Returns:
    -------
    CompiledStateGraph
        The compiled multi-agent system.
    """

    llm = model

    routing_preprocessor_prompt = PromptTemplate(
        template="""
        You are an expert in routing decisions for a SQL Database Agent, a Charting Visualization Agent, and a Pandas Table Agent. Your job is to:
        
        1. Determine what the correct format for a Users Question should be for use with a SQL Database Agent based on the incoming user question. Anything related to database and data manipulation should be passed along.
        2. Determine whether or not a chart should be generated or a table should be returned based on the users question.
        3. If a chart is requested, determine the correct format of a Users Question should be used with a Data Visualization Agent. Anything related to plotting and visualization should be passed along.
        
        Use the following criteria on how to route the the initial user question:
        
        From the incoming user question, remove any details about the format of the final response as either a Chart or Table and return only the important part of the incoming user question that is relevant for the SQL generator agent. This will be the 'user_instructions_sql_database'. If 'None' is found, return the original user question.
        
        Next, determine if the user would like a data visualization ('chart') or a 'table' returned with the results of the Data Wrangling Agent. If unknown, not specified or 'None' is found, then select 'table'.  
        
        If a 'chart' is requested, return the 'user_instructions_data_visualization'. If 'None' is found, return None.
        
        Return JSON with 'user_instructions_sql_database', 'user_instructions_data_visualization' and 'routing_preprocessor_decision'.
        
        INITIAL_USER_QUESTION: {user_instructions}
        """,
        input_variables=["user_instructions"],
    )

    routing_preprocessor = routing_preprocessor_prompt | llm | JsonOutputParser()

    class PrimaryState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        user_instructions: str
        user_instructions_sql_database: str
        user_instructions_data_visualization: str
        routing_preprocessor_decision: str
        sql_query_code: str
        sql_database_function: str
        data_sql: dict
        data_raw: dict
        plot_required: bool
        data_visualization_function: str
        plotly_graph: dict
        plotly_error: str
        max_retries: int
        retry_count: int

    def prepare_messages(state: PrimaryState):
        print("---SQL DATA ANALYST---")
        print("*************************")
        print("---PREPARE MESSAGES---")
        msgs = state.get("messages", [])
        ui = state.get("user_instructions")
        if not msgs:
            system_hint = (
                "You are a SQL data analyst orchestrator. Route the user's question to SQL querying "
                "and optional visualization. Prefer tables unless the user clearly requests a chart."
            )
            msgs = [("system", system_hint), ("user", ui)]
        if not ui:
            for msg in reversed(msgs):
                if getattr(msg, "type", None) == "human" or getattr(msg, "role", None) == "user":
                    ui = msg.content
                    break
        # Normalize any tuple messages into BaseMessage objects
        normalized = []
        for msg in msgs:
            if isinstance(msg, BaseMessage):
                normalized.append(msg)
            elif isinstance(msg, tuple) and len(msg) == 2:
                role, content = msg
                if role in ("user", "human"):
                    normalized.append(HumanMessage(content=content))
                elif role == "system":
                    normalized.append(SystemMessage(content=content))
                elif role in ("assistant", "ai"):
                    normalized.append(AIMessage(content=content))
                else:
                    normalized.append(HumanMessage(content=str(content)))
            else:
                normalized.append(HumanMessage(content=str(msg)))

        return {"messages": normalized, "user_instructions": ui}

    def preprocess_routing(state: PrimaryState):
        print("---PREPROCESS ROUTER---")
        question = state.get("user_instructions")
        try:
            response = routing_preprocessor.invoke({"user_instructions": question})
        except Exception:
            response = {
                "user_instructions_sql_database": question,
                "user_instructions_data_visualization": None,
                "routing_preprocessor_decision": "table",
            }

        return {
            "user_instructions_sql_database": response.get(
                "user_instructions_sql_database", question
            ),
            "user_instructions_data_visualization": response.get(
                "user_instructions_data_visualization"
            ),
            "routing_preprocessor_decision": response.get(
                "routing_preprocessor_decision", "table"
            ),
        }

    def router_chart_or_table(state: PrimaryState):
        print("---ROUTER: CHART OR TABLE---")
        return (
            "chart"
            if state.get("routing_preprocessor_decision") == "chart"
            else "table"
        )

    def invoke_sql_database_agent(state: PrimaryState):
        response = sql_database_agent.invoke(
            {
                "user_instructions": state.get("user_instructions_sql_database"),
                "max_retries": state.get("max_retries"),
                "retry_count": state.get("retry_count"),
            }
        )

        return {
            "messages": response.get("messages"),
            "data_sql": response.get("data_sql"),
            "sql_query_code": response.get("sql_query_code"),
            "sql_database_function": response.get("sql_database_function"),
        }

    def invoke_data_visualization_agent(state: PrimaryState):
        data_sql = state.get("data_sql")
        if data_sql is None or (isinstance(data_sql, dict) and len(data_sql) == 0):
            return {
                "messages": [],
                "data_visualization_function": None,
                "plotly_graph": None,
                "plotly_error": "No data returned from SQL step; visualization skipped.",
            }
        response = data_visualization_agent.invoke(
            {
                "user_instructions": state.get("user_instructions_data_visualization"),
                "data_raw": data_sql,
                "max_retries": state.get("max_retries"),
                "retry_count": state.get("retry_count"),
            }
        )

        return {
            "messages": response.get("messages"),
            "data_visualization_function": response.get("data_visualization_function"),
            "plotly_graph": response.get("plotly_graph"),
            "plotly_error": response.get("data_visualization_error"),
        }

    def finalize_output(state: PrimaryState):
        print("---FINALIZE OUTPUT---")
        route = state.get("routing_preprocessor_decision", "table")
        data_sql = state.get("data_sql")
        plot = state.get("plotly_graph")
        plot_err = state.get("plotly_error")
        sql_query = state.get("sql_query_code")
        parts = []
        if sql_query:
            parts.append("SQL query generated and executed.")
        if data_sql:
            try:
                df = pd.DataFrame(data_sql)
                parts.append(f"Returned table shape: {df.shape[0]} rows x {df.shape[1]} cols.")
            except Exception:
                parts.append("Returned table available.")
        if route == "chart":
            if plot:
                parts.append("Chart created from query results.")
            elif plot_err:
                parts.append(f"Chart not created: {plot_err}")
        summary = " ".join(parts) or "Workflow completed."
        ai_msg = AIMessage(content=summary, role=AGENT_NAME)
        msgs = state.get("messages", [])
        msgs = msgs + [ai_msg]
        return {"messages": msgs}

    workflow = StateGraph(PrimaryState)

    workflow.add_node("prepare_messages", prepare_messages)
    workflow.add_node("routing_preprocessor", preprocess_routing)
    workflow.add_node("sql_database_agent", invoke_sql_database_agent)
    workflow.add_node("data_visualization_agent", invoke_data_visualization_agent)
    workflow.add_node("finalize_output", finalize_output)

    workflow.add_edge(START, "prepare_messages")
    workflow.add_edge("prepare_messages", "routing_preprocessor")
    workflow.add_edge("routing_preprocessor", "sql_database_agent")

    workflow.add_conditional_edges(
        "sql_database_agent",
        router_chart_or_table,
        {"chart": "data_visualization_agent", "table": "finalize_output"},
    )

    workflow.add_edge("data_visualization_agent", "finalize_output")
    workflow.add_edge("finalize_output", END)

    app = workflow.compile(checkpointer=checkpointer, name=AGENT_NAME)

    return app
