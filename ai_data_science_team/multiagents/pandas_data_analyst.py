from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.types import Checkpointer
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages

from typing_extensions import TypedDict, Annotated, Sequence, Union

import pandas as pd
import json
from IPython.display import Markdown

from ai_data_science_team.templates import BaseAgent
from ai_data_science_team.agents import DataWranglingAgent, DataVisualizationAgent
from ai_data_science_team.utils.plotly import plotly_from_dict
from ai_data_science_team.utils.regex import (
    remove_consecutive_duplicates,
    get_generic_summary,
)

AGENT_NAME = "pandas_data_analyst"


class PandasDataAnalyst(BaseAgent):
    """
    PandasDataAnalyst is a multi-agent class that combines data wrangling and visualization capabilities.

    Parameters:
    -----------
    model:
        The language model to be used for the agents.
    data_wrangling_agent: DataWranglingAgent
        The Data Wrangling Agent for transforming raw data.
    data_visualization_agent: DataVisualizationAgent
        The Data Visualization Agent for generating plots.
    checkpointer: Checkpointer (optional)
        The checkpointer to save the state of the multi-agent system.

    Methods:
    --------
    ainvoke_agent(user_instructions, data_raw, **kwargs)
        Asynchronously invokes the multi-agent with user instructions and raw data.
    invoke_agent(user_instructions, data_raw, **kwargs)
        Synchronously invokes the multi-agent with user instructions and raw data.
    get_data_wrangled()
        Returns the wrangled data as a Pandas DataFrame.
    get_plotly_graph()
        Returns the Plotly graph as a Plotly object.
    get_data_wrangler_function(markdown=False)
        Returns the data wrangling function as a string, optionally in Markdown.
    get_data_visualization_function(markdown=False)
        Returns the data visualization function as a string, optionally in Markdown.
    """

    def __init__(
        self,
        model,
        data_wrangling_agent: DataWranglingAgent,
        data_visualization_agent: DataVisualizationAgent,
        checkpointer: Checkpointer = None,
    ):
        self._params = {
            "model": model,
            "data_wrangling_agent": data_wrangling_agent,
            "data_visualization_agent": data_visualization_agent,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        """Create or rebuild the compiled graph. Resets response to None."""
        self.response = None
        return make_pandas_data_analyst(
            model=self._params["model"],
            data_wrangling_agent=self._params["data_wrangling_agent"]._compiled_graph,
            data_visualization_agent=self._params[
                "data_visualization_agent"
            ]._compiled_graph,
            checkpointer=self._params["checkpointer"],
        )

    def update_params(self, **kwargs):
        """Updates parameters and rebuilds the compiled graph."""
        for k, v in kwargs.items():
            self._params[k] = v
        self._compiled_graph = self._make_compiled_graph()

    async def ainvoke_agent(
        self,
        user_instructions,
        data_raw: Union[pd.DataFrame, dict, list],
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
        """Asynchronously invokes the multi-agent."""
        response = await self._compiled_graph.ainvoke(
            {
                "user_instructions": user_instructions,
                "data_raw": self._convert_data_input(data_raw),
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        if response.get("messages"):
            response["messages"] = remove_consecutive_duplicates(response["messages"])
        self.response = response

    def invoke_agent(
        self,
        user_instructions,
        data_raw: Union[pd.DataFrame, dict, list],
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
        """Synchronously invokes the multi-agent."""
        response = self._compiled_graph.invoke(
            {
                "user_instructions": user_instructions,
                "data_raw": self._convert_data_input(data_raw),
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
        data_raw: Union[pd.DataFrame, dict, list],
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
        """
        Invoke the multi-agent with an explicit message list (preferred for supervisors/teams).
        """
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
                "data_raw": self._convert_data_input(data_raw),
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
        data_raw: Union[pd.DataFrame, dict, list],
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
                "data_raw": self._convert_data_input(data_raw),
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        if response.get("messages"):
            response["messages"] = remove_consecutive_duplicates(response["messages"])
        self.response = response
        return None

    def get_data_wrangled(self):
        """Returns the wrangled data as a Pandas DataFrame."""
        if self.response and self.response.get("data_wrangled"):
            return pd.DataFrame(self.response.get("data_wrangled"))

    def get_plotly_graph(self):
        """Returns the Plotly graph as a Plotly object."""
        if self.response and self.response.get("plotly_graph"):
            return plotly_from_dict(self.response.get("plotly_graph"))

    def get_data_wrangler_function(self, markdown=False):
        """Returns the data wrangling function as a string."""
        if self.response and self.response.get("data_wrangler_function"):
            code = self.response.get("data_wrangler_function")
            return Markdown(f"```python\n{code}\n```") if markdown else code

    def get_data_visualization_function(self, markdown=False):
        """Returns the data visualization function as a string."""
        if self.response and self.response.get("data_visualization_function"):
            code = self.response.get("data_visualization_function")
            return Markdown(f"```python\n{code}\n```") if markdown else code

    def get_workflow_summary(self, markdown=False):
        """Returns a summary of the workflow."""
        if self.response and self.response.get("messages"):
            agents = []
            seen = set()
            # Only list sub-agents (exclude assistant/system/user)
            allowed = {"data_wrangling_agent", "data_visualization_agent"}
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
                f"# Pandas Data Analyst Workflow Summary\n\nThis workflow contains {len(agents)} agents:\n\n"
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

    @staticmethod
    def _convert_data_input(
        data_raw: Union[pd.DataFrame, dict, list],
    ) -> Union[dict, list]:
        """Converts input data to the expected format (dict or list of dicts)."""
        if isinstance(data_raw, pd.DataFrame):
            return data_raw.to_dict()
        if isinstance(data_raw, dict):
            return data_raw
        if isinstance(data_raw, list):
            return [
                item.to_dict() if isinstance(item, pd.DataFrame) else item
                for item in data_raw
            ]
        raise ValueError(
            "data_raw must be a DataFrame, dict, or list of DataFrames/dicts"
        )


def make_pandas_data_analyst(
    model,
    data_wrangling_agent: CompiledStateGraph,
    data_visualization_agent: CompiledStateGraph,
    checkpointer: Checkpointer = None,
):
    """
    Creates a multi-agent system that wrangles data and optionally visualizes it.

    Parameters:
    -----------
    model: The language model to be used.
    data_wrangling_agent: CompiledStateGraph
        The Data Wrangling Agent.
    data_visualization_agent: CompiledStateGraph
        The Data Visualization Agent.
    checkpointer: Checkpointer (optional)
        The checkpointer to save the state.

    Returns:
    --------
    CompiledStateGraph: The compiled multi-agent system.
    """

    llm = model

    routing_preprocessor_prompt = PromptTemplate(
        template="""
        You are an expert in routing decisions for a Pandas Data Manipulation Wrangling Agent, a Charting Visualization Agent, and a Pandas Table Agent. Your job is to tell the agents which actions to perform and determine the correct routing for the incoming user question:
        
        1. Determine what the correct format for a Users Question should be for use with a Pandas Data Wrangling Agent based on the incoming user question. Anything related to data wrangling and manipulation should be passed along. Anything related to data analysis can be handled by the Pandas Agent. Anything that uses Pandas can be passed along. Tables can be returned from this agent. Don't pass along anything about plotting or visualization.
        2. Determine whether or not a chart should be generated or a table should be returned based on the users question.
        3. If a chart is requested, determine the correct format of a Users Question should be used with a Data Visualization Agent. Anything related to plotting and visualization should be passed along.
        
        Use the following criteria on how to route the the initial user question:
        
        From the incoming user question, remove any details about the format of the final response as either a Chart or Table and return only the important part of the incoming user question that is relevant for the Pandas Data Wrangling and Transformation agent. This will be the 'user_instructions_data_wrangling'. If 'None' is found, return the original user question.
        
        Next, determine if the user would like a data visualization ('chart') or a 'table' returned with the results of the Data Wrangling Agent. If unknown, not specified or 'None' is found, then select 'table'.  
        
        If a 'chart' is requested, return the 'user_instructions_data_visualization'. If 'None' is found, return None.
        
        Return JSON with 'user_instructions_data_wrangling', 'user_instructions_data_visualization' and 'routing_preprocessor_decision'.
        
        INITIAL_USER_QUESTION: {user_instructions}
        """,
        input_variables=["user_instructions"],
    )

    routing_preprocessor = routing_preprocessor_prompt | llm | JsonOutputParser()

    class PrimaryState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        user_instructions: str
        user_instructions_data_wrangling: str
        user_instructions_data_visualization: str
        routing_preprocessor_decision: str
        data_raw: Union[dict, list]
        data_wrangled: dict
        data_wrangler_function: str
        data_visualization_function: str
        plotly_graph: dict
        plotly_error: str
        max_retries: int
        retry_count: int

    def prepare_messages(state: PrimaryState):
        print("---PANDAS DATA ANALYST---")
        print("*************************")
        print("---PREPARE MESSAGES---")
        msgs = state.get("messages", [])
        ui = state.get("user_instructions")
        if not msgs:
            system_hint = (
                "You are a pandas data analyst orchestrator. Route the user's question to data wrangling "
                "and optional visualization. Prefer tables unless the user clearly requests a chart."
            )
            msgs = [("system", system_hint), ("user", ui)]
        if not ui:
            for msg in reversed(msgs):
                if getattr(msg, "type", None) == "human" or getattr(msg, "role", None) == "user":
                    ui = msg.content
                    break
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
                "user_instructions_data_wrangling": question,
                "user_instructions_data_visualization": None,
                "routing_preprocessor_decision": "table",
            }

        return {
            "user_instructions_data_wrangling": response.get(
                "user_instructions_data_wrangling", question
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

    def invoke_data_wrangling_agent(state: PrimaryState):
        response = data_wrangling_agent.invoke(
            {
                "user_instructions": state.get("user_instructions_data_wrangling"),
                "data_raw": state.get("data_raw"),
                "max_retries": state.get("max_retries"),
                "retry_count": state.get("retry_count"),
            }
        )

        return {
            "messages": response.get("messages"),
            "data_wrangled": response.get("data_wrangled"),
            "data_wrangler_function": response.get("data_wrangler_function"),
            "plotly_error": response.get("data_visualization_error"),
        }

    def invoke_data_visualization_agent(state: PrimaryState):
        data_for_viz = state.get("data_wrangled") or state.get("data_raw")
        if data_for_viz is None:
            return {
                "messages": [],
                "data_visualization_function": None,
                "plotly_graph": None,
                "plotly_error": "No data available for visualization; skipped.",
            }
        response = data_visualization_agent.invoke(
            {
                "user_instructions": state.get("user_instructions_data_visualization"),
                "data_raw": data_for_viz,
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
        data_wrangled = state.get("data_wrangled")
        plot = state.get("plotly_graph")
        plot_err = state.get("plotly_error")
        parts = []
        if data_wrangled:
            try:
                df = pd.DataFrame(data_wrangled)
                parts.append(f"Wrangled table shape: {df.shape[0]} rows x {df.shape[1]} cols.")
            except Exception:
                parts.append("Wrangled data available.")
        if route == "chart":
            if plot:
                parts.append("Chart created from wrangled data.")
            elif plot_err:
                parts.append(f"Chart not created: {plot_err}")
        summary = " ".join(parts) or "Workflow completed."
        ai_msg = AIMessage(content=summary, role="assistant")
        msgs = state.get("messages", [])
        msgs = msgs + [ai_msg]
        return {"messages": msgs}

    workflow = StateGraph(PrimaryState)

    workflow.add_node("prepare_messages", prepare_messages)
    workflow.add_node("routing_preprocessor", preprocess_routing)
    workflow.add_node("data_wrangling_agent", invoke_data_wrangling_agent)
    workflow.add_node("data_visualization_agent", invoke_data_visualization_agent)
    workflow.add_node("finalize_output", finalize_output)

    workflow.add_edge(START, "prepare_messages")
    workflow.add_edge("prepare_messages", "routing_preprocessor")
    workflow.add_edge("routing_preprocessor", "data_wrangling_agent")

    workflow.add_conditional_edges(
        "data_wrangling_agent",
        router_chart_or_table,
        {"chart": "data_visualization_agent", "table": "finalize_output"},
    )

    workflow.add_edge("data_visualization_agent", "finalize_output")
    workflow.add_edge("finalize_output", END)

    app = workflow.compile(checkpointer=checkpointer, name=AGENT_NAME)

    return app
