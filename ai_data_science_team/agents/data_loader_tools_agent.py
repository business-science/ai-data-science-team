from typing import Any, Optional, Annotated, Sequence, List, Dict
import operator

import pandas as pd

from IPython.display import Markdown

from langchain_core.messages import BaseMessage, AIMessage

from langgraph.prebuilt import ToolNode

try:
    from langgraph.prebuilt import create_react_agent
    from langgraph.prebuilt.chat_agent_executor import AgentState
except ImportError:
    from langchain.agents import create_react_agent, AgentState

from langgraph.types import Checkpointer
from langgraph.graph import START, END, StateGraph

from ai_data_science_team.templates import BaseAgent
from ai_data_science_team.utils.regex import format_agent_name
from ai_data_science_team.tools.data_loader import (
    load_directory,
    load_file,
    list_directory_contents,
    list_directory_recursive,
    get_file_info,
    search_files_by_pattern,
)
from ai_data_science_team.utils.messages import get_tool_call_names

AGENT_NAME = "data_loader_tools_agent"

tools = [
    load_directory,
    load_file,
    list_directory_contents,
    list_directory_recursive,
    get_file_info,
    search_files_by_pattern,
]


class DataLoaderToolsAgent(BaseAgent):
    """
    A Data Loader Agent that can interact with data loading tools and search for files in your file system.

    Parameters:
    ----------
    model : langchain.llms.base.LLM
        The language model used to generate the tool calling agent.
    react_agent_kwargs : dict
        Additional keyword arguments to pass to the create_react_agent function.
    invoke_react_agent_kwargs : dict
        Additional keyword arguments to pass to the invoke method of the react agent.
    checkpointer : langgraph.types.Checkpointer
        A checkpointer to use for saving and loading the agent's state.

    Methods:
    --------
    update_params(**kwargs)
        Updates the agent's parameters and rebuilds the compiled graph.
    ainvoke_agent(user_instructions: str=None, **kwargs)
        Runs the agent with the given user instructions asynchronously.
    invoke_agent(user_instructions: str=None, **kwargs)
        Runs the agent with the given user instructions.
    get_internal_messages(markdown: bool=False)
        Returns the internal messages from the agent's response.
    get_artifacts(as_dataframe: bool=False)
        Returns the MLflow artifacts from the agent's response.
    get_ai_message(markdown: bool=False)
        Returns the AI message from the agent's response.

    """

    def __init__(
        self,
        model: Any,
        create_react_agent_kwargs: Optional[Dict] = {},
        invoke_react_agent_kwargs: Optional[Dict] = {},
        checkpointer: Optional[Checkpointer] = None,
    ):
        self._params = {
            "model": model,
            "create_react_agent_kwargs": create_react_agent_kwargs,
            "invoke_react_agent_kwargs": invoke_react_agent_kwargs,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        """
        Creates the compiled graph for the agent.
        """
        self.response = None
        return make_data_loader_tools_agent(**self._params)

    def update_params(self, **kwargs):
        """
        Updates the agent's parameters and rebuilds the compiled graph.
        """
        for k, v in kwargs.items():
            self._params[k] = v
        self._compiled_graph = self._make_compiled_graph()

    async def ainvoke_agent(self, user_instructions: str = None, **kwargs):
        """
        Runs the agent with the given user instructions.

        Parameters:
        ----------
        user_instructions : str, optional
            The user instructions to pass to the agent.
        kwargs : dict, optional
            Additional keyword arguments to pass to the agents ainvoke method.

        """
        response = await self._compiled_graph.ainvoke(
            {
                "user_instructions": user_instructions,
            },
            **kwargs,
        )
        self.response = response
        return None

    def invoke_agent(self, user_instructions: str = None, **kwargs):
        """
        Runs the agent with the given user instructions.

        Parameters:
        ----------
        user_instructions : str, optional
            The user instructions to pass to the agent.
        kwargs : dict, optional
            Additional keyword arguments to pass to the agents invoke method.

        """
        response = self._compiled_graph.invoke(
            {
                "user_instructions": user_instructions,
            },
            **kwargs,
        )
        self.response = response
        return None

    def get_internal_messages(self, markdown: bool = False):
        """
        Returns the internal messages from the agent's response.
        """
        pretty_print = "\n\n".join(
            [
                f"### {msg.type.upper()}\n\nID: {msg.id}\n\nContent:\n\n{msg.content}"
                for msg in self.response["internal_messages"]
            ]
        )
        if markdown:
            return Markdown(pretty_print)
        else:
            return self.response["internal_messages"]

    def get_artifacts(self, as_dataframe: bool = False):
        """
        Returns the MLflow artifacts from the agent's response.
        """
        if as_dataframe:
            return pd.DataFrame(self.response["data_loader_artifacts"])
        else:
            return self.response["data_loader_artifacts"]

    def get_ai_message(self, markdown: bool = False):
        """
        Returns the AI message from the agent's response.
        """
        if markdown:
            return Markdown(self.response["messages"][0].content)
        else:
            return self.response["messages"][0].content

    def get_tool_calls(self):
        """
        Returns the tool calls made by the agent.
        """
        return self.response["tool_calls"]


def make_data_loader_tools_agent(
    model: Any,
    create_react_agent_kwargs: Optional[Dict] = {},
    invoke_react_agent_kwargs: Optional[Dict] = {},
    checkpointer: Optional[Checkpointer] = None,
):
    """
    Creates a Data Loader Agent that can interact with data loading tools.

    Parameters:
    ----------
    model : langchain.llms.base.LLM
        The language model used to generate the tool calling agent.
    react_agent_kwargs : dict
        Additional keyword arguments to pass to the create_react_agent function.
    invoke_react_agent_kwargs : dict
        Additional keyword arguments to pass to the invoke method of the react agent.
    checkpointer : langgraph.types.Checkpointer
        A checkpointer to use for saving and loading the agent's state.

    Returns:
    --------
    app : langchain.graphs.CompiledStateGraph
        An agent that can interact with data loading tools.
    """

    class GraphState(AgentState):
        internal_messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        data_loader_artifacts: dict
        tool_calls: List[str]

    def data_loader_agent(state):
        print(format_agent_name(AGENT_NAME))
        print("    ")

        print("    * RUN REACT TOOL-CALLING AGENT")

        tool_node = ToolNode(tools=tools)

        data_loader_agent = create_react_agent(
            model,
            tools=tool_node,
            state_schema=GraphState,
            checkpointer=checkpointer,
            **create_react_agent_kwargs,
        )

        response = data_loader_agent.invoke(
            {
                "messages": [("user", state["user_instructions"])],
            },
            invoke_react_agent_kwargs,
        )

        print("    * POST-PROCESS RESULTS")

        internal_messages = response["messages"]

        # Ensure there is at least one AI message
        if not internal_messages:
            return {
                "internal_messages": [],
                "mlflow_artifacts": None,
            }

        # Get the last AI message
        last_ai_message = AIMessage(internal_messages[-1].content, role=AGENT_NAME)

        # Get the last tool artifact safely
        last_tool_artifact = None
        if len(internal_messages) > 1:
            last_message = internal_messages[-2]  # Get second-to-last message
            if hasattr(last_message, "artifact"):  # Check if it has an "artifact"
                last_tool_artifact = last_message.artifact
            elif isinstance(last_message, dict) and "artifact" in last_message:
                last_tool_artifact = last_message["artifact"]

        tool_calls = get_tool_call_names(internal_messages)

        return {
            "messages": [last_ai_message],
            "internal_messages": internal_messages,
            "data_loader_artifacts": last_tool_artifact,
            "tool_calls": tool_calls,
        }

    workflow = StateGraph(GraphState)

    workflow.add_node("data_loader_agent", data_loader_agent)

    workflow.add_edge(START, "data_loader_agent")
    workflow.add_edge("data_loader_agent", END)

    app = workflow.compile(
        checkpointer=checkpointer,
        name=AGENT_NAME,
    )

    return app
