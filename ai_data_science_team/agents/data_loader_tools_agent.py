from typing_extensions import Any, Optional, Annotated, Sequence, List, Dict
import operator

import pandas as pd
import os

from IPython.display import Markdown

from langchain_core.messages import BaseMessage, AIMessage

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
        log_tool_calls: bool = True,
    ):
        self._params = {
            "model": model,
            "create_react_agent_kwargs": create_react_agent_kwargs,
            "invoke_react_agent_kwargs": invoke_react_agent_kwargs,
            "checkpointer": checkpointer,
            "log_tool_calls": log_tool_calls,
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
        messages = kwargs.pop("messages", None)
        if messages is None:
            messages = [("user", user_instructions)]
        response = await self._compiled_graph.ainvoke(
            {
                "messages": messages,
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
        messages = kwargs.pop("messages", None)
        if messages is None:
            messages = [("user", user_instructions)]
        response = self._compiled_graph.invoke(
            {
                "messages": messages,
                "user_instructions": user_instructions,
            },
            **kwargs,
        )
        self.response = response
        return None

    def invoke_messages(self, messages: Sequence[BaseMessage], **kwargs):
        """
        Runs the agent given an explicit message list (preferred for supervisors/teams).
        """
        response = self._compiled_graph.invoke(
            {
                "messages": messages,
                "user_instructions": None,
            },
            **kwargs,
        )
        self.response = response
        return None

    async def ainvoke_messages(self, messages: Sequence[BaseMessage], **kwargs):
        """
        Async version of invoke_messages for supervisors/teams.
        """
        response = await self._compiled_graph.ainvoke(
            {
                "messages": messages,
                "user_instructions": None,
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
        artifact = None
        if self.response:
            artifact = self.response.get("data_loader_artifacts")

        # Back-compat: if exactly one tool artifact and caller didn't request DF, unwrap to legacy shape
        if (
            not as_dataframe
            and isinstance(artifact, dict)
            and len(artifact) == 1
            and isinstance(next(iter(artifact.values())), dict)
        ):
            artifact = next(iter(artifact.values()))

        # Handle directory-style artifacts: {filename: {"status","data","error"}}
        if isinstance(artifact, dict) and all(
            isinstance(v, dict) and "data" in v for v in artifact.values()
        ):
            dataframes = {
                k: pd.DataFrame(v["data"])
                if v.get("data") is not None
                else pd.DataFrame()
                for k, v in artifact.items()
            }
            return dataframes if as_dataframe else dataframes

        if not as_dataframe:
            return artifact

        # Try to coerce to a DataFrame sensibly
        if isinstance(artifact, pd.DataFrame):
            return artifact
        if isinstance(artifact, dict):
            if "data" in artifact:
                try:
                    return pd.DataFrame(artifact["data"])
                except Exception:
                    return pd.DataFrame([artifact])
            try:
                return pd.DataFrame(artifact)
            except Exception:
                return pd.DataFrame({"artifact": [artifact]})
        if isinstance(artifact, list):
            try:
                return pd.DataFrame(artifact)
            except Exception:
                return pd.DataFrame({"artifact": artifact})

        return pd.DataFrame()

    def get_ai_message(self, markdown: bool = False):
        """
        Returns the AI message from the agent's response.
        """
        if not self.response or "messages" not in self.response:
            return None

        msgs = self.response.get("messages", [])
        last_ai = None
        for msg in reversed(msgs):
            role = getattr(msg, "role", None) or getattr(msg, "type", None)
            if role in ("assistant", "ai"):
                last_ai = msg
                break
        if last_ai is None and msgs:
            last_ai = msgs[-1]

        if last_ai is None:
            return None

        content = getattr(last_ai, "content", "")
        if markdown:
            return Markdown(content)
        return content

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
    log_tool_calls: bool = True,
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

    react_agent = create_react_agent(
        model,
        tools=tools,
        state_schema=AgentState,
        checkpointer=checkpointer,
        **create_react_agent_kwargs,
    )

    class GraphState(AgentState):
        user_instructions: str
        messages: Annotated[Sequence[BaseMessage], operator.add]
        data_loader_artifacts: dict
        tool_calls: List[str]

    def prepare_messages(state: GraphState):
        print(format_agent_name(AGENT_NAME))
        print("    * PREPARE MESSAGES")
        if state.get("messages"):
            return {}
        return {"messages": [("user", state.get("user_instructions"))]}

    def run_react_agent(state: GraphState):
        print("    * RUN REACT TOOL-CALLING AGENT")
        system_hint = (
            "You are a data loader + file system tools agent.\n"
            "- If the user asks to LIST files (e.g., 'what files are in ./data', 'list only CSVs'), "
            "use listing/search tools (search_files_by_pattern, list_directory_contents, list_directory_recursive). "
            "Do NOT load file contents.\n"
            "- Use load_file only when the user explicitly asks to load/read a specific file.\n"
            "- Use load_directory only when the user explicitly asks to load ALL files in a directory.\n"
            "Prefer search_files_by_pattern for extension filters (e.g., pattern='*.csv')."
        )
        base_messages = state.get("messages", []) or [
            ("user", state.get("user_instructions"))
        ]
        messages = [("system", system_hint)] + base_messages
        input_payload = {"messages": messages}
        return react_agent.invoke(input_payload, invoke_react_agent_kwargs)

    def post_process(state: GraphState):
        print("    * POST-PROCESS RESULTS")
        internal_messages = state.get("messages", [])

        if not internal_messages:
            return {
                "messages": [],
                "data_loader_artifacts": None,
                "tool_calls": [],
            }

        # Prefer the last assistant/ai message; fall back to last message
        last_ai = None
        for msg in reversed(internal_messages):
            role = getattr(msg, "role", None) or getattr(msg, "type", None)
            if role in ("assistant", "ai"):
                last_ai = msg
                break
        if last_ai is None:
            last_ai = internal_messages[-1]

        last_ai_message = AIMessage(getattr(last_ai, "content", ""), role=AGENT_NAME)

        # Collect artifacts per tool if possible
        artifacts = {}
        last_tool_artifact = None
        key_counts: dict[str, int] = {}

        def _next_key(base: str) -> str:
            b = (base or "artifact").strip() or "artifact"
            key_counts[b] = key_counts.get(b, 0) + 1
            n = key_counts[b]
            return b if n == 1 else f"{b}_{n}"

        for msg in internal_messages:
            art = getattr(msg, "artifact", None)
            name = getattr(msg, "name", None)
            if art is not None:
                key = _next_key(str(name) if name else "artifact")
                artifacts[key] = art
                last_tool_artifact = art
            elif isinstance(msg, dict) and "artifact" in msg:
                key = _next_key(str(msg.get("name")) if msg.get("name") else "artifact")
                artifacts[key] = msg["artifact"]
                last_tool_artifact = msg["artifact"]

        tool_calls = get_tool_call_names(internal_messages)
        if tool_calls and log_tool_calls:
            for name in tool_calls:
                # try to include artifact path if present in the prior message
                path_hint = ""
                # search for first artifact-bearing message
                for msg in reversed(internal_messages):
                    art = getattr(msg, "artifact", None)
                    if art:
                        # if the artifact looks like a dict with path info
                        if isinstance(art, dict):
                            for k, v in art.items():
                                if isinstance(v, str) and os.path.exists(v):
                                    path_hint = f" | {v}"
                                    break
                        break
                print(f"    * Tool: {name}{path_hint}")
            try:
                if isinstance(artifacts, dict) and artifacts:
                    keys = list(artifacts.keys())
                    print(f"    * Artifacts captured: {keys}")
                else:
                    print("    * Artifacts captured: none")
            except Exception:
                pass

        return {
            "messages": [last_ai_message],
            "internal_messages": internal_messages,
            "data_loader_artifacts": artifacts if artifacts else last_tool_artifact,
            "tool_calls": tool_calls,
        }

    workflow = StateGraph(GraphState)

    workflow.add_node("prepare_messages", prepare_messages)
    workflow.add_node("react_agent", run_react_agent)
    workflow.add_node("post_process", post_process)

    workflow.add_edge(START, "prepare_messages")
    workflow.add_edge("prepare_messages", "react_agent")
    workflow.add_edge("react_agent", "post_process")
    workflow.add_edge("post_process", END)

    app = workflow.compile(
        checkpointer=checkpointer,
        name=AGENT_NAME,
    )

    return app
