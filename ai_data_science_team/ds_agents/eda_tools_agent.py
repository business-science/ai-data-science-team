from typing_extensions import Any, Optional, Annotated, Sequence, Dict
import operator
import pandas as pd

from IPython.display import Markdown

from langchain_core.messages import BaseMessage, AIMessage

try:
    from langgraph.prebuilt import create_react_agent
    from langgraph.prebuilt.chat_agent_executor import AgentState
except ImportError:
    from langchain.agents import create_react_agent, AgentState

from langgraph.graph import START, END, StateGraph
from langgraph.types import Checkpointer

from ai_data_science_team.templates import BaseAgent
from ai_data_science_team.utils.regex import format_agent_name

from ai_data_science_team.tools.eda import (
    explain_data,
    describe_dataset,
    visualize_missing,
    generate_correlation_funnel,
    generate_sweetviz_report,
    generate_dtale_report,
)
from ai_data_science_team.utils.messages import get_tool_call_names


AGENT_NAME = "exploratory_data_analyst_agent"

# Updated tool list for EDA
EDA_TOOLS = [
    explain_data,
    describe_dataset,
    visualize_missing,
    generate_correlation_funnel,
    generate_sweetviz_report,
    generate_dtale_report,
]


class EDAToolsAgent(BaseAgent):
    """
    An Exploratory Data Analysis Tools Agent that interacts with EDA tools to generate summary statistics,
    missing data visualizations, correlation funnels, EDA reports, etc.

    Parameters:
    ----------
    model : langchain.llms.base.LLM
        The language model for generating the tool-calling agent.
    create_react_agent_kwargs : dict
        Additional kwargs for create_react_agent.
    invoke_react_agent_kwargs : dict
        Additional kwargs for agent invocation.
    checkpointer : Checkpointer, optional
        The checkpointer for the agent.
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
        Creates the compiled state graph for the EDA agent.
        """
        self.response = None
        return make_eda_tools_agent(**self._params)

    def update_params(self, **kwargs):
        """
        Updates the agent's parameters and rebuilds the compiled graph.
        """
        for k, v in kwargs.items():
            self._params[k] = v
        self._compiled_graph = self._make_compiled_graph()

    async def ainvoke_agent(
        self, user_instructions: str = None, data_raw: pd.DataFrame = None, **kwargs
    ):
        """
        Asynchronously runs the agent with user instructions and data.

        Parameters:
        ----------
        user_instructions : str, optional
            The instructions for the agent.
        data_raw : pd.DataFrame, optional
            The input data as a DataFrame.
        """
        messages = kwargs.pop("messages", None)
        if messages is None:
            messages = [("user", user_instructions)]
        response = await self._compiled_graph.ainvoke(
            {
                "messages": messages,
                "user_instructions": user_instructions,
                "data_raw": data_raw.to_dict() if data_raw is not None else None,
            },
            **kwargs,
        )
        self.response = response
        return None

    def invoke_agent(
        self, user_instructions: str = None, data_raw: pd.DataFrame = None, **kwargs
    ):
        """
        Synchronously runs the agent with user instructions and data.

        Parameters:
        ----------
        user_instructions : str, optional
            The instructions for the agent.
        data_raw : pd.DataFrame, optional
            The input data as a DataFrame.
        """
        messages = kwargs.pop("messages", None)
        if messages is None:
            messages = [("user", user_instructions)]
        response = self._compiled_graph.invoke(
            {
                "messages": messages,
                "user_instructions": user_instructions,
                "data_raw": data_raw.to_dict() if data_raw is not None else None,
            },
            **kwargs,
        )
        self.response = response
        return None

    def invoke_messages(
        self, messages: Sequence[BaseMessage], data_raw: pd.DataFrame = None, **kwargs
    ):
        """
        Runs the agent with an explicit message list (preferred for supervisors/teams).
        """
        response = self._compiled_graph.invoke(
            {
                "messages": messages,
                "user_instructions": None,
                "data_raw": data_raw.to_dict() if data_raw is not None else None,
            },
            **kwargs,
        )
        self.response = response
        return None

    async def ainvoke_messages(
        self, messages: Sequence[BaseMessage], data_raw: pd.DataFrame = None, **kwargs
    ):
        """
        Async version of invoke_messages.
        """
        response = await self._compiled_graph.ainvoke(
            {
                "messages": messages,
                "user_instructions": None,
                "data_raw": data_raw.to_dict() if data_raw is not None else None,
            },
            **kwargs,
        )
        self.response = response
        return None

    def get_internal_messages(self, markdown: bool = False):
        """
        Returns internal messages from the agent response.
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
        Returns the EDA artifacts from the agent response.
        """
        artifact = None
        if self.response:
            artifact = self.response.get("eda_artifacts")
        # Back-compat: if there is exactly one tool artifact, unwrap it so older
        # callers that expected a flat dict still work.
        if (
            not as_dataframe
            and isinstance(artifact, dict)
            and len(artifact) == 1
            and isinstance(next(iter(artifact.values())), dict)
        ):
            artifact = next(iter(artifact.values()))
        if not as_dataframe:
            return artifact
        # Try to coerce to DataFrame sensibly
        if isinstance(artifact, pd.DataFrame):
            return artifact
        if isinstance(artifact, dict):
            # If this is a dict of tool_name -> artifact
            if any(isinstance(v, dict) for v in artifact.values()):
                converted = {}
                for k, v in artifact.items():
                    # Prefer the flattened describe output when present
                    if isinstance(v, dict) and "describe_df_flat" in v:
                        try:
                            converted[k] = pd.DataFrame(v["describe_df_flat"])
                            continue
                        except Exception:
                            pass
                    if (
                        isinstance(v, dict)
                        and "describe_df" in v
                        and isinstance(v["describe_df"], dict)
                    ):
                        converted[k] = pd.DataFrame.from_dict(v["describe_df"]).T
                    elif isinstance(v, dict) and all(
                        isinstance(val, dict) for val in v.values()
                    ):
                        converted[k] = pd.DataFrame(v).T
                    elif isinstance(v, dict):
                        converted[k] = pd.DataFrame(v)
                    else:
                        converted[k] = v
                return converted
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
        Returns the AI message from the agent response.
        """
        if not self.response or "messages" not in self.response:
            return None
        msgs = self.response.get("messages", [])
        last_ai = None
        for msg in reversed(msgs):
            role = getattr(msg, "role", None) or getattr(msg, "type", None)
            if role in ("assistant", "ai", AGENT_NAME):
                last_ai = msg
                break
        if last_ai is None and msgs:
            last_ai = msgs[-1]
        if last_ai is None:
            print("No AI message found in response.")
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


def make_eda_tools_agent(
    model: Any,
    create_react_agent_kwargs: Optional[Dict] = {},
    invoke_react_agent_kwargs: Optional[Dict] = {},
    checkpointer: Optional[Checkpointer] = None,
    log_tool_calls: bool = True,
):
    """
    Creates an Exploratory Data Analyst Agent that can interact with EDA tools.

    Parameters:
    ----------
    model : Any
        The language model used for tool-calling.
    create_react_agent_kwargs : dict
        Additional kwargs for create_react_agent.
    invoke_react_agent_kwargs : dict
        Additional kwargs for agent invocation.
    checkpointer : Checkpointer, optional
        The checkpointer for the agent.

    Returns:
    -------
    app : langgraph.graph.CompiledStateGraph
        The compiled state graph for the EDA agent.
    """

    class GraphState(AgentState):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        data_raw: dict
        eda_artifacts: dict
        tool_calls: list

    # Build the React subgraph once so it shows in .show(xray=1)
    react_agent = create_react_agent(
        model,
        tools=EDA_TOOLS,
        state_schema=GraphState,
        **create_react_agent_kwargs,
        checkpointer=checkpointer,
    )

    def prepare_messages(state: GraphState):
        print(format_agent_name(AGENT_NAME))
        print("    * PREPARE MESSAGES")
        if state.get("messages"):
            return {}
        return {"messages": [("user", state.get("user_instructions"))]}

    def run_react_agent(state: GraphState):
        print("    * RUN REACT TOOL-CALLING AGENT FOR EDA")
        data_raw = state.get("data_raw")
        if data_raw is None:
            print("    * No data_raw provided to EDA agent")
        else:
            try:
                n_rows = (
                    len(next(iter(data_raw.values())))
                    if isinstance(data_raw, dict)
                    else "?"
                )
            except Exception:
                n_rows = "?"
            print(f"    * data_raw rows: {n_rows}")

        system_hint = (
            "You are an EDA agent. You have access to the dataset in state as data_raw. "
            "Use the provided EDA tools to summarize or visualize the data, then return concise results."
        )
        base_messages = state.get("messages", []) or [
            ("user", state.get("user_instructions"))
        ]
        messages = [("system", system_hint)] + base_messages

        input_payload = {
            "messages": messages,
            "data_raw": data_raw,
        }
        return react_agent.invoke(input_payload, invoke_react_agent_kwargs)

    def post_process(state: GraphState):
        print("    * POST-PROCESSING EDA RESULTS")

        internal_messages = state.get("messages", [])
        if not internal_messages:
            return {"messages": [], "eda_artifacts": None, "tool_calls": []}

        last_ai_message = None
        for msg in reversed(internal_messages):
            role = getattr(msg, "role", None) or getattr(msg, "type", None)
            if role in ("assistant", "ai"):
                last_ai_message = AIMessage(
                    content=getattr(msg, "content", ""),
                    name=AGENT_NAME,
                )
                break
        if last_ai_message is None:
            last_ai_message = AIMessage(
                content=getattr(internal_messages[-1], "content", ""),
                name=AGENT_NAME,
            )
        # If the AI content is empty, synthesize a minimal summary so users see something helpful.
        if not getattr(last_ai_message, "content", "").strip():
            last_ai_message = AIMessage(
                content="Summary statistics have been computed for the dataset. See artifact preview below.",
                name=AGENT_NAME,
            )

        # Collect artifacts per tool if possible
        artifacts = {}
        for msg in internal_messages:
            art = getattr(msg, "artifact", None)
            name = getattr(msg, "name", None)
            if art is not None:
                key = name or f"artifact_{len(artifacts) + 1}"
                artifacts[key] = art
            elif isinstance(msg, dict) and "artifact" in msg:
                key = msg.get("name") or f"artifact_{len(artifacts) + 1}"
                artifacts[key] = msg["artifact"]
        # Fallback to last artifact
        last_tool_artifact = None
        if artifacts:
            last_tool_artifact = list(artifacts.values())[-1]

        # If the AI message content is sparse and we have an artifact, append a brief summary
        if last_tool_artifact is not None:
            try:
                summary_snippet = None
                if isinstance(last_tool_artifact, str):
                    summary_snippet = last_tool_artifact[:1000]
                elif isinstance(last_tool_artifact, dict):
                    # Try to detect flattened describe output: {'describe_df_flat': {...}}
                    if "describe_df_flat" in last_tool_artifact and isinstance(
                        last_tool_artifact["describe_df_flat"], dict
                    ):
                        df_preview = pd.DataFrame(
                            last_tool_artifact["describe_df_flat"]
                        ).head()
                        summary_snippet = df_preview.to_markdown(index=False)
                    # Try legacy describe_df
                    elif "describe_df" in last_tool_artifact and isinstance(
                        last_tool_artifact["describe_df"], dict
                    ):
                        df_preview = pd.DataFrame.from_dict(
                            last_tool_artifact["describe_df"]
                        ).T.head()
                        summary_snippet = df_preview.to_markdown()
                    elif all(isinstance(v, dict) for v in last_tool_artifact.values()):
                        df_preview = pd.DataFrame(last_tool_artifact).T.head()
                        summary_snippet = df_preview.to_markdown()
                    else:
                        df_preview = pd.DataFrame(last_tool_artifact).head()
                        summary_snippet = df_preview.to_markdown(index=False)
                elif isinstance(last_tool_artifact, list):
                    summary_snippet = str(last_tool_artifact)[:1000]
                else:
                    summary_snippet = str(last_tool_artifact)[:1000]
                if summary_snippet:
                    last_ai_message = AIMessage(
                        content=f"{last_ai_message.content}\n\nArtifact preview:\n{summary_snippet}",
                        name=AGENT_NAME,
                    )
            except Exception:
                pass

        tool_calls = get_tool_call_names(internal_messages)
        if log_tool_calls and tool_calls:
            for name in tool_calls:
                print(f"    * Tool: {name}")

        return {
            "messages": [last_ai_message],
            "internal_messages": internal_messages,
            "eda_artifacts": artifacts if artifacts else last_tool_artifact,
            "tool_calls": tool_calls,
        }

    workflow = StateGraph(GraphState)
    workflow.add_node("prepare_messages", prepare_messages)
    workflow.add_node("react_agent", react_agent)
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
