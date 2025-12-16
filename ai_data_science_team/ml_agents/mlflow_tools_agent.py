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

from langgraph.types import Checkpointer
from langgraph.graph import START, END, StateGraph

from ai_data_science_team.templates import BaseAgent
from ai_data_science_team.utils.regex import format_agent_name
from ai_data_science_team.tools.mlflow import (
    mlflow_search_experiments,
    mlflow_search_runs,
    mlflow_create_experiment,
    mlflow_set_tags,
    mlflow_log_params,
    mlflow_log_metrics,
    mlflow_log_table,
    mlflow_log_dict,
    mlflow_log_figure,
    mlflow_log_artifact,
    mlflow_predict_from_run_id,
    mlflow_launch_ui,
    mlflow_stop_ui,
    mlflow_list_artifacts,
    mlflow_download_artifacts,
    mlflow_list_registered_models,
    mlflow_search_registered_models,
    mlflow_get_model_version_details,
    mlflow_get_run_details,
    mlflow_transition_model_version_stage,
    mlflow_tracking_info,
    mlflow_ui_status,
)
from ai_data_science_team.utils.messages import get_tool_call_names

AGENT_NAME = "mlflow_tools_agent"

# TOOL SETUP
tools = [
    mlflow_search_experiments,
    mlflow_search_runs,
    mlflow_create_experiment,
    mlflow_set_tags,
    mlflow_log_params,
    mlflow_log_metrics,
    mlflow_log_table,
    mlflow_log_dict,
    mlflow_log_figure,
    mlflow_log_artifact,
    mlflow_predict_from_run_id,
    mlflow_launch_ui,
    mlflow_stop_ui,
    mlflow_list_artifacts,
    mlflow_download_artifacts,
    mlflow_list_registered_models,
    mlflow_search_registered_models,
    mlflow_get_model_version_details,
    mlflow_get_run_details,
    mlflow_transition_model_version_stage,
    mlflow_tracking_info,
    mlflow_ui_status,
]


class MLflowToolsAgent(BaseAgent):
    """
    An agent that can interact with MLflow by calling tools.

    Current tools include:
    - List Experiments
    - Search Runs
    - Create Experiment
    - Predict (from a Run ID)

    Parameters:
    ----------
    model : langchain.llms.base.LLM
        The language model used to generate the tool calling agent.
    mlfow_tracking_uri : str, optional
        The tracking URI for MLflow. Defaults to None.
    mlflow_registry_uri : str, optional
        The registry URI for MLflow. Defaults to None.
    react_agent_kwargs : dict
        Additional keyword arguments to pass to the create_react_agent function.
    invoke_react_agent_kwargs : dict
        Additional keyword arguments to pass to the invoke method of the react agent.
    checkpointer : langchain.checkpointing.Checkpointer, optional
        A checkpointer to use for saving and loading the agent's state. Defaults to None.

    Methods:
    --------
    update_params(**kwargs):
        Updates the agent's parameters and rebuilds the compiled graph.
    ainvoke_agent(user_instructions: str=None, data_raw: pd.DataFrame=None, **kwargs):
        Asynchronously runs the agent with the given user instructions.
    invoke_agent(user_instructions: str=None, data_raw: pd.DataFrame=None, **kwargs):
        Runs the agent with the given user instructions.
    get_internal_messages(markdown: bool=False):
        Returns the internal messages from the agent's response.
    get_mlflow_artifacts(as_dataframe: bool=False):
        Returns the MLflow artifacts from the agent's response.
    get_ai_message(markdown: bool=False):
        Returns the AI message from the agent's response



    Examples:
    --------
    ```python
    from ai_data_science_team.ml_agents import MLflowToolsAgent

    mlflow_agent = MLflowToolsAgent(llm)

    mlflow_agent.invoke_agent(user_instructions="List the MLflow experiments")

    mlflow_agent.get_response()

    mlflow_agent.get_internal_messages(markdown=True)

    mlflow_agent.get_ai_message(markdown=True)

    mlflow_agent.get_mlflow_artifacts(as_dataframe=True)

    ```

    Returns
    -------
    MLflowToolsAgent : langchain.graphs.CompiledStateGraph
        An instance of the MLflow Tools Agent.

    """

    def __init__(
        self,
        model: Any,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_registry_uri: Optional[str] = None,
        create_react_agent_kwargs: Optional[Dict] = {},
        invoke_react_agent_kwargs: Optional[Dict] = {},
        checkpointer: Optional[Checkpointer] = None,
        log_tool_calls: bool = True,
    ):
        self._params = {
            "model": model,
            "mlflow_tracking_uri": mlflow_tracking_uri,
            "mlflow_registry_uri": mlflow_registry_uri,
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
        return make_mlflow_tools_agent(**self._params)

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
        Runs the agent with the given user instructions.

        Parameters:
        ----------
        user_instructions : str, optional
            The user instructions to pass to the agent.
        data_raw : pd.DataFrame, optional
            The data to pass to the agent. Used for prediction and tool calls where data is required.
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
        Runs the agent with the given user instructions.

        Parameters:
        ----------
        user_instructions : str, optional
            The user instructions to pass to the agent.
        data_raw : pd.DataFrame, optional
            The raw data to pass to the agent. Used for prediction and tool calls where data is required.

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

    def get_mlflow_artifacts(self, as_dataframe: bool = False):
        """
        Returns the MLflow artifacts from the agent's response.
        """
        artifact = None
        if self.response:
            artifact = self.response.get("mlflow_artifacts")

        # Back-compat: if exactly one tool artifact and caller didn't request DF, unwrap to legacy shape
        if (
            not as_dataframe
            and isinstance(artifact, dict)
            and len(artifact) == 1
            and isinstance(next(iter(artifact.values())), dict)
        ):
            artifact = next(iter(artifact.values()))

        if not as_dataframe:
            return artifact

        # Try to convert to DataFrame sensibly
        try:
            if isinstance(artifact, dict):
                if isinstance(artifact.get("runs"), list):
                    return pd.DataFrame(artifact.get("runs"))
                if isinstance(artifact.get("experiments"), list):
                    return pd.DataFrame(artifact.get("experiments"))
            return pd.DataFrame(artifact)
        except Exception:
            return pd.DataFrame({"artifact": [artifact]})

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


def make_mlflow_tools_agent(
    model: Any,
    mlflow_tracking_uri: str = None,
    mlflow_registry_uri: str = None,
    create_react_agent_kwargs: Optional[Dict] = {},
    invoke_react_agent_kwargs: Optional[Dict] = {},
    checkpointer: Optional[Checkpointer] = None,
    log_tool_calls: bool = True,
):
    """
    MLflow Tool Calling Agent

    Parameters:
    ----------
    model : Any
        The language model used to generate the agent.
    mlflow_tracking_uri : str, optional
        The tracking URI for MLflow. Defaults to None.
    mlflow_registry_uri : str, optional
        The registry URI for MLflow. Defaults to None.
    create_react_agent_kwargs : dict, optional
        Additional keyword arguments to pass to the agent's create_react_agent method.
    invoke_react_agent_kwargs : dict, optional
        Additional keyword arguments to pass to the agent's invoke method.
    checkpointer : langchain.checkpointing.Checkpointer, optional
        A checkpointer to use for saving and loading the agent's state. Defaults to None.

    Returns
    -------
    app : langchain.graphs.CompiledStateGraph
        A compiled state graph for the MLflow Tool Calling Agent.

    """

    try:
        import mlflow
    except ImportError:
        raise ImportError(
            "MLflow is not installed. Please install it by running: pip install mlflow"
        )

    if mlflow_tracking_uri is not None:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    if mlflow_registry_uri is not None:
        mlflow.set_registry_uri(mlflow_registry_uri)

    class GraphState(AgentState):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        data_raw: dict
        mlflow_artifacts: dict
        tool_calls: list

    # Build React subgraph once so it appears in .show(xray=1)
    react_agent = create_react_agent(
        model,
        tools=tools,
        state_schema=GraphState,
        checkpointer=checkpointer,
        **create_react_agent_kwargs,
    )
    if invoke_react_agent_kwargs:
        react_agent = react_agent.with_config(invoke_react_agent_kwargs)

    def prepare_messages(state: GraphState):
        print(format_agent_name(AGENT_NAME))
        print("    * PREPARE MESSAGES")
        if state.get("messages"):
            return {}
        system_hint = (
            "You are an MLflow tools agent. Use the provided MLflow tools to inspect or manage MLflow, "
            "and return concise results. The tracking URI and registry URI are already configured."
        )
        base_messages = [("user", state.get("user_instructions"))]
        return {"messages": [("system", system_hint)] + base_messages}

    def post_process(state: GraphState):
        print("    * POST-PROCESS RESULTS")
        internal_messages = state.get("messages", [])

        def _escape_md_cell(value: Any) -> str:
            s = "" if value is None else str(value)
            return s.replace("|", "\\|").replace("\n", " ").replace("\r", " ")

        def _records_to_md_table(
            records: list[dict], columns: list[str], max_rows: int = 15
        ) -> str:
            if not records:
                return ""
            cols = [c for c in columns if c]
            rows = records[: max_rows if max_rows and max_rows > 0 else len(records)]
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            body = [
                "| "
                + " | ".join(_escape_md_cell(r.get(c)) for c in cols)
                + " |"
                for r in rows
            ]
            return "\n".join([header, sep] + body)

        if not internal_messages:
            return {
                "messages": [],
                "mlflow_artifacts": None,
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

        last_ai_content = getattr(last_ai, "content", "")

        # Collect artifacts per tool if possible
        artifacts = {}
        last_tool_artifact = None
        for msg in internal_messages:
            art = getattr(msg, "artifact", None)
            name = getattr(msg, "name", None)
            if art is not None:
                key = name or f"artifact_{len(artifacts)+1}"
                artifacts[key] = art
                last_tool_artifact = art
            elif isinstance(msg, dict) and "artifact" in msg:
                key = msg.get("name") or f"artifact_{len(artifacts)+1}"
                artifacts[key] = msg["artifact"]
                last_tool_artifact = msg["artifact"]

        tool_calls = get_tool_call_names(internal_messages)
        if log_tool_calls and tool_calls:
            for name in tool_calls:
                print(f"    * Tool: {name}")

        def _format_artifacts_table(art: Any) -> str | None:
            if not isinstance(art, dict):
                return None

            parts: list[str] = []
            exp_art = art.get("mlflow_search_experiments")
            if isinstance(exp_art, dict) and isinstance(exp_art.get("experiments"), list):
                exps = exp_art.get("experiments") or []
                count = exp_art.get("count") if isinstance(exp_art.get("count"), int) else len(exps)
                parts.append(f"Found {count} MLflow experiment(s).")
                parts.append(
                    _records_to_md_table(
                        exps,
                        columns=[
                            "experiment_id",
                            "name",
                            "lifecycle_stage",
                            "creation_time",
                            "last_update_time",
                            "artifact_location",
                        ],
                        max_rows=15,
                    )
                )

            runs_art = art.get("mlflow_search_runs")
            if isinstance(runs_art, dict) and isinstance(runs_art.get("runs"), list):
                runs = runs_art.get("runs") or []
                count = runs_art.get("count") if isinstance(runs_art.get("count"), int) else len(runs)
                max_results = runs_art.get("max_results")
                header = (
                    f"Showing {count} most recent run(s) (max_results={max_results})."
                    if isinstance(max_results, int)
                    else f"Showing {count} run(s)."
                )
                parts.append(header)
                parts.append(
                    _records_to_md_table(
                        runs,
                        columns=[
                            "run_id",
                            "run_name",
                            "status",
                            "start_time",
                            "duration_seconds",
                            "has_model",
                            "model_uri",
                            "params_preview",
                            "metrics_preview",
                        ],
                        max_rows=15,
                    )
                )

            return "\n\n".join([p for p in parts if isinstance(p, str) and p.strip()]) or None

        formatted = _format_artifacts_table(artifacts) or _format_artifacts_table(last_tool_artifact)
        last_ai_message = AIMessage(
            content=formatted or last_ai_content,
            name=AGENT_NAME,
        )

        return {
            "messages": [last_ai_message],
            "internal_messages": internal_messages,
            "mlflow_artifacts": artifacts if artifacts else last_tool_artifact,
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
