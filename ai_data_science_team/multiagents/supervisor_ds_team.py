from typing import Sequence, TypedDict, Annotated, Optional, Dict, Any
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from IPython.display import Markdown
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.types import Checkpointer


class SupervisorDSState(TypedDict):
    """
    Shared state for the supervisor-led data science team.
    """

    # Team conversation
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

    # Shared data/artifacts
    data_raw: Optional[dict]
    data_sql: Optional[dict]
    data_wrangled: Optional[dict]
    data_cleaned: Optional[dict]
    eda_artifacts: Optional[dict]
    viz_graph: Optional[dict]
    feature_data: Optional[dict]
    model_info: Optional[dict]
    mlflow_artifacts: Optional[dict]
    artifacts: Dict[str, Any]


def make_supervisor_ds_team(
    model: Any,
    data_loader_agent,
    data_wrangling_agent,
    data_cleaning_agent,
    eda_tools_agent,
    data_visualization_agent,
    sql_database_agent,
    feature_engineering_agent,
    h2o_ml_agent,
    mlflow_tools_agent,
    checkpointer: Optional[Checkpointer] = None,
    temperature: float = 0,
):
    """
    Build a supervisor-led data science team using existing sub-agents.

    Args:
        model: LLM (or model name) for the supervisor router.
        data_loader_agent: DataLoaderToolsAgent instance.
        data_wrangling_agent: DataWranglingAgent instance.
        data_cleaning_agent: DataCleaningAgent instance.
        eda_tools_agent: EDAToolsAgent instance.
        data_visualization_agent: DataVisualizationAgent instance.
        sql_database_agent: SQLDatabaseAgent instance.
        feature_engineering_agent: FeatureEngineeringAgent instance.
        h2o_ml_agent: H2OMLAgent instance.
        mlflow_tools_agent: MLflowToolsAgent instance.
        checkpointer: optional LangGraph checkpointer.
        temperature: supervisor routing temperature.
    """

    subagent_names = [
        "Data_Loader_Tools_Agent",
        "Data_Wrangling_Agent",
        "Data_Cleaning_Agent",
        "EDA_Tools_Agent",
        "Data_Visualization_Agent",
        "SQL_Database_Agent",
        "Feature_Engineering_Agent",
        "H2O_ML_Agent",
        "MLflow_Tools_Agent",
    ]

    llm = ChatOpenAI(model=model) if isinstance(model, str) else model
    llm.temperature = temperature

    system_prompt = """
You are a supervisor managing a data science team with these workers: {subagent_names}.

Routing guidance:
- Use Data_Loader_Tools_Agent to find/load data.
- Use Data_Wrangling_Agent for pandas prep/transform; Data_Cleaning_Agent for deeper cleaning.
- Use SQL_Database_Agent for SQL query generation/execution.
- Use EDA_Tools_Agent for describe/missingness/correlation/EDA reports.
- Use Data_Visualization_Agent for charts.
- Use Feature_Engineering_Agent for feature creation.
- Use H2O_ML_Agent for AutoML training/eval.
- Use MLflow_Tools_Agent for experiment/registry ops.

Rules:
- Avoid assigning the same worker twice in a row unless necessary.
- Prefer tables unless the user explicitly requests charts/models.
- Stop when the task appears complete and respond with FINISH.
"""

    route_options = ["FINISH"] + subagent_names

    function_def = {
        "name": "route",
        "description": "Select the next worker.",
        "parameters": {
            "title": "route_schema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [{"enum": route_options}],
                }
            },
            "required": ["next"],
        },
    }

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next? Or FINISH? Select one of: {route_options}",
            ),
        ]
    ).partial(
        route_options=str(route_options), subagent_names=", ".join(subagent_names)
    )

    supervisor_chain = (
        prompt
        | llm.bind(functions=[function_def], function_call={"name": "route"})
        | JsonOutputFunctionsParser()
    )

    def _clean_messages(msgs: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
        """
        Strip tool call payloads to avoid OpenAI 'tool_calls' vs 'functions' conflicts.
        Skip tool/function role messages; drop tool_calls field from AI messages.
        """
        cleaned: list[BaseMessage] = []
        for m in msgs or []:
            role = getattr(m, "role", getattr(m, "type", None))
            if role in ("tool", "function"):
                continue
            if hasattr(m, "tool_calls"):
                cleaned.append(AIMessage(content=getattr(m, "content", ""), name=getattr(m, "name", None)))
            else:
                cleaned.append(m)
        return cleaned

    def supervisor_node(state: SupervisorDSState):
        print("---SUPERVISOR---")
        clean_msgs = _clean_messages(state.get("messages", []))
        result = supervisor_chain.invoke({"messages": clean_msgs})
        return {"next": result.get("next")}

    def _merge_messages(state: SupervisorDSState, response: dict):
        msgs = state.get("messages", [])
        new_msgs = response.get("messages") or []
        return {"messages": msgs + new_msgs}

    def node_loader(state: SupervisorDSState):
        print("---DATA LOADER---")
        data_loader_agent.invoke_messages(messages=state.get("messages", []))
        merged = _merge_messages(state, data_loader_agent.response or {})

        loader_artifacts = (data_loader_agent.response or {}).get(
            "data_loader_artifacts"
        )

        # Heuristic: if caller has no data_raw, populate from the first returned artifact
        data_raw = state.get("data_raw")
        if data_raw is None and isinstance(loader_artifacts, dict) and loader_artifacts:
            first_val = next(iter(loader_artifacts.values()))
            # Many tools return {"status","data","error"}
            if isinstance(first_val, dict) and "data" in first_val:
                data_raw = first_val.get("data")
            else:
                data_raw = loader_artifacts

        return {
            **merged,
            "data_raw": data_raw,
            "artifacts": {
                **state.get("artifacts", {}),
                "data_loader": loader_artifacts,
            },
        }

    def node_wrangling(state: SupervisorDSState):
        print("---DATA WRANGLING---")
        data_wrangling_agent.invoke_messages(
            messages=state.get("messages", []),
            data_raw=state.get("data_raw"),
        )
        merged = _merge_messages(state, data_wrangling_agent.response or {})
        return {
            **merged,
            "data_wrangled": (data_wrangling_agent.response or {}).get("data_wrangled"),
            "artifacts": {
                **state.get("artifacts", {}),
                "data_wrangling": (data_wrangling_agent.response or {}).get(
                    "data_wrangled"
                ),
            },
        }

    def node_cleaning(state: SupervisorDSState):
        print("---DATA CLEANING---")
        data_cleaning_agent.invoke_messages(
            messages=state.get("messages", []),
            data_raw=state.get("data_wrangled") or state.get("data_raw"),
        )
        merged = _merge_messages(state, data_cleaning_agent.response or {})
        return {
            **merged,
            "data_cleaned": (data_cleaning_agent.response or {}).get("data_cleaned"),
            "artifacts": {
                **state.get("artifacts", {}),
                "data_cleaning": (data_cleaning_agent.response or {}).get(
                    "data_cleaned"
                ),
            },
        }

    def node_sql(state: SupervisorDSState):
        print("---SQL DATABASE---")
        sql_database_agent.invoke_messages(
            messages=state.get("messages", []),
        )
        merged = _merge_messages(state, sql_database_agent.response or {})
        return {
            **merged,
            "data_sql": (sql_database_agent.response or {}).get("data_sql"),
            "artifacts": {
                **state.get("artifacts", {}),
                "sql": {
                    "sql_query_code": (sql_database_agent.response or {}).get(
                        "sql_query_code"
                    ),
                    "sql_database_function": (sql_database_agent.response or {}).get(
                        "sql_database_function"
                    ),
                    "data_sql": (sql_database_agent.response or {}).get("data_sql"),
                },
            },
        }

    def node_eda(state: SupervisorDSState):
        print("---EDA TOOLS---")
        eda_tools_agent.invoke_messages(
            messages=state.get("messages", []),
            data_raw=state.get("data_cleaned")
            or state.get("data_wrangled")
            or state.get("data_raw"),
        )
        merged = _merge_messages(state, eda_tools_agent.response or {})
        return {
            **merged,
            "eda_artifacts": (eda_tools_agent.response or {}).get("eda_artifacts"),
            "artifacts": {
                **state.get("artifacts", {}),
                "eda": (eda_tools_agent.response or {}).get("eda_artifacts"),
            },
        }

    def node_viz(state: SupervisorDSState):
        print("---DATA VISUALIZATION---")
        data_visualization_agent.invoke_messages(
            messages=state.get("messages", []),
            data_raw=state.get("data_cleaned")
            or state.get("data_wrangled")
            or state.get("data_sql")
            or state.get("data_raw"),
        )
        merged = _merge_messages(state, data_visualization_agent.response or {})
        return {
            **merged,
            "viz_graph": (data_visualization_agent.response or {}).get("plotly_graph"),
            "artifacts": {
                **state.get("artifacts", {}),
                "viz": {
                    "plotly_graph": (data_visualization_agent.response or {}).get(
                        "plotly_graph"
                    ),
                    "data_visualization_function": (
                        data_visualization_agent.response or {}
                    ).get("data_visualization_function"),
                },
            },
        }

    def node_fe(state: SupervisorDSState):
        print("---FEATURE ENGINEERING---")
        feature_engineering_agent.invoke_messages(
            messages=state.get("messages", []),
            data_raw=state.get("data_cleaned")
            or state.get("data_wrangled")
            or state.get("data_sql")
            or state.get("data_raw"),
        )
        merged = _merge_messages(state, feature_engineering_agent.response or {})
        return {
            **merged,
            "feature_data": (feature_engineering_agent.response or {}).get(
                "feature_engineered_data"
            ),
            "artifacts": {
                **state.get("artifacts", {}),
                "feature_engineering": (feature_engineering_agent.response or {}),
            },
        }

    def node_h2o(state: SupervisorDSState):
        print("---H2O ML---")
        h2o_ml_agent.invoke_messages(
            messages=state.get("messages", []),
            data_raw=state.get("feature_data")
            or state.get("data_cleaned")
            or state.get("data_wrangled")
            or state.get("data_raw"),
        )
        merged = _merge_messages(state, h2o_ml_agent.response or {})
        return {
            **merged,
            "model_info": (h2o_ml_agent.response or {}).get("leaderboard"),
            "artifacts": {
                **state.get("artifacts", {}),
                "h2o": (h2o_ml_agent.response or {}),
            },
        }

    def node_mlflow(state: SupervisorDSState):
        print("---MLFLOW TOOLS---")
        mlflow_tools_agent.invoke_messages(
            messages=state.get("messages", []),
        )
        merged = _merge_messages(state, mlflow_tools_agent.response or {})
        return {
            **merged,
            "mlflow_artifacts": (mlflow_tools_agent.response or {}).get(
                "mlflow_artifacts"
            ),
            "artifacts": {
                **state.get("artifacts", {}),
                "mlflow": (mlflow_tools_agent.response or {}).get("mlflow_artifacts"),
            },
        }

    workflow = StateGraph(SupervisorDSState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("Data_Loader_Tools_Agent", node_loader)
    workflow.add_node("Data_Wrangling_Agent", node_wrangling)
    workflow.add_node("Data_Cleaning_Agent", node_cleaning)
    workflow.add_node("EDA_Tools_Agent", node_eda)
    workflow.add_node("Data_Visualization_Agent", node_viz)
    workflow.add_node("SQL_Database_Agent", node_sql)
    workflow.add_node("Feature_Engineering_Agent", node_fe)
    workflow.add_node("H2O_ML_Agent", node_h2o)
    workflow.add_node("MLflow_Tools_Agent", node_mlflow)

    workflow.set_entry_point("supervisor")

    # After any worker, return to supervisor
    for node in subagent_names:
        workflow.add_edge(node, "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state.get("next"),
        {name: name for name in subagent_names} | {"FINISH": END},
    )

    app = workflow.compile(checkpointer=checkpointer, name="supervisor_ds_team")
    return app


class SupervisorDSTeam:
    """
    OO wrapper for the supervisor-led data science team.

    Mirrors the pattern used by other agents: holds a compiled graph,
    exposes message-first helpers, and keeps the latest response.
    """

    def __init__(
        self,
        model: Any,
        data_loader_agent,
        data_wrangling_agent,
        data_cleaning_agent,
        eda_tools_agent,
        data_visualization_agent,
        sql_database_agent,
        feature_engineering_agent,
        h2o_ml_agent,
        mlflow_tools_agent,
        checkpointer: Optional[Checkpointer] = None,
        temperature: float = 0.0,
    ):
        self._params = {
            "model": model,
            "data_loader_agent": data_loader_agent,
            "data_wrangling_agent": data_wrangling_agent,
            "data_cleaning_agent": data_cleaning_agent,
            "eda_tools_agent": eda_tools_agent,
            "data_visualization_agent": data_visualization_agent,
            "sql_database_agent": sql_database_agent,
            "feature_engineering_agent": feature_engineering_agent,
            "h2o_ml_agent": h2o_ml_agent,
            "mlflow_tools_agent": mlflow_tools_agent,
            "checkpointer": checkpointer,
            "temperature": temperature,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response: Optional[dict] = None

    def _make_compiled_graph(self):
        self.response = None
        return make_supervisor_ds_team(
            model=self._params["model"],
            data_loader_agent=self._params["data_loader_agent"],
            data_wrangling_agent=self._params["data_wrangling_agent"],
            data_cleaning_agent=self._params["data_cleaning_agent"],
            eda_tools_agent=self._params["eda_tools_agent"],
            data_visualization_agent=self._params["data_visualization_agent"],
            sql_database_agent=self._params["sql_database_agent"],
            feature_engineering_agent=self._params["feature_engineering_agent"],
            h2o_ml_agent=self._params["h2o_ml_agent"],
            mlflow_tools_agent=self._params["mlflow_tools_agent"],
            checkpointer=self._params["checkpointer"],
            temperature=self._params["temperature"],
        )

    def update_params(self, **kwargs):
        """
        Update parameters (e.g., swap sub-agents or model) and rebuild the graph.
        """
        for k, v in kwargs.items():
            self._params[k] = v
        self._compiled_graph = self._make_compiled_graph()

    def invoke_messages(self, messages: Sequence[BaseMessage], artifacts: Optional[dict] = None, **kwargs):
        """
        Invoke the team with a message list (recommended for supervisor/teams).
        """
        self.response = self._compiled_graph.invoke(
            {"messages": messages, "artifacts": artifacts or {}},
            **kwargs,
        )
        return None

    async def ainvoke_messages(self, messages: Sequence[BaseMessage], artifacts: Optional[dict] = None, **kwargs):
        """
        Async version of invoke_messages.
        """
        self.response = await self._compiled_graph.ainvoke(
            {"messages": messages, "artifacts": artifacts or {}},
            **kwargs,
        )
        return None

    def invoke_agent(self, user_instructions: str, artifacts: Optional[dict] = None, **kwargs):
        """
        Convenience wrapper for a single human prompt.
        """
        msg = HumanMessage(content=user_instructions)
        return self.invoke_messages(messages=[msg], artifacts=artifacts, **kwargs)

    async def ainvoke_agent(self, user_instructions: str, artifacts: Optional[dict] = None, **kwargs):
        msg = HumanMessage(content=user_instructions)
        return await self.ainvoke_messages(messages=[msg], artifacts=artifacts, **kwargs)

    def invoke(self, input: dict, **kwargs):
        """
        Generic invoke passthrough (for backward compatibility).
        """
        self.response = self._compiled_graph.invoke(input, **kwargs)
        return self.response

    async def ainvoke(self, input: dict, **kwargs):
        self.response = await self._compiled_graph.ainvoke(input, **kwargs)
        return self.response

    def get_ai_message(self, markdown: bool = False):
        """
        Return the last assistant/ai message.
        """
        if not self.response or "messages" not in self.response:
            return None
        last_ai = None
        for msg in reversed(self.response.get("messages", [])):
            if isinstance(msg, AIMessage) or getattr(msg, "role", None) in ("assistant", "ai"):
                last_ai = msg
                break
        if last_ai is None:
            return None
        content = getattr(last_ai, "content", "")
        return Markdown(content) if markdown else content

    def get_artifacts(self):
        """
        Return aggregated artifacts dict from the supervisor state.
        """
        if self.response:
            return self.response.get("artifacts")
        return None
