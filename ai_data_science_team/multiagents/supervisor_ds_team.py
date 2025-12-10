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
    last_worker: Optional[str]

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

Critical rule: only route to workers when the user explicitly asks for their capabilities. Do not assume the next steps.

Routing guidance:
- If no data_raw and the user asks to load/import/read a file (e.g., "load data/churn_data.csv"), call Data_Loader_Tools_Agent ONCE, then FINISH unless they explicitly ask for cleaning, EDA, visualization, feature engineering, SQL, or modeling.
- If data_raw exists AND the user explicitly asks to clean, wrangle, impute, or otherwise prepare the data, route to Data_Wrangling_Agent or Data_Cleaning_Agent. Do NOT automatically run them after loading.
- Use SQL_Database_Agent only for SQL/database questions.
- Use EDA_Tools_Agent only for describe/missingness/correlation/Sweetviz when requested.
- Use Data_Visualization_Agent only when the user asks for a plot/chart/graph.
- Use Feature_Engineering_Agent only when asked for feature creation/encoding.
- Use H2O_ML_Agent only when asked to train/evaluate a model/AutoML.
- Use MLflow_Tools_Agent only when asked about experiment tracking or model registry.

Rules:
- Track which worker acted last and do NOT select the same worker twice in a row unless explicitly required.
- Prefer tables unless the user explicitly requests charts/models.
- If the user request appears satisfied, respond with FINISH.
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
            ("system", "Last worker: {last_worker}"),
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

    def _parse_intent(msgs: Sequence[BaseMessage]):
        last_human = ""
        for m in reversed(msgs or []):
            role = getattr(m, "role", getattr(m, "type", None))
            if role in ("human", "user"):
                last_human = (getattr(m, "content", "") or "").lower()
                break
        def has(*words):
            return any(w in last_human for w in words)
        wants_preview = has("head", "first 5", "first five", "preview", "show rows", "top 5", "first five rows", "first 5 rows")
        wants_viz = has("plot", "chart", "visual", "graph")
        wants_sql = has("sql", "query", "database", "table")
        wants_clean = has("clean", "impute", "missing", "null", "na", "outlier")
        wants_wrangling = has("wrangle", "transform", "rename", "format", "standardize")
        wants_eda = has("describe", "eda", "summary", "correlation", "sweetviz", "missingness")
        wants_feature = has("feature", "encode", "one-hot", "feat eng")
        wants_model = has("train", "model", "automl", "classify", "regression", "predict")

        wants_load = has("load", "import", "read csv", "read file", "open file")
        mentions_file = (".csv" in last_human) or (".parquet" in last_human) or (".xlsx" in last_human) or ("file" in last_human)

        wants_more_processing = any(
            [
                wants_preview,
                wants_viz,
                wants_sql,
                wants_clean,
                wants_wrangling,
                wants_eda,
                wants_feature,
                wants_model,
            ]
        )
        load_only = wants_load and mentions_file and not wants_more_processing
        return {
            "preview": wants_preview,
            "viz": wants_viz,
            "sql": wants_sql,
            "clean": wants_clean,
            "wrangle": wants_wrangling,
            "eda": wants_eda,
            "feature": wants_feature,
            "model": wants_model,
            "load_only": load_only,
        }

    def _get_last_human(msgs: Sequence[BaseMessage]) -> str:
        for m in reversed(msgs or []):
            role = getattr(m, "role", getattr(m, "type", None))
            if role in ("human", "user"):
                return getattr(m, "content", "") or ""
        return ""

    def _suggest_next_worker(state: SupervisorDSState, clean_msgs: Sequence[BaseMessage]):
        """
        Lightweight LLM pre-router to nudge the next worker based on last user ask and data readiness.
        """
        last_human = _get_last_human(clean_msgs)
        if not last_human:
            return None
        data_ready = state.get("data_raw") is not None
        last_worker = state.get("last_worker")
        hint_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a routing helper. Suggest the best next worker from {subagent_names} "
                    "given the last user request. If the request appears satisfied, return FINISH. "
                    "Data loaded: {data_ready}. Last worker: {last_worker}.",
                ),
                ("human", "{last_human}"),
            ]
        ).partial(
            subagent_names=str(subagent_names),
            data_ready=str(data_ready),
            last_worker=str(last_worker),
        )
        route_def = function_def
        try:
            chain = (
                hint_prompt
                | llm.bind(functions=[route_def], function_call={"name": "route"})
                | JsonOutputFunctionsParser()
            )
            res = chain.invoke({"last_human": last_human})
            return res.get("next")
        except Exception:
            return None

    def supervisor_node(state: SupervisorDSState):
        print("---SUPERVISOR---")
        clean_msgs = _clean_messages(state.get("messages", []))
        intents = _parse_intent(clean_msgs)
        data_ready = state.get("data_raw") is not None
        last_worker = state.get("last_worker")

        # Simple load-only overrides before any LLM routing
        if intents.get("load_only") and not data_ready:
            print("  simple load-only request -> forcing Data_Loader_Tools_Agent")
            return {"next": "Data_Loader_Tools_Agent", "last_worker": last_worker}
        if intents.get("load_only") and data_ready:
            print("  simple load-only request with data_ready -> FINISH")
            return {"next": "FINISH", "last_worker": last_worker}

        hint_next = _suggest_next_worker(state, clean_msgs)
        result = supervisor_chain.invoke(
            {"messages": clean_msgs, "last_worker": state.get("last_worker")}
        )
        next_worker = result.get("next")
        print(f"  data_ready={data_ready}, last_worker={last_worker}, router_next={next_worker}, hint_next={hint_next}")

        # Prefer hint when available to avoid over-routing (e.g., finish after load-only requests)
        if hint_next and hint_next != next_worker:
            next_worker = hint_next

        # Intent-aware override when data is present
        if data_ready:
            wants_more = any(
                [
                    intents["viz"],
                    intents["eda"],
                    intents["clean"],
                    intents["wrangle"],
                    intents["sql"],
                    intents["feature"],
                    intents["model"],
                ]
            )
            if next_worker == "Data_Loader_Tools_Agent":
                # Avoid bouncing back to loader once data is available
                if intents["viz"]:
                    next_worker = "Data_Visualization_Agent"
                elif intents["eda"]:
                    next_worker = "EDA_Tools_Agent"
                elif intents["clean"] or intents["wrangle"]:
                    next_worker = "Data_Wrangling_Agent"
                elif intents["feature"]:
                    next_worker = "Feature_Engineering_Agent"
                elif intents["model"]:
                    next_worker = "H2O_ML_Agent"
                elif not wants_more and intents["preview"]:
                    next_worker = "FINISH"
                elif not wants_more:
                    next_worker = "FINISH"
                else:
                    next_worker = "Data_Wrangling_Agent"
            else:
                # If only a preview was requested and loader already ran, finish.
                if (
                    not wants_more
                    and intents["preview"]
                    and last_worker == "Data_Loader_Tools_Agent"
                ):
                    next_worker = "FINISH"

        return {"next": next_worker, "last_worker": next_worker}

    def _trim_messages(msgs: Sequence[BaseMessage], max_messages: int = 20, max_chars: int = 2000):
        trimmed = []
        for m in msgs[-max_messages:]:
            content = getattr(m, "content", "")
            if isinstance(content, str) and len(content) > max_chars:
                content = content[: max_chars] + "\n...[truncated]..."
                if isinstance(m, AIMessage):
                    m = AIMessage(content=content, name=getattr(m, "name", None))
                elif isinstance(m, HumanMessage):
                    m = HumanMessage(content=content)
                else:
                    # fallback to same type without tool_calls
                    m = BaseMessage(type=getattr(m, "type", "ai"), content=content)
            trimmed.append(m)
        return trimmed

    def _merge_messages(state: SupervisorDSState, response: dict):
        msgs = state.get("messages", [])
        new_msgs = response.get("messages") or []
        merged = msgs + new_msgs
        merged = _trim_messages(merged)
        return {"messages": merged}

    def _tag_messages(msgs: Sequence[BaseMessage], default_name: str):
        tagged = []
        for m in msgs or []:
            if isinstance(m, AIMessage) and not getattr(m, "name", None):
                tagged.append(
                    AIMessage(content=getattr(m, "content", ""), name=default_name)
                )
            else:
                tagged.append(m)
        return tagged

    def _ensure_df(data):
        try:
            import pandas as pd
            if data is None:
                return None
            if isinstance(data, dict):
                return pd.DataFrame(data)
            if isinstance(data, list):
                return pd.DataFrame(data)
            return data
        except Exception:
            return data

    def _shape(obj):
        try:
            import pandas as pd
            if isinstance(obj, pd.DataFrame):
                return obj.shape
            if isinstance(obj, dict):
                return (len(obj), len(next(iter(obj.values()))) if obj else 0)
            if isinstance(obj, list):
                return (len(obj),)
        except Exception:
            return None
        return None

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
        print(f"  loader data_raw shape={_shape(data_raw)}")

        # Add a lightweight AI summary message so supervisor can progress
        summary_msg = None
        if isinstance(data_raw, dict):
            try:
                import pandas as pd

                df = pd.DataFrame(data_raw)
                summary_msg = AIMessage(
                    content=f"Loaded dataset with shape {df.shape}. What would you like to do next? I can clean, explore, visualize, engineer features, run SQL, or train a model.",
                    name="data_loader_agent",
                )
            except Exception:
                summary_msg = AIMessage(
                    content="Loaded dataset successfully. What would you like to do next?",
                    name="data_loader_agent",
                )
        elif data_raw is not None:
            summary_msg = AIMessage(
                content="Loaded dataset successfully. What would you like to do next?",
                name="data_loader_agent",
            )

        if summary_msg:
            merged["messages"] = merged.get("messages", []) + [summary_msg]

        merged["messages"] = _tag_messages(merged.get("messages"), "data_loader_agent")
        return {
            **merged,
            "data_raw": data_raw,
            "artifacts": {
                **state.get("artifacts", {}),
                "data_loader": loader_artifacts,
            },
            "last_worker": "Data_Loader_Tools_Agent",
        }

    def node_wrangling(state: SupervisorDSState):
        print("---DATA WRANGLING---")
        data_wrangling_agent.invoke_messages(
            messages=state.get("messages", []),
            data_raw=_ensure_df(state.get("data_raw")),
        )
        merged = _merge_messages(state, data_wrangling_agent.response or {})
        merged["messages"] = _tag_messages(
            merged.get("messages"), "data_wrangling_agent"
        )
        return {
            **merged,
            "data_wrangled": (data_wrangling_agent.response or {}).get("data_wrangled"),
            "artifacts": {
                **state.get("artifacts", {}),
                "data_wrangling": (data_wrangling_agent.response or {}).get(
                    "data_wrangled"
                ),
            },
            "last_worker": "Data_Wrangling_Agent",
        }

    def node_cleaning(state: SupervisorDSState):
        print("---DATA CLEANING---")
        data_cleaning_agent.invoke_messages(
            messages=state.get("messages", []),
            data_raw=_ensure_df(state.get("data_wrangled") or state.get("data_raw")),
        )
        merged = _merge_messages(state, data_cleaning_agent.response or {})
        merged["messages"] = _tag_messages(
            merged.get("messages"), "data_cleaning_agent"
        )
        return {
            **merged,
            "data_cleaned": (data_cleaning_agent.response or {}).get("data_cleaned"),
            "artifacts": {
                **state.get("artifacts", {}),
                "data_cleaning": (data_cleaning_agent.response or {}).get(
                    "data_cleaned"
                ),
            },
            "last_worker": "Data_Cleaning_Agent",
        }

    def node_sql(state: SupervisorDSState):
        print("---SQL DATABASE---")
        sql_database_agent.invoke_messages(
            messages=state.get("messages", []),
        )
        merged = _merge_messages(state, sql_database_agent.response or {})
        merged["messages"] = _tag_messages(
            merged.get("messages"), "sql_database_agent"
        )
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
            "last_worker": "SQL_Database_Agent",
        }

    def node_eda(state: SupervisorDSState):
        print("---EDA TOOLS---")
        eda_tools_agent.invoke_messages(
            messages=state.get("messages", []),
            data_raw=_ensure_df(
                state.get("data_cleaned")
                or state.get("data_wrangled")
                or state.get("data_raw")
            ),
        )
        merged = _merge_messages(state, eda_tools_agent.response or {})
        merged["messages"] = _tag_messages(merged.get("messages"), "eda_tools_agent")
        print(f"  eda artifacts keys={(eda_tools_agent.response or {}).get('eda_artifacts') and list((eda_tools_agent.response or {}).get('eda_artifacts').keys())}")
        return {
            **merged,
            "eda_artifacts": (eda_tools_agent.response or {}).get("eda_artifacts"),
            "artifacts": {
                **state.get("artifacts", {}),
                "eda": (eda_tools_agent.response or {}).get("eda_artifacts"),
            },
            "last_worker": "EDA_Tools_Agent",
        }

    def node_viz(state: SupervisorDSState):
        print("---DATA VISUALIZATION---")
        data_visualization_agent.invoke_messages(
            messages=state.get("messages", []),
            data_raw=_ensure_df(
                state.get("data_cleaned")
                or state.get("data_wrangled")
                or state.get("data_sql")
                or state.get("data_raw")
            ),
        )
        merged = _merge_messages(state, data_visualization_agent.response or {})
        merged["messages"] = _tag_messages(
            merged.get("messages"), "data_visualization_agent"
        )
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
            "last_worker": "Data_Visualization_Agent",
        }

    def node_fe(state: SupervisorDSState):
        print("---FEATURE ENGINEERING---")
        feature_engineering_agent.invoke_messages(
            messages=state.get("messages", []),
            data_raw=_ensure_df(
                state.get("data_cleaned")
                or state.get("data_wrangled")
                or state.get("data_sql")
                or state.get("data_raw")
            ),
        )
        merged = _merge_messages(state, feature_engineering_agent.response or {})
        merged["messages"] = _tag_messages(
            merged.get("messages"), "feature_engineering_agent"
        )
        return {
            **merged,
            "feature_data": (feature_engineering_agent.response or {}).get(
                "feature_engineered_data"
            ),
            "artifacts": {
                **state.get("artifacts", {}),
                "feature_engineering": (feature_engineering_agent.response or {}),
            },
            "last_worker": "Feature_Engineering_Agent",
        }

    def node_h2o(state: SupervisorDSState):
        print("---H2O ML---")
        h2o_ml_agent.invoke_messages(
            messages=state.get("messages", []),
            data_raw=_ensure_df(
                state.get("feature_data")
                or state.get("data_cleaned")
                or state.get("data_wrangled")
                or state.get("data_raw")
            ),
        )
        merged = _merge_messages(state, h2o_ml_agent.response or {})
        merged["messages"] = _tag_messages(merged.get("messages"), "h2o_ml_agent")
        return {
            **merged,
            "model_info": (h2o_ml_agent.response or {}).get("leaderboard"),
            "artifacts": {
                **state.get("artifacts", {}),
                "h2o": (h2o_ml_agent.response or {}),
            },
            "last_worker": "H2O_ML_Agent",
        }

    def node_mlflow(state: SupervisorDSState):
        print("---MLFLOW TOOLS---")
        mlflow_tools_agent.invoke_messages(
            messages=state.get("messages", []),
        )
        merged = _merge_messages(state, mlflow_tools_agent.response or {})
        merged["messages"] = _tag_messages(
            merged.get("messages"), "mlflow_tools_agent"
        )
        return {
            **merged,
            "mlflow_artifacts": (mlflow_tools_agent.response or {}).get(
                "mlflow_artifacts"
            ),
            "artifacts": {
                **state.get("artifacts", {}),
                "mlflow": (mlflow_tools_agent.response or {}).get("mlflow_artifacts"),
            },
            "last_worker": "MLflow_Tools_Agent",
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
