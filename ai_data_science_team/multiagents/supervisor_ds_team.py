from __future__ import annotations

from typing import Sequence, TypedDict, Annotated, Optional, Dict, Any, List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from IPython.display import Markdown
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.types import Checkpointer
from langgraph.graph.message import add_messages


TEAM_MAX_MESSAGES = 30
TEAM_MAX_MESSAGE_CHARS = 4000


def _supervisor_merge_messages(
    left: Sequence[BaseMessage] | None,
    right: Sequence[BaseMessage] | None,
) -> List[BaseMessage]:
    """
    Merge conversation messages safely:
    - Use LangGraph's ID-aware add_messages reducer (prevents duplicates)
    - Drop tool/function role messages (tool outputs can confuse router models)
    - Strip tool_calls payloads from AI messages (avoids tool_calls vs functions conflicts)
    - Truncate very long message bodies
    - Keep only the last N messages
    """
    merged = add_messages(left or [], right or [])

    cleaned: list[BaseMessage] = []
    for m in merged:
        role = getattr(m, "type", None) or getattr(m, "role", None)
        if role in ("tool", "function"):
            continue

        content = getattr(m, "content", "")
        message_id = getattr(m, "id", None)

        if isinstance(content, str) and len(content) > TEAM_MAX_MESSAGE_CHARS:
            content = content[:TEAM_MAX_MESSAGE_CHARS] + "\n...[truncated]..."

        # Remove tool call payloads to keep downstream OpenAI function-calling stable
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            cleaned.append(
                AIMessage(
                    content=content or "",
                    name=getattr(m, "name", None),
                    id=message_id,
                )
            )
            continue

        # Rebuild truncated variants for common message types to avoid mutating originals
        if isinstance(m, AIMessage):
            cleaned.append(
                AIMessage(
                    content=content or "",
                    name=getattr(m, "name", None),
                    id=message_id,
                )
            )
        elif isinstance(m, HumanMessage):
            cleaned.append(HumanMessage(content=content or "", id=message_id))
        elif isinstance(m, SystemMessage):
            cleaned.append(SystemMessage(content=content or "", id=message_id))
        else:
            cleaned.append(m)

    return cleaned[-TEAM_MAX_MESSAGES:]


class SupervisorDSState(TypedDict):
    """
    Shared state for the supervisor-led data science team.
    """

    # Team conversation
    messages: Annotated[Sequence[BaseMessage], _supervisor_merge_messages]
    next: str
    last_worker: Optional[str]
    active_data_key: Optional[str]

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

    if isinstance(model, str):
        llm = ChatOpenAI(model=model, temperature=temperature)
    else:
        llm = model
        # Best-effort: allow callers to pass an already-configured LLM
        try:
            llm.temperature = temperature
        except Exception:
            pass

    system_prompt = """
You are a supervisor managing a data science team with these workers: {subagent_names}.

Each worker has specific tools/capabilities (names are a hint for routing):
- Data_Loader_Tools_Agent: Good for inspecting file folder system, finding files, searching and loading data. Has the following tools: load_file, load_directory, search_files_by_pattern, list_directory_contents/recursive.
- Data_Wrangling_Agent: Can work with one or more datasets, performing operations such as joining/merging multiple datasets, reshaping, aggregating, encoding, creating computed features, and ensuring consistent data types. Capabilities: recommend_wrangling_steps, create_data_wrangling_code, execute_data_wrangling_code (transform/rename/format). Must have data loaded/ready.
- Data_Cleaning_Agent: Strong in cleaning data, removing anomalies, and fixing data issues. Capabilities: recommend_cleaning_steps, create_data_cleaner_code, execute_data_cleaner_code (impute/clean). Must have data loaded/ready.
- EDA_Tools_Agent: Strong in exploring data, analysing data, and providing information about the data. Has several powerful tools: describe_dataset, explain_data, visualize_missing, correlation_funnel, sweetviz (use for previews/head/describe). Must have data loaded/ready.
- Data_Visualization_Agent: Can generate Plotly charts based on user-defined instructions or default visualization steps. Must have data loaded/ready.  
- SQL_Database_Agent: Generate a SQL query based on the recommended steps and user instructions. Executes that SQL query against the provided database connection, returning the data results.
- Feature_Engineering_Agent: The agent applies various feature engineering techniques, such as encoding categorical variables, scaling numeric variables, creating interaction terms,and generating polynomial features. Must have data loaded/ready.
- H2O_ML_Agent: A Machine Learning agent that uses H2O's AutoML for training create_h2o_automl_code, execute_h2o_code (AutoML training/eval).
- MLflow_Tools_Agent: Can interact and run various tools related to accessing, interacting with, and retrieving information from MLflow. Has tools including: mlflow_search_experiments, mlflow_search_runs, mlflow_create_experiment, mlflow_predict_from_run_id, mlflow_launch_ui, mlflow_stop_ui, mlflow_list_artifacts, mlflow_download_artifacts, mlflow_list_registered_models, mlflow_search_registered_models, mlflow_get_model_version_details, mlflow_get_run_details, mlflow_transition_model_version_stage, mlflow_tracking_info, mlflow_ui_status,

Critical rule: only route to workers when the user explicitly asks for their capabilities. Do not assume next steps.

Routing guidance (explicit intent -> worker):
- Load/import/read file (e.g., "load data/churn_data.csv"): Data_Loader_Tools_Agent ONCE, then FINISH unless more is requested.
- Show first N rows / preview / head / describe: EDA_Tools_Agent then FINISH.
- Plot/chart/visual/graph: Data_Visualization_Agent.
- Clean/impute/wrangle/standardize: Data_Wrangling_Agent or Data_Cleaning_Agent.
- SQL/database/query/tables: SQL_Database_Agent.
- Feature creation/encoding: Feature_Engineering_Agent.
- Train/evaluate model/AutoML: H2O_ML_Agent.
- MLflow tracking/registry: MLflow_Tools_Agent.

Rules:
- Track which worker acted last and do NOT select the same worker twice in a row unless explicitly required.
- Prefer tables unless the user explicitly requests charts/models.
- If the user request appears satisfied, respond with FINISH.

Examples:
- "load data/churn_data.csv" -> Data_Loader_Tools_Agent, then FINISH.
- "show the first 5 rows" (data already loaded) -> EDA_Tools_Agent, then FINISH.
- "describe the dataset" -> EDA_Tools_Agent.
- "plot churn by tenure" -> Data_Visualization_Agent.
- "clean missing values" -> Data_Cleaning_Agent.
- "what tables are in the DB?" -> SQL_Database_Agent.
- "engineer one-hot features for churn" -> Feature_Engineering_Agent.
- "train a model for Churn" -> H2O_ML_Agent.
- "list mlflow experiments" -> MLflow_Tools_Agent.
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
            role = getattr(m, "type", None) or getattr(m, "role", None)
            if role in ("tool", "function"):
                continue
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                cleaned.append(
                    AIMessage(
                        content=getattr(m, "content", "") or "",
                        name=getattr(m, "name", None),
                        id=getattr(m, "id", None),
                    )
                )
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

        wants_preview = has(
            "head",
            "first 5",
            "first five",
            "preview",
            "show rows",
            "top 5",
            "first five rows",
            "first 5 rows",
        )
        wants_viz = has("plot", "chart", "visual", "graph")
        wants_sql = has("sql", "query", "database", "table")
        wants_clean = has("clean", "impute", "missing", "null", "na", "outlier")
        wants_wrangling = has("wrangle", "transform", "rename", "format", "standardize")
        wants_eda = has(
            "describe", "eda", "summary", "correlation", "sweetviz", "missingness"
        )
        wants_feature = has("feature", "encode", "one-hot", "feat eng")
        wants_model = has(
            "train", "model", "automl", "classify", "regression", "predict"
        )

        wants_load = has("load", "import", "read csv", "read file", "open file")
        mentions_file = (
            (".csv" in last_human)
            or (".parquet" in last_human)
            or (".xlsx" in last_human)
            or ("file" in last_human)
        )

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

    def _suggest_next_worker(
        state: SupervisorDSState, clean_msgs: Sequence[BaseMessage]
    ):
        """
        Disabled LLM hinting to keep routing deterministic.
        """
        return None

    def supervisor_node(state: SupervisorDSState):
        print("---SUPERVISOR---")
        clean_msgs = _clean_messages(state.get("messages", []))
        intents = _parse_intent(clean_msgs)
        # Infer active dataset if not explicitly tracked yet
        active_data_key = state.get("active_data_key")
        if active_data_key is None:
            if state.get("data_cleaned") is not None:
                active_data_key = "data_cleaned"
            elif state.get("data_wrangled") is not None:
                active_data_key = "data_wrangled"
            elif state.get("data_sql") is not None:
                active_data_key = "data_sql"
            elif state.get("feature_data") is not None:
                active_data_key = "feature_data"
            elif state.get("data_raw") is not None:
                active_data_key = "data_raw"

        data_ready = (
            active_data_key is not None and state.get(active_data_key) is not None
        )
        last_worker = state.get("last_worker")
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

        # Simple load-only overrides before any LLM routing.
        # If the user asks to load a (possibly different) file, always route to the loader.
        # Then, once the loader has responded (same invocation), FINISH.
        if intents.get("load_only"):
            last_role = (
                (getattr(clean_msgs[-1], "type", None) or getattr(clean_msgs[-1], "role", None))
                if clean_msgs
                else None
            )
            if last_worker == "Data_Loader_Tools_Agent" and last_role not in ("human", "user"):
                print("  load-only request already handled by loader -> FINISH")
                return {"next": "FINISH", "active_data_key": active_data_key}
            print("  load-only request -> forcing Data_Loader_Tools_Agent")
            return {"next": "Data_Loader_Tools_Agent", "active_data_key": active_data_key}
        # If we just ran EDA for a preview-only ask, finish instead of looping
        if (
            data_ready
            and intents["preview"]
            and not wants_more
            and last_worker == "EDA_Tools_Agent"
        ):
            print("  preview-only request already handled by EDA -> FINISH")
            return {"next": "FINISH", "active_data_key": active_data_key}
        if data_ready and intents["preview"] and not wants_more:
            # For preview requests, route to EDA to produce a head/summary instead of finishing
            print("  preview-only request with data_ready -> route to EDA_Tools_Agent")
            return {"next": "EDA_Tools_Agent", "active_data_key": active_data_key}

        result = supervisor_chain.invoke(
            {"messages": clean_msgs, "last_worker": state.get("last_worker")}
        )
        next_worker = result.get("next")
        print(f"  data_ready={data_ready}, last_worker={last_worker}, router_next={next_worker}")

        # Intent-aware override when data is present
        if data_ready:
            if next_worker == "Data_Loader_Tools_Agent":
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
                elif not wants_more:
                    next_worker = "FINISH"
                else:
                    next_worker = "Data_Wrangling_Agent"
            else:
                if not wants_more and intents["preview"]:
                    next_worker = "FINISH"

        # Keep active_data_key stable unless a worker changes it.
        return {"next": next_worker, "active_data_key": active_data_key}

    def _trim_messages(
        msgs: Sequence[BaseMessage], max_messages: int = TEAM_MAX_MESSAGES, max_chars: int = TEAM_MAX_MESSAGE_CHARS
    ) -> list[BaseMessage]:
        trimmed: list[BaseMessage] = []
        for m in list(msgs or [])[-max_messages:]:
            content = getattr(m, "content", "")
            if isinstance(content, str) and len(content) > max_chars:
                content = content[:max_chars] + "\n...[truncated]..."
                if isinstance(m, AIMessage):
                    m = AIMessage(
                        content=content,
                        name=getattr(m, "name", None),
                        id=getattr(m, "id", None),
                    )
                elif isinstance(m, HumanMessage):
                    m = HumanMessage(content=content, id=getattr(m, "id", None))
                elif isinstance(m, SystemMessage):
                    m = SystemMessage(content=content, id=getattr(m, "id", None))
            trimmed.append(m)
        return trimmed

    def _extract_new_messages(
        before: Sequence[BaseMessage], after: Sequence[BaseMessage]
    ) -> list[BaseMessage]:
        before_list = list(before or [])
        after_list = list(after or [])
        if not after_list:
            return []
        if not before_list:
            return after_list

        if len(after_list) >= len(before_list):
            # Prefer ID-based matching (LangGraph assigns ids when missing)
            if all(
                getattr(after_list[i], "id", None) == getattr(before_list[i], "id", None)
                for i in range(len(before_list))
            ):
                return after_list[len(before_list) :]

            # Fallback to a content/type prefix match
            if all(
                getattr(after_list[i], "type", None) == getattr(before_list[i], "type", None)
                and getattr(after_list[i], "content", None)
                == getattr(before_list[i], "content", None)
                for i in range(len(before_list))
            ):
                return after_list[len(before_list) :]

        # If the agent returned only new messages (delta), just accept them.
        return after_list

    def _merge_messages(
        before_messages: Sequence[BaseMessage], response: dict
    ) -> dict:
        response_msgs = response.get("messages") or []
        new_msgs = _extract_new_messages(before_messages, response_msgs)
        new_msgs = _clean_messages(new_msgs)
        new_msgs = _trim_messages(new_msgs)
        return {"messages": new_msgs}

    def _tag_messages(msgs: Sequence[BaseMessage], default_name: str):
        tagged: list[BaseMessage] = []
        for m in msgs or []:
            if isinstance(m, AIMessage) and not getattr(m, "name", None):
                tagged.append(
                    AIMessage(
                        content=getattr(m, "content", "") or "",
                        name=default_name,
                        id=getattr(m, "id", None),
                    )
                )
            else:
                tagged.append(m)
        return tagged

    def _format_listing_with_llm(rows: list, last_human: str):
        """
        Ask the supervisor llm to format a short summary + markdown table
        for a directory listing. Falls back to None on error.
        """
        if not rows:
            return None
        limited = rows[:30]  # safety cap
        try:
            # Build a minimal prompt to keep tokens small
            system_tmpl = (
                "You are formatting a directory listing for the user. "
                "Return a concise markdown summary and a markdown table with columns "
                "`filename`, `type`, and `path` (omit missing columns). "
                "Do not add extra narration beyond the summary."
            )
            human_tmpl = (
                "User request: {last_human}\n\n"
                "Rows (JSON list): {rows_json}\n\n"
                "Return:\n"
                "1) One-sentence summary.\n"
                "2) A markdown table."
            )
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_tmpl),
                    ("human", human_tmpl),
                ]
            )
            import json

            rows_json = json.dumps(limited)
            chain = prompt | llm
            resp = chain.invoke({"last_human": last_human, "rows_json": rows_json})
            return getattr(resp, "content", None) or str(resp)
        except Exception:
            return None

    def _format_dataset_with_llm(df_dict: dict, last_human: str, max_rows: int = 10, max_cols: int = 6):
        """
        Ask the supervisor llm to summarize a dataset and include a small markdown table preview.
        df_dict is expected to be a column-oriented dict (like DataFrame.to_dict()).
        """
        if not df_dict:
            return None
        try:
            import pandas as pd
            import json

            df = pd.DataFrame(df_dict)
            df_preview = df.iloc[:max_rows, :max_cols]
            table_md = df_preview.to_markdown(index=False)
            system_tmpl = (
                "You are summarizing a dataset for the user. "
                "Return a concise summary and a small markdown table preview already provided. "
                "Do not add extra narration beyond the summary and the table."
            )
            human_tmpl = (
                "User request: {last_human}\n\n"
                "Preview table (markdown):\n{table_md}\n\n"
                "Dataset shape: {shape}"
            )
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_tmpl),
                    ("human", human_tmpl),
                ]
            )
            chain = prompt | llm
            resp = chain.invoke(
                {
                    "last_human": last_human,
                    "table_md": table_md,
                    "shape": str(df.shape),
                }
            )
            return getattr(resp, "content", None) or str(resp)
        except Exception:
            return None

    def _format_result_with_llm(
        agent_name: str,
        df_dict: Optional[dict],
        last_human: str,
        extra_text: str = "",
        max_rows: int = 6,
        max_cols: int = 6,
    ):
        """
        General formatter: produce a concise summary + markdown table preview via LLM.
        Returns a string or None.
        """
        try:
            preview_md = ""
            import pandas as pd
            import json

            if df_dict:
                df = pd.DataFrame(df_dict)
                df_preview = df.iloc[:max_rows, :max_cols]
                preview_md = df_preview.to_markdown(index=False)
                shape = str(df.shape)
            else:
                shape = "unknown"

            system_tmpl = (
                f"You are summarizing the output of the {agent_name}. "
                "Return a concise summary and, if provided, include the markdown table preview as-is. "
                "Do not add extra narration beyond the summary and table."
            )
            human_tmpl = (
                "User request: {last_human}\n\n"
                "Extra context: {extra_text}\n\n"
                "Preview table (markdown):\n{preview_md}\n\n"
                "Data shape: {shape}"
            )
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_tmpl),
                    ("human", human_tmpl),
                ]
            )
            chain = prompt | llm
            resp = chain.invoke(
                {
                    "last_human": last_human,
                    "extra_text": extra_text or "None",
                    "preview_md": preview_md or "None",
                    "shape": shape,
                }
            )
            return getattr(resp, "content", None) or str(resp)
        except Exception:
            return None

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

    def _get_active_data(state: SupervisorDSState, fallback_keys: Sequence[str]):
        active_key = state.get("active_data_key")
        if active_key and state.get(active_key) is not None:
            return state.get(active_key)
        for key in fallback_keys:
            if state.get(key) is not None:
                return state.get(key)
        return None

    def _is_empty_df(df) -> bool:
        try:
            return df is None or bool(getattr(df, "empty", False))
        except Exception:
            return df is None

    def node_loader(state: SupervisorDSState):
        print("---DATA LOADER---")
        before_msgs = list(state.get("messages", []) or [])
        last_human = _get_last_human(before_msgs)

        # DataLoaderToolsAgent is tool-driven; the latest user request is already in messages.
        data_loader_agent.invoke_messages(messages=before_msgs)
        response = data_loader_agent.response or {}
        merged = _merge_messages(before_msgs, response)

        loader_artifacts = response.get("data_loader_artifacts")

        previous_data_raw = state.get("data_raw")
        data_raw = previous_data_raw
        active_data_key = state.get("active_data_key")

        dir_listing = None
        loaded_dataset = None
        loaded_dataset_label = None
        multiple_loaded_files = None

        # Normalize artifacts into a dict so we can inspect tool intent
        artifacts_map: dict = {}
        if loader_artifacts is None:
            artifacts_map = {}
        elif isinstance(loader_artifacts, dict):
            # Could be {tool_name: artifact} OR a single load_file artifact {status,data,error}
            if {"status", "data"}.issubset(set(loader_artifacts.keys())):
                artifacts_map = {"load_file": loader_artifacts}
            else:
                artifacts_map = loader_artifacts
        else:
            artifacts_map = {"artifact": loader_artifacts}

        # Detect directory listings (do NOT overwrite data_raw)
        for key, val in artifacts_map.items():
            if str(key).startswith("list_directory") or str(key).startswith(
                "search_files_by_pattern"
            ):
                dir_listing = val
                break

        # Detect dataset loads
        for key, val in artifacts_map.items():
            tool_name = str(key)
            # load_file artifact: {"status":"ok","data":{...},"error":None}
            if tool_name.startswith("load_file") and isinstance(val, dict):
                if val.get("status") == "ok" and val.get("data") is not None:
                    loaded_dataset = val.get("data")
                    loaded_dataset_label = tool_name
                    break

            # load_directory artifact: {"file.csv": {"status","data","error"}, ...}
            if tool_name.startswith("load_directory") and isinstance(val, dict):
                ok_items = []
                for fname, info in val.items():
                    if (
                        isinstance(info, dict)
                        and info.get("status") == "ok"
                        and info.get("data") is not None
                    ):
                        ok_items.append((fname, info.get("data")))
                if len(ok_items) == 1:
                    loaded_dataset_label, loaded_dataset = ok_items[0]
                    break
                if len(ok_items) > 1:
                    # Multiple datasets loaded; don't guess which one becomes active.
                    multiple_loaded_files = [fname for fname, _ in ok_items]
                    loaded_dataset = None
                    loaded_dataset_label = None
                    break

        if loaded_dataset is not None:
            data_raw = loaded_dataset
            active_data_key = "data_raw"
            # Prefer dataset summary over any incidental listings
            dir_listing = None

        print(f"  loader data_raw shape={_shape(data_raw)} active_data_key={active_data_key}")

        # Add a lightweight AI summary message so supervisor can progress
        summary_msg = None
        if multiple_loaded_files:
            joined = ", ".join(multiple_loaded_files[:20])
            more = (
                f" (+{len(multiple_loaded_files) - 20} more)"
                if len(multiple_loaded_files) > 20
                else ""
            )
            summary_msg = AIMessage(
                content=(
                    "Loaded multiple datasets from the directory:\n\n"
                    f"{joined}{more}\n\n"
                    "Tell me which file you want to load (e.g., `load <filename>`)."
                ),
                name="data_loader_agent",
            )
        elif dir_listing is not None:
            try:
                # dir_listing could be list/dict; extract filenames
                names = []
                rows = []
                if isinstance(dir_listing, list):
                    for item in dir_listing:
                        if isinstance(item, dict) and "filename" in item:
                            names.append(item["filename"])
                            rows.append(
                                {
                                    "filename": item.get("filename"),
                                    "type": item.get("type"),
                                    "path": item.get("path") or item.get("filepath"),
                                }
                            )
                        else:
                            names.append(str(item))
                            rows.append({"filename": str(item)})
                elif isinstance(dir_listing, dict):
                    # maybe mapping index->filename
                    for v in dir_listing.values():
                        if isinstance(v, dict):
                            names.append(str(v.get("filename", v)))
                            rows.append(
                                {
                                    "filename": v.get("filename"),
                                    "type": v.get("type"),
                                    "path": v.get("path") or v.get("filepath"),
                                }
                            )
                        else:
                            names.append(str(v))
                            rows.append({"filename": str(v)})
                msg_text = "Found files: " + ", ".join(names) if names else "Found directory contents."
                table_text = ""
                if rows:
                    import pandas as pd

                    df_listing = pd.DataFrame(rows)
                    table_cols = [c for c in ["filename", "type", "path"] if c in df_listing.columns]
                    table_text = df_listing[table_cols].to_markdown(index=False)
                # If the user asked for a table or better formatting, try a tiny LLM summary
                last_human = _get_last_human(before_msgs)
                llm_text = _format_listing_with_llm(rows, last_human) if rows else None
                if llm_text:
                    summary_msg = AIMessage(content=llm_text, name="data_loader_agent")
                elif table_text:
                    summary_msg = AIMessage(
                        content=f"{msg_text}\n\n{table_text}",
                        name="data_loader_agent",
                    )
                else:
                    summary_msg = AIMessage(content=msg_text, name="data_loader_agent")
            except Exception:
                summary_msg = AIMessage(content="Listed directory contents.", name="data_loader_agent")
        elif loaded_dataset is not None and isinstance(data_raw, dict):
            try:
                import pandas as pd

                df = pd.DataFrame(data_raw)
                table_md = df.iloc[:5, :6].to_markdown(index=False)
                llm_text = _format_result_with_llm(
                    "data_loader_agent",
                    data_raw,
                    _get_last_human(before_msgs),
                )
                if llm_text:
                    summary_msg = AIMessage(content=llm_text, name="data_loader_agent")
                else:
                    summary_msg = AIMessage(
                        content=f"Loaded dataset with shape {df.shape}.\n\n{table_md}",
                        name="data_loader_agent",
                    )
            except Exception:
                summary_msg = AIMessage(
                    content="Loaded dataset successfully. What would you like to do next?",
                    name="data_loader_agent",
                )
        elif loader_artifacts is not None:
            summary_msg = AIMessage(
                content=(
                    "I couldn't load a tabular dataset from that request. "
                    "Try specifying a concrete file path (e.g., `data/churn_data.csv`) "
                    "or ask me to list files in a directory first."
                ),
                name="data_loader_agent",
            )

        if summary_msg:
            merged["messages"] = merged.get("messages", []) + [summary_msg]

        merged["messages"] = _tag_messages(merged.get("messages"), "data_loader_agent")

        # If the dataset changed, clear downstream artifacts to avoid stale plots/models.
        downstream_resets = {}
        if loaded_dataset is not None:
            downstream_resets = {
                "data_wrangled": None,
                "data_cleaned": None,
                "eda_artifacts": None,
                "viz_graph": None,
                "feature_data": None,
                "model_info": None,
                "mlflow_artifacts": None,
            }

        return {
            **merged,
            "data_raw": data_raw,
            "active_data_key": active_data_key,
            "artifacts": {
                **state.get("artifacts", {}),
                "data_loader": loader_artifacts,
            },
            "last_worker": "Data_Loader_Tools_Agent",
            **downstream_resets,
        }

    def node_wrangling(state: SupervisorDSState):
        print("---DATA WRANGLING---")
        before_msgs = list(state.get("messages", []) or [])
        last_human = _get_last_human(before_msgs)
        active_df = _ensure_df(_get_active_data(state, ["data_raw", "data_sql", "data_wrangled", "data_cleaned", "feature_data"]))
        if _is_empty_df(active_df):
            return {
                "messages": [
                    AIMessage(
                        content="No dataset is available to wrangle. Load a file (or run a SQL query) first.",
                        name="data_wrangling_agent",
                    )
                ],
                "last_worker": "Data_Wrangling_Agent",
            }
        data_wrangling_agent.invoke_messages(
            messages=before_msgs,
            user_instructions=last_human,
            data_raw=active_df,
        )
        response = data_wrangling_agent.response or {}
        merged = _merge_messages(before_msgs, response)
        merged["messages"] = _tag_messages(
            merged.get("messages"), "data_wrangling_agent"
        )
        summary_text = _format_result_with_llm(
            "data_wrangling_agent",
            response.get("data_wrangled"),
            _get_last_human(before_msgs),
            extra_text="Wrangling steps completed.",
        )
        if summary_text:
            merged["messages"].append(
                AIMessage(content=summary_text, name="data_wrangling_agent")
            )
        data_wrangled = response.get("data_wrangled")
        downstream_resets = (
            {
                "data_cleaned": None,
                "eda_artifacts": None,
                "viz_graph": None,
                "feature_data": None,
                "model_info": None,
                "mlflow_artifacts": None,
            }
            if data_wrangled is not None
            else {}
        )
        return {
            **merged,
            "data_wrangled": data_wrangled,
            "active_data_key": "data_wrangled" if data_wrangled is not None else state.get("active_data_key"),
            "artifacts": {
                **state.get("artifacts", {}),
                "data_wrangling": data_wrangled,
            },
            "last_worker": "Data_Wrangling_Agent",
            **downstream_resets,
        }

    def node_cleaning(state: SupervisorDSState):
        print("---DATA CLEANING---")
        before_msgs = list(state.get("messages", []) or [])
        last_human = _get_last_human(before_msgs)
        active_df = _ensure_df(_get_active_data(state, ["data_wrangled", "data_raw", "data_sql", "data_cleaned", "feature_data"]))
        if _is_empty_df(active_df):
            return {
                "messages": [
                    AIMessage(
                        content="No dataset is available to clean. Load a file (or run a SQL query) first.",
                        name="data_cleaning_agent",
                    )
                ],
                "last_worker": "Data_Cleaning_Agent",
            }
        data_cleaning_agent.invoke_messages(
            messages=before_msgs,
            user_instructions=last_human,
            data_raw=active_df,
        )
        response = data_cleaning_agent.response or {}
        merged = _merge_messages(before_msgs, response)
        merged["messages"] = _tag_messages(
            merged.get("messages"), "data_cleaning_agent"
        )
        summary_text = _format_result_with_llm(
            "data_cleaning_agent",
            response.get("data_cleaned"),
            _get_last_human(before_msgs),
            extra_text="Cleaning/imputation completed.",
        )
        if summary_text:
            merged["messages"].append(
                AIMessage(content=summary_text, name="data_cleaning_agent")
            )
        data_cleaned = response.get("data_cleaned")
        downstream_resets = (
            {
                "eda_artifacts": None,
                "viz_graph": None,
                "feature_data": None,
                "model_info": None,
                "mlflow_artifacts": None,
            }
            if data_cleaned is not None
            else {}
        )
        return {
            **merged,
            "data_cleaned": data_cleaned,
            "active_data_key": "data_cleaned" if data_cleaned is not None else state.get("active_data_key"),
            "artifacts": {
                **state.get("artifacts", {}),
                "data_cleaning": data_cleaned,
            },
            "last_worker": "Data_Cleaning_Agent",
            **downstream_resets,
        }

    def node_sql(state: SupervisorDSState):
        print("---SQL DATABASE---")
        before_msgs = list(state.get("messages", []) or [])
        last_human = _get_last_human(before_msgs)
        sql_database_agent.invoke_messages(
            messages=before_msgs,
            user_instructions=last_human,
        )
        response = sql_database_agent.response or {}
        merged = _merge_messages(before_msgs, response)
        merged["messages"] = _tag_messages(merged.get("messages"), "sql_database_agent")
        summary_text = _format_result_with_llm(
            "sql_database_agent",
            response.get("data_sql"),
            _get_last_human(before_msgs),
            extra_text=response.get("sql_query_code", ""),
        )
        if summary_text:
            merged["messages"].append(
                AIMessage(content=summary_text, name="sql_database_agent")
            )
        data_sql = response.get("data_sql")
        return {
            **merged,
            "data_sql": data_sql,
            "active_data_key": "data_sql" if data_sql is not None else state.get("active_data_key"),
            "artifacts": {
                **state.get("artifacts", {}),
                "sql": {
                    "sql_query_code": response.get("sql_query_code"),
                    "sql_database_function": response.get("sql_database_function"),
                    "data_sql": data_sql,
                },
            },
            "last_worker": "SQL_Database_Agent",
        }

    def node_eda(state: SupervisorDSState):
        print("---EDA TOOLS---")
        before_msgs = list(state.get("messages", []) or [])
        active_df = _ensure_df(
            _get_active_data(
                state,
                ["data_cleaned", "data_wrangled", "data_sql", "data_raw", "feature_data"],
            )
        )
        if _is_empty_df(active_df):
            return {
                "messages": [
                    AIMessage(
                        content="No dataset is available for EDA. Load a file (or run a SQL query) first.",
                        name="eda_tools_agent",
                    )
                ],
                "last_worker": "EDA_Tools_Agent",
            }
        eda_tools_agent.invoke_messages(
            messages=before_msgs,
            data_raw=active_df,
        )
        response = eda_tools_agent.response or {}
        merged = _merge_messages(before_msgs, response)
        merged["messages"] = _tag_messages(merged.get("messages"), "eda_tools_agent")
        print(
            f"  eda artifacts keys={response.get('eda_artifacts') and list(response.get('eda_artifacts').keys()) if isinstance(response.get('eda_artifacts'), dict) else None}"
        )
        summary_text = _format_result_with_llm(
            "eda_tools_agent",
            response.get("eda_artifacts", {}).get("describe_dataset")
            if isinstance(response.get("eda_artifacts"), dict)
            else None,
            _get_last_human(before_msgs),
            extra_text="EDA summary.",
        )
        if summary_text:
            merged["messages"].append(
                AIMessage(content=summary_text, name="eda_tools_agent")
            )
        eda_artifacts = response.get("eda_artifacts")
        return {
            **merged,
            "eda_artifacts": eda_artifacts,
            "artifacts": {
                **state.get("artifacts", {}),
                "eda": eda_artifacts,
            },
            "last_worker": "EDA_Tools_Agent",
        }

    def node_viz(state: SupervisorDSState):
        print("---DATA VISUALIZATION---")
        before_msgs = list(state.get("messages", []) or [])
        last_human = _get_last_human(before_msgs)
        active_df = _ensure_df(
            _get_active_data(
                state,
                ["data_cleaned", "data_wrangled", "data_sql", "data_raw", "feature_data"],
            )
        )
        if _is_empty_df(active_df):
            return {
                "messages": [
                    AIMessage(
                        content="No dataset is available to plot. Load a file (or run a SQL query) first.",
                        name="data_visualization_agent",
                    )
                ],
                "last_worker": "Data_Visualization_Agent",
            }
        data_visualization_agent.invoke_messages(
            messages=before_msgs,
            user_instructions=last_human,
            data_raw=active_df,
        )
        response = data_visualization_agent.response or {}
        merged = _merge_messages(before_msgs, response)
        merged["messages"] = _tag_messages(
            merged.get("messages"), "data_visualization_agent"
        )
        summary_text = _format_result_with_llm(
            "data_visualization_agent",
            None,
            _get_last_human(before_msgs),
            extra_text="Visualization generated.",
        )
        if summary_text:
            merged["messages"].append(
                AIMessage(content=summary_text, name="data_visualization_agent")
            )
        plotly_graph = response.get("plotly_graph")
        return {
            **merged,
            "viz_graph": plotly_graph,
            "artifacts": {
                **state.get("artifacts", {}),
                "viz": {
                    "plotly_graph": plotly_graph,
                    "data_visualization_function": response.get(
                        "data_visualization_function"
                    ),
                },
            },
            "last_worker": "Data_Visualization_Agent",
        }

    def node_fe(state: SupervisorDSState):
        print("---FEATURE ENGINEERING---")
        before_msgs = list(state.get("messages", []) or [])
        last_human = _get_last_human(before_msgs)
        active_df = _ensure_df(
            _get_active_data(
                state,
                ["data_cleaned", "data_wrangled", "data_sql", "data_raw", "feature_data"],
            )
        )
        if _is_empty_df(active_df):
            return {
                "messages": [
                    AIMessage(
                        content="No dataset is available for feature engineering. Load a file (or run a SQL query) first.",
                        name="feature_engineering_agent",
                    )
                ],
                "last_worker": "Feature_Engineering_Agent",
            }
        feature_engineering_agent.invoke_messages(
            messages=before_msgs,
            user_instructions=last_human,
            data_raw=active_df,
        )
        response = feature_engineering_agent.response or {}
        merged = _merge_messages(before_msgs, response)
        merged["messages"] = _tag_messages(
            merged.get("messages"), "feature_engineering_agent"
        )
        summary_text = _format_result_with_llm(
            "feature_engineering_agent",
            response.get("feature_engineered_data"),
            _get_last_human(before_msgs),
            extra_text="Feature engineering completed.",
        )
        if summary_text:
            merged["messages"].append(
                AIMessage(content=summary_text, name="feature_engineering_agent")
            )
        feature_data = response.get("feature_engineered_data")
        downstream_resets = (
            {"model_info": None, "mlflow_artifacts": None}
            if feature_data is not None
            else {}
        )
        return {
            **merged,
            "feature_data": feature_data,
            "active_data_key": "feature_data" if feature_data is not None else state.get("active_data_key"),
            "artifacts": {
                **state.get("artifacts", {}),
                "feature_engineering": response,
            },
            "last_worker": "Feature_Engineering_Agent",
            **downstream_resets,
        }

    def node_h2o(state: SupervisorDSState):
        print("---H2O ML---")
        before_msgs = list(state.get("messages", []) or [])
        last_human = _get_last_human(before_msgs)
        active_df = _ensure_df(
            _get_active_data(
                state,
                ["feature_data", "data_cleaned", "data_wrangled", "data_sql", "data_raw"],
            )
        )
        if _is_empty_df(active_df):
            return {
                "messages": [
                    AIMessage(
                        content="No dataset is available for modeling. Load data and (optionally) engineer features first.",
                        name="h2o_ml_agent",
                    )
                ],
                "last_worker": "H2O_ML_Agent",
            }
        h2o_ml_agent.invoke_messages(
            messages=before_msgs,
            user_instructions=last_human,
            data_raw=active_df,
        )
        response = h2o_ml_agent.response or {}
        merged = _merge_messages(before_msgs, response)
        merged["messages"] = _tag_messages(merged.get("messages"), "h2o_ml_agent")
        summary_text = _format_result_with_llm(
            "h2o_ml_agent",
            response.get("leaderboard"),
            _get_last_human(before_msgs),
            extra_text="H2O AutoML results.",
        )
        if summary_text:
            merged["messages"].append(
                AIMessage(content=summary_text, name="h2o_ml_agent")
            )
        leaderboard = response.get("leaderboard")
        return {
            **merged,
            "model_info": leaderboard,
            "artifacts": {
                **state.get("artifacts", {}),
                "h2o": response,
            },
            "last_worker": "H2O_ML_Agent",
        }

    def node_mlflow(state: SupervisorDSState):
        print("---MLFLOW TOOLS---")
        before_msgs = list(state.get("messages", []) or [])
        mlflow_tools_agent.invoke_messages(
            messages=before_msgs,
        )
        response = mlflow_tools_agent.response or {}
        merged = _merge_messages(before_msgs, response)
        merged["messages"] = _tag_messages(merged.get("messages"), "mlflow_tools_agent")
        summary_text = _format_result_with_llm(
            "mlflow_tools_agent",
            response.get("mlflow_artifacts"),
            _get_last_human(before_msgs),
            extra_text="MLflow artifacts.",
        )
        if summary_text:
            merged["messages"].append(
                AIMessage(content=summary_text, name="mlflow_tools_agent")
            )
        mlflow_artifacts = response.get("mlflow_artifacts")
        return {
            **merged,
            "mlflow_artifacts": mlflow_artifacts,
            "artifacts": {
                **state.get("artifacts", {}),
                "mlflow": mlflow_artifacts,
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

    def invoke_messages(
        self,
        messages: Sequence[BaseMessage],
        artifacts: Optional[dict] = None,
        **kwargs,
    ):
        """
        Invoke the team with a message list (recommended for supervisor/teams).
        """
        self.response = self._compiled_graph.invoke(
            {"messages": messages, "artifacts": artifacts or {}},
            **kwargs,
        )
        return None

    async def ainvoke_messages(
        self,
        messages: Sequence[BaseMessage],
        artifacts: Optional[dict] = None,
        **kwargs,
    ):
        """
        Async version of invoke_messages.
        """
        self.response = await self._compiled_graph.ainvoke(
            {"messages": messages, "artifacts": artifacts or {}},
            **kwargs,
        )
        return None

    def invoke_agent(
        self, user_instructions: str, artifacts: Optional[dict] = None, **kwargs
    ):
        """
        Convenience wrapper for a single human prompt.
        """
        msg = HumanMessage(content=user_instructions)
        return self.invoke_messages(messages=[msg], artifacts=artifacts, **kwargs)

    async def ainvoke_agent(
        self, user_instructions: str, artifacts: Optional[dict] = None, **kwargs
    ):
        msg = HumanMessage(content=user_instructions)
        return await self.ainvoke_messages(
            messages=[msg], artifacts=artifacts, **kwargs
        )

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
            if isinstance(msg, AIMessage) or getattr(msg, "role", None) in (
                "assistant",
                "ai",
            ):
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
