from __future__ import annotations

from typing import Sequence, TypedDict, Annotated, Optional, Dict, Any, List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from IPython.display import Markdown
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.json import parse_json_markdown
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.types import Checkpointer
from langgraph.graph.message import add_messages


# Keep the conversation context small to avoid token-rate-limit errors and
# to prevent verbose agent outputs (code/JSON) from bloating subsequent calls.
TEAM_MAX_MESSAGES = 20
TEAM_MAX_MESSAGE_CHARS = 2000


def _is_agent_output_report_message(m: BaseMessage) -> bool:
    """
    Detect verbose JSON "Agent Outputs" reports emitted by node_func_report_agent_outputs.
    These are useful for debugging, but they bloat the LLM context and often cause
    rate-limit errors in multi-step workflows. The UI surfaces details via artifacts/tabs.
    """
    if not isinstance(m, AIMessage):
        return False
    content = getattr(m, "content", None)
    if not isinstance(content, str) or not content:
        return False
    s = content.lstrip()
    if not s.startswith("{"):
        return False
    head = s[:1200]
    return '"report_title"' in head and (
        "Agent Outputs" in head or "Agent Output Summary" in head
    )


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
        if _is_agent_output_report_message(m):
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
    active_dataset_id: Optional[str]
    datasets: Dict[str, Any]
    handled_request_id: Optional[str]
    handled_steps: Dict[str, bool]
    attempted_steps: Dict[str, bool]
    workflow_plan_request_id: Optional[str]
    workflow_plan: Optional[dict]
    target_variable: Optional[str]

    # Shared data/artifacts
    data_raw: Optional[dict]
    data_sql: Optional[dict]
    data_wrangled: Optional[dict]
    data_cleaned: Optional[dict]
    eda_artifacts: Optional[dict]
    viz_graph: Optional[dict]
    feature_data: Optional[dict]
    model_info: Optional[dict]
    eval_artifacts: Optional[dict]
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
    model_evaluation_agent,
    workflow_planner_agent=None,
    checkpointer: Optional[Checkpointer] = None,
    temperature: float = 1.0,
):
    """
    Build a supervisor-led data science team using existing sub-agents.

    Args:
        model: LLM (or model name) for the supervisor router.
        workflow_planner_agent: WorkflowPlannerAgent instance (optional planning for multi-step prompts).
        data_loader_agent: DataLoaderToolsAgent instance.
        data_wrangling_agent: DataWranglingAgent instance.
        data_cleaning_agent: DataCleaningAgent instance.
        eda_tools_agent: EDAToolsAgent instance.
        data_visualization_agent: DataVisualizationAgent instance.
        sql_database_agent: SQLDatabaseAgent instance.
        feature_engineering_agent: FeatureEngineeringAgent instance.
        h2o_ml_agent: H2OMLAgent instance.
        model_evaluation_agent: ModelEvaluationAgent instance.
        mlflow_tools_agent: MLflowToolsAgent instance.
        checkpointer: optional LangGraph checkpointer.
        temperature: supervisor routing temperature.
    """

    subagent_names = [
        "Data_Loader_Tools_Agent",
        "Data_Merge_Agent",
        "Data_Wrangling_Agent",
        "Data_Cleaning_Agent",
        "EDA_Tools_Agent",
        "Data_Visualization_Agent",
        "SQL_Database_Agent",
        "Feature_Engineering_Agent",
        "H2O_ML_Agent",
        "Model_Evaluation_Agent",
        "MLflow_Logging_Agent",
        "MLflow_Tools_Agent",
    ]

    def _openai_requires_responses(model_name: str | None) -> bool:
        model_name = model_name.strip().lower() if isinstance(model_name, str) else ""
        if not model_name:
            return False
        if "codex" in model_name:
            return True
        return model_name in {"gpt-5.1-codex-mini"}

    if isinstance(model, str):
        llm_kwargs: dict[str, object] = {"model": model, "temperature": temperature}
        if _openai_requires_responses(model):
            llm_kwargs["use_responses_api"] = True
            llm_kwargs["output_version"] = "responses/v1"
        llm = ChatOpenAI(**llm_kwargs)
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
- Data_Merge_Agent: Deterministically merges multiple already-loaded datasets (join/concat) based on user/UX configuration. Use for combining datasets into a single modeling table. Must have 2+ datasets loaded/selected.
- Data_Wrangling_Agent: Can work with one or more datasets, performing operations such as joining/merging multiple datasets, reshaping, aggregating, encoding, creating computed features, and ensuring consistent data types. Capabilities: recommend_wrangling_steps, create_data_wrangling_code, execute_data_wrangling_code (transform/rename/format). Must have data loaded/ready.
- Data_Cleaning_Agent: Strong in cleaning data, removing anomalies, and fixing data issues. Capabilities: recommend_cleaning_steps, create_data_cleaner_code, execute_data_cleaner_code (impute/clean). Must have data loaded/ready.
- EDA_Tools_Agent: Strong in exploring data, analysing data, and providing information about the data. Has several powerful tools: describe_dataset, explain_data, visualize_missing, correlation_funnel, sweetviz (use for previews/head/describe). Must have data loaded/ready.
- Data_Visualization_Agent: Can generate Plotly charts based on user-defined instructions or default visualization steps. Must have data loaded/ready.  
- SQL_Database_Agent: Generate a SQL query based on the recommended steps and user instructions. Executes that SQL query against the provided database connection, returning the data results.
- Feature_Engineering_Agent: The agent applies various feature engineering techniques, such as encoding categorical variables, scaling numeric variables, creating interaction terms,and generating polynomial features. Must have data loaded/ready.
- H2O_ML_Agent: A Machine Learning agent that uses H2O's AutoML for training create_h2o_automl_code, execute_h2o_code (AutoML training/eval).
- Model_Evaluation_Agent: Evaluates a trained model on a holdout split and returns standardized metrics + plots (confusion matrix/ROC or residuals).
- MLflow_Logging_Agent: Logs workflow artifacts deterministically to MLflow (tables/figures/metrics) and returns the run id.
- MLflow_Tools_Agent: Can interact and run various tools related to accessing, interacting with, and retrieving information from MLflow. Has tools including: mlflow_search_experiments, mlflow_search_runs, mlflow_create_experiment, mlflow_predict_from_run_id, mlflow_launch_ui, mlflow_stop_ui, mlflow_list_artifacts, mlflow_download_artifacts, mlflow_list_registered_models, mlflow_search_registered_models, mlflow_get_model_version_details, mlflow_get_run_details, mlflow_transition_model_version_stage, mlflow_tracking_info, mlflow_ui_status,

Critical rule: only route to workers when the user explicitly asks for their capabilities. Do not assume next steps.

Routing guidance (explicit intent -> worker):
- Load/import/read file (e.g., "load data/churn_data.csv"): Data_Loader_Tools_Agent ONCE, then FINISH unless more is requested.
- Show first N rows / preview / head / describe: EDA_Tools_Agent then FINISH.
- Plot/chart/visual/graph: Data_Visualization_Agent.
- Merge/join/concat multiple datasets into one: Data_Merge_Agent.
- Clean/impute/wrangle/standardize: Data_Wrangling_Agent or Data_Cleaning_Agent.
- SQL/database/query/tables: SQL_Database_Agent.
- Feature creation/encoding: Feature_Engineering_Agent.
- Train/evaluate model/AutoML: H2O_ML_Agent.
- Evaluate model performance: Model_Evaluation_Agent.
- Log workflow to MLflow: MLflow_Logging_Agent.
- MLflow tracking/registry/UI: MLflow_Tools_Agent.

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
                "Given the conversation above, who should act next? Or FINISH? "
                "Respond with ONLY one of: {route_options}",
            ),
        ]
    ).partial(
        route_options=str(route_options), subagent_names=", ".join(subagent_names)
    )

    def _parse_router_output(text: str) -> dict[str, str]:
        """
        Parse router output into {"next": <route_option>}.

        Supports:
        - OpenAI function-calling JSON via JsonOutputFunctionsParser (handled separately)
        - Raw JSON / JSON-in-markdown
        - Plain-text worker name
        """
        t = (text or "").strip()
        if not t:
            return {"next": "FINISH"}

        # Try JSON (raw or fenced) first.
        try:
            parsed = parse_json_markdown(t)
            if isinstance(parsed, dict):
                nxt = parsed.get("next")
                if isinstance(nxt, str) and nxt in route_options:
                    return {"next": nxt}
        except Exception:
            pass

        lower = t.lower()
        for opt in route_options:
            if opt.lower() in lower:
                return {"next": opt}

        # Best-effort: handle formats like "next: Feature_Engineering_Agent"
        try:
            import re

            m = re.search(r"(?:next|route)\\s*[:=]\\s*([A-Za-z0-9_]+)", t, flags=re.I)
            if m:
                cand = m.group(1).strip()
                for opt in route_options:
                    if cand == opt or cand.lower() == opt.lower():
                        return {"next": opt}
        except Exception:
            pass

        return {"next": "FINISH"}

    # Router chain:
    # - For OpenAI models: use function-calling for high-precision routing.
    # - For other chat models (e.g., Ollama): fall back to strict text parsing.
    if isinstance(llm, ChatOpenAI):
        supervisor_chain = (
            prompt
            | llm.bind(functions=[function_def], function_call={"name": "route"})
            | JsonOutputFunctionsParser()
        )
    else:
        supervisor_chain = (
            prompt | llm | StrOutputParser() | RunnableLambda(_parse_router_output)
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

    # Optional LLM intent parser (enable via artifacts.config.use_llm_intent_parser).
    def _parse_intent(
        msgs: Sequence[BaseMessage], *, use_llm: bool = False
    ) -> dict[str, bool]:
        last_human_text = ""
        for m in reversed(msgs or []):
            role = getattr(m, "role", getattr(m, "type", None))
            if role in ("human", "user"):
                last_human_text = getattr(m, "content", "") or ""
                break
        last_human = last_human_text.lower()

        import re

        def has(*words):
            return any(w in last_human for w in words)

        def has_word(*words):
            return any(re.search(rf"\\b{re.escape(w)}\\b", last_human) for w in words)

        wants_workflow = has(
            "workflow",
            "end-to-end",
            "end to end",
            "full pipeline",
            "full data science",
            "data science workflow",
            "ds workflow",
        )
        wants_list_files = has(
            "what files",
            "list files",
            "show files",
            "files are in",
            "directory contents",
            "list directory",
            "list only",
        ) and has("file", "files", "csv", ".csv", "./", "directory", "folder", "data")
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
        merge_signal = has("merge", "concat", "append", "union", "combine")
        join_signal = has("join")
        merge_context = (
            has("with", "between", "together", "into", "using")
            or (" on " in last_human)
            or has("dataset", "datasets", "dataframe", "table", "tables")
            or has("left join", "right join", "inner join", "outer join")
            or has(".csv", ".parquet", ".xlsx", ".xls", ".json")
        )
        wants_merge = bool(merge_signal or (join_signal and merge_context))
        # NOTE: "standardize" is ambiguous (scaling vs column-name normalization). Treat it as
        # wrangling only when it clearly refers to column naming/structure.
        standardize_column_names = has("standardize") and has(
            "column name",
            "column names",
            "rename column",
            "rename columns",
            "snake case",
            "snake_case",
        )
        wants_wrangling = (
            has(
                "wrangle",
                "transform",
                "reshape",
                "pivot",
                "melt",
                "rename",
                "format",
            )
            or standardize_column_names
        )
        wants_eda = has(
            "describe", "eda", "summary", "correlation", "sweetviz", "missingness"
        )
        # Feature engineering is often referred to as "features", but "feature-engineered data"
        # can also be a *reference* to an existing dataset. Be conservative: require an action signal.
        feature_action = has(
            "encode",
            "one-hot",
            "one hot",
            "label encoding",
            "target encoding",
            "scale",
            "scaling",
            "standardize",
            "normalize",
            "model-ready features",
            "model ready features",
            "feature engineering",
            "feature-engineering",
            "feature engineer",
            "engineer features",
            "create features",
            "build features",
            "make features",
            "generate features",
        )
        references_feature_engineered_data = (
            ("feature-engineered" in last_human or "feature engineered" in last_human)
            and ("data" in last_human or "dataset" in last_human)
            and ("from" in last_human or "using" in last_human or "on" in last_human)
        ) or (
            ("engineered features" in last_human or "engineered feature" in last_human)
            and ("from" in last_human or "using" in last_human or "on" in last_human)
        )
        wants_feature = bool(feature_action)

        # "model" is ambiguous (e.g., a product "bike model" vs an ML model). Prefer a conservative
        # interpretation: only trigger ML modeling when the user explicitly requests training/AutoML/prediction,
        # or when ML terminology appears without a feature-engineering intent.
        explicit_modeling = has(
            "train",
            "automl",
            "fit",
            "tune",
            "cross-validation",
            "cross validation",
            "cv",
            "hyperparameter",
        ) or has_word("predict")
        ml_context = has(
            "classification",
            "classify",
            "regression",
            "xgboost",
            "random forest",
            "lightgbm",
            "catboost",
            "logistic",
            "neural network",
            "deep learning",
        )
        ml_signal = explicit_modeling or (ml_context and not wants_feature)
        model_word = "model" in last_human
        product_model_context = has(
            "bike model",
            "car model",
            "product model",
            "model year",
            "phone model",
            "vehicle model",
        ) or (wants_viz and has("by model", "per model", "for each model"))
        model_ready_context = has(
            "model-ready", "model ready", "model-ready data", "model ready data"
        )
        wants_model = bool(
            ml_signal
            or (
                model_word
                and has("build", "create", "fit", "train", "tune", "predict", "develop")
                and not product_model_context
                and not model_ready_context
            )
        )
        wants_eval = has(
            "evaluate",
            "evaluation",
            "metrics",
            "performance",
            "confusion matrix",
            "roc",
            "auc",
            "precision",
            "recall",
            "f1",
        )

        wants_load = has("load", "import", "read csv", "read file", "open file")
        mentions_file = (
            (".csv" in last_human)
            or (".parquet" in last_human)
            or (".xlsx" in last_human)
            or ("file" in last_human)
        )
        wants_mlflow = "mlflow" in last_human
        wants_mlflow_tools = wants_mlflow and has(
            "ui",
            "launch",
            "stop",
            "status",
            "list",
            "search",
            "experiment",
            "run",
            "artifact",
            "tracking",
            "uri",
            "registry",
            "registered model",
            "model version",
        )
        wants_mlflow_log = wants_mlflow and has(
            "log",
            "logging",
            "save to mlflow",
            "track",
            "record",
        )

        # If the user explicitly wants an end-to-end workflow, enable common steps.
        if wants_workflow:
            wants_clean = True
            wants_eda = True
            wants_viz = True
            wants_model = True
            wants_eval = True

        wants_more_processing = any(
            [
                wants_preview,
                wants_viz,
                wants_sql,
                wants_merge,
                wants_clean,
                wants_wrangling,
                wants_eda,
                wants_feature,
                wants_model,
            ]
        )
        load_only = wants_load and mentions_file and not wants_more_processing
        heuristic_intents: dict[str, bool] = {
            "list_files": wants_list_files,
            "preview": wants_preview,
            "merge": wants_merge,
            "viz": wants_viz,
            "sql": wants_sql,
            "clean": wants_clean,
            "wrangle": wants_wrangling,
            "eda": wants_eda,
            "feature": wants_feature,
            "model": wants_model,
            "evaluate": wants_eval,
            "mlflow": wants_mlflow,
            "mlflow_log": wants_mlflow_log,
            "mlflow_tools": wants_mlflow_tools,
            "workflow": wants_workflow,
            "load": wants_load and mentions_file,
            "load_only": load_only,
        }

        if not use_llm:
            return heuristic_intents

        # LLM-based classification can resolve ambiguous prompts (e.g., "bike model" vs ML model).
        import json

        allowed_keys = list(heuristic_intents.keys())
        llm_intents: dict[str, bool] = {}
        try:
            intent_prompt = (
                "You classify user intent for a data-science assistant router.\n"
                "Return ONLY valid JSON with boolean fields:\n"
                f"{', '.join(allowed_keys)}\n\n"
                "Guidelines:\n"
                "- Set `viz` when the user asks to plot/chart/visualize.\n"
                "- Set `merge` when the user asks to merge/join/concat multiple datasets.\n"
                "- Sweetviz/D-Tale requests are EDA reports: set `eda` true and keep `viz` false unless an additional plot is requested.\n"
                "- Set `model` ONLY for ML modeling (train/AutoML/predict), not product/bike 'model'.\n"
                "- Set `load_only` only when the user only wants data loaded (no preview/eda/viz/etc).\n"
                "- If `mlflow_log` or `mlflow_tools` is true, set `mlflow` true.\n"
                "- If `workflow` is true, you may also set common steps true (clean/eda/viz/model/evaluate).\n"
            )
            intent_llm = llm.bind(temperature=1.0) if hasattr(llm, "bind") else llm
            raw = intent_llm.invoke(
                [
                    SystemMessage(content=intent_prompt),
                    HumanMessage(content=last_human_text),
                ]
            )
            content = getattr(raw, "content", raw)
            if not isinstance(content, str):
                content = str(content)

            try:
                parsed = json.loads(content)
            except Exception:
                m = re.search(r"\{.*\}", content, flags=re.DOTALL)
                parsed = json.loads(m.group(0)) if m else {}

            if isinstance(parsed, dict):
                for k in allowed_keys:
                    if k in parsed:
                        llm_intents[k] = bool(parsed.get(k))
        except Exception:
            llm_intents = {}

        if llm_intents.get("load_only"):
            llm_intents["load"] = True
        if llm_intents.get("mlflow_log") or llm_intents.get("mlflow_tools"):
            llm_intents["mlflow"] = True
        if llm_intents.get("workflow"):
            llm_intents["clean"] = True
            llm_intents["eda"] = True
            llm_intents["viz"] = True
            llm_intents["model"] = True
            llm_intents["evaluate"] = True

        # Sweetviz/D-Tale are EDA report tools; do not route to the charting agent unless the
        # user explicitly asked for a plot/chart/graph in addition to the report.
        wants_eda_report = has(
            "sweetviz",
            "dtale",
            "d-tale",
            "exploratory report",
            "profiling report",
            "eda report",
        )
        if wants_eda_report:
            llm_intents["eda"] = True
        explicit_plot_request = has("plot", "chart", "graph") or has_word("visualize")
        if wants_eda_report and not explicit_plot_request:
            llm_intents["viz"] = False

        # If the user is referencing feature-engineered data, that is typically a dataset selection hint,
        # not a request to run feature engineering again.
        if references_feature_engineered_data and not bool(feature_action):
            llm_intents["feature"] = False

        # Conflict resolution: feature engineering requests often get misclassified as wrangling/cleaning.
        # If the user is clearly asking to engineer model-ready features, keep the plan minimal and avoid
        # running extra transforms unless explicitly requested.
        explicit_wrangling = standardize_column_names or has(
            "wrangle",
            "transform",
            "reshape",
            "pivot",
            "melt",
            "rename",
        )
        explicit_cleaning = has(
            "clean",
            "impute",
            "missing",
            "null",
            "na",
            "outlier",
            "duplicate",
            "deduplicate",
        )
        if bool(feature_action) and not explicit_wrangling:
            llm_intents["wrangle"] = False
        if bool(feature_action) and not explicit_cleaning:
            llm_intents["clean"] = False
        if llm_intents.get("preview") and not llm_intents.get("workflow"):
            llm_intents["merge"] = False

        return {**heuristic_intents, **llm_intents}

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
        # Ensure every message has an ID so per-request step tracking is reliable even
        # when upstream callers don't set message IDs (e.g., Streamlit chat history).
        try:
            clean_msgs = add_messages([], clean_msgs)  # type: ignore[arg-type]
        except Exception:
            pass
        cfg = (state.get("artifacts") or {}).get("config") or {}
        use_llm_intent_parser = (
            bool(cfg.get("use_llm_intent_parser")) if isinstance(cfg, dict) else False
        )
        intents = _parse_intent(clean_msgs, use_llm=use_llm_intent_parser)
        proactive_mode = (
            bool(cfg.get("proactive_workflow_mode")) if isinstance(cfg, dict) else False
        )

        # Track per-user-request steps (within the current user message) to support
        # deterministic sequencing for multi-step prompts.
        last_human_msg = None
        for m in reversed(clean_msgs or []):
            role = getattr(m, "role", getattr(m, "type", None))
            if role in ("human", "user"):
                last_human_msg = m
                break
        current_request_id = (
            getattr(last_human_msg, "id", None) if last_human_msg else None
        )

        handled_request_id = state.get("handled_request_id")
        handled_steps: dict[str, bool] = dict(state.get("handled_steps") or {})
        attempted_steps: dict[str, bool] = dict(state.get("attempted_steps") or {})
        is_new_request = (
            current_request_id is not None and current_request_id != handled_request_id
        )
        if is_new_request:
            handled_request_id = current_request_id
            handled_steps = {}
            attempted_steps = {}
            # Reset workflow plan per user request
            state_plan_req = None
            state_plan = None
        else:
            state_plan_req = state.get("workflow_plan_request_id")
            state_plan = state.get("workflow_plan")

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

        datasets, active_dataset_id = _ensure_dataset_registry(state)
        state_with_datasets: dict = {
            **(state or {}),
            "active_data_key": active_data_key,
            "datasets": datasets,
            "active_dataset_id": active_dataset_id,
        }

        # Handle explicit dataset switching requests (no agent needed).
        last_human_text = _get_last_human(clean_msgs)
        requested_dataset_id = None
        try:
            import re

            lower = (last_human_text or "").lower()
            wants_switch = any(
                k in lower
                for k in (
                    "use dataset",
                    "switch dataset",
                    "set dataset",
                    "use the dataset",
                    "switch to dataset",
                )
            )
            if wants_switch and isinstance(datasets, dict) and datasets:
                m = re.search(r"\\bdataset\\b\\s*[:#]?\\s*([a-zA-Z0-9_\\-]+)", lower)
                token = (m.group(1) if m else "").strip()
                if token.isdigit():
                    ordered = sorted(
                        datasets.items(),
                        key=lambda kv: float(kv[1].get("created_ts") or 0.0)
                        if isinstance(kv[1], dict)
                        else 0.0,
                        reverse=True,
                    )
                    idx = int(token) - 1
                    if 0 <= idx < len(ordered):
                        requested_dataset_id = ordered[idx][0]
                elif token and token in datasets:
                    requested_dataset_id = token

            if requested_dataset_id is None and isinstance(datasets, dict) and datasets:
                # Convenience switching by stage
                stage_hint = None
                if "use sql" in lower or "use sql results" in lower:
                    stage_hint = "sql"
                elif "use cleaned" in lower or "use clean" in lower:
                    stage_hint = "cleaned"
                elif "use wrangled" in lower or "use wrangle" in lower:
                    stage_hint = "wrangled"
                elif "use features" in lower or "use feature" in lower:
                    stage_hint = "feature"
                elif "use raw" in lower:
                    stage_hint = "raw"
                if stage_hint:
                    candidates = [
                        (float(e.get("created_ts") or 0.0), did)
                        for did, e in datasets.items()
                        if isinstance(e, dict) and e.get("stage") == stage_hint
                    ]
                    if candidates:
                        candidates.sort(reverse=True)
                        requested_dataset_id = candidates[0][1]
        except Exception:
            requested_dataset_id = None

        if requested_dataset_id and requested_dataset_id != active_dataset_id:
            selected = (
                datasets.get(requested_dataset_id)
                if isinstance(datasets, dict)
                else None
            )
            label = (
                selected.get("label")
                if isinstance(selected, dict)
                else requested_dataset_id
            )
            msg = AIMessage(
                content=f"Switched active dataset to `{label}` (`{requested_dataset_id}`).",
                name="supervisor",
            )
            return {
                "messages": [msg],
                "next": "FINISH",
                "active_data_key": active_data_key,
                "datasets": datasets,
                "active_dataset_id": requested_dataset_id,
                "handled_request_id": handled_request_id,
                "handled_steps": handled_steps,
                "attempted_steps": attempted_steps,
                "workflow_plan_request_id": state_plan_req,
                "workflow_plan": state_plan,
            }

        data_ready = (
            _get_active_data(
                state_with_datasets,
                [
                    "data_cleaned",
                    "data_wrangled",
                    "data_sql",
                    "data_raw",
                    "feature_data",
                ],
            )
            is not None
        )
        last_worker = state.get("last_worker")

        def _loader_loaded_dataset(loader_artifacts: Any) -> bool:
            """
            Determine whether the loader actually loaded a dataset (vs listing a directory).
            This matters because node_loader intentionally preserves previous data_raw when no load occurred.
            """
            if not loader_artifacts:
                return False
            if isinstance(loader_artifacts, dict):
                # Single load_file artifact shape: {"status":"ok","data":{...},...}
                if (
                    loader_artifacts.get("status") == "ok"
                    and loader_artifacts.get("data") is not None
                ):
                    return True
                for key, val in loader_artifacts.items():
                    tool_name = str(key)
                    if tool_name.startswith("load_file") and isinstance(val, dict):
                        if val.get("status") == "ok" and val.get("data") is not None:
                            return True
                    if tool_name.startswith("load_directory") and isinstance(val, dict):
                        for _fname, info in val.items():
                            if (
                                isinstance(info, dict)
                                and info.get("status") == "ok"
                                and info.get("data") is not None
                            ):
                                return True
            return False

        def _loader_listed_directory(loader_artifacts: Any) -> bool:
            if not loader_artifacts:
                return False
            if isinstance(loader_artifacts, list):
                return True
            if isinstance(loader_artifacts, dict):
                for key in loader_artifacts.keys():
                    tool_name = str(key)
                    if tool_name.startswith("list_directory") or tool_name.startswith(
                        "search_files_by_pattern"
                    ):
                        return True
            return False

        # Mark completed steps for this request based on the last worker.
        if not is_new_request and last_worker:
            if last_worker == "Data_Loader_Tools_Agent":
                loader_art = (state.get("artifacts") or {}).get("data_loader")
                if _loader_loaded_dataset(loader_art):
                    handled_steps["load"] = True
                if _loader_listed_directory(loader_art):
                    handled_steps["list_files"] = True
            elif (
                last_worker == "Data_Merge_Agent"
                and (state.get("artifacts") or {}).get("merge") is not None
            ):
                handled_steps["merge"] = True
            elif (
                last_worker == "SQL_Database_Agent"
                and state.get("data_sql") is not None
            ):
                handled_steps["sql"] = True
            elif (
                last_worker == "Data_Wrangling_Agent"
                and state.get("data_wrangled") is not None
            ):
                handled_steps["wrangle"] = True
            elif (
                last_worker == "Data_Cleaning_Agent"
                and state.get("data_cleaned") is not None
            ):
                handled_steps["clean"] = True
            elif (
                last_worker == "EDA_Tools_Agent"
                and state.get("eda_artifacts") is not None
            ):
                handled_steps["eda"] = True
            elif (
                last_worker == "Data_Visualization_Agent"
                and state.get("viz_graph") is not None
            ):
                handled_steps["viz"] = True
            elif (
                last_worker == "Feature_Engineering_Agent"
                and state.get("feature_data") is not None
            ):
                handled_steps["feature"] = True
            elif last_worker == "H2O_ML_Agent" and state.get("model_info") is not None:
                handled_steps["model"] = True
            elif (
                last_worker == "Model_Evaluation_Agent"
                and state.get("eval_artifacts") is not None
            ):
                handled_steps["evaluate"] = True
            elif (
                last_worker == "MLflow_Logging_Agent"
                and state.get("mlflow_artifacts") is not None
            ):
                handled_steps["mlflow_log"] = True
            elif (
                last_worker == "MLflow_Tools_Agent"
                and state.get("mlflow_artifacts") is not None
            ):
                handled_steps["mlflow_tools"] = True

        step_to_worker = {
            "list_files": "Data_Loader_Tools_Agent",
            "load": "Data_Loader_Tools_Agent",
            "merge": "Data_Merge_Agent",
            "sql": "SQL_Database_Agent",
            "wrangle": "Data_Wrangling_Agent",
            "clean": "Data_Cleaning_Agent",
            "eda": "EDA_Tools_Agent",
            "viz": "Data_Visualization_Agent",
            "feature": "Feature_Engineering_Agent",
            "model": "H2O_ML_Agent",
            "evaluate": "Model_Evaluation_Agent",
            "mlflow_log": "MLflow_Logging_Agent",
            "mlflow_tools": "MLflow_Tools_Agent",
        }

        # Use the workflow planner for multi-step prompts when available.
        wants_steps_count = sum(
            1
            for k in (
                "list_files",
                "load",
                "merge",
                "sql",
                "wrangle",
                "clean",
                "eda",
                "preview",
                "viz",
                "feature",
                "model",
                "evaluate",
                "mlflow_log",
                "mlflow_tools",
                "workflow",
            )
            if intents.get(k)
        )
        use_planner = bool(
            proactive_mode
            or intents.get("workflow")
            or intents.get("model")
            or intents.get("evaluate")
            or intents.get("mlflow_log")
            or intents.get("mlflow_tools")
            or wants_steps_count >= 3
        )

        planned_steps: list[str] | None = None
        plan_questions: list[str] = []
        plan_notes: list[str] = []
        planner_messages: list[BaseMessage] = []
        planned_target: Optional[str] = state.get("target_variable")
        if (
            use_planner
            and workflow_planner_agent is not None
            and current_request_id is not None
        ):
            if state_plan_req == current_request_id and isinstance(state_plan, dict):
                planned_steps = (
                    state_plan.get("steps")
                    if isinstance(state_plan.get("steps"), list)
                    else None
                )
                plan_questions = (
                    state_plan.get("questions")
                    if isinstance(state_plan.get("questions"), list)
                    else []
                )
                plan_notes = (
                    state_plan.get("notes")
                    if isinstance(state_plan.get("notes"), list)
                    else []
                )
            else:
                # Provide a minimal context snapshot to help planning.
                context = {
                    "data_ready": bool(data_ready),
                    "active_data_key": active_data_key,
                    "has_data_raw": state.get("data_raw") is not None,
                    "has_data_cleaned": state.get("data_cleaned") is not None,
                    "has_data_wrangled": state.get("data_wrangled") is not None,
                    "has_feature_data": state.get("feature_data") is not None,
                    "has_sql": state.get("data_sql") is not None,
                    "has_model_info": state.get("model_info") is not None,
                    "proactive_workflow_mode": proactive_mode,
                }
                try:
                    workflow_planner_agent.invoke_messages(
                        messages=clean_msgs,
                        context=context,
                    )
                    plan = workflow_planner_agent.response or {}
                except Exception:
                    plan = {}
                planned_steps = (
                    plan.get("steps") if isinstance(plan.get("steps"), list) else None
                )
                plan_questions = (
                    plan.get("questions")
                    if isinstance(plan.get("questions"), list)
                    else []
                )
                plan_notes = (
                    plan.get("notes") if isinstance(plan.get("notes"), list) else []
                )
                planned_target = plan.get("target_variable") or planned_target
                state_plan_req = current_request_id
                state_plan = {
                    "steps": planned_steps or [],
                    "target_variable": planned_target,
                    "questions": plan_questions,
                    "notes": plan_notes,
                }
                if planned_steps:
                    pretty_steps = " → ".join(str(s) for s in planned_steps)
                    note_text = (
                        "\n".join(f"- {n}" for n in plan_notes) if plan_notes else ""
                    )
                    msg = f"Planned workflow: {pretty_steps}"
                    if note_text:
                        msg = msg + "\n\nNotes:\n" + note_text
                    planner_messages = [
                        AIMessage(content=msg, name="workflow_planner_agent")
                    ]

            # If the planner needs user input, ask and stop.
            if plan_questions and not (planned_steps and len(planned_steps) > 0):
                question_text = "\n".join(f"- {q}" for q in plan_questions)
                note_text = (
                    "\n".join(f"- {n}" for n in plan_notes) if plan_notes else ""
                )
                msg = "To run the workflow, I need:\n" + question_text
                if note_text:
                    msg = msg + "\n\nNotes:\n" + note_text
                return {
                    "messages": [AIMessage(content=msg, name="workflow_planner_agent")],
                    "next": "FINISH",
                    "active_data_key": active_data_key,
                    "datasets": datasets,
                    "active_dataset_id": active_dataset_id,
                    "handled_request_id": handled_request_id,
                    "handled_steps": handled_steps,
                    "attempted_steps": attempted_steps,
                    "workflow_plan_request_id": state_plan_req,
                    "workflow_plan": state_plan,
                }

        recognized_intent = any(
            [
                intents.get("list_files"),
                intents.get("load_only"),
                intents.get("load"),
                intents.get("merge"),
                intents.get("sql"),
                intents.get("wrangle"),
                intents.get("clean"),
                intents.get("eda"),
                intents.get("preview"),
                intents.get("viz"),
                intents.get("feature"),
                intents.get("model"),
                intents.get("evaluate"),
                intents.get("mlflow"),
                intents.get("mlflow_log"),
                intents.get("mlflow_tools"),
                intents.get("workflow"),
            ]
        )
        recognized_intent = bool(planned_steps) or recognized_intent

        # Deterministic, step-aware routing for common data science workflows.
        if recognized_intent:
            steps: list[str] = []

            # If we have a planner-derived step list, trust it.
            if planned_steps:
                steps = [str(s) for s in planned_steps if isinstance(s, str)]
            else:
                if intents.get("list_files"):
                    steps.append("list_files")

                # If the user asked to load a file, do that first.
                if intents.get("load") or intents.get("load_only"):
                    steps.append("load")

                # SQL can also be a data acquisition step.
                if intents.get("sql"):
                    steps.append("sql")

                # If the user requested data-dependent work but no data is present, attempt a load first.
                needs_data = any(
                    [
                        intents.get("merge"),
                        intents.get("wrangle"),
                        intents.get("clean"),
                        intents.get("eda"),
                        intents.get("preview"),
                        intents.get("viz"),
                        intents.get("feature"),
                        intents.get("model"),
                        intents.get("evaluate"),
                    ]
                )
                if (
                    not data_ready
                    and needs_data
                    and not (
                        intents.get("load")
                        or intents.get("load_only")
                        or intents.get("sql")
                    )
                ):
                    steps.insert(0, "load")

                # Transformations
                if intents.get("merge"):
                    steps.append("merge")
                if intents.get("wrangle"):
                    steps.append("wrangle")
                if intents.get("clean"):
                    steps.append("clean")

                # EDA / preview: if the user is explicitly loading, prefer the loader preview and avoid an extra EDA pass.
                wants_preview_via_eda = intents.get("preview") and not (
                    intents.get("load") or intents.get("load_only")
                )
                if intents.get("eda") or wants_preview_via_eda:
                    steps.append("eda")

                # Visualization
                if intents.get("viz"):
                    steps.append("viz")

                # Feature engineering and modeling
                if intents.get("feature"):
                    steps.append("feature")
                if intents.get("model"):
                    steps.append("model")
                if intents.get("evaluate"):
                    steps.append("evaluate")

                # MLflow logging and tools (inspection/UI)
                if intents.get("mlflow_log"):
                    steps.append("mlflow_log")
                if intents.get("mlflow_tools"):
                    steps.append("mlflow_tools")

            if not steps:
                print("  recognized intent but no actionable steps -> fallback router")
            else:
                for step in steps:
                    if handled_steps.get(step):
                        continue
                    worker = step_to_worker.get(step)
                    if not worker:
                        continue

                    # Prevent infinite loops: don't attempt the same step twice within one user request
                    # unless it was actually completed.
                    if attempted_steps.get(step) and not handled_steps.get(step):
                        print(f"  step '{step}' already attempted -> FINISH")
                        return {
                            **(
                                {"messages": planner_messages}
                                if planner_messages
                                else {}
                            ),
                            "next": "FINISH",
                            "active_data_key": active_data_key,
                            "datasets": datasets,
                            "active_dataset_id": active_dataset_id,
                            "handled_request_id": handled_request_id,
                            "handled_steps": handled_steps,
                            "attempted_steps": attempted_steps,
                            "workflow_plan_request_id": state_plan_req,
                            "workflow_plan": state_plan,
                            "target_variable": planned_target,
                        }

                    # Guard data-dependent steps.
                    if (
                        step
                        in (
                            "merge",
                            "wrangle",
                            "clean",
                            "eda",
                            "viz",
                            "feature",
                            "model",
                            "evaluate",
                        )
                        and not data_ready
                    ):
                        print(
                            f"  step '{step}' requires data but none is ready -> Data_Loader_Tools_Agent"
                        )
                        attempted_steps["load"] = True
                        return {
                            **(
                                {"messages": planner_messages}
                                if planner_messages
                                else {}
                            ),
                            "next": "Data_Loader_Tools_Agent",
                            "active_data_key": active_data_key,
                            "datasets": datasets,
                            "active_dataset_id": active_dataset_id,
                            "handled_request_id": handled_request_id,
                            "handled_steps": handled_steps,
                            "attempted_steps": attempted_steps,
                            "workflow_plan_request_id": state_plan_req,
                            "workflow_plan": state_plan,
                            "target_variable": planned_target,
                        }

                    print(f"  next_step='{step}' -> {worker}")
                    attempted_steps[step] = True
                    return {
                        **({"messages": planner_messages} if planner_messages else {}),
                        "next": worker,
                        "active_data_key": active_data_key,
                        "datasets": datasets,
                        "active_dataset_id": active_dataset_id,
                        "handled_request_id": handled_request_id,
                        "handled_steps": handled_steps,
                        "attempted_steps": attempted_steps,
                        "workflow_plan_request_id": state_plan_req,
                        "workflow_plan": state_plan,
                        "target_variable": planned_target,
                    }

                print("  all requested steps handled -> FINISH")
                return {
                    **({"messages": planner_messages} if planner_messages else {}),
                    "next": "FINISH",
                    "active_data_key": active_data_key,
                    "datasets": datasets,
                    "active_dataset_id": active_dataset_id,
                    "handled_request_id": handled_request_id,
                    "handled_steps": handled_steps,
                    "attempted_steps": attempted_steps,
                    "workflow_plan_request_id": state_plan_req,
                    "workflow_plan": state_plan,
                    "target_variable": planned_target,
                }

        result = supervisor_chain.invoke(
            {"messages": clean_msgs, "last_worker": state.get("last_worker")}
        )
        next_worker = result.get("next")
        print(
            f"  data_ready={data_ready}, last_worker={last_worker}, router_next={next_worker}"
        )

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
                elif not any(
                    [
                        intents.get("viz"),
                        intents.get("eda"),
                        intents.get("clean"),
                        intents.get("wrangle"),
                        intents.get("sql"),
                        intents.get("feature"),
                        intents.get("model"),
                        intents.get("mlflow"),
                    ]
                ):
                    next_worker = "FINISH"
                else:
                    next_worker = "Data_Wrangling_Agent"

        # Keep active_data_key stable unless a worker changes it.
        return {
            "next": next_worker,
            "active_data_key": active_data_key,
            "datasets": datasets,
            "active_dataset_id": active_dataset_id,
            "handled_request_id": handled_request_id,
            "handled_steps": handled_steps,
            "attempted_steps": attempted_steps,
            "workflow_plan_request_id": state_plan_req,
            "workflow_plan": state_plan,
            "target_variable": planned_target,
        }

    def _trim_messages(
        msgs: Sequence[BaseMessage],
        max_messages: int = TEAM_MAX_MESSAGES,
        max_chars: int = TEAM_MAX_MESSAGE_CHARS,
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

    def _merge_messages(before_messages: Sequence[BaseMessage], response: dict) -> dict:
        response_msgs = list(response.get("messages") or [])
        if not response_msgs:
            return {"messages": []}

        before_ids = {
            getattr(m, "id", None)
            for m in (before_messages or [])
            if getattr(m, "id", None) is not None
        }

        # Only keep assistant/ai messages created by the sub-agent.
        new_msgs: list[BaseMessage] = []
        seen_new_ids: set[str] = set()
        for m in response_msgs:
            mid = getattr(m, "id", None)
            if mid is not None and mid in before_ids:
                continue
            role = getattr(m, "type", None) or getattr(m, "role", None)
            if role in ("assistant", "ai") or isinstance(m, AIMessage):
                if mid is not None and mid in seen_new_ids:
                    continue
                new_msgs.append(m)
                if mid is not None:
                    seen_new_ids.add(mid)

        # Fallback: if we couldn't compute a clean delta, at least keep the last AI message.
        if not new_msgs:
            for m in reversed(response_msgs):
                role = getattr(m, "type", None) or getattr(m, "role", None)
                if role in ("assistant", "ai") or isinstance(m, AIMessage):
                    new_msgs = [m]
                    break

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

    def _format_dataset_with_llm(
        df_dict: dict, last_human: str, max_rows: int = 10, max_cols: int = 6
    ):
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

    DATASET_REGISTRY_MAX = 10
    DATASET_FINGERPRINT_MAX_ROWS = 200
    DATASET_SCHEMA_MAX_COLS = 200

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

    def _dataset_meta(
        data: Any,
    ) -> tuple[
        Any, list[str] | None, list[dict[str, str]] | None, str | None, str | None
    ]:
        df = _ensure_df(data)
        shape = _shape(df)
        cols = None
        schema = None
        schema_hash = None
        fingerprint = None
        try:
            cols = [str(c) for c in list(getattr(df, "columns", []))]
            cols = cols[:DATASET_SCHEMA_MAX_COLS] if cols else None
        except Exception:
            cols = None

        try:
            import hashlib
            import pandas as pd

            if isinstance(df, pd.DataFrame):
                # Schema (sorted for stability)
                col_order = sorted([str(c) for c in list(df.columns)])
                schema = [
                    {"name": c, "dtype": str(df[c].dtype) if c in df.columns else ""}
                    for c in col_order[:DATASET_SCHEMA_MAX_COLS]
                ]
                schema_str = "|".join(f"{r['name']}:{r['dtype']}" for r in schema)
                schema_hash = (
                    hashlib.sha256(schema_str.encode("utf-8")).hexdigest()
                    if schema_str
                    else None
                )

                # Fingerprint: stable hash over a capped row sample
                df_sample = (
                    df.reindex(columns=col_order)
                    .head(DATASET_FINGERPRINT_MAX_ROWS)
                    .reset_index(drop=True)
                )
                try:
                    from pandas.util import hash_pandas_object

                    row_hashes = hash_pandas_object(df_sample, index=False).values
                    fingerprint = hashlib.sha256(row_hashes.tobytes()).hexdigest()
                except Exception:
                    # Fallback: hash a small JSON snapshot (less stable across versions)
                    snap = df_sample.to_json(orient="split", date_format="iso")
                    fingerprint = hashlib.sha256(snap.encode("utf-8")).hexdigest()
        except Exception:
            schema = None
            schema_hash = None
            fingerprint = None
        return shape, cols, schema, schema_hash, fingerprint

    def _truncate_text(val: Any, max_chars: int) -> Any:
        if not isinstance(val, str):
            return val
        if len(val) <= max_chars:
            return val
        return val[:max_chars] + "\n...[truncated]..."

    def _sha256_text(val: Any) -> str | None:
        try:
            import hashlib

            if not isinstance(val, str) or not val:
                return None
            return hashlib.sha256(val.encode("utf-8")).hexdigest()
        except Exception:
            return None

    def _prune_datasets(datasets: dict[str, Any]) -> dict[str, Any]:
        if len(datasets) <= DATASET_REGISTRY_MAX:
            return datasets
        items: list[tuple[float, str]] = []
        for did, entry in datasets.items():
            ts = 0.0
            if isinstance(entry, dict):
                try:
                    ts = float(entry.get("created_ts") or 0.0)
                except Exception:
                    ts = 0.0
            items.append((ts, did))
        items.sort(reverse=True)
        keep = {did for _ts, did in items[:DATASET_REGISTRY_MAX]}
        return {did: datasets[did] for did in keep if did in datasets}

    def _ensure_dataset_registry(
        state: SupervisorDSState,
    ) -> tuple[dict[str, Any], str | None]:
        datasets = state.get("datasets")
        datasets = datasets if isinstance(datasets, dict) else {}
        active_id = state.get("active_dataset_id")
        active_id = active_id if isinstance(active_id, str) else None

        # Bootstrap from known state slots if registry is empty.
        if not datasets:
            import time
            import uuid
            from datetime import datetime, timezone

            def _add(stage: str, data_key: str):
                nonlocal datasets
                data = state.get(data_key)
                if data is None:
                    return
                did = f"{stage}_{uuid.uuid4().hex[:8]}"
                shape, cols, schema, schema_hash, fingerprint = _dataset_meta(data)
                ts = time.time()
                provenance = {"source_type": "state_slot", "source": data_key}
                # If the app provided explicit provenance for an injected data_raw (upload/sample),
                # prefer it so the pipeline can be reproduced.
                if stage == "raw" and data_key == "data_raw":
                    try:
                        artifacts = state.get("artifacts") or {}
                        input_ds = (
                            artifacts.get("input_dataset")
                            if isinstance(artifacts, dict)
                            else None
                        )
                        if isinstance(input_ds, dict) and input_ds.get("source"):
                            provenance = {**provenance, **input_ds}
                    except Exception:
                        pass

                datasets[did] = {
                    "id": did,
                    "label": data_key,
                    "stage": stage,
                    "data": data,
                    "shape": shape,
                    "columns": cols,
                    "schema": schema,
                    "schema_hash": schema_hash,
                    "fingerprint": fingerprint,
                    "created_ts": ts,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "created_by": "bootstrap",
                    "provenance": provenance,
                    "parent_id": None,
                    "parent_ids": [],
                }

            _add("raw", "data_raw")
            _add("sql", "data_sql")
            _add("wrangled", "data_wrangled")
            _add("cleaned", "data_cleaned")
            _add("feature", "feature_data")

        # Pick a reasonable default active dataset if missing.
        if active_id is None and isinstance(datasets, dict) and datasets:
            # Prefer the dataset whose stage matches active_data_key, else newest.
            active_key = state.get("active_data_key")
            stage_for_key = {
                "data_raw": "raw",
                "data_sql": "sql",
                "data_wrangled": "wrangled",
                "data_cleaned": "cleaned",
                "feature_data": "feature",
            }.get(active_key)
            if stage_for_key:
                matching = [
                    did
                    for did, e in datasets.items()
                    if isinstance(e, dict) and e.get("stage") == stage_for_key
                ]
                if matching:
                    active_id = matching[-1]
            if active_id is None:
                newest = sorted(
                    datasets.items(),
                    key=lambda kv: float(kv[1].get("created_ts") or 0.0)
                    if isinstance(kv[1], dict)
                    else 0.0,
                )
                active_id = newest[-1][0] if newest else None

        if active_id is not None and active_id not in datasets:
            active_id = None

        return _prune_datasets(datasets), active_id

    def _register_dataset(
        state: SupervisorDSState,
        *,
        data: Any,
        stage: str,
        label: str,
        created_by: str,
        provenance: dict[str, Any],
        parent_id: str | None = None,
        parent_ids: Sequence[str] | None = None,
        make_active: bool = True,
    ) -> tuple[dict[str, Any], str | None, str]:
        import time
        import uuid
        from datetime import datetime, timezone

        datasets, current_active = _ensure_dataset_registry(state)
        did = f"{stage}_{uuid.uuid4().hex[:8]}"
        shape, cols, schema, schema_hash, fingerprint = _dataset_meta(data)
        ts = time.time()
        normalized_parents: list[str] = []
        if isinstance(parent_ids, (list, tuple)):
            normalized_parents = [
                str(p) for p in parent_ids if isinstance(p, str) and p
            ]
        if parent_id and parent_id not in normalized_parents:
            normalized_parents = [parent_id, *normalized_parents]
        normalized_parents = [p for p in normalized_parents if p]
        parent_id = normalized_parents[0] if normalized_parents else parent_id
        datasets = {
            **datasets,
            did: {
                "id": did,
                "label": label or did,
                "stage": stage,
                "data": data,
                "shape": shape,
                "columns": cols,
                "schema": schema,
                "schema_hash": schema_hash,
                "fingerprint": fingerprint,
                "created_ts": ts,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "created_by": created_by,
                "provenance": provenance or {},
                "parent_id": parent_id,
                "parent_ids": normalized_parents,
            },
        }
        datasets = _prune_datasets(datasets)
        active_id = did if make_active else current_active
        return datasets, active_id, did

    def _get_active_data(state: SupervisorDSState, fallback_keys: Sequence[str]):
        datasets = state.get("datasets")
        active_id = state.get("active_dataset_id")
        if isinstance(datasets, dict) and isinstance(active_id, str):
            entry = datasets.get(active_id)
            if isinstance(entry, dict) and entry.get("data") is not None:
                return entry.get("data")
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
        cfg = (state.get("artifacts") or {}).get("config") or {}
        debug = bool(cfg.get("debug")) if isinstance(cfg, dict) else False
        if debug:
            print(f"  loader last_human={last_human!r}")

        # DataLoaderToolsAgent is tool-driven; the latest user request is already in messages.
        data_loader_agent.invoke_messages(messages=before_msgs)
        response = data_loader_agent.response or {}
        merged = _merge_messages(before_msgs, response)

        loader_artifacts = response.get("data_loader_artifacts")
        if debug:
            try:
                print(f"  loader response_keys={sorted(list(response.keys()))}")
                if isinstance(loader_artifacts, dict):
                    print(
                        f"  loader artifacts_keys={list(loader_artifacts.keys())[:25]}"
                    )
                else:
                    print(f"  loader artifacts_type={type(loader_artifacts)}")
            except Exception:
                pass

        previous_data_raw = state.get("data_raw")
        data_raw = previous_data_raw
        active_data_key = state.get("active_data_key")

        dir_listing = None
        loaded_dataset = None
        loaded_dataset_label = None
        multiple_loaded_files = None
        multiple_loaded_datasets: list[tuple[str, Any]] | None = None
        fallback_loaded_dataset = False
        multi_file_load = False

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
        if debug:
            try:
                print(f"  loader artifacts_map_keys={list(artifacts_map.keys())[:25]}")
            except Exception:
                pass

        # Detect directory listings (do NOT overwrite data_raw)
        for key, val in artifacts_map.items():
            if str(key).startswith("list_directory") or str(key).startswith(
                "search_files_by_pattern"
            ):
                dir_listing = val
                break

        load_file_ok_items: list[tuple[str, Any]] = []
        # Detect dataset loads
        for key, val in artifacts_map.items():
            tool_name = str(key)
            # load_file artifact: {"status":"ok","data":{...},"error":None}
            if tool_name.startswith("load_file") and isinstance(val, dict):
                if val.get("status") == "ok" and val.get("data") is not None:
                    load_file_ok_items.append((tool_name, val.get("data")))
                continue

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
                if len(ok_items) == 1 and loaded_dataset is None:
                    loaded_dataset_label, loaded_dataset = ok_items[0]
                    continue
                if len(ok_items) > 1:
                    # Multiple datasets loaded; don't guess which one becomes active.
                    multiple_loaded_files = [fname for fname, _ in ok_items]
                    multiple_loaded_datasets = ok_items
                    loaded_dataset = None
                    loaded_dataset_label = None
                    break

        if debug:
            try:
                print(f"  loader load_file_ok_items={len(load_file_ok_items)}")
                for name, data in load_file_ok_items[:3]:
                    print(
                        f"    - ok {name}: data_type={type(data)} shape={_shape(data)}"
                    )
            except Exception:
                pass

        # Fallback: if tool artifacts didn't yield usable data, load file paths directly from the user text.
        if (
            loaded_dataset is None
            and not multiple_loaded_datasets
            and not load_file_ok_items
            and isinstance(last_human, str)
            and last_human.strip()
        ):
            try:
                import re
                import pandas as pd

                from ai_data_science_team.tools.data_loader import (
                    auto_load_file,
                    DEFAULT_MAX_ROWS,
                )

                last_human_lower = last_human.lower()
                if any(
                    w in last_human_lower for w in ("load", "read", "import", "open")
                ):
                    requested = re.findall(
                        r"(?:`|\"|')?([^\s'\"`]+\.(?:csv|tsv|parquet|xlsx?|jsonl|ndjson|json)(?:\.gz)?)",
                        last_human,
                        flags=re.IGNORECASE,
                    )
                    requested = [r.strip() for r in requested if str(r).strip()]
                    seen_req: set[str] = set()
                    requested_unique: list[str] = []
                    for r in requested:
                        if r in seen_req:
                            continue
                        seen_req.add(r)
                        requested_unique.append(r)

                    ok_items: list[tuple[str, Any]] = []
                    errs: list[str] = []
                    for fp in requested_unique:
                        df_or_error = auto_load_file(fp, max_rows=DEFAULT_MAX_ROWS)
                        if isinstance(df_or_error, pd.DataFrame):
                            ok_items.append((fp, df_or_error.to_dict()))
                        else:
                            errs.append(f"{fp}: {df_or_error}")

                    if ok_items:
                        multi_file_load = len(ok_items) > 1
                        multiple_loaded_files = [fp for fp, _ in ok_items]
                        multiple_loaded_datasets = ok_items
                        loaded_dataset_label, loaded_dataset = ok_items[-1]
                        fallback_loaded_dataset = True
                        dir_listing = None
                        if debug:
                            print(
                                f"  loader deterministic_fallback_loaded={len(ok_items)} last={loaded_dataset_label!r}"
                            )
                    if errs and debug:
                        print(f"  loader deterministic_fallback_errors={errs[:3]}")

                    if errs:
                        marker = {
                            "status": "error",
                            "data": None,
                            "error": "; ".join(errs[:3]),
                        }
                        if isinstance(loader_artifacts, dict):
                            loader_artifacts = {
                                **loader_artifacts,
                                "load_file_deterministic_fallback": marker,
                            }
                        elif loader_artifacts is None:
                            loader_artifacts = {
                                "load_file_deterministic_fallback": marker
                            }
            except Exception:
                pass

        # If multiple load_file calls succeeded, keep them all and default the active dataset to the last one.
        if (
            loaded_dataset is None
            and not multiple_loaded_datasets
            and len(load_file_ok_items) > 1
        ):
            import re

            labels: list[str] = []
            try:
                requested = re.findall(
                    r"(?:`|\"|')?([^\s'\"`]+\.(?:csv|tsv|parquet|xlsx?|jsonl|ndjson|json)(?:\.gz)?)",
                    last_human or "",
                    flags=re.IGNORECASE,
                )
                requested = [r.strip() for r in requested if str(r).strip()]
                # De-dupe while preserving order
                seen_req: set[str] = set()
                requested_unique: list[str] = []
                for r in requested:
                    if r in seen_req:
                        continue
                    seen_req.add(r)
                    requested_unique.append(r)
                labels = requested_unique[: len(load_file_ok_items)]
            except Exception:
                labels = []

            if len(labels) != len(load_file_ok_items):
                labels = [name for name, _ in load_file_ok_items]

            multi_file_load = True
            multiple_loaded_files = labels
            multiple_loaded_datasets = [
                (lbl, data) for lbl, (_name, data) in zip(labels, load_file_ok_items)
            ]
            loaded_dataset_label, loaded_dataset = multiple_loaded_datasets[-1]
        elif (
            loaded_dataset is None
            and not multiple_loaded_datasets
            and len(load_file_ok_items) == 1
        ):
            loaded_dataset_label, loaded_dataset = load_file_ok_items[0]

        # If the tool returned only a directory listing but the user requested a specific file to load,
        # attempt to load it deterministically (avoids "listing loop" regressions across turns).
        if loaded_dataset is None and dir_listing is not None:
            try:
                import re
                import os
                from pathlib import Path
                import pandas as pd

                from ai_data_science_team.tools.data_loader import (
                    auto_load_file,
                    DEFAULT_MAX_ROWS,
                )

                last_human_text = _get_last_human(before_msgs) or ""
                last_human_lower = last_human_text.lower()

                if any(
                    w in last_human_lower for w in ("load", "read", "import", "open")
                ):
                    m = re.search(
                        r"(?:`|\"|')?([^\s'\"`]+\.(?:csv|tsv|parquet|xlsx?|jsonl|ndjson|json)(?:\.gz)?)",
                        last_human_text,
                        flags=re.IGNORECASE,
                    )
                    requested = (m.group(1) if m else "").strip()
                    if requested:
                        p = Path(requested).expanduser()
                        if not p.is_absolute():
                            p = (Path(os.getcwd()) / p).resolve()
                        else:
                            p = p.resolve()

                        def _load_path(fp: str) -> Optional[dict]:
                            df_or_error = auto_load_file(fp, max_rows=DEFAULT_MAX_ROWS)
                            if isinstance(df_or_error, pd.DataFrame):
                                return df_or_error.to_dict()
                            return None

                        loaded = _load_path(str(p)) if p.is_file() else None

                        # If the path isn't directly valid, try to match by basename from listing outputs.
                        if loaded is None:
                            basename = Path(requested).name
                            candidate_paths: list[str] = []
                            if isinstance(dir_listing, list):
                                for item in dir_listing:
                                    if isinstance(item, dict):
                                        fp = (
                                            item.get("file_path")
                                            or item.get("absolute_path")
                                            or item.get("path")
                                            or item.get("filepath")
                                        )
                                        if isinstance(fp, str):
                                            candidate_paths.append(fp)
                                    elif isinstance(item, str):
                                        candidate_paths.append(item)
                            elif isinstance(dir_listing, dict):
                                for item in dir_listing.values():
                                    if isinstance(item, dict):
                                        fp = (
                                            item.get("file_path")
                                            or item.get("absolute_path")
                                            or item.get("path")
                                            or item.get("filepath")
                                        )
                                        if isinstance(fp, str):
                                            candidate_paths.append(fp)
                                    elif isinstance(item, str):
                                        candidate_paths.append(item)
                            for fp in candidate_paths:
                                try:
                                    resolved = Path(fp).expanduser().resolve()
                                except Exception:
                                    continue
                                if resolved.is_file() and resolved.name == basename:
                                    loaded = _load_path(str(resolved))
                                    if loaded is not None:
                                        loaded_dataset_label = str(resolved)
                                        break

                        if loaded is not None:
                            loaded_dataset = loaded
                            loaded_dataset_label = loaded_dataset_label or str(p)
                            dir_listing = None
                            fallback_loaded_dataset = True
            except Exception:
                pass

        if loaded_dataset is not None:
            data_raw = loaded_dataset
            active_data_key = "data_raw"
            # Prefer dataset summary over any incidental listings
            dir_listing = None
            if fallback_loaded_dataset:
                # The loader agent likely produced a listing-oriented AI message; suppress it.
                merged["messages"] = []
                # Store a lightweight marker so the supervisor can mark the load step as completed.
                marker = {
                    "status": "ok",
                    "data": {"file_path": loaded_dataset_label},
                    "error": None,
                }
                if isinstance(loader_artifacts, dict):
                    loader_artifacts = {
                        **loader_artifacts,
                        "load_file_fallback": marker,
                    }
                else:
                    loader_artifacts = {"load_file_fallback": marker}

        print(
            f"  loader data_raw shape={_shape(data_raw)} active_data_key={active_data_key}"
        )

        datasets, active_dataset_id = _ensure_dataset_registry(state)
        # Register newly loaded datasets in the dataset registry.
        if multi_file_load and multiple_loaded_datasets:
            try:
                import os
                from ai_data_science_team.tools.data_loader import (
                    resolve_existing_file_path,
                )

                state_for_register = {
                    **state,
                    "datasets": datasets,
                    "active_dataset_id": active_dataset_id,
                }
                to_register = list(multiple_loaded_datasets)[-DATASET_REGISTRY_MAX:]
                for idx, (fname, data) in enumerate(to_register):
                    source = str(fname)
                    try:
                        resolved_path, _matches = resolve_existing_file_path(source)
                        if resolved_path is not None:
                            source = str(resolved_path)
                    except Exception:
                        source = str(fname)

                    label = os.path.basename(source) or str(fname)
                    provenance = {
                        "source_type": "file",
                        "source": source or str(fname),
                        "original_name": os.path.basename(str(fname)) or str(fname),
                        "user_request": last_human,
                        "multi_load": True,
                    }
                    make_active = idx == (len(to_register) - 1)
                    datasets, active_dataset_id, _did = _register_dataset(
                        state_for_register,
                        data=data,
                        stage="raw",
                        label=str(label),
                        created_by="Data_Loader_Tools_Agent",
                        provenance=provenance,
                        parent_id=None,
                        make_active=make_active,
                    )
                    state_for_register = {
                        **state_for_register,
                        "datasets": datasets,
                        "active_dataset_id": active_dataset_id,
                    }
            except Exception:
                # Never fail the load step due to registry bookkeeping.
                pass
        elif loaded_dataset is not None:
            try:
                import os

                # Best-effort: capture the file path from the user request for reproducibility.
                source = loaded_dataset_label
                try:
                    import re
                    from ai_data_science_team.tools.data_loader import (
                        resolve_existing_file_path,
                    )

                    if not (
                        isinstance(source, str)
                        and ("." in source and os.path.sep in source)
                    ):
                        m = re.search(
                            r"(?:`|\"|')?([^\s'\"`]+\.(?:csv|tsv|parquet|xlsx?|jsonl|ndjson|json)(?:\.gz)?)",
                            last_human or "",
                            flags=re.IGNORECASE,
                        )
                        requested = (m.group(1) if m else "").strip()
                        if requested:
                            resolved_path, _matches = resolve_existing_file_path(
                                requested
                            )
                            if resolved_path is not None:
                                source = str(resolved_path)
                            else:
                                source = requested
                    # Also normalize/absolutize an existing-looking path label.
                    if isinstance(source, str) and source.strip():
                        resolved_path, _matches = resolve_existing_file_path(source)
                        if resolved_path is not None:
                            source = str(resolved_path)
                except Exception:
                    pass

                label = source or loaded_dataset_label or "data_raw"
                if isinstance(label, str):
                    label = os.path.basename(label) or label
                provenance = {
                    "source_type": "file",
                    "source": source or loaded_dataset_label,
                    "original_name": os.path.basename(
                        str(source or loaded_dataset_label or "")
                    )
                    or None,
                    "user_request": last_human,
                    "fallback_loader": bool(fallback_loaded_dataset),
                }
                datasets, active_dataset_id, _did = _register_dataset(
                    {
                        **state,
                        "datasets": datasets,
                        "active_dataset_id": active_dataset_id,
                    },
                    data=data_raw,
                    stage="raw",
                    label=str(label),
                    created_by="Data_Loader_Tools_Agent",
                    provenance=provenance,
                    parent_id=None,
                    make_active=True,
                )
            except Exception:
                # Never fail the load step due to registry bookkeeping.
                pass
        elif multiple_loaded_datasets:
            # Keep the already-loaded datasets available for explicit selection, but do not auto-switch.
            try:
                state_for_register = {
                    **state,
                    "datasets": datasets,
                    "active_dataset_id": active_dataset_id,
                }
                # Register only the most recent N to avoid unbounded growth.
                for fname, data in list(multiple_loaded_datasets)[
                    -DATASET_REGISTRY_MAX:
                ]:
                    datasets, active_dataset_id, _did = _register_dataset(
                        state_for_register,
                        data=data,
                        stage="raw",
                        label=str(fname),
                        created_by="Data_Loader_Tools_Agent",
                        provenance={
                            "source_type": "directory_load",
                            "source": fname,
                            "user_request": last_human,
                        },
                        parent_id=None,
                        make_active=False,
                    )
                    state_for_register = {
                        **state_for_register,
                        "datasets": datasets,
                        "active_dataset_id": active_dataset_id,
                    }
            except Exception:
                pass

        # Add a lightweight AI summary message so supervisor can progress
        summary_msg = None
        if multi_file_load and multiple_loaded_datasets:
            try:
                import os
                import pandas as pd

                lines = []
                for fname, data in multiple_loaded_datasets:
                    label = os.path.basename(str(fname)) or str(fname)
                    shape_txt = ""
                    try:
                        df = pd.DataFrame(data)
                        shape_txt = f" ({df.shape[0]} rows × {df.shape[1]} cols)"
                    except Exception:
                        pass
                    lines.append(f"- {label}{shape_txt}")

                active_label = os.path.basename(str(loaded_dataset_label)) or str(
                    loaded_dataset_label or ""
                )
                preview_txt = ""
                try:
                    df_active = (
                        pd.DataFrame(data_raw) if isinstance(data_raw, dict) else None
                    )
                    if df_active is not None:
                        preview_df = df_active.head(5)
                        max_cols = 10
                        if preview_df.shape[1] > max_cols:
                            preview_df = preview_df.iloc[:, :max_cols]
                        preview_txt = (
                            "\n\nPreview (first 5 rows):\n\n"
                            + preview_df.to_markdown(index=False)
                        )
                except Exception:
                    pass

                summary_msg = AIMessage(
                    content=(
                        f"Loaded {len(multiple_loaded_datasets)} datasets:\n\n"
                        + "\n".join(lines)
                        + (
                            f"\n\nActive dataset: {active_label}."
                            if active_label
                            else ""
                        )
                        + preview_txt
                        + "\n\nUse the sidebar dataset selector to switch the active dataset, or use Pipeline Studio to merge them."
                    ),
                    name="data_loader_agent",
                )
            except Exception:
                summary_msg = AIMessage(
                    content=(
                        f"Loaded {len(multiple_loaded_datasets)} datasets. "
                        "Use the sidebar dataset selector to switch the active dataset, or use Pipeline Studio to merge them."
                    ),
                    name="data_loader_agent",
                )
        elif multiple_loaded_files:
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
                        if isinstance(item, dict):
                            if "filename" in item:
                                names.append(item.get("filename"))
                                rows.append(
                                    {
                                        "filename": item.get("filename"),
                                        "type": item.get("type"),
                                        "path": item.get("path")
                                        or item.get("filepath"),
                                    }
                                )
                                continue
                            if "file_path" in item:
                                fp = item.get("file_path")
                                import os

                                fn = (
                                    os.path.basename(fp)
                                    if isinstance(fp, str)
                                    else str(fp)
                                )
                                names.append(fn)
                                rows.append(
                                    {"filename": fn, "type": "file", "path": fp}
                                )
                                continue
                            if "absolute_path" in item or "name" in item:
                                ap = item.get("absolute_path")
                                import os

                                fn = item.get("name") or (
                                    os.path.basename(ap)
                                    if isinstance(ap, str)
                                    else str(ap)
                                )
                                names.append(fn)
                                rows.append(
                                    {
                                        "filename": fn,
                                        "type": item.get("type"),
                                        "path": ap,
                                    }
                                )
                                continue

                        names.append(str(item))
                        rows.append({"filename": str(item)})
                elif isinstance(dir_listing, dict):
                    # maybe mapping index->filename
                    for v in dir_listing.values():
                        if isinstance(v, dict):
                            if "filename" in v:
                                names.append(str(v.get("filename")))
                                rows.append(
                                    {
                                        "filename": v.get("filename"),
                                        "type": v.get("type"),
                                        "path": v.get("path") or v.get("filepath"),
                                    }
                                )
                            elif "file_path" in v:
                                fp = v.get("file_path")
                                import os

                                fn = (
                                    os.path.basename(fp)
                                    if isinstance(fp, str)
                                    else str(fp)
                                )
                                names.append(fn)
                                rows.append(
                                    {"filename": fn, "type": "file", "path": fp}
                                )
                            elif "absolute_path" in v or "name" in v:
                                ap = v.get("absolute_path")
                                import os

                                fn = v.get("name") or (
                                    os.path.basename(ap)
                                    if isinstance(ap, str)
                                    else str(ap)
                                )
                                names.append(fn)
                                rows.append(
                                    {"filename": fn, "type": v.get("type"), "path": ap}
                                )
                            else:
                                names.append(str(v))
                                rows.append({"filename": str(v)})
                        else:
                            names.append(str(v))
                            rows.append({"filename": str(v)})

                last_human = _get_last_human(before_msgs).lower()
                wants_csv_only = "csv" in last_human and (
                    "list" in last_human or "files" in last_human
                )
                if wants_csv_only and rows:
                    rows = [
                        r
                        for r in rows
                        if str(r.get("filename", "")).lower().endswith(".csv")
                    ]
                    names = [r.get("filename") for r in rows if r.get("filename")]
                    if not rows:
                        summary_msg = AIMessage(
                            content="No CSV files found in that directory.",
                            name="data_loader_agent",
                        )
                        dir_listing = None

                if summary_msg is None:
                    msg_text = (
                        "Found files: " + ", ".join(names)
                        if names
                        else "Found directory contents."
                    )
                    table_text = ""
                    if rows:
                        import pandas as pd

                        df_listing = pd.DataFrame(rows)
                        table_cols = [
                            c
                            for c in ["filename", "type", "path"]
                            if c in df_listing.columns
                        ]
                        table_text = df_listing[table_cols].to_markdown(index=False)
                    # If the user asked for a table or better formatting, try a tiny LLM summary
                    llm_text = (
                        _format_listing_with_llm(rows, last_human) if rows else None
                    )
                    if llm_text:
                        summary_msg = AIMessage(
                            content=llm_text, name="data_loader_agent"
                        )
                    elif table_text:
                        summary_msg = AIMessage(
                            content=f"{msg_text}\n\n{table_text}",
                            name="data_loader_agent",
                        )
                    else:
                        summary_msg = AIMessage(
                            content=msg_text, name="data_loader_agent"
                        )
            except Exception:
                summary_msg = AIMessage(
                    content="Listed directory contents.", name="data_loader_agent"
                )
        elif loaded_dataset is not None and isinstance(data_raw, dict):
            try:
                import pandas as pd

                df = pd.DataFrame(data_raw)
                last_human_lower = (_get_last_human(before_msgs) or "").lower()
                wants_preview_rows = any(
                    k in last_human_lower
                    for k in (
                        "head",
                        "preview",
                        "first 5",
                        "first five",
                        "first 5 rows",
                        "first five rows",
                        "show the first",
                        "show first",
                        "show rows",
                    )
                )

                max_cols = 10
                preview_df = df.head(5)
                col_note = ""
                if preview_df.shape[1] > max_cols:
                    preview_df = preview_df.iloc[:, :max_cols]
                    col_note = f" (showing first {max_cols} of {df.shape[1]} columns)"
                table_md = preview_df.to_markdown(index=False)

                if wants_preview_rows:
                    summary_msg = AIMessage(
                        content=f"Loaded dataset with shape {df.shape}.{col_note}\n\n{table_md}",
                        name="data_loader_agent",
                    )
                else:
                    llm_text = _format_result_with_llm(
                        "data_loader_agent",
                        data_raw,
                        _get_last_human(before_msgs),
                    )
                    if llm_text:
                        summary_msg = AIMessage(
                            content=llm_text, name="data_loader_agent"
                        )
                    else:
                        summary_msg = AIMessage(
                            content=f"Loaded dataset with shape {df.shape}.{col_note}\n\n{table_md}",
                            name="data_loader_agent",
                        )
            except Exception:
                summary_msg = AIMessage(
                    content="Loaded dataset successfully. What would you like to do next?",
                    name="data_loader_agent",
                )
        elif loader_artifacts is not None:
            # Include tool errors (if any) to help the user correct paths quickly.
            errors: list[str] = []
            try:
                if isinstance(loader_artifacts, dict):
                    # Single load_file artifact shape: {"status","data","error"}
                    if {"status", "data"}.issubset(set(loader_artifacts.keys())):
                        err = loader_artifacts.get("error")
                        if isinstance(err, str) and err.strip():
                            errors.append(err.strip())
                    else:
                        for k, v in loader_artifacts.items():
                            if not str(k).startswith("load_file"):
                                continue
                            if isinstance(v, dict):
                                if v.get("status") != "ok":
                                    err = v.get("error")
                                    if isinstance(err, str) and err.strip():
                                        errors.append(err.strip())
            except Exception:
                errors = []

            errors_txt = ""
            if errors:
                uniq: list[str] = []
                seen: set[str] = set()
                for e in errors:
                    if e in seen:
                        continue
                    seen.add(e)
                    uniq.append(e)
                shown = uniq[:3]
                errors_txt = "\n\nErrors:\n" + "\n".join([f"- {e}" for e in shown])
                if len(uniq) > 3:
                    errors_txt += f"\n- (+{len(uniq) - 3} more)"

            summary_msg = AIMessage(
                content=(
                    "I couldn't load a tabular dataset from that request. "
                    f"{errors_txt}\n\n"
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
            "datasets": datasets,
            "active_dataset_id": active_dataset_id,
            "artifacts": {
                **state.get("artifacts", {}),
                "data_loader": loader_artifacts,
            },
            "last_worker": "Data_Loader_Tools_Agent",
            **downstream_resets,
        }

    def node_merge(state: SupervisorDSState):
        print("---DATA MERGE---")
        before_msgs = list(state.get("messages", []) or [])
        last_human = _get_last_human(before_msgs)
        datasets, active_dataset_id = _ensure_dataset_registry(state)
        state_with_datasets = {
            **(state or {}),
            "datasets": datasets,
            "active_dataset_id": active_dataset_id,
        }

        cfg = (state.get("artifacts") or {}).get("config") or {}
        merge_cfg = cfg.get("merge") if isinstance(cfg, dict) else None
        merge_cfg = merge_cfg if isinstance(merge_cfg, dict) else {}

        def _parse_list(val: Any) -> list[str]:
            if isinstance(val, list):
                return [str(x).strip() for x in val if str(x).strip()]
            if isinstance(val, tuple):
                return [str(x).strip() for x in val if str(x).strip()]
            if isinstance(val, str):
                parts = [p.strip() for p in val.split(",")]
                return [p for p in parts if p]
            return []

        def _dedupe_keep_order(items: list[str]) -> list[str]:
            out: list[str] = []
            seen: set[str] = set()
            for x in items:
                if x in seen:
                    continue
                seen.add(x)
                out.append(x)
            return out

        def _resolve_dataset_ids_from_text(text: str) -> list[str]:
            if not text or not isinstance(datasets, dict):
                return []
            t = text.lower()
            matches: list[str] = []
            for did, entry in datasets.items():
                if not isinstance(entry, dict):
                    continue
                label = str(entry.get("label") or "").lower()
                prov = (
                    entry.get("provenance")
                    if isinstance(entry.get("provenance"), dict)
                    else {}
                )
                original = str(prov.get("original_name") or "").lower()
                source = str(prov.get("source") or "").lower()
                if did.lower() in t:
                    matches.append(did)
                    continue
                if label and label in t:
                    matches.append(did)
                    continue
                if original and original in t:
                    matches.append(did)
                    continue
                if source and source in t:
                    matches.append(did)
                    continue
            return matches

        selected_ids = _parse_list(merge_cfg.get("dataset_ids"))
        selected_ids = [d for d in selected_ids if d in datasets]

        # Fallback: attempt to infer datasets from the user request
        if len(selected_ids) < 2:
            inferred = _resolve_dataset_ids_from_text(last_human)
            selected_ids = [*selected_ids, *inferred]

        # Fallback: if only one is identified, use the active dataset plus the newest other dataset
        if (
            len(selected_ids) < 2
            and isinstance(active_dataset_id, str)
            and active_dataset_id in datasets
        ):
            selected_ids = [active_dataset_id]
            ordered = sorted(
                datasets.items(),
                key=lambda kv: float(kv[1].get("created_ts") or 0.0)
                if isinstance(kv[1], dict)
                else 0.0,
                reverse=True,
            )
            for did, _e in ordered:
                if did != active_dataset_id:
                    selected_ids.append(did)
                    break

        selected_ids = _dedupe_keep_order([d for d in selected_ids if d in datasets])

        if len(selected_ids) < 2:
            available = []
            try:
                ordered = sorted(
                    datasets.items(),
                    key=lambda kv: float(kv[1].get("created_ts") or 0.0)
                    if isinstance(kv[1], dict)
                    else 0.0,
                    reverse=True,
                )
                for did, e in ordered[:10]:
                    if not isinstance(e, dict):
                        continue
                    available.append(f"- `{did}` ({e.get('stage')}:{e.get('label')})")
            except Exception:
                pass
            msg = (
                "To merge datasets, mention 2+ dataset IDs in your request (or use Pipeline Studio to create a merge node).\n\n"
                + ("Available datasets:\n" + "\n".join(available) if available else "")
            ).strip()
            return {
                "messages": [AIMessage(content=msg, name="data_merge_agent")],
                "last_worker": "Data_Merge_Agent",
            }

        dfs = []
        for did in selected_ids:
            entry = datasets.get(did)
            df = _ensure_df(entry.get("data") if isinstance(entry, dict) else None)
            if _is_empty_df(df):
                return {
                    "messages": [
                        AIMessage(
                            content=f"Dataset `{did}` is empty/unavailable; load it again before merging.",
                            name="data_merge_agent",
                        )
                    ],
                    "last_worker": "Data_Merge_Agent",
                }
            dfs.append(df)

        op = str(merge_cfg.get("operation") or "join").strip().lower()
        if any(w in last_human.lower() for w in ("concat", "append", "union")):
            op = "concat"
        if op not in ("join", "concat"):
            op = "join"

        import pandas as pd

        merge_code_lines: list[str] = ["# Auto-generated merge step"]
        merged_df = None
        merge_meta: dict[str, Any] = {"operation": op, "dataset_ids": selected_ids}

        if op == "concat":
            axis = merge_cfg.get("axis", 0)
            try:
                axis = int(axis)
            except Exception:
                axis = 0
            ignore_index = bool(merge_cfg.get("ignore_index", True))
            merged_df = pd.concat(
                dfs, axis=axis, ignore_index=(ignore_index if axis == 0 else False)
            )
            merge_meta.update({"axis": axis, "ignore_index": ignore_index})
            merge_code_lines.append(
                f"df = pd.concat([{', '.join([f'df_{i}' for i in range(len(dfs))])}], axis={axis}, ignore_index={ignore_index if axis == 0 else False})"
            )
        else:
            how = str(merge_cfg.get("how") or "inner").strip().lower()
            if how not in ("inner", "left", "right", "outer"):
                how = "inner"
            on_cols = _parse_list(merge_cfg.get("on"))
            left_on = _parse_list(merge_cfg.get("left_on"))
            right_on = _parse_list(merge_cfg.get("right_on"))
            suffixes_raw = str(merge_cfg.get("suffixes") or "_x,_y")
            suffixes_parts = [p.strip() for p in suffixes_raw.split(",") if p.strip()]
            suffixes = (
                (suffixes_parts[0], suffixes_parts[1])
                if len(suffixes_parts) >= 2
                else ("_x", "_y")
            )

            # Infer join keys when not provided.
            if not on_cols and not (left_on and right_on):
                common = set(dfs[0].columns)
                for df in dfs[1:]:
                    common = common.intersection(set(df.columns))
                if common:
                    preferred = sorted(
                        list(common),
                        key=lambda c: (
                            0 if "id" in str(c).lower() else 1,
                            str(c).lower(),
                        ),
                    )
                    on_cols = [preferred[0]]

            if not on_cols and not (left_on and right_on):
                return {
                    "messages": [
                        AIMessage(
                            content=(
                                "I couldn't infer join keys for the selected datasets. "
                                "Specify join keys in your request (e.g., `join on customerID`), "
                                "or configure them in Pipeline Studio."
                            ),
                            name="data_merge_agent",
                        )
                    ],
                    "last_worker": "Data_Merge_Agent",
                }

            merged_df = dfs[0]
            merge_code_lines.append("df = df_0")
            for i in range(1, len(dfs)):
                if left_on and right_on:
                    merged_df = merged_df.merge(
                        dfs[i],
                        how=how,
                        left_on=left_on,
                        right_on=right_on,
                        suffixes=suffixes,
                    )
                    merge_code_lines.append(
                        f"df = df.merge(df_{i}, how={how!r}, left_on={left_on!r}, right_on={right_on!r}, suffixes={suffixes!r})"
                    )
                else:
                    merged_df = merged_df.merge(
                        dfs[i],
                        how=how,
                        on=on_cols,
                        suffixes=suffixes,
                    )
                    merge_code_lines.append(
                        f"df = df.merge(df_{i}, how={how!r}, on={on_cols!r}, suffixes={suffixes!r})"
                    )

            merge_meta.update(
                {
                    "how": how,
                    "on": on_cols,
                    "left_on": left_on,
                    "right_on": right_on,
                    "suffixes": suffixes,
                }
            )

        merge_code = "\n".join(merge_code_lines).strip() + "\n"
        merge_code_hash = _sha256_text(merge_code)

        merged_data = merged_df
        try:
            import pandas as pd

            if isinstance(merged_df, pd.DataFrame):
                merged_data = merged_df.to_dict()
        except Exception:
            merged_data = merged_df

        datasets, active_dataset_id, merged_id = _register_dataset(
            state_with_datasets,
            data=merged_data,
            stage="wrangled",
            label="data_merged",
            created_by="Data_Merge_Agent",
            provenance={
                "source_type": "agent",
                "user_request": last_human,
                "transform": {
                    "kind": "python_merge",
                    "merge": merge_meta,
                    "merge_code": _truncate_text(merge_code, 12000),
                    "code_sha256": merge_code_hash,
                },
            },
            parent_ids=selected_ids,
            make_active=True,
        )

        msg_lines = [
            f"Merged {len(selected_ids)} datasets ({op}).",
            f"Result shape: {getattr(merged_df, 'shape', None)}.",
            f"Active dataset id: `{merged_id}`.",
        ]
        merged = {
            "messages": [
                AIMessage(
                    content=" ".join([m for m in msg_lines if m]),
                    name="data_merge_agent",
                )
            ]
        }
        merged["messages"] = _tag_messages(merged.get("messages"), "data_merge_agent")
        downstream_resets = {
            "data_cleaned": None,
            "eda_artifacts": None,
            "viz_graph": None,
            "feature_data": None,
            "model_info": None,
            "mlflow_artifacts": None,
        }
        return {
            **merged,
            "data_wrangled": merged_data,
            "active_data_key": "data_wrangled",
            "datasets": datasets,
            "active_dataset_id": active_dataset_id,
            "artifacts": {
                **state.get("artifacts", {}),
                "merge": {
                    "dataset_ids": selected_ids,
                    "operation": op,
                    "active_dataset_id": merged_id,
                    "merge_config": merge_cfg,
                },
            },
            "last_worker": "Data_Merge_Agent",
            **downstream_resets,
        }

    def node_wrangling(state: SupervisorDSState):
        print("---DATA WRANGLING---")
        before_msgs = list(state.get("messages", []) or [])
        last_human = _get_last_human(before_msgs)
        datasets, active_dataset_id = _ensure_dataset_registry(state)
        state_with_datasets = {
            **(state or {}),
            "datasets": datasets,
            "active_dataset_id": active_dataset_id,
        }
        active_df = _ensure_df(
            _get_active_data(
                state_with_datasets,
                [
                    "data_raw",
                    "data_sql",
                    "data_wrangled",
                    "data_cleaned",
                    "feature_data",
                ],
            )
        )
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
        if data_wrangled is not None:
            try:
                wrangler_code = response.get("data_wrangler_function")
                wrangler_code_hash = _sha256_text(wrangler_code)
                datasets, active_dataset_id, _did = _register_dataset(
                    state_with_datasets,
                    data=data_wrangled,
                    stage="wrangled",
                    label="data_wrangled",
                    created_by="Data_Wrangling_Agent",
                    provenance={
                        "source_type": "agent",
                        "user_request": last_human,
                        "transform": {
                            "kind": "python_function",
                            "function_name": response.get(
                                "data_wrangler_function_name"
                            ),
                            "function_path": response.get(
                                "data_wrangler_function_path"
                            ),
                            "function_code": _truncate_text(wrangler_code, 12000),
                            "code_sha256": wrangler_code_hash,
                            "recommended_steps": response.get("recommended_steps"),
                            "summary": response.get("data_wrangling_summary"),
                            "error": response.get("data_wrangler_error"),
                            "error_log_path": response.get(
                                "data_wrangler_error_log_path"
                            ),
                        },
                    },
                    parent_id=active_dataset_id,
                    make_active=True,
                )
            except Exception:
                pass
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
            "active_data_key": "data_wrangled"
            if data_wrangled is not None
            else state.get("active_data_key"),
            "datasets": datasets,
            "active_dataset_id": active_dataset_id,
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
        datasets, active_dataset_id = _ensure_dataset_registry(state)
        state_with_datasets = {
            **(state or {}),
            "datasets": datasets,
            "active_dataset_id": active_dataset_id,
        }
        active_df = _ensure_df(
            _get_active_data(
                state_with_datasets,
                [
                    "data_wrangled",
                    "data_raw",
                    "data_sql",
                    "data_cleaned",
                    "feature_data",
                ],
            )
        )
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
        if data_cleaned is not None:
            try:
                cleaner_code = response.get("data_cleaner_function")
                cleaner_code_hash = _sha256_text(cleaner_code)
                datasets, active_dataset_id, _did = _register_dataset(
                    state_with_datasets,
                    data=data_cleaned,
                    stage="cleaned",
                    label="data_cleaned",
                    created_by="Data_Cleaning_Agent",
                    provenance={
                        "source_type": "agent",
                        "user_request": last_human,
                        "transform": {
                            "kind": "python_function",
                            "function_name": response.get("data_cleaner_function_name"),
                            "function_path": response.get("data_cleaner_function_path"),
                            "function_code": _truncate_text(cleaner_code, 12000),
                            "code_sha256": cleaner_code_hash,
                            "recommended_steps": response.get("recommended_steps"),
                            "summary": response.get("data_cleaning_summary"),
                            "error": response.get("data_cleaner_error"),
                            "error_log_path": response.get(
                                "data_cleaner_error_log_path"
                            ),
                        },
                    },
                    parent_id=active_dataset_id,
                    make_active=True,
                )
            except Exception:
                pass
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
            "active_data_key": "data_cleaned"
            if data_cleaned is not None
            else state.get("active_data_key"),
            "datasets": datasets,
            "active_dataset_id": active_dataset_id,
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
        datasets, active_dataset_id = _ensure_dataset_registry(state)
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
        if data_sql is not None:
            try:
                sql_code_full = response.get("sql_query_code")
                sql_code_hash = _sha256_text(sql_code_full)
                sql_code = _truncate_text(sql_code_full, 12000)
                sql_fn_full = response.get("sql_database_function")
                sql_fn_hash = _sha256_text(sql_fn_full)
                sql_fn = _truncate_text(sql_fn_full, 6000)
                datasets, active_dataset_id, _did = _register_dataset(
                    {
                        **state,
                        "datasets": datasets,
                        "active_dataset_id": active_dataset_id,
                    },
                    data=data_sql,
                    stage="sql",
                    label="data_sql",
                    created_by="SQL_Database_Agent",
                    provenance={
                        "source_type": "sql",
                        "user_request": last_human,
                        "transform": {
                            "kind": "sql_query",
                            "sql_query_code": sql_code,
                            "sql_sha256": sql_code_hash,
                            "sql_database_function": sql_fn,
                            "sql_database_function_sha256": sql_fn_hash,
                            "sql_database_function_path": response.get(
                                "sql_database_function_path"
                            ),
                            "sql_database_function_name": response.get(
                                "sql_database_function_name"
                            ),
                        },
                    },
                    parent_id=None,
                    make_active=True,
                )
            except Exception:
                pass
        return {
            **merged,
            "data_sql": data_sql,
            "active_data_key": "data_sql"
            if data_sql is not None
            else state.get("active_data_key"),
            "datasets": datasets,
            "active_dataset_id": active_dataset_id,
            "artifacts": {
                **state.get("artifacts", {}),
                "sql": {
                    "sql_query_code": response.get("sql_query_code"),
                    "sql_database_function": response.get("sql_database_function"),
                    "sql_database_function_path": response.get(
                        "sql_database_function_path"
                    ),
                    "sql_database_function_name": response.get(
                        "sql_database_function_name"
                    ),
                    "data_sql": data_sql,
                },
            },
            "last_worker": "SQL_Database_Agent",
        }

    def node_eda(state: SupervisorDSState):
        print("---EDA TOOLS---")
        before_msgs = list(state.get("messages", []) or [])
        last_human = _get_last_human(before_msgs).lower()
        feature_df = _ensure_df(state.get("feature_data"))
        wants_feature_engineered_report = (
            ("feature-engineered" in last_human or "feature engineered" in last_human)
            and (
                "data" in last_human
                or "dataset" in last_human
                or "features" in last_human
            )
        ) or ("engineered features" in last_human)
        active_df = _ensure_df(
            _get_active_data(
                state,
                [
                    "data_cleaned",
                    "data_wrangled",
                    "data_sql",
                    "data_raw",
                    "feature_data",
                ],
            )
        )
        # If the user explicitly references feature-engineered data, prefer it for EDA/reporting.
        if wants_feature_engineered_report and not _is_empty_df(feature_df):
            active_df = feature_df
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
                [
                    "data_cleaned",
                    "data_wrangled",
                    "data_sql",
                    "data_raw",
                    "feature_data",
                ],
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
        plotly_graph = response.get("plotly_graph")
        viz_error = response.get("data_visualization_error")
        viz_error_path = response.get("data_visualization_error_log_path")
        viz_warning = response.get("data_visualization_warning")
        try:
            from ai_data_science_team.utils.plotly import plotly_from_dict

            fig = plotly_from_dict(plotly_graph) if plotly_graph else None
            trace_types = (
                sorted(
                    {
                        getattr(t, "type", None)
                        for t in getattr(fig, "data", [])
                        if getattr(t, "type", None)
                    }
                )
                if fig is not None
                else []
            )
            title = None
            if fig is not None:
                try:
                    title = getattr(getattr(fig.layout, "title", None), "text", None)
                except Exception:
                    title = None
            viz_summary = (
                response.get("data_visualization_summary") or "Visualization generated."
            )
            if trace_types:
                viz_summary = f"{viz_summary} Trace types: {', '.join(trace_types)}."
            if title:
                viz_summary = f"{viz_summary} Title: {title}."
            merged["messages"].append(
                AIMessage(content=viz_summary, name="data_visualization_agent")
            )
        except Exception:
            pass
        if isinstance(viz_error, str) and viz_error:
            err_bits = [viz_error]
            if isinstance(viz_error_path, str) and viz_error_path:
                err_bits.append(f"Log: {viz_error_path}")
            merged["messages"].append(
                AIMessage(
                    content="Visualization error:\n" + "\n".join(err_bits),
                    name="data_visualization_agent",
                )
            )
        if isinstance(viz_warning, str) and viz_warning:
            merged["messages"].append(
                AIMessage(
                    content="Visualization warning:\n" + viz_warning,
                    name="data_visualization_agent",
                )
            )
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
                    "error": viz_error,
                    "error_log_path": viz_error_path,
                    "warning": viz_warning,
                },
            },
            "last_worker": "Data_Visualization_Agent",
        }

    def node_fe(state: SupervisorDSState):
        print("---FEATURE ENGINEERING---")
        before_msgs = list(state.get("messages", []) or [])
        last_human = _get_last_human(before_msgs)
        datasets, active_dataset_id = _ensure_dataset_registry(state)
        state_with_datasets = {
            **(state or {}),
            "datasets": datasets,
            "active_dataset_id": active_dataset_id,
        }
        active_df = _ensure_df(
            _get_active_data(
                state_with_datasets,
                [
                    "data_cleaned",
                    "data_wrangled",
                    "data_sql",
                    "data_raw",
                    "feature_data",
                ],
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
            response.get("data_engineered"),
            _get_last_human(before_msgs),
            extra_text="Feature engineering completed.",
        )
        if summary_text:
            merged["messages"].append(
                AIMessage(content=summary_text, name="feature_engineering_agent")
            )
        feature_data = response.get("data_engineered")
        if feature_data is not None:
            try:
                fe_code = response.get("feature_engineer_function")
                fe_code_hash = _sha256_text(fe_code)
                datasets, active_dataset_id, _did = _register_dataset(
                    state_with_datasets,
                    data=feature_data,
                    stage="feature",
                    label="feature_data",
                    created_by="Feature_Engineering_Agent",
                    provenance={
                        "source_type": "agent",
                        "user_request": last_human,
                        "transform": {
                            "kind": "python_function",
                            "function_name": response.get(
                                "feature_engineer_function_name"
                            ),
                            "function_path": response.get(
                                "feature_engineer_function_path"
                            ),
                            "function_code": _truncate_text(fe_code, 12000),
                            "code_sha256": fe_code_hash,
                            "recommended_steps": response.get("recommended_steps"),
                            "error": response.get("feature_engineer_error"),
                            "error_log_path": response.get(
                                "feature_engineer_error_log_path"
                            ),
                        },
                    },
                    parent_id=active_dataset_id,
                    make_active=True,
                )
            except Exception:
                pass
        downstream_resets = (
            {
                "eda_artifacts": None,
                "viz_graph": None,
                "model_info": None,
                "mlflow_artifacts": None,
            }
            if feature_data is not None
            else {}
        )
        return {
            **merged,
            "feature_data": feature_data,
            "active_data_key": "feature_data"
            if feature_data is not None
            else state.get("active_data_key"),
            "datasets": datasets,
            "active_dataset_id": active_dataset_id,
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
        # Respect the supervisor's active dataset selection (dataset registry / active_dataset_id),
        # falling back to known state keys when the registry is absent.
        active_df = _ensure_df(
            _get_active_data(
                state,
                [
                    "feature_data",
                    "data_cleaned",
                    "data_wrangled",
                    "data_sql",
                    "data_raw",
                ],
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

        # If user asks for prediction/scoring, use an existing model in the H2O cluster
        # instead of retraining AutoML.
        if isinstance(last_human, str) and any(
            w in last_human.lower()
            for w in ("predict", "prediction", "score", "scoring", "inference")
        ):
            import re

            def _extract_run_id(text: str) -> str | None:
                t = text or ""
                m = re.search(r"\b([0-9a-f]{32})\b", t, flags=re.IGNORECASE)
                return m.group(1) if m else None

            def _extract_model_id(text: str) -> str | None:
                t = text or ""
                # Prefer backticked/quoted ids
                m = re.search(r"(?:`|\"|')(?P<mid>[^`\"']+)(?:`|\"|')", t)
                if m and m.group("mid"):
                    mid = m.group("mid").strip()
                    if len(mid) >= 8:
                        return mid
                # Common H2O AutoML id patterns
                m = re.search(r"\b([A-Za-z0-9_]+AutoML_[A-Za-z0-9_]+)\b", t)
                if m and m.group(1):
                    return m.group(1).strip()
                m = re.search(r"\b([A-Za-z0-9_]+_AutoML_[A-Za-z0-9_]+_model_\d+)\b", t)
                if m and m.group(1):
                    return m.group(1).strip()
                return None

            model_id = _extract_model_id(last_human)
            h2o_art = (state.get("artifacts") or {}).get("h2o")
            h2o_art = h2o_art if isinstance(h2o_art, dict) else {}
            cfg = (state.get("artifacts") or {}).get("config") or {}
            cfg = cfg if isinstance(cfg, dict) else {}
            run_id = _extract_run_id(last_human) or h2o_art.get("mlflow_run_id")
            wants_mlflow = "mlflow" in (last_human or "").lower() or bool(run_id)
            if not model_id:
                model_id = h2o_art.get("best_model_id") or None
            if not model_id and isinstance(h2o_art.get("h2o_train_result"), dict):
                model_id = h2o_art["h2o_train_result"].get("best_model_id")

            # Optional: score via MLflow (preferred when available), so predictions work across restarts.
            if wants_mlflow:
                # If no explicit run_id, try newest run in the configured experiment.
                if not (isinstance(run_id, str) and run_id.strip()):
                    try:
                        import mlflow
                        from mlflow.tracking import MlflowClient

                        tracking_uri = cfg.get("mlflow_tracking_uri")
                        if isinstance(tracking_uri, str) and tracking_uri.strip():
                            mlflow.set_tracking_uri(tracking_uri.strip())
                        exp_name = cfg.get("mlflow_experiment_name") or "H2O AutoML"
                        client = MlflowClient()
                        exp = client.get_experiment_by_name(str(exp_name))
                        if exp is not None:
                            runs = client.search_runs(
                                experiment_ids=[exp.experiment_id],
                                order_by=["attributes.start_time DESC"],
                                max_results=25,
                            )

                            def _run_has_model_artifact(rid: str) -> bool:
                                try:
                                    return bool(
                                        client.list_artifacts(rid, path="model")
                                    )
                                except Exception:
                                    return False

                            # Prefer the newest run that actually contains a logged model.
                            for r in runs or []:
                                rid = getattr(getattr(r, "info", None), "run_id", None)
                                if (
                                    isinstance(rid, str)
                                    and rid
                                    and _run_has_model_artifact(rid)
                                ):
                                    run_id = rid
                                    break
                    except Exception:
                        pass

                if isinstance(run_id, str) and run_id.strip():
                    # Best-effort: drop target column if present so we score only features.
                    target = state.get("target_variable")
                    target = (
                        target
                        if isinstance(target, str) and target in active_df.columns
                        else None
                    )
                    x_df = active_df.drop(columns=[target]) if target else active_df
                    try:
                        import mlflow
                        import pandas as pd
                        import h2o
                        from mlflow.tracking import MlflowClient

                        tracking_uri = cfg.get("mlflow_tracking_uri")
                        if isinstance(tracking_uri, str) and tracking_uri.strip():
                            mlflow.set_tracking_uri(tracking_uri.strip())

                        model_uri = f"runs:/{run_id.strip()}/model"
                        # Validate this run actually has a model logged; otherwise provide a helpful message.
                        try:
                            client = MlflowClient()
                            has_model = any(
                                getattr(item, "path", None) == "model"
                                for item in client.list_artifacts(
                                    run_id.strip(), path=""
                                )
                            )
                        except Exception:
                            has_model = True
                        if not has_model:
                            return {
                                "messages": [
                                    AIMessage(
                                        content=(
                                            f"MLflow run `{run_id}` does not contain a logged model at artifact path `model/`.\n\n"
                                            "This usually means you logged workflow artifacts (tables/json) but did not log a model. "
                                            "Train with MLflow enabled (H2O training logs to `model/`), or provide a run id that contains a model."
                                        ),
                                        name="h2o_ml_agent",
                                    )
                                ],
                                "last_worker": "H2O_ML_Agent",
                            }
                        # Prefer mlflow.h2o flavor for stable scoring (handles H2O models and
                        # lets us coerce categorical columns to match training).
                        h2o.init()
                        try:
                            model = mlflow.h2o.load_model(model_uri)
                        except Exception:
                            model = mlflow.pyfunc.load_model(model_uri)

                        if hasattr(model, "predict") and not hasattr(
                            model, "_model_json"
                        ):
                            # Likely a pyfunc wrapper; predict directly.
                            raw_preds = model.predict(x_df)
                            if isinstance(raw_preds, pd.DataFrame):
                                preds_df = raw_preds
                            elif isinstance(raw_preds, pd.Series):
                                preds_df = raw_preds.to_frame(name="prediction")
                            else:
                                preds_df = pd.DataFrame({"prediction": list(raw_preds)})
                        else:
                            frame = h2o.H2OFrame(x_df)
                            # Coerce expected categorical columns to factor.
                            try:
                                out_json = getattr(model, "_model_json", {}) or {}
                                output = (
                                    out_json.get("output")
                                    if isinstance(out_json, dict)
                                    else {}
                                )
                                names = (
                                    output.get("names")
                                    if isinstance(output, dict)
                                    else None
                                )
                                domains = (
                                    output.get("domains")
                                    if isinstance(output, dict)
                                    else None
                                )
                                if isinstance(names, list) and isinstance(
                                    domains, list
                                ):
                                    for col, dom in zip(names, domains):
                                        if dom is None:
                                            continue
                                        if col in frame.columns:
                                            try:
                                                frame[col] = frame[col].asfactor()
                                            except Exception:
                                                pass
                            except Exception:
                                pass

                            preds_h2o = model.predict(frame)
                            preds_df = preds_h2o.as_data_frame(use_pandas=True)

                        try:
                            preds_df.insert(0, "row_id", range(len(preds_df)))
                            if target:
                                preds_df.insert(
                                    1,
                                    f"actual_{target}",
                                    active_df[target].reset_index(drop=True),
                                )
                        except Exception:
                            pass

                        preds_data = preds_df.to_dict()
                    except Exception as e:
                        return {
                            "messages": [
                                AIMessage(
                                    content=(
                                        f"Failed to score with MLflow run `{run_id}`: {e}\n\n"
                                        f"Tried model URI: `runs:/{run_id}/model`.\n\n"
                                        "Tip: scoring must use the same feature schema as training. "
                                        "If you trained on engineered features, set the active dataset to that feature dataset before scoring."
                                    ),
                                    name="h2o_ml_agent",
                                )
                            ],
                            "last_worker": "H2O_ML_Agent",
                        }

                    datasets, active_dataset_id = _ensure_dataset_registry(state)
                    try:
                        label = f"predictions_mlflow_{run_id}"[:80]
                        datasets, active_dataset_id, pred_id = _register_dataset(
                            {
                                **state,
                                "datasets": datasets,
                                "active_dataset_id": active_dataset_id,
                            },
                            data=preds_data,
                            stage="wrangled",
                            label=label,
                            created_by="H2O_ML_Agent",
                            provenance={
                                "source_type": "agent",
                                "user_request": last_human,
                                "transform": {
                                    "kind": "mlflow_predict",
                                    "run_id": run_id,
                                    "model_uri": f"runs:/{run_id.strip()}/model",
                                    "dropped_target": bool(target),
                                },
                            },
                            parent_id=active_dataset_id,
                            make_active=True,
                        )
                    except Exception:
                        pred_id = None

                    try:
                        preview_md = preds_df.head(5).to_markdown(index=False)
                        msg = f"Scored dataset with MLflow run `{run_id}`. Predictions shape: {preds_df.shape}.\n\n{preview_md}"
                    except Exception:
                        msg = f"Scored dataset with MLflow run `{run_id}`."

                    return {
                        "messages": [AIMessage(content=msg, name="h2o_ml_agent")],
                        "data_wrangled": preds_data,
                        "active_data_key": "data_wrangled",
                        "datasets": datasets,
                        "active_dataset_id": active_dataset_id,
                        "artifacts": {
                            **state.get("artifacts", {}),
                            "mlflow_predictions": {
                                "run_id": run_id,
                                "predictions_dataset_id": pred_id,
                            },
                        },
                        "last_worker": "H2O_ML_Agent",
                    }
            if not isinstance(model_id, str) or not model_id.strip():
                return {
                    "messages": [
                        AIMessage(
                            content=(
                                "To make predictions, provide an H2O `model_id` (or train a model first). "
                                "Example: `predict with model `XGBoost_grid_...` on the dataset`."
                            ),
                            name="h2o_ml_agent",
                        )
                    ],
                    "last_worker": "H2O_ML_Agent",
                }

            # Best-effort: drop target column if present so we score only features.
            target = state.get("target_variable")
            target = (
                target
                if isinstance(target, str) and target in active_df.columns
                else None
            )
            x_df = active_df.drop(columns=[target]) if target else active_df

            try:
                import h2o

                h2o.init()
                model = h2o.get_model(model_id.strip())
                frame = h2o.H2OFrame(x_df)
                preds_h2o = model.predict(frame)
                preds_df = preds_h2o.as_data_frame(use_pandas=True)
                try:
                    preds_df.insert(0, "row_id", range(len(preds_df)))
                    if target:
                        preds_df.insert(
                            1,
                            f"actual_{target}",
                            active_df[target].reset_index(drop=True),
                        )
                except Exception:
                    pass
                preds_data = preds_df.to_dict()
            except Exception as e:
                return {
                    "messages": [
                        AIMessage(
                            content=(
                                f"Failed to score with model `{model_id}`: {e}\n\n"
                                "Tip: model IDs are only available while the H2O cluster is running. "
                                "If you restarted, retrain or load a saved model."
                            ),
                            name="h2o_ml_agent",
                        )
                    ],
                    "last_worker": "H2O_ML_Agent",
                }

            # Register predictions as a new dataset (tabular output) for downstream viz/EDA.
            datasets, active_dataset_id = _ensure_dataset_registry(state)
            try:
                label = f"predictions_{model_id}"[:80]
                datasets, active_dataset_id, pred_id = _register_dataset(
                    {
                        **state,
                        "datasets": datasets,
                        "active_dataset_id": active_dataset_id,
                    },
                    data=preds_data,
                    stage="wrangled",
                    label=label,
                    created_by="H2O_ML_Agent",
                    provenance={
                        "source_type": "agent",
                        "user_request": last_human,
                        "transform": {
                            "kind": "h2o_predict",
                            "model_id": model_id,
                            "dropped_target": bool(target),
                            "n_rows": int(getattr(x_df, "shape", (0, 0))[0] or 0),
                            "n_cols": int(getattr(x_df, "shape", (0, 0))[1] or 0),
                        },
                    },
                    parent_id=active_dataset_id,
                    make_active=True,
                )
            except Exception:
                pred_id = None

            try:
                preview_md = preds_df.head(5).to_markdown(index=False)
                msg = f"Scored dataset with model `{model_id}`. Predictions shape: {preds_df.shape}.\n\n{preview_md}"
            except Exception:
                msg = f"Scored dataset with model `{model_id}`."

            return {
                "messages": [AIMessage(content=msg, name="h2o_ml_agent")],
                "data_wrangled": preds_data,
                "active_data_key": "data_wrangled",
                "datasets": datasets,
                "active_dataset_id": active_dataset_id,
                "artifacts": {
                    **state.get("artifacts", {}),
                    "h2o_predictions": {
                        "model_id": model_id,
                        "predictions_dataset_id": pred_id,
                    },
                },
                "last_worker": "H2O_ML_Agent",
            }

        h2o_ml_agent.invoke_messages(
            messages=before_msgs,
            user_instructions=last_human,
            data_raw=active_df,
            target_variable=state.get("target_variable"),
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
        mlflow_run_id = response.get("mlflow_run_id")
        if mlflow_run_id:
            merged["messages"].append(
                AIMessage(
                    content=f"MLflow logging enabled. Run ID: `{mlflow_run_id}`",
                    name="h2o_ml_agent",
                )
            )
            model_uri = response.get("mlflow_model_uri")
            if isinstance(model_uri, str) and model_uri.strip():
                merged["messages"].append(
                    AIMessage(
                        content=f"MLflow model URI: `{model_uri.strip()}`",
                        name="h2o_ml_agent",
                    )
                )
        leaderboard = response.get("leaderboard")
        return {
            **merged,
            "model_info": leaderboard,
            "mlflow_artifacts": response.get("mlflow_model")
            or (
                {
                    "run_id": mlflow_run_id,
                    "model_uri": response.get("mlflow_model_uri"),
                }
                if mlflow_run_id
                else None
            ),
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

    def node_eval(state: SupervisorDSState):
        print("---MODEL EVALUATION---")
        before_msgs = list(state.get("messages", []) or [])
        feature_df = _ensure_df(state.get("feature_data"))
        active_df = (
            feature_df
            if not _is_empty_df(feature_df)
            else _ensure_df(
                _get_active_data(
                    state, ["data_cleaned", "data_wrangled", "data_sql", "data_raw"]
                )
            )
        )
        if _is_empty_df(active_df):
            return {
                "messages": [
                    AIMessage(
                        content="No dataset is available for evaluation. Load data and train a model first.",
                        name="model_evaluation_agent",
                    )
                ],
                "last_worker": "Model_Evaluation_Agent",
            }
        h2o_art = (state.get("artifacts") or {}).get("h2o")
        model_artifacts = h2o_art if isinstance(h2o_art, dict) else {}
        model_evaluation_agent.invoke_messages(
            messages=before_msgs,
            data_raw=active_df,
            model_artifacts=model_artifacts,
            target_variable=state.get("target_variable"),
        )
        response = model_evaluation_agent.response or {}
        merged = _merge_messages(before_msgs, response)
        merged["messages"] = _tag_messages(
            merged.get("messages"), "model_evaluation_agent"
        )
        eval_artifacts = response.get("eval_artifacts")
        plotly_graph = response.get("plotly_graph")
        return {
            **merged,
            "eval_artifacts": eval_artifacts,
            "artifacts": {
                **state.get("artifacts", {}),
                "eval": {
                    "eval_artifacts": eval_artifacts,
                    "plotly_graph": plotly_graph,
                },
            },
            "last_worker": "Model_Evaluation_Agent",
        }

    def node_mlflow_log(state: SupervisorDSState):
        print("---MLFLOW LOGGING---")
        before_msgs = list(state.get("messages", []) or [])

        # Pull config from the supervisor artifacts (optional).
        cfg = {}
        try:
            cfg = (state.get("artifacts") or {}).get("config") or {}
        except Exception:
            cfg = {}

        tracking_uri = cfg.get("mlflow_tracking_uri") if isinstance(cfg, dict) else None
        artifact_root = (
            cfg.get("mlflow_artifact_root") if isinstance(cfg, dict) else None
        )
        experiment_name = (
            cfg.get("mlflow_experiment_name") if isinstance(cfg, dict) else None
        )

        # Attempt to reuse an existing run id (from H2O training) if present.
        run_id = None
        h2o_art = (state.get("artifacts") or {}).get("h2o")
        if isinstance(h2o_art, dict):
            run_id = h2o_art.get("mlflow_run_id")
            if not run_id and isinstance(h2o_art.get("h2o_train_result"), dict):
                run_id = h2o_art["h2o_train_result"].get("mlflow_run_id")
            if not run_id and isinstance(h2o_art.get("model_results"), dict):
                run_id = h2o_art["model_results"].get("mlflow_run_id")

        feature_df = _ensure_df(state.get("feature_data"))
        active_df = (
            feature_df
            if not _is_empty_df(feature_df)
            else _ensure_df(
                _get_active_data(
                    state, ["data_cleaned", "data_wrangled", "data_sql", "data_raw"]
                )
            )
        )
        viz_graph = state.get("viz_graph")
        eval_payload = (state.get("artifacts") or {}).get("eval")
        eval_artifacts = state.get("eval_artifacts")
        eval_plot = None
        if isinstance(eval_payload, dict):
            eval_plot = eval_payload.get("plotly_graph")

        logged: dict = {"tables": [], "figures": [], "dicts": [], "metrics": []}
        message_lines: list[str] = []

        try:
            import mlflow
            import json
            from pathlib import Path

            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            if experiment_name:
                # Best-effort: if an artifact root is configured, ensure the experiment exists
                # with that artifact location (applies only when creating new experiments).
                try:
                    from mlflow.tracking import MlflowClient
                    import re

                    if isinstance(artifact_root, str) and artifact_root.strip():
                        root = Path(artifact_root).expanduser().resolve()
                        root.mkdir(parents=True, exist_ok=True)
                        safe = re.sub(
                            r"[^A-Za-z0-9._-]+", "_", str(experiment_name)
                        ).strip("_")
                        safe = safe or "experiment"
                        artifact_location = (root / safe).as_uri()
                        client = MlflowClient(tracking_uri=tracking_uri)
                        exp = client.get_experiment_by_name(str(experiment_name))
                        if exp is None:
                            client.create_experiment(
                                name=str(experiment_name),
                                artifact_location=artifact_location,
                            )
                except Exception:
                    pass
                mlflow.set_experiment(experiment_name)

            # Start or resume the run
            with mlflow.start_run(run_id=run_id) as run:
                run_id = run.info.run_id

                # Basic tags/params
                try:
                    mlflow.set_tags(
                        {
                            "app": "supervisor_ds_team",
                            "active_data_key": state.get("active_data_key") or "",
                            "active_dataset_id": state.get("active_dataset_id") or "",
                        }
                    )
                except Exception:
                    pass

                # Log a small dataset preview + schema
                if active_df is not None and not _is_empty_df(active_df):
                    try:
                        mlflow.log_table(
                            active_df.head(200),
                            artifact_file="tables/data_preview.json",
                        )
                        logged["tables"].append("tables/data_preview.json")
                    except Exception:
                        pass
                    try:
                        schema = {
                            "columns": [
                                {"name": str(c), "dtype": str(active_df[c].dtype)}
                                for c in list(active_df.columns)
                            ],
                            "shape": list(active_df.shape),
                        }
                        mlflow.log_dict(schema, artifact_file="tables/schema.json")
                        logged["dicts"].append("tables/schema.json")
                    except Exception:
                        pass

                # Log pipeline (dataset lineage + reproduction script)
                try:
                    from ai_data_science_team.utils.pipeline import (
                        build_pipeline_snapshot,
                    )

                    ds = state.get("datasets")
                    ds = ds if isinstance(ds, dict) else {}
                    pipe = build_pipeline_snapshot(
                        ds, active_dataset_id=state.get("active_dataset_id")
                    )
                    if isinstance(pipe, dict) and pipe.get("lineage"):
                        pipe_spec = dict(pipe)
                        script = pipe_spec.pop("script", None)
                        mlflow.log_dict(
                            pipe_spec, artifact_file="pipeline/pipeline_spec.json"
                        )
                        logged["dicts"].append("pipeline/pipeline_spec.json")
                        if isinstance(script, str) and script.strip():
                            if hasattr(mlflow, "log_text"):
                                mlflow.log_text(
                                    script, artifact_file="pipeline/pipeline_repro.py"
                                )
                                logged["dicts"].append("pipeline/pipeline_repro.py")
                            else:
                                mlflow.log_dict(
                                    {"script": script},
                                    artifact_file="pipeline/pipeline_repro.json",
                                )
                                logged["dicts"].append("pipeline/pipeline_repro.json")
                        try:
                            if pipe.get("pipeline_hash"):
                                mlflow.set_tag(
                                    "pipeline_hash", str(pipe.get("pipeline_hash"))
                                )
                        except Exception:
                            pass
                except Exception:
                    pass

                # Log visualization plot (if any)
                if viz_graph:
                    try:
                        mlflow.log_dict(viz_graph, artifact_file="plots/viz.json")
                        logged["dicts"].append("plots/viz.json")
                    except Exception:
                        pass
                    try:
                        import plotly.io as pio

                        fig = pio.from_json(json.dumps(viz_graph))
                        mlflow.log_figure(fig, artifact_file="plots/viz.html")
                        logged["figures"].append("plots/viz.html")
                    except Exception:
                        pass

                # Log evaluation artifacts + metrics + plot
                if eval_artifacts:
                    try:
                        mlflow.log_dict(
                            eval_artifacts,
                            artifact_file="evaluation/eval_artifacts.json",
                        )
                        logged["dicts"].append("evaluation/eval_artifacts.json")
                    except Exception:
                        pass
                    try:
                        metrics = (
                            eval_artifacts.get("metrics")
                            if isinstance(eval_artifacts, dict)
                            else None
                        )
                        if isinstance(metrics, dict):
                            safe = {}
                            for k, v in metrics.items():
                                try:
                                    safe[str(k)] = float(v)
                                except Exception:
                                    continue
                            if safe:
                                mlflow.log_metrics(safe)
                                logged["metrics"].extend(list(safe.keys()))
                    except Exception:
                        pass
                if eval_plot:
                    try:
                        mlflow.log_dict(
                            eval_plot, artifact_file="evaluation/eval_plot.json"
                        )
                        logged["dicts"].append("evaluation/eval_plot.json")
                    except Exception:
                        pass
                    try:
                        import plotly.io as pio

                        fig = pio.from_json(json.dumps(eval_plot))
                        mlflow.log_figure(
                            fig, artifact_file="evaluation/eval_plot.html"
                        )
                        logged["figures"].append("evaluation/eval_plot.html")
                    except Exception:
                        pass

        except Exception as e:
            message_lines.append(f"MLflow logging failed: {e}")

        if run_id:
            message_lines.append(f"Logged workflow artifacts to MLflow run `{run_id}`.")
        if any(logged.values()):
            message_lines.append(
                "Logged: "
                + ", ".join(
                    [
                        *(
                            [f"{len(logged['tables'])} table(s)"]
                            if logged["tables"]
                            else []
                        ),
                        *(
                            [f"{len(logged['figures'])} figure(s)"]
                            if logged["figures"]
                            else []
                        ),
                        *(
                            [f"{len(logged['dicts'])} json artifact(s)"]
                            if logged["dicts"]
                            else []
                        ),
                        *(
                            [f"{len(logged['metrics'])} metric(s)"]
                            if logged["metrics"]
                            else []
                        ),
                    ]
                )
                + "."
            )
        if not message_lines:
            message_lines.append(
                "No artifacts were available to log yet. Train a model and/or create a chart first."
            )

        msg = "\n".join(message_lines)
        merged = {"messages": [AIMessage(content=msg, name="mlflow_logging_agent")]}
        merged["messages"] = _tag_messages(
            merged.get("messages"), "mlflow_logging_agent"
        )
        return {
            **merged,
            "mlflow_artifacts": {"run_id": run_id, "logged": logged},
            "artifacts": {
                **state.get("artifacts", {}),
                "mlflow_log": {"run_id": run_id, "logged": logged},
            },
            "last_worker": "MLflow_Logging_Agent",
        }

    workflow = StateGraph(SupervisorDSState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("Data_Loader_Tools_Agent", node_loader)
    workflow.add_node("Data_Merge_Agent", node_merge)
    workflow.add_node("Data_Wrangling_Agent", node_wrangling)
    workflow.add_node("Data_Cleaning_Agent", node_cleaning)
    workflow.add_node("EDA_Tools_Agent", node_eda)
    workflow.add_node("Data_Visualization_Agent", node_viz)
    workflow.add_node("SQL_Database_Agent", node_sql)
    workflow.add_node("Feature_Engineering_Agent", node_fe)
    workflow.add_node("H2O_ML_Agent", node_h2o)
    workflow.add_node("Model_Evaluation_Agent", node_eval)
    workflow.add_node("MLflow_Logging_Agent", node_mlflow_log)
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
        model_evaluation_agent,
        workflow_planner_agent=None,
        checkpointer: Optional[Checkpointer] = None,
        temperature: float = 1.0,
    ):
        self._params = {
            "model": model,
            "workflow_planner_agent": workflow_planner_agent,
            "data_loader_agent": data_loader_agent,
            "data_wrangling_agent": data_wrangling_agent,
            "data_cleaning_agent": data_cleaning_agent,
            "eda_tools_agent": eda_tools_agent,
            "data_visualization_agent": data_visualization_agent,
            "sql_database_agent": sql_database_agent,
            "feature_engineering_agent": feature_engineering_agent,
            "h2o_ml_agent": h2o_ml_agent,
            "mlflow_tools_agent": mlflow_tools_agent,
            "model_evaluation_agent": model_evaluation_agent,
            "checkpointer": checkpointer,
            "temperature": temperature,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response: Optional[dict] = None

    def _make_compiled_graph(self):
        self.response = None
        return make_supervisor_ds_team(
            model=self._params["model"],
            workflow_planner_agent=self._params["workflow_planner_agent"],
            data_loader_agent=self._params["data_loader_agent"],
            data_wrangling_agent=self._params["data_wrangling_agent"],
            data_cleaning_agent=self._params["data_cleaning_agent"],
            eda_tools_agent=self._params["eda_tools_agent"],
            data_visualization_agent=self._params["data_visualization_agent"],
            sql_database_agent=self._params["sql_database_agent"],
            feature_engineering_agent=self._params["feature_engineering_agent"],
            h2o_ml_agent=self._params["h2o_ml_agent"],
            mlflow_tools_agent=self._params["mlflow_tools_agent"],
            model_evaluation_agent=self._params["model_evaluation_agent"],
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
