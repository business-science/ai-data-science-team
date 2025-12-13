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
    temperature: float = 0,
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
- Model_Evaluation_Agent: Evaluates a trained model on a holdout split and returns standardized metrics + plots (confusion matrix/ROC or residuals).
- MLflow_Logging_Agent: Logs workflow artifacts deterministically to MLflow (tables/figures/metrics) and returns the run id.
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
        wants_wrangling = has("wrangle", "transform", "rename", "format", "standardize")
        wants_eda = has(
            "describe", "eda", "summary", "correlation", "sweetviz", "missingness"
        )
        wants_feature = has("feature", "encode", "one-hot", "feat eng")

        # "model" is ambiguous (e.g., "bike model" vs ML model). Only treat as ML intent when
        # there are explicit ML/training signals, or action verbs paired with "model".
        ml_signal = has(
            "train",
            "automl",
            "predict",
            "classification",
            "classify",
            "regression",
            "mlflow",
            "cross-validation",
            "cross validation",
            "cv",
            "hyperparameter",
            "tune",
            "xgboost",
            "random forest",
            "lightgbm",
            "catboost",
            "logistic",
            "neural network",
            "deep learning",
        )
        model_word = "model" in last_human
        product_model_context = has(
            "bike model",
            "car model",
            "product model",
            "model year",
            "phone model",
            "vehicle model",
        ) or (wants_viz and has("by model", "per model", "for each model"))
        wants_model = bool(
            ml_signal
            or (
                model_word
                and has("build", "create", "fit", "train", "tune", "predict", "develop")
                and not product_model_context
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
                wants_clean,
                wants_wrangling,
                wants_eda,
                wants_feature,
                wants_model,
            ]
        )
        load_only = wants_load and mentions_file and not wants_more_processing
        return {
            "list_files": wants_list_files,
            "preview": wants_preview,
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
        cfg = (state.get("artifacts") or {}).get("config") or {}
        proactive_mode = bool(cfg.get("proactive_workflow_mode")) if isinstance(cfg, dict) else False

        # Track per-user-request steps (within the current user message) to support
        # deterministic sequencing for multi-step prompts.
        last_human_msg = None
        for m in reversed(clean_msgs or []):
            role = getattr(m, "role", getattr(m, "type", None))
            if role in ("human", "user"):
                last_human_msg = m
                break
        current_request_id = getattr(last_human_msg, "id", None) if last_human_msg else None

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

        data_ready = (
            active_data_key is not None and state.get(active_data_key) is not None
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
                if loader_artifacts.get("status") == "ok" and loader_artifacts.get("data") is not None:
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
            elif last_worker == "SQL_Database_Agent" and state.get("data_sql") is not None:
                handled_steps["sql"] = True
            elif last_worker == "Data_Wrangling_Agent" and state.get("data_wrangled") is not None:
                handled_steps["wrangle"] = True
            elif last_worker == "Data_Cleaning_Agent" and state.get("data_cleaned") is not None:
                handled_steps["clean"] = True
            elif last_worker == "EDA_Tools_Agent" and state.get("eda_artifacts") is not None:
                handled_steps["eda"] = True
            elif last_worker == "Data_Visualization_Agent" and state.get("viz_graph") is not None:
                handled_steps["viz"] = True
            elif last_worker == "Feature_Engineering_Agent" and state.get("feature_data") is not None:
                handled_steps["feature"] = True
            elif last_worker == "H2O_ML_Agent" and state.get("model_info") is not None:
                handled_steps["model"] = True
            elif last_worker == "Model_Evaluation_Agent" and state.get("eval_artifacts") is not None:
                handled_steps["evaluate"] = True
            elif last_worker == "MLflow_Logging_Agent" and state.get("mlflow_artifacts") is not None:
                handled_steps["mlflow_log"] = True
            elif last_worker == "MLflow_Tools_Agent" and state.get("mlflow_artifacts") is not None:
                handled_steps["mlflow_tools"] = True

        step_to_worker = {
            "list_files": "Data_Loader_Tools_Agent",
            "load": "Data_Loader_Tools_Agent",
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
        if use_planner and workflow_planner_agent is not None and current_request_id is not None:
            if state_plan_req == current_request_id and isinstance(state_plan, dict):
                planned_steps = state_plan.get("steps") if isinstance(state_plan.get("steps"), list) else None
                plan_questions = state_plan.get("questions") if isinstance(state_plan.get("questions"), list) else []
                plan_notes = state_plan.get("notes") if isinstance(state_plan.get("notes"), list) else []
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
                planned_steps = plan.get("steps") if isinstance(plan.get("steps"), list) else None
                plan_questions = plan.get("questions") if isinstance(plan.get("questions"), list) else []
                plan_notes = plan.get("notes") if isinstance(plan.get("notes"), list) else []
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
                    note_text = "\n".join(f"- {n}" for n in plan_notes) if plan_notes else ""
                    msg = f"Planned workflow: {pretty_steps}"
                    if note_text:
                        msg = msg + "\n\nNotes:\n" + note_text
                    planner_messages = [AIMessage(content=msg, name="workflow_planner_agent")]

            # If the planner needs user input, ask and stop.
            if plan_questions and not (planned_steps and len(planned_steps) > 0):
                question_text = "\n".join(f"- {q}" for q in plan_questions)
                note_text = "\n".join(f"- {n}" for n in plan_notes) if plan_notes else ""
                msg = "To run the workflow, I need:\n" + question_text
                if note_text:
                    msg = msg + "\n\nNotes:\n" + note_text
                return {
                    "messages": [AIMessage(content=msg, name="workflow_planner_agent")],
                    "next": "FINISH",
                    "active_data_key": active_data_key,
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
                if not data_ready and needs_data and not (
                    intents.get("load") or intents.get("load_only") or intents.get("sql")
                ):
                    steps.insert(0, "load")

                # Transformations
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
                            **({"messages": planner_messages} if planner_messages else {}),
                            "next": "FINISH",
                            "active_data_key": active_data_key,
                            "handled_request_id": handled_request_id,
                            "handled_steps": handled_steps,
                            "attempted_steps": attempted_steps,
                            "workflow_plan_request_id": state_plan_req,
                            "workflow_plan": state_plan,
                            "target_variable": planned_target,
                        }

                    # Guard data-dependent steps.
                    if step in ("wrangle", "clean", "eda", "viz", "feature", "model", "evaluate") and not data_ready:
                        print(f"  step '{step}' requires data but none is ready -> Data_Loader_Tools_Agent")
                        attempted_steps["load"] = True
                        return {
                            **({"messages": planner_messages} if planner_messages else {}),
                            "next": "Data_Loader_Tools_Agent",
                            "active_data_key": active_data_key,
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
            "handled_request_id": handled_request_id,
            "handled_steps": handled_steps,
            "attempted_steps": attempted_steps,
            "workflow_plan_request_id": state_plan_req,
            "workflow_plan": state_plan,
            "target_variable": planned_target,
        }

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

    def _merge_messages(
        before_messages: Sequence[BaseMessage], response: dict
    ) -> dict:
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
        fallback_loaded_dataset = False

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

                if any(w in last_human_lower for w in ("load", "read", "import", "open")):
                    m = re.search(
                        r"(?:`|\"|')?([\\w\\-./~]+\\.(?:csv|tsv|parquet|xlsx?|jsonl|ndjson|json)(?:\\.gz)?)",
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
                    loader_artifacts = {**loader_artifacts, "load_file_fallback": marker}
                else:
                    loader_artifacts = {"load_file_fallback": marker}

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
                        if isinstance(item, dict):
                            if "filename" in item:
                                names.append(item.get("filename"))
                                rows.append(
                                    {
                                        "filename": item.get("filename"),
                                        "type": item.get("type"),
                                        "path": item.get("path") or item.get("filepath"),
                                    }
                                )
                                continue
                            if "file_path" in item:
                                fp = item.get("file_path")
                                import os

                                fn = os.path.basename(fp) if isinstance(fp, str) else str(fp)
                                names.append(fn)
                                rows.append({"filename": fn, "type": "file", "path": fp})
                                continue
                            if "absolute_path" in item or "name" in item:
                                ap = item.get("absolute_path")
                                import os

                                fn = item.get("name") or (os.path.basename(ap) if isinstance(ap, str) else str(ap))
                                names.append(fn)
                                rows.append(
                                    {"filename": fn, "type": item.get("type"), "path": ap}
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

                                fn = os.path.basename(fp) if isinstance(fp, str) else str(fp)
                                names.append(fn)
                                rows.append({"filename": fn, "type": "file", "path": fp})
                            elif "absolute_path" in v or "name" in v:
                                ap = v.get("absolute_path")
                                import os

                                fn = v.get("name") or (os.path.basename(ap) if isinstance(ap, str) else str(ap))
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
                wants_csv_only = "csv" in last_human and ("list" in last_human or "files" in last_human)
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
                            c for c in ["filename", "type", "path"] if c in df_listing.columns
                        ]
                        table_text = df_listing[table_cols].to_markdown(index=False)
                    # If the user asked for a table or better formatting, try a tiny LLM summary
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
                        summary_msg = AIMessage(content=llm_text, name="data_loader_agent")
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
        plotly_graph = response.get("plotly_graph")
        try:
            from ai_data_science_team.utils.plotly import plotly_from_dict

            fig = plotly_from_dict(plotly_graph) if plotly_graph else None
            trace_types = (
                sorted({getattr(t, "type", None) for t in getattr(fig, "data", []) if getattr(t, "type", None)})
                if fig is not None
                else []
            )
            title = None
            if fig is not None:
                try:
                    title = getattr(getattr(fig.layout, "title", None), "text", None)
                except Exception:
                    title = None
            viz_summary = response.get("data_visualization_summary") or "Visualization generated."
            if trace_types:
                viz_summary = f"{viz_summary} Trace types: {', '.join(trace_types)}."
            if title:
                viz_summary = f"{viz_summary} Title: {title}."
            merged["messages"].append(
                AIMessage(content=viz_summary, name="data_visualization_agent")
            )
        except Exception:
            pass
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

    def node_eval(state: SupervisorDSState):
        print("---MODEL EVALUATION---")
        before_msgs = list(state.get("messages", []) or [])
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
        merged["messages"] = _tag_messages(merged.get("messages"), "model_evaluation_agent")
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

        active_df = _ensure_df(
            _get_active_data(
                state,
                ["feature_data", "data_cleaned", "data_wrangled", "data_sql", "data_raw"],
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

            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            if experiment_name:
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
                        }
                    )
                except Exception:
                    pass

                # Log a small dataset preview + schema
                if active_df is not None and not _is_empty_df(active_df):
                    try:
                        mlflow.log_table(active_df.head(200), artifact_file="tables/data_preview.json")
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
                        mlflow.log_dict(eval_artifacts, artifact_file="evaluation/eval_artifacts.json")
                        logged["dicts"].append("evaluation/eval_artifacts.json")
                    except Exception:
                        pass
                    try:
                        metrics = eval_artifacts.get("metrics") if isinstance(eval_artifacts, dict) else None
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
                        mlflow.log_dict(eval_plot, artifact_file="evaluation/eval_plot.json")
                        logged["dicts"].append("evaluation/eval_plot.json")
                    except Exception:
                        pass
                    try:
                        import plotly.io as pio

                        fig = pio.from_json(json.dumps(eval_plot))
                        mlflow.log_figure(fig, artifact_file="evaluation/eval_plot.html")
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
                        *([f"{len(logged['tables'])} table(s)"] if logged["tables"] else []),
                        *([f"{len(logged['figures'])} figure(s)"] if logged["figures"] else []),
                        *([f"{len(logged['dicts'])} json artifact(s)"] if logged["dicts"] else []),
                        *([f"{len(logged['metrics'])} metric(s)"] if logged["metrics"] else []),
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
        merged["messages"] = _tag_messages(merged.get("messages"), "mlflow_logging_agent")
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
        temperature: float = 0.0,
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
