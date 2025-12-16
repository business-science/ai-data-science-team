# BUSINESS SCIENCE UNIVERSITY
# AI DATA SCIENCE TEAM
# ***
# * Agents: H2O Machine Learning Agent

import os
import json
from typing_extensions import TypedDict, Annotated, Sequence, Literal, Optional
import operator

import pandas as pd
from IPython.display import Markdown

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage

from langgraph.types import Command, Checkpointer
from langgraph.checkpoint.memory import MemorySaver

from ai_data_science_team.templates import (
    node_func_execute_agent_code_on_data,
    node_func_human_review,
    node_func_fix_agent_code,
    node_func_report_agent_outputs,
    create_coding_agent_graph,
    BaseAgent,
)
from ai_data_science_team.parsers.parsers import PythonOutputParser
from ai_data_science_team.utils.regex import (
    relocate_imports_inside_function,
    add_comments_to_top,
    format_agent_name,
    format_recommended_steps,
    get_generic_summary,
)
from ai_data_science_team.tools.dataframe import get_dataframe_summary
from ai_data_science_team.utils.logging import log_ai_function, log_ai_error
from ai_data_science_team.utils.messages import get_last_user_message_content
from ai_data_science_team.tools.h2o import H2O_AUTOML_DOCUMENTATION

AGENT_NAME = "h2o_ml_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")
MAX_SUMMARY_COLUMNS = 30
MAX_SUMMARY_CHARS = 5000

DEFAULT_ML_STEPS = format_recommended_steps(
    """
1. Verify target column exists and has sufficient non-null values; ensure target is categorical for classification or numeric for regression.
2. Review column types; let H2OAutoML handle encoding, avoid heavy feature engineering here.
3. Cap runtime and/or max_models to fit resource budget; disable deep learning if not needed.
4. Use stratified folds (nfolds) if target is imbalanced; consider balance_classes=True for classification.
5. Save leaderboard and best model; optionally log metrics/artifacts to MLflow if enabled.
    """,
    heading="# Recommended ML Steps:",
)


class H2OMLAgent(BaseAgent):
    """
    A Machine Learning agent that uses H2O's AutoML for training,
    allowing the user to specify a model directory for saving the best model.
    If neither model_directory nor log_path is provided, model saving is skipped.

    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model used to generate the ML code.
    n_samples : int, optional
        Number of samples used when summarizing the dataset. Defaults to 30.
    log : bool, optional
        Whether to log the generated code and errors. Defaults to False.
    log_path : str, optional
        Directory path for storing log files. Defaults to None.
    file_name : str, optional
        Name of the Python file for saving the generated code. Defaults to "h2o_automl.py".
    function_name : str, optional
        Name of the function that performs the AutoML training. Defaults to "h2o_automl".
    model_directory : str or None, optional
        Directory to save the H2O Machine Learning model. If None, defaults to log_path (if available).
        If both are None, no model is saved. Defaults to None.
    overwrite : bool, optional
        Whether to overwrite the log file if it exists. Defaults to True.
    human_in_the_loop : bool, optional
        Enables user review of the code. Defaults to False.
    bypass_recommended_steps : bool, optional
        If True, skips the recommended steps prompt. Defaults to False.
    bypass_explain_code : bool, optional
        If True, skips the code-explanation step. Defaults to False.
    enable_mlflow : bool, default False
        Whether to enable MLflow logging. If False, skip MLflow entirely.
    mlflow_tracking_uri : str or None
        If provided, sets MLflow tracking URI at runtime.
    mlflow_experiment_name : str
        Name of the MLflow experiment (created if doesn't exist).
    mlflow_run_name : str, default None
        A custom name for the MLflow run.
    checkpointer : langgraph.checkpoint.memory.MemorySaver, optional
        A checkpointer object for saving the agent's state. Defaults to None.


    Methods
    -------
    update_params(**kwargs)
        Updates the agent's parameters and rebuilds the compiled state graph.
    ainvoke_agent(user_instructions, data_raw, target_variable, ...)
        Asynchronously runs the agent to produce an H2O AutoML model, optionally saving the model to disk.
    invoke_agent(user_instructions, data_raw, target_variable, ...)
        Synchronously runs the agent to produce an H2O AutoML model, optionally saving the model to disk.
    get_leaderboard()
        Retrieves the H2O AutoML leaderboard from the agent's response.
    get_best_model_id()
        Retrieves the best model ID from the agent's response.
    get_model_path()
        Retrieves the saved model path from the agent's response (or None if not saved).
    get_data_raw()
        Retrieves the raw data as a DataFrame from the agent's response.
    get_h2o_train_function(markdown=False)
        Retrieves the H2O AutoML function code generated by the agent.
    get_recommended_ml_steps(markdown=False)
        Retrieves recommended ML steps from the agent's response.
    get_workflow_summary()
        Retrieves a summary of the agent's workflow.
    get_response()
        Returns the entire response dictionary.
    show()
        Visualizes the compiled graph as a Mermaid diagram.

    Examples
    --------
    ```python
    from langchain_openai import ChatOpenAI
    import pandas as pd
    from ai_data_science_team.ml_agents import H2OMLAgent

    llm = ChatOpenAI(model="gpt-4o-mini")

    df = pd.read_csv("data/churn_data.csv")

    ml_agent = H2OMLAgent(
        model=llm,
        log=True,
        log_path=LOG_PATH,
        model_directory=MODEL_PATH,
    )

    ml_agent.invoke_agent(
        data_raw=df.drop(columns=["customerID"]),
        user_instructions="Please do classification on 'Churn'. Use a max runtime of 30 seconds.",
        target_variable="Churn"
    )

    # Retrieve and display the leaderboard of models
    ml_agent.get_leaderboard()

    # Get the H2O training function in markdown format
    ml_agent.get_h2o_train_function(markdown=True)

    # Get the recommended machine learning steps in markdown format
    ml_agent.get_recommended_ml_steps(markdown=True)

    # Get a summary of the workflow in markdown format
    ml_agent.get_workflow_summary(markdown=True)

    # Get a summary of the logs in markdown format
    ml_agent.get_log_summary(markdown=True)

    # Get the path to the saved model
    model_path = ml_agent.get_model_path()
    model_path
    ```

    Returns
    -------
    H2OMLAgent : langchain.graphs.CompiledStateGraph
        An instance of the H2O ML agent.

    """

    def __init__(
        self,
        model,
        n_samples=30,
        log=False,
        log_path=None,
        file_name="h2o_automl.py",
        function_name="h2o_automl",
        model_directory=None,
        overwrite=True,
        human_in_the_loop=False,
        bypass_recommended_steps=False,
        bypass_explain_code=False,
        enable_mlflow=False,
        mlflow_tracking_uri=None,
        mlflow_artifact_root=None,
        mlflow_experiment_name="H2O AutoML",
        mlflow_run_name=None,
        checkpointer: Optional[Checkpointer] = None,
    ):
        self._params = {
            "model": model,
            "n_samples": n_samples,
            "log": log,
            "log_path": log_path,
            "file_name": file_name,
            "function_name": function_name,
            "model_directory": model_directory,
            "overwrite": overwrite,
            "human_in_the_loop": human_in_the_loop,
            "bypass_recommended_steps": bypass_recommended_steps,
            "bypass_explain_code": bypass_explain_code,
            "enable_mlflow": enable_mlflow,
            "mlflow_tracking_uri": mlflow_tracking_uri,
            "mlflow_artifact_root": mlflow_artifact_root,
            "mlflow_experiment_name": mlflow_experiment_name,
            "mlflow_run_name": mlflow_run_name,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        """
        Creates the compiled graph for the agent.
        """
        self.response = None
        return make_h2o_ml_agent(**self._params)

    def update_params(self, **kwargs):
        """
        Updates the agent's parameters and rebuilds the compiled graph.
        """
        for k, v in kwargs.items():
            self._params[k] = v
        self._compiled_graph = self._make_compiled_graph()

    async def ainvoke_agent(
        self,
        data_raw: pd.DataFrame,
        user_instructions: str = None,
        target_variable: str = None,
        max_retries=3,
        retry_count=0,
        **kwargs,
    ):
        """
        Asynchronously trains an H2O AutoML model for the provided dataset,
        saving the best model to disk if model_directory or log_path is available.
        """
        response = await self._compiled_graph.ainvoke(
            {
                "messages": [("user", user_instructions)] if user_instructions else [],
                "user_instructions": user_instructions,
                "data_raw": data_raw.to_dict(),
                "target_variable": target_variable,
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        self.response = response
        return None

    def invoke_agent(
        self,
        data_raw: pd.DataFrame,
        user_instructions: str = None,
        target_variable: str = None,
        max_retries=3,
        retry_count=0,
        **kwargs,
    ):
        """
        Synchronously trains an H2O AutoML model for the provided dataset,
        saving the best model to disk if model_directory or log_path is available.
        """
        response = self._compiled_graph.invoke(
            {
                "messages": [("user", user_instructions)] if user_instructions else [],
                "user_instructions": user_instructions,
                "data_raw": data_raw.to_dict(),
                "target_variable": target_variable,
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        self.response = response
        return None

    def invoke_messages(
        self,
        messages: Sequence[BaseMessage],
        data_raw: pd.DataFrame,
        target_variable: str = None,
        max_retries=3,
        retry_count=0,
        **kwargs,
    ):
        """
        Invokes the agent with an explicit message list (preferred for supervisors/teams).
        """
        user_instructions = kwargs.pop("user_instructions", None)
        if user_instructions is None:
            user_instructions = get_last_user_message_content(messages)
        response = self._compiled_graph.invoke(
            {
                "messages": messages,
                "user_instructions": user_instructions,
                "data_raw": data_raw.to_dict(),
                "target_variable": target_variable,
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        self.response = response
        return None

    async def ainvoke_messages(
        self,
        messages: Sequence[BaseMessage],
        data_raw: pd.DataFrame,
        target_variable: str = None,
        max_retries=3,
        retry_count=0,
        **kwargs,
    ):
        """
        Async version of invoke_messages.
        """
        user_instructions = kwargs.pop("user_instructions", None)
        if user_instructions is None:
            user_instructions = get_last_user_message_content(messages)
        response = await self._compiled_graph.ainvoke(
            {
                "messages": messages,
                "user_instructions": user_instructions,
                "data_raw": data_raw.to_dict(),
                "target_variable": target_variable,
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        self.response = response
        return None

    def get_leaderboard(self):
        """Returns the H2O AutoML leaderboard as a DataFrame."""
        if self.response and "leaderboard" in self.response:
            return pd.DataFrame(self.response["leaderboard"])
        return None

    def get_best_model_id(self):
        """Returns the best model id from the AutoML run."""
        if self.response and "best_model_id" in self.response:
            return self.response["best_model_id"]
        return None

    def get_model_path(self):
        """Returns the file path to the saved best model, or None if not saved."""
        if self.response and "model_path" in self.response:
            return self.response["model_path"]
        return None

    def get_data_raw(self):
        """Retrieves the raw data as a DataFrame from the response."""
        if self.response and "data_raw" in self.response:
            return pd.DataFrame(self.response["data_raw"])
        return None

    def get_h2o_train_function(self, markdown=False):
        """Retrieves the H2O AutoML function code generated by the agent."""
        if self.response and "h2o_train_function" in self.response:
            code = self.response["h2o_train_function"]
            if markdown:
                return Markdown(f"```python\n{code}\n```")
            return code
        return None

    def get_recommended_ml_steps(self, markdown=False):
        """Retrieves recommended ML steps from the agent's response."""
        if self.response and "recommended_steps" in self.response:
            steps = self.response["recommended_steps"]
            if markdown:
                return Markdown(steps)
            return steps
        return None

    def get_workflow_summary(self, markdown=False):
        """
        Retrieves the agent's workflow summary, if logging is enabled.
        """
        if self.response and self.response.get("messages"):
            summary = get_generic_summary(
                json.loads(self.response.get("messages")[-1].content)
            )
            if markdown:
                return Markdown(summary)
            else:
                return summary

    def get_log_summary(self, markdown=False):
        """
        Logs a summary of the agent's operations, if logging is enabled.
        """
        if self.response:
            if self.response.get("h2o_train_function_path"):
                log_details = f"""
## H2O Machine Learning Agent Log Summary:

Function Path: {self.response.get("h2o_train_function_path")}

Function Name: {self.response.get("h2o_train_function_name")}

Best Model ID: {self.get_best_model_id()}

Model Path: {self.get_model_path()}
                """
                if markdown:
                    return Markdown(log_details)
                else:
                    return log_details


def make_h2o_ml_agent(
    model,
    n_samples=30,
    log=False,
    log_path=None,
    file_name="h2o_automl.py",
    function_name="h2o_automl",
    model_directory=None,
    overwrite=True,
    human_in_the_loop=False,
    bypass_recommended_steps=False,
    bypass_explain_code=False,
    enable_mlflow=False,
    mlflow_tracking_uri=None,
    mlflow_artifact_root=None,
    mlflow_experiment_name="H2O AutoML",
    mlflow_run_name=None,
    checkpointer=None,
):
    """
    Creates a machine learning agent that uses H2O for AutoML.
    The agent will:
      1. Optionally recommend ML steps,
      2. Creates Python code that sets up H2OAutoML,
      3. Executes that code (optionally saving the best model to disk),
      4. Fixes errors if needed,
      5. Optionally explains the code.

    model_directory: Directory to save the model.
                    If None, defaults to log_path.
                    If both are None, skip saving.
    """

    llm = model

    # Handle logging directory
    if log:
        if log_path is None:
            log_path = "logs/"
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    # Check if H2O is installed
    try:
        import h2o
        from h2o.automl import H2OAutoML
    except ImportError as e:
        raise ImportError(
            "The 'h2o' library is not installed. Please install it using pip:\n\n"
            "    pip install h2o\n\n"
            "Visit https://docs.h2o.ai/h2o/latest-stable/h2o-docs/downloading.html for details."
        ) from e

    if human_in_the_loop:
        if checkpointer is None:
            print(
                "Human in the loop is enabled. A checkpointer is required. Setting to MemorySaver()."
            )
            checkpointer = MemorySaver()

    # Define GraphState
    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        recommended_steps: str
        data_raw: dict
        leaderboard: dict
        best_model_id: str
        model_path: str
        model_results: dict
        target_variable: str
        all_datasets_summary: str
        h2o_train_function: str
        h2o_train_function_path: str
        h2o_train_file_name: str
        h2o_train_function_name: str
        h2o_train_error: str
        h2o_train_error_log_path: str
        max_retries: int
        retry_count: int

    # 1) Recommend ML steps (optional)
    def recommend_ml_steps(state: GraphState):
        print(format_agent_name(AGENT_NAME))
        print("    * RECOMMEND MACHINE LEARNING STEPS")

        recommend_steps_prompt = PromptTemplate(
            template="""
                You are an AutoML Expert using H2O. 
                
                We have the following dataset summary, user instructions, and H2O AutoML documentation:

                User instructions:
                    {user_instructions}

                Data Summary:
                    {all_datasets_summary}
                    
                H2O AutoML Documentation:
                    {h2o_automl_documentation}

                Please recommend a short list of steps or considerations for performing H2OAutoML on this data. Specifically focus on maximizing model accuracy while remaining flexible to user instructions and the dataset.
                
                - Recommend any paramters and values that might improve performance (predictive accuracy).
                - Recommend the Loss Function, Stopping Criteria, and other advanced parameters.
                - Use the H2O AutoML documentation to your advantage.
                - Exclude deep learning algorithms since these are typically low performance.
                
                Avoid these:
                
                - Do not perform data cleaning or feature engineering here. We will handle that separately.
                - Do not limit memory size or CPU usage unless the user specifies it. 
                
                Return as a numbered list. You can return short code snippets to demonstrate actions. But do not return a fully coded solution. The H2O AutoML code will be generated separately by a Coding Agent.
            """,
            input_variables=[
                "user_instructions",
                "all_datasets_summary",
                "h2o_automl_documentation",
            ],
        )

        data_raw = state.get("data_raw")
        df = pd.DataFrame.from_dict(data_raw)
        all_datasets_summary = get_dataframe_summary(
            [df], n_sample=min(n_samples, 5), skip_stats=True
        )
        all_datasets_summary_str = "\n\n".join(
            [s[:MAX_SUMMARY_CHARS] for s in all_datasets_summary]
        )

        steps_agent = recommend_steps_prompt | llm
        recommended_steps = steps_agent.invoke(
            {
                "user_instructions": state.get("user_instructions"),
                "all_datasets_summary": all_datasets_summary_str,
                "h2o_automl_documentation": H2O_AUTOML_DOCUMENTATION,
            }
        )

        return {
            "recommended_steps": format_recommended_steps(
                recommended_steps.content.strip(), heading="# Recommended ML Steps:"
            ),
            "all_datasets_summary": all_datasets_summary_str,
        }

    # 2) Create code
    def create_h2o_code(state: GraphState):
        if bypass_recommended_steps:
            print(format_agent_name(AGENT_NAME))

            data_raw = state.get("data_raw")
            df = pd.DataFrame.from_dict(data_raw)
            all_datasets_summary = get_dataframe_summary(
                [df], n_sample=min(n_samples, 5), skip_stats=True
            )
            all_datasets_summary_str = "\n\n".join(
                [s[:MAX_SUMMARY_CHARS] for s in all_datasets_summary]
            )
        else:
            all_datasets_summary_str = state.get("all_datasets_summary")

        steps_for_prompt = state.get("recommended_steps") or DEFAULT_ML_STEPS

        print("    * CREATE H2O AUTOML CODE")

        code_prompt = PromptTemplate(
            template="""
            You are an H2O AutoML agent. Create a Python function named {function_name}(data_raw)
            that runs H2OAutoML on the provided data with a focus on maximizing model accuracy and 
            incorporating user instructions for flexibility.
            
            Do not perform substantial data cleaning or feature engineering here. We will handle that separately.

            We have two variables for deciding where to save the model:
            model_directory = {model_directory} 
            log_path = {log_path}
            
            IMPORTANT: MLflow Parameters if the user wants to enable MLflow with H2O AutoML:
                enable_mlflow: {enable_mlflow}
                mlflow_tracking_uri: {mlflow_tracking_uri}
                mlflow_experiment_name: {mlflow_experiment_name}
                mlflow_run_name: {mlflow_run_name}


            Additional Requirements:
            - Convert `data_raw` (pandas DataFrame) into an H2OFrame.
            - Identify the target variable from {target_variable} (if provided).
            - Start H2O if not already started.
            - Use Recommended Steps to guide any advanced parameters (e.g., cross-validation folds, 
            balancing classes, extended training time, stacking) that might improve performance.
            - If the user does not specify anything special, use H2OAutoML defaults (including stacked ensembles).
            - Include safe defaults: max_runtime_secs (e.g., 30) and max_models (e.g., 20) to avoid runaway jobs; exclude deep learning by default.
            - Focus on maximizing accuracy (or the most relevant metric if it's not classification) 
            while remaining flexible to user instructions.
            - Return a dict with keys: leaderboard, best_model_id, model_path, and model_results.
            - If enable_mlfow is True, log the top metrics and save the model as an artifact. (See example function)
            - IMPORTANT: if enable_mlflow is True, make sure to set enable_mlflow to True in the function definition.
            - Function signature must be valid Python: place **kwargs at the end of the parameter list.
            
            Initial User Instructions (Disregard any instructions that are unrelated to modeling):
                {user_instructions}
            
            Recommended Steps:
                {recommended_steps}

            Data summary for reference:
                {all_datasets_summary}

        Return only code in ```python``` with a single function definition. Use this as an example starting template:
        ```python
            def {function_name}(
                data_raw,
                target: str,
                max_runtime_secs: int,
                exclude_algos: list[str],
                balance_classes: bool,
                nfolds: int,
                seed: int,
                max_models: int,
                stopping_metric: str,
                stopping_tolerance: float,
                stopping_rounds: int,
                sort_metric: str ,
                model_directory: str | None = None,
                log_path: str | None = None,
                enable_mlflow: bool, # If use has specified to enable MLflow, make sure to make this True              
                mlflow_tracking_uri: str | None, 
                mlflow_experiment_name: str,
                mlflow_run_name: str,
                **kwargs # Additional parameters for H2OAutoML (feel free to add these based on user instructions and recommended steps)
            ):

                import h2o
                from h2o.automl import H2OAutoML
                import pandas as pd
                import json

                # Optional MLflow usage
                if enable_mlflow:
                    import mlflow
                    if mlflow_tracking_uri:
                        mlflow.set_tracking_uri(mlflow_tracking_uri)
                    mlflow.set_experiment(mlflow_experiment_name)
                    run_context = mlflow.start_run(run_name=mlflow_run_name)
                else:
                    # Dummy context manager to skip MLflow if not enabled
                    from contextlib import nullcontext
                    run_context = nullcontext()

                exclude_algos = exclude_algos or ["DeepLearning"]  # default if not provided

                # Convert data to DataFrame
                df = pd.DataFrame(data_raw)

                with run_context as run:
                    # If using MLflow, track run ID
                    run_id = None
                    if enable_mlflow and run is not None:
                        run_id = run.info.run_id
                        import mlflow
                        

                    # Initialize H2O
                    h2o.init()

                    # Create H2OFrame
                    data_h2o = h2o.H2OFrame(df)

                    # Setup AutoML
                    aml = H2OAutoML(
                        max_runtime_secs=max_runtime_secs,
                        exclude_algos=exclude_algos,
                        balance_classes=balance_classes,
                        nfolds=nfolds,
                        seed=seed,
                        max_models=max_models,
                        stopping_metric=stopping_metric,
                        stopping_tolerance=stopping_tolerance,
                        stopping_rounds=stopping_rounds,
                        sort_metric=sort_metric,
                        **kwargs
                    )

                    # Train
                    x = [col for col in data_h2o.columns if col != target]
                    aml.train(x=x, y=target, training_frame=data_h2o)

                    # Save model if we have a directory/log path
                    if model_directory is None and log_path is None:
                        model_path = None
                    else:
                        path_to_save = model_directory if model_directory else log_path
                        model_path = h2o.save_model(model=aml.leader, path=path_to_save, force=True)

                    # Leaderboard (DataFrame -> dict)
                    leaderboard_df = pd.DataFrame(aml.leaderboard)
                    leaderboard_dict = leaderboard_df.to_dict()

                    # Gather top-model metrics from the first row
                    top_metrics = leaderboard_df.iloc[0].to_dict()  

                    # Construct model_results
                    model_results = dict(
                        model_flavor= "H2O AutoML",
                        model_path= model_path,
                        best_model_id= aml.leader.model_id,
                        metrics= top_metrics  # all metrics from the top row
                    )

                    # IMPORTANT: Log these to MLflow if enabled
                    if enable_mlflow and run is not None:
                        
                        # Log the top metrics if numeric
                        numeric_metrics = {{k: v for k, v in top_metrics.items() if isinstance(v, (int, float))}}
                        mlflow.log_metrics(numeric_metrics)

                        # Log artifact if we saved the model
                        # MLflow 3.x prefers `name=` (artifact_path is deprecated)
                        try:
                            mlflow.h2o.log_model(aml.leader, name="model")
                        except TypeError:
                            mlflow.h2o.log_model(aml.leader, artifact_path="model")
                        
                        # Log the leaderboard
                        mlflow.log_table(leaderboard_dict, "leaderboard.json")
                        
                        # Log these parameters (if specified)
                        mlflow.log_params(dict(
                            target= target,
                            max_runtime_secs= max_runtime_secs,
                            exclude_algos= str(exclude_algos),
                            balance_classes= balance_classes,
                            nfolds= nfolds,
                            seed= seed,
                            max_models= max_models,
                            stopping_metric= stopping_metric,
                            stopping_tolerance= stopping_tolerance,
                            stopping_rounds= stopping_rounds,
                            sort_metric= sort_metric,
                            model_directory= model_directory,
                            log_path= log_path
                        ))

                    # Build the output
                    output = dict(
                        leaderboard= leaderboard_dict,
                        best_model_id= aml.leader.model_id,
                        model_path= model_path,
                        model_results= model_results,
                        mlflow_run_id= run_id
                    )

                return output
            ```
            
            Avoid these errors:
            
            - WARNING mlflow.models.model: Model logged without a signature and input example. Please set `input_example` parameter when logging the model to auto infer the model signature.
            
            - 'list' object has no attribute 'tolist'
            
            - with h2o.utils.threading.local_context(polars_enabled=True, datatable_enabled=True):  pandas_df = h2o_df.as_data_frame() # Convert to pandas DataFrame using pd.DataFrame(h2o_df)
            
            - dtype is only supported for one column frames
            
            - h2o.is_running() module 'h2o' has no attribute 'is_running'. Solution: just do h2o.init() and it will check if H2O is running.

            Critical requirements (prevents common AutoML failures):
            - If the target is binary (2 unique values) or non-numeric, make sure H2O treats it as classification by converting to a factor:
              `data_h2o[target] = data_h2o[target].asfactor()` after creating the H2OFrame.
            - Only use classification metrics (AUC/logloss/AUCPR) when the target is categorical; use regression metrics (RMSE/MAE) only for numeric targets.
            - Do not set extremely small stopping_tolerance; prefer H2O defaults unless the user explicitly requests tuning.
            
            
            """,
            input_variables=[
                "user_instructions",
                "function_name",
                "target_variable",
                "recommended_steps",
                "all_datasets_summary",
                "model_directory",
                "log_path",
                "enable_mlflow",
                "mlflow_tracking_uri",
                "mlflow_experiment_name",
                "mlflow_run_name",
            ],
        )

        recommended_steps = steps_for_prompt
        h2o_code_agent = code_prompt | llm | PythonOutputParser()

        resp = h2o_code_agent.invoke(
            {
                "user_instructions": state.get("user_instructions"),
                "function_name": function_name,
                "target_variable": state.get("target_variable"),
                "recommended_steps": recommended_steps,
                "all_datasets_summary": all_datasets_summary_str,
                "model_directory": model_directory,
                "log_path": log_path,
                "enable_mlflow": enable_mlflow,
                "mlflow_tracking_uri": mlflow_tracking_uri,
                "mlflow_experiment_name": mlflow_experiment_name,
                "mlflow_run_name": mlflow_run_name,
            }
        )

        resp = relocate_imports_inside_function(resp)
        resp = add_comments_to_top(resp, agent_name=AGENT_NAME)

        # Log the code snippet if requested
        file_path, f_name = log_ai_function(
            response=resp,
            file_name=file_name,
            log=log,
            log_path=log_path,
            overwrite=overwrite,
        )

        return {
            "h2o_train_function": resp,
            "h2o_train_function_path": file_path,
            "h2o_train_file_name": f_name,
            "h2o_train_function_name": function_name,
        }

    # Human Review
    prompt_text_human_review = "Are the following Machine Learning instructions correct? (Answer 'yes' or provide modifications)\n{steps}"

    if not bypass_explain_code:

        def human_review(
            state: GraphState,
        ) -> Command[Literal["recommend_ml_steps", "report_agent_outputs"]]:
            return node_func_human_review(
                state=state,
                prompt_text=prompt_text_human_review,
                yes_goto="report_agent_outputs",
                no_goto="recommend_ml_steps",
                user_instructions_key="user_instructions",
                recommended_steps_key="recommended_steps",
                code_snippet_key="h2o_train_function",
            )
    else:

        def human_review(
            state: GraphState,
        ) -> Command[Literal["recommend_ml_steps", "__end__"]]:
            return node_func_human_review(
                state=state,
                prompt_text=prompt_text_human_review,
                yes_goto="__end__",
                no_goto="recommend_ml_steps",
                user_instructions_key="user_instructions",
                recommended_steps_key="recommended_steps",
                code_snippet_key="h2o_train_function",
            )

    # 3) Execute code
    def execute_h2o_code(state):
        user_instructions = state.get("user_instructions") or get_last_user_message_content(
            state.get("messages", [])
        )
        target_col = state.get("target_variable")
        target_col = str(target_col).strip() if isinstance(target_col, str) else ""

        def _infer_target(df: pd.DataFrame) -> str:
            if target_col and target_col in df.columns:
                return target_col
            # Heuristic for churn-like requests
            if isinstance(user_instructions, str) and "churn" in user_instructions.lower():
                if "Churn" in df.columns:
                    return "Churn"
                if "churn" in df.columns:
                    return "churn"
            # Common target names
            for cand in ("target", "Target", "label", "Label", "y", "Y"):
                if cand in df.columns:
                    return cand
            # Fall back to last column
            try:
                return str(df.columns[-1])
            except Exception:
                return ""

        target_col_final = ""

        def _preprocess(data_dict):
            nonlocal target_col_final
            df = pd.DataFrame.from_dict(data_dict)
            target_col_final = _infer_target(df)
            if target_col_final:
                if target_col_final not in df.columns:
                    raise ValueError(
                        f"Target variable '{target_col_final}' not found in data."
                    )
                non_null = df[target_col_final].notnull().sum()
                if non_null == 0:
                    raise ValueError(
                        f"Target variable '{target_col_final}' has no non-null values."
                    )
                nunique = df[target_col_final].dropna().nunique()
                if nunique < 2:
                    vc = (
                        df[target_col_final]
                        .value_counts(dropna=False)
                        .head(5)
                        .to_dict()
                    )
                    raise ValueError(
                        f"Target variable '{target_col_final}' has <2 classes (nunique={nunique}). "
                        f"Value counts (top 5): {vc}"
                    )
                # If the target looks binary/categorical, coerce it to a categorical dtype so H2O
                # treats this as classification (avoids regression + AUC/logloss metric errors).
                try:
                    is_bool = pd.api.types.is_bool_dtype(df[target_col_final])
                    is_object = pd.api.types.is_object_dtype(df[target_col_final])
                    is_category = pd.api.types.is_categorical_dtype(df[target_col_final])
                    is_numeric = pd.api.types.is_numeric_dtype(df[target_col_final])
                    if is_bool or is_object or is_category or (is_numeric and nunique == 2):
                        # Normalize numeric 0/1 or 1/2, etc. to strings then category for stable factor inference.
                        df[target_col_final] = df[target_col_final].astype(str).astype("category")
                except Exception:
                    pass
            return df

        result = node_func_execute_agent_code_on_data(
            state=state,
            data_key="data_raw",
            code_snippet_key="h2o_train_function",
            result_key="h2o_train_result",
            error_key="h2o_train_error",
            agent_function_name=state.get("h2o_train_function_name"),
            pre_processing=_preprocess,
            post_processing=lambda x: x,
            error_message_prefix="Error occurred during H2O AutoML: ",
        )
        if target_col_final:
            result["target_variable"] = target_col_final

        # If no error, extract leaderboard, best_model_id, and model_path
        if not result["h2o_train_error"]:
            if result["h2o_train_result"] and isinstance(
                result["h2o_train_result"], dict
            ):
                lb = result["h2o_train_result"].get("leaderboard", {})
                best_id = result["h2o_train_result"].get("best_model_id", None)
                mpath = result["h2o_train_result"].get("model_path", None)
                model_results = result["h2o_train_result"].get("model_results", {})

                result["leaderboard"] = lb
                result["best_model_id"] = best_id
                result["model_path"] = mpath
                result["model_results"] = model_results

                # Robust MLflow logging: do it here so we don't rely on the LLM-generated code
                # to correctly log models/params (and so `predict using mlflow` can find `model/`).
                if enable_mlflow:
                    existing_run_id = (
                        result["h2o_train_result"].get("mlflow_run_id")
                        if isinstance(result.get("h2o_train_result"), dict)
                        else None
                    )
                    try:
                        import mlflow

                        if mlflow_tracking_uri:
                            mlflow.set_tracking_uri(mlflow_tracking_uri)
                        exp_name = mlflow_experiment_name or "H2O AutoML"
                        # Best-effort: if an artifact root is configured, ensure the experiment
                        # exists with that artifact location. Existing experiments keep their
                        # original artifact location in MLflow.
                        try:
                            from pathlib import Path
                            from mlflow.tracking import MlflowClient
                            import re

                            if isinstance(mlflow_artifact_root, str) and mlflow_artifact_root.strip():
                                root = Path(mlflow_artifact_root).expanduser().resolve()
                                root.mkdir(parents=True, exist_ok=True)
                                safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(exp_name)).strip("_")
                                safe = safe or "H2O_AutoML"
                                artifact_location = (root / safe).as_uri()
                                client = MlflowClient(tracking_uri=mlflow_tracking_uri)
                                exp = client.get_experiment_by_name(str(exp_name))
                                if exp is None:
                                    client.create_experiment(
                                        name=str(exp_name),
                                        artifact_location=artifact_location,
                                    )
                        except Exception:
                            pass

                        mlflow.set_experiment(exp_name)

                        def _sanitize_metric_name(name: str) -> str:
                            n = str(name or "").strip()
                            n = n.replace(" ", "_").replace("/", "_").replace("-", "_")
                            return n

                        def _extract_metrics_from_leaderboard(lb_dict: dict) -> dict:
                            if not lb_dict:
                                return {}
                            # Supported shapes:
                            # 1) MLflow table json: {"columns": [...], "data": [[...], ...]}
                            # 2) pandas.DataFrame.to_dict() outputs (dict-of-dicts or dict-of-lists)
                            # 3) list-of-records
                            cols = lb_dict.get("columns") if isinstance(lb_dict, dict) else None
                            rows = lb_dict.get("data") if isinstance(lb_dict, dict) else None
                            if isinstance(cols, list) and isinstance(rows, list) and rows:
                                first = rows[0]
                                if isinstance(first, list) and len(first) == len(cols):
                                    out = {}
                                    for c, v in zip(cols, first):
                                        if c in (None, "", "model_id"):
                                            continue
                                        try:
                                            fv = float(v)
                                        except Exception:
                                            continue
                                        if fv != fv:  # NaN
                                            continue
                                        out[_sanitize_metric_name(str(c))] = fv
                                    return out

                            try:
                                import pandas as pd

                                df = pd.DataFrame(lb_dict)
                                if df is None or df.empty:
                                    return {}
                                row = df.iloc[0].to_dict()
                                out = {}
                                for c, v in row.items():
                                    if c in (None, "", "model_id"):
                                        continue
                                    try:
                                        fv = float(v)
                                    except Exception:
                                        continue
                                    if fv != fv:  # NaN
                                        continue
                                    out[_sanitize_metric_name(str(c))] = fv
                                return out
                            except Exception:
                                return {}

                        # Only start our own run if the generated code didn't already do it.
                        run_ctx = (
                            mlflow.start_run(run_id=str(existing_run_id))
                            if isinstance(existing_run_id, str) and existing_run_id.strip()
                            else mlflow.start_run(run_name=mlflow_run_name)
                        )
                        with run_ctx as run:
                            run_id = run.info.run_id if run is not None else None
                            if isinstance(run_id, str) and run_id:
                                result["mlflow_run_id"] = run_id
                                if isinstance(result.get("h2o_train_result"), dict):
                                    result["h2o_train_result"]["mlflow_run_id"] = run_id

                            # Tags/params
                            try:
                                mlflow.set_tag("agent", AGENT_NAME)
                                if isinstance(user_instructions, str) and user_instructions.strip():
                                    mlflow.set_tag("user_instructions", user_instructions.strip()[:5000])
                            except Exception:
                                pass
                            try:
                                mlflow.log_params(
                                    {
                                        "target_variable": target_col_final or target_col or "",
                                        "function_name": state.get("h2o_train_function_name") or "",
                                        "max_runtime_secs": 30,
                                        "best_model_id": best_id or "",
                                    }
                                )
                            except Exception:
                                pass

                            # Leaderboard + metrics
                            try:
                                if isinstance(lb, dict) and lb:
                                    mlflow.log_table(lb, "leaderboard.json")
                            except Exception:
                                try:
                                    if isinstance(lb, dict) and lb:
                                        mlflow.log_dict(lb, "leaderboard.json")
                                except Exception:
                                    pass
                            try:
                                metrics = _extract_metrics_from_leaderboard(lb)
                                if metrics:
                                    mlflow.log_metrics(metrics)
                            except Exception:
                                pass

                            # Log leader model (H2O)
                            try:
                                import h2o

                                h2o.init()
                                if isinstance(best_id, str) and best_id.strip():
                                    leader_model = h2o.get_model(best_id.strip())
                                    try:
                                        logged_model = mlflow.h2o.log_model(
                                            leader_model, name="model"
                                        )
                                    except TypeError:
                                        logged_model = mlflow.h2o.log_model(
                                            leader_model, artifact_path="model"
                                        )

                                    # Store a stable model URI for downstream scoring.
                                    model_uri = (
                                        getattr(logged_model, "model_uri", None)
                                        if logged_model is not None
                                        else None
                                    )
                                    model_uri = (
                                        model_uri
                                        if isinstance(model_uri, str) and model_uri.strip()
                                        else f"runs:/{run_id}/model"
                                    )
                                    result["mlflow_model_uri"] = model_uri
                                    if isinstance(result.get("h2o_train_result"), dict):
                                        result["h2o_train_result"]["mlflow_model_uri"] = model_uri

                                    # Best-effort: capture a lightweight listing of model files
                                    # so users can see that the model is persisted even if MLflow
                                    # UI separates "Logged models" from run artifacts.
                                    model_files: list[str] = []
                                    try:
                                        from mlflow.tracking import MlflowClient

                                        client = MlflowClient()
                                        model_files = [
                                            getattr(a, "path", None)
                                            for a in client.list_artifacts(run_id, path="model")
                                        ]
                                        model_files = [p for p in model_files if isinstance(p, str)]
                                    except Exception:
                                        model_files = []

                                    model_art = {
                                        "run_id": run_id,
                                        "model_uri": model_uri,
                                        "best_model_id": best_id,
                                        "model_files": model_files,
                                    }
                                    result["mlflow_model"] = model_art
                                    if isinstance(result.get("h2o_train_result"), dict):
                                        result["h2o_train_result"]["mlflow_model"] = model_art

                                    try:
                                        mlflow.log_dict(model_art, artifact_file="model_info.json")
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                    except Exception:
                        # Never fail training because MLflow logging failed.
                        pass

        if result.get("h2o_train_error") and log:
            error_log_path = log_ai_error(
                error_message=result["h2o_train_error"],
                file_name=f"{file_name}_errors.log",
                log=log,
                log_path=log_path if log_path is not None else LOG_PATH,
                overwrite=False,
            )
            result["h2o_train_error_log_path"] = error_log_path

        return result

    # 4) Fix code if there's an error
    def fix_h2o_code(state: GraphState):
        fix_prompt = """
        You are an H2O AutoML agent. The function {function_name} currently has errors. 
        Please fix it. Return only the corrected function in ```python``` format.
        
        Broken code:
        {code_snippet}

        Last Known Error:
        {error}
        """
        return node_func_fix_agent_code(
            state=state,
            code_snippet_key="h2o_train_function",
            error_key="h2o_train_error",
            llm=llm,
            prompt_template=fix_prompt,
            agent_name=AGENT_NAME,
            file_path=state.get("h2o_train_function_path"),
            function_name=state.get("h2o_train_function_name"),
            log=log,
        )

    # 5) Final reporting node
    def report_agent_outputs(state: GraphState):
        return node_func_report_agent_outputs(
            state=state,
            keys_to_include=[
                "recommended_steps",
                "h2o_train_function",
                "h2o_train_function_path",
                "h2o_train_function_name",
                "h2o_train_error",
                "h2o_train_error_log_path",
                "model_path",
                "best_model_id",
            ],
            result_key="messages",
            role=AGENT_NAME,
            custom_title="H2O Machine Learning Agent Outputs",
        )

    node_functions = {
        "recommend_ml_steps": recommend_ml_steps,
        "human_review": human_review,
        "create_h2o_code": create_h2o_code,
        "execute_h2o_code": execute_h2o_code,
        "fix_h2o_code": fix_h2o_code,
        "report_agent_outputs": report_agent_outputs,
    }

    app = create_coding_agent_graph(
        GraphState=GraphState,
        node_functions=node_functions,
        recommended_steps_node_name="recommend_ml_steps",
        create_code_node_name="create_h2o_code",
        execute_code_node_name="execute_h2o_code",
        fix_code_node_name="fix_h2o_code",
        explain_code_node_name="report_agent_outputs",
        error_key="h2o_train_error",
        max_retries_key="max_retries",
        retry_count_key="retry_count",
        human_in_the_loop=human_in_the_loop,
        human_review_node_name="human_review",
        checkpointer=checkpointer,
        bypass_recommended_steps=bypass_recommended_steps,
        bypass_explain_code=bypass_explain_code,
        agent_name=AGENT_NAME,
    )

    return app
