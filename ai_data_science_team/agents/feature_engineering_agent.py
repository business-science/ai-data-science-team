# BUSINESS SCIENCE UNIVERSITY
# AI DATA SCIENCE TEAM
# ***
# * Agents: Feature Engineering Agent

# Libraries
from typing_extensions import TypedDict, Annotated, Sequence, Literal
import operator

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage

from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

import os
import json
import pandas as pd

from IPython.display import Markdown

from ai_data_science_team.templates import (
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
from ai_data_science_team.utils.sandbox import run_code_sandboxed_subprocess
from ai_data_science_team.utils.messages import get_last_user_message_content

# Setup
AGENT_NAME = "feature_engineering_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")

# Class


class FeatureEngineeringAgent(BaseAgent):
    """
    Creates a feature engineering agent that can process datasets based on user-defined instructions or
    default feature engineering steps. The agent generates a Python function to engineer features, executes it,
    and logs the process, including code and errors. It is designed to facilitate reproducible and
    customizable feature engineering workflows.

    The agent can perform the following default feature engineering steps unless instructed otherwise:
    - Convert features to appropriate data types
    - Remove features that have unique values for each row
    - Remove constant features
    - Encode high-cardinality categoricals (threshold <= 5% of dataset) as 'other'
    - One-hot-encode categorical variables
    - Convert booleans to integer (1/0)
    - Create datetime-based features (if applicable)
    - Handle target variable encoding if specified
    - Any user-provided instructions to add, remove, or modify steps

    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model used to generate the feature engineering function.
    n_samples : int, optional
        Number of samples used when summarizing the dataset. Defaults to 30.
    log : bool, optional
        Whether to log the generated code and errors. Defaults to False.
    log_path : str, optional
        Directory path for storing log files. Defaults to None.
    file_name : str, optional
        Name of the file for saving the generated response. Defaults to "feature_engineer.py".
    function_name : str, optional
        Name of the function for data visualization. Defaults to "feature_engineer".
    overwrite : bool, optional
        Whether to overwrite the log file if it exists. If False, a unique file name is created. Defaults to True.
    human_in_the_loop : bool, optional
        Enables user review of feature engineering instructions. Defaults to False.
    bypass_recommended_steps : bool, optional
        If True, skips the default recommended steps. Defaults to False.
    bypass_explain_code : bool, optional
        If True, skips the step that provides code explanations. Defaults to False.
    checkpointer : Checkpointer, optional
        Checkpointer to save and load the agent's state. Defaults to None.

    Methods
    -------
    update_params(**kwargs)
        Updates the agent's parameters and rebuilds the compiled state graph.
    ainvoke_agent(
        user_instructions: str,
        data_raw: pd.DataFrame,
        target_variable: str = None,
        max_retries=3,
        retry_count=0
    )
        Engineers features from the provided dataset asynchronously based on user instructions.
    invoke_agent(
        user_instructions: str,
        data_raw: pd.DataFrame,
        target_variable: str = None,
        max_retries=3,
        retry_count=0
    )
        Engineers features from the provided dataset synchronously based on user instructions.
    get_workflow_summary()
        Retrieves a summary of the agent's workflow.
    get_log_summary()
        Retrieves a summary of logged operations if logging is enabled.
    get_data_engineered()
        Retrieves the feature-engineered dataset as a pandas DataFrame.
    get_data_raw()
        Retrieves the raw dataset as a pandas DataFrame.
    get_feature_engineer_function()
        Retrieves the generated Python function used for feature engineering.
    get_recommended_feature_engineering_steps()
        Retrieves the agent's recommended feature engineering steps.
    get_response()
        Returns the response from the agent as a dictionary.
    show()
        Displays the agent's mermaid diagram.

    Examples
    --------
    ```python
    import pandas as pd
    from langchain_openai import ChatOpenAI
    from ai_data_science_team.agents import FeatureEngineeringAgent

    llm = ChatOpenAI(model="gpt-4o-mini")

    feature_agent = FeatureEngineeringAgent(
        model=llm,
        n_samples=30,
        log=True,
        log_path="logs",
        human_in_the_loop=True
    )

    df = pd.read_csv("https://raw.githubusercontent.com/business-science/ai-data-science-team/refs/heads/master/data/churn_data.csv")

    feature_agent.invoke_agent(
        user_instructions="Also encode the 'PaymentMethod' column with one-hot encoding.",
        data_raw=df,
        target_variable="Churn",
        max_retries=3,
        retry_count=0
    )

    engineered_data = feature_agent.get_data_engineered()
    response = feature_agent.get_response()
    ```

    Returns
    -------
    FeatureEngineeringAgent : langchain.graphs.CompiledStateGraph
        A feature engineering agent implemented as a compiled state graph.
    """

    def __init__(
        self,
        model,
        n_samples=30,
        log=False,
        log_path=None,
        file_name="feature_engineer.py",
        function_name="feature_engineer",
        overwrite=True,
        human_in_the_loop=False,
        bypass_recommended_steps=False,
        bypass_explain_code=False,
        checkpointer=None,
    ):
        self._params = {
            "model": model,
            "n_samples": n_samples,
            "log": log,
            "log_path": log_path,
            "file_name": file_name,
            "function_name": function_name,
            "overwrite": overwrite,
            "human_in_the_loop": human_in_the_loop,
            "bypass_recommended_steps": bypass_recommended_steps,
            "bypass_explain_code": bypass_explain_code,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        """
        Create the compiled graph for the feature engineering agent.
        Running this method will reset the response to None.
        """
        self.response = None
        return make_feature_engineering_agent(**self._params)

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
        Asynchronously engineers features for the provided dataset.
        The response is stored in the 'response' attribute.

        Parameters
        ----------
        data_raw : pd.DataFrame
            The raw dataset to be processed.
        user_instructions : str, optional
            Instructions for feature engineering.
        target_variable : str, optional
            The name of the target variable (if any).
        max_retries : int
            Maximum retry attempts.
        retry_count : int
            Current retry attempt count.
        **kwargs
            Additional keyword arguments to pass to ainvoke().

        Returns
        -------
        None
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
        Synchronously engineers features for the provided dataset.
        The response is stored in the 'response' attribute.

        Parameters
        ----------
        data_raw : pd.DataFrame
            The raw dataset to be processed.
        user_instructions : str
            Instructions for feature engineering agent.
        target_variable : str, optional
            The name of the target variable (if any).
        max_retries : int
            Maximum retry attempts.
        retry_count : int
            Current retry attempt count.
        **kwargs
            Additional keyword arguments to pass to invoke().

        Returns
        -------
        None
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
            if self.response.get("feature_engineer_function_path"):
                log_details = f"""
## Featuring Engineering Agent Log Summary:

Function Path: {self.response.get("feature_engineer_function_path")}

Function Name: {self.response.get("feature_engineer_function_name")}
                """
                if markdown:
                    return Markdown(log_details)
                else:
                    return log_details

    def get_data_engineered(self):
        """
        Retrieves the engineered data stored after running invoke/ainvoke.

        Returns
        -------
        pd.DataFrame or None
            The engineered dataset as a pandas DataFrame.
        """
        if self.response and "data_engineered" in self.response:
            return pd.DataFrame(self.response["data_engineered"])
        return None

    def get_data_raw(self):
        """
        Retrieves the raw data.

        Returns
        -------
        pd.DataFrame or None
            The raw dataset as a pandas DataFrame if available.
        """
        if self.response and "data_raw" in self.response:
            return pd.DataFrame(self.response["data_raw"])
        return None

    def get_feature_engineer_function(self, markdown=False):
        """
        Retrieves the feature engineering function generated by the agent.

        Parameters
        ----------
        markdown : bool, optional
            If True, returns the function in Markdown code block format.

        Returns
        -------
        str or None
            The Python function code, or None if unavailable.
        """
        if self.response and "feature_engineer_function" in self.response:
            code = self.response["feature_engineer_function"]
            if markdown:
                return Markdown(f"```python\n{code}\n```")
            return code
        return None

    def get_recommended_feature_engineering_steps(self, markdown=False):
        """
        Retrieves the agent's recommended feature engineering steps.

        Parameters
        ----------
        markdown : bool, optional
            If True, returns the steps in Markdown format.

        Returns
        -------
        str or None
            The recommended steps, or None if not available.
        """
        if self.response and "recommended_steps" in self.response:
            steps = self.response["recommended_steps"]
            if markdown:
                return Markdown(steps)
            return steps
        return None


# * Feature Engineering Agent


def make_feature_engineering_agent(
    model,
    n_samples=30,
    log=False,
    log_path=None,
    file_name="feature_engineer.py",
    function_name="feature_engineer",
    overwrite=True,
    human_in_the_loop=False,
    bypass_recommended_steps=False,
    bypass_explain_code=False,
    checkpointer=None,
):
    """
    Creates a feature engineering agent that can be run on a dataset. The agent applies various feature engineering
    techniques, such as encoding categorical variables, scaling numeric variables, creating interaction terms,
    and generating polynomial features. The agent takes in a dataset and user instructions and outputs a Python
    function for feature engineering. It also logs the code generated and any errors that occur.

    The agent is instructed to apply the following feature engineering techniques:

    - Remove string or categorical features with unique values equal to the size of the dataset
    - Remove constant features with the same value in all rows
    - High cardinality categorical features should be encoded by a threshold <= 5 percent of the dataset, by converting infrequent values to "other"
    - Encoding categorical variables using OneHotEncoding
    - Numeric features should be left untransformed
    - Create datetime-based features if datetime columns are present
    - If a target variable is provided:
        - If a categorical target variable is provided, encode it using LabelEncoding
        - All other target variables should be converted to numeric and unscaled
    - Convert any boolean True/False values to 1/0
    - Return a single data frame containing the transformed features and target variable, if one is provided.
    - Any specific instructions provided by the user

    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model to use to generate code.
    n_samples : int, optional
        The number of data samples to use for generating the feature engineering code. Defaults to 30.
        If you get an error due to maximum tokens, try reducing this number.
        > "This model's maximum context length is 128000 tokens. However, your messages resulted in 333858 tokens. Please reduce the length of the messages."
    log : bool, optional
        Whether or not to log the code generated and any errors that occur.
        Defaults to False.
    log_path : str, optional
        The path to the directory where the log files should be stored. Defaults to "logs/".
    file_name : str, optional
        The name of the file to save the log to. Defaults to "feature_engineer.py".
    function_name : str, optional
        The name of the function that will be generated. Defaults to "feature_engineer".
    overwrite : bool, optional
        Whether or not to overwrite the log file if it already exists. If False, a unique file name will be created.
        Defaults to True.
    human_in_the_loop : bool, optional
        Whether or not to use human in the loop. If True, adds an interput and human in the loop step that asks the user to review the feature engineering instructions. Defaults to False.
    bypass_recommended_steps : bool, optional
        Bypass the recommendation step, by default False
    bypass_explain_code : bool, optional
        Bypass the code explanation step, by default False.
    checkpointer : Checkpointer, optional
        Checkpointer to save and load the agent's state. Defaults to None.

    Examples
    -------
    ``` python
    import pandas as pd
    from langchain_openai import ChatOpenAI
    from ai_data_science_team.agents import feature_engineering_agent

    llm = ChatOpenAI(model="gpt-4o-mini")

    feature_engineering_agent = make_feature_engineering_agent(llm)

    df = pd.read_csv("https://raw.githubusercontent.com/business-science/ai-data-science-team/refs/heads/master/data/churn_data.csv")

    response = feature_engineering_agent.invoke({
        "user_instructions": None,
        "target_variable": "Churn",
        "data_raw": df.to_dict(),
        "max_retries": 3,
        "retry_count": 0
    })

    pd.DataFrame(response['data_engineered'])
    ```

    Returns
    -------
    app : langchain.graphs.CompiledStateGraph
        The feature engineering agent as a state graph.
    """
    llm = model

    if human_in_the_loop:
        if checkpointer is None:
            print(
                "Human in the loop is enabled. A checkpointer is required. Setting to MemorySaver()."
            )
            checkpointer = MemorySaver()

    # Human in th loop requires recommended steps
    if bypass_recommended_steps and human_in_the_loop:
        bypass_recommended_steps = False
        print("Bypass recommended steps set to False to enable human in the loop.")

    # Setup Log Directory
    if log:
        if log_path is None:
            log_path = "logs/"
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    # Defaults and helpers
    MAX_SUMMARY_COLUMNS = 30

    DEFAULT_FEATURE_STEPS = format_recommended_steps(
        """
1. Convert features to appropriate data types based on sample values.
2. Remove string/categorical features with unique values equal to the dataset size.
3. Remove constant features (single unique value).
4. For high-cardinality categoricals (category frequency < 5%): bucket infrequent values to "Other".
5. One-hot encode categorical variables after bucketing.
6. Convert boolean features to integers (0/1) after encoding.
7. Create datetime-based features when datetime columns exist (year, month, day, etc.).
8. If a target variable is provided:
   - If categorical, label-encode it.
   - Otherwise, convert to numeric if needed and keep in the returned DataFrame.
9. Return a single DataFrame with engineered features (and target if provided).
        """,
        heading="# Recommended Feature Engineering Steps:",
    )

    def _summarize_df_for_prompt(df: pd.DataFrame) -> str:
        """
        Lightweight schema summary to keep prompts small and focused.
        """
        df_limited = df.iloc[:, :MAX_SUMMARY_COLUMNS] if df.shape[1] > MAX_SUMMARY_COLUMNS else df
        schema = []
        n_rows = len(df_limited)
        for col in df_limited.columns:
            series = df_limited[col]
            dtype = str(series.dtype)
            missing_pct = float(series.isna().mean() * 100) if n_rows > 0 else 0.0
            n_unique = int(series.nunique(dropna=True))
            sample_vals = series.dropna().unique()[:3].tolist()
            schema.append(
                {
                    "column": col,
                    "dtype": dtype,
                    "missing_pct": round(missing_pct, 2),
                    "n_unique": n_unique,
                    "sample_values": sample_vals,
                }
            )
        return json.dumps({"n_rows": n_rows, "n_cols": df_limited.shape[1], "schema": schema}, indent=2)

    # Define GraphState for the router
    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        recommended_steps: str
        data_raw: dict
        data_engineered: dict
        target_variable: str
        all_datasets_summary: str
        feature_engineer_function: str
        feature_engineer_function_path: str
        feature_engineer_file_name: str
        feature_engineer_function_name: str
        feature_engineer_error: str
        feature_engineer_error_log_path: str
        max_retries: int
        retry_count: int

    def recommend_feature_engineering_steps(state: GraphState):
        """
        Recommend a series of feature engineering steps based on the input data.
        These recommended steps will be appended to the user_instructions.
        """
        print(format_agent_name(AGENT_NAME))
        print("    * RECOMMEND FEATURE ENGINEERING STEPS")

        # Prompt to get recommended steps from the LLM
        recommend_steps_prompt = PromptTemplate(
            template="""
            You are a Feature Engineering Expert. Given the following information about the data, 
            recommend a series of numbered steps to take to engineer features. 
            The steps should be tailored to the data characteristics and should be helpful 
            for a feature engineering agent that will be implemented.
            
            General Steps:
            Things that should be considered in the feature engineering steps:
            
            * Convert features to the appropriate data types based on their sample data values
            * Remove string or categorical features with unique values equal to the size of the dataset
            * Remove constant features with the same value in all rows
            * High cardinality categorical features should be encoded by a threshold <= 5 percent of the dataset, by converting infrequent values to "other"
            * Encoding categorical variables using OneHotEncoding
            * Numeric features should be left untransformed
            * Create datetime-based features if datetime columns are present
            * If a target variable is provided:
                * If a categorical target variable is provided, encode it using LabelEncoding
                * All other target variables should be converted to numeric and unscaled
            * Convert any Boolean (True/False) values to integer (1/0) values. This should be performed after one-hot encoding.
            
            Custom Steps:
            * Analyze the data to determine if any additional feature engineering steps are needed.
            * Recommend steps that are specific to the data provided. Include why these steps are necessary or beneficial.
            * If no additional steps are needed, simply state that no additional steps are required.
            
            IMPORTANT:
            Make sure to take into account any additional user instructions that may add, remove or modify some of these steps. Include comments in your code to explain your reasoning for each step. Include comments if something is not done because a user requested. Include comments if something is done because a user requested.
            
            User instructions:
            {user_instructions}
            
            Previously Recommended Steps (if any):
            {recommended_steps}
            
            Below are summaries of all datasets provided:
            {all_datasets_summary}

            Return steps as a numbered list. You can return short code snippets to demonstrate actions. But do not return a fully coded solution. The code will be generated separately by a Coding Agent.
            
            Avoid these:
            1. Do not include steps to save files.
            2. Do not include unrelated user instructions that are not related to the feature engineering.
            """,
            input_variables=[
                "user_instructions",
                "recommended_steps",
                "all_datasets_summary",
            ],
        )

        data_raw = state.get("data_raw")
        df = pd.DataFrame.from_dict(data_raw)

        all_datasets_summary_str = _summarize_df_for_prompt(df)

        steps_agent = recommend_steps_prompt | llm
        recommended_steps = steps_agent.invoke(
            {
                "user_instructions": state.get("user_instructions"),
                "recommended_steps": state.get("recommended_steps"),
                "all_datasets_summary": all_datasets_summary_str,
            }
        )

        return {
            "recommended_steps": format_recommended_steps(
                recommended_steps.content.strip(),
                heading="# Recommended Feature Engineering Steps:",
            ),
            "all_datasets_summary": all_datasets_summary_str,
        }

    # Human Review

    prompt_text_human_review = "Are the following feature engineering instructions correct? (Answer 'yes' or provide modifications)\n{steps}"

    if not bypass_explain_code:

        def human_review(
            state: GraphState,
        ) -> Command[
            Literal[
                "recommend_feature_engineering_steps",
                "report_agent_outputs",
            ]
        ]:
            return node_func_human_review(
                state=state,
                prompt_text=prompt_text_human_review,
                yes_goto="report_agent_outputs",
                no_goto="recommend_feature_engineering_steps",
                user_instructions_key="user_instructions",
                recommended_steps_key="recommended_steps",
                code_snippet_key="feature_engineer_function",
            )
    else:

        def human_review(
            state: GraphState,
        ) -> Command[Literal["recommend_feature_engineering_steps", "__end__"]]:
            return node_func_human_review(
                state=state,
                prompt_text=prompt_text_human_review,
                yes_goto="__end__",
                no_goto="recommend_feature_engineering_steps",
                user_instructions_key="user_instructions",
                recommended_steps_key="recommended_steps",
                code_snippet_key="feature_engineer_function",
            )

    def create_feature_engineering_code(state: GraphState):
        if bypass_recommended_steps:
            print(format_agent_name(AGENT_NAME))

            data_raw = state.get("data_raw")
            df = pd.DataFrame.from_dict(data_raw)

            all_datasets_summary_str = _summarize_df_for_prompt(df)
            steps_for_prompt = DEFAULT_FEATURE_STEPS
        else:
            all_datasets_summary_str = state.get("all_datasets_summary")
            steps_for_prompt = state.get("recommended_steps", DEFAULT_FEATURE_STEPS)

            if not steps_for_prompt:
                steps_for_prompt = DEFAULT_FEATURE_STEPS

        print("    * CREATE FEATURE ENGINEERING CODE")

        feature_engineering_prompt = PromptTemplate(
            template="""
            You are a Feature Engineering Agent. Your job is to create a {function_name}() function that can be run on the data provided using the following recommended steps.
            
            Recommended Steps:
            {recommended_steps}
            
            Use the schema summary below to decide appropriate transformations. Keep the prompt small; do not request more data.

            Target Variable (if provided): {target_variable}

            Dataset Schema Summary (capped columns):
            {all_datasets_summary}
            
            You can use Pandas, Numpy, and Scikit Learn libraries to feature engineer the data.
            Prefer deterministic, dependency-light transforms (use pandas and numpy; avoid external libraries unless necessary). If you must use sklearn, include the import and keep it minimal.
            Do NOT hardcode domain-specific column names or derived features unless explicitly requested by the user. Derive column lists from the schema (dtypes, unique counts) and user instructions only.
            Do NOT invent domain-specific engineered features (e.g., tenure buckets, custom ratios) unless the user requested them. Stick to generic transformations: type coercion, missing imputation, high-cardinality bucketing, one-hot encoding, boolean to int.
            Bucket high-cardinality categoricals using a frequency threshold (<5% -> "Other") before one-hot encoding.

            Return Python code in ```python``` format with a single function definition, {function_name}(data_raw), including all imports inside the function. The function must:
            - Separate features and target (if provided), encode the target only if categorical, and reattach target to the returned DataFrame.
            - Impute missing values (simple strategies are fine).
            - One-hot encode categorical columns, with high-cardinality bucketing first.
            - Convert booleans to integers after encoding.
            - Return a pandas DataFrame (not fit objects), and do not persist models.
            
            Best Practices and Error Preventions:
            - Handle missing values in numeric and categorical features before transformations.
            - Avoid creating highly correlated features unless explicitly instructed.
            - Convert Boolean to integer values (0/1) after one-hot encoding unless otherwise instructed.
            - Preserve the target variable in the returned DataFrame when provided.
            - Do not return fit objects (encoders/imputers); only return the transformed DataFrame.
            - For high-cardinality categoricals: bucket categories with frequency < 5% to 'Other' before encoding.
            - Avoid hardcoded column names; build feature lists programmatically from the schema and user instructions.
            
            Avoid the following errors:
            
            - name 'OneHotEncoder' is not defined
            
            - Shape of passed values is (7043, 48), indices imply (7043, 47)
            
            - name 'numeric_features' is not defined
            
            - name 'categorical_features' is not defined


            """,
            input_variables=[
                "recommended_steps",
                "target_variable",
                "all_datasets_summary",
                "function_name",
            ],
        )

        feature_engineering_agent = (
            feature_engineering_prompt | llm | PythonOutputParser()
        )

        response = feature_engineering_agent.invoke(
            {
                "recommended_steps": steps_for_prompt,
                "target_variable": state.get("target_variable"),
                "all_datasets_summary": all_datasets_summary_str,
                "function_name": function_name,
            }
        )

        response = relocate_imports_inside_function(response)
        response = add_comments_to_top(response, agent_name=AGENT_NAME)

        # For logging: store the code generated
        file_path, file_name_2 = log_ai_function(
            response=response,
            file_name=file_name,
            log=log,
            log_path=log_path,
            overwrite=overwrite,
        )

        return {
            "feature_engineer_function": response,
            "feature_engineer_function_path": file_path,
            "feature_engineer_file_name": file_name_2,
            "feature_engineer_function_name": function_name,
            "all_datasets_summary": all_datasets_summary_str,
        }

    def execute_feature_engineering_code(state):
        print("    * EXECUTE FEATURE ENGINEERING CODE (SANDBOXED)")

        result, error = run_code_sandboxed_subprocess(
            code_snippet=state.get("feature_engineer_function"),
            function_name=state.get("feature_engineer_function_name"),
            data=state.get("data_raw"),
            timeout=10,
            memory_limit_mb=512,
            data_format="dataframe",
        )

        # Validate result structure
        validation_error = None
        df_out = None
        if error is None:
            try:
                df_out = pd.DataFrame(result)
                target = state.get("target_variable")
                if target and target not in df_out.columns:
                    validation_error = (
                        f"Target column '{target}' missing from engineered output."
                    )
            except Exception as exc:
                validation_error = f"Engineered output is not a valid table: {exc}"
        else:
            validation_error = error

        error_prefixed = (
            f"An error occurred during feature engineering: {validation_error}"
            if validation_error
            else None
        )

        error_log_path = None
        if error_prefixed and log:
            error_log_path = log_ai_error(
                error_message=error_prefixed,
                file_name=f"{function_name}_errors.log",
                log=log,
                log_path=log_path if log_path is not None else "logs/",
                overwrite=False,
            )
            if error_log_path:
                print(f"      Error logged to: {error_log_path}")

        result_payload = df_out.to_dict() if error_prefixed is None else None

        return {
            "data_engineered": result_payload,
            "feature_engineer_error": error_prefixed,
            "feature_engineer_error_log_path": error_log_path,
        }

    def fix_feature_engineering_code(state: GraphState):
        feature_engineer_prompt = """
        You are a Feature Engineering Agent. Your job is to fix the {function_name}() function that currently contains errors.
        
        Provide only the corrected function definition for {function_name}().
        
        Return Python code in ```python``` format with a single function definition, {function_name}(data_raw), that includes all imports inside the function.
        
        This is the broken code (please fix): 
        {code_snippet}

        Last Known Error:
        {error}
        """

        return node_func_fix_agent_code(
            state=state,
            code_snippet_key="feature_engineer_function",
            error_key="feature_engineer_error",
            llm=llm,
            prompt_template=feature_engineer_prompt,
            agent_name=AGENT_NAME,
            log=log,
            file_path=state.get("feature_engineer_function_path"),
            function_name=state.get("feature_engineer_function_name"),
        )

    # Final reporting node
    def report_agent_outputs(state: GraphState):
        return node_func_report_agent_outputs(
            state=state,
            keys_to_include=[
                "recommended_steps",
                "feature_engineer_function",
                "feature_engineer_function_path",
                "feature_engineer_function_name",
                "feature_engineer_error",
                "feature_engineer_error_log_path",
            ],
            result_key="messages",
            role=AGENT_NAME,
            custom_title="Feature Engineering Agent Outputs",
        )

    # Create the graph
    node_functions = {
        "recommend_feature_engineering_steps": recommend_feature_engineering_steps,
        "human_review": human_review,
        "create_feature_engineering_code": create_feature_engineering_code,
        "execute_feature_engineering_code": execute_feature_engineering_code,
        "fix_feature_engineering_code": fix_feature_engineering_code,
        "report_agent_outputs": report_agent_outputs,
    }

    app = create_coding_agent_graph(
        GraphState=GraphState,
        node_functions=node_functions,
        recommended_steps_node_name="recommend_feature_engineering_steps",
        create_code_node_name="create_feature_engineering_code",
        execute_code_node_name="execute_feature_engineering_code",
        fix_code_node_name="fix_feature_engineering_code",
        explain_code_node_name="report_agent_outputs",
        error_key="feature_engineer_error",
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
