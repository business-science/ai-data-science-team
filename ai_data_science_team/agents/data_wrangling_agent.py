# BUSINESS SCIENCE UNIVERSITY
# AI DATA SCIENCE TEAM
# ***
# * Agents: Data Wrangling Agent

# Libraries
from typing_extensions import TypedDict, Annotated, Sequence, Literal, Union, Optional
import operator
import os
import json
import pandas as pd
from IPython.display import Markdown

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage
from langgraph.types import Command
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
from ai_data_science_team.utils.logging import log_ai_function

# Setup Logging Path
AGENT_NAME = "data_wrangling_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")

# Class


class DataWranglingAgent(BaseAgent):
    """
    Creates a data wrangling agent that can work with one or more datasets, performing operations such as
    joining/merging multiple datasets, reshaping, aggregating, encoding, creating computed features,
    and ensuring consistent data types. The agent generates a Python function to wrangle the data,
    executes the function, and logs the process (if enabled).

    The agent can handle:
    - A single dataset (provided as a dictionary of {column: list_of_values})
    - Multiple datasets (provided as a list of such dictionaries)

    Key wrangling steps can include:
    - Merging or joining datasets
    - Pivoting/melting data for reshaping
    - GroupBy aggregations (sums, means, counts, etc.)
    - Encoding categorical variables
    - Computing new columns from existing ones
    - Dropping or rearranging columns
    - Any additional user instructions

    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model used to generate the data wrangling function.
    n_samples : int, optional
        Number of samples to show in the data summary for wrangling. Defaults to 30.
    log : bool, optional
        Whether to log the generated code and errors. Defaults to False.
    log_path : str, optional
        Directory path for storing log files. Defaults to None.
    file_name : str, optional
        Name of the file for saving the generated response. Defaults to "data_wrangler.py".
    function_name : str, optional
        Name of the function to be generated. Defaults to "data_wrangler".
    overwrite : bool, optional
        Whether to overwrite the log file if it exists. If False, a unique file name is created. Defaults to True.
    human_in_the_loop : bool, optional
        Enables user review of data wrangling instructions. Defaults to False.
    bypass_recommended_steps : bool, optional
        If True, skips the step that generates recommended data wrangling steps. Defaults to False.
    bypass_explain_code : bool, optional
        If True, skips the step that provides code explanations. Defaults to False.
    checkpointer : Checkpointer, optional
        A checkpointer object to save and load the agent's state. Defaults to None.

    Methods
    -------
    update_params(**kwargs)
        Updates the agent's parameters and rebuilds the compiled state graph.

    ainvoke_agent(user_instructions: str, data_raw: Union[dict, list], max_retries=3, retry_count=0)
        Asynchronously wrangles the provided dataset(s) based on user instructions.

    invoke_agent(user_instructions: str, data_raw: Union[dict, list], max_retries=3, retry_count=0)
        Synchronously wrangles the provided dataset(s) based on user instructions.

    get_workflow_summary()
        Retrieves a summary of the agent's workflow.

    get_log_summary()
        Retrieves a summary of logged operations if logging is enabled.

    get_data_wrangled()
        Retrieves the final wrangled dataset (as a dictionary of {column: list_of_values}).

    get_data_raw()
        Retrieves the raw dataset(s).

    get_data_wrangler_function()
        Retrieves the generated Python function used for data wrangling.

    get_recommended_wrangling_steps()
        Retrieves the agent's recommended wrangling steps.

    get_response()
        Returns the full response dictionary from the agent.

    show()
        Displays the agent's mermaid diagram for visual inspection of the compiled graph.

    Examples
    --------
    ```python
    import pandas as pd
    from langchain_openai import ChatOpenAI
    from ai_data_science_team.agents import DataWranglingAgent

    # Single dataset example
    llm = ChatOpenAI(model="gpt-4o-mini")

    data_wrangling_agent = DataWranglingAgent(
        model=llm,
        n_samples=30,
        log=True,
        log_path="logs",
        human_in_the_loop=True
    )

    df = pd.read_csv("https://raw.githubusercontent.com/business-science/ai-data-science-team/refs/heads/master/data/churn_data.csv")

    data_wrangling_agent.invoke_agent(
        user_instructions="Group by 'gender' and compute mean of 'tenure'.",
        data_raw=df,  # data_raw can be df.to_dict() or just a DataFrame
        max_retries=3,
        retry_count=0
    )

    data_wrangled = data_wrangling_agent.get_data_wrangled()
    response = data_wrangling_agent.get_response()

    # Multiple dataset example (list of dicts)
    df1 = pd.DataFrame({'id': [1,2,3], 'val1': [10,20,30]})
    df2 = pd.DataFrame({'id': [1,2,3], 'val2': [40,50,60]})

    data_wrangling_agent.invoke_agent(
        user_instructions="Merge these two datasets on 'id' and compute a new column 'val_sum' = val1+val2",
        data_raw=[df1, df2],   # multiple datasets
        max_retries=3,
        retry_count=0
    )

    data_wrangled = data_wrangling_agent.get_data_wrangled()
    ```

    Returns
    -------
    DataWranglingAgent : langchain.graphs.CompiledStateGraph
        A data wrangling agent implemented as a compiled state graph.
    """

    def __init__(
        self,
        model,
        n_samples=30,
        log=False,
        log_path=None,
        file_name="data_wrangler.py",
        function_name="data_wrangler",
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
        Create the compiled graph for the data wrangling agent.
        Running this method will reset the response to None.
        """
        self.response = None
        return make_data_wrangling_agent(**self._params)

    def update_params(self, **kwargs):
        """
        Updates the agent's parameters and rebuilds the compiled graph.
        """
        for k, v in kwargs.items():
            self._params[k] = v
        self._compiled_graph = self._make_compiled_graph()

    async def ainvoke_agent(
        self,
        data_raw: Union[pd.DataFrame, dict, list],
        user_instructions: str = None,
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
        """
        Asynchronously wrangles the provided dataset(s) based on user instructions.
        The response is stored in the 'response' attribute.

        Parameters
        ----------
        data_raw : Union[pd.DataFrame, dict, list]
            The raw dataset(s) to be wrangled.
            Can be a single DataFrame, a single dict ({col: list_of_values}),
              or a list of dicts if multiple datasets are provided.
        user_instructions : str
            Instructions for data wrangling.
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
        data_input = self._convert_data_input(data_raw)
        response = await self._compiled_graph.ainvoke(
            {
                "user_instructions": user_instructions,
                "data_raw": data_input,
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        self.response = response
        return None

    def invoke_agent(
        self,
        data_raw: Union[pd.DataFrame, dict, list],
        user_instructions: str = None,
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
        """
        Synchronously wrangles the provided dataset(s) based on user instructions.
        The response is stored in the 'response' attribute.

        Parameters
        ----------
        data_raw : Union[pd.DataFrame, dict, list]
            The raw dataset(s) to be wrangled.
            Can be a single DataFrame, a single dict, or a list of dicts.
        user_instructions : str
            Instructions for data wrangling agent.
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
        data_input = self._convert_data_input(data_raw)
        response = self._compiled_graph.invoke(
            {
                "user_instructions": user_instructions,
                "data_raw": data_input,
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
            if self.response.get("data_wrangler_function_path"):
                log_details = f"""
## Data Wrangling Agent Log Summary:

Function Path: {self.response.get("data_wrangler_function_path")}

Function Name: {self.response.get("data_wrangler_function_name")}
                """
                if markdown:
                    return Markdown(log_details)
                else:
                    return log_details

    def get_data_wrangled(self) -> Optional[pd.DataFrame]:
        """
        Retrieves the wrangled data after running invoke_agent() or ainvoke_agent().

        Returns
        -------
        pd.DataFrame or None
            The wrangled dataset as a pandas DataFrame (if available).
        """
        if self.response and "data_wrangled" in self.response:
            return pd.DataFrame(self.response["data_wrangled"])
        return None

    def get_data_raw(self) -> Union[dict, list, None]:
        """
        Retrieves the original raw data from the last invocation.

        Returns
        -------
        Union[dict, list, None]
            The original dataset(s) as a single dict or a list of dicts, or None if not available.
        """
        if self.response and "data_raw" in self.response:
            return self.response["data_raw"]
        return None

    def get_data_wrangler_function(self, markdown=False) -> Optional[str]:
        """
        Retrieves the generated data wrangling function code.

        Parameters
        ----------
        markdown : bool, optional
            If True, returns the function in Markdown code block format.

        Returns
        -------
        str or None
            The Python function code, or None if not available.
        """
        if self.response and "data_wrangler_function" in self.response:
            code = self.response["data_wrangler_function"]
            if markdown:
                return Markdown(f"```python\n{code}\n```")
            return code
        return None

    def get_recommended_wrangling_steps(self, markdown=False) -> Optional[str]:
        """
        Retrieves the agent's recommended data wrangling steps.

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

    @staticmethod
    def _convert_data_input(
        data_raw: Union[pd.DataFrame, dict, list],
    ) -> Union[dict, list]:
        """
        Internal utility to convert data_raw (which could be a DataFrame, dict, or list of dicts)
        into the format expected by the underlying agent (dict or list of dicts).

        Parameters
        ----------
        data_raw : Union[pd.DataFrame, dict, list]
            The raw input data to be converted.

        Returns
        -------
        Union[dict, list]
            The data in a dictionary or list-of-dictionaries format.
        """
        # If a single DataFrame, convert to dict
        if isinstance(data_raw, pd.DataFrame):
            return data_raw.to_dict()

        # If it's already a dict (single dataset)
        if isinstance(data_raw, dict):
            return data_raw

        # If it's already a list, check if it's a list of DataFrames or dicts
        if isinstance(data_raw, list):
            # Convert any DataFrame item to dict
            converted_list = []
            for item in data_raw:
                if isinstance(item, pd.DataFrame):
                    converted_list.append(item.to_dict())
                elif isinstance(item, dict):
                    converted_list.append(item)
                else:
                    raise ValueError(
                        "List must contain only DataFrames or dictionaries."
                    )
            return converted_list

        raise ValueError(
            "data_raw must be a DataFrame, a dict, or a list of dicts/DataFrames."
        )


# Function


def make_data_wrangling_agent(
    model,
    n_samples=30,
    log=False,
    log_path=None,
    file_name="data_wrangler.py",
    function_name="data_wrangler",
    overwrite=True,
    human_in_the_loop=False,
    bypass_recommended_steps=False,
    bypass_explain_code=False,
    checkpointer=None,
):
    """
    Creates a data wrangling agent that can be run on one or more datasets. The agent can be
    instructed to perform common data wrangling steps such as:

    - Joining or merging multiple datasets
    - Reshaping data (pivoting, melting)
    - Aggregating data via groupby operations
    - Encoding categorical variables (one-hot, label encoding)
    - Creating computed features (e.g., ratio of two columns)
    - Ensuring consistent data types
    - Dropping or rearranging columns

    The agent takes in one or more datasets (passed as a dictionary or list of dictionaries if working on multiple dictionaries), user instructions,
    and outputs a python function that can be used to wrangle the data. If multiple datasets
    are provided, the agent should combine or transform them according to user instructions.

    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model to use to generate code.
    n_samples : int, optional
        The number of samples to show in the data summary. Defaults to 30.
        If you get an error due to maximum tokens, try reducing this number.
        > "This model's maximum context length is 128000 tokens. However, your messages resulted in 333858 tokens. Please reduce the length of the messages."
    log : bool, optional
        Whether or not to log the code generated and any errors that occur.
        Defaults to False.
    log_path : str, optional
        The path to the directory where the log files should be stored. Defaults to "logs/".
    file_name : str, optional
        The name of the file to save the response to. Defaults to "data_wrangler.py".
    function_name : str, optional
        The name of the function to be generated. Defaults to "data_wrangler".
    overwrite : bool, optional
        Whether or not to overwrite the log file if it already exists. If False, a unique file name will be created.
        Defaults to True.
    human_in_the_loop : bool, optional
        Whether or not to use human in the loop. If True, adds an interrupt and human-in-the-loop
        step that asks the user to review the data wrangling instructions. Defaults to False.
    bypass_recommended_steps : bool, optional
        Bypass the recommendation step, by default False
    bypass_explain_code : bool, optional
        Bypass the code explanation step, by default False.
    checkpointer : Checkpointer, optional
        A checkpointer object to save and load the agent's state. Defaults to None.

    Example
    -------
    ``` python
    from langchain_openai import ChatOpenAI
    import pandas as pd

    df = pd.DataFrame({
        'category': ['A', 'B', 'A', 'C'],
        'value': [10, 20, 15, 5]
    })

    llm = ChatOpenAI(model="gpt-4o-mini")

    data_wrangling_agent = make_data_wrangling_agent(llm)

    response = data_wrangling_agent.invoke({
        "user_instructions": "Calculate the sum and mean of 'value' by 'category'.",
        "data_raw": df.to_dict(),
        "max_retries":3,
        "retry_count":0
    })
    pd.DataFrame(response['data_wrangled'])
    ```

    Returns
    -------
    app : langchain.graphs.CompiledStateGraph
        The data wrangling agent as a state graph.
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
            log_path = LOG_PATH
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        recommended_steps: str
        # data_raw should be a dict for a single dataset or a list of dicts for multiple datasets
        data_raw: Union[dict, list]
        data_wrangled: dict
        all_datasets_summary: str
        data_wrangler_function: str
        data_wrangler_function_path: str
        data_wrangler_function_name: str
        data_wrangler_error: str
        max_retries: int
        retry_count: int

    def recommend_wrangling_steps(state: GraphState):
        print(format_agent_name(AGENT_NAME))
        print("    * RECOMMEND WRANGLING STEPS")

        data_raw = state.get("data_raw")

        if isinstance(data_raw, dict):
            # Single dataset scenario
            primary_dataset_name = "main"
            datasets = {primary_dataset_name: data_raw}
        elif isinstance(data_raw, list) and all(
            isinstance(item, dict) for item in data_raw
        ):
            # Multiple datasets scenario
            datasets = {f"dataset_{i}": d for i, d in enumerate(data_raw, start=1)}
            primary_dataset_name = "dataset_1"
        else:
            raise ValueError("data_raw must be a dict or a list of dicts.")

        # Convert all datasets to DataFrames for inspection
        dataframes = {name: pd.DataFrame.from_dict(d) for name, d in datasets.items()}

        # Create a summary for all datasets
        # We'll include a short sample and info for each dataset
        all_datasets_summary = get_dataframe_summary(
            dataframes, n_sample=n_samples, skip_stats=True
        )

        # Join all datasets summaries into one big text block
        all_datasets_summary_str = "\n\n".join(all_datasets_summary)

        # Prepare the prompt:
        # We now include summaries for all datasets, not just the primary dataset.
        # The LLM can then use all this info to recommend steps that consider merging/joining.
        recommend_steps_prompt = PromptTemplate(
            template="""
            You are a Data Wrangling Expert. Given the following data (one or multiple datasets) and user instructions, 
            recommend a series of numbered steps to wrangle the data based on a user's needs. 
            
            You can use any common data wrangling techniques such as joining, reshaping, aggregating, encoding, etc. 
            
            If multiple datasets are provided, you may need to recommend how to merge or join them. 
            
            Also consider any special transformations requested by the user. If the user instructions 
            say to do something else or not to do certain steps, follow those instructions.
            
            User instructions:
            {user_instructions}

            Previously Recommended Steps (if any):
            {recommended_steps}

            Below are summaries of all datasets provided:
            {all_datasets_summary}

            Return steps as a numbered list. You can return short code snippets to demonstrate actions. But do not return a fully coded solution. The code will be generated separately by a Coding Agent.
            
            Avoid these:
            1. Do not include steps to save files.
            2. Do not include unrelated user instructions that are not related to the data wrangling.
            """,
            input_variables=[
                "user_instructions",
                "recommended_steps",
                "all_datasets_summary",
            ],
        )

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
                heading="# Recommended Data Wrangling Steps:",
            ),
            "all_datasets_summary": all_datasets_summary_str,
        }

    def create_data_wrangler_code(state: GraphState):
        if bypass_recommended_steps:
            print(format_agent_name(AGENT_NAME))

            data_raw = state.get("data_raw")

            if isinstance(data_raw, dict):
                # Single dataset scenario
                primary_dataset_name = "main"
                datasets = {primary_dataset_name: data_raw}
            elif isinstance(data_raw, list) and all(
                isinstance(item, dict) for item in data_raw
            ):
                # Multiple datasets scenario
                datasets = {f"dataset_{i}": d for i, d in enumerate(data_raw, start=1)}
                primary_dataset_name = "dataset_1"
            else:
                raise ValueError("data_raw must be a dict or a list of dicts.")

            # Convert all datasets to DataFrames for inspection
            dataframes = {
                name: pd.DataFrame.from_dict(d) for name, d in datasets.items()
            }

            # Create a summary for all datasets
            # We'll include a short sample and info for each dataset
            all_datasets_summary = get_dataframe_summary(
                dataframes, n_sample=n_samples, skip_stats=True
            )

            # Join all datasets summaries into one big text block
            all_datasets_summary_str = "\n\n".join(all_datasets_summary)

        else:
            all_datasets_summary_str = state.get("all_datasets_summary")

        print("    * CREATE DATA WRANGLER CODE")

        data_wrangling_prompt = PromptTemplate(
            template="""
            You are a Pandas Data Wrangling Coding Agent. Your job is to create a {function_name}() function that can be run on the provided data. You should use Pandas and NumPy for data wrangling operations.
            
            User instructions:
            {user_instructions}
            
            Follow these recommended steps (if present):
            {recommended_steps}
            
            If multiple datasets are provided, you may need to merge or join them. Make sure to handle that scenario based on the recommended steps and user instructions.
            
            Below are summaries of all datasets provided. If more than one dataset is provided, you may need to merge or join them.:
            {all_datasets_summary}
            
            Return Python code in ```python``` format with a single function definition, {function_name}(), that includes all imports inside the function. And returns a single pandas data frame.

            ```python
            def {function_name}(data_list):
                '''
                Wrangle the data provided in data.
                
                data_list: A list of one or more pandas data frames containing the raw data to be wrangled.
                '''
                import pandas as pd
                import numpy as np
                # Implement the wrangling steps here
                
                # Return a single DataFrame 
                return data_wrangled
            ```
            
            Avoid Errors:
            1. If the incoming data is not a list. Convert it to a list first. 
            2. Do not specify data types inside the function arguments.
            
            Important Notes:
            1. Do Not use Print statements to display the data. Return the data frame instead with the data wrangling operation performed.
            2. Do not plot graphs. Only return the data frame.
            
            Make sure to explain any non-trivial steps with inline comments. Follow user instructions. Comment code thoroughly.
            
            
            """,
            input_variables=[
                "recommended_steps",
                "user_instructions",
                "all_datasets_summary",
                "function_name",
            ],
        )

        data_wrangling_agent = data_wrangling_prompt | llm | PythonOutputParser()

        response = data_wrangling_agent.invoke(
            {
                "recommended_steps": state.get("recommended_steps"),
                "user_instructions": state.get("user_instructions"),
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
            "data_wrangler_function": response,
            "data_wrangler_function_path": file_path,
            "data_wrangler_file_name": file_name_2,
            "data_wrangler_function_name": function_name,
            "all_datasets_summary": all_datasets_summary_str,
        }

    def human_review(
        state: GraphState,
    ) -> Command[Literal["recommend_wrangling_steps", "create_data_wrangler_code"]]:
        return node_func_human_review(
            state=state,
            prompt_text="Are the following data wrangling steps correct? (Answer 'yes' or provide modifications)\n{steps}",
            yes_goto="create_data_wrangler_code",
            no_goto="recommend_wrangling_steps",
            user_instructions_key="user_instructions",
            recommended_steps_key="recommended_steps",
        )

    # Human Review

    prompt_text_human_review = "Are the following data wrangling instructions correct? (Answer 'yes' or provide modifications)\n{steps}"

    if not bypass_explain_code:

        def human_review(
            state: GraphState,
        ) -> Command[
            Literal["recommend_wrangling_steps", "explain_data_wrangler_code"]
        ]:
            return node_func_human_review(
                state=state,
                prompt_text=prompt_text_human_review,
                yes_goto="explain_data_wrangler_code",
                no_goto="recommend_wrangling_steps",
                user_instructions_key="user_instructions",
                recommended_steps_key="recommended_steps",
                code_snippet_key="data_wrangler_function",
            )
    else:

        def human_review(
            state: GraphState,
        ) -> Command[Literal["recommend_wrangling_steps", "__end__"]]:
            return node_func_human_review(
                state=state,
                prompt_text=prompt_text_human_review,
                yes_goto="__end__",
                no_goto="recommend_wrangling_steps",
                user_instructions_key="user_instructions",
                recommended_steps_key="recommended_steps",
                code_snippet_key="data_wrangler_function",
            )

    def execute_data_wrangler_code(state: GraphState):
        return node_func_execute_agent_code_on_data(
            state=state,
            data_key="data_raw",
            result_key="data_wrangled",
            error_key="data_wrangler_error",
            code_snippet_key="data_wrangler_function",
            agent_function_name=state.get("data_wrangler_function_name"),
            # pre_processing=pre_processing,
            post_processing=lambda df: df.to_dict()
            if isinstance(df, pd.DataFrame)
            else df,
            error_message_prefix="An error occurred during data wrangling: ",
        )

    def fix_data_wrangler_code(state: GraphState):
        data_wrangler_prompt = """
        You are a Data Wrangling Agent. Your job is to create a {function_name}() function that can be run on the data provided. The function is currently broken and needs to be fixed.
        
        Make sure to only return the function definition for {function_name}().
        
        Return Python code in ```python``` format with a single function definition, {function_name}(data_raw), that includes all imports inside the function.
        
        This is the broken code (please fix): 
        {code_snippet}

        Last Known Error:
        {error}
        """

        return node_func_fix_agent_code(
            state=state,
            code_snippet_key="data_wrangler_function",
            error_key="data_wrangler_error",
            llm=llm,
            prompt_template=data_wrangler_prompt,
            agent_name=AGENT_NAME,
            log=log,
            file_path=state.get("data_wrangler_function_path"),
            function_name=state.get("data_wrangler_function_name"),
        )

    # Final reporting node
    def report_agent_outputs(state: GraphState):
        return node_func_report_agent_outputs(
            state=state,
            keys_to_include=[
                "recommended_steps",
                "data_wrangler_function",
                "data_wrangler_function_path",
                "data_wrangler_function_name",
                "data_wrangler_error",
            ],
            result_key="messages",
            role=AGENT_NAME,
            custom_title="Data Wrangling Agent Outputs",
        )

    # Define the graph
    node_functions = {
        "recommend_wrangling_steps": recommend_wrangling_steps,
        "human_review": human_review,
        "create_data_wrangler_code": create_data_wrangler_code,
        "execute_data_wrangler_code": execute_data_wrangler_code,
        "fix_data_wrangler_code": fix_data_wrangler_code,
        "report_agent_outputs": report_agent_outputs,
    }

    app = create_coding_agent_graph(
        GraphState=GraphState,
        node_functions=node_functions,
        recommended_steps_node_name="recommend_wrangling_steps",
        create_code_node_name="create_data_wrangler_code",
        execute_code_node_name="execute_data_wrangler_code",
        fix_code_node_name="fix_data_wrangler_code",
        explain_code_node_name="report_agent_outputs",
        error_key="data_wrangler_error",
        human_in_the_loop=human_in_the_loop,
        human_review_node_name="human_review",
        checkpointer=checkpointer,
        bypass_recommended_steps=bypass_recommended_steps,
        bypass_explain_code=bypass_explain_code,
        agent_name=AGENT_NAME,
    )

    return app
