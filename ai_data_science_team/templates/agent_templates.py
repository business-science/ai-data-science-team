from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.graph.state import CompiledStateGraph

from langchain_core.runnables import RunnableConfig
from langgraph.pregel.types import StreamMode

import pandas as pd
import sqlalchemy as sql
import json

from typing_extensions import Any, Callable, Dict, Type, Optional, Union, List

from ai_data_science_team.parsers.parsers import PythonOutputParser
from ai_data_science_team.utils.regex import (
    relocate_imports_inside_function,
    add_comments_to_top,
    remove_consecutive_duplicates,
)

from IPython.display import Image, display
import pandas as pd


class BaseAgent(CompiledStateGraph):
    """
    A generic base class for agents that interact with compiled state graphs.

    Provides shared functionality for handling parameters, responses, and state
    graph operations.
    """

    def __init__(self, **params):
        """
        Initialize the agent with provided parameters.

        Parameters:
            **params: Arbitrary keyword arguments representing the agent's parameters.
        """
        self._params = params
        self._compiled_graph = self._make_compiled_graph()
        self.response = None
        self.name = self._compiled_graph.name
        self.checkpointer = self._compiled_graph.checkpointer
        self.store = self._compiled_graph.store
        self.output_channels = self._compiled_graph.output_channels
        self.nodes = self._compiled_graph.nodes
        self.stream_mode = self._compiled_graph.stream_mode
        self.builder = self._compiled_graph.builder
        self.channels = self._compiled_graph.channels
        self.input_channels = self._compiled_graph.input_channels
        self.input_schema = self._compiled_graph.input_schema
        self.output_schema = self._compiled_graph.output_schema
        self.debug = self._compiled_graph.debug
        self.interrupt_after_nodes = self._compiled_graph.interrupt_after_nodes
        self.interrupt_before_nodes = self._compiled_graph.interrupt_before_nodes
        self.config = self._compiled_graph.config

    def _make_compiled_graph(self):
        """
        Subclasses should override this method to create a specific compiled graph.
        """
        raise NotImplementedError(
            "Subclasses must implement the `_make_compiled_graph` method."
        )

    def update_params(self, **kwargs):
        """
        Update one or more parameters and rebuild the compiled graph.

        Parameters:
            **kwargs: Parameters to update.
        """
        self._params.update(kwargs)
        self._compiled_graph = self._make_compiled_graph()

    def __getattr__(self, name: str):
        """
        Delegate attribute access to the compiled graph if the attribute is not found.

        Parameters:
            name (str): The attribute name.

        Returns:
            Any: The attribute from the compiled graph.
        """
        return getattr(self._compiled_graph, name)

    def invoke(
        self,
        input: Union[dict[str, Any], Any],
        config: Optional[RunnableConfig] = None,
        **kwargs,
    ):
        """
        Wrapper for self._compiled_graph.invoke()

        Parameters:
            input: The input data for the graph. It can be a dictionary or any other type.
            config: Optional. The configuration for the graph run.
            **kwarg: Arguments to pass to self._compiled_graph.invoke()

        Returns:
            Any: The agent's response.
        """
        self.response = self._compiled_graph.invoke(
            input=input, config=config, **kwargs
        )

        if self.response.get("messages"):
            self.response["messages"] = remove_consecutive_duplicates(
                self.response["messages"]
            )

        return self.response

    async def ainvoke(
        self,
        input: Union[dict[str, Any], Any],
        config: Optional[RunnableConfig] = None,
        **kwargs,
    ):
        """
        Wrapper for self._compiled_graph.ainvoke()

        Parameters:
            input: The input data for the graph. It can be a dictionary or any other type.
            config: Optional. The configuration for the graph run.
            **kwarg: Arguments to pass to self._compiled_graph.ainvoke()

        Returns:
            Any: The agent's response.
        """
        self.response = await self._compiled_graph.ainvoke(
            input=input, config=config, **kwargs
        )

        if self.response.get("messages"):
            self.response["messages"] = remove_consecutive_duplicates(
                self.response["messages"]
            )

        return self.response

    def stream(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        stream_mode: StreamMode | list[StreamMode] | None = None,
        **kwargs,
    ):
        """
        Wrapper for self._compiled_graph.stream()

        Parameters:
            input: The input to the graph.
            config: The configuration to use for the run.
            stream_mode: The mode to stream output, defaults to self.stream_mode.
                Options are 'values', 'updates', and 'debug'.
                values: Emit the current values of the state for each step.
                updates: Emit only the updates to the state for each step.
                    Output is a dict with the node name as key and the updated values as value.
                debug: Emit debug events for each step.
            **kwarg: Arguments to pass to self._compiled_graph.stream()

        Returns:
            Any: The agent's response.
        """
        self.response = self._compiled_graph.stream(
            input=input, config=config, stream_mode=stream_mode, **kwargs
        )

        if self.response.get("messages"):
            self.response["messages"] = remove_consecutive_duplicates(
                self.response["messages"]
            )

        return self.response

    async def astream(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        stream_mode: StreamMode | list[StreamMode] | None = None,
        **kwargs,
    ):
        """
        Wrapper for self._compiled_graph.astream()

        Parameters:
            input: The input to the graph.
            config: The configuration to use for the run.
            stream_mode: The mode to stream output, defaults to self.stream_mode.
                Options are 'values', 'updates', and 'debug'.
                values: Emit the current values of the state for each step.
                updates: Emit only the updates to the state for each step.
                    Output is a dict with the node name as key and the updated values as value.
                debug: Emit debug events for each step.
            **kwarg: Arguments to pass to self._compiled_graph.astream()

        Returns:
            Any: The agent's response.
        """
        self.response = await self._compiled_graph.astream(
            input=input, config=config, stream_mode=stream_mode, **kwargs
        )

        if self.response.get("messages"):
            self.response["messages"] = remove_consecutive_duplicates(
                self.response["messages"]
            )

        return self.response

    def get_state_keys(self):
        """
        Returns a list of keys that the state graph response contains.

        Returns:
            list: A list of keys in the response.
        """
        return list(self.get_output_jsonschema()["properties"].keys())

    def get_state_properties(self):
        """
        Returns detailed properties of the state graph response.

        Returns:
            dict: The properties of the response.
        """
        return self.get_output_jsonschema()["properties"]

    def get_state(self, config, *, subgraphs=False):
        """
        Returns the state of the agent.
        """
        return self._compiled_graph.get_state(config, subgraphs=subgraphs)

    def get_state_history(self, config, *, filter=None, before=None, limit=None):
        """
        Returns the state history of the agent.
        """
        return self._compiled_graph.get_state_history(
            config, filter=filter, before=before, limit=limit
        )

    def update_state(self, config, values, as_node=None):
        """
        Updates the state of the agent.
        """
        return self._compiled_graph.update_state(config, values, as_node)

    def get_response(self):
        """
        Returns the response generated by the agent.

        Returns:
            Any: The agent's response.
        """
        if self.response.get("messages"):
            self.response["messages"] = remove_consecutive_duplicates(
                self.response["messages"]
            )

        return self.response

    def show(self, xray: int = 0):
        """
        Displays the agent's state graph as a Mermaid diagram.

        Parameters:
            xray (int): If set to 1, displays subgraph levels. Defaults to 0.
        """
        display(Image(self.get_graph(xray=xray).draw_mermaid_png()))


def create_coding_agent_graph(
    GraphState: Type,
    node_functions: Dict[str, Callable],
    recommended_steps_node_name: str,
    create_code_node_name: str,
    execute_code_node_name: str,
    fix_code_node_name: str,
    explain_code_node_name: str,
    error_key: str,
    max_retries_key: str = "max_retries",
    retry_count_key: str = "retry_count",
    human_in_the_loop: bool = False,
    human_review_node_name: str = "human_review",
    checkpointer: Optional[Callable] = None,
    bypass_recommended_steps: bool = False,
    bypass_explain_code: bool = False,
    agent_name: str = "coding_agent",
):
    """
    Creates a generic agent graph using the provided node functions and node names.

    Parameters
    ----------
    GraphState : Type
        The TypedDict or class used as state for the workflow.
    node_functions : dict
        A dictionary mapping node names to their respective functions.
        Example: {
            "recommend_cleaning_steps": recommend_cleaning_steps,
            "human_review": human_review,
            "create_data_cleaner_code": create_data_cleaner_code,
            "execute_data_cleaner_code": execute_data_cleaner_code,
            "fix_data_cleaner_code": fix_data_cleaner_code,
            "explain_data_cleaner_code": explain_data_cleaner_code
        }
    recommended_steps_node_name : str
        The node name that recommends steps.
    create_code_node_name : str
        The node name that creates the code.
    execute_code_node_name : str
        The node name that executes the generated code.
    fix_code_node_name : str
        The node name that fixes code if errors occur.
    explain_code_node_name : str
        The node name that explains the final code.
    error_key : str
        The state key used to check for errors.
    max_retries_key : str, optional
        The state key used for the maximum number of retries.
    retry_count_key : str, optional
        The state key for the current retry count.
    human_in_the_loop : bool, optional
        Whether to include a human review step.
    human_review_node_name : str, optional
        The node name for human review if human_in_the_loop is True.
    checkpointer : callable, optional
        A checkpointer callable if desired.
    bypass_recommended_steps : bool, optional
        Whether to skip the recommended steps node.
    bypass_explain_code : bool, optional
        Whether to skip the final explain code node.
    name : str, optional
        The name of the agent graph.

    Returns
    -------
    app : langchain.graphs.StateGraph
        The compiled workflow application.
    """

    workflow = StateGraph(GraphState)

    # * NODES

    # Always add create, execute, and fix nodes
    workflow.add_node(create_code_node_name, node_functions[create_code_node_name])
    workflow.add_node(execute_code_node_name, node_functions[execute_code_node_name])
    workflow.add_node(fix_code_node_name, node_functions[fix_code_node_name])

    # Conditionally add the recommended-steps node
    if not bypass_recommended_steps:
        workflow.add_node(
            recommended_steps_node_name, node_functions[recommended_steps_node_name]
        )

    # Conditionally add the human review node
    if human_in_the_loop:
        workflow.add_node(
            human_review_node_name, node_functions[human_review_node_name]
        )

    # Conditionally add the explanation node
    if not bypass_explain_code:
        workflow.add_node(
            explain_code_node_name, node_functions[explain_code_node_name]
        )

    # * EDGES

    # Set the entry point
    entry_point = (
        create_code_node_name
        if bypass_recommended_steps
        else recommended_steps_node_name
    )

    workflow.set_entry_point(entry_point)

    if not bypass_recommended_steps:
        workflow.add_edge(recommended_steps_node_name, create_code_node_name)

    workflow.add_edge(create_code_node_name, execute_code_node_name)
    workflow.add_edge(fix_code_node_name, execute_code_node_name)

    # Define a helper to check if we have an error & can still retry
    def error_and_can_retry(state):
        return (
            state.get(error_key) is not None
            and state.get(retry_count_key) is not None
            and state.get(max_retries_key) is not None
            and state[retry_count_key] < state[max_retries_key]
        )

    # If human in the loop, add a branch for human review
    if human_in_the_loop:
        workflow.add_conditional_edges(
            execute_code_node_name,
            lambda s: "fix_code" if error_and_can_retry(s) else "human_review",
            {
                "human_review": human_review_node_name,
                "fix_code": fix_code_node_name,
            },
        )
    else:
        # If no human review, the next node is fix_code if error, else explain_code.
        if not bypass_explain_code:
            workflow.add_conditional_edges(
                execute_code_node_name,
                lambda s: "fix_code" if error_and_can_retry(s) else "explain_code",
                {
                    "fix_code": fix_code_node_name,
                    "explain_code": explain_code_node_name,
                },
            )
        else:
            workflow.add_conditional_edges(
                execute_code_node_name,
                lambda s: "fix_code" if error_and_can_retry(s) else "END",
                {
                    "fix_code": fix_code_node_name,
                    "END": END,
                },
            )

    if not bypass_explain_code:
        workflow.add_edge(explain_code_node_name, END)

    # Finally, compile
    app = workflow.compile(
        checkpointer=checkpointer,
        name=agent_name,
    )

    return app


def node_func_human_review(
    state: Any,
    prompt_text: str,
    yes_goto: str,
    no_goto: str,
    user_instructions_key: str = "user_instructions",
    recommended_steps_key: str = "recommended_steps",
    code_snippet_key: str = "code_snippet",
    code_type: str = "python",
) -> Command[str]:
    """
    A generic function to handle human review steps.

    Parameters
    ----------
    state : GraphState
        The current GraphState.
    prompt_text : str
        The text to display to the user before their input.
    yes_goto : str
        The node to go to if the user confirms (answers "yes").
    no_goto : str
        The node to go to if the user suggests modifications.
    user_instructions_key : str, optional
        The key in the state to store user instructions.
    recommended_steps_key : str, optional
        The key in the state to store recommended steps.
    code_snippet_key : str, optional
        The key in the state to store the code snippet.
    code_type : str, optional
        The type of code snippet to display (e.g., "python").

    Returns
    -------
    Command[str]
        A Command object directing the next state and updates to the state.
    """
    print("    * HUMAN REVIEW")

    code_markdown = f"```{code_type}\n" + state.get(code_snippet_key) + "\n```"

    # Display instructions and get user response
    user_input = interrupt(
        value=prompt_text.format(
            steps=state.get(recommended_steps_key, "") + "\n\n" + code_markdown
        )
    )

    # Decide next steps based on user input
    if user_input.strip().lower() == "yes":
        goto = yes_goto
        update = {}
    else:
        goto = no_goto
        modifications = (
            "User Has Requested Modifications To Previous Code: \n" + user_input
        )
        if state.get(user_instructions_key) is None:
            update = {
                user_instructions_key: modifications
                + "\n\nPrevious Code:\n"
                + code_markdown
            }
        else:
            update = {
                user_instructions_key: state.get(user_instructions_key)
                + modifications
                + "\n\nPrevious Code:\n"
                + code_markdown
            }

    return Command(goto=goto, update=update)


def node_func_execute_agent_code_on_data(
    state: Any,
    data_key: str,
    code_snippet_key: str,
    result_key: str,
    error_key: str,
    agent_function_name: str,
    pre_processing: Optional[Callable[[Any], Any]] = None,
    post_processing: Optional[Callable[[Any], Any]] = None,
    error_message_prefix: str = "An error occurred during agent execution: ",
) -> Dict[str, Any]:
    """
    Execute a generic agent code defined in a code snippet retrieved from the state on input data and return the result.

    Parameters
    ----------
    state : Any
        A state object that supports `get(key: str)` method to retrieve values.
    data_key : str
        The key in the state used to retrieve the input data.
    code_snippet_key : str
        The key in the state used to retrieve the Python code snippet defining the agent function.
    result_key : str
        The key in the state used to store the result of the agent function.
    error_key : str
        The key in the state used to store the error message if any.
    agent_function_name : str
        The name of the function (e.g., 'data_cleaner') expected to be defined in the code snippet.
    pre_processing : Callable[[Any], Any], optional
        A function to preprocess the data before passing it to the agent function.
        This might be used to convert raw data into a DataFrame or otherwise transform it.
        If not provided, a default approach will be used if data is a dict.
    post_processing : Callable[[Any], Any], optional
        A function to postprocess the output of the agent function before returning it.
    error_message_prefix : str, optional
        A prefix or full message to use in the error output if an exception occurs.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the result and/or error messages. Keys are arbitrary,
        but typically include something like "result" or "error".
    """

    print("    * EXECUTING AGENT CODE")

    # Retrieve raw data and code snippet from the state
    data = state.get(data_key)
    agent_code = state.get(code_snippet_key)

    # Preprocessing: If no pre-processing function is given, attempt a default handling.
    # Important: do not crash the whole team if preprocessing fails; return an error instead.
    try:
        if pre_processing is None:
            if isinstance(data, dict):
                df = pd.DataFrame.from_dict(data)
            elif isinstance(data, list):
                df = [pd.DataFrame.from_dict(item) for item in data]
            else:
                raise ValueError(
                    "Data is not a dictionary or list and no pre_processing function was provided."
                )
        else:
            df = pre_processing(data)
    except Exception as e:
        return {result_key: None, error_key: f"{error_message_prefix}{str(e)}"}

    # Execute the code snippet to define the agent function
    local_vars = {}
    global_vars = {}
    agent_function = None
    exec_error = None
    try:
        exec(agent_code, global_vars, local_vars)
        # Retrieve the agent function from the executed code
        agent_function = local_vars.get(agent_function_name, None)
        if agent_function is None or not callable(agent_function):
            raise ValueError(
                f"Agent function '{agent_function_name}' not found or not callable in the provided code."
            )
    except Exception as e:
        exec_error = f"{error_message_prefix}{str(e)}"

    # If code couldn't be executed/loaded, return early with an error (do not crash the team).
    if exec_error:
        return {result_key: None, error_key: exec_error}

    # Execute the agent function
    agent_error = None
    result = None
    try:
        result = agent_function(df)

        # Test an error
        # if state.get("retry_count") == 0:
        #     10/0

        # Apply post-processing if provided
        if post_processing is not None:
            result = post_processing(result)
        else:
            if isinstance(result, pd.DataFrame):
                result = result.to_dict()

    except Exception as e:
        print(e)
        agent_error = f"{error_message_prefix}{str(e)}"

    # Return results
    output = {result_key: result, error_key: agent_error}
    return output


def node_func_execute_agent_from_sql_connection(
    state: Any,
    connection: Any,
    code_snippet_key: str,
    result_key: str,
    error_key: str,
    agent_function_name: str,
    post_processing: Optional[Callable[[Any], Any]] = None,
    error_message_prefix: str = "An error occurred during agent execution: ",
) -> Dict[str, Any]:
    """
    Execute a generic agent code defined in a code snippet retrieved from the state on a SQLAlchemy connection object
    and return the result.

    Parameters
    ----------
    state : Any
        A state object that supports `get(key: str)` method to retrieve values.
    connection : str
        The SQLAlchemy connection object to use for executing the agent function.
    code_snippet_key : str
        The key in the state used to retrieve the Python code snippet defining the agent function.
    result_key : str
        The key in the state used to store the result of the agent function.
    error_key : str
        The key in the state used to store the error message if any.
    agent_function_name : str
        The name of the function (e.g., 'sql_database_agent') expected to be defined in the code snippet.
    post_processing : Callable[[Any], Any], optional
        A function to postprocess the output of the agent function before returning it.
    error_message_prefix : str, optional
        A prefix or full message to use in the error output if an exception occurs.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the result and/or error messages. Keys are arbitrary,
        but typically include something like "result" or "error".
    """

    print("    * EXECUTING AGENT CODE ON SQL CONNECTION")

    # Retrieve SQLAlchemy connection and code snippet from the state
    is_engine = isinstance(connection, sql.engine.base.Engine)
    connection = connection.connect() if is_engine else connection
    agent_code = state.get(code_snippet_key)

    # Ensure the connection object is provided
    if connection is None:
        raise ValueError(f"Connection object not found.")

    # Execute the code snippet to define the agent function
    local_vars = {}
    global_vars = {}
    exec(agent_code, global_vars, local_vars)

    # Retrieve the agent function from the executed code
    agent_function = local_vars.get(agent_function_name, None)
    if agent_function is None or not callable(agent_function):
        raise ValueError(
            f"Agent function '{agent_function_name}' not found or not callable in the provided code."
        )

    # Execute the agent function
    agent_error = None
    result = None
    try:
        result = agent_function(connection)

        # Apply post-processing if provided
        if post_processing is not None:
            result = post_processing(result)
    except Exception as e:
        print(e)
        agent_error = f"{error_message_prefix}{str(e)}"

    # Return results
    output = {result_key: result, error_key: agent_error}
    return output


def node_func_fix_agent_code(
    state: Any,
    code_snippet_key: str,
    error_key: str,
    llm: Any,
    prompt_template: str,
    agent_name: str,
    retry_count_key: str = "retry_count",
    log: bool = False,
    file_path: str = "logs/agent_function.py",
    function_name: str = "agent_function",
) -> dict:
    """
    Generic function to fix a given piece of agent code using an LLM and a prompt template.

    Parameters
    ----------
    state : Any
        A state object that supports `get(key: str)` method to retrieve values.
    code_snippet_key : str
        The key in the state used to retrieve the broken code snippet.
    error_key : str
        The key in the state used to retrieve the related error message.
    llm : Any
        The language model or pipeline capable of receiving prompts and returning responses.
        It should support a call like `(llm | PythonOutputParser()).invoke(prompt)`.
    prompt_template : str
        A string template for the prompt that will be sent to the LLM. It should contain
        placeholders `{code_snippet}` and `{error}` which will be formatted with the actual values.
    agent_name : str
        The name of the agent being fixed. This is used to add comments to the top of the code.
    retry_count_key : str, optional
        The key in the state that tracks how many times we've retried fixing the code.
    log : bool, optional
        Whether to log the returned code to a file.
    file_path : str, optional
        The path to the file where the code will be logged.
    function_name : str, optional
        The name of the function in the code snippet that will be fixed.

    Returns
    -------
    dict
        A dictionary containing updated code, cleared error, and incremented retry count.
    """
    print("    * FIX AGENT CODE")
    print("      retry_count:" + str(state.get(retry_count_key)))

    # Retrieve the code snippet and the error from the state
    code_snippet = state.get(code_snippet_key)
    error_message = state.get(error_key)

    # Format the prompt with the code snippet and the error
    prompt = prompt_template.format(
        code_snippet=code_snippet,
        error=error_message,
        function_name=function_name,
        user_instructions=state.get("user_instructions"),
        recommended_steps=state.get("recommended_steps"),
    )

    # Execute the prompt with the LLM
    response = (llm | PythonOutputParser()).invoke(prompt)

    response = relocate_imports_inside_function(response)
    response = add_comments_to_top(response, agent_name=agent_name)

    # Log the response if requested
    if log:
        with open(file_path, "w") as file:
            file.write(response)
            print(f"      File saved to: {file_path}")

    # Return updated results
    return {
        code_snippet_key: response,
        error_key: None,
        retry_count_key: state.get(retry_count_key) + 1,
    }


def node_func_explain_agent_code(
    state: Any,
    code_snippet_key: str,
    result_key: str,
    error_key: str,
    llm: Any,
    role: str,
    explanation_prompt_template: str,
    success_prefix: str = "# Agent Explanation:\n\n",
    error_message: str = "The agent encountered an error during execution and cannot be explained.",
) -> Dict[str, Any]:
    """
    Generic function to explain what a given agent code snippet does.

    Parameters
    ----------
    state : Any
        A state object that supports `get(key: str)` to retrieve values.
    code_snippet_key : str
        The key in `state` where the agent code snippet is stored.
    result_key : str
        The key in `state` where the LLM's explanation is stored. Typically this is "messages".
    error_key : str
        The key in `state` where any error messages related to the code snippet are stored.
    llm : Any
        The language model used to explain the code. Should support `.invoke(prompt)`.
    role : str
        The role of the agent explaining the code snippet. Examples: "Data Scientist", "Data Engineer", etc.
    explanation_prompt_template : str
        A prompt template that can be used to explain the code. It should contain a placeholder
        for inserting the agent code snippet. For example:

        "Explain the steps performed by this agent code in a succinct manner:\n\n{code}"

    success_prefix : str, optional
        A prefix to add before the LLM's explanation, helping format the final message.
    error_message : str, optional
        Message to return if the agent code snippet cannot be explained due to an error.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing one key "messages", which is a list of messages (e.g., AIMessage)
        describing the explanation or the error.
    """
    print("    * EXPLAIN AGENT CODE")

    # Check if there's an error associated with the code
    agent_error = state.get(error_key)
    if agent_error is None:
        # Retrieve the code snippet
        code_snippet = state.get(code_snippet_key)

        # Format the prompt by inserting the code snippet
        prompt = explanation_prompt_template.format(code=code_snippet)

        # Invoke the LLM to get an explanation
        response = llm.invoke(prompt)

        # Prepare the success message
        message = AIMessage(content=f"{success_prefix}{response.content}", role=role)
        return {"messages": [message]}
    else:
        # Return an error message if there was a problem with the code
        message = AIMessage(content=error_message)
        return {result_key: [message]}


def node_func_report_agent_outputs(
    state: Dict[str, Any],
    keys_to_include: List[str],
    result_key: str,
    role: str,
    custom_title: str = "Agent Output Summary",
) -> Dict[str, Any]:
    """
    Gathers relevant data directly from the state (filtered by `keys_to_include`)
    and returns them as a structured message in `state[result_key]`.

    No LLM is used.

    Parameters
    ----------
    state : Dict[str, Any]
        The current state dictionary holding all agent variables.
    keys_to_include : List[str]
        The list of keys in `state` to include in the output.
    result_key : str
        The key in `state` under which we'll store the final structured message.
    role : str
        The role that will be used in the final AIMessage (e.g., "DataCleaningAgent").
    custom_title : str, optional
        A title or heading for your report. Defaults to "Agent Output Summary".
    """
    print("    * REPORT AGENT OUTPUTS")

    final_report = {"report_title": custom_title}

    for key in keys_to_include:
        final_report[key] = state.get(key, f"<{key}_not_found_in_state>")

    # Wrap it in a list of messages (like the current "messages" pattern).
    # You can serialize this dictionary as JSON or just cast it to string.
    return {
        result_key: [AIMessage(content=json.dumps(final_report, indent=2), role=role)]
    }
