# BUSINESS SCIENCE UNIVERSITY
# AI DATA SCIENCE TEAM
# ***
# * Agents: Data Visualization Agent


# Libraries
from typing_extensions import TypedDict, Annotated, Sequence, Literal
import operator

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage

from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

import os
import json
import difflib
import re
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
from ai_data_science_team.utils.plotly import plotly_from_dict
from ai_data_science_team.utils.sandbox import run_code_sandboxed_subprocess
from ai_data_science_team.utils.messages import get_last_user_message_content

# Setup
AGENT_NAME = "data_visualization_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")

# Class


class DataVisualizationAgent(BaseAgent):
    """
    Creates a data visualization agent that can generate Plotly charts based on user-defined instructions or
    default visualization steps (if any). The agent generates a Python function to produce the visualization,
    executes it, and logs the process, including code and errors. It is designed to facilitate reproducible
    and customizable data visualization workflows.

    The agent may use default instructions for creating charts unless instructed otherwise, such as:
    - Generating a recommended chart type (bar, scatter, line, etc.)
    - Creating user-friendly titles and axis labels
    - Applying consistent styling (template, font sizes, color themes)
    - Handling theme details (white background, base font size, line size, etc.)

    User instructions can modify, add, or remove any of these steps to tailor the visualization process.

    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model used to generate the data visualization function.
    n_samples : int, optional
        Number of samples used when summarizing the dataset for chart instructions. Defaults to 30.
        Reducing this number can help avoid exceeding the model's token limits.
    log : bool, optional
        Whether to log the generated code and errors. Defaults to False.
    log_path : str, optional
        Directory path for storing log files. Defaults to None.
    file_name : str, optional
        Name of the file for saving the generated response. Defaults to "data_visualization.py".
    function_name : str, optional
        Name of the function for data visualization. Defaults to "data_visualization".
    overwrite : bool, optional
        Whether to overwrite the log file if it exists. If False, a unique file name is created. Defaults to True.
    human_in_the_loop : bool, optional
        Enables user review of data visualization instructions. Defaults to False.
    bypass_recommended_steps : bool, optional
        If True, skips the default recommended visualization steps. Defaults to False.
    bypass_explain_code : bool, optional
        If True, skips the step that provides code explanations. Defaults to False.
    checkpointer : langgraph.types.Checkpointer
        A checkpointer to use for saving and loading the agent

    Methods
    -------
    update_params(**kwargs)
        Updates the agent's parameters and rebuilds the compiled state graph.
    ainvoke_agent(user_instructions: str, data_raw: pd.DataFrame, max_retries=3, retry_count=0)
        Asynchronously generates a visualization based on user instructions.
    invoke_agent(user_instructions: str, data_raw: pd.DataFrame, max_retries=3, retry_count=0)
        Synchronously generates a visualization based on user instructions.
    get_workflow_summary()
        Retrieves a summary of the agent's workflow.
    get_log_summary()
        Retrieves a summary of logged operations if logging is enabled.
    get_plotly_graph()
        Retrieves the Plotly graph (as a dictionary) produced by the agent.
    get_data_raw()
        Retrieves the raw dataset as a pandas DataFrame (based on the last response).
    get_data_visualization_function()
        Retrieves the generated Python function used for data visualization.
    get_recommended_visualization_steps()
        Retrieves the agent's recommended visualization steps.
    get_response()
        Returns the response from the agent as a dictionary.
    show()
        Displays the agent's mermaid diagram.

    Examples
    --------
    ```python
    import pandas as pd
    from langchain_openai import ChatOpenAI
    from ai_data_science_team.agents import DataVisualizationAgent

    llm = ChatOpenAI(model="gpt-4o-mini")

    data_visualization_agent = DataVisualizationAgent(
        model=llm,
        n_samples=30,
        log=True,
        log_path="logs",
        human_in_the_loop=True
    )

    df = pd.read_csv("https://raw.githubusercontent.com/business-science/ai-data-science-team/refs/heads/master/data/churn_data.csv")

    data_visualization_agent.invoke_agent(
        user_instructions="Generate a scatter plot of age vs. total charges with a trend line.",
        data_raw=df,
        max_retries=3,
        retry_count=0
    )

    plotly_graph_dict = data_visualization_agent.get_plotly_graph()
    # You can render plotly_graph_dict with plotly.io.from_json or
    # something similar in a Jupyter Notebook.

    response = data_visualization_agent.get_response()
    ```

    Returns
    --------
    DataVisualizationAgent : langchain.graphs.CompiledStateGraph
        A data visualization agent implemented as a compiled state graph.
    """

    def __init__(
        self,
        model,
        n_samples=30,
        log=False,
        log_path=None,
        file_name="data_visualization.py",
        function_name="data_visualization",
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
        Create the compiled graph for the data visualization agent.
        Running this method will reset the response to None.
        """
        self.response = None
        return make_data_visualization_agent(**self._params)

    def update_params(self, **kwargs):
        """
        Updates the agent's parameters and rebuilds the compiled graph.
        """
        # Update parameters
        for k, v in kwargs.items():
            self._params[k] = v
        # Rebuild the compiled graph
        self._compiled_graph = self._make_compiled_graph()

    async def ainvoke_agent(
        self,
        data_raw: pd.DataFrame,
        user_instructions: str = None,
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
        """
        Asynchronously invokes the agent to generate a visualization.
        The response is stored in the 'response' attribute.

        Parameters
        ----------
        data_raw : pd.DataFrame
            The raw dataset to be visualized.
        user_instructions : str
            Instructions for data visualization.
        max_retries : int
            Maximum retry attempts.
        retry_count : int
            Current retry attempt count.
        **kwargs : dict
            Additional keyword arguments passed to ainvoke().

        Returns
        -------
        None
        """
        response = await self._compiled_graph.ainvoke(
            {
                "messages": [("user", user_instructions)] if user_instructions else [],
                "user_instructions": user_instructions,
                "data_raw": data_raw.to_dict(),
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
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
        """
        Synchronously invokes the agent to generate a visualization.
        The response is stored in the 'response' attribute.

        Parameters
        ----------
        data_raw : pd.DataFrame
            The raw dataset to be visualized.
        user_instructions : str
            Instructions for data visualization agent.
        max_retries : int
            Maximum retry attempts.
        retry_count : int
            Current retry attempt count.
        **kwargs : dict
            Additional keyword arguments passed to invoke().

        Returns
        -------
        None
        """
        response = self._compiled_graph.invoke(
            {
                "messages": [("user", user_instructions)] if user_instructions else [],
                "user_instructions": user_instructions,
                "data_raw": data_raw.to_dict(),
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
        max_retries: int = 3,
        retry_count: int = 0,
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
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
        """
        Async version of invoke_messages for supervisors/teams.
        """
        user_instructions = kwargs.pop("user_instructions", None)
        if user_instructions is None:
            user_instructions = get_last_user_message_content(messages)
        response = await self._compiled_graph.ainvoke(
            {
                "messages": messages,
                "user_instructions": user_instructions,
                "data_raw": data_raw.to_dict(),
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        self.response = response
        return None

    def run_smoke_tests(
        self,
        data_raw: pd.DataFrame,
        prompts: list[str] | None = None,
        max_retries: int = 1,
    ) -> dict:
        """
        Run a small suite of visualization prompts and report pass/fail.
        Intended for quick sanity checks after agent updates.
        """
        if not isinstance(data_raw, pd.DataFrame):
            return {
                "passed": False,
                "error": "data_raw must be a pandas DataFrame",
            }
        test_prompts = prompts or [
            "Plot a histogram of a numeric column.",
            "Create a bar chart for a categorical column.",
            "Make a scatter plot between two numeric columns.",
            "Create a line chart over a date/time column.",
            "Make a violin+box plot grouped by a categorical column.",
        ]
        results = []
        for prompt in test_prompts:
            try:
                self.invoke_agent(
                    data_raw=data_raw,
                    user_instructions=prompt,
                    max_retries=max_retries,
                    retry_count=0,
                )
                resp = self.response or {}
                results.append(
                    {
                        "prompt": prompt,
                        "has_plot": bool(resp.get("plotly_graph")),
                        "error": resp.get("data_visualization_error"),
                        "warning": resp.get("data_visualization_warning"),
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "prompt": prompt,
                        "has_plot": False,
                        "error": str(exc),
                        "warning": None,
                    }
                )
        passed = all(r.get("has_plot") and not r.get("error") for r in results)
        return {"passed": passed, "results": results}

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
            if self.response.get("data_visualization_function_path"):
                log_details = f"""
## Data Visualization Agent Log Summary:

Function Path: {self.response.get("data_visualization_function_path")}

Function Name: {self.response.get("data_visualization_function_name")}
                """
                if markdown:
                    return Markdown(log_details)
                else:
                    return log_details

    def get_plotly_graph(self):
        """
        Retrieves the Plotly graph (in dictionary form) produced by the agent.

        Returns
        -------
        dict or None
            The Plotly graph dictionary if available, otherwise None.
        """
        if self.response:
            return plotly_from_dict(self.response.get("plotly_graph", None))
        return None

    def get_data_raw(self):
        """
        Retrieves the raw dataset used in the last invocation.

        Returns
        -------
        pd.DataFrame or None
            The raw dataset as a DataFrame if available, otherwise None.
        """
        if self.response and self.response.get("data_raw"):
            return pd.DataFrame(self.response.get("data_raw"))
        return None

    def get_data_visualization_function(self, markdown=False):
        """
        Retrieves the generated Python function used for data visualization.

        Parameters
        ----------
        markdown : bool, optional
            If True, returns the function in Markdown code block format.

        Returns
        -------
        str or None
            The Python function code as a string if available, otherwise None.
        """
        if self.response:
            func_code = self.response.get("data_visualization_function", "")
            if markdown:
                return Markdown(f"```python\n{func_code}\n```")
            return func_code
        return None

    def get_recommended_visualization_steps(self, markdown=False):
        """
        Retrieves the agent's recommended visualization steps.

        Parameters
        ----------
        markdown : bool, optional
            If True, returns the steps in Markdown format.

        Returns
        -------
        str or None
            The recommended steps if available, otherwise None.
        """
        if self.response:
            steps = self.response.get("recommended_steps", "")
            if markdown:
                return Markdown(steps)
            return steps
        return None

    def get_response(self):
        """
        Returns the agent's full response dictionary.

        Returns
        -------
        dict or None
            The response dictionary if available, otherwise None.
        """
        return self.response

    def show(self):
        """
        Displays the agent's mermaid diagram for visual inspection of the compiled graph.
        """
        return self._compiled_graph.show()


# Agent


def make_data_visualization_agent(
    model,
    n_samples=30,
    log=False,
    log_path=None,
    file_name="data_visualization.py",
    function_name="data_visualization",
    overwrite=True,
    human_in_the_loop=False,
    bypass_recommended_steps=False,
    bypass_explain_code=False,
    checkpointer=None,
):
    """
    Creates a data visualization agent that can generate Plotly charts based on user-defined instructions or
    default visualization steps. The agent generates a Python function to produce the visualization, executes it,
    and logs the process, including code and errors. It is designed to facilitate reproducible and customizable
    data visualization workflows.

    The agent can perform the following default visualization steps unless instructed otherwise:
    - Generating a recommended chart type (bar, scatter, line, etc.)
    - Creating user-friendly titles and axis labels
    - Applying consistent styling (template, font sizes, color themes)
    - Handling theme details (white background, base font size, line size, etc.)

    User instructions can modify, add, or remove any of these steps to tailor the visualization process.

    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model used to generate the data visualization function.
    n_samples : int, optional
        Number of samples used when summarizing the dataset for chart instructions. Defaults to 30.
    log : bool, optional
        Whether to log the generated code and errors. Defaults to False.
    log_path : str, optional
        Directory path for storing log files. Defaults to None.
    file_name : str, optional
        Name of the file for saving the generated response. Defaults to "data_visualization.py".
    function_name : str, optional
        Name of the function for data visualization. Defaults to "data_visualization".
    overwrite : bool, optional
        Whether to overwrite the log file if it exists. If False, a unique file name is created. Defaults to True.
    human_in_the_loop : bool, optional
        Enables user review of data visualization instructions. Defaults to False.
    bypass_recommended_steps : bool, optional
        If True, skips the default recommended visualization steps. Defaults to False.
    bypass_explain_code : bool, optional
        If True, skips the step that provides code explanations. Defaults to False.
    checkpointer : langgraph.types.Checkpointer
        A checkpointer to use for saving and loading the agent

    Examples
    --------
    ``` python
    import pandas as pd
    from langchain_openai import ChatOpenAI
    from ai_data_science_team.agents import data_visualization_agent

    llm = ChatOpenAI(model="gpt-4o-mini")

    data_visualization_agent = make_data_visualization_agent(llm)

    df = pd.read_csv("https://raw.githubusercontent.com/business-science/ai-data-science-team/refs/heads/master/data/churn_data.csv")

    response = data_visualization_agent.invoke({
        "user_instructions": "Generate a scatter plot of tenure vs. total charges with a trend line.",
        "data_raw": df.to_dict(),
        "max_retries": 3,
        "retry_count": 0
    })

    pd.DataFrame(response['plotly_graph'])
    ```

    Returns
    -------
    app : langchain.graphs.CompiledStateGraph
        The data visualization agent as a state graph.
    """

    llm = model

    MAX_SUMMARY_COLUMNS = 30
    MAX_SUMMARY_CHARS = 5000

    DEFAULT_VISUALIZATION_INSTRUCTIONS = """
Use an appropriate chart type based on column types (categorical vs numeric). Derive columns from the provided schema; do not hardcode names. Handle missing values gracefully. Prefer plotly express for simplicity. Do not save files or print; just return a JSON-serializable Plotly figure dictionary. Always set chart title and axis labels; use unit hints when available.
    """

    def _normalize_column_name(value: str) -> str:
        if not isinstance(value, str):
            return ""
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    def _profile_dataframe(df: pd.DataFrame) -> dict:
        df = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        n_rows = int(getattr(df, "shape", (0, 0))[0] or 0)
        sample = df
        if n_rows > 5000:
            sample = df.head(5000)

        columns = [str(c) for c in list(sample.columns)]
        numeric_cols: list[str] = []
        categorical_cols: list[str] = []
        datetime_cols: list[str] = []
        boolean_cols: list[str] = []
        low_card_numeric: list[str] = []
        high_card_categorical: list[str] = []

        for col in columns:
            s = sample[col]
            try:
                nunique = int(s.nunique(dropna=True))
            except Exception:
                nunique = 0

            if pd.api.types.is_bool_dtype(s):
                boolean_cols.append(col)
                categorical_cols.append(col)
                continue
            if pd.api.types.is_datetime64_any_dtype(s):
                datetime_cols.append(col)
                continue
            if pd.api.types.is_numeric_dtype(s):
                numeric_cols.append(col)
                if nunique <= 10:
                    low_card_numeric.append(col)
                    categorical_cols.append(col)
                continue

            categorical_cols.append(col)
            if nunique >= max(20, int(0.2 * max(n_rows, 1))):
                high_card_categorical.append(col)

        return {
            "n_rows": n_rows,
            "columns": columns,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "datetime_cols": datetime_cols,
            "boolean_cols": boolean_cols,
            "low_cardinality_numeric": low_card_numeric,
            "high_cardinality_categorical": high_card_categorical,
        }

    def _infer_units(columns: list[str]) -> dict[str, str]:
        units = {}
        for col in columns:
            col_lower = col.lower()
            unit = None
            if "%" in col_lower or "pct" in col_lower or "percent" in col_lower:
                unit = "%"
            elif "usd" in col_lower or "price" in col_lower or "amount" in col_lower:
                unit = "USD"
            elif "cost" in col_lower or "charge" in col_lower:
                unit = "USD"
            elif "date" in col_lower or "time" in col_lower:
                unit = "date/time"
            elif "age" in col_lower:
                unit = "years"
            elif col_lower.endswith("_id") or col_lower == "id":
                unit = None
            if unit:
                units[col] = unit
        return units

    def _format_profile_for_prompt(profile: dict) -> str:
        if not isinstance(profile, dict):
            return ""
        def _fmt(values: list[str]) -> str:
            return ", ".join(values[:12]) if values else "None"
        return "\n".join(
            [
                f"Rows: {profile.get('n_rows')}",
                f"Numeric: {_fmt(profile.get('numeric_cols') or [])}",
                f"Categorical: {_fmt(profile.get('categorical_cols') or [])}",
                f"Datetime: {_fmt(profile.get('datetime_cols') or [])}",
                f"Boolean: {_fmt(profile.get('boolean_cols') or [])}",
                f"Low-card numeric: {_fmt(profile.get('low_cardinality_numeric') or [])}",
                f"High-card categorical: {_fmt(profile.get('high_cardinality_categorical') or [])}",
            ]
        )

    def _format_units_for_prompt(units: dict[str, str]) -> str:
        if not isinstance(units, dict) or not units:
            return "None"
        items = [f"{k} -> {v}" for k, v in list(units.items())[:12]]
        return ", ".join(items)

    def _resolve_column_aliases(text: str, columns: list[str]) -> dict[str, str]:
        if not isinstance(text, str) or not text.strip():
            return {}
        columns = [str(c) for c in columns if isinstance(c, str)]
        if not columns:
            return {}
        col_norm_map = {c: _normalize_column_name(c) for c in columns}
        tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
        candidates = set(tokens)
        for i in range(len(tokens) - 1):
            candidates.add(tokens[i] + tokens[i + 1])
            candidates.add(f"{tokens[i]}_{tokens[i + 1]}")
        aliases: dict[str, str] = {}
        for cand in list(candidates):
            cand_norm = _normalize_column_name(cand)
            if not cand_norm or len(cand_norm) < 3:
                continue
            best = None
            best_score = 0.0
            for col, col_norm in col_norm_map.items():
                if not col_norm:
                    continue
                if cand_norm == col_norm or cand_norm in col_norm:
                    best = col
                    best_score = 1.0
                    break
                score = difflib.SequenceMatcher(None, cand_norm, col_norm).ratio()
                if score > best_score:
                    best_score = score
                    best = col
            if best and best_score >= 0.82:
                aliases[cand] = best
        return aliases

    def _format_aliases_for_prompt(aliases: dict[str, str]) -> str:
        if not isinstance(aliases, dict) or not aliases:
            return "None"
        items = [f"{k} -> {v}" for k, v in list(aliases.items())[:12]]
        return ", ".join(items)

    def _build_prompt_context(
        df: pd.DataFrame, user_text: str | None
    ) -> tuple[str, dict]:
        base = _summarize_df_for_prompt(df)
        profile = _profile_dataframe(df)
        units = _infer_units(profile.get("columns") or [])
        aliases = _resolve_column_aliases(user_text or "", profile.get("columns") or [])
        sections = [
            base,
            "COLUMN PROFILE:\n" + _format_profile_for_prompt(profile),
            "COLUMN ALIASES (user -> dataset):\n" + _format_aliases_for_prompt(aliases),
            "UNIT HINTS:\n" + _format_units_for_prompt(units),
        ]
        context = "\n\n".join([s for s in sections if s])
        return context[:MAX_SUMMARY_CHARS], profile

    def _label_for_column(col: str, units: dict[str, str]) -> str:
        label = str(col).replace("_", " ").strip().title()
        unit = units.get(col)
        if unit:
            label = f"{label} ({unit})"
        return label

    def _extract_missing_columns(error_text: str) -> list[str]:
        if not isinstance(error_text, str) or not error_text:
            return []
        missing = set()
        for match in re.findall(r"KeyError:\\s*['\\\"]([^'\\\"]+)['\\\"]", error_text):
            missing.add(match)
        for match in re.findall(r"\\['([^']+)'\\]\\s+not in index", error_text):
            missing.add(match)
        list_match = re.search(r"None of \\[(.*)\\] are in the \\[columns\\]", error_text)
        if list_match:
            raw = list_match.group(1)
            for col in re.findall(r"'([^']+)'", raw):
                missing.add(col)
        return [m for m in missing if isinstance(m, str) and m.strip()]

    def _suggest_column_fallbacks(missing: list[str], columns: list[str]) -> dict[str, str]:
        suggestions = {}
        if not missing or not columns:
            return suggestions
        norm_cols = {c: _normalize_column_name(c) for c in columns}
        for miss in missing:
            miss_norm = _normalize_column_name(miss)
            if not miss_norm:
                continue
            best = None
            best_score = 0.0
            for col, col_norm in norm_cols.items():
                if not col_norm:
                    continue
                score = difflib.SequenceMatcher(None, miss_norm, col_norm).ratio()
                if score > best_score:
                    best_score = score
                    best = col
            if best and best_score >= 0.7:
                suggestions[miss] = best
        return suggestions

    def _patch_missing_columns(code: str, mapping: dict[str, str]) -> tuple[str, bool]:
        if not isinstance(code, str) or not mapping:
            return code, False
        patched = code
        for old, new in mapping.items():
            if not isinstance(old, str) or not isinstance(new, str):
                continue
            patched = re.sub(rf"'{re.escape(old)}'", f"'{new}'", patched)
            patched = re.sub(rf"\\\"{re.escape(old)}\\\"", f'\"{new}\"', patched)
        return patched, patched != code

    def _build_fallback_chart(df: pd.DataFrame, profile: dict) -> tuple[dict | None, str | None]:
        try:
            import plotly.express as px
            import plotly.io as pio
            import json as _json
        except Exception:
            return None, "Plotly is not available for fallback."

        if not isinstance(df, pd.DataFrame) or df.empty:
            return None, "No data available for fallback."
        sample = df.head(5000)
        units = _infer_units(profile.get("columns") or [])
        numeric_cols = profile.get("numeric_cols") or []
        categorical_cols = profile.get("categorical_cols") or []
        datetime_cols = profile.get("datetime_cols") or []

        fig = None
        note = None
        if datetime_cols and numeric_cols:
            x = datetime_cols[0]
            y = numeric_cols[0]
            fig = px.line(
                sample,
                x=x,
                y=y,
                labels={x: _label_for_column(x, units), y: _label_for_column(y, units)},
                title=f"{_label_for_column(y, units)} over {_label_for_column(x, units)}",
            )
            note = f"Fallback line chart using {x} vs {y}."
        elif categorical_cols and numeric_cols:
            x = categorical_cols[0]
            y = numeric_cols[0]
            fig = px.bar(
                sample,
                x=x,
                y=y,
                labels={x: _label_for_column(x, units), y: _label_for_column(y, units)},
                title=f"{_label_for_column(y, units)} by {_label_for_column(x, units)}",
            )
            note = f"Fallback bar chart using {x} vs {y}."
        elif numeric_cols:
            x = numeric_cols[0]
            fig = px.histogram(
                sample,
                x=x,
                labels={x: _label_for_column(x, units)},
                title=f"Distribution of {_label_for_column(x, units)}",
            )
            note = f"Fallback histogram using {x}."
        elif categorical_cols:
            x = categorical_cols[0]
            fig = px.bar(
                sample,
                x=x,
                labels={x: _label_for_column(x, units)},
                title=f"Counts by {_label_for_column(x, units)}",
            )
            note = f"Fallback bar chart using {x}."
        if fig is None:
            return None, "No suitable columns found for fallback."
        fig_dict = _json.loads(pio.to_json(fig))
        return fig_dict, note

    def _summarize_df_for_prompt(df: pd.DataFrame) -> str:
        df_limited = (
            df.iloc[:, :MAX_SUMMARY_COLUMNS] if df.shape[1] > MAX_SUMMARY_COLUMNS else df
        )
        summary = "\n\n".join(
            get_dataframe_summary(
                [df_limited],
                n_sample=min(n_samples, 5),
                skip_stats=True,
            )
        )
        return summary[:MAX_SUMMARY_CHARS]

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

    # Define GraphState for the router
    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        user_instructions_processed: str
        recommended_steps: str
        data_raw: dict
        plotly_graph: dict
        all_datasets_summary: str
        data_visualization_function: str
        data_visualization_function_path: str
        data_visualization_function_file_name: str
        data_visualization_function_name: str
        data_visualization_error: str
        data_visualization_error_log_path: str
        data_visualization_summary: str
        data_visualization_warning: str
        max_retries: int
        retry_count: int

    def chart_instructor(state: GraphState):
        print(format_agent_name(AGENT_NAME))
        print("    * CREATE CHART GENERATOR INSTRUCTIONS")

        recommend_steps_prompt = PromptTemplate(
            template="""
            You are a supervisor that is an expert in providing instructions to a chart generator agent for plotting. 
    
            You will take a question that a user has and the data that was generated to answer the question, and create instructions to create a chart from the data that will be passed to a chart generator agent.
            
            USER QUESTION / INSTRUCTIONS: 
            {user_instructions}
            
            Previously Recommended Instructions (if any):
            {recommended_steps}
            
            DATA SUMMARY: 
            {all_datasets_summary}

            IMPORTANT:
            
            - Formulate chart generator instructions by informing the chart generator of what type of plotly plot to use (e.g. bar, line, scatter, etc) to best represent the data. 
            - Think about how best to convey the information in the data to the user.
            - The data summary includes COLUMN PROFILE, COLUMN ALIASES, and UNIT HINTS; use them when selecting columns and labeling axes.
            - If the user specifies a chart type (e.g., violin, box, scatter, histogram, line), you MUST use that chart type. Do NOT substitute a different chart type.
            - If the user does not specify a type of plot, select the appropriate chart type based on the data summary provided and the user's question and how best to show the results.
            - Come up with an informative title from the user's question and data provided. Also provide X and Y axis titles.
            - If the user requests a combined \"violin+box\" plot, instruct the generator to use a violin plot with an embedded box plot (e.g., `plotly.express.violin(..., box=True)`).
            - Only use columns present in the schema (or the alias map). Never guess column names.
            
            CHART TYPE SELECTION TIPS:
            
            - If a numeric column has less than 10 unique values, consider this column to be treated as a categorical column. Pick a chart that is appropriate for categorical data.
            - If a numeric column has more than 10 unique values, consider this column to be treated as a continuous column. Pick a chart that is appropriate for continuous data.       
            
            
            RETURN FORMAT:
            
            Return your instructions in the following format:
            CHART GENERATOR INSTRUCTIONS: 
            FILL IN THE INSTRUCTIONS HERE
            
            Avoid these:
            1. Do not include steps to save files.
            2. Do not include unrelated user instructions that are not related to the chart generation.
            """,
            input_variables=[
                "user_instructions",
                "recommended_steps",
                "all_datasets_summary",
            ],
        )

        data_raw = state.get("data_raw")
        df = pd.DataFrame.from_dict(data_raw)
        user_text = " ".join(
            [
                str(state.get("user_instructions") or ""),
                str(state.get("recommended_steps") or ""),
            ]
        ).strip()
        all_datasets_summary_str, _profile = _build_prompt_context(df, user_text)

        chart_instructor = recommend_steps_prompt | llm

        recommended_steps = chart_instructor.invoke(
            {
                "user_instructions": state.get("user_instructions"),
                "recommended_steps": state.get("recommended_steps"),
                "all_datasets_summary": all_datasets_summary_str,
            }
        )

        return {
            "recommended_steps": format_recommended_steps(
                recommended_steps.content.strip(),
                heading="# Recommended Data Visualization Steps:",
            ),
            "all_datasets_summary": all_datasets_summary_str,
        }

    def chart_generator(state: GraphState):
        print("    * CREATE DATA VISUALIZATION CODE")

        if bypass_recommended_steps:
            print(format_agent_name(AGENT_NAME))

            data_raw = state.get("data_raw")
            df = pd.DataFrame.from_dict(data_raw)

            user_text = str(state.get("user_instructions") or "").strip()
            all_datasets_summary_str, _profile = _build_prompt_context(df, user_text)

            chart_generator_instructions = (
                state.get("user_instructions") or DEFAULT_VISUALIZATION_INSTRUCTIONS
            )

        else:
            all_datasets_summary_str = state.get("all_datasets_summary")
            chart_generator_instructions = (
                state.get("recommended_steps") or DEFAULT_VISUALIZATION_INSTRUCTIONS
            )

        prompt_template = PromptTemplate(
            template="""
            You are a chart generator agent that is an expert in generating plotly charts. You must use plotly or plotly.express to produce plots.
    
            Your job is to produce python code to generate visualizations with a function named {function_name}.
            
            You will take instructions from a Chart Instructor and generate a plotly chart from the data provided.
            
            CHART INSTRUCTIONS: 
            {chart_generator_instructions}
            
            DATA: 
            {all_datasets_summary}

            IMPORTANT:
            - The data summary includes COLUMN PROFILE, COLUMN ALIASES, and UNIT HINTS; use them for column selection and axis labels.
            - Only use columns present in the schema (or alias map). Never guess column names.
            
            RETURN:
            
            Return Python code in ```python ``` format with a single function definition, {function_name}(data_raw), that includes all imports inside the function.
            
            Return the plotly chart as a dictionary.
            
            Return code to provide the data visualization function:
            
            def {function_name}(data_raw):
                import pandas as pd
                import numpy as np
                import json
                import plotly.graph_objects as go
                import plotly.io as pio
                
                ...
                
                fig_json = pio.to_json(fig)
                fig_dict = json.loads(fig_json)
                
                return fig_dict
            
            Avoid these:
            1. Do not include steps to save files.
            2. Do not include unrelated user instructions that are not related to the chart generation.
            
            """,
            input_variables=[
                "chart_generator_instructions",
                "all_datasets_summary",
                "function_name",
            ],
        )

        data_visualization_agent = prompt_template | llm | PythonOutputParser()

        response = data_visualization_agent.invoke(
            {
                "chart_generator_instructions": chart_generator_instructions,
                "all_datasets_summary": all_datasets_summary_str,
                "function_name": function_name,
            }
        )

        response = relocate_imports_inside_function(response)
        response = add_comments_to_top(response, agent_name=AGENT_NAME)

        # For logging: store the code generated:
        file_path, file_name_2 = log_ai_function(
            response=response,
            file_name=file_name,
            log=log,
            log_path=log_path,
            overwrite=overwrite,
        )

        return {
            "data_visualization_function": response,
            "data_visualization_function_path": file_path,
            "data_visualization_function_file_name": file_name_2,
            "data_visualization_function_name": function_name,
            "all_datasets_summary": all_datasets_summary_str,
        }

    # Human Review

    prompt_text_human_review = "Are the following data visualization instructions correct? (Answer 'yes' or provide modifications)\n{steps}"

    if not bypass_explain_code:

        def human_review(
            state: GraphState,
        ) -> Command[Literal["chart_instructor", "report_agent_outputs"]]:
            return node_func_human_review(
                state=state,
                prompt_text=prompt_text_human_review,
                yes_goto="report_agent_outputs",
                no_goto="chart_instructor",
                user_instructions_key="user_instructions",
                recommended_steps_key="recommended_steps",
                code_snippet_key="data_visualization_function",
            )
    else:

        def human_review(
            state: GraphState,
        ) -> Command[Literal["chart_instructor", "__end__"]]:
            return node_func_human_review(
                state=state,
                prompt_text=prompt_text_human_review,
                yes_goto="__end__",
                no_goto="chart_instructor",
                user_instructions_key="user_instructions",
                recommended_steps_key="recommended_steps",
                code_snippet_key="data_visualization_function",
            )

    def execute_data_visualization_code(state):
        print("    * EXECUTE DATA VISUALIZATION CODE (SANDBOXED)")

        data_raw = state.get("data_raw") or {}
        df = pd.DataFrame.from_dict(data_raw) if isinstance(data_raw, dict) else pd.DataFrame()
        profile = _profile_dataframe(df)

        code_snippet = state.get("data_visualization_function")
        result, error = run_code_sandboxed_subprocess(
            code_snippet=code_snippet,
            function_name=state.get("data_visualization_function_name"),
            data=state.get("data_raw"),
            timeout=15,
            memory_limit_mb=512,
            data_format="dataframe",
        )

        warning_message = None
        patched_code_used = False
        if error:
            missing_cols = _extract_missing_columns(error)
            if missing_cols:
                suggestions = _suggest_column_fallbacks(
                    missing_cols, profile.get("columns") or []
                )
                patched_code, changed = _patch_missing_columns(
                    code_snippet or "", suggestions
                )
                if changed:
                    result, error = run_code_sandboxed_subprocess(
                        code_snippet=patched_code,
                        function_name=state.get("data_visualization_function_name"),
                        data=state.get("data_raw"),
                        timeout=15,
                        memory_limit_mb=512,
                        data_format="dataframe",
                    )
                    if error is None and suggestions:
                        replaced = ", ".join(
                            [f"{k} -> {v}" for k, v in suggestions.items()]
                        )
                        warning_message = (
                            "Auto-substituted missing columns: " + replaced
                        )
                        patched_code_used = True
                        code_snippet = patched_code

        validation_error = error
        viz_summary = None

        if error is None:
            try:
                fig = plotly_from_dict(result)
                if fig is None:
                    validation_error = "Plotly figure could not be reconstructed."
                else:
                    traces = len(fig.data) if hasattr(fig, "data") else 0
                    viz_summary = f"Plotly figure with {traces} trace(s) generated."
                    # Validate chart type against explicit user request when possible.
                    req_raw = state.get("user_instructions") or ""
                    if isinstance(req_raw, str) and "[Pipeline Studio context]" in req_raw:
                        req_raw = req_raw.split("[Pipeline Studio context]", 1)[0]
                    req = req_raw.lower() if isinstance(req_raw, str) else ""
                    if req:
                        import re

                        def _has_word(word: str) -> bool:
                            return re.search(rf"\\b{re.escape(word)}s?\\b", req) is not None

                        expected = set()
                        if _has_word("violin"):
                            expected.add("violin")
                        if _has_word("box") or "boxplot" in req:
                            expected.add("box")
                        if re.search(r"\\bhist(ogram)?\\b", req):
                            expected.add("histogram")
                        if _has_word("scatter"):
                            expected.add("scatter")
                        if _has_word("heatmap"):
                            expected.add("heatmap")
                        if _has_word("bar") or "barplot" in req:
                            expected.add("bar")
                        if _has_word("line") or "line chart" in req:
                            expected.add("line")

                        actual_types = {
                            getattr(t, "type", None)
                            for t in getattr(fig, "data", []) or []
                            if getattr(t, "type", None)
                        }

                        def has_line() -> bool:
                            for t in getattr(fig, "data", []) or []:
                                if getattr(t, "type", None) == "scatter":
                                    mode = (getattr(t, "mode", "") or "").lower()
                                    if "lines" in mode:
                                        return True
                            return False

                        mismatch = None
                        if "violin" in expected and "violin" not in actual_types:
                            mismatch = "violin"
                        elif "box" in expected and "violin" not in expected and "box" not in actual_types:
                            mismatch = "box"
                        elif "histogram" in expected and "histogram" not in actual_types:
                            mismatch = "histogram"
                        elif "bar" in expected and "bar" not in actual_types:
                            mismatch = "bar"
                        elif "heatmap" in expected and "heatmap" not in actual_types:
                            mismatch = "heatmap"
                        elif "scatter" in expected and "scatter" not in actual_types:
                            mismatch = "scatter"
                        elif "line" in expected and not has_line():
                            mismatch = "line"

                        if mismatch:
                            got = (
                                ", ".join(sorted(actual_types))
                                if actual_types
                                else "unknown"
                            )
                            warning_message = (
                                "Chart type warning. "
                                f"User requested '{mismatch}' style, but got '{got}'. "
                                f"User instructions: {state.get('user_instructions')!r}"
                            )
            except Exception as exc:
                validation_error = f"Plotly reconstruction failed: {exc}"

        if validation_error:
            fallback_fig, fallback_note = _build_fallback_chart(df, profile)
            if fallback_fig is not None:
                result = fallback_fig
                validation_error = None
                note = fallback_note or "Fallback chart used."
                if warning_message:
                    warning_message = f"{warning_message}\n{note}"
                else:
                    warning_message = note

        error_prefixed = (
            f"An error occurred during data visualization: {validation_error}"
            if validation_error
            else None
        )

        error_log_path = None
        if error_prefixed and log:
            error_log_path = log_ai_error(
                error_message=error_prefixed,
                file_name=f"{file_name}_errors.log",
                log=log,
                log_path=log_path if log_path is not None else LOG_PATH,
                overwrite=False,
            )
            if error_log_path:
                print(f"      Error logged to: {error_log_path}")

        output = {
            "plotly_graph": result if error_prefixed is None else None,
            "data_visualization_error": error_prefixed,
            "data_visualization_error_log_path": error_log_path,
            "data_visualization_summary": viz_summary,
            "data_visualization_warning": warning_message,
        }
        if patched_code_used and isinstance(code_snippet, str) and code_snippet.strip():
            output["data_visualization_function"] = code_snippet
        return output

    def fix_data_visualization_code(state: GraphState):
        prompt = """
        You are a Data Visualization Agent. Your job is to create a {function_name}() function that can be run on the data provided. The function is currently broken and needs to be fixed.
        
        Make sure to only return the function definition for {function_name}().
        
        Return Python code in ```python``` format with a single function definition, {function_name}(data_raw), that includes all imports inside the function.
        
        This is the broken code (please fix): 
        {code_snippet}

        User instructions:
        {user_instructions}

        Recommended steps (if any):
        {recommended_steps}

        Last Known Error:
        {error}
        """

        return node_func_fix_agent_code(
            state=state,
            code_snippet_key="data_visualization_function",
            error_key="data_visualization_error",
            llm=llm,
            prompt_template=prompt,
            agent_name=AGENT_NAME,
            log=log,
            file_path=state.get("data_visualization_function_path"),
            function_name=state.get("data_visualization_function_name"),
        )

    # Final reporting node
    def report_agent_outputs(state: GraphState):
        return node_func_report_agent_outputs(
            state=state,
            keys_to_include=[
                "recommended_steps",
                "data_visualization_function",
                "data_visualization_function_path",
                "data_visualization_function_name",
                "data_visualization_error",
                "data_visualization_error_log_path",
                "data_visualization_summary",
                "data_visualization_warning",
            ],
            result_key="messages",
            role=AGENT_NAME,
            custom_title="Data Visualization Agent Outputs",
        )

    # Define the graph
    node_functions = {
        "chart_instructor": chart_instructor,
        "human_review": human_review,
        "chart_generator": chart_generator,
        "execute_data_visualization_code": execute_data_visualization_code,
        "fix_data_visualization_code": fix_data_visualization_code,
        "report_agent_outputs": report_agent_outputs,
    }

    app = create_coding_agent_graph(
        GraphState=GraphState,
        node_functions=node_functions,
        recommended_steps_node_name="chart_instructor",
        create_code_node_name="chart_generator",
        execute_code_node_name="execute_data_visualization_code",
        fix_code_node_name="fix_data_visualization_code",
        explain_code_node_name="report_agent_outputs",
        error_key="data_visualization_error",
        human_in_the_loop=human_in_the_loop,  # or False
        human_review_node_name="human_review",
        checkpointer=checkpointer,
        bypass_recommended_steps=bypass_recommended_steps,
        bypass_explain_code=bypass_explain_code,
        agent_name=AGENT_NAME,
    )

    return app
