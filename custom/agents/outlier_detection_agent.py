"""
Outlier Detection Agent

An AI agent specialized in detecting, analyzing, and treating outliers in datasets.
Supports multiple detection methods and provides treatment recommendations.
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from typing import List, Optional, Dict, Any
import pandas as pd
import json

from custom.tools.outlier_detection import (
    detect_outliers_zscore,
    detect_outliers_iqr,
    detect_outliers_isolation_forest,
    detect_outliers_lof,
    visualize_outliers,
    suggest_outlier_treatment,
)


class OutlierDetectionAgent:
    """
    AI Agent for comprehensive outlier detection and treatment.

    This agent can:
    - Detect outliers using multiple methods (Z-score, IQR, Isolation Forest, LOF)
    - Visualize outliers in the data
    - Compare detection methods
    - Suggest appropriate treatment strategies
    - Handle both univariate and multivariate outliers

    Args:
        model: The language model to use (e.g., ChatOpenAI)
        verbose: Whether to print detailed execution logs
        **kwargs: Additional arguments

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> import pandas as pd
        >>>
        >>> # Load data
        >>> df = pd.read_csv("data.csv")
        >>>
        >>> # Initialize agent
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> agent = OutlierDetectionAgent(model=llm)
        >>>
        >>> # Detect outliers
        >>> result = agent.detect_outliers(
        ...     data=df,
        ...     columns=['price', 'age', 'income'],
        ...     methods=['zscore', 'iqr', 'isolation_forest']
        ... )
    """

    def __init__(self, model, verbose: bool = True, **kwargs):
        self.model = model
        self.verbose = verbose
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent()

    def _create_tools(self):
        """Create the tools available to the agent"""
        return [
            detect_outliers_zscore,
            detect_outliers_iqr,
            detect_outliers_isolation_forest,
            detect_outliers_lof,
            visualize_outliers,
            suggest_outlier_treatment,
        ]

    def _create_agent(self):
        """Create the agent executor"""
        prompt = PromptTemplate.from_template(
            """You are an Outlier Detection Specialist AI agent. Your role is to identify anomalous data points and recommend appropriate treatment strategies.

You have access to the following tools:
{tools}

When detecting outliers:
1. Understand the data distribution and context
2. Apply appropriate detection methods:
   - Z-score: For normally distributed data
   - IQR: For skewed or non-normal distributions
   - Isolation Forest: For multivariate outliers
   - LOF: For local density-based outliers
3. Visualize outliers for human review
4. Compare results across methods
5. Recommend treatment strategies based on outlier percentage and domain context

Use the following format:

Question: the input question or task you must complete
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""
        )

        agent = create_react_agent(
            llm=self.model,
            tools=self.tools,
            prompt=prompt
        )

        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            handle_parsing_errors=True,
            max_iterations=10,
        )

        return agent_executor

    def detect_outliers(
        self,
        data: pd.DataFrame,
        columns: List[str],
        methods: List[str] = ["iqr"],
        compare_methods: bool = False,
        visualize: bool = True
    ):
        """
        Detect outliers using specified methods.

        Args:
            data: DataFrame to analyze
            columns: List of columns to check for outliers
            methods: List of detection methods ('zscore', 'iqr', 'isolation_forest', 'lof')
            compare_methods: Whether to compare results across methods
            visualize: Whether to create visualizations

        Returns:
            Agent's outlier analysis and recommendations
        """
        # Build task description
        task = f"Detect outliers in the dataset.\n"
        task += f"Columns to analyze: {', '.join(columns)}\n"
        task += f"Methods to use: {', '.join(methods)}\n"
        task += f"Total rows: {len(data)}\n\n"

        if compare_methods and len(methods) > 1:
            task += "Compare the results from different detection methods.\n"

        if visualize:
            task += "Create visualizations of the detected outliers.\n"

        task += "\nProvide insights on the outliers and recommend treatment strategies."

        # Store data for tools to access
        self.current_data = data
        self.current_columns = columns

        # Invoke agent
        result = self.agent_executor.invoke({"input": task})

        return result

    def quick_detect(
        self,
        data: pd.DataFrame,
        columns: List[str],
        method: str = "iqr"
    ) -> dict:
        """
        Quick outlier detection without full agent reasoning.

        Args:
            data: DataFrame to analyze
            columns: Columns to check
            method: Detection method to use

        Returns:
            Dictionary with outlier information
        """
        data_dict = data.to_dict()

        if method == "zscore":
            result_json = detect_outliers_zscore.invoke({
                "data": data_dict,
                "columns": columns,
                "threshold": 3.0
            })
        elif method == "iqr":
            result_json = detect_outliers_iqr.invoke({
                "data": data_dict,
                "columns": columns,
                "multiplier": 1.5
            })
        elif method == "isolation_forest":
            result_json = detect_outliers_isolation_forest.invoke({
                "data": data_dict,
                "columns": columns,
                "contamination": 0.1
            })
        elif method == "lof":
            result_json = detect_outliers_lof.invoke({
                "data": data_dict,
                "columns": columns,
                "n_neighbors": 20
            })
        else:
            raise ValueError(f"Unknown method: {method}")

        # Get treatment recommendations
        treatment = suggest_outlier_treatment.invoke({
            "outlier_info": result_json,
            "treatment_strategy": "auto"
        })

        return {
            "outlier_info": json.loads(result_json),
            "treatment_recommendations": treatment
        }

    def compare_methods(
        self,
        data: pd.DataFrame,
        columns: List[str],
        methods: List[str] = ["zscore", "iqr", "isolation_forest"]
    ) -> pd.DataFrame:
        """
        Compare outlier detection across multiple methods.

        Args:
            data: DataFrame to analyze
            columns: Columns to check
            methods: List of methods to compare

        Returns:
            DataFrame comparing outlier counts across methods
        """
        data_dict = data.to_dict()
        comparison_results = []

        for method in methods:
            if method == "zscore":
                result_json = detect_outliers_zscore.invoke({
                    "data": data_dict,
                    "columns": columns
                })
            elif method == "iqr":
                result_json = detect_outliers_iqr.invoke({
                    "data": data_dict,
                    "columns": columns
                })
            elif method == "isolation_forest":
                result_json = detect_outliers_isolation_forest.invoke({
                    "data": data_dict,
                    "columns": columns
                })
                # Isolation forest returns different structure
                result = json.loads(result_json)
                if 'outlier_count' in result:
                    comparison_results.append({
                        'method': method,
                        'column': 'multivariate',
                        'outlier_count': result['outlier_count'],
                        'outlier_pct': result['outlier_pct']
                    })
                continue
            elif method == "lof":
                result_json = detect_outliers_lof.invoke({
                    "data": data_dict,
                    "columns": columns
                })
                # LOF returns different structure
                result = json.loads(result_json)
                if 'outlier_count' in result:
                    comparison_results.append({
                        'method': method,
                        'column': 'multivariate',
                        'outlier_count': result['outlier_count'],
                        'outlier_pct': result['outlier_pct']
                    })
                continue

            result = json.loads(result_json)
            for col, col_info in result.items():
                if isinstance(col_info, dict) and 'outlier_count' in col_info:
                    comparison_results.append({
                        'method': method,
                        'column': col,
                        'outlier_count': col_info['outlier_count'],
                        'outlier_pct': col_info['outlier_pct']
                    })

        return pd.DataFrame(comparison_results)

    def get_consensus_outliers(
        self,
        data: pd.DataFrame,
        columns: List[str],
        methods: List[str] = ["zscore", "iqr"],
        min_methods: int = 2
    ) -> List[int]:
        """
        Get outliers that are flagged by multiple methods (consensus).

        Args:
            data: DataFrame to analyze
            columns: Columns to check
            methods: Detection methods to use
            min_methods: Minimum number of methods that must flag a point as outlier

        Returns:
            List of row indices flagged by at least min_methods
        """
        data_dict = data.to_dict()
        outlier_sets = []

        for method in methods:
            if method == "zscore":
                result_json = detect_outliers_zscore.invoke({
                    "data": data_dict,
                    "columns": columns
                })
            elif method == "iqr":
                result_json = detect_outliers_iqr.invoke({
                    "data": data_dict,
                    "columns": columns
                })
            else:
                continue

            result = json.loads(result_json)
            method_outliers = set()
            for col, col_info in result.items():
                if isinstance(col_info, dict) and 'outlier_indices' in col_info:
                    method_outliers.update(col_info['outlier_indices'])

            outlier_sets.append(method_outliers)

        # Count how many methods flagged each index
        from collections import Counter
        all_outliers = []
        for outlier_set in outlier_sets:
            all_outliers.extend(list(outlier_set))

        outlier_counts = Counter(all_outliers)

        # Return indices flagged by at least min_methods
        consensus = [idx for idx, count in outlier_counts.items() if count >= min_methods]

        return consensus


if __name__ == "__main__":
    # Example usage
    print("Outlier Detection Agent loaded successfully!")
    print("\nExample usage:")
    print("""
    from langchain_openai import ChatOpenAI
    from custom.agents.outlier_detection_agent import OutlierDetectionAgent
    import pandas as pd
    import numpy as np

    # Create sample data with outliers
    np.random.seed(42)
    data = {
        'price': np.concatenate([np.random.normal(100, 15, 95), [500, 600, 700, 800, 900]]),
        'age': np.concatenate([np.random.normal(35, 10, 95), [150, 160, -5, -10, 200]]),
        'income': np.concatenate([np.random.normal(50000, 10000, 95), [500000, 600000, 700000, 800000, 900000]])
    }
    df = pd.DataFrame(data)

    # Initialize agent
    llm = ChatOpenAI(model="gpt-4")
    agent = OutlierDetectionAgent(model=llm)

    # Full analysis
    result = agent.detect_outliers(
        data=df,
        columns=['price', 'age', 'income'],
        methods=['zscore', 'iqr', 'isolation_forest'],
        compare_methods=True,
        visualize=True
    )
    print(result['output'])

    # Quick detection
    quick_result = agent.quick_detect(
        data=df,
        columns=['price', 'age'],
        method='iqr'
    )
    print(quick_result['treatment_recommendations'])

    # Compare methods
    comparison = agent.compare_methods(
        data=df,
        columns=['price', 'age', 'income'],
        methods=['zscore', 'iqr']
    )
    print(comparison)

    # Get consensus outliers
    consensus = agent.get_consensus_outliers(
        data=df,
        columns=['price', 'age', 'income'],
        methods=['zscore', 'iqr'],
        min_methods=2
    )
    print(f"Consensus outliers (flagged by ≥2 methods): {consensus}")
    """)
