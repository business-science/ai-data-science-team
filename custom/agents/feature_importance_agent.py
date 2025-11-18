"""
Feature Importance Analysis Agent

An AI agent specialized in analyzing and visualizing feature importance from ML models.
Supports multiple importance calculation methods and comparison across methods.
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from typing import Optional, List, Any
import pandas as pd

from custom.tools.feature_importance import (
    calculate_permutation_importance,
    extract_tree_importance,
    calculate_shap_importance,
    plot_feature_importance_bar,
    compare_importance_methods,
    generate_importance_report,
)


class FeatureImportanceAgent:
    """
    AI Agent for feature importance analysis and visualization.

    This agent can:
    - Extract importance from tree-based models
    - Calculate permutation importance
    - Compute SHAP values
    - Create visualizations
    - Compare different importance methods
    - Generate comprehensive reports

    Args:
        model: The language model to use (e.g., ChatOpenAI)
        verbose: Whether to print detailed execution logs
        **kwargs: Additional arguments

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> import pandas as pd
        >>>
        >>> # Train a model
        >>> rf_model = RandomForestClassifier()
        >>> rf_model.fit(X_train, y_train)
        >>>
        >>> # Initialize agent
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> agent = FeatureImportanceAgent(model=llm)
        >>>
        >>> # Analyze importance
        >>> result = agent.analyze_importance(
        ...     model=rf_model,
        ...     X=X_test,
        ...     y=y_test,
        ...     feature_names=X_train.columns.tolist()
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
            extract_tree_importance,
            calculate_permutation_importance,
            calculate_shap_importance,
            plot_feature_importance_bar,
            compare_importance_methods,
            generate_importance_report,
        ]

    def _create_agent(self):
        """Create the agent executor"""
        prompt = PromptTemplate.from_template(
            """You are a Feature Importance Analysis AI agent. Your role is to analyze and explain which features are most important in machine learning models.

You have access to the following tools:
{tools}

When analyzing feature importance:
1. Identify the type of model to choose the appropriate importance method
2. Extract or calculate importance scores using available methods
3. Visualize the results for easy interpretation
4. Compare different methods when requested
5. Generate reports explaining the findings
6. Provide recommendations on feature selection

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

    def analyze_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None,
        method: str = "auto",
        compare_methods: bool = False,
    ):
        """
        Analyze feature importance for a trained model.

        Args:
            model: Trained ML model
            X: Feature data
            y: Target data (required for permutation importance)
            feature_names: List of feature names
            method: Importance method ('auto', 'tree', 'permutation', 'shap')
            compare_methods: Whether to compare multiple methods

        Returns:
            Agent's analysis and visualizations
        """
        if feature_names is None:
            feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]

        # Build task description
        task = f"Analyze feature importance for the trained model.\n"
        task += f"Features: {len(feature_names)}\n"
        task += f"Samples: {len(X)}\n"

        if method != "auto":
            task += f"Use the '{method}' method.\n"

        if compare_methods:
            task += "Compare multiple importance calculation methods.\n"

        task += "\nProvide insights on which features are most important and why."

        # Store data for tools to access
        self.current_model = model
        self.current_X = X
        self.current_y = y
        self.current_feature_names = feature_names

        # Invoke agent
        result = self.agent_executor.invoke({"input": task})

        return result

    def quick_importance(
        self,
        model: Any,
        feature_names: List[str],
        top_n: int = 20
    ) -> dict:
        """
        Quick feature importance extraction without full agent reasoning.

        Args:
            model: Trained model
            feature_names: List of feature names
            top_n: Number of top features to return

        Returns:
            Dictionary with importance data and plot
        """
        import json

        # Extract importance
        importance_json = extract_tree_importance.invoke({
            "model": model,
            "feature_names": feature_names
        })

        # Create plot
        plot_json = plot_feature_importance_bar.invoke({
            "importance_data": importance_json,
            "top_n": top_n,
            "title": f"Top {top_n} Most Important Features"
        })

        # Generate report
        report = generate_importance_report.invoke({
            "importance_data": importance_json,
            "model_name": type(model).__name__,
            "method": "Tree-based"
        })

        return {
            "importance_scores": json.loads(importance_json),
            "plot": json.loads(plot_json),
            "report": report
        }

    def compare_methods(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str],
        methods: List[str] = ["tree", "permutation"]
    ) -> dict:
        """
        Compare feature importance across different calculation methods.

        Args:
            model: Trained model
            X: Feature data
            y: Target data
            feature_names: List of feature names
            methods: List of methods to compare

        Returns:
            Comparison results with visualizations
        """
        import json

        importance_data_list = []
        method_names = []

        for method in methods:
            if method == "tree":
                importance_json = extract_tree_importance.invoke({
                    "model": model,
                    "feature_names": feature_names
                })
                method_names.append("Tree-based")
            elif method == "permutation":
                importance_json = calculate_permutation_importance.invoke({
                    "model": model,
                    "X": X.to_dict(),
                    "y": y.to_dict() if hasattr(y, 'to_dict') else y,
                    "n_repeats": 5
                })
                method_names.append("Permutation")
            elif method == "shap":
                importance_json = calculate_shap_importance.invoke({
                    "model": model,
                    "X": X.to_dict(),
                    "sample_size": min(100, len(X))
                })
                method_names.append("SHAP")

            importance_data_list.append(importance_json)

        # Compare methods
        comparison_json = compare_importance_methods.invoke({
            "importance_data_list": importance_data_list,
            "method_names": method_names
        })

        return json.loads(comparison_json)

    def identify_redundant_features(
        self,
        model: Any,
        feature_names: List[str],
        threshold: float = 0.01
    ) -> List[str]:
        """
        Identify features with very low importance that might be redundant.

        Args:
            model: Trained model
            feature_names: List of feature names
            threshold: Importance threshold below which features are considered redundant

        Returns:
            List of potentially redundant feature names
        """
        import json

        # Extract importance
        importance_json = extract_tree_importance.invoke({
            "model": model,
            "feature_names": feature_names
        })

        importance_df = pd.read_json(importance_json)

        # Identify low importance features
        redundant = importance_df[importance_df['importance'] < threshold]['feature'].tolist()

        return redundant


if __name__ == "__main__":
    # Example usage
    print("Feature Importance Agent loaded successfully!")
    print("\nExample usage:")
    print("""
    from langchain_openai import ChatOpenAI
    from custom.agents.feature_importance_agent import FeatureImportanceAgent
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd

    # Train a model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Initialize agent
    llm = ChatOpenAI(model="gpt-4")
    agent = FeatureImportanceAgent(model=llm)

    # Full analysis
    result = agent.analyze_importance(
        model=rf,
        X=X_test,
        y=y_test,
        feature_names=X_train.columns.tolist(),
        compare_methods=True
    )
    print(result['output'])

    # Quick importance check
    quick_result = agent.quick_importance(
        model=rf,
        feature_names=X_train.columns.tolist(),
        top_n=15
    )
    print(quick_result['report'])

    # Identify redundant features
    redundant = agent.identify_redundant_features(
        model=rf,
        feature_names=X_train.columns.tolist(),
        threshold=0.01
    )
    print(f"Potentially redundant features: {redundant}")
    """)
