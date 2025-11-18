"""
Model Comparison Agent

An AI agent specialized in comparing multiple ML models across various metrics.
Provides visualizations and recommendations for model selection.
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from custom.tools.model_comparison import (
    compare_classification_metrics,
    compare_regression_metrics,
    plot_roc_curves,
    plot_prediction_comparison,
    create_model_comparison_table,
    generate_model_comparison_report,
)


class ModelComparisonAgent:
    """
    AI Agent for comparing multiple ML models.

    This agent can:
    - Compare classification models (accuracy, precision, recall, F1, AUC)
    - Compare regression models (MSE, RMSE, MAE, R², MAPE)
    - Generate ROC curves
    - Create prediction comparison plots
    - Generate comparison tables and reports
    - Provide model selection recommendations

    Args:
        model: The language model to use (e.g., ChatOpenAI)
        verbose: Whether to print detailed execution logs
        **kwargs: Additional arguments

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.linear_model import LogisticRegression
        >>>
        >>> # Train models
        >>> rf = RandomForestClassifier()
        >>> lr = LogisticRegression()
        >>> rf.fit(X_train, y_train)
        >>> lr.fit(X_train, y_train)
        >>>
        >>> # Get predictions
        >>> predictions = {
        ...     'RandomForest': rf.predict(X_test),
        ...     'LogisticRegression': lr.predict(X_test),
        ...     'RandomForest_proba': rf.predict_proba(X_test),
        ...     'LogisticRegression_proba': lr.predict_proba(X_test),
        ... }
        >>>
        >>> # Initialize agent
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> agent = ModelComparisonAgent(model=llm)
        >>>
        >>> # Compare models
        >>> result = agent.compare_models(
        ...     predictions=predictions,
        ...     y_true=y_test,
        ...     model_names=['RandomForest', 'LogisticRegression'],
        ...     task_type='classification'
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
            compare_classification_metrics,
            compare_regression_metrics,
            plot_roc_curves,
            plot_prediction_comparison,
            create_model_comparison_table,
            generate_model_comparison_report,
        ]

    def _create_agent(self):
        """Create the agent executor"""
        prompt = PromptTemplate.from_template(
            """You are a Model Comparison Specialist AI agent. Your role is to compare multiple machine learning models and help users select the best model for their task.

You have access to the following tools:
{tools}

When comparing models:
1. Identify the task type (classification or regression)
2. Calculate appropriate metrics for all models
3. Generate visualizations (ROC curves, prediction plots)
4. Create comparison tables
5. Analyze trade-offs between models
6. Provide clear recommendations based on the use case

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

    def compare_models(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        model_names: List[str],
        task_type: str = "classification",
        generate_plots: bool = True,
    ):
        """
        Compare multiple models across metrics.

        Args:
            predictions: Dict with model predictions (and probabilities if available)
            y_true: True labels/values
            model_names: List of model names
            task_type: 'classification' or 'regression'
            generate_plots: Whether to generate visualizations

        Returns:
            Agent's comparison analysis
        """
        # Build task description
        task = f"Compare {len(model_names)} {task_type} models.\n"
        task += f"Models: {', '.join(model_names)}\n"
        task += f"Samples: {len(y_true)}\n\n"

        if generate_plots:
            task += "Generate visualizations for the comparison.\n"

        task += "Provide a comprehensive analysis and recommend the best model."

        # Store data for tools to access
        self.current_predictions = predictions
        self.current_y_true = y_true
        self.current_model_names = model_names
        self.current_task_type = task_type

        # Invoke agent
        result = self.agent_executor.invoke({"input": task})

        return result

    def quick_comparison(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        model_names: List[str],
        task_type: str = "classification"
    ) -> dict:
        """
        Quick model comparison without full agent reasoning.

        Args:
            predictions: Dict with model predictions
            y_true: True labels/values
            model_names: List of model names
            task_type: 'classification' or 'regression'

        Returns:
            Dictionary with metrics and report
        """
        import json

        # Calculate metrics
        if task_type == "classification":
            metrics_json = compare_classification_metrics.invoke({
                "models_predictions": predictions,
                "y_true": y_true.tolist() if hasattr(y_true, 'tolist') else y_true,
                "model_names": model_names
            })
        else:
            metrics_json = compare_regression_metrics.invoke({
                "models_predictions": predictions,
                "y_true": y_true.tolist() if hasattr(y_true, 'tolist') else y_true,
                "model_names": model_names
            })

        # Generate report
        report = generate_model_comparison_report.invoke({
            "metrics_data": metrics_json,
            "model_names": model_names,
            "task_type": task_type
        })

        return {
            "metrics": json.loads(metrics_json),
            "report": report
        }

    def plot_roc_comparison(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        model_names: List[str]
    ) -> dict:
        """
        Generate ROC curve comparison plot.

        Args:
            predictions: Dict with model predictions and probabilities
            y_true: True labels
            model_names: List of model names

        Returns:
            Plotly figure as dict
        """
        import json

        plot_json = plot_roc_curves.invoke({
            "models_predictions": predictions,
            "y_true": y_true.tolist() if hasattr(y_true, 'tolist') else y_true,
            "model_names": model_names
        })

        return json.loads(plot_json)

    def rank_models(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        model_names: List[str],
        task_type: str = "classification",
        ranking_metric: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Rank models based on a specific metric.

        Args:
            predictions: Dict with model predictions
            y_true: True labels/values
            model_names: List of model names
            task_type: 'classification' or 'regression'
            ranking_metric: Metric to use for ranking (default: f1_score for classification, r2 for regression)

        Returns:
            List of (model_name, score) tuples, sorted by score
        """
        import json

        # Calculate metrics
        if task_type == "classification":
            metrics_json = compare_classification_metrics.invoke({
                "models_predictions": predictions,
                "y_true": y_true.tolist() if hasattr(y_true, 'tolist') else y_true,
                "model_names": model_names
            })
            default_metric = 'f1_score'
        else:
            metrics_json = compare_regression_metrics.invoke({
                "models_predictions": predictions,
                "y_true": y_true.tolist() if hasattr(y_true, 'tolist') else y_true,
                "model_names": model_names
            })
            default_metric = 'r2'

        metrics_df = pd.DataFrame(json.loads(metrics_json))

        # Use specified metric or default
        metric = ranking_metric if ranking_metric else default_metric

        if metric not in metrics_df.columns:
            raise ValueError(f"Metric '{metric}' not found in results")

        # Sort by metric
        minimize_metrics = ['mse', 'rmse', 'mae', 'mape']
        ascending = metric in minimize_metrics

        ranked = metrics_df.sort_values(metric, ascending=ascending)

        return list(zip(ranked['model'], ranked[metric]))


if __name__ == "__main__":
    # Example usage
    print("Model Comparison Agent loaded successfully!")
    print("\nExample usage:")
    print("""
    from langchain_openai import ChatOpenAI
    from custom.agents.model_comparison_agent import ModelComparisonAgent
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    # Train models
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100),
        'LogisticRegression': LogisticRegression()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    # Get predictions
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            predictions[f'{name}_proba'] = model.predict_proba(X_test)

    # Initialize agent
    llm = ChatOpenAI(model="gpt-4")
    agent = ModelComparisonAgent(model=llm)

    # Full comparison
    result = agent.compare_models(
        predictions=predictions,
        y_true=y_test,
        model_names=list(models.keys()),
        task_type='classification',
        generate_plots=True
    )
    print(result['output'])

    # Quick comparison
    quick_result = agent.quick_comparison(
        predictions=predictions,
        y_true=y_test,
        model_names=list(models.keys()),
        task_type='classification'
    )
    print(quick_result['report'])

    # Rank models
    rankings = agent.rank_models(
        predictions=predictions,
        y_true=y_test,
        model_names=list(models.keys()),
        task_type='classification',
        ranking_metric='f1_score'
    )
    print("Rankings:", rankings)
    """)
