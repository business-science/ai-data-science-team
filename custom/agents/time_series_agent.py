"""
Time Series Analysis Agent

An AI agent specialized in analyzing time series data including trend detection,
seasonality analysis, decomposition, stationarity testing, and forecasting.
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from typing import Optional, Dict
import pandas as pd
import json

from custom.tools.time_series import (
    detect_seasonality,
    decompose_time_series,
    check_stationarity,
    forecast_baseline,
    visualize_time_series,
    calculate_time_series_metrics,
)


class TimeSeriesAgent:
    """
    AI Agent for comprehensive time series analysis.

    This agent can:
    - Detect seasonality patterns (daily, weekly, monthly, yearly)
    - Decompose series into trend, seasonal, and residual components
    - Check stationarity (ADF test)
    - Create baseline forecasts
    - Calculate time series metrics
    - Visualize trends and forecasts

    Args:
        model: The language model to use (e.g., ChatOpenAI)
        verbose: Whether to print detailed execution logs
        **kwargs: Additional arguments

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> import pandas as pd
        >>>
        >>> # Load time series data
        >>> df = pd.read_csv("sales_data.csv")
        >>>
        >>> # Initialize agent
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> agent = TimeSeriesAgent(model=llm)
        >>>
        >>> # Analyze time series
        >>> result = agent.analyze_series(
        ...     data=df,
        ...     value_column='sales',
        ...     date_column='date',
        ...     forecast_periods=30
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
            detect_seasonality,
            decompose_time_series,
            check_stationarity,
            forecast_baseline,
            visualize_time_series,
            calculate_time_series_metrics,
        ]

    def _create_agent(self):
        """Create the agent executor"""
        prompt = PromptTemplate.from_template(
            """You are a Time Series Analysis Specialist AI agent. Your role is to analyze temporal data, identify patterns, and create forecasts.

You have access to the following tools:
{tools}

When analyzing time series:
1. Calculate basic metrics to understand the data
2. Check for stationarity (important for many models)
3. Detect seasonality patterns
4. Decompose into trend, seasonal, and residual components
5. Create baseline forecasts for comparison
6. Visualize results for interpretation
7. Provide insights on patterns and recommendations for modeling

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

    def analyze_series(
        self,
        data: pd.DataFrame,
        value_column: str,
        date_column: str,
        forecast_periods: int = 0,
        check_seasonality: bool = True,
        decompose: bool = True
    ):
        """
        Perform comprehensive time series analysis.

        Args:
            data: DataFrame with time series data
            value_column: Column containing values
            date_column: Column containing dates
            forecast_periods: Number of periods to forecast (0 for no forecast)
            check_seasonality: Whether to check for seasonality
            decompose: Whether to decompose the series

        Returns:
            Agent's analysis and insights
        """
        # Build task description
        task = f"Analyze the time series data.\n"
        task += f"Value column: {value_column}\n"
        task += f"Date column: {date_column}\n"
        task += f"Number of observations: {len(data)}\n\n"

        if check_seasonality:
            task += "Detect any seasonality patterns in the data.\n"

        if decompose:
            task += "Decompose the series into trend, seasonal, and residual components.\n"

        if forecast_periods > 0:
            task += f"Create a baseline forecast for the next {forecast_periods} periods.\n"

        task += "\nProvide insights on patterns, trends, and recommendations for forecasting."

        # Store data for tools to access
        self.current_data = data
        self.current_value_column = value_column
        self.current_date_column = date_column

        # Invoke agent
        result = self.agent_executor.invoke({"input": task})

        return result

    def quick_analysis(
        self,
        data: pd.DataFrame,
        value_column: str,
        date_column: str
    ) -> dict:
        """
        Quick time series analysis without full agent reasoning.

        Args:
            data: DataFrame with time series
            value_column: Value column
            date_column: Date column

        Returns:
            Dictionary with key metrics and insights
        """
        data_dict = data.to_dict()

        # Calculate metrics
        metrics_json = calculate_time_series_metrics.invoke({
            "data": data_dict,
            "value_column": value_column,
            "date_column": date_column
        })

        # Check stationarity
        stationarity_json = check_stationarity.invoke({
            "data": data_dict,
            "value_column": value_column,
            "date_column": date_column
        })

        return {
            "metrics": json.loads(metrics_json),
            "stationarity": json.loads(stationarity_json)
        }

    def detect_patterns(
        self,
        data: pd.DataFrame,
        value_column: str,
        date_column: str,
        max_lag: int = 365
    ) -> dict:
        """
        Detect seasonality and patterns in time series.

        Args:
            data: DataFrame with time series
            value_column: Value column
            date_column: Date column
            max_lag: Maximum lag for seasonality detection

        Returns:
            Dictionary with seasonality information
        """
        seasonality_json = detect_seasonality.invoke({
            "data": data.to_dict(),
            "value_column": value_column,
            "date_column": date_column,
            "max_lag": max_lag
        })

        return json.loads(seasonality_json)

    def create_forecast(
        self,
        data: pd.DataFrame,
        value_column: str,
        date_column: str,
        forecast_periods: int = 30,
        methods: list = ["naive", "mean", "drift"]
    ) -> dict:
        """
        Create baseline forecasts using multiple methods.

        Args:
            data: DataFrame with time series
            value_column: Value column
            date_column: Date column
            forecast_periods: Number of periods to forecast
            methods: List of forecast methods to use

        Returns:
            Dictionary with forecasts from each method
        """
        data_dict = data.to_dict()
        forecasts = {}

        for method in methods:
            forecast_json = forecast_baseline.invoke({
                "data": data_dict,
                "value_column": value_column,
                "date_column": date_column,
                "forecast_periods": forecast_periods,
                "method": method
            })
            forecasts[method] = json.loads(forecast_json)

        return forecasts

    def decompose_series(
        self,
        data: pd.DataFrame,
        value_column: str,
        date_column: str,
        model: str = "additive",
        period: Optional[int] = None
    ) -> dict:
        """
        Decompose time series into components.

        Args:
            data: DataFrame with time series
            value_column: Value column
            date_column: Date column
            model: 'additive' or 'multiplicative'
            period: Seasonal period (auto-detected if None)

        Returns:
            Dictionary with trend, seasonal, and residual components
        """
        decomposition_json = decompose_time_series.invoke({
            "data": data.to_dict(),
            "value_column": value_column,
            "date_column": date_column,
            "model": model,
            "period": period
        })

        return json.loads(decomposition_json)

    def generate_report(
        self,
        data: pd.DataFrame,
        value_column: str,
        date_column: str
    ) -> str:
        """
        Generate a comprehensive time series analysis report.

        Args:
            data: DataFrame with time series
            value_column: Value column
            date_column: Date column

        Returns:
            Formatted text report
        """
        # Get all analyses
        metrics = self.quick_analysis(data, value_column, date_column)
        patterns = self.detect_patterns(data, value_column, date_column)

        report = "TIME SERIES ANALYSIS REPORT\n"
        report += "=" * 80 + "\n\n"

        # Basic info
        report += f"Series: {value_column}\n"
        report += f"Date Range: {metrics['metrics']['start_date']} to {metrics['metrics']['end_date']}\n"
        report += f"Observations: {metrics['metrics']['count']}\n"
        report += f"Duration: {metrics['metrics']['duration_days']} days\n\n"

        # Key metrics
        report += "KEY METRICS:\n"
        report += "-" * 80 + "\n"
        report += f"  Mean:                  {metrics['metrics']['mean']:.2f}\n"
        report += f"  Std Deviation:         {metrics['metrics']['std']:.2f}\n"
        report += f"  Min / Max:             {metrics['metrics']['min']:.2f} / {metrics['metrics']['max']:.2f}\n"
        report += f"  Range:                 {metrics['metrics']['range']:.2f}\n"

        if 'percent_change' in metrics['metrics'] and metrics['metrics']['percent_change'] is not None:
            report += f"  Total Change:          {metrics['metrics']['percent_change']:.2f}%\n"

        if 'trend_direction' in metrics['metrics']:
            report += f"  Trend Direction:       {metrics['metrics']['trend_direction']}\n"

        report += "\n"

        # Stationarity
        report += "STATIONARITY TEST:\n"
        report += "-" * 80 + "\n"
        report += f"  Test:                  {metrics['stationarity']['test']}\n"
        report += f"  P-value:               {metrics['stationarity']['p_value']:.4f}\n"
        report += f"  Result:                {metrics['stationarity']['interpretation']}\n"

        if 'recommendation' in metrics['stationarity']:
            report += f"  Recommendation:        {metrics['stationarity']['recommendation']}\n"

        report += "\n"

        # Seasonality
        report += "SEASONALITY DETECTION:\n"
        report += "-" * 80 + "\n"

        if patterns['seasonality_found']:
            report += "  Patterns detected:\n"
            for pattern in patterns['patterns']:
                report += f"    - {pattern['type'].capitalize()}: period={pattern['period']}, strength={pattern['strength']:.3f}\n"
        else:
            report += "  No strong seasonality patterns detected.\n"

        report += "\n" + "=" * 80 + "\n"

        return report


if __name__ == "__main__":
    # Example usage
    print("Time Series Agent loaded successfully!")
    print("\nExample usage:")
    print("""
    from langchain_openai import ChatOpenAI
    from custom.agents.time_series_agent import TimeSeriesAgent
    import pandas as pd
    import numpy as np

    # Create sample time series data
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    # Add trend + seasonality + noise
    trend = np.linspace(100, 150, 365)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(365) / 7)  # Weekly
    noise = np.random.normal(0, 3, 365)
    values = trend + seasonality + noise

    df = pd.DataFrame({
        'date': dates,
        'sales': values
    })

    # Initialize agent
    llm = ChatOpenAI(model="gpt-4")
    agent = TimeSeriesAgent(model=llm)

    # Full analysis
    result = agent.analyze_series(
        data=df,
        value_column='sales',
        date_column='date',
        forecast_periods=30,
        check_seasonality=True,
        decompose=True
    )
    print(result['output'])

    # Quick analysis
    quick_result = agent.quick_analysis(
        data=df,
        value_column='sales',
        date_column='date'
    )
    print(quick_result)

    # Detect patterns
    patterns = agent.detect_patterns(
        data=df,
        value_column='sales',
        date_column='date'
    )
    print(f"Seasonality found: {patterns['seasonality_found']}")

    # Create forecasts
    forecasts = agent.create_forecast(
        data=df,
        value_column='sales',
        date_column='date',
        forecast_periods=30,
        methods=['naive', 'mean', 'drift']
    )
    print(f"Naive forecast: {forecasts['naive']['forecast_values'][:5]}")

    # Generate report
    report = agent.generate_report(
        data=df,
        value_column='sales',
        date_column='date'
    )
    print(report)
    """)
