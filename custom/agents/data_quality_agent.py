"""
Data Quality Validation Agent

An AI agent specialized in data quality assessment, validation, and reporting.
Uses comprehensive tools to evaluate data quality across multiple dimensions.
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from typing import Optional, Dict, Any
import pandas as pd

from custom.tools.data_quality import (
    check_schema_compliance,
    detect_data_anomalies,
    validate_business_rules,
    calculate_data_quality_score,
    generate_data_quality_report,
)


class DataQualityAgent:
    """
    AI Agent for comprehensive data quality validation and assessment.

    This agent can:
    - Validate schema compliance
    - Detect data anomalies (missing values, outliers, duplicates)
    - Enforce business rules
    - Calculate quality scores
    - Generate detailed quality reports

    Args:
        model: The language model to use (e.g., ChatOpenAI)
        verbose: Whether to print detailed execution logs
        **kwargs: Additional arguments

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> import pandas as pd
        >>>
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> agent = DataQualityAgent(model=llm)
        >>>
        >>> df = pd.read_csv("data.csv")
        >>> result = agent.invoke(
        ...     data=df,
        ...     task="Perform a comprehensive data quality assessment"
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
            check_schema_compliance,
            detect_data_anomalies,
            validate_business_rules,
            calculate_data_quality_score,
            generate_data_quality_report,
        ]

    def _create_agent(self):
        """Create the agent executor"""
        prompt = PromptTemplate.from_template(
            """You are a Data Quality Specialist AI agent. Your role is to assess and validate data quality using the tools available to you.

You have access to the following tools:
{tools}

When analyzing data quality:
1. Start with basic anomaly detection to understand the data
2. Calculate quality scores to quantify issues
3. Use specific validation tools for targeted checks
4. Generate comprehensive reports when requested
5. Provide actionable recommendations for data cleaning

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

    def invoke(
        self,
        data: pd.DataFrame,
        task: str,
        expected_schema: Optional[Dict[str, str]] = None,
        business_rules: Optional[Dict[str, Any]] = None,
    ):
        """
        Invoke the data quality agent with a specific task.

        Args:
            data: DataFrame to analyze
            task: Description of the quality check task
            expected_schema: Optional schema definition for validation
            business_rules: Optional business rules to validate

        Returns:
            Agent's analysis and recommendations
        """
        # Prepare context
        context = f"Dataset shape: {data.shape}\n"
        context += f"Columns: {', '.join(data.columns)}\n\n"
        context += f"Task: {task}\n\n"

        if expected_schema:
            context += f"Expected schema provided: {expected_schema}\n"

        if business_rules:
            context += f"Business rules provided: {list(business_rules.keys())}\n"

        # Store data in a format tools can access
        # In production, you might use a shared state or database
        self.current_data = data.to_dict()
        self.expected_schema = expected_schema
        self.business_rules = business_rules

        # Invoke agent
        result = self.agent_executor.invoke({"input": context})

        return result

    def quick_check(self, data: pd.DataFrame) -> str:
        """
        Perform a quick quality check without full agent reasoning.

        Args:
            data: DataFrame to check

        Returns:
            Quick quality assessment
        """
        data_dict = data.to_dict()

        # Run quick checks
        anomalies = detect_data_anomalies.invoke({"data": data_dict})
        score = calculate_data_quality_score.invoke({"data": data_dict})

        result = "QUICK QUALITY CHECK\n"
        result += "=" * 50 + "\n\n"
        result += anomalies + "\n\n"
        result += score

        return result

    def validate_schema(
        self,
        data: pd.DataFrame,
        expected_schema: Dict[str, str]
    ) -> str:
        """
        Validate data against expected schema.

        Args:
            data: DataFrame to validate
            expected_schema: Dict mapping column names to expected dtypes

        Returns:
            Schema validation report
        """
        return check_schema_compliance.invoke({
            "data": data.to_dict(),
            "expected_schema": expected_schema
        })

    def check_business_rules(
        self,
        data: pd.DataFrame,
        rules: Dict[str, Any]
    ) -> str:
        """
        Validate data against business rules.

        Args:
            data: DataFrame to validate
            rules: Dict of rule_name -> rule_function

        Returns:
            Business rule validation report
        """
        return validate_business_rules.invoke({
            "data": data.to_dict(),
            "rules": rules
        })

    def generate_report(
        self,
        data: pd.DataFrame,
        report_name: str = "data_quality_report"
    ) -> str:
        """
        Generate comprehensive quality report.

        Args:
            data: DataFrame to analyze
            report_name: Name for the report

        Returns:
            Full quality report
        """
        return generate_data_quality_report.invoke({
            "data": data.to_dict(),
            "report_name": report_name
        })


if __name__ == "__main__":
    # Example usage
    print("Data Quality Agent loaded successfully!")
    print("\nExample usage:")
    print("""
    from langchain_openai import ChatOpenAI
    from custom.agents.data_quality_agent import DataQualityAgent
    import pandas as pd

    # Initialize
    llm = ChatOpenAI(model="gpt-4")
    agent = DataQualityAgent(model=llm)

    # Load data
    df = pd.read_csv("your_data.csv")

    # Full quality assessment
    result = agent.invoke(
        data=df,
        task="Perform comprehensive quality check and provide recommendations"
    )

    # Quick check
    quick_result = agent.quick_check(df)
    print(quick_result)

    # Schema validation
    schema = {"age": "int64", "name": "object", "salary": "float64"}
    schema_result = agent.validate_schema(df, schema)
    print(schema_result)
    """)
