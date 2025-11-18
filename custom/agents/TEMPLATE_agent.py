"""
Template for creating a custom agent

Copy this file and rename it to create your own agent.
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool


class MyCustomAgent:
    """
    Description of what your custom agent does.

    Args:
        model: The language model to use
        verbose: Whether to print verbose output
        **kwargs: Additional arguments

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> agent = MyCustomAgent(model=llm)
        >>> result = agent.invoke("Do something")
    """

    def __init__(self, model, verbose=True, **kwargs):
        self.model = model
        self.verbose = verbose
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent()

    def _create_tools(self):
        """Define the tools your agent can use"""
        tools = [
            Tool(
                name="example_tool",
                func=self._example_tool_function,
                description="Description of what this tool does"
            ),
            # Add more tools here
        ]
        return tools

    def _example_tool_function(self, input_text: str) -> str:
        """
        Example tool function

        Args:
            input_text: Input from the agent

        Returns:
            Result of the tool operation
        """
        # Your tool logic here
        return f"Processed: {input_text}"

    def _create_agent(self):
        """Create the agent executor"""
        prompt = PromptTemplate.from_template(
            """You are a helpful AI assistant.

            You have access to the following tools:
            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

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
            handle_parsing_errors=True
        )

        return agent_executor

    def invoke(self, user_input: str):
        """
        Invoke the agent with a user input

        Args:
            user_input: The user's question or request

        Returns:
            The agent's response
        """
        result = self.agent_executor.invoke({"input": user_input})
        return result


# Example usage
if __name__ == "__main__":
    # This code runs only when you execute this file directly
    # Uncomment to test:

    # from langchain_openai import ChatOpenAI
    # import os
    #
    # os.environ['OPENAI_API_KEY'] = 'your-api-key'
    # llm = ChatOpenAI(model="gpt-4")
    #
    # agent = MyCustomAgent(model=llm)
    # result = agent.invoke("Test my custom agent")
    # print(result)

    print("Template agent loaded. Customize this file for your needs!")
