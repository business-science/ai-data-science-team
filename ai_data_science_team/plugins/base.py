"""
Base classes for AI Data Science Team plugins.

This module defines the abstract base classes that all plugins must inherit from.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type
from datetime import datetime


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""

    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    author_email: str = ""
    license: str = "MIT"
    homepage: str = ""
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "author_email": self.author_email,
            "license": self.license,
            "homepage": self.homepage,
            "tags": self.tags,
            "dependencies": self.dependencies,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginMetadata":
        """Create metadata from dictionary."""
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


class BasePlugin(ABC):
    """Abstract base class for all plugins."""

    # Override these in subclasses
    plugin_type: str = "base"
    metadata: Optional[PluginMetadata] = None

    def __init__(self, **kwargs):
        """Initialize the plugin with optional configuration."""
        self.config = kwargs
        self._initialized = False

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        """Get plugin metadata."""
        if cls.metadata is not None:
            return cls.metadata
        return PluginMetadata(
            name=cls.__name__,
            description=cls.__doc__ or "",
        )

    def initialize(self) -> None:
        """Initialize the plugin. Override for custom initialization."""
        self._initialized = True

    def cleanup(self) -> None:
        """Cleanup resources. Override for custom cleanup."""
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if the plugin is initialized."""
        return self._initialized

    def validate_config(self) -> bool:
        """Validate plugin configuration. Override for custom validation."""
        return True

    def __repr__(self) -> str:
        meta = self.get_metadata()
        return f"<{self.__class__.__name__}(name={meta.name}, version={meta.version})>"


class AgentPlugin(BasePlugin):
    """
    Base class for custom agent plugins.

    Custom agents must implement the `create_agent` method which returns
    a compiled LangGraph state graph.

    Example:
        class MyAgent(AgentPlugin):
            metadata = PluginMetadata(
                name="my_agent",
                version="1.0.0",
                description="My custom agent"
            )

            def create_agent(self, model, **kwargs):
                # Build and return your agent graph
                return compiled_graph

            def invoke(self, **kwargs):
                # Invoke the agent
                return self.agent.invoke(kwargs)
    """

    plugin_type: str = "agent"

    def __init__(self, model=None, **kwargs):
        """
        Initialize the agent plugin.

        Parameters
        ----------
        model : langchain LLM, optional
            The language model to use for the agent.
        **kwargs
            Additional configuration options.
        """
        super().__init__(**kwargs)
        self.model = model
        self.agent = None

    @abstractmethod
    def create_agent(self, model, **kwargs):
        """
        Create and return the agent graph.

        Parameters
        ----------
        model : langchain LLM
            The language model to use.
        **kwargs
            Additional arguments for agent creation.

        Returns
        -------
        CompiledStateGraph
            The compiled LangGraph state graph.
        """
        pass

    def initialize(self) -> None:
        """Initialize the agent."""
        if self.model is not None:
            self.agent = self.create_agent(self.model, **self.config)
        super().initialize()

    def invoke(self, **kwargs) -> Any:
        """
        Invoke the agent.

        Parameters
        ----------
        **kwargs
            Arguments to pass to the agent.

        Returns
        -------
        Any
            The agent's response.
        """
        if self.agent is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return self.agent.invoke(kwargs)

    async def ainvoke(self, **kwargs) -> Any:
        """
        Asynchronously invoke the agent.

        Parameters
        ----------
        **kwargs
            Arguments to pass to the agent.

        Returns
        -------
        Any
            The agent's response.
        """
        if self.agent is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return await self.agent.ainvoke(kwargs)


class ToolPlugin(BasePlugin):
    """
    Base class for custom tool plugins.

    Custom tools must implement the `create_tool` method which returns
    a LangChain tool.

    Example:
        class MyTool(ToolPlugin):
            metadata = PluginMetadata(
                name="my_tool",
                version="1.0.0",
                description="My custom tool"
            )

            def create_tool(self, **kwargs):
                @tool
                def my_tool_func(input: str) -> str:
                    return f"Processed: {input}"
                return my_tool_func
    """

    plugin_type: str = "tool"

    def __init__(self, **kwargs):
        """Initialize the tool plugin."""
        super().__init__(**kwargs)
        self.tool = None

    @abstractmethod
    def create_tool(self, **kwargs):
        """
        Create and return the LangChain tool.

        Parameters
        ----------
        **kwargs
            Additional arguments for tool creation.

        Returns
        -------
        BaseTool
            The LangChain tool.
        """
        pass

    def initialize(self) -> None:
        """Initialize the tool."""
        self.tool = self.create_tool(**self.config)
        super().initialize()

    def invoke(self, input: Any) -> Any:
        """
        Invoke the tool.

        Parameters
        ----------
        input : Any
            Input to the tool.

        Returns
        -------
        Any
            The tool's output.
        """
        if self.tool is None:
            raise RuntimeError("Tool not initialized. Call initialize() first.")
        return self.tool.invoke(input)


class WorkflowPlugin(BasePlugin):
    """
    Base class for custom workflow plugins.

    Workflows combine multiple agents and tools into a pipeline.

    Example:
        class MyWorkflow(WorkflowPlugin):
            metadata = PluginMetadata(
                name="my_workflow",
                version="1.0.0",
                description="My custom workflow"
            )

            def create_workflow(self, agents, tools, **kwargs):
                # Build workflow using agents and tools
                return workflow_graph
    """

    plugin_type: str = "workflow"

    def __init__(self, agents: Optional[List[AgentPlugin]] = None,
                 tools: Optional[List[ToolPlugin]] = None, **kwargs):
        """
        Initialize the workflow plugin.

        Parameters
        ----------
        agents : list of AgentPlugin, optional
            Agents to use in the workflow.
        tools : list of ToolPlugin, optional
            Tools to use in the workflow.
        **kwargs
            Additional configuration options.
        """
        super().__init__(**kwargs)
        self.agents = agents or []
        self.tools = tools or []
        self.workflow = None

    @abstractmethod
    def create_workflow(self, agents: List[AgentPlugin],
                        tools: List[ToolPlugin], **kwargs):
        """
        Create and return the workflow graph.

        Parameters
        ----------
        agents : list of AgentPlugin
            Available agents.
        tools : list of ToolPlugin
            Available tools.
        **kwargs
            Additional arguments.

        Returns
        -------
        CompiledStateGraph
            The compiled workflow graph.
        """
        pass

    def initialize(self) -> None:
        """Initialize the workflow."""
        # Initialize all agents and tools first
        for agent in self.agents:
            if not agent.is_initialized:
                agent.initialize()
        for tool in self.tools:
            if not tool.is_initialized:
                tool.initialize()

        self.workflow = self.create_workflow(self.agents, self.tools, **self.config)
        super().initialize()

    def invoke(self, **kwargs) -> Any:
        """
        Invoke the workflow.

        Parameters
        ----------
        **kwargs
            Arguments to pass to the workflow.

        Returns
        -------
        Any
            The workflow's response.
        """
        if self.workflow is None:
            raise RuntimeError("Workflow not initialized. Call initialize() first.")
        return self.workflow.invoke(kwargs)
