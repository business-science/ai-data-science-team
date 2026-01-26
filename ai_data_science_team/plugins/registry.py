"""
Plugin Registry for AI Data Science Team.

This module provides a centralized registry for discovering, registering,
and managing plugins.
"""

from typing import Any, Callable, Dict, List, Optional, Type, Union
from functools import wraps
import logging

from ai_data_science_team.plugins.base import (
    AgentPlugin,
    ToolPlugin,
    WorkflowPlugin,
    BasePlugin,
    PluginMetadata,
)

logger = logging.getLogger(__name__)

# Global registry instance
_global_registry: Optional["PluginRegistry"] = None


class PluginRegistry:
    """
    Central registry for all plugins.

    The registry maintains separate namespaces for agents, tools, and workflows,
    allowing for easy discovery and instantiation of plugins.

    Example:
        registry = PluginRegistry()

        # Register plugins
        registry.register_agent("my_agent", MyAgentClass)
        registry.register_tool("my_tool", MyToolClass)

        # List available plugins
        print(registry.list_agents())

        # Get and instantiate a plugin
        agent_class = registry.get_agent("my_agent")
        agent = agent_class(model=llm)
    """

    def __init__(self):
        """Initialize the registry."""
        self._agents: Dict[str, Type[AgentPlugin]] = {}
        self._tools: Dict[str, Type[ToolPlugin]] = {}
        self._workflows: Dict[str, Type[WorkflowPlugin]] = {}
        self._metadata: Dict[str, PluginMetadata] = {}

    # =========================================================================
    # Agent Registration
    # =========================================================================

    def register_agent(
        self,
        name: str,
        agent_class: Type[AgentPlugin],
        metadata: Optional[PluginMetadata] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Register an agent plugin.

        Parameters
        ----------
        name : str
            Unique name for the agent.
        agent_class : Type[AgentPlugin]
            The agent class to register.
        metadata : PluginMetadata, optional
            Plugin metadata. If not provided, will be extracted from the class.
        overwrite : bool, default False
            Whether to overwrite an existing agent with the same name.

        Raises
        ------
        ValueError
            If an agent with the same name already exists and overwrite is False.
        TypeError
            If agent_class is not a subclass of AgentPlugin.
        """
        if not isinstance(agent_class, type) or not issubclass(agent_class, AgentPlugin):
            raise TypeError(f"agent_class must be a subclass of AgentPlugin, got {type(agent_class)}")

        if name in self._agents and not overwrite:
            raise ValueError(f"Agent '{name}' already registered. Use overwrite=True to replace.")

        self._agents[name] = agent_class
        self._metadata[f"agent:{name}"] = metadata or agent_class.get_metadata()
        logger.info(f"Registered agent: {name}")

    def unregister_agent(self, name: str) -> bool:
        """
        Unregister an agent plugin.

        Parameters
        ----------
        name : str
            Name of the agent to unregister.

        Returns
        -------
        bool
            True if the agent was unregistered, False if it didn't exist.
        """
        if name in self._agents:
            del self._agents[name]
            del self._metadata[f"agent:{name}"]
            logger.info(f"Unregistered agent: {name}")
            return True
        return False

    def get_agent(self, name: str) -> Type[AgentPlugin]:
        """
        Get an agent class by name.

        Parameters
        ----------
        name : str
            Name of the agent.

        Returns
        -------
        Type[AgentPlugin]
            The agent class.

        Raises
        ------
        KeyError
            If the agent is not found.
        """
        if name not in self._agents:
            available = ", ".join(self._agents.keys()) or "none"
            raise KeyError(f"Agent '{name}' not found. Available agents: {available}")
        return self._agents[name]

    def list_agents(self) -> List[str]:
        """List all registered agent names."""
        return list(self._agents.keys())

    def has_agent(self, name: str) -> bool:
        """Check if an agent is registered."""
        return name in self._agents

    # =========================================================================
    # Tool Registration
    # =========================================================================

    def register_tool(
        self,
        name: str,
        tool_class: Type[ToolPlugin],
        metadata: Optional[PluginMetadata] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Register a tool plugin.

        Parameters
        ----------
        name : str
            Unique name for the tool.
        tool_class : Type[ToolPlugin]
            The tool class to register.
        metadata : PluginMetadata, optional
            Plugin metadata.
        overwrite : bool, default False
            Whether to overwrite an existing tool with the same name.
        """
        if not isinstance(tool_class, type) or not issubclass(tool_class, ToolPlugin):
            raise TypeError(f"tool_class must be a subclass of ToolPlugin, got {type(tool_class)}")

        if name in self._tools and not overwrite:
            raise ValueError(f"Tool '{name}' already registered. Use overwrite=True to replace.")

        self._tools[name] = tool_class
        self._metadata[f"tool:{name}"] = metadata or tool_class.get_metadata()
        logger.info(f"Registered tool: {name}")

    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool plugin."""
        if name in self._tools:
            del self._tools[name]
            del self._metadata[f"tool:{name}"]
            logger.info(f"Unregistered tool: {name}")
            return True
        return False

    def get_tool(self, name: str) -> Type[ToolPlugin]:
        """Get a tool class by name."""
        if name not in self._tools:
            available = ", ".join(self._tools.keys()) or "none"
            raise KeyError(f"Tool '{name}' not found. Available tools: {available}")
        return self._tools[name]

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    # =========================================================================
    # Workflow Registration
    # =========================================================================

    def register_workflow(
        self,
        name: str,
        workflow_class: Type[WorkflowPlugin],
        metadata: Optional[PluginMetadata] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Register a workflow plugin.

        Parameters
        ----------
        name : str
            Unique name for the workflow.
        workflow_class : Type[WorkflowPlugin]
            The workflow class to register.
        metadata : PluginMetadata, optional
            Plugin metadata.
        overwrite : bool, default False
            Whether to overwrite an existing workflow with the same name.
        """
        if not isinstance(workflow_class, type) or not issubclass(workflow_class, WorkflowPlugin):
            raise TypeError(f"workflow_class must be a subclass of WorkflowPlugin")

        if name in self._workflows and not overwrite:
            raise ValueError(f"Workflow '{name}' already registered. Use overwrite=True to replace.")

        self._workflows[name] = workflow_class
        self._metadata[f"workflow:{name}"] = metadata or workflow_class.get_metadata()
        logger.info(f"Registered workflow: {name}")

    def unregister_workflow(self, name: str) -> bool:
        """Unregister a workflow plugin."""
        if name in self._workflows:
            del self._workflows[name]
            del self._metadata[f"workflow:{name}"]
            logger.info(f"Unregistered workflow: {name}")
            return True
        return False

    def get_workflow(self, name: str) -> Type[WorkflowPlugin]:
        """Get a workflow class by name."""
        if name not in self._workflows:
            available = ", ".join(self._workflows.keys()) or "none"
            raise KeyError(f"Workflow '{name}' not found. Available workflows: {available}")
        return self._workflows[name]

    def list_workflows(self) -> List[str]:
        """List all registered workflow names."""
        return list(self._workflows.keys())

    def has_workflow(self, name: str) -> bool:
        """Check if a workflow is registered."""
        return name in self._workflows

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_metadata(self, plugin_type: str, name: str) -> Optional[PluginMetadata]:
        """
        Get metadata for a plugin.

        Parameters
        ----------
        plugin_type : str
            Type of plugin ("agent", "tool", "workflow").
        name : str
            Name of the plugin.

        Returns
        -------
        PluginMetadata or None
            The plugin metadata, or None if not found.
        """
        return self._metadata.get(f"{plugin_type}:{name}")

    def list_all(self) -> Dict[str, List[str]]:
        """
        List all registered plugins by type.

        Returns
        -------
        dict
            Dictionary with keys "agents", "tools", "workflows" and lists of names.
        """
        return {
            "agents": self.list_agents(),
            "tools": self.list_tools(),
            "workflows": self.list_workflows(),
        }

    def clear(self) -> None:
        """Clear all registered plugins."""
        self._agents.clear()
        self._tools.clear()
        self._workflows.clear()
        self._metadata.clear()
        logger.info("Cleared all registered plugins")

    def __repr__(self) -> str:
        return (
            f"PluginRegistry("
            f"agents={len(self._agents)}, "
            f"tools={len(self._tools)}, "
            f"workflows={len(self._workflows)})"
        )


def get_registry() -> PluginRegistry:
    """
    Get the global plugin registry.

    Returns
    -------
    PluginRegistry
        The global registry instance.
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = PluginRegistry()
    return _global_registry


def register_agent(name: str, metadata: Optional[PluginMetadata] = None):
    """
    Decorator to register an agent plugin with the global registry.

    Parameters
    ----------
    name : str
        Name for the agent.
    metadata : PluginMetadata, optional
        Plugin metadata.

    Example:
        @register_agent("my_agent")
        class MyAgent(AgentPlugin):
            def create_agent(self, model, **kwargs):
                ...
    """
    def decorator(cls: Type[AgentPlugin]) -> Type[AgentPlugin]:
        get_registry().register_agent(name, cls, metadata)
        return cls
    return decorator


def register_tool(name: str, metadata: Optional[PluginMetadata] = None):
    """
    Decorator to register a tool plugin with the global registry.

    Parameters
    ----------
    name : str
        Name for the tool.
    metadata : PluginMetadata, optional
        Plugin metadata.

    Example:
        @register_tool("my_tool")
        class MyTool(ToolPlugin):
            def create_tool(self, **kwargs):
                ...
    """
    def decorator(cls: Type[ToolPlugin]) -> Type[ToolPlugin]:
        get_registry().register_tool(name, cls, metadata)
        return cls
    return decorator


def register_workflow(name: str, metadata: Optional[PluginMetadata] = None):
    """
    Decorator to register a workflow plugin with the global registry.

    Parameters
    ----------
    name : str
        Name for the workflow.
    metadata : PluginMetadata, optional
        Plugin metadata.

    Example:
        @register_workflow("my_workflow")
        class MyWorkflow(WorkflowPlugin):
            def create_workflow(self, agents, tools, **kwargs):
                ...
    """
    def decorator(cls: Type[WorkflowPlugin]) -> Type[WorkflowPlugin]:
        get_registry().register_workflow(name, cls, metadata)
        return cls
    return decorator
