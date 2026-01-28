"""
AI Data Science Team - Plugin System

This module provides a plugin architecture for extending the AI Data Science Team
with custom agents, tools, and workflows.

Example usage:
    from ai_data_science_team.plugins import PluginRegistry, AgentPlugin, register_agent

    # Register a custom agent using decorator
    @register_agent("my_custom_agent")
    class MyCustomAgent(AgentPlugin):
        def invoke(self, **kwargs):
            # Custom logic here
            pass

    # Or register programmatically
    registry = PluginRegistry()
    registry.register_agent("another_agent", AnotherAgentClass)

    # List available plugins
    print(registry.list_agents())

    # Get a plugin
    agent_class = registry.get_agent("my_custom_agent")
"""

from ai_data_science_team.plugins.base import (
    AgentPlugin,
    ToolPlugin,
    WorkflowPlugin,
    PluginMetadata,
)
from ai_data_science_team.plugins.registry import (
    PluginRegistry,
    register_agent,
    register_tool,
    register_workflow,
    get_registry,
)
from ai_data_science_team.plugins.loader import (
    PluginLoader,
    load_plugins_from_directory,
    load_plugin_from_module,
)

__all__ = [
    # Base classes
    "AgentPlugin",
    "ToolPlugin",
    "WorkflowPlugin",
    "PluginMetadata",
    # Registry
    "PluginRegistry",
    "register_agent",
    "register_tool",
    "register_workflow",
    "get_registry",
    # Loader
    "PluginLoader",
    "load_plugins_from_directory",
    "load_plugin_from_module",
]
