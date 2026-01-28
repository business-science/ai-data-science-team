"""
Unit tests for the plugin system.
"""

import pytest
from unittest.mock import MagicMock
from pathlib import Path


class TestPluginBase:
    """Tests for plugin base classes."""

    def test_plugin_metadata_creation(self):
        """Test creating plugin metadata."""
        from ai_data_science_team.plugins import PluginMetadata

        meta = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            description="A test plugin",
            author="Test Author",
            tags=["test", "example"],
        )

        assert meta.name == "test_plugin"
        assert meta.version == "1.0.0"
        assert meta.description == "A test plugin"
        assert "test" in meta.tags

    def test_plugin_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        from ai_data_science_team.plugins import PluginMetadata

        meta = PluginMetadata(name="test", version="1.0.0")
        data = meta.to_dict()

        assert isinstance(data, dict)
        assert data["name"] == "test"
        assert data["version"] == "1.0.0"

    def test_plugin_metadata_from_dict(self):
        """Test creating metadata from dictionary."""
        from ai_data_science_team.plugins import PluginMetadata

        data = {"name": "test", "version": "2.0.0", "description": "Test"}
        meta = PluginMetadata.from_dict(data)

        assert meta.name == "test"
        assert meta.version == "2.0.0"

    def test_agent_plugin_base_class(self):
        """Test AgentPlugin base class."""
        from ai_data_science_team.plugins import AgentPlugin, PluginMetadata

        class TestAgent(AgentPlugin):
            metadata = PluginMetadata(name="test_agent", version="1.0.0")

            def create_agent(self, model, **kwargs):
                return MagicMock()

        agent = TestAgent()
        assert agent.plugin_type == "agent"
        assert agent.get_metadata().name == "test_agent"

    def test_tool_plugin_base_class(self):
        """Test ToolPlugin base class."""
        from ai_data_science_team.plugins import ToolPlugin, PluginMetadata

        class TestTool(ToolPlugin):
            metadata = PluginMetadata(name="test_tool", version="1.0.0")

            def create_tool(self, **kwargs):
                return MagicMock()

        tool = TestTool()
        assert tool.plugin_type == "tool"
        assert tool.get_metadata().name == "test_tool"

    def test_workflow_plugin_base_class(self):
        """Test WorkflowPlugin base class."""
        from ai_data_science_team.plugins import WorkflowPlugin, PluginMetadata

        class TestWorkflow(WorkflowPlugin):
            metadata = PluginMetadata(name="test_workflow", version="1.0.0")

            def create_workflow(self, agents, tools, **kwargs):
                return MagicMock()

        workflow = TestWorkflow()
        assert workflow.plugin_type == "workflow"
        assert workflow.get_metadata().name == "test_workflow"


class TestPluginRegistry:
    """Tests for the plugin registry."""

    def test_registry_creation(self):
        """Test creating a new registry."""
        from ai_data_science_team.plugins import PluginRegistry

        registry = PluginRegistry()
        assert registry is not None
        assert registry.list_agents() == []
        assert registry.list_tools() == []
        assert registry.list_workflows() == []

    def test_register_agent(self):
        """Test registering an agent."""
        from ai_data_science_team.plugins import PluginRegistry, AgentPlugin

        class MyAgent(AgentPlugin):
            def create_agent(self, model, **kwargs):
                return MagicMock()

        registry = PluginRegistry()
        registry.register_agent("my_agent", MyAgent)

        assert "my_agent" in registry.list_agents()
        assert registry.has_agent("my_agent")
        assert registry.get_agent("my_agent") == MyAgent

    def test_register_agent_duplicate_raises(self):
        """Test that registering duplicate agent raises error."""
        from ai_data_science_team.plugins import PluginRegistry, AgentPlugin

        class MyAgent(AgentPlugin):
            def create_agent(self, model, **kwargs):
                return MagicMock()

        registry = PluginRegistry()
        registry.register_agent("my_agent", MyAgent)

        with pytest.raises(ValueError, match="already registered"):
            registry.register_agent("my_agent", MyAgent)

    def test_register_agent_overwrite(self):
        """Test overwriting an existing agent."""
        from ai_data_science_team.plugins import PluginRegistry, AgentPlugin

        class MyAgent1(AgentPlugin):
            def create_agent(self, model, **kwargs):
                return MagicMock()

        class MyAgent2(AgentPlugin):
            def create_agent(self, model, **kwargs):
                return MagicMock()

        registry = PluginRegistry()
        registry.register_agent("my_agent", MyAgent1)
        registry.register_agent("my_agent", MyAgent2, overwrite=True)

        assert registry.get_agent("my_agent") == MyAgent2

    def test_unregister_agent(self):
        """Test unregistering an agent."""
        from ai_data_science_team.plugins import PluginRegistry, AgentPlugin

        class MyAgent(AgentPlugin):
            def create_agent(self, model, **kwargs):
                return MagicMock()

        registry = PluginRegistry()
        registry.register_agent("my_agent", MyAgent)

        result = registry.unregister_agent("my_agent")
        assert result is True
        assert not registry.has_agent("my_agent")

    def test_get_nonexistent_agent_raises(self):
        """Test that getting non-existent agent raises KeyError."""
        from ai_data_science_team.plugins import PluginRegistry

        registry = PluginRegistry()

        with pytest.raises(KeyError, match="not found"):
            registry.get_agent("nonexistent")

    def test_register_tool(self):
        """Test registering a tool."""
        from ai_data_science_team.plugins import PluginRegistry, ToolPlugin

        class MyTool(ToolPlugin):
            def create_tool(self, **kwargs):
                return MagicMock()

        registry = PluginRegistry()
        registry.register_tool("my_tool", MyTool)

        assert "my_tool" in registry.list_tools()
        assert registry.has_tool("my_tool")

    def test_register_workflow(self):
        """Test registering a workflow."""
        from ai_data_science_team.plugins import PluginRegistry, WorkflowPlugin

        class MyWorkflow(WorkflowPlugin):
            def create_workflow(self, agents, tools, **kwargs):
                return MagicMock()

        registry = PluginRegistry()
        registry.register_workflow("my_workflow", MyWorkflow)

        assert "my_workflow" in registry.list_workflows()
        assert registry.has_workflow("my_workflow")

    def test_list_all(self):
        """Test listing all plugins."""
        from ai_data_science_team.plugins import PluginRegistry, AgentPlugin, ToolPlugin

        class MyAgent(AgentPlugin):
            def create_agent(self, model, **kwargs):
                return MagicMock()

        class MyTool(ToolPlugin):
            def create_tool(self, **kwargs):
                return MagicMock()

        registry = PluginRegistry()
        registry.register_agent("agent1", MyAgent)
        registry.register_tool("tool1", MyTool)

        all_plugins = registry.list_all()

        assert "agents" in all_plugins
        assert "tools" in all_plugins
        assert "workflows" in all_plugins
        assert "agent1" in all_plugins["agents"]
        assert "tool1" in all_plugins["tools"]

    def test_clear_registry(self):
        """Test clearing the registry."""
        from ai_data_science_team.plugins import PluginRegistry, AgentPlugin

        class MyAgent(AgentPlugin):
            def create_agent(self, model, **kwargs):
                return MagicMock()

        registry = PluginRegistry()
        registry.register_agent("my_agent", MyAgent)
        registry.clear()

        assert registry.list_agents() == []

    def test_register_invalid_class_raises(self):
        """Test that registering non-plugin class raises TypeError."""
        from ai_data_science_team.plugins import PluginRegistry

        class NotAPlugin:
            pass

        registry = PluginRegistry()

        with pytest.raises(TypeError):
            registry.register_agent("invalid", NotAPlugin)


class TestPluginDecorators:
    """Tests for plugin registration decorators."""

    def test_register_agent_decorator(self):
        """Test @register_agent decorator."""
        from ai_data_science_team.plugins import (
            register_agent,
            get_registry,
            AgentPlugin,
            PluginRegistry,
        )

        # Use a fresh registry
        import ai_data_science_team.plugins.registry as reg_module
        old_registry = reg_module._global_registry
        reg_module._global_registry = PluginRegistry()

        try:
            @register_agent("decorated_agent")
            class DecoratedAgent(AgentPlugin):
                def create_agent(self, model, **kwargs):
                    return MagicMock()

            registry = get_registry()
            assert registry.has_agent("decorated_agent")
            assert registry.get_agent("decorated_agent") == DecoratedAgent

        finally:
            reg_module._global_registry = old_registry

    def test_register_tool_decorator(self):
        """Test @register_tool decorator."""
        from ai_data_science_team.plugins import (
            register_tool,
            get_registry,
            ToolPlugin,
            PluginRegistry,
        )

        import ai_data_science_team.plugins.registry as reg_module
        old_registry = reg_module._global_registry
        reg_module._global_registry = PluginRegistry()

        try:
            @register_tool("decorated_tool")
            class DecoratedTool(ToolPlugin):
                def create_tool(self, **kwargs):
                    return MagicMock()

            registry = get_registry()
            assert registry.has_tool("decorated_tool")

        finally:
            reg_module._global_registry = old_registry


class TestPluginLoader:
    """Tests for the plugin loader."""

    def test_loader_creation(self):
        """Test creating a plugin loader."""
        from ai_data_science_team.plugins import PluginLoader, PluginRegistry

        registry = PluginRegistry()
        loader = PluginLoader(registry)

        assert loader.registry is registry

    def test_load_from_nonexistent_file_raises(self):
        """Test loading from non-existent file raises error."""
        from ai_data_science_team.plugins import PluginLoader

        loader = PluginLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_from_file("/nonexistent/path/plugin.py")

    def test_load_from_nonexistent_directory_raises(self):
        """Test loading from non-existent directory raises error."""
        from ai_data_science_team.plugins import PluginLoader

        loader = PluginLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_from_directory("/nonexistent/path")

    def test_load_from_file(self, tmp_path):
        """Test loading plugins from a file."""
        from ai_data_science_team.plugins import PluginLoader, PluginRegistry

        # Create a plugin file
        plugin_code = '''
from ai_data_science_team.plugins import AgentPlugin

class FileLoadedAgent(AgentPlugin):
    plugin_name = "file_loaded_agent"

    def create_agent(self, model, **kwargs):
        return None
'''
        plugin_file = tmp_path / "test_plugin.py"
        plugin_file.write_text(plugin_code)

        registry = PluginRegistry()
        loader = PluginLoader(registry)
        loaded = loader.load_from_file(plugin_file)

        assert len(loaded) > 0
        assert registry.has_agent("file_loaded_agent")

    def test_load_from_directory(self, tmp_path):
        """Test loading plugins from a directory."""
        from ai_data_science_team.plugins import PluginLoader, PluginRegistry

        # Create plugin files
        plugin1 = '''
from ai_data_science_team.plugins import AgentPlugin

class DirAgent1(AgentPlugin):
    plugin_name = "dir_agent_1"
    def create_agent(self, model, **kwargs):
        return None
'''
        plugin2 = '''
from ai_data_science_team.plugins import ToolPlugin

class DirTool1(ToolPlugin):
    plugin_name = "dir_tool_1"
    def create_tool(self, **kwargs):
        return None
'''
        (tmp_path / "agent_plugin.py").write_text(plugin1)
        (tmp_path / "tool_plugin.py").write_text(plugin2)

        registry = PluginRegistry()
        loader = PluginLoader(registry)
        results = loader.load_from_directory(tmp_path)

        assert len(results) >= 2
        assert registry.has_agent("dir_agent_1")
        assert registry.has_tool("dir_tool_1")


class TestPluginInvocation:
    """Tests for invoking plugins."""

    def test_agent_plugin_invoke_without_init_raises(self):
        """Test that invoking uninitialized agent raises error."""
        from ai_data_science_team.plugins import AgentPlugin

        class TestAgent(AgentPlugin):
            def create_agent(self, model, **kwargs):
                return MagicMock()

        agent = TestAgent()

        with pytest.raises(RuntimeError, match="not initialized"):
            agent.invoke(data="test")

    def test_agent_plugin_initialize_and_invoke(self):
        """Test initializing and invoking an agent."""
        from ai_data_science_team.plugins import AgentPlugin

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"result": "success"}

        class TestAgent(AgentPlugin):
            def create_agent(self, model, **kwargs):
                return mock_graph

        agent = TestAgent(model=MagicMock())
        agent.initialize()

        assert agent.is_initialized
        result = agent.invoke(data="test")
        assert result == {"result": "success"}

    def test_tool_plugin_invoke(self):
        """Test invoking a tool plugin."""
        from ai_data_science_team.plugins import ToolPlugin

        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "tool result"

        class TestTool(ToolPlugin):
            def create_tool(self, **kwargs):
                return mock_tool

        tool = TestTool()
        tool.initialize()

        result = tool.invoke("test input")
        assert result == "tool result"
