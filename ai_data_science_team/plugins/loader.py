"""
Plugin Loader for AI Data Science Team.

This module provides utilities for dynamically loading plugins from
files, directories, and Python modules.
"""

import importlib
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from ai_data_science_team.plugins.base import (
    AgentPlugin,
    ToolPlugin,
    WorkflowPlugin,
    BasePlugin,
)
from ai_data_science_team.plugins.registry import PluginRegistry, get_registry

logger = logging.getLogger(__name__)


class PluginLoader:
    """
    Utility class for loading plugins from various sources.

    Example:
        loader = PluginLoader()

        # Load from a Python file
        loader.load_from_file("my_plugins.py")

        # Load from a directory
        loader.load_from_directory("./plugins")

        # Load from an installed package
        loader.load_from_module("my_plugin_package")
    """

    def __init__(self, registry: Optional[PluginRegistry] = None):
        """
        Initialize the plugin loader.

        Parameters
        ----------
        registry : PluginRegistry, optional
            Registry to load plugins into. Uses global registry if not provided.
        """
        self.registry = registry or get_registry()
        self._loaded_modules: Dict[str, Any] = {}

    def load_from_file(self, filepath: Union[str, Path]) -> List[str]:
        """
        Load plugins from a Python file.

        Parameters
        ----------
        filepath : str or Path
            Path to the Python file containing plugin definitions.

        Returns
        -------
        list of str
            Names of loaded plugins.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ImportError
            If the file cannot be imported.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Plugin file not found: {filepath}")

        if not filepath.suffix == ".py":
            raise ValueError(f"Plugin file must be a .py file: {filepath}")

        # Generate a unique module name
        module_name = f"ai_ds_plugin_{filepath.stem}_{id(filepath)}"

        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load plugin file: {filepath}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            del sys.modules[module_name]
            raise ImportError(f"Error loading plugin file {filepath}: {e}") from e

        self._loaded_modules[str(filepath)] = module

        # Find and register plugins defined in the module
        loaded = self._register_plugins_from_module(module)
        logger.info(f"Loaded {len(loaded)} plugins from {filepath}")
        return loaded

    def load_from_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.py",
        recursive: bool = False,
    ) -> Dict[str, List[str]]:
        """
        Load plugins from all Python files in a directory.

        Parameters
        ----------
        directory : str or Path
            Path to the directory containing plugin files.
        pattern : str, default "*.py"
            Glob pattern for plugin files.
        recursive : bool, default False
            Whether to search recursively.

        Returns
        -------
        dict
            Dictionary mapping file paths to lists of loaded plugin names.
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Plugin directory not found: {directory}")

        if not directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        results = {}
        glob_method = directory.rglob if recursive else directory.glob

        for filepath in glob_method(pattern):
            if filepath.name.startswith("_"):
                continue  # Skip private files

            try:
                loaded = self.load_from_file(filepath)
                results[str(filepath)] = loaded
            except Exception as e:
                logger.warning(f"Failed to load plugins from {filepath}: {e}")
                results[str(filepath)] = []

        return results

    def load_from_module(self, module_name: str) -> List[str]:
        """
        Load plugins from an installed Python module/package.

        Parameters
        ----------
        module_name : str
            Name of the module to import (e.g., "my_plugin_package").

        Returns
        -------
        list of str
            Names of loaded plugins.
        """
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise ImportError(f"Could not import module {module_name}: {e}") from e

        self._loaded_modules[module_name] = module
        loaded = self._register_plugins_from_module(module)
        logger.info(f"Loaded {len(loaded)} plugins from module {module_name}")
        return loaded

    def _register_plugins_from_module(self, module: Any) -> List[str]:
        """
        Find and register all plugin classes defined in a module.

        Parameters
        ----------
        module : module
            The loaded Python module.

        Returns
        -------
        list of str
            Names of registered plugins.
        """
        loaded = []

        for name in dir(module):
            if name.startswith("_"):
                continue

            obj = getattr(module, name)

            # Skip non-classes and base classes
            if not isinstance(obj, type):
                continue
            if obj in (AgentPlugin, ToolPlugin, WorkflowPlugin, BasePlugin):
                continue

            # Check if it's a plugin subclass
            try:
                if issubclass(obj, AgentPlugin):
                    plugin_name = getattr(obj, "plugin_name", name.lower())
                    if not self.registry.has_agent(plugin_name):
                        self.registry.register_agent(plugin_name, obj)
                        loaded.append(f"agent:{plugin_name}")

                elif issubclass(obj, ToolPlugin):
                    plugin_name = getattr(obj, "plugin_name", name.lower())
                    if not self.registry.has_tool(plugin_name):
                        self.registry.register_tool(plugin_name, obj)
                        loaded.append(f"tool:{plugin_name}")

                elif issubclass(obj, WorkflowPlugin):
                    plugin_name = getattr(obj, "plugin_name", name.lower())
                    if not self.registry.has_workflow(plugin_name):
                        self.registry.register_workflow(plugin_name, obj)
                        loaded.append(f"workflow:{plugin_name}")

            except TypeError:
                # Not a class or can't check subclass
                continue

        return loaded

    def get_loaded_modules(self) -> List[str]:
        """Get list of loaded module/file paths."""
        return list(self._loaded_modules.keys())

    def unload_all(self) -> None:
        """Unload all loaded modules and clear the registry."""
        for module_name in list(self._loaded_modules.keys()):
            if module_name in sys.modules:
                del sys.modules[module_name]
        self._loaded_modules.clear()
        self.registry.clear()
        logger.info("Unloaded all plugins")


# Convenience functions using the global registry

def load_plugins_from_directory(
    directory: Union[str, Path],
    pattern: str = "*.py",
    recursive: bool = False,
) -> Dict[str, List[str]]:
    """
    Load plugins from a directory into the global registry.

    Parameters
    ----------
    directory : str or Path
        Path to the directory containing plugin files.
    pattern : str, default "*.py"
        Glob pattern for plugin files.
    recursive : bool, default False
        Whether to search recursively.

    Returns
    -------
    dict
        Dictionary mapping file paths to lists of loaded plugin names.
    """
    loader = PluginLoader()
    return loader.load_from_directory(directory, pattern, recursive)


def load_plugin_from_module(module_name: str) -> List[str]:
    """
    Load plugins from an installed module into the global registry.

    Parameters
    ----------
    module_name : str
        Name of the module to import.

    Returns
    -------
    list of str
        Names of loaded plugins.
    """
    loader = PluginLoader()
    return loader.load_from_module(module_name)
