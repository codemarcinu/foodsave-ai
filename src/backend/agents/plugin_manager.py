from typing import Dict, List, Optional

from .enhanced_base_agent import ImprovedBaseAgent
from .plugin_interface import AgentPlugin


class PluginManager:
    """Manages registration and execution of agent plugins"""

    def __init__(self):
        self._plugins: Dict[str, AgentPlugin] = {}

    def register_plugin(self, name: str, plugin: AgentPlugin) -> None:
        """Register a new plugin

        Args:
            name: Unique plugin identifier
            plugin: Plugin instance implementing AgentPlugin
        """
        if name in self._plugins:
            raise ValueError(f"Plugin {name} already registered")
        self._plugins[name] = plugin

    def unregister_plugin(self, name: str) -> None:
        """Remove a registered plugin

        Args:
            name: Plugin identifier to remove
        """
        self._plugins.pop(name, None)

    def initialize_plugins(self, agent: ImprovedBaseAgent) -> None:
        """Initialize all plugins with agent instance"""
        for plugin in self._plugins.values():
            plugin.initialize(agent)

    def apply_before_processing(self, input_data: Dict) -> Dict:
        """Apply all before_process hooks"""
        result = input_data
        for plugin in self._plugins.values():
            result = plugin.before_process(result)
        return result

    def apply_after_processing(self, output_data: Dict) -> Dict:
        """Apply all after_process hooks"""
        result = output_data
        for plugin in self._plugins.values():
            result = plugin.after_process(result)
        return result

    def get_plugin(self, name: str) -> Optional[AgentPlugin]:
        """Get registered plugin by name"""
        return self._plugins.get(name)

    def list_plugins(self) -> List[str]:
        """Get names of all registered plugins"""
        return list(self._plugins.keys())
