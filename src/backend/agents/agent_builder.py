from typing import Any, Dict, Optional

from .agent_container import AgentContainer
from .enhanced_base_agent import ImprovedBaseAgent


class AgentBuilder:
    """Builder pattern for configuring and creating agents"""

    def __init__(self, container: AgentContainer, factory: Any = None):
        self.container = container
        self._factory = factory
        self._config: Dict[str, Any] = {}
        self._agent_type: Optional[str] = None

    def of_type(self, agent_type: str) -> "AgentBuilder":
        """Set the agent type to build"""
        self._agent_type = agent_type
        return self

    def with_config(self, config: Dict[str, Any]) -> "AgentBuilder":
        """Add configuration for the agent"""
        self._config.update(config)
        return self

    def build(self) -> ImprovedBaseAgent:
        """Build and configure the agent instance"""
        if not self._agent_type:
            raise ValueError("Agent type must be specified")

        # Get core dependencies
        db = self.container.get("db")
        profile_manager = self.container.get("profile_manager")
        llm_client = self.container.get("llm_client")

        # Create agent with dependencies
        agent = self._create_agent_instance()

        # Inject common dependencies
        if hasattr(agent, "set_db"):
            agent.set_db(db)
        if hasattr(agent, "set_profile_manager"):
            agent.set_profile_manager(profile_manager)
        if hasattr(agent, "set_llm_client"):
            agent.set_llm_client(llm_client)

        # Apply additional configuration
        for key, value in self._config.items():
            if hasattr(agent, key):
                setattr(agent, key, value)

        return agent

    def _create_agent_instance(self) -> ImprovedBaseAgent:
        """Create the appropriate agent instance"""
        if not self._agent_type:
            raise ValueError("Agent type must be specified")

        # Get agent class from registry
        agent_class = self._factory._registry.get(self._agent_type)
        if not agent_class:
            raise ValueError(f"Unsupported agent type: {self._agent_type}")

        # Create agent instance with basic dependencies
        agent_name = f"{self._agent_type}Agent"
        return agent_class(
            name=agent_name,
            error_handler=self.container.get("error_handler"),
            fallback_manager=self.container.get("fallback_manager"),
            alert_service=self.container.get("alert_service"),
        )
