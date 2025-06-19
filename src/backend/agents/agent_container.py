from typing import Any, Dict

from ..core.sqlalchemy_compat import AsyncSession


class AgentContainer:
    """Dependency Injection container for agent dependencies"""

    def __init__(self):
        self._services: Dict[str, Any] = {}

    def register(self, name: str, service: Any) -> None:
        """Register a service in the container"""
        self._services[name] = service

    def get(self, name: str) -> Any:
        """Get a registered service"""
        return self._services.get(name)

    def register_core_services(self, db: AsyncSession) -> None:
        """Register core services used by agents"""
        from ..core.enhanced_vector_store import EnhancedVectorStore
        from ..core.hybrid_llm_client import hybrid_llm_client
        from ..core.profile_manager import ProfileManager

        self.register("db", db)
        self.register("profile_manager", ProfileManager(db))
        self.register("llm_client", hybrid_llm_client)
        self.register("vector_store", EnhancedVectorStore())
