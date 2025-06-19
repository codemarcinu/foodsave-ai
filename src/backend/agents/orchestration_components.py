import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class IntentData:
    def __init__(self, type: str, entities: Dict = None):
        self.type = type
        self.entities = entities if entities is not None else {}


class MemoryContext:
    def __init__(self, session_id: str, history: List[Dict] = None):
        self.session_id = session_id
        self.history = history if history is not None else []


class IIntentDetector(ABC):
    @abstractmethod
    async def detect_intent(self, query: str, context: MemoryContext) -> IntentData:
        """Detects user intent based on query and context."""
        pass


class IAgentRouter(ABC):
    @abstractmethod
    async def route_to_agent(
        self, intent: IntentData, context: MemoryContext
    ) -> Dict[str, Any]:
        """Routes the intent to the appropriate agent and returns its response."""
        pass


class IMemoryManager(ABC):
    @abstractmethod
    async def get_context(self, session_id: str) -> MemoryContext:
        """Retrieves the session context."""
        pass

    @abstractmethod
    async def update_context(self, context: MemoryContext, new_data: Dict[str, Any]):
        """Updates the session context with new data."""
        pass


class IResponseGenerator(ABC):
    @abstractmethod
    async def generate_response(
        self, agent_response: Dict[str, Any], context: MemoryContext
    ) -> str:
        """Generates a text response for the user."""
        pass


class SimpleIntentDetector(IIntentDetector):
    async def detect_intent(self, query: str, context: MemoryContext) -> IntentData:
        query_lower = query.lower()
        if "przepis" in query_lower:
            return IntentData("recipe_request")
        elif "dodaj" in query_lower or "kup" in query_lower:
            return IntentData("add_to_list")
        return IntentData("general_query")


class BasicMemoryManager(IMemoryManager):
    def __init__(self):
        self.contexts = {}

    async def get_context(self, session_id: str) -> MemoryContext:
        if session_id not in self.contexts:
            self.contexts[session_id] = MemoryContext(session_id)
        return self.contexts[session_id]

    async def update_context(self, context: MemoryContext, new_data: Dict[str, Any]):
        context.history.append(
            {"timestamp": datetime.now().isoformat(), "data": new_data}
        )


class BasicResponseGenerator(IResponseGenerator):
    async def generate_response(
        self, agent_response: Dict[str, Any], context: MemoryContext
    ) -> str:
        if "response" in agent_response:
            return agent_response["response"]
        return "Przepraszam, wystąpił problem podczas przetwarzania żądania."
