# Skopiuj i wklej ten kod jako całą zawartość pliku state.py

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class AgentStatus(Enum):
    """Status of an agent"""

    IDLE = auto()
    PROCESSING = auto()
    WAITING_FOR_INPUT = auto()
    ERROR = auto()


@dataclass
class AgentState:
    """Represents the state of an individual agent"""

    name: str
    status: AgentStatus = AgentStatus.IDLE
    last_activity: float = 0.0  # timestamp
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationState:
    """Represents the state of a conversation."""

    session_id: str
    history: List[Dict[str, Any]] = field(default_factory=list)
    is_awaiting_clarification: bool = False
    is_cooking_confirmation: bool = False
    original_intent: Optional[str] = None
    original_entities: Optional[Dict[str, Any]] = None
    ambiguous_options: List[Any] = field(default_factory=list)
    cooking_ingredients: List[Dict[str, Any]] = field(default_factory=list)
    agent_states: Dict[str, bool] = field(
        default_factory=lambda: {
            "weather": True,
            "search": True,
            "shopping": False,
            "cooking": False,
        }
    )
    current_model: str = "gemma3:12b"  # Default model to use for LLM operations

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def set_clarification_mode(
        self, intent: str, entities: Dict[str, Any], options: List[Any]
    ):
        self.is_awaiting_clarification = True
        self.original_intent = intent
        self.original_entities = entities
        self.ambiguous_options = options

    def set_cooking_state(self, ingredients: List[Dict[str, Any]]):
        """Set state for cooking confirmation flow."""
        self.is_cooking_confirmation = True
        self.cooking_ingredients = ingredients

    def reset(self):
        self.is_awaiting_clarification = False
        self.is_cooking_confirmation = False
        self.original_intent = None
        self.original_entities = None
        self.ambiguous_options = []
        self.cooking_ingredients = []

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "session_id": self.session_id,
            "history_length": len(self.history),
            "is_awaiting_clarification": self.is_awaiting_clarification,
            "is_cooking_confirmation": self.is_cooking_confirmation,
            "cooking_ingredients": self.cooking_ingredients,
            "original_intent": self.original_intent,
            "ambiguous_options": self.ambiguous_options,
            "agent_states": self.agent_states,
            "current_model": self.current_model,
        }
        return result
