"""
Base agent implementation for the AI assistant framework.
Provides core functionality for all agent types.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, ValidationError

from ..core.llm_client import llm_client

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


class AgentResponse:
    """Standard response format for all agents"""

    def __init__(
        self,
        success: bool,
        text: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        text_stream: Optional[AsyncGenerator[str, None]] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
    ):
        self.success = success
        self.text = text
        self.data = data or {}
        self.text_stream = text_stream
        self.message = message
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary format"""
        return {
            "success": self.success,
            "text": self.text,
            "data": self.data,
            "message": self.message,
            "error": self.error,
        }


class BaseAgent(ABC):
    """
    Base agent interface that all agent implementations must follow.
    Provides basic functionality for input validation and processing.
    """

    def __init__(self, name: str):
        self.name = name

    def _validate_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data against model if available"""
        return data

    def _handle_error(self, error: Exception) -> AgentResponse:
        """Standard error handling"""
        logger.error(f"Error in {self.name}: {str(error)}")
        return AgentResponse(
            success=False,
            error=str(error),
            message=f"An error occurred in the {self.name}",
        )

    @abstractmethod
    async def process(self, context: Dict[str, Any]) -> AgentResponse:
        """
        Process a request and return a response.
        Must be implemented by all agent subclasses.
        """
        pass

    async def _stream_llm_response(
        self, model: str, messages: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """Stream LLM response with basic error handling"""
        try:
            async for chunk in llm_client.chat(
                model=model, messages=messages, stream=True
            ):
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]
        except Exception as e:
            logger.error(f"Error streaming LLM response: {str(e)}")
            yield f"Error generating response: {str(e)}"


class EnhancedBaseAgent(BaseAgent, Generic[T]):
    """
    Enhanced base agent with additional features:
    - Type validation through generic input model
    - Improved error handling
    - Streaming LLM support
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.input_model: Optional[type[T]] = None

    def _validate_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data against model"""
        if self.input_model:
            try:
                validated = self.input_model.parse_obj(data)
                return validated.dict()
            except ValidationError as ve:
                raise ValueError(f"Invalid input data for {self.name}: {ve}") from ve
        return data

    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Process input data and return response.
        Must be implemented by all agent subclasses.
        """
        pass
