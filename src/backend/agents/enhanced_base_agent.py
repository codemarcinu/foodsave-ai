import asyncio
import logging
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Callable, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, ValidationError

from ..core.hybrid_llm_client import ModelComplexity, hybrid_llm_client
from .adapters.alert_service import AlertService
from .adapters.error_handler import ErrorHandler
from .adapters.fallback_manager import FallbackManager
from .core.agent_interface import IAlertService, IErrorHandler
from .error_types import EnhancedAgentResponse, ErrorSeverity

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


class ImprovedBaseAgent(ABC, Generic[T]):
    """
    Enhanced base agent with improved error handling, fallbacks, and alerting.
    This extends the functionality of EnhancedBaseAgent with more robust error
    handling and recovery mechanisms.
    """

    def __init__(
        self,
        name: str,
        error_handler: Optional[IErrorHandler] = None,
        fallback_manager: Optional[FallbackManager] = None,
        alert_service: Optional[IAlertService] = None,
    ):
        self.name = name
        self.input_model: Optional[type[T]] = None
        self.error_handler = error_handler or ErrorHandler(name)
        self.fallback_manager = fallback_manager
        self.alert_service = alert_service or AlertService(name)
        self.fallback_attempts = 0
        self.max_fallback_attempts = 3

    async def process(self, input_data: Dict[str, Any]) -> EnhancedAgentResponse:
        """Main processing method to be implemented by each agent"""
        raise NotImplementedError("process() must be implemented by subclass")

    def _validate_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data against model"""
        if self.input_model:
            try:
                validated = self.input_model.parse_obj(data)
                return validated.dict()
            except ValidationError as ve:
                raise ValueError(f"Invalid input data for {self.name}: {ve}") from ve
        return data

    async def execute_with_fallback(
        self,
        func: Callable[..., Any],
        *args,
        fallback_handler: Optional[Callable[..., Any]] = None,
        error_severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        **kwargs,
    ) -> Any:
        """
        Execute a function with automatic fallback and error handling
        Delegates to ErrorHandler
        """
        return await self.error_handler.execute_with_fallback(
            func,
            *args,
            fallback_handler=fallback_handler,
            error_severity=error_severity,
            **kwargs,
        )

    async def safe_process(self, input_data: Dict[str, Any]) -> EnhancedAgentResponse:
        """
        Process input with comprehensive error handling and fallbacks

        This wraps the process method with additional safety and recovery mechanisms,
        including multi-tiered fallback approaches for different error scenarios.
        """
        start_time = time.time()
        self.fallback_attempts = 0

        try:
            # First try normal processing
            response = await self.process(input_data)
            response.processing_time = time.time() - start_time
            return response

        except Exception as e:
            logger.error(f"Error in {self.name}.process: {str(e)}")

            # Begin fallback sequence
            return await self._multi_tiered_fallback(input_data, e, start_time)

    async def _multi_tiered_fallback(
        self, input_data: Dict[str, Any], original_error: Exception, start_time: float
    ) -> EnhancedAgentResponse:
        """Delegate to FallbackManager"""
        if self.fallback_manager is None:
            raise ValueError("Fallback manager must be initialized")
        error_info = {
            "agent": self.name,
            "error_type": type(original_error).__name__,
            "message": str(original_error),
            "traceback": traceback.format_exc(),
        }

        if self.alert_service.should_alert(str(original_error), ErrorSeverity.HIGH):
            await self.alert_service.send_alert(
                f"Critical error in {self.name}", error_info, ErrorSeverity.HIGH
            )

        response = await self.fallback_manager.execute_fallback(
            input_data, original_error
        )
        response.processing_time = time.time() - start_time
        response.processed_with_fallback = True
        return response

    async def _try_prompt_rewriting(
        self, input_data: Dict[str, Any], original_error: Exception
    ) -> Optional[EnhancedAgentResponse]:
        """
        Attempt to rewrite the prompt to avoid the error
        """
        self.fallback_attempts += 1
        logger.info(f"Trying prompt rewriting (attempt {self.fallback_attempts})")

        try:
            # Extract query from input data
            query = input_data.get("query", "")
            if not query:
                # Try other common fields
                for field in ["prompt", "text", "content", "message"]:
                    if field in input_data and isinstance(input_data[field], str):
                        query = input_data[field]
                        break

            if not query or len(query) < 3:
                return None

            # Create rewriting prompt
            rewrite_prompt = (
                f"Poniższe zapytanie spowodowało błąd: '{str(original_error)}'. "
                f"Przepisz zapytanie, aby było jaśniejsze i łatwiejsze do przetworzenia. "
                f"Zachowaj intencję, ale uprość język i strukturę.\n\n"
                f"Oryginalne zapytanie: '{query}'\n\n"
                f"Przepisane zapytanie:"
            )

            # Get rewritten query
            response = await hybrid_llm_client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "Jesteś pomocnym asystentem przepisującym zapytania.",
                    },
                    {"role": "user", "content": rewrite_prompt},
                ],
                model="gemma2:2b",  # Use simple model for rewriting
                force_complexity=ModelComplexity.SIMPLE,
            )

            if response and "message" in response:
                rewritten_query = response["message"]["content"].strip()
                logger.info(f"Rewritten query: {rewritten_query}")

                # Create new input data with rewritten query
                new_input = input_data.copy()

                # Replace the query in the correct field
                if "query" in new_input:
                    new_input["query"] = rewritten_query
                elif "prompt" in new_input:
                    new_input["prompt"] = rewritten_query
                elif "text" in new_input:
                    new_input["text"] = rewritten_query
                elif "content" in new_input:
                    new_input["content"] = rewritten_query
                elif "message" in new_input:
                    new_input["message"] = rewritten_query

                # Try processing with rewritten query
                response = await self.process(new_input)
                if response.success:
                    response.metadata["original_query"] = query
                    response.metadata["rewritten_query"] = rewritten_query
                    return response

        except Exception as e:
            logger.error(f"Error in prompt rewriting: {str(e)}")

        return None

    async def _try_simplified_model(
        self, input_data: Dict[str, Any], original_error: Exception
    ) -> Optional[EnhancedAgentResponse]:
        """
        Attempt to process with a simplified model
        """
        self.fallback_attempts += 1
        logger.info(f"Trying simplified model (attempt {self.fallback_attempts})")

        try:
            # Extract query for simplification
            query = input_data.get("query", "")
            if not query:
                # Try other common fields
                for field in ["prompt", "text", "content", "message"]:
                    if field in input_data and isinstance(input_data[field], str):
                        query = input_data[field]
                        break

            if not query:
                return None

            # Create simplification prompt
            simplify_prompt = (
                f"Zapytanie użytkownika: '{query}'\n\n"
                f"Udziel prostej, zwięzłej odpowiedzi na to zapytanie. "
                f"Jeśli zapytanie jest zbyt złożone lub niejasne, poproś o doprecyzowanie."
            )

            # Use smaller model with simpler approach
            response = await hybrid_llm_client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "Jesteś pomocnym asystentem udzielającym prostych odpowiedzi.",
                    },
                    {"role": "user", "content": simplify_prompt},
                ],
                model="llama3:8b",  # Use mid-size model
                force_complexity=ModelComplexity.STANDARD,
            )

            if response and "message" in response:
                simplified_answer = response["message"]["content"].strip()

                # Create successful response
                return EnhancedAgentResponse(
                    success=True,
                    text=simplified_answer,
                    message="Odpowiedź wygenerowana w trybie uproszczonym",
                    metadata={"simplified": True, "original_query": query},
                )

        except Exception as e:
            logger.error(f"Error in simplified model approach: {str(e)}")

        return None

    # Removed methods as they're now handled by AlertService

    async def _stream_llm_response(
        self,
        model: str,
        messages: List[Dict[str, str]],
        retries: int = 3,
        backoff_factor: float = 0.5,
    ) -> AsyncGenerator[str, None]:
        """Stream LLM response with retries"""
        attempt = 0

        while attempt < retries:
            try:
                async for chunk in hybrid_llm_client.chat(
                    messages=messages, model=model, stream=True
                ):
                    if "message" in chunk and "content" in chunk["message"]:
                        yield chunk["message"]["content"]
                return
            except Exception as e:
                logger.debug(f"LLM streaming error: {str(e)}")
                attempt += 1
                if attempt < retries:
                    wait_time = backoff_factor * (2 ** (attempt - 1))
                    logger.warning(
                        f"LLM streaming attempt {attempt} failed, retrying in {wait_time}s"
                    )
                    await asyncio.sleep(wait_time)

        logger.error(f"LLM streaming failed after {retries} attempts")
        yield f"Error: Failed to generate response after {retries} attempts"


# Use ImprovedBaseAgent in place of EnhancedBaseAgent for improved error handling
EnhancedBaseAgentWithFallback = ImprovedBaseAgent
EnhancedBaseAgent = ImprovedBaseAgent  # Alias for backwards compatibility
