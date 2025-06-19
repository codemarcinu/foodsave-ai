import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from backend.core.hybrid_llm_client import ModelComplexity, hybrid_llm_client

from ..core.agent_interface import IFallbackProvider
from ..error_types import EnhancedAgentResponse, ErrorSeverity


class FallbackStrategy(ABC):
    """Base class for fallback strategies"""

    @abstractmethod
    async def execute(
        self, input_data: Dict[str, Any], error: Exception
    ) -> Optional[EnhancedAgentResponse]:
        pass


class PromptRewritingStrategy(FallbackStrategy):
    """Fallback strategy that rewrites the prompt"""

    async def execute(
        self, input_data: Dict[str, Any], error: Exception
    ) -> Optional[EnhancedAgentResponse]:
        query = self._extract_query(input_data)
        if not query:
            return None

        rewrite_prompt = (
            f"Poniższe zapytanie spowodowało błąd: '{str(error)}'. "
            f"Przepisz zapytanie, aby było jaśniejsze i łatwiejsze do przetworzenia. "
            f"Zachowaj intencję, ale uprość język i strukturę.\n\n"
            f"Oryginalne zapytanie: '{query}'\n\n"
            f"Przepisane zapytanie:"
        )

        response = await hybrid_llm_client.chat(
            messages=[
                {
                    "role": "system",
                    "content": "Jesteś pomocnym asystentem przepisującym zapytania.",
                },
                {"role": "user", "content": rewrite_prompt},
            ],
            model="gemma2:2b",
            force_complexity=ModelComplexity.SIMPLE,
        )

        if response and "message" in response:
            rewritten_query = response["message"]["content"].strip()
            new_input = input_data.copy()
            self._update_input(new_input, rewritten_query)
            return EnhancedAgentResponse(
                success=True,
                data=new_input,
                message="Prompt rewritten successfully",
                metadata={"original_query": query, "rewritten_query": rewritten_query},
            )
        return None

    def _extract_query(self, input_data: Dict[str, Any]) -> Optional[str]:
        for field in ["query", "prompt", "text", "content", "message"]:
            if field in input_data and isinstance(input_data[field], str):
                return input_data[field]
        return None

    def _update_input(self, input_data: Dict[str, Any], new_query: str) -> None:
        for field in ["query", "prompt", "text", "content", "message"]:
            if field in input_data:
                input_data[field] = new_query
                break


class SimplifiedModelStrategy(FallbackStrategy):
    """Fallback strategy using simplified model"""

    async def execute(
        self, input_data: Dict[str, Any], error: Exception
    ) -> Optional[EnhancedAgentResponse]:
        query = self._extract_query(input_data)
        if not query:
            return None

        simplify_prompt = (
            f"Zapytanie użytkownika: '{query}'\n\n"
            f"Udziel prostej, zwięzłej odpowiedzi na to zapytanie. "
            f"Jeśli zapytanie jest zbyt złożone lub niejasne, poproś o doprecyzowanie."
        )

        response = await hybrid_llm_client.chat(
            messages=[
                {
                    "role": "system",
                    "content": "Jesteś pomocnym asystentem udzielającym prostych odpowiedzi.",
                },
                {"role": "user", "content": simplify_prompt},
            ],
            model="llama3:8b",
            force_complexity=ModelComplexity.STANDARD,
        )

        if response and "message" in response:
            return EnhancedAgentResponse(
                success=True,
                text=response["message"]["content"].strip(),
                message="Odpowiedź wygenerowana w trybie uproszczonym",
                metadata={"simplified": True, "original_query": query},
            )
        return None

    def _extract_query(self, input_data: Dict[str, Any]) -> Optional[str]:
        for field in ["query", "prompt", "text", "content", "message"]:
            if field in input_data and isinstance(input_data[field], str):
                return input_data[field]
        return None


class MinimalResponseStrategy(FallbackStrategy):
    """Fallback strategy returning minimal response"""

    async def execute(
        self, input_data: Dict[str, Any], error: Exception
    ) -> EnhancedAgentResponse:
        return EnhancedAgentResponse(
            success=False,
            error=f"Przepraszam, nie mogłem przetworzyć Twojego zapytania: {str(error)}",
            error_severity=ErrorSeverity.MEDIUM,
            message="Wystąpił problem podczas przetwarzania zapytania. Proszę spróbować inaczej sformułować pytanie.",
            metadata={"fallback_tier": "minimal_response"},
        )


class FallbackManager(IFallbackProvider):
    """Manages fallback strategies execution"""

    def __init__(self):
        self.strategies = [
            PromptRewritingStrategy(),
            SimplifiedModelStrategy(),
            MinimalResponseStrategy(),
        ]

    async def execute_fallback(
        self, input_data: Dict[str, Any], error: Exception
    ) -> EnhancedAgentResponse:
        """Execute fallback strategies in order"""
        for strategy in self.strategies:
            try:
                result = await strategy.execute(input_data, error)
                if result:
                    return result
            except Exception as e:
                logging.error(f"Fallback strategy failed: {str(e)}")

        # If all strategies fail, return minimal response
        return await MinimalResponseStrategy().execute(input_data, error)
