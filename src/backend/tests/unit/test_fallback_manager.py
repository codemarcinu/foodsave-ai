from unittest.mock import AsyncMock, patch

import pytest

from backend.agents.adapters.fallback_manager import (
    FallbackManager,
    MinimalResponseStrategy,
    PromptRewritingStrategy,
    SimplifiedModelStrategy,
)
from backend.agents.error_types import EnhancedAgentResponse


class TestFallbackStrategies:
    @pytest.fixture
    def input_data(self):
        return {"query": "test query"}

    @pytest.fixture
    def error(self):
        return Exception("test error")

    @pytest.mark.asyncio
    async def test_prompt_rewriting_strategy(self, input_data, error):
        strategy = PromptRewritingStrategy()
        with patch(
            "backend.core.hybrid_llm_client.hybrid_llm_client.chat"
        ) as mock_chat:
            mock_chat.return_value = {"message": {"content": "rewritten query"}}
            result = await strategy.execute(input_data, error)
            assert result.success
            assert result.data == {"query": "rewritten query"}
            assert result.metadata["original_query"] == "test query"
            assert result.metadata["rewritten_query"] == "rewritten query"

    @pytest.mark.asyncio
    async def test_simplified_model_strategy(self, input_data, error):
        strategy = SimplifiedModelStrategy()
        with patch(
            "backend.core.hybrid_llm_client.hybrid_llm_client.chat"
        ) as mock_chat:
            mock_chat.return_value = {"message": {"content": "simplified answer"}}
            result = await strategy.execute(input_data, error)
            assert isinstance(result, EnhancedAgentResponse)
            assert result.text == "simplified answer"

    @pytest.mark.asyncio
    async def test_minimal_response_strategy(self, input_data, error):
        strategy = MinimalResponseStrategy()
        result = await strategy.execute(input_data, error)
        assert isinstance(result, EnhancedAgentResponse)
        assert not result.success
        assert "Przepraszam" in result.error


class TestFallbackManager:
    @pytest.fixture
    def manager(self):
        return FallbackManager()

    @pytest.mark.asyncio
    async def test_execute_fallback(self, manager):
        input_data = {"query": "test"}
        error = Exception("test error")

        with patch.object(
            PromptRewritingStrategy, "execute", new_callable=AsyncMock
        ) as mock_strategy:
            mock_strategy.return_value = EnhancedAgentResponse(success=True)
            response = await manager.execute_fallback(input_data, error)
            assert response.success

    @pytest.mark.asyncio
    async def test_execute_fallback_all_fail(self, manager):
        input_data = {"query": "test"}
        error = Exception("test error")

        with patch.object(
            PromptRewritingStrategy, "execute", new_callable=AsyncMock
        ) as mock_strategy1, patch.object(
            SimplifiedModelStrategy, "execute", new_callable=AsyncMock
        ) as mock_strategy2:
            mock_strategy1.return_value = None
            mock_strategy2.return_value = None

            response = await manager.execute_fallback(input_data, error)
            assert not response.success
            assert "Przepraszam" in response.error
