import asyncio
import os

# Dodanie ścieżki do sys.path dla importów
import sys
from typing import Any, AsyncGenerator, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))

from backend.agents.enhanced_base_agent import EnhancedBaseAgent
from backend.agents.error_types import AgentError
from backend.agents.state import AgentState


class TestEnhancedBaseAgent:
    """Testy dla Enhanced Base Agent - bazowej klasy wszystkich agentów"""

    @pytest.fixture
    def base_agent(self):
        """Fixture dla Enhanced Base Agent"""
        return EnhancedBaseAgent()

    @pytest.fixture
    def mock_llm_client(self):
        """Mock klienta LLM"""
        with patch("backend.agents.enhanced_base_agent.llm_client") as mock_client:
            mock_client.chat.return_value = {"message": {"content": "LLM response"}}
            yield mock_client

    @pytest.mark.asyncio
    async def test_process_success(self, base_agent):
        """Test pomyślnego przetwarzania"""
        # Given
        input_data = {"key": "value"}

        # When
        response = await base_agent.process(input_data)

        # Then
        assert response.success is True
        assert response.data == input_data

    @pytest.mark.asyncio
    async def test_process_with_validation(self, base_agent):
        """Test walidacji danych wejściowych"""
        # Given
        base_agent.input_schema = {"key": {"type": "string", "required": True}}
        input_data = {"key": "value"}

        # When
        response = await base_agent.process(input_data)

        # Then
        assert response.success is True

    @pytest.mark.asyncio
    async def test_process_with_validation_error(self, base_agent):
        """Test błędu walidacji danych wejściowych"""
        # Given
        base_agent.input_schema = {"key": {"type": "string", "required": True}}
        input_data = {"wrong_key": "value"}

        # When
        response = await base_agent.process(input_data)

        # Then
        assert response.success is False
        assert "Invalid input" in response.error
        assert response.error_type == AgentError.INVALID_INPUT

    @pytest.mark.asyncio
    async def test_process_with_llm_assistance(self, base_agent, mock_llm_client):
        """Test asystencji LLM w przetwarzaniu"""
        # Given
        input_data = {"key": "value"}
        base_agent.use_llm = True

        # When
        response = await base_agent.process(input_data)

        # Then
        assert response.success is True
        assert response.data["llm_assisted"] == "LLM response"
        mock_llm_client.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_with_error_handling(self, base_agent):
        """Test obsługi błędów w przetwarzaniu"""

        # Given
        async def mock_process():
            raise Exception("Processing error")

        base_agent._process = mock_process

        # When
        response = await base_agent.process({})

        # Then
        assert response.success is False
        assert "Processing error" in response.error
        assert response.error_type == AgentError.PROCESSING_ERROR

    @pytest.mark.asyncio
    async def test_process_with_timeout(self, base_agent):
        """Test przekroczenia czasu przetwarzania"""
        # Given
        base_agent.TIMEOUT = 0.1

        async def slow_process():
            await asyncio.sleep(0.2)
            return {}

        base_agent._process = slow_process

        # When
        response = await base_agent.process({})

        # Then
        assert response.success is False
        assert "Timeout" in response.error
        assert response.error_type == AgentError.TIMEOUT

    @pytest.mark.asyncio
    async def test_process_with_retry(self, base_agent):
        """Test mechanizmu ponawiania"""
        # Given
        base_agent.MAX_RETRIES = 2
        call_count = 0

        async def retry_process():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary error")
            return {}

        base_agent._process = retry_process

        # When
        response = await base_agent.process({})

        # Then
        assert response.success is True
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_process_with_state_management(self, base_agent):
        """Test zarządzania stanem agenta"""
        # Given
        input_data = {"key": "value"}
        base_agent.state = AgentState(conversation_history=[])

        # When
        await base_agent.process(input_data)

        # Then
        assert len(base_agent.state.conversation_history) == 1
        assert base_agent.state.conversation_history[0].input == input_data

    @pytest.mark.asyncio
    async def test_process_with_plugins(self, base_agent):
        """Test wykorzystania pluginów"""
        # Given
        mock_plugin = Mock()
        mock_plugin.execute = AsyncMock(return_value={"plugin_data": "result"})
        base_agent.plugins = [mock_plugin]

        # When
        response = await base_agent.process({})

        # Then
        assert response.data["plugin_data"] == "result"
        mock_plugin.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_with_metrics(self, base_agent):
        """Test zbierania metryk"""
        # Given
        base_agent.track_metrics = True

        # When
        await base_agent.process({})

        # Then
        assert base_agent.metrics["processing_time"] > 0
        assert base_agent.metrics["success_count"] == 1

    @pytest.mark.asyncio
    async def test_process_with_streaming(self, base_agent):
        """Test streamowania odpowiedzi"""

        # Given
        async def stream_process():
            yield {"chunk": 1}
            yield {"chunk": 2}

        base_agent._process = stream_process

        # When
        response = await base_agent.process({}, stream=True)

        # Then
        assert response.stream is not None
        collected = []
        async for chunk in response.stream:
            collected.append(chunk)
        assert len(collected) == 2

    @pytest.mark.asyncio
    async def test_process_with_authentication(self, base_agent):
        """Test uwierzytelniania"""
        # Given
        base_agent.requires_auth = True
        input_data = {"api_key": "valid_key"}
        base_agent.auth_service = Mock()
        base_agent.auth_service.authenticate.return_value = True

        # When
        response = await base_agent.process(input_data)

        # Then
        assert response.success is True

    @pytest.mark.asyncio
    async def test_process_with_authentication_error(self, base_agent):
        """Test błędu uwierzytelniania"""
        # Given
        base_agent.requires_auth = True
        input_data = {"api_key": "invalid_key"}
        base_agent.auth_service = Mock()
        base_agent.auth_service.authenticate.return_value = False

        # When
        response = await base_agent.process(input_data)

        # Then
        assert response.success is False
        assert "Authentication failed" in response.error
        assert response.error_type == AgentError.AUTHENTICATION_ERROR

    @pytest.mark.asyncio
    async def test_process_with_rate_limiting(self, base_agent):
        """Test ograniczania zapytań"""
        # Given
        base_agent.rate_limiter = Mock()
        base_agent.rate_limiter.acquire.side_effect = Exception("Rate limit exceeded")

        # When
        response = await base_agent.process({})

        # Then
        assert response.success is False
        assert "Rate limit exceeded" in response.error
        assert response.error_type == AgentError.RATE_LIMIT_EXCEEDED

    @pytest.mark.asyncio
    async def test_process_with_caching(self, base_agent):
        """Test mechanizmu cache'owania"""
        # Given
        input_data = {"key": "value"}
        base_agent.enable_caching = True

        # First call
        await base_agent.process(input_data)
        # Second call
        response = await base_agent.process(input_data)

        # Then
        assert response.cached is True

    @pytest.mark.asyncio
    async def test_process_with_logging(self, base_agent):
        """Test logowania"""
        # Given
        base_agent.logger = Mock()

        # When
        await base_agent.process({})

        # Then
        assert base_agent.logger.info.call_count == 2  # Start and end

    @pytest.mark.asyncio
    async def test_process_with_analytics(self, base_agent):
        """Test integracji z analityką"""
        # Given
        base_agent.analytics_service = Mock()

        # When
        await base_agent.process({})

        # Then
        base_agent.analytics_service.track_event.assert_called()

    @pytest.mark.asyncio
    async def test_process_with_circuit_breaker(self, base_agent):
        """Test wyłącznika bezpieczeństwa"""
        # Given
        base_agent.circuit_breaker = Mock()
        base_agent.circuit_breaker.allow_request.return_value = False

        # When
        response = await base_agent.process({})

        # Then
        assert response.success is False
        assert "Circuit breaker" in response.error
        assert response.error_type == AgentError.SERVICE_UNAVAILABLE

    @pytest.mark.asyncio
    async def test_process_performance(self, base_agent):
        """Test wydajności przetwarzania"""
        # Given
        base_agent.TIMEOUT = 1.0

        # When
        start_time = asyncio.get_event_loop().time()
        await base_agent.process({})
        end_time = asyncio.get_event_loop().time()

        # Then
        duration = end_time - start_time
        assert duration < 0.1  # Przetwarzanie powinno być szybkie
