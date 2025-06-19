import asyncio
import os

# Dodanie ścieżki do sys.path dla importów
import sys
from typing import Any, AsyncGenerator, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))

from backend.agents.error_types import AgentError
from backend.core.hybrid_llm_client import HybridLLMClient
from backend.core.llm_client import LLMClient


class TestHybridLLMClient:
    """Testy dla Hybrid LLM Client - zarządzania lokalnymi modelami LLM"""

    @pytest.fixture
    def hybrid_client(self):
        """Fixture dla Hybrid LLM Client"""
        return HybridLLMClient()

    @pytest.fixture
    def mock_local_client(self):
        """Mock lokalnego klienta LLM"""
        with patch("backend.core.hybrid_llm_client.LocalLLMClient") as mock_client:
            mock_instance = mock_client.return_value
            mock_instance.chat = AsyncMock()
            yield mock_instance

    @pytest.fixture
    def mock_remote_client(self):
        """Mock zdalnego klienta LLM"""
        with patch("backend.core.hybrid_llm_client.RemoteLLMClient") as mock_client:
            mock_instance = mock_client.return_value
            mock_instance.chat = AsyncMock()
            yield mock_instance

    @pytest.fixture
    def mock_performance_monitor(self):
        """Mock monitora wydajności"""
        with patch("backend.core.hybrid_llm_client.PerformanceMonitor") as mock_monitor:
            mock_instance = mock_monitor.return_value
            mock_instance.measure = AsyncMock()
            yield mock_instance

    @pytest.mark.asyncio
    async def test_chat_with_local_model(self, hybrid_client, mock_local_client):
        """Test czatu z lokalnym modelem"""
        # Given
        input_data = {"messages": [{"role": "user", "content": "Hello"}]}
        mock_local_client.chat.return_value = {"message": {"content": "Local response"}}

        # When
        response = await hybrid_client.chat(input_data)

        # Then
        assert response["message"]["content"] == "Local response"
        mock_local_client.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_with_remote_model(self, hybrid_client, mock_remote_client):
        """Test czatu ze zdalnym modelem"""
        # Given
        input_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4",
        }
        mock_remote_client.chat.return_value = {
            "message": {"content": "Remote response"}
        }

        # When
        response = await hybrid_client.chat(input_data)

        # Then
        assert response["message"]["content"] == "Remote response"
        mock_remote_client.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_with_auto_selection(
        self, hybrid_client, mock_local_client, mock_remote_client
    ):
        """Test automatycznego wyboru modelu"""
        # Given
        input_data = {"messages": [{"role": "user", "content": "Hello"}]}
        mock_local_client.chat.return_value = {"message": {"content": "Local response"}}

        # When
        response = await hybrid_client.chat(input_data, auto_select=True)

        # Then
        assert response["message"]["content"] == "Local response"
        mock_local_client.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_with_fallback(
        self, hybrid_client, mock_local_client, mock_remote_client
    ):
        """Test mechanizmu fallback"""
        # Given
        input_data = {"messages": [{"role": "user", "content": "Hello"}]}
        mock_local_client.chat.side_effect = Exception("Local error")
        mock_remote_client.chat.return_value = {
            "message": {"content": "Fallback response"}
        }

        # When
        response = await hybrid_client.chat(input_data)

        # Then
        assert response["message"]["content"] == "Fallback response"
        mock_remote_client.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_with_performance_monitoring(
        self, hybrid_client, mock_performance_monitor
    ):
        """Test monitorowania wydajności"""
        # Given
        input_data = {"messages": [{"role": "user", "content": "Hello"}]}

        # When
        await hybrid_client.chat(input_data, monitor_performance=True)

        # Then
        mock_performance_monitor.measure.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_with_streaming(self, hybrid_client, mock_local_client):
        """Test streamowania odpowiedzi"""
        # Given
        input_data = {"messages": [{"role": "user", "content": "Hello"}]}

        async def mock_stream():
            yield {"content": "Stream"}
            yield {"content": " response"}

        mock_local_client.chat.return_value = mock_stream()

        # When
        response_stream = await hybrid_client.chat(input_data, stream=True)

        # Then
        collected_chunks = []
        async for chunk in response_stream:
            collected_chunks.append(chunk)

        assert len(collected_chunks) == 2
        assert collected_chunks[0]["content"] == "Stream"
        assert collected_chunks[1]["content"] == " response"

    @pytest.mark.asyncio
    async def test_chat_with_context_window(self, hybrid_client, mock_local_client):
        """Test zarządzania kontekstem"""
        # Given
        long_message = "A" * 5000
        input_data = {"messages": [{"role": "user", "content": long_message}]}

        # When
        await hybrid_client.chat(input_data)

        # Then
        call_args = mock_local_client.chat.call_args
        processed_messages = call_args[0][0]["messages"]
        assert (
            len(processed_messages[0]["content"]) < 4000
        )  # Sprawdzenie skrócenia wiadomości

    @pytest.mark.asyncio
    async def test_chat_with_custom_parameters(self, hybrid_client, mock_local_client):
        """Test niestandardowych parametrów"""
        # Given
        input_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 100,
        }

        # When
        await hybrid_client.chat(input_data)

        # Then
        call_args = mock_local_client.chat.call_args
        assert call_args[0][0]["temperature"] == 0.7
        assert call_args[0][0]["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_chat_with_function_calling(self, hybrid_client, mock_local_client):
        """Test wywoływania funkcji"""
        # Given
        input_data = {
            "messages": [{"role": "user", "content": "What's the weather in Warsaw?"}],
            "functions": [
                {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                }
            ],
        }
        mock_local_client.chat.return_value = {
            "function_call": {
                "name": "get_weather",
                "arguments": '{"location": "Warsaw"}',
            }
        }

        # When
        response = await hybrid_client.chat(input_data)

        # Then
        assert "function_call" in response
        assert response["function_call"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_chat_with_history(self, hybrid_client, mock_local_client):
        """Test czatu z historią konwersacji"""
        # Given
        input_data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "What's the capital of Poland?"},
                {"role": "assistant", "content": "Warsaw"},
                {"role": "user", "content": "What's its population?"},
            ]
        }

        # When
        await hybrid_client.chat(input_data)

        # Then
        call_args = mock_local_client.chat.call_args
        assert len(call_args[0][0]["messages"]) == 4

    @pytest.mark.asyncio
    async def test_chat_with_model_override(self, hybrid_client, mock_remote_client):
        """Test nadpisywania modelu"""
        # Given
        input_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "claude-2",
        }

        # When
        await hybrid_client.chat(input_data)

        # Then
        call_args = mock_remote_client.chat.call_args
        assert call_args[0][0]["model"] == "claude-2"

    @pytest.mark.asyncio
    async def test_chat_with_error_handling(
        self, hybrid_client, mock_local_client, mock_remote_client
    ):
        """Test obsługi błędów"""
        # Given
        input_data = {"messages": [{"role": "user", "content": "Hello"}]}
        mock_local_client.chat.side_effect = Exception("Local error")
        mock_remote_client.chat.side_effect = Exception("Remote error")

        # When
        with pytest.raises(AgentError) as excinfo:
            await hybrid_client.chat(input_data)

        # Then
        assert "All models failed" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_chat_with_retry(self, hybrid_client, mock_local_client):
        """Test mechanizmu ponawiania"""
        # Given
        input_data = {"messages": [{"role": "user", "content": "Hello"}]}
        mock_local_client.chat.side_effect = [
            Exception("Temporary error"),
            {"message": {"content": "Retry response"}},
        ]

        # When
        response = await hybrid_client.chat(input_data, retries=2)

        # Then
        assert response["message"]["content"] == "Retry response"
        assert mock_local_client.chat.call_count == 2

    @pytest.mark.asyncio
    async def test_chat_with_cost_tracking(self, hybrid_client, mock_local_client):
        """Test śledzenia kosztów"""
        # Given
        input_data = {"messages": [{"role": "user", "content": "Hello"}]}

        # When
        await hybrid_client.chat(input_data, track_cost=True)

        # Then
        assert hybrid_client.total_cost > 0

    @pytest.mark.asyncio
    async def test_chat_with_custom_prompt(self, hybrid_client, mock_local_client):
        """Test niestandardowego promptu"""
        # Given
        input_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "prompt_template": "Custom template: {input}",
        }

        # When
        await hybrid_client.chat(input_data)

        # Then
        call_args = mock_local_client.chat.call_args
        assert "Custom template: Hello" in call_args[0][0]["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_chat_with_safety_checks(self, hybrid_client, mock_local_client):
        """Test kontroli bezpieczeństwa"""
        # Given
        input_data = {
            "messages": [{"role": "user", "content": "Harmful content"}],
            "safety_level": "high",
        }
        mock_local_client.chat.return_value = {"message": {"content": "Blocked"}}

        # When
        response = await hybrid_client.chat(input_data)

        # Then
        assert response["message"]["content"] == "Blocked"

    @pytest.mark.asyncio
    async def test_chat_with_multilingual_support(
        self, hybrid_client, mock_local_client
    ):
        """Test wsparcia wielojęzykowego"""
        # Given
        input_data = {
            "messages": [{"role": "user", "content": "Cześć"}],
            "language": "pl",
        }

        # When
        await hybrid_client.chat(input_data)

        # Then
        call_args = mock_local_client.chat.call_args
        assert call_args[0][0]["language"] == "pl"

    @pytest.mark.asyncio
    async def test_chat_with_model_ensembling(
        self, hybrid_client, mock_local_client, mock_remote_client
    ):
        """Test łączenia modeli (ensembling)"""
        # Given
        input_data = {"messages": [{"role": "user", "content": "Complex question"}]}
        mock_local_client.chat.return_value = {"message": {"content": "Local answer"}}
        mock_remote_client.chat.return_value = {"message": {"content": "Remote answer"}}

        # When
        response = await hybrid_client.chat(input_data, ensemble=True)

        # Then
        assert "Local answer" in response["message"]["content"]
        assert "Remote answer" in response["message"]["content"]
        assert "Consensus" in response["message"]["content"]

    @pytest.mark.asyncio
    async def test_chat_performance(self, hybrid_client):
        """Test wydajności czatu"""
        # Given
        input_data = {"messages": [{"role": "user", "content": "Hello"}]}

        # When
        start_time = asyncio.get_event_loop().time()
        await hybrid_client.chat(input_data)
        end_time = asyncio.get_event_loop().time()

        # Then
        duration = end_time - start_time
        assert duration < 1.0  # Czat powinien trwać mniej niż 1 sekundę
