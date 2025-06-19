import asyncio
import os

# Dodanie ścieżki do sys.path dla importów
import sys
from typing import Any, AsyncGenerator, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))

from src.backend.agents.enhanced_orchestrator import EnhancedOrchestrator
from src.backend.agents.error_types import OrchestratorError
from src.backend.agents.state import AgentState


class TestEnhancedOrchestrator:
    """Testy dla Enhanced Orchestrator - centralnego kontrolera systemu"""

    @pytest.fixture
    def orchestrator(self):
        """Fixture dla Enhanced Orchestrator"""
        return EnhancedOrchestrator()

    @pytest.fixture
    def mock_agent_factory(self):
        """Mock fabryki agentów"""
        with patch(
            "src.backend.agents.enhanced_orchestrator.AgentFactory"
        ) as mock_factory:
            yield mock_factory

    @pytest.fixture
    def mock_agent(self):
        """Mock agenta"""
        agent = Mock()
        agent.process = AsyncMock()
        return agent

    @pytest.mark.asyncio
    async def test_process_command_basic_flow(
        self, orchestrator, mock_agent_factory, mock_agent
    ):
        """Test podstawowego przepływu przetwarzania komendy"""
        # Given
        command = "What's the weather in Warsaw?"
        mock_agent_factory.create_agent.return_value = mock_agent
        mock_agent.process.return_value = {"weather": "sunny", "temperature": 25}

        # When
        response = await orchestrator.process_command(command)

        # Then
        assert response.success is True
        assert "weather" in response.data
        assert response.data["temperature"] == 25
        mock_agent_factory.create_agent.assert_called_once()
        mock_agent.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_command_with_multiple_agents(
        self, orchestrator, mock_agent_factory, mock_agent
    ):
        """Test przetwarzania komendy wymagającej wielu agentów"""
        # Given
        command = "Find recipes using tomatoes and check availability in local stores"

        # Konfiguracja mock'ów dla różnych agentów
        chef_agent = Mock()
        chef_agent.process = AsyncMock(return_value={"recipes": ["Pasta", "Salad"]})

        shopping_agent = Mock()
        shopping_agent.process = AsyncMock(
            return_value={"available": ["Pasta ingredients"]}
        )

        mock_agent_factory.create_agent.side_effect = [chef_agent, shopping_agent]

        # When
        response = await orchestrator.process_command(command)

        # Then
        assert response.success is True
        assert "recipes" in response.data
        assert "available" in response.data
        assert mock_agent_factory.create_agent.call_count == 2
        chef_agent.process.assert_called_once()
        shopping_agent.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_command_with_agent_error(
        self, orchestrator, mock_agent_factory, mock_agent
    ):
        """Test obsługi błędu agenta podczas przetwarzania"""
        # Given
        command = "Check weather in Berlin"
        mock_agent_factory.create_agent.return_value = mock_agent
        mock_agent.process.side_effect = Exception("Agent error")

        # When
        response = await orchestrator.process_command(command)

        # Then
        assert response.success is False
        assert "Agent error" in response.error
        assert response.error_type == OrchestratorError.AGENT_EXECUTION_ERROR

    @pytest.mark.asyncio
    async def test_process_command_with_fallback(
        self, orchestrator, mock_agent_factory, mock_agent
    ):
        """Test mechanizmu fallback w przypadku błędu"""
        # Given
        command = "Translate this text to French"
        mock_agent_factory.create_agent.return_value = mock_agent

        # Pierwszy agent zwraca błąd, drugi działa
        mock_agent.process.side_effect = [
            Exception("Translation error"),
            {"translation": "Bonjour"},
        ]

        # When
        response = await orchestrator.process_command(command)

        # Then
        assert response.success is True
        assert "translation" in response.data
        assert mock_agent_factory.create_agent.call_count == 2

    @pytest.mark.asyncio
    async def test_process_file_image_handling(
        self, orchestrator, mock_agent_factory, mock_agent
    ):
        """Test obsługi plików obrazów"""
        # Given
        file_path = "/path/to/receipt.jpg"
        file_type = "image/jpeg"
        mock_agent_factory.create_agent.return_value = mock_agent
        mock_agent.process.return_value = {"items": ["Milk", "Bread"], "total": 15.99}

        # When
        response = await orchestrator.process_file(file_path, file_type)

        # Then
        assert response.success is True
        assert "items" in response.data
        assert response.data["total"] == 15.99
        mock_agent_factory.create_agent.assert_called_with("OCR")
        mock_agent.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_file_pdf_handling(
        self, orchestrator, mock_agent_factory, mock_agent
    ):
        """Test obsługi plików PDF"""
        # Given
        file_path = "/path/to/document.pdf"
        file_type = "application/pdf"
        mock_agent_factory.create_agent.return_value = mock_agent
        mock_agent.process.return_value = {"text": "Extracted PDF content"}

        # When
        response = await orchestrator.process_file(file_path, file_type)

        # Then
        assert response.success is True
        assert "text" in response.data
        mock_agent_factory.create_agent.assert_called_with("DocumentProcessing")
        mock_agent.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_file_unsupported_type(self, orchestrator):
        """Test obsługi nieobsługiwanego typu pliku"""
        # Given
        file_path = "/path/to/file.xyz"
        file_type = "application/xyz"

        # When
        response = await orchestrator.process_file(file_path, file_type)

        # Then
        assert response.success is False
        assert "Unsupported file type" in response.error
        assert response.error_type == OrchestratorError.UNSUPPORTED_FILE_TYPE

    @pytest.mark.asyncio
    async def test_handle_conversation_context(
        self, orchestrator, mock_agent_factory, mock_agent
    ):
        """Test zarządzania kontekstem konwersacji"""
        # Given
        conversation = [
            {"role": "user", "content": "What's the weather in Krakow?"},
            {"role": "assistant", "content": "It's sunny, 22°C"},
            {"role": "user", "content": "What about tomorrow?"},
        ]
        mock_agent_factory.create_agent.return_value = mock_agent
        mock_agent.process.return_value = {"weather": "cloudy", "temperature": 20}

        # When
        response = await orchestrator.handle_conversation(conversation)

        # Then
        assert response.success is True
        assert "weather" in response.data
        assert (
            mock_agent.process.call_args[0][0]["context"]
            == "What's the weather in Krakow? It's sunny, 22°C What about tomorrow?"
        )

    @pytest.mark.asyncio
    async def test_handle_conversation_with_state(self, orchestrator):
        """Test zarządzania stanem podczas konwersacji"""
        # Given
        conversation = [
            {"role": "user", "content": "Start shopping list"},
            {"role": "assistant", "content": "What would you like to add?"},
            {"role": "user", "content": "Add milk and eggs"},
        ]

        # When
        await orchestrator.handle_conversation(conversation)

        # Then
        assert orchestrator.conversation_state is not None
        assert "shopping" in orchestrator.conversation_state

    @pytest.mark.asyncio
    async def test_error_handling_invalid_command(self, orchestrator):
        """Test obsługi nieprawidłowej komendy"""
        # Given
        command = ""

        # When
        response = await orchestrator.process_command(command)

        # Then
        assert response.success is False
        assert "Invalid command" in response.error
        assert response.error_type == OrchestratorError.INVALID_INPUT

    @pytest.mark.asyncio
    async def test_performance_monitoring(
        self, orchestrator, mock_agent_factory, mock_agent
    ):
        """Test monitorowania wydajności"""
        # Given
        command = "Test performance"
        mock_agent_factory.create_agent.return_value = mock_agent

        # When
        await orchestrator.process_command(command)

        # Then
        assert orchestrator.performance_metrics.total_requests == 1
        assert orchestrator.performance_metrics.last_processing_time > 0

    @pytest.mark.asyncio
    async def test_concurrent_processing(
        self, orchestrator, mock_agent_factory, mock_agent
    ):
        """Test równoczesnego przetwarzania wielu żądań"""
        # Given
        commands = ["Command 1", "Command 2", "Command 3"]
        mock_agent_factory.create_agent.return_value = mock_agent

        # When
        tasks = [orchestrator.process_command(cmd) for cmd in commands]
        responses = await asyncio.gather(*tasks)

        # Then
        assert len(responses) == 3
        assert all(resp.success is True for resp in responses)
        assert orchestrator.performance_metrics.total_requests == 3

    @pytest.mark.asyncio
    async def test_agent_selection_logic(self, orchestrator, mock_agent_factory):
        """Test logiki wyboru agenta"""
        # Given
        test_cases = [
            ("What's the weather?", "Weather"),
            ("Search for AI news", "Search"),
            ("Read this receipt", "OCR"),
            ("Plan meals for next week", "MealPlanner"),
        ]

        for command, expected_agent in test_cases:
            # When
            await orchestrator.process_command(command)

            # Then
            mock_agent_factory.create_agent.assert_called_with(expected_agent)
            mock_agent_factory.reset_mock()

    @pytest.mark.asyncio
    async def test_alerting_system(self, orchestrator, mock_agent_factory, mock_agent):
        """Test systemu alertowania dla błędów krytycznych"""
        # Given
        command = "Critical command"
        mock_agent_factory.create_agent.return_value = mock_agent
        mock_agent.process.side_effect = Exception("Critical error")

        # When
        response = await orchestrator.process_command(command)

        # Then
        assert response.requires_alert is True
        assert orchestrator.alerts_sent > 0

    @pytest.mark.asyncio
    async def test_retry_mechanism(self, orchestrator, mock_agent_factory, mock_agent):
        """Test mechanizmu ponawiania prób"""
        # Given
        command = "Unstable command"
        mock_agent_factory.create_agent.return_value = mock_agent

        # Symulacja: pierwsza próba nieudana, druga udana
        mock_agent.process.side_effect = [
            Exception("Temporary error"),
            {"result": "Success after retry"},
        ]

        # When
        response = await orchestrator.process_command(command)

        # Then
        assert response.success is True
        assert mock_agent.process.call_count == 2

    @pytest.mark.asyncio
    async def test_state_persistence(self, orchestrator):
        """Test utrwalania stanu orkiestratora"""
        # Given
        orchestrator.conversation_state = {"key": "value"}

        # When
        await orchestrator.save_state()
        new_orchestrator = EnhancedOrchestrator()
        await new_orchestrator.load_state()

        # Then
        assert new_orchestrator.conversation_state == {"key": "value"}

    @pytest.mark.asyncio
    async def test_timeout_handling(self, orchestrator, mock_agent_factory, mock_agent):
        """Test obsługi przekroczenia czasu"""
        # Given
        command = "Slow command"
        mock_agent_factory.create_agent.return_value = mock_agent

        # Symulacja długiego przetwarzania
        async def slow_processing():
            await asyncio.sleep(10)
            return {"result": "Too late"}

        mock_agent.process = slow_processing

        # Ustawienie krótkiego timeoutu
        orchestrator.TIMEOUT = 0.1

        # When
        response = await orchestrator.process_command(command)

        # Then
        assert response.success is False
        assert "Timeout" in response.error
        assert response.error_type == OrchestratorError.TIMEOUT
