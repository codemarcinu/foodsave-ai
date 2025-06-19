import asyncio
import os

# Dodanie ścieżki do sys.path dla importów
import sys
from typing import Any, AsyncGenerator, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))

from backend.agents.agent_factory import AgentFactory
from backend.agents.enhanced_base_agent import EnhancedBaseAgent
from backend.agents.enhanced_orchestrator import EnhancedOrchestrator
from backend.agents.error_types import AgentError
from backend.agents.ocr_agent import OCRAgent
from backend.agents.search_agent import SearchAgent


class TestAgentFactory:
    """Testy dla Agent Factory - fabryki tworzenia agentów"""

    @pytest.fixture
    def agent_factory(self):
        """Fixture dla Agent Factory"""
        return AgentFactory()

    @pytest.fixture
    def mock_llm_client(self):
        """Mock klienta LLM"""
        with patch("backend.agents.agent_factory.llm_client") as mock_client:
            yield mock_client

    @pytest.fixture
    def mock_config(self):
        """Mock konfiguracji"""
        with patch("backend.agents.agent_factory.config") as mock_config:
            mock_config.AGENT_CONFIG = {
                "Weather": {"module": "weather_agent", "class": "EnhancedWeatherAgent"},
                "Search": {"module": "search_agent", "class": "SearchAgent"},
                "OCR": {"module": "ocr_agent", "class": "OCRAgent"},
            }
            yield mock_config

    @pytest.mark.asyncio
    async def test_create_agent_success(self, agent_factory, mock_config):
        """Test pomyślnego tworzenia agenta"""
        # Given
        agent_type = "Search"

        # When
        agent = agent_factory.create_agent(agent_type)

        # Then
        assert agent is not None
        assert isinstance(agent, SearchAgent)

    @pytest.mark.asyncio
    async def test_create_agent_invalid_type(self, agent_factory):
        """Test tworzenia agenta o nieprawidłowym typie"""
        # Given
        agent_type = "InvalidAgentType"

        # When
        with pytest.raises(ValueError) as excinfo:
            agent_factory.create_agent(agent_type)

        # Then
        assert "Unsupported agent type" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_create_orchestrator(self, agent_factory):
        """Test tworzenia orkiestratora"""
        # When
        orchestrator = agent_factory.create_agent("Orchestrator")

        # Then
        assert orchestrator is not None
        assert isinstance(orchestrator, EnhancedOrchestrator)

    @pytest.mark.asyncio
    async def test_create_agent_with_dependencies(
        self, agent_factory, mock_config, mock_llm_client
    ):
        """Test tworzenia agenta z wstrzykiwaniem zależności"""
        # Given
        agent_type = "Weather"
        dependencies = {"llm_client": mock_llm_client}

        # When
        agent = agent_factory.create_agent(agent_type, **dependencies)

        # Then
        assert agent is not None
        assert hasattr(agent, "llm_client")
        assert agent.llm_client == mock_llm_client

    @pytest.mark.asyncio
    async def test_create_agent_with_configuration(self, agent_factory, mock_config):
        """Test tworzenia agenta z konfiguracją"""
        # Given
        agent_type = "OCR"
        config = {"timeout": 30, "language": "pol"}

        # When
        agent = agent_factory.create_agent(agent_type, **config)

        # Then
        assert agent is not None
        assert agent.TIMEOUT == 30
        assert agent.default_language == "pol"

    @pytest.mark.asyncio
    async def test_create_agent_caching(self, agent_factory, mock_config):
        """Test mechanizmu cache'owania agentów"""
        # Given
        agent_type = "Search"

        # When
        agent1 = agent_factory.create_agent(agent_type)
        agent2 = agent_factory.create_agent(agent_type)

        # Then
        assert agent1 is agent2  # Powinien być ten sam obiekt

    @pytest.mark.asyncio
    async def test_create_agent_without_caching(self, agent_factory, mock_config):
        """Test tworzenia nowej instancji agenta bez cache'owania"""
        # Given
        agent_type = "Search"

        # When
        agent1 = agent_factory.create_agent(agent_type)
        agent2 = agent_factory.create_agent(agent_type, use_cache=False)

        # Then
        assert agent1 is not agent2  # Powinny być różne instancje

    @pytest.mark.asyncio
    async def test_create_agent_with_custom_modules(self, agent_factory, mock_config):
        """Test tworzenia agenta z niestandardowych modułów"""
        # Given
        agent_type = "CustomAgent"
        custom_config = {"module": "custom_agent_module", "class": "CustomAgentClass"}
        agent_factory.agent_config[agent_type] = custom_config

        with patch("importlib.import_module") as mock_import:
            mock_module = Mock()
            mock_module.CustomAgentClass = Mock(return_value="CustomAgentInstance")
            mock_import.return_value = mock_module

            # When
            agent = agent_factory.create_agent(agent_type)

            # Then
            assert agent == "CustomAgentInstance"

    @pytest.mark.asyncio
    async def test_create_agent_with_fallback(self, agent_factory, mock_config):
        """Test mechanizmu fallback przy tworzeniu agenta"""
        # Given
        agent_type = "Weather"

        # Symulacja błędu przy tworzeniu głównego agenta
        with patch("importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("Module not found")

            # Konfiguracja fallback
            agent_factory.agent_config[agent_type]["fallback"] = {
                "module": "search_agent",
                "class": "SearchAgent",
            }

            # When
            agent = agent_factory.create_agent(agent_type)

            # Then
            assert isinstance(agent, SearchAgent)

    @pytest.mark.asyncio
    async def test_create_agent_with_initialization_error(
        self, agent_factory, mock_config
    ):
        """Test błędu inicjalizacji agenta"""
        # Given
        agent_type = "OCR"

        with patch(
            "backend.agents.agent_factory.importlib.import_module"
        ) as mock_import:
            mock_module = Mock()
            mock_module.OCRAgent = Mock(side_effect=Exception("Init error"))
            mock_import.return_value = mock_module

            # When
            with pytest.raises(AgentError) as excinfo:
                agent_factory.create_agent(agent_type)

            # Then
            assert "Failed to create agent" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_create_agent_with_invalid_config(self, agent_factory):
        """Test tworzenia agenta z nieprawidłową konfiguracją"""
        # Given
        agent_type = "Search"
        agent_factory.agent_config[agent_type] = {"module": "invalid_module"}

        # When
        with pytest.raises(ValueError) as excinfo:
            agent_factory.create_agent(agent_type)

        # Then
        assert "Invalid configuration" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_create_agent_with_async_initialization(
        self, agent_factory, mock_config
    ):
        """Test asynchronicznej inicjalizacji agenta"""
        # Given
        agent_type = "Search"

        async def async_init():
            await asyncio.sleep(0.1)
            return "AsyncAgent"

        with patch(
            "backend.agents.agent_factory.importlib.import_module"
        ) as mock_import:
            mock_module = Mock()
            mock_module.SearchAgent = Mock(return_value=async_init())
            mock_import.return_value = mock_module

            # When
            agent = agent_factory.create_agent(agent_type)

            # Then
            assert asyncio.iscoroutine(agent)
            result = await agent
            assert result == "AsyncAgent"

    @pytest.mark.asyncio
    async def test_create_multiple_agent_types(self, agent_factory, mock_config):
        """Test tworzenia różnych typów agentów"""
        # Given
        agent_types = ["Weather", "Search", "OCR"]

        # When
        agents = [agent_factory.create_agent(t) for t in agent_types]

        # Then
        assert len(agents) == 3
        assert all(agent is not None for agent in agents)
        assert isinstance(
            agents[0], EnhancedBaseAgent
        )  # WeatherAgent nie jest zdefiniowany, ale zakładamy że dziedziczy po EnhancedBaseAgent
        assert isinstance(agents[1], SearchAgent)
        assert isinstance(agents[2], OCRAgent)

    @pytest.mark.asyncio
    async def test_create_agent_with_plugins(self, agent_factory, mock_config):
        """Test tworzenia agenta z pluginami"""
        # Given
        agent_type = "Search"
        plugins = [Mock(), Mock()]

        # When
        agent = agent_factory.create_agent(agent_type, plugins=plugins)

        # Then
        assert agent is not None
        assert hasattr(agent, "plugins")
        assert len(agent.plugins) == 2

    @pytest.mark.asyncio
    async def test_create_agent_with_state(self, agent_factory, mock_config):
        """Test tworzenia agenta z przekazanym stanem"""
        # Given
        agent_type = "Weather"
        initial_state = {"location": "Warsaw"}

        # When
        agent = agent_factory.create_agent(agent_type, initial_state=initial_state)

        # Then
        assert agent is not None
        assert agent.state == initial_state

    @pytest.mark.asyncio
    async def test_create_agent_performance(self, agent_factory, mock_config):
        """Test wydajności tworzenia agentów"""
        # Given
        agent_type = "Search"

        # When
        start_time = asyncio.get_event_loop().time()
        for _ in range(100):
            agent_factory.create_agent(agent_type, use_cache=False)
        end_time = asyncio.get_event_loop().time()

        # Then
        duration = end_time - start_time
        assert duration < 1.0  # Tworzenie 100 agentów powinno trwać mniej niż 1 sekundę
