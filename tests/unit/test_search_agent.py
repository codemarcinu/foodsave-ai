import asyncio
import os

# Dodanie ścieżki do sys.path dla importów
import sys
from typing import Any, AsyncGenerator, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))

from backend.agents.error_types import AgentError
from backend.agents.search_agent import SearchAgent
from backend.integrations.web_search import WebSearch


class TestSearchAgent:
    """Testy dla Search Agent - agenta wyszukiwania internetowego"""

    @pytest.fixture
    def search_agent(self):
        """Fixture dla Search Agent"""
        return SearchAgent()

    @pytest.fixture
    def mock_web_search(self):
        """Mock usługi wyszukiwania internetowego"""
        with patch("backend.agents.search_agent.WebSearch") as mock_search:
            mock_instance = mock_search.return_value
            mock_instance.search = AsyncMock()
            yield mock_instance

    @pytest.fixture
    def mock_llm_client(self):
        """Mock klienta LLM"""
        with patch("backend.agents.search_agent.llm_client") as mock_client:
            mock_client.chat.return_value = {
                "message": {"content": "Refined search results"}
            }
            yield mock_client

    @pytest.mark.asyncio
    async def test_web_search_success(self, search_agent, mock_web_search):
        """Test pomyślnego wyszukiwania internetowego"""
        # Given
        input_data = {"query": "weather in Warsaw"}
        mock_web_search.search.return_value = [
            {
                "title": "Weather Forecast",
                "url": "https://example.com/weather",
                "snippet": "Current weather...",
            }
        ]

        # When
        response = await search_agent.process(input_data)

        # Then
        assert response.success is True
        assert "results" in response.data
        assert len(response.data["results"]) == 1
        assert response.data["results"][0]["title"] == "Weather Forecast"

    @pytest.mark.asyncio
    async def test_web_search_with_refinement(
        self, search_agent, mock_web_search, mock_llm_client
    ):
        """Test wyszukiwania z refinem za pomocą LLM"""
        # Given
        input_data = {"query": "latest AI news", "refine_results": True}
        mock_web_search.search.return_value = [
            {"title": "AI Breakthrough", "snippet": "New AI model..."},
            {"title": "Tech News", "snippet": "Recent developments..."},
        ]

        # When
        response = await search_agent.process(input_data)

        # Then
        assert response.success is True
        assert "refined_results" in response.data
        assert response.data["refined_results"] == "Refined search results"
        mock_llm_client.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_web_search_with_summarization(
        self, search_agent, mock_web_search, mock_llm_client
    ):
        """Test podsumowania wyników wyszukiwania"""
        # Given
        input_data = {"query": "climate change impacts", "summarize": True}
        mock_web_search.search.return_value = [
            {"title": "Impact Report", "snippet": "Sea levels rising..."},
            {"title": "Research Study", "snippet": "Temperature changes..."},
        ]

        # When
        response = await search_agent.process(input_data)

        # Then
        assert response.success is True
        assert "summary" in response.data
        assert response.data["summary"] == "Refined search results"

    @pytest.mark.asyncio
    async def test_web_search_with_time_filter(self, search_agent, mock_web_search):
        """Test wyszukiwania z filtrem czasowym"""
        # Given
        input_data = {"query": "Python updates", "time_range": "week"}

        # When
        await search_agent.process(input_data)

        # Then
        call_args = mock_web_search.search.call_args
        assert call_args[1]["time_range"] == "week"

    @pytest.mark.asyncio
    async def test_web_search_with_region_filter(self, search_agent, mock_web_search):
        """Test wyszukiwania z filtrem regionalnym"""
        # Given
        input_data = {"query": "local news", "region": "pl"}

        # When
        await search_agent.process(input_data)

        # Then
        call_args = mock_web_search.search.call_args
        assert call_args[1]["region"] == "pl"

    @pytest.mark.asyncio
    async def test_web_search_with_site_filter(self, search_agent, mock_web_search):
        """Test wyszukiwania z filtrem domeny"""
        # Given
        input_data = {"query": "AI research", "site": "arxiv.org"}

        # When
        await search_agent.process(input_data)

        # Then
        call_args = mock_web_search.search.call_args
        assert call_args[1]["site"] == "arxiv.org"

    @pytest.mark.asyncio
    async def test_web_search_with_type_filter(self, search_agent, mock_web_search):
        """Test wyszukiwania z filtrem typu"""
        # Given
        input_data = {"query": "Python tutorial", "type": "video"}

        # When
        await search_agent.process(input_data)

        # Then
        call_args = mock_web_search.search.call_args
        assert call_args[1]["type"] == "video"

    @pytest.mark.asyncio
    async def test_web_search_with_num_results(self, search_agent, mock_web_search):
        """Test wyszukiwania z określoną liczbą wyników"""
        # Given
        input_data = {"query": "machine learning", "num_results": 20}

        # When
        await search_agent.process(input_data)

        # Then
        call_args = mock_web_search.search.call_args
        assert call_args[1]["num_results"] == 20

    @pytest.mark.asyncio
    async def test_web_search_error_handling(self, search_agent, mock_web_search):
        """Test obsługi błędów wyszukiwania"""
        # Given
        input_data = {"query": "test error"}
        mock_web_search.search.side_effect = Exception("Search error")

        # When
        response = await search_agent.process(input_data)

        # Then
        assert response.success is False
        assert "Search error" in response.error
        assert response.error_type == AgentError.SEARCH_ERROR

    @pytest.mark.asyncio
    async def test_web_search_with_fallback(self, search_agent, mock_web_search):
        """Test mechanizmu fallback w przypadku błędu"""
        # Given
        input_data = {"query": "fallback test"}
        mock_web_search.search.side_effect = [
            Exception("Primary search error"),
            [{"title": "Fallback result"}],
        ]

        # When
        response = await search_agent.process(input_data)

        # Then
        assert response.success is True
        assert len(response.data["results"]) == 1
        assert mock_web_search.search.call_count == 2

    @pytest.mark.asyncio
    async def test_web_search_with_cache(self, search_agent, mock_web_search):
        """Test użycia cache wyszukiwania"""
        # Given
        input_data = {"query": "cached query"}

        # When
        # Pierwsze wyszukiwanie
        await search_agent.process(input_data)
        # Drugie wyszukiwanie (powinno użyć cache)
        await search_agent.process(input_data)

        # Then
        assert mock_web_search.search.call_count == 1

    @pytest.mark.asyncio
    async def test_web_search_without_cache(self, search_agent, mock_web_search):
        """Test wyszukiwania bez cache"""
        # Given
        input_data = {"query": "no cache query", "use_cache": False}

        # When
        await search_agent.process(input_data)
        await search_agent.process(input_data)

        # Then
        assert mock_web_search.search.call_count == 2

    @pytest.mark.asyncio
    async def test_news_search(self, search_agent, mock_web_search):
        """Test wyszukiwania wiadomości"""
        # Given
        input_data = {"query": "breaking news", "search_type": "news"}

        # When
        await search_agent.process(input_data)

        # Then
        call_args = mock_web_search.search.call_args
        assert call_args[1]["search_type"] == "news"

    @pytest.mark.asyncio
    async def test_image_search(self, search_agent, mock_web_search):
        """Test wyszukiwania obrazów"""
        # Given
        input_data = {"query": "cats", "search_type": "images"}

        # When
        await search_agent.process(input_data)

        # Then
        call_args = mock_web_search.search.call_args
        assert call_args[1]["search_type"] == "images"

    @pytest.mark.asyncio
    async def test_video_search(self, search_agent, mock_web_search):
        """Test wyszukiwania wideo"""
        # Given
        input_data = {"query": "tutorial", "search_type": "videos"}

        # When
        await search_agent.process(input_data)

        # Then
        call_args = mock_web_search.search.call_args
        assert call_args[1]["search_type"] == "videos"

    @pytest.mark.asyncio
    async def test_safe_search(self, search_agent, mock_web_search):
        """Test bezpiecznego wyszukiwania"""
        # Given
        input_data = {"query": "sensitive content", "safe_search": True}

        # When
        await search_agent.process(input_data)

        # Then
        call_args = mock_web_search.search.call_args
        assert call_args[1]["safe_search"] is True

    @pytest.mark.asyncio
    async def test_search_with_pagination(self, search_agent, mock_web_search):
        """Test wyszukiwania z paginacją"""
        # Given
        input_data = {"query": "long list", "page": 2}

        # When
        await search_agent.process(input_data)

        # Then
        call_args = mock_web_search.search.call_args
        assert call_args[1]["page"] == 2

    @pytest.mark.asyncio
    async def test_search_performance(self, search_agent, mock_web_search):
        """Test wydajności wyszukiwania"""
        # Given
        input_data = {"query": "performance test"}

        # When
        start_time = asyncio.get_event_loop().time()
        await search_agent.process(input_data)
        end_time = asyncio.get_event_loop().time()

        # Then
        duration = end_time - start_time
        assert duration < 2.0  # Wyszukiwanie powinno trwać mniej niż 2 sekundy

    @pytest.mark.asyncio
    async def test_search_with_language_filter(self, search_agent, mock_web_search):
        """Test wyszukiwania z filtrem językowym"""
        # Given
        input_data = {"query": "warsaw events", "language": "pl"}

        # When
        await search_agent.process(input_data)

        # Then
        call_args = mock_web_search.search.call_args
        assert call_args[1]["language"] == "pl"

    @pytest.mark.asyncio
    async def test_search_with_multiple_filters(self, search_agent, mock_web_search):
        """Test wyszukiwania z wieloma filtrami"""
        # Given
        input_data = {
            "query": "AI conference",
            "time_range": "year",
            "region": "us",
            "num_results": 15,
            "type": "news",
        }

        # When
        await search_agent.process(input_data)

        # Then
        call_args = mock_web_search.search.call_args
        assert call_args[1]["time_range"] == "year"
        assert call_args[1]["region"] == "us"
        assert call_args[1]["num_results"] == 15
        assert call_args[1]["type"] == "news"
