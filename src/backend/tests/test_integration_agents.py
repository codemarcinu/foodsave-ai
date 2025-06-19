import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from backend.agents.base_agent import AgentResponse
from backend.agents.chef_agent import ChefAgent
from backend.agents.enhanced_orchestrator import EnhancedOrchestrator
from backend.agents.ocr_agent import OCRAgent
from backend.agents.search_agent import SearchAgent
from backend.agents.weather_agent import WeatherAgent


@pytest.mark.asyncio
async def test_integration_chat_flow():
    orchestrator = EnhancedOrchestrator()

    # Mock all agents
    with patch("backend.agents.enhanced_orchestrator.ChefAgent") as mock_chef, patch(
        "backend.agents.enhanced_orchestrator.OCRAgent"
    ) as mock_ocr, patch(
        "backend.agents.enhanced_orchestrator.SearchAgent"
    ) as mock_search, patch(
        "backend.agents.enhanced_orchestrator.WeatherAgent"
    ) as mock_weather:

        # Setup mock responses
        mock_chef.return_value.process.return_value = AgentResponse(
            success=True, message="Recipe suggestion"
        )
        mock_ocr.return_value.process.return_value = AgentResponse(
            success=True, message="Extracted text"
        )
        mock_search.return_value.process.return_value = AgentResponse(
            success=True, message="Search results"
        )
        mock_weather.return_value.process.return_value = AgentResponse(
            success=True, message="Weather forecast"
        )

        # Test cooking flow
        response = await orchestrator.process(
            {"intent": "cooking", "ingredients": ["chicken", "rice"]}
        )
        assert response.success
        assert "Recipe suggestion" in response.message

        # Test OCR flow
        response = await orchestrator.process(
            {"intent": "receipt", "file_bytes": b"test", "file_type": "image"}
        )
        assert response.success
        assert "Extracted text" in response.message

        # Test search flow
        response = await orchestrator.process(
            {"intent": "search", "query": "test query"}
        )
        assert response.success
        assert "Search results" in response.message

        # Test weather flow
        response = await orchestrator.process(
            {"intent": "weather", "query": "weather in Warsaw"}
        )
        assert response.success
        assert "Weather forecast" in response.message


@pytest.mark.asyncio
async def test_integration_error_handling():
    orchestrator = EnhancedOrchestrator()

    with patch("backend.agents.enhanced_orchestrator.ChefAgent") as mock_chef:
        mock_chef.return_value.process.return_value = AgentResponse(
            success=False, error="No ingredients provided"
        )

        response = await orchestrator.process({"intent": "cooking", "ingredients": []})
        assert not response.success
        assert "No ingredients provided" in response.error


@pytest.mark.asyncio
async def test_integration_combined_flow():
    """Test złożonego przepływu używającego wielu agentów"""
    orchestrator = EnhancedOrchestrator()

    with patch("backend.agents.enhanced_orchestrator.ChefAgent") as mock_chef, patch(
        "backend.agents.enhanced_orchestrator.WeatherAgent"
    ) as mock_weather:

        mock_chef.return_value.process.return_value = AgentResponse(
            success=True, message="Summer salad recipe"
        )
        mock_weather.return_value.process.return_value = AgentResponse(
            success=True, message="Sunny, 25°C"
        )

        # Przepływ: pogoda -> przepis dopasowany do pogody
        weather_response = await orchestrator.process(
            {"intent": "weather", "query": "weather in Warsaw"}
        )
        assert weather_response.success

        recipe_response = await orchestrator.process(
            {
                "intent": "cooking",
                "ingredients": ["tomato", "cucumber"],
                "context": {"weather": "sunny"},
            }
        )
        assert recipe_response.success
        assert "Summer salad" in recipe_response.message


@pytest.mark.asyncio
async def test_integration_performance():
    """Test wydajnościowy z wieloma równoległymi żądaniami"""
    orchestrator = EnhancedOrchestrator()

    with patch("backend.agents.enhanced_orchestrator.SearchAgent") as mock_search:
        mock_search.return_value.process.return_value = AgentResponse(
            success=True, message="Search results"
        )

        # Symulacja wielu równoległych żądań
        tasks = [
            orchestrator.process({"intent": "search", "query": f"test query {i}"})
            for i in range(5)
        ]

        responses = await asyncio.gather(*tasks)
        assert all(r.success for r in responses)


@pytest.mark.asyncio
async def test_integration_fallback_mechanism():
    """Test mechanizmu fallback gdy główny agent zawiedzie"""
    orchestrator = EnhancedOrchestrator()

    with patch(
        "backend.agents.enhanced_orchestrator.SearchAgent"
    ) as mock_search, patch(
        "backend.agents.enhanced_orchestrator.EnhancedRAGAgent"
    ) as mock_rag:

        mock_search.return_value.process.return_value = AgentResponse(
            success=False, error="Search failed"
        )
        mock_rag.return_value.process.return_value = AgentResponse(
            success=True, message="Fallback answer from RAG"
        )

        response = await orchestrator.process(
            {"intent": "search", "query": "important query"}
        )
        assert response.success
        assert "Fallback answer" in response.message
