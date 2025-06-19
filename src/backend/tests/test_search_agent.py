from unittest.mock import AsyncMock, patch

import pytest

from backend.agents.base_agent import AgentResponse
from backend.agents.search_agent import SearchAgent


@pytest.mark.asyncio
async def test_search_agent_with_results():
    agent = SearchAgent()
    mock_context = {"query": "test query", "model": "llama3"}

    with patch(
        "backend.agents.search_agent.SearchAgent._perform_search"
    ) as mock_search:
        mock_search.return_value = [
            {
                "title": "Test Result",
                "url": "http://test.com",
                "snippet": "Test snippet",
            }
        ]

        # Mock streaming response
        mock_stream = AsyncMock()
        mock_stream.__aiter__.return_value = [
            {"message": {"content": "Search result 1"}},
            {"message": {"content": "Search result 2"}},
        ]

        with patch("backend.core.llm_client.llm_client.chat", return_value=mock_stream):
            response = await agent.process(mock_context)
            assert isinstance(response, AgentResponse)
            assert response.success is True

            content = ""
            async for chunk in response.text_stream:
                content += chunk
            assert "Search result 1Search result 2" in content


@pytest.mark.asyncio
async def test_search_agent_empty_results():
    agent = SearchAgent()
    mock_context = {"query": "test query"}

    with patch(
        "backend.agents.search_agent.SearchAgent._perform_search"
    ) as mock_search:
        mock_search.return_value = []

        response = await agent.process(mock_context)
        assert response.success is True  # Empty results are handled gracefully
        assert "Brak wyników" in response.text


@pytest.mark.asyncio
async def test_search_agent_input_validation():
    agent = SearchAgent()

    # Test empty query
    mock_context = {"query": ""}
    response = await agent.process(mock_context)
    assert response.success is False
    assert "Brak zapytania" in response.error

    # Test invalid input type
    mock_context = {"query": 123}
    response = await agent.process(mock_context)
    assert response.success is False
    assert "Nieprawidłowy typ zapytania" in response.error


@pytest.mark.asyncio
async def test_search_agent_llm_error():
    agent = SearchAgent()
    mock_context = {"query": "test query", "model": "llama3"}

    with patch(
        "backend.agents.search_agent.SearchAgent._perform_search"
    ) as mock_search:
        mock_search.return_value = [{"title": "Test", "url": "http://test.com"}]

        # Mock streaming error
        mock_stream = AsyncMock()
        mock_stream.__aiter__.side_effect = Exception("LLM error")

        with patch("backend.core.llm_client.llm_client.chat", return_value=mock_stream):
            response = await agent.process(mock_context)
            assert response.success is False
            assert "LLM error" in response.error
