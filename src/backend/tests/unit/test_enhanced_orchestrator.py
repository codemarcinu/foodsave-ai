import asyncio
from datetime import datetime
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.agents.enhanced_orchestrator import EnhancedOrchestrator
from backend.core.hybrid_llm_client import ModelComplexity
from backend.core.memory import ConversationMemoryManager
from backend.models.conversation import Conversation
from backend.models.user_profile import InteractionType


@pytest.fixture
def db_session():
    return AsyncMock()


@pytest.fixture
def weather_agent():
    return AsyncMock()


@pytest.fixture
def mock_conversation():
    return Conversation(id=1, session_id="test_session")


@pytest.fixture
def orchestrator(db_session, weather_agent):
    orchestrator = EnhancedOrchestrator(db_session)
    orchestrator.weather_agent = weather_agent
    return orchestrator


@pytest.mark.asyncio
async def test_process_command_basic(orchestrator, db_session, mock_conversation):
    """Test basic command processing"""
    with patch(
        "backend.core.crud.get_conversation_by_session_id",
        new_callable=AsyncMock,
        return_value=mock_conversation,
    ):
        with patch.object(
            ConversationMemoryManager,
            "retrieve_context",
            new_callable=AsyncMock,
            return_value=[],
        ):
            with patch.object(
                EnhancedOrchestrator, "_route_by_intent", new_callable=AsyncMock
            ) as mock_route:
                mock_route.return_value = {"response": "test response"}

                result = await orchestrator.process_command(
                    "test command", "test_session"
                )

                assert result["response"] == "test response"
                mock_route.assert_awaited_once()


@pytest.mark.asyncio
async def test_model_selection(orchestrator, db_session, mock_conversation):
    """Test model selection based on complexity"""
    test_cases = [
        (ModelComplexity.SIMPLE, "gemma:2b"),
        (ModelComplexity.STANDARD, "gemma3:latest"),
        (ModelComplexity.COMPLEX, "gemma3:latest"),
    ]

    for complexity, expected_model in test_cases:
        with patch(
            "backend.core.crud.get_conversation_by_session_id",
            new_callable=AsyncMock,
            return_value=mock_conversation,
        ):
            with patch.object(
                ConversationMemoryManager,
                "retrieve_context",
                new_callable=AsyncMock,
                return_value=[],
            ):
                with patch.object(
                    EnhancedOrchestrator,
                    "_determine_command_complexity",
                    new_callable=AsyncMock,
                    return_value=complexity,
                ):
                    with patch.object(
                        EnhancedOrchestrator, "_route_by_intent", new_callable=AsyncMock
                    ) as mock_route:
                        mock_route.return_value = {"response": "test"}

                        await orchestrator.process_command(
                            "test command", "test_session"
                        )

                        # Verify model selection in general chat
                        if mock_route.call_args[0][1].get("intent") == "general_chat":
                            assert mock_route.call_args[0][3] == complexity
                            assert (
                                mock_route.call_args[1]["model_used"] == expected_model
                            )


@pytest.mark.asyncio
async def test_error_handling(orchestrator, db_session):
    """Test error handling during command processing"""
    with patch(
        "backend.core.crud.get_conversation_by_session_id",
        new_callable=AsyncMock,
        side_effect=Exception("Test error"),
    ):
        result = await orchestrator.process_command("test command", "test_session")

        assert "Brak dostępnych agentów" in result["response"]


@pytest.mark.asyncio
async def test_weather_handling(orchestrator, db_session, mock_conversation):
    """Test weather command routing"""
    with patch(
        "backend.core.crud.get_conversation_by_session_id",
        new_callable=AsyncMock,
        return_value=mock_conversation,
    ):
        with patch.object(
            ConversationMemoryManager,
            "retrieve_context",
            new_callable=AsyncMock,
            return_value=[],
        ):
            with patch.object(
                EnhancedOrchestrator, "_detect_intent", new_callable=AsyncMock
            ) as mock_detect:
                mock_detect.return_value = {
                    "intent": "weather",
                    "requires_clarification": False,
                }

                with patch.object(
                    EnhancedOrchestrator, "_handle_weather", new_callable=AsyncMock
                ) as mock_weather:
                    mock_weather.return_value = {"response": "weather response"}

                    result = await orchestrator.process_command(
                        "jaka jest pogoda w Warszawie?", "test_session"
                    )

                    assert result["response"] == "weather response"
                    mock_weather.assert_awaited_once()
                    # Verify Bielik model is used
                    assert (
                        mock_weather.call_args[0][2]
                        == "SpeakLeash/bielik-11b-v2.3-instruct:Q8_0"
                    )


@pytest.mark.asyncio
async def test_search_handling(orchestrator, db_session, mock_conversation):
    """Test search command routing"""
    with patch(
        "backend.core.crud.get_conversation_by_session_id",
        new_callable=AsyncMock,
        return_value=mock_conversation,
    ):
        with patch.object(
            ConversationMemoryManager,
            "retrieve_context",
            new_callable=AsyncMock,
            return_value=[],
        ):
            with patch.object(
                EnhancedOrchestrator, "_detect_intent", new_callable=AsyncMock
            ) as mock_detect:
                mock_detect.return_value = {
                    "intent": "search",
                    "requires_clarification": False,
                }

                with patch.object(
                    EnhancedOrchestrator, "_handle_search", new_callable=AsyncMock
                ) as mock_search:
                    mock_search.return_value = {"response": "search response"}

                    result = await orchestrator.process_command(
                        "wyszukaj informacje o Pythonie", "test_session"
                    )

                    assert result["response"] == "search response"
                    mock_search.assert_awaited_once()
                    # Verify Bielik model is used
                    assert (
                        mock_search.call_args[0][2]
                        == "SpeakLeash/bielik-11b-v2.3-instruct:Q8_0"
                    )


@pytest.mark.asyncio
async def test_rag_handling(orchestrator, db_session, mock_conversation):
    """Test RAG command routing"""
    with patch(
        "backend.core.crud.get_conversation_by_session_id",
        new_callable=AsyncMock,
        return_value=mock_conversation,
    ):
        with patch.object(
            ConversationMemoryManager,
            "retrieve_context",
            new_callable=AsyncMock,
            return_value=[],
        ):
            with patch.object(
                EnhancedOrchestrator, "_detect_intent", new_callable=AsyncMock
            ) as mock_detect:
                mock_detect.return_value = {
                    "intent": "rag",
                    "requires_clarification": False,
                }

                with patch.object(
                    EnhancedOrchestrator, "_handle_rag", new_callable=AsyncMock
                ) as mock_rag:
                    mock_rag.return_value = {"response": "rag response"}

                    result = await orchestrator.process_command(
                        "co wiesz o sztucznej inteligencji?", "test_session"
                    )

                    assert result["response"] == "rag response"
                    mock_rag.assert_awaited_once()
                    # Verify Bielik model is used
                    assert (
                        mock_rag.call_args[0][2]
                        == "SpeakLeash/bielik-11b-v2.3-instruct:Q8_0"
                    )


@pytest.mark.asyncio
async def test_cooking_handling(orchestrator, db_session, mock_conversation):
    """Test cooking command routing"""
    with patch(
        "backend.core.crud.get_conversation_by_session_id",
        new_callable=AsyncMock,
        return_value=mock_conversation,
    ):
        with patch.object(
            ConversationMemoryManager,
            "retrieve_context",
            new_callable=AsyncMock,
            return_value=[],
        ):
            with patch.object(
                EnhancedOrchestrator, "_detect_intent", new_callable=AsyncMock
            ) as mock_detect:
                mock_detect.return_value = {
                    "intent": "cooking",
                    "requires_clarification": False,
                }

                with patch.object(
                    EnhancedOrchestrator, "_handle_cooking", new_callable=AsyncMock
                ) as mock_cooking:
                    mock_cooking.return_value = {"response": "cooking response"}

                    result = await orchestrator.process_command(
                        "jak przygotować spaghetti?", "test_session"
                    )

                    assert result["response"] == "cooking response"
                    mock_cooking.assert_awaited_once()
                    # Verify Bielik model is used
                    assert (
                        mock_cooking.call_args[0][2]
                        == "SpeakLeash/bielik-11b-v2.3-instruct:Q6_K"
                    )


@pytest.mark.asyncio
async def test_clarification_handling(orchestrator, db_session, mock_conversation):
    """Test clarification request handling"""
    with patch(
        "backend.core.crud.get_conversation_by_session_id",
        new_callable=AsyncMock,
        return_value=mock_conversation,
    ):
        with patch.object(
            ConversationMemoryManager,
            "retrieve_context",
            new_callable=AsyncMock,
            return_value=[],
        ):
            with patch.object(
                EnhancedOrchestrator, "_detect_intent", new_callable=AsyncMock
            ) as mock_detect:
                mock_detect.return_value = {
                    "intent": "weather",
                    "requires_clarification": True,
                    "ambiguous_options": ["Warszawa", "Kraków"],
                }

                result = await orchestrator.process_command("pogoda", "test_session")

                assert "doprecyzować" in result["response"]
                assert result["requires_clarification"]
                assert len(result["options"]) == 2


@pytest.mark.asyncio
async def test_memory_integration(orchestrator, db_session, mock_conversation):
    """Test memory integration in command processing"""
    test_context = [{"content": "previous conversation"}]

    with patch(
        "backend.core.crud.get_conversation_by_session_id",
        new_callable=AsyncMock,
        return_value=mock_conversation,
    ):
        with patch.object(
            ConversationMemoryManager,
            "retrieve_context",
            new_callable=AsyncMock,
            return_value=test_context,
        ):
            with patch.object(
                EnhancedOrchestrator, "_route_by_intent", new_callable=AsyncMock
            ) as mock_route:
                mock_route.return_value = {"response": "test response"}

                result = await orchestrator.process_command(
                    "test command", "test_session"
                )

                assert (
                    "Kontekst z wcześniejszych rozmów"
                    in mock_route.call_args[1]["memory_context"]
                )


@pytest.mark.asyncio
async def test_personalization(orchestrator, db_session, mock_conversation):
    """Test personalization in command processing"""
    with patch(
        "backend.core.crud.get_conversation_by_session_id",
        new_callable=AsyncMock,
        return_value=mock_conversation,
    ):
        with patch.object(
            ConversationMemoryManager,
            "retrieve_context",
            new_callable=AsyncMock,
            return_value=[],
        ):
            with patch.object(
                EnhancedOrchestrator, "_route_by_intent", new_callable=AsyncMock
            ) as mock_route:
                mock_route.return_value = {"response": "test response"}

                # Test with short command (should trigger suggestions)
                await orchestrator.process_command("cześć", "test_session")

                assert "Sugestie" in mock_route.call_args[1]["personalized_context"]


@pytest.mark.asyncio
async def test_shutdown(orchestrator):
    """Test orchestrator shutdown sequence"""
    with patch.object(orchestrator.weather_agent, "close", new_callable=AsyncMock):
        await orchestrator.shutdown()
        orchestrator.weather_agent.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_model_fallback(orchestrator, db_session, mock_conversation):
    """Test model fallback when primary model fails"""
    with patch(
        "backend.core.crud.get_conversation_by_session_id",
        new_callable=AsyncMock,
        return_value=mock_conversation,
    ):
        with patch.object(
            ConversationMemoryManager,
            "retrieve_context",
            new_callable=AsyncMock,
            return_value=[],
        ):
            with patch.object(
                EnhancedOrchestrator, "_route_by_intent", new_callable=AsyncMock
            ) as mock_route:
                # First call fails, second succeeds
                mock_route.side_effect = [
                    Exception("Model error"),
                    {"response": "fallback response"},
                ]

                result = await orchestrator.process_command(
                    "test command", "test_session"
                )

                assert result["response"] == "fallback response"
                assert mock_route.call_count == 2


@pytest.mark.asyncio
async def test_invalid_command(orchestrator, db_session):
    """Test handling of invalid/empty commands"""
    result = await orchestrator.process_command("", "test_session")
    assert "Nie rozpoznano" in result["response"]
    assert not result["success"]


@pytest.mark.asyncio
async def test_concurrent_commands(orchestrator, db_session, mock_conversation):
    """Test handling of concurrent commands"""

    async def run_command():
        with patch(
            "backend.core.crud.get_conversation_by_session_id",
            new_callable=AsyncMock,
            return_value=mock_conversation,
        ):
            with patch.object(
                ConversationMemoryManager,
                "retrieve_context",
                new_callable=AsyncMock,
                return_value=[],
            ):
                with patch.object(
                    EnhancedOrchestrator, "_route_by_intent", new_callable=AsyncMock
                ) as mock_route:
                    mock_route.return_value = {"response": "test response"}
                    return await orchestrator.process_command(
                        "test command", "test_session"
                    )

    # Run multiple commands concurrently
    results = await asyncio.gather(*[run_command() for _ in range(5)])
    assert all(r["response"] == "test response" for r in results)


@pytest.mark.asyncio
async def test_performance_metrics(orchestrator, db_session, mock_conversation):
    """Test performance metrics collection"""
    with patch(
        "backend.core.crud.get_conversation_by_session_id",
        new_callable=AsyncMock,
        return_value=mock_conversation,
    ):
        with patch.object(
            ConversationMemoryManager,
            "retrieve_context",
            new_callable=AsyncMock,
            return_value=[],
        ):
            with patch.object(
                EnhancedOrchestrator, "_route_by_intent", new_callable=AsyncMock
            ) as mock_route:
                mock_route.return_value = {"response": "test response"}

                await orchestrator.process_command("test command", "test_session")

                assert "processing_time" in mock_route.call_args[1]
                assert isinstance(mock_route.call_args[1]["processing_time"], float)
