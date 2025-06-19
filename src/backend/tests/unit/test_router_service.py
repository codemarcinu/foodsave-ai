from unittest.mock import MagicMock, patch

import pytest

from backend.agents.error_types import EnhancedAgentResponse
from backend.agents.router_service import IntentionHandler, RouterService
from backend.core.hybrid_llm_client import ModelComplexity


class TestRouterService:
    @pytest.fixture
    def router(self):
        return RouterService()

    @pytest.mark.asyncio
    async def test_detect_intent(self, router):
        with patch(
            "backend.core.hybrid_llm_client.hybrid_llm_client.chat"
        ) as mock_chat:
            mock_chat.return_value = {
                "message": {"content": '{"intent": "WEATHER", "confidence": 0.9}'}
            }
            intent = await router.detect_intent("Jaka jest pogoda?")
            assert intent["intent"] == "WEATHER"
            assert intent["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_route_command(self, router):
        mock_handler = MagicMock(spec=IntentionHandler)
        mock_handler.handle.return_value = EnhancedAgentResponse(
            success=True, text="Test response"
        )
        router.handlers = mock_handler

        response = await router.route_command(
            command="test",
            intent_data={"intent": "TEST"},
            session_id="123",
            complexity_level=ModelComplexity.STANDARD,
            agent_states={},
        )
        assert response.success
        assert response.text == "Test response"


class TestIntentionHandlers:
    @pytest.fixture
    def weather_handler(self):
        from backend.agents.enhanced_weather_agent import EnhancedWeatherAgent

        mock_agent = MagicMock(spec=EnhancedWeatherAgent)
        mock_agent.process.return_value = EnhancedAgentResponse(
            success=True, text="Weather response"
        )
        from backend.agents.router_service import WeatherHandler

        return WeatherHandler(mock_agent)

    @pytest.mark.asyncio
    async def test_weather_handler(self, weather_handler):
        response = await weather_handler.handle(
            command="Jaka jest pogoda?",
            intent_data={"intent": "WEATHER"},
            session_id="123",
            complexity_level=ModelComplexity.STANDARD,
            agent_states={"weather": True},
        )
        assert response.success
        assert response.text == "Weather response"

    @pytest.mark.asyncio
    async def test_weather_handler_not_handled(self, weather_handler):
        response = await weather_handler.handle(
            command="Test",
            intent_data={"intent": "OTHER"},
            session_id="123",
            complexity_level=ModelComplexity.STANDARD,
            agent_states={},
        )
        assert response is None
