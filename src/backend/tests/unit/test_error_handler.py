from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from backend.agents.error_types import ErrorSeverity
from backend.agents.mixins.error_handler import ErrorHandler


class TestErrorHandler:
    @pytest.fixture
    def handler(self):
        return ErrorHandler("test_agent")

    @pytest.mark.asyncio
    async def test_execute_with_fallback_success(self, handler):
        mock_func = AsyncMock(return_value="success")
        result, used_fallback, error, time = await handler.execute_with_fallback(
            mock_func
        )
        assert result == "success"
        assert not used_fallback
        assert error is None
        assert time > 0

    @pytest.mark.asyncio
    async def test_execute_with_fallback_error(self, handler):
        mock_func = AsyncMock(side_effect=Exception("test error"))
        result, used_fallback, error, time = await handler.execute_with_fallback(
            mock_func
        )
        assert result is None
        assert used_fallback
        assert error == "test error"
        assert time > 0

    @pytest.mark.asyncio
    async def test_execute_with_fallback_fallback_success(self, handler):
        mock_func = AsyncMock(side_effect=Exception("test error"))
        mock_fallback = AsyncMock(return_value="fallback success")
        result, used_fallback, error, time = await handler.execute_with_fallback(
            mock_func, fallback_handler=mock_fallback
        )
        assert result == "fallback success"
        assert used_fallback
        assert error == "test error"
        assert time > 0

    def test_should_alert_below_threshold(self, handler):
        assert not handler._should_alert("test", ErrorSeverity.LOW)

    def test_should_alert_above_threshold(self, handler):
        assert handler._should_alert("test", ErrorSeverity.HIGH)

    def test_should_alert_throttling(self, handler):
        error_key = "test_agent:test"
        handler.last_alerts[error_key] = datetime.now() - timedelta(seconds=100)
        assert handler._should_alert("test", ErrorSeverity.HIGH)

        handler.last_alerts[error_key] = datetime.now()
        assert not handler._should_alert("test", ErrorSeverity.HIGH)

    @pytest.mark.asyncio
    async def test_send_alert(self, handler):
        with patch("logging.warning") as mock_warning:
            await handler._send_alert("test", {}, ErrorSeverity.HIGH)
            mock_warning.assert_called_once()
