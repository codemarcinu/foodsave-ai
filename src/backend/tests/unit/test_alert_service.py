from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from backend.agents.adapters.alert_service import AlertService
from backend.agents.error_types import ErrorSeverity


class TestAlertService:
    @pytest.fixture
    def service(self):
        return AlertService("test_agent")

    def test_should_alert_below_threshold(self, service):
        assert not service.should_alert("test", ErrorSeverity.LOW)

    def test_should_alert_above_threshold(self, service):
        assert service.should_alert("test", ErrorSeverity.HIGH)

    def test_should_alert_throttling(self, service):
        error_key = "test_agent:test"
        service.last_alerts[error_key] = datetime.now() - timedelta(seconds=3600)
        assert service.should_alert("test", ErrorSeverity.HIGH)

        service.last_alerts[error_key] = datetime.now()
        assert not service.should_alert("test", ErrorSeverity.HIGH)

    @pytest.mark.asyncio
    async def test_send_alert(self, service):
        with patch.object(service, "should_alert", return_value=True), patch(
            "logging.warning"
        ) as mock_warning:
            await service.send_alert("test", {"error": "test"}, ErrorSeverity.HIGH)
            mock_warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_alert_not_triggered(self, service):
        with patch.object(service, "should_alert", return_value=False), patch(
            "logging.warning"
        ) as mock_warning:
            await service.send_alert("test", {"error": "test"}, ErrorSeverity.HIGH)
            mock_warning.assert_not_called()
