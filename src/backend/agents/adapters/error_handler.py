import logging
import time
import traceback
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Optional

from ..core.agent_interface import IErrorHandler
from ..error_types import ErrorSeverity


class ErrorHandler(IErrorHandler):
    """Handler for error processing and recovery with alerting capabilities"""

    def __init__(self, name: str, alert_config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.alert_config = alert_config or {
            "enabled": True,
            "min_severity": ErrorSeverity.HIGH,
            "throttle_period": 3600,
        }
        self.last_alerts: Dict[str, datetime] = {}

    async def execute_with_fallback(
        self,
        func: Callable[..., Awaitable[Any]],
        *args,
        fallback_handler: Optional[Callable[..., Any]] = None,
        error_severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        **kwargs,
    ) -> Any:
        """
        Execute a function with automatic fallback and error handling
        Returns the result of the operation or fallback.
        Logs additional debug information internally.
        """
        start_time = time.time()

        try:
            result = await func(*args, **kwargs)
            return result

        except Exception as e:
            error_info = {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat(),
            }
            logging.error(f"Error in {self.name}.{func.__name__}: {str(e)}")

            if self._should_alert(str(e), error_severity):
                await self._send_alert(
                    f"Error in {self.name}.{func.__name__}", error_info, error_severity
                )

            if fallback_handler:
                try:
                    logging.info(f"Attempting fallback for {func.__name__}")
                    result = await fallback_handler(*args, error=e, **kwargs)
                    return result
                except Exception as fallback_error:
                    logging.error(f"Fallback also failed: {str(fallback_error)}")
                    error_info["fallback_error"] = str(fallback_error)

            return None

    def _should_alert(self, error_message: str, severity: ErrorSeverity) -> bool:
        """Determine if an alert should be sent based on severity and throttling"""
        if not self.alert_config["enabled"]:
            return False

        severity_levels = {
            ErrorSeverity.LOW: 1,
            ErrorSeverity.MEDIUM: 2,
            ErrorSeverity.HIGH: 3,
            ErrorSeverity.CRITICAL: 4,
        }

        if (
            severity_levels[severity]
            < severity_levels[self.alert_config["min_severity"]]
        ):
            return False

        error_key = f"{self.name}:{error_message[:50]}"
        now = datetime.now()

        if error_key in self.last_alerts:
            time_since_last = (now - self.last_alerts[error_key]).total_seconds()
            if time_since_last < self.alert_config["throttle_period"]:
                return False

        self.last_alerts[error_key] = now
        return True

    def should_alert(self, error_message: str, severity: ErrorSeverity) -> bool:
        """Public interface for checking if an alert should be sent"""
        return self._should_alert(error_message, severity)

    async def _send_alert(
        self, subject: str, error_info: Dict[str, Any], severity: ErrorSeverity
    ) -> None:
        """Send alert notification via configured channels"""
        if not self.alert_config["enabled"]:
            return

        logging.warning(f"AGENT ALERT: {subject} ({severity})")
        # In production would send email/Slack alerts here
