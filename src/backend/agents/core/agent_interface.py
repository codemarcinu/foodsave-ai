from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, Optional

from ..error_types import EnhancedAgentResponse, ErrorSeverity


class IErrorHandler(ABC):
    """Interface for error handling functionality"""

    @abstractmethod
    async def execute_with_fallback(
        self,
        func: Callable[..., Awaitable[Any]],
        *args,
        fallback_handler: Optional[Callable[..., Any]] = None,
        error_severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        **kwargs
    ) -> Any:
        """Execute a function with automatic fallback and error handling"""
        pass


class IFallbackProvider(ABC):
    """Interface for fallback functionality"""

    @abstractmethod
    async def execute_fallback(
        self, input_data: Dict[str, Any], original_error: Exception
    ) -> EnhancedAgentResponse:
        """Execute fallback logic for failed operations"""
        pass


class IAlertService(ABC):
    """Interface for alerting functionality"""

    @abstractmethod
    def should_alert(self, error_message: str, severity: ErrorSeverity) -> bool:
        """Determine if an alert should be sent for this error"""
        pass

    @abstractmethod
    async def send_alert(
        self, title: str, error_info: Dict[str, Any], severity: ErrorSeverity
    ) -> None:
        """Send an alert notification"""
        pass
