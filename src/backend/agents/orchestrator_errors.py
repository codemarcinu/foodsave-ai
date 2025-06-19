from typing import Any, Dict, Optional

from fastapi import status

from ..core.exceptions import ErrorCodes, FoodSaveException


class OrchestratorError(FoodSaveException):
    """Base class for orchestrator errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code=ErrorCodes.INTERNAL_ERROR,
            message=message,
            details=details,
        )


class AgentProcessingError(OrchestratorError):
    """Error during agent processing."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=f"Agent processing error: {message}", details=details)


class ServiceUnavailableError(OrchestratorError):
    """Required service is unavailable."""

    def __init__(self, service_name: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code=ErrorCodes.SERVICE_UNAVAILABLE,
            message=f"Service {service_name} is unavailable",
            details=details,
        )


class IntentRecognitionError(OrchestratorError):
    """Error during intent recognition."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Intent recognition error: {message}", details=details
        )


class MemoryManagerError(OrchestratorError):
    """Error related to memory management."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=f"Memory manager error: {message}", details=details)


class ProfileManagerError(OrchestratorError):
    """Error related to user profile management."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=f"Profile manager error: {message}", details=details)
