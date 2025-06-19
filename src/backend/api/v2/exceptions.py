from typing import Any, Dict, Optional

from fastapi import status
from pydantic import BaseModel


class APIErrorDetail(BaseModel):
    """Standardized API error response format for v2"""

    status_code: int
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class APIException(Exception):
    """Base exception for API v2 with standardized error format"""

    def __init__(
        self,
        status_code: int,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.details = details
        super().__init__(message)


# Common error codes for API v2
class APIErrorCodes:
    # Client errors (4xx)
    INVALID_INPUT = "invalid_input"
    MISSING_FIELD = "missing_field"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    NOT_FOUND = "not_found"
    METHOD_NOT_ALLOWED = "method_not_allowed"
    CONFLICT = "conflict"
    UNPROCESSABLE_ENTITY = "unprocessable_entity"

    # Server errors (5xx)
    INTERNAL_ERROR = "internal_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    EXTERNAL_SERVICE_ERROR = "external_service_error"

    # Domain-specific errors
    RECEIPT_PROCESSING_ERROR = "receipt_processing_error"
    FILE_TYPE_UNSUPPORTED = "file_type_unsupported"


# Specific API exceptions
class BadRequestError(APIException):
    """400 Bad Request errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code=APIErrorCodes.INVALID_INPUT,
            message=message,
            details=details,
        )


class UnauthorizedError(APIException):
    """401 Unauthorized errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code=APIErrorCodes.UNAUTHORIZED,
            message=message,
            details=details,
        )


class ForbiddenError(APIException):
    """403 Forbidden errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            error_code=APIErrorCodes.FORBIDDEN,
            message=message,
            details=details,
        )


class NotFoundError(APIException):
    """404 Not Found errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code=APIErrorCodes.NOT_FOUND,
            message=message,
            details=details,
        )


class MethodNotAllowedError(APIException):
    """405 Method Not Allowed errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
            error_code=APIErrorCodes.METHOD_NOT_ALLOWED,
            message=message,
            details=details,
        )


class ConflictError(APIException):
    """409 Conflict errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            error_code=APIErrorCodes.CONFLICT,
            message=message,
            details=details,
        )


class UnprocessableEntityError(APIException):
    """422 Unprocessable Entity errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code=APIErrorCodes.UNPROCESSABLE_ENTITY,
            message=message,
            details=details,
        )


class InternalServerError(APIException):
    """500 Internal Server Error"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code=APIErrorCodes.INTERNAL_ERROR,
            message=message,
            details=details,
        )


class ServiceUnavailableError(APIException):
    """503 Service Unavailable"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code=APIErrorCodes.SERVICE_UNAVAILABLE,
            message=message,
            details=details,
        )


class ExternalServiceError(APIException):
    """Errors from external services"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_502_BAD_GATEWAY,
            error_code=APIErrorCodes.EXTERNAL_SERVICE_ERROR,
            message=message,
            details=details,
        )
