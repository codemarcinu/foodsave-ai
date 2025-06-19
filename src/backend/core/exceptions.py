from typing import Any, Dict, Optional

from fastapi import HTTPException, status
from pydantic import BaseModel


class ErrorDetail(BaseModel):
    """Standardized error detail structure"""

    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class FoodSaveException(HTTPException):
    """Base exception for FoodSave application with standardized error format"""

    def __init__(
        self,
        status_code: int,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            status_code=status_code,
            detail=ErrorDetail(
                code=error_code, message=message, details=details
            ).dict(),
        )


# Common error codes
class ErrorCodes:
    DATABASE_ERROR = "database_error"
    VALIDATION_ERROR = "validation_error"
    AUTH_ERROR = "authentication_error"
    PERMISSION_ERROR = "permission_error"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    SERVICE_UNAVAILABLE = "service_unavailable"
    INTERNAL_ERROR = "internal_error"


# Specific exceptions
class DatabaseError(FoodSaveException):
    """Database related errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code=ErrorCodes.DATABASE_ERROR,
            message=message,
            details=details,
        )


class ValidationError(FoodSaveException):
    """Data validation errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code=ErrorCodes.VALIDATION_ERROR,
            message=message,
            details=details,
        )


class AuthenticationError(FoodSaveException):
    """Authentication related errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code=ErrorCodes.AUTH_ERROR,
            message=message,
            details=details,
        )


class PermissionError(FoodSaveException):
    """Permission related errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            error_code=ErrorCodes.PERMISSION_ERROR,
            message=message,
            details=details,
        )


class NotFoundError(FoodSaveException):
    """Resource not found errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code=ErrorCodes.NOT_FOUND,
            message=message,
            details=details,
        )


class ConflictError(FoodSaveException):
    """Resource conflict errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            error_code=ErrorCodes.CONFLICT,
            message=message,
            details=details,
        )


class ServiceUnavailableError(FoodSaveException):
    """Service unavailable errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code=ErrorCodes.SERVICE_UNAVAILABLE,
            message=message,
            details=details,
        )


class InternalServerError(FoodSaveException):
    """Internal server errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code=ErrorCodes.INTERNAL_ERROR,
            message=message,
            details=details,
        )
