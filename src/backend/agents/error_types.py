from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

from pydantic import BaseModel


class ErrorSeverity(str, Enum):
    """Severity levels for agent errors"""

    LOW = "low"  # Non-critical errors, can be handled silently
    MEDIUM = "medium"  # User-visible errors that don't require developer attention
    HIGH = "high"  # Serious errors that should be logged for review
    CRITICAL = "critical"  # Urgent errors requiring immediate attention


class EnhancedAgentResponse(BaseModel):
    """Enhanced agent response with additional context and error handling"""

    success: bool
    data: Optional[Dict[str, Any]] = None
    text: Optional[str] = None
    text_stream: Optional[AsyncGenerator[str, None]] = None
    message: Optional[str] = None
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    error_severity: Optional[ErrorSeverity] = None
    processed_with_fallback: bool = False
    processing_time: float = 0.0
    metadata: Dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True


class AlertConfig(BaseModel):
    """Configuration for alert notifications"""

    enabled: bool = True
    email_alerts: bool = False
    email_recipients: List[str] = []
    slack_alerts: bool = False
    slack_webhook: Optional[str] = None
    min_severity: ErrorSeverity = ErrorSeverity.HIGH
    throttle_period: int = 3600  # seconds between similar alerts


class AgentError(Exception):
    """Base class for agent errors"""

    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        self.message = message
        self.severity = severity
        super().__init__(message)


class OrchestratorError(AgentError):
    """Error specific to orchestrator operations"""

    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.HIGH):
        super().__init__(message, severity)
