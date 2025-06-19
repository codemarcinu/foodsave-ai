import functools
import logging
import time
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


class CircuitBreaker:
    """Implementation of Circuit Breaker pattern for agent operations"""

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        half_open_threshold: int = 1,
    ):
        self.state = CircuitState.CLOSED
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_threshold = half_open_threshold
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_attempts = 0

    def __call__(self, func: Callable[..., Coroutine[Any, Any, Any]]):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if self.state == CircuitState.OPEN:
                current_time = time.time()
                if (
                    current_time - (self.last_failure_time or 0)
                ) > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_attempts = 0
                else:
                    raise CircuitOpenException(
                        f"Circuit is OPEN. Retry after {self.recovery_timeout - (current_time - (self.last_failure_time or 0)):.1f}s"
                    )

            try:
                result = await func(*args, **kwargs)
                if self.state == CircuitState.HALF_OPEN:
                    self.half_open_attempts += 1
                    if self.half_open_attempts >= self.half_open_threshold:
                        self._reset()
                return result
            except Exception:
                await self._record_failure()
                raise

        return wrapper

    async def _record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()

        if (
            self.state == CircuitState.CLOSED
            and self.failure_count >= self.failure_threshold
        ):
            self.state = CircuitState.OPEN
            logger.error(f"Circuit breaker OPEN after {self.failure_count} failures")
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.error("Circuit breaker re-OPENED during half-open state")

    def _reset(self):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_attempts = 0
        logger.info("Circuit breaker RESET to CLOSED state")


class CircuitOpenException(Exception):
    """Exception raised when circuit breaker is open"""

    pass


def circuit_breaker(
    failure_threshold: int = 3,
    recovery_timeout: float = 30.0,
    half_open_threshold: int = 1,
):
    """Decorator factory for circuit breaker pattern"""

    def decorator(func: Callable[..., Coroutine[Any, Any, Any]]):
        breaker = CircuitBreaker(
            failure_threshold, recovery_timeout, half_open_threshold
        )
        return breaker(func)

    return decorator
