import asyncio
import logging
import time
from collections import defaultdict
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class TokenBucket:
    """Implementation of token bucket algorithm for rate limiting"""

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.time()
        self.lock = asyncio.Lock()

    async def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from bucket"""
        async with self.lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        refill_amount = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + refill_amount)
        self.last_refill = now


class RateLimiter:
    """Rate limiter for agents with multi-level limits"""

    def __init__(self):
        self.global_limits: Dict[str, TokenBucket] = {}
        self.user_limits: Dict[str, Dict[str, TokenBucket]] = defaultdict(dict)
        self.lock = asyncio.Lock()

    async def set_global_limit(
        self, agent_type: str, capacity: int, refill_rate: float
    ):
        """Set global rate limit for agent type"""
        async with self.lock:
            self.global_limits[agent_type] = TokenBucket(capacity, refill_rate)

    async def set_user_limit(
        self, agent_type: str, user_id: str, capacity: int, refill_rate: float
    ):
        """Set user-specific rate limit for agent type"""
        async with self.lock:
            self.user_limits[agent_type][user_id] = TokenBucket(capacity, refill_rate)

    async def check_limit(
        self, agent_type: str, user_id: Optional[str] = None, tokens: int = 1
    ) -> bool:
        """
        Check if request is allowed
        Args:
            tokens: Number of tokens to consume (default=1)
        Returns:
            bool: True if allowed, False if rate limited
        """
        # Check user-specific limit first if provided
        if (
            user_id
            and agent_type in self.user_limits
            and user_id in self.user_limits[agent_type]
        ):
            if not await self.user_limits[agent_type][user_id].consume(tokens):
                return False

        # Check global limit
        if agent_type in self.global_limits:
            if not await self.global_limits[agent_type].consume(tokens):
                return False

        return True


def rate_limited(agent_type: str, user_id_key: Optional[str] = None):
    """Decorator for rate limiting agent methods"""

    def decorator(func: Callable) -> Callable:
        async def wrapper(self, *args, **kwargs) -> Any:
            # Extract user_id from kwargs or args
            user_id = None
            if user_id_key:
                if user_id_key in kwargs:
                    user_id = kwargs[user_id_key]
                else:
                    # Try to get from args by parameter name
                    try:
                        # Get function signature
                        import inspect

                        sig = inspect.signature(func)
                        params = list(sig.parameters.keys())
                        if user_id_key in params:
                            idx = params.index(user_id_key) - 1  # subtract 1 for 'self'
                            if idx < len(args):
                                user_id = args[idx]
                    except Exception as e:
                        logger.warning(f"Error extracting user_id: {e}")

            if not await self.rate_limiter.check_limit(agent_type, user_id):
                raise RateLimitExceeded(f"Rate limit exceeded for {agent_type}")
            return await func(self, *args, **kwargs)

        return wrapper

    return decorator


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded"""

    pass
