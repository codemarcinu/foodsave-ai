import asyncio

import pytest

from backend.agents.mixins.rate_limiter import (
    RateLimiter,
    RateLimitExceeded,
    TokenBucket,
    rate_limited,
)


class TestTokenBucket:
    @pytest.mark.asyncio
    async def test_consume_tokens(self):
        bucket = TokenBucket(capacity=10, refill_rate=1)
        assert await bucket.consume(5) is True
        assert bucket.tokens == 5

    @pytest.mark.asyncio
    async def test_refill_tokens(self):
        bucket = TokenBucket(capacity=10, refill_rate=1)
        await bucket.consume(10)
        assert bucket.tokens == 0

        # Wait for refill
        await asyncio.sleep(1.1)
        assert await bucket.consume(1) is True

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self):
        bucket = TokenBucket(capacity=5, refill_rate=0.1)
        for _ in range(5):
            assert await bucket.consume() is True
        assert await bucket.consume() is False


class TestRateLimiter:
    @pytest.fixture
    def limiter(self):
        return RateLimiter()

    @pytest.mark.asyncio
    async def test_global_limit(self, limiter):
        await limiter.set_global_limit("test_agent", 2, 0.1)
        assert await limiter.check_limit("test_agent") is True
        assert await limiter.check_limit("test_agent") is True
        assert await limiter.check_limit("test_agent") is False

    @pytest.mark.asyncio
    async def test_user_limit(self, limiter):
        await limiter.set_user_limit("test_agent", "user1", 1, 0.1)
        assert await limiter.check_limit("test_agent", "user1") is True
        assert await limiter.check_limit("test_agent", "user1") is False
        assert await limiter.check_limit("test_agent", "user2") is True

    @pytest.mark.asyncio
    async def test_combined_limits(self, limiter):
        await limiter.set_global_limit("test_agent", 2, 0.1)
        await limiter.set_user_limit("test_agent", "user1", 1, 0.1)

        # User1 hits both limits
        assert await limiter.check_limit("test_agent", "user1") is True
        assert await limiter.check_limit("test_agent", "user1") is False

        # User2 only hits global limit
        assert await limiter.check_limit("test_agent", "user2") is True
        assert await limiter.check_limit("test_agent", "user2") is False


class TestRateLimitedDecorator:
    @pytest.mark.asyncio
    async def test_rate_limited_method(self):
        class TestAgent:
            def __init__(self):
                self.rate_limiter = RateLimiter()

            @rate_limited("test_agent", "user_id")
            async def test_method(self, user_id):
                return "success"

        agent = TestAgent()
        await agent.rate_limiter.set_user_limit("test_agent", "user1", 1, 0.1)

        # First call should succeed
        assert await agent.test_method("user1") == "success"

        # Second call should fail
        with pytest.raises(RateLimitExceeded):
            await agent.test_method("user1")
