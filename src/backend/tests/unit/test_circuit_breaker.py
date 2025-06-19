import time
from unittest.mock import AsyncMock

import pytest

from backend.agents.mixins.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenException,
    CircuitState,
    circuit_breaker,
)


class TestCircuitBreaker:
    @pytest.fixture
    def breaker(self):
        return CircuitBreaker(
            failure_threshold=2, recovery_timeout=0.1, half_open_threshold=1
        )

    @pytest.mark.asyncio
    async def test_closed_state_success(self, breaker):
        mock_func = AsyncMock(return_value="success")
        wrapped = breaker(mock_func)
        result = await wrapped()
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_closed_to_open_transition(self, breaker):
        mock_func = AsyncMock(side_effect=Exception("error"))
        wrapped = breaker(mock_func)

        # First failure
        with pytest.raises(Exception):
            await wrapped()
        assert breaker.state == CircuitState.CLOSED

        # Second failure triggers OPEN
        with pytest.raises(Exception):
            await wrapped()
        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_open_state_blocks_calls(self, breaker):
        breaker.state = CircuitState.OPEN
        breaker.last_failure_time = time.time()

        mock_func = AsyncMock()
        wrapped = breaker(mock_func)

        with pytest.raises(CircuitOpenException):
            await wrapped()
        mock_func.assert_not_called()

    @pytest.mark.asyncio
    async def test_open_to_half_open_transition(self, breaker):
        breaker.state = CircuitState.OPEN
        breaker.last_failure_time = time.time() - 0.2  # Past recovery timeout

        mock_func = AsyncMock(return_value="success")
        wrapped = breaker(mock_func)

        result = await wrapped()
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_to_closed(self, breaker):
        breaker.state = CircuitState.HALF_OPEN
        mock_func = AsyncMock(return_value="success")
        wrapped = breaker(mock_func)

        await wrapped()
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_to_open(self, breaker):
        breaker.state = CircuitState.HALF_OPEN
        mock_func = AsyncMock(side_effect=Exception("error"))
        wrapped = breaker(mock_func)

        with pytest.raises(Exception):
            await wrapped()
        assert breaker.state == CircuitState.OPEN


class TestCircuitBreakerDecorator:
    @pytest.mark.asyncio
    async def test_decorator_success(self):
        @circuit_breaker(failure_threshold=1)
        async def test_func():
            return "success"

        result = await test_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_decorator_failure(self):
        @circuit_breaker(failure_threshold=1)
        async def test_func():
            raise Exception("error")

        # First call fails and opens circuit
        with pytest.raises(Exception):
            await test_func()

        # Second call should be blocked
        with pytest.raises(CircuitOpenException):
            await test_func()
