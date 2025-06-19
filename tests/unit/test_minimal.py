import pytest

from src.backend.agents.orchestration_components import SimpleIntentDetector


@pytest.mark.asyncio
async def test_minimal_intent_detection():
    """Minimal test to verify test infrastructure works"""
    detector = SimpleIntentDetector()
    result = await detector.detect_intent("test query", None)
    assert result is not None
    assert hasattr(result, "type")
