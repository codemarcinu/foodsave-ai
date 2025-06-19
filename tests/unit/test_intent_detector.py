import pytest

from src.backend.agents.orchestration_components import (
    MemoryContext,
    SimpleIntentDetector,
)


@pytest.mark.asyncio
async def test_detect_recipe_intent():
    detector = SimpleIntentDetector()
    context = MemoryContext(session_id="test")
    intent = await detector.detect_intent("please give me a cake recipe", context)
    assert intent.type == "recipe_request"


@pytest.mark.asyncio
async def test_detect_add_to_list_intent():
    detector = SimpleIntentDetector()
    context = MemoryContext(session_id="test")
    intent = await detector.detect_intent("add eggs to shopping", context)
    assert intent.type == "add_to_list"


@pytest.mark.asyncio
async def test_detect_unknown_intent():
    detector = SimpleIntentDetector()
    context = MemoryContext(session_id="test")
    intent = await detector.detect_intent(
        "random text that doesn't match any intent", context
    )
    assert intent.type == "unknown"
