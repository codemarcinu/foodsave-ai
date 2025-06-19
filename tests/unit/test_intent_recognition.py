import json
import os
from typing import Any, Dict

import pytest

from src.backend.agents.tools.tools import recognize_intent

# Ladowanie danych testowych bezposrednio z pliku JSON
TEST_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "test_data.json"
)
with open(TEST_DATA_PATH, "r") as f:
    TEST_DATA = json.load(f)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_prompt, expected_intent",
    [(item["prompt"], item["intent"]) for item in TEST_DATA],
)
async def test_intent_recognition_with_mock(
    user_prompt: str, expected_intent: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Testuje funkcję recognize_intent z użyciem atrapy (mock) dla klienta LLM.
    """
    # Definiujemy atrapę odpowiedzi, którą ma zwrócić llm_client.chat
    mock_response = {"message": {"content": json.dumps({"intent": expected_intent})}}

    # Definiujemy asynchroniczną funkcję-atrapę
    async def mock_chat(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return mock_response

    # Używamy monkeypatch do zastąpienia prawdziwej metody `chat` naszą atrapą
    monkeypatch.setattr("backend.core.llm_client.llm_client.chat", mock_chat)

    # Wywołujemy testowaną funkcję
    raw_json_response = await recognize_intent(user_prompt)

    # Parsujemy odpowiedź i sprawdzamy asercję
    parsed_response = json.loads(raw_json_response)
    recognized_intent = parsed_response.get("intent")

    assert (
        recognized_intent == expected_intent
    ), f"Oczekiwano intencji '{expected_intent}', a otrzymano '{recognized_intent}'"
