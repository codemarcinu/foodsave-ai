from unittest.mock import AsyncMock, patch

import pytest

from src.backend.agents.tools.tools import extract_entities, recognize_intent


@pytest.mark.asyncio
@patch("backend.agents.tools.tools.llm_client", new_callable=AsyncMock)
async def test_recognize_intent_success(mock_llm_client):
    """
    Tests if recognize_intent successfully calls the LLM and returns the intent.
    """
    # Arrange
    mock_llm_client.chat.return_value = {
        "message": {"content": '{"intent": "DODAJ_ZAKUPY"}'}
    }
    prompt = "Dodaj paragon z Biedronki za 100 zł"

    # Act
    result = await recognize_intent(prompt)

    # Assert
    mock_llm_client.chat.assert_called_once()
    assert '"intent": "DODAJ_ZAKUPY"' in result


@pytest.mark.asyncio
@patch("backend.agents.tools.tools.llm_client", new_callable=AsyncMock)
async def test_recognize_intent_failure(mock_llm_client):
    """
    Tests if recognize_intent handles LLM client failure gracefully.
    """
    # Arrange
    mock_llm_client.chat.side_effect = Exception("LLM is down")
    prompt = "Dodaj paragon"

    # Act
    result = await recognize_intent(prompt)

    # Assert
    assert result == '{"intent": "UNKNOWN"}'


@pytest.mark.asyncio
@patch("backend.agents.tools.tools.llm_client", new_callable=AsyncMock)
async def test_extract_entities_success(mock_llm_client):
    """
    Tests if extract_entities successfully calls the LLM and returns entities.
    """
    # Arrange
    mock_llm_client.chat.return_value = {
        "message": {"content": '{"sklep": "Biedronka", "kwota": 100}'}
    }
    prompt = "Paragon z Biedronki za 100 zł"

    # Act
    result = await extract_entities(prompt)

    # Assert
    mock_llm_client.chat.assert_called_once()
    assert '"sklep": "Biedronka"' in result
    assert '"kwota": 100' in result


@pytest.mark.asyncio
@patch("backend.agents.tools.tools.llm_client", new_callable=AsyncMock)
async def test_extract_entities_failure(mock_llm_client):
    """
    Tests if extract_entities handles LLM client failure gracefully.
    """
    # Arrange
    mock_llm_client.chat.side_effect = Exception("LLM is down")
    prompt = "Paragon z Biedronki"

    # Act
    result = await extract_entities(prompt)

    # Assert
    assert result == "{}"
