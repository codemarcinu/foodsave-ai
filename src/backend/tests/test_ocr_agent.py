from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from backend.agents.enhanced_base_agent import EnhancedAgentResponse
from backend.agents.ocr_agent import OCRAgent, OCRAgentInput


@pytest.mark.asyncio
async def test_ocr_agent_success_image():
    agent = OCRAgent()
    mock_input = {"file_bytes": b"test_image_bytes", "file_type": "image"}

    with patch("backend.agents.ocr_agent.process_image_file") as mock_process:
        mock_process.return_value = "Rozpoznany tekst z obrazu"

        response = await agent.process(mock_input)
        assert isinstance(response, EnhancedAgentResponse)
        assert response.success is True
        assert "Rozpoznany tekst z obrazu" in response.text


@pytest.mark.asyncio
async def test_ocr_agent_success_pdf():
    agent = OCRAgent()
    mock_input = {"file_bytes": b"test_pdf_bytes", "file_type": "pdf"}

    with patch("backend.agents.ocr_agent.process_pdf_file") as mock_process:
        mock_process.return_value = "Rozpoznany tekst z PDF"

        response = await agent.process(mock_input)
        assert isinstance(response, EnhancedAgentResponse)
        assert response.success is True
        assert "Rozpoznany tekst z PDF" in response.text


@pytest.mark.asyncio
async def test_ocr_agent_input_validation():
    agent = OCRAgent()

    # Test missing required field
    with pytest.raises(ValidationError):
        await agent.process({"file_type": "image"})

    # Test invalid file type
    with pytest.raises(ValidationError):
        await agent.process({"file_bytes": b"test", "file_type": "invalid"})


@pytest.mark.asyncio
async def test_ocr_agent_unsupported_file_type():
    agent = OCRAgent()
    mock_input = {"file_bytes": b"test", "file_type": "docx"}

    response = await agent.process(mock_input)
    assert not response.success
    assert "Nieobsługiwany typ pliku" in response.error


@pytest.mark.asyncio
async def test_ocr_agent_empty_result():
    agent = OCRAgent()
    mock_input = {"file_bytes": b"test", "file_type": "image"}

    with patch("backend.agents.ocr_agent.process_image_file") as mock_process:
        mock_process.return_value = ""

        response = await agent.process(mock_input)
        assert not response.success
        assert "Nie udało się rozpoznać tekstu" in response.error


@pytest.mark.asyncio
async def test_ocr_agent_processing_error():
    agent = OCRAgent()
    mock_input = {"file_bytes": b"test", "file_type": "image"}

    with patch("backend.agents.ocr_agent.process_image_file") as mock_process:
        mock_process.side_effect = Exception("OCR processing error")

        response = await agent.process(mock_input)
        assert not response.success
        assert "Wystąpił błąd podczas przetwarzania pliku" in response.error
