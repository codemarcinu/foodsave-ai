import asyncio
import os

# Dodanie ścieżki do sys.path dla importów
import sys
from typing import Any, AsyncGenerator, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "src"))

from src.backend.agents.error_types import AgentError
from src.backend.agents.ocr_agent import OCRAgent
from src.backend.core.ocr import OCRProcessor


class TestOCRAgent:
    """Testy dla OCR Agent - agenta przetwarzającego obrazy i dokumenty"""

    @pytest.fixture
    def ocr_agent(self):
        """Fixture dla OCR Agent"""
        return OCRAgent()

    @pytest.fixture
    def mock_ocr_processor(self):
        """Mock procesora OCR"""
        with patch("src.backend.agents.ocr_agent.OCRProcessor") as mock_processor:
            mock_processor.return_value.process_image.return_value = {
                "text": "Extracted text",
                "items": ["Milk", "Bread"],
                "total": 15.99,
            }
            yield mock_processor

    @pytest.fixture
    def mock_llm_client(self):
        """Mock klienta LLM"""
        with patch("src.backend.agents.ocr_agent.llm_client") as mock_client:
            mock_client.chat.return_value = {
                "message": {"content": "Structured receipt data"}
            }
            yield mock_client

    @pytest.mark.asyncio
    async def test_process_image_success(self, ocr_agent, mock_ocr_processor):
        """Test pomyślnego przetwarzania obrazu"""
        # Given
        input_data = {
            "image_path": "/path/to/receipt.jpg",
            "image_data": b"fake_image_data",
        }

        # When
        response = await ocr_agent.process(input_data)

        # Then
        assert response.success is True
        assert "text" in response.data
        assert "items" in response.data
        assert response.data["total"] == 15.99
        mock_ocr_processor.return_value.process_image.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_image_missing_path(self, ocr_agent):
        """Test przetwarzania bez podania ścieżki obrazu"""
        # Given
        input_data = {"image_data": b"fake_image_data"}

        # When
        response = await ocr_agent.process(input_data)

        # Then
        assert response.success is False
        assert "Missing image_path or image_data" in response.error
        assert response.error_type == AgentError.INVALID_INPUT

    @pytest.mark.asyncio
    async def test_process_image_ocr_error(self, ocr_agent, mock_ocr_processor):
        """Test błędu OCR podczas przetwarzania"""
        # Given
        input_data = {"image_path": "/path/to/receipt.jpg"}
        mock_ocr_processor.return_value.process_image.side_effect = Exception(
            "OCR error"
        )

        # When
        response = await ocr_agent.process(input_data)

        # Then
        assert response.success is False
        assert "OCR error" in response.error
        assert response.error_type == AgentError.PROCESSING_ERROR

    @pytest.mark.asyncio
    async def test_process_image_with_llm_enhancement(
        self, ocr_agent, mock_ocr_processor, mock_llm_client
    ):
        """Test wzbogacania danych OCR przez LLM"""
        # Given
        input_data = {"image_path": "/path/to/receipt.jpg", "enhance_with_llm": True}

        # When
        response = await ocr_agent.process(input_data)

        # Then
        assert response.success is True
        assert "llm_enhanced" in response.data
        assert response.data["llm_enhanced"] == "Structured receipt data"
        mock_llm_client.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_pdf_success(self, ocr_agent, mock_ocr_processor):
        """Test pomyślnego przetwarzania pliku PDF"""
        # Given
        input_data = {"file_path": "/path/to/document.pdf"}

        # When
        response = await ocr_agent.process(input_data)

        # Then
        assert response.success is True
        assert "text" in response.data
        mock_ocr_processor.return_value.process_pdf.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_unsupported_type(self, ocr_agent):
        """Test przetwarzania nieobsługiwanego typu pliku"""
        # Given
        input_data = {"file_path": "/path/to/file.xyz"}

        # When
        response = await ocr_agent.process(input_data)

        # Then
        assert response.success is False
        assert "Unsupported file type" in response.error
        assert response.error_type == AgentError.UNSUPPORTED_FILE_TYPE

    @pytest.mark.asyncio
    async def test_process_large_image(self, ocr_agent, mock_ocr_processor):
        """Test przetwarzania dużego obrazu"""
        # Given
        # Symulacja dużego obrazu (5MB)
        large_image = b"0" * (5 * 1024 * 1024)
        input_data = {"image_data": large_image}

        # When
        response = await ocr_agent.process(input_data)

        # Then
        assert response.success is True
        # Sprawdzenie czy obraz został zmniejszony
        processed_data = mock_ocr_processor.return_value.process_image.call_args[0][0]
        assert len(processed_data) < len(large_image)

    @pytest.mark.asyncio
    async def test_process_with_multiple_pages(self, ocr_agent, mock_ocr_processor):
        """Test przetwarzania dokumentu wielostronicowego"""
        # Given
        input_data = {"file_path": "/path/to/multi_page.pdf"}
        mock_ocr_processor.return_value.process_pdf.return_value = {
            "pages": [{"text": "Page 1 content"}, {"text": "Page 2 content"}]
        }

        # When
        response = await ocr_agent.process(input_data)

        # Then
        assert response.success is True
        assert "pages" in response.data
        assert len(response.data["pages"]) == 2

    @pytest.mark.asyncio
    async def test_process_with_language_hint(self, ocr_agent, mock_ocr_processor):
        """Test przetwarzania z podpowiedzią języka"""
        # Given
        input_data = {"image_path": "/path/to/receipt.jpg", "language": "pol"}

        # When
        await ocr_agent.process(input_data)

        # Then
        call_args = mock_ocr_processor.return_value.process_image.call_args
        assert call_args[1]["language"] == "pol"

    @pytest.mark.asyncio
    async def test_process_with_rotation_correction(
        self, ocr_agent, mock_ocr_processor
    ):
        """Test automatycznej korekty rotacji obrazu"""
        # Given
        input_data = {"image_path": "/path/to/rotated.jpg"}

        # Konfiguracja mocka do zwrócenia informacji o rotacji
        mock_ocr_processor.return_value.detect_rotation.return_value = 90
        mock_ocr_processor.return_value.correct_rotation.return_value = (
            b"corrected_image"
        )

        # When
        await ocr_agent.process(input_data)

        # Then
        mock_ocr_processor.return_value.detect_rotation.assert_called()
        mock_ocr_processor.return_value.correct_rotation.assert_called()

    @pytest.mark.asyncio
    async def test_process_with_table_extraction(self, ocr_agent, mock_ocr_processor):
        """Test ekstrakcji tabel z dokumentu"""
        # Given
        input_data = {
            "file_path": "/path/to/document_with_table.pdf",
            "extract_tables": True,
        }
        mock_ocr_processor.return_value.extract_tables.return_value = [
            {"rows": 3, "columns": 2}
        ]

        # When
        response = await ocr_agent.process(input_data)

        # Then
        assert response.success is True
        assert "tables" in response.data
        assert len(response.data["tables"]) > 0

    @pytest.mark.asyncio
    async def test_process_with_specific_region(self, ocr_agent, mock_ocr_processor):
        """Test przetwarzania określonego regionu obrazu"""
        # Given
        input_data = {
            "image_path": "/path/to/receipt.jpg",
            "region": {"x": 100, "y": 200, "width": 300, "height": 400},
        }

        # When
        await ocr_agent.process(input_data)

        # Then
        call_args = mock_ocr_processor.return_value.process_image.call_args
        assert call_args[1]["region"] == (100, 200, 300, 400)

    @pytest.mark.asyncio
    async def test_process_with_fallback_engine(self, ocr_agent, mock_ocr_processor):
        """Test użycia zapasowego silnika OCR"""
        # Given
        input_data = {"image_path": "/path/to/receipt.jpg"}
        mock_ocr_processor.return_value.process_image.side_effect = [
            Exception("Primary engine failed"),
            {"text": "Fallback text"},
        ]

        # When
        response = await ocr_agent.process(input_data)

        # Then
        assert response.success is True
        assert response.data["text"] == "Fallback text"
        assert mock_ocr_processor.return_value.process_image.call_count == 2

    @pytest.mark.asyncio
    async def test_process_with_quality_metrics(self, ocr_agent, mock_ocr_processor):
        """Test zbierania metryk jakości OCR"""
        # Given
        input_data = {"image_path": "/path/to/receipt.jpg"}
        mock_ocr_processor.return_value.process_image.return_value[
            "quality_metrics"
        ] = {"confidence": 0.92, "resolution": 300}

        # When
        response = await ocr_agent.process(input_data)

        # Then
        assert "quality_metrics" in response.data
        assert response.data["quality_metrics"]["confidence"] == 0.92

    @pytest.mark.asyncio
    async def test_process_with_image_enhancement(self, ocr_agent, mock_ocr_processor):
        """Test poprawy jakości obrazu przed OCR"""
        # Given
        input_data = {"image_path": "/path/to/low_quality.jpg"}

        # Konfiguracja mocka dla operacji poprawy jakości
        mock_ocr_processor.return_value.enhance_image.return_value = b"enhanced_image"

        # When
        await ocr_agent.process(input_data)

        # Then
        mock_ocr_processor.return_value.enhance_image.assert_called()

    @pytest.mark.asyncio
    async def test_process_with_barcode_detection(self, ocr_agent, mock_ocr_processor):
        """Test wykrywania kodów kreskowych"""
        # Given
        input_data = {"image_path": "/path/to/product.jpg", "detect_barcodes": True}
        mock_ocr_processor.return_value.detect_barcodes.return_value = [
            {"type": "EAN-13", "data": "123456789012"}
        ]

        # When
        response = await ocr_agent.process(input_data)

        # Then
        assert response.success is True
        assert "barcodes" in response.data
        assert len(response.data["barcodes"]) > 0

    @pytest.mark.asyncio
    async def test_process_with_specialized_model(self, ocr_agent, mock_ocr_processor):
        """Test użycia specjalizowanego modelu OCR"""
        # Given
        input_data = {"image_path": "/path/to/receipt.jpg", "model": "receipt"}

        # When
        await ocr_agent.process(input_data)

        # Then
        call_args = mock_ocr_processor.return_value.process_image.call_args
        assert call_args[1]["model"] == "receipt"

    @pytest.mark.asyncio
    async def test_process_with_streaming_output(self, ocr_agent):
        """Test streamowania wyników OCR"""
        # Given
        input_data = {"image_path": "/path/to/large_document.pdf"}

        # Konfiguracja mocka dla streamingu
        async def mock_streaming_processor():
            yield {"page": 1, "text": "Page 1 content"}
            yield {"page": 2, "text": "Page 2 content"}

        with patch("src.backend.agents.ocr_agent.OCRProcessor") as mock_processor:
            mock_processor.return_value.process_pdf_stream.return_value = (
                mock_streaming_processor()
            )

            # When
            response = await ocr_agent.process(input_data, stream=True)

            # Then
            assert response.text_stream is not None
            collected_chunks = []
            async for chunk in response.text_stream:
                collected_chunks.append(chunk)
            assert len(collected_chunks) == 2
