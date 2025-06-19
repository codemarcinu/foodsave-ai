from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.backend.api.v2.exceptions import APIErrorCodes
from src.backend.main import app

client = TestClient(app)


@pytest.fixture
def mock_ocr_agent():
    with patch("src.backend.agents.ocr_agent.OCRAgent") as mock:
        yield mock


def test_upload_receipt_success_image(mock_ocr_agent):
    """Test successful receipt upload with image"""
    mock_agent = MagicMock()
    mock_agent.process.return_value = MagicMock(
        success=True, text="Test receipt text", message="Processed successfully"
    )
    mock_ocr_agent.return_value = mock_agent

    test_image = BytesIO(b"fake image data")
    response = client.post(
        "/api/v2/receipts/upload",
        files={"file": ("receipt.jpg", test_image, "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status_code": 200,
        "message": "Receipt processed successfully",
        "data": {"text": "Test receipt text", "message": "Processed successfully"},
    }


def test_upload_receipt_success_pdf(mock_ocr_agent):
    """Test successful receipt upload with PDF"""
    mock_agent = MagicMock()
    mock_agent.process.return_value = MagicMock(
        success=True, text="Test PDF receipt", message="PDF processed"
    )
    mock_ocr_agent.return_value = mock_agent

    test_pdf = BytesIO(b"fake pdf data")
    response = client.post(
        "/api/v2/receipts/upload",
        files={"file": ("receipt.pdf", test_pdf, "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["data"]["text"] == "Test PDF receipt"


def test_upload_receipt_missing_content_type():
    """Test missing content type header"""
    response = client.post(
        "/api/v2/receipts/upload", files={"file": ("receipt.jpg", b"fake data")}
    )

    assert response.status_code == 400
    assert response.json() == {
        "status_code": 400,
        "error_code": APIErrorCodes.INVALID_INPUT,
        "message": "Missing content type header",
        "details": {"field": "file", "error": "Content-Type header is required"},
    }


def test_upload_receipt_unsupported_type():
    """Test unsupported file type"""
    test_file = BytesIO(b"fake data")
    response = client.post(
        "/api/v2/receipts/upload",
        files={"file": ("receipt.txt", test_file, "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == APIErrorCodes.INVALID_INPUT
    assert "Unsupported file type" in response.json()["message"]


def test_upload_receipt_processing_error(mock_ocr_agent):
    """Test receipt processing failure"""
    mock_agent = MagicMock()
    mock_agent.process.return_value = MagicMock(
        success=False, error="OCR processing failed"
    )
    mock_ocr_agent.return_value = mock_agent

    test_image = BytesIO(b"fake image data")
    response = client.post(
        "/api/v2/receipts/upload",
        files={"file": ("receipt.jpg", test_image, "image/jpeg")},
    )

    assert response.status_code == 422
    assert response.json()["error_code"] == APIErrorCodes.RECEIPT_PROCESSING_ERROR


def test_upload_receipt_internal_error(mock_ocr_agent):
    """Test unexpected internal error"""
    mock_agent = MagicMock()
    mock_agent.process.side_effect = Exception("Unexpected error")
    mock_ocr_agent.return_value = mock_agent

    test_image = BytesIO(b"fake image data")
    response = client.post(
        "/api/v2/receipts/upload",
        files={"file": ("receipt.jpg", test_image, "image/jpeg")},
    )

    assert response.status_code == 500
    assert response.json()["error_code"] == APIErrorCodes.INTERNAL_ERROR
