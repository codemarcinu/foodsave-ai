from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.backend.api.v2.endpoints.receipts import router
from src.backend.api.v2.exceptions import APIErrorCodes

app = FastAPI()
app.include_router(router)
client = TestClient(app)


@pytest.fixture
def mock_ocr_agent():
    with patch("src.backend.api.v2.endpoints.receipts.OCRAgent") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance

        # Create an async mock for the process method
        async def async_process(*args, **kwargs):
            return MagicMock(
                success=True, text="Test receipt text", message="Processed successfully"
            )

        mock_instance.process = async_process
        yield mock_instance


def test_upload_receipt_success_image(mock_ocr_agent):
    """Test successful receipt upload with image"""
    test_image = BytesIO(b"fake image data")
    response = client.post(
        "/receipts/upload", files={"file": ("receipt.jpg", test_image, "image/jpeg")}
    )

    assert response.status_code == 200
    assert response.json() == {
        "status_code": 200,
        "message": "Receipt processed successfully",
        "data": {"text": "Test receipt text", "message": "Processed successfully"},
    }


def test_upload_receipt_missing_content_type(mock_ocr_agent):
    """Test missing content type header"""
    response = client.post(
        "/receipts/upload",
        files={
            "file": ("receipt.jpg", b"fake data", "")
        },  # Set content_type to empty string
    )

    assert response.status_code == 400
    response_json = response.json()
    assert response_json["status_code"] == 400
    assert response_json["error_code"] == APIErrorCodes.INVALID_INPUT
    assert response_json["message"] == "Missing content type header"
    assert response_json["details"] == {
        "field": "file",
        "error": "Content-Type header is required",
    }
