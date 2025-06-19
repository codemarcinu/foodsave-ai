from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.backend.agents.ocr_agent import OCRAgent, OCRAgentInput
from src.backend.api.v2.exceptions import (
    APIErrorCodes,
    BadRequestError,
    UnprocessableEntityError,
)

router = APIRouter(prefix="/receipts", tags=["Receipts"])


@router.post("/upload", response_model=None)
async def upload_receipt(file: UploadFile = File(...)):
    """Endpoint for uploading and processing receipt images with OCR.

    Returns:
        JSONResponse: Standardized success or error response
    """
    try:
        # Validate file content type
        if not file.content_type:
            raise HTTPException(
                status_code=400,
                detail={
                    "status_code": 400,
                    "error_code": "BAD_REQUEST",
                    "message": "Missing content type header",
                    "details": {
                        "field": "file",
                        "error": "Content-Type header is required",
                    },
                },
            )

        if file.content_type.startswith("image/"):
            file_type = "image"
        elif file.content_type == "application/pdf":
            file_type = "pdf"
        else:
            raise BadRequestError(
                message="Unsupported file type",
                details={
                    "content_type": file.content_type,
                    "supported_types": ["image/*", "application/pdf"],
                },
            )

        # Read and process file
        file_bytes = await file.read()
        agent = OCRAgent()
        input_data = OCRAgentInput(file_bytes=file_bytes, file_type=file_type)
        result = await agent.process(input_data)

        if not result.success:
            raise UnprocessableEntityError(
                message="Failed to process receipt",
                details={
                    "error": result.error,
                    "error_code": APIErrorCodes.RECEIPT_PROCESSING_ERROR,
                },
            )

        return JSONResponse(
            status_code=200,
            content={
                "status_code": 200,
                "message": "Receipt processed successfully",
                "data": {"text": result.text, "message": result.message},
            },
        )

    except HTTPException as he:
        # Re-raise HTTPException to preserve the status code and detail
        raise he
    except Exception as e:
        # For other exceptions, return a 500 error with standardized format
        return JSONResponse(
            status_code=500,
            content={
                "status_code": 500,
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "Unexpected error processing receipt",
                "details": {"error": str(e)},
            },
        )
