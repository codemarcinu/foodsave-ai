from typing import Any, Dict

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from src.backend.infrastructure.database.database import get_db

from ..agents.orchestrator import Orchestrator

router = APIRouter()


@router.post("/upload")
async def upload_file(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Endpoint to upload files (images or PDFs) to be processed by the Orchestrator.
    """
    try:
        content_type = file.content_type
        if not content_type:
            raise HTTPException(status_code=400, detail="Cannot determine file type")

        if not content_type.startswith("image/") and content_type != "application/pdf":
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {content_type}. Supported types: image/*, application/pdf",
            )

        file_bytes = await file.read()

        orchestrator = Orchestrator(db)
        response = await orchestrator.process_file(
            file_bytes=file_bytes,
            filename=file.filename,
            session_id=session_id,
            content_type=content_type,
        )

        if "error" in response:
            raise HTTPException(status_code=500, detail=response["error"])

        return response

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )
