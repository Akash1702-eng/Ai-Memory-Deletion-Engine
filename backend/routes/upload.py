"""
Upload route — POST /api/upload-csv.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.models.schemas import UploadResponse
from backend.services.upload_service import UploadService
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["upload"])


@router.post("/upload-csv", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)):
    """Upload a CSV file with Q&A training data."""
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    try:
        content = await file.read()
        service = UploadService()
        result = await service.process_csv(file.filename, content)
        return UploadResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Upload failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
