"""
Hugging Face routes — upload and download adapters.
"""

from fastapi import APIRouter, HTTPException

from backend.models.schemas import HFUploadRequest, HFUploadResponse, HFDownloadRequest, HFDownloadResponse
from backend.services.hf_service import HuggingFaceService
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["huggingface"])


@router.post("/hf-upload", response_model=HFUploadResponse)
async def hf_upload(request: HFUploadRequest):
    """Upload LoRA adapter to Hugging Face Hub."""
    try:
        service = HuggingFaceService()
        result = service.upload_adapter(
            adapter_type=request.adapter_type,
            repo_id=request.repo_id,
        )
        return HFUploadResponse(**result)
    except Exception as e:
        logger.error("HF upload failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hf-download", response_model=HFDownloadResponse)
async def hf_download(request: HFDownloadRequest):
    """Download LoRA adapter from Hugging Face Hub."""
    try:
        service = HuggingFaceService()
        result = service.download_adapter(
            adapter_type=request.adapter_type,
            repo_id=request.repo_id,
        )
        return HFDownloadResponse(**result)
    except Exception as e:
        logger.error("HF download failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
