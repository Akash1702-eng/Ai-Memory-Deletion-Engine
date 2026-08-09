"""
Unlearning route — POST /api/run-unlearning + GET /api/run-unlearning/status.

Submits gradient ascent unlearning to Kaggle GPU and provides status polling.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from backend.models.schemas import UnlearningRequest, KaggleStatusResponse
from backend.services.unlearning_service import get_unlearning_service
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["unlearning"])


@router.post("/run-unlearning")
async def run_unlearning(request: UnlearningRequest):
    """
    Submit gradient ascent unlearning job to Kaggle GPU.

    Returns immediately with job info. Poll /api/run-unlearning/status for progress.
    """
    try:
        service = get_unlearning_service()
        result = await service.start_unlearning(
            forget_texts=request.forget_texts,
            retain_texts=request.retain_texts,
            test_queries=request.test_queries,
            num_epochs=request.epochs,
            learning_rate=request.learning_rate,
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Unlearning submission failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/run-unlearning/status")
async def unlearning_status():
    """Poll Kaggle kernel status for unlearning job."""
    try:
        service = get_unlearning_service()
        result = await service.check_unlearning_status()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Status check failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
