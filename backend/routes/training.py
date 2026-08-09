"""
Training route — POST /api/train-model + GET /api/train-model/status.

Submits fine-tuning to Kaggle GPU and provides status polling.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from backend.models.schemas import TrainRequest, KaggleJobResponse, KaggleStatusResponse
from backend.services.training_service import TrainingService
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["training"])


@router.post("/train-model")
async def train_model(request: TrainRequest):
    """
    Submit fine-tuning job to Kaggle GPU.

    Returns immediately with job info. Poll /api/train-model/status for progress.
    """
    try:
        service = TrainingService()
        result = await service.start_training(
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Training submission failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/train-model/status")
async def training_status():
    """Poll Kaggle kernel status for fine-tuning job."""
    try:
        service = TrainingService()
        result = await service.check_training_status()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Status check failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
