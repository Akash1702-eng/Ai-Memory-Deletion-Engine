"""
Evaluation route — POST /api/evaluation + POST /api/evaluation/before-after.
"""

from fastapi import APIRouter, HTTPException

from backend.models.schemas import (
    EvaluationRequest, EvaluationResponse,
    BeforeAfterEvaluationRequest, BeforeAfterEvaluationResponse,
)
from backend.services.evaluation_service import get_evaluation_service
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["evaluation"])


@router.post("/evaluation", response_model=EvaluationResponse)
async def run_evaluation(request: EvaluationRequest):
    """Run model evaluation and optional MIA."""
    try:
        service = get_evaluation_service()
        result = await service.run_evaluation(
            test_queries=request.test_queries,
            model_type=request.model_type,
            forgotten_texts=request.forgotten_texts,
            non_member_texts=request.non_member_texts,
        )
        return EvaluationResponse(**result)
    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluation/before-after", response_model=BeforeAfterEvaluationResponse)
async def run_before_after_evaluation(request: BeforeAfterEvaluationRequest):
    """
    Run evaluation comparing fine-tuned (before) vs unlearned (after) models.

    This is the primary evaluation endpoint that demonstrates unlearning effectiveness.
    """
    try:
        service = get_evaluation_service()
        result = await service.run_before_after_evaluation(
            test_queries=request.test_queries,
            forgotten_texts=request.forgotten_texts,
            non_member_texts=request.non_member_texts,
        )
        return BeforeAfterEvaluationResponse(**result)
    except Exception as e:
        logger.error("Before/After evaluation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
