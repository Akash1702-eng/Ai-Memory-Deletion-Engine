"""
Evaluation route — POST /api/evaluation + POST /api/evaluation/before-after.
"""

from fastapi import APIRouter, HTTPException

from backend.models.schemas import (
    EvaluationRequest, EvaluationResponse,
    BeforeAfterEvaluationRequest, BeforeAfterEvaluationResponse,
)
from backend.models.database import get_latest_active_user
from backend.services.email_service import send_evaluation_report_email
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
    Run evaluation comparing fine-tuned (before) vs unlearned (after) models,
    and automatically send the audit report to the corresponding user via email.
    """
    try:
        service = get_evaluation_service()
        result = await service.run_before_after_evaluation(
            test_queries=request.test_queries,
            forgotten_texts=request.forgotten_texts,
            non_member_texts=request.non_member_texts,
        )

        # ── Send Report via Email to User ────────────────────────────────────
        target_email = request.user_email
        if not target_email:
            # Fallback to the latest active user from the database
            latest = get_latest_active_user()
            if latest and latest.get("email"):
                target_email = latest["email"]

        email_sent = False
        if target_email and "@" in target_email:
            try:
                email_sent = send_evaluation_report_email(target_email, result)
                if email_sent:
                    logger.info("Evaluation audit report successfully emailed to %s", target_email)
                else:
                    logger.warning("Evaluation report email sending returned false for %s", target_email)
            except Exception as mail_err:
                logger.error("Failed to email evaluation report to %s: %s", target_email, mail_err)

        result["email_sent"] = email_sent
        result["email_recipient"] = target_email if email_sent else None

        return BeforeAfterEvaluationResponse(**result)
    except Exception as e:
        logger.error("Before/After evaluation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
