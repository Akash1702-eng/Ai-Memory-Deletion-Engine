"""
Settings route — manages compute target settings (Kaggle GPU vs Local Machine).
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.models.database import get_compute_target, set_compute_target
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/settings", tags=["settings"])


class ComputeTargetRequest(BaseModel):
    target: str = Field(..., description="Compute target: 'kaggle' or 'local'")


@router.get("/compute_target")
async def get_target():
    """Get current compute target ('kaggle' or 'local')."""
    try:
        target = get_compute_target()
        return {"compute_target": target}
    except Exception as e:
        logger.error("Error getting compute target: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compute_target")
async def set_target(request: ComputeTargetRequest):
    """Set current compute target ('kaggle' or 'local')."""
    try:
        target = set_compute_target(request.target)
        return {
            "compute_target": target,
            "message": f"Compute execution target set to {target.upper()}",
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error setting compute target: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
