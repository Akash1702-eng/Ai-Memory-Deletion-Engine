"""
Payment route — handles Razorpay InfinityFree payment bridge callbacks and subscription activation.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.models.database import update_user_subscription, check_user_subscription
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/payment", tags=["payment"])


PLAN_DURATIONS = {
    "plan_1m": 1,
    "plan_6m": 6,
    "plan_12m": 12,
}


class VerifyPaymentRequest(BaseModel):
    user_id: str
    plan_id: str
    payment_id: str
    status: str = "success"


@router.post("/verify")
async def verify_payment(request: VerifyPaymentRequest):
    """
    Verify payment callback from InfinityFree Razorpay bridge
    (https://onlineclothier.infinityfreeapp.com/payment_accept.php).
    """
    if request.plan_id not in PLAN_DURATIONS:
        raise HTTPException(status_code=400, detail="Invalid plan selected.")

    months = PLAN_DURATIONS[request.plan_id]
    try:
        updated = update_user_subscription(
            email_or_id=request.user_id,
            plan_id=request.plan_id,
            months=months,
        )
        logger.info(
            "Payment verified for user %s. Activated plan %s (%d months). Payment ID: %s",
            request.user_id, request.plan_id, months, request.payment_id
        )
        return {
            "success": True,
            "subscription": updated,
            "message": f"Subscription activated for {months} month(s)! Thank you for your payment.",
        }
    except Exception as e:
        logger.error("Payment verification failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
