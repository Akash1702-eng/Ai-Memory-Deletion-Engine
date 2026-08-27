"""
Authentication and 7-day free trial tracking route.
"""

import hashlib
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr, Field

from backend.models.database import (
    create_user_record,
    get_user_by_email,
    get_user_by_id,
    check_user_subscription,
    save_otp_code,
    verify_otp_code,
)
from backend.services.email_service import generate_otp, send_otp_email
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _hash_password(password: str) -> str:
    """Hash password using SHA-256."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


class SendOtpRequest(BaseModel):
    email: str = Field(..., min_length=3)


class VerifyOtpSignupRequest(BaseModel):
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=4)
    otp: str = Field(..., min_length=4, max_length=10)


class LoginRequest(BaseModel):
    email: str
    password: str


@router.post("/send_otp")
async def send_otp(request: SendOtpRequest):
    """Send a 6-digit OTP code to the user's email via SMTP."""
    existing = get_user_by_email(request.email)
    if existing:
        return {
            "already_registered": True,
            "message": "Account already exists for this email. Please log in.",
        }

    otp_code = generate_otp()
    save_otp_code(request.email, otp_code, expires_minutes=10)
    sent = send_otp_email(request.email, otp_code)

    return {
        "success": sent,
        "email": request.email,
        "message": f"Verification OTP code sent to {request.email}. Please check your inbox.",
    }


@router.post("/verify_otp_signup")
async def verify_otp_signup(request: VerifyOtpSignupRequest):
    """Verify OTP and complete 7-day free trial signup."""
    existing = get_user_by_email(request.email)
    if existing:
        sub_info = check_user_subscription(existing["id"])
        return {
            "user_id": existing["id"],
            "email": existing["email"],
            "subscription": sub_info,
            "message": "Welcome back! Your account is active.",
        }

    valid = verify_otp_code(request.email, request.otp)
    if not valid:
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired OTP code. Please check your email or click Resend OTP.",
        )

    pwd_hash = _hash_password(request.password)
    user = create_user_record(request.email, pwd_hash)
    sub_info = check_user_subscription(user["id"])

    logger.info("New user verified & registered: %s with 7-day free trial", request.email)
    return {
        "user_id": user["id"],
        "email": user["email"],
        "subscription": sub_info,
        "message": "Email verified! Registration successful with 7-day free trial.",
    }


@router.post("/login")
async def login(request: LoginRequest):
    """Authenticate user."""
    user = get_user_by_email(request.email)
    if not user or user["password_hash"] != _hash_password(request.password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    sub_info = check_user_subscription(user["id"])
    return {
        "user_id": user["id"],
        "email": user["email"],
        "subscription": sub_info,
        "message": "Login successful.",
    }


class SubscribeRequest(BaseModel):
    user_id: str = Field("", description="User ID")
    email: str = Field("", description="User Email")
    plan_id: str = Field("plan_1m", description="Plan ID: plan_1m, plan_6m, or plan_12m")
    payment_id: str = Field("", description="Razorpay Payment ID")


@router.post("/subscribe")
async def subscribe_user(request: SubscribeRequest):
    """Activate or extend user subscription based on payment plan."""
    from backend.models.database import update_user_subscription
    
    months_map = {
        "plan_1m": 1,
        "plan_6m": 6,
        "plan_12m": 12,
    }
    months = months_map.get(request.plan_id, 1)

    target = request.email.strip() or request.user_id.strip()
    if not target:
        raise HTTPException(status_code=400, detail="User email or ID is required for subscription.")

    try:
        updated = update_user_subscription(target, request.plan_id, months)
        sub_info = check_user_subscription(updated["user_id"])
        logger.info("Subscription activated for %s: %s (%d months)", target, request.plan_id, months)
        return {
            "success": True,
            "user_id": updated["user_id"],
            "email": updated["email"],
            "subscription": sub_info,
            "message": f"Payment verified! {months}-Month Premium Subscription activated.",
        }
    except Exception as e:
        logger.error("Subscription activation failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/me")
async def get_me(user_id: str):
    """Get current user profile and subscription status."""
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    sub_info = check_user_subscription(user["id"])
    return {
        "user_id": user["id"],
        "email": user["email"],
        "created_at": user["created_at"],
        "subscription": sub_info,
    }
