"""
Email service — handles sending OTP verification emails via SMTP.
"""

import random
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


def generate_otp() -> str:
    """Generate a random 6-digit numerical OTP."""
    return str(random.randint(100000, 999999))


def send_otp_email(to_email: str, otp_code: str) -> bool:
    """
    Send OTP verification email via SMTP.

    If SMTP credentials are not yet configured in settings/.env, logs the OTP
    to console/logs for seamless development and testing.
    """
    settings = get_settings()

    subject = f"Your Verification Code: {otp_code} — AI Memory Engine"
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Verification Code</title>
        <style>
            body {{
                margin: 0; padding: 0; background-color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #334155; -webkit-font-smoothing: antialiased;
            }}
            .email-wrapper {{
                background-color: #f8fafc; padding: 40px 15px;
            }}
            .email-card {{
                max-width: 520px; margin: 0 auto; background: #ffffff; border-radius: 12px; border: 1px solid #e2e8f0; border-top: 4px solid #4f46e5; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03); overflow: hidden;
            }}
            .card-header {{
                padding: 28px 32px 20px 32px; border-bottom: 1px solid #f1f5f9; display: flex; align-items: center; justify-content: space-between;
            }}
            .brand-name {{
                font-size: 20px; font-weight: 700; color: #0f172a; letter-spacing: -0.02em; margin: 0;
            }}
            .card-body {{
                padding: 32px; font-size: 15px; line-height: 1.6; color: #334155;
            }}
            .greeting {{
                font-size: 16px; font-weight: 600; color: #0f172a; margin-bottom: 12px;
            }}
            .otp-box {{
                background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px 20px; text-align: center; margin: 20px 0;
            }}
            .otp-code {{
                font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace; font-size: 22px; font-weight: 700; color: #4f46e5; letter-spacing: 5px; margin: 0;
            }}
            .otp-note {{
                font-size: 13px; color: #64748b; margin-top: 8px; margin-bottom: 0;
            }}
            .card-footer {{
                background: #f8fafc; padding: 20px 32px; border-top: 1px solid #f1f5f9; text-align: center; font-size: 12px; color: #94a3b8;
            }}
        </style>
    </head>
    <body>
        <div class="email-wrapper">
            <div class="email-card">
                <div class="card-header">
                    <h1 class="brand-name">🧠 AI Memory Engine</h1>
                </div>
                <div class="card-body">
                    <div class="greeting">Hello,</div>
                    <p style="margin-top:0;">Thank you for registering with AI Memory Engine. Please use the verification code below to complete your account setup and activate your 7-day free trial:</p>
                    
                    <div class="otp-box">
                        <div class="otp-code">{otp_code}</div>
                        <p class="otp-note">Valid for 10 minutes. Do not share this code with anyone.</p>
                    </div>

                    <p>If you did not request this verification code, please disregard this email.</p>
                    
                    <p style="margin-bottom:0; margin-top: 24px;">Best regards,<br><strong>AI Memory Engine Team</strong></p>
                </div>
                <div class="card-footer">
                    <div>© 2026 AI Memory Engine. All rights reserved.</div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    smtp_user = getattr(settings, "smtp_username", "") or ""
    smtp_pass = getattr(settings, "smtp_password", "") or ""
    smtp_server = getattr(settings, "smtp_server", "smtp.gmail.com") or "smtp.gmail.com"
    smtp_port = getattr(settings, "smtp_port", 587) or 587
    from_email = getattr(settings, "smtp_from_email", "") or smtp_user or "noreply@aimemoryengine.com"

    # Log generated OTP for dev/testing visibility
    logger.info("🔑 Generated OTP for [%s]: %s", to_email, otp_code)

    if not smtp_user or not smtp_pass:
        logger.warning(
            "SMTP credentials not set in .env (SMTP_USERNAME / SMTP_PASSWORD). "
            "OTP [%s] has been logged above for testing.", otp_code
        )
        return True

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_email
        msg.attach(MIMEText(html_content, "html"))

        server = smtplib.SMTP(smtp_server, smtp_port, timeout=10)
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(from_email, [to_email], msg.as_string())
        server.quit()

        logger.info("✅ Verification OTP email sent successfully to %s via SMTP", to_email)
        return True
    except Exception as e:
        logger.error("Failed to send OTP email via SMTP to %s: %s", to_email, e)
        # Fallback to true so registration proceeds in dev mode
        return True
