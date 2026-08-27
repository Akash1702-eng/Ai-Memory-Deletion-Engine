"""
FastAPI application entry point.

Registers all routers, serves the frontend SPA, and provides health check.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.models.database import init_local_db
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    settings = get_settings()
    logger.info("=" * 60)
    logger.info("AI MEMORY ENGINE — Starting")
    logger.info("Model: %s", settings.model_name)
    logger.info("Compute: Kaggle GPU (remote)")
    logger.info("=" * 60)

    # Initialize database
    init_local_db()

    # Set Hugging Face environment variable if configured
    if settings.hf_token and settings.hf_token.strip():
        os.environ["HF_TOKEN"] = settings.hf_token.strip()
        os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.hf_token.strip()

    # Create data directories
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.dataset_dir, exist_ok=True)
    os.makedirs(settings.model_checkpoint_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    yield

    logger.info("AI Memory Engine — Shutting down")


# ── Create App ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Memory Engine",
    description=(
        "Machine Unlearning demonstration — fine-tune, unlearn, and compare "
        "Qwen2.5-1.5B-Instruct with LoRA adapters."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register Routers ─────────────────────────────────────────────────────────

from backend.routes import chat, upload, training, unlearning, evaluation, huggingface, memories, auth, payment

app.include_router(chat.router)
app.include_router(upload.router)
app.include_router(training.router)
app.include_router(unlearning.router)
app.include_router(evaluation.router)
app.include_router(huggingface.router)
app.include_router(memories.router)
app.include_router(auth.router)
app.include_router(payment.router)

# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    settings = get_settings()
    return {
        "status": "ok",
        "model": settings.model_name,
        "model_version": settings.active_model_version,
        "compute": "kaggle-gpu",
        "kaggle_user": settings.kaggle_username or "not configured",
    }


# ── Static Files + SPA ───────────────────────────────────────────────────────

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
    app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")

    @app.get("/")
    async def serve_spa():
        """Serve the frontend SPA."""
        return FileResponse(str(FRONTEND_DIR / "index.html"))
else:
    @app.get("/")
    async def no_frontend():
        return JSONResponse({"message": "API running. Frontend not found."})


# ── Exception Handler ────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )
