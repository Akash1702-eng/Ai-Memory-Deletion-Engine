"""
Centralized logging with Loguru.

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Server started on port {}", port)
"""

import sys
from functools import lru_cache
from pathlib import Path

from loguru import logger as _loguru_logger

# Remove Loguru's default handler so we can configure our own.
_loguru_logger.remove()

_CONFIGURED = False


def _configure_once() -> None:
    """Add stderr + file sinks the first time any logger is requested."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    from config.settings import get_settings
    settings = get_settings()

    # ── Console sink (human-readable) ─────────────────────────────────
    _loguru_logger.add(
        sys.stderr,
        level=settings.log_level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # ── File sink (rotating) ──────────────────────────────────────────
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    _loguru_logger.add(
        str(log_path),
        level=settings.log_level.upper(),
        format="{time:YYYY-MM-DDTHH:mm:ss.SSSZ} | {level} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="30 days",
        compression="gz",
        serialize=False,
        enqueue=True,
    )

    _CONFIGURED = True


def get_logger(name: str = "ai_memory_engine"):
    """Return a Loguru logger bound to *name*."""
    _configure_once()
    return _loguru_logger.bind(name=name)
