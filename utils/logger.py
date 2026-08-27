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


class _LogAdapter:
    """Wrapper around Loguru logger to support both standard printf-style (%s) and str.format ({}) args."""

    def __init__(self, logger_inst):
        self._logger = logger_inst

    @staticmethod
    def _format(msg, args):
        if not args:
            return msg
        if isinstance(msg, str):
            if "%" in msg and not ("{" in msg and "}" in msg):
                try:
                    return msg % args
                except Exception:
                    pass
            elif "{" in msg and "}" in msg:
                try:
                    return msg.format(*args)
                except Exception:
                    pass
            else:
                try:
                    return msg % args
                except Exception:
                    try:
                        return msg.format(*args)
                    except Exception:
                        pass
        return msg

    def info(self, msg, *args, **kwargs):
        self._logger.opt(depth=1).info(self._format(msg, args), **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._logger.opt(depth=1).warning(self._format(msg, args), **kwargs)

    def error(self, msg, *args, **kwargs):
        self._logger.opt(depth=1).error(self._format(msg, args), **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._logger.opt(depth=1).debug(self._format(msg, args), **kwargs)

    def exception(self, msg, *args, **kwargs):
        self._logger.opt(depth=1).exception(self._format(msg, args), **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._logger.opt(depth=1).critical(self._format(msg, args), **kwargs)

    def log(self, level, msg, *args, **kwargs):
        self._logger.opt(depth=1).log(level, self._format(msg, args), **kwargs)

    def bind(self, **kwargs):
        return _LogAdapter(self._logger.bind(**kwargs))

    def opt(self, **kwargs):
        return _LogAdapter(self._logger.opt(**kwargs))

    def __getattr__(self, name):
        return getattr(self._logger, name)


def get_logger(name: str = "ai_memory_engine"):
    """Return a Loguru logger bound to *name* with formatting support."""
    _configure_once()
    return _LogAdapter(_loguru_logger.bind(name=name))
