"""
Structured logging setup.
Call setup_logging() once at startup (in main.py / app factory).
All modules get their logger via: logger = logging.getLogger(__name__)
"""
import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", json_format: bool = False) -> None:
    """
    Configure root logger for the application.

    Args:
        level:       Log level string — DEBUG | INFO | WARNING | ERROR
        json_format: Use JSON lines format (for log aggregation in production).
                     Plain text format is used when False (good for local dev).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    if json_format:
        formatter = _JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()
    root.addHandler(handler)

    # Quiet down noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


class _JsonFormatter(logging.Formatter):
    """Minimal JSON-lines formatter — one JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        import traceback

        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = traceback.format_exception(*record.exc_info)
        return json.dumps(payload)
