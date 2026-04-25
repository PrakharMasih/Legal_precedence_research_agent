from __future__ import annotations

import contextvars
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

import structlog

correlation_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id",
    default=None,
)

# Create logs directory if it doesn't exist
_LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
_LOGS_DIR.mkdir(exist_ok=True)


def _add_correlation_id(_: Any, __: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    correlation_id = correlation_id_var.get()
    if correlation_id is not None:
        event_dict.setdefault("correlation_id", correlation_id)
    return event_dict


def configure_logging(log_level: str = "INFO") -> None:
    """Configure logging with both console and file handlers with rotation."""
    log_level_upper = log_level.upper()
    numeric_level = getattr(logging, log_level_upper, logging.INFO)

    # Configure root logger with both stdout and file handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove any existing handlers
    root_logger.handlers = []

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(console_handler)

    # File handler with time-based rotation (daily)
    # Keeps 7 days of logs (7 backup files + current)
    timed_file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=str(_LOGS_DIR / "app.log"),
        when="midnight",
        interval=1,
        backupCount=7,
        utc=True,
    )
    timed_file_handler.setLevel(numeric_level)
    timed_file_handler.setFormatter(logging.Formatter("%(message)s"))
    timed_file_handler.suffix = "%Y-%m-%d"  # Readable suffix format
    root_logger.addHandler(timed_file_handler)

    # File handler with size-based rotation
    # Keeps 10 files, each max 10MB
    sized_file_handler = logging.handlers.RotatingFileHandler(
        filename=str(_LOGS_DIR / "app-size.log"),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=10,
    )
    sized_file_handler.setLevel(numeric_level)
    sized_file_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(sized_file_handler)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _add_correlation_id,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def bind_correlation_id(correlation_id: str | None) -> None:
    correlation_id_var.set(correlation_id)
    if correlation_id is not None:
        structlog.contextvars.bind_contextvars(correlation_id=correlation_id)


def clear_correlation_id() -> None:
    correlation_id_var.set(None)
    structlog.contextvars.clear_contextvars()


def get_logger(*, component: str | None = None) -> structlog.stdlib.BoundLogger:
    logger = structlog.get_logger()
    if component is not None:
        return logger.bind(component=component)
    return logger


def get_logs_directory() -> Path:
    """Get the logs directory path."""
    return _LOGS_DIR
