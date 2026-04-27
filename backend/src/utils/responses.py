"""Utilities for building standardized HTTP responses."""

from __future__ import annotations

from uuid import uuid4

from fastapi.responses import JSONResponse

from src.api.v1.schemas import ErrorResponse
from src.constants import HEADER_CORRELATION_ID
from src.utils.timestamps import timestamp_iso


def get_correlation_id_from_headers(headers: dict[str, str]) -> str:
    """
    Extract correlation ID from request headers or generate a new one.

    Args:
        headers: Request headers dict.

    Returns:
        Existing correlation ID from headers or a newly generated UUID.
    """
    return headers.get(HEADER_CORRELATION_ID) or str(uuid4())


def build_error_response(
    *,
    correlation_id: str,
    status_code: int,
    error_code: str,
    message: str,
) -> JSONResponse:
    """
    Build a standardized error response.

    Args:
        correlation_id: Request correlation ID.
        status_code: HTTP status code.
        error_code: Application error code.
        message: Human-readable error message.

    Returns:
        JSONResponse with error details and correlation ID header.
    """
    payload = ErrorResponse(
        correlation_id=correlation_id,
        error_code=error_code,
        message=message,
        timestamp=timestamp_iso(),
    )
    response = JSONResponse(status_code=status_code, content=payload.model_dump())
    response.headers[HEADER_CORRELATION_ID] = correlation_id
    return response
