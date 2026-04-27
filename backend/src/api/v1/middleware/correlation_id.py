"""Middleware for correlation ID injection and context binding."""

from __future__ import annotations

from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from src.constants import HEADER_CORRELATION_ID
from src.core.logging import bind_correlation_id, clear_correlation_id


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware that injects correlation IDs into request context.

    This middleware:
    1. Extracts correlation ID from request headers or generates a new one
    2. Binds it to the application context for use in logging
    3. Includes it in response headers
    """

    async def dispatch(self, request: Request, call_next: callable) -> Response:
        """
        Process request and inject correlation ID context.

        Args:
            request: FastAPI request object.
            call_next: Next middleware/handler in the chain.

        Returns:
            Response with correlation ID header added.
        """
        # Extract or generate correlation ID
        correlation_id = request.headers.get(HEADER_CORRELATION_ID) or str(uuid4())

        # Bind to context for logging
        bind_correlation_id(correlation_id)

        try:
            # Process request
            response = await call_next(request)
        finally:
            # Clean up context
            clear_correlation_id()

        # Add correlation ID to response headers
        response.headers[HEADER_CORRELATION_ID] = correlation_id
        return response
