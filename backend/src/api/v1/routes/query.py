"""Legal research query endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.api.v1.schemas import ErrorResponse, QueryRequest, QueryResponse
from src.constants import ERROR_CODE_CORPUS_NOT_INDEXED, ERROR_CODE_LLM_UNAVAILABLE
from src.core.exceptions import CorpusNotIndexedError, LLMUnavailableError
from src.core.runtime import ensure_runtime
from src.utils.responses import get_correlation_id_from_headers
from src.utils.timestamps import timestamp_iso

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def submit_query(request: Request, payload: QueryRequest) -> QueryResponse | JSONResponse:
    """
    Execute a legal research query.

    Processes the user's query through the autonomous legal research agent,
    which decomposes the query, retrieves relevant precedents, and provides
    analysis and recommendations.

    Args:
        request: FastAPI request object.
        payload: QueryRequest with the user's query text.

    Returns:
        QueryResponse with agent findings and message IDs.

    Raises:
        HTTP 503 if LLM provider is unavailable.
        HTTP 409 if corpus has not been indexed yet.
    """
    runtime = await ensure_runtime(request.app)
    query_service = runtime.query_service
    correlation_id = get_correlation_id_from_headers(dict(request.headers))

    # Get recent conversation context for the LLM
    history = await query_service.get_recent_context()

    try:
        result = await query_service.execute_query(
            query_text=payload.query,
            correlation_id=correlation_id,
            history=history,
        )
    except LLMUnavailableError as exc:
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                correlation_id=correlation_id,
                error_code=ERROR_CODE_LLM_UNAVAILABLE,
                message=str(exc),
                timestamp=timestamp_iso(),
            ).model_dump(),
        )
    except CorpusNotIndexedError as exc:
        return JSONResponse(
            status_code=409,
            content=ErrorResponse(
                correlation_id=correlation_id,
                error_code=ERROR_CODE_CORPUS_NOT_INDEXED,
                message=str(exc),
                timestamp=timestamp_iso(),
            ).model_dump(),
        )

    return QueryResponse(
        correlation_id=correlation_id,
        query_type=result.get("query_type", "unknown"),
        chat_response=result.get("chat_response", ""),
        response=result.get("response"),
        sources_searched=result.get("sources_searched", 0),
        processing_time_ms=result.get("processing_time_ms", 0),
        user_message_id=result.get("user_message_id", ""),
        assistant_message_id=result.get("assistant_message_id", ""),
    )
