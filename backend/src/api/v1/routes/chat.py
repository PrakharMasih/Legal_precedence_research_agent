"""Chat history endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from src.api.v1.schemas import ChatHistoryResponse, MessageItem
from src.constants import (
    DEFAULT_CHAT_HISTORY_LIMIT,
    DEFAULT_CHAT_HISTORY_OFFSET,
    MAX_CHAT_HISTORY_LIMIT,
    MIN_CHAT_HISTORY_LIMIT,
    MIN_CHAT_HISTORY_OFFSET,
)
from src.core.runtime import ensure_runtime

router = APIRouter(prefix="/chat", tags=["chat"])


@router.get("/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    request: Request,
    limit: int = Query(
        default=DEFAULT_CHAT_HISTORY_LIMIT, ge=MIN_CHAT_HISTORY_LIMIT, le=MAX_CHAT_HISTORY_LIMIT
    ),
    offset: int = Query(default=DEFAULT_CHAT_HISTORY_OFFSET, ge=MIN_CHAT_HISTORY_OFFSET),
) -> ChatHistoryResponse:
    """
    Get paginated chat history.

    Returns a chronological list of all messages in the conversation.

    Args:
        request: FastAPI request object.
        limit: Number of messages to retrieve (1-200).
        offset: Number of messages to skip (>= 0).

    Returns:
        ChatHistoryResponse with paginated messages and metadata.
    """
    runtime = await ensure_runtime(request.app)
    chat_service = runtime.chat_service

    messages, total = await chat_service.get_chat_history(limit=limit, offset=offset)

    return ChatHistoryResponse(
        total=total,
        limit=limit,
        offset=offset,
        messages=[
            MessageItem(
                id=m.id,
                role=m.role,
                content=m.content,
                query_type=m.query_type,
                sources_searched=m.sources_searched,
                created_at=m.created_at.isoformat(),
                raw_response=m.raw_response,
                agent_steps=m.agent_steps,
            )
            for m in messages
        ],
    )


@router.delete("", status_code=200)
async def clear_chat_history(request: Request) -> dict:
    """
    Clear all chat history.

    Deletes all messages from the conversation.

    Args:
        request: FastAPI request object.

    Returns:
        Confirmation dict with cleared flag.
    """
    runtime = await ensure_runtime(request.app)
    chat_service = runtime.chat_service
    await chat_service.clear_history()
    return {"cleared": True}
