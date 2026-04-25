from __future__ import annotations

from fastapi import APIRouter, Query, Request

from src.api.v1.schemas import ChatHistoryResponse, MessageItem
from src.core.runtime import ensure_runtime

router = APIRouter(prefix="/chat", tags=["chat"])


@router.get("/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> ChatHistoryResponse:
    """Return a paginated chronological list of all messages."""
    runtime = await ensure_runtime(request.app)
    total = await runtime.chat_repository.count()
    messages = await runtime.chat_repository.get_history(limit=limit, offset=offset)
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
    """Delete all messages."""
    runtime = await ensure_runtime(request.app)
    await runtime.chat_repository.clear()
    return {"cleared": True}
