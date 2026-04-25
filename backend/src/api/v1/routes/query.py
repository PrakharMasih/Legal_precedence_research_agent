from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.agent.agent import LegalResearchAgent
from src.agent.tools import ResearchToolbox
from src.api.v1.schemas import ErrorResponse, QueryRequest, QueryResponse
from src.core.exceptions import CorpusNotIndexedError, LLMUnavailableError
from src.core.runtime import ensure_runtime
from src.models.conversation import Message
from src.retrieval.retriever import Retriever

router = APIRouter(prefix="/query", tags=["query"])


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _build_agent(runtime: Any) -> tuple[LegalResearchAgent, Any]:
    retriever = Retriever(
        session_factory=runtime.session_factory,
        vector_store=runtime.vector_store,
        embedder=runtime.embedder,
    )
    toolbox = ResearchToolbox(
        retriever=retriever,
        document_repository=runtime.document_repository,
        chunk_repository=runtime.chunk_repository,
    )
    return LegalResearchAgent(llm_provider=runtime.llm_provider, toolbox=toolbox), toolbox


@router.post("", response_model=QueryResponse)
async def submit_query(request: Request, payload: QueryRequest) -> QueryResponse | JSONResponse:
    runtime = await ensure_runtime(request.app)
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid4()))
    now = datetime.now(UTC)

    recent_messages = await runtime.chat_repository.get_recent_context()
    history = [{"role": m.role, "content": m.content} for m in recent_messages]

    agent, _ = _build_agent(runtime)

    try:
        result = await agent.run(payload.query, correlation_id, history=history)
    except LLMUnavailableError as exc:
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                correlation_id=correlation_id,
                error_code="LLM_UNAVAILABLE",
                message=str(exc),
                timestamp=_timestamp(),
            ).model_dump(),
        )
    except CorpusNotIndexedError as exc:
        return JSONResponse(
            status_code=409,
            content=ErrorResponse(
                correlation_id=correlation_id,
                error_code="CORPUS_NOT_INDEXED",
                message=str(exc),
                timestamp=_timestamp(),
            ).model_dump(),
        )

    user_msg = Message(
        id=str(uuid4()),
        role="user",
        content=payload.query,
        query_type=result["query_type"],
        sources_searched=result["sources_searched"],
        created_at=now,
    )
    assistant_msg = Message(
        id=str(uuid4()),
        role="assistant",
        content=result.get("chat_response") or "",
        query_type=result["query_type"],
        sources_searched=result["sources_searched"],
        raw_response=result.get("response"),
        agent_steps=result.get("agent_steps"),
        created_at=datetime.now(UTC),
    )
    await runtime.chat_repository.append(user_msg)
    await runtime.chat_repository.append(assistant_msg)

    return QueryResponse(
        correlation_id=correlation_id,
        query_type=result["query_type"],
        chat_response=result.get("chat_response", ""),
        response=result["response"],
        sources_searched=result["sources_searched"],
        processing_time_ms=result["processing_time_ms"],
    )
