"""WebSocket endpoint for real-time agentic research.

Protocol (server → client):
    {"type": "agent_started", "correlation_id": "...", "message": "..."}  ← sent before pipeline
    {"type": "thinking",     "step": N, "phase": "planning|retrieval|reflection", "message": "..."}
    {"type": "tool_result",  "step": N, "tool": "search_corpus", "query": "...",
                              "total_returned": N, "top_results": [...]}
    {"type": "reasoning",    "step": N, "message": "...", "issue": "...",
                              "rules_count": N, "precedent_strengths": {...}}
    {"type": "synthesizing", "step": N, "message": "...", "unique_documents": [...]}
    {"type": "query_type",   "step": N, "is_research": bool, "message": "..."}
    {"type": "streaming",    "step": N, "message": "..."}
    {"type": "stream_chunk",             "content": "..."}   ← many, no step
    {"type": "completed",   "message_id": "...", "query_type": "...", "sources_searched": N}
    {"type": "error",       "error_code": "...", "message": "..."}

Client → server:
    {"query": "...", "mode": "auto|research|general"}
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.agent.agent import LegalResearchAgent
from src.agent.tools import ResearchToolbox
from src.core.exceptions import CorpusNotIndexedError, LLMUnavailableError
from src.core.runtime import ensure_runtime
from src.models.conversation import Message
from src.retrieval.retriever import Retriever

router = APIRouter(tags=["websocket"])

_logger = logging.getLogger(__name__)

_VALID_MODES = frozenset({"auto", "research", "general"})


def _build_agent(runtime: Any) -> LegalResearchAgent:
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
    return LegalResearchAgent(llm_provider=runtime.llm_provider, toolbox=toolbox)


@router.websocket("/ws/query")
async def ws_query(websocket: WebSocket) -> None:
    """Real-time agentic research endpoint."""
    correlation_id = str(uuid4())
    _logger.info("ws_query: connection accepted", extra={"correlation_id": correlation_id})

    await websocket.accept()
    _logger.info("ws_query: waiting for message", extra={"correlation_id": correlation_id})

    try:
        # Use receive_text for max compatibility (text frames from browsers/clients)
        raw_text = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
        _logger.info(
            "ws_query: received text frame",
            extra={"correlation_id": correlation_id, "text_length": len(raw_text)},
        )
        raw: dict[str, Any] = json.loads(raw_text)
        _logger.info(
            "ws_query: parsed message",
            extra={"correlation_id": correlation_id, "message_keys": list(raw.keys())},
        )
    except TimeoutError:
        _logger.error(
            "ws_query: receive_text() timeout after 60s — client connected but never sent",
            extra={"correlation_id": correlation_id},
        )
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "error_code": "RECEIVE_TIMEOUT",
                    "message": "Client did not send message within 60 seconds.",
                }
            )
            await websocket.close()
        except Exception:
            pass
        return
    except WebSocketDisconnect:
        _logger.warning(
            "ws_query: client disconnected before sending message",
            extra={"correlation_id": correlation_id},
        )
        return
    except ValueError:
        _logger.error(
            "ws_query: invalid JSON received",
            extra={"correlation_id": correlation_id},
        )
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "error_code": "INVALID_JSON",
                    "message": "Request must be valid JSON.",
                }
            )
            await websocket.close()
        except Exception:
            pass
        return
    except Exception as exc:
        _logger.exception(
            "ws_query: unexpected error receiving message",
            extra={"correlation_id": correlation_id, "error_type": type(exc).__name__},
        )
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "error_code": "RECEIVE_ERROR",
                    "message": f"Failed to receive message: {type(exc).__name__}",
                }
            )
            await websocket.close()
        except Exception:
            pass
        return

    query: str = str(raw.get("query", "")).strip()
    mode: str = str(raw.get("mode", "auto")).strip()
    _logger.info(
        "ws_query: parsed request",
        extra={"correlation_id": correlation_id, "query_length": len(query), "mode": mode},
    )

    if mode not in _VALID_MODES:
        mode = "auto"

    if not query:
        _logger.warning("ws_query: empty query", extra={"correlation_id": correlation_id})
        await websocket.send_json(
            {"type": "error", "error_code": "EMPTY_QUERY", "message": "Query must not be empty."}
        )
        await websocket.close()
        return

    # Inject app reference through websocket scope (starlette stores it on scope["app"])
    app = websocket.scope["app"]

    try:
        _logger.info("ws_query: ensuring runtime", extra={"correlation_id": correlation_id})
        runtime = await ensure_runtime(app)
        settings = runtime.settings
        _logger.info(
            "ws_query: runtime initialized",
            extra={"correlation_id": correlation_id, "llm_provider": settings.llm_provider},
        )

        # Load conversation history for context
        _logger.info("ws_query: loading chat history", extra={"correlation_id": correlation_id})
        messages = await runtime.chat_repository.get_recent_context()
        # Convert Message objects to dict format for agent
        history = [
            {"role": msg.role, "content": msg.content}
            for msg in messages[-10:]
            if msg.role in ("user", "assistant")
        ]
        _logger.info(
            "ws_query: chat history loaded",
            extra={"correlation_id": correlation_id, "history_messages": len(history)},
        )

        # Persist user message
        user_msg = Message(
            id=str(uuid4()),
            role="user",
            content=query,
            sources_searched=0,
            created_at=datetime.now(UTC),
        )
        await runtime.chat_repository.append(user_msg)
        _logger.info(
            "ws_query: user message persisted",
            extra={"correlation_id": correlation_id, "message_id": user_msg.id},
        )

        # Build agent
        _logger.info("ws_query: building agent", extra={"correlation_id": correlation_id})
        agent = _build_agent(runtime)
        _logger.info("ws_query: agent built", extra={"correlation_id": correlation_id})

        async def on_event(event: dict[str, Any]) -> None:
            """Forward every event to the WebSocket client."""
            try:
                await websocket.send_json(
                    json.loads(json.dumps(event, ensure_ascii=False, default=str))
                )
            except WebSocketDisconnect:
                _logger.info(
                    "ws_query: client disconnected during event",
                    extra={"correlation_id": correlation_id},
                )
                raise
            except Exception:  # noqa: BLE001
                _logger.exception(
                    "ws_query: failed to send event",
                    extra={"correlation_id": correlation_id, "event_type": event.get("type")},
                )
                # Don't re-raise; client may be temporarily unavailable

        # Signal client before the pipeline starts — lets the frontend mount the
        # thinking-process panel so it captures step 1 (planning) without a race condition.
        await websocket.send_json(
            {
                "type": "agent_started",
                "correlation_id": correlation_id,
                "message": "Agent initialized — starting research pipeline…",
            }
        )

        # Run agent with streaming
        force_mode: str | None = None if mode == "auto" else mode
        _logger.info(
            "ws_query: starting agent.run()",
            extra={
                "correlation_id": correlation_id,
                "force_mode": force_mode,
                "query_length": len(query),
            },
        )

        result = await agent.run(
            query_text=query,
            correlation_id=correlation_id,
            history=history,
            on_event=on_event,
            force_mode=force_mode,
        )

        query_type: str = result.get("query_type", "general_query")
        chat_response: str = result.get("chat_response", "")
        raw_response: dict[str, Any] = result.get("response", {})
        sources_searched: int = int(result.get("sources_searched", 0))
        agent_steps: list[dict[str, Any]] = result.get("agent_steps", [])

        # Persist assistant message with full thinking trace
        assistant_msg = Message(
            id=str(uuid4()),
            role="assistant",
            content=chat_response,
            query_type=query_type,
            sources_searched=sources_searched,
            raw_response=raw_response,
            agent_steps=agent_steps,
            created_at=datetime.now(UTC),
        )
        await runtime.chat_repository.append(assistant_msg)

        # Signal completion — frontend fetches history after this
        await websocket.send_json(
            {
                "type": "completed",
                "message_id": assistant_msg.id,
                "query_type": query_type,
                "sources_searched": sources_searched,
                "processing_time_ms": result.get("processing_time_ms", 0),
            }
        )
        _logger.info(
            "ws_query: completed successfully",
            extra={"correlation_id": correlation_id, "query_type": query_type},
        )

    except WebSocketDisconnect:
        _logger.info("ws_query: client disconnected", extra={"correlation_id": correlation_id})

    except CorpusNotIndexedError:
        _logger.warning(
            "ws_query: corpus not indexed",
            extra={"correlation_id": correlation_id},
        )
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "error_code": "CORPUS_NOT_INDEXED",
                    "message": (
                        "No documents have been indexed yet. Please ingest your PDF corpus first."
                    ),
                }
            )
        except Exception:  # noqa: BLE001
            _logger.exception(
                "ws_query: failed to send CORPUS_NOT_INDEXED error",
                extra={"correlation_id": correlation_id},
            )

    except LLMUnavailableError:
        _logger.warning(
            "ws_query: LLM unavailable",
            extra={"correlation_id": correlation_id},
        )
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "error_code": "LLM_UNAVAILABLE",
                    "message": (
                        "The LLM service is currently unavailable. Please try again shortly."
                    ),
                }
            )
        except Exception:  # noqa: BLE001
            _logger.exception(
                "ws_query: failed to send LLM_UNAVAILABLE error",
                extra={"correlation_id": correlation_id},
            )

    except Exception:  # noqa: BLE001
        _logger.exception("ws_query: unexpected error", extra={"correlation_id": correlation_id})
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "error_code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred. Please try again.",
                }
            )
        except Exception:  # noqa: BLE001
            _logger.exception(
                "ws_query: failed to send INTERNAL_ERROR",
                extra={"correlation_id": correlation_id},
            )

    finally:
        try:
            await websocket.close()
            _logger.info("ws_query: connection closed", extra={"correlation_id": correlation_id})
        except Exception:  # noqa: BLE001
            _logger.debug(
                "ws_query: error closing connection",
                extra={"correlation_id": correlation_id},
            )
            pass
