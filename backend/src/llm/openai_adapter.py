from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from src.core.exceptions import LLMUnavailableError
from src.llm.base import LLMProvider, LLMResponse

_logger = logging.getLogger(__name__)

# Retry settings for rate-limit (429) and transient (5xx) errors
# Note: _MAX_RETRIES is overridden by settings.llm_max_retries
_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 3.0  # seconds — doubles each attempt: 3, 6, 12, 24, 30 (capped)
_RETRY_MAX_DELAY = 30.0  # cap per attempt
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def _is_retryable(exc: Exception) -> tuple[bool, float]:
    """Return (should_retry, retry_after_seconds) by inspecting the exception."""
    msg = str(exc).lower()

    # Model errors (404 not_found, 400 decommissioned/invalid_request) are NOT retryable
    is_model_not_found = "404" in str(exc) and "model_not_found" in msg
    is_decommissioned = "decommissioned" in msg
    is_tool_failed = "tool_use_failed" in msg
    is_invalid_request = "400" in str(exc) and "invalid_request" in msg
    if is_model_not_found or is_decommissioned or is_tool_failed or is_invalid_request:
        return False, 0.0

    # Extract Retry-After from the message if present (Groq/OpenAI send it)
    retry_after: float = 0.0
    for part in str(exc).split():
        try:
            val = float(part.rstrip("s"))
            if 0 < val < 120:
                retry_after = val
                break
        except ValueError:
            pass

    if "429" in str(exc) or "rate limit" in msg or "too many requests" in msg:
        return True, retry_after or 5.0
    if any(str(code) in str(exc) for code in (500, 502, 503, 504)):
        return True, retry_after or 2.0
    return False, 0.0


def _format_tool_error(exc: Exception, model: str) -> str:
    """Format a tool use error with helpful guidance."""
    msg = str(exc)
    error_msg = f"LLM call failed: {msg}"

    # Check for model not found (404) error
    if "model_not_found" in msg.lower() or "does not exist" in msg.lower():
        groq_recommendation = (
            "\n\nMODEL NOT FOUND (404):\n"
            "The model is not available in your account or region.\n\n"
            "PRODUCTION MODELS WITH TOOL SUPPORT (as of Apr 2026):\n"
            "1. llama-3.3-70b-versatile (recommended - production, good tool use)\n"
            "2. llama-3.1-8b-instant (smaller, faster)\n"
            "3. openai/gpt-oss-120b (high quality)\n\n"
            "Update in .env: LLM_MODEL=llama-3.3-70b-versatile\n\n"
            "Check available models at: https://console.groq.com/docs/models"
        )
        return error_msg + groq_recommendation

    # Check for model decommissioned error
    if "decommissioned" in msg.lower():
        groq_recommendation = (
            "\n\nMODEL DECOMMISSIONED:\n"
            "Your Groq model is no longer available.\n\n"
            "PRODUCTION MODELS WITH TOOL SUPPORT (as of Apr 2026):\n"
            "1. llama-3.3-70b-versatile (recommended - latest, good tool use)\n"
            "2. llama-3.1-8b-instant (smaller, faster)\n"
            "3. openai/gpt-oss-120b (high quality)\n\n"
            "Update in .env: LLM_MODEL=llama-3.3-70b-versatile\n"
            "https://console.groq.com/docs/models"
        )
        return error_msg + groq_recommendation

    # Check for Groq tool_use_failed error
    if "tool_use_failed" in msg:
        groq_recommendation = (
            "\n\nTOOL CALL GENERATION FAILED:\n"
            "The model has trouble with tool formatting. Try:\n"
            "1. openai/gpt-oss-120b (best for tool use - recommended)\n"
            "2. openai/gpt-oss-20b (alternative with strong tool support)\n"
            "3. llama-3.1-8b-instant (if OSS models unavailable)\n\n"
            "Update in .env: LLM_MODEL=openai/gpt-oss-120b\n"
            "See: https://console.groq.com/docs/tool-use/overview for supported models"
        )
        return error_msg + groq_recommendation

    return error_msg


class OpenAIAdapter(LLMProvider):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        request_timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self._model = model
        self._max_retries = max_retries
        init_params: dict[str, Any] = {
            "openai_api_key": api_key,
            "model_name": model,
            "request_timeout": request_timeout,
            "max_retries": max_retries,
        }
        if base_url:
            init_params["openai_api_base"] = base_url
        self._client = ChatOpenAI(**init_params)  # type: ignore[call-arg]
        _logger.info(
            "openai_adapter: initialized",
            extra={
                "model": model,
                "base_url": base_url or "default",
                "request_timeout": request_timeout,
                "max_retries": max_retries,
            },
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        runnable = self._client.bind(tools=tools) if tools else self._client
        lc_messages = self._to_langchain_messages(messages)

        _logger.debug(
            "llm.chat: starting call",
            extra={
                "model": self._model,
                "message_count": len(lc_messages),
                "has_tools": bool(tools),
            },
        )

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await runnable.ainvoke(lc_messages)
                _logger.debug("llm.chat: call succeeded", extra={"attempt": attempt + 1})
                break
            except Exception as exc:
                last_exc = exc
                retryable, hint = _is_retryable(exc)
                if not retryable or attempt == self._max_retries:
                    _logger.error(
                        "llm.chat: non-retryable or final attempt",
                        extra={
                            "attempt": attempt + 1,
                            "retryable": retryable,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        },
                    )
                    formatted_error = _format_tool_error(exc, self._model)
                    raise LLMUnavailableError(formatted_error) from exc
                delay = min(_RETRY_BASE_DELAY * (2**attempt), _RETRY_MAX_DELAY)
                wait = max(delay, hint)
                _logger.warning(
                    "llm.chat: retryable error, backing off",
                    extra={"attempt": attempt + 1, "wait_seconds": wait, "error": str(exc)},
                )
                await asyncio.sleep(wait)
        else:
            raise LLMUnavailableError("LLM call failed after all retries") from last_exc

        tool_calls = [self._serialize_tool_call(tc) for tc in response.tool_calls]
        return LLMResponse(
            content=response.text if hasattr(response, "text") else str(response.content or ""),
            tool_calls=tool_calls,
            raw=response.model_dump(),
        )

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        """Yield content tokens one by one as the LLM generates them."""
        lc_messages = self._to_langchain_messages(messages)

        for attempt in range(self._max_retries + 1):
            try:
                async for chunk in self._client.astream(lc_messages):
                    content = chunk.content
                    if content:
                        yield str(content)
                return  # success
            except Exception as exc:
                retryable, hint = _is_retryable(exc)
                if not retryable or attempt == self._max_retries:
                    formatted_error = _format_tool_error(exc, self._model)
                    raise LLMUnavailableError(formatted_error) from exc
                delay = min(_RETRY_BASE_DELAY * (2**attempt), _RETRY_MAX_DELAY)
                wait = max(delay, hint)
                _logger.warning(
                    "llm.chat_stream: retryable error, backing off",
                    extra={"attempt": attempt + 1, "wait_seconds": wait, "error": str(exc)},
                )
                await asyncio.sleep(wait)

    def _to_langchain_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[SystemMessage | HumanMessage | AIMessage | ToolMessage]:
        converted: list[SystemMessage | HumanMessage | AIMessage | ToolMessage] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content", "") or ""
            if role == "system":
                converted.append(SystemMessage(content=content))
            elif role == "assistant":
                tool_calls_raw = message.get("tool_calls", [])
                if tool_calls_raw:
                    lc_tool_calls = []
                    for tc in tool_calls_raw:
                        raw_args = tc["function"].get("arguments", "{}")
                        try:
                            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                        except (json.JSONDecodeError, TypeError, ValueError):
                            args = {}
                        lc_tool_calls.append(
                            {
                                "name": tc["function"]["name"],
                                "args": args,
                                "id": tc.get("id", ""),
                                "type": "tool_call",
                            }
                        )
                    converted.append(AIMessage(content=content, tool_calls=lc_tool_calls))
                else:
                    converted.append(AIMessage(content=content))
            elif role == "tool":
                converted.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=message.get("tool_call_id", ""),
                    )
                )
            else:
                converted.append(HumanMessage(content=content))
        return converted

    def _serialize_tool_call(self, tool_call: Any) -> dict[str, Any]:
        """Serialize a LangChain ToolCall to OpenAI tool_call dict format."""
        arguments = tool_call.args if hasattr(tool_call, "args") else tool_call.get("args", {})
        tool_id = tool_call.id if hasattr(tool_call, "id") else tool_call.get("id", "")
        tool_name = tool_call.name if hasattr(tool_call, "name") else tool_call.get("name", "")
        return {
            "id": tool_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(arguments),
            },
        }
