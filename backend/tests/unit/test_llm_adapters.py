from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from src.core.config import Settings
from src.core.exceptions import LLMUnavailableError
from src.llm.factory import GROQ_BASE_URL, LLMFactory
from src.llm.openai_adapter import OpenAIAdapter


class FakeChatOpenAI:
    def __init__(self, *args, **kwargs):
        self._response = kwargs.pop("response", None)
        self._tools = None

    def bind(self, **kwargs):
        self._tools = kwargs.get("tools")
        return self

    async def ainvoke(self, messages):
        return self._response


def make_completion_response(content: str = "ok"):
    return AIMessage(content=content)


@pytest.mark.asyncio
async def test_openai_adapter_returns_llm_response(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeChatOpenAI(response=make_completion_response("hello"))
    monkeypatch.setattr("src.llm.openai_adapter.ChatOpenAI", lambda **kwargs: fake_client)

    adapter = OpenAIAdapter(api_key="test", model="gpt-4o-mini")
    response = await adapter.chat(messages=[{"role": "user", "content": "hi"}])

    assert response.content == "hello"
    assert response.tool_calls == []
    assert response.raw["content"] == "hello"


@pytest.mark.asyncio
async def test_openai_adapter_raises_llm_unavailable_on_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ErrorClient:
        def bind(self, **kwargs):
            return self

        async def ainvoke(self, messages):
            raise RuntimeError("Connection refused")

    monkeypatch.setattr("src.llm.openai_adapter.ChatOpenAI", lambda **kwargs: ErrorClient())

    adapter = OpenAIAdapter(api_key="test", model="gpt-4o-mini")
    with pytest.raises(LLMUnavailableError):
        await adapter.chat(messages=[{"role": "user", "content": "hi"}])


def test_llm_factory_returns_openai_adapter() -> None:
    settings = Settings(
        LLM_PROVIDER="openai",
        LLM_MODEL="gpt-4o-mini",
        LLM_API_KEY="test-key",
        CORPUS_DIR="judgement_pdfs",
        SQLITE_DB_PATH="data/lexi.db",
    )

    adapter = LLMFactory.from_config(settings)

    assert isinstance(adapter, OpenAIAdapter)


def test_llm_factory_returns_groq_adapter() -> None:
    settings = Settings(
        LLM_PROVIDER="groq",
        LLM_MODEL="llama-3.3-70b-versatile",
        LLM_API_KEY="test-key",
        CORPUS_DIR="judgement_pdfs",
        SQLITE_DB_PATH="data/lexi.db",
    )

    adapter = LLMFactory.from_config(settings)

    assert isinstance(adapter, OpenAIAdapter)
    assert str(adapter._client.openai_api_base).rstrip("/") == GROQ_BASE_URL


def test_llm_factory_raises_for_unknown_provider() -> None:
    settings = Settings(
        LLM_PROVIDER="unknown",
        LLM_MODEL="gpt-4o-mini",
        LLM_API_KEY="test-key",
        CORPUS_DIR="judgement_pdfs",
        SQLITE_DB_PATH="data/lexi.db",
    )

    with pytest.raises(ValueError):
        LLMFactory.from_config(settings)
