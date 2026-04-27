from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from reportlab.pdfgen import canvas

from src.core.config import get_settings
from src.core.exceptions import LLMUnavailableError
from src.core.runtime import ensure_runtime
from src.llm.base import LLMResponse
from src.main import create_app


def create_pdf(path: Path, text: str) -> None:
    pdf = canvas.Canvas(str(path))
    text_object = pdf.beginText(40, 800)
    for line in text.splitlines():
        text_object.textLine(line)
    pdf.drawText(text_object)
    pdf.save()


class SearchingLLM:
    """
    Issues two search_corpus tool calls on the first tool-loop round, then stops.
    Returns 'research' for non-tool calls (classifier / synthesis / stream).
    This ensures the agent has retrieved chunks and takes the precedent_research path.
    """

    def __init__(self) -> None:
        self._tool_loop_calls = 0

    async def chat(self, messages, tools=None):
        if not tools:
            return LLMResponse(content="research", tool_calls=[], raw={})
        self._tool_loop_calls += 1
        if self._tool_loop_calls == 1:
            return LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "tc-1",
                        "type": "function",
                        "function": {
                            "name": "search_corpus",
                            "arguments": '{"query": "insurer liability unlicensed driver"}',
                        },
                    },
                ],
                raw={},
            )
        return LLMResponse(content="", tool_calls=[], raw={})


class PassiveLLM:
    """Returns no tool calls and empty content — agent finds no chunks → general_query."""

    async def chat(self, messages, tools=None):
        return LLMResponse(
            content="",
            tool_calls=[],
            raw={"messages": messages, "tools": tools or []},
        )


class UnavailableLLM:
    async def chat(self, messages, tools=None):
        raise LLMUnavailableError("provider unavailable")


@pytest.fixture
async def indexed_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    create_pdf(
        corpus_dir / "doc_001.pdf",
        "National Insurance Co. versus Swaran Singh\nSUPREME COURT OF INDIA\n15 March 2018\n"
        "The insurer remained liable to third-party claimants despite a licence breach."
        "The court applied pay and recover principles in a motor accident claim.",
    )
    create_pdf(
        corpus_dir / "doc_002.pdf",
        "Transport Company versus Claimant\nHIGH COURT OF DELHI\n12 July 2019\n"
        "A commercial vehicle caused a fatal accident. The driver had no valid licence.",
    )

    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "casey.db"))
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("CORPUS_DIR", str(corpus_dir))
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    get_settings.cache_clear()

    app = create_app()
    runtime = await ensure_runtime(app)
    run_id = str(uuid4())
    await runtime.ingestion_run_repository.create(
        {
            "id": run_id,
            "corpus_dir": str(corpus_dir),
            "started_at": datetime.now(UTC).isoformat(),
            "status": "running",
        }
    )
    await runtime.ingestion_pipeline.run(str(corpus_dir), run_id)
    yield app
    await runtime.engine.dispose()
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_post_query_returns_precedent_research_shape(
    indexed_app,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core.runtime import ensure_runtime

    runtime = await ensure_runtime(indexed_app)
    monkeypatch.setattr(runtime, "llm_provider", SearchingLLM())

    async with AsyncClient(
        transport=ASGITransport(app=indexed_app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/v1/query",
            json={
                "query": (
                    "Client: Mrs. Lakshmi Devi. Her husband was killed in a road"
                    " accident involving a commercial truck. The driver had no valid"
                    " licence. What precedents support our case?"
                )
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query_type"] == "precedent_research"
    assert isinstance(payload["response"]["supporting_precedents"], list)
    assert isinstance(payload["response"]["adverse_precedents"], list)
    assert set(payload["response"]["strategy_recommendation"].keys()) == {
        "priority_arguments",
        "compensation_range",
        "risks",
    }


@pytest.mark.asyncio
async def test_post_query_returns_general_query_shape(
    indexed_app,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core.runtime import ensure_runtime

    runtime = await ensure_runtime(indexed_app)
    monkeypatch.setattr(runtime, "llm_provider", PassiveLLM())

    async with AsyncClient(
        transport=ASGITransport(app=indexed_app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/v1/query",
            json={"query": "Which of these judgments involve commercial vehicles?"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query_type"] == "general_query"
    assert isinstance(payload["response"]["answer"], str)
    assert isinstance(payload["response"]["supporting_documents"], list)


@pytest.mark.asyncio
async def test_post_query_returns_503_when_llm_is_unavailable(
    indexed_app,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core.runtime import ensure_runtime

    runtime = await ensure_runtime(indexed_app)
    monkeypatch.setattr(runtime, "llm_provider", UnavailableLLM())

    async with AsyncClient(
        transport=ASGITransport(app=indexed_app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/api/v1/query",
            json={"query": "Find precedents for insurer liability"},
        )

    assert response.status_code == 503
    payload = response.json()
    assert payload["error_code"] == "LLM_UNAVAILABLE"
    assert isinstance(payload["message"], str)
