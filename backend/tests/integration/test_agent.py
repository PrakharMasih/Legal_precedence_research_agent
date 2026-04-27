from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from reportlab.pdfgen import canvas

from src.agent.agent import LegalResearchAgent
from src.agent.tools import ResearchToolbox
from src.core.exceptions import LLMUnavailableError
from src.ingestion.chunker import Chunker
from src.ingestion.embedder import Embedder
from src.ingestion.pipeline import IngestionPipeline
from src.llm.base import LLMResponse
from src.retrieval.retriever import Retriever
from src.storage.database import create_db_engine, init_schema, make_session_factory
from src.storage.repositories import (
    ChunkRepository,
    DocumentRepository,
    IngestionFailureRepository,
    IngestionRunRepository,
)
from src.storage.vector_store import VectorStore


def create_pdf(path: Path, text: str) -> None:
    pdf = canvas.Canvas(str(path))
    text_object = pdf.beginText(40, 800)
    for line in text.splitlines():
        text_object.textLine(line)
    pdf.drawText(text_object)
    pdf.save()


class SearchingLLM:
    """
    On the first tool-loop call returns two search_corpus tool calls.
    On classify / synthesis calls (no tools list) returns 'research'.
    All subsequent tool-loop calls return empty to stop the agentic loop.
    """

    def __init__(self) -> None:
        self._tool_loop_calls = 0

    async def chat(self, messages, tools=None):
        # Non-tool calls: classify, synthesize, stream-fallback — return "research" so the
        # agent takes the precedent-research path (not general_query) for the first run.
        if not tools:
            return LLMResponse(content="research", tool_calls=[], raw={})
        # First tool-loop call — instruct two distinct searches
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
                    {
                        "id": "tc-2",
                        "type": "function",
                        "function": {
                            "name": "search_corpus",
                            "arguments": '{"query": "third party motor accident compensation"}',
                        },
                    },
                ],
                raw={},
            )
        # Subsequent tool-loop calls — no more tools, proceed to synthesis
        return LLMResponse(content="", tool_calls=[], raw={})


# PassiveLLM kept for the UnavailableLLM test below
class PassiveLLM:
    async def chat(self, messages, tools=None):
        return LLMResponse(content="", tool_calls=[], raw={})


class UnavailableLLM:
    async def chat(self, messages, tools=None):
        raise LLMUnavailableError("provider unavailable")


@pytest.mark.asyncio
async def test_agent_returns_research_and_general_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    create_pdf(
        corpus_dir / "doc_001.pdf",
        "National Insurance Co. versus Swaran Singh\nSUPREME COURT OF INDIA\n"
        "The insurer remained liable despite the licence breach.",
    )

    engine = create_db_engine(tmp_path / "casey.db")
    await init_schema(engine)
    session_factory = make_session_factory(engine)
    document_repository = DocumentRepository(session_factory)
    chunk_repository = ChunkRepository(session_factory)
    ingestion_run_repository = IngestionRunRepository(session_factory)
    ingestion_failure_repository = IngestionFailureRepository(session_factory)
    vector_store = VectorStore(path=tmp_path / "qdrant")
    await vector_store.init_collection()
    pipeline = IngestionPipeline(
        document_repository=document_repository,
        chunk_repository=chunk_repository,
        ingestion_run_repository=ingestion_run_repository,
        ingestion_failure_repository=ingestion_failure_repository,
        vector_store=vector_store,
        embedder=Embedder(),
        chunker=Chunker(),
    )
    run_id = str(uuid4())
    await ingestion_run_repository.create(
        {
            "id": run_id,
            "corpus_dir": str(corpus_dir),
            "started_at": datetime.now(UTC).isoformat(),
            "status": "running",
        }
    )
    await pipeline.run(str(corpus_dir), run_id)

    retriever = Retriever(
        session_factory=session_factory,
        vector_store=vector_store,
        embedder=Embedder(),
    )
    toolbox = ResearchToolbox(
        retriever=retriever,
        document_repository=document_repository,
        chunk_repository=chunk_repository,
    )
    search_call_count = 0
    original_search = ResearchToolbox.search_corpus

    async def counting_search(self, *args, **kwargs):
        nonlocal search_call_count
        search_call_count += 1
        return await original_search(self, *args, **kwargs)

    monkeypatch.setattr(ResearchToolbox, "search_corpus", counting_search)

    agent = LegalResearchAgent(llm_provider=SearchingLLM(), toolbox=toolbox)
    research_result = await agent.run(
        "Client: find precedents supporting insurer liability.",
        "corr-1",
    )
    general_result = await agent.run(
        "Which judgments involve commercial vehicles?", "corr-2", force_mode="general"
    )

    assert search_call_count >= 2
    assert research_result["query_type"] == "precedent_research"
    assert isinstance(research_result["response"]["supporting_precedents"], list)
    assert general_result["query_type"] == "general_query"
    assert isinstance(general_result["response"]["supporting_documents"], list)

    failing_agent = LegalResearchAgent(llm_provider=UnavailableLLM(), toolbox=toolbox)
    with pytest.raises(LLMUnavailableError):
        await failing_agent.run("Find precedents", "corr-3")

    await engine.dispose()
