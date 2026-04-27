from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from reportlab.pdfgen import canvas

from src.ingestion.chunker import Chunker
from src.ingestion.embedder import Embedder
from src.ingestion.pipeline import IngestionPipeline
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


@pytest.mark.asyncio
async def test_retriever_returns_ranked_chunks_from_indexed_corpus(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    create_pdf(
        corpus_dir / "doc_001.pdf",
        "National Insurance Co. versus Swaran Singh\nSUPREME COURT OF INDIA\n"
        "The insurer remained liable despite the licence breach.",
    )
    create_pdf(
        corpus_dir / "doc_002.pdf",
        "Transport Company versus Claimant\nHIGH COURT OF DELHI\n"
        "The commercial vehicle owner contested compensation and negligence.",
    )

    engine = create_db_engine(tmp_path / "casey.db")
    await init_schema(engine)
    session_factory = make_session_factory(engine)
    vector_store = VectorStore(path=tmp_path / "qdrant")
    await vector_store.init_collection()
    pipeline = IngestionPipeline(
        document_repository=DocumentRepository(session_factory),
        chunk_repository=ChunkRepository(session_factory),
        ingestion_run_repository=IngestionRunRepository(session_factory),
        ingestion_failure_repository=IngestionFailureRepository(session_factory),
        vector_store=vector_store,
        embedder=Embedder(),
        chunker=Chunker(),
    )
    run_id = str(uuid4())
    ingestion_runs = IngestionRunRepository(session_factory)
    await ingestion_runs.create(
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
    results = await retriever.retrieve("unlicensed driver insurance liability", n=5)

    assert results
    assert all(result.document_id for result in results)
    assert all(result.rrf_score > 0 for result in results)
    assert [result.rrf_score for result in results] == sorted(
        [result.rrf_score for result in results],
        reverse=True,
    )
    assert await retriever.retrieve("   ", n=5) == []

    await engine.dispose()
