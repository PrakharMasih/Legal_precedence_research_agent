from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from reportlab.pdfgen import canvas
from sqlalchemy import text

from src.ingestion.chunker import Chunker
from src.ingestion.embedder import Embedder
from src.ingestion.pipeline import IngestionPipeline
from src.models.document import Document
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
async def test_ingestion_pipeline_processes_pdfs_and_records_failures(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    create_pdf(
        corpus_dir / "doc_001.pdf",
        "Lakshmi Devi versus National Insurance Co.\nSUPREME COURT OF INDIA\n15 March 2018\n"
        "The insurer denied liability because the truck driver had no valid licence.",
    )
    create_pdf(
        corpus_dir / "doc_002.pdf",
        "Transport Company versus Claimant\nHIGH COURT OF DELHI\n12 July 2019\n"
        "The commercial vehicle owner disputed the multiplier used for compensation.",
    )
    create_pdf(
        corpus_dir / "doc_003.pdf",
        "Claimant versus Insurer\nHIGH COURT OF MADRAS\n11 January 2020\n"
        "The court applied pay and recover after finding a licence breach.",
    )
    (corpus_dir / "doc_bad.pdf").write_bytes(b"not a real pdf")

    database_path = tmp_path / "casey.db"
    engine = create_db_engine(database_path)
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

    assert await document_repository.count() == 3
    first_document = (await document_repository.list_all(limit=1))[0]
    assert await chunk_repository.count_for_document(first_document.id) > 0
    assert await vector_store.count() > 0

    async with session_factory() as _session:
        fts_result = await _session.execute(
            text("SELECT COUNT(*) AS total FROM chunks_fts WHERE chunks_fts MATCH :q"),
            {"q": "licence"},
        )
        fts_match_count = int(fts_result.scalar() or 0)
    assert fts_match_count > 0

    failures = await ingestion_failure_repository.get_by_run_id(run_id)
    assert len(failures) == 1
    assert failures[0]["file_name"] == "doc_bad.pdf"

    await pipeline.run(str(corpus_dir), run_id)
    assert await document_repository.count() == 3

    await engine.dispose()


@pytest.mark.asyncio
async def test_ingestion_pipeline_reprocesses_incomplete_success_documents(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    create_pdf(
        corpus_dir / "doc_001.pdf",
        "Lakshmi Devi versus National Insurance Co.\nSUPREME COURT OF INDIA\n15 March 2018\n"
        "The insurer denied liability because the truck driver had no valid licence.",
    )

    database_path = tmp_path / "casey.db"
    engine = create_db_engine(database_path)
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

    broken_document = Document(
        id="f" * 64,
        file_name="doc_001.pdf",
        case_name="Broken import row",
        court_name="SUPREME COURT OF INDIA",
        judgment_date="15 March 2018",
        page_count=1,
        char_count=100,
        ingested_at=datetime.now(UTC),
        status="success",
    )
    await document_repository.insert(broken_document)

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

    assert await document_repository.count() == 1
    repaired_document = await document_repository.get_by_filename("doc_001.pdf")
    assert repaired_document is not None
    assert repaired_document.id != broken_document.id
    assert await chunk_repository.count_for_document(repaired_document.id) > 0

    await engine.dispose()
