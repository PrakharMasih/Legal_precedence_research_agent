from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from reportlab.pdfgen import canvas

from src.core.config import get_settings
from src.core.runtime import ensure_runtime
from src.main import create_app


def create_pdf(path: Path, text: str) -> None:
    pdf = canvas.Canvas(str(path))
    text_object = pdf.beginText(40, 800)
    for line in text.splitlines():
        text_object.textLine(line)
    pdf.drawText(text_object)
    pdf.save()


@pytest.fixture
async def indexed_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    create_pdf(
        corpus_dir / "doc_001.pdf",
        "National Insurance Co. versus Swaran Singh\nSUPREME COURT OF INDIA\n15 March 2018\n"
        "The insurer remained liable to third-party claimants despite a licence breach.",
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
async def test_get_documents_returns_paginated_document_summaries(indexed_app) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=indexed_app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/v1/documents", params={"page": 1, "page_size": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 2
    assert payload["page"] == 1
    assert payload["page_size"] == 1
    assert len(payload["documents"]) == 1
    assert set(payload["documents"][0].keys()) >= {
        "document_id",
        "file_name",
        "case_name",
        "court_name",
        "judgment_date",
        "page_count",
        "chunk_count",
        "ingested_at",
    }
