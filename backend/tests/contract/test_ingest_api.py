from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from src.core.config import get_settings
from src.main import create_app


@pytest.fixture
def test_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "lexi.db"))
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("CORPUS_DIR", str((Path.cwd() / "judgement_pdfs").resolve()))
    get_settings.cache_clear()
    app = create_app()
    yield app
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_post_ingest_returns_accepted_with_run_id(test_app) -> None:
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        response = await client.post("/api/v1/ingest", json={"corpus_dir": "judgement_pdfs"})

    assert response.status_code == 202
    payload = response.json()
    assert isinstance(payload["correlation_id"], str)
    assert isinstance(payload["run_id"], str)
    assert payload["status"] == "running"
    assert isinstance(payload["message"], str)


@pytest.mark.asyncio
async def test_get_ingest_status_returns_report_shape(test_app) -> None:
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        post_response = await client.post("/api/v1/ingest", json={"corpus_dir": "judgement_pdfs"})
        run_id = post_response.json()["run_id"]
        response = await client.get(f"/api/v1/ingest/{run_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == run_id
    assert payload["status"] in {"running", "complete", "failed"}
    assert isinstance(payload["correlation_id"], str)
    assert isinstance(payload["corpus_dir"], str)
    assert isinstance(payload["total_files"], int)
    assert isinstance(payload["succeeded"], int)
    assert isinstance(payload["failed"], int)
    assert isinstance(payload["total_chunks"], int)
    assert isinstance(payload["failures"], list)
    datetime.fromisoformat(payload["started_at"].replace("Z", "+00:00"))


@pytest.mark.asyncio
async def test_post_ingest_returns_conflict_when_run_is_active(test_app) -> None:
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        first = await client.post("/api/v1/ingest", json={"corpus_dir": "judgement_pdfs"})
        second = await client.post("/api/v1/ingest", json={"corpus_dir": "judgement_pdfs"})

    assert first.status_code == 202
    assert second.status_code == 409
    payload = second.json()
    assert payload["error_code"] == "INGESTION_IN_PROGRESS"
    assert isinstance(payload["message"], str)
    datetime.fromisoformat(payload["timestamp"].replace("Z", "+00:00"))


@pytest.mark.asyncio
async def test_get_ingest_status_returns_404_for_unknown_run(test_app) -> None:
    unknown_run_id = "00000000-0000-0000-0000-000000000000"

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        response = await client.get(f"/api/v1/ingest/{unknown_run_id}")

    assert response.status_code == 404
    payload = response.json()
    assert payload["error_code"] == "INGESTION_RUN_NOT_FOUND"
    assert isinstance(payload["correlation_id"], str)
    assert isinstance(payload["message"], str)
    datetime.fromisoformat(payload["timestamp"].replace("Z", "+00:00"))
