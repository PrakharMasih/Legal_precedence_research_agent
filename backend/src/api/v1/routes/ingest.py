from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.api.v1.schemas import ErrorResponse, IngestAccepted, IngestionReport, IngestRequest
from src.core.logging import get_logger
from src.core.runtime import ensure_runtime

router = APIRouter(prefix="/ingest", tags=["ingest"])
logger = get_logger(component="api.ingest")


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _log_ingestion_task_completion(
    *,
    task: asyncio.Task[None],
    run_id: str,
    corpus_dir: str,
) -> None:
    if task.cancelled():
        logger.warning(
            "api.ingest.background_task_cancelled",
            run_id=run_id,
            corpus_dir=corpus_dir,
        )
        return

    exception = task.exception()
    if exception is not None:
        logger.exception(
            "api.ingest.background_task_failed",
            run_id=run_id,
            corpus_dir=corpus_dir,
            error=str(exception),
            exc_info=exception,
        )
        return

    logger.info(
        "api.ingest.background_task_completed",
        run_id=run_id,
        corpus_dir=corpus_dir,
    )


def _error_response(
    *,
    correlation_id: str,
    status_code: int,
    error_code: str,
    message: str,
) -> JSONResponse:
    payload = ErrorResponse(
        correlation_id=correlation_id,
        error_code=error_code,
        message=message,
        timestamp=_timestamp(),
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump())


@router.post("", response_model=IngestAccepted, status_code=202)
async def trigger_ingestion(
    request: Request,
    payload: IngestRequest,
) -> IngestAccepted | JSONResponse:
    runtime = await ensure_runtime(request.app)
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid4()))
    requested_corpus_dir = payload.corpus_dir or str(runtime.settings.corpus_dir)
    logger.info(
        "api.ingest.request_received",
        correlation_id=correlation_id,
        corpus_dir=requested_corpus_dir,
    )
    active_run = await runtime.ingestion_run_repository.get_active_run()
    if active_run is not None:
        logger.warning(
            "api.ingest.request_rejected_active_run",
            correlation_id=correlation_id,
            requested_corpus_dir=requested_corpus_dir,
            active_run_id=active_run["id"],
        )
        return _error_response(
            correlation_id=correlation_id,
            status_code=409,
            error_code="INGESTION_IN_PROGRESS",
            message=(
                "An ingestion run is already running. Check the existing run status"
                " before starting another."
            ),
        )

    run_id = str(uuid4())
    corpus_dir = requested_corpus_dir
    await runtime.ingestion_run_repository.create(
        {
            "id": run_id,
            "corpus_dir": corpus_dir,
            "started_at": _timestamp(),
            "status": "running",
        }
    )
    task = asyncio.create_task(runtime.ingestion_pipeline.run(corpus_dir, run_id))
    request.app.state.ingestion_tasks[run_id] = task
    logger.info(
        "api.ingest.run_accepted",
        correlation_id=correlation_id,
        run_id=run_id,
        corpus_dir=corpus_dir,
    )

    def _cleanup(background_task: asyncio.Task[None]) -> None:
        request.app.state.ingestion_tasks.pop(run_id, None)
        _log_ingestion_task_completion(
            task=background_task,
            run_id=run_id,
            corpus_dir=corpus_dir,
        )

    task.add_done_callback(_cleanup)
    return IngestAccepted(
        correlation_id=correlation_id,
        run_id=run_id,
        status="running",
        message=f"Ingestion started for {corpus_dir}",
    )


@router.get("/{run_id}", response_model=IngestionReport)
async def get_ingestion_status(run_id: str, request: Request) -> IngestionReport | JSONResponse:
    runtime = await ensure_runtime(request.app)
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid4()))
    logger.info(
        "api.ingest.status_requested",
        correlation_id=correlation_id,
        run_id=run_id,
    )
    run = await runtime.ingestion_run_repository.get_by_id(run_id)
    if run is None:
        logger.warning(
            "api.ingest.status_not_found",
            correlation_id=correlation_id,
            run_id=run_id,
        )
        return _error_response(
            correlation_id=correlation_id,
            status_code=404,
            error_code="INGESTION_RUN_NOT_FOUND",
            message=f"No ingestion run found for run_id={run_id}",
        )

    failures = await runtime.ingestion_failure_repository.get_by_run_id(run_id)
    logger.info(
        "api.ingest.status_returned",
        correlation_id=correlation_id,
        run_id=run_id,
        status=run["status"],
        total_files=run["total_files"],
        succeeded=run["succeeded"],
        failed=run["failed"],
    )
    return IngestionReport(
        correlation_id=correlation_id,
        run_id=run["id"],
        status=run["status"],
        corpus_dir=run["corpus_dir"],
        total_files=run["total_files"],
        succeeded=run["succeeded"],
        failed=run["failed"],
        total_chunks=run["total_chunks"],
        failures=failures,
        started_at=run["started_at"],
        completed_at=run["completed_at"],
    )
