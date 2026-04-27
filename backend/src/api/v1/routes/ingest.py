"""Document ingestion endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.api.v1.schemas import IngestAccepted, IngestionReport, IngestRequest
from src.constants import (
    ERROR_CODE_INGESTION_IN_PROGRESS,
    ERROR_CODE_INGESTION_RUN_NOT_FOUND,
    ERROR_MSG_INGESTION_IN_PROGRESS,
    ERROR_MSG_INGESTION_RUN_NOT_FOUND,
    STATUS_RUNNING,
)
from src.core.logging import get_logger
from src.core.runtime import ensure_runtime
from src.utils.responses import build_error_response, get_correlation_id_from_headers

router = APIRouter(prefix="/ingest", tags=["ingest"])
logger = get_logger(component="api.ingest")


def _log_ingestion_task_completion(
    *,
    task: asyncio.Task[None],
    run_id: str,
    corpus_dir: str,
) -> None:
    """Log the result of an ingestion background task."""
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


@router.post("", response_model=IngestAccepted, status_code=202)
async def trigger_ingestion(
    request: Request,
    payload: IngestRequest,
) -> IngestAccepted | JSONResponse:
    """
    Trigger document ingestion.

    Starts an asynchronous ingestion run to process PDFs from the corpus directory.
    Returns immediately with a run ID for tracking progress.

    Args:
        request: FastAPI request object.
        payload: IngestRequest with optional corpus_dir override.

    Returns:
        IngestAccepted response with run_id and status.

    Raises:
        HTTP 409 if an ingestion run is already active.
    """
    runtime = await ensure_runtime(request.app)
    ingestion_service = runtime.ingestion_service
    correlation_id = get_correlation_id_from_headers(dict(request.headers))

    requested_corpus_dir = payload.corpus_dir or str(runtime.settings.corpus_dir)

    logger.info(
        "api.ingest.request_received",
        correlation_id=correlation_id,
        corpus_dir=requested_corpus_dir,
    )

    try:
        run_id, _ = await ingestion_service.start_ingestion(requested_corpus_dir)
    except ValueError as exc:
        logger.warning(
            "api.ingest.request_rejected_active_run",
            correlation_id=correlation_id,
            requested_corpus_dir=requested_corpus_dir,
            error=str(exc),
        )
        return build_error_response(
            correlation_id=correlation_id,
            status_code=409,
            error_code=ERROR_CODE_INGESTION_IN_PROGRESS,
            message=ERROR_MSG_INGESTION_IN_PROGRESS,
        )

    # Schedule background ingestion task
    task = asyncio.create_task(ingestion_service.execute_ingestion(requested_corpus_dir, run_id))
    request.app.state.ingestion_tasks[run_id] = task

    logger.info(
        "api.ingest.run_accepted",
        correlation_id=correlation_id,
        run_id=run_id,
        corpus_dir=requested_corpus_dir,
    )

    def _cleanup(background_task: asyncio.Task[None]) -> None:
        """Clean up after background task completion."""
        request.app.state.ingestion_tasks.pop(run_id, None)
        _log_ingestion_task_completion(
            task=background_task,
            run_id=run_id,
            corpus_dir=requested_corpus_dir,
        )

    task.add_done_callback(_cleanup)

    return IngestAccepted(
        correlation_id=correlation_id,
        run_id=run_id,
        status=STATUS_RUNNING,
        message=f"Ingestion started for {requested_corpus_dir}",
    )


@router.get("/{run_id}", response_model=IngestionReport)
async def get_ingestion_status(run_id: str, request: Request) -> IngestionReport | JSONResponse:
    """
    Get status of an ingestion run.

    Returns the current status, progress, and any failures for the ingestion run.

    Args:
        request: FastAPI request object.
        run_id: ID of the ingestion run.

    Returns:
        IngestionReport with status and progress details.

    Raises:
        HTTP 404 if run_id is not found.
    """
    runtime = await ensure_runtime(request.app)
    ingestion_service = runtime.ingestion_service
    correlation_id = get_correlation_id_from_headers(dict(request.headers))

    logger.info(
        "api.ingest.status_requested",
        correlation_id=correlation_id,
        run_id=run_id,
    )

    run = await ingestion_service.get_run_status(run_id)
    if run is None:
        logger.warning(
            "api.ingest.status_not_found",
            correlation_id=correlation_id,
            run_id=run_id,
        )
        return build_error_response(
            correlation_id=correlation_id,
            status_code=404,
            error_code=ERROR_CODE_INGESTION_RUN_NOT_FOUND,
            message=ERROR_MSG_INGESTION_RUN_NOT_FOUND.format(run_id=run_id),
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
