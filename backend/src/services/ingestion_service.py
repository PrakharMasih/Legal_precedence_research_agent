"""Service layer for document ingestion operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from src.core.logging import get_logger
from src.utils.timestamps import timestamp_iso

if TYPE_CHECKING:
    from src.ingestion.pipeline import IngestionPipeline
    from src.storage.repositories import IngestionRunRepository

logger = get_logger(component="services.ingestion")


class IngestionService:
    """
    Service for managing document ingestion operations.

    Orchestrates:
    - Ingestion run lifecycle
    - Status tracking
    - Error handling
    """

    def __init__(
        self,
        *,
        ingestion_pipeline: IngestionPipeline,
        ingestion_run_repository: IngestionRunRepository,
    ) -> None:
        """
        Initialize IngestionService.

        Args:
            ingestion_pipeline: Pipeline for document processing.
            ingestion_run_repository: Repository for tracking ingestion runs.
        """
        self._pipeline = ingestion_pipeline
        self._run_repository = ingestion_run_repository

    async def check_active_run(self) -> dict[str, Any] | None:
        """
        Check if an ingestion run is currently active.

        Returns:
            Active run dict if one exists, None otherwise.
        """
        return await self._run_repository.get_active_run()

    async def start_ingestion(self, corpus_dir: str) -> tuple[str, dict[str, Any]]:
        """
        Start a new ingestion run.

        Args:
            corpus_dir: Directory path containing PDF files to ingest.

        Returns:
            Tuple of (run_id, run_record).

        Raises:
            ValueError: If an ingestion run is already active.
        """
        active_run = await self.check_active_run()
        if active_run is not None:
            raise ValueError(
                f"Ingestion already in progress (run_id={active_run['id']}). "
                "Wait for it to complete or check its status."
            )

        run_id = str(uuid4())
        run_record = {
            "id": run_id,
            "corpus_dir": corpus_dir,
            "started_at": timestamp_iso(),
            "status": "running",
            "total_files": 0,
            "succeeded": 0,
            "failed": 0,
            "total_chunks": 0,
            "completed_at": None,
        }

        await self._run_repository.create(run_record)
        logger.info(
            "ingestion_service.run_started",
            run_id=run_id,
            corpus_dir=corpus_dir,
        )

        return run_id, run_record

    async def get_run_status(self, run_id: str) -> dict[str, Any] | None:
        """
        Get status of an ingestion run.

        Args:
            run_id: ID of the ingestion run.

        Returns:
            Run record if found, None otherwise.
        """
        return await self._run_repository.get_by_id(run_id)

    async def execute_ingestion(self, corpus_dir: str, run_id: str) -> None:
        """
        Execute the ingestion pipeline.

        This is typically run as a background task.

        Args:
            corpus_dir: Directory containing PDFs to ingest.
            run_id: ID of the ingestion run.
        """
        logger.info(
            "ingestion_service.executing_pipeline",
            run_id=run_id,
            corpus_dir=corpus_dir,
        )
        await self._pipeline.run(corpus_dir, run_id)
