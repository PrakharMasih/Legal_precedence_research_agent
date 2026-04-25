from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from src.core.logging import get_logger
from src.ingestion.chunker import Chunker
from src.ingestion.embedder import Embedder
from src.ingestion.parser import parse_pdf
from src.models.document import Chunk, Document
from src.storage.repositories import (
    ChunkRepository,
    DocumentRepository,
    IngestionFailureRecord,
    IngestionFailureRepository,
    IngestionRunRepository,
)
from src.storage.vector_store import VectorStore

logger = get_logger(component="ingestion")


@dataclass(slots=True)
class IngestionPipeline:
    document_repository: DocumentRepository
    chunk_repository: ChunkRepository
    ingestion_run_repository: IngestionRunRepository
    ingestion_failure_repository: IngestionFailureRepository
    vector_store: VectorStore
    embedder: Embedder
    chunker: Chunker

    async def run(self, corpus_dir: str, run_id: str) -> None:
        corpus_path = Path(corpus_dir)
        files = sorted(corpus_path.glob("*.pdf"))
        logger.info(
            "ingestion.run_started",
            run_id=run_id,
            corpus_dir=corpus_dir,
            total_files=len(files),
        )
        await self.ingestion_run_repository.update(
            run_id,
            {"total_files": len(files), "status": "running"},
        )

        succeeded = 0
        failed = 0
        total_chunks = 0

        for file_path in files:
            existing = None
            try:
                parsed = await parse_pdf(file_path)
                existing = await self.document_repository.get_by_id(parsed.file_hash)
                if existing is None:
                    existing = await self.document_repository.get_by_filename(parsed.file_name)

                existing_chunk_count = 0
                if existing is not None:
                    existing_chunk_count = await self.chunk_repository.count_for_document(
                        parsed.file_hash
                    )
                    if existing.id != parsed.file_hash:
                        existing_chunk_count = await self.chunk_repository.count_for_document(
                            existing.id
                        )

                if existing is not None and existing.id != parsed.file_hash:
                    logger.warning(
                        "ingestion.document_replacing_stale_filename_match",
                        run_id=run_id,
                        file_name=parsed.file_name,
                        stale_document_id=existing.id,
                        document_id=parsed.file_hash,
                    )
                    await self.vector_store.delete_by_document_id(existing.id)
                    await self.document_repository.delete_by_id(existing.id)
                    existing = None
                    existing_chunk_count = 0

                if (
                    existing is not None
                    and existing.status == "success"
                    and existing_chunk_count > 0
                ):
                    succeeded += 1
                    logger.info(
                        "ingestion.document_skipped_existing",
                        run_id=run_id,
                        file_name=parsed.file_name,
                        document_id=parsed.file_hash,
                    )
                    continue

                if existing is not None and existing_chunk_count == 0:
                    logger.warning(
                        "ingestion.document_reprocessing_incomplete",
                        run_id=run_id,
                        file_name=parsed.file_name,
                        document_id=parsed.file_hash,
                        status=existing.status,
                    )

                document = Document(
                    id=parsed.file_hash,
                    file_name=parsed.file_name,
                    case_name=parsed.case_name,
                    court_name=parsed.court_name,
                    judgment_date=parsed.judgment_date,
                    page_count=parsed.page_count,
                    char_count=parsed.char_count,
                    ingested_at=datetime.now(UTC),
                    status="success",
                )

                chunk_slices = await self.chunker.chunk_text(parsed.raw_text)
                if not chunk_slices:
                    raise ValueError("No chunks were produced from readable text")

                # ── Separate parents (context blocks) from children (retrieval units) ──
                parent_slices = [s for s in chunk_slices if s.chunk_type == "parent"]
                child_slices = [s for s in chunk_slices if s.chunk_type == "child"]

                if not child_slices:
                    raise ValueError("Chunker produced no child chunks")

                # Only embed child chunks (precise retrieval units)
                embeddings = await self.embedder.embed_batch([s.content for s in child_slices])

                now = datetime.now(UTC)

                # Create parent Chunk objects (stored in SQLite, NOT in vector store)
                parent_chunks = [
                    Chunk(
                        id=str(uuid4()),
                        document_id=parsed.file_hash,
                        content=s.content,
                        char_start=s.char_start,
                        char_end=s.char_end,
                        chunk_index=i,
                        embedded_at=now,
                        section=s.section,
                        chunk_type="parent",
                        parent_id=None,
                    )
                    for i, s in enumerate(parent_slices)
                ]
                parent_id_by_index = {i: chunk.id for i, chunk in enumerate(parent_chunks)}

                # Create child Chunk objects (stored in SQLite AND vector store)
                child_chunks = [
                    Chunk(
                        id=str(uuid4()),
                        document_id=parsed.file_hash,
                        content=s.content,
                        char_start=s.char_start,
                        char_end=s.char_end,
                        chunk_index=len(parent_chunks) + i,
                        embedded_at=now,
                        section=s.section,
                        chunk_type="child",
                        parent_id=(
                            parent_id_by_index.get(s.parent_index)
                            if s.parent_index is not None
                            else None
                        ),
                    )
                    for i, s in enumerate(child_slices)
                ]

                all_chunks = parent_chunks + child_chunks

                if existing is None:
                    await self.document_repository.insert(document)

                await self.vector_store.delete_by_document_id(parsed.file_hash)
                await self.chunk_repository.delete_by_document_id(parsed.file_hash)
                await self.chunk_repository.insert_batch(all_chunks)

                # Vector-index only child chunks (with section + hierarchy metadata)
                for child, embedding in zip(child_chunks, embeddings, strict=True):
                    await self.vector_store.add_embeddings(
                        child.id,
                        embedding,
                        {
                            "document_id": parsed.file_hash,
                            "file_name": parsed.file_name,
                            "case_name": parsed.case_name or "",
                            "chunk_index": child.chunk_index,
                            "char_start": child.char_start,
                            "char_end": child.char_end,
                            "section": child.section,
                            "chunk_type": "child",
                            "parent_id": child.parent_id or "",
                        },
                        child.content,
                    )

                if existing is not None and existing.status != "success":
                    await self.document_repository.update_status(parsed.file_hash, "success")

                succeeded += 1
                total_chunks += len(child_chunks)
                logger.info(
                    "ingestion.document_processed",
                    run_id=run_id,
                    file_name=parsed.file_name,
                    document_id=parsed.file_hash,
                    parent_chunks=len(parent_chunks),
                    child_chunks=len(child_chunks),
                )
            except Exception as exc:
                failed += 1
                await self.ingestion_failure_repository.insert(
                    run_id,
                    IngestionFailureRecord(file_name=file_path.name, error_message=str(exc)),
                )
                if existing is not None:
                    await self.document_repository.update_status(parsed.file_hash, "failed")
                logger.error(
                    "ingestion.document_failed",
                    run_id=run_id,
                    file_name=file_path.name,
                    error=str(exc),
                )

        final_status = "complete" if failed == 0 else "failed"
        await self.ingestion_run_repository.update(
            run_id,
            {
                "completed_at": datetime.now(UTC).isoformat(),
                "succeeded": succeeded,
                "failed": failed,
                "total_chunks": total_chunks,
                "status": final_status,
            },
        )
        logger.info(
            "ingestion.run_completed",
            run_id=run_id,
            corpus_dir=corpus_dir,
            status=final_status,
            succeeded=succeeded,
            failed=failed,
            total_chunks=total_chunks,
        )
