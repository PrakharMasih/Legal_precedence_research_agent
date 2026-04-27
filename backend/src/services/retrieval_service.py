"""Service layer for document retrieval operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.models.document import Chunk

if TYPE_CHECKING:
    from src.storage.repositories import ChunkRepository, DocumentRepository


class RetrievalService:
    """
    Service for retrieving documents and chunks.

    Provides a business logic layer above document/chunk repositories.
    """

    def __init__(
        self,
        *,
        document_repository: DocumentRepository,
        chunk_repository: ChunkRepository,
    ) -> None:
        """
        Initialize RetrievalService.

        Args:
            document_repository: Repository for document retrieval.
            chunk_repository: Repository for chunk retrieval.
        """
        self._document_repository = document_repository
        self._chunk_repository = chunk_repository

    async def list_documents(self, page: int = 1, page_size: int = 50) -> tuple[list, int]:
        """
        List documents in the corpus.

        Args:
            page: Page number (1-indexed).
            page_size: Number of documents per page.

        Returns:
            Tuple of (documents, total_count).
        """
        offset = max(page - 1, 0) * page_size
        documents = await self._document_repository.list_all(limit=page_size, offset=offset)
        total = await self._document_repository.count()
        return documents, total

    async def get_chunks_for_document(self, document_id: str) -> list[Chunk]:
        """
        Get all chunks for a document.

        Args:
            document_id: ID of the document.

        Returns:
            List of chunks.
        """
        return await self._chunk_repository.get_all_for_document(document_id)

    async def count_chunks_for_document(self, document_id: str) -> int:
        """
        Count chunks in a document.

        Args:
            document_id: ID of the document.

        Returns:
            Number of chunks.
        """
        return await self._chunk_repository.count_for_document(document_id)
