"""Document listing endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from src.api.v1.schemas import DocumentListResponse, DocumentSummary
from src.constants import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, MIN_PAGE_SIZE
from src.core.runtime import ensure_runtime
from src.utils.responses import get_correlation_id_from_headers

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=MIN_PAGE_SIZE, le=MAX_PAGE_SIZE),
) -> DocumentListResponse:
    """
    List ingested documents.

    Returns paginated list of all documents in the corpus.

    Args:
        request: FastAPI request object.
        page: Page number (1-indexed).
        page_size: Number of documents per page.

    Returns:
        DocumentListResponse with paginated documents and metadata.
    """
    runtime = await ensure_runtime(request.app)
    retrieval_service = runtime.retrieval_service
    correlation_id = get_correlation_id_from_headers(dict(request.headers))

    documents, total = await retrieval_service.list_documents(page=page, page_size=page_size)

    summaries = []
    for document in documents:
        chunk_count = await retrieval_service.count_chunks_for_document(document.id)
        summaries.append(
            DocumentSummary(
                document_id=document.id,
                file_name=document.file_name,
                case_name=document.case_name,
                court_name=document.court_name,
                judgment_date=document.judgment_date,
                page_count=document.page_count,
                chunk_count=chunk_count,
                ingested_at=document.ingested_at.isoformat(),
            )
        )

    return DocumentListResponse(
        correlation_id=correlation_id,
        total=total,
        page=page,
        page_size=page_size,
        documents=summaries,
    )
