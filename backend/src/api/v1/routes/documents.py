from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Request

from src.api.v1.schemas import DocumentListResponse, DocumentSummary
from src.core.runtime import ensure_runtime

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    request: Request,
    page: int = 1,
    page_size: int = 50,
) -> DocumentListResponse:
    runtime = await ensure_runtime(request.app)
    offset = max(page - 1, 0) * page_size
    documents = await runtime.document_repository.list_all(limit=page_size, offset=offset)
    summaries = []
    for document in documents:
        summaries.append(
            DocumentSummary(
                document_id=document.id,
                file_name=document.file_name,
                case_name=document.case_name,
                court_name=document.court_name,
                judgment_date=document.judgment_date,
                page_count=document.page_count,
                chunk_count=await runtime.chunk_repository.count_for_document(document.id),
                ingested_at=document.ingested_at.isoformat(),
            )
        )

    return DocumentListResponse(
        correlation_id=request.headers.get("X-Correlation-ID", str(uuid4())),
        total=await runtime.document_repository.count(),
        page=page,
        page_size=page_size,
        documents=summaries,
    )
