from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.agent.output_schemas import GeneralQueryResponse, PrecedentAnalysis


class HealthResponse(BaseModel):
    status: str = "ok"


class IngestRequest(BaseModel):
    corpus_dir: str | None = None


class IngestAccepted(BaseModel):
    correlation_id: str
    run_id: str
    status: str = "running"
    message: str


class IngestionFailureItem(BaseModel):
    file_name: str
    error_message: str


class IngestionReport(BaseModel):
    correlation_id: str
    run_id: str
    status: str
    corpus_dir: str
    total_files: int = 0
    succeeded: int = 0
    failed: int = 0
    total_chunks: int = 0
    failures: list[IngestionFailureItem] = Field(default_factory=list)
    started_at: str
    completed_at: str | None = None


class ErrorResponse(BaseModel):
    correlation_id: str
    error_code: str
    message: str
    timestamp: str


class QueryOptions(BaseModel):
    max_precedents: int = 10
    include_excerpts: bool = True


class QueryRequest(BaseModel):
    query: str
    options: QueryOptions = Field(default_factory=QueryOptions)


class QueryResponse(BaseModel):
    correlation_id: str
    query_type: str
    chat_response: str
    response: PrecedentAnalysis | GeneralQueryResponse
    sources_searched: int
    processing_time_ms: int
    user_message_id: str
    assistant_message_id: str


class DocumentSummary(BaseModel):
    document_id: str
    file_name: str
    case_name: str | None = None
    court_name: str | None = None
    judgment_date: str | None = None
    page_count: int = 0
    chunk_count: int = 0
    ingested_at: str


class DocumentListResponse(BaseModel):
    correlation_id: str
    total: int
    page: int
    page_size: int
    documents: list[DocumentSummary]


# ── Chat history schemas ──────────────────────────────────────────────────────


class WsQueryRequest(BaseModel):
    query: str
    mode: Literal["auto", "research", "general"] = "auto"
    options: QueryOptions = Field(default_factory=QueryOptions)


class MessageItem(BaseModel):
    id: str
    role: str
    content: str
    query_type: str | None = None
    sources_searched: int
    created_at: str
    raw_response: dict[str, Any] | None = None
    agent_steps: list[dict[str, Any]] | None = None


class ChatHistoryResponse(BaseModel):
    total: int
    limit: int
    offset: int
    messages: list[MessageItem] = Field(default_factory=list)
