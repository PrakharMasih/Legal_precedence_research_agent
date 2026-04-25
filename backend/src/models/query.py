from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class SearchMode(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class QueryRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    correlation_id: str
    query_text: str
    submitted_at: datetime


class RankedChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    file_name: str
    case_name: str | None = None  # legal case name (e.g., "Party A vs Party B")
    content: str
    char_start: int
    char_end: int
    rrf_score: float = 0.0
    section: str | None = None
    parent_content: str | None = None  # parent chunk content supplied as LLM reasoning context
