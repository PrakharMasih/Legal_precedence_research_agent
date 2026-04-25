from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=64, max_length=64)
    file_name: str
    case_name: str | None = None
    court_name: str | None = None
    judgment_date: str | None = None
    page_count: int = 0
    char_count: int = 0
    ingested_at: datetime
    status: Literal["success", "failed"] = "success"


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    document_id: str
    content: str
    char_start: int
    char_end: int
    chunk_index: int
    embedded_at: datetime
    section: str = "other"
    chunk_type: str = "child"
    parent_id: str | None = None
