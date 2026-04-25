from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    role: Literal["user", "assistant"]
    content: str
    query_type: str | None = None
    sources_searched: int = 0
    raw_response: dict[str, Any] | None = None
    agent_steps: list[dict[str, Any]] | None = None
    created_at: datetime

    @field_validator("raw_response", mode="before")
    @classmethod
    def _parse_raw_response(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, ValueError):
                return None
        return v

    @field_validator("agent_steps", mode="before")
    @classmethod
    def _parse_agent_steps(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, ValueError):
                return None
        return v
