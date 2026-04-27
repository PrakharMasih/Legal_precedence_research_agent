from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import delete, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.core.cache import (
    TTL_CHAT_COUNT,
    TTL_CHAT_HISTORY,
    TTL_CHAT_RECENT,
    TTL_CHUNK_COUNT,
    TTL_CHUNK_LIST,
    TTL_DOC,
    TTL_DOC_COUNT,
    TTL_DOC_LIST,
    CacheClient,
)
from src.models.conversation import Message
from src.models.document import Chunk, Document
from src.storage.database import (
    chunks_table,
    documents_table,
    ingestion_failures_table,
    ingestion_runs_table,
    messages_table,
)


def _isoformat(value: datetime) -> str:
    return value.isoformat()


@dataclass(slots=True)
class IngestionFailureRecord:
    file_name: str
    error_message: str


# ---------------------------------------------------------------------------
# DocumentRepository
# ---------------------------------------------------------------------------


class DocumentRepository:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        cache: CacheClient | None = None,
    ) -> None:
        self._sf = session_factory
        self._cache = cache or CacheClient()

    # -- cache key helpers --------------------------------------------------

    @staticmethod
    def _key_doc(document_id: str) -> str:
        return f"casey:doc:{document_id}"

    @staticmethod
    def _key_list(limit: int, offset: int) -> str:
        return f"casey:doc:list:{limit}:{offset}"

    _KEY_COUNT = "casey:doc:count"

    async def _invalidate_lists(self) -> None:
        """Remove all cached list pages and the count key."""
        await self._cache.delete_pattern("casey:doc:list:*")
        await self._cache.delete(self._KEY_COUNT)

    # -- queries -------------------------------------------------------------

    async def get_by_id(self, document_id: str) -> Document | None:
        key = self._key_doc(document_id)
        cached = await self._cache.get(key)
        if cached is not None:
            return Document.model_validate(cached)

        async with self._sf() as session:
            result = await session.execute(
                select(documents_table).where(documents_table.c.id == document_id)
            )
            row = result.mappings().first()

        doc = Document.model_validate(dict(row)) if row else None
        if doc is not None:
            await self._cache.set(key, doc.model_dump(mode="json"), ttl=TTL_DOC)
        return doc

    async def get_by_filename(self, file_name: str) -> Document | None:
        async with self._sf() as session:
            result = await session.execute(
                select(documents_table).where(documents_table.c.file_name == file_name)
            )
            row = result.mappings().first()
            return Document.model_validate(dict(row)) if row else None

    async def insert(self, document: Document) -> None:
        async with self._sf() as session:
            await session.execute(
                insert(documents_table).values(
                    id=document.id,
                    file_name=document.file_name,
                    case_name=document.case_name,
                    court_name=document.court_name,
                    judgment_date=document.judgment_date,
                    page_count=document.page_count,
                    char_count=document.char_count,
                    ingested_at=_isoformat(document.ingested_at),
                    status=document.status,
                )
            )
            await session.commit()
        # Seed the per-document cache; bust list caches
        await self._cache.set(
            self._key_doc(document.id), document.model_dump(mode="json"), ttl=TTL_DOC
        )
        await self._invalidate_lists()

    async def delete_by_id(self, document_id: str) -> None:
        async with self._sf() as session:
            await session.execute(
                delete(documents_table).where(documents_table.c.id == document_id)
            )
            await session.commit()
        await self._cache.delete(self._key_doc(document_id))
        await self._invalidate_lists()

    async def update_status(self, document_id: str, status: str) -> None:
        async with self._sf() as session:
            await session.execute(
                update(documents_table)
                .where(documents_table.c.id == document_id)
                .values(status=status)
            )
            await session.commit()
        # Invalidate the cached document so the new status is visible
        await self._cache.delete(self._key_doc(document_id))

    async def list_all(self, limit: int = 50, offset: int = 0) -> list[Document]:
        key = self._key_list(limit, offset)
        cached = await self._cache.get(key)
        if cached is not None:
            return [Document.model_validate(d) for d in cached]

        async with self._sf() as session:
            result = await session.execute(
                select(documents_table)
                .order_by(documents_table.c.ingested_at.desc())
                .limit(limit)
                .offset(offset)
            )
            docs = [Document.model_validate(dict(row)) for row in result.mappings()]

        await self._cache.set(key, [d.model_dump(mode="json") for d in docs], ttl=TTL_DOC_LIST)
        return docs

    async def count(self) -> int:
        cached = await self._cache.get(self._KEY_COUNT)
        if cached is not None:
            return int(cached)

        async with self._sf() as session:
            result = await session.execute(select(documents_table.c.id))
            total = len(result.all())

        await self._cache.set(self._KEY_COUNT, total, ttl=TTL_DOC_COUNT)
        return total


# ---------------------------------------------------------------------------
# ChunkRepository
# ---------------------------------------------------------------------------


class ChunkRepository:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        cache: CacheClient | None = None,
    ) -> None:
        self._sf = session_factory
        self._cache = cache or CacheClient()

    # -- cache key helpers --------------------------------------------------

    @staticmethod
    def _key_count(document_id: str) -> str:
        return f"casey:chunk:count:{document_id}"

    @staticmethod
    def _key_list(document_id: str) -> str:
        return f"casey:chunk:list:{document_id}"

    # -- mutations ----------------------------------------------------------

    async def insert_batch(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        async with self._sf() as session:
            await session.execute(
                insert(chunks_table),
                [
                    {
                        "id": chunk.id,
                        "document_id": chunk.document_id,
                        "content": chunk.content,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                        "chunk_index": chunk.chunk_index,
                        "embedded_at": _isoformat(chunk.embedded_at),
                        "section": chunk.section,
                        "chunk_type": chunk.chunk_type,
                        "parent_id": chunk.parent_id,
                    }
                    for chunk in chunks
                ],
            )
            await session.commit()
        # Bust chunk caches for every affected document
        doc_ids = {c.document_id for c in chunks}
        for doc_id in doc_ids:
            await self._cache.delete(self._key_count(doc_id), self._key_list(doc_id))

    async def delete_by_document_id(self, document_id: str) -> None:
        async with self._sf() as session:
            await session.execute(
                delete(chunks_table).where(chunks_table.c.document_id == document_id)
            )
            await session.commit()
        await self._cache.delete(self._key_count(document_id), self._key_list(document_id))

    # -- queries ------------------------------------------------------------

    async def get_by_id(self, chunk_id: str) -> Chunk | None:
        async with self._sf() as session:
            result = await session.execute(
                select(chunks_table).where(chunks_table.c.id == chunk_id)
            )
            row = result.mappings().first()
            return Chunk.model_validate(dict(row)) if row else None

    async def get_by_document_id(self, document_id: str) -> list[Chunk]:
        key = self._key_list(document_id)
        cached = await self._cache.get(key)
        if cached is not None:
            return [Chunk.model_validate(c) for c in cached]

        async with self._sf() as session:
            result = await session.execute(
                select(chunks_table)
                .where(chunks_table.c.document_id == document_id)
                .order_by(chunks_table.c.chunk_index.asc())
            )
            chunks = [Chunk.model_validate(dict(row)) for row in result.mappings()]

        await self._cache.set(key, [c.model_dump(mode="json") for c in chunks], ttl=TTL_CHUNK_LIST)
        return chunks

    async def count_for_document(self, document_id: str) -> int:
        key = self._key_count(document_id)
        cached = await self._cache.get(key)
        if cached is not None:
            return int(cached)

        async with self._sf() as session:
            result = await session.execute(
                select(chunks_table.c.id).where(chunks_table.c.document_id == document_id)
            )
            total = len(result.all())

        await self._cache.set(key, total, ttl=TTL_CHUNK_COUNT)
        return total


# ---------------------------------------------------------------------------
# IngestionRunRepository
# ---------------------------------------------------------------------------


class IngestionRunRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    async def create(self, payload: dict[str, Any]) -> None:
        async with self._sf() as session:
            await session.execute(
                insert(ingestion_runs_table).values(
                    id=payload["id"],
                    corpus_dir=payload["corpus_dir"],
                    started_at=payload["started_at"],
                    completed_at=payload.get("completed_at"),
                    total_files=payload.get("total_files", 0),
                    succeeded=payload.get("succeeded", 0),
                    failed=payload.get("failed", 0),
                    total_chunks=payload.get("total_chunks", 0),
                    status=payload.get("status", "running"),
                )
            )
            await session.commit()

    async def update(self, run_id: str, payload: dict[str, Any]) -> None:
        async with self._sf() as session:
            await session.execute(
                update(ingestion_runs_table)
                .where(ingestion_runs_table.c.id == run_id)
                .values(**payload)
            )
            await session.commit()

    async def get_by_id(self, run_id: str) -> dict[str, Any] | None:
        async with self._sf() as session:
            result = await session.execute(
                select(ingestion_runs_table).where(ingestion_runs_table.c.id == run_id)
            )
            row = result.mappings().first()
            return dict(row) if row else None

    async def get_active_run(self) -> dict[str, Any] | None:
        async with self._sf() as session:
            result = await session.execute(
                select(ingestion_runs_table)
                .where(ingestion_runs_table.c.status == "running")
                .order_by(ingestion_runs_table.c.started_at.desc())
                .limit(1)
            )
            row = result.mappings().first()
            return dict(row) if row else None


# ---------------------------------------------------------------------------
# IngestionFailureRepository
# ---------------------------------------------------------------------------


class IngestionFailureRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    async def insert(self, run_id: str, failure: IngestionFailureRecord) -> None:
        async with self._sf() as session:
            await session.execute(
                insert(ingestion_failures_table).values(
                    run_id=run_id,
                    file_name=failure.file_name,
                    error_message=failure.error_message,
                )
            )
            await session.commit()

    async def get_by_run_id(self, run_id: str) -> list[dict[str, Any]]:
        async with self._sf() as session:
            result = await session.execute(
                select(
                    ingestion_failures_table.c.file_name,
                    ingestion_failures_table.c.error_message,
                )
                .where(ingestion_failures_table.c.run_id == run_id)
                .order_by(ingestion_failures_table.c.id.asc())
            )
            return [dict(row) for row in result.mappings()]


# ---------------------------------------------------------------------------
# ChatRepository
# ---------------------------------------------------------------------------


class ChatRepository:
    CONTEXT_WINDOW = 20

    _KEY_RECENT = "casey:chat:recent"
    _KEY_COUNT = "casey:chat:count"

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        cache: CacheClient | None = None,
    ) -> None:
        self._sf = session_factory
        self._cache = cache or CacheClient()

    @staticmethod
    def _key_history(limit: int, offset: int) -> str:
        return f"casey:chat:history:{limit}:{offset}"

    async def _invalidate_all(self) -> None:
        """Bust every chat cache entry — called on append and clear."""
        await self._cache.delete(self._KEY_RECENT, self._KEY_COUNT)
        await self._cache.delete_pattern("casey:chat:history:*")

    async def append(self, message: Message) -> None:
        async with self._sf() as session:
            await session.execute(
                insert(messages_table).values(
                    id=message.id,
                    role=message.role,
                    content=message.content,
                    raw_response=json.dumps(message.raw_response)
                    if message.raw_response is not None
                    else None,
                    agent_steps=json.dumps(message.agent_steps)
                    if message.agent_steps is not None
                    else None,
                    query_type=message.query_type,
                    sources_searched=message.sources_searched,
                    created_at=_isoformat(message.created_at),
                )
            )
            await session.commit()
        # Invalidate after the write so the next read is always fresh
        await self._invalidate_all()

    async def get_by_id(self, message_id: str) -> Message | None:
        async with self._sf() as session:
            result = await session.execute(
                select(messages_table).where(messages_table.c.id == message_id)
            )
            row = result.mappings().first()
            return Message.model_validate(dict(row)) if row else None

    async def get_history(self, limit: int = 50, offset: int = 0) -> list[Message]:
        """Return a chronological page of messages for display."""
        key = self._key_history(limit, offset)
        cached = await self._cache.get(key)
        if cached is not None:
            return [Message.model_validate(m) for m in cached]

        async with self._sf() as session:
            result = await session.execute(
                select(messages_table)
                .order_by(messages_table.c.created_at.asc())
                .limit(limit)
                .offset(offset)
            )
            messages = [Message.model_validate(dict(row)) for row in result.mappings()]

        await self._cache.set(
            key,
            [m.model_dump(mode="json") for m in messages],
            ttl=TTL_CHAT_HISTORY,
        )
        return messages

    async def get_recent_context(self) -> list[Message]:
        """Return the most-recent CONTEXT_WINDOW messages in chronological order."""
        cached = await self._cache.get(self._KEY_RECENT)
        if cached is not None:
            return [Message.model_validate(m) for m in cached]

        async with self._sf() as session:
            subq = (
                select(messages_table)
                .order_by(messages_table.c.created_at.desc())
                .limit(self.CONTEXT_WINDOW)
                .subquery()
            )
            result = await session.execute(select(subq).order_by(subq.c.created_at.asc()))
            messages = [Message.model_validate(dict(row)) for row in result.mappings()]

        await self._cache.set(
            self._KEY_RECENT,
            [m.model_dump(mode="json") for m in messages],
            ttl=TTL_CHAT_RECENT,
        )
        return messages

    async def count(self) -> int:
        cached = await self._cache.get(self._KEY_COUNT)
        if cached is not None:
            return int(cached)

        async with self._sf() as session:
            result = await session.execute(select(messages_table.c.id))
            total = len(result.all())

        await self._cache.set(self._KEY_COUNT, total, ttl=TTL_CHAT_COUNT)
        return total

    async def clear(self) -> None:
        async with self._sf() as session:
            await session.execute(delete(messages_table))
            await session.commit()
        await self._invalidate_all()
