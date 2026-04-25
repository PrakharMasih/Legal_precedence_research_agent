from __future__ import annotations

import re

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.models.query import RankedChunk


class SparseRetriever:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    async def search(self, query_text: str, n_results: int = 20) -> list[RankedChunk]:
        sanitized_query = self._sanitize_query(query_text)
        if not sanitized_query:
            return []

        async with self._sf() as session:
            result = await session.execute(
                text("""
                    SELECT c.id, c.document_id, d.file_name, d.case_name, c.content,
                           c.char_start, c.char_end, c.section
                    FROM chunks_fts
                    JOIN chunks c ON chunks_fts.rowid = c.rowid
                    JOIN documents d ON c.document_id = d.id
                    WHERE chunks_fts MATCH :query
                      AND (c.chunk_type = 'child' OR c.chunk_type IS NULL)
                    ORDER BY rank
                    LIMIT :n
                """),
                {"query": sanitized_query, "n": n_results},
            )
            rows = result.mappings().all()

        return [
            RankedChunk(
                chunk_id=row["id"],
                document_id=row["document_id"],
                file_name=row["file_name"],
                case_name=row["case_name"],
                content=row["content"],
                char_start=row["char_start"],
                char_end=row["char_end"],
                section=row["section"] or None,
            )
            for row in rows
        ]

    def _sanitize_query(self, query_text: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9]+", query_text.lower())
        return " AND ".join(tokens)
