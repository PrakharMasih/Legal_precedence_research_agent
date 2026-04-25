from __future__ import annotations

import asyncio

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.core.exceptions import CorpusNotIndexedError
from src.ingestion.embedder import Embedder
from src.models.query import RankedChunk, SearchMode
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import rrf_fuse
from src.retrieval.sparse import SparseRetriever
from src.storage.vector_store import VectorStore


class Retriever:
    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        vector_store: VectorStore,
        embedder: Embedder,
    ) -> None:
        self._sf = session_factory
        self._dense = DenseRetriever(vector_store)
        self._sparse = SparseRetriever(session_factory)
        self._embedder = embedder

    async def retrieve(
        self,
        query_text: str,
        *,
        n: int = 10,
        search_mode: SearchMode = SearchMode.HYBRID,
    ) -> list[RankedChunk]:
        if not query_text.strip():
            return []

        if await self._count_documents() == 0:
            raise CorpusNotIndexedError("Corpus has not been indexed yet")

        query_embedding = await self._embedder.embed_one(query_text)
        if search_mode == SearchMode.DENSE:
            ranked = (await self._dense.search(query_embedding, n_results=n))[:n]
        elif search_mode == SearchMode.SPARSE:
            ranked = (await self._sparse.search(query_text, n_results=n))[:n]
        else:
            dense_results, sparse_results = await asyncio.gather(
                self._dense.search(query_embedding, n_results=max(n, 20)),
                self._sparse.search(query_text, n_results=max(n, 20)),
            )
            ranked = rrf_fuse(dense_results, sparse_results, limit=n)

        return await self._expand_parent_context(ranked)

    # -- Parent-context expansion -------------------------------------------

    async def _expand_parent_context(self, chunks: list[RankedChunk]) -> list[RankedChunk]:
        """Attach each child chunk's parent content for LLM reasoning context."""
        if not chunks:
            return chunks

        chunk_ids = [c.chunk_id for c in chunks]
        stmt = text(
            """
            SELECT c_child.id AS child_id, c_parent.content AS parent_content
            FROM chunks c_child
            JOIN chunks c_parent ON c_child.parent_id = c_parent.id
            WHERE c_child.id IN :ids
            """
        ).bindparams(bindparam("ids", expanding=True))

        async with self._sf() as session:
            result = await session.execute(stmt, {"ids": chunk_ids})
            rows = result.mappings().all()

        parent_map: dict[str, str] = {
            row["child_id"]: row["parent_content"] for row in rows
        }
        return [
            chunk.model_copy(update={"parent_content": parent_map.get(chunk.chunk_id)})
            for chunk in chunks
        ]

    async def _count_documents(self) -> int:
        async with self._sf() as session:
            result = await session.execute(text("SELECT COUNT(*) AS total FROM documents"))
            row = result.first()
            return int(row[0]) if row else 0
