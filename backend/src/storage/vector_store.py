from __future__ import annotations

from pathlib import Path
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import QueryResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

# Hard-coded to match all-MiniLM-L6-v2 output dimension.
_VECTOR_DIM = 384


class VectorStore:
    """Async Qdrant-backed vector store.

    Modes (resolved in priority order):
      1. Remote  – ``url`` is set (optionally with ``api_key`` for Qdrant Cloud).
      2. Embedded – ``path`` is set; uses qdrant-client local file storage.
      3. In-memory – fallback (useful for tests; pass ``path=None, url=None``).
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        api_key: str | None = None,
        path: Path | None = None,
        collection: str = "judgments",
        vector_dim: int = _VECTOR_DIM,
    ) -> None:
        self._url = url
        self._api_key = api_key or None
        self._path = path
        self._collection = collection
        self._vector_dim = vector_dim
        self._client: AsyncQdrantClient | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def init_collection(self) -> None:
        """Create the Qdrant client and ensure the collection exists."""
        self._client = self._build_client()
        exists = await self._client.collection_exists(self._collection)
        if not exists:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._vector_dim, distance=Distance.COSINE),
            )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def add_embeddings(
        self,
        chunk_id: str,
        embedding: list[float],
        metadata: dict[str, Any],
        content: str,
    ) -> None:
        """Upsert a single embedding with its payload into the collection."""
        client = await self._require_client()
        payload = {**metadata, "content": content}
        await client.upsert(
            collection_name=self._collection,
            points=[PointStruct(id=chunk_id, vector=embedding, payload=payload)],
            wait=True,
        )

    async def delete_by_document_id(self, document_id: str) -> None:
        """Remove all points whose payload ``document_id`` matches."""
        client = await self._require_client()
        await client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
                )
            ),
            wait=True,
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def query_dense(
        self,
        embedding: list[float],
        n_results: int = 20,
    ) -> list[ScoredPoint]:
        """Return the top-n nearest neighbours as Qdrant ``ScoredPoint`` objects."""
        client = await self._require_client()
        response: QueryResponse = await client.query_points(
            collection_name=self._collection,
            query=embedding,
            limit=n_results,
            with_payload=True,
            with_vectors=False,
        )
        return response.points

    async def count(self) -> int:
        client = await self._require_client()
        result = await client.count(collection_name=self._collection, exact=True)
        return result.count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_client(self) -> AsyncQdrantClient:
        if self._url:
            return AsyncQdrantClient(url=self._url, api_key=self._api_key)
        if self._path is not None:
            self._path.mkdir(parents=True, exist_ok=True)
            return AsyncQdrantClient(path=str(self._path))
        # Fallback: in-memory (useful for isolated tests)
        return AsyncQdrantClient(location=":memory:")

    async def _require_client(self) -> AsyncQdrantClient:
        if self._client is None:
            await self.init_collection()
        return self._client
