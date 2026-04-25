from __future__ import annotations

from qdrant_client.models import ScoredPoint

from src.models.query import RankedChunk
from src.storage.vector_store import VectorStore


class DenseRetriever:
    def __init__(self, vector_store: VectorStore):
        self._vector_store = vector_store

    async def search(self, query_embedding: list[float], n_results: int = 20) -> list[RankedChunk]:
        hits: list[ScoredPoint] = await self._vector_store.query_dense(
            query_embedding, n_results=n_results
        )
        ranked: list[RankedChunk] = []
        for hit in hits:
            payload = hit.payload or {}
            ranked.append(
                RankedChunk(
                    chunk_id=str(hit.id),
                    document_id=payload.get("document_id", ""),
                    file_name=payload.get("file_name", ""),
                    case_name=payload.get("case_name") or None,
                    content=payload.get("content", ""),
                    char_start=int(payload.get("char_start", 0)),
                    char_end=int(payload.get("char_end", 0)),
                    section=payload.get("section") or None,
                )
            )
        return ranked
