from __future__ import annotations

from src.models.query import RankedChunk


def rrf_fuse(
    dense_results: list[RankedChunk],
    sparse_results: list[RankedChunk],
    *,
    k: int = 60,
    limit: int = 10,
) -> list[RankedChunk]:
    if not dense_results and not sparse_results:
        return []

    merged: dict[str, RankedChunk] = {}
    scores: dict[str, float] = {}

    for results in (dense_results, sparse_results):
        for index, chunk in enumerate(results, start=1):
            merged.setdefault(chunk.chunk_id, chunk)
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1 / (k + index)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [
        merged[chunk_id].model_copy(update={"rrf_score": score})
        for chunk_id, score in ranked[:limit]
    ]
