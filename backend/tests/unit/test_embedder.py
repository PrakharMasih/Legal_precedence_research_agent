from __future__ import annotations

import asyncio

import pytest

from src.ingestion.embedder import Embedder


@pytest.mark.asyncio
async def test_embed_one_returns_384_dimensions() -> None:
    embedder = Embedder()

    vector = await embedder.embed_one("pay and recover doctrine")

    assert len(vector) == 384
    assert all(isinstance(value, float) for value in vector)


@pytest.mark.asyncio
async def test_embed_batch_returns_one_vector_per_input() -> None:
    embedder = Embedder()

    vectors = await embedder.embed_batch(["alpha", "beta", "gamma"])

    assert len(vectors) == 3
    assert all(len(vector) == 384 for vector in vectors)


@pytest.mark.asyncio
async def test_embed_batch_uses_to_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    embedder = Embedder()
    called = False

    async def fake_to_thread(func, *args, **kwargs):
        nonlocal called
        called = True
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    vectors = await embedder.embed_batch(["insurer liability"])

    assert called is True
    assert len(vectors) == 1
    assert len(vectors[0]) == 384
