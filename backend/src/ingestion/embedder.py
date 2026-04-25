from __future__ import annotations

import asyncio
import hashlib
from math import sqrt

from langchain_community.embeddings import HuggingFaceEmbeddings


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: HuggingFaceEmbeddings | None = None
        self._load_failed = False

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_batch_sync, texts)

    async def embed_one(self, text: str) -> list[float]:
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    def _embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        if model is None:
            return [self._fallback_embedding(text) for text in texts]

        vectors = model.embed_documents(texts)
        return [self._validate_embedding(vector) for vector in vectors]

    def _get_model(self) -> HuggingFaceEmbeddings | None:
        if self._load_failed:
            return None
        if self._model is None:
            try:
                self._model = HuggingFaceEmbeddings(
                    model_name=self._model_name,
                    encode_kwargs={"batch_size": 32},
                )
            except Exception:
                self._load_failed = True
                return None
        return self._model

    def _fallback_embedding(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [digest[index % len(digest)] / 255.0 for index in range(384)]
        norm = sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]

    def _validate_embedding(self, vector: list[float]) -> list[float]:
        if len(vector) != 384:
            raise ValueError(f"Expected 384-dim embedding, received {len(vector)}")
        return vector
