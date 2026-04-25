from __future__ import annotations

import asyncio
from dataclasses import dataclass

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from src.core.cache import CacheClient
from src.core.config import Settings, get_settings
from src.ingestion.chunker import Chunker
from src.ingestion.embedder import Embedder
from src.ingestion.pipeline import IngestionPipeline
from src.llm.base import LLMProvider
from src.llm.factory import LLMFactory
from src.storage.database import create_db_engine, init_schema, make_session_factory
from src.storage.repositories import (
    ChatRepository,
    ChunkRepository,
    DocumentRepository,
    IngestionFailureRepository,
    IngestionRunRepository,
)
from src.storage.vector_store import VectorStore


@dataclass(slots=True)
class RuntimeServices:
    settings: Settings
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    vector_store: VectorStore
    document_repository: DocumentRepository
    chunk_repository: ChunkRepository
    ingestion_run_repository: IngestionRunRepository
    ingestion_failure_repository: IngestionFailureRepository
    chat_repository: ChatRepository
    embedder: Embedder
    chunker: Chunker
    ingestion_pipeline: IngestionPipeline
    llm_provider: LLMProvider  # Cached LLM provider for connection reuse
    cache: CacheClient  # Redis cache client (no-op when Redis is not configured)


async def ensure_runtime(app: FastAPI) -> RuntimeServices:
    runtime = getattr(app.state, "runtime", None)
    if runtime is not None:
        return runtime

    lock = getattr(app.state, "runtime_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        app.state.runtime_lock = lock

    async with lock:
        runtime = getattr(app.state, "runtime", None)
        if runtime is not None:
            return runtime

        settings = get_settings()
        engine = create_db_engine(settings.sqlite_db_path)
        await init_schema(engine)
        session_factory = make_session_factory(engine)

        vector_store = VectorStore(
            url=settings.qdrant_url or None,
            api_key=settings.qdrant_api_key or None,
            path=None if settings.qdrant_url else settings.qdrant_path,
            collection=settings.qdrant_collection,
        )
        await vector_store.init_collection()

        cache = CacheClient(settings.redis_url)
        await cache.connect()

        document_repository = DocumentRepository(session_factory, cache=cache)
        chunk_repository = ChunkRepository(session_factory, cache=cache)
        ingestion_run_repository = IngestionRunRepository(session_factory)
        ingestion_failure_repository = IngestionFailureRepository(session_factory)
        chat_repository = ChatRepository(session_factory, cache=cache)
        embedder = Embedder()
        chunker = Chunker()
        ingestion_pipeline = IngestionPipeline(
            document_repository=document_repository,
            chunk_repository=chunk_repository,
            ingestion_run_repository=ingestion_run_repository,
            ingestion_failure_repository=ingestion_failure_repository,
            vector_store=vector_store,
            embedder=embedder,
            chunker=chunker,
        )
        llm_provider = LLMFactory.from_config(settings)

        runtime = RuntimeServices(
            settings=settings,
            engine=engine,
            session_factory=session_factory,
            vector_store=vector_store,
            document_repository=document_repository,
            chunk_repository=chunk_repository,
            ingestion_run_repository=ingestion_run_repository,
            ingestion_failure_repository=ingestion_failure_repository,
            chat_repository=chat_repository,
            embedder=embedder,
            chunker=chunker,
            ingestion_pipeline=ingestion_pipeline,
            llm_provider=llm_provider,
            cache=cache,
        )
        app.state.runtime = runtime
        app.state.ingestion_tasks = {}
        return runtime


async def close_runtime(app: FastAPI) -> None:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        return
    await runtime.cache.close()
    await runtime.vector_store.close()
    await runtime.engine.dispose()
    app.state.runtime = None
