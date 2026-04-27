"""Runtime initialization and lifecycle management."""

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
from src.services.chat_service import ChatService
from src.services.ingestion_service import IngestionService
from src.services.query_service import QueryService
from src.services.retrieval_service import RetrievalService
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
    """Application runtime dependencies and services."""

    # ── Configuration & Core ──────────────────────────────────────────────
    settings: Settings
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]

    # ── Storage & Infrastructure ──────────────────────────────────────────
    vector_store: VectorStore
    cache: CacheClient

    # ── Repositories (Data Access Layer) ──────────────────────────────────
    document_repository: DocumentRepository
    chunk_repository: ChunkRepository
    ingestion_run_repository: IngestionRunRepository
    ingestion_failure_repository: IngestionFailureRepository
    chat_repository: ChatRepository

    # ── Ingestion Components ──────────────────────────────────────────────
    embedder: Embedder
    chunker: Chunker
    ingestion_pipeline: IngestionPipeline

    # ── LLM Provider ──────────────────────────────────────────────────────
    llm_provider: LLMProvider

    # ── Services (Business Logic Layer) ───────────────────────────────────
    chat_service: ChatService
    ingestion_service: IngestionService
    query_service: QueryService
    retrieval_service: RetrievalService


async def ensure_runtime(app: FastAPI) -> RuntimeServices:
    """
    Ensure runtime services are initialized.

    Lazy-initializes all runtime dependencies on first access. Subsequent calls
    return the cached RuntimeServices instance.

    Args:
        app: FastAPI application instance.

    Returns:
        RuntimeServices with all dependencies initialized.
    """
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

        # ── Initialize configuration ──────────────────────────────────────
        settings = get_settings()

        # ── Initialize storage ────────────────────────────────────────────
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

        # ── Initialize repositories ───────────────────────────────────────
        document_repository = DocumentRepository(session_factory, cache=cache)
        chunk_repository = ChunkRepository(session_factory, cache=cache)
        ingestion_run_repository = IngestionRunRepository(session_factory)
        ingestion_failure_repository = IngestionFailureRepository(session_factory)
        chat_repository = ChatRepository(session_factory, cache=cache)

        # ── Initialize ingestion components ───────────────────────────────
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

        # ── Initialize LLM provider ───────────────────────────────────────
        llm_provider = LLMFactory.from_config(settings)

        # ── Initialize services ───────────────────────────────────────────
        chat_service = ChatService(chat_repository=chat_repository)

        ingestion_service = IngestionService(
            ingestion_pipeline=ingestion_pipeline,
            ingestion_run_repository=ingestion_run_repository,
        )

        query_service = QueryService(
            llm_provider=llm_provider,
            chat_repository=chat_repository,
            document_repository=document_repository,
            chunk_repository=chunk_repository,
            vector_store=vector_store,
            embedder=embedder,
        )

        retrieval_service = RetrievalService(
            document_repository=document_repository,
            chunk_repository=chunk_repository,
        )

        # ── Create runtime ────────────────────────────────────────────────
        runtime = RuntimeServices(
            settings=settings,
            engine=engine,
            session_factory=session_factory,
            vector_store=vector_store,
            cache=cache,
            document_repository=document_repository,
            chunk_repository=chunk_repository,
            ingestion_run_repository=ingestion_run_repository,
            ingestion_failure_repository=ingestion_failure_repository,
            chat_repository=chat_repository,
            embedder=embedder,
            chunker=chunker,
            ingestion_pipeline=ingestion_pipeline,
            llm_provider=llm_provider,
            chat_service=chat_service,
            ingestion_service=ingestion_service,
            query_service=query_service,
            retrieval_service=retrieval_service,
        )
        app.state.runtime = runtime
        app.state.ingestion_tasks = {}
        return runtime


async def close_runtime(app: FastAPI) -> None:
    """
    Close and cleanup runtime resources.

    Gracefully shuts down all runtime services and connections.

    Args:
        app: FastAPI application instance.
    """
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        return
    await runtime.cache.close()
    await runtime.vector_store.close()
    await runtime.engine.dispose()
    app.state.runtime = None
