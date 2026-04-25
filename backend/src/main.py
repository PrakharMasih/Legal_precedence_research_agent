from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.v1.routes.chat import router as chat_router
from src.api.v1.routes.documents import router as documents_router
from src.api.v1.routes.ingest import router as ingest_router
from src.api.v1.routes.query import router as query_router
from src.api.v1.routes.ws import router as ws_router
from src.core.config import get_settings
from src.core.exceptions import (
    CorpusNotIndexedError,
    IngestionError,
    LexiError,
    LLMUnavailableError,
    RetrievalError,
    ValidationError,
)
from src.core.logging import (
    bind_correlation_id,
    clear_correlation_id,
    configure_logging,
    correlation_id_var,
    get_logger,
)
from src.core.runtime import close_runtime, ensure_runtime

logger = get_logger(component="app")


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _current_correlation_id(request: Request) -> str:
    return request.headers.get("X-Correlation-ID") or correlation_id_var.get() or str(uuid4())


def _error_response(
    *,
    request: Request,
    status_code: int,
    error_code: str,
    message: str,
) -> JSONResponse:
    correlation_id = _current_correlation_id(request)
    payload = {
        "correlation_id": correlation_id,
        "error_code": error_code,
        "message": message,
        "timestamp": _timestamp(),
    }
    response = JSONResponse(status_code=status_code, content=payload)
    response.headers["X-Correlation-ID"] = correlation_id
    return response


def _verify_writable_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe_path = path / ".write_probe"
    probe_path.write_text("ok", encoding="utf-8")
    probe_path.unlink(missing_ok=True)


def _log_startup_configuration() -> None:
    settings = get_settings()
    corpus_exists = settings.corpus_dir.exists()
    if not corpus_exists:
        logger.warning("application.corpus_dir_missing", corpus_dir=str(settings.corpus_dir))

    _verify_writable_directory(settings.sqlite_db_path.parent)
    logger.info(
        "application.startup_config",
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        corpus_dir=str(settings.corpus_dir),
        corpus_dir_exists=corpus_exists,
        sqlite_db_path=str(settings.sqlite_db_path),
        qdrant_url=settings.qdrant_url or f"embedded:{settings.qdrant_path}",
        qdrant_collection=settings.qdrant_collection,
        log_level=settings.log_level,
        log_format=settings.log_format,
        host=settings.host,
        port=settings.port,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    _log_startup_configuration()
    try:
        await ensure_runtime(app)
        yield
    finally:
        await close_runtime(app)
        logger.info("application.shutdown")


def create_app() -> FastAPI:
    application = FastAPI(
        title="Lexi Legal Precedent Research API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Configure CORS for frontend integration
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins (configure for production)
        allow_credentials=True,
        allow_methods=["*"],  # Allow all HTTP methods
        allow_headers=["*"],  # Allow all headers
    )

    @application.middleware("http")
    async def correlation_id_middleware(request: Request, call_next):
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid4()))
        bind_correlation_id(correlation_id)
        try:
            response = await call_next(request)
        finally:
            clear_correlation_id()
        response.headers["X-Correlation-ID"] = correlation_id
        return response

    @application.get("/health")
    async def healthcheck() -> JSONResponse:
        return JSONResponse(content={"status": "ok"})

    @application.exception_handler(IngestionError)
    async def handle_ingestion_error(request: Request, exc: IngestionError) -> JSONResponse:
        return _error_response(
            request=request,
            status_code=400,
            error_code="INGESTION_ERROR",
            message=str(exc),
        )

    @application.exception_handler(CorpusNotIndexedError)
    async def handle_corpus_not_indexed(
        request: Request,
        exc: CorpusNotIndexedError,
    ) -> JSONResponse:
        return _error_response(
            request=request,
            status_code=409,
            error_code="CORPUS_NOT_INDEXED",
            message=str(exc),
        )

    @application.exception_handler(LLMUnavailableError)
    async def handle_llm_unavailable(
        request: Request,
        exc: LLMUnavailableError,
    ) -> JSONResponse:
        return _error_response(
            request=request,
            status_code=503,
            error_code="LLM_UNAVAILABLE",
            message=str(exc),
        )

    @application.exception_handler(ValidationError)
    async def handle_validation_error(request: Request, exc: ValidationError) -> JSONResponse:
        return _error_response(
            request=request,
            status_code=400,
            error_code="VALIDATION_ERROR",
            message=str(exc),
        )

    @application.exception_handler(RetrievalError)
    async def handle_retrieval_error(request: Request, exc: RetrievalError) -> JSONResponse:
        return _error_response(
            request=request,
            status_code=500,
            error_code="RETRIEVAL_ERROR",
            message=str(exc),
        )

    @application.exception_handler(LexiError)
    async def handle_lexi_error(request: Request, exc: LexiError) -> JSONResponse:
        return _error_response(
            request=request,
            status_code=400,
            error_code="LEXI_ERROR",
            message=str(exc),
        )

    @application.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return _error_response(
            request=request,
            status_code=422,
            error_code="REQUEST_VALIDATION_ERROR",
            message=str(exc),
        )

    @application.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("application.unhandled_exception", error=str(exc))
        return _error_response(
            request=request,
            status_code=500,
            error_code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred.",
        )

    application.include_router(ingest_router, prefix="/api/v1")
    application.include_router(query_router, prefix="/api/v1")
    application.include_router(documents_router, prefix="/api/v1")
    application.include_router(chat_router, prefix="/api/v1")
    application.include_router(ws_router)  # WebSocket — no HTTP prefix
    return application


app = create_app()
