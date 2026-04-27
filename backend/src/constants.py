"""
Centralized constants for the application.

This module defines all magic strings, error codes, HTTP status codes,
and other constants used throughout the codebase.
"""

from __future__ import annotations

# ── HTTP Status Codes ─────────────────────────────────────────────────────────

HTTP_200_OK = 200
HTTP_202_ACCEPTED = 202
HTTP_400_BAD_REQUEST = 400
HTTP_404_NOT_FOUND = 404
HTTP_409_CONFLICT = 409
HTTP_500_INTERNAL_SERVER_ERROR = 500
HTTP_503_SERVICE_UNAVAILABLE = 503

# ── Request Headers ──────────────────────────────────────────────────────────

HEADER_CORRELATION_ID = "X-Correlation-ID"

# ── Error Codes & Messages ──────────────────────────────────────────────────

# Ingestion errors
ERROR_CODE_INGESTION_IN_PROGRESS = "INGESTION_IN_PROGRESS"
ERROR_MSG_INGESTION_IN_PROGRESS = (
    "An ingestion run is already running. Check the existing run status before starting another."
)

ERROR_CODE_INGESTION_RUN_NOT_FOUND = "INGESTION_RUN_NOT_FOUND"
ERROR_MSG_INGESTION_RUN_NOT_FOUND = "No ingestion run found for run_id={run_id}"

# Query & LLM errors
ERROR_CODE_LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
ERROR_CODE_CORPUS_NOT_INDEXED = "CORPUS_NOT_INDEXED"
ERROR_CODE_CASEY_ERROR = "CASEY_ERROR"

# WebSocket & Input errors
ERROR_CODE_RECEIVE_TIMEOUT = "RECEIVE_TIMEOUT"
ERROR_MSG_RECEIVE_TIMEOUT = "Client did not send message within 60 seconds."

ERROR_CODE_INVALID_JSON = "INVALID_JSON"
ERROR_MSG_INVALID_JSON = "Request must be valid JSON."

ERROR_CODE_RECEIVE_ERROR = "RECEIVE_ERROR"
ERROR_MSG_RECEIVE_ERROR = "Failed to receive message: {error_type}"

ERROR_CODE_INVALID_MODE = "INVALID_MODE"
ERROR_MSG_INVALID_MODE = "Invalid query mode. Must be one of: auto, research, general"

ERROR_CODE_EMPTY_QUERY = "EMPTY_QUERY"
ERROR_MSG_EMPTY_QUERY = "Query cannot be empty."

# ── Status Strings ───────────────────────────────────────────────────────────

STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

# ── Query Modes ──────────────────────────────────────────────────────────────

QUERY_MODE_AUTO = "auto"
QUERY_MODE_RESEARCH = "research"
QUERY_MODE_GENERAL = "general"
VALID_QUERY_MODES = frozenset({QUERY_MODE_AUTO, QUERY_MODE_RESEARCH, QUERY_MODE_GENERAL})

# ── WebSocket Message Types ──────────────────────────────────────────────────

WS_MSG_TYPE_AGENT_STARTED = "agent_started"
WS_MSG_TYPE_THINKING = "thinking"
WS_MSG_TYPE_TOOL_RESULT = "tool_result"
WS_MSG_TYPE_REASONING = "reasoning"
WS_MSG_TYPE_SYNTHESIZING = "synthesizing"
WS_MSG_TYPE_QUERY_TYPE = "query_type"
WS_MSG_TYPE_STREAMING = "streaming"
WS_MSG_TYPE_STREAM_CHUNK = "stream_chunk"
WS_MSG_TYPE_COMPLETED = "completed"
WS_MSG_TYPE_ERROR = "error"

# ── Document & Ingestion Constants ───────────────────────────────────────────

DEFAULT_CORPUS_DIR = "judgement_pdfs"
DEFAULT_SQLITE_DB_PATH = "data/casey.db"
DEFAULT_QDRANT_PATH = "data/qdrant"
DEFAULT_QDRANT_COLLECTION = "judgments"

# ── Pagination ───────────────────────────────────────────────────────────────

DEFAULT_PAGE_SIZE = 50
MIN_PAGE_SIZE = 1
MAX_PAGE_SIZE = 200

# ── Chat History ─────────────────────────────────────────────────────────────

DEFAULT_CHAT_HISTORY_LIMIT = 50
MIN_CHAT_HISTORY_LIMIT = 1
MAX_CHAT_HISTORY_LIMIT = 200
DEFAULT_CHAT_HISTORY_OFFSET = 0
MIN_CHAT_HISTORY_OFFSET = 0

# ── WebSocket ────────────────────────────────────────────────────────────────

WS_RECEIVE_TIMEOUT_SECONDS = 60.0
