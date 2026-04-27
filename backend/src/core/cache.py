"""Async Redis cache client with graceful degradation.

All operations are no-ops when Redis is not configured or unreachable —
the application continues to function correctly, just without caching.

Cache key namespaces:
    casey:chat:recent           – recent conversation context (20 s TTL)
    casey:chat:history:{l}:{o}  – paginated chat history (30 s TTL)
    casey:chat:count            – total message count (30 s TTL)
    casey:doc:{id}              – single document metadata (600 s TTL)
    casey:doc:list:{l}:{o}      – paginated document list (300 s TTL)
    casey:doc:count             – total document count (300 s TTL)
    casey:chunk:count:{doc_id}  – chunk count per document (600 s TTL)
    casey:chunk:list:{doc_id}   – chunk list per document (600 s TTL)
"""

from __future__ import annotations

import json
import logging
from typing import Any

_logger = logging.getLogger(__name__)

# redis >=5 ships asyncio support natively (no extra required)
try:
    import redis.asyncio as aioredis  # type: ignore[import]

    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


# ---------------------------------------------------------------------------
# TTLs (seconds)
# ---------------------------------------------------------------------------
TTL_CHAT_RECENT = 20  # recent context — very short; chat is real-time
TTL_CHAT_HISTORY = 30  # paginated history displayed in UI
TTL_CHAT_COUNT = 30  # message count for pagination
TTL_DOC = 600  # single document — immutable after ingestion
TTL_DOC_LIST = 300  # document list page
TTL_DOC_COUNT = 300  # total document count
TTL_CHUNK_COUNT = 600  # chunk count per document — immutable after ingestion
TTL_CHUNK_LIST = 600  # chunk list per document — immutable after ingestion


class CacheClient:
    """Thin async wrapper around redis.asyncio with graceful degradation."""

    def __init__(self, redis_url: str | None = None) -> None:
        self._url = redis_url
        self._client: Any | None = None

    @property
    def available(self) -> bool:
        return self._client is not None

    async def connect(self) -> None:
        if not self._url or not _REDIS_AVAILABLE:
            _logger.info("cache: Redis not configured or package missing — caching disabled")
            return
        try:
            self._client = aioredis.from_url(
                self._url,
                decode_responses=True,
                socket_connect_timeout=2.0,
                socket_timeout=2.0,
            )
            await self._client.ping()
            _logger.info("cache: connected to Redis", extra={"url": self._url})
        except Exception as exc:
            _logger.warning(
                "cache: Redis unreachable, caching disabled",
                extra={"error": str(exc)},
            )
            self._client = None

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None

    # ------------------------------------------------------------------
    # Core get / set / delete
    # ------------------------------------------------------------------

    async def get(self, key: str) -> Any | None:
        if self._client is None:
            return None
        try:
            raw = await self._client.get(key)
            return json.loads(raw) if raw is not None else None
        except Exception as exc:
            _logger.debug("cache.get failed", extra={"key": key, "error": str(exc)})
            return None

    async def set(self, key: str, value: Any, ttl: int = 300) -> None:
        if self._client is None:
            return
        try:
            await self._client.setex(key, ttl, json.dumps(value, default=str))
        except Exception as exc:
            _logger.debug("cache.set failed", extra={"key": key, "error": str(exc)})

    async def delete(self, *keys: str) -> None:
        if self._client is None or not keys:
            return
        try:
            await self._client.delete(*keys)
        except Exception as exc:
            _logger.debug("cache.delete failed", extra={"keys": list(keys), "error": str(exc)})

    async def delete_pattern(self, pattern: str) -> None:
        """Delete all keys matching a Redis glob pattern."""
        if self._client is None:
            return
        try:
            cursor = 0
            while True:
                cursor, found = await self._client.scan(cursor, match=pattern, count=200)
                if found:
                    await self._client.delete(*found)
                if cursor == 0:
                    break
        except Exception as exc:
            _logger.debug(
                "cache.delete_pattern failed",
                extra={"pattern": pattern, "error": str(exc)},
            )
