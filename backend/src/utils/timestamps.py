"""Utilities for handling timestamps consistently across the application."""

from __future__ import annotations

from datetime import UTC, datetime


def get_utc_now() -> datetime:
    """Get current UTC time."""
    return datetime.now(UTC)


def timestamp_iso() -> str:
    """Get current UTC timestamp in ISO 8601 format."""
    return get_utc_now().isoformat()


def datetime_to_iso(dt: datetime) -> str:
    """Convert a datetime object to ISO 8601 string format."""
    return dt.isoformat()
