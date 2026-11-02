"""Shared timestamp parsing and Beijing-time presentation helpers.

Storage timestamps in AdaptiveAOI remain UTC, timezone-aware ISO-8601
strings.  This module is deliberately presentation-only: it never rewrites
the persisted value and never changes ordering semantics.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo


BEIJING_TIMEZONE = ZoneInfo("Asia/Shanghai")
UTC = timezone.utc
BEIJING_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def parse_timestamp(value: Any, *, naive_timezone=UTC) -> datetime | None:
    """Parse supported persisted/legacy values into an aware UTC datetime.

    Naive values are treated as UTC because all backend-created timestamps
    are UTC; this explicit assumption avoids silently applying +08 twice.
    Callers that receive a known local-naive value can pass its timezone.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, datetime.min.time())
    elif isinstance(value, (int, float)):
        try:
            parsed = datetime.fromtimestamp(float(value), tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    else:
        text = str(value).strip()
        if not text:
            return None
        normalized = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
        parsed = None
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            for fmt in (
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
                "%Y/%m/%d %H:%M:%S",
                "%Y-%m-%d",
            ):
                try:
                    parsed = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            if parsed is None:
                return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=naive_timezone)
    return parsed.astimezone(UTC)


def format_beijing_datetime(
    value: Any,
    *,
    fallback: str = "—",
    naive_timezone=UTC,
    include_timezone: bool = False,
) -> str:
    """Return a user-facing timestamp in Asia/Shanghai.

    The default format is ``YYYY-MM-DD HH:mm:ss``.  ``include_timezone`` is
    useful for an occasional explicit page caption, while ordinary tables
    should use the compact default.
    """
    parsed = parse_timestamp(value, naive_timezone=naive_timezone)
    if parsed is None:
        return fallback
    rendered = parsed.astimezone(BEIJING_TIMEZONE).strftime(BEIJING_DATETIME_FORMAT)
    return f"{rendered}（北京时间）" if include_timezone else rendered


__all__ = [
    "BEIJING_DATETIME_FORMAT",
    "BEIJING_TIMEZONE",
    "format_beijing_datetime",
    "parse_timestamp",
]
