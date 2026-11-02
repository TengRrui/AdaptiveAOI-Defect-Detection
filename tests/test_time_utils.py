from __future__ import annotations

from datetime import datetime, timezone

from app.time_utils import format_beijing_datetime, parse_timestamp


def test_utc_iso_is_rendered_in_beijing_without_double_offset() -> None:
    assert format_beijing_datetime("2026-08-16T01:37:52+00:00") == "2026-08-16 09:37:52"
    assert format_beijing_datetime("2026-08-16T09:37:52+08:00") == "2026-08-16 09:37:52"


def test_aware_datetime_and_sqlite_naive_value() -> None:
    aware = datetime(2026, 8, 16, 1, 37, 52, tzinfo=timezone.utc)
    assert format_beijing_datetime(aware) == "2026-08-16 09:37:52"
    # Backend-created SQLite legacy timestamps are UTC when no offset exists.
    assert format_beijing_datetime("2026-08-16 01:37:52") == "2026-08-16 09:37:52"
    assert parse_timestamp("2026/08/16 01:37:52") is not None


def test_timestamp_fallbacks_and_numeric_epoch() -> None:
    assert format_beijing_datetime(None) == "—"
    assert format_beijing_datetime("not-a-timestamp", fallback="未记录") == "未记录"
    assert format_beijing_datetime(0) == "1970-01-01 08:00:00"
