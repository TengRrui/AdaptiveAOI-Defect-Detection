from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


MIGRATION_ID = "20260825_01_detection_history_runtime_revision"


def apply(conn: sqlite3.Connection) -> None:
    """Add nullable Runtime Revision persistence without rewriting history."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            migration_id TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """
    )
    already_applied = conn.execute(
        "SELECT 1 FROM schema_migrations WHERE migration_id = ?",
        (MIGRATION_ID,),
    ).fetchone()
    columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(detection_results)").fetchall()
    }
    if "runtime_revision" not in columns:
        conn.execute("ALTER TABLE detection_results ADD COLUMN runtime_revision INTEGER")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_detection_runtime_revision ON detection_results(runtime_revision)"
    )
    if not already_applied:
        conn.execute(
            "INSERT INTO schema_migrations(migration_id, applied_at) VALUES(?, ?)",
            (MIGRATION_ID, datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )


__all__ = ["MIGRATION_ID", "apply"]
