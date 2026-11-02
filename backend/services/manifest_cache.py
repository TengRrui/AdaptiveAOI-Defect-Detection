from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
import sqlite3

from backend.database import DatabaseManager


class FileManifestCache:
    """Persistent size/mtime keyed SHA-256 cache for managed artifacts."""

    def __init__(self, database: DatabaseManager) -> None:
        self.database = database
        self.hits = 0
        self.misses = 0
        with self.database.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS file_manifest_cache (
                    file_path TEXT PRIMARY KEY,
                    size_bytes INTEGER NOT NULL,
                    modified_ns INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_file_manifest_signature
                  ON file_manifest_cache(size_bytes, modified_ns);
                """
            )
            rows = conn.execute(
                "SELECT file_path,size_bytes,modified_ns,sha256 FROM file_manifest_cache"
            ).fetchall()
        self._entries = {
            str(row["file_path"]): (int(row["size_bytes"]), int(row["modified_ns"]), str(row["sha256"]))
            for row in rows
        }

    def reset_metrics(self) -> None:
        self.hits = 0
        self.misses = 0

    def digest(self, path: str | Path, *, connection: sqlite3.Connection | None = None) -> str:
        resolved = Path(path).resolve()
        stat = resolved.stat()
        key = str(resolved)
        cached = self._entries.get(key)
        if cached is not None and cached[:2] == (int(stat.st_size), int(stat.st_mtime_ns)):
            self.hits += 1
            return cached[2]
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
        def persist(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO file_manifest_cache(file_path,size_bytes,modified_ns,sha256,updated_at) VALUES(?,?,?,?,CURRENT_TIMESTAMP) "
                "ON CONFLICT(file_path) DO UPDATE SET size_bytes=excluded.size_bytes,modified_ns=excluded.modified_ns,sha256=excluded.sha256,updated_at=CURRENT_TIMESTAMP",
                (key, int(stat.st_size), int(stat.st_mtime_ns), value),
            )
        if connection is not None:
            persist(connection)
        else:
            with self.database.connect() as conn:
                persist(conn)
        self._entries[key] = (int(stat.st_size), int(stat.st_mtime_ns), value)
        self.misses += 1
        return value

    def signature_is_current(self, path: str | Path) -> bool:
        """Check a cached artifact signature without reading its contents."""

        try:
            resolved = Path(path).resolve()
            stat = resolved.stat()
        except OSError:
            return False
        cached = self._entries.get(str(resolved))
        return bool(
            cached is not None
            and cached[:2] == (int(stat.st_size), int(stat.st_mtime_ns))
        )

    def metrics(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }
