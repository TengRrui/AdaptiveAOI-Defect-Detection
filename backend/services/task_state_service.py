from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.database import DatabaseManager


CANONICAL_ACTIVE_STATES = {"pending", "running", "paused"}
CANONICAL_TERMINAL_STATES = {"completed", "failed", "interrupted"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_task_state(raw_status: Any) -> str:
    """Map workflow-specific stages to the durable task state contract."""
    status = str(raw_status or "pending").strip().lower()
    if status in {"queued", "pending"}:
        return "pending"
    if status == "paused":
        return "paused"
    if status in {
        "completed", "complete", "partial", "ready_to_apply", "frozen",
        "applied", "rolled_back", "validated",
    }:
        return "completed"
    if status in {"failed", "error", "rejected"}:
        return "failed"
    if status in {"interrupted", "cancelled", "canceled"}:
        return "interrupted"
    return "running"


class TaskStateService:
    """SQLite-backed source of truth for long-running task lifecycle state."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.database = DatabaseManager(self.path)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return self.database.connect()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS task_states (
                    task_id TEXT PRIMARY KEY,
                    task_kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    runtime_status TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    interrupted_at TEXT,
                    interruption_reason TEXT,
                    parent_task_id TEXT,
                    rerun_count INTEGER NOT NULL DEFAULT 0,
                    idempotency_key TEXT,
                    request_fingerprint TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_task_states_kind_status
                ON task_states(task_kind, status, updated_at DESC);
                CREATE TABLE IF NOT EXISTS task_state_metadata (
                    state_key TEXT PRIMARY KEY,
                    state_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            # Existing installations predate durable idempotency.  SQLite has
            # no ``ADD COLUMN IF NOT EXISTS``, so migrate by inspecting the
            # live schema and keep the operation idempotent.
            columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(task_states)").fetchall()
            }
            if "idempotency_key" not in columns:
                conn.execute("ALTER TABLE task_states ADD COLUMN idempotency_key TEXT")
            if "request_fingerprint" not in columns:
                conn.execute("ALTER TABLE task_states ADD COLUMN request_fingerprint TEXT")
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_task_states_kind_idempotency
                ON task_states(task_kind, idempotency_key)
                WHERE idempotency_key IS NOT NULL AND idempotency_key != ''
                """
            )

    @staticmethod
    def _public(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        item = dict(row)
        try:
            item["payload"] = json.loads(item.pop("payload_json") or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            item["payload"] = {}
        item["rerunnable"] = item.get("status") == "interrupted"
        return item

    def upsert(
        self,
        task_id: str,
        task_kind: str,
        runtime_status: Any,
        payload: dict[str, Any] | None = None,
        *,
        parent_task_id: str | None = None,
        idempotency_key: str | None = None,
        request_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        now = _utc_now()
        raw = str(runtime_status or "pending").strip().lower()
        state = canonical_task_state(raw)
        body = json.dumps(payload or {}, ensure_ascii=False, default=str)
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT created_at, started_at, rerun_count FROM task_states WHERE task_id=?",
                (str(task_id),),
            ).fetchone()
            created_at = str((existing or {}).get("created_at") if isinstance(existing, dict) else existing["created_at"] if existing else now)
            started_at = (
                str(existing["started_at"] or "") if existing else ""
            ) or (now if state == "running" else None)
            rerun_count = int(existing["rerun_count"] or 0) if existing else 0
            finished_at = now if state in CANONICAL_TERMINAL_STATES else None
            conn.execute(
                """
                INSERT INTO task_states(
                    task_id, task_kind, status, runtime_status, payload_json,
                    created_at, updated_at, started_at, finished_at,
                    interrupted_at, interruption_reason, parent_task_id, rerun_count,
                    idempotency_key, request_fingerprint
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(task_id) DO UPDATE SET
                    task_kind=excluded.task_kind,
                    status=excluded.status,
                    runtime_status=excluded.runtime_status,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at,
                    started_at=COALESCE(task_states.started_at, excluded.started_at),
                    finished_at=excluded.finished_at,
                    interrupted_at=CASE WHEN excluded.status='interrupted' THEN excluded.updated_at ELSE task_states.interrupted_at END,
                    parent_task_id=COALESCE(excluded.parent_task_id, task_states.parent_task_id),
                    idempotency_key=COALESCE(task_states.idempotency_key, excluded.idempotency_key),
                    request_fingerprint=COALESCE(task_states.request_fingerprint, excluded.request_fingerprint)
                """,
                (
                    str(task_id), str(task_kind), state, raw, body,
                    created_at, now, started_at, finished_at,
                    now if state == "interrupted" else None,
                    None, parent_task_id, rerun_count,
                    str(idempotency_key or "") or None,
                    str(request_fingerprint or "") or None,
                ),
            )
            row = conn.execute("SELECT * FROM task_states WHERE task_id=?", (str(task_id),)).fetchone()
        return self._public(row)

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM task_states WHERE task_id=?", (str(task_id),)).fetchone()
        return self._public(row) if row else None

    def find_by_idempotency(self, task_kind: str, idempotency_key: str) -> dict[str, Any] | None:
        key = str(idempotency_key or "").strip()
        if not key:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM task_states WHERE task_kind=? AND idempotency_key=? LIMIT 1",
                (str(task_kind), key),
            ).fetchone()
        return self._public(row) if row else None

    def claim_idempotent_task(
        self,
        task_id: str,
        task_kind: str,
        runtime_status: Any,
        payload: dict[str, Any] | None,
        *,
        idempotency_key: str,
        request_fingerprint: str,
        parent_task_id: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Atomically claim an idempotent task submission.

        A read followed by ``upsert`` is subject to a cross-process race: two
        API workers can both observe no row and one then leaks a raw UNIQUE
        constraint error.  ``BEGIN IMMEDIATE`` serializes the lookup and insert
        at SQLite's write boundary.  The returned boolean is true only for the
        process that is allowed to enqueue the background worker.
        """

        key = str(idempotency_key or "").strip()
        if not key:
            raise ValueError("idempotency_key is required for an atomic task claim")
        now = _utc_now()
        raw = str(runtime_status or "pending").strip().lower()
        state = canonical_task_state(raw)
        body = json.dumps(payload or {}, ensure_ascii=False, default=str)
        with self._lock, self.database.transaction(immediate=True) as conn:
            existing = conn.execute(
                "SELECT * FROM task_states WHERE task_kind=? AND idempotency_key=? LIMIT 1",
                (str(task_kind), key),
            ).fetchone()
            if existing is not None:
                return self._public(existing), False
            conn.execute(
                """
                INSERT INTO task_states(
                    task_id, task_kind, status, runtime_status, payload_json,
                    created_at, updated_at, started_at, finished_at,
                    interrupted_at, interruption_reason, parent_task_id, rerun_count,
                    idempotency_key, request_fingerprint
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    str(task_id), str(task_kind), state, raw, body,
                    now, now, now if state == "running" else None,
                    now if state in CANONICAL_TERMINAL_STATES else None,
                    now if state == "interrupted" else None,
                    None, parent_task_id, 0, key,
                    str(request_fingerprint or "") or None,
                ),
            )
            claimed = conn.execute(
                "SELECT * FROM task_states WHERE task_id=?",
                (str(task_id),),
            ).fetchone()
        return self._public(claimed), True

    def latest(self, task_kind: str | None = None) -> dict[str, Any] | None:
        query = "SELECT * FROM task_states"
        values: tuple[Any, ...] = ()
        if task_kind:
            query += " WHERE task_kind=?"
            values = (str(task_kind),)
        query += " ORDER BY updated_at DESC LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(query, values).fetchone()
        return self._public(row) if row else None

    def list(
        self,
        task_kind: str | None = None,
        *,
        statuses: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return durable task snapshots for lifecycle/protection checks.

        The domain-draft delete flow must be able to protect a draft even
        when the in-memory worker object is unavailable (for example while a
        process is recovering).  Keeping this read-only query here avoids
        making callers inspect the SQLite schema directly.
        """
        clauses: list[str] = []
        values: list[Any] = []
        if task_kind:
            clauses.append("task_kind=?")
            values.append(str(task_kind))
        if statuses:
            normalized = [str(value).strip().lower() for value in statuses if str(value).strip()]
            if normalized:
                placeholders = ",".join("?" for _ in normalized)
                clauses.append(f"status IN ({placeholders})")
                values.extend(normalized)
        query = "SELECT * FROM task_states"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC"
        with self._connect() as conn:
            rows = conn.execute(query, tuple(values)).fetchall()
        return [self._public(row) for row in rows]

    def known_task_ids(self) -> set[str]:
        """Return indexed task ids without decoding every persisted payload.

        Startup reconciliation only needs to distinguish already-indexed task
        directories from legacy manifests.  Reading ``payload_json`` for every
        completed task made cold startup proportional to the full task history,
        even though almost all rows required no work.
        """

        with self._connect() as conn:
            rows = conn.execute("SELECT task_id FROM task_states").fetchall()
        return {str(row["task_id"]) for row in rows if str(row["task_id"] or "")}

    def legacy_manifest_reconciliation_complete(self) -> bool:
        """Return whether the one-time pre-SQLite manifest migration finished."""

        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_value FROM task_state_metadata WHERE state_key=?",
                ("legacy_manifest_reconciliation_complete",),
            ).fetchone()
        return bool(row and str(row["state_value"] or "").lower() == "true")

    def mark_legacy_manifest_reconciliation_complete(self) -> None:
        now = _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO task_state_metadata(state_key,state_value,updated_at)
                VALUES(?,?,?)
                ON CONFLICT(state_key) DO UPDATE SET
                    state_value=excluded.state_value,
                    updated_at=excluded.updated_at
                """,
                ("legacy_manifest_reconciliation_complete", "true", now),
            )

    def mark_interrupted(self, task_id: str, reason: str = "service_restart") -> dict[str, Any] | None:
        now = _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE task_states
                SET status='interrupted', runtime_status='interrupted', updated_at=?,
                    finished_at=?, interrupted_at=?, interruption_reason=?
                WHERE task_id=? AND status IN ('pending','running','paused')
                """,
                (now, now, now, str(reason), str(task_id)),
            )
            row = conn.execute("SELECT * FROM task_states WHERE task_id=?", (str(task_id),)).fetchone()
        return self._public(row) if row else None

    def recover_unfinished(self, reason: str = "service_restart") -> list[str]:
        now = _utc_now()
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT task_id FROM task_states WHERE status IN ('pending','running','paused')"
            ).fetchall()
            ids = [str(row["task_id"]) for row in rows]
            if ids:
                conn.execute(
                    """
                    UPDATE task_states
                    SET status='interrupted', runtime_status='interrupted', updated_at=?,
                        finished_at=?, interrupted_at=?, interruption_reason=?
                    WHERE status IN ('pending','running','paused')
                    """,
                    (now, now, now, str(reason)),
                )
        return ids

    def record_rerun(self, source_task_id: str, new_task_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE task_states SET rerun_count=rerun_count+1, updated_at=? WHERE task_id=?",
                (_utc_now(), str(source_task_id)),
            )
            conn.execute(
                "UPDATE task_states SET parent_task_id=? WHERE task_id=?",
                (str(source_task_id), str(new_task_id)),
            )
