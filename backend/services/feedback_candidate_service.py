"""Audited candidate-pool storage for feedback-driven optimization.

This service is intentionally separate from :mod:`result_storage`: detection
rows are immutable, while feedback candidates and their review/annotation
history are append-or-transition records.  All values returned by this module
are browser-safe (absolute source paths stay server-side).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import os
import secrets
import shutil
import sqlite3
import time
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from backend.errors import ApiError
from backend.database import DatabaseManager
from backend.services.detection_record_protection import (
    candidate_protection_codes,
    candidate_source_ids,
)
from backend.services.feedback_task_status import ACTIVE_TASK_STATUSES


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(value: Any, default: Any) -> Any:
    try:
        parsed = json.loads(value or "") if isinstance(value, str) else value
        return parsed if parsed is not None else default
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _task_started_at(task_id: Any) -> str:
    """Recover the task start time embedded by the formal job-id factory.

    Older feedback rows may outlive their detection-result row or job manifest.
    In that case the task id remains the only original detection-task clock.  It
    is deliberately treated as a *task* time, never as the feedback time.
    """
    parts = str(task_id or "").split("_")
    for index in range(len(parts) - 1):
        date_part, time_part = parts[index], parts[index + 1]
        if len(date_part) != 8 or len(time_part) != 6 or not (date_part + time_part).isdigit():
            continue
        try:
            local = datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S").replace(
                tzinfo=ZoneInfo("Asia/Shanghai")
            )
            return local.astimezone(timezone.utc).isoformat(timespec="seconds")
        except ValueError:
            continue
    return ""


class FeedbackCandidateService:
    """Candidate pool, human review, mask revisions and audit history."""

    # These groups are the single public statistics vocabulary used by both
    # the feedback center and the statistics dashboard.  Keep audit-only
    # ``deleted`` rows outside every group: they exist solely for version
    # traceability after the operator has removed the feedback record.
    _PENDING_STATUSES = frozenset({
        "collected", "legacy_unresolved", "pending_review", "in_review",
        "conflict", "record_only",
    })
    _APPROVED_STATUSES = frozenset({"accepted"})
    _WITHDRAWN_STATUSES = frozenset({
        "withdrawn", "rejected", "archived", "cancelled", "invalid_source",
    })
    _USED_STATUSES = frozenset({
        "queued", "running", "optimizing", "ready_to_apply", "consumed",
        "validation_failed", "frozen", "applied", "rolled_back", "completed",
    })
    # Legacy releases stored task progress in the review row itself.  These
    # values still represent an approved human correction; reusable evidence
    # is now tracked per task by feedback_optimization_items.
    _REUSABLE_REVIEW_STATUSES = frozenset({
        "accepted", "queued", "running", "optimizing", "ready_to_apply",
        "consumed", "validation_failed", "frozen", "applied", "rolled_back",
        "completed",
    })

    # The candidate state is intentionally explicit.  Optimization workers
    # and review requests must use the same transition table so a stale UI
    # request cannot jump directly from a review state to an applied profile.
    _TRANSITIONS: dict[str, set[str]] = {
        "collected": {"pending_review", "rejected", "withdrawn", "archived"},
        "legacy_unresolved": {"pending_review", "rejected", "withdrawn", "archived"},
        "pending_review": {"in_review", "accepted", "rejected", "conflict", "withdrawn", "archived"},
        "in_review": {"pending_review", "accepted", "rejected", "conflict", "withdrawn", "archived"},
        "conflict": {"pending_review", "accepted", "rejected", "withdrawn", "archived"},
        "accepted": {"pending_review", "rejected", "queued", "ready_to_apply", "applied", "rolled_back", "archived"},
        "ready_to_apply": {"consumed", "applied", "validation_failed", "accepted", "rolled_back"},
        "queued": {"accepted", "optimizing", "validation_failed", "rejected", "invalid_source"},
        "optimizing": {"accepted", "validation_failed", "queued", "ready_to_apply", "consumed", "invalid_source"},
        "consumed": {"applied", "archived"},
        "validation_failed": {"accepted", "rejected", "queued"},
        "applied": {"rolled_back", "archived"},
        "rolled_back": set(),
        # ``record_only`` describes application policy, not an absence of a
        # human-review workflow.  Operators may correct and review these rows;
        # the immutable context flag still prevents optimization or profile
        # mutation in FeedbackOptimizationService.
        "record_only": {"pending_review", "in_review", "accepted", "rejected", "withdrawn"},
        "rejected": {"pending_review", "archived", "withdrawn"},
        "withdrawn": {"archived"},
        "invalid_source": {"pending_review", "rejected", "withdrawn"},
        "archived": set(),
        "deleted": set(),
    }

    def __init__(self, settings: Any, storage: Any, logger: Any | None = None):
        self.settings = settings
        self.storage = storage
        self.logger = logger
        self.path = storage.path
        self.database = getattr(storage, "database", DatabaseManager(self.path))
        self.settings.ensure_runtime_dirs()
        # ResultStorage creates the schema first.  Calling this is harmless for
        # older installations whose database was created before this feature.
        self._ensure_schema()
        self._migrate_legacy_candidates()

    def _connect(self) -> sqlite3.Connection:
        database = getattr(self, "database", None)
        if database is None:
            database = DatabaseManager(self.path)
            self.database = database
        return database.connect()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS feedback_candidates (
                    candidate_id TEXT PRIMARY KEY, feedback_id TEXT NOT NULL,
                    feedback_key TEXT NOT NULL UNIQUE, mode TEXT NOT NULL,
                    task_id TEXT NOT NULL, sample_id TEXT NOT NULL, result_id TEXT,
                    frame_index INTEGER, timestamp_ms REAL, dataset TEXT NOT NULL,
                    category TEXT NOT NULL, model_version TEXT, config_version TEXT,
                    config_profile_id TEXT, config_profile_version INTEGER,
                    base_profile_id TEXT, base_memory_bank_id TEXT, base_config_version INTEGER,
                    memory_bank_id TEXT, method_mode TEXT, inference_mode TEXT,
                    original_prediction TEXT, original_score REAL, threshold REAL,
                    corrected_label TEXT, corrected_defect_type TEXT, assessment TEXT,
                    issue_tags_json TEXT NOT NULL DEFAULT '[]', note TEXT,
                    status TEXT NOT NULL, conflict_status TEXT NOT NULL DEFAULT 'none',
                    priority_score REAL NOT NULL DEFAULT 0, priority_reason TEXT NOT NULL DEFAULT '',
                    source_artifact_ref TEXT, source_sha256 TEXT, perceptual_hash TEXT, reviewer TEXT,
                    review_note TEXT, reviewed_at TEXT, created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL, context_json TEXT NOT NULL,
                    candidate_version INTEGER NOT NULL DEFAULT 1,
                    source_snapshot_ref TEXT, source_snapshot_sha256 TEXT,
                    source_snapshot_json TEXT NOT NULL DEFAULT '{}',
                    protection_released_at TEXT, snapshot_error TEXT,
                    detection_time TEXT, feedback_submitted_at TEXT,
                    deleted_at TEXT, audit_only INTEGER NOT NULL DEFAULT 0,
                    source_memory_bank_deleted_at TEXT
                );
                CREATE TABLE IF NOT EXISTS feedback_delete_plans (
                    plan_id TEXT PRIMARY KEY,
                    preview_token_hash TEXT NOT NULL,
                    candidate_ids_json TEXT NOT NULL,
                    plan_hash TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    consumed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS feedback_annotations (
                    annotation_id TEXT PRIMARY KEY, candidate_id TEXT NOT NULL,
                    revision INTEGER NOT NULL, annotation_type TEXT NOT NULL,
                    mask_path TEXT, mask_artifact_ref TEXT, mask_sha256 TEXT, mask_width INTEGER,
                    mask_height INTEGER, mask_format TEXT, bbox_json TEXT, polygon_json TEXT, note TEXT,
                    created_by TEXT, created_at TEXT NOT NULL, updated_at TEXT,
                    UNIQUE(candidate_id, revision)
                );
                CREATE TABLE IF NOT EXISTS feedback_audit_log (
                    audit_id TEXT PRIMARY KEY, candidate_id TEXT, job_id TEXT,
                    action TEXT NOT NULL, actor TEXT NOT NULL, from_status TEXT,
                    to_status TEXT, details_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback_optimization_jobs (
                    job_id TEXT PRIMARY KEY, base_profile_id TEXT NOT NULL,
                    base_config_profile_id TEXT, base_config_profile_version INTEGER,
                    candidate_ids_json TEXT NOT NULL, candidate_count INTEGER NOT NULL DEFAULT 0,
                    strategy TEXT NOT NULL, objective TEXT NOT NULL,
                    update_threshold INTEGER NOT NULL DEFAULT 1,
                    update_normal_memory INTEGER NOT NULL DEFAULT 0,
                    update_defect_memory INTEGER NOT NULL DEFAULT 0,
                    update_lightweight_calibrator INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0,
                    stage TEXT NOT NULL DEFAULT 'queued', message TEXT NOT NULL DEFAULT '',
                    constraints_json TEXT NOT NULL DEFAULT '{}', candidate_profile_id TEXT,
                    candidate_config_profile_id TEXT, candidate_config_profile_version INTEGER,
                    metrics_before_json TEXT NOT NULL DEFAULT '{}', metrics_after_json TEXT NOT NULL DEFAULT '{}',
                    complexity_before_json TEXT NOT NULL DEFAULT '{}', complexity_after_json TEXT NOT NULL DEFAULT '{}',
                    validation_json TEXT NOT NULL DEFAULT '{}', error_code TEXT, error_message TEXT,
                    created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, applied_at TEXT,
                    rolled_back_at TEXT, previous_profile_id TEXT, previous_config_profile_id TEXT,
                    previous_config_profile_version INTEGER,
                    idempotency_key TEXT NOT NULL DEFAULT '', current_stage_id TEXT NOT NULL DEFAULT '',
                    stages_json TEXT NOT NULL DEFAULT '[]', timings_json TEXT NOT NULL DEFAULT '{}',
                    stage_started_at TEXT, last_progress_at TEXT, heartbeat_at TEXT
                );
                CREATE TABLE IF NOT EXISTS feedback_optimization_items (
                    job_id TEXT NOT NULL, candidate_id TEXT NOT NULL, role TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued', message TEXT NOT NULL DEFAULT '',
                    result_code TEXT NOT NULL DEFAULT '', components_json TEXT NOT NULL DEFAULT '[]',
                    started_at TEXT, finished_at TEXT, updated_at TEXT,
                    source_memory_bank_id TEXT, target_memory_bank_id TEXT,
                    target_memory_bank_version TEXT, strategy TEXT, used_at TEXT,
                    usage_result TEXT, generated_profile_id TEXT,
                    PRIMARY KEY(job_id, candidate_id)
                );
                CREATE TABLE IF NOT EXISTS feedback_state (
                    singleton_id INTEGER PRIMARY KEY CHECK(singleton_id=1),
                    revision INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT,
                    reason TEXT
                );
                INSERT OR IGNORE INTO feedback_state(singleton_id,revision,updated_at,reason)
                VALUES(1,0,CURRENT_TIMESTAMP,'schema_initialized');
                """
            )
            migrations = {
                "feedback_candidates": {
                    "feedback_id": "TEXT",
                    "feedback_key": "TEXT",
                    "mode": "TEXT",
                    "task_id": "TEXT",
                    "sample_id": "TEXT",
                    "result_id": "TEXT",
                    "frame_index": "INTEGER",
                    "timestamp_ms": "REAL",
                    "dataset": "TEXT",
                    "category": "TEXT",
                    "model_version": "TEXT",
                    "config_version": "TEXT",
                    "config_profile_id": "TEXT",
                    "config_profile_version": "INTEGER",
                    "base_profile_id": "TEXT",
                    "base_memory_bank_id": "TEXT",
                    "base_config_version": "INTEGER",
                    "memory_bank_id": "TEXT",
                    "method_mode": "TEXT",
                    "inference_mode": "TEXT",
                    "original_prediction": "TEXT",
                    "original_score": "REAL",
                    "threshold": "REAL",
                    "corrected_label": "TEXT",
                    "corrected_defect_type": "TEXT",
                    "assessment": "TEXT",
                    "issue_tags_json": "TEXT DEFAULT '[]'",
                    "note": "TEXT",
                    "status": "TEXT DEFAULT 'pending_review'",
                    "conflict_status": "TEXT DEFAULT 'none'",
                    "priority_score": "REAL DEFAULT 0",
                    "priority_reason": "TEXT DEFAULT ''",
                    "source_artifact_ref": "TEXT",
                    "source_sha256": "TEXT",
                    "perceptual_hash": "TEXT",
                    "reviewer": "TEXT",
                    "review_note": "TEXT",
                    "reviewed_at": "TEXT",
                    "created_at": "TEXT",
                    "updated_at": "TEXT",
                    "context_json": "TEXT DEFAULT '{}'",
                    "candidate_version": "INTEGER NOT NULL DEFAULT 1",
                    "source_snapshot_ref": "TEXT",
                    "source_snapshot_sha256": "TEXT",
                    "source_snapshot_json": "TEXT DEFAULT '{}'",
                    "protection_released_at": "TEXT",
                    "snapshot_error": "TEXT",
                    "detection_time": "TEXT",
                    "feedback_submitted_at": "TEXT",
                    "deleted_at": "TEXT",
                    "audit_only": "INTEGER NOT NULL DEFAULT 0",
                    "source_memory_bank_deleted_at": "TEXT",
                },
                "feedback_annotations": {
                    "candidate_id": "TEXT",
                    "revision": "INTEGER DEFAULT 1",
                    "annotation_type": "TEXT DEFAULT 'mask'",
                    "mask_path": "TEXT",
                    "mask_artifact_ref": "TEXT",
                    "mask_sha256": "TEXT",
                    "mask_width": "INTEGER",
                    "mask_height": "INTEGER",
                    "mask_format": "TEXT",
                    "bbox_json": "TEXT",
                    "polygon_json": "TEXT",
                    "note": "TEXT",
                    "created_by": "TEXT",
                    "created_at": "TEXT",
                    "updated_at": "TEXT",
                },
                "feedback_optimization_jobs": {
                    "base_profile_id": "TEXT",
                    "base_config_profile_id": "TEXT",
                    "base_config_profile_version": "INTEGER",
                    "candidate_ids_json": "TEXT DEFAULT '[]'",
                    "candidate_count": "INTEGER NOT NULL DEFAULT 0",
                    "strategy": "TEXT DEFAULT 'safe_incremental'",
                    "objective": "TEXT DEFAULT 'f1'",
                    "update_threshold": "INTEGER NOT NULL DEFAULT 1",
                    "update_normal_memory": "INTEGER NOT NULL DEFAULT 0",
                    "update_defect_memory": "INTEGER NOT NULL DEFAULT 0",
                    "update_lightweight_calibrator": "INTEGER NOT NULL DEFAULT 0",
                    "status": "TEXT DEFAULT 'queued'",
                    "progress": "REAL DEFAULT 0",
                    "stage": "TEXT DEFAULT 'queued'",
                    "message": "TEXT DEFAULT ''",
                    "constraints_json": "TEXT DEFAULT '{}'",
                    "candidate_profile_id": "TEXT",
                    "candidate_config_profile_id": "TEXT",
                    "candidate_config_profile_version": "INTEGER",
                    "metrics_before_json": "TEXT DEFAULT '{}'",
                    "metrics_after_json": "TEXT DEFAULT '{}'",
                    "complexity_before_json": "TEXT NOT NULL DEFAULT '{}'",
                    "complexity_after_json": "TEXT NOT NULL DEFAULT '{}'",
                    "validation_json": "TEXT DEFAULT '{}'",
                    "error_code": "TEXT",
                    "error_message": "TEXT",
                    "created_at": "TEXT",
                    "started_at": "TEXT",
                    "finished_at": "TEXT",
                    "applied_at": "TEXT",
                    "rolled_back_at": "TEXT",
                    "previous_profile_id": "TEXT",
                    "previous_config_profile_id": "TEXT",
                    "previous_config_profile_version": "INTEGER",
                    "idempotency_key": "TEXT NOT NULL DEFAULT ''",
                    "current_stage_id": "TEXT NOT NULL DEFAULT ''",
                    "stages_json": "TEXT NOT NULL DEFAULT '[]'",
                    "timings_json": "TEXT NOT NULL DEFAULT '{}'",
                    "stage_started_at": "TEXT",
                    "last_progress_at": "TEXT",
                    "heartbeat_at": "TEXT",
                },
                "feedback_optimization_items": {
                    "job_id": "TEXT",
                    "candidate_id": "TEXT",
                    "role": "TEXT DEFAULT 'unresolved'",
                    "status": "TEXT DEFAULT 'queued'",
                    "message": "TEXT DEFAULT ''",
                    "result_code": "TEXT NOT NULL DEFAULT ''",
                    "components_json": "TEXT NOT NULL DEFAULT '[]'",
                    "started_at": "TEXT",
                    "finished_at": "TEXT",
                    "updated_at": "TEXT",
                    "source_memory_bank_id": "TEXT",
                    "target_memory_bank_id": "TEXT",
                    "target_memory_bank_version": "TEXT",
                    "strategy": "TEXT",
                    "used_at": "TEXT",
                    "usage_result": "TEXT",
                    "generated_profile_id": "TEXT",
                },
                "feedback_audit_log": {
                    "candidate_id": "TEXT",
                    "job_id": "TEXT",
                    "action": "TEXT DEFAULT 'legacy'",
                    "actor": "TEXT DEFAULT 'system'",
                    "from_status": "TEXT",
                    "to_status": "TEXT",
                    "details_json": "TEXT DEFAULT '{}'",
                    "created_at": "TEXT",
                },
            }
            for table, columns in migrations.items():
                existing = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
                for name, definition in columns.items():
                    if name not in existing:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
            # Indexes are deliberately created only after the additive
            # migrations.  Older installations may have a partial table with
            # no ``status``/``updated_at`` columns; creating the index before
            # migration would make application startup fail.
            conn.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_feedback_candidate_status ON feedback_candidates(status, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_feedback_candidate_profile ON feedback_candidates(memory_bank_id, dataset, category);
                CREATE INDEX IF NOT EXISTS idx_feedback_candidate_memory_status ON feedback_candidates(memory_bank_id, status, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_feedback_candidate_mode_updated ON feedback_candidates(mode, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_feedback_candidate_detection_time ON feedback_candidates(detection_time DESC);
                CREATE INDEX IF NOT EXISTS idx_feedback_annotation_candidate ON feedback_annotations(candidate_id, revision DESC);
                CREATE INDEX IF NOT EXISTS idx_feedback_optimization_status ON feedback_optimization_jobs(status, created_at DESC);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_optimization_item_unique ON feedback_optimization_items(job_id, candidate_id);
                CREATE INDEX IF NOT EXISTS idx_feedback_optimization_item_candidate_usage ON feedback_optimization_items(candidate_id, used_at DESC);
                CREATE INDEX IF NOT EXISTS idx_feedback_audit_candidate ON feedback_audit_log(candidate_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_feedback_audit_job ON feedback_audit_log(job_id, created_at DESC);
                CREATE TRIGGER IF NOT EXISTS feedback_revision_candidate_insert
                AFTER INSERT ON feedback_candidates BEGIN
                    UPDATE feedback_state SET revision=revision+1,updated_at=CURRENT_TIMESTAMP,reason='candidate_insert' WHERE singleton_id=1;
                END;
                CREATE TRIGGER IF NOT EXISTS feedback_revision_candidate_update
                AFTER UPDATE ON feedback_candidates BEGIN
                    UPDATE feedback_state SET revision=revision+1,updated_at=CURRENT_TIMESTAMP,reason='candidate_update' WHERE singleton_id=1;
                END;
                CREATE TRIGGER IF NOT EXISTS feedback_revision_candidate_delete
                AFTER DELETE ON feedback_candidates BEGIN
                    UPDATE feedback_state SET revision=revision+1,updated_at=CURRENT_TIMESTAMP,reason='candidate_delete' WHERE singleton_id=1;
                END;
                CREATE TRIGGER IF NOT EXISTS feedback_revision_annotation_insert
                AFTER INSERT ON feedback_annotations BEGIN
                    UPDATE feedback_state SET revision=revision+1,updated_at=CURRENT_TIMESTAMP,reason='annotation_insert' WHERE singleton_id=1;
                END;
                CREATE TRIGGER IF NOT EXISTS feedback_revision_annotation_update
                AFTER UPDATE ON feedback_annotations BEGIN
                    UPDATE feedback_state SET revision=revision+1,updated_at=CURRENT_TIMESTAMP,reason='annotation_update' WHERE singleton_id=1;
                END;
                CREATE TRIGGER IF NOT EXISTS feedback_revision_annotation_delete
                AFTER DELETE ON feedback_annotations BEGIN
                    UPDATE feedback_state SET revision=revision+1,updated_at=CURRENT_TIMESTAMP,reason='annotation_delete' WHERE singleton_id=1;
                END;
                CREATE TRIGGER IF NOT EXISTS feedback_revision_job_insert
                AFTER INSERT ON feedback_optimization_jobs BEGIN
                    UPDATE feedback_state SET revision=revision+1,updated_at=CURRENT_TIMESTAMP,reason='optimization_insert' WHERE singleton_id=1;
                END;
                CREATE TRIGGER IF NOT EXISTS feedback_revision_job_update
                AFTER UPDATE ON feedback_optimization_jobs BEGIN
                    UPDATE feedback_state SET revision=revision+1,updated_at=CURRENT_TIMESTAMP,reason='optimization_update' WHERE singleton_id=1;
                END;
                CREATE TRIGGER IF NOT EXISTS feedback_revision_job_delete
                AFTER DELETE ON feedback_optimization_jobs BEGIN
                    UPDATE feedback_state SET revision=revision+1,updated_at=CURRENT_TIMESTAMP,reason='optimization_delete' WHERE singleton_id=1;
                END;
                CREATE TRIGGER IF NOT EXISTS feedback_statistics_candidate_insert
                AFTER INSERT ON feedback_candidates BEGIN
                    UPDATE statistics_state SET revision=revision+1,updated_at=CURRENT_TIMESTAMP,reason='feedback_candidate_insert' WHERE singleton_id=1;
                END;
                CREATE TRIGGER IF NOT EXISTS feedback_statistics_candidate_update
                AFTER UPDATE ON feedback_candidates BEGIN
                    UPDATE statistics_state SET revision=revision+1,updated_at=CURRENT_TIMESTAMP,reason='feedback_candidate_update' WHERE singleton_id=1;
                END;
                CREATE TRIGGER IF NOT EXISTS feedback_statistics_candidate_delete
                AFTER DELETE ON feedback_candidates BEGIN
                    UPDATE statistics_state SET revision=revision+1,updated_at=CURRENT_TIMESTAMP,reason='feedback_candidate_delete' WHERE singleton_id=1;
                END;
                CREATE TRIGGER IF NOT EXISTS feedback_statistics_annotation_insert
                AFTER INSERT ON feedback_annotations BEGIN
                    UPDATE statistics_state SET revision=revision+1,updated_at=CURRENT_TIMESTAMP,reason='feedback_annotation_insert' WHERE singleton_id=1;
                END;
                CREATE TRIGGER IF NOT EXISTS feedback_statistics_annotation_update
                AFTER UPDATE ON feedback_annotations BEGIN
                    UPDATE statistics_state SET revision=revision+1,updated_at=CURRENT_TIMESTAMP,reason='feedback_annotation_update' WHERE singleton_id=1;
                END;
                CREATE TRIGGER IF NOT EXISTS feedback_statistics_annotation_delete
                AFTER DELETE ON feedback_annotations BEGIN
                    UPDATE statistics_state SET revision=revision+1,updated_at=CURRENT_TIMESTAMP,reason='feedback_annotation_delete' WHERE singleton_id=1;
                END;
                CREATE TRIGGER IF NOT EXISTS feedback_statistics_job_insert
                AFTER INSERT ON feedback_optimization_jobs BEGIN
                    UPDATE statistics_state SET revision=revision+1,updated_at=CURRENT_TIMESTAMP,reason='feedback_optimization_insert' WHERE singleton_id=1;
                END;
                CREATE TRIGGER IF NOT EXISTS feedback_statistics_job_update
                AFTER UPDATE ON feedback_optimization_jobs BEGIN
                    UPDATE statistics_state SET revision=revision+1,updated_at=CURRENT_TIMESTAMP,reason='feedback_optimization_update' WHERE singleton_id=1;
                END;
                CREATE TRIGGER IF NOT EXISTS feedback_statistics_job_delete
                AFTER DELETE ON feedback_optimization_jobs BEGIN
                    UPDATE statistics_state SET revision=revision+1,updated_at=CURRENT_TIMESTAMP,reason='feedback_optimization_delete' WHERE singleton_id=1;
                END;
                """
            )
            conn.execute(
                """UPDATE feedback_candidates
                   SET feedback_submitted_at=created_at
                   WHERE feedback_submitted_at IS NULL OR feedback_submitted_at=''"""
            )
            conn.execute(
                """UPDATE feedback_candidates
                   SET detection_time=COALESCE(
                       detection_time,
                       (SELECT created_at FROM detection_results r WHERE r.result_id=feedback_candidates.result_id)
                   )
                   WHERE detection_time IS NULL OR detection_time=''"""
            )
            # Audit-only rows keep their candidate_id and usage provenance,
            # but deleting a feedback record must release the active feedback
            # identity so the same detection result can be submitted again.
            conn.execute(
                """UPDATE feedback_candidates
                   SET feedback_key='audit:' || candidate_id || ':' || feedback_key
                   WHERE (status='deleted' OR audit_only=1)
                     AND feedback_key NOT LIKE 'audit:%'"""
            )
            # Bring legacy image feedback onto result-scoped identities.  This
            # preserves repeated-click idempotency after upgrading while still
            # allowing identical bytes in a later detection result.
            conn.execute(
                """UPDATE inspection_feedback
                   SET feedback_key='result:' || result_id
                   WHERE result_id IS NOT NULL AND result_id<>''
                     AND feedback_key<>'result:' || result_id"""
            )
            conn.execute(
                """UPDATE feedback_candidates
                   SET feedback_key='result:' || result_id
                   WHERE status<>'deleted' AND audit_only=0
                     AND result_id IS NOT NULL AND result_id<>''
                     AND feedback_key<>'result:' || result_id"""
            )
            conn.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS idx_inspection_feedback_result
                   ON inspection_feedback(result_id)
                   WHERE result_id IS NOT NULL AND result_id<>''"""
            )
            conn.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_candidate_active_result
                   ON feedback_candidates(result_id)
                   WHERE status<>'deleted' AND audit_only=0
                     AND result_id IS NOT NULL AND result_id<>''"""
            )

    def revision(self) -> dict[str, Any]:
        """Return the durable cache revision for feedback-facing data."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT revision,updated_at,reason FROM feedback_state WHERE singleton_id=1"
            ).fetchone()
        return {
            "revision": int(row["revision"] or 0) if row else 0,
            "updated_at": row["updated_at"] if row else None,
            "reason": row["reason"] if row else None,
        }

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["issue_tags"] = _json(item.pop("issue_tags_json", "[]"), [])
        item["context"] = _json(item.pop("context_json", "{}"), {})
        item["source_snapshot"] = _json(item.pop("source_snapshot_json", "{}"), {})
        item["candidate_version"] = int(item.get("candidate_version") or 1)
        # The backend optimization worker may use ``_source_path``.  API
        # routes call ``public_candidate`` before returning this object so the
        # internal path never reaches the browser.
        return item

    @staticmethod
    def _usage_history_by_candidate(
        conn: sqlite3.Connection,
        candidate_ids: Iterable[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Return durable many-to-many usage records for reviewed samples."""
        ids = list(dict.fromkeys(str(value) for value in candidate_ids if str(value)))
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = conn.execute(
            f"""
            SELECT i.candidate_id,i.job_id,
                   COALESCE(NULLIF(i.source_memory_bank_id,''),j.base_profile_id,'') AS source_memory_bank_id,
                   COALESCE(NULLIF(i.target_memory_bank_id,''),j.candidate_profile_id,'') AS target_memory_bank_id,
                   COALESCE(i.target_memory_bank_version,'') AS target_memory_bank_version,
                   COALESCE(NULLIF(i.strategy,''),j.strategy,'') AS strategy,
                   COALESCE(NULLIF(i.used_at,''),i.finished_at,j.finished_at,j.applied_at,j.created_at) AS used_at,
                   COALESCE(NULLIF(i.usage_result,''),j.status,i.status,'') AS usage_result,
                   COALESCE(NULLIF(i.generated_profile_id,''),j.candidate_profile_id,'') AS generated_profile_id,
                   i.status AS item_status,j.status AS task_status
            FROM feedback_optimization_items i
            LEFT JOIN feedback_optimization_jobs j ON j.job_id=i.job_id
            WHERE i.candidate_id IN ({placeholders})
              AND (
                    COALESCE(i.used_at,'')<>''
                    OR (i.status IN ('completed','ready_to_apply','applied')
                        AND COALESCE(j.candidate_profile_id,'')<>'')
                  )
            ORDER BY COALESCE(NULLIF(i.used_at,''),i.finished_at,j.finished_at,j.applied_at,j.created_at) DESC
            """,
            ids,
        ).fetchall()
        result: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            item = dict(row)
            candidate_id = str(item.pop("candidate_id") or "")
            result.setdefault(candidate_id, []).append({
                "feedback_sample_id": candidate_id,
                "optimization_task_id": str(item.pop("job_id") or ""),
                **item,
            })
        return result

    @classmethod
    def _attach_usage_metadata(
        cls,
        conn: sqlite3.Connection,
        items: list[dict[str, Any]],
    ) -> None:
        histories = cls._usage_history_by_candidate(
            conn,
            [str(item.get("candidate_id") or "") for item in items],
        )
        for item in items:
            history = list(histories.get(str(item.get("candidate_id") or ""), []))
            item["usage_history"] = history
            item["usage_count"] = len(history)
            item["latest_used_at"] = history[0].get("used_at") if history else None
            item["consumed_status"] = "used" if history else "unused"

    def _enrich_candidate(self, item: dict[str, Any]) -> dict[str, Any]:
        """Attach browser-safe review facts without confusing feedback time with detection time."""
        result = dict(item)
        context = dict(result.get("context") or {})
        detection_time = str(result.get("detection_time") or context.get("detection_time") or "")
        detection_time_source = "detection_result" if detection_time else ""
        if not detection_time and result.get("result_id"):
            detection = self.storage.get_result(str(result["result_id"])) or {}
            detection_time = str(detection.get("created_at") or "")
            detection_time_source = "detection_result" if detection_time else ""
        if not detection_time and str(result.get("mode") or "") == "video":
            try:
                manifest = self.settings.jobs_root.joinpath(str(result.get("task_id") or ""), "job_manifest.json")
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                detection_time = str(payload.get("created_at") or "")
                detection_time_source = "task_manifest" if detection_time else ""
            except (OSError, ValueError, json.JSONDecodeError):
                detection_time = ""
        if not detection_time:
            detection_time = _task_started_at(result.get("task_id"))
            detection_time_source = "task_id" if detection_time else ""
        result["detection_time"] = detection_time or None
        result["detection_time_source"] = detection_time_source or None
        result["feedback_submitted_at"] = str(result.get("feedback_submitted_at") or result.get("created_at") or "") or None
        result["sample_name"] = str(context.get("source_label") or result.get("sample_id") or "检测样本")
        result["source_profile_id"] = str(context.get("source_profile_id") or result.get("base_profile_id") or result.get("memory_bank_id") or "") or None
        result["source_memory_bank_id"] = str(context.get("source_memory_bank_id") or result.get("base_memory_bank_id") or result.get("memory_bank_id") or "") or None
        result["source_target_domain_id"] = str(context.get("source_target_domain_id") or result["source_profile_id"] or "") or None
        result["source_config_snapshot_id"] = str(context.get("source_config_snapshot_id") or context.get("config_profile_id") or result.get("config_version") or "") or None
        result["source_memory_bank_deleted"] = bool(
            result.get("source_memory_bank_deleted_at")
            or context.get("source_memory_bank_deleted_at")
            or context.get("source_memory_bank_deleted")
        )
        # ``consumed_status`` is attached from the task usage relation by list
        # and detail readers.  This fallback keeps isolated legacy callers
        # safe until that relation is available.
        result["consumed_status"] = "unused"
        result["usage_count"] = 0
        result["latest_used_at"] = None
        result["usage_history"] = []
        result["audit_only"] = bool(result.get("audit_only"))
        return result

    @classmethod
    def public_candidate(cls, item: dict[str, Any]) -> dict[str, Any]:
        result = dict(item)
        context = dict(result.get("context") or {})
        context.pop("_source_path", None)
        context.pop("source_path", None)
        result["context"] = context
        return result

    def _domain_snapshot_source(self, candidate: dict[str, Any]) -> tuple[Path | None, Path | None]:
        reference = str(candidate.get("source_snapshot_ref") or "").strip().replace("\\", "/")
        if not reference:
            return None, None
        root = (self.settings.jobs_root / reference).resolve()
        try:
            root.relative_to(self.settings.jobs_root.resolve())
        except ValueError:
            return None, None
        if not root.is_dir():
            return None, None
        # Image snapshots use ``source.<ext>`` while reviewed video frames use
        # ``source_frame.<ext>``.  Both are immutable correction evidence and
        # must follow the same pool eligibility rules.
        source = next(
            (
                path
                for pattern in ("source.*", "source_frame.*")
                for path in sorted(root.glob(pattern))
                if path.is_file()
            ),
            None,
        )
        mask = root / "reviewed_mask.png"
        return source, mask if mask.is_file() else None

    @staticmethod
    def _runtime_resource_identity(row: sqlite3.Row) -> set[str]:
        """Return the user-facing identities represented by one live resource."""
        item = dict(row)
        metadata = _json(item.get("metadata_json"), {})
        return {
            str(value).strip()
            for value in (
                item.get("resource_id"),
                item.get("profile_id"),
                metadata.get("memory_bank_id"),
                metadata.get("normal_memory_bank_id"),
                metadata.get("target_domain_id"),
                metadata.get("target_domain_version"),
            )
            if str(value or "").strip()
        }

    def _live_correction_source_ids(self, conn: sqlite3.Connection) -> set[str]:
        """Return existing, ordinary-user-visible target-domain identities."""
        try:
            rows = conn.execute(
                """
                SELECT resource_id,profile_id,metadata_json
                FROM runtime_resources
                WHERE resource_type IN ('target_profile','candidate_profile','normal_memory_bank')
                  AND COALESCE(archived,0)=0
                """
            ).fetchall()
        except sqlite3.Error:
            return set()
        identities: set[str] = set()
        for row in rows:
            identities.update(self._runtime_resource_identity(row))
        return identities

    def _correction_pool_rejection_codes(
        self,
        candidate: dict[str, Any],
        *,
        live_source_ids: set[str],
        active_job_count: int,
        snapshot_source: Path | None,
    ) -> list[str]:
        """Explain why a historical feedback row is not a reusable sample.

        Feedback history and the correction sample pool intentionally have
        different grains: the former is audit evidence, while the latter is a
        current input resource.  Only this method defines pool eligibility.
        """
        context = dict(candidate.get("context") or {})
        codes: list[str] = []
        status = str(candidate.get("status") or "").strip().lower()
        if status not in self._REUSABLE_REVIEW_STATUSES:
            codes.append("not_approved")
        if candidate.get("deleted_at") or bool(candidate.get("audit_only")):
            codes.append("feedback_deleted")
        if status in {"withdrawn", "rejected", "archived", "deleted", "invalid_source"}:
            codes.append("feedback_unavailable")
        if candidate.get("source_memory_bank_deleted_at") or context.get("source_memory_bank_deleted"):
            codes.append("source_memory_bank_deleted")
        source_ids = candidate_source_ids(candidate)
        if not source_ids or not source_ids.intersection(live_source_ids):
            codes.append("source_memory_bank_missing")
        if not candidate.get("source_snapshot_sha256") or snapshot_source is None:
            codes.append("snapshot_missing")
        if str(candidate.get("corrected_label") or "") not in {"normal", "defect"}:
            codes.append("corrected_label_missing")
        if bool(
            context.get("official_test_mode")
            or context.get("feedback_optimization_locked")
            or context.get("source_scope") == "official_test"
        ):
            codes.append("official_test_locked")
        if active_job_count:
            codes.append("optimization_active")
        return list(dict.fromkeys(codes))

    def list_available_reviewed_samples(
        self,
        candidate_ids: list[str] | None = None,
        *,
        memory_bank_id: str | None = None,
        source_profile_id: str | None = None,
        source_target_domain_id: str | None = None,
        include_internal_paths: bool = False,
    ) -> list[dict[str, Any]]:
        """Return the canonical, currently reusable correction sample pool.

        Prior use never makes an approved sample unavailable.  A source
        detection result may be gone only when the independently reviewed,
        immutable snapshot remains valid.  A permanently deleted/archived
        source Memory Bank still removes the row from the ordinary pool while
        leaving its audit and task-usage history untouched.
        """
        selected = {str(value) for value in (candidate_ids or []) if str(value)}
        with self._connect() as conn:
            reusable_statuses = sorted(self._REUSABLE_REVIEW_STATUSES)
            status_marks = ",".join("?" for _ in reusable_statuses)
            rows = conn.execute(
                f"""
                SELECT * FROM feedback_candidates
                WHERE status IN ({status_marks})
                  AND COALESCE(deleted_at,'')=''
                  AND COALESCE(audit_only,0)=0
                ORDER BY reviewed_at DESC, updated_at DESC
                """,
                reusable_statuses,
            ).fetchall()
            candidates = [self._row(row) for row in rows]
            self._attach_usage_metadata(conn, candidates)
            active_jobs_by_candidate = self._active_optimization_jobs_by_candidate(
                conn,
                [str(candidate.get("candidate_id") or "") for candidate in candidates],
            )
            live_source_ids = self._live_correction_source_ids(conn)
            annotations = {
                str(row["candidate_id"]): dict(row)
                for row in conn.execute(
                    "SELECT a.* FROM feedback_annotations a JOIN (SELECT candidate_id,MAX(revision) revision FROM feedback_annotations GROUP BY candidate_id) x ON x.candidate_id=a.candidate_id AND x.revision=a.revision"
                ).fetchall()
            }
        result: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate_id = str(candidate.get("candidate_id") or "")
            if selected and candidate_id not in selected:
                continue
            context = dict(candidate.get("context") or {})
            label = str(candidate.get("corrected_label") or "")
            source, snapshot_mask = self._domain_snapshot_source(candidate)
            active_jobs = list(active_jobs_by_candidate.get(candidate_id) or [])
            rejection_codes = self._correction_pool_rejection_codes(
                candidate,
                live_source_ids=live_source_ids,
                active_job_count=len(active_jobs),
                snapshot_source=source,
            )
            latest = annotations.get(candidate_id) or {}
            mask_ref = str(latest.get("mask_artifact_ref") or latest.get("mask_path") or "")
            annotation_mask = None
            if mask_ref:
                possible = (self.settings.jobs_root / mask_ref).resolve()
                try:
                    possible.relative_to(self.settings.jobs_root.resolve())
                    annotation_mask = possible if possible.is_file() else None
                except ValueError:
                    annotation_mask = None
            mask = snapshot_mask or annotation_mask
            if rejection_codes:
                continue
            source_memory_bank_id = str(
                context.get("source_memory_bank_id")
                or candidate.get("base_memory_bank_id")
                or candidate.get("memory_bank_id")
                or ""
            )
            item = {
                "candidate_id": candidate_id,
                "feedback_id": candidate.get("feedback_id"),
                "mode": candidate.get("mode"),
                "task_id": candidate.get("task_id"),
                "sample_id": candidate.get("sample_id"),
                "corrected_label": label,
                "corrected_defect_type": candidate.get("corrected_defect_type") or "",
                "dataset": candidate.get("dataset") or "",
                "category": candidate.get("category") or "",
                "source_memory_bank_id": source_memory_bank_id,
                "source_profile_id": str(
                    context.get("source_profile_id")
                    or candidate.get("base_profile_id")
                    or source_memory_bank_id
                ),
                "source_target_domain_id": str(
                    context.get("source_target_domain_id")
                    or context.get("source_profile_id")
                    or candidate.get("base_profile_id")
                    or source_memory_bank_id
                ),
                "reviewed_at": candidate.get("reviewed_at"),
                "reviewer": candidate.get("reviewer") or "local_operator",
                "snapshot_sha256": candidate.get("source_snapshot_sha256"),
                "source_name": source.name,
                "mask_available": bool(mask),
                "status": "accepted",
                "consumed_status": candidate.get("consumed_status") or "unused",
                "usage_count": int(candidate.get("usage_count") or 0),
                "latest_used_at": candidate.get("latest_used_at"),
                "usage_history": list(candidate.get("usage_history") or []),
                "eligible": True,
            }
            if include_internal_paths:
                item["_source_path"] = str(source)
                item["_mask_path"] = str(mask) if mask else ""
            result.append(item)
        if memory_bank_id:
            wanted = str(memory_bank_id)
            result = [item for item in result if str(item.get("source_memory_bank_id") or "") == wanted]
        if source_profile_id:
            wanted = str(source_profile_id)
            result = [item for item in result if str(item.get("source_profile_id") or "") == wanted]
        if source_target_domain_id:
            wanted = str(source_target_domain_id)
            result = [item for item in result if str(item.get("source_target_domain_id") or "") == wanted]
        if selected:
            resolved = {str(item["candidate_id"]) for item in result}
            missing = sorted(selected - resolved)
            if missing:
                raise ApiError(
                    "DOMAIN_FEEDBACK_NOT_ELIGIBLE",
                    "部分反馈样本未审核、来自正式测试范围或缺少独立快照，不能用于目标域构建：" + "、".join(value[:8] for value in missing),
                    409,
                )
        return result

    def domain_build_candidates(self, candidate_ids: list[str] | None = None, *, include_internal_paths: bool = False) -> list[dict[str, Any]]:
        """Compatibility entry point for the canonical correction pool."""
        return self.list_available_reviewed_samples(
            candidate_ids,
            include_internal_paths=include_internal_paths,
        )

    def _audit(self, conn: sqlite3.Connection, *, candidate_id: str | None, job_id: str | None,
               action: str, actor: str, from_status: str | None, to_status: str | None,
               details: dict[str, Any] | None = None) -> None:
        conn.execute(
            "INSERT INTO feedback_audit_log(audit_id,candidate_id,job_id,action,actor,from_status,to_status,details_json,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (uuid.uuid4().hex, candidate_id, job_id, action, actor, from_status, to_status,
             json.dumps(details or {}, ensure_ascii=False, default=str), _utc_now()),
        )

    @classmethod
    def _transition_allowed(cls, current: str, target: str) -> bool:
        current = str(current or "pending_review")
        target = str(target or current)
        return current == target or target in cls._TRANSITIONS.get(current, set())

    @staticmethod
    def _hamming_hash(left: str, right: str) -> int | None:
        """Return Hamming distance for two binary perceptual hashes."""
        left = str(left or "")
        right = str(right or "")
        if not left or len(left) != len(right) or any(char not in "01" for char in left + right):
            return None
        return sum(a != b for a, b in zip(left, right))

    @staticmethod
    def _priority(context: dict[str, Any], values: dict[str, Any]) -> tuple[float, str]:
        score = values.get("score", context.get("score"))
        threshold = values.get("threshold", context.get("threshold"))
        try:
            margin = abs(float(score) - float(threshold)) / max(abs(float(threshold)), 1e-6)
        except (TypeError, ValueError):
            margin = 0.0
        tags = set(values.get("issue_tags") or [])
        bonus = (0.7 if "missed_detection" in tags else 0.0) + (0.5 if "false_alarm" in tags else 0.0)
        if values.get("corrected_label") in {"normal", "defect"}:
            bonus += 0.2
        reasons = ["分数接近阈值" if margin < 0.25 else "分数远离阈值"]
        if "missed_detection" in tags:
            reasons.append("漏检反馈")
        if "false_alarm" in tags:
            reasons.append("误检反馈")
        return round(float(min(10.0, margin + bonus)), 6), "、".join(reasons)

    def _migrate_legacy_candidates(self) -> None:
        """Make the old calibration_candidate flag visible in the real pool."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM inspection_feedback WHERE application_mode = 'calibration_candidate'"
                ).fetchall()
                for row in rows:
                    if conn.execute("SELECT 1 FROM feedback_candidates WHERE feedback_key = ?", (row["feedback_key"],)).fetchone():
                        continue
                    context = _json(row["context_json"], {})
                    values = {
                        "assessment": row["assessment"],
                        "corrected_label": row["corrected_label"],
                        "corrected_defect_type": row["corrected_defect_type"],
                        "issue_tags": _json(row["issue_tags_json"], []),
                        "note": row["note"] or "",
                    }
                    legacy_official = bool(
                        context.get("official_test_mode")
                        or context.get("feedback_optimization_locked")
                        or context.get("source_scope") == "official_test"
                    )
                    legacy_status = (
                        "record_only"
                        if legacy_official
                        else "pending_review"
                        if context.get("task_id") and context.get("memory_bank_id")
                        else "legacy_unresolved"
                    )
                    self._insert_candidate(conn, dict(row), values, context, status=legacy_status)
        except sqlite3.OperationalError:
            # A partially upgraded database is repaired on the next startup.
            return

    def _insert_candidate(self, conn: sqlite3.Connection, record: dict[str, Any], values: dict[str, Any],
                          context: dict[str, Any], *, status: str | None = None) -> dict[str, Any]:
        now = _utc_now()
        candidate_id = str(record.get("candidate_id") or uuid.uuid4().hex)
        corrected = str(values.get("corrected_label") or record.get("corrected_label") or "keep")
        assessment = str(values.get("assessment") or record.get("assessment") or "uncertain")
        score = context.get("score")
        threshold = context.get("threshold")
        priority, reason = self._priority({**context, "score": score, "threshold": threshold}, {**values, "score": score, "threshold": threshold})
        # Detection is deliberately always ``inference_only``: that flag
        # prevents a request from mutating the active model/profile.  It must
        # not, by itself, prevent a separately reviewed offline optimization
        # job from creating a new version.  Only the explicit evaluation lock
        # (or a record captured from the official-test scope) is record-only.
        official_test = bool(
            context.get("official_test_mode")
            or context.get("feedback_optimization_locked")
            or context.get("source_scope") == "official_test"
        )
        final_status = status or ("record_only" if official_test else "pending_review")
        issue_tags = values.get("issue_tags") or _json(record.get("issue_tags_json"), []) or []
        context_to_store = dict(context)
        source_path = str(context_to_store.pop("_source_path", "") or context_to_store.pop("source_path", ""))
        source_ref = str(context_to_store.get("source_artifact_ref") or "")
        source_sha = str(context_to_store.get("source_sha256") or "")
        perceptual_hash = str(context_to_store.get("perceptual_hash") or "")
        if source_path:
            try:
                source_file = Path(source_path)
                source_sha = hashlib.sha256(source_file.read_bytes()).hexdigest()
                try:
                    from PIL import Image

                    image = Image.open(source_file).convert("L").resize((8, 8))
                    # Do not reuse the ``values`` argument here: it contains
                    # the feedback labels/notes used by the INSERT below.
                    pixel_values = list(image.getdata())
                    mean = sum(pixel_values) / max(1, len(pixel_values))
                    perceptual_hash = "".join("1" if value >= mean else "0" for value in pixel_values)
                except Exception:
                    perceptual_hash = ""
            except OSError:
                source_sha = ""
        context_to_store["_source_path"] = source_path
        insert_sql = """INSERT INTO feedback_candidates(
                candidate_id,feedback_id,feedback_key,mode,task_id,sample_id,result_id,frame_index,timestamp_ms,
                dataset,category,model_version,config_version,config_profile_id,config_profile_version,
                base_profile_id,base_memory_bank_id,base_config_version,memory_bank_id,
                method_mode,inference_mode,original_prediction,original_score,threshold,corrected_label,corrected_defect_type,
                assessment,issue_tags_json,note,status,conflict_status,priority_score,priority_reason,source_artifact_ref,
                source_sha256,perceptual_hash,created_at,updated_at,context_json
                ) VALUES (""" + ",".join("?" for _ in range(39)) + ")"
        conn.execute(insert_sql, (
                candidate_id, str(record.get("feedback_id") or uuid.uuid4().hex), str(record.get("feedback_key") or ""),
                str(record.get("mode") or context.get("mode") or "single"), str(record.get("task_id") or context.get("task_id") or ""),
                str(record.get("sample_id") or context.get("sample_id") or ""), record.get("result_id") or context.get("result_id"),
                record.get("frame_index") if record.get("frame_index") not in (-1, "-1") else None, record.get("timestamp_ms") or context.get("timestamp_ms"),
                str(context.get("dataset") or ""), str(context.get("category") or ""), str(context.get("model_version") or ""),
                str(context.get("config_version") or ""), str(context.get("config_profile_id") or ""), int(context.get("config_profile_version") or 0),
                str(context.get("source_profile_id") or context.get("memory_bank_id") or ""),
                str(context.get("source_memory_bank_id") or context.get("memory_bank_id") or ""),
                int(context.get("config_profile_version") or 0),
                str(context.get("source_memory_bank_id") or context.get("memory_bank_id") or ""),
                str(context.get("profile_method_mode") or context.get("method_mode") or ""), str(context.get("inference_mode") or ""),
                str(context.get("prediction") or ""), score, threshold, corrected, values.get("corrected_defect_type"), assessment,
                json.dumps(issue_tags, ensure_ascii=False), str(values.get("note") or record.get("note") or ""), final_status, "none",
                priority, reason, source_ref, source_sha, perceptual_hash, str(record.get("created_at") or now), now,
                json.dumps(context_to_store, ensure_ascii=False, default=str),
            ))
        conn.execute(
            "UPDATE feedback_candidates SET detection_time=?,feedback_submitted_at=? WHERE candidate_id=?",
            (str(context.get("detection_time") or "") or None, str(record.get("created_at") or now), candidate_id),
        )
        self._audit(conn, candidate_id=candidate_id, job_id=None, action="candidate_created", actor="system",
                    from_status=None, to_status=final_status, details={"priority_reason": reason})
        row = conn.execute("SELECT * FROM feedback_candidates WHERE candidate_id = ?", (candidate_id,)).fetchone()
        assert row is not None
        return self._row(row)

    def ensure_candidate(
        self,
        record: dict[str, Any],
        context: dict[str, Any],
        *,
        connection: sqlite3.Connection | None = None,
    ) -> dict[str, Any]:
        key = str(record.get("feedback_key") or "").strip()
        if not key:
            raise ApiError("INVALID_FEEDBACK_KEY", "反馈样本标识无效。", 422)
        values = dict(record)
        manager = nullcontext(connection) if connection is not None else self._connect()
        with manager as conn:
            assert conn is not None
            existing = conn.execute("SELECT * FROM feedback_candidates WHERE feedback_key = ?", (key,)).fetchone()
            if existing:
                current = self._row(existing)
                old_label = str(current.get("corrected_label") or "keep")
                new_label = str(values.get("corrected_label") or old_label)
                conflict = "label_conflict" if old_label not in {"keep", new_label} and new_label not in {"keep", old_label} else current.get("conflict_status", "none")
                current_status = str(current.get("status") or "pending_review")
                new_status = "conflict" if conflict != "none" else current_status
                if current_status == "legacy_unresolved" and conflict == "none":
                    new_status = "pending_review"
                # A later duplicate feedback update must not rewind a queued,
                # validated, applied or rolled-back candidate.  It may still
                # create a conflict, which then requires explicit review.
                if conflict == "none" and not self._transition_allowed(current_status, new_status):
                    new_status = current_status
                priority, priority_reason = self._priority(context, {**values, "score": context.get("score"), "threshold": context.get("threshold")})
                conn.execute(
                    "UPDATE feedback_candidates SET corrected_label=?, corrected_defect_type=?, assessment=?, issue_tags_json=?, note=?, conflict_status=?, status=?, priority_score=?, priority_reason=?, updated_at=? WHERE feedback_key=?",
                    (new_label, values.get("corrected_defect_type"), values.get("assessment"), json.dumps(values.get("issue_tags") or [], ensure_ascii=False), str(values.get("note") or ""), conflict, new_status, priority, priority_reason, _utc_now(), key),
                )
                self._audit(conn, candidate_id=current["candidate_id"], job_id=None, action="feedback_updated", actor="system", from_status=current.get("status"), to_status=new_status, details={"conflict_status": conflict})
                row = conn.execute("SELECT * FROM feedback_candidates WHERE feedback_key = ?", (key,)).fetchone()
                assert row is not None
                return self._row(row)
            # Content and perceptual hashes remain useful evidence metadata,
            # but are not feedback identities.  Different result_id values
            # (or video task/frame pairs) are independent detection events,
            # even when their bytes are identical.  The exact feedback_key
            # branch above already provides repeated-click idempotency.
            return self._insert_candidate(conn, {**record, "feedback_key": key}, values, context)

    def submit_feedback(
        self,
        values: dict[str, Any],
        context: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Persist feedback and its candidate in one SQLite transaction."""
        try:
            with self.database.transaction(immediate=True) as conn:
                record = self.storage.upsert_feedback(values, context, connection=conn)
                candidate = self.ensure_candidate(record, context, connection=conn)
                if str(candidate.get("feedback_id") or "") != str(record.get("feedback_id") or ""):
                    raise ApiError(
                        "FEEDBACK_CANDIDATE_LINK_FAILED",
                        "反馈记录未能绑定到本次检测结果，提交已回滚。",
                        409,
                    )
                result_id = str(context.get("result_id") or "")
                if result_id and str(candidate.get("result_id") or "") != result_id:
                    raise ApiError(
                        "FEEDBACK_CANDIDATE_RESULT_MISMATCH",
                        "反馈候选与本次检测结果不一致，提交已回滚。",
                        409,
                    )
                return record, candidate
        except ApiError:
            raise
        except sqlite3.Error as exc:
            if self.logger:
                self.logger.exception("feedback_submission_transaction_failed")
            raise ApiError(
                "FEEDBACK_SUBMISSION_FAILED",
                "反馈记录与待审核样本未能完整保存，请重试。",
                500,
            ) from exc

    def _candidate_filter(
        self, *, status: str | None = None, memory_bank_id: str | None = None,
                        source_profile_id: str | None = None, source_target_domain_id: str | None = None,
                        dataset: str | None = None, category: str | None = None,
                        task_id: str | None = None, mode: str | None = None,
                        search: str | None = None, label: str | None = None,
                        issue: str | None = None, mask: str | None = None,
                        detection_since: str | None = None,
                        feedback_since: str | None = None,
                        min_score: float | None = None,
                        method: str | None = None,
                        usage: str | None = None,
    ) -> tuple[str, list[Any]]:
        conditions: list[str] = []
        params: list[Any] = []
        # Deleted rows that are retained only as immutable version-audit
        # summaries must never reappear in ordinary feedback searches.  The
        # explicit ``status=deleted`` query remains available to maintenance
        # code when an audit investigation needs it.
        if not status:
            conditions.append("status <> 'deleted'")
        for field, value in (("status", status), ("memory_bank_id", memory_bank_id), ("dataset", dataset), ("category", category), ("task_id", task_id), ("mode", mode)):
            if not value:
                continue
            if field == "status" and "," in str(value):
                values = [part.strip() for part in str(value).split(",") if part.strip()]
                if values:
                    conditions.append(f"{field} IN ({','.join('?' for _ in values)})")
                    params.extend(values)
            else:
                conditions.append(f"{field} = ?")
                params.append(str(value))
        if memory_bank_id:
            conditions.append("COALESCE(NULLIF(base_memory_bank_id,''), memory_bank_id) = ?")
            params.append(str(memory_bank_id))
        if source_profile_id:
            conditions.append("COALESCE(NULLIF(base_profile_id,''), memory_bank_id) = ?")
            params.append(str(source_profile_id))
        if source_target_domain_id:
            conditions.append(
                "COALESCE(NULLIF(json_extract(context_json,'$.source_target_domain_id'),''), "
                "NULLIF(base_profile_id,''), memory_bank_id) = ?"
            )
            params.append(str(source_target_domain_id))
        # Search and secondary filters stay in the same SQL query as the
        # paginated list.  The UI must never fetch hundreds of candidates and
        # then filter them in Python; this also keeps ``total`` truthful for
        # the current query.  ``source_label`` is the durable sample display
        # name written into context_json by the detection bridge.
        if search and str(search).strip():
            pattern = f"%{str(search).strip().lower()}%"
            conditions.append(
                "(LOWER(COALESCE(json_extract(context_json,'$.source_label'),'')) LIKE ? "
                "OR LOWER(COALESCE(sample_id,'')) LIKE ? "
                "OR LOWER(COALESCE(corrected_defect_type,'')) LIKE ? "
                "OR LOWER(COALESCE(note,'')) LIKE ? "
                "OR LOWER(COALESCE(review_note,'')) LIKE ?)"
            )
            params.extend([pattern] * 5)
        if label:
            normalized_label = str(label).strip().lower()
            if normalized_label == "unknown":
                conditions.append("(corrected_label IS NULL OR LOWER(COALESCE(corrected_label,'')) NOT IN ('normal','defect'))")
            elif normalized_label in {"normal", "defect"}:
                conditions.append("LOWER(COALESCE(corrected_label,'')) = ?")
                params.append(normalized_label)
        if issue:
            normalized_issue = str(issue).strip().lower()
            if normalized_issue in {"false_alarm", "missed_detection"}:
                conditions.append("issue_tags_json LIKE ?")
                params.append(f'%"{normalized_issue}"%')
            elif normalized_issue in {"correct", "uncertain"}:
                conditions.append("LOWER(COALESCE(assessment,'')) = ?")
                params.append(normalized_issue)
        if mask in {"with", "without"}:
            exists_clause = (
                "EXISTS (SELECT 1 FROM feedback_annotations a "
                "WHERE a.candidate_id=feedback_candidates.candidate_id "
                "AND COALESCE(a.mask_path,'')<>'')"
            )
            conditions.append(exists_clause if mask == "with" else f"NOT {exists_clause}")
        if detection_since:
            conditions.append("COALESCE(detection_time,'') >= ?")
            params.append(str(detection_since))
        if feedback_since:
            conditions.append("COALESCE(feedback_submitted_at,created_at,'') >= ?")
            params.append(str(feedback_since))
        if min_score is not None:
            try:
                conditions.append("COALESCE(original_score,0) >= ?")
                params.append(float(min_score))
            except (TypeError, ValueError):
                pass
        if method and str(method).strip():
            conditions.append("LOWER(COALESCE(method_mode,'')) LIKE ?")
            params.append(f"%{str(method).strip().lower()}%")
        if usage in {"used", "unused"}:
            usage_exists = (
                "EXISTS (SELECT 1 FROM feedback_optimization_items i "
                "LEFT JOIN feedback_optimization_jobs j ON j.job_id=i.job_id "
                "WHERE i.candidate_id=feedback_candidates.candidate_id AND ("
                "COALESCE(i.used_at,'')<>'' OR (i.status IN ('completed','ready_to_apply','applied') "
                "AND COALESCE(j.candidate_profile_id,'')<>'')))"
            )
            conditions.append(usage_exists if usage == "used" else f"NOT {usage_exists}")
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        return where, params

    def count_candidates(self, **filters: Any) -> int:
        where, params = self._candidate_filter(**filters)
        with self._connect() as conn:
            return int(conn.execute(f"SELECT COUNT(*) FROM feedback_candidates{where}", params).fetchone()[0])

    def list_candidates(self, *, status: str | None = None, memory_bank_id: str | None = None,
                        source_profile_id: str | None = None, source_target_domain_id: str | None = None,
                        dataset: str | None = None, category: str | None = None,
                        task_id: str | None = None, mode: str | None = None,
                        search: str | None = None, label: str | None = None,
                        issue: str | None = None, mask: str | None = None,
                        detection_since: str | None = None,
                        feedback_since: str | None = None,
                        min_score: float | None = None,
                        method: str | None = None,
                        usage: str | None = None,
                        limit: int = 200,
                        offset: int = 0) -> list[dict[str, Any]]:
        where, params = self._candidate_filter(
            status=status,
            memory_bank_id=memory_bank_id,
            source_profile_id=source_profile_id,
            source_target_domain_id=source_target_domain_id,
            dataset=dataset,
            category=category,
            task_id=task_id,
            mode=mode,
            search=search,
            label=label,
            issue=issue,
            mask=mask,
            detection_since=detection_since,
            feedback_since=feedback_since,
            min_score=min_score,
            method=method,
            usage=usage,
        )
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM feedback_candidates{where} ORDER BY priority_score DESC, updated_at DESC LIMIT ? OFFSET ?",
                [*params, min(500, max(1, int(limit))), max(0, int(offset))],
            ).fetchall()
            items = [self._enrich_candidate(self._row(row)) for row in rows]
            if items:
                self._attach_usage_metadata(conn, items)
                candidate_ids = [str(item.get("candidate_id") or "") for item in items if item.get("candidate_id")]
                placeholders = ",".join("?" for _ in candidate_ids)
                mask_rows = conn.execute(
                    f"SELECT DISTINCT candidate_id FROM feedback_annotations WHERE candidate_id IN ({placeholders}) AND mask_path IS NOT NULL AND mask_path <> ''",
                    candidate_ids,
                ).fetchall()
                mask_ids = {str(row["candidate_id"]) for row in mask_rows}
                for item in items:
                    item["has_mask"] = str(item.get("candidate_id") or "") in mask_ids
                result_ids = [
                    str(item.get("result_id") or "")
                    for item in items
                    if str(item.get("result_id") or "")
                ]
                result_assets: dict[str, dict[str, Any]] = {}
                if result_ids:
                    result_placeholders = ",".join("?" for _ in result_ids)
                    result_rows = conn.execute(
                        f"""
                        SELECT result_id, original_image_url, result_image_url,
                               heatmap_url, mask_url
                        FROM detection_results
                        WHERE result_id IN ({result_placeholders})
                        """,
                        result_ids,
                    ).fetchall()
                    for result_row in result_rows:
                        assets = {
                            "original_image_url": self._safe_artifact_url(result_row["original_image_url"]),
                            "overlay_url": self._safe_artifact_url(result_row["result_image_url"]),
                            "heatmap_url": self._safe_artifact_url(result_row["heatmap_url"]),
                            "mask_url": self._safe_artifact_url(result_row["mask_url"]),
                        }
                        result_assets[str(result_row["result_id"])] = assets
                for item in items:
                    assets = dict(result_assets.get(str(item.get("result_id") or "")) or {})
                    item["source_assets"] = assets
                    item["has_visualization"] = any(
                        bool(assets.get(key)) for key in ("overlay_url", "heatmap_url", "mask_url")
                    )
        return items

    def get_candidate(self, candidate_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM feedback_candidates WHERE candidate_id = ?", (str(candidate_id),)).fetchone()
            if row is None:
                raise ApiError("FEEDBACK_CANDIDATE_NOT_FOUND", "反馈候选样本不存在。", 404)
            item = self._enrich_candidate(self._row(row))
            self._attach_usage_metadata(conn, [item])
            annotations = conn.execute("SELECT * FROM feedback_annotations WHERE candidate_id = ? ORDER BY revision DESC", (str(candidate_id),)).fetchall()
        item["annotations"] = [dict(annotation) for annotation in annotations]
        item["source_artifact_ref"] = item.get("source_artifact_ref") or None
        return item

    def _snapshot_source_path(self, candidate: dict[str, Any]) -> Path | None:
        context = dict(candidate.get("context") or {})
        raw = str(context.get("_source_path") or "").strip()
        if raw:
            path = Path(raw)
            if path.is_file():
                return path
        if candidate.get("result_id"):
            detection = self.storage.get_result(str(candidate["result_id"])) or {}
            for key in ("original_image_url", "result_image_url"):
                ref = str(detection.get(key) or "").strip().replace("\\", "/")
                if ref.startswith("/artifacts/"):
                    relative = ref.removeprefix("/artifacts/")
                    path = (self.settings.jobs_root / relative).resolve()
                    try:
                        path.relative_to(self.settings.jobs_root.resolve())
                    except ValueError:
                        continue
                    if path.is_file():
                        return path
        return None

    def _extract_persisted_video_frame(self, candidate: dict[str, Any], target: Path) -> Path:
        """Extract one reviewed frame without requiring an in-memory video job.

        Completed video jobs and their upload artifacts survive a backend
        restart, while ``VideoJobService`` deliberately does not recreate the
        old worker object.  Review snapshots must therefore be able to read
        the persisted source directly instead of treating an absent worker as
        an absent sample.
        """
        source = self._snapshot_source_path(candidate)
        if source is None or not source.is_file():
            raise ApiError("FEEDBACK_SOURCE_UNAVAILABLE", "原始视频文件不可用，无法提取审核帧。", 409)

        # A few legacy frame results already point at a decoded still image.
        # Preserve it directly instead of sending it through VideoCapture.
        try:
            from PIL import Image

            with Image.open(source) as image:
                image.convert("RGB").save(target, format="PNG")
            return target
        except Exception:
            pass

        try:
            import cv2

            capture = cv2.VideoCapture(str(source))
            if not capture.isOpened():
                raise RuntimeError("video_open_failed")
            try:
                frame_index = max(0, int(candidate.get("frame_index") or 0))
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError("video_frame_decode_failed")
                if not cv2.imwrite(str(target), frame):
                    raise RuntimeError("video_frame_save_failed")
            finally:
                capture.release()
        except ApiError:
            raise
        except Exception as exc:
            raise ApiError("FEEDBACK_FRAME_UNAVAILABLE", "对应视频帧无法读取，尚不能完成审核。", 409) from exc
        return target

    def _build_source_snapshot(self, candidate: dict[str, Any], video_jobs: Any | None = None) -> dict[str, Any]:
        """Create the immutable, minimal evidence needed after history removal."""
        candidate_id = str(candidate.get("candidate_id") or "")
        version = int(candidate.get("candidate_version") or 1)
        root = self._mask_root(candidate_id) / "snapshots"
        root.mkdir(parents=True, exist_ok=True)
        final = root / f"snapshot_v{version:04d}"
        if final.exists():
            manifest_path = final / "snapshot.json"
            if manifest_path.is_file():
                digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                return {
                    "source_snapshot_ref": final.relative_to(self.settings.jobs_root).as_posix(),
                    "source_snapshot_sha256": digest,
                    "source_snapshot": json.loads(manifest_path.read_text(encoding="utf-8")),
                }
        temporary = root / f".snapshot_v{version:04d}_{uuid.uuid4().hex}.tmp"
        temporary.mkdir(parents=False, exist_ok=False)
        copied: list[dict[str, Any]] = []
        try:
            # A video candidate represents one reviewed frame, not the source
            # video file.  Prefer extracting that immutable frame even when the
            # detection context still contains the staged video path.  The old
            # ordering copied the whole video as ``source.bin``; the feedback
            # optimizer then tried to decode it as an image and failed at the
            # first video item.
            source = None
            if str(candidate.get("mode") or "") == "video":
                source = temporary / "source_frame.png"
                extracted = False
                if video_jobs is not None:
                    try:
                        video_jobs.extract_feedback_frame(
                            str(candidate.get("task_id") or ""),
                            int(candidate.get("frame_index") or 0),
                            source,
                        )
                        extracted = source.is_file()
                    except ApiError as exc:
                        # A completed persisted job is intentionally absent
                        # from the live worker registry after restart.
                        if exc.code != "VIDEO_JOB_NOT_ACTIVE":
                            raise
                if not extracted:
                    source = self._extract_persisted_video_frame(candidate, source)
            else:
                source = self._snapshot_source_path(candidate)
            if source is not None and source.parent != temporary:
                suffix = source.suffix.lower() if source.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"} else ".bin"
                copied_source = temporary / f"source{suffix}"
                shutil.copy2(source, copied_source)
                source = copied_source
            if source is None or not source.is_file():
                raise ApiError("FEEDBACK_SOURCE_UNAVAILABLE", "原始样本不可用，尚不能解除检测记录保护。", 409)
            copied.append({"name": source.name, "size_bytes": source.stat().st_size, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
            annotations = list(candidate.get("annotations") or [])
            latest = annotations[0] if annotations else {}
            mask_ref = str(latest.get("mask_artifact_ref") or latest.get("mask_path") or "")
            if mask_ref:
                mask = (self.settings.jobs_root / mask_ref).resolve()
                try:
                    mask.relative_to(self.settings.jobs_root.resolve())
                except ValueError as exc:
                    raise ApiError("FEEDBACK_MASK_REFERENCE_INVALID", "缺陷掩码引用无效。", 409) from exc
                if mask.is_file():
                    target = temporary / "reviewed_mask.png"
                    shutil.copy2(mask, target)
                    copied.append({"name": target.name, "size_bytes": target.stat().st_size, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
            context = dict(candidate.get("context") or {})
            context.pop("_source_path", None)
            context.pop("source_path", None)
            manifest = {
                "schema_version": 1,
                "candidate_id": candidate_id,
                "feedback_id": candidate.get("feedback_id"),
                "mode": candidate.get("mode"),
                "task_id": candidate.get("task_id"),
                "result_id": candidate.get("result_id"),
                "sample_id": candidate.get("sample_id"),
                "frame_index": candidate.get("frame_index"),
                "timestamp_ms": candidate.get("timestamp_ms"),
                "original_prediction": candidate.get("original_prediction"),
                "original_score": candidate.get("original_score"),
                "threshold": candidate.get("threshold"),
                "corrected_label": candidate.get("corrected_label"),
                "corrected_defect_type": candidate.get("corrected_defect_type"),
                "reviewer": "local_operator",
                "review_note": candidate.get("review_note") or "",
                "memory_bank_id": candidate.get("memory_bank_id"),
                "config_profile_id": candidate.get("config_profile_id"),
                "config_profile_version": candidate.get("config_profile_version"),
                "method_mode": candidate.get("method_mode"),
                "context": context,
                "files": copied,
                "created_at": _utc_now(),
            }
            manifest_path = temporary / "snapshot.json"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            os.replace(temporary, final)
            for path in final.rglob("*"):
                if path.is_file():
                    try:
                        path.chmod(0o444)
                    except OSError:
                        pass
            return {
                "source_snapshot_ref": final.relative_to(self.settings.jobs_root).as_posix(),
                "source_snapshot_sha256": digest,
                "source_snapshot": manifest,
            }
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def create_snapshot(self, candidate_id: str, video_jobs: Any | None = None) -> dict[str, Any]:
        candidate = self.get_candidate(candidate_id)
        if str(candidate.get("corrected_label") or "keep") not in {"normal", "defect"}:
            raise ApiError("REVIEW_LABEL_REQUIRED", "生成审核快照前必须确认正常或缺陷标签。", 422)
        snapshot = self._build_source_snapshot(candidate, video_jobs)
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE feedback_candidates SET source_snapshot_ref=?,source_snapshot_sha256=?,source_snapshot_json=?,snapshot_error=NULL,protection_released_at=?,candidate_version=candidate_version+1,updated_at=? WHERE candidate_id=?",
                (snapshot["source_snapshot_ref"], snapshot["source_snapshot_sha256"], json.dumps(snapshot["source_snapshot"], ensure_ascii=False, default=str), now, now, str(candidate_id)),
            )
            self._audit(conn, candidate_id=str(candidate_id), job_id=None, action="source_snapshot_created", actor="local_operator", from_status=str(candidate.get("status") or ""), to_status=str(candidate.get("status") or ""), details={"sha256": snapshot["source_snapshot_sha256"]})
        return self.get_candidate(candidate_id)

    def protection_status(self, candidate_id: str) -> dict[str, Any]:
        candidate = self.get_candidate(candidate_id)
        status = str(candidate.get("status") or "pending_review")
        with self._connect() as conn:
            active_jobs = self._active_optimization_jobs(conn, str(candidate_id))
        codes = candidate_protection_codes(
            candidate,
            active_optimization_jobs=len(active_jobs),
        )
        reason_labels = {
            "feedback_pending": "存在待审核反馈",
            "feedback_reviewing": "反馈正在审核或存在标签冲突",
            "feedback_snapshot_pending": "反馈已通过，但独立样本快照尚未完成",
            "optimization_active": "反馈正被运行中的更新任务使用",
        }
        return {
            "candidate_id": str(candidate_id),
            "feedback_id": candidate.get("feedback_id"),
            "status": status,
            "protected": bool(codes),
            "reason": "；".join(reason_labels[code] for code in codes),
            "protection_codes": codes,
            "snapshot_ready": bool(candidate.get("source_snapshot_sha256")),
            "protection_released_at": candidate.get("protection_released_at"),
            "candidate_version": int(candidate.get("candidate_version") or 1),
            "active_optimization_jobs": active_jobs,
            "source_memory_bank_deleted": bool(candidate.get("source_memory_bank_deleted")),
        }

    @staticmethod
    def _active_optimization_jobs(conn: sqlite3.Connection, candidate_id: str) -> list[dict[str, str]]:
        """Return only genuinely running jobs that still consume a candidate.

        Candidate status values such as ``consumed`` and ``ready_to_apply`` are
        historical facts, not live locks.  The task row is the authority for
        whether an optimization dependency is still active.
        """
        return FeedbackCandidateService._active_optimization_jobs_by_candidate(conn, [candidate_id]).get(str(candidate_id), [])

    @staticmethod
    def _active_optimization_jobs_by_candidate(
        conn: sqlite3.Connection,
        candidate_ids: Iterable[str],
    ) -> dict[str, list[dict[str, str]]]:
        """Batch live optimization dependency checks for one visible page."""

        ids = list(dict.fromkeys(str(value) for value in candidate_ids if str(value)))
        if not ids:
            return {}
        statuses = sorted(ACTIVE_TASK_STATUSES)
        status_marks = ",".join("?" for _ in statuses)
        id_marks = ",".join("?" for _ in ids)
        result: dict[str, list[dict[str, str]]] = {candidate_id: [] for candidate_id in ids}
        known: dict[str, set[str]] = {candidate_id: set() for candidate_id in ids}
        rows = conn.execute(
            f"""
            SELECT DISTINCT i.candidate_id,j.job_id,j.status
            FROM feedback_optimization_items i
            JOIN feedback_optimization_jobs j ON j.job_id=i.job_id
            WHERE i.candidate_id IN ({id_marks})
              AND j.status IN ({status_marks})
            """,
            [*ids, *statuses],
        ).fetchall()
        for row in rows:
            candidate_id = str(row["candidate_id"] or "")
            job_id = str(row["job_id"] or "")
            if candidate_id not in result or not job_id:
                continue
            result[candidate_id].append({"job_id": job_id, "status": str(row["status"] or "")})
            known[candidate_id].add(job_id)

        # Older jobs may predate item-row materialization.  Read the active
        # set once and fan out its immutable candidate id list.
        legacy_rows = conn.execute(
            f"SELECT job_id,status,candidate_ids_json FROM feedback_optimization_jobs WHERE status IN ({status_marks})",
            statuses,
        ).fetchall()
        for row in legacy_rows:
            job_id = str(row["job_id"] or "")
            for candidate_id in {str(value) for value in _json(row["candidate_ids_json"], []) if value}:
                if candidate_id not in result or job_id in known[candidate_id]:
                    continue
                result[candidate_id].append({"job_id": job_id, "status": str(row["status"] or "")})
        return result

    def mark_source_memory_bank_deleted(
        self,
        *,
        profile_id: str,
        resource_id: str | None = None,
        actor: str = "local_operator",
        connection: sqlite3.Connection | None = None,
    ) -> dict[str, Any]:
        """Record a deleted source without discarding feedback audit evidence.

        The source ids remain available for traceability.  Once the source is
        gone, an inaccessible review row is audit evidence rather than a lock;
        only a genuinely active optimization transaction remains protected.
        """
        source_ids = {str(value).strip() for value in (profile_id, resource_id) if str(value or "").strip()}
        if not source_ids:
            return {"affected": 0, "released": 0, "still_protected": 0, "active_jobs": 0}

        def _apply(conn: sqlite3.Connection) -> dict[str, Any]:
            rows = conn.execute(
                "SELECT * FROM feedback_candidates WHERE COALESCE(deleted_at,'')=''"
            ).fetchall()
            now = _utc_now()
            affected = released = still_protected = active_job_count = 0
            for row in rows:
                candidate = self._row(row)
                context = dict(candidate.get("context") or {})
                references = candidate_source_ids(candidate)
                if not references.intersection(source_ids):
                    continue
                candidate_id = str(candidate["candidate_id"])
                active_jobs = self._active_optimization_jobs(conn, candidate_id)
                status = str(candidate.get("status") or "pending_review")
                protected = bool(candidate_protection_codes(
                    candidate,
                    active_optimization_jobs=len(active_jobs),
                    source_deleted_override=True,
                ))
                deleted_at = str(candidate.get("source_memory_bank_deleted_at") or now)
                already_released = bool(candidate.get("protection_released_at"))
                archive_stale_review = not protected and status in {
                    "collected", "legacy_unresolved", "pending_review", "in_review", "conflict"
                }
                target_status = "archived" if archive_stale_review else status
                needs_update = (
                    not candidate.get("source_memory_bank_deleted_at")
                    or (not protected and not already_released)
                    or target_status != status
                )
                if not needs_update:
                    if protected:
                        still_protected += 1
                        active_job_count += len(active_jobs)
                    continue
                context.update(
                    {
                        "source_memory_bank_deleted": True,
                        "source_memory_bank_deleted_at": deleted_at,
                    }
                )
                conn.execute(
                    """
                    UPDATE feedback_candidates
                    SET source_memory_bank_deleted_at=?,context_json=?,status=?,
                        protection_released_at=CASE WHEN ?=0 THEN COALESCE(protection_released_at,?) ELSE protection_released_at END,
                        candidate_version=candidate_version+1,updated_at=?
                    WHERE candidate_id=?
                    """,
                    (deleted_at, json.dumps(context, ensure_ascii=False, default=str), target_status, int(protected), now, now, candidate_id),
                )
                self._audit(
                    conn,
                    candidate_id=candidate_id,
                    job_id=None,
                    action="source_memory_bank_deleted",
                    actor=actor,
                    from_status=status,
                    to_status=target_status,
                    details={
                        "source_profile_id": str(profile_id),
                        "source_resource_id": str(resource_id or ""),
                        "protection_released": not protected,
                        "stale_review_lock_reconciled": bool(candidate.get("source_memory_bank_deleted_at")) and not protected,
                        "stale_review_archived": archive_stale_review,
                        "active_job_count": len(active_jobs),
                    },
                )
                affected += 1
                active_job_count += len(active_jobs)
                if protected:
                    still_protected += 1
                else:
                    released += 1
            return {
                "affected": affected,
                "released": released,
                "still_protected": still_protected,
                "active_jobs": active_job_count,
            }

        if connection is not None:
            return _apply(connection)
        with self._connect() as conn:
            return _apply(conn)

    def review(self, candidate_id: str, *, action: str, reviewer: str = "local_operator", corrected_label: str | None = None,
               corrected_defect_type: str | None = None, note: str | None = None,
               require_snapshot: bool = True, expected_version: int | None = None,
               video_jobs: Any | None = None) -> dict[str, Any]:
        action = str(action).lower()
        if action not in {"accept", "reject", "withdraw", "update", "resolve_conflict", "start_review", "archive", "dequeue"}:
            raise ApiError("INVALID_REVIEW_ACTION", "审核操作无效。", 422)
        reviewer = "local_operator"
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM feedback_candidates WHERE candidate_id = ?", (str(candidate_id),)).fetchone()
            if row is None:
                raise ApiError("FEEDBACK_CANDIDATE_NOT_FOUND", "反馈候选样本不存在。", 404)
            current = self._row(row)
            from_status = str(current.get("status") or "pending_review")
            if expected_version is not None and int(current.get("candidate_version") or 1) != int(expected_version):
                raise ApiError("FEEDBACK_VERSION_CONFLICT", "反馈已被其他操作更新，请刷新后重试。", 409)
            if from_status in {"queued", "optimizing", "consumed", "ready_to_apply", "rolled_back", "archived"}:
                raise ApiError("FEEDBACK_STATUS_LOCKED", "该反馈当前处于优化或只读锁定状态，不能重复修改。", 409)
            label = str(corrected_label or current.get("corrected_label") or "keep")
            if label not in {"keep", "normal", "defect"}:
                raise ApiError("INVALID_CORRECTED_LABEL", "修正标签只能是保留、正常或缺陷。", 422)
            if action in {"accept", "resolve_conflict", "update"} and label == "keep" and action != "update":
                raise ApiError("REVIEW_LABEL_REQUIRED", "进入优化候选池前必须确认正常或缺陷标签。", 422)
            # Editing a label is not the same as approving it.  Always return
            # an edited sample to the review queue; only an explicit accept or
            # conflict-resolution action can make it optimization-eligible.
            to_status = (
                "accepted" if action in {"accept", "resolve_conflict"}
                else "in_review" if action == "start_review"
                else "archived" if action == "archive"
                else "withdrawn" if action == "withdraw"
                else "pending_review" if action in {"update", "dequeue"}
                else "rejected"
            )
            conflict = "none" if action == "resolve_conflict" else str(current.get("conflict_status") or "none")
            if conflict != "none" and action != "resolve_conflict":
                raise ApiError("FEEDBACK_CONFLICT", "该反馈存在标签冲突，请先解决冲突。", 409)
            if not self._transition_allowed(from_status, to_status):
                raise ApiError("FEEDBACK_INVALID_STATE_TRANSITION", "当前反馈状态不允许执行该审核操作。", 409)
        snapshot: dict[str, Any] = {}
        if to_status == "accepted" and require_snapshot:
            staged = {**current, "corrected_label": label, "corrected_defect_type": corrected_defect_type or current.get("corrected_defect_type"), "review_note": note or ""}
            try:
                snapshot = self._build_source_snapshot(staged, video_jobs)
            except ApiError as exc:
                with self._connect() as conn:
                    conn.execute("UPDATE feedback_candidates SET snapshot_error=?,updated_at=? WHERE candidate_id=?", (exc.message, _utc_now(), str(candidate_id)))
                raise
        now = _utc_now()
        released_at = now if to_status in {"rejected", "withdrawn", "archived"} or (to_status == "accepted" and snapshot.get("source_snapshot_sha256")) else current.get("protection_released_at")
        with self.database.transaction(immediate=True) as conn:
            rowcount = conn.execute(
                "UPDATE feedback_candidates SET corrected_label=?,corrected_defect_type=?,status=?,conflict_status=?,reviewer=?,review_note=?,reviewed_at=?,source_snapshot_ref=COALESCE(?,source_snapshot_ref),source_snapshot_sha256=COALESCE(?,source_snapshot_sha256),source_snapshot_json=CASE WHEN ?<>'' THEN ? ELSE source_snapshot_json END,snapshot_error=NULL,protection_released_at=?,candidate_version=candidate_version+1,updated_at=? WHERE candidate_id=? AND candidate_version=?",
                (label, corrected_defect_type or current.get("corrected_defect_type"), to_status, conflict, reviewer, str(note or "")[:1000], now, snapshot.get("source_snapshot_ref"), snapshot.get("source_snapshot_sha256"), str(snapshot.get("source_snapshot_sha256") or ""), json.dumps(snapshot.get("source_snapshot") or {}, ensure_ascii=False, default=str), released_at, now, str(candidate_id), int(current.get("candidate_version") or 1)),
            ).rowcount
            if not rowcount:
                raise ApiError("FEEDBACK_VERSION_CONFLICT", "反馈已被其他操作更新，请刷新后重试。", 409)
            self._audit(conn, candidate_id=str(candidate_id), job_id=None, action=f"review_{action}", actor=reviewer, from_status=from_status, to_status=to_status, details={"corrected_label": label, "corrected_defect_type": corrected_defect_type, "review_note": str(note or "")[:1000], "snapshot_sha256": snapshot.get("source_snapshot_sha256")})
            fresh = conn.execute("SELECT * FROM feedback_candidates WHERE candidate_id = ?", (str(candidate_id),)).fetchone()
        assert fresh is not None
        return self._row(fresh)

    def delete_preview(self, candidate_ids: Iterable[str], *, create_plan: bool = True) -> dict[str, Any]:
        selected = list(dict.fromkeys(str(value) for value in candidate_ids if str(value)))
        rows: list[dict[str, Any]] = []
        with self._connect() as conn:
            for candidate_id in selected:
                row = conn.execute("SELECT * FROM feedback_candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
                if row is None:
                    rows.append({"candidate_id": candidate_id, "action": "missing", "message": "记录不存在"})
                    continue
                item = self._row(row)
                used = bool(conn.execute("SELECT 1 FROM feedback_optimization_items WHERE candidate_id=? LIMIT 1", (candidate_id,)).fetchone())
                status = str(item.get("status") or "pending_review")
                if used or status in {"queued", "optimizing", "ready_to_apply", "consumed", "frozen", "applied", "rolled_back", "completed"}:
                    action, message = "audit_only", "删除反馈历史记录，已生成版本保持不变并保留最小版本审计摘要"
                else:
                    action, message = "physical_delete", "直接删除反馈、候选、标注和未使用附件"
                annotation_rows = conn.execute(
                    "SELECT mask_path,mask_artifact_ref FROM feedback_annotations WHERE candidate_id=?",
                    (candidate_id,),
                ).fetchall()
                root = self.settings.jobs_root / "feedback_candidates" / candidate_id
                files = {path.resolve() for path in root.rglob("*") if path.is_file()} if root.is_dir() else set()
                mask_files: set[Path] = set()
                jobs_root = self.settings.jobs_root.resolve()
                for annotation in annotation_rows:
                    for raw_ref in (annotation["mask_artifact_ref"], annotation["mask_path"]):
                        ref = str(raw_ref or "").strip()
                        if not ref:
                            continue
                        path = Path(ref)
                        candidate_path = path.resolve() if path.is_absolute() else (self.settings.jobs_root / path).resolve()
                        try:
                            candidate_path.relative_to(jobs_root)
                        except ValueError:
                            continue
                        if candidate_path.is_file():
                            mask_files.add(candidate_path)
                rows.append({
                    "candidate_id": candidate_id,
                    "sample_name": str((item.get("context") or {}).get("source_label") or item.get("sample_id") or "检测样本"),
                    "status": status,
                    "action": action,
                    "message": message,
                    "mask_count": len(mask_files),
                    "attachment_count": len(files - mask_files),
                })
        plan_payload = json.dumps(selected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        plan_hash = hashlib.sha256(plan_payload.encode("utf-8")).hexdigest()
        plan_id = ""
        preview_token = ""
        if create_plan:
            plan_id = f"feedback_delete_plan_{uuid.uuid4().hex}"
            preview_token = secrets.token_urlsafe(32)
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO feedback_delete_plans(plan_id,preview_token_hash,candidate_ids_json,plan_hash,status,created_at,expires_at) VALUES(?,?,?,?,?,?,?)",
                    (plan_id, hashlib.sha256(preview_token.encode("utf-8")).hexdigest(), plan_payload, plan_hash, "pending", _utc_now(), time.time() + 600),
                )
        return {
            "items": rows,
            "selected_count": len(selected),
            "feedback_count": sum(item["action"] in {"physical_delete", "audit_only"} for item in rows),
            # Compatibility field for older clients.  Deletion no longer
            # routes pending feedback through the review-withdraw workflow.
            "withdraw_count": 0,
            "physical_delete_count": sum(item["action"] == "physical_delete" for item in rows),
            "audit_only_count": sum(item["action"] == "audit_only" for item in rows),
            "mask_count": sum(int(item.get("mask_count") or 0) for item in rows),
            "attachment_count": sum(int(item.get("attachment_count") or 0) for item in rows),
            "plan_id": plan_id,
            "preview_token": preview_token,
            "plan_hash": plan_hash,
            "expires_in_seconds": 600,
        }

    def delete_candidates(
        self,
        candidate_ids: Iterable[str],
        *,
        preview_token: str | None = None,
        acknowledged: bool = False,
        actor: str = "local_operator",
    ) -> dict[str, Any]:
        if not acknowledged:
            raise ApiError("CONFIRMATION_REQUIRED", "请先核对删除预览并勾选确认。", 422)
        selected = list(dict.fromkeys(str(value) for value in candidate_ids if str(value)))
        if preview_token:
            with self._connect() as conn:
                plan = conn.execute("SELECT * FROM feedback_delete_plans WHERE preview_token_hash=? AND status='pending'", (hashlib.sha256(str(preview_token).encode("utf-8")).hexdigest(),)).fetchone()
                if plan is None or float(plan["expires_at"] or 0) <= time.time():
                    raise ApiError("FEEDBACK_DELETE_PREVIEW_EXPIRED", "删除预览已过期，请重新生成。", 409)
                planned = [str(value) for value in _json(plan["candidate_ids_json"], [])]
                if planned != selected:
                    raise ApiError("FEEDBACK_DELETE_PREVIEW_STALE", "反馈删除对象已变化，请重新生成预览。", 409)
                conn.execute("UPDATE feedback_delete_plans SET status='consumed',consumed_at=? WHERE plan_id=? AND status='pending'", (_utc_now(), plan["plan_id"]))
        # Re-evaluate current rows without creating a second unused plan.  The
        # original immutable plan has already been consumed above.
        preview = self.delete_preview(candidate_ids, create_plan=False)
        results: list[dict[str, Any]] = []
        artifact_roots: list[Path] = []
        for plan in preview["items"]:
            candidate_id = str(plan["candidate_id"])
            action = str(plan["action"])
            if action == "missing":
                results.append({**plan, "success": False})
                continue
            with self._connect() as conn:
                row = conn.execute("SELECT * FROM feedback_candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
                if row is None:
                    results.append({**plan, "success": False})
                    continue
                item = self._row(row)
                latest_mask = conn.execute(
                    "SELECT mask_sha256 FROM feedback_annotations WHERE candidate_id=? ORDER BY revision DESC LIMIT 1",
                    (candidate_id,),
                ).fetchone()
                details = {
                    "feedback_id": item.get("feedback_id"),
                    "source_profile_id": item.get("base_profile_id") or item.get("memory_bank_id"),
                    "source_memory_bank_id": item.get("base_memory_bank_id") or item.get("memory_bank_id"),
                    "corrected_label": item.get("corrected_label"),
                    "source_sha256": item.get("source_snapshot_sha256") or item.get("source_sha256"),
                    "mask_sha256": latest_mask["mask_sha256"] if latest_mask else None,
                }
                now = _utc_now()
                if action == "audit_only":
                    self._audit(conn, candidate_id=candidate_id, job_id=None, action="feedback_deleted_audit_retained", actor=actor, from_status=str(item.get("status") or ""), to_status="deleted", details=details)
                    conn.execute("DELETE FROM feedback_annotations WHERE candidate_id=?", (candidate_id,))
                    conn.execute("DELETE FROM inspection_feedback WHERE feedback_id=?", (str(item.get("feedback_id") or ""),))
                    conn.execute("DELETE FROM feedback_records WHERE feedback_id=?", (str(item.get("feedback_id") or ""),))
                    conn.execute(
                        "UPDATE feedback_candidates SET status='deleted',feedback_key=CASE WHEN feedback_key LIKE 'audit:%' THEN feedback_key ELSE 'audit:' || candidate_id || ':' || feedback_key END,note='',review_note='',context_json='{}',source_artifact_ref=NULL,source_snapshot_ref=NULL,source_snapshot_json='{}',deleted_at=?,audit_only=1,protection_released_at=COALESCE(protection_released_at,?),candidate_version=candidate_version+1,updated_at=? WHERE candidate_id=?",
                        (now, now, now, candidate_id),
                    )
                else:
                    self._audit(conn, candidate_id=candidate_id, job_id=None, action="feedback_physically_deleted", actor=actor, from_status=str(item.get("status") or ""), to_status="deleted", details=details)
                    conn.execute("DELETE FROM feedback_annotations WHERE candidate_id=?", (candidate_id,))
                    conn.execute("DELETE FROM inspection_feedback WHERE feedback_id=?", (str(item.get("feedback_id") or ""),))
                    conn.execute("DELETE FROM feedback_records WHERE feedback_id=?", (str(item.get("feedback_id") or ""),))
                    conn.execute("DELETE FROM feedback_candidates WHERE candidate_id=?", (candidate_id,))
                artifact_roots.append(self.settings.jobs_root / "feedback_candidates" / candidate_id)
            for root in artifact_roots[-1:]:
                if root.is_dir():
                    shutil.rmtree(root, ignore_errors=True)
            results.append({**plan, "success": True, "status": "deleted"})
        return {
            "results": results,
            "success_count": sum(bool(item.get("success")) for item in results),
            "failed_count": sum(not bool(item.get("success")) for item in results),
        }

    def bulk_review(self, candidate_ids: list[str], *, action: str, note: str = "", require_snapshot: bool = True,
                    video_jobs: Any | None = None) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for candidate_id in [str(value) for value in candidate_ids if str(value)]:
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            sample_name = "检测样本"
            try:
                current = self.get_candidate(candidate_id)
                sample_name = str((current.get("context") or {}).get("source_label") or current.get("sample_id") or "检测样本")
                validation_errors: list[str] = []
                if action == "accept":
                    corrected_label = str(current.get("corrected_label") or "keep")
                    if corrected_label not in {"normal", "defect"}:
                        validation_errors.append("尚未确认真实标签")
                    if str(current.get("conflict_status") or "none") != "none" or str(current.get("status") or "") == "conflict":
                        validation_errors.append("存在标签冲突")
                    if not str(current.get("memory_bank_id") or current.get("base_memory_bank_id") or "").strip():
                        validation_errors.append("缺少来源记忆库")
                    if not str(current.get("sample_id") or "").strip():
                        validation_errors.append("缺少样本标识")
                if validation_errors:
                    results.append({
                        "candidate_id": candidate_id,
                        "sample_name": sample_name,
                        "success": False,
                        "error": {
                            "code": "FEEDBACK_BULK_REVIEW_VALIDATION_FAILED",
                            "message": "；".join(validation_errors),
                        },
                    })
                    continue
                candidate = self.review(
                    candidate_id,
                    action=action,
                    reviewer="local_operator",
                    note=note,
                    require_snapshot=require_snapshot,
                    video_jobs=video_jobs,
                )
                results.append({
                    "candidate_id": candidate_id,
                    "sample_name": sample_name,
                    "success": True,
                    "candidate": self.public_candidate(candidate),
                })
            except ApiError as exc:
                results.append({
                    "candidate_id": candidate_id,
                    "sample_name": sample_name,
                    "success": False,
                    "error": {"code": exc.code, "message": exc.message},
                })
        return {
            "results": results,
            "success_count": sum(bool(item.get("success")) for item in results),
            "failed_count": sum(not bool(item.get("success")) for item in results),
        }

    def _mask_root(self, candidate_id: str) -> Path:
        if not candidate_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in candidate_id):
            raise ApiError("INVALID_CANDIDATE_ID", "候选样本标识无效。", 422)
        path = self.settings.jobs_root / "feedback_candidates" / candidate_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_annotation(self, candidate_id: str, *, annotation_type: str, mask_base64: str | None = None,
                        use_original_mask: bool = False,
                        mask_width: int | None = None, mask_height: int | None = None, bbox_json: Any = None,
                        polygon_json: Any = None, note: str | None = None, created_by: str = "operator") -> dict[str, Any]:
        candidate = self.get_candidate(candidate_id)
        candidate_context = dict(candidate.get("context") or {})
        # When the source dimensions are known, a user-provided mask must use
        # exactly the same raster dimensions.  This keeps patch/pixel labels
        # aligned and prevents silently applying a resized annotation.  Older
        # candidates without dimensions remain accepted for compatibility.
        expected_width = int(candidate_context.get("source_width") or 0)
        expected_height = int(candidate_context.get("source_height") or 0)
        source_path = str(candidate_context.get("_source_path") or "")
        if (expected_width <= 0 or expected_height <= 0) and source_path:
            try:
                from PIL import Image

                with Image.open(source_path) as source_image:
                    expected_width, expected_height = source_image.size
            except Exception:
                # The source may have been archived or be a video.  The
                # worker still validates the decoded mask before use.
                expected_width = expected_height = 0
        data = None
        if use_original_mask:
            source = self.source_info(candidate_id)
            reference = str(source.get("mask_url") or "").strip().replace("\\", "/")
            if reference.startswith("/artifacts/"):
                reference = reference.removeprefix("/artifacts/")
            elif reference.startswith("artifacts/"):
                reference = reference.removeprefix("artifacts/")
            else:
                raise ApiError("ORIGINAL_MASK_UNAVAILABLE", "当前检测结果没有可复用的系统 Mask。", 409)
            original = (self.settings.jobs_root / reference).resolve()
            try:
                original.relative_to(self.settings.jobs_root.resolve())
            except ValueError as exc:
                raise ApiError("ORIGINAL_MASK_INVALID", "系统 Mask 引用无效。", 409) from exc
            if not original.is_file():
                raise ApiError("ORIGINAL_MASK_UNAVAILABLE", "当前检测结果的系统 Mask 已不可用。", 409)
            data = original.read_bytes()
        if mask_base64:
            encoded = str(mask_base64).split(",", 1)[-1]
            try:
                data = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ApiError("INVALID_MASK", "Mask 文件编码无效。", 422) from exc
            if len(data) > 20 * 1024 * 1024:
                raise ApiError("MASK_TOO_LARGE", "Mask 文件不能超过 20 MB。", 422)
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ApiError("INVALID_MASK", "Mask 必须是 PNG 图像。", 422)
            # Decode and normalize on the server.  This prevents an arbitrary
            # RGBA/HTML payload from being treated as a mask and guarantees a
            # single-channel binary PNG whose dimensions match the source
            # annotation.  The original detection mask is never overwritten.
            try:
                from PIL import Image

                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                    width, height = image.size
                    if width > 10000 or height > 10000 or width * height > 100_000_000:
                        raise ApiError("MASK_TOO_LARGE", "Mask 分辨率过大。", 422)
                    if mask_width and int(mask_width) != width or mask_height and int(mask_height) != height:
                        raise ApiError("MASK_DIMENSION_MISMATCH", "Mask 尺寸与上传内容不一致。", 422)
                    if expected_width > 0 and expected_height > 0 and (width != expected_width or height != expected_height):
                        raise ApiError("MASK_SOURCE_DIMENSION_MISMATCH", "Mask 尺寸必须与源图片完全一致。", 422)
                    gray = image.convert("L")
                    binary = gray.point(lambda value: 255 if value >= 128 else 0, mode="1")
                    output = io.BytesIO()
                    binary.save(output, format="PNG", optimize=True)
                    data = output.getvalue()
                    mask_width, mask_height = width, height
            except ApiError:
                raise
            except Exception as exc:
                raise ApiError("INVALID_MASK", "Mask 图像无法读取或转换。", 422) from exc
        if data:
            try:
                from PIL import Image

                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                    width, height = image.size
                    if expected_width > 0 and expected_height > 0 and (width != expected_width or height != expected_height):
                        raise ApiError("MASK_SOURCE_DIMENSION_MISMATCH", "Mask 尺寸必须与源图片完全一致。", 422)
                    gray = image.convert("L")
                    binary = gray.point(lambda value: 255 if value >= 128 else 0, mode="1")
                    output = io.BytesIO()
                    binary.save(output, format="PNG", optimize=True)
                    data = output.getvalue()
                    mask_width, mask_height = width, height
            except ApiError:
                raise
            except Exception as exc:
                raise ApiError("INVALID_MASK", "Mask 图像无法读取或转换。", 422) from exc
        with self._connect() as conn:
            revision = int(conn.execute("SELECT COALESCE(MAX(revision),0)+1 FROM feedback_annotations WHERE candidate_id=?", (str(candidate_id),)).fetchone()[0])
            ref = None
            digest = None
            if data:
                target = self._mask_root(candidate_id) / f"mask_r{revision:04d}.png"
                temporary = target.with_suffix(".tmp")
                temporary.write_bytes(data)
                temporary.replace(target)
                digest = hashlib.sha256(data).hexdigest()
                ref = f"feedback_candidates/{candidate_id}/{target.name}"
            annotation_id = uuid.uuid4().hex
            now = _utc_now()
            conn.execute("INSERT INTO feedback_annotations(annotation_id,candidate_id,revision,annotation_type,mask_path,mask_artifact_ref,mask_sha256,mask_width,mask_height,mask_format,bbox_json,polygon_json,note,created_by,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (annotation_id, str(candidate_id), revision, str(annotation_type or "region"), ref, ref, digest, mask_width, mask_height, "png" if ref else None, json.dumps(bbox_json, ensure_ascii=False, default=str) if bbox_json is not None else None, json.dumps(polygon_json, ensure_ascii=False, default=str) if polygon_json is not None else None, str(note or "")[:1000], str(created_by or "operator")[:80], now, now))
            self._audit(conn, candidate_id=str(candidate_id), job_id=None, action="annotation_saved", actor=str(created_by or "operator"), from_status=None, to_status=None, details={"revision": revision, "mask_sha256": digest})
            row = conn.execute("SELECT * FROM feedback_annotations WHERE annotation_id=?", (annotation_id,)).fetchone()
        return dict(row) if row else {"annotation_id": annotation_id, "revision": revision, "mask_artifact_ref": ref, "mask_sha256": digest}

    def annotation_history(self, candidate_id: str) -> list[dict[str, Any]]:
        self.get_candidate(candidate_id)
        with self._connect() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM feedback_annotations WHERE candidate_id=? ORDER BY revision DESC", (str(candidate_id),)).fetchall()]

    def source_info(self, candidate_id: str) -> dict[str, Any]:
        candidate = self.get_candidate(candidate_id)
        context = dict(candidate.get("context") or {})
        source_width = int(context.get("source_width") or 0)
        source_height = int(context.get("source_height") or 0)
        if (source_width <= 0 or source_height <= 0) and str(context.get("_source_path") or ""):
            try:
                from PIL import Image

                with Image.open(str(context.get("_source_path"))) as source_image:
                    source_width, source_height = source_image.size
            except Exception:
                source_width = source_height = 0
        result: dict[str, Any] = {
            "mode": candidate.get("mode"),
            "task_id": candidate.get("task_id"),
            "sample_id": candidate.get("sample_id"),
            "frame_index": candidate.get("frame_index"),
            "timestamp_ms": candidate.get("timestamp_ms"),
            "source_artifact_ref": candidate.get("source_artifact_ref"),
            "source_width": source_width or None,
            "source_height": source_height or None,
        }
        if candidate.get("result_id"):
            detection = self.storage.get_result(str(candidate["result_id"])) or {}
            result.update({
                "original_image_url": self._safe_artifact_url(detection.get("original_image_url")),
                "result_image_url": self._safe_artifact_url(detection.get("result_image_url")),
                "heatmap_url": self._safe_artifact_url(detection.get("heatmap_url")),
                "mask_url": self._safe_artifact_url(detection.get("mask_url")),
            })
        with self._connect() as conn:
            latest_annotation = conn.execute(
                "SELECT revision, mask_artifact_ref FROM feedback_annotations "
                "WHERE candidate_id=? AND mask_artifact_ref IS NOT NULL ORDER BY revision DESC LIMIT 1",
                (str(candidate_id),),
            ).fetchone()
        if latest_annotation and str(latest_annotation["mask_artifact_ref"] or ""):
            reference = str(latest_annotation["mask_artifact_ref"]).replace("\\", "/").lstrip("/")
            result["review_mask_url"] = f"/artifacts/{reference}"
            result["review_mask_revision"] = int(latest_annotation["revision"] or 0)
        return result

    @staticmethod
    def _safe_artifact_url(value: Any) -> str | None:
        text = str(value or "").strip()
        if not text or "\\" in text or (len(text) > 2 and text[1] == ":"):
            return None
        if text.startswith(("/artifacts/", "artifacts/", "http://", "https://")):
            return text
        return None

    def source_preview(self, candidate_id: str, video_jobs: Any | None = None) -> dict[str, Any]:
        """Return a controlled preview reference without exposing local paths."""
        candidate = self.get_candidate(candidate_id)
        if candidate.get("result_id"):
            detection = self.storage.get_result(str(candidate["result_id"])) or {}
            value = self._safe_artifact_url(detection.get("original_image_url"))
            if value:
                return {"kind": "image", "url": str(value), "source_artifact_ref": candidate.get("source_artifact_ref")}
        if candidate.get("mode") == "video" and video_jobs is not None:
            try:
                target = self._mask_root(str(candidate_id)) / "source_frame.png"
                video_jobs.extract_feedback_frame(str(candidate["task_id"]), int(candidate.get("frame_index") or 0), target)
                return {
                    "kind": "video_frame",
                    "url": f"/artifacts/feedback_candidates/{candidate_id}/source_frame.png",
                    "source_artifact_ref": candidate.get("source_artifact_ref"),
                    "frame_index": candidate.get("frame_index"),
                    "timestamp_ms": candidate.get("timestamp_ms"),
                }
            except Exception:
                pass
        source_path = Path(str((candidate.get("context") or {}).get("_source_path") or ""))
        if source_path.is_file():
            try:
                target = self._mask_root(str(candidate_id)) / (
                    "source_frame.png" if candidate.get("mode") == "video" else "source_image.png"
                )
                if candidate.get("mode") == "video":
                    import cv2

                    capture = cv2.VideoCapture(str(source_path))
                    try:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, int(candidate.get("frame_index") or 0))
                        ok, frame = capture.read()
                        if not ok or frame is None:
                            raise ValueError("视频帧无法读取")
                        temporary = target.with_name(f"{target.stem}.part{target.suffix}")
                        if not cv2.imwrite(str(temporary), frame):
                            raise ValueError("视频帧无法保存")
                        os.replace(temporary, target)
                    finally:
                        capture.release()
                elif not target.is_file() or target.stat().st_mtime_ns < source_path.stat().st_mtime_ns:
                    from PIL import Image

                    with Image.open(source_path) as source_image:
                        source_image.convert("RGB").save(target, format="PNG", optimize=True)
                return {
                    "kind": "video_frame" if candidate.get("mode") == "video" else "image",
                    "url": f"/artifacts/feedback_candidates/{candidate_id}/{target.name}",
                    "source_artifact_ref": candidate.get("source_artifact_ref"),
                    "frame_index": candidate.get("frame_index") if candidate.get("mode") == "video" else None,
                    "timestamp_ms": candidate.get("timestamp_ms") if candidate.get("mode") == "video" else None,
                }
            except Exception:
                pass
        return {"kind": "unavailable", "url": None, "source_artifact_ref": candidate.get("source_artifact_ref")}

    def statistics(
        self,
        *,
        memory_bank_id: str | None = None,
        memory_bank_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        visible_ids: tuple[str, ...] | None = None
        if memory_bank_id:
            where = " WHERE memory_bank_id = ?"
            params: list[Any] = [str(memory_bank_id)]
            scope = "memory_bank"
        elif memory_bank_ids is not None:
            visible_ids = tuple(dict.fromkeys(str(value) for value in memory_bank_ids if str(value)))
            if visible_ids:
                marks = ",".join("?" for _ in visible_ids)
                where = f" WHERE memory_bank_id IN ({marks})"
                params = list(visible_ids)
            else:
                where = " WHERE 1=0"
                params = []
            scope = "active_memory_banks"
        else:
            where = ""
            params = []
            scope = "all_records"
        with self._connect() as conn:
            active_where = f"{where} AND status<>'deleted'" if where else " WHERE status<>'deleted'"
            total = int(conn.execute(f"SELECT COUNT(*) FROM feedback_candidates{active_where}", params).fetchone()[0])
            rows = conn.execute(
                f"SELECT status, corrected_label, mode, COUNT(*) AS count FROM feedback_candidates{active_where} GROUP BY status, corrected_label, mode",
                params,
            ).fetchall()
            issue_where = f"{where} AND status NOT IN ('withdrawn','deleted')" if where else " WHERE status NOT IN ('withdrawn','deleted')"
            issue_rows = conn.execute(f"SELECT issue_tags_json, COUNT(*) AS count FROM feedback_candidates{issue_where} GROUP BY issue_tags_json", params).fetchall()
            mask_query = "SELECT COUNT(DISTINCT c.candidate_id) FROM feedback_candidates c JOIN feedback_annotations a ON a.candidate_id=c.candidate_id AND a.mask_path IS NOT NULL WHERE c.status<>'deleted'"
            mask_params: list[Any] = []
            if memory_bank_id:
                mask_query += " AND c.memory_bank_id = ?"
                mask_params.append(str(memory_bank_id))
            elif visible_ids is not None:
                if visible_ids:
                    mask_query += f" AND c.memory_bank_id IN ({','.join('?' for _ in visible_ids)})"
                    mask_params.extend(visible_ids)
                else:
                    mask_query += " AND 1=0"
            mask_count = int(conn.execute(mask_query, mask_params).fetchone()[0])
            profile_rows = conn.execute(
                f"SELECT COALESCE(memory_bank_id,'') AS profile_id, COUNT(*) AS count FROM feedback_candidates{active_where} GROUP BY memory_bank_id",
                params,
            ).fetchall()
            job_rows = conn.execute("SELECT status, COUNT(*) AS count FROM feedback_optimization_jobs GROUP BY status").fetchall()
            audit_deleted_count = int(conn.execute(
                "SELECT COUNT(*) FROM feedback_audit_log WHERE action IN ('feedback_physically_deleted','feedback_deleted_audit_retained')"
            ).fetchone()[0])
            trend_rows = conn.execute(
                f"SELECT substr(feedback_submitted_at,1,10) AS day,COUNT(*) AS count FROM feedback_candidates{active_where} GROUP BY substr(feedback_submitted_at,1,10) ORDER BY day",
                params,
            ).fetchall()
            usage_clauses = ["c.status<>'deleted'"]
            usage_params: list[Any] = []
            if memory_bank_id:
                usage_clauses.append("c.memory_bank_id=?")
                usage_params.append(str(memory_bank_id))
            elif visible_ids is not None:
                if visible_ids:
                    usage_clauses.append(f"c.memory_bank_id IN ({','.join('?' for _ in visible_ids)})")
                    usage_params.extend(visible_ids)
                else:
                    usage_clauses.append("1=0")
            legacy_used = sorted(self._USED_STATUSES)
            legacy_marks = ",".join("?" for _ in legacy_used)
            usage_params.extend(legacy_used)
            used_count = int(conn.execute(
                f"""
                SELECT COUNT(*) FROM feedback_candidates c
                WHERE {' AND '.join(usage_clauses)}
                  AND (
                    c.status IN ({legacy_marks})
                    OR EXISTS(
                        SELECT 1 FROM feedback_optimization_items i
                        LEFT JOIN feedback_optimization_jobs j ON j.job_id=i.job_id
                        WHERE i.candidate_id=c.candidate_id
                          AND (COALESCE(i.used_at,'')<>''
                               OR (i.status IN ('completed','ready_to_apply','applied')
                                   AND COALESCE(j.candidate_profile_id,'')<>''))
                    )
                  )
                """,
                usage_params,
            ).fetchone()[0])
        by_status: dict[str, int] = {}
        by_label: dict[str, int] = {}
        by_mode: dict[str, int] = {}
        by_issue_tag: dict[str, int] = {}
        for row in rows:
            by_status[row["status"]] = by_status.get(row["status"], 0) + int(row["count"])
            if str(row["status"] or "") != "withdrawn":
                by_label[row["corrected_label"] or "keep"] = by_label.get(row["corrected_label"] or "keep", 0) + int(row["count"])
            by_mode[row["mode"]] = by_mode.get(row["mode"], 0) + int(row["count"])
        for row in issue_rows:
            for tag in _json(row["issue_tags_json"], []):
                value = str(tag or "").strip()
                if value:
                    by_issue_tag[value] = by_issue_tag.get(value, 0) + int(row["count"])
        pending_count = sum(by_status.get(status, 0) for status in self._PENDING_STATUSES)
        approved_count = sum(by_status.get(status, 0) for status in self._REUSABLE_REVIEW_STATUSES)
        withdrawn_count = sum(by_status.get(status, 0) for status in self._WITHDRAWN_STATUSES)
        feedback_revision = self.revision()
        return {
            "total": total,
            "pending_count": pending_count,
            "approved_count": approved_count,
            "withdrawn_count": withdrawn_count,
            "used_count": used_count,
            "by_status": by_status,
            "by_label": by_label,
            "by_mode": by_mode,
            "by_issue_tag": by_issue_tag,
            "mask_count": mask_count,
            "without_mask_count": max(0, total - mask_count),
            "by_profile": {str(row["profile_id"] or "未关联"): int(row["count"]) for row in profile_rows},
            "optimization_jobs": {str(row["status"]): int(row["count"]) for row in job_rows},
            # Deleted feedback is no longer an ordinary record.  The audit
            # count is exposed under an explicit maintenance-only name so no
            # dashboard can accidentally present it as live feedback.
            "deleted_count": 0,
            "audit_deleted_count": audit_deleted_count,
            "by_day": {str(row["day"] or "未知日期"): int(row["count"]) for row in trend_rows},
            "feedback_revision": int(feedback_revision.get("revision") or 0),
            "source": "feedback_candidates",
            "scope": scope,
        }

    @staticmethod
    def active_memory_bank_ids(profiles: Iterable[dict[str, Any]]) -> tuple[str, ...]:
        """Return the exact target-domain scope exposed by the feedback UI."""
        identifiers: list[str] = []
        for profile in profiles:
            if bool(profile.get("archived")):
                continue
            memory_bank_id = str(profile.get("memory_bank_id") or profile.get("profile_id") or "").strip()
            if memory_bank_id:
                identifiers.append(memory_bank_id)
        return tuple(dict.fromkeys(identifiers))

    def resource_summaries(self, profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build target-domain feedback cards without exposing raw resource paths."""
        pending_statuses = tuple(sorted(self._PENDING_STATUSES))
        pending_marks = ",".join("?" for _ in pending_statuses)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT COALESCE(memory_bank_id, '') AS memory_bank_id,
                       COUNT(*) AS feedback_count,
                       SUM(CASE WHEN status IN ({pending_marks}) THEN 1 ELSE 0 END) AS pending_count,
                       SUM(CASE WHEN status='accepted' THEN 1 ELSE 0 END) AS approved_count,
                       SUM(CASE WHEN status='accepted'
                                 AND COALESCE(json_extract(context_json, '$.official_test_mode'), 0)=0
                                 AND COALESCE(json_extract(context_json, '$.feedback_optimization_locked'), 0)=0
                                 AND COALESCE(json_extract(context_json, '$.source_scope'), '')<>'official_test'
                                THEN 1 ELSE 0 END) AS updateable_count,
                       MAX(COALESCE(feedback_submitted_at, updated_at, created_at)) AS latest_feedback_at
                FROM feedback_candidates
                WHERE status <> 'deleted' AND COALESCE(memory_bank_id, '') <> ''
                GROUP BY memory_bank_id
                """,
                pending_statuses,
            ).fetchall()
        grouped = {str(row["memory_bank_id"]): dict(row) for row in rows}
        output: list[dict[str, Any]] = []
        visible_ids = set(self.active_memory_bank_ids(profiles))
        for profile in profiles:
            memory_bank_id = str(profile.get("memory_bank_id") or profile.get("profile_id") or "").strip()
            if memory_bank_id not in visible_ids:
                continue
            metadata = dict(profile.get("metadata") or {})
            counts = grouped.get(memory_bank_id, {})
            output.append(
                {
                    "memory_bank_id": memory_bank_id,
                    "source_profile_id": str(profile.get("profile_id") or memory_bank_id),
                    "source_target_domain_id": str(
                        metadata.get("target_domain_id")
                        or metadata.get("target_domain_version")
                        or profile.get("profile_id")
                        or memory_bank_id
                    ),
                    "friendly_name": str(
                        profile.get("friendly_name")
                        or profile.get("display_name")
                        or metadata.get("memory_bank_name")
                        or "目标域记忆库"
                    ),
                    "dataset": str(profile.get("dataset") or metadata.get("dataset") or ""),
                    "category": str(profile.get("category") or metadata.get("category") or ""),
                    "current_version": int(
                        profile.get("config_profile_version")
                        or metadata.get("config_profile_version")
                        or metadata.get("version")
                        or 1
                    ),
                    "feedback_count": int(counts.get("feedback_count") or 0),
                    "pending_count": int(counts.get("pending_count") or 0),
                    "approved_count": int(counts.get("approved_count") or 0),
                    # The landing cards intentionally use persisted eligibility
                    # facts only.  Full snapshot/filesystem validation remains
                    # lazy and runs after the operator enters the update tab.
                    "updateable_count": int(counts.get("updateable_count") or 0),
                    "latest_feedback_at": counts.get("latest_feedback_at"),
                    "frozen": bool(profile.get("frozen", metadata.get("frozen", False))),
                }
            )
        output.sort(
            key=lambda item: (
                int(item.get("feedback_count") or 0) > 0,
                str(item.get("latest_feedback_at") or ""),
                str(item.get("friendly_name") or ""),
            ),
            reverse=True,
        )
        return output

    def audit(self, *, candidate_id: str | None = None, job_id: str | None = None,
              memory_bank_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if candidate_id:
            clauses.append("a.candidate_id=?")
            params.append(candidate_id)
        if job_id:
            clauses.append("a.job_id=?")
            params.append(job_id)
        if memory_bank_id:
            clauses.append("c.memory_bank_id=?")
            params.append(memory_bank_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        join = " LEFT JOIN feedback_candidates c ON c.candidate_id=a.candidate_id" if memory_bank_id else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT a.* FROM feedback_audit_log a{join}{where} ORDER BY a.created_at DESC LIMIT ?",
                [*params, min(500, max(1, int(limit)))],
            ).fetchall()
        return [{**dict(row), "details": _json(row["details_json"], {})} for row in rows]
