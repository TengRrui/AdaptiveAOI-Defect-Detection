from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config import IntegrationSettings
from backend.database import DatabaseManager
from backend.migrations.detection_history_v0002_runtime_revision import apply as apply_runtime_revision_migration


class ResultStorage:
    """Small SQLite history store populated only by completed real inference jobs."""

    def __init__(self, settings: IntegrationSettings):
        self.settings = settings
        self.settings.ensure_runtime_dirs()
        self.path = self.settings.runtime_root / "detection_history.sqlite3"
        self.database = DatabaseManager(self.path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return self.database.connect()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS detection_results (
                    result_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    batch_id TEXT,
                    created_at TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    category TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    source_path TEXT,
                    prediction TEXT NOT NULL,
                    is_anomaly INTEGER NOT NULL,
                    anomaly_score REAL,
                    threshold REAL,
                    defect_type TEXT,
                    inference_time_ms REAL,
                    device TEXT,
                    original_image_url TEXT,
                    result_image_url TEXT,
                    marked_image_url TEXT,
                    heatmap_url TEXT,
                    mask_url TEXT,
                    result_json_url TEXT,
                    metadata_json TEXT NOT NULL,
                    runtime_revision INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_detection_created_at ON detection_results(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_detection_model ON detection_results(model_id);
                CREATE INDEX IF NOT EXISTS idx_detection_prediction ON detection_results(prediction);
                CREATE INDEX IF NOT EXISTS idx_detection_batch_created
                ON detection_results(batch_id, created_at DESC, result_id);
                CREATE INDEX IF NOT EXISTS idx_detection_dataset_category_created
                ON detection_results(dataset, category, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_detection_task_created
                ON detection_results(json_extract(metadata_json, '$.task_id'), created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_detection_mode_status_created
                ON detection_results(
                    json_extract(metadata_json, '$.mode'),
                    json_extract(metadata_json, '$.status'),
                    created_at DESC
                );
                CREATE TABLE IF NOT EXISTS statistics_state (
                    singleton_id INTEGER PRIMARY KEY CHECK(singleton_id = 1),
                    revision INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT ''
                );
                INSERT OR IGNORE INTO statistics_state(singleton_id, revision, updated_at, reason)
                VALUES (1, 0, '', 'initialized');
                CREATE TABLE IF NOT EXISTS detection_history_tombstones (
                    record_key TEXT PRIMARY KEY,
                    deleted_at TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT 'history_deleted'
                );
                CREATE INDEX IF NOT EXISTS idx_history_tombstones_deleted_at
                ON detection_history_tombstones(deleted_at DESC);
                CREATE TABLE IF NOT EXISTS feedback_records (
                    feedback_id TEXT PRIMARY KEY,
                    result_id TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    note TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS inspection_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    feedback_key TEXT NOT NULL UNIQUE,
                    mode TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    result_id TEXT,
                    frame_index INTEGER,
                    timestamp_ms REAL,
                    assessment TEXT NOT NULL,
                    corrected_label TEXT NOT NULL,
                    corrected_defect_type TEXT,
                    issue_tags_json TEXT NOT NULL,
                    note TEXT,
                    application_mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    context_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_feedback_task ON inspection_feedback(task_id, mode);
                CREATE INDEX IF NOT EXISTS idx_feedback_sample ON inspection_feedback(sample_id, frame_index);
                CREATE INDEX IF NOT EXISTS idx_inspection_feedback_created
                ON inspection_feedback(created_at DESC, mode, status);
                /*
                 * Feedback-driven optimization is deliberately kept beside the
                 * immutable detection history.  These tables are additive: old
                 * feedback rows remain readable and are never rewritten in place.
                 */
                CREATE TABLE IF NOT EXISTS feedback_candidates (
                    candidate_id TEXT PRIMARY KEY,
                    feedback_id TEXT NOT NULL,
                    feedback_key TEXT NOT NULL UNIQUE,
                    mode TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    result_id TEXT,
                    frame_index INTEGER,
                    timestamp_ms REAL,
                    dataset TEXT NOT NULL,
                    category TEXT NOT NULL,
                    model_version TEXT,
                    config_version TEXT,
                    config_profile_id TEXT,
                    config_profile_version INTEGER,
                    base_profile_id TEXT,
                    base_memory_bank_id TEXT,
                    base_config_version INTEGER,
                    memory_bank_id TEXT,
                    method_mode TEXT,
                    inference_mode TEXT,
                    original_prediction TEXT,
                    original_score REAL,
                    threshold REAL,
                    corrected_label TEXT,
                    corrected_defect_type TEXT,
                    assessment TEXT,
                    issue_tags_json TEXT NOT NULL DEFAULT '[]',
                    note TEXT,
                    status TEXT NOT NULL,
                    conflict_status TEXT NOT NULL DEFAULT 'none',
                    priority_score REAL NOT NULL DEFAULT 0,
                    priority_reason TEXT NOT NULL DEFAULT '',
                    source_artifact_ref TEXT,
                    source_sha256 TEXT,
                    perceptual_hash TEXT,
                    reviewer TEXT,
                    review_note TEXT,
                    reviewed_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    context_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback_annotations (
                    annotation_id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    annotation_type TEXT NOT NULL,
                    mask_path TEXT,
                    mask_artifact_ref TEXT,
                    mask_sha256 TEXT,
                    mask_width INTEGER,
                    mask_height INTEGER,
                    mask_format TEXT,
                    bbox_json TEXT,
                    polygon_json TEXT,
                    note TEXT,
                    created_by TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT,
                    UNIQUE(candidate_id, revision),
                    FOREIGN KEY(candidate_id) REFERENCES feedback_candidates(candidate_id)
                );
                CREATE TABLE IF NOT EXISTS feedback_optimization_jobs (
                    job_id TEXT PRIMARY KEY,
                    base_profile_id TEXT NOT NULL,
                    base_config_profile_id TEXT,
                    base_config_profile_version INTEGER,
                    candidate_ids_json TEXT NOT NULL,
                    candidate_count INTEGER NOT NULL DEFAULT 0,
                    strategy TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    update_threshold INTEGER NOT NULL DEFAULT 1,
                    update_normal_memory INTEGER NOT NULL DEFAULT 0,
                    update_defect_memory INTEGER NOT NULL DEFAULT 0,
                    update_lightweight_calibrator INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    stage TEXT NOT NULL DEFAULT 'queued',
                    message TEXT NOT NULL DEFAULT '',
                    constraints_json TEXT NOT NULL DEFAULT '{}',
                    candidate_profile_id TEXT,
                    candidate_config_profile_id TEXT,
                    candidate_config_profile_version INTEGER,
                    metrics_before_json TEXT NOT NULL DEFAULT '{}',
                    metrics_after_json TEXT NOT NULL DEFAULT '{}',
                    complexity_before_json TEXT NOT NULL DEFAULT '{}',
                    complexity_after_json TEXT NOT NULL DEFAULT '{}',
                    validation_json TEXT NOT NULL DEFAULT '{}',
                    error_code TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    applied_at TEXT,
                    rolled_back_at TEXT,
                    previous_profile_id TEXT,
                    previous_config_profile_id TEXT,
                    previous_config_profile_version INTEGER,
                    idempotency_key TEXT NOT NULL DEFAULT '',
                    current_stage_id TEXT NOT NULL DEFAULT '',
                    stages_json TEXT NOT NULL DEFAULT '[]',
                    timings_json TEXT NOT NULL DEFAULT '{}',
                    stage_started_at TEXT,
                    last_progress_at TEXT,
                    heartbeat_at TEXT
                );
                CREATE TABLE IF NOT EXISTS feedback_optimization_items (
                    job_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    message TEXT NOT NULL DEFAULT '',
                    result_code TEXT NOT NULL DEFAULT '',
                    components_json TEXT NOT NULL DEFAULT '[]',
                    started_at TEXT,
                    finished_at TEXT,
                    updated_at TEXT,
                    source_memory_bank_id TEXT,
                    target_memory_bank_id TEXT,
                    target_memory_bank_version TEXT,
                    strategy TEXT,
                    used_at TEXT,
                    usage_result TEXT,
                    generated_profile_id TEXT,
                    PRIMARY KEY(job_id, candidate_id),
                    FOREIGN KEY(job_id) REFERENCES feedback_optimization_jobs(job_id),
                    FOREIGN KEY(candidate_id) REFERENCES feedback_candidates(candidate_id)
                );
                CREATE TABLE IF NOT EXISTS feedback_audit_log (
                    audit_id TEXT PRIMARY KEY,
                    candidate_id TEXT,
                    job_id TEXT,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    from_status TEXT,
                    to_status TEXT,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                """
            )
            # Existing installations may have been initialized before the
            # red-box artifact was introduced.  Add the nullable column in
            # place so historical records remain readable and untouched.
            columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(detection_results)").fetchall()
            }
            if "marked_image_url" not in columns:
                conn.execute("ALTER TABLE detection_results ADD COLUMN marked_image_url TEXT")
            apply_runtime_revision_migration(conn)

    def save_result(self, result: dict[str, Any], request_id: str, batch_id: str | None = None) -> dict[str, Any]:
        result_id = str(result.get("result_id") or uuid.uuid4().hex)
        record = dict(result)
        record["result_id"] = result_id
        metadata = dict(record.get("metadata") or {})
        context = dict(metadata.get("detection_context") or record.get("detection_context") or {})
        raw_revision = record.get("runtime_revision")
        if raw_revision is None:
            raw_revision = metadata.get("runtime_revision")
        if raw_revision is None:
            raw_revision = context.get("runtime_revision")
        runtime_revision = int(raw_revision) if raw_revision is not None else None
        record["runtime_revision"] = runtime_revision
        if runtime_revision is not None:
            metadata["runtime_revision"] = runtime_revision
            if context:
                context["runtime_revision"] = runtime_revision
                metadata["detection_context"] = context
        record["metadata"] = metadata
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO detection_results (
                    result_id, request_id, batch_id, created_at, model_id, dataset, category, filename, source_path,
                    prediction, is_anomaly, anomaly_score, threshold, defect_type, inference_time_ms, device,
                    original_image_url, result_image_url, marked_image_url, heatmap_url, mask_url, result_json_url,
                    metadata_json, runtime_revision
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result_id,
                    request_id,
                    batch_id,
                    created_at,
                    str(record.get("model_id", "")),
                    str(record.get("metadata", {}).get("dataset", "")),
                    str(record.get("metadata", {}).get("category", "")),
                    str(record.get("metadata", {}).get("filename", "")),
                    str(record.get("metadata", {}).get("source_path", "")),
                    str(record.get("prediction", "")),
                    int(bool(record.get("is_anomaly"))),
                    record.get("anomaly_score"),
                    record.get("threshold"),
                    record.get("defect_type"),
                    record.get("inference_time_ms"),
                    str(record.get("device", "")),
                    record.get("original_image_url"),
                    record.get("result_image_url"),
                    record.get("marked_image_url"),
                    record.get("heatmap_url"),
                    record.get("mask_url"),
                    record.get("metadata", {}).get("result_json_url"),
                    json.dumps(metadata, ensure_ascii=False, default=str),
                    runtime_revision,
                ),
            )
            self.bump_statistics_revision(conn=conn, reason="detection_result_saved")
        record["created_at"] = created_at
        return record

    def statistics_revision(self) -> int:
        """Return the durable revision for every effective-history mutation."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT revision FROM statistics_state WHERE singleton_id=1"
            ).fetchone()
        return int(row["revision"] if row else 0)

    def bump_statistics_revision(
        self,
        *,
        conn: sqlite3.Connection | None = None,
        reason: str = "history_changed",
    ) -> int:
        """Advance the revision in the caller's transaction when provided."""
        own_connection = conn is None
        target = conn or self._connect()
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            target.execute(
                """
                INSERT INTO statistics_state(singleton_id, revision, updated_at, reason)
                VALUES(1, 1, ?, ?)
                ON CONFLICT(singleton_id) DO UPDATE SET
                    revision=statistics_state.revision + 1,
                    updated_at=excluded.updated_at,
                    reason=excluded.reason
                """,
                (now, str(reason or "history_changed")[:80]),
            )
            row = target.execute(
                "SELECT revision FROM statistics_state WHERE singleton_id=1"
            ).fetchone()
            if own_connection:
                target.commit()
            return int(row["revision"] if row else 0)
        finally:
            if own_connection:
                target.close()

    def mark_history_deleted(
        self,
        conn: sqlite3.Connection,
        record_keys: list[str],
        *,
        reason: str = "history_deleted",
    ) -> None:
        """Hide retained task manifests after their logical history was deleted."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows = [
            (str(key), now, str(reason or "history_deleted")[:80])
            for key in dict.fromkeys(record_keys)
            if str(key).strip()
        ]
        if rows:
            conn.executemany(
                """
                INSERT INTO detection_history_tombstones(record_key, deleted_at, reason)
                VALUES(?, ?, ?)
                ON CONFLICT(record_key) DO UPDATE SET
                    deleted_at=excluded.deleted_at,
                    reason=excluded.reason
                """,
                rows,
            )

    def deleted_history_keys(self) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT record_key FROM detection_history_tombstones"
            ).fetchall()
        return {str(row["record_key"]) for row in rows if row["record_key"]}

    @staticmethod
    def _row_to_result(row: sqlite3.Row) -> dict[str, Any]:
        payload = dict(row)
        try:
            metadata = json.loads(payload.pop("metadata_json") or "{}")
        except json.JSONDecodeError:
            metadata = {}
        payload["metadata"] = metadata
        payload["is_anomaly"] = bool(payload["is_anomaly"])
        revision = payload.get("runtime_revision")
        payload["runtime_revision"] = int(revision) if revision is not None else None
        payload["runtime_revision_display"] = (
            str(payload["runtime_revision"])
            if payload["runtime_revision"] is not None
            else "历史记录未记录"
        )
        if payload["runtime_revision"] is not None:
            metadata.setdefault("runtime_revision", payload["runtime_revision"])
        return payload

    def list_results(
        self,
        page: int = 1,
        page_size: int = 20,
        model_id: str | None = None,
        prediction: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        filename: str | None = None,
        defect_type: str | None = None,
    ) -> dict[str, Any]:
        conditions: list[str] = []
        params: list[Any] = []
        for field, value in (("model_id", model_id), ("prediction", prediction), ("defect_type", defect_type)):
            if value:
                conditions.append(f"{field} = ?")
                params.append(value)
        if date_from:
            conditions.append("created_at >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("created_at <= ?")
            params.append(date_to)
        if filename:
            conditions.append("filename LIKE ?")
            params.append(f"%{filename}%")
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        safe_page = max(1, int(page))
        safe_size = min(100, max(1, int(page_size)))
        with self._connect() as conn:
            total = int(conn.execute(f"SELECT COUNT(*) FROM detection_results{where}", params).fetchone()[0])
            rows = conn.execute(
                f"SELECT * FROM detection_results{where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                [*params, safe_size, (safe_page - 1) * safe_size],
            ).fetchall()
        return {"items": [self._row_to_result(row) for row in rows], "total": total, "page": safe_page, "page_size": safe_size}

    def statistics_rows(self, date_from: datetime | None = None, date_to: datetime | None = None) -> list[dict[str, Any]]:
        """Return real inference rows for backend-only aggregation.

        Time predicates are parameterized and evaluated in UTC.  The dashboard
        never receives source filesystem paths from this internal row set.
        """
        conditions: list[str] = []
        params: list[Any] = []
        if date_from is not None:
            conditions.append("created_at >= ?")
            params.append(date_from.astimezone(timezone.utc).isoformat(timespec="seconds"))
        if date_to is not None:
            conditions.append("created_at <= ?")
            params.append(date_to.astimezone(timezone.utc).isoformat(timespec="seconds"))
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        with self._connect() as conn:
            rows = conn.execute(f"SELECT * FROM detection_results{where} ORDER BY created_at DESC", params).fetchall()
        return [self._row_to_result(row) for row in rows]

    def get_result(self, result_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM detection_results WHERE result_id = ?", (result_id,)).fetchone()
        return self._row_to_result(row) if row else None

    def detection_feedback_context(
        self,
        result_id: str,
        *,
        task_id: str | None = None,
        mode: str | None = None,
        sample_id: str | None = None,
    ) -> dict[str, Any]:
        """Resolve immutable inference facts for image feedback on the server."""
        result = self.get_result(str(result_id or ""))
        if result is None:
            raise KeyError(f"Result not found: {result_id}")
        metadata = result.get("metadata") or {}
        stored_mode = "batch" if result.get("batch_id") else "single"
        if mode and mode != stored_mode:
            raise ValueError("反馈模式与检测结果类型不一致。")

        stored_task_id = str(result.get("batch_id") or metadata.get("task_id") or "").strip()
        requested_task_id = str(task_id or "").strip()
        if requested_task_id and stored_task_id and requested_task_id != stored_task_id:
            raise ValueError("检测任务与反馈样本不匹配。")
        if requested_task_id and not stored_task_id:
            manifest_path = self.settings.jobs_root / requested_task_id / "job_manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {}
            if str(manifest.get("result_id") or "") != str(result_id):
                raise ValueError("无法确认该样本属于指定检测任务。")
            stored_task_id = requested_task_id
        canonical_task_id = stored_task_id or f"legacy_result_{result_id}"
        canonical_sample_id = str(result_id)
        if sample_id and str(sample_id) != canonical_sample_id:
            raise ValueError("样本标识与检测结果不匹配。")

        calibration_value = str(metadata.get("calibration_path") or "").strip()
        config_version = Path(calibration_value).name if calibration_value else str(metadata.get("config_version") or "base")
        source_path = str(result.get("source_path") or metadata.get("source_path") or "").strip()
        profile_method_mode = str(
            metadata.get("profile_method_mode")
            or metadata.get("method_mode_resolved")
            or metadata.get("method_mode")
            or "normal_only"
        )
        scoring_method_mode = str(
            metadata.get("scoring_method_mode")
            or (metadata.get("core_result") or {}).get("method_mode")
            or "normal_memory"
        )
        memory_bank_id = str(metadata.get("memory_bank_id") or "").strip()
        config_profile_id = str(metadata.get("config_profile_id") or "").strip()
        try:
            config_profile_version = int(metadata.get("config_profile_version") or 0)
        except (TypeError, ValueError):
            config_profile_version = 0
        source_profile_id = str(
            metadata.get("source_profile_id")
            or metadata.get("profile_id")
            or memory_bank_id
            or ""
        ).strip()
        source_target_domain_id = str(
            metadata.get("source_target_domain_id")
            or metadata.get("target_domain_id")
            or metadata.get("target_domain_version")
            or source_profile_id
            or ""
        ).strip()
        source_config_snapshot_id = str(
            metadata.get("source_config_snapshot_id")
            or metadata.get("config_snapshot_id")
            or (
                f"{config_profile_id}:v{config_profile_version}"
                if config_profile_id and config_profile_version > 0
                else config_profile_id
            )
            or config_version
            or ""
        ).strip()
        return {
            "mode": stored_mode,
            "task_id": canonical_task_id,
            "sample_id": canonical_sample_id,
            "result_id": str(result_id),
            "frame_index": None,
            "timestamp_ms": None,
            "source_label": str(result.get("filename") or metadata.get("filename") or "检测图片"),
            "detection_time": str(result.get("created_at") or ""),
            "dataset": str(result.get("dataset") or metadata.get("dataset") or ""),
            "category": str(result.get("category") or metadata.get("category") or ""),
            "model_version": str(result.get("model_id") or ""),
            "config_version": config_version,
            "memory_bank_id": memory_bank_id,
            "source_memory_bank_id": memory_bank_id,
            "source_profile_id": source_profile_id,
            "source_target_domain_id": source_target_domain_id,
            "source_config_snapshot_id": source_config_snapshot_id,
            "threshold": result.get("threshold"),
            "prediction": str(result.get("prediction") or ""),
            "score": result.get("anomaly_score"),
            "configuration_frozen": bool(metadata.get("configuration_frozen", True)),
            "config_profile_id": config_profile_id,
            "config_profile_version": config_profile_version,
            "method_mode": profile_method_mode,
            "profile_method_mode": profile_method_mode,
            "scoring_method_mode": scoring_method_mode,
            "image_size": metadata.get("image_size"),
            "feature_dimension": metadata.get("feature_dimension"),
            "feature_layers": metadata.get("feature_layers") or [],
            "backbone": metadata.get("backbone"),
            "inference_mode": str(metadata.get("inference_mode") or "inference_only"),
            "official_test_mode": bool(metadata.get("official_test_mode", False)),
            "source_artifact_ref": f"result:{result_id}",
            # The absolute source path is retained only in the immutable result
            # row and is intentionally not returned to the browser.
            "_source_path": source_path,
        }

    def summary(self) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN is_anomaly = 1 THEN 1 ELSE 0 END) AS anomaly,
                       AVG(inference_time_ms) AS average_inference_time_ms
                FROM detection_results
                """
            ).fetchone()
            recent_rows = conn.execute("SELECT * FROM detection_results ORDER BY created_at DESC LIMIT 8").fetchall()
        total = int(row["total"] or 0)
        anomaly = int(row["anomaly"] or 0)
        normal = total - anomaly
        return {
            "total": total,
            "normal": normal,
            "anomaly": anomaly,
            "pass_rate": (normal / total * 100.0) if total else None,
            "average_inference_time_ms": float(row["average_inference_time_ms"]) if row["average_inference_time_ms"] is not None else None,
            "recent_results": [self._row_to_result(item) for item in recent_rows],
            "source": "detection_history.sqlite3",
            "statistics_revision": self.statistics_revision(),
        }

    @staticmethod
    def _row_to_feedback(row: sqlite3.Row) -> dict[str, Any]:
        payload = dict(row)
        try:
            payload["issue_tags"] = json.loads(payload.pop("issue_tags_json") or "[]")
        except json.JSONDecodeError:
            payload["issue_tags"] = []
        try:
            payload["context"] = json.loads(payload.pop("context_json") or "{}")
        except json.JSONDecodeError:
            payload["context"] = {}
        if payload.get("frame_index") == -1:
            payload["frame_index"] = None
        return payload

    @staticmethod
    def _feedback_key(context: dict[str, Any]) -> str:
        """Return the identity of one feedbackable detection event.

        Image content, filenames and source paths are deliberately excluded:
        the same image may be detected again with a different Memory Bank or
        Config and that new result must be allowed to create new feedback.
        """
        result_id = str(context.get("result_id") or "").strip()
        if result_id:
            return f"result:{result_id}"
        mode = str(context.get("mode") or "").strip()
        task_id = str(context.get("task_id") or "").strip()
        frame_index = context.get("frame_index")
        if mode == "video" and task_id and frame_index is not None:
            return f"video:{task_id}:frame:{int(frame_index)}"
        sample_id = str(context.get("sample_id") or "").strip()
        frame_key = int(frame_index) if frame_index is not None else -1
        return f"legacy:{mode}:{task_id}:{sample_id}:{frame_key}"

    def upsert_feedback(
        self,
        payload: dict[str, Any],
        context: dict[str, Any],
        *,
        connection: sqlite3.Connection | None = None,
    ) -> dict[str, Any]:
        """Save a correction beside, never over, the immutable detection record."""
        mode = str(context["mode"])
        task_id = str(context["task_id"])
        sample_id = str(context["sample_id"])
        frame_index = context.get("frame_index")
        frame_key = int(frame_index) if frame_index is not None else -1
        feedback_key = self._feedback_key(context)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        application_mode = str(payload.get("application_mode") or "save_only")
        assessment = str(payload.get("assessment") or "uncertain")
        corrected_label = str(payload.get("corrected_label") or "keep")
        status = (
            "pending_application"
            if application_mode == "calibration_candidate"
            else "confirmed"
            if assessment == "correct" and corrected_label == "keep"
            else "corrected"
            if assessment == "incorrect" or corrected_label != "keep"
            else "saved"
        )
        def persist(conn: sqlite3.Connection) -> dict[str, Any]:
            existing = conn.execute(
                "SELECT feedback_id, created_at FROM inspection_feedback WHERE feedback_key = ?",
                (feedback_key,),
            ).fetchone()
            feedback_id = str(existing["feedback_id"]) if existing else uuid.uuid4().hex
            created_at = str(existing["created_at"]) if existing else now
            values = (
                feedback_id,
                feedback_key,
                mode,
                task_id,
                sample_id,
                context.get("result_id"),
                frame_key,
                context.get("timestamp_ms"),
                assessment,
                corrected_label,
                payload.get("corrected_defect_type"),
                json.dumps(payload.get("issue_tags") or [], ensure_ascii=False),
                str(payload.get("note") or "").strip(),
                application_mode,
                status,
                created_at,
                now,
                json.dumps(context, ensure_ascii=False, default=str),
            )
            conn.execute(
                """
                INSERT INTO inspection_feedback (
                    feedback_id, feedback_key, mode, task_id, sample_id, result_id, frame_index,
                    timestamp_ms, assessment, corrected_label, corrected_defect_type, issue_tags_json,
                    note, application_mode, status, created_at, updated_at, context_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(feedback_key) DO UPDATE SET
                    result_id=excluded.result_id,
                    timestamp_ms=excluded.timestamp_ms,
                    assessment=excluded.assessment,
                    corrected_label=excluded.corrected_label,
                    corrected_defect_type=excluded.corrected_defect_type,
                    issue_tags_json=excluded.issue_tags_json,
                    note=excluded.note,
                    application_mode=excluded.application_mode,
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    context_json=excluded.context_json
                """,
                values,
            )
            row = conn.execute("SELECT * FROM inspection_feedback WHERE feedback_id = ?", (feedback_id,)).fetchone()
            assert row is not None
            return self._row_to_feedback(row)

        if connection is not None:
            return persist(connection)
        with self.database.transaction(immediate=True) as conn:
            return persist(conn)

    def list_feedback(
        self,
        *,
        task_id: str,
        mode: str | None = None,
        sample_id: str | None = None,
        frame_index: int | None = None,
    ) -> list[dict[str, Any]]:
        conditions = ["task_id = ?"]
        params: list[Any] = [task_id]
        if mode:
            conditions.append("mode = ?")
            params.append(mode)
        if sample_id:
            conditions.append("sample_id = ?")
            params.append(sample_id)
        if frame_index is not None:
            conditions.append("frame_index = ?")
            params.append(int(frame_index))
        where = " AND ".join(conditions)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM inspection_feedback WHERE {where} ORDER BY updated_at DESC LIMIT 500",
                params,
            ).fetchall()
        return [self._row_to_feedback(row) for row in rows]

    def get_feedback(self, feedback_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM inspection_feedback WHERE feedback_id = ?", (feedback_id,)).fetchone()
        return self._row_to_feedback(row) if row else None

    def save_feedback(self, result_id: str, feedback_type: str, note: str | None = None) -> dict[str, Any]:
        """Compatibility adapter for first-generation feedback callers."""
        mapping = {
            "false positive": ("incorrect", "normal", ["false_alarm"]),
            "false negative": ("incorrect", "defect", ["missed_detection"]),
            "true positive": ("correct", "keep", []),
            "true negative": ("correct", "keep", []),
        }
        if feedback_type not in mapping:
            raise ValueError("不支持的反馈类型。")
        assessment, corrected_label, issue_tags = mapping[feedback_type]
        context = self.detection_feedback_context(result_id)
        return self.upsert_feedback(
            {
                "assessment": assessment,
                "corrected_label": corrected_label,
                "issue_tags": issue_tags,
                "application_mode": "save_only",
                "note": note,
            },
            context,
        )
