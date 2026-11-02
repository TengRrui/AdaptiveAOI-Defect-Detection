from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from backend.errors import ApiError
from backend.database import DatabaseManager
from backend.adapters.result_serializer import artifact_url
from backend.services.detection_record_protection import candidate_protection_codes
from backend.services.feedback_task_status import ACTIVE_TASK_STATUSES


_MODE_LABELS = {"single": "单图检测", "batch": "批量检测", "video": "视频文件检测"}
_STATUS_LABELS = {
    "success": "成功",
    "failed": "失败",
    "cancelled": "已取消",
    "running": "运行中",
}
_PREDICTION_LABELS = {"normal": "正常", "anomaly": "缺陷", "failed": "失败", "unknown": "未记录"}
_RUNNING_STATUSES = {"running", "processing", "stopping", "paused", "queued"}
_DELETABLE_STATUSES = {"success", "failed", "cancelled"}
_PLAN_TTL_SECONDS = 30 * 60
_HISTORY_SORT_FIELDS = {"created_at", "score", "mode", "status", "prediction", "filename"}
_SQLITE_IN_CHUNK_SIZE = 500
_CLEANUP_PROGRESS_BATCH_SIZE = 25

_PROTECTION_DETAILS: dict[str, dict[str, str]] = {
    "task_running": {
        "label": "🔒 检测任务运行中",
        "message": "该检测任务尚未结束，当前不能删除其结果。",
        "resolution": "请等待任务完成，或先在对应检测页面取消任务。",
        "action": "detection",
    },
    "feedback_pending": {
        "label": "🔒 待审核反馈",
        "message": "该记录包含尚未完成审核的反馈，因此暂时不能删除。",
        "resolution": "请先前往反馈优化中心完成审核或撤销误提交反馈。",
        "action": "feedback_review",
    },
    "feedback_reviewing": {
        "label": "🔒 反馈审核处理中",
        "message": "该记录的人工纠错仍在处理，暂时不能删除。",
        "resolution": "请先在反馈优化中心完成审核、退回或撤销。",
        "action": "feedback_review",
    },
    "feedback_snapshot_pending": {
        "label": "🔒 等待独立样本快照",
        "message": "审核已经通过，但用于审计的独立样本快照尚未保存完成。",
        "resolution": "请在反馈优化中心重新完成审核快照，或撤销该反馈。",
        "action": "feedback_review",
    },
    "optimization_active": {
        "label": "🔒 记忆库更新任务运行中",
        "message": "该记录正在被反馈驱动的记忆库更新或验证任务使用。",
        "resolution": "请等待任务结束，或先取消相关更新任务。",
        "action": "feedback_update",
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _number(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_name(value: Any, fallback: str = "未命名") -> str:
    text = str(value or "").replace("\\", "/").rstrip("/")
    if not text:
        return fallback
    return Path(text).name or fallback


def _json(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, type(default)) else default
        except (TypeError, ValueError, json.JSONDecodeError):
            return default
    return value if isinstance(value, type(default)) else default


def _short_id(value: str) -> str:
    text = str(value or "")
    return text if len(text) <= 18 else f"{text[:8]}…{text[-6:]}"


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, float(percentile))) * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class HistoryManagementService:
    """Safe, auditable management of persisted detection history.

    The service intentionally knows only controlled task/result references. It
    never accepts a filesystem path from the client and never deletes inputs,
    datasets, profiles, memory banks, configurations, or documentation.
    """

    def __init__(self, settings: Any, storage: Any, logger: Any | None = None):
        self.settings = settings
        self.storage = storage
        self.logger = logger
        self.database = getattr(storage, "database", DatabaseManager(storage.path, timeout=15))
        self._lock = threading.RLock()
        self._file_cache: dict[str, list[Path]] = {}
        self._file_cache_signatures: dict[str, tuple[int, int]] = {}
        self._projection_cache: list[dict[str, Any]] | None = None
        self._projection_cache_signature: tuple[Any, ...] | None = None
        self._cleanup_threads: dict[str, threading.Thread] = {}
        self._cleanup_cancel_requested: set[str] = set()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        database = getattr(self, "database", None)
        if database is None:
            database = DatabaseManager(self.storage.path, timeout=15)
            self.database = database
        return database.connect()

    def _deleted_target_profile_ids(self) -> set[str]:
        """Return every deleted target-domain identity used by old/new rows."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT profile_id, resource_id
                    FROM runtime_resource_tombstones
                    WHERE profile_id IS NOT NULL
                      AND profile_id <> ''
                      AND resource_type IN ('target_profile', 'candidate_profile', 'normal_memory_bank')
                    """
                ).fetchall()
        except sqlite3.Error:
            return set()
        return {
            str(value)
            for row in rows
            for value in (row["profile_id"], row["resource_id"])
            if value
        }

    @staticmethod
    def _file_size(path: Path) -> int:
        try:
            return int(path.stat().st_size)
        except OSError:
            return 0

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS history_management_audit (
                    audit_id TEXT PRIMARY KEY,
                    operation TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    target_type TEXT NOT NULL DEFAULT 'detection_history',
                    delete_scope TEXT NOT NULL DEFAULT '',
                    filter_json TEXT NOT NULL DEFAULT '{}',
                    affected_task_count INTEGER NOT NULL DEFAULT 0,
                    affected_result_count INTEGER NOT NULL DEFAULT 0,
                    protected_count INTEGER NOT NULL DEFAULT 0,
                    affected_file_count INTEGER NOT NULL DEFAULT 0,
                    cleanup_file_count INTEGER NOT NULL DEFAULT 0,
                    cleanup_failed_count INTEGER NOT NULL DEFAULT 0,
                    released_bytes INTEGER NOT NULL DEFAULT 0,
                    snapshot_name TEXT,
                    backup_path TEXT,
                    operator TEXT NOT NULL DEFAULT 'local_operator',
                    result TEXT,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS history_cleanup_queue (
                    cleanup_id TEXT PRIMARY KEY,
                    audit_id TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    expected_size INTEGER NOT NULL DEFAULT 0,
                    sha256 TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(audit_id) REFERENCES history_management_audit(audit_id)
                );
                CREATE INDEX IF NOT EXISTS idx_history_audit_created
                    ON history_management_audit(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_history_cleanup_status
                    ON history_cleanup_queue(status, updated_at DESC);
                CREATE TABLE IF NOT EXISTS history_cleanup_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    message TEXT NOT NULL DEFAULT '',
                    total_records INTEGER NOT NULL DEFAULT 0,
                    processed_records INTEGER NOT NULL DEFAULT 0,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    skipped_count INTEGER NOT NULL DEFAULT 0,
                    failed_count INTEGER NOT NULL DEFAULT 0,
                    total_files INTEGER NOT NULL DEFAULT 0,
                    processed_files INTEGER NOT NULL DEFAULT 0,
                    estimated_bytes INTEGER NOT NULL DEFAULT 0,
                    released_bytes INTEGER NOT NULL DEFAULT 0,
                    audit_id TEXT,
                    snapshot_name TEXT,
                    deleted_record_keys_json TEXT NOT NULL DEFAULT '[]',
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    finished_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_history_cleanup_jobs_updated
                    ON history_cleanup_jobs(updated_at DESC);
                CREATE TABLE IF NOT EXISTS history_delete_plans (
                    plan_id TEXT PRIMARY KEY,
                    operation_type TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    filter_json TEXT NOT NULL,
                    all_clear INTEGER NOT NULL DEFAULT 0,
                    delete_feedback INTEGER NOT NULL DEFAULT 0,
                    cleanup_files INTEGER NOT NULL DEFAULT 1,
                    selected_count INTEGER NOT NULL DEFAULT 0,
                    deletable_count INTEGER NOT NULL DEFAULT 0,
                    protected_count INTEGER NOT NULL DEFAULT 0,
                    result_count INTEGER NOT NULL DEFAULT 0,
                    file_count INTEGER NOT NULL DEFAULT 0,
                    estimated_bytes INTEGER NOT NULL DEFAULT 0,
                    plan_hash TEXT NOT NULL,
                    preview_token_hash TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    consumed_at TEXT,
                    finished_at TEXT
                );
                CREATE TABLE IF NOT EXISTS history_delete_plan_items (
                    plan_id TEXT NOT NULL,
                    record_key TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    preview_deletable INTEGER NOT NULL DEFAULT 0,
                    preview_protected_reason TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(plan_id, record_key),
                    FOREIGN KEY(plan_id) REFERENCES history_delete_plans(plan_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_history_delete_plans_status
                    ON history_delete_plans(status, expires_at);
                CREATE INDEX IF NOT EXISTS idx_history_delete_plan_items_plan
                    ON history_delete_plan_items(plan_id, ordinal);
                """
            )
            existing_columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(history_management_audit)").fetchall()
            }
            for column, definition in {
                "target_type": "TEXT NOT NULL DEFAULT 'detection_history'",
                "delete_scope": "TEXT NOT NULL DEFAULT ''",
                "backup_path": "TEXT",
                "operator": "TEXT NOT NULL DEFAULT 'local_operator'",
                "result": "TEXT",
                "affected_file_count": "INTEGER NOT NULL DEFAULT 0",
            }.items():
                if column not in existing_columns:
                    conn.execute(f"ALTER TABLE history_management_audit ADD COLUMN {column} {definition}")
            detection_columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(detection_results)").fetchall()
            }
            for column in ("created_at", "batch_id", "request_id", "model_id", "prediction"):
                if column in detection_columns:
                    conn.execute(
                        f"CREATE INDEX IF NOT EXISTS idx_detection_results_{column} "
                        f"ON detection_results({column})"
                    )
            conn.execute(
                """
                UPDATE history_cleanup_jobs
                SET status='failed', stage='failed',
                    message='后端服务曾在清理任务执行期间停止，请重新预览后再执行。',
                    error_message='cleanup_interrupted_by_service_restart',
                    updated_at=?, finished_at=?
                WHERE status IN ('queued','running','cancelling')
                """,
                (_utc_now(), _utc_now()),
            )
            conn.execute(
                """
                UPDATE history_cleanup_queue
                SET status='pending',
                    error_message='cleanup_interrupted_before_completion',
                    updated_at=?
                WHERE status IN ('processing','retrying')
                """,
                (_utc_now(),),
            )
            conn.execute(
                """
                UPDATE history_delete_plans
                SET status='interrupted', finished_at=?
                WHERE status IN ('consumed','executing')
                """,
                (_utc_now(),),
            )

    @staticmethod
    def _cleanup_status_label(status: str) -> str:
        return {
            "queued": "等待中",
            "running": "运行中",
            "cancelling": "取消中",
            "cancelled": "已取消",
            "completed": "已完成",
            "partial": "部分完成",
            "failed": "失败",
        }.get(str(status), "未知")

    @staticmethod
    def _cleanup_stage_label(stage: str) -> str:
        return {
            "queued": "等待执行",
            "preparing": "准备删除范围",
            "snapshot": "创建数据库快照与文件清单",
            "database": "提交数据库事务",
            "files": "清理受控结果文件",
            "finalizing": "整理审计记录",
            "completed": "清理完成",
            "cancelled": "已取消",
            "failed": "清理失败",
        }.get(str(stage), "处理中")

    def _update_cleanup_job(
        self,
        job_id: str,
        *,
        conn: sqlite3.Connection | None = None,
        **values: Any,
    ) -> None:
        if not values:
            return
        allowed = {
            "status", "stage", "message", "total_records", "processed_records",
            "success_count", "skipped_count", "failed_count", "total_files",
            "processed_files", "estimated_bytes", "released_bytes", "audit_id",
            "snapshot_name", "deleted_record_keys_json", "error_message", "finished_at",
        }
        payload = {key: value for key, value in values.items() if key in allowed}
        payload["updated_at"] = _utc_now()
        assignments = ", ".join(f"{key}=?" for key in payload)
        if conn is not None:
            conn.execute(
                f"UPDATE history_cleanup_jobs SET {assignments} WHERE job_id=?",
                [*payload.values(), job_id],
            )
            return
        with self._connect() as target:
            target.execute(
                f"UPDATE history_cleanup_jobs SET {assignments} WHERE job_id=?",
                [*payload.values(), job_id],
            )

    def cleanup_job(self, job_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM history_cleanup_jobs WHERE job_id=?", (str(job_id),)).fetchone()
        if row is None:
            raise ApiError("HISTORY_CLEANUP_JOB_NOT_FOUND", "未找到该检测记录清理任务。", 404)
        item = dict(row)
        total_records = int(item.get("total_records") or 0)
        total_files = int(item.get("total_files") or 0)
        processed_records = int(item.get("processed_records") or 0)
        processed_files = int(item.get("processed_files") or 0)
        if total_files:
            progress = 0.35 + 0.65 * min(1.0, processed_files / total_files)
        elif total_records:
            progress = 0.1 + 0.6 * min(1.0, processed_records / total_records)
        else:
            progress = 0.0
        if item["status"] in {"completed", "partial", "cancelled", "failed"}:
            progress = 1.0 if item["status"] in {"completed", "partial"} else progress
        return {
            "cleanup_job_id": item["job_id"],
            "status": item["status"],
            "status_label": self._cleanup_status_label(item["status"]),
            "stage": item["stage"],
            "stage_label": self._cleanup_stage_label(item["stage"]),
            "message": item.get("message") or "",
            "progress": round(max(0.0, min(1.0, progress)), 4),
            "total_records": total_records,
            "processed_records": processed_records,
            "success_count": int(item.get("success_count") or 0),
            "skipped_count": int(item.get("skipped_count") or 0),
            "failed_count": int(item.get("failed_count") or 0),
            "total_files": total_files,
            "processed_files": processed_files,
            "estimated_bytes": int(item.get("estimated_bytes") or 0),
            "released_bytes": int(item.get("released_bytes") or 0),
            "audit_id": _short_id(item.get("audit_id") or ""),
            "snapshot_name": item.get("snapshot_name") or "",
            "deleted_record_keys": _json(item.get("deleted_record_keys_json"), []),
            "error_message": "清理任务执行失败，请查看审计记录。" if item.get("error_message") else "",
            "created_at": item.get("created_at"),
            "updated_at": item.get("updated_at"),
            "finished_at": item.get("finished_at"),
            "can_cancel": item["status"] == "queued",
        }

    def cleanup_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit or 20), 100))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT job_id FROM history_cleanup_jobs ORDER BY created_at DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
        return [self.cleanup_job(str(row["job_id"])) for row in rows]

    def cancel_cleanup_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute("SELECT status FROM history_cleanup_jobs WHERE job_id=?", (str(job_id),)).fetchone()
                if row is None:
                    raise ApiError("HISTORY_CLEANUP_JOB_NOT_FOUND", "未找到该检测记录清理任务。", 404)
                if str(row["status"]) != "queued":
                    raise ApiError("HISTORY_CLEANUP_NOT_CANCELLABLE", "该任务已开始执行，不能在数据库事务中途取消。", 409)
                self._cleanup_cancel_requested.add(str(job_id))
                conn.execute(
                    "UPDATE history_cleanup_jobs SET status='cancelled',stage='cancelled',message=?,updated_at=?,finished_at=? WHERE job_id=?",
                    ("任务已在开始前取消。", _utc_now(), _utc_now(), str(job_id)),
                )
        return self.cleanup_job(str(job_id))

    def _projection_signature(self) -> tuple[Any, ...]:
        """Cheaply detect history changes without walking result folders."""
        # The database also stores delete plans, cleanup jobs and audit rows.
        # Their writes must not invalidate the expensive detection projection.
        # ``statistics_revision`` advances only for effective history/resource
        # dependency changes and is updated in the same transaction as deletes.
        history_revision = int(self.storage.statistics_revision())
        manifests: list[tuple[str, int, int]] = []
        root = self.settings.jobs_root
        if root.exists():
            for directory in root.iterdir():
                if not directory.is_dir() or directory.name in {"domain_video_samples", "feedback_candidates"}:
                    continue
                path = directory / "job_manifest.json"
                try:
                    stat = path.stat()
                except OSError:
                    continue
                manifests.append((directory.name, stat.st_mtime_ns, stat.st_size))
        manifests.sort()
        return (history_revision, tuple(manifests))

    def _invalidate_projection(self) -> None:
        with self._lock:
            self._projection_cache = None
            self._projection_cache_signature = None
            self._file_cache.clear()
            self._file_cache_signatures.clear()

    def resource_dependency_changed(self, reason: str = "resource_dependency_changed") -> int:
        """Invalidate history protection and dashboard statistics after resource lifecycle changes."""
        self._invalidate_projection()
        return int(self.storage.bump_statistics_revision(reason=str(reason)))

    def _records_projection(self, *, force: bool = False) -> list[dict[str, Any]]:
        signature = self._projection_signature()
        with self._lock:
            if not force and self._projection_cache is not None and signature == self._projection_cache_signature:
                return self._projection_cache
        records = self._build_records()
        records.sort(
            key=lambda item: _parse_datetime(item.get("created_at"))
            or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        with self._lock:
            self._projection_cache = records
            self._projection_cache_signature = signature
        return records

    # ------------------------------------------------------------------
    # Read-side record projection
    # ------------------------------------------------------------------
    def _load_rows(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM detection_results ORDER BY created_at DESC").fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["_metadata"] = _json(item.pop("metadata_json", "{}"), {})
            output.append(item)
        return output

    def _load_manifests(self) -> list[dict[str, Any]]:
        root = self.settings.jobs_root
        if not root.exists():
            return []
        deleted_record_keys = self.storage.deleted_history_keys()
        manifests: list[dict[str, Any]] = []
        for directory in root.iterdir():
            if not directory.is_dir() or directory.name in {"domain_video_samples", "feedback_candidates"}:
                continue
            path = directory / "job_manifest.json"
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or str(payload.get("kind") or "") not in {"single", "batch", "video"}:
                continue
            if (
                directory.name in deleted_record_keys
                or str(payload.get("job_id") or "") in deleted_record_keys
                or str(payload.get("result_id") or "") in deleted_record_keys
            ):
                continue
            payload["_job_root"] = directory
            payload["_created_at"] = _parse_datetime(payload.get("created_at") or payload.get("updated_at"))
            manifests.append(payload)
        return manifests

    @staticmethod
    def _normal_status(value: Any, has_result: bool = False) -> str:
        status = str(value or "").lower()
        if status in {"completed", "success", "succeeded", "done"}:
            return "success"
        if status in {"failed", "error"}:
            return "failed"
        if status in {"cancelled", "canceled", "stopped"}:
            return "cancelled"
        if status in _RUNNING_STATUSES:
            return "running"
        return "success" if has_result else "failed"

    @staticmethod
    def _row_metadata(row: dict[str, Any]) -> dict[str, Any]:
        metadata = row.get("_metadata")
        return metadata if isinstance(metadata, dict) else {}

    def _task_rows(self, rows: Iterable[dict[str, Any]], task_id: str, mode: str) -> list[dict[str, Any]]:
        task = str(task_id or "")
        output: list[dict[str, Any]] = []
        for row in rows:
            metadata = self._row_metadata(row)
            if mode == "batch" and str(row.get("batch_id") or "") == task:
                output.append(row)
            elif mode == "single" and (
                str(row.get("result_id") or "") == task
                or str(metadata.get("task_id") or "") == task
                or str(row.get("request_id") or "") == task
            ):
                output.append(row)
            elif mode == "video" and str(metadata.get("task_id") or "") == task:
                output.append(row)
        return output

    def _path_from_reference(self, reference: Any, task_root: Path | None) -> Path | None:
        if reference in (None, ""):
            return None
        text = str(reference)
        if text.startswith("/artifacts/"):
            candidate = self.settings.jobs_root / text[len("/artifacts/") :]
        else:
            candidate = Path(text)
            if not candidate.is_absolute() and task_root is not None:
                candidate = task_root / candidate
        try:
            resolved = candidate.resolve()
            jobs_root = self.settings.jobs_root.resolve()
            resolved.relative_to(jobs_root)
            if task_root is not None:
                resolved.relative_to(task_root.resolve())
            return resolved
        except (OSError, ValueError):
            return None

    def _collect_result_files(
        self,
        task_root: Path | None,
        rows: list[dict[str, Any]],
        manifest: dict[str, Any] | None,
    ) -> list[Path]:
        cache_key = str(task_root.resolve()) if task_root is not None else ""
        cache_signature: tuple[int, int] | None = None
        if task_root is not None:
            try:
                manifest_mtime = int((task_root / "job_manifest.json").stat().st_mtime_ns)
                results_root = (task_root / "results").resolve()
                results_mtime = int(results_root.stat().st_mtime_ns) if results_root.exists() else 0
                cache_signature = (manifest_mtime, results_mtime)
            except OSError:
                cache_signature = None
        if (
            cache_key
            and cache_key in self._file_cache
            and cache_signature is not None
            and self._file_cache_signatures.get(cache_key) == cache_signature
        ):
            return list(self._file_cache[cache_key])
        files: set[Path] = set()
        if task_root is not None:
            try:
                results_root = (task_root / "results").resolve()
                results_root.relative_to(self.settings.jobs_root.resolve())
                if results_root.exists():
                    files.update(path.resolve() for path in results_root.rglob("*") if path.is_file())
                manifest_path = (task_root / "job_manifest.json").resolve()
                if manifest_path.is_file():
                    files.add(manifest_path)
            except (OSError, ValueError):
                pass
        references: list[Any] = []
        for row in rows:
            references.extend(row.get(key) for key in ("result_image_url", "heatmap_url", "mask_url", "result_json_url"))
            metadata = self._row_metadata(row)
            references.extend([metadata.get("result_json_url"), metadata.get("visualization_artifacts")])
        if manifest:
            def walk(value: Any, key: str = "") -> None:
                if isinstance(value, dict):
                    for child_key, child_value in value.items():
                        if child_key not in {
                            "source_path", "input_video_path", "memory_path", "config_path",
                            "calibration_path", "image_path", "staged_input_path",
                        }:
                            walk(child_value, child_key)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        walk(item, key)
                elif isinstance(value, str) and (key.endswith("_path") or key.endswith("_url") or key in {"path", "file"}):
                    references.append(value)
            walk(manifest)
        for reference in references:
            if isinstance(reference, dict):
                for value in reference.values():
                    path = self._path_from_reference(value, task_root)
                    if path and path.is_file():
                        files.add(path)
            elif isinstance(reference, (list, tuple)):
                for value in reference:
                    path = self._path_from_reference(value, task_root)
                    if path and path.is_file():
                        files.add(path)
            else:
                path = self._path_from_reference(reference, task_root)
                if path and path.is_file():
                    files.add(path)
        result = sorted(files)
        if cache_key:
            self._file_cache[cache_key] = result
            if cache_signature is not None:
                self._file_cache_signatures[cache_key] = cache_signature
        return result

    @staticmethod
    def _manifest_has_visuals(manifest: dict[str, Any]) -> bool:
        selection = manifest.get("selection") or {}
        if selection.get("generate_visualizations") is False:
            return False
        visual_tokens = ("heatmap", "overlay", "mask", "visual", "preview", "result_video")

        def contains(value: Any, key: str = "") -> bool:
            if isinstance(value, dict):
                return any(contains(child, str(child_key)) for child_key, child in value.items())
            if isinstance(value, (list, tuple)):
                return any(contains(child, key) for child in value)
            return bool(value) and any(token in key.lower() for token in visual_tokens)

        return contains(manifest)

    def _ensure_record_files(self, record: dict[str, Any]) -> list[Path]:
        if record.get("_files_loaded"):
            return list(record.get("_file_paths") or [])
        with self._lock:
            if record.get("_files_loaded"):
                return list(record.get("_file_paths") or [])
            paths = self._collect_result_files(
                record.get("_job_root"),
                list(record.get("_task_rows") or []),
                record.get("_manifest") if isinstance(record.get("_manifest"), dict) else None,
            )
            record["_file_paths"] = paths
            record["_files_loaded"] = True
            record["file_bytes"] = sum(self._file_size(path) for path in paths if path.is_file())
            record["visual_count"] = sum(
                1 for item in paths if item.parent.name in {"visualizations", "preview", "video", "keyframes"}
            )
            return list(paths)

    def _feedback_index(
        self,
    ) -> tuple[dict[str, int], dict[str, int], set[str], set[str], dict[str, list[str]], dict[str, list[str]]]:
        """Return feedback counts and candidate result/task references.

        Candidate references are kept separate from ordinary feedback so an
        explicitly confirmed high-risk delete can remove a plain feedback
        record while still protecting samples waiting for review.
        """
        result_counts: dict[str, int] = {}
        task_counts: dict[str, int] = {}
        task_refs: set[str] = set()
        result_refs: set[str] = set()
        candidate_task_refs: dict[str, list[str]] = {}
        candidate_result_refs: dict[str, list[str]] = {}
        deleted_source_ids = self._deleted_target_profile_ids()
        with self._connect() as conn:
            for table in ("inspection_feedback", "feedback_records", "feedback_candidates"):
                try:
                    rows = conn.execute(
                        f"SELECT * FROM {table}" if table != "feedback_records" else "SELECT * FROM feedback_records"
                    ).fetchall()
                except sqlite3.Error:
                    rows = []
                for row in rows:
                    item = dict(row)
                    if str(item.get("status") or "").lower() == "deleted":
                        continue
                    task_id = str(item.get("task_id") or "")
                    result_id = str(item.get("result_id") or "")
                    if task_id:
                        task_refs.add(task_id)
                        task_counts[task_id] = task_counts.get(task_id, 0) + 1
                    if result_id:
                        result_refs.add(result_id)
                        result_counts[result_id] = result_counts.get(result_id, 0) + 1
                    if table == "feedback_candidates":
                        reasons = candidate_protection_codes(
                            item,
                            deleted_source_ids=deleted_source_ids,
                        )
                        for reason in reasons:
                            if reason and task_id:
                                candidate_task_refs.setdefault(task_id, []).append(reason)
                            if reason and result_id:
                                candidate_result_refs.setdefault(result_id, []).append(reason)
        return result_counts, task_counts, task_refs, result_refs, candidate_task_refs, candidate_result_refs

    def _optimization_refs(self) -> tuple[set[str], set[str], set[str]]:
        candidate_refs: set[str] = set()
        active_tasks: set[str] = set()
        active_results: set[str] = set()
        candidate_to_record: dict[str, tuple[str, str]] = {}
        with self._connect() as conn:
            try:
                candidates = conn.execute("SELECT candidate_id, task_id, result_id FROM feedback_candidates").fetchall()
                for row in candidates:
                    candidate_id = str(row["candidate_id"] or "")
                    candidate_refs.add(candidate_id)
                    candidate_to_record[candidate_id] = (str(row["task_id"] or ""), str(row["result_id"] or ""))
            except sqlite3.Error:
                candidates = []
            try:
                jobs = conn.execute("SELECT candidate_ids_json, status FROM feedback_optimization_jobs").fetchall()
            except sqlite3.Error:
                jobs = []
            active_jobs: set[str] = set()
            for row in jobs:
                ids = _json(row["candidate_ids_json"], [])
                if str(row["status"] or "") in ACTIVE_TASK_STATUSES:
                    active_jobs.update(str(item) for item in ids if item)
            for candidate_id in active_jobs:
                task_id, result_id = candidate_to_record.get(candidate_id, ("", ""))
                if task_id:
                    active_tasks.add(task_id)
                if result_id:
                    active_results.add(result_id)
        return candidate_refs, active_tasks, active_results

    def _related_feedback_counts(self, records: list[dict[str, Any]]) -> dict[str, int]:
        task_ids = sorted({str(item.get("task_id") or "") for item in records if item.get("task_id")})
        result_ids = sorted({result_id for item in records for result_id in item.get("result_ids", []) if result_id})
        counts = {"feedback": 0, "candidates": 0, "masks": 0, "optimization_jobs": 0}
        with self._connect() as conn:
            def where_for(columns: tuple[str, ...]) -> tuple[str, list[str]]:
                clauses: list[str] = []
                params: list[str] = []
                if task_ids:
                    marks = ",".join("?" for _ in task_ids)
                    clauses.append(f"({columns[0]} IN ({marks}))")
                    params.extend(task_ids)
                if result_ids and len(columns) > 1:
                    marks = ",".join("?" for _ in result_ids)
                    clauses.append(f"({columns[1]} IN ({marks}))")
                    params.extend(result_ids)
                return (" OR ".join(clauses) or "1=0"), params
            try:
                clause, params = where_for(("task_id", "result_id"))
                counts["feedback"] = int(conn.execute(f"SELECT COUNT(*) FROM inspection_feedback WHERE {clause}", params).fetchone()[0])
            except sqlite3.Error:
                pass
            candidate_ids: list[str] = []
            try:
                clause, params = where_for(("task_id", "result_id"))
                rows = conn.execute(f"SELECT candidate_id FROM feedback_candidates WHERE {clause}", params).fetchall()
                candidate_ids = [str(row[0]) for row in rows if row[0]]
                counts["candidates"] = len(candidate_ids)
            except sqlite3.Error:
                pass
            if candidate_ids:
                marks = ",".join("?" for _ in candidate_ids)
                try:
                    counts["masks"] = int(
                        conn.execute(
                            f"SELECT COUNT(*) FROM feedback_annotations WHERE candidate_id IN ({marks}) AND (mask_path IS NOT NULL OR mask_artifact_ref IS NOT NULL)",
                            candidate_ids,
                        ).fetchone()[0]
                    )
                except sqlite3.Error:
                    pass
                try:
                    jobs = conn.execute("SELECT candidate_ids_json FROM feedback_optimization_jobs").fetchall()
                    target = set(candidate_ids)
                    counts["optimization_jobs"] = sum(
                        bool(target.intersection(set(_json(row[0], [])))) for row in jobs
                    )
                except sqlite3.Error:
                    pass
        return counts

    def _build_records(self) -> list[dict[str, Any]]:
        rows = self._load_rows()
        manifests = self._load_manifests()
        by_result = {str(row.get("result_id")): row for row in rows if row.get("result_id")}
        by_batch: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            batch_id = str(row.get("batch_id") or "")
            if batch_id:
                by_batch.setdefault(batch_id, []).append(row)
        (
            feedback_counts,
            feedback_task_counts,
            feedback_tasks,
            feedback_results,
            candidate_tasks,
            candidate_results,
        ) = self._feedback_index()
        _candidate_refs, active_optimization_tasks, active_optimization_results = self._optimization_refs()
        deleted_target_profile_ids = self._deleted_target_profile_ids()
        represented: set[str] = set()
        records: list[dict[str, Any]] = []

        def add_record(manifest: dict[str, Any], mode: str, task_rows: list[dict[str, Any]]) -> None:
            task_id = str(manifest.get("job_id") or "")
            selection = manifest.get("selection") or {}
            summary = manifest.get("summary") or {}
            first = task_rows[0] if task_rows else {}
            first_meta = self._row_metadata(first)
            result_ids = [str(item.get("result_id")) for item in task_rows if item.get("result_id")]
            represented.update(result_ids)
            status = self._normal_status(manifest.get("status"), bool(task_rows) or bool(manifest.get("result_id")))
            created = manifest.get("_created_at") or _parse_datetime(first.get("created_at"))
            dataset = str(selection.get("dataset") or summary.get("dataset") or first.get("dataset") or first_meta.get("dataset") or "")
            category = str(selection.get("category") or summary.get("category") or first.get("category") or first_meta.get("category") or "")
            config = _safe_name(
                first_meta.get("config_version") or selection.get("config_profile_id")
                or summary.get("config_path") or selection.get("config") or "",
                "",
            )
            model = str(selection.get("model_id") or summary.get("model") or first.get("model_id") or "")
            device = str(selection.get("device") or summary.get("device") or first.get("device") or "")
            memory_bank = str(selection.get("memory_bank_id") or summary.get("memory_bank_id") or first_meta.get("memory_bank_id") or "")
            method_mode = str(
                first_meta.get("method_mode")
                or (first_meta.get("core_result") or {}).get("method_mode")
                or selection.get("method_mode")
                or (selection.get("memory_bank_profile") or {}).get("method_mode_resolved")
                or summary.get("method_mode")
                or ""
            )
            profile_id = str(first_meta.get("config_profile_id") or selection.get("config_profile_id") or summary.get("config_profile_id") or "")
            profile_name = str(
                (selection.get("memory_bank_profile") or {}).get("friendly_name")
                or summary.get("profile_friendly_name")
                or first_meta.get("profile_friendly_name")
                or ""
            )
            resource_deleted = bool(
                (memory_bank and memory_bank in deleted_target_profile_ids)
                or (profile_id and profile_id in deleted_target_profile_ids)
            )
            config_version = str(
                first_meta.get("config_profile_version")
                or selection.get("config_profile_version")
                or summary.get("calibration_version")
                or ""
            )
            revision_value = first.get("runtime_revision")
            if revision_value is None:
                revision_value = first_meta.get("runtime_revision")
            if revision_value is None:
                revision_value = (selection.get("detection_context") or {}).get("runtime_revision")
            if revision_value is None:
                revision_value = selection.get("runtime_revision")
            if revision_value is None:
                revision_value = (summary.get("detection_context") or {}).get("runtime_revision")
            if revision_value is None:
                revision_value = summary.get("runtime_revision")
            if revision_value is None:
                revision_value = manifest.get("runtime_revision")
            runtime_revision = int(revision_value) if revision_value is not None else None
            if mode == "single":
                result = task_rows[0] if task_rows else by_result.get(str(manifest.get("result_id") or ""))
                if result:
                    prediction = str(result.get("prediction") or "unknown").lower()
                    score = _number(result.get("anomaly_score"))
                    threshold = _number(result.get("threshold"))
                    filename = _safe_name(result.get("filename") or first_meta.get("filename"), "检测图片")
                    result_ids = [str(result.get("result_id"))]
                else:
                    prediction, score, threshold, filename = "failed", None, None, "检测图片"
                total_items = 1 if result else 0
                anomaly_count = int(prediction == "anomaly")
                normal_count = int(prediction == "normal")
                failed_count = int(prediction == "failed" or status == "failed")
            elif mode == "batch":
                normalized_predictions = [str(item.get("prediction") or "").lower() for item in task_rows]
                anomaly_count = sum(bool(item.get("is_anomaly")) or prediction == "anomaly" for item, prediction in zip(task_rows, normalized_predictions))
                # The management projection represents currently retained child
                # records, not the task's historical input count.  This keeps a
                # parent batch summary consistent after one or more children are
                # explicitly deleted.
                total_items = len(task_rows)
                prediction = "anomaly" if anomaly_count else ("normal" if task_rows else "failed")
                score = None
                threshold = _number(selection.get("threshold"))
                filename = _safe_name(first_meta.get("source_path"), "批量输入目录")
                normal_count = sum(
                    (not bool(item.get("is_anomaly")))
                    and prediction not in {"anomaly", "failed", "error"}
                    for item, prediction in zip(task_rows, normalized_predictions)
                )
                failed_result_count = sum(prediction in {"failed", "error"} for prediction in normalized_predictions)
                failed_count = failed_result_count
            else:
                anomaly_count = int(summary.get("anomaly_frames") or 0)
                total_items = int(summary.get("total_decoded_records") or summary.get("reported_input_frames") or 0)
                prediction = "anomaly" if anomaly_count else ("normal" if int(summary.get("successful_processed_frames") or 0) else "failed")
                score = _number(summary.get("highest_anomaly_score"))
                threshold = _number(summary.get("threshold") or selection.get("threshold"))
                filename = _safe_name(manifest.get("original_filename") or summary.get("input_video_path"), "视频文件")
                normal_count = int(summary.get("normal_frames") or max(0, int(summary.get("successful_processed_frames") or 0) - anomaly_count))
                failed_count = int(summary.get("failed_frames") or 0)
            timing = dict(summary.get("timing") or manifest.get("timing") or {})
            inference_stage = dict((timing.get("stages") or {}).get("inference_ms") or {})
            input_video = dict(summary.get("input_video_metadata") or {})
            p50_ms = _number(
                summary.get("p50_ms")
                or summary.get("p50_inference_ms")
                or inference_stage.get("p50_ms")
            )
            p95_ms = _number(
                summary.get("p95_ms")
                or summary.get("p95_inference_ms")
                or inference_stage.get("p95_ms")
            )
            throughput = _number(
                summary.get("throughput_images_per_second")
                or summary.get("throughput")
                or summary.get("processing_throughput_fps")
                or timing.get("full_pipeline_fps")
            )
            if mode == "batch":
                retained_timings = [
                    float(value)
                    for value in (_number(item.get("inference_time_ms")) for item in task_rows)
                    if value is not None and value >= 0
                ]
                if retained_timings:
                    p50_ms = _percentile(retained_timings, 0.50)
                    p95_ms = _percentile(retained_timings, 0.95)
                    average_ms = sum(retained_timings) / len(retained_timings)
                    throughput = 1000.0 / average_ms if average_ms > 0 else None
            pure_inference_fps = _number(summary.get("pure_inference_fps") or timing.get("pure_inference_fps"))
            full_processing_fps = _number(
                summary.get("full_pipeline_fps")
                or summary.get("processing_throughput_fps")
                or timing.get("full_pipeline_fps")
            )
            duration_seconds = _number(input_video.get("duration_seconds") or summary.get("duration_seconds"))
            # Browsing history must not recursively scan every result folder.
            # File paths, sizes and hashes are resolved only for records that
            # enter an explicit deletion preview.
            file_paths: list[Path] = []
            visual_count = int(self._manifest_has_visuals(manifest))
            feedback_count = max(
                feedback_task_counts.get(task_id, 0),
                sum(feedback_counts.get(ref, 0) for ref in result_ids),
            )
            has_feedback = bool(feedback_count or task_id in feedback_tasks or any(ref in feedback_results for ref in result_ids))
            protected_reasons: list[str] = []
            management_warnings: list[str] = []
            if status == "running":
                protected_reasons.append("task_running")
            candidate_reasons = list(candidate_tasks.get(task_id) or [])
            for ref in result_ids:
                candidate_reasons.extend(candidate_results.get(ref) or [])
            protected_reasons.extend(dict.fromkeys(candidate_reasons))
            if task_id in active_optimization_tasks or any(ref in active_optimization_results for ref in result_ids):
                protected_reasons.append("optimization_active")
            if status == "success" and mode == "single" and not task_rows and (
                not manifest.get("result_id") or str(manifest.get("result_id")) not in by_result
            ):
                management_warnings.append("数据库或任务清单存在不一致")
            if status == "success" and mode == "batch" and not task_rows and int(manifest.get("input_count") or manifest.get("total_count") or 0) > 0:
                management_warnings.append("数据库或任务清单存在不一致")
            protected_reasons = list(dict.fromkeys(protected_reasons))
            protection_details = [dict(code=code, **_PROTECTION_DETAILS[code]) for code in protected_reasons]
            stable_fallback_source = "|".join(
                (
                    mode,
                    str(manifest.get("_job_root") or ""),
                    str(manifest.get("created_at") or ""),
                    str(manifest.get("original_filename") or manifest.get("input_path") or ""),
                )
            )
            stable_fallback_key = (
                "orphan_"
                + hashlib.sha256(stable_fallback_source.encode("utf-8")).hexdigest()[:24]
            )
            record = {
                "record_key": task_id or (result_ids[0] if result_ids else stable_fallback_key),
                "task_id": task_id or (result_ids[0] if result_ids else ""),
                "mode": mode,
                "created_at": created.isoformat(timespec="seconds") if isinstance(created, datetime) else str(created or ""),
                "dataset": dataset,
                "category": category,
                "config": config,
                "model_version": model,
                "device": device,
                "memory_bank_id": memory_bank,
                "filename": filename,
                "status": status,
                "prediction": prediction if prediction in _PREDICTION_LABELS else "unknown",
                "score": score,
                "threshold": threshold,
                "total_items": total_items,
                "normal_count": normal_count,
                "anomaly_count": anomaly_count,
                "failed_count": failed_count,
                # Video frame summaries are persisted in the task manifest/
                # JSONL rather than one row per frame in detection_results.
                # Expose the real frame count to management while keeping the
                # DB result_ids separately for transactional row deletion.
                "result_count": total_items if mode == "video" else len(result_ids),
                "p50_ms": p50_ms,
                "p95_ms": p95_ms,
                "throughput": throughput,
                "pure_inference_fps": pure_inference_fps,
                "full_processing_fps": full_processing_fps,
                "duration_seconds": duration_seconds,
                "feedback_count": feedback_count,
                "visual_count": visual_count,
                "file_bytes": None,
                "method_mode": method_mode,
                "profile_id": profile_id,
                "profile_name": _safe_name(profile_name, "") if profile_name else "",
                "resource_deleted": resource_deleted,
                "config_version": config_version,
                "runtime_revision": runtime_revision,
                "runtime_revision_display": str(runtime_revision) if runtime_revision is not None else "历史记录未记录",
                "task_label": f"{_MODE_LABELS.get(mode, mode)} · {filename}",
                "protected": bool(protected_reasons),
                "protected_reasons": protected_reasons,
                "protection_details": protection_details,
                "management_warnings": management_warnings,
                "result_ids": result_ids,
                "_job_root": manifest.get("_job_root"),
                "_file_paths": file_paths,
                "_files_loaded": False,
                "_manifest": manifest,
                "_task_rows": task_rows,
            }
            records.append(record)

        for manifest in manifests:
            mode = str(manifest.get("kind") or "")
            task_id = str(manifest.get("job_id") or "")
            if mode == "single":
                result_id = str(manifest.get("result_id") or "")
                task_rows = []
                if result_id and result_id in by_result:
                    task_rows = [by_result[result_id]]
                else:
                    task_rows = self._task_rows(rows, task_id, mode)
            elif mode == "batch":
                task_rows = by_batch.get(task_id, [])
            else:
                task_rows = self._task_rows(rows, task_id, mode)
            add_record(manifest, mode, task_rows)

        for row in rows:
            result_id = str(row.get("result_id") or "")
            if not result_id or result_id in represented:
                continue
            metadata = self._row_metadata(row)
            task_id = str(metadata.get("task_id") or result_id)
            manifest = {
                "job_id": task_id,
                "kind": "single",
                "status": "completed",
                "created_at": row.get("created_at"),
                "result_id": result_id,
                "_created_at": _parse_datetime(row.get("created_at")),
                "_job_root": self._path_from_reference(row.get("result_json_url"), None).parents[2] if self._path_from_reference(row.get("result_json_url"), None) else None,
            }
            add_record(manifest, "single", [row])
        return records

    @staticmethod
    def _normalize_filters(payload: dict[str, Any] | None) -> dict[str, Any]:
        source = dict(payload or {})
        allowed_modes = {"all", "single", "batch", "video"}
        allowed_status = {"all", "success", "failed", "cancelled", "running"}
        allowed_prediction = {"all", "normal", "anomaly", "failed", "unknown"}
        allowed_feedback = {"all", "has_feedback", "no_feedback"}
        allowed_visual = {"all", "generated", "not_generated"}
        filters = {
            "start_time": str(source.get("start_time") or "").strip() or None,
            "end_time": str(source.get("end_time") or "").strip() or None,
            "mode": str(source.get("mode") or "all").strip().lower(),
            "status": str(source.get("status") or "all").strip().lower(),
            "prediction": str(source.get("prediction") or "all").strip().lower(),
            "feedback_state": str(source.get("feedback_state") or "all").strip().lower(),
            "visuals_state": str(source.get("visuals_state") or "all").strip().lower(),
            "dataset": str(source.get("dataset") or "").strip() or None,
            "category": str(source.get("category") or "").strip() or None,
            "config": str(source.get("config") or "").strip() or None,
            "config_version": str(source.get("config_version") or "").strip() or None,
            "method_mode": str(source.get("method_mode") or "").strip() or None,
            "profile_id": str(source.get("profile_id") or "").strip() or None,
            "model_version": str(source.get("model_version") or "").strip() or None,
            "memory_bank_id": str(source.get("memory_bank_id") or "").strip() or None,
            "filename_keyword": str(source.get("filename_keyword") or "").strip()[:200] or None,
            "task_id": str(source.get("task_id") or "").strip()[:160] or None,
            "score_min": _number(source.get("score_min")),
            "score_max": _number(source.get("score_max")),
        }
        if filters["mode"] not in allowed_modes:
            raise ApiError("INVALID_HISTORY_FILTER", "检测模式筛选条件无效。", 422)
        if filters["status"] not in allowed_status:
            raise ApiError("INVALID_HISTORY_FILTER", "任务状态筛选条件无效。", 422)
        if filters["prediction"] not in allowed_prediction:
            raise ApiError("INVALID_HISTORY_FILTER", "检测结果筛选条件无效。", 422)
        if filters["feedback_state"] not in allowed_feedback or filters["visuals_state"] not in allowed_visual:
            raise ApiError("INVALID_HISTORY_FILTER", "记录筛选条件无效。", 422)
        if filters["score_min"] is not None and filters["score_max"] is not None and filters["score_min"] > filters["score_max"]:
            raise ApiError("INVALID_HISTORY_FILTER", "异常分数下限不能高于上限。", 422)
        for key in ("start_time", "end_time"):
            if filters[key] and _parse_datetime(filters[key]) is None:
                raise ApiError("INVALID_HISTORY_FILTER", "时间筛选格式无效。", 422)
        start = _parse_datetime(filters["start_time"])
        end = _parse_datetime(filters["end_time"])
        if start and end and start > end:
            raise ApiError("INVALID_HISTORY_FILTER", "开始时间不得晚于结束时间。", 422)
        return filters

    @staticmethod
    def _matches(record: dict[str, Any], filters: dict[str, Any]) -> bool:
        if filters["mode"] != "all" and record["mode"] != filters["mode"]:
            return False
        if filters["status"] != "all" and record["status"] != filters["status"]:
            return False
        if filters["prediction"] != "all" and record["prediction"] != filters["prediction"]:
            return False
        for key in ("dataset", "category", "config", "config_version", "method_mode", "profile_id", "model_version", "memory_bank_id"):
            if filters[key] and str(record.get(key) or "") != filters[key]:
                return False
        if filters["filename_keyword"] and filters["filename_keyword"].casefold() not in str(record.get("filename") or "").casefold():
            return False
        if filters["task_id"] and filters["task_id"] != str(record.get("task_id") or ""):
            return False
        if filters["feedback_state"] == "has_feedback" and not record["feedback_count"]:
            return False
        if filters["feedback_state"] == "no_feedback" and record["feedback_count"]:
            return False
        if filters["visuals_state"] == "generated" and not record["visual_count"]:
            return False
        if filters["visuals_state"] == "not_generated" and record["visual_count"]:
            return False
        score = record.get("score")
        if filters["score_min"] is not None and (score is None or score < filters["score_min"]):
            return False
        if filters["score_max"] is not None and (score is None or score > filters["score_max"]):
            return False
        created = _parse_datetime(record.get("created_at"))
        start = _parse_datetime(filters["start_time"])
        end = _parse_datetime(filters["end_time"])
        if start and (created is None or created < start):
            return False
        if end and (created is None or created > end):
            return False
        return True

    def _filtered(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        return [item for item in self._records_projection() if self._matches(item, filters)]

    @staticmethod
    def _public_record(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "record_key": record["record_key"],
            "task_id_display": _short_id(record.get("task_id")),
            "task_label": record.get("task_label") or "检测任务",
            "mode": record["mode"],
            "mode_label": _MODE_LABELS.get(record["mode"], record["mode"]),
            "created_at": record.get("created_at") or "",
            "filename": record.get("filename") or "未命名",
            "dataset": record.get("dataset") or "-",
            "category": record.get("category") or "-",
            "config": record.get("config") or "-",
            "config_version": record.get("config_version") or "-",
            "method_mode": record.get("method_mode") or "-",
            "profile_id": _short_id(record.get("profile_id")) if record.get("profile_id") else "-",
            "profile_name": record.get("profile_name") or "-",
            "resource_deleted": bool(record.get("resource_deleted")),
            "resource_state": "原目标域资源已删除" if record.get("resource_deleted") else "可用或未记录",
            "model_version": record.get("model_version") or "-",
            "device": record.get("device") or "-",
            "memory_bank": _short_id(record.get("memory_bank_id")) if record.get("memory_bank_id") else "-",
            "runtime_revision": record.get("runtime_revision"),
            "runtime_revision_display": record.get("runtime_revision_display") or "历史记录未记录",
            "status": record["status"],
            "status_label": _STATUS_LABELS.get(record["status"], record["status"]),
            "prediction": record["prediction"],
            "prediction_label": _PREDICTION_LABELS.get(record["prediction"], "未记录"),
            "score": record.get("score"),
            "threshold": record.get("threshold"),
            "total_items": record.get("total_items", 0),
            "normal_count": record.get("normal_count", 0),
            "anomaly_count": record.get("anomaly_count", 0),
            "failed_count": record.get("failed_count", 0),
            "result_count": record.get("result_count", 0),
            "p50_ms": record.get("p50_ms"),
            "p95_ms": record.get("p95_ms"),
            "throughput": record.get("throughput"),
            "pure_inference_fps": record.get("pure_inference_fps"),
            "full_processing_fps": record.get("full_processing_fps"),
            "duration_seconds": record.get("duration_seconds"),
            "feedback_count": record.get("feedback_count", 0),
            "feedback_state": "已反馈" if record.get("feedback_count") else "未反馈",
            "visual_state": "结果可查看" if record.get("visual_count") else "未生成或已清理",
            "file_bytes": record.get("file_bytes", 0),
            "protected": bool(record.get("protected")),
            "protected_reason": "；".join(
                str(item.get("label") or "") for item in (record.get("protection_details") or [])
            ),
            "protection_details": list(record.get("protection_details") or []),
            "protection_codes": list(record.get("protected_reasons") or []),
            "management_warnings": list(record.get("management_warnings") or []),
            "management_state": (
                str((record.get("protection_details") or [{}])[0].get("label") or "🔒 受保护")
                if record.get("protected")
                else "— 无保护"
            ),
        }

    def summary(self, filters_payload: dict[str, Any] | None = None) -> dict[str, Any]:
        filters = self._normalize_filters(filters_payload)
        all_records = self._records_projection()
        records = [item for item in all_records if self._matches(item, filters)]
        records.sort(key=lambda item: _parse_datetime(item.get("created_at")) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        eligible = [item for item in records if not item.get("protected") and item.get("status") in _DELETABLE_STATUSES]
        mode_counts = {mode: sum(item["mode"] == mode for item in records) for mode in ("single", "batch", "video")}
        status_counts = {status: sum(item["status"] == status for item in records) for status in _STATUS_LABELS}
        feedback_count = sum(int(item.get("feedback_count") or 0) for item in records)
        created_values = sorted(
            value for value in (_parse_datetime(item.get("created_at")) for item in records) if value is not None
        )
        return {
            "total_records": len(records),
            "database_results": sum(int(item.get("result_count") or 0) for item in records),
            "normal_results": sum(int(item.get("normal_count") or 0) for item in records),
            "anomaly_results": sum(int(item.get("anomaly_count") or 0) for item in records),
            "failed_results": sum(int(item.get("failed_count") or 0) for item in records),
            "success_records": status_counts["success"],
            "failed_records": status_counts["failed"],
            "cancelled_records": status_counts["cancelled"],
            "running_records": status_counts["running"],
            "anomaly_records": sum(item["prediction"] == "anomaly" for item in records),
            "feedback_records": feedback_count,
            "visualized_records": sum(bool(item.get("visual_count")) for item in records),
            "mode_counts": mode_counts,
            "deletable_records": len(eligible),
            "protected_records": len(records) - len(eligible),
            "estimated_reclaim_bytes": None,
            "result_file_bytes": None,
            "database_bytes": self._file_size(self.storage.path),
            "earliest_record": created_values[0].isoformat(timespec="seconds") if created_values else None,
            "latest_record": created_values[-1].isoformat(timespec="seconds") if created_values else None,
            "all_time_records": len(all_records),
            "source": "detection_history.sqlite3 与受控任务清单",
        }

    def records(
        self,
        filters_payload: dict[str, Any] | None = None,
        *,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> dict[str, Any]:
        filters = self._normalize_filters(filters_payload)
        # Build the controlled projection once per request.  Rebuilding it for
        # every filter option repeatedly scans manifests, detection rows and
        # result folders, which made a paginated request appear to hang on a
        # real history database.  Pagination still limits the payload returned
        # to the UI while this single projection keeps the read path bounded.
        all_records = self._records_projection()
        items = [item for item in all_records if self._matches(item, filters)]
        safe_sort_by = str(sort_by or "created_at").strip().lower()
        safe_sort_order = str(sort_order or "desc").strip().lower()
        if safe_sort_by not in _HISTORY_SORT_FIELDS or safe_sort_order not in {"asc", "desc"}:
            raise ApiError("INVALID_HISTORY_SORT", "检测记录排序条件无效。", 422)

        def sort_value(item: dict[str, Any]) -> Any:
            value = item.get(safe_sort_by)
            if safe_sort_by == "created_at":
                return _parse_datetime(value) or datetime.min.replace(tzinfo=timezone.utc)
            if safe_sort_by == "score":
                return _number(value)
            return str(value or "").casefold()

        def has_sort_value(item: dict[str, Any]) -> bool:
            if safe_sort_by == "created_at":
                return _parse_datetime(item.get("created_at")) is not None
            if safe_sort_by == "score":
                return _number(item.get("score")) is not None
            return bool(str(item.get(safe_sort_by) or "").strip())

        present = [item for item in items if has_sort_value(item)]
        missing = [item for item in items if not has_sort_value(item)]
        present.sort(key=lambda item: str(item.get("record_key") or ""))
        present.sort(key=sort_value, reverse=safe_sort_order == "desc")
        items = present + missing
        safe_size = min(50, max(5, int(page_size)))
        total = len(items)
        pages = max(1, (total + safe_size - 1) // safe_size)
        safe_page = min(max(1, int(page)), pages)
        start = (safe_page - 1) * safe_size
        return {
            "items": [self._public_record(item) for item in items[start : start + safe_size]],
            "total": total,
            "page": safe_page,
            "page_size": safe_size,
            "total_pages": pages,
            "filter": filters,
            "sort_by": safe_sort_by,
            "sort_order": safe_sort_order,
            "filter_options": {
                key: sorted({str(item.get(key) or "") for item in all_records if item.get(key) not in (None, "")})
                for key in ("mode", "status", "prediction", "dataset", "category", "config", "config_version", "method_mode", "profile_id", "device", "model_version")
            },
        }

    # ------------------------------------------------------------------
    # Server-paged child records for batch and video tasks
    # ------------------------------------------------------------------
    def _task_manifest(self, task_id: str) -> tuple[dict[str, Any], Path]:
        safe_id = str(task_id or "").strip()
        if not safe_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in safe_id):
            raise ApiError("INVALID_TASK_ID", "检测任务标识无效。", 422)
        job_root = (self.settings.jobs_root / safe_id).resolve()
        try:
            job_root.relative_to(self.settings.jobs_root.resolve())
        except ValueError as exc:
            raise ApiError("INVALID_TASK_ID", "检测任务标识无效。", 422) from exc
        manifest_path = job_root / "job_manifest.json"
        if not manifest_path.exists():
            raise ApiError("HISTORY_TASK_NOT_FOUND", "检测任务记录不存在。", 404)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ApiError("HISTORY_TASK_INVALID", "检测任务记录无法读取。", 500) from exc
        if not isinstance(manifest, dict):
            raise ApiError("HISTORY_TASK_INVALID", "检测任务记录格式无效。", 500)
        return manifest, job_root

    @staticmethod
    def _feedback_filter_clause(value: str | None) -> str:
        normalized = str(value or "all").strip().lower()
        if normalized in {"all", "", "全部"}:
            return "all"
        if normalized in {"submitted", "feedback", "已反馈"}:
            return "submitted"
        if normalized in {"unsubmitted", "none", "未反馈"}:
            return "unsubmitted"
        raise ApiError("INVALID_FEEDBACK_FILTER", "反馈状态筛选条件无效。", 422)

    @staticmethod
    def _prediction_filter(value: str | None) -> str:
        normalized = str(value or "all").strip().lower()
        mapping = {
            "": "all", "all": "all", "全部": "all",
            "normal": "normal", "正常": "normal", "0": "normal",
            "anomaly": "anomaly", "abnormal": "anomaly", "defect": "anomaly", "缺陷": "anomaly", "1": "anomaly",
            "failed": "failed", "error": "failed", "失败": "failed",
        }
        if normalized not in mapping:
            raise ApiError("INVALID_PREDICTION_FILTER", "检测结果筛选条件无效。", 422)
        return mapping[normalized]

    def _batch_task_items(
        self,
        task_id: str,
        *,
        page: int,
        page_size: int,
        query: str | None,
        prediction: str | None,
        feedback_status: str | None,
    ) -> dict[str, Any]:
        safe_page = max(1, int(page))
        safe_size = min(100, max(5, int(page_size)))
        prediction_filter = self._prediction_filter(prediction)
        feedback_filter = self._feedback_filter_clause(feedback_status)
        conditions = ["d.batch_id = ?"]
        params: list[Any] = [str(task_id)]
        keyword = str(query or "").strip()
        if keyword:
            conditions.append("LOWER(d.filename) LIKE ?")
            params.append(f"%{keyword.casefold()}%")
        if prediction_filter == "normal":
            conditions.append("LOWER(d.prediction) IN ('normal','ok','0')")
        elif prediction_filter == "anomaly":
            conditions.append("LOWER(d.prediction) IN ('anomaly','abnormal','defect','ng','1')")
        elif prediction_filter == "failed":
            conditions.append("LOWER(d.prediction) IN ('failed','error')")
        feedback_exists = (
            "EXISTS(SELECT 1 FROM inspection_feedback f "
            "WHERE f.task_id=d.batch_id AND f.mode='batch' "
            "AND (f.result_id=d.result_id OR f.sample_id=d.result_id))"
        )
        if feedback_filter == "submitted":
            conditions.append(feedback_exists)
        elif feedback_filter == "unsubmitted":
            conditions.append(f"NOT {feedback_exists}")
        where = " AND ".join(conditions)
        with self._connect() as conn:
            total = int(conn.execute(f"SELECT COUNT(*) FROM detection_results d WHERE {where}", params).fetchone()[0])
            pages = max(1, (total + safe_size - 1) // safe_size)
            safe_page = min(safe_page, pages)
            rows = conn.execute(
                f"""
                SELECT d.*,
                       (SELECT f.status FROM inspection_feedback f
                        WHERE f.task_id=d.batch_id AND f.mode='batch'
                          AND (f.result_id=d.result_id OR f.sample_id=d.result_id)
                        ORDER BY f.updated_at DESC LIMIT 1) AS feedback_status
                FROM detection_results d
                WHERE {where}
                ORDER BY d.created_at ASC, d.result_id ASC
                LIMIT ? OFFSET ?
                """,
                [*params, safe_size, (safe_page - 1) * safe_size],
            ).fetchall()
        items: list[dict[str, Any]] = []
        for offset, row in enumerate(rows):
            item = dict(row)
            feedback_value = str(item.get("feedback_status") or "")
            items.append(
                {
                    "item_id": str(item.get("result_id") or ""),
                    "result_id": str(item.get("result_id") or ""),
                    "row_number": (safe_page - 1) * safe_size + offset + 1,
                    "mode": "batch",
                    "filename": str(item.get("filename") or "检测图片"),
                    "created_at": str(item.get("created_at") or ""),
                    "prediction": str(item.get("prediction") or "unknown").lower(),
                    "score": _number(item.get("anomaly_score")),
                    "threshold": _number(item.get("threshold")),
                    "inference_time_ms": _number(item.get("inference_time_ms")),
                    "runtime_revision": int(item["runtime_revision"]) if item.get("runtime_revision") is not None else None,
                    "runtime_revision_display": str(item["runtime_revision"]) if item.get("runtime_revision") is not None else "历史记录未记录",
                    "status": "failed" if str(item.get("prediction") or "").lower() in {"failed", "error"} else "success",
                    "feedback_status": feedback_value or "unsubmitted",
                    "feedback_label": "已反馈" if feedback_value else "未反馈",
                    "original_image_url": item.get("original_image_url"),
                    "result_image_url": item.get("result_image_url"),
                    "heatmap_url": item.get("heatmap_url"),
                    "mask_url": item.get("mask_url"),
                    "visualization_available": bool(item.get("result_image_url") or item.get("heatmap_url") or item.get("mask_url")),
                    "can_delete": True,
                }
            )
        return {
            "items": items,
            "total": total,
            "page": safe_page,
            "page_size": safe_size,
            "total_pages": max(1, (total + safe_size - 1) // safe_size),
            "mode": "batch",
            "task_id": str(task_id),
            "detail_source": "detection_history.sqlite3",
            "can_delete_items": True,
            "management_message": "批量子结果可独立删除；父任务统计会按保留结果重新计算。",
        }

    def _video_task_items(
        self,
        task_id: str,
        *,
        page: int,
        page_size: int,
        query: str | None,
        prediction: str | None,
        feedback_status: str | None,
        time_from_ms: float | None,
        time_to_ms: float | None,
    ) -> dict[str, Any]:
        manifest, job_root = self._task_manifest(task_id)
        summary = dict(manifest.get("summary") or {})
        parent_revision = (summary.get("detection_context") or {}).get("runtime_revision")
        if parent_revision is None:
            parent_revision = summary.get("runtime_revision")
        if parent_revision is None:
            parent_revision = (manifest.get("detection_context") or {}).get("runtime_revision")
        if parent_revision is None:
            parent_revision = (manifest.get("selection") or {}).get("runtime_revision")
        parent_revision = int(parent_revision) if parent_revision is not None else None
        jsonl_value = str(summary.get("jsonl_path") or "")
        path = Path(jsonl_value).resolve() if jsonl_value else (job_root / "results" / "video" / "frame_results.jsonl").resolve()
        try:
            path.relative_to(job_root)
        except ValueError as exc:
            raise ApiError("VIDEO_FRAME_INDEX_INVALID", "视频帧级明细路径不在任务目录内。", 500) from exc
        if not path.exists():
            return {
                "items": [], "total": 0, "page": 1, "page_size": max(5, int(page_size)), "total_pages": 1,
                "mode": "video", "task_id": str(task_id), "detail_source": "none", "frame_details_persisted": False,
                "can_delete_items": False, "management_message": "该任务没有保存帧级检测明细。",
            }
        feedback_rows = self.storage.list_feedback(task_id=str(task_id), mode="video")
        feedback_by_frame = {
            int(item.get("frame_index")): item
            for item in feedback_rows
            if item.get("frame_index") is not None
        }
        prediction_filter = self._prediction_filter(prediction)
        feedback_filter = self._feedback_filter_clause(feedback_status)
        keyword = str(query or "").strip().casefold()
        safe_size = min(100, max(5, int(page_size)))
        requested_page = max(1, int(page))
        requested_start = (requested_page - 1) * safe_size
        requested_end = requested_start + safe_size
        page_items: list[dict[str, Any]] = []
        tail_items: list[dict[str, Any]] = []
        matched_total = 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict) or str(row.get("status") or "success").lower() != "success":
                        continue
                    frame_index = int(row.get("frame_index", -1))
                    timestamp_ms = float(row.get("timestamp_ms") or 0.0)
                    normalized_prediction = self._prediction_filter(str(row.get("prediction") or row.get("prediction_label") or "all"))
                    if keyword and keyword not in str(frame_index).casefold() and keyword not in f"frame:{frame_index}":
                        continue
                    if prediction_filter != "all" and normalized_prediction != prediction_filter:
                        continue
                    if time_from_ms is not None and timestamp_ms < float(time_from_ms):
                        continue
                    if time_to_ms is not None and timestamp_ms > float(time_to_ms):
                        continue
                    feedback = feedback_by_frame.get(frame_index)
                    if feedback_filter == "submitted" and not feedback:
                        continue
                    if feedback_filter == "unsubmitted" and feedback:
                        continue
                    heatmap_path = str(row.get("heatmap_path") or "")
                    mask_path = str(row.get("mask_path") or "")
                    heatmap_value = artifact_url(heatmap_path, self.settings.jobs_root) if heatmap_path else None
                    mask_value = artifact_url(mask_path, self.settings.jobs_root) if mask_path else None
                    item = {
                        "item_id": f"frame:{frame_index}",
                        "frame_index": frame_index,
                        "timestamp_ms": timestamp_ms,
                        "mode": "video",
                        "filename": f"帧 {frame_index}",
                        "prediction": normalized_prediction,
                        "score": _number(row.get("anomaly_score") if row.get("anomaly_score") is not None else row.get("score")),
                        "threshold": _number(row.get("threshold")),
                        "inference_time_ms": _number(row.get("inference_time_ms")),
                        "runtime_revision": int(row["runtime_revision"]) if row.get("runtime_revision") is not None else parent_revision,
                        "runtime_revision_display": str(
                            int(row["runtime_revision"]) if row.get("runtime_revision") is not None else parent_revision
                        ) if (row.get("runtime_revision") is not None or parent_revision is not None) else "历史记录未记录",
                        "frame_pipeline_ms": _number(
                            row.get("frame_pipeline_ms")
                            if row.get("frame_pipeline_ms") is not None
                            else row.get("frame_total_ms")
                        ),
                        "status": "success",
                        "feedback_status": str((feedback or {}).get("status") or "unsubmitted"),
                        "feedback_label": "已反馈" if feedback else "未反馈",
                        "original_image_url": f"/api/history/management/tasks/{task_id}/frames/{frame_index}/image",
                        "result_image_url": None,
                        "heatmap_url": heatmap_value,
                        "mask_url": mask_value,
                        "visualization_available": bool(heatmap_value or mask_value),
                        "can_delete": False,
                    }
                    if requested_start <= matched_total < requested_end:
                        page_items.append(item)
                    tail_items.append(item)
                    if len(tail_items) > safe_size:
                        tail_items.pop(0)
                    matched_total += 1
        except OSError as exc:
            raise ApiError("VIDEO_FRAME_INDEX_UNREADABLE", "视频帧级明细无法读取。", 500) from exc
        total = matched_total
        pages = max(1, (total + safe_size - 1) // safe_size)
        safe_page = min(requested_page, pages)
        start = (safe_page - 1) * safe_size
        if safe_page == requested_page:
            items = page_items
        else:
            final_page_count = max(0, total - start)
            items = tail_items[-final_page_count:] if final_page_count else []
        for offset, item in enumerate(items):
            item["row_number"] = start + offset + 1
        return {
            "items": items,
            "total": total,
            "page": safe_page,
            "page_size": safe_size,
            "total_pages": pages,
            "mode": "video",
            "task_id": str(task_id),
            "detail_source": "frame_results.jsonl",
            "frame_details_persisted": True,
            "can_delete_items": False,
            "management_message": "帧级明细与结果视频、时间轴和 JSONL 共享顺序，当前仅支持查看、筛选和反馈，不支持单帧物理删除。",
        }

    def task_items(
        self,
        task_id: str,
        *,
        mode: str,
        page: int = 1,
        page_size: int = 20,
        query: str | None = None,
        prediction: str | None = None,
        feedback_status: str | None = None,
        time_from_ms: float | None = None,
        time_to_ms: float | None = None,
    ) -> dict[str, Any]:
        normalized_mode = str(mode or "").strip().lower()
        if normalized_mode == "batch":
            return self._batch_task_items(
                task_id, page=page, page_size=page_size, query=query,
                prediction=prediction, feedback_status=feedback_status,
            )
        if normalized_mode == "video":
            return self._video_task_items(
                task_id, page=page, page_size=page_size, query=query,
                prediction=prediction, feedback_status=feedback_status,
                time_from_ms=time_from_ms, time_to_ms=time_to_ms,
            )
        raise ApiError("UNSUPPORTED_TASK_ITEM_MODE", "只有批量和视频任务提供子级明细。", 422)

    def delete_task_items(
        self,
        task_id: str,
        item_ids: list[str],
        *,
        acknowledged: bool = False,
    ) -> dict[str, Any]:
        if not acknowledged:
            raise ApiError("CONFIRMATION_REQUIRED", "请先核对删除预览并勾选确认。", 422)
        wanted = list(dict.fromkeys(str(item) for item in item_ids if str(item).strip()))
        if not wanted:
            raise ApiError("EMPTY_ITEM_SELECTION", "请先选择需要删除的批量样本记录。", 422)
        marks = ",".join("?" for _ in wanted)
        deleted_source_ids = self._deleted_target_profile_ids()
        protected: list[dict[str, Any]] = []
        deletable: list[str] = []
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT result_id,filename FROM detection_results WHERE batch_id=? AND result_id IN ({marks})",
                [str(task_id), *wanted],
            ).fetchall()
            found = {str(row["result_id"]): dict(row) for row in rows}
            for result_id in wanted:
                if result_id not in found:
                    continue
                candidate_rows = conn.execute(
                    "SELECT * FROM feedback_candidates WHERE result_id=? AND status<>'deleted'",
                    (result_id,),
                ).fetchall()
                reasons: list[str] = []
                candidate_ids: list[str] = []
                for candidate_row in candidate_rows:
                    candidate = dict(candidate_row)
                    candidate_ids.append(str(candidate.get("candidate_id") or ""))
                    reasons.extend(candidate_protection_codes(candidate, deleted_source_ids=deleted_source_ids))
                if candidate_ids:
                    candidate_marks = ",".join("?" for _ in candidate_ids)
                    active = conn.execute(
                        f"""SELECT 1 FROM feedback_optimization_items i
                            JOIN feedback_optimization_jobs j ON j.job_id=i.job_id
                            WHERE i.candidate_id IN ({candidate_marks})
                              AND j.status IN ({','.join('?' for _ in ACTIVE_TASK_STATUSES)}) LIMIT 1""",
                        [*candidate_ids, *sorted(ACTIVE_TASK_STATUSES)],
                    ).fetchone()
                    if active:
                        reasons.append("optimization_active")
                reasons = list(dict.fromkeys(reason for reason in reasons if reason))
                if reasons:
                    protected.append(
                        {
                            "item_id": result_id,
                            "filename": str(found[result_id].get("filename") or "检测图片"),
                            "reasons": reasons,
                            "details": [dict(code=code, **_PROTECTION_DETAILS[code]) for code in reasons if code in _PROTECTION_DETAILS],
                        }
                    )
                else:
                    deletable.append(result_id)
            if deletable:
                delete_marks = ",".join("?" for _ in deletable)
                conn.execute(f"DELETE FROM detection_results WHERE result_id IN ({delete_marks})", deletable)
                self.storage.bump_statistics_revision(conn=conn, reason="batch_child_results_deleted")
        self._invalidate_projection()
        return {
            "task_id": str(task_id),
            "requested_count": len(wanted),
            "deleted_count": len(deletable),
            "deleted_item_ids": deletable,
            "protected_count": len(protected),
            "protected_items": protected,
            "statistics_revision": self.storage.statistics_revision(),
            "parent_policy": "父任务保留；数量、正常/缺陷/失败和耗时统计按剩余子记录重新计算，全部删除后显示 0 条有效结果。",
        }

    # ------------------------------------------------------------------
    # Preview, snapshots and controlled deletion
    # ------------------------------------------------------------------
    @staticmethod
    def _scope_records(records: list[dict[str, Any]], scope: str, record_keys: list[str]) -> list[dict[str, Any]]:
        if scope == "selected":
            wanted = {str(item) for item in record_keys if str(item)}
            return [item for item in records if item["record_key"] in wanted]
        if scope in {"current_filtered", "all"}:
            return list(records)
        raise ApiError("INVALID_HISTORY_SCOPE", "删除范围无效。", 422)

    @staticmethod
    def _can_delete(item: dict[str, Any], delete_feedback: bool) -> bool:
        if item.get("status") not in _DELETABLE_STATUSES:
            return False
        reasons = set(item.get("protected_reasons") or [])
        if not reasons:
            return True
        # A normal delete never cascades feedback.  With the explicit high-
        # risk option, plain feedback may be removed, but running work,
        # pending candidates and optimization/profile references stay locked.
        if delete_feedback and reasons.issubset({"plain_feedback"}):
            return True
        return False

    def _selection(
        self,
        filters_payload: dict[str, Any] | None,
        *,
        scope: str,
        record_keys: list[str] | None = None,
        delete_feedback: bool = False,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        filters = self._normalize_filters(filters_payload)
        all_records = self._records_projection()
        filtered = [item for item in all_records if self._matches(item, filters)]
        # Explicit record ids are the durable selection source across pagination.
        # Filtered and clear-all operations are both restricted by their frozen
        # filter set (for example, clearing only single-image history).
        selection_source = all_records if scope == "selected" else filtered
        selected = self._scope_records(selection_source, scope, record_keys or [])
        eligible = [item for item in selected if self._can_delete(item, delete_feedback)]
        protected = [item for item in selected if item not in eligible]
        return filters, selected, eligible, protected

    @staticmethod
    def _public_protected(item: dict[str, Any]) -> dict[str, Any]:
        public = HistoryManagementService._public_record(item)
        public["protected_reason"] = public.get("protected_reason") or "当前记录不可安全删除"
        return public

    @staticmethod
    def _delete_plan_digest(core: dict[str, Any], items: list[dict[str, Any]]) -> str:
        payload = {
            "core": core,
            "items": [
                {
                    "record_key": str(item.get("record_key") or ""),
                    "ordinal": int(item.get("ordinal") or 0),
                    "preview_deletable": bool(item.get("preview_deletable")),
                    "preview_protected_reason": str(item.get("preview_protected_reason") or ""),
                }
                for item in items
            ],
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _create_delete_plan(
        self,
        *,
        operation_type: str,
        scope: str,
        filters: dict[str, Any],
        selected: list[dict[str, Any]],
        eligible: list[dict[str, Any]],
        protected: list[dict[str, Any]],
        all_clear: bool,
        delete_feedback: bool,
        cleanup_files: bool,
        result_count: int,
        file_count: int,
        estimated_bytes: int,
    ) -> dict[str, Any]:
        plan_id = f"history_plan_{uuid.uuid4().hex}"
        preview_token = secrets.token_urlsafe(32)
        created_at = _utc_now()
        expires_at = datetime.fromtimestamp(
            time.time() + _PLAN_TTL_SECONDS, tz=timezone.utc
        ).isoformat(timespec="seconds")
        eligible_keys = {str(item.get("record_key") or "") for item in eligible}
        items = [
            {
                "record_key": str(item.get("record_key") or ""),
                "ordinal": ordinal,
                "preview_deletable": str(item.get("record_key") or "") in eligible_keys,
                "preview_protected_reason": "；".join(item.get("protected_reasons") or []),
            }
            for ordinal, item in enumerate(selected, start=1)
            if str(item.get("record_key") or "")
        ]
        core = {
            "plan_id": plan_id,
            "operation_type": operation_type,
            "scope": scope,
            "filters": filters,
            "all_clear": bool(all_clear),
            "delete_feedback": bool(delete_feedback),
            "cleanup_files": bool(cleanup_files),
            "selected_count": len(selected),
            "deletable_count": len(eligible),
            "protected_count": len(protected),
            "result_count": int(result_count),
            "file_count": int(file_count),
            "estimated_bytes": int(estimated_bytes),
            "created_at": created_at,
            "expires_at": expires_at,
        }
        plan_hash = self._delete_plan_digest(core, items)
        token_hash = hashlib.sha256(preview_token.encode("utf-8")).hexdigest()
        with self._connect() as conn:
            conn.execute(
                "UPDATE history_delete_plans SET status='expired' WHERE status='pending' AND expires_at<=?",
                (created_at,),
            )
            conn.execute(
                """
                INSERT INTO history_delete_plans(
                    plan_id,operation_type,scope,filter_json,all_clear,delete_feedback,cleanup_files,
                    selected_count,deletable_count,protected_count,result_count,file_count,estimated_bytes,
                    plan_hash,preview_token_hash,status,created_at,expires_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'pending',?,?)
                """,
                (
                    plan_id,
                    operation_type,
                    scope,
                    json.dumps(filters, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    int(bool(all_clear)),
                    int(bool(delete_feedback)),
                    int(bool(cleanup_files)),
                    len(selected),
                    len(eligible),
                    len(protected),
                    int(result_count),
                    int(file_count),
                    int(estimated_bytes),
                    plan_hash,
                    token_hash,
                    created_at,
                    expires_at,
                ),
            )
            conn.executemany(
                """
                INSERT INTO history_delete_plan_items(
                    plan_id,record_key,ordinal,preview_deletable,preview_protected_reason
                ) VALUES (?,?,?,?,?)
                """,
                [
                    (
                        plan_id,
                        item["record_key"],
                        item["ordinal"],
                        int(item["preview_deletable"]),
                        item["preview_protected_reason"],
                    )
                    for item in items
                ],
            )
        return {
            **core,
            "plan_hash": plan_hash,
            "plan_status": "pending",
            "preview_token": preview_token,
            "record_keys": [item["record_key"] for item in items],
        }

    def _consume_delete_plan(self, payload: dict[str, Any], *, all_clear: bool) -> dict[str, Any]:
        plan_id = str(payload.get("plan_id") or "").strip()
        preview_token = str(payload.get("preview_token") or "").strip()
        if not plan_id or not preview_token:
            raise ApiError("HISTORY_CONFIRMATION_REQUIRED", "请先生成删除预览。", 409)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM history_delete_plans WHERE plan_id=?",
                (plan_id,),
            ).fetchone()
            if row is None:
                raise ApiError("HISTORY_PLAN_NOT_FOUND", "删除计划不存在，请重新生成预览。", 409)
            plan_row = dict(row)
            if str(plan_row.get("status") or "") != "pending":
                raise ApiError("HISTORY_PLAN_ALREADY_USED", "删除计划已使用或失效，请重新生成预览。", 409)
            expires_at = _parse_datetime(plan_row.get("expires_at"))
            if expires_at is None or expires_at <= datetime.now(timezone.utc):
                conn.execute(
                    "UPDATE history_delete_plans SET status='expired', finished_at=? WHERE plan_id=?",
                    (_utc_now(), plan_id),
                )
                conn.commit()
                raise ApiError("HISTORY_CONFIRMATION_EXPIRED", "删除预览已过期，请重新生成预览。", 409)
            if bool(plan_row.get("all_clear")) != bool(all_clear):
                raise ApiError("HISTORY_PLAN_TYPE_MISMATCH", "删除计划类型不匹配，请重新生成预览。", 409)
            supplied_hash = hashlib.sha256(preview_token.encode("utf-8")).hexdigest()
            if not secrets.compare_digest(supplied_hash, str(plan_row.get("preview_token_hash") or "")):
                raise ApiError("HISTORY_CONFIRMATION_INVALID", "删除预览令牌无效。", 409)
            item_rows = conn.execute(
                """
                SELECT record_key,ordinal,preview_deletable,preview_protected_reason
                FROM history_delete_plan_items WHERE plan_id=? ORDER BY ordinal
                """,
                (plan_id,),
            ).fetchall()
            items = [dict(item) for item in item_rows]
            filters = _json(plan_row.get("filter_json"), {})
            core = {
                "plan_id": plan_id,
                "operation_type": str(plan_row.get("operation_type") or ""),
                "scope": str(plan_row.get("scope") or ""),
                "filters": filters,
                "all_clear": bool(plan_row.get("all_clear")),
                "delete_feedback": bool(plan_row.get("delete_feedback")),
                "cleanup_files": bool(plan_row.get("cleanup_files")),
                "selected_count": int(plan_row.get("selected_count") or 0),
                "deletable_count": int(plan_row.get("deletable_count") or 0),
                "protected_count": int(plan_row.get("protected_count") or 0),
                "result_count": int(plan_row.get("result_count") or 0),
                "file_count": int(plan_row.get("file_count") or 0),
                "estimated_bytes": int(plan_row.get("estimated_bytes") or 0),
                "created_at": str(plan_row.get("created_at") or ""),
                "expires_at": str(plan_row.get("expires_at") or ""),
            }
            if not secrets.compare_digest(
                self._delete_plan_digest(core, items), str(plan_row.get("plan_hash") or "")
            ):
                conn.execute(
                    "UPDATE history_delete_plans SET status='invalid', finished_at=? WHERE plan_id=?",
                    (_utc_now(), plan_id),
                )
                conn.commit()
                raise ApiError("HISTORY_PLAN_INTEGRITY_ERROR", "删除计划完整性校验失败，未执行删除。", 409)
            conn.execute(
                "UPDATE history_delete_plans SET status='consumed', consumed_at=? WHERE plan_id=? AND status='pending'",
                (_utc_now(), plan_id),
            )
            conn.commit()
        return {
            **core,
            "plan_hash": str(plan_row.get("plan_hash") or ""),
            "record_keys": [str(item.get("record_key") or "") for item in items],
            "plan_status": "consumed",
        }

    def preview(self, payload: dict[str, Any], *, all_clear: bool = False) -> dict[str, Any]:
        scope = "all" if all_clear else str(payload.get("scope") or "current_filtered")
        record_keys = [str(item) for item in payload.get("record_keys") or []][:500]
        delete_feedback = bool(payload.get("delete_feedback", False))
        cleanup_files = bool(payload.get("cleanup_files", True))
        filters, selected, eligible, protected = self._selection(
            payload.get("filters"), scope=scope, record_keys=record_keys, delete_feedback=delete_feedback
        )
        if cleanup_files:
            for record in eligible:
                self._ensure_record_files(record)
        files = sorted({path for item in eligible for path in item.get("_file_paths", []) if path.is_file()}) if cleanup_files else []
        mode_counts = {mode: sum(item["mode"] == mode for item in eligible) for mode in ("single", "batch", "video")}
        feedback_count = sum(int(item.get("feedback_count") or 0) for item in selected)
        running_count = sum(item.get("status") == "running" for item in selected)
        related = self._related_feedback_counts(selected)
        result_count = sum(int(item.get("result_count") or 0) for item in eligible)
        estimated_bytes = sum(self._file_size(path) for path in files)
        plan = self._create_delete_plan(
            operation_type="clear_all" if all_clear else "delete_records",
            scope=scope,
            filters=filters,
            selected=selected,
            eligible=eligible,
            protected=protected,
            all_clear=all_clear,
            delete_feedback=delete_feedback,
            cleanup_files=cleanup_files,
            result_count=result_count,
            file_count=len(files),
            estimated_bytes=estimated_bytes,
        )
        return {
            "plan_id": plan["plan_id"],
            "preview_token": plan["preview_token"],
            "confirmation_token": plan["preview_token"],
            "plan_hash": plan["plan_hash"],
            "plan_status": plan["plan_status"],
            "plan_created_at": plan["created_at"],
            "expires_at": plan["expires_at"],
            "expires_in_seconds": _PLAN_TTL_SECONDS,
            "scope": scope,
            "all_clear": all_clear,
            "selected_records": len(selected),
            "deletable_records": len(eligible),
            "protected_records": len(protected),
            "protected_items": [self._public_protected(item) for item in protected[:100]],
            "mode_counts": mode_counts,
            "result_count": result_count,
            "normal_result_count": sum(int(item.get("normal_count") or 0) for item in eligible),
            "anomaly_result_count": sum(int(item.get("anomaly_count") or 0) for item in eligible),
            "failed_result_count": sum(int(item.get("failed_count") or 0) for item in eligible),
            "feedback_count": max(feedback_count, related["feedback"]),
            "feedback_candidate_count": related["candidates"],
            "mask_annotation_count": related["masks"],
            "optimization_job_count": related["optimization_jobs"],
            "current_running_count": running_count,
            "file_count": len(files),
            "estimated_reclaim_bytes": estimated_bytes,
            "affects_statistics": bool(eligible),
            "affects_feedback_optimization": bool(
                feedback_count or related["candidates"] or related["optimization_jobs"]
            ),
            "affects_profile_validation": any(
                "optimization_active" in set(item.get("protected_reasons") or [])
                for item in protected
            ),
            "delete_feedback": delete_feedback,
            "cleanup_files": cleanup_files,
            "warning": "删除操作不可恢复；原始数据集、模型、记忆库、配置和文档不会被删除。",
        }

    def _backup_root(self) -> Path:
        return self.settings.project_root.parent / "AdaptiveAOI_Backup"

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _cleanup_artifact_matches(
        self,
        path: Path,
        *,
        expected_size: int,
        expected_sha256: str | None,
    ) -> tuple[bool, str | None]:
        """Refuse to unlink a path that no longer identifies the planned file."""
        try:
            if int(path.stat().st_size) != int(expected_size):
                return False, "结果文件已发生变化，已保留并等待人工复核。"
            if expected_sha256 and not secrets.compare_digest(self._sha256(path), str(expected_sha256)):
                return False, "结果文件已发生变化，已保留并等待人工复核。"
        except OSError:
            return False, "结果文件状态无法核验，已保留并等待重试。"
        return True, None

    def _create_snapshot(
        self,
        operation: str,
        scope: str,
        filters: dict[str, Any],
        selected: list[dict[str, Any]],
        eligible: list[dict[str, Any]],
        cleanup_files: bool,
        cleanup_job_id: str | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        root = self._backup_root()
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot = root / f"检测记录删除快照_{stamp}_{secrets.token_hex(4)}"
        snapshot.mkdir(parents=True, exist_ok=False)
        database_copy = snapshot / "detection_history.sqlite3"
        # Use SQLite's online backup API instead of copying the file bytes
        # directly.  This keeps the pre-delete snapshot transaction-safe even
        # when a WAL/journal is present or another reader has the database open.
        source_conn = self.database.connect()
        target_conn = DatabaseManager(database_copy).connect()
        try:
            source_conn.backup(target_conn)
            target_conn.commit()
        finally:
            target_conn.close()
            source_conn.close()
        file_manifest: list[dict[str, Any]] = []
        seen_files: set[str] = set()
        candidate_files = sorted(
            {
                path
                for item in eligible
                for path in (item.get("_file_paths", []) if cleanup_files else [])
                if path.is_file()
            }
        )
        if cleanup_job_id:
            self._update_cleanup_job(
                cleanup_job_id,
                stage="snapshot",
                message=f"正在创建安全快照并校验 {len(candidate_files)} 个结果文件。",
                total_files=len(candidate_files),
                estimated_bytes=sum(self._file_size(path) for path in candidate_files),
            )
        for item in eligible:
            for path in item.get("_file_paths", []) if cleanup_files else []:
                if not path.is_file():
                    continue
                try:
                    relative = path.resolve().relative_to(self.settings.jobs_root.resolve()).as_posix()
                except ValueError:
                    continue
                if relative in seen_files:
                    continue
                seen_files.add(relative)
                file_manifest.append(
                    {
                        "relative_path": relative,
                        "size_bytes": self._file_size(path),
                        "sha256": self._sha256(path),
                    }
                )
        public_selected = [self._public_record(item) for item in selected]
        snapshot_payload = {
            "created_at": _utc_now(),
            "operation": operation,
            "scope": scope,
            "filters": filters,
            "selected_records": public_selected,
            "eligible_record_keys": [item["record_key"] for item in eligible],
            "affected_task_count": len(eligible),
            "affected_result_count": sum(int(item.get("result_count") or 0) for item in eligible),
            "estimated_reclaim_bytes": sum(self._file_size(path) for item in eligible for path in item.get("_file_paths", []) if path.is_file()) if cleanup_files else 0,
            "operator": "local_operator",
            "audit_status": "snapshot_created_before_database_transaction",
            "database_backup": database_copy.name,
            "note": "仅保存数据库副本和结果文件清单，不复制原始输入、数据集、模型、记忆库或配置文件。",
        }
        (snapshot / "snapshot.json").write_text(json.dumps(snapshot_payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (snapshot / "file_manifest.json").write_text(json.dumps(file_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return snapshot.name, file_manifest

    @staticmethod
    def _delete_in_chunks(
        conn: sqlite3.Connection,
        *,
        table: str,
        column: str,
        values: Iterable[str],
    ) -> int:
        allowed = {
            ("detection_results", "result_id"),
            ("detection_results", "batch_id"),
            ("feedback_records", "result_id"),
            ("inspection_feedback", "result_id"),
            ("inspection_feedback", "task_id"),
        }
        if (table, column) not in allowed:
            raise ValueError("Unsupported history bulk-delete target")
        unique_values = [str(value) for value in dict.fromkeys(values) if str(value)]
        deleted = 0
        for offset in range(0, len(unique_values), _SQLITE_IN_CHUNK_SIZE):
            chunk = unique_values[offset : offset + _SQLITE_IN_CHUNK_SIZE]
            marks = ",".join("?" for _ in chunk)
            deleted += int(
                conn.execute(
                    f"DELETE FROM {table} WHERE {column} IN ({marks})",
                    chunk,
                ).rowcount
                or 0
            )
        return deleted

    def _delete_db_rows(self, conn: sqlite3.Connection, result_ids: list[str], task_ids: list[str], delete_feedback: bool) -> int:
        deleted = self._delete_in_chunks(
            conn,
            table="detection_results",
            column="result_id",
            values=result_ids,
        )
        if delete_feedback:
            self._delete_in_chunks(conn, table="feedback_records", column="result_id", values=result_ids)
            self._delete_in_chunks(conn, table="inspection_feedback", column="result_id", values=result_ids)
        deleted += self._delete_in_chunks(
            conn,
            table="detection_results",
            column="batch_id",
            values=task_ids,
        )
        if delete_feedback:
            self._delete_in_chunks(conn, table="inspection_feedback", column="task_id", values=task_ids)
        return deleted

    def _delete_sync(
        self,
        payload: dict[str, Any],
        *,
        all_clear: bool = False,
        delete_plan: dict[str, Any] | None = None,
        cleanup_job_id: str | None = None,
    ) -> dict[str, Any]:
        delete_plan = delete_plan or self._consume_delete_plan(payload, all_clear=all_clear)
        scope = str(delete_plan.get("scope") or ("all" if all_clear else "current_filtered"))
        filters = self._normalize_filters(delete_plan.get("filters"))
        delete_feedback = bool(delete_plan.get("delete_feedback", False))
        cleanup_files = bool(delete_plan.get("cleanup_files", True))
        selection_keys = [str(item) for item in delete_plan.get("record_keys") or [] if str(item)]
        _, selected, eligible, protected = self._selection(
            filters,
            scope="selected",
            record_keys=selection_keys,
            delete_feedback=delete_feedback,
        )
        selected_keys = {str(item.get("record_key") or "") for item in selected}
        missing_record_keys = [item for item in selection_keys if item not in selected_keys]
        skipped_record_count = len(protected) + len(missing_record_keys)
        for record in eligible:
            self._ensure_record_files(record)
        if cleanup_job_id:
            self._update_cleanup_job(
                cleanup_job_id,
                stage="preparing",
                message=(
                    f"已重新校验删除计划：{len(eligible)} 条可删除，"
                    f"{len(protected)} 条受保护，{len(missing_record_keys)} 条已不存在。"
                ),
                total_records=len(selection_keys),
            )
        try:
            snapshot_name, file_manifest = self._create_snapshot(
                str(delete_plan.get("operation_type") or ("clear_all" if all_clear else "delete_records")),
                scope,
                filters,
                selected,
                eligible,
                cleanup_files,
                cleanup_job_id,
            )
        except Exception as exc:
            if self.logger:
                self.logger.exception("history_management_snapshot_failed")
            raise ApiError("HISTORY_SNAPSHOT_FAILED", "数据库备份失败，未执行删除。", 500) from exc
        now = _utc_now()
        if cleanup_job_id:
            self._update_cleanup_job(
                cleanup_job_id,
                stage="database",
                message="安全快照已完成，正在提交检测历史数据库事务。",
                snapshot_name=snapshot_name,
            )
        try:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                planned_record_keys = [str(item.get("record_key") or "") for item in eligible]
                existing_tombstones: set[str] = set()
                for offset in range(0, len(planned_record_keys), _SQLITE_IN_CHUNK_SIZE):
                    chunk = planned_record_keys[offset : offset + _SQLITE_IN_CHUNK_SIZE]
                    if not chunk:
                        continue
                    marks = ",".join("?" for _ in chunk)
                    rows = conn.execute(
                        f"SELECT record_key FROM detection_history_tombstones WHERE record_key IN ({marks})",
                        chunk,
                    ).fetchall()
                    existing_tombstones.update(str(row["record_key"]) for row in rows)
                claimed_eligible = [
                    item
                    for item in eligible
                    if str(item.get("record_key") or "") not in existing_tombstones
                ]
                skipped_record_count += len(eligible) - len(claimed_eligible)
                eligible = claimed_eligible
                deleted_record_keys = [str(item.get("record_key") or "") for item in eligible]
                result_ids = sorted({result_id for item in eligible for result_id in item.get("result_ids", [])})
                task_ids = sorted({str(item.get("task_id") or "") for item in eligible})
                planned_result_count = sum(int(item.get("result_count") or 0) for item in eligible)

                claimed_files: set[str] = set()
                if cleanup_files:
                    jobs_root = self.settings.jobs_root.resolve()
                    for item in eligible:
                        for path in item.get("_file_paths", []):
                            try:
                                claimed_files.add(path.resolve().relative_to(jobs_root).as_posix())
                            except (OSError, ValueError):
                                continue
                file_manifest = [
                    item
                    for item in file_manifest
                    if str(item.get("relative_path") or "") in claimed_files
                ]
                audit_id = uuid.uuid4().hex if eligible else ""
                cleanup_queue_rows = [
                    (
                        uuid.uuid4().hex,
                        audit_id,
                        str(item.get("relative_path") or ""),
                        int(item.get("size_bytes") or 0),
                        item.get("sha256"),
                        "processing",
                        None,
                        now,
                        now,
                    )
                    for item in file_manifest
                ]
                if eligible:
                    deleted_results = self._delete_db_rows(conn, result_ids, task_ids, delete_feedback)
                    self.storage.mark_history_deleted(
                        conn,
                        deleted_record_keys,
                        reason=str(delete_plan.get("operation_type") or "history_deleted"),
                    )
                    statistics_revision = self.storage.bump_statistics_revision(
                        conn=conn,
                        reason=str(delete_plan.get("operation_type") or "history_deleted"),
                    )
                    conn.execute(
                        """
                        INSERT INTO history_management_audit(
                            audit_id, operation, scope, target_type, delete_scope, filter_json, affected_task_count,
                            affected_result_count, protected_count, affected_file_count, cleanup_file_count,
                            released_bytes, snapshot_name, backup_path, operator, result, status, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            audit_id,
                            str(delete_plan.get("operation_type") or ("clear_all" if all_clear else "delete_records")),
                            scope,
                            "detection_history",
                            scope,
                            json.dumps(filters, ensure_ascii=False, default=str),
                            len(eligible),
                            planned_result_count,
                            skipped_record_count,
                            len(file_manifest),
                            len(file_manifest),
                            sum(int(item.get("size_bytes") or 0) for item in file_manifest),
                            snapshot_name,
                            snapshot_name,
                            "local_operator",
                            None,
                            "db_committed_cleanup_pending",
                            now,
                        ),
                    )
                    if cleanup_queue_rows:
                        conn.executemany(
                            """
                            INSERT INTO history_cleanup_queue(
                                cleanup_id, audit_id, relative_path, expected_size, sha256,
                                status, error_message, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            cleanup_queue_rows,
                        )
                else:
                    deleted_results = 0
                    revision_row = conn.execute(
                        "SELECT revision FROM statistics_state WHERE singleton_id=1"
                    ).fetchone()
                    statistics_revision = int(revision_row["revision"] if revision_row else 0)
                if cleanup_job_id:
                    self._update_cleanup_job(
                        cleanup_job_id,
                        conn=conn,
                        stage="files",
                        message=(
                            "数据库事务已完成，正在清理受控结果文件。"
                            if eligible
                            else "记录已被其他删除任务处理，本任务未重复删除。"
                        ),
                        processed_records=len(eligible),
                        success_count=len(eligible),
                        skipped_count=skipped_record_count,
                        audit_id=audit_id or None,
                        total_files=len(file_manifest),
                        processed_files=0,
                        deleted_record_keys_json=json.dumps(deleted_record_keys, ensure_ascii=False),
                    )
                conn.commit()
        except Exception as exc:
            if self.logger:
                self.logger.exception("history_management_db_delete_failed")
            raise ApiError("HISTORY_DELETE_FAILED", "检测记录删除失败，数据库事务已回滚。", 500) from exc
        cleanup_ids_by_path = {row[2]: row[0] for row in cleanup_queue_rows}
        cleaned = 0
        skipped = 0
        failed = 0
        released_bytes = 0
        queue_updates: list[tuple[str, str | None, str, str]] = []
        for index, item in enumerate(file_manifest, start=1):
            relative = str(item.get("relative_path") or "")
            path = (self.settings.jobs_root / relative).resolve()
            cleanup_id = cleanup_ids_by_path[relative]
            status = "cleaned"
            error = None
            try:
                path.relative_to(self.settings.jobs_root.resolve())
                if path.is_file():
                    matches, identity_error = self._cleanup_artifact_matches(
                        path,
                        expected_size=int(item.get("size_bytes") or 0),
                        expected_sha256=item.get("sha256"),
                    )
                    if not matches:
                        status = "pending"
                        error = identity_error
                        failed += 1
                    else:
                        path.unlink()
                        cleaned += 1
                        released_bytes += int(item.get("size_bytes") or 0)
                elif path.exists():
                    status = "failed"
                    error = "结果路径不是文件。"
                    failed += 1
                else:
                    status = "skipped"
                    skipped += 1
            except (OSError, ValueError) as exc:
                status = "pending"
                error = "结果文件清理失败，可稍后重试。"
                failed += 1
                if self.logger:
                    self.logger.warning("history_file_cleanup_failed relative=%s error=%s", relative, exc)
            queue_updates.append((status, error, _utc_now(), cleanup_id))
            flush_progress = (
                index == len(file_manifest)
                or index % _CLEANUP_PROGRESS_BATCH_SIZE == 0
            )
            if flush_progress:
                with self._connect() as conn:
                    conn.executemany(
                        "UPDATE history_cleanup_queue SET status=?, error_message=?, updated_at=? WHERE cleanup_id=?",
                        queue_updates,
                    )
                queue_updates.clear()
            if cleanup_job_id and flush_progress:
                self._update_cleanup_job(
                    cleanup_job_id,
                    processed_files=index,
                    skipped_count=skipped_record_count + skipped,
                    failed_count=failed,
                    released_bytes=released_bytes,
                    message=f"已处理 {index}/{len(file_manifest)} 个结果文件。",
                )
        audit_status = "completed" if failed == 0 else "partial_cleanup"
        if audit_id:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE history_management_audit SET affected_file_count=?, cleanup_file_count=?, cleanup_failed_count=?, status=?, result=? WHERE audit_id=?",
                    (len(file_manifest), cleaned, failed, audit_status, audit_status, audit_id),
                )
        if eligible:
            self._invalidate_projection()
        return {
            "audit_id": audit_id,
            "status": audit_status,
            "snapshot_name": snapshot_name,
            "deleted_records": len(eligible),
            "deleted_results": planned_result_count,
            "deleted_database_results": deleted_results,
            "protected_records": len(protected),
            "missing_records": len(missing_record_keys),
            "skipped_records": skipped_record_count,
            "cleaned_files": cleaned,
            "skipped_files": skipped,
            "cleanup_failed": failed,
            "released_bytes": released_bytes,
            "deleted_record_keys": [item["record_key"] for item in eligible],
            "statistics_revision": statistics_revision,
            "message": (
                "删除计划已执行。"
                if failed == 0
                else "删除计划已执行，部分结果文件待重试清理。"
            ),
        }

    def _run_cleanup_job(
        self,
        job_id: str,
        payload: dict[str, Any],
        all_clear: bool,
        delete_plan: dict[str, Any],
    ) -> None:
        try:
            with self._lock:
                cancelled = job_id in self._cleanup_cancel_requested
            if cancelled:
                return
            self._update_cleanup_job(
                job_id,
                status="running",
                stage="preparing",
                message="正在确认受保护记录与可删除范围。",
            )
            result = self._delete_sync(
                payload,
                all_clear=all_clear,
                delete_plan=delete_plan,
                cleanup_job_id=job_id,
            )
            status = "completed" if not int(result.get("cleanup_failed") or 0) else "partial"
            self._update_cleanup_job(
                job_id,
                status=status,
                stage="completed",
                message=result.get("message") or "检测记录清理完成。",
                processed_records=int(result.get("deleted_records") or 0)
                + int(result.get("skipped_records") or 0),
                success_count=int(result.get("deleted_records") or 0),
                skipped_count=int(result.get("skipped_records") or 0)
                + int(result.get("skipped_files") or 0),
                failed_count=int(result.get("cleanup_failed") or 0),
                processed_files=int(result.get("cleaned_files") or 0)
                + int(result.get("skipped_files") or 0)
                + int(result.get("cleanup_failed") or 0),
                released_bytes=int(result.get("released_bytes") or 0),
                audit_id=result.get("audit_id"),
                snapshot_name=result.get("snapshot_name"),
                deleted_record_keys_json=json.dumps(result.get("deleted_record_keys") or [], ensure_ascii=False),
                finished_at=_utc_now(),
            )
            with self._connect() as conn:
                conn.execute(
                    "UPDATE history_delete_plans SET status=?, finished_at=? WHERE plan_id=?",
                    ("completed" if status == "completed" else "partial", _utc_now(), delete_plan.get("plan_id")),
                )
        except Exception as exc:
            if self.logger:
                self.logger.exception("history_cleanup_job_failed job_id=%s", job_id)
            message = exc.message if isinstance(exc, ApiError) else "检测记录清理任务失败，数据库未完成的事务已回滚。"
            with self._connect() as conn:
                job_row = conn.execute(
                    "SELECT audit_id FROM history_cleanup_jobs WHERE job_id=?",
                    (job_id,),
                ).fetchone()
                audit_id = str(job_row["audit_id"] or "") if job_row else ""
                if audit_id:
                    conn.execute(
                        """
                        UPDATE history_cleanup_queue
                        SET status='pending', error_message=?, updated_at=?
                        WHERE audit_id=? AND status='processing'
                        """,
                        ("cleanup_worker_interrupted", _utc_now(), audit_id),
                    )
            self._update_cleanup_job(
                job_id,
                status="failed",
                stage="failed",
                message=message,
                error_message=type(exc).__name__,
                failed_count=1,
                finished_at=_utc_now(),
            )
            if delete_plan.get("plan_id"):
                with self._connect() as conn:
                    conn.execute(
                        "UPDATE history_delete_plans SET status='failed', finished_at=? WHERE plan_id=?",
                        (_utc_now(), delete_plan.get("plan_id")),
                    )
        finally:
            with self._lock:
                self._cleanup_threads.pop(job_id, None)
                self._cleanup_cancel_requested.discard(job_id)

    def delete(self, payload: dict[str, Any], *, all_clear: bool = False) -> dict[str, Any]:
        delete_plan = self._consume_delete_plan(payload, all_clear=all_clear)
        job_id = f"cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:10]}"
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO history_cleanup_jobs(
                    job_id,status,stage,message,total_records,created_at,updated_at
                ) VALUES (?, 'queued', 'queued', ?, ?, ?, ?)
                """,
                (
                    job_id,
                    "删除计划已确认，等待后台安全重校验。",
                    int(delete_plan.get("selected_count") or len(delete_plan.get("record_keys") or [])),
                    now,
                    now,
                ),
            )
            conn.execute(
                "UPDATE history_delete_plans SET status='executing' WHERE plan_id=? AND status='consumed'",
                (delete_plan.get("plan_id"),),
            )
        thread = threading.Thread(
            target=self._run_cleanup_job,
            args=(job_id, dict(payload), bool(all_clear), delete_plan),
            name=f"history-cleanup-{job_id[-10:]}",
            daemon=True,
        )
        with self._lock:
            self._cleanup_threads[job_id] = thread
        thread.start()
        return self.cleanup_job(job_id)

    def audit(self, limit: int = 50) -> list[dict[str, Any]]:
        safe_limit = min(100, max(1, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT audit_id, operation, scope, affected_task_count, affected_result_count,
                       protected_count, affected_file_count, cleanup_file_count, cleanup_failed_count,
                       released_bytes, snapshot_name, backup_path, target_type, operator,
                       result, status, error_message, created_at
                FROM history_management_audit ORDER BY created_at DESC LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        return [
            {
                "audit_id": _short_id(row["audit_id"]),
                "operation_label": "全部清空" if row["operation"] == "clear_all" else "删除记录",
                "scope": row["scope"],
                "affected_task_count": row["affected_task_count"],
                "affected_result_count": row["affected_result_count"],
                "protected_count": row["protected_count"],
                "affected_file_count": row["affected_file_count"],
                "cleanup_file_count": row["cleanup_file_count"],
                "cleanup_failed_count": row["cleanup_failed_count"],
                "released_bytes": row["released_bytes"],
                "snapshot_name": row["snapshot_name"] or "-",
                "backup_path": row["backup_path"] or row["snapshot_name"] or "-",
                "target_type": row["target_type"] or "detection_history",
                "operator": row["operator"] or "local_operator",
                "result": row["result"] or row["status"],
                "status": row["status"],
                "error_message": row["error_message"] or "",
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def retry_file_cleanup(self, cleanup_ids: list[str] | None = None) -> dict[str, Any]:
        wanted = [str(item) for item in (cleanup_ids or []) if str(item)][:100]
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if wanted:
                marks = ",".join("?" for _ in wanted)
                rows = conn.execute(
                    f"""SELECT * FROM history_cleanup_queue
                        WHERE cleanup_id IN ({marks}) AND status IN ('pending','failed')""",
                    wanted,
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM history_cleanup_queue
                       WHERE status IN ('pending','failed') ORDER BY updated_at LIMIT 100"""
                ).fetchall()
            claimed_ids = [str(row["cleanup_id"]) for row in rows]
            if claimed_ids:
                marks = ",".join("?" for _ in claimed_ids)
                conn.execute(
                    f"""UPDATE history_cleanup_queue
                        SET status='retrying', error_message=NULL, updated_at=?
                        WHERE cleanup_id IN ({marks}) AND status IN ('pending','failed')""",
                    [_utc_now(), *claimed_ids],
                )
            conn.commit()
        cleaned = 0
        failed = 0
        updates: list[tuple[str, str | None, str, str]] = []
        try:
            for row in rows:
                relative = str(row["relative_path"] or "")
                path = (self.settings.jobs_root / relative).resolve()
                status, error = "cleaned", None
                try:
                    path.relative_to(self.settings.jobs_root.resolve())
                    if path.is_file():
                        matches, identity_error = self._cleanup_artifact_matches(
                            path,
                            expected_size=int(row["expected_size"] or 0),
                            expected_sha256=row["sha256"],
                        )
                        if not matches:
                            status, error = "pending", identity_error
                            failed += 1
                        else:
                            path.unlink()
                            cleaned += 1
                    elif path.exists():
                        status, error = "pending", "结果路径不是文件，未执行删除。"
                        failed += 1
                except (OSError, ValueError):
                    status, error = "pending", "结果文件仍无法清理。"
                    failed += 1
                updates.append((status, error, _utc_now(), str(row["cleanup_id"])))
            if updates:
                with self._connect() as conn:
                    conn.executemany(
                        "UPDATE history_cleanup_queue SET status=?, error_message=?, updated_at=? WHERE cleanup_id=? AND status='retrying'",
                        updates,
                    )
        except Exception:
            if claimed_ids:
                with self._connect() as conn:
                    marks = ",".join("?" for _ in claimed_ids)
                    conn.execute(
                        f"""UPDATE history_cleanup_queue
                            SET status='pending', error_message=?, updated_at=?
                            WHERE cleanup_id IN ({marks}) AND status='retrying'""",
                        ["cleanup_retry_interrupted", _utc_now(), *claimed_ids],
                    )
            raise
        return {"attempted": len(rows), "cleaned": cleaned, "failed": failed, "message": "清理重试已完成。"}
