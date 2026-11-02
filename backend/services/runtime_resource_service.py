"""Unified, server-owned registry for runtime resources and profile bindings.

The existing configuration-profile and target-domain services remain the
source of truth.  This service is an additive index over those resources: it
stores stable resource IDs, dependency edges and validation/audit state in the
runtime SQLite database without moving or rewriting model and memory-bank
artifacts.  All mutating operations resolve paths from a server-side
registered reference; the UI never submits a file path.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import sqlite3
import stat
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.errors import ApiError
from backend.database import DatabaseManager
from backend.services.runtime_profile_service import deployment_scope
from backend.services.manifest_cache import FileManifestCache
from backend.services.feedback_task_status import ACTIVE_TASK_STATUSES


RESOURCE_TYPES = {
    "base_model",
    "model_weight",
    "configuration",
    "effective_config",
    "target_profile",
    "candidate_profile",
    "normal_memory_bank",
    "coreset_index",
    "defect_prototype_bank",
    "lightweight_calibrator",
    "image_threshold",
    "pixel_threshold",
    "fusion_parameters",
}
TARGET_DOMAIN_RESOURCE_TYPES = RESOURCE_TYPES - {"base_model", "model_weight"}
STATUS_LABELS = {
    "normal": "正常",
    "warning": "警告",
    "incomplete": "不完整",
    "incompatible": "不兼容",
    "hash_mismatch": "哈希异常",
    "building": "构建中",
    "managed": "后端托管",
    "archived": "已归档",
    "orphan_candidate": "来源待确认",
}
RESOURCE_ID_RE = re.compile(r"^res_[0-9a-f]{24}$")
DELETE_OPERATION_ID_RE = re.compile(r"^delete_[0-9a-f]{16}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _is_reparse_or_symlink(path: Path) -> bool:
    """Treat symlinks and Windows junction/reparse points as unsafe roots."""
    try:
        if path.is_symlink():
            return True
        attributes = int(getattr(path.stat(), "st_file_attributes", 0) or 0)
        return bool(attributes & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)))
    except OSError:
        return True


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True, default=str)


def _process_is_alive(process_id: int) -> bool:
    """Check a persisted job owner without relying on Windows signal 0.

    ``os.kill(pid, 0)`` is the usual POSIX probe, but Windows can report an
    invalid-parameter error for a healthy process.  Querying the process exit
    code keeps active resource leases effective on the deployment platform.
    """
    if process_id <= 0:
        return False
    if process_id == os.getpid():
        return True
    if os.name == "nt":
        try:
            import ctypes

            process_query_limited_information = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                process_query_limited_information, False, process_id
            )
            if not handle:
                return False
            try:
                exit_code = ctypes.c_ulong()
                if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return False
                return int(exit_code.value) == 259  # STILL_ACTIVE
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        except (AttributeError, OSError, ValueError):
            return False
    try:
        os.kill(process_id, 0)
        return True
    except (OSError, ProcessLookupError, PermissionError):
        return False


class RuntimeResourceService:
    """Index and lifecycle guard for the current, versioned runtime graph."""

    def __init__(
        self,
        settings: Any,
        config_profiles: Any,
        domain_profiles: Any,
        models: Any,
        logger: Any,
        registry: Any | None = None,
        jobs: Any | None = None,
    ):
        self.settings = settings
        self.config_profiles = config_profiles
        self.domain_profiles = domain_profiles
        self.models = models
        self.logger = logger
        self.registry = registry
        self.jobs = jobs
        self.runtime_profiles: Any | None = None
        self.test_lock: Any | None = None
        self.feedback_candidates: Any | None = None
        self.history_management: Any | None = None
        self.path = settings.runtime_root / "detection_history.sqlite3"
        self.database = DatabaseManager(self.path)
        self.manifest_cache = FileManifestCache(self.database)
        self._lock = threading.RLock()
        self._apply_lock = threading.RLock()
        self._delete_lock = threading.RLock()
        self._activity_lock = threading.RLock()
        self._active_profile_validations = 0
        self._delete_previews: dict[str, dict[str, Any]] = {}
        self._batch_delete_previews: dict[str, dict[str, Any]] = {}
        self._clear_all_plans: dict[str, dict[str, Any]] = {}
        self._last_scan_at = 0.0
        self._initialize()
        # A service restart is also a convergence point: legacy manifests can
        # otherwise keep resources falsely "in use" until an operator opens a
        # delete preview.  The same conservative heartbeat/PID checks used by
        # lifecycle operations are applied here without deleting any output.
        self._running_tasks()

    def _runtime_state(self) -> dict[str, Any]:
        if self.runtime_profiles is not None:
            return self.runtime_profiles.raw_runtime()
        return self.settings.dynamic_settings()

    def reconcile_deleted_feedback_dependencies(self) -> dict[str, int]:
        """Idempotently repair stale feedback locks left by deleted resources.

        This also revisits candidates marked by an earlier release policy: a
        deleted source plus no active transaction must converge to unlocked on
        every startup.
        """
        if self.feedback_candidates is None:
            return {"affected": 0, "released": 0, "still_protected": 0, "active_jobs": 0}
        total = {"affected": 0, "released": 0, "still_protected": 0, "active_jobs": 0}
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT COALESCE(profile_id,''),COALESCE(resource_id,'')
                FROM runtime_resource_tombstones
                WHERE resource_type IN ('target_profile','candidate_profile','normal_memory_bank')
                """
            ).fetchall()
            for row in rows:
                result = self.feedback_candidates.mark_source_memory_bank_deleted(
                    profile_id=str(row[0] or ""), resource_id=str(row[1] or ""),
                    actor="system_reconciliation", connection=conn,
                )
                for key in total:
                    total[key] += int(result.get(key) or 0)
        if total["affected"] and self.history_management is not None:
            self.history_management.resource_dependency_changed(
                "deleted_memory_bank_feedback_dependency_reconciled"
            )
        if total["affected"]:
            self.logger.info("deleted_memory_bank_feedback_dependency_reconciled %s", total)
        return total

    def _write_runtime_state(
        self,
        updates: dict[str, Any],
        *,
        runtime_status: str | None = None,
        bump_revision: bool = True,
    ) -> dict[str, Any]:
        if self.runtime_profiles is not None:
            return self.runtime_profiles.persist(
                updates,
                runtime_status=runtime_status,
                bump_revision=bump_revision,
            )
        # Compatibility for isolated service tests.  The formal application
        # always attaches RuntimeProfileService before any mutation endpoint is
        # exposed.
        return self.settings.update_dynamic_settings(updates)

    def _connect(self) -> sqlite3.Connection:
        return self.database.connect()

    def _initialize(self) -> None:
        self.settings.ensure_runtime_dirs()
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runtime_resources (
                    resource_id TEXT PRIMARY KEY,
                    resource_key TEXT NOT NULL UNIQUE,
                    resource_type TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    version TEXT NOT NULL DEFAULT '1',
                    parent_resource_id TEXT,
                    profile_id TEXT,
                    dataset TEXT,
                    category TEXT,
                    method_mode TEXT,
                    artifact_ref TEXT NOT NULL DEFAULT '',
                    file_size INTEGER,
                    sha256 TEXT,
                    frozen INTEGER NOT NULL DEFAULT 0,
                    active INTEGER NOT NULL DEFAULT 0,
                    archived INTEGER NOT NULL DEFAULT 0,
                    source_type TEXT NOT NULL DEFAULT 'managed',
                    compatibility_version TEXT NOT NULL DEFAULT '1',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    integrity_status TEXT NOT NULL DEFAULT 'warning',
                    created_at TEXT NOT NULL,
                    archived_at TEXT,
                    last_archived_at TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_runtime_resources_type ON runtime_resources(resource_type, archived);
                CREATE INDEX IF NOT EXISTS idx_runtime_resources_binding ON runtime_resources(dataset, category, profile_id);
                CREATE INDEX IF NOT EXISTS idx_runtime_resources_status_updated
                ON runtime_resources(archived, integrity_status, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_runtime_resources_active_updated
                ON runtime_resources(active DESC, frozen DESC, updated_at DESC);
                CREATE TABLE IF NOT EXISTS runtime_resource_dependencies (
                    relation_id TEXT PRIMARY KEY,
                    parent_resource_id TEXT NOT NULL,
                    child_resource_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    required INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    UNIQUE(parent_resource_id, child_resource_id, relation_type)
                );
                CREATE INDEX IF NOT EXISTS idx_runtime_dependency_parent ON runtime_resource_dependencies(parent_resource_id);
                CREATE INDEX IF NOT EXISTS idx_runtime_dependency_child ON runtime_resource_dependencies(child_resource_id);
                CREATE TABLE IF NOT EXISTS runtime_resource_audit (
                    audit_id TEXT PRIMARY KEY,
                    action TEXT NOT NULL,
                    resource_id TEXT,
                    actor TEXT NOT NULL,
                    before_json TEXT NOT NULL DEFAULT '{}',
                    after_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runtime_resource_runtime_state (
                    state_key TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runtime_resource_catalog_state (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    revision INTEGER NOT NULL DEFAULT 0,
                    last_event_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                );
                INSERT OR IGNORE INTO runtime_resource_catalog_state(singleton,revision,last_event_json,updated_at)
                VALUES(1,0,'{}','1970-01-01T00:00:00+00:00');
                CREATE TABLE IF NOT EXISTS runtime_resource_catalog_events (
                    revision INTEGER PRIMARY KEY,
                    event_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runtime_resource_tombstones (
                    tombstone_id TEXT PRIMARY KEY,
                    resource_id TEXT NOT NULL,
                    profile_id TEXT,
                    resource_type TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    version TEXT,
                    dataset TEXT,
                    category TEXT,
                    method_mode TEXT,
                    sha256 TEXT,
                    file_size INTEGER,
                    deleted_at TEXT NOT NULL,
                    delete_operation_id TEXT NOT NULL,
                    deletion_reason TEXT,
                    former_profile_references_json TEXT NOT NULL DEFAULT '[]',
                    former_task_reference_count INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_runtime_tombstone_resource ON runtime_resource_tombstones(resource_id,deleted_at DESC);
                CREATE TABLE IF NOT EXISTS runtime_resource_batch_operations (
                    operation_id TEXT PRIMARY KEY,
                    operation_type TEXT NOT NULL,
                    selected_count INTEGER NOT NULL,
                    success_count INTEGER NOT NULL,
                    skipped_count INTEGER NOT NULL,
                    failed_count INTEGER NOT NULL,
                    released_bytes INTEGER NOT NULL DEFAULT 0,
                    details_json TEXT NOT NULL DEFAULT '[]',
                    actor TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            tombstone_columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(runtime_resource_tombstones)").fetchall()
            }
            if "profile_id" not in tombstone_columns:
                conn.execute("ALTER TABLE runtime_resource_tombstones ADD COLUMN profile_id TEXT")
            resource_columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(runtime_resources)").fetchall()
            }
            # Existing installations predate explicit archive timestamps.  A
            # nullable migration preserves their creation/update history and
            # lets the next real archive operation write an unambiguous event
            # time without rewriting old lifecycle data.
            if "archived_at" not in resource_columns:
                conn.execute("ALTER TABLE runtime_resources ADD COLUMN archived_at TEXT")
            if "last_archived_at" not in resource_columns:
                conn.execute("ALTER TABLE runtime_resources ADD COLUMN last_archived_at TEXT")

    def catalog_revision(self, since: int | None = None) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT revision,last_event_json,updated_at FROM runtime_resource_catalog_state WHERE singleton=1").fetchone()
            revision = int(row["revision"] or 0) if row else 0
            event = json.loads(row["last_event_json"] or "{}") if row else {}
            changes = []
            if since is not None and int(since) < revision:
                changes = [json.loads(item["event_json"] or "{}") for item in conn.execute("SELECT event_json FROM runtime_resource_catalog_events WHERE revision>? ORDER BY revision LIMIT 100", (max(0, int(since)),)).fetchall()]
        return {"revision": revision, "changed": since is not None and int(since) != revision, "last_event": event, "changes": changes, "updated_at": row["updated_at"] if row else None}

    def _bump_revision(self, *, action: str, resource_ids: list[str] | None = None,
                       profile_ids: list[str] | None = None, resource_types: list[str] | None = None,
                       current_profile_changed: bool = False) -> dict[str, Any]:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT revision FROM runtime_resource_catalog_state WHERE singleton=1").fetchone()
            revision = int(row[0] or 0) + 1
            event = {
                "revision": revision,
                "action": str(action),
                "affected_resource_ids": list(dict.fromkeys(resource_ids or [])),
                "affected_profile_ids": list(dict.fromkeys(profile_ids or [])),
                "invalidated_resource_types": list(dict.fromkeys(resource_types or [])),
                "current_profile_changed": bool(current_profile_changed),
                "created_at": now,
            }
            encoded = _json(event)
            conn.execute("UPDATE runtime_resource_catalog_state SET revision=?,last_event_json=?,updated_at=? WHERE singleton=1", (revision, encoded, now))
            conn.execute("INSERT INTO runtime_resource_catalog_events(revision,event_json,created_at) VALUES(?,?,?)", (revision, encoded, now))
            conn.execute("DELETE FROM runtime_resource_catalog_events WHERE revision<?", (max(0, revision - 500),))
        return event

    def _relative_ref(self, path: Path | None) -> str:
        if path is None:
            return ""
        try:
            return path.resolve().relative_to(self.settings.project_root).as_posix()
        except (ValueError, OSError):
            return ""

    def _safe_path(self, artifact_ref: str) -> Path:
        ref = str(artifact_ref or "").strip().replace("\\", "/")
        if not ref or ref.startswith("/") or ".." in Path(ref).parts:
            raise ApiError("RESOURCE_PATH_INVALID", "资源引用无效。", 422)
        raw_path = self.settings.project_root / ref
        if raw_path.is_symlink() or any(parent.is_symlink() for parent in raw_path.parents if parent != self.settings.project_root):
            raise ApiError("RESOURCE_PATH_FORBIDDEN", "不允许通过符号链接访问资源。", 403)
        path = raw_path.resolve()
        roots = (
            self.settings.runtime_root.resolve(),
            self.settings.algorithm_outputs_root.resolve(),
            self.settings.memory_banks_root.resolve(),
            self.settings.config_profiles_root.resolve(),
            (self.settings.project_root / "configs").resolve(),
            (self.settings.project_root / "models").resolve(),
            (self.settings.project_root / "weights").resolve(),
        )
        if not any(_within(path, root) for root in roots):
            raise ApiError("RESOURCE_PATH_FORBIDDEN", "资源引用不在受控项目资源目录内。", 403)
        return path

    @staticmethod
    def _resource_id(resource_type: str, resource_key: str) -> str:
        return "res_" + hashlib.sha256(f"{resource_type}:{resource_key}".encode("utf-8")).hexdigest()[:24]

    def _row(self, row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        item = dict(row)
        try:
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            item["metadata"] = {}
        for key in ("frozen", "active", "archived"):
            item[key] = bool(item.get(key))
        item["integrity_status_label"] = STATUS_LABELS.get(str(item.get("integrity_status")), "未知")
        item["current"] = bool(item.get("active"))
        return item

    def _get(self, resource_id: str) -> dict[str, Any]:
        if not RESOURCE_ID_RE.fullmatch(str(resource_id or "")):
            raise ApiError("RESOURCE_NOT_FOUND", "资源标识无效。", 404)
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runtime_resources WHERE resource_id=?", (resource_id,)).fetchone()
        if row is None:
            raise ApiError("RESOURCE_NOT_FOUND", "资源不存在或尚未建立索引。", 404)
        return self._row(row)

    def _record_audit(self, action: str, resource_id: str | None, actor: str, before: Any = None, after: Any = None) -> None:
        # Several dependency checks can write the same action for the same
        # resource within one second.  A timestamp-derived id therefore
        # collided under profile validation and could abort an otherwise safe
        # apply/restore transaction.  Audit rows need uniqueness, not content
        # addressing; retain the timestamp as data and use a random id.
        audit_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO runtime_resource_audit(audit_id,action,resource_id,actor,before_json,after_json,created_at) VALUES(?,?,?,?,?,?,?)",
                (audit_id, action, resource_id, str(actor or "operator")[:80], _json(before), _json(after), _utc_now()),
            )

    def _upsert(
        self,
        spec: dict[str, Any],
        seen: set[str],
        id_by_key: dict[str, str],
        *,
        connection: sqlite3.Connection | None = None,
    ) -> str:
        resource_type = str(spec.get("resource_type") or "")
        if resource_type not in RESOURCE_TYPES:
            raise ValueError(f"unsupported resource type: {resource_type}")
        resource_key = str(spec.get("resource_key") or f"{resource_type}:{spec.get('artifact_ref') or spec.get('profile_id') or spec.get('display_name')}")
        resource_id = self._resource_id(resource_type, resource_key)
        # Child artifact paths change when a profile is archived/restored
        # (the reference moves between the live root and ``_archive``). Keep
        # the existing resource id for the same owner/type so a rescan does
        # not create duplicate child rows or orphan dependency edges.
        owner_id = str(spec.get("profile_id") or "")
        if owner_id and resource_type not in {"target_profile"}:
            if connection is not None:
                conn = connection
                existing = conn.execute("SELECT resource_id FROM runtime_resources WHERE resource_key=?", (resource_key,)).fetchone()
                if existing is None:
                    desired_archived = int(bool(spec.get("archived", False)))
                    existing = conn.execute(
                        "SELECT resource_id FROM runtime_resources WHERE profile_id=? AND resource_type=? ORDER BY CASE WHEN archived=? THEN 0 ELSE 1 END, updated_at DESC LIMIT 1",
                        (owner_id, resource_type, desired_archived),
                    ).fetchone()
            else:
                with self._connect() as conn:
                    existing = conn.execute("SELECT resource_id FROM runtime_resources WHERE resource_key=?", (resource_key,)).fetchone()
                    if existing is None:
                        desired_archived = int(bool(spec.get("archived", False)))
                        existing = conn.execute(
                            "SELECT resource_id FROM runtime_resources WHERE profile_id=? AND resource_type=? ORDER BY CASE WHEN archived=? THEN 0 ELSE 1 END, updated_at DESC LIMIT 1",
                            (owner_id, resource_type, desired_archived),
                        ).fetchone()
            if existing is not None:
                resource_id = str(existing["resource_id"])
        seen.add(resource_id)
        id_by_key[resource_key] = resource_id
        now = _utc_now()
        artifact_ref = str(spec.get("artifact_ref") or "")
        path = None
        if artifact_ref:
            try:
                path = self._safe_path(artifact_ref)
            except ApiError:
                path = None
        size = spec.get("file_size")
        digest = str(spec.get("sha256") or "")
        metadata = dict(spec.get("metadata") or {})
        if path is not None and path.exists() and not path.is_dir():
            try:
                size = int(path.stat().st_size)
                if not digest and size <= 256 * 1024 * 1024:
                    digest = self.manifest_cache.digest(path, connection=connection)
                elif size > 256 * 1024 * 1024:
                    metadata.setdefault("hash_policy", "large_file_requires_explicit_validation")
            except OSError:
                pass
        status = str(spec.get("integrity_status") or "warning")
        if metadata.get("virtual") or (not artifact_ref and status == "managed"):
            status = str(spec.get("integrity_status") or "normal")
        elif path is None or not path.exists():
            status = "incomplete"
        elif digest and metadata.get("expected_sha256") and digest.lower() != str(metadata["expected_sha256"]).lower():
            status = "hash_mismatch"
        elif resource_type == "candidate_profile" and status in {"building", "warning", "incomplete", "incompatible"}:
            # A staging candidate may be present on disk while it is still
            # waiting for review/application; keep that lifecycle state.
            pass
        elif bool(spec.get("archived")) and path.is_dir():
            # Keep the lifecycle state visible even though the archived
            # directory is still readable and hashable.
            status = "archived"
        elif digest or path.is_dir():
            status = "normal"
        metadata.setdefault("artifact_ref", artifact_ref)
        def persist(conn: sqlite3.Connection) -> None:
            old = conn.execute(
                "SELECT archived, created_at, archived_at, last_archived_at "
                "FROM runtime_resources WHERE resource_id=?",
                (resource_id,),
            ).fetchone()
            # The domain/config registries and their controlled live/archive
            # directories are authoritative.  Preserving an old catalog flag
            # made a legacy-archived profile appear selectable forever even
            # though its files had already moved to ``_archive``.
            archived = int(bool(spec.get("archived", False)))
            spec_created_at = str(spec.get("created_at") or metadata.get("created_at") or "").strip()
            created_at = str(old["created_at"]) if old else spec_created_at or now
            metadata_archived_at = str(spec.get("archived_at") or metadata.get("archived_at") or "").strip()
            metadata_last_archived_at = str(spec.get("last_archived_at") or metadata.get("last_archived_at") or "").strip()
            if archived:
                archived_at = metadata_archived_at or (str(old["archived_at"] or "") if old else "") or now
                last_archived_at = metadata_last_archived_at or (str(old["last_archived_at"] or "") if old else "") or None
            else:
                archived_at = None
                last_archived_at = metadata_last_archived_at or (str(old["last_archived_at"] or "") if old else "") or None
            conn.execute(
                """
                INSERT INTO runtime_resources(resource_id,resource_key,resource_type,display_name,version,parent_resource_id,profile_id,dataset,category,method_mode,artifact_ref,file_size,sha256,frozen,active,archived,source_type,compatibility_version,metadata_json,integrity_status,created_at,archived_at,last_archived_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(resource_id) DO UPDATE SET
                  resource_key=excluded.resource_key, resource_type=excluded.resource_type, display_name=excluded.display_name,
                  version=excluded.version, parent_resource_id=excluded.parent_resource_id, profile_id=excluded.profile_id,
                  dataset=excluded.dataset, category=excluded.category, method_mode=excluded.method_mode,
                  artifact_ref=excluded.artifact_ref, file_size=excluded.file_size, sha256=excluded.sha256,
                  frozen=excluded.frozen, active=excluded.active, archived=excluded.archived, source_type=excluded.source_type,
                  compatibility_version=excluded.compatibility_version, metadata_json=excluded.metadata_json,
                  integrity_status=excluded.integrity_status, archived_at=excluded.archived_at,
                  last_archived_at=excluded.last_archived_at, updated_at=excluded.updated_at
                """,
                (
                    resource_id,
                    resource_key,
                    resource_type,
                    str(spec.get("display_name") or resource_type),
                    str(spec.get("version") or "1"),
                    spec.get("parent_resource_id"),
                    spec.get("profile_id"),
                    spec.get("dataset"),
                    spec.get("category"),
                    spec.get("method_mode"),
                    artifact_ref,
                    size,
                    digest or None,
                    int(bool(spec.get("frozen"))),
                    int(bool(spec.get("active"))),
                    archived,
                    str(spec.get("source_type") or "managed"),
                    str(spec.get("compatibility_version") or "1"),
                    _json(metadata),
                    status,
                    created_at,
                    archived_at,
                    last_archived_at,
                    now,
                ),
            )
        if connection is not None:
            persist(connection)
        else:
            with self._connect() as conn:
                persist(conn)
        return resource_id

    def _dependency(
        self,
        parent: str,
        child: str,
        relation_type: str,
        required: bool = True,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        if not parent or not child or parent == child:
            return
        relation_id = hashlib.sha256(f"{parent}:{child}:{relation_type}".encode("utf-8")).hexdigest()[:32]
        def persist(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT OR REPLACE INTO runtime_resource_dependencies(relation_id,parent_resource_id,child_resource_id,relation_type,required,created_at) VALUES(?,?,?,?,?,?)",
                (relation_id, parent, child, relation_type, int(required), _utc_now()),
            )
        if connection is not None:
            persist(connection)
        else:
            with self._connect() as conn:
                persist(conn)

    def _model_specs(self, id_by_key: dict[str, str]) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        try:
            model_rows = self.models.list_models()
        except Exception as exc:  # pragma: no cover - depends on optional torch/model files
            self.logger.warning("runtime_resource_model_scan_failed: %s", exc)
            model_rows = []
        # A model registry may refuse one entry when an optional dataset
        # artifact is unavailable.  Keep the stable model definitions indexed
        # so the maintenance UI can report the missing dependency instead of
        # silently dropping the entire model family.
        known_ids = {str(item.get("model_id") or "") for item in model_rows}
        for model_id, definition in self.settings.models.items():
            if str(model_id) in known_ids:
                continue
            try:
                profile_path = self.settings.profile_path(str(model_id))
            except Exception:
                profile_path = None
            try:
                profile = self.settings.algorithm_profile(str(model_id))
            except Exception:
                profile = {}
            model_rows.append(
                {
                    "model_id": str(model_id),
                    "display_name": definition.get("display_name") or str(model_id),
                    "model_type": definition.get("model_type") or "patchcore",
                    "backbone": profile.get("backbone"),
                    "profile_path": str(profile_path) if profile_path else "",
                    "dataset": profile.get("dataset"),
                    "category": profile.get("category"),
                }
            )
        for model in model_rows:
            model_id = str(model.get("model_id") or "")
            if not model_id:
                continue
            profile_value = str(model.get("profile_path") or "").strip()
            profile_path = Path(profile_value) if profile_value else None
            profile_ref = self._relative_ref(profile_path if profile_path and profile_path.exists() else None)
            profile_invalid = bool(profile_value) and not profile_ref
            key = f"base_model:{model_id}"
            specs.append(
                {
                    "resource_type": "base_model",
                    "resource_key": key,
                    "display_name": str(model.get("display_name") or model_id),
                    "version": str(model.get("backbone") or "1"),
                    "artifact_ref": profile_ref,
                    "profile_id": model_id,
                    "dataset": model.get("dataset"),
                    "category": model.get("category"),
                    "source_type": "backend_managed",
                    "frozen": True,
                    "metadata": {"model_id": model_id, "model_type": model.get("model_type") or "patchcore", "backbone": model.get("backbone"), "managed_weight": True, "path_policy_violation": profile_invalid},
                    "integrity_status": "incompatible" if profile_invalid else ("managed" if not profile_ref else "normal"),
                }
            )
            definition = self.settings.models.get(model_id, {})
            weight_value = definition.get("weights_path") or definition.get("weight_path") or definition.get("weights")
            weight_ref = ""
            weight_invalid = False
            if weight_value:
                path = self.settings.resolve_path(str(weight_value))
                weight_ref = self._relative_ref(path if path.exists() else None)
                weight_invalid = not bool(weight_ref)
            weight_key = f"model_weight:{model_id}:{weight_ref or 'backend-managed'}"
            specs.append(
                {
                    "resource_type": "model_weight",
                    "resource_key": weight_key,
                    "display_name": f"{model.get('display_name') or model_id} · 骨干权重",
                    "version": str(model.get("backbone") or "managed"),
                    "artifact_ref": weight_ref,
                    "profile_id": model_id,
                    "dataset": model.get("dataset"),
                    "category": model.get("category"),
                    "source_type": "backend_managed",
                    "frozen": True,
                    "metadata": {"model_id": model_id, "backbone": model.get("backbone"), "managed_weight": not bool(weight_ref), "security_policy": "禁止联网下载与任意可执行上传", "path_policy_violation": weight_invalid},
                    "integrity_status": "incompatible" if weight_invalid else ("managed" if not weight_ref else "normal"),
                }
            )
        return specs

    def _candidate_specs(self) -> list[dict[str, Any]]:
        """Index feedback-optimization candidates without making them usable.

        Candidates live in a hidden staging directory until the feedback
        optimizer explicitly promotes one.  They are visible for audit and
        dependency inspection, but ``apply_profile`` only accepts a frozen
        ``target_profile`` resource, so a candidate can never become active by
        selecting it in the runtime page.
        """
        root = self.settings.memory_banks_root / ".feedback_candidates"
        candidate_paths: list[Path] = []
        if root.is_dir():
            candidate_paths.extend(path for path in sorted(root.iterdir()) if path.is_dir() and not path.is_symlink())
        # Target-domain builds publish candidates directly below the controlled
        # memory-bank root.  They were previously invisible here because this
        # scanner only indexed feedback candidates from the hidden staging
        # folder.  A build could therefore finish its files but fail the final
        # catalog-registration check.  Index both candidate sources, while a
        # frozen target continues to be indexed exclusively by
        # ``list_all_profiles`` below.
        if self.settings.memory_banks_root.is_dir():
            candidate_paths.extend(
                path
                for path in sorted(self.settings.memory_banks_root.iterdir())
                if path.is_dir()
                and not path.is_symlink()
                and not path.name.startswith(".")
                and (path / "candidate.lock").is_file()
                and not (path / "frozen.lock").is_file()
            )
        if not candidate_paths:
            return []
        specs: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for path in candidate_paths:
            path_key = str(path.resolve()).lower()
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)
            metadata_path = path / "metadata.json"
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                metadata = {}
            candidate_id = str(metadata.get("memory_bank_id") or path.name)
            if not candidate_id:
                continue
            ref = self._relative_ref(path)
            method_mode = str(metadata.get("method_mode_resolved") or metadata.get("method_mode_requested") or "normal_only")
            feedback_candidate = path.parent.name == ".feedback_candidates"
            validation = metadata.get("feedback_validation") if isinstance(metadata.get("feedback_validation"), dict) else {}
            if not validation and isinstance(metadata.get("candidate_validation"), dict):
                validation = metadata.get("candidate_validation") or {}
            validated = validation.get("status") == "passed" or bool(validation.get("validated"))
            status = "warning" if validated else "building"
            parent_key = f"candidate_profile:{candidate_id}"
            spec: dict[str, Any] = {
                "resource_type": "candidate_profile",
                "resource_key": parent_key,
                "display_name": str(metadata.get("memory_bank_name") or f"反馈优化候选 · {candidate_id[-8:]}"),
                "version": str(metadata.get("profile_version") or metadata.get("config_profile_version") or "1"),
                "profile_id": candidate_id,
                "dataset": metadata.get("dataset"),
                "category": metadata.get("category"),
                "method_mode": method_mode,
                "artifact_ref": ref,
                "source_type": "feedback_candidate" if feedback_candidate else "target_domain_candidate",
                "frozen": bool(metadata.get("frozen")),
                "active": False,
                "metadata": {"feedback_optimization_job_id": metadata.get("feedback_optimization_job_id"), "parent_profile_id": metadata.get("feedback_parent_profile_id") or metadata.get("parent_profile_id"), "config_profile_id": metadata.get("config_profile_id"), "config_profile_version": metadata.get("config_profile_version"), "config_hash": metadata.get("config_hash"), "validation_status": validation.get("status") or ("passed" if validated else "pending"), "candidate_only": True},
                "integrity_status": status,
            }
            children: list[tuple[str, str, str, bool]] = []
            file_specs = [
                ("normal_memory_bank", "正常记忆库", "memory_bank.pt", True),
                ("coreset_index", "coreset 索引", "memory_bank.pt", False),
                ("image_threshold", "图像级阈值", "threshold.json", True),
                ("effective_config", "目标域生效配置", "effective_config.yaml", True),
                ("defect_prototype_bank", "缺陷原型库", "defect_prototypes.pt", method_mode == "normal_defect_dual_memory"),
                ("lightweight_calibrator", "轻量校准器", "lightweight_calibrator.json", method_mode == "normal_memory_defect_guided_lite"),
                ("fusion_parameters", "融合与门控参数", "metadata.json", False),
            ]
            for child_type, label, filename, required in file_specs:
                child_path = path / filename
                child_ref = self._relative_ref(child_path if child_path.is_file() else None)
                child_key = f"{child_type}:{candidate_id}:{child_ref or filename}"
                specs.append({"resource_type": child_type, "resource_key": child_key, "display_name": f"{spec['display_name']} · {label}", "version": spec["version"], "profile_id": candidate_id, "dataset": spec["dataset"], "category": spec["category"], "method_mode": method_mode, "artifact_ref": child_ref, "source_type": spec["source_type"], "frozen": False, "active": False, "metadata": {"required": required, "candidate_only": True, "virtual": not bool(child_ref)}, "integrity_status": "normal" if child_ref else ("managed" if not required else "incomplete")})
                children.append((child_type, child_key, label, required))
            spec["_children"] = children
            specs.insert(0, spec)
        return specs

    def startup_catalog_ready(self) -> bool:
        """Return whether the active formal graph can reuse the persisted catalog.

        Startup only needs the currently bound Profile and its required files.
        The old unconditional ``scan(force=True)`` walked every candidate,
        archived resource and configuration before Runtime warm-up.  A
        persisted catalog plus mtime/size signatures is sufficient for this
        narrow safety check; management actions still perform a full scan.
        """

        runtime = self._runtime_state()
        if bool(runtime.get("runtime_deactivated")) or not str(runtime.get("memory_bank_id") or ""):
            return True
        memory_bank_id = str(runtime.get("memory_bank_id") or "")
        try:
            with self._connect() as conn:
                table = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='runtime_resources'"
                ).fetchone()
                if table is None:
                    return False
                profile = conn.execute(
                    "SELECT resource_id,artifact_ref,integrity_status FROM runtime_resources "
                    "WHERE resource_type='target_profile' AND profile_id=? AND archived=0 LIMIT 1",
                    (memory_bank_id,),
                ).fetchone()
                if profile is None or str(profile["integrity_status"] or "") in {"incomplete", "incompatible", "hash_mismatch"}:
                    return False
                required = conn.execute(
                    "SELECT artifact_ref,integrity_status FROM runtime_resources "
                    "WHERE profile_id=? AND resource_type IN ('normal_memory_bank','effective_config','image_threshold') "
                    "AND archived=0",
                    (memory_bank_id,),
                ).fetchall()
        except (OSError, sqlite3.Error):
            return False
        profile_ref = str(profile["artifact_ref"] or "")
        if not profile_ref:
            return False
        try:
            profile_root = self._safe_path(profile_ref)
        except ApiError:
            return False
        if not profile_root.is_dir():
            return False
        for row in required:
            status = str(row["integrity_status"] or "")
            ref = str(row["artifact_ref"] or "")
            if status in {"incomplete", "incompatible", "hash_mismatch"} or not ref:
                return False
            try:
                path = self._safe_path(ref)
            except ApiError:
                return False
            if not path.is_file() or not self.manifest_cache.signature_is_current(path):
                return False
        return True

    def scan(self, *, force: bool = False, include_resources: bool = True) -> dict[str, Any]:
        """Rebuild the index from existing authoritative metadata."""
        with self._lock:
            if not force and self._last_scan_at and time.monotonic() - self._last_scan_at < 5.0:
                resource_count = self.count_resources(include_archived=True)
                return {
                    "resource_count": resource_count,
                    "dependency_count": self._dependency_count(),
                    "resources": self.list_resources() if include_resources else [],
                    "summary": self.maintenance_summary(scan=False),
                    "skipped": True,
                }
            self.manifest_cache.reset_metrics()
            seen: set[str] = set()
            id_by_key: dict[str, str] = {}
            specs = self._model_specs(id_by_key)
            specs.extend(self._candidate_specs())
            try:
                bank_rows = self.domain_profiles.list_all_profiles()
                active_ids = {str(item.get("memory_bank_id") or "") for item in bank_rows}
                for archived in self.domain_profiles.list_archived_profiles():
                    archived_id = str(archived.get("memory_bank_id") or "")
                    if archived_id and archived_id not in active_ids:
                        archived = dict(archived)
                        archived["archived"] = True
                        bank_rows.append(archived)
            except Exception as exc:
                self.logger.warning("runtime_resource_memory_scan_failed: %s", exc)
                bank_rows = []
            bank_name_by_id = {
                str(item.get("memory_bank_id") or ""): str(item.get("friendly_name") or "目标域记忆库")
                for item in bank_rows
                if str(item.get("memory_bank_id") or "")
            }
            try:
                config_rows = self.config_profiles.list_profiles(include_archived=True)
            except Exception as exc:
                self.logger.warning("runtime_resource_config_scan_failed: %s", exc)
                config_rows = []
            for item in config_rows:
                pid = str(item.get("config_profile_id") or "")
                version = int(item.get("version") or 1)
                owner_id = str(item.get("memory_bank_id") or item.get("target_domain_version") or "")
                short_config_name = str(item.get("friendly_name") or "配置方案")
                global_config_name = (
                    f"{bank_name_by_id[owner_id]} · {short_config_name}"
                    if owner_id in bank_name_by_id
                    else short_config_name
                )
                archived_config = bool(item.get("archived"))
                if archived_config:
                    try:
                        config_root = self.config_profiles.archived_version_directory(pid, version)
                    except ApiError:
                        config_root = None
                else:
                    config_root = self.settings.config_profiles_root / pid / f"v{version:04d}"
                ref = self._relative_ref(config_root if config_root is not None and config_root.exists() else None)
                specs.append(
                    {
                        "resource_type": "configuration",
                        "resource_key": f"configuration:{pid}:v{version}",
                        "display_name": global_config_name,
                        "version": f"v{version}",
                        "artifact_ref": ref,
                        "profile_id": f"{pid}:v{version}",
                        "created_at": item.get("created_at"),
                        "dataset": item.get("dataset"),
                        "category": item.get("category"),
                        "method_mode": (item.get("config") or {}).get("method_mode") if isinstance(item.get("config"), dict) else None,
                        "source_type": str(item.get("source_kind") or "configuration"),
                        "frozen": bool(item.get("frozen")),
                        "archived": archived_config,
                        "archived_at": item.get("archived_at"),
                        "last_archived_at": item.get("last_archived_at"),
                        "integrity_status": "archived" if archived_config else "normal",
                        "metadata": {
                            "config_profile_id": pid,
                            "config_profile_version": version,
                            "config_snapshot_id": item.get("config_snapshot_id") or pid,
                            "config_snapshot_hash": item.get("config_snapshot_hash") or item.get("config_hash"),
                            "target_domain_version": item.get("target_domain_version") or item.get("memory_bank_id"),
                            "build_job_id": item.get("build_job_id"),
                            "validated": bool(item.get("validated")),
                            "applied": bool(item.get("applied")),
                            "ordinary_visible": bool(item.get("ordinary_visible", True)),
                        },
                    }
                )
                specs.append(
                    {
                        "resource_type": "effective_config",
                        "resource_key": f"effective_config:{pid}:v{version}",
                        "display_name": f"{global_config_name} · 生效配置",
                        "version": f"v{version}",
                        "artifact_ref": f"{ref}/profile.yaml" if ref else "",
                        "profile_id": f"{pid}:v{version}",
                        "dataset": item.get("dataset"),
                        "category": item.get("category"),
                        "source_type": "configuration",
                        "frozen": bool(item.get("frozen")),
                        "archived": archived_config,
                        "integrity_status": "archived" if archived_config else "normal",
                        "metadata": {
                            "config_profile_id": pid,
                            "config_profile_version": version,
                            "target_domain_version": item.get("target_domain_version") or item.get("memory_bank_id"),
                            "ordinary_visible": False,
                            "technical_child": True,
                            "managed_with_profile": True,
                        },
                    }
                )
            dynamic = self._runtime_state()
            current_bank_id = str(dynamic.get("memory_bank_id") or "")
            config_count_by_bank: dict[tuple[str, bool], int] = {}
            for config_row in config_rows:
                owner_id = str(config_row.get("memory_bank_id") or config_row.get("target_domain_version") or "")
                if owner_id:
                    state_key = (owner_id, bool(config_row.get("archived")))
                    config_count_by_bank[state_key] = config_count_by_bank.get(state_key, 0) + 1
            for bank in bank_rows:
                bank_id = str(bank.get("memory_bank_id") or "")
                if not bank_id:
                    continue
                builtin = str(bank.get("profile_kind") or "") == "builtin_public_baseline" or bank_id.startswith("builtin_")
                archived = bool(bank.get("archived"))
                if builtin:
                    root_path = self.settings.algorithm_outputs_root / "checkpoints" / str(bank.get("dataset") or "").lower() / str(bank.get("category") or "")
                elif archived:
                    try:
                        root_path = self.domain_profiles.archived_profile_directory(bank_id)
                    except ApiError:
                        root_path = None
                else:
                    root_path = self.settings.memory_banks_root / bank_id
                root_ref = self._relative_ref(root_path if root_path is not None and root_path.exists() else None)
                profile_key = f"target_profile:{bank_id}"
                profile_id = self._resource_id("target_profile", profile_key)
                method_mode = str(bank.get("method_mode_resolved") or bank.get("method_mode_requested") or "normal_only")
                scope = deployment_scope(bank)
                profile_spec = {
                    "resource_type": "target_profile",
                    "resource_key": profile_key,
                    "display_name": str(bank.get("friendly_name") or f"目标域方案 · {bank.get('dataset')} / {bank.get('category')}"),
                    "version": str(bank.get("profile_version") or bank.get("config_profile_version") or "1"),
                    "artifact_ref": root_ref,
                    "profile_id": bank_id,
                    "created_at": bank.get("created_at"),
                    "dataset": bank.get("dataset"),
                    "category": bank.get("category"),
                    "method_mode": method_mode,
                    "source_type": str(bank.get("profile_kind") or "target_domain"),
                    "frozen": bool(bank.get("frozen")),
                    "active": bank_id == current_bank_id and not archived,
                    "archived": archived,
                    "archived_at": bank.get("archived_at"),
                    "last_archived_at": bank.get("last_archived_at"),
                    "metadata": {
                        "memory_bank_id": bank_id,
                        "profile_kind": bank.get("profile_kind"),
                        "is_default": bool(bank.get("is_default")) and scope == "formal",
                        "deployment_scope": scope,
                        "config_profile_id": bank.get("config_profile_id"),
                        "config_profile_version": bank.get("config_profile_version"),
                        "config_snapshot_id": bank.get("config_snapshot_id") or bank.get("config_profile_id"),
                        "config_hash": bank.get("config_hash"),
                        "config_snapshot_sha256": bank.get("config_snapshot_sha256"),
                        "build_job_id": bank.get("build_job_id"),
                        "threshold": bank.get("threshold"),
                        "pixel_threshold": bank.get("pixel_threshold"),
                        "feature_dimension": bank.get("feature_dimension"),
                        "backbone": bank.get("backbone"),
                        "model_id": bank.get("model_id"),
                        # Preserve the build-time numeric execution contract
                        # in the runtime registry.  These values describe how
                        # the immutable artifact was produced; they do not
                        # change when the resource is merely scanned again.
                        "precision": bank.get("precision") or bank.get("requested_precision"),
                        "requested_precision": bank.get("requested_precision") or bank.get("precision"),
                        "actual_precision": bank.get("actual_precision"),
                        "amp_enabled": bank.get("amp_enabled"),
                        "feature_dtype": bank.get("feature_dtype"),
                        "backbone_output_dtype": bank.get("backbone_output_dtype"),
                        "memory_bank_dtype": bank.get("memory_bank_dtype"),
                        "index_dtype": bank.get("index_dtype"),
                        "precision_fallback_reason": bank.get("precision_fallback_reason") or bank.get("fallback_reason"),
                        "normal_sample_count": bank.get("normal_sample_count"),
                        "defect_calibration_sample_count": bank.get("defect_calibration_sample_count"),
                        "defect_prototype_count": bank.get("defect_prototype_count"),
                        "lightweight_calibration": bank.get("lightweight_calibration"),
                        "algorithm_mode": bank.get("algorithm_mode") or method_mode,
                        "normal_memory_enabled": bool(bank.get("normal_memory_enabled", True)),
                        "lightweight_calibration_enabled": bool(
                            bank.get("lightweight_calibration_enabled", bank.get("lightweight_calibration", False))
                        ),
                        "defect_guidance_enabled": bool(
                            bank.get("defect_guidance_enabled", bank.get("lightweight_calibration", False))
                        ),
                        "resource_size_bytes": bank.get("resource_size_bytes"),
                        "resource_integrity_status": bank.get("resource_integrity_status"),
                        "frozen": bool(bank.get("frozen")),
                        "archive_name": bank.get("archive_name"),
                        "config_count": int(config_count_by_bank.get((bank_id, archived), 0)),
                    },
                    "integrity_status": "normal" if bank.get("frozen") else "incomplete",
                }
                if archived:
                    profile_spec["integrity_status"] = "archived" if root_path is not None else "incomplete"
                specs.append(profile_spec)
                children: list[tuple[str, str, str, bool]] = []
                if builtin:
                    memory_path = root_path / "memory_bank.pt" if root_path is not None else None
                    threshold_path = self.settings.active_calibration_path(str(bank.get("dataset") or ""), str(bank.get("category") or ""))
                    config_path = self.settings.profile_path(str(bank.get("model_id") or "")) if bank.get("model_id") else None
                else:
                    memory_path = root_path / "memory_bank.pt" if root_path is not None else None
                    threshold_path = root_path / "threshold.json" if root_path is not None else None
                    config_path = root_path / "effective_config.yaml" if root_path is not None else None
                child_specs = [
                    ("normal_memory_bank", "正常记忆库", memory_path, True),
                    ("coreset_index", "coreset 索引", memory_path, False),
                    ("image_threshold", "图像级阈值", threshold_path, True),
                    ("effective_config", "目标域生效配置", config_path, True),
                ]
                if not builtin and root_path is not None and (root_path / "defect_prototypes.pt").is_file():
                    child_specs.append(("defect_prototype_bank", "缺陷原型库", root_path / "defect_prototypes.pt", method_mode == "normal_defect_dual_memory"))
                if not builtin and root_path is not None and (root_path / "lightweight_calibrator.json").is_file():
                    child_specs.append(("lightweight_calibrator", "轻量校准器", root_path / "lightweight_calibrator.json", method_mode == "normal_memory_defect_guided_lite"))
                if bank.get("pixel_threshold") is not None:
                    child_specs.append(("pixel_threshold", "像素级阈值", threshold_path, False))
                if method_mode in {"normal_defect_dual_memory", "normal_memory_defect_guided_lite"}:
                    child_specs.append(("fusion_parameters", "融合与门控参数", root_path / "metadata.json", False))
                for child_type, label, child_path, required in child_specs:
                    child_ref = self._relative_ref(child_path if child_path and child_path.exists() else None)
                    child_name = child_path.name if child_path is not None else label
                    # Use the logical filename rather than its live/archive
                    # path so restore cycles keep one stable child identity.
                    child_key = f"{child_type}:{bank_id}:{child_name}"
                    child_status = "archived" if archived and child_ref else ("normal" if child_ref else ("managed" if child_type in {"coreset_index", "fusion_parameters"} else "incomplete"))
                    specs.append({"resource_type": child_type, "resource_key": child_key, "display_name": f"{bank.get('friendly_name') or '目标域方案'} · {label}", "version": str(bank.get("config_profile_version") or "1"), "artifact_ref": child_ref, "profile_id": bank_id, "dataset": bank.get("dataset"), "category": bank.get("category"), "method_mode": method_mode, "source_type": str(bank.get("profile_kind") or "target_domain"), "frozen": bool(bank.get("frozen")), "archived": archived, "metadata": {"required": required, "memory_bank_id": bank_id, "virtual": not bool(child_ref), "expected_sha256": ((bank.get("file_sha256") or {}).get(child_name) if isinstance(bank.get("file_sha256"), dict) else None), "ordinary_visible": False, "technical_child": True, "managed_with_profile": True}, "integrity_status": child_status})
                    children.append((child_type, child_key, label, required))
                # Record dependencies after all specs receive stable IDs.
                profile_spec["_children"] = children
            # Backfill the target-version relationship for legacy snapshots
            # from the authoritative profile binding.  Earlier builds stored
            # the configuration ID on the target profile but omitted the
            # reverse link in configuration metadata, which made one owned
            # snapshot look like an unrelated configuration row in the UI.
            target_config_owners: dict[str, dict[str, Any]] = {}
            for owner in specs:
                if owner.get("resource_type") not in {"target_profile", "candidate_profile"}:
                    continue
                owner_metadata = owner.get("metadata") or {}
                config_id = str(owner_metadata.get("config_profile_id") or "")
                config_version = int(owner_metadata.get("config_profile_version") or 0)
                if not config_id or not config_version:
                    continue
                target_config_owners[f"{config_id}:v{config_version}"] = {
                    "target_domain_version": str(owner.get("profile_id") or ""),
                    "memory_bank_id": str(owner.get("profile_id") or ""),
                    "build_job_id": str(owner_metadata.get("build_job_id") or ""),
                }
            for spec in specs:
                if spec.get("resource_type") not in {"configuration", "effective_config"}:
                    continue
                owner_binding = target_config_owners.get(str(spec.get("profile_id") or ""))
                if not owner_binding:
                    continue
                spec_metadata = spec.setdefault("metadata", {})
                for key, value in owner_binding.items():
                    if value and not spec_metadata.get(key):
                        spec_metadata[key] = value
                if spec.get("resource_type") == "effective_config":
                    spec_metadata.update(ordinary_visible=False, technical_child=True, managed_with_profile=True)

            # Upsert after the full discovery list has been assembled.
            # Reuse one bounded transaction for the scan's catalog writes.
            # Opening a fresh SQLite connection for every resource made a
            # warm manifest-cache scan slower than hashing itself.
            scan_conn = self._connect()
            try:
                for spec in specs:
                    self._upsert(spec, seen, id_by_key, connection=scan_conn)
                scan_conn.commit()
            except Exception:
                scan_conn.rollback()
                raise
            finally:
                scan_conn.close()
            # A profile may have been indexed once before and once after an
            # archive/restore move. Remove only stale *index rows* for the
            # same discovered profile; physical artifacts are never touched.
            discovered_profiles = {str(spec.get("profile_id") or "") for spec in specs if spec.get("resource_type") == "target_profile" and spec.get("profile_id")}
            if discovered_profiles:
                placeholders = ",".join("?" for _ in discovered_profiles)
                with self._connect() as conn:
                    stale_rows = conn.execute(
                        "SELECT resource_id FROM runtime_resources WHERE profile_id IN (" + placeholders + ") AND resource_id NOT IN (" + ",".join("?" for _ in seen) + ")",
                        tuple(discovered_profiles) + tuple(seen),
                    ).fetchall()
                    stale_ids = [str(row["resource_id"]) for row in stale_rows]
                    if stale_ids:
                        stale_placeholders = ",".join("?" for _ in stale_ids)
                        conn.execute("DELETE FROM runtime_resource_dependencies WHERE parent_resource_id IN (" + stale_placeholders + ") OR child_resource_id IN (" + stale_placeholders + ")", tuple(stale_ids) + tuple(stale_ids))
                        conn.execute("DELETE FROM runtime_resources WHERE resource_id IN (" + stale_placeholders + ")", tuple(stale_ids))
            # A target-profile deletion can own and physically remove a
            # dedicated configuration version.  Its configuration row is
            # removed as part of the same transaction, but an older derived
            # ``effective_config`` index row used to survive and reappear as a
            # phantom resource.  Reconcile only bindings explicitly recorded
            # in a deletion record and no longer returned by the authoritative
            # configuration service; unrelated missing files remain visible
            # as incomplete resources for maintenance and recovery.
            authoritative_config_bindings = {
                f"{str(item.get('config_profile_id') or '')}:v{int(item.get('version') or 1)}"
                for item in config_rows
                if item.get("config_profile_id")
            }
            deleted_owned_config_bindings: set[str] = set()
            with self._connect() as conn:
                tombstone_rows = conn.execute(
                    "SELECT metadata_json FROM runtime_resource_tombstones"
                ).fetchall()
                for tombstone_row in tombstone_rows:
                    try:
                        tombstone_metadata = json.loads(tombstone_row["metadata_json"] or "{}")
                    except (TypeError, json.JSONDecodeError):
                        continue
                    deleted_owned_config_bindings.update(
                        str(binding)
                        for binding in list(tombstone_metadata.get("owned_configuration_profiles") or [])
                        if binding
                    )
                stale_config_bindings = sorted(deleted_owned_config_bindings - authoritative_config_bindings)
                if stale_config_bindings:
                    binding_placeholders = ",".join("?" for _ in stale_config_bindings)
                    stale_rows = conn.execute(
                        "SELECT resource_id FROM runtime_resources "
                        "WHERE source_type='configuration' AND profile_id IN (" + binding_placeholders + ")",
                        tuple(stale_config_bindings),
                    ).fetchall()
                    stale_config_ids = [str(row["resource_id"]) for row in stale_rows]
                    if stale_config_ids:
                        stale_placeholders = ",".join("?" for _ in stale_config_ids)
                        conn.execute(
                            "DELETE FROM runtime_resource_dependencies WHERE parent_resource_id IN (" + stale_placeholders + ") OR child_resource_id IN (" + stale_placeholders + ")",
                            tuple(stale_config_ids) + tuple(stale_config_ids),
                        )
                        conn.execute(
                            "DELETE FROM runtime_resources WHERE resource_id IN (" + stale_placeholders + ")",
                            tuple(stale_config_ids),
                        )
            dependency_conn = self._connect()
            try:
                dependency_conn.execute("DELETE FROM runtime_resource_dependencies")
                for spec in specs:
                    parent_key = str(spec.get("resource_key") or "")
                    parent_id = id_by_key.get(parent_key)
                    for child_type, child_key, label, required in list(spec.get("_children") or []):
                        self._dependency(parent_id or "", id_by_key.get(child_key) or "", label, required, connection=dependency_conn)
                    if spec.get("resource_type") in {"target_profile", "candidate_profile"} and parent_id:
                        bank_id = str(spec.get("profile_id") or "")
                        # Every config owned by this memory bank is a managed
                        # child.  The currently selected config is runtime state,
                        # not the sole ownership relationship.
                        for config_row in config_rows:
                            if bool(config_row.get("archived")) != bool(spec.get("archived")):
                                continue
                            owner_id = str(config_row.get("memory_bank_id") or config_row.get("target_domain_version") or "")
                            if owner_id != bank_id:
                                continue
                            cfg_id = str(config_row.get("config_profile_id") or "")
                            cfg_ver = int(config_row.get("version") or 0)
                            if cfg_id and cfg_ver:
                                cfg_id_key = f"configuration:{cfg_id}:v{cfg_ver}"
                                self._dependency(parent_id, id_by_key.get(cfg_id_key) or "", "记忆库配置版本", True, connection=dependency_conn)
                        model_key = f"base_model:{str((spec.get('metadata') or {}).get('model_id') or '')}"
                        if id_by_key.get(model_key):
                            self._dependency(parent_id, id_by_key[model_key], "基础模型", True, connection=dependency_conn)
                dependency_conn.commit()
            except Exception:
                dependency_conn.rollback()
                raise
            finally:
                dependency_conn.close()
            # Mark previously indexed records whose files disappeared, but do
            # not delete them: the maintenance tab must be able to explain it.
            with self._connect() as conn:
                old_rows = conn.execute(
                    "SELECT resource_id,resource_type,source_type,artifact_ref,archived,metadata_json "
                    "FROM runtime_resources"
                ).fetchall()
                for row in old_rows:
                    if row["resource_id"] in seen:
                        continue
                    status = "archived" if row["archived"] else "incomplete"
                    if not row["archived"] and row["artifact_ref"]:
                        try:
                            status = "normal" if self._safe_path(row["artifact_ref"]).exists() else "incomplete"
                        except ApiError:
                            status = "incompatible"
                    try:
                        metadata = json.loads(row["metadata_json"] or "{}")
                    except (TypeError, json.JSONDecodeError):
                        metadata = {}
                    # Old target-domain builds could leave configuration index
                    # rows behind after the owned snapshot disappeared.  Keep
                    # the audit/index evidence for advanced maintenance, but do
                    # not expose an unusable technical row as a normal user
                    # configuration.  No artifact is removed here.
                    if row["resource_type"] in {"configuration", "effective_config"} and row["source_type"] == "configuration":
                        metadata.update(
                            ordinary_visible=False,
                            technical_child=True,
                            managed_with_profile=True,
                            legacy_orphan=True,
                        )
                    conn.execute(
                        "UPDATE runtime_resources SET integrity_status=?, metadata_json=?, updated_at=? WHERE resource_id=?",
                        (status, json.dumps(metadata, ensure_ascii=False), _utc_now(), row["resource_id"]),
                    )
            self._record_audit("rebuild_index", None, "operator", None, {"resource_count": len(seen)})
            self._last_scan_at = time.monotonic()
            event = self._bump_revision(action="rebuild_index", resource_ids=list(seen), resource_types=["resource_catalog"])
            return {
                "resource_count": len(seen),
                "dependency_count": self._dependency_count(),
                # Build publication only needs the catalog transaction and its
                # revision.  Rendering the fully enriched resource list here
                # also inspects active jobs and the current Runtime, which can
                # wait on task/profile locks while this very build is still
                # finalising.  Keep that optional for normal catalog APIs and
                # let build publication use the lightweight contract.
                "resources": self.list_resources() if include_resources else [],
                "summary": self.maintenance_summary(scan=False),
                "manifest_cache": self.manifest_cache.metrics(),
                **event,
            }

    def _dependency_count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM runtime_resource_dependencies").fetchone()[0])

    def _effective_runtime_binding(self) -> dict[str, Any]:
        """Resolve the same fallback profile that inference actually uses."""
        runtime = {**dict(self.settings.defaults or {}), **self._runtime_state()}
        if runtime.get("memory_bank_id"):
            return runtime
        # Never turn an explicit user deactivation back into an implicit
        # default binding.  Inference routes use RuntimeProfileService and will
        # return the friendly no-memory-bank state until a profile is applied.
        if bool(runtime.get("runtime_deactivated")):
            return runtime
        try:
            bank = self.domain_profiles.resolve_for_inference(
                None,
                str(runtime.get("model_id") or ""),
                str(runtime.get("dataset") or ""),
                str(runtime.get("category") or ""),
            )
        except Exception:
            return runtime
        runtime["memory_bank_id"] = str(bank.get("memory_bank_id") or "")
        runtime["config_profile_id"] = str(bank.get("config_profile_id") or runtime.get("config_profile_id") or "")
        runtime["config_profile_version"] = int(bank.get("config_profile_version") or runtime.get("config_profile_version") or 0)
        return runtime

    @staticmethod
    def _resource_filter_sql(
        *,
        search: str = "",
        resource_type: str = "",
        dataset: str = "",
        category: str = "",
        status: str = "",
        include_archived: bool = False,
        include_bound: bool = True,
        include_frozen: bool = True,
    ) -> tuple[list[str], list[Any]]:
        del include_bound, include_frozen
        clauses = []
        values: list[Any] = []
        if not include_archived:
            clauses.append("archived=0")
        if resource_type:
            clauses.append("resource_type=?"); values.append(resource_type)
        if dataset:
            clauses.append("lower(COALESCE(dataset,''))=lower(?)"); values.append(dataset)
        if category:
            clauses.append("COALESCE(category,'')=?"); values.append(category)
        if status in STATUS_LABELS:
            clauses.append("integrity_status=?"); values.append(status)
        if search:
            clauses.append("(display_name LIKE ? OR resource_id LIKE ? OR profile_id LIKE ?)"); token = f"%{search}%"; values.extend([token, token, token])
        return clauses, values

    def count_resources(self, **filters: Any) -> int:
        clauses, values = self._resource_filter_sql(**filters)
        query = "SELECT COUNT(*) FROM runtime_resources" + (" WHERE " + " AND ".join(clauses) if clauses else "")
        with self._connect() as conn:
            return int(conn.execute(query, values).fetchone()[0])

    def profile_registered(
        self,
        profile_id: str,
        *,
        resource_type: str,
        include_archived: bool = False,
    ) -> bool:
        """Check catalog publication without resolving Runtime or active jobs."""
        clauses = ["profile_id=?", "resource_type=?"]
        values: list[Any] = [str(profile_id or ""), str(resource_type or "")]
        if not include_archived:
            clauses.append("archived=0")
        query = "SELECT 1 FROM runtime_resources WHERE " + " AND ".join(clauses) + " LIMIT 1"
        with self._connect() as conn:
            return conn.execute(query, values).fetchone() is not None

    def list_resources(
        self,
        *,
        search: str = "",
        resource_type: str = "",
        dataset: str = "",
        category: str = "",
        status: str = "",
        include_archived: bool = False,
        include_bound: bool = True,
        include_frozen: bool = True,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses, values = self._resource_filter_sql(
            search=search,
            resource_type=resource_type,
            dataset=dataset,
            category=category,
            status=status,
            include_archived=include_archived,
            include_bound=include_bound,
            include_frozen=include_frozen,
        )
        query = "SELECT * FROM runtime_resources" + (" WHERE " + " AND ".join(clauses) if clauses else "") + " ORDER BY active DESC, frozen DESC, updated_at DESC, display_name"
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            values.extend([max(1, int(limit)), max(0, int(offset))])
        with self._connect() as conn:
            rows = conn.execute(query, values).fetchall()
            catalog_rows = conn.execute("SELECT * FROM runtime_resources").fetchall()
            dependency_rows = conn.execute(
                "SELECT parent_resource_id,child_resource_id FROM runtime_resource_dependencies"
            ).fetchall()
        items = [self._row(row) for row in rows]
        if not include_bound:
            bound_ids = {str(row["child_resource_id"]) for row in dependency_rows}
            items = [item for item in items if str(item.get("resource_id") or "") not in bound_ids]
        if not include_frozen:
            items = [item for item in items if not item.get("frozen")]

        all_by_id = {
            str(item.get("resource_id") or ""): item
            for item in (self._row(row) for row in catalog_rows)
        }
        parent_ids_by_child: dict[str, list[str]] = {}
        for row in dependency_rows:
            parent_ids_by_child.setdefault(str(row["child_resource_id"]), []).append(str(row["parent_resource_id"]))
        running_tasks = self._running_tasks()
        current_runtime = self._effective_runtime_binding()
        current_profile_id = str(current_runtime.get("memory_bank_id") or "")
        current_config_id = str(current_runtime.get("config_profile_id") or "")
        current_config_version = int(current_runtime.get("config_profile_version") or 0)
        deactivated_profile_ids = {
            str(value)
            for value in list(current_runtime.get("deactivated_profile_ids") or [])
            if str(value)
        }

        target_by_profile = {
            str(item.get("profile_id") or ""): item
            for item in all_by_id.values()
            if item.get("resource_type") in {"target_profile", "candidate_profile"}
        }
        for item in items:
            resource_id = str(item.get("resource_id") or "")
            profile_id = str(item.get("profile_id") or "")
            parents = [all_by_id[parent_id] for parent_id in parent_ids_by_child.get(resource_id, []) if parent_id in all_by_id]
            owner = item if item.get("resource_type") in {"target_profile", "candidate_profile", "configuration"} else target_by_profile.get(profile_id)
            if owner is None:
                owner = next((parent for parent in parents if parent.get("resource_type") in {"target_profile", "candidate_profile", "configuration"}), None)
            operation_target_id = str((owner or item).get("resource_id") or resource_id)
            config_profile_id = ""
            config_profile_version = 0
            if item.get("resource_type") == "configuration":
                config_profile_id, _, raw_version = profile_id.partition(":v")
                try:
                    config_profile_version = int(raw_version or str(item.get("version") or "1").removeprefix("v"))
                except ValueError:
                    config_profile_version = 0
            current_use = bool(
                (profile_id and profile_id == current_profile_id)
                or (
                    item.get("resource_type") == "configuration"
                    and config_profile_id == current_config_id
                    and config_profile_version == current_config_version
                )
                or (owner is not None and str(owner.get("profile_id") or "") == current_profile_id)
            )
            active_tasks = []
            for task in running_tasks:
                task_profiles = {str(task.get("memory_bank_id") or ""), str(task.get("profile_id") or "")}
                uses_profile = bool(profile_id and profile_id in task_profiles)
                uses_config = bool(config_profile_id and config_profile_id == str(task.get("config_profile_id") or ""))
                if uses_profile or uses_config:
                    active_tasks.append(task)
            profile_bound = bool(parents or (owner is not None and owner is not item))
            source_type = str(item.get("source_type") or "")
            if item.get("resource_type") == "configuration":
                if profile_bound or source_type in {"target_domain", "target_domain_snapshot"}:
                    subtype = "profile 配置快照"
                elif item.get("archived"):
                    subtype = "已归档配置"
                elif item.get("frozen"):
                    subtype = "配置模板"
                else:
                    subtype = "可编辑草稿"
            else:
                subtype = str(item.get("resource_type") or "历史资源，类型待确认")
            builtin_profile = self._is_builtin_profile(item)
            if active_tasks:
                management_state = "正在运行任务使用"
            elif current_use and builtin_profile:
                management_state = "可归档（将自动暂停当前方案）"
            elif current_use:
                management_state = "需要先切换当前运行方案"
            elif item.get("archived") or bool((owner or {}).get("archived")):
                management_state = "可删除"
            elif owner is not None and owner is not item:
                management_state = "随所属 profile 管理"
            elif item.get("resource_type") in {"target_profile", "candidate_profile", "configuration"}:
                management_state = "可直接归档"
            else:
                management_state = "只读共享资源"
            item.update(
                profile_bound=profile_bound,
                resource_subtype=subtype,
                current_use=current_use,
                active_task_count=len(active_tasks),
                current_usage_count=len(active_tasks),
                active_tasks=active_tasks,
                management_state=management_state,
                operation_target_id=operation_target_id,
                operation_target_type=str((owner or item).get("resource_type") or ""),
                owner_profile_resource_id=(str(owner.get("resource_id") or "") if owner and owner.get("resource_type") in {"target_profile", "candidate_profile"} else ""),
                lifecycle_status=(
                    "当前使用"
                    if current_use
                    else (
                        "已归档"
                        if item.get("archived") or bool((owner or {}).get("archived"))
                        else ("已停用" if profile_id in deactivated_profile_ids else "可用")
                    )
                ),
            )

        if status:
            status_filters = {
                "current": lambda row: bool(row.get("current_use")),
                "available": lambda row: not row.get("archived") and row.get("integrity_status") in {"normal", "managed", "warning"},
                "frozen": lambda row: bool(row.get("frozen")),
                "profile_bound": lambda row: bool(row.get("profile_bound")),
                "template": lambda row: row.get("resource_subtype") == "配置模板",
                "draft": lambda row: row.get("resource_subtype") == "可编辑草稿",
                "archived": lambda row: bool(row.get("archived")),
                "incomplete": lambda row: row.get("integrity_status") in {"incomplete", "incompatible", "hash_mismatch"},
            }
            predicate = status_filters.get(status)
            if predicate:
                items = [item for item in items if predicate(item)]
        return items

    def get_resource(self, resource_id: str) -> dict[str, Any]:
        item = self._get(resource_id)
        deps = self.dependencies(resource_id)
        refs = self.references(resource_id)
        item["dependencies"] = deps
        item["references"] = refs
        return item

    def usage(self, resource_id: str) -> dict[str, Any]:
        """Explain live use separately from current and historical bindings."""
        item = self._get(resource_id)
        blockers = self._hard_blockers(item)
        current_binding = self._is_current_binding(item)
        references = self.references(resource_id)
        owner_profile = next(
            (
                row
                for row in references
                if row.get("resource_type") in {"target_profile", "candidate_profile"}
            ),
            None,
        )
        if owner_profile and not current_binding:
            current_binding = self._is_current_binding(owner_profile)
        return {
            "resource_id": resource_id,
            "active_task_count": len(blockers),
            "active_tasks": blockers,
            "current_profile_dependency": bool(current_binding),
            "current_profile": {
                "resource_id": (owner_profile or item).get("resource_id"),
                "display_name": (owner_profile or item).get("display_name"),
            } if current_binding else None,
            "historical_profile_references": [
                {
                    "resource_id": row.get("resource_id"),
                    "profile_id": row.get("profile_id"),
                    "display_name": row.get("display_name"),
                    "available": row.get("integrity_status") not in {"incomplete", "incompatible", "hash_mismatch"},
                }
                for row in references
                if row.get("resource_type") in {"target_profile", "candidate_profile"}
                and not self._is_current_binding(row)
            ],
            "management_state": "正在运行任务使用" if blockers else ("需要先切换当前运行方案" if current_binding else "当前没有运行任务占用"),
        }

    def dependencies(self, resource_id: str) -> list[dict[str, Any]]:
        self._get(resource_id)
        with self._connect() as conn:
            rows = conn.execute("SELECT r.* , d.relation_type, d.required FROM runtime_resource_dependencies d JOIN runtime_resources r ON r.resource_id=d.child_resource_id WHERE d.parent_resource_id=? ORDER BY d.required DESC, r.resource_type", (resource_id,)).fetchall()
        result = []
        for row in rows:
            item = self._row(row); item["relation_type"] = row["relation_type"]; item["required"] = bool(row["required"]); result.append(item)
        return result

    def references(self, resource_id: str) -> list[dict[str, Any]]:
        self._get(resource_id)
        with self._connect() as conn:
            rows = conn.execute("SELECT r.* , d.relation_type, d.required FROM runtime_resource_dependencies d JOIN runtime_resources r ON r.resource_id=d.parent_resource_id WHERE d.child_resource_id=? ORDER BY r.resource_type", (resource_id,)).fetchall()
        result = []
        for row in rows:
            item = self._row(row); item["relation_type"] = row["relation_type"]; item["required"] = bool(row["required"]); result.append(item)
        return result

    def validate_resource(self, resource_id: str) -> dict[str, Any]:
        item = self._get(resource_id)
        status = "normal"
        message = "资源完整性验证通过。"
        if item.get("metadata", {}).get("virtual") or item.get("integrity_status") == "managed":
            status = "managed" if item.get("integrity_status") == "managed" else "normal"
        elif not item.get("artifact_ref"):
            status, message = "incomplete", "资源尚未绑定受控文件引用。"
        else:
            try:
                path = self._safe_path(item["artifact_ref"])
                if not path.exists():
                    status, message = "incomplete", "资源文件不存在。"
                elif path.is_file() and item.get("sha256"):
                    digest = _sha256(path) if path.stat().st_size <= 256 * 1024 * 1024 else item.get("sha256")
                    if digest.lower() != str(item["sha256"]).lower():
                        status, message = "hash_mismatch", "资源文件 SHA-256 与注册值不一致。"
                elif path.is_file() and not item.get("sha256"):
                    status, message = "warning", "文件存在，但尚未登记 SHA-256；请在维护页重新扫描。"
            except ApiError as exc:
                status, message = "incompatible", exc.message
        if item.get("resource_type") == "target_profile":
            try:
                bank = self.domain_profiles.get_profile(str(item.get("profile_id")), str(item.get("dataset") or ""), str(item.get("category") or ""), str((item.get("metadata") or {}).get("model_id") or "") or None)
                if not bank.get("frozen"):
                    status, message = "incomplete", "目标域 profile 尚未冻结。"
            except Exception as exc:
                status, message = "incomplete", "目标域 profile 的冻结文件或绑定关系不完整。"
                self.logger.debug("runtime_profile_validation_failed: %s", exc)
        with self._connect() as conn:
            conn.execute("UPDATE runtime_resources SET integrity_status=?, updated_at=? WHERE resource_id=?", (status, _utc_now(), resource_id))
        self._record_audit("validate", resource_id, "operator", {"status": item.get("integrity_status")}, {"status": status, "message": message})
        result = self._get(resource_id); result["validation_message"] = message; return result

    def validate_profile(self, resource_id: str) -> dict[str, Any]:
        with self._activity_lock:
            self._active_profile_validations += 1
        try:
            item = self._get(resource_id)
            if item.get("resource_type") != "target_profile":
                raise ApiError("RESOURCE_NOT_PROFILE", "该资源不是可应用的目标域 profile。", 422)
            validated = [self.validate_resource(resource_id)]
            validated.extend(self.validate_resource(row["resource_id"]) for row in self.dependencies(resource_id))
            failures = [row for row in validated if row.get("integrity_status") in {"incomplete", "incompatible", "hash_mismatch"}]
            if failures:
                return {"valid": False, "profile": self._get(resource_id), "dependencies": validated, "message": "存在缺失、不兼容或哈希异常的依赖，不能应用。"}
            return {"valid": True, "profile": self._get(resource_id), "dependencies": validated, "message": "profile 及其冻结依赖验证通过。"}
        finally:
            with self._activity_lock:
                self._active_profile_validations = max(0, self._active_profile_validations - 1)

    def current_profile(self) -> dict[str, Any]:
        current_id = str(self._runtime_state().get("memory_bank_id") or "")
        rows = self.list_resources(resource_type="target_profile", include_archived=False)
        item = next((row for row in rows if str(row.get("profile_id")) == current_id), None)
        return {"resource": item, "runtime": self._runtime_state(), "inference_mode": "inference_only"}

    def _memory_bank_context(self, memory_bank_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        bank_id = str(memory_bank_id or "")
        item = next(
            (row for row in self.list_resources(resource_type="target_profile", include_archived=False)
             if str(row.get("profile_id") or "") == bank_id),
            None,
        )
        if item is None:
            raise ApiError("MEMORY_BANK_NOT_FOUND", "记忆库不存在或已归档。", 404)
        bank = self.domain_profiles.resolve_for_inference(
            bank_id,
            str((item.get("metadata") or {}).get("model_id") or self.settings.defaults.get("model_id") or ""),
            str(item.get("dataset") or ""),
            str(item.get("category") or ""),
        )
        return item, bank

    def memory_bank_configs(self, memory_bank_id: str, *, include_archived: bool = False) -> dict[str, Any]:
        item, bank = self._memory_bank_context(memory_bank_id)
        configs = self.config_profiles.list_for_memory_bank(memory_bank_id, include_archived=include_archived)
        runtime = self._runtime_state()
        current_bank = str(runtime.get("memory_bank_id") or "") == str(memory_bank_id)
        current_id = str(runtime.get("config_profile_id") or "") if current_bank else ""
        current_version = int(runtime.get("config_profile_version") or 0) if current_bank else 0
        for config in configs:
            config["current"] = (
                str(config.get("config_profile_id") or "") == current_id
                and int(config.get("version") or 0) == current_version
            )
        return {
            "memory_bank": {**bank, "resource_id": item.get("resource_id")},
            "configs": configs,
            "next_default_name": self.config_profiles.next_memory_bank_config_name(memory_bank_id),
        }

    def create_memory_bank_config(self, memory_bank_id: str, values: dict[str, Any]) -> dict[str, Any]:
        if self.test_lock is not None:
            self.test_lock.require_write_allowed("memory_update")
        item, bank = self._memory_bank_context(memory_bank_id)
        if self.test_lock is not None:
            self.test_lock.require_write_allowed("memory_update", profile=item, context=bank)
        result = self.config_profiles.create_for_memory_bank(
            bank,
            values,
            source_profile_id=values.get("source_config_profile_id"),
            source_version=values.get("source_config_profile_version"),
        )
        scan_result = self.scan(force=True)
        self._record_audit("create_memory_bank_config", None, "operator", None, {"memory_bank_id": memory_bank_id, "config": f"{result.get('config_profile_id')}:v{result.get('version')}"})
        return {"config": result, "catalog_revision": scan_result.get("revision")}

    def revise_memory_bank_config(self, memory_bank_id: str, profile_id: str, version: int, values: dict[str, Any]) -> dict[str, Any]:
        if self.test_lock is not None:
            self.test_lock.require_write_allowed("memory_update")
        item, bank = self._memory_bank_context(memory_bank_id)
        if self.test_lock is not None:
            self.test_lock.require_write_allowed("memory_update", profile=item, context=bank)
        result = self.config_profiles.revise_for_memory_bank(bank, profile_id, version, values)
        scan_result = self.scan(force=True)
        self._record_audit("revise_memory_bank_config", None, "operator", {"config": f"{profile_id}:v{version}"}, {"config": f"{profile_id}:v{result.get('version')}"})
        return {"config": result, "catalog_revision": scan_result.get("revision")}

    def delete_memory_bank_config(self, memory_bank_id: str, profile_id: str, version: int, *, acknowledged: bool = False) -> dict[str, Any]:
        if self.test_lock is not None:
            self.test_lock.require_write_allowed("memory_update")
        if not acknowledged:
            raise ApiError("CONFIRMATION_REQUIRED", "请先核对删除预览并勾选确认。", 422)
        self._ensure_resource_mutation_idle()
        item, bank = self._memory_bank_context(memory_bank_id)
        if self.test_lock is not None:
            self.test_lock.require_write_allowed("memory_update", profile=item, context=bank)
        runtime = self._runtime_state()
        result = self.config_profiles.remove_from_memory_bank(
            memory_bank_id,
            profile_id,
            version,
            current_profile_id=str(runtime.get("config_profile_id") or "") if str(runtime.get("memory_bank_id") or "") == str(memory_bank_id) else "",
            current_version=int(runtime.get("config_profile_version") or 0),
        )
        scan_result = self.scan(force=True)
        self._record_audit("delete_memory_bank_config", None, "operator", {"config": f"{profile_id}:v{version}"}, {"logically_deleted": True, "memory_bank_id": memory_bank_id})
        return {**result, "catalog_revision": scan_result.get("revision")}

    @staticmethod
    def _is_feedback_activity(item: dict[str, Any]) -> bool:
        kind = str(item.get("kind") or "").strip().lower()
        return "feedback" in kind or "反馈" in kind

    @classmethod
    def _is_feedback_apply_self(cls, item: dict[str, Any], job_id: str | None) -> bool:
        return bool(
            job_id
            and str(item.get("job_id") or "") == str(job_id)
            and cls._is_feedback_activity(item)
            and str(item.get("status") or "").strip().lower() == "applying"
        )

    def _running_tasks(self, *, exclude_feedback_job_id: str | None = None) -> list[dict[str, str]]:
        """Return active server jobs before an atomic profile switch.

        JobService persists a small manifest for every task.  Reading those
        manifests keeps this registry independent of any one task service and
        also protects the switch after a backend restart, when an in-memory
        task list may no longer be available.
        """
        active_states = {"queued", "loading_model", "running", "paused", "stopping", "cancelling", "calibrating", "building", "validating", "optimizing"}
        heartbeat_ttl_seconds = 180.0
        durable_states = getattr(getattr(self.jobs, "states", None), "list", None)
        if callable(durable_states):
            # JobService keeps this indexed snapshot in SQLite on every state
            # transition.  It is the same persistent source used for restart
            # recovery, but avoids walking every historical job directory for
            # ordinary resource-list and button refresh requests.
            snapshots = durable_states(statuses={"pending", "running", "paused"})
            active: list[dict[str, str]] = []
            for snapshot in snapshots:
                payload = dict(snapshot.get("payload") or {})
                status = str(snapshot.get("runtime_status") or payload.get("status") or "running").lower()
                heartbeat = payload.get("heartbeat_at") or snapshot.get("updated_at") or payload.get("updated_at")
                try:
                    stamped = datetime.fromisoformat(str(heartbeat or "").replace("Z", "+00:00"))
                    if stamped.tzinfo is None:
                        stamped = stamped.replace(tzinfo=timezone.utc)
                    heartbeat_fresh = (datetime.now(timezone.utc) - stamped.astimezone(timezone.utc)).total_seconds() <= heartbeat_ttl_seconds
                except (TypeError, ValueError):
                    heartbeat_fresh = False
                try:
                    owner_pid = int(payload.get("owner_process_id") or 0)
                except (TypeError, ValueError):
                    owner_pid = 0
                owner_alive = _process_is_alive(owner_pid) if owner_pid else heartbeat_fresh
                if not (heartbeat_fresh and owner_alive):
                    try:
                        self.jobs.states.mark_interrupted(
                            str(snapshot.get("task_id") or payload.get("job_id") or ""),
                            "任务心跳已失效或执行进程不存在，系统已自动释放资源占用。",
                        )
                    except Exception:
                        pass
                    continue
                item = {
                    "job_id": str(snapshot.get("task_id") or payload.get("job_id") or ""),
                    "kind": str(snapshot.get("task_kind") or payload.get("kind") or "unknown"),
                    "status": status,
                    "memory_bank_id": str(payload.get("memory_bank_id") or payload.get("profile_id") or ""),
                    "profile_id": str(payload.get("profile_id") or payload.get("memory_bank_id") or ""),
                    "config_profile_id": str(payload.get("config_profile_id") or ""),
                    "config_profile_version": str(payload.get("config_profile_version") or ""),
                    "heartbeat_at": str(heartbeat or ""),
                    "owner_process_id": str(owner_pid or ""),
                    "heartbeat_ttl_seconds": str(int(heartbeat_ttl_seconds)),
                }
                if not self._is_feedback_apply_self(item, exclude_feedback_job_id):
                    active.append(item)
            return active

        # Compatibility fallback for isolated/legacy service construction.
        # The formal application always injects JobService and uses the fast
        # indexed path above.
        root = getattr(self.settings, "jobs_root", None)
        if root is None or not Path(root).is_dir():
            return []
        active: list[dict[str, str]] = []
        try:
            manifests = Path(root).glob("*/job_manifest.json")
        except OSError:
            return []
        for manifest in manifests:
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            status = str(payload.get("status") or "").lower()
            if status not in active_states:
                continue
            heartbeat = payload.get("heartbeat_at") or payload.get("updated_at") or payload.get("created_at")
            try:
                stamped = datetime.fromisoformat(str(heartbeat or "").replace("Z", "+00:00"))
                if stamped.tzinfo is None:
                    stamped = stamped.replace(tzinfo=timezone.utc)
                heartbeat_fresh = (datetime.now(timezone.utc) - stamped.astimezone(timezone.utc)).total_seconds() <= heartbeat_ttl_seconds
            except (TypeError, ValueError):
                heartbeat_fresh = False
            try:
                owner_pid = int(payload.get("owner_process_id") or 0)
            except (TypeError, ValueError):
                owner_pid = 0
            # Older but otherwise live task manifests may not contain an owner
            # PID.  In that case a fresh heartbeat is the task service lease;
            # a present PID must additionally refer to a live process.
            owner_alive = _process_is_alive(owner_pid) if owner_pid else heartbeat_fresh
            if not (heartbeat_fresh and owner_alive):
                # A stale persisted state is history, not an active lease.  Mark
                # it interrupted so later resource checks and the UI converge
                # on the same state after a backend restart or worker crash.
                now = _utc_now()
                payload.update(
                    status="interrupted",
                    updated_at=now,
                    finished_at=payload.get("finished_at") or now,
                    interruption_reason="任务心跳已失效或执行进程不存在，系统已自动释放资源占用。",
                )
                try:
                    temporary = manifest.with_suffix(".json.tmp")
                    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
                    temporary.replace(manifest)
                    self._record_audit(
                        "converge_stale_task",
                        None,
                        "system",
                        {"job_id": payload.get("job_id") or manifest.parent.name, "status": status, "heartbeat_at": heartbeat, "owner_process_id": owner_pid},
                        {"status": "interrupted", "released_resource_lock": True},
                    )
                except Exception:
                    pass
                continue
            if status in active_states:
                item = {
                    "job_id": str(payload.get("job_id") or manifest.parent.name),
                    "kind": str(payload.get("kind") or "unknown"),
                    "status": status,
                    "memory_bank_id": str(payload.get("memory_bank_id") or payload.get("profile_id") or ""),
                    "profile_id": str(payload.get("profile_id") or payload.get("memory_bank_id") or ""),
                    "config_profile_id": str(payload.get("config_profile_id") or ""),
                    "config_profile_version": str(payload.get("config_profile_version") or ""),
                    "heartbeat_at": str(heartbeat or ""),
                    "owner_process_id": str(owner_pid or ""),
                    "heartbeat_ttl_seconds": str(int(heartbeat_ttl_seconds)),
                }
                if not self._is_feedback_apply_self(item, exclude_feedback_job_id):
                    active.append(item)
        return active

    def _runtime_work_activity(self, *, exclude_feedback_job_id: str | None = None) -> list[dict[str, Any]]:
        """Return real work that must finish before a runtime transition."""

        if exclude_feedback_job_id:
            running = self._running_tasks(exclude_feedback_job_id=exclude_feedback_job_id)
        else:
            running = self._running_tasks()
        active: list[dict[str, Any]] = [dict(item) for item in running]
        active.extend(self._domain_build_references())
        try:
            with self._connect() as conn:
                placeholders = ",".join("?" for _ in ACTIVE_TASK_STATUSES)
                rows = conn.execute(
                    f"SELECT job_id,status FROM feedback_optimization_jobs WHERE status IN ({placeholders})",
                    sorted(ACTIVE_TASK_STATUSES),
                ).fetchall()
            active.extend(
                {"job_id": str(row["job_id"]), "kind": "反馈优化", "status": str(row["status"])}
                for row in rows
                if not self._is_feedback_apply_self(
                    {"job_id": str(row["job_id"]), "kind": "反馈优化", "status": str(row["status"])},
                    exclude_feedback_job_id,
                )
            )
        except sqlite3.Error:
            pass
        with self._activity_lock:
            if self._active_profile_validations:
                active.append({"job_id": "profile_validation", "kind": "profile 验证", "status": "validating"})
        warmup = getattr(self, "warmup", None)
        if warmup is not None:
            warmup_status = dict(warmup.status() or {})
            if str(warmup_status.get("state") or "").lower() == "loading":
                active.append({
                    "job_id": "runtime_warmup",
                    "kind": "运行资源初始化",
                    "status": str(warmup_status.get("phase") or "loading"),
                })
        if exclude_feedback_job_id:
            active = [
                item for item in active
                if not self._is_feedback_apply_self(item, exclude_feedback_job_id)
            ]
        unique: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in active:
            key = (str(item.get("job_id") or item.get("kind") or ""), str(item.get("status") or ""))
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    def _runtime_registry_activity(self) -> list[dict[str, Any]]:
        """Return real inference leases that make a Runtime switch unsafe."""

        active: list[dict[str, Any]] = []
        if self.registry is not None:
            try:
                snapshot = dict(self.registry.status() or {})
                for index, runtime in enumerate(snapshot.get("runtimes") or []):
                    count = int(runtime.get("active_count") or 0)
                    if count <= 0:
                        continue
                    active.append({
                        "job_id": f"runtime_lease_{index}",
                        "kind": "runtime_lease",
                        "status": "running",
                        "active_count": count,
                        "model_id": str(runtime.get("model_id") or ""),
                        "dataset": str(runtime.get("dataset") or ""),
                        "category": str(runtime.get("category") or ""),
                    })
            except (AttributeError, TypeError, ValueError):
                self.logger.exception("runtime_switch_registry_status_failed")
        reservations_lock = getattr(self.domain_profiles, "_lock", None)
        if reservations_lock is not None:
            with reservations_lock:
                reservations = {
                    str(profile_id): set(owners or set())
                    for profile_id, owners in dict(getattr(self.domain_profiles, "_reservations", {}) or {}).items()
                }
        else:
            reservations = {
                str(profile_id): set(owners or set())
                for profile_id, owners in dict(getattr(self.domain_profiles, "_reservations", {}) or {}).items()
            }
        for profile_id, owners in reservations.items():
            for owner_id in set(owners or set()):
                active.append({
                    "job_id": str(owner_id),
                    "kind": "runtime_lease",
                    "status": "running",
                    "profile_id": str(profile_id),
                })
        return active

    @staticmethod
    def _runtime_blocker_message(item: dict[str, Any]) -> str:
        kind = str(item.get("kind") or "").strip().lower()
        status = str(item.get("status") or "").strip().lower()
        if kind == "runtime_lease":
            return "检测任务正在使用当前运行资源"
        if "batch" in kind or "批量" in kind:
            return "批量检测任务正在运行"
        if "video" in kind or "视频" in kind:
            return "视频检测任务正在运行"
        if "single" in kind or "image" in kind or "单图" in kind:
            return "单图检测任务正在运行"
        if "feedback" in kind or "反馈" in kind:
            return "反馈候选版本正在应用" if status == "applying" else "反馈优化任务正在运行"
        if "domain" in kind or "目标域" in kind:
            return "目标域资源正在构建"
        if "validation" in kind or "验证" in kind:
            return "候选版本正在验证"
        if "warmup" in kind or "初始化" in kind:
            return "运行资源正在预热"
        if "profile_switch" in kind or "切换" in kind:
            return "运行方案切换事务正在执行"
        return "后台任务正在运行"

    def can_switch_runtime(
        self,
        *,
        exclude_feedback_job_id: str | None = None,
        include_switch_transaction: bool = True,
    ) -> dict[str, Any]:
        """Return one authoritative decision for every Runtime transition.

        A public/operator switch never supplies an exclusion.  The Feedback
        apply workflow may exclude only its exact own job id; all other jobs,
        validations, warmups and inference leases remain blockers.
        """

        blockers = self._runtime_work_activity(exclude_feedback_job_id=exclude_feedback_job_id)
        blockers.extend(self._runtime_registry_activity())
        if include_switch_transaction:
            acquired = self._apply_lock.acquire(blocking=False)
            if acquired:
                self._apply_lock.release()
            else:
                blockers.append({"job_id": "profile_switch", "kind": "profile_switch", "status": "applying"})
        unique: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for blocker in blockers:
            key = (
                str(blocker.get("job_id") or blocker.get("kind") or ""),
                str(blocker.get("kind") or ""),
                str(blocker.get("status") or ""),
            )
            if key not in seen:
                seen.add(key)
                unique.append(dict(blocker))
        summaries = list(dict.fromkeys(self._runtime_blocker_message(item) for item in unique))
        return {
            "allowed": not unique,
            "blockers": unique,
            "summary": "；".join(summaries) if summaries else "当前没有活动任务，可以切换运行方案。",
        }

    def _ensure_runtime_switch_allowed(self, *, exclude_feedback_job_id: str | None = None) -> dict[str, Any]:
        decision = self.can_switch_runtime(exclude_feedback_job_id=exclude_feedback_job_id)
        if not decision["allowed"]:
            only_runtime_leases = all(
                str(item.get("kind") or "") == "runtime_lease"
                for item in decision["blockers"]
            )
            raise ApiError(
                "RUNTIME_IN_USE" if only_runtime_leases else "RUNTIME_TASK_ACTIVE",
                f"当前无法切换运行方案：{decision['summary']}。",
                409,
                {"tasks": decision["blockers"][:30], "summary": decision["summary"]},
            )
        return decision

    def _global_runtime_activity(self, *, exclude_feedback_job_id: str | None = None) -> list[dict[str, Any]]:
        """Return every activity that makes resource mutation unsafe.

        Resource archive/delete is a catalog-wide transaction.  Even a build
        that does not yet own the selected profile can publish new resources
        while the transaction is running, so it must block mutation globally.
        """
        return list(
            self.can_switch_runtime(exclude_feedback_job_id=exclude_feedback_job_id)["blockers"]
        )

    def _ensure_resource_mutation_idle(self, *, exclude_feedback_job_id: str | None = None) -> None:
        active = self._global_runtime_activity(exclude_feedback_job_id=exclude_feedback_job_id)
        if active:
            raise ApiError(
                "RUNTIME_TASK_ACTIVE",
                "当前仍有检测、目标域构建、校准、反馈优化、验证或 profile 切换任务运行，完成后才能归档、恢复或删除资源。",
                409,
                {"tasks": active[:30]},
            )

    def _hard_blockers(self, item: dict[str, Any], *, exclude_feedback_job_id: str | None = None) -> list[dict[str, Any]]:
        """Only active work and lifecycle transactions are hard blockers."""
        profile_id = str(item.get("profile_id") or "")
        resource_type = str(item.get("resource_type") or "")
        metadata = dict(item.get("metadata") or {})
        config_id = str(metadata.get("config_profile_id") or "")
        config_version = int(metadata.get("config_profile_version") or 0)
        if resource_type == "configuration":
            config_id, _, raw_version = profile_id.partition(":v")
            try:
                config_version = int(raw_version or str(item.get("version") or "1").removeprefix("v"))
            except ValueError:
                config_version = 0
        blockers: list[dict[str, Any]] = []
        running = (
            self._running_tasks(exclude_feedback_job_id=exclude_feedback_job_id)
            if exclude_feedback_job_id
            else self._running_tasks()
        )
        for task in running:
            uses_profile = profile_id and profile_id in {
                str(task.get("memory_bank_id") or ""), str(task.get("profile_id") or "")
            }
            uses_config = config_id and config_id == str(task.get("config_profile_id") or "")
            if uses_profile or uses_config:
                blockers.append({**task, "reason": "正在运行的检测任务使用该资源"})
        if resource_type == "target_profile" and profile_id:
            # A loaded registry cache is not a running task lease.  Only a
            # fresh task heartbeat or an explicit reservation may block the
            # lifecycle operation; the cache is invalidated before the move.
            reservations = (getattr(self.domain_profiles, "_reservations", {}) or {}).get(profile_id) or set()
            for owner in reservations:
                blockers.append({"job_id": str(owner), "kind": "检测任务", "status": "running", "reason": "任务仍持有记忆库租约"})
            blockers.extend(
                {**row, "reason": "反馈优化任务正在使用该记忆库"}
                for row in self._optimization_references(profile_id, exclude_feedback_job_id=exclude_feedback_job_id)
            )
        if resource_type == "configuration" and config_id and config_version:
            try:
                with self._connect() as conn:
                    rows = conn.execute(
                        "SELECT job_id,status FROM feedback_optimization_jobs WHERE status IN ('queued','running','optimizing','validating','applying','cancelling') AND ((base_config_profile_id=? AND base_config_profile_version=?) OR (candidate_config_profile_id=? AND candidate_config_profile_version=?))",
                        (config_id, config_version, config_id, config_version),
                    ).fetchall()
                blockers.extend({"job_id": str(row["job_id"]), "kind": "反馈优化", "status": str(row["status"]), "reason": "反馈优化或验证任务正在使用该配置"} for row in rows)
            except sqlite3.Error:
                pass
        if str(item.get("integrity_status") or "") in {"building", "validating", "deleting", "migrating"}:
            blockers.append({"kind": "资源生命周期", "status": str(item.get("integrity_status")), "reason": "资源正在写入、验证或迁移"})
        # Deduplicate restarted-worker observations of the same job.
        unique: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for blocker in blockers:
            key = (str(blocker.get("job_id") or blocker.get("kind") or ""), str(blocker.get("reason") or ""))
            if key not in seen:
                seen.add(key)
                unique.append(blocker)
        return unique

    def _exclusive_configuration_dependencies(
        self,
        item: dict[str, Any],
        dependencies: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Return target-domain config snapshots owned only by one profile.

        Reusable templates never appear here.  A build copies a template into
        a new ``target_domain`` snapshot, and that snapshot follows the owning
        profile through archive, restore and physical deletion.
        """
        if str(item.get("resource_type") or "") != "target_profile":
            return []
        result: list[dict[str, Any]] = []
        for child in list(dependencies if dependencies is not None else self.dependencies(str(item.get("resource_id") or ""))):
            if str(child.get("resource_type") or "") != "configuration":
                continue
            if str(child.get("source_type") or "") not in {"target_domain", "target_domain_snapshot", "memory_bank_config", "legacy_compatible", "builtin_public_baseline"}:
                continue
            other_refs = [
                ref for ref in self.references(str(child.get("resource_id") or ""))
                if str(ref.get("resource_id") or "") != str(item.get("resource_id") or "")
            ]
            if not other_refs:
                result.append(child)
        return result

    def _configuration_resource_ids(self, configurations: list[dict[str, Any]]) -> list[str]:
        """Return config rows and their technical effective-config children.

        A memory-bank config is represented by one user-facing ``configuration``
        row and one hidden ``effective_config`` row.  Both share the same
        ``profile_id`` binding and must follow the parent memory bank through
        archive, restore and deletion.
        """
        bindings = sorted(
            {
                str(configuration.get("profile_id") or "")
                for configuration in configurations
                if configuration.get("profile_id")
            }
        )
        if not bindings:
            return []
        placeholders = ",".join("?" for _ in bindings)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT resource_id FROM runtime_resources "
                "WHERE profile_id IN (" + placeholders + ") "
                "AND resource_type IN ('configuration','effective_config')",
                tuple(bindings),
            ).fetchall()
        return list(dict.fromkeys(str(row["resource_id"]) for row in rows))

    def _is_current_binding(self, item: dict[str, Any]) -> bool:
        runtime = self._effective_runtime_binding()
        if str(item.get("resource_type") or "") == "target_profile":
            return str(item.get("profile_id") or "") == str(runtime.get("memory_bank_id") or "")
        if str(item.get("resource_type") or "") == "configuration":
            profile_id, _, version = str(item.get("profile_id") or "").partition(":v")
            return profile_id == str(runtime.get("config_profile_id") or "") and int(version or 0) == int(runtime.get("config_profile_version") or 0)
        return False

    @staticmethod
    def _is_builtin_profile(item: dict[str, Any]) -> bool:
        return (
            str(item.get("resource_type") or "") == "target_profile"
            and (
                str(item.get("source_type") or "") == "builtin_public_baseline"
                or str(item.get("profile_id") or "").startswith("builtin_")
            )
        )

    def _builtin_archive_root(self) -> Path:
        root = (self.settings.memory_banks_root / "_archive" / "_builtin_public_baselines").resolve()
        if not _within(root, self.settings.memory_banks_root.resolve()):
            raise ApiError("RESOURCE_PATH_FORBIDDEN", "内置记忆库归档目录不受控。", 403)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _move_builtin_profile_to_archive(self, item: dict[str, Any]) -> dict[str, Any]:
        """Move an opt-out built-in memory bank into the managed archive.

        Built-in public baselines live under ``outputs/checkpoints`` rather
        than the regular target-profile directory.  They are still ordinary
        local memory-bank artifacts from the operator's point of view, so the
        lifecycle manager must be able to archive, restore and delete them
        without treating the shared model weights as owned files.
        """
        profile_id = str(item.get("profile_id") or "")
        if not profile_id.startswith("builtin_"):
            raise ApiError("BUILTIN_PROFILE_ID_INVALID", "内置记忆库标识无效。", 422)
        live_ref = str(item.get("artifact_ref") or "")
        source = self._safe_path(live_ref)
        checkpoints_root = (self.settings.algorithm_outputs_root / "checkpoints").resolve()
        if not source.is_dir() or not _within(source, checkpoints_root) or source == checkpoints_root:
            raise ApiError("BUILTIN_MEMORY_BANK_MISSING", "内置记忆库目录不存在或不在受控检查点目录中。", 409)
        destination = self._builtin_archive_root() / profile_id
        if destination.exists():
            raise ApiError("MEMORY_BANK_ARCHIVE_CONFLICT", "该内置记忆库已经存在归档副本。", 409)
        if self.registry is not None:
            self.registry.invalidate(str(item.get("dataset") or "").lower(), str(item.get("category") or ""))
        os.replace(source, destination)
        return {
            "source": source,
            "destination": destination,
            "live_ref": live_ref,
            "archive_ref": self._relative_ref(destination),
        }

    def _restore_builtin_profile(self, item: dict[str, Any], actor: str) -> dict[str, Any]:
        profile_id = str(item.get("profile_id") or "")
        metadata = dict(item.get("metadata") or {})
        archive_ref = str(item.get("artifact_ref") or "")
        live_ref = str(metadata.get("builtin_live_artifact_ref") or "")
        if not live_ref:
            raise ApiError("BUILTIN_RESTORE_TARGET_MISSING", "内置记忆库缺少原始位置记录，无法恢复。", 409)
        source = self._safe_path(archive_ref)
        target = self._safe_path(live_ref)
        archive_root = self._builtin_archive_root()
        checkpoints_root = (self.settings.algorithm_outputs_root / "checkpoints").resolve()
        if not source.is_dir() or not _within(source, archive_root):
            raise ApiError("MEMORY_BANK_ARCHIVE_NOT_FOUND", "内置记忆库归档文件不存在。", 404)
        if not _within(target, checkpoints_root) or target == checkpoints_root:
            raise ApiError("BUILTIN_RESTORE_TARGET_INVALID", "内置记忆库原始位置不受控。", 409)
        if target.exists():
            raise ApiError("MEMORY_BANK_RESTORE_CONFLICT", "原始位置已经存在同名记忆库，无法恢复。", 409)
        target.parent.mkdir(parents=True, exist_ok=True)
        before = dict(item)
        os.replace(source, target)
        try:
            restored_metadata = dict(metadata)
            restored_metadata.pop("builtin_archived_artifact_ref", None)
            restored_metadata["builtin_last_restored_at"] = _utc_now()
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT resource_id,artifact_ref FROM runtime_resources WHERE profile_id=?",
                    (profile_id,),
                ).fetchall()
                for row in rows:
                    ref = str(row["artifact_ref"] or "")
                    restored_ref = live_ref + ref[len(archive_ref):] if ref.startswith(archive_ref) else ref
                    conn.execute(
                        "UPDATE runtime_resources SET artifact_ref=?,archived=0,active=0,integrity_status='normal',last_archived_at=COALESCE(archived_at,last_archived_at),archived_at=NULL,updated_at=? WHERE resource_id=?",
                        (restored_ref, _utc_now(), str(row["resource_id"])),
                    )
                conn.execute(
                    "UPDATE runtime_resources SET artifact_ref=?,metadata_json=?,archived=0,active=0,integrity_status='normal',last_archived_at=COALESCE(archived_at,last_archived_at),archived_at=NULL,updated_at=? WHERE resource_id=?",
                    (live_ref, _json(restored_metadata), _utc_now(), str(item.get("resource_id") or "")),
                )
        except Exception:
            if target.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                os.replace(target, source)
            raise
        if self.registry is not None:
            self.registry.invalidate(str(item.get("dataset") or "").lower(), str(item.get("category") or ""))
        result = self._get(str(item.get("resource_id") or ""))
        self._record_audit("restore", str(item.get("resource_id") or ""), actor, before, result)
        event = self._bump_revision(
            action="restore",
            resource_ids=[str(item.get("resource_id") or "")],
            profile_ids=[profile_id],
            resource_types=["target_profile", "normal_memory_bank", "coreset_index"],
        )
        return {**result, **event}

    def _pause_runtime_profile(self, actor: str) -> dict[str, Any]:
        before = self._runtime_state()
        previous_profile_id = str(before.get("memory_bank_id") or "")
        if not previous_profile_id and bool(before.get("runtime_deactivated")):
            return {"runtime": before, "deactivated": True, "changed": False, "previous_runtime": before}
        deactivated_profile_ids = {
            str(value) for value in list(before.get("deactivated_profile_ids") or []) if str(value)
        }
        if previous_profile_id:
            deactivated_profile_ids.add(previous_profile_id)
        updates = {
            **before,
            "model_id": "",
            "dataset": "",
            "category": "",
            "config": "",
            "memory_bank_id": "",
            "threshold": None,
            "config_profile_id": "",
            "config_profile_version": 0,
            "config_hash": "",
            "frozen": False,
            "inference_mode": "inference_only",
            "runtime_deactivated": True,
            "deactivated_profile_ids": sorted(deactivated_profile_ids),
            "deactivated_at": _utc_now(),
        }
        previous_config_id = str(before.get("config_profile_id") or "")
        previous_config_version = int(before.get("config_profile_version") or 0)
        try:
            # Publish the complete empty binding before releasing the old
            # runtime.  Health, status, configuration and detection endpoints
            # now observe the same revisioned RuntimeContext while cleanup is
            # still in progress.
            transition_runtime = self._write_runtime_state(updates, runtime_status="deactivating")
            if self.registry is not None:
                self.registry.invalidate(fail_on_active=True)
            if previous_profile_id and hasattr(self.domain_profiles, "deactivate"):
                self.domain_profiles.deactivate(previous_profile_id)
            if previous_config_id and previous_config_version:
                self.config_profiles.mark_applied(previous_config_id, previous_config_version, False)
            with self._connect() as conn:
                conn.execute("UPDATE runtime_resources SET active=0,updated_at=? WHERE active=1", (_utc_now(),))
        except Exception:
            previous_status = str(before.get("runtime_status") or ("ready" if previous_profile_id else "not_ready"))
            self._write_runtime_state(before, runtime_status=previous_status)
            if previous_profile_id:
                try:
                    self.domain_profiles.activate(
                        previous_profile_id,
                        str(before.get("dataset") or ""),
                        str(before.get("category") or ""),
                        str(before.get("model_id") or ""),
                    )
                except Exception:
                    self.logger.exception("runtime_profile_deactivate_domain_rollback_failed")
            if previous_config_id and previous_config_version:
                try:
                    self.config_profiles.mark_applied(previous_config_id, previous_config_version, True)
                except Exception:
                    self.logger.exception("runtime_profile_deactivate_config_rollback_failed")
            with self._connect() as conn:
                conn.execute("UPDATE runtime_resources SET active=0,updated_at=? WHERE resource_type='target_profile'", (_utc_now(),))
                if previous_profile_id:
                    conn.execute("UPDATE runtime_resources SET active=1,updated_at=? WHERE resource_type='target_profile' AND profile_id=?", (_utc_now(), previous_profile_id))
            raise
        if self.runtime_profiles is not None:
            self.runtime_profiles.set_runtime_status(
                "inactive",
                expected_revision=int(transition_runtime.get("runtime_revision") or 0),
                expected_profile_id="",
            )
        after = self._runtime_state()
        self._record_audit("deactivate_runtime_profile", None, actor, before, after)
        event = self._bump_revision(
            action="deactivate_runtime_profile",
            profile_ids=[previous_profile_id] if previous_profile_id else [],
            resource_types=["target_profile", "configuration", "runtime"],
            current_profile_changed=True,
        )
        warmup = getattr(self, "warmup", None)
        if warmup is not None:
            warmup.restart()
        return {"runtime": after, "deactivated": True, "changed": True, "previous_runtime": before, **event}

    def deactivate_current_profile(self, actor: str = "operator") -> dict[str, Any]:
        """Atomically clear the complete runtime binding after work is idle."""

        with self._apply_lock:
            self._ensure_runtime_switch_allowed()
            return self._pause_runtime_profile(actor)

    def _transition_current_profile(self, item: dict[str, Any], action: str, replacement_resource_id: str | None,
                                    actor: str) -> dict[str, Any]:
        if not self._is_current_binding(item):
            return {"changed": False, "before": self._runtime_state()}
        self._ensure_runtime_switch_allowed()
        before = self._runtime_state()
        action = str(action or "cancel")
        # Public baselines are no longer deletion-protected.  A direct
        # lifecycle action on the currently selected built-in profile safely
        # pauses new inference instead of forcing the operator through an
        # otherwise impossible replacement-profile requirement.
        if action == "cancel" and self._is_builtin_profile(item):
            action = "pause"
        if action == "switch":
            if not replacement_resource_id or str(replacement_resource_id) == str(item.get("resource_id") or ""):
                raise ApiError("REPLACEMENT_PROFILE_REQUIRED", "请选择另一个完整、已冻结且验证通过的运行方案。", 422)
            self.apply_profile(str(replacement_resource_id), actor)
        elif action == "restore_default":
            default = next(
                (
                    row for row in self.list_resources(resource_type="target_profile")
                    if str(row.get("resource_id")) != str(item.get("resource_id"))
                    and bool((row.get("metadata") or {}).get("is_default"))
                    and deployment_scope(row) == "formal"
                ),
                None,
            )
            if default is None:
                raise ApiError("DEFAULT_PROFILE_NOT_FOUND", "没有可替换的默认冻结方案。", 409)
            self.apply_profile(str(default["resource_id"]), actor)
        elif action == "pause":
            self._pause_runtime_profile(actor)
        else:
            raise ApiError("CURRENT_PROFILE_ACTION_REQUIRED", "当前运行方案引用该资源，请先选择切换、恢复默认或暂停检测。", 409)
        return {"changed": True, "before": before, "runtime": self._runtime_state()}

    def _restore_runtime_snapshot(self, before: dict[str, Any], actor: str = "system") -> None:
        self._write_runtime_state(dict(before or {}))
        if self.registry is not None:
            self.registry.invalidate()
        previous_id = str((before or {}).get("memory_bank_id") or "")
        if previous_id:
            try:
                self.domain_profiles.activate(previous_id, str(before.get("dataset") or ""), str(before.get("category") or ""), str(before.get("model_id") or ""))
            except Exception:
                self.logger.exception("runtime_profile_restore_after_resource_failure")
        with self._connect() as conn:
            conn.execute("UPDATE runtime_resources SET active=0,updated_at=? WHERE resource_type='target_profile'", (_utc_now(),))
            if previous_id:
                conn.execute("UPDATE runtime_resources SET active=1,updated_at=? WHERE resource_type='target_profile' AND profile_id=?", (_utc_now(), previous_id))

    def apply_profile(
        self,
        resource_id: str,
        actor: str = "operator",
        *,
        config_profile_id: str | None = None,
        config_profile_version: int | None = None,
        feedback_job_id: str | None = None,
    ) -> dict[str, Any]:
        if self.test_lock is not None:
            self.test_lock.require_write_allowed("profile_apply")
        with self._apply_lock:
            # Recheck under the atomic switch lock.  Public callers never pass
            # an exclusion; a Feedback apply may ignore only its own exact job.
            self._ensure_runtime_switch_allowed(exclude_feedback_job_id=feedback_job_id)
            item = self._get(resource_id)
            if item.get("resource_type") != "target_profile":
                raise ApiError("RESOURCE_NOT_PROFILE", "只能应用完整目标域 profile，不能单独替换模型、记忆库或阈值。", 422)
            if deployment_scope(item) == "test_only":
                raise ApiError("TEST_PROFILE_NOT_APPLICABLE", "测试 Profile 已隔离，不能设为当前正式运行方案。", 409)
            if not item.get("frozen") or item.get("archived"):
                raise ApiError("PROFILE_NOT_APPLICABLE", "候选、未冻结或已归档 profile 不能用于正式推理。", 409)
            check = self.validate_profile(resource_id)
            if not check.get("valid"):
                raise ApiError("PROFILE_INTEGRITY_FAILED", str(check.get("message") or "profile 验证失败。"), 409, {"dependencies": check.get("dependencies") or []})
            bank = self.domain_profiles.resolve_for_inference(str(item.get("profile_id")), str((item.get("metadata") or {}).get("model_id") or self.settings.defaults.get("model_id") or ""), str(item.get("dataset") or ""), str(item.get("category") or ""))
            before = self._runtime_state()
            requested_is_current_bank = str(before.get("memory_bank_id") or "") == str(bank.get("memory_bank_id") or "")
            config_id = str(
                config_profile_id
                or (before.get("config_profile_id") if requested_is_current_bank else "")
                or bank.get("config_profile_id")
                or item.get("metadata", {}).get("config_profile_id")
                or ""
            )
            config_version = int(
                config_profile_version
                or (before.get("config_profile_version") if requested_is_current_bank else 0)
                or bank.get("config_profile_version")
                or item.get("metadata", {}).get("config_profile_version")
                or 0
            )
            if not config_id or not config_version:
                raise ApiError("PROFILE_CONFIG_MISSING", "profile 缺少冻结配置绑定，不能应用。", 409)
            config = self.config_profiles.get(config_id, config_version)
            compatibility = self.config_profiles.compatibility(config_id, config_version, bank)
            if not compatibility.get("compatible"):
                raise ApiError("PROFILE_CONFIG_INCOMPATIBLE", "profile 与配置方案不兼容，不能应用。", 409, {"reasons": compatibility.get("reasons") or []})
            previous_id = str(before.get("memory_bank_id") or "")
            previous_config_id = str(before.get("config_profile_id") or "")
            previous_config_version = int(before.get("config_profile_version") or 0)
            activated_new = False
            applied_new = False
            try:
                updates = {
                    "model_id": str(bank.get("model_id") or config.get("config", {}).get("model_id") or self.settings.defaults.get("model_id") or ""),
                    "dataset": str(bank.get("dataset") or "").lower(),
                    "category": str(bank.get("category") or ""),
                    "device": str(config.get("config", {}).get("device") or bank.get("device") or before.get("device") or "auto"),
                    "config": str(config.get("config", {}).get("config") or before.get("config") or ""),
                    "memory_bank_id": str(bank.get("memory_bank_id") or item.get("profile_id")),
                    "threshold": config.get("threshold") if config.get("threshold") is not None else bank.get("threshold"),
                    "config_profile_id": config_id,
                    "config_profile_version": config_version,
                    "config_hash": str(config.get("config_hash") or bank.get("config_hash") or ""),
                    "inference_mode": "inference_only",
                    "runtime_deactivated": False,
                    "deactivated_at": None,
                    "save_visuals": bool(config.get("config", {}).get("save_visuals", True)),
                    "save_original": bool(config.get("config", {}).get("save_original", True)),
                }
                deactivated_profile_ids = {
                    str(value) for value in list(before.get("deactivated_profile_ids") or []) if str(value)
                }
                if previous_id and previous_id != str(bank.get("memory_bank_id") or ""):
                    deactivated_profile_ids.add(previous_id)
                deactivated_profile_ids.discard(str(bank.get("memory_bank_id") or ""))
                updates["deactivated_profile_ids"] = sorted(deactivated_profile_ids)
                # Publish the complete binding as one revision, but keep it
                # non-inferable until ModelWarmup verifies the Runtime entry.
                # Health/readiness and all detection routes therefore observe
                # either the old ready context or the new loading context—never
                # a partially switched mixture.
                self._write_runtime_state(updates, runtime_status="loading")
                self.domain_profiles.activate(str(item.get("profile_id")), str(bank.get("dataset")), str(bank.get("category")), str(bank.get("model_id") or updates["model_id"]))
                activated_new = True
                self.config_profiles.mark_applied(config_id, config_version, True)
                applied_new = True
                if self.registry is not None:
                    self.registry.invalidate()
                with self._connect() as conn:
                    applied_at = _utc_now()
                    metadata = dict(item.get("metadata") or {})
                    metadata["last_applied_at"] = applied_at
                    conn.execute("UPDATE runtime_resources SET active=0, updated_at=? WHERE resource_type='target_profile'", (_utc_now(),))
                    conn.execute("UPDATE runtime_resources SET active=1, metadata_json=?, updated_at=? WHERE resource_id=?", (_json(metadata), applied_at, resource_id))
                after = self._runtime_state()
                self._record_audit("apply_profile", resource_id, actor, before, after)
                event = self._bump_revision(action="apply_profile", resource_ids=[resource_id], profile_ids=[str(item.get("profile_id") or "")], resource_types=["target_profile", "configuration", "runtime"], current_profile_changed=True)
                warmup = getattr(self, "warmup", None)
                if warmup is not None:
                    warmup.restart()
                return {"runtime": after, "resource": self._get(resource_id), "previous_profile_id": previous_id or None, **event}
            except Exception:
                self._write_runtime_state(before)
                if self.registry is not None:
                    try:
                        self.registry.invalidate()
                    except Exception:
                        self.logger.exception("runtime_profile_registry_rollback_failed")
                # Restore every side effect that may have completed before a
                # later dependency or database write failed.  This keeps the
                # active profile, config applied flags and dynamic settings in
                # one consistent snapshot.
                if activated_new and previous_id:
                    try:
                        self.domain_profiles.activate(previous_id, str(before.get("dataset") or ""), str(before.get("category") or ""), str(before.get("model_id") or ""))
                    except Exception:
                        self.logger.exception("runtime_profile_domain_rollback_failed")
                if applied_new:
                    try:
                        self.config_profiles.mark_applied(config_id, config_version, False)
                        if previous_config_id and previous_config_version:
                            self.config_profiles.mark_applied(previous_config_id, previous_config_version, True)
                    except Exception:
                        self.logger.exception("runtime_profile_config_rollback_failed")
                try:
                    with self._connect() as conn:
                        conn.execute("UPDATE runtime_resources SET active=0, updated_at=? WHERE resource_type='target_profile'", (_utc_now(),))
                        if previous_id:
                            conn.execute("UPDATE runtime_resources SET active=1, updated_at=? WHERE profile_id=? AND resource_type='target_profile'", (_utc_now(), previous_id))
                except Exception:
                    self.logger.exception("runtime_profile_index_rollback_failed")
                raise

    def rollback(self, actor: str = "operator") -> dict[str, Any]:
        if self.test_lock is not None:
            self.test_lock.require_write_allowed("profile_rollback")
        with self._apply_lock:
            self._ensure_runtime_switch_allowed()
            with self._connect() as conn:
                row = conn.execute("SELECT before_json,resource_id FROM runtime_resource_audit WHERE action='apply_profile' ORDER BY created_at DESC LIMIT 1").fetchone()
            if row is None:
                raise ApiError("RUNTIME_ROLLBACK_EMPTY", "暂无可回滚的运行方案。", 409)
            try:
                before = json.loads(row["before_json"] or "{}")
            except json.JSONDecodeError as exc:
                raise ApiError("RUNTIME_ROLLBACK_INVALID", "上一个运行方案快照无法读取。", 409) from exc
            current = self._runtime_state()
            previous_id = str(before.get("memory_bank_id") or "")
            if previous_id:
                previous = next((item for item in self.list_resources(resource_type="target_profile") if str(item.get("profile_id")) == previous_id), None)
                if previous is None:
                    raise ApiError("RUNTIME_ROLLBACK_PROFILE_MISSING", "上一个 profile 已不在资源索引中，无法安全回滚。", 409)
                if deployment_scope(previous) == "test_only":
                    raise ApiError("RUNTIME_ROLLBACK_TEST_PROFILE", "上一个方案属于测试 Profile，正式运行环境禁止回滚到该资源。", 409)
                check = self.validate_profile(str(previous["resource_id"]))
                if not check.get("valid"):
                    raise ApiError("RUNTIME_ROLLBACK_PROFILE_INVALID", "上一个 profile 完整性校验失败，未执行回滚。", 409)
            self._write_runtime_state(before)
            if previous_id:
                self.domain_profiles.activate(previous_id, str(before.get("dataset") or ""), str(before.get("category") or ""), str(before.get("model_id") or ""))
            with self._connect() as conn:
                conn.execute("UPDATE runtime_resources SET active=0, updated_at=? WHERE resource_type='target_profile'", (_utc_now(),))
                if previous_id:
                    conn.execute("UPDATE runtime_resources SET active=1, updated_at=? WHERE profile_id=? AND resource_type='target_profile'", (_utc_now(), previous_id))
            self._record_audit("rollback_profile", row["resource_id"], actor, current, before)
            event = self._bump_revision(action="rollback_profile", resource_ids=[str(row["resource_id"] or "")], profile_ids=[previous_id], resource_types=["target_profile", "configuration", "runtime"], current_profile_changed=True)
            return {"runtime": self._runtime_state(), "rolled_back_profile_id": previous_id or None, **event}

    def archive(self, resource_id: str, actor: str = "local_operator", *, current_profile_action: str = "cancel",
                replacement_resource_id: str | None = None, exclude_feedback_job_id: str | None = None) -> dict[str, Any]:
        if self.test_lock is not None:
            self.test_lock.require_write_allowed("resource_lifecycle")
        self._ensure_resource_mutation_idle(exclude_feedback_job_id=exclude_feedback_job_id)
        item = self._get(resource_id)
        profile_id = str(item.get("profile_id") or "")
        blockers = self._hard_blockers(item, exclude_feedback_job_id=exclude_feedback_job_id)
        if blockers:
            raise ApiError("RESOURCE_IN_USE", "该资源正被运行中的任务使用，暂时不能归档。", 409, {"tasks": blockers})
        before = item
        kind = str(item.get("resource_type"))
        if kind not in {"target_profile", "candidate_profile", "configuration"}:
            raise ApiError("RESOURCE_ARCHIVE_UNSUPPORTED", "只能归档完整目标域 profile、候选 profile 或配置方案。", 422)
        transition = {"changed": False, "before": self._runtime_state()}
        affected_profiles: list[str] = []
        archive_applied = False
        builtin_move: dict[str, Any] | None = None
        archived_configurations: list[dict[str, Any]] = []
        archive_timestamp = _utc_now()
        try:
            transition = self._transition_current_profile(item, current_profile_action, replacement_resource_id, actor)
            refs = self.references(resource_id)
            if kind == "target_profile":
                if self._is_builtin_profile(item):
                    builtin_move = self._move_builtin_profile_to_archive(item)
                    archive_applied = True
                else:
                    builtin_move = None
                    self.domain_profiles.archive(profile_id)
                    archive_applied = True
                    for configuration in self._exclusive_configuration_dependencies(item):
                        config_profile_id, _, raw_version = str(configuration.get("profile_id") or "").partition(":v")
                        config_version = int(raw_version or str(configuration.get("version") or "1").removeprefix("v"))
                        self.config_profiles.archive_version(config_profile_id, config_version)
                        archived_configurations.append(configuration)
            elif kind == "candidate_profile":
                source = self._safe_path(str(item.get("artifact_ref") or ""))
                if source.is_dir():
                    destination_root = self.settings.memory_banks_root / "_archive"
                    destination_root.mkdir(parents=True, exist_ok=True)
                    destination = destination_root / source.name
                    if destination.exists():
                        destination = destination_root / f"{source.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    shutil.move(str(source), str(destination))
                    archive_applied = True
            else:
                pid, _, raw_version = profile_id.partition(":v")
                version = int(raw_version or str(item.get("version") or "1").removeprefix("v"))
                self.config_profiles.archive_version(pid, version)
                archive_applied = True
                affected_profiles = [str(ref.get("profile_id") or "") for ref in refs if ref.get("resource_type") == "target_profile" and ref.get("profile_id")]
                if affected_profiles:
                    self.domain_profiles.mark_unavailable(affected_profiles, "resource_archived", resource_id)
            with self._connect() as conn:
                conn.execute(
                    "UPDATE runtime_resources SET archived=1, active=0, integrity_status='archived', archived_at=?, updated_at=? WHERE resource_id=?",
                    (archive_timestamp, archive_timestamp, resource_id),
                )
                if kind in {"target_profile", "candidate_profile"}:
                    conn.execute(
                        "UPDATE runtime_resources SET archived=1, active=0, integrity_status='archived', archived_at=?, updated_at=? WHERE profile_id=? AND resource_type NOT IN ('target_profile','configuration','base_model','model_weight')",
                        (archive_timestamp, archive_timestamp, profile_id),
                    )
                if kind == "target_profile" and self._is_builtin_profile(item) and builtin_move:
                    metadata = dict(item.get("metadata") or {})
                    metadata.update(
                        builtin_live_artifact_ref=builtin_move["live_ref"],
                        builtin_archived_artifact_ref=builtin_move["archive_ref"],
                        builtin_archived_at=archive_timestamp,
                    )
                    rows = conn.execute(
                        "SELECT resource_id,artifact_ref FROM runtime_resources WHERE profile_id=?",
                        (profile_id,),
                    ).fetchall()
                    for row in rows:
                        ref = str(row["artifact_ref"] or "")
                        archived_ref = builtin_move["archive_ref"] + ref[len(builtin_move["live_ref"]):] if ref.startswith(builtin_move["live_ref"]) else ref
                        conn.execute(
                            "UPDATE runtime_resources SET artifact_ref=? WHERE resource_id=?",
                            (archived_ref, str(row["resource_id"])),
                        )
                    conn.execute(
                        "UPDATE runtime_resources SET artifact_ref=?,metadata_json=? WHERE resource_id=?",
                        (builtin_move["archive_ref"], _json(metadata), resource_id),
                    )
                for configuration in archived_configurations:
                    conn.execute(
                        "UPDATE runtime_resources SET archived=1, active=0, integrity_status='archived', archived_at=?, updated_at=? "
                        "WHERE profile_id=? AND resource_type IN ('configuration','effective_config')",
                        (archive_timestamp, archive_timestamp, str(configuration.get("profile_id") or "")),
                    )
            result = self._get(resource_id)
            self._record_audit("archive", resource_id, actor, before, {"resource": result, "historical_references": refs, "affected_profiles": affected_profiles})
            configuration_ids = [str(configuration.get("resource_id") or "") for configuration in archived_configurations]
            event = self._bump_revision(action="archive", resource_ids=[resource_id, *configuration_ids], profile_ids=[profile_id, *affected_profiles], resource_types=[kind, *("configuration" for _ in configuration_ids)], current_profile_changed=bool(transition.get("changed")))
            return {**result, **event, "affected_profile_ids": affected_profiles, "archived_configuration_ids": configuration_ids}
        except Exception:
            if archive_applied:
                try:
                    if kind == "target_profile":
                        if self._is_builtin_profile(item) and builtin_move:
                            source = Path(builtin_move["source"])
                            destination = Path(builtin_move["destination"])
                            if destination.exists() and not source.exists():
                                source.parent.mkdir(parents=True, exist_ok=True)
                                os.replace(destination, source)
                        else:
                            self.domain_profiles.restore(profile_id)
                            for configuration in archived_configurations:
                                config_profile_id, _, raw_version = str(configuration.get("profile_id") or "").partition(":v")
                                config_version = int(raw_version or str(configuration.get("version") or "1").removeprefix("v"))
                                self.config_profiles.restore_version(config_profile_id, config_version)
                    elif kind == "configuration":
                        pid, _, raw_version = profile_id.partition(":v")
                        version = int(raw_version or str(item.get("version") or "1").removeprefix("v"))
                        self.config_profiles.restore_version(pid, version)
                        if affected_profiles:
                            self.domain_profiles.clear_unavailable(affected_profiles)
                except Exception:
                    self.logger.exception("runtime_resource_archive_rollback_failed resource_id=%s", resource_id)
            if transition.get("changed"):
                self._restore_runtime_snapshot(dict(transition.get("before") or {}))
            raise

    def restore(self, resource_id: str, actor: str = "local_operator") -> dict[str, Any]:
        if self.test_lock is not None:
            self.test_lock.require_write_allowed("resource_lifecycle")
        self._ensure_resource_mutation_idle()
        item = self._get(resource_id)
        if item.get("resource_type") not in {"target_profile", "configuration"}:
            raise ApiError("RESOURCE_NOT_PROFILE", "只有目标域 profile 和配置方案支持归档恢复。", 422)
        if not item.get("archived"):
            raise ApiError("RESOURCE_NOT_ARCHIVED", "该资源当前不是已归档状态。", 409)
        if item.get("active"):
            raise ApiError("RESOURCE_IN_USE", "当前运行方案不能恢复或修改。", 409)
        if self._is_builtin_profile(item):
            return self._restore_builtin_profile(item, actor)
        before = item
        affected_profiles: list[str] = []
        restored_configurations: list[dict[str, Any]] = []
        restored_profile = False
        restored_configuration_versions: list[tuple[str, int]] = []
        try:
            if item.get("resource_type") == "target_profile":
                restored_configurations = self._exclusive_configuration_dependencies(item)
                restored = self.domain_profiles.restore(str(item.get("profile_id")))
                restored_profile = True
                for configuration in restored_configurations:
                    if not configuration.get("archived"):
                        continue
                    config_profile_id, _, raw_version = str(configuration.get("profile_id") or "").partition(":v")
                    config_version = int(raw_version or str(configuration.get("version") or "1").removeprefix("v"))
                    self.config_profiles.restore_version(config_profile_id, config_version)
                    restored_configuration_versions.append((config_profile_id, config_version))
            else:
                pid, _, raw_version = str(item.get("profile_id") or "").partition(":v")
                version = int(raw_version or str(item.get("version") or "1").removeprefix("v"))
                restored = self.config_profiles.restore_version(pid, version)
                restored_configuration_versions.append((pid, version))
                refs = self.references(resource_id)
                affected_profiles = [str(ref.get("profile_id") or "") for ref in refs if ref.get("resource_type") == "target_profile" and ref.get("profile_id")]
                if affected_profiles:
                    self.domain_profiles.clear_unavailable(affected_profiles)
        except Exception:
            # Restore is one profile-level transaction.  If any owned config
            # snapshot cannot be restored, put every already-restored artifact
            # back into the archive so the catalog never exposes a partial
            # runnable profile.
            for config_profile_id, config_version in reversed(restored_configuration_versions):
                try:
                    self.config_profiles.archive_version(config_profile_id, config_version)
                except Exception:
                    self.logger.exception(
                        "runtime_resource_restore_config_rollback_failed profile=%s version=%s",
                        config_profile_id,
                        config_version,
                    )
            if restored_profile:
                try:
                    self.domain_profiles.archive(str(item.get("profile_id") or ""))
                except Exception:
                    self.logger.exception("runtime_resource_restore_profile_rollback_failed resource_id=%s", resource_id)
            raise
        restore_timestamp = _utc_now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE runtime_resources SET archived=0, active=0, integrity_status='warning', last_archived_at=COALESCE(archived_at,last_archived_at), archived_at=NULL, updated_at=? WHERE resource_id=? OR profile_id=?",
                (restore_timestamp, resource_id, str(item.get("profile_id") or "")),
            )
            for configuration in restored_configurations:
                conn.execute(
                    "UPDATE runtime_resources SET archived=0, active=0, integrity_status='warning', last_archived_at=COALESCE(archived_at,last_archived_at), archived_at=NULL, updated_at=? "
                    "WHERE profile_id=? AND resource_type IN ('configuration','effective_config')",
                    (restore_timestamp, str(configuration.get("profile_id") or "")),
                )
        self.scan(force=True)
        result = self._get(resource_id)
        self._record_audit("restore", resource_id, actor, before, {"resource": result, "active": False})
        configuration_ids = [str(configuration.get("resource_id") or "") for configuration in restored_configurations]
        event = self._bump_revision(action="restore", resource_ids=[resource_id, *configuration_ids], profile_ids=[str(item.get("profile_id") or ""), *affected_profiles], resource_types=[str(item.get("resource_type") or ""), *("configuration" for _ in configuration_ids)])
        return {"resource": result, "restored": True, "active": False, "profile": restored, **event, "affected_profile_ids": affected_profiles, "restored_configuration_ids": configuration_ids}

    def _optimization_references(self, profile_id: str, *, exclude_feedback_job_id: str | None = None) -> list[dict[str, Any]]:
        """Find active feedback-optimization jobs in the shared runtime DB."""
        try:
            with self._connect() as conn:
                placeholders = ",".join("?" for _ in ACTIVE_TASK_STATUSES)
                rows = conn.execute(
                    f"SELECT job_id,status,base_profile_id,candidate_profile_id FROM feedback_optimization_jobs WHERE status IN ({placeholders}) AND (base_profile_id=? OR candidate_profile_id=?)",
                    [*sorted(ACTIVE_TASK_STATUSES), str(profile_id), str(profile_id)],
                ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [
            {"job_id": str(row["job_id"]), "status": str(row["status"]), "kind": "反馈优化"}
            for row in rows
            if not self._is_feedback_apply_self(
                {"job_id": str(row["job_id"]), "status": str(row["status"]), "kind": "反馈优化"},
                exclude_feedback_job_id,
            )
        ]

    def _domain_build_references(self) -> list[dict[str, Any]]:
        tasks = getattr(self.domain_profiles, "_tasks", {}) or {}
        active_states = {"queued", "running", "cancelling", "validating", "building"}
        result = []
        for job_id, task in tasks.items():
            state = str(getattr(task, "state", "") or "").lower()
            if state in active_states:
                result.append({"job_id": str(job_id), "status": state, "kind": "目标域构建"})
        return result

    def _artifact_listing(self, source: Path) -> tuple[list[dict[str, Any]], list[str]]:
        files: list[dict[str, Any]] = []
        symlinks: list[str] = []
        for path in source.rglob("*"):
            if _is_reparse_or_symlink(path):
                symlinks.append(path.name)
                continue
            if path.is_file():
                try:
                    files.append({"name": path.relative_to(source).as_posix(), "size_bytes": int(path.stat().st_size)})
                except OSError as exc:
                    raise ApiError("RESOURCE_FILE_STAT_FAILED", "资源文件状态无法读取，已阻止删除。", 409) from exc
        return files, symlinks

    def _owned_resource_ids(self, item: dict[str, Any], dependencies: list[dict[str, Any]], source: Path) -> list[str]:
        owned = [str(item["resource_id"])]
        source = source.resolve()
        for child in dependencies:
            ref = str(child.get("artifact_ref") or "")
            is_owned = False
            # Target-profile children are registered with the owning profile
            # id.  Prefer that ownership signal because archived artifacts no
            # longer resolve under the live ``project_root/artifacts`` path;
            # relying only on ``artifact_ref`` would leave dangling child
            # rows after a successful physical deletion.
            if str(child.get("profile_id") or "") == str(item.get("profile_id") or "") and child.get("resource_type") not in {"base_model", "model_weight", "configuration"}:
                is_owned = True
            elif ref:
                try:
                    candidate = self._safe_path(ref).resolve()
                    is_owned = _within(candidate, source)
                except ApiError:
                    is_owned = False
            if is_owned:
                owned.append(str(child["resource_id"]))
        return list(dict.fromkeys(owned))

    def _history_reference_count(self, profile_id: str) -> int:
        target = str(profile_id or "").strip()
        if not target:
            return 0

        def contains_target(value: Any) -> bool:
            if isinstance(value, dict):
                return any(contains_target(child) for child in value.values())
            if isinstance(value, (list, tuple)):
                return any(contains_target(child) for child in value)
            return isinstance(value, str) and value == target

        with self._connect() as conn:
            try:
                rows = conn.execute(
                    "SELECT metadata_json FROM detection_results WHERE metadata_json LIKE ?",
                    (f"%{target}%",),
                ).fetchall()
            except sqlite3.Error:
                return 0
        count = 0
        for row in rows:
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            count += int(contains_target(metadata))
        return count

    def _delete_preview(
        self,
        resource_id: str,
        *,
        issue_token: bool,
        exclude_feedback_job_id: str | None = None,
    ) -> dict[str, Any]:
        item = self._get(resource_id)
        refs = self.references(resource_id)
        dependencies = self.dependencies(resource_id)
        reasons: list[str] = []
        profile_id = str(item.get("profile_id") or "")
        kind = str(item.get("resource_type") or "")
        builtin_profile = self._is_builtin_profile(item)
        source: Path | None = None
        sources: list[Path] = []
        # The generated built-in binding is metadata for inference, not an
        # owned user configuration file.  Archiving/deleting the public memory
        # bank must never remove shared configuration or model resources.
        exclusive_configurations = [] if builtin_profile else self._exclusive_configuration_dependencies(item, dependencies)
        files: list[dict[str, Any]] = []
        symlinks: list[str] = []
        if kind not in {"target_profile", "configuration"}:
            reasons.append("当前仅支持删除已归档目标域 profile 或配置方案；基础模型和共享资源受保护")
        if not item.get("archived"):
            reasons.append("资源尚未归档")
        requires_profile_transition = self._is_current_binding(item)
        if item.get("frozen") and not item.get("archived"):
            reasons.append("资源已冻结，必须先归档")
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        if not builtin_profile and any(bool(metadata.get(key)) for key in ("demo_required", "required_for_demo", "protected", "delete_protected")):
            reasons.append("演示或系统保护资源禁止删除")
        try:
            if kind == "target_profile" and profile_id:
                if builtin_profile:
                    source = self._safe_path(str(item.get("artifact_ref") or ""))
                    if not source.is_dir() or not _within(source, self._builtin_archive_root()):
                        raise ApiError("MEMORY_BANK_ARCHIVE_NOT_FOUND", "内置记忆库归档文件不存在。", 404)
                else:
                    source = self.domain_profiles.archived_profile_directory(profile_id)
                    # A preview must prove that the frozen metadata and all
                    # resource hashes are readable before a delete token is issued.
                    # This blocks incomplete/corrupt archives instead of turning
                    # an integrity problem into a physical deletion.
                    self.domain_profiles._read_target_metadata(source, verify_artifacts=True)
                files, symlinks = self._artifact_listing(source)
                sources.append(source)
                if symlinks:
                    reasons.append("受控目录中存在 symbolic link，已阻止删除")
                for configuration in exclusive_configurations:
                    if not configuration.get("archived"):
                        reasons.append("profile 专属配置快照尚未归档")
                        continue
                    config_profile_id, _, raw_version = str(configuration.get("profile_id") or "").partition(":v")
                    config_version = int(raw_version or str(configuration.get("version") or "1").removeprefix("v"))
                    config_source = self.config_profiles.archived_version_directory(config_profile_id, config_version)
                    required = [config_source / "profile.yaml", config_source / "metadata.json"]
                    if not all(path.is_file() for path in required):
                        reasons.append("profile 专属归档配置快照不完整")
                    config_files, config_symlinks = self._artifact_listing(config_source)
                    files.extend({**entry, "name": f"configuration/{entry['name']}"} for entry in config_files)
                    symlinks.extend(config_symlinks)
                    sources.append(config_source)
            elif kind == "configuration" and profile_id:
                pid, _, raw_version = profile_id.partition(":v")
                version = int(raw_version or str(item.get("version") or "1").removeprefix("v"))
                source = self.config_profiles.archived_version_directory(pid, version)
                sources.append(source)
                required = [source / "profile.yaml", source / "metadata.json"]
                if not all(path.is_file() for path in required):
                    reasons.append("归档配置文件不完整")
                files, symlinks = self._artifact_listing(source)
                if symlinks:
                    reasons.append("受控目录中存在 symbolic link，已阻止删除")
        except ApiError as exc:
            reasons.append(exc.message)
        running = self._hard_blockers(item, exclude_feedback_job_id=exclude_feedback_job_id)
        if running:
            reasons.append(f"被正在运行的 {len(running)} 个任务引用")
        optimization_refs = (
            self._optimization_references(profile_id, exclude_feedback_job_id=exclude_feedback_job_id)
            if kind == "target_profile" and profile_id
            else []
        )
        history_reference_count = self._history_reference_count(profile_id)
        owned_ids = self._owned_resource_ids(item, dependencies, source) if source is not None else [str(item["resource_id"])]
        owned_ids.extend(self._configuration_resource_ids(exclusive_configurations))
        owned_ids = list(dict.fromkeys(value for value in owned_ids if value))
        shared_resource_ids: list[str] = []
        for child in dependencies:
            if str(child.get("resource_id")) not in owned_ids:
                continue
            child_refs = [ref for ref in self.references(str(child["resource_id"])) if str(ref.get("resource_id")) != str(item["resource_id"])]
            if child_refs:
                shared_resource_ids.append(str(child["resource_id"]))
        if shared_resource_ids:
            owned_ids = [value for value in owned_ids if value not in set(shared_resource_ids)]
        allowed = not reasons
        token = secrets.token_urlsafe(24) if issue_token and allowed else ""
        expires_at = datetime.fromtimestamp(time.time() + 600, tz=timezone.utc).isoformat(timespec="seconds") if token else ""
        if token and source is not None:
            self._delete_previews[token] = {
                "resource_id": str(resource_id),
                "expires_at": time.time() + 600,
                "source": str(source),
                "sources": [str(path) for path in sources],
                "owned_resource_ids": owned_ids,
                "owned_configuration_resource_ids": [str(configuration.get("resource_id") or "") for configuration in exclusive_configurations],
                "owned_configuration_profiles": [str(configuration.get("profile_id") or "") for configuration in exclusive_configurations],
                "resource_type": kind,
                "requires_profile_transition": requires_profile_transition,
            }
        public_item = {key: value for key, value in item.items() if key not in {"artifact_ref", "metadata"}}
        return {
            "resource": public_item,
            "resource_id": str(resource_id),
            "resource_type": item.get("resource_type"),
            "references": refs,
            "dependencies": [{key: child.get(key) for key in ("resource_id", "resource_type", "display_name", "version", "profile_id", "integrity_status", "required")} for child in dependencies],
            "owned_resource_count": len(owned_ids),
            "owned_configuration_count": len(exclusive_configurations),
            "file_count": len(files),
            "total_bytes": sum(int(file.get("size_bytes") or 0) for file in files),
            "optimization_references": optimization_refs,
            "running_task_references": running,
            "history_reference_count": history_reference_count,
            "historical_profile_references": [{"resource_id": ref.get("resource_id"), "profile_id": ref.get("profile_id"), "display_name": ref.get("display_name"), "active": bool(ref.get("active"))} for ref in refs],
            "requires_profile_transition": requires_profile_transition,
            "shared_resource_ids": shared_resource_ids,
            "allowed": allowed,
            "reasons": reasons or ["依赖检查通过，可以安全删除该记忆库。"],
            "preview_token": token,
            "expires_at": expires_at,
        }

    def delete_preview(self, resource_id: str, *, exclude_feedback_job_id: str | None = None) -> dict[str, Any]:
        if self.test_lock is not None:
            self.test_lock.require_write_allowed("resource_lifecycle")
        with self._delete_lock:
            self._ensure_resource_mutation_idle(exclude_feedback_job_id=exclude_feedback_job_id)
            return self._delete_preview(
                resource_id,
                issue_token=True,
                exclude_feedback_job_id=exclude_feedback_job_id,
            )

    def delete(self, resource_id: str, preview_token: str, actor: str = "local_operator", *,
               acknowledged: bool = False, delete_artifacts: bool = True, reason: str = "",
               current_profile_action: str = "cancel", replacement_resource_id: str | None = None,
               exclude_feedback_job_id: str | None = None) -> dict[str, Any]:
        with self._delete_lock:
            if not acknowledged:
                raise ApiError("CONFIRMATION_REQUIRED", "请先核对删除预览并勾选确认。", 422)
            self._ensure_resource_mutation_idle(exclude_feedback_job_id=exclude_feedback_job_id)
            saved = self._delete_previews.get(str(preview_token or ""))
            if not saved or saved.get("resource_id") != str(resource_id) or float(saved.get("expires_at") or 0) <= time.time():
                raise ApiError("RESOURCE_DELETE_PREVIEW_EXPIRED", "删除预览已过期，请重新生成预览。", 409)
            if not delete_artifacts:
                raise ApiError("RESOURCE_DELETE_ARTIFACTS_REQUIRED", "安全删除必须同时处理受控专属资源。", 422)
            item = self._get(resource_id)
            transition = self._transition_current_profile(item, current_profile_action, replacement_resource_id, actor)
            try:
                preview = self._delete_preview(
                    resource_id,
                    issue_token=False,
                    exclude_feedback_job_id=exclude_feedback_job_id,
                )
                if not preview.get("allowed"):
                    raise ApiError("RESOURCE_DELETE_BLOCKED", "该资源当前不能物理删除。", 409, {"reasons": preview.get("reasons") or []})
                kind = str(item.get("resource_type") or "")
                if kind == "target_profile":
                    source = (
                        self._safe_path(str(item.get("artifact_ref") or ""))
                        if self._is_builtin_profile(item)
                        else self.domain_profiles.archived_profile_directory(str(item.get("profile_id") or ""))
                    )
                elif kind == "configuration":
                    pid, _, raw_version = str(item.get("profile_id") or "").partition(":v")
                    version = int(raw_version or str(item.get("version") or "1").removeprefix("v"))
                    source = self.config_profiles.archived_version_directory(pid, version)
                else:
                    raise ApiError("RESOURCE_DELETE_UNSUPPORTED", "该资源类型不支持物理删除。", 422)
                saved_sources = [Path(value).resolve() for value in list(saved.get("sources") or [saved.get("source")]) if value]
                if str(source.resolve()) not in {str(path) for path in saved_sources} or any(not path.exists() for path in saved_sources):
                    raise ApiError("RESOURCE_DELETE_PREVIEW_STALE", "资源位置已变化，请重新生成删除预览。", 409)
            except Exception:
                if transition.get("changed"):
                    self._restore_runtime_snapshot(dict(transition.get("before") or {}))
                raise
            operation_id = f"delete_{uuid.uuid4().hex[:16]}"
            trash_root = (self.settings.runtime_root / "resource_trash").resolve()
            trash_root.mkdir(parents=True, exist_ok=True)
            isolated = trash_root / operation_id
            before = {"resource": item, "preview": preview, "operation_id": operation_id, "reason": str(reason or "")[:240]}
            moved_records: list[tuple[Path, Path]] = []
            db_committed = False
            former_refs = list(preview.get("historical_profile_references") or [])
            affected_profiles = [str(ref.get("profile_id") or "") for ref in former_refs if ref.get("profile_id")]
            try:
                files_before: list[dict[str, Any]] = []
                for source_path in saved_sources:
                    source_files, source_symlinks = self._artifact_listing(source_path)
                    if source_symlinks:
                        raise ApiError("RESOURCE_DELETE_SYMLINK", "受控目录中存在 symbolic link，已阻止删除。", 409)
                    files_before.extend(source_files)
                isolated.mkdir(parents=True, exist_ok=False)
                files_after: list[dict[str, Any]] = []
                for index, source_path in enumerate(saved_sources):
                    moved_root = isolated / f"{index:02d}_{source_path.name}"
                    shutil.move(str(source_path), str(moved_root))
                    moved_records.append((source_path, moved_root))
                    moved_files, symlinks_after = self._artifact_listing(moved_root)
                    if symlinks_after:
                        raise ApiError("RESOURCE_DELETE_ISOLATION_FAILED", "资源隔离目录出现 symbolic link，未提交删除。", 409)
                    files_after.extend(moved_files)
                if len(files_before) != len(files_after) or sum(int(x.get("size_bytes") or 0) for x in files_before) != sum(int(x.get("size_bytes") or 0) for x in files_after):
                    raise ApiError("RESOURCE_DELETE_ISOLATION_FAILED", "资源隔离校验失败，未提交删除。", 409)
                owned_ids = list(saved.get("owned_resource_ids") or [str(resource_id)])
                placeholders = ",".join("?" for _ in owned_ids)
                with self._connect() as conn:
                    tombstone = {
                        "resource_id": str(resource_id),
                        "profile_id": item.get("profile_id"),
                        "source_type": item.get("source_type"),
                        "historical_references": former_refs,
                        "history_reference_count": int(preview.get("history_reference_count") or 0),
                        "shared_resource_ids": list(preview.get("shared_resource_ids") or []),
                        "owned_configuration_profiles": list(saved.get("owned_configuration_profiles") or []),
                    }
                    conn.execute(
                        "INSERT INTO runtime_resource_tombstones(tombstone_id,resource_id,profile_id,resource_type,display_name,version,dataset,category,method_mode,sha256,file_size,deleted_at,delete_operation_id,deletion_reason,former_profile_references_json,former_task_reference_count,metadata_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (uuid.uuid4().hex, str(resource_id), item.get("profile_id"), kind, str(item.get("display_name") or "资源"), str(item.get("version") or ""), item.get("dataset"), item.get("category"), item.get("method_mode"), item.get("sha256"), item.get("file_size"), _utc_now(), operation_id, str(reason or "")[:240], _json(former_refs), int(preview.get("history_reference_count") or 0), _json(tombstone)),
                    )
                    feedback_release = {"affected": 0, "released": 0, "still_protected": 0, "active_jobs": 0}
                    if kind == "target_profile" and self.feedback_candidates is not None:
                        feedback_release = self.feedback_candidates.mark_source_memory_bank_deleted(
                            profile_id=str(item.get("profile_id") or ""),
                            resource_id=str(resource_id),
                            actor=actor,
                            connection=conn,
                        )
                    conn.execute("DELETE FROM runtime_resource_dependencies WHERE parent_resource_id IN (" + placeholders + ") OR child_resource_id IN (" + placeholders + ")", tuple(owned_ids) + tuple(owned_ids))
                    conn.execute("DELETE FROM runtime_resources WHERE resource_id IN (" + placeholders + ")", tuple(owned_ids))
                db_committed = True
                try:
                    if kind == "target_profile":
                        if not self._is_builtin_profile(item):
                            self.domain_profiles.forget_archived(str(item.get("profile_id") or ""), operation_id, reason=str(reason or "")[:240])
                        for config_binding in list(saved.get("owned_configuration_profiles") or []):
                            config_profile_id, _, raw_version = str(config_binding).partition(":v")
                            self.config_profiles.forget_archived(config_profile_id, int(raw_version or 1), operation_id, reason=str(reason or "")[:240])
                    else:
                        pid, _, raw_version = str(item.get("profile_id") or "").partition(":v")
                        version = int(raw_version or str(item.get("version") or "1").removeprefix("v"))
                        self.config_profiles.forget_archived(pid, version, operation_id, reason=str(reason or "")[:240])
                        if affected_profiles:
                            self.domain_profiles.mark_unavailable(affected_profiles, "resource_missing", resource_id)
                except Exception as exc:
                    self.logger.warning("runtime_resource_delete_tombstone_pending: %s", exc)
                released = sum(int(x.get("size_bytes") or 0) for x in files_after)
                cleanup_pending = False
                try:
                    shutil.rmtree(isolated)
                except OSError:
                    cleanup_pending = True
                after = {"operation_id": operation_id, "state": "deleted", "released_bytes": released, "cleanup_pending": cleanup_pending}
                # The resource and database transaction are already committed
                # here.  An audit-write outage must not report a failed delete
                # and attempt to restore only the files while leaving the
                # resource index deleted.
                try:
                    self._record_audit("delete", resource_id, actor, before, after)
                except Exception:
                    self.logger.exception("runtime_resource_delete_audit_failed operation_id=%s", operation_id)
                self._delete_previews.pop(str(preview_token), None)
                owned_configuration_ids = list(saved.get("owned_configuration_resource_ids") or [])
                event_profile_ids = (
                    [str(item.get("profile_id") or "")]
                    if kind == "target_profile"
                    else []
                ) + affected_profiles
                event = self._bump_revision(action="delete", resource_ids=[str(resource_id), *owned_configuration_ids], profile_ids=event_profile_ids, resource_types=[kind, *("configuration" for _ in owned_configuration_ids)], current_profile_changed=bool(transition.get("changed")))
                statistics_revision = None
                if self.history_management is not None:
                    try:
                        statistics_revision = self.history_management.resource_dependency_changed(
                            "memory_bank_deleted_history_protection_recalculated"
                        )
                    except Exception:
                        self.logger.exception("history_protection_invalidation_failed operation_id=%s", operation_id)
                return {"deleted": True, "physical_delete": True, "operation_id": operation_id, "released_bytes": released, "cleanup_pending": cleanup_pending, "resource_id": str(resource_id), "affected_profile_ids": affected_profiles, "deleted_configuration_ids": owned_configuration_ids, "feedback_dependency_release": feedback_release, "statistics_revision": statistics_revision, **event}
            except Exception as exc:
                if moved_records and not db_committed:
                    try:
                        for restore_target, moved_root in reversed(moved_records):
                            if moved_root.exists() and not restore_target.exists():
                                shutil.move(str(moved_root), str(restore_target))
                        if isolated.exists():
                            shutil.rmtree(isolated, ignore_errors=True)
                    except OSError:
                        self.logger.exception("runtime_resource_delete_restore_failed")
                try:
                    self._record_audit("delete_failed", resource_id, actor, before, {"state": "archived", "error": str(exc)[:240], "operation_id": operation_id})
                except Exception:
                    self.logger.exception("runtime_resource_delete_failure_audit_failed operation_id=%s", operation_id)
                if isinstance(exc, ApiError):
                    if transition.get("changed") and not db_committed:
                        self._restore_runtime_snapshot(dict(transition.get("before") or {}))
                    raise
                if transition.get("changed") and not db_committed:
                    self._restore_runtime_snapshot(dict(transition.get("before") or {}))
                raise ApiError("RESOURCE_DELETE_FAILED", "安全删除失败，资源已恢复为归档状态。", 500) from exc

    def replace_profile(
        self,
        old_resource_id: str,
        new_resource_id: str,
        *,
        actor: str = "operator",
        feedback_job_id: str | None = None,
    ) -> dict[str, Any]:
        """Safely replace one active target profile with another.

        Replacement is intentionally a lifecycle transaction, not an in-place
        edit.  The normal profile-apply path first switches and warms the
        Runtime binding (including integrity validation), then the old profile
        and its exclusive configuration snapshots are physically deleted
        through the normal preview/token path.  A feedback
        job may exclude only itself from the global activity check; all other
        detection/build/update activity and registry leases remain hard
        blockers.
        """
        old_id = str(old_resource_id or "").strip()
        new_id = str(new_resource_id or "").strip()
        if not old_id or not new_id or old_id == new_id:
            raise ApiError("REPLACEMENT_PROFILE_REQUIRED", "替换操作必须选择与原版本不同的新目标域版本。", 422)
        if self.test_lock is not None:
            self.test_lock.require_write_allowed("resource_lifecycle")
        with self._apply_lock:
            self._ensure_resource_mutation_idle(exclude_feedback_job_id=feedback_job_id)
            if self.registry is not None:
                try:
                    runtime_status = self.registry.status()
                    active_count = sum(
                        int(item.get("active_count") or 0)
                        for item in runtime_status.get("runtimes") or []
                    )
                except (AttributeError, TypeError, ValueError):
                    active_count = 0
                if active_count:
                    raise ApiError(
                        "RUNTIME_IN_USE",
                        "当前仍有检测正在使用原版本，完成后才能执行替换。",
                        409,
                        {"active_count": active_count},
                    )
            old_item = self._get(old_id)
            new_item = self._get(new_id)
            if str(old_item.get("resource_type") or "") != "target_profile" or str(new_item.get("resource_type") or "") != "target_profile":
                raise ApiError("RESOURCE_NOT_PROFILE", "替换对象必须是完整目标域 Profile。", 422)
            if not self._is_current_binding(old_item):
                raise ApiError("REPLACEMENT_BASE_NOT_CURRENT", "原版本已不是当前运行方案，未执行替换。", 409)
            if not bool(new_item.get("frozen")) or bool(new_item.get("archived")):
                raise ApiError("PROFILE_NOT_APPLICABLE", "新版本必须先通过完整性验证并冻结，才能替换原版本。", 409)

            # Switch and warm up the candidate first.  If either runtime load
            # or warmup fails, the old profile is still live and can be
            # restored before any archive/delete operation is attempted.
            try:
                self.apply_profile(new_id, actor=actor, feedback_job_id=feedback_job_id)
                warmup = getattr(self, "warmup", None)
                if warmup is not None:
                    deadline = time.monotonic() + 120.0
                    while True:
                        warmup_state = dict(warmup.status() or {})
                        state = str(warmup_state.get("state") or "").lower()
                        if state == "ready":
                            break
                        if state in {"failed", "not_ready", "deactivating"} or time.monotonic() >= deadline:
                            raise ApiError(
                                "REPLACEMENT_RUNTIME_NOT_READY",
                                "新版本运行资源未完成预热，原版本未删除。",
                                409,
                                {"warmup": warmup_state},
                            )
                        time.sleep(0.1)
            except Exception:
                # apply_profile itself rolls back partial runtime writes.  If
                # warmup failed after the new binding was written, explicitly
                # restore the old live profile before surfacing the failure.
                try:
                    if str(self._runtime_state().get("memory_bank_id") or "") == str(new_item.get("profile_id") or ""):
                        self.apply_profile(old_id, actor=actor, feedback_job_id=feedback_job_id)
                except Exception:
                    self.logger.exception("replacement_runtime_rollback_failed old=%s new=%s", old_id, new_id)
                raise

            try:
                archived = self.archive(
                    old_id,
                    actor=actor,
                    current_profile_action="cancel",
                    exclude_feedback_job_id=feedback_job_id,
                )
            except Exception:
                # The candidate is warm, but retiring the old graph failed.
                # Restore the old runtime before reporting failure so a
                # filesystem/catalog problem cannot leave the system running
                # the new profile while claiming replacement did not happen.
                try:
                    if str(self._runtime_state().get("memory_bank_id") or "") == str(new_item.get("profile_id") or ""):
                        self.apply_profile(old_id, actor=actor, feedback_job_id=feedback_job_id)
                except Exception:
                    self.logger.exception("replacement_archive_rollback_failed old=%s new=%s", old_id, new_id)
                raise
            result: dict[str, Any] = {
                "replaced": True,
                "old_resource_id": old_id,
                "new_resource_id": new_id,
                "old_profile_id": str(old_item.get("profile_id") or ""),
                "new_profile_id": str(new_item.get("profile_id") or ""),
                "archived": True,
                "deleted": False,
                "archive": archived,
            }
            try:
                preview = self.delete_preview(old_id, exclude_feedback_job_id=feedback_job_id)
                if not preview.get("allowed"):
                    raise ApiError(
                        "REPLACEMENT_CLEANUP_BLOCKED",
                        "新版本已切换，但原版本清理仍被保护条件阻止；原版本已归档，可在高级维护中重试删除。",
                        409,
                        {"reasons": preview.get("reasons") or []},
                    )
                deleted = self.delete(
                    old_id,
                    preview_token=str(preview.get("preview_token") or ""),
                    actor=actor,
                    acknowledged=True,
                    reason="feedback_replace_original",
                    exclude_feedback_job_id=feedback_job_id,
                )
                result.update({"deleted": True, "delete": deleted, "cleanup_status": "completed"})
            except Exception as exc:
                # The new runtime is already valid and active; never switch it
                # back merely because filesystem cleanup needs maintenance.
                # The old profile remains archived (and therefore absent from
                # ordinary runnable lists), with a durable warning for retry.
                result.update({
                    "cleanup_status": "archived_pending_delete",
                    "cleanup_warning": str(getattr(exc, "message", exc)),
                })
                self.logger.warning("replacement_cleanup_pending old=%s new=%s error=%s", old_id, new_id, exc)
            return result

    def batch_archive_preview(self, resource_ids: list[str], *, resource_type: str | None = None) -> dict[str, Any]:
        self._ensure_resource_mutation_idle()
        items: list[dict[str, Any]] = []
        for resource_id in dict.fromkeys(str(value) for value in resource_ids if str(value)):
            try:
                item = self._get(resource_id)
                if resource_type and str(item.get("resource_type")) != resource_type:
                    raise ApiError("RESOURCE_TYPE_MISMATCH", "资源类型与当前操作不匹配。", 422)
                blockers = self._hard_blockers(item)
                current = self._is_current_binding(item)
                state = "already_archived" if item.get("archived") else "blocked" if blockers else "needs_profile_transition" if current else "ready"
                items.append({
                    "resource_id": resource_id,
                    "display_name": item.get("display_name"),
                    "resource_type": item.get("resource_type"),
                    "state": state,
                    "can_archive": state in {"ready", "needs_profile_transition"},
                    "requires_profile_transition": current,
                    "blockers": blockers,
                    "historical_references": self.references(resource_id),
                })
            except ApiError as exc:
                items.append({"resource_id": resource_id, "state": "failed", "can_archive": False, "error": {"code": exc.code, "message": exc.message}})
        return {
            "selected_count": len(items),
            "ready_count": sum(bool(item.get("can_archive")) for item in items),
            "blocked_count": sum(not bool(item.get("can_archive")) for item in items),
            "items": items,
        }

    def _record_batch_operation(self, operation_type: str, results: list[dict[str, Any]], actor: str) -> dict[str, Any]:
        operation_id = f"batch_{uuid.uuid4().hex[:20]}"
        success_count = sum(bool(item.get("success")) for item in results)
        skipped_count = sum(str(item.get("state") or "") == "skipped" for item in results)
        failed_count = len(results) - success_count - skipped_count
        released = sum(int(item.get("released_bytes") or 0) for item in results if item.get("success"))
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO runtime_resource_batch_operations(operation_id,operation_type,selected_count,success_count,skipped_count,failed_count,released_bytes,details_json,actor,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (operation_id, operation_type, len(results), success_count, skipped_count, failed_count, released, _json(results), str(actor or "local_operator")[:80], _utc_now()),
            )
        return {"operation_id": operation_id, "selected_count": len(results), "success_count": success_count, "skipped_count": skipped_count, "failed_count": failed_count, "released_bytes": released, "results": results}

    def batch_archive(self, resource_ids: list[str], *, actor: str = "local_operator",
                      current_profile_action: str = "cancel", replacement_resource_id: str | None = None,
                      resource_type: str | None = None) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for resource_id in dict.fromkeys(str(value) for value in resource_ids if str(value)):
            try:
                item = self._get(resource_id)
                if resource_type and str(item.get("resource_type")) != resource_type:
                    raise ApiError("RESOURCE_TYPE_MISMATCH", "资源类型与当前操作不匹配。", 422)
                if item.get("archived"):
                    results.append({"resource_id": resource_id, "success": False, "state": "skipped", "message": "资源已经归档。"})
                    continue
                archived = self.archive(resource_id, actor, current_profile_action=current_profile_action, replacement_resource_id=replacement_resource_id)
                results.append({"resource_id": resource_id, "success": True, "state": "archived", "revision": archived.get("revision")})
            except ApiError as exc:
                state = "skipped" if exc.code in {"RESOURCE_IN_USE", "CURRENT_PROFILE_ACTION_REQUIRED"} else "failed"
                results.append({"resource_id": resource_id, "success": False, "state": state, "error": {"code": exc.code, "message": exc.message}})
        return self._record_batch_operation("batch_archive", results, actor)

    def batch_restore(self, resource_ids: list[str], *, actor: str = "local_operator",
                      resource_type: str | None = None) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for resource_id in dict.fromkeys(str(value) for value in resource_ids if str(value)):
            try:
                item = self._get(resource_id)
                if resource_type and str(item.get("resource_type")) != resource_type:
                    raise ApiError("RESOURCE_TYPE_MISMATCH", "资源类型与当前操作不匹配。", 422)
                if not item.get("archived"):
                    results.append({"resource_id": resource_id, "success": False, "state": "skipped", "message": "资源未归档。"})
                    continue
                restored = self.restore(resource_id, actor)
                results.append({"resource_id": resource_id, "success": True, "state": "restored", "revision": restored.get("revision")})
            except ApiError as exc:
                results.append({"resource_id": resource_id, "success": False, "state": "failed", "error": {"code": exc.code, "message": exc.message}})
        return self._record_batch_operation("batch_restore", results, actor)

    def batch_delete_preview(self, resource_ids: list[str], *, resource_type: str | None = None) -> dict[str, Any]:
        self._ensure_resource_mutation_idle()
        items: list[dict[str, Any]] = []
        plan_items: dict[str, str] = {}
        for resource_id in dict.fromkeys(str(value) for value in resource_ids if str(value)):
            try:
                item = self._get(resource_id)
                if resource_type and str(item.get("resource_type")) != resource_type:
                    raise ApiError("RESOURCE_TYPE_MISMATCH", "资源类型与当前操作不匹配。", 422)
                preview = self._delete_preview(resource_id, issue_token=True)
                state = "ready" if preview.get("allowed") else "blocked"
                if preview.get("requires_profile_transition") and preview.get("allowed"):
                    state = "needs_profile_transition"
                items.append({**preview, "state": state})
                if preview.get("preview_token"):
                    plan_items[resource_id] = str(preview["preview_token"])
            except ApiError as exc:
                items.append({"resource_id": resource_id, "state": "failed", "allowed": False, "error": {"code": exc.code, "message": exc.message}})
        token = secrets.token_urlsafe(32) if plan_items else ""
        if token:
            self._batch_delete_previews[token] = {"items": plan_items, "expires_at": time.time() + 600, "resource_type": resource_type}
        return {
            "selected_count": len(items),
            "deletable_count": sum(bool(item.get("allowed")) for item in items),
            "blocked_count": sum(not bool(item.get("allowed")) for item in items),
            "file_count": sum(int(item.get("file_count") or 0) for item in items if item.get("allowed")),
            "total_bytes": sum(int(item.get("total_bytes") or 0) for item in items if item.get("allowed")),
            "items": items,
            "preview_token": token,
        }

    def batch_delete(self, resource_ids: list[str], *, preview_token: str, acknowledged: bool = False,
                     actor: str = "local_operator", reason: str = "", current_profile_action: str = "cancel",
                     replacement_resource_id: str | None = None, resource_type: str | None = None) -> dict[str, Any]:
        if self.test_lock is not None:
            self.test_lock.require_write_allowed("resource_lifecycle")
        if not acknowledged:
            raise ApiError("CONFIRMATION_REQUIRED", "请先核对删除预览并勾选确认。", 422)
        plan = self._batch_delete_previews.pop(str(preview_token or ""), None)
        if not plan or float(plan.get("expires_at") or 0) <= time.time():
            raise ApiError("RESOURCE_DELETE_PREVIEW_EXPIRED", "批量删除预览已过期，请重新生成。", 409)
        if resource_type and plan.get("resource_type") not in {None, resource_type}:
            raise ApiError("RESOURCE_DELETE_PREVIEW_STALE", "批量删除资源类型已变化，请重新预览。", 409)
        planned = dict(plan.get("items") or {})
        results: list[dict[str, Any]] = []
        for resource_id in dict.fromkeys(str(value) for value in resource_ids if str(value)):
            single_token = str(planned.get(resource_id) or "")
            if not single_token:
                results.append({"resource_id": resource_id, "success": False, "state": "skipped", "message": "该资源未通过删除预览。"})
                continue
            try:
                result = self.delete(
                    resource_id,
                    preview_token=single_token,
                    actor=actor,
                    acknowledged=True,
                    delete_artifacts=True,
                    reason=reason,
                    current_profile_action=current_profile_action,
                    replacement_resource_id=replacement_resource_id,
                )
                results.append({"resource_id": resource_id, "success": True, "state": "deleted", "released_bytes": int(result.get("released_bytes") or 0), "operation_id": result.get("operation_id")})
            except ApiError as exc:
                state = "skipped" if exc.code in {"RESOURCE_IN_USE", "CURRENT_PROFILE_ACTION_REQUIRED", "RESOURCE_DELETE_BLOCKED"} else "failed"
                results.append({"resource_id": resource_id, "success": False, "state": state, "error": {"code": exc.code, "message": exc.message}})
        return self._record_batch_operation("batch_delete", results, actor)

    def _target_domain_rows(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runtime_resources WHERE resource_type NOT IN ('base_model','model_weight') "
                "ORDER BY resource_type,resource_id"
            ).fetchall()
        return [self._row(row) for row in rows]

    def _target_domain_cleanup_paths(self) -> list[Path]:
        """Return only server-owned target-domain paths, never shared models/data."""
        paths: list[Path] = []
        for root in (self.settings.memory_banks_root.resolve(), self.settings.config_profiles_root.resolve()):
            root.mkdir(parents=True, exist_ok=True)
            paths.extend(sorted(root.iterdir(), key=lambda item: item.name.lower()))
        jobs_root = self.settings.jobs_root.resolve()
        for name in ("domain_drafts", "domain_video_samples"):
            candidate = jobs_root / name
            if candidate.exists():
                paths.append(candidate)
        paths.extend(sorted(jobs_root.glob("calibration_*"), key=lambda item: item.name.lower()))
        for candidate in (self.settings.memory_bank_registry_path.resolve(), self.settings.active_calibrations_path.resolve()):
            if candidate.exists():
                paths.append(candidate)
        unique: list[Path] = []
        seen: set[str] = set()
        allowed_roots = (
            self.settings.memory_banks_root.resolve(),
            self.settings.config_profiles_root.resolve(),
            self.settings.runtime_root.resolve(),
        )
        for path in paths:
            resolved = path.resolve()
            if str(resolved) in seen:
                continue
            if not any(_within(resolved, root) for root in allowed_roots):
                raise ApiError("RESOURCE_PATH_FORBIDDEN", "目标域清理计划包含非受控路径，已阻止操作。", 409)
            if _is_reparse_or_symlink(resolved):
                raise ApiError("RESOURCE_DELETE_SYMLINK", "目标域资源中存在符号链接或重解析点，已阻止清理。", 409)
            seen.add(str(resolved))
            unique.append(resolved)
        return unique

    def _cleanup_path_summary(self, paths: list[Path]) -> tuple[list[dict[str, Any]], int, int]:
        entries: list[dict[str, Any]] = []
        total_files = 0
        total_bytes = 0
        for path in paths:
            if path.is_file():
                size = int(path.stat().st_size)
                count = 1
            elif path.is_dir():
                files, symlinks = self._artifact_listing(path)
                if symlinks:
                    raise ApiError("RESOURCE_DELETE_SYMLINK", "目标域资源目录中存在符号链接，已阻止清理。", 409)
                count = len(files)
                size = sum(int(item.get("size_bytes") or 0) for item in files)
            else:
                continue
            entries.append({"ref": self._relative_ref(path), "file_count": count, "total_bytes": size})
            total_files += count
            total_bytes += size
        return entries, total_files, total_bytes

    @staticmethod
    def _target_domain_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        def count(*types: str) -> int:
            return sum(str(item.get("resource_type") or "") in types for item in rows)

        return {
            "profile_count": count("target_profile", "candidate_profile"),
            "target_profile_count": count("target_profile"),
            "candidate_profile_count": count("candidate_profile"),
            "normal_memory_bank_count": count("normal_memory_bank"),
            "defect_prototype_bank_count": count("defect_prototype_bank"),
            "configuration_count": count("configuration"),
            "effective_config_count": count("effective_config"),
            "threshold_count": count("image_threshold", "pixel_threshold"),
            "calibrator_count": count("lightweight_calibrator"),
            "coreset_count": count("coreset_index"),
            "fusion_parameter_count": count("fusion_parameters"),
        }

    def cleanup_summary(self) -> dict[str, Any]:
        rows = self._target_domain_rows()
        paths = self._target_domain_cleanup_paths()
        path_entries, file_count, total_bytes = self._cleanup_path_summary(paths)
        return {
            **self._target_domain_counts(rows),
            "resource_count": len(rows),
            "resource_ids": [str(item.get("resource_id") or "") for item in rows if item.get("resource_id")],
            "file_count": file_count,
            "total_bytes": total_bytes,
            "managed_path_count": len(path_entries),
            "current_profile": self.current_profile().get("resource"),
            "active_tasks": self._global_runtime_activity(),
        }

    def clear_all_preview(self) -> dict[str, Any]:
        with self._delete_lock:
            active = self._global_runtime_activity()
            rows = self._target_domain_rows()
            paths = self._target_domain_cleanup_paths()
            path_entries, file_count, total_bytes = self._cleanup_path_summary(paths)
            resource_ids = [str(item.get("resource_id") or "") for item in rows if item.get("resource_id")]
            profile_ids = [
                str(item.get("profile_id") or "")
                for item in rows
                if item.get("resource_type") in {"target_profile", "candidate_profile"} and item.get("profile_id")
            ]
            plan_payload = {
                "resource_ids": resource_ids,
                "profile_ids": profile_ids,
                "path_entries": path_entries,
                "file_count": file_count,
                "total_bytes": total_bytes,
                "current_profile_id": str((self.current_profile().get("resource") or {}).get("profile_id") or ""),
            }
            plan_hash = hashlib.sha256(_json(plan_payload).encode("utf-8")).hexdigest()
            plan_id = f"clear_{uuid.uuid4().hex[:20]}"
            preview_token = secrets.token_urlsafe(32)
            expires_at_epoch = time.time() + 900
            self._clear_all_plans[plan_id] = {
                **plan_payload,
                "plan_id": plan_id,
                "preview_token": preview_token,
                "plan_hash": plan_hash,
                "paths": [str(path) for path in paths],
                "expires_at": expires_at_epoch,
            }
            return {
                "plan_id": plan_id,
                "preview_token": preview_token,
                "created_at": _utc_now(),
                "expires_at": datetime.fromtimestamp(expires_at_epoch, tz=timezone.utc).isoformat(timespec="seconds"),
                "plan_hash": plan_hash,
                "status": "blocked" if active else "ready",
                "blocked": bool(active),
                "blocking_items": active,
                "resource_ids": resource_ids,
                "profile_ids": profile_ids,
                "memory_bank_ids": [str(item.get("profile_id") or "") for item in rows if item.get("resource_type") == "normal_memory_bank"],
                "configuration_ids": [str(item.get("resource_id") or "") for item in rows if item.get("resource_type") == "configuration"],
                "files": path_entries,
                "file_count": file_count,
                "total_bytes": total_bytes,
                "current_profile": self.current_profile().get("resource"),
                "shared_resources_preserved": ["base_model", "model_weight"],
                **self._target_domain_counts(rows),
                "resource_count": len(rows),
            }

    def clear_all_target_domain_resources(
        self,
        *,
        plan_id: str,
        preview_token: str,
        acknowledged: bool,
        actor: str = "local_operator",
    ) -> dict[str, Any]:
        if self.test_lock is not None:
            self.test_lock.require_write_allowed("resource_lifecycle")
        if not acknowledged:
            raise ApiError("CONFIRMATION_REQUIRED", "请先核对删除预览并勾选确认。", 422)
        with self._delete_lock:
            plan = self._clear_all_plans.get(str(plan_id or ""))
            if (
                not plan
                or str(plan.get("preview_token") or "") != str(preview_token or "")
                or float(plan.get("expires_at") or 0) <= time.time()
            ):
                raise ApiError("RESOURCE_CLEAR_ALL_PLAN_EXPIRED", "全部资源清理计划不存在或已过期，请重新预览。", 409)
            self._ensure_resource_mutation_idle()
            rows = self._target_domain_rows()
            resource_ids = [str(item.get("resource_id") or "") for item in rows if item.get("resource_id")]
            paths = self._target_domain_cleanup_paths()
            path_entries, file_count, total_bytes = self._cleanup_path_summary(paths)
            current_payload = {
                "resource_ids": resource_ids,
                "profile_ids": [str(item.get("profile_id") or "") for item in rows if item.get("resource_type") in {"target_profile", "candidate_profile"} and item.get("profile_id")],
                "path_entries": path_entries,
                "file_count": file_count,
                "total_bytes": total_bytes,
                "current_profile_id": str((self.current_profile().get("resource") or {}).get("profile_id") or ""),
            }
            current_hash = hashlib.sha256(_json(current_payload).encode("utf-8")).hexdigest()
            if current_hash != str(plan.get("plan_hash") or ""):
                raise ApiError("RESOURCE_CLEAR_ALL_PLAN_STALE", "目标域资源在预览后发生变化，请重新生成清理计划。", 409)

            operation_id = f"clear_{uuid.uuid4().hex[:20]}"
            trash_root = (self.settings.runtime_root / "resource_trash").resolve()
            isolated = trash_root / operation_id
            trash_root.mkdir(parents=True, exist_ok=True)
            isolated.mkdir(parents=True, exist_ok=False)
            moved: list[tuple[Path, Path]] = []
            runtime_before = self._runtime_state()
            db_committed = False
            fresh_paths: list[Path] = []
            try:
                for index, source in enumerate(paths):
                    if not source.exists():
                        raise ApiError("RESOURCE_CLEAR_ALL_PLAN_STALE", "目标域资源文件在执行前发生变化，请重新预览。", 409)
                    destination = isolated / f"{index:04d}_{source.name}"
                    shutil.move(str(source), str(destination))
                    moved.append((source, destination))
                moved_entries, moved_file_count, moved_total_bytes = self._cleanup_path_summary([item[1] for item in moved])
                if moved_file_count != file_count or moved_total_bytes != total_bytes:
                    raise ApiError("RESOURCE_CLEAR_ALL_ISOLATION_FAILED", "隔离目录文件数量或大小不一致，未提交清理。", 409)

                self.settings.memory_banks_root.mkdir(parents=True, exist_ok=True)
                self.settings.config_profiles_root.mkdir(parents=True, exist_ok=True)
                archive_root = self.settings.config_profiles_root / "_archived"
                archive_root.mkdir(parents=True, exist_ok=True)
                archive_registry = archive_root / "registry.json"
                archive_registry.write_text(json.dumps({"archived": {}, "deleted": {}}, ensure_ascii=False, indent=2), encoding="utf-8")
                fresh_paths.extend([archive_registry, archive_root])
                registry_path = self.settings.memory_bank_registry_path
                registry_path.write_text(json.dumps({"defaults": {}, "friendly_names": {}, "archived": {}, "unavailable": {}}, ensure_ascii=False, indent=2), encoding="utf-8")
                fresh_paths.append(registry_path)
                for name in ("domain_drafts", "domain_video_samples"):
                    directory = self.settings.jobs_root / name
                    directory.mkdir(parents=True, exist_ok=True)
                    fresh_paths.append(directory)

                now = _utc_now()
                with self._connect() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    feedback_release = {"affected": 0, "released": 0, "still_protected": 0, "active_jobs": 0}
                    for item in rows:
                        tombstone = {
                            "resource_id": item.get("resource_id"),
                            "profile_id": item.get("profile_id"),
                            "source_type": item.get("source_type"),
                            "clear_all": True,
                            "plan_id": plan_id,
                            "plan_hash": current_hash,
                        }
                        conn.execute(
                            "INSERT INTO runtime_resource_tombstones(tombstone_id,resource_id,profile_id,resource_type,display_name,version,dataset,category,method_mode,sha256,file_size,deleted_at,delete_operation_id,deletion_reason,former_profile_references_json,former_task_reference_count,metadata_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (
                                uuid.uuid4().hex,
                                str(item.get("resource_id") or ""),
                                item.get("profile_id"),
                                str(item.get("resource_type") or "resource"),
                                str(item.get("display_name") or "目标域资源"),
                                str(item.get("version") or ""),
                                item.get("dataset"),
                                item.get("category"),
                                item.get("method_mode"),
                                item.get("sha256"),
                                item.get("file_size"),
                                now,
                                operation_id,
                                "清空全部目标域资源",
                                "[]",
                                0,
                                _json(tombstone),
                            ),
                        )
                        if (
                            str(item.get("resource_type") or "") in {"target_profile", "candidate_profile", "normal_memory_bank"}
                            and self.feedback_candidates is not None
                        ):
                            released_feedback = self.feedback_candidates.mark_source_memory_bank_deleted(
                                profile_id=str(item.get("profile_id") or ""),
                                resource_id=str(item.get("resource_id") or ""),
                                actor=actor,
                                connection=conn,
                            )
                            for key in feedback_release:
                                feedback_release[key] += int(released_feedback.get(key) or 0)
                    if resource_ids:
                        placeholders = ",".join("?" for _ in resource_ids)
                        conn.execute(
                            "DELETE FROM runtime_resource_dependencies WHERE parent_resource_id IN (" + placeholders + ") OR child_resource_id IN (" + placeholders + ")",
                            tuple(resource_ids) + tuple(resource_ids),
                        )
                        conn.execute("DELETE FROM runtime_resources WHERE resource_id IN (" + placeholders + ")", tuple(resource_ids))
                    conn.execute(
                        "INSERT INTO runtime_resource_batch_operations(operation_id,operation_type,selected_count,success_count,skipped_count,failed_count,released_bytes,details_json,actor,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (operation_id, "clear_all_target_domain_resources", len(rows), len(rows), 0, 0, total_bytes, _json({"plan_id": plan_id, "plan_hash": current_hash, "counts": self._target_domain_counts(rows)}), str(actor or "local_operator")[:80], now),
                    )
                db_committed = True
                self._write_runtime_state(
                    {
                        "dataset": "",
                        "category": "",
                        "memory_bank_id": "",
                        "config_profile_id": "",
                        "config_profile_version": 1,
                        "config_hash": "",
                        "inference_mode": "inference_only",
                    }
                )
                self.domain_profiles.reset_after_resource_clear()
                event = self._bump_revision(
                    action="clear_all_target_domain_resources",
                    resource_ids=resource_ids,
                    profile_ids=current_payload["profile_ids"],
                    resource_types=sorted(TARGET_DOMAIN_RESOURCE_TYPES),
                    current_profile_changed=bool(current_payload["current_profile_id"]),
                )
                statistics_revision = None
                if self.history_management is not None:
                    statistics_revision = self.history_management.resource_dependency_changed(
                        "target_domain_resources_cleared_history_protection_recalculated"
                    )
                cleanup_pending = False
                try:
                    shutil.rmtree(isolated)
                except OSError:
                    cleanup_pending = True
                self._clear_all_plans.pop(str(plan_id), None)
                try:
                    self._record_audit(
                        "clear_all_target_domain_resources",
                        None,
                        actor,
                        {"plan_id": plan_id, "resource_count": len(rows), "file_count": file_count, "total_bytes": total_bytes},
                        {"operation_id": operation_id, "cleanup_pending": cleanup_pending},
                    )
                except Exception:
                    self.logger.exception("runtime_resource_clear_all_audit_failed operation_id=%s", operation_id)
                return {
                    "cleared": True,
                    "operation_id": operation_id,
                    "plan_id": plan_id,
                    "deleted_resource_count": len(rows),
                    "deleted_profile_count": self._target_domain_counts(rows)["profile_count"],
                    "deleted_memory_bank_count": self._target_domain_counts(rows)["normal_memory_bank_count"],
                    "deleted_configuration_count": self._target_domain_counts(rows)["configuration_count"],
                    "deleted_file_count": file_count,
                    "released_bytes": total_bytes,
                    "cleanup_pending": cleanup_pending,
                    "feedback_dependency_release": feedback_release,
                    "statistics_revision": statistics_revision,
                    "preserved_resource_types": ["base_model", "model_weight"],
                    **event,
                }
            except Exception as exc:
                if not db_committed:
                    try:
                        for fresh in reversed(fresh_paths):
                            if fresh.is_file():
                                fresh.unlink(missing_ok=True)
                            elif fresh.is_dir() and fresh.exists():
                                shutil.rmtree(fresh, ignore_errors=True)
                        for restore_target, moved_path in reversed(moved):
                            if moved_path.exists() and not restore_target.exists():
                                restore_target.parent.mkdir(parents=True, exist_ok=True)
                                shutil.move(str(moved_path), str(restore_target))
                        if isolated.exists():
                            shutil.rmtree(isolated, ignore_errors=True)
                        self._write_runtime_state(runtime_before)
                    except Exception:
                        self.logger.exception("runtime_resource_clear_all_restore_failed operation_id=%s", operation_id)
                if isinstance(exc, ApiError):
                    raise
                raise ApiError("RESOURCE_CLEAR_ALL_FAILED", "全部目标域资源清理失败；未提交的资源已经恢复。", 500) from exc

    def tombstones(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM runtime_resource_tombstones ORDER BY deleted_at DESC LIMIT ?", (max(1, min(int(limit), 500)),)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["former_profile_references"] = json.loads(item.pop("former_profile_references_json") or "[]")
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            item["profile_id"] = str(item.get("profile_id") or item["metadata"].get("profile_id") or "")
            result.append(item)
        return result

    def operation_history(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 50), 200))
        with self._connect() as conn:
            rows = conn.execute("SELECT audit_id,action,resource_id,actor,before_json,after_json,created_at FROM runtime_resource_audit WHERE action IN ('archive','restore','delete','delete_failed','delete_cleanup_retry','rebuild_index','validate') ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        result = []
        for row in rows:
            try:
                before = json.loads(row["before_json"] or "{}")
            except json.JSONDecodeError:
                before = {}
            try:
                after = json.loads(row["after_json"] or "{}")
            except json.JSONDecodeError:
                after = {}
            resource = before.get("resource") if isinstance(before, dict) and isinstance(before.get("resource"), dict) else (before if isinstance(before, dict) else {})
            result.append({
                "operation_id": after.get("operation_id") or row["audit_id"],
                "operation": row["action"],
                "resource_id": row["resource_id"],
                "resource_name": (resource or {}).get("display_name") or (after.get("resource") or {}).get("display_name") or "资源索引",
                "result": "失败" if row["action"] == "delete_failed" else "成功",
                "released_bytes": int(after.get("released_bytes") or 0),
                "error_message": str(after.get("error") or ""),
                "created_at": row["created_at"],
                "actor": row["actor"],
            })
        return result

    def pending_cleanup(self) -> dict[str, Any]:
        """List isolated deletion directories that still need final cleanup.

        A failed final ``rmtree`` must never make a resource selectable again;
        the database already treats it as deleted.  This small, path-free
        view lets the maintenance page retry only server-owned trash entries.
        """
        root = (self.settings.runtime_root / "resource_trash").resolve()
        items: list[dict[str, Any]] = []
        if not root.exists():
            return {"count": 0, "items": []}
        for candidate in sorted(root.iterdir(), key=lambda path: path.name):
            if not candidate.is_dir() or not DELETE_OPERATION_ID_RE.fullmatch(candidate.name):
                continue
            try:
                files, symlinks = self._artifact_listing(candidate)
                total = sum(int(item.get("size_bytes") or 0) for item in files)
                items.append({
                    "operation_id": candidate.name,
                    "file_count": len(files),
                    "total_bytes": total,
                    "unsafe": bool(symlinks),
                    "status": "需要人工处理" if symlinks else "待清理",
                })
            except OSError:
                items.append({"operation_id": candidate.name, "file_count": 0, "total_bytes": 0, "unsafe": True, "status": "需要人工处理"})
        return {"count": len(items), "items": items}

    def retry_cleanup(self, operation_id: str, actor: str = "operator") -> dict[str, Any]:
        """Retry final removal of one server-owned deletion trash directory."""
        operation_id = str(operation_id or "").strip()
        if not DELETE_OPERATION_ID_RE.fullmatch(operation_id):
            raise ApiError("RESOURCE_TRASH_INVALID", "待清理操作标识无效。", 422)
        with self._delete_lock:
            root = (self.settings.runtime_root / "resource_trash").resolve()
            target = (root / operation_id).resolve()
            if not _within(target, root) or target == root or not target.is_dir():
                raise ApiError("RESOURCE_TRASH_NOT_FOUND", "待清理隔离目录不存在，可能已经清理完成。", 404)
            files, symlinks = self._artifact_listing(target)
            if symlinks:
                raise ApiError("RESOURCE_TRASH_UNSAFE", "待清理目录存在 symbolic link，已停止自动清理。", 409)
            size = sum(int(item.get("size_bytes") or 0) for item in files)
            before = {"operation_id": operation_id, "file_count": len(files), "total_bytes": size, "state": "cleanup_pending"}
            try:
                shutil.rmtree(target)
            except OSError as exc:
                self._record_audit("delete_cleanup_retry", None, actor, before, {"operation_id": operation_id, "state": "cleanup_pending", "error": str(exc)[:240]})
                raise ApiError("RESOURCE_TRASH_CLEANUP_FAILED", "隔离目录仍无法清理，请稍后重试或由管理员处理。", 409) from exc
            if target.exists():
                self._record_audit("delete_cleanup_retry", None, actor, before, {"operation_id": operation_id, "state": "cleanup_pending", "error": "目录清理后仍存在"})
                raise ApiError("RESOURCE_TRASH_CLEANUP_FAILED", "隔离目录清理未完成，请稍后重试。", 409)
            self._record_audit("delete_cleanup_retry", None, actor, before, {"operation_id": operation_id, "state": "cleaned", "released_bytes": size})
            return {"cleaned": True, "operation_id": operation_id, "released_bytes": size, "remaining": False}

    def find_orphans(self) -> dict[str, Any]:
        indexed = {str(item.get("artifact_ref")) for item in self.list_resources(include_archived=True) if item.get("artifact_ref")}
        candidates: list[dict[str, Any]] = []
        for root, kind in ((self.settings.memory_banks_root, "memory_bank"), (self.settings.config_profiles_root, "configuration")):
            if not root.exists():
                continue
            for path in root.iterdir():
                if path.name.startswith("_") or path.name.startswith(".") or not path.is_dir():
                    continue
                ref = self._relative_ref(path)
                if ref not in indexed and not any(ref == str(item).split("/v", 1)[0] for item in indexed):
                    candidates.append({"kind": kind, "display_name": path.name, "artifact_ref": ref, "status": "待确认"})
        return {"count": len(candidates), "candidates": candidates[:200]}

    def maintenance_summary(self, *, scan: bool = True) -> dict[str, Any]:
        if scan:
            try:
                self.scan()
            except Exception as exc:
                self.logger.warning("runtime_resource_summary_scan_failed: %s", exc)
        with self._connect() as conn:
            total = int(conn.execute("SELECT COUNT(*) FROM runtime_resources WHERE archived=0").fetchone()[0])
            missing = int(conn.execute("SELECT COUNT(*) FROM runtime_resources WHERE archived=0 AND integrity_status IN ('incomplete','incompatible')").fetchone()[0])
            hashes = int(conn.execute("SELECT COUNT(*) FROM runtime_resources WHERE archived=0 AND integrity_status='hash_mismatch'").fetchone()[0])
            frozen = int(conn.execute("SELECT COUNT(*) FROM runtime_resources WHERE archived=0 AND frozen=1").fetchone()[0])
            archived = int(conn.execute("SELECT COUNT(*) FROM runtime_resources WHERE archived=1").fetchone()[0])
            dependency_count = int(conn.execute("SELECT COUNT(*) FROM runtime_resource_dependencies").fetchone()[0])
            indexed_orphans = int(conn.execute("SELECT COUNT(*) FROM runtime_resources WHERE integrity_status='orphan_candidate'").fetchone()[0])
        # ``scan=False`` is the lightweight list-page contract.  Calling
        # ``find_orphans`` here used to load the resource graph a second time
        # and enumerate both artifact roots after every ordinary button click.
        # A deliberate maintenance scan still performs the live filesystem
        # check; normal UI refreshes use the last indexed result.
        orphan_count = self.find_orphans().get("count", 0) if scan else indexed_orphans
        return {
            "total": total,
            "missing_or_incompatible": missing,
            "hash_anomaly": hashes,
            "frozen": frozen,
            "archived": archived,
            "dependency_count": dependency_count,
            "orphan_count": orphan_count,
        }

    def rebuild_index(self) -> dict[str, Any]:
        return self.scan(force=True)

    def cleanup_preview(self) -> dict[str, Any]:
        candidates = []
        for item in self.list_resources(include_archived=False):
            preview = self.delete_preview(str(item["resource_id"]))
            # An incomplete/hash-anomalous resource is still protected when it
            # is active, frozen or referenced by a profile.  It must be
            # repaired or archived through its owning profile instead of being
            # offered as a one-click cleanup candidate.
            safe_failed = (
                item.get("integrity_status") in {"incomplete", "hash_mismatch"}
                and not item.get("active")
                and not item.get("frozen")
                and not preview.get("references")
            )
            if preview.get("allowed") or safe_failed:
                candidates.append({"resource_id": item["resource_id"], "display_name": item["display_name"], "resource_type": item["resource_type"], "status": item.get("integrity_status_label"), "allowed": bool(preview.get("allowed")), "reasons": preview.get("reasons")})
        return {"count": len(candidates), "candidates": candidates}

    def cleanup(self, resource_ids: list[str], actor: str = "operator") -> dict[str, Any]:
        archived = []
        for resource_id in list(resource_ids or [])[:100]:
            archived.append(self.archive(resource_id, actor))
        return {"archived_count": len(archived), "resources": archived}

    def backup_index(self, actor: str = "operator") -> dict[str, Any]:
        """Copy the runtime SQLite index to a unique backup location.

        The database also contains historical records, so the operation is
        explicitly opt-in and never overwrites an existing backup.  No model,
        image or memory-bank artifact is copied.
        """
        source = self.path
        if not source.is_file():
            raise ApiError("RUNTIME_INDEX_MISSING", "资源索引数据库尚未建立，暂时无法备份。", 404)
        backup_root = self.settings.project_root.parent / "AdaptiveAOI_Backup" / "08_修改前文件快照"
        backup_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = backup_root / f"运行资源索引备份_{stamp}"
        suffix = 1
        while target_dir.exists():
            suffix += 1
            target_dir = backup_root / f"运行资源索引备份_{stamp}_{suffix:02d}"
        target_dir.mkdir(parents=True)
        target = target_dir / "detection_history.sqlite3"
        with self._lock:
            shutil.copy2(source, target)
        digest = _sha256(target)
        manifest = {"source": "outputs/system_runtime/detection_history.sqlite3", "backup": target.name, "size_bytes": target.stat().st_size, "sha256": digest, "created_at": _utc_now(), "actor": str(actor or "operator")[:80], "note": "仅备份运行索引数据库，不包含模型、图片或记忆库本体。"}
        (target_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        self._record_audit("backup_index", None, actor, None, manifest)
        return {"backup_name": target_dir.name, "size_bytes": manifest["size_bytes"], "sha256": digest}

    def restore_default(self, actor: str = "operator") -> dict[str, Any]:
        default = next(
            (
                item for item in self.list_resources(resource_type="target_profile")
                if bool((item.get("metadata") or {}).get("is_default"))
                and deployment_scope(item) == "formal"
            ),
            None,
        )
        if default is None:
            raise ApiError("DEFAULT_PROFILE_NOT_FOUND", "系统没有可验证的默认冻结方案。", 409)
        result = self.apply_profile(str(default["resource_id"]), actor)
        return {"restored": True, **result}


__all__ = ["RuntimeResourceService", "RESOURCE_TYPES", "STATUS_LABELS"]
