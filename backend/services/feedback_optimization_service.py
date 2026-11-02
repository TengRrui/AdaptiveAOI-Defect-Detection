"""Safe, asynchronous optimization from reviewed feedback candidates.

The worker never mutates an active memory bank.  It computes a candidate
profile in a job-owned directory and validates it against reviewed evidence.
Normal modes require an explicit apply call; the explicit ``replace_original``
mode performs the same apply transaction in the worker only after the user has
confirmed replacement and the candidate has passed validation.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sqlite3
import statistics
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from backend.errors import ApiError
from backend.database import DatabaseManager
from backend.services.feedback_task_status import (
    ACTIVE_ITEM_STATUSES,
    TERMINAL_ITEM_STATUSES,
    TERMINAL_TASK_STATUSES,
    get_feedback_item_status_display,
    get_feedback_task_status_display,
)
from backend.services.feedback_update_result import RESULT_SCHEMA_VERSION, build_feedback_update_result
from backend.services.job_service import JobPaths, JobService
from backend.services.strategy_eligibility_policy import (
    evaluate_strategy_eligibility,
    get_strategy_policy,
    list_strategy_policies,
)


PROFILE_METHOD_ALIASES = {
    "normal_memory": "normal_only",
    "normal_only": "normal_only",
    "normal_defect_calibration": "normal_defect_calibration",
    "lightweight_calibration": "normal_memory_defect_guided_lite",
    "normal_memory_defect_guided_lite": "normal_memory_defect_guided_lite",
    "dual_memory": "normal_defect_dual_memory",
    "normal_defect_dual_memory": "normal_defect_dual_memory",
}


FEEDBACK_UPDATE_STAGES = (
    {"stage_id": "feedback_validation", "stage_name": "校验反馈样本", "weight": 5.0},
    {"stage_id": "profile_compatibility", "stage_name": "检查来源记忆库与配置兼容性", "weight": 5.0},
    {"stage_id": "prepare_normal", "stage_name": "准备正常纠错样本", "weight": 4.0},
    {"stage_id": "prepare_defect", "stage_name": "准备缺陷纠错样本", "weight": 4.0},
    {"stage_id": "runtime_loading", "stage_name": "加载并复用特征提取 Runtime", "weight": 7.0},
    {"stage_id": "feature_extraction", "stage_name": "提取新增样本特征", "weight": 28.0},
    {"stage_id": "profile_workspace", "stage_name": "创建候选版本工作区", "weight": 7.0},
    {"stage_id": "normal_memory_update", "stage_name": "更新正常特征资源", "weight": 12.0},
    {"stage_id": "defect_reference_update", "stage_name": "更新缺陷参考特征", "weight": 8.0},
    {"stage_id": "parameter_calibration", "stage_name": "重新计算判定参数", "weight": 7.0},
    {"stage_id": "ann_index", "stage_name": "构建检索索引", "weight": 5.0},
    {"stage_id": "integrity", "stage_name": "资源完整性与 SHA-256 校验", "weight": 4.0},
    {"stage_id": "validation", "stage_name": "验证候选新版本", "weight": 7.0},
    {"stage_id": "config_profile", "stage_name": "生成新 Config 与 Profile", "weight": 4.0},
    {"stage_id": "save_result", "stage_name": "保存更新结果", "weight": 3.0},
)

TERMINAL_OPTIMIZATION_STATES = set(TERMINAL_TASK_STATUSES)
REUSABLE_REVIEW_STATUSES = frozenset({
    "accepted", "queued", "running", "optimizing", "ready_to_apply",
    "consumed", "validation_failed", "frozen", "applied", "rolled_back",
    "completed",
})
USAGE_EVENT_STATUSES = frozenset({
    "queued", "running", "optimizing", "ready_to_apply", "consumed",
    "frozen", "applied", "rolled_back", "completed",
})


def _profile_method_mode(value: Any) -> str:
    """Normalize profile/build mode names without confusing them with scoring kernels."""
    raw = str(value or "").strip().lower()
    return PROFILE_METHOD_ALIASES.get(raw, raw or "normal_only")


def _target_method_mode(base_mode: Any, strategy: Any) -> str:
    """Return the immutable candidate profile mode produced by an update strategy."""
    base = _profile_method_mode(base_mode)
    selected = str(strategy or "safe_incremental").strip().lower()
    # Threshold fitting with reviewed defect evidence upgrades a pure normal
    # profile to the existing normal-memory + defect-calibration profile.  It
    # does not alter the normal bank or the feature extractor.
    if selected == "threshold_only" and base == "normal_only":
        return "normal_defect_calibration"
    return base


def _utc_now() -> str:
    # Millisecond timestamps keep short stages observable while remaining ISO
    # compatible with databases created by older releases.
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value: Any, fallback: Any) -> Any:
    try:
        result = json.loads(value or "") if isinstance(value, str) else value
        return result if result is not None else fallback
    except (TypeError, ValueError, json.JSONDecodeError):
        return fallback


def _int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _parse_metric_float(value: Any, *, default: float | None = None) -> float | None:
    """Parse a persisted metric without treating a legal zero as missing.

    Metrics are persisted in JSON and older jobs may contain ``None`` or an
    empty string.  ``0``/``0.0`` are valid measurements (especially FPR,
    recall, and growth), so do not use ``value or default`` here.  Non-finite
    and non-numeric values are treated as unavailable and return ``default``.
    """
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _metric_or_default(value: Any, default: float) -> float:
    """Return a finite metric or an explicit default, preserving zero."""
    parsed = _parse_metric_float(value)
    return float(default) if parsed is None else parsed


class FeedbackOptimizationService:
    """Create, validate, apply and roll back immutable feedback profiles."""

    def __init__(
        self,
        settings: Any,
        jobs: JobService,
        storage: Any,
        candidates: Any,
        domain_profiles: Any,
        config_profiles: Any,
        registry: Any,
        video_jobs: Any,
        logger: Any,
    ):
        self.settings = settings
        self.jobs = jobs
        self.storage = storage
        self.candidates = candidates
        self.domain_profiles = domain_profiles
        self.config_profiles = config_profiles
        self.registry = registry
        self.video_jobs = video_jobs
        self.logger = logger
        self.path = storage.path
        self.database = getattr(storage, "database", DatabaseManager(self.path))
        # Bound by ``build_services`` after RuntimeResourceService is created.
        # Keeping the dependency optional avoids a construction cycle while
        # still allowing feedback publication/application to update the shared
        # resource catalog revision used by every page.
        self.runtime_resources: Any | None = None
        self.runtime_profiles: Any | None = None
        self.test_lock: Any | None = None
        self._lock = threading.RLock()
        self._cancel: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._ensure_schema()
        self._recover_interrupted_jobs()
        self._repair_legacy_item_states()
        self._stage_history = self._load_stage_history()

    def _runtime_state(self) -> dict[str, Any]:
        runtime_profiles = getattr(self, "runtime_profiles", None)
        if runtime_profiles is not None:
            return runtime_profiles.raw_runtime()
        return self.settings.dynamic_settings()

    def _require_test_write_allowed(self, operation: str, *, context: dict[str, Any] | None = None) -> None:
        lock = getattr(self, "test_lock", None)
        if lock is not None:
            lock.require_write_allowed(operation, context=context)

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
                CREATE TABLE IF NOT EXISTS feedback_optimization_jobs (
                    job_id TEXT PRIMARY KEY, base_profile_id TEXT NOT NULL,
                    base_config_profile_id TEXT, base_config_profile_version INTEGER,
                    candidate_ids_json TEXT NOT NULL, strategy TEXT NOT NULL,
                    objective TEXT NOT NULL, status TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0,
                    stage TEXT NOT NULL DEFAULT 'queued', message TEXT NOT NULL DEFAULT '',
                    constraints_json TEXT NOT NULL DEFAULT '{}', candidate_profile_id TEXT,
                    candidate_config_profile_id TEXT, candidate_config_profile_version INTEGER,
                    metrics_before_json TEXT NOT NULL DEFAULT '{}', metrics_after_json TEXT NOT NULL DEFAULT '{}',
                    validation_json TEXT NOT NULL DEFAULT '{}', result_json TEXT NOT NULL DEFAULT '{}',
                    error_code TEXT, error_message TEXT,
                    created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, applied_at TEXT,
                    rolled_back_at TEXT, previous_profile_id TEXT, previous_config_profile_id TEXT,
                    previous_config_profile_version INTEGER,
                    replacement_old_resource_id TEXT,
                    replacement_cleanup_json TEXT NOT NULL DEFAULT '{}',
                    idempotency_key TEXT NOT NULL DEFAULT '',
                    current_stage_id TEXT NOT NULL DEFAULT '',
                    stages_json TEXT NOT NULL DEFAULT '[]', timings_json TEXT NOT NULL DEFAULT '{}',
                    stage_started_at TEXT, last_progress_at TEXT, heartbeat_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_feedback_optimization_status ON feedback_optimization_jobs(status, created_at DESC);
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
                """
            )
            columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(feedback_optimization_jobs)").fetchall()}
            migrations = {
                "idempotency_key": "TEXT NOT NULL DEFAULT ''",
                "current_stage_id": "TEXT NOT NULL DEFAULT ''",
                "stages_json": "TEXT NOT NULL DEFAULT '[]'",
                "timings_json": "TEXT NOT NULL DEFAULT '{}'",
                "stage_started_at": "TEXT",
                "last_progress_at": "TEXT",
                "heartbeat_at": "TEXT",
                "result_json": "TEXT NOT NULL DEFAULT '{}'",
                "replacement_old_resource_id": "TEXT",
                "replacement_cleanup_json": "TEXT NOT NULL DEFAULT '{}'",
            }
            for name, definition in migrations.items():
                if name not in columns:
                    conn.execute(f"ALTER TABLE feedback_optimization_jobs ADD COLUMN {name} {definition}")
            item_columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(feedback_optimization_items)").fetchall()}
            item_migrations = {
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
            }
            for name, definition in item_migrations.items():
                if name not in item_columns:
                    conn.execute(f"ALTER TABLE feedback_optimization_items ADD COLUMN {name} {definition}")
            conn.execute("DROP INDEX IF EXISTS idx_feedback_optimization_idempotency")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_optimization_idempotency "
                "ON feedback_optimization_jobs(idempotency_key, created_at DESC)"
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_optimization_item_unique "
                "ON feedback_optimization_items(job_id, candidate_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_optimization_item_candidate_usage "
                "ON feedback_optimization_items(candidate_id, used_at DESC)"
            )
            # Populate the durable usage facts for candidates generated by
            # earlier releases.  Failed/skipped items are intentionally not
            # counted as use.
            conn.execute(
                """
                UPDATE feedback_optimization_items
                SET source_memory_bank_id=COALESCE(NULLIF(source_memory_bank_id,''),(
                        SELECT j.base_profile_id FROM feedback_optimization_jobs j
                        WHERE j.job_id=feedback_optimization_items.job_id
                    )),
                    target_memory_bank_id=COALESCE(NULLIF(target_memory_bank_id,''),(
                        SELECT j.candidate_profile_id FROM feedback_optimization_jobs j
                        WHERE j.job_id=feedback_optimization_items.job_id
                    )),
                    target_memory_bank_version=COALESCE(NULLIF(target_memory_bank_version,''),'1'),
                    strategy=COALESCE(NULLIF(strategy,''),(
                        SELECT j.strategy FROM feedback_optimization_jobs j
                        WHERE j.job_id=feedback_optimization_items.job_id
                    )),
                    used_at=COALESCE(NULLIF(used_at,''),finished_at,(
                        SELECT COALESCE(j.finished_at,j.applied_at,j.created_at)
                        FROM feedback_optimization_jobs j
                        WHERE j.job_id=feedback_optimization_items.job_id
                    )),
                    usage_result=COALESCE(NULLIF(usage_result,''),(
                        SELECT j.status FROM feedback_optimization_jobs j
                        WHERE j.job_id=feedback_optimization_items.job_id
                    )),
                    generated_profile_id=COALESCE(NULLIF(generated_profile_id,''),(
                        SELECT j.candidate_profile_id FROM feedback_optimization_jobs j
                        WHERE j.job_id=feedback_optimization_items.job_id
                    ))
                WHERE status IN ('completed','ready_to_apply','applied')
                  AND EXISTS(
                      SELECT 1 FROM feedback_optimization_jobs j
                      WHERE j.job_id=feedback_optimization_items.job_id
                        AND COALESCE(j.candidate_profile_id,'')<>''
                  )
                """
            )

    @staticmethod
    def _item_transition_allowed(previous: str, target: str) -> bool:
        if previous == target:
            return True
        transitions = {
            "queued": {"validating", "cancelled", "failed", "interrupted", "legacy_state_unknown"},
            "validating": {"processing", "skipped", "cancelled", "failed", "interrupted"},
            "processing": {"completed", "skipped", "cancelled", "failed", "interrupted"},
        }
        return target in transitions.get(previous, set())

    def _set_items_status(
        self,
        job_id: str,
        status: str,
        *,
        candidate_ids: list[str] | None = None,
        from_statuses: set[str] | frozenset[str] | None = None,
        message: str = "",
        result_code: str = "",
        components_by_id: dict[str, list[str]] | None = None,
        allow_legacy_repair: bool = False,
    ) -> int:
        """Advance item rows atomically; never move a completed row backwards."""
        target = str(status or "").strip().lower()
        if target not in {*ACTIVE_ITEM_STATUSES, *TERMINAL_ITEM_STATUSES}:
            raise ApiError("FEEDBACK_ITEM_STATUS_INVALID", "反馈更新明细状态无效。", 500)
        selected = {str(value) for value in (candidate_ids or []) if str(value)}
        now = _utc_now()
        changed = 0
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM feedback_optimization_items WHERE job_id=?",
                (str(job_id),),
            ).fetchall()
            columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(feedback_optimization_items)").fetchall()}
            for row in rows:
                candidate_id = str(row["candidate_id"])
                previous = str(row["status"] or "queued")
                if selected and candidate_id not in selected:
                    continue
                if from_statuses is not None and previous not in from_statuses:
                    continue
                if not allow_legacy_repair and not self._item_transition_allowed(previous, target):
                    self.logger.warning(
                        "feedback_item_transition_rejected job_id=%s candidate_id=%s from=%s to=%s",
                        job_id,
                        candidate_id,
                        previous,
                        target,
                    )
                    continue
                values: dict[str, Any] = {"status": target, "message": str(message)}
                if "result_code" in columns:
                    values["result_code"] = str(result_code)
                if "updated_at" in columns:
                    values["updated_at"] = now
                if "started_at" in columns and target in {"validating", "processing"} and not row["started_at"]:
                    values["started_at"] = now
                if "finished_at" in columns and target in TERMINAL_ITEM_STATUSES:
                    values["finished_at"] = now
                if "components_json" in columns and components_by_id is not None:
                    values["components_json"] = json.dumps(
                        list(components_by_id.get(candidate_id) or []),
                        ensure_ascii=False,
                    )
                conn.execute(
                    f"UPDATE feedback_optimization_items SET {', '.join(f'{key}=?' for key in values)} "
                    "WHERE job_id=? AND candidate_id=?",
                    [*values.values(), str(job_id), candidate_id],
                )
                changed += 1
        return changed

    @staticmethod
    def _candidate_update_components(candidate: dict[str, Any], payload: dict[str, Any]) -> list[str]:
        """Identify real update outputs that can consume one reviewed sample."""
        label = str(candidate.get("corrected_label") or "")
        prediction = str(candidate.get("original_prediction") or "").strip().lower()
        annotations = list(candidate.get("annotations") or [])
        has_mask = bool(candidate.get("has_mask")) or any(
            str(item.get("mask_path") or item.get("mask_artifact_ref") or "") for item in annotations
        )
        components: list[str] = []
        if bool(payload.get("update_threshold")) and candidate.get("original_score") is not None and label in {"normal", "defect"}:
            components.append("threshold")
        if (
            bool(payload.get("update_normal_memory"))
            and label == "normal"
            and prediction in {"1", "anomaly", "abnormal", "defect", "ng", "异常", "缺陷"}
            and not has_mask
        ):
            components.append("normal_memory")
        if bool(payload.get("update_defect_memory")) and label == "defect" and has_mask:
            components.append("defect_prototypes")
        if bool(payload.get("update_lightweight_calibrator")) and label in {"normal", "defect"}:
            components.append("lightweight_calibrator")
        return components

    def _prepare_processing_items(
        self,
        job_id: str,
        candidates: list[dict[str, Any]],
        payload: dict[str, Any],
    ) -> dict[str, list[str]]:
        components = {
            str(row["candidate_id"]): self._candidate_update_components(row, payload)
            for row in candidates
        }
        used_ids = [candidate_id for candidate_id, values in components.items() if values]
        skipped_ids = [candidate_id for candidate_id, values in components.items() if not values]
        if used_ids:
            self._set_items_status(
                job_id,
                "processing",
                candidate_ids=used_ids,
                from_statuses={"validating"},
                message="已通过校验，正在参与候选版本生成",
                result_code="accepted_for_update",
                components_by_id=components,
            )
        if skipped_ids:
            self._set_items_status(
                job_id,
                "skipped",
                candidate_ids=skipped_ids,
                from_statuses={"validating"},
                message="当前选择的更新内容不会使用此类样本",
                result_code="strategy_not_applicable",
                components_by_id=components,
            )
        return components

    def _item_summary(self, job_id: str) -> dict[str, int]:
        counts = {key: 0 for key in (
            "total", "queued", "validating", "processing", "completed", "skipped",
            "failed", "cancelled", "interrupted", "legacy_state_unknown",
        )}
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status,COUNT(*) AS count FROM feedback_optimization_items WHERE job_id=? GROUP BY status",
                (str(job_id),),
            ).fetchall()
        for row in rows:
            status = str(row["status"] or "queued")
            value = int(row["count"] or 0)
            counts[status] = counts.get(status, 0) + value
            if status in {"ready_to_apply", "frozen", "applied", "consumed", "rolled_back"}:
                # These are terminal usage states of an item; expose them as
                # completed work in the result summary while retaining the
                # raw status count for audit/debug views.
                counts["completed"] += value
            counts["total"] += value
        return counts

    def _item_ids(self, job_id: str, *statuses: str) -> list[str]:
        normalized = [str(value) for value in statuses if str(value)]
        with self._connect() as conn:
            if normalized:
                placeholders = ",".join("?" for _ in normalized)
                rows = conn.execute(
                    f"SELECT candidate_id FROM feedback_optimization_items WHERE job_id=? AND status IN ({placeholders})",
                    [str(job_id), *normalized],
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT candidate_id FROM feedback_optimization_items WHERE job_id=?",
                    (str(job_id),),
                ).fetchall()
        return [str(row["candidate_id"]) for row in rows]

    def _assert_items_finalized(self, job_id: str, target_status: str) -> None:
        summary = self._item_summary(job_id)
        active = sum(int(summary.get(status) or 0) for status in ACTIVE_ITEM_STATUSES)
        if active <= 0:
            return
        self.logger.error(
            "inconsistent_feedback_task_state job_id=%s target=%s item_summary=%s",
            job_id,
            target_status,
            summary,
        )
        self._insert_audit(job_id, "inconsistent_feedback_task_state", {
            "target_status": target_status,
            "item_summary": summary,
        })
        raise ApiError(
            "FEEDBACK_ITEM_STATE_INCOMPLETE",
            "反馈样本明细尚未完成收敛，任务不能进入完成状态。",
            500,
            {"item_summary": summary},
        )

    def _legacy_consumed_candidate_ids(self, row: sqlite3.Row) -> set[str]:
        candidate_profile_id = str(row["candidate_profile_id"] or "")
        if not candidate_profile_id:
            return set()
        roots = [
            self.settings.memory_banks_root / candidate_profile_id,
            self.settings.memory_banks_root / ".feedback_candidates" / candidate_profile_id,
        ]
        roots.extend((self.settings.memory_banks_root / ".failed_feedback_updates").glob(f"{row['job_id']}_*"))
        for root in roots:
            manifest = root / "feedback_update_manifest.json"
            if manifest.is_file():
                payload = _json(manifest.read_text(encoding="utf-8"), {})
                values = {
                    str(item.get("candidate_id") or "")
                    for item in list(payload.get("samples") or [])
                    if str(item.get("candidate_id") or "")
                }
                if values:
                    return values
            metadata_path = root / "metadata.json"
            if metadata_path.is_file():
                metadata = _json(metadata_path.read_text(encoding="utf-8"), {})
                values = {str(value) for value in list(metadata.get("feedback_candidate_ids") or []) if str(value)}
                if values:
                    return values
        return set()

    def _repair_legacy_item_states(self) -> None:
        """Repair only item outcomes that can be inferred from durable task evidence."""
        repaired_total = 0
        with self._connect() as conn:
            jobs = conn.execute(
                "SELECT * FROM feedback_optimization_jobs WHERE status IN "
                "('ready_to_apply','frozen','applied','rolled_back','completed','validation_failed','cancelled','failed','interrupted') "
                "AND EXISTS(SELECT 1 FROM feedback_optimization_items i WHERE i.job_id=feedback_optimization_jobs.job_id AND i.status='queued')"
            ).fetchall()
        for row in jobs:
            job_id = str(row["job_id"])
            status = str(row["status"] or "")
            queued_ids: list[str]
            with self._connect() as conn:
                queued_ids = [
                    str(item["candidate_id"])
                    for item in conn.execute(
                        "SELECT candidate_id FROM feedback_optimization_items WHERE job_id=? AND status='queued'",
                        (job_id,),
                    ).fetchall()
                ]
            repaired: dict[str, int] = {}
            if status in {"ready_to_apply", "frozen", "applied", "rolled_back", "completed"}:
                consumed = self._legacy_consumed_candidate_ids(row)
                completed_ids = [value for value in queued_ids if value in consumed]
                unknown_ids = [value for value in queued_ids if value not in consumed]
                if completed_ids:
                    repaired["completed"] = self._set_items_status(
                        job_id,
                        "completed",
                        candidate_ids=completed_ids,
                        from_statuses={"queued"},
                        message="依据已生成候选版本清单修复历史明细状态",
                        result_code="legacy_manifest_consumed",
                        allow_legacy_repair=True,
                    )
                if unknown_ids:
                    repaired["legacy_state_unknown"] = self._set_items_status(
                        job_id,
                        "legacy_state_unknown",
                        candidate_ids=unknown_ids,
                        from_statuses={"queued"},
                        message="历史任务缺少逐样本处理记录，无法安全推断",
                        result_code="legacy_evidence_missing",
                        allow_legacy_repair=True,
                    )
            elif status == "cancelled":
                repaired["cancelled"] = self._set_items_status(
                    job_id, "cancelled", from_statuses={"queued"},
                    message="历史任务已取消，明细未继续处理", result_code="cancelled_by_task",
                    allow_legacy_repair=True,
                )
            elif status == "validation_failed":
                repaired["failed"] = self._set_items_status(
                    job_id, "failed", from_statuses={"queued"},
                    message="候选版本验证未通过，样本未用于可应用版本", result_code="candidate_validation_failed",
                    allow_legacy_repair=True,
                )
            elif status == "interrupted":
                repaired["interrupted"] = self._set_items_status(
                    job_id, "interrupted", from_statuses={"queued"},
                    message="历史任务因服务重启中断", result_code="worker_interrupted",
                    allow_legacy_repair=True,
                )
            else:
                repaired["legacy_state_unknown"] = self._set_items_status(
                    job_id, "legacy_state_unknown", from_statuses={"queued"},
                    message="历史失败任务缺少逐样本结果，无法安全推断", result_code="legacy_evidence_missing",
                    allow_legacy_repair=True,
                )
            repaired_count = sum(repaired.values())
            if repaired_count:
                repaired_total += repaired_count
                self._insert_audit(job_id, "legacy_item_state_repaired", repaired)
        if repaired_total:
            self.logger.info("feedback_optimization_legacy_items_repaired count=%s", repaired_total)

    def _recover_interrupted_jobs(self) -> None:
        """A worker thread cannot survive a backend restart; persist that fact."""
        now = _utc_now()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT job_id,candidate_ids_json FROM feedback_optimization_jobs WHERE status IN "
                "('queued','preparing','running','building','updating','optimizing','validating','applying','cancelling')"
            ).fetchall()
            for row in rows:
                ids = [str(value) for value in _json(row["candidate_ids_json"], [])]
                conn.execute(
                    "UPDATE feedback_optimization_jobs SET status='interrupted',stage='服务重启中断',message='任务未完成，可重新运行。',finished_at=? WHERE job_id=?",
                    (now, str(row["job_id"])),
                )
                conn.execute(
                    "UPDATE feedback_optimization_items SET status='interrupted',message='服务重启导致任务中断',"
                    "result_code='worker_interrupted',finished_at=?,updated_at=? "
                    "WHERE job_id=? AND status IN ('queued','validating','processing')",
                    (now, now, str(row["job_id"])),
                )
                for candidate_id in ids:
                    conn.execute(
                        "UPDATE feedback_candidates SET status='accepted',updated_at=? WHERE candidate_id=? AND status IN ('queued','optimizing')",
                        (now, candidate_id),
                    )

    def _load_stage_history(self) -> dict[str, list[float]]:
        history: dict[str, list[float]] = {}
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT stages_json FROM feedback_optimization_jobs "
                "WHERE status IN ('ready_to_apply','frozen','applied','rolled_back') "
                "ORDER BY finished_at DESC LIMIT 40"
            ).fetchall()
        for row in rows:
            for stage in _json(row["stages_json"], []):
                if str(stage.get("status") or "") not in {"completed", "warning"}:
                    continue
                duration = float(stage.get("elapsed_seconds") or 0.0)
                if duration >= 0.0:
                    history.setdefault(str(stage.get("stage_id") or ""), []).append(duration)
        return {key: values[:20] for key, values in history.items() if key}

    @staticmethod
    def _stage_applicability(payload: dict[str, Any], stage_id: str, candidates: list[dict[str, Any]]) -> tuple[bool, str]:
        has_normal = any(str(row.get("corrected_label") or "") == "normal" for row in candidates)
        has_defect = any(str(row.get("corrected_label") or "") == "defect" for row in candidates)
        resource_update = any(bool(payload.get(key)) for key in (
            "update_normal_memory", "update_defect_memory", "update_lightweight_calibrator",
        ))
        rules = {
            "prepare_normal": (has_normal, "当前选择中没有正常纠错样本"),
            "prepare_defect": (has_defect, "当前选择中没有缺陷纠错样本"),
            "runtime_loading": (resource_update, "仅调整记录分数与判定参数，无需加载特征 Runtime"),
            "feature_extraction": (resource_update, "当前更新内容无需重新提取图像特征"),
            "normal_memory_update": (bool(payload.get("update_normal_memory")), "当前更新内容不修改正常特征资源"),
            "defect_reference_update": (bool(payload.get("update_defect_memory")), "当前更新内容不修改缺陷参考特征"),
            "parameter_calibration": (
                bool(payload.get("update_threshold")) or bool(payload.get("update_lightweight_calibrator")),
                "当前更新内容不重新计算判定参数",
            ),
            "ann_index": (bool(payload.get("update_normal_memory")), "正常特征资源未改变，无需重建检索索引"),
        }
        return rules.get(stage_id, (True, "等待执行"))

    def _initialize_stages(self, payload: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        now = _utc_now()
        stages: list[dict[str, Any]] = []
        for definition in FEEDBACK_UPDATE_STAGES:
            applicable, message = self._stage_applicability(payload, str(definition["stage_id"]), candidates)
            stages.append({
                **definition,
                "status": "pending" if applicable else "skipped",
                "progress": 0.0,
                "started_at": None,
                "finished_at": now if not applicable else None,
                "elapsed_seconds": 0.0,
                "message": "等待执行" if applicable else message,
            })
        return stages

    @staticmethod
    def _weighted_progress(stages: list[dict[str, Any]]) -> float:
        applicable = [stage for stage in stages if str(stage.get("status") or "") != "skipped"]
        total = sum(float(stage.get("weight") or 0.0) for stage in applicable)
        if total <= 0.0:
            return 0.0
        done = 0.0
        for stage in applicable:
            weight = float(stage.get("weight") or 0.0)
            status = str(stage.get("status") or "")
            if status in {"completed", "warning"}:
                done += weight
            elif status == "running":
                done += weight * min(1.0, max(0.0, float(stage.get("progress") or 0.0)))
        return min(0.99, max(0.0, done / total))

    def _set_stage(
        self,
        job_id: str,
        stage_id: str,
        *,
        status: str = "running",
        progress: float | None = None,
        message: str | None = None,
        completed_items: int | None = None,
        total_items: int | None = None,
        current_item: str | None = None,
    ) -> dict[str, Any]:
        row = self._public(self._row(job_id))
        stages = list(row.get("stages") or [])
        stage = next((item for item in stages if str(item.get("stage_id") or "") == str(stage_id)), None)
        if stage is None or str(stage.get("status") or "") == "skipped":
            return row
        now = _utc_now()
        if status in {"running", "completed", "warning", "failed"} and not stage.get("started_at"):
            stage["started_at"] = now
        if progress is not None:
            stage["progress"] = max(float(stage.get("progress") or 0.0), min(1.0, float(progress)))
        if message is not None:
            stage["message"] = str(message)
        if completed_items is not None:
            stage["completed_items"] = max(0, int(completed_items))
        if total_items is not None:
            stage["total_items"] = max(0, int(total_items))
        if current_item is not None:
            stage["current_item"] = str(current_item)
        stage["status"] = status
        if status in {"completed", "warning", "failed"}:
            stage["finished_at"] = now
            if status in {"completed", "warning"}:
                stage["progress"] = 1.0
            try:
                stage["elapsed_seconds"] = max(
                    0.0,
                    (datetime.fromisoformat(now) - datetime.fromisoformat(str(stage.get("started_at") or now))).total_seconds(),
                )
            except ValueError:
                stage["elapsed_seconds"] = float(stage.get("elapsed_seconds") or 0.0)
        overall = self._weighted_progress(stages)
        return self._update(
            job_id,
            current_stage_id=str(stage_id),
            stage=str(stage.get("stage_name") or stage_id),
            message=str(stage.get("message") or ""),
            progress=overall,
            stages_json=json.dumps(stages, ensure_ascii=False, default=str),
            stage_started_at=stage.get("started_at"),
            last_progress_at=now,
            heartbeat_at=now,
        )

    def _record_timing(self, job_id: str, name: str, seconds: float, **details: Any) -> None:
        row = self._public(self._row(job_id))
        timings = dict(row.get("timings") or {})
        timings[str(name)] = {"seconds": round(max(0.0, float(seconds)), 6), **details}
        try:
            self._update(job_id, timings_json=json.dumps(timings, ensure_ascii=False, default=str))
        except sqlite3.OperationalError as exc:
            # Legacy embedded databases created before the timing migration
            # remain operable; the formal service constructor always runs the
            # migration before accepting requests.
            if "timings_json" not in str(exc):
                raise

    def _heartbeat_loop(self, job_id: str, stop: threading.Event) -> None:
        while not stop.wait(2.0):
            try:
                with self._connect() as conn:
                    row = conn.execute("SELECT status FROM feedback_optimization_jobs WHERE job_id=?", (job_id,)).fetchone()
                    if row is None or str(row["status"] or "") in TERMINAL_OPTIMIZATION_STATES:
                        return
                    conn.execute(
                        "UPDATE feedback_optimization_jobs SET heartbeat_at=? WHERE job_id=?",
                        (_utc_now(), job_id),
                    )
            except sqlite3.Error:
                return

    @staticmethod
    def _check_cancel(cancel: threading.Event) -> None:
        if cancel.is_set():
            raise ApiError("FEEDBACK_OPTIMIZATION_CANCELLED", "反馈优化任务已取消。", 409)

    def _remember_stage_history(self, stages: list[dict[str, Any]]) -> None:
        for stage in stages:
            if str(stage.get("status") or "") not in {"completed", "warning"}:
                continue
            stage_id = str(stage.get("stage_id") or "")
            if not stage_id:
                continue
            values = self._stage_history.setdefault(stage_id, [])
            values.insert(0, float(stage.get("elapsed_seconds") or 0.0))
            self._stage_history[stage_id] = values[:20]

    def _public(self, row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        item = dict(row)
        for key in (
            "candidate_ids_json", "constraints_json", "metrics_before_json", "metrics_after_json",
            "complexity_before_json", "complexity_after_json", "validation_json", "result_json",
            "stages_json", "timings_json", "replacement_cleanup_json",
        ):
            if key in item:
                list_value = key in {"candidate_ids_json", "stages_json"}
                item[key.removesuffix("_json")] = _json(item.pop(key), [] if list_value else {})
        stages = list(item.get("stages") or [])
        now = datetime.now(timezone.utc)
        running_stage = next((stage for stage in stages if str(stage.get("status") or "") == "running"), None)
        if running_stage and running_stage.get("started_at"):
            try:
                running_stage["elapsed_seconds"] = max(
                    float(running_stage.get("elapsed_seconds") or 0.0),
                    (now - datetime.fromisoformat(str(running_stage["started_at"]))).total_seconds(),
                )
            except ValueError:
                pass
        started_at = item.get("started_at")
        finished_at = item.get("finished_at")
        elapsed_seconds = 0.0
        if started_at:
            try:
                end_time = datetime.fromisoformat(str(finished_at)) if finished_at else now
                elapsed_seconds = max(
                    0.0,
                    (end_time - datetime.fromisoformat(str(started_at))).total_seconds(),
                )
            except ValueError:
                elapsed_seconds = 0.0
        status = str(item.get("status") or "")
        # ``completed`` is a legacy terminal state from earlier workers.  It
        # still represents a successfully generated candidate and must not be
        # rendered with an in-progress progress bar after a restart.
        successful = status in {"ready_to_apply", "frozen", "applied", "rolled_back", "completed"}
        item["progress"] = 1.0 if successful else min(0.99, self._weighted_progress(stages) if stages else float(item.get("progress") or 0.0))
        eta: float | None = 0.0 if successful else None
        if status in {"queued", "running", "cancelling"} and stages:
            estimates: list[float] = []
            known = True
            for stage in stages:
                if str(stage.get("status") or "") in {"completed", "warning", "skipped"}:
                    continue
                history = self._stage_history.get(str(stage.get("stage_id") or "")) or []
                if not history:
                    known = False
                    break
                median = float(statistics.median(history))
                estimates.append(max(0.0, median - float(stage.get("elapsed_seconds") or 0.0)) if str(stage.get("status") or "") == "running" else median)
            if known:
                eta = sum(estimates)
        def age(value: Any) -> float | None:
            if not value:
                return None
            try:
                return max(0.0, (now - datetime.fromisoformat(str(value))).total_seconds())
            except ValueError:
                return None
        heartbeat_age = age(item.get("heartbeat_at"))
        progress_age = age(item.get("last_progress_at"))
        activity = "active"
        if status == "running" and heartbeat_age is not None and heartbeat_age > 15.0:
            activity = "possibly_interrupted"
        elif status == "running" and progress_age is not None and progress_age > 20.0:
            activity = "long_running"
        presentation = get_feedback_task_status_display(status)
        item.update(
            elapsed_seconds=round(elapsed_seconds, 3),
            current_stage_id=str((running_stage or {}).get("stage_id") or item.get("current_stage_id") or ""),
            current_stage_elapsed_seconds=round(float((running_stage or {}).get("elapsed_seconds") or 0.0), 3),
            eta_seconds=None if eta is None else round(float(eta), 3),
            estimated_remaining_seconds=None if eta is None else round(float(eta), 3),
            activity_status=activity,
            heartbeat_age_seconds=heartbeat_age,
            last_progress_age_seconds=progress_age,
            rerunnable=bool(presentation["can_retry"]),
            **presentation,
        )
        return item

    def _row(self, job_id: str) -> sqlite3.Row:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM feedback_optimization_jobs WHERE job_id=?", (str(job_id),)).fetchone()
        if row is None:
            raise ApiError("FEEDBACK_OPTIMIZATION_JOB_NOT_FOUND", "反馈优化任务不存在。", 404)
        return row

    def _store_result(self, job_id: str, public: dict[str, Any], item_summary: dict[str, Any] | None = None) -> dict[str, Any]:
        """Attach and persist the stable result, including for legacy rows."""
        result = build_feedback_update_result(public, item_summary or {})
        public["result"] = result.as_dict()
        try:
            with self._connect() as conn:
                columns = {
                    str(value["name"])
                    for value in conn.execute("PRAGMA table_info(feedback_optimization_jobs)").fetchall()
                }
                if "result_json" in columns:
                    conn.execute(
                        "UPDATE feedback_optimization_jobs SET result_json=? WHERE job_id=?",
                        (json.dumps(public["result"], ensure_ascii=False, default=str), str(job_id)),
                    )
        except sqlite3.Error:
            # Formal databases are migrated at construction.  This fallback
            # keeps isolated legacy test databases readable while returning
            # the same in-memory contract.
            pass
        return public

    def _update(self, job_id: str, **values: Any) -> dict[str, Any]:
        allowed = {
            "status", "progress", "stage", "message", "candidate_profile_id", "candidate_config_profile_id",
            "candidate_config_profile_version", "metrics_before_json", "metrics_after_json", "validation_json",
            "result_json",
            "complexity_before_json", "complexity_after_json",
            "error_code", "error_message", "started_at", "finished_at", "applied_at", "rolled_back_at",
            "previous_profile_id", "previous_config_profile_id", "previous_config_profile_version",
            "replacement_old_resource_id", "replacement_cleanup_json",
            "idempotency_key", "current_stage_id", "stages_json", "timings_json",
            "stage_started_at", "last_progress_at", "heartbeat_at",
        }
        values = {k: v for k, v in values.items() if k in allowed}
        target_status = str(values.get("status") or "")
        if target_status in {"validation_failed", "failed", "cancelled", "interrupted"}:
            item_status = "failed" if target_status in {"validation_failed", "failed"} else target_status
            item_message = str(values.get("message") or "任务未完成")
            self._set_items_status(
                job_id,
                item_status,
                from_statuses=ACTIVE_ITEM_STATUSES,
                message=item_message,
                result_code={
                    "validation_failed": "candidate_validation_failed",
                    "failed": "task_failed",
                    "cancelled": "cancelled_by_task",
                    "interrupted": "worker_interrupted",
                }[target_status],
            )
        elif target_status in {"ready_to_apply", "frozen", "applied", "completed", "rolled_back"}:
            self._assert_items_finalized(job_id, target_status)
        if values:
            with self._connect() as conn:
                conn.execute(
                    f"UPDATE feedback_optimization_jobs SET {', '.join(f'{key}=?' for key in values)} WHERE job_id=?",
                    [*values.values(), str(job_id)],
                )
        public = self._public(self._row(job_id))
        # Every terminal task receives a durable, user-facing result.  Keep
        # this beside the raw validation JSON so historical tasks can be
        # explained after a restart without asking the UI to infer an
        # outcome from internal status strings.
        if target_status in TERMINAL_TASK_STATUSES:
            try:
                item_summary = self._item_summary(job_id)
                public = self._store_result(job_id, public, item_summary)
            except (sqlite3.Error, TypeError, ValueError):
                # Isolated legacy test databases may not have the migration;
                # the formal service always does.  Preserve the state update
                # and expose a best-effort in-memory result in that case.
                pass
            if "result" not in public:
                public = self._store_result(job_id, public, {})

        jobs = getattr(self, "jobs", None)
        manifest = jobs.get(str(job_id), "feedback_optimization") if jobs is not None else None
        if manifest is not None and manifest.manifest_path.exists():
            self.jobs.update(
                manifest,
                status=public.get("status"),
                stage=public.get("stage"),
                progress=public.get("progress"),
                message=public.get("message"),
                candidate_profile_id=public.get("candidate_profile_id"),
                result=public.get("result"),
                error={"code": public.get("error_code"), "message": public.get("error_message")} if public.get("error_code") else None,
            )
        return public

    def _insert_audit(self, job_id: str, action: str, details: dict[str, Any] | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO feedback_audit_log(audit_id,job_id,action,actor,details_json,created_at) VALUES(?,?,?,?,?,?)",
                (
                    uuid.uuid4().hex,
                    str(job_id),
                    action,
                    "system",
                    json.dumps(details or {}, ensure_ascii=False, default=str),
                    _utc_now(),
                ),
            )

    def _set_candidate_status(self, candidate_ids: list[str], status: str, job_id: str, message: str = "") -> None:
        if not candidate_ids:
            return
        with self._connect() as conn:
            for candidate_id in candidate_ids:
                row = conn.execute("SELECT status FROM feedback_candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
                if not row:
                    continue
                previous = str(row["status"] or "")
                if status in USAGE_EVENT_STATUSES and previous in REUSABLE_REVIEW_STATUSES:
                    # Usage is a task/sample relation, not a terminal state of
                    # the human-reviewed sample.  Normalize legacy rows back
                    # to the durable approval state while retaining the task
                    # event in the audit log and item relation.
                    conn.execute(
                        "UPDATE feedback_candidates SET status='accepted', updated_at=? WHERE candidate_id=?",
                        (_utc_now(), candidate_id),
                    )
                    conn.execute(
                        "INSERT INTO feedback_audit_log(audit_id,candidate_id,job_id,action,actor,from_status,to_status,details_json,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                        (
                            uuid.uuid4().hex, candidate_id, job_id,
                            "candidate_usage_status", "system", previous, status,
                            json.dumps({"message": message, "persisted_review_status": "accepted"}, ensure_ascii=False),
                            _utc_now(),
                        ),
                    )
                    continue
                if previous != status and hasattr(self.candidates, "_transition_allowed") and not self.candidates._transition_allowed(previous, status):
                    # A stale worker must never force a candidate through an
                    # illegal state transition.  Keep the candidate state
                    # unchanged and retain an audit trail for diagnosis.
                    conn.execute(
                        "INSERT INTO feedback_audit_log(audit_id,candidate_id,job_id,action,actor,from_status,to_status,details_json,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                        (uuid.uuid4().hex, candidate_id, job_id, "candidate_transition_rejected", "system", previous, status, json.dumps({"message": message}, ensure_ascii=False), _utc_now()),
                    )
                    continue
                conn.execute("UPDATE feedback_candidates SET status=?, updated_at=? WHERE candidate_id=?", (status, _utc_now(), candidate_id))
                conn.execute("INSERT INTO feedback_audit_log(audit_id,candidate_id,job_id,action,actor,from_status,to_status,details_json,created_at) VALUES(?,?,?,?,?,?,?,?,?)", (uuid.uuid4().hex, candidate_id, job_id, "candidate_status", "system", previous, status, json.dumps({"message": message}, ensure_ascii=False), _utc_now()))

    def _record_usage_outcome(self, job_id: str, result: str) -> int:
        """Persist one immutable task/sample usage fact for every used item.

        Candidate approval is intentionally not changed here.  Reusability is
        a property of the reviewed correction snapshot; use is represented by
        the many-to-many item row keyed by ``(job_id, candidate_id)``.
        """
        row = self._row(job_id)
        target_id = str(row["candidate_profile_id"] or "")
        if not target_id:
            return 0
        target_version = "1"
        for root in (self.settings.memory_banks_root / ".feedback_candidates", self.settings.memory_banks_root):
            metadata_path = root / target_id / "metadata.json"
            if not metadata_path.is_file():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                target_version = str(
                    metadata.get("target_domain_version")
                    or metadata.get("profile_version")
                    or metadata.get("version")
                    or "1"
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
            break
        now = _utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE feedback_optimization_items
                SET source_memory_bank_id=COALESCE(NULLIF(source_memory_bank_id,''),?),
                    target_memory_bank_id=?, target_memory_bank_version=?,
                    strategy=COALESCE(NULLIF(strategy,''),?),
                    used_at=COALESCE(NULLIF(used_at,''),finished_at,?),
                    usage_result=?, generated_profile_id=?
                WHERE job_id=? AND status IN ('completed','ready_to_apply','applied')
                """,
                (
                    str(row["base_profile_id"] or ""), target_id, target_version,
                    str(row["strategy"] or ""), now, str(result or "completed"),
                    target_id, str(job_id),
                ),
            )
            affected = int(cursor.rowcount or 0)
            try:
                conn.execute(
                    "UPDATE feedback_state SET revision=revision+1,updated_at=?,reason=? WHERE singleton=1",
                    (now, "optimization_usage_update"),
                )
            except sqlite3.Error:
                pass
            try:
                conn.execute(
                    "UPDATE statistics_state SET revision=revision+1,updated_at=?,reason=? WHERE singleton=1",
                    (now, "optimization_usage_update"),
                )
            except sqlite3.Error:
                pass
        return affected

    def _official_test_lock(self, candidate_rows: list[dict[str, Any]]) -> bool:
        dynamic = self._runtime_state()
        if bool(dynamic.get("official_test_mode")) or bool(dynamic.get("feedback_optimization_locked")):
            return True
        return any(
            bool((row.get("context") or {}).get("official_test_mode")) or row.get("status") == "record_only"
            for row in candidate_rows
        )

    @staticmethod
    def _metrics(scores: list[float], labels: list[int], threshold: float) -> dict[str, Any]:
        if not scores or len(scores) != len(labels):
            return {"sample_count": 0, "status": "insufficient_samples"}
        pred = [int(float(score) >= float(threshold)) for score in scores]
        tp = sum(p == 1 and y == 1 for p, y in zip(pred, labels))
        tn = sum(p == 0 and y == 0 for p, y in zip(pred, labels))
        fp = sum(p == 1 and y == 0 for p, y in zip(pred, labels))
        fn = sum(p == 0 and y == 1 for p, y in zip(pred, labels))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        normal_count = fp + tn
        defect_count = fn + tp
        return {
            "sample_count": len(scores),
            "threshold": float(threshold),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            # A zero numerator is a valid metric.  A zero denominator is not
            # a zero rate: it means this evidence contains no samples for the
            # corresponding class and the rate cannot be evaluated.
            "false_positive_rate": round(fp / normal_count, 6) if normal_count else None,
            "false_positive_rate_status": "ok" if normal_count else "not_evaluable",
            "false_positive_sample_count": normal_count,
            "false_negative_rate": round(fn / defect_count, 6) if defect_count else None,
            "false_negative_rate_status": "ok" if defect_count else "not_evaluable",
            "false_negative_sample_count": defect_count,
        }

    @classmethod
    def _bootstrap_stability(cls, scores: list[float], labels: list[int], threshold: float, seed: int = 42) -> dict[str, Any]:
        """Estimate small-sample metric stability without changing the model.

        This is deliberately a light bootstrap over the reviewed evidence; it
        is a validation diagnostic, not a second training loop.
        """
        if len(scores) < 5 or len(scores) != len(labels) or len(set(labels)) < 2:
            return {"status": "insufficient_samples"}
        rng = np.random.default_rng(int(seed))
        values: list[tuple[float, float, float]] = []
        size = len(scores)
        score_array = np.asarray(scores, dtype=float)
        label_array = np.asarray(labels, dtype=int)
        for _ in range(200):
            indices = rng.integers(0, size, size=size)
            metrics = cls._metrics(score_array[indices].tolist(), label_array[indices].tolist(), threshold)
            # A bootstrap draw can contain only one class even when the full
            # evidence is valid.  Do not turn that non-evaluable FPR into a
            # fabricated zero; simply omit the draw from the stability
            # estimate.
            if metrics.get("false_positive_rate") is None:
                continue
            values.append((float(metrics["f1"]), float(metrics["recall"]), float(metrics["false_positive_rate"])))
        if not values:
            return {"status": "not_evaluable", "reason": "no_normal_samples_in_bootstrap"}
        matrix = np.asarray(values, dtype=float)
        return {
            "status": "ok",
            "iterations": int(len(values)),
            "f1_p05": round(float(np.quantile(matrix[:, 0], 0.05)), 6),
            "f1_p50": round(float(np.quantile(matrix[:, 0], 0.50)), 6),
            "f1_p95": round(float(np.quantile(matrix[:, 0], 0.95)), 6),
            "recall_p05": round(float(np.quantile(matrix[:, 1], 0.05)), 6),
            "false_positive_rate_p95": round(float(np.quantile(matrix[:, 2], 0.95)), 6),
        }

    @classmethod
    def _choose_threshold(
        cls,
        rows: list[dict[str, Any]],
        current: float,
        objective: str,
        min_recall: float,
        max_fpr: float,
    ) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
        evidence = [
            (float(row["original_score"]), 1 if row.get("corrected_label") == "defect" else 0)
            for row in rows
            if row.get("original_score") is not None and row.get("corrected_label") in {"normal", "defect"}
        ]
        if not evidence:
            return float(current), cls._metrics([], [], current), []
        if len({label for _, label in evidence}) < 2:
            # Threshold search needs both normal and defect evidence.  In
            # particular, never treat a defect-only set as FPR=0.
            metrics = cls._metrics([item[0] for item in evidence], [item[1] for item in evidence], current)
            metrics["eligible"] = False
            return float(current), metrics, []
        candidates = sorted({float(current), *[score for score, _ in evidence], *[score + 1e-6 for score, _ in evidence]})
        scored: list[dict[str, Any]] = []
        for threshold in candidates:
            metrics = cls._metrics([item[0] for item in evidence], [item[1] for item in evidence], threshold)
            recall = _parse_metric_float(metrics.get("recall"))
            fpr = _parse_metric_float(metrics.get("false_positive_rate"))
            metrics["eligible"] = recall is not None and fpr is not None and recall >= float(min_recall) and fpr <= float(max_fpr)
            scored.append(metrics)
        eligible = [item for item in scored if item.get("eligible")]
        pool = eligible or scored
        if objective == "recall_priority":
            chosen = max(pool, key=lambda item: (item["recall"], item["f1"], -item["false_positive_rate"], -item["threshold"]))
        elif objective == "balanced":
            chosen = max(
                pool,
                key=lambda item: (
                    item["recall"] + (1.0 - item["false_positive_rate"]) + item["f1"],
                    item["f1"],
                ),
            )
        else:
            chosen = max(pool, key=lambda item: (item["f1"], item["recall"], -item["false_positive_rate"]))
        return float(chosen["threshold"]), chosen, scored

    @staticmethod
    def _strategy_flags(payload: dict[str, Any], base_mode: str | None = None) -> dict[str, bool]:
        """Return one unambiguous set of resource updates for a UI strategy.

        The visible strategy is authoritative.  Explicit resource flags are
        retained only for legacy callers whose strategy is not one of the
        current five choices.  ``comprehensive`` is resolved against the base
        method so it cannot request an unsupported mutually-exclusive resource.
        """
        strategy = str(payload.get("strategy") or "safe_incremental").strip().lower()
        if strategy == "threshold_only":
            return {
                "update_threshold": True,
                "update_normal_memory": False,
                "update_defect_memory": False,
                "update_lightweight_calibrator": False,
            }
        if strategy == "normal_memory":
            return {
                "update_threshold": False,
                "update_normal_memory": True,
                "update_defect_memory": False,
                "update_lightweight_calibrator": False,
            }
        if strategy == "defect_prototypes":
            return {
                "update_threshold": False,
                "update_normal_memory": False,
                "update_defect_memory": True,
                "update_lightweight_calibrator": False,
            }
        if strategy == "lightweight_calibrator":
            return {
                "update_threshold": True,
                "update_normal_memory": False,
                "update_defect_memory": False,
                "update_lightweight_calibrator": True,
            }
        if strategy == "comprehensive":
            normalized_mode = str(base_mode or "normal_only")
            return {
                "update_threshold": True,
                "update_normal_memory": True,
                "update_defect_memory": normalized_mode == "normal_defect_dual_memory",
                "update_lightweight_calibrator": normalized_mode == "normal_memory_defect_guided_lite",
            }
        # Compatibility with old API clients that sent only resource flags.
        return {
            "update_threshold": bool(payload.get("update_threshold", True)),
            "update_normal_memory": bool(payload.get("update_normal_memory", False)),
            "update_defect_memory": bool(payload.get("update_defect_memory", False)),
            "update_lightweight_calibrator": bool(payload.get("update_lightweight_calibrator", False)),
        }

    @staticmethod
    def strategy_policies(base_method_mode: str | None = None) -> list[dict[str, Any]]:
        """Return the same strategy policy used by compatibility checks."""
        return list_strategy_policies(base_method_mode)

    def compatibility(self, payload: dict[str, Any], *, include_rows: bool = False) -> dict[str, Any]:
        """Validate reviewed evidence against one immutable base profile.

        This is deliberately side-effect free so the page can show every
        incompatibility before a background job is created.  Scores collected
        from another memory bank are not mixed into threshold fitting even if
        the dataset/category happen to match.
        """
        base_id = str(payload.get("base_profile_id") or "").strip()
        candidate_ids = list(dict.fromkeys(
            str(value).strip() for value in payload.get("candidate_ids") or [] if str(value).strip()
        ))
        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        if not base_id or base_id.startswith("builtin_"):
            errors.append({"code": "FEEDBACK_BASE_PROFILE_INVALID", "message": "请选择已冻结的目标域 profile，公开基线不能作为反馈增量更新对象。"})
        if not candidate_ids:
            errors.append({"code": "FEEDBACK_CANDIDATES_REQUIRED", "message": "请选择至少一个已审核反馈样本。"})
        for candidate_id in candidate_ids:
            try:
                item = self.candidates.get_candidate(candidate_id)
            except ApiError as exc:
                errors.append({"code": exc.code, "message": exc.message, "candidate_id": candidate_id})
                continue
            rows.append(item)
            if str(item.get("status") or "") not in REUSABLE_REVIEW_STATUSES:
                errors.append({"code": "FEEDBACK_CANDIDATE_NOT_ACCEPTED", "message": "只有已审核通过且当前有效的纠错样本才能加入更新任务。", "candidate_id": candidate_id})
            source_bank_id = str(
                item.get("source_memory_bank_id")
                or item.get("memory_bank_id")
                or item.get("base_memory_bank_id")
                or ""
            )
            if source_bank_id and source_bank_id != base_id:
                warnings.append({
                    "code": "FEEDBACK_CROSS_MEMORY_BANK_SOURCE",
                    "message": "该样本来自其他目标域版本；系统将继续校验数据集、类别、特征模型和特征维度。",
                    "candidate_id": candidate_id,
                    "advanced_details": {
                        "source_memory_bank_id": source_bank_id,
                        "target_memory_bank_id": base_id,
                    },
                })
            context = dict(item.get("context") or {})
            lock = getattr(self, "test_lock", None)
            if lock is not None and lock.state(context=context).get("test_only_source"):
                errors.append({"code": "OFFICIAL_TEST_WRITE_LOCKED", "message": "正式测试样本只能保留检测记录，不能参与反馈优化。", "candidate_id": candidate_id})
            if bool(context.get("official_test_mode")) or bool(context.get("feedback_optimization_locked")) or str(item.get("status") or "") == "record_only":
                errors.append({"code": "OFFICIAL_TEST_FEEDBACK_LOCKED", "message": "正式测试或显式锁定样本只能记录，不能参与资源更新。", "candidate_id": candidate_id})
            if not item.get("source_snapshot_sha256"):
                errors.append({"code": "FEEDBACK_SNAPSHOT_REQUIRED", "message": "审核样本缺少不可变来源快照，请重新完成审核。", "candidate_id": candidate_id})

        base_profile: dict[str, Any] = {}
        base_metadata: dict[str, Any] = {}
        base_effective_config: dict[str, Any] = {}
        if base_id and rows:
            first = rows[0]
            try:
                resolved = self.domain_profiles.resolve_for_inference(
                    base_id,
                    str(first.get("model_version") or ""),
                    str(first.get("dataset") or ""),
                    str(first.get("category") or ""),
                )
                base_profile = {key: value for key, value in resolved.items() if key not in {"memory_path", "calibration_path", "config"}}
                base_effective_config = dict(resolved.get("config") or {})
                metadata_path = Path(resolved["memory_path"]).parent / "metadata.json"
                base_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if not bool(base_metadata.get("frozen")):
                    errors.append({"code": "FEEDBACK_BASE_NOT_FROZEN", "message": "基础 profile 尚未冻结。"})
            except (ApiError, OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append({"code": getattr(exc, "code", "FEEDBACK_BASE_PROFILE_INVALID"), "message": getattr(exc, "message", "基础 profile 不存在、不完整或不兼容。")})

        selected_config_id = str(base_metadata.get("config_profile_id") or "")
        selected_config_version = int(base_metadata.get("config_profile_version") or 0)
        source_config_keys = {
            (str(item.get("config_profile_id") or ""), _int_or_zero(item.get("config_profile_version")))
            for item in rows
            if str(item.get("config_profile_id") or "") and _int_or_zero(item.get("config_profile_version"))
        }
        if len(source_config_keys) > 1:
            warnings.append({
                "code": "FEEDBACK_MULTIPLE_CONFIG_VARIANTS",
                "message": "所选反馈来自同一记忆库的多个配置；系统将把新判定参数写入基础版本当前配置，其余配置仅继承。",
            })
        elif source_config_keys:
            selected_config_id, selected_config_version = next(iter(source_config_keys))
            config_profiles = getattr(self, "config_profiles", None)
            if config_profiles is not None:
                try:
                    selected_config = config_profiles.get(selected_config_id, selected_config_version)
                    selected_owner = str(selected_config.get("memory_bank_id") or selected_config.get("target_domain_version") or "")
                    if selected_owner and selected_owner != base_id:
                        warnings.append({
                            "code": "FEEDBACK_SOURCE_CONFIG_FROM_OTHER_BANK",
                            "message": "反馈来源配置属于其他目标域；只在特征空间兼容时允许复用。",
                            "advanced_details": {
                                "source_memory_bank_id": selected_owner,
                                "target_memory_bank_id": base_id,
                            },
                        })
                    if base_metadata and config_profiles.feature_hash(selected_config.get("config") or {}) != config_profiles.feature_hash(base_effective_config):
                        errors.append({"code": "FEEDBACK_CONFIG_FEATURE_MISMATCH", "message": "反馈使用的配置与基础记忆库特征空间不兼容。"})
                except (ApiError, OSError, TypeError, ValueError):
                    errors.append({"code": "FEEDBACK_SOURCE_CONFIG_MISSING", "message": "反馈使用的配置已不存在，不能安全生成新版本。"})

        if base_metadata:
            expected = {
                "dataset": str(base_metadata.get("dataset") or "").lower(),
                "category": str(base_metadata.get("category") or ""),
                "model_version": str(base_metadata.get("model_id") or ""),
                "config_profile_id": selected_config_id,
                "config_profile_version": selected_config_version,
            }
            config_variant_warned = False
            for item in rows:
                candidate_id = str(item.get("candidate_id") or "")
                actual_config_version = _int_or_zero(item.get("config_profile_version"))
                checks = (
                    (str(item.get("dataset") or "").lower() == expected["dataset"], "数据集不一致"),
                    (str(item.get("category") or "") == expected["category"], "类别不一致"),
                    (str(item.get("model_version") or "") == expected["model_version"], "特征提取模型不一致"),
                )
                for passed, message in checks:
                    if not passed:
                        errors.append({"code": "FEEDBACK_PROFILE_COMPATIBILITY_FAILED", "message": message, "candidate_id": candidate_id})
                source_config_id = str(item.get("config_profile_id") or "")
                if not config_variant_warned and (
                    (source_config_id and expected["config_profile_id"] and source_config_id != expected["config_profile_id"])
                    or (actual_config_version and expected["config_profile_version"] and actual_config_version != expected["config_profile_version"])
                ):
                    config_variant_warned = True
                    warnings.append({
                        "code": "FEEDBACK_CONFIG_VARIANT",
                        "message": "部分反馈使用了该记忆库的其他配置版本；候选版本会以当前基础方案重新生成配置快照。",
                        "candidate_id": candidate_id,
                        "advanced_details": {
                            "source_config_profile_id": source_config_id,
                            "source_config_profile_version": actual_config_version,
                            "base_config_profile_id": expected["config_profile_id"],
                            "base_config_profile_version": expected["config_profile_version"],
                        },
                    })
                context = dict(item.get("context") or {})
                source_size = _int_or_zero(context.get("image_size") or context.get("input_size"))
                source_dimension = _int_or_zero(context.get("feature_dimension") or context.get("embedding_dimension"))
                base_size = _int_or_zero(base_metadata.get("image_size"))
                base_dimension = _int_or_zero(base_metadata.get("feature_dimension"))
                if source_size and base_size and source_size != base_size:
                    errors.append({"code": "FEEDBACK_INPUT_SIZE_MISMATCH", "message": "输入尺寸与基础 profile 不一致。", "candidate_id": candidate_id})
                if source_dimension and base_dimension and source_dimension != base_dimension:
                    errors.append({"code": "FEEDBACK_EMBEDDING_DIMENSION_MISMATCH", "message": "特征维度与基础 profile 不一致。", "candidate_id": candidate_id})

        base_mode = _profile_method_mode(
            base_metadata.get("method_mode_resolved")
            or base_metadata.get("method_mode_requested")
            or base_profile.get("method_mode_resolved")
            or "normal_only"
        )
        flags = self._strategy_flags(payload, base_mode)
        mode = _target_method_mode(base_mode, payload.get("strategy"))
        reviewed_normal = [row for row in rows if row.get("corrected_label") == "normal"]
        reviewed_defect = [row for row in rows if row.get("corrected_label") == "defect"]
        source_modes: set[str] = set()
        raw_source_modes: set[str] = set()
        for row in rows:
            context = dict(row.get("context") or {})
            raw_source_mode = str(
                context.get("profile_method_mode")
                or context.get("method_mode_resolved")
                or row.get("method_mode")
                or context.get("method_mode")
                or ""
            ).strip()
            if raw_source_mode:
                raw_source_modes.add(raw_source_mode)
                source_modes.add(_profile_method_mode(raw_source_mode))
        if source_modes and (source_modes != {base_mode} or raw_source_modes != {base_mode}):
            warnings.append({
                "code": "FEEDBACK_METHOD_MODE_VARIANT",
                "message": "当前反馈样本来源于基础检测方式，与目标域版本的校准方式不同；系统已改用记忆库、模型和特征空间进行兼容校验。",
                "suggestions": ["创建新的目标域版本", "仅调整判定参数", "重新选择基础记忆库"],
                "advanced_details": {
                    "source_method_modes": sorted(raw_source_modes),
                    "base_profile_method_mode": base_mode,
                    "target_profile_method_mode": mode,
                },
            })
        safe_normal = [
            row for row in reviewed_normal
            if str(row.get("original_prediction") or "").strip().lower() in {"1", "anomaly", "abnormal", "defect", "ng", "异常", "缺陷"}
            and not bool(row.get("has_mask"))
            and not any(str(annotation.get("mask_path") or annotation.get("mask_artifact_ref") or "") for annotation in row.get("annotations") or [])
        ]
        masked_defect = [
            row for row in reviewed_defect
            if bool(row.get("has_mask"))
            or any(str(annotation.get("mask_path") or annotation.get("mask_artifact_ref") or "") for annotation in row.get("annotations") or [])
        ]
        eligibility = evaluate_strategy_eligibility(
            str(payload.get("strategy") or "threshold_only"),
            base_mode,
            {
                "normal": len(reviewed_normal),
                "defect": len(reviewed_defect),
                "safe_normal": len(safe_normal),
                "reliable_mask_defect": len(masked_defect),
            },
        )
        errors.extend(eligibility.get("blockers") or [])
        warnings.extend(eligibility.get("warnings") or [])
        if flags["update_defect_memory"]:
            if mode != "normal_defect_dual_memory":
                unsupported = {"code": "FEEDBACK_DEFECT_MEMORY_UNSUPPORTED", "message": "当前目标域版本尚未启用缺陷原型库。请先创建双记忆目标域版本，或改为仅调整判定参数。"}
                errors.append(unsupported)
                eligibility.setdefault("blockers", []).append(unsupported)
                eligibility["can_start"] = False
                eligibility["can_auto_apply"] = False
        if flags["update_lightweight_calibrator"]:
            if mode != "normal_memory_defect_guided_lite":
                unsupported = {"code": "FEEDBACK_LIGHTWEIGHT_UNSUPPORTED", "message": "当前目标域版本尚未启用轻量校准器。请先创建轻量校准目标域版本，或改为仅调整判定参数。"}
                errors.append(unsupported)
                eligibility.setdefault("blockers", []).append(unsupported)
                eligibility["can_start"] = False
                eligibility["can_auto_apply"] = False
        if not any(flags.values()):
            errors.append({"code": "FEEDBACK_UPDATE_REQUIRED", "message": "请至少选择一种更新内容。"})
        publish_mode = str(payload.get("publish_mode") or "update_current")
        if publish_mode in {"update_current", "replace_original"}:
            current_id = str(self._runtime_state().get("memory_bank_id") or "")
            if current_id != base_id:
                message = (
                    "‘替换原版本’必须以当前运行 profile 为基础；如需保留其他基础版本，请选择‘生成新版本但暂不应用’。"
                    if publish_mode == "replace_original"
                    else "‘生成新版本并在验证通过后应用’必须以当前运行 profile 为基础；如需保留其他基础版本，请选择‘生成新版本但暂不应用’。"
                )
                errors.append({"code": "FEEDBACK_CURRENT_PROFILE_MISMATCH", "message": message})

        groups: dict[str, int] = {}
        for row in rows:
            source_profile = str(row.get("memory_bank_id") or "未关联")
            groups[source_profile] = groups.get(source_profile, 0) + 1
        result = {
            "compatible": not errors,
            "errors": errors,
            "warnings": warnings,
            "groups": groups,
            "base_profile": base_profile,
            "method_mode": mode,
            "base_method_mode": base_mode,
            "target_method_mode": mode,
            "source_method_modes": sorted(raw_source_modes),
            "mode_transition": base_mode != mode,
            "update_flags": flags,
            "eligibility": eligibility,
            "strategy_policy": eligibility.get("policy") or {},
            "eligibility_requirements": eligibility.get("eligibility_requirements") or {},
            "recommended_requirements": eligibility.get("recommended_requirements") or {},
            "current_counts": eligibility.get("current_counts") or {},
            "blockers": eligibility.get("blockers") or [],
            "can_start": bool(eligibility.get("can_start")),
            "normal_count": len(reviewed_normal),
            "defect_count": len(reviewed_defect),
            "safe_normal_count": len(safe_normal),
            "masked_defect_count": len(masked_defect),
            "active_config_profile_id": selected_config_id,
            "active_config_profile_version": selected_config_version,
        }
        if include_rows:
            result["_rows"] = rows
        return result

    def create_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        preflight_started = time.perf_counter()
        self._require_test_write_allowed("feedback_update", context=payload)
        dynamic = self._runtime_state()
        if bool(dynamic.get("official_test_mode")) or bool(dynamic.get("feedback_optimization_locked")):
            raise ApiError("OFFICIAL_TEST_FEEDBACK_LOCKED", "当前处于正式测试锁定状态，反馈只能记录，不能创建优化任务。", 409)
        base_id = str(payload.get("base_profile_id") or "").strip()
        if not base_id or base_id.startswith("builtin_"):
            raise ApiError("FEEDBACK_BASE_PROFILE_INVALID", "反馈优化必须基于已冻结的目标域记忆库，不能修改公开基线。", 409)
        ids = list(dict.fromkeys(str(item).strip() for item in payload.get("candidate_ids") or [] if str(item).strip()))
        if not ids:
            raise ApiError("FEEDBACK_CANDIDATES_REQUIRED", "请选择至少一个已审核反馈样本。", 422)
        publish_mode = str(payload.get("publish_mode") or "update_current")
        if publish_mode == "replace_original" and not bool(payload.get("replace_confirm")):
            raise ApiError(
                "REPLACEMENT_CONFIRMATION_REQUIRED",
                "替换原版本前必须确认：替换成功后原版本将从普通资源列表中删除，仅保留新版本。",
                422,
            )
        idempotency_key = str(payload.get("idempotency_key") or "").strip()
        if not idempotency_key:
            idempotency_key = hashlib.sha256(json.dumps({
                "base_profile_id": base_id,
                "candidate_ids": sorted(ids),
                "strategy": str(payload.get("strategy") or "safe_incremental"),
                "publish_mode": publish_mode,
                "version_name": str(payload.get("version_name") or ""),
            }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM feedback_optimization_jobs WHERE idempotency_key=? "
                "AND status IN ('queued','preparing','running','building','updating','optimizing','validating','applying','cancelling','ready_to_apply') ORDER BY created_at DESC LIMIT 1",
                (idempotency_key,),
            ).fetchone()
        if existing is not None:
            return self._public(existing)
        compatibility_started = time.perf_counter()
        preview = self.compatibility(payload, include_rows=True)
        compatibility_seconds = time.perf_counter() - compatibility_started
        rows = list(preview.pop("_rows", []))
        if not preview["compatible"]:
            first_error = (preview.get("errors") or [{}])[0]
            raise ApiError(
                str(first_error.get("code") or "FEEDBACK_COMPATIBILITY_FAILED"),
                str(first_error.get("message") or "所选样本与基础 profile 不兼容。"),
                409,
                {"compatibility": preview},
            )
        payload.update(preview.get("update_flags") or {})
        payload["active_config_profile_id"] = preview.get("active_config_profile_id")
        payload["active_config_profile_version"] = preview.get("active_config_profile_version")
        if self._official_test_lock(rows):
            raise ApiError("OFFICIAL_TEST_FEEDBACK_LOCKED", "正式测试或冻结测试锁定状态下，反馈只能记录，不能启动优化。", 409)
        with self._connect() as conn:
            active = conn.execute(
                "SELECT job_id FROM feedback_optimization_jobs WHERE base_profile_id=? AND status IN ('queued','preparing','running','building','updating','optimizing','validating','applying','cancelling','ready_to_apply')",
                (base_id,),
            ).fetchone()
            if active:
                raise ApiError("FEEDBACK_OPTIMIZATION_BUSY", "该记忆库已有未完成的反馈优化任务。", 409)
        job = self.jobs.create("feedback_optimization")
        stages = self._initialize_stages(payload, rows)
        created_at = _utc_now()
        initial_timings = {
            "feedback_query_and_compatibility": {
                "seconds": round(float(compatibility_seconds), 6),
                "candidate_count": len(rows),
            },
            "job_preflight": {"seconds": round(float(time.perf_counter() - preflight_started), 6)},
        }
        constraints = {
            key: payload.get(key)
            for key in (
                "min_recall", "max_false_positive_rate", "max_memory_growth", "max_latency_growth",
                "publish_mode", "replace_confirm", "version_name", "note",
            )
        }
        now = created_at
        base_profile = self.domain_profiles.get_profile(
            base_id,
            str(rows[0].get("dataset") or ""),
            str(rows[0].get("category") or ""),
            str(rows[0].get("model_version") or ""),
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO feedback_optimization_jobs(job_id,base_profile_id,base_config_profile_id,base_config_profile_version,candidate_ids_json,candidate_count,strategy,objective,update_threshold,update_normal_memory,update_defect_memory,update_lightweight_calibrator,status,progress,stage,message,constraints_json,created_at,idempotency_key,current_stage_id,stages_json,timings_json,last_progress_at,heartbeat_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    job.job_id,
                    base_id,
                    payload.get("active_config_profile_id") or base_profile.get("config_profile_id"),
                    int(payload.get("active_config_profile_version") or base_profile.get("config_profile_version") or 0),
                    json.dumps(ids),
                    len(ids),
                    str(payload.get("strategy") or "safe_incremental"),
                    str(payload.get("objective") or "f1"),
                    int(bool(payload.get("update_threshold", True))),
                    int(bool(payload.get("update_normal_memory", False))),
                    int(bool(payload.get("update_defect_memory", False))),
                    int(bool(payload.get("update_lightweight_calibrator", False))),
                    "queued",
                    0.0,
                    "等待开始",
                    "已创建反馈优化任务",
                    json.dumps(constraints),
                    now,
                    idempotency_key,
                    "",
                    json.dumps(stages, ensure_ascii=False),
                    json.dumps(initial_timings, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            for item in rows:
                role = "normal" if item.get("corrected_label") == "normal" else "defect" if item.get("corrected_label") == "defect" else "unresolved"
                source_bank_id = str(
                    item.get("source_memory_bank_id")
                    or item.get("memory_bank_id")
                    or item.get("base_memory_bank_id")
                    or ""
                )
                conn.execute(
                    """
                    INSERT INTO feedback_optimization_items(
                        job_id,candidate_id,role,status,source_memory_bank_id,strategy,updated_at
                    ) VALUES(?,?,?,?,?,?,?)
                    """,
                    (
                        job.job_id, item["candidate_id"], role, "queued",
                        source_bank_id, str(payload.get("strategy") or "safe_incremental"), now,
                    ),
                )
        self.jobs.update(
            job,
            status="queued",
            stage="等待开始",
            progress=0.0,
            base_profile_id=base_id,
            candidate_ids=ids,
            rerun_payload={
                "base_profile_id": base_id,
                "candidate_ids": ids,
                "strategy": str(payload.get("strategy") or "safe_incremental"),
                "objective": str(payload.get("objective") or "f1"),
                **(preview.get("update_flags") or {}),
                **constraints,
                "idempotency_key": idempotency_key,
            },
        )
        self._set_candidate_status(ids, "queued", job.job_id, "已加入反馈优化任务")
        self._insert_audit(job.job_id, "optimization_created", {"base_profile_id": base_id, "candidate_count": len(ids)})
        cancel = threading.Event()
        with self._lock:
            self._cancel[job.job_id] = cancel
            thread = threading.Thread(
                target=self._run,
                args=(job, dict(payload), rows, cancel),
                name=f"AdaptiveAOI-Feedback-{job.job_id[-10:]}",
                daemon=False,
            )
            self._threads[job.job_id] = thread
            thread.start()
        return self._public(self._row(job.job_id))

    def rerun(self, job_id: str) -> dict[str, Any]:
        row = self._row(job_id)
        if str(row["status"] or "") not in {"interrupted", "cancelled", "failed", "validation_failed"}:
            raise ApiError("FEEDBACK_JOB_NOT_RERUNNABLE", "只有已中断、已取消或失败的反馈更新任务可以重新运行。", 409)
        ids = [str(value) for value in _json(row["candidate_ids_json"], [])]
        constraints = _json(row["constraints_json"], {})
        # A failed/cancelled worker never consumes the reviewed evidence.  Put
        # every still-owned row back into the reviewed pool before the same
        # compatibility gates are evaluated again.
        self._set_candidate_status(ids, "accepted", job_id, "重新运行前恢复为已审核样本")
        payload = {
            "base_profile_id": str(row["base_profile_id"]),
            "candidate_ids": ids,
            "strategy": str(row["strategy"] or "safe_incremental"),
            "objective": str(row["objective"] or "f1"),
            "update_threshold": bool(row["update_threshold"]),
            "update_normal_memory": bool(row["update_normal_memory"]),
            "update_defect_memory": bool(row["update_defect_memory"]),
            "update_lightweight_calibrator": bool(row["update_lightweight_calibrator"]),
            **dict(constraints or {}),
            # Keep normal duplicate-click protection, while a deliberate
            # rerun receives a new operation key and therefore a new job.
            "idempotency_key": f"rerun:{job_id}:{uuid.uuid4().hex}",
        }
        created = self.create_job(payload)
        self.jobs.states.record_rerun(job_id, str(created["job_id"]))
        return created

    def _source_image(self, candidate: dict[str, Any], workdir: Path) -> Path | None:
        snapshot_ref = str(candidate.get("source_snapshot_ref") or "").strip()
        if snapshot_ref:
            snapshot_source, _snapshot_mask = self._snapshot_evidence(candidate)
            # A reviewed candidate with a declared snapshot must never fall
            # back to a mutable original path when the snapshot is missing or
            # fails its manifest/hash checks.
            if str(candidate.get("mode") or "") == "video" and snapshot_source is not None:
                return self._snapshot_video_frame(candidate, snapshot_source, workdir)
            return snapshot_source
        context = candidate.get("context") or {}
        source = str(context.get("_source_path") or "").strip()
        if candidate.get("mode") == "video":
            try:
                return self.video_jobs.extract_feedback_frame(
                    candidate["task_id"],
                    int(candidate.get("frame_index") or 0),
                    workdir / f"{candidate['candidate_id']}.png",
                )
            except Exception:
                return None
        if not source:
            result = self.storage.get_result(str(candidate.get("result_id") or ""))
            source = str((result or {}).get("source_path") or "")
        if not source:
            return None
        path = Path(source).expanduser().resolve()
        allowed = [
            self.settings.jobs_root.resolve(),
            self.settings.upload_root.resolve(),
            self.settings.project_root.resolve(),
        ]
        if not any(self._within(path, root) for root in allowed) or not path.is_file():
            return None
        return path

    def _snapshot_video_frame(
        self,
        candidate: dict[str, Any],
        snapshot_source: Path,
        workdir: Path,
    ) -> Path | None:
        """Return an image for old and new immutable video evidence snapshots.

        New snapshots contain ``source_frame.png``.  Releases before this fix
        could contain the complete source video as ``source.bin``.  That file is
        still immutable and hash-verified, so extract the reviewed frame from it
        instead of discarding historical evidence or falling back to a mutable
        task path.
        """
        import cv2

        if cv2.imread(str(snapshot_source), cv2.IMREAD_COLOR) is not None:
            return snapshot_source
        target = workdir / f"{candidate.get('candidate_id') or uuid.uuid4().hex}_source_frame.png"
        capture = cv2.VideoCapture(str(snapshot_source))
        try:
            if not capture.isOpened():
                return None
            frame_index = max(0, int(candidate.get("frame_index") or 0))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                return None
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f"{target.stem}.part{target.suffix}")
            if not cv2.imwrite(str(temporary), frame):
                return None
            temporary.replace(target)
            return target
        finally:
            capture.release()

    def _snapshot_evidence(self, candidate: dict[str, Any]) -> tuple[Path | None, Path | None]:
        """Resolve and verify one immutable review snapshot and optional Mask."""
        reference = str(candidate.get("source_snapshot_ref") or "").strip().replace("\\", "/")
        if not reference:
            return None, None
        root = (self.settings.jobs_root / reference).resolve()
        if not self._within(root, self.settings.jobs_root.resolve()) or not root.is_dir():
            return None, None
        manifest_path = root / "snapshot.json"
        if not manifest_path.is_file():
            return None, None
        expected_manifest_hash = str(candidate.get("source_snapshot_sha256") or "").strip().lower()
        if not expected_manifest_hash or _sha256(manifest_path).lower() != expected_manifest_hash:
            return None, None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return None, None
        verified: dict[str, Path] = {}
        for entry in manifest.get("files") or []:
            name = str(entry.get("name") or "")
            if not name or Path(name).name != name:
                return None, None
            path = (root / name).resolve()
            if not self._within(path, root) or not path.is_file():
                return None, None
            try:
                if int(entry.get("size_bytes") or -1) != int(path.stat().st_size):
                    return None, None
            except (TypeError, ValueError, OSError):
                return None, None
            if str(entry.get("sha256") or "").lower() != _sha256(path).lower():
                return None, None
            verified[name] = path
        source = next((path for name, path in verified.items() if name.startswith("source")), None)
        mask = verified.get("reviewed_mask.png")
        return source, mask

    @staticmethod
    def _within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _extract_features(
        self,
        candidate_rows: list[dict[str, Any]],
        resolved: dict[str, Any],
        workdir: Path,
        cancel: threading.Event,
        job_id: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        from core.image_io import prepare_image, preprocess_rgb
        import torch

        config = dict(resolved.get("config") or {})
        requested = str(config.get("device") or "cpu").lower()
        config["device"] = "cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu"
        config.setdefault("image_size", int(config.get("image_size") or 256))
        config["output_root"] = str(workdir)
        runtime_spec = {
            "model_id": str(resolved.get("model_id") or config.get("model_id") or candidate_rows[0].get("model_version") or ""),
            "dataset": str(resolved.get("dataset") or candidate_rows[0].get("dataset") or ""),
            "category": str(resolved.get("category") or candidate_rows[0].get("category") or ""),
            "device": str(config["device"]),
            "config": config,
            "memory_path": Path(resolved["memory_path"]),
            "calibration_path": resolved.get("calibration_path"),
            "defect_direction_path": resolved.get("defect_direction_path"),
            "defect_prototype_path": resolved.get("defect_prototype_path"),
            "lightweight_calibrator_path": resolved.get("lightweight_calibrator_path"),
        }
        self._set_stage(job_id, "runtime_loading", status="running", progress=0.1, message="正在获取当前目标域 Runtime")
        runtime_started = time.perf_counter()
        entry, cache_hit = self.registry.get_or_create(runtime_spec)
        runtime_seconds = time.perf_counter() - runtime_started
        self._record_timing(
            job_id,
            "runtime_loading",
            runtime_seconds,
            cache_hit=bool(cache_hit),
            model_load_seconds=0.0 if cache_hit else round(float(getattr(entry.inferencer, "model_load_ms", 0.0)) / 1000.0, 6),
            memory_bank_load_seconds=0.0 if cache_hit else round(float(getattr(entry.inferencer, "memory_bank_load_ms", 0.0)) / 1000.0, 6),
        )
        self._set_stage(
            job_id,
            "runtime_loading",
            status="completed",
            message="已复用当前 Runtime" if cache_hit else "Runtime 已加载并加入缓存",
        )
        self._check_cancel(cancel)
        outputs: list[dict[str, Any]] = []
        source_seconds = 0.0
        mask_seconds = 0.0
        feature_seconds = 0.0
        total_items = len(candidate_rows)
        self._set_stage(
            job_id,
            "feature_extraction",
            status="running",
            progress=0.0,
            message=f"正在读取纠错样本（0/{total_items}）",
            completed_items=0,
            total_items=total_items,
            current_item="",
        )
        with self.registry.execution(entry, workdir, resolved.get("threshold")) as inferencer:
            model = inferencer.model
            for index, candidate in enumerate(candidate_rows):
                self._check_cancel(cancel)
                context = dict(candidate.get("context") or {})
                current_name = str(
                    context.get("source_label")
                    or candidate.get("sample_id")
                    or candidate.get("candidate_id")
                    or f"样本 {index + 1}"
                )
                self._set_stage(
                    job_id,
                    "feature_extraction",
                    status="running",
                    progress=index / max(1, total_items),
                    message=f"正在处理 {index + 1}/{total_items}：{Path(current_name).name}",
                    completed_items=index,
                    total_items=total_items,
                    current_item=Path(current_name).name,
                )
                source_started = time.perf_counter()
                source = self._source_image(candidate, workdir)
                source_seconds += time.perf_counter() - source_started
                if source is None:
                    candidate["source_error"] = "样本源文件不可用"
                    self._set_stage(
                        job_id,
                        "feature_extraction",
                        status="running",
                        progress=(index + 1) / max(1, total_items),
                        message=f"已跳过无法读取样本 {index + 1}/{total_items}",
                        completed_items=index + 1,
                        total_items=total_items,
                        current_item=Path(current_name).name,
                    )
                    continue
                try:
                    prepared = prepare_image(source)
                except RuntimeError as exc:
                    candidate["source_error"] = "样本文件不是可读取图像"
                    candidate["source_error_detail"] = str(exc)
                    self._set_stage(
                        job_id,
                        "feature_extraction",
                        status="running",
                        progress=(index + 1) / max(1, total_items),
                        message=f"已跳过无法解码样本 {index + 1}/{total_items}",
                        completed_items=index + 1,
                        total_items=total_items,
                        current_item=Path(current_name).name,
                    )
                    continue
                tensor, _ = preprocess_rgb(prepared.image, int(config["image_size"]))
                started = time.perf_counter()
                with torch.inference_mode():
                    patches, _shape = model.extract(tensor.unsqueeze(0).to(model.device))
                    patch_array = patches[0].detach().cpu().numpy().astype(np.float32)
                    distances = model.memory_bank.score(patch_array)
                    distances = distances.detach().cpu().numpy() if hasattr(distances, "detach") else np.asarray(distances)
                elapsed = time.perf_counter() - started
                feature_seconds += elapsed
                mask_started = time.perf_counter()
                mask_path = self._annotation_mask_path(candidate)
                mask_seconds += time.perf_counter() - mask_started
                outputs.append(
                    {
                        "candidate": candidate,
                        "patches": patch_array,
                        "patch_distances": np.asarray(distances, dtype=np.float32),
                        "patch_shape": tuple(int(value) for value in _shape),
                        "mask_path": mask_path,
                        "source": str(source),
                        "score": float(candidate.get("original_score") or 0.0),
                        "feature_and_ann_ms": float(elapsed * 1000.0),
                    }
                )
                self._set_stage(
                    job_id,
                    "feature_extraction",
                    status="running",
                    progress=(index + 1) / max(1, len(candidate_rows)),
                    message=f"已处理 {index + 1}/{len(candidate_rows)} 个纠错样本",
                    completed_items=index + 1,
                    total_items=total_items,
                    current_item=Path(current_name).name,
                )
            base_features = np.asarray(model.memory_bank.features, dtype=np.float32)
            base_runtime_metadata = dict(model.memory_bank.runtime_metadata())
        self._record_timing(job_id, "original_image_read", source_seconds, candidate_count=len(candidate_rows))
        self._record_timing(job_id, "mask_read", mask_seconds, mask_count=sum(bool(item.get("mask_path")) for item in outputs))
        self._record_timing(job_id, "feedback_feature_extraction", feature_seconds, extracted_count=len(outputs))
        self._set_stage(job_id, "feature_extraction", status="completed", message=f"已提取 {len(outputs)} 个样本的特征")
        return outputs, {
            "cache_hit": bool(cache_hit),
            "base_features": base_features,
            "base_runtime_metadata": base_runtime_metadata,
        }

    def _annotation_mask_path(self, candidate: dict[str, Any]) -> Path | None:
        """Resolve only the latest server-owned mask artifact for a candidate."""
        annotations = candidate.get("annotations") or []
        for annotation in sorted(annotations, key=lambda item: int(item.get("revision") or 0), reverse=True):
            reference = str(annotation.get("mask_path") or annotation.get("mask_artifact_ref") or "")
            if not reference:
                continue
            path = (self.settings.jobs_root / reference).resolve()
            try:
                path.relative_to(self.settings.jobs_root.resolve())
            except ValueError:
                continue
            if path.is_file():
                return path
        _source, snapshot_mask = self._snapshot_evidence(candidate)
        if snapshot_mask is not None:
            return snapshot_mask
        return None

    def _clone_profile(
        self,
        base_path: Path,
        candidate_path: Path,
        candidate_id: str,
        base_meta: dict[str, Any],
        job_id: str,
        *,
        version_name: str = "",
        version_note: str = "",
    ) -> dict[str, Any]:
        if candidate_path.exists():
            shutil.rmtree(candidate_path)
        shutil.copytree(base_path, candidate_path)
        metadata_path = candidate_path / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(
            {
                "memory_bank_id": candidate_id,
                "memory_bank_name": str(version_name or f"反馈优化候选 · {candidate_id[-8:]}")[:80],
                "created_at": _utc_now(),
                "feedback_optimization_job_id": job_id,
                "feedback_parent_profile_id": base_meta.get("memory_bank_id"),
                "feedback_version_note": str(version_note or "")[:1000],
                "profile_kind": "feedback_candidate",
                "applied": False,
                "frozen": True,
                "test_data_update_allowed": False,
            }
        )
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata

    @staticmethod
    def _refresh_lock(
        profile_path: Path,
        metadata: dict[str, Any],
        *,
        precomputed_hashes: dict[str, str] | None = None,
    ) -> dict[str, str]:
        metadata_path = profile_path / "metadata.json"
        hashes = dict(metadata.get("file_sha256") or {})
        hashes.update(dict(precomputed_hashes or {}))
        for name in (
            "memory_bank.pt",
            "memory_bank.json",
            "coreset_indices.pt",
            "threshold.json",
            "effective_config.yaml",
            "defect_prototypes.pt",
            "lightweight_calibrator.json",
            "normal_samples_manifest.csv",
            "feedback_update_manifest.json",
        ):
            if (profile_path / name).is_file():
                if name in (precomputed_hashes or {}):
                    continue
                hashes[name] = _sha256(profile_path / name)
        metadata["file_sha256"] = hashes
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        (profile_path / "frozen.lock").write_text(
            json.dumps(
                {
                    "memory_bank_id": metadata.get("memory_bank_id"),
                    "frozen_at": _utc_now(),
                    "metadata_sha256": _sha256(metadata_path),
                    "write_policy": "inference_only",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return hashes

    def _update_normal_memory(
        self,
        extracted: list[dict[str, Any]],
        candidate_path: Path,
        resolved: dict[str, Any],
        metadata: dict[str, Any],
        constraints: dict[str, Any],
        *,
        cached_base_features: np.ndarray | None = None,
        cached_runtime_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # A reviewed normal sample carrying a defect Mask is not safe for the
        # normal bank: the patch boundary may still contain the defect.  Keep
        # it available for defect/lightweight calibration but exclude it here.
        def was_anomaly(item: dict[str, Any]) -> bool:
            value = str(item["candidate"].get("original_prediction") or "").strip().lower()
            return value in {"1", "anomaly", "abnormal", "defect", "ng", "异常", "缺陷"}

        normal = [
            item
            for item in extracted
            if item["candidate"].get("corrected_label") == "normal"
            and was_anomaly(item)
            and not item.get("mask_path")
        ]
        masked_normal_count = sum(
            1 for item in extracted
            if item["candidate"].get("corrected_label") == "normal" and item.get("mask_path")
        )
        non_anomaly_count = sum(
            1 for item in extracted
            if item["candidate"].get("corrected_label") == "normal" and not was_anomaly(item)
        )
        if not normal:
            return {
                "updated": False,
                "reason": "没有由误检修正且无缺陷 Mask 的已审核正常样本",
                "excluded_masked_normal_count": masked_normal_count,
                "excluded_non_anomaly_count": non_anomaly_count,
            }
        from core.memory_bank import MemoryBank, sample_patches
        import torch

        base_read_started = time.perf_counter()
        base_bank = None
        if cached_base_features is None:
            base_bank = MemoryBank.load(resolved["memory_path"], device="cpu")[0]
            base_features = np.asarray(base_bank.features, dtype=np.float32)
            runtime_metadata = dict(base_bank.runtime_metadata())
        else:
            base_features = np.asarray(cached_base_features, dtype=np.float32)
            runtime_metadata = dict(cached_runtime_metadata or {})
        base_read_seconds = time.perf_counter() - base_read_started
        # Normal-memory updates use bounded replacement rather than append.
        # The resulting bank keeps exactly the original entry budget, so ANN
        # query count and serialized growth cannot increase without an explicit
        # future migration.  At most 10% of the old coreset is replaced.
        replacement_ratio = min(0.10, max(0.0, _metric_or_default(constraints.get("max_memory_growth"), 0.05)))
        max_replace = max(1, int(len(base_features) * replacement_ratio))
        coreset_started = time.perf_counter()
        additions = [sample_patches(torch.from_numpy(item["patches"]), ratio=0.02, max_count=max_replace).numpy() for item in normal]
        added = np.concatenate(additions, axis=0)[:max_replace] if additions else np.empty((0, base_features.shape[1]), dtype=np.float32)
        if not len(added):
            return {"updated": False, "reason": "正常样本未产生可用特征"}
        replace_count = min(len(added), max_replace, len(base_features))
        generator = np.random.default_rng(42)
        keep_indices = generator.choice(
            len(base_features),
            size=max(0, len(base_features) - replace_count),
            replace=False,
        )
        replacement = np.asarray(added[:replace_count], dtype=np.float32)
        combined = np.concatenate([base_features[np.sort(keep_indices)], replacement], axis=0)
        coreset_seconds = time.perf_counter() - coreset_started
        ann_started = time.perf_counter()
        bank = MemoryBank(knn_k=int((resolved.get("config") or {}).get("knn_k", 1)), device="cpu", backend="auto").fit(combined)
        ann_seconds = time.perf_counter() - ann_started
        meta = runtime_metadata
        meta.update(
            metadata,
            feedback_incremental=True,
            feedback_added_feature_count=int(replace_count),
            feedback_replaced_feature_count=int(replace_count),
            feedback_base_feature_count=int(len(base_features)),
            feedback_memory_growth=0.0,
            feedback_memory_budget_preserved=True,
        )
        save_started = time.perf_counter()
        bank.save(candidate_path / "memory_bank.pt", meta)
        # The cloned coreset index belongs to the base bank.  Persist an index
        # for the candidate bank itself so future integrity checks never pair
        # the new features with stale base indices.
        torch.save(torch.arange(len(combined), dtype=torch.long), candidate_path / "coreset_indices.pt")
        save_seconds = time.perf_counter() - save_started
        return {
            "updated": True,
            "base_feature_count": len(base_features),
            "added_feature_count": int(replace_count),
            "replaced_feature_count": int(replace_count),
            "final_feature_count": int(len(combined)),
            "memory_growth": 0.0,
            "memory_budget_preserved": True,
            "excluded_masked_normal_count": masked_normal_count,
            "excluded_non_anomaly_count": non_anomaly_count,
            "timings": {
                "base_memory_read_seconds": round(base_read_seconds, 6),
                "coreset_seconds": round(coreset_seconds, 6),
                "ann_index_seconds": round(ann_seconds, 6),
                "memory_save_seconds": round(save_seconds, 6),
                "base_memory_cache_reused": cached_base_features is not None,
            },
        }

    def _update_defect_memory(self, extracted: list[dict[str, Any]], candidate_path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
        defect = [item for item in extracted if item["candidate"].get("corrected_label") == "defect"]
        if not defect:
            return {"updated": False, "reason": "没有已审核缺陷样本"}
        if str(metadata.get("method_mode_resolved") or "") != "normal_defect_dual_memory":
            return {"updated": False, "reason": "当前方法模式没有独立缺陷原型库"}
        from methods.adaptive_dual_memory import DefectPrototypeMemory, mine_defect_prototypes
        normal_items = [item for item in extracted if item["candidate"].get("corrected_label") == "normal"]
        normal_distances = (
            np.concatenate([item["patch_distances"] for item in normal_items], axis=0)
            if normal_items
            else np.asarray([0.0], dtype=np.float32)
        )
        defect_patches: list[np.ndarray] = []
        defect_distances: list[np.ndarray] = []
        masked_count = 0
        for item in defect:
            patches = np.asarray(item["patches"])
            distances = np.asarray(item["patch_distances"])
            selector = None
            if item.get("mask_path"):
                from methods.defect_calibration.prototype import mask_to_patch_selector

                try:
                    selector = mask_to_patch_selector(item["mask_path"], tuple(item["patch_shape"]))
                    masked_count += 1
                except (OSError, ValueError, TypeError):
                    selector = None
            else:
                from methods.defect_calibration.prototype import topk_selector

                try:
                    selector = topk_selector(distances.reshape(tuple(item["patch_shape"])), top_ratio=0.05, min_patches=4)
                except (TypeError, ValueError):
                    selector = np.zeros(len(distances), dtype=bool)
                    if len(distances):
                        keep = min(len(distances), max(4, int(len(distances) * 0.05)))
                        selector[np.argpartition(distances, -keep)[-keep:]] = True
            if selector is None:
                # A malformed annotation never becomes a reason to ingest an
                # entire defect image.  Fall back only to the highest-score
                # patches, which is the existing high-confidence path.
                selector = np.zeros(len(distances), dtype=bool)
                if len(distances):
                    keep = min(len(distances), max(4, int(len(distances) * 0.05)))
                    selector[np.argpartition(distances, -keep)[-keep:]] = True
            if selector is not None and bool(np.any(selector)):
                defect_patches.append(patches[selector])
                defect_distances.append(distances[selector])
        if not defect_patches:
            return {"updated": False, "reason": "缺陷样本没有可用的 Mask 或高置信 Patch"}
        memory = mine_defect_prototypes(
            defect_patches,
            defect_distances,
            normal_distances,
            max_prototypes=max(1, int(_metric_or_default(metadata.get("max_defect_prototypes"), 32))),
            min_image_support=1,
        )
        existing_path = candidate_path / "defect_prototypes.pt"
        if existing_path.is_file():
            try:
                old = DefectPrototypeMemory.load(existing_path)
                import torch

                prototypes = torch.cat([old.prototypes, memory.prototypes], dim=0)
                reliability = torch.cat([old.reliability, memory.reliability], dim=0)
                radius = torch.cat([old.radius, memory.radius], dim=0)
                keep = min(max(1, int(_metric_or_default(metadata.get("max_defect_prototypes"), 32))), int(prototypes.shape[0]))
                order = np.argsort(-reliability.numpy())[:keep]
                memory = DefectPrototypeMemory(
                    prototypes[order],
                    reliability[order],
                    radius[order],
                    old.normal_floor,
                    old.normal_scale,
                    {**old.metadata, "feedback_added": True},
                )
            except Exception:
                pass
        memory.save(existing_path)
        metadata["defect_prototype_count"] = int(memory.count)
        metadata["feedback_defect_prototype_update"] = True
        return {"updated": True, "prototype_count": int(memory.count), "masked_defect_count": masked_count}

    def _update_lightweight(self, extracted: list[dict[str, Any]], candidate_path: Path, metadata: dict[str, Any], resolved: dict[str, Any]) -> dict[str, Any]:
        normal = [item for item in extracted if item["candidate"].get("corrected_label") == "normal"]
        defect = [item for item in extracted if item["candidate"].get("corrected_label") == "defect"]
        if not normal or not defect:
            return {"updated": False, "reason": "轻量校准至少需要正常和缺陷样本"}
        from methods.lightweight_calibration import fit_lightweight_profile, write_lightweight_profile

        profile = fit_lightweight_profile(
            [item["patch_distances"] for item in normal],
            [item["patch_distances"] for item in defect],
            [item["score"] for item in normal],
            [item["score"] for item in defect],
            threshold=_metric_or_default(metadata.get("threshold"), 0.5),
            config=dict(resolved.get("config") or {}),
            seed=int((resolved.get("config") or {}).get("random_seed", 42)),
        )
        write_lightweight_profile(candidate_path / "lightweight_calibrator.json", profile)
        metadata["lightweight_calibration"] = True
        metadata["lightweight_calibrator_parameters"] = int(profile.get("parameter_count") or 0)
        return {"updated": True, "parameter_count": metadata["lightweight_calibrator_parameters"]}

    def _finish_active_stage(self, job_id: str, status: str, message: str) -> None:
        """Finish only the currently running stage without inventing progress."""
        detail = self._public(self._row(job_id))
        stage_id = str(detail.get("current_stage_id") or "")
        if stage_id:
            self._set_stage(job_id, stage_id, status=status, message=message)

    def _quarantine_candidate(self, candidate_path: Path | None, job_id: str) -> None:
        """Keep failed work auditable but outside the public resource catalog."""
        if candidate_path is None or not candidate_path.exists():
            return
        failed_root = self.settings.memory_banks_root / ".failed_feedback_updates"
        failed_root.mkdir(parents=True, exist_ok=True)
        target = failed_root / f"{job_id}_{candidate_path.name}"
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        shutil.move(str(candidate_path), str(target))

    def _run(self, job: JobPaths, payload: dict[str, Any], candidates: list[dict[str, Any]], cancel: threading.Event) -> None:
        job_id = job.job_id
        workdir = job.results_dir / "candidate_work"
        candidate_path: Path | None = None
        candidate_config_refs: list[tuple[str, int]] = []
        heartbeat_stop = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat_loop,
            args=(job_id, heartbeat_stop),
            name=f"AdaptiveAOI-FeedbackHeartbeat-{job_id[-8:]}",
            daemon=True,
        )
        heartbeat.start()
        try:
            base_id = str(payload.get("base_profile_id") or "").strip()
            candidate_ids = [str(row["candidate_id"]) for row in candidates]
            self._set_items_status(
                job_id,
                "validating",
                from_statuses={"queued"},
                message="正在校验标签、来源与更新策略",
                result_code="validation_started",
            )
            self._update(
                job_id,
                status="running",
                started_at=_utc_now(),
                message="正在校验已审核纠错样本",
                heartbeat_at=_utc_now(),
                last_progress_at=_utc_now(),
            )
            self._set_candidate_status([str(row["candidate_id"]) for row in candidates], "optimizing", job_id, "后台优化处理中")

            self._set_stage(job_id, "feedback_validation", status="running", progress=0.1, message="正在核对标签、来源快照和可用异常分数")
            validation_started = time.perf_counter()
            evidence = [
                row for row in candidates
                if row.get("original_score") is not None and row.get("corrected_label") in {"normal", "defect"}
            ]
            scores = [float(row["original_score"]) for row in evidence]
            labels = [1 if row.get("corrected_label") == "defect" else 0 for row in evidence]
            normal_count, defect_count = labels.count(0), labels.count(1)
            self._record_timing(job_id, "feedback_validation", time.perf_counter() - validation_started, normal_count=normal_count, defect_count=defect_count)
            self._prepare_processing_items(job_id, candidates, payload)
            self._set_stage(job_id, "feedback_validation", status="completed", message=f"已确认正常 {normal_count}、缺陷 {defect_count} 个样本")
            self._check_cancel(cancel)

            self._set_stage(job_id, "profile_compatibility", status="running", progress=0.1, message="正在解析基础目标域资源")
            profile_started = time.perf_counter()
            first = candidates[0]
            resolved = self.domain_profiles.resolve_for_inference(
                base_id,
                str(first.get("model_version") or ""),
                str(first.get("dataset") or ""),
                str(first.get("category") or ""),
            )
            # A Memory Bank may own several frozen configurations.  Feedback
            # evidence is bound to the configuration that produced it, which
            # can differ from the Bank metadata's initialization snapshot.
            # Overlay that immutable configuration before calibration so both
            # the baseline metric and the new candidate preserve its complete
            # parameter set; sibling configurations are copied unchanged later.
            active_config_id = str(payload.get("active_config_profile_id") or "")
            active_config_version = int(payload.get("active_config_profile_version") or 0)
            active_base_config: dict[str, Any] = {}
            if active_config_id and active_config_version:
                active_config_detail = self.config_profiles.get(active_config_id, active_config_version)
                active_base_config = dict(active_config_detail.get("config") or {})
                merged_runtime_config = dict(resolved.get("config") or {})
                merged_runtime_config.update(active_base_config)
                resolved["config"] = merged_runtime_config
                if active_config_detail.get("threshold") is not None:
                    resolved["threshold"] = float(active_config_detail["threshold"])
            base_path = Path(resolved["memory_path"]).parent
            base_metadata = json.loads((base_path / "metadata.json").read_text(encoding="utf-8"))
            base_mode = _profile_method_mode(
                base_metadata.get("method_mode_resolved")
                or base_metadata.get("method_mode_requested")
                or "normal_only"
            )
            reviewed_normal = [row for row in candidates if row.get("corrected_label") == "normal"]
            reviewed_defect = [row for row in candidates if row.get("corrected_label") == "defect"]
            safe_normal = [
                row for row in reviewed_normal
                if str(row.get("original_prediction") or "").strip().lower() in {"1", "anomaly", "abnormal", "defect", "ng", "异常", "缺陷"}
                and not bool(row.get("has_mask"))
                and not any(str(annotation.get("mask_path") or annotation.get("mask_artifact_ref") or "") for annotation in row.get("annotations") or [])
            ]
            masked_defect = [
                row for row in reviewed_defect
                if bool(row.get("has_mask"))
                or any(str(annotation.get("mask_path") or annotation.get("mask_artifact_ref") or "") for annotation in row.get("annotations") or [])
            ]
            eligibility = evaluate_strategy_eligibility(
                str(payload.get("strategy") or "threshold_only"),
                base_mode,
                {
                    "normal": normal_count,
                    "defect": defect_count,
                    "safe_normal": len(safe_normal),
                    "reliable_mask_defect": len(masked_defect),
                },
            )
            if eligibility.get("blockers"):
                message = "；".join(str(item.get("message") or "样本不满足当前策略准入条件") for item in eligibility["blockers"])
                self._set_stage(job_id, "feedback_validation", status="failed", message=message)
                self._update(
                    job_id,
                    status="validation_failed",
                    stage="纠错样本准入未通过",
                    message=message,
                    validation_json=json.dumps({"status": "eligibility_failed", "eligibility": eligibility}, ensure_ascii=False, default=str),
                    finished_at=_utc_now(),
                )
                self._set_candidate_status(candidate_ids, "accepted", job_id, "样本仍保留在已审核池，可补充符合当前策略的证据后重新运行")
                return
            hash_started = time.perf_counter()
            base_memory_hash = _sha256(base_path / "memory_bank.pt")
            self._record_timing(job_id, "base_memory_sha256_before", time.perf_counter() - hash_started)
            self._record_timing(job_id, "profile_compatibility", time.perf_counter() - profile_started)
            self._set_stage(job_id, "profile_compatibility", status="completed", message="基础模型、特征维度、数据集和类别兼容")

            self._set_stage(job_id, "prepare_normal", status="running", progress=0.5, message="正在整理已确认正常样本")
            self._set_stage(job_id, "prepare_normal", status="completed", message=f"已准备 {normal_count} 个正常证据")
            self._set_stage(job_id, "prepare_defect", status="running", progress=0.5, message="正在整理已确认缺陷样本")
            self._set_stage(job_id, "prepare_defect", status="completed", message=f"已准备 {defect_count} 个缺陷证据")
            self._check_cancel(cancel)

            resolved_threshold = resolved.get("threshold")
            if resolved_threshold is None:
                resolved_threshold = base_metadata.get("threshold")
            current_threshold = _metric_or_default(resolved_threshold, 0.5)
            before = self._metrics(scores, labels, current_threshold)
            # Candidate metadata is assembled before the parameter stage so
            # resource-specific workers can enrich it.  Until calibration
            # finishes the effective threshold remains the frozen baseline.
            new_threshold = current_threshold
            after = dict(before)
            threshold_options: list[dict[str, Any]] = []
            workdir.mkdir(parents=True, exist_ok=True)
            extracted: list[dict[str, Any]] = []
            runtime_context: dict[str, Any] = {}
            if any(payload.get(key) for key in ("update_normal_memory", "update_defect_memory", "update_lightweight_calibrator")):
                extracted, runtime_context = self._extract_features(candidates, resolved, workdir, cancel, job_id)
            # A missing source must be visible as an invalid candidate rather
            # than silently becoming an empty feature vector.  Threshold-only
            # evidence can still use the immutable score recorded by the
            # detection task; resource updates use only source-valid rows.
            invalid_rows = [row for row in candidates if row.get("source_error")]
            if invalid_rows:
                self._set_items_status(
                    job_id,
                    "skipped",
                    candidate_ids=[str(row["candidate_id"]) for row in invalid_rows],
                    from_statuses={"processing"},
                    message="原始图片或视频帧无法安全读取，本次未使用",
                    result_code="invalid_source",
                )
                self._set_candidate_status(
                    [str(row["candidate_id"]) for row in invalid_rows],
                    "invalid_source",
                    job_id,
                    "原始图片或视频帧无法安全读取",
                )
                candidates = [row for row in candidates if not row.get("source_error")]
                if not candidates:
                    self._finish_active_stage(job_id, "failed", "参与资源更新的反馈源文件均不可用")
                    self._update(
                        job_id,
                        status="validation_failed",
                        stage="源文件失效",
                        message="参与资源更新的反馈源文件均不可用，未创建候选 profile。",
                        validation_json=json.dumps({"status": "invalid_source", "invalid_count": len(invalid_rows)}, ensure_ascii=False),
                        finished_at=_utc_now(),
                    )
                    return
            self._check_cancel(cancel)

            self._set_stage(job_id, "profile_workspace", status="running", progress=0.05, message="正在复制冻结基础版本到隔离工作区")
            candidate_id = f"feedback_{uuid.uuid4().hex[:24]}"
            candidate_path = self.settings.memory_banks_root / ".feedback_candidates" / candidate_id
            workspace_started = time.perf_counter()
            metadata = self._clone_profile(
                base_path,
                candidate_path,
                candidate_id,
                base_metadata,
                job_id,
                version_name=str(payload.get("version_name") or ""),
                version_note=str(payload.get("note") or ""),
            )
            self._record_timing(job_id, "profile_workspace", time.perf_counter() - workspace_started, bytes=int(sum(path.stat().st_size for path in candidate_path.rglob("*") if path.is_file())))
            self._set_stage(job_id, "profile_workspace", status="completed", message="候选版本工作区已创建，原冻结版本未修改")
            target_mode = _target_method_mode(base_mode, payload.get("strategy"))
            method_names = {
                "normal_only": "正常记忆异常检测",
                "normal_defect_calibration": "正常记忆与缺陷阈值校准",
                "normal_memory_defect_guided_lite": "单正常记忆·缺陷引导轻量校准",
                "normal_defect_dual_memory": "可靠性门控双记忆异常检测",
            }
            metadata.update(
                threshold=float(new_threshold),
                threshold_method="feedback_reviewed_constrained_search",
                method_mode_requested=target_mode,
                method_mode_resolved=target_mode,
                method_name=method_names.get(target_mode, "目标域异常检测"),
                feedback_base_method_mode=base_mode,
                feedback_target_method_mode=target_mode,
                feedback_candidate_count=len(candidates),
                feedback_candidate_ids=[str(row["candidate_id"]) for row in candidates],
                feedback_metrics_before=before,
                feedback_metrics_after=after,
                feedback_publish_mode=str(payload.get("publish_mode") or "update_current"),
                feedback_base_memory_sha256=base_memory_hash,
                evaluation_mode="calibration_only",
                calibration_result={
                    "data_role": "reviewed_feedback_calibration_set",
                    "metrics_before": before,
                    "metrics_after": after,
                    "sample_count": len(candidates),
                },
                validation_result={
                    "status": "not_run",
                    "data_role": "independent_validation_required",
                    "message": "审核反馈样本用于校准，不作为最终测试精度；尚未运行独立验证集。",
                },
                official_test_mode=False,
                test_data_update_allowed=False,
            )

            feedback_manifest = {
                "schema_version": 1,
                "job_id": job_id,
                "base_profile_id": base_id,
                "base_memory_sha256": base_memory_hash,
                "created_at": _utc_now(),
                "samples": [
                    {
                        "candidate_id": str(row.get("candidate_id") or ""),
                        "source_profile_id": str(row.get("profile_id") or ""),
                        "mode": str(row.get("mode") or ""),
                        "issue_tag": str(row.get("issue_tag") or ""),
                        "corrected_label": str(row.get("corrected_label") or ""),
                        "source_snapshot_sha256": str(row.get("source_snapshot_sha256") or ""),
                        "has_mask": bool(
                            row.get("has_mask")
                            or row.get("latest_annotation_id")
                            or any(
                                str(annotation.get("mask_path") or annotation.get("mask_artifact_ref") or "")
                                for annotation in row.get("annotations") or []
                            )
                        ),
                        "official_test": bool(row.get("official_test")),
                    }
                    for row in candidates
                ],
            }
            (candidate_path / "feedback_update_manifest.json").write_text(
                json.dumps(feedback_manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            metadata["feedback_update_manifest"] = "feedback_update_manifest.json"
            metadata["normal_samples_manifest_scope"] = "base_profile_sources"
            updates: dict[str, Any] = {}
            constraints = dict(payload)
            if payload.get("update_normal_memory"):
                self._set_stage(job_id, "normal_memory_update", status="running", progress=0.1, message="正在受限预算内替换正常特征")
                update_started = time.perf_counter()
                updates["normal_memory"] = self._update_normal_memory(
                    extracted,
                    candidate_path,
                    resolved,
                    metadata,
                    constraints,
                    cached_base_features=runtime_context.get("base_features"),
                    cached_runtime_metadata=runtime_context.get("base_runtime_metadata"),
                )
                normal_timing = dict((updates["normal_memory"] or {}).get("timings") or {})
                self._record_timing(job_id, "normal_memory_update", time.perf_counter() - update_started, **normal_timing)
                normal_status = "completed" if updates["normal_memory"].get("updated") else "warning"
                self._set_stage(job_id, "normal_memory_update", status=normal_status, message="正常特征资源已更新" if normal_status == "completed" else str(updates["normal_memory"].get("reason") or "没有安全可用的正常样本"))
                self._set_stage(job_id, "ann_index", status="running", progress=0.8, message="正在确认候选检索索引")
                self._set_stage(job_id, "ann_index", status=normal_status, message="候选检索索引已生成" if normal_status == "completed" else "正常特征未变化，无需替换索引")
                self._record_timing(job_id, "ann_index", float(normal_timing.get("ann_index_seconds") or 0.0))
            if payload.get("update_defect_memory"):
                self._set_stage(job_id, "defect_reference_update", status="running", progress=0.1, message="正在从 Mask 或高置信区域提取缺陷参考")
                defect_started = time.perf_counter()
                updates["defect_memory"] = self._update_defect_memory(extracted, candidate_path, metadata)
                self._record_timing(job_id, "defect_reference_update", time.perf_counter() - defect_started)
                defect_status = "completed" if updates["defect_memory"].get("updated") else "warning"
                self._set_stage(job_id, "defect_reference_update", status=defect_status, message="缺陷参考特征已更新" if defect_status == "completed" else str(updates["defect_memory"].get("reason") or "没有可用缺陷参考"))
            self._check_cancel(cancel)

            self._set_stage(job_id, "parameter_calibration", status="running", progress=0.1, message="正在重新计算阈值与轻量判定参数")
            calibration_started = time.perf_counter()
            if payload.get("update_threshold", True):
                new_threshold, after, threshold_options = self._choose_threshold(
                    candidates,
                    current_threshold,
                    str(payload.get("objective") or "f1"),
                    _metric_or_default(payload.get("min_recall"), 0.95),
                    _metric_or_default(payload.get("max_false_positive_rate"), 0.10),
                )
            else:
                new_threshold, after, threshold_options = current_threshold, dict(before), []
            before["threshold_before"] = float(current_threshold)
            after["threshold_after"] = float(new_threshold)
            after["threshold_delta"] = float(new_threshold - current_threshold)
            after["stability"] = self._bootstrap_stability(scores, labels, new_threshold)
            metadata["threshold"] = float(new_threshold)
            metadata["feedback_metrics_before"] = before
            metadata["feedback_metrics_after"] = after
            if payload.get("update_lightweight_calibrator"):
                updates["lightweight_calibrator"] = self._update_lightweight(extracted, candidate_path, metadata, resolved)
            self._record_timing(job_id, "parameter_calibration", time.perf_counter() - calibration_started, threshold_before=current_threshold, threshold_after=new_threshold)
            self._update(job_id, metrics_before_json=json.dumps(before), metrics_after_json=json.dumps(after))
            calibration_warning = bool(payload.get("update_lightweight_calibrator")) and not bool((updates.get("lightweight_calibrator") or {}).get("updated"))
            self._set_stage(job_id, "parameter_calibration", status="warning" if calibration_warning else "completed", message="判定参数已重新计算" if not calibration_warning else str((updates.get("lightweight_calibrator") or {}).get("reason") or "轻量校准器未更新"))

            threshold_path = candidate_path / "threshold.json"
            if threshold_path.is_file():
                threshold_payload = _json(threshold_path.read_text(encoding="utf-8"), {})
                threshold_payload.update(
                    image_threshold=float(new_threshold),
                    threshold_strategy="feedback_reviewed_constrained_search",
                    method_mode_requested=target_mode,
                    method_mode_resolved=target_mode,
                    feedback_optimization_job_id=job_id,
                    frozen=True,
                    test_data_update_allowed=False,
                )
                threshold_path.write_text(json.dumps(threshold_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            effective_path = candidate_path / "effective_config.yaml"
            if effective_path.is_file():
                effective = yaml.safe_load(effective_path.read_text(encoding="utf-8")) or {}
                effective.update(active_base_config)
                effective.update(
                    threshold=float(new_threshold),
                    method_mode=target_mode,
                    method_mode_resolved=target_mode,
                    feedback_optimization_job_id=job_id,
                    inference_mode="inference_only",
                    memory_bank_write_policy="frozen_read_only",
                    test_data_update_allowed=False,
                )
                effective_path.write_text(yaml.safe_dump(effective, allow_unicode=True, sort_keys=False), encoding="utf-8")

            self._set_stage(job_id, "validation", status="running", progress=0.1, message="正在执行安全门禁与复杂度校验")
            offline_validation_started = time.perf_counter()
            candidate_memory_hash = _sha256(candidate_path / "memory_bank.pt")
            latency_values = np.asarray(
                [float(item.get("feature_and_ann_ms") or 0.0) for item in extracted if float(item.get("feature_and_ann_ms") or 0.0) > 0.0],
                dtype=float,
            )
            latency_metrics = {
                "sample_count": int(latency_values.size),
                "p50_ms": round(float(np.quantile(latency_values, 0.50)), 3) if latency_values.size else None,
                "p95_ms": round(float(np.quantile(latency_values, 0.95)), 3) if latency_values.size else None,
            }
            validation = {
                "status": "passed",
                "evaluation_mode": "calibration_only",
                "evidence_role": "reviewed_feedback_calibration_set",
                "final_test_metrics_available": False,
                "metrics_before": before,
                "metrics_after": after,
                "calibration_result": {"metrics_before": before, "metrics_after": after},
                "validation_result": {
                    "status": "not_run",
                    "data_role": "independent_validation_required",
                    "message": "当前准入检查使用校准证据，不代表独立测试性能。",
                },
                "threshold_options": threshold_options,
                "threshold_scope": {
                    "image_level": "optimized" if payload.get("update_threshold", True) else "unchanged",
                    "pixel_level": "unchanged_without_independent_pixel_validation",
                },
                "validation_method": "reviewed_evidence_plus_200_bootstrap",
                "base_method_mode": base_mode,
                "target_method_mode": target_mode,
                "mode_transition": base_mode != target_mode,
                "updates": updates,
                "eligibility": eligibility,
                "normal_count": normal_count,
                "defect_count": defect_count,
                "false_negative_rate_before": before.get("false_negative_rate"),
                "false_negative_rate_after": after.get("false_negative_rate"),
                "pixel_metrics": {
                    "status": "not_available",
                    "reason": "当前审核集没有独立的候选推理像素结果；人工 Mask 仅用于缺陷 Patch 提取，不虚构 IoU。",
                },
                "latency": latency_metrics,
                "base_memory_sha256_before": base_memory_hash,
                "base_memory_sha256_after": None,
                "candidate_memory_sha256": candidate_memory_hash,
                "base_memory_unchanged": None,
            }
            base_hash_after_started = time.perf_counter()
            base_memory_hash_after = _sha256(base_path / "memory_bank.pt")
            self._record_timing(job_id, "base_memory_sha256_after", time.perf_counter() - base_hash_after_started)
            validation["base_memory_sha256_after"] = base_memory_hash_after
            validation["base_memory_unchanged"] = base_memory_hash_after == base_memory_hash
            raw_memory_growth = (updates.get("normal_memory") or {}).get("memory_growth")
            memory_growth = _parse_metric_float(raw_memory_growth)
            # A strategy that does not update the normal bank has no memory
            # growth to measure.  An update strategy that omits the metric is
            # different: it is not evaluable and must not silently pass as 0.
            memory_growth_for_estimate = 0.0 if memory_growth is None and not payload.get("update_normal_memory") else memory_growth
            latency_growth_upper_bound = memory_growth_for_estimate
            validation["complexity_estimate"] = {
                "p95_latency_growth": round(float(memory_growth_for_estimate), 6) if memory_growth_for_estimate is not None else None,
                "measurement": "fixed_budget_memory_ratio_upper_bound",
            }
            max_growth_limit = min(0.10, max(0.0, _metric_or_default(payload.get("max_memory_growth"), 0.05)))
            validation["candidate_validation"] = {
                "ready": bool(eligibility.get("candidate_validation_ready")),
                "minimum_total_samples": int(eligibility.get("policy", {}).get("validation_min_total_samples") or 0),
                "requires_both_classes": bool(eligibility.get("policy", {}).get("validation_require_both_classes")),
                "normal_count": normal_count,
                "defect_count": defect_count,
            }
            failed_checks: list[dict[str, Any]] = []
            if not bool(eligibility.get("candidate_validation_ready")):
                failed_checks.append({
                    "code": "CANDIDATE_VALIDATION_EVIDENCE_INSUFFICIENT",
                    "message": "候选版本验证证据不足：Recall/FPR 验证需要两类标签且至少 5 条总证据；当前无法作为可冻结版本准入。",
                })
            else:
                min_recall = _metric_or_default(payload.get("min_recall"), 0.95)
                max_fpr = _metric_or_default(payload.get("max_false_positive_rate"), 0.10)
                max_latency = _metric_or_default(payload.get("max_latency_growth"), 0.10)
                actual_recall = _parse_metric_float(after.get("recall"))
                actual_fpr = _parse_metric_float(after.get("false_positive_rate"))
                if actual_recall is None:
                    failed_checks.append({
                        "code": "RECALL_NOT_EVALUABLE",
                        "message": "无法计算缺陷召回率：验证结果缺少有效召回率指标。",
                        "actual": None,
                        "required": min_recall,
                    })
                elif actual_recall < min_recall:
                    failed_checks.append({
                        "code": "RECALL_BELOW_MINIMUM",
                        "message": f"Recall {actual_recall:.3f} < 要求 {min_recall:.3f}",
                        "actual": actual_recall,
                        "required": min_recall,
                    })
                if actual_fpr is None:
                    fpr_reason = "当前验证样本没有正常样本，误报率不可计算。" if after.get("false_positive_rate_status") == "not_evaluable" else "验证结果缺少有效误报率指标。"
                    failed_checks.append({
                        "code": "FALSE_POSITIVE_RATE_NOT_EVALUABLE",
                        "message": f"无法验证误报率：{fpr_reason}",
                        "actual": None,
                        "required": max_fpr,
                    })
                elif actual_fpr > max_fpr:
                    failed_checks.append({
                        "code": "FALSE_POSITIVE_RATE_ABOVE_MAXIMUM",
                        "message": f"误报率 {actual_fpr:.3f} > 上限 {max_fpr:.3f}",
                        "actual": actual_fpr,
                        "required": max_fpr,
                    })
                if memory_growth is None and payload.get("update_normal_memory"):
                    failed_checks.append({
                        "code": "MEMORY_GROWTH_NOT_EVALUABLE",
                        "message": "无法验证记忆库增长：候选更新没有保存有效增长指标。",
                        "actual": None,
                        "required": max_growth_limit,
                    })
                elif memory_growth is not None and memory_growth > max_growth_limit + 1e-6:
                    failed_checks.append({
                        "code": "MEMORY_GROWTH_ABOVE_MAXIMUM",
                        "message": f"记忆库增长 {memory_growth:.3%} > 上限 {max_growth_limit:.3%}",
                        "actual": memory_growth,
                        "required": max_growth_limit,
                    })
                if latency_growth_upper_bound is None:
                    failed_checks.append({
                        "code": "LATENCY_GROWTH_NOT_EVALUABLE",
                        "message": "无法验证延迟增长：候选更新没有可用的增长指标。",
                        "actual": None,
                        "required": max_latency,
                    })
                elif latency_growth_upper_bound > max_latency + 1e-6:
                    failed_checks.append({
                        "code": "LATENCY_GROWTH_ABOVE_MAXIMUM",
                        "message": f"延迟增长上界 {latency_growth_upper_bound:.3%} > 上限 {max_latency:.3%}",
                        "actual": latency_growth_upper_bound,
                        "required": max_latency,
                    })
                if not bool(validation.get("base_memory_unchanged")):
                    failed_checks.append({
                        "code": "BASE_MEMORY_CHANGED",
                        "message": "基础冻结记忆库哈希发生变化",
                    })
            if failed_checks:
                validation["status"] = "failed"
                validation["failed_checks"] = failed_checks
                validation["reason"] = "候选版本验证未通过：" + "；".join(
                    str(item.get("message") or "存在未通过的验证条件") for item in failed_checks
                )
            if validation["status"] != "passed":
                self._set_stage(job_id, "validation", status="failed", message=str(validation.get("reason") or "候选版本未通过安全门禁"))
                self._quarantine_candidate(candidate_path, job_id)
                candidate_path = None
                self._update(job_id, status="validation_failed", stage="候选版本验证失败", message=str(validation.get("reason") or "候选方案未通过安全门禁"), validation_json=json.dumps(validation, ensure_ascii=False, default=str), finished_at=_utc_now())
                self._set_candidate_status(candidate_ids, "accepted", job_id, "候选版本未通过验证，样本可调整后重新运行")
                return
            metadata["feedback_validation"] = validation
            self._record_timing(job_id, "offline_validation", time.perf_counter() - offline_validation_started)
            self._set_stage(job_id, "validation", status="completed", message="候选版本通过当前校准证据与安全约束验证")

            complexity_before = {
                "memory_patch_count": (base_metadata.get("coreset") or {}).get("memory_patch_count"),
                "ann_query_count": 1 + int(base_metadata.get("additional_ann_count") or 0),
                "p95_latency_ms": base_metadata.get("p95_latency_ms"),
                "memory_bank_bytes": int((base_path / "memory_bank.pt").stat().st_size),
            }
            base_patch_count = int((updates.get("normal_memory") or {}).get("base_feature_count") or (base_metadata.get("coreset") or {}).get("memory_patch_count") or 0)
            complexity_after = {
                "memory_patch_count": int((updates.get("normal_memory") or {}).get("final_feature_count") or base_patch_count),
                "ann_query_count": 1 + int(metadata.get("additional_ann_count") or 0),
                "p95_latency_growth": round(latency_growth_upper_bound, 6),
                "p50_feature_and_ann_ms": latency_metrics.get("p50_ms"),
                "p95_feature_and_ann_ms": latency_metrics.get("p95_ms"),
                "memory_bank_bytes": int((candidate_path / "memory_bank.pt").stat().st_size),
                "p95_latency_measurement": "fixed_budget_memory_ratio_upper_bound",
            }
            current_job = self._row(job_id)
            base_cfg_id = str(current_job["base_config_profile_id"] or base_metadata.get("config_profile_id") or "")
            base_cfg_ver = int(current_job["base_config_profile_version"] or base_metadata.get("config_profile_version") or 0)
            candidate_cfg_id, candidate_cfg_ver = base_cfg_id, base_cfg_ver
            inherited_configs: list[dict[str, Any]] = []
            self._set_stage(job_id, "config_profile", status="running", progress=0.1, message="正在继承全部有效配置并创建候选 Profile")
            config_started = time.perf_counter()
            if base_cfg_id and base_cfg_ver:
                candidate_config = yaml.safe_load((candidate_path / "effective_config.yaml").read_text(encoding="utf-8")) or {}
                candidate_config.update(
                    target_domain_version=candidate_id,
                    build_job_id=job_id,
                    threshold=float(new_threshold),
                    method_mode=target_mode,
                    method_mode_resolved=target_mode,
                )
                inheritance = self.config_profiles.clone_all_for_feedback_candidate(
                    source_memory_bank_id=str(base_id),
                    target_memory_bank_id=candidate_id,
                    active_profile_id=base_cfg_id,
                    active_version=base_cfg_ver,
                    active_effective_config=candidate_config,
                    build_job_id=job_id,
                    note=str(payload.get("note") or "")[:500],
                )
                copied = dict(inheritance["active_config"])
                inherited_configs = [
                    {
                        "config_profile_id": str(item.get("config_profile_id") or ""),
                        "config_profile_version": int(item.get("version") or 1),
                        "friendly_name": str(item.get("friendly_name") or "未命名配置"),
                        "source_config_profile_id": str(item.get("source_config_profile_id") or ""),
                        "source_config_profile_version": int(item.get("source_config_profile_version") or 0),
                        "source_config_hash": str(item.get("source_config_hash") or ""),
                        "source_threshold": item.get("source_threshold"),
                        "active": bool(item.get("active")),
                    }
                    for item in inheritance.get("configs") or []
                ]
                candidate_config_refs = [
                    (str(item["config_profile_id"]), int(item["config_profile_version"]))
                    for item in inherited_configs
                ]
                candidate_cfg_id, candidate_cfg_ver = str(copied["config_profile_id"]), int(copied["version"])
            metadata["config_profile_id"] = candidate_cfg_id
            metadata["config_profile_version"] = candidate_cfg_ver
            metadata["config_profiles"] = inherited_configs
            metadata["config_count"] = len(inherited_configs) or (1 if candidate_cfg_id else 0)
            metadata["default_config_name"] = next(
                (str(item.get("friendly_name") or "未命名配置") for item in inherited_configs if item.get("active")),
                "当前绑定配置",
            )
            self._record_timing(job_id, "config_profile", time.perf_counter() - config_started)
            self._set_stage(
                job_id,
                "config_profile",
                status="completed",
                message=f"已继承并创建 {metadata['config_count']} 个独立配置；仅当前反馈配置写入新参数",
            )

            self._set_stage(job_id, "integrity", status="running", progress=0.1, message="正在计算资源校验值并生成冻结锁")
            integrity_started = time.perf_counter()
            self._refresh_lock(candidate_path, metadata, precomputed_hashes={"memory_bank.pt": candidate_memory_hash})
            self._record_timing(job_id, "integrity", time.perf_counter() - integrity_started, candidate_memory_sha256=candidate_memory_hash)
            self._set_stage(job_id, "integrity", status="completed", message="文件完整性与 SHA-256 校验完成")

            self._set_stage(job_id, "save_result", status="running", progress=0.2, message="正在提交候选版本结果")
            self._update(
                job_id,
                candidate_profile_id=candidate_id,
                candidate_config_profile_id=candidate_cfg_id,
                candidate_config_profile_version=candidate_cfg_ver,
                validation_json=json.dumps(validation, ensure_ascii=False, default=str),
                complexity_before_json=json.dumps(complexity_before, ensure_ascii=False, default=str),
                complexity_after_json=json.dumps(complexity_after, ensure_ascii=False, default=str),
            )
            self._set_stage(job_id, "save_result", status="completed", message="候选版本已保存，等待用户确认冻结或应用")
            detail = self._public(self._row(job_id))
            self._remember_stage_history(list(detail.get("stages") or []))
            # The terminal transition is deliberately last.  While resources,
            # validation and stage state are being committed the public
            # progress remains capped at 99%; 100% therefore means the job is
            # already a durable, usable candidate rather than merely that the
            # expensive feature stage has ended.
            self._set_items_status(
                job_id,
                "completed",
                from_statuses={"processing"},
                message="已成功用于候选新版本生成",
                result_code="candidate_version_generated",
            )
            self._update(
                job_id,
                status="ready_to_apply",
                progress=1.0,
                stage="结果与应用",
                message=(
                    "候选新版本已生成并验证通过；即将安全替换原版本。"
                    if str(payload.get("publish_mode") or "update_current") == "replace_original"
                    else "候选新版本已生成并验证通过；尚未替换当前运行方案。"
                ),
                finished_at=_utc_now(),
            )
            publish_mode = str(payload.get("publish_mode") or "update_current")
            if publish_mode == "replace_original":
                # Replacement is deliberately confirmed at creation time and
                # then completed by the worker only after validation.  The
                # resource service performs the atomic runtime switch before
                # archiving/deleting the old graph.
                try:
                    self.apply(job_id, confirm=True, actor="feedback_replace_worker")
                except ApiError:
                    # ``apply`` has already persisted a safe fallback state
                    # (ready/frozen) when runtime loading fails.  Do not let
                    # the outer worker handler overwrite that explanation as
                    # a generic build failure.
                    return
            else:
                self._record_usage_outcome(job_id, "ready_to_apply")
                self._set_candidate_status(
                    self._item_ids(job_id, "completed"),
                    "ready_to_apply",
                    job_id,
                    "已成功用于候选 profile，等待应用",
                )
            self._set_candidate_status(
                self._item_ids(job_id, "skipped"),
                "accepted",
                job_id,
                "本次更新未使用，仍保留在已审核样本池",
            )
            self._insert_audit(
                job_id,
                "optimization_ready_to_apply",
                {"candidate_profile_id": candidate_id, "validation": validation, "publish_mode": publish_mode},
            )
        except ApiError as exc:
            self.logger.warning("feedback_optimization_failed job_id=%s code=%s", job_id, exc.code)
            terminal = "cancelled" if exc.code == "FEEDBACK_OPTIMIZATION_CANCELLED" else "failed"
            self._finish_active_stage(job_id, "warning" if terminal == "cancelled" else "failed", exc.message)
            self._quarantine_candidate(candidate_path, job_id)
            for config_id, config_version in candidate_config_refs:
                try:
                    self.config_profiles.archive_draft(config_id, config_version)
                except (ApiError, OSError, ValueError):
                    pass
            self._set_candidate_status([str(row["candidate_id"]) for row in candidates], "accepted", job_id, "任务未消费样本，可重新运行")
            self._update(job_id, status=terminal, stage="已取消" if terminal == "cancelled" else "失败", message=exc.message, error_code=exc.code, error_message=exc.message, finished_at=_utc_now())
        except Exception as exc:
            self.logger.exception("feedback_optimization_failed job_id=%s", job_id)
            self._finish_active_stage(job_id, "failed", "反馈更新未完成")
            self._quarantine_candidate(candidate_path, job_id)
            for config_id, config_version in candidate_config_refs:
                try:
                    self.config_profiles.archive_draft(config_id, config_version)
                except (ApiError, OSError, ValueError):
                    pass
            self._set_candidate_status([str(row["candidate_id"]) for row in candidates], "accepted", job_id, "后台任务失败，样本可重新运行")
            self._update(job_id, status="failed", stage="失败", message="反馈更新未完成，请查看系统日志。", error_code="FEEDBACK_OPTIMIZATION_FAILED", error_message=f"{type(exc).__name__}: {exc}", finished_at=_utc_now())
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=1.0)
            with self._lock:
                self._cancel.pop(job_id, None)
                self._threads.pop(job_id, None)
            # If a source Memory Bank was deleted while this job was active,
            # the terminal task no longer needs its detection artifacts.  Run
            # the same idempotent convergence used at service startup so the
            # history lock is released without waiting for a restart.
            if self.runtime_resources is not None:
                try:
                    self.runtime_resources.reconcile_deleted_feedback_dependencies()
                except Exception:
                    self.logger.exception(
                        "feedback_terminal_protection_reconciliation_failed job_id=%s",
                        job_id,
                    )

    def status(self, job_id: str, *, include_items: bool = True, compact: bool = False) -> dict[str, Any]:
        row = self._row(job_id)
        item = self._public(row)
        if str(item.get("status") or "") in {
            "queued", "preparing", "running", "building", "updating", "optimizing", "validating", "applying", "cancelling",
        }:
            heartbeat_age = item.get("heartbeat_age_seconds")
            with self._lock:
                thread = self._threads.get(job_id)
                worker_alive = bool(thread and thread.is_alive())
            if heartbeat_age is not None and float(heartbeat_age) > 30.0 and not worker_alive:
                candidate_ids = [str(value) for value in _json(row["candidate_ids_json"], [])]
                self._finish_active_stage(job_id, "failed", "后台任务心跳已停止")
                self._set_candidate_status(candidate_ids, "accepted", job_id, "任务中断，样本可重新运行")
                self._update(
                    job_id,
                    status="interrupted",
                    stage="已中断",
                    message="后台任务心跳已停止，可重新运行该任务。",
                    error_code="FEEDBACK_OPTIMIZATION_INTERRUPTED",
                    error_message="后台任务心跳已停止",
                    finished_at=_utc_now(),
                )
                item = self._public(self._row(job_id))
        if not include_items:
            item["item_summary"] = self._item_summary(job_id)
            if bool(item.get("is_terminal")) and (
                not item.get("result") or (item.get("result") or {}).get("result_version") != RESULT_SCHEMA_VERSION
            ):
                item = self._store_result(job_id, item, item["item_summary"])
            if compact:
                for key in (
                    "candidate_ids", "metrics_before", "metrics_after", "complexity_before",
                    "complexity_after", "validation",
                ):
                    item.pop(key, None)
            return item
        with self._connect() as conn:
            tables = {str(value["name"]) for value in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            candidate_columns = {
                str(value["name"])
                for value in conn.execute("PRAGMA table_info(feedback_candidates)").fetchall()
            } if "feedback_candidates" in tables else set()
            sample_expr = "c.sample_id" if "sample_id" in candidate_columns else "NULL"
            label_expr = "c.corrected_label" if "corrected_label" in candidate_columns else "NULL"
            mode_expr = "c.mode" if "mode" in candidate_columns else "NULL"
            mask_expr = (
                "EXISTS(SELECT 1 FROM feedback_annotations a WHERE a.candidate_id=i.candidate_id "
                "AND COALESCE(a.mask_path,a.mask_artifact_ref,'')<>'')"
                if "feedback_annotations" in tables else "0"
            )
            values = conn.execute(
                f"SELECT i.*,{sample_expr} AS sample_id,{label_expr} AS corrected_label,{mode_expr} AS mode,"
                f"{mask_expr} AS has_mask "
                "FROM feedback_optimization_items i LEFT JOIN feedback_candidates c ON c.candidate_id=i.candidate_id "
                "WHERE i.job_id=? ORDER BY i.candidate_id",
                (job_id,),
            ).fetchall()
        public_items: list[dict[str, Any]] = []
        for value in values:
            detail = dict(value)
            detail["components"] = _json(detail.pop("components_json", "[]"), [])
            detail["status_display"] = get_feedback_item_status_display(detail.get("status"))
            detail["sample_name"] = str(detail.pop("sample_id", "") or "未命名样本")
            public_items.append(detail)
        item["items"] = public_items
        item["item_summary"] = self._item_summary(job_id)
        if bool(item.get("is_terminal")) and (
            not item.get("result") or (item.get("result") or {}).get("result_version") != RESULT_SCHEMA_VERSION
        ):
            item = self._store_result(job_id, item, item["item_summary"])
        return item

    def list_jobs(self, limit: int = 100, *, summary: bool = False) -> list[dict[str, Any]]:
        limit = min(200, max(1, int(limit)))
        with self._connect() as conn:
            if summary:
                rows = conn.execute(
                    "SELECT job_id,base_profile_id,candidate_profile_id,candidate_config_profile_id,"
                    "candidate_config_profile_version,status,progress,stage,message,strategy,update_normal_memory,"
                    "constraints_json,"
                    "metrics_before_json,metrics_after_json,complexity_before_json,complexity_after_json,"
                    "validation_json,created_at,started_at,finished_at,applied_at,error_code,error_message,result_json "
                    "FROM feedback_optimization_jobs ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM feedback_optimization_jobs ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        items = [self._public(row) for row in rows]
        job_ids = [str(item.get("job_id") or "") for item in items]
        summary_by_job: dict[str, dict[str, int]] = {}
        if job_ids:
            placeholders = ",".join("?" for _ in job_ids)
            with self._connect() as conn:
                counts = conn.execute(
                    f"SELECT job_id,status,COUNT(*) AS count FROM feedback_optimization_items "
                    f"WHERE job_id IN ({placeholders}) GROUP BY job_id,status",
                    job_ids,
                ).fetchall()
            for count in counts:
                job_summary = summary_by_job.setdefault(
                    str(count["job_id"]),
                    {key: 0 for key in (
                        "total", "queued", "validating", "processing", "completed", "skipped",
                        "failed", "cancelled", "interrupted", "legacy_state_unknown",
                    )},
                )
                state = str(count["status"] or "queued")
                value = int(count["count"] or 0)
                job_summary[state] = job_summary.get(state, 0) + value
                job_summary["total"] += value
        for item in items:
            item["item_summary"] = summary_by_job.get(
                str(item.get("job_id") or ""),
                {key: 0 for key in (
                    "total", "queued", "validating", "processing", "completed", "skipped",
                    "failed", "cancelled", "interrupted", "legacy_state_unknown",
                )},
            )
            if bool(item.get("is_terminal")) and (
                not item.get("result") or (item.get("result") or {}).get("result_version") != RESULT_SCHEMA_VERSION
            ):
                item = self._store_result(str(item.get("job_id") or ""), item, item["item_summary"])
            if summary:
                keep = {
                    "job_id", "base_profile_id", "candidate_profile_id", "status", "progress", "stage",
                    "message", "created_at", "started_at", "finished_at", "applied_at", "error_code",
                    "task_status", "task_status_display", "status_group", "status_icon", "severity", "result",
                    "is_terminal", "can_retry", "can_cancel", "can_apply", "item_summary",
                }
                for key in tuple(item):
                    if key not in keep:
                        item.pop(key, None)
        return items

    def profile_metadata(self, profile_id: str) -> dict[str, Any]:
        """Read one profile manifest directly without scanning the full catalog."""
        profile_id = str(profile_id or "").strip()
        if not profile_id:
            return {}
        candidates = (
            self.settings.memory_banks_root / profile_id / "metadata.json",
            self.settings.memory_banks_root / ".feedback_candidates" / profile_id / "metadata.json",
        )
        for path in candidates:
            try:
                if path.is_file():
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    return dict(payload) if isinstance(payload, dict) else {}
            except (OSError, ValueError, TypeError):
                continue
        return {}

    def cancel(self, job_id: str) -> dict[str, Any]:
        row = self._row(job_id)
        if row["status"] not in {"queued", "running"}:
            return self._public(row)
        with self._lock:
            event = self._cancel.get(job_id)
            if event:
                event.set()
        self._update(job_id, status="cancelling", stage="正在取消", message="正在安全停止后台任务")
        return self.status(job_id)

    def _runtime_resource_for_profile(self, profile_id: str) -> dict[str, Any] | None:
        if self.runtime_resources is None:
            return None
        resource = next(
            (
                item
                for item in self.runtime_resources.list_resources(resource_type="target_profile")
                if str(item.get("profile_id") or "") == str(profile_id)
            ),
            None,
        )
        if resource is not None:
            return resource
        self.runtime_resources.rebuild_index()
        return next(
            (
                item
                for item in self.runtime_resources.list_resources(resource_type="target_profile")
                if str(item.get("profile_id") or "") == str(profile_id)
            ),
            None,
        )

    def _publish_candidate_files(self, row: sqlite3.Row, actor: str) -> tuple[Path, dict[str, Any]]:
        """Freeze a validated candidate as a new public version without activation."""
        candidate_id = str(row["candidate_profile_id"] or "")
        hidden = self.settings.memory_banks_root / ".feedback_candidates" / candidate_id
        public = self.settings.memory_banks_root / candidate_id
        if public.is_dir():
            metadata = json.loads((public / "metadata.json").read_text(encoding="utf-8"))
            return public, metadata
        if not hidden.is_dir() or public.exists():
            raise ApiError("FEEDBACK_CANDIDATE_PROFILE_MISSING", "候选 profile 文件不完整或目标版本名称冲突。", 409)
        metadata = json.loads((hidden / "metadata.json").read_text(encoding="utf-8"))
        validation = _json(row["validation_json"], {})
        if str(validation.get("status") or "") != "passed" or not bool(validation.get("base_memory_unchanged")):
            raise ApiError("FEEDBACK_VALIDATION_REQUIRED", "候选 profile 尚未通过完整离线验证。", 409)
        candidate_cfg_id = str(row["candidate_config_profile_id"] or metadata.get("config_profile_id") or "")
        candidate_cfg_ver = int(row["candidate_config_profile_version"] or metadata.get("config_profile_version") or 0)
        if candidate_cfg_id and candidate_cfg_ver:
            config_refs = list(metadata.get("config_profiles") or [])
            if not config_refs:
                config_refs = [{
                    "config_profile_id": candidate_cfg_id,
                    "config_profile_version": candidate_cfg_ver,
                    "friendly_name": str(metadata.get("default_config_name") or metadata.get("memory_bank_name") or "初始化配置"),
                    "active": True,
                }]
            active_effective_path = hidden / "effective_config.yaml"
            active_effective = yaml.safe_load(active_effective_path.read_text(encoding="utf-8")) or {}
            frozen_refs: list[dict[str, Any]] = []
            active_frozen: dict[str, Any] | None = None
            active_hash = ""
            for ref in config_refs:
                ref_id = str(ref.get("config_profile_id") or "")
                ref_version = int(ref.get("config_profile_version") or 0)
                if not ref_id or not ref_version:
                    raise ApiError("FEEDBACK_CANDIDATE_CONFIG_MISSING", "候选版本包含无效配置引用。", 409)
                cfg = self.config_profiles.get(ref_id, ref_version)
                bound_target = str(cfg.get("target_domain_version") or cfg.get("memory_bank_id") or "")
                if bound_target and bound_target != candidate_id:
                # Compatibility repair for candidates produced by the former
                # copy_version path: that path retained the base bank binding,
                # so the new immutable profile could never be frozen.  Create
                # one target-owned snapshot instead of weakening the binding
                # invariant or editing the old snapshot in place.
                    replacement_id = "snapshot_" + hashlib.sha256(
                        f"{candidate_id}:{ref_id}:v{ref_version}".encode("utf-8")
                    ).hexdigest()[:24]
                    replacement_config = dict(active_effective if ref.get("active") else (cfg.get("config") or {}))
                    replacement_config.update(config_profile_id=replacement_id, config_profile_version=1, target_domain_version=candidate_id, build_job_id=str(row["job_id"]))
                    cfg = self.config_profiles.create_from_template(
                        {
                            "config_profile_id": replacement_id,
                            "friendly_name": str(ref.get("friendly_name") or "初始化配置")[:80],
                            "note": "旧候选配置绑定兼容修复",
                            "source_kind": "target_domain_snapshot",
                            "config_origin": "feedback_inherited",
                            "config": replacement_config,
                            "target_domain_version": candidate_id,
                            "build_job_id": str(row["job_id"]),
                            "ordinary_visible": True,
                        },
                        template_profile_id=ref_id,
                        template_version=ref_version,
                    )
                    ref_id, ref_version = str(cfg["config_profile_id"]), int(cfg["version"])
                    if ref.get("active"):
                        candidate_cfg_id, candidate_cfg_ver = ref_id, ref_version
                        self._update(str(row["job_id"]), candidate_config_profile_id=ref_id, candidate_config_profile_version=ref_version)
                effective = dict(active_effective if ref.get("active") else (cfg.get("config") or {}))
                effective.update(config_profile_id=ref_id, config_profile_version=ref_version, target_domain_version=candidate_id, build_job_id=str(row["job_id"]))
                if ref.get("active"):
                    threshold_value = effective.get("threshold")
                    if threshold_value is None:
                        threshold_value = cfg.get("threshold")
                    if threshold_value is None:
                        threshold_value = metadata.get("threshold")
                else:
                    # A sibling config is inherited verbatim.  Never use the
                    # candidate's active threshold as its fallback, otherwise
                    # publishing one updated config silently mutates every
                    # other config owned by the new Memory Bank.
                    threshold_value = ref.get("source_threshold")
                    if threshold_value is None:
                        threshold_value = effective.get("threshold")
                    if threshold_value is None:
                        threshold_value = cfg.get("threshold")
                threshold = float(0.5 if threshold_value is None else threshold_value)
                if bool(cfg.get("frozen")) and str(cfg.get("memory_bank_id") or "") == candidate_id:
                    frozen = dict(cfg.get("config") or {})
                    cfg_hash = str(cfg.get("config_hash") or self.config_profiles.config_hash(frozen))
                else:
                    self.config_profiles.validate(ref_id, ref_version)
                    frozen, cfg_hash = self.config_profiles.frozen_snapshot(ref_id, ref_version, effective, threshold)
                    self.config_profiles.bind_and_freeze(ref_id, ref_version, candidate_id, threshold, frozen, cfg_hash)
                source_hash = str(ref.get("source_config_hash") or "")
                if not ref.get("active") and source_hash and cfg_hash != source_hash:
                    raise ApiError(
                        "FEEDBACK_INHERITED_CONFIG_CHANGED",
                        f"配置“{ref.get('friendly_name') or ref_id}”在继承过程中发生变化，候选版本未发布。",
                        409,
                    )
                frozen_ref = {
                    **ref,
                    "config_profile_id": ref_id,
                    "config_profile_version": ref_version,
                    "config_hash": cfg_hash,
                    "threshold": threshold,
                }
                frozen_refs.append(frozen_ref)
                if ref.get("active"):
                    active_frozen = frozen
                    active_hash = cfg_hash
                    candidate_cfg_id, candidate_cfg_ver = ref_id, ref_version
            if active_frozen is None:
                raise ApiError("FEEDBACK_ACTIVE_CONFIG_MISSING", "候选版本缺少当前默认配置。", 409)
            metadata.update(
                config_hash=active_hash,
                config_profile_id=candidate_cfg_id,
                config_profile_version=candidate_cfg_ver,
                config_profiles=frozen_refs,
                config_count=len(frozen_refs),
            )
            active_effective.update(
                active_frozen,
                config_hash=active_hash,
                config_profile_id=candidate_cfg_id,
                config_profile_version=candidate_cfg_ver,
                inference_mode="inference_only",
            )
            active_effective_path.write_text(yaml.safe_dump(active_effective, allow_unicode=True, sort_keys=False), encoding="utf-8")
        metadata.update(frozen=True, profile_status="frozen", frozen_at=_utc_now(), applied=False)
        self._refresh_lock(hidden, metadata)
        shutil.move(str(hidden), str(public))
        self.registry.invalidate(str(metadata.get("dataset") or "").lower(), str(metadata.get("category") or ""))
        if self.runtime_resources is not None:
            self.runtime_resources.rebuild_index()
        self._insert_audit(str(row["job_id"]), "optimization_frozen", {"candidate_profile_id": candidate_id, "actor": actor})
        return public, metadata

    def freeze(self, job_id: str, *, confirm: bool, actor: str = "operator") -> dict[str, Any]:
        self._require_test_write_allowed("profile_freeze")
        if not confirm:
            raise ApiError("FREEZE_CONFIRMATION_REQUIRED", "冻结候选版本前需要明确确认。", 422)
        runtime = self._runtime_state()
        if bool(runtime.get("official_test_mode")) or bool(runtime.get("feedback_optimization_locked")):
            raise ApiError("OFFICIAL_TEST_FEEDBACK_LOCKED", "当前处于正式测试锁定状态，不能冻结反馈优化版本。", 409)
        row = self._row(job_id)
        if row["status"] != "ready_to_apply":
            raise ApiError("FEEDBACK_JOB_NOT_READY", "只有通过离线验证的候选方案才能冻结。", 409)
        started = time.perf_counter()
        self._publish_candidate_files(row, actor)
        self._record_timing(job_id, "profile_registration", time.perf_counter() - started, action="freeze")
        self._update(job_id, status="frozen", stage="已冻结", message="新 profile 已冻结保存，但尚未切换当前运行方案。")
        self._record_usage_outcome(job_id, "frozen")
        self._set_candidate_status(self._item_ids(job_id, "completed", "ready_to_apply"), "consumed", job_id, "已用于冻结候选版本")
        return self.status(job_id)

    def apply(self, job_id: str, *, confirm: bool, actor: str = "operator") -> dict[str, Any]:
        self._require_test_write_allowed("profile_apply")
        if not confirm:
            raise ApiError("APPLY_CONFIRMATION_REQUIRED", "应用候选方案前需要明确确认。", 422)
        runtime = self._runtime_state()
        if bool(runtime.get("official_test_mode")) or bool(runtime.get("feedback_optimization_locked")):
            raise ApiError("OFFICIAL_TEST_FEEDBACK_LOCKED", "当前处于正式测试锁定状态，不能应用反馈优化版本。", 409)
        row = self._row(job_id)
        if row["status"] not in {"ready_to_apply", "frozen"}:
            raise ApiError("FEEDBACK_JOB_NOT_READY", "只有通过离线验证的候选方案才能应用。", 409)
        previous_status = str(row["status"])
        self._update(
            job_id,
            status="applying",
            stage="正在应用",
            message="正在原子切换当前运行方案；候选版本与原版本均不会被原地覆盖。",
            error_code=None,
            error_message=None,
            finished_at=None,
        )
        try:
            with self._connect() as conn:
                now = _utc_now()
                conn.execute(
                    "UPDATE feedback_optimization_jobs SET heartbeat_at=?,last_progress_at=? WHERE job_id=?",
                    (now, now, job_id),
                )
        except sqlite3.OperationalError:
            # Isolated legacy embeds may not have timing columns.  The formal
            # database is migrated before serving requests.
            pass
        apply_heartbeat_stop = threading.Event()
        apply_heartbeat = threading.Thread(
            target=self._heartbeat_loop,
            args=(job_id, apply_heartbeat_stop),
            name=f"feedback-apply-heartbeat-{job_id}",
            daemon=True,
        )
        apply_heartbeat.start()
        candidate_id = str(row["candidate_profile_id"] or "")
        try:
            publish_started = time.perf_counter()
            public, metadata = self._publish_candidate_files(row, actor)
            self._record_timing(job_id, "profile_registration", time.perf_counter() - publish_started, action="publish")
            base_profile_id = str(row["base_profile_id"])
            constraints = _json(row["constraints_json"], {})
            publish_mode = str(constraints.get("publish_mode") or "update_current")
            dataset, category, model_id = str(metadata.get("dataset") or ""), str(metadata.get("category") or ""), str(metadata.get("model_id") or "")
            previous = self.domain_profiles.get_profile(base_profile_id, dataset, category, model_id)
            previous_cfg_id, previous_cfg_ver = str(previous.get("config_profile_id") or ""), int(previous.get("config_profile_version") or 0)
            candidate_cfg_id = str(row["candidate_config_profile_id"] or metadata.get("config_profile_id") or "")
            candidate_cfg_ver = int(row["candidate_config_profile_version"] or metadata.get("config_profile_version") or 0)
            runtime_started = time.perf_counter()
            resource = self._runtime_resource_for_profile(candidate_id)
            replacement_cleanup: dict[str, Any] = {}
            old_resource: dict[str, Any] | None = None
            if publish_mode == "replace_original":
                if self.runtime_resources is None or not hasattr(self.runtime_resources, "replace_profile"):
                    raise ApiError("REPLACEMENT_SERVICE_UNAVAILABLE", "安全替换服务不可用，原版本未修改。", 503)
                old_resource = self._runtime_resource_for_profile(base_profile_id)
                if old_resource is None:
                    raise ApiError("REPLACEMENT_BASE_RESOURCE_MISSING", "原运行版本不在统一资源目录中，未执行替换。", 409)
                if str(self._runtime_state().get("memory_bank_id") or "") != base_profile_id:
                    raise ApiError("REPLACEMENT_BASE_NOT_CURRENT", "原版本已不是当前运行方案，未执行替换。", 409)
            if resource is not None and publish_mode != "replace_original":
                if not hasattr(self, "runtime_profiles"):
                    # Compatibility for isolated service tests/legacy embeds
                    # constructed with ``__new__``.  The formal application
                    # always attaches RuntimeProfileService in build_services.
                    self.runtime_resources.apply_profile(
                        str(resource["resource_id"]),
                        actor=actor,
                        feedback_job_id=job_id,
                    )
                elif self.runtime_profiles is None:
                    raise ApiError("RUNTIME_PROFILE_SERVICE_UNAVAILABLE", "运行 Profile 服务不可用，未切换当前方案。", 503)
                else:
                    self.runtime_profiles.apply_profile(
                        str(resource["resource_id"]),
                        actor=actor,
                        feedback_job_id=job_id,
                    )
            else:
                if resource is None:
                    raise ApiError("RUNTIME_PROFILE_NOT_INDEXED", "候选 Profile 尚未进入统一资源目录，未切换当前方案。", 409)
                replacement_cleanup = self.runtime_resources.replace_profile(
                    str(old_resource["resource_id"]),
                    str(resource["resource_id"]),
                    actor=actor,
                    feedback_job_id=job_id,
                )
            self._record_timing(job_id, "runtime_refresh", time.perf_counter() - runtime_started, candidate_profile_id=candidate_id)
            if publish_mode == "replace_original":
                self._update(
                    job_id,
                    replacement_old_resource_id=str(old_resource["resource_id"]),
                    replacement_cleanup_json=json.dumps(replacement_cleanup, ensure_ascii=False, default=str),
                )
            if candidate_cfg_id and candidate_cfg_ver:
                self.config_profiles.mark_applied(candidate_cfg_id, candidate_cfg_ver, True)
            applied_message = (
                "新 profile 已通过验证并安全替换原版本；原版本已从普通运行资源中移除，历史审计仍保留。"
                if publish_mode == "replace_original"
                else "候选 profile 已显式应用；原 profile 仍保留，可回滚。"
            )
            if replacement_cleanup.get("cleanup_status") == "archived_pending_delete":
                applied_message += " 原版本已归档但物理清理待维护任务完成。"
            completed_at = _utc_now()
            self._update(job_id, status="applied", applied_at=completed_at, finished_at=completed_at, previous_profile_id=base_profile_id, previous_config_profile_id=previous_cfg_id, previous_config_profile_version=previous_cfg_ver, message=applied_message)
            self._record_usage_outcome(job_id, "applied")
            self._set_candidate_status(self._item_ids(job_id, "completed", "ready_to_apply"), "applied", job_id, "候选 profile 已应用")
            self._insert_audit(job_id, "optimization_applied", {"candidate_profile_id": candidate_id, "actor": actor, "previous_profile_id": base_profile_id, "publish_mode": publish_mode, "replacement_cleanup": replacement_cleanup})
            return self.status(job_id)
        except Exception as exc:
            published = (self.settings.memory_banks_root / candidate_id).is_dir()
            fallback_status = "frozen" if published or previous_status == "frozen" else "ready_to_apply"
            error_code = exc.code if isinstance(exc, ApiError) else "FEEDBACK_APPLY_FAILED"
            self._update(
                job_id,
                status=fallback_status,
                stage="应用失败",
                message="版本生成成功，但应用失败；候选版本已保留，可重新应用。",
                error_code=error_code,
                error_message=str(getattr(exc, "message", exc)),
                finished_at=_utc_now(),
            )
            self._insert_audit(job_id, "optimization_apply_failed", {
                "candidate_profile_id": candidate_id,
                "error_code": error_code,
                "candidate_preserved": True,
            })
            if isinstance(exc, ApiError):
                raise
            raise ApiError("FEEDBACK_APPLY_FAILED", "候选 profile 应用失败，当前运行方案未切换；已冻结候选版本仍保留。", 500) from exc
        finally:
            apply_heartbeat_stop.set()
            apply_heartbeat.join(timeout=1.0)

    def rollback(self, job_id: str, *, confirm: bool = False, actor: str = "operator") -> dict[str, Any]:
        self._require_test_write_allowed("profile_rollback")
        if not confirm:
            raise ApiError("ROLLBACK_CONFIRMATION_REQUIRED", "回滚运行 profile 前需要明确确认。", 422)
        runtime = self._runtime_state()
        if bool(runtime.get("official_test_mode")) or bool(runtime.get("feedback_optimization_locked")):
            raise ApiError("OFFICIAL_TEST_FEEDBACK_LOCKED", "正式测试锁定状态下不能切换运行 profile。", 409)
        row = self._row(job_id)
        if row["status"] != "applied":
            raise ApiError("FEEDBACK_JOB_NOT_APPLIED", "只有已应用的反馈优化任务才能回滚。", 409)
        if str(_json(row["constraints_json"], {}).get("publish_mode") or "") == "replace_original":
            raise ApiError(
                "REPLACEMENT_ROLLBACK_UNAVAILABLE",
                "该任务已物理替换并删除原版本，不能回滚到已删除资源；请在运行配置中选择其他仍保留的冻结版本。",
                409,
            )
        previous_id = str(row["previous_profile_id"] or "")
        candidate_id = str(row["candidate_profile_id"] or "")
        if not previous_id:
            raise ApiError("FEEDBACK_ROLLBACK_TARGET_MISSING", "找不到原始 profile，无法回滚。", 409)
        candidate_public = self.settings.memory_banks_root / candidate_id
        if not candidate_public.is_dir():
            raise ApiError("FEEDBACK_CANDIDATE_PROFILE_MISSING", "当前候选 profile 不存在，无法回滚。", 409)
        candidate_meta = json.loads((candidate_public / "metadata.json").read_text(encoding="utf-8"))
        dataset, category, model_id = str(candidate_meta.get("dataset") or ""), str(candidate_meta.get("category") or ""), str(candidate_meta.get("model_id") or "")
        previous = self.domain_profiles.get_profile(previous_id, dataset, category, model_id)
        resource = self._runtime_resource_for_profile(previous_id)
        if resource is not None:
            if not hasattr(self, "runtime_profiles"):
                self.runtime_resources.apply_profile(str(resource["resource_id"]), actor=actor)
            elif self.runtime_profiles is None:
                raise ApiError("RUNTIME_PROFILE_SERVICE_UNAVAILABLE", "运行 Profile 服务不可用，未执行回滚。", 503)
            else:
                self.runtime_profiles.apply_profile(str(resource["resource_id"]), actor=actor)
        else:
            raise ApiError("RUNTIME_PROFILE_NOT_INDEXED", "原始 Profile 尚未进入统一资源目录，未执行回滚。", 409)
        if row["previous_config_profile_id"] and row["previous_config_profile_version"]:
            self.config_profiles.mark_applied(str(row["previous_config_profile_id"]), int(row["previous_config_profile_version"]), True)
        self._update(job_id, status="rolled_back", rolled_back_at=_utc_now(), message="已回滚到原始冻结 profile；候选版本仍保留。")
        self._record_usage_outcome(job_id, "rolled_back")
        self._set_candidate_status([str(value) for value in _json(row["candidate_ids_json"], [])], "rolled_back", job_id, "候选 profile 已回滚")
        self._insert_audit(job_id, "optimization_rolled_back", {"actor": actor, "restored_profile_id": previous_id})
        return self.status(job_id)

    def shutdown(self) -> None:
        """Request cancellation without deleting candidate/job artifacts."""
        with self._lock:
            events = list(self._cancel.values())
            threads = list(self._threads.values())
        for event in events:
            event.set()
        for thread in threads:
            thread.join(timeout=5.0)
