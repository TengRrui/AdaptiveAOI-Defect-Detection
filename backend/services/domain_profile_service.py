"""Immutable target-domain memory-bank lifecycle and asynchronous build jobs."""
from __future__ import annotations

import csv
import base64
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import secrets
import statistics
import sys
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, UnidentifiedImageError

from backend.errors import ApiError
from backend.path_validation import validate_dataset_category
from backend.services.video_sample_service import VideoSampleCancelled, VideoSampleService
from core.domain_build_contract import (
    CREATE_ALLOWED_MODES,
    CREATE_MODE_LABELS,
    DEFAULT_CREATE_MODE,
    LIGHTWEIGHT_CREATE_MODE,
    READ_COMPATIBLE_MODES,
    default_feature_layers,
    normalize_precision_request,
    precision_uses_fp16,
)


def _domain_adaptation_module():
    """Load the target-domain algorithm only for an actual domain operation.

    Importing this module initializes PyTorch and the feature stack.  Keeping
    it out of backend construction lets liveness become available before model
    warm-up without changing any algorithm implementation.
    """
    from methods.patchcore import domain_adaptation

    return domain_adaptation


PROFILE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{8,128}$")
SAMPLE_SCAN_RULE_VERSION = "image-verify-sha256-v2"
SYSTEM_MANAGED_MODEL_FIELDS = {
    "model_id",
    "model_version",
    "base_model_version",
    "backbone",
    "weights",
    "weights_sha256",
    "model_manifest",
    "model_manifest_sha256",
}
BUILD_ENVIRONMENT_WARMUP_TIMEOUT_SECONDS = 120.0
BUILD_ENVIRONMENT_TERMINATE_TIMEOUT_SECONDS = 2.0
CREATABLE_DOMAIN_METHOD_MODES = CREATE_ALLOWED_MODES
CREATABLE_DOMAIN_METHOD_LABELS = CREATE_MODE_LABELS
READ_COMPATIBLE_DOMAIN_METHOD_MODES = READ_COMPATIBLE_MODES


DOMAIN_BUILD_STAGES = (
    {"stage_id": "initialization", "stage_name": "任务初始化", "weight": 3.0},
    {"stage_id": "sample_validation", "stage_name": "样本扫描与合法性检查", "weight": 7.0},
    {"stage_id": "video_preparation", "stage_name": "视频抽帧、质量筛选与去重", "weight": 10.0},
    {"stage_id": "feature_extraction", "stage_name": "图像预处理与特征提取", "weight": 35.0},
    {"stage_id": "memory_index", "stage_name": "Memory Bank、coreset 与检索索引构建", "weight": 18.0},
    {"stage_id": "threshold_calibration", "stage_name": "判定阈值生成与校准", "weight": 10.0},
    {"stage_id": "defect_extension", "stage_name": "缺陷引导轻量校准", "weight": 8.0},
    {"stage_id": "config_profile", "stage_name": "Config 与目标域 Profile 生成", "weight": 4.0},
    # Integrity, immutable freeze, catalog registration and transaction commit
    # are one internal finalization boundary.  They remain individually logged
    # in messages/metadata but are not exposed as operator workflow steps.
    {"stage_id": "finalizing", "stage_name": "完成目标域资源创建", "weight": 5.0},
)

TERMINAL_BUILD_STATES = {"completed", "cancelled", "failed", "interrupted"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_token(value: str, fallback: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip()).strip("-._")
    return token[:36] or fallback


def _request_fingerprint(payload: dict[str, Any]) -> str:
    canonical = {
        str(key): value
        for key, value in dict(payload).items()
        if str(key) != "idempotency_key" and not str(key).startswith("_")
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class DomainBuildTask:
    job_id: str
    payload: dict[str, Any]
    idempotency_key: str = ""
    request_fingerprint: str = ""
    state: str = "queued"
    lifecycle_phase: str = "submitted"
    stage: str = "已提交"
    message: str = "目标域构建任务已提交，正在等待后台工作线程。"
    current: int = 0
    total: int = 0
    normal_processed: int = 0
    defect_processed: int = 0
    created_at: str = field(default_factory=_utc_now)
    api_received_at: str | None = None
    task_created_at: str | None = None
    queued_at: str | None = None
    worker_acquired_at: str | None = None
    first_progress_at: str | None = None
    build_started_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_seconds: float = 0.0
    error: dict[str, str] | None = None
    result: dict[str, Any] | None = None
    registration_status: str = "pending"
    resource_catalog_revision: int | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    future: Future[None] | None = None
    started_monotonic: float | None = None
    stages: list[dict[str, Any]] = field(default_factory=list)
    last_progress_at: str | None = None
    heartbeat_at: str | None = None
    heartbeat_thread: threading.Thread | None = None
    failed_stage: str | None = None
    dataset_manifest: dict[str, Any] | None = None


class DomainProfileService:
    """Owns target-domain scans, immutable profiles, and one build task at a time."""

    def __init__(
        self,
        settings: Any,
        registry: Any,
        logger: Any,
        *,
        config_profiles: Any,
        jobs: Any | None = None,
        models: Any | None = None,
    ):
        self.settings = settings
        self.registry = registry
        self.logger = logger
        self.config_profiles = config_profiles
        self.jobs = jobs
        self.models = models
        self.root = settings.memory_banks_root.resolve()
        self.archive_root = self.root / "_archive"
        self.draft_root = (settings.jobs_root / "domain_drafts").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.draft_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._tasks: dict[str, DomainBuildTask] = {}
        self._tasks_by_idempotency: dict[str, str] = {}
        self._active_job_id: str | None = None
        self._reservations: dict[str, set[str]] = {}
        self.runtime_resources: Any | None = None
        self.test_lock: Any | None = None
        self.failed_build_root = self.root / "_failed_builds"
        self.failed_build_root.mkdir(parents=True, exist_ok=True)
        self.video_samples = VideoSampleService(settings.jobs_root / "domain_video_samples")
        self.stage_history_path = (settings.jobs_root / "domain_stage_history.json").resolve()
        self._stage_history = self._load_stage_history()
        self._sample_scan_cache: OrderedDict[tuple[str, str, str], dict[str, Any]] = OrderedDict()
        self._sample_scan_cache_lock = threading.RLock()
        self._sample_scan_cache_limit = 16
        self.feedback_candidates: Any | None = None
        self._build_environment_warmup_lock = threading.RLock()
        self._build_environment_warmup_thread: threading.Thread | None = None
        self._build_environment_warmup_process: subprocess.Popen[Any] | None = None
        self._build_environment_warmup_done = threading.Event()
        self._build_environment_warmup_stop = threading.Event()
        self._build_environment_warmup_state = "not_started"
        self._build_environment_warmup_error = ""
        # Reuse one long-lived execution thread for GPU target-domain builds.
        # Repeatedly tearing down a CUDA-owning Python thread can leave the
        # Windows IOCP loop unable to accept further API traffic even after the
        # durable task status is complete.  One build is already the product
        # concurrency contract, so a single persistent worker is sufficient.
        self._build_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="AdaptiveAOI-Domain",
        )
        # Draft deletion uses the same preview/acknowledgement contract as
        # other destructive resources.  Plans are intentionally short-lived
        # and bound to the draft file fingerprint.
        self._draft_delete_plans: dict[str, dict[str, Any]] = {}
        self._draft_revision_path = (self.draft_root / ".revision").resolve()
        self._draft_revision = self._load_draft_revision()

    def _load_draft_revision(self) -> int:
        try:
            return max(0, int(self._draft_revision_path.read_text(encoding="utf-8").strip() or 0))
        except (OSError, ValueError):
            return 0

    def _bump_draft_revision(self) -> int:
        self._draft_revision = max(0, int(self._draft_revision or 0)) + 1
        temporary = self._draft_revision_path.with_suffix(".tmp")
        try:
            temporary.write_text(str(self._draft_revision), encoding="utf-8")
            os.replace(temporary, self._draft_revision_path)
        except OSError as exc:
            self.logger.warning("domain_draft_revision_write_failed error=%s", exc)
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        return self._draft_revision

    def draft_revision(self) -> int:
        with self._lock:
            return int(self._draft_revision)

    def _draft_artifacts(self, draft_id: str) -> list[Path]:
        """Return only artifacts owned by a draft under the draft root."""
        path = self._draft_path(draft_id)
        artifacts: list[Path] = [path] if path.is_file() else []
        temporary = path.with_suffix(".tmp")
        if temporary.is_file():
            artifacts.append(temporary)
        directory = (self.draft_root / str(draft_id)).resolve()
        try:
            directory.relative_to(self.draft_root.resolve())
        except ValueError:
            return artifacts
        if directory.is_dir():
            artifacts.extend(item for item in directory.rglob("*") if item.is_file())
        return artifacts

    def _draft_fingerprint(self, draft_id: str) -> str:
        digest = hashlib.sha256()
        root = self.draft_root.resolve()
        for path in sorted(self._draft_artifacts(draft_id), key=lambda item: item.as_posix()):
            try:
                stat = path.stat()
                relative = path.resolve().relative_to(root).as_posix()
                digest.update(relative.encode("utf-8", errors="surrogatepass"))
                digest.update(str(stat.st_size).encode("ascii"))
                digest.update(str(stat.st_mtime_ns).encode("ascii"))
            except (OSError, ValueError):
                continue
        return digest.hexdigest()

    def _active_draft_protection(self, draft_id: str) -> list[dict[str, Any]]:
        """Find live build tasks linked to a draft, including durable state."""
        active: list[dict[str, Any]] = []
        active_states = {"queued", "initializing", "running", "finalizing", "cancelling"}
        with self._lock:
            for task in self._tasks.values():
                if str(task.payload.get("draft_id") or "") != str(draft_id):
                    continue
                if task.state in active_states:
                    active.append({"job_id": task.job_id, "status": task.state, "source": "memory"})
        if self.jobs is not None and hasattr(self.jobs.states, "list"):
            try:
                durable_rows = self.jobs.states.list("domain_adaptation", statuses={"pending", "running"})
            except Exception as exc:
                self.logger.warning("domain_draft_task_protection_lookup_failed error=%s", exc)
                durable_rows = []
            known_ids = {str(item.get("job_id") or "") for item in active}
            for row in durable_rows:
                payload = dict(row.get("payload") or {})
                request_payload = dict(payload.get("request_payload") or {})
                if str(request_payload.get("draft_id") or "") != str(draft_id):
                    continue
                job_id = str(row.get("task_id") or "")
                if job_id not in known_ids:
                    active.append({
                        "job_id": job_id,
                        "status": str(row.get("runtime_status") or row.get("status") or "running"),
                        "source": "durable",
                    })
        return active

    def _official_base_model(
        self,
        *,
        device: str | None = None,
        verify_sha256: bool = True,
        require_ready: bool = False,
    ) -> dict[str, Any]:
        """Resolve the build feature model independently from Runtime state."""
        if self.models is not None:
            return dict(
                self.models.official_base_model(
                    device=device,
                    verify_sha256=verify_sha256,
                    require_ready=require_ready,
                )
            )

        # Compatibility path for focused service tests and older embeddings.
        # Production construction always injects ModelRegistry above.
        official_id = str(self.settings.defaults.get("model_id") or "").strip()
        try:
            model_id, definition = self.settings.model_definition(official_id)
            profile_path = self.settings.profile_path(model_id)
            profile = self.settings.algorithm_profile_unbound(model_id)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise ApiError("OFFICIAL_BASE_MODEL_INVALID", "正式模型资源缺失或配置无效。", 503) from exc
        from core.feature_extractor import backbone_resource_status

        backbone = str(profile.get("backbone") or "").strip().lower()
        resource = dict(backbone_resource_status(backbone, verify_sha256=verify_sha256))
        resolved_device, warnings = self.registry.resolve_device(device)
        ready = bool(resource.get("ready"))
        if require_ready and not ready:
            raise ApiError("OFFICIAL_BASE_MODEL_UNAVAILABLE", "正式模型资源缺失或完整性校验失败。", 503)
        return {
            "model_id": model_id,
            "model_version": str(definition.get("version") or self.settings.backend.get("version") or "1.0.0"),
            "display_name": str(definition.get("display_name") or model_id),
            "profile_path": str(profile_path),
            "profile_sha256": _sha256(profile_path),
            "profile": profile,
            "backbone": backbone,
            "weights": {
                "source": resource.get("source") or "本机受控资源",
                "present": bool(resource.get("weight_present")),
                "sha256": str(resource.get("expected_sha256") or ""),
                "size_bytes": int(resource.get("expected_size") or 0),
            },
            "manifest": {
                "present": bool(resource.get("manifest_present")),
                "sha256": "",
                "integrity_status": str(resource.get("reason_code") or "unknown"),
            },
            "resource_status": resource,
            "ready": ready,
            "device_compatibility": {
                "requested": str(device or "auto"),
                "resolved": resolved_device,
                "warnings": list(warnings or []),
            },
            "source": "system_official_model_registry",
        }

    def _load_stage_history(self) -> dict[str, list[float]]:
        try:
            payload = json.loads(self.stage_history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return {}
        return {
            str(key): [float(value) for value in list(values or [])[-20:] if float(value) >= 0.0]
            for key, values in dict(payload or {}).items()
            if isinstance(values, list)
        }

    def _save_stage_history(self) -> None:
        self.stage_history_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.stage_history_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._stage_history, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.stage_history_path)

    @staticmethod
    def _stage_applicability(payload: dict[str, Any], stage_id: str) -> tuple[bool, str]:
        mode = str(payload.get("method_mode") or DEFAULT_CREATE_MODE).lower()
        if stage_id == "video_preparation":
            source_types = {str(payload.get("normal_source_type") or "image").lower()}
            if mode == LIGHTWEIGHT_CREATE_MODE:
                source_types.add(str(payload.get("defect_source_type") or "image").lower())
            applicable = bool(source_types & {"video", "mixed"})
            return applicable, "当前任务无需视频抽帧" if not applicable else "等待处理视频来源"
        if stage_id == "defect_extension":
            applicable = mode == LIGHTWEIGHT_CREATE_MODE
            return applicable, "当前构建模式无需轻量校准" if not applicable else "等待执行缺陷引导轻量校准"
        return True, "等待执行"

    def _initialize_stages(self, task: DomainBuildTask) -> None:
        if task.stages:
            return
        task.stages = []
        for definition in DOMAIN_BUILD_STAGES:
            applicable, message = self._stage_applicability(task.payload, str(definition["stage_id"]))
            task.stages.append(
                {
                    **definition,
                    "status": "pending" if applicable else "skipped",
                    "progress": 0.0,
                    "started_at": None,
                    "finished_at": _utc_now() if not applicable else None,
                    "elapsed_seconds": 0.0,
                    "message": message,
                }
            )

    @staticmethod
    def _stage(task: DomainBuildTask, stage_id: str) -> dict[str, Any]:
        for stage in task.stages:
            if str(stage.get("stage_id")) == stage_id:
                return stage
        raise KeyError(stage_id)

    def _set_stage(
        self,
        task: DomainBuildTask,
        stage_id: str,
        *,
        status: str = "running",
        progress: float | None = None,
        message: str | None = None,
    ) -> None:
        with self._lock:
            stage = self._stage(task, stage_id)
            now = _utc_now()
            if stage.get("status") == "skipped":
                return
            if stage.get("status") in {"completed", "warning"}:
                return
            if status in {"running", "completed", "warning", "failed"} and not stage.get("started_at"):
                stage["started_at"] = now
            if progress is not None:
                stage["progress"] = max(float(stage.get("progress") or 0.0), min(1.0, float(progress)))
            if message is not None:
                stage["message"] = str(message)
            stage["status"] = status
            if status in {"completed", "warning", "failed"}:
                stage["finished_at"] = now
                if status in {"completed", "warning"}:
                    stage["progress"] = 1.0
                started = stage.get("started_at")
                if started:
                    try:
                        stage["elapsed_seconds"] = max(
                            0.0,
                            (datetime.fromisoformat(now) - datetime.fromisoformat(str(started))).total_seconds(),
                        )
                    except ValueError:
                        pass
                if status in {"completed", "warning"} and not stage.get("history_recorded"):
                    duration = float(stage.get("elapsed_seconds") or 0.0)
                    history = self._stage_history.setdefault(stage_id, [])
                    history.append(duration)
                    self._stage_history[stage_id] = history[-20:]
                    stage["history_recorded"] = True
                    try:
                        self._save_stage_history()
                    except OSError as exc:
                        self.logger.warning("domain_stage_history_write_failed stage=%s error=%s", stage_id, exc)
            task.stage = str(stage.get("stage_name") or stage_id)
            task.message = str(stage.get("message") or "")
            task.last_progress_at = now
            task.heartbeat_at = now
            if task.started_monotonic is not None:
                task.elapsed_seconds = time.monotonic() - task.started_monotonic
            self._persist_task(task)

    def _heartbeat_worker(self, task: DomainBuildTask) -> None:
        while task.state not in TERMINAL_BUILD_STATES:
            time.sleep(2.0)
            if task.state in TERMINAL_BUILD_STATES:
                return
            if task.future is not None and task.future.done():
                return
            if task.thread is not None and not task.thread.is_alive():
                return
            with self._lock:
                task.heartbeat_at = _utc_now()
                self._persist_task(task)

    def _persist_task(self, task: DomainBuildTask) -> None:
        if self.jobs is None:
            return
        public = self._task_payload(task)
        public["request_payload"] = dict(task.payload)
        self.jobs.states.upsert(
            task.job_id,
            "domain_adaptation",
            task.state,
            public,
            idempotency_key=task.idempotency_key,
            request_fingerprint=task.request_fingerprint,
        )

    def _task_payload(self, task: DomainBuildTask) -> dict[str, Any]:
        if task.started_monotonic is not None and task.state in {"queued", "running", "cancelling"}:
            task.elapsed_seconds = time.monotonic() - task.started_monotonic
        now = datetime.now(timezone.utc)
        public_stages: list[dict[str, Any]] = []
        weighted_total = 0.0
        weighted_done = 0.0
        running_stage: dict[str, Any] | None = None
        for source in task.stages:
            stage = {key: value for key, value in source.items() if key != "history_recorded"}
            if stage.get("status") == "running" and stage.get("started_at"):
                try:
                    stage["elapsed_seconds"] = max(
                        float(stage.get("elapsed_seconds") or 0.0),
                        (now - datetime.fromisoformat(str(stage["started_at"]))).total_seconds(),
                    )
                except ValueError:
                    pass
                running_stage = stage
            if stage.get("status") != "skipped":
                weight = float(stage.get("weight") or 0.0)
                weighted_total += weight
                if stage.get("status") in {"completed", "warning"}:
                    weighted_done += weight
                elif stage.get("status") == "running":
                    weighted_done += weight * min(1.0, max(0.0, float(stage.get("progress") or 0.0)))
            public_stages.append(stage)
        progress = weighted_done / weighted_total if weighted_total else 0.0
        completion_ready = bool(
            task.state == "completed"
            and task.registration_status == "registered"
            and task.result
            and task.result.get("memory_bank_id")
            and task.result.get("config_profile_id")
            and task.result.get("resource_integrity_status") == "passed"
        )
        progress = 1.0 if completion_ready else min(0.99, max(0.0, progress))

        eta_seconds: float | None = 0.0 if completion_ready else None
        if task.state in {"queued", "running", "cancelling"} and public_stages:
            estimates: list[float] = []
            known = True
            for stage in public_stages:
                if stage.get("status") in {"completed", "warning", "skipped"}:
                    continue
                history = list(self._stage_history.get(str(stage.get("stage_id"))) or [])
                if not history:
                    known = False
                    break
                median_seconds = float(statistics.median(history))
                if stage.get("status") == "running":
                    estimates.append(max(0.0, median_seconds - float(stage.get("elapsed_seconds") or 0.0)))
                else:
                    estimates.append(median_seconds)
            if known:
                eta_seconds = sum(estimates)

        heartbeat_age: float | None = None
        progress_age: float | None = None
        for value, assign in ((task.heartbeat_at, "heartbeat"), (task.last_progress_at, "progress")):
            if not value:
                continue
            try:
                age = max(0.0, (now - datetime.fromisoformat(str(value))).total_seconds())
            except ValueError:
                continue
            if assign == "heartbeat":
                heartbeat_age = age
            else:
                progress_age = age
        activity_status = "active"
        if task.state == "running" and heartbeat_age is not None and heartbeat_age > 15.0:
            activity_status = "possibly_interrupted"
        elif task.state == "running" and progress_age is not None and progress_age > 20.0:
            activity_status = "long_running"
        return {
            "job_id": task.job_id,
            "idempotency_key": task.idempotency_key,
            "request_fingerprint": task.request_fingerprint,
            "status": task.state,
            "lifecycle_phase": task.lifecycle_phase,
            "stage": task.stage,
            "message": task.message,
            "progress": progress,
            "current": task.current,
            "total": task.total,
            "normal_processed": task.normal_processed,
            "defect_processed": task.defect_processed,
            "elapsed_seconds": round(task.elapsed_seconds, 3),
            "eta_seconds": None if eta_seconds is None else round(float(eta_seconds), 3),
            "estimated_remaining_seconds": None if eta_seconds is None else round(float(eta_seconds), 3),
            "created_at": task.created_at,
            "api_received_at": task.api_received_at,
            "task_created_at": task.task_created_at or task.created_at,
            "queued_at": task.queued_at,
            "worker_acquired_at": task.worker_acquired_at,
            "first_progress_at": task.first_progress_at,
            "build_started_at": task.build_started_at,
            "started_at": task.started_at,
            "finished_at": task.finished_at,
            "device": task.payload.get("device", "auto"),
            "error": task.error,
            "result": task.result,
            "registration_status": task.registration_status,
            "resource_catalog_revision": task.resource_catalog_revision,
            "stages": public_stages,
            "current_stage_id": str((running_stage or {}).get("stage_id") or ""),
            "last_progress_at": task.last_progress_at,
            "heartbeat_at": task.heartbeat_at,
            "activity_status": activity_status,
            "failed_stage": task.failed_stage,
            "dataset_manifest_created_at": (task.dataset_manifest or {}).get("created_at"),
            "rerunnable": task.state == "interrupted",
        }

    def _forbidden_roots(self) -> list[Path]:
        workspace_root = self.settings.project_root.parent.resolve()
        roots = [
            self.settings.algorithm_outputs_root.resolve(),
            self.root,
            (self.settings.project_root / "assets" / "video_demo").resolve(),
            (workspace_root / "AdaptiveAOI_Backup").resolve(),
            (workspace_root / "Docyment_report").resolve(),
            (self.settings.project_root / "data" / "evaluation_demo" / "test").resolve(),
            (self.settings.project_root / "data" / "demo_videos" / "inference").resolve(),
        ]
        return roots

    @staticmethod
    def _looks_like_test_source(path: Path) -> bool:
        blocked = {"test", "testing", "evaluation", "inference", "official_test", "non_public_test", "非公开测试", "测试集"}
        return any(part.casefold() in blocked for part in path.parts)

    def _source_directory(self, value: str | Path, role: str) -> Path:
        try:
            raw = str(value).strip()
            if not raw:
                raise ValueError("empty path")
            folder = Path(raw).expanduser().resolve()
        except (OSError, RuntimeError) as exc:
            raise ApiError("INVALID_SAMPLE_DIRECTORY", f"{role}样本文件夹路径无效。", 422) from exc
        if not folder.exists() or not folder.is_dir():
            raise ApiError("SAMPLE_DIRECTORY_MISSING", f"{role}样本文件夹不存在或不是目录。", 404)
        if self._looks_like_test_source(folder) or _is_within(folder, (self.settings.project_root / "data" / "evaluation_demo" / "test").resolve()):
            raise ApiError("TEST_DATA_FORBIDDEN", f"{role}样本路径属于测试数据，禁止用于目标域构建。", 403)
        if any(_is_within(folder, root) for root in self._forbidden_roots()):
            raise ApiError(
                "SAMPLE_DIRECTORY_FORBIDDEN",
                f"{role}样本不能来自检测输出、记忆库、备份或临时报告目录。",
                403,
            )
        return folder

    def _scan_one(self, value: str | Path, role: str) -> dict[str, Any]:
        scan_started = time.perf_counter()
        folder = self._source_directory(value, role)
        suffixes = self.settings.supported_suffixes()
        inventory: list[tuple[Path, str, int, int]] = []
        signature = hashlib.sha256()
        signature.update(SAMPLE_SCAN_RULE_VERSION.encode("utf-8"))
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            stat = path.stat()
            relative_path = path.relative_to(folder).as_posix()
            inventory.append((path, relative_path, int(stat.st_size), int(stat.st_mtime_ns)))
            signature.update(relative_path.encode("utf-8", errors="surrogatepass"))
            signature.update(str(stat.st_size).encode("ascii"))
            signature.update(str(stat.st_mtime_ns).encode("ascii"))
        inventory_signature = signature.hexdigest()
        cache_key = (str(folder), role, inventory_signature)
        with self._sample_scan_cache_lock:
            cached = self._sample_scan_cache.get(cache_key)
            if cached is not None:
                self._sample_scan_cache.move_to_end(cache_key)
                result = deepcopy(cached)
                result["cache_hit"] = True
                result["scan_elapsed_ms"] = round((time.perf_counter() - scan_started) * 1000.0, 1)
                return result

        valid: list[dict[str, Any]] = []
        invalid: list[dict[str, str]] = []
        for path, relative_path, size_bytes, _mtime_ns in inventory:
            try:
                with Image.open(path) as image:
                    image.verify()
                valid.append(
                    {
                        "path": path,
                        "relative_path": relative_path,
                        "size_bytes": size_bytes,
                        "sha256": _sha256(path),
                    }
                )
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                invalid.append({"relative_path": relative_path, "reason": f"{type(exc).__name__}: {exc}"})
        expected = 100 if role == "正常" else 30
        warnings = []
        if len(valid) != expected:
            warnings.append(f"期望 {expected} 张{role}图片，当前有效图片为 {len(valid)} 张。")
        if not valid:
            warnings.append(f"{role}样本文件夹中没有可读取的支持格式图片。")
        result = {
            "role": role,
            "root": folder,
            "discovered_count": len(inventory),
            "valid_count": len(valid),
            "invalid_count": len(invalid),
            "valid": valid,
            "invalid": invalid,
            "warnings": warnings,
            "cache_hit": False,
            "inventory_signature": inventory_signature,
            "scan_rule_version": SAMPLE_SCAN_RULE_VERSION,
            "scan_elapsed_ms": round((time.perf_counter() - scan_started) * 1000.0, 1),
        }
        with self._sample_scan_cache_lock:
            # A changed directory signature supersedes its previous entry; keep
            # the cache bounded so a long operator session cannot retain every
            # directory ever visited.
            for old_key in [key for key in self._sample_scan_cache if key[:2] == cache_key[:2] and key != cache_key]:
                self._sample_scan_cache.pop(old_key, None)
            self._sample_scan_cache[cache_key] = deepcopy(result)
            self._sample_scan_cache.move_to_end(cache_key)
            while len(self._sample_scan_cache) > self._sample_scan_cache_limit:
                self._sample_scan_cache.popitem(last=False)
        return result

    @staticmethod
    def _public_scan(scan: dict[str, Any]) -> dict[str, Any]:
        return {
            "role": scan["role"],
            "source_type": scan.get("source_type", "image"),
            "folder_name": scan["root"].name if scan.get("root") is not None else "",
            "discovered_count": scan["discovered_count"],
            "valid_count": scan["valid_count"],
            "invalid_count": scan["invalid_count"],
            "invalid_files": scan["invalid"][:20],
            "warnings": scan["warnings"],
            "video_summary": scan.get("video_summary") or {},
            "cache_hit": bool(scan.get("cache_hit")),
            "scan_elapsed_ms": float(scan.get("scan_elapsed_ms") or 0.0),
            "inventory_signature": scan.get("inventory_signature"),
            "scan_rule_version": scan.get("scan_rule_version"),
        }

    @staticmethod
    def _empty_scan(role: str, source_type: str) -> dict[str, Any]:
        return {
            "role": role,
            "source_type": source_type,
            "root": None,
            "discovered_count": 0,
            "valid_count": 0,
            "invalid_count": 0,
            "valid": [],
            "invalid": [],
            "warnings": [],
            "video_summary": {"video_count": 0, "valid_video_count": 0, "invalid_video_count": 0, "total_duration_seconds": 0.0, "warnings": []},
        }

    def _validate_video_inputs(self, files: list[str], directory: str, role: str) -> tuple[list[str], list[str]]:
        valid_files: list[str] = []
        for raw in files:
            try:
                path = Path(str(raw).strip()).expanduser().resolve()
            except (OSError, RuntimeError, ValueError) as exc:
                raise ApiError("INVALID_VIDEO_SOURCE", f"{role}视频文件路径无效。", 422) from exc
            if self._looks_like_test_source(path) or _is_within(path, (self.settings.project_root / "data" / "demo_videos" / "inference").resolve()):
                raise ApiError("TEST_DATA_FORBIDDEN", f"{role}视频属于测试数据，禁止用于目标域构建。", 403)
            if any(_is_within(path, root) for root in self._forbidden_roots()):
                raise ApiError("SAMPLE_DIRECTORY_FORBIDDEN", f"{role}视频不能来自输出、备份或临时报告目录。", 403)
            valid_files.append(str(path))
        valid_dirs: list[str] = []
        if directory:
            valid_dirs.append(str(self._source_directory(directory, role)))
        return valid_files, valid_dirs

    def _source_scan(self, payload: dict[str, Any], role: str) -> dict[str, Any]:
        prefix = "normal" if role == "正常" else "defect"
        source_type = str(payload.get(f"{prefix}_source_type") or "image").lower()
        if source_type not in {"image", "video", "mixed"}:
            raise ApiError("INVALID_SAMPLE_SOURCE_TYPE", f"{role}样本来源类型无效。", 422)
        image_dir = str(payload.get(f"{prefix}_image_dir") or payload.get("normal_dir" if prefix == "normal" else "defect_dir") or "").strip()
        files = [str(item).strip() for item in list(payload.get(f"{prefix}_video_files") or []) if str(item).strip()]
        video_dir = str(payload.get(f"{prefix}_video_dir") or "").strip()
        scan = self._empty_scan(role, source_type)
        if source_type in {"image", "mixed"}:
            if not image_dir:
                scan["warnings"].append(f"{role}图片文件夹尚未选择。")
            else:
                image_scan = self._scan_one(image_dir, role)
                scan.update(
                    {
                        key: image_scan[key]
                        for key in (
                            "root",
                            "discovered_count",
                            "valid_count",
                            "invalid_count",
                            "valid",
                            "invalid",
                            "warnings",
                            "cache_hit",
                            "scan_elapsed_ms",
                            "inventory_signature",
                            "scan_rule_version",
                        )
                    }
                )
        if source_type in {"video", "mixed"}:
            files, dirs = self._validate_video_inputs(files, video_dir, role)
            video_summary = self.video_samples.scan(files, dirs)
            scan["video_summary"] = video_summary
            scan["warnings"].extend(video_summary.get("warnings") or [])
            scan["discovered_count"] += int(video_summary.get("video_count") or 0)
            scan["valid_count"] += int(video_summary.get("valid_video_count") or 0)
            scan["invalid_count"] += int(video_summary.get("invalid_video_count") or 0)
        feedback_label = "normal" if prefix == "normal" else "defect"
        feedback_rows = [
            item for item in list(payload.get("_feedback_samples") or [])
            if str(item.get("corrected_label") or "") == feedback_label and item.get("_source_path")
        ]
        for item in feedback_rows:
            source_path = Path(str(item["_source_path"])).resolve()
            scan["valid"].append(
                {
                    "path": source_path,
                    "relative_path": f"feedback/{item.get('candidate_id')}/{source_path.name}",
                    "size_bytes": source_path.stat().st_size,
                    "sha256": _sha256(source_path),
                    "source_type": "feedback",
                    "feedback_candidate_id": item.get("candidate_id"),
                    "mask_path": item.get("_mask_path"),
                }
            )
        if feedback_rows:
            scan["discovered_count"] += len(feedback_rows)
            scan["valid_count"] += len(feedback_rows)
            scan["warnings"] = [warning for warning in scan["warnings"] if "尚未选择" not in warning]
        scan["source_type"] = source_type
        return scan

    def _creation_mode(self, payload: dict[str, Any]) -> str:
        """Resolve and validate the mode used by a *new* target-domain build.

        Historical profiles keep their original mode and remain readable.  A
        new draft/job, including one submitted through a hidden API field or a
        legacy config template, may only use the current creation modes.
        """

        mode = str(payload.get("method_mode") or DEFAULT_CREATE_MODE).strip().lower()
        config_profile_id = str(payload.get("config_profile_id") or "")
        config_profile_version = int(payload.get("config_profile_version") or 0)
        if config_profile_id and config_profile_version:
            selected = self.config_profiles.get(config_profile_id, config_profile_version)
            mode = str((selected.get("config") or {}).get("method_mode") or mode).strip().lower()
        if mode not in CREATABLE_DOMAIN_METHOD_MODES:
            raise ApiError(
                "DOMAIN_BUILD_MODE_RETIRED",
                "该请求使用已停止新建的历史构建方式。请在目标域页面明确选择当前支持的构建模式后重试。",
                422,
                details={"allowed_modes": list(CREATABLE_DOMAIN_METHOD_LABELS.values())},
            )
        return mode

    @staticmethod
    def _requested_precision(payload: dict[str, Any]) -> str:
        try:
            return normalize_precision_request(payload, default="fp16")
        except ValueError as exc:
            raise ApiError("INVALID_DOMAIN_BUILD_PRECISION", str(exc), 422) from exc

    def _draft_path(self, draft_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9._-]{8,128}", str(draft_id)):
            raise ApiError("INVALID_DOMAIN_DRAFT_ID", "目标域草稿标识无效。", 422)
        return self.draft_root / f"{draft_id}.json"

    @staticmethod
    def _public_draft(payload: dict[str, Any]) -> dict[str, Any]:
        draft_payload = dict(payload.get("payload") or {})
        for field in SYSTEM_MANAGED_MODEL_FIELDS:
            draft_payload.pop(field, None)
        return {
            "draft_id": payload["draft_id"],
            "name": str(draft_payload.get("friendly_name") or "未命名目标域"),
            "current_step": int(payload.get("current_step") or 1),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
            "payload": draft_payload,
        }

    def save_draft(self, data: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            draft_id = str(data.get("draft_id") or f"domain-draft-{uuid.uuid4().hex[:16]}")
            path = self._draft_path(draft_id)
            previous: dict[str, Any] = {}
            if path.is_file():
                try:
                    previous = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    raise ApiError("DOMAIN_DRAFT_INVALID", "目标域草稿内容损坏，无法继续保存。", 409) from exc
            draft_payload = dict(data.get("payload") or {})
            for field in SYSTEM_MANAGED_MODEL_FIELDS:
                draft_payload.pop(field, None)
            draft_payload["method_mode"] = self._creation_mode(draft_payload)
            draft_payload.setdefault("layers", default_feature_layers())
            requested_precision = self._requested_precision(draft_payload)
            draft_payload["precision"] = requested_precision
            draft_payload["use_fp16"] = precision_uses_fp16(requested_precision)
            document = {
                "draft_id": draft_id,
                "current_step": max(1, min(5, int(data.get("current_step") or 1))),
                "payload": draft_payload,
                "created_at": previous.get("created_at") or _utc_now(),
                "updated_at": _utc_now(),
                "status": "draft",
            }
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, path)
            self._bump_draft_revision()
            return self._public_draft(document)

    def list_drafts(self) -> list[dict[str, Any]]:
        result = []
        for path in sorted(self.draft_root.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                result.append(self._public_draft(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, ValueError, KeyError):
                continue
        return result

    def get_draft(self, draft_id: str) -> dict[str, Any]:
        path = self._draft_path(draft_id)
        if not path.is_file():
            raise ApiError("DOMAIN_DRAFT_NOT_FOUND", "目标域草稿不存在。", 404)
        return self._public_draft(json.loads(path.read_text(encoding="utf-8")))

    def delete_draft_preview(self, draft_id: str) -> dict[str, Any]:
        with self._lock:
            path = self._draft_path(draft_id)
            if not path.is_file():
                raise ApiError("DOMAIN_DRAFT_NOT_FOUND", "目标域草稿不存在。", 404)
            artifacts = self._draft_artifacts(draft_id)
            protected_tasks = self._active_draft_protection(draft_id)
            plan_id = f"domain_draft_delete_{uuid.uuid4().hex}"
            preview_token = secrets.token_urlsafe(32)
            expires_at = time.time() + 600
            self._draft_delete_plans[plan_id] = {
                "draft_id": str(draft_id),
                "preview_token": preview_token,
                "fingerprint": self._draft_fingerprint(draft_id),
                "expires_at": expires_at,
            }
            total_bytes = 0
            for artifact in artifacts:
                try:
                    total_bytes += int(artifact.stat().st_size)
                except OSError:
                    continue
            protected_items = [
                {
                    "job_id": str(item.get("job_id") or ""),
                    "action": "protected",
                    "protected_reason": "该草稿当前关联正在执行的目标域构建任务，请先取消任务，待任务结束后再删除草稿。",
                }
                for item in protected_tasks
            ]
            allowed = not protected_tasks
            return {
                "plan_id": plan_id,
                "preview_token": preview_token,
                "draft_id": str(draft_id),
                "selected_count": 1,
                "deletable_count": 1 if allowed else 0,
                "protected_count": len(protected_tasks),
                "protected_items": protected_items,
                "file_count": len(artifacts),
                "total_bytes": total_bytes,
                "allowed": allowed,
                "expires_in_seconds": 600,
                "draft_revision": self._draft_revision,
            }

    def delete_draft(self, draft_id: str, *, plan_id: str = "", preview_token: str = "") -> dict[str, Any]:
        with self._lock:
            path = self._draft_path(draft_id)
            if not path.is_file():
                # DELETE is intentionally idempotent: a retry after a
                # successful request must not turn a cleaned draft into a
                # misleading 500/traceback.
                return {"draft_id": str(draft_id), "deleted": False, "already_deleted": True, "message": "草稿已经删除。"}
            plan_key = str(plan_id or "")
            plan = self._draft_delete_plans.get(plan_key)
            if not plan or str(plan.get("draft_id")) != str(draft_id) or str(plan.get("preview_token")) != str(preview_token) or float(plan.get("expires_at") or 0) <= time.time():
                self._draft_delete_plans.pop(plan_key, None)
                raise ApiError("DOMAIN_DRAFT_DELETE_PREVIEW_EXPIRED", "删除预览已过期，请重新生成。", 409)
            protected_tasks = self._active_draft_protection(draft_id)
            if protected_tasks:
                raise ApiError(
                    "DOMAIN_DRAFT_IN_USE",
                    "该草稿当前存在正在执行的目标域构建任务，请先取消任务后再删除草稿。",
                    409,
                    details={"job_ids": [str(item.get("job_id") or "") for item in protected_tasks]},
                )
            fingerprint = self._draft_fingerprint(draft_id)
            if fingerprint != str(plan.get("fingerprint") or ""):
                raise ApiError("DOMAIN_DRAFT_DELETE_PREVIEW_STALE", "草稿已变化，请重新生成删除预览。", 409)
            artifacts = self._draft_artifacts(draft_id)
            deleted_files = 0
            for artifact in artifacts:
                try:
                    artifact.unlink()
                    deleted_files += 1
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise ApiError("DOMAIN_DRAFT_DELETE_FAILED", "草稿临时文件无法清理，请稍后重试。", 500) from exc
            owned_directory = (self.draft_root / str(draft_id)).resolve()
            try:
                owned_directory.relative_to(self.draft_root.resolve())
            except ValueError:
                owned_directory = None
            if owned_directory is not None and owned_directory.is_dir():
                shutil.rmtree(owned_directory, ignore_errors=False)
            self._draft_delete_plans.pop(plan_key, None)
            revision = self._bump_draft_revision()
            return {
                "draft_id": str(draft_id),
                "deleted": True,
                "deleted_files": deleted_files,
                "plan_id": plan_key,
                "draft_revision": revision,
            }

    def scan_domain_samples(self, payload: dict[str, Any] | str, defect_dir: str | None = None) -> dict[str, Any]:
        if isinstance(payload, str):
            payload = {"normal_dir": payload, "defect_dir": str(defect_dir or ""), "normal_source_type": "image", "defect_source_type": "image"}
        payload = dict(payload or {})
        payload["method_mode"] = self._creation_mode(payload)
        normal = self._source_scan(payload, "正常")
        defect = self._source_scan(payload, "缺陷校准")
        normal_root = normal.get("root")
        defect_root = defect.get("root")
        if normal_root and defect_root and (normal_root == defect_root or _is_within(normal_root, defect_root) or _is_within(defect_root, normal_root)):
            raise ApiError("SAMPLE_DIRECTORIES_OVERLAP", "正常样本和缺陷校准样本必须使用彼此独立的文件夹。", 422)
        normal_video_values = [str(item) for item in list(payload.get("normal_video_files") or []) if str(item).strip()]
        normal_video_values += [str(payload.get("normal_video_dir") or "")]
        defect_video_values = [str(item) for item in list(payload.get("defect_video_files") or []) if str(item).strip()]
        defect_video_values += [str(payload.get("defect_video_dir") or "")]
        normal_video_paths = []
        defect_video_paths = []
        for value in normal_video_values:
            if not value:
                continue
            try:
                normal_video_paths.append(Path(value).expanduser().resolve())
            except (OSError, RuntimeError, ValueError):
                continue
        for value in defect_video_values:
            if not value:
                continue
            try:
                defect_video_paths.append(Path(value).expanduser().resolve())
            except (OSError, RuntimeError, ValueError):
                continue
        if any(a == b or _is_within(a, b) or _is_within(b, a) for a in normal_video_paths for b in defect_video_paths):
            raise ApiError("SAMPLE_SOURCES_OVERLAP", "正常和缺陷视频来源必须相互独立，不能选择同一视频或嵌套目录。", 422)
        normal_hashes = {item["sha256"] for item in normal["valid"]}
        duplicate_hashes = normal_hashes & {item["sha256"] for item in defect["valid"]}
        if duplicate_hashes:
            raise ApiError("SAMPLE_CONTENT_OVERLAP", "正常样本与缺陷校准样本包含内容相同的图片，已阻止构建。", 422)
        normal_has_source = bool(normal["valid"] or (normal.get("video_summary") or {}).get("valid_video_count"))
        defect_has_source = bool(defect["valid"] or (defect.get("video_summary") or {}).get("valid_video_count"))
        mode = str(payload.get("method_mode") or DEFAULT_CREATE_MODE).lower()
        defect_required = mode == LIGHTWEIGHT_CREATE_MODE
        normal_target_count = int(payload.get("normal_target_count") or 100)
        defect_target_count = int(payload.get("defect_target_count") or 30)
        budget_mode = str(payload.get("sample_budget_mode") or "final_frame_count")
        normal_source_type = str(payload.get("normal_source_type") or "image")
        defect_source_type = str(payload.get("defect_source_type") or "image")

        # A pure-image scan knows the final usable sample count immediately.
        # Video/mixed inputs can only prove the final frame budget after the
        # deterministic extraction/de-duplication stage, so keep that state
        # explicit instead of reporting a false evaluation-ready result.
        normal_budget_pending = normal_source_type in {"video", "mixed"} and budget_mode != "video_file_count"
        defect_budget_pending = defect_required and defect_source_type in {"video", "mixed"} and budget_mode != "video_file_count"
        if budget_mode == "video_file_count":
            normal_available = int((normal.get("video_summary") or {}).get("valid_video_count") or 0)
            defect_available = int((defect.get("video_summary") or {}).get("valid_video_count") or 0)
        else:
            normal_available = int(normal.get("valid_count") or 0)
            defect_available = int(defect.get("valid_count") or 0)
        normal_budget_ready = normal_available >= normal_target_count if not normal_budget_pending else False
        defect_budget_ready = (
            True
            if not defect_required
            else (defect_available >= defect_target_count if not defect_budget_pending else False)
        )
        source_ready = bool(normal_has_source and (defect_has_source or not defect_required))
        budget_ready = bool(normal_budget_ready and defect_budget_ready)
        budget_validation_pending = bool(normal_budget_pending or defect_budget_pending)
        normal_public = self._public_scan(normal)
        defect_public = self._public_scan(defect)
        normal_public.update(
            budget_target=normal_target_count,
            budget_available=normal_available,
            budget_ready=normal_budget_ready,
            budget_validation_pending=normal_budget_pending,
        )
        defect_public.update(
            budget_target=defect_target_count,
            budget_available=defect_available,
            budget_ready=defect_budget_ready,
            budget_validation_pending=defect_budget_pending,
        )
        return {
            "normal": normal_public,
            "defect": defect_public,
            "category": "由当前运行配置指定",
            "sample_budget_mode": budget_mode,
            "normal_target_count": normal_target_count,
            "defect_target_count": defect_target_count,
            "source_ready": source_ready,
            "sample_budget_ready": budget_ready,
            "budget_validation_pending": budget_validation_pending,
            "ready": bool(source_ready and (budget_ready or budget_validation_pending)),
        }

    def preview_videos(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Extract a small, disposable representative-frame preview.

        The endpoint returns only safe frame metadata; materialized JPEGs stay
        in a task-owned temporary directory and are removed before returning.
        Full training extraction is performed only by the build worker.
        """
        payload = dict(payload or {})
        sampling = dict(payload.get("video_sampling_config") or {})
        sampling["final_max_samples"] = min(12, int(sampling.get("final_max_samples", 12) or 12))
        sampling["max_frames_per_video"] = min(12, int(sampling.get("max_frames_per_video", 12) or 12))
        previews: dict[str, Any] = {}
        for prefix, role_label in (("normal", "normal"), ("defect", "defect")):
            source_type = str(payload.get(f"{prefix}_source_type") or "image").lower()
            if source_type not in {"video", "mixed"}:
                continue
            files, dirs = self._validate_video_inputs(
                [str(item) for item in list(payload.get(f"{prefix}_video_files") or []) if str(item).strip()],
                str(payload.get(f"{prefix}_video_dir") or ""),
                "正常" if prefix == "normal" else "缺陷校准",
            )
            if not files and not dirs:
                previews[prefix] = {"frames": [], "summary": {"video_count": 0}, "message": "尚未选择视频样本。"}
                continue
            try:
                result = self.video_samples.extract(
                    role_label,
                    files,
                    dirs,
                    sampling,
                    target_count=12,
                    intervals=payload.get("defect_video_intervals") if prefix == "defect" else None,
                    excluded_sample_ids=payload.get("excluded_sample_ids") or [],
                )
            except ValueError as exc:
                # Keep malformed ranges and decoder failures actionable in the
                # UI instead of leaking a Python traceback through the API.
                raise ApiError("VIDEO_SAMPLE_INVALID", str(exc), 422) from exc
            try:
                frames: list[dict[str, Any]] = []
                for row in result.get("manifest") or []:
                    safe_row = {
                        key: row.get(key)
                        for key in ("sample_id", "source_video_id", "source_video_name", "frame_index", "timestamp_ms", "original_fps", "width", "height", "extraction_method", "quality_score", "selected", "usage", "defect_interval")
                    }
                    try:
                        with Image.open(Path(str(row.get("path")))) as image:
                            image.thumbnail((240, 240))
                            buffer = io.BytesIO()
                            image.convert("RGB").save(buffer, format="JPEG", quality=82)
                            safe_row["thumbnail_base64"] = base64.b64encode(buffer.getvalue()).decode("ascii")
                    except (OSError, UnidentifiedImageError):
                        safe_row["thumbnail_base64"] = ""
                    frames.append(safe_row)
                previews[prefix] = {
                    "summary": result.get("summary") or {},
                    "frames": frames,
                }
            finally:
                self.video_samples.cleanup(result)
        return {"previews": previews}

    def defaults(self, model_id: str | None, dataset: str, category: str, device: str | None) -> dict[str, Any]:
        # ``model_id`` remains accepted for old clients, but target-domain
        # creation always uses the packaged official feature model.
        official_model = self._official_base_model(device=device, verify_sha256=True)
        resolved_model_id = str(official_model["model_id"])
        profile = dict(official_model["profile"])
        compatibility = dict(official_model.get("device_compatibility") or {})
        resolved_device = str(compatibility.get("resolved") or "cpu")
        warnings = list(compatibility.get("warnings") or [])
        return {
            "model_id": resolved_model_id,
            "model_version": str(official_model.get("model_version") or ""),
            "dataset": str(dataset).lower(),
            "category": str(category),
            "device": resolved_device,
            "image_size": int(profile.get("image_size", 256)),
            "feature_layers": list(profile.get("layers", default_feature_layers())),
            "embedding_dimension": "由所选骨干网络和特征层自动确定",
            "coreset_ratio": float(profile.get("coreset_ratio", 0.1)),
            "max_memory_patches": int(profile.get("max_memory_patches", 50000)),
            "knn_k": int(profile.get("knn_k", 1)),
            "distance_metric": "euclidean",
            "batch_size": int(profile.get("batch_size", 1)),
            "threshold_method": str(profile.get("threshold_method", "normal_percentile")),
            "threshold_percentile": float(profile.get("threshold_percentile", 99.0)),
            "method_mode": DEFAULT_CREATE_MODE,
            "method_modes": list(CREATABLE_DOMAIN_METHOD_LABELS),
            "method_mode_labels": dict(CREATABLE_DOMAIN_METHOD_LABELS),
            "precision": "fp16",
            "use_fp16": True,
            "precision_options": ["fp16", "fp32"],
            "requested_precision": "fp16",
            "actual_precision": "fp16" if resolved_device == "cuda" else "fp32",
            "precision_fallback_reason": "" if resolved_device == "cuda" else "cpu",
            "lightweight_calibration": {
                "enabled": True,
                "method_mode": "normal_memory_defect_guided_lite",
                "display_name": "单正常记忆·缺陷引导轻量校准",
                "pooling_candidates": [0.005, 0.01, 0.02, 0.05],
                "gate_percentile": 95.0,
                "l2": 0.01,
                "cv_folds": 5,
            },
            "max_defect_prototypes": int(profile.get("max_defect_prototypes", 32)),
            "defect_top_ratio": float(profile.get("defect_top_ratio", 0.05)),
            "normal_floor_percentile": float(profile.get("normal_floor_percentile", 99.0)),
            "min_defect_image_support": int(profile.get("min_defect_image_support", 2)),
            "random_seed": int(profile.get("random_seed", 42)),
            "threshold_methods": sorted(_domain_adaptation_module().SUPPORTED_DOMAIN_THRESHOLD_METHODS),
            "warnings": warnings,
        }

    def _validated_build_config(self, payload: dict[str, Any]) -> tuple[dict[str, Any], str, str, str, int]:
        config_profile_id = str(payload.get("config_profile_id") or "")
        config_profile_version = int(payload.get("config_profile_version") or 0)
        bound_profile = False
        bound_dataset = ""
        bound_category = ""
        official_model = self._official_base_model(
            device=str(payload.get("device") or "auto"),
            verify_sha256=True,
            require_ready=True,
        )
        model_id = str(official_model["model_id"])
        profile = dict(official_model["profile"])
        official_backbone = str(official_model.get("backbone") or profile.get("backbone") or "")
        if config_profile_id and config_profile_version:
            selected = self.config_profiles.get(config_profile_id, config_profile_version)
            if selected.get("frozen"):
                raise ApiError("FROZEN_CONFIG_PROFILE", "冻结配置不能再次构建，请先复制为新版本。", 409)
            if not selected.get("validated"):
                raise ApiError("CONFIG_PROFILE_NOT_VALIDATED", "配置方案必须先验证，再生成记忆库。", 409)
            profile.update(selected.get("config") or {})
            bound_dataset = str(profile.get("dataset") or "").lower()
            bound_category = str(profile.get("category") or "")
            bound_profile = True
        # A template may carry historical model fields, but it cannot switch
        # the packaged feature space used by a new target-domain build.
        profile["model_id"] = model_id
        profile["backbone"] = official_backbone
        dataset = str(payload.get("dataset") or profile.get("dataset") or "custom").lower().strip()
        category = str(payload.get("category") or profile.get("category") or "custom").strip()
        if not dataset or not category or any(token in category for token in ("/", "\\", "..")):
            raise ApiError("INVALID_DOMAIN_SELECTION", "数据集或类别值无效。", 422)
        if bound_profile and (bound_dataset != dataset or bound_category != category):
            raise ApiError("CONFIG_PROFILE_DOMAIN_MISMATCH", "配置方案与目标域的数据集或类别不一致。", 409)
        device, warnings = self.registry.resolve_device(profile.get("device") if bound_profile else payload.get("device"))
        method = str(profile.get("threshold_method") if bound_profile else payload.get("threshold_method") or profile.get("threshold_method") or "best_f1").lower()
        if method not in _domain_adaptation_module().SUPPORTED_DOMAIN_THRESHOLD_METHODS:
            raise ApiError("UNSUPPORTED_THRESHOLD_METHOD", f"当前不支持阈值方法：{method}", 422)
        def selected_value(key: str, default: Any) -> Any:
            return profile.get(key, default) if bound_profile else payload.get(key, profile.get(key, default))

        image_size = int(selected_value("image_size", 256))
        ratio = float(selected_value("coreset_ratio", 0.1))
        max_patches = int(selected_value("max_memory_patches", 50000))
        knn_k = int(selected_value("knn_k", 1))
        batch_size = int(selected_value("batch_size", 1))
        percentile = float(selected_value("threshold_percentile", 99.0))
        seed = int(selected_value("random_seed", 42))
        method_mode = self._creation_mode({
            **payload,
            "method_mode": selected_value("method_mode", DEFAULT_CREATE_MODE),
        })
        max_defect_prototypes = int(selected_value("max_defect_prototypes", 32))
        defect_top_ratio = float(selected_value("defect_top_ratio", 0.05))
        normal_floor_percentile = float(selected_value("normal_floor_percentile", 99.0))
        min_defect_image_support = int(selected_value("min_defect_image_support", 2))
        lightweight_gate_percentile = float(selected_value("lightweight_gate_percentile", 95.0))
        lightweight_l2 = float(selected_value("lightweight_l2", 0.01))
        lightweight_score_scale = float(selected_value("lightweight_score_scale", 1.0))
        lightweight_cv_folds = int(selected_value("lightweight_cv_folds", 5))
        if not 64 <= image_size <= 2048:
            raise ApiError("INVALID_IMAGE_SIZE", "输入尺寸必须在 64 到 2048 之间。", 422)
        if not 0.001 <= ratio <= 1.0:
            raise ApiError("INVALID_CORESET_RATIO", "coreset 采样比例必须在 0.001 到 1.0 之间。", 422)
        if not 1 <= max_patches <= 200000:
            raise ApiError("INVALID_CORESET_SIZE", "记忆库目标大小必须在 1 到 200000 之间。", 422)
        if not 1 <= knn_k <= 64 or not 1 <= batch_size <= 32 or not 50.0 <= percentile <= 100.0 or seed < 0:
            raise ApiError("INVALID_DOMAIN_PARAMETERS", "目标域适配参数超出允许范围。", 422)
        if method_mode not in CREATABLE_DOMAIN_METHOD_MODES:
            raise ApiError("DOMAIN_BUILD_MODE_RETIRED", "该构建模式已停止用于创建新目标域资源。", 422)
        if not 1 <= max_defect_prototypes <= 128 or not 0.005 <= defect_top_ratio <= 0.25:
            raise ApiError("INVALID_DEFECT_MEMORY_PARAMETERS", "缺陷原型数量或补丁挖掘比例超出允许范围。", 422)
        if not 90.0 <= normal_floor_percentile <= 99.9 or not 1 <= min_defect_image_support <= 30:
            raise ApiError("INVALID_DEFECT_MEMORY_PARAMETERS", "缺陷原型可靠性参数超出允许范围。", 422)
        if not 90.0 <= lightweight_gate_percentile <= 99.9 or not 0.0 <= lightweight_l2 <= 10.0 or not 0.0 < lightweight_score_scale <= 100.0:
            raise ApiError("INVALID_LIGHTWEIGHT_CALIBRATION_PARAMETERS", "轻量校准参数超出允许范围。", 422)
        if not 2 <= lightweight_cv_folds <= 10:
            raise ApiError("INVALID_LIGHTWEIGHT_CALIBRATION_PARAMETERS", "轻量校准交叉验证折数必须在 2 到 10 之间。", 422)
        selected_precision = selected_value("precision", None)
        precision_source = (
            {"precision": selected_precision}
            if selected_precision not in {None, ""}
            else {"use_fp16": selected_value("use_fp16", True)}
        )
        requested_precision = self._requested_precision(precision_source)
        actual_precision = "fp16" if requested_precision == "fp16" and device == "cuda" else "fp32"
        fallback_reason = "cpu" if requested_precision == "fp16" and actual_precision == "fp32" else ""
        if fallback_reason:
            warnings = [*warnings, "当前设备不支持半精度构建，将使用单精度模式。"]
        profile.update(
            {
                "dataset": dataset,
                "category": category,
                "model_id": model_id,
                "model_version": str(official_model.get("model_version") or ""),
                "base_model_profile_sha256": str(official_model.get("profile_sha256") or ""),
                "base_model_weights_sha256": str((official_model.get("weights") or {}).get("sha256") or ""),
                "base_model_manifest_sha256": str((official_model.get("manifest") or {}).get("sha256") or ""),
                "base_model_integrity_status": str((official_model.get("resource_status") or {}).get("reason_code") or "unknown"),
                "base_model_source": "system_official_model_registry",
                "device": device,
                "precision": requested_precision,
                "requested_precision": requested_precision,
                "actual_precision": actual_precision,
                "use_fp16": precision_uses_fp16(actual_precision),
                "amp_enabled": precision_uses_fp16(actual_precision),
                "precision_fallback_reason": fallback_reason,
                "backbone": official_backbone,
                "layers": list(selected_value("layers", default_feature_layers())),
                "image_size": image_size,
                "coreset_ratio": ratio,
                "max_memory_patches": max_patches,
                "knn_k": knn_k,
                "memory_backend": str(selected_value("memory_backend", "auto")),
                "memory_backend_fallback": str(selected_value("memory_backend_fallback", "faiss_cpu_flat")),
                "memory_ivf_nlist": int(selected_value("memory_ivf_nlist", 256)),
                "memory_ivf_nprobe": int(selected_value("memory_ivf_nprobe", 32)),
                "score_aggregation": str(selected_value("score_aggregation", "maximum_patch")),
                "batch_size": batch_size,
                "threshold_method": method,
                "threshold_percentile": percentile,
                "threshold_target_recall": float(selected_value("threshold_target_recall", 0.95)),
                "threshold_max_false_positive_rate": float(selected_value("threshold_max_false_positive_rate", 0.10)),
                "pixel_threshold_mode": str(selected_value("pixel_threshold_mode", "automatic_from_normal_map")),
                "random_seed": seed,
                "method_mode": method_mode,
                "algorithm_mode": method_mode,
                "normal_memory_enabled": True,
                "lightweight_calibration_enabled": method_mode == LIGHTWEIGHT_CREATE_MODE,
                "defect_guidance_enabled": method_mode == LIGHTWEIGHT_CREATE_MODE,
                "max_defect_prototypes": max_defect_prototypes,
                "defect_coreset_ratio": float(selected_value("defect_coreset_ratio", 0.1)),
                "defect_region_filter": str(selected_value("defect_region_filter", "high_anomaly_patches")),
                "dual_fusion_strategy": str(selected_value("dual_fusion_strategy", "reliability_gated_relative_distance")),
                "defect_top_ratio": defect_top_ratio,
                "normal_floor_percentile": normal_floor_percentile,
                "min_defect_image_support": min_defect_image_support,
                "lightweight_gate_percentile": lightweight_gate_percentile,
                "lightweight_l2": lightweight_l2,
                "lightweight_score_scale": lightweight_score_scale,
                "lightweight_cv_folds": lightweight_cv_folds,
                "lightweight_pooling_candidates": [0.005, 0.01, 0.02, 0.05],
                "dual_alpha_candidates": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
                "domain_profile_mode": "adaptation_then_frozen_inference",
                "inference_only": True,
                "warnings": warnings,
            }
        )
        # A build never freezes or reuses a mutable configuration record.
        # The exclusive snapshot is created only after the expensive algorithm
        # phase has succeeded, immediately before the profile is published.
        # This prevents failed builds from leaving ordinary users with orphaned
        # configuration entries.
        if bound_profile:
            payload.setdefault("template_config_profile_id", config_profile_id)
            payload.setdefault("template_config_profile_version", config_profile_version)
        config_profile_id = ""
        config_profile_version = 0
        return profile, model_id, device, config_profile_id, config_profile_version

    def warm_build_environment_async(self) -> bool:
        """Warm the lazy target-domain module without gating API readiness.

        Native ML imports can hold the interpreter long enough to starve the
        FastAPI event loop even when launched from a Python thread.  Use an
        isolated helper process to validate imports and warm the operating
        system's file/native-library caches.  The API process keeps the actual
        module lazy, so a task can publish ``preparing_environment`` before its
        own import and a stuck helper can never block liveness/readiness.
        """

        with self._build_environment_warmup_lock:
            current = self._build_environment_warmup_thread
            if self._build_environment_warmup_state == "warming":
                return False
            if current is not None and current.is_alive():
                return False
            if self._build_environment_warmup_state == "ready":
                return False
            self._build_environment_warmup_state = "warming"
            self._build_environment_warmup_error = ""
            self._build_environment_warmup_done.clear()
            self._build_environment_warmup_stop.clear()
            worker = threading.Thread(
                target=self._warm_build_environment,
                name="AdaptiveAOI-Domain-Environment-Warmup",
                daemon=True,
            )
            self._build_environment_warmup_thread = worker
            try:
                # Start while the state lock still owns the ``warming`` claim.
                # The helper thread may block briefly when registering its
                # process, but a concurrent caller can no longer observe a
                # non-alive thread and create a second helper.
                worker.start()
            except RuntimeError as exc:
                self._build_environment_warmup_thread = None
                self._build_environment_warmup_state = "failed"
                self._build_environment_warmup_error = f"{type(exc).__name__}: {exc}"
                self._build_environment_warmup_done.set()
                self.logger.warning("domain_build_environment_warmup_start_failed error=%s", exc)
                return False
        return True

    @staticmethod
    def _reclaim_warmup_process(process: subprocess.Popen[Any]) -> None:
        if process.poll() is not None:
            return
        try:
            process.terminate()
        except OSError:
            pass
        try:
            process.wait(timeout=BUILD_ENVIRONMENT_TERMINATE_TIMEOUT_SECONDS)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=BUILD_ENVIRONMENT_TERMINATE_TIMEOUT_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _warm_build_environment(self) -> None:
        started = time.perf_counter()
        try:
            creation_flags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "from methods.patchcore import domain_adaptation as _domain_adaptation",
                ],
                cwd=str(self.settings.project_root),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creation_flags,
            )
            with self._build_environment_warmup_lock:
                self._build_environment_warmup_process = process
                stopping = self._build_environment_warmup_stop.is_set()
            if stopping:
                self._reclaim_warmup_process(process)
                raise RuntimeError("isolated warm-up stopped during shutdown")
            try:
                return_code = process.wait(timeout=BUILD_ENVIRONMENT_WARMUP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired as exc:
                self._reclaim_warmup_process(process)
                raise RuntimeError(
                    f"isolated warm-up exceeded {BUILD_ENVIRONMENT_WARMUP_TIMEOUT_SECONDS:.0f}s"
                ) from exc
            if return_code != 0:
                raise RuntimeError(f"isolated warm-up exited with code {return_code}")
        except Exception as exc:
            with self._build_environment_warmup_lock:
                if self._build_environment_warmup_stop.is_set():
                    # The isolated helper is only an opportunistic cache warm-up.
                    # A real queued build is allowed to take ownership of loading
                    # the module instead of waiting for this helper to finish.
                    if self._build_environment_warmup_state != "ready":
                        self._build_environment_warmup_state = "preempted"
                    self._build_environment_warmup_error = ""
                    stopped = True
                else:
                    self._build_environment_warmup_state = "failed"
                    self._build_environment_warmup_error = f"{type(exc).__name__}: {exc}"
                    stopped = False
            if stopped:
                self.logger.info("domain_build_environment_warmup_preempted")
            else:
                self.logger.warning("domain_build_environment_warmup_failed error=%s", exc)
        else:
            with self._build_environment_warmup_lock:
                self._build_environment_warmup_state = "ready"
            self.logger.info(
                "domain_build_environment_warmup_completed elapsed_seconds=%.3f",
                time.perf_counter() - started,
            )
        finally:
            with self._build_environment_warmup_lock:
                self._build_environment_warmup_process = None
                self._build_environment_warmup_done.set()

    def _preempt_optional_build_environment_warmup(self) -> bool:
        """Let a real build own module loading instead of waiting for warm-up.

        The helper process only warms operating-system caches; it does not
        import the module into this API process.  Waiting for it before the
        real import serializes the same expensive work twice on a cold start.
        """

        with self._build_environment_warmup_lock:
            if self._build_environment_warmup_state != "warming":
                return False
            self._build_environment_warmup_stop.set()
            process = self._build_environment_warmup_process
        if process is not None:
            self._reclaim_warmup_process(process)
        return True

    def _load_build_environment_for_task(self) -> Any:
        self._preempt_optional_build_environment_warmup()
        try:
            module = _domain_adaptation_module()
        except Exception as exc:
            with self._build_environment_warmup_lock:
                self._build_environment_warmup_state = "failed"
                self._build_environment_warmup_error = f"{type(exc).__name__}: {exc}"
                self._build_environment_warmup_done.set()
            raise
        with self._build_environment_warmup_lock:
            self._build_environment_warmup_state = "ready"
            self._build_environment_warmup_error = ""
            self._build_environment_warmup_done.set()
        return module

    def _resolve_build_feedback_samples(self, task: DomainBuildTask) -> None:
        candidate_ids = [
            str(value)
            for value in list(task.payload.get("feedback_candidate_ids") or [])
            if str(value)
        ]
        if not candidate_ids:
            task.payload["_feedback_samples"] = []
            return
        if self.feedback_candidates is None:
            raise ApiError(
                "FEEDBACK_SAMPLE_SERVICE_UNAVAILABLE",
                "已审核反馈样本服务尚未就绪，无法准备本次目标域构建。",
                503,
            )
        task.payload["_feedback_samples"] = self.feedback_candidates.domain_build_candidates(
            candidate_ids,
            include_internal_paths=True,
        )

    def _mark_worker_acquired(self, task: DomainBuildTask) -> None:
        now = _utc_now()
        with self._lock:
            task.state = "running"
            task.lifecycle_phase = "preparing_environment"
            task.stage = "准备构建环境"
            task.message = "后台工作线程已接收任务，正在加载构建模块并校验资源。"
            task.worker_acquired_at = now
            task.first_progress_at = task.first_progress_at or now
            task.started_at = task.started_at or now
            task.started_monotonic = task.started_monotonic or time.monotonic()
            task.heartbeat_at = now
            task.last_progress_at = now
            self._persist_task(task)
        self.logger.info(
            "domain_build_timing job_id=%s event=worker_acquired at=%s",
            task.job_id,
            now,
        )

    def _prepare_build_environment(
        self,
        task: DomainBuildTask,
    ) -> tuple[Any, tuple[dict[str, Any], str, str, str, int]]:
        domain_adaptation = self._load_build_environment_for_task()
        self._resolve_build_feedback_samples(task)
        validated = self._validated_build_config(task.payload)
        return domain_adaptation, validated

    def _mark_build_started(self, task: DomainBuildTask) -> None:
        now = _utc_now()
        with self._lock:
            task.lifecycle_phase = "building"
            task.build_started_at = task.build_started_at or now
            task.stage = "正式构建"
            task.message = "构建环境已就绪，正在执行目标域资源构建。"
            task.last_progress_at = now
            task.heartbeat_at = now
            self._persist_task(task)
        self.logger.info(
            "domain_build_timing job_id=%s event=build_started at=%s",
            task.job_id,
            now,
        )

    def start_build(
        self,
        payload: dict[str, Any],
        *,
        api_received_at: str | None = None,
    ) -> dict[str, Any]:
        # Reject retired modes synchronously.  This keeps hidden API parameters
        # and legacy config templates from creating a doomed background job.
        payload = dict(payload)
        payload["method_mode"] = self._creation_mode(payload)
        requested_precision = self._requested_precision(payload)
        payload["precision"] = requested_precision
        payload["use_fp16"] = precision_uses_fp16(requested_precision)
        if self.test_lock is not None:
            self.test_lock.require_write_allowed("domain_build", context=payload)
        if bool(payload.get("overwrite_existing")):
            raise ApiError("FROZEN_PROFILE_OVERWRITE_FORBIDDEN", "已冻结记忆库不允许覆盖；请创建新版本。", 422)
        idempotency_key = str(payload.get("idempotency_key") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9._:-]{16,128}", idempotency_key):
            raise ApiError("INVALID_BUILD_IDEMPOTENCY_KEY", "构建幂等标识无效，请重新提交。", 422)
        request_fingerprint = _request_fingerprint(payload)
        with self._lock:
            existing_job_id = self._tasks_by_idempotency.get(idempotency_key)
            if existing_job_id and existing_job_id in self._tasks:
                existing = self._tasks[existing_job_id]
                if existing.request_fingerprint != request_fingerprint:
                    raise ApiError(
                        "DOMAIN_BUILD_IDEMPOTENCY_CONFLICT",
                        "同一构建提交标识对应的参数已变化，请重新发起构建。",
                        409,
                    )
                return self.task_status(existing_job_id)
            find_durable = getattr(getattr(self.jobs, "states", None), "find_by_idempotency", None)
            durable = find_durable("domain_adaptation", idempotency_key) if callable(find_durable) else None
            if durable is not None:
                durable_payload = dict(durable.get("payload") or {})
                durable_fingerprint = str(
                    durable.get("request_fingerprint")
                    or durable_payload.get("request_fingerprint")
                    or ""
                )
                if not durable_fingerprint:
                    prior_request = dict(durable_payload.get("request_payload") or {})
                    if prior_request:
                        durable_fingerprint = _request_fingerprint(prior_request)
                if durable_fingerprint != request_fingerprint:
                    raise ApiError(
                        "DOMAIN_BUILD_IDEMPOTENCY_CONFLICT",
                        "同一构建提交标识对应的参数已变化，请重新发起构建。",
                        409,
                    )
                return self.task_status(str(durable.get("task_id") or durable_payload.get("job_id") or ""))
            active = self._tasks.get(self._active_job_id or "")
            if active and active.state in {"queued", "running", "cancelling"}:
                raise ApiError("DOMAIN_BUILD_ALREADY_ACTIVE", "已有目标域适配任务正在运行，不能重复启动。", 409)
            job_id = uuid.uuid4().hex
            task_created_at = _utc_now()
            task = DomainBuildTask(
                job_id=job_id,
                payload=dict(payload),
                idempotency_key=idempotency_key,
                request_fingerprint=request_fingerprint,
                created_at=task_created_at,
                api_received_at=api_received_at or task_created_at,
                task_created_at=task_created_at,
                queued_at=_utc_now(),
            )
            self._initialize_stages(task)
            claim_idempotent = getattr(
                getattr(self.jobs, "states", None),
                "claim_idempotent_task",
                None,
            )
            if callable(claim_idempotent):
                public = self._task_payload(task)
                public["request_payload"] = dict(task.payload)
                durable, claimed = claim_idempotent(
                    task.job_id,
                    "domain_adaptation",
                    task.state,
                    public,
                    idempotency_key=task.idempotency_key,
                    request_fingerprint=task.request_fingerprint,
                )
                if not claimed:
                    durable_payload = dict(durable.get("payload") or {})
                    durable_fingerprint = str(
                        durable.get("request_fingerprint")
                        or durable_payload.get("request_fingerprint")
                        or ""
                    )
                    if not durable_fingerprint:
                        prior_request = dict(durable_payload.get("request_payload") or {})
                        if prior_request:
                            durable_fingerprint = _request_fingerprint(prior_request)
                    if durable_fingerprint != request_fingerprint:
                        raise ApiError(
                            "DOMAIN_BUILD_IDEMPOTENCY_CONFLICT",
                            "同一构建提交标识对应的参数已变化，请重新发起构建。",
                            409,
                        )
                    durable_job_id = str(durable.get("task_id") or durable_payload.get("job_id") or "")
                    return self.task_status(durable_job_id)
            self._tasks[job_id] = task
            self._tasks_by_idempotency[idempotency_key] = job_id
            self._active_job_id = job_id
            if not callable(claim_idempotent):
                self._persist_task(task)
            task.future = self._build_executor.submit(self._run_build, task)
            heartbeat = threading.Thread(
                target=self._heartbeat_worker,
                args=(task,),
                name=f"AdaptiveAOI-Domain-Heartbeat-{job_id[:8]}",
                daemon=True,
            )
            task.heartbeat_thread = heartbeat
            heartbeat.start()
            self.logger.info(
                "domain_build_timing job_id=%s event=task_created api_received_at=%s "
                "task_created_at=%s queued_at=%s",
                job_id,
                task.api_received_at,
                task.task_created_at,
                task.queued_at,
            )
        return self.task_status(job_id)

    def _progress(self, task: DomainBuildTask, stage: str, current: int, total: int, message: str) -> None:
        stage_labels = {
            "loading_model": "加载模型",
            "video_sampling": "视频抽帧与筛选",
            "building_memory": "生成记忆库",
            "calibrating_threshold": "阈值校准",
            "mining_defect_patches": "缺陷补丁挖掘",
            "validating_dual_memory": "双记忆可靠性验证",
            "calibrating_lightweight": "轻量校准",
            "calibration_complete": "校准完成",
        }
        with self._lock:
            task.stage = stage_labels.get(stage, stage)
            task.current = current
            task.total = total
            task.message = message
            if stage == "building_memory":
                task.normal_processed = min(current, int(task.payload.get("normal_count", current)))
            elif stage == "calibrating_threshold":
                normal_count = int(task.payload.get("normal_count", 0))
                task.normal_processed = normal_count
                task.defect_processed = max(0, current - 2 * normal_count)
            if task.started_monotonic is not None:
                task.elapsed_seconds = time.monotonic() - task.started_monotonic
        ratio = min(1.0, max(0.0, float(current) / float(total))) if total else 0.0
        if stage == "loading_model":
            self._set_stage(task, "feature_extraction", progress=0.0, message=message)
        elif stage == "video_sampling":
            self._set_stage(task, "video_preparation", progress=ratio, message=message)
        elif stage == "building_memory":
            normal_total = max(1, int(task.payload.get("normal_count") or current or 1))
            feature_ratio = min(0.98, max(0.0, float(current) / float(normal_total)))
            self._set_stage(task, "feature_extraction", progress=feature_ratio, message=message)
            if current >= normal_total:
                self._set_stage(task, "memory_index", progress=0.05, message="正常特征已提取，正在生成 coreset、Memory Bank 与检索索引")
        elif stage == "calibrating_threshold":
            self._set_stage(task, "feature_extraction", status="completed", progress=1.0, message="图像预处理与特征提取完成")
            self._set_stage(task, "memory_index", status="completed", progress=1.0, message="正常记忆、coreset 与检索索引已生成")
            self._set_stage(task, "threshold_calibration", progress=min(0.95, ratio), message=message)
        elif stage in {"calibrating_lightweight", "mining_defect_patches", "validating_dual_memory"}:
            self._set_stage(task, "defect_extension", progress=min(0.95, ratio), message=message)
        elif stage == "calibration_complete":
            self._set_stage(task, "threshold_calibration", status="completed", progress=1.0, message="判定阈值生成与校准完成")
            extension = self._stage(task, "defect_extension")
            if extension.get("status") != "skipped":
                self._set_stage(task, "defect_extension", status="completed", progress=1.0, message="当前方法的缺陷扩展步骤完成")
        else:
            self._persist_task(task)

    @staticmethod
    def _write_manifest(path: Path, scan: dict[str, Any], role: str) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_role", "relative_path", "size_bytes", "sha256"])
            writer.writeheader()
            for item in scan["valid"]:
                writer.writerow(
                    {
                        "sample_role": role,
                        "relative_path": item["relative_path"],
                        "size_bytes": item["size_bytes"],
                        "sha256": item["sha256"],
                    }
                )

    @staticmethod
    def _write_sample_manifest(path: Path, rows: list[dict[str, Any]], role: str) -> None:
        fields = [
            "sample_id", "sample_role", "source_type", "source_video_id", "source_video_name",
            "relative_path", "frame_index", "timestamp_ms", "original_fps", "width", "height",
            "frame_hash", "extraction_method", "quality_score", "selected", "usage",
            "defect_interval", "anomaly_confidence", "enters_defect_prototype",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                item = dict(row)
                item["sample_role"] = role
                item["relative_path"] = str(item.get("relative_path") or Path(str(item.get("path") or "")).name)
                item["defect_interval"] = json.dumps(item.get("defect_interval") or [], ensure_ascii=False, separators=(",", ":"))
                item["enters_defect_prototype"] = bool(item.get("enters_defect_prototype", False))
                writer.writerow(item)

    def _image_sample_rows(self, scan: dict[str, Any], role: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        root = scan.get("root")
        for item in scan.get("valid") or []:
            path = Path(str(item["path"]))
            width = height = 0
            try:
                with Image.open(path) as image:
                    width, height = image.size
                    pixel_hash = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
            except (OSError, UnidentifiedImageError):
                continue
            digest = pixel_hash or str(item.get("sha256") or _sha256(path))
            rows.append(
                {
                    "sample_id": f"image_{digest[:16]}",
                    "source_type": str(item.get("source_type") or "image"),
                    "source_video_id": "",
                    "source_video_name": "",
                    "frame_index": "",
                    "timestamp_ms": "",
                    "original_fps": "",
                    "width": width,
                    "height": height,
                    "frame_hash": digest,
                    "extraction_method": "reviewed_feedback_snapshot" if item.get("source_type") == "feedback" else "image_file",
                    "quality_score": "",
                    "selected": True,
                    "usage": "normal_memory" if role == "normal" else "defect_calibration_only",
                    "enters_defect_prototype": False,
                    "path": str(path),
                    "relative_path": str(item.get("relative_path") or path.name),
                    "feedback_candidate_id": item.get("feedback_candidate_id") or "",
                    "mask_path": item.get("mask_path") or "",
                }
            )
        return rows

    def _materialize_source_samples(
        self,
        task: DomainBuildTask,
        payload: dict[str, Any],
        role: str,
        target_count: int,
        intervals: Any = None,
    ) -> tuple[dict[str, Any], list[Path], list[dict[str, Any]], dict[str, Any] | None]:
        scan = self._source_scan(payload, "正常" if role == "normal" else "缺陷校准")
        image_rows = self._image_sample_rows(scan, role)
        # Keep every scanned row for the frozen audit manifest, while the
        # selected rows are tracked separately so a video manifest's
        # budget-rejected candidates cannot be paired with the returned paths.
        rows = list(image_rows)
        selected_rows = list(image_rows)
        paths = [Path(str(row["path"])) for row in selected_rows]
        video_result: dict[str, Any] | None = None
        source_type = str(scan.get("source_type") or "image").lower()
        if source_type in {"video", "mixed"}:
            prefix = "normal" if role == "normal" else "defect"
            files, dirs = self._validate_video_inputs(
                [str(item) for item in list(payload.get(f"{prefix}_video_files") or []) if str(item).strip()],
                str(payload.get(f"{prefix}_video_dir") or ""),
                "正常" if role == "normal" else "缺陷校准",
            )
            video_config = dict(payload.get("video_sampling_config") or {})
            if str(payload.get("sample_budget_mode") or "final_frame_count") == "video_file_count":
                video_count = int((scan.get("video_summary") or {}).get("valid_video_count") or 0)
                # In file-count mode each valid source video contributes one
                # representative frame; the numeric target is not used to
                # duplicate frames or to fabricate missing videos.
                video_config["max_frames_per_video"] = 1
                video_config["final_max_samples"] = max(1, video_count)
                target_count = max(1, video_count)
            video_result = self.video_samples.extract(
                role,
                files,
                dirs,
                video_config,
                target_count=target_count,
                intervals=intervals,
                task_id=task.job_id,
                progress=lambda stage, current, total, message: self._progress(task, stage, current, total, message),
                cancelled=task.cancel_event.is_set,
                excluded_sample_ids=payload.get("excluded_sample_ids") or [],
            )
            video_paths = [Path(str(item)) for item in video_result.get("paths") or []]
            video_manifest = list(video_result.get("manifest") or [])
            selected_video_paths = set(video_paths)
            selected_video_rows = [
                row for row in video_manifest
                if row.get("selected") and Path(str(row.get("path"))) in selected_video_paths
            ]
            paths.extend(video_paths)
            selected_rows.extend(selected_video_rows)
            rows.extend(video_manifest)

        # For image-only and mixed sources, the target count is a real sample
        # budget rather than a display hint.  Selection is deterministic and
        # round-robin by source video so a mixed build does not silently spend
        # the entire budget on the first image folder.  In explicit
        # video-file-count mode each valid video already contributes one frame;
        # image inputs remain available as ordinary source samples.
        budget_mode = str(payload.get("sample_budget_mode") or "final_frame_count")
        valid_video_count = int((video_result or {}).get("summary", {}).get("valid_video_count") or 0)
        if budget_mode != "video_file_count" or valid_video_count <= 0:
            paths, selected_rows = self._limit_selected_source_rows(
                paths,
                selected_rows,
                max(1, int(target_count)),
                int((payload.get("video_sampling_config") or {}).get("seed", 42)),
            )
        if video_result is not None:
            # A mixed-source budget may trim some of the initially extracted
            # video representatives after images and videos are balanced.
            # Keep the public summary truthful about frames actually handed to
            # the feature builder.
            video_result.setdefault("summary", {})["selected_frame_count"] = sum(
                1 for row in selected_rows if row.get("source_type") == "video" and row.get("selected", True)
            )
        return scan, paths, rows, video_result

    @staticmethod
    def _limit_selected_source_rows(
        paths: list[Path],
        selected_rows: list[dict[str, Any]],
        limit: int,
        seed: int,
    ) -> tuple[list[Path], list[dict[str, Any]]]:
        if limit <= 0 or len(paths) <= limit:
            return paths, selected_rows
        buckets: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
        for path, row in zip(paths, selected_rows):
            key = str(row.get("source_video_id") or "image")
            buckets.setdefault(key, []).append((path, row))
        keys = sorted(buckets)
        if keys:
            offset = int(seed) % len(keys)
            keys = keys[offset:] + keys[:offset]
        chosen: list[tuple[Path, dict[str, Any]]] = []
        while len(chosen) < limit and any(buckets.values()):
            for key in keys:
                bucket = buckets.get(key) or []
                if bucket and len(chosen) < limit:
                    chosen.append(bucket.pop(0))
        chosen_paths = [item[0] for item in chosen]
        chosen_ids = {id(item[1]) for item in chosen}
        for row in selected_rows:
            if id(row) not in chosen_ids:
                row["selected"] = False
        for path, row in zip(paths, selected_rows):
            if id(row) not in chosen_ids and str(row.get("source_type")) == "video":
                try:
                    path.unlink()
                except OSError:
                    pass
        return chosen_paths, [item[1] for item in chosen]

    def _materialize_task_dataset(
        self,
        task: DomainBuildTask,
        *,
        defect_optional: bool,
    ) -> dict[str, Any]:
        """Create one immutable in-task dataset manifest and reuse it.

        The public frozen CSV files remain role-specific for compatibility,
        while directory traversal, decoding, hashes and video extraction are
        performed exactly once for this task.
        """

        if task.dataset_manifest is not None:
            return task.dataset_manifest
        normal = self._materialize_source_samples(
            task,
            task.payload,
            "normal",
            int(task.payload.get("normal_target_count") or 100),
        )
        try:
            defect = (
                (
                    self._empty_scan("缺陷校准", str(task.payload.get("defect_source_type") or "image")),
                    [],
                    [],
                    None,
                )
                if defect_optional
                else self._materialize_source_samples(
                    task,
                    task.payload,
                    "defect",
                    int(task.payload.get("defect_target_count") or 30),
                    intervals=task.payload.get("defect_video_intervals"),
                )
            )
        except Exception:
            normal_video_result = normal[3]
            if isinstance(normal_video_result, dict):
                self.video_samples.cleanup(normal_video_result)
            raise
        task.dataset_manifest = {
            "created_at": _utc_now(),
            "normal": normal,
            "defect": defect,
        }
        return task.dataset_manifest

    def _run_build(self, task: DomainBuildTask) -> None:
        partial: Path | None = None
        published_final: Path | None = None
        created_config: tuple[str, int] | None = None
        domain_adaptation = None
        try:
            # Publish a durable first progress snapshot before any lazy ML
            # import or resource validation.  A cold PyTorch import can take
            # tens of seconds, but it is now visible as real preparation rather
            # than an unexplained queued gap.
            self._mark_worker_acquired(task)
            domain_adaptation, validated = self._prepare_build_environment(task)
            config, model_id, device, config_profile_id, config_profile_version = validated
            self._mark_build_started(task)
            self._set_stage(task, "initialization", message="构建环境已就绪，正在解析正式构建参数")
            requested_mode = str(task.payload.get("method_mode") or DEFAULT_CREATE_MODE).lower()
            profile_id = str(task.payload.get("config_profile_id") or "")
            profile_version = int(task.payload.get("config_profile_version") or 0)
            if profile_id and profile_version:
                bound = self.config_profiles.get(profile_id, profile_version)
                requested_mode = str((bound.get("config") or {}).get("method_mode") or requested_mode).lower()
            defect_optional = requested_mode == "normal_only"
            requested_mode = str(config.get("method_mode") or requested_mode).lower()
            defect_optional = requested_mode == "normal_only"
            task.payload["device"] = device
            task.payload["config_profile_id"] = config_profile_id
            task.payload["config_profile_version"] = config_profile_version
            self._set_stage(task, "initialization", status="completed", progress=1.0, message="构建参数与基础模型配置已就绪")
            self._set_stage(task, "sample_validation", message="正在扫描正常与缺陷来源并检查样本合法性")
            dataset_manifest = self._materialize_task_dataset(task, defect_optional=defect_optional)
            normal, normal_paths, normal_rows, normal_video_result = dataset_manifest["normal"]
            defect, defect_paths, defect_rows, defect_video_result = dataset_manifest["defect"]
            if not normal_paths or (not defect_paths and not defect_optional):
                raise ApiError(
                    "DOMAIN_SAMPLES_REQUIRED",
                    "正常样本必须至少包含一张有效图片或视频代表帧；当前方法模式还需要有效缺陷样本。",
                    422,
                )
            normal_target = int(task.payload.get("normal_target_count") or 100)
            defect_target = int(task.payload.get("defect_target_count") or 30)
            # ``normal_target_count``/``defect_target_count`` describe the
            # recommended sample budget and the deterministic sampling
            # cap.  They are not a second hidden readiness gate: the public
            # readiness endpoint already reports a yellow warning when the
            # available data is below that target.  A build with at least one
            # valid sample for every required role must therefore remain
            # executable, otherwise the UI can truthfully say "可创建" while
            # the worker immediately fails with a contradictory hard block.
            budget_warnings: list[str] = []
            if len(normal_paths) < normal_target:
                budget_warnings.append(
                    f"正常样本实际 {len(normal_paths)}，低于建议目标 {normal_target}"
                )
            if not defect_optional and len(defect_paths) < defect_target:
                budget_warnings.append(
                    f"缺陷样本实际 {len(defect_paths)}，低于建议目标 {defect_target}"
                )
            task.payload["sample_budget_warnings"] = budget_warnings
            normal_root = normal.get("root")
            defect_root = defect.get("root")
            if defect_paths and normal_root and defect_root and (normal_root == defect_root or _is_within(normal_root, defect_root) or _is_within(defect_root, normal_root)):
                raise ApiError("SAMPLE_DIRECTORIES_OVERLAP", "正常样本和缺陷校准样本必须使用彼此独立的文件夹。", 422)
            normal_hashes = {str(item.get("frame_hash") or item.get("sha256")) for item in normal_rows}
            defect_hashes = {str(item.get("frame_hash") or item.get("sha256")) for item in defect_rows}
            duplicate_hashes = normal_hashes & defect_hashes if defect_rows else set()
            if duplicate_hashes:
                raise ApiError("SAMPLE_CONTENT_OVERLAP", "正常与缺陷样本包含相同图片或视频帧，已阻止冻结。", 422)
            video_stage = self._stage(task, "video_preparation")
            if video_stage.get("status") != "skipped":
                self._set_stage(
                    task,
                    "video_preparation",
                    status="completed",
                    progress=1.0,
                    message="视频代表帧抽取、质量筛选与去重完成",
                )
            sample_status = "warning" if budget_warnings else "completed"
            sample_message = "；".join(budget_warnings) if budget_warnings else "样本扫描、可读性、角色隔离与重复检查通过"
            self._set_stage(task, "sample_validation", status=sample_status, progress=1.0, message=sample_message)
            task.payload["normal_count"] = len(normal_paths)
            task.payload["defect_count"] = len(defect_paths)
            task.payload["normal_source_summary"] = normal.get("video_summary") or {}
            task.payload["defect_source_summary"] = defect.get("video_summary") or {}
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            profile_id = f"{_safe_token(config['dataset'], 'dataset')}_{_safe_token(config['category'], 'category')}_{stamp}_{uuid.uuid4().hex[:8]}"
            partial = self.root / f".{profile_id}.partial"
            final = self.root / profile_id
            if partial.exists() or final.exists():
                raise ApiError("DOMAIN_PROFILE_CONFLICT", "目标域记忆库版本目录发生冲突，请重新启动任务。", 409)
            partial.mkdir(parents=True, exist_ok=False)
            config["output_root"] = str(partial)
            self._set_stage(task, "feature_extraction", message="正在加载特征资源并准备图像预处理")
            with self.registry.exclusive():
                # Release the matching inference runtime before loading the one
                # adaptation extractor. No test inference can overlap this write phase.
                self.registry.invalidate(config["dataset"], config["category"])
                algorithm_result = domain_adaptation.build_and_calibrate_domain(
                    [str(item) for item in normal_paths],
                    [str(item) for item in defect_paths],
                    config,
                    partial,
                    progress=lambda stage, current, total, message: self._progress(task, stage, current, total, message),
                    cancelled=task.cancel_event.is_set,
                )
            self._set_stage(task, "feature_extraction", status="completed", progress=1.0, message="图像预处理与特征提取完成")
            self._set_stage(task, "memory_index", status="completed", progress=1.0, message="Memory Bank、coreset 与检索索引构建完成")
            self._set_stage(task, "threshold_calibration", status="completed", progress=1.0, message="判定阈值生成与校准完成")
            extension = self._stage(task, "defect_extension")
            if extension.get("status") != "skipped":
                self._set_stage(task, "defect_extension", status="completed", progress=1.0, message="当前方法的缺陷扩展步骤完成")
            if task.cancel_event.is_set():
                raise domain_adaptation.DomainAdaptationCancelled("目标域适配任务已取消。")

            publish_mode = str(task.payload.get("publish_mode") or "frozen").lower()
            is_candidate = publish_mode == "candidate"
            self._set_stage(
                task,
                "config_profile",
                message="正在组装候选配置快照与完整目标域 Profile" if is_candidate else "正在生成并冻结有效配置与目标域 Profile",
            )
            threshold = algorithm_result["threshold"]
            resolved_mode = str(threshold.get("method_mode_resolved") or requested_mode)
            if resolved_mode not in CREATABLE_DOMAIN_METHOD_MODES:
                raise ApiError(
                    "DOMAIN_BUILD_MODE_RETIRED",
                    "构建器返回了已停止用于新建资源的历史模式，资源发布已终止。",
                    409,
                )
            precision_metadata = dict(algorithm_result.get("precision_metadata") or {})
            config.update(
                precision=str(precision_metadata.get("requested_precision") or config.get("precision") or "fp16"),
                requested_precision=str(precision_metadata.get("requested_precision") or config.get("requested_precision") or "fp16"),
                actual_precision=str(precision_metadata.get("actual_precision") or config.get("actual_precision") or "fp32"),
                amp_enabled=bool(precision_metadata.get("amp_enabled", config.get("amp_enabled", False))),
                feature_dtype=str(precision_metadata.get("feature_dtype") or "unknown"),
                memory_bank_dtype=str(precision_metadata.get("memory_bank_dtype") or "float32"),
                index_dtype=str(precision_metadata.get("index_dtype") or "float32"),
                precision_fallback_reason=str(precision_metadata.get("fallback_reason") or config.get("precision_fallback_reason") or ""),
            )
            task.payload.update(
                requested_precision=config["requested_precision"],
                actual_precision=config["actual_precision"],
                amp_enabled=config["amp_enabled"],
            )
            complexity_snapshot = (
                json.loads((partial / "complexity_summary.json").read_text(encoding="utf-8"))
                if (partial / "complexity_summary.json").is_file()
                else {}
            )
            calibration_snapshot = dict(algorithm_result.get("calibration_summary") or {})
            evaluation_fields = (
                self.test_lock.evaluation_fields(calibration_snapshot)
                if self.test_lock is not None
                else {
                    "evaluation_mode": "calibration_only",
                    "calibration_result": calibration_snapshot,
                    "validation_result": {"status": "not_run", "message": "校准集指标不是最终测试精度。"},
                }
            )
            config_snapshot_id = f"snapshot_{hashlib.sha256(profile_id.encode('utf-8')).hexdigest()[:24]}"
            snapshot_config = dict(config)
            snapshot_config.update(
                config_profile_id=config_snapshot_id,
                config_profile_version=1,
                target_domain_version=profile_id,
                build_job_id=task.job_id,
                threshold=float(threshold["image_threshold"]),
                pixel_threshold=threshold.get("pixel_threshold"),
                inference_mode="inference_only",
                method_mode_resolved=resolved_mode,
                algorithm_mode=resolved_mode,
                normal_memory_enabled=True,
                lightweight_calibration_enabled=resolved_mode == LIGHTWEIGHT_CREATE_MODE,
                defect_guidance_enabled=resolved_mode == LIGHTWEIGHT_CREATE_MODE,
                use_dual_memory=resolved_mode == "normal_defect_dual_memory",
                dual_memory_alpha=float(threshold.get("dual_memory_alpha") or 0.0),
                lightweight_calibrator_path="lightweight_calibrator.json" if resolved_mode == "normal_memory_defect_guided_lite" else "",
                calibration_snapshot=calibration_snapshot,
                calibration_result=evaluation_fields["calibration_result"],
                validation_result=evaluation_fields["validation_result"],
                evaluation_mode=evaluation_fields["evaluation_mode"],
                data_roles={"memory_build": "normal_build_set", "calibration": "defect_calibration_set", "test": "official_test_set"},
                official_test_mode=False,
                complexity_snapshot=complexity_snapshot,
                video_sampling_config=dict(task.payload.get("video_sampling_config") or {}),
                memory_bank_write_policy="frozen_read_only",
                test_data_update_allowed=False,
            )
            created = self.config_profiles.create_from_template(
                {
                    "config_profile_id": config_snapshot_id,
                    "friendly_name": str(task.payload.get("config_profile_name") or "初始化配置"),
                    "note": str(task.payload.get("note") or ""),
                    "source_kind": "target_domain_snapshot",
                    "config_origin": "target_domain_initial",
                    "config": snapshot_config,
                    "target_domain_version": profile_id,
                    "build_job_id": task.job_id,
                    "build_idempotency_key": task.idempotency_key,
                    "ordinary_visible": True,
                },
                template_profile_id=str(task.payload.get("template_config_profile_id") or "") or None,
                template_version=int(task.payload.get("template_config_profile_version") or 0) or None,
            )
            config_profile_id = str(created["config_profile_id"])
            config_profile_version = int(created["version"])
            created_config = (config_profile_id, config_profile_version)
            self.config_profiles.validate(config_profile_id, config_profile_version)
            task.payload["config_profile_id"] = config_profile_id
            task.payload["config_profile_version"] = config_profile_version
            self._set_stage(task, "config_profile", progress=0.65, message="配置快照已生成，正在写入样本清单与生效配置")
            self._write_sample_manifest(partial / "normal_samples_manifest.csv", normal_rows, "normal_memory_source")
            defect_role = (
                "defect_calibration_and_separate_prototype_source"
                if resolved_mode == "normal_defect_dual_memory"
                else ("not_used_by_normal_only_mode" if resolved_mode == "normal_only" else ("defect_lightweight_calibration_only" if resolved_mode == "normal_memory_defect_guided_lite" else "defect_threshold_only"))
            )
            self._write_sample_manifest(partial / "defect_calibration_manifest.csv", defect_rows, defect_role)
            self._write_sample_manifest(partial / "defect_samples_manifest.csv", defect_rows, defect_role)
            frozen_profile, frozen_config_hash = self.config_profiles.frozen_snapshot(
                config_profile_id,
                config_profile_version,
                snapshot_config,
                float(threshold["image_threshold"]),
            )
            effective_config = dict(frozen_profile)
            effective_config.update(
                config_profile_id=config_profile_id,
                config_profile_version=config_profile_version,
                config_hash=frozen_config_hash,
                threshold=float(threshold["image_threshold"]),
                inference_mode="inference_only",
                method_mode_resolved=resolved_mode,
                algorithm_mode=resolved_mode,
                normal_memory_enabled=True,
                lightweight_calibration_enabled=resolved_mode == LIGHTWEIGHT_CREATE_MODE,
                defect_guidance_enabled=resolved_mode == LIGHTWEIGHT_CREATE_MODE,
                use_dual_memory=resolved_mode == "normal_defect_dual_memory",
                dual_memory_alpha=float(threshold.get("dual_memory_alpha") or 0.0),
                lightweight_calibrator_path="lightweight_calibrator.json" if resolved_mode == "normal_memory_defect_guided_lite" else "",
            )
            effective_config["output_root"] = "由推理任务动态指定"
            effective_config["memory_bank_write_policy"] = "frozen_read_only"
            effective_config["test_data_update_allowed"] = False
            effective_config["official_test_mode"] = False
            effective_config["evaluation_mode"] = evaluation_fields["evaluation_mode"]
            effective_config["calibration_result"] = evaluation_fields["calibration_result"]
            effective_config["validation_result"] = evaluation_fields["validation_result"]
            (partial / "effective_config.yaml").write_text(yaml.safe_dump(effective_config, allow_unicode=True, sort_keys=False), encoding="utf-8")

            threshold_path = partial / "threshold.json"
            threshold_payload = json.loads(threshold_path.read_text(encoding="utf-8"))
            threshold_payload.update(
                evaluation_mode=evaluation_fields["evaluation_mode"],
                metric_data_role="defect_calibration_set",
                final_test_metrics_available=False,
                test_data_update_allowed=False,
            )
            threshold_path.write_text(json.dumps(threshold_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self._set_stage(task, "config_profile", status="completed", progress=1.0, message="Config 与目标域 Profile 文件已生成")
            self._set_stage(task, "finalizing", progress=0.05, message="正在完成资源创建")

            artifact_names = [
                "memory_bank.pt",
                "memory_bank.json",
                "coreset_indices.pt",
                "normal_coreset_index.pt",
                "threshold.json",
                "scoring_config.json",
                "fusion_parameters.json",
                "complexity_summary.json",
                "effective_config.yaml",
                "normal_samples_manifest.csv",
                "defect_calibration_manifest.csv",
                "defect_samples_manifest.csv",
                "calibration_summary.json",
            ]
            if algorithm_result.get("defect_prototype_path"):
                artifact_names.extend(["defect_prototypes.pt", "defect_prototype_index.pt"])
            if algorithm_result.get("lightweight_calibrator_path"):
                artifact_names.append("lightweight_calibrator.json")
            hashes = {name: _sha256(partial / name) for name in artifact_names}
            memory_meta = algorithm_result["memory_metadata"]
            normal_source_types = []
            defect_source_types = []
            if any(row.get("source_type") == "image" and row.get("selected", True) for row in normal_rows):
                normal_source_types.append("image")
            if normal_video_result and int((normal_video_result.get("summary") or {}).get("valid_video_count") or 0) > 0:
                normal_source_types.append("video")
            if any(row.get("source_type") == "image" and row.get("selected", True) for row in defect_rows):
                defect_source_types.append("image")
            if defect_video_result and int((defect_video_result.get("summary") or {}).get("valid_video_count") or 0) > 0:
                defect_source_types.append("video")
            metadata = {
                "memory_bank_name": str(task.payload.get("friendly_name") or f"{config['dataset']} / {config['category']} / {stamp}"),
                "memory_bank_id": profile_id,
                "profile_version": 1,
                "created_at": _utc_now(),
                "dataset": config["dataset"],
                "category": config["category"],
                "normal_sample_count": len(normal_paths),
                "defect_calibration_sample_count": len(defect_paths),
                "normal_source_types": normal_source_types or [str(normal.get("source_type") or "image")],
                "defect_source_types": defect_source_types or [str(defect.get("source_type") or "image")],
                "normal_image_count": sum(1 for row in normal_rows if row.get("source_type") == "image" and row.get("selected", True)),
                "normal_video_count": int((normal_video_result or {}).get("summary", {}).get("valid_video_count") or 0),
                "defect_image_count": sum(1 for row in defect_rows if row.get("source_type") == "image" and row.get("selected", True)),
                "defect_video_count": int((defect_video_result or {}).get("summary", {}).get("valid_video_count") or 0),
                "normal_extracted_frame_count": int((normal_video_result or {}).get("summary", {}).get("selected_frame_count") or 0),
                "defect_extracted_frame_count": int((defect_video_result or {}).get("summary", {}).get("selected_frame_count") or 0),
                "deduplicated_frame_count": int((normal_video_result or {}).get("summary", {}).get("deduplicated_frame_count") or 0) + int((defect_video_result or {}).get("summary", {}).get("deduplicated_frame_count") or 0),
                "rejected_frame_count": int((normal_video_result or {}).get("summary", {}).get("rejected_frame_count") or 0) + int((defect_video_result or {}).get("summary", {}).get("rejected_frame_count") or 0),
                "video_sampling_config": dict(task.payload.get("video_sampling_config") or {}),
                "sample_budget_mode": str(task.payload.get("sample_budget_mode") or "final_frame_count"),
                "defect_video_intervals": task.payload.get("defect_video_intervals") or {},
                "source_video_hashes": {
                    "normal": (normal_video_result or {}).get("summary", {}).get("source_video_hashes", {}),
                    "defect": (defect_video_result or {}).get("summary", {}).get("source_video_hashes", {}),
                },
                "frame_selection_seed": int((task.payload.get("video_sampling_config") or {}).get("seed", config.get("random_seed", 42))),
                "video_decoder_version": f"opencv-{getattr(__import__('cv2'), '__version__', 'unknown')}",
                "sample_manifest": {
                    "normal": "normal_samples_manifest.csv",
                    "defect": "defect_calibration_manifest.csv",
                },
                "normal_source_summary": {
                    "source_type": normal.get("source_type"),
                    "folder_name": normal["root"].name if normal.get("root") else "",
                    "content_digest": hashlib.sha256("".join(sorted(normal_hashes)).encode()).hexdigest(),
                    "video_summary": (normal_video_result or {}).get("summary", {}),
                },
                "defect_source_summary": {
                    "source_type": defect.get("source_type"),
                    "folder_name": defect["root"].name if defect.get("root") else "",
                    "content_digest": hashlib.sha256("".join(sorted(defect_hashes)).encode()).hexdigest(),
                    "video_summary": (defect_video_result or {}).get("summary", {}),
                },
                "model_id": model_id,
                "model_version": str(config.get("model_version") or self.settings.backend.get("version", "1.0.0")),
                "base_model_source": str(config.get("base_model_source") or "system_official_model_registry"),
                "base_model_profile_sha256": str(config.get("base_model_profile_sha256") or ""),
                "base_model_weights_sha256": str(config.get("base_model_weights_sha256") or ""),
                "base_model_manifest_sha256": str(config.get("base_model_manifest_sha256") or ""),
                "base_model_integrity_status": str(config.get("base_model_integrity_status") or "unknown"),
                "precision": str(config.get("precision") or "fp16"),
                "requested_precision": str(config.get("requested_precision") or "fp16"),
                "actual_precision": str(config.get("actual_precision") or "fp32"),
                "amp_enabled": bool(config.get("amp_enabled", False)),
                "feature_dtype": str(config.get("feature_dtype") or "unknown"),
                "backbone_output_dtype": str(precision_metadata.get("backbone_output_dtype") or "unknown"),
                "memory_bank_dtype": str(config.get("memory_bank_dtype") or "float32"),
                "index_dtype": str(config.get("index_dtype") or "float32"),
                "precision_fallback_reason": str(config.get("precision_fallback_reason") or ""),
                "config_profile_id": config_profile_id,
                "config_profile_version": config_profile_version,
                "config_snapshot_id": config_profile_id,
                "config_hash": frozen_config_hash,
                "config_snapshot_sha256": _sha256(partial / "effective_config.yaml"),
                "build_job_id": task.job_id,
                "build_idempotency_key": task.idempotency_key,
                "backbone": memory_meta["backbone"],
                "config_profile": str(self.settings.profile_path(model_id).relative_to(self.settings.project_root)).replace("\\", "/"),
                "feature_layers": memory_meta["feature_layers"],
                "feature_dimension": memory_meta["feature_dim"],
                "image_size": memory_meta["image_size"],
                "coreset": {
                    "strategy": memory_meta["coreset_strategy"],
                    "ratio": memory_meta["coreset_ratio"],
                    "source_patch_count": memory_meta["original_memory_size"],
                    "memory_patch_count": memory_meta["coreset_memory_size"],
                },
                "knn_k": memory_meta["knn_k"],
                "distance_metric": "euclidean",
                "threshold": threshold["image_threshold"],
                "threshold_method": threshold["threshold_strategy"],
                "pixel_threshold": threshold.get("pixel_threshold"),
                "method_name": "单正常记忆·缺陷引导轻量校准" if resolved_mode == "normal_memory_defect_guided_lite" else ("可靠性门控双记忆异常检测" if resolved_mode == "normal_defect_dual_memory" else "正常记忆异常检测"),
                "method_mode_requested": threshold.get("method_mode_requested"),
                "method_mode_resolved": resolved_mode,
                "algorithm_mode": resolved_mode,
                "normal_memory_enabled": True,
                "lightweight_calibration_enabled": resolved_mode == LIGHTWEIGHT_CREATE_MODE,
                "defect_guidance_enabled": resolved_mode == LIGHTWEIGHT_CREATE_MODE,
                "base_method": "normal patch memory nearest-neighbour anomaly detection",
                "references": ["PatchCore (CVPR 2022)", "M3DM (CVPR 2023)", "SegAD (CVPR 2024)"],
                "defect_prototype_count": int((algorithm_result.get("calibration_summary") or {}).get("prototype_metadata", {}).get("prototype_count", 0)) if resolved_mode == "normal_defect_dual_memory" else 0,
                "dual_memory_alpha": float(threshold.get("dual_memory_alpha") or 0.0),
                "lightweight_calibration": bool(resolved_mode == "normal_memory_defect_guided_lite"),
                "lightweight_calibrator_parameters": int((algorithm_result.get("calibration_summary") or {}).get("lightweight_profile", {}).get("parameter_count", 0)) if resolved_mode == "normal_memory_defect_guided_lite" else 0,
                "lightweight_calibrator_path": "lightweight_calibrator.json" if resolved_mode == "normal_memory_defect_guided_lite" else None,
                "defect_features_saved": False,
                "additional_ann_count": 0,
                "device": device,
                "random_seed": int(config["random_seed"]),
                "frozen": not is_candidate,
                "profile_status": "candidate" if is_candidate else "frozen",
                "profile_kind": "target_domain",
                "normal_features_only": True,
                "defect_samples_written_to_memory_bank": False,
                "test_data_update_allowed": False,
                "official_test_mode": False,
                "evaluation_mode": evaluation_fields["evaluation_mode"],
                "data_roles": {"memory_build": "normal_build_set", "calibration": "defect_calibration_set", "test": "official_test_set"},
                "compatible_inference_interface": "domain-profile-v2",
                "resource_size_bytes": sum((partial / name).stat().st_size for name in artifact_names if (partial / name).is_file()),
                "calibration_metrics": calibration_snapshot,
                "calibration_result": evaluation_fields["calibration_result"],
                "validation_result": evaluation_fields["validation_result"],
                "complexity_summary": complexity_snapshot,
                "file_sha256": hashes,
                "resource_integrity_status": "passed",
            }
            metadata_path = partial / "metadata.json"
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            lock_payload = {
                "memory_bank_id": profile_id,
                "created_at": _utc_now(),
                "metadata_sha256": _sha256(metadata_path),
                "write_policy": "candidate_read_only" if is_candidate else "inference_only",
            }
            lock_name = "candidate.lock" if is_candidate else "frozen.lock"
            (partial / lock_name).write_text(json.dumps(lock_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            for name, digest in hashes.items():
                if _sha256(partial / name) != digest:
                    raise RuntimeError(f"冻结前文件哈希校验失败：{name}")
            self._set_stage(task, "finalizing", progress=0.55, message="内部完整性确认通过，正在冻结并注册资源")
            os.replace(partial, final)
            published_final = final
            partial = None
            if not is_candidate:
                self.config_profiles.bind_and_freeze(
                    config_profile_id,
                    config_profile_version,
                    profile_id,
                    float(threshold["image_threshold"]),
                    frozen_profile,
                    frozen_config_hash,
                )
                self.registry.invalidate(config["dataset"], config["category"])
            # Publication is complete only after the unified resource catalog
            # can discover the profile and every owned dependency.  A failure
            # here moves the directory to the failed-build isolation area and
            # archives the unbound snapshot in the exception handler below.
            if self.runtime_resources is None:
                raise RuntimeError("统一资源目录服务尚未绑定，不能发布目标域版本。")
            catalog = self.runtime_resources.scan(force=True, include_resources=False)
            resource_type = "candidate_profile" if is_candidate else "target_profile"
            registered = self.runtime_resources.profile_registered(
                profile_id,
                resource_type=resource_type,
                include_archived=False,
            )
            if not registered:
                raise RuntimeError("目标域文件已生成，但统一资源目录未确认注册。")
            task.registration_status = "registered"
            task.resource_catalog_revision = int(catalog.get("resource_catalog_revision") or catalog.get("revision") or 0)
            self._set_stage(task, "finalizing", progress=0.9, message="资源目录已确认，正在提交最终任务状态")
            # Building or freezing creates an available version; applying it is
            # a separate, explicit and atomic operation performed by the user.
            task.result = {
                **self._public_metadata(metadata, is_default=False),
                "registration_status": task.registration_status,
                "resource_catalog_revision": task.resource_catalog_revision,
            }
            self._set_stage(
                task,
                "finalizing",
                status="completed",
                progress=1.0,
                message="目标域资源已完成完整性确认、冻结、注册与事务提交",
            )
            task.state = "completed"
            task.stage = "创建完成"
            task.message = "目标域记忆库、配置和 Profile 已创建并冻结，可选择应用。"
            self._persist_task(task)
        except VideoSampleCancelled as exc:
            task.state = "cancelled"
            task.lifecycle_phase = "cancelled"
            task.stage = "已取消"
            task.message = str(exc)
        except ApiError as exc:
            task.state = "failed"
            preparing = task.build_started_at is None
            task.lifecycle_phase = "failed"
            running = next((item for item in task.stages if item.get("status") == "running"), None)
            task.failed_stage = "preparing_environment" if preparing else str((running or {}).get("stage_id") or "execution")
            task.stage = "准备构建环境失败" if preparing else "构建执行失败"
            task.error = {"code": exc.code, "message": exc.message, "stage": task.failed_stage}
            self.logger.warning("domain_profile_build_failed job_id=%s code=%s", task.job_id, exc.code)
        except Exception as exc:
            if domain_adaptation is not None and isinstance(exc, domain_adaptation.DomainAdaptationCancelled):
                task.state = "cancelled"
                task.lifecycle_phase = "cancelled"
                task.stage = "已取消"
                task.message = str(exc)
            else:
                task.state = "failed"
                preparing = task.build_started_at is None
                task.lifecycle_phase = "failed"
                running = next((item for item in task.stages if item.get("status") == "running"), None)
                task.failed_stage = "preparing_environment" if preparing else str((running or {}).get("stage_id") or "execution")
                task.stage = "准备构建环境失败" if preparing else "构建执行失败"
                task.error = {
                    "code": "DOMAIN_BUILD_PREPARATION_FAILED" if preparing else "DOMAIN_BUILD_FAILED",
                    "message": f"目标域适配失败：{type(exc).__name__}: {exc}",
                    "stage": task.failed_stage,
                }
                self.logger.exception("domain_profile_build_failed job_id=%s", task.job_id)
        finally:
            if task.state == "failed":
                for stage in task.stages:
                    if stage.get("status") == "running":
                        self._set_stage(
                            task,
                            str(stage.get("stage_id")),
                            status="failed",
                            message=str((task.error or {}).get("message") or "当前阶段执行失败"),
                        )
                        break
            if task.state != "completed" and published_final is not None and published_final.exists():
                isolated = self.failed_build_root / f"{published_final.name}_{task.job_id[:8]}"
                try:
                    os.replace(published_final, isolated)
                    task.registration_status = "isolated_after_failure"
                except OSError as exc:
                    self.logger.error("domain_failed_build_isolation_failed job_id=%s error=%s", task.job_id, exc)
            if task.state != "completed" and created_config is not None:
                try:
                    self.config_profiles.archive_draft(*created_config)
                except Exception as exc:
                    self.logger.error("domain_failed_build_config_archive_failed job_id=%s error=%s", task.job_id, exc)
            if task.state != "completed" and self.runtime_resources is not None:
                try:
                    self.runtime_resources.scan(force=True)
                except Exception as exc:
                    self.logger.error("domain_failed_build_catalog_reconcile_failed job_id=%s error=%s", task.job_id, exc)
            for video_result in (locals().get("normal_video_result"), locals().get("defect_video_result")):
                if isinstance(video_result, dict):
                    self.video_samples.cleanup(video_result)
            if partial is not None and partial.exists():
                shutil.rmtree(partial, ignore_errors=True)
            task.finished_at = _utc_now()
            if task.state == "completed":
                task.lifecycle_phase = "completed"
            if task.started_monotonic is not None:
                task.elapsed_seconds = time.monotonic() - task.started_monotonic
            with self._lock:
                if self._active_job_id == task.job_id:
                    self._active_job_id = None
            self._persist_task(task)

    def _require_task(self, job_id: str) -> DomainBuildTask:
        with self._lock:
            task = self._tasks.get(job_id)
        if task is None:
            raise ApiError("DOMAIN_JOB_NOT_FOUND", "目标域适配任务不存在，或后端已经重启。", 404)
        return task

    def task_status(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            task = self._tasks.get(job_id)
        if task is not None:
            return self._task_payload(task)
        durable = self.jobs.states.get(job_id) if self.jobs is not None else None
        if durable and durable.get("task_kind") == "domain_adaptation":
            payload = dict(durable.get("payload") or {})
            payload.update(status=durable.get("runtime_status") or durable.get("status"), rerunnable=bool(durable.get("rerunnable")))
            return payload
        raise ApiError("DOMAIN_JOB_NOT_FOUND", "目标域适配任务不存在。", 404)

    def current_task(self) -> dict[str, Any] | None:
        """Return the active build or the most recent build for UI recovery."""
        with self._lock:
            active = self._tasks.get(self._active_job_id or "")
            if active is not None:
                job_id = active.job_id
            elif self._tasks:
                job_id = max(self._tasks.values(), key=lambda item: item.created_at).job_id
            else:
                durable = self.jobs.states.latest("domain_adaptation") if self.jobs is not None else None
                if durable is None:
                    return None
                payload = dict(durable.get("payload") or {})
                payload.update(status=durable.get("runtime_status") or durable.get("status"), rerunnable=bool(durable.get("rerunnable")))
                return payload
        return self.task_status(job_id)

    def rerun(self, job_id: str) -> dict[str, Any]:
        durable = self.jobs.states.get(job_id) if self.jobs is not None else None
        if not durable or durable.get("task_kind") != "domain_adaptation":
            raise ApiError("DOMAIN_JOB_NOT_FOUND", "目标域适配任务不存在。", 404)
        if durable.get("status") != "interrupted":
            raise ApiError("DOMAIN_JOB_NOT_INTERRUPTED", "只有因服务重启中断的任务可以重新运行。", 409)
        payload = dict((durable.get("payload") or {}).get("request_payload") or {})
        if not payload:
            raise ApiError("DOMAIN_JOB_RERUN_DATA_MISSING", "中断任务缺少可重新运行的构建参数。", 409)
        payload["idempotency_key"] = f"rerun:{uuid.uuid4().hex}"
        created = self.start_build(payload)
        self.jobs.states.record_rerun(job_id, str(created["job_id"]))
        return created

    def cancel(self, job_id: str) -> dict[str, Any]:
        task = self._require_task(job_id)
        if task.state in {"queued", "running"}:
            task.cancel_event.set()
            task.state = "cancelling"
            task.stage = "正在取消"
            task.message = "正在安全结束当前阶段；不会保留有效的半成品记忆库。"
            self._persist_task(task)
        return self.task_status(job_id)

    def _registry_data(self) -> dict[str, Any]:
        path = self.settings.memory_bank_registry_path
        if not path.exists():
            return {"defaults": {}, "friendly_names": {}, "archived": {}, "unavailable": {}}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {"defaults": {}, "friendly_names": {}, "archived": {}, "unavailable": {}}
        except Exception:
            return {"defaults": {}, "friendly_names": {}, "archived": {}, "unavailable": {}}

    def mark_unavailable(self, profile_ids: list[str], reason: str, resource_id: str = "") -> list[str]:
        changed: list[str] = []
        with self._lock:
            data = self._registry_data()
            unavailable = data.setdefault("unavailable", {})
            defaults = data.setdefault("defaults", {})
            for profile_id in dict.fromkeys(str(value) for value in profile_ids if str(value)):
                unavailable[profile_id] = {
                    "reason": str(reason or "resource_missing"),
                    "resource_id": str(resource_id or ""),
                    "updated_at": _utc_now(),
                }
                for key, value in list(defaults.items()):
                    if str(value) == profile_id:
                        defaults.pop(key, None)
                changed.append(profile_id)
            self._write_registry(data)
        return changed

    def clear_unavailable(self, profile_ids: list[str]) -> list[str]:
        changed: list[str] = []
        with self._lock:
            data = self._registry_data()
            unavailable = data.setdefault("unavailable", {})
            for profile_id in dict.fromkeys(str(value) for value in profile_ids if str(value)):
                if unavailable.pop(profile_id, None) is not None:
                    changed.append(profile_id)
            self._write_registry(data)
        return changed

    def _write_registry(self, payload: dict[str, Any]) -> None:
        path = self.settings.memory_bank_registry_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)

    @staticmethod
    def _default_key(model_id: str, dataset: str, category: str) -> str:
        return f"{model_id}/{str(dataset).lower()}/{category}"

    def _set_default(self, profile_id: str, model_id: str, dataset: str, category: str) -> None:
        with self._lock:
            data = self._registry_data()
            data.setdefault("defaults", {})[self._default_key(model_id, dataset, category)] = profile_id
            self._write_registry(data)

    def build_conditions(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Evaluate a draft without creating configuration or model resources."""
        payload = dict(payload)
        payload["method_mode"] = self._creation_mode(payload)
        requested_precision = self._requested_precision(payload)
        payload["precision"] = requested_precision
        payload["use_fp16"] = precision_uses_fp16(requested_precision)
        items: list[dict[str, Any]] = []

        def add(
            key: str,
            ok: bool,
            title: str,
            summary: str,
            repair: str = "",
            *,
            warning: bool = False,
            applicable: bool = True,
            details: dict[str, Any] | None = None,
        ) -> None:
            if not applicable:
                status, icon = "not_applicable", "—"
            elif warning:
                status, icon = "warning", "⚠️"
            else:
                status, icon = ("passed", "✅") if ok else ("failed", "❌")
            items.append(
                {
                    "id": key,
                    "key": key,
                    "status": status,
                    "icon": icon,
                    "title": title,
                    "summary": summary,
                    "message": summary,
                    "repair": repair,
                    "reason": summary,
                    "resolution": repair,
                    "blocking": bool(applicable and not warning and not ok),
                    "details": dict(details or {}),
                }
            )

        name = str(payload.get("friendly_name") or "").strip()
        dataset = str(payload.get("dataset") or "").strip()
        category = str(payload.get("category") or "").strip()
        mode = str(payload.get("method_mode") or "").strip()
        duplicate = False
        if name:
            for path in self.root.glob("*/metadata.json"):
                try:
                    if str(json.loads(path.read_text(encoding="utf-8")).get("memory_bank_name") or "") == name:
                        duplicate = True
                        break
                except (OSError, ValueError):
                    continue
        name_ok = bool(name) and not duplicate
        add(
            "name",
            name_ok,
            "目标域名称",
            f"已填写：{name}" if name and not duplicate else (f"名称“{name}”已存在，系统不会覆盖已有资源。" if duplicate else "尚未填写目标域名称。"),
            "填写便于识别且未被使用的目标域名称。",
        )
        add(
            "domain",
            bool(dataset and category),
            "数据集与类别",
            f"已选择：{dataset} / {category}" if dataset and category else "数据集或类别尚未设置。",
            "补全数据集与类别；部署时可以填写新的受控业务名称。",
        )

        scan: dict[str, Any] = {}
        scan_error = ""
        normal_count = 0
        defect_count = 0
        normal_video_count = 0
        defect_video_count = 0
        try:
            scan = self.scan_domain_samples(payload)
            normal = dict(scan.get("normal") or {})
            defect = dict(scan.get("defect") or {})
            normal_count = int(normal.get("valid_count") or 0)
            defect_count = int(defect.get("valid_count") or 0)
            normal_video_count = int((normal.get("video_summary") or {}).get("valid_video_count") or 0)
            defect_video_count = int((defect.get("video_summary") or {}).get("valid_video_count") or 0)
        except ApiError as exc:
            scan_error = exc.message

        normal_ready = bool(normal_count or normal_video_count)
        add(
            "normal_samples",
            normal_ready and not scan_error,
            "正常样本",
            scan_error or (f"已找到 {normal_count} 个有效样本；有效视频 {normal_video_count} 个。" if normal_ready else "没有找到可读取的正常图片或视频样本。"),
            "选择并确认正常样本来源；测试数据、输出目录和备份目录不能用于构建。",
        )
        needs_defect = mode == "normal_memory_defect_guided_lite"
        defect_ready = bool(defect_count or defect_video_count)
        add(
            "defect_samples",
            defect_ready,
            "缺陷样本",
            f"已找到 {defect_count} 个有效样本；有效视频 {defect_video_count} 个。" if defect_ready else ("当前模式不要求缺陷样本。" if not needs_defect else "当前方法模式需要缺陷样本，但尚未找到有效来源。"),
            "选择缺陷图片或缺陷视频；缺陷样本只用于轻量校准，不会写入正常记忆库。",
            applicable=needs_defect or defect_ready,
        )

        normal_target = max(1, int(payload.get("normal_target_count") or 100))
        defect_target = max(0, int(payload.get("defect_target_count") or 30))
        normal_total = normal_count + normal_video_count
        defect_total = defect_count + defect_video_count
        budget_messages: list[str] = []
        if normal_total < normal_target:
            budget_messages.append(f"正常样本当前 {normal_total}，建议目标 {normal_target}")
        if needs_defect and defect_total < defect_target:
            budget_messages.append(f"缺陷样本当前 {defect_total}，建议目标 {defect_target}")
        add(
            "sample_budget",
            not budget_messages,
            "建议样本预算",
            "已达到当前建议样本预算。" if not budget_messages else "；".join(budget_messages) + "。可以构建候选版本，但正式使用前应确认代表性。",
            "补充具有代表性的目标域样本，或在明确了解风险后继续构建候选版本。",
            warning=bool(budget_messages),
        )

        mode_ok = mode in CREATABLE_DOMAIN_METHOD_MODES
        add(
            "mode",
            mode_ok,
            "构建模式",
            CREATABLE_DOMAIN_METHOD_LABELS.get(mode, "尚未选择当前支持的构建模式。"),
            "选择页面提供的“基础正常记忆”或“单正常记忆 · 缺陷引导轻量校准”。",
        )

        model_ok = False
        model_summary = "系统基础特征资源尚未就绪。"
        model_repair = "请恢复正式安装包中的模型配置、模型清单和受控权重资源。"
        model_details: dict[str, Any] = {}
        resolved_device = ""
        try:
            official_model = self._official_base_model(
                device=str(payload.get("device") or "auto"),
                verify_sha256=True,
            )
            model_id = str(official_model.get("model_id") or "")
            backbone = str(official_model.get("backbone") or "")
            resource = dict(official_model.get("resource_status") or {})
            compatibility = dict(official_model.get("device_compatibility") or {})
            resolved_device = str(compatibility.get("resolved") or "cpu")
            device_warnings = list(compatibility.get("warnings") or [])
            model_ok = bool(official_model.get("ready"))
            model_details = {
                "模型版本": official_model.get("model_version") or "未登记",
                "内部模型标识": model_id,
                "特征提取器": backbone or "未解析",
                "运行设备": resolved_device.upper(),
                "权重来源": resource.get("source") or "本机受控资源",
                "完整性状态": resource.get("reason_code") or "unknown",
                "配置摘要": str(official_model.get("profile_sha256") or "")[:12] or "未生成",
                "权重摘要": str((official_model.get("weights") or {}).get("sha256") or "")[:12] or "未生成",
                "模型清单摘要": str((official_model.get("manifest") or {}).get("sha256") or "")[:12] or "未生成",
            }
            if device_warnings:
                model_details["设备说明"] = "；".join(str(item) for item in device_warnings)
            if model_ok:
                model_summary = (
                    "系统用于构建目标域特征记忆的基础特征资源已准备完成；"
                    "该资源由正式模型注册服务自动解析，不依赖当前 Config、Profile 或 Memory Bank。"
                )
                model_repair = "无需处理。"
            else:
                model_summary = "正式模型资源缺失或完整性校验失败。"
                model_repair = "请恢复正式安装包中与当前系统版本配套的模型配置、模型清单和受控权重。"
                model_details["失败原因"] = str(resource.get("reason") or "资源状态不可用")
        except ApiError as exc:
            model_summary = exc.message or "正式模型资源缺失或完整性校验失败。"
            model_repair = str(exc.details.get("resolution") or model_repair)
            model_details = {"检查结果": str(exc.details.get("reason") or exc.message)}
        except (ValueError, RuntimeError, OSError) as exc:
            model_summary = f"基础特征资源检查失败：{exc}"
            model_repair = "请检查正式模型资源完整性并恢复与系统版本配套的资源。"
            model_details = {"检查异常": f"{type(exc).__name__}: {exc}"}
        add("model", model_ok, "基础特征资源", model_summary, model_repair, details=model_details)

        requested_fp16 = precision_uses_fp16(requested_precision)
        if model_ok and requested_fp16 and resolved_device == "cuda":
            add(
                "precision",
                True,
                "构建精度",
                "当前设备支持半精度构建，将使用半精度（FP16）提取特征。",
                details={"用户选择": "半精度（FP16）", "实际执行": "半精度（FP16）"},
            )
        elif model_ok and requested_fp16:
            add(
                "precision",
                True,
                "构建精度",
                "当前设备不支持半精度构建，将使用单精度模式。",
                "如需半精度构建，请使用受支持的 CUDA 设备；也可继续使用本次单精度回退。",
                warning=True,
                details={"用户选择": "半精度（FP16）", "实际执行": "单精度（FP32）", "运行设备": resolved_device.upper() or "CPU"},
            )
        elif model_ok:
            add(
                "precision",
                True,
                "构建精度",
                "将使用单精度（FP32）构建。",
                details={"用户选择": "单精度（FP32）", "实际执行": "单精度（FP32）"},
            )
        else:
            add(
                "precision",
                False,
                "构建精度",
                "基础特征资源尚未就绪，暂时无法确认实际构建精度。",
                applicable=False,
            )

        config_errors: list[str] = []
        try:
            ratio = float(payload.get("coreset_ratio", 0.1))
            knn = int(payload.get("knn_k", 1))
            image_size = int(payload.get("image_size", 256))
            layers = list(payload.get("layers") or [])
            max_memory = int(payload.get("max_memory_patches", 0))
            if not 0.001 <= ratio <= 1.0:
                config_errors.append("coreset 比例必须为 0.001–1.0")
            if not 1 <= knn <= 64:
                config_errors.append("最近邻数量必须为 1–64")
            if image_size < 64 or not layers:
                config_errors.append("输入尺寸或特征层无效")
            if max_memory <= 0:
                config_errors.append("最大记忆条目必须大于 0")
            if mode == "normal_memory_defect_guided_lite" and str(payload.get("threshold_method") or "") not in {"normal_percentile", "mean_std", "best_f1", "recall_priority"}:
                config_errors.append("阈值校准方法无效")
            if mode == "normal_memory_defect_guided_lite" and not 90.0 <= float(payload.get("lightweight_gate_percentile", 95.0)) <= 99.9:
                config_errors.append("轻量门控分位数必须为 90.0–99.9")
        except (TypeError, ValueError):
            config_errors.append("存在无法解析的构建参数")
        add(
            "parameters",
            not config_errors,
            "构建配置",
            "配置完整：输入尺寸、特征层、coreset、最近邻和当前方法参数均有效。" if not config_errors else "；".join(config_errors),
            "按提示修正当前方法模式显示的字段；无需跳转运行配置页面。",
        )

        with self._lock:
            active = self._tasks.get(self._active_job_id or "")
            conflict = bool(active and active.state in {"queued", "running", "cancelling"})
        output_ok = False
        try:
            probe = self.root / f".write_probe_{uuid.uuid4().hex}"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            output_ok = True
        except OSError:
            output_ok = False
        runtime_ok = output_ok and not conflict
        add(
            "runtime",
            runtime_ok,
            "输出与运行状态",
            "输出目录可写；当前没有冲突构建任务；构建使用独立写入上下文。" if runtime_ok else ("已有目标域构建任务正在运行。" if conflict else "目标域资源输出目录不可写。"),
            "等待当前任务结束或检查目标域资源目录权限和磁盘空间。",
        )
        blocking_items = [item for item in items if item["status"] == "failed"]
        warning_items = [item for item in items if item["status"] == "warning"]
        not_applicable_items = [item for item in items if item["status"] == "not_applicable"]
        can_create = not blocking_items
        return {
            # ``ready`` is retained for old callers.  New UI and API consumers
            # must use the single authoritative build gate below.
            "ready": can_create,
            "can_create": can_create,
            "blocking_count": len(blocking_items),
            "warning_count": len(warning_items),
            "not_applicable_count": len(not_applicable_items),
            "blocking_items": blocking_items,
            "items": items,
            "summary": {
                "friendly_name": name,
                "dataset": dataset,
                "category": category,
                "method_mode": mode,
                "normal_sample_count": normal_count,
                "defect_sample_count": defect_count,
                "normal_video_count": normal_video_count,
                "defect_video_count": defect_video_count,
                "representative_frame_count": normal_count + defect_count,
                "coreset_ratio": float(payload.get("coreset_ratio") or 0.1),
                "knn_k": int(payload.get("knn_k") or 1),
                "expected_version": "新候选版本",
            },
        }

    def reset_after_resource_clear(self) -> None:
        """Forget completed/stale build handles after a full catalog reset."""
        with self._lock:
            active = self._tasks.get(self._active_job_id or "")
            if active and active.state in {"queued", "running", "cancelling"}:
                raise ApiError("DOMAIN_BUILD_ACTIVE", "目标域构建仍在运行，不能重置资源状态。", 409)
            self._tasks.clear()
            self._tasks_by_idempotency.clear()
            self._active_job_id = None

    def create_parameter_candidate(self, base_profile_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Create an immutable parameter-only successor of a frozen profile.

        The normal memory and optional defect resources are copied unchanged;
        every editable runtime parameter is persisted in a new exclusive
        configuration snapshot.  The returned profile is a candidate and
        therefore cannot be applied until it is validated and frozen.
        """
        if self.test_lock is not None:
            self.test_lock.require_write_allowed("calibration", context=payload)
        allowed = {"threshold", "pixel_threshold", "score_aggregation", "save_visuals", "save_original"}
        unknown = sorted(set(payload) - allowed - {"friendly_name", "note"})
        if unknown:
            raise ApiError(
                "TARGET_PROFILE_REBUILD_REQUIRED",
                "所选字段会影响记忆库内容，必须在目标域适配中重新构建新版本。",
                409,
            )
        friendly_name = str(payload.get("friendly_name") or "").strip()
        if not friendly_name or len(friendly_name) > 80 or any(token in friendly_name for token in ("/", "\\", "..")):
            raise ApiError("INVALID_FRIENDLY_NAME", "新版本名称必须为 1 到 80 个字符，且不能包含路径符号。", 422)
        threshold_value = float(payload.get("threshold"))
        pixel_value = payload.get("pixel_threshold")
        if threshold_value < 0 or (pixel_value is not None and float(pixel_value) < 0):
            raise ApiError("INVALID_THRESHOLD", "图像级和像素级阈值不能小于 0。", 422)

        final: Path | None = None
        partial: Path | None = None
        created_config: tuple[str, int] | None = None
        with self._lock:
            try:
                base_path = self._profile_directory(base_profile_id)
                base_metadata = self._read_target_metadata(base_path)
                base_config = yaml.safe_load((base_path / "effective_config.yaml").read_text(encoding="utf-8")) or {}
                base_threshold = json.loads((base_path / "threshold.json").read_text(encoding="utf-8"))
                config_profile_id = str(base_metadata.get("config_profile_id") or "")
                config_profile_version = int(base_metadata.get("config_profile_version") or 0)
                if not config_profile_id or not config_profile_version:
                    raise ApiError("TARGET_PROFILE_CONFIG_MISSING", "基础目标域版本缺少专属配置快照，不能派生新版本。", 409)

                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                candidate_id = (
                    f"{_safe_token(str(base_metadata.get('dataset') or 'dataset'), 'dataset')}_"
                    f"{_safe_token(str(base_metadata.get('category') or 'category'), 'category')}_"
                    f"params_{stamp}_{uuid.uuid4().hex[:8]}"
                )
                new_snapshot_id = f"snapshot_{hashlib.sha256(candidate_id.encode('utf-8')).hexdigest()[:24]}"

                updated_config = dict(base_config)
                updated_config.update(
                    config_profile_id=new_snapshot_id,
                    config_profile_version=1,
                    target_domain_version=candidate_id,
                    build_job_id=f"parameter:{candidate_id}",
                    score_aggregation=str(payload.get("score_aggregation") or base_config.get("score_aggregation") or "maximum_patch"),
                    save_visuals=bool(payload.get("save_visuals", base_config.get("save_visuals", True))),
                    save_original=bool(payload.get("save_original", base_config.get("save_original", True))),
                )
                created = self.config_profiles.create_from_template(
                    {
                        "config_profile_id": new_snapshot_id,
                        "friendly_name": f"{friendly_name} · 专属配置快照",
                        "note": str(payload.get("note") or "")[:500],
                        "source_kind": "target_domain_snapshot",
                        "config": updated_config,
                        "target_domain_version": candidate_id,
                        "build_job_id": f"parameter:{candidate_id}",
                        "ordinary_visible": True,
                    },
                    template_profile_id=config_profile_id,
                    template_version=config_profile_version,
                )
                new_config_id = str(created["config_profile_id"])
                new_config_version = int(created["version"])
                created_config = (new_config_id, new_config_version)
                self.config_profiles.validate(new_config_id, new_config_version)
                frozen_config, frozen_hash = self.config_profiles.frozen_snapshot(
                    new_config_id,
                    new_config_version,
                    updated_config,
                    threshold_value,
                )

                partial = self.root / f".{candidate_id}.partial"
                final = self.root / candidate_id
                if partial.exists() or final.exists():
                    raise ApiError("DOMAIN_PROFILE_CONFLICT", "新目标域版本目录发生冲突，请重试。", 409)
                shutil.copytree(base_path, partial)
                for obsolete in ("frozen.lock", "candidate.lock", "candidate_validation.json"):
                    (partial / obsolete).unlink(missing_ok=True)

                threshold_payload = dict(base_threshold)
                threshold_payload["image_threshold"] = threshold_value
                threshold_payload["threshold"] = threshold_value
                if pixel_value is not None:
                    threshold_payload["pixel_threshold"] = float(pixel_value)
                (partial / "threshold.json").write_text(
                    json.dumps(threshold_payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                effective = dict(frozen_config)
                effective.update(
                    config_profile_id=new_config_id,
                    config_profile_version=new_config_version,
                    target_domain_version=candidate_id,
                    build_job_id=f"parameter:{candidate_id}",
                    config_hash=frozen_hash,
                    threshold=threshold_value,
                    inference_mode="inference_only",
                    output_root="由推理任务动态指定",
                    memory_bank_write_policy="frozen_read_only",
                    test_data_update_allowed=False,
                )
                (partial / "effective_config.yaml").write_text(
                    yaml.safe_dump(effective, allow_unicode=True, sort_keys=False), encoding="utf-8"
                )

                metadata = dict(base_metadata)
                hashes = dict(metadata.get("file_sha256") or {})
                hashes.pop("candidate_validation.json", None)
                hashes["threshold.json"] = _sha256(partial / "threshold.json")
                hashes["effective_config.yaml"] = _sha256(partial / "effective_config.yaml")
                parent_version = int(metadata.get("profile_version") or metadata.get("config_profile_version") or 1)
                metadata.update(
                    memory_bank_id=candidate_id,
                    memory_bank_name=friendly_name,
                    created_at=_utc_now(),
                    profile_version=parent_version + 1,
                    parent_profile_id=base_profile_id,
                    config_profile_id=new_config_id,
                    config_profile_version=new_config_version,
                    config_snapshot_id=new_config_id,
                    config_hash=frozen_hash,
                    config_snapshot_sha256=_sha256(partial / "effective_config.yaml"),
                    build_job_id=f"parameter:{candidate_id}",
                    build_idempotency_key=f"parameter:{candidate_id}",
                    threshold=threshold_value,
                    pixel_threshold=float(pixel_value) if pixel_value is not None else metadata.get("pixel_threshold"),
                    frozen=False,
                    profile_status="candidate",
                    profile_kind="target_parameter_candidate",
                    candidate_validation={},
                    file_sha256=hashes,
                    parameter_change_kind="inference_only",
                    parameter_change_note=str(payload.get("note") or "")[:500],
                    test_data_update_allowed=False,
                )
                metadata["resource_size_bytes"] = sum(
                    path.stat().st_size for path in partial.iterdir() if path.is_file()
                )
                metadata_path = partial / "metadata.json"
                metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
                (partial / "candidate.lock").write_text(
                    json.dumps(
                        {
                            "memory_bank_id": candidate_id,
                            "created_at": _utc_now(),
                            "metadata_sha256": _sha256(metadata_path),
                            "write_policy": "candidate_read_only",
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                os.replace(partial, final)
                partial = None
                if self.runtime_resources is None:
                    raise RuntimeError("统一资源目录服务尚未绑定，不能注册新目标域版本。")
                catalog = self.runtime_resources.scan(force=True)
                registered = any(
                    str(item.get("profile_id") or "") == candidate_id
                    for item in self.runtime_resources.list_resources(resource_type="candidate_profile", include_archived=False)
                )
                if not registered:
                    raise RuntimeError("参数候选文件已生成，但统一资源目录未确认注册。")
                result = self._public_metadata(metadata, is_default=False)
                result.update(
                    registration_status="registered",
                    resource_catalog_revision=int(catalog.get("resource_catalog_revision") or catalog.get("revision") or 0),
                    parent_profile_id=base_profile_id,
                )
                return result
            except Exception:
                if final is not None and final.exists():
                    isolated = self.failed_build_root / f"{final.name}_parameter_failure"
                    if isolated.exists():
                        isolated = self.failed_build_root / f"{final.name}_parameter_failure_{uuid.uuid4().hex[:8]}"
                    os.replace(final, isolated)
                if created_config is not None:
                    try:
                        self.config_profiles.archive_draft(*created_config)
                    except Exception as cleanup_error:
                        self.logger.error("parameter_candidate_config_archive_failed error=%s", cleanup_error)
                if self.runtime_resources is not None:
                    self.runtime_resources.scan(force=True)
                raise
            finally:
                if partial is not None and partial.exists():
                    shutil.rmtree(partial, ignore_errors=True)

    def validate_candidate(self, profile_id: str) -> dict[str, Any]:
        path = self._profile_directory(profile_id)
        candidate_lock = path / "candidate.lock"
        if not candidate_lock.is_file() or (path / "frozen.lock").exists():
            raise ApiError("DOMAIN_CANDIDATE_REQUIRED", "该资源不是待验证候选版本。", 409)
        metadata_path = path / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        required = ["memory_bank.pt", "threshold.json", "effective_config.yaml", "normal_samples_manifest.csv"]
        missing = [name for name in required if not (path / name).is_file()]
        if missing:
            raise ApiError("DOMAIN_CANDIDATE_INCOMPLETE", f"候选版本缺少资源：{', '.join(missing)}", 409)
        for name, digest in dict(metadata.get("file_sha256") or {}).items():
            if not (path / name).is_file() or _sha256(path / name) != digest:
                raise ApiError("DOMAIN_CANDIDATE_INTEGRITY_FAILED", f"候选资源校验失败：{name}", 409)
        with (path / "normal_samples_manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            roles = {str(row.get("sample_role") or "") for row in csv.DictReader(handle)}
        if roles - {"normal_memory_source"}:
            raise ApiError("NORMAL_MEMORY_CONTAMINATION", "正常记忆库清单包含非正常来源。", 409)
        mode = str(metadata.get("method_mode_resolved") or "")
        if mode == "normal_defect_dual_memory" and not (path / "defect_prototypes.pt").is_file():
            raise ApiError("DOMAIN_CANDIDATE_INCOMPLETE", "双记忆候选缺少缺陷原型。", 409)
        if mode == "normal_memory_defect_guided_lite" and not (path / "lightweight_calibrator.json").is_file():
            raise ApiError("DOMAIN_CANDIDATE_INCOMPLETE", "轻量校准候选缺少校准器。", 409)
        validation = {"validated": True, "validated_at": _utc_now(), "integrity": "passed", "normal_memory_isolated": True}
        (path / "candidate_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
        metadata["candidate_validation"] = validation
        metadata["resource_integrity_status"] = "passed"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        lock = json.loads(candidate_lock.read_text(encoding="utf-8"))
        lock["metadata_sha256"] = _sha256(metadata_path)
        candidate_lock.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
        return {**self._public_metadata(metadata), **validation}

    def candidate_details(self, profile_id: str) -> dict[str, Any]:
        """Return one server-owned candidate without exposing local paths."""
        path = self._profile_directory(profile_id)
        if not (path / "candidate.lock").is_file() or (path / "frozen.lock").exists():
            raise ApiError("DOMAIN_CANDIDATE_REQUIRED", "该资源不是待处理的目标域候选版本。", 409)
        metadata_path = path / "metadata.json"
        if not metadata_path.is_file():
            raise ApiError("DOMAIN_CANDIDATE_INCOMPLETE", "候选版本缺少元数据。", 409)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return self._public_metadata(metadata, is_default=False)

    def freeze_candidate(self, profile_id: str, *, confirm: bool) -> dict[str, Any]:
        if self.test_lock is not None:
            self.test_lock.require_write_allowed("profile_freeze")
        if not confirm:
            raise ApiError("CONFIRMATION_REQUIRED", "冻结候选版本需要明确确认。", 409)
        self.validate_candidate(profile_id)
        path = self._profile_directory(profile_id)
        metadata_path = path / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        config = yaml.safe_load((path / "effective_config.yaml").read_text(encoding="utf-8")) or {}
        threshold = json.loads((path / "threshold.json").read_text(encoding="utf-8"))
        config_profile_id = str(metadata["config_profile_id"])
        config_profile_version = int(metadata["config_profile_version"])
        frozen_profile, frozen_hash = self.config_profiles.frozen_snapshot(
            config_profile_id, config_profile_version, config, float(threshold["image_threshold"])
        )
        metadata.update(
            frozen=True,
            profile_status="frozen",
            frozen_at=_utc_now(),
            config_hash=frozen_hash,
            config_snapshot_id=config_profile_id,
            config_snapshot_sha256=_sha256(path / "effective_config.yaml"),
            resource_integrity_status="passed",
        )
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        lock = {"memory_bank_id": profile_id, "frozen_at": _utc_now(), "metadata_sha256": _sha256(metadata_path), "write_policy": "inference_only"}
        (path / "frozen.lock").write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
        (path / "candidate.lock").unlink(missing_ok=True)
        self.config_profiles.bind_and_freeze(
            config_profile_id, config_profile_version, profile_id, float(threshold["image_threshold"]), frozen_profile, frozen_hash
        )
        self.registry.invalidate(str(metadata.get("dataset")), str(metadata.get("category")))
        return self._public_metadata(metadata)

    def _profile_directory(self, profile_id: str) -> Path:
        if not PROFILE_ID_PATTERN.fullmatch(str(profile_id)):
            raise ApiError("INVALID_MEMORY_BANK_ID", "记忆库标识无效。", 422)
        path = (self.root / profile_id).resolve()
        if not _is_within(path, self.root) or not path.is_dir():
            raise ApiError("MEMORY_BANK_NOT_FOUND", "记忆库不存在或已归档。", 404)
        return path

    def _read_target_metadata(self, path: Path, *, verify_artifacts: bool = True) -> dict[str, Any]:
        required = ["memory_bank.pt", "threshold.json", "effective_config.yaml", "metadata.json", "frozen.lock"]
        if any(not (path / name).is_file() for name in required):
            raise ApiError("MEMORY_BANK_NOT_FROZEN", "记忆库文件不完整或尚未冻结。", 409)
        try:
            metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            lock = json.loads((path / "frozen.lock").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ApiError("MEMORY_BANK_METADATA_INVALID", "记忆库元数据无法读取。", 500) from exc
        if not metadata.get("frozen") or lock.get("write_policy") != "inference_only":
            raise ApiError("MEMORY_BANK_NOT_FROZEN", "该记忆库未处于推理只读冻结状态。", 409)
        if metadata.get("method_mode_resolved") == "normal_defect_dual_memory" and not (path / "defect_prototypes.pt").is_file():
            raise ApiError("MEMORY_BANK_NOT_FROZEN", "双记忆方案缺少独立缺陷原型文件。", 409)
        if metadata.get("method_mode_resolved") == "normal_memory_defect_guided_lite" and not (path / "lightweight_calibrator.json").is_file():
            raise ApiError("MEMORY_BANK_NOT_FROZEN", "轻量校准方案缺少冻结校准器文件。", 409)
        if _sha256(path / "metadata.json") != lock.get("metadata_sha256"):
            raise ApiError("MEMORY_BANK_INTEGRITY_FAILED", "记忆库冻结元数据已发生变化。", 409)
        if verify_artifacts:
            hashes = metadata.get("file_sha256") or {}
            checked = ["memory_bank.pt", "threshold.json", "effective_config.yaml"]
            if metadata.get("method_mode_resolved") == "normal_defect_dual_memory":
                checked.append("defect_prototypes.pt")
            if metadata.get("method_mode_resolved") == "normal_memory_defect_guided_lite":
                checked.append("lightweight_calibrator.json")
            for name in checked:
                if hashes.get(name) != _sha256(path / name):
                    raise ApiError("MEMORY_BANK_INTEGRITY_FAILED", f"记忆库冻结文件校验失败：{name}", 409)
        if not metadata.get("config_profile_id"):
            effective = yaml.safe_load((path / "effective_config.yaml").read_text(encoding="utf-8")) or {}
            legacy = self.config_profiles.ensure_legacy_profile(metadata, effective)
            metadata = {
                **metadata,
                "config_profile_id": legacy.get("config_profile_id"),
                "config_profile_version": legacy.get("version"),
                "config_hash": legacy.get("config_hash"),
            }
        return metadata

    @staticmethod
    def _public_metadata(metadata: dict[str, Any], *, is_default: bool = False) -> dict[str, Any]:
        return {
            "memory_bank_id": metadata.get("memory_bank_id"),
            "friendly_name": metadata.get("memory_bank_name"),
            "dataset": metadata.get("dataset"),
            "category": metadata.get("category"),
            "created_at": metadata.get("created_at"),
            "profile_version": int(metadata.get("profile_version") or metadata.get("config_profile_version") or 1),
            "parent_profile_id": metadata.get("parent_profile_id") or metadata.get("feedback_parent_profile_id"),
            "normal_sample_count": metadata.get("normal_sample_count"),
            "defect_calibration_sample_count": metadata.get("defect_calibration_sample_count"),
            "normal_source_types": metadata.get("normal_source_types") or [],
            "defect_source_types": metadata.get("defect_source_types") or [],
            "normal_image_count": metadata.get("normal_image_count"),
            "normal_video_count": metadata.get("normal_video_count"),
            "normal_extracted_frame_count": metadata.get("normal_extracted_frame_count"),
            "defect_image_count": metadata.get("defect_image_count"),
            "defect_video_count": metadata.get("defect_video_count"),
            "defect_extracted_frame_count": metadata.get("defect_extracted_frame_count"),
            "deduplicated_frame_count": metadata.get("deduplicated_frame_count"),
            "rejected_frame_count": metadata.get("rejected_frame_count"),
            "video_sampling_config": metadata.get("video_sampling_config") or {},
            "sample_budget_mode": metadata.get("sample_budget_mode") or "final_frame_count",
            "defect_video_intervals": metadata.get("defect_video_intervals") or {},
            "source_video_hashes": metadata.get("source_video_hashes") or {},
            "frame_selection_seed": metadata.get("frame_selection_seed"),
            "video_decoder_version": metadata.get("video_decoder_version"),
            "sample_manifest": metadata.get("sample_manifest") or {},
            "normal_source_summary": metadata.get("normal_source_summary") or {},
            "defect_source_summary": metadata.get("defect_source_summary") or {},
            "threshold": metadata.get("threshold"),
            "threshold_method": metadata.get("threshold_method"),
            "pixel_threshold": metadata.get("pixel_threshold"),
            "method_name": metadata.get("method_name") or "正常记忆异常检测",
            "method_mode_requested": metadata.get("method_mode_requested"),
            "method_mode_resolved": metadata.get("method_mode_resolved") or "normal_defect_calibration",
            "algorithm_mode": metadata.get("algorithm_mode") or metadata.get("method_mode_resolved") or metadata.get("method_mode_requested"),
            "normal_memory_enabled": bool(metadata.get("normal_memory_enabled", True)),
            "lightweight_calibration_enabled": bool(
                metadata.get("lightweight_calibration_enabled", metadata.get("lightweight_calibration", False))
            ),
            "defect_guidance_enabled": bool(
                metadata.get("defect_guidance_enabled", metadata.get("lightweight_calibration", False))
            ),
            "base_method": metadata.get("base_method"),
            "references": metadata.get("references") or [],
            "defect_prototype_count": int(metadata.get("defect_prototype_count") or 0),
            "dual_memory_alpha": float(metadata.get("dual_memory_alpha") or 0.0),
            "lightweight_calibration": bool(metadata.get("lightweight_calibration", False)),
            "lightweight_calibrator_parameters": int(metadata.get("lightweight_calibrator_parameters") or 0),
            "defect_features_saved": bool(metadata.get("defect_features_saved", False)),
            "additional_ann_count": int(metadata.get("additional_ann_count") or 0),
            "feature_layers": metadata.get("feature_layers"),
            "feature_dimension": metadata.get("feature_dimension"),
            "image_size": metadata.get("image_size"),
            "coreset": metadata.get("coreset"),
            "knn_k": metadata.get("knn_k"),
            "model_id": metadata.get("model_id"),
            "model_type": "patchcore",
            "backbone": metadata.get("backbone"),
            "device": metadata.get("device"),
            "precision": metadata.get("precision") or metadata.get("requested_precision") or ("fp16" if metadata.get("amp_enabled") else "fp32"),
            "requested_precision": metadata.get("requested_precision") or metadata.get("precision") or ("fp16" if metadata.get("amp_enabled") else "fp32"),
            "actual_precision": metadata.get("actual_precision") or ("fp16" if metadata.get("amp_enabled") else "fp32"),
            "amp_enabled": bool(metadata.get("amp_enabled", False)),
            "feature_dtype": metadata.get("feature_dtype") or "unknown",
            "backbone_output_dtype": metadata.get("backbone_output_dtype") or "unknown",
            "memory_bank_dtype": metadata.get("memory_bank_dtype") or "unknown",
            "index_dtype": metadata.get("index_dtype") or "unknown",
            "precision_fallback_reason": metadata.get("precision_fallback_reason") or "",
            "config_profile_id": metadata.get("config_profile_id"),
            "config_profile_version": metadata.get("config_profile_version"),
            "config_snapshot_id": metadata.get("config_snapshot_id") or metadata.get("config_profile_id"),
            "config_hash": metadata.get("config_hash"),
            "config_snapshot_sha256": metadata.get("config_snapshot_sha256") or "",
            "build_job_id": metadata.get("build_job_id") or "",
            "build_idempotency_key": metadata.get("build_idempotency_key") or "",
            "resource_integrity_status": metadata.get("resource_integrity_status") or ("passed" if metadata.get("frozen") else "pending"),
            "frozen": bool(metadata.get("frozen")),
            "profile_status": metadata.get("profile_status") or ("frozen" if metadata.get("frozen") else "candidate"),
            "file_sha256": metadata.get("file_sha256") or {},
            "candidate_validation": metadata.get("candidate_validation") or {},
            "resource_size_bytes": int(metadata.get("resource_size_bytes") or 0),
            "calibration_metrics": metadata.get("calibration_metrics") or {},
            "evaluation_mode": metadata.get("evaluation_mode") or "calibration_only",
            "calibration_result": metadata.get("calibration_result") or metadata.get("calibration_metrics") or {},
            "validation_result": metadata.get("validation_result") or {"status": "not_run"},
            "data_roles": metadata.get("data_roles") or {},
            "complexity_summary": metadata.get("complexity_summary") or {},
            "profile_kind": metadata.get("profile_kind", "target_domain"),
            "normal_features_only": bool(metadata.get("normal_features_only")),
            "test_data_update_allowed": bool(metadata.get("test_data_update_allowed", False)),
            "official_test_mode": bool(metadata.get("official_test_mode", False)),
            "is_default": is_default,
        }

    def _builtin_memory_path(self, dataset: str, category: str) -> Path:
        root = (self.settings.algorithm_outputs_root / "checkpoints").resolve()
        path = (root / dataset / category / "memory_bank.pt").resolve()
        if not _is_within(path, root):
            raise ApiError("MEMORY_BANK_PATH_UNSAFE", "记忆库资源路径超出受控目录。", 422)
        return path

    def _builtin_profile(self, model_id: str, dataset: str, category: str) -> dict[str, Any] | None:
        dataset, category = validate_dataset_category(dataset, category)
        memory_path = self._builtin_memory_path(dataset, category)
        if not memory_path.is_file():
            return None
        calibration_path = self.settings.active_calibration_path(str(dataset).lower(), category)
        threshold = None
        method = "模型内置阈值"
        if calibration_path.is_file():
            try:
                calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
                threshold = calibration.get("image_threshold")
                method = calibration.get("threshold_strategy", "已有校准")
            except Exception:
                pass
        profile_id = f"builtin_{hashlib.sha256(f'{model_id}|{dataset}|{category}|{memory_path}'.encode()).hexdigest()[:20]}"
        binding = self.config_profiles.builtin_binding(
            profile_id,
            threshold,
            model_id=model_id,
            dataset=str(dataset).lower(),
            category=category,
        )
        return {
            "memory_bank_id": profile_id,
            "friendly_name": "内置公开数据基线",
            "dataset": str(dataset).lower(),
            "category": category,
            "created_at": datetime.fromtimestamp(memory_path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
            "normal_sample_count": None,
            "defect_calibration_sample_count": None,
            "threshold": threshold,
            "threshold_method": method,
            "model_id": model_id,
            "model_type": "patchcore",
            "method_name": "正常记忆异常检测",
            "method_mode_requested": "normal_only",
            "method_mode_resolved": "normal_only",
            "base_method": "normal patch memory nearest-neighbour anomaly detection",
            "references": ["PatchCore (CVPR 2022)"],
            "defect_prototype_count": 0,
            "dual_memory_alpha": 0.0,
            "backbone": binding.get("backbone"),
            "config_profile_id": binding.get("config_profile_id"),
            "config_profile_version": binding.get("version"),
            "config_hash": binding.get("config_hash"),
            "frozen": True,
            "profile_kind": "builtin_public_baseline",
            "normal_features_only": True,
            "test_data_update_allowed": False,
            "is_default": False,
        }

    def list_profiles(self, dataset: str, category: str, model_id: str | None = None) -> list[dict[str, Any]]:
        dataset, category = validate_dataset_category(dataset, category)
        resolved_model_id, _ = self.settings.model_definition(model_id)
        data = self._registry_data()
        default_id = data.get("defaults", {}).get(self._default_key(resolved_model_id, dataset, category))
        friendly_names = data.get("friendly_names", {})
        unavailable = data.get("unavailable", {})
        profiles: list[dict[str, Any]] = []
        for path in sorted(self.root.iterdir()) if self.root.exists() else []:
            if not path.is_dir() or path.name.startswith((".", "_")):
                continue
            if path.name in unavailable:
                continue
            try:
                metadata = self._read_target_metadata(path, verify_artifacts=False)
            except ApiError:
                continue
            if str(metadata.get("dataset", "")).lower() != str(dataset).lower() or str(metadata.get("category")) != str(category):
                continue
            if str(metadata.get("model_id")) != resolved_model_id:
                continue
            public = self._public_metadata(metadata, is_default=path.name == default_id)
            if path.name in friendly_names:
                public["friendly_name"] = friendly_names[path.name]
            profiles.append(public)
        profiles.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        builtin = self._builtin_profile(resolved_model_id, dataset, category)
        if builtin:
            builtin["is_default"] = builtin["memory_bank_id"] == default_id
            profiles.append(builtin)
        if profiles and not any(item["is_default"] for item in profiles):
            # With no explicit selection, retain the repository's built-in
            # baseline.  Merely building a newer target-domain version must not
            # silently change the effective inference profile.
            fallback = next(
                (item for item in profiles if item.get("profile_kind") == "builtin_public_baseline"),
                profiles[0],
            )
            fallback["is_default"] = True
        return profiles

    def list_all_profiles(self) -> list[dict[str, Any]]:
        """List frozen target-domain resources without requiring unknown field values."""
        data = self._registry_data()
        default_ids = set(str(value) for value in (data.get("defaults") or {}).values())
        friendly_names = data.get("friendly_names", {})
        unavailable = data.get("unavailable", {})
        profiles: list[dict[str, Any]] = []
        for path in sorted(self.root.iterdir()) if self.root.exists() else []:
            if not path.is_dir() or path.name.startswith((".", "_")):
                continue
            if path.name in unavailable:
                continue
            try:
                metadata = self._read_target_metadata(path, verify_artifacts=False)
            except ApiError:
                continue
            public = self._public_metadata(metadata, is_default=path.name in default_ids)
            if path.name in friendly_names:
                public["friendly_name"] = friendly_names[path.name]
            profiles.append(public)
        # Register every built-in profile that can currently be selected, not
        # only the repository default.  Otherwise a valid built-in profile
        # chosen in runtime settings disappears from the unified catalog and
        # the UI incorrectly reports that the current scheme has no owner.
        builtin_specs: list[tuple[str, str, str]] = []
        for settings_row in (self.settings.defaults, self.settings.dynamic_settings()):
            try:
                model_id, _ = self.settings.model_definition(str(settings_row.get("model_id") or ""))
                spec = (
                    model_id,
                    str(settings_row.get("dataset") or "mvtec").lower(),
                    str(settings_row.get("category") or "bottle"),
                )
                if spec not in builtin_specs:
                    builtin_specs.append(spec)
            except (KeyError, ValueError):
                continue
        known_ids = {str(item.get("memory_bank_id") or "") for item in profiles}
        for model_id, dataset, category in builtin_specs:
            builtin = self._builtin_profile(model_id, dataset, category)
            if builtin and str(builtin.get("memory_bank_id") or "") not in known_ids:
                builtin["is_default"] = str(builtin.get("memory_bank_id") or "") in default_ids
                profiles.append(builtin)
                known_ids.add(str(builtin.get("memory_bank_id") or ""))
        profiles.sort(key=lambda item: (not bool(item.get("is_default")), str(item.get("created_at") or "")), reverse=False)
        return profiles

    def resolve_for_inference(self, profile_id: str | None, model_id: str, dataset: str, category: str) -> dict[str, Any]:
        dataset, category = validate_dataset_category(dataset, category)
        selected = None
        if profile_id:
            # Startup already resolved the active formal profile from the
            # persisted runtime-resource graph.  Do not enumerate every target
            # domain directory again here: that scan is a management/listing
            # operation and can dominate cold start as the repository grows.
            # Validate only the explicitly bound directory, then perform the
            # authoritative artifact check below before returning it.
            try:
                bound_path = self._profile_directory(str(profile_id))
                bound_metadata = self._read_target_metadata(bound_path, verify_artifacts=False)
                if (
                    str(bound_metadata.get("memory_bank_id") or bound_path.name) != str(profile_id)
                    or str(bound_metadata.get("dataset") or "").lower() != str(dataset).lower()
                    or str(bound_metadata.get("category") or "") != str(category)
                    or str(bound_metadata.get("model_id") or "") != str(model_id)
                ):
                    raise ApiError("MEMORY_BANK_NOT_FOUND", "所选记忆库不存在、未冻结或与当前模型配置不兼容。", 409)
                registry_data = self._registry_data()
                default_id = registry_data.get("defaults", {}).get(self._default_key(model_id, dataset, category))
                selected = self._public_metadata(bound_metadata, is_default=str(default_id or "") == str(profile_id))
                friendly_name = (registry_data.get("friendly_names") or {}).get(str(profile_id))
                if friendly_name:
                    selected["friendly_name"] = friendly_name
            except ApiError:
                raise
            except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
                raise ApiError("MEMORY_BANK_NOT_FOUND", "所选记忆库不存在、未冻结或与当前模型配置不兼容。", 409) from exc
        else:
            # Omitted profile IDs are used by management and compatibility
            # callers; retain the complete selectable-profile behavior there.
            profiles = self.list_profiles(dataset, category, model_id)
            selected = next((item for item in profiles if item.get("is_default")), profiles[0] if profiles else None)
        if selected is None:
            raise ApiError("MEMORY_BANK_MISSING", "没有可用记忆库，请先进入“目标域适配”页面构建并冻结。", 409)
        if selected["profile_kind"] == "builtin_public_baseline":
            return {
                **selected,
                "memory_path": self._builtin_memory_path(dataset, category),
                "calibration_path": self.settings.active_calibration_path(str(dataset).lower(), category),
                "config": self.settings.algorithm_profile_unbound(model_id),
            }
        path = self._profile_directory(selected["memory_bank_id"])
        metadata = self._read_target_metadata(path)
        config = yaml.safe_load((path / "effective_config.yaml").read_text(encoding="utf-8")) or {}
        return {
            **self._public_metadata(metadata, is_default=selected.get("is_default", False)),
            "memory_path": path / "memory_bank.pt",
            "calibration_path": path / "threshold.json",
            "defect_prototype_path": path / "defect_prototypes.pt" if (path / "defect_prototypes.pt").is_file() else None,
            "lightweight_calibrator_path": path / "lightweight_calibrator.json" if (path / "lightweight_calibrator.json").is_file() else None,
            "config": config,
        }

    def get_profile(self, profile_id: str, dataset: str, category: str, model_id: str | None = None) -> dict[str, Any]:
        resolved_model_id, _ = self.settings.model_definition(model_id)
        selected = self.resolve_for_inference(profile_id, resolved_model_id, dataset, category)
        return {key: value for key, value in selected.items() if key not in {"memory_path", "calibration_path", "config"}}

    def activate(self, profile_id: str, dataset: str, category: str, model_id: str | None = None) -> dict[str, Any]:
        resolved_model_id, _ = self.settings.model_definition(model_id)
        selected = self.resolve_for_inference(profile_id, resolved_model_id, dataset, category)
        self._set_default(profile_id, resolved_model_id, dataset, category)
        self.registry.invalidate(str(dataset).lower(), category)
        return {key: value for key, value in selected.items() if key not in {"memory_path", "calibration_path", "config"}}

    def deactivate(self, profile_id: str) -> dict[str, Any]:
        """Remove only the runtime-default pointer; immutable artifacts stay intact."""

        removed: list[str] = []
        with self._lock:
            data = self._registry_data()
            defaults = data.setdefault("defaults", {})
            for key, value in list(defaults.items()):
                if str(value or "") == str(profile_id or ""):
                    removed.append(str(key))
                    defaults.pop(key, None)
            self._write_registry(data)
        return {"profile_id": str(profile_id or ""), "deactivated": True, "removed_default_bindings": removed}

    def rename(self, profile_id: str, friendly_name: str) -> dict[str, Any]:
        if self.test_lock is not None:
            self.test_lock.require_write_allowed("resource_lifecycle")
        path = self._profile_directory(profile_id)
        metadata = self._read_target_metadata(path)
        name = str(friendly_name).strip()
        if not name or len(name) > 80:
            raise ApiError("INVALID_FRIENDLY_NAME", "友好名称必须为 1 到 80 个字符。", 422)
        with self._lock:
            data = self._registry_data()
            data.setdefault("friendly_names", {})[profile_id] = name
            self._write_registry(data)
        result = self._public_metadata(metadata)
        result["friendly_name"] = name
        return result

    def archive(self, profile_id: str) -> dict[str, Any]:
        if self.test_lock is not None:
            self.test_lock.require_write_allowed("resource_lifecycle")
        path = self._profile_directory(profile_id)
        metadata = self._read_target_metadata(path)
        memory_path = path / "memory_bank.pt"
        if self._reservations.get(profile_id):
            raise ApiError("MEMORY_BANK_IN_USE", "该记忆库正在被检测任务使用，不能归档。", 409)
        # A registry cache may retain a loaded model after the last request,
        # but that is not an active task.  Explicit reservations are the lease;
        # invalidate the idle cache before moving the immutable profile.
        self.registry.invalidate(str(metadata.get("dataset", "")).lower(), str(metadata.get("category", "")))
        self.archive_root.mkdir(parents=True, exist_ok=True)
        destination = self.archive_root / profile_id
        if destination.exists():
            destination = self.archive_root / f"{profile_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.replace(path, destination)
        with self._lock:
            data = self._registry_data()
            friendly_name = (data.get("friendly_names") or {}).get(profile_id) or metadata.get("memory_bank_name") or "目标域记忆库"
            data.setdefault("archived", {})[profile_id] = {"archived_at": _utc_now(), "archive_name": destination.name, "friendly_name": friendly_name}
            data.setdefault("friendly_names", {}).pop(profile_id, None)
            defaults = data.setdefault("defaults", {})
            for key, value in list(defaults.items()):
                if value == profile_id:
                    defaults.pop(key, None)
            self._write_registry(data)
        return {"memory_bank_id": profile_id, "archived": True}

    def archived_profile_directory(self, profile_id: str) -> Path:
        """Resolve an archived profile only through the server registry."""
        if not PROFILE_ID_PATTERN.fullmatch(str(profile_id)):
            raise ApiError("INVALID_MEMORY_BANK_ID", "记忆库标识无效。", 422)
        data = self._registry_data()
        entry = (data.get("archived") or {}).get(str(profile_id)) or {}
        archive_name = str(entry.get("archive_name") or profile_id)
        candidate = (self.archive_root / archive_name).resolve()
        if not _is_within(candidate, self.archive_root.resolve()) or not candidate.is_dir() or candidate.is_symlink():
            raise ApiError("MEMORY_BANK_ARCHIVE_NOT_FOUND", "归档记忆库文件不存在或路径不受控。", 404)
        return candidate

    def list_archived_profiles(self) -> list[dict[str, Any]]:
        """Return archived profiles without making them selectable for inference."""
        data = self._registry_data()
        archived = data.get("archived") or {}
        rows: list[dict[str, Any]] = []
        for profile_id, entry in archived.items():
            profile_id = str(profile_id)
            try:
                path = self.archived_profile_directory(profile_id)
            except ApiError as exc:
                rows.append({"memory_bank_id": profile_id, "friendly_name": str(entry.get("friendly_name") or "目标域记忆库"), "archived": True, "archived_at": entry.get("archived_at"), "integrity_status": "incomplete", "archive_error": exc.message})
                continue
            try:
                metadata = self._read_target_metadata(path, verify_artifacts=False)
                item = self._public_metadata(metadata)
                item["friendly_name"] = str(entry.get("friendly_name") or item.get("friendly_name") or "目标域记忆库")
                item["archived"] = True
                item["archived_at"] = entry.get("archived_at")
                item["archive_name"] = path.name
                item["integrity_status"] = "archived"
            except ApiError as exc:
                item = {"memory_bank_id": profile_id, "friendly_name": str(entry.get("friendly_name") or "目标域记忆库"), "archived": True, "archived_at": entry.get("archived_at"), "integrity_status": "incomplete", "archive_name": path.name, "archive_error": exc.message}
            rows.append(item)
        rows.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return rows

    def restore(self, profile_id: str) -> dict[str, Any]:
        """Restore an archived profile without activating it."""
        with self._lock:
            source = self.archived_profile_directory(profile_id)
            target = (self.root / str(profile_id)).resolve()
            if target.exists():
                raise ApiError("MEMORY_BANK_RESTORE_CONFLICT", "同名记忆库已存在，无法恢复归档。", 409)
            metadata = self._read_target_metadata(source, verify_artifacts=True)
            if target.is_symlink() or not _is_within(target, self.root.resolve()):
                raise ApiError("MEMORY_BANK_PATH_UNSAFE", "恢复目标路径不受控。", 409)
            os.replace(source, target)
            data = self._registry_data()
            entry = (data.get("archived") or {}).pop(str(profile_id), {})
            data.setdefault("friendly_names", {})[str(profile_id)] = str(entry.get("friendly_name") or metadata.get("memory_bank_name") or "目标域记忆库")
            data.setdefault("unavailable", {}).pop(str(profile_id), None)
            self._write_registry(data)
            return {**self._public_metadata(metadata), "memory_bank_id": str(profile_id), "archived": False, "restored": True, "active": False}

    def forget_archived(self, profile_id: str, operation_id: str, *, reason: str = "") -> dict[str, Any]:
        """Remove an archived profile from the selectable registry and keep a tombstone."""
        with self._lock:
            data = self._registry_data()
            archived = data.setdefault("archived", {})
            entry = dict(archived.pop(str(profile_id), {}) or {})
            data.setdefault("friendly_names", {}).pop(str(profile_id), None)
            data.setdefault("deleted", {})[str(profile_id)] = {
                "deleted_at": _utc_now(),
                "operation_id": str(operation_id),
                "archive_name": entry.get("archive_name"),
                "friendly_name": entry.get("friendly_name") or "目标域记忆库",
                "reason": str(reason or "")[:240],
            }
            self._write_registry(data)
        return {"memory_bank_id": str(profile_id), "deleted": True, "operation_id": str(operation_id)}

    def reserve(self, profile_id: str | None, owner_id: str) -> None:
        if not profile_id or str(profile_id).startswith("builtin_"):
            return
        self._profile_directory(str(profile_id))
        with self._lock:
            self._reservations.setdefault(str(profile_id), set()).add(str(owner_id))

    def release(self, profile_id: str | None, owner_id: str) -> None:
        if not profile_id:
            return
        with self._lock:
            owners = self._reservations.get(str(profile_id))
            if owners is not None:
                owners.discard(str(owner_id))
                if not owners:
                    self._reservations.pop(str(profile_id), None)

    def shutdown(self) -> None:
        with self._lock:
            tasks = [task for task in self._tasks.values() if task.state in {"queued", "running", "cancelling"}]
        for task in tasks:
            task.cancel_event.set()
        for task in tasks:
            if task.future is not None:
                try:
                    task.future.result(timeout=10.0)
                except Exception:
                    pass
            elif task.thread and task.thread.is_alive():
                task.thread.join(timeout=10.0)
        self._build_environment_warmup_stop.set()
        with self._build_environment_warmup_lock:
            warmup_process = self._build_environment_warmup_process
            warmup_thread = self._build_environment_warmup_thread
        if warmup_process is not None:
            self._reclaim_warmup_process(warmup_process)
        if warmup_thread is not None and warmup_thread.is_alive():
            warmup_thread.join(timeout=2.0 * BUILD_ENVIRONMENT_TERMINATE_TIMEOUT_SECONDS + 1.0)
        self._build_executor.shutdown(wait=False, cancel_futures=True)
