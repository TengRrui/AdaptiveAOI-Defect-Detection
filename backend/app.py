from __future__ import annotations

import logging
import threading
import time
import uuid
from copy import deepcopy
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router
from backend.config import IntegrationSettings
from backend.errors import ApiError
from backend.logging_config import configure_logging


MODEL_WARMUP_WORKER_KIND = "model_warmup_initializer"
WORKER_HEARTBEAT_TTL_MS = 30_000


def runtime_state_code(
    state: str | None,
    *,
    runtime_status: str | None = None,
    phase: str | None = None,
) -> str:
    """Return the canonical Runtime lifecycle state.

    Runtime readiness is deliberately independent from the one-shot thread
    that performs model warm-up.  A completed warm-up may therefore yield
    READY while its initializer thread is STOPPED.
    """

    raw_state = str(state or "").strip().lower()
    raw_runtime = str(runtime_status or "").strip().lower()
    raw_phase = str(phase or "").strip().lower()
    if raw_state == "failed" or raw_runtime == "failed":
        return "FAILED"
    if raw_state == "degraded" or raw_runtime in {"degraded", "deactivating"}:
        return "DEGRADED"
    if raw_phase == "warming_up" or raw_state == "warming_up":
        return "WARMING_UP"
    if raw_state == "ready" and raw_runtime in {"", "ready"}:
        return "READY"
    if raw_runtime == "ready" and raw_state in {"", "ready"}:
        return "READY"
    if raw_state in {"loading", "starting"} or raw_runtime in {"loading", "starting"}:
        return "LOADING"
    if raw_state in {"pending", ""} or raw_runtime in {"pending", "starting"}:
        return "STARTING"
    if raw_state in {"not_ready", "inactive"} or raw_runtime in {"not_ready", "inactive"}:
        return "NOT_READY"
    return "STARTING"


def _parse_status_time(value: str | datetime | None) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif value:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def worker_state_snapshot(
    *,
    thread_alive: bool,
    lifecycle_state: str | None,
    required_for_runtime: bool,
    worker_kind: str = MODEL_WARMUP_WORKER_KIND,
    last_heartbeat_at: str | datetime | None = None,
    now: str | datetime | None = None,
    error: str | None = None,
) -> dict[str, object]:
    """Describe a worker without making it an implicit Runtime gate.

    The current model warm-up thread is one-shot.  Once it has completed, its
    state is STOPPED and Runtime can remain READY.  Heartbeat age is only
    meaningful while a worker is alive; a stopped auxiliary initializer does
    not become a failed Runtime merely because its last transition is old.
    """

    lifecycle = str(lifecycle_state or "").strip().lower()
    alive = bool(thread_alive)
    if lifecycle == "failed":
        worker_state = "FAILED"
    elif alive and lifecycle in {"pending", "starting", "loading", "warming_up"}:
        worker_state = "STARTING"
    elif alive:
        worker_state = "ALIVE"
    else:
        worker_state = "STOPPED"

    heartbeat = _parse_status_time(last_heartbeat_at)
    observed = _parse_status_time(now) or datetime.now(timezone.utc)
    heartbeat_age_ms: int | None = None
    if alive and heartbeat is not None:
        heartbeat_age_ms = max(0, int((observed - heartbeat).total_seconds() * 1000))
    return {
        "worker_kind": worker_kind,
        "worker_state": worker_state,
        "worker_alive": alive,
        "worker_required_for_runtime": bool(required_for_runtime),
        "worker_heartbeat_relevant": bool(required_for_runtime),
        "worker_last_seen": heartbeat.isoformat(timespec="milliseconds") if heartbeat else None,
        "worker_heartbeat_age_ms": heartbeat_age_ms,
        "worker_heartbeat_ttl_ms": WORKER_HEARTBEAT_TTL_MS,
        "worker_error": str(error or "") or None,
    }


@dataclass
class ServiceContainer:
    settings: IntegrationSettings
    logger: logging.Logger
    jobs: Any
    storage: Any
    registry: Any
    models: Any
    adapter: Any
    video_jobs: Any
    folder_dialogs: Any
    config_profiles: Any
    domain_profiles: Any
    statistics: Any
    feedback_candidates: Any
    feedback_optimization: Any
    history_management: Any
    runtime_resources: Any
    runtime_profiles: Any
    test_lock: Any
    warmup: "ModelWarmup"


class StartupCoordinator:
    """Expose a real API shell before service and ML initialization finishes.

    Uvicorn's lifespan must yield before it opens the listening socket.  The
    previous application built every service while importing ``backend.app``,
    so even process-only liveness was delayed by profile reconciliation and
    native ML imports.  This coordinator owns exactly one background builder,
    publishes a lock-bounded status snapshot, then delegates readiness to the
    one ``ModelWarmup`` instance owned by the service container.
    """

    def __init__(self, settings: IntegrationSettings, logger: logging.Logger):
        self.settings = settings
        self.logger = logger
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._app: FastAPI | None = None
        self._thread: threading.Thread | None = None
        self._warmup: ModelWarmup | None = None
        self._state = "pending"
        self._phase = "api_ready"
        self._stage = "API 已就绪，等待后台初始化"
        self._progress = 5
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._last_transition_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        self._last_heartbeat_at = self._last_transition_at
        self._error = ""

    def _set_stage(self, phase: str, stage: str, progress: int) -> None:
        with self._lock:
            self._state = "loading"
            self._phase = phase
            self._stage = stage
            self._progress = max(0, min(99, int(progress)))
            self._last_transition_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            self._last_heartbeat_at = self._last_transition_at
            elapsed = time.monotonic() - self._started_at if self._started_at else 0.0
        self.logger.info(
            "startup_stage phase=%s stage=%s progress=%s elapsed_seconds=%.3f",
            phase,
            stage,
            self._progress,
            elapsed,
        )

    def start(self, app: FastAPI) -> bool:
        with self._lock:
            self._app = app
            if self._thread is not None and self._thread.is_alive():
                return False
            if self._warmup is not None:
                return False
            self._state = "loading"
            self._phase = "loading_settings"
            self._stage = "正在加载服务配置"
            self._progress = 8
            self._started_at = time.monotonic()
            self._finished_at = None
            self._error = ""
            self._last_transition_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            self._thread = threading.Thread(
                target=self._run,
                name="AdaptiveAOI-ServiceInitializer",
                daemon=True,
            )
            worker = self._thread
        worker.start()
        return True

    def _run(self) -> None:
        try:
            app = self._app
            if app is None:
                raise RuntimeError("FastAPI application is unavailable")
            self._set_stage("loading_settings", "正在加载服务配置", 8)
            services = build_services(self.settings, self.logger, progress_callback=self._set_stage)
            app.state.services = services
            app.state.model_warmup = services.warmup
            with self._lock:
                self._warmup = services.warmup
            services.warmup.start()

            # Dependency cleanup is not required for the first screen or the
            # current frozen inference runtime.  Run it only after warm-up has
            # reached a terminal state so it cannot invert resource locks with
            # model construction.
            while not self._stop.wait(0.25):
                state = str(services.warmup.status().get("state") or "loading")
                if state not in {"pending", "loading", "starting"}:
                    break
            if not self._stop.is_set():
                # Start the build-only lazy import only after inference warm-up
                # reaches any terminal state (READY, NOT_READY or FAILED).
                # Concurrent cold imports can monopolize the GIL/native loader
                # long enough to delay liveness.  This ordering still covers a
                # clean installation with no Memory Bank, without making the
                # build warm-up a readiness gate.
                services.domain_profiles.warm_build_environment_async()
            if not self._stop.is_set() and str(services.warmup.status().get("state") or "") == "ready":
                try:
                    maintenance_started = time.perf_counter()
                    services.runtime_resources.reconcile_deleted_feedback_dependencies()
                    self.logger.info(
                        "deferred_startup_maintenance_completed task=reconcile_deleted_feedback_dependencies elapsed_seconds=%.3f",
                        time.perf_counter() - maintenance_started,
                    )
                except Exception:
                    # Maintenance is deliberately outside readiness.  Preserve
                    # the valid runtime and report the cleanup failure only in
                    # the service log for a later controlled retry.
                    self.logger.exception(
                        "deferred_startup_maintenance_failed task=reconcile_deleted_feedback_dependencies"
                    )
        except Exception as exc:
            with self._lock:
                self._state = "failed"
                self._phase = "failed"
                self._stage = "检测服务初始化失败"
                self._progress = min(95, max(5, self._progress))
                self._error = f"{type(exc).__name__}: {exc}"
                self._finished_at = time.monotonic()
                self._last_transition_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            self.logger.exception("service_initialization_failed")

    def restart(self) -> tuple[bool, str]:
        with self._lock:
            warmup = self._warmup
            app = self._app
            worker_alive = bool(self._thread and self._thread.is_alive())
            state = self._state
        if warmup is not None:
            snapshot = warmup.status()
            if str(snapshot.get("state") or "") == "loading":
                return False, "检测资源仍在加载，未创建重复任务。"
            warmup.restart()
            return True, "已重新启动检测资源加载。"
        if state != "failed" or app is None or worker_alive:
            return False, "后台服务仍在初始化，未创建重复任务。"
        return self.start(app), "已重新启动后台服务初始化。"

    def status(self) -> dict[str, object]:
        with self._lock:
            warmup = self._warmup
            elapsed = (self._finished_at or time.monotonic()) - self._started_at if self._started_at else 0.0
            snapshot_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            initializer_alive = bool(self._thread and self._thread.is_alive())
            snapshot = {
                "state": self._state,
                "phase": self._phase,
                "current_stage": self._stage,
                "progress": self._progress,
                "ready": False,
                "runtime_ready": False,
                "runtime_state": runtime_state_code(self._state, phase=self._phase),
                "elapsed_seconds": round(max(0.0, elapsed), 3),
                "error": self._error or None,
                "current_profile": None,
                "runtime": {},
                "runtime_context": {},
                "timings": {},
                "last_transition_at": self._last_transition_at,
                "status_snapshot_at": snapshot_at,
            }
            snapshot.update(
                worker_state_snapshot(
                    thread_alive=initializer_alive,
                    lifecycle_state=self._state,
                    required_for_runtime=False,
                    last_heartbeat_at=self._last_heartbeat_at,
                    now=snapshot_at,
                    error=self._error,
                )
            )
        if warmup is not None:
            snapshot.update(warmup.status())
        snapshot["api_ready"] = True
        snapshot["heartbeat_at"] = snapshot.get("status_snapshot_at")
        snapshot["runtime_ready"] = bool(snapshot.get("ready"))
        snapshot["runtime_state"] = runtime_state_code(
            str(snapshot.get("state") or ""),
            runtime_status=str((snapshot.get("runtime_context") or {}).get("runtime_status") or ""),
            phase=str(snapshot.get("phase") or ""),
        )
        return snapshot

    def shutdown(self) -> None:
        self._stop.set()
        app = self._app
        services = getattr(app.state, "services", None) if app is not None else None
        if services is None:
            return
        services.feedback_optimization.shutdown()
        services.video_jobs.shutdown()
        services.domain_profiles.shutdown()
        services.registry.shutdown()
        services.logger.info("service_stopped")


class ModelWarmup:
    """One non-blocking preload task with a lock-free public readiness view.

    Model/runtime construction holds the registry lock while PatchCore loads
    its backbone, memory bank and ANN index.  Health handlers therefore read
    only this small status snapshot; they never inspect the registry during
    warm-up.
    """

    def __init__(
        self,
        settings: IntegrationSettings,
        runtime_profiles: Any,
        adapter: Any,
        logger: logging.Logger,
        initial_profile: dict[str, object] | None = None,
    ):
        self.settings = settings
        self.runtime_profiles = runtime_profiles
        self.adapter = adapter
        self.logger = logger
        self._lock = threading.RLock()
        self._state = "pending"
        self._phase = "waiting"
        self._stage = "等待后台预热"
        self._progress = 0
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._error = ""
        self._last_heartbeat_at: str | None = None
        self._profile = self._profile_summary(initial_profile or {})
        self._runtime: dict[str, object] = dict((initial_profile or {}).get("runtime") or {})
        self._timings: dict[str, float] = {}
        self._thread: threading.Thread | None = None

    @staticmethod
    def _profile_summary(profile_state: dict[str, object]) -> dict[str, object]:
        resource = dict(profile_state.get("resource") or {})
        runtime = dict(profile_state.get("runtime") or {})
        return {
            "profile_id": resource.get("profile_id") or runtime.get("memory_bank_id"),
            "name": resource.get("display_name") or resource.get("profile_id"),
            "version": resource.get("version"),
            "memory_bank_id": resource.get("profile_id") or runtime.get("memory_bank_id"),
            "config_profile_id": runtime.get("config_profile_id"),
            "config_profile_version": runtime.get("config_profile_version"),
            "dataset": resource.get("dataset") or runtime.get("dataset"),
            "category": resource.get("category") or runtime.get("category"),
            "deployment_scope": resource.get("deployment_scope") or (resource.get("metadata") or {}).get("deployment_scope"),
            "frozen": bool(resource.get("frozen")),
        }

    def _set_stage(self, phase: str, stage: str, progress: int) -> None:
        with self._lock:
            self._phase = phase
            self._stage = stage
            self._progress = max(0, min(100, int(progress)))
            self._last_heartbeat_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            elapsed = time.monotonic() - self._started_at if self._started_at else 0.0
        self.logger.info(
            "startup_stage phase=%s stage=%s progress=%s elapsed_seconds=%.3f",
            phase,
            stage,
            self._progress,
            elapsed,
        )

    def start(self) -> None:
        context = self.runtime_profiles.context_snapshot()
        if not context.get("current_memory_bank_id"):
            with self._lock:
                self._state = "not_ready"
                self._phase = "missing_profile"
                self._stage = "当前未加载目标域记忆库"
                self._progress = 0
                self._started_at = None
                self._finished_at = time.monotonic()
                self._error = ""
                self._profile = {}
                self._runtime = dict(context.get("runtime") or {})
                self._timings = {}
                self._last_heartbeat_at = None
            return
        with self._lock:
            if self._state not in {"pending", "failed", "not_ready"}:
                return
            self._state = "loading"
            self._phase = "resolving_runtime"
            self._stage = "正在解析当前正式运行方案"
            self._progress = 20
            self._started_at = time.monotonic()
            self._finished_at = None
            self._error = ""
            self._timings = {}
            self._last_heartbeat_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            self._thread = threading.Thread(target=self._run, name="AdaptiveAOI-ModelWarmup", daemon=True)
            thread = self._thread
        self.logger.info("startup_stage phase=resolving_runtime stage=正在解析当前正式运行方案 progress=20 elapsed_seconds=0.000")
        thread.start()

    def restart(self) -> None:
        """Reload the currently applied profile after an atomic runtime switch."""
        context = self.runtime_profiles.context_snapshot()
        if not context.get("current_memory_bank_id"):
            with self._lock:
                self._state = "not_ready"
                self._phase = "missing_profile"
                self._stage = "当前未加载目标域记忆库"
                self._progress = 0
                self._started_at = None
                self._finished_at = time.monotonic()
                self._error = ""
                self._profile = {}
                self._runtime = dict(context.get("runtime") or {})
                self._timings = {}
                self._last_heartbeat_at = None
            return
        with self._lock:
            if self._state == "loading":
                return
            self._state = "pending"
            self._phase = "waiting"
            self._stage = "等待后台预热"
            self._progress = 0
            self._started_at = None
            self._finished_at = None
            self._error = ""
            self._last_heartbeat_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        self.start()

    def _run(self) -> None:
        expected_revision: int | None = None
        expected_profile_id: str | None = None
        try:
            defaults = self.settings.defaults
            self._set_stage("resolving_runtime", "正在解析当前正式运行方案", 25)
            profile_started = time.monotonic()
            runtime_config = self.runtime_profiles.configuration()
            profile_state = self.runtime_profiles.current_profile()
            context = self.runtime_profiles.context_snapshot()
            expected_revision = int(context.get("runtime_revision") or 0)
            expected_profile_id = str(context.get("current_profile_id") or "")
            profile_seconds = time.monotonic() - profile_started
            dynamic = dict(profile_state.get("runtime") or {})
            memory_bank_id = runtime_config.get("memory_bank_id")
            with self._lock:
                self._profile = self._profile_summary(profile_state)
                self._runtime = {
                    key: runtime_config.get(key)
                    for key in (
                        "model_id", "dataset", "category", "device", "threshold",
                        "memory_bank_id", "config_profile_id", "config_profile_version",
                        "inference_mode", "frozen", "memory_bank_ready", "threshold_ready",
                    )
                }
            if not memory_bank_id:
                with self._lock:
                    self._state = "not_ready"
                    self._phase = "missing_profile"
                    self._stage = "当前未加载目标域记忆库"
                    self._progress = 0
                    self._error = ""
                    self._finished_at = time.monotonic()
                self.runtime_profiles.set_runtime_status(
                    "not_ready",
                    expected_revision=expected_revision,
                    expected_profile_id=expected_profile_id,
                )
                self.logger.info("model_warmup_skipped_no_target_profile")
                return
            # Device resolution imports the Python ML runtime (PyTorch) before
            # the inferencer module can be constructed.  Publish that real
            # boundary before calling the adapter so a long native import is
            # never mislabeled as model construction.
            self._set_stage("initializing_python_runtime", "正在加载检测算法运行时组件", 30)
            runtime_started = time.monotonic()
            resolved, entry, cache_hit = self.adapter.resolve_video_runtime(
                runtime_config.get("model_id") or dynamic.get("model_id") or defaults.get("model_id"),
                runtime_config.get("dataset") or dynamic.get("dataset") or defaults.get("dataset"),
                runtime_config.get("category") or dynamic.get("category") or defaults.get("category"),
                runtime_config.get("device") or dynamic.get("device") or "auto",
                None if memory_bank_id else dynamic.get("threshold"),
                memory_bank_id,
                "inference_only",
                progress_callback=self._runtime_progress,
            )
            model_seconds = 0.0 if cache_hit else float(entry.inferencer.model_load_ms) / 1000.0
            memory_seconds = 0.0 if cache_hit else float(entry.inferencer.memory_bank_load_ms) / 1000.0
            runtime_seconds = time.monotonic() - runtime_started
            initialization_seconds = max(0.0, runtime_seconds - model_seconds - memory_seconds)
            import torch

            with self._lock:
                self._timings.update(
                    profile_resolution_seconds=round(profile_seconds, 3),
                    runtime_initialization_seconds=round(initialization_seconds, 3),
                    model_load_seconds=round(model_seconds, 3),
                    memory_bank_load_seconds=round(memory_seconds, 3),
                    cache_hit=bool(cache_hit),
                )
                self._runtime.update(
                    model_id=resolved.get("model_id"),
                    dataset=resolved.get("dataset"),
                    category=resolved.get("category"),
                    device=resolved.get("device"),
                    memory_bank_id=resolved.get("memory_bank_id"),
                    torch_version=str(torch.__version__),
                    cuda_version=str(torch.version.cuda or ""),
                    gpu_name=None,
                )
            self.logger.info("startup_stage_timing phase=loading_model stage=加载模型 stage_seconds=%.3f", model_seconds)
            self._set_stage("loading_memory_bank", "正在加载当前正式 Memory Bank", 82)
            self.logger.info("startup_stage_timing phase=loading_memory_bank stage=加载 Memory Bank stage_seconds=%.3f", memory_seconds)
            self._set_stage("loading_index", "正在加载并校验近邻检索索引", 90)
            self._set_stage("warming_up", "正在完成检测资源预热", 95)
            if not self.runtime_profiles.set_runtime_status(
                "ready",
                expected_revision=expected_revision,
                expected_profile_id=expected_profile_id,
            ):
                self.logger.info(
                    "model_warmup_result_discarded_stale_context profile=%s revision=%s",
                    expected_profile_id,
                    expected_revision,
                )
                return
            with self._lock:
                self._state = "ready"
                self._phase = "ready"
                self._stage = "预热完成"
                self._progress = 100
                self._finished_at = time.monotonic()
                total_seconds = self._finished_at - self._started_at if self._started_at else 0.0
                self._timings["total_warmup_seconds"] = round(total_seconds, 3)
            self.logger.info("startup_stage phase=ready stage=预热完成 progress=100 elapsed_seconds=%.3f", total_seconds)
        except Exception as exc:
            if expected_revision is not None:
                self.runtime_profiles.set_runtime_status(
                    "failed",
                    expected_revision=expected_revision,
                    expected_profile_id=expected_profile_id,
                )
            with self._lock:
                self._state = "failed"
                self._phase = "failed"
                self._stage = "预热失败"
                self._progress = min(95, max(5, self._progress))
                self._error = f"{type(exc).__name__}: {exc}"
                self._finished_at = time.monotonic()
            self.logger.exception("model_warmup_failed")

    def _runtime_progress(self, phase: str, stage: str, progress: int) -> None:
        """Receive real constructor milestones from the sole RuntimeRegistry."""
        self._set_stage(phase, stage, progress)

    def status(self) -> dict[str, object]:
        # Resource identity comes only from RuntimeProfileService.  Warm-up owns
        # loading progress, never a second independently cached current Profile.
        context = self.runtime_profiles.context_snapshot()
        with self._lock:
            snapshot_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            initializer_alive = bool(self._thread and self._thread.is_alive())
            elapsed = (self._finished_at or time.monotonic()) - self._started_at if self._started_at else 0.0
            runtime_snapshot = deepcopy(dict(context.get("runtime") or {}))
            for key in ("torch_version", "cuda_version", "gpu_name"):
                if self._runtime.get(key) is not None:
                    runtime_snapshot[key] = self._runtime.get(key)
            runtime_status = str(context.get("runtime_status") or "not_ready")
            if runtime_status == "deactivating":
                public_state = "deactivating"
                phase = "deactivating"
                stage = "正在停用当前目标域"
                progress = 0
            elif runtime_status in {"inactive", "not_ready"}:
                public_state = "not_ready"
                phase = "missing_profile"
                stage = "当前未加载目标域记忆库"
                progress = 0
            else:
                public_state = "ready" if self._state == "ready" else ("failed" if self._state == "failed" else "loading")
                phase = self._phase
                stage = self._stage
                progress = self._progress
            worker_snapshot = worker_state_snapshot(
                thread_alive=initializer_alive,
                lifecycle_state=self._state,
                required_for_runtime=False,
                last_heartbeat_at=self._last_heartbeat_at,
                now=snapshot_at,
                error=self._error,
            )
            runtime_state = runtime_state_code(
                public_state,
                runtime_status=runtime_status,
                phase=phase,
            )
            snapshot = {
                "state": public_state,
                "phase": phase,
                "current_stage": stage,
                "progress": progress,
                "ready": public_state == "ready",
                "runtime_ready": public_state == "ready",
                "runtime_state": runtime_state,
                "elapsed_seconds": round(max(0.0, elapsed), 3),
                "error": self._error or None,
                "current_profile": deepcopy(context.get("current_profile")),
                "runtime": runtime_snapshot,
                "runtime_context": context,
                "timings": dict(self._timings),
                "heartbeat_at": snapshot_at,
                "status_snapshot_at": snapshot_at,
            }
            snapshot.update(worker_snapshot)
            return snapshot


def build_services(
    settings: IntegrationSettings,
    logger: logging.Logger | None = None,
    *,
    progress_callback: Callable[[str, str, int], None] | None = None,
) -> ServiceContainer:
    # Import service implementations only after Uvicorn has exposed the
    # process-only liveness endpoint.  Several of these modules eventually
    # reach native image/ML dependencies and must not delay socket binding.
    from backend.adapters.patchcore_adapter import PatchCoreAdapter
    from backend.services.config_profile_service import ConfigProfileService
    from backend.services.evaluation_lock_service import EvaluationLockService
    from backend.services.domain_profile_service import DomainProfileService
    from backend.services.feedback_candidate_service import FeedbackCandidateService
    from backend.services.feedback_optimization_service import FeedbackOptimizationService
    from backend.services.folder_dialog_service import FolderDialogService
    from backend.services.history_management_service import HistoryManagementService
    from backend.services.job_service import JobService
    from backend.services.model_registry import ModelRegistry
    from backend.services.result_storage import ResultStorage
    from backend.services.runtime_profile_service import RuntimeProfileService
    from backend.services.runtime_registry import RuntimeRegistry
    from backend.services.runtime_resource_service import RuntimeResourceService
    from backend.services.statistics_service import StatisticsService
    from backend.services.video_job_service import VideoJobService

    build_started = time.perf_counter()
    settings.ensure_runtime_dirs()
    logger = logger or configure_logging(settings)
    logger.info("startup_stage phase=backend stage=启动后端 progress=5 elapsed_seconds=0.000")
    logger.info("startup_stage phase=configuration stage=加载配置 progress=12 elapsed_seconds=%.3f", time.perf_counter() - build_started)
    if progress_callback is not None:
        progress_callback("loading_settings", "正在构造轻量业务服务", 12)
    jobs = JobService(settings)
    test_lock = EvaluationLockService(settings)
    storage = ResultStorage(settings)
    registry = RuntimeRegistry(settings)
    models = ModelRegistry(settings, registry)
    config_profiles = ConfigProfileService(settings, logger)
    domain_profiles = DomainProfileService(
        settings,
        registry,
        logger,
        config_profiles=config_profiles,
        jobs=jobs,
        models=models,
    )
    adapter = PatchCoreAdapter(settings, jobs, storage, registry, models, domain_profiles, logger)
    video_jobs = VideoJobService(settings, jobs, adapter, registry, logger)
    feedback_candidates = FeedbackCandidateService(settings, storage, logger)
    domain_profiles.feedback_candidates = feedback_candidates
    feedback_optimization = FeedbackOptimizationService(
        settings,
        jobs,
        storage,
        feedback_candidates,
        domain_profiles,
        config_profiles,
        registry,
        video_jobs,
        logger,
    )
    folder_dialogs = FolderDialogService(settings.runtime_root / "recent_folder_dialog_paths.json")
    statistics = StatisticsService(settings, storage)
    history_management = HistoryManagementService(settings, storage, logger)
    runtime_resources = RuntimeResourceService(
        settings,
        config_profiles,
        domain_profiles,
        models,
        logger,
        registry=registry,
        jobs=jobs,
    )
    runtime_profiles = RuntimeProfileService(settings, runtime_resources, logger, test_lock=test_lock)
    runtime_resources.runtime_profiles = runtime_profiles
    runtime_resources.test_lock = test_lock
    runtime_resources.feedback_candidates = feedback_candidates
    runtime_resources.history_management = history_management
    domain_profiles.runtime_resources = runtime_resources
    domain_profiles.test_lock = test_lock
    feedback_optimization.runtime_resources = runtime_resources
    feedback_optimization.runtime_profiles = runtime_profiles
    feedback_optimization.test_lock = test_lock
    # Resolve an invalid/test-only persisted binding before any model warm-up
    # or HTTP endpoint can observe it.
    if progress_callback is not None:
        progress_callback("resolving_runtime", "正在解析当前正式运行方案", 20)
    profile_started = time.perf_counter()
    initial_profile = runtime_profiles.bootstrap()
    logger.info("startup_stage phase=profile stage=加载 Profile progress=20 elapsed_seconds=%.3f", time.perf_counter() - profile_started)
    warmup = ModelWarmup(settings, runtime_profiles, adapter, logger, initial_profile=initial_profile)
    runtime_resources.warmup = warmup
    return ServiceContainer(settings, logger, jobs, storage, registry, models, adapter, video_jobs, folder_dialogs, config_profiles, domain_profiles, statistics, feedback_candidates, feedback_optimization, history_management, runtime_resources, runtime_profiles, test_lock, warmup)


def create_app(config_path: str | None = None) -> FastAPI:
    settings = IntegrationSettings.load(config_path)
    settings.ensure_runtime_dirs()
    logger = configure_logging(settings)
    startup = StartupCoordinator(settings, logger)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("service_started host=%s port=%s", settings.host(), settings.port())
        startup.start(app)
        yield
        startup.shutdown()

    app = FastAPI(
        title="AdaptiveAOI Integration API",
        version=str(settings.backend.get("version", "1.0.0")),
        description="Typed API binding the Streamlit system to the existing AdaptiveAOI PatchCore experiment core.",
        lifespan=lifespan,
    )
    app.state.services = None
    app.state.logger = logger
    app.state.startup = startup
    app.state.model_warmup = None
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.backend.get("cors_origins", [])),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT"],
        allow_headers=["*"],
    )

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None) or request.headers.get("X-Request-ID") or uuid.uuid4().hex
        fields = [
            {
                "field": ".".join(str(part) for part in error.get("loc", []) if part not in {"body", "query", "path"}),
                "message": str(error.get("msg") or "参数无效"),
                "type": str(error.get("type") or "validation_error"),
            }
            for error in exc.errors()
        ]
        response = JSONResponse(
            status_code=422,
            content={
                "success": False,
                "request_id": request_id,
                "result": None,
                "error": {
                    "code": "REQUEST_VALIDATION_FAILED",
                    "message": "请求参数不符合接口要求，请检查输入后重试。",
                    "details": {"fields": fields},
                },
            },
        )
        response.headers["X-Request-ID"] = request_id
        return response

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        try:
            response = await call_next(request)
        except ApiError as exc:
            response = JSONResponse(
                status_code=exc.status_code,
                # Service errors may contain pathlib.Path/numpy scalar values
                # from compatibility metadata.  Encode the complete error
                # payload before constructing JSONResponse so a business
                # 409/4xx is never turned into a secondary 500 while trying
                # to serialize its diagnostics.
                content=jsonable_encoder({
                    "success": False,
                    "request_id": request.state.request_id,
                    "result": None,
                    "error": {"code": exc.code, "message": exc.message, "details": exc.details},
                }),
            )
        except Exception:
            request.app.state.logger.exception("unhandled_request_error request_id=%s path=%s", request.state.request_id, request.url.path)
            response = JSONResponse(
                status_code=500,
                content={"success": False, "request_id": request.state.request_id, "result": None, "error": {"code": "INTERNAL_ERROR", "message": "请求未能完成，请检查算法服务日志及模型资源。", "details": {}}},
            )
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    app.mount("/artifacts", StaticFiles(directory=str(settings.jobs_root)), name="artifacts")
    app.include_router(router)
    return app


app = create_app()
