from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from bridge.api_client import client
from bridge.config_loader import load_software_config


@dataclass(frozen=True, slots=True)
class RuntimeStatusSnapshot:
    """Compact authoritative Runtime fact used by every detection page.

    Readiness owns the transition state while ``runtime_context`` owns the
    immutable inference binding.  Keeping both in one value prevents a page
    from combining a cached ready configuration with a newer loading state.
    """

    runtime_revision: int = 0
    status: str = "not_ready"
    runtime_state: str = "NOT_READY"
    runtime_ready: bool | None = None
    worker_state: str = "STOPPED"
    worker_alive: bool = False
    worker_required_for_runtime: bool = False
    worker_kind: str = "model_warmup_initializer"
    worker_last_seen: str = ""
    worker_heartbeat_age_ms: int | None = None
    worker_heartbeat_ttl_ms: int = 30000
    worker_heartbeat_relevant: bool = False
    worker_error: str = ""
    status_snapshot_at: str = ""
    profile_id: str = ""
    memory_bank_id: str = ""
    config_id: str = ""
    model_id: str = ""
    dataset: str = ""
    category: str = ""
    updated_at: str = ""
    profile_name: str = ""
    profile_version: str = ""
    config_version: int = 0
    device: str = "auto"
    config: str = ""
    threshold: float | None = None
    frozen: bool = False
    memory_bank_ready: bool = False
    config_hash: str = ""
    method_mode: str = ""
    phase: str = ""
    stage: str = ""
    progress: int = 0
    message: str = ""
    error: str = ""

    @property
    def ready(self) -> bool:
        return bool(self.runtime_ready if self.runtime_ready is not None else self.status == "ready")

    @property
    def binding(self) -> tuple[int, str, str, str, str]:
        return (
            self.runtime_revision,
            self.profile_id,
            self.memory_bank_id,
            self.config_id,
            self.model_id,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

def _fallback_runtime() -> dict[str, Any]:
    config = load_software_config()
    selected_config = str(config["default_config"])
    return {
        # The browser shell must never resurrect a stale local selection while
        # the backend source of truth is unavailable.
        "dataset": "",
        "category": "",
        "config": selected_config,
        "device": str(config["device"]),
        "model_id": "",
        "memory_bank_id": "",
        "threshold": None,
        "inference_mode": "inference_only",
        "mode": "inference_only",
        "frozen": False,
        "memory_bank_ready": False,
        "threshold_ready": False,
        "memory_bank_profile": {},
        "config_profile_id": "",
        "config_profile_version": None,
        "config_hash": "",
        "config_profile": {},
        "config_profile_name": "未选择配置方案",
        "runtime_error": "运行配置服务暂时不可用，检测入口已保持禁用。",
        "runtime_context": {},
    }


def _normalize_runtime_context(context: dict[str, Any], runtime: dict[str, Any] | None = None) -> dict[str, Any]:
    runtime = dict(runtime or context.get("runtime") or {})
    current_profile = dict(context.get("current_profile") or {})
    resource = dict(context.get("resource") or {})
    metadata = dict(resource.get("metadata") or {})
    model_id = str(context.get("model_id") or runtime.get("model_id") or metadata.get("model_id") or "")
    return {
        "runtime_revision": int(context.get("runtime_revision") or runtime.get("runtime_revision") or 0),
        "profile_id": str(context.get("profile_id") or context.get("current_profile_id") or current_profile.get("profile_id") or ""),
        "profile_version": context.get("profile_version") or current_profile.get("version"),
        "memory_bank_id": str(context.get("memory_bank_id") or context.get("current_memory_bank_id") or runtime.get("memory_bank_id") or ""),
        "config_profile_id": str(context.get("config_profile_id") or context.get("current_config_id") or runtime.get("config_profile_id") or ""),
        "config_version": int(context.get("config_version") or context.get("current_config_version") or runtime.get("config_profile_version") or 0),
        "model_id": model_id,
        "model_version": str(context.get("model_version") or model_id),
        "dataset": str(context.get("dataset") or runtime.get("dataset") or ""),
        "category": str(context.get("category") or runtime.get("category") or ""),
        "method_mode": str(context.get("method_mode") or resource.get("method_mode") or metadata.get("method_mode") or ""),
        "status": str(context.get("status") or context.get("runtime_status") or runtime.get("runtime_status") or "not_ready"),
        "updated_at": str(context.get("updated_at") or runtime.get("runtime_updated_at") or ""),
    }


def runtime_status_snapshot_from_readiness(payload: dict[str, Any]) -> RuntimeStatusSnapshot:
    """Normalize one readiness response without consulting page-local state."""

    if not payload.get("success"):
        raw_error = payload.get("error")
        error = dict(raw_error) if isinstance(raw_error, dict) else {"message": str(raw_error or "")}
        return RuntimeStatusSnapshot(
            status="starting",
            stage="正在连接算法服务",
            message=str(error.get("message") or "算法服务正在启动。"),
            error=str(error.get("message") or ""),
        )
    raw_context = dict(payload.get("runtime_context") or {})
    context = _normalize_runtime_context(raw_context)
    current_profile = dict(payload.get("current_profile") or raw_context.get("current_profile") or {})
    runtime = dict(raw_context.get("runtime") or {})
    resource = dict(raw_context.get("resource") or {})
    metadata = dict(resource.get("metadata") or {})
    threshold_value = runtime.get("threshold")
    if threshold_value is None:
        threshold_value = metadata.get("threshold")
    threshold = None if threshold_value is None else float(threshold_value)
    frozen = bool(current_profile.get("frozen", resource.get("frozen", metadata.get("frozen", False))))
    status = str(payload.get("status") or context.get("status") or "loading").strip().lower()
    if status not in {"pending", "starting", "loading", "ready", "not_ready", "failed", "inactive"}:
        status = "not_ready"
    explicit_runtime_state = str(payload.get("runtime_state") or "").strip().upper()
    if explicit_runtime_state in {
        "STARTING", "LOADING", "WARMING_UP", "READY", "DEGRADED", "FAILED", "NOT_READY"
    }:
        runtime_state = explicit_runtime_state
    else:
        runtime_state = {
            "pending": "STARTING",
            "starting": "LOADING",
            "loading": "LOADING",
            "ready": "READY",
            "failed": "FAILED",
            "not_ready": "NOT_READY",
            "inactive": "NOT_READY",
        }.get(status, "STARTING")
    worker_state = str(payload.get("worker_state") or "STOPPED").strip().upper()
    if worker_state not in {"STOPPED", "STARTING", "ALIVE", "IDLE", "FAILED"}:
        worker_state = "STOPPED"
    worker_age = payload.get("worker_heartbeat_age_ms")
    worker_age = None if worker_age is None else max(0, int(worker_age))
    return RuntimeStatusSnapshot(
        runtime_revision=int(context.get("runtime_revision") or 0),
        status=status,
        runtime_state=runtime_state,
        runtime_ready=bool(payload.get("runtime_ready", status == "ready")),
        worker_state=worker_state,
        worker_alive=bool(payload.get("worker_alive")),
        worker_required_for_runtime=bool(payload.get("worker_required_for_runtime")),
        worker_kind=str(payload.get("worker_kind") or "model_warmup_initializer"),
        worker_last_seen=str(payload.get("worker_last_seen") or ""),
        worker_heartbeat_age_ms=worker_age,
        worker_heartbeat_ttl_ms=max(1, int(payload.get("worker_heartbeat_ttl_ms") or 30000)),
        worker_heartbeat_relevant=bool(payload.get("worker_heartbeat_relevant")),
        worker_error=str(payload.get("worker_error") or ""),
        status_snapshot_at=str(payload.get("status_snapshot_at") or payload.get("heartbeat_at") or ""),
        profile_id=str(context.get("profile_id") or current_profile.get("profile_id") or ""),
        memory_bank_id=str(context.get("memory_bank_id") or current_profile.get("memory_bank_id") or ""),
        config_id=str(context.get("config_profile_id") or current_profile.get("config_profile_id") or ""),
        model_id=str(context.get("model_id") or ""),
        dataset=str(context.get("dataset") or current_profile.get("dataset") or ""),
        category=str(context.get("category") or current_profile.get("category") or ""),
        updated_at=str(context.get("updated_at") or ""),
        profile_name=str(current_profile.get("name") or resource.get("display_name") or ""),
        profile_version=str(context.get("profile_version") or current_profile.get("version") or ""),
        config_version=int(context.get("config_version") or current_profile.get("config_profile_version") or 0),
        device=str(runtime.get("device") or "auto"),
        config=str(runtime.get("config") or ""),
        threshold=threshold,
        frozen=frozen,
        memory_bank_ready=bool(frozen and threshold is not None and context.get("memory_bank_id")),
        config_hash=str(runtime.get("config_hash") or metadata.get("config_hash") or ""),
        method_mode=str(context.get("method_mode") or resource.get("method_mode") or ""),
        phase=str(payload.get("phase") or ""),
        stage=str(payload.get("current_stage") or ""),
        progress=max(0, min(100, int(payload.get("progress") or 0))),
        message=str(payload.get("message") or ""),
        error=str(payload.get("error") or ""),
    )


def load_runtime_status_snapshot() -> RuntimeStatusSnapshot:
    """Fetch the single lightweight Runtime UI fact."""

    return runtime_status_snapshot_from_readiness(client().readiness())


def merge_runtime_status(
    runtime: dict[str, Any],
    snapshot: RuntimeStatusSnapshot,
) -> dict[str, Any]:
    """Overlay authoritative transition facts on a cached full configuration."""

    merged = dict(runtime)
    context = dict(merged.get("runtime_context") or {})
    context.update(
        {
            "runtime_revision": snapshot.runtime_revision,
            "status": snapshot.status,
            "profile_id": snapshot.profile_id,
            "memory_bank_id": snapshot.memory_bank_id,
            "config_profile_id": snapshot.config_id,
            "model_id": snapshot.model_id,
            "dataset": snapshot.dataset,
            "category": snapshot.category,
            "updated_at": snapshot.updated_at,
        }
    )
    merged["runtime_context"] = context
    merged["runtime_status"] = snapshot.status
    merged["runtime_revision"] = snapshot.runtime_revision
    merged["runtime_ready"] = snapshot.ready
    merged["runtime_state"] = snapshot.runtime_state
    merged["worker_state"] = snapshot.worker_state
    merged["worker_alive"] = snapshot.worker_alive
    merged["worker_required_for_runtime"] = snapshot.worker_required_for_runtime
    merged["worker_kind"] = snapshot.worker_kind
    merged["worker_last_seen"] = snapshot.worker_last_seen
    merged["worker_heartbeat_age_ms"] = snapshot.worker_heartbeat_age_ms
    merged["worker_heartbeat_ttl_ms"] = snapshot.worker_heartbeat_ttl_ms
    merged["worker_heartbeat_relevant"] = snapshot.worker_heartbeat_relevant
    merged["worker_error"] = snapshot.worker_error
    merged["status_snapshot_at"] = snapshot.status_snapshot_at
    merged["_ui_runtime_snapshot"] = snapshot.as_dict()
    # Identity fields from the authoritative snapshot override stale cached
    # values whenever they are present. During the earliest startup stage the
    # cached descriptive values may remain for layout continuity, but status
    # still remains loading and detection stays disabled.
    for key, value in (
        ("memory_bank_id", snapshot.memory_bank_id),
        ("config_profile_id", snapshot.config_id),
        ("model_id", snapshot.model_id),
        ("dataset", snapshot.dataset),
        ("category", snapshot.category),
    ):
        if value:
            merged[key] = value
    merged["device"] = snapshot.device or "auto"
    merged["config"] = snapshot.config
    merged["threshold"] = snapshot.threshold
    merged["frozen"] = snapshot.frozen
    merged["memory_bank_ready"] = snapshot.memory_bank_ready
    merged["threshold_ready"] = snapshot.threshold is not None
    merged["config_profile_version"] = snapshot.config_version or None
    merged["config_hash"] = snapshot.config_hash
    merged["profile_version"] = snapshot.profile_version
    merged["method_mode"] = snapshot.method_mode
    existing_profile = dict(merged.get("memory_bank_profile") or {})
    existing_profile.update(
        {
            "friendly_name": snapshot.profile_name or snapshot.profile_id or "当前目标域版本",
            "memory_bank_name": snapshot.profile_name or snapshot.memory_bank_id,
            "profile_id": snapshot.profile_id,
            "memory_bank_id": snapshot.memory_bank_id,
            "method_mode": snapshot.method_mode,
            "method_name": snapshot.method_mode,
            "frozen": snapshot.frozen,
        }
    )
    merged["memory_bank_profile"] = existing_profile
    merged["config_profile_name"] = (
        str(merged.get("config_profile_name") or "")
        if str(merged.get("config_profile_id") or "") == snapshot.config_id
        else ""
    ) or ("当前冻结配置" if snapshot.config_id else "未选择配置方案")
    return merged


def load_runtime_configuration() -> dict[str, Any]:
    payload = client().config()
    config_payload = dict(payload.get("config") or {}) if payload.get("success") else {}
    runtime = dict(config_payload.get("runtime") or {})
    if not runtime:
        return _fallback_runtime()
    normalized = dict(runtime)
    normalized["runtime_context"] = _normalize_runtime_context(
        dict(config_payload.get("runtime_context") or {}),
        normalized,
    )
    normalized["inference_mode"] = "inference_only"
    normalized["mode"] = "inference_only"
    normalized["memory_bank_ready"] = bool(
        normalized.get("memory_bank_ready")
        and normalized.get("frozen")
        and normalized.get("threshold") is not None
    )
    return normalized


def load_runtime_context() -> dict[str, Any]:
    """Fetch the current authoritative binding immediately before inference."""

    payload = client().runtime_context()
    if not payload.get("success"):
        error = dict(payload.get("error") or {})
        return {
            "ok": False,
            "context": {},
            "error": str(error.get("message") or "当前正式运行配置暂时无法读取。"),
            "api_error": error,
        }
    context = _normalize_runtime_context(dict(payload.get("runtime_context") or {}))
    required = ("profile_id", "memory_bank_id", "config_profile_id", "model_id", "dataset", "category")
    missing = [key for key in required if not context.get(key)]
    if context.get("status") != "ready" or missing:
        return {
            "ok": False,
            "context": context,
            "error": "当前正式运行方案尚未就绪，请等待配置应用完成。",
            "api_error": {"code": "RUNTIME_PROFILE_NOT_READY", "details": {"missing_fields": missing}},
        }
    return {"ok": True, "context": context}


def apply_runtime_configuration(updates: dict[str, Any]) -> dict[str, Any]:
    payload = client().update_config(updates)
    if not payload.get("success"):
        error = payload.get("error") or {}
        return {"ok": False, "error": str(error.get("message") or "运行配置应用失败，请稍后重试。")}
    runtime = dict(payload.get("runtime") or {})
    if not runtime:
        return {"ok": False, "error": "后端没有返回已应用的运行配置。"}
    runtime["inference_mode"] = "inference_only"
    runtime["mode"] = "inference_only"
    return {"ok": True, "runtime": runtime}


def compatible_memory_banks(dataset: str, category: str, model_id: str) -> dict[str, Any]:
    payload = client().memory_banks(dataset=dataset, category=category, model_id=model_id)
    if not payload.get("success"):
        error = payload.get("error") or {}
        return {"ok": False, "profiles": [], "error": str(error.get("message") or "记忆库列表读取失败。")}
    profiles = [
        item
        for item in list(payload.get("memory_banks") or [])
        if bool(item.get("frozen")) and item.get("threshold") is not None
    ]
    profiles.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    profiles.sort(key=lambda item: (not bool(item.get("frozen")), not bool(item.get("is_default"))))
    return {"ok": True, "profiles": profiles}


__all__ = [
    "RuntimeStatusSnapshot",
    "apply_runtime_configuration",
    "compatible_memory_banks",
    "load_runtime_configuration",
    "load_runtime_context",
    "load_runtime_status_snapshot",
    "merge_runtime_status",
    "runtime_status_snapshot_from_readiness",
]
