from __future__ import annotations

import json
import mimetypes
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse

from backend.api.schemas import (
    CalibrationActivateRequest,
    CalibrationRequest,
    ConfigProfileCopyRequest,
    ConfigProfileCreateRequest,
    ConfigProfileImportRequest,
    ConfigProfileUpdateRequest,
    ConfigUpdateRequest,
    DomainBuildRequest,
    DomainCandidateActionRequest,
    DomainParameterVersionRequest,
    DomainDraftSaveRequest,
    DomainScanRequest,
    FeedbackBulkRequest,
    FeedbackCandidateBulkReviewRequest,
    FeedbackCandidateDeleteRequest,
    FeedbackCandidateReviewRequest,
    FeedbackAnnotationRequest,
    FeedbackOptimizationApplyRequest,
    FeedbackOptimizationCreateRequest,
    FeedbackOptimizationRollbackRequest,
    FeedbackRequest,
    FolderDialogRecentDirectoryRequest,
    FolderDialogRequest,
    HistoryAllDeletePreviewRequest,
    HistoryAllDeleteRequest,
    HistoryCleanupRetryRequest,
    HistoryDeletePreviewRequest,
    HistoryDeleteRequest,
    HistoryTaskItemsDeleteRequest,
    MemoryBankActivateRequest,
    MemoryBankConfigCreateRequest,
    MemoryBankConfigDeleteRequest,
    MemoryBankConfigUpdateRequest,
    MemoryBankRenameRequest,
    ModelLoadRequest,
    OfficialTestLockRequest,
    RuntimeCleanupRequest,
    RuntimeBatchActionRequest,
    RuntimeClearAllRequest,
    RuntimeConfigurationBatchRequest,
    RuntimeProfileApplyRequest,
    RuntimeResourceActionRequest,
    VideoJobCreateRequest,
)
from backend.errors import ApiError
from backend.services.detection_context import DetectionContext
from backend.services.upload_stream_service import stream_upload_to_path


router = APIRouter(prefix="/api")


def _services(request: Request) -> Any:
    services = getattr(request.app.state, "services", None)
    if services is None:
        startup = getattr(request.app.state, "startup", None)
        snapshot = startup.status() if startup is not None else {}
        raise ApiError(
            "SERVICE_INITIALIZING",
            "算法服务正在加载运行资源，请稍候重试。",
            503,
            {
                "phase": snapshot.get("phase"),
                "current_stage": snapshot.get("current_stage"),
                "progress": snapshot.get("progress"),
            },
        )
    return services


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "unknown"))


def _success(request: Request, **payload: Any) -> dict[str, Any]:
    return {"success": True, "request_id": _request_id(request), "error": None, **payload}


def _optional_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _require_local_request(request: Request) -> None:
    host = str(request.client.host if request.client else "").lower()
    if host not in {"127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost"}:
        raise ApiError("LOCAL_REQUEST_REQUIRED", "文件夹选择仅允许从本机界面调用。", 403)


def _validated_config_name(settings: Any, value: str) -> str:
    path = settings.resolve_path(value)
    config_root = (settings.project_root / "configs").resolve()
    try:
        relative = path.relative_to(settings.project_root)
        path.relative_to(config_root)
    except ValueError as exc:
        raise ApiError("INVALID_RUNTIME_CONFIG", "配置文件必须来自当前项目的 configs 目录。", 422) from exc
    if path.suffix.lower() not in {".yaml", ".yml"} or not path.is_file():
        raise ApiError("INVALID_RUNTIME_CONFIG", "所选配置文件不存在或不是有效的 YAML 配置。", 422)
    return relative.as_posix()


def _validate_inference_binding(
    services: Any,
    *,
    profile_id: str | None,
    runtime_revision: int | None,
    memory_bank_id: str | None,
    model_id: str | None,
    dataset: str | None,
    category: str | None,
    config_profile_id: str | None,
    config_profile_version: int | None,
) -> dict[str, Any]:
    """Resolve omitted inference fields to the one applied frozen runtime.

    Local callers historically could omit ``memory_bank_id`` and profile
    binding fields.  The algorithm adapter then selected the newest compatible
    resource, which could differ from the profile displayed on the runtime
    configuration page.  Resolve and validate the complete binding here so
    every entry point uses the same frozen resource.
    """
    context = services.runtime_profiles.context_snapshot()
    runtime = _runtime_configuration(services)
    if (
        str(runtime.get("profile_status") or "") != "ready"
        or str(context.get("status") or context.get("runtime_status") or "") != "ready"
    ):
        raise ApiError("RUNTIME_PROFILE_NOT_READY", "检测服务已启动，当前未加载目标域记忆库。", 409)
    expected = {
        "profile_id": str(context.get("profile_id") or context.get("current_profile_id") or ""),
        "runtime_revision": int(context.get("runtime_revision") or 0),
        "memory_bank_id": str(runtime.get("memory_bank_id") or ""),
        "model_id": str(runtime.get("model_id") or ""),
        "dataset": str(runtime.get("dataset") or ""),
        "category": str(runtime.get("category") or ""),
        "config_profile_id": str(runtime.get("config_profile_id") or ""),
        "config_profile_version": int(runtime.get("config_profile_version") or 0),
    }
    supplied = {
        "profile_id": profile_id,
        "runtime_revision": runtime_revision,
        "memory_bank_id": memory_bank_id,
        "model_id": model_id,
        "dataset": dataset,
        "category": category,
        "config_profile_id": config_profile_id,
        "config_profile_version": config_profile_version,
    }
    mismatched = [
        key for key, value in supplied.items()
        if value not in (None, "") and str(value).lower() != str(expected[key]).lower()
    ]
    if mismatched:
        raise ApiError(
            "INFERENCE_RUNTIME_BINDING_MISMATCH",
            "运行配置刚刚发生变化，已拒绝使用过期资源开始检测。请重新开始检测。",
            409,
            {
                "mismatched_fields": mismatched,
                "runtime_revision": expected["runtime_revision"],
                "profile_id": expected["profile_id"],
            },
        )
    resolved_memory_bank_id = expected["memory_bank_id"]
    resolved_model_id = expected["model_id"]
    resolved_dataset = expected["dataset"]
    resolved_category = expected["category"]
    resolved_profile_id = expected["config_profile_id"]
    resolved_profile_version = expected["config_profile_version"]
    bank = services.domain_profiles.get_profile(
        resolved_memory_bank_id,
        resolved_dataset,
        resolved_category,
        resolved_model_id,
    )
    if not bank.get("frozen") or bank.get("threshold") is None:
        raise ApiError("INFERENCE_RESOURCE_NOT_READY", "当前运行方案缺少已冻结的记忆库或阈值。", 409)
    compatibility = services.config_profiles.compatibility(resolved_profile_id, resolved_profile_version, bank)
    if not compatibility.get("compatible"):
        raise ApiError(
            "INFERENCE_CONFIG_BINDING_MISMATCH",
            "检测请求的配置不属于当前记忆库或与其特征空间不兼容。",
            409,
            {"reasons": compatibility.get("reasons") or []},
        )
    config_detail = services.config_profiles.get(resolved_profile_id, resolved_profile_version)
    config_values = dict(config_detail.get("config") or {})
    resolved_threshold = config_detail.get("threshold") if config_detail.get("threshold") is not None else bank.get("threshold")
    precision = str(
        bank.get("actual_precision")
        or bank.get("requested_precision")
        or bank.get("precision")
        or config_values.get("actual_precision")
        or config_values.get("requested_precision")
        or config_values.get("precision")
        or ("fp16" if config_values.get("use_fp16") else "fp32")
    ).lower()
    feature_layers = (
        bank.get("feature_layers")
        or config_values.get("feature_layers")
        or context.get("feature_layers")
        or []
    )
    algorithm_mode = str(
        bank.get("algorithm_mode")
        or bank.get("method_mode_resolved")
        or config_values.get("algorithm_mode")
        or config_values.get("method_mode_resolved")
        or context.get("method_mode")
        or "unknown"
    )
    detection_context = DetectionContext.create(
        profile_id=expected["profile_id"],
        memory_bank_id=resolved_memory_bank_id,
        config_id=resolved_profile_id,
        model_id=resolved_model_id,
        threshold=resolved_threshold,
        precision=precision,
        feature_layers=feature_layers,
        runtime_revision=expected["runtime_revision"],
        algorithm_mode=algorithm_mode,
    )
    return {
        "profile_id": expected["profile_id"],
        "profile_version": context.get("profile_version"),
        "runtime_revision": expected["runtime_revision"],
        "memory_bank_id": resolved_memory_bank_id,
        "model_id": resolved_model_id,
        "dataset": resolved_dataset,
        "category": resolved_category,
        "config_profile_id": resolved_profile_id,
        "config_profile_version": resolved_profile_version,
        "config_hash": str(config_detail.get("config_hash") or ""),
        "config_values": config_values,
        "threshold": resolved_threshold,
        "device": str(config_values.get("device") or runtime.get("device") or "auto"),
        "detection_context": detection_context,
    }


def _runtime_configuration(services: Any) -> dict[str, Any]:
    return services.runtime_profiles.configuration()


def _runtime_worker_fields(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Project one startup snapshot consistently across health APIs.

    worker_alive remains for compatibility, but it is not a Runtime
    readiness gate.  The explicit worker fields explain whether a stopped
    worker is expected for the current process.
    """

    return {
        "runtime_ready": bool(snapshot.get("runtime_ready", snapshot.get("ready"))),
        "runtime_state": str(snapshot.get("runtime_state") or "STARTING"),
        "worker_kind": str(snapshot.get("worker_kind") or "model_warmup_initializer"),
        "worker_state": str(snapshot.get("worker_state") or "STOPPED"),
        "worker_alive": bool(snapshot.get("worker_alive")),
        "worker_required_for_runtime": bool(snapshot.get("worker_required_for_runtime")),
        "worker_heartbeat_relevant": bool(snapshot.get("worker_heartbeat_relevant")),
        "worker_last_seen": snapshot.get("worker_last_seen"),
        "worker_heartbeat_age_ms": snapshot.get("worker_heartbeat_age_ms"),
        "worker_heartbeat_ttl_ms": int(snapshot.get("worker_heartbeat_ttl_ms") or 30000),
        "worker_error": snapshot.get("worker_error"),
        "status_snapshot_at": snapshot.get("status_snapshot_at") or snapshot.get("heartbeat_at"),
    }


@router.post("/local/folder-dialog")
def start_folder_dialog(payload: FolderDialogRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    try:
        dialog = _services(request).folder_dialogs.start(
            payload.initial_dir,
            dialog_type=payload.dialog_type,
            multiple=payload.multiple,
            file_types=payload.file_types,
            title="选择视频文件" if payload.dialog_type in {"file", "files"} else "选择文件夹",
            recent_key=payload.recent_key,
            fallback_dirs=payload.fallback_dirs,
        )
    except ValueError as exc:
        raise ApiError("INVALID_FOLDER_DIALOG_TYPE", str(exc), 422) from exc
    return _success(request, folder_dialog=dialog)


@router.post("/local/folder-dialog/recent-directory")
def remember_folder_dialog_directory(
    payload: FolderDialogRecentDirectoryRequest,
    request: Request,
) -> dict[str, Any]:
    _require_local_request(request)
    try:
        recent_directory = _services(request).folder_dialogs.remember_directory(
            payload.recent_key,
            payload.directory,
        )
    except ValueError as exc:
        raise ApiError("INVALID_RECENT_DIRECTORY", str(exc), 422) from exc
    return _success(request, recent_directory=recent_directory)


@router.get("/local/folder-dialog/{dialog_id}")
def folder_dialog_status(dialog_id: str, request: Request, consume: bool = False) -> dict[str, Any]:
    _require_local_request(request)
    if len(dialog_id) != 32 or any(character not in "0123456789abcdef" for character in dialog_id.lower()):
        raise ApiError("INVALID_FOLDER_DIALOG_ID", "文件夹选择请求无效。", 422)
    dialog = _services(request).folder_dialogs.status(dialog_id, consume=consume)
    if dialog is None:
        raise ApiError("FOLDER_DIALOG_NOT_FOUND", "文件夹选择请求不存在或已结束。", 404)
    return _success(request, folder_dialog=dialog)


@router.post("/local/folder-dialog/{dialog_id}/cancel")
def cancel_folder_dialog(dialog_id: str, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    if len(dialog_id) != 32 or any(character not in "0123456789abcdef" for character in dialog_id.lower()):
        raise ApiError("INVALID_FOLDER_DIALOG_ID", "文件夹选择请求无效。", 422)
    dialog = _services(request).folder_dialogs.cancel(dialog_id)
    if dialog is None:
        raise ApiError("FOLDER_DIALOG_NOT_FOUND", "文件夹选择请求不存在或已结束。", 404)
    return _success(request, folder_dialog=dialog)


@router.get("/domain-adaptation/defaults")
def domain_adaptation_defaults(
    request: Request,
    dataset: str,
    category: str,
    model_id: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    _require_local_request(request)
    defaults = _services(request).domain_profiles.defaults(model_id, dataset, category, device)
    return _success(request, defaults=defaults)


def _domain_payload_with_feedback(request: Request, payload: DomainScanRequest | DomainBuildRequest) -> dict[str, Any]:
    data = payload.model_dump()
    candidate_ids = list(data.get("feedback_candidate_ids") or [])
    data["_feedback_samples"] = _services(request).feedback_candidates.domain_build_candidates(
        candidate_ids, include_internal_paths=True
    ) if candidate_ids else []
    return data


@router.get("/domain-adaptation/drafts")
def list_domain_drafts(request: Request) -> dict[str, Any]:
    _require_local_request(request)
    service = _services(request).domain_profiles
    return _success(request, drafts=service.list_drafts(), draft_revision=service.draft_revision())


@router.get("/domain-adaptation/drafts/{draft_id}")
def get_domain_draft(draft_id: str, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, draft=_services(request).domain_profiles.get_draft(draft_id))


@router.put("/domain-adaptation/drafts")
def save_domain_draft(payload: DomainDraftSaveRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, draft=_services(request).domain_profiles.save_draft(payload.model_dump()))


@router.post("/domain-adaptation/drafts/{draft_id}/delete-preview")
def preview_domain_draft_delete(draft_id: str, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, preview=_services(request).domain_profiles.delete_draft_preview(draft_id))


@router.delete("/domain-adaptation/drafts/{draft_id}")
def delete_domain_draft(
    draft_id: str,
    request: Request,
    acknowledged: bool = False,
    plan_id: str = "",
    preview_token: str = "",
) -> dict[str, Any]:
    _require_local_request(request)
    if not acknowledged:
        raise ApiError("CONFIRMATION_REQUIRED", "请先勾选“我已核对删除预览”。", 422)
    return _success(request, result=_services(request).domain_profiles.delete_draft(draft_id, plan_id=plan_id, preview_token=preview_token))


@router.get("/domain-adaptation/feedback-candidates")
def domain_feedback_candidates(request: Request) -> dict[str, Any]:
    _require_local_request(request)
    service = _services(request).feedback_candidates
    candidates = service.list_available_reviewed_samples(include_internal_paths=False)
    return _success(
        request,
        candidates=candidates,
        available_count=len(candidates),
        feedback_revision=int(service.revision().get("revision") or 0),
        source="feedback_correction_pool",
    )


@router.get("/feedback/correction-pool/available")
def feedback_correction_pool_available(
    request: Request,
    memory_bank_id: str | None = None,
    source_profile_id: str | None = None,
    source_target_domain_id: str | None = None,
) -> dict[str, Any]:
    """Expose the same reusable-sample truth used by target-domain builds."""
    service = _services(request).feedback_candidates
    candidates = service.list_available_reviewed_samples(
        memory_bank_id=memory_bank_id,
        source_profile_id=source_profile_id,
        source_target_domain_id=source_target_domain_id,
        include_internal_paths=False,
    )
    return _success(
        request,
        candidates=candidates,
        available_count=len(candidates),
        feedback_revision=int(service.revision().get("revision") or 0),
        source="feedback_correction_pool",
    )


@router.post("/domain-adaptation/scan")
def scan_domain_samples(payload: DomainScanRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    scan = _services(request).domain_profiles.scan_domain_samples(_domain_payload_with_feedback(request, payload))
    return _success(request, scan=scan)


@router.post("/domain-adaptation/scan-samples")
def scan_domain_samples_alias(payload: DomainScanRequest, request: Request) -> dict[str, Any]:
    """Stable, explicit alias used by the target-domain resource workflow."""
    _require_local_request(request)
    scan = _services(request).domain_profiles.scan_domain_samples(_domain_payload_with_feedback(request, payload))
    return _success(request, scan=scan)


@router.post("/domain-adaptation/check")
def check_domain_build(payload: DomainBuildRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    conditions = _services(request).domain_profiles.build_conditions(_domain_payload_with_feedback(request, payload))
    return _success(request, conditions=conditions)


@router.post("/domain-adaptation/build-readiness")
def domain_build_readiness(payload: DomainBuildRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    conditions = _services(request).domain_profiles.build_conditions(_domain_payload_with_feedback(request, payload))
    return _success(request, build_readiness=conditions, conditions=conditions)


@router.post("/domain-adaptation/video-preview")
def preview_domain_videos(payload: DomainScanRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    preview = _services(request).domain_profiles.preview_videos(_domain_payload_with_feedback(request, payload))
    return _success(request, preview=preview)


@router.post("/domain-adaptation/jobs")
def start_domain_adaptation(payload: DomainBuildRequest, request: Request) -> dict[str, Any]:
    api_received_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    _require_local_request(request)
    # Persist and queue the task before resolving feedback snapshots or touching
    # any image/model resource.  Those operations belong to the worker's
    # visible preparation stage.
    task = _services(request).domain_profiles.start_build(
        payload.model_dump(),
        api_received_at=api_received_at,
    )
    return _success(request, domain_job=task)


@router.post("/domain-adaptation/build")
def start_domain_adaptation_alias(payload: DomainBuildRequest, request: Request) -> dict[str, Any]:
    """Stable creation alias; uses the same idempotent worker as /jobs."""
    api_received_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    _require_local_request(request)
    task = _services(request).domain_profiles.start_build(
        payload.model_dump(),
        api_received_at=api_received_at,
    )
    return _success(request, domain_job=task)


@router.get("/domain-adaptation/jobs/current")
def current_domain_adaptation(request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, domain_job=_services(request).domain_profiles.current_task())


@router.get("/domain-adaptation/jobs/{job_id}")
def domain_adaptation_status(job_id: str, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    if len(job_id) != 32 or any(character not in "0123456789abcdef" for character in job_id.lower()):
        raise ApiError("INVALID_DOMAIN_JOB_ID", "目标域适配任务标识无效。", 422)
    return _success(request, domain_job=_services(request).domain_profiles.task_status(job_id))


@router.post("/domain-adaptation/jobs/{job_id}/cancel")
def cancel_domain_adaptation(job_id: str, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, domain_job=_services(request).domain_profiles.cancel(job_id))


@router.post("/domain-adaptation/jobs/{job_id}/rerun")
def rerun_domain_adaptation(job_id: str, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, domain_job=_services(request).domain_profiles.rerun(job_id))


@router.get("/domain-adaptation/candidates/{profile_id}")
def domain_candidate_details(profile_id: str, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, candidate=_services(request).domain_profiles.candidate_details(profile_id))


@router.post("/domain-adaptation/profiles/{profile_id}/parameter-version")
def create_domain_parameter_version(
    profile_id: str,
    payload: DomainParameterVersionRequest,
    request: Request,
) -> dict[str, Any]:
    _require_local_request(request)
    result = _services(request).domain_profiles.create_parameter_candidate(
        profile_id,
        payload.model_dump(exclude_none=True),
    )
    return _success(request, candidate=result)


@router.post("/domain-adaptation/candidates/{profile_id}/validate")
def validate_domain_candidate(profile_id: str, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, candidate=_services(request).domain_profiles.validate_candidate(profile_id))


@router.post("/domain-adaptation/candidates/{profile_id}/freeze")
def freeze_domain_candidate(profile_id: str, payload: DomainCandidateActionRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    result = _services(request).domain_profiles.freeze_candidate(profile_id, confirm=payload.confirm)
    _services(request).runtime_resources.scan(force=True)
    return _success(request, candidate=result)


@router.post("/domain-adaptation/candidates/{profile_id}/apply")
def apply_domain_candidate(profile_id: str, payload: DomainCandidateActionRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    if not payload.confirm:
        raise ApiError("CONFIRMATION_REQUIRED", "应用目标域版本需要明确确认。", 409)
    services = _services(request)
    resource = next(
        (item for item in services.runtime_resources.list_resources(resource_type="target_profile", include_archived=False) if str(item.get("profile_id")) == profile_id),
        None,
    )
    # A completed target-domain build registers its frozen profile before it
    # reports success.  Use that indexed fact on the first click; only repair a
    # genuinely missing index entry with one forced scan.
    if resource is None:
        services.runtime_resources.scan(force=True)
        resource = next(
            (
                item
                for item in services.runtime_resources.list_resources(
                    resource_type="target_profile", include_archived=False
                )
                if str(item.get("profile_id")) == profile_id
            ),
            None,
        )
    if resource is None:
        raise ApiError("DOMAIN_PROFILE_NOT_DISCOVERED", "冻结 profile 尚未进入资源索引。", 409)
    return _success(request, result=services.runtime_profiles.apply_profile(str(resource["resource_id"]), payload.actor))


@router.get("/memory-banks")
def list_memory_banks(
    request: Request,
    dataset: str | None = None,
    category: str | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    services = _services(request)
    if dataset and category:
        profiles = services.domain_profiles.list_profiles(dataset, category, model_id)
    else:
        profiles = services.domain_profiles.list_all_profiles()
    return _success(request, memory_banks=profiles)


@router.get("/config-profiles")
def list_config_profiles(request: Request, dataset: str | None = None, category: str | None = None) -> dict[str, Any]:
    _require_local_request(request)
    profiles = _services(request).config_profiles.list_profiles(dataset, category)
    return _success(
        request,
        config_profiles=[item for item in profiles if str(item.get("memory_bank_id") or item.get("target_domain_version") or "")],
    )


@router.post("/config-profiles")
def create_config_profile(payload: ConfigProfileCreateRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    raise ApiError(
        "CONFIG_MUST_BELONG_TO_MEMORY_BANK",
        "配置必须在目标域记忆库内部创建，请使用运行配置中的“新增配置”。",
        status_code=409,
    )


@router.post("/config-profiles/import")
def import_config_profile(payload: ConfigProfileImportRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    raise ApiError(
        "CONFIG_MUST_BELONG_TO_MEMORY_BANK",
        "配置不能作为独立一级资源导入，请先选择目标域记忆库。",
        status_code=409,
    )


@router.get("/config-profiles/{profile_id}/versions/{version}")
def get_config_profile(profile_id: str, version: int, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, config_profile=_services(request).config_profiles.get(profile_id, version))


@router.get("/config-profiles/{profile_id}/versions/{version}/export")
def export_config_profile(profile_id: str, version: int, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    profile = _services(request).config_profiles.get(profile_id, version)
    return _success(
        request,
        config_profile={key: value for key, value in profile.items() if key not in {"config"}},
        yaml_text=profile.get("yaml_text") or "",
    )


@router.put("/config-profiles/{profile_id}/versions/{version}")
def update_config_profile(profile_id: str, version: int, payload: ConfigProfileUpdateRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    raise ApiError(
        "USE_MEMORY_BANK_CONFIG_VERSIONING",
        "配置版本必须在所属记忆库内修订，冻结配置不会被原地覆盖。",
        status_code=409,
    )


@router.post("/config-profiles/{profile_id}/versions/{version}/copy")
def copy_config_profile(profile_id: str, version: int, payload: ConfigProfileCopyRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    raise ApiError(
        "USE_MEMORY_BANK_CONFIG_VERSIONING",
        "复制配置时必须指定所属记忆库，请从记忆库详情执行。",
        status_code=409,
    )


@router.post("/config-profiles/{profile_id}/versions/{version}/validate")
def validate_config_profile(profile_id: str, version: int, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, config_profile=_services(request).config_profiles.validate(profile_id, version))


@router.post("/config-profiles/{profile_id}/versions/{version}/archive")
def archive_config_profile(profile_id: str, version: int, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, config_profile=_services(request).config_profiles.archive_draft(profile_id, version))


@router.get("/memory-banks/{memory_bank_id}")
def memory_bank_detail(
    memory_bank_id: str,
    request: Request,
    dataset: str,
    category: str,
    model_id: str | None = None,
) -> dict[str, Any]:
    profile = _services(request).domain_profiles.get_profile(memory_bank_id, dataset, category, model_id)
    return _success(request, memory_bank=profile)


@router.get("/memory-banks/{memory_bank_id}/configs")
def memory_bank_configs(memory_bank_id: str, request: Request, include_archived: bool = False) -> dict[str, Any]:
    _require_local_request(request)
    return _success(
        request,
        **_services(request).runtime_resources.memory_bank_configs(memory_bank_id, include_archived=include_archived),
    )


@router.post("/memory-banks/{memory_bank_id}/configs")
def create_memory_bank_config(memory_bank_id: str, payload: MemoryBankConfigCreateRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(
        request,
        result=_services(request).runtime_resources.create_memory_bank_config(memory_bank_id, payload.model_dump(exclude_none=True)),
    )


@router.post("/memory-banks/{memory_bank_id}/configs/{profile_id}/versions/{version}/revise")
def revise_memory_bank_config(memory_bank_id: str, profile_id: str, version: int, payload: MemoryBankConfigUpdateRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(
        request,
        result=_services(request).runtime_resources.revise_memory_bank_config(
            memory_bank_id, profile_id, version, payload.model_dump(exclude_none=True)
        ),
    )


@router.post("/memory-banks/{memory_bank_id}/configs/{profile_id}/versions/{version}/delete")
def delete_memory_bank_config(memory_bank_id: str, profile_id: str, version: int, payload: MemoryBankConfigDeleteRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    if not payload.acknowledged:
        raise ApiError("CONFIRMATION_REQUIRED", "请先勾选“我已核对删除预览”。", 422)
    return _success(
        request,
        result=_services(request).runtime_resources.delete_memory_bank_config(
            memory_bank_id,
            profile_id,
            version,
            acknowledged=payload.acknowledged,
        ),
    )


@router.post("/memory-banks/{memory_bank_id}/activate")
def activate_memory_bank(memory_bank_id: str, payload: MemoryBankActivateRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    services = _services(request)
    profile = next(
        (item for item in services.runtime_profiles.list_profiles() if str(item.get("profile_id") or "") == str(memory_bank_id)),
        None,
    )
    if profile is None:
        raise ApiError("RUNTIME_PROFILE_MISSING", "所选目标域 Profile 不存在或已归档。", 404)
    if str(profile.get("dataset") or "").lower() != str(payload.dataset or "").lower() or str(profile.get("category") or "") != str(payload.category or ""):
        raise ApiError("RUNTIME_PROFILE_BINDING_MISMATCH", "所选记忆库与数据集或类别不一致。", 409)
    result = services.runtime_profiles.apply_profile(str(profile["resource_id"]), actor="memory_bank_activate")
    return _success(request, memory_bank=result.get("current", {}).get("resource"), result=result)


@router.post("/memory-banks/{memory_bank_id}/rename")
def rename_memory_bank(memory_bank_id: str, payload: MemoryBankRenameRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, memory_bank=_services(request).domain_profiles.rename(memory_bank_id, payload.friendly_name))


@router.post("/memory-banks/{memory_bank_id}/archive")
def archive_memory_bank(memory_bank_id: str, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, memory_bank=_services(request).domain_profiles.archive(memory_bank_id))


# ---------------------------------------------------------------------------
# Unified runtime resource registry.  These endpoints are intentionally
# additive; the existing config-profile and domain-profile APIs remain the
# authoritative writers for their on-disk formats.
# ---------------------------------------------------------------------------
@router.get("/runtime/resources")
def runtime_resources(
    request: Request,
    search: str = "",
    resource_type: str = "",
    dataset: str = "",
    category: str = "",
    status: str = "",
    include_archived: bool = False,
    include_bound: bool = True,
    include_frozen: bool = True,
    include_deleted: bool = False,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=500, ge=1, le=500),
) -> dict[str, Any]:
    _require_local_request(request)
    service = _services(request).runtime_resources
    filters = dict(
        search=search,
        resource_type=resource_type,
        dataset=dataset,
        category=category,
        status=status,
        include_archived=include_archived,
        include_bound=include_bound,
        include_frozen=include_frozen,
    )
    start = (page - 1) * page_size
    post_filter_status = status in {"current", "available", "frozen", "profile_bound", "template", "draft", "archived", "incomplete"}
    if post_filter_status or not include_bound or not include_frozen:
        resources = service.list_resources(**filters)
        total = len(resources)
        page_resources = resources[start:start + page_size]
    else:
        total = service.count_resources(**filters)
        page_resources = service.list_resources(**filters, limit=page_size, offset=start)
    deletion_records = service.tombstones(page_size) if include_deleted else []
    return _success(
        request,
        resources=page_resources,
        total=total,
        page=page,
        page_size=page_size,
        deletion_records=deletion_records,
        summary=service.maintenance_summary(scan=False),
    )


@router.get("/runtime/resources/revision")
def runtime_resource_revision(request: Request, since: int | None = None) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, **_services(request).runtime_resources.catalog_revision(since))


@router.get("/runtime/resources/tombstones")
def runtime_resource_tombstones(request: Request, limit: int = Query(default=200, ge=1, le=500)) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, tombstones=_services(request).runtime_resources.tombstones(limit))


@router.get("/runtime/resources/deletion-records")
def runtime_resource_deletion_records(request: Request, limit: int = Query(default=200, ge=1, le=500)) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, deletion_records=_services(request).runtime_resources.tombstones(limit))


@router.get("/runtime/resources/{resource_id}")
def runtime_resource_detail(resource_id: str, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, resource=_services(request).runtime_resources.get_resource(resource_id))


@router.get("/runtime/resources/{resource_id}/usage")
def runtime_resource_usage(resource_id: str, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, usage=_services(request).runtime_resources.usage(resource_id))


@router.post("/runtime/resources/scan")
def runtime_resource_scan(request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, scan=_services(request).runtime_resources.scan(force=True))


@router.post("/runtime/resources/validate")
def runtime_resource_validate_all(request: Request) -> dict[str, Any]:
    _require_local_request(request)
    resources = _services(request).runtime_resources.list_resources(include_archived=False)
    validated = [_services(request).runtime_resources.validate_resource(str(item["resource_id"])) for item in resources]
    return _success(request, resources=validated, summary=_services(request).runtime_resources.maintenance_summary(scan=False))


@router.post("/runtime/resources/{resource_id}/validate")
def runtime_resource_validate(resource_id: str, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, resource=_services(request).runtime_resources.validate_resource(resource_id))


@router.post("/runtime/resources/{resource_id}/archive")
def runtime_resource_archive(resource_id: str, payload: RuntimeResourceActionRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    if not payload.confirm:
        raise ApiError("CONFIRMATION_REQUIRED", "归档资源前请勾选确认。", 422)
    return _success(request, resource=_services(request).runtime_resources.archive(resource_id, payload.actor, current_profile_action=payload.current_profile_action, replacement_resource_id=payload.replacement_resource_id))


@router.post("/runtime/resources/{resource_id}/restore")
def runtime_resource_restore(resource_id: str, payload: RuntimeResourceActionRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    if not payload.confirm:
        raise ApiError("CONFIRMATION_REQUIRED", "恢复归档前请勾选确认。", 422)
    return _success(request, resource=_services(request).runtime_resources.restore(resource_id, payload.actor))


@router.post("/runtime/resources/{resource_id}/delete-preview")
def runtime_resource_delete_preview(resource_id: str, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, preview=_services(request).runtime_resources.delete_preview(resource_id))


@router.post("/runtime/resources/{resource_id}/delete")
def runtime_resource_delete(resource_id: str, payload: RuntimeResourceActionRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    if not payload.acknowledged:
        raise ApiError("CONFIRMATION_REQUIRED", "请先勾选“我已核对删除预览”。", 422)
    return _success(request, result=_services(request).runtime_resources.delete(
        resource_id,
        preview_token=payload.preview_token,
        actor=payload.actor,
        acknowledged=payload.acknowledged,
        delete_artifacts=payload.delete_artifacts,
        reason=payload.reason,
        current_profile_action=payload.current_profile_action,
        replacement_resource_id=payload.replacement_resource_id,
    ))


@router.post("/runtime/resources/batch-archive-preview")
def runtime_resource_batch_archive_preview(payload: RuntimeBatchActionRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, preview=_services(request).runtime_resources.batch_archive_preview(payload.resource_ids))


@router.post("/runtime/resources/batch-archive")
def runtime_resource_batch_archive(payload: RuntimeBatchActionRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    if not payload.confirm:
        raise ApiError("CONFIRMATION_REQUIRED", "请先勾选操作确认后再归档。", 422)
    return _success(request, result=_services(request).runtime_resources.batch_archive(payload.resource_ids, actor=payload.actor, current_profile_action=payload.current_profile_action, replacement_resource_id=payload.replacement_resource_id))


@router.post("/runtime/resources/batch-restore")
def runtime_resource_batch_restore(payload: RuntimeBatchActionRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    if not payload.confirm:
        raise ApiError("CONFIRMATION_REQUIRED", "请先勾选操作确认后再恢复。", 422)
    return _success(request, result=_services(request).runtime_resources.batch_restore(payload.resource_ids, actor=payload.actor))


@router.post("/runtime/resources/batch-delete-preview")
def runtime_resource_batch_delete_preview(payload: RuntimeBatchActionRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, preview=_services(request).runtime_resources.batch_delete_preview(payload.resource_ids))


@router.post("/runtime/resources/batch-delete")
def runtime_resource_batch_delete(payload: RuntimeBatchActionRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    if not payload.acknowledged:
        raise ApiError("CONFIRMATION_REQUIRED", "请先勾选“我已核对删除预览”。", 422)
    return _success(request, result=_services(request).runtime_resources.batch_delete(
        payload.resource_ids,
        preview_token=payload.preview_token,
        acknowledged=payload.acknowledged,
        actor=payload.actor,
        reason=payload.reason,
        current_profile_action=payload.current_profile_action,
        replacement_resource_id=payload.replacement_resource_id,
    ))


@router.get("/runtime/target-domains/cleanup-summary")
def runtime_target_domain_cleanup_summary(request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, summary=_services(request).runtime_resources.cleanup_summary())


@router.post("/runtime/target-domains/batch-archive-preview")
def runtime_target_domain_batch_archive_preview(payload: RuntimeBatchActionRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, preview=_services(request).runtime_resources.batch_archive_preview(payload.resource_ids))


@router.post("/runtime/target-domains/batch-archive")
def runtime_target_domain_batch_archive(payload: RuntimeBatchActionRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    if not payload.confirm:
        raise ApiError("CONFIRMATION_REQUIRED", "请先勾选操作确认后再归档目标域版本。", 422)
    return _success(request, result=_services(request).runtime_resources.batch_archive(payload.resource_ids, actor=payload.actor, current_profile_action=payload.current_profile_action, replacement_resource_id=payload.replacement_resource_id))


@router.post("/runtime/target-domains/batch-restore")
def runtime_target_domain_batch_restore(payload: RuntimeBatchActionRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    if not payload.confirm:
        raise ApiError("CONFIRMATION_REQUIRED", "请先勾选操作确认后再恢复目标域版本。", 422)
    return _success(request, result=_services(request).runtime_resources.batch_restore(payload.resource_ids, actor=payload.actor))


@router.post("/runtime/target-domains/batch-delete-preview")
def runtime_target_domain_batch_delete_preview(payload: RuntimeBatchActionRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, preview=_services(request).runtime_resources.batch_delete_preview(payload.resource_ids))


@router.post("/runtime/target-domains/batch-delete")
def runtime_target_domain_batch_delete(payload: RuntimeBatchActionRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    if not payload.acknowledged:
        raise ApiError("CONFIRMATION_REQUIRED", "请先勾选“我已核对删除预览”。", 422)
    return _success(request, result=_services(request).runtime_resources.batch_delete(
        payload.resource_ids,
        preview_token=payload.preview_token,
        acknowledged=payload.acknowledged,
        actor=payload.actor,
        reason=payload.reason,
        current_profile_action=payload.current_profile_action,
        replacement_resource_id=payload.replacement_resource_id,
    ))


@router.post("/runtime/target-domains/clear-all-preview")
def runtime_target_domain_clear_all_preview(request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, preview=_services(request).runtime_resources.clear_all_preview())


@router.post("/runtime/target-domains/clear-all")
def runtime_target_domain_clear_all(payload: RuntimeClearAllRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    if not payload.acknowledged:
        raise ApiError("CONFIRMATION_REQUIRED", "请先勾选“我已核对删除预览”。", 422)
    result = _services(request).runtime_resources.clear_all_target_domain_resources(
        plan_id=payload.plan_id,
        preview_token=payload.preview_token,
        acknowledged=payload.acknowledged,
        actor=payload.actor,
    )
    warmup = getattr(request.app.state, "model_warmup", None)
    if warmup is not None:
        warmup.restart()
    return _success(request, result=result)


@router.post("/runtime/configurations/batch-archive-preview")
def runtime_configuration_batch_archive_preview(payload: RuntimeConfigurationBatchRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, preview=_services(request).runtime_resources.batch_archive_preview(payload.configuration_ids, resource_type="configuration"))


@router.post("/runtime/configurations/batch-archive")
def runtime_configuration_batch_archive(payload: RuntimeConfigurationBatchRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    if not payload.confirm:
        raise ApiError("CONFIRMATION_REQUIRED", "请先勾选操作确认后再归档配置。", 422)
    return _success(request, result=_services(request).runtime_resources.batch_archive(payload.configuration_ids, actor=payload.actor, current_profile_action=payload.current_profile_action, replacement_resource_id=payload.replacement_resource_id, resource_type="configuration"))


@router.post("/runtime/configurations/batch-restore")
def runtime_configuration_batch_restore(payload: RuntimeConfigurationBatchRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    if not payload.confirm:
        raise ApiError("CONFIRMATION_REQUIRED", "请先勾选操作确认后再恢复配置。", 422)
    return _success(request, result=_services(request).runtime_resources.batch_restore(payload.configuration_ids, actor=payload.actor, resource_type="configuration"))


@router.post("/runtime/configurations/batch-delete-preview")
def runtime_configuration_batch_delete_preview(payload: RuntimeConfigurationBatchRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, preview=_services(request).runtime_resources.batch_delete_preview(payload.configuration_ids, resource_type="configuration"))


@router.post("/runtime/configurations/batch-delete")
def runtime_configuration_batch_delete(payload: RuntimeConfigurationBatchRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    if not payload.acknowledged:
        raise ApiError("CONFIRMATION_REQUIRED", "请先勾选“我已核对删除预览”。", 422)
    return _success(request, result=_services(request).runtime_resources.batch_delete(
        payload.configuration_ids,
        preview_token=payload.preview_token,
        acknowledged=payload.acknowledged,
        actor=payload.actor,
        reason=payload.reason,
        current_profile_action=payload.current_profile_action,
        replacement_resource_id=payload.replacement_resource_id,
        resource_type="configuration",
    ))


@router.get("/runtime/resources/{resource_id}/dependencies")
def runtime_resource_dependencies(resource_id: str, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, dependencies=_services(request).runtime_resources.dependencies(resource_id))


@router.get("/runtime/resources/{resource_id}/references")
def runtime_resource_references(resource_id: str, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, references=_services(request).runtime_resources.references(resource_id))


@router.get("/runtime/profiles")
def runtime_profiles(request: Request) -> dict[str, Any]:
    _require_local_request(request)
    runtime_profiles_service = _services(request).runtime_profiles
    context = runtime_profiles_service.context_snapshot()
    current = runtime_profiles_service.current_profile(context)
    profiles = runtime_profiles_service.list_profiles()
    return _success(request, profiles=profiles, current=current)


@router.get("/runtime/context")
def runtime_context(request: Request) -> dict[str, Any]:
    """Return the compact authoritative binding used for new inference work."""

    _require_local_request(request)
    startup_snapshot = request.app.state.startup.status()
    # Keep the binding and lifecycle fields from the same immutable startup
    # snapshot.  A second live context read here could otherwise combine a
    # new profile revision with the previous worker/readiness state during a
    # runtime switch.
    runtime_context_snapshot = startup_snapshot.get("runtime_context")
    if not isinstance(runtime_context_snapshot, dict):
        runtime_context_snapshot = _services(request).runtime_profiles.context_snapshot()
    return _success(
        request,
        runtime_context=runtime_context_snapshot,
        **_runtime_worker_fields(startup_snapshot),
    )


@router.get("/runtime/profiles/{resource_id}")
def runtime_profile_detail(resource_id: str, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, profile=_services(request).runtime_resources.get_resource(resource_id))


@router.post("/runtime/profiles/{resource_id}/validate")
def runtime_profile_validate(resource_id: str, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, validation=_services(request).runtime_resources.validate_profile(resource_id))


@router.post("/runtime/profiles/apply")
def runtime_profile_apply(payload: RuntimeProfileApplyRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(
        request,
        result=_services(request).runtime_profiles.apply_profile(
            payload.resource_id,
            payload.actor,
            config_profile_id=payload.config_profile_id,
            config_profile_version=payload.config_profile_version,
        ),
    )


@router.post("/runtime/profiles/deactivate")
def runtime_profile_deactivate(payload: RuntimeResourceActionRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    if not payload.confirm:
        raise ApiError("CONFIRMATION_REQUIRED", "停用当前运行方案前请勾选确认。", 422)
    return _success(request, result=_services(request).runtime_profiles.deactivate(payload.actor))


@router.post("/runtime/profiles/rollback")
def runtime_profile_rollback(payload: RuntimeResourceActionRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    if not payload.confirm:
        raise ApiError("CONFIRMATION_REQUIRED", "回滚运行方案前请勾选确认。", 422)
    return _success(request, result=_services(request).runtime_profiles.rollback(payload.actor))


@router.get("/runtime/maintenance/summary")
def runtime_maintenance_summary(request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, summary=_services(request).runtime_resources.maintenance_summary(scan=False))


@router.post("/runtime/maintenance/rebuild-index")
def runtime_maintenance_rebuild(request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, result=_services(request).runtime_resources.rebuild_index())


@router.post("/runtime/maintenance/find-orphans")
def runtime_maintenance_orphans(request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, result=_services(request).runtime_resources.find_orphans())


@router.post("/runtime/maintenance/cleanup-preview")
def runtime_maintenance_cleanup_preview(request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, result=_services(request).runtime_resources.cleanup_preview())


@router.post("/runtime/maintenance/cleanup")
def runtime_maintenance_cleanup(payload: RuntimeCleanupRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    if not payload.confirm:
        raise ApiError("CONFIRMATION_REQUIRED", "清理资源前请勾选确认。", 422)
    return _success(request, result=_services(request).runtime_resources.cleanup(payload.resource_ids, payload.actor))


@router.post("/runtime/maintenance/backup-index")
def runtime_maintenance_backup_index(payload: RuntimeResourceActionRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    if not payload.confirm:
        raise ApiError("CONFIRMATION_REQUIRED", "备份资源索引前请勾选确认。", 422)
    return _success(request, result=_services(request).runtime_resources.backup_index(payload.actor))


@router.post("/runtime/maintenance/restore-default")
def runtime_maintenance_restore_default(payload: RuntimeResourceActionRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    if not payload.confirm:
        raise ApiError("CONFIRMATION_REQUIRED", "恢复默认方案前请勾选确认。", 422)
    return _success(request, result=_services(request).runtime_profiles.restore_default(payload.actor))


@router.get("/runtime/maintenance/audit")
def runtime_maintenance_audit(request: Request, limit: int = 50) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, entries=_services(request).runtime_resources.operation_history(limit=limit))


@router.get("/runtime/maintenance/resource-trash")
def runtime_maintenance_pending_cleanup(request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, cleanup=_services(request).runtime_resources.pending_cleanup())


@router.post("/runtime/maintenance/resource-trash/{operation_id}/cleanup")
def runtime_maintenance_retry_cleanup(operation_id: str, payload: RuntimeResourceActionRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    if not payload.confirm:
        raise ApiError("CONFIRMATION_REQUIRED", "清理隔离目录前请勾选确认。", 422)
    return _success(request, result=_services(request).runtime_resources.retry_cleanup(operation_id, payload.actor))


@router.post("/video/uploads")
async def upload_video_for_inspection(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    """Stage and decode-check a selected local video before enabling Start."""
    services = _services(request)
    reservation: dict[str, str] | None = None
    try:
        reservation = services.video_jobs.reserve_upload(file.filename or "upload.mp4")
        streamed = await stream_upload_to_path(
            file,
            reservation["source_path"],
            max_bytes=500 * 1024 * 1024,
        )
        upload = services.video_jobs.inspect_staged_upload(
            reservation["upload_id"],
            streamed.path,
            size_bytes=streamed.size_bytes,
            source_sha256=streamed.sha256,
        )
        return _success(request, upload=upload)
    except ApiError:
        raise
    except Exception as exc:
        raise ApiError("VIDEO_UPLOAD_FAILED", f"视频选择或信息读取失败：{type(exc).__name__}: {exc}", 500) from exc
    finally:
        if reservation is not None:
            manifest = services.video_jobs.upload_root / reservation["upload_id"] / "upload_manifest.json"
            if not manifest.exists():
                services.video_jobs.discard_upload(reservation["upload_id"])


@router.get("/video/uploads/{upload_id}")
def video_upload(upload_id: str, request: Request) -> dict[str, Any]:
    return _success(request, upload=_services(request).video_jobs.public_upload(upload_id))


@router.get("/video/uploads/{upload_id}/content", response_class=FileResponse)
def video_upload_content(upload_id: str, request: Request) -> FileResponse:
    path, media_type = _services(request).video_jobs.upload_content(upload_id)
    return FileResponse(path, media_type=media_type)


@router.post("/video/jobs")
def start_video_job(payload: VideoJobCreateRequest, request: Request) -> dict[str, Any]:
    services = _services(request)
    binding = _validate_inference_binding(
        services,
        profile_id=payload.profile_id,
        runtime_revision=payload.runtime_revision,
        memory_bank_id=payload.memory_bank_id,
        model_id=payload.model_id,
        dataset=payload.dataset,
        category=payload.category,
        config_profile_id=payload.config_profile_id,
        config_profile_version=payload.config_profile_version,
    )
    options = payload.model_dump(exclude={"upload_id"})
    options.update(binding)
    task = services.video_jobs.start(payload.upload_id, options, _request_id(request))
    return _success(request, video_job=task)


@router.get("/video/jobs/{job_id}")
def video_job_status(job_id: str, request: Request, heartbeat: bool = False) -> dict[str, Any]:
    return _success(request, video_job=_services(request).video_jobs.status(job_id, heartbeat=heartbeat))


@router.get("/video/jobs/{job_id}/feedback-interval")
def video_feedback_interval(
    job_id: str,
    request: Request,
    start_ms: float = Query(default=0.0, ge=0.0),
    end_ms: float = Query(default=0.0, ge=0.0),
    max_frames: int = Query(default=8, ge=1, le=32),
) -> dict[str, Any]:
    """Select auditable representative frames for an operator interval.

    The route returns frame metadata only.  The subsequent normal feedback
    endpoint still validates each selected frame before creating a candidate.
    """
    if end_ms < start_ms:
        raise ApiError("INVALID_VIDEO_INTERVAL", "视频反馈时间区间无效。", 422)
    frames = _services(request).video_jobs.feedback_interval_contexts(job_id, start_ms, end_ms, max_frames)
    public_frames = []
    for frame in frames:
        item = dict(frame)
        item.pop("_source_path", None)
        item.pop("source_path", None)
        public_frames.append(item)
    return _success(request, frames=public_frames)


@router.post("/video/jobs/{job_id}/pause")
def pause_video_job(job_id: str, request: Request) -> dict[str, Any]:
    return _success(request, video_job=_services(request).video_jobs.pause(job_id))


@router.post("/video/jobs/{job_id}/resume")
def resume_video_job(job_id: str, request: Request) -> dict[str, Any]:
    return _success(request, video_job=_services(request).video_jobs.resume(job_id))


@router.post("/video/jobs/{job_id}/stop")
def stop_video_job(job_id: str, request: Request) -> dict[str, Any]:
    return _success(request, video_job=_services(request).video_jobs.stop(job_id))


@router.post("/video/jobs/{job_id}/cancel")
def cancel_video_job(job_id: str, request: Request) -> dict[str, Any]:
    return _success(request, video_job=_services(request).video_jobs.cancel(job_id))


@router.post("/video/jobs/{job_id}/rerun")
def rerun_video_job(job_id: str, request: Request) -> dict[str, Any]:
    return _success(request, video_job=_services(request).video_jobs.rerun(job_id, _request_id(request)))


@router.post("/video/jobs/{job_id}/open-output")
def open_video_output_directory(job_id: str, request: Request) -> dict[str, Any]:
    return _success(request, video_job=_services(request).video_jobs.open_result_directory(job_id))


@router.post("/video/jobs/{job_id}/open-video")
def open_video_result(job_id: str, request: Request) -> dict[str, Any]:
    return _success(request, video_job=_services(request).video_jobs.open_result_video(job_id))


@router.get("/health/live")
async def health_live(request: Request) -> dict[str, Any]:
    """Process liveness only; never waits for Profile or model resources."""

    return _success(
        request,
        status="live",
        alive=True,
        service="AdaptiveAOI",
        instance_root=str(request.app.state.startup.settings.project_root.resolve()),
    )


@router.get("/health/ready")
async def health_ready(request: Request) -> dict[str, Any]:
    """Lock-free readiness snapshot produced by the background warm-up task."""

    preload = request.app.state.startup.status()
    state = str(preload.get("state") or "loading")
    stage = str(preload.get("current_stage") or "正在准备检测资源")
    message = (
        "检测资源已就绪。"
        if state == "ready"
        else (f"检测资源加载失败：{preload.get('error') or stage}" if state == "failed" else stage)
    )
    return _success(
        request,
        status=state,
        ready=bool(preload.get("ready")),
        current_stage=stage,
        phase=preload.get("phase"),
        progress=int(preload.get("progress") or 0),
        elapsed_seconds=preload.get("elapsed_seconds"),
        message=message,
        current_profile=preload.get("current_profile"),
        runtime_context=preload.get("runtime_context") or {},
        timings=preload.get("timings") or {},
        error=preload.get("error"),
        api_ready=bool(preload.get("api_ready", True)),
        heartbeat_at=preload.get("heartbeat_at"),
        **_runtime_worker_fields(preload),
    )


@router.post("/health/ready/retry")
async def health_ready_retry(request: Request) -> dict[str, Any]:
    """Retry one failed initializer without creating a duplicate worker."""

    _require_local_request(request)
    startup = request.app.state.startup
    accepted, message = startup.restart()
    return _success(request, accepted=accepted, message=message, readiness=startup.status())


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    """Backward-compatible detailed health built from the readiness snapshot.

    This endpoint deliberately avoids RuntimeProfileService.configuration()
    and RuntimeRegistry.status() while warm-up is running.  Both may touch
    resources protected by the same lock used during model construction.
    """

    services = getattr(request.app.state, "services", None)
    startup = request.app.state.startup
    settings = services.settings if services is not None else startup.settings
    preload = startup.status()
    warmup_state = str(preload.get("state") or "loading").lower()
    runtime_ready = bool(preload.get("runtime_ready", preload.get("ready")))
    runtime_config = dict(preload.get("runtime") or {})
    current_profile_data = preload.get("current_profile")
    current_profile = dict(current_profile_data) if current_profile_data else None
    if current_profile is not None:
        current_profile["status"] = warmup_state
    current_profile_values = current_profile or {}
    model_id = str(runtime_config.get("model_id") or "")
    dataset = str(runtime_config.get("dataset") or current_profile_values.get("dataset") or "")
    category = str(runtime_config.get("category") or current_profile_values.get("category") or "")
    model_loaded = runtime_ready
    model = {
        "model_id": model_id,
        "model_type": "patchcore",
        "backbone": None,
        "profile_valid": bool(current_profile_values.get("profile_id") and current_profile_values.get("frozen")),
        "weights_valid": bool(runtime_config.get("memory_bank_id")),
        "calibration_valid": runtime_config.get("threshold") is not None,
        "loaded": model_loaded,
        "default_threshold": runtime_config.get("threshold"),
        "current_threshold": runtime_config.get("threshold"),
        "dataset": dataset,
        "category": category,
        "memory_bank_id": runtime_config.get("memory_bank_id"),
        "config_profile_id": runtime_config.get("config_profile_id"),
        "config_profile_version": runtime_config.get("config_profile_version"),
        "inference_mode": runtime_config.get("inference_mode"),
        "frozen": bool(runtime_config.get("frozen")),
    }
    model_ready = bool(runtime_ready and model["weights_valid"] and model["profile_valid"])
    memory_configured = bool(runtime_config.get("memory_bank_id"))
    profile_status = (
        "ready"
        if warmup_state == "ready"
        else ("missing" if warmup_state in {"not_ready", "deactivating"} else ("invalid" if warmup_state == "failed" else "loading"))
    )
    runtime_status = services.registry.status() if services is not None and runtime_ready else {"cached_runtime_count": 0, "runtimes": []}
    service_status = (
        "ready"
        if warmup_state == "ready"
        else ("not_ready" if warmup_state in {"not_ready", "deactivating"} else ("degraded" if warmup_state == "failed" else "loading"))
    )
    service_message = (
        "算法服务已连接，但运行资源预热失败。"
        if warmup_state == "failed"
        else (
            "当前未加载目标域记忆库。"
            if warmup_state == "not_ready"
            else (
                "正在停用当前目标域记忆库。"
                if warmup_state == "deactivating"
                else ("算法服务正在后台加载运行资源。" if warmup_state == "loading" else "算法服务已就绪。")
            )
        )
    )
    cuda_ready = bool(model_loaded and str(runtime_config.get("device") or "").lower() == "cuda")
    # CUDA metadata is captured once by the model warm-up owner.  Re-probing
    # the Windows driver from every health request can serialize the API after
    # readiness and make an otherwise healthy service appear hung.
    torch_version = str(runtime_config.get("torch_version") or "")
    cuda_version = str(runtime_config.get("cuda_version") or "")
    gpu_name = runtime_config.get("gpu_name") if cuda_ready else None
    return _success(
        request,
        status=service_status,
        ready=warmup_state == "ready",
        service="AdaptiveAOI",
        api_ready=True,
        database_ready=bool(settings.runtime_root.exists()),
        configuration_ready=bool(settings.config_path.exists()),
        model_service_ready=warmup_state == "ready",
        message=service_message,
        version=str(settings.backend.get("version", "1.0.0")),
        system_time=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        device=str(runtime_config.get("device") or "unknown"),
        cuda_available=cuda_ready,
        torch_version=torch_version,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        current_model=model,
        profile_status=profile_status,
        current_profile=current_profile,
        runtime_context=preload.get("runtime_context") or {},
        model_ready=model_ready,
        model_runtime_ready=runtime_ready and memory_configured,
        model_preload=preload,
        runtime=runtime_status,
        **_runtime_worker_fields(preload),
        api_docs="/docs",
    )


@router.get("/models")
def list_models(request: Request) -> dict[str, Any]:
    return _success(request, models=_services(request).models.list_models())


@router.post("/models/load")
def load_model(payload: ModelLoadRequest, request: Request) -> dict[str, Any]:
    services = _services(request)
    binding = _validate_inference_binding(
        services,
        profile_id=None,
        runtime_revision=None,
        memory_bank_id=payload.memory_bank_id,
        model_id=payload.model_id,
        dataset=payload.dataset,
        category=payload.category,
        config_profile_id=None,
        config_profile_version=None,
    )
    model = services.adapter.load_model(
        binding["model_id"],
        binding["dataset"],
        binding["category"],
        payload.device or binding["device"],
        binding["threshold"],
        binding["memory_bank_id"],
        payload.mode,
    )
    return _success(request, model=model)


@router.post("/detect")
async def detect(
    request: Request,
    file: UploadFile = File(...),
    model_id: str | None = Form(default=None),
    dataset: str | None = Form(default=None),
    category: str | None = Form(default=None),
    device: str | None = Form(default=None),
    threshold: float | None = Form(default=None),
    generate_visualizations: str | None = Form(default=None),
    save_visuals: str | None = Form(default=None),
    memory_bank_id: str | None = Form(default=None),
    config_profile_id: str | None = Form(default=None),
    config_profile_version: int | None = Form(default=None),
    profile_id: str | None = Form(default=None),
    runtime_revision: int | None = Form(default=None),
    mode: str = Form(default="inference_only"),
) -> dict[str, Any]:
    services = _services(request)
    binding = _validate_inference_binding(
        services,
        profile_id=profile_id,
        runtime_revision=runtime_revision,
        memory_bank_id=memory_bank_id,
        model_id=model_id,
        dataset=dataset,
        category=category,
        config_profile_id=config_profile_id,
        config_profile_version=config_profile_version,
    )
    model_id = binding["model_id"]
    dataset = binding["dataset"]
    category = binding["category"]
    memory_bank_id = binding["memory_bank_id"]
    config_profile_id = binding["config_profile_id"]
    config_profile_version = binding["config_profile_version"]
    threshold = threshold if threshold is not None else binding["threshold"]
    device = device or binding["device"]
    job = services.adapter.start_job("single")
    try:
        source = services.adapter.upload_target(job, file.filename or "upload.png")
        streamed = await stream_upload_to_path(file, source, max_bytes=services.settings.max_upload_bytes())
        source = streamed.path
        services.adapter.validate_image(source)
        payload = services.adapter.detect(
            job,
            source,
            _request_id(request),
            model_id=model_id,
            dataset=dataset,
            category=category,
            device=device,
            threshold=threshold,
            save_visuals=_optional_bool(generate_visualizations) if generate_visualizations is not None else _optional_bool(save_visuals),
            memory_bank_id=memory_bank_id,
            config_profile_id=config_profile_id,
            config_profile_version=config_profile_version,
            config_hash=binding["config_hash"],
            config_values=binding["config_values"],
            detection_context=binding["detection_context"],
            mode=mode,
        )
        return _success(request, **payload)
    except ApiError:
        raise
    except Exception as exc:
        services.jobs.fail(job, "DETECT_REQUEST_FAILED", str(exc))
        raise


@router.post("/detect/batch")
async def detect_batch(
    request: Request,
    files: list[UploadFile] | None = File(default=None),
    input_dir: str | None = Form(default=None),
    model_id: str | None = Form(default=None),
    dataset: str | None = Form(default=None),
    category: str | None = Form(default=None),
    device: str | None = Form(default=None),
    threshold: float | None = Form(default=None),
    generate_visualizations: str | None = Form(default=None),
    save_visuals: str | None = Form(default=None),
    memory_bank_id: str | None = Form(default=None),
    config_profile_id: str | None = Form(default=None),
    config_profile_version: int | None = Form(default=None),
    profile_id: str | None = Form(default=None),
    runtime_revision: int | None = Form(default=None),
    mode: str = Form(default="inference_only"),
    async_job: str | None = Form(default=None),
) -> dict[str, Any]:
    services = _services(request)
    binding = _validate_inference_binding(
        services,
        profile_id=profile_id,
        runtime_revision=runtime_revision,
        memory_bank_id=memory_bank_id,
        model_id=model_id,
        dataset=dataset,
        category=category,
        config_profile_id=config_profile_id,
        config_profile_version=config_profile_version,
    )
    model_id = binding["model_id"]
    dataset = binding["dataset"]
    category = binding["category"]
    memory_bank_id = binding["memory_bank_id"]
    config_profile_id = binding["config_profile_id"]
    config_profile_version = binding["config_profile_version"]
    threshold = threshold if threshold is not None else binding["threshold"]
    device = device or binding["device"]
    job = services.adapter.start_job("batch")
    try:
        uploaded = files or []
        if input_dir and uploaded:
            raise ApiError("AMBIGUOUS_BATCH_INPUT", "Provide uploaded files or input_dir, not both.", 422)
        if input_dir:
            _require_local_request(request)
            sources = services.adapter.list_directory_images(input_dir)
        elif uploaded:
            if len(uploaded) > services.settings.max_batch_files():
                raise ApiError("BATCH_TOO_LARGE", "The upload exceeds the configured batch file limit.", 413)
            sources = []
            for item in uploaded:
                target = services.adapter.upload_target(job, item.filename or "upload.png")
                streamed = await stream_upload_to_path(item, target, max_bytes=services.settings.max_upload_bytes())
                services.adapter.validate_image(streamed.path)
                sources.append(streamed.path)
        else:
            raise ApiError("BATCH_INPUT_REQUIRED", "Provide one or more image files or an input directory.", 422)
        request_id = _request_id(request)
        selection = {
            "model_id": model_id,
            "dataset": dataset,
            "category": category,
            "device": device,
            "threshold": threshold,
            "save_visuals": _optional_bool(generate_visualizations) if generate_visualizations is not None else _optional_bool(save_visuals),
            "memory_bank_id": memory_bank_id,
            "config_profile_id": config_profile_id,
            "config_profile_version": config_profile_version,
            "config_hash": binding["config_hash"],
            "config_values": binding["config_values"],
            "detection_context": binding["detection_context"],
            "mode": mode,
        }
        if _optional_bool(async_job):
            services.jobs.update(
                job,
                status="queued",
                stage="preparing",
                progress=0.0,
                processed=0,
                success_count=0,
                normal_count=0,
                anomaly_count=0,
                failure_count=0,
                input_count=len(sources),
                current_file=None,
                request_id=request_id,
            )

            def run_in_background() -> None:
                try:
                    result = services.adapter.detect_batch(job, sources, request_id, **selection)
                    response_path = job.root / "batch_response.json"
                    temporary = response_path.with_suffix(".json.tmp")
                    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
                    temporary.replace(response_path)
                    services.jobs.update(job, response_file=response_path.name)
                except ApiError as exc:
                    services.jobs.fail(job, exc.code, exc.message)
                except Exception as exc:
                    services.logger.exception("async_batch_failed job_id=%s", job.job_id)
                    services.jobs.fail(job, "BATCH_REQUEST_FAILED", str(exc))

            threading.Thread(
                target=run_in_background,
                name=f"AdaptiveAOI-Batch-{job.job_id[-10:]}",
                daemon=True,
            ).start()
            return _success(
                request,
                job_id=job.job_id,
                batch_id=job.job_id,
                status="queued",
                total=len(sources),
            )
        payload = services.adapter.detect_batch(job, sources, request_id, **selection)
        return _success(request, **payload)
    except ApiError:
        raise
    except Exception as exc:
        services.jobs.fail(job, "BATCH_REQUEST_FAILED", str(exc))
        raise


@router.get("/results")
def results(
    request: Request,
    page: int = 1,
    page_size: int = 20,
    model_id: str | None = None,
    prediction: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    filename: str | None = None,
    defect_type: str | None = None,
) -> dict[str, Any]:
    data = _services(request).storage.list_results(page, page_size, model_id, prediction, date_from, date_to, filename, defect_type)
    return _success(request, **data)


@router.get("/results/{result_id}")
def result_detail(result_id: str, request: Request) -> dict[str, Any]:
    result = _services(request).storage.get_result(result_id)
    if result is None:
        raise ApiError("RESULT_NOT_FOUND", "Detection result was not found.", 404)
    return _success(request, result=result)


@router.get("/statistics/summary")
def statistics_summary(request: Request) -> dict[str, Any]:
    return _success(request, summary=_services(request).storage.summary())


@router.get("/statistics/revision")
def statistics_revision(request: Request, since: int | None = None) -> dict[str, Any]:
    revision = _services(request).storage.statistics_revision()
    return _success(
        request,
        statistics_revision=revision,
        changed=since is None or int(since) != revision,
    )


@router.get("/statistics/dashboard")
def statistics_dashboard(
    request: Request,
    module: Literal["single", "batch", "video"] = "single",
    start_time: str | None = None,
    end_time: str | None = None,
    dataset: str | None = None,
    category: str | None = None,
    config: str | None = None,
    device: str | None = None,
    model_version: str | None = None,
    memory_bank_id: str | None = None,
    prediction: Literal["normal", "anomaly"] | None = None,
    task_status: Literal["success", "failed", "cancelled", "running"] | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=5, le=100),
    sort_by: Literal[
        "created_at", "score", "end_to_end_ms", "throughput", "item_count", "anomaly_rate", "full_pipeline_fps"
    ] = "created_at",
    sort_order: Literal["asc", "desc"] = "desc",
) -> dict[str, Any]:
    data = _services(request).statistics.query(
        module=module,
        start_time=start_time,
        end_time=end_time,
        dataset=dataset,
        category=category,
        config=config,
        device=device,
        model_version=model_version,
        memory_bank_id=memory_bank_id,
        prediction=prediction,
        task_status=task_status,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    return _success(request, dashboard=data)


def _history_filter_query(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ApiError("INVALID_HISTORY_FILTER", "检测记录筛选条件格式无效。", 422) from exc
    if not isinstance(payload, dict):
        raise ApiError("INVALID_HISTORY_FILTER", "检测记录筛选条件格式无效。", 422)
    return payload


@router.get("/history/management/summary")
def history_management_summary(request: Request, filters: str | None = None) -> dict[str, Any]:
    return _success(request, summary=_services(request).history_management.summary(_history_filter_query(filters)))


@router.get("/history/management/records")
def history_management_records(
    request: Request,
    filters: str | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=5, le=50),
    sort_by: str = Query(default="created_at", max_length=40),
    sort_order: str = Query(default="desc", max_length=8),
) -> dict[str, Any]:
    return _success(
        request,
        records=_services(request).history_management.records(
            _history_filter_query(filters),
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
        ),
    )


@router.get("/history/management/tasks/{task_id}/items")
def history_management_task_items(
    task_id: str,
    request: Request,
    mode: Literal["batch", "video"] = Query(...),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=5, le=100),
    query: str | None = Query(default=None, max_length=200),
    prediction: str | None = Query(default=None, max_length=24),
    feedback_status: str | None = Query(default=None, max_length=24),
    time_from_ms: float | None = Query(default=None, ge=0.0),
    time_to_ms: float | None = Query(default=None, ge=0.0),
) -> dict[str, Any]:
    if time_from_ms is not None and time_to_ms is not None and time_to_ms < time_from_ms:
        raise ApiError("INVALID_FRAME_TIME_RANGE", "视频帧时间范围无效。", 422)
    items = _services(request).history_management.task_items(
        task_id,
        mode=mode,
        page=page,
        page_size=page_size,
        query=query,
        prediction=prediction,
        feedback_status=feedback_status,
        time_from_ms=time_from_ms,
        time_to_ms=time_to_ms,
    )
    return _success(request, task_items=items)


@router.post("/history/management/tasks/{task_id}/items/delete")
def history_management_delete_task_items(
    task_id: str,
    payload: HistoryTaskItemsDeleteRequest,
    request: Request,
) -> dict[str, Any]:
    _require_local_request(request)
    if not payload.acknowledged:
        raise ApiError("CONFIRMATION_REQUIRED", "请先勾选“我已核对删除预览”。", 422)
    result = _services(request).history_management.delete_task_items(
        task_id,
        payload.item_ids,
        acknowledged=True,
    )
    return _success(request, result=result)


@router.get(
    "/history/management/tasks/{task_id}/frames/{frame_index}/image",
    response_class=FileResponse,
)
def history_management_video_frame_image(task_id: str, frame_index: int, request: Request) -> FileResponse:
    path = _services(request).video_jobs.historical_frame_preview(task_id, frame_index)
    return FileResponse(path, media_type="image/jpeg", filename=path.name)


@router.post("/history/management/delete-preview")
def history_management_delete_preview(payload: HistoryDeletePreviewRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, preview=_services(request).history_management.preview(payload.model_dump()))


@router.post("/history/management/delete")
def history_management_delete(payload: HistoryDeleteRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    if not payload.acknowledged:
        raise ApiError("CONFIRMATION_REQUIRED", "请先勾选“我已核对删除预览”。", 422)
    return _success(request, result=_services(request).history_management.delete(payload.model_dump()))


@router.post("/history/management/delete-all-preview")
def history_management_delete_all_preview(
    payload: HistoryAllDeletePreviewRequest, request: Request
) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, preview=_services(request).history_management.preview(payload.model_dump(), all_clear=True))


@router.post("/history/management/delete-all")
def history_management_delete_all(payload: HistoryAllDeleteRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    if not payload.acknowledged:
        raise ApiError("CONFIRMATION_REQUIRED", "请先勾选“我已核对删除预览”。", 422)
    return _success(request, result=_services(request).history_management.delete(payload.model_dump(), all_clear=True))


@router.get("/history/management/cleanup-jobs")
def history_management_cleanup_jobs(
    request: Request, limit: int = Query(default=20, ge=1, le=100)
) -> dict[str, Any]:
    return _success(request, jobs=_services(request).history_management.cleanup_jobs(limit))


@router.get("/history/management/cleanup-jobs/{job_id}")
def history_management_cleanup_job(job_id: str, request: Request) -> dict[str, Any]:
    return _success(request, job=_services(request).history_management.cleanup_job(job_id))


@router.post("/history/management/cleanup-jobs/{job_id}/cancel")
def history_management_cancel_cleanup_job(job_id: str, request: Request) -> dict[str, Any]:
    return _success(request, job=_services(request).history_management.cancel_cleanup_job(job_id))


@router.get("/history/management/audit")
def history_management_audit(request: Request, limit: int = Query(default=50, ge=1, le=100)) -> dict[str, Any]:
    return _success(request, audit=_services(request).history_management.audit(limit))


@router.post("/history/management/retry-file-cleanup")
def history_management_retry_file_cleanup(
    payload: HistoryCleanupRetryRequest, request: Request
) -> dict[str, Any]:
    return _success(request, result=_services(request).history_management.retry_file_cleanup(payload.cleanup_ids))


@router.get("/experiments")
def experiments(request: Request) -> dict[str, Any]:
    root = _services(request).settings.experiment_root
    if not root.exists():
        return _success(request, experiments=[], warnings=["Experiment result directory does not exist."])
    items: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".json", ".md"}:
            continue
        relative = path.relative_to(root).as_posix()
        items.append(
            {
                "experiment_id": relative.encode("utf-8").hex(),
                "name": path.name,
                "relative_path": relative,
                "type": path.suffix.lower().lstrip("."),
                "size_bytes": path.stat().st_size,
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
            }
        )
        if len(items) >= 300:
            break
    return _success(request, experiments=items, truncated=len(items) >= 300)


@router.get("/experiments/{experiment_id}")
def experiment_detail(experiment_id: str, request: Request) -> dict[str, Any]:
    services = _services(request)
    try:
        relative = bytes.fromhex(experiment_id).decode("utf-8")
    except ValueError as exc:
        raise ApiError("INVALID_EXPERIMENT_ID", "Experiment ID is invalid.", 422) from exc
    root = services.settings.experiment_root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ApiError("INVALID_EXPERIMENT_PATH", "Experiment path is outside the result directory.", 403) from exc
    if not path.exists() or not path.is_file():
        raise ApiError("EXPERIMENT_NOT_FOUND", "Experiment artifact was not found.", 404)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ApiError("EXPERIMENT_READ_FAILED", "Experiment artifact could not be read.", 500) from exc
    preview = text[:100_000]
    payload: dict[str, Any] = {"name": path.name, "relative_path": str(path.relative_to(root)), "type": path.suffix.lower().lstrip("."), "truncated": len(text) > len(preview)}
    if path.suffix.lower() == ".json":
        try:
            payload["content"] = json.loads(preview)
        except json.JSONDecodeError:
            payload["content"] = preview
            payload["warning"] = "JSON could not be parsed; text preview is returned."
    else:
        payload["content"] = preview
    return _success(request, experiment=payload)


@router.get("/config")
def get_config(request: Request) -> dict[str, Any]:
    services = _services(request)
    settings = services.settings
    context = services.runtime_profiles.context_snapshot()
    return _success(
        request,
        config={
            "dynamic": context.get("runtime") or {},
            "defaults": settings.defaults,
            "models": list(settings.models.keys()),
            "backend": {"host": settings.host(), "port": settings.port(), "debug": settings.debug()},
            "supported_image_formats": sorted(settings.supported_suffixes()),
            "runtime": services.runtime_profiles.configuration(context),
            "runtime_context": context,
        },
    )


@router.put("/config")
def update_config(payload: ConfigUpdateRequest, request: Request) -> dict[str, Any]:
    services = _services(request)
    updates = payload.model_dump(exclude_none=True)
    if str(updates.get("inference_mode") or "inference_only") != "inference_only":
        raise ApiError("INFERENCE_ONLY_REQUIRED", "正式测试模式只能使用只读推理。", 409)
    result = services.runtime_profiles.apply_configuration(updates)
    return _success(request, config=result.get("runtime") or services.runtime_profiles.raw_runtime(), runtime=_runtime_configuration(services))


@router.get("/runtime/test-lock")
def runtime_test_lock(request: Request) -> dict[str, Any]:
    _require_local_request(request)
    return _success(request, test_lock=_services(request).runtime_profiles.test_lock_state())


@router.post("/runtime/test-lock")
def set_runtime_test_lock(payload: OfficialTestLockRequest, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    result = _services(request).runtime_profiles.set_official_test_mode(payload.enabled, payload.actor)
    return _success(request, result=result, test_lock=result.get("test_lock"))


@router.post("/calibrations")
def calibrate(payload: CalibrationRequest, request: Request) -> dict[str, Any]:
    services = _services(request)
    services.test_lock.require_write_allowed(
        "calibration",
        context={"dataset": payload.dataset, "category": payload.category},
    )
    job = services.adapter.start_job("calibration")
    result = services.adapter.calibrate(job, _request_id(request), **payload.model_dump())
    result.update(services.test_lock.evaluation_fields(result.get("calibration") or {}))
    return _success(request, calibration=result)


@router.post("/calibrations/activate")
def activate_calibration(payload: CalibrationActivateRequest, request: Request) -> dict[str, Any]:
    services = _services(request)
    services.test_lock.require_write_allowed(
        "calibration",
        context={"dataset": payload.dataset, "category": payload.category},
    )
    try:
        path = services.settings.activate_calibration(payload.dataset, payload.category, payload.calibration_path)
    except (FileNotFoundError, ValueError) as exc:
        raise ApiError("CALIBRATION_ACTIVATION_FAILED", str(exc), 422) from exc
    invalidated = services.registry.invalidate(str(payload.dataset).lower(), payload.category)
    return _success(request, active_calibration_path=str(path), invalidated_runtime_count=invalidated)


def _feedback_values(payload: FeedbackRequest) -> dict[str, Any]:
    values = payload.model_dump(exclude_none=True)
    legacy = payload.feedback_type
    if legacy and not payload.assessment:
        mapping = {
            "false positive": ("incorrect", "normal", ["false_alarm"]),
            "false negative": ("incorrect", "defect", ["missed_detection"]),
            "true positive": ("correct", "keep", []),
            "true negative": ("correct", "keep", []),
        }
        assessment, corrected_label, issue_tags = mapping[legacy]
        values.update(assessment=assessment, corrected_label=corrected_label, issue_tags=issue_tags, application_mode="save_only")
    if values.get("assessment") not in {"correct", "incorrect", "uncertain"}:
        raise ApiError("INVALID_FEEDBACK", "请选择当前检测结果是否正确。", 422)
    if values.get("corrected_label") not in {"keep", "normal", "defect"}:
        raise ApiError("INVALID_FEEDBACK", "请选择有效的用户修正结果。", 422)
    return values


def _prediction_feedback_label(value: Any) -> str | None:
    prediction = str(value or "").strip().lower()
    if prediction in {"0", "normal", "ok", "good", "negative"}:
        return "normal"
    if prediction in {"1", "anomaly", "abnormal", "defect", "ng", "positive"}:
        return "defect"
    return None


def _resolve_video_feedback_label(values: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Make every newly submitted video feedback carry an explicit human label.

    A correct assessment confirms the immutable frame prediction.  An
    incorrect/uncertain assessment must name the real label explicitly.  The
    server performs this validation as well as the UI so old clients cannot
    create another ``corrected_label=keep`` video candidate.
    """
    if str(context.get("mode") or "") != "video":
        return values
    resolved = dict(values)
    assessment = str(resolved.get("assessment") or "")
    corrected_label = str(resolved.get("corrected_label") or "keep")
    prediction_label = _prediction_feedback_label(context.get("prediction"))
    if assessment == "correct":
        if prediction_label is None:
            raise ApiError(
                "VIDEO_FEEDBACK_LABEL_REQUIRED",
                "当前视频帧的原始判定无法解析，请明确选择人工标签。",
                422,
            )
        if corrected_label in {"normal", "defect"} and corrected_label != prediction_label:
            raise ApiError(
                "VIDEO_FEEDBACK_LABEL_CONFLICT",
                "人工标签与“结果正确”不一致；请改为“结果有误”后提交。",
                422,
            )
        resolved["corrected_label"] = prediction_label
    else:
        if corrected_label not in {"normal", "defect"}:
            raise ApiError(
                "VIDEO_FEEDBACK_LABEL_REQUIRED",
                "视频反馈必须明确人工标签为正常或缺陷。",
                422,
            )
        if assessment == "incorrect" and prediction_label == corrected_label:
            raise ApiError(
                "VIDEO_FEEDBACK_LABEL_CONFLICT",
                "人工标签与原始判定相同；请确认是“结果正确”还是修改人工标签。",
                422,
            )
    return resolved


def _feedback_context(payload: FeedbackRequest, request: Request) -> dict[str, Any]:
    services = _services(request)
    if payload.mode == "video" or payload.frame_index is not None:
        if payload.mode != "video" or not payload.task_id or payload.frame_index is None:
            raise ApiError("INVALID_VIDEO_FEEDBACK", "视频反馈必须关联视频任务和具体帧。", 422)
        context = services.video_jobs.feedback_context(payload.task_id, payload.frame_index)
        if payload.sample_id and payload.sample_id != context["sample_id"]:
            raise ApiError("VIDEO_FRAME_MISMATCH", "视频帧标识与任务记录不匹配。", 422)
        if payload.interval_start_ms is not None or payload.interval_end_ms is not None:
            start_ms = float(payload.interval_start_ms or 0.0)
            end_ms = float(payload.interval_end_ms if payload.interval_end_ms is not None else start_ms)
            if end_ms < start_ms:
                raise ApiError("INVALID_VIDEO_INTERVAL", "视频反馈时间区间无效。", 422)
            context["interval_start_ms"] = start_ms
            context["interval_end_ms"] = end_ms
    else:
        if not payload.result_id:
            raise ApiError("INVALID_IMAGE_FEEDBACK", "图像反馈必须关联具体检测结果。", 422)
        try:
            context = services.storage.detection_feedback_context(
                payload.result_id,
                task_id=payload.task_id,
                mode=payload.mode,
                sample_id=payload.sample_id,
            )
        except KeyError as exc:
            raise ApiError("RESULT_NOT_FOUND", "反馈对应的检测结果不存在。", 404) from exc
        except ValueError as exc:
            raise ApiError("FEEDBACK_TARGET_MISMATCH", str(exc), 422) from exc

    memory_bank_id = str(context.get("memory_bank_id") or "")
    if not memory_bank_id:
        raise ApiError(
            "FEEDBACK_SOURCE_MEMORY_REQUIRED",
            "该检测结果没有可追溯的来源记忆库，不能加入反馈优化中心。",
            422,
        )
    try:
        services.domain_profiles.resolve_for_inference(
            memory_bank_id,
            str(context.get("model_version") or ""),
            str(context.get("dataset") or ""),
            str(context.get("category") or ""),
        )
    except ApiError:
        raise
    except (FileNotFoundError, ValueError) as exc:
        raise ApiError("MEMORY_BANK_NOT_FOUND", "检测记录对应的记忆库已不存在，反馈未保存。", 422) from exc
    # These aliases are derived only from the immutable task/result context.
    # Never infer a historical feedback source from the current runtime profile.
    context["source_memory_bank_id"] = str(context.get("source_memory_bank_id") or memory_bank_id)
    context["source_profile_id"] = str(context.get("source_profile_id") or memory_bank_id)
    context["source_target_domain_id"] = str(
        context.get("source_target_domain_id") or context["source_profile_id"]
    )
    context["source_config_snapshot_id"] = str(
        context.get("source_config_snapshot_id")
        or context.get("config_profile_id")
        or context.get("config_version")
        or ""
    )
    runtime = services.runtime_profiles.raw_runtime()
    # Preserve the immutable detection facts while binding the candidate to
    # the operator's explicit evaluation lock at submission time.  Normal
    # inference remains read-only regardless of whether offline feedback
    # optimization is permitted.
    context["official_test_mode"] = bool(
        context.get("official_test_mode") or runtime.get("official_test_mode")
    )
    context["feedback_optimization_locked"] = bool(
        context.get("feedback_optimization_locked")
        or runtime.get("feedback_optimization_locked")
    )
    return context


@router.get("/feedback")
def feedback_records(
    request: Request,
    task_id: str,
    mode: str | None = None,
    sample_id: str | None = None,
    frame_index: int | None = None,
) -> dict[str, Any]:
    if not task_id or len(task_id) > 128:
        raise ApiError("INVALID_TASK_ID", "反馈任务标识无效。", 422)
    records = _services(request).storage.list_feedback(
        task_id=task_id,
        mode=mode,
        sample_id=sample_id,
        frame_index=frame_index,
    )
    return _success(request, feedbacks=[_public_feedback_record(record) for record in records])


@router.post("/feedback/bulk")
def bulk_feedback(payload: FeedbackBulkRequest, request: Request) -> dict[str, Any]:
    services = _services(request)
    records = []
    candidates = []
    failures: list[dict[str, Any]] = []
    for index, item in enumerate(payload.feedbacks):
        try:
            values = _feedback_values(item)
            context = _feedback_context(item, request)
            values = _resolve_video_feedback_label(values, context)
            record, candidate = services.feedback_candidates.submit_feedback(values, context)
            records.append(record)
            candidates.append(candidate)
        except ApiError as exc:
            # One stale/deleted frame must not prevent other valid selected
            # frames from being recorded.  The UI keeps only these failures
            # selected so the operator can review or retry them individually.
            failures.append(
                {
                    "index": index,
                    "task_id": item.task_id,
                    "sample_id": item.sample_id,
                    "frame_index": item.frame_index,
                    "code": exc.code,
                    "message": exc.message,
                }
            )
    candidate_statuses = {
        str(candidate.get("status") or "pending_review") for candidate in candidates
    }
    submission_status = (
        next(iter(candidate_statuses)) if len(candidate_statuses) == 1
        else "mixed" if candidate_statuses
        else "recorded"
    )
    return _success(
        request,
        feedbacks=[_public_feedback_record(record) for record in records],
        candidates=[services.feedback_candidates.public_candidate(item) for item in candidates],
        saved_count=len(records),
        failed_count=len(failures),
        failures=failures,
        status=submission_status,
    )


@router.post("/feedback")
def feedback(payload: FeedbackRequest, request: Request) -> dict[str, Any]:
    services = _services(request)
    values = _feedback_values(payload)
    context = _feedback_context(payload, request)
    values = _resolve_video_feedback_label(values, context)
    record, candidate = services.feedback_candidates.submit_feedback(values, context)
    return _success(
        request,
        feedback=_public_feedback_record(record),
        candidate=services.feedback_candidates.public_candidate(candidate),
        feedback_id=str(record.get("feedback_id") or ""),
        candidate_id=str(candidate.get("candidate_id") or ""),
        status=str(candidate.get("status") or "pending_review"),
    )


def _public_feedback_record(record: dict[str, Any]) -> dict[str, Any]:
    """Strip backend-only source paths from legacy feedback responses."""
    value = dict(record)
    context = dict(value.get("context") or {})
    context.pop("_source_path", None)
    context.pop("source_path", None)
    value["context"] = context
    return value


@router.get("/feedback/candidates")
def feedback_candidates(
    request: Request,
    status: str | None = None,
    memory_bank_id: str | None = None,
    source_profile_id: str | None = None,
    source_target_domain_id: str | None = None,
    dataset: str | None = None,
    category: str | None = None,
    task_id: str | None = None,
    mode: Literal["single", "batch", "video"] | None = None,
    search: str | None = None,
    label: Literal["normal", "defect", "unknown"] | None = None,
    issue: Literal["false_alarm", "missed_detection", "correct", "uncertain"] | None = None,
    mask: Literal["with", "without"] | None = None,
    detection_since: str | None = None,
    feedback_since: str | None = None,
    min_score: float | None = None,
    method: str | None = None,
    usage: Literal["used", "unused"] | None = None,
    limit: int | None = Query(default=None, ge=1, le=500),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    services = _services(request)
    effective_size = int(limit or page_size)
    filters = {
        "status": status,
        "memory_bank_id": memory_bank_id,
        "source_profile_id": source_profile_id,
        "source_target_domain_id": source_target_domain_id,
        "dataset": dataset,
        "category": category,
        "task_id": task_id,
        "mode": mode,
        "search": search,
        "label": label,
        "issue": issue,
        "mask": mask,
        "detection_since": detection_since,
        "feedback_since": feedback_since,
        "min_score": min_score,
        "method": method,
        "usage": usage,
    }
    rows = services.feedback_candidates.list_candidates(
        **filters,
        limit=effective_size,
        offset=(page - 1) * effective_size,
    )
    total = services.feedback_candidates.count_candidates(**filters)
    return _success(
        request,
        candidates=[services.feedback_candidates.public_candidate(item) for item in rows],
        total=total,
        page=page,
        page_size=effective_size,
        total_pages=max(1, (total + effective_size - 1) // effective_size),
    )


@router.get("/feedback/revision")
def feedback_revision(request: Request, since: int | None = None) -> dict[str, Any]:
    current = _services(request).feedback_candidates.revision()
    revision = int(current.get("revision") or 0)
    return _success(
        request,
        feedback_revision=revision,
        changed=since is None or int(since) != revision,
        updated_at=current.get("updated_at"),
        reason=current.get("reason"),
    )


@router.get("/feedback/resources")
def feedback_resources(request: Request) -> dict[str, Any]:
    services = _services(request)
    profiles = services.domain_profiles.list_all_profiles()
    return _success(
        request,
        resources=services.feedback_candidates.resource_summaries(profiles),
    )


@router.post("/feedback/candidates/delete-preview")
def feedback_candidates_delete_preview(payload: FeedbackCandidateDeleteRequest, request: Request) -> dict[str, Any]:
    return _success(request, preview=_services(request).feedback_candidates.delete_preview(payload.candidate_ids))


@router.post("/feedback/candidates/delete")
def delete_feedback_candidates(payload: FeedbackCandidateDeleteRequest, request: Request) -> dict[str, Any]:
    if not payload.acknowledged:
        raise ApiError("CONFIRMATION_REQUIRED", "请先勾选“我已核对删除预览”。", 422)
    result = _services(request).feedback_candidates.delete_candidates(
        payload.candidate_ids,
        preview_token=payload.preview_token,
        acknowledged=True,
        actor=payload.actor,
    )
    return _success(request, **result)


@router.get("/feedback/candidates/{candidate_id}")
def feedback_candidate(candidate_id: str, request: Request) -> dict[str, Any]:
    services = _services(request)
    return _success(request, candidate=services.feedback_candidates.public_candidate(services.feedback_candidates.get_candidate(candidate_id)))


@router.post("/feedback/candidates/{candidate_id}/review")
def review_feedback_candidate(candidate_id: str, payload: FeedbackCandidateReviewRequest, request: Request) -> dict[str, Any]:
    services = _services(request)
    result = services.feedback_candidates.review(
        candidate_id,
        action=payload.action,
        reviewer=payload.reviewer,
        corrected_label=payload.corrected_label,
        corrected_defect_type=payload.corrected_defect_type,
        note=payload.note,
        require_snapshot=payload.require_snapshot,
        expected_version=payload.candidate_version,
        video_jobs=services.video_jobs,
    )
    return _success(request, candidate=services.feedback_candidates.public_candidate(result))


@router.post("/feedback/candidates/review-bulk")
def review_feedback_candidates_bulk(payload: FeedbackCandidateBulkReviewRequest, request: Request) -> dict[str, Any]:
    services = _services(request)
    result = services.feedback_candidates.bulk_review(
        payload.candidate_ids,
        action=payload.action,
        note=payload.note or "",
        require_snapshot=payload.require_snapshot,
        video_jobs=services.video_jobs,
    )
    return _success(request, **result, reviewed_count=result["success_count"])


@router.post("/feedback/candidates/bulk-review")
def review_feedback_candidates_bulk_compat(payload: FeedbackCandidateBulkReviewRequest, request: Request) -> dict[str, Any]:
    return review_feedback_candidates_bulk(payload, request)


@router.post("/feedback/candidates/{candidate_id}/reject")
def reject_feedback_candidate(candidate_id: str, request: Request) -> dict[str, Any]:
    result = _services(request).feedback_candidates.review(candidate_id, action="reject", reviewer="local_operator")
    return _success(request, candidate=_services(request).feedback_candidates.public_candidate(result))


@router.post("/feedback/candidates/{candidate_id}/snapshot")
def snapshot_feedback_candidate(candidate_id: str, request: Request) -> dict[str, Any]:
    services = _services(request)
    candidate = services.feedback_candidates.create_snapshot(candidate_id, services.video_jobs)
    return _success(request, candidate=services.feedback_candidates.public_candidate(candidate))


@router.get("/feedback/candidates/{candidate_id}/protection-status")
def feedback_candidate_protection_status(candidate_id: str, request: Request) -> dict[str, Any]:
    return _success(request, protection=_services(request).feedback_candidates.protection_status(candidate_id))


@router.get("/feedback/candidates/{candidate_id}/annotations")
def feedback_annotation_history(candidate_id: str, request: Request) -> dict[str, Any]:
    return _success(request, annotations=_services(request).feedback_candidates.annotation_history(candidate_id))


@router.get("/feedback/candidates/{candidate_id}/annotation")
def feedback_annotation_latest(candidate_id: str, request: Request) -> dict[str, Any]:
    history = _services(request).feedback_candidates.annotation_history(candidate_id)
    return _success(request, annotation=history[0] if history else None, annotations=history)


@router.get("/feedback/candidates/{candidate_id}/source")
def feedback_candidate_source(candidate_id: str, request: Request) -> dict[str, Any]:
    return _success(request, source=_services(request).feedback_candidates.source_info(candidate_id))


@router.get("/feedback/candidates/{candidate_id}/source-preview")
def feedback_candidate_source_preview(candidate_id: str, request: Request) -> dict[str, Any]:
    services = _services(request)
    return _success(request, source=services.feedback_candidates.source_preview(candidate_id, services.video_jobs))


@router.post("/feedback/candidates/{candidate_id}/annotations")
def save_feedback_annotation(candidate_id: str, payload: FeedbackAnnotationRequest, request: Request) -> dict[str, Any]:
    annotation = _services(request).feedback_candidates.save_annotation(
        candidate_id,
        annotation_type=payload.annotation_type,
        mask_base64=payload.mask_base64,
        use_original_mask=payload.use_original_mask,
        mask_width=payload.mask_width,
        mask_height=payload.mask_height,
        bbox_json=payload.bbox,
        polygon_json=payload.polygon,
        note=payload.note,
        created_by=payload.created_by,
    )
    return _success(request, annotation=annotation)


@router.put("/feedback/candidates/{candidate_id}/annotation")
def update_feedback_annotation(candidate_id: str, payload: FeedbackAnnotationRequest, request: Request) -> dict[str, Any]:
    return save_feedback_annotation(candidate_id, payload, request)


@router.get("/feedback/candidates/{candidate_id}/annotation/history")
def feedback_annotation_history_compat(candidate_id: str, request: Request) -> dict[str, Any]:
    return feedback_annotation_history(candidate_id, request)


@router.get("/feedback/statistics")
def feedback_statistics(request: Request, memory_bank_id: str | None = None) -> dict[str, Any]:
    services = _services(request)
    visible_memory_bank_ids = None
    if not memory_bank_id:
        profiles = services.domain_profiles.list_all_profiles()
        visible_memory_bank_ids = services.feedback_candidates.active_memory_bank_ids(profiles)
    return _success(
        request,
        statistics=services.feedback_candidates.statistics(
            memory_bank_id=memory_bank_id,
            memory_bank_ids=visible_memory_bank_ids,
        ),
    )


@router.get("/feedback/audit")
def feedback_audit(request: Request, candidate_id: str | None = None, job_id: str | None = None,
                   memory_bank_id: str | None = None,
                   limit: int = Query(default=200, ge=1, le=500)) -> dict[str, Any]:
    return _success(
        request,
        audit=_services(request).feedback_candidates.audit(
            candidate_id=candidate_id,
            job_id=job_id,
            memory_bank_id=memory_bank_id,
            limit=limit,
        ),
    )


@router.post("/feedback/optimization/jobs")
def create_feedback_optimization(payload: FeedbackOptimizationCreateRequest, request: Request) -> dict[str, Any]:
    return _success(request, job=_services(request).feedback_optimization.create_job(payload.model_dump()))


@router.post("/feedback/optimization/compatibility")
def feedback_optimization_compatibility(payload: FeedbackOptimizationCreateRequest, request: Request) -> dict[str, Any]:
    return _success(request, compatibility=_services(request).feedback_optimization.compatibility(payload.model_dump()))


@router.get("/feedback/optimization/strategies")
def feedback_optimization_strategies(
    request: Request,
    method_mode: str | None = Query(default=None, max_length=80),
) -> dict[str, Any]:
    """Return strategy admission policies from the same backend source used by compatibility."""
    return _success(
        request,
        strategies=_services(request).feedback_optimization.strategy_policies(method_mode),
    )


@router.get("/feedback/optimization/jobs")
def list_feedback_optimization_jobs(
    request: Request,
    limit: int = Query(default=100, ge=1, le=200),
    summary: bool = Query(default=False),
) -> dict[str, Any]:
    return _success(request, jobs=_services(request).feedback_optimization.list_jobs(limit, summary=summary))


@router.get("/feedback/optimization/jobs/{job_id}")
def feedback_optimization_status(
    job_id: str,
    request: Request,
    compact: bool = Query(default=False),
    include_items: bool = Query(default=True),
) -> dict[str, Any]:
    return _success(
        request,
        job=_services(request).feedback_optimization.status(
            job_id,
            include_items=include_items,
            compact=compact,
        ),
    )


@router.post("/feedback/optimization/jobs/{job_id}/cancel")
def cancel_feedback_optimization(job_id: str, request: Request) -> dict[str, Any]:
    return _success(request, job=_services(request).feedback_optimization.cancel(job_id))


@router.post("/feedback/optimization/jobs/{job_id}/rerun")
def rerun_feedback_optimization(job_id: str, request: Request) -> dict[str, Any]:
    return _success(request, job=_services(request).feedback_optimization.rerun(job_id))


@router.post("/feedback/optimization/jobs/{job_id}/validate")
def validate_feedback_optimization(job_id: str, request: Request) -> dict[str, Any]:
    # Validation happens as the final worker gate.  This endpoint is an
    # idempotent status read and never promotes a failed job.
    job = _services(request).feedback_optimization.status(job_id)
    return _success(request, job=job, validated=job.get("status") in {"ready_to_apply", "frozen", "applied", "rolled_back"})


@router.get("/feedback/optimization/jobs/{job_id}/candidate-profile")
def feedback_candidate_profile(job_id: str, request: Request) -> dict[str, Any]:
    services = _services(request)
    job = services.feedback_optimization.status(job_id, include_items=False)
    if not job.get("candidate_profile_id"):
        raise ApiError("FEEDBACK_CANDIDATE_PROFILE_NOT_READY", "候选 profile 尚未生成。", 409)
    candidate_id = str(job.get("candidate_profile_id") or "")
    base_id = str(job.get("base_profile_id") or "")
    constraints = dict(job.get("constraints") or {})
    validation = dict(job.get("validation") or {})

    base_profile = services.feedback_optimization.profile_metadata(base_id)
    candidate_metadata = services.feedback_optimization.profile_metadata(candidate_id)

    def manifest_config_refs(metadata: dict[str, Any], profile_id: Any, version: Any) -> list[dict[str, Any]]:
        refs = [dict(item) for item in metadata.get("config_profiles") or [] if isinstance(item, dict)]
        if not refs and profile_id and version:
            refs = [{
                "config_profile_id": str(profile_id),
                "config_profile_version": int(version),
                "friendly_name": "初始化配置",
                "active": True,
            }]
        return refs

    base_configs = manifest_config_refs(
        base_profile,
        job.get("base_config_profile_id"),
        job.get("base_config_profile_version"),
    )
    candidate_configs = manifest_config_refs(
        candidate_metadata,
        job.get("candidate_config_profile_id"),
        job.get("candidate_config_profile_version"),
    )

    def config_summary(refs: list[dict[str, Any]], profile_id: Any, version: Any) -> dict[str, Any]:
        for ref in refs:
            ref_version = ref.get("version") or ref.get("config_profile_version")
            if (
                str(ref.get("config_profile_id") or "") == str(profile_id or "")
                and int(ref_version or 0) == int(version or 0)
            ):
                return {
                    "friendly_name": str(ref.get("friendly_name") or "当前绑定配置"),
                    "version": int(ref_version or version or 1),
                }
        return {"friendly_name": "当前绑定配置", "version": int(version or 1)} if profile_id else {}

    base_config = config_summary(base_configs, job.get("base_config_profile_id"), job.get("base_config_profile_version"))
    candidate_config = config_summary(
        candidate_configs,
        job.get("candidate_config_profile_id"),
        job.get("candidate_config_profile_version"),
    )
    inherited_configs = [
        {
            "friendly_name": str(item.get("friendly_name") or "未命名配置"),
            "version": int(item.get("version") or item.get("config_profile_version") or 1),
            "active": (
                str(item.get("config_profile_id") or "") == str(job.get("candidate_config_profile_id") or "")
                and int(item.get("version") or item.get("config_profile_version") or 0)
                == int(job.get("candidate_config_profile_version") or 0)
            ),
            "source_label": str(item.get("source_label") or "反馈更新继承"),
        }
        for item in candidate_configs
        if not bool(item.get("active"))
    ]
    integrity_passed = str(validation.get("status") or "").lower() == "passed"
    return _success(
        request,
        candidate_profile={
            "memory_bank_id": candidate_id,
            "friendly_name": str(constraints.get("version_name") or f"反馈更新版本 · {candidate_id[-8:]}"),
            "config_profile_id": job.get("candidate_config_profile_id"),
            "config_profile_version": job.get("candidate_config_profile_version"),
            "config_count": len(candidate_configs) or (1 if job.get("candidate_config_profile_id") else 0),
            "default_config_name": candidate_config.get("friendly_name") or "当前绑定配置",
            "inherited_configs": inherited_configs,
            "inherited_config_names": [item["friendly_name"] for item in inherited_configs if not item["active"]],
            "validation": validation,
            "complexity_before": job.get("complexity_before") or {},
            "complexity_after": job.get("complexity_after") or {},
            "publish_mode": constraints.get("publish_mode") or "save_new_version",
            "version_name": constraints.get("version_name") or "",
            "status": job.get("status"),
            "status_display": job.get("task_status_display") or "状态未知",
            "integrity_passed": integrity_passed,
            "integrity_display": "资源完整性校验通过" if integrity_passed else "资源完整性尚未通过",
            "base_profile": {
                "memory_bank_id": base_id,
                "friendly_name": str(base_profile.get("friendly_name") or "基础目标域记忆库"),
                "version": int(base_profile.get("version") or base_profile.get("current_version") or 1),
                "config_count": len(base_configs),
                "default_config_name": base_config.get("friendly_name") or "当前绑定配置",
                "default_config_version": base_config.get("version") or job.get("base_config_profile_version"),
            },
        },
    )


@router.post("/feedback/optimization/jobs/{job_id}/freeze")
def freeze_feedback_optimization(job_id: str, payload: FeedbackOptimizationApplyRequest, request: Request) -> dict[str, Any]:
    return _success(request, job=_services(request).feedback_optimization.freeze(job_id, confirm=payload.confirm, actor=payload.actor))


@router.post("/feedback/optimization/jobs/{job_id}/apply")
def apply_feedback_optimization(job_id: str, payload: FeedbackOptimizationApplyRequest, request: Request) -> dict[str, Any]:
    return _success(request, job=_services(request).feedback_optimization.apply(job_id, confirm=payload.confirm, actor=payload.actor))


@router.post("/feedback/optimization/jobs/{job_id}/rollback")
def rollback_feedback_optimization(job_id: str, payload: FeedbackOptimizationRollbackRequest, request: Request) -> dict[str, Any]:
    return _success(request, job=_services(request).feedback_optimization.rollback(job_id, confirm=payload.confirm, actor=payload.actor))


@router.get("/jobs/{job_id}")
def job_status(job_id: str, request: Request) -> dict[str, Any]:
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in job_id):
        raise ApiError("INVALID_JOB_ID", "Job ID contains invalid characters.", 422)
    root = _services(request).settings.jobs_root / job_id
    manifest = root / "job_manifest.json"
    if not manifest.exists():
        durable = _services(request).jobs.durable_state(job_id)
        if durable is None:
            raise ApiError("JOB_NOT_FOUND", "Job was not found.", 404)
        return _success(request, job={**dict(durable.get("payload") or {}), "status": durable.get("runtime_status"), "rerunnable": durable.get("rerunnable")}, result=None)
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ApiError("JOB_MANIFEST_INVALID", "Job manifest could not be read.", 500) from exc
    result = None
    response_file = str(payload.get("response_file") or "")
    if response_file:
        response_path = (root / Path(response_file).name).resolve()
        try:
            response_path.relative_to(root.resolve())
            result_payload = json.loads(response_path.read_text(encoding="utf-8"))
            if isinstance(result_payload, dict):
                result = result_payload
        except (OSError, json.JSONDecodeError, ValueError):
            result = None
    return _success(request, job=payload, result=result)


@router.post("/jobs/{job_id}/rerun")
def rerun_job(job_id: str, request: Request) -> dict[str, Any]:
    _require_local_request(request)
    services = _services(request)
    durable = services.jobs.durable_state(job_id)
    if not durable or durable.get("task_kind") != "batch":
        raise ApiError("JOB_RERUN_UNSUPPORTED", "当前任务不支持重新运行。", 409)
    if durable.get("status") != "interrupted":
        raise ApiError("JOB_NOT_INTERRUPTED", "只有因服务重启中断的批量任务可以重新运行。", 409)
    manifest = dict(durable.get("payload") or {})
    persisted_selection = dict(manifest.get("selection") or {})
    selection = {
        key: persisted_selection.get(key)
        for key in (
            "model_id", "dataset", "category", "device", "threshold",
            "save_visuals", "memory_bank_id", "mode",
        )
    }
    source_values = [str(value) for value in manifest.get("input_files") or [] if str(value)]
    sources = [Path(value).expanduser().resolve() for value in source_values]
    missing = [path.name for path in sources if not path.is_file()]
    if not sources or missing:
        raise ApiError(
            "JOB_RERUN_INPUT_MISSING",
            "中断任务的输入文件已不存在，无法重新运行。",
            409,
            {"missing_files": missing[:20]},
        )
    job = services.adapter.start_job("batch")
    request_id = _request_id(request)
    services.jobs.update(
        job,
        status="queued",
        stage="preparing",
        request_id=request_id,
        rerun_of=job_id,
        input_files=[str(path) for path in sources],
        selection=selection,
        input_count=len(sources),
    )

    def run_rerun() -> None:
        try:
            result = services.adapter.detect_batch(job, sources, request_id, **selection)
            response_path = job.root / "batch_response.json"
            temporary = response_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            temporary.replace(response_path)
            services.jobs.update(job, response_file=response_path.name)
        except ApiError as exc:
            services.jobs.fail(job, exc.code, exc.message)
        except Exception as exc:
            services.logger.exception("batch_rerun_failed job_id=%s", job.job_id)
            services.jobs.fail(job, "BATCH_RERUN_FAILED", str(exc))

    threading.Thread(
        target=run_rerun,
        name=f"AdaptiveAOI-Batch-Rerun-{job.job_id[-10:]}",
        daemon=True,
    ).start()
    services.jobs.states.record_rerun(job_id, job.job_id)
    return _success(request, job_id=job.job_id, batch_id=job.job_id, status="queued", rerun_of=job_id)


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request) -> dict[str, Any]:
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in job_id):
        raise ApiError("INVALID_JOB_ID", "任务标识无效。", 422)
    services = _services(request)
    root = services.settings.jobs_root / job_id
    manifest = root / "job_manifest.json"
    if not manifest.exists():
        raise ApiError("JOB_NOT_FOUND", "任务不存在。", 404)
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ApiError("JOB_MANIFEST_INVALID", "任务状态无法读取。", 500) from exc
    if str(payload.get("kind") or "") != "batch":
        raise ApiError("JOB_CANCEL_UNSUPPORTED", "该任务不支持此取消方式。", 409)
    status = str(payload.get("status") or "")
    if status in {"completed", "partial", "cancelled", "failed"}:
        return _success(request, job=payload)
    job = services.jobs.get(job_id, "batch")
    updated = services.jobs.update(
        job,
        cancel_requested=True,
        cancel_requested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        status="cancelling",
        stage="cancelling",
        eta_seconds=None,
    )
    return _success(request, job=updated)
