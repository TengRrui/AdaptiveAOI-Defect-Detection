from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from core.domain_build_contract import (
    DEFAULT_CREATE_MODE,
    default_feature_layers,
    normalize_precision_request,
    precision_uses_fp16,
)


class ModelLoadRequest(BaseModel):
    model_id: str | None = None
    dataset: str | None = None
    category: str | None = None
    device: Literal["auto", "cpu", "cuda"] | None = None
    threshold: float | None = Field(default=None, ge=0.0)
    memory_bank_id: str | None = Field(default=None, max_length=128)
    mode: Literal["inference_only"] = "inference_only"


class ConfigUpdateRequest(BaseModel):
    model_id: str | None = None
    dataset: str | None = Field(default=None, min_length=1, max_length=80)
    category: str | None = Field(default=None, min_length=1, max_length=120)
    config: str | None = Field(default=None, min_length=1, max_length=260)
    device: Literal["auto", "cpu", "cuda"] | None = None
    threshold: float | None = Field(default=None, ge=0.0)
    memory_bank_id: str | None = Field(default=None, min_length=1, max_length=128)
    inference_mode: Literal["inference_only"] | None = None
    save_visuals: bool | None = None
    save_original: bool | None = None
    config_profile_id: str | None = Field(default=None, min_length=3, max_length=64)
    config_profile_version: int | None = Field(default=None, ge=1)
    config_hash: str | None = Field(default=None, min_length=64, max_length=64)


class OfficialTestLockRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    actor: str = Field(default="operator", min_length=1, max_length=80)


class ConfigProfileCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    friendly_name: str = Field(min_length=1, max_length=80)
    note: str | None = Field(default=None, max_length=500)
    source_kind: str | None = Field(default="custom", max_length=40)
    config: dict[str, Any] = Field(default_factory=dict)


class ConfigProfileUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    friendly_name: str | None = Field(default=None, min_length=1, max_length=80)
    note: str | None = Field(default=None, max_length=500)
    config: dict[str, Any] = Field(default_factory=dict)


class ConfigProfileCopyRequest(BaseModel):
    friendly_name: str | None = Field(default=None, min_length=1, max_length=80)


class MemoryBankConfigCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    friendly_name: str = Field(min_length=1, max_length=80)
    note: str | None = Field(default=None, max_length=500)
    source_config_profile_id: str | None = Field(default=None, min_length=3, max_length=64)
    source_config_profile_version: int | None = Field(default=None, ge=1)
    config: dict[str, Any] = Field(default_factory=dict)


class MemoryBankConfigUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    friendly_name: str | None = Field(default=None, min_length=1, max_length=80)
    note: str | None = Field(default=None, max_length=500)
    config: dict[str, Any] = Field(default_factory=dict)


class MemoryBankConfigDeleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Deprecated compatibility field.  The formal UI uses the immutable
    # preview plus ``acknowledged`` instead of typing a phrase.
    confirmation_text: str = Field(default="", max_length=20)
    acknowledged: bool = False


class ConfigProfileImportRequest(BaseModel):
    friendly_name: str = Field(min_length=1, max_length=80)
    yaml_text: str = Field(min_length=1, max_length=200000)


class CalibrationRequest(BaseModel):
    model_id: str | None = None
    dataset: str
    category: str
    normal_shot: int = Field(default=100, ge=1)
    defect_shot: int = Field(default=30, ge=0)
    seed: int = Field(default=0, ge=0)
    use_defect_direction: bool = False


class CalibrationActivateRequest(BaseModel):
    dataset: str
    category: str
    calibration_path: str


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # ``feedback_type`` keeps the first-generation API compatible.  New UI
    # callers use the mode/task/sample fields and the explicit human review
    # fields below.  Runtime/model facts are resolved by the server from the
    # referenced detection record; they are intentionally not accepted here.
    result_id: str | None = Field(default=None, max_length=128)
    feedback_type: Literal["false positive", "false negative", "true positive", "true negative"] | None = None
    mode: Literal["single", "batch", "video"] | None = None
    task_id: str | None = Field(default=None, max_length=128)
    sample_id: str | None = Field(default=None, max_length=256)
    frame_index: int | None = Field(default=None, ge=0)
    timestamp_ms: float | None = Field(default=None, ge=0.0)
    interval_start_ms: float | None = Field(default=None, ge=0.0)
    interval_end_ms: float | None = Field(default=None, ge=0.0)
    # Optional server-issued perceptual hash for interval-selected video
    # frames.  It is evidence for de-duplication only; it never identifies a
    # file or grants access to a path.
    perceptual_hash: str | None = Field(default=None, max_length=128)
    assessment: Literal["correct", "incorrect", "uncertain"] | None = None
    corrected_label: Literal["keep", "normal", "defect"] | None = None
    corrected_defect_type: str | None = Field(default=None, max_length=120)
    issue_tags: list[
        Literal[
            "score_high",
            "score_low",
            "region_inaccurate",
            "missed_detection",
            "false_alarm",
            "other",
        ]
    ] = Field(default_factory=list, max_length=6)
    application_mode: Literal["save_only", "calibration_candidate"] = "save_only"
    note: str | None = Field(default=None, max_length=1000)


class FeedbackBulkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feedbacks: list[FeedbackRequest] = Field(min_length=1, max_length=500)


class FeedbackCandidateReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["accept", "reject", "withdraw", "update", "resolve_conflict", "start_review", "archive", "dequeue"] = Field(
        validation_alias=AliasChoices("action", "decision")
    )
    reviewer: str = Field(default="local_operator", min_length=1, max_length=80)
    corrected_label: Literal["keep", "normal", "defect"] | None = None
    corrected_defect_type: str | None = Field(default=None, max_length=120)
    note: str | None = Field(default=None, max_length=1000, validation_alias=AliasChoices("note", "review_note"))
    require_snapshot: bool = True
    candidate_version: int | None = Field(default=None, ge=1)


class FeedbackCandidateBulkReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_ids: list[str] = Field(min_length=1, max_length=500)
    action: Literal["accept", "reject", "withdraw"]
    reviewer: str = Field(default="local_operator", min_length=1, max_length=80)
    note: str | None = Field(default=None, max_length=1000)
    require_snapshot: bool = True


class FeedbackCandidateDeleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_ids: list[str] = Field(min_length=1, max_length=500)
    confirm_text: str = Field(default="", max_length=20)
    acknowledged: bool = False
    preview_token: str = Field(default="", max_length=200)
    actor: str = Field(default="local_operator", min_length=1, max_length=80)


class FeedbackAnnotationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    annotation_type: Literal["mask", "bbox", "polygon", "region"] = "mask"
    mask_base64: str | None = Field(default=None, max_length=30_000_000)
    use_original_mask: bool = False
    mask_width: int | None = Field(default=None, ge=1, le=10000)
    mask_height: int | None = Field(default=None, ge=1, le=10000)
    bbox: dict[str, float] | None = None
    polygon: list[list[float]] | None = Field(default=None, max_length=5000)
    note: str | None = Field(default=None, max_length=1000)
    created_by: str = Field(default="operator", min_length=1, max_length=80)


class FeedbackOptimizationCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_profile_id: str = Field(min_length=3, max_length=128)
    candidate_ids: list[str] = Field(min_length=1, max_length=500)
    strategy: Literal[
        "safe_incremental",
        "comprehensive",
        "threshold_only",
        "normal_memory",
        "defect_prototypes",
        "lightweight_calibrator",
    ] = "safe_incremental"
    objective: Literal["f1", "recall_priority", "balanced"] = "f1"
    update_threshold: bool = True
    update_normal_memory: bool = False
    update_defect_memory: bool = False
    update_lightweight_calibrator: bool = False
    min_recall: float = Field(default=0.95, ge=0.0, le=1.0)
    max_false_positive_rate: float = Field(default=0.10, ge=0.0, le=1.0)
    max_memory_growth: float = Field(default=0.05, ge=0.0, le=0.10)
    max_latency_growth: float = Field(default=0.10, ge=0.0, le=1.0)
    publish_mode: Literal["replace_original", "update_current", "save_new_version"] = "update_current"
    replace_confirm: bool = False
    version_name: str = Field(default="", max_length=80)
    note: str | None = Field(default=None, max_length=1000)
    idempotency_key: str = Field(default="", max_length=128)


class FeedbackOptimizationApplyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm: bool = False
    actor: str = Field(default="operator", min_length=1, max_length=80)


class FeedbackOptimizationRollbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm: bool = False
    actor: str = Field(default="operator", min_length=1, max_length=80)


class FolderDialogRequest(BaseModel):
    initial_dir: str | None = Field(default=None, max_length=4096)
    dialog_type: Literal["folder", "file", "files"] = "folder"
    multiple: bool = False
    file_types: list[str] = Field(default_factory=list, max_length=20)
    recent_key: str | None = Field(default=None, pattern=r"^[a-z0-9_]{1,64}$")
    fallback_dirs: list[str] = Field(default_factory=list, max_length=5)


class FolderDialogRecentDirectoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recent_key: str = Field(pattern=r"^[a-z0-9_]{1,64}$")
    directory: str = Field(min_length=1, max_length=4096)


class DomainScanRequest(BaseModel):
    normal_dir: str = Field(default="", max_length=4096)
    defect_dir: str = Field(default="", max_length=4096)
    normal_source_type: Literal["image", "video", "mixed"] = "image"
    defect_source_type: Literal["image", "video", "mixed"] = "image"
    normal_image_dir: str = Field(default="", max_length=4096)
    normal_video_files: list[str] = Field(default_factory=list, max_length=256)
    normal_video_dir: str = Field(default="", max_length=4096)
    defect_image_dir: str = Field(default="", max_length=4096)
    defect_video_files: list[str] = Field(default_factory=list, max_length=256)
    defect_video_dir: str = Field(default="", max_length=4096)
    video_sampling_config: dict[str, Any] = Field(default_factory=dict)
    defect_video_intervals: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    excluded_sample_ids: list[str] = Field(default_factory=list, max_length=2000)
    sample_budget_mode: Literal["final_frame_count", "video_file_count", "custom"] = "final_frame_count"
    normal_target_count: int = Field(default=100, ge=1, le=100000)
    defect_target_count: int = Field(default=30, ge=0, le=100000)
    # Creation-mode policy is enforced by DomainProfileService so retired
    # clients receive a stable business error instead of a generic enum parse
    # failure.  Historical modes remain readable through GET resources.
    method_mode: str = Field(default=DEFAULT_CREATE_MODE, min_length=1, max_length=80)
    feedback_candidate_ids: list[str] = Field(default_factory=list, max_length=500)


class DomainBuildRequest(DomainScanRequest):
    # Optional link back to the persisted target-domain draft.  It is used
    # only for lifecycle protection/cleanup; the draft remains the UI's
    # source of truth and is never treated as a model input.
    draft_id: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9][a-z0-9_-]{7,63}$",
        description="与本次构建关联的目标域草稿标识。",
    )
    idempotency_key: str | None = Field(
        default=None,
        min_length=16,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
        description="一次用户构建操作的稳定幂等标识；页面重绘或网络重试必须复用同一值。",
    )
    model_id: str | None = None
    dataset: str = Field(min_length=1, max_length=80)
    category: str = Field(min_length=1, max_length=120)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    friendly_name: str | None = Field(default=None, max_length=80)
    backbone: Literal["wide_resnet50_2", "resnet50"] = "wide_resnet50_2"
    layers: list[Literal["layer1", "layer2", "layer3", "layer4"]] = Field(default_factory=default_feature_layers, min_length=1, max_length=4)
    image_size: int = Field(default=256, ge=64, le=2048)
    precision: Literal["fp16", "fp32"] = "fp16"
    # Deprecated request alias retained for old clients.  The validator below
    # always normalizes it to the canonical ``precision`` contract.
    use_fp16: bool | None = None
    coreset_ratio: float = Field(default=0.1, ge=0.001, le=1.0)
    max_memory_patches: int = Field(default=50000, ge=1, le=200000)
    knn_k: int = Field(default=1, ge=1, le=64)
    memory_backend: Literal["auto", "faiss_gpu_flat", "faiss_gpu_ivf", "faiss_cpu_flat", "sklearn", "torch_gpu_flat"] = "auto"
    memory_backend_fallback: Literal["faiss_cpu_flat", "sklearn", "torch_gpu_flat"] = "faiss_cpu_flat"
    memory_ivf_nlist: int = Field(default=256, ge=1, le=65536)
    memory_ivf_nprobe: int = Field(default=32, ge=1, le=65536)
    score_aggregation: Literal["maximum_patch", "top_k_mean"] = "maximum_patch"
    batch_size: int = Field(default=1, ge=1, le=32)
    threshold_method: Literal["normal_percentile", "mean_std", "best_f1", "recall_priority"] = "best_f1"
    threshold_percentile: float = Field(default=99.0, ge=50.0, le=100.0)
    threshold_target_recall: float = Field(default=0.95, ge=0.0, le=1.0)
    threshold_max_false_positive_rate: float = Field(default=0.10, ge=0.0, le=1.0)
    pixel_threshold_mode: Literal["automatic_from_normal_map"] = "automatic_from_normal_map"
    method_mode: str = Field(default=DEFAULT_CREATE_MODE, min_length=1, max_length=80)
    max_defect_prototypes: int = Field(default=32, ge=1, le=128)
    defect_top_ratio: float = Field(default=0.05, ge=0.005, le=0.25)
    defect_coreset_ratio: float = Field(default=0.25, ge=0.01, le=1.0)
    defect_region_filter: Literal["high_anomaly_patches", "reviewed_mask_preferred"] = "high_anomaly_patches"
    dual_fusion_strategy: Literal["reliability_gated_relative_distance"] = "reliability_gated_relative_distance"
    normal_floor_percentile: float = Field(default=99.0, ge=90.0, le=99.9)
    min_defect_image_support: int = Field(default=2, ge=1, le=30)
    lightweight_gate_percentile: float = Field(default=95.0, ge=90.0, le=99.9)
    lightweight_l2: float = Field(default=0.01, ge=0.0, le=10.0)
    lightweight_score_scale: float = Field(default=1.0, gt=0.0, le=100.0)
    lightweight_cv_folds: int = Field(default=5, ge=2, le=10)
    lightweight_pooling_candidates: list[float] = Field(default_factory=lambda: [0.005, 0.01, 0.02, 0.05], min_length=1, max_length=16)
    random_seed: int = Field(default=42, ge=0)
    overwrite_existing: bool = False
    publish_mode: Literal["candidate", "frozen"] = "frozen"
    config_profile_id: str | None = Field(default=None, min_length=3, max_length=64)
    config_profile_version: int | None = Field(default=None, ge=1)
    template_config_profile_id: str | None = Field(default=None, min_length=3, max_length=64)
    template_config_profile_version: int | None = Field(default=None, ge=1)
    config_profile_name: str | None = Field(default=None, max_length=80)
    note: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def normalize_build_precision(self) -> "DomainBuildRequest":
        fields = self.model_fields_set
        request: dict[str, Any] = {}
        if "precision" in fields:
            request["precision"] = self.precision
        if "use_fp16" in fields:
            request["use_fp16"] = self.use_fp16
        precision = normalize_precision_request(request, default="fp16")
        self.precision = precision
        self.use_fp16 = precision_uses_fp16(precision)
        return self


class DomainDraftSaveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    draft_id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9_-]{7,63}$")
    current_step: int = Field(default=1, ge=1, le=5)
    payload: dict[str, Any] = Field(default_factory=dict)


class DomainCandidateActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm: bool = False
    actor: str = Field(default="operator", min_length=1, max_length=80)


class DomainParameterVersionRequest(BaseModel):
    """Create a new candidate version without mutating a frozen profile."""

    model_config = ConfigDict(extra="forbid")

    friendly_name: str = Field(min_length=1, max_length=80)
    note: str | None = Field(default=None, max_length=500)
    threshold: float = Field(ge=0.0)
    pixel_threshold: float | None = Field(default=None, ge=0.0)
    score_aggregation: Literal["maximum_patch", "top_k_mean"] = "maximum_patch"
    save_visuals: bool = True
    save_original: bool = True


class MemoryBankActivateRequest(BaseModel):
    model_id: str | None = None
    dataset: str
    category: str


class MemoryBankRenameRequest(BaseModel):
    friendly_name: str = Field(min_length=1, max_length=80)


class RuntimeResourceActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm: bool = False
    acknowledged: bool = False
    # Deprecated: retained for old clients; formal UI never sends it.
    confirmation_text: str = Field(
        default="",
        max_length=160,
        validation_alias=AliasChoices("confirmation_text", "confirm_text"),
    )
    preview_token: str = Field(default="", max_length=160)
    delete_artifacts: bool = True
    reason: str = Field(default="", max_length=240)
    actor: str = Field(default="operator", min_length=1, max_length=80)
    current_profile_action: Literal["cancel", "switch", "restore_default", "pause"] = "cancel"
    replacement_resource_id: str | None = Field(default=None, max_length=40)


class RuntimeProfileApplyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resource_id: str = Field(min_length=28, max_length=40)
    actor: str = Field(default="operator", min_length=1, max_length=80)
    config_profile_id: str | None = Field(default=None, min_length=3, max_length=64)
    config_profile_version: int | None = Field(default=None, ge=1)


class RuntimeCleanupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resource_ids: list[str] = Field(min_length=1, max_length=100)
    confirm: bool = False
    actor: str = Field(default="operator", min_length=1, max_length=80)


class RuntimeBatchActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resource_ids: list[str] = Field(min_length=1, max_length=500)
    confirm: bool = False
    acknowledged: bool = False
    # Deprecated: retained for old clients; acknowledgement is a checkbox.
    confirmation_text: str = Field(default="", max_length=160)
    preview_token: str = Field(default="", max_length=200)
    actor: str = Field(default="local_operator", min_length=1, max_length=80)
    reason: str = Field(default="", max_length=240)
    current_profile_action: Literal["cancel", "switch", "restore_default", "pause"] = "cancel"
    replacement_resource_id: str | None = Field(default=None, max_length=40)


class RuntimeConfigurationBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    configuration_ids: list[str] = Field(min_length=1, max_length=500)
    confirm: bool = False
    acknowledged: bool = False
    # Deprecated: retained for old clients; acknowledgement is a checkbox.
    confirmation_text: str = Field(default="", max_length=160)
    preview_token: str = Field(default="", max_length=200)
    actor: str = Field(default="local_operator", min_length=1, max_length=80)
    reason: str = Field(default="", max_length=240)
    current_profile_action: Literal["cancel", "switch", "restore_default", "pause"] = "cancel"
    replacement_resource_id: str | None = Field(default=None, max_length=40)


class RuntimeClearAllRequest(BaseModel):
    """Confirm an immutable, server-generated target-domain cleanup plan."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(min_length=10, max_length=80)
    preview_token: str = Field(min_length=20, max_length=200)
    confirmation_text: str = Field(default="", max_length=80)
    acknowledged: bool = False
    actor: str = Field(default="local_operator", min_length=1, max_length=80)


class VideoJobCreateRequest(BaseModel):
    upload_id: str = Field(min_length=1, max_length=128)
    profile_id: str | None = Field(default=None, max_length=128)
    runtime_revision: int | None = Field(default=None, ge=0)
    model_id: str | None = None
    dataset: str | None = None
    category: str | None = None
    device: Literal["auto", "cpu", "cuda"] | None = None
    threshold: float | None = Field(default=None, ge=0.0)
    memory_bank_id: str | None = Field(default=None, max_length=128)
    config_profile_id: str | None = Field(default=None, max_length=64)
    config_profile_version: int | None = Field(default=None, ge=1)
    mode: Literal["inference_only"] = "inference_only"
    generate_visualizations: bool = True
    frame_stride: int = Field(default=1, ge=1, le=120)
    save_heatmap: bool = False
    save_mask: bool = False
    save_jsonl: bool = True


class HistoryManagementFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_time: str | None = Field(default=None, max_length=64)
    end_time: str | None = Field(default=None, max_length=64)
    mode: Literal["all", "single", "batch", "video"] = "all"
    status: Literal["all", "success", "failed", "cancelled", "running"] = "all"
    prediction: Literal["all", "normal", "anomaly", "failed", "unknown"] = "all"
    feedback_state: Literal["all", "has_feedback", "no_feedback"] = "all"
    visuals_state: Literal["all", "generated", "not_generated"] = "all"
    dataset: str | None = Field(default=None, max_length=80)
    category: str | None = Field(default=None, max_length=120)
    config: str | None = Field(default=None, max_length=260)
    config_version: str | None = Field(default=None, max_length=80)
    method_mode: str | None = Field(default=None, max_length=120)
    profile_id: str | None = Field(default=None, max_length=128)
    model_version: str | None = Field(default=None, max_length=128)
    memory_bank_id: str | None = Field(default=None, max_length=128)
    filename_keyword: str | None = Field(default=None, max_length=200)
    task_id: str | None = Field(default=None, max_length=160)
    score_min: float | None = Field(default=None, ge=0.0)
    score_max: float | None = Field(default=None, ge=0.0)


class HistoryDeletePreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filters: HistoryManagementFilter = Field(default_factory=HistoryManagementFilter)
    scope: Literal["selected", "current_filtered"] = "current_filtered"
    record_keys: list[str] = Field(default_factory=list, max_length=500)
    delete_feedback: bool = False
    cleanup_files: bool = True


class HistoryDeleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(min_length=20, max_length=96)
    preview_token: str = Field(min_length=20, max_length=200)
    confirmation_text: str = Field(default="", max_length=80)
    acknowledged: bool = False


class HistoryTaskItemsDeleteRequest(BaseModel):
    """Delete selected persisted children from one completed batch task."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["batch"] = "batch"
    item_ids: list[str] = Field(min_length=1, max_length=500)
    confirmation_text: str = Field(default="", max_length=80)
    acknowledged: bool = False


class HistoryAllDeletePreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filters: HistoryManagementFilter = Field(default_factory=HistoryManagementFilter)
    delete_feedback: bool = False
    cleanup_files: bool = True


class HistoryAllDeleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(min_length=20, max_length=96)
    preview_token: str = Field(min_length=20, max_length=200)
    confirmation_text: str = Field(default="", max_length=80)
    acknowledged: bool = False


class HistoryCleanupRetryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cleanup_ids: list[str] = Field(default_factory=list, max_length=100)
