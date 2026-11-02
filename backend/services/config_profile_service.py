from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from backend.errors import ApiError
from core.domain_build_contract import (
    DEFAULT_CREATE_MODE,
    LIGHTWEIGHT_CREATE_MODE,
    PRECISION_MODES,
    READ_COMPATIBLE_MODES,
    default_feature_layers,
)


PROFILE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")
NUMBERED_CONFIG_NAME_PATTERN = re.compile(r"^配置\s*(\d+)$")
SUPPORTED_BACKBONES = {"wide_resnet50_2", "resnet50"}
SUPPORTED_LAYERS = {"layer1", "layer2", "layer3", "layer4"}
SUPPORTED_THRESHOLD_METHODS = {"normal_percentile", "mean_std", "best_f1", "recall_priority"}
SUPPORTED_METHOD_MODES = READ_COMPATIBLE_MODES
HASH_KEYS = (
    "dataset",
    "category",
    "model_id",
    "model_version",
    "model_type",
    "base_model_profile_sha256",
    "base_model_weights_sha256",
    "base_model_manifest_sha256",
    "backbone",
    "layers",
    "image_size",
    "use_fp16",
    "precision",
    "requested_precision",
    "actual_precision",
    "amp_enabled",
    "feature_dtype",
    "memory_bank_dtype",
    "index_dtype",
    "precision_fallback_reason",
    "coreset_ratio",
    "max_memory_patches",
    "knn_k",
    "distance_metric",
    "memory_backend",
    "memory_backend_fallback",
    "memory_ivf_nlist",
    "memory_ivf_nprobe",
    "score_aggregation",
    "batch_size",
    "threshold_method",
    "threshold_percentile",
    "threshold_target_recall",
    "threshold_max_false_positive_rate",
    "pixel_threshold_mode",
    "method_mode",
    "algorithm_mode",
    "normal_memory_enabled",
    "lightweight_calibration_enabled",
    "defect_guidance_enabled",
    "max_defect_prototypes",
    "defect_top_ratio",
    "defect_coreset_ratio",
    "defect_region_filter",
    "dual_fusion_strategy",
    "normal_floor_percentile",
    "min_defect_image_support",
    "lightweight_gate_percentile",
    "lightweight_l2",
    "lightweight_score_scale",
    "lightweight_pooling_candidates",
    "lightweight_cv_folds",
    "random_seed",
    "save_visuals",
    "save_original",
    "threshold",
    "inference_mode",
)
SNAPSHOT_EXTRA_KEYS = (
    "config_profile_id",
    "config_profile_version",
    "target_domain_version",
    "build_job_id",
    "method_mode_resolved",
    "algorithm_mode",
    "normal_memory_enabled",
    "lightweight_calibration_enabled",
    "defect_guidance_enabled",
    "pixel_threshold",
    "dual_memory_alpha",
    "lightweight_calibrator_path",
    "calibration_snapshot",
    "complexity_snapshot",
    "video_sampling_config",
    "memory_bank_write_policy",
    "test_data_update_allowed",
    "official_test_mode",
    "evaluation_mode",
    "calibration_result",
    "validation_result",
    "data_roles",
    "backbone_output_dtype",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_text(value: Any, label: str, maximum: int) -> str:
    text = str(value or "").strip()
    if not text or len(text) > maximum or any(ord(char) < 32 for char in text):
        raise ApiError("INVALID_CONFIG_PROFILE", f"{label}必须为 1 到 {maximum} 个可见字符。", 422)
    if any(token in text for token in ("/", "\\", "..")):
        raise ApiError("INVALID_CONFIG_PROFILE", f"{label}不能包含路径分隔符或相对路径符号。", 422)
    return text


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


class ConfigProfileService:
    """Versioned, server-controlled target-domain configuration snapshots."""

    def __init__(self, settings: Any, logger: Any):
        self.settings = settings
        self.logger = logger
        self.root = settings.config_profiles_root.resolve()
        self.archive_root = self.root / "_archived"
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self.archive_root.mkdir(parents=True, exist_ok=True)
        self._register_builtin_baseline()
        try:
            self.reconcile_target_snapshot_duplicates()
        except Exception as exc:
            self.logger.warning("config_snapshot_reconcile_failed: %s", exc)

    @staticmethod
    def config_hash(config: dict[str, Any]) -> str:
        canonical = {key: config.get(key) for key in HASH_KEYS if key in config}
        encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def feature_hash(config: dict[str, Any]) -> str:
        # Only fields that define the extracted embedding space belong here.
        # Query-time parameters (K, threshold, aggregation, device, video
        # sampling) may legitimately differ between configurations owned by
        # the same memory bank.
        keys = (
            "model_id",
            "model_version",
            "base_model_weights_sha256",
            "backbone",
            "layers",
            "image_size",
            "use_fp16",
        )
        encoded = json.dumps({key: config.get(key) for key in keys}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def snapshot_hash(config: dict[str, Any]) -> str:
        encoded = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _legacy_initial_name(name: Any) -> bool:
        """Recognise only names emitted by older automatic build flows."""
        text = str(name or "").strip()
        return (
            text in {"有效配置快照", "配置1", "初始化配置"}
            or text.endswith(" · 有效配置快照")
            or text.endswith(" / 有效配置快照")
        )

    def _visible_name(self, metadata: dict[str, Any]) -> str:
        raw = str(metadata.get("friendly_name") or "").strip() or "未命名配置"
        if self._legacy_initial_name(raw):
            return "初始化配置"
        if (
            raw.endswith("有效快照")
            and str(metadata.get("config_profile_id") or "").startswith("snapshot_")
            and str(metadata.get("source_kind") or "") in {"target_domain_snapshot", "memory_bank_config"}
        ):
            return "初始化配置"
        origin = str(metadata.get("config_origin") or "")
        parent_id = str(metadata.get("parent_config_id") or "")
        parent_version = int(metadata.get("parent_config_version") or 0)
        # Older feedback candidates used the target-domain name for the cloned
        # active config.  Follow the immutable parent once so their UI name is
        # recovered without rewriting historical metadata or custom names.
        if not origin and parent_id and parent_version and parent_id != str(metadata.get("config_profile_id") or ""):
            try:
                _, parent_metadata = self._read(parent_id, parent_version)
            except (ApiError, OSError, ValueError):
                parent_metadata = {}
            parent_raw = str(parent_metadata.get("friendly_name") or "").strip()
            if self._legacy_initial_name(parent_raw):
                return "初始化配置"
        return raw

    def _source_label(self, metadata: dict[str, Any]) -> str:
        origin = str(metadata.get("config_origin") or "")
        build_job_id = str(metadata.get("build_job_id") or "")
        if origin == "feedback_inherited" or build_job_id.startswith("feedback_optimization_"):
            return "反馈更新继承"
        if origin == "user_created" or str(metadata.get("config_profile_id") or "").startswith("mbcfg_"):
            return "用户新增"
        parent_id = str(metadata.get("parent_config_id") or "")
        parent_version = int(metadata.get("parent_config_version") or 0)
        current_owner = str(metadata.get("memory_bank_id") or metadata.get("target_domain_version") or "")
        if parent_id and parent_version and parent_id != str(metadata.get("config_profile_id") or ""):
            try:
                _, parent_metadata = self._read(parent_id, parent_version)
            except (ApiError, OSError, ValueError):
                parent_metadata = {}
            parent_owner = str(parent_metadata.get("memory_bank_id") or parent_metadata.get("target_domain_version") or "")
            if parent_owner and current_owner and parent_owner != current_owner:
                return "反馈更新继承"
        return "目标域初始化"

    def _profile_dir(self, profile_id: str) -> Path:
        if not PROFILE_ID_PATTERN.fullmatch(str(profile_id)):
            raise ApiError("INVALID_CONFIG_PROFILE_ID", "配置方案标识无效。", 422)
        path = (self.root / profile_id).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ApiError("INVALID_CONFIG_PROFILE_ID", "配置方案标识无效。", 422) from exc
        return path

    def _version_dir(self, profile_id: str, version: int) -> Path:
        if int(version) < 1:
            raise ApiError("INVALID_CONFIG_PROFILE_VERSION", "配置方案版本无效。", 422)
        return self._profile_dir(profile_id) / f"v{int(version):04d}"

    @property
    def _archive_registry_path(self) -> Path:
        return self.archive_root / "registry.json"

    def _archive_registry(self) -> dict[str, Any]:
        try:
            payload = json.loads(self._archive_registry_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {"archived": {}, "deleted": {}}
        except (OSError, json.JSONDecodeError):
            return {"archived": {}, "deleted": {}}

    def _write_archive_registry(self, payload: dict[str, Any]) -> None:
        _atomic_text(self._archive_registry_path, json.dumps(payload, ensure_ascii=False, indent=2))

    @staticmethod
    def _archive_key(profile_id: str, version: int) -> str:
        return f"{profile_id}:v{int(version)}"

    def _read(self, profile_id: str, version: int) -> tuple[dict[str, Any], dict[str, Any]]:
        directory = self._version_dir(profile_id, version)
        profile_path = directory / "profile.yaml"
        metadata_path = directory / "metadata.json"
        if not profile_path.is_file() or not metadata_path.is_file():
            raise ApiError("CONFIG_PROFILE_NOT_FOUND", "配置方案不存在或已归档。", 404)
        try:
            profile = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError) as exc:
            raise ApiError("CONFIG_PROFILE_INVALID", "配置方案文件无法读取。", 500) from exc
        if not isinstance(profile, dict) or not isinstance(metadata, dict):
            raise ApiError("CONFIG_PROFILE_INVALID", "配置方案结构无效。", 500)
        return profile, metadata

    def _write(self, profile_id: str, version: int, profile: dict[str, Any], metadata: dict[str, Any]) -> None:
        directory = self._version_dir(profile_id, version)
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_text(directory / "profile.yaml", yaml.safe_dump(profile, allow_unicode=True, sort_keys=False))
        _atomic_text(directory / "metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))

    def _base_config(self, model_id: str | None = None) -> tuple[str, dict[str, Any], dict[str, Any]]:
        try:
            resolved_model_id, definition = self.settings.model_definition(model_id)
            profile = self.settings.algorithm_profile_unbound(resolved_model_id)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise ApiError("MODEL_NOT_FOUND", "所选异常检测模型配置不存在。", 422) from exc
        return resolved_model_id, definition, profile

    def normalize(self, values: dict[str, Any], *, base: dict[str, Any] | None = None) -> dict[str, Any]:
        incoming = dict(values or {})
        requested_model = str(incoming.get("model_id") or (base or {}).get("model_id") or self.settings.defaults.get("model_id") or "")
        model_id, definition, default_profile = self._base_config(requested_model)
        merged = {**default_profile, **dict(base or {}), **incoming}
        dataset = _safe_text(merged.get("dataset") or "custom", "数据集或项目名称", 80).lower()
        category = _safe_text(merged.get("category") or "custom", "类别名称", 120)
        device = str(merged.get("device") or "auto").lower()
        if device not in {"auto", "cpu", "cuda"}:
            raise ApiError("INVALID_CONFIG_PROFILE", "运行设备只能为 auto、cpu 或 cuda。", 422)
        backbone = str(merged.get("backbone") or "wide_resnet50_2")
        if backbone not in SUPPORTED_BACKBONES:
            raise ApiError("INVALID_CONFIG_PROFILE", "当前代码仅支持 wide_resnet50_2 或 resnet50 骨干网络。", 422)
        raw_layers = merged.get("layers", merged.get("feature_layers", default_feature_layers()))
        layers = [item.strip() for item in (raw_layers.split(",") if isinstance(raw_layers, str) else list(raw_layers or [])) if str(item).strip()]
        if not layers or len(set(layers)) != len(layers) or any(layer not in SUPPORTED_LAYERS for layer in layers):
            raise ApiError("INVALID_CONFIG_PROFILE", "特征层必须从 layer1 至 layer4 中选择，且不能重复。", 422)
        image_size = int(merged.get("image_size", 256))
        coreset_ratio = float(merged.get("coreset_ratio", 0.1))
        max_memory_patches = int(merged.get("max_memory_patches", 50000))
        knn_k = int(merged.get("knn_k", 1))
        batch_size = int(merged.get("batch_size", 1))
        random_seed = int(merged.get("random_seed", 42))
        threshold_method = str(merged.get("threshold_method") or "best_f1")
        threshold_percentile = float(merged.get("threshold_percentile", 99.0))
        threshold_target_recall = float(merged.get("threshold_target_recall", 0.95))
        threshold_max_false_positive_rate = float(merged.get("threshold_max_false_positive_rate", 0.10))
        # The packaged model profile is a read-compatible historical baseline,
        # not the policy default for a newly created target-domain Config.
        # Explicit input and an inherited Memory Bank Config keep their mode;
        # otherwise new configuration construction follows the formal default.
        method_mode = str(
            incoming.get("method_mode")
            or (base or {}).get("method_mode")
            or DEFAULT_CREATE_MODE
        ).lower()
        max_defect_prototypes = int(merged.get("max_defect_prototypes", 32))
        defect_top_ratio = float(merged.get("defect_top_ratio", 0.05))
        normal_floor_percentile = float(merged.get("normal_floor_percentile", 99.0))
        min_defect_image_support = int(merged.get("min_defect_image_support", 2))
        lightweight_gate_percentile = float(merged.get("lightweight_gate_percentile", 95.0))
        lightweight_l2 = float(merged.get("lightweight_l2", 0.01))
        lightweight_score_scale = float(merged.get("lightweight_score_scale", 1.0))
        lightweight_cv_folds = int(merged.get("lightweight_cv_folds", 5))
        raw_pooling_candidates = merged.get("lightweight_pooling_candidates", [0.005, 0.01, 0.02, 0.05])
        if isinstance(raw_pooling_candidates, str):
            raw_pooling_candidates = [item.strip() for item in raw_pooling_candidates.split(",") if item.strip()]
        lightweight_pooling_candidates = [float(item) for item in list(raw_pooling_candidates or [])]
        memory_backend = str(merged.get("memory_backend") or "auto").lower()
        memory_backend_fallback = str(merged.get("memory_backend_fallback") or "faiss_cpu_flat").lower()
        memory_ivf_nlist = int(merged.get("memory_ivf_nlist", 256))
        memory_ivf_nprobe = int(merged.get("memory_ivf_nprobe", 32))
        score_aggregation = str(merged.get("score_aggregation") or "maximum_patch").lower()
        pixel_threshold_mode = str(merged.get("pixel_threshold_mode") or "automatic_from_normal_map").lower()
        requested_precision = str(
            merged.get("requested_precision")
            or merged.get("precision")
            or ("fp16" if bool(merged.get("use_fp16", False)) else "fp32")
        ).lower()
        actual_precision = str(
            merged.get("actual_precision")
            or ("fp16" if bool(merged.get("use_fp16", False)) else "fp32")
        ).lower()
        if requested_precision not in PRECISION_MODES or actual_precision not in PRECISION_MODES:
            raise ApiError("INVALID_CONFIG_PROFILE", "构建精度只支持 fp16 或 fp32。", 422)
        amp_enabled = bool(merged.get("amp_enabled", actual_precision == "fp16"))
        if amp_enabled != (actual_precision == "fp16"):
            raise ApiError("INVALID_CONFIG_PROFILE", "实际构建精度与AMP执行状态不一致。", 422)
        defect_coreset_ratio = float(merged.get("defect_coreset_ratio", 0.25))
        defect_region_filter = str(merged.get("defect_region_filter") or "high_anomaly_patches").lower()
        dual_fusion_strategy = str(merged.get("dual_fusion_strategy") or "reliability_gated_relative_distance").lower()
        if not 64 <= image_size <= 2048:
            raise ApiError("INVALID_CONFIG_PROFILE", "输入尺寸必须在 64 到 2048 之间。", 422)
        if not 0.001 <= coreset_ratio <= 1.0 or not 1 <= max_memory_patches <= 200000:
            raise ApiError("INVALID_CONFIG_PROFILE", "coreset 参数超出允许范围。", 422)
        if not 1 <= knn_k <= 64 or not 1 <= batch_size <= 32 or random_seed < 0:
            raise ApiError("INVALID_CONFIG_PROFILE", "最近邻、批处理大小或随机种子无效。", 422)
        if threshold_method not in SUPPORTED_THRESHOLD_METHODS or not 50.0 <= threshold_percentile <= 100.0:
            raise ApiError("INVALID_CONFIG_PROFILE", "阈值校准方法或正常样本分位数无效。", 422)
        if not 0.0 <= threshold_target_recall <= 1.0 or not 0.0 <= threshold_max_false_positive_rate <= 1.0:
            raise ApiError("INVALID_CONFIG_PROFILE", "召回率下限或误检率上限无效。", 422)
        if method_mode not in SUPPORTED_METHOD_MODES:
            raise ApiError("INVALID_CONFIG_PROFILE", "异常检测方法模式无效。", 422)
        if not 1 <= max_defect_prototypes <= 128 or not 0.005 <= defect_top_ratio <= 0.25:
            raise ApiError("INVALID_CONFIG_PROFILE", "缺陷原型数量或补丁挖掘比例超出允许范围。", 422)
        if not 90.0 <= normal_floor_percentile <= 99.9 or not 1 <= min_defect_image_support <= 30:
            raise ApiError("INVALID_CONFIG_PROFILE", "原型可靠性参数超出允许范围。", 422)
        if not 90.0 <= lightweight_gate_percentile <= 99.9 or not 0.0 <= lightweight_l2 <= 10.0 or not 0.0 < lightweight_score_scale <= 100.0:
            raise ApiError("INVALID_CONFIG_PROFILE", "轻量校准参数超出允许范围。", 422)
        if not 2 <= lightweight_cv_folds <= 10 or not lightweight_pooling_candidates or any(not 0.001 <= value <= 0.25 for value in lightweight_pooling_candidates):
            raise ApiError("INVALID_CONFIG_PROFILE", "轻量校准候选比例或交叉验证折数无效。", 422)
        if memory_backend not in {"auto", "faiss_gpu_flat", "faiss_gpu_ivf", "faiss_cpu_flat", "sklearn", "torch_gpu_flat"}:
            raise ApiError("INVALID_CONFIG_PROFILE", "近邻检索后端无效。", 422)
        if memory_backend_fallback not in {"faiss_cpu_flat", "sklearn", "torch_gpu_flat"}:
            raise ApiError("INVALID_CONFIG_PROFILE", "近邻检索回退后端无效。", 422)
        if not 1 <= memory_ivf_nlist <= 65536 or not 1 <= memory_ivf_nprobe <= 65536:
            raise ApiError("INVALID_CONFIG_PROFILE", "FAISS IVF 聚类数或探测数无效。", 422)
        if score_aggregation not in {"maximum_patch", "top_k_mean"}:
            raise ApiError("INVALID_CONFIG_PROFILE", "异常分数聚合方式无效。", 422)
        if pixel_threshold_mode != "automatic_from_normal_map":
            raise ApiError("INVALID_CONFIG_PROFILE", "当前只支持根据正常热力图自动估计像素级阈值。", 422)
        if not 0.01 <= defect_coreset_ratio <= 1.0:
            raise ApiError("INVALID_CONFIG_PROFILE", "缺陷原型 coreset 比例无效。", 422)
        if defect_region_filter not in {"high_anomaly_patches", "reviewed_mask_preferred"} or dual_fusion_strategy != "reliability_gated_relative_distance":
            raise ApiError("INVALID_CONFIG_PROFILE", "缺陷区域筛选或双记忆融合策略无效。", 422)
        model_version = str(merged.get("model_version") or definition.get("version") or self.settings.backend.get("version") or "1.0.0")
        model_digests = {
            key: str(merged.get(key) or "").lower()
            for key in (
                "base_model_profile_sha256",
                "base_model_weights_sha256",
                "base_model_manifest_sha256",
            )
        }
        if any(value and not re.fullmatch(r"[0-9a-f]{64}", value) for value in model_digests.values()):
            raise ApiError("INVALID_CONFIG_PROFILE", "基础特征资源校验摘要无效。", 422)
        return {
            "dataset": dataset,
            "category": category,
            "model_id": model_id,
            "model_version": model_version,
            "model_type": str(definition.get("model_type") or "patchcore"),
            "base_model_source": str(merged.get("base_model_source") or "system_official_model_registry"),
            "base_model_profile_sha256": model_digests["base_model_profile_sha256"],
            "base_model_weights_sha256": model_digests["base_model_weights_sha256"],
            "base_model_manifest_sha256": model_digests["base_model_manifest_sha256"],
            "base_model_integrity_status": str(merged.get("base_model_integrity_status") or "unknown"),
            "device": device,
            "random_seed": random_seed,
            "backbone": backbone,
            "layers": layers,
            "image_size": image_size,
            "precision": requested_precision,
            "requested_precision": requested_precision,
            "actual_precision": actual_precision,
            "use_fp16": actual_precision == "fp16",
            "amp_enabled": amp_enabled,
            "feature_dtype": str(merged.get("feature_dtype") or "unknown"),
            "backbone_output_dtype": str(merged.get("backbone_output_dtype") or "unknown"),
            "memory_bank_dtype": str(merged.get("memory_bank_dtype") or "unknown"),
            "index_dtype": str(merged.get("index_dtype") or "unknown"),
            "precision_fallback_reason": str(merged.get("precision_fallback_reason") or ""),
            "embedding_dimension": "由骨干网络和特征层自动确定",
            "coreset_ratio": coreset_ratio,
            "max_memory_patches": max_memory_patches,
            "knn_k": knn_k,
            "distance_metric": "euclidean",
            "memory_backend": memory_backend,
            "memory_backend_fallback": memory_backend_fallback,
            "memory_ivf_nlist": memory_ivf_nlist,
            "memory_ivf_nprobe": memory_ivf_nprobe,
            "score_aggregation": score_aggregation,
            "batch_size": batch_size,
            "threshold_method": threshold_method,
            "threshold_percentile": threshold_percentile,
            "threshold_target_recall": threshold_target_recall,
            "threshold_max_false_positive_rate": threshold_max_false_positive_rate,
            "pixel_threshold_mode": pixel_threshold_mode,
            "method_mode": method_mode,
            "algorithm_mode": str(merged.get("algorithm_mode") or method_mode),
            "normal_memory_enabled": bool(merged.get("normal_memory_enabled", True)),
            "lightweight_calibration_enabled": bool(
                merged.get("lightweight_calibration_enabled", method_mode == LIGHTWEIGHT_CREATE_MODE)
            ),
            "defect_guidance_enabled": bool(
                merged.get("defect_guidance_enabled", method_mode == LIGHTWEIGHT_CREATE_MODE)
            ),
            "max_defect_prototypes": max_defect_prototypes,
            "defect_top_ratio": defect_top_ratio,
            "defect_coreset_ratio": defect_coreset_ratio,
            "defect_region_filter": defect_region_filter,
            "dual_fusion_strategy": dual_fusion_strategy,
            "normal_floor_percentile": normal_floor_percentile,
            "min_defect_image_support": min_defect_image_support,
            "lightweight_gate_percentile": lightweight_gate_percentile,
            "lightweight_l2": lightweight_l2,
            "lightweight_score_scale": lightweight_score_scale,
            "lightweight_pooling_candidates": lightweight_pooling_candidates,
            "lightweight_cv_folds": lightweight_cv_folds,
            "save_visuals": bool(merged.get("save_visuals", not bool(merged.get("no_save_visuals", False)))),
            "save_original": bool(merged.get("save_original", True)),
            "save_numeric_results": True,
        }

    def _public(self, profile: dict[str, Any], metadata: dict[str, Any], *, include_config: bool = False) -> dict[str, Any]:
        stored_name = str(metadata.get("friendly_name") or "")
        visible_name = self._visible_name(metadata)
        result = {
            "config_profile_id": metadata.get("config_profile_id"),
            "version": metadata.get("version"),
            "friendly_name": visible_name,
            "display_name": visible_name,
            "stored_friendly_name": stored_name,
            "dataset": profile.get("dataset"),
            "category": profile.get("category"),
            "model_id": profile.get("model_id"),
            "model_version": profile.get("model_version"),
            "model_type": profile.get("model_type"),
            "backbone": profile.get("backbone"),
            "created_at": metadata.get("created_at"),
            "updated_at": metadata.get("updated_at"),
            "status": metadata.get("status", "draft"),
            "validated": bool(metadata.get("validated")),
            "frozen": bool(metadata.get("frozen")),
            "applied": bool(metadata.get("applied")),
            "source_kind": metadata.get("source_kind", "custom"),
            "config_origin": metadata.get("config_origin") or "",
            "source_label": self._source_label(metadata),
            "memory_bank_id": metadata.get("memory_bank_id") or "",
            "threshold": metadata.get("threshold"),
            "config_hash": metadata.get("config_hash"),
            "feature_hash": metadata.get("feature_hash"),
            "note": metadata.get("note") or "",
            "parent_config_id": metadata.get("parent_config_id"),
            "parent_config_version": metadata.get("parent_config_version"),
            "copied_from_version": metadata.get("copied_from_version"),
            "config_snapshot_id": metadata.get("config_snapshot_id") or metadata.get("config_profile_id"),
            "config_snapshot_hash": metadata.get("config_snapshot_hash") or "",
            "target_domain_version": metadata.get("target_domain_version") or "",
            "build_job_id": metadata.get("build_job_id") or "",
            "build_idempotency_key": metadata.get("build_idempotency_key") or "",
            "integrity_status": metadata.get("integrity_status") or ("normal" if metadata.get("validated") else "pending"),
            "ordinary_visible": bool(metadata.get("ordinary_visible", True)),
            "merged_into": metadata.get("merged_into") or "",
            "merged_from": list(metadata.get("merged_from") or []),
            "parameter_count": len([key for key in HASH_KEYS if key in profile]),
        }
        if include_config:
            result["config"] = deepcopy(profile)
            result["yaml_text"] = yaml.safe_dump(profile, allow_unicode=True, sort_keys=False)
        return result

    def list_profiles(
        self,
        dataset: str | None = None,
        category: str | None = None,
        *,
        include_archived: bool = False,
        include_hidden: bool = False,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for profile_dir in sorted(self.root.iterdir()) if self.root.exists() else []:
            if not profile_dir.is_dir() or profile_dir.name.startswith("_"):
                continue
            for version_dir in sorted(profile_dir.glob("v[0-9][0-9][0-9][0-9]")):
                try:
                    version = int(version_dir.name[1:])
                    profile, metadata = self._read(profile_dir.name, version)
                except (ApiError, ValueError):
                    continue
                if dataset and str(profile.get("dataset", "")).lower() != str(dataset).lower():
                    continue
                if category and str(profile.get("category")) != str(category):
                    continue
                item = self._public(profile, metadata)
                if not include_hidden and not item.get("ordinary_visible", True):
                    continue
                rows.append(item)
        rows.sort(key=lambda item: (str(item.get("updated_at") or ""), int(item.get("version") or 0)), reverse=True)
        if include_archived:
            # The immutable built-in baseline may be recreated on startup
            # after an older copy was archived.  Both copies have the same
            # logical (profile_id, version); exposing both lets the later
            # archived row overwrite the live row in the resource catalog.
            # Prefer the authoritative live version and only add archive
            # entries whose logical version is not currently present.
            live_keys = {
                (str(item.get("config_profile_id") or ""), int(item.get("version") or 0))
                for item in rows
            }
            archived = [
                item
                for item in self.list_archived_profiles(dataset=dataset, category=category)
                if (str(item.get("config_profile_id") or ""), int(item.get("version") or 0)) not in live_keys
            ]
            rows.extend(archived)
        return rows

    def list_archived_profiles(self, dataset: str | None = None, category: str | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        registry = self._archive_registry()
        for key, entry in (registry.get("archived") or {}).items():
            try:
                profile_id = str(entry.get("config_profile_id") or str(key).split(":v", 1)[0])
                version = int(entry.get("version") or str(key).rsplit(":v", 1)[-1])
                directory = self.archived_version_directory(profile_id, version)
                profile = yaml.safe_load((directory / "profile.yaml").read_text(encoding="utf-8")) or {}
                metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
                if dataset and str(profile.get("dataset") or "").lower() != str(dataset).lower():
                    continue
                if category and str(profile.get("category") or "") != str(category):
                    continue
                item = self._public(profile, metadata)
                item.update(
                    archived=True,
                    status="archived",
                    archive_name=directory.name,
                    archived_at=entry.get("archived_at"),
                    last_archived_at=entry.get("last_archived_at"),
                )
                rows.append(item)
            except (ApiError, OSError, ValueError, yaml.YAMLError, json.JSONDecodeError):
                continue
        rows.sort(key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""), reverse=True)
        return rows

    def get(self, profile_id: str, version: int) -> dict[str, Any]:
        profile, metadata = self._read(profile_id, version)
        return self._public(profile, metadata, include_config=True)

    def _next_version(self, profile_id: str) -> int:
        directory = self._profile_dir(profile_id)
        versions = [int(path.name[1:]) for path in directory.glob("v[0-9][0-9][0-9][0-9]") if path.is_dir()]
        return max(versions, default=0) + 1

    def create(self, values: dict[str, Any]) -> dict[str, Any]:
        friendly_name = _safe_text(values.get("friendly_name") or "新配置方案", "配置方案名称", 80)
        requested_id = str(values.get("config_profile_id") or "").strip().lower()
        profile_id = requested_id if requested_id else f"profile_{uuid.uuid4().hex[:16]}"
        if not PROFILE_ID_PATTERN.fullmatch(profile_id):
            raise ApiError("CONFIG_PROFILE_CONFLICT", "配置方案标识已存在或无效。", 409)
        raw_config = dict(values.get("config") or values)
        normalized = self.normalize(raw_config)
        # Immutable Memory Bank configurations also carry their runtime
        # threshold, resolved calibration mode and lineage snapshot fields.
        # ``normalize`` intentionally limits editable parameters, so restore
        # these server-owned values after validation instead of silently
        # dropping them while cloning a sibling configuration.
        for key in (*SNAPSHOT_EXTRA_KEYS, "threshold", "inference_mode"):
            if key in raw_config:
                normalized[key] = deepcopy(raw_config[key])
        now = _utc_now()
        metadata = {
            "config_profile_id": profile_id,
            "version": 1,
            "friendly_name": friendly_name,
            "created_at": now,
            "updated_at": now,
            "status": "draft",
            "validated": False,
            "frozen": False,
            "applied": False,
            "source_kind": str(values.get("source_kind") or "custom"),
            "memory_bank_id": "",
            "threshold": None,
            "config_hash": self.config_hash(normalized),
            "feature_hash": self.feature_hash(normalized),
            "note": str(values.get("note") or "")[:500],
            "config_snapshot_id": profile_id,
            "config_snapshot_hash": self.snapshot_hash(normalized),
            "target_domain_version": str(values.get("target_domain_version") or ""),
            "build_job_id": str(values.get("build_job_id") or ""),
            "build_idempotency_key": str(values.get("build_idempotency_key") or ""),
            "integrity_status": "pending",
            "ordinary_visible": bool(values.get("ordinary_visible", True)),
            "config_origin": str(values.get("config_origin") or ""),
        }
        with self._lock:
            if self._profile_dir(profile_id).exists():
                raise ApiError("CONFIG_PROFILE_CONFLICT", "配置方案标识已存在或无效。", 409)
            self._write(profile_id, 1, normalized, metadata)
        return self._public(normalized, metadata, include_config=True)

    def create_from_template(
        self,
        values: dict[str, Any],
        *,
        template_profile_id: str | None = None,
        template_version: int | None = None,
    ) -> dict[str, Any]:
        """Create one independent build snapshot while preserving template lineage.

        A template remains reusable.  Only the newly created target-domain
        configuration version is later bound to and frozen with a memory bank.
        """
        template_config: dict[str, Any] = {}
        if template_profile_id and template_version:
            template = self.get(str(template_profile_id), int(template_version))
            template_config = dict(template.get("config") or {})
        payload = dict(values)
        raw_config = dict(values.get("config") or values)
        payload["config"] = self.normalize(raw_config, base=template_config)
        for key in (*SNAPSHOT_EXTRA_KEYS, "threshold", "inference_mode"):
            if key in raw_config:
                payload["config"][key] = deepcopy(raw_config[key])
        requested_id = str(payload.get("config_profile_id") or "").strip().lower()
        if requested_id and self._profile_dir(requested_id).exists():
            existing = self.get(requested_id, 1)
            same_binding = (
                str(existing.get("target_domain_version") or "") == str(payload.get("target_domain_version") or "")
                and str(existing.get("build_job_id") or "") == str(payload.get("build_job_id") or "")
                and str(existing.get("config_hash") or "") == self.config_hash(payload["config"])
            )
            if same_binding:
                return existing
            raise ApiError("CONFIG_SNAPSHOT_BINDING_CONFLICT", "该目标域版本已存在不兼容的配置快照。", 409)
        created = self.create(payload)
        with self._lock:
            profile, metadata = self._read(str(created["config_profile_id"]), int(created["version"]))
            metadata.update(source_kind="target_domain_snapshot")
            if template_profile_id and template_version:
                metadata.update(
                    parent_config_id=str(template_profile_id),
                    parent_config_version=int(template_version),
                    copied_from_version=int(template_version),
                )
            self._write(str(created["config_profile_id"]), int(created["version"]), profile, metadata)
        return self.get(str(created["config_profile_id"]), int(created["version"]))

    def reconcile_target_snapshot_duplicates(self) -> dict[str, Any]:
        """Migrate legacy snapshots into memory-bank-owned config versions.

        Older releases assumed a 1:1 target/config relationship and hid every
        additional snapshot as a duplicate.  A memory bank now owns one or
        more immutable configuration versions, so all bound historical rows
        remain visible inside that bank.  No file is deleted by this migration.
        """
        rows = [
            item
            for item in self.list_profiles(include_archived=False, include_hidden=True)
            if str(item.get("source_kind") or "") == "target_domain_snapshot"
            and str(item.get("memory_bank_id") or item.get("target_domain_version") or "")
        ]
        groups: dict[str, list[dict[str, Any]]] = {}
        for item in rows:
            binding = str(item.get("memory_bank_id") or item.get("target_domain_version") or "")
            groups.setdefault(binding, []).append(item)
        migrated_count = 0
        with self._lock:
            for item in rows:
                profile_id = str(item["config_profile_id"])
                version = int(item["version"])
                profile, metadata = self._read(profile_id, version)
                before = json.dumps(metadata, ensure_ascii=False, sort_keys=True, default=str)
                binding = str(metadata.get("memory_bank_id") or metadata.get("target_domain_version") or "")
                metadata.setdefault("config_snapshot_id", profile_id)
                metadata.setdefault("config_snapshot_hash", self.snapshot_hash(profile))
                metadata["memory_bank_id"] = binding
                if not metadata.get("target_domain_version"):
                    metadata["target_domain_version"] = binding
                if not metadata.get("build_job_id"):
                    metadata["build_job_id"] = f"legacy:{binding}" if binding else ""
                metadata.setdefault("integrity_status", "normal" if metadata.get("validated") else "pending")
                metadata.update(
                    source_kind="memory_bank_config",
                    ordinary_visible=True,
                    merged_into="",
                    resource_parent_type="memory_bank",
                )
                if json.dumps(metadata, ensure_ascii=False, sort_keys=True, default=str) != before:
                    metadata["updated_at"] = _utc_now()
                    self._write(profile_id, version, profile, metadata)
                    migrated_count += 1
        return {
            "target_binding_count": len(groups),
            "migrated_snapshot_count": migrated_count,
            "duplicate_snapshot_count": 0,
            "physical_files_deleted": False,
        }

    def update(self, profile_id: str, version: int, values: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            current, metadata = self._read(profile_id, version)
            if metadata.get("frozen"):
                raise ApiError("FROZEN_CONFIG_PROFILE", "冻结配置不能原地修改，请复制为新版本。", 409)
            friendly = values.get("friendly_name")
            if friendly is not None:
                metadata["friendly_name"] = _safe_text(friendly, "配置方案名称", 80)
            metadata["note"] = str(values.get("note", metadata.get("note") or ""))[:500]
            normalized = self.normalize(dict(values.get("config") or values), base=current)
            metadata.update(
                updated_at=_utc_now(),
                status="draft",
                validated=False,
                config_hash=self.config_hash(normalized),
                feature_hash=self.feature_hash(normalized),
            )
            self._write(profile_id, version, normalized, metadata)
        return self._public(normalized, metadata, include_config=True)

    def validate(self, profile_id: str, version: int) -> dict[str, Any]:
        with self._lock:
            profile, metadata = self._read(profile_id, version)
            normalized = self.normalize(profile)
            if metadata.get("frozen"):
                # Frozen historical snapshots must be verified against the
                # exact payload that was persisted.  Normalising an older
                # snapshot now also supplies newer precision provenance
                # fields, which must not silently rewrite the immutable file
                # or make its original hash look corrupt.
                stored_hash = str(metadata.get("config_hash") or "")
                if stored_hash != self.config_hash(profile):
                    raise ApiError("FROZEN_CONFIG_INTEGRITY_FAILED", "冻结配置的哈希不一致，不能继续使用。", 409)
                return self._public(profile, metadata, include_config=True)
            metadata.update(
                updated_at=_utc_now(),
                status="frozen" if metadata.get("frozen") else "validated",
                validated=True,
                config_hash=self.config_hash(normalized),
                feature_hash=self.feature_hash(normalized),
            )
            self._write(profile_id, version, normalized, metadata)
        return self._public(normalized, metadata, include_config=True)

    def copy_version(self, profile_id: str, version: int, friendly_name: str | None = None) -> dict[str, Any]:
        with self._lock:
            profile, source = self._read(profile_id, version)
            new_version = self._next_version(profile_id)
            now = _utc_now()
            metadata = {
                **source,
                "version": new_version,
                "friendly_name": _safe_text(friendly_name or f"{source.get('friendly_name') or '配置方案'} · 版本 {new_version}", "配置方案名称", 80),
                "created_at": now,
                "updated_at": now,
                "status": "draft",
                "validated": False,
                "frozen": False,
                "applied": False,
                "source_kind": "copied_version",
                "memory_bank_id": "",
                "threshold": None,
                "parent_config_id": str(profile_id),
                "parent_config_version": int(version),
                "copied_from_version": int(version),
            }
            self._write(profile_id, new_version, profile, metadata)
        return self._public(profile, metadata, include_config=True)

    def archive_draft(self, profile_id: str, version: int) -> dict[str, Any]:
        with self._lock:
            _, metadata = self._read(profile_id, version)
            if metadata.get("frozen") or metadata.get("memory_bank_id") or metadata.get("applied"):
                raise ApiError("CONFIG_PROFILE_IN_USE", "已冻结、已绑定或已应用的配置不能删除。", 409)
        return self.archive_version(profile_id, version)

    def archive_version(self, profile_id: str, version: int) -> dict[str, Any]:
        with self._lock:
            profile, metadata = self._read(profile_id, version)
            source = self._version_dir(profile_id, version)
            target = self.archive_root / f"{profile_id}_v{version:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if target.exists():
                target = self.archive_root / f"{target.name}_{uuid.uuid4().hex[:8]}"
            shutil.move(str(source), str(target))
            registry = self._archive_registry()
            registry.setdefault("archived", {})[self._archive_key(profile_id, version)] = {
                "config_profile_id": profile_id,
                "version": int(version),
                "archive_name": target.name,
                "friendly_name": metadata.get("friendly_name") or "配置方案",
                "dataset": profile.get("dataset"),
                "category": profile.get("category"),
                "archived_at": _utc_now(),
            }
            self._write_archive_registry(registry)
            profile_root = self._profile_dir(profile_id)
            if profile_root.is_dir() and not any(profile_root.iterdir()):
                profile_root.rmdir()
        return {"config_profile_id": profile_id, "version": version, "archived": True, "archive_name": target.name}

    def archived_version_directory(self, profile_id: str, version: int) -> Path:
        if not PROFILE_ID_PATTERN.fullmatch(str(profile_id)) or int(version) < 1:
            raise ApiError("INVALID_CONFIG_PROFILE_ID", "配置方案标识无效。", 422)
        entry = (self._archive_registry().get("archived") or {}).get(self._archive_key(profile_id, version)) or {}
        archive_name = str(entry.get("archive_name") or "")
        if not archive_name:
            raise ApiError("CONFIG_PROFILE_ARCHIVE_NOT_FOUND", "归档配置不存在。", 404)
        path = (self.archive_root / archive_name).resolve()
        try:
            path.relative_to(self.archive_root.resolve())
        except ValueError as exc:
            raise ApiError("CONFIG_PROFILE_ARCHIVE_INVALID", "归档配置路径不受控。", 409) from exc
        if not path.is_dir() or path.is_symlink():
            raise ApiError("CONFIG_PROFILE_ARCHIVE_NOT_FOUND", "归档配置文件不存在。", 404)
        return path

    def restore_version(self, profile_id: str, version: int) -> dict[str, Any]:
        with self._lock:
            source = self.archived_version_directory(profile_id, version)
            target = self._version_dir(profile_id, version)
            if target.exists():
                raise ApiError("CONFIG_PROFILE_RESTORE_CONFLICT", "同版本配置已存在，不能恢复。", 409)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            registry = self._archive_registry()
            registry.setdefault("archived", {}).pop(self._archive_key(profile_id, version), None)
            self._write_archive_registry(registry)
            profile, metadata = self._read(profile_id, version)
        result = self._public(profile, metadata)
        result.update(archived=False, restored=True)
        return result

    def forget_archived(self, profile_id: str, version: int, operation_id: str, *, reason: str = "") -> dict[str, Any]:
        with self._lock:
            registry = self._archive_registry()
            entry = dict(registry.setdefault("archived", {}).pop(self._archive_key(profile_id, version), {}) or {})
            registry.setdefault("deleted", {})[self._archive_key(profile_id, version)] = {
                **entry,
                "config_profile_id": profile_id,
                "version": int(version),
                "deleted_at": _utc_now(),
                "operation_id": str(operation_id),
                "reason": str(reason or "")[:240],
            }
            self._write_archive_registry(registry)
        return {"config_profile_id": profile_id, "version": int(version), "deleted": True, "operation_id": operation_id}

    def import_yaml(self, yaml_text: str, friendly_name: str) -> dict[str, Any]:
        try:
            parsed = yaml.safe_load(str(yaml_text or ""))
        except yaml.YAMLError as exc:
            raise ApiError("INVALID_PROFILE_YAML", "YAML 解析失败，请检查缩进和字段格式。", 422) from exc
        if not isinstance(parsed, dict):
            raise ApiError("INVALID_PROFILE_YAML", "导入的 YAML 必须是配置映射。", 422)
        return self.create({"friendly_name": friendly_name or "导入的配置方案", "config": parsed, "source_kind": "imported_yaml"})

    def frozen_snapshot(self, profile_id: str, version: int, effective: dict[str, Any], threshold: float) -> tuple[dict[str, Any], str]:
        profile, metadata = self._read(profile_id, version)
        if metadata.get("frozen"):
            raise ApiError("FROZEN_CONFIG_PROFILE", "冻结配置不能重复构建，请复制为新版本。", 409)
        if not metadata.get("validated"):
            raise ApiError("CONFIG_PROFILE_NOT_VALIDATED", "配置方案必须先验证，再生成记忆库。", 409)
        normalized = self.normalize(effective, base=profile)
        if self.feature_hash(normalized) != metadata.get("feature_hash"):
            raise ApiError("CONFIG_PROFILE_CHANGED", "构建参数与已验证配置不一致，请重新保存并验证。", 409)
        normalized.update(threshold=float(threshold), inference_mode="inference_only")
        for key in SNAPSHOT_EXTRA_KEYS:
            if key in effective:
                normalized[key] = deepcopy(effective[key])
        return normalized, self.config_hash(normalized)

    def bind_and_freeze(
        self,
        profile_id: str,
        version: int,
        memory_bank_id: str,
        threshold: float,
        frozen_profile: dict[str, Any],
        config_hash: str,
    ) -> dict[str, Any]:
        with self._lock:
            _, metadata = self._read(profile_id, version)
            if metadata.get("frozen"):
                raise ApiError("FROZEN_CONFIG_PROFILE", "配置方案已经冻结。", 409)
            if self.config_hash(frozen_profile) != str(config_hash):
                raise ApiError("CONFIG_PROFILE_HASH_MISMATCH", "配置冻结哈希校验失败。", 409)
            expected_target = str(metadata.get("target_domain_version") or "")
            if expected_target and expected_target != str(memory_bank_id):
                raise ApiError("CONFIG_SNAPSHOT_BINDING_CONFLICT", "配置快照与目标域版本不一致，不能冻结。", 409)
            metadata.update(
                updated_at=_utc_now(),
                status="frozen",
                validated=True,
                frozen=True,
                memory_bank_id=str(memory_bank_id),
                threshold=float(threshold),
                config_hash=str(config_hash),
                feature_hash=self.feature_hash(frozen_profile),
                inference_mode="inference_only",
                config_snapshot_id=str(profile_id),
                config_snapshot_hash=self.snapshot_hash(frozen_profile),
                target_domain_version=str(memory_bank_id),
                integrity_status="normal",
                ordinary_visible=True,
                source_kind="memory_bank_config",
                resource_parent_type="memory_bank",
            )
            self._write(profile_id, version, frozen_profile, metadata)
        return self._public(frozen_profile, metadata, include_config=True)

    def compatibility(self, profile_id: str, version: int, memory_bank: dict[str, Any]) -> dict[str, Any]:
        profile, metadata = self._read(profile_id, version)
        reasons: list[str] = []
        for key in ("dataset", "category", "model_id"):
            if str(profile.get(key) or "").lower() != str(memory_bank.get(key) or "").lower():
                reasons.append(f"{key} 不匹配")
        if metadata.get("source_kind") != "builtin_public_baseline" and str(metadata.get("memory_bank_id") or "") != str(memory_bank.get("memory_bank_id") or ""):
            reasons.append("记忆库绑定关系不匹配")
        # Full config hashes intentionally differ between sibling configs.
        # Only the embedding-space contract must match the memory-bank build.
        for key in ("backbone", "image_size"):
            expected = memory_bank.get(key)
            if expected is not None and str(profile.get(key) or "") != str(expected):
                reasons.append(f"{key} 与记忆库特征空间不匹配")
        expected_layers = list(memory_bank.get("feature_layers") or memory_bank.get("layers") or [])
        if expected_layers and list(profile.get("layers") or []) != expected_layers:
            reasons.append("特征层与记忆库不匹配")
        if not metadata.get("frozen") or not memory_bank.get("frozen") or memory_bank.get("threshold") is None:
            reasons.append("配置、记忆库或阈值尚未冻结")
        return {"compatible": not reasons, "reasons": reasons, "config_profile": self._public(profile, metadata)}

    def list_for_memory_bank(self, memory_bank_id: str, *, include_archived: bool = False) -> list[dict[str, Any]]:
        binding = str(memory_bank_id or "").strip()
        if not binding:
            raise ApiError("MEMORY_BANK_REQUIRED", "必须指定记忆库。", 422)
        rows = [
            item for item in self.list_profiles(include_archived=include_archived, include_hidden=True)
            if str(item.get("memory_bank_id") or item.get("target_domain_version") or "") == binding
        ]
        rows.sort(key=lambda item: (bool(item.get("applied")), str(item.get("updated_at") or "")), reverse=True)
        return rows

    def next_memory_bank_config_name(self, memory_bank_id: str) -> str:
        """Return the next stable default label without reusing deleted numbers."""
        highest = 1
        for item in self.list_for_memory_bank(memory_bank_id, include_archived=True):
            name = str(item.get("friendly_name") or "").strip()
            if name == "初始化配置":
                highest = max(highest, 1)
                continue
            match = NUMBERED_CONFIG_NAME_PATTERN.fullmatch(name)
            if match:
                highest = max(highest, int(match.group(1)))
        return f"配置{highest + 1}"

    def clone_all_for_feedback_candidate(
        self,
        *,
        source_memory_bank_id: str,
        target_memory_bank_id: str,
        active_profile_id: str,
        active_version: int,
        active_effective_config: dict[str, Any],
        build_job_id: str,
        note: str = "",
    ) -> dict[str, Any]:
        """Clone every valid source config and update only the active branch.

        The returned configs are independent draft snapshots owned by the
        candidate Memory Bank.  Source files and hashes are never modified.
        """
        source_rows = self.list_for_memory_bank(source_memory_bank_id)
        source_keys = {
            (str(item.get("config_profile_id") or ""), int(item.get("version") or 0))
            for item in source_rows
        }
        active_key = (str(active_profile_id or ""), int(active_version or 0))
        if active_key[0] and active_key[1] and active_key not in source_keys:
            active = self.get(active_key[0], active_key[1])
            if str(active.get("memory_bank_id") or active.get("target_domain_version") or "") != str(source_memory_bank_id):
                raise ApiError("FEEDBACK_ACTIVE_CONFIG_MISMATCH", "当前反馈更新配置不属于基础记忆库。", 409)
            source_rows.append(active)
        if not source_rows or active_key not in {
            (str(item.get("config_profile_id") or ""), int(item.get("version") or 0))
            for item in source_rows
        }:
            raise ApiError("FEEDBACK_ACTIVE_CONFIG_MISSING", "基础记忆库缺少本次反馈所使用的配置。", 409)

        detailed: list[dict[str, Any]] = []
        active_feature_hash = ""
        for row in source_rows:
            detail = self.get(str(row["config_profile_id"]), int(row["version"]))
            key = (str(detail["config_profile_id"]), int(detail["version"]))
            if key == active_key:
                active_feature_hash = str(detail.get("feature_hash") or self.feature_hash(detail.get("config") or {}))
            detailed.append(detail)
        incompatible = [
            str(item.get("friendly_name") or "未命名配置")
            for item in detailed
            if str(item.get("feature_hash") or self.feature_hash(item.get("config") or {})) != active_feature_hash
        ]
        if incompatible:
            names = "、".join(incompatible[:5])
            raise ApiError(
                "FEEDBACK_CONFIG_INHERITANCE_INCOMPATIBLE",
                f"基础记忆库包含与当前特征空间不兼容的配置：{names}。请先处理这些配置，再创建反馈更新版本。",
                409,
            )

        # Put the active/default configuration first; keep remaining labels and
        # parameters unchanged so the target version is immediately usable.
        detailed.sort(
            key=lambda item: (
                (str(item.get("config_profile_id") or ""), int(item.get("version") or 0)) != active_key,
                str(item.get("created_at") or ""),
            )
        )
        created_rows: list[dict[str, Any]] = []
        for source in detailed:
            source_id = str(source["config_profile_id"])
            source_version = int(source["version"])
            is_active = (source_id, source_version) == active_key
            target_id = "snapshot_" + hashlib.sha256(
                f"{target_memory_bank_id}:{source_id}:v{source_version}".encode("utf-8")
            ).hexdigest()[:24]
            config = deepcopy(source.get("config") or {})
            if is_active:
                config.update(deepcopy(active_effective_config or {}))
            config.update(
                config_profile_id=target_id,
                config_profile_version=1,
                target_domain_version=target_memory_bank_id,
                build_job_id=build_job_id,
            )
            created = self.create_from_template(
                {
                    "config_profile_id": target_id,
                    "friendly_name": str(source.get("friendly_name") or "未命名配置")[:80],
                    "note": str(note or source.get("note") or "")[:500],
                    "source_kind": "target_domain_snapshot",
                    "config_origin": "feedback_inherited",
                    "config": config,
                    "target_domain_version": target_memory_bank_id,
                    "build_job_id": build_job_id,
                    "ordinary_visible": True,
                },
                template_profile_id=source_id,
                template_version=source_version,
            )
            self.validate(str(created["config_profile_id"]), int(created["version"]))
            created_rows.append(
                {
                    **created,
                    "source_config_profile_id": source_id,
                    "source_config_profile_version": source_version,
                    "source_config_hash": str(
                        source.get("config_hash") or self.config_hash(source.get("config") or {})
                    ),
                    "source_threshold": (
                        source.get("config", {}).get("threshold")
                        if source.get("config", {}).get("threshold") is not None
                        else source.get("threshold")
                    ),
                    "active": is_active,
                }
            )
        active_created = next(item for item in created_rows if item.get("active"))
        return {
            "configs": created_rows,
            "active_config": active_created,
            "config_count": len(created_rows),
            "inherited_config_names": [str(item.get("friendly_name") or "未命名配置") for item in created_rows],
        }

    def create_for_memory_bank(
        self,
        memory_bank: dict[str, Any],
        values: dict[str, Any],
        *,
        source_profile_id: str | None = None,
        source_version: int | None = None,
    ) -> dict[str, Any]:
        """Create one immutable config owned by ``memory_bank``."""
        memory_bank_id = str(memory_bank.get("memory_bank_id") or "")
        if not memory_bank_id or not memory_bank.get("frozen"):
            raise ApiError("MEMORY_BANK_NOT_READY", "只能为已冻结记忆库创建配置。", 409)
        base: dict[str, Any] = {}
        if source_profile_id and source_version:
            source = self.get(str(source_profile_id), int(source_version))
            if str(source.get("memory_bank_id") or "") != memory_bank_id:
                raise ApiError("CONFIG_MEMORY_BANK_MISMATCH", "基础配置不属于当前记忆库。", 409)
            base = dict(source.get("config") or {})
        elif memory_bank.get("config_profile_id") and memory_bank.get("config_profile_version"):
            source = self.get(str(memory_bank["config_profile_id"]), int(memory_bank["config_profile_version"]))
            base = dict(source.get("config") or {})
        incoming = dict(values.get("config") or {})
        # A config may tune inference/calibration behaviour, but cannot change
        # the embeddings already stored in its parent memory bank.
        for key, expected in {
            "dataset": memory_bank.get("dataset"),
            "category": memory_bank.get("category"),
            "model_id": memory_bank.get("model_id"),
            "backbone": memory_bank.get("backbone"),
            "layers": memory_bank.get("feature_layers") or memory_bank.get("layers"),
            "image_size": memory_bank.get("image_size"),
        }.items():
            if expected not in (None, "", []):
                incoming[key] = expected
        normalized = self.normalize(incoming, base=base)
        sampling = dict(incoming.get("video_sampling_config") or base.get("video_sampling_config") or {})
        if sampling:
            normalized["video_sampling_config"] = {
                "frame_interval_seconds": max(0.01, float(sampling.get("frame_interval_seconds") or 1.0)),
                "max_representative_frames": max(1, int(sampling.get("max_representative_frames") or 500)),
                "similarity_threshold": min(1.0, max(0.0, float(sampling.get("similarity_threshold") or 0.98))),
            }
        threshold = float(incoming.get("threshold", values.get("threshold", base.get("threshold", memory_bank.get("threshold")))))
        normalized.update(threshold=threshold, inference_mode="inference_only")
        profile_id = f"mbcfg_{uuid.uuid4().hex[:16]}"
        requested_name = str(values.get("friendly_name") or "").strip()
        if not requested_name or requested_name == "新增配置":
            requested_name = self.next_memory_bank_config_name(memory_bank_id)
        now = _utc_now()
        metadata = {
            "config_profile_id": profile_id,
            "version": 1,
            "friendly_name": _safe_text(requested_name, "配置名称", 80),
            "created_at": now,
            "updated_at": now,
            "status": "frozen",
            "validated": True,
            "frozen": True,
            "applied": False,
            "source_kind": "memory_bank_config",
            "config_origin": "user_created",
            "memory_bank_id": memory_bank_id,
            "target_domain_version": memory_bank_id,
            "threshold": threshold,
            "config_hash": self.config_hash(normalized),
            "feature_hash": self.feature_hash(normalized),
            "config_snapshot_id": profile_id,
            "config_snapshot_hash": self.snapshot_hash(normalized),
            "integrity_status": "normal",
            "ordinary_visible": True,
            "resource_parent_type": "memory_bank",
            "parent_config_id": str(source_profile_id or memory_bank.get("config_profile_id") or ""),
            "parent_config_version": int(source_version or memory_bank.get("config_profile_version") or 0),
            "note": str(values.get("note") or "")[:500],
        }
        with self._lock:
            self._write(profile_id, 1, normalized, metadata)
        return self._public(normalized, metadata, include_config=True)

    def revise_for_memory_bank(
        self,
        memory_bank: dict[str, Any],
        profile_id: str,
        version: int,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        source = self.get(profile_id, version)
        if str(source.get("memory_bank_id") or "") != str(memory_bank.get("memory_bank_id") or ""):
            raise ApiError("CONFIG_MEMORY_BANK_MISMATCH", "配置不属于当前记忆库。", 409)
        base = dict(source.get("config") or {})
        incoming = {**base, **dict(values.get("config") or {})}
        # Editing never overwrites a frozen version; it creates vN+1.
        for key in ("dataset", "category", "model_id", "backbone", "layers", "image_size"):
            incoming[key] = base.get(key)
        normalized = self.normalize(incoming, base=base)
        sampling = dict(incoming.get("video_sampling_config") or base.get("video_sampling_config") or {})
        if sampling:
            normalized["video_sampling_config"] = {
                "frame_interval_seconds": max(0.01, float(sampling.get("frame_interval_seconds") or 1.0)),
                "max_representative_frames": max(1, int(sampling.get("max_representative_frames") or 500)),
                "similarity_threshold": min(1.0, max(0.0, float(sampling.get("similarity_threshold") or 0.98))),
            }
        threshold = float(incoming.get("threshold", source.get("threshold") or memory_bank.get("threshold")))
        normalized.update(threshold=threshold, inference_mode="inference_only")
        with self._lock:
            new_version = self._next_version(profile_id)
            now = _utc_now()
            metadata = {
                **{key: value for key, value in source.items() if key not in {"config", "yaml_text"}},
                "config_profile_id": profile_id,
                "version": new_version,
                "friendly_name": _safe_text(values.get("friendly_name") or source.get("friendly_name") or "配置", "配置名称", 80),
                "created_at": now,
                "updated_at": now,
                "status": "frozen",
                "validated": True,
                "frozen": True,
                "applied": False,
                "source_kind": "memory_bank_config",
                "memory_bank_id": str(memory_bank.get("memory_bank_id") or ""),
                "target_domain_version": str(memory_bank.get("memory_bank_id") or ""),
                "threshold": threshold,
                "config_hash": self.config_hash(normalized),
                "feature_hash": self.feature_hash(normalized),
                "config_snapshot_id": profile_id,
                "config_snapshot_hash": self.snapshot_hash(normalized),
                "parent_config_id": profile_id,
                "parent_config_version": int(version),
                "copied_from_version": int(version),
                "ordinary_visible": True,
                "merged_into": "",
                "note": str(values.get("note", source.get("note") or ""))[:500],
            }
            self._write(profile_id, new_version, normalized, metadata)
        return self.get(profile_id, new_version)

    def remove_from_memory_bank(
        self,
        memory_bank_id: str,
        profile_id: str,
        version: int,
        *,
        current_profile_id: str = "",
        current_version: int = 0,
    ) -> dict[str, Any]:
        rows = [item for item in self.list_for_memory_bank(memory_bank_id) if not item.get("archived")]
        target = next((item for item in rows if str(item.get("config_profile_id")) == str(profile_id) and int(item.get("version") or 0) == int(version)), None)
        if target is None:
            raise ApiError("CONFIG_PROFILE_NOT_FOUND", "该配置不属于当前记忆库。", 404)
        if len(rows) <= 1:
            raise ApiError("LAST_MEMORY_BANK_CONFIG", "一个记忆库至少保留一个配置，当前唯一配置不能删除。", 409)
        if str(profile_id) == str(current_profile_id) and int(version) == int(current_version):
            raise ApiError("ACTIVE_MEMORY_BANK_CONFIG", "当前正在使用该配置，请先切换到同一记忆库的其他配置。", 409)
        result = self.archive_version(profile_id, version)
        result.update(memory_bank_id=memory_bank_id, logically_deleted=True)
        return result

    def mark_applied(self, profile_id: str, version: int, applied: bool = True) -> None:
        with self._lock:
            if applied:
                for profile_dir in self.root.iterdir():
                    if not profile_dir.is_dir() or profile_dir.name.startswith("_"):
                        continue
                    for version_dir in profile_dir.glob("v*"):
                        metadata_path = version_dir / "metadata.json"
                        profile_path = version_dir / "profile.yaml"
                        if not metadata_path.is_file() or not profile_path.is_file():
                            continue
                        other_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                        if not other_metadata.get("applied"):
                            continue
                        other_profile_id = str(other_metadata.get("profile_id") or profile_dir.name)
                        try:
                            other_version = int(other_metadata.get("version") or version_dir.name.removeprefix("v"))
                        except (TypeError, ValueError):
                            continue
                        is_current = (
                            other_profile_id == str(profile_id)
                            and other_version == int(version)
                        )
                        if is_current:
                            continue
                        other_profile = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
                        other_metadata.setdefault("profile_id", other_profile_id)
                        other_metadata.setdefault("version", other_version)
                        other_metadata["applied"] = False
                        other_metadata["updated_at"] = _utc_now()
                        self._write(
                            other_profile_id,
                            other_version,
                            other_profile,
                            other_metadata,
                        )
            profile, metadata = self._read(profile_id, version)
            metadata["applied"] = bool(applied)
            metadata["updated_at"] = _utc_now()
            self._write(profile_id, version, profile, metadata)

    def ensure_legacy_profile(self, memory_metadata: dict[str, Any], effective_config: dict[str, Any]) -> dict[str, Any]:
        memory_id = str(memory_metadata.get("memory_bank_id") or "")
        profile_id = f"legacy_{hashlib.sha256(memory_id.encode()).hexdigest()[:20]}"
        version = 1
        try:
            existing = self.get(profile_id, version)
            return existing
        except ApiError as exc:
            if exc.code != "CONFIG_PROFILE_NOT_FOUND":
                raise
        normalized = self.normalize(effective_config)
        normalized.update(threshold=float(memory_metadata.get("threshold")), inference_mode="inference_only")
        now = str(memory_metadata.get("created_at") or _utc_now())
        metadata = {
            "config_profile_id": profile_id,
            "version": version,
            "friendly_name": "旧版兼容配置",
            "created_at": now,
            "updated_at": now,
            "status": "frozen",
            "validated": True,
            "frozen": True,
            "applied": False,
            "source_kind": "legacy_compatible",
            "memory_bank_id": memory_id,
            "threshold": memory_metadata.get("threshold"),
            "config_hash": self.config_hash(normalized),
            "feature_hash": self.feature_hash(normalized),
            "note": "由旧版记忆库中已有的 effective_config.yaml 注册；未补造缺失参数。",
        }
        with self._lock:
            if not self._version_dir(profile_id, version).exists():
                self._write(profile_id, version, normalized, metadata)
        return self._public(normalized, metadata, include_config=True)

    def builtin_binding(
        self,
        memory_bank_id: str,
        threshold: float | None,
        *,
        model_id: str,
        dataset: str,
        category: str,
    ) -> dict[str, Any]:
        """Return an immutable config snapshot compatible with one built-in bank.

        The historical implementation reused the MVTec/bottle baseline for every
        built-in memory bank.  That made otherwise valid VisA and other built-in
        profiles fail the profile/config compatibility gate during an atomic
        switch.  A built-in bank now receives its own deterministic snapshot;
        the legacy baseline remains untouched for historical traceability.
        """
        profile_id = f"builtin_cfg_{hashlib.sha256(str(memory_bank_id).encode()).hexdigest()[:16]}"
        version = 1
        try:
            profile, metadata = self._read(profile_id, version)
        except ApiError as exc:
            if exc.code != "CONFIG_PROFILE_NOT_FOUND":
                raise
            _, _, base = self._base_config(model_id)
            profile = self.normalize(
                {
                    **base,
                    "model_id": model_id,
                    "dataset": str(dataset).lower(),
                    "category": category,
                    "method_mode": "normal_only",
                }
            )
            profile.update(
                threshold=float(threshold) if threshold is not None else None,
                inference_mode="inference_only",
            )
            now = _utc_now()
            metadata = {
                "config_profile_id": profile_id,
                "version": version,
                "friendly_name": f"内置基线配置 · {str(dataset).lower()} / {category}",
                "created_at": now,
                "updated_at": now,
                "status": "frozen",
                "validated": True,
                "frozen": True,
                "applied": False,
                "source_kind": "builtin_public_baseline",
                "memory_bank_id": str(memory_bank_id),
                "threshold": threshold,
                "config_hash": self.config_hash(profile),
                "feature_hash": self.feature_hash(profile),
                "note": "由内置公开数据记忆库自动注册的专属只读配置快照。",
            }
            with self._lock:
                if not self._version_dir(profile_id, version).exists():
                    self._write(profile_id, version, profile, metadata)
                else:
                    profile, metadata = self._read(profile_id, version)
        return self._public(profile, metadata, include_config=True)

    def _register_builtin_baseline(self) -> None:
        profile_id = "builtin_public_baseline"
        version = 1
        if self._version_dir(profile_id, version).exists():
            return
        model_id = "patchcore-mvtec-final" if "patchcore-mvtec-final" in self.settings.models else None
        _, _, base = self._base_config(model_id)
        normalized = self.normalize(base)
        now = _utc_now()
        metadata = {
            "config_profile_id": profile_id,
            "version": version,
            "friendly_name": "内置公开数据基线",
            "created_at": now,
            "updated_at": now,
            "status": "frozen",
            "validated": True,
            "frozen": True,
            "applied": False,
            "source_kind": "builtin_public_baseline",
            "memory_bank_id": "",
            "threshold": None,
            "config_hash": self.config_hash(normalized),
            "feature_hash": self.feature_hash(normalized),
            "note": "由 configs/final_balanced.yaml 注册，仅用于公开数据演示和开发。",
        }
        self._write(profile_id, version, normalized, metadata)
