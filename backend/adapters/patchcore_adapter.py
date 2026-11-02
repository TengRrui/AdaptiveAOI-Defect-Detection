from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, UnidentifiedImageError

from backend.adapters.result_serializer import artifact_url, json_safe, normalized_detection, structured_timing
from backend.config import IntegrationSettings
from backend.errors import ApiError
from backend.path_validation import validate_dataset_category, validate_resource_segment
from backend.services.detection_context import DetectionContext, detection_context_dict
from backend.services.job_service import JobPaths, JobService, utc_now
from backend.services.model_registry import ModelRegistry
from backend.services.result_storage import ResultStorage
from backend.services.runtime_registry import RuntimeRegistry


class PatchCoreAdapter:
    """Uses the existing PatchCore implementation without copying algorithm logic."""

    def __init__(
        self,
        settings: IntegrationSettings,
        jobs: JobService,
        storage: ResultStorage,
        registry: RuntimeRegistry,
        models: ModelRegistry,
        domain_profiles: Any,
        logger: Any,
    ):
        self.settings = settings
        self.jobs = jobs
        self.storage = storage
        self.registry = registry
        self.models = models
        self.domain_profiles = domain_profiles
        self.logger = logger

    def _resolve(
        self,
        model_id: str | None,
        dataset: str | None,
        category: str | None,
        device: str | None,
        threshold: float | None,
        save_visuals: bool | None,
        memory_bank_id: str | None = None,
        mode: str = "inference_only",
        config_profile_id: str | None = None,
        config_profile_version: int | None = None,
        config_hash: str | None = None,
        config_values: dict[str, Any] | None = None,
        detection_context: DetectionContext | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Reject supplied paths before resolving local models or dataset roots.
        if dataset is not None:
            validate_resource_segment(dataset, "dataset")
        if category is not None:
            validate_resource_segment(category, "category")
        try:
            resolved_model_id, definition = self.settings.model_definition(model_id)
            try:
                profile = self.settings.algorithm_profile(resolved_model_id)
            except ValueError:
                # Uploaded images and frozen target-domain profiles do not need a
                # dataset root. Dataset-backed CLI jobs still validate it later.
                profile = self.settings.algorithm_profile_unbound(resolved_model_id)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise ApiError("MODEL_NOT_FOUND", str(exc), 404) from exc

        resolved_dataset, resolved_category = validate_dataset_category(
            dataset or profile.get("dataset") or self.settings.defaults.get("dataset", "mvtec"),
            category or profile.get("category") or self.settings.defaults.get("category", "bottle"),
        )
        if str(mode or "inference_only") != "inference_only":
            raise ApiError(
                "INFERENCE_ONLY_REQUIRED",
                "正式测试模式只允许推理，不能利用测试图片更新记忆库或阈值。",
                403,
            )
        resolved_device, warnings = self.registry.resolve_device(device)
        try:
            data_root = self.settings.data_root_for(profile)
        except ValueError:
            data_root = self.settings.project_root
        selected_profile = self.domain_profiles.resolve_for_inference(
            memory_bank_id,
            resolved_model_id,
            resolved_dataset,
            resolved_category,
        )
        memory_path = Path(selected_profile["memory_path"])
        calibration_path = Path(selected_profile["calibration_path"])
        profile_kind = str(selected_profile.get("profile_kind") or "builtin_public_baseline")
        if profile_kind == "target_domain" and threshold is not None:
            frozen_threshold = selected_profile.get("threshold")
            # Video and other clients may echo the read-only threshold shown by
            # the API.  Echoing the exact frozen value is not an override; a
            # different value remains forbidden.  The effective runtime value
            # is still loaded from the frozen profile below.
            if frozen_threshold is None or abs(float(threshold) - float(frozen_threshold)) > 1e-9:
                raise ApiError(
                    "FROZEN_THRESHOLD_OVERRIDE_FORBIDDEN",
                    "所选目标域记忆库已冻结，正式测试模式不能覆盖阈值。",
                    403,
                )
        defect_path = calibration_path.parent / "defect_direction.pt" if profile_kind != "target_domain" else None
        runtime_config = deepcopy(selected_profile.get("config") or profile)
        # The Memory Bank owns multiple immutable configuration versions.  The
        # applied runtime binding, rather than the Bank's original default
        # snapshot, is the authoritative configuration for this inference.
        # Routes resolve and compatibility-check this payload before it reaches
        # the adapter; copying it here keeps scoring and persisted provenance on
        # the same configuration version.
        if config_values:
            runtime_config.update(deepcopy(config_values))
        prototype_value = selected_profile.get("defect_prototype_path")
        prototype_path = Path(prototype_value) if prototype_value else calibration_path.parent / "defect_prototypes.pt"
        lightweight_value = selected_profile.get("lightweight_calibrator_path")
        lightweight_path = Path(lightweight_value) if lightweight_value else calibration_path.parent / "lightweight_calibrator.json"
        selected_mode = str(selected_profile.get("method_mode_resolved") or runtime_config.get("method_mode_resolved") or "")
        if lightweight_path.is_file():
            from methods.lightweight_calibration.lightweight_profile import load_lightweight_profile

            lightweight_profile = load_lightweight_profile(lightweight_path)
        else:
            lightweight_profile = None
        lightweight_ready = bool(
            profile_kind == "target_domain"
            and selected_mode == "normal_memory_defect_guided_lite"
            and lightweight_profile is not None
        )
        dual_memory_ready = bool(profile_kind == "target_domain" and prototype_path.is_file() and selected_mode == "normal_defect_dual_memory")
        # A frozen lightweight profile must never silently degrade to baseline
        # scoring when its calibration artifact is missing or unreadable.  The
        # domain-profile service checks the file and its hash; this adapter
        # guard also covers malformed/invalid runtime selections.
        if profile_kind == "target_domain" and selected_mode == "normal_memory_defect_guided_lite" and not lightweight_ready:
            raise ApiError(
                "MEMORY_BANK_NOT_FROZEN",
                "轻量校准方案缺少可用的冻结校准器文件，不能静默使用错误配置。",
                409,
            )
        runtime_config.update(
            {
                "project_root": str(self.settings.project_root),
                "data_root": str(data_root),
                "dataset": resolved_dataset,
                "category": resolved_category,
                "device": resolved_device,
                "use_fp16": bool(profile.get("use_fp16", False) and resolved_device == "cuda"),
                "use_calibrated_threshold": calibration_path.exists(),
                "calibration_path": str(calibration_path) if calibration_path.exists() else "",
                "defect_direction_path": str(defect_path) if defect_path is not None and defect_path.exists() else "",
                "defect_prototype_path": str(prototype_path) if dual_memory_ready else "",
                "lightweight_calibrator_path": str(lightweight_path) if lightweight_ready else "",
                "use_dual_memory": dual_memory_ready,
                "method_mode_resolved": str(selected_profile.get("method_mode_resolved") or runtime_config.get("method_mode_resolved") or "normal_only"),
                "save_heatmap": bool(self.settings.dynamic_settings().get("save_visuals", self.settings.defaults.get("save_visuals", True)) if save_visuals is None else save_visuals),
                "save_mask": bool(self.settings.dynamic_settings().get("save_visuals", self.settings.defaults.get("save_visuals", True)) if save_visuals is None else save_visuals),
                "save_overlay": bool(self.settings.dynamic_settings().get("save_visuals", self.settings.defaults.get("save_visuals", True)) if save_visuals is None else save_visuals),
                "no_save_visuals": not bool(self.settings.dynamic_settings().get("save_visuals", self.settings.defaults.get("save_visuals", True)) if save_visuals is None else save_visuals),
                "async_save_outputs": False,
                "domain_profile_id": selected_profile["memory_bank_id"],
                "domain_profile_kind": profile_kind,
                "inference_only": True,
                "test_data_update_allowed": False,
            }
        )
        if profile_kind == "target_domain":
            runtime_config["use_defect_direction"] = False
            warnings.append("已启用冻结目标域方案；当前任务为推理只读模式。")
            if dual_memory_ready:
                warnings.append("图像判定使用可靠性门控双记忆评分；定位热力图继续使用正常记忆基线。")
            if lightweight_ready:
                warnings.append("图像判定使用单正常记忆与缺陷引导轻量校准；定位热力图继续使用正常记忆基线。")
        if resolved_device == "cpu" and bool(profile.get("use_fp16", False)):
            warnings.append("当前设备为 CPU，已自动关闭 FP16。")
        if not calibration_path.exists():
            warnings.append("未找到 calibration.json，当前使用模型配置中的阈值。")
        resolved_threshold = (
            None
            if profile_kind == "target_domain"
            else (float(threshold) if threshold is not None else self.settings.dynamic_settings().get("threshold"))
        )
        frozen_context = DetectionContext.from_value(detection_context) if detection_context is not None else None
        if frozen_context is not None:
            mismatched = {
                "model_id": (frozen_context.model_id, resolved_model_id),
                "memory_bank_id": (frozen_context.memory_bank_id, str(selected_profile["memory_bank_id"])),
                "config_id": (frozen_context.config_id, str(config_profile_id or selected_profile.get("config_profile_id") or "")),
            }
            invalid = [key for key, pair in mismatched.items() if pair[0] and pair[0] != pair[1]]
            if invalid:
                raise ApiError(
                    "DETECTION_CONTEXT_MISMATCH",
                    "检测任务的冻结运行身份与实际资源不一致，已拒绝执行。",
                    409,
                    {"mismatched_fields": invalid},
                )
        return {
            "model_id": resolved_model_id,
            "model_definition": definition,
            "dataset": resolved_dataset,
            "category": resolved_category,
            "device": resolved_device,
            "threshold": resolved_threshold,
            "save_visuals": not bool(runtime_config["no_save_visuals"]),
            "config": runtime_config,
            "memory_path": memory_path,
            "calibration_path": calibration_path if calibration_path.exists() else None,
            "defect_direction_path": defect_path if defect_path is not None and defect_path.exists() else None,
            "defect_prototype_path": prototype_path if dual_memory_ready else None,
            "lightweight_calibrator_path": lightweight_path if lightweight_ready else None,
            "memory_bank_id": selected_profile["memory_bank_id"],
            "memory_bank_profile": selected_profile,
            "config_profile_id": str(config_profile_id or selected_profile.get("config_profile_id") or ""),
            "config_profile_version": int(config_profile_version or selected_profile.get("config_profile_version") or 0),
            "config_hash": str(config_hash or selected_profile.get("config_hash") or ""),
            "mode": "inference_only",
            "warnings": warnings,
            "detection_context": frozen_context,
        }

    @staticmethod
    def _manifest_context(resolved: dict[str, Any]) -> dict[str, Any] | None:
        return detection_context_dict(resolved.get("detection_context"))

    @classmethod
    def _manifest_selection(cls, resolved: dict[str, Any]) -> dict[str, Any]:
        selection = {
            key: value
            for key, value in resolved.items()
            if key not in {"config", "model_definition", "detection_context"}
        }
        context = cls._manifest_context(resolved)
        if context is not None:
            selection["runtime_revision"] = context["runtime_revision"]
            selection["detection_context"] = context
        return selection

    def start_job(self, kind: str) -> JobPaths:
        return self.jobs.create(kind)

    def load_model(
        self,
        model_id: str | None,
        dataset: str | None,
        category: str | None,
        device: str | None,
        threshold: float | None,
        memory_bank_id: str | None = None,
        mode: str = "inference_only",
    ) -> dict[str, Any]:
        resolved = self._resolve(model_id, dataset, category, device, threshold, None, memory_bank_id, mode)
        entry, cache_hit = self.registry.get_or_create(resolved)
        return {
            "model_id": resolved["model_id"],
            "dataset": resolved["dataset"],
            "category": resolved["category"],
            "device": resolved["device"],
            "threshold": resolved["threshold"] if resolved["threshold"] is not None else entry.inferencer.threshold_value,
            "cache_hit": cache_hit,
            "memory_bank_id": resolved["memory_bank_id"],
            "mode": "inference_only",
            "method_name": resolved["memory_bank_profile"].get("method_name") or "正常记忆异常检测",
            "method_mode": resolved["memory_bank_profile"].get("method_mode_resolved") or "normal_only",
            "warnings": resolved["warnings"],
        }

    def resolve_video_runtime(
        self,
        model_id: str | None = None,
        dataset: str | None = None,
        category: str | None = None,
        device: str | None = None,
        threshold: float | None = None,
        memory_bank_id: str | None = None,
        mode: str = "inference_only",
        config_profile_id: str | None = None,
        config_profile_version: int | None = None,
        config_hash: str | None = None,
        config_values: dict[str, Any] | None = None,
        detection_context: DetectionContext | dict[str, Any] | None = None,
        progress_callback: Any | None = None,
        **_: Any,
    ) -> tuple[dict[str, Any], Any, bool]:
        """Return one cached official runtime for the public video API.

        The video service deliberately uses this public adapter method rather
        than duplicating PatchCore setup in the software layer.
        """
        resolved = self._resolve(
            model_id,
            dataset,
            category,
            device,
            threshold,
            None,
            memory_bank_id,
            mode,
            config_profile_id,
            config_profile_version,
            config_hash,
            config_values,
            detection_context,
        )
        entry, cache_hit = self.registry.get_or_create(resolved, progress_callback=progress_callback)
        return resolved, entry, cache_hit

    def validate_image(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            raise ApiError("INPUT_FILE_MISSING", "所选图像文件不存在。", 404)
        if path.suffix.lower() not in self.settings.supported_suffixes():
            raise ApiError("UNSUPPORTED_FILE_TYPE", f"不支持的图像类型：{path.suffix}", 415, {"supported": sorted(self.settings.supported_suffixes())})
        try:
            with Image.open(path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise ApiError("INVALID_IMAGE", "上传文件不是有效图像。", 422) from exc

    def save_upload(self, job: JobPaths, filename: str, content: bytes) -> Path:
        if len(content) > self.settings.max_upload_bytes():
            raise ApiError("FILE_TOO_LARGE", "上传图像超过配置的大小限制。", 413, {"max_bytes": self.settings.max_upload_bytes()})
        safe_name = Path(filename or "upload.png").name
        target = job.input_dir / safe_name
        counter = 1
        while target.exists():
            target = job.input_dir / f"{Path(safe_name).stem}_{counter}{Path(safe_name).suffix}"
            counter += 1
        target.write_bytes(content)
        self.validate_image(target)
        return target

    def upload_target(self, job: JobPaths, filename: str) -> Path:
        """Reserve a unique target for a streamed image upload."""
        safe_name = Path(filename or "upload.png").name
        target = job.input_dir / safe_name
        counter = 1
        while target.exists() or target.with_name(f".{target.name}.uploading").exists():
            target = job.input_dir / f"{Path(safe_name).stem}_{counter}{Path(safe_name).suffix}"
            counter += 1
        return target

    def list_directory_images(self, input_dir: str | Path) -> list[Path]:
        folder = Path(input_dir).expanduser().resolve()
        if not folder.exists() or not folder.is_dir():
            raise ApiError("INPUT_DIRECTORY_MISSING", "输入目录不存在。", 404)
        workspace_root = self.settings.project_root.parent.resolve()
        forbidden_roots = [
            self.settings.algorithm_outputs_root.resolve(),
            self.settings.memory_banks_root.resolve(),
            (workspace_root / "AdaptiveAOI_Backup").resolve(),
            (workspace_root / "Docyment_report").resolve(),
        ]
        if any(folder == root or root in folder.parents for root in forbidden_roots):
            raise ApiError("INPUT_DIRECTORY_NOT_ALLOWED", "不能把检测输出、记忆库、备份或临时报告目录作为批量测试输入。", 403)
        images = [path for path in sorted(folder.rglob("*")) if path.is_file() and path.suffix.lower() in self.settings.supported_suffixes()]
        if not images:
            raise ApiError("NO_SUPPORTED_IMAGES", "输入目录中未找到支持的图像文件。", 422)
        if len(images) > self.settings.max_batch_files():
            raise ApiError("BATCH_TOO_LARGE", "输入目录中的图像数量超过配置的批量上限。", 413, {"max_batch_files": self.settings.max_batch_files()})
        return images

    def _original_artifact(self, source: Path, job: JobPaths, index: int | None = None) -> Path | None:
        save_original = bool(self.settings.dynamic_settings().get("save_original", self.settings.defaults.get("save_original", True)))
        if not save_original:
            return None
        if source.is_relative_to(job.root):
            return source
        suffix = f"{index:04d}_" if index is not None else ""
        target = job.results_dir / "originals" / f"{suffix}{source.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return target

    def _run_one(self, job: JobPaths, source: Path, resolved: dict[str, Any], request_id: str, batch_id: str | None = None, index: int | None = None) -> dict[str, Any]:
        end_to_end_started = time.perf_counter()
        self.validate_image(source)
        entry, cache_hit = self.registry.get_or_create(resolved)
        result_json_path = job.results_dir / "records" / f"{index if index is not None else 0:04d}_{source.stem}_result.json"
        result_json_path.parent.mkdir(parents=True, exist_ok=True)
        queue_started = time.perf_counter()
        with self.registry.execution(entry, job.root, resolved["threshold"]) as inferencer:
            queue_wait_ms = float((time.perf_counter() - queue_started) * 1000.0)
            visual_keys = ("save_heatmap", "save_mask", "save_overlay", "no_save_visuals", "timing_mode")
            original_visual_config = {key: inferencer.config.get(key) for key in visual_keys}
            inferencer.config.update(
                {
                    "save_heatmap": bool(resolved["save_visuals"]),
                    "save_mask": bool(resolved["save_visuals"]),
                    "save_overlay": bool(resolved["save_visuals"]),
                    "no_save_visuals": not bool(resolved["save_visuals"]),
                    "timing_mode": "strict_total" if resolved["save_visuals"] else "no_save_total",
                }
            )
            try:
                core_result = inferencer.infer_image(source, save_outputs=True, generate_visuals=resolved["save_visuals"])
                inferencer.finalize_save_results([core_result], close=False)
            finally:
                for key, value in original_visual_config.items():
                    if value is None:
                        inferencer.config.pop(key, None)
                    else:
                        inferencer.config[key] = value
        clean = {key: json_safe(value) for key, value in core_result.items() if not str(key).startswith("_")}
        original_save_started = time.perf_counter()
        original_path = self._original_artifact(source, job, index)
        original_save_ms = float((time.perf_counter() - original_save_started) * 1000.0)
        end_to_end_ms = float((time.perf_counter() - end_to_end_started) * 1000.0)
        timing = structured_timing(
            core_result,
            model_load_ms=None if cache_hit else float(entry.inferencer.model_load_ms),
            memory_bank_load_ms=None if cache_hit else float(entry.inferencer.memory_bank_load_ms),
            queue_wait_ms=queue_wait_ms,
            extra_save_ms=original_save_ms,
            end_to_end_ms=end_to_end_ms,
            generate_visualizations=bool(core_result.get("generate_visualizations")),
        )
        clean["timing"] = timing
        result_save_started = time.perf_counter()
        result_json_path.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
        result_json_save_ms = float((time.perf_counter() - result_save_started) * 1000.0)
        timing["save_ms"] = float(timing["save_ms"] + result_json_save_ms)
        timing["end_to_end_ms"] = float((time.perf_counter() - end_to_end_started) * 1000.0)
        clean["timing"] = timing
        result_json_path.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
        normalized = normalized_detection(
            clean,
            resolved["model_id"],
            resolved["device"],
            self.settings.jobs_root,
            original_path,
            result_json_path,
            resolved.get("detection_context"),
        )
        frozen_context = self._manifest_context(resolved)
        if frozen_context is not None:
            normalized["runtime_revision"] = int(frozen_context["runtime_revision"])
            normalized["detection_context"] = frozen_context
            normalized["metadata"]["runtime_revision"] = int(frozen_context["runtime_revision"])
            normalized["metadata"]["detection_context"] = frozen_context
            normalized["metadata"]["started_at"] = frozen_context["started_at"]
            normalized["metadata"]["precision"] = frozen_context["precision"]
            normalized["metadata"]["algorithm_mode"] = frozen_context["algorithm_mode"]
        normalized["metadata"]["cache_hit"] = cache_hit
        normalized["metadata"]["task_id"] = job.job_id
        input_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        normalized["input_hash"] = input_hash
        normalized["task_id"] = job.job_id
        normalized["profile_id"] = frozen_context["profile_id"] if frozen_context is not None else resolved["memory_bank_id"]
        normalized["metadata"]["input_hash"] = input_hash
        normalized["metadata"]["profile_id"] = normalized["profile_id"]
        normalized["metadata"]["config_version"] = self.settings.profile_path(resolved["model_id"]).name
        normalized["metadata"]["calibration_path"] = str(resolved["calibration_path"] or "")
        normalized["metadata"]["memory_bank_id"] = resolved["memory_bank_id"]
        normalized["metadata"]["config_profile_id"] = resolved.get("config_profile_id")
        normalized["metadata"]["config_id"] = frozen_context["config_id"] if frozen_context is not None else resolved.get("config_profile_id")
        normalized["metadata"]["config_profile_version"] = resolved.get("config_profile_version")
        normalized["metadata"]["config_hash"] = resolved.get("config_hash")
        normalized["metadata"]["memory_bank_threshold"] = float(core_result.get("threshold", entry.inferencer.threshold_value))
        normalized["metadata"]["inference_mode"] = "inference_only"
        runtime_policy = dict(self.settings.dynamic_settings() or {})
        official_test_mode = bool(runtime_policy.get("official_test_mode"))
        normalized["metadata"]["official_test_mode"] = official_test_mode
        normalized["metadata"]["test_data_update_allowed"] = False
        normalized["metadata"]["data_role"] = "official_test_set" if official_test_mode else "inference_input"
        normalized["metadata"]["evaluation_mode"] = "official_test" if official_test_mode else "inference_only"
        profile_kind = str(resolved["memory_bank_profile"].get("profile_kind") or "")
        normalized["metadata"]["configuration_frozen"] = bool(
            resolved["memory_bank_profile"].get("frozen")
            and profile_kind.startswith("target_")
        )
        normalized["metadata"]["method_name"] = resolved["memory_bank_profile"].get("method_name") or "正常记忆异常检测"
        profile_method_mode = str(resolved["memory_bank_profile"].get("method_mode_resolved") or "normal_only")
        scoring_method_mode = str(core_result.get("method_mode") or "normal_memory")
        # Keep the profile/build mode and the runtime scoring kernel separate.
        # Feedback compatibility is bound to the immutable profile resources;
        # a calibrated profile may still execute the normal-memory scorer.
        normalized["metadata"]["method_mode"] = profile_method_mode
        normalized["metadata"]["method_mode_resolved"] = profile_method_mode
        normalized["metadata"]["profile_method_mode"] = profile_method_mode
        normalized["metadata"]["scoring_method_mode"] = scoring_method_mode
        normalized["metadata"]["image_size"] = resolved["memory_bank_profile"].get("image_size")
        normalized["metadata"]["feature_dimension"] = resolved["memory_bank_profile"].get("feature_dimension")
        normalized["metadata"]["feature_layers"] = resolved["memory_bank_profile"].get("feature_layers")
        normalized["metadata"]["backbone"] = resolved["memory_bank_profile"].get("backbone")
        normalized["metadata"]["base_method"] = resolved["memory_bank_profile"].get("base_method") or "normal patch memory nearest-neighbour anomaly detection"
        normalized["metadata"]["defect_prototype_count"] = int(resolved["memory_bank_profile"].get("defect_prototype_count") or 0)
        normalized["metadata"]["sample_source_summary"] = {
            "normal_source_types": resolved["memory_bank_profile"].get("normal_source_types") or [],
            "defect_source_types": resolved["memory_bank_profile"].get("defect_source_types") or [],
            "normal_image_count": resolved["memory_bank_profile"].get("normal_image_count"),
            "normal_video_count": resolved["memory_bank_profile"].get("normal_video_count"),
            "normal_extracted_frame_count": resolved["memory_bank_profile"].get("normal_extracted_frame_count"),
            "defect_image_count": resolved["memory_bank_profile"].get("defect_image_count"),
            "defect_video_count": resolved["memory_bank_profile"].get("defect_video_count"),
            "defect_extracted_frame_count": resolved["memory_bank_profile"].get("defect_extracted_frame_count"),
            "method_mode": resolved["memory_bank_profile"].get("method_mode_resolved"),
            "frozen": bool(resolved["memory_bank_profile"].get("frozen")),
        }
        normalized["metadata"]["generate_visualizations"] = bool(clean.get("generate_visualizations"))
        normalized["metadata"]["visualization_status"] = clean.get("visualization_status")
        normalized["metadata"]["timing"] = timing
        normalized["metadata"]["timing_mode"] = clean.get("timing_mode")
        normalized["metadata"]["warnings"] = list(resolved["warnings"])
        stored = self.storage.save_result(normalized, request_id=request_id, batch_id=batch_id)
        self.logger.info("inference_completed request_id=%s result_id=%s file=%s cache_hit=%s time_ms=%.3f", request_id, stored["result_id"], source.name, cache_hit, stored["inference_time_ms"])
        return stored

    @staticmethod
    def _visual_preview(item: dict[str, Any]) -> dict[str, Any] | None:
        """Return a compact, manifest-safe preview for live batch rendering."""
        if not bool(item.get("generate_visualizations", False)):
            return None
        metadata = item.get("metadata") or {}
        if not any(item.get(key) for key in ("original_image_url", "heatmap_url", "marked_image_url", "mask_url", "result_image_url")):
            return None
        return {
            "result_id": item.get("result_id"),
            "image_name": metadata.get("filename") or Path(str(metadata.get("source_path") or "")).name,
            "prediction": item.get("prediction"),
            "is_anomaly": bool(item.get("is_anomaly")),
            "anomaly_score": item.get("anomaly_score"),
            "threshold": item.get("threshold"),
            "original_image_url": item.get("original_image_url"),
            "heatmap_url": item.get("heatmap_url"),
            "marked_image_url": item.get("marked_image_url"),
            "mask_url": item.get("mask_url"),
            "result_image_url": item.get("result_image_url"),
        }

    def detect(self, job: JobPaths, source: Path, request_id: str, **selection: Any) -> dict[str, Any]:
        resolved = self._resolve(**selection)
        context = self._manifest_context(resolved)
        self.jobs.update(
            job,
            status="running",
            request_id=request_id,
            runtime_revision=context.get("runtime_revision") if context else None,
            detection_context=context,
            selection=self._manifest_selection(resolved),
            input_files=[str(source)],
        )
        try:
            result = self._run_one(job, source, resolved, request_id)
            self.jobs.update(
                job,
                status="completed",
                result_id=result["result_id"],
                artifacts=result["metadata"].get("result_json_url"),
                generate_visualizations=bool(result.get("generate_visualizations", True)),
                visualization_status=result.get("visualization_status", "generated"),
                timing=result.get("timing") or {},
                timing_mode=result.get("timing_mode"),
                warnings=resolved["warnings"],
            )
            return {"result": result, "warnings": resolved["warnings"], "job_id": job.job_id}
        except ApiError as exc:
            self.jobs.fail(job, exc.code, exc.message)
            raise
        except Exception as exc:
            self.logger.exception("inference_failed request_id=%s file=%s", request_id, source.name)
            self.jobs.fail(job, "INFERENCE_FAILED", str(exc))
            raise ApiError("INFERENCE_FAILED", "推理失败，请检查算法服务日志和模型资源。", 500) from exc

    def detect_batch(self, job: JobPaths, sources: Iterable[Path], request_id: str, **selection: Any) -> dict[str, Any]:
        wall_clock_started = time.perf_counter()
        resolved = self._resolve(**selection)
        context = self._manifest_context(resolved)
        items = list(sources)
        batch_id = job.job_id
        batch_started = time.perf_counter()
        self.jobs.update(
            job,
            status="running",
            stage="detecting",
            progress=0.0,
            processed=0,
            request_id=request_id,
            runtime_revision=context.get("runtime_revision") if context else None,
            detection_context=context,
            selection=self._manifest_selection(resolved),
            input_files=[str(item) for item in items],
            input_count=len(items),
            total_items=len(items),
            completed_items=0,
            active_processing_seconds=0.0,
            paused_seconds=0.0,
            current_fps=None,
            inference_fps=None,
            complete_fps=None,
            eta_seconds=None,
            paused_at=None,
            last_progress_at=None,
            generate_visualizations=bool(resolved["save_visuals"]),
            latest_visualized_result=None,
            visualized_results=[],
            started_at=utc_now(),
        )
        successful: list[dict[str, Any]] = []
        failed: list[dict[str, str]] = []
        visualized_results: list[dict[str, Any]] = []
        cancelled = False
        for index, source in enumerate(items):
            if bool(self.jobs.read_manifest(job).get("cancel_requested")):
                cancelled = True
                break
            self.jobs.update(job, current_file=source.name, current_index=index + 1)
            item_started = time.perf_counter()
            latest_visualized_result = None
            try:
                item = self._run_one(job, source, resolved, request_id, batch_id=batch_id, index=index)
                successful.append(item)
                latest_visualized_result = self._visual_preview(item)
                if latest_visualized_result:
                    visualized_results = [
                        previous
                        for previous in visualized_results
                        if str(previous.get("result_id") or "") != str(latest_visualized_result.get("result_id") or "")
                    ]
                    visualized_results.append(latest_visualized_result)
                    visualized_results = visualized_results[-20:]
            except ApiError as exc:
                failed.append({"filename": source.name, "code": exc.code, "message": exc.message})
                self.logger.warning("batch_item_failed request_id=%s file=%s code=%s", request_id, source.name, exc.code)
            except Exception as exc:
                failed.append({"filename": source.name, "code": "INFERENCE_FAILED", "message": "Inference failed for this image."})
                self.logger.exception("batch_item_failed request_id=%s file=%s", request_id, source.name)
            processed = len(successful) + len(failed)
            anomaly_count = sum(1 for item in successful if item.get("is_anomaly"))
            elapsed = max(0.000001, time.perf_counter() - batch_started)
            average_ms = elapsed * 1000.0 / max(1, processed)
            remaining = max(0, len(items) - processed)
            cancel_requested = bool(self.jobs.read_manifest(job).get("cancel_requested"))
            live_status = "cancelling" if cancel_requested else "running"
            inference_times = sorted(float(item["inference_time_ms"]) for item in successful)
            p50_live = inference_times[max(0, math.ceil(len(inference_times) * 0.50) - 1)] if inference_times else None
            p95_live = inference_times[max(0, math.ceil(len(inference_times) * 0.95) - 1)] if inference_times else None
            inference_total_ms = sum(inference_times)
            inference_fps = len(inference_times) / (inference_total_ms / 1000.0) if inference_total_ms > 0 else None
            throughput = processed / elapsed
            self.jobs.update(
                job,
                status=live_status,
                stage="cancelling" if cancel_requested else "detecting",
                processed=processed,
                completed_items=processed,
                total_items=len(items),
                success_count=len(successful),
                normal_count=len(successful) - anomaly_count,
                anomaly_count=anomaly_count,
                failure_count=len(failed),
                progress=processed / max(1, len(items)),
                current_file=source.name,
                current_item_ms=(time.perf_counter() - item_started) * 1000.0,
                average_ms=average_ms,
                p50_ms=p50_live,
                p95_ms=p95_live,
                throughput=throughput,
                current_fps=throughput,
                inference_fps=inference_fps,
                complete_fps=throughput,
                eta_seconds=None if cancel_requested else remaining * average_ms / 1000.0,
                elapsed_seconds=elapsed,
                active_processing_seconds=elapsed,
                paused_seconds=0.0,
                last_progress_at=utc_now(),
                generate_visualizations=bool(resolved["save_visuals"]),
                latest_visualized_result=latest_visualized_result or (visualized_results[-1] if visualized_results else None),
                visualized_results=visualized_results,
            )
            if cancel_requested:
                cancelled = True
                break
        active_processing_seconds = max(0.000001, time.perf_counter() - batch_started)
        csv_path = job.results_dir / "batch_result.csv"
        fields = [
            "result_id",
            "filename",
            "prediction",
            "is_anomaly",
            "anomaly_score",
            "threshold",
            "inference_time_ms",
            "model_id",
            "memory_bank_id",
            "runtime_revision",
            "inference_mode",
            "device",
            "generate_visualizations",
            "input_read_ms",
            "preprocess_ms",
            "feature_extraction_ms",
            "ann_search_ms",
            "inference_ms",
            "postprocess_ms",
            "visualization_ms",
            "encode_ms",
            "save_ms",
            "end_to_end_ms",
        ]
        csv_save_started = time.perf_counter()
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for item in successful:
                writer.writerow(
                    {
                        "result_id": item["result_id"],
                        "filename": item["metadata"].get("filename"),
                        "prediction": item["prediction"],
                        "is_anomaly": item["is_anomaly"],
                        "anomaly_score": item["anomaly_score"],
                        "threshold": item["threshold"],
                        "inference_time_ms": item["inference_time_ms"],
                        "model_id": item["model_id"],
                        "memory_bank_id": item.get("metadata", {}).get("memory_bank_id"),
                        "runtime_revision": item.get("runtime_revision"),
                        "inference_mode": item.get("metadata", {}).get("inference_mode"),
                        "device": item["device"],
                        "generate_visualizations": item.get("generate_visualizations"),
                        **{
                            key: (item.get("timing") or {}).get(key)
                            for key in fields
                            if key.endswith("_ms") and key != "inference_time_ms"
                        },
                    }
                )
        csv_save_ms = float((time.perf_counter() - csv_save_started) * 1000.0)
        anomaly = sum(1 for item in successful if item["is_anomaly"])
        mean_time = sum(item["inference_time_ms"] for item in successful) / len(successful) if successful else None
        ordered_times = sorted(item["inference_time_ms"] for item in successful)
        p95_time = ordered_times[max(0, math.ceil(len(ordered_times) * 0.95) - 1)] if ordered_times else None
        batch_end_to_end_ms = float((time.perf_counter() - wall_clock_started) * 1000.0)
        batch_timing = self._batch_timing(successful, batch_end_to_end_ms, csv_save_ms)
        final_status = "cancelled" if cancelled else ("partial" if failed else "completed")
        processed = len(successful) + len(failed)
        final_throughput = processed / active_processing_seconds if processed else None
        final_inference_fps = (
            len(ordered_times) / (sum(ordered_times) / 1000.0)
            if ordered_times and sum(ordered_times) > 0
            else None
        )
        self.jobs.update(
            job,
            status=final_status,
            stage="cancelled" if cancelled else "completed",
            progress=(len(successful) + len(failed)) / max(1, len(items)),
            processed=processed,
            completed_items=processed,
            total_items=len(items),
            current_file=None,
            output_csv=str(csv_path),
            success_count=len(successful),
            normal_count=len(successful) - anomaly,
            anomaly_count=anomaly,
            failure_count=len(failed),
            elapsed_seconds=batch_end_to_end_ms / 1000.0,
            active_processing_seconds=active_processing_seconds,
            paused_seconds=0.0,
            current_fps=final_throughput,
            inference_fps=final_inference_fps,
            complete_fps=final_throughput,
            throughput=final_throughput,
            average_ms=(active_processing_seconds * 1000.0 / processed) if processed else None,
            p50_ms=ordered_times[max(0, math.ceil(len(ordered_times) * 0.50) - 1)] if ordered_times else None,
            p95_ms=p95_time,
            eta_seconds=0.0 if final_status in {"completed", "partial"} else None,
            paused_at=None,
            last_progress_at=utc_now() if processed else None,
            generate_visualizations=bool(resolved["save_visuals"]),
            timing=batch_timing,
            warnings=resolved["warnings"],
            latest_visualized_result=visualized_results[-1] if visualized_results else None,
            visualized_results=visualized_results,
        )
        return {
            "job_id": job.job_id,
            "batch_id": batch_id,
            "runtime_revision": context.get("runtime_revision") if context else None,
            "detection_context": context,
            "total": len(items),
            "successful": len(successful),
            "failed": len(failed),
            "normal": len(successful) - anomaly,
            "anomaly": anomaly,
            "average_inference_time_ms": mean_time,
            "p95_inference_time_ms": p95_time,
            "generate_visualizations": bool(resolved["save_visuals"]),
            "visualization_status": "generated" if resolved["save_visuals"] else "not_generated",
            "visualization_artifacts": [
                {
                    "result_id": item.get("result_id"),
                    "artifacts": item.get("visualization_artifacts") or [],
                }
                for item in successful
                if item.get("visualization_artifacts")
            ],
            "timing": batch_timing,
            "timing_mode": "strict_total" if resolved["save_visuals"] else "no_save_total",
            "results": successful,
            "errors": failed,
            "result_csv_url": artifact_url(csv_path, self.settings.jobs_root),
            "warnings": resolved["warnings"],
            "status": final_status,
        }

    @staticmethod
    def _batch_timing(successful: list[dict[str, Any]], task_end_to_end_ms: float, csv_save_ms: float) -> dict[str, Any]:
        stage_keys = (
            "input_read_ms",
            "preprocess_ms",
            "feature_extraction_ms",
            "ann_search_ms",
            "anomaly_scoring_ms",
            "inference_ms",
            "postprocess_ms",
            "visualization_ms",
            "encode_ms",
            "save_ms",
            "end_to_end_ms",
        )

        def stats(values: list[float]) -> dict[str, float | int] | None:
            if not values:
                return None
            ordered = sorted(values)

            def percentile(percent: float) -> float:
                position = (len(ordered) - 1) * percent / 100.0
                lower = int(position)
                upper = min(lower + 1, len(ordered) - 1)
                weight = position - lower
                return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)

            return {
                "count": len(ordered),
                "total_ms": float(sum(ordered)),
                "mean_ms": float(sum(ordered) / len(ordered)),
                "p50_ms": percentile(50.0),
                "p95_ms": percentile(95.0),
                "min_ms": float(ordered[0]),
                "max_ms": float(ordered[-1]),
            }

        stage_statistics: dict[str, Any] = {}
        for key in stage_keys:
            values = [float((item.get("timing") or {}).get(key)) for item in successful if (item.get("timing") or {}).get(key) is not None]
            stage_statistics[key] = stats(values)
        model_load = next(((item.get("timing") or {}).get("model_load_ms") for item in successful if (item.get("timing") or {}).get("model_load_ms") is not None), None)
        memory_load = next(((item.get("timing") or {}).get("memory_bank_load_ms") for item in successful if (item.get("timing") or {}).get("memory_bank_load_ms") is not None), None)
        return {
            "successful_samples": len(successful),
            "model_load_ms": model_load,
            "memory_bank_load_ms": memory_load,
            "model_reused": model_load is None,
            "memory_bank_reused": memory_load is None,
            "task_end_to_end_ms": float(task_end_to_end_ms),
            "result_csv_save_ms": float(csv_save_ms),
            "throughput_images_per_second": float(len(successful) / (task_end_to_end_ms / 1000.0)) if successful and task_end_to_end_ms > 0 else None,
            "stages": stage_statistics,
        }

    def calibrate(self, job: JobPaths, request_id: str, dataset: str, category: str, model_id: str | None, normal_shot: int, defect_shot: int, seed: int, use_defect_direction: bool) -> dict[str, Any]:
        resolved = self._resolve(model_id, dataset, category, None, None, True)
        memory_target = job.root / "checkpoints" / resolved["dataset"] / resolved["category"] / "memory_bank.pt"
        memory_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolved["memory_path"], memory_target)
        config = dict(resolved["config"])
        config["output_root"] = str(job.root)
        config_path = job.root / "calibration_runtime_config.yaml"
        import yaml

        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        # The standalone calibration script consumes a dataset manifest from
        # the job output directory.  API-triggered calibration jobs are
        # isolated under ``system_runtime/jobs`` and therefore cannot rely on
        # a manifest generated by an unrelated offline experiment.  Generate a
        # fresh, read-only manifest from the configured data root before
        # launching the script; this does not copy or modify source images.
        manifest_output = job.root / "results"
        manifest_output.mkdir(parents=True, exist_ok=True)
        data_root = Path(str(config.get("data_root") or "")).expanduser()
        if not data_root.is_dir():
            self.jobs.fail(job, "CALIBRATION_DATA_ROOT_MISSING", "校准数据根目录不存在或不可访问。")
            raise ApiError("CALIBRATION_DATA_ROOT_MISSING", "校准数据根目录不存在或不可访问。", 422)
        try:
            # Build only the requested dataset.  ``write_manifest_outputs``
            # scans every configured dataset (which can be tens of thousands
            # of files) and made an API calibration appear hung on a normal
            # workstation.  A calibration job needs exactly one manifest.
            from datasets.manifest import (
                build_dagm_manifest,
                build_mvtec_manifest,
                build_visa_from_pytorch,
                build_visa_manifest,
                find_data_child,
            )

            dataset_name = str(resolved["dataset"]).lower()
            if dataset_name == "mvtec":
                dataset_root = find_data_child(data_root, ("mvtec", "mvtec_ad"))
                if dataset_root is None:
                    raise FileNotFoundError("MVTec data directory not found")
                manifest_df = build_mvtec_manifest(dataset_root)
            elif dataset_name == "visa":
                dataset_root = find_data_child(data_root, ("visa", "VisA"))
                if dataset_root is not None:
                    manifest_df = build_visa_manifest(dataset_root, data_root=data_root)
                elif (data_root / "visa_pytorch").exists():
                    manifest_df = build_visa_from_pytorch(data_root / "visa_pytorch")
                else:
                    raise FileNotFoundError("VisA data directory not found")
            elif dataset_name == "dagm":
                dataset_root = find_data_child(data_root, ("dagm",))
                if dataset_root is None:
                    raise FileNotFoundError("DAGM data directory not found")
                manifest_df, _ = build_dagm_manifest(dataset_root)
            else:
                raise ValueError(f"unsupported calibration dataset: {dataset_name}")
            expected_manifest = manifest_output / f"manifest_{dataset_name}.csv"
            manifest_df.to_csv(expected_manifest, index=False, encoding="utf-8-sig")
            manifest_paths = {f"manifest_{dataset_name}": expected_manifest}
            expected_manifest = manifest_output / f"manifest_{resolved['dataset']}.csv"
            if not expected_manifest.is_file():
                self.jobs.fail(job, "CALIBRATION_MANIFEST_MISSING", "校准数据清单未生成。")
                raise ApiError("CALIBRATION_MANIFEST_MISSING", "未找到当前数据集的校准清单。", 422)
            self.jobs.update(job, manifest_paths={key: str(value) for key, value in manifest_paths.items()})
        except ApiError:
            raise
        except Exception as exc:
            self.logger.exception("calibration_manifest_build_failed")
            self.jobs.fail(job, "CALIBRATION_MANIFEST_FAILED", "校准数据清单生成失败。")
            raise ApiError("CALIBRATION_MANIFEST_FAILED", "校准数据清单生成失败，请检查数据根目录和日志。", 422) from exc
        command = [
            sys.executable,
            str(self.settings.project_root / "scripts" / "calibrate.py"),
            "--config",
            str(config_path),
            "--dataset",
            resolved["dataset"],
            "--category",
            resolved["category"],
            "--normal_shot",
            str(int(normal_shot)),
            "--defect_shot",
            str(int(defect_shot)),
            "--seed",
            str(int(seed)),
        ]
        if use_defect_direction:
            command.append("--use_defect_direction")
        self.jobs.update(job, request_id=request_id, command=command, status="running")
        completed = subprocess.run(command, cwd=str(self.settings.project_root), text=True, encoding="utf-8", errors="replace", capture_output=True, check=False)
        stdout_path = job.results_dir / "calibration_stdout.log"
        stderr_path = job.results_dir / "calibration_stderr.log"
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        calibration_path = job.root / "calibrations" / resolved["dataset"] / resolved["category"] / "calibration.json"
        if completed.returncode != 0 or not calibration_path.exists():
            self.jobs.fail(job, "CALIBRATION_FAILED", "Calibration script failed or did not produce calibration.json.")
            raise ApiError("CALIBRATION_FAILED", "Calibration failed. Check job logs for details.", 500, {"job_id": job.job_id})
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        self.jobs.update(job, status="completed", calibration_path=str(calibration_path), calibration=payload)
        return {"job_id": job.job_id, "dataset": resolved["dataset"], "category": resolved["category"], "calibration": payload, "calibration_path": str(calibration_path), "active": False}
