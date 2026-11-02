from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from typing import Any

from backend.config import IntegrationSettings
from backend.errors import ApiError


class ModelRegistry:
    """Maps stable API model IDs to existing algorithm configuration profiles."""

    def __init__(self, settings: IntegrationSettings, runtime_registry: Any):
        self.settings = settings
        self.runtime_registry = runtime_registry

    def model_info(self, model_id: str, dataset: str | None = None, category: str | None = None) -> dict[str, Any]:
        try:
            resolved_id, definition = self.settings.model_definition(model_id)
            profile_path = self.settings.profile_path(resolved_id)
            # The model catalog describes packaged feature extractors.  It
            # must remain available before the first target-domain Memory Bank
            # exists, so do not require a bound dataset root here.
            profile = self.settings.algorithm_profile_unbound(resolved_id)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise ApiError("MODEL_NOT_FOUND", str(exc), 404) from exc
        effective_dataset = str(dataset or profile.get("dataset") or self.settings.defaults.get("dataset", "mvtec")).lower()
        effective_category = str(category or profile.get("category") or self.settings.defaults.get("category", "bottle"))
        memory_path = self.settings.algorithm_outputs_root / "checkpoints" / effective_dataset / effective_category / "memory_bank.pt"
        calibration_path = self.settings.active_calibration_path(effective_dataset, effective_category)
        threshold = profile.get("threshold")
        if calibration_path.exists():
            try:
                calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
                threshold = calibration.get("image_threshold", threshold)
            except Exception:
                pass
        return {
            "model_id": resolved_id,
            "display_name": definition.get("display_name", resolved_id),
            "model_type": definition.get("model_type", "patchcore"),
            "profile_path": str(profile_path),
            "profile_valid": profile_path.exists(),
            "backbone": profile.get("backbone"),
            "memory_bank_path": str(memory_path),
            "weights_valid": memory_path.exists(),
            "calibration_path": str(calibration_path),
            "calibration_valid": calibration_path.exists(),
            "default_threshold": threshold,
            "dataset": effective_dataset,
            "category": effective_category,
            "is_default": resolved_id == str(self.settings.defaults.get("model_id") or ""),
            "loaded": self.runtime_registry.has_model(resolved_id, effective_dataset, effective_category),
        }

    def list_models(self) -> list[dict[str, Any]]:
        return [self.model_info(model_id) for model_id in self.settings.models]

    def official_base_model(
        self,
        *,
        device: str | None = None,
        verify_sha256: bool = True,
        require_ready: bool = False,
    ) -> dict[str, Any]:
        """Resolve the packaged feature model used for new target domains.

        The source is deliberately the immutable integration default, not the
        currently applied Runtime Profile.  This keeps creation of the first
        Memory Bank possible and prevents a stale target-domain Config from
        selecting a different feature space.
        """
        official_id = str(self.settings.defaults.get("model_id") or "").strip()
        if not official_id:
            raise ApiError(
                "OFFICIAL_BASE_MODEL_NOT_CONFIGURED",
                "正式基础特征资源尚未配置。",
                503,
            )
        try:
            model_id, definition = self.settings.model_definition(official_id)
            profile_path = self.settings.profile_path(model_id)
            profile = self.settings.algorithm_profile_unbound(model_id)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise ApiError(
                "OFFICIAL_BASE_MODEL_INVALID",
                "正式模型资源缺失或配置无效。",
                503,
                details={"reason": str(exc)},
            ) from exc

        backbone = str(profile.get("backbone") or "").strip().lower()
        from core.feature_extractor import backbone_resource_status

        resource = dict(backbone_resource_status(backbone, verify_sha256=verify_sha256))
        requested_device = str(device or "auto").strip().lower()
        resolved_device, device_warnings = self.runtime_registry.resolve_device(requested_device)
        backbone_root = Path(os.environ.get("ADAPTIVEAOI_BACKBONE_ROOT") or (self.settings.project_root / "artifacts" / "models" / "backbones")).expanduser().resolve()
        manifest_path = backbone_root / "manifest.json"
        manifest_sha256 = ""
        if manifest_path.is_file():
            digest = hashlib.sha256()
            with manifest_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            manifest_sha256 = digest.hexdigest()

        ready = bool(resource.get("ready"))
        if require_ready and not ready:
            raise ApiError(
                "OFFICIAL_BASE_MODEL_UNAVAILABLE",
                "正式模型资源缺失或完整性校验失败。",
                503,
                details={
                    "reason_code": resource.get("reason_code"),
                    "reason": resource.get("reason"),
                    "resolution": resource.get("resolution"),
                },
            )
        return {
            "model_id": model_id,
            "model_version": str(definition.get("version") or self.settings.backend.get("version") or "1.0.0"),
            "display_name": str(definition.get("display_name") or model_id),
            "model_type": str(definition.get("model_type") or "feature_memory_anomaly_detection"),
            "profile_path": str(profile_path),
            "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
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
                "sha256": manifest_sha256,
                "integrity_status": str(resource.get("reason_code") or "unknown"),
            },
            "resource_status": resource,
            "ready": ready,
            "device_compatibility": {
                "requested": requested_device,
                "resolved": resolved_device,
                "warnings": list(device_warnings or []),
            },
            "source": "system_official_model_registry",
        }
