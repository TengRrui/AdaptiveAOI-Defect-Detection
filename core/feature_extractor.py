from __future__ import annotations

from contextlib import nullcontext
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import (
    resnet50,
    wide_resnet50_2,
)
from torchvision.models.feature_extraction import create_feature_extractor

from core.domain_build_contract import DEFAULT_FEATURE_LAYERS, dtype_name


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BACKBONE_SPECS = {
    "wide_resnet50_2": {
        "filename": "wide_resnet50_2-9ba9bcbe.pth",
        "size": 275_905_729,
        "sha256": "9ba9bcbebc349d733a72eb7608143fd754e4689ac8d8ce2916ca0bdff6443950",
        "factory": wide_resnet50_2,
    },
    "resnet50": {
        "filename": "resnet50-11ad3fa6.pth",
        "size": 102_540_417,
        "sha256": "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca",
        "factory": resnet50,
    },
}


def _append_warning(output_root: str | Path, message: str):
    log_dir = Path(output_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "backbone_warning.log").open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def _backbone_root() -> Path:
    configured = os.environ.get("ADAPTIVEAOI_BACKBONE_ROOT", "").strip()
    return Path(configured).expanduser().resolve() if configured else _PROJECT_ROOT / "artifacts" / "models" / "backbones"


def backbone_resource_status(name: str, *, verify_sha256: bool = True) -> dict[str, object]:
    """Return the offline backbone resource status without creating a model.

    Target-domain build readiness must verify the packaged feature weights, but
    it must not depend on an already loaded detection runtime or Memory Bank.
    The same immutable specification is used by ``load_backbone`` so readiness
    cannot disagree with the real build worker.
    """
    normalized = str(name or "").strip().lower()
    spec = _BACKBONE_SPECS.get(normalized)
    if spec is None:
        return {
            "ready": False,
            "backbone": normalized,
            "reason_code": "unsupported_backbone",
            "reason": f"不支持的特征提取器：{normalized or '未选择'}。",
            "resolution": "请在“模型与构建配置”中选择系统支持的特征提取器。",
        }
    root = _backbone_root()
    manifest_path = root / "manifest.json"
    weight_path = root / str(spec["filename"])
    details: dict[str, object] = {
        "backbone": normalized,
        "manifest_present": manifest_path.is_file(),
        "weight_present": weight_path.is_file(),
        "expected_size": int(spec["size"]),
        "expected_sha256": str(spec["sha256"]),
        "source": "artifacts/models/backbones",
    }
    if not manifest_path.is_file():
        return {
            **details,
            "ready": False,
            "reason_code": "manifest_missing",
            "reason": "基础特征模型清单缺失，无法确认本地权重版本。",
            "resolution": "请恢复正式安装包中的模型清单与受控权重文件。",
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_spec = dict((manifest.get("weights") or {}).get(normalized) or {})
    except (OSError, ValueError, TypeError):
        manifest_spec = {}
    if (
        str(manifest_spec.get("filename") or "") != str(spec["filename"])
        or int(manifest_spec.get("size_bytes") or 0) != int(spec["size"])
        or str(manifest_spec.get("sha256") or "").lower() != str(spec["sha256"]).lower()
    ):
        return {
            **details,
            "ready": False,
            "reason_code": "manifest_mismatch",
            "reason": "基础特征模型清单与当前算法版本不匹配。",
            "resolution": "请恢复与当前正式系统版本配套的模型清单。",
        }
    if not weight_path.is_file():
        return {
            **details,
            "ready": False,
            "reason_code": "weight_missing",
            "reason": f"未找到所选特征提取器的本地权重文件：{spec['filename']}。",
            "resolution": "请恢复正式安装包中的受控权重，或选择本机已完整安装的特征提取器。",
        }
    try:
        if verify_sha256:
            _verify_packaged_weight(str(weight_path), int(spec["size"]), str(spec["sha256"]))
        elif weight_path.stat().st_size != int(spec["size"]):
            raise RuntimeError("weight size mismatch")
    except (OSError, RuntimeError) as exc:
        return {
            **details,
            "ready": False,
            "reason_code": "weight_integrity_failed",
            "reason": f"基础特征模型完整性校验失败：{exc}",
            "resolution": "请恢复正式安装包中的原始权重文件后重新检查。",
        }
    return {
        **details,
        "ready": True,
        "reason_code": "ready",
        "reason": "受控本地权重、模型清单与 SHA-256 校验均已通过。",
        "resolution": "无需处理。",
    }


@lru_cache(maxsize=8)
def _verify_packaged_weight(path_text: str, expected_size: int, expected_sha256: str) -> Path:
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"受控骨干权重不存在：{path}")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(f"受控骨干权重大小不一致：{path.name}，期望 {expected_size}，实际 {actual_size}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest().lower()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"受控骨干权重 SHA-256 校验失败：{path.name}")
    return path


def load_backbone(name: str, output_root: str | Path) -> tuple[nn.Module, str]:
    name = name.lower()
    spec = _BACKBONE_SPECS.get(name)
    if spec is None:
        raise ValueError(f"不支持的骨干网络：{name}。生产推理不允许自动切换骨干网络。")
    weight_path = _backbone_root() / str(spec["filename"])
    try:
        verified_path = _verify_packaged_weight(str(weight_path), int(spec["size"]), str(spec["sha256"]))
        state_dict = torch.load(verified_path, map_location="cpu", weights_only=True)
        model = spec["factory"](weights=None)
        model.load_state_dict(state_dict, strict=True)
        return model, name
    except Exception as exc:
        message = f"受控骨干权重加载失败，已拒绝随机权重退化：{type(exc).__name__}: {exc}"
        _append_warning(output_root, message)
        raise RuntimeError(message) from exc


class PatchFeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: Iterable[str] = DEFAULT_FEATURE_LAYERS,
        device: str | torch.device = "cuda",
        output_root: str | Path = "outputs",
        use_fp16: bool = False,
        non_blocking_transfer: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.layers = list(layers)
        model, actual_name = load_backbone(backbone, output_root)
        self.backbone_name = actual_name
        return_nodes = {layer: layer for layer in self.layers}
        self.model = create_feature_extractor(model, return_nodes=return_nodes).to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.use_fp16 = bool(use_fp16 and self.device.type == "cuda")
        self.non_blocking_transfer = bool(non_blocking_transfer and self.device.type == "cuda")
        self._last_precision_metadata: dict[str, object] = {
            "device": self.device.type,
            "amp_enabled": self.use_fp16,
            "compute_precision": "fp16" if self.use_fp16 else "fp32",
            "input_dtype": "unknown",
            "backbone_output_dtype": "unknown",
            "feature_dtype": "unknown",
        }

    def precision_metadata(self) -> dict[str, object]:
        return dict(self._last_precision_metadata)

    @torch.inference_mode()
    def forward(self, images: torch.Tensor, timer=None) -> tuple[torch.Tensor, tuple[int, int]]:
        host_to_device_timer = timer.track("host_to_device_ms") if timer is not None else nullcontext()
        with host_to_device_timer:
            images = images.to(self.device, non_blocking=self.non_blocking_transfer)
        autocast_device = "cuda" if self.device.type == "cuda" else "cpu"
        feature_timer = timer.track("feature_ms") if timer is not None else nullcontext()
        with feature_timer:
            with torch.autocast(device_type=autocast_device, dtype=torch.float16, enabled=self.use_fp16):
                features = self.model(images)
                ordered = [features[layer] for layer in self.layers]
                backbone_output_dtype = dtype_name(ordered[0].dtype) if ordered else "unknown"
                target_hw = ordered[0].shape[-2:]
                aligned = [
                    feat if feat.shape[-2:] == target_hw else F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)
                    for feat in ordered
                ]
                feat = torch.cat(aligned, dim=1)
                feat = F.normalize(feat.float(), p=2, dim=1)
            b, c, h, w = feat.shape
            patches = feat.permute(0, 2, 3, 1).reshape(b, h * w, c).contiguous()
            self._last_precision_metadata = {
                "device": self.device.type,
                "amp_enabled": self.use_fp16,
                "compute_precision": "fp16" if self.use_fp16 else "fp32",
                "input_dtype": dtype_name(images.dtype),
                "backbone_output_dtype": backbone_output_dtype,
                # Normalization intentionally returns FP32 for stable cosine
                # geometry even when the backbone convolution uses autocast.
                "feature_dtype": dtype_name(patches.dtype),
            }
        return patches, (h, w)
