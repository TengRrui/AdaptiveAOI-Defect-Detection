from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from core.anomaly_scorer import image_score, patch_scores_to_map
from methods.defect_calibration.prototype import (
    l2_normalize,
    mask_to_patch_selector,
    projection_scores,
    prototype,
    topk_selector,
)


class DefectDirection:
    def __init__(self, normal_proto: torch.Tensor, defect_proto: torch.Tensor, alpha: float = 0.5, metadata: dict | None = None):
        self.normal_proto = normal_proto.float().cpu()
        self.defect_proto = defect_proto.float().cpu()
        self.direction = l2_normalize(self.defect_proto - self.normal_proto).float().cpu()
        self.alpha = float(alpha)
        self.metadata = metadata or {}

    def score_patches(self, patch_features: torch.Tensor) -> np.ndarray:
        return projection_scores(patch_features.detach().cpu(), self.normal_proto, self.direction)

    def score_map(self, patch_features: torch.Tensor, patch_shape: tuple[int, int]) -> np.ndarray:
        scores = self.score_patches(patch_features)
        return scores.reshape(patch_shape)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "normal_proto": self.normal_proto,
                "defect_proto": self.defect_proto,
                "direction": self.direction,
                "alpha": self.alpha,
                "metadata": self.metadata,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        obj = cls(payload["normal_proto"], payload["defect_proto"], payload.get("alpha", 0.5), payload.get("metadata", {}))
        if "direction" in payload:
            obj.direction = payload["direction"].float().cpu()
        return obj


def build_defect_direction(
    model,
    calibration_rows,
    top_ratio: float = 0.05,
    max_normal_patches: int = 20000,
    max_defect_patches: int = 20000,
    seed: int = 0,
    alpha: float = 0.5,
) -> tuple[DefectDirection, dict]:
    rng = np.random.default_rng(seed)
    normal_chunks = []
    defect_chunks = []
    used_mask = 0
    used_topk = 0

    for item in calibration_rows.to_dict("records"):
        from datasets.transforms import image_transform, load_rgb

        transform = image_transform(int(model.config["image_size"]))
        tensor = transform(load_rgb(item["image_path"]))
        patches, patch_shape = model.extract(tensor)
        patch_features = patches[0].detach().cpu()

        if int(item["label"]) == 0:
            normal_chunks.append(patch_features)
            continue

        selector = None
        mask_path = str(item.get("mask_path", ""))
        if mask_path and mask_path.lower() != "nan" and Path(mask_path).exists():
            selector = mask_to_patch_selector(mask_path, patch_shape)
            used_mask += 1
        else:
            if model.memory_bank is not None:
                memory_scores = model.memory_bank.score(patch_features)
                if isinstance(memory_scores, torch.Tensor):
                    memory_scores = memory_scores.detach().cpu().numpy()
                score_map = patch_scores_to_map(memory_scores, patch_shape)
                selector = topk_selector(score_map, top_ratio=top_ratio)
                used_topk += 1
        if selector is None or not selector.any():
            selector = np.ones(patch_features.shape[0], dtype=bool)
        defect_chunks.append(patch_features[selector])

    if not normal_chunks:
        raise RuntimeError("No normal samples available for defect direction calibration.")
    if not defect_chunks:
        raise RuntimeError("No defect samples available for defect direction calibration.")

    normal_features = torch.cat(normal_chunks, dim=0)
    defect_features = torch.cat(defect_chunks, dim=0)
    if normal_features.shape[0] > max_normal_patches:
        idx = rng.choice(normal_features.shape[0], size=max_normal_patches, replace=False)
        normal_features = normal_features[idx]
    if defect_features.shape[0] > max_defect_patches:
        idx = rng.choice(defect_features.shape[0], size=max_defect_patches, replace=False)
        defect_features = defect_features[idx]

    normal_proto = prototype(normal_features)
    defect_proto = prototype(defect_features)
    direction = DefectDirection(normal_proto, defect_proto, alpha=alpha)
    metadata = {
        "normal_patch_count": int(normal_features.shape[0]),
        "defect_patch_count": int(defect_features.shape[0]),
        "used_mask_samples": int(used_mask),
        "used_topk_samples": int(used_topk),
        "top_ratio": float(top_ratio),
        "alpha": float(alpha),
        "defect_direction_norm": float(torch.norm(direction.direction).item()),
        "normal_defect_cosine": float(torch.dot(normal_proto, defect_proto).item()),
    }
    direction.metadata = metadata
    return direction, metadata


def search_alpha(base_scores: list[float], defect_scores: list[float], labels: list[int], candidates: list[float] | None = None) -> tuple[float, float]:
    from core.calibration import best_f1_threshold

    candidates = candidates or [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    best_alpha = 0.0
    best_f1 = -1.0
    b = np.asarray(base_scores, dtype=np.float32)
    d = np.asarray(defect_scores, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    for alpha in candidates:
        _, f1 = best_f1_threshold(y, b + float(alpha) * d)
        if f1 > best_f1:
            best_alpha = float(alpha)
            best_f1 = float(f1)
    return best_alpha, best_f1


def save_defect_calibration_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
