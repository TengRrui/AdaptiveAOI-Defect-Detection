"""Normal-only robust score normalization.

The normal image and patch distributions are summarized with median/MAD and
quantiles.  The implementation is intentionally dependency-light and keeps
all values scalar so the resulting profile is safe to freeze and audit.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def _finite(values: Iterable[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float32)
    array = array[np.isfinite(array)]
    return array.reshape(-1)


def _center_scale(values: np.ndarray) -> tuple[float, float]:
    values = _finite(values)
    if values.size == 0:
        return 0.0, 1.0
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    q75, q25 = np.percentile(values, [75.0, 25.0]) if values.size > 1 else (center, center)
    scale = max(1.4826 * mad, float((q75 - q25) / 1.349), 1e-6)
    return center, scale


def fit_robust_normalizer(normal_patch_rows: list[np.ndarray], normal_image_scores: list[float]) -> dict:
    """Fit scalar normal-only statistics; no defect values are accepted."""
    patch_values = np.concatenate([_finite(row) for row in normal_patch_rows if _finite(row).size], axis=0) if normal_patch_rows else np.empty(0, dtype=np.float32)
    patch_center, patch_scale = _center_scale(patch_values)
    image_values = _finite(normal_image_scores)
    image_center, image_scale = _center_scale(image_values)
    image_q95 = float(np.quantile(image_values, 0.95)) if image_values.size else image_center + 2.0 * image_scale
    patch_q95 = float(np.quantile(patch_values, 0.95)) if patch_values.size else patch_center + 2.0 * patch_scale
    return {
        "method": "median_mad_quantile_normal_only_v1",
        "patch_center": patch_center,
        "patch_scale": patch_scale,
        "patch_q95": patch_q95,
        "image_center": image_center,
        "image_scale": image_scale,
        "image_q95": image_q95,
        "normal_sample_count": int(len(normal_patch_rows)),
        "normal_patch_count": int(patch_values.size),
    }


def normalize_patch_scores(values: np.ndarray, normalizer: dict) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    center = float(normalizer.get("patch_center", 0.0))
    scale = max(float(normalizer.get("patch_scale", 1.0)), 1e-6)
    return (array - center) / scale


def normalize_image_score(value: float, normalizer: dict) -> float:
    center = float(normalizer.get("image_center", 0.0))
    scale = max(float(normalizer.get("image_scale", 1.0)), 1e-6)
    return float((float(value) - center) / scale)
