"""Defect-guided spatial pooling selection using only normal-bank distances."""
from __future__ import annotations

from typing import Iterable

import numpy as np


POOLING_CANDIDATES = (0.005, 0.01, 0.02, 0.05)


def extract_pooling_features(values: Iterable[float] | np.ndarray, ratio: float) -> dict[str, float]:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float32).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"top_mean": 0.0, "p95": 0.0, "maximum": 0.0, "std": 0.0, "top_k": 0.0}
    keep = max(1, int(np.ceil(array.size * float(ratio))))
    top = np.partition(array, array.size - keep)[-keep:]
    return {
        "top_mean": float(np.mean(top)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
        "std": float(np.std(array)),
        "top_k": float(keep),
    }


def select_pooling_ratio(normal_rows: list[np.ndarray], defect_rows: list[np.ndarray], normalizer: dict, candidates: Iterable[float] = POOLING_CANDIDATES) -> tuple[float, list[dict]]:
    """Select a ratio by a deterministic separation proxy.

    The selector uses only calibration labels and patch distances.  The
    returned rows are persisted in the calibration summary for auditability.
    """
    normal_center = float(normalizer.get("image_center", 0.0))
    normal_scale = max(float(normalizer.get("image_scale", 1.0)), 1e-6)
    rows: list[dict] = []
    candidate_values = [float(value) for value in candidates]
    if not candidate_values:
        candidate_values = [0.01]
    best = (float("-inf"), float(candidate_values[0]))
    for ratio in candidate_values:
        normal_values = [extract_pooling_features(row, ratio)["top_mean"] for row in normal_rows]
        defect_values = [extract_pooling_features(row, ratio)["top_mean"] for row in defect_rows]
        normal_mean = float(np.mean(normal_values)) if normal_values else normal_center
        defect_mean = float(np.mean(defect_values)) if defect_values else normal_mean
        separation = (defect_mean - normal_mean) / normal_scale if defect_values else 0.0
        rows.append({"pooling": "top_k_mean", "top_ratio": ratio, "normal_mean": normal_mean, "defect_mean": defect_mean, "separation_proxy": separation})
        ordering = (separation, -abs(ratio - 0.01))
        if ordering > (best[0], -abs(best[1] - 0.01)):
            best = (separation, ratio)
    return float(best[1]), rows
