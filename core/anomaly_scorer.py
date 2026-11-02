from __future__ import annotations

import numpy as np


def patch_scores_to_map(scores: np.ndarray, patch_shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(scores, dtype=np.float32).reshape(patch_shape)


def image_score(score_map: np.ndarray, top_ratio: float = 0.01) -> float:
    flat = np.asarray(score_map, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return 0.0
    k = max(1, int(flat.size * top_ratio))
    return float(np.partition(flat, -k)[-k:].mean())


def percentile_threshold(scores: list[float], percentile: float = 99.0) -> float:
    if not scores:
        return 0.5
    return float(np.percentile(np.asarray(scores, dtype=np.float32), percentile))
