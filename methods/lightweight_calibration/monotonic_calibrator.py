"""Tiny non-negative monotonic calibrator."""
from __future__ import annotations

import numpy as np


FEATURE_NAMES = ("baseline_z", "top_mean_z", "p95_z", "maximum_z")


def _projected_fit(features: np.ndarray, labels: np.ndarray, l2: float, steps: int = 180) -> tuple[np.ndarray, float]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.ones(4, dtype=np.float32) / 4.0, 0.0
    w = np.ones(x.shape[1], dtype=np.float64) / max(1, x.shape[1])
    b = float(np.mean(y) - np.mean(x @ w))
    lr = 0.05 / max(1.0, float(np.linalg.norm(x, ord=2) ** 2 / max(1, x.shape[0])))
    for _ in range(int(steps)):
        pred = x @ w + b
        residual = pred - y
        grad_w = (x.T @ residual) / max(1, x.shape[0]) + float(l2) * w
        grad_b = float(np.mean(residual))
        w = np.maximum(0.0, w - lr * grad_w)
        b -= lr * grad_b
    return w.astype(np.float32), float(b)


def fit_monotonic_calibrator(features: np.ndarray, labels: np.ndarray, *, l2: float = 0.01, score_scale: float = 1.0) -> dict:
    weights, bias = _projected_fit(features, labels, l2)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-8:
        weights = np.zeros_like(weights)
        weights[0] = 1.0
        weight_sum = 1.0
    weights = weights / weight_sum
    return {
        "method": "nonnegative_linear_monotonic_v1",
        "feature_names": list(FEATURE_NAMES),
        "weights": [float(value) for value in weights],
        "bias": float(bias),
        "score_scale": max(float(score_scale), 1e-6),
        "l2": float(l2),
        "parameter_count": int(len(weights) + 2),
        "monotonic_nonnegative": True,
    }


def apply_monotonic_calibrator(features: np.ndarray, calibrator: dict) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    weights = np.asarray(calibrator.get("weights") or [1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != weights.shape[0]:
        weights = np.resize(weights, x.shape[1])
    weights = np.maximum(weights, 0.0)
    return (x @ weights + float(calibrator.get("bias", 0.0))).astype(np.float32)
