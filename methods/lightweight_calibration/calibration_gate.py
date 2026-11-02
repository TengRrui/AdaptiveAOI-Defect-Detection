"""Safety gate and explicit baseline fallback for lite calibration."""
from __future__ import annotations

import numpy as np


def fit_gate(normal_signal: np.ndarray, *, percentile: float = 95.0) -> dict:
    values = np.asarray(normal_signal, dtype=np.float32).reshape(-1)
    values = values[np.isfinite(values)]
    threshold = float(np.quantile(values, percentile / 100.0)) if values.size else 0.0
    return {
        "method": "normal_quantile_safety_gate_v1",
        "signal_threshold": threshold,
        "normal_percentile": float(percentile),
        "fallback": "baseline_score",
        "parameter_count": 1,
    }


def apply_gate(signal: float, gate: dict) -> tuple[bool, str]:
    if not np.isfinite(float(signal)):
        return False, "invalid_signal_fallback"
    threshold = float(gate.get("signal_threshold", 0.0))
    if float(signal) >= threshold:
        return True, "calibration_applied"
    return False, "below_normal_tail_fallback"
