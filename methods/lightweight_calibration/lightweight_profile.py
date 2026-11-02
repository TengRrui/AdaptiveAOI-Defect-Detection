"""Fit/apply the frozen NMDG-LC profile."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .calibration_gate import apply_gate, fit_gate
from .monotonic_calibrator import apply_monotonic_calibrator, fit_monotonic_calibrator
from .robust_score_normalizer import fit_robust_normalizer, normalize_image_score, normalize_patch_scores
from .spatial_score_pooling import POOLING_CANDIDATES, extract_pooling_features, select_pooling_ratio


MODE = "normal_memory_defect_guided_lite"
DISPLAY_NAME = "单正常记忆·缺陷引导轻量校准"
VERSION = "nmdg-lc-v1"


def _feature_row(patch_scores: np.ndarray, baseline_score: float, normalizer: dict, pooling: dict) -> tuple[np.ndarray, dict]:
    pooled = extract_pooling_features(patch_scores, float(pooling.get("top_ratio", 0.01)))
    patch_z = normalize_patch_scores(patch_scores, normalizer)
    patch_scale = max(float(normalizer.get("patch_scale", 1.0)), 1e-6)
    image_scale = max(float(normalizer.get("image_scale", 1.0)), 1e-6)
    image_center = float(normalizer.get("image_center", 0.0))
    feature = np.asarray(
        [
            max(0.0, normalize_image_score(baseline_score, normalizer)),
            max(0.0, (float(pooled["top_mean"]) - image_center) / image_scale),
            max(0.0, (float(pooled["p95"]) - float(normalizer.get("patch_center", 0.0))) / patch_scale),
            max(0.0, (float(pooled["maximum"]) - float(normalizer.get("patch_center", 0.0))) / patch_scale),
        ],
        dtype=np.float32,
    )
    return feature, pooled


def _apply_rows(patch_rows: list[np.ndarray], base_scores: list[float], normalizer: dict, pooling: dict, calibrator: dict, gate: dict, score_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    features: list[np.ndarray] = []
    pooled_rows: list[dict] = []
    for row, base in zip(patch_rows, base_scores):
        feature, pooled = _feature_row(row, float(base), normalizer, pooling)
        features.append(feature)
        pooled_rows.append(pooled)
    feature_array = np.asarray(features, dtype=np.float32)
    signals = apply_monotonic_calibrator(feature_array, calibrator)
    gate_threshold = float(gate.get("signal_threshold", 0.0))
    applied = signals >= gate_threshold
    calibrated = np.asarray(base_scores, dtype=np.float32) + float(score_scale) * np.maximum(signals - gate_threshold, 0.0)
    final = np.where(applied, calibrated, np.asarray(base_scores, dtype=np.float32))
    return final.astype(np.float32), signals.astype(np.float32), applied.astype(bool), pooled_rows


def _stratified_cv_summary(features: np.ndarray, labels: np.ndarray, *, l2: float, score_scale: float, seed: int, requested_folds: int = 5) -> dict[str, Any]:
    """Evaluate the tiny calibrator without retaining fold-level sample data."""
    # These metrics are needed only while fitting a new profile.  Importing
    # scikit-learn during formal inference startup used to add a large native
    # import tail even though loading a frozen JSON profile never evaluates CV.
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    normal_indices = np.flatnonzero(labels == 0)
    defect_indices = np.flatnonzero(labels == 1)
    fold_count = min(int(requested_folds), int(len(normal_indices)), int(len(defect_indices)))
    if fold_count < 2:
        return {"method": "stratified_kfold", "fold_count": 0, "status": "insufficient_samples"}
    rng = np.random.default_rng(seed)
    normal_folds = np.array_split(rng.permutation(normal_indices), fold_count)
    defect_folds = np.array_split(rng.permutation(defect_indices), fold_count)
    fold_rows: list[dict[str, float]] = []
    for fold in range(fold_count):
        validation = np.concatenate([normal_folds[fold], defect_folds[fold]])
        training = np.setdiff1d(np.arange(len(labels)), validation, assume_unique=False)
        calibrator = fit_monotonic_calibrator(features[training], labels[training], l2=l2, score_scale=score_scale)
        train_signal = apply_monotonic_calibrator(features[training], calibrator)
        normal_train_signal = train_signal[labels[training] == 0]
        validation_signal = apply_monotonic_calibrator(features[validation], calibrator)
        threshold = float(np.percentile(normal_train_signal, 95.0)) if normal_train_signal.size else 0.0
        validation_labels = labels[validation].astype(np.int32)
        validation_prediction = (validation_signal >= threshold).astype(np.int32)
        fold_rows.append(
            {
                "auroc": float(roc_auc_score(validation_labels, validation_signal)),
                "average_precision": float(average_precision_score(validation_labels, validation_signal)),
                "f1": float(f1_score(validation_labels, validation_prediction, zero_division=0)),
            }
        )
    return {
        "method": "stratified_kfold",
        "fold_count": fold_count,
        "seed": int(seed),
        "threshold_rule": "95th_percentile_of_training_normal_signal",
        "auroc_mean": float(np.mean([row["auroc"] for row in fold_rows])),
        "average_precision_mean": float(np.mean([row["average_precision"] for row in fold_rows])),
        "f1_mean": float(np.mean([row["f1"] for row in fold_rows])),
        "sample_count": int(len(labels)),
        "normal_sample_count": int(len(normal_indices)),
        "defect_sample_count": int(len(defect_indices)),
    }


def fit_lightweight_profile(
    normal_patch_rows: list[np.ndarray],
    defect_patch_rows: list[np.ndarray],
    normal_base_scores: list[float],
    defect_base_scores: list[float],
    *,
    threshold: float,
    threshold_strategy: str = "best_f1",
    config: dict[str, Any] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit the tiny profile while retaining no defect features on disk."""
    del threshold_strategy
    config = dict(config or {})
    normalizer = fit_robust_normalizer(normal_patch_rows, normal_base_scores)
    selected_ratio, pooling_candidates = select_pooling_ratio(normal_patch_rows, defect_patch_rows, normalizer, config.get("lightweight_pooling_candidates", POOLING_CANDIDATES))
    pooling = {"method": "top_k_mean_plus_tail_v1", "top_ratio": selected_ratio, "candidate_ratios": [float(value) for value in config.get("lightweight_pooling_candidates", POOLING_CANDIDATES)]}

    all_rows = list(normal_patch_rows) + list(defect_patch_rows)
    all_base = list(normal_base_scores) + list(defect_base_scores)
    labels = np.asarray([0] * len(normal_patch_rows) + [1] * len(defect_patch_rows), dtype=np.float32)
    features = np.asarray([_feature_row(row, base, normalizer, pooling)[0] for row, base in zip(all_rows, all_base)], dtype=np.float32)
    calibrator = fit_monotonic_calibrator(features, labels, l2=float(config.get("lightweight_l2", 0.01)), score_scale=float(normalizer.get("image_scale", 1.0)))
    cv_summary = _stratified_cv_summary(
        features,
        labels,
        l2=float(config.get("lightweight_l2", 0.01)),
        score_scale=float(normalizer.get("image_scale", 1.0)),
        seed=seed,
        requested_folds=int(config.get("lightweight_cv_folds", 5)),
    )
    signals = apply_monotonic_calibrator(features, calibrator)
    gate = fit_gate(signals[: len(normal_patch_rows)], percentile=float(config.get("lightweight_gate_percentile", 95.0)))
    score_scale = max(float(config.get("lightweight_score_scale", normalizer.get("image_scale", 1.0))), 1e-6)
    final_scores, signal_values, applied, pooled_rows = _apply_rows(all_rows, all_base, normalizer, pooling, calibrator, gate, score_scale)
    normal_signal = signal_values[: len(normal_patch_rows)]
    defect_signal = signal_values[len(normal_patch_rows) :]
    profile = {
        "profile_version": VERSION,
        "method_mode": MODE,
        "method_name": DISPLAY_NAME,
        "normal_bank_id": str(config.get("normal_bank_id") or ""),
        "normalizer": normalizer,
        "spatial_pooling": pooling,
        "calibrator": calibrator,
        "calibration_gate": {**gate, "score_scale": score_scale},
        "image_threshold": float(threshold),
        "baseline_threshold": float(config.get("baseline_threshold", threshold)),
        "calibration_method": "normal_tail_mad_plus_defect_guided_nonnegative_linear",
        "normal_sample_count": int(len(normal_patch_rows)),
        "defect_calibration_sample_count": int(len(defect_patch_rows)),
        "random_seed": int(seed),
        "frozen": True,
        "test_time_update_allowed": False,
        "defect_features_saved": False,
        "defect_images_saved": False,
        "additional_ann_count": 0,
        "parameter_count": int(calibrator.get("parameter_count", 0) + gate.get("parameter_count", 0) + 6),
        "config_hash": str(config.get("config_hash") or ""),
        "created_at": str(config.get("created_at") or ""),
    }
    profile["calibration_metrics"] = {
        "normal_signal_mean": float(np.mean(normal_signal)) if normal_signal.size else 0.0,
        "defect_signal_mean": float(np.mean(defect_signal)) if defect_signal.size else 0.0,
        "gate_applied_normal_count": int(np.sum(applied[: len(normal_patch_rows)])),
        "gate_applied_defect_count": int(np.sum(applied[len(normal_patch_rows) :])),
        "final_scores": [float(value) for value in final_scores],
        "signals": [float(value) for value in signal_values],
        "pooled_features": pooled_rows,
        "cross_validation": cv_summary,
    }
    profile["component_comparison"] = pooling_candidates
    return profile


def apply_lightweight_profile(patch_scores: np.ndarray, baseline_score: float, profile: dict[str, Any]) -> dict[str, Any]:
    normalizer = profile.get("normalizer") or {}
    pooling = profile.get("spatial_pooling") or {"top_ratio": 0.01}
    calibrator = profile.get("calibrator") or {}
    gate = profile.get("calibration_gate") or {}
    feature, pooled = _feature_row(np.asarray(patch_scores, dtype=np.float32), float(baseline_score), normalizer, pooling)
    signal = float(apply_monotonic_calibrator(feature, calibrator)[0])
    enabled, gate_status = apply_gate(signal, gate)
    score_scale = max(float(gate.get("score_scale", normalizer.get("image_scale", 1.0))), 1e-6)
    calibrated = float(float(baseline_score) + score_scale * max(signal - float(gate.get("signal_threshold", 0.0)), 0.0))
    final = calibrated if enabled else float(baseline_score)
    return {
        "baseline_score": float(baseline_score),
        "calibrated_score": float(calibrated),
        "final_score": float(final),
        "calibration_applied": bool(enabled),
        "gate_status": gate_status,
        "fallback_used": not bool(enabled),
        "calibration_version": str(profile.get("profile_version") or VERSION),
        "calibration_signal": signal,
        "pooling_top_ratio": float(pooling.get("top_ratio", 0.01)),
        "pooling_top_mean": float(pooled.get("top_mean", 0.0)),
    }


def load_lightweight_profile(path: str | Path) -> dict[str, Any] | None:
    candidate = Path(path)
    if not candidate.is_file():
        return None
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("method_mode") != MODE or not value.get("frozen"):
        return None
    return value


def write_lightweight_profile(path: str | Path, profile: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The persisted contract contains only frozen scalar metadata.  Keep
    # calibration summaries useful for audit, but never serialize per-sample
    # signals, pooled vectors, feature tensors, image bytes, or test arrays.
    payload = dict(profile)
    metrics = dict(payload.get("calibration_metrics") or {})
    for field in ("final_scores", "signals", "pooled_features", "defect_features", "test_scores", "image_paths"):
        metrics.pop(field, None)
        payload.pop(field, None)
    payload["calibration_metrics"] = metrics
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return destination
