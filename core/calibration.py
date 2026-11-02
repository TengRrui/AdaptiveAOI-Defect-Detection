from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


def sample_calibration_rows(
    manifest_df: object,
    normal_shot: int,
    defect_shot: int,
    seed: int = 0,
) -> object:
    import pandas as pd

    rng = np.random.default_rng(seed)
    normal_pool = manifest_df[manifest_df["label"].astype(int) == 0]
    train_normal = normal_pool[normal_pool["split"].astype(str).str.lower() == "train"]
    if len(train_normal) > 0:
        normal_pool = train_normal
    defect_pool = manifest_df[manifest_df["label"].astype(int) == 1]

    normal_n = min(int(normal_shot), len(normal_pool))
    defect_n = min(int(defect_shot), len(defect_pool))
    normal_idx = rng.choice(normal_pool.index.to_numpy(), size=normal_n, replace=False) if normal_n else []
    defect_idx = rng.choice(defect_pool.index.to_numpy(), size=defect_n, replace=False) if defect_n else []
    return pd.concat([normal_pool.loc[normal_idx], defect_pool.loc[defect_idx]], ignore_index=True)


def threshold_percentile(scores: Iterable[float], percentile: float = 99.0) -> float:
    arr = np.asarray(list(scores), dtype=np.float32)
    if arr.size == 0:
        return 0.5
    return float(np.percentile(arr, float(percentile)))


def threshold_mean_std(scores: Iterable[float], k: float = 3.0) -> float:
    arr = np.asarray(list(scores), dtype=np.float32)
    if arr.size == 0:
        return 0.5
    return float(arr.mean() + float(k) * arr.std())


def best_f1_threshold(y_true: Iterable[int], scores: Iterable[float]) -> tuple[float, float]:
    y = np.asarray(list(y_true), dtype=np.int32)
    s = np.asarray(list(scores), dtype=np.float32)
    if y.size == 0:
        return 0.5, 0.0
    if len(np.unique(y)) < 2:
        return threshold_percentile(s, 99.0), 0.0
    order = np.argsort(-s)
    sorted_scores = s[order]
    sorted_y = y[order]
    tp = np.cumsum(sorted_y)
    fp = np.cumsum(1 - sorted_y)
    positives = int(sorted_y.sum())
    fn = positives - tp
    denom = (2 * tp + fp + fn).astype(np.float32)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(denom, dtype=np.float32), where=denom > 0)
    unique_ends = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    candidate_idx = np.where(unique_ends)[0]
    best_idx = int(candidate_idx[np.argmax(f1[candidate_idx])])
    return float(sorted_scores[best_idx]), float(f1[best_idx])


def recall_priority_threshold(
    y_true: Iterable[int],
    scores: Iterable[float],
    target_recall: float = 0.95,
) -> tuple[float, float, float]:
    from sklearn.metrics import precision_recall_curve

    y = np.asarray(list(y_true), dtype=np.int32)
    s = np.asarray(list(scores), dtype=np.float32)
    if y.size == 0 or len(np.unique(y)) < 2:
        return threshold_percentile(s, 99.0), 0.0, 0.0
    precision, recall, thresholds = precision_recall_curve(y, s)
    best = None
    for idx, t in enumerate(thresholds):
        r = float(recall[idx])
        p = float(precision[idx])
        if r >= target_recall and (best is None or p > best[1]):
            best = (float(t), p, r)
    if best is not None:
        return best
    best_idx = int(np.argmax(recall[:-1]))
    return float(thresholds[best_idx]), float(precision[best_idx]), float(recall[best_idx])


def choose_image_threshold(
    labels: Iterable[int],
    scores: Iterable[float],
    strategy: str,
    percentile: float = 99.0,
    mean_std_k: float = 3.0,
    target_recall: float = 0.95,
) -> dict:
    labels = list(labels)
    scores = list(scores)
    strategy = strategy.lower()
    if strategy in {"normal_percentile", "percentile"}:
        normal_scores = [s for y, s in zip(labels, scores) if int(y) == 0]
        threshold = threshold_percentile(normal_scores or scores, percentile)
        return {"strategy": strategy, "threshold": threshold, "score": None}
    if strategy in {"mean_std", "mean+kstd", "mean_std_k"}:
        normal_scores = [s for y, s in zip(labels, scores) if int(y) == 0]
        threshold = threshold_mean_std(normal_scores or scores, mean_std_k)
        return {"strategy": strategy, "threshold": threshold, "score": None}
    if strategy == "best_f1":
        threshold, best_f1 = best_f1_threshold(labels, scores)
        return {"strategy": strategy, "threshold": threshold, "score": best_f1}
    if strategy in {"recall_priority", "recall-priority"}:
        threshold, precision, recall = recall_priority_threshold(labels, scores, target_recall)
        return {"strategy": strategy, "threshold": threshold, "precision": precision, "recall": recall}
    raise ValueError(f"Unsupported threshold strategy: {strategy}")


def choose_pixel_threshold(
    pixel_labels: Iterable[int],
    pixel_scores: Iterable[float],
    fallback_scores: Iterable[float],
    strategy: str,
    percentile: float = 99.0,
) -> dict:
    y = np.asarray(list(pixel_labels), dtype=np.int32)
    s = np.asarray(list(pixel_scores), dtype=np.float32)
    if y.size and len(np.unique(y)) >= 2 and strategy.lower() in {"best_f1", "recall_priority", "recall-priority"}:
        threshold, best_f1 = best_f1_threshold(y, s)
        return {"strategy": "best_f1", "threshold": threshold, "score": best_f1}
    return {"strategy": "percentile_fallback", "threshold": threshold_percentile(fallback_scores, percentile), "score": None}


def save_calibration(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_calibration(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def calibration_dir(config: dict, dataset: str, category: str) -> Path:
    return Path(config["output_root"]) / "calibrations" / dataset / category


def default_calibration_path(config: dict, dataset: str, category: str) -> Path:
    return calibration_dir(config, dataset, category) / "calibration.json"
