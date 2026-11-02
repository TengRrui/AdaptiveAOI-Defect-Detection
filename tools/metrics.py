from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def safe_auroc(y_true, y_score) -> float:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return math.nan
    return float(roc_auc_score(y_true, y_score))


def binary_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def read_mask_flat(path: str) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    return (arr > 0).astype(np.uint8).reshape(-1)


def read_heatmap_flat(path: str) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    return arr.astype(np.float32).reshape(-1) / 255.0


def evaluate_result_csv(result_csv: str | Path) -> dict[str, float]:
    df = pd.read_csv(result_csv)
    image = binary_metrics(df["label_gt"], df["pred_label"])
    out = {
        "I-AUROC": safe_auroc(df["label_gt"], df["score"]),
        "image_F1": image["f1"],
        "Precision": image["precision"],
        "Recall": image["recall"],
        "mean_ms": float(df["total_ms"].mean()),
        "p95_ms": float(df["total_ms"].quantile(0.95)),
    }
    pixel_y = []
    pixel_score = []
    pixel_pred = []
    for item in df.to_dict("records"):
        mask_path = str(item.get("gt_mask_path", ""))
        heatmap_path = str(item.get("heatmap_path", ""))
        candidate_pred = str(item.get("mask_path", ""))
        if mask_path.lower() == "nan":
            mask_path = ""
        if heatmap_path.lower() == "nan":
            heatmap_path = ""
        if candidate_pred.lower() == "nan":
            candidate_pred = ""
        if mask_path and heatmap_path and Path(mask_path).exists() and Path(heatmap_path).exists():
            pixel_y.append(read_mask_flat(mask_path))
            pixel_score.append(read_heatmap_flat(heatmap_path))
            if candidate_pred and Path(candidate_pred).exists() and candidate_pred != mask_path:
                pixel_pred.append(read_mask_flat(candidate_pred))
    if pixel_y and pixel_score:
        y = np.concatenate(pixel_y)
        s = np.concatenate(pixel_score)
        out["P-AUROC"] = safe_auroc(y, s)
        if pixel_pred:
            p = np.concatenate(pixel_pred)
            pix = binary_metrics(y, p)
            out["pixel_F1"] = pix["f1"]
        else:
            out["pixel_F1"] = math.nan
    else:
        out["P-AUROC"] = math.nan
        out["pixel_F1"] = math.nan
    return out
