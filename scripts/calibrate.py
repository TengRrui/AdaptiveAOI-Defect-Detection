from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.anomaly_scorer import image_score, patch_scores_to_map
from core.calibration import (
    calibration_dir,
    choose_image_threshold,
    choose_pixel_threshold,
    sample_calibration_rows,
    save_calibration,
)
from core.postprocess import resize_heatmap
from datasets.transforms import image_transform, load_rgb
from methods.defect_calibration.defect_direction import DefectDirection, save_defect_calibration_json, search_alpha
from methods.defect_calibration.prototype import mask_to_patch_selector, prototype, topk_selector
from methods.patchcore.build_memory import build_memory
from methods.patchcore.infer import default_memory_path
from methods.patchcore.model import FastPatchCoreAOI
from tools.path_utils import append_error_report, load_config


def load_mask_binary(mask_path: str, height: int, width: int) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    arr = (np.array(mask) > 0).astype(np.uint8)
    if arr.shape != (height, width):
        arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_NEAREST)
    return arr.reshape(-1)


def sample_tensor_rows(features: torch.Tensor, max_rows: int, rng) -> torch.Tensor:
    if features.shape[0] <= max_rows:
        return features
    idx = rng.choice(features.shape[0], size=max_rows, replace=False)
    return features[idx]


def sample_pixels(labels: np.ndarray, scores: np.ndarray, max_pixels: int, rng) -> tuple[np.ndarray, np.ndarray]:
    if labels.size <= max_pixels:
        return labels, scores
    idx = rng.choice(labels.size, size=max_pixels, replace=False)
    return labels[idx], scores[idx]


def extract_cache(model, rows: pd.DataFrame):
    transform = image_transform(int(model.config["image_size"]))
    cache = []
    for item in rows.to_dict("records"):
        image = load_rgb(item["image_path"])
        width, height = image.size
        tensor = transform(image)
        patches, patch_shape = model.extract(tensor)
        patch_features = patches[0].detach().cpu()
        memory_scores = model.memory_bank.score(patch_features)
        memory_map = patch_scores_to_map(memory_scores, patch_shape)
        cache.append(
            {
                "item": item,
                "label": int(item["label"]),
                "height": int(height),
                "width": int(width),
                "patch_shape": patch_shape,
                "patch_features": patch_features.half(),
                "memory_map": memory_map,
                "base_score": image_score(memory_map),
            }
        )
    return cache


def build_direction_from_cache(cache, top_ratio: float, seed: int, alpha: float):
    rng = np.random.default_rng(seed)
    normal_chunks = []
    defect_chunks = []
    used_mask = 0
    used_topk = 0
    for entry in cache:
        features = entry["patch_features"].float()
        if entry["label"] == 0:
            normal_chunks.append(sample_tensor_rows(features, 512, rng))
            continue
        item = entry["item"]
        selector = None
        mask_path = str(item.get("mask_path", ""))
        if mask_path and mask_path.lower() != "nan" and Path(mask_path).exists():
            selector = mask_to_patch_selector(mask_path, entry["patch_shape"])
            used_mask += 1
        else:
            selector = topk_selector(entry["memory_map"], top_ratio=top_ratio)
            used_topk += 1
        if selector is None or not selector.any():
            selector = np.ones(features.shape[0], dtype=bool)
        defect_chunks.append(features[selector])

    if not normal_chunks:
        raise RuntimeError("No normal features available for defect direction.")
    if not defect_chunks:
        raise RuntimeError("No defect features available for defect direction.")
    normal_features = sample_tensor_rows(torch.cat(normal_chunks, dim=0), 20000, rng)
    defect_features = sample_tensor_rows(torch.cat(defect_chunks, dim=0), 20000, rng)
    normal_proto = prototype(normal_features)
    defect_proto = prototype(defect_features)
    direction = DefectDirection(normal_proto, defect_proto, alpha=alpha)
    meta = {
        "normal_patch_count": int(normal_features.shape[0]),
        "defect_patch_count": int(defect_features.shape[0]),
        "used_mask_samples": int(used_mask),
        "used_topk_samples": int(used_topk),
        "top_ratio": float(top_ratio),
        "alpha": float(alpha),
        "defect_direction_norm": float(torch.norm(direction.direction).item()),
        "normal_defect_cosine": float(torch.dot(normal_proto, defect_proto).item()),
    }
    direction.metadata = meta
    return direction, meta


def score_cache(cache, direction=None, pixel_sample_per_image: int = 12000, seed: int = 0):
    rng = np.random.default_rng(seed)
    records = []
    pixel_labels = []
    pixel_scores = []
    normal_pixel_scores = []
    for entry in cache:
        item = entry["item"]
        score_map = entry["memory_map"]
        defect_img_score = 0.0
        alpha = 0.0
        if direction is not None:
            defect_map = direction.score_map(entry["patch_features"].float(), entry["patch_shape"])
            defect_img_score = image_score(defect_map)
            alpha = float(direction.alpha)
            score_map = score_map + alpha * defect_map
        heatmap = resize_heatmap(score_map, height=entry["height"], width=entry["width"])
        score = image_score(heatmap)
        records.append(
            {
                "image_path": item["image_path"],
                "label": int(item["label"]),
                "label_name": item["label_name"],
                "mask_path": item.get("mask_path", ""),
                "base_score": entry["base_score"],
                "defect_score": defect_img_score,
                "score": score,
                "alpha": alpha,
            }
        )
        flat_scores = heatmap.reshape(-1).astype(np.float32)
        if int(item["label"]) == 0:
            labels = np.zeros_like(flat_scores, dtype=np.uint8)
            sampled_y, sampled_s = sample_pixels(labels, flat_scores, pixel_sample_per_image, rng)
            pixel_labels.append(sampled_y)
            pixel_scores.append(sampled_s)
            normal_pixel_scores.extend(sampled_s.tolist())
        else:
            mask_path = str(item.get("mask_path", ""))
            if mask_path and mask_path.lower() != "nan" and Path(mask_path).exists():
                labels = load_mask_binary(mask_path, entry["height"], entry["width"])
                sampled_y, sampled_s = sample_pixels(labels, flat_scores, pixel_sample_per_image, rng)
                pixel_labels.append(sampled_y)
                pixel_scores.append(sampled_s)
    pixel_y = np.concatenate(pixel_labels) if pixel_labels else np.array([], dtype=np.uint8)
    pixel_s = np.concatenate(pixel_scores) if pixel_scores else np.array([], dtype=np.float32)
    return pd.DataFrame(records), pixel_y, pixel_s, normal_pixel_scores


def main():
    parser = argparse.ArgumentParser(description="Calibrate thresholds and optional defect direction.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True, choices=["mvtec", "visa", "dagm"])
    parser.add_argument("--category", required=True)
    parser.add_argument("--normal_shot", type=int, default=100)
    parser.add_argument("--defect_shot", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_defect_direction", action="store_true")
    parser.add_argument("--threshold_strategy", default="best_f1", choices=["normal_percentile", "mean_std", "best_f1", "recall_priority"])
    parser.add_argument("--threshold_percentile", type=float, default=99.0)
    parser.add_argument("--mean_std_k", type=float, default=3.0)
    parser.add_argument("--target_recall", type=float, default=0.95)
    parser.add_argument("--alpha", default="auto")
    parser.add_argument("--defect_top_ratio", type=float, default=0.05)
    args = parser.parse_args()

    config = load_config(args.config)
    config["dataset"] = args.dataset
    config["category"] = args.category

    try:
        if not default_memory_path(config, args.dataset, args.category).exists():
            print(f"memory missing, building first: {args.dataset}/{args.category}")
            build_memory(config, category=args.category)

        manifest_path = Path(config["output_root"]) / "results" / f"manifest_{args.dataset}.csv"
        manifest = pd.read_csv(manifest_path)
        manifest = manifest[(manifest["dataset"].astype(str).str.lower() == args.dataset) & (manifest["category"].astype(str) == args.category)].reset_index(drop=True)
        calibration_rows = sample_calibration_rows(manifest, args.normal_shot, args.defect_shot, seed=args.seed)

        model = FastPatchCoreAOI(config).load_memory(default_memory_path(config, args.dataset, args.category))
        cache = extract_cache(model, calibration_rows)
        direction = None
        defect_payload = None
        if args.use_defect_direction and int(args.defect_shot) > 0:
            initial_alpha = 0.5 if args.alpha == "auto" else float(args.alpha)
            direction, defect_meta = build_direction_from_cache(cache, args.defect_top_ratio, args.seed, initial_alpha)
            base_df, _, _, _ = score_cache(cache, direction=None, seed=args.seed)
            direction.alpha = 1.0
            defect_df, _, _, _ = score_cache(cache, direction=direction, seed=args.seed)
            if args.alpha == "auto":
                best_alpha, alpha_f1 = search_alpha(
                    base_df["base_score"].tolist(),
                    defect_df["defect_score"].tolist(),
                    base_df["label"].astype(int).tolist(),
                )
                direction.alpha = best_alpha
            else:
                alpha_f1 = None
                direction.alpha = float(args.alpha)
            out_dir = calibration_dir(config, args.dataset, args.category)
            direction_path = out_dir / "defect_direction.pt"
            direction.save(direction_path)
            defect_meta["alpha"] = float(direction.alpha)
            defect_payload = {
                "dataset": args.dataset,
                "category": args.category,
                "normal_shot": int(args.normal_shot),
                "defect_shot": int(args.defect_shot),
                "seed": int(args.seed),
                "alpha": float(direction.alpha),
                "alpha_search_f1": alpha_f1,
                "defect_direction_path": str(direction_path),
                **defect_meta,
            }
            save_defect_calibration_json(out_dir / "defect_calibration.json", defect_payload)

        score_df, pixel_y, pixel_s, normal_pixel_scores = score_cache(cache, direction=direction, seed=args.seed)
        image_threshold = choose_image_threshold(
            score_df["label"].astype(int).tolist(),
            score_df["score"].tolist(),
            strategy=args.threshold_strategy,
            percentile=args.threshold_percentile,
            mean_std_k=args.mean_std_k,
            target_recall=args.target_recall,
        )
        pixel_threshold = choose_pixel_threshold(
            pixel_y,
            pixel_s,
            fallback_scores=normal_pixel_scores or score_df["score"].tolist(),
            strategy=args.threshold_strategy,
            percentile=args.threshold_percentile,
        )

        out_dir = calibration_dir(config, args.dataset, args.category)
        payload = {
            "dataset": args.dataset,
            "category": args.category,
            "normal_shot_requested": int(args.normal_shot),
            "defect_shot_requested": int(args.defect_shot),
            "normal_shot_used": int((calibration_rows["label"].astype(int) == 0).sum()),
            "defect_shot_used": int((calibration_rows["label"].astype(int) == 1).sum()),
            "seed": int(args.seed),
            "threshold_strategy": args.threshold_strategy,
            "image_threshold": float(image_threshold["threshold"]),
            "pixel_threshold": float(pixel_threshold["threshold"]),
            "image_threshold_detail": image_threshold,
            "pixel_threshold_detail": pixel_threshold,
            "use_defect_direction": bool(direction is not None),
            "defect_alpha": float(direction.alpha) if direction is not None else 0.0,
            "defect_calibration": defect_payload,
        }
        calibration_path = save_calibration(out_dir / "calibration.json", payload)
        score_df.to_csv(out_dir / "calibration_scores.csv", index=False, encoding="utf-8-sig")

        report_path = Path(config["output_root"]) / "results" / "threshold_report.csv"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "dataset": args.dataset,
            "category": args.category,
            "normal_shot": payload["normal_shot_used"],
            "defect_shot": payload["defect_shot_used"],
            "strategy": args.threshold_strategy,
            "image_threshold": payload["image_threshold"],
            "pixel_threshold": payload["pixel_threshold"],
            "use_defect_direction": payload["use_defect_direction"],
            "defect_alpha": payload["defect_alpha"],
            "calibration_path": str(calibration_path),
        }
        if report_path.exists():
            report = pd.read_csv(report_path)
            report = pd.concat([report, pd.DataFrame([row])], ignore_index=True)
        else:
            report = pd.DataFrame([row])
        report.to_csv(report_path, index=False, encoding="utf-8-sig")
        print(json.dumps(row, ensure_ascii=False, indent=2))
    except Exception as exc:
        append_error_report(config, "calibrate failed", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
