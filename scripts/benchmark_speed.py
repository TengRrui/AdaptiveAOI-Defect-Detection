from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime import peak_memory_mb, reset_peak_memory
from methods.patchcore.build_memory import build_memory
from methods.patchcore.infer import PatchCoreInferencer, default_memory_path
from tools.path_utils import append_error_report, load_config


def pick_source_image(config: dict, dataset: str, category: str) -> Path:
    manifest_path = Path(config["output_root"]) / "results" / f"manifest_{dataset}.csv"
    df = pd.read_csv(manifest_path)
    df = df[(df["dataset"].astype(str).str.lower() == dataset) & (df["category"].astype(str) == category) & (df["split"].astype(str).str.lower() == "test")]
    if df.empty:
        raise RuntimeError(f"No test image found for {dataset}/{category}.")
    return Path(df.iloc[0]["image_path"])


def prepare_benchmark_image(config: dict, source: Path, image_size: int, dataset: str, category: str) -> Path:
    out_dir = Path(config["output_root"]) / "speed_reports" / "benchmark_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_{category}_{image_size}.png"
    if not out_path.exists():
        image = Image.open(source).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        image.save(out_path)
    return out_path


def summarize_times(df: pd.DataFrame, peak_memory: float) -> dict:
    summary = {"peak_memory": float(peak_memory)}
    for col in [c for c in df.columns if c.endswith("_ms")]:
        values = df[col].astype(float).to_numpy()
        summary[col] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark AdaptiveAOI speed.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True, choices=["mvtec", "visa", "dagm"])
    parser.add_argument("--category", required=True)
    parser.add_argument("--image_size", type=int, required=True)
    parser.add_argument("--model_image_size", type=int, default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--use_tiling", action="store_true")
    parser.add_argument("--tile_size", type=int, default=None)
    parser.add_argument("--tile_overlap", type=int, default=None)
    parser.add_argument("--tile_batch_size", type=int, default=None)
    parser.add_argument("--tile_fusion", default=None, choices=["max", "average"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--no_save_visuals", action="store_true")
    parser.add_argument("--timing_mode", default="strict_total", choices=["strict_total", "online_total", "detection_only", "no_save_total"])
    parser.add_argument("--speed_mode", action="store_true")
    parser.add_argument("--extreme_speed_mode", action="store_true")
    parser.add_argument("--coarse_only", action="store_true")
    parser.add_argument("--coarse_to_fine", action="store_true")
    parser.add_argument("--coarse_image_size", type=int, default=None)
    parser.add_argument("--coarse_threshold", type=float, default=None)
    parser.add_argument("--roi_topk", type=int, default=None)
    parser.add_argument("--roi_min_area", type=int, default=None)
    parser.add_argument("--roi_expand_ratio", type=float, default=None)
    parser.add_argument("--roi_merge_iou", type=float, default=None)
    parser.add_argument("--fine_tile_size", type=int, default=None)
    parser.add_argument("--fine_image_size", type=int, default=None)
    parser.add_argument("--max_fine_rois", type=int, default=None)
    parser.add_argument("--coarse_fine_fusion", default=None, choices=["max", "average", "replace"])
    args = parser.parse_args()

    config = load_config(args.config)
    config["dataset"] = args.dataset
    config["category"] = args.category
    config["device"] = args.device
    config["use_fp16"] = bool(args.precision == "fp16" and args.device == "cuda")
    config["timing_mode"] = args.timing_mode
    config["speed_mode"] = bool(args.speed_mode or args.extreme_speed_mode)
    config["extreme_speed_mode"] = bool(args.extreme_speed_mode)
    config["coarse_only"] = bool(args.coarse_only)
    config["use_coarse_to_fine"] = bool(args.coarse_to_fine)
    config["use_tiling"] = bool(args.use_tiling and not args.coarse_only and not args.coarse_to_fine)
    config["tile_size"] = int(args.tile_size if args.tile_size is not None else config.get("tile_size", 768))
    config["tile_overlap"] = int(args.tile_overlap if args.tile_overlap is not None else config.get("tile_overlap", 128))
    config["tile_batch_size"] = int(args.tile_batch_size if args.tile_batch_size is not None else config.get("tile_batch_size", 4))
    config["tile_fusion"] = str(args.tile_fusion if args.tile_fusion is not None else config.get("tile_fusion", "max"))
    for name in (
        "coarse_image_size",
        "roi_topk",
        "roi_min_area",
        "fine_tile_size",
        "fine_image_size",
        "max_fine_rois",
    ):
        value = getattr(args, name)
        if value is not None:
            config[name] = int(value)
    for name in ("coarse_threshold", "roi_expand_ratio", "roi_merge_iou"):
        value = getattr(args, name)
        if value is not None:
            config[name] = float(value)
    if args.coarse_fine_fusion is not None:
        config["coarse_fine_fusion"] = str(args.coarse_fine_fusion)
    config["no_save_visuals"] = bool(args.no_save_visuals or not args.save_outputs)
    config["save_heatmap"] = bool(args.save_outputs and not args.no_save_visuals)
    config["save_mask"] = bool(args.save_outputs and not args.no_save_visuals)
    config["save_overlay"] = bool(args.save_outputs and not args.no_save_visuals)
    if args.model_image_size is not None:
        config["image_size"] = int(args.model_image_size)
    elif args.speed_mode or args.extreme_speed_mode or args.coarse_only or args.coarse_to_fine:
        config["image_size"] = int(config["image_size"])
    elif not args.use_tiling:
        config["image_size"] = int(args.image_size)

    try:
        if not default_memory_path(config, args.dataset, args.category).exists():
            build_memory(config, category=args.category)
        source = pick_source_image(config, args.dataset, args.category)
        image_path = prepare_benchmark_image(config, source, args.image_size, args.dataset, args.category)
        inferencer = PatchCoreInferencer(config, category=args.category)
        for _ in range(int(args.warmup)):
            inferencer.infer_image(image_path, save_outputs=args.save_outputs)

        reset_peak_memory(inferencer.model.device)
        rows = []
        for _ in range(int(args.runs)):
            result = inferencer.infer_image(image_path, save_outputs=args.save_outputs)
            rows.append({k: v for k, v in result.items() if not k.startswith("_")})
        peak = peak_memory_mb(inferencer.model.device)
        df = pd.DataFrame(rows)
        out_dir = Path(config["output_root"]) / "speed_reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"speed_{args.image_size}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        summary = summarize_times(df, peak)
        summary.update(
            {
                "dataset": args.dataset,
                "category": args.category,
                "image_size": int(args.image_size),
                "model_image_size": int(config["image_size"]),
                "device": args.device,
                "precision": args.precision,
                "use_tiling": bool(args.use_tiling),
                "timing_mode": args.timing_mode,
                "speed_mode": bool(args.speed_mode),
                "extreme_speed_mode": bool(args.extreme_speed_mode),
                "coarse_only": bool(args.coarse_only),
                "coarse_to_fine": bool(args.coarse_to_fine),
                "coarse_image_size": int(config.get("coarse_image_size", 0)),
                "fine_image_size": int(config.get("fine_image_size", 0)),
                "fine_tile_size": int(config.get("fine_tile_size", 0)),
                "max_fine_rois": int(config.get("max_fine_rois", 0)),
                "no_save_visuals": bool(args.no_save_visuals or not args.save_outputs),
                "tile_size": int(config["tile_size"]),
                "tile_overlap": int(config["tile_overlap"]),
                "tile_batch_size": int(config["tile_batch_size"]),
                "runs": int(args.runs),
                "warmup": int(args.warmup),
                "csv_path": str(csv_path),
            }
        )
        summary_path = out_dir / "speed_summary.json"
        if summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                payload = [payload]
        else:
            payload = []
        payload.append(summary)
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as exc:
        append_error_report(config, "benchmark_speed failed", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
