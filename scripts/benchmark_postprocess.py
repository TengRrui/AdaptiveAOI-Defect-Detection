"""Benchmark visual postprocessing and output persistence without changing scores."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime import peak_memory_mb, reset_peak_memory
from methods.patchcore.infer import PatchCoreInferencer
from tools.path_utils import load_config


def prepare_2500_input(root: Path, source: Path) -> Path:
    existing = PROJECT_ROOT / "outputs" / "optimization" / "async_io" / "benchmark_inputs" / "mvtec_bottle_resize_2500.png"
    if existing.exists():
        return existing
    if not source.is_file():
        raise FileNotFoundError(f"Benchmark source image not found: {source}")
    target = root / "benchmark_inputs" / "mvtec_bottle_resize_2500.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.open(source).convert("RGB").resize((2500, 2500), Image.Resampling.BILINEAR).save(target)
    return target


def file_pixel_stats(left: Path, right: Path) -> dict:
    a = np.asarray(Image.open(left))
    b = np.asarray(Image.open(right))
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {"shape_equal": bool(a.shape == b.shape), "max_abs_diff": int(diff.max()), "mean_abs_diff": float(diff.mean()), "different_values": int(np.count_nonzero(diff))}


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.copy()
    for column in view.columns:
        if pd.api.types.is_numeric_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.3f}")
        else:
            view[column] = view[column].fillna("").astype(str)
    header = [str(column).replace("|", "\\|") for column in view.columns]
    rows = [[str(value).replace("|", "\\|") for value in row] for row in view.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |", *["| " + " | ".join(row) + " |" for row in rows]])


def run_case(base_config: dict, memory_path: Path, image_path: Path, output_root: Path, name: str, options: dict, warmup: int, runs: int) -> tuple[list[dict], list[dict]]:
    config = dict(base_config)
    config.update(options)
    config["output_root"] = str(output_root / "artifacts" / name)
    inferencer = PatchCoreInferencer(config, category="bottle", memory_path=memory_path)
    for _ in range(warmup):
        inferencer.infer_image(image_path, save_outputs=False, generate_visuals=False)
    reset_peak_memory(inferencer.model.device)
    results = []
    started = time.perf_counter()
    for _ in range(runs):
        results.append(inferencer.infer_image(image_path, save_outputs=True))
    inferencer.finalize_save_results(results, close=True)
    wall_per_image_ms = (time.perf_counter() - started) * 1000.0 / max(1, runs)
    peak = peak_memory_mb(inferencer.model.device)
    rows, profiles = [], []
    for index, result in enumerate(results):
        row = {key: value for key, value in result.items() if not key.startswith("_")}
        row.update({"case": name, "run": index, "peak_gpu_memory_mb": peak, "case_wall_per_image_ms": wall_per_image_ms})
        rows.append(row)
        for profile in result.get("_postprocess_array_profile", []):
            profiles.append({"case": name, "run": index, "peak_gpu_memory_mb": peak, **profile})
    return rows, profiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AdaptiveAOI large-image postprocess/save modes.")
    parser.add_argument("--config", default="configs/mvtec.yaml")
    parser.add_argument("--source_image", default=None, help="Optional source image; defaults to MVTec bottle under configured data_root.")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=6)
    args = parser.parse_args()

    base_config = load_config(args.config)
    output_root = Path(base_config["output_root"]) / "optimization" / "postprocess"
    output_root.mkdir(parents=True, exist_ok=True)
    source_image = Path(args.source_image).expanduser().resolve() if args.source_image else Path(base_config["data_root"]) / "mvtec" / "bottle" / "test" / "good" / "000.png"
    image_path = prepare_2500_input(output_root, source_image)
    memory_path = Path(base_config["output_root"]) / "checkpoints" / "mvtec" / "bottle" / "memory_bank.pt"
    base_config.update({
        "dataset": "mvtec", "category": "bottle", "device": "cuda" if torch.cuda.is_available() else "cpu",
        "memory_backend": "faiss_gpu_flat", "memory_backend_fallback": "torch_gpu_flat",
        "no_save_visuals": False, "return_mask_array": False, "retain_postprocess_arrays": False,
    })
    cases = {
        "no_save": {"save_heatmap": False, "save_mask": False, "save_overlay": False, "no_save_visuals": True, "async_save_outputs": False},
        "sync_mask": {"save_heatmap": False, "save_mask": True, "save_overlay": False, "async_save_outputs": False},
        "sync_heatmap_mask": {"save_heatmap": True, "save_mask": True, "save_overlay": False, "async_save_outputs": False},
        "sync_all": {"save_heatmap": True, "save_mask": True, "save_overlay": True, "async_save_outputs": False},
        "async_all": {"save_heatmap": True, "save_mask": True, "save_overlay": True, "async_save_outputs": True, "save_queue_size": 2},
        "async_heatmap_mask": {"save_heatmap": True, "save_mask": True, "save_overlay": False, "async_save_outputs": True, "save_queue_size": 2},
    }
    benchmark_rows, profile_rows, case_results = [], [], {}
    for name, options in cases.items():
        rows, profiles = run_case(base_config, memory_path, image_path, output_root, name, options, args.warmup, args.runs)
        benchmark_rows.extend(rows)
        profile_rows.extend(profiles)
        case_results[name] = rows
    benchmark = pd.DataFrame(benchmark_rows)
    benchmark.to_csv(output_root / "benchmark.csv", index=False, encoding="utf-8-sig")
    memory_profile = pd.DataFrame(profile_rows)
    memory_profile.to_csv(output_root / "memory_profile.csv", index=False, encoding="utf-8-sig")

    reference = case_results["no_save"]
    checks = []
    for name, rows in case_results.items():
        prediction_equal = all(a["pred_label"] == b["pred_label"] for a, b in zip(reference, rows))
        score_difference = max(abs(float(a["score"]) - float(b["score"])) for a, b in zip(reference, rows))
        expected_paths = {path for row in rows for key, path in [("heatmap_path", row.get("heatmap_path", "")), ("mask_path", row.get("mask_path", "")), ("overlay_path", row.get("overlay_path", ""))] if path}
        existing_paths = {path for path in expected_paths if Path(path).exists()}
        checks.append({"case": name, "pred_label_identical": prediction_equal, "max_image_score_abs_diff": score_difference, "unique_result_paths": len(expected_paths), "existing_result_files": len(existing_paths), "paths_valid": len(expected_paths) == len(existing_paths)})
    pixel_checks = []
    for key in ["heatmap_path", "mask_path", "overlay_path"]:
        left = Path(case_results["sync_all"][-1][key])
        right = Path(case_results["async_all"][-1][key])
        pixel_checks.append({"artifact": key, **file_pixel_stats(left, right)})
    save_failures = not all(item["paths_valid"] for item in checks)
    regression = [
        "# Visual Output Regression", "",
        "- Reference: `no_save`; all score/prediction checks use exactly the same 2500x2500 input and memory bank.",
        "- `sync_all` and `async_all` files are compared after the async save queue flushes.", "",
        "## Prediction And File Checks", "", markdown_table(pd.DataFrame(checks)), "",
        "## Sync/Async Pixel Difference", "", markdown_table(pd.DataFrame(pixel_checks)), "",
        f"- Overall result: **{'FAIL' if save_failures or not all(item['pred_label_identical'] and item['max_image_score_abs_diff'] <= 1e-6 for item in checks) else 'PASS'}**", "",
    ]
    (output_root / "visual_regression.md").write_text("\n".join(regression), encoding="utf-8")

    timing_columns = ["heatmap_resize_ms", "mask_ms", "overlay_ms", "encode_ms", "disk_write_ms", "save_queue_wait_ms", "strict_total_ms", "no_save_total_ms", "online_total_ms", "peak_gpu_memory_mb", "case_wall_per_image_ms"]
    summary = benchmark.groupby("case", as_index=False)[timing_columns].mean()
    per_run_memory = memory_profile.groupby(["case", "run"], as_index=False).agg(estimated_postprocess_arrays_mb=("megabytes", "sum"), array_records=("array_name", "count"))
    array_summary = per_run_memory.groupby("case", as_index=False).agg(estimated_postprocess_arrays_mb=("estimated_postprocess_arrays_mb", "mean"), array_records=("array_records", "mean"))
    summary = summary.merge(array_summary, on="case", how="left").fillna({"estimated_postprocess_arrays_mb": 0.0, "array_records": 0})
    inventory = pd.DataFrame([
        {"array": "score_heatmap", "shape_at_2500": "2500x2500", "dtype": "float32", "megabytes": 2500 * 2500 * 4 / 1024 ** 2, "created_at": "_infer_full_image/_infer_tiled", "reuse": "score, mask, visualization"},
        {"array": "binary_mask", "shape_at_2500": "2500x2500", "dtype": "uint8", "megabytes": 2500 * 2500 / 1024 ** 2, "created_at": "infer_prepared only when requested", "reuse": "mask PNG"},
        {"array": "heatmap_uint8", "shape_at_2500": "2500x2500", "dtype": "uint8", "megabytes": 2500 * 2500 / 1024 ** 2, "created_at": "infer_prepared only heatmap/overlay", "reuse": "heatmap PNG and overlay color lookup"},
        {"array": "overlay_rgb", "shape_at_2500": "2500x2500x3", "dtype": "uint8", "megabytes": 2500 * 2500 * 3 / 1024 ** 2, "created_at": "infer_prepared only overlay", "reuse": "overlay PNG"},
        {"array": "overlay_colormap_temp", "shape_at_2500": "2500x2500x3", "dtype": "uint8", "megabytes": 2500 * 2500 * 3 / 1024 ** 2, "created_at": "make_overlay", "reuse": "released after alpha blend"},
    ])
    report = [
        "# Large Image Postprocess And Save Benchmark", "",
        "## Array Inventory", "", markdown_table(inventory), "",
        "## Design", "",
        "- Score heatmap stays float32. Mask, grayscale heatmap PNG source, and overlay are uint8.",
        "- Mask is created only for mask output or explicit return_mask_array. The uint8 heatmap is created only for heatmap/overlay output and is shared by both.",
        "- The decoded RGB image is reused for overlay. JET RGB lookup avoids per-image BGR/RGB conversion.",
        "- Async saving uses one bounded worker. Paths are populated only after the queue flush confirms successful atomic writes; failed writes log and raise.", "",
        "## Timing And Memory", "", markdown_table(summary), "",
        "## Consistency", "", markdown_table(pd.DataFrame(checks)), "", markdown_table(pd.DataFrame(pixel_checks)), "",
        "- `peak_gpu_memory_mb` is CUDA allocation peak; CPU postprocess memory is represented by the explicit ndarray profile in memory_profile.csv.", "",
    ]
    (output_root / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"benchmark": str(output_root / "benchmark.csv"), "visual_regression": str(output_root / "visual_regression.md"), "save_failures": save_failures}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
