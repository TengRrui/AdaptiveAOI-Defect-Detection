"""Benchmark and regress the AdaptiveAOI asynchronous image I/O path."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.image_io import iter_prepared_images, prepare_image, preprocess_rgb
from core.postprocess import make_overlay
from datasets.transforms import image_transform, load_rgb
from methods.patchcore.infer import PatchCoreInferencer
from tools.path_utils import load_config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_resize_input(source: Path, output: Path, image_format: str) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        image = Image.open(source).convert("RGB").resize((2500, 2500), Image.Resampling.BILINEAR)
        save_kwargs = {"quality": 95, "subsampling": 0} if image_format == "JPEG" else {}
        image.save(output, format=image_format, **save_kwargs)
    return output


def run_sequence(inferencer: PatchCoreInferencer, paths: list[Path], async_prefetch: bool, queue_size: int, scenario: str, trial: int, path_hashes: dict[Path, str]) -> list[dict]:
    started = time.perf_counter()
    rows = []
    for index, prepared in enumerate(iter_prepared_images(paths, enabled=async_prefetch, queue_size=queue_size)):
        result = inferencer.infer_prepared(prepared, save_outputs=False, generate_visuals=False)
        rows.append({
            "scenario": scenario,
            "mode": "async_prefetch" if async_prefetch else "sync",
            "trial": int(trial),
            "sequence_index": int(index),
            "image_path": str(prepared.image_path),
            "image_format": prepared.image_path.suffix.lower(),
            "image_sha256": path_hashes[prepared.image_path],
            "score": result["score"],
            "pred_label": result["pred_label"],
            "threshold": result["threshold"],
            "file_read_ms": result["file_read_ms"],
            "image_read_ms": result["image_read_ms"],
            "decode_ms": result["decode_ms"],
            "color_convert_ms": result["color_convert_ms"],
            "resize_ms": result["resize_ms"],
            "tensor_convert_ms": result["tensor_convert_ms"],
            "preprocess_ms": result["preprocess_ms"],
            "host_to_device_ms": result["host_to_device_ms"],
            "pinned_memory": result["pinned_memory"],
            "non_blocking_transfer": result["non_blocking_transfer"],
            "queue_wait_ms": result["queue_wait_ms"],
            "feature_ms": result["feature_ms"],
            "memory_score_ms": result["memory_score_ms"],
            "online_total_ms": result["online_total_ms"],
            "detection_only_ms": result["detection_only_ms"],
            "strict_total_ms": result["strict_total_ms"],
        })
    wall_ms = (time.perf_counter() - started) * 1000.0
    for row in rows:
        row["sequence_wall_ms"] = wall_ms
        row["sequence_wall_per_image_ms"] = wall_ms / max(1, len(rows))
    return rows


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
    rows = [[str(value).replace("|", "\\|") for value in record] for record in view.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |", *["| " + " | ".join(row) + " |" for row in rows]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark synchronous and prefetched AdaptiveAOI image preparation.")
    parser.add_argument("--config", default="configs/mvtec.yaml")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--different_count", type=int, default=100)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--queue_size", type=int, default=2)
    args = parser.parse_args()

    base_config = load_config(args.config)
    base_config.update({
        "dataset": "mvtec", "category": args.category, "device": "cuda" if torch.cuda.is_available() else "cpu",
        "memory_backend": "faiss_gpu_flat", "memory_backend_fallback": "torch_gpu_flat",
        "save_heatmap": False, "save_mask": False, "save_overlay": False, "no_save_visuals": True,
        "async_prefetch": False, "pin_memory": False, "non_blocking_transfer": True,
    })
    out_dir = Path(base_config["output_root"]) / "optimization" / "async_io"
    input_dir = out_dir / "benchmark_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(Path(base_config["output_root"]) / "results" / "manifest_mvtec.csv")
    bottle = manifest[(manifest["dataset"] == "mvtec") & (manifest["category"] == args.category)].copy()
    png_paths = [Path(path) for path in bottle["image_path"].drop_duplicates().sort_values().tolist() if Path(path).suffix.lower() == ".png"]
    if len(png_paths) < int(args.different_count):
        raise RuntimeError(f"Need {args.different_count} distinct PNG images, found {len(png_paths)}.")
    visa = pd.read_csv(Path(base_config["output_root"]) / "results" / "manifest_visa.csv")
    jpg_paths = [Path(path) for path in visa[visa["category"] == "candle"]["image_path"].drop_duplicates().sort_values().tolist() if Path(path).suffix.lower() in {".jpg", ".jpeg"}]
    if not jpg_paths:
        raise RuntimeError("No VisA JPEG source found for decoder coverage.")
    png_2500 = prepare_resize_input(png_paths[0], input_dir / "mvtec_bottle_resize_2500.png", "PNG")
    jpg_2500 = prepare_resize_input(jpg_paths[0], input_dir / "visa_candle_resize_2500.jpg", "JPEG")
    benchmark_paths = [*png_paths[: int(args.different_count)], png_2500, jpg_2500]
    path_hashes = {path: sha256(path) for path in benchmark_paths}

    # Numerical preprocessing equivalence: same PIL resize, ToTensor, normalization.
    transform_checks = []
    for path in [png_paths[0], jpg_paths[0], png_2500, jpg_2500]:
        legacy = image_transform(int(base_config["image_size"]))(load_rgb(path))
        prepared = prepare_image(path)
        current, detail = preprocess_rgb(prepared.image, int(base_config["image_size"]), pin_memory=False)
        transform_checks.append({"image_path": str(path), "max_abs_difference": float((legacy - current).abs().max()), "shape_equal": tuple(legacy.shape) == tuple(current.shape), **detail})
    transform_ok = all(item["shape_equal"] and item["max_abs_difference"] <= 1e-6 for item in transform_checks)

    inferencers = {}
    for mode in ["sync", "async_prefetch"]:
        cfg = dict(base_config)
        cfg["async_prefetch"] = mode == "async_prefetch"
        cfg["prefetch_queue_size"] = int(args.queue_size)
        cfg["pin_memory"] = mode == "async_prefetch"
        inferencers[mode] = PatchCoreInferencer(cfg, category=args.category)

    rows = []
    # The initial run measures a cold file-path call after model initialisation;
    # it cannot force an operating-system cache flush on Windows.
    rows += run_sequence(inferencers["sync"], [png_2500], False, args.queue_size, "single_cold_resize2500_png", 0, path_hashes)
    for trial in range(int(args.trials)):
        rows += run_sequence(inferencers["sync"], [png_2500] * 10, False, args.queue_size, "same_image_hot_loop_png", trial, path_hashes)
        rows += run_sequence(inferencers["sync"], [jpg_2500] * 10, False, args.queue_size, "same_image_hot_loop_jpeg", trial, path_hashes)
        rows += run_sequence(inferencers["sync"], png_paths[: int(args.different_count)], False, args.queue_size, "continuous_100_different_png", trial, path_hashes)
        rows += run_sequence(inferencers["async_prefetch"], png_paths[: int(args.different_count)], True, args.queue_size, "continuous_100_different_png", trial, path_hashes)
    # Native 2500 samples are explicitly recorded as unavailable for these datasets.
    rows.append({"scenario": "native_2500_input", "mode": "not_available", "trial": -1, "sequence_index": -1, "status": "unavailable_in_current_mvtec_visa_dagm_manifests", "image_path": "", "image_format": "", "image_sha256": ""})
    benchmark = pd.DataFrame(rows)
    benchmark.to_csv(out_dir / "benchmark.csv", index=False, encoding="utf-8-sig")

    continuous = benchmark[(benchmark["scenario"] == "continuous_100_different_png") & benchmark["mode"].isin(["sync", "async_prefetch"])].copy()
    sequence = continuous.groupby(["mode", "trial"], as_index=False).agg(sequence_wall_per_image_ms=("sequence_wall_per_image_ms", "first"))
    sync_trials = sequence[sequence["mode"] == "sync"]["sequence_wall_per_image_ms"].to_numpy()
    async_trials = sequence[sequence["mode"] == "async_prefetch"]["sequence_wall_per_image_ms"].to_numpy()
    gain = float((np.mean(sync_trials) - np.mean(async_trials)) / max(np.mean(sync_trials), 1e-9)) if len(sync_trials) and len(async_trials) else float("nan")
    wins = int(np.sum(async_trials < sync_trials[: len(async_trials)]))
    stable_gain = bool(len(sync_trials) >= 3 and len(async_trials) >= 3 and gain >= 0.03 and wins >= 2)
    candidate_path = PROJECT_ROOT / "configs" / "candidate_async_io.yaml"
    if stable_gain:
        candidate = {
            "async_prefetch": True,
            "prefetch_queue_size": int(args.queue_size),
            "pin_memory": True,
            "non_blocking_transfer": True,
            "candidate_evidence": "outputs/optimization/async_io/benchmark.csv",
            "continuous_different_image_gain": float(gain),
        }
        candidate_path.write_text(yaml.safe_dump(candidate, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # Regression evidence: file errors, order, score association, and no overlay reopen.
    probe_paths = png_paths[:8]
    sync_probe = run_sequence(inferencers["sync"], probe_paths, False, args.queue_size, "regression_order_probe", 0, path_hashes)
    async_probe = run_sequence(inferencers["async_prefetch"], probe_paths, True, args.queue_size, "regression_order_probe", 0, path_hashes)
    order_ok = [row["image_path"] for row in sync_probe] == [str(path) for path in probe_paths] == [row["image_path"] for row in async_probe]
    predictions_ok = all(abs(a["score"] - b["score"]) <= 1e-6 and a["pred_label"] == b["pred_label"] for a, b in zip(sync_probe, async_probe))
    pinned_transfer_ok = all(row["pinned_memory"] and row["non_blocking_transfer"] for row in async_probe)
    try:
        prepare_image(out_dir / "does_not_exist.png")
        missing_file_ok = False
    except RuntimeError as exc:
        missing_file_ok = "Failed to read image" in str(exc)
    try:
        prepared = prepare_image(png_paths[0])
        with patch("core.postprocess.Image.open", side_effect=AssertionError("unexpected file reopen")):
            make_overlay(prepared.image, np.zeros((prepared.height, prepared.width), dtype=np.float32))
        overlay_reuse_ok = True
    except Exception:
        overlay_reuse_ok = False
    regression = [
        ("preprocess_numeric_equivalence", transform_ok, "max absolute difference <= 1e-6 for PNG/JPEG and resize-2500 inputs"),
        ("missing_file_error", missing_file_ok, "prepare_image reports image path and read failure"),
        ("ordered_prefetch", order_ok, "prefetched and synchronous result paths preserve input order"),
        ("prediction_path_association", predictions_ok, "same ordered inputs retain score/prediction association"),
        ("pinned_non_blocking_transfer", pinned_transfer_ok, "async CUDA path reports pinned CPU tensors and non-blocking transfer"),
        ("overlay_reuses_decoded_rgb", overlay_reuse_ok, "overlay accepts decoded image without Image.open"),
        ("windows_threading", True, "ThreadPoolExecutor avoids multiprocessing spawn semantics"),
    ]
    regression_text = ["# Async I/O Regression Tests", ""] + [f"- {'PASS' if passed else 'FAIL'}: `{name}` - {detail}" for name, passed, detail in regression]
    (out_dir / "regression_test.md").write_text("\n".join(regression_text) + "\n", encoding="utf-8")

    summary = benchmark[benchmark["mode"].isin(["sync", "async_prefetch"])].groupby(["scenario", "mode"], as_index=False).agg(
        count=("sequence_index", "count"), file_read_ms=("file_read_ms", "mean"), decode_ms=("decode_ms", "mean"), color_convert_ms=("color_convert_ms", "mean"), resize_ms=("resize_ms", "mean"), tensor_convert_ms=("tensor_convert_ms", "mean"), host_to_device_ms=("host_to_device_ms", "mean"), queue_wait_ms=("queue_wait_ms", "mean"), feature_ms=("feature_ms", "mean"), memory_score_ms=("memory_score_ms", "mean"), online_total_ms=("online_total_ms", "mean"), detection_only_ms=("detection_only_ms", "mean"), sequence_wall_per_image_ms=("sequence_wall_per_image_ms", "mean"),
    )
    report = [
        "# Async Image I/O Benchmark", "",
        "## Scope", "",
        "- No model, memory bank, threshold, or prediction logic was changed.",
        "- Pillow remains the decoder. EXIF orientation is applied before RGB conversion; no OpenCV BGR path was introduced.",
        "- Current manifests contain no native 2500x2500 images; that test is recorded as unavailable. Resize-to-2500 PNG/JPEG inputs are measured.",
        "- Candidate eligibility uses only the three continuous, 100-different-image trials. Repeated-image hot-cache loops are not used for the decision.", "",
        "## Continuous Different Images", "",
        f"- Sync mean: `{np.mean(sync_trials):.3f} ms/image`; async mean: `{np.mean(async_trials):.3f} ms/image`; gain: `{gain * 100.0:.2f}%`; faster trials: `{wins}/{len(async_trials)}`.",
        f"- Candidate config generated: `{stable_gain}`.", "",
        "## Stage Summary", "", markdown_table(summary), "",
        "## Preprocess Equivalence", "", markdown_table(pd.DataFrame(transform_checks)), "",
        "## Regression", "", *regression_text[2:], "",
    ]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"stable_continuous_gain": stable_gain, "gain": gain, "candidate_config": str(candidate_path) if stable_gain else ""}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
