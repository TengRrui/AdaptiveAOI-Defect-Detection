from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.memory_bank import MemoryBank
from core.runtime import peak_memory_mb, reset_peak_memory
from datasets.transforms import load_rgb
from methods.patchcore.infer import PatchCoreInferencer
from tools.metrics import binary_metrics, safe_auroc
from tools.path_utils import load_config, load_yaml


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "optimization" / "gpu_ann"
TIMING_MODES = ("online_total", "detection_only")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stats(values: pd.Series) -> dict[str, float]:
    array = values.astype(float).to_numpy()
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def effective_size(inferencer: PatchCoreInferencer) -> int:
    return int(inferencer.model.metadata.get("scoring_memory_size", inferencer.model.memory_bank.features.shape[0]))


def find_case(base_config: dict, dataset: str, category: str) -> dict:
    for case in base_config["benchmark_cases"]:
        if str(case["dataset"]).lower() == dataset and str(case["category"]) == category:
            return dict(case)
    raise KeyError(f"No baseline case for {dataset}/{category}.")


def find_benchmark_input(base_config: dict, dataset: str, category: str) -> dict:
    manifest = Path(base_config["output_root"]) / "speed_reports" / "final_benchmark_input_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing fixed input manifest: {manifest}")
    df = pd.read_csv(manifest)
    selected = df[(df["dataset"].astype(str).str.lower() == dataset) & (df["category"].astype(str) == category)]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one benchmark input for {dataset}/{category}; got {len(selected)}.")
    return selected.iloc[0].to_dict()


def make_config(base_config: dict, case: dict, backend: str, fallback: str, score_limit: int, timing_mode: str, bench_config: dict) -> dict:
    config = dict(base_config)
    config.update(
        {
            "dataset": str(case["dataset"]).lower(),
            "category": str(case["category"]),
            "memory_backend": backend,
            "memory_backend_fallback": fallback,
            "memory_ivf_nlist": int(bench_config["memory_ivf_nlist"]),
            "memory_ivf_nprobe": int(bench_config["memory_ivf_nprobe"]),
            "torch_gpu_chunk_size": int(bench_config["torch_gpu_chunk_size"]),
            "score_memory_limit": int(score_limit),
            "timing_mode": timing_mode,
            "save_heatmap": False,
            "save_mask": False,
            "save_overlay": False,
            "no_save_visuals": True,
        }
    )
    return config


def make_inferencer(base_config: dict, case: dict, backend: str, fallback: str, score_limit: int, timing_mode: str, bench_config: dict) -> PatchCoreInferencer:
    config = make_config(base_config, case, backend, fallback, score_limit, timing_mode, bench_config)
    inferencer = PatchCoreInferencer(config, category=config["category"], memory_path=case["memory_bank_path"])
    expected_dim = int(inferencer.model.metadata.get("feature_dim", inferencer.model.memory_bank.feature_dim))
    if expected_dim != inferencer.model.memory_bank.feature_dim:
        raise RuntimeError(
            f"Memory feature dimension mismatch: metadata={expected_dim}, bank={inferencer.model.memory_bank.feature_dim}."
        )
    return inferencer


def run_speed(inferencer: PatchCoreInferencer, image_path: str, warmup: int, runs: int, timing_mode: str) -> tuple[pd.DataFrame, float]:
    inferencer.config["timing_mode"] = timing_mode
    for _ in range(warmup):
        inferencer.infer_image(image_path, save_outputs=False, generate_visuals=False)
    reset_peak_memory(inferencer.model.device)
    rows = []
    for run_index in range(runs):
        result = inferencer.infer_image(image_path, save_outputs=False, generate_visuals=False)
        rows.append({
            "run_index": run_index,
            "timing_mode": timing_mode,
            "memory_score_ms": float(result["memory_score_ms"]),
            "device_to_host_ms": float(result["device_to_host_ms"]),
            "online_total_ms": float(result["online_total_ms"]),
            "detection_only_ms": float(result["detection_only_ms"]),
            "total_ms": float(result["total_ms"]),
        })
    return pd.DataFrame(rows), peak_memory_mb(inferencer.model.device)


def sanity_subset(base_config: dict, dataset: str, category: str, samples_per_label: int) -> pd.DataFrame:
    manifest_path = Path(base_config["output_root"]) / "results" / f"manifest_{dataset}.csv"
    df = pd.read_csv(manifest_path)
    subset = df[
        (df["dataset"].astype(str).str.lower() == dataset)
        & (df["category"].astype(str) == category)
        & (df["split"].astype(str).str.lower() == "test")
    ].sort_values(["label", "image_path"])
    selected = pd.concat([subset[subset["label"] == label].head(samples_per_label) for label in (0, 1)], ignore_index=True)
    if selected["label"].nunique() != 2:
        raise RuntimeError("Sanity subset must contain normal and anomaly samples.")
    return selected


def run_accuracy(inferencer: PatchCoreInferencer, subset: pd.DataFrame) -> dict[str, float]:
    labels, scores, predictions = [], [], []
    for item in subset.to_dict("records"):
        result = inferencer.infer_image(item["image_path"], meta=item, save_outputs=False, generate_visuals=False)
        labels.append(int(item["label"]))
        scores.append(float(result["score"]))
        predictions.append(int(result["pred_label"]))
    metrics = binary_metrics(labels, predictions)
    return {
        "I-AUROC": safe_auroc(labels, scores),
        "F1": metrics["f1"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
    }


def consistency_check(cpu: PatchCoreInferencer, gpu: PatchCoreInferencer, source_image: str) -> dict[str, float | str]:
    image = load_rgb(source_image)
    tensor = gpu.transform(image)
    patches, _ = gpu.model.extract(tensor)
    cpu_scores = cpu.model.memory_bank.score(patches[0])
    gpu_scores = gpu.model.memory_bank.score(patches[0])
    if isinstance(cpu_scores, torch.Tensor):
        cpu_scores = cpu_scores.detach().cpu().numpy()
    if isinstance(gpu_scores, torch.Tensor):
        gpu_scores = gpu_scores.detach().cpu().numpy()
    cpu_indices = cpu.model.memory_bank.nearest_indices(patches[0])
    gpu_indices = gpu.model.memory_bank.nearest_indices(patches[0])
    if isinstance(cpu_indices, torch.Tensor):
        cpu_indices = cpu_indices.detach().cpu().numpy()
    if isinstance(gpu_indices, torch.Tensor):
        gpu_indices = gpu_indices.detach().cpu().numpy()
    return {
        "nearest_agreement": float(np.mean(np.asarray(cpu_indices) == np.asarray(gpu_indices))),
        "score_max_abs_diff": float(np.max(np.abs(np.asarray(cpu_scores) - np.asarray(gpu_scores)))),
        "score_mean_abs_diff": float(np.mean(np.abs(np.asarray(cpu_scores) - np.asarray(gpu_scores)))),
        "cpu_backend": cpu.model.memory_bank.backend,
        "gpu_backend_actual": gpu.model.memory_bank.backend,
    }


def write_implementation_report(
    path: Path,
    bench_config: dict,
    rows: pd.DataFrame,
    tradeoff: pd.DataFrame,
    faiss_gpu: tuple[bool, str],
    candidate_message: str,
) -> None:
    actual = rows[["backend_requested", "backend_actual", "backend_fallback_reason"]].drop_duplicates()
    lines = [
        "# GPU kNN Implementation Report",
        "",
        "## Scope",
        "",
        "- Only memory-bank scoring was changed. Backbone, threshold, tiling, postprocess and support-set selection are unchanged.",
        "- Serialized `memory_bank.pt` files remain CPU tensors and are loaded read-only; backend metadata is runtime-only.",
        "",
        "## Backends",
        "",
        "- `faiss_gpu_flat`: exact GPU FAISS when the installed FAISS build exposes GPU resources and CUDA tensor search.",
        "- `faiss_gpu_ivf`: GPU IVF-Flat with recorded `nlist`/`nprobe` when available.",
        "- `faiss_cpu_flat`: exact CPU L2 baseline.",
        "- `sklearn`: CPU fallback.",
        "- `torch_gpu_flat`: exact CUDA L2 fallback used only when GPU FAISS is unavailable or explicitly selected.",
        "",
        "## Environment Decision",
        "",
        f"- FAISS GPU available: `{faiss_gpu[0]}`.",
        f"- FAISS GPU probe: `{faiss_gpu[1]}`.",
        "",
        "## Requested vs Actual Backends",
        "",
        "| Requested | Actual | Fallback reason |",
        "|---|---|---|",
    ]
    for row in actual.to_dict("records"):
        reason = row["backend_fallback_reason"] if pd.notna(row["backend_fallback_reason"]) else ""
        lines.append(f"| {row['backend_requested']} | {row['backend_actual']} | {reason} |")
    full_rows = tradeoff[tradeoff["score_memory_limit"] == 0]
    lines.extend(
        [
            "",
            "## Full-Memory Result",
            "",
            "| Requested | Actual | Memory score mean ms | Detection-only mean ms | I-AUROC | F1 | Index GPU bytes |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in full_rows.to_dict("records"):
        lines.append(
            f"| {row['backend_requested']} | {row['backend_actual']} | {row['memory_score_mean_ms']:.3f} | "
            f"{row['detection_only_mean_ms']:.3f} | {row['I-AUROC']:.4f} | {row['F1']:.4f} | {int(row['memory_index_gpu_bytes'])} |"
        )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            f"- Score memory sizes: `{bench_config['score_memory_sizes']}`; warmup: `{bench_config['warmup']}`; runs: `{bench_config['runs']}`.",
            f"- Timing modes: `{bench_config['timing_modes']}`; fixed seed: `{bench_config['random_seed']}`.",
            "- Accuracy is reported on a deterministic subset only and is never fed back into threshold/calibration selection.",
            f"- Candidate decision: {candidate_message}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_regression_report(path: Path, checks: list[tuple[str, bool, str]]) -> None:
    lines = ["# GPU kNN Regression Tests", "", "| Check | Status | Evidence |", "|---|---|---|"]
    for name, passed, evidence in checks:
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} | {evidence} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_write_candidate_config(base_config: dict, bench_config: dict, tradeoff: pd.DataFrame) -> str:
    tolerance = bench_config["accuracy_tolerance"]
    cpu = tradeoff[tradeoff["backend_actual"] == "faiss_cpu_flat"].set_index("score_memory_size")
    gpu = tradeoff[tradeoff["backend_actual"].isin(["faiss_gpu_flat", "faiss_gpu_ivf"])]
    candidates = []
    for row in gpu.to_dict("records"):
        size = int(row["score_memory_size"])
        if size not in cpu.index:
            continue
        reference = cpu.loc[size]
        faster = float(row["detection_only_mean_ms"]) < float(reference["detection_only_mean_ms"])
        within_tolerance = all(
            float(reference[name]) - float(row[name]) <= float(tolerance[f"{name}_drop"])
            for name in ("I-AUROC", "F1", "Precision", "Recall")
        )
        if faster and within_tolerance:
            candidates.append((float(row["detection_only_mean_ms"]), row))
    if not candidates:
        return "No candidate_gpu_ann.yaml generated: no actual FAISS GPU backend met speed and accuracy gates."
    _, best = min(candidates, key=lambda item: item[0])
    candidate = dict(base_config)
    candidate.update(
        {
            "memory_backend": best["backend_requested"],
            "memory_backend_fallback": bench_config["memory_backend_fallback"],
            "score_memory_limit": int(best["score_memory_limit"]),
            "memory_ivf_nlist": int(bench_config["memory_ivf_nlist"]),
            "memory_ivf_nprobe": int(bench_config["memory_ivf_nprobe"]),
            "torch_gpu_chunk_size": int(bench_config["torch_gpu_chunk_size"]),
        }
    )
    candidate_path = PROJECT_ROOT / "configs" / "candidate_gpu_ann.yaml"
    candidate_path.write_text(yaml.safe_dump(candidate, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return f"Generated {candidate_path} after passing exact CPU comparison gates."


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CPU/GPU memory-bank backends without changing AOI behavior.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "gpu_ann_benchmark.yaml"))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    bench_config = load_yaml(config_path)
    base_config_path = Path(str(bench_config["base_config"]))
    if not base_config_path.is_absolute():
        base_config_path = config_path.parent / base_config_path
    base_config = load_config(base_config_path)
    set_seed(int(bench_config["random_seed"]))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = str(bench_config["dataset"]).lower()
    category = str(bench_config["category"])
    case = find_case(base_config, dataset, category)
    benchmark_input = find_benchmark_input(base_config, dataset, category)
    sanity = sanity_subset(base_config, dataset, category, int(bench_config["sanity_samples_per_label"]))
    faiss_gpu = MemoryBank.faiss_gpu_available()

    benchmark_rows = []
    tradeoff_rows = []
    regression_checks: list[tuple[str, bool, str]] = []
    backends = [
        (str(bench_config["memory_backend_cpu"]), str(bench_config["memory_backend_cpu"])),
        (str(bench_config["memory_backend_gpu"]), str(bench_config["memory_backend_fallback"])),
    ]

    for requested_backend, fallback_backend in backends:
        for score_limit in [int(value) for value in bench_config["score_memory_sizes"]]:
            inferencers = {}
            for timing_mode in bench_config["timing_modes"]:
                inferencer = make_inferencer(
                    base_config, case, requested_backend, fallback_backend, score_limit, timing_mode, bench_config
                )
                inferencers[timing_mode] = inferencer
                samples, peak = run_speed(
                    inferencer,
                    benchmark_input["benchmark_image_path"],
                    int(bench_config["warmup"]),
                    int(bench_config["runs"]),
                    timing_mode,
                )
                row = {
                    "dataset": dataset,
                    "category": category,
                    "backend_requested": requested_backend,
                    "backend_actual": inferencer.model.memory_bank.backend,
                    "backend_fallback_reason": inferencer.model.memory_bank.backend_info.get("memory_backend_fallback_reason", ""),
                    "score_memory_limit": score_limit,
                    "score_memory_size": effective_size(inferencer),
                    "timing_mode": timing_mode,
                    "memory_score_mean_ms": stats(samples["memory_score_ms"])["mean"],
                    "memory_score_p95_ms": stats(samples["memory_score_ms"])["p95"],
                    "device_to_host_mean_ms": stats(samples["device_to_host_ms"])["mean"],
                    "device_to_host_p95_ms": stats(samples["device_to_host_ms"])["p95"],
                    "total_mean_ms": stats(samples["total_ms"])["mean"],
                    "total_p95_ms": stats(samples["total_ms"])["p95"],
                    "count": int(len(samples)),
                    "peak_memory_mb": float(peak),
                    **inferencer.model.memory_bank.runtime_metadata(),
                }
                benchmark_rows.append(row)

            accuracy_inferencer = make_inferencer(
                base_config, case, requested_backend, fallback_backend, score_limit, "online_total", bench_config
            )
            accuracy = run_accuracy(accuracy_inferencer, sanity)
            online = next(row for row in benchmark_rows if row["backend_requested"] == requested_backend and row["score_memory_limit"] == score_limit and row["timing_mode"] == "online_total")
            detection = next(row for row in benchmark_rows if row["backend_requested"] == requested_backend and row["score_memory_limit"] == score_limit and row["timing_mode"] == "detection_only")
            tradeoff_rows.append(
                {
                    "dataset": dataset,
                    "category": category,
                    "backend_requested": requested_backend,
                    "backend_actual": accuracy_inferencer.model.memory_bank.backend,
                    "backend_fallback_reason": accuracy_inferencer.model.memory_bank.backend_info.get("memory_backend_fallback_reason", ""),
                    "score_memory_limit": score_limit,
                    "score_memory_size": effective_size(accuracy_inferencer),
                    "I-AUROC": accuracy["I-AUROC"],
                    "F1": accuracy["F1"],
                    "Precision": accuracy["Precision"],
                    "Recall": accuracy["Recall"],
                    "online_total_mean_ms": online["total_mean_ms"],
                    "online_total_p95_ms": online["total_p95_ms"],
                    "detection_only_mean_ms": detection["total_mean_ms"],
                    "detection_only_p95_ms": detection["total_p95_ms"],
                    "memory_score_mean_ms": detection["memory_score_mean_ms"],
                    "memory_score_p95_ms": detection["memory_score_p95_ms"],
                    "memory_index_build_ms": float(accuracy_inferencer.model.memory_bank.backend_info.get("memory_index_build_ms", 0.0)),
                    "memory_index_gpu_bytes": int(accuracy_inferencer.model.memory_bank.backend_info.get("memory_index_gpu_bytes", 0)),
                    "memory_ivf_nlist": int(accuracy_inferencer.model.memory_bank.backend_info.get("memory_ivf_nlist", 0)),
                    "memory_ivf_nprobe": int(accuracy_inferencer.model.memory_bank.backend_info.get("memory_ivf_nprobe", 0)),
                }
            )

    benchmark_df = pd.DataFrame(benchmark_rows)
    tradeoff_df = pd.DataFrame(tradeoff_rows)
    benchmark_df.to_csv(OUTPUT_DIR / "benchmark.csv", index=False, encoding="utf-8-sig")
    tradeoff_df.to_csv(OUTPUT_DIR / "accuracy_speed_tradeoff.csv", index=False, encoding="utf-8-sig")

    for score_limit in [int(value) for value in bench_config["score_memory_sizes"]]:
        cpu = make_inferencer(base_config, case, bench_config["memory_backend_cpu"], bench_config["memory_backend_cpu"], score_limit, "online_total", bench_config)
        gpu = make_inferencer(base_config, case, bench_config["memory_backend_gpu"], bench_config["memory_backend_fallback"], score_limit, "online_total", bench_config)
        check = consistency_check(cpu, gpu, case["source_image_path"])
        regression_checks.append(
            (f"score finite / output dimension at score_memory_limit={score_limit}", math.isfinite(check["score_max_abs_diff"]), f"max_abs_diff={check['score_max_abs_diff']:.8f}")
        )
        regression_checks.append(
            (f"nearest consistency at score_memory_limit={score_limit}", check["nearest_agreement"] >= 0.999, f"agreement={check['nearest_agreement']:.6f}; {check['cpu_backend']} vs {check['gpu_backend_actual']}")
        )
        tradeoff_df.loc[
            (tradeoff_df["backend_requested"] == bench_config["memory_backend_gpu"])
            & (tradeoff_df["score_memory_limit"] == score_limit),
            ["nearest_agreement", "score_max_abs_diff", "score_mean_abs_diff"],
        ] = [check["nearest_agreement"], check["score_max_abs_diff"], check["score_mean_abs_diff"]]

    cpu_probe = make_inferencer(base_config, case, bench_config["memory_backend_cpu"], bench_config["memory_backend_cpu"], 0, "online_total", bench_config)
    regression_checks.append(("original CPU path still runs", cpu_probe.model.memory_bank.backend == "faiss_cpu_flat", f"actual={cpu_probe.model.memory_bank.backend}"))
    gpu_probe = make_inferencer(base_config, case, bench_config["memory_backend_gpu"], bench_config["memory_backend_fallback"], 0, "online_total", bench_config)
    regression_checks.append(
        ("GPU request has explicit fallback", gpu_probe.model.memory_bank.backend in {"faiss_gpu_flat", "faiss_gpu_ivf", "torch_gpu_flat", "faiss_cpu_flat", "sklearn"}, f"actual={gpu_probe.model.memory_bank.backend}; reason={gpu_probe.model.memory_bank.backend_info.get('memory_backend_fallback_reason', '')}")
    )
    regression_checks.append(
        ("memory metadata feature dimension matches", int(gpu_probe.model.metadata["feature_dim"]) == gpu_probe.model.memory_bank.feature_dim, f"metadata={gpu_probe.model.metadata['feature_dim']}; bank={gpu_probe.model.memory_bank.feature_dim}")
    )
    try:
        gpu_probe.model.memory_bank.score(torch.zeros((1, gpu_probe.model.memory_bank.feature_dim + 1), device=gpu_probe.model.device))
        regression_checks.append(("feature dimension guard", False, "mismatched query unexpectedly accepted"))
    except ValueError as exc:
        regression_checks.append(("feature dimension guard", True, str(exc)))

    tradeoff_df.to_csv(OUTPUT_DIR / "accuracy_speed_tradeoff.csv", index=False, encoding="utf-8-sig")
    candidate_message = maybe_write_candidate_config(base_config, bench_config, tradeoff_df)
    write_implementation_report(
        OUTPUT_DIR / "implementation_report.md", bench_config, benchmark_df, tradeoff_df, faiss_gpu, candidate_message
    )
    write_regression_report(OUTPUT_DIR / "regression_test.md", regression_checks)
    (OUTPUT_DIR / "candidate_decision.txt").write_text(candidate_message + "\n", encoding="utf-8")
    print(json.dumps({"benchmark_rows": len(benchmark_df), "tradeoff_rows": len(tradeoff_df), "candidate": candidate_message}, ensure_ascii=False))


if __name__ == "__main__":
    main()
