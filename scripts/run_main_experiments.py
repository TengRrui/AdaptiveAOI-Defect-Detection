from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.patchcore.build_memory import build_memory
from methods.patchcore.infer import default_memory_path, run_batch_inference
from tools.metrics import evaluate_result_csv
from tools.path_utils import load_config


def log_error(output_root: Path, title: str, message: str):
    log_path = output_root / "logs" / "full_experiment_error_report.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {title}\n\n{message.strip()}\n")


def categories(config: dict, dataset: str) -> list[str]:
    manifest_path = Path(config["output_root"]) / "results" / f"manifest_{dataset}.csv"
    df = pd.read_csv(manifest_path)
    return sorted(df["category"].astype(str).unique().tolist())


def run_one(config: dict, dataset: str, category: str, result_name: str) -> dict:
    run_cfg = dict(config)
    run_cfg["dataset"] = dataset
    run_cfg["category"] = category
    if not default_memory_path(run_cfg, dataset, category).exists():
        build_memory(run_cfg, category=category)
    device = torch.device(run_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _, result_csv = run_batch_inference(run_cfg, category=category, split="test", result_name=result_name)
    metrics = evaluate_result_csv(result_csv)
    peak_memory = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else 0.0
    metrics.update(
        {
            "dataset": dataset,
            "category": category,
            "algorithm": "A0_baseline",
            "peak_memory": peak_memory,
            "status": "ok",
            "result_csv": str(result_csv),
        }
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run main full-category experiments.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True, choices=["mvtec", "visa", "dagm"])
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--limit_categories", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = args.dataset
    output_root = Path(config["output_root"])
    cats = args.categories or categories(config, dataset)
    if args.limit_categories:
        cats = cats[: int(args.limit_categories)]

    rows = []
    for category in cats:
        print(f"[main] {dataset}/{category}")
        try:
            rows.append(run_one(config, dataset, category, result_name="main_result.csv"))
        except Exception as exc:
            message = f"{dataset}/{category}: {type(exc).__name__}: {exc}"
            log_error(output_root, "main experiment failed", message)
            rows.append({"dataset": dataset, "category": category, "status": "failed", "error": message})

    out_path = output_root / "results" / f"main_{dataset}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    summary_path = output_root / "results" / "main_summary.csv"
    existing = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    combined = pd.concat([existing[existing.get("dataset", pd.Series(dtype=str)) != dataset], df], ignore_index=True) if not existing.empty else df
    combined.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(json.dumps({"output": str(out_path), "categories": len(cats), "ok": int((df["status"] == "ok").sum())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
