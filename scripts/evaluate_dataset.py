from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.patchcore.build_memory import build_memory
from methods.patchcore.infer import default_memory_path
from methods.patchcore.infer import run_batch_inference
from tools.metrics import evaluate_result_csv
from tools.path_utils import append_error_report, load_config


def categories_from_manifest(config: dict, dataset: str) -> list[str]:
    manifest_path = Path(config["output_root"]) / "results" / f"manifest_{dataset}.csv"
    df = pd.read_csv(manifest_path)
    return sorted(df["category"].astype(str).unique().tolist())


def evaluate_category(config: dict, dataset: str, category: str) -> dict:
    config = dict(config)
    config["dataset"] = dataset
    config["category"] = category
    if not default_memory_path(config, dataset, category).exists():
        print(f"memory missing, building first: {dataset}/{category}")
        build_memory(config, category=category)
    _, result_csv = run_batch_inference(config, category=category, split="test")
    metrics = evaluate_result_csv(result_csv)
    metrics.update({"dataset": dataset, "category": category, "result_csv": str(result_csv)})
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate AdaptiveAOI on one or all categories.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True, choices=["mvtec", "visa", "dagm"])
    parser.add_argument("--category", default=None)
    parser.add_argument("--all_categories", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config["dataset"] = args.dataset
    try:
        if args.all_categories:
            categories = categories_from_manifest(config, args.dataset)
        elif args.category:
            categories = [args.category]
        else:
            categories = [str(config["category"])]

        rows = []
        for category in categories:
            rows.append(evaluate_category(config, args.dataset, category))
            print(f"evaluated: {args.dataset}/{category}")

        out_path = Path(config["output_root"]) / "results" / f"metrics_{args.dataset}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"metrics_csv: {out_path}")
        print(pd.DataFrame(rows).to_string(index=False))
    except Exception as exc:
        append_error_report(config, "evaluate_dataset failed", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
