from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.calibration import choose_image_threshold, sample_calibration_rows
from methods.patchcore.build_memory import build_memory
from methods.patchcore.infer import default_memory_path, run_batch_inference
from tools.metrics import binary_metrics, safe_auroc
from tools.path_utils import load_config


DEFAULT_CATEGORIES = {
    "mvtec": ["bottle", "grid", "toothbrush"],
    "visa": ["candle", "cashew", "pcb1"],
    "dagm": ["Class1", "Class2", "Class3"],
}


def evaluate_scores(df: pd.DataFrame, threshold: float) -> dict:
    pred = (df["score"].astype(float).to_numpy() >= threshold).astype(int)
    y = df["label_gt"].astype(int).to_numpy()
    image = binary_metrics(y, pred)
    return {
        "I-AUROC": safe_auroc(y, df["score"].astype(float).to_numpy()),
        "image_F1": image["f1"],
        "Precision": image["precision"],
        "Recall": image["recall"],
    }


def ensure_result(config: dict, dataset: str, category: str) -> Path:
    run_cfg = dict(config)
    run_cfg["dataset"] = dataset
    run_cfg["category"] = category
    if not default_memory_path(run_cfg, dataset, category).exists():
        build_memory(run_cfg, category=category)
    _, result_csv = run_batch_inference(run_cfg, category=category, split="test", result_name="fewshot_base_result.csv")
    return result_csv


def main():
    parser = argparse.ArgumentParser(description="Run efficient few-shot threshold calibration experiments.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True, choices=["mvtec", "visa", "dagm"])
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--normal_shot", type=int, default=100)
    parser.add_argument("--defect_shots", nargs="*", type=int, default=[0, 1, 5, 10, 30])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = args.dataset
    cats = args.categories or DEFAULT_CATEGORIES[dataset]
    output_root = Path(config["output_root"])
    manifest = pd.read_csv(output_root / "results" / f"manifest_{dataset}.csv")
    rows = []

    for category in cats:
        print(f"[fewshot] {dataset}/{category}")
        result_csv = ensure_result(config, dataset, category)
        result_df = pd.read_csv(result_csv)
        cat_manifest = manifest[(manifest["dataset"].astype(str).str.lower() == dataset) & (manifest["category"].astype(str) == category)].reset_index(drop=True)
        for defect_shot in args.defect_shots:
            for seed in args.seeds:
                split_df = sample_calibration_rows(cat_manifest, args.normal_shot, defect_shot, seed=seed)
                normal_used = int((split_df["label"].astype(int) == 0).sum())
                defect_available = int((cat_manifest["label"].astype(int) == 1).sum())
                defect_used = int((split_df["label"].astype(int) == 1).sum())
                split_dir = output_root / "calibrations" / dataset / category / "fewshot" / f"defect_{defect_shot}" / f"seed_{seed}"
                split_dir.mkdir(parents=True, exist_ok=True)
                split_payload = {
                    "dataset": dataset,
                    "category": category,
                    "normal_shot_requested": int(args.normal_shot),
                    "normal_shot_used": normal_used,
                    "defect_shot_requested": int(defect_shot),
                    "defect_shot_used": defect_used,
                    "seed": int(seed),
                    "samples": split_df[["image_path", "label", "label_name", "defect_type", "mask_path"]].to_dict("records"),
                    "note": "defect samples capped by availability" if defect_used < defect_shot else "",
                }
                (split_dir / "calibration_split.json").write_text(json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8")

                joined = split_df.merge(result_df[["image_path", "score"]], on="image_path", how="inner")
                if defect_shot == 0:
                    threshold_detail = choose_image_threshold(joined["label"].astype(int), joined["score"].astype(float), "normal_percentile", percentile=99.0)
                else:
                    threshold_detail = choose_image_threshold(joined["label"].astype(int), joined["score"].astype(float), "best_f1")
                threshold = float(threshold_detail["threshold"])
                eval_df = result_df[~result_df["image_path"].isin(set(split_df["image_path"].astype(str)))]
                if eval_df.empty:
                    eval_df = result_df
                metrics = evaluate_scores(eval_df, threshold)
                rows.append(
                    {
                        "dataset": dataset,
                        "category": category,
                        "normal_shot": normal_used,
                        "defect_shot_requested": defect_shot,
                        "defect_shot_used": defect_used,
                        "defect_available": defect_available,
                        "seed": seed,
                        "threshold": threshold,
                        "eval_count": len(eval_df),
                        "calibration_split": str(split_dir / "calibration_split.json"),
                        **metrics,
                    }
                )

    out_path = output_root / "results" / f"fewshot_{dataset}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    all_df = pd.DataFrame(rows)
    summary = all_df.groupby(["dataset", "category", "defect_shot_requested"], as_index=False).agg(
        I_AUROC_mean=("I-AUROC", "mean"),
        I_AUROC_std=("I-AUROC", "std"),
        image_F1_mean=("image_F1", "mean"),
        image_F1_std=("image_F1", "std"),
        Precision_mean=("Precision", "mean"),
        Recall_mean=("Recall", "mean"),
    )
    summary_path = output_root / "results" / "fewshot_summary.csv"
    existing = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    combined = pd.concat([existing[existing.get("dataset", pd.Series(dtype=str)) != dataset], summary], ignore_index=True) if not existing.empty else summary
    combined.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(json.dumps({"output": str(out_path), "rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
