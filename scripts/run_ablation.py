from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.calibration import calibration_dir
from methods.patchcore.build_memory import build_memory
from methods.patchcore.infer import default_memory_path, run_batch_inference
from tools.metrics import evaluate_result_csv
from tools.path_utils import append_error_report, load_config


STAGE_CONFIGS = {
    "A0": {"name": "baseline", "use_calibrated_threshold": False, "use_defect_direction": False, "use_tiling": False, "use_fp16": False},
    "A1": {"name": "baseline + threshold calibration", "use_calibrated_threshold": True, "use_defect_direction": False, "use_tiling": False, "use_fp16": False},
    "A2": {"name": "baseline + defect direction", "use_calibrated_threshold": False, "use_defect_direction": True, "use_tiling": False, "use_fp16": False},
    "A3": {"name": "baseline + defect direction + threshold calibration", "use_calibrated_threshold": True, "use_defect_direction": True, "use_tiling": False, "use_fp16": False},
    "A4": {"name": "full algorithm", "use_calibrated_threshold": True, "use_defect_direction": True, "use_tiling": True, "use_fp16": True},
}


def run_calibrate(config_path: str, dataset: str, category: str, normal_shot: int, defect_shot: int, seed: int, use_defect: bool, threshold_strategy: str):
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "calibrate.py"),
        "--config",
        config_path,
        "--dataset",
        dataset,
        "--category",
        category,
        "--normal_shot",
        str(normal_shot),
        "--defect_shot",
        str(defect_shot),
        "--seed",
        str(seed),
        "--threshold_strategy",
        threshold_strategy,
    ]
    if use_defect:
        cmd.append("--use_defect_direction")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def calibration_uses_defect(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("use_defect_direction", False))
    except Exception:
        return False


def prepare_calibrations(config_path: str, config: dict, dataset: str, category: str, normal_shot: int, defect_shot: int, seed: int, threshold_strategy: str, stages: list[str]):
    out_dir = calibration_dir(config, dataset, category)
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold_only = out_dir / "calibration_threshold_only.json"
    with_defect = out_dir / "calibration_with_defect_direction.json"

    need_threshold_only = any(stage.upper() == "A1" for stage in stages)
    need_defect = any(stage.upper() in {"A2", "A3", "A4"} for stage in stages)

    if need_threshold_only and not threshold_only.exists():
        run_calibrate(config_path, dataset, category, normal_shot, defect_shot, seed, False, threshold_strategy)
        shutil.copy2(out_dir / "calibration.json", threshold_only)
    if need_defect and (not with_defect.exists() or not (out_dir / "defect_direction.pt").exists()):
        if calibration_uses_defect(out_dir / "calibration.json") and (out_dir / "defect_direction.pt").exists():
            shutil.copy2(out_dir / "calibration.json", with_defect)
        else:
            run_calibrate(config_path, dataset, category, normal_shot, defect_shot, seed, True, threshold_strategy)
            shutil.copy2(out_dir / "calibration.json", with_defect)
    if need_defect and not with_defect.exists():
        run_calibrate(config_path, dataset, category, normal_shot, defect_shot, seed, True, threshold_strategy)
        shutil.copy2(out_dir / "calibration.json", with_defect)
    return threshold_only, with_defect


def main():
    parser = argparse.ArgumentParser(description="Run A0-A4 ablation for AdaptiveAOI algorithm.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True, choices=["mvtec", "visa", "dagm"])
    parser.add_argument("--category", required=True)
    parser.add_argument("--stages", nargs="*", default=["A0", "A1", "A2", "A3", "A4"])
    parser.add_argument("--normal_shot", type=int, default=100)
    parser.add_argument("--defect_shot", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold_strategy", default="best_f1")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config["dataset"] = args.dataset
    config["category"] = args.category
    try:
        if not default_memory_path(config, args.dataset, args.category).exists():
            build_memory(config, category=args.category)
        threshold_only, with_defect = prepare_calibrations(
            args.config,
            config,
            args.dataset,
            args.category,
            args.normal_shot,
            args.defect_shot,
            args.seed,
            args.threshold_strategy,
            args.stages,
        )

        rows = []
        for stage in args.stages:
            stage = stage.upper()
            stage_cfg = STAGE_CONFIGS[stage]
            run_cfg = dict(config)
            run_cfg.update(stage_cfg)
            run_cfg["device"] = "cuda" if str(config.get("device", "cuda")) == "cuda" else str(config.get("device", "cpu"))
            if stage in {"A1"}:
                run_cfg["calibration_path"] = str(threshold_only)
            if stage in {"A3", "A4"}:
                run_cfg["calibration_path"] = str(with_defect)
            if stage == "A4":
                run_cfg["tile_size"] = int(config.get("tile_size", 768))
                run_cfg["tile_overlap"] = int(config.get("tile_overlap", 128))
                run_cfg["tile_batch_size"] = int(config.get("tile_batch_size", 4))
                run_cfg["tile_fusion"] = str(config.get("tile_fusion", "max"))
            result_name = f"ablation_{stage}_result.csv"
            _, result_csv = run_batch_inference(run_cfg, category=args.category, split="test", limit=args.limit, result_name=result_name)
            metrics = evaluate_result_csv(result_csv)
            metrics.update(
                {
                    "stage": stage,
                    "stage_name": stage_cfg["name"],
                    "dataset": args.dataset,
                    "category": args.category,
                    "use_calibrated_threshold": bool(stage_cfg["use_calibrated_threshold"]),
                    "use_defect_direction": bool(stage_cfg["use_defect_direction"]),
                    "use_tiling": bool(stage_cfg["use_tiling"]),
                    "use_fp16": bool(stage_cfg["use_fp16"]),
                    "result_csv": str(result_csv),
                }
            )
            rows.append(metrics)
            print(f"finished {stage}: {args.dataset}/{args.category}")

        out_path = Path(config["output_root"]) / "results" / f"ablation_{args.dataset}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame(rows)
        if out_path.exists():
            old_df = pd.read_csv(out_path)
            if {"dataset", "category"}.issubset(old_df.columns):
                old_df = old_df[
                    ~(
                        (old_df["dataset"].astype(str).str.lower() == args.dataset)
                        & (old_df["category"].astype(str) == args.category)
                    )
                ]
                new_df = pd.concat([old_df, new_df], ignore_index=True)
        new_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        summary_path = Path(config["output_root"]) / "results" / "ablation_summary.csv"
        all_frames = []
        for path in (Path(config["output_root"]) / "results").glob("ablation_*.csv"):
            if path.name == "ablation_summary.csv":
                continue
            all_frames.append(pd.read_csv(path))
        if all_frames:
            pd.concat(all_frames, ignore_index=True).to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"ablation_csv: {out_path}")
        print(pd.DataFrame(rows).to_string(index=False))
    except Exception as exc:
        append_error_report(config, "run_ablation failed", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
