from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.patchcore.build_memory import build_memory
from methods.patchcore.infer import PatchCoreInferencer, default_memory_path
from tools.path_utils import load_config


DEFAULT_CATEGORIES = {
    "mvtec": ["bottle", "grid", "toothbrush"],
    "visa": ["candle", "cashew", "pcb1"],
    "dagm": ["Class1", "Class2", "Class3"],
}


def copy_if_exists(src: str, dst: Path):
    if src and str(src).lower() != "nan" and Path(src).exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Collect normal/anomaly visualization samples.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True, choices=["mvtec", "visa", "dagm"])
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--normal_count", type=int, default=3)
    parser.add_argument("--anomaly_count", type=int, default=3)
    args = parser.parse_args()

    config = load_config(args.config)
    config["dataset"] = args.dataset
    cats = args.categories or DEFAULT_CATEGORIES[args.dataset]
    manifest = pd.read_csv(Path(config["output_root"]) / "results" / f"manifest_{args.dataset}.csv")

    for category in cats:
        config["category"] = category
        if not default_memory_path(config, args.dataset, category).exists():
            build_memory(config, category=category)
        inferencer = PatchCoreInferencer(config, category=category)
        cat_df = manifest[(manifest["category"].astype(str) == category) & (manifest["split"].astype(str).str.lower() == "test")]
        samples = pd.concat(
            [
                cat_df[cat_df["label"].astype(int) == 0].head(args.normal_count),
                cat_df[cat_df["label"].astype(int) == 1].head(args.anomaly_count),
            ],
            ignore_index=True,
        )
        out_dir = Path(config["output_root"]) / "visualizations" / args.dataset / category
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, item in enumerate(samples.to_dict("records")):
            prefix = f"{idx:02d}_{'normal' if int(item['label']) == 0 else 'anomaly'}_{Path(item['image_path']).stem}"
            result = inferencer.infer_image(item["image_path"], meta=item, save_outputs=True)
            copy_if_exists(item["image_path"], out_dir / f"{prefix}_original{Path(item['image_path']).suffix}")
            copy_if_exists(item.get("mask_path", ""), out_dir / f"{prefix}_gt_mask.png")
            copy_if_exists(result["heatmap_path"], out_dir / f"{prefix}_heatmap.png")
            copy_if_exists(result["mask_path"], out_dir / f"{prefix}_binary_mask.png")
            copy_if_exists(result["overlay_path"], out_dir / f"{prefix}_overlay.png")
        print(f"visualizations: {args.dataset}/{category} -> {out_dir}")


if __name__ == "__main__":
    main()
