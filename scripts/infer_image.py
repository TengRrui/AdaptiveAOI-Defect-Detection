from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.patchcore.infer import PatchCoreInferencer
from tools.path_utils import append_error_report, load_config


def main():
    parser = argparse.ArgumentParser(description="Infer one image with AdaptiveAOI.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--category", default=None)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--memory_path", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.category:
        config["category"] = args.category
    try:
        inferencer = PatchCoreInferencer(config, category=config["category"], memory_path=args.memory_path)
        result = inferencer.infer_image(args.image_path, save_outputs=True)
        result_path = inferencer.save_single_result(result)
        clean = {k: v for k, v in result.items() if not k.startswith("_")}
        print(json.dumps(
            {
                "image_score": clean["score"],
                "threshold": clean["threshold"],
                "pred_label": clean["pred_label"],
                "total_ms": clean["total_ms"],
                "result_json": str(result_path),
                "heatmap_path": clean["heatmap_path"],
                "mask_path": clean["mask_path"],
                "overlay_path": clean["overlay_path"],
            },
            ensure_ascii=False,
            indent=2,
        ))
    except Exception as exc:
        append_error_report(config, "infer_image failed", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
