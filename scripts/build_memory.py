from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.patchcore.build_memory import build_memory
from tools.path_utils import append_error_report, load_config


def main():
    parser = argparse.ArgumentParser(description="Build AdaptiveAOI memory bank.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--category", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.category:
        config["category"] = args.category
    try:
        metadata = build_memory(config, category=args.category)
        print("Memory build finished.")
        for key in [
            "dataset",
            "category",
            "backbone",
            "feature_layers",
            "image_size",
            "original_memory_size",
            "coreset_memory_size",
            "feature_dim",
            "build_time",
            "sample_count",
            "threshold",
            "memory_path",
        ]:
            print(f"{key}: {metadata.get(key)}")
    except Exception as exc:
        append_error_report(config, "build_memory failed", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
