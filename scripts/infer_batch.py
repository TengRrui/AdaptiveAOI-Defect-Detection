from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.patchcore.infer import run_batch_inference
from tools.path_utils import append_error_report, load_config


def main():
    parser = argparse.ArgumentParser(description="Run batch inference with AdaptiveAOI.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--category", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--async_prefetch", action="store_true")
    parser.add_argument("--prefetch_queue_size", type=int, default=None)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--result_name", default="result.csv")
    parser.add_argument("--async_save_outputs", action="store_true")
    parser.add_argument("--save_queue_size", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.category:
        config["category"] = args.category
    if args.async_prefetch:
        config["async_prefetch"] = True
    if args.prefetch_queue_size is not None:
        config["prefetch_queue_size"] = max(1, int(args.prefetch_queue_size))
    if args.pin_memory:
        config["pin_memory"] = True
    if args.async_save_outputs:
        config["async_save_outputs"] = True
    if args.save_queue_size is not None:
        config["save_queue_size"] = max(1, int(args.save_queue_size))
    try:
        df, result_csv = run_batch_inference(config, category=config["category"], split=args.split, limit=args.limit, result_name=args.result_name)
        print("Batch inference finished.")
        print(f"result_csv: {result_csv}")
        print(f"image_count: {len(df)}")
        if len(df):
            print(f"mean_ms: {df['total_ms'].mean():.2f}")
            print(f"p95_ms: {df['total_ms'].quantile(0.95):.2f}")
    except Exception as exc:
        append_error_report(config, "infer_batch failed", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
