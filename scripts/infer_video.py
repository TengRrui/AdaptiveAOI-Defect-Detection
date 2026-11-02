"""Command-line entry for AdaptiveAOI offline video-file frame-by-frame detection."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.path_utils import load_config
from video.video_inference import VideoDetectionOptions, run_video_detection


def load_runtime_config(args: argparse.Namespace) -> tuple[dict, Path]:
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    previous_data_root = os.environ.get("ADAPTIVEAOI_DATA_ROOT")
    try:
        if args.data_root:
            os.environ["ADAPTIVEAOI_DATA_ROOT"] = str(Path(args.data_root).expanduser().resolve())
        config = load_config(config_path)
    finally:
        if args.data_root:
            if previous_data_root is None:
                os.environ.pop("ADAPTIVEAOI_DATA_ROOT", None)
            else:
                os.environ["ADAPTIVEAOI_DATA_ROOT"] = previous_data_root
    if args.category:
        config["category"] = args.category
    if args.dataset:
        config["dataset"] = args.dataset
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    config["device"] = device
    config["use_fp16"] = bool(config.get("use_fp16", False) and device == "cuda")
    config["timing_mode"] = args.timing_mode
    calibration_path = Path(args.calibration_path).expanduser().resolve() if args.calibration_path else Path(config["output_root"]) / "calibrations" / str(config["dataset"]).lower() / str(config["category"]) / "calibration.json"
    if args.threshold is None and calibration_path.is_file():
        config["use_calibrated_threshold"] = True
        config["calibration_path"] = str(calibration_path)
    elif args.threshold is not None:
        config["threshold"] = float(args.threshold)
        config["use_calibrated_threshold"] = False
    memory_path = Path(args.memory_path).expanduser().resolve() if args.memory_path else Path(config["output_root"]) / "checkpoints" / str(config["dataset"]).lower() / str(config["category"]) / "memory_bank.pt"
    if not memory_path.is_file():
        raise FileNotFoundError(f"模型记忆库不存在：{memory_path}。请检查 --memory-path、数据集和类别。")
    return config, memory_path


def main() -> int:
    parser = argparse.ArgumentParser(description="AdaptiveAOI 离线视频文件逐帧检测（复用当前正式 PatchCore 图像推理接口）。")
    parser.add_argument("--input", required=True, help="输入视频路径。")
    parser.add_argument("--output", required=True, help="带检测可视化的视频输出路径。")
    parser.add_argument("--config", required=True, help="当前正式算法配置 YAML。")
    parser.add_argument("--data-root", default=None, help="仅当前进程使用的外部数据根目录，不修改 Windows 环境变量。")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--memory-path", default=None)
    parser.add_argument("--calibration-path", default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--timing-mode", choices=["strict_total", "online_total", "detection_only", "no_save_total"], default="strict_total")
    parser.add_argument("--output-dir", default=None, help="逐帧 JSONL/CSV/汇总与关键帧目录；默认为输出视频所在目录。")
    parser.add_argument("--save-heatmap", action="store_true")
    parser.add_argument("--save-mask", action="store_true")
    parser.add_argument("--save-jsonl", dest="save_jsonl", action="store_true", default=True)
    parser.add_argument("--no-save-jsonl", dest="save_jsonl", action="store_false")
    parser.add_argument("--save-csv", dest="save_csv", action="store_true", default=True)
    parser.add_argument("--no-save-csv", dest="save_csv", action="store_false")
    parser.add_argument("--save-keyframes", dest="save_keyframes", action="store_true", default=True)
    parser.add_argument("--no-save-keyframes", dest="save_keyframes", action="store_false")
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        input_path = Path(args.input).expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"输入视频不存在：{input_path}")
        config, memory_path = load_runtime_config(args)
        summary = run_video_detection(
            VideoDetectionOptions(
                input_video=input_path,
                output_video=args.output,
                output_dir=args.output_dir,
                config=config,
                memory_path=memory_path,
                config_path=str(Path(args.config).expanduser().resolve()),
                device=str(config["device"]),
                threshold=args.threshold,
                frame_stride=args.frame_stride,
                start_time=args.start_time,
                end_time=args.end_time,
                max_frames=args.max_frames,
                save_overlay=not args.no_overlay,
                save_heatmap=args.save_heatmap,
                save_mask=args.save_mask,
                save_jsonl=args.save_jsonl,
                save_csv=args.save_csv,
                save_keyframes=args.save_keyframes,
                overwrite=args.overwrite,
            )
        )
    except Exception as exc:
        print(f"视频检测失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
