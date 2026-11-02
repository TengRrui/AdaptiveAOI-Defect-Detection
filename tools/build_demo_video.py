"""Build a small, reproducible, right-to-left conveyor-style dataset demo video.

The source images are copied beforehand into assets/video_demo/source_images;
this script only reads that small demo copy and never changes the raw dataset.
OpenCV writes a video-only stream, so the generated video has no audio track.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from video.video_writer import VideoWriter


SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def parse_resolution(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        width, height = int(width_text), int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("分辨率格式应为例如 960x540。") from exc
    if width < 320 or height < 240:
        raise argparse.ArgumentTypeError("分辨率不得小于 320x240。")
    return width, height


def read_image(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"无法读取或已损坏的图像：{path}")
    return image


def letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized = cv2.resize(image, (max(1, round(source_width * scale)), max(1, round(source_height * scale))), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), 242, dtype=np.uint8)
    top = (height - resized.shape[0]) // 2
    left = (width - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def conveyor_background(width: int, height: int, elapsed: float, direction: str) -> np.ndarray:
    frame = np.full((height, width, 3), (36, 36, 36), dtype=np.uint8)
    belt_top = int(height * 0.17)
    belt_bottom = int(height * 0.84)
    cv2.rectangle(frame, (0, belt_top), (width, belt_bottom), (76, 76, 76), -1)
    cv2.rectangle(frame, (0, belt_top), (width, belt_bottom), (142, 142, 142), 3)
    sign = -1 if direction == "right_to_left" else 1
    offset = int((elapsed * 140 * sign) % 80)
    for x in range(-160, width + 160, 80):
        start = x + offset
        cv2.line(frame, (start, belt_bottom), (start + 130, belt_top), (96, 96, 96), 4, cv2.LINE_AA)
    cv2.putText(frame, "AdaptiveAOI DATASET DEMO", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (238, 238, 238), 2, cv2.LINE_AA)
    cv2.putText(frame, "offline test video | no pre-drawn predictions", (18, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return frame


def source_canvas_background(source: np.ndarray, width: int, height: int) -> np.ndarray:
    """Keep a current normal sample intact when source and video dimensions match.

    This mode is useful for a true image-to-video regression: the untouched
    background frame still goes through the same encoded/decoded path as a
    normal single image, while independently moving samples enter from the
    right and leave to the left.
    """
    if source.shape[1] == width and source.shape[0] == height:
        return source.copy()
    return cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA if source.shape[1] > width or source.shape[0] > height else cv2.INTER_LINEAR)


def build_demo_video(
    input_dir: str | Path,
    output_path: str | Path,
    fps: float = 20.0,
    max_duration: float = 12.0,
    resolution: tuple[int, int] = (960, 540),
    direction: str = "right_to_left",
    background_mode: str = "conveyor",
    seed: int = 20260729,
    sample_count: int = 8,
    seconds_per_sample: float = 1.05,
    overwrite: bool = False,
) -> dict:
    """Generate and re-read a video-only MP4/AVI.  Returns its manifest."""

    if direction not in {"right_to_left", "left_to_right"}:
        raise ValueError("direction 仅支持 right_to_left 或 left_to_right。")
    if background_mode not in {"conveyor", "source_canvas"}:
        raise ValueError("background_mode 仅支持 conveyor 或 source_canvas。")
    if fps <= 0 or max_duration <= 0 or sample_count < 1 or seconds_per_sample <= 0:
        raise ValueError("FPS、时长、样本数和每样本时长必须为正数。")
    source_dir = Path(input_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"输入图像目录不存在：{source_dir}")
    candidates = [path for path in sorted(source_dir.rglob("*")) if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES]
    if len(candidates) < sample_count:
        raise ValueError(f"输入目录只有 {len(candidates)} 张可读候选图像，少于请求的 {sample_count} 张。")
    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(len(candidates), size=sample_count, replace=False)
    selected_paths = [candidates[int(index)] for index in selected_indices]
    selected: list[tuple[Path, np.ndarray]] = []
    source_records = []
    for index, path in enumerate(selected_paths, start=1):
        image = read_image(path)
        selected.append((path, image))
        source_records.append(
            {
                "sample_id": index,
                "path": str(path),
                "file_size_bytes": path.stat().st_size,
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "sha256": sha256(path),
            }
        )
    width, height = resolution
    if background_mode == "source_canvas":
        item_width = item_height = max(150, int(min(width, height) * 0.32))
    else:
        item_width = max(150, int(width * 0.29))
        item_height = max(150, int(height * 0.58))
    belt_y = (height - item_height) // 2
    speed = max(260.0, width * 0.62)
    transit_seconds = (width + item_width + 80) / speed
    natural_duration = (sample_count - 1) * seconds_per_sample + transit_seconds + 0.20
    duration = min(float(max_duration), natural_duration)
    frame_total = max(1, int(round(duration * fps)))
    writer = VideoWriter(output_path, fps, width, height, overwrite=overwrite)
    writer_info = writer.open()
    prepared_items = [(path, letterbox(image, item_width, item_height)) for path, image in selected]
    normal_background = next((image for path, image in selected if "normal" in path.stem.lower() or "good" in path.stem.lower()), selected[0][1])
    try:
        for frame_index in range(frame_total):
            elapsed = frame_index / fps
            frame = source_canvas_background(normal_background, width, height) if background_mode == "source_canvas" else conveyor_background(width, height, elapsed, direction)
            for sample_index, (_, item) in enumerate(prepared_items):
                launched_at = sample_index * seconds_per_sample
                delta = elapsed - launched_at
                if direction == "right_to_left":
                    x = int(round(width + 40 - speed * delta))
                else:
                    x = int(round(-item_width - 40 + speed * delta))
                if x <= -item_width or x >= width:
                    continue
                source_x0 = max(0, -x)
                source_x1 = min(item_width, width - x)
                if source_x1 <= source_x0:
                    continue
                target_x0 = max(0, x)
                target_x1 = target_x0 + (source_x1 - source_x0)
                frame[belt_y : belt_y + item_height, target_x0:target_x1] = item[:, source_x0:source_x1]
                cv2.rectangle(frame, (target_x0, belt_y), (target_x1 - 1, belt_y + item_height - 1), (238, 238, 238), 2)
                if target_x1 - target_x0 > 100:
                    cv2.putText(frame, f"Sample {sample_index + 1}", (target_x0 + 10, belt_y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Sample {sample_index + 1}", (target_x0 + 10, belt_y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (250, 250, 250), 1, cv2.LINE_AA)
            writer.write(frame)
        writer_info = writer.close_and_promote()
    except Exception:
        writer.abort()
        raise
    capture = cv2.VideoCapture(str(writer_info.final_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"生成后的视频无法重新打开：{writer_info.final_path}")
        actual_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        actual_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0))
        actual_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0))
        actual_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0))
        ok_first, first = capture.read()
        last = first if ok_first else None
        while True:
            ok, current = capture.read()
            if not ok:
                break
            last = current
        if not ok_first or first is None or last is None or actual_frames <= 0:
            raise RuntimeError("生成视频的首帧或最后一帧读取失败。")
        if float(np.mean(first)) < 1.0 or float(np.mean(last)) < 1.0:
            raise RuntimeError("生成视频出现黑色首帧或结束帧。")
    finally:
        capture.release()
    manifest = {
        "source_kind": "dataset_image_synthesis",
        "source_image_directory": str(source_dir),
        "selected_sources": source_records,
        "output_video": str(writer_info.final_path),
        "output_sha256": sha256(writer_info.final_path),
        "file_size_bytes": writer_info.final_path.stat().st_size,
        "no_audio": True,
        "requested": {"fps": fps, "max_duration_seconds": max_duration, "resolution": [width, height], "direction": direction, "background_mode": background_mode, "seed": seed, "sample_count": sample_count},
        "actual": {"fps": actual_fps, "width": actual_width, "height": actual_height, "frame_count": actual_frames, "duration_seconds": actual_frames / actual_fps if actual_fps > 0 else None, "codec": writer_info.codec, "codec_fallback_used": writer_info.fallback_used, "readback_verified": True},
    }
    manifest_path = writer_info.final_path.with_suffix(".manifest.json")
    if manifest_path.exists() and not overwrite:
        counter = 1
        while manifest_path.exists():
            manifest_path = writer_info.final_path.with_name(f"{writer_info.final_path.stem}_{counter}.manifest.json")
            counter += 1
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="基于当前演示图像生成无音频右向左流动测试视频。")
    parser.add_argument("--input-dir", default=str(PROJECT_ROOT / "assets" / "video_demo" / "source_images"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "assets" / "video_demo" / "AdaptiveAOI_右向左流动测试视频.mp4"))
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--max-duration", type=float, default=12.0)
    parser.add_argument("--resolution", type=parse_resolution, default=(960, 540))
    parser.add_argument("--direction", choices=["right_to_left", "left_to_right"], default="right_to_left")
    parser.add_argument("--background-mode", choices=["conveyor", "source_canvas"], default="conveyor")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--seconds-per-sample", type=float, default=1.05)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = build_demo_video(
        input_dir=args.input_dir,
        output_path=args.output,
        fps=args.fps,
        max_duration=args.max_duration,
        resolution=args.resolution,
        direction=args.direction,
        background_mode=args.background_mode,
        seed=args.seed,
        sample_count=args.sample_count,
        seconds_per_sample=args.seconds_per_sample,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
