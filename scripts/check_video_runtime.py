"""Fast, side-effect-free video runtime probe used by the launcher."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np


def probe() -> dict[str, object]:
    build = cv2.getBuildInformation()
    with tempfile.TemporaryDirectory(prefix="adaptiveaoi_video_probe_") as directory:
        target = Path(directory) / "probe.mp4"
        writer = cv2.VideoWriter(
            str(target),
            cv2.VideoWriter_fourcc(*"mp4v"),
            8.0,
            (96, 64),
        )
        writer_opened = writer.isOpened()
        if writer_opened:
            for index in range(4):
                frame = np.full((64, 96, 3), 40 + index * 30, dtype=np.uint8)
                writer.write(frame)
        writer.release()

        capture = cv2.VideoCapture(str(target))
        read_opened = capture.isOpened()
        readable, frame = capture.read() if read_opened else (False, None)
        capture.release()

        result = {
            "opencv_version": cv2.__version__,
            "opencv_ffmpeg": "FFMPEG:                      YES" in build,
            "ffmpeg_cli": shutil.which("ffmpeg") is not None,
            "ffprobe_cli": shutil.which("ffprobe") is not None,
            "mp4v_writer": bool(writer_opened),
            "mp4_readback": bool(readable and frame is not None and frame.size > 0),
        }
        result["ready"] = bool(result["mp4v_writer"] and result["mp4_readback"])
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the AdaptiveAOI video runtime.")
    parser.parse_args()
    result = probe()
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
