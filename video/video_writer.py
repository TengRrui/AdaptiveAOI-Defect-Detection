"""Safe OpenCV video output creation and post-write validation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2


class VideoOutputError(RuntimeError):
    """Raised when no usable writer/codec can be created or validated."""


@dataclass
class WriterInfo:
    final_path: Path
    temporary_path: Path
    fps: float
    width: int
    height: int
    codec: str
    fallback_used: bool


def safe_output_path(requested: str | Path, overwrite: bool = False) -> Path:
    path = Path(requested).expanduser().resolve()
    if not path.suffix:
        path = path.with_suffix(".mp4")
    if not path.exists() or overwrite:
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


class VideoWriter:
    """Writes to a temporary sibling file then validates and atomically promotes it."""

    def __init__(self, output_path: str | Path, fps: float, width: int, height: int, overwrite: bool = False):
        self.final_path = safe_output_path(output_path, overwrite=overwrite)
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self.writer: cv2.VideoWriter | None = None
        self.info: WriterInfo | None = None
        self.frame_count = 0

    def open(self) -> WriterInfo:
        if self.final_path.suffix.lower() not in {".mp4", ".avi", ".webm"}:
            self.final_path = self.final_path.with_suffix(".mp4")
        # Prefer browser-native codecs.  OpenCV Windows builds often lack an
        # H.264 encoder, so VP8/WebM is the portable browser fallback.  MPEG-4
        # Part 2 and MJPEG remain available for system-player compatibility.
        candidates = [
            ("avc1", ".mp4", False),
            ("VP80", ".webm", True),
            ("mp4v", ".mp4", True),
            ("MJPG", ".avi", True),
        ]
        base = self.final_path
        for codec, suffix, fallback in candidates:
            final = base.with_suffix(suffix)
            if final.exists() and final != self.final_path:
                final = safe_output_path(final, overwrite=False)
            temporary = final.with_name(f"{final.stem}.writing{final.suffix}")
            if temporary.exists():
                raise VideoOutputError(f"检测到未清理的临时视频，拒绝覆盖：{temporary}")
            writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*codec), self.fps, (self.width, self.height))
            if writer.isOpened():
                self.final_path = final
                self.writer = writer
                self.info = WriterInfo(final, temporary, self.fps, self.width, self.height, codec, fallback)
                return self.info
            writer.release()
        raise VideoOutputError("无法创建浏览器兼容的视频编码器；已尝试 H.264、VP8、MPEG-4 和 MJPEG。")

    def write(self, bgr_frame) -> None:
        if self.writer is None or self.info is None:
            raise RuntimeError("VideoWriter 尚未打开。")
        if bgr_frame is None or bgr_frame.shape[1] != self.width or bgr_frame.shape[0] != self.height:
            raise VideoOutputError("输出帧尺寸与 VideoWriter 初始化尺寸不一致。")
        self.writer.write(bgr_frame)
        self.frame_count += 1

    def close_and_promote(self) -> WriterInfo:
        if self.writer is None or self.info is None:
            raise RuntimeError("VideoWriter 尚未打开。")
        self.writer.release()
        self.writer = None
        temporary = self.info.temporary_path
        if not temporary.exists() or temporary.stat().st_size <= 0 or self.frame_count <= 0:
            raise VideoOutputError(f"输出视频为空或没有写入任何帧：{temporary}")
        capture = cv2.VideoCapture(str(temporary))
        try:
            if not capture.isOpened():
                raise VideoOutputError(f"输出视频无法重新读取：{temporary}")
            actual_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0))
            actual_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0))
            ok_first, first = capture.read()
            if not ok_first or first is None or actual_width != self.width or actual_height != self.height:
                raise VideoOutputError(f"输出视频重新读取尺寸或首帧校验失败：{temporary}")
            last = first
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                last = frame
            if last is None:
                raise VideoOutputError(f"输出视频最后一帧无法读取：{temporary}")
        finally:
            capture.release()
        os.replace(temporary, self.info.final_path)
        return self.info

    def abort(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
