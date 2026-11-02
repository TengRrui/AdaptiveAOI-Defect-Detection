"""OpenCV-backed, sequential video-file decoding with deterministic timestamps."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import cv2

from .video_types import FramePacket, VideoMetadata


class VideoInputError(RuntimeError):
    """Raised when a requested input cannot be decoded as a video file."""


def _fourcc_to_text(value: float) -> str:
    integer = int(value)
    return "".join(chr((integer >> (8 * index)) & 0xFF) for index in range(4)).strip("\x00")


class VideoReader:
    """Context-managed standard video reader; it never buffers the entire video."""

    def __init__(self, input_video: str | Path):
        self.path = Path(input_video).expanduser().resolve()
        self.capture: cv2.VideoCapture | None = None
        self.metadata: VideoMetadata | None = None

    def open(self) -> VideoMetadata:
        if not self.path.exists() or not self.path.is_file():
            raise VideoInputError(f"输入视频文件不存在：{self.path}")
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise VideoInputError(f"视频无法打开或解码：{self.path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0))
        if fps <= 0.0 or width <= 0 or height <= 0:
            capture.release()
            raise VideoInputError(f"视频元数据无效（FPS/尺寸为空）：{self.path}")
        duration = (frame_count / fps) if frame_count > 0 else None
        self.capture = capture
        self.metadata = VideoMetadata(
            path=self.path,
            source_fps=fps,
            width=width,
            height=height,
            reported_frame_count=frame_count,
            duration_seconds=duration,
            fourcc=_fourcc_to_text(capture.get(cv2.CAP_PROP_FOURCC)),
        )
        return self.metadata

    def __enter__(self) -> "VideoReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def iter_frames(
        self,
        start_time: float | None = None,
        end_time: float | None = None,
        max_frames: int | None = None,
    ) -> Iterator[FramePacket]:
        if self.capture is None or self.metadata is None:
            raise RuntimeError("VideoReader 尚未打开。")
        if start_time is not None and start_time < 0:
            raise ValueError("start_time 不能小于 0。")
        if end_time is not None and start_time is not None and end_time < start_time:
            raise ValueError("end_time 不能小于 start_time。")
        fps = self.metadata.source_fps
        start_index = int(round(float(start_time or 0.0) * fps))
        if start_index > 0:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, start_index)
        frame_index = start_index
        emitted = 0
        previous_timestamp = -1.0
        while True:
            if max_frames is not None and emitted >= int(max_frames):
                return
            started = time.perf_counter()
            ok, frame = self.capture.read()
            decode_ms = (time.perf_counter() - started) * 1000.0
            if not ok or frame is None:
                return
            timestamp_ms = float(frame_index / fps * 1000.0)
            if timestamp_ms <= previous_timestamp:
                timestamp_ms = previous_timestamp + (1000.0 / fps)
            if end_time is not None and timestamp_ms > float(end_time) * 1000.0 + 1e-6:
                return
            previous_timestamp = timestamp_ms
            yield FramePacket(frame_index=frame_index, timestamp_ms=timestamp_ms, bgr_frame=frame, decode_ms=decode_ms)
            frame_index += 1
            emitted += 1
