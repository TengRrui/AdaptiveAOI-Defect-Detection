"""Typed value objects used by the AdaptiveAOI video-file pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    source_fps: float
    width: int
    height: int
    reported_frame_count: int
    duration_seconds: float | None
    fourcc: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        return payload


@dataclass
class FramePacket:
    frame_index: int
    timestamp_ms: float
    bgr_frame: np.ndarray
    decode_ms: float


@dataclass
class FrameResult:
    frame_index: int
    timestamp_ms: float
    source_fps: float
    original_width: int
    original_height: int
    inference_width: int
    inference_height: int
    inference_time_ms: float | None
    frame_pipeline_ms: float | None
    anomaly_score: float | None
    prediction: str
    prediction_label: int | None
    threshold: float | None
    baseline_score: float | None = None
    calibrated_score: float | None = None
    final_score: float | None = None
    calibration_applied: bool = False
    gate_status: str = "not_applicable"
    fallback_used: bool = False
    calibration_version: str = ""
    method_mode: str = "normal_memory"
    heatmap_shape: list[int] | None = None
    heatmap_min: float | None = None
    heatmap_max: float | None = None
    boxes: list[dict[str, Any]] = field(default_factory=list)
    heatmap_path: str = ""
    mask_path: str = ""
    error: str = ""
    status: str = "success"
    decode_ms: float = 0.0
    preprocess_ms: float = 0.0
    feature_extraction_ms: float = 0.0
    ann_search_ms: float = 0.0
    anomaly_scoring_ms: float = 0.0
    postprocess_ms: float = 0.0
    visualization_ms: float | None = None
    encode_ms: float | None = None
    save_ms: float | None = None
    frame_total_ms: float | None = None
    visualisation_ms: float = 0.0
    encode_write_ms: float = 0.0
    runtime_revision: int | None = None
    detection_context: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


ProgressCallback = Callable[[dict[str, Any]], None]
CancelCallback = Callable[[], bool]
PreviewCallback = Callable[[np.ndarray, FrameResult], None]
PauseCallback = Callable[[], bool]
PausedSecondsCallback = Callable[[], float]
