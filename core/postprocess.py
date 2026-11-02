from __future__ import annotations

import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# Build a RGB lookup table once.  The overlay is intentionally an industrial
# alert ramp rather than JET: low anomaly response is green, the middle is
# yellow/orange, and high response is red.  Keeping the LUT here means single
# image and batch/video artifacts use the same visual language without changing
# the anomaly score or threshold calculation.
_RED_GREEN_STOPS = np.asarray(
    [
        (34, 139, 86),    # low: deep green
        (82, 180, 88),    # low-mid: green
        (238, 220, 66),   # middle: yellow
        (246, 148, 42),   # high-mid: orange
        (211, 38, 48),     # high: red
    ],
    dtype=np.float32,
)
_RED_GREEN_POSITIONS = np.asarray([0, 64, 128, 192, 255], dtype=np.float32)
_RED_GREEN_RGB_LUT = np.column_stack(
    [
        np.interp(np.arange(256, dtype=np.float32), _RED_GREEN_POSITIONS, _RED_GREEN_STOPS[:, channel])
        for channel in range(3)
    ]
).round().astype(np.uint8)


def resize_heatmap(score_map: np.ndarray, height: int, width: int) -> np.ndarray:
    source = np.asarray(score_map, dtype=np.float32)
    heatmap = cv2.resize(source, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
    return np.asarray(heatmap, dtype=np.float32)


def binary_mask(heatmap: np.ndarray, threshold: float) -> np.ndarray:
    """Threshold directly into uint8, without an intermediate full bool array."""
    mask = np.empty(np.asarray(heatmap).shape, dtype=np.uint8)
    np.greater_equal(heatmap, float(threshold), out=mask)
    mask *= 255
    return mask


def normalize_uint8(heatmap: np.ndarray) -> np.ndarray:
    arr = np.asarray(heatmap, dtype=np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    # Keep the legacy normalization formula so visual PNG values remain stable.
    return ((arr - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)


def make_overlay(
    image_source: str | Path | Image.Image | np.ndarray,
    heatmap: np.ndarray | None = None,
    alpha: float = 0.45,
    heatmap_u8: np.ndarray | None = None,
) -> np.ndarray:
    """Make a RGB uint8 overlay, reusing a supplied decoded image/heatmap array."""
    if isinstance(image_source, np.ndarray):
        image = image_source.astype(np.uint8, copy=False)
    elif isinstance(image_source, Image.Image):
        image = np.asarray(image_source.convert("RGB"))
    else:
        with Image.open(image_source) as source:
            image = np.asarray(source.convert("RGB"))
    if heatmap_u8 is None:
        if heatmap is None:
            raise ValueError("heatmap or heatmap_u8 is required to create an overlay")
        heatmap_u8 = normalize_uint8(heatmap)
    color = _RED_GREEN_RGB_LUT[np.asarray(heatmap_u8, dtype=np.uint8)]
    if color.shape[:2] != image.shape[:2]:
        color = cv2.resize(color, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    blend_alpha = min(1.0, max(0.0, float(alpha)))
    return cv2.addWeighted(image, 1.0 - blend_alpha, color, blend_alpha, 0)


def make_marked_image(
    image_source: str | Path | Image.Image | np.ndarray,
    mask: np.ndarray | None,
    *,
    enabled: bool = True,
    min_region_area: int = 1,
    thickness: int = 4,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Draw display-only red boxes from the final anomaly mask.

    This deliberately does not make a second anomaly decision.  The caller
    supplies the already thresholded final mask and controls whether the
    image-level prediction is anomalous.  A normal sample therefore returns
    the untouched original image and no regions, while an anomalous sample
    gets one box per connected mask component.
    """
    if isinstance(image_source, np.ndarray):
        image = np.asarray(image_source).astype(np.uint8, copy=True)
    elif isinstance(image_source, Image.Image):
        image = np.asarray(image_source.convert("RGB")).astype(np.uint8, copy=True)
    else:
        with Image.open(image_source) as source:
            image = np.asarray(source.convert("RGB")).astype(np.uint8, copy=True)

    if not enabled or mask is None:
        return image, []
    mask_array = np.asarray(mask)
    if mask_array.ndim == 3:
        mask_array = mask_array[..., 0]
    if mask_array.ndim != 2 or not mask_array.size:
        return image, []
    if mask_array.shape != image.shape[:2]:
        mask_array = cv2.resize(
            mask_array.astype(np.uint8, copy=False),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    binary = (mask_array > 0).astype(np.uint8)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    minimum = max(1, int(min_region_area))
    boxes: list[tuple[int, int, int, int]] = []
    canvas = image.copy()
    for index in range(1, int(count)):
        x, y, width, height, area = (int(value) for value in stats[index])
        if area < minimum or width <= 0 or height <= 0:
            continue
        boxes.append((x, y, width, height))
        line_width = max(2, min(int(thickness), max(2, min(width, height))))
        cv2.rectangle(canvas, (x, y), (x + width - 1, y + height - 1), (255, 0, 0), line_width)
    return canvas, boxes


def array_profile(name: str, array: np.ndarray, created_at: str, reusable: bool) -> dict:
    value = np.asarray(array)
    return {
        "array_name": name,
        "shape": "x".join(str(part) for part in value.shape),
        "dtype": str(value.dtype),
        "bytes": int(value.nbytes),
        "megabytes": float(value.nbytes / (1024 ** 2)),
        "created_at": created_at,
        "reusable": bool(reusable),
    }


@dataclass
class SaveOutcome:
    paths: dict[str, str] = field(default_factory=dict)
    encode_ms: float = 0.0
    disk_write_ms: float = 0.0
    queue_wait_ms: float = 0.0
    errors: list[str] = field(default_factory=list)


@dataclass
class SaveTicket:
    submitted_at: float
    future: Future | None = None
    outcome: SaveOutcome | None = None


def _encode_png(array: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _save_arrays(submitted_at: float, payloads: dict[str, tuple[Path, np.ndarray]]) -> SaveOutcome:
    outcome = SaveOutcome(queue_wait_ms=float((time.perf_counter() - submitted_at) * 1000.0))
    for name, (path, array) in payloads.items():
        try:
            started = time.perf_counter()
            encoded = _encode_png(array)
            outcome.encode_ms += (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            _atomic_write(path, encoded)
            outcome.disk_write_ms += (time.perf_counter() - started) * 1000.0
            outcome.paths[name] = str(path)
        except Exception as exc:
            outcome.errors.append(f"{name} -> {path}: {type(exc).__name__}: {exc}")
    return outcome


class VisualSaveQueue:
    """Bounded thread-backed PNG encoder/writer with explicit result resolution."""

    def __init__(self, enabled: bool = False, queue_size: int = 2):
        self.enabled = bool(enabled)
        self.queue_size = max(1, int(queue_size))
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="AdaptiveAOIVisualSave") if self.enabled else None
        self.tickets: list[SaveTicket] = []

    def submit(self, payloads: dict[str, tuple[Path, np.ndarray]]) -> SaveTicket | None:
        if not payloads:
            return None
        ticket = SaveTicket(submitted_at=time.perf_counter())
        if self.executor is None:
            ticket.outcome = _save_arrays(ticket.submitted_at, payloads)
        else:
            # Backpressure is intentional: it prevents 2500x2500 arrays from
            # accumulating unboundedly while preserving background overlap.
            while len([item for item in self.tickets if item.future is not None and not item.future.done()]) >= self.queue_size:
                self.tickets[0].future.result()
                self.tickets.pop(0)
            ticket.future = self.executor.submit(_save_arrays, ticket.submitted_at, payloads)
            self.tickets.append(ticket)
        return ticket

    def resolve(self, ticket: SaveTicket | None) -> SaveOutcome:
        if ticket is None:
            return SaveOutcome()
        if ticket.outcome is None:
            try:
                ticket.outcome = ticket.future.result()
            except Exception as exc:
                ticket.outcome = SaveOutcome(errors=[f"visual save worker failed: {type(exc).__name__}: {exc}"])
        return ticket.outcome

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=False)
            self.executor = None


def save_heatmap(path: str | Path, heatmap: np.ndarray) -> Path:
    _atomic_write(Path(path), _encode_png(normalize_uint8(heatmap)))
    return Path(path)


def save_mask(path: str | Path, mask: np.ndarray) -> Path:
    _atomic_write(Path(path), _encode_png(mask.astype(np.uint8, copy=False)))
    return Path(path)


def save_overlay(path: str | Path, overlay: np.ndarray) -> Path:
    _atomic_write(Path(path), _encode_png(overlay.astype(np.uint8, copy=False)))
    return Path(path)
