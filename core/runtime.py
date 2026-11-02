from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

import torch
from PIL import Image


def select_device(requested: str = "cuda") -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class StageTimer:
    def __init__(self, device: torch.device):
        self.device = device
        self.times = {}

    @contextmanager
    def track(self, name: str):
        sync_if_cuda(self.device)
        start = time.perf_counter()
        yield
        sync_if_cuda(self.device)
        self.add(name, (time.perf_counter() - start) * 1000.0)

    def add(self, name: str, elapsed_ms: float):
        self.times[name] = float(self.times.get(name, 0.0) + elapsed_ms)

    def total_ms(self) -> float:
        return float(sum(self.times.values()))

    def get(self, name: str) -> float:
        return float(self.times.get(name, 0.0))

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.times.items()}


RUNTIME_FIELDS = [
    "image_read_ms",
    "file_read_ms",
    "decode_ms",
    "color_convert_ms",
    "preprocess_ms",
    "resize_ms",
    "tensor_convert_ms",
    "queue_wait_ms",
    "tile_schedule_ms",
    "host_to_device_ms",
    "feature_ms",
    "device_to_host_ms",
    "memory_score_ms",
    "defect_score_ms",
    "fusion_ms",
    "image_score_ms",
    "heatmap_resize_ms",
    "heatmap_normalize_ms",
    "mask_ms",
    "overlay_ms",
    "visualization_ms",
    "encode_ms",
    "disk_write_ms",
    "save_queue_wait_ms",
    "postprocess_ms",
    "save_output_ms",
    "total_ms",
]


def timing_totals(times: dict[str, float], timing_mode: str = "strict_total") -> dict[str, float | str]:
    image_read = float(times.get("image_read_ms", 0.0))
    decode = float(times.get("decode_ms", 0.0))
    color_convert = float(times.get("color_convert_ms", 0.0))
    preprocess = float(times.get("preprocess_ms", 0.0))
    queue_wait = float(times.get("queue_wait_ms", 0.0))
    tile_schedule = float(times.get("tile_schedule_ms", 0.0))
    host_to_device = float(times.get("host_to_device_ms", 0.0))
    feature = float(times.get("feature_ms", 0.0))
    device_to_host = float(times.get("device_to_host_ms", 0.0))
    memory = float(times.get("memory_score_ms", 0.0))
    defect = float(times.get("defect_score_ms", 0.0))
    fusion = float(times.get("fusion_ms", 0.0))
    image_score_elapsed = float(times.get("image_score_ms", 0.0))
    heatmap_resize = float(times.get("heatmap_resize_ms", 0.0))
    heatmap_normalize = float(times.get("heatmap_normalize_ms", 0.0))
    mask = float(times.get("mask_ms", 0.0))
    overlay = float(times.get("overlay_ms", 0.0))
    encode = float(times.get("encode_ms", 0.0))
    save_output = float(times.get("save_output_ms", 0.0))
    strict_total = (
        image_read
        + decode
        + color_convert
        + queue_wait
        + preprocess
        + tile_schedule
        + host_to_device
        + feature
        + device_to_host
        + memory
        + defect
        + fusion
        + image_score_elapsed
        + heatmap_resize
        + heatmap_normalize
        + mask
        + overlay
        + encode
        + save_output
    )
    no_save_total = strict_total - encode - save_output
    online_total = (
        preprocess
        + tile_schedule
        + host_to_device
        + feature
        + device_to_host
        + memory
        + defect
        + fusion
        + image_score_elapsed
        + heatmap_resize
        + heatmap_normalize
        + mask
    )
    detection_only = tile_schedule + host_to_device + feature + device_to_host + memory + defect + fusion + image_score_elapsed
    mode = str(timing_mode or "strict_total").lower()
    selected = {
        "strict_total": strict_total,
        "online_total": online_total,
        "detection_only": detection_only,
        "no_save_total": no_save_total,
    }.get(mode, strict_total)
    return {
        "timing_mode": mode,
        "strict_total_ms": float(strict_total),
        "online_total_ms": float(online_total),
        "detection_only_ms": float(detection_only),
        "no_save_total_ms": float(no_save_total),
        "total_ms": float(selected),
    }


def peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))


def reset_peak_memory(device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def load_pil_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")
