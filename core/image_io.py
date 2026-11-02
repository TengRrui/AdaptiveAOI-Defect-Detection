"""Image preparation and ordered background prefetch for inference.

The module keeps Pillow as the canonical decoder so RGB conversion and resize
semantics remain aligned with the existing torchvision/PIL inference path.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Iterator

import torch
from PIL import Image, ImageOps
from torchvision.transforms import InterpolationMode, functional as tvf


@dataclass
class PreparedImage:
    image_path: Path
    image: Image.Image
    width: int
    height: int
    file_read_ms: float
    decode_ms: float
    color_convert_ms: float
    queue_wait_ms: float = 0.0


def prepare_image(image_path: str | Path) -> PreparedImage:
    """Read, decode, orient, and RGB-convert one image exactly once."""
    path = Path(image_path)
    try:
        started = time.perf_counter()
        raw = path.read_bytes()
        file_read_ms = (time.perf_counter() - started) * 1000.0
    except Exception as exc:
        raise RuntimeError(f"Failed to read image '{path}': {type(exc).__name__}: {exc}") from exc
    try:
        started = time.perf_counter()
        with Image.open(BytesIO(raw)) as decoded:
            decoded.load()
            oriented = ImageOps.exif_transpose(decoded)
            width, height = oriented.size
            decode_ms = (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            image = oriented.convert("RGB")
            color_convert_ms = (time.perf_counter() - started) * 1000.0
    except Exception as exc:
        raise RuntimeError(f"Failed to decode image '{path}': {type(exc).__name__}: {exc}") from exc
    return PreparedImage(path, image, int(width), int(height), float(file_read_ms), float(decode_ms), float(color_convert_ms))


def preprocess_rgb(image: Image.Image, image_size: int, pin_memory: bool = False) -> tuple[torch.Tensor, dict[str, float | bool]]:
    """Apply the existing PIL bilinear resize, ToTensor, and ImageNet normalize steps."""
    started = time.perf_counter()
    resized = tvf.resize(image, [int(image_size), int(image_size)], interpolation=InterpolationMode.BILINEAR, antialias=True)
    resize_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    tensor = tvf.to_tensor(resized)
    tensor = tvf.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).contiguous()
    pinned = False
    if pin_memory and torch.cuda.is_available():
        try:
            tensor = tensor.pin_memory()
            pinned = True
        except RuntimeError:
            # Pinned memory is an optional transfer optimisation; inference must
            # remain usable on CPU-only or constrained Windows environments.
            pinned = False
    tensor_convert_ms = (time.perf_counter() - started) * 1000.0
    return tensor, {
        "resize_ms": float(resize_ms),
        "tensor_convert_ms": float(tensor_convert_ms),
        "pinned_memory": bool(pinned),
    }


class OrderedImagePrefetcher:
    """One-worker ordered prefetcher for Windows-safe read/decode overlap."""

    def __init__(self, image_paths: Iterable[str | Path], enabled: bool = False, queue_size: int = 2):
        self.paths = [Path(path) for path in image_paths]
        self.enabled = bool(enabled)
        self.queue_size = max(1, int(queue_size))

    def __iter__(self) -> Iterator[PreparedImage]:
        if not self.enabled:
            for path in self.paths:
                yield prepare_image(path)
            return
        # A single background worker gives deterministic path ordering, avoids
        # process-spawn issues on Windows, and can decode the next image while the
        # foreground thread waits for CUDA feature extraction/scoring.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="AdaptiveAOIImageIO") as executor:
            futures: dict[int, Future[PreparedImage]] = {}
            next_submit = 0

            def schedule() -> None:
                nonlocal next_submit
                while next_submit < len(self.paths) and len(futures) < self.queue_size:
                    futures[next_submit] = executor.submit(prepare_image, self.paths[next_submit])
                    next_submit += 1

            schedule()
            for index in range(len(self.paths)):
                future = futures.pop(index)
                started = time.perf_counter()
                prepared = future.result()
                prepared.queue_wait_ms = float((time.perf_counter() - started) * 1000.0)
                schedule()
                yield prepared


def iter_prepared_images(image_paths: Iterable[str | Path], enabled: bool = False, queue_size: int = 2) -> Iterator[PreparedImage]:
    return iter(OrderedImagePrefetcher(image_paths, enabled=enabled, queue_size=queue_size))
