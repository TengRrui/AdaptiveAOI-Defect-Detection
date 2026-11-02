from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image


def fit_canvas_size(
    source_width: int,
    source_height: int,
    *,
    max_width: int = 760,
    max_height: int = 520,
) -> tuple[int, int]:
    """Fit a source image inside the editor without letterboxing or distortion."""
    width = max(1, int(source_width))
    height = max(1, int(source_height))
    scale = min(1.0, float(max_width) / width, float(max_height) / height)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def binary_mask(image: Image.Image | np.ndarray | None, size: tuple[int, int]) -> Image.Image:
    """Return an exact-size, single-channel binary mask."""
    if image is None:
        return Image.new("L", size, 0)
    value = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
    mask = value.convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return mask.point(lambda pixel: 255 if pixel >= 128 else 0, mode="L")


def compose_editor_background(
    source: Image.Image | np.ndarray,
    *,
    source_opacity: float = 1.0,
    mask: Image.Image | np.ndarray | None = None,
    mask_opacity: float = 0.45,
) -> Image.Image:
    """Compose the immutable source and reviewed-mask layers for display only."""
    original = source if isinstance(source, Image.Image) else Image.fromarray(np.asarray(source))
    original = original.convert("RGB")
    source_alpha = min(1.0, max(0.2, float(source_opacity)))
    if source_alpha < 1.0:
        original = Image.blend(Image.new("RGB", original.size, (238, 244, 248)), original, source_alpha)
    if mask is None:
        return original

    reviewed = binary_mask(mask, original.size)
    opacity = int(round(255 * min(0.9, max(0.1, float(mask_opacity)))))
    alpha = reviewed.point(lambda pixel: opacity if pixel else 0, mode="L")
    overlay = Image.new("RGBA", original.size, (255, 64, 96, 0))
    overlay.putalpha(alpha)
    return Image.alpha_composite(original.convert("RGBA"), overlay).convert("RGB")


def canvas_drawing_to_source_mask(
    canvas_rgba: Any,
    *,
    source_size: tuple[int, int],
    initial_mask: Image.Image | np.ndarray | None = None,
) -> Image.Image:
    """Map the transparent drawing layer back to source-image coordinates.

    ``streamlit-drawable-canvas`` does not return its background image in
    ``image_data``.  Transparent pixels are therefore untouched, cyan/red
    pixels add Mask, and white pixels (the eraser tool) remove Mask.  The
    display canvas has the exact source aspect ratio, so a nearest-neighbour
    resize is the complete inverse coordinate mapping and introduces no
    letterbox/padding offset.
    """
    width, height = max(1, int(source_size[0])), max(1, int(source_size[1]))
    result = np.asarray(binary_mask(initial_mask, (width, height)), dtype=np.uint8).copy()
    if canvas_rgba is None:
        return Image.fromarray(result, mode="L")

    pixels = np.asarray(canvas_rgba).astype(np.uint8, copy=False)
    if pixels.ndim != 3 or pixels.shape[2] < 4:
        raise ValueError("画布数据必须是 RGBA 图层。")
    active = pixels[:, :, 3] >= 12
    white = (pixels[:, :, 0] >= 225) & (pixels[:, :, 1] >= 225) & (pixels[:, :, 2] >= 225)
    add_display = (active & ~white).astype(np.uint8) * 255
    erase_display = (active & white).astype(np.uint8) * 255
    add_source = np.asarray(
        Image.fromarray(add_display, mode="L").resize((width, height), Image.Resampling.NEAREST)
    ) >= 128
    erase_source = np.asarray(
        Image.fromarray(erase_display, mode="L").resize((width, height), Image.Resampling.NEAREST)
    ) >= 128
    result[erase_source] = 0
    result[add_source] = 255
    return Image.fromarray(result, mode="L")


def mask_png_bytes(mask: Image.Image | np.ndarray) -> bytes:
    output = BytesIO()
    value = mask if isinstance(mask, Image.Image) else Image.fromarray(np.asarray(mask))
    value.convert("L").point(lambda pixel: 255 if pixel >= 128 else 0, mode="L").save(
        output,
        format="PNG",
        optimize=True,
    )
    return output.getvalue()
