"""Import and compatibility checks for the shared heatmap overlay component."""

from __future__ import annotations

import importlib
from io import BytesIO
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "app"


def _ui_common():
    # Streamlit pages import ``ui_common`` as a top-level module because the
    # app directory is on the Streamlit script path.  Mirror that launch
    # topology in the import-level tests.
    for path in (str(PROJECT_ROOT), str(APP_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return importlib.import_module("ui_common")


def test_display_heatmap_overlay_is_public_import() -> None:
    ui_common = _ui_common()
    from ui_common import display_heatmap_overlay

    assert callable(display_heatmap_overlay)
    assert ui_common.display_heatmap_overlay is display_heatmap_overlay


def test_legacy_original_and_heatmap_inputs_keep_red_green_overlay() -> None:
    ui_common = _ui_common()
    original = Image.fromarray(np.full((8, 8, 3), 220, dtype=np.uint8), mode="RGB")
    heatmap = np.zeros((8, 8), dtype=np.uint8)
    heatmap[:, 4:] = 255

    rendered = ui_common.make_overlay(original, heatmap_u8=heatmap, alpha=0.45)

    assert rendered.shape == (8, 8, 3)
    # Low response remains green-biased and high response red-biased after
    # blending, proving that the fallback did not regress to JET/pure heatmap.
    low = rendered[:, 1].mean(axis=0)
    high = rendered[:, 6].mean(axis=0)
    assert low[1] > low[0]
    assert high[0] > high[1]


def test_visual_input_normalization_accepts_pil_path_numpy_and_bytes(tmp_path: Path) -> None:
    ui_common = _ui_common()
    image = Image.fromarray(np.full((4, 5, 3), 120, dtype=np.uint8), mode="RGB")
    path = tmp_path / "sample.png"
    image.save(path)
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    for value in (image, path, np.asarray(image), buffer.getvalue()):
        loaded = ui_common._load_visual_image(value)
        assert loaded is not None
        assert loaded.size == (5, 4)


def test_detection_pages_render_without_import_error() -> None:
    from streamlit.testing.v1 import AppTest

    _ui_common()
    for page_name in ("1_单图检测.py", "2_批量检测.py", "9_视频文件检测.py"):
        page = APP_ROOT / "pages" / page_name
        app = AppTest.from_file(str(page)).run(timeout=120)
        assert not app.exception, [str(item.value) for item in app.exception]
