from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


VISUAL_KEYS = {
    "heatmap": ["heatmap_path", "heatmap", "heatmap_file"],
    "mask": ["mask_path", "mask", "mask_file"],
    "overlay": ["overlay_path", "overlay", "overlay_file"],
}


def _missing(path: Path, kind: str) -> dict[str, Any]:
    return {"ok": False, "kind": kind, "path": str(path), "error": f"{kind} file not found: {path}"}


def _read_json(path: Path, kind: str) -> dict[str, Any]:
    if not path.exists():
        return _missing(path, kind)
    try:
        return {"ok": True, "kind": kind, "path": str(path), "data": json.loads(path.read_text(encoding="utf-8"))}
    except Exception as exc:
        return {"ok": False, "kind": kind, "path": str(path), "error": f"Failed to read {kind}: {type(exc).__name__}: {exc}"}


def _read_csv(path: Path, kind: str) -> dict[str, Any]:
    if not path.exists():
        return _missing(path, kind)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        return {"ok": True, "kind": kind, "path": str(path), "rows": rows, "row_count": len(rows)}
    except Exception as exc:
        return {"ok": False, "kind": kind, "path": str(path), "error": f"Failed to read {kind}: {type(exc).__name__}: {exc}"}


def _resolve_path(path_value: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(path_value)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def _first_existing(paths: list[Path]) -> str | None:
    for path in paths:
        if path.exists():
            return str(path)
    return None


def find_visual_outputs(base_dir: str | Path, image_stem: str | None = None) -> dict[str, str | None]:
    base = Path(base_dir).resolve()
    if not base.exists():
        return {"heatmap": None, "mask": None, "overlay": None}

    def candidates(label: str) -> list[Path]:
        patterns = []
        if image_stem:
            patterns.extend([f"*{image_stem}*{label}*.png", f"*{label}*{image_stem}*.png"])
        patterns.extend([f"*{label}*.png", f"*{label}*.jpg", f"*{label}*.jpeg"])
        found: list[Path] = []
        for pattern in patterns:
            found.extend(sorted(base.rglob(pattern)))
        return found

    return {
        "heatmap": _first_existing(candidates("heatmap")),
        "mask": _first_existing(candidates("mask")),
        "overlay": _first_existing(candidates("overlay")),
    }


def _visuals_from_payload(payload: dict[str, Any], base_dir: Path) -> dict[str, str | None]:
    visuals: dict[str, str | None] = {}
    for visual_name, keys in VISUAL_KEYS.items():
        value = None
        for key in keys:
            if payload.get(key):
                candidate = _resolve_path(str(payload[key]), base_dir)
                value = str(candidate) if candidate.exists() else str(candidate)
                break
        visuals[visual_name] = value
    if not any(visuals.values()):
        stem = Path(str(payload.get("image_path", ""))).stem or None
        visuals = find_visual_outputs(base_dir, stem)
    return visuals


def read_single_result(result_json: str | Path) -> dict[str, Any]:
    path = Path(result_json).resolve()
    result = _read_json(path, "single_result")
    if result["ok"]:
        result["visuals"] = _visuals_from_payload(result["data"], path.parent)
    return result


def read_batch_result(result_csv: str | Path) -> dict[str, Any]:
    return _read_csv(Path(result_csv).resolve(), "batch_result")


def read_calibration(calibration_json: str | Path) -> dict[str, Any]:
    return _read_json(Path(calibration_json).resolve(), "calibration")


def read_speed_report(speed_report_csv: str | Path) -> dict[str, Any]:
    return _read_csv(Path(speed_report_csv).resolve(), "speed_report")


def read_speed_summary(speed_summary_json: str | Path) -> dict[str, Any]:
    return _read_json(Path(speed_summary_json).resolve(), "speed_summary")


def load_speed_outputs(output_dir: str | Path) -> dict[str, Any]:
    base = Path(output_dir).resolve()
    if not base.exists():
        return {"ok": False, "path": str(base), "error": f"Speed output directory not found: {base}"}
    csv_candidates = sorted(base.glob("speed_report.csv")) + sorted(base.glob("speed_*.csv"))
    summary_candidates = sorted(base.glob("speed_summary.json"))
    return {
        "ok": True,
        "path": str(base),
        "speed_report": read_speed_report(csv_candidates[0]) if csv_candidates else {"ok": False, "error": "No speed_report.csv or speed_*.csv found."},
        "speed_summary": read_speed_summary(summary_candidates[0]) if summary_candidates else {"ok": False, "error": "No speed_summary.json found."},
    }
