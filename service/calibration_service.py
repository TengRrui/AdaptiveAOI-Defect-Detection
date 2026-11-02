from __future__ import annotations

from pathlib import Path
from typing import Any

from bridge.api_client import client
from bridge.algorithm_bridge import log_software_error


def _model_id(config: str | Path) -> str:
    name = Path(str(config)).stem.lower()
    return {"mvtec": "patchcore-mvtec", "visa": "patchcore-visa", "dagm": "patchcore-dagm"}.get(name, "patchcore-default")


def _fail(message: str, **extra: Any) -> dict[str, Any]:
    log_software_error("calibration_service", message)
    return {"ok": False, "error": message, **extra}


def run_calibration(
    dataset: str,
    category: str,
    config: str,
    normal_shot: int,
    defect_shot: int,
    seed: int,
    use_defect_direction: bool,
    execute: bool = True,
) -> dict[str, Any]:
    """Create an immutable calibration job through the integration API."""
    del execute
    payload = client().calibrate(
        {
            "model_id": _model_id(config),
            "dataset": dataset,
            "category": category,
            "normal_shot": int(normal_shot),
            "defect_shot": int(defect_shot),
            "seed": int(seed),
            "use_defect_direction": bool(use_defect_direction),
        }
    )
    if not payload.get("success"):
        error = payload.get("error") or {}
        return _fail(error.get("message", "Calibration failed."), api_error=error, request_id=payload.get("request_id"))
    result = payload.get("calibration") or {}
    raw = result.get("calibration") or {}
    return {
        "ok": True,
        "request_id": payload.get("request_id"),
        "job_id": result.get("job_id"),
        "dataset": result.get("dataset", dataset),
        "category": result.get("category", category),
        "calibration_path": result.get("calibration_path"),
        "raw_calibration": raw,
        "summary": {
            "image_threshold": raw.get("image_threshold"),
            "pixel_threshold": raw.get("pixel_threshold"),
            "threshold_strategy": raw.get("threshold_strategy"),
            "use_defect_direction": raw.get("use_defect_direction"),
            "alpha": raw.get("alpha"),
            "calibration_version": raw.get("calibration_version") or result.get("job_id"),
            "normal_shot": raw.get("normal_shot", normal_shot),
            "defect_shot": raw.get("defect_shot", defect_shot),
        },
        "active": bool(result.get("active", False)),
        "bridge": {"mode": "integration_api", "request_id": payload.get("request_id"), "job_id": result.get("job_id")},
    }
