from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

from backend.services.detection_context import DetectionContext, detection_context_dict


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return value


def artifact_url(path: str | Path | None, jobs_root: Path) -> str | None:
    if not path:
        return None
    target = Path(path).resolve()
    try:
        relative = target.relative_to(jobs_root.resolve())
    except ValueError:
        return None
    return "/artifacts/" + quote(relative.as_posix())


def structured_timing(
    core_result: dict[str, Any],
    *,
    model_load_ms: float | None,
    memory_bank_load_ms: float | None,
    queue_wait_ms: float | None,
    extra_save_ms: float,
    end_to_end_ms: float,
    generate_visualizations: bool,
) -> dict[str, Any]:
    """Map the core's CUDA-synchronised stage timings without inventing stages."""
    input_read_ms = sum(float(core_result.get(key, 0.0) or 0.0) for key in ("file_read_ms", "decode_ms", "color_convert_ms"))
    visualization_ms = float(core_result.get("visualization_ms", 0.0) or 0.0) if generate_visualizations else None
    encode_ms = float(core_result.get("encode_ms", 0.0) or 0.0) if generate_visualizations else None
    return {
        "model_load_ms": model_load_ms,
        "memory_bank_load_ms": memory_bank_load_ms,
        "input_read_ms": input_read_ms,
        "preprocess_ms": float(core_result.get("preprocess_ms", 0.0) or 0.0),
        "feature_extraction_ms": float(core_result.get("feature_ms", 0.0) or 0.0),
        "ann_search_ms": float(core_result.get("memory_score_ms", 0.0) or 0.0),
        "anomaly_scoring_ms": (
            float(core_result.get("defect_score_ms", 0.0) or 0.0)
            + float(core_result.get("fusion_ms", 0.0) or 0.0)
            + float(core_result.get("image_score_ms", 0.0) or 0.0)
        ),
        "inference_ms": float(core_result.get("detection_only_ms", core_result.get("inference_time_ms", 0.0)) or 0.0),
        "postprocess_ms": float(core_result.get("heatmap_resize_ms", 0.0) or 0.0),
        "visualization_ms": visualization_ms,
        "encode_ms": encode_ms,
        "save_ms": float(core_result.get("disk_write_ms", 0.0) or 0.0) + float(extra_save_ms),
        "queue_wait_ms": queue_wait_ms,
        "end_to_end_ms": float(end_to_end_ms),
        "model_reused": model_load_ms is None,
        "memory_bank_reused": memory_bank_load_ms is None,
        "inference_includes_feature_and_scoring": True,
    }


def normalized_detection(
    core_result: dict[str, Any],
    model_id: str,
    device: str,
    jobs_root: Path,
    original_path: Path | None,
    result_json_path: Path,
    detection_context: DetectionContext | dict[str, Any] | None = None,
) -> dict[str, Any]:
    prediction = "anomaly" if int(core_result.get("pred_label", 0)) == 1 else "normal"
    source_path = Path(str(core_result.get("image_path", "")))
    result_image_url = artifact_url(core_result.get("overlay_path"), jobs_root)
    marked_image_url = artifact_url(core_result.get("marked_image_path"), jobs_root)
    heatmap_url = artifact_url(core_result.get("heatmap_path"), jobs_root)
    mask_url = artifact_url(core_result.get("mask_path"), jobs_root)
    visualization_artifacts = [
        {"kind": kind, "url": url}
        for kind, url in (
            ("overlay", result_image_url),
            ("marked_image", marked_image_url),
            ("heatmap", heatmap_url),
            ("mask", mask_url),
        )
        if url
    ]
    frozen_context = detection_context_dict(detection_context)
    metadata = {
        "dataset": core_result.get("dataset"),
        "category": core_result.get("category"),
        "filename": source_path.name,
        "source_path": str(source_path),
        "memory_backend": core_result.get("memory_backend"),
        "precision_mode": core_result.get("precision_mode"),
        "tile_count": core_result.get("tile_count"),
        "use_tiling": core_result.get("use_tiling"),
        "result_json_url": artifact_url(result_json_path, jobs_root),
        "core_result": json_safe({key: value for key, value in core_result.items() if not str(key).startswith("_")}),
        "visualization_artifacts": visualization_artifacts,
        "marked_image_url": marked_image_url,
    }
    if frozen_context is not None:
        metadata["runtime_revision"] = int(frozen_context["runtime_revision"])
        metadata["detection_context"] = frozen_context
    result = {
        "prediction": prediction,
        "is_anomaly": prediction == "anomaly",
        "anomaly_score": float(core_result.get("score", core_result.get("image_score", 0.0))),
        "baseline_score": float(core_result.get("baseline_score", core_result.get("score", 0.0))),
        "calibrated_score": float(core_result.get("calibrated_score", core_result.get("score", 0.0))),
        "final_score": float(core_result.get("final_score", core_result.get("score", 0.0))),
        "calibration_applied": bool(core_result.get("calibration_applied", False)),
        "gate_status": str(core_result.get("gate_status") or "not_applicable"),
        "fallback_used": bool(core_result.get("fallback_used", False)),
        "calibration_version": str(core_result.get("calibration_version") or ""),
        "method_mode": str(core_result.get("method_mode") or "normal_memory"),
        "threshold": float(core_result.get("threshold", 0.0)),
        "defect_type": core_result.get("defect_type") or None,
        "inference_time_ms": float(core_result.get("inference_time_ms", core_result.get("total_ms", 0.0))),
        "generate_visualizations": bool(core_result.get("generate_visualizations", False)),
        "visualization_status": str(core_result.get("visualization_status") or "not_generated"),
        "visualization_artifacts": visualization_artifacts,
        "timing": json_safe(core_result.get("timing") or {}),
        "timing_mode": str(core_result.get("timing_mode") or "strict_total"),
        "model_id": model_id,
        "device": device,
        "original_image_url": artifact_url(original_path, jobs_root),
        "result_image_url": result_image_url,
        "marked_image_url": marked_image_url,
        "heatmap_url": heatmap_url,
        "mask_url": mask_url,
        "marked_region_count": int(core_result.get("marked_region_count", 0) or 0),
        "regions": [],
        "metadata": metadata,
    }
    if frozen_context is not None:
        result["runtime_revision"] = int(frozen_context["runtime_revision"])
        result["detection_context"] = frozen_context
    return result
