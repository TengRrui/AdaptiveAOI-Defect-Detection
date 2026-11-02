from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from bridge.api_client import client
from bridge.algorithm_bridge import log_software_error
from service.runtime_config_service import load_runtime_context
from service.status_service import LATEST_INFERENCE_PATH, save_latest_inference


def _fail(message: str, **extra: Any) -> dict[str, Any]:
    log_software_error("inference_service", message)
    return {"ok": False, "error": message, **extra}


class _LocalUpload:
    def __init__(self, path: Path):
        self.name = path.name
        self.size = path.stat().st_size
        self._handle = path.open("rb")

    def read(self, size: int = -1) -> bytes:
        return self._handle.read(size)

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._handle.seek(offset, whence)

    def tell(self) -> int:
        return self._handle.tell()


def uploaded_input_hash(uploaded_file: Any | None) -> str:
    if uploaded_file is None:
        return ""
    if not hasattr(uploaded_file, "read"):
        return hashlib.sha256(bytes(uploaded_file)).hexdigest()
    current = uploaded_file.tell() if hasattr(uploaded_file, "tell") else None
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)
    digest = hashlib.sha256()
    for chunk in iter(lambda: uploaded_file.read(1024 * 1024), b""):
        digest.update(chunk)
    if current is not None and hasattr(uploaded_file, "seek"):
        uploaded_file.seek(current)
    return digest.hexdigest()


def result_matches_input(result: dict[str, Any] | None, input_hash: str) -> bool:
    return bool(result and input_hash and str(result.get("input_hash") or "") == str(input_hash))


def run_image_inference(
    dataset: str,
    category: str,
    config: str,
    uploaded_file: Any | None = None,
    image_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    execute: bool = True,
    device: str = "auto",
    memory_bank_id: str | None = None,
    generate_visualizations: bool = True,
) -> dict[str, Any]:
    """Call the typed backend API; no Streamlit service invokes experiment scripts directly."""
    del output_dir, execute
    if uploaded_file is None and image_path is None:
        return _fail("请上传图像文件，或提供 image_path。")
    if uploaded_file is None:
        path = Path(image_path).expanduser().resolve()
        if not path.exists():
            return _fail(f"图像文件不存在：{path}")
        uploaded_file = _LocalUpload(path)

    runtime_state = load_runtime_context()
    if not runtime_state.get("ok"):
        return _fail(
            str(runtime_state.get("error") or "当前正式运行配置尚未就绪。"),
            api_error=runtime_state.get("api_error") or {},
        )
    context = dict(runtime_state.get("context") or {})
    runtime = {
        "runtime_context": context,
        "generate_visualizations": bool(generate_visualizations),
    }
    payload = client().detect_upload(uploaded_file, runtime)
    if not payload.get("success"):
        error = payload.get("error") or {}
        if str(error.get("code") or "") == "INFERENCE_RUNTIME_BINDING_MISMATCH":
            latest = load_runtime_context()
            return _fail(
                "运行配置刚刚发生变化，已同步最新配置，请重新开始检测。",
                api_error=error,
                runtime_context=latest.get("context") or {},
                request_id=payload.get("request_id"),
            )
        return _fail(error.get("message", "算法服务检测失败，请检查后端日志和模型资源。"), api_error=error, request_id=payload.get("request_id"))

    result = payload.get("result") or {}
    metadata = result.get("metadata") or {}
    api = client()
    mapped = {
        "ok": True,
        "request_id": payload.get("request_id"),
        "job_id": payload.get("job_id"),
        "task_id": result.get("task_id") or metadata.get("task_id") or payload.get("job_id"),
        "result_id": result.get("result_id"),
        "input_hash": result.get("input_hash") or metadata.get("input_hash"),
        "input_name": metadata.get("filename") or getattr(uploaded_file, "name", ""),
        "profile_id": result.get("profile_id") or metadata.get("profile_id") or metadata.get("memory_bank_id"),
        "runtime_revision": metadata.get("runtime_revision") or context.get("runtime_revision"),
        "dataset": metadata.get("dataset", context.get("dataset") or dataset),
        "category": metadata.get("category", context.get("category") or category),
        "config": metadata.get("config_profile_id") or context.get("config_profile_id") or config,
        "config_id": metadata.get("config_profile_id") or context.get("config_profile_id") or config,
        "calibration_version": metadata.get("calibration_path") or "base",
        "image_path": api.artifact_url(result.get("original_image_url")),
        "current_file_ref": api.artifact_url(result.get("original_image_url")),
        "original_image_ref": api.artifact_url(result.get("original_image_url")),
        "result_json": api.artifact_url(metadata.get("result_json_url")),
        "heatmap_path": api.artifact_url(result.get("heatmap_url")),
        "heatmap_ref": api.artifact_url(result.get("heatmap_url")),
        "mask_path": api.artifact_url(result.get("mask_url")),
        "mask_ref": api.artifact_url(result.get("mask_url")),
        "overlay_path": api.artifact_url(result.get("result_image_url")),
        "overlay_ref": api.artifact_url(result.get("result_image_url")),
        "marked_image_path": api.artifact_url(result.get("marked_image_url")),
        "marked_image_ref": api.artifact_url(result.get("marked_image_url")),
        "marked_region_count": int(result.get("marked_region_count", 0) or 0),
        "score": result.get("anomaly_score"),
        "anomaly_score": result.get("anomaly_score"),
        "baseline_score": result.get("baseline_score", result.get("anomaly_score")),
        "calibrated_score": result.get("calibrated_score", result.get("anomaly_score")),
        "final_score": result.get("final_score", result.get("anomaly_score")),
        "calibration_applied": bool(result.get("calibration_applied", False)),
        "gate_status": result.get("gate_status", "not_applicable"),
        "fallback_used": bool(result.get("fallback_used", False)),
        "calibration_version": result.get("calibration_version") or metadata.get("calibration_path") or "base",
        "method_mode": result.get("method_mode") or metadata.get("method_mode") or "normal_memory",
        "threshold": result.get("threshold"),
        "pred_label": 1 if result.get("is_anomaly") else 0,
        "prediction": "anomaly" if result.get("is_anomaly") else "normal",
        "feedback_status": "unsubmitted",
        "inference_time": result.get("inference_time_ms"),
        "generate_visualizations": bool(result.get("generate_visualizations", True)),
        "visualization_status": result.get("visualization_status", "generated"),
        "timing": result.get("timing") or metadata.get("timing") or {},
        "timing_mode": result.get("timing_mode") or metadata.get("timing_mode") or "strict_total",
        "model_id": result.get("model_id"),
        "memory_bank_id": metadata.get("memory_bank_id"),
        "inference_mode": metadata.get("inference_mode", "inference_only"),
        "device": result.get("device"),
        "regions": result.get("regions", []),
        "warnings": payload.get("warnings", []),
        "raw_result": result,
        "bridge": {"mode": "integration_api", "request_id": payload.get("request_id"), "job_id": payload.get("job_id")},
        "latest_inference_path": str(LATEST_INFERENCE_PATH),
    }
    save_latest_inference(mapped)
    return mapped
