from __future__ import annotations

from pathlib import Path
from typing import Any

from bridge.api_client import client
from bridge.algorithm_bridge import log_software_error
from service.runtime_config_service import load_runtime_context


def _fail(message: str, **extra: Any) -> dict[str, Any]:
    log_software_error("batch_service", message)
    return {"ok": False, "error": message, **extra}


def _to_row(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") or {}
    return {
        "result_id": item.get("result_id"),
        "image_name": metadata.get("filename"),
        "prediction": item.get("prediction"),
        "pred_label": 1 if item.get("is_anomaly") else 0,
        "score": item.get("anomaly_score"),
        "baseline_score": item.get("baseline_score", item.get("anomaly_score")),
        "calibrated_score": item.get("calibrated_score", item.get("anomaly_score")),
        "final_score": item.get("final_score", item.get("anomaly_score")),
        "calibration_applied": bool(item.get("calibration_applied", False)),
        "gate_status": item.get("gate_status", "not_applicable"),
        "fallback_used": bool(item.get("fallback_used", False)),
        "calibration_version": item.get("calibration_version") or metadata.get("calibration_path") or "base",
        "method_mode": item.get("method_mode") or metadata.get("method_mode") or "normal_memory",
        "threshold": item.get("threshold"),
        "memory_bank_id": metadata.get("memory_bank_id"),
        "inference_mode": metadata.get("inference_mode", "inference_only"),
        "total_ms": item.get("inference_time_ms"),
        "original_image_url": client().artifact_url(item.get("original_image_url")),
        "image_url": client().artifact_url(item.get("original_image_url")),
        "heatmap_url": client().artifact_url(item.get("heatmap_url")),
        "mask_url": client().artifact_url(item.get("mask_url")),
        "marked_image_url": client().artifact_url(item.get("marked_image_url")),
        "overlay_url": client().artifact_url(item.get("result_image_url")),
        "visualization_artifacts": item.get("visualization_artifacts") or [],
        "marked_region_count": int(item.get("marked_region_count", 0) or 0),
        "generate_visualizations": bool(item.get("generate_visualizations", True)),
        "visualization_status": item.get("visualization_status", "generated"),
        "timing": item.get("timing") or metadata.get("timing") or {},
        "timing_mode": item.get("timing_mode") or metadata.get("timing_mode") or "strict_total",
    }


def _runtime(generate_visualizations: bool = True) -> dict[str, Any]:
    state = load_runtime_context()
    if not state.get("ok"):
        return {
            "ok": False,
            "error": state.get("error") or "当前正式运行配置尚未就绪。",
            "api_error": state.get("api_error") or {},
        }
    return {
        "ok": True,
        "runtime_context": dict(state.get("context") or {}),
        "generate_visualizations": bool(generate_visualizations),
    }


def _input_path(input_dir: str | Path) -> Path | None:
    try:
        input_path = Path(input_dir).expanduser().resolve()
        if not input_path.exists() or not input_path.is_dir():
            return None
    except (OSError, RuntimeError):
        return None
    return input_path


def _result_from_payload(payload: dict[str, Any], *, input_dir: str = "", job_id: str = "") -> dict[str, Any]:
    rows = [_to_row(item) for item in payload.get("results", [])]
    return {
        "ok": True,
        "request_id": payload.get("request_id"),
        "job_id": payload.get("job_id") or job_id,
        "batch_id": payload.get("batch_id") or job_id,
        "input_dir": input_dir,
        "result_csv": client().artifact_url(payload.get("result_csv_url")),
        "export_csv": client().artifact_url(payload.get("result_csv_url")),
        "preview_rows": rows,
        "row_count": payload.get("total", 0),
        "mean_ms": payload.get("average_inference_time_ms"),
        "p95_ms": payload.get("p95_inference_time_ms"),
        "generate_visualizations": bool(payload.get("generate_visualizations", True)),
        "visualization_status": payload.get("visualization_status", "generated"),
        "timing": payload.get("timing") or {},
        "timing_mode": payload.get("timing_mode") or "strict_total",
        "normal": payload.get("normal", 0),
        "anomaly": payload.get("anomaly", 0),
        "errors": payload.get("errors", []),
        "warnings": payload.get("warnings", []),
        "status": payload.get("status") or "completed",
        "bridge": {"mode": "integration_api", "request_id": payload.get("request_id"), "job_id": payload.get("job_id") or job_id},
        "note": "结果已保存在本次批量任务的受控输出目录。",
    }


def start_batch_inference(
    dataset: str,
    category: str,
    config: str,
    input_dir: str | Path,
    device: str = "auto",
    memory_bank_id: str | None = None,
    generate_visualizations: bool = True,
) -> dict[str, Any]:
    input_path = _input_path(input_dir)
    if input_path is None:
        return _fail("输入图片文件夹不存在、无法访问或不是有效目录。")
    runtime = _runtime(generate_visualizations)
    if not runtime.get("ok"):
        return _fail(str(runtime.get("error")), api_error=runtime.get("api_error") or {})
    payload = client().detect_directory(str(input_path), runtime, async_job=True)
    if not payload.get("success"):
        error = payload.get("error") or {}
        if str(error.get("code") or "") == "INFERENCE_RUNTIME_BINDING_MISMATCH":
            latest = load_runtime_context()
            return _fail(
                "运行配置刚刚发生变化，已同步最新配置，请重新开始检测。",
                api_error=error,
                runtime_context=latest.get("context") or {},
            )
        return _fail(error.get("message", "批量检测任务创建失败。"), api_error=error)
    return {
        "ok": True,
        "job_id": payload.get("job_id"),
        "batch_id": payload.get("batch_id"),
        "status": payload.get("status") or "queued",
        "total": payload.get("total") or 0,
        "input_dir": str(input_path),
        "dataset": runtime["runtime_context"].get("dataset") or dataset,
        "category": runtime["runtime_context"].get("category") or category,
        "config": runtime["runtime_context"].get("config_profile_id") or config,
        "generate_visualizations": bool(generate_visualizations),
    }


def batch_inference_status(job_id: str) -> dict[str, Any]:
    payload = client().job(job_id)
    if not payload.get("success"):
        error = payload.get("error") or {}
        return _fail(error.get("message", "批量任务状态读取失败。"), api_error=error)
    manifest = payload.get("job") or {}
    result_payload = payload.get("result") or {}
    result = _result_from_payload(result_payload, job_id=job_id) if result_payload else None
    return {"ok": True, "job": manifest, "result": result}


def run_batch_inference(
    dataset: str,
    category: str,
    config: str,
    input_dir: str | Path,
    output_dir: str | Path,
    execute: bool = True,
    device: str = "auto",
    memory_bank_id: str | None = None,
    generate_visualizations: bool = True,
) -> dict[str, Any]:
    """Compatibility synchronous path retained for non-UI callers."""
    del output_dir, execute
    input_path = _input_path(input_dir)
    if input_path is None:
        return _fail("输入图片文件夹不存在、无法访问或不是有效目录。")
    runtime = _runtime(generate_visualizations)
    if not runtime.get("ok"):
        return _fail(str(runtime.get("error")), api_error=runtime.get("api_error") or {})
    payload = client().detect_directory(str(input_path), runtime)
    if not payload.get("success"):
        error = payload.get("error") or {}
        return _fail(error.get("message", "Integration API batch detection failed."), api_error=error, request_id=payload.get("request_id"))
    return _result_from_payload(payload, input_dir=str(input_path))
