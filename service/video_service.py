"""Streamlit-side client facade for the public AdaptiveAOI video API.

This module never imports a model or video decoder.  The API owns the
background worker and returns an isolated, uniquely named algorithm job.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from bridge.api_client import client
from service.runtime_config_service import load_runtime_context


VIDEO_SUFFIXES = ("mp4", "avi", "mov", "mkv")


def _failure(payload: dict[str, Any]) -> dict[str, Any]:
    error = payload.get("error") or {}
    return {"ok": False, "error": error.get("message", "视频检测服务调用失败。"), "api_error": error, "request_id": payload.get("request_id")}


def inspect_selected_video(uploaded_file: Any, progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    payload = client().upload_video(uploaded_file, progress=progress)
    if not payload.get("success"):
        return _failure(payload)
    upload = payload.get("upload") or {}
    return {"ok": True, "upload": upload, "request_id": payload.get("request_id")}


def inspect_saved_video(upload_id: str) -> dict[str, Any]:
    payload = client().video_upload(upload_id)
    if not payload.get("success"):
        return _failure(payload)
    return {"ok": True, "upload": payload.get("upload") or {}, "request_id": payload.get("request_id")}


def start_video_detection(upload_id: str, runtime: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    del runtime
    runtime_state = load_runtime_context()
    if not runtime_state.get("ok"):
        return {
            "ok": False,
            "error": runtime_state.get("error") or "当前正式运行配置尚未就绪。",
            "api_error": runtime_state.get("api_error") or {},
        }
    context = dict(runtime_state.get("context") or {})
    generate_visualizations = bool(options.get("generate_visualizations", True))
    payload = {
        "upload_id": upload_id,
        "profile_id": context.get("profile_id"),
        "runtime_revision": context.get("runtime_revision"),
        "threshold": options.get("threshold"),
        "mode": "inference_only",
        "frame_stride": int(options.get("frame_stride", 1)),
        "generate_visualizations": generate_visualizations,
        "save_heatmap": generate_visualizations and bool(options.get("save_heatmap", False)),
        "save_mask": generate_visualizations and bool(options.get("save_mask", False)),
        "save_jsonl": bool(options.get("save_jsonl", True)),
    }
    response = client().start_video_job(payload)
    if not response.get("success"):
        error = response.get("error") or {}
        if str(error.get("code") or "") == "INFERENCE_RUNTIME_BINDING_MISMATCH":
            latest = load_runtime_context()
            return {
                "ok": False,
                "error": "运行配置刚刚发生变化，已同步最新配置，请重新开始检测。",
                "api_error": error,
                "runtime_context": latest.get("context") or {},
                "request_id": response.get("request_id"),
            }
        return _failure(response)
    return {"ok": True, "video_job": response.get("video_job") or {}, "request_id": response.get("request_id")}


def video_job_status(job_id: str, heartbeat: bool = True) -> dict[str, Any]:
    response = client().video_job(job_id, heartbeat=heartbeat)
    if not response.get("success"):
        return _failure(response)
    return {"ok": True, "video_job": response.get("video_job") or {}, "request_id": response.get("request_id")}


def video_job_action(job_id: str, action: str) -> dict[str, Any]:
    actions = {
        "pause": client().pause_video_job,
        "resume": client().resume_video_job,
        "stop": client().cancel_video_job,
        "cancel": client().cancel_video_job,
        "rerun": client().rerun_video_job,
        "open_output": client().open_video_output,
        "open_result_video": client().open_video_result,
    }
    handler = actions.get(action)
    if handler is None:
        return {"ok": False, "error": f"不支持的视频任务操作：{action}"}
    response = handler(job_id)
    if not response.get("success"):
        return _failure(response)
    return {"ok": True, "video_job": response.get("video_job") or {}, "request_id": response.get("request_id")}


def video_filename(upload: dict[str, Any]) -> str:
    return Path(str(upload.get("original_filename") or upload.get("metadata", {}).get("file_name") or "")).name


__all__ = [
    "VIDEO_SUFFIXES",
    "inspect_selected_video",
    "inspect_saved_video",
    "start_video_detection",
    "video_filename",
    "video_job_action",
    "video_job_status",
]
