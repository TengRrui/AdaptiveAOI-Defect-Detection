from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from bridge.algorithm_bridge import log_software_error
from bridge.api_client import client
from service.status_service import latest_inference


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEEDBACK_DIR = PROJECT_ROOT / "outputs" / "system_runtime" / "feedback"
FEEDBACK_LOG_PATH = FEEDBACK_DIR / "feedback_log.csv"
ROLLBACK_LOG_PATH = FEEDBACK_DIR / "rollback_log.csv"

FEEDBACK_FIELDS = [
    "image_path",
    "old_score",
    "old_pred_label",
    "old_threshold",
    "feedback_label",
    "feedback_type",
    "timestamp",
    "dataset",
    "category",
    "calibration_version",
    "result_id",
]

FEEDBACK_TYPES = {
    "false positive": "误检",
    "false negative": "漏检",
    "true positive": "正确异常",
    "true negative": "正确正常",
}


def _api_error(payload: dict[str, Any], fallback: str) -> dict[str, Any]:
    error = payload.get("error") or {}
    message = str(error.get("message") or fallback)
    log_software_error("feedback_service", message)
    return {"ok": False, "error": message}


def _feedback_payload(context: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    """Only send controlled identifiers and human input, never local paths."""
    payload = {
        "mode": context.get("mode"),
        "task_id": context.get("task_id"),
        "sample_id": context.get("sample_id"),
        "result_id": context.get("result_id"),
        "frame_index": context.get("frame_index"),
        "timestamp_ms": context.get("timestamp_ms"),
        "interval_start_ms": context.get("interval_start_ms"),
        "interval_end_ms": context.get("interval_end_ms"),
        "perceptual_hash": context.get("perceptual_hash"),
        "assessment": values.get("assessment"),
        "corrected_label": values.get("corrected_label"),
        "corrected_defect_type": values.get("corrected_defect_type"),
        "issue_tags": list(values.get("issue_tags") or []),
        "note": str(values.get("note") or ""),
        "application_mode": values.get("application_mode", "save_only"),
    }
    return {key: value for key, value in payload.items() if value is not None}


def submit_feedback(context: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    payload = client().save_feedback(_feedback_payload(context, values))
    if not payload.get("success"):
        return _api_error(payload, "反馈保存失败。")
    return {"ok": True, "feedback": payload.get("feedback") or {}, "candidate": payload.get("candidate") or {}}


def _human_label_for_prediction(value: Any) -> str | None:
    prediction = str(value or "").strip().lower()
    if prediction in {"0", "normal", "ok", "good", "negative"}:
        return "normal"
    if prediction in {"1", "anomaly", "abnormal", "defect", "ng", "positive"}:
        return "defect"
    return None


def submit_bulk_feedback(contexts: list[dict[str, Any]], values: dict[str, Any]) -> dict[str, Any]:
    if not contexts:
        return {"ok": False, "error": "请先选择需要反馈的检测结果。"}
    feedback_payloads: list[dict[str, Any]] = []
    for context in contexts:
        item_values = dict(values)
        if item_values.get("assessment") == "derive_from_label":
            corrected_label = str(item_values.get("corrected_label") or "")
            predicted_label = _human_label_for_prediction(context.get("prediction"))
            item_values["assessment"] = (
                "correct" if predicted_label == corrected_label
                else "incorrect" if predicted_label in {"normal", "defect"}
                else "uncertain"
            )
        feedback_payloads.append(_feedback_payload(context, item_values))
    payload = client().save_bulk_feedback(feedback_payloads)
    if not payload.get("success"):
        return _api_error(payload, "批量反馈保存失败。")
    return {
        "ok": True,
        "feedbacks": payload.get("feedbacks") or [],
        "candidates": payload.get("candidates") or [],
        "saved_count": int(payload.get("saved_count") or 0),
        "failed_count": int(payload.get("failed_count") or 0),
        "failures": payload.get("failures") or [],
    }


def task_feedback(
    task_id: str,
    *,
    mode: str | None = None,
    sample_id: str | None = None,
    frame_index: int | None = None,
) -> dict[str, Any]:
    if not task_id:
        return {"ok": False, "error": "当前检测记录缺少任务标识。", "feedbacks": []}
    payload = client().feedbacks(task_id=task_id, mode=mode, sample_id=sample_id, frame_index=frame_index)
    if not payload.get("success"):
        failed = _api_error(payload, "反馈状态读取失败。")
        failed["feedbacks"] = []
        return failed
    return {"ok": True, "feedbacks": payload.get("feedbacks") or []}


def detection_task_items(
    task_id: str,
    *,
    mode: str,
    page: int = 1,
    page_size: int = 20,
    query: str | None = None,
    prediction: str | None = None,
    feedback_status: str | None = None,
    time_from_ms: float | None = None,
    time_to_ms: float | None = None,
) -> dict[str, Any]:
    if not task_id:
        return {"ok": False, "error": "当前检测记录缺少任务标识。", "items": [], "total": 0}
    payload = client().history_management_task_items(
        task_id,
        mode=mode,
        page=page,
        page_size=page_size,
        query=query,
        prediction=prediction,
        feedback_status=feedback_status,
        time_from_ms=time_from_ms,
        time_to_ms=time_to_ms,
    )
    if not payload.get("success"):
        failed = _api_error(payload, "检测明细读取失败。")
        failed.update(items=[], total=0, page=1, page_size=page_size, total_pages=1)
        return failed
    result = dict(payload.get("task_items") or {})
    result["ok"] = True
    return result


def video_feedback_interval(task_id: str, *, start_ms: float, end_ms: float, max_frames: int = 8) -> dict[str, Any]:
    """Ask the backend for non-adjacent representative frames in an interval."""
    if not task_id:
        return {"ok": False, "error": "当前视频任务缺少任务标识。", "frames": []}
    payload = client().video_feedback_interval(task_id, start_ms=start_ms, end_ms=end_ms, max_frames=max_frames)
    if not payload.get("success"):
        failed = _api_error(payload, "视频反馈区间读取失败。")
        failed["frames"] = []
        return failed
    return {"ok": True, "frames": payload.get("frames") or []}


def existing_feedback(context: dict[str, Any]) -> dict[str, Any] | None:
    response = task_feedback(
        str(context.get("task_id") or ""),
        mode=str(context.get("mode") or "") or None,
        sample_id=str(context.get("sample_id") or "") or None,
        frame_index=context.get("frame_index"),
    )
    records = response.get("feedbacks") or []
    return records[0] if records else None


def save_feedback(payload: dict[str, Any]) -> dict[str, Any]:
    feedback_type = str(payload.get("feedback_type", "")).strip().lower()
    if feedback_type not in FEEDBACK_TYPES:
        message = f"不支持的反馈类型: {feedback_type}"
        log_software_error("feedback_service", message)
        return {"ok": False, "error": message}
    latest = latest_inference() or {}
    result_id = str(payload.get("result_id") or latest.get("result_id") or "")
    if not result_id:
        return {"ok": False, "error": "当前记录没有后端 result_id，请先使用集成 API 完成一次检测。"}
    api_payload = client().save_feedback({"result_id": result_id, "feedback_type": feedback_type, "note": str(payload.get("note") or "")})
    if not api_payload.get("success"):
        error = api_payload.get("error") or {}
        message = error.get("message", "后端反馈保存失败。")
        log_software_error("feedback_service", message)
        return {"ok": False, "error": message, "api_error": error}
    record = {
        "image_path": payload.get("image_path") or latest.get("image_path", ""),
        "old_score": payload.get("old_score", latest.get("score", "")),
        "old_pred_label": payload.get("old_pred_label", latest.get("pred_label", "")),
        "old_threshold": payload.get("old_threshold", latest.get("threshold", "")),
        "feedback_label": payload.get("feedback_label", FEEDBACK_TYPES[feedback_type]),
        "feedback_type": feedback_type,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": payload.get("dataset", latest.get("dataset", "")),
        "category": payload.get("category", latest.get("category", "")),
        "calibration_version": payload.get("calibration_version", latest.get("calibration_version", "base")),
        "result_id": result_id,
    }
    _append_csv(FEEDBACK_LOG_PATH, FEEDBACK_FIELDS, record)
    return {"ok": True, "path": str(FEEDBACK_LOG_PATH), "record": record, "backend": api_payload.get("feedback")}


def update_threshold_from_feedback(
    dataset: str,
    category: str,
    config: str,
    old_threshold: float | int | str | None,
    calibration_version: str = "base",
) -> dict[str, Any]:
    del dataset, category, config, old_threshold, calibration_version
    return {
        "ok": False,
        "error": "反馈已写入真实检测历史。请在目标域适配的“校准与验证”区域创建独立校准任务，并在后端显式激活后再用于推理。",
    }


def record_rollback(dataset: str, category: str, from_version: str, to_version: str, reason: str = "") -> dict[str, Any]:
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset,
        "category": category,
        "from_version": from_version,
        "to_version": to_version,
        "reason": reason,
    }
    _append_csv(ROLLBACK_LOG_PATH, list(record.keys()), record)
    return {"ok": True, "path": str(ROLLBACK_LOG_PATH), "record": record}


def read_feedback_rows(dataset: str | None = None, category: str | None = None) -> list[dict[str, str]]:
    if not FEEDBACK_LOG_PATH.exists():
        return []
    with FEEDBACK_LOG_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if dataset:
        rows = [row for row in rows if str(row.get("dataset", "")).lower() == str(dataset).lower()]
    if category:
        rows = [row for row in rows if str(row.get("category", "")) == str(category)]
    return rows


def _append_csv(path: Path, fields: list[str], record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: record.get(field, "") for field in fields})
