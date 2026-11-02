from __future__ import annotations

import csv
import io
import json
import re
from html import escape
from pathlib import Path
from typing import Any, Iterable

import streamlit as st

from ui_common import sanitized_text
from time_utils import format_beijing_datetime

FIELD_LABELS = {
    "model_type": "模型类型",
    "backbone": "骨干网络",
    "profile_valid": "配置有效性",
    "weights_valid": "记忆库状态",
    "memory_bank_ready": "记忆库状态",
    "calibration_valid": "校准状态",
    "calibration_ready": "校准状态",
    "loaded": "模型加载状态",
    "default_threshold": "默认阈值",
    "dataset": "数据集",
    "category": "类别",
    "created_at": "创建时间",
    "updated_at": "更新时间",
    "completed_at": "完成时间",
    "normal_sample_count": "正常样本数",
    "defect_sample_count": "缺陷校准样本数",
    "defect_calibration_sample_count": "缺陷校准样本数",
    "normal_source_types": "正常样本来源",
    "defect_source_types": "缺陷样本来源",
    "normal_image_count": "正常图片数量",
    "normal_video_count": "正常视频数量",
    "normal_extracted_frame_count": "正常有效帧数量",
    "defect_image_count": "缺陷图片数量",
    "defect_video_count": "缺陷视频数量",
    "defect_extracted_frame_count": "缺陷有效帧数量",
    "deduplicated_frame_count": "去重帧数量",
    "rejected_frame_count": "过滤帧数量",
    "video_sampling_config": "视频抽帧设置",
    "sample_budget_mode": "样本预算方式",
    "defect_video_intervals": "缺陷时间区间",
    "source_video_hashes": "来源视频校验",
    "frame_selection_seed": "帧选择种子",
    "video_decoder_version": "视频解码器",
    "sample_manifest": "样本清单",
    "threshold": "检测阈值",
    "image_threshold": "图像检测阈值",
    "pixel_threshold": "像素检测阈值",
    "threshold_method": "阈值校准方法",
    "threshold_strategy": "阈值校准方法",
    "frozen": "冻结状态",
    "inference_mode": "推理模式",
    "mode": "运行模式",
    "time": "发生时间",
    "action": "操作模块",
    "error": "错误详情",
    "message": "提示信息",
    "status": "状态",
    "duration": "耗时",
    "duration_ms": "耗时",
    "elapsed_seconds": "已用时间",
    "device": "运行设备",
    "precision": "计算精度",
    "friendly_name": "友好名称",
    "profile_kind": "记忆库类型",
    "normal_features_only": "记忆库内容",
    "test_data_update_allowed": "测试数据更新权限",
    "method_name": "当前检测方法",
    "method_mode": "方法模式",
    "method_mode_requested": "请求模式",
    "method_mode_resolved": "实际模式",
    "defect_prototype_count": "独立缺陷原型数",
    "dual_memory_alpha": "门控融合权重",
    "is_default": "默认记忆库",
    "pred_label": "检测结果",
    "prediction": "检测结果",
    "score": "异常分数",
    "anomaly_score": "异常分数",
    "normal_shot": "正常样本数",
    "defect_shot": "缺陷样本数",
    "use_defect_direction": "缺陷方向校准",
    "alpha": "融合权重",
    "calibration_version": "校准版本",
    "total_ms": "总耗时",
    "mean": "平均值",
    "p50": "P50",
    "p95": "P95",
    "min": "最小值",
    "max": "最大值",
    "torch": "PyTorch 版本",
    "cuda_available": "CUDA 状态",
    "exists": "资源状态",
    "decodable": "解码状态",
    "level": "错误级别",
    "success": "执行结果",
    "ok": "执行结果",
    "model_id": "模型版本",
    "memory_bank_id": "记忆库标识",
    "job_id": "任务标识",
    "task_id": "任务标识",
    "result_id": "结果标识",
    "config": "配置文件",
    "config_path": "配置文件",
    "image_path": "图像资源",
    "input_path": "输入资源",
    "output_path": "输出资源",
    "overlay_path": "叠加图资源",
    "heatmap_path": "热力图资源",
    "mask_path": "掩码资源",
    "filename": "文件名",
    "file_name": "文件名",
    "command": "执行命令",
    "returncode": "返回状态码",
    "stdout": "标准输出",
    "stderr": "错误输出",
    "details": "详细信息",
    "metadata": "任务元信息",
    "bridge": "算法调用信息",
    "generate_visualizations": "可视化结果",
    "inference_time": "推理耗时",
    "inference_ms": "纯推理耗时",
    "end_to_end_ms": "端到端耗时",
    "model_load_ms": "模型加载耗时",
    "memory_bank_load_ms": "记忆库加载耗时",
    "sample_count": "样本数",
    "item_count": "项目数",
    "frame_index": "帧编号",
    "timestamp_ms": "视频时间戳",
}

ENUM_LABELS = {
    "patchcore": "正常记忆异常检测",
    "wide_resnet50_2": "Wide-ResNet50-2",
    "best_f1": "最佳 F1",
    "recall_priority": "召回率优先",
    "normal_percentile": "正常样本百分位",
    "mean_std": "均值与标准差",
    "inference_only": "只读推理",
    "readonly": "只读推理",
    "target_domain": "目标域记忆库",
    "builtin_public_baseline": "内置公开数据基线",
    "success": "成功",
    "completed": "已完成",
    "failed": "失败",
    "cancelled": "已取消",
    "running": "运行中",
    "queued": "等待处理",
    "loading": "加载中",
    "initializing_python_runtime": "正在加载检测算法运行时组件",
    "loading_model": "正在加载模型",
    "loading_memory_bank": "正在加载当前正式 Memory Bank",
    "loading_index": "正在加载并校验近邻检索索引",
    "warming_up": "正在完成检测资源预热",
    "paused": "已暂停",
    "stopping": "正在安全停止",
    "unknown": "状态未知",
    "generated": "已生成",
    "not_generated": "未生成",
    "strict_total": "严格全流程计时",
    "normal": "正常",
    "anomaly": "异常",
    "auto": "自动选择",
    "normal_only": "仅正常样本基线",
    "normal_defect_calibration": "正常记忆与缺陷阈值校准",
    "normal_defect_dual_memory": "可靠性门控双记忆",
    "normal_memory_defect_guided_lite": "单正常记忆·缺陷引导轻量校准",
    "normal_memory": "正常记忆异常检测",
    "cuda": "CUDA",
    "cpu": "CPU",
    "true": "已启用",
    "false": "未启用",
}


def friendly_model_name(value: Any) -> str:
    """Map internal algorithm/model identifiers to the one public product name."""
    text = str(value or "").strip()
    lowered = text.lower().replace("_", "-")
    if "patchcore" in lowered or lowered.startswith("patch-core"):
        return "AdaptiveAOI 特征记忆检测模型"
    return text or "AdaptiveAOI 基础特征模型"

ACTION_LABELS = {
    "inference_service": "推理服务",
    "model_load": "模型加载",
    "memory_bank_load": "记忆库加载",
    "batch_inference": "批量检测",
    "video_inference": "视频检测",
    "calibration_service": "校准服务",
    "domain_adaptation": "目标域适配",
    "folder_dialog": "文件夹选择",
    "system": "系统",
    "batch_service": "批量检测服务",
    "video_service": "视频检测服务",
    "runtime_config": "运行配置",
    "feedback_service": "反馈服务",
}


def install_friendly_display_styles() -> None:
    st.markdown(
        """
        <style>
          .friendly-status { display:inline-flex; align-items:center; gap:.28rem; padding:.18rem .5rem; border:1px solid rgba(112,139,160,.24); border-radius:999px; font-size:.82rem; font-weight:720; }
          .friendly-status-ok { background:rgba(22,138,173,.09); color:#126782; }
          .friendly-status-warn { background:rgba(238,155,0,.10); color:#8a5a00; }
          .friendly-status-error { background:rgba(209,73,91,.09); color:#a12f40; }
          .friendly-log-meta { color:#6b7e91; font-size:.82rem; margin-bottom:.3rem; }
          .friendly-log-title { color:inherit; font-weight:760; margin:.2rem 0; }
          .friendly-log-copy { color:inherit; line-height:1.6; overflow-wrap:anywhere; }
          .friendly-log-advice { margin-top:.45rem; padding:.48rem .62rem; border-radius:8px; background:rgba(22,138,173,.06); line-height:1.5; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def field_label(field: str) -> str:
    return FIELD_LABELS.get(str(field), str(field).replace("_", " ").strip().title())


def format_local_time(value: Any) -> str:
    return format_beijing_datetime(value, fallback="暂无记录")


def _missing_text(field: str) -> str:
    if field in {"normal_sample_count", "defect_sample_count", "defect_calibration_sample_count", "created_at", "time"}:
        return "暂无记录"
    if "duration" in field or field.endswith("_ms"):
        return "未统计"
    if field in {"threshold", "default_threshold", "image_threshold", "pixel_threshold"}:
        return "未提供"
    return "未提供"


def _boolean_text(field: str, value: bool) -> str:
    mapping = {
        "profile_valid": ("✓ 已通过", "⚠ 无效"),
        "weights_valid": ("✓ 已就绪", "⚠ 未就绪"),
        "memory_bank_ready": ("✓ 已就绪", "⚠ 未就绪"),
        "calibration_valid": ("✓ 已完成", "⚠ 未完成"),
        "calibration_ready": ("✓ 已完成", "⚠ 未完成"),
        "loaded": ("✓ 已加载", "○ 未加载"),
        "frozen": ("🔒 已冻结", "⚠ 未冻结"),
        "normal_features_only": ("✓ 仅正常特征", "⚠ 包含非正常特征"),
        "test_data_update_allowed": ("⚠ 允许", "⛔ 禁止"),
        "cuda_available": ("✓ 可用", "○ 不可用"),
        "exists": ("✓ 已存在", "⚠ 缺失"),
        "decodable": ("✓ 可解码", "⚠ 无法解码"),
        "is_default": ("✓ 当前默认", "普通版本"),
        "use_defect_direction": ("✓ 已启用", "未启用"),
    }
    positive, negative = mapping.get(field, ("✓ 已启用", "○ 未启用"))
    return positive if value else negative


def _friendly_path(value: str) -> str:
    text = sanitized_text(value)
    text = re.sub(
        r"(?i)\b(token|password|passwd|secret|api[_-]?key|authorization)\b\s*[:=]\s*[^\s,;]+",
        r"\1=已隐藏",
        text,
    )
    if text != value:
        if re.fullmatch(r"[A-Za-z]:\\[^\r\n\"']+", value.strip()):
            candidate = Path(value).name
            return candidate if candidate and candidate not in {".", "\\"} else "内部位置（已隐藏）"
        return text
    if "/artifacts/" in value or value.startswith(("http://", "https://")):
        return "内部资源（已隐藏）"
    return value


def friendly_value(value: Any, field: str = "", *, digits: int = 4) -> str:
    field = str(field or "")
    if value is None or str(value).strip().lower() in {"null", "none", "undefined"}:
        return _missing_text(field)
    sensitive_field = field.lower()
    if any(token in sensitive_field for token in ("password", "passwd", "secret", "token", "authorization", "api_key")):
        return "敏感信息（已隐藏）"
    if isinstance(value, bool):
        return _boolean_text(field, value)
    if isinstance(value, (list, tuple, set)):
        if not value:
            return "暂无记录"
        if all(not isinstance(item, (dict, list, tuple, set)) for item in value):
            return "、".join(friendly_value(item, field) for item in value)
        return f"共 {len(value)} 项"
    if isinstance(value, dict):
        return f"包含 {len(value)} 项结构化信息"
    if field in {"created_at", "updated_at", "completed_at", "time", "timestamp"} or field.endswith("_at"):
        return format_local_time(value)
    if isinstance(value, (int, float)):
        if field in {"threshold", "default_threshold", "image_threshold", "pixel_threshold", "score", "anomaly_score", "alpha"}:
            return f"{float(value):.{digits}f}"
        if field.endswith("_ms") or field in {"duration_ms", "total_ms"}:
            return f"{float(value) / 1000:.2f} 秒" if float(value) >= 1000 else f"{float(value):.2f} 毫秒"
        if field.endswith("_seconds") or field == "elapsed_seconds":
            return f"{float(value):.1f} 秒"
        return f"{value:,}" if isinstance(value, int) else f"{float(value):.4f}".rstrip("0").rstrip(".")
    text = str(value).strip()
    lowered = text.lower()
    if field in {"model_id", "model_version", "model_type", "algorithm", "algorithm_version"}:
        return friendly_model_name(text)
    if lowered in {"true", "false"}:
        return _boolean_text(field, lowered == "true")
    if any(token in field.lower() for token in ("path", "url", "root", "directory", "folder", "artifact", "log_file")):
        candidate = Path(text).name
        return candidate if candidate and candidate not in {".", "\\", "/"} else "内部位置（已隐藏）"
    if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", text):
        return format_local_time(text)
    if lowered in ENUM_LABELS:
        return ENUM_LABELS[lowered]
    if field == "action":
        return ACTION_LABELS.get(lowered, text.replace("_", " "))
    if field in {"pred_label", "prediction"}:
        return "异常" if lowered in {"1", "true", "anomaly", "defect", "ng"} else "正常"
    if field.endswith("_id") and len(text) > 18:
        return f"{text[:8]}…{text[-4:]}"
    return _friendly_path(text)


def friendly_error_message(value: Any, fallback: str = "操作未能完成。") -> str:
    """从接口错误对象中提取可面向用户展示的说明。"""
    current = value
    for _ in range(4):
        if not isinstance(current, dict):
            break
        current = current.get("message") or current.get("detail") or current.get("error") or fallback
    if isinstance(current, (list, tuple)):
        current = "；".join(str(item) for item in current[:5])
    text = sanitized_text(str(current or fallback)).strip()
    # Transport details are useful in backend logs but are not actionable (or
    # safe) in the ordinary UI.  Keep a single stable Chinese message even if
    # an older service/client still returns the raw requests exception.
    lowered = text.lower()
    if any(token in lowered for token in ("httpconnectionpool", "max retries exceeded", "newconnectionerror", "winerror 10061", "connection refused", "connect timeout", "read timed out")):
        return "算法服务正在启动，请稍候。"
    if "traceback (most recent call last)" in lowered:
        return fallback
    text = re.sub(
        r"(?i)\b(token|password|passwd|secret|api[_-]?key|authorization)\b\s*[:=]\s*[^\s,;]+",
        r"\1=已隐藏",
        text,
    )
    if "Traceback (most recent call last)" in text:
        lines = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("File ")]
        text = lines[-1] if lines else fallback
    return text or fallback


def _status_class(text: str) -> str:
    if any(token in text for token in ("失败", "无效", "缺失", "无法", "错误")):
        return "friendly-status-error"
    if any(token in text for token in ("未就绪", "未完成", "未冻结", "不可用", "警告")):
        return "friendly-status-warn"
    return "friendly-status-ok"


def _is_status_field(field: str, value: Any) -> bool:
    return isinstance(value, bool) or field in {
        "profile_valid", "weights_valid", "memory_bank_ready", "calibration_valid", "calibration_ready",
        "loaded", "frozen", "normal_features_only", "test_data_update_allowed", "cuda_available", "exists",
        "decodable", "is_default", "status", "inference_mode", "mode",
    }


def render_description_list(
    title: str,
    data: dict[str, Any],
    *,
    fields: Iterable[str] | None = None,
    columns: int = 2,
    subtitle: str = "",
) -> None:
    install_friendly_display_styles()
    selected_fields = list(fields or data.keys())
    with st.container(border=True):
        st.markdown(f"#### {title}")
        if subtitle:
            st.caption(subtitle)
        if not selected_fields:
            st.info("暂无可展示的信息。")
            return
        layout = st.columns(max(1, columns))
        for index, field in enumerate(selected_fields):
            value = data.get(field)
            display = friendly_value(value, field)
            with layout[index % len(layout)]:
                st.caption(field_label(field))
                if _is_status_field(field, value):
                    css = _status_class(display)
                    st.markdown(
                        f'<span class="friendly-status {css}">{escape(display)}</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    title_attr = ""
                    if field in {"threshold", "default_threshold", "image_threshold", "pixel_threshold"} and value is not None:
                        title_attr = f' title="{escape(str(value))}"'
                    st.markdown(f'<strong{title_attr}>{escape(display)}</strong>', unsafe_allow_html=True)


def render_model_status(data: dict[str, Any], title: str = "模型状态") -> None:
    render_description_list(
        title,
        data,
        fields=("model_type", "backbone", "profile_valid", "memory_bank_ready", "calibration_ready", "loaded", "default_threshold"),
        columns=2,
        subtitle="模型内部字段已转换为中文状态，不改变后端真实配置。",
    )


def render_memory_bank_details(data: dict[str, Any], title: str = "记忆库详情") -> None:
    payload = dict(data)
    payload.setdefault("inference_mode", "inference_only")
    render_description_list(
        title,
        payload,
        fields=(
            "friendly_name", "dataset", "category", "created_at", "normal_sample_count",
            "defect_calibration_sample_count", "normal_source_types", "defect_source_types",
            "normal_image_count", "normal_video_count", "normal_extracted_frame_count",
            "defect_image_count", "defect_video_count", "defect_extracted_frame_count",
            "method_name", "method_mode_resolved", "defect_prototype_count",
            "threshold", "threshold_method", "frozen", "inference_mode",
        ),
        columns=2,
        subtitle="仅展示友好名称和业务状态，不显示内部路径或完整标识。",
    )


def _flatten(value: Any, prefix: str = "") -> list[tuple[str, Any, str]]:
    rows: list[tuple[str, Any, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix} / {field_label(str(key))}" if prefix else field_label(str(key))
            if isinstance(item, (dict, list, tuple)):
                rows.extend(_flatten(item, path))
            else:
                rows.append((path, item, str(key)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value, start=1):
            path = f"{prefix} / 第 {index} 项" if prefix else f"第 {index} 项"
            if isinstance(item, (dict, list, tuple)):
                rows.extend(_flatten(item, path))
            else:
                rows.append((path, item, ""))
    else:
        rows.append((prefix or "内容", value, ""))
    return rows


def render_technical_details(
    value: Any,
    title: str = "查看技术详情",
    *,
    expanded: bool = False,
    key: str | None = None,
) -> None:
    with st.expander(title, expanded=expanded, key=key):
        rows = [
            {"字段": path, "内容": friendly_value(item, source_field)}
            for path, item, source_field in _flatten(value)
        ]
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
        else:
            st.info("暂无技术详情。")


def render_structured_data(title: str, value: Any, *, subtitle: str = "") -> None:
    if isinstance(value, dict):
        render_description_list(title, value, columns=2, subtitle=subtitle)
        nested = {key: item for key, item in value.items() if isinstance(item, (dict, list, tuple))}
        if nested:
            render_technical_details(nested, "查看分组详情")
        return
    if isinstance(value, list):
        if not value:
            st.info("暂无可展示的记录。")
            return
        if all(isinstance(item, dict) for item in value):
            display_rows = []
            for item in value:
                display_rows.append(
                    {
                        field_label(str(key)): friendly_value(cell, str(key))
                        for key, cell in item.items()
                        if not isinstance(cell, (dict, list, tuple))
                    }
                )
            st.markdown(f"#### {title}")
            if subtitle:
                st.caption(subtitle)
            st.dataframe(display_rows, hide_index=True, width="stretch")
            return
    render_description_list(title, {"message": value}, fields=("message",), columns=1, subtitle=subtitle)


def render_experiment_content(content: Any, file_type: str) -> None:
    kind = str(file_type or "").lower()
    if isinstance(content, (dict, list)):
        render_structured_data("结构化实验结果", content, subtitle="字段已转换为中文友好格式。")
        return
    text = sanitized_text(str(content or "")).strip()
    if not text:
        st.info("实验文件没有可展示的内容。")
        return
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if isinstance(parsed, (dict, list)):
        render_structured_data("结构化实验结果", parsed, subtitle="实验文件已按中文业务字段展示。")
        return
    if kind == "csv":
        try:
            rows = list(csv.DictReader(io.StringIO(text)))
        except csv.Error:
            rows = []
        if rows:
            st.dataframe(
                [{field_label(key): friendly_value(value, key) for key, value in row.items()} for row in rows],
                hide_index=True,
                width="stretch",
            )
            return
    if kind in {"md", "markdown"}:
        st.markdown(text)
        return
    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    with st.container(border=True):
        st.markdown("#### 实验结果内容")
        for paragraph in paragraphs[:300]:
            st.write(paragraph)


def _parse_logs(raw: str, default_level: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in str(raw or "").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {"message": text}
        if not isinstance(payload, dict):
            payload = {"message": str(payload)}
        level = str(payload.get("level") or default_level).lower()
        message = payload.get("error") or payload.get("message") or payload.get("detail") or "未提供详细说明。"
        if isinstance(message, dict):
            message = message.get("message") or message.get("detail") or message.get("error") or "操作未能完成。"
        elif isinstance(message, (list, tuple)):
            message = "；".join(str(item) for item in message[:5]) or "操作未能完成。"
        records.append(
            {
                "time": payload.get("time") or payload.get("timestamp") or payload.get("created_at"),
                "action": payload.get("action") or payload.get("module") or "system",
                "level": level,
                "message": sanitized_text(message),
                "technical": payload,
            }
        )
    records.sort(key=lambda item: str(item.get("time") or ""), reverse=True)
    return records


def _problem_summary(message: str) -> str:
    text = str(message or "")
    for marker in ("：", ":", "。"):
        if marker in text:
            return text.split(marker, 1)[0].strip() or "操作未能完成"
    return text[:56] or "操作未能完成"


def _advice(message: str) -> str:
    text = str(message or "").lower()
    if "不存在" in text or "not found" in text:
        return "请重新选择有效文件或资源，并确认原文件没有被移动。"
    if "权限" in text or "permission" in text or "access" in text:
        return "请检查目标位置的读取或写入权限，然后重试。"
    if "连接" in text or "服务" in text or "timeout" in text:
        return "请确认 AdaptiveAOI 服务仍在运行，稍后重新执行当前操作。"
    if "模型" in text or "memory" in text or "记忆库" in text:
        return "请前往运行配置检查模型、冻结记忆库和阈值状态。"
    return "请根据问题说明检查输入条件；如问题持续，可展开技术详情定位。"


def render_log_list(raw: str, *, default_level: str = "error", empty_message: str = "当前没有需要处理的错误日志。") -> None:
    install_friendly_display_styles()
    records = _parse_logs(raw, default_level)
    if not records:
        st.info(empty_message)
        return
    level_labels = {
        "error": ("错误", "✕", "friendly-status-error"),
        "warning": ("警告", "⚠", "friendly-status-warn"),
        "warn": ("警告", "⚠", "friendly-status-warn"),
        "info": ("信息", "ℹ", "friendly-status-ok"),
    }
    for index, record in enumerate(records[:30]):
        level_label, icon, css = level_labels.get(record["level"], level_labels[default_level])
        module = ACTION_LABELS.get(str(record["action"]).lower(), str(record["action"]).replace("_", " "))
        message = str(record["message"] or "未提供详细说明。")
        with st.container(border=True):
            st.markdown(
                f'<span class="friendly-status {css}">{icon} {level_label}</span> '
                f'<span class="friendly-log-meta">{escape(format_local_time(record.get("time")))} · {escape(module)}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f'<div class="friendly-log-title">{escape(_problem_summary(message))}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="friendly-log-copy">{escape(message)}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="friendly-log-advice"><b>建议：</b>{escape(_advice(message))}</div>', unsafe_allow_html=True)
            render_technical_details(
                record["technical"],
                f"查看{level_label}技术详情 {index + 1}",
                key=f"friendly_log_{default_level}_{index}",
            )


__all__ = [
    "ACTION_LABELS",
    "FIELD_LABELS",
    "field_label",
    "friendly_error_message",
    "friendly_model_name",
    "format_local_time",
    "friendly_value",
    "install_friendly_display_styles",
    "render_description_list",
    "render_experiment_content",
    "render_log_list",
    "render_memory_bank_details",
    "render_model_status",
    "render_structured_data",
    "render_technical_details",
]
