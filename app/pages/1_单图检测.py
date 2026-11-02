from __future__ import annotations

from html import escape
from io import BytesIO
from typing import Any

import streamlit as st

from ui_common import (
    configure_page,
    display_heatmap_overlay,
    display_optional_image,
    format_value,
    initialize_detection_visualization_default,
    label_for_pred,
    render_action_button_with_feedback,
    render_detection_resource_monitor,
    render_runtime_summary,
    runtime_header_items,
    sidebar_runtime_controls,
    update_action_feedback,
)

from feedback_panel import render_feedback_panel
from friendly_display import friendly_error_message, render_technical_details
from service.inference_service import result_matches_input, run_image_inference, uploaded_input_hash
from timing_panel import render_single_timing


SINGLE_INPUT_SNAPSHOT_KEY = "single_input_snapshot"
SINGLE_RUNTIME_BINDING_KEY = "single_runtime_binding"
SINGLE_RUNTIME_CHANGED_KEY = "single_runtime_changed"


class _RetainedUpload(BytesIO):
    """File-uploader compatible view of the current session's retained bytes."""

    def __init__(self, payload: bytes, *, name: str, mime_type: str = "application/octet-stream") -> None:
        super().__init__(payload)
        self.name = name
        self.type = mime_type
        self.size = len(payload)


def _runtime_binding(snapshot: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(snapshot.get("runtime_revision") or 0),
        str(snapshot.get("profile_id") or ""),
        str(snapshot.get("memory_bank_id") or ""),
        str(snapshot.get("config_id") or ""),
        str(snapshot.get("model_id") or ""),
    )


def _retain_single_upload(uploaded: Any | None) -> Any | None:
    """Preserve the selected image across Runtime and multipage rerenders."""

    if uploaded is not None:
        payload = bytes(uploaded.getvalue())
        snapshot = {
            "name": str(getattr(uploaded, "name", "current-image")),
            "type": str(getattr(uploaded, "type", "application/octet-stream")),
            "payload": payload,
        }
        st.session_state[SINGLE_INPUT_SNAPSHOT_KEY] = snapshot
        return uploaded
    snapshot = st.session_state.get(SINGLE_INPUT_SNAPSHOT_KEY)
    if not isinstance(snapshot, dict) or not snapshot.get("payload"):
        return None
    st.caption(f"已保留当前图片：{snapshot.get('name') or '当前图片'}")
    if st.button("移除当前图片", key="single_clear_retained_file", width="stretch"):
        st.session_state.pop(SINGLE_INPUT_SNAPSHOT_KEY, None)
        st.session_state.pop("last_single_result", None)
        st.session_state.pop(SINGLE_RUNTIME_CHANGED_KEY, None)
        st.rerun(scope="fragment")
    return _RetainedUpload(
        bytes(snapshot["payload"]),
        name=str(snapshot.get("name") or "current-image"),
        mime_type=str(snapshot.get("type") or "application/octet-stream"),
    )


def _html(value: Any) -> str:
    return escape(str(value if value not in (None, "") else "-"))


def _is_anomaly(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "anomaly", "abnormal", "defect", "ng", "异常"}:
        return True
    try:
        return int(float(text)) == 1
    except (TypeError, ValueError):
        return False


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_ms(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return "-"
    if number >= 1000:
        return f"{number / 1000:.2f}s"
    return f"{number:.0f}ms"


def _friendly_calibration(value: Any) -> str:
    """Display a calibration identity without exposing a local filesystem path."""
    text = str(value or "").strip()
    if not text:
        return "当前校准配置"
    if "\\" in text or "/" in text:
        filename = text.replace("\\", "/").rsplit("/", 1)[-1]
        return filename or "当前校准配置"
    return text


def _hero(runtime: dict[str, Any], task_context: dict[str, Any] | None = None) -> None:
    badges = "".join(
        f'<span class="aoi-badge">{_html(label)}：{_html(value)}</span>'
        for label, value in runtime_header_items(runtime, task_context)
    )
    st.markdown(
        f"""
        <div class="aoi-workbench-hero">
          <div>
            <div class="aoi-workbench-kicker">Single-image Inspection</div>
            <div class="aoi-workbench-title">单图检测</div>
            <div class="aoi-workbench-copy">
              上传单张工件图像，通过统一算法接口调用当前冻结异常检测方案，输出异常判定、分数阈值以及原始图像、异常热力叠加图、缺陷定位和缺陷 Mask 证据。
            </div>
          </div>
          <div class="aoi-workbench-meta">
            {badges}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _verdict_card(result: dict[str, Any]) -> None:
    anomaly = _is_anomaly(result.get("pred_label"))
    card_class = "aoi-verdict-card-danger" if anomaly else "aoi-verdict-card-normal"
    badge_class = "aoi-badge-danger" if anomaly else "aoi-badge-success"
    label = label_for_pred(result.get("pred_label"))
    copy = "建议复核热力图与掩膜区域，并结合阈值版本判断是否进入反馈校准。" if anomaly else "当前图像低于异常判定阈值，可作为合格样本记录。"
    st.markdown(
        f"""
        <div class="aoi-verdict-card {card_class}">
          <div class="aoi-verdict-label">检测结论</div>
          <div class="aoi-verdict-value">{_html(label)}</div>
          <span class="aoi-badge {badge_class}">Pred Label: {_html(result.get("pred_label"))}</span>
          <div class="aoi-verdict-copy">{_html(copy)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _score_strip(result: dict[str, Any]) -> None:
    st.markdown(
        f"""
        <div class="aoi-score-strip">
          <div class="aoi-score-card">
            <div class="aoi-score-label">异常分数</div>
            <div class="aoi-score-value">{_html(format_value(result.get("score")))}</div>
            <div class="aoi-score-foot">图像级异常分数</div>
          </div>
          <div class="aoi-score-card">
            <div class="aoi-score-label">判定阈值</div>
            <div class="aoi-score-value">{_html(format_value(result.get("threshold")))}</div>
            <div class="aoi-score-foot">当前校准阈值</div>
          </div>
          <div class="aoi-score-card">
            <div class="aoi-score-label">推理耗时</div>
            <div class="aoi-score-value">{_html(_format_ms(result.get("inference_time")))}</div>
            <div class="aoi-score-foot">端到端推理耗时</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _detail_panel(result: dict[str, Any]) -> None:
    st.markdown(
        f"""
        <div class="aoi-panel">
          <div class="aoi-panel-title">检测详情</div>
          <div class="aoi-panel-subtitle">用于检测复核和问题定位的关键记录</div>
          <div class="aoi-mini-row">
            <div class="aoi-mini-label">数据集 / 类别</div>
            <div class="aoi-mini-value">{_html(result.get("dataset"))} / {_html(result.get("category"))}</div>
          </div>
          <div class="aoi-mini-row">
            <div class="aoi-mini-label">校准版本</div>
            <div class="aoi-mini-value">{_html(_friendly_calibration(result.get("calibration_version")))}</div>
          </div>
          <div class="aoi-mini-row">
            <div class="aoi-mini-label">结果记录</div>
            <div class="aoi-mini-value">已保存</div>
          </div>
          <div class="aoi-mini-row">
            <div class="aoi-mini-label">最近记录</div>
            <div class="aoi-mini-value">已同步</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("检测记录已保存，可在当前结果下方提交反馈与优化建议。")


def _image_tile(title: str, note: str, path_value: str | Path | None, missing_note: str | None = None) -> None:
    st.markdown(
        f"""
        <div class="aoi-image-tile-title">{_html(title)}</div>
        <div class="aoi-image-tile-note">{_html(note)}</div>
        """,
        unsafe_allow_html=True,
    )
    if path_value:
        display_optional_image(title, path_value)
    else:
        st.info(missing_note or f"{title} 当前不可用。")


def _result_gallery(result: dict[str, Any]) -> None:
    artifact_status = str(result.get("artifact_status") or "")
    if artifact_status == "unavailable":
        st.warning("最近检测记录仍可追溯，但其原图和可视化文件已不在结果目录中。页面已停止请求失效文件；如需图像证据，请重新执行检测。")
    elif artifact_status == "partial":
        st.info("最近检测记录的部分可视化文件已不可用；仍存在的文件继续展示，缺失项不会触发无效请求。")
    if not bool(result.get("generate_visualizations", True)):
        st.info("本次检测未生成可视化结果；异常分数、阈值、判定和性能耗时已正常保存。")
        _image_tile("原始图像", "输入工件图像，作为检测基准。", result.get("image_path"))
        return
    st.markdown(
        """
        <div class="aoi-result-toolbar">
          <div>
            <div class="aoi-section-title">可视化证据</div>
              <div class="aoi-section-desc">原始图像、红绿异常热力叠加、缺陷定位和缺陷 Mask 形成完整证据链。</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    row1 = st.columns(2)
    with row1[0]:
        _image_tile("原始图像", "输入工件图像，作为检测基准。", result.get("image_path"))
    with row1[1]:
        overlay_path = (
            result.get("overlay_path")
            or result.get("overlay_ref")
            or result.get("result_image_url")
        )
        overlay_note = "以原始图像为底图，叠加红绿异常响应分布；绿色表示低响应，红色表示高响应。"
        st.markdown(
            f"<div class=\"aoi-image-tile-title\">{_html('热力图叠加')}</div>"
            f"<div class=\"aoi-image-tile-note\">{_html(overlay_note)}</div>",
            unsafe_allow_html=True,
        )
        display_heatmap_overlay(
            "热力图叠加",
            overlay_path=overlay_path,
            original_path=result.get("image_path"),
            heatmap_path=result.get("heatmap_path"),
        )
    row2 = st.columns(2)
    with row2[0]:
        anomaly = _is_anomaly(result.get("pred_label"))
        marked_path = result.get("marked_image_path") or result.get("marked_image_ref")
        marked_note = (
            "未检测到需要标记的缺陷区域。"
            if not anomaly
            else "当前检测未生成有效缺陷 Mask，无法绘制缺陷定位框。"
        )
        _image_tile(
            "缺陷定位",
            "在原始图像上以红框标记最终异常 Mask 的连通区域。",
            marked_path or result.get("image_path"),
            missing_note=marked_note,
        )
        if not marked_path:
            st.caption(marked_note)
    with row2[1]:
        _image_tile(
            "缺陷 Mask",
            "显示最终阈值化后的疑似缺陷区域。",
            result.get("mask_path") or result.get("mask_ref"),
            missing_note="当前检测模式未生成缺陷 Mask。",
        )


def _empty_result() -> None:
    st.markdown(
        """
        <div class="aoi-panel">
          <div class="aoi-panel-title">等待检测结果</div>
          <div class="aoi-panel-subtitle">上传图片并点击开始检测后，这里会显示结论、分数和可视化证据。</div>
          <div class="aoi-empty-state">当前上传图片尚未执行检测；历史结果不会代替本次输入显示。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


configure_page("single")

runtime = sidebar_runtime_controls("single", include_memory_bank=True)
session_result = st.session_state.get("last_single_result")
if not runtime.get("memory_bank_id"):
    # Historical records stay in SQLite/statistics, but a clean runtime must
    # not present a deleted target domain as if it were the current result.
    st.session_state.pop("last_single_result", None)
    session_result = None
st.session_state.setdefault("single_detection_running", False)


@st.fragment(run_every="1.5s")
def _single_detection_workspace() -> None:
    """Keep upload, inference feedback and result rendering inside one local region."""
    header_state = render_detection_resource_monitor("single", render_status=False)
    header_runtime = dict(header_state["runtime"])
    header_result = st.session_state.get("last_single_result")
    _hero(header_runtime, header_result if isinstance(header_result, dict) else None)
    render_runtime_summary(header_runtime, header_result if isinstance(header_result, dict) else None)
    resource_state = render_detection_resource_monitor("single")
    runtime_values = dict(resource_state["runtime"])
    resource_ready = bool(resource_state["ready"])
    binding = _runtime_binding(dict(resource_state.get("snapshot") or {}))
    previous_binding = tuple(st.session_state.get(SINGLE_RUNTIME_BINDING_KEY) or ())
    if binding[0] and previous_binding and binding != previous_binding:
        st.session_state.pop("last_single_result", None)
        st.session_state[SINGLE_RUNTIME_CHANGED_KEY] = True
    if binding[0]:
        st.session_state[SINGLE_RUNTIME_BINDING_KEY] = binding

    if st.session_state.get(SINGLE_RUNTIME_CHANGED_KEY):
        st.info("运行配置已更新，请重新检测当前图片。")

    stored_result = st.session_state.get("last_single_result")
    result_value: dict[str, Any] | None = (
        dict(stored_result) if isinstance(stored_result, dict) else None
    )
    input_col, verdict_col = st.columns([0.78, 1.32])
    with input_col:
        st.markdown(
            """
            <div class="aoi-command-card">
              <div class="aoi-command-title">检测输入</div>
              <div class="aoi-command-desc">支持 png、jpg、jpeg、bmp、tif、tiff。上传后由后端创建隔离任务并调用真实模型。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        widget_upload = st.file_uploader(
            "上传待检测图片",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            key="single_selected_file",
        )
        uploaded = _retain_single_upload(widget_upload)
        current_input_hash = uploaded_input_hash(uploaded)
        if uploaded is None:
            # A browser refresh does not restore a file upload.  Keep the
            # session record for a possible fragment rerun, but do not show it
            # without a current input whose hash can be verified.
            result_value = None
        elif not result_matches_input(result_value, current_input_hash):
            # Selecting a genuinely different file is the only input action
            # that invalidates the visible result.  Feedback interactions keep
            # the same upload and therefore retain the complete result.
            result_value = None
            if isinstance(stored_result, dict) and current_input_hash:
                st.session_state.pop("last_single_result", None)
        initialize_detection_visualization_default("single_generate_visualizations")
        generate_visualizations = st.checkbox(
            "生成可视化结果",
            key="single_generate_visualizations",
            disabled=bool(st.session_state["single_detection_running"]),
            help="关闭后仍执行完整异常评分与判定，但不生成异常热力图、缺陷定位和缺陷 Mask。",
        )
        start, start_status = render_action_button_with_feedback(
            "开始检测",
            scope="single-detection::start",
            key="single_start_detection",
            button_type="primary",
            column_ratio=(0.72, 1.28),
            disabled=(
                uploaded is None
                or not resource_ready
                or bool(st.session_state["single_detection_running"])
            ),
        )
        st.caption("检测完成后会保存可追溯结果，并在当前页面提供反馈入口。")

        if start:
            st.session_state["single_detection_running"] = True
            update_action_feedback("single-detection::start", "info", "正在执行单图检测并生成结果...", start_status)
            try:
                with st.spinner("正在通过后端加载或复用模型并检测"):
                    result_value = run_image_inference(
                        dataset=runtime_values["dataset"],
                        category=runtime_values["category"],
                        config=runtime_values["config"],
                        uploaded_file=uploaded,
                        output_dir="outputs/demo_results/single_image",
                        execute=True,
                        device=runtime_values["device"],
                        memory_bank_id=runtime_values.get("memory_bank_id"),
                        generate_visualizations=generate_visualizations,
                    )
            finally:
                st.session_state["single_detection_running"] = False
            if not result_value.get("ok"):
                update_action_feedback(
                    "single-detection::start",
                    "error",
                    friendly_error_message(result_value.get("error"), "检测失败。"),
                    start_status,
                )
                if result_value.get("bridge"):
                    render_technical_details(result_value["bridge"], "查看调用技术详情")
                return
            update_action_feedback("single-detection::start", "success", "检测完成。", start_status)
            if not result_matches_input(result_value, current_input_hash):
                st.error("检测结果与当前上传图片不匹配，已阻止展示。请重新检测。")
                result_value = None
                st.session_state.pop("last_single_result", None)
            else:
                st.session_state["last_single_result"] = result_value
                st.session_state.pop(SINGLE_RUNTIME_CHANGED_KEY, None)

    with verdict_col:
        if result_value:
            _verdict_card(result_value)
            _score_strip(result_value)
        else:
            _empty_result()

    detail_col, gallery_col = st.columns([0.78, 1.32])
    with detail_col:
        if result_value:
            _detail_panel(result_value)
        else:
            st.markdown(
                """
                <div class="aoi-panel">
                  <div class="aoi-panel-title">检测详情</div>
                  <div class="aoi-panel-subtitle">结果生成后会显示数据集、类别、校准版本和保存路径。</div>
                  <div class="aoi-empty-state">暂无检测详情。</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with gallery_col:
        if result_value:
            _result_gallery(result_value)
        else:
            st.markdown(
                """
                <div class="aoi-panel">
                  <div class="aoi-panel-title">结果展示区</div>
                  <div class="aoi-panel-subtitle">检测后展示原始图像、红绿异常热力叠加、缺陷定位和缺陷 Mask。</div>
                  <div class="aoi-empty-state">当前暂无可视化文件。</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if result_value:
        render_single_timing(
            result_value.get("timing") or {},
            generate_visualizations=bool(result_value.get("generate_visualizations", True)),
            timing_mode=str(result_value.get("timing_mode") or ""),
        )

        raw_metadata = (result_value.get("raw_result") or {}).get("metadata") or {}
        feedback_state = render_feedback_panel(
            {
                "mode": "single",
                "task_id": result_value.get("job_id"),
                "sample_id": result_value.get("result_id"),
                "result_id": result_value.get("result_id"),
                "sample_name": raw_metadata.get("filename") or "当前检测图片",
                "prediction": result_value.get("pred_label"),
                "score": result_value.get("score"),
                "threshold": result_value.get("threshold"),
                "memory_bank_id": result_value.get("memory_bank_id"),
                "model_version": result_value.get("model_id"),
            },
            key_prefix=f"single_feedback_{str(result_value.get('result_id') or 'none')[:12]}",
            available_categories=[str(runtime_values.get("category") or "")],
        ) or {}
        if feedback_state.get("submitted"):
            # Feedback is part of the current view state, not a new detection
            # lifecycle.  Persist only feedback fields while retaining every
            # resource identity, score and visualization reference.
            updated_result = dict(result_value)
            updated_result["feedback_status"] = feedback_state.get("feedback_status")
            updated_result["feedback_assessment"] = feedback_state.get("assessment")
            updated_result["feedback_id"] = feedback_state.get("feedback_id")
            st.session_state["last_single_result"] = updated_result


_single_detection_workspace()
