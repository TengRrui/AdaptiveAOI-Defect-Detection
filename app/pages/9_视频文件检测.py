from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from performance_cache import api_client as client
from feedback_panel import render_video_feedback_section
from friendly_display import friendly_error_message, render_technical_details
from ui_common import (
    configure_page,
    format_value,
    initialize_detection_visualization_default,
    render_detection_resource_monitor,
    render_action_button_with_feedback,
    render_action_feedback,
    render_runtime_summary,
    runtime_header_items,
    sanitized_text,
    sidebar_runtime_controls,
    update_action_feedback,
)
from service.video_service import VIDEO_SUFFIXES, inspect_saved_video, inspect_selected_video, start_video_detection, video_job_action, video_job_status
from timing_panel import render_video_timing


VIDEO_RUNTIME_BINDING_KEY = "video_runtime_binding"
VIDEO_RUNTIME_CHANGED_KEY = "video_runtime_changed"


def _runtime_binding(snapshot: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(snapshot.get("runtime_revision") or 0),
        str(snapshot.get("profile_id") or ""),
        str(snapshot.get("memory_bank_id") or ""),
        str(snapshot.get("config_id") or ""),
        str(snapshot.get("model_id") or ""),
    )

def _number(value: Any, digits: int = 2, fallback: str = "-") -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return fallback


def _duration(seconds: Any) -> str:
    try:
        total = max(0, int(round(float(seconds))))
    except (TypeError, ValueError):
        return "-"
    return f"{total // 60:02d}:{total % 60:02d}"


def _artifact_url(value: str | None) -> str | None:
    return client().artifact_url(value)


def _video_hero(runtime: dict[str, Any], task_context: dict[str, Any] | None = None) -> None:
    badges = "".join(
        f'<span class="aoi-badge">{escape(str(label))}：{escape(str(value))}</span>'
        for label, value in runtime_header_items(runtime, task_context)
    )
    st.markdown(
        f"""
        <div class="aoi-workbench-hero">
          <div>
            <div class="aoi-workbench-kicker">Offline Video-file Inspection</div>
            <div class="aoi-workbench-title">视频文件检测</div>
            <div class="aoi-workbench-copy">
              选择本地视频后，软件通过 AdaptiveAOI 的公开 API 在后台执行逐帧检测，
              可按需生成带检测结果的视频和关键帧，并始终保留逐帧数值记录。界面只显示和轮询任务，不在 UI 主线程加载模型或执行推理。
            </div>
          </div>
          <div class="aoi-workbench-meta">
            {badges}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("当前模式对本地视频文件执行逐帧检测，实时性取决于硬件、模型和视频分辨率；不是摄像头实时检测、实时流检测或工业在线检测。")


def _show_video_metadata(upload: dict[str, Any]) -> None:
    metadata = upload.get("metadata") or {}
    rows = [
        ("文件名称", str(upload.get("original_filename") or metadata.get("file_name") or "-")),
        ("分辨率", f"{metadata.get('width', '-')} × {metadata.get('height', '-') }"),
        ("输入 FPS", _number(metadata.get("source_fps"), 2)),
        ("视频时长", _duration(metadata.get("duration_seconds"))),
        ("总帧数", str(metadata.get("reported_frame_count", "-"))),
        ("视频编码", str(metadata.get("fourcc") or "未识别")),
        ("浏览器预览", str(metadata.get("browser_preview_format") or "原始视频")),
        ("文件大小", f"{float(metadata.get('file_size_bytes', 0)) / 1024 / 1024:.2f} MB"),
        ("解码检查", "可以解码" if metadata.get("decodable") else "无法解码"),
    ]
    body = "".join(
        f'<div class="aoi-mini-row"><div class="aoi-mini-label">{escape(str(label))}</div><div class="aoi-mini-value">{escape(str(value))}</div></div>'
        for label, value in rows
    )
    st.markdown(
        f'<div class="aoi-panel"><div class="aoi-panel-title">视频输入信息</div>'
        f'<div class="aoi-panel-subtitle">输入已安全暂存，页面不会显示内部路径。</div>{body}</div>',
        unsafe_allow_html=True,
    )


def _show_live_task(task: dict[str, Any]) -> None:
    status = str(task.get("status") or "unknown")
    progress = task.get("progress") or {}
    summary = task.get("summary") or {}
    artifacts = task.get("artifacts") or {}
    generate_visualizations = bool(task.get("generate_visualizations", summary.get("generate_visualizations", True)))
    is_active = status in {"queued", "initializing", "running", "pausing", "paused", "resuming", "cancelling"}
    labels = {
        "queued": "等待中", "initializing": "初始化", "loading_model": "初始化", "running": "检测中",
        "pausing": "正在暂停", "paused": "已暂停", "resuming": "正在恢复", "cancelling": "正在取消",
        "stopping": "正在取消", "completed": "已完成", "cancelled": "已取消（结果不完整）", "failed": "失败",
        "interrupted": "服务重启中断",
    }
    st.markdown(f"### 当前任务：{labels.get(status, status)}")
    if status == "failed":
        error_payload = task.get("error")
        st.error(friendly_error_message(error_payload, "视频任务失败。"))
        if isinstance(error_payload, dict):
            render_technical_details(error_payload, "查看任务技术详情")
    elif status == "cancelled":
        st.warning("任务已安全停止；已经生成的结果保留在独立目录，但未标记为完整成功结果。")
    elif status == "completed":
        st.success(
            "视频文件已完整逐帧处理并生成正式结果。"
            if generate_visualizations
            else "视频文件已完整逐帧处理；本次按设置未生成可视化视频。"
        )
    elif status == "interrupted":
        st.warning("后端重启时该视频任务尚未完成，已安全标记为中断；可从原始视频重新运行。")
    else:
        st.caption("页面以约 1 秒频率轮询；预览帧可低于实际推理帧率，不影响后台逐帧推理。")

    fraction = progress.get("progress")
    st.progress(float(fraction) if isinstance(fraction, (float, int)) else 0.0, text=f"处理进度：{(float(fraction) * 100 if isinstance(fraction, (float, int)) else 0):.1f}%")
    metrics = st.columns(6)
    metrics[0].metric("当前帧 / 总帧", f"{max(0, int(progress.get('current_frame', -1))) + 1} / {progress.get('total_frames', '-')}")
    metrics[1].metric("当前时间 / 总时长", f"{_duration(float(progress.get('current_time_ms', 0)) / 1000)} / {_duration(progress.get('duration_seconds'))}")
    metrics[2].metric("当前异常分数", format_value(progress.get("anomaly_score")))
    metrics[3].metric("最高异常分数", format_value(progress.get("highest_anomaly_score")))
    metrics[4].metric("当前判定", str(progress.get("prediction") or "等待首帧"))
    metrics[5].metric("异常帧数", str(progress.get("anomaly_frames", 0)))
    extra = st.columns(6)
    extra[0].metric("当前处理 FPS", _number(progress.get("current_fps", task.get("current_fps")), 2))
    extra[1].metric("平均纯推理 FPS", _number(progress.get("pure_inference_fps"), 2))
    extra[2].metric("平均完整处理 FPS", _number(progress.get("full_pipeline_fps", task.get("complete_fps")), 2))
    eta_label = "已暂停" if status in {"pausing", "paused", "resuming"} else (
        "正在取消" if status == "cancelling" else _duration(progress.get("estimated_remaining_seconds"))
    )
    extra[3].metric("预计剩余时间", eta_label)
    extra[4].metric("当前阶段", labels.get(str(progress.get("stage") or status), labels.get(status, "处理中")))
    extra[5].metric("成功 / 失败帧", f"{progress.get('successful_frames', 0)} / {progress.get('failed_frames', 0)}")

    result_url = _artifact_url(artifacts.get("result_video_url"))
    controls = st.columns(4)
    action_scope = f"video-job::{task['job_id']}"
    action_status = st.empty()
    render_action_feedback(action_scope, action_status)
    if status == "running":
        if controls[0].button("暂停检测", key=f"video_pause_{task['job_id']}", width="stretch"):
            result = video_job_action(task["job_id"], "pause")
            if result.get("ok"):
                update_action_feedback(action_scope, "success", "已请求暂停。", action_status)
            else:
                update_action_feedback(action_scope, "error", str(result.get("error") or "暂停请求失败。"), action_status)
    elif status == "paused":
        if controls[0].button("恢复检测", key=f"video_resume_{task['job_id']}", type="primary", width="stretch"):
            result = video_job_action(task["job_id"], "resume")
            if result.get("ok"):
                update_action_feedback(action_scope, "success", "已请求恢复。", action_status)
            else:
                update_action_feedback(action_scope, "error", str(result.get("error") or "恢复请求失败。"), action_status)
    if is_active and controls[1].button("停止检测", key=f"video_stop_{task['job_id']}", width="stretch"):
        result = video_job_action(task["job_id"], "stop")
        if result.get("ok"):
            update_action_feedback(action_scope, "info", "已请求安全停止，当前帧完成后将保留已有部分结果。", action_status)
        else:
            update_action_feedback(action_scope, "error", str(result.get("error") or "停止请求失败。"), action_status)
    elif status == "interrupted" and controls[1].button("重新运行", key=f"video_rerun_{task['job_id']}", type="primary", width="stretch"):
        result = video_job_action(task["job_id"], "rerun")
        if result.get("ok"):
            created_task = dict(result.get("video_job") or {})
            st.session_state["video_last_task"] = created_task
            st.query_params["video_job"] = str(created_task.get("job_id") or "")
            update_action_feedback(action_scope, "success", "已创建新的视频检测任务。", action_status)
        else:
            update_action_feedback(action_scope, "error", str(result.get("error") or "重新运行失败。"), action_status)
    if controls[2].button("打开结果目录", key=f"video_open_{task['job_id']}", disabled=not bool(summary), width="stretch"):
        result = video_job_action(task["job_id"], "open_output")
        if result.get("ok"):
            update_action_feedback(action_scope, "success", "已在系统文件管理器中打开本次结果目录。", action_status)
        else:
            update_action_feedback(action_scope, "error", str(result.get("error") or "结果目录打开失败。"), action_status)
    if controls[3].button("使用系统播放器打开结果视频", key=f"video_open_result_{task['job_id']}", disabled=not bool(result_url), width="stretch"):
        result = video_job_action(task["job_id"], "open_result_video")
        if result.get("ok"):
            update_action_feedback(action_scope, "success", "已请求系统默认播放器打开本次检测结果视频。", action_status)
        else:
            update_action_feedback(action_scope, "error", str(result.get("error") or "结果视频打开失败。"), action_status)

    preview_url = _artifact_url(artifacts.get("preview_url"))
    visual_left, visual_right = st.columns(2)
    with visual_left:
        st.markdown("#### 检测过程预览（输出帧）")
        if preview_url:
            st.image(f"{preview_url}?job={task['job_id']}&frame={progress.get('current_frame', 0)}", caption="后台输出预览；显示频率受限，不代表跳过推理帧。", width="stretch")
        elif not generate_visualizations:
            st.info("本次任务未生成可视化结果；后台仍逐帧完成异常评分和判定。")
        else:
            st.info("等待首个成功处理帧。")
    with visual_right:
        st.markdown("#### 检测结果视频")
        if result_url:
            st.video(result_url)
            st.caption("若当前浏览器不支持该视频编码，可使用“使用系统播放器打开结果视频”。")
        elif not generate_visualizations:
            st.info("本次任务未生成可视化视频。逐帧 JSONL、CSV 和汇总仍会保存。")
        else:
            st.info("完成或安全停止后会在此提供已写入的结果视频。")

    if summary:
        st.markdown("#### 结果统计")
        totals = st.columns(6)
        totals[0].metric("成功帧", str(summary.get("successful_processed_frames", "-")))
        totals[1].metric("失败帧", str(summary.get("failed_frames", "-")))
        totals[2].metric("异常帧", str(summary.get("anomaly_frames", "-")))
        totals[3].metric("最高分 / 帧", f"{format_value(summary.get('highest_anomaly_score'))} / {summary.get('highest_anomaly_frame', '-')}")
        totals[4].metric("全流程 FPS", _number(summary.get("full_pipeline_fps", summary.get("processing_throughput_fps")), 2))
        totals[5].metric("达到输入 FPS（仅离线吞吐）", "是" if summary.get("reached_input_fps") else "否")
        st.caption(
            "逐帧数值记录与汇总已保存至本次独立任务目录；可使用“打开结果目录”查看。"
            if not generate_visualizations
            else "输入视频、结果视频与逐帧记录均已保存至本次独立任务目录；可使用“打开结果目录”查看。"
        )
        render_video_timing(summary)
        keyframes = list(task.get("keyframe_feedback_targets") or [])
        if keyframes:
            st.markdown("#### 关键帧判别摘要")
            st.dataframe(
                [
                    {
                        "关键帧": str(item.get("keyframe_name") or "代表帧"),
                        "帧号": item.get("frame_index"),
                        "时间戳": f"{float(item.get('timestamp_ms') or 0.0) / 1000.0:.2f} 秒",
                        "异常分数": format_value(item.get("score")),
                        "阈值": format_value(item.get("threshold")),
                        "判定": str(item.get("prediction") or "-"),
                    }
                    for item in keyframes
                ],
                width="stretch",
                hide_index=True,
            )

    selection = (task.get("manifest") or {}).get("selection") or {}
    render_video_feedback_section(
        task,
        available_categories=[str(selection.get("category") or "")],
    )


@st.fragment(run_every="1s")
def _video_job_region() -> None:
    cached_task = st.session_state.get("video_last_task") or {}
    job_id = str(cached_task.get("job_id") or st.query_params.get("video_job") or "")
    if not job_id:
        st.info("完成视频选择和解码检查后，点击“开始视频逐帧检测”。")
        return
    cached_status = str(cached_task.get("status") or "")
    if cached_status in {"completed", "cancelled", "failed"}:
        _show_live_task(cached_task)
        return
    response = video_job_status(job_id, heartbeat=True)
    if not response.get("ok"):
        st.error(friendly_error_message(response.get("error"), "无法读取视频任务状态。"))
        return
    task = response["video_job"]
    st.session_state["video_last_task"] = task
    _show_live_task(task)


@st.fragment
def _video_upload_region() -> None:
    """Upload, inspect and preview one video without rebuilding the page."""
    st.markdown("### 1. 选择并检查本地视频")
    st.caption("允许 .mp4、.avi、.mov、.mkv。选择后由 AdaptiveAOI 后端真实读取视频元数据并解码首帧；无法解码时不会启用开始按钮。")
    uploaded = st.file_uploader("选择本地视频", type=list(VIDEO_SUFFIXES), key="video_selected_file")
    upload_state = st.session_state.get("video_upload_state")
    restored_upload_id = str(st.query_params.get("video_upload") or "")
    fingerprint = f"{getattr(uploaded, 'name', '')}:{getattr(uploaded, 'size', '')}" if uploaded is not None else ""
    if uploaded is not None and st.session_state.get("video_upload_fingerprint") != fingerprint:
        upload_progress = st.progress(0, text="正在将视频分块上传到本地算法服务…")

        def _update_upload_progress(sent: int, total: int) -> None:
            ratio = min(1.0, sent / max(1, total))
            upload_progress.progress(ratio, text=f"正在分块上传：{ratio * 100:.0f}%")

        with st.spinner("正在由算法系统读取视频信息并验证可解码性..."):
            inspected = inspect_selected_video(uploaded, progress=_update_upload_progress)
        upload_progress.empty()
        st.session_state["video_upload_fingerprint"] = fingerprint
        st.session_state["video_upload_state"] = inspected
        upload_state = inspected
        if inspected.get("ok"):
            st.query_params["video_upload"] = str((inspected.get("upload") or {}).get("upload_id") or "")
    elif uploaded is None and restored_upload_id and not upload_state:
        upload_state = inspect_saved_video(restored_upload_id)
        st.session_state["video_upload_state"] = upload_state

    if upload_state:
        input_col, info_col = st.columns([1.65, 0.85])
        with input_col:
            st.markdown("#### 输入视频预览")
            if upload_state.get("ok"):
                preview_url = _artifact_url((upload_state.get("upload") or {}).get("preview_url"))
                if preview_url:
                    st.video(preview_url)
                    st.caption("播放器优先使用受控兼容预览；正式检测始终读取原始暂存视频，不改变源文件。")
                else:
                    st.warning("视频已通过解码检查，但暂时无法生成浏览器预览地址。")
        with info_col:
            if upload_state and upload_state.get("ok"):
                _show_video_metadata(upload_state["upload"])
            elif upload_state:
                st.error(sanitized_text(upload_state.get("error", "视频无法解码。")))
                st.caption("请检查视频编码或在系统状态页查看错误摘要。")


configure_page("video")
# Legacy versions kept this notice forever and rendered it on every task poll.
st.session_state.pop("video_action_notice", None)
runtime = sidebar_runtime_controls("video", include_memory_bank=True)
task_context = st.session_state.get("video_last_task") or None
if not runtime.get("memory_bank_id"):
    st.session_state.pop("video_last_task", None)
    task_context = None
@st.fragment(run_every="1.5s")
def _video_runtime_header() -> None:
    state = render_detection_resource_monitor("video", render_status=False)
    current_runtime = dict(state["runtime"])
    current_task = st.session_state.get("video_last_task") or None
    _video_hero(current_runtime, current_task)
    render_runtime_summary(current_runtime, current_task)


_video_runtime_header()
_video_upload_region()


@st.fragment(run_every="1.5s")
def _video_detection_controls() -> None:
    resource_state = render_detection_resource_monitor("video")
    current_runtime = dict(resource_state["runtime"])
    binding = _runtime_binding(dict(resource_state.get("snapshot") or {}))
    previous_binding = tuple(st.session_state.get(VIDEO_RUNTIME_BINDING_KEY) or ())
    if binding[0] and previous_binding and binding != previous_binding:
        st.session_state[VIDEO_RUNTIME_CHANGED_KEY] = True
    if binding[0]:
        st.session_state[VIDEO_RUNTIME_BINDING_KEY] = binding
    if st.session_state.get(VIDEO_RUNTIME_CHANGED_KEY):
        st.info("运行配置已更新，当前视频仍然保留，请重新检测。")
    if current_runtime.get("memory_bank_id") and not current_runtime.get("config_profile_id"):
        st.info("当前运行方案缺少受控配置标识，请前往“运行配置”选择已冻结方案。")

    upload_state = st.session_state.get("video_upload_state")
    st.markdown("### 2. 视频检测设置")
    options_col, action_col = st.columns([1.1, 0.9])
    active_task = st.session_state.get("video_last_task") or {}
    active_status = str(active_task.get("status") or "")
    active = active_status in {"queued", "initializing", "running", "pausing", "paused", "resuming", "cancelling"}
    with options_col:
        with st.form("video_detection_settings_form", border=False):
            setting_a, setting_b, setting_c = st.columns(3)
            frame_stride = setting_a.number_input(
                "frame_stride", min_value=1, max_value=120, value=1, step=1,
                help="1 表示逐帧处理；大于 1 会跳过算法处理帧，输出 FPS 会相应降低。",
            )
            frozen_target = bool(current_runtime.get("frozen"))
            override_threshold = setting_b.checkbox(
                "覆盖阈值",
                value=False,
                disabled=frozen_target,
                help="当前运行配置使用冻结阈值，正式测试模式不允许任务级覆盖。" if frozen_target else "仅未冻结的开发配置允许临时覆盖。",
            )
            threshold = setting_c.number_input("异常阈值", min_value=0.0, value=0.0, step=0.01, disabled=not override_threshold)
            initialize_detection_visualization_default("video_generate_visualizations")
            generate_visualizations = st.checkbox(
                "生成可视化结果",
                key="video_generate_visualizations",
                disabled=active,
                help="关闭后不执行帧绘制、融合和视频编码，但仍执行逐帧推理并保存 JSONL、CSV 和汇总。",
            )
            save_heatmap = st.checkbox("保存每帧热力图", value=False, disabled=active or not generate_visualizations)
            save_mask = st.checkbox("保存每帧掩码", value=False, disabled=active or not generate_visualizations)
            save_jsonl = st.checkbox("保存逐帧 JSONL 记录", value=True)
            st.caption("结果视频：将生成" if generate_visualizations else "结果视频：本次不生成")
            st.form_submit_button("应用视频检测设置", icon=":material/check:")
    with action_col:
        can_start = bool(
            upload_state
            and upload_state.get("ok")
            and not active
            and bool(resource_state["ready"])
        )
        submit_scope = "video-job::submit"
        submit_clicked, submit_status = render_action_button_with_feedback(
            "开始视频逐帧检测",
            scope=submit_scope,
            key="video_start_detection",
            button_type="primary",
            disabled=not can_start,
            column_ratio=(0.72, 1.28),
        )
        if submit_clicked:
            update_action_feedback(submit_scope, "info", "正在创建视频检测任务...", submit_status)
            response = start_video_detection(
                upload_state["upload"]["upload_id"],
                current_runtime,
                {
                    "threshold": float(threshold) if override_threshold else None,
                    "frame_stride": int(frame_stride),
                    "generate_visualizations": generate_visualizations,
                    "save_heatmap": save_heatmap,
                    "save_mask": save_mask,
                    "save_jsonl": save_jsonl,
                },
            )
            if response.get("ok"):
                st.session_state["video_last_task"] = response["video_job"]
                st.query_params["video_job"] = str(response["video_job"].get("job_id") or "")
                st.session_state.pop(VIDEO_RUNTIME_CHANGED_KEY, None)
                update_action_feedback(
                    submit_scope,
                    "success",
                    f"已提交视频任务：{response['video_job'].get('job_id')}",
                    submit_status,
                )
            else:
                update_action_feedback(
                    submit_scope,
                    "error",
                    friendly_error_message(response.get("error"), "视频任务无法启动。"),
                    submit_status,
                )
                st.caption("请在系统状态页查看错误摘要。")
        st.caption("同一时间只允许一个视频任务；重复点击不会启动冲突任务。检测、解码、模型加载与视频写入均在算法后端进行。")


_video_detection_controls()
st.markdown("---")
_video_job_region()
