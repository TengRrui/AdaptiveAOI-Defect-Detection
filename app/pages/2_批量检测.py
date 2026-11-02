from __future__ import annotations

import json
import logging
import os
from collections import Counter
from html import escape
from pathlib import Path
from statistics import mean
from typing import Any

import streamlit as st

from ui_common import (
    configure_page,
    display_heatmap_overlay,
    display_optional_image,
    format_value,
    initialize_detection_visualization_default,
    render_action_button_with_feedback,
    render_detection_resource_monitor,
    render_runtime_summary,
    runtime_header_items,
    sidebar_runtime_controls,
    update_action_feedback,
)

from feedback_panel import render_batch_feedback_section
from friendly_display import friendly_error_message, friendly_value, render_technical_details
from performance_cache import api_client as client
from service.batch_service import batch_inference_status, start_batch_inference
from timing_panel import render_batch_timing


BATCH_INPUT_KEY = "batch_input_dir"
BATCH_RECENT_DIRECTORY_KEY = "batch_last_browse_directory"
BATCH_DEFAULT_BROWSE_DIRECTORY = str(
    (Path(__file__).resolve().parents[2] / "data" / "demo_images" / "inference").resolve()
)
FOLDER_DIALOG_ID_KEY = "batch_folder_dialog_id"
FOLDER_DIALOG_ERROR_KEY = "batch_folder_dialog_error"
FOLDER_DIALOG_STATUS_FAILURES_KEY = "batch_folder_dialog_status_failures"
BATCH_SUCCESS_KEY = "batch_detection_success"
BATCH_RUNNING_KEY = "batch_detection_running"
BATCH_JOB_KEY = "batch_detection_job_id"
BATCH_RUNTIME_BINDING_KEY = "batch_runtime_binding"
BATCH_RUNTIME_CHANGED_KEY = "batch_runtime_changed"
LOGGER = logging.getLogger("adaptiveaoi.frontend.batch")

TASK_STATUS_LABELS = {
    "queued": "等待中",
    "preparing": "初始化",
    "running": "运行中",
    "cancelling": "取消中",
    "cancelled": "已取消",
    "completed": "已完成",
    "partial": "部分完成",
    "failed": "失败",
    "interrupted": "服务重启中断",
}


def _runtime_binding(snapshot: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(snapshot.get("runtime_revision") or 0),
        str(snapshot.get("profile_id") or ""),
        str(snapshot.get("memory_bank_id") or ""),
        str(snapshot.get("config_id") or ""),
        str(snapshot.get("model_id") or ""),
    )


def _html(value: Any) -> str:
    return escape(str(value if value not in (None, "") else "-"))


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_anomaly(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "anomaly", "abnormal", "defect", "ng", "异常"}:
        return True
    try:
        return int(float(text)) == 1
    except (TypeError, ValueError):
        return False


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = (len(values) - 1) * percentile / 100.0
    lower = int(idx)
    upper = min(lower + 1, len(values) - 1)
    if lower == upper:
        return float(values[lower])
    weight = idx - lower
    return float(values[lower] * (1 - weight) + values[upper] * weight)


def _format_ms(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return "-"
    if number >= 1000:
        return f"{number / 1000:.2f}s"
    return f"{number:.0f}ms"


def _format_percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}%"


def _visual_preview_row(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize result/manifest preview fields without loading image bytes."""
    return {
        **item,
        "original_image_url": client().artifact_url(item.get("original_image_url") or item.get("image_url")),
        "overlay_url": client().artifact_url(item.get("overlay_url") or item.get("result_image_url")),
        "heatmap_url": client().artifact_url(item.get("heatmap_url")),
        "marked_image_url": client().artifact_url(item.get("marked_image_url")),
        "mask_url": client().artifact_url(item.get("mask_url")),
    }


def _render_four_visuals(item: dict[str, Any], *, heading: str = "实时检测结果预览") -> None:
    """Render the shared 2x2 evidence layout for one completed sample."""
    item = _visual_preview_row(item)
    image_url = item.get("original_image_url")
    marked_url = item.get("marked_image_url")
    is_anomaly = bool(item.get("is_anomaly")) or _is_anomaly(item.get("prediction"))
    with st.container(border=True):
        st.markdown(f"### {heading}")
        file_name = str(item.get("image_name") or item.get("filename") or "当前样本")
        verdict = "缺陷" if is_anomaly else "正常"
        score = format_value(item.get("anomaly_score", item.get("score")))
        threshold = format_value(item.get("threshold"))
        st.caption(f"当前预览：{file_name} · 判定：{verdict} · 异常分数：{score} · 阈值：{threshold}")
        first = st.columns(2)
        with first[0]:
            st.markdown("**原始图像**")
            display_optional_image("原始图像", image_url)
        with first[1]:
            st.markdown("**热力图叠加**")
            display_heatmap_overlay(
                "热力图叠加",
                overlay_path=item.get("overlay_url") or item.get("result_image_url"),
                original_path=image_url,
                heatmap_path=item.get("heatmap_url"),
            )
        second = st.columns(2)
        with second[0]:
            st.markdown("**缺陷定位**")
            # Normal samples intentionally have no box; showing the original
            # image keeps the four-tile layout without inventing a defect.
            display_optional_image("缺陷定位", marked_url or image_url)
            if not marked_url:
                st.caption("未检测到需要标记的缺陷区域。" if not is_anomaly else "当前检测未生成有效缺陷 Mask，无法绘制定位框。")
        with second[1]:
            st.markdown("**缺陷 Mask**")
            if item.get("mask_url"):
                display_optional_image("缺陷 Mask", item.get("mask_url"))
            else:
                st.info("当前检测模式未生成缺陷 Mask。")


def _render_visual_preview_selector(items: list[dict[str, Any]], *, job_id: str, heading: str) -> None:
    """Show one sample's four artifacts with job-scoped follow/selection state."""
    if not items:
        st.info("等待首个已完成可视化样本。")
        return
    normalized = [_visual_preview_row(item) for item in items]
    by_id = {str(item.get("result_id") or index): item for index, item in enumerate(normalized)}
    ids = list(by_id)
    latest_id = ids[-1]
    follow_key = f"batch_preview_follow_{job_id}"
    selected_key = f"batch_preview_selected_{job_id}"
    choice_key = f"batch_preview_choice_{job_id}"
    st.session_state.setdefault(follow_key, True)
    st.session_state.setdefault(selected_key, latest_id)
    follow = st.checkbox("自动跟随最新检测结果", key=follow_key)
    if follow:
        st.session_state[selected_key] = latest_id
    elif str(st.session_state.get(selected_key) or "") not in by_id:
        st.session_state[selected_key] = latest_id
    selected_id = str(st.session_state.get(selected_key) or latest_id)
    if len(ids) > 1:
        # Keep the selectbox visually aligned with the job-scoped selection;
        # auto-follow must not leave a stale choice in the widget state.
        st.session_state[choice_key] = selected_id
        selected_id = st.selectbox(
            "当前预览样本",
            ids,
            index=max(0, ids.index(selected_id)) if selected_id in ids else len(ids) - 1,
            format_func=lambda value: str(by_id[str(value)].get("image_name") or value),
            key=choice_key,
        )
        if not follow:
            st.session_state[selected_key] = selected_id
    _render_four_visuals(by_id.get(str(selected_id), by_id[latest_id]), heading=heading)


def _hero(runtime: dict[str, Any], task_context: dict[str, Any] | None = None) -> None:
    badges = "".join(
        f'<span class="aoi-badge">{_html(label)}：{_html(value)}</span>'
        for label, value in runtime_header_items(runtime, task_context)
    )
    st.markdown(
        f"""
        <div class="aoi-workbench-hero">
          <div>
            <div class="aoi-workbench-kicker">Batch Quality Inspection</div>
            <div class="aoi-workbench-title">批量检测</div>
            <div class="aoi-workbench-copy">
              面向目录级检测任务，通过集成 API 对所选文件夹逐张调用真实模型，输出隔离 job 的 CSV、耗时统计和结果预览。
            </div>
          </div>
          <div class="aoi-workbench-meta">
            {badges}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(label: str, value: str, foot: str, icon: str) -> None:
    st.markdown(
        f"""
        <div class="aoi-kpi-card">
          <div class="aoi-kpi-top">
            <div class="aoi-kpi-label">{_html(label)}</div>
            <div class="aoi-kpi-icon">{_html(icon)}</div>
          </div>
          <div class="aoi-kpi-value">{_html(value)}</div>
          <div class="aoi-kpi-foot">{_html(foot)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    anomaly = sum(1 for row in rows if _is_anomaly(row.get("pred_label")))
    normal = total - anomaly
    total_ms = [value for value in (_to_float(row.get("total_ms")) for row in rows) if value is not None]
    return {
        "total": total,
        "anomaly": anomaly,
        "normal": normal,
        "pass_rate": normal / total * 100 if total else None,
        "mean_ms": float(mean(total_ms)) if total_ms else None,
        "p95_ms": _percentile(total_ms, 95),
    }


def _rows_for_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    return result.get("preview_rows", [])


def _distribution(rows: list[dict[str, Any]]) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for row in rows:
        if _is_anomaly(row.get("pred_label")):
            label = row.get("defect_type") or row.get("label_name") or "anomaly"
        else:
            label = "normal"
        counter[str(label).strip() or "unknown"] += 1
    return counter.most_common(8)


def _progress_panel(title: str, subtitle: str, items: list[tuple[str, int]]) -> None:
    if not items:
        st.markdown(
            f"""
            <div class="aoi-panel">
              <div class="aoi-panel-title">{_html(title)}</div>
              <div class="aoi-panel-subtitle">{_html(subtitle)}</div>
              <div class="aoi-empty-state">暂无可统计的批量结果。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    maximum = max(count for _, count in items) or 1
    body = "\n".join(
        f"""
        <div class="aoi-progress-row">
          <div class="aoi-progress-meta">
            <span>{_html(label)}</span>
            <span>{count}</span>
          </div>
          <div class="aoi-progress-track">
            <div class="aoi-progress-fill" style="width: {max(4, round(count / maximum * 100))}%;"></div>
          </div>
        </div>
        """
        for label, count in items
    )
    st.markdown(
        f"""
        <div class="aoi-panel">
          <div class="aoi-panel-title">{_html(title)}</div>
          <div class="aoi-panel-subtitle">{_html(subtitle)}</div>
          {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _task_panel(result: dict[str, Any] | None, input_dir: str, output_dir: str) -> None:
    source = "本次任务" if result and result.get("_source") != "existing_result_csv" else "已有结果"
    status = "已完成" if result else "等待执行"
    source_path = str(result.get("input_dir", input_dir) if result else input_dir or "")
    input_display = Path(source_path).name if source_path else "-"
    output_display = "已保存到本次受控任务目录" if result and result.get("export_csv") else output_dir or "-"
    rows = [
        ("任务状态", status),
        ("结果来源", source if result else "-"),
        ("输入目录", input_display),
        ("输出目录", output_display),
    ]
    body = "\n".join(
        f"""
        <div class="aoi-mini-row">
          <div class="aoi-mini-label">{_html(label)}</div>
          <div class="aoi-mini-value">{_html(value)}</div>
        </div>
        """
        for label, value in rows
    )
    st.markdown(
        f"""
        <div class="aoi-panel">
          <div class="aoi-panel-title">任务摘要</div>
          <div class="aoi-panel-subtitle">用于快速核对批量任务状态</div>
          {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _api_error_message(payload: dict[str, Any], fallback: str) -> str:
    error = payload.get("error") or {}
    return str(error.get("message") or fallback)


def _validate_input_directory(value: str) -> tuple[bool, str]:
    text = str(value or "").strip()
    if not text:
        return False, ""
    try:
        folder = Path(text).expanduser()
        if not folder.exists() or not folder.is_dir():
            return False, "输入图片文件夹不存在或不是有效目录。"
        with os.scandir(folder) as entries:
            next(entries, None)
    except PermissionError:
        return False, "当前没有权限访问该文件夹。"
    except OSError:
        return False, "输入图片文件夹当前无法访问。"
    return True, "文件夹可访问，可以开始批量检测。"


def _install_path_tooltip(full_path: str) -> None:
    st.html(
        f"""
        <script>
        (() => {{
          const input = window.parent.document.querySelector('.st-key-{BATCH_INPUT_KEY} input');
          if (input) input.title = {json.dumps(str(full_path or ""), ensure_ascii=False)};
        }})();
        </script>
        """,
        width="content",
        unsafe_allow_javascript=True,
    )


def _poll_folder_dialog() -> None:
    dialog_id = str(st.session_state.get(FOLDER_DIALOG_ID_KEY) or "")
    if not dialog_id:
        return
    response = client().folder_dialog_status(dialog_id, consume=True)
    if not response.get("success"):
        error = response.get("error") or {}
        retryable = bool(error.get("retryable"))
        failures = int(st.session_state.get(FOLDER_DIALOG_STATUS_FAILURES_KEY) or 0) + 1
        st.session_state[FOLDER_DIALOG_STATUS_FAILURES_KEY] = failures
        if retryable:
            st.session_state[FOLDER_DIALOG_ERROR_KEY] = (
                f"状态刷新稍慢，正在重试（第{failures}次）；本地选择窗口仍可继续操作。"
            )
            return
        st.session_state[FOLDER_DIALOG_ERROR_KEY] = _api_error_message(
            response,
            "文件夹选择状态读取失败，请重新打开。",
        )
        st.session_state.pop(FOLDER_DIALOG_ID_KEY, None)
        st.session_state.pop(FOLDER_DIALOG_STATUS_FAILURES_KEY, None)
        st.rerun(scope="fragment")
    dialog = response.get("folder_dialog") or {}
    returned_id = str(dialog.get("dialog_id") or dialog.get("request_id") or "")
    if returned_id and returned_id != dialog_id:
        st.session_state[FOLDER_DIALOG_ERROR_KEY] = "文件夹选择结果已过期，请重新打开选择窗口。"
        st.session_state.pop(FOLDER_DIALOG_ID_KEY, None)
        st.session_state.pop(FOLDER_DIALOG_STATUS_FAILURES_KEY, None)
        st.rerun(scope="fragment")
    st.session_state.pop(FOLDER_DIALOG_STATUS_FAILURES_KEY, None)
    status = str(dialog.get("status") or "")
    if status in {"starting", "open", "busy"}:
        st.session_state.pop(FOLDER_DIALOG_ERROR_KEY, None)
        return
    st.session_state.pop(FOLDER_DIALOG_ID_KEY, None)
    if status == "selected":
        selected_path = str(dialog.get("path") or "")
        if selected_path:
            st.session_state[BATCH_INPUT_KEY] = selected_path
            st.session_state[BATCH_RECENT_DIRECTORY_KEY] = selected_path
            st.session_state.pop(FOLDER_DIALOG_ERROR_KEY, None)
    elif status == "error":
        st.session_state[FOLDER_DIALOG_ERROR_KEY] = str(
            dialog.get("message") or "无法打开文件夹选择窗口，请稍后重试。"
        )
    elif status == "canceled":
        st.session_state.pop(FOLDER_DIALOG_ERROR_KEY, None)
    st.rerun(scope="fragment")


def _run_current_batch(
    runtime: dict[str, Any], input_dir: str, output_dir: str, generate_visualizations: bool,
    *, status_target: Any | None = None,
) -> None:
    try:
        result = start_batch_inference(
            dataset=runtime["dataset"],
            category=runtime["category"],
            config=runtime["config"],
            input_dir=input_dir,
            device=runtime["device"],
            memory_bank_id=runtime.get("memory_bank_id"),
            generate_visualizations=generate_visualizations,
        )
    except Exception:
        LOGGER.exception("batch_job_start_failed")
        message = "批量检测未能启动，请检查文件夹权限和本地服务状态。"
        if status_target is not None:
            update_action_feedback("batch-detection::start", "error", message, status_target)
        else:
            st.error(message)
        return
    if not result.get("ok"):
        message = friendly_error_message(result.get("error"), "批量检测失败。")
        if status_target is not None:
            update_action_feedback("batch-detection::start", "error", message, status_target)
        else:
            st.error(message)
        if result.get("bridge"):
            render_technical_details(result["bridge"], "查看调用技术详情")
        return
    job_id = str(result.get("job_id") or "")
    if not job_id:
        message = "后端未返回有效任务标识，批量任务没有启动。"
        if status_target is not None:
            update_action_feedback("batch-detection::start", "error", message, status_target)
        else:
            st.error(message)
        return
    try:
        resolved_input = str(Path(input_dir).expanduser().resolve())
        st.session_state[BATCH_RECENT_DIRECTORY_KEY] = resolved_input
        remembered = client().remember_folder_dialog_directory(
            BATCH_RECENT_DIRECTORY_KEY,
            resolved_input,
        )
        if not remembered.get("success"):
            LOGGER.warning("batch_recent_directory_persist_failed: %s", remembered.get("error"))
    except (OSError, RuntimeError):
        # The directory was already validated by the task API.  A convenience
        # preference failure must never invalidate a successfully started job.
        LOGGER.warning("batch_recent_directory_resolve_failed", exc_info=True)
    st.session_state[BATCH_JOB_KEY] = job_id
    st.session_state[BATCH_RUNNING_KEY] = True
    st.session_state["batch_detection_input_snapshot"] = input_dir
    st.query_params["batch_job"] = job_id
    if status_target is not None:
        update_action_feedback("batch-detection::start", "success", "批量任务已提交，正在后台处理。", status_target)


@st.fragment(run_every=0.8)
def _render_batch_job() -> None:
    job_id = str(st.session_state.get(BATCH_JOB_KEY) or "")
    if not job_id:
        return
    response = batch_inference_status(job_id)
    if not response.get("ok"):
        st.warning(friendly_error_message(response.get("error"), "批量任务状态暂时无法读取，系统会继续重试。"))
        return
    job = response.get("job") or {}
    status = str(job.get("status") or "queued")
    total = int(job.get("input_count") or 0)
    processed = int(job.get("processed") or 0)
    normal = int(job.get("normal_count") or 0)
    anomaly = int(job.get("anomaly_count") or 0)
    failed = int(job.get("failure_count") or 0)
    progress = min(1.0, max(0.0, float(job.get("progress") or 0.0)))
    label = TASK_STATUS_LABELS.get(status, "处理中")
    st.markdown("### 批量任务实时进度")
    st.progress(progress, text=f"{label} · 已处理 {processed}/{total or '-'}")
    first = st.columns(6)
    values = [
        ("总图片数", total or "-"),
        ("已完成", processed),
        ("正常", normal),
        ("缺陷", anomaly),
        ("失败", failed),
        ("任务状态", label),
    ]
    for column, (title, value) in zip(first, values):
        column.metric(title, value)
    second = st.columns(5)
    second[0].metric("当前单张", _format_ms(job.get("current_item_ms")))
    second[1].metric("平均耗时", _format_ms(job.get("average_ms")))
    second[2].metric("当前吞吐率", f"{float(job.get('throughput') or 0.0):.2f} 张/秒")
    eta_value = job.get("eta_seconds")
    eta_label = "正在取消" if status == "cancelling" else (
        "已结束" if status in {"completed", "partial", "cancelled", "failed", "interrupted"}
        else (f"{float(eta_value):.1f} 秒" if eta_value is not None else "计算中")
    )
    second[3].metric("预计剩余", eta_label)
    second[4].metric("可视化", "生成" if job.get("generate_visualizations") else "不生成")
    current_file = str(job.get("current_file") or "")
    st.caption(f"当前文件：{Path(current_file).name if current_file else '等待下一张图片'}")
    visuals_enabled = bool(job.get("generate_visualizations", (job.get("selection") or {}).get("save_visuals", False)))
    if visuals_enabled:
        live_items = list(job.get("visualized_results") or [])
        latest = job.get("latest_visualized_result")
        if latest and not any(str(item.get("result_id")) == str(latest.get("result_id")) for item in live_items):
            live_items.append(latest)
        _render_visual_preview_selector(live_items, job_id=job_id, heading="实时检测结果预览")
    else:
        st.info("当前任务未开启结果可视化。")
    if status in {"queued", "preparing", "running"}:
        if st.button("取消批量任务", key=f"cancel_batch_{job_id}", icon=":material/stop_circle:"):
            cancel = client().cancel_job(job_id)
            if cancel.get("success"):
                st.info("已提交取消请求，将在当前图片处理结束后安全停止。")
            else:
                st.warning(_api_error_message(cancel, "取消请求提交失败。"))
        return
    if status == "interrupted":
        st.warning("后端重启时该批量任务尚未完成，系统已将其标记为中断，不会继续显示为运行中。")
        if st.button("重新运行批量任务", key=f"rerun_batch_{job_id}", icon=":material/replay:"):
            rerun = client().rerun_job(job_id)
            if rerun.get("success"):
                new_job_id = str(rerun.get("job_id") or "")
                st.session_state[BATCH_JOB_KEY] = new_job_id
                st.session_state[BATCH_RUNNING_KEY] = True
                st.query_params["batch_job"] = new_job_id
                st.rerun(scope="fragment")
            else:
                st.error(_api_error_message(rerun, "批量任务重新运行失败。"))
        return
    result = response.get("result")
    if not result:
        st.info("检测已结束，正在整理结果文件……")
        return
    st.session_state[BATCH_RUNNING_KEY] = False
    if result:
        result["input_dir"] = str(st.session_state.get("batch_detection_input_snapshot") or "")
        st.session_state["last_batch_result"] = result
        st.session_state[BATCH_SUCCESS_KEY] = status in {"completed", "partial"}
    st.session_state.pop(BATCH_JOB_KEY, None)
    if st.query_params.get("batch_job") == job_id:
        del st.query_params["batch_job"]
    st.rerun(scope="app")


def _render_batch_input_controls(runtime: dict[str, Any], output_dir: str, *, poll_dialog: bool) -> None:
    if poll_dialog:
        _poll_folder_dialog()

    dialog_open = bool(st.session_state.get(FOLDER_DIALOG_ID_KEY))
    current_path = str(st.session_state.get(BATCH_INPUT_KEY) or "")
    with st.container(key="batch_folder_picker_row"):
        path_col, picker_col = st.columns([3.7, 1.65], gap="small", vertical_alignment="bottom")
        with path_col:
            input_dir = st.text_input(
                "输入图片文件夹路径",
                key=BATCH_INPUT_KEY,
                placeholder="输入文件夹路径，或使用“选择文件夹”按钮浏览目录",
                help=f"完整路径：{current_path}" if current_path else "支持手动输入、粘贴或使用文件夹选择窗口。",
            )
        with picker_col:
            choose_folder = st.button(
                "选择文件夹",
                icon=":material/folder_open:",
                key="batch_choose_folder",
                disabled=dialog_open,
                width="stretch",
            )
    _install_path_tooltip(input_dir)

    if choose_folder:
        st.session_state.pop(FOLDER_DIALOG_ERROR_KEY, None)
        st.session_state.pop(FOLDER_DIALOG_STATUS_FAILURES_KEY, None)
        initial_directory = str(
            input_dir
            or st.session_state.get(BATCH_RECENT_DIRECTORY_KEY)
            or ""
        )
        response = client().start_folder_dialog(
            initial_directory,
            recent_key=BATCH_RECENT_DIRECTORY_KEY,
            fallback_dirs=[
                str(st.session_state.get("batch_detection_input_snapshot") or ""),
                BATCH_DEFAULT_BROWSE_DIRECTORY,
            ],
        )
        if not response.get("success"):
            st.session_state[FOLDER_DIALOG_ERROR_KEY] = _api_error_message(
                response,
                "无法打开文件夹选择窗口，请确认本地服务已启动。",
            )
        else:
            dialog = response.get("folder_dialog") or {}
            if dialog.get("status") == "busy":
                st.session_state[FOLDER_DIALOG_ERROR_KEY] = "已有文件夹选择窗口，请先完成或关闭。"
            elif dialog.get("dialog_id"):
                st.session_state[FOLDER_DIALOG_ID_KEY] = str(dialog["dialog_id"])
        st.rerun(scope="fragment")

    dialog_error = str(st.session_state.get(FOLDER_DIALOG_ERROR_KEY) or "")
    if dialog_open:
        st.info("Windows 文件夹选择窗口已打开，请完成选择或取消。")
    elif dialog_error:
        st.warning(dialog_error)

    is_valid, validation_message = _validate_input_directory(input_dir)
    if input_dir.strip():
        if is_valid:
            st.caption(validation_message)
        else:
            st.warning(validation_message)

    if st.session_state.pop(BATCH_SUCCESS_KEY, False):
        st.success("批量检测完成")
    initialize_detection_visualization_default("batch_generate_visualizations")
    generate_visualizations = st.checkbox(
        "生成可视化结果",
        key="batch_generate_visualizations",
        disabled=bool(st.session_state.get(BATCH_RUNNING_KEY, False)),
        help="该开关作用于整批任务；关闭后仍保存逐样本判定、分数和耗时，但不生成可视化图片。",
    )
    start, start_status = render_action_button_with_feedback(
        "批量检测",
        scope="batch-detection::start",
        key="batch_start_detection",
        button_type="primary",
        column_ratio=(0.72, 1.28),
        disabled=(
            dialog_open
            or not is_valid
            or not runtime.get("_ui_resource_ready", False)
            or bool(st.session_state.get(BATCH_RUNNING_KEY, False))
        ),
    )
    st.caption("仅展示本会话中通过后端实际执行的批量任务；不会把历史 manifest 结果冒充为当前文件夹结果。")
    if start:
        update_action_feedback("batch-detection::start", "info", "正在创建批量检测任务...", start_status)
        _run_current_batch(
            runtime,
            input_dir,
            output_dir,
            generate_visualizations,
            status_target=start_status,
        )


def _render_pending_batch_input_controls(runtime: dict[str, Any], output_dir: str) -> None:
    _render_batch_input_controls(runtime, output_dir, poll_dialog=True)


@st.fragment(run_every="1.5s")
def _render_batch_control_region(output_dir: str) -> None:
    """Keep path entry, picker launch and lightweight validation local."""
    resource_state = render_detection_resource_monitor("batch")
    runtime = dict(resource_state["runtime"])
    runtime["_ui_resource_ready"] = bool(resource_state["ready"])
    binding = _runtime_binding(dict(resource_state.get("snapshot") or {}))
    previous_binding = tuple(st.session_state.get(BATCH_RUNTIME_BINDING_KEY) or ())
    if binding[0] and previous_binding and binding != previous_binding:
        st.session_state.pop("last_batch_result", None)
        st.session_state[BATCH_RUNTIME_CHANGED_KEY] = True
    if binding[0]:
        st.session_state[BATCH_RUNTIME_BINDING_KEY] = binding
    if st.session_state.get(BATCH_RUNTIME_CHANGED_KEY):
        st.info("运行配置已更新，批量输入路径与筛选状态已保留，请重新检测。")
    st.markdown(
        """
        <div class="aoi-command-card">
          <div class="aoi-command-title">批量任务输入</div>
          <div class="aoi-command-desc">填写数据根目录或上传根目录下的图片文件夹。后端为本次任务生成独立结果目录，不覆盖实验结果。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state.get(FOLDER_DIALOG_ID_KEY):
        _render_pending_batch_input_controls(runtime, output_dir)
    else:
        _render_batch_input_controls(runtime, output_dir, poll_dialog=False)


configure_page("batch")

runtime = sidebar_runtime_controls("batch", include_memory_bank=True)
current_result = st.session_state.get("last_batch_result")
if not runtime.get("memory_bank_id"):
    st.session_state.pop("last_batch_result", None)
    current_result = None
st.session_state.setdefault(BATCH_INPUT_KEY, "")
st.session_state.setdefault(BATCH_RUNNING_KEY, False)
if not st.session_state.get(BATCH_JOB_KEY):
    restored_job = str(st.query_params.get("batch_job") or "")
    if restored_job.startswith("batch_"):
        st.session_state[BATCH_JOB_KEY] = restored_job
        st.session_state[BATCH_RUNNING_KEY] = True
output_dir = "后端 job artifacts"


@st.fragment(run_every="1.5s")
def _render_batch_runtime_header() -> None:
    state = render_detection_resource_monitor("batch", render_status=False)
    current_runtime = dict(state["runtime"])
    result = st.session_state.get("last_batch_result")
    _hero(current_runtime, result if isinstance(result, dict) else None)
    render_runtime_summary(current_runtime, result if isinstance(result, dict) else None)


_render_batch_runtime_header()
control_col, summary_col = st.columns([0.82, 1.18])
with control_col:
    _render_batch_control_region(output_dir)

input_dir = str(st.session_state.get(BATCH_INPUT_KEY) or "")
current_result = st.session_state.get("last_batch_result")
with summary_col:
    _task_panel(current_result, input_dir, output_dir)

_render_batch_job()

if current_result:
    all_rows = _rows_for_result(current_result)
    rows = all_rows[:20]
    summary = _summarize_rows(all_rows)
    if bool(current_result.get("generate_visualizations", True)):
        _render_visual_preview_selector(
            all_rows,
            job_id=str(current_result.get("job_id") or current_result.get("batch_id") or "completed"),
            heading="检测结果预览",
        )
    else:
        st.info("当前任务未开启结果可视化。")
    st.markdown(
        """
        <div class="aoi-section-header">
          <div>
            <div class="aoi-section-title">批量结果摘要</div>
          <div class="aoi-section-desc">统计来自本次后端任务返回的真实检测记录，不补造检测数据。</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    timing = current_result.get("timing") or {}
    total_ms = _to_float(timing.get("task_end_to_end_ms") or timing.get("end_to_end_ms"))
    throughput = _to_float(
        timing.get("throughput_images_per_second")
        or timing.get("throughput_items_per_second")
    )
    failed_count = len(current_result.get("errors") or [])
    metric_cols = st.columns(6)
    cards = [
        ("总样本", str(current_result.get("row_count", 0)), "本次目录扫描结果", "ALL"),
        ("正常", str(summary["normal"]), "判定为正常", "OK"),
        ("缺陷", str(summary["anomaly"]), "判定为异常", "NG"),
        ("失败", str(failed_count), "读取或推理失败", "ERR"),
        ("总耗时", _format_ms(total_ms), "完整批量任务", "TIME"),
        ("吞吐率", f"{throughput:.2f} 张/秒" if throughput is not None else "未统计", "完整任务平均", "RATE"),
    ]
    for column, card in zip(metric_cols, cards):
        with column:
            _metric_card(*card)

    left, right = st.columns([1.05, 0.95])
    with left:
        _progress_panel(
            "样本结果分布",
            "汇总正常、缺陷与处理失败样本",
            [("正常", summary["normal"]), ("缺陷", summary["anomaly"]), ("失败", failed_count)],
        )
    with right:
        st.markdown(
            f"""
            <div class="aoi-panel">
              <div class="aoi-panel-title">结果导出与文件管理</div>
              <div class="aoi-panel-subtitle">结果保存位置：已保存到本次受控任务目录</div>
              <div class="aoi-mini-row"><div class="aoi-mini-label">明细记录</div><div class="aoi-mini-value">{len(all_rows)} 条</div></div>
              <div class="aoi-mini-row"><div class="aoi-mini-label">可视化结果</div><div class="aoi-mini-value">{'已生成' if current_result.get('generate_visualizations', True) else '未生成'}</div></div>
              <div class="aoi-mini-row"><div class="aoi-mini-label">失败记录</div><div class="aoi-mini-value">{failed_count} 条</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        export_url = str(current_result.get("export_csv") or "")
        if export_url:
            st.link_button("打开检测明细", export_url, icon=":material/table_view:", width="stretch")

    _progress_panel("缺陷类型分布", "仅对已判定缺陷的样本按真实类别字段汇总", _distribution(rows))
    st.markdown(
        """
        <div class="aoi-table-panel">
          <div class="aoi-panel-title">前 20 行结果预览</div>
          <div class="aoi-panel-subtitle">默认只保留常用字段；阶段耗时在下方统一统计中查看。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    display_rows = []
    for row in rows:
        display_rows.append({
            "图片名称": friendly_value(row.get("image_name"), "filename"),
            "检测结果": friendly_value(row.get("prediction"), "prediction"),
            "异常分数": friendly_value(row.get("score"), "score"),
            "检测阈值": friendly_value(row.get("threshold"), "threshold"),
            "端到端耗时": friendly_value(row.get("total_ms"), "total_ms"),
            "可视化结果": "已生成" if bool(row.get("generate_visualizations", True)) else "未生成",
        })
    st.dataframe(display_rows, width="stretch", hide_index=True)
    if not bool(current_result.get("generate_visualizations", True)):
        st.info("本次批量任务未生成可视化结果；逐样本数值判定、耗时统计和结果表已正常保存。")
    render_batch_timing(
        current_result.get("timing") or {},
        generate_visualizations=bool(current_result.get("generate_visualizations", True)),
        failed=len(current_result.get("errors") or []),
        timing_mode=str(current_result.get("timing_mode") or ""),
    )
    render_batch_feedback_section(
        task_id=str(current_result.get("job_id") or current_result.get("batch_id") or ""),
        rows=all_rows,
        available_categories=[str(runtime.get("category") or "")],
    )
else:
    st.markdown(
        """
        <div class="aoi-panel">
          <div class="aoi-panel-title">暂无批量结果</div>
          <div class="aoi-panel-subtitle">填写输入目录并执行批量检测后，这里会显示结果表格、统计指标和导出路径。</div>
          <div class="aoi-empty-state">填写允许的输入目录并执行批量检测后，这里会展示本次任务的真实检测记录。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
