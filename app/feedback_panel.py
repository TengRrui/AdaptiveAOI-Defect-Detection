from __future__ import annotations

from typing import Any

import streamlit as st

from performance_cache import api_client as client
from friendly_display import friendly_error_message
from selection_state import (
    checkbox_changed,
    checkbox_key,
    clear_selection,
    remove_selected_ids,
    selected_ids,
    set_page_selection,
    sync_checkbox_states,
)
from service.feedback_service import (
    detection_task_items,
    existing_feedback,
    submit_bulk_feedback,
    submit_feedback,
    task_feedback,
    video_feedback_interval,
)
from ui_common import render_action_button_with_feedback, update_action_feedback


ASSESSMENTS = {
    "correct": "结果正确",
    "incorrect": "结果有误",
    "uncertain": "不确定",
}
CORRECTED_LABELS = {
    "keep": "保持模型结果",
    "normal": "正常",
    "defect": "缺陷",
}
ISSUE_TAGS = {
    "score_high": "异常分数偏高",
    "score_low": "异常分数偏低",
    "region_inaccurate": "缺陷区域不准确",
    "missed_detection": "漏检",
    "false_alarm": "误检",
    "other": "其他说明",
}
STATUS_LABELS = {
    "saved": "已保存",
    "confirmed": "已确认",
    "corrected": "已修正",
    "pending_application": "待应用",
    "applied": "已应用",
    "application_failed": "应用失败",
    "pending_review": "待审核",
    "in_review": "审核中",
    "collected": "已采集",
    "legacy_unresolved": "历史待确认",
    "record_only": "只读测试仅记录",
    "conflict": "标签冲突",
    "queued": "已排队",
    "optimizing": "优化中",
    "validation_failed": "验证失败",
    "ready_to_apply": "已生成，待应用",
    "rolled_back": "已回滚",
    "invalid_source": "源文件失效",
}
KEYFRAME_LABELS = {
    "first_frame": "首帧",
    "normal_representative": "正常代表帧",
    "first_anomaly": "首个异常帧",
    "highest_anomaly_score": "最高异常分数帧",
    "last_frame": "末帧",
}


def _refresh_after_batch_feedback_submission(failures: list[dict[str, Any]]) -> bool:
    """Rebuild the app tree after a complete bulk submission.

    The batch feedback browser is a fragment.  After a successful submission,
    an app-scoped rerun removes the fragment's previous widget tree before the
    next filtered selection is rendered.  Partial failures stay in the current
    fragment so their selected rows and detailed reasons remain visible.
    """

    if failures:
        return False
    st.rerun(scope="app")
    return True


def _feedback_status(record: dict[str, Any] | None) -> str:
    if not record:
        return "未反馈"
    return STATUS_LABELS.get(str(record.get("status") or ""), "已保存")


def _prediction_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"1", "anomaly", "abnormal", "defect", "ng"}:
        return "缺陷"
    if text in {"0", "normal", "ok"}:
        return "正常"
    return str(value or "-")


def _prediction_human_label(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"0", "normal", "ok", "good", "negative"}:
        return "normal"
    if text in {"1", "anomaly", "abnormal", "defect", "ng", "positive"}:
        return "defect"
    return None


def _time_label(timestamp_ms: Any) -> str:
    try:
        seconds = max(0.0, float(timestamp_ms) / 1000.0)
    except (TypeError, ValueError):
        return "-"
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


def _categories(values: list[str] | None, existing: dict[str, Any] | None) -> list[str]:
    items = [str(value).strip() for value in (values or []) if str(value).strip()]
    old = str((existing or {}).get("corrected_defect_type") or "").strip()
    if old:
        items.append(old)
    return list(dict.fromkeys(items))


def _render_editor(
    context: dict[str, Any],
    *,
    key_prefix: str,
    available_categories: list[str] | None,
    existing: dict[str, Any] | None,
) -> dict[str, Any]:
    st.info("当前检测使用冻结配置。反馈将被保存，不会自动修改当前模型、记忆库或阈值。")
    status = _feedback_status(existing)
    editor_state = {
        "submitted": bool(existing),
        "feedback_status": str((existing or {}).get("status") or ""),
        "assessment": str((existing or {}).get("assessment") or ""),
        "feedback_id": str((existing or {}).get("feedback_id") or ""),
    }
    if existing:
        if str(existing.get("assessment") or "") == "correct":
            st.success("✅ 已记录：检测结果正确")
        else:
            st.success("✅ 反馈已提交，等待人工审核")
    if context.get("mode") == "video":
        st.caption(
            f"反馈对象：帧 {context.get('frame_index', '-')} · 时间 {_time_label(context.get('timestamp_ms'))} · 当前状态：{status}"
        )
    else:
        st.caption(f"反馈对象：{context.get('sample_name') or '当前检测结果'} · 当前状态：{status}")

    assessment_options = list(ASSESSMENTS)
    assessment_default = str((existing or {}).get("assessment") or "uncertain")
    assessment = st.radio(
        "当前检测结果是否正确",
        assessment_options,
        index=assessment_options.index(assessment_default) if assessment_default in assessment_options else 2,
        format_func=ASSESSMENTS.get,
        horizontal=True,
        key=f"{key_prefix}_assessment",
    )
    video_feedback = str(context.get("mode") or "") == "video"
    correction_default = str((existing or {}).get("corrected_label") or "keep")
    corrected_label: str | None
    if video_feedback and assessment == "correct":
        corrected_label = _prediction_human_label(context.get("prediction"))
        if corrected_label:
            st.text_input(
                "人工判断标签",
                value=CORRECTED_LABELS[corrected_label],
                disabled=True,
                help="选择“结果正确”时，人工标签自动确认当前帧的原始判定。",
                key=f"{key_prefix}_confirmed_label",
            )
        else:
            st.error("当前帧的原始判定无法解析，请改选“结果有误”并明确人工标签。")
    elif video_feedback:
        video_label_options = ["normal", "defect"]
        selected_index = (
            video_label_options.index(correction_default)
            if correction_default in video_label_options
            else None
        )
        corrected_label = st.radio(
            "人工判断标签",
            video_label_options,
            index=selected_index,
            format_func=CORRECTED_LABELS.get,
            horizontal=True,
            help="视频反馈必须明确人工判断为正常或缺陷；不同标签请分别提交。",
            key=f"{key_prefix}_corrected_label",
        )
        if corrected_label is None:
            st.caption("请选择“正常”或“缺陷”后再提交。")
    else:
        correction_options = list(CORRECTED_LABELS)
        corrected_label = st.selectbox(
            "用户修正结果",
            correction_options,
            index=correction_options.index(correction_default) if correction_default in correction_options else 0,
            format_func=CORRECTED_LABELS.get,
            key=f"{key_prefix}_corrected_label",
        )
    defect_type = None
    if corrected_label == "defect":
        category_options = _categories(available_categories, existing)
        if category_options:
            defect_type = st.selectbox(
                "缺陷类别修正",
                category_options,
                help="类别值按当前项目原值提交，不做翻译或改写。",
                key=f"{key_prefix}_defect_type",
            )
        else:
            st.caption("当前项目未提供可选缺陷类别，反馈将只保存“缺陷”修正。")

    existing_tags = [tag for tag in (existing or {}).get("issue_tags", []) if tag in ISSUE_TAGS]
    issue_tags = st.multiselect(
        "反馈内容（可选）",
        list(ISSUE_TAGS),
        default=existing_tags,
        format_func=ISSUE_TAGS.get,
        key=f"{key_prefix}_issue_tags",
    )
    note = st.text_area(
        "备注（可选）",
        value=str((existing or {}).get("note") or ""),
        max_chars=1000,
        key=f"{key_prefix}_note",
    )
    values = {
        "assessment": assessment,
        "corrected_label": corrected_label,
        "corrected_defect_type": defect_type,
        "issue_tags": issue_tags,
        "note": note,
    }
    busy_key = f"{key_prefix}_submitting"
    st.session_state.setdefault(busy_key, False)
    action_scope = f"feedback-submit::{key_prefix}"
    save_clicked, action_status = render_action_button_with_feedback(
        "更新反馈" if existing else "提交反馈",
        scope=action_scope,
        key=f"{key_prefix}_save",
        button_type="primary",
        disabled=bool(st.session_state.get(busy_key)) or (video_feedback and corrected_label not in {"normal", "defect"}),
        help_text="提交只保存人工反馈，不会自动修改当前检测配置。",
    )
    if save_clicked:
        values["application_mode"] = "save_only"
        st.session_state[busy_key] = True
        update_action_feedback(action_scope, "info", "正在保存反馈...", action_status)
        try:
            result = submit_feedback(context, values)
        finally:
            st.session_state[busy_key] = False
        if result.get("ok"):
            feedback = result.get("feedback") or {}
            candidate = result.get("candidate") or {}
            editor_state = {
                "submitted": True,
                "feedback_status": str(candidate.get("status") or feedback.get("status") or "saved"),
                "assessment": str(values.get("assessment") or ""),
                "feedback_id": str(feedback.get("feedback_id") or candidate.get("feedback_id") or ""),
            }
            if str(values.get("assessment") or "") == "correct":
                update_action_feedback(action_scope, "success", "已记录：检测结果正确。", action_status)
            elif str(candidate.get("status") or "") == "record_only":
                update_action_feedback(action_scope, "info", "反馈已提交，等待人工审核；当前为只读记录。", action_status)
            else:
                update_action_feedback(action_scope, "success", "反馈已提交，等待人工审核。", action_status)
            if st.button("前往反馈优化中心", key=f"{key_prefix}_open_optimization"):
                st.switch_page("pages/11_反馈优化中心.py")
        else:
            update_action_feedback(
                action_scope,
                "error",
                friendly_error_message(result.get("error"), "反馈保存失败。"),
                action_status,
            )
    st.caption("如需应用反馈，请在“反馈优化中心”完成审核、离线验证，并显式应用新版本。")
    return editor_state


def render_feedback_panel(
    context: dict[str, Any],
    *,
    key_prefix: str,
    available_categories: list[str] | None = None,
    expanded: bool = False,
    embedded: bool = False,
) -> dict[str, Any]:
    if not context.get("task_id") or not context.get("sample_id"):
        st.info("当前结果缺少可追溯任务标识，暂不能提交反馈。请重新执行一次检测。")
        return {"submitted": False}
    record = existing_feedback(context)
    surface = st.container(border=True) if embedded else st.expander("反馈与优化", expanded=expanded)
    with surface:
        return _render_editor(
            context,
            key_prefix=key_prefix,
            available_categories=available_categories,
            existing=record,
        )


def _batch_context(task_id: str, row: dict[str, Any]) -> dict[str, Any]:
    result_id = str(row.get("result_id") or "")
    return {
        "mode": "batch",
        "task_id": task_id,
        "sample_id": result_id,
        "result_id": result_id,
        "sample_name": str(row.get("image_name") or row.get("filename") or "检测图片"),
        "prediction": row.get("prediction"),
        "score": row.get("score"),
        "threshold": row.get("threshold"),
        "memory_bank_id": row.get("memory_bank_id"),
    }


def batch_feedback_context(task_id: str, row: dict[str, Any]) -> dict[str, Any]:
    return _batch_context(task_id, row)


def _score_label(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def _browser_state(prefix: str, task_id: str) -> dict[str, Any]:
    base = f"{prefix}_{task_id}"
    page_key = f"{base}_page"
    size_key = f"{base}_page_size"
    st.session_state.setdefault(page_key, 1)
    st.session_state.setdefault(size_key, 10)
    return {
        "base": base,
        "page_key": page_key,
        "size_key": size_key,
        "query_key": f"{base}_query",
        "prediction_key": f"{base}_prediction",
        "feedback_key": f"{base}_feedback",
        "preview_key": f"{base}_preview",
        "editor_key": f"{base}_editor",
    }


def _reset_page_when_filters_change(state: dict[str, Any], values: tuple[Any, ...]) -> None:
    signature_key = f"{state['base']}_filter_signature"
    signature = tuple(str(value) for value in values)
    if st.session_state.get(signature_key) != signature:
        st.session_state[signature_key] = signature
        st.session_state[state["page_key"]] = 1


def _set_state_value(key: str, value: Any) -> None:
    st.session_state[key] = value


def _render_page_navigation(state: dict[str, Any], *, page: int, total_pages: int, total: int) -> None:
    previous, summary, page_input, following = st.columns([0.8, 1.5, 0.9, 0.8], vertical_alignment="center")
    previous.button(
        "上一页", disabled=page <= 1, key=f"{state['base']}_previous", width="stretch",
        on_click=_set_state_value, args=(state["page_key"], max(1, page - 1)),
    )
    summary.caption(f"第 {page}/{max(1, total_pages)} 页 · 共 {total} 条")
    if int(st.session_state.get(state["page_key"], 1)) > max(1, total_pages):
        st.session_state[state["page_key"]] = max(1, total_pages)
    page_input.number_input(
        "页码",
        min_value=1,
        max_value=max(1, total_pages),
        step=1,
        key=state["page_key"],
        label_visibility="collapsed",
    )
    following.button(
        "下一页", disabled=page >= total_pages, key=f"{state['base']}_next", width="stretch",
        on_click=_set_state_value, args=(state["page_key"], min(total_pages, page + 1)),
    )


def _render_selection_controls(
    *,
    state_key: str,
    scope: str,
    page_ids: list[str],
    key_prefix: str,
    item_label: str,
) -> set[str]:
    """Render one stable ID-based selection toolbar shared by batch and video."""
    selected = sync_checkbox_states(
        st.session_state,
        state_key=state_key,
        scope=scope,
        item_ids=page_ids,
    )
    actions = st.columns([1.0, 1.0, 1.0, 2.0])
    if actions[0].button("全选当前页", key=f"{key_prefix}_select_page", width="stretch"):
        selected = set_page_selection(
            st.session_state,
            state_key=state_key,
            scope=scope,
            page_ids=page_ids,
            should_select=True,
        )
    if actions[1].button("取消当前页", key=f"{key_prefix}_unselect_page", width="stretch"):
        selected = set_page_selection(
            st.session_state,
            state_key=state_key,
            scope=scope,
            page_ids=page_ids,
            should_select=False,
        )
    if actions[2].button("清除全部选择", key=f"{key_prefix}_clear", width="stretch"):
        clear_selection(
            st.session_state,
            state_key=state_key,
            scopes=[scope],
            visible_ids=page_ids,
        )
        selected = set()
    current_page_count = len(set(page_ids).intersection(selected))
    actions[3].caption(
        f"当前页已选择 {current_page_count} 条，共选择 {len(selected_ids(st.session_state, state_key))} 条{item_label}。"
    )
    return selected


def _selected_item_registry(
    *,
    registry_key: str,
    visible_items: list[dict[str, Any]],
    selected: set[str],
) -> dict[str, dict[str, Any]]:
    """Keep metadata for cross-page selections without reloading every result."""
    registry_value = st.session_state.get(registry_key)
    registry = dict(registry_value) if isinstance(registry_value, dict) else {}
    for item in visible_items:
        item_id = str(item.get("item_id") or "")
        if item_id in selected:
            registry[item_id] = dict(item)
    for item_id in list(registry):
        if item_id not in selected:
            registry.pop(item_id, None)
    st.session_state[registry_key] = registry
    return registry


def _render_lazy_preview(state: dict[str, Any], items: list[dict[str, Any]]) -> None:
    preview = st.session_state.get(state["preview_key"])
    if not isinstance(preview, dict):
        return
    item_id = str(preview.get("item_id") or "")
    item = next((row for row in items if str(row.get("item_id") or "") == item_id), None)
    if not item:
        # The preview may refer to another page.  Keep the state, but do not
        # fetch a large artifact until that item becomes visible again.
        return
    kind = str(preview.get("kind") or "original")
    if kind == "original":
        values = [("原始样本", item.get("original_image_url"))]
    else:
        values = [
            ("检测叠加图", item.get("result_image_url")),
            ("异常热力图", item.get("heatmap_url")),
            ("分割 Mask", item.get("mask_url")),
        ]
    available = [(label, client().artifact_url(str(url))) for label, url in values if url]
    with st.container(border=True):
        title = str(item.get("filename") or f"帧 {item.get('frame_index', '-')}")
        st.markdown(f"**{title} · {'原始样本' if kind == 'original' else '可视化结果'}**")
        if not available:
            st.info("当前记录没有保存对应图像。")
        else:
            columns = st.columns(min(3, len(available)))
            for index, (label, url) in enumerate(available):
                columns[index % len(columns)].image(url, caption=label, width="stretch")
        if st.button("关闭预览", key=f"{state['base']}_close_preview"):
            st.session_state.pop(state["preview_key"], None)


@st.fragment
def render_batch_feedback_section(
    *,
    task_id: str,
    rows: list[dict[str, Any]],
    available_categories: list[str] | None = None,
) -> None:
    if not task_id or not rows:
        return
    with st.expander("反馈与优化", expanded=False):
        st.info("当前检测使用冻结配置。反馈将被保存，不会自动修改当前模型、记忆库或阈值。")
        st.markdown("#### 反馈样本浏览与选择")
        st.caption("服务端分页只读取当前页元数据；原图和热力图仅在点击查看时加载。")
        state = _browser_state("batch_feedback_browser", task_id)
        filter_columns = st.columns([2.2, 1.0, 1.0, 0.8])
        query = filter_columns[0].text_input("按文件名搜索", key=state["query_key"], placeholder="输入文件名关键词")
        prediction = filter_columns[1].selectbox(
            "检测结果", ["all", "normal", "anomaly", "failed"],
            format_func={"all": "全部", "normal": "正常", "anomaly": "缺陷", "failed": "失败"}.get,
            key=state["prediction_key"],
        )
        feedback_filter = filter_columns[2].selectbox(
            "反馈状态", ["all", "unsubmitted", "submitted"],
            format_func={"all": "全部", "unsubmitted": "未反馈", "submitted": "已反馈"}.get,
            key=state["feedback_key"],
        )
        page_size = int(filter_columns[3].selectbox("每页", [10, 20], key=state["size_key"]))
        _reset_page_when_filters_change(state, (query, prediction, feedback_filter, page_size))
        response = detection_task_items(
            task_id,
            mode="batch",
            page=int(st.session_state[state["page_key"]]),
            page_size=page_size,
            query=query,
            prediction=prediction,
            feedback_status=feedback_filter,
        )
        if not response.get("ok"):
            st.error(friendly_error_message(response.get("error"), "批量反馈样本读取失败。"))
            return
        items = list(response.get("items") or [])
        page = int(response.get("page") or 1)
        total_pages = int(response.get("total_pages") or 1)
        total = int(response.get("total") or 0)
        st.session_state[state["page_key"]] = page
        selection_key = f"batch_feedback_selected_ids_{task_id}"
        selection_scope = f"batch-feedback-{task_id}"
        page_ids = [str(item.get("item_id") or "") for item in items if item.get("item_id")]
        _render_selection_controls(
            state_key=selection_key,
            scope=selection_scope,
            page_ids=page_ids,
            key_prefix=state["base"],
            item_label="检测结果",
        )

        header = st.columns([0.35, 0.45, 2.0, 0.75, 0.7, 0.7, 0.75, 0.85, 1.75])
        for column, label in zip(header, ["选择", "序号", "样本名称", "结果", "分数", "阈值", "状态", "反馈", "操作"]):
            column.caption(label)
        for item in items:
            item_id = str(item.get("item_id") or "")
            row_columns = st.columns([0.35, 0.45, 2.0, 0.75, 0.7, 0.7, 0.75, 0.85, 1.75], vertical_alignment="center")
            row_columns[0].checkbox(
                "选择",
                key=checkbox_key(selection_scope, item_id),
                label_visibility="collapsed",
                on_change=checkbox_changed,
                args=(st.session_state,),
                kwargs={"state_key": selection_key, "scope": selection_scope, "item_id": item_id},
            )
            row_columns[1].write(str(item.get("row_number") or "-"))
            row_columns[2].write(str(item.get("filename") or "检测图片"))
            row_columns[3].write(_prediction_label(item.get("prediction")))
            row_columns[4].write(_score_label(item.get("score")))
            row_columns[5].write(_score_label(item.get("threshold")))
            row_columns[6].write("已完成" if item.get("status") == "success" else "失败")
            row_columns[7].write(str(item.get("feedback_label") or "未反馈"))
            operation_columns = row_columns[8].columns(3)
            if operation_columns[0].button("原图", key=f"{state['base']}_original_{item_id}", width="stretch"):
                st.session_state[state["preview_key"]] = {"item_id": item_id, "kind": "original"}
            if operation_columns[1].button(
                "热图", key=f"{state['base']}_visual_{item_id}", width="stretch",
                disabled=not bool(item.get("visualization_available")),
            ):
                st.session_state[state["preview_key"]] = {"item_id": item_id, "kind": "visual"}
            if operation_columns[2].button("反馈", key=f"{state['base']}_feedback_{item_id}", width="stretch"):
                st.session_state[state["editor_key"]] = item_id
        if not items:
            st.info("当前筛选条件下没有批量检测结果。")
        _render_page_navigation(state, page=page, total_pages=total_pages, total=total)
        _render_lazy_preview(state, items)

        selected_id = str(st.session_state.get(state["editor_key"]) or "")
        selected_row = next((row for row in rows if str(row.get("result_id") or "") == selected_id), None)
        if selected_row is None:
            selected_row = next((row for row in items if str(row.get("item_id") or "") == selected_id), None)
        if selected_row:
            st.markdown("#### 当前样本反馈")
            render_feedback_panel(
                _batch_context(task_id, selected_row),
                key_prefix=f"batch_single_{task_id}_{selected_id[:10]}",
                available_categories=available_categories,
                embedded=True,
            )

        st.divider()
        st.markdown("#### 批量反馈")
        selected_values = selected_ids(st.session_state, selection_key)
        st.caption(f"已选择 {len(selected_values)} 个检测结果；翻页和筛选不会清空选择。")
        assessment = st.selectbox(
            "批量确认",
            list(ASSESSMENTS),
            format_func=ASSESSMENTS.get,
            key=f"batch_feedback_assessment_{task_id}",
        )
        corrected_label = st.selectbox(
            "批量修正结果",
            list(CORRECTED_LABELS),
            format_func=CORRECTED_LABELS.get,
            key=f"batch_feedback_correction_{task_id}",
        )
        defect_type = None
        category_options = _categories(available_categories, None)
        if corrected_label == "defect" and category_options:
            defect_type = st.selectbox("批量缺陷类别", category_options, key=f"batch_feedback_type_{task_id}")
        note = st.text_input("批量备注（可选）", max_chars=1000, key=f"batch_feedback_note_{task_id}")
        action_scope = f"batch-feedback-submit::{task_id}"
        submit_clicked, action_status = render_action_button_with_feedback(
            "提交批量反馈",
            scope=action_scope,
            key=f"batch_feedback_submit_{task_id}",
            button_type="primary",
            disabled=not selected_values,
            help_text="提交后进入人工审核队列；当前检测配置不会自动改变。",
        )
        if submit_clicked:
            update_action_feedback(
                action_scope,
                "info",
                f"正在提交 {len(selected_values)} 条反馈...",
                action_status,
            )
            selected_rows = [row for row in rows if str(row.get("result_id") or "") in selected_values]
            result = submit_bulk_feedback(
                [_batch_context(task_id, row) for row in selected_rows],
                {
                    "assessment": assessment,
                    "corrected_label": corrected_label,
                    "corrected_defect_type": defect_type,
                    "issue_tags": [],
                    "note": note,
                    "application_mode": "save_only",
                },
            )
            if result.get("ok"):
                saved_count = int(result.get("saved_count") or 0)
                failures = list(result.get("failures") or [])
                successful_ids = {
                    str(record.get("sample_id") or record.get("result_id") or "")
                    for record in list(result.get("feedbacks") or [])
                    if record.get("sample_id") or record.get("result_id")
                }
                successful_ids &= selected_values
                if not failures:
                    # The status API can return feedback identifiers rather than the
                    # detection item identifiers used by the checkbox registry.  A
                    # failure-free bulk response means every selected item was
                    # accepted (including idempotent re-submission), so clear the
                    # canonical selection directly instead of relying on cross-ID
                    # matching.
                    successful_ids = set(selected_values)
                remove_selected_ids(
                    st.session_state,
                    state_key=selection_key,
                    removed_ids=successful_ids,
                    scopes=[selection_scope],
                )
                candidates = result.get("candidates") or []
                record_only = bool(candidates) and all(
                    str(item.get("status") or "") == "record_only" for item in candidates
                )
                if failures:
                    update_action_feedback(
                        action_scope,
                        "warning",
                        f"已提交 {saved_count} 条，失败 {len(failures)} 条；失败项仍保持选择。",
                        action_status,
                    )
                    with st.expander("查看批量反馈失败原因", expanded=False):
                        for failure in failures:
                            st.write(f"- {failure.get('sample_name') or failure.get('sample_id') or '检测样本'}：{failure.get('message') or '保存失败'}")
                elif record_only:
                    update_action_feedback(
                        action_scope,
                        "info",
                        f"已提交 {saved_count} 条反馈，等待人工复核（只读记录）。",
                        action_status,
                    )
                else:
                    update_action_feedback(
                        action_scope,
                        "success",
                        f"已提交 {saved_count} 条反馈，等待人工审核。",
                        action_status,
                    )
                _refresh_after_batch_feedback_submission(failures)
            else:
                update_action_feedback(
                    action_scope,
                    "error",
                    friendly_error_message(result.get("error"), "批量反馈保存失败。"),
                    action_status,
                )


def _video_context(task: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    index = int(target.get("frame_index", -1))
    runtime = task.get("feedback_runtime") or {}
    input_info = task.get("input") or {}
    return {
        "mode": "video",
        "task_id": str(task.get("job_id") or ""),
        "sample_id": f"frame:{index}",
        "frame_index": index,
        "timestamp_ms": target.get("timestamp_ms"),
        "interval_start_ms": target.get("interval_start_ms"),
        "interval_end_ms": target.get("interval_end_ms"),
        "perceptual_hash": target.get("perceptual_hash"),
        "sample_name": str(target.get("filename") or f"{input_info.get('original_filename') or '视频'} · 帧 {index}"),
        "prediction": target.get("prediction"),
        "score": target.get("score", target.get("anomaly_score")),
        "threshold": target.get("threshold", runtime.get("threshold")),
        "memory_bank_id": runtime.get("memory_bank_id"),
        "model_version": runtime.get("model_version"),
    }


def video_feedback_context(task: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    return _video_context(task, target)


def render_task_item_feedback(
    *,
    task_id: str,
    mode: str,
    item: dict[str, Any],
    key_prefix: str,
    available_categories: list[str] | None = None,
) -> None:
    if mode == "video":
        context = _video_context({"job_id": task_id, "input": {}}, item)
    else:
        context = _batch_context(task_id, item)
    render_feedback_panel(
        context,
        key_prefix=key_prefix,
        available_categories=available_categories,
        embedded=True,
    )


@st.fragment
def render_video_feedback_section(task: dict[str, Any], *, available_categories: list[str] | None = None) -> None:
    task_id = str(task.get("job_id") or "")
    if not task_id:
        return
    with st.expander("反馈与优化", expanded=False):
        st.markdown("#### 视频反馈与优化")
        st.info("当前检测使用冻结配置。反馈将被保存，不会自动修改当前模型、记忆库或阈值。")
        st.caption(
            "当前列表仅展示实际处理并写入帧级结果的记录；按抽帧策略跳过的帧不会被伪造成可反馈样本。"
        )
        duration = float((task.get("input") or {}).get("metadata", {}).get("duration_seconds") or 1.0)
        task_summary = dict(task.get("summary") or {})
        available_total = int(
            task_summary.get("successful_processed_frames")
            or task.get("completed_items")
            or (task.get("progress") or {}).get("successful_frames")
            or 0
        )
        feedback_response = task_feedback(task_id, mode="video")
        feedback_frames = {
            int(record.get("frame_index"))
            for record in (feedback_response.get("feedbacks") or [])
            if record.get("frame_index") is not None
        }
        feedback_total = min(available_total, len(feedback_frames)) if available_total else len(feedback_frames)
        summary_columns = st.columns(4)
        summary_columns[0].metric("可反馈帧数", available_total)
        summary_columns[1].metric("已反馈", feedback_total)
        summary_columns[2].metric("未反馈", max(0, available_total - feedback_total))

        state = _browser_state("video_feedback_browser", task_id)
        selection_key = f"selected_video_feedback_ids_{task_id}"
        selection_scope = f"video-feedback-{task_id}"
        registry_key = f"{selection_key}_items"
        selection_metric = summary_columns[3].empty()

        filter_columns = st.columns([1.7, 0.9, 0.9, 0.8])
        query = filter_columns[0].text_input(
            "按帧号搜索", key=state["query_key"], placeholder="例如：125 或 frame:125"
        )
        prediction = filter_columns[1].selectbox(
            "检测结果", ["all", "normal", "anomaly", "failed"],
            format_func={"all": "全部", "normal": "正常", "anomaly": "缺陷", "failed": "失败"}.get,
            key=state["prediction_key"],
        )
        feedback_filter = filter_columns[2].selectbox(
            "反馈状态", ["all", "unsubmitted", "submitted"],
            format_func={"all": "全部", "unsubmitted": "未反馈", "submitted": "已反馈"}.get,
            key=state["feedback_key"],
        )
        page_size = int(filter_columns[3].selectbox("每页", [10, 20], key=state["size_key"]))
        time_columns = st.columns(2)
        start_seconds = float(
            time_columns[0].number_input(
                "时间起点（秒）", min_value=0.0, value=0.0, step=0.5,
                key=f"{state['base']}_time_start",
            )
        )
        end_seconds = float(
            time_columns[1].number_input(
                "时间终点（秒）", min_value=0.0, value=max(1.0, duration), step=0.5,
                key=f"{state['base']}_time_end",
            )
        )
        _reset_page_when_filters_change(
            state, (query, prediction, feedback_filter, page_size, start_seconds, end_seconds)
        )
        if end_seconds < start_seconds:
            st.warning("时间终点不能早于时间起点。")
            return
        response = detection_task_items(
            task_id,
            mode="video",
            page=int(st.session_state[state["page_key"]]),
            page_size=page_size,
            query=query,
            prediction=prediction,
            feedback_status=feedback_filter,
            time_from_ms=start_seconds * 1000.0,
            time_to_ms=end_seconds * 1000.0,
        )
        items = list(response.get("items") or []) if response.get("ok") else []
        if not response.get("ok"):
            # A just-started video may not have persisted JSONL rows yet.  Use
            # only the current server-owned preview frame as a temporary row.
            current = dict(task.get("current_frame_context") or {})
            if current.get("frame_index") is not None:
                current.update(
                    item_id=f"frame:{int(current['frame_index'])}",
                    filename=f"帧 {int(current['frame_index'])}",
                    row_number=1,
                    feedback_label="未反馈",
                    visualization_available=False,
                )
                items = [current]
                response = {"ok": True, "page": 1, "total_pages": 1, "total": 1, "items": items}
            else:
                st.info("当前暂无已保存帧级检测明细，请等待视频任务产生检测帧。")
                return
        page = int(response.get("page") or 1)
        total_pages = int(response.get("total_pages") or 1)
        total = int(response.get("total") or 0)
        st.session_state[state["page_key"]] = page
        page_ids = [str(item.get("item_id") or "") for item in items if item.get("item_id")]
        selected = _render_selection_controls(
            state_key=selection_key,
            scope=selection_scope,
            page_ids=page_ids,
            key_prefix=state["base"],
            item_label="帧",
        )
        selection_metric.metric("当前选择", len(selected))
        registry = _selected_item_registry(
            registry_key=registry_key,
            visible_items=items,
            selected=selected,
        )

        header = st.columns([0.35, 0.45, 0.65, 0.85, 0.75, 0.7, 0.65, 0.8, 2.7])
        for column, label in zip(
            header,
            ["选择", "序号", "帧号", "视频时间", "结果", "分数", "阈值", "反馈", "操作"],
        ):
            column.caption(label)
        for item in items:
            item_id = str(item.get("item_id") or f"frame:{item.get('frame_index')}")
            row_columns = st.columns(
                [0.35, 0.45, 0.65, 0.85, 0.75, 0.7, 0.65, 0.8, 2.7],
                vertical_alignment="center",
            )
            row_columns[0].checkbox(
                "选择",
                key=checkbox_key(selection_scope, item_id),
                label_visibility="collapsed",
                on_change=checkbox_changed,
                args=(st.session_state,),
                kwargs={"state_key": selection_key, "scope": selection_scope, "item_id": item_id},
            )
            row_columns[1].write(str(item.get("row_number") or "-"))
            row_columns[2].write(str(item.get("frame_index") if item.get("frame_index") is not None else "-"))
            row_columns[3].write(_time_label(item.get("timestamp_ms")))
            row_columns[4].write(_prediction_label(item.get("prediction")))
            row_columns[5].write(_score_label(item.get("score")))
            row_columns[6].write(_score_label(item.get("threshold")))
            row_columns[7].write(str(item.get("feedback_label") or "未反馈"))
            operation_columns = row_columns[8].columns(3)
            if operation_columns[0].button(
                "查看原始帧", key=f"{state['base']}_original_{item_id}", width="stretch"
            ):
                st.session_state[state["preview_key"]] = {"item_id": item_id, "kind": "original"}
            visualization_available = bool(item.get("visualization_available"))
            if operation_columns[1].button(
                "查看可视化热力图" if visualization_available else "无可视化热力图",
                key=f"{state['base']}_visual_{item_id}",
                width="stretch",
                disabled=not visualization_available,
            ):
                st.session_state[state["preview_key"]] = {"item_id": item_id, "kind": "visual"}
            if operation_columns[2].button(
                "单条反馈", key=f"{state['base']}_feedback_{item_id}", width="stretch"
            ):
                st.session_state[state["editor_key"]] = item_id
        if not items:
            st.info("当前筛选条件下没有视频帧记录。")
        _render_page_navigation(state, page=page, total_pages=total_pages, total=total)
        _render_lazy_preview(state, items)

        selected_id = str(st.session_state.get(state["editor_key"]) or "")
        selected_target = next((item for item in items if str(item.get("item_id") or "") == selected_id), None)
        if selected_target:
            st.markdown("#### 当前帧反馈")
            render_feedback_panel(
                _video_context(task, selected_target),
                key_prefix=f"video_frame_{task_id}_{int(selected_target['frame_index'])}",
                available_categories=available_categories,
                embedded=True,
            )

        st.divider()
        st.markdown("#### 批量反馈")
        selected_values = selected_ids(st.session_state, selection_key)
        st.caption(f"已选择 {len(selected_values)} 个帧级结果；翻页和筛选不会清空选择。")
        if selected_values:
            with st.container(border=True):
                st.markdown("##### 批量反馈设置")
                st.caption("批量反馈适用于人工判断一致的帧；标签不同的帧请分组提交。")
                corrected_label = st.radio(
                    "人工标签",
                    ["normal", "defect"],
                    index=None,
                    format_func=CORRECTED_LABELS.get,
                    horizontal=True,
                    key=f"video_feedback_batch_label_{task_id}",
                )
                if corrected_label is None:
                    st.caption("请选择人工标签；未确定标签的视频帧不会进入待审核池。")
                defect_type = None
                if corrected_label == "defect":
                    category_options = _categories(available_categories, None)
                    if category_options:
                        defect_type = st.selectbox(
                            "缺陷类别（可选）",
                            [""] + category_options + (["其他/未知"] if "其他/未知" not in category_options else []),
                            format_func=lambda value: value or "暂不指定",
                            key=f"video_feedback_batch_type_{task_id}",
                        ) or None
                    else:
                        defect_type = st.text_input(
                            "缺陷类别（可选）",
                            max_chars=120,
                            key=f"video_feedback_batch_type_text_{task_id}",
                        ) or None
                note = st.text_input(
                    "批量备注（可选）",
                    max_chars=1000,
                    key=f"video_feedback_batch_note_{task_id}",
                )
                st.caption("Mask 属于具体帧，批量操作只提交统一人工标签；Mask 请在反馈优化中心逐帧补充或编辑。")
                busy_key = f"video_feedback_batch_busy_{task_id}"
                action_scope = f"video-feedback-submit::{task_id}"
                submit_clicked, action_status = render_action_button_with_feedback(
                    "提交批量反馈",
                    scope=action_scope,
                    key=f"video_feedback_batch_submit_{task_id}",
                    button_type="primary",
                    disabled=bool(st.session_state.get(busy_key)) or corrected_label not in {"normal", "defect"},
                    help_text="提交只保存统一人工标签；Mask 仍按具体帧单独维护。",
                )
                if submit_clicked:
                    update_action_feedback(
                        action_scope,
                        "info",
                        f"正在提交 {len(selected_values)} 条帧反馈...",
                        action_status,
                    )
                    selected_targets = [registry[item_id] for item_id in selected_values if item_id in registry]
                    missing_count = len(selected_values) - len(selected_targets)
                    if missing_count:
                        update_action_feedback(
                            action_scope,
                            "error",
                            f"有 {missing_count} 条跨页选择缺少帧元数据，请清除后重新选择。",
                            action_status,
                        )
                    else:
                        st.session_state[busy_key] = True
                        try:
                            result = submit_bulk_feedback(
                                [_video_context(task, target) for target in selected_targets],
                                {
                                    "assessment": "derive_from_label",
                                    "corrected_label": corrected_label,
                                    "corrected_defect_type": defect_type,
                                    "issue_tags": [],
                                    "note": note,
                                    "application_mode": "save_only",
                                },
                            )
                        finally:
                            st.session_state[busy_key] = False
                        if result.get("ok"):
                            feedbacks = list(result.get("feedbacks") or [])
                            successful_ids = {
                                str(record.get("sample_id") or f"frame:{record.get('frame_index')}")
                                for record in feedbacks
                                if record.get("sample_id") or record.get("frame_index") is not None
                            }
                            failures = list(result.get("failures") or [])
                            if not failures:
                                successful_ids = set(selected_values)
                            remove_selected_ids(
                                st.session_state,
                                state_key=selection_key,
                                removed_ids=successful_ids,
                                scopes=[selection_scope],
                            )
                            for item_id in successful_ids:
                                registry.pop(item_id, None)
                            st.session_state[registry_key] = registry
                            saved_count = int(result.get("saved_count") or 0)
                            if failures:
                                update_action_feedback(
                                    action_scope,
                                    "warning",
                                    f"已提交 {saved_count} 条，失败 {len(failures)} 条；失败项仍保持选择。",
                                    action_status,
                                )
                                with st.expander("查看失败原因", expanded=True):
                                    for failure in failures:
                                        frame_label = failure.get("frame_index")
                                        st.write(f"- 帧 {frame_label if frame_label is not None else '-'}：{failure.get('message') or '保存失败'}")
                            else:
                                update_action_feedback(
                                    action_scope,
                                    "success",
                                    f"已提交 {saved_count} 条帧反馈，等待人工审核。",
                                    action_status,
                                )
                        else:
                            update_action_feedback(
                                action_scope,
                                "error",
                                friendly_error_message(result.get("error"), "视频批量反馈保存失败。"),
                                action_status,
                            )
        else:
            st.caption("请勾选一个或多个帧后进行批量反馈。")

        with st.expander("按时间区间选取代表帧", expanded=False):
            st.caption("从真实帧级检测明细中选择非相邻异常峰值，用于快速复核。")
            interval_columns = st.columns([1.0, 1.0, 0.7])
            interval_start = interval_columns[0].number_input(
                "区间开始（秒）", min_value=0.0, value=0.0, step=0.5,
                key=f"video_feedback_interval_start_{task_id}",
            )
            interval_end = interval_columns[1].number_input(
                "区间结束（秒）", min_value=0.0, value=max(1.0, duration), step=0.5,
                key=f"video_feedback_interval_end_{task_id}",
            )
            max_frames = int(interval_columns[2].number_input(
                "最多帧数", min_value=1, max_value=32, value=8, step=1,
                key=f"video_feedback_interval_limit_{task_id}",
            ))
            interval_scope = f"video-feedback-interval::{task_id}"
            extract_clicked, extract_status = render_action_button_with_feedback(
                "提取代表帧",
                scope=interval_scope,
                key=f"video_feedback_interval_extract_{task_id}",
                disabled=interval_end < interval_start,
            )
            if extract_clicked:
                update_action_feedback(interval_scope, "info", "正在选择异常峰值代表帧...", extract_status)
                with st.spinner("正在选择异常峰值代表帧..."):
                    interval_result = video_feedback_interval(
                        task_id, start_ms=float(interval_start) * 1000.0,
                        end_ms=float(interval_end) * 1000.0, max_frames=max_frames,
                    )
                if interval_result.get("ok"):
                    frames = interval_result.get("frames") or []
                    st.session_state[f"video_feedback_interval_targets_{task_id}"] = frames
                    update_action_feedback(
                        interval_scope,
                        "success",
                        f"已选择 {len(frames)} 个非相邻代表帧，可通过上方帧号搜索逐条反馈。",
                        extract_status,
                    )
                else:
                    update_action_feedback(
                        interval_scope,
                        "error",
                        friendly_error_message(interval_result.get("error"), "视频反馈区间读取失败。"),
                        extract_status,
                    )


__all__ = [
    "batch_feedback_context",
    "render_batch_feedback_section",
    "render_feedback_panel",
    "render_task_item_feedback",
    "render_video_feedback_section",
    "video_feedback_context",
]
