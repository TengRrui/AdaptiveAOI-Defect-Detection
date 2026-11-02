"""Shared delete-preview acknowledgement UI.

The preview is produced by the server and is kept in ``st.session_state`` by
the caller.  This component deliberately has no text confirmation field and
does not open a second dialog: the only client-side acknowledgement is a
checkbox.  The server still receives the immutable plan identifiers and
performs the final dependency/protection recheck.
"""

from __future__ import annotations

from typing import Any, Callable

import streamlit as st

from ui_common import render_action_button_with_feedback, update_action_feedback


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _first_int(preview: dict[str, Any], *keys: str) -> int:
    for key in keys:
        if key in preview and preview.get(key) not in (None, ""):
            return _as_int(preview.get(key))
    return 0


def render_delete_preview_confirmation(
    *,
    preview: dict[str, Any],
    title: str,
    action_label: str,
    acknowledgement_key: str,
    button_key: str,
    execute_callback: Callable[[dict[str, Any]], Any],
    object_label: str = "对象",
    disabled: bool = False,
    extra_summary: list[tuple[str, str]] | None = None,
) -> Any:
    """Render one consistent preview → checkbox → final-action flow.

    ``execute_callback`` is called only after the checkbox is checked and the
    preview reports at least one deletable item.  The callback receives no
    target collection from the browser; it gets only the server-issued plan
    identifiers and ``acknowledged=True``.
    """

    selected = _first_int(preview, "selected_records", "selected_count", "resource_count", "feedback_count")
    deletable = _first_int(preview, "deletable_records", "deletable_count", "physical_delete_count", "feedback_count")
    protected = _first_int(preview, "protected_records", "protected_count", "blocked_count")
    files = _first_int(preview, "file_count", "attachment_count")
    results = _first_int(preview, "result_count", "results_count")
    if not selected and ("allowed" in preview or preview.get("resource_id")):
        selected = 1
    if not deletable and bool(preview.get("allowed")):
        deletable = 1
    plan_id = str(preview.get("plan_id") or "")
    preview_token = str(preview.get("preview_token") or preview.get("confirmation_token") or "")

    with st.container(border=True, key=f"delete_preview_container_{button_key}"):
        st.markdown(f"#### {title}")
        columns = st.columns(5)
        columns[0].metric(f"即将删除{object_label}", f"{selected:,}")
        columns[1].metric("可删除", f"{deletable:,}")
        columns[2].metric("受保护", f"{protected:,}")
        columns[3].metric("相关结果/文件", f"{max(results, files):,}")
        columns[4].metric("计划状态", "有效" if plan_id or preview_token else "待核验")
        if extra_summary:
            st.caption(" · ".join(f"{label}：{value}" for label, value in extra_summary))
        if protected:
            protected_items = list(preview.get("protected_items") or preview.get("items") or [])
            with st.expander(f"查看保护原因（{protected:,} 条）", expanded=False):
                rows = []
                for index, item in enumerate(protected_items, start=1):
                    if str(item.get("action") or "") in {"physical_delete", "ready", "deleted"}:
                        continue
                    rows.append({
                        "序号": index,
                        "对象": item.get("sample_name") or item.get("filename") or item.get("resource_id") or "对象",
                        "原因": item.get("protected_reason") or item.get("message") or item.get("reason") or "当前存在真实保护",
                    })
                if rows:
                    st.dataframe(rows, hide_index=True, width="stretch")
                else:
                    st.caption("当前预览已记录保护数量；执行前服务端仍会再次检查。")
        if preview.get("expires_in_seconds") is not None:
            st.caption(f"删除计划已冻结，有效期约 {max(0, _as_int(preview.get('expires_in_seconds')) // 60)} 分钟。")
        acknowledged = st.checkbox("我已核对删除预览", key=acknowledgement_key, disabled=disabled)
        action_scope = f"delete-confirm::{button_key}::{plan_id or preview_token or 'unplanned'}"
        clicked, action_status = render_action_button_with_feedback(
            action_label,
            scope=action_scope,
            key=button_key,
            button_type="primary",
            disabled=disabled or not acknowledged or deletable <= 0,
            column_ratio=(0.62, 1.38),
        )
        if clicked:
            update_action_feedback(action_scope, "info", "正在删除并复核活动保护...", action_status)
            result = execute_callback({
                "plan_id": plan_id,
                "preview_token": preview_token,
                "acknowledged": True,
            })
            if isinstance(result, dict) and result.get("success"):
                update_action_feedback(action_scope, "success", "删除任务已提交，正在处理记录与附件。", action_status)
            else:
                update_action_feedback(action_scope, "error", "删除未完成，请查看下方具体原因。", action_status)
            return result
        if deletable <= 0:
            st.caption("当前没有可删除对象；请先解除真实保护或重新生成预览。")
    return None
