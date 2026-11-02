from __future__ import annotations

import time
from typing import Any

import streamlit as st

from performance_cache import api_client as client
from delete_preview_confirmation import render_delete_preview_confirmation
from friendly_display import friendly_error_message
from performance_cache import (
    cached_config_profile,
    cached_memory_bank_configs,
    cached_runtime_profiles,
    cached_runtime_resources,
    clear_resource_caches,
)
from time_utils import format_beijing_datetime
from ui_common import (
    configure_page,
    render_action_button_with_feedback,
    render_form_submit_button_with_feedback,
    set_action_feedback,
    sidebar_runtime_controls,
    update_action_feedback,
)


METHOD_LABELS = {
    "normal_only": "单正常记忆",
    "normal_defect_calibration": "正常记忆与缺陷校准",
    "normal_memory_defect_guided_lite": "缺陷引导轻量校准",
    "normal_defect_dual_memory": "正常与缺陷双记忆",
    "auto": "自动推荐",
}
STATUS_LABELS = {
    "normal": "可用",
    "warning": "需检查",
    "incomplete": "不完整",
    "archived": "已归档",
    "frozen": "已冻结",
}
DISPLAY_CONFIG_KEYS = (
    ("coreset_ratio", "coreset 采样比例"),
    ("max_memory_patches", "最大记忆条目"),
    ("knn_k", "最近邻数量"),
    ("image_size", "输入尺寸"),
    ("layers", "特征层"),
    ("threshold", "图像级阈值"),
    ("score_aggregation", "分数聚合"),
    ("device", "推理设备"),
    ("use_fp16", "半精度模式"),
    ("threshold_method", "阈值策略"),
    ("threshold_target_recall", "目标召回率"),
    ("threshold_max_false_positive_rate", "最大误检率"),
)


def _error(response: dict[str, Any], fallback: str) -> str:
    return friendly_error_message((response or {}).get("error"), fallback)


def _rerun_notice(
    scope: str,
    message: str,
    level: str = "success",
    *,
    resource_revision: int = 0,
) -> None:
    set_action_feedback(
        scope,
        level,
        message,
        page_scope="runtime_config",
    )
    clear_resource_caches()
    if resource_revision > 0:
        # Install the revision returned by the mutation before the local
        # fragment reruns, so status labels cannot render one revision behind.
        st.session_state["aoi_resource_catalog_revision"] = int(resource_revision)
    st.session_state["aoi_resource_catalog_checked_at"] = 0.0
    # Both the current-runtime card and management tables live in the same
    # fragment, so refreshing that fragment is sufficient and preserves the
    # page/navigation shell.
    st.rerun(scope="fragment")


def _wait_for_applied_runtime(
    profile_id: str,
    config_profile_id: str,
    config_profile_version: int,
    *,
    timeout_seconds: float = 75.0,
) -> tuple[bool, str]:
    """Wait until the newly bound resources—not just the registry row—are ready."""

    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    last_stage = "正在加载目标域检测资源"
    while time.monotonic() < deadline:
        response = client().readiness()
        if not response.get("success"):
            time.sleep(0.25)
            continue
        last_stage = str(response.get("current_stage") or last_stage)
        context = dict(response.get("runtime_context") or {})
        current_profile = dict(response.get("current_profile") or {})
        matches = (
            str(context.get("profile_id") or context.get("current_profile_id") or current_profile.get("profile_id") or "") == str(profile_id)
            and str(context.get("config_profile_id") or context.get("current_config_id") or current_profile.get("config_profile_id") or "") == str(config_profile_id)
            and int(context.get("config_version") or context.get("current_config_version") or current_profile.get("config_profile_version") or 0) == int(config_profile_version)
        )
        if matches and bool(response.get("ready")) and str(context.get("status") or context.get("runtime_status") or "") == "ready":
            return True, ""
        if matches and str(response.get("status") or "") == "failed":
            return False, str(response.get("error") or "运行资源预热失败。")
        time.sleep(0.25)
    return False, f"运行方案已绑定，但在等待就绪时超时（最后阶段：{last_stage}）。"


def _load_resources() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    revision = int(st.session_state.get("aoi_resource_catalog_revision") or 0)
    response = cached_runtime_resources(revision, "target_profile", True)
    current_response = cached_runtime_profiles(revision)
    if not response.get("success"):
        st.error(_error(response, "记忆库资源暂时无法读取。"))
        return [], {}
    current = (current_response.get("current") or {}).get("runtime") if current_response.get("success") else {}
    rows = [item for item in list(response.get("resources") or []) if item.get("resource_type") == "target_profile"]
    rows.sort(key=lambda item: (bool(item.get("active")), not bool(item.get("archived")), str(item.get("updated_at") or "")), reverse=True)
    return rows, dict(current or {})


def _bank_label(item: dict[str, Any]) -> str:
    name = str(item.get("display_name") or item.get("profile_id") or "未命名记忆库")
    test_only = str((item.get("metadata") or {}).get("deployment_scope") or "") == "test_only"
    state = "测试隔离" if test_only else str(item.get("lifecycle_status") or ("已归档" if item.get("archived") else ("当前使用" if item.get("active") else "可用")))
    return f"{name} · {item.get('dataset') or '-'} / {item.get('category') or '-'} · {state}"


def _config_key(item: dict[str, Any]) -> str:
    return f"{item.get('config_profile_id')}::v{int(item.get('version') or 1)}"


def _config_label(item: dict[str, Any]) -> str:
    state = "当前使用" if item.get("current") else ("历史版本" if not item.get("applied") else "已应用")
    return f"{item.get('friendly_name') or '未命名配置'} · v{int(item.get('version') or 1)} · {state}"


def _config_form(defaults: dict[str, Any], key_prefix: str) -> dict[str, Any]:
    left, right = st.columns(2)
    with left:
        knn_k = st.number_input("最近邻数量", 1, 64, int(defaults.get("knn_k") or 1), key=f"{key_prefix}_knn")
        threshold = st.number_input("图像级阈值", 0.0, 1000000.0, float(defaults.get("threshold") or 0.5), step=0.001, format="%.6f", key=f"{key_prefix}_threshold")
        aggregation = st.selectbox(
            "分数聚合",
            ["maximum_patch", "top_k_mean"],
            index=1 if defaults.get("score_aggregation") == "top_k_mean" else 0,
            format_func=lambda value: "Top-K 高分补丁均值" if value == "top_k_mean" else "最大异常补丁",
            key=f"{key_prefix}_aggregation",
        )
        device = st.selectbox(
            "推理设备",
            ["auto", "cuda", "cpu"],
            index=["auto", "cuda", "cpu"].index(str(defaults.get("device") or "auto")) if str(defaults.get("device") or "auto") in {"auto", "cuda", "cpu"} else 0,
            format_func=lambda value: {"auto": "自动选择", "cuda": "GPU（CUDA）", "cpu": "CPU"}[value],
            key=f"{key_prefix}_device",
        )
    with right:
        threshold_method = st.selectbox(
            "阈值策略",
            ["best_f1", "recall_priority", "normal_percentile", "mean_std"],
            index=["best_f1", "recall_priority", "normal_percentile", "mean_std"].index(str(defaults.get("threshold_method") or "best_f1")),
            format_func=lambda value: {"best_f1": "最佳 F1", "recall_priority": "召回率优先", "normal_percentile": "正常分位数", "mean_std": "均值与标准差"}[value],
            key=f"{key_prefix}_threshold_method",
        )
        target_recall = st.number_input("目标召回率", 0.0, 1.0, float(defaults.get("threshold_target_recall") or 0.95), step=0.01, key=f"{key_prefix}_recall")
        max_fpr = st.number_input("最大误检率", 0.0, 1.0, float(defaults.get("threshold_max_false_positive_rate") or 0.10), step=0.01, key=f"{key_prefix}_fpr")
        batch_size = st.number_input("推理批大小", 1, 32, int(defaults.get("batch_size") or 1), key=f"{key_prefix}_batch")
    with st.expander("视频与校准参数", expanded=False):
        sampling = dict(defaults.get("video_sampling_config") or {})
        a, b, c = st.columns(3)
        frame_interval = a.number_input("视频抽帧间隔（秒）", 0.01, 60.0, float(sampling.get("frame_interval_seconds") or 1.0), step=0.1, key=f"{key_prefix}_frame_interval")
        max_frames = b.number_input("最大代表帧数", 1, 10000, int(sampling.get("max_representative_frames") or 500), key=f"{key_prefix}_max_frames")
        similarity = c.number_input("相似帧去重阈值", 0.0, 1.0, float(sampling.get("similarity_threshold") or 0.98), step=0.01, key=f"{key_prefix}_similarity")
        lite_gate = st.number_input("轻量门控分位数", 90.0, 99.9, float(defaults.get("lightweight_gate_percentile") or 95.0), step=0.1, key=f"{key_prefix}_lite_gate")
    return {
        "knn_k": int(knn_k),
        "threshold": float(threshold),
        "score_aggregation": aggregation,
        "device": device,
        "threshold_method": threshold_method,
        "threshold_target_recall": float(target_recall),
        "threshold_max_false_positive_rate": float(max_fpr),
        "batch_size": int(batch_size),
        "lightweight_gate_percentile": float(lite_gate),
        "video_sampling_config": {
            "frame_interval_seconds": float(frame_interval),
            "max_representative_frames": int(max_frames),
            "similarity_threshold": float(similarity),
        },
    }


def _render_config_details(config: dict[str, Any]) -> None:
    values = dict(config.get("config") or {})
    rows = []
    for key, label in DISPLAY_CONFIG_KEYS:
        value = values.get(key, config.get(key))
        if isinstance(value, list):
            value = "、".join(str(item) for item in value)
        if value is not None:
            if isinstance(value, bool):
                rendered = "是" if value else "否"
            elif isinstance(value, float):
                rendered = f"{value:g}"
            else:
                rendered = str(value)
            rows.append({"参数": label, "当前值": rendered})
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption("输入尺寸、特征层、骨干网络和精度模式与已有记忆特征空间绑定，只读展示；修改这些结构参数需要重新执行目标域适配。")


def _render_memory_lifecycle(item: dict[str, Any]) -> None:
    st.subheader("记忆库生命周期")
    if item.get("active"):
        st.info("当前记忆库正在使用。请先在页面顶部停用当前运行方案，再执行归档或删除。")
        return
    if not item.get("archived"):
        action_scope = f"runtime-archive::{item['resource_id']}"
        archive_clicked, action_status = render_action_button_with_feedback(
            "归档记忆库及其全部配置",
            scope=action_scope,
            key=f"archive_{item['resource_id']}",
            icon=":material/archive:",
        )
        if archive_clicked:
            update_action_feedback(action_scope, "info", "正在归档记忆库及其从属配置...", action_status)
            response = client().runtime_resource_archive(str(item["resource_id"]), confirm=True, current_profile_action="cancel")
            if response.get("success"):
                _rerun_notice(action_scope, "记忆库及其从属配置已归档。")
            else:
                update_action_feedback(action_scope, "error", _error(response, "记忆库归档失败。"), action_status)
        return
    action_scope = f"runtime-restore::{item['resource_id']}"
    restore_clicked, action_status = render_action_button_with_feedback(
        "恢复到可用记忆库",
        scope=action_scope,
        key=f"restore_{item['resource_id']}",
        icon=":material/restore:",
    )
    if restore_clicked:
        update_action_feedback(action_scope, "info", "正在恢复记忆库及从属配置...", action_status)
        response = client().runtime_resource_restore(str(item["resource_id"]), confirm=True)
        if response.get("success"):
            _rerun_notice(action_scope, "记忆库及其从属配置已恢复为可用状态。")
        else:
            update_action_feedback(action_scope, "error", _error(response, "记忆库恢复失败。"), action_status)
    preview_key = f"runtime_delete_preview::{item['resource_id']}"
    preview_revision_key = f"{preview_key}::revision"
    current_revision = int(st.session_state.get("aoi_resource_catalog_revision") or 0)
    if st.session_state.get(preview_revision_key) != current_revision:
        st.session_state.pop(preview_key, None)
    if st.button("生成删除影响预览", icon=":material/preview:", key=f"preview_{item['resource_id']}"):
        with st.spinner("正在核对资源依赖..."):
            preview_response = client().runtime_resource_delete_preview(str(item["resource_id"]))
        if preview_response.get("success"):
            st.session_state[preview_key] = dict(preview_response.get("preview") or {})
            st.session_state[preview_revision_key] = current_revision
            st.session_state[f"delete_ack_{item['resource_id']}"] = False
        else:
            st.warning(_error(preview_response, "删除预览暂不可用。"))
    plan = dict(st.session_state.get(preview_key) or {})
    if not plan:
        st.caption("删除预览仅在点击按钮时计算，普通页面交互不会重复扫描资源文件。")
        return
    response = render_delete_preview_confirmation(
        preview=plan,
        title="删除预览",
        action_label="永久删除该记忆库",
        object_label="记忆库",
        acknowledgement_key=f"delete_ack_{item['resource_id']}",
        button_key=f"delete_{item['resource_id']}",
        execute_callback=lambda payload: client().runtime_resource_delete(
            str(item["resource_id"]),
            confirm=True,
            acknowledged=bool(payload.get("acknowledged")),
            preview_token=str(payload.get("preview_token") or ""),
            reason="普通用户从运行配置删除目标域记忆库",
        ),
        extra_summary=[
            ("配置版本", str((item.get("metadata") or {}).get("config_count") or 0)),
            ("共享模型", "保留"),
            ("历史记录", "保留"),
        ],
    )
    if response is not None:
        if response.get("success"):
            clear_resource_caches()
            st.session_state["aoi_resource_catalog_checked_at"] = 0.0
            st.rerun(scope="fragment")
        else:
            st.error(_error(response, "记忆库删除失败。"))


def _render_bank(item: dict[str, Any], current: dict[str, Any]) -> None:
    metadata = dict(item.get("metadata") or {})
    a, b, c, d = st.columns(4)
    a.metric("数据集", item.get("dataset") or "-")
    b.metric("类别", item.get("category") or "-")
    c.metric("配置数量", int(metadata.get("config_count") or 0))
    d.metric("状态", str(item.get("lifecycle_status") or ("当前使用" if item.get("active") else ("已归档" if item.get("archived") else "可用"))))
    time_label = "归档时间" if item.get("archived") else "创建时间"
    time_value = item.get("archived_at") or item.get("last_archived_at") if item.get("archived") else item.get("created_at")
    st.caption(f"方法模式：{METHOD_LABELS.get(str(item.get('method_mode') or ''), item.get('method_mode') or '-')} · 记忆库大小：{float(metadata.get('resource_size_bytes') or item.get('size_bytes') or 0) / 1024 / 1024:.1f} MB · {time_label}：{format_beijing_datetime(time_value)}")
    if item.get("archived"):
        _render_memory_lifecycle(item)
        return

    revision = int(st.session_state.get("aoi_resource_catalog_revision") or 0)
    response = cached_memory_bank_configs(str(item.get("profile_id") or ""), revision)
    if not response.get("success"):
        st.error(_error(response, "该记忆库的配置无法读取。"))
        return
    configs = list(response.get("configs") or [])
    st.subheader("记忆库内部配置")
    config_rows = [
        {
            "序号": index,
            "配置名称": config.get("friendly_name") or "未命名配置",
            "创建时间": format_beijing_datetime(config.get("created_at")),
            "来源": config.get("source_label") or "目标域初始化",
            "当前使用": "是" if config.get("current") else "否",
            "状态": "可用" if config.get("frozen") and config.get("validated") else "需检查",
        }
        for index, config in enumerate(configs, start=1)
    ]
    st.dataframe(config_rows, hide_index=True, width="stretch")
    if not configs:
        st.error("该记忆库缺少默认配置，不能用于检测；请重新扫描资源或重新构建目标域。")
        return
    options = {_config_key(config): config for config in configs}
    selected_key = st.selectbox("选择配置", list(options), format_func=lambda key: _config_label(options[key]), key=f"select_config_{item['resource_id']}")
    selected = options[selected_key]
    action = st.segmented_control(
        "配置操作",
        ["view", "add", "edit", "delete"],
        format_func={"view": "查看与应用", "add": "新增配置", "edit": "编辑为新版本", "delete": "删除配置"}.get,
        selection_mode="single",
        key=f"runtime_config_action_{item['resource_id']}",
        default="view",
    ) or "view"
    detail: dict[str, Any] = dict(selected)
    if action in {"view", "add", "edit"}:
        detail_response = cached_config_profile(
            str(selected["config_profile_id"]),
            int(selected["version"]),
            revision,
        )
        if not detail_response.get("success"):
            st.error(_error(detail_response, "配置详情暂时无法读取。"))
            return
        detail = dict(detail_response.get("config_profile") or selected)

    if action == "view":
        _render_config_details(detail)
        test_only = str((item.get("metadata") or {}).get("deployment_scope") or "") == "test_only"
        if test_only:
            st.warning("该资源属于测试 Profile，仅保留用于开发核验，不能设为正式运行方案。")
        disabled = bool(selected.get("current")) or test_only
        action_scope = f"runtime-apply::{item['resource_id']}::{selected_key}"
        apply_clicked, action_status = render_action_button_with_feedback(
            "应用该记忆库与配置",
            scope=action_scope,
            key=f"apply_{selected_key}",
            button_type="primary",
            disabled=disabled,
            column_ratio=(0.5, 1.5),
        )
        if apply_clicked:
            update_action_feedback(action_scope, "info", "正在验证并切换运行方案...", action_status)
            result = client().runtime_profile_apply(
                str(item["resource_id"]),
                config_profile_id=str(selected["config_profile_id"]),
                config_profile_version=int(selected["version"]),
            )
            if result.get("success"):
                update_action_feedback(action_scope, "info", "正在加载模型、记忆库与检索索引...", action_status)
                ready, ready_error = _wait_for_applied_runtime(
                    str(item["profile_id"]),
                    str(selected["config_profile_id"]),
                    int(selected["version"]),
                )
                if ready:
                    _rerun_notice(action_scope, "记忆库与配置已原子切换并完成预热，后续检测立即使用新配置。")
                else:
                    update_action_feedback(action_scope, "error", ready_error, action_status)
                return
            else:
                update_action_feedback(action_scope, "error", _error(result, "运行配置切换失败。"), action_status)
    elif action == "add":
        action_scope = f"runtime-config-add::{item['resource_id']}"
        with st.form(f"add_config_{item['resource_id']}"):
            name = st.text_input("新配置名称", value=str(response.get("next_default_name") or "配置2"))
            note = st.text_input("版本说明（可选）")
            values = _config_form(dict(detail.get("config") or {}), f"add_{item['resource_id']}")
            submitted, action_status = render_form_submit_button_with_feedback(
                "创建新配置",
                scope=action_scope,
                button_type="primary",
            )
        if submitted:
            update_action_feedback(action_scope, "info", "正在创建并冻结新配置...", action_status)
            result = client().create_memory_bank_config(
                str(item["profile_id"]),
                {
                    "friendly_name": name,
                    "note": note,
                    "source_config_profile_id": str(selected["config_profile_id"]),
                    "source_config_profile_version": int(selected["version"]),
                    "config": values,
                },
            )
            if result.get("success"):
                _rerun_notice(action_scope, "新配置已创建，并归属于当前记忆库。")
            else:
                update_action_feedback(action_scope, "error", _error(result, "新配置创建失败。"), action_status)
    elif action == "edit":
        st.info("冻结配置不会被覆盖；保存会创建同一配置的下一个版本。")
        action_scope = f"runtime-config-edit::{item['resource_id']}::{selected_key}"
        with st.form(f"revise_config_{selected_key}"):
            name = st.text_input("配置名称", value=str(selected.get("friendly_name") or "配置"), key=f"revise_name_{selected_key}")
            note = st.text_input("版本说明（可选）", value=str(selected.get("note") or ""), key=f"revise_note_{selected_key}")
            values = _config_form(dict(detail.get("config") or {}), f"revise_{selected_key}")
            submitted, action_status = render_form_submit_button_with_feedback(
                "保存为新版本",
                scope=action_scope,
                button_type="primary",
            )
        if submitted:
            update_action_feedback(action_scope, "info", "正在生成新的冻结配置版本...", action_status)
            result = client().revise_memory_bank_config(
                str(item["profile_id"]),
                str(selected["config_profile_id"]),
                int(selected["version"]),
                {"friendly_name": name, "note": note, "config": values},
            )
            if result.get("success"):
                _rerun_notice(action_scope, "已创建新的冻结配置版本，原版本保持不变。")
            else:
                update_action_feedback(action_scope, "error", _error(result, "配置新版本创建失败。"), action_status)
    else:
        if selected.get("current"):
            st.warning("当前正在使用该配置，请先切换到同一记忆库的其他配置。")
        elif len(configs) <= 1:
            st.warning("一个记忆库至少保留一个配置，当前唯一配置不能删除。")
        else:
            st.write(f"将从普通列表移除：{_config_label(selected)}。记忆库本身不会受影响。")
            config_resources_response = cached_runtime_resources(revision, "configuration", True)
            config_binding = f"{selected.get('config_profile_id')}:v{int(selected.get('version') or 1)}"
            config_resource = next(
                (row for row in list(config_resources_response.get("resources") or []) if str(row.get("profile_id") or "") == config_binding),
                None,
            )
            config_preview_key = f"runtime_config_delete_preview::{item['resource_id']}::{selected_key}"
            if config_resource is None:
                st.warning("该配置当前没有可追溯的资源预览，已禁止删除；请先重建资源索引。")
            else:
                if st.button("生成配置删除预览", key=f"preview_config_{selected_key}"):
                    preview_response = client().runtime_resource_delete_preview(str(config_resource.get("resource_id") or ""))
                    if preview_response.get("success"):
                        st.session_state[config_preview_key] = dict(preview_response.get("preview") or {})
                        st.session_state[f"ack_config_{selected_key}"] = False
                    else:
                        st.error(_error(preview_response, "配置删除预览失败。"))
                config_plan = dict(st.session_state.get(config_preview_key) or {})
                if config_plan:
                    result = render_delete_preview_confirmation(
                        preview=config_plan,
                        title="删除配置预览",
                        action_label=f"删除{_config_label(selected)}",
                        object_label="配置",
                        acknowledgement_key=f"ack_config_{selected_key}",
                        button_key=f"delete_config_{selected_key}",
                        execute_callback=lambda payload: client().runtime_resource_delete(
                            str(config_resource.get("resource_id") or ""),
                            confirm=True,
                            acknowledged=bool(payload.get("acknowledged")),
                            preview_token=str(payload.get("preview_token") or ""),
                            reason="运行配置删除历史配置",
                        ),
                        extra_summary=[("所属记忆库", str(item.get("display_name") or item.get("profile_id") or "-")), ("当前使用", "否")],
                    )
                    if result is not None:
                        if result.get("success"):
                            clear_resource_caches()
                            st.session_state["aoi_resource_catalog_checked_at"] = 0.0
                            st.rerun(scope="fragment")
                        else:
                            st.error(_error(result, "配置删除失败。"))
    _render_memory_lifecycle(item)


configure_page("runtime_config")
sidebar_runtime_controls("runtime_config")

# Remove legacy page-level notices.  Operation feedback is now rendered only
# in the status column belonging to the initiating button.
st.session_state.pop("memory_config_notice", None)
st.session_state.pop("_action_feedback::runtime-config::page-notice", None)

st.markdown(
    """
    <style>
      .memory-config-hero{padding:1.2rem 1.35rem;border:1px solid #cbdde8;border-radius:18px;background:linear-gradient(135deg,#f8fbfd,#eef8fb);margin-bottom:1rem}
      .memory-config-hero h1{margin:0;color:#102840;font-size:2rem}.memory-config-hero p{margin:.45rem 0 0;color:#60748a}
    </style>
    <div class="memory-config-hero"><h1>运行配置</h1><p>以目标域记忆库为一级资源管理运行方案；每个记忆库内部维护一个或多个兼容配置版本。资源时间均按北京时间（Asia/Shanghai）显示。</p></div>
    """,
    unsafe_allow_html=True,
)

def _selection(selection_key: str) -> set[str]:
    selected = {str(value) for value in st.session_state.get(selection_key, []) if str(value)}
    st.session_state[selection_key] = sorted(selected)
    return selected


def _toggle_selection(selection_key: str, resource_id: str, widget_key: str) -> None:
    selected = _selection(selection_key)
    if bool(st.session_state.get(widget_key)):
        selected.add(resource_id)
    else:
        selected.discard(resource_id)
    st.session_state[selection_key] = sorted(selected)


def _set_page_selection(selection_key: str, rows: list[dict[str, Any]], checked: bool) -> None:
    selected = _selection(selection_key)
    for row in rows:
        resource_id = str(row.get("resource_id") or "")
        if not resource_id or row.get("active"):
            continue
        widget_key = f"{selection_key}::{resource_id}"
        st.session_state[widget_key] = checked
        if checked:
            selected.add(resource_id)
        else:
            selected.discard(resource_id)
    st.session_state[selection_key] = sorted(selected)


def _render_selectable_rows(rows: list[dict[str, Any]], selection_key: str) -> set[str]:
    selected = _selection(selection_key)
    header = st.columns([0.45, 0.55, 2.45, 1.0, 1.0, 0.9, 1.0, 1.55])
    time_label = "归档时间" if rows and all(bool(row.get("archived")) for row in rows) else "创建时间"
    for column, text in zip(header, ("选择", "序号", "记忆库名称", "数据集", "类别", "配置数量", "状态", time_label)):
        column.caption(text)
    for index, item in enumerate(rows, start=1):
        resource_id = str(item.get("resource_id") or "")
        widget_key = f"{selection_key}::{resource_id}"
        st.session_state[widget_key] = resource_id in selected
        columns = st.columns([0.45, 0.55, 2.45, 1.0, 1.0, 0.9, 1.0, 1.55], vertical_alignment="center")
        columns[0].checkbox(
            f"选择 {item.get('display_name') or resource_id}",
            key=widget_key,
            disabled=bool(item.get("active")),
            label_visibility="collapsed",
            on_change=_toggle_selection,
            args=(selection_key, resource_id, widget_key),
        )
        columns[1].write(index)
        columns[2].write(str(item.get("display_name") or item.get("profile_id") or "未命名记忆库"))
        columns[3].write(str(item.get("dataset") or "-"))
        columns[4].write(str(item.get("category") or "-"))
        columns[5].write(int((item.get("metadata") or {}).get("config_count") or 0))
        columns[6].write(str(item.get("lifecycle_status") or "可用"))
        timestamp = item.get("archived_at") or item.get("last_archived_at") if item.get("archived") else item.get("created_at")
        columns[7].write(format_beijing_datetime(timestamp))
    return _selection(selection_key)


def _render_current_runtime(resources: list[dict[str, Any]], current: dict[str, Any]) -> None:
    current_id = str(current.get("memory_bank_id") or "")
    current_item = next((item for item in resources if str(item.get("profile_id") or "") == current_id), None)
    with st.container(border=True):
        st.subheader("当前运行方案")
        if current_item is None:
            st.info("检测服务已启动，当前未加载目标域记忆库。")
            st.caption("请选择一个可用记忆库及其内部配置并应用，之后单图、批量和视频检测才会启用。")
            render_action_button_with_feedback(
                "停用当前运行方案",
                scope="runtime-deactivate",
                key="deactivate_runtime_profile_empty",
                disabled=True,
                icon=":material/pause_circle:",
            )
            return
        metadata = dict(current_item.get("metadata") or {})
        columns = st.columns(5)
        columns[0].metric("记忆库", current_item.get("display_name") or current_id)
        columns[1].metric("数据集 / 类别", f"{current_item.get('dataset') or '-'} / {current_item.get('category') or '-'}")
        columns[2].metric("配置", f"{current.get('config_profile_id') or '-'} · v{current.get('config_profile_version') or '-'}")
        columns[3].metric("方法模式", METHOD_LABELS.get(str(current_item.get("method_mode") or ""), current_item.get("method_mode") or "-"))
        columns[4].metric("状态", "当前使用")
        st.caption(f"完整性：{STATUS_LABELS.get(str(current_item.get('integrity_status') or ''), '正常')} · 资源大小：{float(metadata.get('resource_size_bytes') or current_item.get('size_bytes') or 0) / 1024 / 1024:.1f} MB")
        confirm = st.checkbox("我确认停用后所有新检测将被阻止，资源本身不会删除或归档", key="confirm_runtime_deactivate")
        action_scope = "runtime-deactivate"
        deactivate_clicked, action_status = render_action_button_with_feedback(
            "停用当前运行方案",
            scope=action_scope,
            key="deactivate_runtime_profile",
            disabled=not confirm,
            icon=":material/pause_circle:",
        )
        if deactivate_clicked:
            update_action_feedback(action_scope, "info", "正在等待活动任务结束并停用运行方案...", action_status)
            with st.spinner("正在等待活动任务结束并停用运行方案..."):
                response = client().runtime_profile_deactivate(confirm=True, actor="runtime_config_ui")
            if response.get("success"):
                result = dict(response.get("result") or {})
                _rerun_notice(
                    action_scope,
                    "当前运行方案已停用；记忆库和配置仍完整保留。",
                    "success",
                    resource_revision=int(result.get("resource_revision") or 0),
                )
            else:
                update_action_feedback(action_scope, "error", _error(response, "运行方案停用失败。"), action_status)


def _render_available_management(rows: list[dict[str, Any]]) -> None:
    st.subheader("可用记忆库")
    st.caption("当前使用的记忆库需先停用；已停用和其他可用记忆库可以直接归档。历史检测引用只作提示，不会永久阻止操作。")
    selection_key = "runtime_available_memory_bank_ids"
    left, right = st.columns([1, 5])
    left.button("全选当前页", on_click=_set_page_selection, args=(selection_key, rows, True), key="select_all_available_banks")
    right.button("清除选择", on_click=_set_page_selection, args=(selection_key, rows, False), key="clear_available_banks")
    selected = _render_selectable_rows(rows, selection_key)
    selected = {value for value in selected if any(str(row.get("resource_id")) == value and not row.get("active") for row in rows)}
    st.caption(f"已选择 {len(selected)} 个可归档记忆库。")
    action_scope = "runtime-batch-archive"
    archive_clicked, action_status = render_action_button_with_feedback(
        "批量归档所选记忆库",
        scope=action_scope,
        key="batch_archive_available_banks",
        icon=":material/archive:",
        disabled=not selected,
    )
    if archive_clicked:
        update_action_feedback(action_scope, "info", f"正在归档 {len(selected)} 个记忆库...", action_status)
        response = client().runtime_resource_batch_archive(sorted(selected), confirm=True, actor="runtime_config_ui")
        if response.get("success"):
            result = dict(response.get("result") or {})
            if int(result.get("failed_count") or 0):
                update_action_feedback(
                    action_scope,
                    "warning",
                    f"已归档 {int(result.get('success_count') or 0)} 个，失败 {int(result.get('failed_count') or 0)} 个。",
                    action_status,
                )
                _rerun_notice(action_scope, f"已归档 {int(result.get('success_count') or 0)} 个，{int(result.get('failed_count') or 0)} 个失败。", "warning")
            else:
                _rerun_notice(action_scope, f"已归档 {int(result.get('success_count') or 0)} 个记忆库。")
        else:
            update_action_feedback(action_scope, "error", _error(response, "批量归档失败。"), action_status)


def _render_archived_management(rows: list[dict[str, Any]]) -> None:
    st.subheader("已归档记忆库")
    st.caption("归档资源不会参与检测，可在这里恢复或永久删除。永久删除会同时移除其专属配置、索引、阈值、校准器和缺陷原型；共享基础模型不受影响。")
    if not rows:
        st.info("当前没有已归档记忆库。")
        return
    selection_key = "runtime_archived_memory_bank_ids"
    first, second, spacer = st.columns([1, 1, 4])
    first.button("全选当前页", on_click=_set_page_selection, args=(selection_key, rows, True), key="select_all_archived_banks")
    second.button("清除选择", on_click=_set_page_selection, args=(selection_key, rows, False), key="clear_archived_banks")
    selected = _render_selectable_rows(rows, selection_key)
    st.caption(f"已选择 {len(selected)} 个已归档记忆库。")
    restore_scope = "runtime-batch-restore"
    restore_clicked, restore_status = render_action_button_with_feedback(
        "批量恢复",
        scope=restore_scope,
        key="batch_restore_archived_banks",
        icon=":material/restore:",
        disabled=not selected,
    )
    if restore_clicked:
        update_action_feedback(restore_scope, "info", f"正在恢复 {len(selected)} 个记忆库...", restore_status)
        response = client().runtime_resource_batch_restore(sorted(selected), confirm=True, actor="runtime_config_ui")
        if response.get("success"):
            result = dict(response.get("result") or {})
            _rerun_notice(restore_scope, f"已恢复 {int(result.get('success_count') or 0)} 个记忆库。")
        else:
            update_action_feedback(restore_scope, "error", _error(response, "批量恢复失败。"), restore_status)
    if st.button("生成批量删除预览", icon=":material/preview:", disabled=not selected, key="batch_delete_preview_archived_banks"):
        with st.spinner("正在核对专属资源、历史引用和可释放空间..."):
            response = client().runtime_resource_batch_delete_preview(sorted(selected))
        if response.get("success"):
            st.session_state["runtime_archived_delete_plan"] = dict(response.get("preview") or {})
            st.session_state["runtime_archived_delete_ids"] = sorted(selected)
            st.session_state["batch_delete_archived_ack"] = False
        else:
            st.error(_error(response, "批量删除预览失败。"))
    plan = dict(st.session_state.get("runtime_archived_delete_plan") or {})
    planned_ids = list(st.session_state.get("runtime_archived_delete_ids") or [])
    if plan and planned_ids:
        response = render_delete_preview_confirmation(
            preview=plan,
            title="批量删除记忆库预览",
            action_label="永久删除所选记忆库",
            object_label="记忆库",
            acknowledgement_key="batch_delete_archived_ack",
            button_key="batch_delete_archived_banks",
            execute_callback=lambda payload: client().runtime_resource_batch_delete(
                planned_ids,
                preview_token=str(payload.get("preview_token") or ""),
                acknowledged=bool(payload.get("acknowledged")),
                confirm=True,
                reason="运行配置批量删除已归档记忆库",
                actor="runtime_config_ui",
            ),
            extra_summary=[("专属配置", "同步处理"), ("共享模型", "保留"), ("历史记录", "保留")],
        )
        if response is not None:
            if response.get("success"):
                result = dict(response.get("result") or {})
                st.session_state.pop("runtime_archived_delete_plan", None)
                st.session_state.pop("runtime_archived_delete_ids", None)
                clear_resource_caches()
                st.session_state["aoi_resource_catalog_checked_at"] = 0.0
                st.rerun(scope="fragment")
            else:
                st.error(_error(response, "批量删除失败。"))


def _render_deleted_records() -> None:
    with st.expander("已删除资源记录", expanded=False):
        if st.button("读取删除记录", key="load_memory_bank_tombstones"):
            response = client().runtime_resource_tombstones(100)
            if response.get("success"):
                st.session_state["runtime_memory_bank_tombstones"] = list(response.get("tombstones") or [])
            else:
                st.error(_error(response, "删除记录暂时无法读取。"))
        rows = [row for row in list(st.session_state.get("runtime_memory_bank_tombstones") or []) if str(row.get("resource_type") or "") == "target_profile"]
        if rows:
            st.dataframe(
                [
                    {
                        "序号": index,
                        "记忆库名称": row.get("display_name") or row.get("profile_id") or "-",
                        "数据集": row.get("dataset") or "-",
                        "类别": row.get("category") or "-",
                        "删除时间": format_beijing_datetime(row.get("deleted_at")),
                        "状态": "已删除",
                    }
                    for index, row in enumerate(rows, start=1)
                ],
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("尚未读取删除记录，或当前没有目标域记忆库删除记录。")

@st.fragment
def _render_runtime_configuration_region() -> None:
    resources, current = _load_resources()
    if not resources:
        _render_current_runtime([], current)
        st.info("当前没有可管理的目标域记忆库，请先在“目标域适配”中创建。")
        return

    _render_current_runtime(resources, current)
    available = [item for item in resources if not item.get("archived")]
    archived = [item for item in resources if item.get("archived")]
    manage_tab, archive_tab = st.tabs(["可用记忆库与配置", "已归档记忆库"])
    with manage_tab:
        _render_available_management(available)
        resource_map = {str(item["resource_id"]): item for item in available}
        if resource_map:
            st.divider()
            selected_resource = st.selectbox("查看和配置记忆库", list(resource_map), format_func=lambda key: _bank_label(resource_map[key]), key="runtime_memory_bank_selected")
            _render_bank(resource_map[selected_resource], current)
    with archive_tab:
        _render_archived_management(archived)
        _render_deleted_records()


_render_runtime_configuration_region()
