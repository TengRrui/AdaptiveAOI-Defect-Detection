from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Callable

import streamlit as st

from core.domain_build_contract import (
    CREATE_MODE_LABELS,
    DEFAULT_CREATE_MODE,
    LIGHTWEIGHT_CREATE_MODE,
    default_feature_layers,
)
from performance_cache import api_client as client
from delete_preview_confirmation import render_delete_preview_confirmation
from domain_draft_autosave import (
    autosave_failure,
    clear_autosave_state,
    queue_autosave,
    retry_failed_autosave,
    run_pending_autosave,
)
from friendly_display import friendly_error_message
from time_utils import format_beijing_datetime
from ui_common import (
    render_action_button_with_feedback,
    render_action_feedback,
    set_action_feedback,
    update_action_feedback,
)


MODE_LABELS = dict(CREATE_MODE_LABELS)
HISTORICAL_MODE_LABELS = {
    **MODE_LABELS,
    "normal_memory": "历史基础正常记忆",
    "normal_defect_calibration": "历史缺陷阈值校准方式",
    "normal_defect_dual_memory": "历史双记忆方式",
    "dual_memory": "历史双记忆方式",
    "auto": "历史自动选择方式",
}
MODE_MIGRATION_REQUIRED = "__select_current_build_mode__"
THRESHOLD_LABELS = {
    "best_f1": "综合 F1 最优",
    "recall_priority": "召回率优先",
    "normal_percentile": "正常样本分位数",
    "mean_std": "正常均值与标准差",
}
FIELD_KEYS = {
    "friendly_name": "domain_rc_name", "dataset": "domain_rc_dataset", "category": "domain_rc_category",
    "method_mode": "domain_rc_mode",
    "image_size": "domain_rc_image_size", "device": "domain_rc_device", "use_fp16": "domain_rc_fp16",
    "coreset_ratio": "domain_rc_coreset", "max_memory_patches": "domain_rc_max_memory", "knn_k": "domain_rc_knn",
    "memory_backend": "domain_rc_memory_backend", "memory_backend_fallback": "domain_rc_fallback",
    "memory_ivf_nlist": "domain_rc_nlist", "memory_ivf_nprobe": "domain_rc_nprobe", "score_aggregation": "domain_rc_aggregation",
    "threshold_method": "domain_rc_threshold_method", "threshold_percentile": "domain_rc_percentile",
    "threshold_target_recall": "domain_rc_recall", "threshold_max_false_positive_rate": "domain_rc_fpr",
    "pixel_threshold_mode": "domain_rc_pixel_threshold", "defect_coreset_ratio": "domain_rc_defect_coreset",
    "max_defect_prototypes": "domain_rc_prototypes", "defect_top_ratio": "domain_rc_defect_ratio",
    "defect_region_filter": "domain_rc_defect_filter", "dual_fusion_strategy": "domain_rc_fusion",
    "lightweight_gate_percentile": "domain_rc_gate", "lightweight_l2": "domain_rc_l2",
    "lightweight_score_scale": "domain_rc_scale", "lightweight_cv_folds": "domain_rc_cv",
    "random_seed": "domain_rc_seed", "note": "domain_rc_note",
}
SOURCE_KEYS = {
    "normal_source_type": "domain:normal:source_type", "defect_source_type": "domain:defect:source_type",
    "normal_image_dir": "domain:normal:image_dir", "normal_dir": "domain:normal:image_dir",
    "defect_image_dir": "domain:defect:image_dir", "defect_dir": "domain:defect:image_dir",
    "normal_video_files": "domain:normal:video_files", "defect_video_files": "domain:defect:video_files",
    "normal_video_dir": "domain:normal:video_dir", "defect_video_dir": "domain:defect:video_dir",
}
DEFAULTS = {
    "domain_rc_name": "", "domain_rc_dataset": "custom", "domain_rc_category": "target_domain", "domain_rc_mode": DEFAULT_CREATE_MODE,
    "domain_rc_layers": default_feature_layers(), "domain_rc_batch": 1,
    "domain_rc_image_size": 256, "domain_rc_device": "auto", "domain_rc_fp16": True,
    "domain_rc_coreset": 0.1, "domain_rc_max_memory": 50000, "domain_rc_knn": 1, "domain_rc_memory_backend": "auto", "domain_rc_fallback": "faiss_cpu_flat",
    "domain_rc_nlist": 256, "domain_rc_nprobe": 32, "domain_rc_aggregation": "maximum_patch", "domain_rc_threshold_method": "best_f1",
    "domain_rc_percentile": 99.0, "domain_rc_recall": 0.95, "domain_rc_fpr": 0.1, "domain_rc_pixel_threshold": "automatic_from_normal_map",
    "domain_rc_defect_coreset": 0.25, "domain_rc_prototypes": 32, "domain_rc_defect_ratio": 0.05, "domain_rc_defect_filter": "high_anomaly_patches",
    "domain_rc_fusion": "reliability_gated_relative_distance", "domain_rc_gate": 95.0, "domain_rc_l2": 0.01, "domain_rc_scale": 1.0, "domain_rc_cv": 5,
    "domain_rc_seed": 42, "domain_rc_note": "",
}

BASIC_INFO_KEYS = (
    "domain_rc_name",
    "domain_rc_dataset",
    "domain_rc_category",
    "domain_rc_note",
)


def _error(response: dict[str, Any], fallback: str) -> str:
    return friendly_error_message(response.get("error"), fallback)


def _human_size(value: Any) -> str:
    size = float(value or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024
    return "0 B"


def _duration(value: Any) -> str:
    try:
        seconds = max(0, int(round(float(value or 0))))
    except (TypeError, ValueError):
        return "--:--:--"
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _metric_value(data: Any, *names: str) -> Any:
    wanted = {str(name).casefold() for name in names}
    if isinstance(data, dict):
        for key, value in data.items():
            if str(key).casefold() in wanted and value is not None:
                return value
        for value in data.values():
            nested = _metric_value(value, *names)
            if nested is not None:
                return nested
    elif isinstance(data, list):
        for value in data:
            nested = _metric_value(value, *names)
            if nested is not None:
                return nested
    return None


def _number(value: Any, digits: int = 4, suffix: str = "") -> str:
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "暂无"


def _file_ready(hashes: dict[str, Any], *names: str) -> bool:
    return any(bool(hashes.get(name)) for name in names)


def _resource_state(ready: bool, *, optional: bool = False) -> str:
    if ready:
        return "✅ 完整"
    return "— 当前模式不适用" if optional else "❌ 不完整"


def _public_build_stages(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse internal safety work into operator-facing build phases."""
    groups = (
        ("sample_check", "样本检查", ("initialization", "sample_validation")),
        ("video_processing", "视频处理", ("video_preparation",)),
        ("feature_extraction", "特征提取", ("feature_extraction",)),
        ("memory_resources", "Memory Bank 与检索资源生成", ("memory_index",)),
        (
            "target_resource",
            "创建目标域资源",
            ("threshold_calibration", "defect_extension", "config_profile", "finalizing", "integrity", "registration"),
        ),
    )
    indexed = {str(stage.get("stage_id") or ""): dict(stage) for stage in stages}
    public: list[dict[str, Any]] = []
    for stage_id, stage_name, member_ids in groups:
        members = [indexed[value] for value in member_ids if value in indexed]
        if not members:
            continue
        statuses = [str(member.get("status") or "pending") for member in members]
        applicable = [member for member in members if str(member.get("status") or "") != "skipped"]
        if not applicable:
            status = "skipped"
        elif "failed" in statuses:
            status = "failed"
        elif "running" in statuses:
            status = "running"
        elif all(value in {"completed", "warning", "skipped"} for value in statuses):
            status = "warning" if "warning" in statuses else "completed"
        else:
            status = "pending"
        total_weight = sum(float(member.get("weight") or 1.0) for member in applicable)
        completed_weight = 0.0
        for member in applicable:
            weight = float(member.get("weight") or 1.0)
            member_status = str(member.get("status") or "pending")
            if member_status in {"completed", "warning"}:
                completed_weight += weight
            elif member_status == "running":
                completed_weight += weight * min(1.0, max(0.0, float(member.get("progress") or 0.0)))
        running = next((member for member in members if str(member.get("status") or "") == "running"), None)
        message_source = running or next(
            (member for member in reversed(members) if str(member.get("status") or "") in {"completed", "warning", "failed"}),
            members[0],
        )
        if stage_id == "target_resource" and status == "running":
            message = "正在完成资源创建..."
        elif stage_id == "target_resource" and status == "completed":
            message = "目标域资源已创建并可使用"
        elif status == "skipped":
            message = str(message_source.get("message") or "当前任务不适用")
        else:
            message = str(message_source.get("message") or "等待执行")
        public.append(
            {
                "stage_id": stage_id,
                "stage_name": stage_name,
                "status": status,
                "progress": completed_weight / total_weight if total_weight else 0.0,
                "started_at": next((member.get("started_at") for member in members if member.get("started_at")), None),
                "elapsed_seconds": sum(float(member.get("elapsed_seconds") or 0.0) for member in members),
                "message": message,
            }
        )
    return public


def _render_build_result(result: dict[str, Any]) -> None:
    mode = str(result.get("method_mode_resolved") or result.get("method_mode_requested") or "normal_only")
    hashes = dict(result.get("file_sha256") or {})
    coreset = dict(result.get("coreset") or {})
    calibration = dict(result.get("calibration_result") or result.get("calibration_metrics") or {})
    validation_result = dict(result.get("validation_result") or {})
    complexity = dict(result.get("complexity_summary") or {})

    st.markdown("### 构建结果摘要")
    first = st.columns(4)
    first[0].metric("目标域名称", str(result.get("friendly_name") or "未命名目标域"))
    first[1].metric("目标域版本", f"v{int(result.get('profile_version') or 1)}")
    first[2].metric("正常样本", int(result.get("normal_sample_count") or 0))
    first[3].metric("缺陷样本", int(result.get("defect_calibration_sample_count") or 0))
    second = st.columns(4)
    second[0].metric("记忆库条目", int(coreset.get("memory_patch_count") or 0))
    second[1].metric("记忆库大小", _human_size(result.get("resource_size_bytes")))
    second[2].metric("构建方式", HISTORICAL_MODE_LABELS.get(mode, "正常记忆异常检测"))
    profile_status = str(result.get("profile_status") or "candidate")
    second[3].metric("构建状态", "候选版本已生成" if profile_status == "candidate" else "构建完成")

    st.markdown("### 资源完整性")
    has_calibrator = mode == LIGHTWEIGHT_CREATE_MODE
    resource_rows = [
        {"资源": "正常记忆库", "状态": _resource_state(_file_ready(hashes, "memory_bank.pt")), "说明": "正常样本特征与检索资源"},
        {"资源": "目标域配置快照", "状态": _resource_state(_file_ready(hashes, "effective_config.yaml") and bool(result.get("config_profile_id"))), "说明": "与当前目标域版本唯一绑定"},
        {"资源": "正常特征索引", "状态": _resource_state(_file_ready(hashes, "normal_coreset_index.pt", "coreset_indices.pt")), "说明": "用于最近邻检索"},
        {"资源": "图像级阈值", "状态": _resource_state(_file_ready(hashes, "threshold.json") and result.get("threshold") is not None), "说明": "用于最终正常/异常判定"},
        {"资源": "像素级阈值", "状态": _resource_state(result.get("pixel_threshold") is not None), "说明": "用于缺陷区域可视化"},
        {"资源": "轻量校准器", "状态": _resource_state(_file_ready(hashes, "lightweight_calibrator.json"), optional=not has_calibrator), "说明": "仅轻量校准模式需要"},
    ]
    resource_columns = st.columns(2)
    for index, item in enumerate(resource_rows):
        with resource_columns[index % 2].container(border=True):
            st.markdown(f"**{item['资源']}**　{item['状态']}")
            st.caption(item["说明"])
    required_states = [str(item["状态"]) for item in resource_rows if "不适用" not in str(item["状态"])]
    if required_states and all(state.startswith("✅") for state in required_states):
        st.success("综合状态：后台完整性确认通过，目标域资源可以安全应用。")
    elif any(state.startswith("❌") for state in required_states):
        st.error("综合状态：资源不完整，请先根据缺失项修复后再验证。")
    else:
        st.warning("综合状态：资源可用但仍有待确认项。")

    st.markdown("### 文件完整性校验")
    valid_hashes = [str(value) for value in hashes.values() if len(str(value)) == 64]
    integrity = str(result.get("resource_integrity_status") or "pending")
    checks = st.columns(4)
    checks[0].metric("已登记校验文件", len(valid_hashes))
    checks[1].metric("校验状态", "✅ 通过" if valid_hashes and integrity in {"passed", "pending"} else "❌ 未通过")
    checks[2].metric("配置摘要", (str(result.get("config_snapshot_sha256") or result.get("config_hash") or "")[:12] or "暂无"))
    validated_at = (result.get("candidate_validation") or {}).get("validated_at") or result.get("created_at")
    checks[3].metric("最近校验", format_beijing_datetime(validated_at, fallback="完成构建时"))
    st.caption("完整哈希仅在技术详情中提供；普通视图只展示校验结果和短摘要。")
    with st.expander("技术详情：完整 SHA-256", expanded=False):
        st.dataframe(
            [{"资源文件": name, "SHA-256": str(digest)} for name, digest in sorted(hashes.items())],
            hide_index=True,
            width="stretch",
        )

    st.markdown("### 判定参数与性能影响")
    params = st.columns(5)
    params[0].metric("图像级阈值", _number(result.get("threshold")))
    params[1].metric("像素级阈值", _number(result.get("pixel_threshold")))
    threshold_method = str(result.get("threshold_method") or "normal_percentile")
    params[2].metric("校准方式", THRESHOLD_LABELS.get(threshold_method, "正常分布估计"))
    params[3].metric("最近邻数量", int(result.get("knn_k") or 1))
    params[4].metric("记忆条目", int(coreset.get("memory_patch_count") or 0))
    aggregation = str(result.get("score_aggregation") or "maximum_patch")
    feature_switches = st.columns(3)
    feature_switches[0].metric("分数聚合", "Top-K 均值" if aggregation == "top_k_mean" else "最大补丁分数")
    feature_switches[1].metric("缺陷校准", "已启用" if mode != "normal_only" else "未启用")
    feature_switches[2].metric("轻量校准", "已启用" if has_calibrator else "未启用")
    st.caption("阈值决定最终正常/异常边界；聚合方式决定补丁分数如何汇总。轻量校准资源只在所选构建方式需要时启用。")
    before_f1 = _metric_value(calibration, "before_f1", "baseline_f1", "f1_before")
    after_f1 = _metric_value(calibration, "after_f1", "calibrated_f1", "best_f1", "f1")
    precision = _metric_value(calibration, "precision", "best_precision")
    recall = _metric_value(calibration, "recall", "best_recall")
    ann_queries = _metric_value(complexity, "ann_query_count", "ann_queries", "additional_ann_count")
    memory_bytes = _metric_value(complexity, "memory_bytes", "memory_bank_bytes", "resource_size_bytes")
    speed_cost = _metric_value(complexity, "latency_increase_percent", "speed_loss_percent", "runtime_overhead_percent")
    st.markdown("#### 校准集指标（非最终测试精度）")
    st.caption("以下指标仅用于使用 30 张已标注缺陷样本选择阈值或轻量参数；未参与独立测试，不能作为独立测试精度。")
    calibration_cards = st.columns(4)
    calibration_cards[0].metric("校准集·调整前 F1", _number(before_f1))
    calibration_cards[1].metric("校准集·调整后 F1", _number(after_f1))
    calibration_cards[2].metric("校准集·Precision", _number(precision))
    calibration_cards[3].metric("校准集·Recall", _number(recall))
    if str(validation_result.get("status") or "not_run") == "not_run":
        st.info("独立验证结果：尚未执行。1000+ 正式测试数据只用于冻结后的只读推理，不会参与建库、校准或参数更新。")
    performance_cards = st.columns(4)
    performance_cards[0].metric("记忆资源占用", _human_size(memory_bytes) if memory_bytes is not None else _human_size(result.get("resource_size_bytes")))
    performance_cards[1].metric("特征条目数量", int(coreset.get("memory_patch_count") or 0))
    performance_cards[2].metric("ANN 查询次数", _number(ann_queries, 0))
    performance_cards[3].metric("相对额外开销", _number(speed_cost, 2, "%") if speed_cost is not None else "暂无实测")
    within_budget = _metric_value(complexity, "within_budget", "complexity_budget_passed", "budget_passed")
    if within_budget is True:
        st.success("复杂度预算：满足当前预算。")
    elif within_budget is False:
        st.error("复杂度预算：未满足当前预算，候选版本不能直接应用。")
    else:
        st.info("复杂度预算：本次结果未提供独立预算结论；系统不会据此虚构性能结果。")


def _restore_payload(payload: dict[str, Any]) -> None:
    # An autosave fingerprint is scoped to the draft that produced it.  A
    # loaded draft must never inherit another draft's duplicate-suppression or
    # failed-request state.
    clear_autosave_state(st.session_state)
    for name, key in FIELD_KEYS.items():
        if name in payload:
            if name == "method_mode":
                restored_mode = str(payload[name] or "").strip().lower()
                if restored_mode in MODE_LABELS:
                    st.session_state[key] = restored_mode
                    st.session_state.pop("domain_rc_legacy_mode", None)
                    st.session_state.pop("domain_rc_mode_migration_required", None)
                else:
                    # Historical drafts remain readable, but a retired mode is
                    # never silently rewritten into a different algorithm.
                    st.session_state[key] = MODE_MIGRATION_REQUIRED
                    st.session_state["domain_rc_legacy_mode"] = restored_mode
                    st.session_state["domain_rc_mode_migration_required"] = True
                continue
            st.session_state[key] = payload[name]
    for name, key in SOURCE_KEYS.items():
        if name in payload:
            st.session_state[key] = payload[name]
    sampling = dict(payload.get("video_sampling_config") or {})
    if sampling:
        st.session_state["domain_video_sampling_config"] = sampling
        for key in list(st.session_state):
            if str(key).startswith("domain:sampling:"):
                st.session_state.pop(key, None)
    st.session_state["domain_sample_budget_config"] = {
        "sample_budget_mode": payload.get("sample_budget_mode", "final_frame_count"),
        "normal_target_count": int(payload.get("normal_target_count", 100)),
        "defect_target_count": int(payload.get("defect_target_count", 30)),
    }
    for key in list(st.session_state):
        if str(key).startswith("domain:budget:"):
            st.session_state.pop(key, None)


def _payload(source_payload: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    payload = dict(source_payload())
    # Target-domain creation no longer accepts reviewed Feedback selections
    # from this page.  Drop the legacy field even if an older source callback
    # still exposes it during a rolling frontend/backend upgrade.
    payload.pop("feedback_candidate_ids", None)
    for name, key in FIELD_KEYS.items():
        value = st.session_state.get(key, DEFAULTS.get(key))
        if name == "method_mode" and value == MODE_MIGRATION_REQUIRED:
            value = ""
        payload[name] = value
    payload.update(
        layers=list(st.session_state.get("domain_rc_layers") or default_feature_layers()),
        batch_size=int(st.session_state.get("domain_rc_batch", 1)),
        precision="fp16" if bool(st.session_state.get("domain_rc_fp16", True)) else "fp32",
        config_profile_name="初始化配置",
        # The backend completes integrity checks, immutable freeze and catalog
        # registration before reporting success.  Applying remains an explicit
        # operator decision.
        publish_mode="frozen",
        sample_note=str(st.session_state.get("domain_rc_note") or ""),
    )
    return payload


def _basic_info_values() -> dict[str, str]:
    return {key: str(st.session_state.get(key) or "") for key in BASIC_INFO_KEYS}


def _queue_basic_info_autosave(changed_field: str) -> None:
    queue_autosave(st.session_state, _basic_info_values(), changed_field=changed_field)


def _retry_basic_info_autosave() -> None:
    retry_failed_autosave(st.session_state)


def _render_basic_info_editor() -> None:
    c1, c2, c3 = st.columns(3)
    c1.text_input(
        "目标域名称",
        key="domain_rc_name",
        help="用于识别此次完整 profile；同名资源不会被覆盖。",
        on_change=_queue_basic_info_autosave,
        args=("domain_rc_name",),
    )
    c2.text_input(
        "数据集",
        key="domain_rc_dataset",
        help="目标域的受控业务名称，可以是实际产线的业务名称。",
        on_change=_queue_basic_info_autosave,
        args=("domain_rc_dataset",),
    )
    c3.text_input(
        "类别",
        key="domain_rc_category",
        help="目标产品类别；实际值会原样写入配置快照。",
        on_change=_queue_basic_info_autosave,
        args=("domain_rc_category",),
    )
    st.text_area(
        "样本备注",
        key="domain_rc_note",
        help="记录来源、采集条件和人工确认信息，不写入检测数据。",
        on_change=_queue_basic_info_autosave,
        args=("domain_rc_note",),
    )


def _process_basic_info_autosave(source_payload: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    outcome = run_pending_autosave(
        st.session_state,
        lambda _snapshot: _save_draft(source_payload, quiet=True),
    )
    if outcome.get("status") == "saved":
        st.toast("已自动保存", icon="✅")
    return outcome


def _render_basic_info_autosave_error() -> None:
    failure = autosave_failure(st.session_state)
    if not failure:
        return
    status, action = st.columns([4, 1], vertical_alignment="center")
    status.error(f"自动保存失败：{failure.get('message') or '后端暂不可用'}。当前输入已保留。")
    action.button(
        "重试保存",
        key="domain_rc_autosave_retry",
        on_click=_retry_basic_info_autosave,
        width="stretch",
    )


def _save_draft(
    source_payload: Callable[[], dict[str, Any]],
    *,
    quiet: bool = False,
    action_status: Any | None = None,
) -> dict[str, Any]:
    if str(st.session_state.get("domain_rc_mode") or "") not in MODE_LABELS:
        response = {
            "success": False,
            "error": {"message": "该草稿使用已停止新建的历史构建方式。请先明确选择当前支持的构建方式。"},
        }
        if not quiet:
            update_action_feedback(
                "domain-draft::action",
                "warning",
                str(response["error"]["message"]),
                action_status,
            )
        return response
    response = client().save_domain_draft({
        "draft_id": st.session_state.get("domain_rc_draft_id") or None,
        "current_step": 1,
        "payload": _payload(source_payload),
    })
    if response.get("success"):
        draft = response.get("draft") or {}
        draft_id = str(draft.get("draft_id") or "")
        if not draft_id:
            response = {
                "success": False,
                "error": {"message": "草稿接口未返回持久化标识。"},
            }
        else:
            st.session_state["domain_rc_draft_id"] = draft_id
            st.query_params["domain_draft"] = draft_id
        if not quiet:
            if response.get("success"):
                set_action_feedback(
                    "domain-draft::action",
                    "success",
                    f"草稿已保存（{draft_id[:12]}）。页面刷新或后端重启后仍可恢复。",
                )
            else:
                update_action_feedback(
                    "domain-draft::action",
                    "error",
                    _error(response, "目标域草稿保存失败。"),
                    action_status,
                )
    if not response.get("success") and not quiet:
        update_action_feedback(
            "domain-draft::action",
            "error",
            _error(response, "目标域草稿保存失败。"),
            action_status,
        )
    return response


def _restore_draft_from_query(index: dict[str, dict[str, Any]]) -> bool:
    """Restore the last autosaved draft after refresh or page re-entry."""

    if str(st.session_state.get("domain_rc_draft_id") or ""):
        return False
    draft_id = str(st.query_params.get("domain_draft") or "")
    draft = index.get(draft_id) if draft_id else None
    if draft_id and not draft:
        st.query_params.pop("domain_draft", None)
    if not draft and index:
        # The backend returns drafts in most-recently-updated order, but use
        # the persisted event time as the explicit fact source as well.
        draft = max(index.values(), key=lambda item: str(item.get("updated_at") or ""))
        draft_id = str(draft.get("draft_id") or "")
    if not draft or not draft_id:
        return False
    _restore_payload(dict(draft.get("payload") or {}))
    st.session_state["domain_rc_draft_id"] = draft_id
    st.query_params["domain_draft"] = draft_id
    return True


def _clear_draft_session_state(draft_id: str, *, clear_build_state: bool) -> None:
    """Return the editor to a genuinely new-draft state after deletion.

    The persisted JSON is the only draft fact source.  Streamlit widget state
    is deliberately cleared here so a rerun cannot repopulate the deleted
    draft from stale paths, scan summaries, selections, or an old job id.
    The recent-directory preference is intentionally retained.
    """
    current_draft_id = str(st.session_state.get("domain_rc_draft_id") or "")
    if current_draft_id != str(draft_id):
        # Deleting a row that is only selected in the draft list must not
        # erase an independently loaded draft (or unsaved new-form values).
        # Only discard the acknowledgement/preview belonging to that row.
        st.session_state.pop(f"domain_draft_delete_ack_{draft_id}", None)
        preview = st.session_state.get("domain_rc_draft_delete_preview")
        if isinstance(preview, dict) and str(preview.get("draft_id") or "") == str(draft_id):
            st.session_state.pop("domain_rc_draft_delete_preview", None)
        return
    clear_autosave_state(st.session_state)
    if str(st.query_params.get("domain_draft") or "") == str(draft_id):
        st.query_params.pop("domain_draft", None)
    transient_keys = set(FIELD_KEYS.values()) | set(SOURCE_KEYS.values()) | {
        "domain_rc_draft_id",
        "domain_rc_legacy_mode",
        "domain_rc_mode_migration_required",
        "domain_rc_last_rendered_mode",
        "domain_rc_draft_delete_preview",
        "domain_rc_conditions",
        "domain_rc_conditions_fingerprint",
        "domain_rc_layers",
        "domain_rc_batch",
        "domain_rc_job_notice",
        "domain_rc_scan_notice",
        "domain_scan_summary",
        "domain_sample_scan_pending",
        "domain_sample_scan_error",
        "domain_video_preview",
        "domain_video_excluded_sample_ids",
        "domain_video_sampling_config",
        "domain_sample_budget_config",
        "domain_sample_budget_mode",
        "domain_config_profile_key",
        "domain_config_profile_pending_key",
        "domain_normal_dialog_id",
        "domain_defect_dialog_id",
        "domain_normal_video_file_dialog_id",
        "domain_defect_video_file_dialog_id",
        "domain_normal_video_dir_dialog_id",
        "domain_defect_video_dir_dialog_id",
        "domain_normal_video_paths_text",
        "domain_defect_video_paths_text",
        "domain_defect_video_intervals_text",
    }
    if clear_build_state:
        transient_keys.update({
            "domain_adaptation_job_id",
            "domain_adaptation_job_active",
            "domain_rc_candidate",
            "domain_rc_candidate_validated",
            "domain_rc_candidate_frozen",
            "domain_rc_candidate_applied",
            "domain_rc_apply_result",
            "domain_rc_completed_job_id",
            "domain_rc_completed_elapsed",
            "domain_rc_build_idempotency_key",
            "domain_rc_build_fingerprint",
        })
    for key in list(st.session_state.keys()):
        if str(key).startswith("domain_draft_delete_ack_"):
            transient_keys.add(str(key))
        if str(key).startswith(("domain:normal:", "domain:defect:", "domain:sampling:", "domain:budget:")):
            transient_keys.add(str(key))
    for key in transient_keys:
        st.session_state.pop(key, None)
    if clear_build_state:
        st.query_params.pop("domain_job", None)
        st.query_params.pop("domain_candidate", None)


@st.fragment
def _draft_bar(source_payload: Callable[[], dict[str, Any]]) -> None:
    response = client().domain_drafts()
    drafts = list(response.get("drafts") or []) if response.get("success") else []
    options = [""] + [str(item["draft_id"]) for item in drafts]
    index = {str(item["draft_id"]): item for item in drafts}
    _restore_draft_from_query(index)
    current = str(st.session_state.get("domain_rc_draft_id") or "")
    if current and current not in options and response.get("success"):
        # Another tab/session may have removed the persisted draft.  Do not
        # keep a phantom option that can later save old fields back under a
        # new draft id; reconcile the editor with the backend fact source.
        _clear_draft_session_state(current, clear_build_state=True)
        current = ""
    elif current not in options:
        options.append(current)
    columns = st.columns([3.2, 1, 1, 1, 2], vertical_alignment="bottom")
    selected = columns[0].selectbox("目标域构建草稿", options, index=options.index(current), format_func=lambda value: "新建草稿" if not value else f"{index.get(value, {}).get('name') or value} · {index.get(value, {}).get('updated_at') or ''}")
    draft_delete_preview = dict(st.session_state.get("domain_rc_draft_delete_preview") or {})
    if draft_delete_preview and str(draft_delete_preview.get("draft_id") or "") != str(selected or ""):
        # A preview belongs to exactly one immutable draft.  Switching the
        # selector must not leave a stale final-action button attached to a
        # different row.
        st.session_state.pop("domain_rc_draft_delete_preview", None)
        draft_delete_preview = {}
    if columns[1].button("加载", icon=":material/folder_open:", width="stretch", disabled=not selected):
        draft = index.get(selected) or {}
        _restore_payload(dict(draft.get("payload") or {}))
        st.session_state["domain_rc_draft_id"] = selected
        st.query_params["domain_draft"] = selected
        st.session_state.pop("domain_rc_conditions", None)
        st.session_state.pop("domain_rc_conditions_fingerprint", None)
        # Loading replaces fields owned by the editor outside this fragment.
        # A fragment-only rerun would update the selector but leave stale form
        # widgets on screen, making a persisted draft look as if it did not
        # load.  Carry feedback through the one intentional page rerun.
        set_action_feedback(
            "domain-draft::action",
            "success",
            "草稿已加载，表单、样本路径和构建参数已恢复。",
        )
        st.rerun()
    with columns[4]:
        draft_action_status = st.empty()
    render_action_feedback("domain-draft::action", draft_action_status)
    if columns[2].button("保存草稿", icon=":material/save:", width="stretch"):
        update_action_feedback("domain-draft::action", "info", "正在保存草稿...", draft_action_status)
        saved = _save_draft(source_payload, action_status=draft_action_status)
        if saved.get("success"):
            # Re-read the persisted list in this small fragment so the new
            # draft is immediately selectable without rebuilding the editor.
            st.rerun(scope="fragment")
    if columns[3].button("删除草稿", icon=":material/delete:", width="stretch", disabled=not selected):
        preview_response = client().domain_draft_delete_preview(selected)
        if preview_response.get("success"):
            st.session_state["domain_rc_draft_delete_preview"] = dict(preview_response.get("preview") or {})
            st.session_state[f"domain_draft_delete_ack_{selected}"] = False
            draft_delete_preview = dict(preview_response.get("preview") or {})
        else:
            st.error(_error(preview_response, "草稿删除预览失败。"))
    if draft_delete_preview and str(draft_delete_preview.get("draft_id") or "") == str(selected or ""):
        deleted = render_delete_preview_confirmation(
            preview=draft_delete_preview,
            title="删除草稿预览",
            action_label="删除该目标域草稿",
            object_label="草稿",
            acknowledgement_key=f"domain_draft_delete_ack_{selected}",
            button_key=f"domain_draft_delete_execute_{selected}",
            execute_callback=lambda payload: client().delete_domain_draft(
                selected,
                acknowledged=True,
                plan_id=str(payload.get("plan_id") or ""),
                preview_token=str(payload.get("preview_token") or ""),
            ),
            extra_summary=[("正式 Memory Bank", "不受影响"), ("检测记录", "不受影响")],
        )
        if deleted is not None:
            if deleted.get("success"):
                clear_build_state = str(st.session_state.get("domain_rc_draft_id") or "") == str(selected)
                _clear_draft_session_state(str(selected), clear_build_state=clear_build_state)
                st.session_state.pop("domain_rc_draft_delete_preview", None)
                # The final rerun is what removes stale widgets/list rows;
                # carry the success message across it so the operator gets a
                # visible result instead of a silent reset.
                set_action_feedback(
                    "domain-draft::action",
                    "success",
                    "草稿已删除。已清理草稿状态；原始样本、检测记录和正式资源未受影响。",
                )
                st.rerun(scope="fragment")
            else:
                st.error(_error(deleted, "草稿删除失败。"))


def _help(text: str, default: str, effect: str) -> str:
    return f"{text} 默认值：{default}。{effect}"


def _config_form() -> bool:
    migration_required = bool(st.session_state.get("domain_rc_mode_migration_required"))
    mode_options = ([MODE_MIGRATION_REQUIRED] if migration_required else []) + list(MODE_LABELS)
    mode = st.selectbox(
        "构建模式",
        mode_options,
        key="domain_rc_mode",
        format_func=lambda value: "请选择当前构建模式" if value == MODE_MIGRATION_REQUIRED else MODE_LABELS[value],
        help="基础正常记忆仅使用正常样本；缺陷引导轻量校准会额外使用已确认缺陷样本拟合小型校准器，但不改变骨干网络。",
    )
    previous_mode = str(st.session_state.get("domain_rc_last_rendered_mode") or "")
    mode_changed = previous_mode != mode
    if mode_changed:
        st.session_state["domain_rc_last_rendered_mode"] = mode
        st.session_state.pop("domain_scan_summary", None)
        st.session_state.pop("domain_rc_conditions", None)
        st.session_state.pop("domain_rc_conditions_fingerprint", None)
    if mode == MODE_MIGRATION_REQUIRED:
        legacy_mode = str(st.session_state.get("domain_rc_legacy_mode") or "")
        legacy_label = HISTORICAL_MODE_LABELS.get(legacy_mode, "历史构建方式")
        st.warning(f"当前草稿使用已停止新建的“{legacy_label}”。继续保存或构建前，请明确选择上方任一当前构建模式；系统不会自动改变算法模式。")
    elif migration_required:
        st.session_state.pop("domain_rc_legacy_mode", None)
        st.session_state.pop("domain_rc_mode_migration_required", None)
    st.caption("轻量校准只调整受控分数映射与门控参数，不训练或替换基础特征网络。")
    st.info(
        "本页使用缺陷校准集得到的 F1、Recall 等仅属于校准指标，不是最终测试精度；"
        "正式 1000+ 测试数据只能在冻结 Profile 下执行只读推理。"
    )
    st.markdown("##### 基础配置")
    st.caption("基础特征资源由系统正式模型注册服务自动解析并校验，无需手动选择；当前运行方案不会影响本次建库使用的基础资源。")
    c1, c2 = st.columns([2, 1])
    c1.multiselect("特征层", ["layer1", "layer2", "layer3", "layer4"], key="domain_rc_layers", help=_help("选择用于构建正常记忆库的特征层。", "仅 layer2", "增加更多层会提高特征维度、记忆库大小和检索开销。"))
    c2.number_input("输入尺寸", 64, 2048, key="domain_rc_image_size", help=_help("统一预处理尺寸。", "256", "增大有利于小缺陷，但降低速度并增加显存。"))
    c1, c2, c3 = st.columns(3)
    c1.selectbox("运行设备", ["auto", "cuda", "cpu"], key="domain_rc_device", help="仅控制本次离线构建；最终 profile 会保存实际解析设备。")
    c2.number_input("特征批大小", 1, 32, key="domain_rc_batch", help=_help("一次提取的图像数。", "1", "增大可提升吞吐，但增加显存。"))
    c3.selectbox(
        "构建精度",
        [True, False],
        key="domain_rc_fp16",
        format_func=lambda value: "半精度（FP16）" if value else "单精度（FP32）",
        help="默认使用半精度提取特征；后端仅在实际设备不支持时明确回退到单精度。记忆库与检索索引仍按正式格式保存。",
    )

    st.markdown("##### 正常记忆库")
    c1, c2, c3 = st.columns(3)
    c1.number_input("coreset 采样比例", 0.001, 1.0, step=0.01, format="%.3f", key="domain_rc_coreset", help=_help("保留正常补丁的比例。", "0.1", "调大可能提升覆盖但增加内存、文件和检索时间。"))
    c2.number_input("最大记忆条目", 1, 200000, step=1000, key="domain_rc_max_memory", help=_help("限制正常记忆条目上限。", "50000", "调小节省资源但可能损失正常外观覆盖。"))
    c3.number_input("最近邻数量", 1, 64, key="domain_rc_knn", help=_help("异常评分使用的邻居数。", "1", "调大更平滑但增加检索成本。"))
    c1, c2, c3 = st.columns(3)
    c1.selectbox("ANN / FAISS 后端", ["auto", "faiss_gpu_flat", "faiss_gpu_ivf", "faiss_cpu_flat", "torch_gpu_flat", "sklearn"], key="domain_rc_memory_backend", help="auto 会选择可用后端；不兼容时可按下方设置安全回退。")
    c2.selectbox("安全回退后端", ["faiss_cpu_flat", "torch_gpu_flat", "sklearn"], key="domain_rc_fallback", help="指定 ANN 后端不可用时回退到兼容实现，不改变记忆库内容。")
    c3.selectbox("分数聚合", ["maximum_patch", "top_k_mean"], key="domain_rc_aggregation", help="决定补丁分数如何汇总为图像分数；Top-K 兼顾局部缺陷与抗噪。")
    if st.session_state.get("domain_rc_memory_backend") == "faiss_gpu_ivf":
        c1, c2 = st.columns(2)
        c1.number_input("IVF 分区数量", 1, 65536, key="domain_rc_nlist", help="分区越多索引更细，但构建和小数据开销增大。")
        c2.number_input("IVF 查询分区", 1, 4096, key="domain_rc_nprobe", help="查询分区越多召回更高，但检索更慢。")

    if mode == LIGHTWEIGHT_CREATE_MODE:
        st.markdown("##### 缺陷引导轻量校准")
        c1, c2, c3 = st.columns(3)
        c1.selectbox("图像阈值目标", ["best_f1", "recall_priority", "normal_percentile", "mean_std"], key="domain_rc_threshold_method", help="根据已确认正常/缺陷分数选择图像级阈值；不修改骨干权重。")
        c2.number_input("Recall 下限", 0.0, 1.0, step=0.01, key="domain_rc_recall", help="候选版本验证时的召回目标；调高可减少漏检但可能增加误报。")
        c3.number_input("误检率上限", 0.0, 1.0, step=0.01, key="domain_rc_fpr", help="候选版本允许的正常误报率上限。")
        c1, c2, c3 = st.columns(3)
        c1.number_input("正常分位阈值", 50.0, 100.0, step=0.1, key="domain_rc_percentile", help="仅正常分布策略使用；调高通常减少误报但可能增加漏检。")
        c2.selectbox("像素阈值", ["automatic_from_normal_map"], key="domain_rc_pixel_threshold", help="根据正常样本异常图自动估计像素级阈值。")
        c3.number_input("随机种子", 0, 2147483647, key="domain_rc_seed", help="保证 coreset、抽帧和校准可复验；不改变业务标签。")
        c1, c2, c3, c4 = st.columns(4)
        c1.number_input("门控分位数", 90.0, 99.9, step=0.1, key="domain_rc_gate", help="限制轻量校准只在可信区域生效，调高更保守。")
        c2.number_input("L2 正则", 0.0, 10.0, step=0.01, key="domain_rc_l2", help="调大可抑制小样本过拟合，但校准幅度更弱。")
        c3.number_input("分数尺度", 0.01, 100.0, step=0.1, key="domain_rc_scale", help="控制轻量分数映射尺度；过大可能放大噪声。")
        c4.number_input("交叉验证折数", 2, 10, key="domain_rc_cv", help="折数增大验证更充分，但校准耗时增加。")
    return mode_changed


def _job_panel() -> None:
    job_id = str(st.session_state.get("domain_adaptation_job_id") or "")
    if not job_id:
        st.markdown("### 第五步：结果与应用")
        _result_actions()
        return
    response = client().domain_adaptation_job(job_id)
    if not response.get("success"):
        error = dict(response.get("error") or {})
        if str(error.get("code") or "") == "DOMAIN_JOB_NOT_FOUND":
            st.session_state["domain_rc_job_notice"] = ("warning", "旧构建任务状态已失效，创建锁已经释放，可以重新创建。")
            st.session_state.pop("domain_adaptation_job_id", None)
            st.session_state.pop("domain_rc_build_idempotency_key", None)
            st.session_state.pop("domain_rc_build_fingerprint", None)
            st.query_params.pop("domain_job", None)
            st.warning("旧构建任务状态已失效，创建锁已经释放，可以重新创建。")
            return
        st.error(_error(response, "无法读取构建进度。")); return
    job = response.get("domain_job") or {}
    state = str(job.get("status") or "")
    progress = float(job.get("progress") or 0.0)
    st.progress(progress, text=f"{job.get('stage')} · {job.get('message')}")
    metrics = st.columns(5)
    metrics[0].metric("已处理正常样本", int(job.get("normal_processed") or 0))
    metrics[1].metric("已处理缺陷样本", int(job.get("defect_processed") or 0))
    metrics[2].metric("有效任务进度", f"{progress * 100:.1f}%")
    elapsed = float(job.get("elapsed_seconds") or 0.0)
    metrics[3].metric("已用时间", _duration(elapsed))
    eta = job.get("eta_seconds")
    metrics[4].metric("预计剩余", _duration(eta) if eta is not None else ("已完成" if state == "completed" else "正在估算"))

    status_ui = {
        "completed": ("✅", "已完成"),
        "running": ("⏳", "正在执行"),
        "pending": ("○", "等待执行"),
        "warning": ("⚠️", "已完成但有警告"),
        "skipped": ("—", "不适用"),
        "failed": ("❌", "失败"),
    }
    st.markdown("#### 构建阶段")
    for index, stage in enumerate(_public_build_stages(list(job.get("stages") or [])), start=1):
        stage_status = str(stage.get("status") or "pending")
        icon, label = status_ui.get(stage_status, ("○", "等待执行"))
        with st.container(border=True):
            columns = st.columns([0.55, 3.1, 1.25, 1.2], vertical_alignment="center")
            columns[0].markdown(f"### {icon}")
            columns[1].markdown(f"**{index}. {stage.get('stage_name') or stage.get('stage_id')}**")
            columns[1].caption(str(stage.get("message") or ""))
            columns[2].caption(label)
            stage_progress = float(stage.get("progress") or 0.0)
            if stage_status == "running":
                columns[2].progress(stage_progress)
            columns[3].caption(
                "— 不适用"
                if stage_status == "skipped"
                else (_duration(stage.get("elapsed_seconds")) if stage.get("started_at") else "尚未开始")
            )

    activity = str(job.get("activity_status") or "active")
    if activity == "long_running":
        st.info("当前阶段计算时间较长，任务心跳正常，仍在运行。")
    elif activity == "possibly_interrupted":
        st.error("任务心跳已停止，任务可能已中断；请先等待状态确认，不要重复创建任务。")
    if state in {"queued", "running", "cancelling"}:
        cancel_scope = f"domain-build-cancel::{job_id}"
        cancel_clicked, cancel_status = render_action_button_with_feedback(
            "安全停止构建",
            scope=cancel_scope,
            key=f"domain_cancel_{job_id}",
            disabled=state == "cancelling",
        )
        if cancel_clicked:
            update_action_feedback(cancel_scope, "info", "正在提交安全停止请求...", cancel_status)
            cancelled = client().cancel_domain_adaptation(job_id)
            if cancelled.get("success"):
                update_action_feedback(cancel_scope, "success", "已提交安全停止请求。", cancel_status)
            else:
                update_action_feedback(cancel_scope, "error", _error(cancelled, "安全停止请求失败。"), cancel_status)
    elif state == "interrupted":
        st.warning("后端重启时该构建任务尚未完成，已标记为中断；半成品不会注册为正式资源。")
        rerun_scope = f"domain-build-rerun::{job_id}"
        rerun_clicked, rerun_status = render_action_button_with_feedback(
            "重新运行目标域构建",
            scope=rerun_scope,
            key=f"domain_rerun_{job_id}",
            button_type="primary",
        )
        if rerun_clicked:
            update_action_feedback(rerun_scope, "info", "正在创建新的构建任务...", rerun_status)
            rerun = client().rerun_domain_adaptation(job_id)
            if rerun.get("success"):
                created = dict(rerun.get("domain_job") or {})
                st.session_state["domain_adaptation_job_id"] = str(created.get("job_id") or "")
                st.query_params["domain_job"] = str(created.get("job_id") or "")
                update_action_feedback(rerun_scope, "success", "已重新创建构建任务。", rerun_status)
            else:
                update_action_feedback(rerun_scope, "error", _error(rerun, "目标域构建重新运行失败。"), rerun_status)
    elif state == "completed":
        result = job.get("result") or {}
        if str(job.get("registration_status") or result.get("registration_status") or "") != "registered":
            st.error("目标域资源尚未完成统一注册，已阻止页面将其标记为成功。")
            return
        existing_result = dict(st.session_state.get("domain_rc_candidate") or {})
        st.session_state["domain_rc_candidate"] = {**existing_result, **result}
        st.session_state["domain_rc_completed_job_id"] = job_id
        st.session_state["domain_rc_completed_elapsed"] = float(job.get("elapsed_seconds") or 0.0)
        if result.get("memory_bank_id"):
            st.query_params["domain_candidate"] = str(result["memory_bank_id"])
        st.success("### ✅ 目标域记忆库创建成功")
        st.markdown("### 结果与应用")
        _result_actions()
    elif state == "failed":
        st.error(friendly_error_message(job.get("error"), "目标域构建失败。"))
        if st.button("关闭失败任务并返回修改", key=f"domain_close_failed_{job_id}"):
            st.session_state.pop("domain_adaptation_job_id", None)
            st.session_state.pop("domain_rc_build_idempotency_key", None)
            st.session_state.pop("domain_rc_build_fingerprint", None)
            st.query_params.pop("domain_job", None)
    elif state == "cancelled":
        st.warning("任务已安全取消，未发布半成品 profile。")
        if st.button("关闭已取消任务", key=f"domain_close_cancelled_{job_id}"):
            st.session_state.pop("domain_adaptation_job_id", None)
            st.session_state.pop("domain_rc_build_idempotency_key", None)
            st.session_state.pop("domain_rc_build_fingerprint", None)
            st.query_params.pop("domain_job", None)
    if state not in {"queued", "running", "cancelling"} and st.session_state.get("domain_rc_terminal_job_id") != job_id:
        st.session_state["domain_rc_terminal_job_id"] = job_id
        st.rerun(scope="app")


def _result_actions() -> None:
    result = dict(st.session_state.get("domain_rc_candidate") or {})
    candidate_id = str(st.query_params.get("domain_candidate") or "")
    if not result and candidate_id:
        response = client().domain_candidate(candidate_id)
        if response.get("success"):
            result = dict(response.get("candidate") or {})
            st.session_state["domain_rc_candidate"] = result
    if not result:
        st.info("完成构建后，这里将显示完整 profile、正常记忆库、配置快照、阈值和可选缺陷资源。")
        return
    profile_status = str(result.get("profile_status") or "frozen")
    is_frozen = bool(
        st.session_state.get("domain_rc_candidate_frozen")
        or result.get("frozen")
        or profile_status == "frozen"
    )
    profile_id = str(result.get("memory_bank_id") or "")
    is_applied = bool(st.session_state.get("domain_rc_candidate_applied"))
    # The progress panel is a timed fragment.  Applying a profile bumps the
    # global resource revision and can trigger another fragment run before the
    # transient success element is painted.  The backend runtime is the source
    # of truth, so recover the applied state from readiness instead of relying
    # solely on a one-run session flag.
    if profile_id and not is_applied:
        readiness = client().readiness()
        current_profile = dict(readiness.get("current_profile") or {}) if readiness.get("success") else {}
        if str(current_profile.get("profile_id") or current_profile.get("memory_bank_id") or "") == profile_id:
            is_applied = True
            st.session_state["domain_rc_candidate_applied"] = True
    summary = st.columns(6)
    summary[0].metric("记忆库", str(result.get("friendly_name") or result.get("memory_bank_name") or "目标域记忆库"))
    summary[1].metric("目标域版本", f"v{int(result.get('profile_version') or 1)}")
    summary[2].metric("配置", str(result.get("default_config_name") or "初始化配置"))
    summary[3].metric("样本数量", int(result.get("normal_sample_count") or 0) + int(result.get("defect_calibration_sample_count") or 0))
    summary[4].metric("创建耗时", _duration(result.get("elapsed_seconds") or st.session_state.get("domain_rc_completed_elapsed")))
    summary[5].metric("状态", "✅ 可使用" if is_frozen else "资源尚未就绪")
    if is_applied:
        st.success("✅ 已应用到当前检测环境。Memory Bank、默认 Config 和 Profile 已完成原子切换；检测仍为只读推理。")
    else:
        st.caption("后台已自动完成完整性确认、SHA-256、不可变冻结、资源注册和事务提交；当前运行方案尚未切换。")
    apply_scope = f"domain-candidate-apply::{profile_id}"
    apply_clicked, apply_status = render_action_button_with_feedback(
        "应用该记忆库",
        scope=apply_scope,
        key=f"domain_apply_{profile_id}",
        button_type="primary",
        icon=":material/check_circle:",
        disabled=not is_frozen or is_applied,
        column_ratio=(0.5, 1.5),
    )
    if apply_clicked:
        update_action_feedback(apply_scope, "info", "正在原子切换 Memory Bank、Config 和 Profile...", apply_status)
        response = client().apply_domain_candidate(profile_id, confirm=True)
        if response.get("success"):
            st.session_state["domain_rc_candidate_applied"] = True
            st.session_state["domain_rc_apply_result"] = dict(response.get("result") or {})
            result_payload = dict(response.get("result") or {})
            resource_revision = int(result_payload.get("resource_revision") or 0)
            if resource_revision:
                st.session_state["aoi_resource_catalog_revision"] = resource_revision
            set_action_feedback(apply_scope, "success", "已应用到当前检测环境。")
            st.rerun(scope="fragment")
        else:
            update_action_feedback(apply_scope, "error", _error(response, "目标域版本应用失败。"), apply_status)
    actions = st.columns(2)
    actions[0].page_link("pages/10_运行配置.py", label="前往运行配置", icon=":material/settings:", width="stretch")
    continue_create = actions[1].button("继续创建", icon=":material/add_circle:", width="stretch")
    with st.expander("查看构建详情与完整配置", expanded=False):
        _render_build_result(result)
    if continue_create:
        st.session_state.pop("domain_rc_candidate", None)
        st.session_state.pop("domain_rc_candidate_validated", None)
        st.session_state.pop("domain_rc_candidate_frozen", None)
        st.session_state.pop("domain_rc_candidate_applied", None)
        st.session_state.pop("domain_rc_apply_result", None)
        st.session_state.pop("domain_rc_completed_job_id", None)
        st.session_state.pop("domain_rc_completed_elapsed", None)
        st.session_state.pop("domain_adaptation_job_id", None)
        st.session_state.pop("domain_rc_build_idempotency_key", None)
        st.session_state.pop("domain_rc_build_fingerprint", None)
        st.query_params.pop("domain_candidate", None)
        st.query_params.pop("domain_job", None)


def render(runtime: dict[str, Any], sample_controls: Callable[[], None], source_payload: Callable[[], dict[str, Any]]) -> None:
    """Render independent local fragments without nesting fragment lifecycles."""
    # Remove legacy widget state left by sessions created before the official
    # base-model resolver became authoritative.
    st.session_state.pop("domain_rc_model", None)
    st.session_state.pop("domain_rc_backbone", None)
    st.session_state.pop("base_model_version", None)
    current_mode = str(st.session_state.get("domain_rc_mode") or "")
    if current_mode and current_mode not in MODE_LABELS and current_mode != MODE_MIGRATION_REQUIRED:
        st.session_state["domain_rc_legacy_mode"] = current_mode
        st.session_state["domain_rc_mode_migration_required"] = True
        st.session_state["domain_rc_mode"] = MODE_MIGRATION_REQUIRED
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)
    st.title("目标域资源创建")
    st.caption("选择构建样本 → 配置构建参数 → 检查条件 → 创建并应用。完整性确认与资源冻结由后台自动完成；测试数据始终禁止参与构建。")
    st.info("最终产物不是一个散落的记忆库文件，而是可追溯的完整 profile：正式基础特征资源 + 配置快照 + 正常记忆库 + 阈值 + 可选轻量校准器。")
    if not st.session_state.get("domain_adaptation_job_id"):
        query_job = str(st.query_params.get("domain_job") or "")
        if query_job:
            st.session_state["domain_adaptation_job_id"] = query_job
    if notice := st.session_state.pop("domain_rc_job_notice", None):
        level, message = notice
        getattr(st, level if level in {"success", "warning", "error", "info"} else "info")(message)
    # Clear the one-off legacy draft notice; current draft actions use the
    # scoped status column rendered by ``_draft_bar``.
    st.session_state.pop("domain_rc_draft_notice", None)
    _draft_bar(source_payload)
    _process_basic_info_autosave(source_payload)
    st.caption("以下内容按单页纵向流程排列；填写内容、构建条件和候选结果均使用同一份草稿状态，不再使用无业务作用的步骤选择控件。")

    st.markdown("### 第一步：样本与目标域")
    _render_basic_info_editor()
    _render_basic_info_autosave_error()
    sample_controls()
    scan_action_status = st.empty()
    st.caption("选择样本目录后会在来源区域自动扫描；耗时校验只刷新该区域，完成后再更新构建条件。")
    if notice := st.session_state.pop("domain_rc_scan_notice", None):
        update_action_feedback("domain-samples::scan", "success", str(notice), scan_action_status)
    scan_summary = dict(st.session_state.get("domain_scan_summary") or {})
    if scan_summary:
        normal_scan = dict(scan_summary.get("normal") or {})
        defect_scan = dict(scan_summary.get("defect") or {})
        metrics = st.columns(4)
        metrics[0].metric("正常有效样本", int(normal_scan.get("valid_count") or 0))
        metrics[1].metric("缺陷有效样本", int(defect_scan.get("valid_count") or 0))
        metrics[2].metric("不可读取样本", int(normal_scan.get("invalid_count") or 0) + int(defect_scan.get("invalid_count") or 0))
        metrics[3].metric("样本预算", "满足" if scan_summary.get("sample_budget_ready") else ("构建时确认" if scan_summary.get("budget_validation_pending") else "未满足"))
    st.markdown("### 第二步：构建模式与参数")
    with st.form("domain_build_parameters_form", border=True):
        current_name = runtime.get("memory_bank_name") or runtime.get("memory_bank_id") or "未应用目标域 profile"
        st.caption(f"当前冻结运行方案（只读参考，不在构建时切换）：{current_name}")
        mode_changed = _config_form()
        parameters_submitted = st.form_submit_button("应用构建参数", type="primary", icon=":material/tune:")
    if parameters_submitted and mode_changed:
        # Streamlit commits form widget values after controls above the form
        # have rendered.  One deliberate rerun lets the earlier sample fragment
        # immediately add/remove defect inputs for the newly chosen mode.
        st.rerun()

    payload = _payload(source_payload)
    st.markdown("### 第三步：构建条件检查")
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "payload": payload,
                "resource_revision": int(st.session_state.get("aoi_resource_catalog_revision") or 0),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    refresh_requested = st.button("重新检查全部构建条件", icon=":material/fact_check:")
    if refresh_requested or st.session_state.get("domain_rc_conditions_fingerprint") != fingerprint:
        if bool(st.session_state.get("domain_rc_mode_migration_required")):
            st.session_state["domain_rc_conditions"] = {
                "ready": False,
                "can_create": False,
                "blocking_count": 1,
                "warning_count": 0,
                "items": [{
                    "id": "mode_migration",
                    "status": "failed",
                    "icon": "❌",
                    "title": "构建模式迁移",
                    "summary": "当前草稿使用已停止新建的历史构建方式，系统没有自动改变其算法模式。",
                    "repair": "返回第二步，明确选择“基础正常记忆”或“单正常记忆 · 缺陷引导轻量校准”。",
                }],
            }
        else:
            response = client().domain_build_readiness(payload)
            if response.get("success"):
                st.session_state["domain_rc_conditions"] = response.get("build_readiness") or response.get("conditions") or {}
            else:
                st.session_state["domain_rc_conditions"] = {
                    "ready": False,
                    "items": [],
                    "error": _error(response, "构建条件检查失败。"),
                }
        st.session_state["domain_rc_conditions_fingerprint"] = fingerprint
    conditions = dict(st.session_state.get("domain_rc_conditions") or {})
    if conditions.get("error"):
        st.error(str(conditions["error"]))
    rows = list(conditions.get("items") or [])
    for index, item in enumerate(rows, start=1):
        with st.container(border=True):
            title = str(item.get("title") or item.get("message") or "构建条件")
            st.markdown(f"**{index}. {item.get('icon') or '—'} {title}**")
            if summary := str(item.get("summary") or item.get("message") or ""):
                if summary != title:
                    st.caption(summary)
            if item.get("repair") and item.get("status") in {"failed", "warning"}:
                st.caption(f"解决建议：{item.get('repair')}")
            details = dict(item.get("details") or {})
            if details:
                with st.expander("技术详情", expanded=False):
                    for label, value in details.items():
                        st.caption(f"{label}：{value}")
    blocking_count = int(conditions.get("blocking_count") or sum(str(item.get("status") or "") == "failed" for item in rows))
    warning_count = int(conditions.get("warning_count") or sum(str(item.get("status") or "") == "warning" for item in rows))
    can_create = bool(conditions.get("can_create", conditions.get("ready", False)))
    if blocking_count:
        st.error(f"当前还有 {blocking_count} 项必须处理，完成后即可创建记忆库。")
    elif rows:
        st.success("全部阻断条件已满足，可以创建完整目标域候选版本。")
    if warning_count:
        st.warning(f"当前有 {warning_count} 项建议性提示；黄色警告不会阻止创建候选版本。")

    st.markdown("### 第四步：开始构建")
    st.caption("构建使用正常样本生成正常记忆；仅在轻量校准模式下读取缺陷样本，且缺陷样本绝不会写入正常记忆库。")
    active_job_id = str(st.session_state.get("domain_adaptation_job_id") or "")
    if active_job_id:
        st.info("正在恢复构建任务状态；下方只刷新任务进度区域，并会在完成后直接显示结果与应用操作。")
    else:
        summary = dict(conditions.get("summary") or {})
        with st.container(border=True):
            st.markdown("#### 创建目标域记忆库")
            if can_create:
                summary_columns = st.columns(4)
                summary_columns[0].caption(f"目标域：{summary.get('friendly_name') or '-'}")
                summary_columns[1].caption(f"构建模式：{MODE_LABELS.get(str(summary.get('method_mode') or ''), '-')}")
                summary_columns[2].caption(f"正常 / 缺陷：{int(summary.get('normal_sample_count') or 0)} / {int(summary.get('defect_sample_count') or 0)}")
                summary_columns[3].caption(f"coreset / K：{summary.get('coreset_ratio', '-')} / {summary.get('knn_k', '-')}")
            else:
                st.caption(f"当前还有 {blocking_count} 项必须处理。按钮保留在固定位置，条件满足后会立即启用。")
        create_scope = "domain-build::start"
        create_requested, create_status = render_action_button_with_feedback(
            "创建目标域记忆库",
            scope=create_scope,
            key="domain_rc_create_memory_bank",
            button_type="primary",
            icon=":material/build:",
            disabled=not can_create,
            column_ratio=(0.62, 1.38),
        )
    if not active_job_id and create_requested:
        update_action_feedback(create_scope, "info", "正在创建后台构建任务...", create_status)
        if st.session_state.get("domain_rc_build_fingerprint") != fingerprint:
            st.session_state["domain_rc_build_fingerprint"] = fingerprint
            st.session_state["domain_rc_build_idempotency_key"] = f"domain-build:{uuid.uuid4().hex}"
        build_payload = {
            **payload,
            "draft_id": str(st.session_state.get("domain_rc_draft_id") or "") or None,
            "idempotency_key": st.session_state["domain_rc_build_idempotency_key"],
        }
        response = client().start_domain_adaptation(build_payload)
        if not response.get("success"):
            recovered = client().current_domain_adaptation_job()
            recovered_job = recovered.get("domain_job") or {}
            if recovered.get("success") and str(recovered_job.get("idempotency_key") or "") == str(build_payload["idempotency_key"]):
                response = {"success": True, "domain_job": recovered_job}
        if response.get("success"):
            job = response.get("domain_job") or {}
            st.session_state["domain_adaptation_job_id"] = job.get("job_id")
            update_action_feedback(
                create_scope,
                "success",
                "目标域构建任务已提交，正在准备构建环境。",
                create_status,
            )
            st.session_state.pop("domain_rc_terminal_job_id", None)
            st.query_params["domain_job"] = str(job.get("job_id") or "")
            st.query_params.pop("domain_candidate", None)
            st.session_state.pop("domain_rc_candidate", None)
        else:
            st.session_state.pop("domain_rc_build_idempotency_key", None)
            st.session_state.pop("domain_rc_build_fingerprint", None)
            update_action_feedback(create_scope, "error", _error(response, "无法启动目标域构建。"), create_status)
    active_job_id = str(st.session_state.get("domain_adaptation_job_id") or "")
    if active_job_id and str(st.session_state.get("domain_rc_terminal_job_id") or "") != active_job_id:
        st.fragment(run_every=1.2)(_job_panel)()
    else:
        st.fragment(_job_panel)()
