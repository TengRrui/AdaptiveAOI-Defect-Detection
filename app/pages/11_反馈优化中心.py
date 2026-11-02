from __future__ import annotations

# 操作员反馈审核与安全优化工作流。
# 页面只编排本地接口；资源更新必须经过审核、验证和显式应用。

import base64
import copy
import time
from io import BytesIO
from datetime import datetime, timedelta, timezone
from typing import Any

import streamlit as st

from backend.services.feedback_task_status import get_feedback_task_status_display
from performance_cache import api_client as client
from delete_preview_confirmation import render_delete_preview_confirmation
from friendly_display import friendly_error_message
from feedback_review_table import (
    REVIEW_TABLE_HEADERS,
    REVIEW_TABLE_WIDTHS,
    compact_sample_name,
    corrected_label_text,
)
from mask_editor import (
    binary_mask,
    canvas_drawing_to_source_mask,
    compose_editor_background,
    fit_canvas_size,
    mask_png_bytes,
)
from performance_cache import (
    cached_runtime_context,
    cached_runtime_liveness,
    cached_feedback_candidates,
    cached_feedback_resources,
    cached_feedback_statistics,
    clear_feedback_caches,
    clear_resource_caches,
)
from time_utils import format_beijing_datetime
from navigation import render_sidebar_navigation
from selection_state import (
    checkbox_changed as update_checkbox_selection,
    checkbox_key as stable_checkbox_key,
    clear_selection as clear_id_selection,
    remove_selected_ids,
    selected_ids as read_selected_ids,
    set_page_selection as update_page_selection,
    persistent_value,
    store_persistent_value,
    store_selected_ids,
    sync_checkbox_states,
)
from styles import render_sidebar_brand
from ui_common import (
    configure_page,
    render_action_button_with_feedback,
    render_action_feedback,
    set_action_feedback,
    update_action_feedback,
)


REVIEW_ACTION_FEEDBACK_SCOPE = "feedback-review::last-action"
FEEDBACK_MEMORY_BANK_KEY = "feedback_selected_memory_bank_id"
FEEDBACK_MEMORY_BANK_PERSISTENCE_KEY = "feedback_center:memory_bank"


configure_page("feedback_optimization")
render_sidebar_brand()
render_sidebar_navigation("feedback_optimization")
st.title("反馈优化中心")
st.caption("检测结果反馈 → 人工纠错审核 → 按来源记忆库更新 → 验证、应用或回滚")
with st.expander("安全说明", expanded=False):
    st.info("反馈不会立即修改当前运行方案。审核通过的样本需经过优化、验证和版本冻结后，才会应用于后续检测。正式测试期间仅记录反馈。")
nav_a, nav_b = st.columns(2)
with nav_a:
    st.page_link("pages/4_目标域适配.py", label="需要完整目标域重建或校准？前往目标域适配", icon=":material/hub:", width="stretch")
with nav_b:
    st.page_link("pages/10_运行配置.py", label="查看当前冻结方案", icon=":material/settings:", width="stretch")

api = client()


def _service_state(payload: dict[str, Any]) -> tuple[str, str]:
    """Translate health/transport states into safe page-level messages."""
    if not isinstance(payload, dict) or not payload.get("success"):
        error = payload.get("error") if isinstance(payload, dict) else {}
        code = str((error or {}).get("code") or "") if isinstance(error, dict) else ""
        if code == "API_CONNECTION_LOST":
            return "disconnected", "后台服务连接中断。"
        if code == "API_STATUS_TIMEOUT":
            return "starting", "状态刷新稍慢，正在重新连接。"
        if code in {"API_UNAVAILABLE", "CONNECT_TIMEOUT", "READ_TIMEOUT"}:
            return "starting", "反馈服务正在启动，请稍候。"
        return "error", "反馈服务暂时不可用，请重新启动系统或稍后重试。"
    status = str(payload.get("status") or "ready").lower()
    if status in {"starting", "pending", "loading"}:
        return "starting", "反馈服务正在启动，请稍候。"
    if status in {"error", "failed"}:
        return "error", "反馈服务启动失败，请查看启动日志后重试。"
    if status == "degraded":
        return "degraded", "反馈服务已连接，但部分资源仍在初始化。"
    return "ready", "反馈服务已就绪。"


def _feedback_wait_content() -> None:
    """Poll only liveness while the feedback API process is unavailable."""
    started = float(st.session_state.setdefault("feedback_center_wait_started", time.monotonic()))
    attempts = int(st.session_state.get("feedback_center_wait_attempts", 0))
    if time.monotonic() - started <= 40.0:
        attempts += 1
        st.session_state["feedback_center_wait_attempts"] = attempts
        state, message = _service_state(cached_runtime_liveness())
        if state in {"ready", "degraded"}:
            st.session_state.pop("feedback_center_wait_started", None)
            st.session_state.pop("feedback_center_wait_attempts", None)
            st.rerun()
            return
        if state == "error":
            st.error(message)
        else:
            st.info(message)
            st.caption(f"系统正在自动重试（第 {attempts} 次）；恢复后会自动加载。")
    else:
        st.warning("反馈服务尚未就绪，自动重试已暂停。")
        st.caption("确认算法服务已启动后，可点击“立即重试”。")
    if st.button("立即重试", icon=":material/refresh:", key="feedback_center_retry_service"):
        st.session_state["feedback_center_wait_started"] = time.monotonic()
        st.session_state["feedback_center_wait_attempts"] = 0


try:
    _feedback_wait_fragment = st.fragment(run_every=2.0)(_feedback_wait_content)
except AttributeError:  # pragma: no cover - compatibility with old Streamlit
    _feedback_wait_fragment = _feedback_wait_content


health_response = cached_runtime_liveness()
service_state, service_message = _service_state(health_response)
if service_state not in {"ready", "degraded"}:
    _feedback_wait_fragment()
    wait_tabs = st.tabs(["待审核反馈", "记忆库更新", "反馈记录管理"])
    for wait_tab in wait_tabs:
        with wait_tab:
            st.info(service_message)
    st.stop()

st.session_state.pop("feedback_center_wait_started", None)
st.session_state.pop("feedback_center_wait_attempts", None)


def _payload(response: dict[str, Any], key: str, default: Any) -> Any:
    if not response or not response.get("success", False):
        return default
    return response.get(key, default)


def _short_id(value: Any, fallback: str = "-") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    return text if len(text) <= 18 else f"{text[:7]}…{text[-5:]}"


def _duration_text(value: Any, *, estimate: bool = False) -> str:
    try:
        raw_seconds = max(0.0, float(value))
    except (TypeError, ValueError):
        return "正在估算" if estimate else "00:00:00"
    if 0.0 < raw_seconds < 1.0 and not estimate:
        return "< 1 秒"
    seconds = max(0, int(round(raw_seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    prefix = "约 " if estimate else ""
    return f"{prefix}{hours:02d}:{minutes:02d}:{seconds:02d}"


def _seconds_text(value: Any) -> str:
    try:
        seconds = max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return "—"
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} 毫秒"
    if seconds < 60.0:
        return f"{seconds:.2f} 秒"
    return _duration_text(seconds)


def _strategy_labels(options: list[str]) -> dict[str, str]:
    """Keep the visible sequence continuous without changing backend codes."""
    return {
        code: f"{chr(ord('A') + index)}. {STRATEGY_DISPLAY_NAMES.get(code, '更新目标域资源')}"
        for index, code in enumerate(options)
    }


def _render_strategy_eligibility(compatibility: dict[str, Any]) -> None:
    """Render backend-owned admission counts without duplicating policy numbers."""
    eligibility = dict(compatibility.get("eligibility") or {})
    if not eligibility:
        return
    counts = dict(eligibility.get("current_counts") or compatibility.get("current_counts") or {})
    hard = dict(eligibility.get("eligibility_requirements") or compatibility.get("eligibility_requirements") or {})
    recommended = dict(eligibility.get("recommended_requirements") or compatibility.get("recommended_requirements") or {})
    labels = (
        ("normal", "正常纠错样本"),
        ("defect", "缺陷纠错样本"),
        ("safe_normal", "安全正常样本"),
        ("reliable_mask_defect", "可靠 Mask 缺陷"),
    )
    st.markdown("#### 本策略样本要求")
    st.caption(
        "满足最低样本数量只表示可以开始计算，不代表候选版本一定能通过最终 Recall、误报率、延迟和记忆库约束验证。"
    )
    columns = st.columns(len(labels))
    for column, (field, label) in zip(columns, labels):
        actual = _count(counts.get(field))
        minimum = _count(hard.get(f"{field}_samples"))
        suggested = _count(recommended.get(f"{field}_samples"))
        if minimum <= 0:
            requirement = "不要求"
            state = "—"
        elif actual >= minimum:
            requirement = f"至少 {minimum} 条"
            state = "✅ 满足"
        else:
            requirement = f"至少 {minimum} 条"
            state = "❌ 不足"
        column.metric(label, f"{actual} / {requirement}")
        column.caption(state)
        if suggested > minimum and actual >= minimum and actual < suggested:
            column.caption(f"建议 {suggested} 条")
    if eligibility.get("can_start"):
        st.success("✅ 已满足本策略最低启动条件，可以开始计算。")
    else:
        st.error("❌ 当前样本不足以启动本策略。请按上方缺项补充审核样本。")
    for warning in eligibility.get("warnings") or []:
        st.warning("⚠️ " + str(warning.get("message") or "建议补充更多证据。"))
    if not eligibility.get("candidate_validation_ready"):
        st.info("候选版本验证尚未具备完整的两类证据；任务可以按策略启动，但不能据此直接冻结或应用。")


def _stage_label(stage_id: Any, fallback: Any = None) -> str:
    internal = str(stage_id or "").strip()
    if internal in FEEDBACK_STAGE_LABELS:
        return FEEDBACK_STAGE_LABELS[internal]
    text = str(fallback or internal or "更新阶段").strip()
    return FEEDBACK_STAGE_LABELS.get(text, text if any("\u4e00" <= char <= "\u9fff" for char in text) else "更新阶段")


def _timing_description(name: str, timing: dict[str, Any]) -> str:
    if name == "feedback_query_and_compatibility":
        return f"已读取并预检 {int(timing.get('candidate_count') or 0)} 条反馈"
    if name == "feedback_validation":
        return f"正常 {int(timing.get('normal_count') or 0)} 条，缺陷 {int(timing.get('defect_count') or 0)} 条"
    if name == "runtime_loading":
        return "复用已加载运行环境" if timing.get("cache_hit") else "已加载特征模型与记忆库"
    if name == "original_image_read":
        return f"读取 {int(timing.get('candidate_count') or 0)} 条原始样本"
    if name == "mask_read":
        return f"读取 {int(timing.get('mask_count') or 0)} 个 Mask"
    if name == "feedback_feature_extraction":
        return f"完成 {int(timing.get('extracted_count') or 0)} 条样本特征提取"
    if name == "profile_workspace":
        size = int(timing.get("bytes") or 0)
        return f"候选工作区约 {size / (1024 * 1024):.1f} MB" if size else "已创建隔离候选工作区"
    if name == "parameter_calibration":
        before, after = timing.get("threshold_before"), timing.get("threshold_after")
        if before is not None and after is not None:
            return f"判定阈值 {float(before):.4f} → {float(after):.4f}"
    if name == "profile_registration":
        return "已冻结候选版本" if timing.get("action") == "freeze" else "已注册并应用新版本"
    if name == "runtime_refresh":
        return "当前检测环境已切换到新版本"
    if timing.get("cache_hit"):
        return "复用缓存完成"
    return "已记录后台真实耗时"


def _parse_candidate_time(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _error(response: dict[str, Any], fallback: str) -> None:
    if response and not response.get("success", False):
        error = dict(response.get("error") or {})
        st.error(friendly_error_message(error, fallback))
        # Keep ordinary pages readable, while making a future contract
        # regression diagnosable without opening browser/network tools.
        if str(error.get("code") or "") == "REQUEST_VALIDATION_FAILED":
            fields = [
                item for item in (dict(error.get("details") or {}).get("fields") or [])
                if isinstance(item, dict) and item.get("field")
            ]
            if fields:
                with st.expander("技术详情 · 请求字段校验", expanded=False):
                    st.write("；".join(
                        f"{item.get('field')}：{item.get('message') or '参数无效'}"
                        for item in fields
                    ))


# The feedback page only needs the compact lock/readiness binding here.  The
# full /api/config payload (about 168 KB) belongs to the Runtime configuration
# page and made every feedback rerun wait on unnecessary serialization.
runtime_response = cached_runtime_context()
context_payload = _payload(runtime_response, "runtime_context", {})
runtime_state = dict(context_payload.get("runtime") or {}) if isinstance(context_payload, dict) else {}
inference_locked = bool(runtime_state.get("official_test_mode")) or bool(
    runtime_state.get("feedback_optimization_locked")
)


def _count(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


STATUS_LABELS = {
    "collected": "待审核", "legacy_unresolved": "待审核", "pending_review": "待审核",
    "in_review": "待审核", "conflict": "待审核", "record_only": "待审核",
    "accepted": "审核通过", "rejected": "已退回", "withdrawn": "已撤销",
    "queued": "已加入更新任务", "running": "更新中", "optimizing": "更新中", "cancelling": "正在取消",
    "ready_to_apply": "已生成，待应用", "validation_failed": "验证未通过", "cancelled": "已取消",
    "frozen": "已用于更新", "applied": "已用于更新", "rolled_back": "已用于更新",
    "consumed": "已用于更新", "archived": "已撤销", "deleted": "已删除",
    "invalid_source": "来源失效", "completed": "已用于更新", "failed": "处理失败",
    "interrupted": "已中断",
}
LABEL_NAMES = {"normal": "正常", "defect": "缺陷", "keep": "保持模型结果", "unknown": "待确认"}
MODE_NAMES = {"single": "单图", "batch": "批量", "video": "视频"}
METHOD_MODE_LABELS = {
    "normal_memory": "基础正常记忆检测",
    "normal_only": "基础正常记忆检测",
    "normal_defect_calibration": "正常记忆与缺陷判定校准",
    "lightweight_calibration": "缺陷引导轻量校准",
    "normal_memory_defect_guided_lite": "缺陷引导轻量校准",
    "dual_memory": "正常与缺陷双记忆",
    "normal_defect_dual_memory": "正常与缺陷双记忆",
    "auto": "自动推荐",
}

STRATEGY_DISPLAY_NAMES = {
    "threshold_only": "仅调整判定参数",
    "normal_memory": "更新正常记忆",
    "defect_prototypes": "更新缺陷原型",
    "lightweight_calibrator": "更新轻量校准器",
    "comprehensive": "综合更新",
}

STRATEGY_HELP = {
    "threshold_only": {
        "summary": "根据人工审核后的正常/缺陷样本重新估计当前目标域的判定边界，不修改正常特征记忆库。",
        "not_update": "不更新正常特征记忆、轻量校准器或骨干模型。",
        "samples": "需要两类已审核证据；具体最低数量见下方策略卡。",
        "mask": "不要求 Mask。",
        "resources": "正常 Memory Bank 保持不变，仅生成新的判定参数与配置快照。",
        "time": "相对最快。",
    },
    "normal_memory": {
        "summary": "使用人工审核确认正常的样本补充或重新优化正常特征空间，并按需要同步更新配套判定参数。",
        "not_update": "人工审核为缺陷的样本绝对不会写入正常 Memory Bank，也不训练骨干模型。",
        "samples": "核心是安全的审核正常样本；缺陷样本不是建库启动的硬条件，但缺少它会影响候选验证。",
        "mask": "通常不要求 Mask；带缺陷 Mask 的样本不能作为正常记忆候选。",
        "resources": "生成新的正常 Memory Bank、检索索引和配套配置。",
        "time": "需要特征提取和索引更新，耗时高于 A。",
    },
    "defect_prototypes": {
        "summary": "保持正常 Memory Bank 不变，使用审核确认的缺陷区域更新当前目标域的缺陷参考特征。",
        "not_update": "不把缺陷整图或正常背景写入正常 Memory Bank，也不训练骨干模型。",
        "samples": "核心是已审核缺陷样本和可靠缺陷区域；正常样本属于后续 FPR 验证证据。",
        "mask": "至少需要可靠 Mask 或当前算法认可的高置信缺陷区域。",
        "resources": "生成新的缺陷参考资源、判定参数和配置快照。",
        "time": "需要缺陷特征提取，耗时通常高于 A。",
    },
    "lightweight_calibrator": {
        "summary": "保持基础 Memory Bank 不变，根据审核样本更新当前目标域的轻量校准参数，使评分映射适应现有数据分布。",
        "not_update": "不重新训练骨干模型，不写入正常记忆库。",
        "samples": "两类监督样本各达到轻量校准器的最低拟合数量。",
        "mask": "当前轻量校准不强制要求 Mask。",
        "resources": "生成新的轻量校准资源、判定参数和配置快照。",
        "time": "一般介于 A 与完整记忆更新之间。",
    },
    "comprehensive": {
        "summary": "综合使用审核确认的正常/缺陷样本，联合更新当前方法真实支持的目标域资源，并生成新的完整目标域版本。",
        "not_update": "不支持的互斥资源不会被创建，骨干模型保持不变。",
        "samples": "按当前基础方法动态合并各子组件的样本要求；正常记忆仍只接受安全的审核正常样本。",
        "mask": "仅当当前方法需要更新缺陷参考资源时要求可靠 Mask。",
        "resources": "更新内容按当前方法动态决定，始终生成新的 Memory Bank/Profile/Config 版本。",
        "time": "执行阶段最多，通常耗时最高。",
    },
}

FEEDBACK_STAGE_LABELS = {
    "feedback_query_and_compatibility": "查询反馈与兼容性预检",
    "job_preflight": "创建任务前安全检查",
    "feedback_validation": "校验反馈样本",
    "profile_compatibility": "检查来源版本兼容性",
    "prepare_normal": "准备正常纠错样本",
    "prepare_defect": "准备缺陷纠错样本",
    "runtime_loading": "加载特征提取运行环境",
    "feature_extraction": "提取纠错样本特征",
    "original_image_read": "读取原始纠错样本",
    "mask_read": "读取缺陷标注",
    "feedback_feature_extraction": "提取纠错样本特征",
    "profile_workspace": "创建候选版本工作区",
    "normal_memory_update": "更新正常特征记忆",
    "defect_reference_update": "更新缺陷参考特征",
    "parameter_calibration": "重新计算判定参数与轻量校准",
    "ann_index": "重建检索索引",
    "base_memory_sha256_before": "校验基础记忆库（更新前）",
    "base_memory_sha256_after": "校验基础记忆库（更新后）",
    "offline_validation": "验证新版本",
    "validation": "验证新版本",
    "config_profile": "生成配置与目标域版本",
    "integrity": "资源完整性校验",
    "profile_registration": "冻结或注册新版本",
    "runtime_refresh": "刷新当前检测环境",
    "save_result": "保存更新结果",
    "complete": "完成",
}

STAGE_STATUS_LABELS = {
    "pending": "○ 等待执行",
    "running": "⏳ 正在执行",
    "completed": "✅ 完成",
    "warning": "⚠️ 完成但有提示",
    "skipped": "— 不适用",
    "failed": "❌ 失败",
    "cancelled": "— 已取消",
    "interrupted": "⚠️ 已中断",
}

UPDATE_COMPONENT_LABELS = {
    "threshold": "判定参数",
    "normal_memory": "正常特征记忆",
    "defect_memory": "缺陷参考特征",
    "lightweight_calibrator": "轻量校准器",
}
PREDICTION_NAMES = {"normal": "正常", "anomaly": "缺陷", "defect": "缺陷", "unknown": "未确认"}
ASSESSMENT_NAMES = {
    "correct": "结果正确", "incorrect": "结果有误", "uncertain": "不确定",
    "false_alarm": "误检", "missed_detection": "漏检",
}
ISSUE_NAMES = {
    "false_alarm": "误检", "missed_detection": "漏检", "score_high": "异常分数偏高",
    "score_low": "异常分数偏低", "region_inaccurate": "缺陷区域不准确", "other": "其他",
}
SELECTED_FEEDBACK_IDS_KEY = "selected_feedback_ids"
FEEDBACK_HISTORY_SELECTION_KEY = "feedback_history_selected_ids"


def _feedback_task_presentation(item: dict[str, Any]) -> dict[str, Any]:
    """Use the backend's canonical task presentation, with one shared fallback."""
    fallback = get_feedback_task_status_display(item.get("status"))
    return {
        key: item.get(key, fallback.get(key))
        for key in fallback
    }


def _result_value(value: Any, unit: str = "") -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "通过" if value else "未通过"
    try:
        number = float(value)
        if unit == "比例" or (abs(number) <= 1.0 and unit == ""):
            return f"{number:.3f}"
        return f"{number:.3f}"
    except (TypeError, ValueError):
        return str(value)


def _metric_display(value: Any) -> str:
    """Display persisted validation metrics without hiding a legal zero."""
    if value is None or value == "":
        return "—"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _render_update_result(detail: dict[str, Any]) -> dict[str, Any]:
    """Render the persisted result contract, never infer an outcome in UI."""
    result = dict(detail.get("result") or {})
    outcome_code = str(result.get("outcome_code") or "")
    if not result or outcome_code in {"", "IN_PROGRESS"}:
        return result

    if outcome_code in {"SUCCESS_READY_TO_APPLY", "SUCCESS_APPLIED"}:
        st.success(str(result.get("title") or "✅ Memory Bank 更新成功"))
    elif outcome_code == "CANCELLED":
        st.info(str(result.get("title") or "— Memory Bank 更新已取消"))
    else:
        st.error(str(result.get("title") or "❌ Memory Bank 更新未完成"))
    st.write(str(result.get("summary") or "暂无结果说明。"))

    header_cols = st.columns(4)
    strategy = dict(result.get("strategy") or {})
    header_cols[0].metric("更新策略", str(strategy.get("name") or strategy.get("code") or "—"))
    used = dict(result.get("used_samples") or {})
    header_cols[1].metric("实际使用样本", int(used.get("completed") or used.get("total") or 0))
    header_cols[2].metric("任务耗时", _duration_text(result.get("elapsed_seconds")))
    failed_stage = str(result.get("failed_stage") or "")
    header_cols[3].metric("失败阶段", _stage_label(failed_stage) if failed_stage else "—")

    source = dict(result.get("source_memory_bank") or {})
    candidate = dict(result.get("candidate_memory_bank") or {})
    if source or candidate:
        st.caption(
            f"基础记忆库：{_short_id(source.get('memory_bank_id'))} · "
            f"候选记忆库：{_short_id(candidate.get('memory_bank_id'))}"
        )

    failed_rules = [dict(item) for item in result.get("failed_rules") or [] if isinstance(item, dict)]
    rules = [dict(item) for item in result.get("validation_rules") or [] if isinstance(item, dict)]
    if rules:
        st.markdown("##### 验证证据与门槛")
        st.dataframe(
            [
                {
                    "验证项目": str(rule.get("label") or rule.get("key") or "—"),
                    "实际值": _result_value(rule.get("actual"), str(rule.get("unit") or "")),
                    "要求": f"{rule.get('operator') or ''} {_result_value(rule.get('required'), str(rule.get('unit') or ''))}".strip(),
                    "结果": "✅" if bool(rule.get("passed")) else "❌",
                }
                for rule in rules
            ],
            width="stretch",
            hide_index=True,
        )
    if failed_rules:
        st.markdown(f"##### 未通过或失败原因（{len(failed_rules)} 项）")
        st.dataframe(
            [
                {
                    "原因": str(rule.get("message") or "未满足规则"),
                    "实际值": _result_value(rule.get("actual")),
                    "要求": _result_value(rule.get("required")),
                }
                for rule in failed_rules
            ],
            width="stretch",
            hide_index=True,
        )
        if outcome_code == "ELIGIBILITY_FAILED":
            suggestions: list[str] = []
            for rule in failed_rules:
                try:
                    missing = max(0, int(rule.get("required") or 0) - int(rule.get("actual") or 0))
                except (TypeError, ValueError):
                    missing = 0
                if missing:
                    suggestions.append(f"{str(rule.get('field') or '对应类别')}还需 {missing} 条")
            if suggestions:
                st.info("建议：请继续收集并审核" + "、".join(suggestions) + "。")

    warnings = [dict(item) for item in result.get("warnings") or [] if isinstance(item, dict)]
    for warning in warnings:
        st.warning("⚠️ " + str(warning.get("message") or "存在注意事项。"))

    generated = dict(result.get("generated_resources") or {})
    if generated:
        with st.expander("生成资源与追溯信息", expanded=False):
            st.write(f"候选 Profile：{generated.get('candidate_profile_id') or '—'}")
            st.write(f"候选 Config：{generated.get('candidate_config_profile_id') or '—'}")
            if generated.get("candidate_config_profile_version") is not None:
                st.write(f"Config 版本：{generated.get('candidate_config_profile_version')}")
    return result


def _feedback_selection_persistence_key(*, history: bool = False) -> str:
    memory_bank_id = str(st.session_state.get(FEEDBACK_MEMORY_BANK_KEY) or "unselected")
    namespace = "history" if history else "review_update"
    return f"feedback_center:{namespace}:{memory_bank_id}"


def _selected_feedback_ids(valid_ids: set[str] | None = None) -> set[str]:
    """Return the one canonical selection set used by review/update tables."""
    return read_selected_ids(
        st.session_state,
        SELECTED_FEEDBACK_IDS_KEY,
        valid_ids=valid_ids,
        legacy_keys=("feedback_review_selected_ids", "feedback_reviewed_selected_ids"),
        persistence_key=_feedback_selection_persistence_key(),
    )


def _feedback_checkbox_key(scope: str, candidate_id: str) -> str:
    return stable_checkbox_key(f"feedback_{scope}", candidate_id)


def _store_feedback_selection(selected: set[str], *, scope: str, visible_ids: set[str]) -> None:
    persistence_key = _feedback_selection_persistence_key()
    store_selected_ids(
        st.session_state,
        SELECTED_FEEDBACK_IDS_KEY,
        selected,
        persistence_key=persistence_key,
    )
    sync_checkbox_states(
        st.session_state,
        state_key=SELECTED_FEEDBACK_IDS_KEY,
        scope=f"feedback_{scope}",
        item_ids=visible_ids,
        persistence_key=persistence_key,
    )


def _feedback_checkbox_changed(scope: str, candidate_id: str) -> None:
    update_checkbox_selection(
        st.session_state,
        state_key=SELECTED_FEEDBACK_IDS_KEY,
        scope=f"feedback_{scope}",
        item_id=candidate_id,
        persistence_key=_feedback_selection_persistence_key(),
    )


def _set_feedback_page_selection(scope: str, candidate_ids: tuple[str, ...], should_select: bool) -> None:
    update_page_selection(
        st.session_state,
        state_key=SELECTED_FEEDBACK_IDS_KEY,
        scope=f"feedback_{scope}",
        page_ids=candidate_ids,
        should_select=should_select,
        persistence_key=_feedback_selection_persistence_key(),
    )


def _clear_feedback_selection(scope: str, visible_ids: tuple[str, ...]) -> None:
    clear_id_selection(
        st.session_state,
        state_key=SELECTED_FEEDBACK_IDS_KEY,
        scopes=(f"feedback_{scope}",),
        visible_ids=visible_ids,
        persistence_key=_feedback_selection_persistence_key(),
    )


def _discard_feedback_selection(candidate_id: str) -> None:
    """Remove one item from the canonical selection and every rendered checkbox."""
    remove_selected_ids(
        st.session_state,
        state_key=SELECTED_FEEDBACK_IDS_KEY,
        removed_ids=(candidate_id,),
        scopes=("feedback_review", "feedback_update"),
        persistence_key=_feedback_selection_persistence_key(),
    )


def _clear_feedback_selection_state() -> None:
    store_selected_ids(
        st.session_state,
        SELECTED_FEEDBACK_IDS_KEY,
        set(),
        persistence_key=_feedback_selection_persistence_key(),
    )
    for key in list(st.session_state):
        if str(key).startswith(("selection::feedback_review::", "selection::feedback_update::")):
            st.session_state.pop(key, None)


def _history_selected_feedback_ids(valid_ids: set[str] | None = None) -> set[str]:
    return read_selected_ids(
        st.session_state,
        FEEDBACK_HISTORY_SELECTION_KEY,
        valid_ids=valid_ids,
        persistence_key=_feedback_selection_persistence_key(history=True),
    )


def _set_feedback_history_page_selection(candidate_ids: tuple[str, ...], should_select: bool) -> None:
    update_page_selection(
        st.session_state,
        state_key=FEEDBACK_HISTORY_SELECTION_KEY,
        scope="feedback_history",
        page_ids=candidate_ids,
        should_select=should_select,
        persistence_key=_feedback_selection_persistence_key(history=True),
    )
    st.session_state["feedback_history_table_revision"] = int(
        st.session_state.get("feedback_history_table_revision", 0)
    ) + 1


def _clear_feedback_history_selection(candidate_ids: tuple[str, ...]) -> None:
    clear_id_selection(
        st.session_state,
        state_key=FEEDBACK_HISTORY_SELECTION_KEY,
        scopes=("feedback_history",),
        visible_ids=candidate_ids,
        persistence_key=_feedback_selection_persistence_key(history=True),
    )
    st.session_state["feedback_history_table_revision"] = int(
        st.session_state.get("feedback_history_table_revision", 0)
    ) + 1


def _user_feedback_text(item: dict[str, Any]) -> str:
    assessment = ASSESSMENT_NAMES.get(str(item.get("assessment") or ""), "")
    issues = [ISSUE_NAMES.get(str(value), str(value)) for value in list(item.get("issue_tags") or [])]
    raw_label = str(item.get("corrected_label") or "keep")
    if str(item.get("mode") or "") == "video" and raw_label not in {"normal", "defect"}:
        correction = "历史记录标签缺失"
    else:
        correction = LABEL_NAMES.get(raw_label, "待确认")
    parts = [value for value in (assessment, "、".join(issues), f"修正为{correction}") if value]
    return "；".join(parts) or "未填写"


def _candidate_optimization_locked(item: dict[str, Any]) -> bool:
    context = dict(item.get("context") or {})
    return bool(
        context.get("official_test_mode")
        or context.get("feedback_optimization_locked")
        or str(context.get("source_scope") or "") == "official_test"
    )


def _candidate_label(item: dict[str, Any]) -> str:
    mode = {"single": "单图", "batch": "批量", "video": "视频"}.get(str(item.get("mode") or ""), "检测")
    raw_label = str(item.get("corrected_label") or "unknown")
    label = LABEL_NAMES.get(raw_label, "待确认")
    score = item.get("original_score")
    score_text = f"{float(score):.4f}" if isinstance(score, (int, float)) else "-"
    frame = item.get("frame_index")
    frame_text = f" · 帧 {frame}" if str(item.get("mode") or "") == "video" and frame not in (None, "", -1, "-1") else ""
    return f"{mode}{frame_text} · {_short_id(item.get('sample_id'), '样本')} · 原分数 {score_text} · {label}"


FEEDBACK_CENTER_TABS = (
    ("review", "待审核反馈"),
    ("optimization", "记忆库更新"),
    ("statistics", "反馈记录管理"),
)


def _feedback_active_tab() -> str:
    """Keep the selected work area stable across Streamlit reruns."""
    valid_keys = {key for key, _ in FEEDBACK_CENTER_TABS}
    legacy_targets = {"reviewed": "optimization", "annotation": "review", "validation": "optimization"}
    requested = str(st.query_params.get("feedback_center_tab") or "").strip()
    requested = legacy_targets.get(requested, requested)
    # Buttons rendered after the segmented control cannot mutate that widget's
    # state in the same Streamlit run.  Consume a pending target before the
    # control is instantiated on the next run instead.
    pending = str(st.session_state.pop("feedback_center_nav_target", "") or "")
    pending = legacy_targets.get(pending, pending)
    if pending in valid_keys:
        st.session_state["feedback_center_active_tab"] = pending
        st.query_params["feedback_center_tab"] = pending
    if requested in valid_keys and "feedback_center_active_tab" not in st.session_state:
        st.session_state["feedback_center_active_tab"] = requested
    st.session_state.setdefault("feedback_center_active_tab", "review")
    selected = st.segmented_control(
        "反馈中心分区",
        options=[key for key, _ in FEEDBACK_CENTER_TABS],
        format_func=lambda value: dict(FEEDBACK_CENTER_TABS).get(value, value),
        selection_mode="single",
        key="feedback_center_active_tab",
    )
    active = str(selected or st.session_state.get("feedback_center_active_tab") or "review")
    if active in valid_keys and str(st.query_params.get("feedback_center_tab") or "") != active:
        st.query_params["feedback_center_tab"] = active
    return active if active in valid_keys else "review"


def _refresh() -> None:
    st.session_state["feedback_center_refresh"] = int(st.session_state.get("feedback_center_refresh", 0)) + 1
    clear_feedback_caches()


def _load_feedback_revision() -> int:
    previous = st.session_state.get("feedback_revision")
    now = time.monotonic()
    last_check = float(st.session_state.get("feedback_revision_checked_at") or 0.0)
    if previous is not None and now - last_check < 0.75:
        return int(previous)
    response = api.feedback_revision(int(previous)) if previous is not None else api.feedback_revision()
    st.session_state["feedback_revision_checked_at"] = now
    if response.get("success"):
        revision = int(response.get("feedback_revision") or 0)
        st.session_state["feedback_revision"] = revision
        if previous is not None and int(previous) != revision:
            clear_feedback_caches()
        return revision
    return int(previous or 0)


def _mark_feedback_changed(*, resources_changed: bool = False) -> None:
    clear_feedback_caches()
    if resources_changed:
        clear_resource_caches()
        st.session_state["aoi_resource_catalog_checked_at"] = 0.0
    response = api.feedback_revision()
    if response.get("success"):
        st.session_state["feedback_revision"] = int(response.get("feedback_revision") or 0)
        st.session_state["feedback_revision_checked_at"] = time.monotonic()


def _display_time(value: Any) -> str:
    return format_beijing_datetime(value, fallback="未记录")


def _display_detection_time(item: dict[str, Any]) -> str:
    rendered = _display_time(item.get("detection_time"))
    if rendered != "未记录" and str(item.get("detection_time_source") or "") == "task_id":
        return f"{rendered}（历史任务时间）"
    return rendered


def _matches_time_range(value: Any, label: str) -> bool:
    if label == "全部":
        return True
    parsed = _parse_candidate_time(value)
    if parsed is None:
        return False
    days = {"今天": 1, "最近 7 天": 7, "最近 30 天": 30, "最近 90 天": 90}[label]
    return parsed >= datetime.now(timezone.utc) - timedelta(days=days)


@st.cache_data(show_spinner=False, max_entries=24)
def _cached_feedback_artifact_image(
    candidate_id: str,
    artifact_ref: str,
    artifact_revision: int = 0,
) -> Any:
    """Decode one candidate-scoped artifact once per component lifecycle."""
    del artifact_revision  # part of the cache key; the URL may stay stable for generated frames
    response = client().artifact_bytes(artifact_ref)
    if not response.get("success"):
        raise ValueError(friendly_error_message(response.get("error"), "样本图像读取失败。"))
    from PIL import Image
    import numpy as np

    with Image.open(BytesIO(response.get("content") or b"")) as image:
        image.load()
        return np.asarray(image.convert("RGB"))


def _editor_source(candidate_id: str, source_info: dict[str, Any]) -> tuple[str, str]:
    original = str(source_info.get("original_image_url") or "")
    if original:
        return original, "video_frame" if str(source_info.get("mode") or "") == "video" else "image"
    preview_response = api.feedback_candidate_source_preview(candidate_id)
    preview = dict(_payload(preview_response, "source", {}))
    kind = str(preview.get("kind") or "unavailable")
    return (str(preview.get("url") or ""), kind) if kind in {"image", "video_frame"} else ("", kind)


def _render_inline_mask_editor(candidate_id: str, item: dict[str, Any]) -> None:
    """Render optional Mask review beside the label decision, not on a separate page."""
    source_response = api.feedback_candidate_source(candidate_id)
    source_info = _payload(source_response, "source", {})
    source_width = int(source_info.get("source_width") or 0)
    source_height = int(source_info.get("source_height") or 0)
    original_mask_available = bool(source_info.get("mask_url"))
    mode_options = ["none"] + (["original"] if original_mask_available else []) + ["upload", "editor"]
    mask_mode = st.radio(
        "Mask（可选）",
        mode_options,
        format_func=lambda value: {
            "none": "不上传 Mask", "original": "使用系统原始 Mask",
            "upload": "上传二值 Mask", "editor": "打开内嵌 Mask 编辑器",
        }[value],
        horizontal=True,
        key=f"feedback_mask_mode_{candidate_id}",
    )
    mask_base64 = None
    mask_width = mask_height = None
    if mask_mode == "upload":
        uploaded = st.file_uploader("上传 PNG 二值 Mask", type=["png"], key=f"feedback_inline_upload_{candidate_id}")
        if uploaded is not None:
            raw = uploaded.getvalue()
            try:
                from PIL import Image

                with Image.open(BytesIO(raw)) as image:
                    image.load()
                    mask_width, mask_height = image.size
                    st.image(image, caption=f"Mask 预览 · {mask_width}×{mask_height}", width="stretch")
                mask_base64 = base64.b64encode(raw).decode("ascii")
            except Exception:
                st.error("Mask 不是可读取的 PNG 图像。")
    elif mask_mode == "editor":
        try:
            from streamlit_drawable_canvas import st_canvas
        except ImportError:
            st_canvas = None
        if st_canvas is None:
            st.warning("当前环境缺少内嵌画布组件，请改用 PNG 上传。")
        else:
            source_ref, source_kind = _editor_source(candidate_id, dict(source_info))
            if not source_ref:
                st.error("该反馈对应的原始检测输入当前不可用，已阻止在错误底图上编辑 Mask。")
            else:
                try:
                    from PIL import Image
                    import numpy as np

                    source_pixels = _cached_feedback_artifact_image(candidate_id, source_ref)
                    source_image = Image.fromarray(np.asarray(source_pixels).astype("uint8"), mode="RGB")
                except Exception as exc:
                    st.error(friendly_error_message(exc, "原始样本读取失败。"))
                    source_image = None

                if source_image is not None:
                    source_width, source_height = source_image.size
                    source_label = "对应视频帧" if source_kind == "video_frame" else "原始检测图片"
                    st.caption(
                        f"底图：当前反馈 candidate 对应的{source_label} · 原始尺寸 {source_width}×{source_height}。"
                    )
                    initial_refs: dict[str, tuple[str, int]] = {}
                    if source_info.get("review_mask_url"):
                        initial_refs["reviewed"] = (
                            str(source_info.get("review_mask_url")),
                            int(source_info.get("review_mask_revision") or 0),
                        )
                    if source_info.get("mask_url"):
                        initial_refs["predicted"] = (str(source_info.get("mask_url")), 0)
                    initial_options = ["none", *initial_refs]
                    initial_key = f"feedback_editor_initial_mask_{candidate_id}"
                    if st.session_state.get(initial_key) not in initial_options:
                        st.session_state.pop(initial_key, None)
                    default_initial = "reviewed" if "reviewed" in initial_refs else "predicted" if "predicted" in initial_refs else "none"
                    initial_choice = st.selectbox(
                        "初始 Mask",
                        initial_options,
                        index=initial_options.index(default_initial),
                        format_func=lambda value: {
                            "none": "不使用初始 Mask",
                            "reviewed": "使用最近保存的人工 Mask",
                            "predicted": "使用当前预测 Mask",
                        }[value],
                        key=initial_key,
                    )
                    initial_mask = None
                    if initial_choice in initial_refs:
                        mask_ref, mask_revision = initial_refs[initial_choice]
                        try:
                            initial_pixels = _cached_feedback_artifact_image(candidate_id, mask_ref, mask_revision)
                            initial_mask = binary_mask(initial_pixels, source_image.size)
                        except Exception as exc:
                            st.warning(friendly_error_message(exc, "初始 Mask 当前不可用，将从空白 Mask 开始。"))

                    opacity_columns = st.columns(2)
                    source_opacity = opacity_columns[0].slider(
                        "原图透明度", 20, 100, 100, 5,
                        key=f"feedback_editor_source_opacity_{candidate_id}",
                        help="只改变编辑器显示，不改变保存的原图或 Mask。",
                    )
                    overlay_opacity = opacity_columns[1].slider(
                        "Mask 覆盖透明度", 10, 90, 45, 5,
                        key=f"feedback_editor_mask_opacity_{candidate_id}",
                        help="已有 Mask 仅在选中区域使用半透明红色覆盖。",
                    )
                    editor_background = compose_editor_background(
                        source_image,
                        source_opacity=source_opacity / 100.0,
                        mask=initial_mask,
                        mask_opacity=overlay_opacity / 100.0,
                    )
                    canvas_width, canvas_height = fit_canvas_size(source_width, source_height)
                    tool_columns = st.columns([1.4, 1.0, 2.6])
                    tool = tool_columns[0].selectbox(
                        "编辑工具", ["brush", "eraser", "polygon"],
                        format_func=lambda value: {"brush": "画笔", "eraser": "橡皮擦", "polygon": "多边形"}[value],
                        key=f"feedback_inline_tool_{candidate_id}",
                    )
                    stroke_width = tool_columns[1].slider(
                        "笔刷宽度", 2, 40, 12 if tool == "eraser" else 6,
                        key=f"feedback_editor_stroke_{candidate_id}",
                    )
                    drawing_key = f"feedback_canvas_drawing_{candidate_id}"
                    epoch_key = f"feedback_canvas_epoch_{candidate_id}"

                    def _clear_drawing() -> None:
                        st.session_state[drawing_key] = {"version": "4.4.0", "objects": []}
                        st.session_state[epoch_key] = int(st.session_state.get(epoch_key, 0)) + 1

                    def _clear_all_mask() -> None:
                        _clear_drawing()
                        st.session_state[initial_key] = "none"

                    tool_columns[2].caption("画布工具栏提供撤销、重做与删除；透明度调整不会丢失当前绘制。")
                    clear_columns = st.columns(2)
                    clear_columns[0].button(
                        "清空当前绘制", key=f"feedback_inline_clear_drawing_{candidate_id}",
                        on_click=_clear_drawing, width="stretch",
                    )
                    clear_columns[1].button(
                        "清空全部 Mask", key=f"feedback_inline_clear_all_{candidate_id}",
                        on_click=_clear_all_mask, width="stretch",
                    )
                    stored_drawing = copy.deepcopy(
                        st.session_state.get(drawing_key) or {"version": "4.4.0", "objects": []}
                    )
                    canvas = st_canvas(
                        fill_color="rgba(16, 168, 216, 0.42)",
                        stroke_width=stroke_width,
                        stroke_color="#ffffff" if tool == "eraser" else "#10a8d8",
                        background_image=editor_background,
                        background_color="",
                        height=canvas_height,
                        width=canvas_width,
                        drawing_mode="polygon" if tool == "polygon" else "freedraw",
                        initial_drawing=stored_drawing,
                        display_toolbar=True,
                        key=f"feedback_inline_canvas_{candidate_id}_{st.session_state.get(epoch_key, 0)}",
                    )
                    if canvas.json_data is not None:
                        st.session_state[drawing_key] = copy.deepcopy(canvas.json_data)
                    try:
                        mask = canvas_drawing_to_source_mask(
                            canvas.image_data,
                            source_size=source_image.size,
                            initial_mask=initial_mask,
                        )
                        raw_mask = mask_png_bytes(mask)
                        mask_base64 = base64.b64encode(raw_mask).decode("ascii")
                        mask_width, mask_height = mask.size
                        mask_ratio = float((np.asarray(mask) >= 128).mean())
                        st.caption(
                            f"保存坐标：画布 {canvas_width}×{canvas_height} → 原图 {source_width}×{source_height}；"
                            f"当前 Mask 覆盖 {mask_ratio * 100:.2f}%。"
                        )
                    except Exception as exc:
                        st.error(friendly_error_message(exc, "当前绘制无法转换为原图 Mask。"))
    if mask_mode != "none":
        mask_note = st.text_input("Mask 审核说明（可选）", max_chars=1000, key=f"feedback_inline_mask_note_{candidate_id}")
        can_save = mask_mode == "original" or bool(mask_base64)
        save_scope = f"feedback-mask-save::{candidate_id}"
        save_clicked, save_status = render_action_button_with_feedback(
            "保存 Mask 修订",
            scope=save_scope,
            key=f"feedback_inline_save_mask_{candidate_id}",
            disabled=not can_save,
        )
        if save_clicked:
            update_action_feedback(save_scope, "info", "正在保存 Mask 修订...", save_status)
            result = api.save_feedback_annotation(candidate_id, {
                "annotation_type": "mask", "mask_base64": mask_base64,
                "use_original_mask": mask_mode == "original",
                "mask_width": mask_width, "mask_height": mask_height,
                "note": mask_note, "created_by": "local_operator",
            })
            _error(result, "Mask 保存失败。")
            if result.get("success"):
                set_action_feedback(REVIEW_ACTION_FEEDBACK_SCOPE, "success", "Mask 修订已保存并生成独立版本。")
                _mark_feedback_changed()
                update_action_feedback(save_scope, "success", "Mask 修订已保存；关闭编辑器后列表状态保持。", save_status)
            else:
                update_action_feedback(
                    save_scope,
                    "error",
                    friendly_error_message(result.get("error"), "Mask 保存失败。"),
                    save_status,
                )
    history = api.feedback_annotation_history(candidate_id)
    annotations = _payload(history, "annotations", [])
    if annotations:
        latest = annotations[0]
        st.caption(f"当前已有 Mask 修订 r{latest.get('revision')} · {_display_time(latest.get('created_at'))}")


@st.dialog("反馈样本详情", width="large")
def _show_feedback_detail_dialog(
    candidate_id: str,
    item: dict[str, Any],
    source_memory_bank_name: str,
) -> None:
    source_response = api.feedback_candidate_source(candidate_id)
    _error(source_response, "反馈样本详情读取失败。")
    source = dict(_payload(source_response, "source", {}))
    assets = dict(item.get("source_assets") or {})
    original_url = api.artifact_url(source.get("original_image_url"))
    source_kind = str(source.get("mode") or item.get("mode") or "")
    if not original_url:
        preview_response = api.feedback_candidate_source_preview(candidate_id)
        _error(preview_response, "原始样本预览读取失败。")
        preview = dict(_payload(preview_response, "source", {}))
        original_url = api.artifact_url(preview.get("url"))
        source_kind = str(preview.get("kind") or source_kind)
    if original_url:
        caption = "原始视频帧" if source_kind in {"video", "video_frame"} else "原始检测图片"
        st.image(original_url, caption=caption, width="stretch")
    else:
        st.warning("原始样本当前不可用，审核记录仍保留；请检查原检测结果文件是否已移动。")

    full_sample_name = str(item.get("sample_name") or "检测样本")
    st.write(f"**完整样本名称：** {full_sample_name}")
    info = st.columns(4)
    info[0].metric("检测时间", _display_detection_time(item))
    info[1].metric("检测类型", MODE_NAMES.get(str(item.get("mode") or ""), "检测"))
    info[2].metric(
        "原始判定",
        PREDICTION_NAMES.get(str(item.get("original_prediction") or "").lower(), "未确认"),
    )
    info[3].metric("人工反馈", corrected_label_text(item.get("corrected_label")))

    evidence = st.columns(4)
    score = item.get("original_score")
    threshold = item.get("threshold")
    evidence[0].metric("异常分数", f"{float(score):.4f}" if isinstance(score, (int, float)) else "-")
    evidence[1].metric("检测阈值", f"{float(threshold):.4f}" if isinstance(threshold, (int, float)) else "-")
    evidence[2].metric("缺陷类别", str(item.get("corrected_defect_type") or "未指定"))
    evidence[3].metric("Mask", "有 Mask" if item.get("has_mask") else "无 Mask")
    st.caption(f"用户反馈记录：{_user_feedback_text(item)}")
    if str(item.get("mode") or "") == "video":
        st.caption(
            f"帧编号：{item.get('frame_index') if item.get('frame_index') not in (None, '') else '-'} · "
            f"视频时间点：{float(item.get('timestamp_ms') or 0.0) / 1000.0:.3f} 秒"
        )

    visualizations = [
        ("热力图", source.get("heatmap_url") or assets.get("heatmap_url")),
        ("缺陷 Mask", source.get("mask_url") or assets.get("mask_url")),
        ("检测叠加图", source.get("result_image_url") or assets.get("overlay_url")),
    ]
    available = [(label, api.artifact_url(url)) for label, url in visualizations if api.artifact_url(url)]
    with st.expander("可视化与 Mask 证据", expanded=bool(available)):
        if not available:
            st.info("该检测结果没有生成可视化热力图、Mask 或叠加图。")
        for offset in range(0, len(available), 2):
            columns = st.columns(2)
            for column, (label, url) in zip(columns, available[offset:offset + 2]):
                with column:
                    st.image(url, caption=label, width="stretch")

    context = dict(item.get("context") or {})
    config_name = str(item.get("config_version") or "配置快照")
    config_profile_version = item.get("config_profile_version")
    config_label = f"{config_name}{f' · v{config_profile_version}' if config_profile_version else ''}"
    source_memory_bank_id = str(item.get("source_memory_bank_id") or context.get("source_memory_bank_id") or "")
    source_profile_id = str(item.get("source_profile_id") or context.get("source_profile_id") or "")
    source_config_id = str(
        context.get("source_config_snapshot_id")
        or item.get("config_profile_id")
        or context.get("config_profile_id")
        or ""
    )
    with st.expander("来源与技术详情", expanded=False):
        st.write(f"**来源 Memory Bank：** {source_memory_bank_name or _short_id(source_memory_bank_id)}")
        st.write(f"**来源 Config：** {config_label}")
        st.write(f"**来源 Profile：** {_short_id(source_profile_id, '未记录')}")
        st.caption(
            f"Memory Bank ID：{source_memory_bank_id or '未记录'}\n\n"
            f"Config ID：{source_config_id or '未记录'}\n\n"
            f"Profile ID：{source_profile_id or '未记录'}\n\n"
            f"Candidate ID：{candidate_id}"
        )


@st.dialog("审核与纠错", width="large")
def _show_review_dialog(candidate_id: str, item: dict[str, Any]) -> None:
    status = str(item.get("status") or "pending_review")
    preview = api.feedback_candidate_source_preview(candidate_id)
    source_preview = dict(_payload(preview, "source", {}))
    source_available = str(source_preview.get("kind") or "unavailable") != "unavailable"
    review_locked = status in {"queued", "optimizing", "ready_to_apply", "applied", "rolled_back", "archived"}
    optimization_locked = _candidate_optimization_locked(item)
    st.caption(
        f"{item.get('sample_name') or '检测样本'} · {_display_detection_time(item)} · "
        f"原判定 {PREDICTION_NAMES.get(str(item.get('original_prediction') or '').lower(), '未确认')} · "
        f"用户反馈 {_user_feedback_text(item)}"
    )
    _render_inline_mask_editor(candidate_id, item)
    if not source_available:
        st.warning("原始样本当前不可用：可以保存修正、退回或撤销，但不能审核通过。")
    elif optimization_locked:
        st.info("该反馈来自正式测试或冻结只读任务：审核结论只保存，不会用于资源更新。")
    reviewable = status in {"collected", "legacy_unresolved", "pending_review", "in_review", "conflict", "record_only"}
    label_options = ["keep", "normal", "defect"]
    current_label = str(item.get("corrected_label") or "keep")
    if current_label not in label_options:
        current_label = "keep"
    defect_options = ["未指定", "划痕", "裂纹", "异物", "变色", "变形", "其他"]
    existing_defect = str(item.get("corrected_defect_type") or "")
    default_defect = existing_defect if existing_defect in defect_options else ("其他" if existing_defect else "未指定")
    with st.form(f"feedback_review_decision_{candidate_id}", border=False):
        label = st.selectbox(
            "人工确认标签", label_options, index=label_options.index(current_label),
            format_func=lambda value: (
                "历史记录标签缺失"
                if str(item.get("mode") or "") == "video" and value == "keep"
                else LABEL_NAMES[value]
            ), key=f"feedback_dialog_label_{candidate_id}", disabled=review_locked,
        )
        defect_choice = st.selectbox(
            "缺陷类别", defect_options, index=defect_options.index(default_defect),
            disabled=review_locked, key=f"feedback_dialog_defect_choice_{candidate_id}",
        )
        other_defect = st.text_input(
            "其他缺陷类别（选择“其他”时填写）",
            value=existing_defect if default_defect == "其他" else "",
            max_chars=120,
            disabled=review_locked,
            key=f"feedback_dialog_defect_{candidate_id}",
        )
        review_note = st.text_area(
            "审核备注", value=str(item.get("review_note") or item.get("note") or ""),
            max_chars=1000, key=f"feedback_dialog_note_{candidate_id}", disabled=review_locked,
        )
        actions = st.columns([1, 1, 1, 1], vertical_alignment="center")
        save_update = actions[0].form_submit_button(
            "仅保存修正", disabled=not reviewable or review_locked, width="stretch",
        )
        accept = actions[1].form_submit_button(
            "保存并审核通过", type="primary",
            disabled=not reviewable or review_locked or not source_available, width="stretch",
        )
        reject = actions[2].form_submit_button(
            "退回", disabled=not reviewable or review_locked, width="stretch",
        )
        withdraw = actions[3].form_submit_button(
            "撤销", disabled=not reviewable or review_locked, width="stretch",
        )
    defect_type = (
        other_defect.strip()
        if defect_choice == "其他" and label == "defect"
        else "" if defect_choice == "未指定" or label != "defect" else defect_choice
    )
    action_scope = f"feedback-review::{candidate_id}"
    action_status = st.empty()
    render_action_feedback(action_scope, action_status)
    if str(item.get("mode") or "") == "video" and current_label == "keep":
        st.warning("该历史视频反馈没有人工主标签。请选择正常或缺陷后，可直接点击“保存并审核通过”。")
    if label == "defect" and not defect_type.strip():
        st.caption("缺陷类别可稍后补充；人工主标签已确认为缺陷即可审核。Mask 同样为可选信息。")
    selected_action = ""
    if save_update:
        selected_action = "update"
    elif accept:
        selected_action = "resolve_conflict" if status == "conflict" else "accept"
    elif reject:
        selected_action = "reject"
    elif withdraw:
        selected_action = "withdraw"
    if selected_action:
        update_action_feedback(action_scope, "info", "正在保存审核结果...", action_status)
        result = api.review_feedback_candidate(candidate_id, {
            "action": selected_action,
            "reviewer": "local_operator",
            "corrected_label": label,
            "corrected_defect_type": defect_type or None,
            "note": review_note or ("人工退回" if selected_action == "reject" else "人工撤销误提交" if selected_action == "withdraw" else ""),
            "require_snapshot": selected_action in {"accept", "resolve_conflict"},
            "candidate_version": int(item.get("candidate_version") or 1),
        })
        if result.get("success"):
            expected_status = {
                "update": "pending_review",
                "accept": "accepted",
                "resolve_conflict": "accepted",
                "reject": "rejected",
                "withdraw": "withdrawn",
            }.get(selected_action, "")
            saved_candidate = dict(result.get("candidate") or {})
            if expected_status and str(saved_candidate.get("status") or "") != expected_status:
                update_action_feedback(
                    action_scope,
                    "error",
                    "审核结果未达到预期状态，请刷新后重试；系统未显示虚假成功。",
                    action_status,
                )
                return
            set_action_feedback(
                REVIEW_ACTION_FEEDBACK_SCOPE,
                "success",
                {
                    "update": "人工修正已保存，反馈仍保留在待审核队列。",
                    "accept": "审核已通过，样本已进入该记忆库的纠错候选。",
                    "resolve_conflict": "标签冲突已解决并审核通过。",
                    "reject": "反馈已退回，不会进入记忆库更新。",
                    "withdraw": "误提交反馈已撤销。",
                }.get(selected_action, "审核操作已完成。"),
            )
            _discard_feedback_selection(candidate_id)
            _mark_feedback_changed()
            st.rerun()
        else:
            update_action_feedback(
                action_scope,
                "error",
                friendly_error_message(result.get("error"), "审核操作失败。"),
                action_status,
            )


def _render_target_resource_selector() -> dict[str, Any] | None:
    st.subheader("选择目标域记忆库")
    st.caption("先选择需要优化的目标域。后续审核、更新和记录管理只读取该记忆库产生的反馈，不会混入其他目标域。")
    resource_revision = int(st.session_state.get("aoi_resource_catalog_revision") or 0)
    resource_loading = st.empty()
    resource_loading.info("正在读取目标域反馈摘要...")
    response = cached_feedback_resources(current_feedback_revision, resource_revision)
    resource_loading.empty()
    _error(response, "目标域反馈资源读取失败。")
    resources = list(_payload(response, "resources", []))
    if not resources:
        st.info("当前没有可用的目标域记忆库。请先在目标域适配中创建并冻结目标域版本。")
        return None
    by_id = {str(item.get("memory_bank_id") or ""): item for item in resources if item.get("memory_bank_id")}
    legacy_selected = str(st.session_state.pop("feedback_center_memory_bank_id", "") or "")
    if legacy_selected and FEEDBACK_MEMORY_BANK_KEY not in st.session_state:
        store_persistent_value(
            st.session_state,
            FEEDBACK_MEMORY_BANK_KEY,
            legacy_selected,
            persistence_key=FEEDBACK_MEMORY_BANK_PERSISTENCE_KEY,
        )
    requested_id = str(st.query_params.get(FEEDBACK_MEMORY_BANK_KEY) or "").strip()
    selected_id = requested_id or str(
        persistent_value(
            st.session_state,
            FEEDBACK_MEMORY_BANK_KEY,
            persistence_key=FEEDBACK_MEMORY_BANK_PERSISTENCE_KEY,
            default="",
        ) or ""
    )
    if selected_id not in by_id:
        selected_id = ""
        st.session_state.pop(FEEDBACK_MEMORY_BANK_KEY, None)
        st.query_params.pop(FEEDBACK_MEMORY_BANK_KEY, None)
        store_persistent_value(
            st.session_state,
            FEEDBACK_MEMORY_BANK_KEY,
            "",
            persistence_key=FEEDBACK_MEMORY_BANK_PERSISTENCE_KEY,
        )
    elif requested_id != selected_id:
        st.query_params[FEEDBACK_MEMORY_BANK_KEY] = selected_id
    store_persistent_value(
        st.session_state,
        FEEDBACK_MEMORY_BANK_KEY,
        selected_id,
        persistence_key=FEEDBACK_MEMORY_BANK_PERSISTENCE_KEY,
    )
    for offset in range(0, len(resources), 2):
        columns = st.columns(2, gap="large")
        for column, resource in zip(columns, resources[offset:offset + 2]):
            memory_bank_id = str(resource.get("memory_bank_id") or "")
            with column.container(border=True):
                title = str(resource.get("friendly_name") or "目标域记忆库")
                active_mark = " · 当前已选择" if memory_bank_id == selected_id else ""
                st.markdown(f"**{title}**{active_mark}")
                st.caption(
                    f"{resource.get('dataset') or '-'} / {resource.get('category') or '-'} · "
                    f"版本 {int(resource.get('current_version') or 1)} · 记忆库 {_short_id(memory_bank_id)}"
                )
                metric_cols = st.columns(4)
                metric_cols[0].metric("反馈", _count(resource.get("feedback_count")))
                metric_cols[1].metric("待审核", _count(resource.get("pending_count")))
                metric_cols[2].metric("已通过", _count(resource.get("approved_count")))
                metric_cols[3].metric("可更新", _count(resource.get("updateable_count")))
                st.caption(f"最近反馈：{_display_time(resource.get('latest_feedback_at'))}")
                actions = st.columns(2)
                if actions[0].button(
                    "查看反馈样本",
                    type="primary" if memory_bank_id == selected_id else "secondary",
                    width="stretch",
                    key=f"feedback_resource_review_{memory_bank_id}",
                ):
                    store_persistent_value(
                        st.session_state,
                        FEEDBACK_MEMORY_BANK_KEY,
                        memory_bank_id,
                        persistence_key=FEEDBACK_MEMORY_BANK_PERSISTENCE_KEY,
                    )
                    st.query_params[FEEDBACK_MEMORY_BANK_KEY] = memory_bank_id
                    st.session_state["feedback_center_nav_target"] = "review"
                    _clear_feedback_selection_state()
                    # The button click has already started a script pass.  Keep
                    # the chosen resource in this pass instead of triggering a
                    # second full-page rerun.
                    selected_id = memory_bank_id
                if actions[1].button(
                    "进入更新",
                    width="stretch",
                    key=f"feedback_resource_update_{memory_bank_id}",
                ):
                    store_persistent_value(
                        st.session_state,
                        FEEDBACK_MEMORY_BANK_KEY,
                        memory_bank_id,
                        persistence_key=FEEDBACK_MEMORY_BANK_PERSISTENCE_KEY,
                    )
                    st.query_params[FEEDBACK_MEMORY_BANK_KEY] = memory_bank_id
                    st.session_state["feedback_center_nav_target"] = "optimization"
                    _clear_feedback_selection_state()
                    selected_id = memory_bank_id
    if not selected_id:
        st.info("请选择一个目标域记忆库后继续。")
        return None
    selected = by_id[selected_id]
    st.success(
        f"当前目标域：{selected.get('friendly_name') or '目标域记忆库'} · "
        f"{selected.get('dataset') or '-'} / {selected.get('category') or '-'}"
    )
    return selected


current_feedback_revision = _load_feedback_revision()
selected_resource = _render_target_resource_selector()
if selected_resource is None:
    st.stop()
selected_memory_bank_id = str(selected_resource.get("memory_bank_id") or "")
selected_source_profile_id = str(selected_resource.get("source_profile_id") or selected_memory_bank_id)
selected_target_domain_id = str(selected_resource.get("source_target_domain_id") or selected_source_profile_id)
active_tab = _feedback_active_tab()
# Remove the legacy page-wide flash channel.  Each action now owns a nearby
# status placeholder; this also prevents ordinary CRUD results from appearing
# in Streamlit's fixed top-right layer.
st.session_state.pop("_action_feedback::feedback-center::page-notice", None)

if active_tab == "review":
    review_heading = st.columns([1, 2], vertical_alignment="center")
    review_heading[0].subheader("待审核反馈")
    with review_heading[1]:
        review_action_status = st.empty()
    render_action_feedback(REVIEW_ACTION_FEEDBACK_SCOPE, review_action_status)
    st.caption("在同一处核对检测信息、修正真实标签、补充可选 Mask，并决定审核通过或撤销误提交。原始检测结果始终保持不变。")
    review_notice = st.session_state.pop("feedback_review_notice", None)
    if isinstance(review_notice, dict):
        review_level = str(review_notice.get("level") or "info")
        review_message = str(review_notice.get("message") or "审核操作已完成。")
        failed_rows = list(review_notice.get("failed_rows") or [])
        if failed_rows:
            getattr(st, review_level if review_level in {"warning", "error"} else "warning")(review_message)
            st.dataframe(failed_rows, hide_index=True, width="stretch")
        else:
            update_action_feedback(REVIEW_ACTION_FEEDBACK_SCOPE, review_level, review_message, review_action_status)
    with st.form("feedback_review_filter_form", border=False):
        filter_cols = st.columns([2.2, 1.05, 1.0, 1.0, 0.95, 0.7], vertical_alignment="bottom")
        keyword_filter = filter_cols[0].text_input(
            "关键词",
            key="feedback_center_keyword",
            placeholder="样本名称、缺陷类别或备注",
        )
        status_filter = filter_cols[1].selectbox(
            "状态", ["all", "pending_review", "in_review", "conflict", "record_only"],
            format_func=lambda value: {"all": "全部待审核", "pending_review": "待审核", "in_review": "审核中", "conflict": "标签冲突", "record_only": "只记录待审核"}.get(value, value),
            key="feedback_center_status",
        )
        label_filter = filter_cols[2].selectbox(
            "人工标签",
            ["all", "normal", "defect", "unknown"],
            format_func=lambda value: {"all": "全部", "normal": "正常", "defect": "缺陷", "unknown": "待确定"}[value],
            key="feedback_center_label",
        )
        candidate_time = filter_cols[3].selectbox(
            "检测时间", ["全部", "今天", "最近 7 天", "最近 30 天", "最近 90 天"], key="feedback_center_time"
        )
        candidate_mask = filter_cols[4].selectbox(
            "Mask", ["all", "with", "without"],
            format_func=lambda value: {"all": "全部", "with": "有 Mask", "without": "无 Mask"}[value],
            key="feedback_center_mask",
        )
        with filter_cols[5]:
            refresh_filters = st.form_submit_button("搜索", width="stretch", icon=":material/search:")
        detail_filter_cols = st.columns([0.9, 1.0, 1.05, 1.05, 1.25, 0.8])
        mode_filter = detail_filter_cols[0].selectbox(
            "检测模式", ["all", "single", "batch", "video"],
            format_func=lambda value: {"all": "全部", "single": "单图", "batch": "批量", "video": "视频"}.get(value, value),
            key="feedback_center_mode",
        )
        issue_filter = detail_filter_cols[1].selectbox(
            "问题类型", ["all", "false_alarm", "missed_detection", "correct", "uncertain"],
            format_func=lambda value: {"all": "全部", "false_alarm": "误检", "missed_detection": "漏检", "correct": "结果正确", "uncertain": "不确定"}.get(value, value),
            key="feedback_center_issue",
        )
        candidate_dataset = detail_filter_cols[2].text_input("数据集", key="feedback_center_dataset", placeholder="全部")
        candidate_category = detail_filter_cols[3].text_input("类别", key="feedback_center_category", placeholder="全部")
        candidate_method = detail_filter_cols[4].text_input("方法模式", key="feedback_center_method", placeholder="全部")
        candidate_score_text = detail_filter_cols[5].text_input("最低分", key="feedback_center_min_score", placeholder="不限")
    if refresh_filters:
        _refresh()
    review_statuses = ("collected", "legacy_unresolved", "pending_review", "in_review", "conflict", "record_only")
    review_status_query = ",".join(review_statuses) if status_filter == "all" else status_filter
    filter_fingerprint = repr((
        keyword_filter.strip().lower(), status_filter, label_filter, candidate_time, candidate_mask,
        mode_filter, issue_filter, candidate_dataset.strip().lower(), candidate_category.strip().lower(),
        candidate_method.strip().lower(), candidate_score_text.strip(),
    ))
    if st.session_state.get("feedback_review_filter_fingerprint") != filter_fingerprint:
        st.session_state["feedback_review_filter_fingerprint"] = filter_fingerprint
        st.session_state["feedback_review_page"] = 1
    page_size_hint = int(st.session_state.get("feedback_review_page_size") or 20)
    if page_size_hint not in {10, 20, 50, 100}:
        page_size_hint = 20
    page_hint = max(1, int(st.session_state.get("feedback_review_page") or 1))
    minimum_score: float | None = None
    if candidate_score_text.strip():
        try:
            minimum_score = float(candidate_score_text.strip())
        except (TypeError, ValueError):
            st.warning("最低分请输入数字，当前暂不应用该筛选。")
    detection_since = ""
    if candidate_time != "全部":
        days = {"今天": 1, "最近 7 天": 7, "最近 30 天": 30, "最近 90 天": 90}[candidate_time]
        detection_since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    response = cached_feedback_candidates(
        current_feedback_revision,
        review_status_query,
        "" if mode_filter == "all" else mode_filter,
        selected_memory_bank_id,
        selected_source_profile_id,
        selected_target_domain_id,
        page_size_hint,
        page=page_hint,
        page_size=page_size_hint,
        search=keyword_filter.strip(),
        label="" if label_filter == "all" else label_filter,
        issue="" if issue_filter == "all" else issue_filter,
        mask="" if candidate_mask == "all" else candidate_mask,
        detection_since=detection_since,
        min_score=minimum_score,
        dataset=candidate_dataset.strip(),
        category=candidate_category.strip(),
        method=candidate_method.strip(),
    )
    _error(response, "候选样本读取失败。")
    page_candidates = list(_payload(response, "candidates", []))
    total_candidates = int(_payload(response, "total", 0) or 0)
    page_count = max(1, int(_payload(response, "total_pages", 1) or 1))
    if page_hint > page_count:
        page_hint = page_count
        st.session_state["feedback_review_page"] = page_count
    if not page_candidates:
        st.info("暂时没有符合当前筛选条件的反馈。可在单图、批量或视频检测结果中提交误检、漏检或结果修正。")
    else:
        pagination_cols = st.columns([0.8, 0.8, 2.4], vertical_alignment="bottom")
        page_size = pagination_cols[0].selectbox("每页", [10, 20, 50, 100], index=1, key="feedback_review_page_size")
        stored_page = max(1, min(page_count, int(st.session_state.get("feedback_review_page") or 1)))
        if st.session_state.get("feedback_review_page") != stored_page:
            st.session_state["feedback_review_page"] = stored_page
        current_page = pagination_cols[1].number_input("页码", min_value=1, max_value=page_count, step=1, key="feedback_review_page")
        start = (int(current_page) - 1) * int(page_size)
        end = start + len(page_candidates)
        pagination_cols[2].caption(
            f"第 {int(current_page)} / {page_count} 页 · 共 {total_candidates} 条 · 当前显示 {start + 1}–{end}"
        )
        reviewable_statuses = {"collected", "legacy_unresolved", "pending_review", "in_review", "conflict", "record_only"}
        visible_reviewable = {str(item.get("candidate_id")) for item in page_candidates if str(item.get("status") or "") in reviewable_statuses}
        selected_ids = _selected_feedback_ids()
        selection_actions = st.columns([1, 1, 1, 1, 2])
        selection_actions[0].button(
            "全选当前页", disabled=not visible_reviewable, key="feedback_select_page",
            on_click=_set_feedback_page_selection,
            args=("review", tuple(sorted(visible_reviewable)), True),
        )
        selection_actions[1].button(
            "取消当前页", disabled=not (selected_ids & visible_reviewable), key="feedback_clear_page",
            on_click=_set_feedback_page_selection,
            args=("review", tuple(sorted(visible_reviewable)), False),
        )
        selection_actions[2].button(
            "全选当前页", disabled=not visible_reviewable, key="feedback_select_filtered",
            on_click=_set_feedback_page_selection,
            args=("review", tuple(sorted(visible_reviewable)), True),
        )
        selection_actions[3].button(
            "清空选择", disabled=not selected_ids, key="feedback_clear_selection",
            on_click=_clear_feedback_selection,
            args=("review", tuple(sorted(visible_reviewable))),
        )
        selection_actions[4].caption(f"已选择 {len(selected_ids)} 条；选择状态按反馈记录 ID 跨页保持。")

        with st.expander("批量审核预览", expanded=bool(selected_ids)):
            chosen = [item for item in page_candidates if str(item.get("candidate_id")) in selected_ids]
            conflicts = sum(str(item.get("status") or "") == "conflict" for item in chosen)
            missing_label_items = [
                item for item in chosen
                if str(item.get("corrected_label") or "keep") not in {"normal", "defect"}
            ]
            missing_labels = len(missing_label_items)
            defect_without_mask = sum(str(item.get("corrected_label") or "") == "defect" and not item.get("has_mask") for item in chosen)
            st.write(f"将处理：{len(chosen)} 条反馈；标签冲突 {conflicts} 条；标签未确认 {missing_labels} 条；缺陷但无掩码 {defect_without_mask} 条。")
            if missing_label_items:
                st.warning("以下历史反馈缺少人工主标签；批量审核只会跳过这些记录，其他有效记录仍会继续处理。")
                st.dataframe(
                    [
                        {
                            "样本名称": str(item.get("sample_name") or "检测样本"),
                            "检测来源": MODE_NAMES.get(str(item.get("mode") or ""), "检测"),
                            "缺失内容": "人工标签（正常/缺陷）",
                            "处理方式": "补充标签后重新审核",
                        }
                        for item in missing_label_items
                    ],
                    hide_index=True,
                    width="stretch",
                )
                missing_video_items = [item for item in missing_label_items if str(item.get("mode") or "") == "video"]
                if missing_video_items:
                    st.caption("同一人工判断的视频帧可在这里统一补充；标签不同的帧请分组选择，或逐条打开“审核与纠错”。")
                    supplement_label = st.radio(
                        "为所选缺失视频反馈补充人工标签",
                        ["normal", "defect"],
                        index=None,
                        format_func=lambda value: LABEL_NAMES[value],
                        horizontal=True,
                        key="feedback_bulk_missing_video_label",
                    )
                    if st.button(
                        "补充所选缺失标签",
                        disabled=supplement_label not in {"normal", "defect"},
                        key="feedback_bulk_missing_video_apply",
                    ):
                        supplement_results: list[dict[str, Any]] = []
                        for item in missing_video_items:
                            candidate_id = str(item.get("candidate_id") or "")
                            response = api.review_feedback_candidate(candidate_id, {
                                "action": "update",
                                "reviewer": "local_operator",
                                "corrected_label": supplement_label,
                                "corrected_defect_type": None,
                                "note": "补充历史视频反馈人工标签",
                                "require_snapshot": False,
                                "candidate_version": int(item.get("candidate_version") or 1),
                            })
                            supplement_results.append({
                                "candidate_id": candidate_id,
                                "sample_name": str(item.get("sample_name") or "检测样本"),
                                "success": bool(response.get("success")),
                                "message": str((response.get("error") or {}).get("message") or ""),
                            })
                        success_count = sum(bool(item.get("success")) for item in supplement_results)
                        failed_rows = [
                            {
                                "样本名称": item["sample_name"],
                                "处理结果": "标签补充失败",
                                "原因": item["message"] or "保存失败",
                            }
                            for item in supplement_results if not item.get("success")
                        ]
                        _mark_feedback_changed()
                        message = f"历史视频标签补充完成：成功 {success_count} 条，失败 {len(failed_rows)} 条。补充后仍需执行审核通过。"
                        if failed_rows:
                            st.session_state["feedback_review_notice"] = {
                                "level": "warning",
                                "message": message,
                                "failed_rows": failed_rows,
                            }
                        else:
                            set_action_feedback(REVIEW_ACTION_FEEDBACK_SCOPE, "success", message)
                        st.rerun()
            bulk_note = st.text_input("批量审核备注", max_chars=1000, key="feedback_bulk_review_note")
            bulk_actions = st.columns([1, 1, 1, 2], vertical_alignment="center")
            bulk_accept = bulk_actions[0].button("批量审核通过", disabled=not selected_ids, key="feedback_review_bulk_accept", width="stretch")
            bulk_reject = bulk_actions[1].button("批量退回", disabled=not selected_ids, key="feedback_review_bulk_reject", width="stretch")
            bulk_withdraw = bulk_actions[2].button("批量撤销", disabled=not selected_ids, key="feedback_review_bulk_withdraw", width="stretch")
            bulk_scope = f"feedback-bulk-review::{selected_memory_bank_id}"
            with bulk_actions[3]:
                bulk_status = st.empty()
            render_action_feedback(bulk_scope, bulk_status)
            if bulk_accept or bulk_reject or bulk_withdraw:
                bulk_action = "accept" if bulk_accept else "reject" if bulk_reject else "withdraw"
                action_name = {"accept": "审核通过", "reject": "退回", "withdraw": "撤销"}[bulk_action]
                update_action_feedback(
                    bulk_scope,
                    "info",
                    f"正在批量{action_name} {len(selected_ids)} 条反馈...",
                    bulk_status,
                )
                result = api.review_feedback_candidates_bulk(
                    sorted(selected_ids), action=bulk_action,
                    reviewer="local_operator", note=bulk_note or "反馈优化中心批量审核", require_snapshot=True,
                )
                _error(result, "批量审核失败。")
                if result.get("success"):
                    _mark_feedback_changed()
                    success_count = int(result.get("success_count") or 0)
                    failed_count = int(result.get("failed_count") or 0)
                    result_rows = list(result.get("results") or [])
                    successful_ids = {str(row.get("candidate_id") or "") for row in result_rows if row.get("success")}
                    remaining = selected_ids - successful_ids
                    _store_feedback_selection(remaining, scope="review", visible_ids=visible_reviewable)
                    failed_rows = [
                        {
                            "样本名称": str(row.get("sample_name") or "检测样本"),
                            "处理结果": "未处理",
                            "原因": str((row.get("error") or {}).get("message") or "审核失败"),
                        }
                        for row in result_rows if not row.get("success")
                    ]
                    if failed_count:
                        set_action_feedback(
                            bulk_scope,
                            "warning",
                            f"批量{action_name}完成：成功 {success_count} 条，失败 {failed_count} 条。",
                        )
                        st.session_state["feedback_review_notice"] = {
                            "level": "warning",
                            "message": f"批量{action_name}完成：成功 {success_count} 条，失败 {failed_count} 条。失败项不会影响其他记录。",
                            "failed_rows": failed_rows,
                        }
                    else:
                        set_action_feedback(
                            bulk_scope,
                            "success",
                            f"批量{action_name}完成：成功 {success_count} 条。",
                        )
                    st.rerun()
                else:
                    update_action_feedback(
                        bulk_scope,
                        "error",
                        friendly_error_message(result.get("error"), "批量审核失败。"),
                        bulk_status,
                    )

    if page_candidates:
        st.markdown("#### 人工审核工作台")
        st.caption("一级表格只保留审核决策信息；检测指标、来源 Config、Profile 和技术 ID 请在“查看”中核对。")
        widths = list(REVIEW_TABLE_WIDTHS)
        header_columns = st.columns(widths, gap="small")
        for column, label in zip(header_columns, REVIEW_TABLE_HEADERS):
            column.markdown(f"**{label}**")
        selected_ids = _selected_feedback_ids()
        source_memory_bank_name = str(
            selected_resource.get("friendly_name") or _short_id(selected_memory_bank_id, "目标域记忆库")
        )
        for sequence, item in enumerate(
            page_candidates,
            start=(int(current_page) - 1) * int(page_size) + 1 if total_candidates else 1,
        ):
            candidate_id = str(item.get("candidate_id") or "")
            status = str(item.get("status") or "pending_review")
            can_select = status in reviewable_statuses
            with st.container(border=True):
                columns = st.columns(widths, gap="small")
                columns[0].markdown(f"**{sequence}**")
                checkbox_key = _feedback_checkbox_key("review", candidate_id)
                # The canonical ID set is authoritative.  Reapply it before the
                # widget is created so dialog/batch actions cannot leave a stale
                # visible tick behind.
                st.session_state[checkbox_key] = candidate_id in selected_ids
                columns[1].checkbox(
                    "选择", key=checkbox_key, disabled=not can_select, label_visibility="collapsed",
                    on_change=_feedback_checkbox_changed, args=("review", candidate_id),
                )
                columns[2].write(_display_detection_time(item))
                full_sample_name = str(item.get("sample_name") or _short_id(item.get("sample_id"), "检测样本"))
                columns[3].markdown(compact_sample_name(full_sample_name), help=full_sample_name)
                if str(item.get("mode") or "") == "video":
                    columns[3].caption(
                        f"帧 {item.get('frame_index') if item.get('frame_index') not in (None, '') else '-'} · "
                        f"{float(item.get('timestamp_ms') or 0.0) / 1000.0:.3f}s"
                    )
                columns[4].write(PREDICTION_NAMES.get(str(item.get("original_prediction") or "").lower(), "未确认"))
                columns[5].write(corrected_label_text(item.get("corrected_label")))
                columns[6].write(
                    str(item.get("corrected_defect_type") or "-")
                    if str(item.get("corrected_label") or "") == "defect"
                    else "-"
                )
                if columns[7].button(
                    "有Mask" if item.get("has_mask") else "无Mask",
                    type="tertiary",
                    help="查看或编辑该样本的 Mask",
                    key=f"feedback_mask_{candidate_id}",
                    width="stretch",
                ):
                    _show_review_dialog(candidate_id, item)
                columns[8].markdown(
                    compact_sample_name(source_memory_bank_name, max_length=20),
                    help=source_memory_bank_name,
                )
                columns[9].write(STATUS_LABELS.get(status, "状态未知"))
                action_columns = columns[10].columns(2, gap="small")
                if action_columns[0].button(
                    "查看",
                    help="查看原图、热力图、检测指标与来源技术信息",
                    key=f"feedback_detail_{candidate_id}",
                    width="stretch",
                ):
                    _show_feedback_detail_dialog(candidate_id, item, source_memory_bank_name)
                if action_columns[1].button(
                    "审核",
                    type="primary",
                    key=f"feedback_open_review_{candidate_id}",
                    width="stretch",
                ):
                    _show_review_dialog(candidate_id, item)

if active_tab == "optimization":
    st.subheader("记忆库更新")
    if inference_locked:
        st.warning("当前处于正式测试锁定状态：反馈可以继续记录，但不能创建、应用或回滚优化版本。")
    else:
        st.info("反馈更新始终在独立候选目录中离线执行。基础记忆库和冻结 profile 不会被原地修改。")
    update_status_query = ",".join((
        "accepted", "queued", "running", "optimizing", "ready_to_apply",
        "consumed", "validation_failed", "frozen", "applied", "rolled_back", "completed",
    ))
    update_page_size_hint = int(st.session_state.get("feedback_update_page_size") or 20)
    if update_page_size_hint not in {10, 20, 50}:
        update_page_size_hint = 20
    update_page_hint = max(1, int(st.session_state.get("feedback_update_page") or 1))
    update_label_hint = str(st.session_state.get("feedback_update_label_filter") or "all")
    update_mask_hint = str(st.session_state.get("feedback_update_mask_filter") or "all")
    update_usage_hint = str(st.session_state.get("feedback_update_usage_filter") or "all")
    update_time_hint = str(st.session_state.get("feedback_update_time_filter") or "全部")
    update_keyword_hint = str(st.session_state.get("feedback_update_keyword") or "")
    update_filter_fingerprint = repr((
        update_label_hint, update_mask_hint, update_usage_hint, update_time_hint,
        update_keyword_hint.strip().lower(), update_page_size_hint,
    ))
    if st.session_state.get("feedback_update_filter_fingerprint") != update_filter_fingerprint:
        st.session_state["feedback_update_filter_fingerprint"] = update_filter_fingerprint
        st.session_state["feedback_update_page"] = 1
        update_page_hint = 1
    update_detection_since = ""
    if update_time_hint != "全部":
        days = {"今天": 1, "最近 7 天": 7, "最近 30 天": 30, "最近 90 天": 90}[update_time_hint]
        update_detection_since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    accepted_response = cached_feedback_candidates(
        current_feedback_revision,
        update_status_query,
        "",
        selected_memory_bank_id,
        selected_source_profile_id,
        selected_target_domain_id,
        update_page_size_hint,
        page=update_page_hint,
        page_size=update_page_size_hint,
        search=update_keyword_hint.strip(),
        label="" if update_label_hint == "all" else update_label_hint,
        mask="" if update_mask_hint == "all" else update_mask_hint,
        usage="" if update_usage_hint == "all" else update_usage_hint,
        detection_since=update_detection_since,
    )
    _error(accepted_response, "纠错样本读取失败。")
    correction_rows = list(_payload(accepted_response, "candidates", []))
    correction_total = int(_payload(accepted_response, "total", 0) or 0)
    correction_pool_response = api.feedback_correction_pool(
        memory_bank_id=selected_memory_bank_id,
        source_profile_id=selected_source_profile_id,
        source_target_domain_id=selected_target_domain_id,
    )
    _error(correction_pool_response, "可用纠错样本池读取失败。")
    available_pool = _payload(correction_pool_response, "candidates", [])
    available_ids = {str(item.get("candidate_id") or "") for item in available_pool}
    accepted = [
        item for item in correction_rows
        if str(item.get("candidate_id") or "") in available_ids
        and not _candidate_optimization_locked(item)
    ]
    if not correction_rows:
        st.info("暂无已审核通过的样本。请先完成标签确认和审核。")
    else:
        st.markdown("#### 目标域反馈资源")
        with st.container(border=True):
            st.markdown(f"**{selected_resource.get('friendly_name') or '目标域记忆库'}**")
            summary_cols = st.columns(5)
            summary_cols[0].metric("可用纠错样本", len(accepted))
            summary_cols[1].metric("正常纠正", sum(str(row.get("corrected_label") or "") == "normal" for row in correction_rows))
            summary_cols[2].metric("缺陷纠正", sum(str(row.get("corrected_label") or "") == "defect" for row in correction_rows))
            summary_cols[3].metric("有 Mask", sum(bool(row.get("has_mask")) for row in correction_rows))
            summary_cols[4].metric("已用于更新", sum(int(row.get("usage_count") or 0) > 0 for row in correction_rows))
            latest_review = max((str(row.get("reviewed_at") or "") for row in correction_rows), default="")
            st.caption(
                f"记忆库 {_short_id(selected_memory_bank_id)} · "
                f"{selected_resource.get('dataset') or '-'} / {selected_resource.get('category') or '-'} · "
                f"最近审核：{_display_time(latest_review)} · 当前筛选共 {correction_total:,} 条"
            )

        st.markdown("#### 第一步：确认目标记忆库")
        profiles_response = api.runtime_profiles()
        _error(profiles_response, "目标域 profile 读取失败。")
        profiles = [
            item for item in _payload(profiles_response, "profiles", [])
            if str(item.get("source_type") or "") != "builtin_public_baseline"
            and bool(item.get("frozen"))
            and not bool(item.get("archived"))
            and selected_memory_bank_id in {
                str(item.get("profile_id") or ""),
                str(item.get("memory_bank_id") or ""),
            }
        ]
        profile_by_id = {str(item.get("profile_id") or ""): item for item in profiles if item.get("profile_id")}
        inferred_profile = next(iter(profile_by_id), "")
        profile_options = list(profile_by_id)
        profile_id = st.selectbox(
            "目标记忆库与基础版本",
            profile_options,
            index=profile_options.index(inferred_profile) if inferred_profile in profile_options else 0,
            format_func=lambda value: f"{profile_by_id[value].get('display_name') or _short_id(value)} · {profile_by_id[value].get('dataset')}/{profile_by_id[value].get('category')} · {METHOD_MODE_LABELS.get(str(profile_by_id[value].get('method_mode') or ''), '目标域检测')}{' · 当前运行' if profile_by_id[value].get('current') else ''}",
            key="feedback_opt_profile",
        ) if profile_options else ""
        base_mode = "normal_only"
        if not profile_options:
            st.error("暂无可作为基础版本的冻结目标域 profile，请先完成目标域适配。")
        elif profile_id:
            base_profile = profile_by_id[profile_id]
            base_metadata = dict(base_profile.get("metadata") or {})
            base_mode = str(base_profile.get("method_mode") or base_metadata.get("method_mode_resolved") or "normal_only")
            with st.container(border=True):
                summary_columns = st.columns(4)
                summary_columns[0].metric("目标记忆库", str(base_profile.get("display_name") or "目标域记忆库"))
                summary_columns[1].metric("当前检测方式", METHOD_MODE_LABELS.get(base_mode, "目标域检测"))
                summary_columns[2].metric("数据集", str(base_profile.get("dataset") or "-"))
                summary_columns[3].metric("类别", str(base_profile.get("category") or "-"))
                st.caption(
                    f"模型：{base_metadata.get('model_id') or '-'} · 特征维度：{base_metadata.get('feature_dimension') or '-'} · "
                    f"完整性：{base_profile.get('integrity_status_label') or '-'} · 当前冻结资源只作为输入，不会被原地修改。"
                )

        st.markdown("#### 第二步：选择纠错样本")
        sample_filter_cols = st.columns([1.0, 1.0, 1.0])
        label_filter = sample_filter_cols[0].selectbox("审核标签", ["all", "normal", "defect"], format_func=lambda value: {"all": "全部", "normal": "正常", "defect": "缺陷"}[value], key="feedback_update_label_filter")
        mask_filter = sample_filter_cols[1].selectbox("Mask", ["all", "with", "without"], format_func=lambda value: {"all": "全部", "with": "有 Mask", "without": "无 Mask"}[value], key="feedback_update_mask_filter")
        usage_filter = sample_filter_cols[2].selectbox("使用状态", ["all", "unused", "used"], format_func=lambda value: {"all": "全部", "unused": "从未使用", "used": "已使用过"}[value], key="feedback_update_usage_filter")
        extra_filter_cols = st.columns([1.0, 1.6, 0.7])
        update_time_filter = extra_filter_cols[0].selectbox("检测时间", ["全部", "今天", "最近 7 天", "最近 30 天", "最近 90 天"], key="feedback_update_time_filter")
        keyword = extra_filter_cols[1].text_input("关键词", placeholder="样本名称、缺陷类型或审核备注", key="feedback_update_keyword")
        page_size = extra_filter_cols[2].selectbox("每页数量", [10, 20, 50], index=1, key="feedback_update_page_size")
        visible = list(correction_rows)
        # Label, Mask, usage, time and keyword predicates are sent to the
        # paginated API above.  Keep this page slice intact so a filter never
        # falls back to downloading the historical pool and filtering in the
        # Streamlit process.
        update_page_count = max(1, (correction_total + int(page_size) - 1) // int(page_size))
        stored_update_page = min(update_page_count, max(1, int(st.session_state.get("feedback_update_page", 1))))
        if st.session_state.get("feedback_update_page") != stored_update_page:
            st.session_state["feedback_update_page"] = stored_update_page
        update_page = st.number_input("当前页", min_value=1, max_value=update_page_count, step=1, key="feedback_update_page")
        update_start = (int(update_page) - 1) * int(page_size)
        # The backend already returned this page; do not slice it a second
        # time, otherwise page 2+ would appear empty.
        visible_page = visible
        accepted_by_id = {str(item["candidate_id"]): item for item in accepted}
        selected_set = _selected_feedback_ids(set(accepted_by_id))
        visible_selectable = {str(row.get("candidate_id") or "") for row in visible_page if str(row.get("candidate_id") or "") in accepted_by_id}
        controls = st.columns([1, 1, 1, 2])
        controls[0].button(
            "全选当前页", disabled=not visible_selectable, key="feedback_update_select_visible",
            on_click=_set_feedback_page_selection,
            args=("update", tuple(sorted(visible_selectable)), True),
        )
        controls[1].button(
            "取消当前页", disabled=not (selected_set & visible_selectable), key="feedback_update_clear_visible",
            on_click=_set_feedback_page_selection,
            args=("update", tuple(sorted(visible_selectable)), False),
        )
        controls[2].button(
            "清除选择", disabled=not selected_set, key="feedback_update_clear_all",
            on_click=_clear_feedback_selection,
            args=("update", tuple(sorted(visible_selectable))),
        )
        update_end = min(correction_total, update_start + len(visible_page))
        controls[3].caption(f"当前筛选 {correction_total} 条，显示第 {update_start + 1 if visible else 0}–{update_end} 条；跨页已选择 {len(selected_set)} 条。")
        for sequence, row in enumerate(visible_page, start=update_start + 1):
            candidate_id = str(row.get("candidate_id") or "")
            selectable = candidate_id in accepted_by_id
            with st.container(border=True):
                row_cols = st.columns([0.45, 0.55, 3.4, 1.0])
                checkbox_key = _feedback_checkbox_key("update", candidate_id)
                st.session_state[checkbox_key] = candidate_id in selected_set
                row_cols[0].checkbox(
                    "选择", key=checkbox_key, disabled=not selectable, label_visibility="collapsed",
                    on_change=_feedback_checkbox_changed, args=("update", candidate_id),
                )
                row_cols[1].markdown(f"`{sequence}`")
                row_cols[2].markdown(f"**{str(row.get('sample_name') or _short_id(row.get('sample_id')))}**")
                usage_count = int(row.get("usage_count") or 0)
                row_cols[3].write(f"使用 {usage_count} 次")
                st.caption(
                    f"检测时间：{_display_detection_time(row)} · 原判定：{row.get('original_prediction') or '-'} → 审核标签：{LABEL_NAMES.get(str(row.get('corrected_label') or ''), '-')} · "
                    f"缺陷类型：{row.get('corrected_defect_type') or '-'} · Mask：{'有' if row.get('has_mask') else '无'}"
                )
                st.caption(
                    f"异常分数/原阈值：{row.get('original_score') if row.get('original_score') is not None else '-'} / {row.get('threshold') if row.get('threshold') is not None else '-'} · "
                    f"审核时间：{_display_time(row.get('reviewed_at'))} · 来源记忆库：{_short_id(row.get('source_memory_bank_id'))} · "
                    f"最近使用：{_display_time(row.get('latest_used_at')) if usage_count else '从未使用'}"
                )
                with st.expander("查看审核详情", expanded=False):
                    st.write(f"审核备注：{row.get('review_note') or '无'}")
                    st.write(f"来源目标域：{_short_id(row.get('source_profile_id'))} · 纠错样本标识：{_short_id(candidate_id)}")
                    usage_history = list(row.get("usage_history") or [])
                    if usage_history:
                        st.markdown("**使用历史**")
                        for usage in usage_history:
                            strategy_name = STRATEGY_DISPLAY_NAMES.get(
                                str(usage.get("strategy") or ""), "目标域资源更新"
                            )
                            st.caption(
                                f"{_display_time(usage.get('used_at'))} · {strategy_name} · "
                                f"任务 {_short_id(usage.get('optimization_task_id'))} · "
                                f"生成版本 {_short_id(usage.get('generated_profile_id'))} · "
                                f"结果 {usage.get('usage_result') or '-'}"
                            )
                    else:
                        st.caption("该纠错样本尚未用于 Memory Bank 更新。")
        selected_set = _selected_feedback_ids(set(accepted_by_id))
        selected_ids = sorted(selected_set)
        selected_rows = [accepted_by_id[value] for value in selected_ids if value in accepted_by_id]
        normal_count = sum(item.get("corrected_label") == "normal" for item in selected_rows)
        defect_count = sum(item.get("corrected_label") == "defect" for item in selected_rows)
        st.caption(
            f"已选择 {len(selected_ids)} 条：正常 {normal_count}、缺陷 {defect_count}。"
            f"全部绑定到当前目标域 {_short_id(selected_memory_bank_id)}。"
        )

        source_modes = sorted({str(item.get("method_mode") or "") for item in selected_rows if item.get("method_mode")})
        if profile_id and source_modes and any(value != base_mode for value in source_modes):
            st.info("反馈样本可能由同一记忆库下的其他检测配置产生。系统按记忆库、模型和特征空间校验，不会只因校准方式不同而阻止更新。")

        st.markdown("#### 第三步：选择更新内容")
        strategy_options = ["threshold_only", "normal_memory"]
        if base_mode in {"normal_defect_dual_memory", "dual_memory"}:
            strategy_options.append("defect_prototypes")
        if base_mode in {"normal_memory_defect_guided_lite", "lightweight_calibration"}:
            strategy_options.append("lightweight_calibrator")
        strategy_options.append("comprehensive")
        strategy_labels = _strategy_labels(strategy_options)
        strategy = st.radio(
            "本次更新内容",
            strategy_options,
            format_func=strategy_labels.get,
            horizontal=True,
            key="feedback_opt_strategy",
            help="选择后，下方会立即显示该策略允许的样本、Mask 要求、实际更新资源和预计耗时。内部策略代码保持不变。",
        )
        defaults = {
            "threshold_only": (True, False, False, False),
            "normal_memory": (False, True, False, False),
            "defect_prototypes": (False, False, True, False),
            "lightweight_calibrator": (True, False, False, True),
            "comprehensive": (
                True,
                True,
                "defect_prototypes" in strategy_options,
                "lightweight_calibrator" in strategy_options,
            ),
        }[strategy]
        update_threshold, update_normal, update_defect, update_lite = defaults
        selected_help = dict(STRATEGY_HELP[strategy])
        if strategy == "comprehensive":
            components = ["判定参数", "正常特征记忆"]
            if "defect_prototypes" in strategy_options:
                components.append("缺陷参考特征")
            if "lightweight_calibrator" in strategy_options:
                components.append("轻量校准器")
            selected_help["resources"] = "本次实际包含：" + "、".join(components) + "；并创建新的完整目标域版本。"
        with st.container(border=True):
            st.markdown(f"**{strategy_labels[strategy]}**")
            st.write(selected_help["summary"])
            help_cols = st.columns(2)
            help_cols[0].write(f"**不更新**：{selected_help['not_update']}")
            help_cols[0].write(f"**允许样本**：{selected_help['samples']}")
            help_cols[1].write(f"**Mask 要求**：{selected_help['mask']}")
            help_cols[1].write(f"**资源变化**：{selected_help['resources']}")
            help_cols[1].write(f"**预计耗时**：{selected_help['time']}")
            st.caption("无论选择哪项，都会创建新的 Memory Bank/Profile/Config 版本，不会原地覆盖冻结资源。")
        constraint_cols = st.columns(2)
        min_recall = constraint_cols[0].slider("最低召回率", 0.5, 1.0, 0.95, 0.01, key="feedback_opt_recall")
        max_fpr = constraint_cols[1].slider("最高误报率", 0.0, 0.5, 0.10, 0.01, key="feedback_opt_fpr")

        st.markdown("#### 第四步：版本生成方式")
        publish_mode_options = ["replace_original", "update_current", "save_new_version"]
        publish_mode = st.radio(
            "候选版本通过验证后",
            publish_mode_options,
            index=1,
            format_func=lambda value: {
                "replace_original": "替换原版本",
                "update_current": "生成新版本并在验证通过后应用",
                "save_new_version": "生成新版本但暂不应用",
            }.get(value, value),
            horizontal=True,
            key="feedback_opt_publish_mode",
        )
        if publish_mode == "replace_original":
            st.warning(
                "⚠️ 安全替换：系统会先生成候选版本、完成离线验证并切换 Runtime；确认新版本可用后，才归档并删除原 Memory Bank、专属 Config 和 Profile。"
                " 任一步骤失败都会保留原版本。"
            )
            st.caption("替换成功后原版本将从普通资源列表中删除，仅保留新版本；历史检测、反馈和审计记录仍保留。")
            replace_confirm = st.checkbox(
                "我确认替换成功后原版本将从普通资源列表中删除，仅保留新版本。更新或验证失败不会影响原版本。",
                key="feedback_opt_replace_confirm",
            )
        elif publish_mode == "update_current":
            st.info("系统会先创建并验证新版本；验证通过后再切换当前运行方案。不会直接覆盖原记忆库，原版本继续保留用于回滚。")
            replace_confirm = False
        else:
            st.info("完成更新与验证后保留新版本，当前检测继续使用原版本。")
            replace_confirm = False
        version_name = st.text_input("新版本友好名称", max_chars=80, placeholder="例如：Candle 反馈优化 2026-08-06", key="feedback_opt_version_name")
        version_note = st.text_area("版本备注", max_chars=1000, placeholder="说明反馈来源、更新目的和验证范围", key="feedback_opt_note")

        payload_error = ""
        try:
            # Use exactly the same canonical body for the side-effect-free
            # compatibility preview and the later POST.  Display labels stay
            # in the UI; only stable wire codes leave this page.
            payload = api.build_feedback_optimization_payload(
                base_profile_id=profile_id,
                candidate_ids=selected_ids,
                strategy=strategy,
                objective="f1",
                update_threshold=update_threshold,
                update_normal_memory=update_normal,
                update_defect_memory=update_defect,
                update_lightweight_calibrator=update_lite,
                min_recall=min_recall,
                max_false_positive_rate=max_fpr,
                max_memory_growth=0.05,
                max_latency_growth=0.10,
                publish_mode=publish_mode,
                replace_confirm=replace_confirm,
                version_name=version_name,
                note=version_note,
            )
        except (TypeError, ValueError) as exc:
            payload = {}
            payload_error = str(exc)
            st.error(f"更新请求参数无效：{payload_error}")
        st.markdown("#### 第五步：执行更新与验证")
        compatibility = {}
        if payload:
            compatibility_response = api.feedback_optimization_compatibility(payload)
            _error(compatibility_response, "兼容性检查失败。")
            compatibility = _payload(compatibility_response, "compatibility", {})
            check_cols = st.columns(4)
            check_cols[0].metric("审核正常", compatibility.get("normal_count", 0))
            check_cols[1].metric("审核缺陷", compatibility.get("defect_count", 0))
            check_cols[2].metric("安全正常候选", compatibility.get("safe_normal_count", 0))
            check_cols[3].metric("有 Mask 缺陷", compatibility.get("masked_defect_count", 0))
            if compatibility.get("compatible"):
                st.success("✅ 记忆库、模型、特征空间、数据集、类别和更新策略兼容。")
            for warning in compatibility.get("warnings") or []:
                warning_code = str(warning.get("code") or "")
                if warning_code.startswith("STRATEGY_") or warning_code == "CANDIDATE_VALIDATION_EVIDENCE_INSUFFICIENT":
                    continue
                st.warning("⚠️ " + str(warning.get("message") or "存在注意事项。"))
                suggestions = list(warning.get("suggestions") or [])
                if suggestions:
                    st.caption("建议：" + "；".join(str(value) for value in suggestions))
                if warning.get("advanced_details"):
                    with st.expander(f"高级兼容详情 · {warning.get('code') or '注意事项'}", expanded=False):
                        details = dict(warning.get("advanced_details") or {})
                        if warning.get("code") == "FEEDBACK_METHOD_MODE_VARIANT":
                            st.write(f"样本内部模式：{', '.join(details.get('source_method_modes') or []) or '-'}")
                            st.write(f"基础版本内部模式：{details.get('base_profile_method_mode') or '-'}")
                            st.write(f"候选版本内部模式：{details.get('target_profile_method_mode') or '-'}")
                        elif warning.get("code") == "FEEDBACK_CONFIG_VARIANT":
                            st.write(f"样本配置标识：{details.get('source_config_profile_id') or '-'}")
                            st.write(f"样本配置版本：{details.get('source_config_profile_version') or '-'}")
                            st.write(f"基础配置标识：{details.get('base_config_profile_id') or '-'}")
                            st.write(f"基础配置版本：{details.get('base_config_profile_version') or '-'}")
            for error in compatibility.get("errors") or []:
                suffix = f"（样本 {_short_id(error.get('candidate_id'))}）" if error.get("candidate_id") else ""
                st.error("❌ " + str(error.get("message") or "兼容性检查失败。") + suffix)
            _render_strategy_eligibility(compatibility)
        create_disabled = (
            inference_locked
            or not profile_id
            or not selected_ids
            or bool(payload_error)
            or not version_name.strip()
            or not bool(compatibility.get("compatible"))
            or (publish_mode == "replace_original" and not replace_confirm)
        )
        create_label = "确认并开始替换" if publish_mode == "replace_original" else "开始更新并生成候选版本"
        if st.button(create_label, type="primary", disabled=create_disabled, key="feedback_opt_create"):
            with st.spinner("正在提交后台更新任务…"):
                result = api.create_feedback_optimization(payload)
            _error(result, "更新任务创建失败。")
            if result.get("success"):
                _mark_feedback_changed()
                st.success(
                    "替换任务已提交到后台；候选版本验证通过并完成预热后才会切换并清理原版本。"
                    if publish_mode == "replace_original"
                    else "更新任务已提交到后台；当前运行 profile 未改变。下方进度区会局部刷新。"
                )
                st.session_state["feedback_last_job"] = (_payload(result, "job", {}) or {}).get("job_id")
                st.session_state["feedback_center_nav_target"] = "optimization"
                st.query_params["feedback_center_tab"] = "optimization"


def _render_feedback_update_job(job_id: str, *, force_refresh: bool = False, polling: bool = False) -> None:
    """Render one job inside a fragment; polling is GET-only and idempotent."""
    cache_key = f"feedback_job_terminal_{job_id}"
    last_good_key = f"feedback_job_last_good_{job_id}"
    cached_terminal = st.session_state.get(cache_key)
    if force_refresh or not isinstance(cached_terminal, dict):
        detail_response = api.feedback_optimization_job(job_id, compact=polling)
        if detail_response.get("success"):
            detail = _payload(detail_response, "job", {})
            st.session_state[last_good_key] = dict(detail)
        else:
            error = dict(detail_response.get("error") or {})
            code = str(error.get("code") or "")
            detail = dict(st.session_state.get(last_good_key) or {})
            if code == "API_STATUS_TIMEOUT" and detail:
                if str(detail.get("status") or "") in {"queued", "running", "applying", "cancelling"}:
                    st.info("状态刷新稍慢，任务仍在运行。已保留上一次成功状态，不会重复创建任务。")
                else:
                    st.info("状态刷新稍慢，正在保留上一次成功状态。")
            elif code == "API_CONNECTION_LOST":
                st.warning("后台服务连接中断。任务区域已保留，可在连接恢复后重新读取状态。")
            else:
                st.warning(str(error.get("message") or "任务状态暂时无法刷新。"))
            if not detail:
                if st.button("重新连接状态", key=f"feedback_reconnect_job_{job_id}"):
                    st.rerun(scope="fragment")
                return
    else:
        detail = dict(cached_terminal)
    status = str(detail.get("status") or "")
    presentation = _feedback_task_presentation(detail)
    if bool(presentation.get("is_terminal")):
        if polling and not detail.get("items"):
            terminal_response = api.feedback_optimization_job(job_id, compact=False)
            if terminal_response.get("success"):
                detail = _payload(terminal_response, "job", detail)
                presentation = _feedback_task_presentation(detail)
                st.session_state[last_good_key] = dict(detail)
        st.session_state[cache_key] = dict(detail)

    progress = min(1.0, max(0.0, float(detail.get("progress") or 0.0)))
    st.progress(progress, text=f"总进度 {progress * 100:.1f}% · {detail.get('message') or '等待后台状态'}")
    time_cols = st.columns(4)
    time_cols[0].metric("已用时间", _duration_text(detail.get("elapsed_seconds")))
    time_cols[1].metric(
        "当前阶段",
        _stage_label(detail.get("current_stage_id"), detail.get("stage") or presentation.get("task_status_display") or "等待中"),
    )
    time_cols[2].metric("本阶段耗时", _duration_text(detail.get("current_stage_elapsed_seconds")))
    time_cols[3].metric("预计剩余", _duration_text(detail.get("eta_seconds"), estimate=True))
    activity = str(detail.get("activity_status") or "")
    if activity == "long_running":
        st.info("当前阶段计算时间较长，但任务心跳正常，仍在执行。")
    elif activity == "possibly_interrupted":
        st.error("任务心跳已停止，后台任务可能中断；可刷新状态或在任务转为中断后重新运行。")

    st.markdown("##### 更新执行阶段")
    for index, stage in enumerate(detail.get("stages") or [], start=1):
        stage_status = str(stage.get("status") or "pending")
        name = _stage_label(stage.get("stage_id"), stage.get("stage_name"))
        message = str(stage.get("message") or "")
        elapsed = _duration_text(stage.get("elapsed_seconds"))
        if stage_status == "running":
            with st.spinner(f"{index}. {name} · {message}"):
                completed_items = stage.get("completed_items")
                total_items = stage.get("total_items")
                item_progress = ""
                if total_items is not None:
                    item_progress = f" · 已处理 {int(completed_items or 0)}/{int(total_items or 0)}"
                current_item = str(stage.get("current_item") or "")
                if current_item:
                    item_progress += f" · 当前：{current_item}"
                st.caption(
                    f"本阶段已用 {elapsed} · 阶段进度 {float(stage.get('progress') or 0.0) * 100:.1f}%{item_progress}"
                )
        else:
            icon = {
                "completed": "✅",
                "warning": "⚠️",
                "skipped": "—",
                "failed": "❌",
                "pending": "○",
            }.get(stage_status, "○")
            suffix = f" · {elapsed}" if stage_status in {"completed", "warning", "failed"} else ""
            st.markdown(f"{icon} **{index}. {name}**{suffix}  ")
            st.caption(message or ("等待执行" if stage_status == "pending" else "当前更新内容无需执行"))

    item_summary = dict(detail.get("item_summary") or {})
    if item_summary:
        st.markdown("##### 反馈样本处理结果")
        result_cols = st.columns(5)
        result_cols[0].metric("反馈样本总数", int(item_summary.get("total") or 0))
        result_cols[1].metric("成功用于更新", int(item_summary.get("completed") or 0))
        result_cols[2].metric("本次跳过", int(item_summary.get("skipped") or 0))
        result_cols[3].metric("处理失败", int(item_summary.get("failed") or 0))
        result_cols[4].metric(
            "取消 / 中断",
            int(item_summary.get("cancelled") or 0) + int(item_summary.get("interrupted") or 0),
        )
        if int(item_summary.get("legacy_state_unknown") or 0):
            st.warning(
                f"有 {int(item_summary.get('legacy_state_unknown') or 0)} 条历史明细缺少可靠处理证据，"
                "已标记为‘历史任务状态信息不完整’，未伪造为成功。"
            )
        rows = list(detail.get("items") or [])
        if rows:
            with st.expander("查看逐样本处理结果", expanded=False):
                st.dataframe(
                    [
                        {
                            "序号": index,
                            "样本": str(row.get("sample_name") or "未命名样本"),
                            "人工标签": LABEL_NAMES.get(str(row.get("corrected_label") or "unknown"), "待确认"),
                            "Mask": "有" if bool(row.get("has_mask")) else "无",
                            "处理结果": str(row.get("status_display") or "状态未知"),
                            "更新内容": "、".join(str(value) for value in list(row.get("components") or [])) or "—",
                            "结果说明": str(row.get("message") or "—"),
                        }
                        for index, row in enumerate(rows, start=1)
                    ],
                    width="stretch",
                    hide_index=True,
                )

    timings = dict(detail.get("timings") or {})
    if timings:
        with st.expander("查看真实阶段计时", expanded=False):
            stages_by_id = {
                str(stage.get("stage_id") or ""): dict(stage)
                for stage in list(detail.get("stages") or [])
            }
            timing_stage_aliases = {
                "feedback_query_and_compatibility": "feedback_validation",
                "job_preflight": "feedback_validation",
                "original_image_read": "feature_extraction",
                "mask_read": "feature_extraction",
                "feedback_feature_extraction": "feature_extraction",
                "base_memory_sha256_before": "profile_compatibility",
                "base_memory_sha256_after": "validation",
                "offline_validation": "validation",
                "profile_registration": "complete",
                "runtime_refresh": "complete",
            }
            timing_rows = []
            for name, timing in timings.items():
                timing = dict(timing or {})
                related_stage_id = timing_stage_aliases.get(str(name), str(name))
                related_stage = stages_by_id.get(related_stage_id, {})
                stage_status = str(related_stage.get("status") or "completed")
                timing_rows.append({
                    "阶段": _stage_label(name),
                    "状态": STAGE_STATUS_LABELS.get(stage_status, "状态未知"),
                    "耗时": _seconds_text(timing.get("seconds")),
                    "说明": _timing_description(str(name), timing),
                })
            st.dataframe(timing_rows, width="stretch", hide_index=True)
        with st.expander("技术详情 · 后台阶段标识", expanded=False):
            st.dataframe(
                [
                    {
                        "内部阶段标识": str(name),
                        "中文阶段": _stage_label(name),
                        "耗时（秒）": round(float((timing or {}).get("seconds") or 0.0), 6),
                    }
                    for name, timing in timings.items()
                ],
                width="stretch",
                hide_index=True,
            )

    metrics_before = detail.get("metrics_before") or {}
    metrics_after = detail.get("metrics_after") or {}
    if metrics_before or metrics_after:
        st.warning("以下为已审核反馈校准集上的准入指标，不是独立测试精度，也不会使用正式测试数据更新资源。")
        st.dataframe([
            {"指标": "阈值", "原方案": _metric_display(metrics_before.get("threshold", metrics_before.get("threshold_before"))), "候选方案": _metric_display(metrics_after.get("threshold", metrics_after.get("threshold_after")))},
            {"指标": "精确率", "原方案": _metric_display(metrics_before.get("precision")), "候选方案": _metric_display(metrics_after.get("precision"))},
            {"指标": "召回率", "原方案": _metric_display(metrics_before.get("recall")), "候选方案": _metric_display(metrics_after.get("recall"))},
            {"指标": "误报率", "原方案": _metric_display(metrics_before.get("false_positive_rate")), "候选方案": _metric_display(metrics_after.get("false_positive_rate"))},
            {"指标": "漏检率", "原方案": _metric_display(metrics_before.get("false_negative_rate")), "候选方案": _metric_display(metrics_after.get("false_negative_rate"))},
            {"指标": "F1", "原方案": _metric_display(metrics_before.get("f1")), "候选方案": _metric_display(metrics_after.get("f1"))},
        ], width="stretch", hide_index=True)

    validation = detail.get("validation") or {}
    complexity_before = detail.get("complexity_before") or {}
    complexity_after = detail.get("complexity_after") or {}
    if validation:
        validation_cols = st.columns(4)
        latency = validation.get("latency") or {}
        validation_cols[0].metric("特征与检索 P50", f"{latency.get('p50_ms')} ms" if latency.get("p50_ms") is not None else "未测得")
        validation_cols[1].metric("特征与检索 P95", f"{latency.get('p95_ms')} ms" if latency.get("p95_ms") is not None else "未测得")
        validation_cols[2].metric("记忆条目", complexity_after.get("memory_patch_count") if complexity_after.get("memory_patch_count") is not None else "未记录")
        validation_cols[3].metric("ANN 查询次数", complexity_after.get("ann_query_count") if complexity_after.get("ann_query_count") is not None else "未记录")
        hash_state = "✅ 原版本哈希未改变" if validation.get("base_memory_unchanged") else "❌ 原版本哈希发生变化"
        st.caption(
            f"{hash_state} · 基础记忆库 {_short_id(validation.get('base_memory_sha256_before'))} · "
            f"候选记忆库 {_short_id(validation.get('candidate_memory_sha256'))} · "
            f"候选资源大小 {complexity_after.get('memory_bank_bytes', '未记录')} 字节"
        )
    update_result = _render_update_result(detail) if bool(_feedback_task_presentation(detail).get("is_terminal")) else {}

    def refresh_fragment() -> None:
        st.session_state.pop(cache_key, None)
        st.rerun(scope="fragment")

    action_cols = st.columns([1, 1, 3], vertical_alignment="center")
    job_action_scope = f"feedback-job-action::{job_id}"
    with action_cols[2]:
        job_action_status = st.empty()
    render_action_feedback(job_action_scope, job_action_status)
    if action_cols[0].button("刷新任务状态", key=f"feedback_refresh_job_{job_id}"):
        refresh_fragment()
    if bool(presentation.get("can_cancel")) or status == "cancelling":
        if action_cols[1].button("取消后台任务", disabled=status == "cancelling", key=f"feedback_cancel_{job_id}"):
            update_action_feedback(job_action_scope, "info", "正在提交取消请求...", job_action_status)
            result = api.cancel_feedback_optimization(job_id)
            _error(result, "取消任务失败。")
            if not result.get("success"):
                update_action_feedback(
                    job_action_scope,
                    "error",
                    friendly_error_message(result.get("error"), "取消任务失败。"),
                    job_action_status,
                )
            refresh_fragment()
    elif bool(presentation.get("can_retry")):
        if action_cols[1].button("重新运行", type="primary", key=f"feedback_rerun_{job_id}"):
            update_action_feedback(job_action_scope, "info", "正在创建新的后台任务...", job_action_status)
            result = api.rerun_feedback_optimization(job_id)
            _error(result, "反馈更新重新运行失败。")
            if result.get("success"):
                created = _payload(result, "job", {})
                st.session_state["feedback_last_job"] = created.get("job_id")
                st.session_state.pop(cache_key, None)
                update_action_feedback(
                    job_action_scope,
                    "success",
                    "已创建新的后台任务；旧失败/取消任务仍保留审计记录。",
                    job_action_status,
                )
            else:
                update_action_feedback(
                    job_action_scope,
                    "error",
                    friendly_error_message(result.get("error"), "反馈更新重新运行失败。"),
                    job_action_status,
                )

    constraints = detail.get("constraints") or {}
    publish_mode = str(constraints.get("publish_mode") or "save_new_version")
    replacement_cleanup = dict(detail.get("replacement_cleanup") or {})
    if status in {"ready_to_apply", "frozen"}:
        profile_response = api.feedback_candidate_profile(job_id)
        _error(profile_response, "候选版本信息读取失败。")
        candidate_profile = _payload(profile_response, "candidate_profile", {})
        if candidate_profile:
            base_profile = dict(candidate_profile.get("base_profile") or {})
            base_name = str(
                base_profile.get("friendly_name")
                or selected_resource.get("friendly_name")
                or "基础目标域记忆库"
            )
            base_version = int(base_profile.get("version") or selected_resource.get("current_version") or 1)
            candidate_name = str(
                candidate_profile.get("friendly_name")
                or candidate_profile.get("version_name")
                or f"{base_name} · 反馈更新版本"
            )
            config_count = int(candidate_profile.get("config_count") or 0)
            default_config_name = str(candidate_profile.get("default_config_name") or "当前绑定配置")
            inherited_config_names = [
                str(name) for name in candidate_profile.get("inherited_config_names") or [] if str(name).strip()
            ]
            integrity_passed = bool(candidate_profile.get("integrity_passed")) or str(validation.get("status") or "") == "passed"
            item_rows = list(detail.get("items") or [])
            corrected_normal = sum(str(item.get("corrected_label") or "") == "normal" for item in item_rows)
            corrected_defect = sum(str(item.get("corrected_label") or "") == "defect" for item in item_rows)
            mask_count = sum(bool(item.get("has_mask")) for item in item_rows)
            used_count = int(item_summary.get("completed") or 0)
            skipped_count = int(item_summary.get("skipped") or 0)

            result_top = st.columns(3)
            result_top[0].metric("基础记忆库", f"{base_name} · v{base_version}")
            result_top[0].caption(
                f"当前配置：{base_profile.get('default_config_name') or '当前绑定配置'}"
            )
            result_top[1].metric("新目标域版本", candidate_name)
            result_top[1].caption(f"默认配置：{default_config_name}")
            result_top[2].metric("纠错样本", f"共 {len(item_rows) or len(detail.get('candidate_ids') or [])} 条")
            result_bottom = st.columns(3)
            result_bottom[0].metric("配置数量", f"{config_count} 个", help=f"当前默认配置：{default_config_name}")
            if inherited_config_names:
                result_bottom[0].caption(f"继承配置：{'、'.join(inherited_config_names)}")
            result_bottom[1].metric("完整性", "✅ 校验通过" if integrity_passed else "❌ 校验未通过")
            result_bottom[2].metric("版本状态", str(presentation.get("task_status_display") or "状态未知"))

            with st.container(border=True):
                st.markdown("**纠错样本构成**")
                sample_cols = st.columns(5)
                sample_cols[0].metric("正常纠正", corrected_normal)
                sample_cols[1].metric("缺陷纠正", corrected_defect)
                sample_cols[2].metric("带 Mask", mask_count)
                sample_cols[3].metric("真正用于更新", used_count)
                sample_cols[4].metric("本次跳过", skipped_count)

            validation_updates = dict(validation.get("updates") or {})
            threshold_before = metrics_before.get("threshold", metrics_before.get("threshold_before"))
            threshold_after = metrics_after.get("threshold", metrics_after.get("threshold_after"))
            threshold_changed = (
                bool(detail.get("update_threshold"))
                and threshold_before is not None
                and threshold_after is not None
                and abs(float(threshold_after) - float(threshold_before)) > 1e-12
            )
            normal_memory_changed = bool((validation_updates.get("normal_memory") or {}).get("updated"))
            defect_reference_changed = bool((validation_updates.get("defect_memory") or {}).get("updated"))
            lightweight_changed = bool((validation_updates.get("lightweight_calibrator") or {}).get("updated"))
            st.markdown("##### 版本变化")
            st.dataframe(
                [
                    {"资源": "正常 Memory Bank", "结果": "已更新" if normal_memory_changed else "保持不变", "说明": "仅使用审核确认正常且符合安全规则的样本。"},
                    {"资源": "判定参数", "结果": "已更新" if threshold_changed else "保持不变", "说明": "根据已审核正常/缺陷证据重新估计判定边界。"},
                    {"资源": "缺陷参考特征", "结果": "已更新" if defect_reference_changed else "保持不变", "说明": "仅在当前方法支持且缺陷区域证据有效时更新。"},
                    {"资源": "轻量校准器", "结果": "已更新" if lightweight_changed else "保持不变", "说明": "不会重新训练骨干模型。"},
                    {"资源": "记忆库内部配置", "结果": f"已生成 {config_count} 个新配置", "说明": f"当前默认：{default_config_name}；其余配置保持来源参数。"},
                ],
                width="stretch",
                hide_index=True,
            )
            with st.expander("技术详情 · 版本标识", expanded=False):
                st.write(f"基础 Profile ID：{detail.get('base_profile_id') or '-'}")
                st.write(f"候选 Profile ID：{candidate_profile.get('memory_bank_id') or '-'}")
                st.write(f"候选 Config ID：{candidate_profile.get('config_profile_id') or '-'}")
                st.write(f"后台任务状态：{status or '-'}")
        if status == "ready_to_apply" and publish_mode == "save_new_version":
            freeze_confirm = st.checkbox("我确认保留此候选新版本但暂不切换当前运行方案。", key=f"feedback_freeze_confirm_{job_id}")
            freeze_scope = f"feedback-freeze::{job_id}"
            freeze_clicked, freeze_status = render_action_button_with_feedback(
                "冻结并保留新版本",
                scope=freeze_scope,
                key=f"feedback_freeze_{job_id}",
                button_type="primary",
                disabled=not freeze_confirm,
            )
            if freeze_clicked:
                update_action_feedback(freeze_scope, "info", "正在冻结并注册候选新版本...", freeze_status)
                result = api.freeze_feedback_optimization(job_id, confirm=True)
                _error(result, "候选版本冻结失败。")
                if result.get("success"):
                    _mark_feedback_changed(resources_changed=True)
                    update_action_feedback(freeze_scope, "success", "候选新版本已冻结保存，当前运行方案未改变。", freeze_status)
                    refresh_fragment()
                else:
                    update_action_feedback(
                        freeze_scope,
                        "error",
                        friendly_error_message(result.get("error"), "候选版本冻结失败。"),
                        freeze_status,
                    )
        elif status == "frozen" and publish_mode == "save_new_version":
            pass
        if publish_mode == "replace_original":
            st.info("该任务会在验证通过后自动完成安全替换；不会原地修改旧文件。旧版本成功替换后不再出现在普通资源列表中。")
        else:
            st.info("应用会原子切换到新版本；不会覆盖原 Memory Bank，原版本保留用于回滚。")
            apply_confirm = st.checkbox("我确认应用此新版本并保留原版本。", key=f"feedback_apply_confirm_{job_id}")
            apply_scope = f"feedback-apply::{job_id}"
            apply_clicked, apply_status = render_action_button_with_feedback(
                "应用新版本",
                scope=apply_scope,
                key=f"feedback_apply_{job_id}",
                button_type="primary",
                disabled=not apply_confirm,
            )
            if apply_clicked:
                update_action_feedback(apply_scope, "info", "正在原子切换到反馈更新版本...", apply_status)
                result = api.apply_feedback_optimization(job_id, confirm=True)
                _error(result, "候选版本应用失败。")
                if result.get("success"):
                    _mark_feedback_changed(resources_changed=True)
                    update_action_feedback(apply_scope, "success", "新版本已切换为当前运行方案。", apply_status)
                    refresh_fragment()
                else:
                    update_action_feedback(
                        apply_scope,
                        "error",
                        friendly_error_message(result.get("error"), "候选版本应用失败。"),
                        apply_status,
                    )
    elif status in {"failed", "interrupted", "validation_failed", "cancelled"}:
        if str((update_result or {}).get("outcome_code") or "") == "ELIGIBILITY_FAILED":
            st.info("请返回上方选择样本，补足当前策略的最低准入条件后重新更新。")
        elif str((update_result or {}).get("outcome_code") or "") == "VALIDATION_FAILED":
            st.info("建议调整样本组成或策略后重新更新；系统不会自动降低 Recall、误报率或复杂度门槛。")
        elif str((update_result or {}).get("outcome_code") or "") == "EXECUTION_FAILED":
            st.info("可检查失败阶段涉及的原始样本或后台资源后重新运行。")
        if detail.get("error_code"):
            with st.expander("技术详情 · 失败标识", expanded=False):
                st.write(f"错误代码：{detail.get('error_code')}")
                if detail.get("error_message"):
                    st.code(str(detail.get("error_message")), language="text")
    if status == "applied":
        if publish_mode == "replace_original":
            if replacement_cleanup.get("cleanup_status") == "archived_pending_delete":
                st.warning("✅ 新版本已切换；原版本已归档，物理清理尚待维护任务完成。当前检测已使用新版本。")
            else:
                st.success("✅ 新版本已通过验证并安全替换原版本；原版本已从普通资源列表移除。")
        if publish_mode == "replace_original":
            st.caption("替换模式会物理移除原版本，因此不能通过本任务回滚到已删除资源；如需回退，请在运行配置中选择其他仍保留的冻结版本。")
        else:
            rollback_confirm = st.checkbox("我确认将当前运行方案切回原始 profile。", key=f"feedback_rollback_confirm_{job_id}")
            rollback_scope = f"feedback-rollback::{job_id}"
            rollback_clicked, rollback_status = render_action_button_with_feedback(
                "回滚到原始版本",
                scope=rollback_scope,
                key=f"feedback_rollback_{job_id}",
                disabled=not rollback_confirm,
            )
            if rollback_clicked:
                update_action_feedback(rollback_scope, "info", "正在切回原始冻结版本...", rollback_status)
                result = api.rollback_feedback_optimization(job_id, confirm=True)
                _error(result, "回滚失败。")
                if result.get("success"):
                    _mark_feedback_changed(resources_changed=True)
                    update_action_feedback(rollback_scope, "success", "已回滚到原始冻结 profile。", rollback_status)
                    refresh_fragment()
                else:
                    update_action_feedback(
                        rollback_scope,
                        "error",
                        friendly_error_message(result.get("error"), "回滚失败。"),
                        rollback_status,
                    )

if active_tab == "optimization":
    st.divider()
    st.markdown("#### 第六步：结果与应用")
    jobs_response = api.feedback_optimization_jobs()
    _error(jobs_response, "优化任务读取失败。")
    jobs = [
        item for item in _payload(jobs_response, "jobs", [])
        if str(item.get("base_profile_id") or item.get("memory_bank_id") or "") == selected_memory_bank_id
    ]
    if not jobs:
        st.info("暂无优化任务。")
    else:
        job_ids = [str(item.get("job_id")) for item in jobs]
        jobs_by_id = {str(item.get("job_id") or ""): item for item in jobs}

        def job_option_label(value: str) -> str:
            job = jobs_by_id.get(str(value), {})
            state = _feedback_task_presentation(job)
            created_text = format_beijing_datetime(job.get("created_at"), fallback="时间未记录")
            memory_name = str(selected_resource.get("friendly_name") or _short_id(selected_memory_bank_id, "目标域"))
            result = dict(job.get("result") or {})
            result_text = str(result.get("title") or state.get("task_status_display") or "状态未知")
            summary = str(result.get("summary") or "").replace("\n", " ").strip()
            if len(summary) > 42:
                summary = summary[:39] + "…"
            return f"{memory_name} · {created_text} · {result_text}" + (f" · {summary}" if summary else "")

        selected_job_id = st.selectbox(
            "选择优化任务",
            job_ids,
            index=job_ids.index(st.session_state.get("feedback_last_job")) if st.session_state.get("feedback_last_job") in job_ids else 0,
            format_func=job_option_label,
            key="feedback_job_select",
        )
        summary_status = str(next((item.get("status") for item in jobs if str(item.get("job_id")) == selected_job_id), ""))
        if summary_status in {"queued", "running", "cancelling"}:
            try:
                st.fragment(run_every=2.0)(_render_feedback_update_job)(selected_job_id, polling=True)
            except AttributeError:  # pragma: no cover - old Streamlit fallback
                _render_feedback_update_job(selected_job_id)
        else:
            st.fragment(_render_feedback_update_job)(selected_job_id)

if active_tab == "statistics":
    history_heading = st.columns([1, 2], vertical_alignment="center")
    history_heading[0].subheader("反馈记录管理")
    history_feedback_scope = f"feedback-history-delete::{selected_memory_bank_id}"
    with history_heading[1]:
        history_action_status = st.empty()
    render_action_feedback(history_feedback_scope, history_action_status)
    st.caption("检索、查看和删除反馈记录。删除反馈不会删除原始检测结果、输入图片、视频或已经生成的目标域版本。")
    stats_response = cached_feedback_statistics(selected_memory_bank_id, current_feedback_revision)
    _error(stats_response, "反馈统计读取失败。")
    stats = _payload(stats_response, "statistics", {})
    if stats:
        total = _count(stats.get("total"))
        cols = st.columns(5)
        cols[0].metric("反馈总数", total)
        cols[1].metric("待审核", _count(stats.get("pending_count")))
        cols[2].metric("已通过", _count(stats.get("approved_count")))
        cols[3].metric("已撤销", _count(stats.get("withdrawn_count")))
        cols[4].metric("已用于更新", _count(stats.get("used_count")))
        if total == 0:
            st.caption("暂无反馈数据。")
    else:
        st.info("暂无反馈记录。检测页面提交反馈后会显示在这里。")

    st.markdown("#### 反馈历史检索与记录管理")
    st.caption("筛选条件只作用于当前反馈记录，不会修改检测历史或运行方案。")
    with st.form("feedback_history_filter_form", border=False):
        search_cols = st.columns([1.7, 1.0, 1.0, 1.0, 1.0])
        history_keyword = search_cols[0].text_input("关键词", placeholder="样本名称、缺陷类别或用户备注", key="feedback_history_keyword")
        history_mode = search_cols[1].selectbox("检测来源", ["all", "single", "batch", "video"], format_func=lambda value: "全部" if value == "all" else MODE_NAMES.get(value, value), key="feedback_history_mode")
        history_status = search_cols[2].selectbox(
            "处理状态", ["all", "pending", "approved", "withdrawn", "used"],
            format_func=lambda value: {"all": "全部", "pending": "待审核", "approved": "已通过", "withdrawn": "已撤销", "used": "已用于更新"}[value],
            key="feedback_history_status",
        )
        history_label = search_cols[3].selectbox("人工标签", ["all", "normal", "defect", "uncertain"], format_func=lambda value: {"all": "全部", "normal": "正常", "defect": "缺陷", "uncertain": "不确定"}[value], key="feedback_history_label")
        history_mask = search_cols[4].selectbox("Mask", ["all", "with", "without"], format_func=lambda value: {"all": "全部", "with": "包含 Mask", "without": "无 Mask"}[value], key="feedback_history_mask")
        more_cols = st.columns([1.0, 1.0, 1.0, 1.4])
        history_time = more_cols[0].selectbox("检测时间范围", ["全部", "今天", "最近 7 天", "最近 30 天", "最近 90 天"], key="feedback_history_time")
        history_feedback_time = more_cols[1].selectbox("反馈时间范围", ["全部", "今天", "最近 7 天", "最近 30 天", "最近 90 天"], key="feedback_history_feedback_time")
        page_size = more_cols[2].selectbox("每页数量", [10, 20, 50, 100], index=1, key="feedback_history_page_size")
        with more_cols[3]:
            history_search = st.form_submit_button("应用检索", type="primary", icon=":material/search:", width="stretch")
    if history_search:
        st.session_state["feedback_history_page"] = 1
    status_groups = {
        "pending": {"collected", "legacy_unresolved", "pending_review", "in_review", "conflict", "record_only"},
        "approved": {"accepted"},
        "withdrawn": {"withdrawn", "rejected", "archived", "cancelled", "invalid_source"},
        "used": {"queued", "running", "optimizing", "ready_to_apply", "consumed", "frozen", "applied", "rolled_back", "completed"},
    }
    status_query = ""
    if history_status != "all":
        status_query = ",".join(sorted(status_groups[history_status]))
    detection_since = ""
    feedback_since = ""
    if history_time != "全部":
        days = {"今天": 1, "最近 7 天": 7, "最近 30 天": 30, "最近 90 天": 90}[history_time]
        detection_since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    if history_feedback_time != "全部":
        days = {"今天": 1, "最近 7 天": 7, "最近 30 天": 30, "最近 90 天": 90}[history_feedback_time]
        feedback_since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    history_page_hint = max(1, int(st.session_state.get("feedback_history_page") or 1))
    records_response = cached_feedback_candidates(
        current_feedback_revision,
        status_query,
        "" if history_mode == "all" else history_mode,
        selected_memory_bank_id,
        selected_source_profile_id,
        selected_target_domain_id,
        int(page_size),
        page=history_page_hint,
        page_size=int(page_size),
        search=history_keyword.strip(),
        label="unknown" if history_label == "uncertain" else ("" if history_label == "all" else history_label),
        mask="" if history_mask == "all" else history_mask,
        detection_since=detection_since,
        feedback_since=feedback_since,
    )
    _error(records_response, "反馈历史读取失败。")
    page_rows = list(_payload(records_response, "candidates", []))
    total_history = int(_payload(records_response, "total", 0) or 0)
    page_count = max(1, int(_payload(records_response, "total_pages", 1) or 1))
    if history_page_hint > page_count:
        history_page_hint = page_count
        st.session_state["feedback_history_page"] = page_count
    history_page = st.number_input("当前页", 1, page_count, key="feedback_history_page")
    start = (int(history_page) - 1) * int(page_size)
    page_ids = {str(row.get("candidate_id") or "") for row in page_rows if str(row.get("status") or "") != "deleted"}
    history_selected = _history_selected_feedback_ids()
    history_controls = st.columns([1, 1, 1, 2])
    history_controls[0].button(
        "全选当前页", disabled=not page_ids, key="feedback_history_select_page",
        on_click=_set_feedback_history_page_selection,
        args=(tuple(sorted(page_ids)), True),
    )
    history_controls[1].button(
        "取消当前页", disabled=not (history_selected & page_ids), key="feedback_history_clear_page",
        on_click=_set_feedback_history_page_selection,
        args=(tuple(sorted(page_ids)), False),
    )
    history_controls[2].button(
        "清除选择", disabled=not history_selected, key="feedback_history_clear_all",
        on_click=_clear_feedback_history_selection,
        args=(tuple(sorted(page_ids)),),
    )
    history_controls[3].caption(f"共 {total_history} 条；跨页已选择 {len(history_selected)} 条。")

    correction_names = {"normal": "正常", "defect": "缺陷", "keep": "不确定", "unknown": "不确定"}
    prediction_names = {"normal": "正常", "anomaly": "异常", "defect": "异常", "unknown": "未确认"}
    table_rows: list[dict[str, Any]] = []
    for sequence, row in enumerate(page_rows, start=start + 1):
        candidate_id = str(row.get("candidate_id") or "")
        table_rows.append({
            "选择": candidate_id in history_selected,
            "序号": sequence,
            "检测时间": _display_detection_time(row),
            "反馈时间": _display_time(row.get("feedback_submitted_at")),
            "样本名称": str(row.get("sample_name") or "检测样本"),
            "检测类型": MODE_NAMES.get(str(row.get("mode") or ""), "检测"),
            "原始结果": prediction_names.get(str(row.get("original_prediction") or "").lower(), "未确认"),
            "人工修正": correction_names.get(str(row.get("corrected_label") or "keep"), "不确定"),
            "缺陷类别": str(row.get("corrected_defect_type") or "-") if str(row.get("corrected_label") or "") == "defect" else "-",
            "Mask状态": "有 Mask" if row.get("has_mask") else "无 Mask",
            "来源目标域": _short_id(row.get("source_profile_id") or row.get("source_memory_bank_id"), "未关联"),
            "状态": STATUS_LABELS.get(str(row.get("status") or ""), "待处理"),
            "操作": "查看详情",
        })
    if table_rows:
        table_revision = int(st.session_state.get("feedback_history_table_revision", 0))
        edited_table = st.data_editor(
            table_rows,
            key=f"feedback_history_table_{int(history_page)}_{table_revision}",
            width="stretch",
            hide_index=True,
            disabled=[column for column in table_rows[0] if column != "选择"],
            column_config={
                "选择": st.column_config.CheckboxColumn("选择", width="small"),
                "序号": st.column_config.NumberColumn("序号", width="small"),
                "检测时间": st.column_config.TextColumn("检测时间", width="medium"),
                "反馈时间": st.column_config.TextColumn("反馈时间", width="medium"),
                "样本名称": st.column_config.TextColumn("样本名称", width="large"),
                "操作": st.column_config.TextColumn("操作", width="small"),
            },
            height=min(720, 38 * len(table_rows) + 40),
        )
        edited_records = edited_table.to_dict("records") if hasattr(edited_table, "to_dict") else list(edited_table)
        history_selected.difference_update(page_ids)
        for source_row, edited_row in zip(page_rows, edited_records):
            if bool(edited_row.get("选择")):
                history_selected.add(str(source_row.get("candidate_id") or ""))
        store_selected_ids(
            st.session_state,
            FEEDBACK_HISTORY_SELECTION_KEY,
            history_selected,
            persistence_key=_feedback_selection_persistence_key(history=True),
        )
    else:
        st.info("当前筛选条件下没有反馈记录。")

    detail_options = [str(row.get("candidate_id") or "") for row in page_rows if row.get("candidate_id")]
    detail_by_id = {str(row.get("candidate_id") or ""): row for row in page_rows}
    if detail_options:
        detail_choice = st.selectbox(
            "查看详情",
            [""] + detail_options,
            format_func=lambda value: "请选择当前页记录" if not value else f"{str(detail_by_id[value].get('sample_name') or '检测样本')} · {_display_detection_time(detail_by_id[value])}",
            key="feedback_history_detail_choice",
        )
        if detail_choice:
            detail = detail_by_id[detail_choice]
            with st.container(border=True):
                st.markdown(f"**{str(detail.get('sample_name') or '检测样本')}**")
                detail_cols = st.columns(4)
                detail_cols[0].caption("数据集 / 类别")
                detail_cols[0].write(f"{detail.get('dataset') or '-'} / {detail.get('category') or '-'}")
                detail_cols[1].caption("异常分数 / 阈值")
                detail_cols[1].write(f"{detail.get('original_score') if detail.get('original_score') is not None else '-'} / {detail.get('threshold') if detail.get('threshold') is not None else '-'}")
                detail_cols[2].caption("缺陷类别")
                detail_cols[2].write(str(detail.get("corrected_defect_type") or "未指定"))
                detail_cols[3].caption("Mask")
                detail_cols[3].write("已提供" if detail.get("has_mask") else "未提供")
                if str(detail.get("mode") or "") == "video":
                    st.caption(f"视频帧：{detail.get('frame_index') if detail.get('frame_index') is not None else '-'} · 视频时间戳：{float(detail.get('timestamp_ms') or 0.0) / 1000.0:.3f} 秒")
                st.caption(f"用户备注：{str(detail.get('note') or '无')}\n\n审核备注：{str(detail.get('review_note') or '无')}")
    if history_selected:
        preview_signature = (current_feedback_revision, tuple(sorted(history_selected)))
        if st.session_state.get("feedback_delete_preview_signature") != preview_signature:
            st.session_state.pop("feedback_delete_preview", None)
        if st.button("生成删除影响预览", icon=":material/preview:", key="feedback_history_preview"):
            with st.spinner("正在核对所选反馈的附件与版本引用..."):
                preview_response = api.feedback_candidates_delete_preview(sorted(history_selected))
            _error(preview_response, "删除影响预览失败。")
            if preview_response.get("success"):
                st.session_state["feedback_delete_preview"] = _payload(preview_response, "preview", {})
                st.session_state["feedback_delete_preview_signature"] = preview_signature
                st.session_state["feedback_history_delete_ack"] = False
        delete_preview = dict(st.session_state.get("feedback_delete_preview") or {})
        if not delete_preview:
            st.caption("删除影响只在点击预览时计算；勾选记录和调整筛选不会扫描附件目录。")
        else:
            result = render_delete_preview_confirmation(
                preview=delete_preview,
                title="反馈记录删除预览",
                action_label="删除所选反馈记录",
                object_label="反馈",
                acknowledgement_key="feedback_history_delete_ack",
                button_key="feedback_history_delete",
                execute_callback=lambda payload: api.delete_feedback_candidates(
                    sorted(history_selected),
                    acknowledged=True,
                    preview_token=str(payload.get("preview_token") or ""),
                ),
                extra_summary=[("原始检测结果", "保留"), ("原始图片/视频", "保留"), ("已生成版本", "保留")],
            )
            if result is not None:
                _error(result, "反馈记录删除失败。")
                if result.get("success"):
                    deleted_ids = {
                        str(item.get("candidate_id") or "")
                        for item in list(result.get("results") or [])
                        if item.get("success") and item.get("candidate_id")
                    }
                    remove_selected_ids(
                        st.session_state,
                        state_key=FEEDBACK_HISTORY_SELECTION_KEY,
                        removed_ids=sorted(deleted_ids),
                        scopes=("feedback_history",),
                        persistence_key=_feedback_selection_persistence_key(history=True),
                    )
                    st.session_state["feedback_history_table_revision"] = int(st.session_state.get("feedback_history_table_revision", 0)) + 1
                    st.session_state.pop("feedback_delete_preview", None)
                    st.session_state.pop("feedback_delete_preview_signature", None)
                    _mark_feedback_changed()
                    set_action_feedback(
                        history_feedback_scope,
                        "success",
                        f"已删除 {int(result.get('success_count') or 0)} 条反馈记录；已使用记录仅保留最小版本审计摘要。",
                    )
                    st.rerun()
    audit_response = api.feedback_audit(memory_bank_id=selected_memory_bank_id, limit=50)
    _error(audit_response, "审计记录读取失败。")
    audit = _payload(audit_response, "audit", [])
    if audit:
        action_labels = {
            "candidate_created": "创建反馈候选", "feedback_updated": "更新反馈",
            "candidate_deduplicated": "合并重复反馈", "video_frame_perceptual_deduplicated": "合并相似视频帧",
            "review_accept": "审核通过", "review_reject": "审核拒绝", "review_update": "更新人工标签",
            "annotation_saved": "保存缺陷掩码", "optimization_created": "创建优化任务",
            "optimization_ready_to_apply": "候选版本通过验证", "optimization_frozen": "冻结候选版本", "optimization_applied": "应用候选版本",
            "optimization_rolled_back": "回滚运行方案",
            "feedback_physically_deleted": "删除未使用反馈", "feedback_deleted_audit_retained": "删除反馈并保留版本审计",
        }
        with st.expander("最近操作审计", expanded=False):
            st.dataframe([{"操作": action_labels.get(str(row.get("action") or ""), "反馈处理"), "时间": _display_time(row.get("created_at")), "操作者": "操作员" if row.get("actor") in {"operator", "local_operator"} else "系统"} for row in audit], width="stretch", hide_index=True)
