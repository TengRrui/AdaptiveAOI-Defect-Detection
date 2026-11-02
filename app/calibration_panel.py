"""Shared operator panel for the unified calibration-and-validation flow.

The algorithm remains in the System backend.  This module only composes the
existing calibration API and presents its immutable candidate result inside
the target-domain workflow; it is deliberately not a second calibration
implementation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from friendly_display import friendly_error_message, render_technical_details
from service.calibration_service import run_calibration
from ui_common import format_value


CALIBRATION_PANEL_TABS = (
    ("recommended", "目标域样本校准（推荐）"),
    ("compatibility", "兼容方向校准"),
)
CALIBRATION_PANEL_TAB_LABELS = dict(CALIBRATION_PANEL_TABS)


def _calibration_panel_active_tab() -> str:
    """Keep the calibration sub-section stable across service/rerun actions."""
    key = "domain_calibration_active_tab"
    valid = {value for value, _ in CALIBRATION_PANEL_TABS}
    requested = str(st.query_params.get("domain_calibration_tab") or "").strip()
    if requested in valid and key not in st.session_state:
        st.session_state[key] = requested
    st.session_state.setdefault(key, "recommended")
    selected = st.segmented_control(
        "校准功能",
        options=[value for value, _ in CALIBRATION_PANEL_TABS],
        format_func=lambda value: CALIBRATION_PANEL_TAB_LABELS.get(value, value),
        selection_mode="single",
        key=key,
    )
    active = str(selected or st.session_state.get(key) or "recommended")
    if active in valid and str(st.query_params.get("domain_calibration_tab") or "") != active:
        st.query_params["domain_calibration_tab"] = active
    return active if active in valid else "recommended"


def _safe_text(value: Any, fallback: str = "-") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    # Keep local paths out of the operator-facing page.
    if "\\" in text or "/" in text:
        return Path(text).name or fallback
    return text


def _summary_rows(summary: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"指标": "图像级阈值", "结果": format_value(summary.get("image_threshold"))},
        {"指标": "像素级阈值", "结果": format_value(summary.get("pixel_threshold"))},
        {"指标": "阈值策略", "结果": _safe_text(summary.get("threshold_strategy"))},
        {"指标": "缺陷方向/轻量参数", "结果": "已启用" if summary.get("use_defect_direction") else "未启用"},
        {"指标": "正常样本", "结果": _safe_text(summary.get("normal_shot"), "0")},
        {"指标": "缺陷样本", "结果": _safe_text(summary.get("defect_shot"), "0")},
    ]


def render_calibration_panel(
    runtime: dict[str, Any],
    *,
    selected_profile: dict[str, Any] | None = None,
    scan_summary: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Render calibration controls inside the target-domain page.

    The recommended target-domain path is the build action directly below
    this panel: it uses the selected image/video sources and the unified
    domain-profile service.  The compatibility tab preserves the old
    defect-direction calibration API for existing datasets and bookmarks;
    it always creates an inactive candidate and never changes the runtime
    profile.
    """
    st.markdown("### 4. 校准与验证")
    st.caption("校准主要调整判定阈值、分数聚合和少量轻量参数，不会默认重新训练骨干网络。")

    profile = selected_profile or {}
    profile_config = profile.get("config") or {}
    profile_id = profile.get("config_profile_id") or ""
    profile_status = "已冻结" if profile.get("frozen") else ("已验证" if profile.get("validated") else "草稿")

    overview = st.container(border=True)
    with overview:
        st.markdown("#### 当前校准上下文")
        columns = st.columns(6)
        values = [
            ("基础方案", _safe_text(profile_id, "未选择")),
            ("数据集", _safe_text(runtime.get("dataset"))),
            ("类别", _safe_text(runtime.get("category"))),
            ("方法模式", _safe_text(profile_config.get("method_mode"), "目标域校准")),
            ("阈值策略", _safe_text(profile_config.get("threshold_method"), "best_f1")),
            ("方案状态", profile_status),
        ]
        for column, (label, value) in zip(columns, values):
            column.metric(label, value)

    target_scan = scan_summary or {}
    normal_scan = target_scan.get("normal") or {}
    defect_scan = target_scan.get("defect") or {}
    source_cols = st.columns(4)
    source_cols[0].metric("正常有效样本", int(normal_scan.get("valid_count") or 0))
    source_cols[1].metric("缺陷校准样本", int(defect_scan.get("valid_count") or 0))
    source_cols[2].metric("当前阈值", format_value(runtime.get("threshold")))
    source_cols[3].metric("当前模式", "只读推理" if runtime.get("inference_mode") == "inference_only" else "开发模式")

    active_tab = _calibration_panel_active_tab()
    if active_tab == "recommended":
        st.info(
            "当前目标域的图片、视频代表帧、去重和质量过滤结果将由下方统一目标域任务使用；"
            "选择方法模式后点击“生成记忆库并完成校准冻结”，一次性生成候选 profile、验证并冻结。",
            icon=":material/verified:",
        )
        st.markdown(
            "**统一职责：** 正常样本只进入正常记忆库；缺陷样本仅用于阈值、轻量校准或独立缺陷原型。"
            " 校准前后指标、资源哈希和隔离状态会写入候选 profile。"
        )
        if not normal_scan or int(normal_scan.get("valid_count") or 0) <= 0:
            st.warning("请先在“样本与来源”区域扫描正常样本。")
        if profile.get("frozen"):
            st.caption("当前基础方案已冻结；如需调整参数，请复制为新版本，旧版本保持可回滚。")

    result: dict[str, Any] | None = None
    if active_tab == "compatibility":
        st.caption("该入口保留旧书签所依赖的校准 API，结果只作为候选验证输出，不会覆盖当前冻结方案。目标域新样本应优先使用上方统一流程。")
        if runtime.get("inference_mode") == "inference_only":
            st.warning("当前为只读推理模式。兼容校准仍可生成离线候选，但不会应用或修改当前运行 profile。")
        param_a, param_b, param_c = st.columns(3)
        normal_shot = param_a.number_input("校准正常样本数", min_value=1, value=100, step=1, key="unified_calibration_normal_shot")
        defect_shot = param_b.number_input("校准缺陷样本数", min_value=0, value=30, step=1, key="unified_calibration_defect_shot")
        seed = param_c.number_input("校准随机种子", min_value=0, value=42, step=1, key="unified_calibration_seed")
        use_direction = st.toggle(
            "启用缺陷方向轻量参数（兼容模式）",
            value=True,
            key="unified_calibration_use_direction",
            help="只拟合缺陷方向相关轻量参数，不训练骨干网络。",
        )
        start = st.button("运行兼容校准验证", type="secondary", width="stretch", key="unified_calibration_start")
        if start:
            with st.spinner("正在调用统一校准服务…"):
                result = run_calibration(
                    dataset=str(runtime.get("dataset") or ""),
                    category=str(runtime.get("category") or ""),
                    config=str(runtime.get("config") or ""),
                    normal_shot=int(normal_shot),
                    defect_shot=int(defect_shot),
                    seed=int(seed),
                    use_defect_direction=bool(use_direction),
                    execute=True,
                )
            if not result.get("ok"):
                st.error(friendly_error_message(result.get("error"), "校准验证失败。"))
                if result.get("bridge"):
                    render_technical_details(result["bridge"], "查看调用技术详情")
            else:
                st.success("校准验证完成，候选结果未应用到当前运行方案。")
                st.session_state["unified_calibration_last_result"] = result

        if result is None:
            result = st.session_state.get("unified_calibration_last_result")
        if result and result.get("ok"):
            summary = result.get("summary") or {}
            st.dataframe(_summary_rows(summary), width="stretch", hide_index=True)
            st.caption(f"候选版本：{_safe_text(summary.get('calibration_version'))} · active={bool(result.get('active'))}")
            render_technical_details(result.get("raw_calibration") or {}, "查看校准技术详情")

    return result


__all__ = ["render_calibration_panel"]
