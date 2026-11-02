from __future__ import annotations

import math
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import altair as alt
import streamlit as st

from feedback_panel import render_task_item_feedback
from service.statistics_service import load_statistics_dashboard, load_statistics_revision
from performance_cache import api_client as client
from delete_preview_confirmation import render_delete_preview_confirmation
from friendly_display import field_label, friendly_error_message, friendly_model_name, friendly_value
from time_utils import format_beijing_datetime
from performance_cache import cached_feedback_statistics, clear_feedback_caches
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
from ui_common import (
    configure_page,
    render_action_button_with_feedback,
    render_action_feedback,
    render_form_submit_button_with_feedback,
    set_action_feedback,
    sidebar_runtime_controls,
    update_action_feedback,
)


LOCAL_TIMEZONE = ZoneInfo("Asia/Shanghai")
MODULE_LABELS = {"单图检测": "single", "批量检测": "batch", "视频检测": "video"}
RESULT_LABELS = {"normal": "正常", "anomaly": "异常", "failed": "失败"}
STATUS_LABELS = {"success": "成功", "failed": "失败", "cancelled": "已取消", "running": "运行中"}
STAGE_LABELS = {
    "input_read_ms": "输入读取",
    "decode_ms": "视频解码",
    "preprocess_ms": "预处理",
    "feature_extraction_ms": "特征提取",
    "ann_search_ms": "近邻检索",
    "anomaly_scoring_ms": "异常评分",
    "inference_ms": "纯推理",
    "postprocess_ms": "后处理",
    "visualization_ms": "可视化",
    "encode_ms": "编码",
    "save_ms": "保存",
    "end_to_end_ms": "端到端",
    "frame_total_ms": "逐帧全流程",
}
PALETTE = ["#168AAD", "#4361EE", "#EF8354", "#6C757D", "#6D597A"]
RESULT_COLORS = alt.Scale(domain=["正常", "异常", "失败"], range=["#168AAD", "#D1495B", "#6C757D"])
SERIES_COLORS = {
    "总检测数": "#4361EE", "任务数": "#4361EE", "处理图片数": "#168AAD", "处理帧数": "#168AAD",
    "正常数": "#168AAD", "异常数": "#D1495B", "成功任务数": "#168AAD", "失败任务数": "#6C757D",
    "平均纯推理": "#EF8354", "平均端到端": "#4361EE", "P95 端到端": "#168AAD",
    "平均": "#4361EE", "P50": "#168AAD", "P95": "#EF8354", "最低": "#6C757D", "最高": "#6D597A",
}

HISTORY_SELECTION_STATE_KEY = "history_selected_record_ids"
HISTORY_LEGACY_SELECTION_STATE_KEY = "history_management_selected"
HISTORY_CHECKBOX_PREFIX = "history_select_"
HISTORY_LEGACY_CHECKBOX_PREFIX = "history_record_"
HISTORY_PENDING_DELETED_IDS_KEY = "history_pending_deleted_record_ids"
HISTORY_DELETE_PLAN_ID_KEY = "history_delete_plan_id"
HISTORY_DELETE_PREVIEW_TOKEN_KEY = "history_delete_preview_token"
HISTORY_DELETE_PREVIEW_SUMMARY_KEY = "history_delete_preview_summary"
HISTORY_DELETE_PLAN_MODE_KEY = "history_delete_plan_mode"
HISTORY_DELETE_PLAN_CREATED_AT_KEY = "history_delete_plan_created_at"
HISTORY_MODE_OPTIONS = {
    "单图检测记录": "single",
    "批量检测记录": "batch",
    "视频检测记录": "video",
}
HISTORY_MODE_SELECTOR_KEY = "history_management_record_section"
HISTORY_MODE_SNAPSHOTS_KEY = "history_management_mode_snapshots"
HISTORY_LOADED_MODE_KEY = "history_management_loaded_mode"
HISTORY_EXPANDED_JOB_ID_KEY = "history_expanded_job_id"
HISTORY_EXPANDED_JOB_MODE_KEY = "history_expanded_job_mode"
HISTORY_EXPANDED_JOB_TITLE_KEY = "history_expanded_job_title"
HISTORY_METHOD_LABELS = {
    "normal_only": "单正常记忆",
    "normal_memory": "单正常记忆",
    "normal_defect_calibration": "正常记忆与缺陷校准",
    "normal_memory_defect_guided_lite": "单正常记忆·缺陷引导轻量校准",
    "normal_defect_dual_memory": "正常与缺陷双记忆",
    "dual_memory": "正常与缺陷双记忆",
    "auto": "自动推荐",
}


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _count(value: Any) -> str:
    number = _number(value)
    return "-" if number is None else f"{int(number):,}"


def _percent(value: Any) -> str:
    number = _number(value)
    return "-" if number is None else f"{number:.1f}%"


def _duration_ms(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "-"
    return f"{number / 1000:.2f} s" if number >= 1000 else f"{number:.2f} ms"


def _duration_seconds(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "-"
    hours = int(number // 3600)
    minutes = int((number % 3600) // 60)
    seconds = int(round(number % 60))
    return f"{hours:d}小时 {minutes:02d}分 {seconds:02d}秒" if hours else f"{minutes:d}分 {seconds:02d}秒"


def _fps(value: Any) -> str:
    number = _number(value)
    return "-" if number is None else f"{number:.2f} FPS"


def _throughput(value: Any) -> str:
    number = _number(value)
    return "-" if number is None else f"{number:.2f} 张/秒"


def _friendly_time(value: Any) -> str:
    return format_beijing_datetime(value, fallback="-")


def _history_method_label(value: Any) -> str:
    method = str(value or "").strip()
    return HISTORY_METHOD_LABELS.get(method, method or "未记录")


def _hero() -> None:
    st.markdown(
        """
        <div class="aoi-workbench-hero analytics-hero">
          <div>
            <div class="aoi-workbench-kicker">Quality & Performance Analytics</div>
            <div class="aoi-workbench-title">统计分析</div>
            <div class="aoi-workbench-copy">汇总单图、批量和视频检测任务的质量结果、性能耗时与运行趋势。</div>
          </div>
          <div class="analytics-hero-badge">真实历史任务</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _install_styles() -> None:
    st.markdown(
        """
        <style>
          .analytics-hero { margin-bottom: .8rem; }
          .analytics-hero-badge {
            align-self: flex-start; padding: .45rem .75rem; border: 1px solid rgba(22,138,173,.25);
            border-radius: 999px; background: rgba(22,138,173,.08); color: #126782; font-weight: 700; font-size: .82rem;
          }
          .analytics-range {
            padding: .72rem .9rem; border: 1px solid rgba(120,139,164,.23); border-radius: 12px;
            background: rgba(248,250,252,.9); color: #38506a; font-size: .9rem; margin: .35rem 0 .75rem;
          }
          .analytics-section-title { font-size: 1.06rem; font-weight: 760; margin: .55rem 0 .08rem; color: #18344f; }
          .analytics-section-copy { color: #687b91; font-size: .88rem; margin-bottom: .45rem; }
          .analytics-chart-header {
            display:flex; align-items:flex-start; justify-content:space-between; gap:.75rem;
            min-height:3.35rem; margin:.02rem 0 .18rem;
          }
          .analytics-chart-heading { min-width:0; }
          .analytics-chart-title { color:#152f49; font-size:1rem; line-height:1.25; font-weight:790; }
          .analytics-chart-copy { margin-top:.2rem; color:#6c7f93; font-size:.76rem; line-height:1.45; }
          .analytics-chart-metric {
            flex:0 0 auto; max-width:46%; padding:.28rem .5rem; border:1px solid rgba(22,138,173,.18);
            border-radius:999px; background:rgba(22,138,173,.065); color:#236a80; font-size:.68rem;
            font-weight:720; line-height:1.25; text-align:right;
          }
          div[class*="st-key-analytics_chart_"] {
            min-width:0; height:100%; border:1px solid rgba(112,139,160,.18) !important; border-radius:14px !important;
            background:linear-gradient(145deg,#ffffff 0%,#fbfdff 100%);
            box-shadow:0 7px 18px rgba(15,39,71,.05);
            transition:border-color 180ms ease, box-shadow 180ms ease;
          }
          div[class*="st-key-analytics_chart_"]:hover {
            border-color:rgba(22,138,173,.28); box-shadow:0 9px 22px rgba(15,39,71,.075);
          }
          div[class*="st-key-analytics_chart_"] div[data-testid="stCaptionContainer"] {
            margin-top:-.2rem; padding-top:.45rem; border-top:1px solid rgba(112,139,160,.12);
          }
          div[data-testid="stMetric"] {
            border: 1px solid rgba(120,139,164,.22); border-radius: 12px; padding: .7rem .8rem;
            background: rgba(255,255,255,.94); min-height: 104px;
          }
          div[data-testid="stMetric"] label { color: #5a6d82; }
          div[data-testid="stMetricValue"] { color: #163550; }
          div[class*="st-key-history_child_panel_"] {
            width: calc(100% - 1.05rem);
            margin: .2rem 0 .85rem 1.05rem;
            border-color: rgba(22,138,173,.24) !important;
            border-left: 3px solid rgba(22,138,173,.72) !important;
            background: linear-gradient(145deg,rgba(246,251,254,.98),rgba(251,253,255,.98));
            box-shadow: 0 5px 15px rgba(30,74,104,.045);
          }
          div[data-testid="stAltairChart"] { min-width:0; border:0; padding:0; background:transparent; }
          div[data-testid="stSegmentedControl"] { margin-bottom: .5rem; }
          @media (max-width: 900px) {
            .analytics-hero-badge { display:none; }
            .analytics-chart-header { min-height:0; flex-wrap:wrap; }
            .analytics-chart-metric { max-width:100%; }
            div[class*="st-key-history_child_panel_"] {
              width: calc(100% - .35rem); margin-left: .35rem;
            }
          }
          @media (max-width: 1300px) {
            div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-analytics_chart_"]) { flex-wrap:wrap; }
            div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-analytics_chart_"]) > div[data-testid="stColumn"] {
              flex:1 1 100% !important; width:100% !important; min-width:0 !important;
            }
          }
          @media (prefers-reduced-motion:reduce) {
            div[class*="st-key-analytics_chart_"] { transition:none; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _range_values() -> tuple[datetime | None, datetime]:
    now = datetime.now(LOCAL_TIMEZONE).replace(microsecond=0)
    preset = st.session_state.get("analytics_range_preset", "最近 7 天")
    if preset == "今天":
        return datetime.combine(now.date(), time.min, LOCAL_TIMEZONE), now
    if preset == "最近 30 天":
        return now - timedelta(days=30), now
    if preset == "最近 90 天":
        return now - timedelta(days=90), now
    if preset == "全部":
        return None, now
    if preset == "自定义时间":
        start = st.session_state.get("analytics_custom_start_applied")
        end = st.session_state.get("analytics_custom_end_applied")
        if isinstance(start, datetime) and isinstance(end, datetime):
            return start, min(end, now)
    return now - timedelta(days=7), now


def _render_time_controls() -> tuple[datetime | None, datetime]:
    presets = ["今天", "最近 7 天", "最近 30 天", "最近 90 天", "全部", "自定义时间"]
    # Keep enough width for the icon + four-character label.  The shared
    # action row has its own button/status columns, so an overly narrow outer
    # column would squeeze the button down to a vertical strip.
    top_left, top_right = st.columns([3.6, 2.4], vertical_alignment="bottom")
    with top_left:
        st.radio("统计时间", presets, horizontal=True, key="analytics_range_preset")
    with top_right:
        refresh_scope = "statistics::refresh"
        refresh_clicked, refresh_status = render_action_button_with_feedback(
            "刷新数据",
            scope=refresh_scope,
            key="analytics_refresh",
            icon=":material/refresh:",
            column_ratio=(0.85, 1.15),
        )
    if refresh_clicked:
        update_action_feedback(refresh_scope, "info", "正在刷新统计数据...", refresh_status)
        st.session_state["analytics_refresh_token"] = int(st.session_state.get("analytics_refresh_token", 0)) + 1

    if st.session_state.get("analytics_range_preset") == "自定义时间":
        now = datetime.now(LOCAL_TIMEZONE).replace(microsecond=0)
        default_start = st.session_state.get("analytics_custom_start_applied", now - timedelta(days=7))
        default_end = st.session_state.get("analytics_custom_end_applied", now)
        with st.container(border=True):
            date_columns = st.columns(4)
            start_date = date_columns[0].date_input("开始日期", value=default_start.date(), key="analytics_start_date")
            start_clock = date_columns[1].time_input("开始时间", value=default_start.time(), key="analytics_start_time")
            end_date = date_columns[2].date_input("结束日期", value=default_end.date(), key="analytics_end_date")
            end_clock = date_columns[3].time_input("结束时间", value=default_end.time(), key="analytics_end_time")
            actions = st.columns([1, 1, 4])
            apply_custom = actions[0].button("应用", type="primary", width="stretch", key="analytics_apply_custom")
            reset_custom = actions[1].button("重置", width="stretch", key="analytics_reset_custom")
            if apply_custom:
                start = datetime.combine(start_date, start_clock, LOCAL_TIMEZONE)
                end = datetime.combine(end_date, end_clock, LOCAL_TIMEZONE)
                if start > end:
                    st.error("开始时间不能晚于结束时间。")
                elif end > now:
                    st.error("结束时间不能晚于当前时间。")
                else:
                    st.session_state["analytics_custom_start_applied"] = start
                    st.session_state["analytics_custom_end_applied"] = end
                    st.session_state["analytics_detail_page"] = 1
            if reset_custom:
                st.session_state["analytics_custom_start_applied"] = now - timedelta(days=7)
                st.session_state["analytics_custom_end_applied"] = now
    return _range_values()


def _filter_value(key: str) -> str | None:
    value = st.session_state.get(key)
    return None if value in (None, "", "全部") else str(value)


def _select_filter(label: str, key: str, values: list[str], *, format_map: dict[str, str] | None = None, long_value: bool = False) -> None:
    options = ["全部", *values]
    current = str(st.session_state.get(key, "全部"))
    if current not in options:
        st.session_state[key] = "全部"
    selected = str(st.session_state.get(key, "全部"))

    def display(value: str) -> str:
        text = (format_map or {}).get(value, value)
        return f"{text[:30]}…" if long_value and len(text) > 31 else text

    st.selectbox(
        label,
        options,
        key=key,
        format_func=display,
        help=(f"当前完整值：{selected}" if long_value and selected != "全部" else None),
    )


def _clear_analytics_filters() -> None:
    for key in (
        "analytics_filter_dataset", "analytics_filter_category", "analytics_filter_config", "analytics_filter_device",
        "analytics_filter_model", "analytics_filter_memory", "analytics_filter_prediction", "analytics_filter_status",
    ):
        st.session_state[key] = "全部"
    st.session_state["analytics_detail_page"] = 1


def _render_filters(options: dict[str, list[str]]) -> None:
    with st.expander("筛选条件", expanded=True):
        first = st.columns(4)
        with first[0]:
            _select_filter("数据集", "analytics_filter_dataset", options.get("dataset", []))
        with first[1]:
            _select_filter("类别", "analytics_filter_category", options.get("category", []))
        with first[2]:
            _select_filter("配置文件", "analytics_filter_config", options.get("config", []), long_value=True)
        with first[3]:
            _select_filter("运行设备", "analytics_filter_device", options.get("device", []))
        second = st.columns(4)
        with second[0]:
            _select_filter("模型版本", "analytics_filter_model", options.get("model_version", []))
        with second[1]:
            _select_filter("记忆库", "analytics_filter_memory", options.get("memory_bank_id", []), long_value=True)
        with second[2]:
            _select_filter("检测结论", "analytics_filter_prediction", options.get("prediction", []), format_map=RESULT_LABELS)
        with second[3]:
            _select_filter("任务状态", "analytics_filter_status", options.get("status", []), format_map=STATUS_LABELS)
        st.button(
            "清除筛选",
            icon=":material/filter_alt_off:",
            key="analytics_clear_filters",
            on_click=_clear_analytics_filters,
        )


@st.cache_data(ttl=15, show_spinner=False)
def _cached_dashboard(module: str, params: tuple[tuple[str, Any], ...], refresh_token: int,
                      statistics_revision: int) -> dict[str, Any]:
    del refresh_token, statistics_revision
    return load_statistics_dashboard(module, **dict(params))


@st.cache_data(ttl=2, show_spinner=False)
def _cached_statistics_revision() -> int:
    """Share the lightweight revision read across dashboard fragments."""

    return int(load_statistics_revision() or 0)


def _metric_grid(items: list[tuple[str, str, str]]) -> None:
    for offset in range(0, len(items), 5):
        columns = st.columns(min(5, len(items) - offset))
        for column, (label, value, help_text) in zip(columns, items[offset : offset + 5]):
            column.metric(label, value, help=help_text)


def _section(title: str, copy: str) -> None:
    st.markdown(f'<div class="analytics-section-title">{title}</div><div class="analytics-section-copy">{copy}</div>', unsafe_allow_html=True)


@contextmanager
def _chart_card(title: str, copy: str, key: str, metric: str | None = None):
    with st.container(border=True, key=f"analytics_chart_{key}"):
        metric_html = f'<div class="analytics-chart-metric">{metric}</div>' if metric else ""
        st.markdown(
            '<div class="analytics-chart-header"><div class="analytics-chart-heading">'
            f'<div class="analytics-chart-title">{title}</div>'
            f'<div class="analytics-chart-copy">{copy}</div></div>{metric_html}</div>',
            unsafe_allow_html=True,
        )
        yield


def _series_scale(labels: list[str]) -> alt.Scale:
    colors = [SERIES_COLORS.get(label, PALETTE[index % len(PALETTE)]) for index, label in enumerate(labels)]
    return alt.Scale(domain=labels, range=colors)


def _value_axis() -> alt.Axis:
    return alt.Axis(
        title=None,
        grid=True,
        gridColor="#E7EDF3",
        gridOpacity=1,
        domainColor="#C9D5E1",
        tickColor="#C9D5E1",
        labelColor="#6B7C90",
        labelPadding=6,
    )


def _category_axis(*, angle: int = 0, limit: int = 110) -> alt.Axis:
    return alt.Axis(
        labelAngle=angle,
        labelLimit=limit,
        labelOverlap="greedy",
        labelColor="#6B7C90",
        labelPadding=7,
        domainColor="#C9D5E1",
        tickColor="#C9D5E1",
        grid=False,
    )


def _long_series(rows: list[dict[str, Any]], fields: dict[str, str]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        for field, label in fields.items():
            value = _number(row.get(field))
            if value is not None:
                output.append({"时间": row.get("bucket"), "系列": label, "数值": value})
    return output


def _line_chart(rows: list[dict[str, Any]], fields: dict[str, str], unit: str, *, mixed_units: bool = False) -> None:
    data = _long_series(rows, fields)
    if not data:
        st.info("当前筛选范围内暂无可绘制的趋势数据。")
        return
    selection = alt.selection_point(fields=["系列"], bind="legend")
    labels = list(fields.values())
    base = (
        alt.Chart(alt.Data(values=data))
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("时间:T", title="时间", axis=alt.Axis(labelAngle=0, labelOverlap="greedy", tickCount=6, labelColor="#6B7C90", titleColor="#53697F", grid=False)),
            y=alt.Y("数值:Q", title=None, scale=alt.Scale(zero=False), axis=_value_axis()),
            color=alt.Color("系列:N", scale=_series_scale(labels), legend=alt.Legend(title=None, orient="top", direction="horizontal", symbolType="circle")),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.18)),
            strokeDash=alt.StrokeDash("系列:N", legend=None, scale=alt.Scale(domain=labels, range=[[1, 0], [7, 3], [2, 2], [9, 3], [4, 2]])),
            tooltip=[alt.Tooltip("时间:T", title="时间", format="%Y-%m-%d %H:%M"), "系列:N", alt.Tooltip("数值:Q", title=unit, format=",.2f")],
        )
        .add_params(selection)
    )
    if mixed_units:
        base = base.properties(height=90).facet(
            row=alt.Row("系列:N", title=None, header=alt.Header(labelOrient="top"))
        ).resolve_scale(y="independent")
    else:
        base = base.properties(height=270)
    _render_chart(base)


def _render_chart(chart: alt.TopLevelMixin) -> None:
    try:
        st.altair_chart(chart, width="stretch")
    except Exception:
        st.error("图表暂时无法渲染，请刷新数据后重试。")


def _donut_chart(rows: list[dict[str, Any]], unit: str) -> None:
    data = []
    for row in rows:
        count = int(row.get("count", 0))
        data.append(
            {
                "结果": RESULT_LABELS.get(str(row.get("result")), str(row.get("result"))),
                "数量": count,
                "占比": _number(row.get("percentage")),
            }
        )
    if not any(row["数量"] for row in data):
        st.info("当前筛选范围内暂无结果分布数据。")
        return
    total = sum(row["数量"] for row in data)
    selection = alt.selection_point(fields=["结果"], bind="legend")
    arc = (
        alt.Chart(alt.Data(values=data))
        .mark_arc(innerRadius=62, outerRadius=105, stroke="#ffffff", strokeWidth=2)
        .encode(
            theta=alt.Theta("数量:Q"),
            color=alt.Color("结果:N", scale=RESULT_COLORS, legend=alt.Legend(title=None, orient="bottom", direction="horizontal")),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.16)),
            tooltip=["结果:N", alt.Tooltip("数量:Q", title=unit, format=","), alt.Tooltip("占比:Q", format=".1f", title="占比(%)")],
        )
        .add_params(selection)
    )
    center_value = alt.Chart(alt.Data(values=[{"总数": total}])).mark_text(fontSize=25, fontWeight=800, color="#18344F", dy=-5).encode(text=alt.Text("总数:Q", format=","))
    center_label = alt.Chart(alt.Data(values=[{"标签": "总检测数"}])).mark_text(fontSize=11, color="#718196", dy=17).encode(text="标签:N")
    chart = (arc + center_value + center_label).properties(height=285)
    _render_chart(chart)
    st.caption(" · ".join(f"{row['结果']} {row['数量']:,}（{row['占比']:.1f}%）" for row in data if row["占比"] is not None))


def _bar_chart(rows: list[dict[str, Any]], x: str, y: str, *, x_title: str, y_title: str, horizontal: bool = False, color: str = "#168AAD") -> None:
    data = [row for row in rows if _number(row.get(y)) is not None]
    if not data:
        st.info("当前筛选范围内暂无可绘制数据。")
        return
    chart = alt.Chart(alt.Data(values=data)).mark_bar(color=color, cornerRadiusEnd=4)
    if horizontal:
        chart = chart.encode(
            y=alt.Y(f"{x}:N", title=None, sort="-x", axis=_category_axis(limit=150)),
            x=alt.X(f"{y}:Q", title=y_title, axis=_value_axis()),
            tooltip=[alt.Tooltip(f"{x}:N", title=x_title), alt.Tooltip(f"{y}:Q", title=y_title, format=",.2f")],
        )
    else:
        label_angle = -20 if len(data) > 5 or max((len(str(row.get(x) or "")) for row in data), default=0) > 8 else 0
        chart = chart.encode(
            x=alt.X(f"{x}:N", title=x_title, sort=None, axis=_category_axis(angle=label_angle, limit=105)),
            y=alt.Y(f"{y}:Q", title=None, axis=_value_axis()),
            tooltip=[alt.Tooltip(f"{x}:N", title=x_title), alt.Tooltip(f"{y}:Q", title=y_title, format=",.2f")],
        )
    _render_chart(chart.properties(height=270))


def _category_charts(rows: list[dict[str, Any]], *, key_prefix: str) -> None:
    if not rows:
        st.info("当前筛选范围内暂无类别统计。")
        return
    left, right = st.columns(2)
    long = []
    for row in rows:
        long.extend(
            [
                {"类别": row.get("category"), "系列": "检测数量", "数量": row.get("total")},
                {"类别": row.get("category"), "系列": "异常数量", "数量": row.get("anomaly")},
            ]
        )
    with left:
        with _chart_card("类别检测数量", "按类别对比检测总量与异常数量。", f"{key_prefix}_category_count", "图片数量（张）"):
            selection = alt.selection_point(fields=["系列"], bind="legend")
            chart = (
                alt.Chart(alt.Data(values=long))
                .mark_bar(cornerRadiusEnd=3)
                .encode(
                    y=alt.Y("类别:N", sort="-x", title=None, axis=_category_axis(limit=150)),
                    x=alt.X("数量:Q", title=None, axis=_value_axis()),
                    color=alt.Color("系列:N", scale=alt.Scale(domain=["检测数量", "异常数量"], range=["#168AAD", "#D1495B"]), legend=alt.Legend(title=None, orient="top")),
                    opacity=alt.condition(selection, alt.value(1), alt.value(0.16)),
                    yOffset="系列:N",
                    tooltip=["类别:N", "系列:N", alt.Tooltip("数量:Q", title="图片数量（张）", format=",")],
                )
                .add_params(selection)
                .properties(height=max(240, len(rows) * 42))
            )
            _render_chart(chart)
    with right:
        with _chart_card("类别异常率", "比较各类别异常结果所占比例。", f"{key_prefix}_category_rate", "异常率（%）"):
            _bar_chart(rows, "category", "anomaly_rate", x_title="类别", y_title="异常率（%）", horizontal=True, color="#EF8354")


def _timing_charts(rows: list[dict[str, Any]], *, task_level: bool, key_prefix: str) -> None:
    data = [
        {
            "阶段": STAGE_LABELS.get(str(row.get("stage")), str(row.get("stage"))),
            "累计耗时": row.get("total_ms"),
            "平均耗时": row.get("mean_ms"),
            "P95": row.get("p95_ms"),
        }
        for row in rows
        if row.get("mean_ms") is not None or row.get("total_ms") is not None
    ]
    if not data:
        st.info("旧记录尚未保存可分解的阶段耗时，当前不显示空阶段。")
        return
    left, right = st.columns(2)
    with left:
        with _chart_card("阶段平均耗时", "比较检测链路各阶段的平均处理耗时。", f"{key_prefix}_timing_mean", "平均耗时（ms）"):
            _bar_chart(data, "阶段", "平均耗时", x_title="阶段", y_title="平均耗时（ms）", color="#168AAD")
    with right:
        field = "累计耗时" if task_level else "P95"
        title = "累计耗时（ms）" if task_level else "P95 耗时（ms）"
        copy = "汇总各阶段在当前任务范围内的累计耗时。" if task_level else "展示各阶段高分位处理耗时。"
        with _chart_card("阶段累计耗时" if task_level else "阶段 P95 耗时", copy, f"{key_prefix}_timing_secondary", title):
            _bar_chart(data, "阶段", field, x_title="阶段", y_title=title, color="#4361EE")


def _score_histogram(payload: dict[str, Any]) -> None:
    rows = payload.get("bins") or []
    if not rows:
        st.info("当前筛选范围内暂无异常分数数据。")
        return
    labels = [str(row.get("label") or "-") for row in rows]
    bars = (
        alt.Chart(alt.Data(values=rows))
        .mark_bar(color="#4361EE", opacity=.9, stroke="#3148C8", strokeWidth=.5)
        .encode(
            x=alt.X("label:N", title="异常分数区间", sort=labels, axis=_category_axis(angle=-20, limit=105)),
            y=alt.Y("count:Q", title=None, axis=_value_axis()),
            tooltip=[alt.Tooltip("label:N", title="分数区间"), alt.Tooltip("count:Q", title="图片数量（张）")],
        )
    )
    stats = payload.get("statistics") or {}
    markers = []
    for label, value, color in (
        ("平均分数", stats.get("mean"), "#168AAD"),
        ("P50", stats.get("p50"), "#2E7D6E"),
        ("P95", stats.get("p95"), "#EF8354"),
        ("当前平均阈值", payload.get("average_threshold"), "#D1495B"),
    ):
        number = _number(value)
        if number is not None:
            marker_bin = next(
                (
                    str(row.get("label") or "-")
                    for row in rows
                    if _number(row.get("lower")) is not None
                    and _number(row.get("upper")) is not None
                    and float(row["lower"]) <= number <= float(row["upper"])
                ),
                None,
            )
            if marker_bin is not None:
                markers.append({"标记": label, "数值": number, "分数区间": marker_bin, "颜色": color})
    chart = bars
    if markers:
        marker_scale = alt.Scale(domain=[row["标记"] for row in markers], range=[row["颜色"] for row in markers])
        rules = (
            alt.Chart(alt.Data(values=markers))
            .mark_rule(strokeWidth=2, strokeDash=[5, 3])
            .encode(
                x=alt.X("分数区间:N", sort=labels),
                color=alt.Color("标记:N", scale=marker_scale, legend=alt.Legend(title=None, orient="top", direction="horizontal")),
                tooltip=["标记:N", alt.Tooltip("数值:Q", title="分数", format=".4f")],
            )
        )
        chart = bars + rules
    chart = chart.properties(height=270)
    _render_chart(chart)

    def score_text(value: Any) -> str:
        number = _number(value)
        return "未记录" if number is None else f"{number:.4f}"

    st.caption(
        f"平均值 {score_text(stats.get('mean'))} · P50 {score_text(stats.get('p50'))} · "
        f"P95 {score_text(stats.get('p95'))} · 最大值 {score_text(stats.get('max'))} · "
        f"当前平均阈值 {score_text(payload.get('average_threshold'))}"
    )


def _visualization_comparison(rows: list[dict[str, Any]], module: str) -> None:
    if len(rows) < 2 or not any(row.get("generate_visualizations") is False for row in rows):
        st.info("暂无足够数据：需要同时存在生成和不生成可视化的真实成功任务。")
        return
    metrics = {
        "batch": {"average_end_to_end_ms": "平均任务耗时（ms）", "average_item_ms": "平均单图耗时（ms）", "average_throughput": "吞吐率（图片/秒）"},
        "video": {"average_end_to_end_ms": "平均任务耗时（ms）", "average_frame_ms": "平均每帧耗时（ms）", "average_encode_ms": "平均编码耗时（ms）", "average_full_pipeline_fps": "全流程 FPS"},
    }[module]
    data = []
    for row in rows:
        group = "生成可视化" if row.get("generate_visualizations") else "不生成可视化"
        for key, label in metrics.items():
            value = _number(row.get(key))
            if value is not None:
                data.append({"模式": group, "指标": label, "数值": value})
    if not data:
        st.info("暂无足够的可比较耗时数据。")
        return
    selection = alt.selection_point(fields=["模式"], bind="legend")
    chart = (
        alt.Chart(alt.Data(values=data))
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("指标:N", title=None, axis=_category_axis(limit=150)),
            xOffset=alt.XOffset("模式:N"),
            y=alt.Y("数值:Q", title=None, axis=_value_axis()),
            color=alt.Color("模式:N", scale=alt.Scale(range=["#168AAD", "#EF8354"]), legend=alt.Legend(title=None, orient="top")),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.16)),
            tooltip=["模式:N", "指标:N", alt.Tooltip("数值:Q", format=",.2f")],
        )
        .add_params(selection)
        .properties(height=235)
    )
    _render_chart(chart)


def _single_dashboard(data: dict[str, Any]) -> None:
    summary = data.get("summary") or {}
    _metric_grid(
        [
            ("检测任务数", _count(summary.get("task_count")), "当前筛选范围内的单图任务"),
            ("成功任务数", _count(summary.get("success_tasks")), "已保存真实结果的任务"),
            ("失败任务数", _count(summary.get("failed_tasks")), "后端明确标记失败的任务"),
            ("正常图片数", _count(summary.get("normal_count")), "模型判定为正常"),
            ("异常图片数", _count(summary.get("anomaly_count")), "模型判定为异常"),
            ("异常占比", _percent(summary.get("anomaly_rate")), "异常图片 / 有效图片"),
            ("平均异常分数", "-" if _number(summary.get("average_score")) is None else f"{summary['average_score']:.4f}", "仅统计有效分数"),
            ("平均端到端时间", _duration_ms(summary.get("average_end_to_end_ms")), "输入开始至结果可用"),
            ("平均纯推理时间", _duration_ms(summary.get("average_inference_ms")), "不包含可视化与写盘"),
            ("已生成可视化任务", _count(summary.get("visualized_tasks")), "有明确可视化产物的任务"),
        ]
    )
    _section("质量结果与趋势", "所有图表共享当前时间和公共筛选条件。")
    left, right = st.columns([1.35, 0.65])
    with left:
        with _chart_card("检测数量趋势", "展示当前时间区间内正常、异常和全部检测数量的变化。", "single_count_trend", "检测数量（次）"):
            _line_chart(data.get("time_series") or [], {"task_count": "总检测数", "normal": "正常数", "anomaly": "异常数"}, "检测数量（次）")
    with right:
        with _chart_card("检测结果分布", "汇总正常、异常和失败结果的数量及占比。", "single_result_distribution", "结果数量及占比"):
            _donut_chart(data.get("result_distribution") or [], "图片数量（张）")
    left, right = st.columns(2)
    with left:
        with _chart_card("异常分数分布", "展示异常分数所在区间，并标记平均值、分位数与当前阈值。", "single_score_distribution", "图片数量（张）"):
            _score_histogram(data.get("score_distribution") or {})
    with right:
        with _chart_card("性能趋势", "对比纯推理耗时、端到端耗时及 P95 耗时变化。", "single_performance_trend", "平均耗时（ms）"):
            _line_chart(
                data.get("time_series") or [],
                {"average_inference_ms": "平均纯推理", "average_end_to_end_ms": "平均端到端", "p95_end_to_end_ms": "P95 端到端"},
                "平均耗时（ms）",
            )
    _section("类别检测分布", "按类别比较检测规模与异常比例。")
    _category_charts(data.get("category_distribution") or [], key_prefix="single")
    _section("性能耗时组成", "对比检测链路各阶段的平均与高分位耗时。")
    _timing_charts((data.get("timing_statistics") or {}).get("stages") or [], task_level=False, key_prefix="single")


def _batch_dashboard(data: dict[str, Any]) -> None:
    summary = data.get("summary") or {}
    _metric_grid(
        [
            ("批量任务数", _count(summary.get("task_count")), "当前筛选范围内的批量任务"),
            ("成功任务数", _count(summary.get("success_tasks")), "完成的批量任务"),
            ("失败任务数", _count(summary.get("failed_tasks")), "明确失败的批量任务"),
            ("累计处理图片数", _count(summary.get("total_images")), "任务输入图片总量"),
            ("正常图片数", _count(summary.get("normal_count")), "有效结果中的正常图片"),
            ("异常图片数", _count(summary.get("anomaly_count")), "有效结果中的异常图片"),
            ("平均每批图片数", "-" if _number(summary.get("average_batch_size")) is None else f"{summary['average_batch_size']:.1f} 张", "累计图片 / 批量任务"),
            ("平均批量任务耗时", _duration_ms(summary.get("average_task_ms")), "仅使用真实任务级计时"),
            ("平均单图耗时", _duration_ms(summary.get("average_item_ms")), "排除失败样本"),
            ("平均吞吐率", _throughput(summary.get("average_throughput")), "仅使用已记录任务吞吐率"),
        ]
    )
    _section("批量规模、质量与吞吐", "旧任务缺少任务级耗时或吞吐率时保留为空，不使用壁钟时间补造。")
    left, right = st.columns([1.35, 0.65])
    with left:
        with _chart_card("批量任务趋势", "展示批量任务、处理图片和任务状态数量的变化。", "batch_task_trend", "任务与图片数量"):
            _line_chart(
                data.get("time_series") or [],
                {"task_count": "任务数", "image_count": "处理图片数", "success_tasks": "成功任务数", "failed_tasks": "失败任务数"},
                "任务与图片数量",
            )
    with right:
        with _chart_card("批量结果分布", "汇总批量检测中的正常、异常和失败图片。", "batch_result_distribution", "结果数量及占比"):
            _donut_chart(data.get("result_distribution") or [], "图片数量（张）")
    left, right = st.columns(2)
    with left:
        with _chart_card("批量规模分布", "展示每个图片数量区间内的批量任务数量。", "batch_scale_distribution", "任务数量（个）"):
            _bar_chart(data.get("scale_distribution") or [], "label", "count", x_title="每批图片数量区间", y_title="任务数量（个）")
    with right:
        with _chart_card("批量吞吐率趋势", "对比平均、分位数及吞吐率范围的变化。", "batch_throughput_trend", "吞吐率（图片/秒）"):
            _line_chart(
                data.get("time_series") or [],
                {"average_throughput": "平均", "p50_throughput": "P50", "p95_throughput": "P95", "min_throughput": "最低", "max_throughput": "最高"},
                "吞吐率（图片/秒）",
            )
    _section("批量耗时分解", "比较批量任务各处理阶段的耗时构成。")
    _timing_charts((data.get("timing_statistics") or {}).get("stages") or [], task_level=True, key_prefix="batch")
    _section("类别异常率对比", "按类别比较检测数量和异常比例。")
    _category_charts(data.get("category_distribution") or [], key_prefix="batch")
    with _chart_card("可视化开关影响", "比较是否生成可视化结果时的真实性能记录。", "batch_visualization_impact", "实际记录值（按指标）"):
        _visualization_comparison(data.get("visualization_comparison") or [], "batch")


def _fps_chart(rows: list[dict[str, Any]]) -> None:
    data = []
    for row in rows:
        for key, label in (("input_fps", "输入 FPS"), ("pure_inference_fps", "纯推理 FPS"), ("full_pipeline_fps", "完整处理 FPS")):
            value = _number(row.get(key))
            if value is not None:
                data.append({"任务": row.get("label"), "系列": label, "FPS": value})
    if not data:
        st.info("当前视频任务没有可比较的 FPS 记录。")
        return
    selection = alt.selection_point(fields=["系列"], bind="legend")
    chart = (
        alt.Chart(alt.Data(values=data))
        .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("任务:N", title="视频任务", axis=alt.Axis(labels=False, ticks=False, grid=False, domainColor="#C9D5E1", titleColor="#53697F")),
            y=alt.Y("FPS:Q", title=None, axis=_value_axis()),
            color=alt.Color("系列:N", scale=alt.Scale(range=["#6C757D", "#4361EE", "#168AAD"]), legend=alt.Legend(title=None, orient="top")),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.16)),
            xOffset="系列:N",
            tooltip=["任务:N", "系列:N", alt.Tooltip("FPS:Q", title="处理速度（FPS）", format=",.2f")],
        )
        .add_params(selection)
        .properties(height=285)
    )
    _render_chart(chart)


def _video_dashboard(data: dict[str, Any]) -> None:
    summary = data.get("summary") or {}
    _metric_grid(
        [
            ("视频任务数", _count(summary.get("task_count")), "离线视频文件检测任务"),
            ("成功任务数", _count(summary.get("success_tasks")), "已完整完成任务"),
            ("失败任务数", _count(summary.get("failed_tasks")), "明确失败任务"),
            ("累计视频时长", _duration_seconds(summary.get("total_duration_seconds")), "输入视频时长合计"),
            ("累计处理帧数", _count(summary.get("total_frames")), "真实解码记录帧数"),
            ("异常帧数", _count(summary.get("anomaly_frames")), "判定为异常的帧"),
            ("平均异常帧比例", _percent(summary.get("average_anomaly_rate")), "按任务异常帧率平均"),
            ("平均完整处理 FPS", _fps(summary.get("average_full_pipeline_fps")), "包含解码、后处理和保存"),
            ("平均纯推理 FPS", _fps(summary.get("average_pure_inference_fps")), "不等于完整处理 FPS"),
            ("达到输入 FPS 的任务比例", _percent(summary.get("reached_input_fps_rate")), "仅离线处理吞吐对比"),
        ]
    )
    _section("视频帧质量与处理性能", "完整处理 FPS 与纯推理 FPS 分开统计，不描述为摄像头实时检测。")
    left, right = st.columns([1.35, 0.65])
    with left:
        with _chart_card("视频任务趋势", "展示视频任务量、处理帧数和视频时长变化。", "video_task_trend", "各系列原单位"):
            _line_chart(data.get("time_series") or [], {"task_count": "任务数", "frame_count": "处理帧数", "duration_seconds": "视频时长（秒）"}, "各系列原单位", mixed_units=True)
    with right:
        with _chart_card("视频检测结果分布", "汇总正常、异常和失败帧的数量及占比。", "video_result_distribution", "帧数量及占比"):
            _donut_chart(data.get("result_distribution") or [], "帧数量（帧）")
    left, right = st.columns(2)
    with left:
        with _chart_card("输入 FPS 与处理 FPS", "对比输入、纯推理和完整处理速度。", "video_fps_comparison", "处理速度（FPS）"):
            _fps_chart(data.get("fps_comparison") or [])
    with right:
        with _chart_card("视频性能趋势", "分别展示逐帧耗时与完整处理速度变化。", "video_performance_trend", "耗时（ms）/ 速度（FPS）"):
            _line_chart(
                data.get("time_series") or [],
                {"average_frame_ms": "平均每帧", "p50_frame_ms": "P50 每帧", "p95_frame_ms": "P95 每帧", "average_full_pipeline_fps": "完整处理 FPS"},
                "耗时（ms）/ 速度（FPS）",
                mixed_units=True,
            )
    _section("视频阶段耗时组成", "比较视频处理链路各阶段的平均与累计耗时。")
    _timing_charts((data.get("timing_statistics") or {}).get("stages") or [], task_level=True, key_prefix="video")
    left, right = st.columns(2)
    with left:
        with _chart_card("异常帧比例分布", "展示各异常帧比例区间内的视频任务数量。", "video_anomaly_ratio", "任务数量（个）"):
            _bar_chart(data.get("anomaly_ratio_distribution") or [], "label", "count", x_title="异常帧比例区间", y_title="任务数量（个）", color="#EF8354")
    with right:
        with _chart_card("可视化视频生成影响", "比较是否生成结果视频时的真实性能记录。", "video_visualization_impact", "实际记录值（按指标）"):
            _visualization_comparison(data.get("visualization_comparison") or [], "video")


def _detail_columns(module: str) -> dict[str, str]:
    common = {"time": "时间", "dataset": "数据集", "category": "类别", "status": "任务状态", "generate_visualizations": "可视化状态"}
    if module == "single":
        return {
            "time": "时间", "filename": "文件名称", "dataset": "数据集", "category": "类别", "prediction": "检测结果",
            "score": "异常分数", "threshold": "阈值", "pure_inference_ms": "纯推理(ms)",
            "end_to_end_ms": "端到端(ms)", "generate_visualizations": "可视化状态", "status": "任务状态",
        }
    if module == "batch":
        return {
            "time": "任务时间", "input_name": "输入目录", "total_images": "图片总数", "successful_items": "成功数",
            "failed_items": "失败数", "anomaly_count": "异常数", "anomaly_rate": "异常率(%)", "end_to_end_ms": "总耗时(ms)",
            "average_item_ms": "平均单图(ms)", "throughput": "吞吐率(张/秒)", "generate_visualizations": "可视化状态", "status": "任务状态",
        }
    return {
        "time": "任务时间", "filename": "视频文件", "duration_seconds": "时长(s)", "total_frames": "总帧数",
        "anomaly_count": "异常帧数", "anomaly_rate": "异常帧率(%)", "input_fps": "输入 FPS",
        "pure_inference_fps": "纯推理 FPS", "full_pipeline_fps": "完整处理 FPS", "end_to_end_ms": "总耗时(ms)",
        "generate_visualizations": "可视化视频状态", "status": "任务状态",
    }


def _render_details(data: dict[str, Any], module: str) -> None:
    _section("任务明细", "明细与上方图表使用完全相同的筛选条件；每页最多展示当前选择的记录数。")
    pagination = data.get("pagination") or {}
    total_pages = max(1, int(pagination.get("total_pages", 1)))
    if int(st.session_state.get("analytics_detail_page", 1)) > total_pages:
        st.session_state["analytics_detail_page"] = total_pages
        st.rerun()
    sort_options = {
        "single": {"created_at": "时间", "score": "异常分数", "end_to_end_ms": "端到端耗时"},
        "batch": {"created_at": "时间", "item_count": "图片数量", "end_to_end_ms": "总耗时", "throughput": "吞吐率", "anomaly_rate": "异常率"},
        "video": {"created_at": "时间", "item_count": "帧数", "end_to_end_ms": "总耗时", "full_pipeline_fps": "完整处理 FPS", "anomaly_rate": "异常帧率"},
    }[module]
    if st.session_state.get("analytics_detail_sort") not in sort_options:
        st.session_state["analytics_detail_sort"] = "created_at"
    controls = st.columns([1.5, 1, 1, 1])
    with controls[0]:
        st.selectbox("排序字段", list(sort_options), format_func=sort_options.get, key="analytics_detail_sort")
    with controls[1]:
        st.selectbox("排序方向", ["desc", "asc"], format_func=lambda value: "降序" if value == "desc" else "升序", key="analytics_detail_order")
    with controls[2]:
        st.selectbox("每页数量", [10, 20, 50], key="analytics_detail_page_size")
    with controls[3]:
        st.number_input("页码", min_value=1, max_value=total_pages, step=1, key="analytics_detail_page")
    rows = data.get("task_details") or []
    if not rows:
        st.info("当前筛选范围内暂无任务明细。")
        return
    columns = _detail_columns(module)
    display_rows = []
    for row in rows:
        rendered = {}
        for key, label in columns.items():
            value = row.get(key)
            if key == "time":
                value = _friendly_time(value)
            elif key == "status":
                value = STATUS_LABELS.get(str(value), str(value or "-"))
            elif key == "prediction":
                value = RESULT_LABELS.get(str(value), str(value or "-"))
            elif key == "generate_visualizations":
                value = "已生成" if value is True else ("未生成" if value is False else "未记录")
            elif value in (None, ""):
                value = friendly_value(value, key)
            rendered[label] = value
        display_rows.append(rendered)
    st.dataframe(display_rows, hide_index=True, width="stretch", height=min(470, 62 + len(display_rows) * 35))
    st.caption(f"共 {int(pagination.get('total', 0)):,} 条 · 第 {int(pagination.get('page', 1))} / {total_pages} 页")
    choices = [f"{index + 1}. {row.get('filename') or row.get('input_name') or '任务'} · {_friendly_time(row.get('time'))}" for index, row in enumerate(rows)]
    selected = st.selectbox("查看任务详情", choices, key=f"analytics_detail_view_{module}")
    if selected:
        item = rows[choices.index(selected)]
        with st.expander("当前任务详情", expanded=False):
            detail_columns = st.columns(3)
            safe_items = [(key, value) for key, value in item.items() if key != "record_key"]
            labels = _detail_columns(module)
            for index, (key, value) in enumerate(safe_items):
                label = labels.get(key, field_label(key))
                if key == "time":
                    value = _friendly_time(value)
                elif key == "status":
                    value = STATUS_LABELS.get(str(value), value)
                elif key == "prediction":
                    value = RESULT_LABELS.get(str(value), value)
                elif key == "generate_visualizations":
                    value = "已生成" if value is True else ("未生成" if value is False else "未记录")
                else:
                    value = friendly_value(value, key)
                detail_columns[index % 3].metric(label, str(value))


def _management_bytes(value: Any) -> str:
    if value is None:
        return "按需计算"
    try:
        size = float(value or 0)
    except (TypeError, ValueError):
        size = 0
    units = ("B", "KB", "MB", "GB")
    index = 0
    while size >= 1024 and index < len(units) - 1:
        size /= 1024
        index += 1
    return f"{size:.1f} {units[index]}"


def _history_checkbox_key(record_id: str) -> str:
    mode = str(st.session_state.get("history_management_mode") or "single")
    return stable_checkbox_key(f"history_{mode}", record_id)


def _history_selection_persistence_key(mode: str | None = None) -> str:
    selected_mode = str(mode or st.session_state.get("history_management_mode") or "single")
    return f"detection_history:selected:{selected_mode}"


def _history_switch_mode(mode: str) -> None:
    """Persist each record type's filters, paging, selection and cleanup state."""
    mode = mode if mode in set(HISTORY_MODE_OPTIONS.values()) else "single"
    loaded = str(st.session_state.get(HISTORY_LOADED_MODE_KEY) or "")
    if loaded == mode:
        st.session_state["history_management_mode"] = mode
        store_persistent_value(
            st.session_state,
            "history_management_mode",
            mode,
            persistence_key="detection_history:active_mode",
        )
        return
    snapshots = dict(st.session_state.get(HISTORY_MODE_SNAPSHOTS_KEY) or {})
    excluded = {
        HISTORY_MODE_SELECTOR_KEY,
        HISTORY_MODE_SNAPSHOTS_KEY,
        HISTORY_LOADED_MODE_KEY,
        # Streamlit button values are transient widget state and cannot be
        # restored through session_state.  Keeping them in a mode snapshot
        # makes the next record type fail before its checkboxes are rendered.
        "history_management_apply_filter",
        "history_management_refresh",
        "history_management_toggle_page_selection",
        "history_management_clear_selection",
    }
    managed = {
        key
        for key in list(st.session_state)
        if (key.startswith("history_management_") or key in {
            HISTORY_SELECTION_STATE_KEY,
            HISTORY_PENDING_DELETED_IDS_KEY,
            HISTORY_DELETE_PLAN_ID_KEY,
            HISTORY_DELETE_PREVIEW_TOKEN_KEY,
            HISTORY_DELETE_PREVIEW_SUMMARY_KEY,
            HISTORY_DELETE_PLAN_MODE_KEY,
            HISTORY_DELETE_PLAN_CREATED_AT_KEY,
        })
        and key not in excluded
    }
    if loaded:
        snapshots[loaded] = {key: st.session_state.get(key) for key in managed}
    for key in managed:
        st.session_state.pop(key, None)
    for key, value in dict(snapshots.get(mode) or {}).items():
        if key not in excluded:
            st.session_state[key] = value
    st.session_state[HISTORY_MODE_SNAPSHOTS_KEY] = snapshots
    st.session_state[HISTORY_LOADED_MODE_KEY] = mode
    st.session_state["history_management_mode"] = mode
    store_persistent_value(
        st.session_state,
        "history_management_mode",
        mode,
        persistence_key="detection_history:active_mode",
    )


def _history_selected_record_ids() -> set[str]:
    """Return the canonical, stable record-id selection for this session."""
    return read_selected_ids(
        st.session_state,
        HISTORY_SELECTION_STATE_KEY,
        legacy_keys=(HISTORY_LEGACY_SELECTION_STATE_KEY,),
        persistence_key=_history_selection_persistence_key(),
    )


def _history_store_selected_record_ids(selected: set[str]) -> None:
    store_selected_ids(
        st.session_state,
        HISTORY_SELECTION_STATE_KEY,
        selected,
        persistence_key=_history_selection_persistence_key(),
    )


def _history_sync_checkbox_states(record_ids: set[str], selected: set[str]) -> None:
    """Synchronize widget state before the corresponding checkboxes are created."""
    for record_id in record_ids:
        st.session_state.pop(f"{HISTORY_LEGACY_CHECKBOX_PREFIX}{record_id}", None)
    mode = str(st.session_state.get("history_management_mode") or "single")
    persistence_key = _history_selection_persistence_key(mode)
    store_selected_ids(
        st.session_state,
        HISTORY_SELECTION_STATE_KEY,
        selected,
        persistence_key=persistence_key,
    )
    sync_checkbox_states(
        st.session_state,
        state_key=HISTORY_SELECTION_STATE_KEY,
        scope=f"history_{mode}",
        item_ids=record_ids,
        persistence_key=persistence_key,
    )


def _history_record_selection_changed(record_id: str) -> None:
    """Fold a row checkbox change back into the one canonical selection set."""
    mode = str(st.session_state.get("history_management_mode") or "single")
    update_checkbox_selection(
        st.session_state,
        state_key=HISTORY_SELECTION_STATE_KEY,
        scope=f"history_{mode}",
        item_id=record_id,
        persistence_key=_history_selection_persistence_key(mode),
    )


def _history_set_page_selection(record_ids: tuple[str, ...], should_select: bool) -> None:
    """Apply one page-level action before Streamlit performs its normal rerun."""
    mode = str(st.session_state.get("history_management_mode") or "single")
    update_page_selection(
        st.session_state,
        state_key=HISTORY_SELECTION_STATE_KEY,
        scope=f"history_{mode}",
        page_ids=record_ids,
        should_select=should_select,
        persistence_key=_history_selection_persistence_key(mode),
    )


def _history_clear_all_selections(current_page_record_ids: tuple[str, ...]) -> None:
    mode = str(st.session_state.get("history_management_mode") or "single")
    clear_id_selection(
        st.session_state,
        state_key=HISTORY_SELECTION_STATE_KEY,
        scopes=(f"history_{mode}",),
        visible_ids=current_page_record_ids,
        persistence_key=_history_selection_persistence_key(mode),
    )


def _history_apply_pending_deleted_selection_cleanup() -> None:
    """Remove only successfully deleted records and their obsolete widget keys."""
    deleted = {
        str(record_id)
        for record_id in (st.session_state.pop(HISTORY_PENDING_DELETED_IDS_KEY, set()) or set())
        if str(record_id)
    }
    if not deleted:
        return
    mode = str(st.session_state.get("history_management_mode") or "single")
    remove_selected_ids(
        st.session_state,
        state_key=HISTORY_SELECTION_STATE_KEY,
        removed_ids=deleted,
        scopes=(f"history_{mode}",),
        persistence_key=_history_selection_persistence_key(mode),
    )
    for record_id in deleted:
        st.session_state.pop(f"{HISTORY_LEGACY_CHECKBOX_PREFIX}{record_id}", None)


def _history_store_delete_plan(preview: dict[str, Any], confirmation_mode: str) -> None:
    st.session_state[HISTORY_DELETE_PLAN_ID_KEY] = str(preview.get("plan_id") or "")
    st.session_state[HISTORY_DELETE_PREVIEW_TOKEN_KEY] = str(preview.get("preview_token") or "")
    st.session_state[HISTORY_DELETE_PREVIEW_SUMMARY_KEY] = dict(preview)
    st.session_state[HISTORY_DELETE_PLAN_MODE_KEY] = str(confirmation_mode or "records")
    st.session_state[HISTORY_DELETE_PLAN_CREATED_AT_KEY] = str(preview.get("plan_created_at") or "")


def _history_clear_delete_plan() -> None:
    for key in (
        HISTORY_DELETE_PLAN_ID_KEY,
        HISTORY_DELETE_PREVIEW_TOKEN_KEY,
        HISTORY_DELETE_PREVIEW_SUMMARY_KEY,
        HISTORY_DELETE_PLAN_MODE_KEY,
        HISTORY_DELETE_PLAN_CREATED_AT_KEY,
    ):
        st.session_state.pop(key, None)


def _render_clear_all_confirmation(
    *,
    active_mode: str,
    record_type: str,
    record_short_name: str,
    preview: dict[str, Any],
    execute_callback: Any,
) -> dict[str, Any] | None:
    """Render the shared, plan-bound high-risk confirmation for all record types."""
    return render_delete_preview_confirmation(
        preview={
            **dict(preview),
            "plan_id": preview.get("plan_id") or st.session_state.get(HISTORY_DELETE_PLAN_ID_KEY),
            "preview_token": preview.get("preview_token") or st.session_state.get(HISTORY_DELETE_PREVIEW_TOKEN_KEY),
        },
        title=f"清空全部{record_type}删除预览",
        action_label=f"清空全部可删除的{record_short_name}记录",
        object_label=record_type,
        acknowledgement_key=f"history_clear_all_ack_{active_mode}",
        button_key=f"history_execute_all_{active_mode}",
        execute_callback=execute_callback,
        extra_summary=[("原始数据集/模型", "保留"), ("Memory Bank/配置", "保留")],
    )


def _history_register_cleanup_job(job: dict[str, Any], *, preview_key: str | None = None) -> str:
    """Register one submitted cleanup job without forcing an extra page rerun."""
    job_id = str(job.get("cleanup_job_id") or "")
    st.session_state["history_management_cleanup_job_id"] = job_id
    st.session_state["history_management_preview"] = None
    if preview_key:
        st.session_state[preview_key] = None
    _history_clear_delete_plan()
    return job_id


def _history_cleanup_job_panel(job_id: str) -> None:
    processed_key = f"history_cleanup_processed_{job_id}"
    terminal_snapshot_key = f"history_cleanup_terminal_snapshot_{job_id}"
    job = st.session_state.get(terminal_snapshot_key)
    if not isinstance(job, dict):
        response = client().history_management_cleanup_job(job_id)
        if not response.get("success"):
            st.error(friendly_error_message(response.get("error"), "清理任务状态暂时无法读取。"))
            return
        job = response.get("job") or {}
    status = str(job.get("status") or "")
    stage = str(job.get("stage") or "")
    terminal = status in {"completed", "partial", "cancelled", "failed"}
    with st.container(border=True):
        st.markdown("#### 检测记录清理进度")
        st.progress(
            float(job.get("progress") or 0.0),
            text=f"{job.get('status_label') or '处理中'} · {job.get('stage_label') or '正在处理'}",
        )
        metrics = st.columns(6)
        metrics[0].metric("待处理记录", f"{int(job.get('total_records') or 0):,}")
        metrics[1].metric("已处理记录", f"{int(job.get('processed_records') or 0):,}")
        metrics[2].metric("成功", f"{int(job.get('success_count') or 0):,}")
        metrics[3].metric("跳过", f"{int(job.get('skipped_count') or 0):,}")
        metrics[4].metric("失败", f"{int(job.get('failed_count') or 0):,}")
        metrics[5].metric("已释放", _management_bytes(job.get("released_bytes")))
        st.caption(
            f"结果文件 {int(job.get('processed_files') or 0):,}/{int(job.get('total_files') or 0):,} · "
            f"预计释放 {_management_bytes(job.get('estimated_bytes'))} · {job.get('message') or '正在处理'}"
        )
        if job.get("can_cancel") and st.button("取消尚未开始的清理任务", key=f"cancel_cleanup_{job_id}"):
            cancel_response = client().history_management_cancel_cleanup_job(job_id)
            if not cancel_response.get("success"):
                st.error(friendly_error_message(cancel_response.get("error"), "取消清理任务失败。"))
        deleted = set(job.get("deleted_record_keys") or [])
        transaction_finished = stage in {"files", "finalizing", "completed"}
        no_op_after_other_delete = (
            not deleted
            and int(job.get("success_count") or 0) == 0
            and int(job.get("skipped_count") or 0) > 0
        )
        database_committed = (
            bool(deleted) and (transaction_finished or terminal)
        ) or (
            transaction_finished and no_op_after_other_delete
        )
        refresh_key = f"history_cleanup_database_refreshed_{job_id}"
        if database_committed and not st.session_state.get(refresh_key):
            selected = _history_selected_record_ids()
            _history_store_selected_record_ids(selected - deleted)
            st.session_state[HISTORY_PENDING_DELETED_IDS_KEY] = deleted
            st.session_state["history_management_preview"] = None
            st.session_state["history_management_all_preview"] = None
            _history_clear_delete_plan()
            st.session_state[refresh_key] = True
            # Refresh records/statistics once when the database transaction is
            # durable.  File cleanup remains in this local progress fragment;
            # reaching a terminal file state does not trigger another app rerun.
            st.rerun(scope="app")
        if terminal:
            st.session_state[processed_key] = True
            st.session_state[terminal_snapshot_key] = dict(job)
            st.session_state["history_management_cleanup_job_id"] = ""
            if status == "completed":
                st.success("检测记录与受控结果文件清理完成。")
            elif status == "partial":
                st.warning("检测记录已删除，部分结果文件可在下方重试清理。")
            elif status == "cancelled":
                st.info("清理任务已在开始前取消。")
            else:
                st.error(job.get("message") or "清理任务失败，请查看审计记录。")


def _management_filters() -> dict[str, Any]:
    preset = st.session_state.get("history_management_range", "最近 30 天")
    now = datetime.now(LOCAL_TIMEZONE).replace(microsecond=0)
    start: datetime | None = None
    end: datetime | None = now
    if preset == "今天":
        start = datetime.combine(now.date(), time.min, LOCAL_TIMEZONE)
    elif preset == "最近 7 天":
        start = now - timedelta(days=7)
    elif preset == "最近 30 天":
        start = now - timedelta(days=30)
    elif preset == "最近 90 天":
        start = now - timedelta(days=90)
    elif preset == "自定义时间":
        custom_start = st.session_state.get("history_management_start")
        custom_end = st.session_state.get("history_management_end")
        if isinstance(custom_start, datetime):
            start = custom_start
        if isinstance(custom_end, datetime):
            end = min(custom_end, now)
    elif preset == "全部":
        start, end = None, None
    filters: dict[str, Any] = {
        "start_time": start.isoformat() if start else None,
        "end_time": end.isoformat() if end else None,
        "mode": st.session_state.get("history_management_mode", "all"),
        "status": st.session_state.get("history_management_status", "all"),
        "prediction": st.session_state.get("history_management_prediction", "all"),
        "feedback_state": st.session_state.get("history_management_feedback", "all"),
        "visuals_state": st.session_state.get("history_management_visuals", "all"),
        "dataset": st.session_state.get("history_management_dataset") or None,
        "category": st.session_state.get("history_management_category") or None,
        "config": st.session_state.get("history_management_config") or None,
        "config_version": st.session_state.get("history_management_config_version") or None,
        "method_mode": st.session_state.get("history_management_method_mode") or None,
        "profile_id": st.session_state.get("history_management_profile") or None,
        "model_version": st.session_state.get("history_management_model") or None,
        "memory_bank_id": st.session_state.get("history_management_memory") or None,
        "filename_keyword": st.session_state.get("history_management_filename") or None,
        "task_id": st.session_state.get("history_management_task") or None,
        "score_min": st.session_state.get("history_management_score_min"),
        "score_max": st.session_state.get("history_management_score_max"),
    }
    return {key: value for key, value in filters.items() if value not in (None, "")}


def _history_clear_expanded_job() -> None:
    for key in (
        HISTORY_EXPANDED_JOB_ID_KEY,
        HISTORY_EXPANDED_JOB_MODE_KEY,
        HISTORY_EXPANDED_JOB_TITLE_KEY,
        "history_child_task",
    ):
        st.session_state.pop(key, None)


def _history_toggle_child_items(task_id: str, mode: str, title: str) -> None:
    """Use one job-id-bound inline drill-down instead of a detached page footer."""
    normalized_task_id = str(task_id or "")
    normalized_mode = str(mode or "")
    if (
        str(st.session_state.get(HISTORY_EXPANDED_JOB_ID_KEY) or "") == normalized_task_id
        and str(st.session_state.get(HISTORY_EXPANDED_JOB_MODE_KEY) or "") == normalized_mode
    ):
        _history_clear_expanded_job()
        return
    st.session_state[HISTORY_EXPANDED_JOB_ID_KEY] = normalized_task_id
    st.session_state[HISTORY_EXPANDED_JOB_MODE_KEY] = normalized_mode
    st.session_state[HISTORY_EXPANDED_JOB_TITLE_KEY] = str(title or "检测任务")
    # Drop the legacy detached target so stale sessions cannot resurrect it.
    st.session_state.pop("history_child_task", None)


def _history_child_set_value(key: str, value: Any) -> None:
    st.session_state[key] = value


def _history_render_child_items(
    *, task_id: str, mode: str, title: str, parent_item: dict[str, Any]
) -> None:
    if not task_id or mode not in {"batch", "video"}:
        return
    prefix = f"history_child_{mode}_{task_id}"
    page_key = f"{prefix}_page"
    size_key = f"{prefix}_size"
    st.session_state.setdefault(page_key, 1)
    st.session_state.setdefault(size_key, 10)
    with st.container(border=True, key=f"history_child_panel_{mode}_{task_id}"):
        title_columns = st.columns([4.0, 0.7])
        title_columns[0].markdown(
            f"### {'本次批量检测明细' if mode == 'batch' else '已保存的帧级检测明细'}"
        )
        title_columns[0].caption(f"所属任务：{title or '检测任务'}")
        title_columns[1].button(
            "收起",
            key=f"{prefix}_close",
            width="stretch",
            on_click=_history_clear_expanded_job,
        )
        if mode == "batch":
            st.caption(
                f"总样本 {int(parent_item.get('total_items') or 0):,} · "
                f"正常 {int(parent_item.get('normal_count') or 0):,} · "
                f"缺陷 {int(parent_item.get('anomaly_count') or 0):,} · "
                f"失败 {int(parent_item.get('failed_count') or 0):,}"
            )
        else:
            full_fps = parent_item.get("full_processing_fps")
            st.caption(
                f"总帧 {int(parent_item.get('total_items') or 0):,} · "
                f"异常帧 {int(parent_item.get('anomaly_count') or 0):,} · "
                f"失败 {int(parent_item.get('failed_count') or 0):,} · "
                f"完整处理 {'-' if full_fps is None else f'{float(full_fps):.2f} FPS'}"
            )
        st.caption("仅在展开后按任务标识查询明细；父任务筛选、页码和选择状态保持不变。")
        filters = st.columns([1.7, 0.9, 0.9, 0.7])
        query = filters[0].text_input(
            "搜索", key=f"{prefix}_query",
            placeholder="文件名关键词" if mode == "batch" else "帧号",
        )
        prediction = filters[1].selectbox(
            "检测结果", ["all", "normal", "anomaly", "failed"],
            format_func={"all": "全部", "normal": "正常", "anomaly": "缺陷", "failed": "失败"}.get,
            key=f"{prefix}_prediction",
        )
        feedback_status = filters[2].selectbox(
            "反馈状态", ["all", "unsubmitted", "submitted"],
            format_func={"all": "全部", "unsubmitted": "未反馈", "submitted": "已反馈"}.get,
            key=f"{prefix}_feedback",
        )
        page_size = int(filters[3].selectbox("每页", [10, 20, 50], key=size_key))
        time_from_ms: float | None = None
        time_to_ms: float | None = None
        if mode == "video":
            time_columns = st.columns(2)
            time_from = float(time_columns[0].number_input(
                "视频时间起点（秒）", min_value=0.0, value=0.0, step=0.5,
                key=f"{prefix}_time_from",
            ))
            time_to = float(time_columns[1].number_input(
                "视频时间终点（秒，0 表示不限）", min_value=0.0, value=0.0, step=0.5,
                key=f"{prefix}_time_to",
            ))
            if time_to and time_to < time_from:
                st.warning("时间终点不能早于时间起点。")
                return
            time_from_ms = time_from * 1000.0
            time_to_ms = time_to * 1000.0 if time_to else None
        signature = (query, prediction, feedback_status, page_size, time_from_ms, time_to_ms)
        signature_key = f"{prefix}_signature"
        if st.session_state.get(signature_key) != signature:
            st.session_state[signature_key] = signature
            st.session_state[page_key] = 1
        response = client().history_management_task_items(
            task_id,
            mode=mode,
            page=int(st.session_state.get(page_key, 1)),
            page_size=page_size,
            query=query,
            prediction=prediction,
            feedback_status=feedback_status,
            time_from_ms=time_from_ms,
            time_to_ms=time_to_ms,
        )
        if not response.get("success"):
            st.error(friendly_error_message(response.get("error"), "任务明细加载失败。"))
            return
        payload = dict(response.get("task_items") or {})
        items = list(payload.get("items") or [])
        page = int(payload.get("page") or 1)
        total_pages = max(1, int(payload.get("total_pages") or 1))
        total = int(payload.get("total") or 0)
        st.session_state[page_key] = page
        st.info(str(payload.get("management_message") or "明细来自持久化检测结果。"))
        selection_key = f"{prefix}_selected_ids"
        selection_scope = f"history-child-{mode}-{task_id}"
        page_ids = [str(item.get("item_id") or "") for item in items if item.get("item_id")]
        if mode == "batch":
            sync_checkbox_states(
                st.session_state,
                state_key=selection_key,
                scope=selection_scope,
                item_ids=page_ids,
            )
            controls = st.columns([1.0, 1.0, 1.0, 2.0])
            controls[0].button(
                "全选当前页", key=f"{prefix}_select_page", width="stretch",
                on_click=update_page_selection,
                args=(st.session_state,),
                kwargs={"state_key": selection_key, "scope": selection_scope, "page_ids": page_ids, "should_select": True},
            )
            controls[1].button(
                "取消当前页", key=f"{prefix}_unselect_page", width="stretch",
                on_click=update_page_selection,
                args=(st.session_state,),
                kwargs={"state_key": selection_key, "scope": selection_scope, "page_ids": page_ids, "should_select": False},
            )
            controls[2].button(
                "清除选择", key=f"{prefix}_clear", width="stretch",
                on_click=clear_id_selection,
                args=(st.session_state,),
                kwargs={"state_key": selection_key, "scopes": [selection_scope], "visible_ids": page_ids},
            )
            controls[3].caption(f"已跨页选择 {len(read_selected_ids(st.session_state, selection_key))} 条")
        header_weights = (
            [0.32, 0.38, 1.45, 1.05, 0.62, 0.62, 0.62, 0.72, 0.72, 1.55]
            if mode == "batch"
            else [0.38, 0.52, 0.72, 0.62, 0.62, 0.62, 0.76, 0.72, 1.55]
        )
        header_labels = (
            ["选择", "序号", "样本名称", "检测时间", "结果", "分数", "阈值", "耗时", "反馈", "操作"]
            if mode == "batch"
            else ["序号", "帧号", "视频时间", "结果", "分数", "阈值", "单帧耗时", "反馈", "操作"]
        )
        header = st.columns(header_weights)
        for column, label in zip(header, header_labels):
            column.caption(label)
        for item in items:
            item_id = str(item.get("item_id") or "")
            columns = st.columns(header_weights, vertical_alignment="center")
            offset = 0
            if mode == "batch":
                columns[0].checkbox(
                    "选择", key=stable_checkbox_key(selection_scope, item_id), label_visibility="collapsed",
                    on_change=update_checkbox_selection,
                    args=(st.session_state,),
                    kwargs={"state_key": selection_key, "scope": selection_scope, "item_id": item_id},
                )
                columns[1].write(str(item.get("row_number") or "-"))
                columns[2].write(str(item.get("filename") or "检测图片"))
                created_at = _friendly_time(item.get("created_at"))
                columns[3].write(created_at[5:] if created_at != "-" else created_at)
                columns[4].write(RESULT_LABELS.get(str(item.get("prediction") or ""), "未记录"))
                columns[5].write("-" if item.get("score") is None else f"{float(item['score']):.4f}")
                columns[6].write("-" if item.get("threshold") is None else f"{float(item['threshold']):.4f}")
                columns[7].write("-" if item.get("inference_time_ms") is None else f"{float(item['inference_time_ms']):.1f} ms")
                columns[8].write(str(item.get("feedback_label") or "未反馈"))
                operation_column = columns[9]
            else:
                columns[0].write(str(item.get("row_number") or "-"))
                columns[1].write(str(item.get("frame_index") or 0))
                seconds = max(0.0, float(item.get("timestamp_ms") or 0.0) / 1000.0)
                columns[2].write(f"{int(seconds // 60):02d}:{seconds % 60:05.2f}")
                columns[3].write(RESULT_LABELS.get(str(item.get("prediction") or ""), "未记录"))
                columns[4].write("-" if item.get("score") is None else f"{float(item['score']):.4f}")
                columns[5].write("-" if item.get("threshold") is None else f"{float(item['threshold']):.4f}")
                frame_ms = item.get("frame_pipeline_ms")
                columns[6].write("-" if frame_ms is None else f"{float(frame_ms):.1f} ms")
                columns[7].write(str(item.get("feedback_label") or "未反馈"))
                operation_column = columns[8]
            actions = operation_column.columns(4)
            if actions[0].button("原图" if mode == "batch" else "原帧", key=f"{prefix}_original_{item_id}", width="stretch"):
                st.session_state[f"{prefix}_preview"] = {"item_id": item_id, "kind": "original"}
            if actions[1].button(
                "热图", key=f"{prefix}_visual_{item_id}", width="stretch",
                disabled=not bool(item.get("visualization_available")),
            ):
                st.session_state[f"{prefix}_preview"] = {"item_id": item_id, "kind": "visual"}
            with actions[2].popover("详情", width="stretch"):
                st.markdown(f"**{item.get('filename') or ('检测图片' if mode == 'batch' else '视频帧')}**")
                score_value = "-" if item.get("score") is None else f"{float(item['score']):.4f}"
                threshold_value = "-" if item.get("threshold") is None else f"{float(item['threshold']):.4f}"
                st.caption(
                    f"判定：{RESULT_LABELS.get(str(item.get('prediction') or ''), '未记录')} · "
                    f"异常分数：{score_value} · 阈值：{threshold_value}"
                )
                if mode == "video":
                    st.caption(
                        f"帧号：{int(item.get('frame_index') or 0)} · "
                        f"视频时间：{max(0.0, float(item.get('timestamp_ms') or 0.0)) / 1000.0:.3f} 秒"
                    )
            if actions[3].button("反馈", key=f"{prefix}_feedback_{item_id}", width="stretch"):
                st.session_state[f"{prefix}_feedback_item"] = item_id
        if not items:
            st.info("当前筛选条件下没有明细记录。")

        navigation = st.columns([0.8, 1.5, 0.8, 0.8])
        navigation[0].button(
            "上一页", disabled=page <= 1, key=f"{prefix}_previous", width="stretch",
            on_click=_history_child_set_value, args=(page_key, max(1, page - 1)),
        )
        navigation[1].caption(f"第 {page}/{total_pages} 页 · 共 {total} 条")
        if int(st.session_state.get(page_key, 1)) > total_pages:
            st.session_state[page_key] = total_pages
        navigation[2].number_input("页码", min_value=1, max_value=total_pages, step=1, key=page_key, label_visibility="collapsed")
        navigation[3].button(
            "下一页", disabled=page >= total_pages, key=f"{prefix}_next", width="stretch",
            on_click=_history_child_set_value, args=(page_key, min(total_pages, page + 1)),
        )

        preview = st.session_state.get(f"{prefix}_preview")
        preview_item = next((item for item in items if str(item.get("item_id") or "") == str((preview or {}).get("item_id") or "")), None)
        if preview_item:
            with st.container(border=True):
                kind = str(preview.get("kind") or "original")
                values = (
                    [("原始样本" if mode == "batch" else "原始视频帧", preview_item.get("original_image_url"))]
                    if kind == "original"
                    else [
                        ("检测叠加图", preview_item.get("result_image_url")),
                        ("异常热力图", preview_item.get("heatmap_url")),
                        ("分割 Mask", preview_item.get("mask_url")),
                    ]
                )
                available = [(label, client().artifact_url(str(url))) for label, url in values if url]
                if available:
                    preview_columns = st.columns(min(3, len(available)))
                    for index, (label, url) in enumerate(available):
                        preview_columns[index % len(preview_columns)].image(url, caption=label, width="stretch")
                else:
                    st.info("当前记录没有保存对应图像。")
                if st.button("关闭图像预览", key=f"{prefix}_close_preview"):
                    st.session_state.pop(f"{prefix}_preview", None)
        feedback_item_id = str(st.session_state.get(f"{prefix}_feedback_item") or "")
        feedback_item = next((item for item in items if str(item.get("item_id") or "") == feedback_item_id), None)
        if feedback_item:
            st.markdown("#### 提交或修改反馈")
            render_task_item_feedback(
                task_id=task_id,
                mode=mode,
                item=feedback_item,
                key_prefix=f"{prefix}_feedback_editor_{feedback_item_id.replace(':', '_')}",
            )

        if mode == "batch":
            selected = read_selected_ids(st.session_state, selection_key)
            with st.expander("删除所选样本明细", expanded=False):
                st.warning(
                    f"已选择 {len(selected)} 条。删除只移除所选子结果；父任务保留并按剩余结果重算数量与 P50/P95。"
                )
                acknowledged = st.checkbox("我已核对删除预览", key=f"{prefix}_delete_ack")
                delete_scope = f"statistics-detail-delete::{task_id}"
                delete_clicked, delete_status = render_action_button_with_feedback(
                    f"删除 {len(selected)} 条样本明细",
                    scope=delete_scope,
                    key=f"{prefix}_delete",
                    button_type="primary",
                    disabled=not selected or not acknowledged,
                )
                if delete_clicked:
                    update_action_feedback(delete_scope, "info", "正在删除所选样本明细...", delete_status)
                    result = client().history_management_delete_task_items(
                        task_id, sorted(selected), acknowledged=True,
                    )
                    if result.get("success"):
                        outcome = dict(result.get("result") or {})
                        deleted_ids = list(outcome.get("deleted_item_ids") or [])
                        remove_selected_ids(
                            st.session_state,
                            state_key=selection_key,
                            removed_ids=deleted_ids,
                            scopes=[selection_scope],
                        )
                        set_action_feedback(
                            delete_scope,
                            "success",
                            f"已删除 {int(outcome.get('deleted_count') or 0)} 条；"
                            f"受保护未删除 {int(outcome.get('protected_count') or 0)} 条。",
                        )
                        st.session_state.pop(f"{prefix}_preview", None)
                        st.session_state.pop(f"{prefix}_feedback_item", None)
                        st.rerun(scope="fragment")
                    else:
                        update_action_feedback(
                            delete_scope,
                            "error",
                            friendly_error_message(result.get("error"), "批量样本明细删除失败。"),
                            delete_status,
                        )


@st.fragment
def _render_history_management() -> None:
    """Render record filters, pagination and selection as one local region."""
    st.markdown("### 检测记录管理")
    st.caption("单图、批量与视频记录分别管理。删除前会先预览影响范围并创建可追溯快照；原始数据集、模型、记忆库、配置和文档始终保留。")
    persisted_mode = str(
        persistent_value(
            st.session_state,
            "history_management_mode",
            persistence_key="detection_history:active_mode",
            default="single",
        )
        or "single"
    )
    persisted_label = next(
        (label for label, mode in HISTORY_MODE_OPTIONS.items() if mode == persisted_mode),
        "单图检测记录",
    )
    st.session_state.setdefault(HISTORY_MODE_SELECTOR_KEY, persisted_label)
    mode_label = st.segmented_control(
        "记录类型",
        list(HISTORY_MODE_OPTIONS),
        key=HISTORY_MODE_SELECTOR_KEY,
        selection_mode="single",
    )
    active_mode = HISTORY_MODE_OPTIONS.get(str(mode_label or "单图检测记录"), "single")
    _history_switch_mode(active_mode)
    st.markdown(f"#### {next(label for label, mode in HISTORY_MODE_OPTIONS.items() if mode == active_mode)}")
    st.session_state.setdefault("history_management_range", "最近 30 天")
    st.session_state["history_management_mode"] = active_mode
    st.session_state.setdefault("history_management_status", "all")
    st.session_state.setdefault("history_management_prediction", "all")
    st.session_state.setdefault("history_management_feedback", "all")
    st.session_state.setdefault("history_management_visuals", "all")
    st.session_state.setdefault("history_management_page", 1)
    st.session_state.setdefault("history_management_page_size", 20)
    st.session_state.setdefault("history_management_sort_by", "created_at")
    st.session_state.setdefault("history_management_sort_order", "desc")
    _history_apply_pending_deleted_selection_cleanup()
    _history_selected_record_ids()
    st.session_state.setdefault("history_management_cleanup_files", True)
    st.session_state.setdefault("history_management_delete_feedback", False)

    with st.form("history_management_filter_form", border=True):
        st.markdown("#### 查询条件")
        first = st.columns([1.2, 1, 1, 1])
        first[0].selectbox("时间范围", ["今天", "最近 7 天", "最近 30 天", "最近 90 天", "全部", "自定义时间"], key="history_management_range")
        first[1].selectbox("任务状态", ["all", "success", "failed", "cancelled", "running"], format_func=lambda value: {"all": "全部状态", **STATUS_LABELS}.get(value, value), key="history_management_status")
        first[2].selectbox("检测结果", ["all", "normal", "anomaly", "failed"], format_func=lambda value: {"all": "全部结果", **RESULT_LABELS}.get(value, value), key="history_management_prediction")
        first[3].selectbox("反馈关联", ["all", "has_feedback", "no_feedback"], format_func=lambda value: {"all": "全部记录", "has_feedback": "有反馈", "no_feedback": "无反馈"}.get(value, value), key="history_management_feedback")
        if st.session_state.get("history_management_range") == "自定义时间":
            now = datetime.now(LOCAL_TIMEZONE)
            custom = st.columns(2)
            custom[0].date_input("开始日期", value=(now - timedelta(days=30)).date(), key="history_management_start_date")
            custom[1].date_input("结束日期", value=now.date(), key="history_management_end_date")
            st.session_state["history_management_start"] = datetime.combine(
                st.session_state["history_management_start_date"], time.min, LOCAL_TIMEZONE
            )
            st.session_state["history_management_end"] = datetime.combine(
                st.session_state["history_management_end_date"], time.max, LOCAL_TIMEZONE
            )
        second = st.columns(4)
        second[0].text_input("数据集（可选）", key="history_management_dataset")
        second[1].text_input("类别（可选）", key="history_management_category")
        second[2].text_input("配置标识（可选）", key="history_management_config")
        second[3].text_input("文件名关键词（可选）", key="history_management_filename")
        third = st.columns(4)
        third[0].text_input("方法模式（可选）", key="history_management_method_mode")
        third[1].text_input("profile（可选）", key="history_management_profile")
        third[2].text_input("配置版本（可选）", key="history_management_config_version")
        third[3].text_input("模型版本（可选）", key="history_management_model")
        fourth = st.columns(4)
        fourth[0].text_input("记忆库友好标识（可选）", key="history_management_memory")
        fourth[1].text_input("任务编号（可选）", key="history_management_task")
        fourth[2].selectbox("可视化结果", ["all", "generated", "not_generated"], format_func=lambda value: {"all": "全部", "generated": "已生成", "not_generated": "未生成"}.get(value, value), key="history_management_visuals")
        fourth[3].write("")
        score_columns = st.columns(2)
        score_columns[0].number_input("异常分数下限", min_value=0.0, value=None, step=0.01, key="history_management_score_min")
        score_columns[1].number_input("异常分数上限", min_value=0.0, value=None, step=0.01, key="history_management_score_max")
        filter_scope = "statistics-history::filter"
        filter_clicked, filter_status = render_form_submit_button_with_feedback(
            "应用查询条件",
            scope=filter_scope,
            button_type="primary",
            icon=":material/search:",
        )
        if filter_clicked:
            update_action_feedback(filter_scope, "info", "正在应用查询条件...", filter_status)
            st.session_state["history_management_page"] = 1
            st.session_state["history_management_filter_notice"] = (
                "查询条件已应用；跨页选择仍保留。如不再需要，请使用“清除全部选择”。"
            )

    if st.session_state.pop("history_management_filter_notice", None):
        st.info("查询条件已应用；跨页选择仍保留。如不再需要，请使用“清除全部选择”。")

    filters = _management_filters()
    summary_response = client().history_management_summary(filters)
    if not summary_response.get("success"):
        st.error(friendly_error_message(summary_response.get("error"), "检测记录管理服务暂时不可用。"))
        return
    summary = summary_response.get("summary") or {}
    metric_columns = st.columns(5)
    metric_columns[0].metric("当前类型任务", f"{int(summary.get('total_records') or 0):,}")
    metric_columns[1].metric("检测结果", f"{int(summary.get('database_results') or 0):,}")
    metric_columns[2].metric("正常结果", f"{int(summary.get('normal_results') or 0):,}")
    metric_columns[3].metric("异常结果", f"{int(summary.get('anomaly_results') or 0):,}")
    metric_columns[4].metric("失败结果", f"{int(summary.get('failed_results') or 0):,}")
    st.caption(
        f"成功任务 {int(summary.get('success_records') or 0):,} · "
        f"失败 {int(summary.get('failed_records') or 0):,} · "
        f"已取消 {int(summary.get('cancelled_records') or 0):,} · "
        f"反馈关联 {int(summary.get('feedback_records') or 0):,} · "
        f"结果文件 {_management_bytes(summary.get('result_file_bytes'))} · "
        f"数据库 {_management_bytes(summary.get('database_bytes'))} · "
        f"最早 {_friendly_time(summary.get('earliest_record'))} · 最新 {_friendly_time(summary.get('latest_record'))} · "
        "只有尚未完成的真实依赖会暂时阻止删除。"
    )
    with st.expander("为什么有些记录无法删除？", expanded=False):
        st.markdown(
            "- 🔒 **待审核反馈**：请先到反馈优化中心完成审核或撤销误提交反馈。\n"
            "- 🔒 **记忆库更新任务运行中**：请等待任务结束，或先取消相关更新任务。\n"
            "- ✅ **已完成更新或已删除来源记忆库**：不存在活动依赖时会自动解除保护。\n"
            "- 历史反馈、审计记录和已完成版本不会永久阻止检测记录删除。"
        )
        try:
            st.page_link(
                "pages/11_反馈优化中心.py",
                label="前往反馈优化中心",
                icon=":material/rule:",
            )
        except (KeyError, RuntimeError):
            st.caption("请从左侧“智能优化 → 反馈优化中心”进入。")

    records_response = client().history_management_records(
        filters,
        page=int(st.session_state.get("history_management_page", 1)),
        page_size=int(st.session_state.get("history_management_page_size", 20)),
        sort_by=str(st.session_state.get("history_management_sort_by") or "created_at"),
        sort_order=str(st.session_state.get("history_management_sort_order") or "desc"),
    )
    if not records_response.get("success"):
        st.error(friendly_error_message(records_response.get("error"), "检测记录列表加载失败。"))
        return
    records_payload = records_response.get("records") or {}
    items = records_payload.get("items") or []
    total_pages = max(1, int(records_payload.get("total_pages") or 1))
    if int(st.session_state.get("history_management_page", 1)) > total_pages:
        st.session_state["history_management_page"] = total_pages
        st.rerun(scope="fragment")
    page_controls = st.columns([0.8, 0.8, 1.2, 0.9, 0.8, 2.4])
    with page_controls[0]:
        st.number_input("每页", min_value=5, max_value=50, step=5, key="history_management_page_size")
    with page_controls[1]:
        st.number_input("页码", min_value=1, max_value=total_pages, step=1, key="history_management_page")
    with page_controls[2]:
        st.selectbox(
            "排序字段",
            ["created_at", "score", "mode", "status", "prediction", "filename"],
            format_func=lambda value: {
                "created_at": "检测时间",
                "score": "异常分数",
                "mode": "检测类型",
                "status": "任务状态",
                "prediction": "检测结果",
                "filename": "样本名称",
            }.get(value, value),
            key="history_management_sort_by",
        )
    with page_controls[3]:
        st.selectbox(
            "排序方向",
            ["desc", "asc"],
            format_func=lambda value: "降序" if value == "desc" else "升序",
            key="history_management_sort_order",
        )
    with page_controls[4]:
        if st.button("刷新列表", icon=":material/refresh:", key="history_management_refresh"):
            st.rerun(scope="fragment")
    current_page = int(records_payload.get("page") or st.session_state.get("history_management_page", 1))
    current_page_size = int(records_payload.get("page_size") or st.session_state.get("history_management_page_size", 20))
    first_display_index = (current_page - 1) * current_page_size + 1 if items else 0
    last_display_index = first_display_index + len(items) - 1 if items else 0
    page_controls[5].caption(
        f"共 {int(records_payload.get('total') or 0):,} 条 · 当前显示 {first_display_index:,}–{last_display_index:,} · "
        f"第 {current_page}/{total_pages} 页"
    )
    visible_parent_ids = {
        str(item.get("record_key") or "") for item in items if item.get("record_key")
    }
    expanded_job_id = str(st.session_state.get(HISTORY_EXPANDED_JOB_ID_KEY) or "")
    expanded_job_mode = str(st.session_state.get(HISTORY_EXPANDED_JOB_MODE_KEY) or "")
    if expanded_job_id and (
        expanded_job_mode != active_mode or expanded_job_id not in visible_parent_ids
    ):
        # Page/filter/mode changes and parent deletion must not leave detached
        # details from a task that is no longer in the visible result set.
        _history_clear_expanded_job()
        expanded_job_id = ""
        expanded_job_mode = ""
    selected = _history_selected_record_ids()
    if items:
        current_page_record_ids = {
            str(item.get("record_key")) for item in items if item.get("record_key")
        }
        current_page_selectable_ids = {
            str(item.get("record_key"))
            for item in items
            if item.get("record_key") and not bool(item.get("protected"))
        }
        protected_on_page = current_page_record_ids - current_page_selectable_ids
        if selected.intersection(protected_on_page):
            selected.difference_update(protected_on_page)
            _history_store_selected_record_ids(selected)

        current_page_selected_ids = selected.intersection(current_page_selectable_ids)
        current_page_selected_count = len(current_page_selected_ids)
        current_page_selectable_count = len(current_page_selectable_ids)
        page_fully_selected = bool(current_page_selectable_ids) and (
            current_page_selected_count == current_page_selectable_count
        )
        if page_fully_selected:
            page_selection_label = "取消当前页全选"
        elif current_page_selected_count:
            page_selection_label = "补选当前页"
        else:
            page_selection_label = "全选当前页"

        _history_sync_checkbox_states(current_page_record_ids, selected)
        selection_actions = st.columns([1, 1, 2])
        selection_actions[0].button(
            page_selection_label,
            key="history_management_toggle_page_selection",
            disabled=not current_page_selectable_ids,
            on_click=_history_set_page_selection,
            args=(tuple(sorted(current_page_selectable_ids)), not page_fully_selected),
        )
        selection_actions[1].button(
            "清除全部选择",
            key="history_management_clear_selection",
            disabled=not selected,
            on_click=_history_clear_all_selections,
            args=(tuple(sorted(current_page_record_ids)),),
        )
        if not current_page_selectable_ids:
            selection_actions[2].caption("当前页没有可选择的记录。")
        else:
            selection_actions[2].caption(
                f"当前页已选 {current_page_selected_count:,}/{current_page_selectable_count:,} 条，"
                f"共选择 {len(selected):,} 条记录。"
            )
        header = st.columns([0.45, 0.45, 1.25, 2.0, 1.45, 1.45, 1.25, 1.0])
        header_labels = {
            "single": ("序号", "选择", "检测时间", "图片名称", "判定 / 分数", "目标域版本", "反馈 / 结果", "状态"),
            "batch": ("序号", "选择", "开始时间", "任务名称", "数量统计", "性能统计", "目标域版本", "状态"),
            "video": ("序号", "选择", "检测时间", "视频名称", "帧统计", "速度 / 时长", "目标域版本", "状态"),
        }[active_mode]
        for column, label in zip(
            header,
            header_labels,
        ):
            column.markdown(f"**{label}**")
        for display_index, item in enumerate(items, start=first_display_index):
            record_key = str(item.get("record_key") or "")
            profile_display = (
                "原目标域资源已删除"
                if item.get("resource_deleted")
                else (item.get("profile_name") or "默认运行方案")
            )
            with st.container(border=True):
                choice_columns = st.columns([0.45, 0.45, 1.25, 2.0, 1.45, 1.45, 1.25, 1.0])
                choice_columns[0].markdown(f"**{display_index:,}**")
                choice_columns[1].checkbox(
                    "选择",
                    label_visibility="collapsed",
                    key=_history_checkbox_key(record_key),
                    disabled=bool(item.get("protected")),
                    help=item.get("protected_reason") or None,
                    on_change=_history_record_selection_changed,
                    args=(record_key,),
                )
                choice_columns[2].write(_friendly_time(item.get("created_at")))
                choice_columns[2].caption(item.get("mode_label") or "-")
                choice_columns[3].write(item.get("filename") or "未命名")
                choice_columns[3].caption(f"{item.get('dataset') or '-'} / {item.get('category') or '-'}")
                score_text = "-" if item.get("score") is None else f"{float(item.get('score')):.4f}"
                threshold_text = "-" if item.get("threshold") is None else f"{float(item.get('threshold')):.4f}"
                if active_mode == "single":
                    choice_columns[4].write(item.get("prediction_label") or "-")
                    choice_columns[4].caption(f"分数 {score_text} · 阈值 {threshold_text}")
                    choice_columns[5].write(profile_display)
                    choice_columns[5].caption(f"{_history_method_label(item.get('method_mode'))} · 配置版本 {item.get('config_version') or '未记录'}")
                    choice_columns[6].write(item.get("feedback_state") or "未反馈")
                    choice_columns[6].caption(item.get("visual_state") or "未生成或已清理")
                elif active_mode == "batch":
                    choice_columns[4].write(f"总计 {int(item.get('total_items') or 0):,}")
                    choice_columns[4].caption(f"正常 {int(item.get('normal_count') or 0):,} · 缺陷 {int(item.get('anomaly_count') or 0):,} · 失败 {int(item.get('failed_count') or 0):,}")
                    p50 = "-" if item.get("p50_ms") is None else f"{float(item['p50_ms']):.1f}ms"
                    p95 = "-" if item.get("p95_ms") is None else f"{float(item['p95_ms']):.1f}ms"
                    speed = "-" if item.get("throughput") is None else f"{float(item['throughput']):.2f} 张/秒"
                    choice_columns[5].write(f"P50 {p50}")
                    choice_columns[5].caption(f"P95 {p95} · {speed}")
                    choice_columns[6].write(profile_display)
                    choice_columns[6].caption(_history_method_label(item.get("method_mode")))
                else:
                    choice_columns[4].write(f"总帧 {int(item.get('total_items') or 0):,}")
                    choice_columns[4].caption(f"异常 {int(item.get('anomaly_count') or 0):,} · 失败 {int(item.get('failed_count') or 0):,}")
                    pure_fps = "-" if item.get("pure_inference_fps") is None else f"{float(item['pure_inference_fps']):.2f} FPS"
                    full_fps = "-" if item.get("full_processing_fps") is None else f"{float(item['full_processing_fps']):.2f} FPS"
                    duration = "-" if item.get("duration_seconds") is None else f"{float(item['duration_seconds']):.1f} 秒"
                    choice_columns[5].write(f"纯推理 {pure_fps}")
                    choice_columns[5].caption(f"完整处理 {full_fps} · 时长 {duration}")
                    choice_columns[6].write(profile_display)
                    choice_columns[6].caption(item.get("visual_state") or "结果视频未生成或已清理")
                choice_columns[7].write(item.get("status_label") or "-")
                choice_columns[7].caption(item.get("management_state") or ("🔒 受保护" if item.get("protected") else "— 无保护"))
                if active_mode in {"batch", "video"}:
                    is_expanded = expanded_job_id == record_key and expanded_job_mode == active_mode
                    action_columns = st.columns([1.25, 4.75])
                    action_columns[0].button(
                        "收起检测明细" if is_expanded else "查看检测明细",
                        key=f"history_toggle_child_{active_mode}_{record_key}",
                        type="primary" if is_expanded else "secondary",
                        width="stretch",
                        on_click=_history_toggle_child_items,
                        args=(record_key, active_mode, str(item.get("filename") or "检测任务")),
                    )
                    action_columns[1].caption(
                        "当前任务明细已展开，下一条父任务位于明细区域之后。"
                        if is_expanded else "明细将在本任务下方展开；一次只展开一个任务。"
                    )
            with st.expander(f"查看详情 · {item.get('filename') or '未命名'}", expanded=False):
                detail = st.columns(4)
                detail[0].metric("数据集/类别", f"{item.get('dataset') or '-'} / {item.get('category') or '-'}")
                detail[1].metric("异常分数", "-" if item.get("score") is None else f"{float(item.get('score')):.4f}")
                detail[2].metric("结果数量", f"{int(item.get('result_count') or 0):,}")
                detail[3].metric("反馈状态", item.get("feedback_state") or "未反馈")
                st.caption(
                    f"profile：{'原目标域资源已删除' if item.get('resource_deleted') else (item.get('profile_name') or '-')} · 配置版本：{item.get('config_version') or '-'} · "
                    f"方法模式：{_history_method_label(item.get('method_mode'))} · "
                    f"阈值：{item.get('threshold') if item.get('threshold') is not None else '-'} · "
                    f"结果文件：{item.get('visual_state') or '未生成或已清理'}"
                )
                if item.get("protected"):
                    st.markdown("**为什么受保护？**")
                    for protection in list(item.get("protection_details") or []):
                        st.warning(
                            f"{protection.get('label') or '🔒 受保护'}\n\n"
                            f"{protection.get('message') or ''}\n\n"
                            f"**解除方式：** {protection.get('resolution') or '请先完成相关操作。'}"
                        )
                    protection_actions = {
                        str(protection.get("action") or "")
                        for protection in list(item.get("protection_details") or [])
                    }
                    if "feedback_review" in protection_actions:
                        try:
                            st.page_link(
                                "pages/11_反馈优化中心.py",
                                label="前往反馈优化中心完成审核",
                                icon=":material/rule:",
                            )
                        except (KeyError, RuntimeError):
                            st.caption("请从左侧“智能优化 → 反馈优化中心”完成审核。")
                    if "feedback_update" in protection_actions:
                        try:
                            st.page_link(
                                "pages/11_反馈优化中心.py",
                                label="查看反馈更新任务",
                                icon=":material/update:",
                            )
                        except (KeyError, RuntimeError):
                            st.caption("请从左侧“智能优化 → 反馈优化中心”查看更新任务。")
                elif item.get("resource_deleted"):
                    st.success("来源 Memory Bank 已删除，且不存在活动反馈或更新任务，当前记录已自动解除保护。")
                for warning in list(item.get("management_warnings") or []):
                    st.warning(f"数据状态提示：{warning}")
                with st.popover("技术信息"):
                    st.caption(
                        f"任务标识：{item.get('task_id_display') or '-'} · "
                        f"profile 标识：{'原目标域资源已删除' if item.get('resource_deleted') else (item.get('profile_id') or '-')} · "
                        f"记忆库标识：{'原目标域资源已删除' if item.get('resource_deleted') else (item.get('memory_bank') or '-')} · "
                        f"模型版本：{friendly_model_name(item.get('model_version'))} · 设备：{item.get('device') or '-'} · "
                        f"Runtime Revision：{item.get('runtime_revision_display') or '历史记录未记录'}"
                    )
            if expanded_job_id == record_key and expanded_job_mode == active_mode:
                _history_render_child_items(
                    task_id=record_key,
                    mode=active_mode,
                    title=str(item.get("filename") or "检测任务"),
                    parent_item=dict(item),
                )
        selected = _history_selected_record_ids()
    else:
        st.info("当前筛选条件下没有检测记录。")

    st.divider()
    st.markdown("#### 删除与清理")
    cleanup_scope_text = {
        "single": "单图删除预览会列出数据库结果、可视化文件及反馈关联；受保护反馈不会被静默删除。",
        "batch": "批量任务删除预览会同时统计其全部单张结果和结果文件，并明确反馈关联。",
        "video": "视频任务删除预览会同时统计帧级结果、关键帧、结果视频、时间轴记录及视频反馈关联。",
    }[active_mode]
    st.caption(f"{cleanup_scope_text} 原始输入数据集、模型、记忆库、配置和文档始终保留。")
    cleanup_files = st.checkbox(
        "同时清理关联结果文件（默认开启；关闭时仅删除数据库历史记录）",
        key="history_management_cleanup_files",
    )
    delete_feedback = st.checkbox(
        "高级操作：同时删除普通反馈记录（待审核候选、Mask 标注和优化引用仍受保护）",
        key="history_management_delete_feedback",
        help="默认不级联删除反馈。启用后预览会单独列出可删除和受保护的反馈；已进入候选池或优化流程的记录仍不会删除。",
    )
    if delete_feedback:
        st.warning("已启用反馈级联删除：仅普通反馈可被删除，待审核候选、人工 Mask 和优化引用仍会保留并受保护。")
    cleanup_job_id = str(st.session_state.get("history_management_cleanup_job_id") or "")
    if not cleanup_job_id:
        recent_jobs_response = client().history_management_cleanup_jobs(5)
        if recent_jobs_response.get("success"):
            recent_jobs = recent_jobs_response.get("jobs") or []
            active_job = next(
                (item for item in recent_jobs if str(item.get("status") or "") in {"queued", "running"}),
                None,
            )
            if active_job:
                cleanup_job_id = str(active_job.get("cleanup_job_id") or "")
                st.session_state["history_management_cleanup_job_id"] = cleanup_job_id
    if cleanup_job_id:
        processed_key = f"history_cleanup_processed_{cleanup_job_id}"
        if st.session_state.get(processed_key):
            st.fragment(_history_cleanup_job_panel)(cleanup_job_id)
        else:
            st.fragment(run_every=1.0)(_history_cleanup_job_panel)(cleanup_job_id)
    action_columns = st.columns([1, 1, 1, 2], vertical_alignment="center")
    if action_columns[0].button("预览删除已选记录", disabled=not selected, key="history_preview_selected"):
        response = client().history_management_delete_preview(
            {"filters": filters, "scope": "selected", "record_keys": sorted(selected), "delete_feedback": delete_feedback, "cleanup_files": cleanup_files}
        )
        if response.get("success"):
            preview_result = response.get("preview") or {}
            _history_store_delete_plan(preview_result, "records")
            st.session_state["history_management_preview"] = preview_result
            st.session_state["history_delete_ack"] = False
        else:
            _history_clear_delete_plan()
            st.session_state["history_management_preview"] = {"error": response.get("error")}
    if action_columns[1].button("预览删除当前筛选结果", disabled=not bool(items), key="history_preview_filtered"):
        response = client().history_management_delete_preview(
            {"filters": filters, "scope": "current_filtered", "record_keys": [], "delete_feedback": delete_feedback, "cleanup_files": cleanup_files}
        )
        if response.get("success"):
            preview_result = response.get("preview") or {}
            _history_store_delete_plan(preview_result, "records")
            st.session_state["history_management_preview"] = preview_result
            st.session_state["history_delete_ack"] = False
        else:
            _history_clear_delete_plan()
            st.session_state["history_management_preview"] = {"error": response.get("error")}
    retry_scope = "statistics-cleanup::retry"
    with action_columns[3]:
        retry_status = st.empty()
    render_action_feedback(retry_scope, retry_status)
    if action_columns[2].button("重试失败的文件清理", key="history_retry_cleanup", width="stretch"):
        update_action_feedback(retry_scope, "info", "正在重试失败的文件清理...", retry_status)
        response = client().history_management_retry_cleanup([])
        if response.get("success"):
            update_action_feedback(
                retry_scope,
                "success",
                (response.get("result") or {}).get("message") or "清理重试已完成。",
                retry_status,
            )
        else:
            update_action_feedback(
                retry_scope,
                "error",
                friendly_error_message(response.get("error"), "清理重试失败。"),
                retry_status,
            )

    preview = (
        st.session_state.get(HISTORY_DELETE_PREVIEW_SUMMARY_KEY)
        if st.session_state.get(HISTORY_DELETE_PLAN_MODE_KEY) == "records"
        else st.session_state.get("history_management_preview")
    )
    if preview:
        if preview.get("error"):
            st.error(friendly_error_message(preview.get("error"), "删除预览失败。"))
        else:
            st.warning(
                f"预览：将处理 {int(preview.get('deletable_records') or 0):,} 条记录、"
                f"{int(preview.get('result_count') or 0):,} 个结果、{int(preview.get('file_count') or 0):,} 个结果文件，"
                f"预计清理 {_management_bytes(preview.get('estimated_reclaim_bytes'))}。"
            )
            preview_modes = preview.get("mode_counts") or {}
            st.caption(
                f"单图 {int(preview_modes.get('single', 0)):,} · 批量 {int(preview_modes.get('batch', 0)):,} · "
                f"视频 {int(preview_modes.get('video', 0)):,} · 正常结果 {int(preview.get('normal_result_count') or 0):,} · "
                f"异常结果 {int(preview.get('anomaly_result_count') or 0):,} · 失败结果 {int(preview.get('failed_result_count') or 0):,} · "
                f"反馈关联 {int(preview.get('feedback_count') or 0):,} · 候选 {int(preview.get('feedback_candidate_count') or 0):,} · "
                f"Mask {int(preview.get('mask_annotation_count') or 0):,} · 优化任务 {int(preview.get('optimization_job_count') or 0):,} · "
                f"运行中 {int(preview.get('current_running_count') or 0):,}"
            )
            if int(preview.get("protected_records") or 0):
                st.info(f"其中 {int(preview.get('protected_records') or 0):,} 条记录受保护，将不会删除。")
            st.caption(
                f"删除计划已冻结 · 创建时间 {_friendly_time(preview.get('plan_created_at'))} · "
                f"有效期 {int(preview.get('expires_in_seconds') or 0) // 60} 分钟。"
            )
            response = render_delete_preview_confirmation(
                preview=preview,
                title="检测记录删除预览",
                action_label="删除可删除的检测记录",
                object_label="检测记录",
                acknowledgement_key="history_delete_ack",
                button_key="history_execute_delete",
                execute_callback=lambda payload: client().history_management_delete(payload),
                extra_summary=[("原始数据集/模型", "保留"), ("Memory Bank/配置", "保留")],
            )
            if response is not None:
                if response.get("success"):
                    job = response.get("result") or {}
                    job_id = _history_register_cleanup_job(job)
                    st.fragment(run_every=1.0)(_history_cleanup_job_panel)(job_id)
                else:
                    st.error(friendly_error_message(response.get("error"), "检测记录删除失败。"))

    mode_short_label = {"single": "单图", "batch": "批量", "video": "视频"}[active_mode]
    mode_clear_label = f"{mode_short_label}检测记录"
    all_preview_key = f"history_management_all_preview_{active_mode}"
    all_ack_key = f"history_clear_all_ack_{active_mode}"
    with st.expander(f"清空全部{mode_clear_label}（高风险操作）", expanded=False):
        feedback_note = "普通反馈会按高级选项处理，待审核候选、Mask 标注和优化引用仍会保留。" if delete_feedback else "含反馈、待审核候选或被优化任务引用的记录会保留。"
        st.warning(
            f"本操作只会清空所有可删除的{mode_clear_label}及其受控结果文件，不会影响另外两类检测记录；"
            f"正在运行的任务会保留。{feedback_note}"
        )
        if st.button(f"清空全部{mode_clear_label}", key=f"history_preview_all_{active_mode}"):
            response = client().history_management_delete_all_preview(
                {
                    "filters": {"mode": active_mode},
                    "delete_feedback": delete_feedback,
                    "cleanup_files": cleanup_files,
                }
            )
            if response.get("success"):
                preview_result = response.get("preview") or {}
                _history_store_delete_plan(preview_result, "all")
                st.session_state[all_preview_key] = preview_result
            else:
                _history_clear_delete_plan()
                st.session_state[all_preview_key] = {"error": response.get("error")}
            st.session_state[all_ack_key] = False
        all_preview = st.session_state.get(all_preview_key)
        if all_preview:
            if all_preview.get("error"):
                st.error(friendly_error_message(all_preview.get("error"), "全部清空预览失败。"))
            else:
                response = _render_clear_all_confirmation(
                    active_mode=active_mode,
                    record_type=mode_clear_label,
                    record_short_name=mode_short_label,
                    preview=all_preview,
                    execute_callback=client().history_management_delete_all,
                )
                if response is not None:
                    if response.get("success"):
                        job = response.get("result") or {}
                        job_id = _history_register_cleanup_job(job, preview_key=all_preview_key)
                        st.fragment(run_every=1.0)(_history_cleanup_job_panel)(job_id)
                    else:
                        st.error(friendly_error_message(response.get("error"), "全部清空失败。"))

    audit_response = client().history_management_audit(30)
    if audit_response.get("success"):
        with st.expander("删除审计记录", expanded=False):
            audit_rows = audit_response.get("audit") or []
            if audit_rows:
                st.dataframe(audit_rows, hide_index=True, width="stretch")
            else:
                st.caption("尚无删除审计记录。")


ANALYTICS_PAGE_TABS = (
    ("dashboard", "统计看板"),
    ("history", "检测记录管理"),
)


def _analytics_active_page_tab() -> str:
    """Keep dashboard/history selection stable across button-triggered reruns."""
    valid_keys = {key for key, _ in ANALYTICS_PAGE_TABS}
    requested = str(st.query_params.get("analytics_page_tab") or "").strip()
    if requested in valid_keys and "analytics_page_tab" not in st.session_state:
        st.session_state["analytics_page_tab"] = requested
    st.session_state.setdefault("analytics_page_tab", "dashboard")
    selected = st.segmented_control(
        "统计分析分区",
        options=[key for key, _ in ANALYTICS_PAGE_TABS],
        format_func=lambda value: dict(ANALYTICS_PAGE_TABS).get(value, value),
        selection_mode="single",
        key="analytics_page_tab",
    )
    active = str(selected or st.session_state.get("analytics_page_tab") or "dashboard")
    if active in valid_keys and str(st.query_params.get("analytics_page_tab") or "") != active:
        st.query_params["analytics_page_tab"] = active
    return active if active in valid_keys else "dashboard"


configure_page("statistics")
sidebar_runtime_controls("statistics")
# Discard the legacy page-level flash slot.  Delete/cleanup actions now own a
# status placeholder beside their initiating control.
st.session_state.pop("_action_feedback::statistics::page-notice", None)
_install_styles()
_hero()

active_page_tab = _analytics_active_page_tab()
if active_page_tab == "history":
    _render_history_management()

if active_page_tab == "dashboard":
    st.session_state.setdefault("analytics_range_preset", "最近 7 天")
    st.session_state.setdefault("analytics_module_label", "单图检测")
    st.session_state.setdefault("analytics_refresh_token", 0)
    st.session_state.setdefault("analytics_detail_page", 1)
    st.session_state.setdefault("analytics_detail_page_size", 20)
    st.session_state.setdefault("analytics_detail_sort", "created_at")
    st.session_state.setdefault("analytics_detail_order", "desc")

    start_time, end_time = _render_time_controls()
    module_label = st.segmented_control(
        "分析模块",
        list(MODULE_LABELS),
        selection_mode="single",
        key="analytics_module_label",
    )
    module = MODULE_LABELS.get(str(module_label), "single")

    if st.session_state.get("analytics_last_module") != module:
        st.session_state["analytics_last_module"] = module
        for key in (
            "analytics_filter_dataset", "analytics_filter_category", "analytics_filter_config", "analytics_filter_device",
            "analytics_filter_model", "analytics_filter_memory", "analytics_filter_prediction", "analytics_filter_status",
        ):
            st.session_state[key] = "全部"
        st.session_state["analytics_detail_page"] = 1
        st.session_state["analytics_detail_sort"] = "created_at"

    filter_signature = (
        module,
        start_time.isoformat() if start_time else "all",
        end_time.isoformat(),
        *(
            str(st.session_state.get(key, "全部"))
            for key in (
                "analytics_filter_dataset", "analytics_filter_category", "analytics_filter_config", "analytics_filter_device",
                "analytics_filter_model", "analytics_filter_memory", "analytics_filter_prediction", "analytics_filter_status",
            )
        ),
    )
    if st.session_state.get("analytics_filter_signature") != filter_signature:
        st.session_state["analytics_filter_signature"] = filter_signature
        st.session_state["analytics_detail_page"] = 1

    query_params = {
        "start_time": start_time.isoformat() if start_time else None,
        "end_time": end_time.isoformat(),
        "dataset": _filter_value("analytics_filter_dataset"),
        "category": _filter_value("analytics_filter_category"),
        "config": _filter_value("analytics_filter_config"),
        "device": _filter_value("analytics_filter_device"),
        "model_version": _filter_value("analytics_filter_model"),
        "memory_bank_id": _filter_value("analytics_filter_memory"),
        "prediction": _filter_value("analytics_filter_prediction"),
        "task_status": _filter_value("analytics_filter_status"),
        "page": int(st.session_state.get("analytics_detail_page", 1)),
        "page_size": int(st.session_state.get("analytics_detail_page_size", 20)),
        "sort_by": str(st.session_state.get("analytics_detail_sort", "created_at")),
        "sort_order": str(st.session_state.get("analytics_detail_order", "desc")),
    }

    with st.spinner("正在汇总真实历史任务..."):
        response = _cached_dashboard(
            module,
            tuple(sorted(query_params.items(), key=lambda item: item[0])),
            int(st.session_state.get("analytics_refresh_token", 0)),
            _cached_statistics_revision(),
        )
    if not response.get("ok"):
        st.error(friendly_error_message(response.get("error"), "统计数据加载失败。"))
        st.stop()
    dashboard = response.get("dashboard") or {}

    _render_filters(dashboard.get("filter_options") or {})
    start_text = format_beijing_datetime(start_time, fallback="最早记录")
    end_text = format_beijing_datetime(end_time)
    st.markdown(
        f'<div class="analytics-range">当前统计区间：<b>{start_text}</b> 至 <b>{end_text}</b>（Asia/Shanghai，开始与结束边界均包含）</div>',
        unsafe_allow_html=True,
    )

    if int((dashboard.get("summary") or {}).get("task_count", 0)) == 0:
        st.info("当前时间区间和筛选条件下暂无真实任务数据。可扩大时间范围或清除筛选条件。")
    else:
        if module == "single":
            _single_dashboard(dashboard)
        elif module == "batch":
            _batch_dashboard(dashboard)
        else:
            _video_dashboard(dashboard)

    # Feedback is summarized here without copying the detailed review UI.  The
    # candidate center remains the place for labels, Mask revisions and version
    # operations; this card only reads the same persisted counters.
    previous_feedback_revision = st.session_state.get("statistics_feedback_revision")
    feedback_revision_response = client().feedback_revision(
        int(previous_feedback_revision) if previous_feedback_revision is not None else None
    )
    feedback_revision = int(previous_feedback_revision or 0)
    if feedback_revision_response.get("success"):
        feedback_revision = int(feedback_revision_response.get("feedback_revision") or 0)
        if previous_feedback_revision is not None and int(previous_feedback_revision) != feedback_revision:
            clear_feedback_caches()
        st.session_state["statistics_feedback_revision"] = feedback_revision
    feedback_response = cached_feedback_statistics("", feedback_revision)
    if feedback_response.get("success"):
        feedback_stats = feedback_response.get("statistics") or {}
        with st.container(border=True):
            st.markdown("#### 用户反馈概览")
            feedback_cols = st.columns(5)
            feedback_cols[0].metric("反馈总数", _count(feedback_stats.get("total")))
            feedback_cols[1].metric("待审核", _count(feedback_stats.get("pending_count")))
            feedback_cols[2].metric("审核通过", _count(feedback_stats.get("approved_count")))
            feedback_cols[3].metric("已撤销", _count(feedback_stats.get("withdrawn_count")))
            feedback_cols[4].metric("已用于更新", _count(feedback_stats.get("used_count")))
            if _count(feedback_stats.get("total")) == 0:
                st.caption("暂无反馈数据。")
            else:
                st.caption("反馈不会覆盖原始检测记录；详情审核、Mask 修订、离线验证和版本回滚请前往“反馈优化中心”。")
    _render_details(dashboard, module)
