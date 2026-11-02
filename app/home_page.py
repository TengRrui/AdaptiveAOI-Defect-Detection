from __future__ import annotations

import math
import time as time_module
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import streamlit as st

from performance_cache import api_client as client
from dashboard_components import install_dashboard_styles, render_performance_cards, render_stat_cards, render_system_hero
from friendly_display import friendly_error_message, friendly_value
from navigation import get_page
from time_utils import format_beijing_datetime
from ui_common import format_value, sidebar_runtime_controls


LOCAL_TIMEZONE = ZoneInfo("Asia/Shanghai")
OVERVIEW_CACHE_BUCKET_SECONDS = 15
MODULE_LABELS = {"single": "单图检测", "batch": "批量检测", "video": "视频检测"}
RESULT_LABELS = {"normal": "正常", "anomaly": "异常", "failed": "失败"}
STATUS_LABELS = {"success": "成功", "failed": "失败", "cancelled": "已取消", "running": "运行中"}
def _result_colors(alt):
    return alt.Scale(domain=["正常", "异常", "失败"], range=["#168AAD", "#D1495B", "#6C757D"])


def _series_colors(alt):
    return alt.Scale(domain=["任务数", "正常结果", "异常结果"], range=["#4361EE", "#168AAD", "#D1495B"])


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


def _duration(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "-"
    return f"{number / 1000:.2f} s" if number >= 1000 else f"{number:.1f} ms"


def _fps(value: Any) -> str:
    number = _number(value)
    return "-" if number is None else f"{number:.2f} FPS"


def _throughput(value: Any) -> str:
    number = _number(value)
    return "-" if number is None else f"{number:.2f} 张/秒"


def _score(value: Any) -> str:
    number = _number(value)
    return "-" if number is None else f"{number:.4f}"


def _friendly_time(value: Any) -> str:
    return format_beijing_datetime(value, fallback="-")


def _install_styles() -> None:
    install_dashboard_styles()
    st.markdown(
        """
        <style>
          .overview-range { padding:.55rem .72rem; margin:.28rem 0 .68rem; border:1px solid rgba(112,139,160,.18); border-radius:10px; background:rgba(251,253,255,.9); color:#486176; font-size:.78rem; }
          .overview-section { display:flex; align-items:center; gap:.45rem; color:#183b56; font-size:1.08rem; font-weight:800; margin:.82rem 0 .08rem; }
          .overview-section::before { content:""; width:.24rem; height:1rem; border-radius:999px; background:linear-gradient(#22d3ee,#0e7490); box-shadow:0 0 10px rgba(34,211,238,.25); }
          .overview-section-copy { color:#718096; font-size:.88rem; margin-bottom:.48rem; }
          div[data-testid="stAltairChart"] { padding:.35rem; border:1px solid rgba(112,139,160,.16); border-radius:12px; background:#fff; }
          .overview-module-title { color:#173a56; font-size:1.06rem; font-weight:780; }
          .overview-module-copy { color:#718096; font-size:.82rem; margin-bottom:.45rem; }
          .st-key-overview_time_toolbar { margin:.12rem 0 .08rem; padding:.48rem .58rem .2rem; border:1px solid rgba(112,139,160,.16); border-radius:12px; background:rgba(255,255,255,.82); }
          .st-key-overview_time_toolbar [data-testid="stSegmentedControl"] { margin:0 !important; }
          .st-key-overview_time_toolbar [data-testid="stSegmentedControl"] button { min-height:2.15rem; font-size:.76rem; }
          .overview-chart-title { color:#294861; font-size:.85rem; font-weight:800; margin-bottom:.08rem; }
          .overview-chart-copy { color:#7c8c9d; font-size:.68rem; margin-bottom:.24rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _cacheable_overview_now(now: datetime | None = None) -> datetime:
    """Return a stable short-lived end time for the default dashboard range.

    Streamlit reruns a fragment for every widget interaction.  Using the exact
    current second as part of the statistics cache key made an unrelated click
    bypass the 15-second dashboard cache and reload all three statistics
    modules.  A small time bucket keeps ordinary interactions on the same
    snapshot; explicit refreshes and statistics revisions still have their own
    cache-busting keys.
    """

    current = now or datetime.now(LOCAL_TIMEZONE)
    if current.tzinfo is None:
        current = current.replace(tzinfo=LOCAL_TIMEZONE)
    else:
        current = current.astimezone(LOCAL_TIMEZONE)
    current = current.replace(microsecond=0)
    return current - timedelta(seconds=current.second % OVERVIEW_CACHE_BUCKET_SECONDS)


def _range_values() -> tuple[datetime | None, datetime]:
    current_now = datetime.now(LOCAL_TIMEZONE).replace(microsecond=0)
    cacheable_now = _cacheable_overview_now(current_now)
    preset = st.session_state.get("overview_range_preset", "最近 7 天")
    if preset == "今天":
        return datetime.combine(cacheable_now.date(), time.min, LOCAL_TIMEZONE), cacheable_now
    if preset == "最近 30 天":
        return cacheable_now - timedelta(days=30), cacheable_now
    if preset == "最近 90 天":
        return cacheable_now - timedelta(days=90), cacheable_now
    if preset == "全部":
        return None, cacheable_now
    if preset == "自定义时间":
        start = st.session_state.get("overview_custom_start_applied")
        end = st.session_state.get("overview_custom_end_applied")
        if isinstance(start, datetime) and isinstance(end, datetime):
            return start, min(end, current_now)
    return cacheable_now - timedelta(days=7), cacheable_now


def _time_controls() -> tuple[datetime | None, datetime]:
    st.session_state.setdefault("overview_range_preset", "最近 7 天")
    with st.container(key="overview_time_toolbar", border=False):
        left, right = st.columns([5, 1], vertical_alignment="bottom")
        with left:
            st.segmented_control(
                "统计时间",
                ["今天", "最近 7 天", "最近 30 天", "最近 90 天", "全部", "自定义时间"],
                selection_mode="single",
                key="overview_range_preset",
            )
        with right:
            if st.button("刷新数据", icon=":material/refresh:", width="stretch", key="overview_refresh"):
                st.session_state["overview_refresh_token"] = int(st.session_state.get("overview_refresh_token", 0)) + 1
    if st.session_state["overview_range_preset"] == "自定义时间":
        now = datetime.now(LOCAL_TIMEZONE).replace(microsecond=0)
        default_start = st.session_state.get("overview_custom_start_applied", now - timedelta(days=7))
        default_end = st.session_state.get("overview_custom_end_applied", now)
        with st.container(border=True):
            fields = st.columns(4)
            start_date = fields[0].date_input("开始日期", value=default_start.date(), key="overview_start_date")
            start_clock = fields[1].time_input("开始时间", value=default_start.time(), key="overview_start_time")
            end_date = fields[2].date_input("结束日期", value=default_end.date(), key="overview_end_date")
            end_clock = fields[3].time_input("结束时间", value=default_end.time(), key="overview_end_time")
            actions = st.columns([1, 1, 4])
            if actions[0].button("应用", type="primary", width="stretch", key="overview_apply_time"):
                start = datetime.combine(start_date, start_clock, LOCAL_TIMEZONE)
                end = datetime.combine(end_date, end_clock, LOCAL_TIMEZONE)
                if start > end:
                    st.error("开始时间不能晚于结束时间。")
                elif end > now:
                    st.error("结束时间不能晚于当前时间。")
                else:
                    st.session_state["overview_custom_start_applied"] = start
                    st.session_state["overview_custom_end_applied"] = end
            if actions[1].button("重置", width="stretch", key="overview_reset_time"):
                st.session_state["overview_custom_start_applied"] = now - timedelta(days=7)
                st.session_state["overview_custom_end_applied"] = now
    return _range_values()


@st.cache_data(ttl=15, show_spinner=False)
def _dashboard(module: str, start_time: str | None, end_time: str, refresh_token: int,
               statistics_revision: int) -> dict[str, Any]:
    from service.statistics_service import load_statistics_dashboard

    del refresh_token, statistics_revision
    response = load_statistics_dashboard(
        module,
        start_time=start_time,
        end_time=end_time,
        page=1,
        page_size=10,
        sort_by="created_at",
        sort_order="desc",
    )
    if not response.get("ok"):
        return {"ok": False, "error": response.get("error", "统计数据读取失败。"), "dashboard": {}}
    return {"ok": True, "dashboard": response.get("dashboard") or {}}


@st.cache_data(ttl=15, show_spinner=False)
def _dashboard_bundle(
    start_time: str | None,
    end_time: str,
    refresh_token: int,
    statistics_revision: int,
) -> dict[str, dict[str, Any]]:
    from service.statistics_service import load_statistics_dashboard

    """Load independent module summaries concurrently for the overview."""
    del refresh_token, statistics_revision
    modules = ("single", "batch", "video")
    responses: dict[str, dict[str, Any]] = {}

    def load_one(module: str) -> dict[str, Any]:
        response = load_statistics_dashboard(
            module,
            start_time=start_time,
            end_time=end_time,
            page=1,
            page_size=10,
            sort_by="created_at",
            sort_order="desc",
        )
        if not response.get("ok"):
            return {"ok": False, "error": response.get("error", "统计数据读取失败。"), "dashboard": {}}
        return {"ok": True, "dashboard": response.get("dashboard") or {}}

    with ThreadPoolExecutor(max_workers=len(modules), thread_name_prefix="overview-stats") as executor:
        futures = {
            executor.submit(load_one, module): module
            for module in modules
        }
        for future in as_completed(futures):
            module = futures[future]
            try:
                responses[module] = future.result()
            except Exception as exc:
                responses[module] = {
                    "ok": False,
                    "error": f"{module} 统计读取失败：{exc}",
                    "dashboard": {},
                }
    return {module: responses.get(module, {"ok": False, "error": "统计响应缺失。", "dashboard": {}}) for module in modules}


@st.cache_data(ttl=2, show_spinner=False)
def _health_status(refresh_token: int) -> dict[str, Any]:
    del refresh_token
    payload = client().health()
    return payload if isinstance(payload, dict) else {}


def _service_state(payload: dict[str, Any]) -> tuple[str, str]:
    """Map transport/readiness responses to user-facing startup states."""
    if not isinstance(payload, dict) or not payload.get("success"):
        error = payload.get("error") if isinstance(payload, dict) else {}
        code = str((error or {}).get("code") or "") if isinstance(error, dict) else ""
        if code in {"API_UNAVAILABLE", "CONNECT_TIMEOUT", "READ_TIMEOUT"}:
            return "starting", "算法服务正在启动，请稍候。"
        return "error", "算法服务启动失败，请重新启动系统或查看启动日志。"
    state = str(payload.get("status") or "ready").lower()
    if state in {"starting", "pending", "loading"}:
        return "starting", "算法服务正在启动，请稍候。"
    if state in {"error", "failed"}:
        return "error", "算法服务启动失败，请重新启动系统或查看启动日志。"
    if state == "degraded":
        return "degraded", "算法服务已连接，但部分资源仍在初始化。"
    return "ready", "算法服务已就绪。"


def _service_wait_content() -> None:
    """Poll liveness in a bounded fragment without blocking the page shell."""
    started = float(st.session_state.setdefault("overview_service_wait_started", time_module.monotonic()))
    attempts = int(st.session_state.get("overview_service_wait_attempts", 0))
    elapsed = time_module.monotonic() - started
    if elapsed <= 300.0:
        attempts += 1
        st.session_state["overview_service_wait_attempts"] = attempts
        payload = client().health()
        state, message = _service_state(payload)
        if state in {"ready", "degraded"}:
            st.session_state.pop("overview_service_wait_started", None)
            st.session_state.pop("overview_service_wait_attempts", None)
            try:
                _health_status.clear()
            except Exception:
                pass
            st.rerun()
            return
        if state == "error":
            st.error(message)
        else:
            preload = dict(payload.get("model_preload") or {}) if isinstance(payload, dict) else {}
            stage = str(preload.get("current_stage") or "正在启动后台 API 服务")
            progress = max(0, min(99, int(preload.get("progress") or 0)))
            startup_elapsed = float(preload.get("elapsed_seconds") or elapsed)
            st.markdown("### 系统正在启动")
            st.info(f"当前阶段：{stage}")
            st.progress(progress / 100.0, text=f"检测资源准备进度 {progress}%")
            st.caption(
                f"已用时间：{startup_elapsed:.1f} 秒 · 检测功能暂不可用 · "
                f"页面和菜单可以正常浏览。系统会自动恢复，无需刷新浏览器（第 {attempts} 次检查）。"
            )
    else:
        st.warning("检测资源准备时间超过 5 分钟，请检查模型资源或启动日志。")
        st.caption("页面仍可浏览；自动重试已暂停，确认资源后可立即重试。")
    if st.button("立即重试", icon=":material/refresh:", key="overview_retry_service", width="content"):
        st.session_state["overview_service_wait_started"] = time_module.monotonic()
        st.session_state["overview_service_wait_attempts"] = 0
        try:
            _health_status.clear()
        except Exception:
            pass


def _service_health_monitor_content() -> None:
    """Watch an already-open overview for a backend restart or disconnect."""
    state, _message = _service_state(client().health())
    previous = str(st.session_state.get("overview_service_state") or state)
    st.session_state["overview_service_state"] = state
    if previous in {"ready", "degraded"} and state not in {"ready", "degraded"}:
        st.session_state["overview_service_wait_started"] = time_module.monotonic()
        st.session_state["overview_service_wait_attempts"] = 0
        try:
            _health_status.clear()
        except Exception:
            pass
        st.rerun()


try:
    _service_wait_fragment = st.fragment(run_every=2.0)(_service_wait_content)
    _service_health_monitor = st.fragment(run_every=5.0)(_service_health_monitor_content)
except AttributeError:  # pragma: no cover - compatibility with old Streamlit
    _service_wait_fragment = _service_wait_content
    _service_health_monitor = _service_health_monitor_content


def _statistics_revision_monitor_content() -> None:
    from service.statistics_service import load_statistics_revision

    current = load_statistics_revision()
    previous = st.session_state.get("overview_statistics_revision")
    st.session_state["overview_statistics_revision"] = current
    if previous is not None and int(previous) != int(current):
        _dashboard.clear()
        _dashboard_bundle.clear()
        st.session_state["overview_refresh_token"] = int(
            st.session_state.get("overview_refresh_token", 0)
        ) + 1


try:
    _statistics_revision_monitor = st.fragment(run_every=3.0)(_statistics_revision_monitor_content)
except AttributeError:  # pragma: no cover - compatibility with old Streamlit
    _statistics_revision_monitor = _statistics_revision_monitor_content


def _render_chart(chart: alt.TopLevelMixin) -> None:
    try:
        st.altair_chart(chart, width="stretch")
    except Exception:
        st.error("图表暂时无法渲染，请刷新数据后重试。")


def _section(title: str, copy: str) -> None:
    st.markdown(f'<div class="overview-section">{title}</div><div class="overview-section-copy">{copy}</div>', unsafe_allow_html=True)


def _distribution(dashboards: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    counts = {"normal": 0, "anomaly": 0, "failed": 0}
    for dashboard in dashboards.values():
        for row in dashboard.get("result_distribution") or []:
            result = str(row.get("result") or "")
            if result in counts:
                counts[result] += int(row.get("count") or 0)
    total = sum(counts.values())
    return [
        {
            "结果": RESULT_LABELS[key],
            "数量": value,
            "占比": float(value / total * 100.0) if total else 0.0,
        }
        for key, value in counts.items()
    ]


def _module_distribution(dashboards: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for module, dashboard in dashboards.items():
        for item in dashboard.get("result_distribution") or []:
            rows.append(
                {
                    "模块": MODULE_LABELS[module],
                    "结果": RESULT_LABELS.get(str(item.get("result")), str(item.get("result"))),
                    "数量": int(item.get("count") or 0),
                }
            )
    return rows


def _trend(dashboards: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, float]] = {}
    for module, dashboard in dashboards.items():
        for row in dashboard.get("time_series") or []:
            bucket = str(row.get("bucket") or "")
            if not bucket:
                continue
            target = buckets.setdefault(bucket, {"任务数": 0.0, "正常结果": 0.0, "异常结果": 0.0})
            task_count = float(row.get("task_count") or 0)
            target["任务数"] += task_count
            target["正常结果"] += float(row.get("normal") or 0)
            target["异常结果"] += float(row.get("anomaly") or 0)
    output: list[dict[str, Any]] = []
    for bucket, values in sorted(buckets.items()):
        for series, value in values.items():
            output.append({"时间": bucket, "系列": series, "数值": value})
    return output


def _weighted_average(summaries: dict[str, dict[str, Any]]) -> float | None:
    fields = {"single": "average_end_to_end_ms", "batch": "average_task_ms", "video": "average_task_ms"}
    numerator = 0.0
    denominator = 0
    for module, summary in summaries.items():
        average = _number(summary.get(fields[module]))
        count = int(summary.get("task_count") or 0)
        if average is not None and count > 0:
            numerator += average * count
            denominator += count
    return numerator / denominator if denominator else None


def _donut(rows: list[dict[str, Any]]) -> None:
    import altair as alt

    if not any(int(row.get("数量") or 0) for row in rows):
        st.info("当前时间范围内暂无检测结果数据。")
        return
    selection = alt.selection_point(fields=["结果"], bind="legend")
    chart = (
        alt.Chart(alt.Data(values=rows))
        .mark_arc(innerRadius=62, outerRadius=106, stroke="#ffffff", strokeWidth=2)
        .encode(
            theta=alt.Theta("数量:Q"),
            color=alt.Color("结果:N", scale=_result_colors(alt), legend=alt.Legend(orient="bottom")),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.16)),
            tooltip=["结果:N", alt.Tooltip("数量:Q", format=","), alt.Tooltip("占比:Q", title="占比(%)", format=".1f")],
        )
        .add_params(selection)
        .properties(height=285)
    )
    _render_chart(chart)
    st.caption(" · ".join(f"{row['结果']} {row['数量']:,}（{row['占比']:.1f}%）" for row in rows))


def _grouped_results(rows: list[dict[str, Any]]) -> None:
    import altair as alt

    if not any(int(row.get("数量") or 0) for row in rows):
        st.info("当前时间范围内暂无可比较的模块结果。")
        return
    selection = alt.selection_point(fields=["结果"], bind="legend")
    chart = (
        alt.Chart(alt.Data(values=rows))
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("模块:N", title=None),
            y=alt.Y("数量:Q", title="结果数量"),
            color=alt.Color("结果:N", scale=_result_colors(alt), legend=alt.Legend(orient="top")),
            xOffset="结果:N",
            opacity=alt.condition(selection, alt.value(1), alt.value(0.16)),
            tooltip=["模块:N", "结果:N", alt.Tooltip("数量:Q", format=",")],
        )
        .add_params(selection)
        .properties(height=285)
    )
    _render_chart(chart)


def _task_composition(summaries: dict[str, dict[str, Any]]) -> None:
    import altair as alt

    rows = [
        {"检测类型": MODULE_LABELS[module], "任务数": int(summary.get("task_count") or 0)}
        for module, summary in summaries.items()
    ]
    if not any(row["任务数"] for row in rows):
        st.info("当前时间范围内暂无检测任务构成数据。")
        return
    chart = (
        alt.Chart(alt.Data(values=rows))
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, color="#168AAD")
        .encode(
            x=alt.X("检测类型:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("任务数:Q", title="任务数", scale=alt.Scale(zero=True)),
            color=alt.Color("检测类型:N", scale=alt.Scale(range=["#168AAD", "#4361EE", "#0E7490"]), legend=None),
            tooltip=["检测类型:N", alt.Tooltip("任务数:Q", format=",")],
        )
        .properties(height=245)
    )
    _render_chart(chart)


def _chart_heading(title: str, copy: str) -> None:
    st.markdown(
        f'<div class="overview-chart-title">{title}</div><div class="overview-chart-copy">{copy}</div>',
        unsafe_allow_html=True,
    )


def _trend_chart(rows: list[dict[str, Any]]) -> None:
    import altair as alt

    if not rows:
        st.info("当前时间范围内暂无可绘制的运行趋势。")
        return
    selection = alt.selection_point(fields=["系列"], bind="legend")
    chart = (
        alt.Chart(alt.Data(values=rows))
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("时间:T", title="时间", axis=alt.Axis(labelOverlap=True)),
            y=alt.Y("数值:Q", title="数量", scale=alt.Scale(zero=True)),
            color=alt.Color("系列:N", scale=_series_colors(alt), legend=alt.Legend(orient="top")),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.16)),
            tooltip=[alt.Tooltip("时间:T", title="时间", format="%Y-%m-%d %H:%M"), "系列:N", alt.Tooltip("数值:Q", format=",")],
        )
        .add_params(selection)
        .properties(height=315)
    )
    _render_chart(chart)


def _mini_distribution(dashboard: dict[str, Any]) -> None:
    import altair as alt

    rows = [
        {"结果": RESULT_LABELS.get(str(item.get("result")), str(item.get("result"))), "数量": int(item.get("count") or 0)}
        for item in dashboard.get("result_distribution") or []
    ]
    if not any(row["数量"] for row in rows):
        st.caption("暂无结果分布")
        return
    chart = (
        alt.Chart(alt.Data(values=rows))
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("结果:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("数量:Q", title=None, axis=None),
            color=alt.Color("结果:N", scale=_result_colors(alt), legend=None),
            tooltip=["结果:N", alt.Tooltip("数量:Q", format=",")],
        )
        .properties(height=105)
    )
    _render_chart(chart)


def _module_cards(dashboards: dict[str, dict[str, Any]]) -> None:
    columns = st.columns(3, gap="large")
    module_metrics = {
        "single": lambda summary: [
            ("任务数", _count(summary.get("task_count"))), ("正常数", _count(summary.get("normal_count"))),
            ("异常数", _count(summary.get("anomaly_count"))), ("平均异常分数", _score(summary.get("average_score"))),
            ("平均端到端", _duration(summary.get("average_end_to_end_ms"))), ("平均纯推理", _duration(summary.get("average_inference_ms"))),
        ],
        "batch": lambda summary: [
            ("任务数", _count(summary.get("task_count"))), ("累计图片", _count(summary.get("total_images"))),
            ("异常图片", _count(summary.get("anomaly_count"))), ("平均任务耗时", _duration(summary.get("average_task_ms"))),
            ("平均单图耗时", _duration(summary.get("average_item_ms"))), ("平均吞吐率", _throughput(summary.get("average_throughput"))),
        ],
        "video": lambda summary: [
            ("任务数", _count(summary.get("task_count"))), ("累计帧数", _count(summary.get("total_frames"))),
            ("异常帧数", _count(summary.get("anomaly_frames"))), ("完整处理 FPS", _fps(summary.get("average_full_pipeline_fps"))),
            ("纯推理 FPS", _fps(summary.get("average_pure_inference_fps"))), ("平均任务耗时", _duration(summary.get("average_task_ms"))),
        ],
    }
    tab_labels = {"single": "单图检测", "batch": "批量检测", "video": "视频检测"}
    for column, module in zip(columns, ("single", "batch", "video")):
        dashboard = dashboards[module]
        summary = dashboard.get("summary") or {}
        with column:
            with st.container(border=True):
                st.markdown(f'<div class="overview-module-title">{MODULE_LABELS[module]}</div>', unsafe_allow_html=True)
                st.markdown('<div class="overview-module-copy">当前时间和筛选范围内的核心质量与性能概况。</div>', unsafe_allow_html=True)
                metrics = module_metrics[module](summary)
                cells = st.columns(2)
                for index, (label, value) in enumerate(metrics):
                    cells[index % 2].caption(label)
                    cells[index % 2].markdown(f"**{value}**")
                _mini_distribution(dashboard)
                if st.button("查看统计分析", key=f"overview_open_{module}", icon=":material/monitoring:", width="stretch"):
                    st.session_state["analytics_module_label"] = tab_labels[module]
                    st.switch_page(str(get_page("statistics").page_path))


def _recent_rows(dashboards: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for module, dashboard in dashboards.items():
        for item in dashboard.get("task_details") or []:
            status = str(item.get("status") or "")
            if module == "single":
                prediction = RESULT_LABELS.get(str(item.get("prediction")), str(item.get("prediction") or "-"))
                core = f"异常分数 {_score(item.get('score'))}"
                task_name = item.get("filename") or "单图检测任务"
            elif module == "batch":
                anomaly = int(item.get("anomaly_count") or 0)
                prediction = "失败" if status == "failed" else ("异常" if anomaly else "正常")
                core = f"{_count(item.get('total_images'))} 张 · 异常 {anomaly:,}"
                task_name = item.get("input_name") or "批量检测任务"
            else:
                anomaly = int(item.get("anomaly_count") or 0)
                prediction = "失败" if status == "failed" else ("异常" if anomaly else "正常")
                core = f"{_count(item.get('total_frames'))} 帧 · FPS {_score(item.get('full_pipeline_fps'))}"
                task_name = item.get("filename") or "视频检测任务"
            rows.append(
                {
                    "_time": str(item.get("time") or ""),
                    "时间": _friendly_time(item.get("time")),
                    "任务类型": MODULE_LABELS[module],
                    "任务名称": friendly_value(task_name, "filename"),
                    "结果": prediction,
                    "核心数值": core,
                    "任务状态": STATUS_LABELS.get(status, status or "-"),
                    "查看详情": "统计分析",
                }
            )
    rows.sort(key=lambda item: item["_time"], reverse=True)
    return [{key: value for key, value in item.items() if key != "_time"} for item in rows[:8]]


@st.fragment
def _render_home_main(runtime: dict[str, Any]) -> None:
    from service.statistics_service import load_statistics_revision

    """Render overview interactions without rerunning the navigation shell."""
    _install_styles()
    refresh_token = int(st.session_state.get("overview_refresh_token", 0))
    statistics_revision = load_statistics_revision()
    st.session_state["overview_statistics_revision"] = statistics_revision
    health = _health_status(refresh_token)
    render_system_hero(runtime, health)
    service_state, _service_message = _service_state(health)
    start_time, end_time = _time_controls()
    if service_state not in {"ready", "degraded"}:
        # Keep the shell and hero visible while the launcher finishes starting
        # the API.  Statistics are intentionally not requested until liveness
        # is confirmed, so a cold connection cannot block or poison the page.
        _service_wait_fragment()
        start_text = format_beijing_datetime(start_time, fallback="最早记录")
        st.markdown(
            f'<div class="overview-range">当前统计区间：<b>{start_text}</b> 至 <b>{format_beijing_datetime(end_time)}</b></div>',
            unsafe_allow_html=True,
        )
        _section("整体系统运行概览", "统计服务正在连接；服务恢复后将自动填充真实历史任务数据。")
        render_stat_cards(
            [
                {"icon": "dataset", "label": "总检测任务数", "value": "加载中…", "help": "等待统计服务"},
                {"icon": "image", "label": "单图检测任务", "value": "加载中…", "help": "等待统计服务"},
                {"icon": "photo_library", "label": "批量检测任务", "value": "加载中…", "help": "等待统计服务"},
                {"icon": "video_file", "label": "视频检测任务", "value": "加载中…", "help": "等待统计服务"},
                {"icon": "task_alt", "label": "成功任务数", "value": "加载中…", "help": "等待统计服务"},
            ]
        )
        _section("关键运行图表", "统计服务正在连接，请稍候。")
        left, right = st.columns([1.45, 1], gap="large")
        with left:
            with st.container(border=True):
                _chart_heading("近期检测趋势", "服务恢复后显示真实趋势")
                st.info("统计服务正在连接，请稍候。")
        with right:
            with st.container(border=True):
                _chart_heading("检测结果分布", "服务恢复后显示真实分布")
                st.info("统计服务正在连接，请稍候。")
        return
    st.session_state["overview_service_state"] = service_state
    _service_health_monitor()
    _statistics_revision_monitor()
    st.session_state.pop("overview_service_wait_started", None)
    st.session_state.pop("overview_service_wait_attempts", None)
    start_iso = start_time.isoformat() if start_time else None
    end_iso = end_time.isoformat()
    with st.spinner("正在汇总真实历史任务..."):
        responses = _dashboard_bundle(
            start_iso,
            end_iso,
            refresh_token,
            statistics_revision,
        )
    errors = [friendly_error_message(item.get("error"), "系统总览数据加载失败。") for item in responses.values() if not item.get("ok")]
    if errors:
        st.error(errors[0] or "系统总览数据加载失败，请稍后重试。")
        return
    dashboards = {module: response["dashboard"] for module, response in responses.items()}
    summaries = {module: dashboard.get("summary") or {} for module, dashboard in dashboards.items()}
    start_text = format_beijing_datetime(start_time, fallback="最早记录")
    st.markdown(
        f'<div class="overview-range">当前统计区间：<b>{start_text}</b> 至 <b>{format_beijing_datetime(end_time)}</b>（Asia/Shanghai，开始与结束边界均包含）</div>',
        unsafe_allow_html=True,
    )

    total_tasks = sum(int(summary.get("task_count") or 0) for summary in summaries.values())
    success_tasks = sum(int(summary.get("success_tasks") or 0) for summary in summaries.values())
    failed_tasks = sum(int(summary.get("failed_tasks") or 0) for summary in summaries.values())
    anomalies = (
        int(summaries["single"].get("anomaly_count") or 0)
        + int(summaries["batch"].get("anomaly_count") or 0)
        + int(summaries["video"].get("anomaly_frames") or 0)
    )
    profile = runtime.get("memory_bank_profile") or {}
    memory_status = "已冻结" if runtime.get("memory_bank_ready") else "未就绪"
    memory_help = str(profile.get("friendly_name") or "当前没有可用冻结记忆库")
    mode = str(runtime.get("inference_mode") or runtime.get("mode") or "")
    mode_status = "只读推理" if mode in {"inference_only", "readonly", "只读"} else friendly_value(mode, "inference_mode")
    _section("整体系统运行概览", "任务指标来自当前时间范围；只读模式与记忆库状态来自当前后端已应用配置。")
    render_stat_cards(
        [
            {"icon": "dataset", "label": "总检测任务数", "value": _count(total_tasks), "help": "单图、批量与视频任务合计"},
            {"icon": "image", "label": "单图检测任务", "value": _count(summaries["single"].get("task_count")), "help": "单图检测任务"},
            {"icon": "photo_library", "label": "批量检测任务", "value": _count(summaries["batch"].get("task_count")), "help": "批量目录检测任务"},
            {"icon": "video_file", "label": "视频检测任务", "value": _count(summaries["video"].get("task_count")), "help": "本地视频逐帧任务"},
            {"icon": "task_alt", "label": "成功任务数", "value": _count(success_tasks), "help": "明确完成的任务"},
            {"icon": "error", "label": "失败任务数", "value": _count(failed_tasks), "help": "明确标记失败的任务", "kind": "alert"},
            {"icon": "crisis_alert", "label": "异常结果总数", "value": _count(anomalies), "help": "异常图片与异常视频帧累计", "kind": "alert"},
            {"icon": "timer", "label": "平均端到端耗时", "value": _duration(_weighted_average(summaries)), "help": "按具有真实耗时的任务加权"},
            {"icon": "lock", "label": "当前推理模式", "value": mode_status or "状态未知", "help": "来自当前已应用运行配置", "kind": "state"},
            {"icon": "database", "label": "默认记忆库", "value": memory_status, "help": memory_help, "kind": "state"},
        ]
    )
    if total_tasks == 0:
        st.info("当前时间范围内暂无真实检测任务，可扩大时间范围或刷新数据。")

    _section("关键运行图表", "使用同一统计时间范围展示工作量、质量分布和近期变化；全部来自真实历史任务。")
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        with st.container(border=True):
            _chart_heading("近期检测趋势", "任务、正常结果和异常结果随时间变化")
            _trend_chart(_trend(dashboards))
    with right:
        with st.container(border=True):
            _chart_heading("检测结果分布", "正常、异常、失败的数量与比例")
            _donut(_distribution(dashboards))
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            _chart_heading("检测类型构成", "单图、批量和视频任务数量")
            _task_composition(summaries)
    with right:
        with st.container(border=True):
            _chart_heading("三类检测质量对比", "按检测类型对比正常、异常和失败结果")
            _grouped_results(_module_distribution(dashboards))

    with st.container(border=True):
        _chart_heading("运行性能概况", "分别保留端到端、纯推理、批量吞吐和视频完整处理口径")
        render_performance_cards(
            [
                {"label": "单图平均纯推理时间", "value": _duration(summaries["single"].get("average_inference_ms")), "help": "仅模型推理阶段"},
                {"label": "全任务平均端到端耗时", "value": _duration(_weighted_average(summaries)), "help": "按具备真实耗时的任务加权"},
                {"label": "批量平均吞吐率", "value": _throughput(summaries["batch"].get("average_throughput")), "help": "完成图片数 / 任务总耗时"},
                {"label": "视频平均完整处理 FPS", "value": _fps(summaries["video"].get("average_full_pipeline_fps")), "help": "包含解码、推理与结果写入"},
            ]
        )

    _section("检测类型概览", "三个模块使用相同时间范围；旧记录缺失的耗时字段保持为空，不填充伪造数值。")
    _module_cards(dashboards)

    _section("最近任务摘要", "最近任务仅作为辅助摘要，不显示内部路径、服务地址或任务 UUID。")
    recent = _recent_rows(dashboards)
    if recent:
        st.dataframe(
            recent,
            hide_index=True,
            width="stretch",
            height=min(380, 62 + len(recent) * 35),
            column_config={"查看详情": st.column_config.LinkColumn("查看详情", display_text="查看详情")},
        )
    else:
        st.info("当前时间范围内暂无最近任务。")
    st.page_link(
        get_page("statistics").page_path,
        label="前往统计分析查看完整数据",
        icon=":material/bar_chart:",
        width="stretch",
    )


def render_home() -> None:
    """Keep sidebar controls outside the independently rerunnable main view."""
    runtime = sidebar_runtime_controls("home")
    _render_home_main(runtime)


__all__ = ["render_home"]
