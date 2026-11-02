from __future__ import annotations

from html import escape
from typing import Any, Iterable

import streamlit as st

from flow_stage_labels import flow_stage_label
from friendly_display import friendly_value


FLOW_DEFINITION = (
    ("input", "waiting", "选择图片、图片目录或本地视频文件。"),
    ("video_file", "waiting", "读取图像；视频模式逐帧解码。"),
    ("tune", "waiting", "按当前冻结配置执行缩放和张量预处理。"),
    ("neurology", "model", "由当前骨干网络提取特征。"),
    ("manage_search", "memory", "在冻结的正常特征记忆库中执行近邻检索。"),
    ("rule", "calibration", "计算异常分数并使用冻结阈值完成判定。"),
    ("save", "waiting", "按任务设置生成可视化并保存受控结果。"),
)

STAT_ICON_GLYPHS = {
    "dataset": "总",
    "image": "图",
    "photo_library": "批",
    "video_file": "视",
    "task_alt": "成",
    "error": "败",
    "crisis_alert": "异",
    "timer": "时",
    "lock": "锁",
    "database": "库",
}


def _status_class(value: str) -> str:
    if any(token in value for token in ("异常", "失败", "未配置", "缺失")):
        return "is-error"
    if any(token in value for token in ("未知", "尚未", "等待", "未就绪", "启动中", "加载中")):
        return "is-idle"
    return "is-ready"


def _status_snapshot(runtime: dict[str, Any], health: dict[str, Any]) -> dict[str, Any]:
    health_state = str(health.get("status") or "").lower()
    health_ok = bool(health.get("success") and health_state in {"ok", "ready", "degraded"})
    model = health.get("current_model") or {}
    model_loaded = bool(model.get("loaded"))
    memory_ready = bool(runtime.get("memory_bank_ready") and model.get("weights_valid", True))
    calibration_ready = bool(memory_ready and model.get("calibration_valid") and runtime.get("threshold") is not None)
    mode = str(runtime.get("inference_mode") or runtime.get("mode") or "")
    preload_state = str((health.get("model_preload") or {}).get("state") or "pending").lower()
    if not health.get("success"):
        service_status = "启动中" if str((health.get("error") or {}).get("code") or "") == "API_UNAVAILABLE" else "启动失败"
    elif health_state in {"starting", "pending", "loading"}:
        service_status = "启动中"
    elif health_state == "degraded":
        service_status = "部分就绪"
    else:
        service_status = "运行中"
    if not runtime.get("memory_bank_id"):
        model_status = "等待目标域"
    elif preload_state == "loading":
        model_status = "加载中"
    elif preload_state == "failed":
        model_status = "加载失败"
    else:
        model_status = "已加载" if model_loaded else "尚未加载"
    runtime_state = str(health.get("runtime_state") or "").upper()
    worker_state = str(health.get("worker_state") or "STOPPED").upper()
    worker_required = bool(health.get("worker_required_for_runtime"))
    if worker_required:
        worker_status = {
            "FAILED": "失败（检测不可用）",
            "STOPPED": "已停止（需检查）",
            "STARTING": "启动中",
            "ALIVE": "运行中",
            "IDLE": "空闲",
        }.get(worker_state, "状态未知")
    else:
        worker_status = {
            "FAILED": "辅助异常（不影响检测）",
            "STOPPED": "未常驻（不影响检测）",
            "STARTING": "辅助启动中",
            "ALIVE": "辅助运行中",
            "IDLE": "辅助空闲",
        }.get(worker_state, "辅助状态未知")
    return {
        "health_ok": health_ok,
        "model_loaded": model_loaded,
        "memory_ready": memory_ready,
        "calibration_ready": calibration_ready,
        "service_status": service_status,
        "model_status": model_status,
        "memory_status": "已冻结" if memory_ready else "未配置",
        "calibration_status": "已完成" if calibration_ready else "未配置",
        "mode_status": "只读推理" if mode in {"inference_only", "readonly", "只读"} else (friendly_value(mode, "inference_mode") if mode else "状态未知"),
        "runtime_state": runtime_state or ("READY" if health_ok else "STARTING"),
        "worker_status": worker_status,
        "worker_required": worker_required,
    }


def install_dashboard_styles() -> None:
    st.markdown(
        """
        <style>
          .tech-hero {
            --hero-deep:#102A45; --hero-mid:#16485C; --hero-teal:#176B77;
            --hero-border:rgba(125,211,252,.22); --hero-line:rgba(148,211,224,.18);
            --hero-surface:rgba(7,31,49,.18); --hero-surface-strong:rgba(7,34,53,.34);
            --hero-accent:#67E8F9; --hero-text:#F4FAFD; --hero-muted:#B8CFDC;
            position: relative; isolation: isolate; overflow: hidden;
            margin: .05rem 0 .82rem; padding: .94rem 1rem .9rem;
            border: 1px solid var(--hero-border); border-radius: 18px;
            background:
              radial-gradient(circle at 88% 8%, rgba(91,211,224,.11), transparent 30rem),
              linear-gradient(122deg, var(--hero-deep) 0%, var(--hero-mid) 58%, var(--hero-teal) 100%);
            box-shadow: 0 12px 28px rgba(8,29,47,.14), inset 0 1px 0 rgba(207,250,254,.08);
          }
          .tech-hero::before {
            content:""; position:absolute; z-index:-1; width:24rem; height:14rem;
            right:-6rem; top:-6rem; border-radius:50%; background:rgba(103,232,249,.10);
            filter:blur(48px); pointer-events:none; animation:techGlowDrift 10s ease-in-out infinite alternate;
          }
          .tech-hero::after {
            content:""; position:absolute; inset:0; z-index:-1; pointer-events:none;
            border-top:1px solid rgba(207,250,254,.22); border-radius:inherit;
          }
          .tech-hero-main { display:grid; grid-template-columns:minmax(235px,.9fr) minmax(540px,2.1fr); gap:.82rem; align-items:stretch; }
          .tech-hero-copy { display:flex; flex-direction:column; justify-content:center; min-width:0; padding:.08rem .18rem .08rem .06rem; }
          .tech-kicker { color:var(--hero-accent); font-size:.7rem; font-weight:850; letter-spacing:.095em; }
          .tech-title { margin:.16rem 0 .28rem; color:var(--hero-text); font-size:1.9rem; line-height:1.06; font-weight:850; text-shadow:0 0 20px rgba(103,232,249,.09); }
          .tech-description { max-width:29rem; color:var(--hero-muted); font-size:.84rem; line-height:1.52; }
          .tech-status-row { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:.34rem; max-width:19rem; margin-top:.62rem; }
          .tech-status {
            display:inline-flex; align-items:center; gap:.28rem; min-width:0; min-height:1.72rem; padding:.22rem .46rem;
            border:1px solid var(--hero-line); border-radius:999px; background:var(--hero-surface-strong);
            color:#D4E5EF; font-size:.7rem; font-weight:760; white-space:nowrap; overflow:hidden;
            transition:border-color 180ms ease, background 180ms ease;
          }
          .tech-status:hover { border-color:rgba(125,211,252,.34); background:rgba(9,54,70,.42); }
          .tech-status::before { content:""; width:.42rem; height:.42rem; border-radius:50%; background:#94a3b8; }
          .tech-status.is-ready { color:#D7FAFF; border-color:rgba(34,211,238,.30); background:rgba(8,72,88,.32); }
          .tech-status.is-ready::before { background:#22D3EE; box-shadow:0 0 0 4px rgba(34,211,238,.12); }
          .tech-status.is-error { color:#FFE4E6; border-color:rgba(251,113,133,.34); background:rgba(127,29,29,.24); }
          .tech-status.is-error::before { background:#FB7185; }
          .tech-flow-panel {
            min-width:0; padding:.28rem .02rem .24rem .88rem; border:0; border-left:1px solid var(--hero-line);
            border-radius:0; background:var(--hero-surface); box-shadow:none;
          }
          .tech-panel-head { display:flex; align-items:flex-start; justify-content:space-between; gap:.7rem; margin:0 .02rem .58rem .04rem; }
          .tech-panel-heading { min-width:0; }
          .tech-panel-title { color:var(--hero-text); font-size:.98rem; font-weight:820; }
          .tech-panel-note { margin-top:.12rem; color:#9FB5C6; font-size:.69rem; white-space:nowrap; }
          .tech-flow { display:grid; grid-template-columns:repeat(7,minmax(0,1fr)); gap:.16rem; align-items:stretch; }
          .tech-flow-node {
            position:relative; min-width:0; text-align:center; padding:.35rem .04rem .32rem;
            border:1px solid rgba(125,211,252,.12); border-radius:10px; background:rgba(7,34,53,.17);
            transition:border-color 180ms ease, background 180ms ease, transform 180ms ease;
          }
          .tech-flow-node:hover { border-color:rgba(103,232,249,.28); background:rgba(9,64,78,.24); transform:translateY(-2px); }
          .tech-flow-node.is-ready { border-color:rgba(103,232,249,.20); }
          .tech-flow-node:not(:last-child)::after {
            content:""; position:absolute; z-index:1; top:1.5rem; left:calc(50% + 1.2rem); right:calc(-50% - .18rem);
            height:1px; background:linear-gradient(90deg, rgba(103,232,249,.46), rgba(34,211,238,.13));
          }
          .tech-flow-node:not(:last-child)::before {
            content:""; position:absolute; z-index:2; top:1.37rem; left:calc(50% + 1.2rem); width:.28rem; height:.28rem;
            border-radius:50%; background:#67E8F9; box-shadow:0 0 9px rgba(103,232,249,.72);
            animation:techSignal 5.6s linear infinite;
          }
          .tech-flow-number {
            position:relative; z-index:3; display:inline-flex; align-items:center; justify-content:center;
            width:2.28rem; height:2.28rem; border:1px solid rgba(103,232,249,.30); border-radius:50%;
            background:rgba(12,61,78,.72); color:#D8FAFF;
            box-shadow:0 4px 11px rgba(2,12,27,.16), inset 0 1px 0 rgba(224,242,254,.08);
            font-size:.73rem; font-weight:830; letter-spacing:.035em;
          }
          .tech-flow-node.is-ready .tech-flow-number { animation:techReady 3.8s ease-in-out infinite; }
          .tech-flow-name { min-width:0; margin-top:.34rem; color:#F1F9FC; font-size:clamp(.74rem,.96vw,.83rem); font-weight:800; line-height:1.25; letter-spacing:-.04em; white-space:nowrap; }
          .tech-flow-state { display:inline-flex; align-items:center; justify-content:center; gap:.28rem; margin-top:.27rem; color:#9FB5C6; font-size:.75rem; line-height:1.2; white-space:nowrap; }
          .tech-flow-state::before { content:""; width:.38rem; height:.38rem; flex:0 0 auto; border-radius:50%; background:#7C91A3; }
          .tech-flow-node.is-ready .tech-flow-state { color:#A5F3FC; }
          .tech-flow-node.is-ready .tech-flow-state::before { background:#22D3EE; box-shadow:0 0 7px rgba(34,211,238,.56); }
          .tech-flow-node.is-error .tech-flow-state { color:#FECDD3; }
          .tech-flow-node.is-error .tech-flow-state::before { background:#FB7185; }
          .tech-config-link {
            display:inline-flex; flex:0 0 auto; align-items:center; justify-content:center; min-height:1.78rem; padding:.28rem .58rem;
            border:1px solid rgba(125,211,252,.24); border-radius:8px; background:rgba(8,56,72,.28);
            color:#CFFAFE !important; font-size:.7rem; font-weight:780; text-decoration:none !important;
            transition:background 180ms ease, border-color 180ms ease, transform 180ms ease;
          }
          .tech-config-link:hover { color:#ECFEFF !important; border-color:rgba(103,232,249,.42); background:rgba(8,76,90,.40); transform:translateY(-1px); }
          .overview-stat-grid { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:.58rem; margin:.48rem 0 .75rem; }
          .overview-stat-card {
            position:relative; min-width:0; min-height:6.35rem; padding:.68rem .72rem .62rem; overflow:hidden;
            border:1px solid rgba(112,139,160,.18); border-radius:12px; background:linear-gradient(145deg,#fff,#fbfdff);
            box-shadow:0 7px 18px rgba(15,39,71,.055);
          }
          .overview-stat-card::after { content:""; position:absolute; right:-1.7rem; bottom:-1.9rem; width:4.6rem; height:4.6rem; border-radius:50%; background:rgba(34,211,238,.055); }
          .overview-stat-top { display:flex; align-items:center; justify-content:space-between; gap:.45rem; }
          .overview-stat-icon { display:inline-flex; align-items:center; justify-content:center; width:1.75rem; height:1.75rem; border-radius:8px; background:rgba(14,116,144,.08); color:#0e7490; }
          .overview-stat-glyph { font-size:.74rem; font-weight:850; line-height:1; }
          .overview-stat-label { flex:1; min-width:0; color:#60758b; font-size:.7rem; font-weight:720; line-height:1.25; }
          .overview-stat-value { position:relative; z-index:1; margin-top:.38rem; color:#12324d; font-size:1.3rem; font-weight:850; line-height:1.05; overflow-wrap:anywhere; }
          .overview-stat-foot { position:relative; z-index:1; margin-top:.3rem; color:#8291a2; font-size:.6rem; line-height:1.3; }
          .overview-stat-card.is-alert .overview-stat-icon { color:#b23a4d; background:rgba(209,73,91,.08); }
          .overview-stat-card.is-state .overview-stat-value { font-size:1.05rem; color:#0e6478; }
          .overview-performance-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.55rem; margin:.25rem 0 .2rem; }
          .overview-performance-item { min-width:0; padding:.62rem .66rem; border:1px solid rgba(112,139,160,.16); border-radius:10px; background:linear-gradient(145deg,#fbfdff,#f3fbfd); }
          .overview-performance-label { color:#6d8094; font-size:.65rem; line-height:1.25; }
          .overview-performance-value { margin-top:.26rem; color:#0e6478; font-size:1.05rem; font-weight:840; overflow-wrap:anywhere; }
          .overview-performance-note { margin-top:.22rem; color:#8a98a8; font-size:.58rem; line-height:1.3; }
          @keyframes techGlowDrift { from { transform:translate3d(0,0,0) scale(.96); opacity:.64; } to { transform:translate3d(-2rem,1.4rem,0) scale(1.08); opacity:1; } }
          @keyframes techSignal { 0% { transform:translateX(0); opacity:0; } 18% { opacity:1; } 80% { opacity:1; } 100% { transform:translateX(2rem); opacity:0; } }
          @keyframes techReady { 0%,100% { box-shadow:0 6px 15px rgba(2,12,27,.25), inset 0 1px 0 rgba(224,242,254,.10); } 50% { box-shadow:0 6px 20px rgba(34,211,238,.24), inset 0 1px 0 rgba(224,242,254,.14); } }
          @media (max-width:1270px) {
            .tech-hero-main { grid-template-columns:1fr; }
            .tech-status-row { grid-template-columns:repeat(4,minmax(0,1fr)); max-width:none; }
            .tech-flow-panel { padding:.72rem .04rem .08rem; border-left:0; border-top:1px solid var(--hero-line); }
            .tech-flow { grid-template-columns:repeat(4,minmax(0,1fr)); row-gap:.62rem; }
            .tech-flow-node::before, .tech-flow-node::after { display:none; }
            .tech-panel-note { white-space:normal; }
            .overview-stat-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
            .overview-performance-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
          }
          @media (max-width:900px) {
            .tech-hero { padding:.82rem; }
            .tech-title { font-size:1.65rem; }
            .tech-status-row { grid-template-columns:repeat(2,minmax(0,1fr)); }
            .tech-panel-head { flex-wrap:wrap; }
          }
          @media (max-width:560px) {
            .tech-flow { grid-template-columns:repeat(2,minmax(0,1fr)); }
            .overview-stat-grid { grid-template-columns:1fr; }
          }
          @media (prefers-reduced-motion:reduce) {
            .tech-hero::before, .tech-flow-node, .tech-flow-number, .tech-flow-node::before, .tech-status, .tech-config-link, .overview-stat-card { animation:none !important; transition:none !important; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_system_hero(runtime: dict[str, Any], health: dict[str, Any]) -> None:
    snapshot = _status_snapshot(runtime, health)
    statuses = (
        ("系统服务", snapshot["service_status"]),
        ("运行时", snapshot["runtime_state"]),
        ("模型状态", snapshot["model_status"]),
        ("记忆库状态", snapshot["memory_status"]),
        ("推理模式", snapshot["mode_status"]),
        ("后台Worker", snapshot["worker_status"]),
    )
    status_html = "".join(
        f'<span class="tech-status {_status_class(value)}"><span>{escape(label)}：</span>{escape(value)}</span>'
        for label, value in statuses
    )
    flow_states = {
        "waiting": "等待任务",
        "model": "已就绪" if snapshot["model_loaded"] else "尚未加载",
        "memory": snapshot["memory_status"],
        "calibration": "已就绪" if snapshot["calibration_ready"] else "未配置",
    }
    nodes = []
    for index, (stage_key, state_key, help_text) in enumerate(FLOW_DEFINITION, start=1):
        name = flow_stage_label(stage_key)
        state = flow_states.get(state_key, state_key)
        nodes.append(
            f'<div class="tech-flow-node {_status_class(state)}" title="{escape(help_text)}" style="animation-delay:{(index - 1) * 55}ms">'
            f'<div class="tech-flow-number" aria-hidden="true">{index:02d}</div>'
            f'<div class="tech-flow-name">{escape(name)}</div>'
            f'<span class="tech-flow-state">{escape(state)}</span></div>'
        )
    hero_html = (
        '<section class="tech-hero"><div class="tech-hero-main"><div class="tech-hero-copy">'
        '<div class="tech-kicker">AdaptiveAOI · 工业视觉智能</div>'
        '<div class="tech-title">系统运行总览</div>'
        '<div class="tech-description">面向工业视觉质检，统一呈现系统状态、检测流程与运行指标。</div>'
        f'<div class="tech-status-row">{status_html}</div></div><div class="tech-flow-panel">'
        '<div class="tech-panel-head"><div class="tech-panel-heading">'
        '<div class="tech-panel-title">智能缺陷检测流程</div>'
        '<div class="tech-panel-note">状态同步自当前系统配置</div></div>'
        '<a class="tech-config-link" href="运行配置" target="_self">查看运行配置</a></div>'
        f'<div class="tech-flow">{"".join(nodes)}</div></div></div></section>'
    )
    st.markdown(hero_html, unsafe_allow_html=True)


def render_stat_cards(items: Iterable[dict[str, str]]) -> None:
    cards = []
    for index, item in enumerate(items):
        kind = item.get("kind", "")
        glyph = STAT_ICON_GLYPHS.get(item.get("icon", ""), "·")
        classes = "overview-stat-card"
        if kind == "alert":
            classes += " is-alert"
        elif kind == "state":
            classes += " is-state"
        cards.append(
            f'<div class="{classes}" style="animation-delay:{min(index, 9) * 35}ms">'
            f'<div class="overview-stat-top"><span class="overview-stat-icon" aria-hidden="true"><span class="overview-stat-glyph">{escape(glyph)}</span></span>'
            f'<span class="overview-stat-label">{escape(item.get("label", ""))}</span></div>'
            f'<div class="overview-stat-value">{escape(item.get("value", "-"))}</div>'
            f'<div class="overview-stat-foot">{escape(item.get("help", ""))}</div></div>'
        )
    st.markdown(f'<div class="overview-stat-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_performance_cards(items: Iterable[dict[str, str]]) -> None:
    cards = "".join(
        f'<div class="overview-performance-item">'
        f'<div class="overview-performance-label">{escape(item.get("label", ""))}</div>'
        f'<div class="overview-performance-value">{escape(item.get("value", "-"))}</div>'
        f'<div class="overview-performance-note">{escape(item.get("help", ""))}</div></div>'
        for item in items
    )
    st.markdown(f'<div class="overview-performance-grid">{cards}</div>', unsafe_allow_html=True)


__all__ = ["install_dashboard_styles", "render_performance_cards", "render_stat_cards", "render_system_hero"]
