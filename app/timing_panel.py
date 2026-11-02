from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st


STAGE_LABELS = {
    "input_read_ms": "输入读取",
    "decode_ms": "视频读取与解码",
    "preprocess_ms": "图像预处理",
    "feature_extraction_ms": "特征提取",
    "ann_search_ms": "近邻检索",
    "anomaly_scoring_ms": "异常评分",
    "inference_ms": "纯推理",
    "postprocess_ms": "后处理",
    "visualization_ms": "可视化生成",
    "encode_ms": "视频编码",
    "save_ms": "结果保存",
    "end_to_end_ms": "端到端总计",
    "frame_total_ms": "逐帧全流程",
}

STAGE_DESCRIPTIONS = {
    "model_load_ms": "加载或复用当前模型资源。",
    "memory_bank_load_ms": "加载或复用冻结的正常特征记忆库。",
    "input_read_ms": "读取输入图片或帧数据。",
    "decode_ms": "读取视频并将压缩数据解码为图像帧。",
    "preprocess_ms": "调整尺寸并转换为模型输入张量。",
    "feature_extraction_ms": "使用当前骨干网络生成图像特征。",
    "ann_search_ms": "将图像特征与正常记忆库进行匹配。",
    "anomaly_scoring_ms": "聚合近邻距离并计算异常分数。",
    "inference_ms": "完成特征提取、检索与异常评分。",
    "postprocess_ms": "生成最终判定、热力图和 Mask 数据。",
    "visualization_ms": "绘制叠加图、热力图或 Mask。",
    "encode_ms": "将已处理帧编码为结果视频。",
    "save_ms": "保存检测记录与可视化结果。",
    "end_to_end_ms": "从接收输入到结果可用的完整耗时。",
    "frame_total_ms": "单帧从读取到处理完成的完整耗时。",
}


def _install_styles() -> None:
    st.markdown(
        """
        <style>
        .aoi-timing-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.72rem;margin:.55rem 0 .2rem}
        .aoi-timing-card{border:1px solid rgba(143,166,195,.28);border-radius:14px;background:linear-gradient(145deg,rgba(255,255,255,.94),rgba(245,249,253,.92));padding:.84rem .9rem;min-width:0}
        .aoi-timing-head{display:flex;align-items:flex-start;justify-content:space-between;gap:.8rem;margin-bottom:.48rem}
        .aoi-timing-title{font-size:.94rem;font-weight:760;color:#142945;line-height:1.35}
        .aoi-timing-value{font-size:.94rem;font-weight:760;color:#117b9f;white-space:nowrap}
        .aoi-timing-description{font-size:.78rem;line-height:1.5;color:#667994;margin-bottom:.58rem}
        .aoi-timing-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.42rem}
        .aoi-timing-stat{min-width:0;border-radius:9px;background:rgba(231,240,249,.65);padding:.42rem .48rem}
        .aoi-timing-stat span{display:block;font-size:.68rem;color:#71829a;margin-bottom:.16rem}
        .aoi-timing-stat strong{display:block;font-size:.78rem;color:#233b59;white-space:normal;overflow-wrap:anywhere}
        .aoi-timing-share{height:5px;border-radius:999px;background:#e4edf5;overflow:hidden;margin-top:.58rem}
        .aoi-timing-share>span{display:block;height:100%;border-radius:inherit;background:linear-gradient(90deg,#249ec2,#4e6ce7)}
        @media(max-width:900px){.aoi-timing-grid{grid-template-columns:1fr}.aoi-timing-stats{grid-template-columns:repeat(2,minmax(0,1fr))}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _number(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def _ms(value: Any, *, skipped: bool = False) -> str:
    number = _number(value)
    if number is None:
        return "未执行" if skipped else "未单独统计"
    return f"{number / 1000:.2f} s" if number >= 1000 else f"{number:.2f} ms"


def _fps(value: Any) -> str:
    number = _number(value)
    return f"{number:.2f} FPS" if number is not None else "暂无"


def _share(value: Any, total: Any) -> tuple[str, float]:
    part = _number(value)
    whole = _number(total)
    if part is None or whole is None or whole <= 0:
        return "未单独统计", 0.0
    ratio = max(0.0, min(1.0, part / whole))
    return f"{ratio * 100:.1f}%", ratio


def _render_stage_cards(rows: list[dict[str, Any]], total_ms: Any) -> None:
    cards: list[str] = []
    for row in rows:
        key = str(row.get("key") or "")
        skipped = bool(row.get("skipped"))
        total_value = row.get("total_ms")
        share_text, share_ratio = _share(total_value, total_ms)
        stats = row.get("stats") or {}
        detail_html = "".join(
            f'<div class="aoi-timing-stat"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>'
            for label, value in stats.items()
        )
        cards.append(
            '<div class="aoi-timing-card">'
            '<div class="aoi-timing-head">'
            f'<div class="aoi-timing-title">{escape(str(row.get("label") or STAGE_LABELS.get(key, key)))}</div>'
            f'<div class="aoi-timing-value">{escape(_ms(total_value, skipped=skipped))}</div>'
            '</div>'
            f'<div class="aoi-timing-description">{escape(STAGE_DESCRIPTIONS.get(key, "当前阶段的真实耗时统计。"))}</div>'
            f'<div class="aoi-timing-stats">{detail_html}</div>'
            f'<div class="aoi-timing-share" title="占总耗时 {escape(share_text)}"><span style="width:{share_ratio * 100:.2f}%"></span></div>'
            '</div>'
        )
    if cards:
        st.markdown(f'<div class="aoi-timing-grid">{"".join(cards)}</div>', unsafe_allow_html=True)
    else:
        st.info("当前结果未记录可展示的阶段耗时。")


def _summary_cards(values: list[tuple[str, str]]) -> None:
    for start in range(0, len(values), 4):
        batch = values[start : start + 4]
        columns = st.columns(len(batch))
        for column, (label, value) in zip(columns, batch):
            column.metric(label, value)


def render_single_timing(
    timing: dict[str, Any] | None, *, generate_visualizations: bool, timing_mode: str = ""
) -> None:
    timing = timing or {}
    if not timing:
        return
    _install_styles()
    total = timing.get("end_to_end_ms")
    model_load = _number(timing.get("model_load_ms"))
    memory_load = _number(timing.get("memory_bank_load_ms"))
    model_prepare = None if model_load is None and memory_load is None else (model_load or 0.0) + (memory_load or 0.0)
    st.markdown("#### 性能耗时")
    st.caption(
        f"计时模式：{timing_mode or '当前执行模式'} · 可视化：{'已生成' if generate_visualizations else '未执行'}"
    )
    _summary_cards(
        [
            ("端到端总耗时", _ms(total)),
            ("模型准备", _ms(model_prepare)),
            ("图像预处理", _ms(timing.get("preprocess_ms"))),
            ("特征提取", _ms(timing.get("feature_extraction_ms"))),
            ("近邻检索", _ms(timing.get("ann_search_ms"))),
            ("后处理", _ms(timing.get("postprocess_ms"))),
            ("结果保存", _ms(timing.get("save_ms"))),
        ]
    )
    with st.expander("查看阶段耗时", expanded=False):
        st.caption(
            "模型与记忆库已复用，因此准备阶段可能显示为未单独统计。"
            if timing.get("model_reused") and timing.get("memory_bank_reused")
            else "首次任务会单独记录模型和记忆库加载耗时。"
        )
        rows = [
            {"key": "model_load_ms", "label": "模型加载", "total_ms": timing.get("model_load_ms")},
            {"key": "memory_bank_load_ms", "label": "记忆库加载", "total_ms": timing.get("memory_bank_load_ms")},
        ]
        for key, label in STAGE_LABELS.items():
            if key == "frame_total_ms":
                continue
            rows.append(
                {
                    "key": key,
                    "label": label,
                    "total_ms": timing.get(key),
                    "skipped": not generate_visualizations and key in {"visualization_ms", "encode_ms"},
                    "stats": {"占总耗时": _share(timing.get(key), total)[0]},
                }
            )
        _render_stage_cards(rows, total)


def render_batch_timing(
    timing: dict[str, Any] | None,
    *,
    generate_visualizations: bool,
    failed: int = 0,
    timing_mode: str = "",
) -> None:
    timing = timing or {}
    if not timing:
        return
    _install_styles()
    total = timing.get("task_end_to_end_ms") or timing.get("end_to_end_ms")
    st.markdown("#### 批量性能统计")
    st.caption(
        f"计时模式：{timing_mode or '当前执行模式'} · 可视化：{'已生成' if generate_visualizations else '未执行'} · 阶段分布仅统计成功样本。"
    )
    _summary_cards(
        [
            ("任务端到端", _ms(total)),
            ("处理吞吐", _fps(timing.get("throughput_fps") or timing.get("throughput_images_per_second"))),
            ("成功样本", str(int(timing.get("successful_samples") or 0))),
            ("失败样本", str(int(failed or 0))),
        ]
    )
    with st.expander("查看各阶段总计与样本分布", expanded=False):
        rows = [
            {"key": "model_load_ms", "label": "模型加载", "total_ms": timing.get("model_load_ms"), "stats": {"说明": "任务级一次"}},
            {"key": "memory_bank_load_ms", "label": "记忆库加载", "total_ms": timing.get("memory_bank_load_ms"), "stats": {"说明": "任务级一次"}},
        ]
        for key, label in STAGE_LABELS.items():
            stats = (timing.get("stages") or {}).get(key) or {}
            if not stats and key not in {"visualization_ms", "encode_ms"}:
                continue
            rows.append(
                {
                    "key": key,
                    "label": label,
                    "total_ms": stats.get("total_ms"),
                    "skipped": not generate_visualizations and key in {"visualization_ms", "encode_ms"},
                    "stats": {
                        "平均": _ms(stats.get("mean_ms")),
                        "P50": _ms(stats.get("p50_ms")),
                        "P95": _ms(stats.get("p95_ms")),
                        "最大": _ms(stats.get("max_ms")),
                    },
                }
            )
        _render_stage_cards(rows, total)


def render_video_timing(summary: dict[str, Any] | None) -> None:
    summary = summary or {}
    timing = summary.get("timing") or {}
    if not timing:
        return
    _install_styles()
    generate_visualizations = bool(summary.get("generate_visualizations", True))
    total = timing.get("end_to_end_ms")
    frame_stats = (timing.get("stages") or {}).get("frame_total_ms") or {}
    st.markdown("#### 视频性能耗时")
    st.caption(
        f"计时模式：{summary.get('timing_mode') or '当前执行模式'} · 可视化：{'已生成' if generate_visualizations else '未执行'}"
    )
    _summary_cards(
        [
            ("端到端耗时", _ms(total)),
            ("纯推理 FPS", _fps(timing.get("pure_inference_fps"))),
            ("完整处理 FPS", _fps(timing.get("full_pipeline_fps"))),
            ("平均逐帧耗时", _ms(frame_stats.get("mean_ms"))),
        ]
    )
    st.caption(
        f"逐帧全流程 P50 {_ms(frame_stats.get('p50_ms'))} · P95 {_ms(frame_stats.get('p95_ms'))}；"
        "纯推理 FPS 不包含视频解码、可视化和编码。"
    )
    with st.expander("查看视频各阶段统计", expanded=False):
        rows = [
            {"key": "model_load_ms", "label": "模型加载", "total_ms": timing.get("model_load_ms"), "stats": {"说明": "任务级一次"}},
            {"key": "memory_bank_load_ms", "label": "记忆库加载", "total_ms": timing.get("memory_bank_load_ms"), "stats": {"说明": "任务级一次"}},
            {"key": "save_ms", "label": "结果保存（任务总计）", "total_ms": timing.get("save_ms"), "stats": {"说明": "任务级统计"}},
        ]
        for key, label in STAGE_LABELS.items():
            stats = (timing.get("stages") or {}).get(key) or {}
            if not stats and key not in {"visualization_ms", "encode_ms"}:
                continue
            rows.append(
                {
                    "key": key,
                    "label": label,
                    "total_ms": stats.get("total_ms"),
                    "skipped": not generate_visualizations and key in {"visualization_ms", "encode_ms"},
                    "stats": {
                        "平均": _ms(stats.get("mean_ms")),
                        "P50": _ms(stats.get("p50_ms")),
                        "P95": _ms(stats.get("p95_ms")),
                        "最大": _ms(stats.get("max_ms")),
                    },
                }
            )
        _render_stage_cards(rows, total)


__all__ = ["render_single_timing", "render_batch_timing", "render_video_timing"]
