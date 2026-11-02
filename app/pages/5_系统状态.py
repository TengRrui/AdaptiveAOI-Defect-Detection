from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

from friendly_display import friendly_value, render_log_list, render_model_status
from ui_common import configure_page, format_value, label_for_pred, sidebar_runtime_controls

from service.status_service import (
    latest_inference,
    read_json,
    read_text_tail,
    speed_summary_path,
    system_status,
)


def _html(value: Any) -> str:
    return escape(str(value if value not in (None, "") else "-"))


def _metric_card(label: str, value: str, foot: str, icon: str) -> None:
    st.markdown(
        f"""
        <div class="aoi-kpi-card">
          <div class="aoi-kpi-top">
            <div class="aoi-kpi-label">{_html(label)}</div>
            <div class="aoi-kpi-icon">{_html(icon)}</div>
          </div>
          <div class="aoi-kpi-value">{_html(value)}</div>
          <div class="aoi-kpi-foot">{_html(foot)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _hero(runtime: dict[str, Any]) -> None:
    st.markdown(
        f"""
        <div class="aoi-workbench-hero">
          <div>
            <div class="aoi-workbench-kicker">系统运行观测</div>
            <div class="aoi-workbench-title">系统状态</div>
            <div class="aoi-workbench-copy">
              集中查看算法服务、功能模块、模型资源、最近推理与真实性能摘要；内部路径和标识默认隐藏。
            </div>
          </div>
          <div class="aoi-workbench-meta">
            <span class="aoi-badge">数据集：{_html(runtime.get("dataset"))}</span>
            <span class="aoi-badge">类别：{_html(runtime.get("category"))}</span>
            <span class="aoi-badge">运行设备：{_html(runtime.get("device"))}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _panel_rows(title: str, subtitle: str, rows: list[tuple[str, Any]]) -> None:
    body = "\n".join(
        f"""
        <div class="aoi-mini-row">
          <div class="aoi-mini-label">{_html(label)}</div>
          <div class="aoi-mini-value">{_html(value)}</div>
        </div>
        """
        for label, value in rows
    )
    st.markdown(
        f"""
        <div class="aoi-panel">
          <div class="aoi-panel-title">{_html(title)}</div>
          <div class="aoi-panel-subtitle">{_html(subtitle)}</div>
          {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _status_badge(label: str, ok: bool) -> str:
    klass = "aoi-badge-success" if ok else "aoi-badge-danger"
    return f'<span class="aoi-badge {klass}">{_html(label)}</span>'


def _speed_summary_payload(path: Path) -> dict[str, Any] | None:
    payload = read_json(path)
    if isinstance(payload, list) and payload:
        last = payload[-1]
        return last if isinstance(last, dict) else None
    return payload if isinstance(payload, dict) else None


def _render_inference_technical(payload: dict[str, Any]) -> None:
    with st.expander("查看最近推理技术详情", expanded=False):
        scores = st.columns(3)
        scores[0].metric("异常分数", friendly_value(payload.get("score", payload.get("anomaly_score")), "score"))
        scores[1].metric("判定阈值", friendly_value(payload.get("threshold"), "threshold"))
        scores[2].metric("最终结果", label_for_pred(payload.get("pred_label", payload.get("prediction"))))
        st.dataframe(
            [
                {"分组": "运行配置", "项目": "数据集", "当前值": friendly_value(payload.get("dataset"), "dataset"), "说明": "当前检测使用的数据来源标识"},
                {"分组": "运行配置", "项目": "类别", "当前值": friendly_value(payload.get("category"), "category"), "说明": "当前产品或检测类别"},
                {"分组": "判别逻辑", "项目": "方法模式", "当前值": friendly_value(payload.get("method_mode"), "method_mode"), "说明": "本次检测实际使用的评分方式"},
                {"分组": "判别逻辑", "项目": "校准状态", "当前值": friendly_value(payload.get("calibration_applied"), "calibration_applied"), "说明": "是否应用缺陷引导校准"},
                {"分组": "耗时", "项目": "端到端总耗时", "当前值": friendly_value(payload.get("total_ms") or (payload.get("timing") or {}).get("end_to_end_ms"), "total_ms"), "说明": "从读取输入到结果可用的完整时间"},
                {"分组": "高级标识", "项目": "任务标识", "当前值": friendly_value(payload.get("task_id") or payload.get("job_id"), "task_id"), "说明": "用于日志追踪的缩略标识"},
            ],
            width="stretch",
            hide_index=True,
        )


def _render_speed_technical(payload: dict[str, Any]) -> None:
    with st.expander("查看最近性能技术详情", expanded=False):
        rows = []
        groups = [
            ("图像读取", "image_read_ms", "从磁盘读取测试图片"),
            ("图像预处理", "preprocess_ms", "尺寸调整和输入转换"),
            ("特征提取", "feature_ms", "骨干网络生成特征"),
            ("记忆库检索", "memory_score_ms", "与正常特征执行近邻匹配"),
            ("后处理", "postprocess_ms", "生成最终分数和区域结果"),
            ("完整处理", "total_ms", "当前计时口径下的端到端时间"),
        ]
        for label, key, description in groups:
            values = payload.get(key) or {}
            rows.append({
                "阶段": label,
                "平均耗时": friendly_value(values.get("mean") if isinstance(values, dict) else None, "total_ms"),
                "P50": friendly_value(values.get("median") if isinstance(values, dict) else None, "total_ms"),
                "P95": friendly_value(values.get("p95") if isinstance(values, dict) else None, "total_ms"),
                "说明": description,
            })
        st.dataframe(rows, width="stretch", hide_index=True)
        st.caption(
            f"运行设备：{friendly_value(payload.get('device'), 'device')} · "
            f"数值精度：{friendly_value(payload.get('precision'), 'precision')} · "
            f"测试轮数：{friendly_value(payload.get('runs'), 'count')}"
        )


configure_page("status")

runtime = sidebar_runtime_controls("status")
_hero(runtime)
# ``sidebar_runtime_controls`` is intentionally backed by the lightweight
# readiness snapshot.  During startup, after a resource is deactivated, or
# while the backend is reconnecting, that snapshot may not contain a complete
# descriptive runtime binding.  The status page should still render its
# service diagnostics in those states; direct indexing here used to turn a
# normal not-ready state into ``KeyError: 'dataset'`` (and hid the useful
# health information from the user).
status = system_status(
    runtime.get("dataset"),
    runtime.get("category"),
    runtime.get("config"),
)

path_status = status.get("path_status", {})
scripts = path_status.get("scripts", {})
found_scripts = path_status.get("found_scripts", [])
missing_scripts = path_status.get("missing_scripts", [])
memory = status.get("memory_bank") or {}
calibration = status.get("calibration") or {}
cuda = status.get("cuda", {})
api_health = (status.get("api") or {}).get("health") or {}
current_model = (api_health.get("current_model") or {})

if not status.get("ok", True):
    st.error(status.get("error", "系统状态读取失败。"))

cols = st.columns(5)
with cols[0]:
    _metric_card("算法服务", "正常" if status.get("ok") else "不可用", "当前后端健康状态", "服务")
with cols[1]:
    _metric_card("功能模块", f"{len(found_scripts)}/{len(scripts)}", "已通过文件可用性检查", "模块")
with cols[2]:
    memory_foot = str(memory.get("friendly_name") or (f"{memory.get('size_mb', 0)} MB" if memory.get("exists") else "当前无可用资源"))
    _metric_card("正常记忆库", "存在" if memory.get("exists") else "缺失", memory_foot, "记忆")
with cols[3]:
    threshold_text = friendly_value(calibration.get("threshold"), "threshold") if calibration.get("threshold") is not None else "当前阈值资源"
    _metric_card("校准参数", "存在" if calibration.get("exists") else "缺失", threshold_text, "校准")
with cols[4]:
    _metric_card("图形加速", "可用" if cuda.get("cuda_available") else "不可用", str(cuda.get("gpu_name") or cuda.get("torch") or "未识别")[:28], "GPU")

overview_tab, asset_tab, result_tab, log_tab = st.tabs(["运行概览", "脚本与资产", "结果与速度", "日志"])

with overview_tab:
    left, right = st.columns([1.05, 0.95])
    with left:
        _panel_rows(
            "算法运行环境",
            "用于确认本机和算法运行环境",
            [
                ("运行解释器", "已配置" if status.get("algorithm_python") else "未找到"),
                ("运行设备", status.get("device", "未配置")),
                ("CUDA 状态", "✓ 可用" if cuda.get("cuda_available") else "○ 不可用"),
                ("PyTorch 版本", cuda.get("torch") or "未识别"),
                ("CUDA 说明", cuda.get("error") or "当前无异常"),
            ],
        )
    with right:
        st.markdown(
            f"""
            <div class="aoi-panel">
              <div class="aoi-panel-title">项目路径</div>
              <div class="aoi-panel-subtitle">软件桥接到算法核心，不复制算法代码</div>
              <div class="aoi-mini-row">
                <div class="aoi-mini-label">算法项目</div>
                <div class="aoi-mini-value">{_status_badge("存在", bool(status.get("ok")))}</div>
              </div>
              <div class="aoi-soft-divider"></div>
              <div class="aoi-mini-row">
                <div class="aoi-mini-label">算法输出目录</div>
                <div class="aoi-mini-value">outputs</div>
              </div>
              <div class="aoi-panel-subtitle">路径信息由后端管理，未在检测界面显示。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if current_model:
        render_model_status(
            {
                "model_type": current_model.get("model_type"),
                "backbone": current_model.get("backbone"),
                "profile_valid": current_model.get("profile_valid"),
                "memory_bank_ready": current_model.get("weights_valid"),
                "calibration_ready": current_model.get("calibration_valid"),
                "loaded": current_model.get("loaded"),
                "default_threshold": current_model.get("default_threshold"),
            },
            "当前运行方案模型状态",
        )
    else:
        st.info("当前暂无可用模型状态。")

    worker_state = str(api_health.get("worker_state") or "STOPPED").upper()
    worker_required = bool(api_health.get("worker_required_for_runtime"))
    worker_alive = bool(api_health.get("worker_alive"))
    worker_heartbeat_age = api_health.get("worker_heartbeat_age_ms")
    if worker_required:
        worker_label = {
            "FAILED": "失败（检测链路不可用）",
            "STOPPED": "已停止（需检查）",
            "STARTING": "启动中",
            "ALIVE": "运行中",
            "IDLE": "空闲",
        }.get(worker_state, "状态未知")
        worker_note = "该 Worker 是 Runtime Ready 的必要条件。"
    else:
        worker_label = {
            "FAILED": "辅助异常（不影响检测）",
            "STOPPED": "未常驻（初始化已完成）",
            "STARTING": "辅助启动中",
            "ALIVE": "辅助运行中",
            "IDLE": "辅助空闲",
        }.get(worker_state, "辅助状态未知")
        worker_note = "当前 Worker 仅负责一次性启动/模型预热，不是检测 Ready 的必要条件。"
    heartbeat_text = (
        f"{int(worker_heartbeat_age)} ms"
        if worker_alive and worker_heartbeat_age is not None
        else "不适用（Worker 未常驻）"
    )
    _panel_rows(
        "运行时与后台 Worker",
        "两个生命周期独立展示，避免把辅助线程结束误判为检测服务故障。",
        [
            ("运行时状态", str(api_health.get("runtime_state") or api_health.get("status") or "未知")),
            ("检测 Ready", "✓ 可用" if api_health.get("runtime_ready", api_health.get("ready")) else "○ 不可用"),
            ("Worker 状态", worker_label),
            ("Worker 存活", "是" if worker_alive else "否"),
            ("Worker 心跳年龄", heartbeat_text),
            ("说明", worker_note),
        ],
    )

with asset_tab:
    module_map = {
        "scripts/build_manifest.py": ("样本清单服务", "整理并校验输入样本清单"),
        "scripts/build_memory.py": ("记忆库管理服务", "构建正常特征记忆库"),
        "scripts/calibrate.py": ("阈值校准服务", "根据标注样本计算检测阈值"),
        "scripts/infer_image.py": ("单图检测服务", "处理单张图片并生成判定"),
        "scripts/infer_batch.py": ("批量检测服务", "处理目录中的多张图片"),
        "scripts/infer_video.py": ("视频检测服务", "对本地视频执行逐帧检测"),
        "scripts/benchmark_speed.py": ("性能统计服务", "测量推理阶段和整体耗时"),
    }
    checked_at = str(path_status.get("checked_at") or "-")
    script_rows = []
    for rel_path, item in scripts.items():
        module_name, description = module_map.get(rel_path, ("算法功能模块", "系统运行所需模块"))
        available = bool(item.get("exists"))
        script_rows.append({
            "功能模块": module_name,
            "当前状态": "正常" if available else "不可用",
            "最近检查时间": checked_at,
            "是否可用": "是" if available else "否",
            "简短说明": description,
        })
    st.markdown(
        """
        <div class="aoi-section-header">
          <div>
            <div class="aoi-section-title">算法脚本状态</div>
            <div class="aoi-section-desc">缺失脚本会导致对应页面显示友好错误，不会崩溃。</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(script_rows, width="stretch", hide_index=True)
    with st.expander("查看模块技术详情", expanded=False):
        st.dataframe(
            [
                {
                    "功能模块": module_map.get(rel_path, ("算法功能模块", ""))[0],
                    "Python 模块": rel_path.replace("scripts/", "").replace(".py", ""),
                    "文件状态": "存在" if item.get("exists") else "缺失",
                }
                for rel_path, item in scripts.items()
            ],
            width="stretch",
            hide_index=True,
        )

    left, right = st.columns(2)
    with left:
        _panel_rows(
            "正常记忆库",
            "当前数据集和类别使用的冻结记忆库",
            [
                ("资源状态", "✓ 已就绪" if memory.get("exists") else "⚠ 缺失"),
                ("正常样本", friendly_value(memory.get("normal_sample_count"), "count")),
                ("资源名称", memory.get("friendly_name") or "当前冻结记忆库"),
                ("位置", "算法核心受控资产"),
            ],
        )
    with right:
        _panel_rows(
            "校准参数",
            "当前数据集和类别使用的冻结校准参数",
            [
                ("资源状态", "✓ 已完成" if calibration.get("exists") else "⚠ 缺失"),
                ("当前阈值", friendly_value(calibration.get("threshold"), "threshold")),
                ("校准方法", calibration.get("method") or "当前冻结配置"),
                ("位置", "算法核心受控资产"),
            ],
        )
    if missing_scripts:
        st.warning("部分算法功能模块不可用；请展开技术详情查看模块简称。")

with result_tab:
    latest = latest_inference()
    left, right = st.columns([0.95, 1.05])
    with left:
        if latest:
            st.markdown("#### 最近推理记录")
            latest_name = latest.get("filename") or latest.get("image_name") or latest.get("sample_name") or "最近样本"
            st.dataframe(
                [{
                    "时间": friendly_value(latest.get("created_at") or latest.get("time"), "created_at"),
                    "检测类型": "单图检测",
                    "样本名称": friendly_value(latest_name, "filename"),
                    "检测结果": label_for_pred(latest.get("pred_label", latest.get("prediction"))),
                    "异常分数": format_value(latest.get("score", latest.get("anomaly_score"))),
                    "总耗时": friendly_value(latest.get("total_ms") or (latest.get("timing") or {}).get("end_to_end_ms"), "total_ms"),
                    "当前方案": friendly_value(latest.get("profile_name") or latest.get("method_mode") or "当前运行方案", "method_mode"),
                    "状态": "已完成",
                }],
                width="stretch",
                hide_index=True,
            )
        else:
            st.markdown(
                """
                <div class="aoi-empty-state">暂无最近推理记录。完成一次单图检测后这里会自动更新。</div>
                """,
                unsafe_allow_html=True,
            )
    with right:
        speed_path = speed_summary_path(runtime.get("config"))
        speed_payload = _speed_summary_payload(speed_path)
        if speed_payload:
            total_ms = speed_payload.get("total_ms", {})
            timing_mode = str(speed_payload.get("timing_mode") or "strict_total")
            mode_label = {
                "detection_only": "纯检测耗时",
                "no_save_total": "不保存结果耗时",
                "online": "在线处理耗时",
                "strict": "完整严格耗时",
                "strict_total": "完整严格耗时",
            }.get(timing_mode, "完整任务耗时")
            mean_ms = total_ms.get("mean") if isinstance(total_ms, dict) else None
            p50_ms = total_ms.get("median") if isinstance(total_ms, dict) else None
            p95_ms = total_ms.get("p95") if isinstance(total_ms, dict) else None
            throughput = 1000.0 / float(mean_ms) if mean_ms not in (None, 0) else None
            st.markdown("#### 最近速度摘要")
            st.dataframe(
                [{
                    "检测类型": "单图检测性能测试",
                    "样本数量": speed_payload.get("runs") or "未记录",
                    "平均耗时": friendly_value(mean_ms, "total_ms"),
                    "P50": friendly_value(p50_ms, "total_ms"),
                    "P95": friendly_value(p95_ms, "total_ms"),
                    "吞吐率": f"{throughput:.2f} 张/秒" if throughput is not None else "未记录",
                    "可视化": "未单独记录",
                    "计时口径": mode_label,
                }],
                width="stretch",
                hide_index=True,
            )
        else:
            st.markdown(
                f"""
                <div class="aoi-empty-state">未找到最近速度摘要。</div>
                """,
                unsafe_allow_html=True,
            )
    if latest:
        _render_inference_technical(latest)
    else:
        st.info("暂无最近推理记录。")
    speed_path = speed_summary_path(runtime.get("config"))
    speed_payload_raw = read_json(speed_path)
    if speed_payload_raw:
        _render_speed_technical(speed_payload_raw[-1] if isinstance(speed_payload_raw, list) else speed_payload_raw)
    else:
        st.info("未找到最近速度摘要。")

with log_tab:
    st.markdown(
        """
        <div class="aoi-section-header">
          <div>
            <div class="aoi-section-title">软件调用日志</div>
            <div class="aoi-section-desc">用于运行前排查桥接命令、脚本报错和路径问题。</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### 近期调用记录")
    render_log_list(
        read_text_tail(status["call_history_path"], max_lines=80),
        default_level="info",
        empty_message="当前没有调用记录。",
    )
    st.markdown("#### 错误与处理建议")
    render_log_list(
        read_text_tail(status["error_log_path"], max_lines=80),
        default_level="error",
        empty_message="当前没有需要处理的错误日志。",
    )
