from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from core.domain_build_contract import DEFAULT_CREATE_MODE, LIGHTWEIGHT_CREATE_MODE
from performance_cache import api_client as client
from friendly_display import friendly_error_message
from ui_common import configure_page, sidebar_runtime_controls
from domain_resource_creation_page import render as render_domain_resource_creation


NORMAL_PATH_KEY = "domain:normal:image_dir"
DEFECT_PATH_KEY = "domain:defect:image_dir"
NORMAL_DIALOG_KEY = "domain:normal:image_dialog"
DEFECT_DIALOG_KEY = "domain:defect:image_dialog"
NORMAL_SOURCE_TYPE_KEY = "domain:normal:source_type"
DEFECT_SOURCE_TYPE_KEY = "domain:defect:source_type"
NORMAL_VIDEO_FILES_KEY = "domain:normal:video_files"
DEFECT_VIDEO_FILES_KEY = "domain:defect:video_files"
NORMAL_VIDEO_DIR_KEY = "domain:normal:video_dir"
DEFECT_VIDEO_DIR_KEY = "domain:defect:video_dir"
NORMAL_VIDEO_FILE_DIALOG_KEY = "domain:normal:video_file_dialog"
DEFECT_VIDEO_FILE_DIALOG_KEY = "domain:defect:video_file_dialog"
NORMAL_VIDEO_DIR_DIALOG_KEY = "domain:normal:video_dir_dialog"
DEFECT_VIDEO_DIR_DIALOG_KEY = "domain:defect:video_dir_dialog"
NORMAL_VIDEO_TEXT_KEY = "domain:normal:video_paths"
DEFECT_VIDEO_TEXT_KEY = "domain:defect:video_paths"
DEFECT_INTERVAL_TEXT_KEY = "domain:defect:video_intervals"
DOMAIN_SAMPLING_KEY = "domain_video_sampling_config"
DOMAIN_BUDGET_KEY = "domain_sample_budget_config"
DOMAIN_SCAN_KEY = "domain_scan_summary"
DOMAIN_SCAN_PENDING_KEY = "domain_sample_scan_pending"
DOMAIN_SCAN_ERROR_KEY = "domain_sample_scan_error"
DOMAIN_PREVIEW_KEY = "domain_video_preview"
DOMAIN_EXCLUDED_KEY = "domain_video_excluded_sample_ids"
DOMAIN_JOB_KEY = "domain_adaptation_job_id"
DOMAIN_JOB_ACTIVE_KEY = "domain_adaptation_job_active"
DOMAIN_PROFILE_KEY = "domain_config_profile_key"
DOMAIN_PROFILE_PENDING_KEY = "domain_config_profile_pending_key"
TARGET_DOMAIN_RECENT_KEY = "target_domain_last_browse_directory"

LEGACY_DOMAIN_STATE_KEYS = {
    "domain_normal_dir": NORMAL_PATH_KEY,
    "domain_defect_dir": DEFECT_PATH_KEY,
    "domain_normal_dialog_id": NORMAL_DIALOG_KEY,
    "domain_defect_dialog_id": DEFECT_DIALOG_KEY,
    "domain_normal_source_type": NORMAL_SOURCE_TYPE_KEY,
    "domain_defect_source_type": DEFECT_SOURCE_TYPE_KEY,
    "domain_normal_video_files": NORMAL_VIDEO_FILES_KEY,
    "domain_defect_video_files": DEFECT_VIDEO_FILES_KEY,
    "domain_normal_video_dir": NORMAL_VIDEO_DIR_KEY,
    "domain_defect_video_dir": DEFECT_VIDEO_DIR_KEY,
    "domain_normal_video_file_dialog_id": NORMAL_VIDEO_FILE_DIALOG_KEY,
    "domain_defect_video_file_dialog_id": DEFECT_VIDEO_FILE_DIALOG_KEY,
    "domain_normal_video_dir_dialog_id": NORMAL_VIDEO_DIR_DIALOG_KEY,
    "domain_defect_video_dir_dialog_id": DEFECT_VIDEO_DIR_DIALOG_KEY,
    "domain_normal_video_paths_text": NORMAL_VIDEO_TEXT_KEY,
    "domain_defect_video_paths_text": DEFECT_VIDEO_TEXT_KEY,
    "domain_defect_video_intervals_text": DEFECT_INTERVAL_TEXT_KEY,
}

SAMPLING_WIDGET_KEYS = {
    field: f"domain:sampling:{field}"
    for field in (
        "sampling_method", "frame_interval", "time_interval_seconds", "target_fps",
        "max_frames_per_video", "final_max_samples", "seed", "start_time", "end_time",
        "similarity_threshold", "deduplicate", "filter_blur", "filter_brightness",
        "blur_threshold", "dark_threshold", "bright_threshold",
    )
}
BUDGET_MODE_WIDGET_KEY = "domain:budget:mode"
NORMAL_BUDGET_WIDGET_KEY = "domain:budget:normal_target_count"
DEFECT_BUDGET_WIDGET_KEY = "domain:budget:defect_target_count"

SOURCE_TYPE_LABELS = {"image": "图片", "video": "视频", "mixed": "图片与视频混合"}
VIDEO_SAMPLING_LABELS = {
    "auto": "自动抽取（代表帧）",
    "fixed_frame_interval": "固定帧间隔",
    "fixed_time_interval": "固定时间间隔",
    "target_fps": "目标抽帧 FPS",
    "uniform_count": "均匀抽取指定数量",
}
VIDEO_FILE_TYPES = ["mp4", "avi", "mov", "mkv", "m4v", "webm"]
LIGHTWEIGHT_MODE = LIGHTWEIGHT_CREATE_MODE


def _error(payload: dict[str, Any], fallback: str) -> str:
    return friendly_error_message(payload.get("error"), fallback)


def _migrate_legacy_domain_state() -> None:
    """Move old session keys once into the stable page/section/field namespace."""

    for legacy_key, stable_key in LEGACY_DOMAIN_STATE_KEYS.items():
        if stable_key not in st.session_state and legacy_key in st.session_state:
            st.session_state[stable_key] = st.session_state[legacy_key]
        st.session_state.pop(legacy_key, None)


def _invalidate_domain_inputs() -> None:
    """Discard every derived view when a target-domain input changes."""
    for key in (
        DOMAIN_SCAN_KEY,
        DOMAIN_PREVIEW_KEY,
        "domain_rc_conditions",
        "domain_rc_conditions_fingerprint",
        "domain_rc_candidate_validated",
        "domain_rc_candidate_frozen",
    ):
        st.session_state.pop(key, None)
    st.session_state[DOMAIN_SCAN_PENDING_KEY] = True


def _poll_dialog(dialog_key: str, path_key: str) -> None:
    dialog_id = str(st.session_state.get(dialog_key) or "")
    if not dialog_id:
        return
    response = client().folder_dialog_status(dialog_id, consume=True)
    if not response.get("success"):
        if bool((response.get("error") or {}).get("retryable")):
            st.session_state[f"{dialog_key}_error"] = "状态刷新稍慢，本地选择窗口仍可继续操作。"
            return
        st.session_state.pop(dialog_key, None)
        st.session_state[f"{dialog_key}_error"] = _error(response, "文件夹选择状态读取失败。")
        return
    dialog = response.get("folder_dialog") or {}
    status = str(dialog.get("status") or "")
    if status in {"starting", "open", "busy"}:
        st.session_state.pop(f"{dialog_key}_error", None)
        return
    st.session_state.pop(dialog_key, None)
    if status == "selected" and dialog.get("path"):
        st.session_state[path_key] = str(dialog["path"])
        # The next normal/defect selection should start at the sibling-friendly
        # parent directory, not inside the directory that was just selected.
        st.session_state[TARGET_DOMAIN_RECENT_KEY] = str(Path(str(dialog["path"])).parent)
        st.session_state.pop(f"{dialog_key}_error", None)
        _invalidate_domain_inputs()
    elif status == "error":
        st.session_state[f"{dialog_key}_error"] = str(dialog.get("message") or "无法打开文件夹选择窗口。")


def _start_dialog(dialog_key: str, path_key: str, recent_key: str) -> None:
    response = client().start_folder_dialog(
        str(st.session_state.get(path_key) or st.session_state.get(TARGET_DOMAIN_RECENT_KEY) or ""),
        recent_key=TARGET_DOMAIN_RECENT_KEY,
    )
    if not response.get("success"):
        st.session_state[f"{dialog_key}_error"] = _error(response, "无法打开 Windows 文件夹选择窗口。")
        return
    dialog = response.get("folder_dialog") or {}
    if dialog.get("status") == "busy":
        st.session_state[f"{dialog_key}_error"] = "已有文件夹选择窗口，请先完成或取消。"
    elif dialog.get("dialog_id"):
        st.session_state[dialog_key] = str(dialog["dialog_id"])


def _poll_video_dialog(dialog_key: str, files_key: str, text_key: str) -> None:
    dialog_id = str(st.session_state.get(dialog_key) or "")
    if not dialog_id:
        return
    response = client().folder_dialog_status(dialog_id, consume=True)
    if not response.get("success"):
        if bool((response.get("error") or {}).get("retryable")):
            st.session_state[f"{dialog_key}_error"] = "状态刷新稍慢，本地选择窗口仍可继续操作。"
            return
        st.session_state.pop(dialog_key, None)
        st.session_state[f"{dialog_key}_error"] = _error(response, "视频选择窗口状态读取失败。")
        return
    dialog = response.get("folder_dialog") or {}
    status = str(dialog.get("status") or "")
    if status in {"starting", "open", "busy"}:
        st.session_state.pop(f"{dialog_key}_error", None)
        return
    st.session_state.pop(dialog_key, None)
    if status == "selected":
        values = [str(item) for item in (dialog.get("paths") or []) if str(item).strip()]
        if not values and dialog.get("path"):
            values = [str(dialog["path"])]
        st.session_state[files_key] = values
        st.session_state[text_key] = "\n".join(values)
        if values:
            st.session_state[TARGET_DOMAIN_RECENT_KEY] = str(Path(values[0]).parent)
        st.session_state.pop(f"{dialog_key}_error", None)
        _invalidate_domain_inputs()
    elif status == "error":
        st.session_state[f"{dialog_key}_error"] = str(dialog.get("message") or "无法打开视频选择窗口。")


def _poll_single_dialog(dialog_key: str, path_key: str) -> None:
    dialog_id = str(st.session_state.get(dialog_key) or "")
    if not dialog_id:
        return
    response = client().folder_dialog_status(dialog_id, consume=True)
    if not response.get("success"):
        if bool((response.get("error") or {}).get("retryable")):
            st.session_state[f"{dialog_key}_error"] = "状态刷新稍慢，本地选择窗口仍可继续操作。"
            return
        st.session_state.pop(dialog_key, None)
        st.session_state[f"{dialog_key}_error"] = _error(response, "文件夹选择窗口状态读取失败。")
        return
    dialog = response.get("folder_dialog") or {}
    status = str(dialog.get("status") or "")
    if status in {"starting", "open", "busy"}:
        st.session_state.pop(f"{dialog_key}_error", None)
        return
    st.session_state.pop(dialog_key, None)
    if status == "selected" and dialog.get("path"):
        st.session_state[path_key] = str(dialog["path"])
        st.session_state[TARGET_DOMAIN_RECENT_KEY] = str(Path(str(dialog["path"])).parent)
        _invalidate_domain_inputs()
    elif status == "error":
        st.session_state[f"{dialog_key}_error"] = str(dialog.get("message") or "无法打开文件夹选择窗口。")


def _start_video_dialog(
    dialog_key: str,
    initial_key: str,
    *,
    multiple: bool,
    recent_key: str,
    files_key: str = "",
    text_key: str = "",
) -> None:
    response = client().start_folder_dialog(
        str(st.session_state.get(initial_key) or st.session_state.get(TARGET_DOMAIN_RECENT_KEY) or ""),
        dialog_type="files" if multiple else "folder",
        multiple=multiple,
        file_types=VIDEO_FILE_TYPES,
        recent_key=TARGET_DOMAIN_RECENT_KEY,
    )
    if not response.get("success"):
        st.session_state[f"{dialog_key}_error"] = _error(response, "无法打开 Windows 视频选择窗口。")
        return
    dialog = response.get("folder_dialog") or {}
    if dialog.get("status") == "busy":
        st.session_state[f"{dialog_key}_error"] = "已有本地选择窗口，请先完成或取消。"
    elif dialog.get("dialog_id"):
        st.session_state[dialog_key] = str(dialog["dialog_id"])


def _parse_video_intervals(value: str) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for line in str(value or "").splitlines():
        text = line.strip()
        if not text or ":" not in text:
            continue
        name, ranges = text.split(":", 1)
        name = name.strip()
        if not name:
            continue
        values: list[dict[str, Any]] = []
        for item in ranges.split(","):
            token = item.strip()
            if not token or "-" not in token:
                continue
            start, end = [part.strip() for part in token.split("-", 1)]
            try:
                if start.lower().endswith("f") or end.lower().endswith("f"):
                    values.append({"start_frame": int(start.rstrip("fF")), "end_frame": int(end.rstrip("fF"))})
                else:
                    values.append({"start_time": float(start), "end_time": float(end)})
            except ValueError:
                continue
        if values:
            result[name] = values
    return result


def _domain_input_payload() -> dict[str, Any]:
    sampling = dict(st.session_state.get(DOMAIN_SAMPLING_KEY) or {})
    budget = dict(st.session_state.get(DOMAIN_BUDGET_KEY) or {})
    needs_defect = str(st.session_state.get("domain_rc_mode") or DEFAULT_CREATE_MODE) == LIGHTWEIGHT_MODE
    return {
        "normal_dir": str(st.session_state.get(NORMAL_PATH_KEY) or "").strip(),
        "defect_dir": str(st.session_state.get(DEFECT_PATH_KEY) or "").strip() if needs_defect else "",
        "normal_source_type": str(st.session_state.get(NORMAL_SOURCE_TYPE_KEY) or "image"),
        "defect_source_type": str(st.session_state.get(DEFECT_SOURCE_TYPE_KEY) or "image") if needs_defect else "image",
        "normal_image_dir": str(st.session_state.get(NORMAL_PATH_KEY) or "").strip(),
        "defect_image_dir": str(st.session_state.get(DEFECT_PATH_KEY) or "").strip() if needs_defect else "",
        "normal_video_files": [str(value).strip() for value in list(st.session_state.get(NORMAL_VIDEO_FILES_KEY) or []) if str(value).strip()],
        "defect_video_files": [str(value).strip() for value in list(st.session_state.get(DEFECT_VIDEO_FILES_KEY) or []) if str(value).strip()] if needs_defect else [],
        "normal_video_dir": str(st.session_state.get(NORMAL_VIDEO_DIR_KEY) or "").strip(),
        "defect_video_dir": str(st.session_state.get(DEFECT_VIDEO_DIR_KEY) or "").strip() if needs_defect else "",
        "video_sampling_config": sampling,
        "defect_video_intervals": _parse_video_intervals(str(st.session_state.get(DEFECT_INTERVAL_TEXT_KEY) or "")) if needs_defect else {},
        "excluded_sample_ids": list(st.session_state.get(DOMAIN_EXCLUDED_KEY) or []),
        "sample_budget_mode": str(budget.get("sample_budget_mode") or "final_frame_count"),
        "normal_target_count": int(budget.get("normal_target_count") or 100),
        "defect_target_count": int(budget.get("defect_target_count") or 30) if needs_defect else 0,
    }


def _sample_scan_payload() -> dict[str, Any]:
    payload = _domain_input_payload()
    payload.update(
        method_mode=str(st.session_state.get("domain_rc_mode") or DEFAULT_CREATE_MODE),
    )
    return payload


def _run_pending_sample_scan() -> None:
    if not st.session_state.get(DOMAIN_SCAN_PENDING_KEY):
        return
    if any(
        st.session_state.get(key)
        for key in (
            NORMAL_DIALOG_KEY,
            DEFECT_DIALOG_KEY,
            NORMAL_VIDEO_FILE_DIALOG_KEY,
            DEFECT_VIDEO_FILE_DIALOG_KEY,
            NORMAL_VIDEO_DIR_DIALOG_KEY,
            DEFECT_VIDEO_DIR_DIALOG_KEY,
        )
    ):
        return
    payload = _sample_scan_payload()
    has_source = any(
        (
            payload.get("normal_image_dir"),
            payload.get("defect_image_dir"),
            payload.get("normal_video_files"),
            payload.get("defect_video_files"),
            payload.get("normal_video_dir"),
            payload.get("defect_video_dir"),
        )
    )
    if not has_source:
        st.session_state.pop(DOMAIN_SCAN_KEY, None)
        st.session_state[DOMAIN_SCAN_PENDING_KEY] = False
        return
    with st.status("正在扫描所选样本…", expanded=True) as status:
        st.write("只读取刚选择的目录；重复扫描会复用目录签名缓存。")
        response = client().scan_domain_samples(**payload)
        if response.get("success"):
            scan = response.get("scan") or {}
            st.session_state[DOMAIN_SCAN_KEY] = scan
            st.session_state[DOMAIN_SCAN_PENDING_KEY] = False
            st.session_state.pop(DOMAIN_SCAN_ERROR_KEY, None)
            st.session_state.pop("domain_rc_conditions_fingerprint", None)
            st.session_state.pop("domain_rc_conditions", None)
            st.session_state["domain_rc_scan_notice"] = "样本扫描完成，构建条件已按当前路径重新计算。"
            status.update(label="样本扫描完成", state="complete", expanded=False)
        else:
            message = _error(response, "样本扫描失败，请检查路径、权限和样本格式。")
            st.session_state[DOMAIN_SCAN_PENDING_KEY] = False
            st.session_state[DOMAIN_SCAN_ERROR_KEY] = message
            status.update(label="样本扫描失败", state="error", expanded=True)
            st.error(message)


def _folder_row(label: str, path_key: str, dialog_key: str, help_text: str, recent_key: str) -> None:
    # Every native selector shares the backend's single-dialog lock.  Keep the
    # image-folder buttons disabled while a video selector is open as well, so
    # a fast double click cannot create competing Windows dialogs.
    any_dialog_open = any(
        st.session_state.get(key)
        for key in (
            NORMAL_DIALOG_KEY,
            DEFECT_DIALOG_KEY,
            NORMAL_VIDEO_FILE_DIALOG_KEY,
            DEFECT_VIDEO_FILE_DIALOG_KEY,
            NORMAL_VIDEO_DIR_DIALOG_KEY,
            DEFECT_VIDEO_DIR_DIALOG_KEY,
        )
    )
    path_col, button_col = st.columns([4.1, 1.35], gap="small", vertical_alignment="bottom")
    with path_col:
        st.text_input(
            label,
            key=path_key,
            placeholder="输入文件夹路径，或使用“选择文件夹”按钮浏览目录",
            help=help_text,
            on_change=_invalidate_domain_inputs,
        )
    with button_col:
        if st.button(
            "选择文件夹",
            icon=":material/folder_open:",
            key=f"choose_{dialog_key}",
            disabled=any_dialog_open,
            width="stretch",
        ):
            _start_dialog(dialog_key, path_key, recent_key)
    if st.session_state.get(dialog_key):
        st.info("Windows 文件夹选择窗口已打开，请完成选择或取消。")
    if error := str(st.session_state.get(f"{dialog_key}_error") or ""):
        st.warning(error)


def _sample_folder_controls_content() -> None:
    needs_defect = str(st.session_state.get("domain_rc_mode") or DEFAULT_CREATE_MODE) == LIGHTWEIGHT_MODE
    st.markdown("#### 样本来源类型")
    st.session_state.setdefault(NORMAL_SOURCE_TYPE_KEY, "image")
    st.session_state.setdefault(DEFECT_SOURCE_TYPE_KEY, "image")
    source_columns = st.columns(2) if needs_defect else (st,)
    source_columns[0].selectbox("正常样本来源", list(SOURCE_TYPE_LABELS), key=NORMAL_SOURCE_TYPE_KEY, format_func=lambda value: SOURCE_TYPE_LABELS[value], on_change=_invalidate_domain_inputs)
    if needs_defect:
        source_columns[1].selectbox("缺陷样本来源", list(SOURCE_TYPE_LABELS), key=DEFECT_SOURCE_TYPE_KEY, format_func=lambda value: SOURCE_TYPE_LABELS[value], on_change=_invalidate_domain_inputs)
    else:
        st.caption("基础正常记忆只需要正常样本；切换到缺陷引导轻量校准后，页面才会显示缺陷样本来源。")

    any_dialog_open = any(st.session_state.get(key) for key in (
        NORMAL_DIALOG_KEY, DEFECT_DIALOG_KEY, NORMAL_VIDEO_FILE_DIALOG_KEY,
        DEFECT_VIDEO_FILE_DIALOG_KEY, NORMAL_VIDEO_DIR_DIALOG_KEY, DEFECT_VIDEO_DIR_DIALOG_KEY,
    ))

    def render_role(prefix: str, label: str, source_key: str, image_path_key: str, image_dialog_key: str, video_files_key: str, video_text_key: str, video_dir_key: str, video_file_dialog_key: str, video_dir_dialog_key: str) -> None:
        source_type = str(st.session_state.get(source_key) or "image")
        st.markdown(f"#### {label}")
        if prefix == "normal":
            st.caption("正常图片和正常视频代表帧可以进入正常记忆库；连续相似帧不会重复计入样本预算。")
        else:
            st.caption("缺陷视频帧不会写入正常记忆库，仅用于当前轻量校准器的受控拟合与验证。")
        if source_type in {"image", "mixed"}:
            _folder_row("正常图片文件夹" if prefix == "normal" else "缺陷校准图片文件夹", image_path_key, image_dialog_key, "支持中文、空格和长路径；不能选择测试输出、备份或报告目录。", f"domain_{prefix}_images")
        if source_type in {"video", "mixed"}:
            st.session_state.setdefault(video_files_key, [])
            st.session_state.setdefault(video_text_key, "")
            st.session_state.setdefault(video_dir_key, "")
            text = st.text_area(
                "正常视频文件（每行一个路径）" if prefix == "normal" else "缺陷视频文件（每行一个路径）",
                key=video_text_key,
                height=78,
                placeholder="可手动填写，也可使用“选择视频文件”按钮浏览并选择多个文件",
                help="支持 mp4、avi、mov、mkv、m4v、webm；路径仅用于本机目标域构建。",
                on_change=_invalidate_domain_inputs,
            )
            st.session_state[video_files_key] = [line.strip() for line in str(text or "").splitlines() if line.strip()]
            st.text_input(
                "正常视频文件夹路径（可选）" if prefix == "normal" else "缺陷视频文件夹路径（可选）",
                key=video_dir_key,
                placeholder="输入文件夹路径，或使用“选择视频文件夹”按钮浏览目录",
                help="后端只扫描该目录中的支持视频格式，不会扫描整个磁盘。",
                on_change=_invalidate_domain_inputs,
            )
            file_col, dir_col = st.columns(2)
            with file_col:
                if st.button("选择视频文件", icon=":material/video_file:", key=f"choose_{video_file_dialog_key}", disabled=any_dialog_open, width="stretch"):
                    _start_video_dialog(
                        video_file_dialog_key,
                        video_dir_key,
                        multiple=True,
                        recent_key=TARGET_DOMAIN_RECENT_KEY,
                        files_key=video_files_key,
                        text_key=video_text_key,
                    )
            with dir_col:
                if st.button("选择视频文件夹", icon=":material/folder_open:", key=f"choose_{video_dir_dialog_key}", disabled=any_dialog_open, width="stretch"):
                    _start_video_dialog(
                        video_dir_dialog_key,
                        video_dir_key,
                        multiple=False,
                        recent_key=TARGET_DOMAIN_RECENT_KEY,
                    )
            if st.session_state.get(video_dir_key):
                st.caption(f"视频文件夹：{Path(str(st.session_state[video_dir_key])).name}")
            if st.session_state.get(video_file_dialog_key) or st.session_state.get(video_dir_dialog_key):
                st.info("Windows 视频选择窗口已打开，请完成选择或取消。")
            for dialog_key in (video_file_dialog_key, video_dir_dialog_key):
                if error := str(st.session_state.get(f"{dialog_key}_error") or ""):
                    st.warning(error)
            st.caption(f"已选视频文件：{len(st.session_state.get(video_files_key) or [])} 个")
        else:
            for key in (video_files_key, video_text_key, video_dir_key, video_file_dialog_key, video_dir_dialog_key):
                st.session_state.pop(key, None)

    render_role("normal", "正常样本 · 正常记忆来源", NORMAL_SOURCE_TYPE_KEY, NORMAL_PATH_KEY, NORMAL_DIALOG_KEY, NORMAL_VIDEO_FILES_KEY, NORMAL_VIDEO_TEXT_KEY, NORMAL_VIDEO_DIR_KEY, NORMAL_VIDEO_FILE_DIALOG_KEY, NORMAL_VIDEO_DIR_DIALOG_KEY)
    if needs_defect:
        render_role("defect", "缺陷样本 · 轻量校准来源", DEFECT_SOURCE_TYPE_KEY, DEFECT_PATH_KEY, DEFECT_DIALOG_KEY, DEFECT_VIDEO_FILES_KEY, DEFECT_VIDEO_TEXT_KEY, DEFECT_VIDEO_DIR_KEY, DEFECT_VIDEO_FILE_DIALOG_KEY, DEFECT_VIDEO_DIR_DIALOG_KEY)
    if needs_defect and str(st.session_state.get(DEFECT_SOURCE_TYPE_KEY) or "image") in {"video", "mixed"}:
        st.text_area("缺陷视频时间区间（可选）", key=DEFECT_INTERVAL_TEXT_KEY, height=70, placeholder="每行：视频文件名: 0-10, 20-30；也可写帧区间 120f-240f", on_change=_invalidate_domain_inputs)
        st.caption("未提供帧级区间时，系统将通过正常记忆库筛选高置信异常代表帧；不会假设缺陷视频每一帧都有缺陷。")

    relevant_source_keys = (NORMAL_SOURCE_TYPE_KEY, DEFECT_SOURCE_TYPE_KEY) if needs_defect else (NORMAL_SOURCE_TYPE_KEY,)
    has_video_sources = any(str(st.session_state.get(key) or "image") in {"video", "mixed"} for key in relevant_source_keys)
    if has_video_sources:
        st.markdown("#### 视频抽帧设置")
        with st.expander("抽帧、去重和质量过滤", expanded=False):
            current = dict(st.session_state.get(DOMAIN_SAMPLING_KEY) or {})
            sampling_defaults = {
                "sampling_method": "auto", "frame_interval": 15, "time_interval_seconds": 1.0,
                "target_fps": 1.0, "max_frames_per_video": 100, "final_max_samples": 100,
                "seed": 42, "start_time": 0.0, "end_time": 0.0,
                "similarity_threshold": 0.95, "deduplicate": True, "filter_blur": True,
                "filter_brightness": True, "blur_threshold": 40.0,
                "dark_threshold": 20.0, "bright_threshold": 235.0,
            }
            for field, default in sampling_defaults.items():
                st.session_state.setdefault(SAMPLING_WIDGET_KEYS[field], current.get(field, default))
            method = st.selectbox("抽帧方式", list(VIDEO_SAMPLING_LABELS), key=SAMPLING_WIDGET_KEYS["sampling_method"], format_func=lambda value: VIDEO_SAMPLING_LABELS[value], help="控制如何从视频中抽取候选帧。默认自动模式兼顾时间覆盖和样本预算，固定间隔适合帧率稳定的视频。")
            a, b, c = st.columns(3)
            frame_interval = a.number_input("帧间隔", min_value=1, step=1, key=SAMPLING_WIDGET_KEYS["frame_interval"], help="固定帧间隔模式下每隔多少帧取一帧。数值越小样本越密集，但抽帧与去重耗时越高；默认 15 帧。")
            time_interval = b.number_input("时间间隔（秒）", min_value=0.05, step=0.05, key=SAMPLING_WIDGET_KEYS["time_interval_seconds"], help="固定时间间隔模式下两次抽帧之间的秒数。减小会获得更密集的时间覆盖，也会产生更多相似帧。")
            target_fps = c.number_input("目标抽帧 FPS", min_value=0.1, step=0.1, key=SAMPLING_WIDGET_KEYS["target_fps"], help="目标 FPS 模式希望每秒保留的候选帧数。默认 1 FPS，适合变化较慢的工业视频。")
            d, e, f = st.columns(3)
            max_per_video = d.number_input("每个视频最大帧数", min_value=1, step=1, key=SAMPLING_WIDGET_KEYS["max_frames_per_video"], help="单个视频进入质量过滤前最多保留的候选帧数，防止长视频独占样本预算。")
            final_max = e.number_input("最终最大样本数", min_value=1, step=1, key=SAMPLING_WIDGET_KEYS["final_max_samples"], help="完成抽帧、质量过滤和相似帧去重后允许保留的总样本上限。正常样本数可按实际资源预算设置。")
            seed = f.number_input("帧选择随机种子", min_value=0, step=1, key=SAMPLING_WIDGET_KEYS["seed"], help="控制同等候选帧取舍的可复现随机过程。相同视频和参数使用相同种子便于复验。")
            g, h, i = st.columns(3)
            start_time = g.number_input("起始时间（秒）", min_value=0.0, step=0.5, key=SAMPLING_WIDGET_KEYS["start_time"])
            end_time = h.number_input("结束时间（秒，0 表示到末尾）", min_value=0.0, step=0.5, key=SAMPLING_WIDGET_KEYS["end_time"])
            similarity = i.number_input("相似度去重阈值", min_value=0.5, max_value=0.999, step=0.01, format="%.3f", key=SAMPLING_WIDGET_KEYS["similarity_threshold"], help="相邻候选帧相似度达到该阈值时视为重复。阈值越低去重越强，可能丢失细微变化；默认 0.95。")
            j, k, l = st.columns(3)
            deduplicate = j.checkbox("相似帧去重", key=SAMPLING_WIDGET_KEYS["deduplicate"])
            filter_blur = k.checkbox("过滤模糊帧", key=SAMPLING_WIDGET_KEYS["filter_blur"])
            filter_brightness = l.checkbox("过滤过暗/过亮帧", key=SAMPLING_WIDGET_KEYS["filter_brightness"])
            m, n = st.columns(2)
            blur_threshold = m.number_input("模糊度阈值", min_value=0.0, step=5.0, key=SAMPLING_WIDGET_KEYS["blur_threshold"], help="低于该清晰度分数的帧会被过滤。提高阈值会更严格，但可能丢弃轻微失焦仍可用的样本；默认 40。")
            dark_threshold = n.number_input("过暗阈值", min_value=0.0, max_value=255.0, step=1.0, key=SAMPLING_WIDGET_KEYS["dark_threshold"], help="平均亮度低于该值的帧视为过暗。调高会过滤更多暗帧；默认 20（8 位灰度尺度）。")
            bright_threshold = st.number_input("过亮阈值", min_value=0.0, max_value=255.0, step=1.0, key=SAMPLING_WIDGET_KEYS["bright_threshold"], help="平均亮度高于该值的帧视为过亮。调低会过滤更多高曝光帧；默认 235（8 位灰度尺度）。")
            st.session_state[DOMAIN_SAMPLING_KEY] = {
                "sampling_method": method, "frame_interval": int(frame_interval), "time_interval_seconds": float(time_interval),
                "target_fps": float(target_fps), "max_frames_per_video": int(max_per_video), "final_max_samples": int(final_max),
                "seed": int(seed), "start_time": float(start_time), "end_time": float(end_time) if end_time > 0 else None,
                "deduplicate": bool(deduplicate), "similarity_threshold": float(similarity), "filter_blur": bool(filter_blur),
                "blur_threshold": float(blur_threshold), "filter_brightness": bool(filter_brightness),
                "dark_threshold": float(dark_threshold), "bright_threshold": float(bright_threshold),
            }
    else:
        st.session_state.pop(DOMAIN_SAMPLING_KEY, None)
    st.markdown("#### 样本预算")
    budget_state = dict(st.session_state.get(DOMAIN_BUDGET_KEY) or {})
    st.session_state.setdefault(BUDGET_MODE_WIDGET_KEY, budget_state.get("sample_budget_mode", "final_frame_count"))
    st.session_state.setdefault(NORMAL_BUDGET_WIDGET_KEY, int(budget_state.get("normal_target_count", 100)))
    st.session_state.setdefault(DEFECT_BUDGET_WIDGET_KEY, int(budget_state.get("defect_target_count", 30)))
    if needs_defect:
        budget_a, budget_b, budget_c = st.columns(3)
    else:
        budget_a, budget_b = st.columns(2)
    budget_mode = budget_a.selectbox("样本预算方式", ["final_frame_count", "video_file_count", "custom"], format_func=lambda value: {"final_frame_count": "按最终有效帧数", "video_file_count": "按视频文件数", "custom": "自定义预算"}[value], key=BUDGET_MODE_WIDGET_KEY)
    normal_target = budget_b.number_input("正常目标有效帧", min_value=1, step=1, key=NORMAL_BUDGET_WIDGET_KEY)
    defect_target = budget_c.number_input("缺陷目标有效帧", min_value=0, step=1, key=DEFECT_BUDGET_WIDGET_KEY) if needs_defect else 0
    st.session_state[DOMAIN_BUDGET_KEY] = {"sample_budget_mode": budget_mode, "normal_target_count": int(normal_target), "defect_target_count": int(defect_target)}
    st.caption("视频来源按有效代表帧计数；纯图片来源按可读取图片计数。系统不会自动混用正常与缺陷样本。")
    action_col, hint_col = st.columns([1.4, 3.6], vertical_alignment="center")
    if action_col.button("重新扫描所选样本", icon=":material/refresh:", width="stretch"):
        st.session_state[DOMAIN_SCAN_PENDING_KEY] = True
    hint_col.caption("选择目录后会自动扫描；目录内容和修改时间未变化时直接复用缓存。")
    if scan_error := str(st.session_state.get(DOMAIN_SCAN_ERROR_KEY) or ""):
        st.error(scan_error)
    _run_pending_sample_scan()


FOLDER_DIALOG_KEYS = (
    NORMAL_DIALOG_KEY,
    DEFECT_DIALOG_KEY,
    NORMAL_VIDEO_FILE_DIALOG_KEY,
    DEFECT_VIDEO_FILE_DIALOG_KEY,
    NORMAL_VIDEO_DIR_DIALOG_KEY,
    DEFECT_VIDEO_DIR_DIALOG_KEY,
)


def _has_open_sample_dialog() -> bool:
    return any(bool(st.session_state.get(key)) for key in FOLDER_DIALOG_KEYS)


@st.fragment
def _sample_folder_controls() -> None:
    """One local fragment owns every sample widget and native-picker action."""
    _sample_folder_controls_content()


@st.fragment(run_every=0.4)
def _sample_dialog_status() -> None:
    """Poll only the native picker while it is open; never block a click rerun."""

    before = {key: str(st.session_state.get(key) or "") for key in FOLDER_DIALOG_KEYS}
    if before.get(NORMAL_DIALOG_KEY):
        _poll_dialog(NORMAL_DIALOG_KEY, NORMAL_PATH_KEY)
    if before.get(DEFECT_DIALOG_KEY):
        _poll_dialog(DEFECT_DIALOG_KEY, DEFECT_PATH_KEY)
    if before.get(NORMAL_VIDEO_FILE_DIALOG_KEY):
        _poll_video_dialog(NORMAL_VIDEO_FILE_DIALOG_KEY, NORMAL_VIDEO_FILES_KEY, NORMAL_VIDEO_TEXT_KEY)
    if before.get(DEFECT_VIDEO_FILE_DIALOG_KEY):
        _poll_video_dialog(DEFECT_VIDEO_FILE_DIALOG_KEY, DEFECT_VIDEO_FILES_KEY, DEFECT_VIDEO_TEXT_KEY)
    if before.get(NORMAL_VIDEO_DIR_DIALOG_KEY):
        _poll_single_dialog(NORMAL_VIDEO_DIR_DIALOG_KEY, NORMAL_VIDEO_DIR_KEY)
    if before.get(DEFECT_VIDEO_DIR_DIALOG_KEY):
        _poll_single_dialog(DEFECT_VIDEO_DIR_DIALOG_KEY, DEFECT_VIDEO_DIR_KEY)
    after = {key: str(st.session_state.get(key) or "") for key in FOLDER_DIALOG_KEYS}
    active = next((value for value in after.values() if value), "")
    if active:
        status_col, cancel_col = st.columns([4, 1], vertical_alignment="center")
        status_col.info("Windows 选择窗口已打开；完成选择或取消后，本区域会自动更新。")
        if cancel_col.button("取消选择", key="domain_cancel_native_picker", width="stretch"):
            client().cancel_folder_dialog(active)
            for key in FOLDER_DIALOG_KEYS:
                st.session_state.pop(key, None)
            st.rerun(scope="app")
    elif any(before.values()) and before != after:
        # The selected path belongs to the main editor fragment; one app-scope
        # refresh makes that widget and the derived scan state coherent.
        st.rerun(scope="app")


_migrate_legacy_domain_state()
configure_page("domain_adaptation")
runtime = sidebar_runtime_controls("domain_adaptation")
_sample_dialog_status()
render_domain_resource_creation(runtime, _sample_folder_controls, _domain_input_payload)
