from __future__ import annotations

import re
import time
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st


from performance_cache import api_client as client
from navigation import get_page, render_sidebar_navigation
from performance_cache import cached_runtime_status_snapshot, clear_resource_caches, runtime_ui_configuration
from styles import apply_global_styles, render_sidebar_brand


DATASETS = ["mvtec", "visa", "dagm", "custom"]
DEVICES = ["auto", "cuda", "cpu"]

DATASET_DISPLAY = {
    "mvtec": "MVTec AD",
    "visa": "VisA",
    "dagm": "DAGM",
    "custom": "自定义数据集",
}
DEVICE_DISPLAY = {
    "auto": "自动选择",
    "cuda": "GPU（CUDA）",
    "cpu": "CPU",
    "mps": "GPU（MPS）",
}


DETECTION_VISUAL_DEFAULT_REVISION = 1
ACTION_FEEDBACK_LEVELS = {"success", "warning", "error", "info"}
ACTION_FEEDBACK_TTL_SECONDS = 5.0
ACTION_FEEDBACK_PREFIX = "_action_feedback::"
ACTION_FEEDBACK_ICONS = {
    "success": "✅",
    "warning": "⚠️",
    "error": "❌",
    "info": "⏳",
}


def initialize_detection_visualization_default(session_key: str) -> None:
    """Migrate one detection-page visualization checkbox to default-off.

    The revision marker resets a stale ``True`` left by the previous UI only
    once.  After that, an operator's explicit choice remains stable for the
    current Streamlit session while every new/cold session starts disabled.
    """
    marker_key = f"_detection_visual_default_revision::{session_key}"
    if int(st.session_state.get(marker_key, 0) or 0) < DETECTION_VISUAL_DEFAULT_REVISION:
        st.session_state[session_key] = False
        st.session_state[marker_key] = DETECTION_VISUAL_DEFAULT_REVISION


def _action_feedback_key(scope: str) -> str:
    return f"{ACTION_FEEDBACK_PREFIX}{scope}"


def _action_feedback_page_scope(scope: str, page_scope: str | None = None) -> str:
    return str(
        page_scope
        or st.session_state.get("aoi_current_page_key")
        or str(scope).partition("::")[0]
        or "global"
    )


def _clear_transient_action_feedback(*, page_scope: str | None = None, except_scope: str | None = None) -> None:
    """Discard short-lived messages without touching actionable warnings/errors."""
    for key in list(st.session_state):
        if not str(key).startswith(ACTION_FEEDBACK_PREFIX):
            continue
        payload = st.session_state.get(key)
        if not isinstance(payload, dict) or bool(payload.get("persistent")):
            continue
        if page_scope is not None and str(payload.get("page_scope") or "") != str(page_scope):
            continue
        if except_scope is not None and str(payload.get("action_scope") or "") == str(except_scope):
            continue
        st.session_state.pop(key, None)


def set_action_feedback(
    scope: str,
    level: str,
    message: str,
    *,
    ttl_seconds: float = ACTION_FEEDBACK_TTL_SECONDS,
    page_scope: str | None = None,
    persistent: bool | None = None,
) -> None:
    """Store one scoped flash message.

    Success and ordinary information are transient by default.  Warnings and
    errors remain until the next relevant action or an explicit clear so an
    operator cannot miss an actionable failure.
    """
    normalized_level = level if level in ACTION_FEEDBACK_LEVELS else "info"
    normalized_scope = str(scope or "global")
    normalized_page = _action_feedback_page_scope(normalized_scope, page_scope)
    should_persist = normalized_level in {"warning", "error"} if persistent is None else bool(persistent)
    now = time.time()
    if not should_persist:
        _clear_transient_action_feedback(page_scope=normalized_page, except_scope=normalized_scope)
    st.session_state[_action_feedback_key(scope)] = {
        "level": normalized_level,
        "message": str(message or "操作已完成。"),
        "created_at": now,
        "expires_at": None if should_persist else now + max(0.1, float(ttl_seconds)),
        "page_scope": normalized_page,
        "action_scope": normalized_scope,
        "consumed": False,
        "persistent": should_persist,
    }


def clear_action_feedback(scope: str) -> None:
    st.session_state.pop(_action_feedback_key(scope), None)


def _render_transient_action_feedback(target: Any | None, level: str, message: str, ttl_seconds: float) -> None:
    """Render a non-blocking inline flash at the current action location.

    Ordinary CRUD feedback must never fall back to Streamlit's fixed
    top-right toast layer: that divorces the result from the button that
    caused it and becomes especially confusing on long pages.  Legacy call
    sites that have not supplied an explicit status placeholder still get an
    inline placeholder at their current render position.
    """
    icon = ACTION_FEEDBACK_ICONS[level]
    if target is None:
        target = st.empty()
    if hasattr(target, "markdown"):
        duration = max(0.1, float(ttl_seconds))
        safe_message = escape(message)
        target.markdown(
            f"""
            <style>
            @keyframes aoi-action-flash-hide {{
              0%, 82% {{ opacity: 1; visibility: visible; max-height: 6rem; padding-top: .55rem; padding-bottom: .55rem; }}
              100% {{ opacity: 0; visibility: hidden; max-height: 0; padding-top: 0; padding-bottom: 0; border-width: 0; }}
            }}
            .aoi-action-flash {{
              padding: .55rem .75rem; border-radius: .55rem; line-height: 1.35;
              max-height: 6rem; overflow: hidden;
              animation: aoi-action-flash-hide {duration:.2f}s ease forwards;
            }}
            .aoi-action-flash-success {{ background: #eaf7ef; color: #176b3a; border: 1px solid #a8d8bb; }}
            .aoi-action-flash-info {{ background: #eef6fb; color: #24556f; border: 1px solid #b8d8e8; }}
            </style>
            <div class="aoi-action-flash aoi-action-flash-{level}" role="status">{icon} {safe_message}</div>
            """,
            unsafe_allow_html=True,
        )
        return
    getattr(target, level)(f"{icon} {message}")


def render_action_feedback(scope: str, target: Any | None = None) -> None:
    """Render a scoped action message and consume transient feedback once."""
    key = _action_feedback_key(scope)
    payload = st.session_state.get(key)
    if not isinstance(payload, dict):
        return
    if bool(payload.get("consumed")):
        st.session_state.pop(key, None)
        return
    level = str(payload.get("level") or "info")
    if level not in ACTION_FEEDBACK_LEVELS:
        level = "info"
    message = str(payload.get("message") or "操作已完成。")
    persistent = bool(payload.get("persistent", level in {"warning", "error"}))
    expires_at = payload.get("expires_at")
    if not persistent and expires_at is not None and time.time() >= float(expires_at):
        st.session_state.pop(key, None)
        return
    output = target if target is not None else st
    if persistent:
        getattr(output, level)(f"{ACTION_FEEDBACK_ICONS[level]} {message}")
        return
    remaining = max(0.1, float(expires_at or time.time()) - time.time())
    _render_transient_action_feedback(target, level, message, remaining)
    payload["consumed"] = True
    st.session_state.pop(key, None)


def update_action_feedback(
    scope: str,
    level: str,
    message: str,
    target: Any | None = None,
    *,
    ttl_seconds: float = ACTION_FEEDBACK_TTL_SECONDS,
    page_scope: str | None = None,
    persistent: bool | None = None,
) -> None:
    """Update session state and replace the visible adjacent status in-place."""
    set_action_feedback(
        scope,
        level,
        message,
        ttl_seconds=ttl_seconds,
        page_scope=page_scope,
        persistent=persistent,
    )
    render_action_feedback(scope, target)


def render_action_button_with_feedback(
    label: str,
    *,
    scope: str,
    key: str,
    disabled: bool = False,
    button_type: str = "secondary",
    icon: str | None = None,
    help_text: str | None = None,
    column_ratio: tuple[float, float] = (0.42, 1.58),
) -> tuple[bool, Any]:
    """Render a stable button/status row that wraps naturally on narrow pages."""
    button_column, status_column = st.columns(list(column_ratio), vertical_alignment="center")
    button_options: dict[str, Any] = {
        "key": key,
        "disabled": disabled,
        "type": button_type,
        "width": "stretch",
    }
    if icon:
        button_options["icon"] = icon
    if help_text:
        button_options["help"] = help_text
    clicked = button_column.button(label, **button_options)
    with status_column:
        status_placeholder = st.empty()
    render_action_feedback(scope, status_placeholder)
    return clicked, status_placeholder


def render_form_submit_button_with_feedback(
    label: str,
    *,
    scope: str,
    disabled: bool = False,
    button_type: str = "secondary",
    icon: str | None = None,
    help_text: str | None = None,
    column_ratio: tuple[float, float] = (0.42, 1.58),
) -> tuple[bool, Any]:
    """Form-submit equivalent of :func:`render_action_button_with_feedback`.

    It must be called inside an ``st.form``.  The status column is independent
    from the submit widget, so running/success/error feedback replaces itself
    in one stable location and wraps naturally on narrower screens.
    """
    button_column, status_column = st.columns(list(column_ratio), vertical_alignment="center")
    button_options: dict[str, Any] = {
        "disabled": disabled,
        "type": button_type,
        "width": "stretch",
    }
    if icon:
        button_options["icon"] = icon
    if help_text:
        button_options["help"] = help_text
    clicked = button_column.form_submit_button(label, **button_options)
    with status_column:
        status_placeholder = st.empty()
    render_action_feedback(scope, status_placeholder)
    return clicked, status_placeholder
CATEGORY_DISPLAY = {
    "bottle": "瓶子",
    "cable": "线缆",
    "capsule": "胶囊",
    "carpet": "地毯",
    "grid": "网格",
    "hazelnut": "榛子",
    "leather": "皮革",
    "metal_nut": "金属螺母",
    "pill": "药片",
    "screw": "螺丝",
    "tile": "瓷砖",
    "toothbrush": "牙刷",
    "transistor": "晶体管",
    "wood": "木材",
    "zipper": "拉链",
}
CONFIG_DISPLAY = {
    "configs/final_accuracy.yaml": "最终精度配置",
    "configs/final_balanced.yaml": "最终平衡配置",
    "configs/final_speed.yaml": "最终速度配置",
}


def _dataset_display(value: str) -> str:
    return DATASET_DISPLAY.get(value, value)


def _config_display(value: str) -> str:
    return CONFIG_DISPLAY.get(value, Path(value).stem.replace("_", " "))


def _category_display(value: str) -> str:
    return f"{CATEGORY_DISPLAY.get(value, value)}（{value}）"


def _device_display(value: str) -> str:
    return DEVICE_DISPLAY.get(value, value)


def configure_page(page_key: str, *, page_configured: bool = False) -> None:
    """Apply page configuration from the central navigation metadata."""
    page = get_page(page_key)
    if not page_configured:
        st.set_page_config(page_title=page.browser_title if page.key != "unknown" else "AdaptiveAOI", layout="wide", initial_sidebar_state="expanded")
    previous_page = str(st.session_state.get("aoi_current_page_key") or "")
    if previous_page and previous_page != page.key:
        _clear_transient_action_feedback()
    st.session_state["aoi_current_page_key"] = page.key
    apply_global_styles()
    # The resource catalog is not part of the detection-page loading shell.
    # During process cold start the catalog endpoint can still be initializing;
    # querying it here used to hold the entire page before even the truthful
    # Runtime loading state was visible.
    if cached_runtime_status_snapshot().ready:
        synchronize_resource_catalog()
    install_sidebar_accessibility_controls()


def synchronize_resource_catalog() -> dict[str, Any]:
    """Invalidate only resource selectors affected by a server-side catalog change."""
    previous = st.session_state.get("aoi_resource_catalog_revision")
    now = time.monotonic()
    last_check = float(st.session_state.get("aoi_resource_catalog_checked_at") or 0.0)
    if previous is not None and now - last_check < 0.75:
        return {"changed": False, "available": True, "revision": int(previous), "throttled": True}
    response = client().runtime_resource_revision(int(previous)) if previous is not None else client().runtime_resource_revision()
    st.session_state["aoi_resource_catalog_checked_at"] = now
    if not response.get("success"):
        return {"changed": False, "available": False}
    revision = int(response.get("revision") or 0)
    st.session_state["aoi_resource_catalog_revision"] = revision
    if previous is None or not response.get("changed"):
        return {"changed": False, "available": True, "revision": revision}

    changes = list(response.get("changes") or [])
    affected_resources = {
        str(resource_id)
        for event in changes
        for resource_id in list((event or {}).get("affected_resource_ids") or [])
        if resource_id
    }
    affected_profiles = {
        str(profile_id)
        for event in changes
        for profile_id in list((event or {}).get("affected_profile_ids") or [])
        if profile_id
    }
    # These are selector/widget keys only. Detection inputs and task results
    # are deliberately left intact when the resource catalog changes.
    selector_keys = {
        "runtime_selected_profile_resource",
        "runtime_target_profile_resource",
        "runtime_bank_resource",
        "runtime_edit_profile",
        "runtime_dependency_profile",
        "domain_config_profile_key",
    }
    for key in selector_keys:
        value = str(st.session_state.get(key) or "")
        value_profile = value.partition(":v")[0]
        if value in affected_resources or value in affected_profiles or value_profile in affected_profiles:
            st.session_state.pop(key, None)
    selection_keys = {
        "runtime_archive_profile_selected",
        "runtime_archive_template_selected",
        "runtime_memory_selected",
        "runtime_configuration_selected",
    }
    for key in selection_keys:
        values = [str(value) for value in st.session_state.get(key, []) if value]
        if any(value in affected_resources for value in values):
            st.session_state[key] = [value for value in values if value not in affected_resources]
    for key in list(st.session_state):
        if key.startswith("runtime_") and "_row_res_" in key and any(key.endswith(resource_id) for resource_id in affected_resources):
            st.session_state.pop(key, None)
    st.session_state["aoi_resource_catalog_changed"] = {
        "revision": revision,
        "actions": [str((event or {}).get("action") or "") for event in changes],
    }
    clear_resource_caches()
    return {"changed": True, "available": True, "revision": revision, "changes": changes}


def install_sidebar_accessibility_controls() -> None:
    """Persist the navigation shell and proxy the grid-aligned sidebar control."""
    st.html(
        """
        <script>
        (() => {
          try {
            const host = window.parent;
            const doc = host.document;
            const previous = host.__adaptiveAoiShellControlsV5;
            previous?.controller?.abort();
            previous?.observer?.disconnect();
            if (previous?.interval) host.clearInterval(previous.interval);
            const controller = new host.AbortController();
            const signal = controller.signal;

            const nativeSidebar = doc.querySelector('[data-testid="stSidebar"]');
            const identity = host.__adaptiveAoiSidebarIdentityV1 || {
              root: nativeSidebar,
              replacements: 0,
              pageTransitions: 0,
              lastPage: '',
            };
            if (identity.root && nativeSidebar && identity.root !== nativeSidebar) {
              identity.replacements += 1;
              identity.root = nativeSidebar;
            } else if (!identity.root && nativeSidebar) {
              identity.root = nativeSidebar;
            }
            host.__adaptiveAoiSidebarIdentityV1 = identity;
            if (nativeSidebar) {
              nativeSidebar.setAttribute('data-aoi-app-sidebar', 'true');
              nativeSidebar.setAttribute('data-aoi-root-replacements', String(identity.replacements));
            }

            const SCROLL_KEY = 'adaptiveAoiNavScrollV1';
            const GROUP_STATE_KEY = 'adaptiveAoiNavGroupsV2';
            let applyFrame = 0;
            let restoredScrollContainer = null;
            let lastPage = identity.lastPage || '';

            const readGroupState = () => {
              try {
                const value = JSON.parse(host.localStorage.getItem(GROUP_STATE_KEY) || '{}');
                return value && typeof value === 'object' ? value : {};
              } catch (_) {
                return {};
              }
            };

            const applyNavigationGroups = () => {
              const state = readGroupState();
              doc.querySelectorAll('[data-aoi-nav-group]').forEach((button) => {
                const key = button.dataset.aoiNavGroup || '';
                const expanded = state[key] !== false;
                const group = button.closest('[class*="st-key-aoi_nav_group_"]');
                const marker = group?.querySelector('.aoi-nav-group-open, .aoi-nav-group-closed');
                if (marker) {
                  marker.classList.toggle('aoi-nav-group-open', expanded);
                  marker.classList.toggle('aoi-nav-group-closed', !expanded);
                }
                if (button.getAttribute('aria-expanded') !== String(expanded)) {
                  button.setAttribute('aria-expanded', String(expanded));
                }
              });
            };

            const sidebarExpanded = () =>
              doc.querySelector('[data-testid="stSidebar"]')?.getAttribute('aria-expanded') !== 'false';

            const updateToggle = () => {
              const expanded = sidebarExpanded();
              doc.querySelectorAll('[data-aoi-sidebar-toggle]').forEach((button) => {
                button.setAttribute('aria-label', expanded ? '收起侧栏' : '展开侧栏');
                button.setAttribute('title', expanded ? '收起侧栏' : '展开侧栏');
                const icon = button.querySelector('.aoi-sidebar-toggle-icon');
                if (icon) icon.textContent = expanded ? '‹‹' : '››';
              });
            };

            const applyStoredState = () => {
              applyFrame = 0;
              const navigation = doc.querySelector('.st-key-aoi_nav_scroll');
              if (navigation && navigation !== restoredScrollContainer) {
                const storedTop = Number(host.localStorage.getItem(SCROLL_KEY) || 0);
                navigation.scrollTop = Number.isFinite(storedTop) ? storedTop : 0;
                restoredScrollContainer = navigation;
              }
              const marker = doc.querySelector('.aoi-current-route-marker');
              const currentPage = marker?.dataset.aoiCurrentPage || '';
              if (lastPage && currentPage && lastPage !== currentPage) {
                identity.pageTransitions += 1;
              }
              if (currentPage) lastPage = currentPage;
              identity.lastPage = lastPage;
              if (nativeSidebar) {
                nativeSidebar.setAttribute('data-aoi-root-replacements', String(identity.replacements));
                nativeSidebar.setAttribute('data-aoi-page-transitions', String(identity.pageTransitions));
              }
              applyNavigationGroups();
              updateToggle();
            };

            const scheduleApply = () => {
              if (applyFrame) return;
              applyFrame = host.requestAnimationFrame(applyStoredState);
            };

            const proxySidebarToggle = (event) => {
              const custom = event.target.closest?.('[data-aoi-sidebar-toggle]');
              if (!custom) return false;
              event.preventDefault();
              event.stopPropagation();
              const nativeControl = sidebarExpanded()
                ? doc.querySelector('[data-testid="stSidebarCollapseButton"] button')
                : doc.querySelector('button[data-testid="stExpandSidebarButton"]');
              nativeControl?.click();
              host.setTimeout(updateToggle, 60);
              return true;
            };

            const handleShortcut = (event) => {
              if (event.ctrlKey && !event.shiftKey && !event.altKey && String(event.key).toLowerCase() === 'b') {
                const control = doc.querySelector('[data-aoi-sidebar-toggle]');
                if (!control) return;
                event.preventDefault();
                event.stopImmediatePropagation();
                control.click();
              }
            };

            doc.addEventListener('click', (event) => {
              if (proxySidebarToggle(event)) return;
              const groupToggle = event.target.closest?.('[data-aoi-nav-group]');
              if (groupToggle) {
                event.preventDefault();
                event.stopImmediatePropagation();
                const key = groupToggle.dataset.aoiNavGroup || '';
                const state = readGroupState();
                state[key] = groupToggle.getAttribute('aria-expanded') !== 'true';
                host.localStorage.setItem(GROUP_STATE_KEY, JSON.stringify(state));
                applyNavigationGroups();
                return;
              }
              const pageLink = event.target.closest?.('[data-testid="stPageLink"] a');
              if (pageLink) {
                const navigation = doc.querySelector('.st-key-aoi_nav_scroll');
                if (navigation) host.localStorage.setItem(SCROLL_KEY, String(navigation.scrollTop));
              }
            }, { capture: true, signal });

            doc.addEventListener('scroll', (event) => {
              if (event.target?.classList?.contains('st-key-aoi_nav_scroll')) {
                host.localStorage.setItem(SCROLL_KEY, String(event.target.scrollTop));
              }
            }, { capture: true, signal });

            doc.addEventListener('keydown', handleShortcut, { capture: true, signal });
            const interval = host.setInterval(scheduleApply, 120);
            host.__adaptiveAoiShellControlsV5 = { controller, interval };
            scheduleApply();
          } catch (_) { /* Native controls remain available via CSS. */ }
        })();
        </script>
        """,
        width="content",
        unsafe_allow_javascript=True,
    )

def sidebar_runtime_controls(key_prefix: str = "runtime", *, include_memory_bank: bool = False) -> dict[str, Any]:
    del key_prefix, include_memory_bank
    current_page_key = str(st.session_state.get("aoi_current_page_key") or "overview")
    render_sidebar_brand()
    render_sidebar_navigation(current_page_key)
    runtime, _snapshot = runtime_ui_configuration()
    return runtime


def detection_resource_state(runtime: dict[str, Any], model_status: dict[str, Any] | None = None) -> dict[str, str | bool]:
    """Return a resource-aware detection state, not merely model-preload state."""
    snapshot = dict(runtime.get("_ui_runtime_snapshot") or {})
    preload_state = str(
        (model_status or {}).get("state")
        or snapshot.get("status")
        or runtime.get("runtime_status")
        or "not_ready"
    )
    if preload_state in {"pending", "loading", "starting"}:
        return {
            "code": "loading",
            "ready": False,
            "message": str((model_status or {}).get("message") or "⏳ 正在准备检测资源"),
        }
    if preload_state == "failed":
        return {
            "code": "failed",
            "ready": False,
            "message": str((model_status or {}).get("message") or "❌ 检测资源加载失败"),
        }
    if preload_state in {"not_ready", "inactive"}:
        return {
            "code": "not_ready",
            "ready": False,
            "message": str((model_status or {}).get("message") or "检测 Runtime 当前未就绪。"),
        }
    if runtime.get("runtime_error"):
        return {"code": "disconnected", "ready": False, "message": "检测服务暂时未连接"}
    if not str(runtime.get("memory_bank_id") or ""):
        return {"code": "missing_memory", "ready": False, "message": "检测服务已启动，当前未加载目标域记忆库。"}
    required = (
        bool(runtime.get("frozen")),
        runtime.get("threshold") is not None,
        bool(runtime.get("config_profile_id")),
        bool(runtime.get("config_profile_version")),
        bool(runtime.get("config_hash")),
        bool(runtime.get("memory_bank_ready")),
    )
    if not all(required):
        return {"code": "incomplete", "ready": False, "message": "当前目标域版本资源不完整，请重新选择或修复。"}
    return {"code": "ready", "ready": True, "message": "检测资源已就绪"}


def render_detection_resource_status(
    runtime: dict[str, Any],
    model_status: dict[str, Any] | None = None,
    *,
    page_key: str = "detection",
) -> dict[str, str | bool]:
    state = detection_resource_state(runtime, model_status)
    if state["code"] == "ready":
        st.success(str(state["message"]))
    elif state["code"] == "loading":
        st.info(str(state["message"]))
        st.caption("检测按钮暂不可用；后台会继续初始化，完成后本页自动恢复，无需刷新浏览器。")
    elif state["code"] == "failed":
        st.error(sanitized_text(state["message"]))
        if st.button(
            "重新加载检测资源",
            icon=":material/refresh:",
            key=f"retry_detection_resource_loading::{page_key}",
        ):
            response = client().retry_readiness()
            if response.get("success") and response.get("accepted"):
                clear_resource_caches()
                st.info(str(response.get("message") or "已重新启动检测资源加载。"))
            else:
                st.warning(str(response.get("message") or "检测资源仍在加载，未创建重复任务。"))
    elif state["code"] == "missing_memory":
        st.warning(str(state["message"]))
        links = st.columns(2)
        with links[0]:
            st.page_link(get_page("domain_adaptation").source_path, label="前往目标域适配", icon=":material/hub:", width="stretch")
        with links[1]:
            st.page_link(get_page("runtime_config").source_path, label="前往运行配置选择目标域版本", icon=":material/settings:", width="stretch")
    elif state["code"] == "disconnected":
        st.error(str(state["message"]))
    else:
        st.error(str(state["message"]))
        st.page_link(get_page("runtime_config").source_path, label="检查目标域版本完整性", icon=":material/build:")
    return state


def render_detection_resource_monitor(page_key: str, *, render_status: bool = True) -> dict[str, Any]:
    """Render one page-local Runtime state from the authoritative backend fact.

    The caller owns the timed fragment.  That fragment contains both this
    status and its detection controls, so a loading-to-ready transition only
    replaces that small region.  It deliberately never requests an app-wide
    rerun: doing so used to remove the scheduled fragment that initiated the
    transition and produced both a white page and ``fragment not found``.
    """

    runtime, snapshot = runtime_ui_configuration()
    snapshot_values = snapshot.as_dict()
    if snapshot.status in {"pending", "starting", "loading"}:
        stage = snapshot.stage or "正在加载目标域资源"
        message = f"⏳ 正在准备检测资源 · 当前阶段：{stage} · {snapshot.progress}%"
    elif snapshot.status == "ready":
        message = "检测资源已就绪"
    elif snapshot.status == "failed":
        message = f"检测资源加载失败：{snapshot.error or snapshot.stage or '请稍后重试。'}"
    else:
        message = snapshot.message or "检测 Runtime 当前未就绪。"
    model_status = {
        "available": snapshot.ready,
        "state": snapshot.status,
        "message": message,
        "detail": snapshot.error,
        "phase": snapshot.phase,
        "progress": snapshot.progress,
    }
    transition_key = f"_runtime_ui_transition::{page_key}"
    transition = dict(st.session_state.get(transition_key) or {})
    if str(transition.get("status") or "") != snapshot.status or int(transition.get("revision") or 0) != snapshot.runtime_revision:
        transition = {
            "status": snapshot.status,
            "revision": snapshot.runtime_revision,
            "observed_at_ms": int(time.time() * 1000),
        }
        st.session_state[transition_key] = transition
    state = (
        render_detection_resource_status(runtime, model_status, page_key=page_key)
        if render_status
        else detection_resource_state(runtime, model_status)
    )
    if render_status:
        st.html(
            '<span style="display:none" '
            f'data-aoi-runtime-page="{escape(page_key)}" '
            f'data-aoi-runtime-status="{escape(snapshot.status)}" '
            f'data-aoi-runtime-revision="{snapshot.runtime_revision}" '
            f'data-aoi-runtime-observed-ms="{int(transition.get("observed_at_ms") or 0)}"></span>'
        )
    return {
        **state,
        "runtime": runtime,
        "snapshot": snapshot_values,
        "observed_at_ms": int(transition.get("observed_at_ms") or 0),
    }


def runtime_header_items(runtime: dict[str, Any], task_context: dict[str, Any] | None = None) -> list[tuple[str, str]]:
    profile = dict(runtime.get("memory_bank_profile") or {})
    profile_name = str(profile.get("friendly_name") or profile.get("memory_bank_name") or "当前目标域版本")
    runtime_status = str(
        dict(runtime.get("_ui_runtime_snapshot") or {}).get("status")
        or runtime.get("runtime_status")
        or ""
    )
    if runtime_status in {"pending", "starting", "loading"}:
        return [("状态", "正在准备检测资源")]
    if not str(runtime.get("memory_bank_id") or ""):
        return [("状态", "当前未加载目标域记忆库")]
    if not task_context:
        return [
            ("当前目标域版本", profile_name),
            ("当前配置", str(runtime.get("config_profile_name") or f"版本 {runtime.get('config_profile_version') or '-'}")),
            ("状态", "已加载"),
        ]
    context = dict(task_context)
    return [
        ("数据集", str(context.get("dataset") or runtime.get("dataset") or "-")),
        ("类别", str(context.get("category") or runtime.get("category") or "-")),
        ("运行设备", str(context.get("device") or runtime.get("device") or "-")),
        ("目标域版本", str(context.get("profile_name") or context.get("memory_bank_name") or profile_name)),
        ("配置", str(context.get("config_profile_name") or runtime.get("config_profile_name") or f"版本 {runtime.get('config_profile_version') or '-'}")),
        ("方法模式", str(context.get("method_name") or context.get("method_mode") or profile.get("method_name") or "正常特征记忆检测")),
        ("推理模式", "只读推理"),
    ]


def render_runtime_summary(runtime: dict[str, Any], task_context: dict[str, Any] | None = None) -> None:
    profile = runtime.get("memory_bank_profile") or {}
    memory_name = str(profile.get("friendly_name") or ("已选择冻结记忆库" if runtime.get("memory_bank_id") else "未选择"))
    with st.container(border=True):
        heading, action = st.columns([5, 1.3], vertical_alignment="center")
        with heading:
            st.markdown("#### 当前运行方案")
            st.caption("以下为后端已确认并由当前检测任务统一使用的冻结方案。")
        with action:
            runtime_page = get_page("runtime_config")
            try:
                st.page_link(
                    runtime_page.source_path,
                    label="修改运行配置",
                    icon=":material/settings:",
                    width="stretch",
                )
            except (KeyError, RuntimeError):
                # Direct page tests and older Streamlit registries may not
                # expose ``url_pathname`` yet.  Keep the same SPA navigation
                # path instead of allowing the detection page to fail.
                if st.button("修改运行配置", icon=":material/settings:", width="stretch", key="runtime_summary_open_config_fallback"):
                    st.switch_page(runtime_page.source_path)
        values = runtime_header_items(runtime, task_context)
        columns = st.columns(len(values))
        for column, (label, value) in zip(columns, values):
            column.caption(label)
            column.markdown(f"**{value if value not in (None, '') else '-'}**")
        if not runtime.get("memory_bank_id"):
            st.caption("加载完整目标域版本后，检测任务才会启用。")


def script_available(rel_script: str) -> bool:
    try:
        from bridge.path_resolver import write_path_status

        status = write_path_status()
        item = status.get("scripts", {}).get(rel_script, {})
        return bool(item.get("exists"))
    except Exception:
        return False


def require_script(rel_script: str) -> bool:
    if script_available(rel_script):
        return True
    st.error(f"算法脚本未找到: {rel_script}")
    return False


def display_optional_image(label: str, path_value: str | Path | None) -> None:
    if not path_value:
        st.info(f"{label} 不存在")
        return
    if str(path_value).startswith(("http://", "https://")):
        st.image(str(path_value), caption=label, width="stretch")
        return
    path = Path(path_value)
    if path.exists():
        st.image(str(path), caption=label, width="stretch")
    else:
        st.info(f"{label} 当前不可用，请重新执行检测或在结果记录中确认任务状态。")


def _load_visual_image(path_value: Any) -> Image.Image | None:
    """Load one visualization input without assuming it is a filesystem path.

    Result records normally carry artifact paths/URLs, while AppTest and
    legacy callers may pass an already decoded PIL image, NumPy array, or
    bytes.  Keeping this normalization in the shared component prevents one
    page from accepting a type that another page rejects.
    """
    if path_value is None:
        return None
    try:
        # Visualization dependencies are only needed after a user opens a
        # result tile. Keep them out of the common page import path so the
        # Streamlit shell and Runtime monitor start with UI/HTTP dependencies.
        from io import BytesIO

        import numpy as np
        from PIL import Image, UnidentifiedImageError

        if isinstance(path_value, Image.Image):
            return path_value.convert("RGB")
        if isinstance(path_value, np.ndarray):
            array = np.asarray(path_value)
            if not array.size:
                return None
            if array.ndim == 2:
                if array.dtype != np.uint8:
                    scale = 255.0 if float(np.nanmax(array)) <= 1.0 else 1.0
                    array = np.clip(array * scale, 0, 255).astype(np.uint8)
                return Image.fromarray(array, mode="L").convert("RGB")
            if array.ndim == 3 and array.shape[2] >= 3:
                array = array[..., :3]
                if array.dtype != np.uint8:
                    scale = 255.0 if float(np.nanmax(array)) <= 1.0 else 1.0
                    array = np.clip(array * scale, 0, 255).astype(np.uint8)
                return Image.fromarray(array, mode="RGB")
            return None
        if isinstance(path_value, (bytes, bytearray, memoryview)):
            with Image.open(BytesIO(bytes(path_value))) as image:
                return image.convert("RGB")
        value = str(path_value)
        if value.startswith(("http://", "https://")):
            payload = client().artifact_bytes(value)
            if not payload.get("success"):
                return None
            with Image.open(BytesIO(payload.get("content") or b"")) as image:
                return image.convert("RGB")
        path = Path(value)
        if not path.exists():
            return None
        with Image.open(path) as image:
            return image.convert("RGB")
    except (OSError, UnidentifiedImageError, ValueError):
        return None


def make_overlay(*args: Any, **kwargs: Any) -> Any:
    """Render legacy overlay inputs without loading OpenCV during page import."""
    from core.postprocess import make_overlay as render_overlay

    return render_overlay(*args, **kwargs)


def display_heatmap_overlay(
    label: str,
    *,
    overlay_path: Any,
    original_path: Any,
    heatmap_path: Any,
    alpha: float = 0.45,
) -> None:
    """Display a red-green overlay, rebuilding it only for legacy records.

    New results use the persisted overlay artifact.  Older results that only
    contain an original image and heatmap are composed in memory so the second
    tile keeps the same semantics without modifying historical files.
    """
    if overlay_path is not None:
        # Persisted URLs/paths should be rendered directly so a rerun does
        # not decode large artifacts in Python.  In-memory values use the
        # same loader as legacy fallback records.
        if isinstance(overlay_path, (str, Path)):
            value = str(overlay_path)
            if value.startswith(("http://", "https://")) or Path(value).exists():
                display_optional_image(label, overlay_path)
                return
        overlay = _load_visual_image(overlay_path)
        if overlay is not None:
            st.image(overlay, caption=label, width="stretch")
            return
    original = _load_visual_image(original_path)
    heatmap = _load_visual_image(heatmap_path)
    if original is not None and heatmap is not None:
        try:
            import numpy as np

            from core.postprocess import make_overlay

            heatmap_u8 = np.asarray(heatmap.convert("L"), dtype=np.uint8)
            st.image(make_overlay(original, heatmap_u8=heatmap_u8, alpha=alpha), caption=label, width="stretch")
            return
        except (TypeError, ValueError):
            pass
    st.info(f"{label} 当前不可用，请重新执行检测或在结果记录中确认任务状态。")


def sanitized_display(value: Any) -> Any:
    """Keep technical routes and local filesystem paths out of user-facing UI."""
    hidden_keys = ("path", "url", "root", "directory", "folder", "artifact", "log")
    if isinstance(value, dict):
        return {str(key): ("已隐藏（内部定位信息）" if any(token in str(key).lower() for token in hidden_keys) else sanitized_display(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitized_display(item) for item in value]
    if isinstance(value, str) and (value.startswith(("http://", "https://")) or ":\\" in value or "/artifacts/" in value):
        return "已隐藏（内部定位信息）"
    return value


def sanitized_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"https?://(?:127\.0\.0\.1|localhost)(?::\d+)?[^\s'\"]*", "[内部服务地址已隐藏]", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<![\w.])(?:127\.0\.0\.1|localhost)(?::\d+)?", "[内部服务地址已隐藏]", text, flags=re.IGNORECASE)
    text = re.sub(r"[A-Za-z]:\\[^\r\n]*", "[本地路径已隐藏]", text)
    return text


def model_preload_status() -> dict[str, Any]:
    _runtime, snapshot = runtime_ui_configuration()
    return {
        "available": snapshot.ready,
        "state": snapshot.status,
        "message": snapshot.message,
        "detail": snapshot.error,
        "phase": snapshot.phase,
        "progress": snapshot.progress,
    }


def label_for_pred(value: Any) -> str:
    try:
        return "异常" if int(value) == 1 else "正常"
    except Exception:
        return str(value)


def format_value(value: Any, digits: int = 4) -> str:
    if value in (None, ""):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)
