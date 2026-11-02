"""Single source of truth for AdaptiveAOI user-facing page metadata."""
from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable

import streamlit as st


APP_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PageMeta:
    key: str
    title_zh: str
    group: str
    source_path: str
    icon: str
    description: str
    order: int
    # A page may keep a descriptive browser/title name while using a shorter
    # label in the compact sidebar.  Keeping this separate avoids changing
    # route keys or duplicating navigation definitions.
    nav_label: str | None = None

    @property
    def browser_title(self) -> str:
        return f"AdaptiveAOI - {self.title_zh}"

    @property
    def page_path(self) -> Path:
        return APP_ROOT / self.source_path


GROUP_ORDER = ("总览", "检测中心", "智能优化", "运行与分析")
GROUP_ICONS = {
    "总览": ":material/dashboard:",
    "检测中心": ":material/center_focus_strong:",
    "智能优化": ":material/auto_awesome:",
    "运行与分析": ":material/monitoring:",
}
GROUP_ICON_SVGS = {
    "总览": '<svg viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>',
    "检测中心": '<svg viewBox="0 0 24 24"><path d="M8 4H5a1 1 0 0 0-1 1v3M16 4h3a1 1 0 0 1 1 1v3M8 20H5a1 1 0 0 1-1-1v-3M16 20h3a1 1 0 0 0 1-1v-3"/><circle cx="12" cy="12" r="3"/></svg>',
    "智能优化": '<svg viewBox="0 0 24 24"><path d="m12 3 1.2 3.3L16.5 7.5l-3.3 1.2L12 12l-1.2-3.3-3.3-1.2 3.3-1.2L12 3ZM18.5 13l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8.8-2.2ZM5.5 13l.7 1.8 1.8.7-1.8.7-.7 1.8-.7-1.8-1.8-.7 1.8-.7.7-1.8Z"/></svg>',
    "运行与分析": '<svg viewBox="0 0 24 24"><path d="M4 19V11M10 19V7M16 19v-4M22 19V4"/><path d="m4 8 5-3 6 5 7-7"/></svg>',
}
GROUP_ARROW_SVG = '<svg viewBox="0 0 24 24"><path d="m7 9.5 5 5 5-5"/></svg>'
GROUP_KEYS = {
    "总览": "overview",
    "检测中心": "inspection",
    "智能优化": "optimization",
    "运行与分析": "analysis",
}

PAGES: tuple[PageMeta, ...] = (
    PageMeta("overview", "系统总览", "总览", "系统总览.py", ":material/dashboard:", "查看系统就绪状态与最近检测概况。", 1),
    PageMeta("single", "单图检测", "检测中心", "pages/1_单图检测.py", ":material/image:", "上传单张图像并查看异常证据。", 2),
    PageMeta("batch", "批量检测", "检测中心", "pages/2_批量检测.py", ":material/photo_library:", "对指定目录执行批量图像检测。", 3),
    PageMeta("video", "视频文件检测", "检测中心", "pages/9_视频文件检测.py", ":material/video_file:", "对本地视频文件执行逐帧检测。", 4),
    PageMeta("domain_adaptation", "目标域适配", "智能优化", "pages/4_目标域适配.py", ":material/hub:", "准备目标域样本，构建记忆库并完成校准与验证。", 5),
    PageMeta("feedback_optimization", "反馈优化中心", "智能优化", "pages/11_反馈优化中心.py", ":material/rule:", "审核反馈样本并创建可验证、可回滚的候选优化版本。", 6),
    PageMeta("status", "系统状态", "运行与分析", "pages/5_系统状态.py", ":material/monitor_heart:", "查看接口、模型与运行状态。", 9),
    PageMeta("runtime_config", "运行配置", "运行与分析", "pages/10_运行配置.py", ":material/settings:", "统一管理检测任务使用的冻结运行配置。", 10),
    PageMeta("statistics", "统计分析", "运行与分析", "pages/7_统计分析.py", ":material/bar_chart:", "查看真实检测历史统计。", 11),
)

PAGE_BY_KEY = {page.key: page for page in PAGES}
PAGE_BY_TITLE = {page.title_zh: page for page in PAGES}


def get_page(key: str) -> PageMeta:
    return PAGE_BY_KEY.get(key, PageMeta("unknown", "页面未找到", "", "", ":material/help:", "", 999))


def pages_in_group(group: str) -> Iterable[PageMeta]:
    return (page for page in PAGES if page.group == group)


def render_sidebar_navigation(current_page_key: str) -> None:
    """Render stable, user-controlled navigation with Streamlit's internal routes."""
    current_page = get_page(current_page_key)
    navigation = st.sidebar.container(key="aoi_nav_scroll", border=False)
    with navigation:
        current_group_key = GROUP_KEYS.get(current_page.group, "")
        st.markdown(
            f'<span class="aoi-current-route-marker" data-aoi-current-page="{current_page.key}" '
            f'data-aoi-current-group="{current_group_key}" aria-hidden="true"></span>',
            unsafe_allow_html=True,
        )
        for group in GROUP_ORDER:
            group_key = GROUP_KEYS[group]
            with st.container(key=f"aoi_nav_group_{group_key}", border=True):
                st.markdown('<span class="aoi-nav-group-open" aria-hidden="true"></span>', unsafe_allow_html=True)
                st.markdown(
                    f'<button type="button" class="aoi-nav-group-toggle" '
                    f'data-aoi-nav-group="{group_key}" aria-expanded="true">'
                    f'<span class="aoi-nav-group-icon" aria-hidden="true">{GROUP_ICON_SVGS[group]}</span>'
                    f'<span class="aoi-nav-group-label">{escape(group)}</span>'
                    f'<span class="aoi-nav-group-arrow" aria-hidden="true">{GROUP_ARROW_SVG}</span>'
                    '</button>',
                    unsafe_allow_html=True,
                )
                with st.container(key=f"aoi_nav_group_body_{group_key}", border=False):
                    for page in pages_in_group(group):
                        with st.container(key=f"aoi_nav_item_{page.key}", border=False):
                            label = page.nav_label or page.title_zh
                            try:
                                st.page_link(
                                    # Resolve multipage entries from the app
                                    # root.  Relative paths are registered
                                    # consistently by the official launcher;
                                    # some Streamlit builds omit absolute
                                    # paths from the page registry.
                                    page.source_path,
                                    label=label,
                                    icon=page.icon,
                                    help=page.description,
                                    width="stretch",
                                )
                            except (KeyError, RuntimeError):
                                # Direct module imports and lightweight test
                                # runners do not always build Streamlit's page
                                # registry.  Keep navigation usable through
                                # the same SPA switch_page mechanism instead
                                # of letting one missing registry entry crash
                                # the whole page.
                                if st.button(label, key=f"aoi_nav_fallback_{page.key}", width="stretch", help=page.description):
                                    st.switch_page(page.source_path)
