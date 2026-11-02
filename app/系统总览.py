from __future__ import annotations

import streamlit as st


# Emit a real first screen before importing Altair, NumPy, statistics services,
# and the rest of the dashboard graph.  On a Windows file-cache cold start
# those imports can take several seconds even though Streamlit's HTTP server is
# already accepting connections.  Keeping this shell dependency-light avoids
# turning that objective import time into a white page.
st.set_page_config(
    page_title="AdaptiveAOI · 系统总览",
    layout="wide",
    initial_sidebar_state="expanded",
)
startup_shell = st.empty()
with startup_shell.container():
    st.markdown("# AdaptiveAOI")
    st.info("系统页面已打开，正在加载界面组件并连接检测资源…")
    st.caption("菜单与检测资源状态会自动出现，无需刷新浏览器。")

from ui_common import configure_page  # noqa: E402
from home_page import render_home  # noqa: E402


configure_page("overview", page_configured=True)
render_home()
startup_shell.empty()
