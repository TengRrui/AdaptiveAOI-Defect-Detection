from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"


def _app_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in APP.rglob("*.py"))


def test_rerun_budget_and_scopes_are_bounded() -> None:
    text = _app_text()
    calls = re.findall(r"st\.rerun\([^\n]*\)", text)
    # One app-scoped refresh is reserved for a completed batch-feedback
    # submission so Streamlit discards the fragment's previous widget tree.
    assert len(calls) <= 34
    assert sum('scope="fragment"' in call for call in calls) >= 15


def test_long_lived_interaction_regions_use_fragments_and_forms() -> None:
    feedback_panel = (APP / "feedback_panel.py").read_text(encoding="utf-8")
    target_domain = (APP / "domain_resource_creation_page.py").read_text(encoding="utf-8")
    statistics = (APP / "pages" / "7_统计分析.py").read_text(encoding="utf-8")
    video = (APP / "pages" / "9_视频文件检测.py").read_text(encoding="utf-8")

    assert "@st.fragment\ndef render_batch_feedback_section" in feedback_panel
    assert "@st.fragment\ndef render_video_feedback_section" in feedback_panel
    assert "domain_build_parameters_form" in target_domain
    assert "history_management_filter_form" in statistics
    assert "video_detection_settings_form" in video


def test_previous_view_is_kept_visible_during_reconciliation() -> None:
    styles = (APP / "styles.py").read_text(encoding="utf-8")
    assert '[data-stale="true"]' in styles
    assert "opacity: 1 !important" in styles


def test_detection_action_labels_stay_on_one_line() -> None:
    single = (APP / "pages" / "1_单图检测.py").read_text(encoding="utf-8")
    batch = (APP / "pages" / "2_批量检测.py").read_text(encoding="utf-8")
    styles = (APP / "styles.py").read_text(encoding="utf-8")

    for source, label in ((single, "开始检测"), (batch, "批量检测")):
        label_pos = source.index(f'"{label}"')
        call_start = source.rfind("render_action_button_with_feedback(\n", 0, label_pos)
        call = source[call_start : source.index("\n        )", label_pos)]
        assert "column_ratio=(0.72, 1.28)" in call

    assert ".st-key-single_start_detection button" in styles
    assert ".st-key-batch_start_detection button" in styles
    assert re.search(
        r"\.st-key-single_start_detection button,\s*"
        r"\.st-key-batch_start_detection button\s*\{[^}]*white-space:\s*nowrap",
        styles,
        flags=re.DOTALL,
    )


def test_statistics_refresh_button_has_room_for_its_label() -> None:
    source = (APP / "pages" / "7_统计分析.py").read_text(encoding="utf-8")
    controls = source.split("def _render_time_controls", 1)[1].split(
        "def _cached_dashboard", 1
    )[0]

    assert "st.columns([3.6, 2.4], vertical_alignment=\"bottom\")" in controls
    assert "column_ratio=(0.85, 1.15)" in controls


def test_streamlit_owns_the_shared_http_resource() -> None:
    cache = (APP / "performance_cache.py").read_text(encoding="utf-8")
    assert "@st.cache_resource" in cache
    assert "def api_client(" in cache


def test_target_domain_has_single_owner_fragments_and_stable_source_keys() -> None:
    page = (APP / "pages" / "4_目标域适配.py").read_text(encoding="utf-8")
    creation = (APP / "domain_resource_creation_page.py").read_text(encoding="utf-8")
    assert "_sample_folder_controls_polling" not in page
    assert "@st.fragment\ndef render(" not in creation
    assert 'NORMAL_SOURCE_TYPE_KEY = "domain:normal:source_type"' in page
    assert 'DEFECT_SOURCE_TYPE_KEY = "domain:defect:source_type"' in page
    draft_bar = creation.split("def _draft_bar", 1)[1].split("def _help", 1)[0]
    load_action = draft_bar.split('button("加载"', 1)[1].split('button("保存草稿"', 1)[0]
    assert "set_action_feedback(" in load_action
    assert "st.rerun()" in load_action


def test_target_domain_widget_keys_have_no_static_duplicates() -> None:
    widget_methods = {
        "button",
        "checkbox",
        "file_uploader",
        "multiselect",
        "number_input",
        "radio",
        "segmented_control",
        "selectbox",
        "slider",
        "text_area",
        "text_input",
        "toggle",
    }
    resolved_keys: list[str] = []
    for path in (APP / "pages" / "4_目标域适配.py", APP / "domain_resource_creation_page.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        constants = {
            target.id: node.value.value
            for node in tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        }
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            if not isinstance(call.func, ast.Attribute) or call.func.attr not in widget_methods:
                continue
            key_node = next((keyword.value for keyword in call.keywords if keyword.arg == "key"), None)
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                resolved_keys.append(key_node.value)
            elif isinstance(key_node, ast.Name) and key_node.id in constants:
                resolved_keys.append(constants[key_node.id])

    duplicates = sorted(key for key, count in Counter(resolved_keys).items() if count > 1)
    assert duplicates == []


def test_feedback_landing_is_lightweight_and_review_is_atomic() -> None:
    service = (ROOT / "backend" / "services" / "feedback_candidate_service.py").read_text(encoding="utf-8")
    resource_summary = service.split("def resource_summaries", 1)[1].split("def audit", 1)[0]
    feedback_page = (APP / "pages" / "11_反馈优化中心.py").read_text(encoding="utf-8")
    assert "list_available_reviewed_samples" not in resource_summary
    assert 'FEEDBACK_MEMORY_BANK_KEY = "feedback_selected_memory_bank_id"' in feedback_page
    assert 'st.form(f"feedback_review_decision_{candidate_id}"' in feedback_page
    assert '"保存并审核通过"' in feedback_page
    assert "expected_status" in feedback_page


def test_runtime_selection_has_no_default_session_state_conflict() -> None:
    runtime_page = (APP / "pages" / "10_运行配置.py").read_text(encoding="utf-8")
    assert "value=resource_id in selected" not in runtime_page
    assert 'st.session_state["aoi_resource_catalog_revision"] = int(resource_revision)' in runtime_page


def test_overview_widget_reruns_reuse_a_stable_statistics_window() -> None:
    overview = (APP / "home_page.py").read_text(encoding="utf-8")
    assert "OVERVIEW_CACHE_BUCKET_SECONDS = 15" in overview
    assert "def _cacheable_overview_now(" in overview
    assert "current.second % OVERVIEW_CACHE_BUCKET_SECONDS" in overview
    # Explicit refresh and data revisions must still invalidate the snapshot.
    assert "overview_refresh_token" in overview
    assert "statistics_revision" in overview


def test_history_cleanup_terminal_state_stops_remote_polling() -> None:
    statistics_page = (APP / "pages" / "7_统计分析.py").read_text(encoding="utf-8")
    delete_confirmation = (APP / "delete_preview_confirmation.py").read_text(encoding="utf-8")
    assert "history_cleanup_terminal_snapshot_" in statistics_page
    assert 'st.session_state["history_management_cleanup_job_id"] = ""' in statistics_page
    assert '{"queued", "running", "partial"}' not in statistics_page
    assert "no_op_after_other_delete" in statistics_page
    assert "bool(deleted) and (transaction_finished or terminal)" in statistics_page
    assert "删除任务已提交，正在处理记录与附件。" in delete_confirmation
    assert "删除已完成，相关列表和统计正在同步。" not in delete_confirmation
