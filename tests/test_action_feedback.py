from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import ui_common


class _Target:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def success(self, message: str) -> None:
        self.calls.append(("success", message))

    def warning(self, message: str) -> None:
        self.calls.append(("warning", message))

    def error(self, message: str) -> None:
        self.calls.append(("error", message))

    def info(self, message: str) -> None:
        self.calls.append(("info", message))


def test_action_feedback_is_scoped_and_replaces_the_current_action(monkeypatch) -> None:
    state: dict = {"aoi_current_page_key": "batch_detection"}
    monkeypatch.setattr(ui_common, "st", SimpleNamespace(session_state=state))

    ui_common.set_action_feedback("batch::job-a", "info", "正在提交 5 条反馈...")
    ui_common.set_action_feedback("batch::job-a", "success", "已提交 5 条反馈。")

    target_a = _Target()
    ui_common.render_action_feedback("batch::job-a", target_a)

    assert target_a.calls == [("success", "✅ 已提交 5 条反馈。")]
    assert "_action_feedback::batch::job-a" not in state

    # A transient result is rendered only once, so later reruns/page entries
    # cannot resurrect it from session state.
    target_after_rerun = _Target()
    ui_common.render_action_feedback("batch::job-a", target_after_rerun)
    assert target_after_rerun.calls == []


def test_new_transient_action_replaces_old_message_on_same_page(monkeypatch) -> None:
    state: dict = {"aoi_current_page_key": "runtime_config"}
    monkeypatch.setattr(ui_common, "st", SimpleNamespace(session_state=state))

    ui_common.set_action_feedback("runtime-archive::a", "success", "已归档。")
    ui_common.set_action_feedback("runtime-restore::b", "success", "已恢复。")

    assert "_action_feedback::runtime-archive::a" not in state
    assert "_action_feedback::runtime-restore::b" in state


def test_action_feedback_failure_and_clear(monkeypatch) -> None:
    state: dict = {}
    monkeypatch.setattr(ui_common, "st", SimpleNamespace(session_state=state))
    target = _Target()

    ui_common.update_action_feedback("delete::plan-a", "error", "删除未完成。", target)
    assert target.calls == [("error", "❌ 删除未完成。")]
    assert "_action_feedback::delete::plan-a" in state

    ui_common.clear_action_feedback("delete::plan-a")
    target_after_clear = _Target()
    ui_common.render_action_feedback("delete::plan-a", target_after_clear)
    assert target_after_clear.calls == []


def test_expired_transient_feedback_is_discarded(monkeypatch) -> None:
    state: dict = {"aoi_current_page_key": "feedback_optimization"}
    now = [100.0]
    monkeypatch.setattr(ui_common, "st", SimpleNamespace(session_state=state))
    monkeypatch.setattr(ui_common.time, "time", lambda: now[0])

    ui_common.set_action_feedback("feedback::bulk", "success", "已审核。", ttl_seconds=5.0)
    now[0] = 106.0
    target = _Target()
    ui_common.render_action_feedback("feedback::bulk", target)

    assert target.calls == []
    assert "_action_feedback::feedback::bulk" not in state


def test_transient_feedback_without_target_uses_inline_placeholder(monkeypatch) -> None:
    state: dict = {"aoi_current_page_key": "runtime_config"}
    target = _Target()
    monkeypatch.setattr(ui_common, "st", SimpleNamespace(session_state=state, empty=lambda: target))
    ui_common.set_action_feedback("runtime::archive", "success", "已归档 2 个记忆库。")
    payload = dict(state["_action_feedback::runtime::archive"])

    assert payload["page_scope"] == "runtime_config"
    assert payload["action_scope"] == "runtime::archive"
    assert payload["consumed"] is False
    assert payload["expires_at"] - payload["created_at"] == 5.0

    ui_common.render_action_feedback("runtime::archive")

    assert target.calls == [("success", "✅ 已归档 2 个记忆库。")]
    assert "_action_feedback::runtime::archive" not in state


def test_page_transition_cleanup_removes_only_transient_feedback(monkeypatch) -> None:
    state: dict = {"aoi_current_page_key": "runtime_config"}
    monkeypatch.setattr(ui_common, "st", SimpleNamespace(session_state=state))
    ui_common.set_action_feedback("runtime::archive", "success", "已归档。")
    ui_common.set_action_feedback("runtime::delete", "error", "删除失败。")

    ui_common._clear_transient_action_feedback()

    assert "_action_feedback::runtime::archive" not in state
    assert "_action_feedback::runtime::delete" in state
