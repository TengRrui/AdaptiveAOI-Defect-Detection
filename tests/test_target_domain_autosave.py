from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))


def _autosave_module() -> Any:
    try:
        return importlib.import_module("domain_draft_autosave")
    except ModuleNotFoundError:
        pytest.fail("目标域草稿尚未提供可测试的自动保存状态机", pytrace=False)


def _creation_module() -> Any:
    return importlib.import_module("domain_resource_creation_page")


def _values(*, name: str = "QA目标域", note: str = "首版说明") -> dict[str, str]:
    return {
        "domain_rc_name": name,
        "domain_rc_dataset": "qa_dataset",
        "domain_rc_category": "connector",
        "domain_rc_note": note,
    }


def test_autosave_queues_one_committed_edit_and_saves_it_once() -> None:
    autosave = _autosave_module()
    state: dict[str, Any] = {}
    pending = autosave.queue_autosave(state, _values(), changed_field="domain_rc_name")

    calls: list[dict[str, Any]] = []

    def save(snapshot: dict[str, Any]) -> dict[str, Any]:
        calls.append(snapshot)
        return {"success": True, "draft": {"draft_id": "domain-draft-qa123456"}}

    first = autosave.run_pending_autosave(state, save)
    second = autosave.run_pending_autosave(state, save)

    assert pending["sequence"] == 1
    assert pending["changed_field"] == "domain_rc_name"
    assert pending["values"] == _values()
    assert first["status"] == "saved"
    assert second["status"] == "idle"
    assert calls == [_values()]


def test_autosave_failure_keeps_the_edited_snapshot_and_can_retry() -> None:
    autosave = _autosave_module()
    state: dict[str, Any] = {}
    autosave.queue_autosave(state, _values(note="需要保留的输入"), changed_field="domain_rc_note")

    failed = autosave.run_pending_autosave(
        state,
        lambda _snapshot: {"success": False, "error": {"message": "后端暂不可用"}},
    )

    assert failed["status"] == "failed"
    assert autosave.autosave_failure(state)["values"] == _values(note="需要保留的输入")
    assert autosave.retry_failed_autosave(state) is True
    retried = autosave.run_pending_autosave(
        state,
        lambda snapshot: {"success": True, "draft": {"draft_id": snapshot["domain_rc_name"]}},
    )
    assert retried["status"] == "saved"
    assert autosave.autosave_failure(state) is None


def test_stale_autosave_completion_does_not_clear_a_newer_edit() -> None:
    autosave = _autosave_module()
    state: dict[str, Any] = {}
    first = autosave.queue_autosave(state, _values(name="第一版"), changed_field="domain_rc_name")
    second = autosave.queue_autosave(state, _values(name="最终版"), changed_field="domain_rc_name")

    outcome = autosave.complete_autosave(
        state,
        first,
        {"success": True, "draft": {"draft_id": "domain-draft-qa123456"}},
    )

    assert outcome["status"] == "stale"
    assert state[autosave.PENDING_KEY]["sequence"] == second["sequence"]
    assert state[autosave.PENDING_KEY]["values"]["domain_rc_name"] == "最终版"


def test_saved_content_fingerprint_prevents_duplicate_network_save() -> None:
    autosave = _autosave_module()
    state: dict[str, Any] = {}
    calls = 0

    def save(_snapshot: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"success": True, "draft": {"draft_id": "domain-draft-qa123456"}}

    autosave.queue_autosave(state, _values(), changed_field="domain_rc_name")
    assert autosave.run_pending_autosave(state, save)["status"] == "saved"
    duplicate = autosave.queue_autosave(state, _values(), changed_field="domain_rc_dataset")
    assert duplicate["status"] == "unchanged"
    assert autosave.run_pending_autosave(state, save)["status"] == "idle"
    assert calls == 1


def test_loading_another_draft_clears_the_previous_draft_autosave_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    autosave = _autosave_module()
    creation = _creation_module()
    state: dict[str, Any] = {autosave.LAST_SAVED_FINGERPRINT_KEY: "draft-a-fingerprint"}
    monkeypatch.setattr(creation, "st", SimpleNamespace(session_state=state))

    creation._restore_payload(
        {
            "friendly_name": "草稿B",
            "dataset": "qa_dataset",
            "category": "connector",
            "method_mode": creation.DEFAULT_CREATE_MODE,
        }
    )

    assert autosave.LAST_SAVED_FINGERPRINT_KEY not in state


def test_successful_autosave_pins_the_current_draft_for_browser_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creation = _creation_module()
    state = dict(creation.DEFAULTS)
    query_params: dict[str, str] = {}

    class Client:
        def save_domain_draft(self, _payload: dict[str, Any]) -> dict[str, Any]:
            return {"success": True, "draft": {"draft_id": "domain-draft-refresh123"}}

    monkeypatch.setattr(
        creation,
        "st",
        SimpleNamespace(session_state=state, query_params=query_params),
    )
    monkeypatch.setattr(creation, "client", lambda: Client())

    response = creation._save_draft(lambda: {}, quiet=True)

    assert response["success"] is True
    assert query_params["domain_draft"] == "domain-draft-refresh123"


def test_browser_refresh_restores_the_draft_named_in_query_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creation = _creation_module()
    restore = getattr(creation, "_restore_draft_from_query", None)
    assert callable(restore), "页面尚未提供从持久化草稿查询状态恢复编辑器的行为"
    state: dict[str, Any] = {}
    query_params: dict[str, str] = {"domain_draft": "domain-draft-refresh123"}
    monkeypatch.setattr(
        creation,
        "st",
        SimpleNamespace(session_state=state, query_params=query_params),
    )

    restored = restore(
        {
            "domain-draft-refresh123": {
                "draft_id": "domain-draft-refresh123",
                "payload": {
                    "friendly_name": "刷新后恢复",
                    "dataset": "qa_dataset_final",
                    "category": "connector",
                    "method_mode": creation.DEFAULT_CREATE_MODE,
                    "note": "已持久化备注",
                },
            }
        }
    )

    assert restored is True
    assert state["domain_rc_draft_id"] == "domain-draft-refresh123"
    assert state["domain_rc_name"] == "刷新后恢复"
    assert state["domain_rc_dataset"] == "qa_dataset_final"
    assert state["domain_rc_note"] == "已持久化备注"


def test_page_reentry_without_query_resumes_the_most_recent_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creation = _creation_module()
    state: dict[str, Any] = {}
    query_params: dict[str, str] = {}
    monkeypatch.setattr(
        creation,
        "st",
        SimpleNamespace(session_state=state, query_params=query_params),
    )

    restored = creation._restore_draft_from_query(
        {
            "domain-draft-latest": {
                "draft_id": "domain-draft-latest",
                "updated_at": "2026-08-26T12:00:00+00:00",
                "payload": {
                    "friendly_name": "最近编辑草稿",
                    "method_mode": creation.DEFAULT_CREATE_MODE,
                },
            },
            "domain-draft-older": {
                "draft_id": "domain-draft-older",
                "updated_at": "2026-08-25T12:00:00+00:00",
                "payload": {
                    "friendly_name": "较早草稿",
                    "method_mode": creation.DEFAULT_CREATE_MODE,
                },
            },
        }
    )

    assert restored is True
    assert state["domain_rc_draft_id"] == "domain-draft-latest"
    assert state["domain_rc_name"] == "最近编辑草稿"
    assert query_params["domain_draft"] == "domain-draft-latest"


def test_target_domain_build_payload_never_carries_feedback_candidate_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    creation = _creation_module()
    state = dict(creation.DEFAULTS)
    state.pop("domain_rc_feedback_ids", None)
    monkeypatch.setattr(creation, "st", SimpleNamespace(session_state=state))

    payload = creation._payload(
        lambda: {
            "normal_dir": "E:/QA/normal",
            "feedback_candidate_ids": ["legacy-feedback-id"],
        }
    )

    assert "feedback_candidate_ids" not in payload
    assert payload["normal_dir"] == "E:/QA/normal"


def test_basic_info_editor_uses_commit_callbacks_without_manual_apply_buttons(monkeypatch: pytest.MonkeyPatch) -> None:
    creation = _creation_module()
    calls: list[tuple[str, str, Any]] = []

    class Column:
        def text_input(self, label: str, **kwargs: Any) -> None:
            calls.append(("text_input", label, kwargs.get("on_change")))

    fake_st = SimpleNamespace(
        columns=lambda _count: (Column(), Column(), Column()),
        text_area=lambda label, **kwargs: calls.append(("text_area", label, kwargs.get("on_change"))),
        session_state={},
    )
    monkeypatch.setattr(creation, "st", fake_st)

    creation._render_basic_info_editor()

    assert [(kind, label) for kind, label, _callback in calls] == [
        ("text_input", "目标域名称"),
        ("text_input", "数据集"),
        ("text_input", "类别"),
        ("text_area", "样本备注"),
    ]
    assert all(callable(callback) for _kind, _label, callback in calls)


def test_target_domain_page_has_no_reviewed_feedback_picker_or_manual_apply_entry() -> None:
    source = (APP / "domain_resource_creation_page.py").read_text(encoding="utf-8")

    assert "cached_domain_feedback_candidates" not in source
    assert "domain_rc_feedback_ids" not in source
    assert "从反馈优化中心选择已审核样本" not in source
    assert "应用目标域基本信息" not in source
    assert "应用样本说明与反馈样本" not in source
