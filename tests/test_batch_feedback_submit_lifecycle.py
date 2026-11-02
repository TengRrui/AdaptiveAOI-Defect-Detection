from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import feedback_panel


def test_batch_feedback_fragment_declares_one_submit_button() -> None:
    source = (APP_DIR / "feedback_panel.py").read_text(encoding="utf-8")
    batch_fragment = source.split("def render_batch_feedback_section(", 1)[1].split(
        "def _video_context(", 1
    )[0]

    assert batch_fragment.count('"提交批量反馈"') == 1


def test_complete_batch_feedback_rebuilds_the_app_tree_once(monkeypatch) -> None:
    rerun_scopes: list[str] = []
    monkeypatch.setattr(
        feedback_panel,
        "st",
        SimpleNamespace(rerun=lambda *, scope: rerun_scopes.append(scope)),
    )

    refreshed = feedback_panel._refresh_after_batch_feedback_submission([])

    assert refreshed is True
    assert rerun_scopes == ["app"]


def test_partial_batch_feedback_keeps_failed_items_without_full_rerun(monkeypatch) -> None:
    rerun_scopes: list[str] = []
    monkeypatch.setattr(
        feedback_panel,
        "st",
        SimpleNamespace(rerun=lambda *, scope: rerun_scopes.append(scope)),
    )

    refreshed = feedback_panel._refresh_after_batch_feedback_submission(
        [{"sample_id": "failed-1", "message": "保存失败"}]
    )

    assert refreshed is False
    assert rerun_scopes == []
