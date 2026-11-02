from __future__ import annotations

from backend.api.schemas import HistoryAllDeleteRequest, HistoryDeleteRequest, HistoryTaskItemsDeleteRequest


def test_history_delete_requests_use_preview_acknowledgement_without_text() -> None:
    common = {
        "plan_id": "plan_12345678901234567890",
        "preview_token": "token_12345678901234567890",
        "acknowledged": True,
    }
    selected = HistoryDeleteRequest(**common)
    cleared = HistoryAllDeleteRequest(**common)
    children = HistoryTaskItemsDeleteRequest(item_ids=["result-1"], acknowledged=True)

    assert selected.acknowledged is True
    assert cleared.acknowledged is True
    assert children.acknowledged is True
    assert selected.confirmation_text == ""
    assert cleared.confirmation_text == ""
    assert children.confirmation_text == ""


def test_history_delete_requests_keep_unacknowledged_execution_disabled() -> None:
    payload = HistoryDeleteRequest(
        plan_id="plan_12345678901234567890",
        preview_token="token_12345678901234567890",
        acknowledged=False,
    )
    assert payload.acknowledged is False
