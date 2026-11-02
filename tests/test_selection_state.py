from __future__ import annotations

from app.selection_state import (
    checkbox_changed,
    checkbox_key,
    clear_selection,
    remove_selected_ids,
    selected_ids,
    set_page_selection,
    store_persistent_value,
    sync_checkbox_states,
    persistent_value,
)


def test_page_selection_is_id_based_and_cross_page_stable() -> None:
    state: dict = {}
    set_page_selection(state, state_key="selected_ids", scope="records", page_ids=["a", "b"], should_select=True)
    set_page_selection(state, state_key="selected_ids", scope="records", page_ids=["c", "d"], should_select=True)
    assert selected_ids(state, "selected_ids") == {"a", "b", "c", "d"}
    assert state[checkbox_key("records", "a")] is True
    assert state[checkbox_key("records", "d")] is True

    set_page_selection(state, state_key="selected_ids", scope="records", page_ids=["a", "b"], should_select=False)
    assert selected_ids(state, "selected_ids") == {"c", "d"}
    assert state[checkbox_key("records", "a")] is False
    assert state[checkbox_key("records", "c")] is True


def test_checkbox_and_refresh_sync_use_canonical_selected_ids() -> None:
    state: dict = {"selected_ids": ["resource-id"]}
    selected = sync_checkbox_states(
        state,
        state_key="selected_ids",
        scope="resources",
        item_ids=["resource-id", "other-id"],
        valid_ids=["resource-id", "other-id"],
    )
    assert selected == {"resource-id"}
    assert state[checkbox_key("resources", "resource-id")] is True
    assert state[checkbox_key("resources", "other-id")] is False

    state[checkbox_key("resources", "other-id")] = True
    checkbox_changed(state, state_key="selected_ids", scope="resources", item_id="other-id")
    assert selected_ids(state, "selected_ids") == {"resource-id", "other-id"}


def test_deleted_ids_and_clear_remove_widget_state() -> None:
    state: dict = {"selected_ids": ["keep", "delete"]}
    sync_checkbox_states(
        state,
        state_key="selected_ids",
        scope="feedback",
        item_ids=["keep", "delete"],
    )
    remaining = remove_selected_ids(
        state,
        state_key="selected_ids",
        removed_ids=["delete"],
        scopes=["feedback"],
    )
    assert remaining == {"keep"}
    assert checkbox_key("feedback", "delete") not in state

    clear_selection(
        state,
        state_key="selected_ids",
        scopes=["feedback"],
        visible_ids=["keep"],
    )
    assert selected_ids(state, "selected_ids") == set()
    assert state[checkbox_key("feedback", "keep")] is False


def test_browser_sessions_are_isolated_while_each_session_remains_stable() -> None:
    first_session: dict = {}
    set_page_selection(
        first_session,
        state_key="selected_ids",
        scope="records",
        page_ids=["stable-id"],
        should_select=True,
        persistence_key="records:batch",
    )
    second_session: dict = {}
    restored = selected_ids(
        second_session,
        "selected_ids",
        persistence_key="records:batch",
    )
    assert restored == set()
    assert persistent_value(second_session, "resource", persistence_key="feedback:resource") is None
    store_persistent_value(first_session, "resource", "bank-a", persistence_key="feedback:resource")
    third_session: dict = {}
    assert persistent_value(third_session, "resource", persistence_key="feedback:resource") is None
    assert persistent_value(first_session, "resource", persistence_key="feedback:resource") == "bank-a"


def test_persistent_value_recovers_after_streamlit_widget_cleanup() -> None:
    state: dict = {}
    store_persistent_value(
        state,
        "feedback_selected_memory_bank_id",
        "bank-a",
        persistence_key="feedback_center:memory_bank",
    )
    # Streamlit can remove a widget-owned key when a page is left.  The
    # session-scoped backing map must survive that cleanup and restore it.
    state.pop("feedback_selected_memory_bank_id")
    assert persistent_value(
        state,
        "feedback_selected_memory_bank_id",
        persistence_key="feedback_center:memory_bank",
    ) == "bank-a"
