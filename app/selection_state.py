"""Stable ID-based selection state for Streamlit tables and card lists.

The canonical selected-ID collection is the source of truth.  Checkbox widget
values are synchronized from that collection before widgets are constructed;
row numbers and CSS never participate in selection identity.
"""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from typing import Any


PERSISTENT_VALUES_KEY = "_aoi_persistent_session_values"


def _persistent_values(state: MutableMapping[str, Any]) -> dict[str, Any]:
    values = state.get(PERSISTENT_VALUES_KEY)
    if not isinstance(values, dict):
        values = {}
        state[PERSISTENT_VALUES_KEY] = values
    return values


def normalize_ids(values: Iterable[Any] | None) -> set[str]:
    return {str(value).strip() for value in (values or []) if str(value).strip()}


def selected_ids(
    state: MutableMapping[str, Any],
    state_key: str,
    *,
    valid_ids: Iterable[Any] | None = None,
    legacy_keys: Iterable[str] = (),
    persistence_key: str | None = None,
) -> set[str]:
    """Read one canonical selection set and optionally discard stale IDs."""
    persisted = _persistent_values(state) if persistence_key else {}
    if state_key not in state:
        migrated: set[str] = set()
        for legacy_key in legacy_keys:
            migrated.update(normalize_ids(state.pop(legacy_key, [])))
        if not migrated and persistence_key:
            migrated.update(normalize_ids(persisted.get(persistence_key)))
        state[state_key] = sorted(migrated)
    selected = normalize_ids(state.get(state_key))
    if valid_ids is not None:
        selected.intersection_update(normalize_ids(valid_ids))
    state[state_key] = sorted(selected)
    if persistence_key:
        persisted[persistence_key] = sorted(selected)
    return selected


def store_selected_ids(
    state: MutableMapping[str, Any],
    state_key: str,
    values: Iterable[Any],
    *,
    persistence_key: str | None = None,
) -> set[str]:
    selected = normalize_ids(values)
    state[state_key] = sorted(selected)
    if persistence_key:
        _persistent_values(state)[persistence_key] = sorted(selected)
    return selected


def checkbox_key(scope: str, item_id: Any) -> str:
    return f"selection::{scope}::{str(item_id).strip()}"


def sync_checkbox_states(
    state: MutableMapping[str, Any],
    *,
    state_key: str,
    scope: str,
    item_ids: Iterable[Any],
    valid_ids: Iterable[Any] | None = None,
    persistence_key: str | None = None,
) -> set[str]:
    """Set visible checkbox values from the canonical ID set before rendering."""
    selected = selected_ids(
        state,
        state_key,
        valid_ids=valid_ids,
        persistence_key=persistence_key,
    )
    for item_id in normalize_ids(item_ids):
        state[checkbox_key(scope, item_id)] = item_id in selected
    return selected


def checkbox_changed(
    state: MutableMapping[str, Any],
    *,
    state_key: str,
    scope: str,
    item_id: Any,
    persistence_key: str | None = None,
) -> set[str]:
    normalized_id = str(item_id).strip()
    selected = selected_ids(state, state_key, persistence_key=persistence_key)
    if bool(state.get(checkbox_key(scope, normalized_id))):
        selected.add(normalized_id)
    else:
        selected.discard(normalized_id)
    return store_selected_ids(state, state_key, selected, persistence_key=persistence_key)


def set_page_selection(
    state: MutableMapping[str, Any],
    *,
    state_key: str,
    scope: str,
    page_ids: Iterable[Any],
    should_select: bool,
    persistence_key: str | None = None,
) -> set[str]:
    """Select or clear one page while preserving selections from other pages."""
    normalized_page_ids = normalize_ids(page_ids)
    selected = selected_ids(state, state_key, persistence_key=persistence_key)
    if should_select:
        selected.update(normalized_page_ids)
    else:
        selected.difference_update(normalized_page_ids)
    store_selected_ids(state, state_key, selected, persistence_key=persistence_key)
    for item_id in normalized_page_ids:
        state[checkbox_key(scope, item_id)] = item_id in selected
    return selected


def clear_selection(
    state: MutableMapping[str, Any],
    *,
    state_key: str,
    scopes: Iterable[str],
    visible_ids: Iterable[Any] = (),
    persistence_key: str | None = None,
) -> None:
    store_selected_ids(state, state_key, set(), persistence_key=persistence_key)
    ids = normalize_ids(visible_ids)
    for scope in scopes:
        for item_id in ids:
            state[checkbox_key(scope, item_id)] = False


def remove_selected_ids(
    state: MutableMapping[str, Any],
    *,
    state_key: str,
    removed_ids: Iterable[Any],
    scopes: Iterable[str] = (),
    persistence_key: str | None = None,
) -> set[str]:
    """Remove successfully deleted/processed rows and discard their widgets."""
    removed = normalize_ids(removed_ids)
    selected = selected_ids(state, state_key, persistence_key=persistence_key)
    selected.difference_update(removed)
    store_selected_ids(state, state_key, selected, persistence_key=persistence_key)
    for scope in scopes:
        for item_id in removed:
            state.pop(checkbox_key(scope, item_id), None)
    return selected


def persistent_value(
    state: MutableMapping[str, Any],
    state_key: str,
    *,
    persistence_key: str | None = None,
    default: Any = None,
) -> Any:
    if state_key not in state:
        persisted = _persistent_values(state) if persistence_key else {}
        state[state_key] = persisted.get(persistence_key, default) if persistence_key else default
    if persistence_key:
        _persistent_values(state)[persistence_key] = state.get(state_key, default)
    return state.get(state_key, default)


def store_persistent_value(
    state: MutableMapping[str, Any],
    state_key: str,
    value: Any,
    *,
    persistence_key: str | None = None,
) -> Any:
    state[state_key] = value
    if persistence_key:
        _persistent_values(state)[persistence_key] = value
    return value
