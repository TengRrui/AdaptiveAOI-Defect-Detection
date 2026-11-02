from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, MutableMapping
from typing import Any


PENDING_KEY = "domain:draft_autosave:pending"
FAILURE_KEY = "domain:draft_autosave:failure"
SEQUENCE_KEY = "domain:draft_autosave:sequence"
LAST_SAVED_FINGERPRINT_KEY = "domain:draft_autosave:last_saved_fingerprint"


def _fingerprint(values: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(values), ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _next_sequence(state: MutableMapping[str, Any]) -> int:
    sequence = int(state.get(SEQUENCE_KEY) or 0) + 1
    state[SEQUENCE_KEY] = sequence
    return sequence


def queue_autosave(
    state: MutableMapping[str, Any],
    values: Mapping[str, Any],
    *,
    changed_field: str,
) -> dict[str, Any]:
    """Queue one committed widget edit without performing network I/O."""

    snapshot = dict(values)
    fingerprint = _fingerprint(snapshot)
    if fingerprint == str(state.get(LAST_SAVED_FINGERPRINT_KEY) or ""):
        state.pop(PENDING_KEY, None)
        state.pop(FAILURE_KEY, None)
        return {"status": "unchanged", "fingerprint": fingerprint}
    pending = {
        "status": "pending",
        "sequence": _next_sequence(state),
        "fingerprint": fingerprint,
        "changed_field": str(changed_field),
        "values": snapshot,
    }
    state[PENDING_KEY] = pending
    state.pop(FAILURE_KEY, None)
    return dict(pending)


def complete_autosave(
    state: MutableMapping[str, Any],
    attempt: Mapping[str, Any],
    response: Mapping[str, Any],
) -> dict[str, Any]:
    """Commit an autosave result only when it still belongs to the latest edit."""

    current = state.get(PENDING_KEY)
    if not isinstance(current, Mapping) or (
        int(current.get("sequence") or 0) != int(attempt.get("sequence") or 0)
        or str(current.get("fingerprint") or "") != str(attempt.get("fingerprint") or "")
    ):
        return {"status": "stale", "response": dict(response)}
    if bool(response.get("success")):
        state[LAST_SAVED_FINGERPRINT_KEY] = str(attempt.get("fingerprint") or "")
        state.pop(PENDING_KEY, None)
        state.pop(FAILURE_KEY, None)
        return {"status": "saved", "response": dict(response), "attempt": dict(attempt)}
    error = response.get("error")
    if isinstance(error, Mapping):
        message = str(error.get("message") or error.get("detail") or "自动保存失败")
    else:
        message = str(error or "自动保存失败")
    failure = {**dict(attempt), "status": "failed", "message": message, "response": dict(response)}
    state.pop(PENDING_KEY, None)
    state[FAILURE_KEY] = failure
    return dict(failure)


def run_pending_autosave(
    state: MutableMapping[str, Any],
    save: Callable[[dict[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    pending = state.get(PENDING_KEY)
    if not isinstance(pending, Mapping):
        return {"status": "idle"}
    attempt = dict(pending)
    response = save(dict(attempt.get("values") or {}))
    return complete_autosave(state, attempt, response)


def autosave_failure(state: Mapping[str, Any]) -> dict[str, Any] | None:
    failure = state.get(FAILURE_KEY)
    return dict(failure) if isinstance(failure, Mapping) else None


def retry_failed_autosave(state: MutableMapping[str, Any]) -> bool:
    failure = autosave_failure(state)
    if not failure:
        return False
    state[PENDING_KEY] = {
        **failure,
        "status": "pending",
        "sequence": _next_sequence(state),
    }
    return True


def clear_autosave_state(state: MutableMapping[str, Any], *, keep_sequence: bool = True) -> None:
    state.pop(PENDING_KEY, None)
    state.pop(FAILURE_KEY, None)
    state.pop(LAST_SAVED_FINGERPRINT_KEY, None)
    if not keep_sequence:
        state.pop(SEQUENCE_KEY, None)
