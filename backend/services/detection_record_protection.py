"""Canonical protection rules for persisted detection records.

Protection is derived from live dependencies.  Historical identifiers and
resource tombstones remain useful audit evidence, but they must never become
permanent deletion locks.
"""

from __future__ import annotations

import json
from typing import Any, Iterable


_PENDING_REVIEW_STATUSES = frozenset({"collected", "legacy_unresolved", "pending_review"})
_REVIEWING_STATUSES = frozenset({"in_review", "conflict"})


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def candidate_source_ids(candidate: dict[str, Any]) -> set[str]:
    """Return every persisted source identity without treating it as live."""
    context = _json_object(candidate.get("context_json") or candidate.get("context"))
    return {
        str(value).strip()
        for value in (
            candidate.get("memory_bank_id"),
            candidate.get("base_memory_bank_id"),
            candidate.get("base_profile_id"),
            context.get("source_memory_bank_id"),
            context.get("source_profile_id"),
            context.get("source_target_domain_id"),
        )
        if str(value or "").strip()
    }


def source_memory_bank_deleted(
    candidate: dict[str, Any],
    *,
    deleted_source_ids: Iterable[str] = (),
) -> bool:
    """Whether the candidate's source target-domain resource is gone."""
    context = _json_object(candidate.get("context_json") or candidate.get("context"))
    if candidate.get("source_memory_bank_deleted_at") or context.get("source_memory_bank_deleted"):
        return True
    deleted = {str(value).strip() for value in deleted_source_ids if str(value or "").strip()}
    return bool(deleted and candidate_source_ids(candidate).intersection(deleted))


def candidate_protection_codes(
    candidate: dict[str, Any],
    *,
    active_optimization_jobs: int = 0,
    deleted_source_ids: Iterable[str] = (),
    source_deleted_override: bool | None = None,
) -> list[str]:
    """Return the live lock reasons for one feedback candidate.

    Once a source Memory Bank is permanently deleted, its now-inaccessible
    review row is historical evidence only.  A genuinely active optimization
    transaction remains a lock until that transaction reaches a terminal
    state.
    """
    if candidate.get("deleted_at") or bool(candidate.get("audit_only")):
        return ["optimization_active"] if active_optimization_jobs else []
    source_deleted = (
        bool(source_deleted_override)
        if source_deleted_override is not None
        else source_memory_bank_deleted(candidate, deleted_source_ids=deleted_source_ids)
    )
    codes: list[str] = []
    if not source_deleted:
        status = str(candidate.get("status") or "pending_review").strip().lower()
        if status in _PENDING_REVIEW_STATUSES:
            codes.append("feedback_pending")
        elif status in _REVIEWING_STATUSES:
            codes.append("feedback_reviewing")
        elif status == "accepted" and not candidate.get("source_snapshot_sha256"):
            codes.append("feedback_snapshot_pending")
    if active_optimization_jobs:
        codes.append("optimization_active")
    return codes


__all__ = [
    "candidate_protection_codes",
    "candidate_source_ids",
    "source_memory_bank_deleted",
]
