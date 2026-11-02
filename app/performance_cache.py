from __future__ import annotations

from typing import Any

import streamlit as st

from bridge.api_client import client as _bridge_client
from service.runtime_config_service import (
    RuntimeStatusSnapshot,
    load_runtime_configuration,
    load_runtime_status_snapshot,
    merge_runtime_status,
)


@st.cache_resource(show_spinner=False, max_entries=4)
def api_client(base_url: str | None = None):
    """One Streamlit-owned HTTP session shared by every page fragment."""
    return _bridge_client(base_url)


@st.cache_data(ttl=30, show_spinner=False)
def cached_runtime_configuration(runtime_revision: int) -> dict[str, Any]:
    """Return the applied configuration for one backend Runtime revision."""
    del runtime_revision
    return load_runtime_configuration()


@st.cache_data(ttl=1.5, show_spinner=False)
def cached_runtime_status_snapshot() -> RuntimeStatusSnapshot:
    """Share one lock-free readiness read across page and local fragments."""

    return load_runtime_status_snapshot()


@st.cache_data(ttl=1.0, show_spinner=False)
def cached_runtime_context() -> dict[str, Any]:
    """Share the compact runtime context between page reruns.

    The full ``/api/config`` payload is intentionally reserved for Runtime
    Config management.  Feedback and lightweight status surfaces only need
    this small, revision-bearing snapshot, and a one-second cache prevents a
    checkbox or tab rerun from issuing the same HTTP request again.
    """

    response = api_client().runtime_context()
    return dict(response or {})


@st.cache_data(ttl=1.0, show_spinner=False)
def cached_runtime_liveness() -> dict[str, Any]:
    """Deduplicate liveness reads shared by the Feedback wait fragment."""

    response = api_client().liveness()
    return dict(response or {})


def runtime_ui_configuration() -> tuple[dict[str, Any], RuntimeStatusSnapshot]:
    """Compose one lightweight Runtime UI value from the live snapshot.

    Detection pages do not need the resource-management payload returned by
    ``/api/config``.  The readiness snapshot already owns their immutable
    binding and control prerequisites, so loading-to-ready recovery never
    queues behind config/resource enumeration.
    """

    snapshot = cached_runtime_status_snapshot()
    # Readiness already carries the complete immutable inference binding and
    # the small runtime fields needed by detection controls.  Never put the
    # heavyweight /api/config endpoint on a loading-to-ready fragment path.
    # Management pages keep their dedicated cached resource/config queries.
    runtime = dict(st.session_state.get("_runtime_ui_last_configuration") or {})
    if (
        str(runtime.get("config_profile_id") or "") != snapshot.config_id
        or int(runtime.get("runtime_revision") or 0) != snapshot.runtime_revision
    ):
        runtime = {}
    runtime.setdefault("runtime_context", {})
    if snapshot.ready:
        st.session_state["_runtime_ui_last_configuration"] = merge_runtime_status(runtime, snapshot)
    return merge_runtime_status(runtime, snapshot), snapshot


@st.cache_data(ttl=30, show_spinner=False)
def cached_runtime_resources(
    resource_revision: int,
    resource_type: str,
    include_archived: bool,
) -> dict[str, Any]:
    del resource_revision
    return api_client().runtime_resources(
        resource_type=resource_type or None,
        include_archived=include_archived,
    )


@st.cache_data(ttl=30, show_spinner=False)
def cached_runtime_profiles(resource_revision: int) -> dict[str, Any]:
    del resource_revision
    return api_client().runtime_profiles()


@st.cache_data(ttl=60, show_spinner=False)
def cached_models(resource_revision: int) -> dict[str, Any]:
    """Model catalog is static between resource catalog revisions."""
    del resource_revision
    return api_client().models()


@st.cache_data(ttl=20, show_spinner=False)
def cached_domain_feedback_candidates(
    feedback_revision: int,
    resource_revision: int,
) -> dict[str, Any]:
    del feedback_revision, resource_revision
    return api_client().domain_feedback_candidates()


@st.cache_data(ttl=30, show_spinner=False)
def cached_memory_bank_configs(
    memory_bank_id: str,
    resource_revision: int,
    include_archived: bool = False,
) -> dict[str, Any]:
    del resource_revision
    return api_client().memory_bank_configs(memory_bank_id, include_archived=include_archived)


@st.cache_data(ttl=30, show_spinner=False)
def cached_config_profile(
    profile_id: str,
    version: int,
    resource_revision: int,
) -> dict[str, Any]:
    del resource_revision
    return api_client().config_profile(profile_id, version)


@st.cache_data(ttl=20, show_spinner=False)
def cached_feedback_resources(
    feedback_revision: int,
    resource_revision: int,
) -> dict[str, Any]:
    del feedback_revision, resource_revision
    return api_client().feedback_resources()


@st.cache_data(ttl=20, show_spinner=False)
def cached_feedback_statistics(
    memory_bank_id: str,
    feedback_revision: int,
) -> dict[str, Any]:
    del feedback_revision
    return api_client().feedback_statistics(memory_bank_id or None)


@st.cache_data(ttl=20, show_spinner=False)
def cached_feedback_candidates(
    feedback_revision: int,
    status: str,
    mode: str,
    memory_bank_id: str,
    source_profile_id: str,
    source_target_domain_id: str,
    limit: int,
    page: int = 1,
    page_size: int | None = None,
    search: str = "",
    label: str = "",
    issue: str = "",
    mask: str = "",
    detection_since: str = "",
    feedback_since: str = "",
    min_score: float | None = None,
    dataset: str = "",
    category: str = "",
    method: str = "",
    usage: str = "",
) -> dict[str, Any]:
    del feedback_revision
    return api_client().feedback_candidates(
        status=status or None,
        mode=mode or None,
        memory_bank_id=memory_bank_id or None,
        source_profile_id=source_profile_id or None,
        source_target_domain_id=source_target_domain_id or None,
        limit=limit if page_size is None else None,
        page=max(1, int(page)),
        page_size=max(1, int(page_size or limit)),
        search=search or None,
        label=label or None,
        issue=issue or None,
        mask=mask or None,
        detection_since=detection_since or None,
        feedback_since=feedback_since or None,
        min_score=min_score,
        dataset=dataset or None,
        category=category or None,
        method=method or None,
        usage=usage or None,
    )


def clear_resource_caches() -> None:
    for cached in (
        cached_runtime_configuration,
        cached_runtime_status_snapshot,
        cached_runtime_context,
        cached_runtime_liveness,
        cached_runtime_resources,
        cached_runtime_profiles,
        cached_models,
        cached_memory_bank_configs,
        cached_config_profile,
        cached_feedback_resources,
        cached_domain_feedback_candidates,
    ):
        cached.clear()


def clear_feedback_caches() -> None:
    for cached in (
        cached_feedback_resources,
        cached_feedback_statistics,
        cached_feedback_candidates,
        cached_domain_feedback_candidates,
    ):
        cached.clear()


__all__ = [
    "api_client",
    "cached_config_profile",
    "cached_feedback_candidates",
    "cached_feedback_resources",
    "cached_feedback_statistics",
    "cached_memory_bank_configs",
    "cached_models",
    "cached_domain_feedback_candidates",
    "cached_runtime_configuration",
    "cached_runtime_context",
    "cached_runtime_liveness",
    "cached_runtime_status_snapshot",
    "cached_runtime_profiles",
    "cached_runtime_resources",
    "clear_feedback_caches",
    "clear_resource_caches",
    "runtime_ui_configuration",
]
