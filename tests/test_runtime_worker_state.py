from __future__ import annotations

from pathlib import Path

from backend.app import runtime_state_code, worker_state_snapshot
from backend.api.routes import _runtime_worker_fields
from service.runtime_config_service import runtime_status_snapshot_from_readiness


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_ready_worker_state_consistency() -> None:
    worker = worker_state_snapshot(
        thread_alive=False,
        lifecycle_state="ready",
        required_for_runtime=False,
    )

    assert worker["worker_state"] == "STOPPED"
    assert worker["worker_alive"] is False
    assert worker["worker_required_for_runtime"] is False
    assert worker["worker_kind"] == "model_warmup_initializer"


def test_runtime_ready_semantics() -> None:
    assert runtime_state_code("ready", runtime_status="ready") == "READY"
    assert runtime_state_code("loading", runtime_status="loading") == "LOADING"
    assert runtime_state_code("failed", runtime_status="failed") == "FAILED"
    assert runtime_state_code("deactivating", runtime_status="deactivating") == "DEGRADED"


def test_worker_heartbeat() -> None:
    worker = worker_state_snapshot(
        thread_alive=True,
        lifecycle_state="loading",
        required_for_runtime=False,
        last_heartbeat_at="2026-08-25T14:00:00+00:00",
        now="2026-08-25T14:00:01+00:00",
    )

    assert worker["worker_state"] == "STARTING"
    assert worker["worker_heartbeat_age_ms"] == 1000
    assert worker["worker_heartbeat_ttl_ms"] == 30000
    assert worker["worker_heartbeat_relevant"] is False


def test_worker_failed_state() -> None:
    worker = worker_state_snapshot(
        thread_alive=False,
        lifecycle_state="failed",
        required_for_runtime=True,
        error="worker crashed",
    )

    assert worker["worker_state"] == "FAILED"
    assert worker["worker_alive"] is False
    assert worker["worker_required_for_runtime"] is True
    assert worker["worker_error"] == "worker crashed"


def test_readiness_snapshot_exposes_runtime_and_worker_states_separately() -> None:
    snapshot = runtime_status_snapshot_from_readiness(
        {
            "success": True,
            "status": "ready",
            "runtime_state": "READY",
            "runtime_ready": True,
            "worker_state": "STOPPED",
            "worker_alive": False,
            "worker_required_for_runtime": False,
            "worker_kind": "model_warmup_initializer",
            "runtime_context": {"runtime_revision": 132, "status": "ready"},
        }
    )

    assert snapshot.ready is True
    assert snapshot.runtime_state == "READY"
    assert snapshot.worker_state == "STOPPED"
    assert snapshot.worker_alive is False
    assert snapshot.worker_required_for_runtime is False


def test_merge_runtime_status_keeps_worker_semantics_in_ui_snapshot() -> None:
    from service.runtime_config_service import RuntimeStatusSnapshot, merge_runtime_status

    merged = merge_runtime_status(
        {},
        RuntimeStatusSnapshot(
            runtime_revision=132,
            status="ready",
            runtime_state="READY",
            worker_state="STOPPED",
            worker_alive=False,
            worker_required_for_runtime=False,
            worker_kind="model_warmup_initializer",
        ),
    )

    assert merged["runtime_ready"] is True
    assert merged["runtime_state"] == "READY"
    assert merged["worker_state"] == "STOPPED"
    assert merged["worker_alive"] is False
    assert merged["worker_required_for_runtime"] is False


def test_health_api_projection_keeps_ready_and_worker_fields_explicit() -> None:
    projected = _runtime_worker_fields(
        {
            "ready": True,
            "runtime_ready": True,
            "runtime_state": "READY",
            "worker_state": "STOPPED",
            "worker_alive": False,
            "worker_required_for_runtime": False,
            "worker_heartbeat_relevant": False,
            "worker_heartbeat_ttl_ms": 30000,
            "status_snapshot_at": "2026-08-25T14:00:00.000+00:00",
        }
    )

    assert projected["runtime_ready"] is True
    assert projected["runtime_state"] == "READY"
    assert projected["worker_state"] == "STOPPED"
    assert projected["worker_alive"] is False
    assert projected["worker_required_for_runtime"] is False
    assert projected["worker_heartbeat_relevant"] is False


def test_system_status_worker_display() -> None:
    status_page = (ROOT / "app/pages/5_系统状态.py").read_text(encoding="utf-8")
    overview = (ROOT / "app/dashboard_components.py").read_text(encoding="utf-8")

    assert "运行时与后台 Worker" in status_page
    assert "worker_required_for_runtime" in status_page
    assert "未常驻（初始化已完成）" in status_page
    assert '("后台Worker", snapshot["worker_status"])' in overview
    assert "未常驻（不影响检测）" in overview
