from __future__ import annotations

import logging
import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient

import backend.app as app_module


def test_liveness_is_available_without_service_container(monkeypatch) -> None:
    application = app_module.create_app()
    coordinator = application.state.startup
    monkeypatch.setattr(coordinator, "start", lambda _app: False)
    monkeypatch.setattr(coordinator, "shutdown", lambda: None)

    with TestClient(application) as client:
        live = client.get("/api/health/live")
        ready = client.get("/api/health/ready")

    assert live.status_code == 200
    assert live.json()["alive"] is True
    assert application.state.services is None
    assert ready.status_code == 200
    assert ready.json()["ready"] is False
    assert ready.json()["status"] == "pending"
    assert ready.json()["phase"] == "api_ready"
    assert ready.json()["progress"] < 100


def test_startup_coordinator_never_creates_duplicate_builder(monkeypatch) -> None:
    entered = threading.Event()
    release = threading.Event()

    def fake_build_services(*_args, **_kwargs):
        entered.set()
        release.wait(timeout=5)
        raise RuntimeError("controlled startup failure")

    monkeypatch.setattr(app_module, "build_services", fake_build_services)
    coordinator = app_module.StartupCoordinator(SimpleNamespace(), logging.getLogger("cold-start-test"))
    application = SimpleNamespace(state=SimpleNamespace(services=None, model_warmup=None))

    assert coordinator.start(application) is True
    assert entered.wait(timeout=2)
    assert coordinator.start(application) is False
    release.set()
    coordinator._thread.join(timeout=5)

    snapshot = coordinator.status()
    assert snapshot["state"] == "failed"
    assert snapshot["ready"] is False
    assert snapshot["progress"] < 100
    assert snapshot["worker_alive"] is False


def test_build_environment_warmup_starts_without_waiting_for_memory_bank_runtime(monkeypatch) -> None:
    calls: list[str] = []
    warmup = SimpleNamespace(
        start=lambda: calls.append("runtime_start"),
        status=lambda: {"state": "not_ready"},
    )
    services = SimpleNamespace(
        warmup=warmup,
        domain_profiles=SimpleNamespace(
            warm_build_environment_async=lambda: calls.append("domain_warmup") or True,
        ),
        runtime_resources=SimpleNamespace(
            reconcile_deleted_feedback_dependencies=lambda: calls.append("maintenance"),
        ),
    )
    monkeypatch.setattr(app_module, "build_services", lambda *_args, **_kwargs: services)
    coordinator = app_module.StartupCoordinator(SimpleNamespace(), logging.getLogger("domain-warmup-test"))
    application = SimpleNamespace(state=SimpleNamespace(services=None, model_warmup=None))
    coordinator._app = application
    coordinator._started_at = 0.0

    coordinator._run()

    assert calls[:2] == ["runtime_start", "domain_warmup"]
    assert "maintenance" not in calls
    assert application.state.services is services
