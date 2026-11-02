from __future__ import annotations

from pathlib import Path

import yaml
from fastapi.testclient import TestClient

import backend.app as app_module
from bridge.config_loader import _cached_software_config, load_software_config


ROOT = Path(__file__).resolve().parents[1]


def test_frontend_api_url_follows_launcher_backend_port(monkeypatch) -> None:
    expected = "http://127.0.0.1:18000"
    monkeypatch.setenv("ADAPTIVEAOI_INTEGRATION_API_URL", expected)
    _cached_software_config.cache_clear()
    try:
        assert load_software_config()["integration_api_url"] == expected
    finally:
        _cached_software_config.cache_clear()


def test_liveness_identifies_the_package_instance(monkeypatch) -> None:
    application = app_module.create_app()
    coordinator = application.state.startup
    monkeypatch.setattr(coordinator, "start", lambda _app: False)
    monkeypatch.setattr(coordinator, "shutdown", lambda: None)

    with TestClient(application) as client:
        response = client.get("/api/health/live")

    payload = response.json()
    assert response.status_code == 200
    assert Path(payload["instance_root"]).resolve() == ROOT.resolve()


def test_visible_speed_profiles_use_the_packaged_backbone() -> None:
    for name in ("speed_mode.yaml", "extreme_speed_mode.yaml"):
        payload = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
        assert payload["backbone"] == "wide_resnet50_2", name


def test_launcher_rejects_other_package_instances_and_propagates_ports() -> None:
    source = (ROOT / "launchers" / "start.ps1").read_text(encoding="utf-8")
    watchdog = (ROOT / "launchers" / "watchdog.ps1").read_text(encoding="utf-8")

    assert "$samePython" not in source
    assert "Test-AdaptiveAOIFrontendProcess" in source
    assert "$env:ADAPTIVEAOI_INTEGRATION_API_URL = $ApiUrl" in source
    assert "instance_root" in source
    assert "Get-ProjectMutexName" in source
    assert "Get-ProjectMutexName" in watchdog
