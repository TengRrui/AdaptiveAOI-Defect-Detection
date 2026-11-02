from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from backend.services.job_service import JobService
from backend.services.runtime_resource_service import RuntimeResourceService
from backend.services.task_state_service import TaskStateService


ROOT = Path(__file__).resolve().parents[1]


def test_known_task_ids_does_not_require_payload_decoding(tmp_path: Path) -> None:
    states = TaskStateService(tmp_path / "task_states.sqlite3")
    states.upsert("task-a", "batch", "completed", {"large": ["value"] * 100})
    states.upsert("task-b", "video", "failed", {"reason": "qa"})

    assert states.known_task_ids() == {"task-a", "task-b"}


def test_job_startup_skips_indexed_terminal_manifest(tmp_path: Path, monkeypatch) -> None:
    runtime_root = tmp_path / "runtime"
    jobs_root = runtime_root / "jobs"
    manifest = jobs_root / "task-a" / "job_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"job_id":"task-a","status":"completed"}', encoding="utf-8")

    states = TaskStateService(runtime_root / "task_states.sqlite3")
    states.upsert("task-a", "batch", "completed", {"job_id": "task-a", "status": "completed"})
    original_read_text = Path.read_text

    def guarded_read_text(path: Path, *args, **kwargs):
        if path == manifest:
            raise AssertionError("indexed terminal manifests must not be decoded during startup")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    settings = SimpleNamespace(
        runtime_root=runtime_root,
        jobs_root=jobs_root,
        ensure_runtime_dirs=lambda: (runtime_root.mkdir(parents=True, exist_ok=True), jobs_root.mkdir(parents=True, exist_ok=True)),
    )

    JobService(settings)


def test_job_startup_skips_history_walk_after_legacy_migration(tmp_path: Path, monkeypatch) -> None:
    runtime_root = tmp_path / "runtime"
    jobs_root = runtime_root / "jobs"
    jobs_root.mkdir(parents=True)
    settings = SimpleNamespace(
        runtime_root=runtime_root,
        jobs_root=jobs_root,
        ensure_runtime_dirs=lambda: (
            runtime_root.mkdir(parents=True, exist_ok=True),
            jobs_root.mkdir(parents=True, exist_ok=True),
        ),
    )

    first = JobService(settings)
    assert first.states.legacy_manifest_reconciliation_complete() is True

    original_glob = Path.glob

    def guarded_glob(path: Path, pattern: str):
        if path == jobs_root and pattern == "*/job_manifest.json":
            raise AssertionError("completed legacy migration must avoid the history walk")
        return original_glob(path, pattern)

    monkeypatch.setattr(Path, "glob", guarded_glob)
    JobService(settings)


def test_runtime_resource_activity_uses_durable_index_without_directory_scan() -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    class FakeStates:
        def __init__(self) -> None:
            self.statuses = None

        def list(self, *, statuses):
            self.statuses = statuses
            return [
                {
                    "task_id": "batch-active",
                    "task_kind": "batch",
                    "status": "running",
                    "runtime_status": "running",
                    "updated_at": now,
                    "payload": {
                        "job_id": "batch-active",
                        "status": "running",
                        "heartbeat_at": now,
                        "owner_process_id": os.getpid(),
                        "memory_bank_id": "memory-a",
                        "config_profile_id": "config-a",
                        "config_profile_version": 2,
                    },
                }
            ]

        def mark_interrupted(self, *_args, **_kwargs):
            raise AssertionError("fresh active tasks must not be interrupted")

    states = FakeStates()
    service = object.__new__(RuntimeResourceService)
    service.jobs = SimpleNamespace(states=states)
    # Accessing this path would prove the slow manifest fallback was used.
    service.settings = SimpleNamespace(jobs_root=None)

    active = service._running_tasks()

    assert states.statuses == {"pending", "running", "paused"}
    assert active == [
        {
            "job_id": "batch-active",
            "kind": "batch",
            "status": "running",
            "memory_bank_id": "memory-a",
            "profile_id": "memory-a",
            "config_profile_id": "config-a",
            "config_profile_version": "2",
            "heartbeat_at": now,
            "owner_process_id": str(os.getpid()),
            "heartbeat_ttl_seconds": "180",
        }
    ]


def test_launcher_checks_listener_before_http_liveness_probe() -> None:
    launcher = (ROOT / "launchers" / "start.ps1").read_text(encoding="utf-8")
    body = launcher.split("function Test-ApiLive", 1)[1].split("function Test-AdaptiveAOIBackendProcess", 1)[0]

    assert body.index("Get-ListeningProcessId -Port $ApiPort") < body.index("Invoke-WebRequest")
    assert "if (-not $listener) { return $false }" in body


def test_lightweight_maintenance_summary_does_not_scan_artifact_roots() -> None:
    service_source = (ROOT / "backend" / "services" / "runtime_resource_service.py").read_text(encoding="utf-8")
    body = service_source.split("def maintenance_summary", 1)[1].split("def rebuild_index", 1)[0]

    assert "self.find_orphans().get(\"count\", 0) if scan else indexed_orphans" in body
