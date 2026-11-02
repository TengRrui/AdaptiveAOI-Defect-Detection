from __future__ import annotations

import logging
import sqlite3
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.api import routes
from backend.api.schemas import DomainBuildRequest
from backend.errors import ApiError
from backend.services.domain_profile_service import DomainBuildTask, DomainProfileService
from backend.services.task_state_service import TaskStateService
from backend.services.video_sample_service import VideoSampleService


class _TaskStates:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, object]] = {}

    def upsert(
        self,
        task_id: str,
        task_kind: str,
        status: str,
        payload: dict[str, object],
        **metadata: object,
    ) -> None:
        self.rows[task_id] = {
            "task_id": task_id,
            "task_kind": task_kind,
            "status": status,
            "runtime_status": status,
            "payload": dict(payload),
            **metadata,
        }

    def get(self, task_id: str) -> dict[str, object] | None:
        return self.rows.get(task_id)


def _service(tmp_path: Path) -> DomainProfileService:
    settings = SimpleNamespace(
        memory_banks_root=tmp_path / "memory_banks",
        jobs_root=tmp_path / "jobs",
        project_root=tmp_path,
    )
    service = DomainProfileService(
        settings,
        SimpleNamespace(),
        logging.getLogger("domain-build-submission-test"),
        config_profiles=SimpleNamespace(),
        jobs=SimpleNamespace(states=_TaskStates()),
        models=SimpleNamespace(),
    )
    # Most unit tests isolate the in-process worker transition and do not need
    # to spawn the real ML-import helper.  Warm-up behavior has dedicated tests
    # below that reset this state explicitly.
    service._build_environment_warmup_state = "ready"
    return service


def _payload(key: str = "domain-build:test:0001") -> dict[str, object]:
    return {
        "idempotency_key": key,
        "method_mode": "normal_memory_defect_guided_lite",
        "precision": "fp16",
        "dataset": "qa",
        "category": "part",
    }


def test_worker_publishes_acquisition_and_first_progress_before_cold_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    import_entered = threading.Event()
    release_import = threading.Event()

    def blocked_import():
        import_entered.set()
        release_import.wait(timeout=5)
        raise RuntimeError("controlled cold import failure")

    monkeypatch.setattr("backend.services.domain_profile_service._domain_adaptation_module", blocked_import)
    try:
        submitted_at = time.perf_counter()
        created = service.start_build(_payload(), api_received_at="2026-08-26T10:00:00.000+00:00")
        submit_elapsed = time.perf_counter() - submitted_at
        assert submit_elapsed < 0.25
        assert import_entered.wait(timeout=2)

        status = service.task_status(str(created["job_id"]))
        assert status["status"] == "running"
        assert status["lifecycle_phase"] == "preparing_environment"
        assert status["stage"] == "准备构建环境"
        assert status["worker_acquired_at"]
        assert status["first_progress_at"]
        assert status["build_started_at"] is None
        assert status["progress"] == 0.0
        assert status["api_received_at"] == "2026-08-26T10:00:00.000+00:00"
        assert status["task_created_at"]
        assert status["queued_at"]
    finally:
        release_import.set()
        service.shutdown()


def test_repeated_submission_reuses_one_task_and_one_worker_queue_item(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    entered = threading.Event()
    release = threading.Event()

    def blocked_import():
        entered.set()
        release.wait(timeout=5)
        raise RuntimeError("controlled stop")

    monkeypatch.setattr("backend.services.domain_profile_service._domain_adaptation_module", blocked_import)
    try:
        first = service.start_build(_payload("domain-build:idempotent:0001"))
        assert entered.wait(timeout=2)
        second = service.start_build(_payload("domain-build:idempotent:0001"))
        assert first["job_id"] == second["job_id"]
        assert len(service._tasks) == 1
    finally:
        release.set()
        service.shutdown()


def test_preparation_failure_is_terminal_and_names_the_failed_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    monkeypatch.setattr(
        "backend.services.domain_profile_service._domain_adaptation_module",
        lambda: (_ for _ in ()).throw(RuntimeError("qa import unavailable")),
    )
    try:
        created = service.start_build(_payload("domain-build:prep-failure:0001"))
        deadline = time.monotonic() + 3
        status = created
        while status["status"] not in {"failed", "completed", "cancelled"} and time.monotonic() < deadline:
            time.sleep(0.01)
            status = service.task_status(str(created["job_id"]))

        assert status["status"] == "failed"
        assert status["lifecycle_phase"] == "failed"
        assert status["stage"] == "准备构建环境失败"
        assert status["failed_stage"] == "preparing_environment"
        assert status["error"]["code"] == "DOMAIN_BUILD_PREPARATION_FAILED"
        assert "qa import unavailable" in status["error"]["message"]
        assert status["finished_at"]
    finally:
        service.shutdown()


def test_submission_timestamps_are_persisted_in_event_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    release = threading.Event()

    def blocked_import():
        release.wait(timeout=5)
        raise RuntimeError("controlled stop")

    monkeypatch.setattr("backend.services.domain_profile_service._domain_adaptation_module", blocked_import)
    try:
        created = service.start_build(
            _payload("domain-build:timestamps:0001"),
            api_received_at="2026-08-26T10:00:00.000+00:00",
        )
        deadline = time.monotonic() + 2
        status = created
        while not status.get("worker_acquired_at") and time.monotonic() < deadline:
            time.sleep(0.01)
            status = service.task_status(str(created["job_id"]))

        ordered = [
            status["api_received_at"],
            status["task_created_at"],
            status["queued_at"],
            status["worker_acquired_at"],
            status["first_progress_at"],
        ]
        parsed = [datetime.fromisoformat(str(value)) for value in ordered]
        assert parsed == sorted(parsed)
        durable = service.jobs.states.get(str(created["job_id"]))
        assert durable is not None
        durable_payload = durable["payload"]
        assert durable_payload["worker_acquired_at"] == status["worker_acquired_at"]
        assert durable_payload["first_progress_at"] == status["first_progress_at"]
    finally:
        release.set()
        service.shutdown()


def test_dataset_manifest_is_materialized_once_per_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    calls: list[str] = []

    def materialize(_task, _payload, role, _target_count, intervals=None):
        calls.append(role)
        return ({"role": role}, [Path(f"{role}.png")], [{"role": role}], None)

    monkeypatch.setattr(service, "_materialize_source_samples", materialize)
    task = DomainBuildTask(job_id="b" * 32, payload={"normal_target_count": 1, "defect_target_count": 1})

    first = service._materialize_task_dataset(task, defect_optional=False)
    second = service._materialize_task_dataset(task, defect_optional=False)

    assert first is second
    assert calls == ["normal", "defect"]
    service.shutdown()


def test_partial_video_materialization_is_cleaned_when_second_role_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    normal_video_result = {"task_dir": str(tmp_path / "normal-video-temp")}
    cleaned: list[dict[str, object]] = []

    def materialize(_task, _payload, role, _target_count, intervals=None):
        if role == "normal":
            return ({"role": role}, [tmp_path / "normal.png"], [{"role": role}], normal_video_result)
        raise ApiError("QA_DEFECT_FAILURE", "缺陷侧受控失败", 422)

    monkeypatch.setattr(service, "_materialize_source_samples", materialize)
    monkeypatch.setattr(service.video_samples, "cleanup", lambda result: cleaned.append(result))
    task = DomainBuildTask(job_id="c" * 32, payload={"normal_target_count": 1, "defect_target_count": 1})

    with pytest.raises(ApiError, match="缺陷侧受控失败"):
        service._materialize_task_dataset(task, defect_optional=False)

    assert cleaned == [normal_video_result]
    service.shutdown()


def test_video_extraction_cleans_task_directory_when_extraction_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    samples = VideoSampleService(tmp_path / "video-samples")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"qa")
    monkeypatch.setattr(samples, "_files_from_inputs", lambda *_args: [source])
    monkeypatch.setattr(
        samples,
        "probe_video",
        lambda _path: {"valid": True, "name": "source.mp4", "duration_seconds": 1.0},
    )

    def fail_after_partial_write(_video, _role, _config, work, *_args):
        (work / "partial.png").write_bytes(b"partial")
        raise RuntimeError("controlled decode failure")

    monkeypatch.setattr(samples, "_extract_one", fail_after_partial_write)

    with pytest.raises(RuntimeError, match="controlled decode failure"):
        samples.extract("normal", [source], [], {}, target_count=1, task_id="qa-video-failure")

    assert not (tmp_path / "video-samples" / "qa-video-failure").exists()


def test_idempotency_survives_service_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_states = TaskStateService(tmp_path / "task_states.sqlite3")

    def make_service() -> DomainProfileService:
        service = _service(tmp_path / f"service-{time.time_ns()}")
        service.jobs = SimpleNamespace(states=task_states)
        return service

    monkeypatch.setattr(
        "backend.services.domain_profile_service._domain_adaptation_module",
        lambda: (_ for _ in ()).throw(RuntimeError("controlled terminal task")),
    )
    first_service = make_service()
    first = first_service.start_build(_payload("domain-build:durable:0001"))
    deadline = time.monotonic() + 3
    first_status = first
    while first_status["status"] != "failed" and time.monotonic() < deadline:
        time.sleep(0.01)
        first_status = first_service.task_status(str(first["job_id"]))
    first_service.shutdown()

    second_service = make_service()
    try:
        recovered = second_service.start_build(_payload("domain-build:durable:0001"))
        assert recovered["job_id"] == first["job_id"]
        assert second_service._tasks == {}
    finally:
        second_service.shutdown()


def test_task_state_schema_migrates_durable_idempotency_columns(tmp_path: Path) -> None:
    path = tmp_path / "legacy_task_states.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE task_states (
                task_id TEXT PRIMARY KEY,
                task_kind TEXT NOT NULL,
                status TEXT NOT NULL,
                runtime_status TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                interrupted_at TEXT,
                interruption_reason TEXT,
                parent_task_id TEXT,
                rerun_count INTEGER NOT NULL DEFAULT 0
            );
            """
        )

    states = TaskStateService(path)

    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(task_states)")}
        indexes = {row[1] for row in connection.execute("PRAGMA index_list(task_states)")}
    assert {"idempotency_key", "request_fingerprint"} <= columns
    assert "idx_task_states_kind_idempotency" in indexes


def test_same_idempotency_key_with_different_payload_is_rejected(tmp_path: Path) -> None:
    service = _service(tmp_path)
    release = threading.Event()

    def blocked_prepare(task):
        release.wait(timeout=5)
        raise ApiError("QA_STOP", "受控结束", 422)

    service._prepare_build_environment = blocked_prepare
    try:
        service.start_build(_payload("domain-build:payload-hash:0001"))
        changed = _payload("domain-build:payload-hash:0001")
        changed["category"] = "different-part"
        with pytest.raises(ApiError) as raised:
            service.start_build(changed)
        assert raised.value.code == "DOMAIN_BUILD_IDEMPOTENCY_CONFLICT"
    finally:
        release.set()
        service.shutdown()


def test_concurrent_services_atomically_reuse_one_idempotent_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "shared-task-states.sqlite3"
    first_states = TaskStateService(database_path)
    second_states = TaskStateService(database_path)
    first_service = _service(tmp_path / "first-service")
    second_service = _service(tmp_path / "second-service")
    first_service.jobs = SimpleNamespace(states=first_states)
    second_service.jobs = SimpleNamespace(states=second_states)

    lookup_barrier = threading.Barrier(2)
    first_find = first_states.find_by_idempotency
    second_find = second_states.find_by_idempotency

    def synchronized_find(original):
        def find(task_kind: str, idempotency_key: str):
            result = original(task_kind, idempotency_key)
            lookup_barrier.wait(timeout=3)
            return result

        return find

    monkeypatch.setattr(first_states, "find_by_idempotency", synchronized_find(first_find))
    monkeypatch.setattr(second_states, "find_by_idempotency", synchronized_find(second_find))
    release_worker = threading.Event()
    monkeypatch.setattr(
        "backend.services.domain_profile_service._domain_adaptation_module",
        lambda: release_worker.wait(timeout=5) or (_ for _ in ()).throw(RuntimeError("controlled stop")),
    )

    results: list[dict[str, object]] = []
    errors: list[BaseException] = []

    def submit(service: DomainProfileService) -> None:
        try:
            results.append(service.start_build(_payload("domain-build:concurrent:0001")))
        except BaseException as exc:  # assertions below must expose raw DB failures
            errors.append(exc)

    threads = [
        threading.Thread(target=submit, args=(first_service,)),
        threading.Thread(target=submit, args=(second_service,)),
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        assert errors == []
        assert len(results) == 2
        assert len({str(result["job_id"]) for result in results}) == 1
        assert len(first_states.list(task_kind="domain_adaptation")) == 1
        assert len(first_service._tasks) + len(second_service._tasks) == 1
    finally:
        release_worker.set()
        first_service.shutdown()
        second_service.shutdown()


def test_build_route_does_not_resolve_feedback_samples_before_task_creation() -> None:
    captured: dict[str, object] = {}

    class DomainProfiles:
        @staticmethod
        def start_build(payload: dict[str, object], *, api_received_at: str | None = None) -> dict[str, object]:
            captured["payload"] = payload
            captured["api_received_at"] = api_received_at
            return {"job_id": "a" * 32, "status": "queued"}

    services = SimpleNamespace(
        domain_profiles=DomainProfiles(),
        feedback_candidates=SimpleNamespace(
            domain_build_candidates=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("feedback samples must be resolved by the worker")
            )
        ),
    )
    request = SimpleNamespace(
        client=SimpleNamespace(host="127.0.0.1"),
        app=SimpleNamespace(state=SimpleNamespace(services=services)),
        state=SimpleNamespace(request_id="qa-request"),
    )
    payload = DomainBuildRequest(
        dataset="qa",
        category="part",
        idempotency_key="domain-build:route:0001",
        feedback_candidate_ids=["candidate-1"],
    )

    result = routes.start_domain_adaptation(payload, request)

    assert result["success"] is True
    assert captured["api_received_at"]
    assert captured["payload"]["feedback_candidate_ids"] == ["candidate-1"]
    assert "_feedback_samples" not in captured["payload"]


def test_build_environment_warmup_is_non_blocking(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    service._build_environment_warmup_state = "not_started"
    process_started = threading.Event()
    release = threading.Event()

    class IsolatedWarmupProcess:
        returncode = None

        def wait(self, timeout=None):
            process_started.set()
            if not release.wait(timeout=timeout or 5):
                raise subprocess.TimeoutExpired("qa-warmup", timeout)
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15
            release.set()

    captured: dict[str, object] = {}

    def popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return IsolatedWarmupProcess()

    monkeypatch.setattr("backend.services.domain_profile_service.subprocess.Popen", popen)
    try:
        started = time.perf_counter()
        launched = service.warm_build_environment_async()
        elapsed = time.perf_counter() - started
        assert launched is True
        assert elapsed < 0.1
        assert process_started.wait(timeout=2)
        assert "from methods.patchcore import domain_adaptation" in " ".join(captured["command"])
        assert service.warm_build_environment_async() is False
    finally:
        release.set()
        warmup_thread = service._build_environment_warmup_thread
        if warmup_thread is not None:
            warmup_thread.join(timeout=5)
        service.shutdown()


def test_warming_state_prevents_duplicate_helper_before_thread_is_alive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    service._build_environment_warmup_state = "warming"
    service._build_environment_warmup_thread = threading.Thread(target=lambda: None)
    popen_calls: list[object] = []
    monkeypatch.setattr(
        "backend.services.domain_profile_service.subprocess.Popen",
        lambda *_args, **_kwargs: popen_calls.append(object()),
    )

    assert service.warm_build_environment_async() is False
    assert popen_calls == []
    service.shutdown()


def test_real_build_preempts_optional_isolated_warmup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    task = DomainBuildTask(job_id="d" * 32, payload={})
    import_entered = threading.Event()
    finished = threading.Event()
    service._build_environment_warmup_state = "warming"
    service._build_environment_warmup_done.clear()
    monkeypatch.setattr(
        "backend.services.domain_profile_service._domain_adaptation_module",
        lambda: import_entered.set() or SimpleNamespace(),
    )
    monkeypatch.setattr(service, "_resolve_build_feedback_samples", lambda _task: None)
    monkeypatch.setattr(service, "_validated_build_config", lambda _task: None)

    worker = threading.Thread(
        target=lambda: (service._prepare_build_environment(task), finished.set()),
        daemon=True,
    )
    try:
        worker.start()
        assert import_entered.wait(timeout=0.25) is True
        assert service._build_environment_warmup_stop.is_set() is True
        assert finished.wait(timeout=2)
    finally:
        service._build_environment_warmup_done.set()
        worker.join(timeout=2)
        service.shutdown()


def test_isolated_warmup_timeout_forces_process_reclamation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    events: list[str] = []

    class StuckWarmupProcess:
        returncode = None

        def wait(self, timeout=None):
            events.append(f"wait:{timeout}")
            if self.returncode is None:
                raise subprocess.TimeoutExpired("qa-warmup", timeout)
            return self.returncode

        def poll(self):
            return self.returncode

        def terminate(self):
            events.append("terminate")

        def kill(self):
            events.append("kill")
            self.returncode = -9

    monkeypatch.setattr(
        "backend.services.domain_profile_service.subprocess.Popen",
        lambda *_args, **_kwargs: StuckWarmupProcess(),
    )
    monkeypatch.setattr(
        "backend.services.domain_profile_service.BUILD_ENVIRONMENT_WARMUP_TIMEOUT_SECONDS",
        0.01,
    )
    monkeypatch.setattr(
        "backend.services.domain_profile_service.BUILD_ENVIRONMENT_TERMINATE_TIMEOUT_SECONDS",
        0.01,
    )

    service._warm_build_environment()

    assert service._build_environment_warmup_state == "failed"
    assert "terminate" in events
    assert "kill" in events
    assert service._build_environment_warmup_done.is_set() is True
    service.shutdown()


def test_create_button_does_not_synchronously_save_draft() -> None:
    source = (Path(__file__).resolve().parents[1] / "app" / "domain_resource_creation_page.py").read_text(
        encoding="utf-8"
    )
    create_branch = source.split("if not active_job_id and create_requested:", 1)[1].split(
        "active_job_id = str(st.session_state.get(\"domain_adaptation_job_id\") or \"\")",
        1,
    )[0]
    assert "_save_draft(" not in create_branch
