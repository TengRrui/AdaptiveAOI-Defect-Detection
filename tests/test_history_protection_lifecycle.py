from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.services.feedback_candidate_service import FeedbackCandidateService
from backend.services.history_management_service import HistoryManagementService
from backend.services.result_storage import ResultStorage
from backend.errors import ApiError


class Settings:
    def __init__(self, root: Path):
        self.project_root = root
        self.runtime_root = root / "runtime"
        self.jobs_root = self.runtime_root / "jobs"
        self.upload_root = self.runtime_root / "uploads"
        self.log_root = self.runtime_root / "logs"
        self.memory_banks_root = root / "memory_banks"
        self.config_profiles_root = root / "config_profiles"

    def ensure_runtime_dirs(self) -> None:
        for path in (
            self.runtime_root, self.jobs_root, self.upload_root, self.log_root,
            self.memory_banks_root, self.config_profiles_root,
        ):
            path.mkdir(parents=True, exist_ok=True)


def _services(tmp_path: Path):
    settings = Settings(tmp_path)
    settings.ensure_runtime_dirs()
    storage = ResultStorage(settings)  # type: ignore[arg-type]
    feedback = FeedbackCandidateService(settings, storage)
    history = HistoryManagementService(settings, storage)
    with storage._connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runtime_resource_tombstones(
                tombstone_id TEXT PRIMARY KEY, resource_id TEXT, profile_id TEXT,
                resource_type TEXT, deleted_at TEXT
            )
            """
        )
    return settings, storage, feedback, history


def _result(storage: ResultStorage, *, mode: str, task_id: str, bank: str, index: int = 0) -> str:
    result_id = f"result-{mode}-{index}"
    storage.save_result(
        {
            "result_id": result_id,
            "model_id": "qa-model",
            "prediction": "normal",
            "is_anomaly": False,
            "anomaly_score": 0.1,
            "threshold": 0.5,
            "device": "cpu",
            "metadata": {
                "task_id": task_id,
                "filename": f"{mode}-{index}.png",
                "dataset": "qa",
                "category": "part",
                "memory_bank_id": bank,
                "source_profile_id": bank,
                "config_profile_id": f"config-{bank}",
                "method_mode": "normal_memory",
            },
        },
        request_id=task_id,
        batch_id=task_id if mode == "batch" else None,
    )
    return result_id


def _manifest(settings: Settings, *, mode: str, task_id: str, bank: str, result_id: str = "") -> None:
    directory = settings.jobs_root / task_id
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": task_id,
        "kind": mode,
        "status": "completed",
        "created_at": "2026-08-13T12:00:00+00:00",
        "selection": {
            "dataset": "qa", "category": "part", "memory_bank_id": bank,
            "config_profile_id": f"config-{bank}", "method_mode": "normal_memory",
        },
    }
    if mode == "single":
        payload["result_id"] = result_id
    elif mode == "batch":
        payload["input_count"] = 1
    else:
        payload["summary"] = {
            "successful_processed_frames": 1, "total_decoded_records": 1,
            "normal_frames": 1, "anomaly_frames": 0, "memory_bank_id": bank,
        }
    (directory / "job_manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _candidate(feedback: FeedbackCandidateService, *, mode: str, task_id: str, result_id: str, bank: str):
    return feedback.ensure_candidate(
        {
            "feedback_key": f"feedback-{task_id}", "feedback_id": f"feedback-{task_id}",
            "mode": mode, "task_id": task_id, "sample_id": f"sample-{task_id}",
            "result_id": result_id, "assessment": "incorrect", "corrected_label": "normal",
            "issue_tags": ["false_alarm"],
        },
        {
            "mode": mode, "task_id": task_id, "sample_id": f"sample-{task_id}",
            "result_id": result_id, "dataset": "qa", "category": "part",
            "memory_bank_id": bank, "source_memory_bank_id": bank,
            "source_profile_id": bank, "method_mode": "normal_memory",
            "prediction": "anomaly", "score": 0.7, "threshold": 0.5,
            "official_test_mode": False,
        },
    )


def _job(feedback: FeedbackCandidateService, candidate_id: str, status: str) -> str:
    job_id = f"job-{candidate_id}-{status}"
    now = "2026-08-13T12:01:00+00:00"
    with feedback._connect() as conn:
        conn.execute(
            """
            INSERT INTO feedback_optimization_jobs(
                job_id,base_profile_id,candidate_ids_json,candidate_count,strategy,objective,
                status,progress,created_at,idempotency_key
            ) VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (job_id, "bank-a", json.dumps([candidate_id]), 1, "threshold_only", "qa", status, 0.5, now, job_id),
        )
        conn.execute(
            "INSERT INTO feedback_optimization_items(job_id,candidate_id,role,status,updated_at) VALUES(?,?,?,?,?)",
            (job_id, candidate_id, "normal", "processing" if status == "running" else "completed", now),
        )
    return job_id


def _record(history: HistoryManagementService, task_id: str) -> dict:
    return next(item for item in history._records_projection(force=True) if item["task_id"] == task_id)


def _wait_cleanup_job(history: HistoryManagementService, job_id: str) -> dict:
    job: dict = {}
    for _ in range(250):
        job = history.cleanup_job(job_id)
        if job["status"] in {"completed", "partial", "failed", "cancelled"}:
            return job
        time.sleep(0.02)
    return job


def test_pending_review_and_withdraw_release_immediately(tmp_path: Path) -> None:
    settings, storage, feedback, history = _services(tmp_path)
    task_id = "single-pending"
    result_id = _result(storage, mode="single", task_id=task_id, bank="bank-a")
    _manifest(settings, mode="single", task_id=task_id, bank="bank-a", result_id=result_id)
    candidate = _candidate(feedback, mode="single", task_id=task_id, result_id=result_id, bank="bank-a")
    assert _record(history, task_id)["protected_reasons"] == ["feedback_pending"]

    feedback.review(candidate["candidate_id"], action="withdraw", require_snapshot=False)
    released = _record(history, task_id)
    assert released["protected"] is False
    preview = history.preview({"scope": "selected", "record_keys": [released["record_key"]], "cleanup_files": False})
    job = history.delete(
        {
            "plan_id": preview["plan_id"], "preview_token": preview["preview_token"],
            "confirmation_text": "确认删除", "cleanup_files": False,
        }
    )
    for _ in range(100):
        job = history.cleanup_job(job["cleanup_job_id"])
        if job["status"] in {"completed", "partial", "failed", "cancelled"}:
            break
        time.sleep(0.02)
    assert job["status"] == "completed"
    assert not history.records({"task_id": task_id})["items"]


@pytest.mark.parametrize("mode", ["single", "batch", "video"])
def test_deleted_bank_terminal_feedback_does_not_lock_any_detection_mode(tmp_path: Path, mode: str) -> None:
    settings, storage, feedback, history = _services(tmp_path)
    task_id = f"{mode}-deleted"
    result_id = _result(storage, mode=mode, task_id=task_id, bank="bank-a")
    _manifest(settings, mode=mode, task_id=task_id, bank="bank-a", result_id=result_id)
    candidate = _candidate(feedback, mode=mode, task_id=task_id, result_id=result_id, bank="bank-a")
    with feedback._connect() as conn:
        conn.execute(
            "UPDATE feedback_candidates SET status='ready_to_apply',source_snapshot_sha256='qa-snapshot' WHERE candidate_id=?",
            (candidate["candidate_id"],),
        )
    _job(feedback, candidate["candidate_id"], "completed")
    with storage._connect() as conn:
        conn.execute(
            "INSERT INTO runtime_resource_tombstones(tombstone_id,resource_id,profile_id,resource_type,deleted_at) VALUES(?,?,?,?,?)",
            (f"tomb-{mode}", "resource-a", "bank-a", "target_profile", "2026-08-13T12:05:00+00:00"),
        )
    release = feedback.mark_source_memory_bank_deleted(profile_id="bank-a", resource_id="resource-a")
    record = _record(history, task_id)
    assert release == {"affected": 1, "released": 1, "still_protected": 0, "active_jobs": 0}
    assert record["resource_deleted"] is True
    assert record["protected"] is False
    preview = history.preview({"scope": "selected", "record_keys": [record["record_key"]], "cleanup_files": False})
    assert preview["deletable_records"] == 1
    assert preview["protected_records"] == 0
    job = history.delete(
        {
            "plan_id": preview["plan_id"], "preview_token": preview["preview_token"],
            "confirmation_text": "确认删除", "cleanup_files": False,
        }
    )
    for _ in range(100):
        job = history.cleanup_job(job["cleanup_job_id"])
        if job["status"] in {"completed", "partial", "failed", "cancelled"}:
            break
        time.sleep(0.02)
    assert job["status"] == "completed"
    assert not history.records({"task_id": task_id})["items"]


def test_active_job_protects_only_its_own_bank_and_terminal_states_release(tmp_path: Path) -> None:
    settings, storage, feedback, history = _services(tmp_path)
    records = {}
    candidates = {}
    for bank in ("bank-a", "bank-b"):
        task_id = f"single-{bank}"
        result_id = _result(storage, mode="single", task_id=task_id, bank=bank, index=len(records))
        _manifest(settings, mode="single", task_id=task_id, bank=bank, result_id=result_id)
        candidates[bank] = _candidate(feedback, mode="single", task_id=task_id, result_id=result_id, bank=bank)
        with feedback._connect() as conn:
            conn.execute(
                "UPDATE feedback_candidates SET status='ready_to_apply',source_snapshot_sha256='qa-snapshot' WHERE candidate_id=?",
                (candidates[bank]["candidate_id"],),
            )
        records[bank] = task_id

    job_a = _job(feedback, candidates["bank-a"]["candidate_id"], "completed")
    job_b = _job(feedback, candidates["bank-b"]["candidate_id"], "running")
    with storage._connect() as conn:
        conn.execute(
            "INSERT INTO runtime_resource_tombstones(tombstone_id,resource_id,profile_id,resource_type,deleted_at) VALUES(?,?,?,?,?)",
            ("tomb-a", "resource-a", "bank-a", "target_profile", "2026-08-13T12:05:00+00:00"),
        )
    feedback.mark_source_memory_bank_deleted(profile_id="bank-a", resource_id="resource-a")
    assert _record(history, records["bank-a"])["protected"] is False
    assert _record(history, records["bank-b"])["protected_reasons"] == ["optimization_active"]

    with feedback._connect() as conn:
        conn.execute("UPDATE feedback_optimization_jobs SET status='cancelled' WHERE job_id=?", (job_b,))
        conn.execute("UPDATE feedback_optimization_items SET status='cancelled' WHERE job_id=?", (job_b,))
    assert _record(history, records["bank-b"])["protected"] is False


@pytest.mark.parametrize("mode", ["single", "batch", "video"])
def test_deleted_bank_pending_review_becomes_historical_and_releases(tmp_path: Path, mode: str) -> None:
    settings, storage, feedback, history = _services(tmp_path)
    task_id = f"{mode}-deleted-pending"
    result_id = _result(storage, mode=mode, task_id=task_id, bank="bank-a")
    _manifest(settings, mode=mode, task_id=task_id, bank="bank-a", result_id=result_id)
    candidate = _candidate(feedback, mode=mode, task_id=task_id, result_id=result_id, bank="bank-a")
    assert _record(history, task_id)["protected_reasons"] == ["feedback_pending"]

    release = feedback.mark_source_memory_bank_deleted(profile_id="bank-a", resource_id="resource-a")
    assert release == {"affected": 1, "released": 1, "still_protected": 0, "active_jobs": 0}
    assert feedback.protection_status(candidate["candidate_id"])["protected"] is False
    assert feedback.get_candidate(candidate["candidate_id"])["status"] == "archived"
    record = _record(history, task_id)
    assert record["protected"] is False
    assert record["resource_deleted"] is False  # no tombstone in this focused service test


def test_deleted_bank_pending_review_keeps_only_live_optimization_lock(tmp_path: Path) -> None:
    settings, storage, feedback, history = _services(tmp_path)
    task_id = "single-deleted-active"
    result_id = _result(storage, mode="single", task_id=task_id, bank="bank-a")
    _manifest(settings, mode="single", task_id=task_id, bank="bank-a", result_id=result_id)
    candidate = _candidate(feedback, mode="single", task_id=task_id, result_id=result_id, bank="bank-a")
    job_id = _job(feedback, candidate["candidate_id"], "running")

    release = feedback.mark_source_memory_bank_deleted(profile_id="bank-a", resource_id="resource-a")
    assert release == {"affected": 1, "released": 0, "still_protected": 1, "active_jobs": 1}
    assert _record(history, task_id)["protected_reasons"] == ["optimization_active"]

    with feedback._connect() as conn:
        conn.execute("UPDATE feedback_optimization_jobs SET status='cancelled' WHERE job_id=?", (job_id,))
        conn.execute("UPDATE feedback_optimization_items SET status='cancelled' WHERE job_id=?", (job_id,))
    second = feedback.mark_source_memory_bank_deleted(profile_id="bank-a", resource_id="resource-a")
    assert second == {"affected": 1, "released": 1, "still_protected": 0, "active_jobs": 0}
    assert feedback.get_candidate(candidate["candidate_id"])["status"] == "archived"
    assert _record(history, task_id)["protected"] is False


def test_deleted_bank_reconciliation_repairs_already_marked_stale_candidate(tmp_path: Path) -> None:
    settings, storage, feedback, history = _services(tmp_path)
    task_id = "single-stale-deleted-source"
    result_id = _result(storage, mode="single", task_id=task_id, bank="bank-a")
    _manifest(settings, mode="single", task_id=task_id, bank="bank-a", result_id=result_id)
    candidate = _candidate(feedback, mode="single", task_id=task_id, result_id=result_id, bank="bank-a")
    with feedback._connect() as conn:
        conn.execute(
            "UPDATE feedback_candidates SET source_memory_bank_deleted_at=?,protection_released_at=NULL WHERE candidate_id=?",
            ("2026-08-13T12:05:00+00:00", candidate["candidate_id"]),
        )
    repaired = feedback.mark_source_memory_bank_deleted(profile_id="bank-a", resource_id="resource-a")
    assert repaired == {"affected": 1, "released": 1, "still_protected": 0, "active_jobs": 0}
    refreshed = feedback.get_candidate(candidate["candidate_id"])
    assert refreshed["protection_released_at"]
    assert refreshed["status"] == "archived"
    assert _record(history, task_id)["protected"] is False


def test_reviewed_snapshot_is_not_a_permanent_lock(tmp_path: Path) -> None:
    settings, storage, feedback, history = _services(tmp_path)
    task_id = "single-reviewed"
    result_id = _result(storage, mode="single", task_id=task_id, bank="bank-a")
    _manifest(settings, mode="single", task_id=task_id, bank="bank-a", result_id=result_id)
    candidate = _candidate(feedback, mode="single", task_id=task_id, result_id=result_id, bank="bank-a")
    with feedback._connect() as conn:
        conn.execute(
            "UPDATE feedback_candidates SET status='accepted',source_snapshot_sha256='immutable-qa-snapshot' WHERE candidate_id=?",
            (candidate["candidate_id"],),
        )
    assert _record(history, task_id)["protected"] is False


@pytest.mark.parametrize("status", ["ready_to_apply", "completed", "cancelled", "failed"])
def test_terminal_optimization_jobs_never_keep_history_locked(tmp_path: Path, status: str) -> None:
    settings, storage, feedback, history = _services(tmp_path)
    task_id = f"single-terminal-{status}"
    result_id = _result(storage, mode="single", task_id=task_id, bank="bank-a")
    _manifest(settings, mode="single", task_id=task_id, bank="bank-a", result_id=result_id)
    candidate = _candidate(feedback, mode="single", task_id=task_id, result_id=result_id, bank="bank-a")
    with feedback._connect() as conn:
        conn.execute(
            "UPDATE feedback_candidates SET status='ready_to_apply',source_snapshot_sha256='immutable-qa-snapshot' WHERE candidate_id=?",
            (candidate["candidate_id"],),
        )
    _job(feedback, candidate["candidate_id"], status)
    assert _record(history, task_id)["protected"] is False


@pytest.mark.parametrize("status", ["queued", "preparing", "running", "building", "validating", "applying"])
def test_all_real_active_job_stages_keep_history_locked(tmp_path: Path, status: str) -> None:
    settings, storage, feedback, history = _services(tmp_path)
    task_id = f"single-active-{status}"
    result_id = _result(storage, mode="single", task_id=task_id, bank="bank-a")
    _manifest(settings, mode="single", task_id=task_id, bank="bank-a", result_id=result_id)
    candidate = _candidate(feedback, mode="single", task_id=task_id, result_id=result_id, bank="bank-a")
    with feedback._connect() as conn:
        conn.execute(
            "UPDATE feedback_candidates SET status='ready_to_apply',source_snapshot_sha256='immutable-qa-snapshot' WHERE candidate_id=?",
            (candidate["candidate_id"],),
        )
    _job(feedback, candidate["candidate_id"], status)
    assert _record(history, task_id)["protected_reasons"] == ["optimization_active"]


def test_clear_all_deletes_all_unprotected_modes(tmp_path: Path) -> None:
    settings, storage, feedback, history = _services(tmp_path)
    for index, mode in enumerate(("single", "batch", "video")):
        task_id = f"clear-{mode}"
        result_id = _result(storage, mode=mode, task_id=task_id, bank="bank-a", index=index)
        _manifest(settings, mode=mode, task_id=task_id, bank="bank-a", result_id=result_id)
    preview = history.preview({"scope": "all", "cleanup_files": False}, all_clear=True)
    assert preview["deletable_records"] == 3
    job = history.delete(
        {
            "plan_id": preview["plan_id"], "preview_token": preview["preview_token"],
            "confirmation_text": "确认删除", "cleanup_files": False,
        },
        all_clear=True,
    )
    for _ in range(100):
        job = history.cleanup_job(job["cleanup_job_id"])
        if job["status"] in {"completed", "partial", "failed", "cancelled"}:
            break
        time.sleep(0.02)
    assert job["status"] == "completed"
    assert history.records()["total"] == 0


@pytest.mark.parametrize("mode", ["single", "batch", "video"])
def test_clear_all_is_scoped_to_one_detection_mode_and_updates_statistics(tmp_path: Path, mode: str) -> None:
    settings, storage, feedback, history = _services(tmp_path)
    for index, record_mode in enumerate(("single", "batch", "video")):
        task_id = f"scoped-{record_mode}"
        result_id = _result(storage, mode=record_mode, task_id=task_id, bank="bank-a", index=index)
        _manifest(settings, mode=record_mode, task_id=task_id, bank="bank-a", result_id=result_id)

    revision_before = storage.statistics_revision()
    preview = history.preview(
        {"filters": {"mode": mode}, "cleanup_files": False},
        all_clear=True,
    )
    assert preview["selected_records"] == 1
    assert preview["deletable_records"] == 1
    assert preview["protected_records"] == 0
    assert preview["mode_counts"] == {
        current: int(current == mode) for current in ("single", "batch", "video")
    }

    job = history.delete(
        {
            "plan_id": preview["plan_id"],
            "preview_token": preview["preview_token"],
            "confirmation_text": "确认删除",
            "cleanup_files": False,
        },
        all_clear=True,
    )
    for _ in range(100):
        job = history.cleanup_job(job["cleanup_job_id"])
        if job["status"] in {"completed", "partial", "failed", "cancelled"}:
            break
        time.sleep(0.02)

    assert job["status"] == "completed"
    assert history.records({"mode": mode})["total"] == 0
    assert history.records()["total"] == 2
    assert storage.statistics_revision() > revision_before


def test_clear_all_deletes_eligible_records_and_reports_protected_records(tmp_path: Path) -> None:
    settings, storage, feedback, history = _services(tmp_path)
    completed_result = _result(storage, mode="single", task_id="clear-eligible", bank="bank-a", index=1)
    _manifest(settings, mode="single", task_id="clear-eligible", bank="bank-a", result_id=completed_result)
    running_result = _result(storage, mode="single", task_id="clear-running", bank="bank-a", index=2)
    _manifest(settings, mode="single", task_id="clear-running", bank="bank-a", result_id=running_result)
    running_manifest = settings.jobs_root / "clear-running" / "job_manifest.json"
    running_payload = json.loads(running_manifest.read_text(encoding="utf-8"))
    running_payload["status"] = "running"
    running_manifest.write_text(json.dumps(running_payload), encoding="utf-8")

    preview = history.preview(
        {"filters": {"mode": "single"}, "cleanup_files": False},
        all_clear=True,
    )
    assert preview["selected_records"] == 2
    assert preview["deletable_records"] == 1
    assert preview["protected_records"] == 1
    assert "检测任务运行中" in preview["protected_items"][0]["protected_reason"]

    job = history.delete(
        {
            "plan_id": preview["plan_id"],
            "preview_token": preview["preview_token"],
            "confirmation_text": "确认删除",
            "cleanup_files": False,
        },
        all_clear=True,
    )
    for _ in range(100):
        job = history.cleanup_job(job["cleanup_job_id"])
        if job["status"] in {"completed", "partial", "failed", "cancelled"}:
            break
        time.sleep(0.02)

    assert job["status"] == "completed"
    assert job["success_count"] == 1
    assert job["skipped_count"] == 1
    remaining = history.records({"mode": "single"})
    assert remaining["total"] == 1
    assert remaining["items"][0]["status"] == "running"


def test_bulk_result_delete_uses_bounded_sql_statements(tmp_path: Path) -> None:
    _, storage, _, history = _services(tmp_path)
    result_ids = [f"bulk-delete-{index:04d}" for index in range(1201)]
    now = "2026-08-26T08:00:00+00:00"
    with storage._connect() as conn:
        conn.executemany(
            """
            INSERT INTO detection_results(
                result_id,request_id,batch_id,created_at,model_id,dataset,category,
                filename,prediction,is_anomaly,metadata_json,runtime_revision
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                (
                    result_id,
                    f"request-{index}",
                    None,
                    now,
                    "qa-model",
                    "qa",
                    "part",
                    f"sample-{index}.png",
                    "normal",
                    0,
                    "{}",
                    1,
                )
                for index, result_id in enumerate(result_ids)
            ],
        )

    statements: list[str] = []
    with storage._connect() as conn:
        conn.set_trace_callback(statements.append)
        deleted = history._delete_db_rows(conn, result_ids, [], False)

    deletes = [
        statement
        for statement in statements
        if statement.lstrip().upper().startswith("DELETE FROM DETECTION_RESULTS")
    ]
    assert deleted == 1201
    assert len(deletes) <= 3


def test_projection_cache_ignores_delete_plan_database_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings, storage, _, history = _services(tmp_path)
    task_id = "projection-cache-single"
    result_id = _result(storage, mode="single", task_id=task_id, bank="bank-a")
    _manifest(settings, mode="single", task_id=task_id, bank="bank-a", result_id=result_id)
    history._records_projection()

    build_calls = 0
    original_build = history._build_records

    def counted_build() -> list[dict]:
        nonlocal build_calls
        build_calls += 1
        return original_build()

    monkeypatch.setattr(history, "_build_records", counted_build)
    history.preview(
        {"scope": "all", "filters": {"mode": "single"}, "cleanup_files": False},
        all_clear=True,
    )
    history._records_projection()

    assert build_calls == 0


def test_database_commit_publishes_cleanup_audit_before_file_unlink_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, storage, _, history = _services(tmp_path)
    task_id = "cleanup-durable-queue"
    result_id = _result(storage, mode="single", task_id=task_id, bank="bank-a")
    _manifest(settings, mode="single", task_id=task_id, bank="bank-a", result_id=result_id)
    results_dir = settings.jobs_root / task_id / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    blocked_file = (results_dir / "blocked-artifact.png").resolve()
    blocked_file.write_bytes(b"qa-artifact")
    history._invalidate_projection()
    record_key = str(_record(history, task_id)["record_key"])

    preview = history.preview(
        {"scope": "all", "filters": {"mode": "single"}, "cleanup_files": True},
        all_clear=True,
    )
    entered = threading.Event()
    release = threading.Event()
    original_unlink = Path.unlink

    def blocking_unlink(path: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if path.resolve() == blocked_file:
            entered.set()
            release.wait(5)
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", blocking_unlink)
    submitted = history.delete(
        {
            "plan_id": preview["plan_id"],
            "preview_token": preview["preview_token"],
            "acknowledged": True,
        },
        all_clear=True,
    )
    job_id = str(submitted["cleanup_job_id"])
    try:
        assert entered.wait(5)
        in_progress = history.cleanup_job(job_id)
        with storage._connect() as conn:
            queued = conn.execute(
                """
                SELECT queue.status, queue.relative_path
                FROM history_cleanup_queue AS queue
                JOIN history_cleanup_jobs AS job ON job.audit_id = queue.audit_id
                WHERE job.job_id=?
                """,
                (job_id,),
            ).fetchall()
        assert in_progress["stage"] == "files"
        assert in_progress["deleted_record_keys"] == [record_key]
        assert len(queued) == int(in_progress["total_files"])
        assert any(str(row["relative_path"]).endswith("blocked-artifact.png") for row in queued)
    finally:
        release.set()
    assert _wait_cleanup_job(history, job_id)["status"] == "completed"


def test_file_cleanup_progress_updates_are_chunked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings, storage, _, history = _services(tmp_path)
    task_id = "cleanup-progress-batched"
    result_id = _result(storage, mode="single", task_id=task_id, bank="bank-a")
    _manifest(settings, mode="single", task_id=task_id, bank="bank-a", result_id=result_id)
    results_dir = settings.jobs_root / task_id / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for index in range(60):
        (results_dir / f"artifact-{index:03d}.bin").write_bytes(b"x")
    history._invalidate_projection()

    progress_updates: list[int] = []
    original_update = history._update_cleanup_job

    def tracked_update(job_id: str, **values):  # type: ignore[no-untyped-def]
        if "processed_files" in values:
            progress_updates.append(int(values["processed_files"] or 0))
        return original_update(job_id, **values)

    monkeypatch.setattr(history, "_update_cleanup_job", tracked_update)
    preview = history.preview(
        {"scope": "all", "filters": {"mode": "single"}, "cleanup_files": True},
        all_clear=True,
    )
    submitted = history.delete(
        {
            "plan_id": preview["plan_id"],
            "preview_token": preview["preview_token"],
            "acknowledged": True,
        },
        all_clear=True,
    )
    finished = _wait_cleanup_job(history, str(submitted["cleanup_job_id"]))

    assert finished["status"] == "completed"
    assert finished["processed_files"] == finished["total_files"]
    assert len(progress_updates) <= 5


def test_batch_parent_delete_removes_all_children_and_advances_one_revision(tmp_path: Path) -> None:
    settings, storage, _, history = _services(tmp_path)
    task_id = "batch-parent-consistency"
    for index in range(7):
        _result(storage, mode="batch", task_id=task_id, bank="bank-a", index=index)
    _manifest(settings, mode="batch", task_id=task_id, bank="bank-a")
    history._invalidate_projection()
    revision_before = storage.statistics_revision()

    preview = history.preview(
        {"scope": "all", "filters": {"mode": "batch"}, "cleanup_files": False},
        all_clear=True,
    )
    assert preview["selected_records"] == 1
    assert preview["result_count"] == 7
    submitted = history.delete(
        {
            "plan_id": preview["plan_id"],
            "preview_token": preview["preview_token"],
            "acknowledged": True,
        },
        all_clear=True,
    )
    finished = _wait_cleanup_job(history, str(submitted["cleanup_job_id"]))

    with storage._connect() as conn:
        child_count = int(
            conn.execute("SELECT COUNT(*) AS count FROM detection_results WHERE batch_id=?", (task_id,)).fetchone()["count"]
        )
        tombstone_count = int(
            conn.execute("SELECT COUNT(*) AS count FROM detection_history_tombstones WHERE record_key=?", (task_id,)).fetchone()["count"]
        )
    assert finished["status"] == "completed"
    assert finished["success_count"] == 1
    assert child_count == 0
    assert tombstone_count == 1
    assert storage.statistics_revision() == revision_before + 1


def test_bulk_delete_rolls_back_when_tombstone_write_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings, storage, _, history = _services(tmp_path)
    task_id = "single-transaction-rollback"
    result_id = _result(storage, mode="single", task_id=task_id, bank="bank-a")
    _manifest(settings, mode="single", task_id=task_id, bank="bank-a", result_id=result_id)
    preview = history.preview(
        {"scope": "all", "filters": {"mode": "single"}, "cleanup_files": False},
        all_clear=True,
    )

    def fail_tombstone(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("qa forced tombstone failure")

    monkeypatch.setattr(storage, "mark_history_deleted", fail_tombstone)
    submitted = history.delete(
        {
            "plan_id": preview["plan_id"],
            "preview_token": preview["preview_token"],
            "acknowledged": True,
        },
        all_clear=True,
    )
    finished = _wait_cleanup_job(history, str(submitted["cleanup_job_id"]))

    with storage._connect() as conn:
        row = conn.execute("SELECT result_id FROM detection_results WHERE result_id=?", (result_id,)).fetchone()
    assert finished["status"] == "failed"
    assert row is not None


def test_failed_artifact_cleanup_remains_auditable_and_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, storage, _, history = _services(tmp_path)
    task_id = "cleanup-retry-audit"
    result_id = _result(storage, mode="single", task_id=task_id, bank="bank-a")
    _manifest(settings, mode="single", task_id=task_id, bank="bank-a", result_id=result_id)
    results_dir = settings.jobs_root / task_id / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    retry_file = (results_dir / "retry-artifact.png").resolve()
    retry_file.write_bytes(b"retry")
    history._invalidate_projection()
    preview = history.preview(
        {"scope": "all", "filters": {"mode": "single"}, "cleanup_files": True},
        all_clear=True,
    )

    original_unlink = Path.unlink

    def fail_target_once(path: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if path.resolve() == retry_file:
            raise OSError("qa forced file lock")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_target_once)
    submitted = history.delete(
        {
            "plan_id": preview["plan_id"],
            "preview_token": preview["preview_token"],
            "acknowledged": True,
        },
        all_clear=True,
    )
    finished = _wait_cleanup_job(history, str(submitted["cleanup_job_id"]))
    assert finished["status"] == "partial"
    with storage._connect() as conn:
        pending = conn.execute(
            "SELECT cleanup_id,status FROM history_cleanup_queue WHERE relative_path LIKE ?",
            ("%retry-artifact.png",),
        ).fetchone()
    assert pending is not None
    assert pending["status"] == "pending"

    monkeypatch.setattr(Path, "unlink", original_unlink)
    retried = history.retry_file_cleanup([str(pending["cleanup_id"])])
    with storage._connect() as conn:
        final_status = conn.execute(
            "SELECT status FROM history_cleanup_queue WHERE cleanup_id=?",
            (pending["cleanup_id"],),
        ).fetchone()["status"]
    assert retried == {"attempted": 1, "cleaned": 1, "failed": 0, "message": "清理重试已完成。"}
    assert final_status == "cleaned"
    assert not retry_file.exists()


def test_delete_plan_can_only_submit_one_cleanup_job(tmp_path: Path) -> None:
    settings, storage, _, history = _services(tmp_path)
    task_id = "delete-plan-single-submit"
    result_id = _result(storage, mode="single", task_id=task_id, bank="bank-a")
    _manifest(settings, mode="single", task_id=task_id, bank="bank-a", result_id=result_id)
    preview = history.preview(
        {"scope": "all", "filters": {"mode": "single"}, "cleanup_files": False},
        all_clear=True,
    )
    payload = {
        "plan_id": preview["plan_id"],
        "preview_token": preview["preview_token"],
        "acknowledged": True,
    }

    submitted = history.delete(payload, all_clear=True)
    with pytest.raises(ApiError) as exc_info:
        history.delete(payload, all_clear=True)

    assert exc_info.value.code == "HISTORY_PLAN_ALREADY_USED"
    assert _wait_cleanup_job(history, str(submitted["cleanup_job_id"]))["status"] == "completed"


def test_cleanup_job_db_commit_marker_is_atomic_with_history_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, storage, _, history = _services(tmp_path)
    task_id = "cleanup-job-atomic-marker"
    result_id = _result(storage, mode="single", task_id=task_id, bank="bank-a")
    _manifest(settings, mode="single", task_id=task_id, bank="bank-a", result_id=result_id)
    revision_before = storage.statistics_revision()
    preview = history.preview(
        {"scope": "all", "filters": {"mode": "single"}, "cleanup_files": False},
        all_clear=True,
    )
    original_update = history._update_cleanup_job

    def fail_db_commit_marker(job_id: str, **values):  # type: ignore[no-untyped-def]
        if values.get("stage") == "files":
            raise RuntimeError("qa forced db-commit marker failure")
        return original_update(job_id, **values)

    monkeypatch.setattr(history, "_update_cleanup_job", fail_db_commit_marker)
    submitted = history.delete(
        {
            "plan_id": preview["plan_id"],
            "preview_token": preview["preview_token"],
            "acknowledged": True,
        },
        all_clear=True,
    )
    finished = _wait_cleanup_job(history, str(submitted["cleanup_job_id"]))

    with storage._connect() as conn:
        result_row = conn.execute(
            "SELECT result_id FROM detection_results WHERE result_id=?",
            (result_id,),
        ).fetchone()
        tombstones = int(
            conn.execute(
                "SELECT COUNT(*) AS count FROM detection_history_tombstones WHERE record_key=?",
                (task_id,),
            ).fetchone()["count"]
        )
        cleanup_rows = int(conn.execute("SELECT COUNT(*) AS count FROM history_cleanup_queue").fetchone()["count"])
        audit_rows = int(conn.execute("SELECT COUNT(*) AS count FROM history_management_audit").fetchone()["count"])
    assert finished["status"] == "failed"
    assert result_row is not None
    assert tombstones == 0
    assert cleanup_rows == 0
    assert audit_rows == 0
    assert storage.statistics_revision() == revision_before


def test_retry_does_not_delete_replaced_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings, storage, _, history = _services(tmp_path)
    task_id = "cleanup-retry-replaced-artifact"
    result_id = _result(storage, mode="single", task_id=task_id, bank="bank-a")
    _manifest(settings, mode="single", task_id=task_id, bank="bank-a", result_id=result_id)
    results_dir = settings.jobs_root / task_id / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    target = (results_dir / "replaced-artifact.bin").resolve()
    target.write_bytes(b"original-artifact")
    history._invalidate_projection()
    preview = history.preview(
        {"scope": "all", "filters": {"mode": "single"}, "cleanup_files": True},
        all_clear=True,
    )
    original_unlink = Path.unlink

    def fail_original_cleanup(path: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if path.resolve() == target:
            raise OSError("qa forced initial cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_original_cleanup)
    submitted = history.delete(
        {
            "plan_id": preview["plan_id"],
            "preview_token": preview["preview_token"],
            "acknowledged": True,
        },
        all_clear=True,
    )
    assert _wait_cleanup_job(history, str(submitted["cleanup_job_id"]))["status"] == "partial"
    with storage._connect() as conn:
        queued = conn.execute(
            "SELECT cleanup_id FROM history_cleanup_queue WHERE relative_path LIKE ?",
            ("%replaced-artifact.bin",),
        ).fetchone()
    assert queued is not None

    target.write_bytes(b"new-artifact-at-same-path")
    monkeypatch.setattr(Path, "unlink", original_unlink)
    retried = history.retry_file_cleanup([str(queued["cleanup_id"])])

    with storage._connect() as conn:
        row = conn.execute(
            "SELECT status,error_message FROM history_cleanup_queue WHERE cleanup_id=?",
            (queued["cleanup_id"],),
        ).fetchone()
    assert retried["attempted"] == 1
    assert retried["cleaned"] == 0
    assert retried["failed"] == 1
    assert target.read_bytes() == b"new-artifact-at-same-path"
    assert row["status"] == "pending"
    assert row["error_message"]


def test_retry_claim_prevents_concurrent_cleanup_of_same_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, storage, _, history = _services(tmp_path)
    task_id = "cleanup-retry-atomic-claim"
    result_id = _result(storage, mode="single", task_id=task_id, bank="bank-a")
    _manifest(settings, mode="single", task_id=task_id, bank="bank-a", result_id=result_id)
    results_dir = settings.jobs_root / task_id / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    target = (results_dir / "claim-artifact.bin").resolve()
    target.write_bytes(b"claim-artifact")
    history._invalidate_projection()
    preview = history.preview(
        {"scope": "all", "filters": {"mode": "single"}, "cleanup_files": True},
        all_clear=True,
    )
    original_unlink = Path.unlink

    def fail_initial(path: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if path.resolve() == target:
            raise OSError("qa forced retryable artifact")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_initial)
    submitted = history.delete(
        {
            "plan_id": preview["plan_id"],
            "preview_token": preview["preview_token"],
            "acknowledged": True,
        },
        all_clear=True,
    )
    assert _wait_cleanup_job(history, str(submitted["cleanup_job_id"]))["status"] == "partial"
    with storage._connect() as conn:
        cleanup_id = str(
            conn.execute(
                "SELECT cleanup_id FROM history_cleanup_queue WHERE relative_path LIKE ?",
                ("%claim-artifact.bin",),
            ).fetchone()["cleanup_id"]
        )

    entered = threading.Event()
    release = threading.Event()

    def blocking_retry(path: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if path.resolve() == target:
            entered.set()
            release.wait(5)
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", blocking_retry)
    first_result: dict[str, object] = {}

    def run_first_retry() -> None:
        first_result.update(history.retry_file_cleanup([cleanup_id]))

    worker = threading.Thread(target=run_first_retry)
    worker.start()
    try:
        assert entered.wait(5)
        second_result = history.retry_file_cleanup([cleanup_id])
    finally:
        release.set()
        worker.join(5)

    with storage._connect() as conn:
        status = conn.execute(
            "SELECT status FROM history_cleanup_queue WHERE cleanup_id=?",
            (cleanup_id,),
        ).fetchone()["status"]
    assert first_result["attempted"] == 1
    assert first_result["cleaned"] == 1
    assert second_result["attempted"] == 0
    assert status == "cleaned"


def test_retry_claim_returns_to_pending_when_final_status_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, storage, _, history = _services(tmp_path)
    task_id = "cleanup-retry-final-write-failure"
    result_id = _result(storage, mode="single", task_id=task_id, bank="bank-a")
    _manifest(settings, mode="single", task_id=task_id, bank="bank-a", result_id=result_id)
    results_dir = settings.jobs_root / task_id / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    target = (results_dir / "final-write-artifact.bin").resolve()
    target.write_bytes(b"retry-final-write")
    history._invalidate_projection()
    preview = history.preview(
        {"scope": "all", "filters": {"mode": "single"}, "cleanup_files": True},
        all_clear=True,
    )
    original_unlink = Path.unlink

    def fail_initial(path: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if path.resolve() == target:
            raise OSError("qa forced retryable artifact")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_initial)
    submitted = history.delete(
        {
            "plan_id": preview["plan_id"],
            "preview_token": preview["preview_token"],
            "acknowledged": True,
        },
        all_clear=True,
    )
    assert _wait_cleanup_job(history, str(submitted["cleanup_job_id"]))["status"] == "partial"
    with storage._connect() as conn:
        cleanup_id = str(
            conn.execute(
                "SELECT cleanup_id FROM history_cleanup_queue WHERE relative_path LIKE ?",
                ("%final-write-artifact.bin",),
            ).fetchone()["cleanup_id"]
        )

    monkeypatch.setattr(Path, "unlink", original_unlink)
    original_connect = history._connect
    connection_calls = 0

    class FinalWriteFailureConnection:
        def __init__(self, connection):  # type: ignore[no-untyped-def]
            self.connection = connection

        def __enter__(self):  # type: ignore[no-untyped-def]
            self.connection.__enter__()
            return self

        def __exit__(self, exc_type, exc, traceback):  # type: ignore[no-untyped-def]
            return self.connection.__exit__(exc_type, exc, traceback)

        def __getattr__(self, name: str):
            return getattr(self.connection, name)

        def executemany(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise sqlite3.OperationalError("qa forced final cleanup status write failure")

    def flaky_connect():  # type: ignore[no-untyped-def]
        nonlocal connection_calls
        connection_calls += 1
        connection = original_connect()
        if connection_calls == 2:
            return FinalWriteFailureConnection(connection)
        return connection

    monkeypatch.setattr(history, "_connect", flaky_connect)
    with pytest.raises(sqlite3.OperationalError):
        history.retry_file_cleanup([cleanup_id])
    with storage._connect() as conn:
        status_after_failure = conn.execute(
            "SELECT status FROM history_cleanup_queue WHERE cleanup_id=?",
            (cleanup_id,),
        ).fetchone()["status"]
    assert status_after_failure == "pending"

    monkeypatch.setattr(history, "_connect", original_connect)
    recovered = history.retry_file_cleanup([cleanup_id])
    with storage._connect() as conn:
        final_status = conn.execute(
            "SELECT status FROM history_cleanup_queue WHERE cleanup_id=?",
            (cleanup_id,),
        ).fetchone()["status"]
    assert recovered["attempted"] == 1
    assert final_status == "cleaned"


def test_overlapping_delete_plans_claim_each_record_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, storage, _, history = _services(tmp_path)
    task_id = "overlapping-delete-plans"
    result_id = _result(storage, mode="single", task_id=task_id, bank="bank-a")
    _manifest(settings, mode="single", task_id=task_id, bank="bank-a", result_id=result_id)
    revision_before = storage.statistics_revision()
    first_preview = history.preview(
        {"scope": "all", "filters": {"mode": "single"}, "cleanup_files": False},
        all_clear=True,
    )
    second_preview = history.preview(
        {"scope": "all", "filters": {"mode": "single"}, "cleanup_files": False},
        all_clear=True,
    )
    barrier = threading.Barrier(2)
    original_snapshot = history._create_snapshot
    original_invalidate = history._invalidate_projection
    invalidations = 0

    def synchronized_snapshot(*args, **kwargs):  # type: ignore[no-untyped-def]
        value = original_snapshot(*args, **kwargs)
        barrier.wait(5)
        return value

    def counted_invalidate() -> None:
        nonlocal invalidations
        invalidations += 1
        original_invalidate()

    monkeypatch.setattr(history, "_create_snapshot", synchronized_snapshot)
    monkeypatch.setattr(history, "_invalidate_projection", counted_invalidate)
    first = history.delete(
        {
            "plan_id": first_preview["plan_id"],
            "preview_token": first_preview["preview_token"],
            "acknowledged": True,
        },
        all_clear=True,
    )
    second = history.delete(
        {
            "plan_id": second_preview["plan_id"],
            "preview_token": second_preview["preview_token"],
            "acknowledged": True,
        },
        all_clear=True,
    )
    finished = [
        _wait_cleanup_job(history, str(first["cleanup_job_id"])),
        _wait_cleanup_job(history, str(second["cleanup_job_id"])),
    ]

    with storage._connect() as conn:
        audit_count = int(conn.execute("SELECT COUNT(*) AS count FROM history_management_audit").fetchone()["count"])
        queue_count = int(conn.execute("SELECT COUNT(*) AS count FROM history_cleanup_queue").fetchone()["count"])
        tombstone_count = int(
            conn.execute(
                "SELECT COUNT(*) AS count FROM detection_history_tombstones WHERE record_key=?",
                (task_id,),
            ).fetchone()["count"]
        )
    assert sum(int(job["success_count"]) for job in finished) == 1
    assert sum(int(job["skipped_count"]) for job in finished) == 1
    assert audit_count == 1
    assert queue_count == 0
    assert tombstone_count == 1
    assert storage.statistics_revision() == revision_before + 1
    assert invalidations == 1
