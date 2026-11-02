from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from backend.services.feedback_candidate_service import FeedbackCandidateService
from backend.services.result_storage import ResultStorage


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
            self.runtime_root,
            self.jobs_root,
            self.upload_root,
            self.log_root,
            self.memory_banks_root,
            self.config_profiles_root,
        ):
            path.mkdir(parents=True, exist_ok=True)


def _service(tmp_path: Path) -> tuple[Settings, ResultStorage, FeedbackCandidateService]:
    settings = Settings(tmp_path)
    settings.ensure_runtime_dirs()
    storage = ResultStorage(settings)  # type: ignore[arg-type]
    feedback = FeedbackCandidateService(settings, storage)
    with feedback._connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runtime_resources(
                resource_id TEXT PRIMARY KEY, profile_id TEXT, resource_type TEXT,
                archived INTEGER NOT NULL DEFAULT 0, metadata_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        conn.execute(
            "INSERT INTO runtime_resources(resource_id,profile_id,resource_type,archived,metadata_json) VALUES(?,?,?,?,?)",
            ("resource-bank-a", "bank-a", "target_profile", 0, json.dumps({"memory_bank_id": "bank-a"})),
        )
    return settings, storage, feedback


def _accepted_candidate(
    settings: Settings,
    feedback: FeedbackCandidateService,
    *,
    candidate_id: str = "candidate-a",
) -> str:
    now = "2026-08-13T12:00:00+00:00"
    snapshot = settings.jobs_root / "feedback_candidates" / candidate_id / "snapshots" / "snapshot_v0001"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "source.png").write_bytes(b"reviewed-source")
    context = {
        "source_memory_bank_id": "bank-a",
        "source_profile_id": "bank-a",
        "source_target_domain_id": "bank-a",
        "official_test_mode": False,
    }
    with feedback._connect() as conn:
        conn.execute(
            """
            INSERT INTO feedback_candidates(
                candidate_id,feedback_id,feedback_key,mode,task_id,sample_id,result_id,
                dataset,category,memory_bank_id,base_memory_bank_id,base_profile_id,
                original_prediction,corrected_label,status,created_at,updated_at,reviewed_at,
                context_json,source_snapshot_ref,source_snapshot_sha256
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                candidate_id,"feedback-a",f"feedback-{candidate_id}","single","task-a","sample-a","result-a",
                "qa","part","bank-a","bank-a","bank-a","anomaly","normal","accepted",now,now,now,
                json.dumps(context),
                f"feedback_candidates/{candidate_id}/snapshots/snapshot_v0001",
                "a" * 64,
            ),
        )
    return candidate_id


def test_reviewed_snapshot_is_available_and_detection_row_is_not_required(tmp_path: Path) -> None:
    settings, storage, feedback = _service(tmp_path)
    candidate_id = _accepted_candidate(settings, feedback)
    # The reviewed immutable snapshot is the build input; the original result
    # may be removed by detection-history maintenance.
    with storage._connect() as conn:
        conn.execute("DELETE FROM detection_results WHERE result_id='result-a'")

    available = feedback.list_available_reviewed_samples()
    assert [item["candidate_id"] for item in available] == [candidate_id]
    assert available[0]["source_memory_bank_id"] == "bank-a"


def test_reviewed_video_frame_snapshot_is_available(tmp_path: Path) -> None:
    settings, _storage, feedback = _service(tmp_path)
    candidate_id = _accepted_candidate(settings, feedback, candidate_id="video-candidate")
    snapshot = settings.jobs_root / "feedback_candidates" / candidate_id / "snapshots" / "snapshot_v0001"
    (snapshot / "source.png").replace(snapshot / "source_frame.png")
    with feedback._connect() as conn:
        conn.execute("UPDATE feedback_candidates SET mode='video' WHERE candidate_id=?", (candidate_id,))

    available = feedback.list_available_reviewed_samples()

    assert [item["candidate_id"] for item in available] == [candidate_id]
    assert available[0]["source_name"] == "source_frame.png"


def test_withdrawn_or_deleted_feedback_is_not_available(tmp_path: Path) -> None:
    settings, _storage, feedback = _service(tmp_path)
    candidate_id = _accepted_candidate(settings, feedback)
    with feedback._connect() as conn:
        conn.execute("UPDATE feedback_candidates SET status='withdrawn' WHERE candidate_id=?", (candidate_id,))
    assert feedback.list_available_reviewed_samples() == []


def test_deleted_source_memory_bank_removes_sample_but_preserves_audit_row(tmp_path: Path) -> None:
    settings, _storage, feedback = _service(tmp_path)
    candidate_id = _accepted_candidate(settings, feedback)
    with feedback._connect() as conn:
        conn.execute("DELETE FROM runtime_resources WHERE profile_id='bank-a'")
        conn.execute(
            "UPDATE feedback_candidates SET source_memory_bank_deleted_at=? WHERE candidate_id=?",
            ("2026-08-13T12:05:00+00:00", candidate_id),
        )

    assert feedback.list_available_reviewed_samples() == []
    historical = feedback.get_candidate(candidate_id)
    assert historical["status"] == "accepted"
    assert historical["source_snapshot_sha256"] == "a" * 64


def test_legacy_consumed_feedback_remains_reusable(tmp_path: Path) -> None:
    settings, _storage, feedback = _service(tmp_path)
    candidate_id = _accepted_candidate(settings, feedback)
    with feedback._connect() as conn:
        conn.execute("UPDATE feedback_candidates SET status='consumed' WHERE candidate_id=?", (candidate_id,))
    available = feedback.list_available_reviewed_samples()
    assert [item["candidate_id"] for item in available] == [candidate_id]


def test_usage_history_is_many_to_many_and_unique_inside_one_task(tmp_path: Path) -> None:
    settings, _storage, feedback = _service(tmp_path)
    candidate_id = _accepted_candidate(settings, feedback)
    with feedback._connect() as conn:
        for index in (1, 2):
            job_id = f"task-{index}"
            target_id = f"bank-a-v{index + 1}"
            conn.execute(
                """
                INSERT INTO feedback_optimization_jobs(
                    job_id,base_profile_id,candidate_ids_json,strategy,objective,status,
                    progress,stage,message,constraints_json,candidate_profile_id,created_at,finished_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    job_id,"bank-a",json.dumps([candidate_id]),"threshold_only","f1",
                    "ready_to_apply",1.0,"result","done","{}",target_id,
                    f"2026-08-13T12:0{index}:00+00:00",f"2026-08-13T12:0{index}:30+00:00",
                ),
            )
            conn.execute(
                """
                INSERT INTO feedback_optimization_items(
                    job_id,candidate_id,role,status,source_memory_bank_id,
                    target_memory_bank_id,target_memory_bank_version,strategy,
                    used_at,usage_result,generated_profile_id
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    job_id,candidate_id,"normal","completed","bank-a",target_id,
                    str(index + 1),"threshold_only",f"2026-08-13T12:0{index}:30+00:00",
                    "ready_to_apply",target_id,
                ),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO feedback_optimization_items(job_id,candidate_id,role,status) VALUES(?,?,?,?)",
                ("task-1", candidate_id, "normal", "completed"),
            )

    available = feedback.list_available_reviewed_samples()
    assert [item["candidate_id"] for item in available] == [candidate_id]
    assert available[0]["usage_count"] == 2
    assert {item["optimization_task_id"] for item in available[0]["usage_history"]} == {"task-1", "task-2"}


def test_pending_review_and_withdrawal_immediately_change_pool_membership(tmp_path: Path) -> None:
    settings, _storage, feedback = _service(tmp_path)
    candidate_id = _accepted_candidate(settings, feedback)
    with feedback._connect() as conn:
        conn.execute(
            "UPDATE feedback_candidates SET status='pending', reviewed_at=NULL WHERE candidate_id=?",
            (candidate_id,),
        )
    assert feedback.list_available_reviewed_samples() == []

    with feedback._connect() as conn:
        conn.execute(
            "UPDATE feedback_candidates SET status='accepted', reviewed_at=? WHERE candidate_id=?",
            ("2026-08-13T12:10:00+00:00", candidate_id),
        )
    assert [item["candidate_id"] for item in feedback.list_available_reviewed_samples()] == [candidate_id]

    with feedback._connect() as conn:
        conn.execute("UPDATE feedback_candidates SET status='withdrawn' WHERE candidate_id=?", (candidate_id,))
    assert feedback.list_available_reviewed_samples() == []
