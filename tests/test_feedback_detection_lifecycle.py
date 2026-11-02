from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from backend.errors import ApiError
from backend.services.feedback_candidate_service import FeedbackCandidateService
from backend.services.result_storage import ResultStorage


class DummySettings:
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


def _service(tmp_path: Path) -> tuple[FeedbackCandidateService, ResultStorage, Path]:
    settings = DummySettings(tmp_path)
    storage = ResultStorage(settings)  # type: ignore[arg-type]
    source = tmp_path / "same-image.png"
    source.write_bytes(b"same-image-bytes")
    return FeedbackCandidateService(settings, storage), storage, source


def _context(source: Path, result_id: str, *, task_id: str | None = None) -> dict:
    task = task_id or f"task-{result_id}"
    return {
        "mode": "batch",
        "task_id": task,
        "sample_id": f"sample-{result_id}",
        "result_id": result_id,
        "dataset": "qa",
        "category": "part",
        "memory_bank_id": "bank-a",
        "source_memory_bank_id": "bank-a",
        "source_profile_id": "bank-a",
        "config_profile_id": "config-a",
        "config_profile_version": 2,
        "method_mode": "normal_memory",
        "prediction": "anomaly",
        "score": 0.7,
        "threshold": 0.5,
        "source_sha256": "identical-content-hash",
        "_source_path": str(source),
        "official_test_mode": False,
    }


VALUES = {
    "assessment": "incorrect",
    "corrected_label": "normal",
    "issue_tags": ["false_alarm"],
    "application_mode": "save_only",
    "note": "qa lifecycle",
}


def test_same_result_is_idempotent_but_new_result_with_same_bytes_is_independent(tmp_path: Path) -> None:
    service, _storage, source = _service(tmp_path)

    first_feedback, first_candidate = service.submit_feedback(VALUES, _context(source, "result-1"))
    repeated_feedback, repeated_candidate = service.submit_feedback(VALUES, _context(source, "result-1"))
    assert repeated_feedback["feedback_id"] == first_feedback["feedback_id"]
    assert repeated_candidate["candidate_id"] == first_candidate["candidate_id"]

    second_feedback, second_candidate = service.submit_feedback(VALUES, _context(source, "result-2"))
    assert second_feedback["feedback_id"] != first_feedback["feedback_id"]
    assert second_candidate["candidate_id"] != first_candidate["candidate_id"]
    assert second_candidate["status"] == "pending_review"
    assert second_candidate["base_memory_bank_id"] == "bank-a"
    assert second_candidate["config_profile_id"] == "config-a"

    with service._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM inspection_feedback").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM feedback_candidates").fetchone()[0] == 2


def test_deleted_audit_candidate_does_not_occupy_feedback_identity(tmp_path: Path) -> None:
    service, _storage, source = _service(tmp_path)
    first_feedback, first_candidate = service.submit_feedback(VALUES, _context(source, "result-1"))
    with service._connect() as conn:
        conn.execute(
            "UPDATE feedback_candidates SET status='applied' WHERE candidate_id=?",
            (first_candidate["candidate_id"],),
        )
    deleted = service.delete_candidates([first_candidate["candidate_id"]], acknowledged=True)
    assert deleted["success_count"] == 1

    new_feedback, new_candidate = service.submit_feedback(VALUES, _context(source, "result-1"))
    assert new_feedback["feedback_id"] != first_feedback["feedback_id"]
    assert new_candidate["candidate_id"] != first_candidate["candidate_id"]
    assert new_candidate["status"] == "pending_review"
    with service._connect() as conn:
        audit = conn.execute(
            "SELECT status,feedback_key FROM feedback_candidates WHERE candidate_id=?",
            (first_candidate["candidate_id"],),
        ).fetchone()
        assert audit["status"] == "deleted"
        assert str(audit["feedback_key"]).startswith("audit:")
        active = conn.execute(
            "SELECT COUNT(*) FROM feedback_candidates WHERE result_id='result-1' AND status<>'deleted'"
        ).fetchone()[0]
        assert active == 1


def test_three_detection_events_of_same_image_create_three_feedback_events(tmp_path: Path) -> None:
    service, _storage, source = _service(tmp_path)
    pairs = [service.submit_feedback(VALUES, _context(source, f"result-{index}")) for index in range(3)]
    assert len({record["feedback_id"] for record, _candidate in pairs}) == 3
    assert len({candidate["candidate_id"] for _record, candidate in pairs}) == 3
    assert {candidate["status"] for _record, candidate in pairs} == {"pending_review"}


def test_candidate_failure_rolls_back_feedback_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service, _storage, source = _service(tmp_path)

    def fail_candidate(*_args, **_kwargs):
        raise sqlite3.IntegrityError("qa candidate failure")

    monkeypatch.setattr(service, "ensure_candidate", fail_candidate)
    with pytest.raises(ApiError) as exc_info:
        service.submit_feedback(VALUES, _context(source, "result-rollback"))
    assert exc_info.value.code == "FEEDBACK_SUBMISSION_FAILED"
    with service._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM inspection_feedback").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM feedback_candidates").fetchone()[0] == 0


def test_mismatched_historical_candidate_cannot_produce_false_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _storage, source = _service(tmp_path)

    def stale_candidate(*_args, **_kwargs):
        return {
            "candidate_id": "historical-candidate",
            "feedback_id": "historical-feedback",
            "result_id": "historical-result",
            "status": "deleted",
        }

    monkeypatch.setattr(service, "ensure_candidate", stale_candidate)
    with pytest.raises(ApiError) as exc_info:
        service.submit_feedback(VALUES, _context(source, "new-result"))
    assert exc_info.value.code == "FEEDBACK_CANDIDATE_LINK_FAILED"
    with service._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM inspection_feedback").fetchone()[0] == 0
