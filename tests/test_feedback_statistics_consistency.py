from __future__ import annotations

from pathlib import Path

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


def _service(tmp_path: Path) -> tuple[FeedbackCandidateService, ResultStorage]:
    settings = DummySettings(tmp_path)
    storage = ResultStorage(settings)  # type: ignore[arg-type]
    return FeedbackCandidateService(settings, storage), storage


def _add_candidate(
    service: FeedbackCandidateService,
    key: str,
    *,
    label: str = "normal",
    memory_bank_id: str = "bank-a",
) -> dict:
    return service.ensure_candidate(
        {
            "feedback_key": key,
            "feedback_id": f"feedback-{key}",
            "mode": "single",
            "task_id": f"task-{key}",
            "sample_id": f"sample-{key}",
            "result_id": f"result-{key}",
            "assessment": "incorrect",
            "corrected_label": label,
            "corrected_defect_type": "scratch" if label == "defect" else None,
            "issue_tags": ["missed_detection" if label == "defect" else "false_alarm"],
        },
        {
            "mode": "single",
            "task_id": f"task-{key}",
            "sample_id": f"sample-{key}",
            "result_id": f"result-{key}",
            "dataset": "qa",
            "category": "part",
            "memory_bank_id": memory_bank_id,
            "source_memory_bank_id": memory_bank_id,
            "source_profile_id": memory_bank_id,
            "method_mode": "normal_memory",
            "prediction": "anomaly" if label == "normal" else "normal",
            "score": 0.5,
            "threshold": 0.6,
            "official_test_mode": False,
        },
    )


def test_feedback_statistics_lifecycle_and_revisions(tmp_path: Path) -> None:
    service, storage = _service(tmp_path)
    assert service.statistics()["total"] == 0
    feedback_revision_before = service.revision()["revision"]
    statistics_revision_before = storage.statistics_revision()

    candidate = _add_candidate(service, "one")
    created = service.statistics()
    assert created["total"] == 1
    assert created["pending_count"] == 1
    assert created["approved_count"] == 0
    assert service.revision()["revision"] > feedback_revision_before
    assert storage.statistics_revision() > statistics_revision_before

    service.review(
        candidate["candidate_id"],
        action="accept",
        corrected_label="normal",
        require_snapshot=False,
    )
    approved = service.statistics()
    assert approved["total"] == 1
    assert approved["pending_count"] == 0
    assert approved["approved_count"] == 1

    service.delete_candidates(
        [candidate["candidate_id"]],
        acknowledged=True,
    )
    deleted = service.statistics()
    assert deleted["total"] == 0
    assert deleted["by_status"] == {}
    assert deleted["deleted_count"] == 0
    assert deleted["audit_deleted_count"] == 1

    # Recreating the service simulates a backend restart; durable revisions
    # and the active-record statistics must not resurrect deleted feedback.
    restarted = FeedbackCandidateService(service.settings, storage)
    assert restarted.statistics()["total"] == 0
    assert restarted.list_candidates(limit=10) == []


def test_withdraw_batch_delete_and_used_audit_are_not_stale_feedback(tmp_path: Path) -> None:
    service, _storage = _service(tmp_path)
    withdrawn = _add_candidate(service, "withdrawn")
    service.review(withdrawn["candidate_id"], action="withdraw", require_snapshot=False)
    withdrawn_stats = service.statistics()
    assert withdrawn_stats["total"] == 1
    assert withdrawn_stats["withdrawn_count"] == 1

    first = _add_candidate(service, "batch-one")
    second = _add_candidate(service, "batch-two", label="defect")
    result = service.delete_candidates(
        [withdrawn["candidate_id"], first["candidate_id"], second["candidate_id"]],
        acknowledged=True,
    )
    assert result["success_count"] == 3
    assert service.statistics()["total"] == 0

    used = _add_candidate(service, "used")
    with service._connect() as conn:
        conn.execute(
            "UPDATE feedback_candidates SET status='applied' WHERE candidate_id=?",
            (used["candidate_id"],),
        )
    assert service.statistics()["used_count"] == 1
    preview = service.delete_preview([used["candidate_id"]])
    assert preview["audit_only_count"] == 1
    service.delete_candidates([used["candidate_id"]], acknowledged=True)

    final = service.statistics()
    assert final["total"] == 0
    assert final["used_count"] == 0
    assert "deleted" not in final["by_status"]
    assert final["audit_deleted_count"] == 4


def test_public_statistics_scope_matches_visible_feedback_resources(tmp_path: Path) -> None:
    service, _storage = _service(tmp_path)
    _add_candidate(service, "visible", memory_bank_id="bank-visible")
    _add_candidate(service, "orphan", memory_bank_id="bank-removed")

    all_records = service.statistics()
    visible_records = service.statistics(memory_bank_ids=("bank-visible",))
    empty_scope = service.statistics(memory_bank_ids=())

    assert all_records["total"] == 2
    assert all_records["scope"] == "all_records"
    assert visible_records["total"] == 1
    assert visible_records["scope"] == "active_memory_banks"
    assert visible_records["by_profile"] == {"bank-visible": 1}
    assert empty_scope["total"] == 0
    assert empty_scope["by_status"] == {}
