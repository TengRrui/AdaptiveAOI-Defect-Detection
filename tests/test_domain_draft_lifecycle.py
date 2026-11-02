from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.errors import ApiError
from backend.services.domain_profile_service import DomainBuildTask, DomainProfileService


def _service(tmp_path: Path) -> DomainProfileService:
    service = DomainProfileService.__new__(DomainProfileService)
    service.draft_root = tmp_path / "domain_drafts"
    service.draft_root.mkdir(parents=True)
    service._draft_revision_path = service.draft_root / ".revision"
    service._draft_revision = 0
    service._draft_delete_plans = {}
    service._lock = threading.RLock()
    service._tasks = {}
    service.jobs = None
    service.logger = SimpleNamespace(warning=lambda *args, **kwargs: None)
    return service


def _save(service: DomainProfileService, draft_id: str = "domain-draft-qa123456") -> dict:
    return service.save_draft(
        {
            "draft_id": draft_id,
            "current_step": 1,
            "payload": {"friendly_name": "QA草稿", "normal_dir": "C:/qa/normal"},
        }
    )


def test_draft_delete_plan_removes_entity_and_is_idempotent(tmp_path: Path) -> None:
    service = _service(tmp_path)
    draft_id = "domain-draft-qa123456"
    _save(service, draft_id)

    preview = service.delete_draft_preview(draft_id)
    assert preview["draft_id"] == draft_id
    assert preview["allowed"] is True
    assert preview["deletable_count"] == 1

    result = service.delete_draft(
        draft_id,
        plan_id=preview["plan_id"],
        preview_token=preview["preview_token"],
    )
    assert result["deleted"] is True
    assert not (service.draft_root / f"{draft_id}.json").exists()
    retry = service.delete_draft(draft_id, plan_id="old", preview_token="old")
    assert retry["already_deleted"] is True


def test_active_build_protects_draft_until_task_is_terminal(tmp_path: Path) -> None:
    service = _service(tmp_path)
    draft_id = "domain-draft-qa123456"
    _save(service, draft_id)
    service._tasks["job-qa"] = DomainBuildTask(
        job_id="job-qa",
        payload={"draft_id": draft_id},
        state="running",
    )

    preview = service.delete_draft_preview(draft_id)
    assert preview["allowed"] is False
    assert preview["deletable_count"] == 0
    assert preview["protected_count"] == 1
    with pytest.raises(ApiError) as error:
        service.delete_draft(
            draft_id,
            plan_id=preview["plan_id"],
            preview_token=preview["preview_token"],
        )
    assert error.value.code == "DOMAIN_DRAFT_IN_USE"

    service._tasks["job-qa"].state = "cancelled"
    preview_after_cancel = service.delete_draft_preview(draft_id)
    assert preview_after_cancel["allowed"] is True
    result = service.delete_draft(
        draft_id,
        plan_id=preview_after_cancel["plan_id"],
        preview_token=preview_after_cancel["preview_token"],
    )
    assert result["deleted"] is True
