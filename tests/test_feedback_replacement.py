from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from backend.errors import ApiError
from backend.services.feedback_optimization_service import FeedbackOptimizationService
from backend.services.runtime_resource_service import RuntimeResourceService


def _service(*, active_count: int = 0):
    service = RuntimeResourceService.__new__(RuntimeResourceService)
    service._apply_lock = threading.RLock()
    service._delete_lock = threading.RLock()
    service.test_lock = None
    service.logger = SimpleNamespace(warning=lambda *_args, **_kwargs: None)
    service.registry = SimpleNamespace(
        status=lambda: {"runtimes": [{"active_count": active_count}]}
    )
    service._ensure_resource_mutation_idle = lambda **_kwargs: None
    rows = {
        "old-resource": {
            "resource_id": "old-resource",
            "resource_type": "target_profile",
            "profile_id": "bank-a",
            "frozen": True,
            "archived": False,
        },
        "new-resource": {
            "resource_id": "new-resource",
            "resource_type": "target_profile",
            "profile_id": "bank-a-v2",
            "frozen": True,
            "archived": False,
        },
    }
    service._get = lambda resource_id: dict(rows[resource_id])
    service._is_current_binding = lambda item: item.get("profile_id") == "bank-a"
    calls: list[tuple[str, dict]] = []
    service.apply_profile = lambda resource_id, **kwargs: calls.append(("apply", {"resource_id": resource_id, **kwargs})) or {"ready": True}
    service.archive = lambda resource_id, **kwargs: calls.append(("archive", {"resource_id": resource_id, **kwargs})) or {"archived": True}
    service.delete_preview = lambda resource_id, **kwargs: calls.append(("preview", {"resource_id": resource_id, **kwargs})) or {"allowed": True, "preview_token": "token"}
    service.delete = lambda resource_id, **kwargs: calls.append(("delete", {"resource_id": resource_id, **kwargs})) or {"deleted": True}
    return service, calls


def test_replace_switches_and_warms_before_cleanup() -> None:
    service, calls = _service()
    result = service.replace_profile("old-resource", "new-resource", actor="qa", feedback_job_id="task-1")

    assert result["replaced"] is True
    assert result["deleted"] is True
    assert [item[0] for item in calls] == ["apply", "archive", "preview", "delete"]
    assert calls[0][1]["resource_id"] == "new-resource"
    assert calls[0][1]["feedback_job_id"] == "task-1"
    assert calls[1][1]["current_profile_action"] == "cancel"
    assert calls[1][1]["exclude_feedback_job_id"] == "task-1"
    assert calls[3][1]["preview_token"] == "token"
    assert calls[3][1]["acknowledged"] is True
    assert "confirmation_text" not in calls[3][1]


def test_replace_blocks_when_runtime_is_in_use() -> None:
    service, calls = _service(active_count=1)
    with pytest.raises(ApiError) as exc_info:
        service.replace_profile("old-resource", "new-resource", feedback_job_id="task-2")
    assert exc_info.value.code == "RUNTIME_IN_USE"
    assert calls == []


def test_backend_requires_explicit_replacement_confirmation() -> None:
    service = FeedbackOptimizationService.__new__(FeedbackOptimizationService)
    service._require_test_write_allowed = lambda *_args, **_kwargs: None
    service._runtime_state = lambda: {"memory_bank_id": "bank-a"}
    with pytest.raises(ApiError) as exc_info:
        service.create_job(
            {
                "base_profile_id": "bank-a",
                "candidate_ids": ["sample-1"],
                "publish_mode": "replace_original",
                "replace_confirm": False,
            }
        )
    assert exc_info.value.code == "REPLACEMENT_CONFIRMATION_REQUIRED"
