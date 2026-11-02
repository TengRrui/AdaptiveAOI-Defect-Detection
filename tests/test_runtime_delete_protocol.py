from __future__ import annotations

import inspect
import threading

import pytest

from backend.api.schemas import RuntimeBatchActionRequest
from backend.errors import ApiError
from backend.services.runtime_resource_service import RuntimeResourceService
from bridge.api_client import IntegrationApiClient


def test_batch_client_uses_checkbox_protocol_without_confirmation_text() -> None:
    api = IntegrationApiClient(base_url="http://127.0.0.1:8000")
    calls: list[dict] = []

    def fake_request(method: str, path: str, **kwargs):
        calls.append({"method": method, "path": path, **kwargs})
        return {"success": True}

    api._request = fake_request  # type: ignore[method-assign]
    api.runtime_resource_batch_delete(
        ["resource_a"],
        preview_token="preview-token",
        acknowledged=True,
    )

    assert calls[0]["path"] == "/api/runtime/resources/batch-delete"
    assert calls[0]["json"]["acknowledged"] is True
    assert "confirmation_text" not in calls[0]["json"]
    assert "confirmation_text" not in str(inspect.signature(IntegrationApiClient.runtime_resource_batch_delete))


def test_batch_request_accepts_acknowledged_without_confirmation_text() -> None:
    request = RuntimeBatchActionRequest(
        resource_ids=["resource_a"],
        preview_token="preview-token",
        acknowledged=True,
    )
    assert request.acknowledged is True
    # The deprecated compatibility field is not required and is never read by
    # the runtime-resource routes.
    assert request.confirmation_text == ""


def test_service_rejects_batch_delete_without_acknowledgement() -> None:
    service = object.__new__(RuntimeResourceService)
    service.test_lock = None

    with pytest.raises(ApiError, match="核对删除预览"):
        service.batch_delete(["resource_a"], preview_token="preview-token")


def test_service_batch_delete_passes_acknowledged_to_single_delete() -> None:
    service = object.__new__(RuntimeResourceService)
    service.test_lock = None
    service._batch_delete_previews = {
        "batch-token": {
            "items": {"resource_a": "single-token"},
            "expires_at": 9_999_999_999.0,
        }
    }
    service._record_batch_operation = lambda *_args, **_kwargs: {"success_count": 1}  # type: ignore[method-assign]
    calls: list[dict] = []

    def fake_delete(resource_id: str, preview_token: str, actor: str, **kwargs):
        calls.append({"resource_id": resource_id, "preview_token": preview_token, "actor": actor, **kwargs})
        return {"released_bytes": 12, "operation_id": "op-1"}

    service.delete = fake_delete  # type: ignore[method-assign]
    result = service.batch_delete(
        ["resource_a"],
        preview_token="batch-token",
        acknowledged=True,
        actor="qa",
    )

    assert result["success_count"] == 1
    assert calls == [{
        "resource_id": "resource_a",
        "preview_token": "single-token",
        "actor": "qa",
        "acknowledged": True,
        "delete_artifacts": True,
        "reason": "",
        "current_profile_action": "cancel",
        "replacement_resource_id": None,
    }]


def test_service_single_delete_requires_acknowledgement_before_plan_use() -> None:
    service = object.__new__(RuntimeResourceService)
    service.test_lock = None
    service._delete_lock = threading.RLock()

    with pytest.raises(ApiError, match="核对删除预览"):
        service.delete("resource_a", "preview-token")
