from __future__ import annotations

import pytest

from backend.errors import ApiError
from backend.services.evaluation_lock_service import EvaluationLockService


@pytest.mark.parametrize("selection", [
    {"data_role": "archived_test"},
    {"metadata": {"data_role": "archived_test"}},
    {"dataset": "archived_test"},
    {"evaluation_mode": "archived_test"},
])
def test_legacy_test_suffix_remains_read_only(selection) -> None:
    service = EvaluationLockService(FakeSettings())
    with pytest.raises(ApiError) as error:
        service.require_write_allowed("memory_update", context=selection)
    assert error.value.code == "OFFICIAL_TEST_WRITE_LOCKED"


class FakeSettings:
    def __init__(self, **values):
        self.values = dict(values)

    def dynamic_settings(self) -> dict:
        return dict(self.values)


@pytest.mark.parametrize(
    "operation",
    ["feedback_update", "memory_update", "profile_apply", "calibration"],
)
def test_official_test_mode_rejects_all_resource_writes(operation: str) -> None:
    service = EvaluationLockService(FakeSettings(official_test_mode=True))

    with pytest.raises(ApiError) as caught:
        service.require_write_allowed(operation)

    assert caught.value.code == "OFFICIAL_TEST_WRITE_LOCKED"
    assert caught.value.status_code == 409
    assert caught.value.details["test_data_update_allowed"] is False


@pytest.mark.parametrize(
    "context",
    [
        {"data_role": "official_test_set"},
        {"deployment_scope": "test_only"},
        {"dataset": "test_data", "category": "test_domain"},
    ],
)
def test_test_only_sources_are_rejected_even_when_global_lock_is_off(context: dict) -> None:
    service = EvaluationLockService(FakeSettings(official_test_mode=False))

    with pytest.raises(ApiError) as caught:
        service.require_write_allowed("memory_update", context=context)

    assert caught.value.code == "OFFICIAL_TEST_WRITE_LOCKED"
    assert caught.value.details["test_only_source"] is True


def test_non_test_calibration_data_remains_writable_before_official_test() -> None:
    service = EvaluationLockService(FakeSettings(official_test_mode=False))

    service.require_write_allowed(
        "calibration",
        context={"data_role": "defect_calibration_set", "dataset": "mvtec", "category": "bottle"},
    )


def test_calibration_metrics_are_not_reported_as_independent_validation() -> None:
    service = EvaluationLockService(FakeSettings())

    result = service.evaluation_fields({"f1": 0.91, "recall": 0.97})

    assert result["evaluation_mode"] == "calibration_only"
    assert result["calibration_result"]["f1"] == 0.91
    assert result["validation_result"]["status"] == "not_run"
    assert result["validation_result"]["data_role"] == "independent_validation_required"
