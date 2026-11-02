"""Evaluation data-role isolation and official-test write lock.

This module is deliberately independent from the anomaly scoring code.  It
provides one policy boundary used by every service that can mutate a profile,
memory bank, threshold or calibration resource.
"""
from __future__ import annotations

from typing import Any

from backend.errors import ApiError


TEST_ONLY_SCOPES = {"test", "test_only", "development", "development_only"}
TEST_DATA_ROLES = {
    "test",
    "test_set",
    "official_test",
    "official_test_set",
    "non_public_test",
    "evaluation_test",
}

TEST_DATA_IDENTIFIERS = {
    "test_data",
    "test_domain",
    "official_test",
    "official_test_set",
    "non_public_test",
    "evaluation_test",
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _is_test_marker(value: Any, known_markers: set[str]) -> bool:
    """Keep legacy names ending in _test read-only without depending on their prefix."""
    token = str(value or "").strip().lower()
    return token in known_markers or token.endswith("_test")


def _scope(value: dict[str, Any]) -> str:
    metadata = _mapping(value.get("metadata"))
    return str(
        value.get("deployment_scope")
        or value.get("profile_scope")
        or metadata.get("deployment_scope")
        or metadata.get("profile_scope")
        or ""
    ).strip().lower()


def _data_role(value: dict[str, Any]) -> str:
    metadata = _mapping(value.get("metadata"))
    return str(
        value.get("data_role")
        or value.get("sample_role")
        or value.get("dataset_role")
        or metadata.get("data_role")
        or metadata.get("sample_role")
        or ""
    ).strip().lower()


def _has_test_identifier(value: dict[str, Any]) -> bool:
    """Recognize legacy test resources that predate explicit role metadata."""

    metadata = _mapping(value.get("metadata"))
    candidates = (
        value.get("dataset"),
        value.get("category"),
        value.get("profile_id"),
        value.get("memory_bank_id"),
        metadata.get("dataset"),
        metadata.get("category"),
        metadata.get("profile_id"),
        metadata.get("memory_bank_id"),
    )
    return any(_is_test_marker(item, TEST_DATA_IDENTIFIERS) for item in candidates)


class EvaluationLockService:
    """Single hard gate for evaluation-test resource mutations."""

    def __init__(self, settings: Any):
        self.settings = settings

    def state(
        self,
        *,
        profile: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        runtime = _mapping(self.settings.dynamic_settings())
        profile_value = _mapping(profile)
        context_value = _mapping(context)
        profile_metadata = _mapping(profile_value.get("metadata"))
        official = bool(
            runtime.get("official_test_mode")
            or profile_value.get("official_test_mode")
            or profile_metadata.get("official_test_mode")
            or context_value.get("official_test_mode")
        )
        test_only = bool(
            context_value.get("test_only")
            or profile_value.get("test_only")
            or _scope(profile_value) in TEST_ONLY_SCOPES
            or _scope(context_value) in TEST_ONLY_SCOPES
            or _is_test_marker(_data_role(profile_value), TEST_DATA_ROLES)
            or _is_test_marker(_data_role(context_value), TEST_DATA_ROLES)
            or _has_test_identifier(profile_value)
            or _has_test_identifier(context_value)
            or _is_test_marker(context_value.get("evaluation_mode"), TEST_DATA_ROLES)
        )
        return {
            "locked": bool(official or test_only),
            "official_test_mode": official,
            "test_only_source": test_only,
            # Evaluation test data is never permitted to update resources.
            "test_data_update_allowed": False,
            "inference_only": True,
        }

    def require_write_allowed(
        self,
        operation: str,
        *,
        profile: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        state = self.state(profile=profile, context=context)
        if not state["locked"]:
            return
        operation_labels = {
            "feedback_update": "反馈优化",
            "memory_update": "记忆库或其配置更新",
            "profile_apply": "运行方案切换",
            "profile_rollback": "运行方案回滚",
            "profile_freeze": "候选方案冻结",
            "calibration": "阈值或校准参数更新",
            "domain_build": "目标域资源构建",
            "resource_lifecycle": "目标域资源维护",
        }
        label = operation_labels.get(str(operation), "资源更新")
        reason = "当前已开启正式测试锁" if state["official_test_mode"] else "请求包含测试专用数据或测试 Profile"
        raise ApiError(
            "OFFICIAL_TEST_WRITE_LOCKED",
            f"{reason}，{label}已被禁止；测试数据只能用于只读推理。",
            409,
            {
                "operation": str(operation),
                "official_test_mode": state["official_test_mode"],
                "test_only_source": state["test_only_source"],
                "test_data_update_allowed": False,
            },
        )

    @staticmethod
    def evaluation_fields(calibration_result: dict[str, Any] | None) -> dict[str, Any]:
        """Describe calibration evidence without presenting it as test accuracy."""

        return {
            "evaluation_mode": "calibration_only",
            "calibration_result": _mapping(calibration_result),
            "validation_result": {
                "status": "not_run",
                "data_role": "independent_validation_required",
                "message": "尚未使用独立验证集评估；校准集指标不是最终测试精度。",
            },
        }


__all__ = [
    "EvaluationLockService",
    "TEST_DATA_IDENTIFIERS",
    "TEST_DATA_ROLES",
    "TEST_ONLY_SCOPES",
]
